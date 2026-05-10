#!/usr/bin/env python3
"""
Level Designation Compliance Checker
=================================

Automated compliance checker for APGI level designation enforcement.
Scans Python files for proper LEVEL DESIGNATION declarations and validates
cross-level bridge invocations.

Usage:
    python utils/level_designation_compliance.py
"""

import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Level definitions
LEVEL_DEFINITIONS = {
    1: "thermodynamic",
    2: "information-theoretic",
    3: "algorithmic/mathematical",
}

# Bridge requirements
BRIDGE_REQUIREMENTS = {
    1: [],  # Level 1 has no higher level to bridge to
    2: [1],  # Level 2 can bridge to Level 1
    3: [2, 1],  # Level 3 can bridge to Level 2 and Level 1
}


class LevelDesignationComplianceChecker:
    """Checks Python files for level designation compliance."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations: List[Dict] = []
        self.compliant_files: List[str] = []

    def check_file(self, file_path: Path) -> Dict:
        """Check a single Python file for level designation compliance."""
        violations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for LEVEL DESIGNATION in raw content first (more reliable)
            if "LEVEL DESIGNATION:" not in content:
                violations.append(
                    {
                        "type": "missing_level_designation",
                        "message": "No LEVEL DESIGNATION declaration found",
                        "line": 1,
                    }
                )
                return {
                    "file": str(file_path),
                    "violations": violations,
                    "compliant": len(violations) == 0,
                }

            # Parse AST to find docstring for additional validation
            try:
                tree = ast.parse(content)
                docstring = ast.get_docstring(tree)
            except SyntaxError:
                violations.append(
                    {
                        "type": "syntax_error",
                        "message": "File has syntax errors, cannot parse",
                        "line": 1,
                    }
                )
                return {
                    "file": str(file_path),
                    "violations": violations,
                    "compliant": len(violations) == 0,
                }

            # Extract declared level from raw content
            level_match = re.search(r"LEVEL DESIGNATION:.*?Level (\d+)", content, re.DOTALL)
            if not level_match:
                violations.append(
                    {
                        "type": "invalid_level_declaration",
                        "message": "LEVEL DESIGNATION found but level not specified correctly",
                        "line": 1,
                    }
                )
                return {
                    "file": str(file_path),
                    "violations": violations,
                    "compliant": len(violations) == 0,
                }

            declared_level = int(level_match.group(1))

            # Check if level is valid
            if declared_level not in LEVEL_DEFINITIONS:
                violations.append(
                    {
                        "type": "invalid_level",
                        "message": f"Invalid level {declared_level}. Valid levels: {list(LEVEL_DEFINITIONS.keys())}",
                        "line": 1,
                    }
                )

            # Check for bridge requirements
            bridge_pattern = re.search(r"Bridge to Level (\d+)", content)
            if bridge_pattern:
                bridge_levels = [int(m.group(1)) for m in re.finditer(r"Bridge to Level (\d+)", content)]
                required_bridges = BRIDGE_REQUIREMENTS.get(declared_level, [])

                for bridge_level in bridge_levels:
                    if bridge_level not in LEVEL_DEFINITIONS:
                        violations.append(
                            {
                                "type": "invalid_bridge_level",
                                "message": f"Invalid bridge level {bridge_level} in {file_path.name}",
                                "line": 1,
                            }
                        )

                # Check if required bridges are mentioned
                for required_bridge in required_bridges:
                    if required_bridge not in bridge_levels:
                        violations.append(
                            {
                                "type": "missing_bridge",
                                "message": f"Missing required bridge to Level {required_bridge}",
                                "line": 1,
                            }
                        )

            # Check for cross-level claims without bridge invocation
            self._check_cross_level_claims(content, declared_level, violations)

            # Check for explicit bridge function usage
            self._check_bridge_usage(content, declared_level, violations)

        except Exception as e:
            violations.append({"type": "file_error", "message": f"Error reading file: {e}", "line": 1})

        return {
            "file": str(file_path),
            "violations": violations,
            "compliant": len(violations) == 0,
            "declared_level": declared_level if "level_match" in locals() else None,
        }

    def _check_cross_level_claims(self, content: str, declared_level: int, violations: List[Dict]):
        """Check for cross-level claims without proper bridge invocation."""
        # Look for patterns that suggest cross-level claims
        cross_level_patterns = [
            (r"\bthermodynamic\b.*?\bclaim\b", "thermodynamic"),
            (r"\binformation.*?\btheory\b", "information-theoretic"),
            (r"\bbandwidth.*?\bbits/s\b", "information-theoretic"),
            (r"\batp.*?\benergy\b", "thermodynamic"),
            (r"\bmetabolic.*?\bcost\b", "thermodynamic"),
        ]

        for pattern_str, target_level in cross_level_patterns:
            if target_level != declared_level and re.search(pattern_str, content, re.IGNORECASE):
                violations.append(
                    {
                        "type": "cross_level_claim_without_bridge",
                        "message": f"Cross-level claim to {target_level} found without bridge invocation",
                        "line": self._find_line_number(content, pattern_str),
                    }
                )

    def _check_bridge_usage(self, content: str, declared_level: int, violations: List[Dict]):
        """Check for proper bridge function usage."""
        # Look for bridge function imports and calls
        bridge_imports = [
            "from utils.bridge_invocations import",
            "import utils.bridge_invocations",
            "bridge_to_thermodynamic",
            "bridge_to_information_theoretic",
            "bridge_l2_to_l1",
            "bridge_l3_to_l2",
        ]

        has_bridge_import = any(imp in content for imp in bridge_imports)

        # Check if bridge is needed but not imported
        if declared_level in [2, 3] and not has_bridge_import:
            violations.append(
                {
                    "type": "missing_bridge_import",
                    "message": f"Level {declared_level} module uses cross-level concepts but missing bridge import",
                    "line": 1,
                }
            )

        # Check for bridge function calls
        bridge_calls = [
            "bridge_to_thermodynamic(",
            "bridge_to_information_theoretic(",
            "bridge_l2_to_l1_with_validation(",
            "bridge_l3_to_l2_with_validation(",
        ]

        has_bridge_call = any(call in content for call in bridge_calls)

        if declared_level in [2, 3] and not has_bridge_call:
            violations.append(
                {
                    "type": "missing_bridge_call",
                    "message": f"Level {declared_level} module has bridge import but no bridge function call",
                    "line": self._find_line_number(content, "bridge_to_"),
                }
            )

    def _find_line_number(self, content: str, pattern: str) -> int:
        """Find the line number where a pattern appears."""
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if pattern in line:
                return i
        return 0

    def scan_project(self) -> Dict[str, Any]:
        """Scan the entire project for level designation compliance."""
        python_files = list(self.project_root.rglob("*.py"))

        results: Dict[str, Any] = {
            "total_files": len(python_files),
            "compliant_files": 0,
            "non_compliant_files": 0,
            "total_violations": 0,
            "violation_types": {},
            "files": [],
            "level_distribution": {1: 0, 2: 0, 3: 0},
        }

        for file_path in python_files:
            # Skip test files and __pycache__
            if "test" in file_path.parts or "__pycache__" in file_path.parts:
                continue

            file_result = self.check_file(file_path)
            results["files"].append(file_result)

            if file_result["compliant"]:
                results["compliant_files"] += 1
            else:
                results["non_compliant_files"] += 1
                results["total_violations"] += len(file_result["violations"])

                # Count violation types
                for violation in file_result["violations"]:
                    vtype = violation["type"]
                    results["violation_types"][vtype] = results["violation_types"].get(vtype, 0) + 1

                # Count level distribution
                if "declared_level" in file_result:
                    level = file_result["declared_level"]
                    results["level_distribution"][level] = results["level_distribution"].get(level, 0) + 1

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate a compliance report."""
        report = []
        report.append("APGI Level Designation Compliance Report")
        report.append("=" * 50)
        report.append(f"Total files scanned: {results['total_files']}")
        report.append(f"Compliant files: {results['compliant_files']}")
        report.append(f"Non-compliant files: {results['non_compliant_files']}")
        report.append(f"Total violations: {results['total_violations']}")
        report.append(f"Compliance rate: {results['compliant_files'] / results['total_files'] * 100:.1f}%")
        report.append("")

        # Level distribution
        report.append("Level Distribution:")
        for level, count in results["level_distribution"].items():
            level_name = LEVEL_DEFINITIONS.get(level, f"Level {level}")
            report.append(f"  {level_name}: {count} files")
        report.append("")

        if results["violation_types"]:
            report.append("Violation Summary:")
            for vtype, count in results["violation_types"].items():
                report.append(f"  {vtype}: {count}")
            report.append("")

        # Show files with violations
        non_compliant_files = [f for f in results["files"] if not f["compliant"]]
        if non_compliant_files:
            report.append("Files with violations:")
            for file_result in non_compliant_files[:10]:  # Show first 10
                report.append(f"  {file_result['file']}")
                for violation in file_result["violations"][:3]:  # Show first 3 violations
                    report.append(f"    Line {violation.get('line', '?')}: {violation['message']}")
                if len(file_result["violations"]) > 3:
                    report.append(f"    ... and {len(file_result['violations']) - 3} more violations")
                report.append("")

            if len(non_compliant_files) > 10:
                report.append(f"... and {len(non_compliant_files) - 10} more files with violations")

        return "\n".join(report)


def main():
    """Main function to run the compliance checker."""
    project_root = Path(__file__).parent.parent

    checker = LevelDesignationComplianceChecker(project_root)
    results = checker.scan_project()

    report = checker.generate_report(results)
    print(report)

    # Return non-zero exit code if violations found
    if results["total_violations"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
