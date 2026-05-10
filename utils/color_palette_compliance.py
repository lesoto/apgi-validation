#!/usr/bin/env python3
"""
Color Palette Compliance Checker
===============================

Automated compliance checker for APGI color palette enforcement.
Scans Python files for hardcoded colors and ensures proper usage of
VISUAL_CONSTANTS palette.

Usage:
    python utils/color_palette_compliance.py
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Standard APGI color palette
APGI_COLORS = {
    "#2166AC": "ST_BLUE",
    "#D6604D": "THETA_RED",
    "#F4A582": "INTERO_AMBER",
    "#41AB5D": "IGNITION_GREEN",
    "#762A83": "ALLOSTATIC_PURPLE",
    "#999999": "HC_GREY",
    "#2ecc71": "STATUS_PASS",
    "#e74c3c": "STATUS_FAIL",
    "#f39c12": "STATUS_ERROR",
    "#95a5a6": "STATUS_UNKNOWN",
}

# Clinical colors (same values as APGI colors)
CLINICAL_COLORS = {
    "#2166AC": "MDD_BLUE",  # Same as ST_BLUE
    "#F4A582": "ANX_ORANGE",  # Same as INTERO_AMBER
    "#762A83": "ADHD_PURPLE",  # Same as ALLOSTATIC_PURPLE
    "#41AB5D": "IGNITION_GREEN_CLINICAL",  # Same as IGNITION_GREEN
    "#999999": "HC_GREY_CLINICAL",  # Same as HC_GREY
}


class ColorPaletteComplianceChecker:
    """Checks Python files for color palette compliance."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.violations: List[Dict] = []
        self.compliant_files: List[str] = []

    def check_file(self, file_path: Path) -> Dict:
        """Check a single Python file for color palette compliance."""
        violations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for hardcoded hex colors
            hex_pattern = r"#[0-9A-Fa-f]{6}"
            hex_colors = re.findall(hex_pattern, content)

            for color in hex_colors:
                if color in APGI_COLORS:
                    # This is an APGI color - check if it's using VISUAL_CONSTANTS
                    if not self._uses_visual_constants(content, color):
                        violations.append(
                            {
                                "type": "hardcoded_apgi_color",
                                "color": color,
                                "constant": APGI_COLORS[color],
                                "line": self._find_line_number(content, color),
                                "message": f"Hardcoded APGI color {color} should use VISUAL_CONSTANTS.{APGI_COLORS[color]}",
                            }
                        )
                else:
                    # Non-APGI color - flag as potential violation
                    violations.append(
                        {
                            "type": "non_apgi_color",
                            "color": color,
                            "line": self._find_line_number(content, color),
                            "message": f"Non-standard color {color} - should use APGI palette",
                        }
                    )

            # Check for proper VISUAL_CONSTANTS import
            if hex_colors and not self._has_visual_constants_import(content):
                violations.append(
                    {
                        "type": "missing_import",
                        "message": "File uses colors but missing VISUAL_CONSTANTS import",
                        "suggestion": "Add: from utils.constants import VISUAL_CONSTANTS",
                    }
                )

        except Exception as e:
            violations.append({"type": "file_error", "message": f"Error reading file: {e}"})

        return {
            "file": str(file_path),
            "violations": violations,
            "compliant": len(violations) == 0,
        }

    def _uses_visual_constants(self, content: str, color: str) -> bool:
        """Check if file uses VISUAL_CONSTANTS for this color."""
        constant_name = APGI_COLORS.get(color, "")
        if constant_name:
            return f"VISUAL_CONSTANTS.{constant_name}" in content
        return False

    def _has_visual_constants_import(self, content: str) -> bool:
        """Check if VISUAL_CONSTANTS is imported."""
        return "VISUAL_CONSTANTS" in content and ("import" in content or "from" in content)

    def _find_line_number(self, content: str, color: str) -> int:
        """Find the line number where a color appears."""
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if color in line:
                return i
        return 0

    def scan_project(self) -> Dict:
        """Scan entire project for color palette compliance."""
        python_files = list(self.project_root.rglob("*.py"))

        results: Dict[str, Any] = {
            "total_files": len(python_files),
            "compliant_files": 0,
            "non_compliant_files": 0,
            "total_violations": 0,
            "violation_types": {},
            "files": [],
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

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate a compliance report."""
        report = []
        report.append("APGI Color Palette Compliance Report")
        report.append("=" * 50)
        report.append(f"Total files scanned: {results['total_files']}")
        report.append(f"Compliant files: {results['compliant_files']}")
        report.append(f"Non-compliant files: {results['non_compliant_files']}")
        report.append(f"Total violations: {results['total_violations']}")
        report.append(f"Compliance rate: {results['compliant_files'] / results['total_files'] * 100:.1f}%")
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
                    report.append(f"    ... and {len(file_result['violations']) - 3} more")
                report.append("")

            if len(non_compliant_files) > 10:
                report.append(f"... and {len(non_compliant_files) - 10} more files with violations")

        return "\n".join(report)


def main():
    """Main function to run the compliance checker."""
    project_root = Path(__file__).parent.parent

    checker = ColorPaletteComplianceChecker(project_root)
    results = checker.scan_project()

    report = checker.generate_report(results)
    print(report)

    # Return non-zero exit code if violations found
    if results["total_violations"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
