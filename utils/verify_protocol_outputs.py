#!/usr/bin/env python3
"""
Verification script to check that ALL protocols output .png, .json, and .csv results.

Usage:
    python verify_protocol_outputs.py [--report]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Union


def get_project_root() -> Path:
    """Get the APGI project root directory."""
    return Path(__file__).parent


# Define expected outputs per protocol (standardized naming with leading zeros)
ExpectedOutputPattern = Union[str, List[str], None]
ExpectedProtocolOutputs = Dict[str, ExpectedOutputPattern]
EXPECTED_OUTPUTS: Dict[int, ExpectedProtocolOutputs] = {
    1: {
        "png": "protocol01_results.png",
        "json": "protocol01_results.json",
        "csv": None,
    },
    2: {
        "png": "protocol02_results.png",
        "json": "protocol02_results.json",
        "csv": None,
    },
    3: {
        "png": "protocol03_results.png",
        "json": "protocol03_results.json",
        "csv": "protocol03_results.csv",
    },
    4: {
        "png": "protocol04_results.png",
        "json": "protocol04_results.json",
        "csv": "protocol04_monte_carlo.csv",
    },
    5: {
        "png": "protocol05_fmri_timeseries.png",
        "json": "protocol05_fmri_results.json",
        "csv": None,
    },
    6: {
        "png": "protocol06_results.png",
        "json": "protocol06_results.json",
        "csv": "protocol06_results.csv",
    },
    7: {
        "png": ["protocol07_dlpfc_results.png", "protocol07_propranolol_results.png"],
        "json": "protocol07_results.json",
        "csv": None,
    },
    8: {
        "png": "protocol08_individual_differences.png",
        "json": "protocol08_results.json",
        "csv": "protocol08_participant.csv",
    },
    9: {
        "png": "protocol09_results.png",
        "json": "protocol09_results.json",
        "csv": None,
    },
    10: {
        "png": "protocol10_results.png",
        "json": "protocol10_results.json",
        "csv": None,
    },
    11: {
        "png": "protocol11_results.png",
        "json": "protocol11_results.json",
        "csv": None,
    },
    12: {
        "png": "protocol12_results.png",
        "json": "protocol12_results.json",
        "csv": None,
    },
}


def find_existing_outputs(root_dir: Path) -> Dict[str, List[Path]]:
    """Find all existing output files in the project root."""
    outputs: Dict[str, List[Path]] = {
        "png": [],
        "json": [],
        "csv": [],
    }

    # Search in root directory only (not recursive to avoid .git, etc.)
    for ext in ["png", "json", "csv"]:
        for path in root_dir.glob(f"*.{ext}"):
            if path.is_file() and path.stat().st_size > 0:
                outputs[ext].append(path)

    # Also check validation_results directory if it exists
    validation_dir = root_dir / "validation_results"
    if validation_dir.exists():
        for ext in ["png", "json", "csv"]:
            for path in validation_dir.rglob(f"*.{ext}"):
                if path.is_file() and path.stat().st_size > 0:
                    outputs[ext].append(path)

    return outputs


def check_protocol_outputs(
    protocol_num: int, outputs: Dict[str, List[Path]]
) -> Dict[str, bool]:
    """Check if a specific protocol has all expected outputs."""
    expected = EXPECTED_OUTPUTS.get(protocol_num, {})

    def check_file(pattern, file_list):
        if pattern is None:
            return True  # Not required
        if isinstance(pattern, list):
            # At least one of the files should exist
            return any(any(p.name == ptn for p in file_list) for ptn in pattern)
        return any(p.name == pattern for p in file_list)

    return {
        "png": check_file(expected.get("png"), outputs["png"]),
        "json": check_file(expected.get("json"), outputs["json"]),
        "csv": check_file(expected.get("csv"), outputs["csv"]),
    }


def generate_report(outputs: Dict[str, List[Path]]) -> str:
    """Generate a markdown report of output status."""
    lines = []
    lines.append("# APGI Protocol Output Verification Report")
    lines.append("")
    lines.append("**Generated:** Auto-generated from file scan")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Protocol | PNG | JSON | CSV | Status |")
    lines.append("|----------|-----|------|-----|--------|")

    complete_count = 0
    for i in range(1, 13):
        status = check_protocol_outputs(i, outputs)
        expected = EXPECTED_OUTPUTS.get(i, {})

        png_required = expected.get("png") is not None
        json_required = expected.get("json") is not None
        csv_required = expected.get("csv") is not None

        png_ok = status["png"] if png_required else True
        json_ok = status["json"] if json_required else True
        csv_ok = status["csv"] if csv_required else True

        png_mark = "✓" if png_ok else ("✗" if png_required else "—")
        json_mark = "✓" if json_ok else ("✗" if json_required else "—")
        csv_mark = "✓" if csv_ok else ("✗" if csv_required else "—")

        all_ok = png_ok and json_ok and csv_ok
        status_str = "✅ Complete" if all_ok else "⚠️ Partial"
        if all_ok:
            complete_count += 1

        lines.append(
            f"| FP_{i:02d} | {png_mark} | {json_mark} | {csv_mark} | {status_str} |"
        )

    lines.append("")
    lines.append(f"**Overall: {complete_count}/12 protocols complete**")

    # List all files found
    lines.append("")
    lines.append("## Existing Output Files")
    lines.append("")

    for ext in ["png", "json", "csv"]:
        files = sorted(outputs[ext], key=lambda p: p.name)
        lines.append(f"### {ext.upper()} Files ({len(files)} total)")
        lines.append("")
        for path in files:
            size = path.stat().st_size
            lines.append(f"- {path.name} ({size:,} bytes)")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Verify APGI protocol outputs")
    parser.add_argument(
        "--report", action="store_true", help="Generate markdown report"
    )
    parser.add_argument(
        "--output",
        default="protocol_output_verification_report.md",
        help="Report output file",
    )
    args = parser.parse_args()

    root = get_project_root()

    # Find all outputs
    outputs = find_existing_outputs(root)

    # Print summary
    print(
        f"Found {len(outputs['png'])} PNG, {len(outputs['json'])} JSON, {len(outputs['csv'])} CSV files"
    )
    print()

    # Check each protocol
    complete_count = 0
    for i in range(1, 13):
        status = check_protocol_outputs(i, outputs)
        expected = EXPECTED_OUTPUTS.get(i, {})

        png_required = expected.get("png") is not None
        json_required = expected.get("json") is not None
        csv_required = expected.get("csv") is not None

        png_ok = status["png"] if png_required else True
        json_ok = status["json"] if json_required else True
        csv_ok = status["csv"] if csv_required else True

        all_ok = png_ok and json_ok and csv_ok

        if all_ok:
            complete_count += 1
            print(f"✅ FP_{i:02d}: Complete")
        else:
            missing = []
            if png_required and not png_ok:
                missing.append("PNG")
            if json_required and not json_ok:
                missing.append("JSON")
            if csv_required and not csv_ok:
                missing.append("CSV")
            print(f"⚠️  FP_{i:02d}: Missing {', '.join(missing)}")

    print()
    print(f"Complete protocols: {complete_count}/12")

    # Generate report
    if args.report:
        report = generate_report(outputs)
        report_path = root / args.output
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")

    return 0


# Stubs for test compatibility
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VerificationResult:
    """Result of protocol output verification."""

    valid: bool
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]

    def is_success(self) -> bool:
        """Check if verification was successful."""
        return self.valid and not self.errors


class OutputVerifier:
    """Verifies protocol output files."""

    def __init__(self):
        """Initialize verifier."""
        pass

    def verify(
        self, filepath: Path, schema: Optional[Dict] = None
    ) -> VerificationResult:
        """Verify an output file."""
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {"checked": 0}

        if not filepath.exists():
            errors.append(f"File does not exist: {filepath}")
            return VerificationResult(
                valid=False, errors=errors, warnings=warnings, details=details
            )

        try:
            with open(filepath, "r") as f:
                content = f.read()
                data = json.loads(content)
                details["checked"] = (
                    len(data)
                    if isinstance(data, dict)
                    else len(data) if isinstance(data, list) else 1
                )

            if schema and isinstance(data, dict):
                # Basic schema validation
                for key in schema.get("properties", {}):
                    if key not in data:
                        warnings.append(f"Missing recommended field: {key}")

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
        except Exception as e:
            errors.append(f"Error reading file: {e}")

        return VerificationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings, details=details
        )


def verify_protocol_output(filepath: Path) -> VerificationResult:
    """Verify a protocol output file."""
    verifier = OutputVerifier()
    return verifier.verify(filepath)


def check_output_structure(
    data: Dict[str, Any], required_fields: List[str]
) -> Dict[str, Any]:
    """Check that data has required structure."""
    result: Dict[str, Any] = {"valid": True, "missing_fields": [], "extra_fields": []}

    for field in required_fields:
        if field not in data:
            result["valid"] = False
            result["missing_fields"].append(field)

    return result


def validate_output_files(files: List[Path]) -> Dict[str, Any]:
    """Validate that output files exist."""
    result: Dict[str, Any] = {"valid": True, "missing": [], "existing": []}

    for filepath in files:
        if filepath.exists():
            result["existing"].append(str(filepath))
        else:
            result["valid"] = False
            result["missing"].append(str(filepath))

    return result


def compare_protocol_outputs(
    output1: Dict[str, Any], output2: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare two protocol outputs."""
    result: Dict[str, Any] = {
        "identical": output1 == output2,
        "differences": [],
        "keys_only_in_first": [],
        "keys_only_in_second": [],
    }

    if not result["identical"]:
        keys1 = set(output1.keys()) if isinstance(output1, dict) else set()
        keys2 = set(output2.keys()) if isinstance(output2, dict) else set()

        result["keys_only_in_first"] = list(keys1 - keys2)
        result["keys_only_in_second"] = list(keys2 - keys1)
        for key in keys1 & keys2:
            if output1[key] != output2[key]:
                result["differences"].append(key)
    return result


if __name__ == "__main__":
    sys.exit(main())
