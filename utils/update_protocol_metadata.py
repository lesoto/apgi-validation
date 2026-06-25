#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.metadata_standardizer import (
    PROTOCOL_DEPENDENCIES,
    PROTOCOL_PREDICTIONS,
    DataSource,
    ProtocolStatus,
    standardize_metadata,
)


def check_protocol_metadata(protocol_id: str, metadata: Dict) -> Tuple[bool, List[str]]:
    """
    Check if protocol metadata follows the standardized schema.

    Args:
        protocol_id: Protocol identifier (e.g., "FP_01", "VP_01")
        metadata: Current metadata dictionary

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check required fields
    required_fields = ["status", "data_source", "predictions"]
    for field in required_fields:
        if field not in metadata:
            issues.append(f"Missing required field: {field}")

    # Check status field
    if "status" in metadata:
        status = metadata["status"]
        try:
            ProtocolStatus(status.upper())
        except ValueError:
            issues.append(f"Invalid status: {status}. Must be one of: {[s.value for s in ProtocolStatus]}")

    # Check data_source field
    if "data_source" in metadata:
        data_source = metadata["data_source"]
        try:
            DataSource(data_source.lower())
        except ValueError:
            issues.append(f"Invalid data_source: {data_source}. Must be one of: {[ds.value for ds in DataSource]}")

    # Check predictions field
    if "predictions" in metadata:
        predictions = metadata["predictions"]
        if not isinstance(predictions, list):
            issues.append("predictions must be a list")
        elif protocol_id in PROTOCOL_PREDICTIONS:
            expected_predictions = PROTOCOL_PREDICTIONS[protocol_id]
            for pred in expected_predictions:
                if pred not in predictions:
                    issues.append(f"Missing expected prediction: {pred}")

    # Check dependencies
    if "dependencies" in metadata:
        dependencies = metadata["dependencies"]
        if not isinstance(dependencies, list):
            issues.append("dependencies must be a list")
        elif protocol_id in PROTOCOL_DEPENDENCIES:
            expected_deps = PROTOCOL_DEPENDENCIES[protocol_id]
            for dep in expected_deps:
                if dep not in dependencies:
                    issues.append(f"Missing expected dependency: {dep}")

    return len(issues) == 0, issues


def update_protocol_metadata(protocol_id: str, metadata: Dict) -> Dict:
    """
    Update protocol metadata to follow standardized schema.

    Args:
        protocol_id: Protocol identifier
        metadata: Current metadata dictionary

    Returns:
        Updated metadata dictionary
    """
    # Start with standardized metadata
    updated = standardize_metadata(protocol_id, metadata)

    # Ensure predictions are included
    if protocol_id in PROTOCOL_PREDICTIONS:
        expected_predictions = PROTOCOL_PREDICTIONS[protocol_id]
        if "predictions" not in updated:
            updated["predictions"] = []
        for pred in expected_predictions:
            if pred not in updated["predictions"]:
                updated["predictions"].append(pred)

    # Ensure dependencies are included
    if protocol_id in PROTOCOL_DEPENDENCIES:
        expected_deps = PROTOCOL_DEPENDENCIES[protocol_id]
        if "dependencies" not in updated:
            updated["dependencies"] = []
        for dep in expected_deps:
            if dep not in updated["dependencies"]:
                updated["dependencies"].append(dep)

    # Add timestamp if not present
    if "last_updated" not in updated:
        from datetime import datetime

        updated["last_updated"] = datetime.now().isoformat()

    return updated


def scan_protocol_files(project_root: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Scan all protocol files for metadata and check compliance.

    Args:
        project_root: Root directory of the project

    Returns:
        Dictionary mapping protocol_id to metadata and compliance status
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    results = {}

    # Scan Falsification protocols
    fp_dir = project_root / "Falsification"
    if fp_dir.exists():
        for fp_file in fp_dir.glob("FP_*.py"):
            protocol_id = fp_file.stem
            metadata = extract_metadata_from_file(fp_file)
            is_valid, issues = check_protocol_metadata(protocol_id, metadata)
            results[protocol_id] = {
                "metadata": metadata,
                "is_valid": is_valid,
                "issues": issues,
                "file_path": str(fp_file),
            }

    # Scan Validation protocols
    vp_dir = project_root / "Validation"
    if vp_dir.exists():
        for vp_file in vp_dir.glob("VP_*.py"):
            protocol_id = vp_file.stem
            metadata = extract_metadata_from_file(vp_file)
            is_valid, issues = check_protocol_metadata(protocol_id, metadata)
            results[protocol_id] = {
                "metadata": metadata,
                "is_valid": is_valid,
                "issues": issues,
                "file_path": str(vp_file),
            }

    return results


def extract_metadata_from_file(file_path: Path) -> Dict:
    """
    Extract metadata from a protocol file.

    Args:
        file_path: Path to the protocol file

    Returns:
        Extracted metadata dictionary
    """
    metadata = {}

    try:
        content = file_path.read_text(encoding="utf-8")

        # Look for metadata in docstring
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1)

            # Extract status
            status_match = re.search(r"Status:\s*(\w+)", docstring, re.IGNORECASE)
            if status_match:
                metadata["status"] = status_match.group(1).upper()

            # Extract data source
            source_match = re.search(r"Data[ _]?[Ss]ource:\s*(\w+)", docstring, re.IGNORECASE)
            if source_match:
                metadata["data_source"] = source_match.group(1).lower()

            # Extract predictions
            predictions_match = re.search(
                r"Predictions?:?\s*(.*?)(?:\n\n|\n\s*\n|$)", docstring, re.IGNORECASE | re.DOTALL
            )
            if predictions_match:
                predictions_text = predictions_match.group(1)
                # Simple extraction - could be enhanced
                predictions = [p.strip() for p in predictions_text.split("\n") if p.strip()]
                metadata["predictions"] = predictions

        # Look for metadata dictionary in code
        metadata_match = re.search(r"metadata\s*=\s*({.*?})", content, re.DOTALL)
        if metadata_match:
            try:
                # Use ast.literal_eval for safe evaluation
                import ast

                metadata_dict = ast.literal_eval(metadata_match.group(1))
                if isinstance(metadata_dict, dict):
                    metadata.update(metadata_dict)
            except (SyntaxError, ValueError):
                pass

    except Exception as e:
        metadata["error"] = str(e)

    return metadata


def update_all_protocols(project_root: Optional[Path] = None, dry_run: bool = True) -> Dict[str, Dict]:
    """
    Update metadata for all protocols.

    Args:
        project_root: Root directory of the project
        dry_run: If True, only show what would be changed without writing files

    Returns:
        Dictionary of update results
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent

    scan_results = scan_protocol_files(project_root)
    update_results = {}

    for protocol_id, result in scan_results.items():
        file_path = Path(result["file_path"])
        current_metadata = result["metadata"]
        updated_metadata = update_protocol_metadata(protocol_id, current_metadata)

        # Check if update is needed
        needs_update = current_metadata != updated_metadata

        update_results[protocol_id] = {
            "needs_update": needs_update,
            "current_metadata": current_metadata,
            "updated_metadata": updated_metadata,
            "file_path": str(file_path),
        }

        if needs_update and not dry_run:
            # Update the file
            update_file_metadata(file_path, updated_metadata)

    return update_results


def update_file_metadata(file_path: Path, metadata: Dict) -> bool:
    """
    Update metadata in a protocol file.

    Args:
        file_path: Path to the protocol file
        metadata: New metadata dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        content = file_path.read_text(encoding="utf-8")

        # Look for existing metadata assignment
        metadata_pattern = r"(metadata\s*=\s*){.*?}"
        if re.search(metadata_pattern, content, re.DOTALL):
            # Replace existing metadata
            new_metadata_str = json.dumps(metadata, indent=2)
            new_content = re.sub(
                metadata_pattern,
                f"\\1{new_metadata_str}",
                content,
                flags=re.DOTALL,
            )
        else:
            # Add metadata at the end of the file (before any final return/if __name__)
            new_metadata_str = f"\n\nmetadata = {json.dumps(metadata, indent=2)}"
            if "__name__" in content:
                # Insert before if __name__ == "__main__"
                parts = content.split('if __name__ == "__main__"')
                new_content = parts[0] + new_metadata_str + '\n\nif __name__ == "__main__"' + parts[1]
            else:
                new_content = content + new_metadata_str

        file_path.write_text(new_content, encoding="utf-8")
        return True

    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False


def generate_compliance_report(scan_results: Dict[str, Dict]) -> str:
    """
    Generate a compliance report from scan results.

    Args:
        scan_results: Results from scan_protocol_files

    Returns:
        Formatted report string
    """
    report_lines = []
    report_lines.append("Protocol Metadata Compliance Report")
    report_lines.append("=" * 50)

    total_protocols = len(scan_results)
    compliant_protocols = sum(1 for r in scan_results.values() if r["is_valid"])
    non_compliant_protocols = total_protocols - compliant_protocols

    report_lines.append(f"Total protocols scanned: {total_protocols}")
    report_lines.append(f"Compliant protocols: {compliant_protocols}")
    report_lines.append(f"Non-compliant protocols: {non_compliant_protocols}")
    report_lines.append(f"Compliance rate: {compliant_protocols / total_protocols * 100:.1f}%")
    report_lines.append("")

    if non_compliant_protocols > 0:
        report_lines.append("Non-compliant protocols:")
        for protocol_id, result in scan_results.items():
            if not result["is_valid"]:
                report_lines.append(f"  {protocol_id}:")
                for issue in result["issues"]:
                    report_lines.append(f"    - {issue}")
        report_lines.append("")

    return "\n".join(report_lines)


def main():
    """Main function to run metadata checks and updates."""
    import argparse

    parser = argparse.ArgumentParser(description="Update and standardize protocol metadata")
    parser.add_argument("--scan", action="store_true", help="Scan protocols for metadata compliance")
    parser.add_argument("--update", action="store_true", help="Update metadata for non-compliant protocols")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without writing files")
    parser.add_argument("--report", action="store_true", help="Generate compliance report")
    args = parser.parse_args()

    if not any([args.scan, args.update, args.report]):
        args.scan = True
        args.report = True

    project_root = Path(__file__).parent.parent

    if args.scan or args.update:
        if args.update:
            print("Updating protocol metadata...")
            update_results = update_all_protocols(project_root, dry_run=args.dry_run)

            if args.dry_run:
                print("\nDry run - no files were modified")
                for protocol_id, result in update_results.items():
                    if result["needs_update"]:
                        print(f"\n{protocol_id} would be updated:")
                        print(f"  Current: {result['current_metadata']}")
                        print(f"  Updated: {result['updated_metadata']}")
            else:
                updated_count = sum(1 for r in update_results.values() if r["needs_update"])
                print(f"\nUpdated {updated_count} protocol(s)")
        else:
            print("Scanning protocol metadata...")
            scan_results = scan_protocol_files(project_root)

    if args.report:
        if "scan_results" not in locals():
            scan_results = scan_protocol_files(project_root)
        report = generate_compliance_report(scan_results)
        print(report)


if __name__ == "__main__":
    main()
