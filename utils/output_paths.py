"""
Centralized output path management for APGI protocols.

This module provides standardized paths for all protocol outputs,
ensuring consistency across PNG, JSON, CSV, and other file formats.
"""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Get the APGI project root directory."""
    return Path(__file__).parent.parent


def get_output_dir(subdir: Optional[str] = None) -> Path:
    """
    Get the standardized output directory.

    Args:
        subdir: Optional subdirectory (e.g., 'visualizations', 'data', 'results')

    Returns:
        Path to the output directory
    """
    root = get_project_root()

    # Primary output directory is validation_results/
    output_dir = root / "validation_results"

    if subdir:
        output_dir = output_dir / subdir

    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_protocol_json_path(protocol_num: int) -> Path:
    """Get standardized JSON output path for a protocol (saved in root)."""
    root = get_project_root()
    return root / f"protocol{protocol_num:02d}_results.json"


def get_protocol_csv_path(protocol_num: int) -> Path:
    """Get standardized CSV output path for a protocol (saved in root)."""
    root = get_project_root()
    return root / f"protocol{protocol_num:02d}_results.csv"


def get_protocol_png_path(protocol_num: int, suffix: Optional[str] = None) -> Path:
    """
    Get standardized PNG output path for a protocol (saved in validation_results/visualizations/).

    Args:
        protocol_num: Protocol number (e.g., 1, 2, 3)
        suffix: Optional suffix for multi-plot protocols (e.g., 'dlpfc', 'propranolol')
    """
    vis_dir = get_output_dir("visualizations")

    if suffix:
        return vis_dir / f"protocol{protocol_num:02d}_{suffix}_results.png"
    return vis_dir / f"protocol{protocol_num:02d}_results.png"


def get_framework_json_path() -> Path:
    """Get standardized JSON path for FP_ALL_Aggregator."""
    return get_output_dir() / "fp_all_results.json"


def migrate_legacy_outputs() -> dict:
    """
    Migrate legacy output files from root to validation_results/.

    Returns:
        Dictionary mapping old paths to new paths
    """
    root = get_project_root()
    migrations = {}

    # Patterns to migrate
    patterns = [
        ("protocol*_results.json", get_output_dir()),
        ("protocol*_results.csv", get_output_dir()),
        ("protocol*.png", get_output_dir("visualizations")),
        ("FP_0*_results.png", get_output_dir("visualizations")),
        ("VP_0*_results.png", get_output_dir("visualizations")),
    ]

    for pattern, dest_dir in patterns:
        for old_path in root.glob(pattern):
            new_path = dest_dir / old_path.name
            if old_path.exists() and not new_path.exists():
                try:
                    import shutil

                    shutil.move(str(old_path), str(new_path))
                    migrations[str(old_path)] = str(new_path)
                except Exception as e:
                    print(f"Warning: Could not migrate {old_path}: {e}")

    return migrations


# Legacy support: root-level paths for backward compatibility
def get_legacy_json_path(protocol_num: int) -> Path:
    """Get root-level JSON path (deprecated, use get_protocol_json_path)."""
    return get_project_root() / f"protocol{protocol_num}_results.json"


def get_legacy_csv_path(protocol_num: int) -> Path:
    """Get root-level CSV path (deprecated, use get_protocol_csv_path)."""
    return get_project_root() / f"protocol{protocol_num}_results.csv"


def get_legacy_png_path(protocol_num: int) -> Path:
    """Get root-level PNG path (deprecated, use get_protocol_png_path)."""
    return get_project_root() / f"protocol{protocol_num}_results.png"
