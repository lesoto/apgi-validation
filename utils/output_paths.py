"""
LEVEL DESIGNATION: Level 2 (information-theoretic)

Bridge to Level 1

Unified Output Path Management
==============================

All outputs are consolidated into apgi_outputs/ directory with the following structure:

    apgi_outputs/
    ├── validation/
    │   ├── VP_01/
    │   │   ├── results.json
    │   │   ├── metadata.json
    │   │   └── visualizations/
    │   ├── VP_02/
    │   └── ...
    ├── falsification/
    │   ├── FP_01/
    │   │   ├── results.json
    │   │   ├── metadata.json
    │   │   └── visualizations/
    │   ├── FP_02/
    │   └── ...
    ├── theory/
    │   ├── module_name/
    │   │   ├── results.json
    │   │   ├── metadata.json
    │   │   └── visualizations/
    │   └── ...
    ├── dashboards/
    │   ├── system_dashboard.html
    │   ├── validation_dashboard.html
    │   └── ...
    ├── logs/
    │   ├── error.log
    │   ├── alerts.jsonl
    │   └── ...
    └── reports/
        ├── batch_report.json
        ├── performance_report.json
        └── ...
"""

from pathlib import Path
from typing import Dict, Optional


def get_project_root() -> Path:
    """Get the APGI project root directory."""
    return Path(__file__).parent.parent


def get_apgi_outputs_root() -> Path:
    """Get the unified APGI outputs root directory."""
    root = get_project_root()
    output_root = root / "apgi_outputs"
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def get_output_dir(subdir: Optional[str] = None) -> Path:
    """
    Get the standardized output directory.

    Args:
        subdir: Optional subdirectory (e.g., 'validation', 'falsification', 'theory', 'dashboards', 'logs', 'reports')

    Returns:
        Path to the output directory
    """
    root = get_apgi_outputs_root()

    if subdir:
        output_dir = root / subdir
    else:
        output_dir = root

    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_validation_output_dir(protocol_num: Optional[int] = None) -> Path:
    """Get output directory for validation protocols."""
    root = get_apgi_outputs_root()
    validation_dir = root / "validation"

    if protocol_num is not None:
        validation_dir = validation_dir / f"VP_{protocol_num:02d}"

    validation_dir.mkdir(parents=True, exist_ok=True)
    return validation_dir


def get_falsification_output_dir(protocol_num: Optional[int] = None) -> Path:
    """Get output directory for falsification protocols."""
    root = get_apgi_outputs_root()
    falsification_dir = root / "falsification"

    if protocol_num is not None:
        falsification_dir = falsification_dir / f"FP_{protocol_num:02d}"

    falsification_dir.mkdir(parents=True, exist_ok=True)
    return falsification_dir


def get_theory_output_dir(module_name: Optional[str] = None) -> Path:
    """Get output directory for theory modules."""
    root = get_apgi_outputs_root()
    theory_dir = root / "theory"

    if module_name:
        theory_dir = theory_dir / module_name

    theory_dir.mkdir(parents=True, exist_ok=True)
    return theory_dir


def get_dashboard_output_dir() -> Path:
    """Get output directory for dashboards."""
    return get_output_dir("dashboards")


def get_logs_output_dir() -> Path:
    """Get output directory for logs."""
    return get_output_dir("logs")


def get_reports_output_dir() -> Path:
    """Get output directory for reports."""
    return get_output_dir("reports")


def get_protocol_json_path(protocol_num: int, protocol_type: str = "validation") -> Path:
    """
    Get standardized JSON output path for a protocol.

    Args:
        protocol_num: Protocol number (1-21 for validation, 1-12 for falsification)
        protocol_type: 'validation', 'falsification', or 'theory'

    Returns:
        Path to the JSON results file
    """
    if protocol_type == "validation":
        output_dir = get_validation_output_dir(protocol_num)
    elif protocol_type == "falsification":
        output_dir = get_falsification_output_dir(protocol_num)
    else:
        output_dir = get_theory_output_dir(protocol_type)

    return output_dir / "results.json"


def get_protocol_csv_path(protocol_num: int, protocol_type: str = "validation") -> Path:
    """
    Get standardized CSV output path for a protocol.

    Args:
        protocol_num: Protocol number
        protocol_type: 'validation', 'falsification', or 'theory'

    Returns:
        Path to the CSV results file
    """
    if protocol_type == "validation":
        output_dir = get_validation_output_dir(protocol_num)
    elif protocol_type == "falsification":
        output_dir = get_falsification_output_dir(protocol_num)
    else:
        output_dir = get_theory_output_dir(protocol_type)

    return output_dir / "results.csv"


def get_protocol_png_path(protocol_num: int, protocol_type: str = "validation", suffix: Optional[str] = None) -> Path:
    """
    Get standardized PNG output path for a protocol.

    Args:
        protocol_num: Protocol number
        protocol_type: 'validation', 'falsification', or 'theory'
        suffix: Optional suffix for the filename (e.g., 'comparison', 'analysis')

    Returns:
        Path to the PNG visualization file
    """
    if protocol_type == "validation":
        output_dir = get_validation_output_dir(protocol_num)
    elif protocol_type == "falsification":
        output_dir = get_falsification_output_dir(protocol_num)
    else:
        output_dir = get_theory_output_dir(protocol_type)

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    if suffix:
        filename = f"results_{suffix}.png"
    else:
        filename = "results.png"

    return viz_dir / filename


def get_metadata_path(protocol_num: int, protocol_type: str = "validation") -> Path:
    """Get standardized metadata JSON path for a protocol."""
    if protocol_type == "validation":
        output_dir = get_validation_output_dir(protocol_num)
    elif protocol_type == "falsification":
        output_dir = get_falsification_output_dir(protocol_num)
    else:
        output_dir = get_theory_output_dir(protocol_type)

    return output_dir / "metadata.json"


def migrate_legacy_outputs() -> dict:
    """
    Migrate legacy output files from old locations to apgi_outputs/.

    Returns:
        Dictionary with migration statistics
    """
    root = get_project_root()
    stats = {"migrated": 0, "errors": 0, "skipped": 0}

    # Legacy output directories to migrate
    legacy_dirs = [
        ("outputs", "legacy"),
        ("validation_results", "legacy"),
        ("results", "legacy"),
        ("reports", "reports"),
        ("logs", "logs"),
    ]

    for legacy_dir_name, target_subdir in legacy_dirs:
        legacy_dir = root / legacy_dir_name
        if not legacy_dir.exists():
            continue

        target_dir = get_output_dir(target_subdir)

        try:
            # Move all files from legacy directory to target
            for file_path in legacy_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(legacy_dir)
                    target_file = target_dir / relative_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file
                    target_file.write_bytes(file_path.read_bytes())
                    stats["migrated"] += 1
        except Exception as e:
            print(f"Error migrating {legacy_dir_name}: {e}")
            stats["errors"] += 1

    return stats


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
    return get_project_root() / f"protocol{protocol_num:02d}_results.png"


# Stubs for test compatibility
# Alias for get_output_dir with different naming
get_output_directory = get_output_dir


def create_output_path(base_dir: Path, filename: str) -> Path:
    """Create output path by joining base dir with filename."""
    return base_dir / filename


def ensure_directory_exists(dir_path: Path) -> None:
    """Ensure directory exists, creating if necessary."""
    dir_path.mkdir(parents=True, exist_ok=True)


def get_unique_filename(dir_path: Path, filename: str) -> Path:
    """Get a unique filename in directory."""
    target = dir_path / filename
    if not target.exists():
        return target
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        new_path = dir_path / new_name
        if not new_path.exists():
            return new_path
        counter += 1


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing special chars."""
    import re

    # Replace special chars with underscore
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    return sanitized


class OutputPathManager:
    """Manage output paths with registration."""

    def __init__(self, base_dir: Path):
        """Initialize path manager."""
        self.base_dir = base_dir
        self._registered: Dict[str, Path] = {}

    def get_path(self, filename: str) -> Path:
        """Get a managed path."""
        return self.base_dir / filename

    def register(self, name: str, filename: str) -> None:
        """Register a named path."""
        self._registered[name] = self.base_dir / filename

    def get_registered(self, name: str) -> Optional[Path]:
        """Get a registered path by name."""
        return self._registered.get(name)
