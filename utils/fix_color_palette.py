#!/usr/bin/env python3
"""
Fix color palette compliance by replacing hardcoded colors with VISUAL_CONSTANTS.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Mapping of hex colors to VISUAL_CONSTANTS
COLOR_MAPPING = {
    "#2166AC": "VISUAL_CONSTANTS.ST_BLUE",
    "#D6604D": "VISUAL_CONSTANTS.THETA_RED",
    "#F4A582": "VISUAL_CONSTANTS.INTERO_AMBER",
    "#41AB5D": "VISUAL_CONSTANTS.IGNITION_GREEN",
    "#762A83": "VISUAL_CONSTANTS.ALLOSTATIC_PURPLE",
    "#999999": "VISUAL_CONSTANTS.HC_GREY",
    "#2ecc71": "VISUAL_CONSTANTS.STATUS_PASS",
    "#e74c3c": "VISUAL_CONSTANTS.STATUS_FAIL",
    "#f39c12": "VISUAL_CONSTANTS.STATUS_ERROR",
    "#95a5a6": "VISUAL_CONSTANTS.STATUS_UNKNOWN",
}

# Non-APGI colors that should be mapped to closest APGI color
NON_APGI_MAPPING = {
    "#2874a6": "VISUAL_CONSTANTS.ST_BLUE",  # Similar to ST_BLUE
    "#155724": "VISUAL_CONSTANTS.IGNITION_GREEN",  # Similar to IGNITION_GREEN
    "#721c24": "VISUAL_CONSTANTS.THETA_RED",  # Similar to THETA_RED
    "#f8f9fa": "VISUAL_CONSTANTS.HC_GREY",  # Light grey
    "#212529": "VISUAL_CONSTANTS.ALLOSTATIC_PURPLE",  # Dark color
    "#ffffff": "VISUAL_CONSTANTS.HC_GREY",  # White/light
    "#3498db": "VISUAL_CONSTANTS.ST_BLUE",  # Similar blue
    "#9b59b6": "VISUAL_CONSTANTS.ALLOSTATIC_PURPLE",  # Similar purple
    "#e67e22": "VISUAL_CONSTANTS.INTERO_AMBER",  # Similar orange
    "#34495e": "VISUAL_CONSTANTS.ALLOSTATIC_PURPLE",  # Dark color
    "#555555": "VISUAL_CONSTANTS.HC_GREY",  # Grey
    "#00FF00": "VISUAL_CONSTANTS.IGNITION_GREEN",  # Green
    "#FF0000": "VISUAL_CONSTANTS.THETA_RED",  # Red
}


def needs_visual_constants_import(content: str) -> bool:
    """Check if file needs VISUAL_CONSTANTS import."""
    # Check if any colors are used
    hex_pattern = r"#[0-9A-Fa-f]{6}"
    has_colors = bool(re.search(hex_pattern, content))

    # Check if VISUAL_CONSTANTS is already imported
    has_import = "VISUAL_CONSTANTS" in content and "import" in content

    return has_colors and not has_import


def add_visual_constants_import(content: str) -> str:
    """Add VISUAL_CONSTANTS import to file."""
    # Find the right place to add import (after other imports)
    lines = content.split("\n")

    # Find the last import line
    last_import_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            last_import_idx = i

    if last_import_idx >= 0:
        # Add after last import
        lines.insert(last_import_idx + 1, "from utils.constants import VISUAL_CONSTANTS")
    else:
        # Add after docstring/shebang
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("#!") or (line.startswith('"""') or line.startswith("'''")):
                insert_idx = i + 1
            elif line.strip() and not line.startswith("#"):
                break
        lines.insert(insert_idx, "from utils.constants import VISUAL_CONSTANTS")

    return "\n".join(lines)


def fix_colors_in_file(file_path: Path) -> Tuple[bool, int]:
    """Fix color palette in a single file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, IOError):
        return False, 0

    original_content = content
    replacements = 0

    # Replace APGI colors
    for hex_color, constant in COLOR_MAPPING.items():
        if hex_color in content:
            # Only replace if not already using VISUAL_CONSTANTS
            if f"{constant}" not in content:
                content = content.replace(f'"{hex_color}"', f'"{constant}"')
                content = content.replace(f"'{hex_color}'", f"'{constant}'")
                replacements += 1

    # Replace non-APGI colors with closest APGI equivalent
    for hex_color, constant in NON_APGI_MAPPING.items():
        if hex_color in content:
            if f"{constant}" not in content:
                content = content.replace(f'"{hex_color}"', f'"{constant}"')
                content = content.replace(f"'{hex_color}'", f"'{constant}'")
                replacements += 1

    # Add import if needed
    if replacements > 0 and needs_visual_constants_import(content):
        content = add_visual_constants_import(content)

    # Write back if changed
    if content != original_content:
        try:
            file_path.write_text(content, encoding="utf-8")
            return True, replacements
        except IOError:
            return False, 0

    return False, 0


def main():
    """Fix color palette compliance across project."""
    project_root = Path(__file__).parent.parent
    python_files = list(project_root.rglob("*.py"))

    fixed = 0
    total_replacements = 0

    for file_path in python_files:
        # Skip test files and __pycache__
        if "test" in file_path.parts or "__pycache__" in file_path.parts or ".claude" in file_path.parts:
            continue

        success, replacements = fix_colors_in_file(file_path)
        if success:
            fixed += 1
            total_replacements += replacements
            print(f"✓ {file_path.relative_to(project_root)} ({replacements} replacements)")

    print(f"\nFixed {fixed} files with {total_replacements} total color replacements")


if __name__ == "__main__":
    main()
