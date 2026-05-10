#!/usr/bin/env python3
"""
Add LEVEL DESIGNATION declarations to Python files.

This script automatically adds LEVEL DESIGNATION declarations to files
that are missing them, based on their location and content.
"""

import ast
import re
from pathlib import Path
from typing import Optional, Tuple

# Mapping of file patterns to level designations
LEVEL_MAPPING = {
    # Level 1: Thermodynamic
    "apgi_core/": (1, "thermodynamic"),
    "Theory/": (2, "information-theoretic"),  # Theory modules bridge to L1
    "Validation/": (3, "algorithmic/mathematical"),  # Validation bridges to L2
    "Falsification/": (3, "algorithmic/mathematical"),  # Falsification bridges to L2
    "utils/": (2, "information-theoretic"),  # Utils bridge to L1
    "main.py": (3, "algorithmic/mathematical"),
}


def get_level_for_file(file_path: Path) -> Optional[Tuple[int, str]]:
    """Determine the level designation for a file based on its path."""
    path_str = str(file_path)

    for pattern, level_info in LEVEL_MAPPING.items():
        if pattern in path_str:
            return level_info

    return None


def get_bridge_requirements(level: int) -> list:
    """Get required bridge levels for a given level."""
    bridges = {
        1: [],
        2: [1],
        3: [2, 1],
    }
    return bridges.get(level, [])


def has_level_designation(content: str) -> bool:
    """Check if file already has LEVEL DESIGNATION."""
    return "LEVEL DESIGNATION:" in content


def add_level_designation(file_path: Path) -> bool:
    """Add LEVEL DESIGNATION to a file if missing."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, IOError):
        return False

    # Skip if already has designation
    if has_level_designation(content):
        return False

    # Get level for this file
    level_info = get_level_for_file(file_path)
    if not level_info:
        return False

    level, level_name = level_info
    bridges = get_bridge_requirements(level)

    # Parse to find docstring location
    try:
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree)
    except SyntaxError:
        return False

    # Build new docstring
    bridge_lines = ""
    if bridges:
        bridge_lines = "\n".join([f"Bridge to Level {b}" for b in bridges])
        bridge_lines = "\n" + bridge_lines

    new_docstring = f'''"""
LEVEL DESIGNATION: Level {level} ({level_name})
{bridge_lines}
"""'''

    # If file has no docstring, add one at the beginning
    if not docstring:
        # Find the first non-comment, non-shebang line
        lines = content.split("\n")
        insert_pos = 0

        for i, line in enumerate(lines):
            if line.startswith("#!") or line.startswith("#"):
                insert_pos = i + 1
            elif line.strip():
                break

        new_lines = lines[:insert_pos] + [new_docstring, ""] + lines[insert_pos:]
        new_content = "\n".join(new_lines)
    else:
        # Replace existing docstring
        # Find the docstring in the content
        docstring_pattern = r'"""[\s\S]*?"""'
        new_content = re.sub(docstring_pattern, new_docstring, content, count=1)

    try:
        file_path.write_text(new_content, encoding="utf-8")
        return True
    except IOError:
        return False


def main():
    """Add level designations to all Python files."""
    project_root = Path(__file__).parent.parent
    python_files = list(project_root.rglob("*.py"))

    added = 0
    skipped = 0

    for file_path in python_files:
        # Skip test files and __pycache__
        if "test" in file_path.parts or "__pycache__" in file_path.parts or ".claude" in file_path.parts:
            continue

        if add_level_designation(file_path):
            added += 1
            print(f"✓ {file_path.relative_to(project_root)}")
        else:
            skipped += 1

    print(f"\nAdded level designations to {added} files")
    print(f"Skipped {skipped} files")


if __name__ == "__main__":
    main()
