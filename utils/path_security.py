"""
Path traversal protection utilities for security.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional


def validate_file_path(file_path: str, base_dir: Optional[str] = None) -> Path:
    """
    Validate file path to prevent path traversal attacks.
    """
    if file_path is None:
        raise TypeError("File path cannot be None")

    path_str = str(file_path)
    if not path_str or not path_str.strip():
        raise ValueError("File path cannot be empty")

    # 1. Immediate check for null bytes
    if "\x00" in path_str:
        raise ValueError("Path traversal detected: Null bytes in path")

    # 2. Check for explicit traversal sequences BEFORE resolution
    # This catches attempts to escape before the OS/Pathlib resolves them
    normalized_input = path_str.replace("\\", "/")

    # Strict check for any ".." component
    parts = normalized_input.split("/")
    if any(p == ".." for p in parts):
        raise ValueError(
            f"Path traversal detected: {file_path} contains parent directory references"
        )

    # 3. Handle Absolute Paths
    input_path = Path(file_path)
    project_root = Path(__file__).parent.parent.absolute()
    temp_dir = Path(tempfile.gettempdir()).absolute()

    # If a base_dir is provided, use it as the primary allowed root
    if base_dir:
        base_path = Path(base_dir).absolute()
        allowed_roots = [base_path]
    else:
        base_path = Path(os.getcwd()).absolute()
        allowed_roots = [base_path, project_root, temp_dir]

    # Ensure Windows absolute paths are caught even on Posix
    is_win_absolute = bool(re.match(r"^[a-zA-Z]:[\\/]", str(file_path)))

    if input_path.is_absolute() or is_win_absolute:
        # On Posix, any path starting with [A-Za-z]: should be rejected as an absolute path attempt
        if os.name != "nt" and is_win_absolute:
            raise ValueError(
                f"Path traversal detected: Absolute Windows path {file_path} is not allowed on this system"
            )

        # Check if absolute path is within any allowed root
        is_in_allowed = False
        for root in allowed_roots:
            try:
                # Resolve both to handle symlinks correctly in containment check
                abs_input = os.path.realpath(str(input_path.absolute()))
                abs_root = os.path.realpath(str(root))
                if os.path.commonpath([abs_input, abs_root]) == abs_root:
                    is_in_allowed = True
                    break
            except (ValueError, OSError):
                continue
        if not is_in_allowed:
            raise ValueError(
                f"Path traversal detected: Absolute path {file_path} is outside allowed directories"
            )

    # 4. Resolve path and check containment (Symlink Traversal Check)
    try:
        # Use realpath to resolve all symlinks and '..'
        resolved_path = Path(os.path.realpath(path_str))

        # The REAL physical path must stay within allowed roots
        is_safe = False
        for root in allowed_roots:
            try:
                root_real = os.path.realpath(str(root))
                if os.path.commonpath([str(resolved_path), root_real]) == root_real:
                    is_safe = True
                    break
            except (ValueError, OSError):
                continue

        if not is_safe:
            raise ValueError(
                f"Path traversal detected: {file_path} points outside allowed directories"
            )
    except (OSError, ValueError) as e:
        if isinstance(e, ValueError) and "Path traversal" in str(e):
            raise
        raise ValueError(f"Invalid file path: {file_path}")

    # 5. Final component checks
    for part in resolved_path.parts:
        if any(char in part for char in ["\u202e", "\u200b", "\ufeff"]):
            raise ValueError("Path traversal detected: Unicode tricks detected")
        if len(part) > 255:
            raise ValueError("Path traversal detected: Path component too long")

    # 6. TOCTOU and existence checks
    if resolved_path.exists():
        if not os.access(resolved_path, os.R_OK):
            raise ValueError("Path traversal detected: Inaccessible file")

        if resolved_path.is_file():
            if resolved_path.stat().st_size > 1024 * 1024 * 1024:
                raise ValueError("Path traversal detected: File too large")

    return resolved_path


def safe_file_operation(
    file_path: str, base_dir: Optional[str] = None, operation: str = "read"
) -> Path:
    """
    Safely perform file operations with path traversal protection.

    Args:
        file_path: The file path to validate
        base_dir: Base directory for file operations
        operation: Type of operation ("read", "write", "config")

    Returns:
        Path object for the validated file path
    """
    if operation == "config":
        # Config files have more restrictive base directory
        if base_dir is None:
            base_dir = os.path.join(os.getcwd(), "config")
    elif operation == "write":
        # Output files can be in current working directory or subdirectories
        if base_dir is None:
            base_dir = os.getcwd()
    else:  # read
        # Input files can be in current working directory or data directories
        if base_dir is None:
            base_dir = os.getcwd()

    return validate_file_path(file_path, base_dir)


def validate_log_path(log_path: str) -> Path:
    """
    Validate log file path to prevent path traversal in environment variables.

    Args:
        log_path: The log file path from environment variable

    Returns:
        Path object for the validated log path

    Raises:
        ValueError: If path traversal is detected
    """
    if not log_path:
        raise ValueError("Log path cannot be empty")

    # Log files should be within the logs directory or current working directory
    base_dirs = [os.path.join(os.getcwd(), "logs"), os.getcwd()]

    for base_dir in base_dirs:
        try:
            return validate_file_path(log_path, base_dir)
        except ValueError:
            continue

    raise ValueError(f"Log path {log_path} is not in allowed directories")
