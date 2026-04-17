#!/usr/bin/env python3
"""
Path Security Utility
=====================

Provides secure file path validation to prevent path traversal attacks,
symlink escapes, and other path-related security vulnerabilities.
"""

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def validate_file_path(file_path: PathLike, base_path: PathLike) -> Path:
    """
    Validate that a file path is secure and within the allowed base directory.

    This function performs comprehensive path validation to prevent:
    - Path traversal attacks (../ sequences)
    - Symlink-based directory escapes
    - Absolute path injection
    - Null byte injection
    - Unicode-based path obfuscation

    Args:
        file_path: The file path to validate (string or Path object)
        base_path: The allowed base/root directory (string or Path object)

    Returns:
        Path: The validated and resolved Path object

    Raises:
        TypeError: If file_path is None
        ValueError: If file_path is empty or path traversal is detected
    """
    # Check for None
    if file_path is None:
        raise TypeError("File path cannot be None")

    # Get raw string input BEFORE Path conversion (Path normalizes // and other patterns)
    raw_path_str = str(file_path)

    # Check for empty path
    if not raw_path_str or raw_path_str == ".":
        raise ValueError("File path cannot be empty")

    # Reject paths with null bytes
    if "\x00" in raw_path_str:
        raise ValueError("Path traversal detected: null byte in path")

    # Reject Windows-style absolute paths (even on non-Windows systems)
    # Matches patterns like C:, C:\, D:\, etc.
    if len(raw_path_str) >= 2 and raw_path_str[1] == ":":
        raise ValueError("Path traversal detected")

    # Check for unicode tricks
    # Right-to-left override character can be used to obfuscate extensions
    if "\u202e" in raw_path_str:
        raise ValueError("Path traversal detected: Unicode RLO character in path")

    # Zero-width space and BOM can cause confusion
    if "\u200b" in raw_path_str or "\ufeff" in raw_path_str:
        raise ValueError("Path traversal detected: Unicode control characters in path")

    # Check for path traversal attempts BEFORE Path conversion
    # Path() normalizes paths, so we check the raw input for attack patterns
    # Match patterns like: ../, ..\, /.., .. at end, or embedded ..
    normalized_str = raw_path_str.replace("\\", "/")
    if (
        "/../" in normalized_str
        or normalized_str.startswith("../")
        or normalized_str.endswith("/..")
        or "/.." in normalized_str
    ):
        raise ValueError("Path traversal detected: '..' sequence in path")

    # Reject double slashes (//) which can be used in path normalization attacks
    # Note: // at the start can be valid for network paths, but // in the middle is suspicious
    if "//" in normalized_str:
        raise ValueError("Path traversal detected: double slash in path")

    # Reject standalone .. sequences (not just as path components)
    if ".." in raw_path_str:
        raise ValueError("Path traversal detected: '..' sequence in path")

    # Now convert to Path object
    path = Path(file_path)
    # Resolve base to its real path (handles macOS /var -> /private/var symlinks)
    base = Path(os.path.realpath(str(base_path)))

    # Resolve the path to its canonical form (resolves symlinks)
    # Handle both relative and absolute paths:
    # - If relative: join with base, then resolve
    # - If absolute: resolve to real path, then check if within base
    if path.is_absolute():
        # For absolute paths, resolve to real path first (handles /var -> /private)
        resolved_input = Path(os.path.realpath(str(path)))
        # Check if resolved path is within base
        try:
            resolved_input.relative_to(base)
            full_path = resolved_input
        except ValueError:
            raise ValueError("Path traversal detected")
    else:
        # For relative paths, join with base
        full_path = base / path

    try:
        resolved_path = Path(os.path.realpath(str(full_path)))
    except (OSError, ValueError) as e:
        raise ValueError(f"Path traversal detected: Cannot resolve path: {e}")

    # Check if path exists and is a symlink pointing elsewhere
    # Use lstat to check for symlinks without following them
    try:
        if full_path.is_symlink():
            # Symlinks are allowed but must resolve within base
            pass
    except (OSError, ValueError):
        pass

    # Final safety check: ensure resolved path is within base directory
    try:
        resolved_path.relative_to(base)
    except ValueError:
        raise ValueError("Path traversal detected")

    # Additional check: ensure file exists and has read permissions
    if resolved_path.exists():
        # Check file size (reject files > 1GB to prevent DoS)
        try:
            size = resolved_path.stat().st_size
            if size > (1024 * 1024 * 1024):  # 1GB
                raise ValueError("Path traversal detected: File too large")
        except OSError:
            pass

        # Check file permissions (reject files with no read permission)
        try:
            mode = resolved_path.stat().st_mode
            # Check if owner has no read permission
            if not (mode & 0o400):  # Owner read permission
                raise ValueError("Path traversal detected: File not readable")
        except OSError:
            pass

    return resolved_path
