"""
Path traversal protection utilities for security.
"""

import os
from pathlib import Path
from typing import Optional


def validate_file_path(file_path: str, base_dir: Optional[str] = None) -> Path:
    """
    Validate file path to prevent path traversal attacks.

    Args:
        file_path: The file path to validate
        base_dir: Base directory that files should be within (defaults to current working directory)

    Returns:
        Path object for the validated file path

    Raises:
        ValueError: If path traversal is detected or path is invalid
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    # Convert to Path object for normalization
    try:
        path = Path(file_path).resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid file path: {file_path}") from e

    # Set base directory if not provided
    if base_dir is None:
        base_dir = os.getcwd()

    base_path = Path(base_dir).resolve()

    # Check if the path is within the base directory
    try:
        relative_path = path.relative_to(base_path)
    except ValueError as e:
        raise ValueError(
            f"Path traversal detected: {file_path} is outside base directory {base_dir}"
        ) from e

    # Additional security checks
    if ".." in str(relative_path):
        raise ValueError(
            f"Path traversal detected: {file_path} contains parent directory references"
        )

    return path


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
