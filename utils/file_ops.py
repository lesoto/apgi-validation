#!/usr/bin/env python3
"""
Unified File Operations Wrapper
===============================

Combines path validation, TOCTOU mitigation, and secure file handling
into a single hardened interface. This module should be the ONLY way
files are accessed in the APGI framework.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Union

# Add parent directory to path for standalone execution
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.path_security import validate_file_path
from utils.toctou_mitigation import get_secure_file_operations

logger = logging.getLogger("file_ops")
_ops = get_secure_file_operations()


def read_text_file(file_path: Union[str, Path], base_path: Union[str, Path]) -> str:
    """Read a text file securely."""
    valid_path = validate_file_path(file_path, base_path)
    content = _ops.safe_read(str(valid_path))
    if content is None:
        raise IOError(f"Could not read file safely: {file_path}")
    return content


def write_text_file(
    file_path: Union[str, Path], base_path: Union[str, Path], content: str
) -> None:
    """Write a text file securely."""
    valid_path = validate_file_path(file_path, base_path)
    success = _ops.safe_write(str(valid_path), content)
    if not success:
        raise IOError(f"Could not write file safely: {file_path}")


def read_json_file(
    file_path: Union[str, Path], base_path: Union[str, Path]
) -> Dict[str, Any]:
    """Read and parse a JSON file securely."""
    content = read_text_file(file_path, base_path)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file {file_path}: {e}")


def write_json_file(
    file_path: Union[str, Path], base_path: Union[str, Path], data: Any, indent: int = 2
) -> None:
    """Serialize and write JSON data to a file securely."""
    content = json.dumps(data, indent=indent)
    write_text_file(file_path, base_path, content)


def delete_file(file_path: Union[str, Path], base_path: Union[str, Path]) -> None:
    """Delete a file securely."""
    valid_path = validate_file_path(file_path, base_path)
    success = _ops.safe_delete(str(valid_path))
    if not success:
        raise IOError(f"Could not delete file safely: {file_path}")


def file_exists(file_path: Union[str, Path], base_path: Union[str, Path]) -> bool:
    """Check if a file exists securely."""
    try:
        valid_path = validate_file_path(file_path, base_path)
        return _ops.safe_exists(str(valid_path))
    except ValueError:
        return False


def make_directory(
    dir_path: Union[str, Path], base_path: Union[str, Path], mode: int = 0o755
) -> None:
    """Create a directory securely."""
    # Allow target path to not exist yet for directory creation
    valid_path = validate_file_path(dir_path, base_path)
    success = _ops.safe_mkdir(str(valid_path), mode)
    if not success:
        raise IOError(f"Could not create directory safely: {dir_path}")
