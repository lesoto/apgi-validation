"""
Security logging integration guide.
This file provides examples and utilities for integrating security audit logging into the codebase.
============================================================================================
"""

# Add project root to Python path for imports
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the audit logger
from utils.security_audit_logger import (
    get_audit_logger,
    log_delete,
    log_import,
    log_read,
    log_write,
)


# Example: Add security logging to file operations
def secure_file_read(file_path: str) -> str:
    """Read file content with security logging."""
    from pathlib import Path

    path = Path(file_path)

    # Log the read operation
    log_read(str(path))

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except Exception as e:
        log_read(str(path), success=False, error=str(e))
        raise


def secure_file_write(file_path: str, content: str) -> None:
    """Write content to file with security logging."""
    from pathlib import Path

    path = Path(file_path)

    # Log the write operation
    log_write(str(path))

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        log_write(str(path), success=False, error=str(e))
        raise


def secure_file_delete(file_path: str) -> None:
    """Delete file with security logging."""
    from pathlib import Path

    path = Path(file_path)

    # Log the delete operation
    log_delete(str(path))

    try:
        path.unlink()
    except Exception as e:
        log_delete(str(path), success=False, error=str(e))
        raise


# Example: Add security logging to module imports
def secure_import_module(module_name: str, module_path: str):
    """Import module with security logging."""
    import importlib.util
    from pathlib import Path

    path = Path(module_path)

    # Log the import operation
    log_import(str(path))

    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    except Exception as e:
        log_import(str(path), success=False, error=str(e))
        raise


# Example: Add security logging to path resolution
def secure_resolve_path(path_str: str) -> str:
    """Resolve path with security logging."""
    from pathlib import Path

    logger = get_audit_logger()

    try:
        resolved = str(Path(path_str).resolve())
        logger.log_path_resolution(path_str, resolved, "resolve", success=True)
        return resolved
    except Exception as e:
        logger.log_path_resolution(
            path_str, "unknown", "resolve", success=False, error=str(e)
        )
        raise


# Integration checklist:
# - [ ] Add security logging to all file read operations
# - [ ] Add security logging to all file write operations
# - [ ] Add security logging to all file delete operations
# - [ ] Add security logging to all module imports
# - [ ] Add security logging to all path resolution operations
# - [ ] Add security logging to all permission checks
# - [ ] Review and update security audit logs regularly
# - [ ] Set up automated security audit log rotation

# Security audit log file location:
# - Default: security_audit.log in project root
# - Can be customized in utils/security_audit_logger.py

# To use the audit logger in your code:
# 1. Import: from utils.security_audit_logger import log_read, log_write, log_delete
# 2. Call log functions before/after file operations
# 3. Use decorators for automatic logging: @audit_file_operation("read")

# Example of using decorators:
# @audit_file_operation("read")
# def read_config(file_path: str) -> dict:
#     with open(file_path, ", encoding=\"utf-8\"r") as f:
#         return json.load(f)

# The decorator will automatically log the operation, success/failure, and any errors

# Security audit log format:
# timestamp | level | operation | message
# 2024-01-01T12:00:00.000000 | INFO | read | File: /path/to/file | User: unknown | Success: True
# 2024-01-01T12:00:01.000000 | WARNING | read | File: /path/to/file | User: unknown | Success: False | Error: File not found
