"""
Security audit logger for file operations.
Logs all file access, path resolution, and permission checks for security auditing.
============================================================================================
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional
from functools import wraps
import traceback


class SecurityAuditLogger:
    """Audit logger for security-relevant file operations."""

    def __init__(
        self, log_file: str = "security_audit.log", log_level: int = logging.INFO
    ):
        """
        Initialize security audit logger.

        Args:
            log_file: Path to audit log file
            log_level: Logging level
        """
        self.log_file = Path(log_file)
        self.log_level = log_level

        # Create audit logger
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(log_level)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            # File handler for audit logs
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(log_level)

            # Format: timestamp | level | operation | details
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(operation)s | %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # In-memory audit trail for recent operations
        self.audit_trail: list = []

    def log_file_access(
        self,
        operation: str,
        file_path: str,
        user: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Log file access operation.

        Args:
            operation: Type of operation (read, write, delete, etc.)
            file_path: Path to file being accessed
            user: User performing the operation (if available)
            success: Whether operation succeeded
            error: Error message if operation failed
            **kwargs: Additional context
        """
        log_entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "operation": operation,
            "file_path": str(file_path),
            "user": user,
            "success": success,
            "error": error,
            "context": kwargs,
        }

        # Add to in-memory trail (keep last 1000 entries)
        self.audit_trail.append(log_entry)
        if len(self.audit_trail) > 1000:
            self.audit_trail.pop(0)

        # Log to file
        extra = {"operation": operation}
        if success:
            self.logger.info(
                f"File: {file_path} | User: {user or 'unknown'} | Success: True",
                extra=extra,
            )
        else:
            self.logger.warning(
                f"File: {file_path} | User: {user or 'unknown'} | Success: False | Error: {error}",
                extra=extra,
            )

    def log_path_resolution(
        self,
        original_path: str,
        resolved_path: str,
        method: str,
        success: bool = True,
        **kwargs,
    ) -> None:
        """
        Log path resolution operation.

        Args:
            original_path: Original path string
            resolved_path: Resolved absolute path
            method: Resolution method used
            success: Whether resolution succeeded
            **kwargs: Additional context
        """
        log_entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "operation": "path_resolution",
            "original_path": original_path,
            "resolved_path": resolved_path,
            "method": method,
            "success": success,
            "context": kwargs,
        }

        self.audit_trail.append(log_entry)
        if len(self.audit_trail) > 1000:
            self.audit_trail.pop(0)

        extra = {"operation": "path_resolution"}
        if success:
            self.logger.info(
                f"Resolved: {original_path} -> {resolved_path} | Method: {method}",
                extra=extra,
            )
        else:
            self.logger.warning(
                f"Resolution failed: {original_path} | Method: {method}",
                extra=extra,
            )

    def log_permission_check(
        self,
        file_path: str,
        permission_type: str,
        granted: bool,
        user: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Log permission check operation.

        Args:
            file_path: Path to file being checked
            permission_type: Type of permission (read, write, execute)
            granted: Whether permission was granted
            user: User requesting permission
            **kwargs: Additional context
        """
        log_entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "operation": "permission_check",
            "file_path": str(file_path),
            "permission_type": permission_type,
            "granted": granted,
            "user": user,
            "context": kwargs,
        }

        self.audit_trail.append(log_entry)
        if len(self.audit_trail) > 1000:
            self.audit_trail.pop(0)

        extra = {"operation": "permission_check"}
        if granted:
            self.logger.info(
                f"Permission granted: {permission_type} | File: {file_path} | User: {user or 'unknown'}",
                extra=extra,
            )
        else:
            self.logger.warning(
                f"Permission denied: {permission_type} | File: {file_path} | User: {user or 'unknown'}",
                extra=extra,
            )

    def log_configuration_change(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        user: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Log configuration change operation.

        Args:
            config_key: Configuration key being changed
            old_value: Previous value
            new_value: New value
            user: User making the change
            **kwargs: Additional context
        """
        log_entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "operation": "config_change",
            "config_key": config_key,
            "old_value": old_value,
            "new_value": new_value,
            "user": user,
            "context": kwargs,
        }

        self.audit_trail.append(log_entry)
        if len(self.audit_trail) > 1000:
            self.audit_trail.pop(0)

        extra = {"operation": "config_change"}
        self.logger.info(
            f"Config changed: {config_key} | Old: {old_value} | New: {new_value} | User: {user or 'unknown'}",
            extra=extra,
        )

    def export_audit_trail(self, output_file: str, format: str = "json") -> None:
        """
        Export audit trail to file.

        Args:
            output_file: Path to output file
            format: Export format (json or csv)
        """
        if format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.audit_trail, f, indent=2)
        elif format == "csv":
            import csv

            with open(output_file, "w", newline="", encoding="utf-8") as f:
                if self.audit_trail:
                    writer = csv.DictWriter(f, fieldnames=self.audit_trail[0].keys())
                    writer.writeheader()
                    writer.writerows(self.audit_trail)

        self.log_file_access(
            "export", output_file, success=True, context={"format": format}
        )

    def get_recent_operations(self, n: int = 100) -> list:
        """
        Get recent audit log entries.

        Args:
            n: Number of recent entries to retrieve

        Returns:
            List of recent audit entries
        """
        return self.audit_trail[-n:]

    def search_audit_trail(
        self,
        operation: Optional[str] = None,
        file_path: Optional[str] = None,
        user: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list:
        """
        Search audit trail for specific operations.

        Args:
            operation: Filter by operation type
            file_path: Filter by file path
            user: Filter by user
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List of matching audit entries
        """
        results = []

        for entry in self.audit_trail:
            match = True

            if operation and entry.get("operation") != operation:
                match = False

            if file_path and file_path not in str(entry.get("file_path", "")):
                match = False

            if user and entry.get("user") != user:
                match = False

            if start_time:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time < start_time:
                    match = False

            if end_time:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time > end_time:
                    match = False

            if match:
                results.append(entry)

        return results


# Global audit logger instance
_audit_logger: Optional[SecurityAuditLogger] = None


def get_audit_logger() -> SecurityAuditLogger:
    """Get or create global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = SecurityAuditLogger()
    return _audit_logger


def audit_file_operation(operation: str, logger: Optional[SecurityAuditLogger] = None):
    """
    Decorator to audit file operations.

    Args:
        operation: Type of operation being audited
        logger: Optional logger instance (uses global if not provided)

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = logger if logger is not None else get_audit_logger()
            file_path = None

            # Try to extract file path from arguments
            if args and isinstance(args[0], (str, Path)):
                file_path = str(args[0])
            elif "file_path" in kwargs:
                file_path = str(kwargs["file_path"])
            elif "path" in kwargs:
                file_path = str(kwargs["path"])

            try:
                result = func(*args, **kwargs)
                if file_path:
                    audit_logger.log_file_access(
                        operation,
                        file_path,
                        success=True,
                        function=func.__name__,
                    )
                return result
            except Exception as e:
                if file_path:
                    audit_logger.log_file_access(
                        operation,
                        file_path,
                        success=False,
                        error=str(e),
                        function=func.__name__,
                        traceback=traceback.format_exc(),
                    )
                raise

        return wrapper

    return decorator


def audit_path_resolution(method: str, logger: Optional[SecurityAuditLogger] = None):
    """
    Decorator to audit path resolution operations.

    Args:
        method: Resolution method being used
        logger: Optional logger instance (uses global if not provided)

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = logger if logger is not None else get_audit_logger()
            original_path = None

            # Try to extract path from arguments
            if args and isinstance(args[0], (str, Path)):
                original_path = str(args[0])

            try:
                result = func(*args, **kwargs)

                if original_path and isinstance(result, (str, Path)):
                    audit_logger.log_path_resolution(
                        original_path, str(result), method, success=True
                    )

                return result
            except Exception as e:
                if original_path:
                    audit_logger.log_path_resolution(
                        original_path,
                        "unknown",
                        method,
                        success=False,
                        error=str(e),
                    )
                raise

        return wrapper

    return decorator


def audit_permission_check(
    permission_type: str, logger: Optional[SecurityAuditLogger] = None
):
    """
    Decorator to audit permission check operations.

    Args:
        permission_type: Type of permission being checked
        logger: Optional logger instance (uses global if not provided)

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = logger if logger is not None else get_audit_logger()
            file_path = None

            # Try to extract file path from arguments
            if args and len(args) > 0 and isinstance(args[0], (str, Path)):
                file_path = str(args[0])
            elif "file_path" in kwargs:
                file_path = str(kwargs["file_path"])

            try:
                result = func(*args, **kwargs)
                granted = bool(result)

                if file_path:
                    audit_logger.log_permission_check(
                        file_path, permission_type, granted, function=func.__name__
                    )

                return result
            except Exception as e:
                if file_path:
                    audit_logger.log_permission_check(
                        file_path,
                        permission_type,
                        granted=False,
                        error=str(e),
                        function=func.__name__,
                    )
                raise

        return wrapper

    return decorator


# Convenience functions for common operations
def log_read(
    file_path: str,
    success: bool = True,
    error: Optional[str] = None,
    logger: Optional[SecurityAuditLogger] = None,
) -> None:
    """Log file read operation."""
    audit_logger = logger if logger is not None else get_audit_logger()
    audit_logger.log_file_access("read", file_path, success=success, error=error)


def log_write(
    file_path: str,
    success: bool = True,
    error: Optional[str] = None,
    logger: Optional[SecurityAuditLogger] = None,
) -> None:
    """Log file write operation."""
    audit_logger = logger if logger is not None else get_audit_logger()
    audit_logger.log_file_access("write", file_path, success=success, error=error)


def log_delete(
    file_path: str,
    success: bool = True,
    error: Optional[str] = None,
    logger: Optional[SecurityAuditLogger] = None,
) -> None:
    """Log file delete operation."""
    audit_logger = logger if logger is not None else get_audit_logger()
    audit_logger.log_file_access("delete", file_path, success=success, error=error)


def log_import(
    module_path: str,
    success: bool = True,
    error: Optional[str] = None,
    logger: Optional[SecurityAuditLogger] = None,
) -> None:
    """Log module import operation."""
    audit_logger = logger if logger is not None else get_audit_logger()
    audit_logger.log_file_access("import", module_path, success=success, error=error)
