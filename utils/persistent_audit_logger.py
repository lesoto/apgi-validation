"""
Audit log persistence implementation.
Implements persistent audit logging to disk for forensic analysis.
=============================================================
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import threading


class PersistentAuditLogger:
    """Persistent audit logger that writes to disk for forensic analysis."""

    def __init__(
        self,
        log_file: str = "audit.log",
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        rotation_enabled: bool = True,
    ):
        """
        Initialize persistent audit logger.

        Args:
            log_file: Path to audit log file
            max_file_size: Maximum size before rotation (bytes)
            backup_count: Number of backup files to keep
            rotation_enabled: Whether to enable log rotation
        """
        self.log_file = Path(log_file)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.rotation_enabled = rotation_enabled

        # Thread lock for concurrent writes
        self._lock = threading.Lock()

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.logger = logging.getLogger("persistent_audit")
        self.logger.setLevel(logging.INFO)

        # File handler with rotation
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(operation)s | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # In-memory audit trail for recent operations
        self.audit_trail: List[Dict] = []
        self.max_trail_size = 1000

        # Statistics
        self.stats = {
            "total_operations": 0,
            "by_operation": {},
            "by_status": {"success": 0, "failure": 0},
            "last_rotation": None,
        }

    def log_operation(
        self,
        operation: str,
        details: Dict,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Log an operation to persistent storage.

        Args:
            operation: Type of operation (read, write, delete, import, etc.)
            details: Operation details
            success: Whether operation succeeded
            error: Error message if operation failed
        """
        with self._lock:
            # Create log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "operation": operation,
                "details": details,
                "success": success,
                "error": error,
            }

            # Update statistics
            self.stats["total_operations"] += 1
            self.stats["by_operation"][operation] = (
                self.stats["by_operation"].get(operation, 0) + 1
            )
            if success:
                self.stats["by_status"]["success"] += 1
            else:
                self.stats["by_status"]["failure"] += 1

            # Add to in-memory trail
            self.audit_trail.append(log_entry)
            if len(self.audit_trail) > self.max_trail_size:
                self.audit_trail.pop(0)

            # Write to file
            self._write_to_file(log_entry)

            # Check for rotation
            if self.rotation_enabled:
                self._check_rotation()

    def _write_to_file(self, log_entry: Dict) -> None:
        """Write log entry to file."""
        try:
            log_line = json.dumps(log_entry)
            self.logger.info(log_line)
        except (KeyboardInterrupt, SystemExit):
            raise
        except (TypeError, ValueError, OSError) as primary_err:
            # Fallback: write directly to file
            try:
                with open(self.log_file, ', encoding="utf-8"a') as f:
                    f.write(f"{json.dumps(log_entry)}\n")
            except (KeyboardInterrupt, SystemExit):
                raise
            except (OSError, TypeError, ValueError) as e2:
                print(f"Failed to write audit log (primary={primary_err!r}): {e2}")

    def _check_rotation(self) -> None:
        """Check if log file needs rotation."""
        try:
            if self.log_file.stat().st_size > self.max_file_size:
                self._rotate_log()
        except (KeyboardInterrupt, SystemExit):
            raise
        except OSError as e:
            # Log at warning level; rotating is best-effort, never silent
            print(f"Audit log rotation check failed: {e}")

    def _rotate_log(self) -> None:
        """Rotate log file."""
        try:
            # Remove oldest backup if needed
            oldest_backup = self.log_file.with_suffix(f".log.{self.backup_count}")
            if oldest_backup.exists():
                oldest_backup.unlink()

            # Shift backup files
            for i in range(self.backup_count - 1, 0, -1):
                old_backup = self.log_file.with_suffix(f".log.{i}")
                new_backup = self.log_file.with_suffix(f".log.{i + 1}")
                if old_backup.exists():
                    old_backup.rename(new_backup)

            # Move current log to .log.1
            backup1 = self.log_file.with_suffix(".log.1")
            self.log_file.rename(backup1)

            # Create new log file
            self.log_file.touch()

            # Update statistics
            self.stats["last_rotation"] = datetime.utcnow().isoformat()

            # Reinitialize logger with new file
            self._reinitialize_logger()

        except Exception as e:
            print(f"Log rotation failed: {e}")

    def _reinitialize_logger(self) -> None:
        """Reinitialize logger after rotation."""
        # Remove old handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add new handler
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(operation)s | %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_recent_operations(
        self, operation: Optional[str] = None, limit: int = 100
    ) -> List[Dict]:
        """
        Get recent operations from audit trail.

        Args:
            operation: Filter by operation type (optional)
            limit: Maximum number of operations to return

        Returns:
            List of recent operations
        """
        with self._lock:
            if operation:
                filtered = [
                    op for op in self.audit_trail if op["operation"] == operation
                ]
                return filtered[-limit:]
            return self.audit_trail[-limit:]

    def get_statistics(self) -> Dict:
        """Get audit statistics."""
        with self._lock:
            return self.stats.copy()

    def export_audit_trail(self, output_file: str) -> None:
        """
        Export complete audit trail to file.

        Args:
            output_file: Path to output file
        """
        with self._lock:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "exported_at": datetime.utcnow().isoformat(),
                        "statistics": self.stats,
                        "operations": self.audit_trail,
                    },
                    f,
                    indent=2,
                )

    def clear_audit_trail(self) -> None:
        """Clear in-memory audit trail (keeps file logs)."""
        with self._lock:
            self.audit_trail.clear()

    def search_audit_trail(self, query: str, field: str = "operation") -> List[Dict]:
        """
        Search audit trail for matching entries.

        Args:
            query: Search query
            field: Field to search in (operation, details, error)

        Returns:
            List of matching operations
        """
        with self._lock:
            results = []
            for entry in self.audit_trail:
                if field == "details":
                    # Search in details dict
                    details_str = json.dumps(entry.get("details", {}))
                    if query.lower() in details_str.lower():
                        results.append(entry)
                else:
                    # Search in operation or error
                    value = entry.get(field, "")
                    if query.lower() in str(value).lower():
                        results.append(entry)
            return results


# Global instance
_persistent_audit_logger: Optional[PersistentAuditLogger] = None


def get_persistent_audit_logger() -> PersistentAuditLogger:
    """Get or create global persistent audit logger instance."""
    global _persistent_audit_logger
    if _persistent_audit_logger is None:
        _persistent_audit_logger = PersistentAuditLogger()
    return _persistent_audit_logger


def log_operation_persistent(
    operation: str, details: Dict, success: bool = True, error: Optional[str] = None
) -> None:
    """
    Log operation to persistent audit logger.

    Args:
        operation: Type of operation
        details: Operation details
        success: Whether operation succeeded
        error: Error message if operation failed
    """
    logger = get_persistent_audit_logger()
    logger.log_operation(operation, details, success, error)


# Convenience functions for common operations
def log_read_persistent(
    file_path: str, success: bool = True, error: Optional[str] = None
) -> None:
    """Log file read operation."""
    log_operation_persistent("read", {"file_path": file_path}, success, error)


def log_write_persistent(
    file_path: str, success: bool = True, error: Optional[str] = None
) -> None:
    """Log file write operation."""
    log_operation_persistent("write", {"file_path": file_path}, success, error)


def log_delete_persistent(
    file_path: str, success: bool = True, error: Optional[str] = None
) -> None:
    """Log file delete operation."""
    log_operation_persistent("delete", {"file_path": file_path}, success, error)


def log_import_persistent(
    module_name: str, success: bool = True, error: Optional[str] = None
) -> None:
    """Log module import operation."""
    log_operation_persistent("import", {"module": module_name}, success, error)
