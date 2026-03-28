"""
TOCTOU mitigation with file locking.
Implements Time-OfCheck-TimeOfUse race condition mitigation with file locking.
=================================================================
"""

import os
import fcntl
import stat
from pathlib import Path
from typing import Optional
import logging


class FileLock:
    """File locking for TOCTOU mitigation."""

    def __init__(self, lock_file: str, timeout: float = 10.0):
        """
        Initialize file lock.

        Args:
            lock_file: Path to lock file
            timeout: Maximum time to wait for lock acquisition
        """
        self.lock_file = Path(lock_file)
        self.timeout = timeout
        self.lock_fd = None
        self.logger = logging.getLogger("file_lock")

        # Ensure lock file directory exists
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def acquire(self) -> bool:
        """
        Acquire file lock.

        Returns:
            True if lock acquired, False otherwise
        """
        try:
            # Open lock file
            self.lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_WRONLY, 0o600)

            # Try to acquire lock (non-blocking)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            self.logger.debug(f"Acquired lock: {self.lock_file}")
            return True

        except (IOError, OSError) as e:
            if self.lock_fd is not None:
                try:
                    os.close(self.lock_fd)
                except (IOError, OSError):
                    pass
                self.lock_fd = None

            self.logger.warning(f"Could not acquire lock: {e}")
            return False

    def acquire_blocking(self) -> bool:
        """
        Acquire file lock with blocking.

        Returns:
            True if lock acquired, False otherwise
        """
        try:
            # Open lock file
            self.lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_WRONLY, 0o600)

            # Try to acquire lock (blocking)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX)

            self.logger.debug(f"Acquired lock (blocking): {self.lock_file}")
            return True

        except (IOError, OSError) as e:
            if self.lock_fd is not None:
                try:
                    os.close(self.lock_fd)
                except (IOError, OSError):
                    pass
                self.lock_fd = None

            self.logger.error(f"Could not acquire lock (blocking): {e}")
            return False

    def release(self) -> None:
        """Release file lock."""
        if self.lock_fd is not None:
            try:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
                self.lock_fd = None
                self.logger.debug(f"Released lock: {self.lock_file}")
            except (IOError, OSError) as e:
                self.logger.error(f"Could not release lock: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self.acquire_blocking()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class SecureFileOperations:
    """Secure file operations with TOCTOU mitigation."""

    def __init__(self, lock_timeout: float = 10.0):
        """
        Initialize secure file operations.

        Args:
            lock_timeout: Timeout for lock acquisition
        """
        self.lock_timeout = lock_timeout
        self.logger = logging.getLogger("secure_file_ops")

    def safe_read(self, file_path: str) -> Optional[str]:
        """
        Safely read file with TOCTOU mitigation.

        Args:
            file_path: Path to file to read

        Returns:
            File content, or None if operation failed
        """
        path = Path(file_path)
        lock_file = path.parent / f".{path.name}.lock"

        lock = FileLock(str(lock_file), self.lock_timeout)

        try:
            if lock.acquire():
                # Double-check file still exists and is accessible
                if path.exists() and os.access(path, os.R_OK):
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    return content
                else:
                    self.logger.warning(
                        f"File not accessible after acquiring lock: {file_path}"
                    )
                    return None
            else:
                self.logger.warning(f"Could not acquire lock for reading: {file_path}")
                return None
        finally:
            lock.release()

    def safe_write(self, file_path: str, content: str) -> bool:
        """
        Safely write to file with TOCTOU mitigation.

        Args:
            file_path: Path to file to write
            content: Content to write

        Returns:
            True if write succeeded, False otherwise
        """
        path = Path(file_path)
        lock_file = path.parent / f".{path.name}.lock"

        lock = FileLock(str(lock_file), self.lock_timeout)

        try:
            if lock.acquire():
                # Write to temporary file first
                temp_file = path.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    f.write(content)

                # Atomic rename
                temp_file.replace(path)

                self.logger.debug(f"Safely wrote to: {file_path}")
                return True
            else:
                self.logger.warning(f"Could not acquire lock for writing: {file_path}")
                return False
        finally:
            lock.release()

    def safe_delete(self, file_path: str) -> bool:
        """
        Safely delete file with TOCTOU mitigation.

        Args:
            file_path: Path to file to delete

        Returns:
            True if delete succeeded, False otherwise
        """
        path = Path(file_path)
        lock_file = path.parent / f".{path.name}.lock"

        lock = FileLock(str(lock_file), self.lock_timeout)

        try:
            if lock.acquire():
                # Double-check file still exists
                if path.exists():
                    path.unlink()
                    self.logger.debug(f"Safely deleted: {file_path}")
                    return True
                else:
                    self.logger.warning(
                        f"File not found after acquiring lock: {file_path}"
                    )
                    return False
            else:
                self.logger.warning(f"Could not acquire lock for deletion: {file_path}")
                return False
        finally:
            lock.release()

    def safe_stat(self, file_path: str) -> Optional[os.stat_result]:
        """
        Safely get file stats with TOCTOU mitigation.

        Args:
            file_path: Path to file

        Returns:
            File stats, or None if operation failed
        """
        path = Path(file_path)
        lock_file = path.parent / f".{path.name}.lock"

        lock = FileLock(str(lock_file), self.lock_timeout)

        try:
            if lock.acquire():
                # Double-check file still exists
                if path.exists():
                    return path.stat()
                else:
                    return None
            else:
                return None
        finally:
            lock.release()

    def safe_exists(self, file_path: str) -> bool:
        """
        Safely check if file exists with TOCTOU mitigation.

        Args:
            file_path: Path to file

        Returns:
            True if file exists, False otherwise
        """
        return self.safe_stat(file_path) is not None

    def safe_isfile(self, file_path: str) -> bool:
        """
        Safely check if path is file with TOCTOU mitigation.

        Args:
            file_path: Path to file

        Returns:
            True if path is a file, False otherwise
        """
        stats = self.safe_stat(file_path)
        if stats is None:
            return False
        return stat.S_ISREG(stats.st_mode)

    def safe_isdir(self, file_path: str) -> bool:
        """
        Safely check if path is directory with TOCTOU mitigation.

        Args:
            file_path: Path to path

        Returns:
            True if path is a directory, False otherwise
        """
        stats = self.safe_stat(file_path)
        if stats is None:
            return False
        return stat.S_ISDIR(stats.st_mode)

    def safe_listdir(self, dir_path: str) -> Optional[list]:
        """
        Safely list directory contents with TOCTOU mitigation.

        Args:
            dir_path: Path to directory

        Returns:
            List of directory entries, or None if operation failed
        """
        path = Path(dir_path)
        lock_file = path / ".dir.lock"

        lock = FileLock(str(lock_file), self.lock_timeout)

        try:
            if lock.acquire():
                if path.exists() and path.is_dir():
                    return os.listdir(dir_path)
                else:
                    return None
            else:
                return None
        finally:
            lock.release()

    def safe_mkdir(self, dir_path: str, mode: int = 0o755) -> bool:
        """
        Safely create directory with TOCTOU mitigation.

        Args:
            dir_path: Path to directory to create
            mode: Directory permissions

        Returns:
            True if creation succeeded, False otherwise
        """
        path = Path(dir_path)
        lock_file = path.parent / f".{path.name}.lock"

        lock = FileLock(str(lock_file), self.lock_timeout)

        try:
            if lock.acquire():
                if not path.exists():
                    path.mkdir(mode=mode)
                    self.logger.debug(f"Safely created directory: {dir_path}")
                    return True
                else:
                    self.logger.warning(f"Directory already exists: {dir_path}")
                    return True  # Already exists is OK
            else:
                return False
        finally:
            lock.release()

    def safe_rmtree(self, dir_path: str) -> bool:
        """
        Safely remove directory tree with TOCTOU mitigation.

        Args:
            dir_path: Path to directory to remove

        Returns:
            True if removal succeeded, False otherwise
        """
        path = Path(dir_path)
        lock_file = path.parent / f".{path.name}.lock"

        lock = FileLock(str(lock_file), self.lock_timeout)

        try:
            if lock.acquire():
                if path.exists() and path.is_dir():
                    import shutil

                    shutil.rmtree(dir_path)
                    self.logger.debug(f"Safely removed directory: {dir_path}")
                    return True
                else:
                    return False
            else:
                return False
        finally:
            lock.release()

    def safe_rename(self, src_path: str, dst_path: str) -> bool:
        """
        Safely rename file with TOCTOU mitigation.

        Args:
            src_path: Source path
            dst_path: Destination path

        Returns:
            True if rename succeeded, False otherwise
        """
        src = Path(src_path)
        dst = Path(dst_path)
        src_lock = src.parent / f".{src.name}.lock"
        dst_lock = dst.parent / f".{dst.name}.lock"

        # Acquire both locks in consistent order
        lock1 = FileLock(str(src_lock), self.lock_timeout)
        lock2 = FileLock(str(dst_lock), self.lock_timeout)

        try:
            if lock1.acquire() and lock2.acquire():
                # Double-check both paths
                if src.exists() and not dst.exists():
                    src.rename(dst)
                    self.logger.debug(f"Safely renamed: {src_path} -> {dst_path}")
                    return True
                else:
                    self.logger.warning(
                        f"Rename preconditions failed: {src_path} -> {dst_path}"
                    )
                    return False
            else:
                return False
        finally:
            lock2.release()
            lock1.release()

    def concurrent_safe_operation(
        self, file_path: str, operation: str, **kwargs
    ) -> Optional:
        """
        Perform concurrent-safe file operation.

        Args:
            file_path: Path to file
            operation: Type of operation (read, write, delete, stat, exists, isfile, isdir)
            **kwargs: Additional arguments for the operation

        Returns:
            Operation result, or None if operation failed
        """
        operations = {
            "read": self.safe_read,
            "write": self.safe_write,
            "delete": self.safe_delete,
            "stat": self.safe_stat,
            "exists": self.safe_exists,
            "isfile": self.safe_isfile,
            "isdir": self.safe_isdir,
        }

        if operation not in operations:
            raise ValueError(f"Unknown operation: {operation}")

        return operations[operation](file_path, **kwargs)


# Global instance
_secure_file_ops: Optional[SecureFileOperations] = None


def get_secure_file_operations() -> SecureFileOperations:
    """Get or create global secure file operations instance."""
    global _secure_file_ops
    if _secure_file_ops is None:
        _secure_file_ops = SecureFileOperations()
    return _secure_file_ops


def safe_read(file_path: str) -> Optional[str]:
    """Safely read file with TOCTOU mitigation."""
    ops = get_secure_file_operations()
    return ops.safe_read(file_path)


def safe_write(file_path: str, content: str) -> bool:
    """Safely write to file with TOCTOU mitigation."""
    ops = get_secure_file_operations()
    return ops.safe_write(file_path, content)


def safe_delete(file_path: str) -> bool:
    """Safely delete file with TOCTOU mitigation."""
    ops = get_secure_file_operations()
    return ops.safe_delete(file_path)


def safe_exists(file_path: str) -> bool:
    """Safely check if file exists with TOCTOU mitigation."""
    ops = get_secure_file_operations()
    return ops.safe_exists(file_path)


def safe_isfile(file_path: str) -> bool:
    """Safely check if path is file with TOCTOU mitigation."""
    ops = get_secure_file_operations()
    return ops.safe_isfile(file_path)


def safe_isdir(file_path: str) -> bool:
    """Safely check if path is directory with TOCTOU mitigation."""
    ops = get_secure_file_operations()
    return ops.safe_isdir(file_path)
