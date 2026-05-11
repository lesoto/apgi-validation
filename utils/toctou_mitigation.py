"""
TOCTOU (Time-of-Check-Time-of-Use) mitigation utilities.
Provides secure file operations with locking to prevent race conditions.
====================================================================
"""

import os
import shutil
import tempfile
import threading
import time
from typing import List, Optional


class FileLock:
    """File-based lock for preventing concurrent access."""

    def __init__(self, lock_file: str, timeout: Optional[float] = None):
        self.lock_file = lock_file
        self.timeout = timeout
        self.lock_fd: Optional[int] = None
        self._locked = False
        self._lock = threading.Lock()  # Internal lock for thread safety

    def acquire(self) -> bool:
        """Acquire the lock."""
        with self._lock:
            if self._locked:
                return False

            try:
                # Try to create lock file exclusively (atomic operation)
                self.lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                self._locked = True

                # Set secure permissions on lock file
                os.chmod(self.lock_file, 0o600)

                # Write process ID to lock file for debugging
                pid = str(os.getpid())
                os.write(self.lock_fd, pid.encode())
                os.fsync(self.lock_fd)

                print(f"DEBUG: Process {pid} acquired lock {self.lock_file}")

                return True
            except (IOError, OSError) as e:
                # Lock file already exists or creation failed
                print(f"DEBUG: Process {os.getpid()} failed to acquire lock {self.lock_file}: {e}")

                if self.lock_fd is not None:
                    try:
                        os.close(self.lock_fd)
                    except OSError:
                        pass
                    self.lock_fd = None
                return False

    def acquire_blocking(self) -> bool:
        """Acquire the lock with blocking."""
        start_time = time.time()

        while True:
            if self.acquire():
                return True

            if self.timeout and (time.time() - start_time) > self.timeout:
                return False

            time.sleep(0.01)  # Small delay before retrying

    def release(self):
        """Release the lock."""
        with self._lock:
            if self.lock_fd is not None:
                try:
                    os.close(self.lock_fd)
                except OSError:
                    pass
                self.lock_fd = None

            if self._locked:
                try:
                    os.unlink(self.lock_file)
                except OSError:
                    pass
                self._locked = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class SecureFileOperations:
    """Secure file operations with TOCTOU mitigation."""

    def __init__(self):
        self.locks = {}

    def _get_lock(self, file_path: str) -> FileLock:
        """Get or create a lock for the file."""
        if file_path not in self.locks:
            lock_path = f"{file_path}.lock"
            self.locks[file_path] = FileLock(lock_path)
        return self.locks[file_path]

    def safe_read(self, file_path: str) -> Optional[str]:
        """Safely read a file."""
        try:
            with open(file_path, "r") as f:
                return f.read()
        except (IOError, OSError):
            return None

    def safe_write(self, file_path: str, content: str) -> bool:
        """Safely write to a file using atomic operation."""
        lock = self._get_lock(file_path)

        try:
            with lock:
                # Write to temporary file first
                temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(file_path), prefix=".tmp_")

                try:
                    with os.fdopen(temp_fd, "w") as temp_f:
                        temp_f.write(content)

                    # Atomic rename
                    os.rename(temp_path, file_path)
                    return True

                except (IOError, OSError):
                    # Clean up temp file if something went wrong
                    try:
                        os.unlink(temp_path)
                    except OSError:
                        pass
                    return False

        except (IOError, OSError):
            return False

    def safe_delete(self, file_path: str) -> bool:
        """Safely delete a file."""
        lock = self._get_lock(file_path)

        try:
            with lock:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    return True
                return False

        except (IOError, OSError):
            return False

    def safe_stat(self, file_path: str) -> Optional[os.stat_result]:
        """Safely get file stats."""
        try:
            return os.stat(file_path)
        except (IOError, OSError):
            return None

    def safe_exists(self, file_path: str) -> bool:
        """Safely check if file exists."""
        return os.path.exists(file_path)

    def safe_isfile(self, file_path: str) -> bool:
        """Safely check if path is a file."""
        return os.path.isfile(file_path)

    def safe_isdir(self, file_path: str) -> bool:
        """Safely check if path is a directory."""
        return os.path.isdir(file_path)

    def safe_listdir(self, file_path: str) -> Optional[List[str]]:
        """Safely list directory contents."""
        try:
            return os.listdir(file_path)
        except (IOError, OSError):
            return None

    def safe_mkdir(self, file_path: str) -> bool:
        """Safely create directory."""
        try:
            os.makedirs(file_path, exist_ok=True)
            return True
        except (IOError, OSError):
            return False

    def safe_rmtree(self, file_path: str) -> bool:
        """Safely remove directory tree."""
        try:
            if os.path.exists(file_path):
                shutil.rmtree(file_path)
            return True
        except (IOError, OSError):
            return False

    def safe_rename(self, src: str, dst: str) -> bool:
        """Safely rename file/directory."""
        try:
            os.rename(src, dst)
            return True
        except (IOError, OSError):
            return False

    def concurrent_safe_operation(self, file_path: str, operation: str, **kwargs):
        """Perform a concurrent-safe operation on a file."""
        lock = self._get_lock(file_path)

        with lock:
            if operation == "write" and "content" in kwargs:
                return self.safe_write(file_path, kwargs["content"])
            elif operation == "read":
                return self.safe_read(file_path)
            elif operation == "delete":
                return self.safe_delete(file_path)
            else:
                raise ValueError(f"Unsupported operation: {operation}")


# Global instance for convenience functions
_secure_file_ops = None


def get_secure_file_operations() -> SecureFileOperations:
    """Get the global secure file operations instance."""
    global _secure_file_ops
    if _secure_file_ops is None:
        _secure_file_ops = SecureFileOperations()
    return _secure_file_ops


# Convenience functions
def safe_read(file_path: str) -> Optional[str]:
    """Convenience function for safe file read."""
    return get_secure_file_operations().safe_read(file_path)


def safe_write(file_path: str, content: str) -> bool:
    """Convenience function for safe file write."""
    return get_secure_file_operations().safe_write(file_path, content)


def safe_delete(file_path: str) -> bool:
    """Convenience function for safe file delete."""
    return get_secure_file_operations().safe_delete(file_path)


def safe_exists(file_path: str) -> bool:
    """Convenience function for safe existence check."""
    return get_secure_file_operations().safe_exists(file_path)


def safe_isfile(file_path: str) -> bool:
    """Convenience function for safe file check."""
    return get_secure_file_operations().safe_isfile(file_path)


def safe_isdir(file_path: str) -> bool:
    """Convenience function for safe directory check."""
    return get_secure_file_operations().safe_isdir(file_path)
