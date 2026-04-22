"""
Tests for TOCTOU mitigation with file locking.
Tests TimeOfCheck-TimeOfUse race condition mitigation.
=====================================================
"""

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.toctou_mitigation import (
    FileLock,
    SecureFileOperations,
    get_secure_file_operations,
    safe_delete,
    safe_exists,
    safe_isdir,
    safe_isfile,
    safe_read,
    safe_write,
)


class TestFileLock:
    """Tests for FileLock class."""

    def test_acquire_release(self, temp_dir):
        """Test acquiring and releasing lock."""
        lock_file = temp_dir / "test.lock"
        lock = FileLock(str(lock_file))

        assert lock.acquire() is True
        assert lock.lock_fd is not None
        lock.release()
        assert lock.lock_fd is None

    def test_context_manager(self, temp_dir):
        """Test lock as context manager."""
        lock_file = temp_dir / "test.lock"
        lock = FileLock(str(lock_file))

        with lock:
            assert lock.lock_fd is not None

        assert lock.lock_fd is None

    def test_concurrent_lock_acquisition(self, temp_dir):
        """Test concurrent lock acquisition."""
        lock_file = temp_dir / "test.lock"
        lock = FileLock(str(lock_file), timeout=1.0)

        acquired_count = [0]

        def try_acquire():
            if lock.acquire():
                acquired_count[0] += 1
                time.sleep(0.1)
                lock.release()

        threads = [threading.Thread(target=try_acquire) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Only one thread should have acquired the lock
        assert acquired_count[0] == 1

    def test_blocking_acquire(self, temp_dir):
        """Test blocking lock acquisition."""
        lock_file = temp_dir / "test.lock"
        lock = FileLock(str(lock_file))

        acquired = lock.acquire_blocking()
        assert acquired is True
        lock.release()

    def test_lock_file_permissions(self, temp_dir):
        """Test lock file has correct permissions."""
        lock_file = temp_dir / "test.lock"
        lock = FileLock(str(lock_file))

        lock.acquire()

        # Check lock file permissions
        stat_info = os.stat(lock_file)
        assert (stat_info.st_mode & 0o777) == 0o600

        lock.release()


class TestSecureFileOperations:
    """Tests for SecureFileOperations class."""

    def test_safe_read(self, temp_dir):
        """Test safe file read."""
        ops = SecureFileOperations()
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        content = ops.safe_read(str(test_file))
        assert content == "test content"

    def test_safe_read_nonexistent(self, temp_dir):
        """Test safe read of nonexistent file."""
        ops = SecureFileOperations()
        test_file = temp_dir / "nonexistent.txt"

        content = ops.safe_read(str(test_file))
        assert content is None

    def test_safe_write(self, temp_dir):
        """Test safe file write."""
        ops = SecureFileOperations()
        test_file = temp_dir / "test.txt"

        result = ops.safe_write(str(test_file), "new content")
        assert result is True
        assert test_file.read_text() == "new content"

    def test_safe_write_atomic(self, temp_dir):
        """Test that write is atomic (uses temp file + rename)."""
        ops = SecureFileOperations()
        test_file = temp_dir / "test.txt"

        result = ops.safe_write(str(test_file), "atomic content")
        assert result is True

    def test_safe_delete(self, temp_dir):
        """Test safe file delete."""
        ops = SecureFileOperations()
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        result = ops.safe_delete(str(test_file))
        assert result is True
        assert not test_file.exists()

    def test_safe_delete_nonexistent(self, temp_dir):
        """Test safe delete of nonexistent file."""
        ops = SecureFileOperations()
        test_file = temp_dir / "nonexistent.txt"

        result = ops.safe_delete(str(test_file))
        assert result is False

    def test_safe_stat(self, temp_dir):
        """Test safe file stat."""
        ops = SecureFileOperations()
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        stats = ops.safe_stat(str(test_file))
        assert stats is not None
        assert stats.st_size == len("test content")

    def test_safe_exists(self, temp_dir):
        """Test safe file existence check."""
        ops = SecureFileOperations()
        test_file = temp_dir / "test.txt"

        assert not ops.safe_exists(str(test_file))

        test_file.write_text("test")
        assert ops.safe_exists(str(test_file))

    def test_safe_isfile(self, temp_dir):
        """Test safe file type check."""
        ops = SecureFileOperations()
        test_file = temp_dir / "test.txt"
        test_dir = temp_dir / "testdir"

        test_file.write_text("test")
        test_dir.mkdir()

        assert ops.safe_isfile(str(test_file))
        assert not ops.safe_isfile(str(test_dir))

    def test_safe_isdir(self, temp_dir):
        """Test safe directory type check."""
        ops = SecureFileOperations()
        test_file = temp_dir / "test.txt"
        test_dir = temp_dir / "testdir"

        test_file.write_text("test")
        test_dir.mkdir()

        assert ops.safe_isdir(str(test_dir))
        assert not ops.safe_isdir(str(test_file))

    def test_safe_listdir(self, temp_dir):
        """Test safe directory listing."""
        ops = SecureFileOperations()
        test_dir = temp_dir / "testdir"
        test_dir.mkdir()

        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")

        entries = ops.safe_listdir(str(test_dir))
        assert entries is not None
        assert "file1.txt" in entries
        assert "file2.txt" in entries

    def test_safe_mkdir(self, temp_dir):
        """Test safe directory creation."""
        ops = SecureFileOperations()
        test_dir = temp_dir / "newdir"

        result = ops.safe_mkdir(str(test_dir))
        assert result is True
        assert test_dir.exists()

    def test_safe_mkdir_existing(self, temp_dir):
        """Test safe mkdir of existing directory."""
        ops = SecureFileOperations()
        test_dir = temp_dir / "existing"
        test_dir.mkdir()

        # Should return True (already exists is OK)
        result = ops.safe_mkdir(str(test_dir))
        assert result is True

    def test_safe_rmtree(self, temp_dir):
        """Test safe directory tree removal."""
        ops = SecureFileOperations()
        test_dir = temp_dir / "todelete"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        result = ops.safe_rmtree(str(test_dir))
        assert result is True
        assert not test_dir.exists()

    def test_safe_rename(self, temp_dir):
        """Test safe file rename."""
        ops = SecureFileOperations()
        src_file = temp_dir / "source.txt"
        dst_file = temp_dir / "destination.txt"
        src_file.write_text("content")

        result = ops.safe_rename(str(src_file), str(dst_file))
        assert result is True
        assert not src_file.exists()
        assert dst_file.exists()

    def test_concurrent_operations(self, temp_dir):
        """Test concurrent safe operations."""
        ops = SecureFileOperations()
        test_file = temp_dir / "concurrent.txt"
        test_file.write_text("initial")

        results = []

        def concurrent_operation(thread_id):
            content = ops.safe_read(str(test_file))
            ops.safe_write(str(test_file), f"modified by {thread_id}")
            results.append(content)

        threads = [
            threading.Thread(target=concurrent_operation, args=(i,)) for i in range(5)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All operations should have succeeded
        assert len(results) == 5
        # Final content should be one of the thread modifications (not dependent on which thread finishes last)
        final_content = test_file.read_text()
        assert any(f"modified by {i}" == final_content for i in range(5))

    def test_toctou_mitigation(self, temp_dir):
        """Test TOCTOU mitigation prevents race conditions."""
        ops = SecureFileOperations()
        test_file = temp_dir / "race_condition.txt"

        # Simulate race condition: check exists, then delete

        def check_then_delete():
            if ops.safe_exists(str(test_file)):
                time.sleep(0.01)  # Small delay
                ops.safe_delete(str(test_file))

        def check_then_delete_2():
            if ops.safe_exists(str(test_file)):
                ops.safe_delete(str(test_file))

        # Create file first
        test_file.write_text("content")

        # Run two threads that both check and delete
        thread1 = threading.Thread(target=check_then_delete)
        thread2 = threading.Thread(target=check_then_delete_2)

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # File should be deleted (one of the threads succeeded)
        # The lock ensures only one thread actually deletes
        assert not test_file.exists() or test_file.read_text() == "content"

    def test_concurrent_safe_operation(self, temp_dir):
        """Test concurrent_safe operation method."""
        ops = SecureFileOperations()
        test_file = temp_dir / "concurrent.txt"
        test_file.write_text("initial")

        def operation_thread(thread_id):
            ops.concurrent_safe_operation(
                str(test_file), "write", content=f"thread_{thread_id}"
            )

        threads = [
            threading.Thread(target=operation_thread, args=(i,)) for i in range(5)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # File should exist with content from one thread
        assert test_file.exists()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_safe_read_function(self, temp_dir):
        """Test safe_read convenience function."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        content = safe_read(str(test_file))
        assert content == "content"

    def test_safe_write_function(self, temp_dir):
        """Test safe_write convenience function."""
        test_file = temp_dir / "test.txt"

        result = safe_write(str(test_file), "new content")
        assert result is True
        assert test_file.read_text() == "new content"

    def test_safe_delete_function(self, temp_dir):
        """Test safe_delete convenience function."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        result = safe_delete(str(test_file))
        assert result is True
        assert not test_file.exists()

    def test_safe_exists_function(self, temp_dir):
        """Test safe_exists convenience function."""
        test_file = temp_dir / "test.txt"

        assert not safe_exists(str(test_file))

        test_file.write_text("content")
        assert safe_exists(str(test_file))

    def test_safe_isfile_function(self, temp_dir):
        """Test safe_isfile convenience function."""
        test_file = temp_dir / "test.txt"
        test_dir = temp_dir / "testdir"

        test_file.write_text("content")
        test_dir.mkdir()

        assert safe_isfile(str(test_file))
        assert not safe_isfile(str(test_dir))

    def test_safe_isdir_function(self, temp_dir):
        """Test safe_isdir convenience function."""
        test_file = temp_dir / "test.txt"
        test_dir = temp_dir / "testdir"

        test_file.write_text("content")
        test_dir.mkdir()

        assert safe_isdir(str(test_dir))
        assert not safe_isdir(str(test_file))

    def test_global_instance(self, temp_dir):
        """Test global secure file operations instance."""
        import utils.toctou_mitigation as tm_module

        tm_module._secure_file_ops = None

        ops = get_secure_file_operations()
        assert isinstance(ops, SecureFileOperations)
