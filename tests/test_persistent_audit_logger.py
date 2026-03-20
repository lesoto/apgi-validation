"""
Tests for persistent audit logger implementation.
Tests audit log persistence to disk functionality.
===================================================
"""

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.persistent_audit_logger import (
    PersistentAuditLogger,
    get_persistent_audit_logger,
    log_read_persistent,
    log_write_persistent,
    log_delete_persistent,
    log_import_persistent,
)


class TestPersistentAuditLogger:
    """Tests for PersistentAuditLogger class."""

    def test_initialization(self, temp_dir):
        """Test audit logger initialization."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        assert logger.log_file == Path(log_file)
        assert logger.max_file_size == 10 * 1024 * 1024
        assert logger.backup_count == 5
        assert len(logger.audit_trail) == 0

    def test_log_operation_success(self, temp_dir):
        """Test logging successful operation."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        logger.log_operation("read", {"file_path": "test.txt"}, success=True)

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "read"
        assert logger.audit_trail[0]["success"] is True
        assert logger.stats["total_operations"] == 1
        assert logger.stats["by_status"]["success"] == 1

    def test_log_operation_failure(self, temp_dir):
        """Test logging failed operation."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        logger.log_operation(
            "write", {"file_path": "test.txt"}, success=False, error="Permission denied"
        )

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["success"] is False
        assert logger.audit_trail[0]["error"] == "Permission denied"
        assert logger.stats["by_status"]["failure"] == 1

    def test_log_file_persistence(self, temp_dir):
        """Test that logs are persisted to file."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        logger.log_operation("read", {"file_path": "test.txt"})

        # Wait for file write
        time.sleep(0.1)

        assert log_file.exists()
        with open(log_file) as f:
            content = f.read()
        assert "read" in content
        assert "test.txt" in content

    def test_multiple_operations(self, temp_dir):
        """Test logging multiple operations."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        operations = ["read", "write", "delete", "import"]
        for op in operations:
            logger.log_operation(op, {"file": f"{op}.txt"})

        assert len(logger.audit_trail) == 4
        assert logger.stats["total_operations"] == 4
        assert logger.stats["by_operation"]["read"] == 1
        assert logger.stats["by_operation"]["write"] == 1

    def test_get_recent_operations(self, temp_dir):
        """Test retrieving recent operations."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        for i in range(10):
            logger.log_operation("read", {"file": f"file{i}.txt"})

        recent = logger.get_recent_operations(limit=5)
        assert len(recent) == 5

    def test_get_recent_operations_filtered(self, temp_dir):
        """Test retrieving recent operations with filter."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        for i in range(5):
            logger.log_operation("read", {"file": f"file{i}.txt"})
        for i in range(5):
            logger.log_operation("write", {"file": f"file{i}.txt"})

        read_ops = logger.get_recent_operations(operation="read")
        assert len(read_ops) == 5
        assert all(op["operation"] == "read" for op in read_ops)

    def test_audit_trail_size_limit(self, temp_dir):
        """Test that audit trail size is limited."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file), max_trail_size=5)

        for i in range(10):
            logger.log_operation("read", {"file": f"file{i}.txt"})

        assert len(logger.audit_trail) == 5
        # Should keep most recent 5
        assert logger.audit_trail[-1]["details"]["file"] == "file9.txt"

    def test_get_statistics(self, temp_dir):
        """Test getting audit statistics."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        logger.log_operation("read", {"file": "test.txt"}, success=True)
        logger.log_operation("write", {"file": "test.txt"}, success=False)
        logger.log_operation("read", {"file": "test2.txt"}, success=True)

        stats = logger.get_statistics()
        assert stats["total_operations"] == 3
        assert stats["by_operation"]["read"] == 2
        assert stats["by_operation"]["write"] == 1
        assert stats["by_status"]["success"] == 2
        assert stats["by_status"]["failure"] == 1

    def test_export_audit_trail(self, temp_dir):
        """Test exporting audit trail to file."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        logger.log_operation("read", {"file": "test.txt"})
        logger.log_operation("write", {"file": "test.txt"})

        export_file = temp_dir / "export.json"
        logger.export_audit_trail(str(export_file))

        assert export_file.exists()
        with open(export_file) as f:
            data = json.load(f)
        assert "operations" in data
        assert "statistics" in data
        assert len(data["operations"]) == 2

    def test_clear_audit_trail(self, temp_dir):
        """Test clearing audit trail."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        logger.log_operation("read", {"file": "test.txt"})
        logger.log_operation("write", {"file": "test.txt"})

        assert len(logger.audit_trail) == 2

        logger.clear_audit_trail()

        assert len(logger.audit_trail) == 0
        # File should still exist
        assert log_file.exists()

    def test_search_audit_trail(self, temp_dir):
        """Test searching audit trail."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        logger.log_operation("read", {"file": "important.txt"})
        logger.log_operation("write", {"file": "test.txt"})
        logger.log_operation("read", {"file": "data.txt"})

        results = logger.search_audit_trail("important")
        assert len(results) == 1
        assert results[0]["details"]["file"] == "important.txt"

    def test_concurrent_logging(self, temp_dir):
        """Test concurrent logging operations."""
        import threading

        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        def log_operations(thread_id):
            for i in range(100):
                logger.log_operation("read", {"file": f"thread{thread_id}_file{i}.txt"})

        threads = [threading.Thread(target=log_operations, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert logger.stats["total_operations"] == 500

    def test_log_rotation(self, temp_dir):
        """Test log file rotation."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file), max_file_size=1000)

        # Write enough to trigger rotation
        for i in range(100):
            logger.log_operation("read", {"file": f"file{i}.txt", "data": "x" * 100})

        # Check that backup was created
        backup1 = temp_dir / "audit.log.1"
        assert backup1.exists()
        assert logger.stats["last_rotation"] is not None


class TestConvenienceFunctions:
    """Tests for convenience logging functions."""

    def test_log_read_persistent(self, temp_dir):
        """Test log_read_persistent function."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        log_read_persistent("test.txt", success=True)

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "read"

    def test_log_write_persistent(self, temp_dir):
        """Test log_write_persistent function."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        log_write_persistent("test.txt", success=False, error="Disk full")

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "write"
        assert logger.audit_trail[0]["error"] == "Disk full"

    def test_log_delete_persistent(self, temp_dir):
        """Test log_delete_persistent function."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        log_delete_persistent("test.txt")

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "delete"

    def test_log_import_persistent(self, temp_dir):
        """Test log_import_persistent function."""
        log_file = temp_dir / "audit.log"
        logger = PersistentAuditLogger(str(log_file))

        log_import_persistent("numpy")

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "import"

    def test_global_logger_instance(self, temp_dir):
        """Test global logger instance."""
        log_file = temp_dir / "audit.log"

        # Set custom log file for global instance
        import utils.persistent_audit_logger as pal_module

        pal_module._persistent_audit_logger = PersistentAuditLogger(str(log_file))

        logger = get_persistent_audit_logger()

        assert isinstance(logger, PersistentAuditLogger)
        assert logger.log_file == Path(log_file)
