"""
Tests for utils/security_audit_logger.py
=========================================
Comprehensive tests for security audit logger.
"""

import pytest
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.security_audit_logger import (
    SecurityAuditLogger,
    get_audit_logger,
    audit_file_operation,
    audit_path_resolution,
    audit_permission_check,
    log_read,
    log_write,
    log_delete,
    log_import,
)


class TestSecurityAuditLoggerInit:
    """Test SecurityAuditLogger initialization."""

    def test_init_default(self, tmp_path):
        """Test initialization with default parameters."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        assert logger.log_file == log_file
        assert logger.log_file.name == "security_audit.log"
        assert logger.audit_trail == []

    def test_init_custom_log_level(self, tmp_path):
        """Test initialization with custom log level."""
        log_file = tmp_path / "security_audit.log"
        import logging

        logger = SecurityAuditLogger(str(log_file), log_level=logging.DEBUG)

        assert logger.log_level == logging.DEBUG


class TestLogFileAccess:
    """Test log_file_access method."""

    def test_log_file_access_success(self, tmp_path):
        """Test logging successful file access."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_file_access("read", "/test/file.txt", user="test_user", success=True)

        assert len(logger.audit_trail) == 1
        entry = logger.audit_trail[0]
        assert entry["operation"] == "read"
        assert entry["file_path"] == "/test/file.txt"
        assert entry["user"] == "test_user"
        assert entry["success"] is True
        assert "timestamp" in entry

    def test_log_file_access_failure(self, tmp_path):
        """Test logging failed file access."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_file_access(
            "write",
            "/test/file.txt",
            user="test_user",
            success=False,
            error="Permission denied",
        )

        assert len(logger.audit_trail) == 1
        entry = logger.audit_trail[0]
        assert entry["operation"] == "write"
        assert entry["success"] is False
        assert entry["error"] == "Permission denied"

    def test_log_file_access_with_context(self, tmp_path):
        """Test logging file access with additional context."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_file_access("read", "/test/file.txt", size=1024, mode="r")

        assert len(logger.audit_trail) == 1
        entry = logger.audit_trail[0]
        assert entry["context"]["size"] == 1024
        assert entry["context"]["mode"] == "r"

    def test_log_file_access_trail_limit(self, tmp_path):
        """Test audit trail limit (1000 entries)."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        # Add 1001 entries
        for i in range(1001):
            logger.log_file_access("read", f"/test/file_{i}.txt")

        assert len(logger.audit_trail) == 1000
        # First entry should be removed
        assert logger.audit_trail[0]["file_path"] == "/test/file_1.txt"


class TestLogPathResolution:
    """Test log_path_resolution method."""

    def test_log_path_resolution_success(self, tmp_path):
        """Test logging successful path resolution."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_path_resolution(
            "../file.txt", "/absolute/path/file.txt", method="resolve"
        )

        assert len(logger.audit_trail) == 1
        entry = logger.audit_trail[0]
        assert entry["operation"] == "path_resolution"
        assert entry["original_path"] == "../file.txt"
        assert entry["resolved_path"] == "/absolute/path/file.txt"
        assert entry["method"] == "resolve"
        assert entry["success"] is True

    def test_log_path_resolution_failure(self, tmp_path):
        """Test logging failed path resolution."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_path_resolution(
            "../file.txt", "unknown", method="resolve", success=False
        )

        assert len(logger.audit_trail) == 1
        entry = logger.audit_trail[0]
        assert entry["success"] is False


class TestLogPermissionCheck:
    """Test log_permission_check method."""

    def test_log_permission_check_granted(self, tmp_path):
        """Test logging granted permission."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_permission_check(
            "/test/file.txt", "read", granted=True, user="test_user"
        )

        assert len(logger.audit_trail) == 1
        entry = logger.audit_trail[0]
        assert entry["operation"] == "permission_check"
        assert entry["file_path"] == "/test/file.txt"
        assert entry["permission_type"] == "read"
        assert entry["granted"] is True
        assert entry["user"] == "test_user"

    def test_log_permission_check_denied(self, tmp_path):
        """Test logging denied permission."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_permission_check(
            "/test/file.txt", "write", granted=False, user="test_user"
        )

        assert len(logger.audit_trail) == 1
        entry = logger.audit_trail[0]
        assert entry["granted"] is False


class TestLogConfigurationChange:
    """Test log_configuration_change method."""

    def test_log_configuration_change(self, tmp_path):
        """Test logging configuration change."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_configuration_change("max_file_size", "100MB", "200MB", user="admin")

        assert len(logger.audit_trail) == 1
        entry = logger.audit_trail[0]
        assert entry["operation"] == "config_change"
        assert entry["config_key"] == "max_file_size"
        assert entry["old_value"] == "100MB"
        assert entry["new_value"] == "200MB"
        assert entry["user"] == "admin"


class TestExportAuditTrail:
    """Test export_audit_trail method."""

    def test_export_audit_trail_json(self, tmp_path):
        """Test exporting audit trail to JSON."""
        log_file = tmp_path / "security_audit.log"
        output_file = tmp_path / "audit_export.json"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_file_access("read", "/test/file1.txt")
        logger.log_file_access("write", "/test/file2.txt")

        logger.export_audit_trail(str(output_file), format="json")

        assert output_file.exists()
        import json

        with open(output_file) as f:
            data = json.load(f)
        assert len(data) == 2

    def test_export_audit_trail_csv(self, tmp_path):
        """Test exporting audit trail to CSV."""
        log_file = tmp_path / "security_audit.log"
        output_file = tmp_path / "audit_export.csv"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_file_access("read", "/test/file1.txt")
        logger.log_file_access("write", "/test/file2.txt")

        logger.export_audit_trail(str(output_file), format="csv")

        assert output_file.exists()
        import csv

        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2


class TestGetRecentOperations:
    """Test get_recent_operations method."""

    def test_get_recent_operations_default(self, tmp_path):
        """Test getting recent operations with default count."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        for i in range(10):
            logger.log_file_access("read", f"/test/file_{i}.txt")

        recent = logger.get_recent_operations()
        assert len(recent) == 100  # Default is 100, but we only have 10

    def test_get_recent_operations_custom_count(self, tmp_path):
        """Test getting recent operations with custom count."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        for i in range(10):
            logger.log_file_access("read", f"/test/file_{i}.txt")

        recent = logger.get_recent_operations(n=5)
        assert len(recent) == 5
        assert recent[0]["file_path"] == "/test/file_5.txt"
        assert recent[-1]["file_path"] == "/test/file_9.txt"


class TestSearchAuditTrail:
    """Test search_audit_trail method."""

    def test_search_by_operation(self, tmp_path):
        """Test searching audit trail by operation."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_file_access("read", "/test/file1.txt")
        logger.log_file_access("write", "/test/file2.txt")
        logger.log_file_access("read", "/test/file3.txt")

        results = logger.search_audit_trail(operation="read")
        assert len(results) == 2

    def test_search_by_file_path(self, tmp_path):
        """Test searching audit trail by file path."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_file_access("read", "/test/file1.txt")
        logger.log_file_access("write", "/test/file2.txt")
        logger.log_file_access("read", "/test/file1.txt")

        results = logger.search_audit_trail(file_path="/test/file1.txt")
        assert len(results) == 2

    def test_search_by_user(self, tmp_path):
        """Test searching audit trail by user."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_file_access("read", "/test/file1.txt", user="user1")
        logger.log_file_access("write", "/test/file2.txt", user="user2")
        logger.log_file_access("read", "/test/file3.txt", user="user1")

        results = logger.search_audit_trail(user="user1")
        assert len(results) == 2

    def test_search_by_time_range(self, tmp_path):
        """Test searching audit trail by time range."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_file_access("read", "/test/file1.txt")

        import time

        time.sleep(0.1)

        start_time = datetime.utcnow()
        logger.log_file_access("write", "/test/file2.txt")

        results = logger.search_audit_trail(start_time=start_time)
        assert len(results) == 1
        assert results[0]["file_path"] == "/test/file2.txt"

    def test_search_combined_filters(self, tmp_path):
        """Test searching with multiple filters."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        logger.log_file_access("read", "/test/file1.txt", user="user1")
        logger.log_file_access("write", "/test/file1.txt", user="user1")
        logger.log_file_access("read", "/test/file2.txt", user="user1")

        results = logger.search_audit_trail(
            operation="read", file_path="/test/file1.txt", user="user1"
        )
        assert len(results) == 1


class TestGetAuditLogger:
    """Test get_audit_logger function."""

    def test_get_audit_logger_singleton(self, tmp_path):
        """Test that get_audit_logger returns singleton instance."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()

        assert logger1 is logger2


class TestAuditFileOperationDecorator:
    """Test audit_file_operation decorator."""

    def test_audit_file_operation_success(self, tmp_path):
        """Test decorator with successful operation."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        @audit_file_operation("test_read", logger=logger)
        def read_file(file_path):
            return "content"

        result = read_file("/test/file.txt")

        assert result == "content"
        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "test_read"
        assert logger.audit_trail[0]["success"] is True

    def test_audit_file_operation_failure(self, tmp_path):
        """Test decorator with failed operation."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        @audit_file_operation("test_read", logger=logger)
        def read_file(file_path):
            raise FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            read_file("/test/file.txt")

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["success"] is False
        assert "File not found" in logger.audit_trail[0]["error"]

    def test_audit_file_operation_with_kwargs(self, tmp_path):
        """Test decorator with keyword arguments."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        @audit_file_operation("test_read", logger=logger)
        def read_file(file_path=None):
            return "content"

        read_file(file_path="/test/file.txt")

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["file_path"] == "/test/file.txt"


class TestAuditPathResolutionDecorator:
    """Test audit_path_resolution decorator."""

    def test_audit_path_resolution_success(self, tmp_path):
        """Test decorator with successful resolution."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        @audit_path_resolution("resolve", logger=logger)
        def resolve_path(path):
            return "/absolute/path"

        result = resolve_path("relative/path")

        assert result == "/absolute/path"
        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "path_resolution"
        assert logger.audit_trail[0]["success"] is True

    def test_audit_path_resolution_failure(self, tmp_path):
        """Test decorator with failed resolution."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        @audit_path_resolution("resolve", logger=logger)
        def resolve_path(path):
            raise ValueError("Invalid path")

        with pytest.raises(ValueError):
            resolve_path("invalid/path")

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["success"] is False


class TestAuditPermissionCheckDecorator:
    """Test audit_permission_check decorator."""

    def test_audit_permission_check_granted(self, tmp_path):
        """Test decorator with granted permission."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        @audit_permission_check("read", logger=logger)
        def check_permission(file_path):
            return True

        result = check_permission("/test/file.txt")

        assert result is True
        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "permission_check"
        assert logger.audit_trail[0]["granted"] is True

    def test_audit_permission_check_denied(self, tmp_path):
        """Test decorator with denied permission."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        @audit_permission_check("write", logger=logger)
        def check_permission(file_path):
            return False

        result = check_permission("/test/file.txt")

        assert result is False
        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["granted"] is False


class TestConvenienceFunctions:
    """Test convenience logging functions."""

    def test_log_read(self, tmp_path):
        """Test log_read convenience function."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        log_read("/test/file.txt", logger=logger)

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "read"

    def test_log_write(self, tmp_path):
        """Test log_write convenience function."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        log_write("/test/file.txt", logger=logger)

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "write"

    def test_log_delete(self, tmp_path):
        """Test log_delete convenience function."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        log_delete("/test/file.txt", logger=logger)

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "delete"

    def test_log_import(self, tmp_path):
        """Test log_import convenience function."""
        log_file = tmp_path / "security_audit.log"
        logger = SecurityAuditLogger(str(log_file))

        log_import("/test/module.py", logger=logger)

        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "import"
