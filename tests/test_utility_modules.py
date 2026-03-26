"""
Comprehensive tests for previously untested utility modules.
Tests for dependency_scanner.py, security_audit_logger.py, and security_logging_integration.py.
===================================================================================
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, Mock
from datetime import datetime
import pytest
import logging
import threading
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dependency_scanner import DependencyScanner
from utils.security_audit_logger import (
    SecurityAuditLogger,
    log_read,
    log_write,
    log_delete,
    log_import,
)
from utils.security_logging_integration import secure_file_read, secure_file_write


class TestDependencyScanner:
    """Comprehensive tests for DependencyScanner."""

    def test_scanner_initialization(self, temp_dir):
        """Test scanner initialization with project root."""
        scanner = DependencyScanner(project_root=str(temp_dir))
        assert scanner.project_root == temp_dir
        assert scanner.requirements_file == temp_dir / "requirements.txt"

    def test_scanner_with_default_project_root(self):
        """Test scanner initialization with default project root."""
        scanner = DependencyScanner()
        assert scanner.project_root == Path.cwd()

    def test_scan_with_pip_audit_success(self, temp_dir):
        """Test successful pip-audit scan."""
        # Create a requirements.txt file
        requirements_file = temp_dir / "requirements.txt"
        requirements_file.write_text("numpy==1.21.0\npandas==1.3.0\n")

        scanner = DependencyScanner(project_root=str(temp_dir))

        # Mock subprocess.run to return successful scan
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps([])  # No vulnerabilities
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = scanner.scan_with_pip_audit()
            assert result["success"] is True
            assert "vulnerabilities" in result

    def test_scan_with_pip_audit_vulnerabilities(self, temp_dir):
        """Test pip-audit scan with vulnerabilities found."""
        requirements_file = temp_dir / "requirements.txt"
        requirements_file.write_text("numpy==1.16.0\n")  # Known vulnerable version

        scanner = DependencyScanner(project_root=str(temp_dir))

        # Mock subprocess.run to return vulnerabilities
        mock_result = Mock()
        mock_result.returncode = 1  # Non-zero return code for vulnerabilities
        mock_result.stdout = json.dumps(
            [
                {
                    "name": "numpy",
                    "version": "1.16.0",
                    "vulnerabilities": [
                        {
                            "id": "CVE-2021-1234",
                            "severity": "high",
                            "description": "Test vulnerability",
                        }
                    ],
                }
            ]
        )
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = scanner.scan_with_pip_audit()
            assert result["success"] is False
            assert len(result["vulnerabilities"]) > 0

    def test_scan_with_missing_requirements_file(self, temp_dir):
        """Test scan with missing requirements.txt file."""
        scanner = DependencyScanner(project_root=str(temp_dir))

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("requirements.txt not found")
            result = scanner.scan_with_pip_audit()
            assert result["success"] is False
            assert "error" in result

    def test_scan_with_timeout(self, temp_dir):
        """Test scan with timeout handling."""
        requirements_file = temp_dir / "requirements.txt"
        requirements_file.write_text("numpy==1.21.0\n")

        scanner = DependencyScanner(project_root=str(temp_dir))

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = TimeoutError("Scan timed out")
            result = scanner.scan_with_pip_audit()
            assert result["success"] is False
            assert (
                "timeout" in result["error"].lower()
                or "timed out" in result["error"].lower()
            )

    def test_scan_with_invalid_json_response(self, temp_dir):
        """Test scan with invalid JSON response."""
        requirements_file = temp_dir / "requirements.txt"
        requirements_file.write_text("numpy==1.21.0\n")

        scanner = DependencyScanner(project_root=str(temp_dir))

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "invalid json"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = scanner.scan_with_pip_audit()
            assert result["success"] is False
            assert "json" in result["error"].lower()

    def test_scan_with_subprocess_error(self, temp_dir):
        """Test scan with subprocess execution error."""
        requirements_file = temp_dir / "requirements.txt"
        requirements_file.write_text("numpy==1.21.0\n")

        scanner = DependencyScanner(project_root=str(temp_dir))

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("pip-audit not found")
            result = scanner.scan_with_pip_audit()
            assert result["success"] is False
            assert "error" in result


class TestSecurityAuditLogger:
    """Comprehensive tests for SecurityAuditLogger."""

    def test_logger_initialization(self, temp_dir):
        """Test logger initialization."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))
        assert logger.log_file == log_file
        assert isinstance(logger.logger, logging.Logger)
        assert isinstance(logger.audit_trail, list)

    def test_log_read_operation(self, temp_dir):
        """Test logging read operations."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        logger.log_read("/path/to/file.txt", success=True)
        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "read"
        assert logger.audit_trail[0]["success"] is True

    def test_log_write_operation(self, temp_dir):
        """Test logging write operations."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        logger.log_write("/path/to/file.txt", success=True, size_bytes=1024)
        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "write"
        assert logger.audit_trail[0]["size_bytes"] == 1024

    def test_log_delete_operation(self, temp_dir):
        """Test logging delete operations."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        logger.log_delete("/path/to/file.txt", success=True)
        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "delete"

    def test_log_import_operation(self, temp_dir):
        """Test logging import operations."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        logger.log_import("some_module", success=True)
        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["operation"] == "import"
        assert logger.audit_trail[0]["module"] == "some_module"

    def test_audit_trail_limit(self, temp_dir):
        """Test audit trail limit (1000 entries)."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        # Add more than 1000 entries
        for i in range(1100):
            logger.log_read(f"/path/to/file_{i}.txt", success=True)

        # Should be limited to 1000 entries
        assert len(logger.audit_trail) <= 1000

    def test_get_recent_operations(self, temp_dir):
        """Test retrieving recent operations."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        for i in range(10):
            logger.log_read(f"/path/to/file_{i}.txt", success=True)

        recent = logger.get_recent_operations(limit=5)
        assert len(recent) == 5
        assert recent[0]["operation"] == "read"

    def test_get_operations_by_type(self, temp_dir):
        """Test filtering operations by type."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        logger.log_read("/path/to/file1.txt", success=True)
        logger.log_write("/path/to/file2.txt", success=True)
        logger.log_read("/path/to/file3.txt", success=True)

        read_ops = logger.get_operations_by_type("read")
        assert len(read_ops) == 2
        assert all(op["operation"] == "read" for op in read_ops)

    def test_log_persistence_to_file(self, temp_dir):
        """Test that logs are written to file."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        logger.log_read("/path/to/file.txt", success=True)

        # Force flush
        for handler in logger.logger.handlers:
            handler.flush()

        # Check file exists and has content
        assert log_file.exists()
        content = log_file.read_text()
        assert "read" in content
        assert "/path/to/file.txt" in content

    def test_log_with_error_details(self, temp_dir):
        """Test logging with error details."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        logger.log_read("/path/to/file.txt", success=False, error="Permission denied")
        assert len(logger.audit_trail) == 1
        assert logger.audit_trail[0]["success"] is False
        assert logger.audit_trail[0]["error"] == "Permission denied"

    def test_get_audit_statistics(self, temp_dir):
        """Test getting audit statistics."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        logger.log_read("/path/to/file1.txt", success=True)
        logger.log_read("/path/to/file2.txt", success=False)
        logger.log_write("/path/to/file3.txt", success=True)

        stats = logger.get_audit_statistics()
        assert stats["total_operations"] == 3
        assert stats["successful_operations"] == 2
        assert stats["failed_operations"] == 1


class TestSecurityLoggingIntegration:
    """Comprehensive tests for security logging integration functions."""

    def test_secure_file_read_success(self, temp_dir):
        """Test secure file read with success."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        content = secure_file_read(str(test_file))
        assert content == "Test content"

    def test_secure_file_read_failure(self, temp_dir):
        """Test secure file read with failure."""
        non_existent_file = temp_dir / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            secure_file_read(str(non_existent_file))

    def test_secure_file_write_success(self, temp_dir):
        """Test secure file write with success."""
        test_file = temp_dir / "test.txt"
        secure_file_write(str(test_file), "Test content")

        assert test_file.exists()
        assert test_file.read_text() == "Test content"

    def test_secure_file_write_failure(self, temp_dir):
        """Test secure file write with permission error."""
        # Create a directory instead of a file
        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()

        with pytest.raises((IsADirectoryError, PermissionError)):
            secure_file_write(str(test_dir), "Test content")

    def test_log_read_function(self, temp_dir):
        """Test log_read convenience function."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        log_read("/path/to/file.txt", success=True)
        assert len(logger.audit_trail) == 1

    def test_log_write_function(self, temp_dir):
        """Test log_write convenience function."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        log_write("/path/to/file.txt", success=True)
        assert len(logger.audit_trail) == 1

    def test_log_delete_function(self, temp_dir):
        """Test log_delete convenience function."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        log_delete("/path/to/file.txt", success=True)
        assert len(logger.audit_trail) == 1

    def test_log_import_function(self, temp_dir):
        """Test log_import convenience function."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        log_import("test_module", success=True)
        assert len(logger.audit_trail) == 1

    def test_concurrent_logging(self, temp_dir):
        """Test concurrent logging operations."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        def log_operations(thread_id):
            for i in range(100):
                logger.log_read(f"/path/file_{thread_id}_{i}.txt", success=True)

        threads = [threading.Thread(target=log_operations, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should have 500 operations logged
        assert len(logger.audit_trail) == 500

    def test_log_rotation(self, temp_dir):
        """Test log file rotation."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        # Add many operations to trigger rotation check
        for i in range(100):
            logger.log_read(f"/path/file_{i}.txt", success=True)

        # Check that log file exists and has content
        assert log_file.exists()
        content = log_file.read_text()
        assert len(content) > 0

    def test_audit_trail_thread_safety(self, temp_dir):
        """Test audit trail thread safety."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        def concurrent_reads():
            for i in range(50):
                logger.log_read(f"/path/file_{i}.txt", success=True)

        threads = [threading.Thread(target=concurrent_reads) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify all operations were logged
        assert len(logger.audit_trail) == 500

    def test_get_operations_by_time_range(self, temp_dir):
        """Test filtering operations by time range."""
        log_file = temp_dir / "test_audit.log"
        logger = SecurityAuditLogger(log_file=str(log_file))

        # Log operations at different times
        logger.log_read("/path/file1.txt", success=True)
        time.sleep(0.1)
        logger.log_read("/path/file2.txt", success=True)

        recent_ops = logger.get_operations_by_time_range(
            start_time=datetime.now().timestamp() - 1,
            end_time=datetime.now().timestamp(),
        )
        assert len(recent_ops) >= 2
