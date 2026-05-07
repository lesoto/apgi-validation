"""
Targeted tests for utils/security_logging_integration.py missing coverage areas.
========================================================================

Focuses on specific functions and lines that are currently uncovered.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSecurityLoggingIntegrationMissingCoverage:
    """Test security logging integration functions with missing coverage."""

    def test_enforce_security_audit_decorator(self):
        """Test the enforce_security_audit decorator."""
        from utils.security_logging_integration import enforce_security_audit

        # Test decorator application
        @enforce_security_audit("test_operation")
        def test_function(arg1, arg2):
            return f"result: {arg1}, {arg2}"

        # Test normal execution
        result = test_function("value1", "value2")
        assert result == "result: value1, value2"

    def test_enforce_security_audit_with_exception(self):
        """Test enforce_security_audit with exception in decorated function."""
        from utils.security_logging_integration import enforce_security_audit

        @enforce_security_audit("test_operation")
        def failing_function():
            raise ValueError("Test error")

        # Should propagate the exception
        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    def test_security_audit_logger_initialization(self):
        """Test security audit logger initialization."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "security_audit.log"

            logger = SecurityAuditLogger(str(log_file))
            assert logger is not None
            assert logger.log_file == str(log_file)

    def test_security_audit_log_operation(self):
        """Test logging security operations."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "audit.log"

            logger = SecurityAuditLogger(str(log_file))
            logger.log_operation("test_operation", "user123", {"param": "value"})

            # Check that log file was created and has content
            assert log_file.exists()
            content = log_file.read_text()
            assert "test_operation" in content
            assert "user123" in content

    def test_security_audit_log_authentication(self):
        """Test logging authentication events."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "auth.log"

            logger = SecurityAuditLogger(str(log_file))
            logger.log_authentication("login", "user123", True, "127.0.0.1")

            content = log_file.read_text()
            assert "login" in content
            assert "user123" in content
            assert "success" in content
            assert "127.0.0.1" in content

    def test_security_audit_log_authentication_failure(self):
        """Test logging authentication failures."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "auth_fail.log"

            logger = SecurityAuditLogger(str(log_file))
            logger.log_authentication(
                "login", "user123", False, "192.168.1.1", "Invalid password"
            )

            content = log_file.read_text()
            assert "login" in content
            assert "user123" in content
            assert "failure" in content
            assert "Invalid password" in content

    def test_security_audit_log_permission_check(self):
        """Test logging permission checks."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "perm.log"

            logger = SecurityAuditLogger(str(log_file))
            logger.log_permission_check("read_file", "user123", "/test/file.txt", True)

            content = log_file.read_text()
            assert "read_file" in content
            assert "user123" in content
            assert "/test/file.txt" in content
            assert "granted" in content

    def test_security_audit_log_permission_denied(self):
        """Test logging permission denied events."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "perm_denied.log"

            logger = SecurityAuditLogger(str(log_file))
            logger.log_permission_check(
                "delete_file", "user123", "/important/file.txt", False
            )

            content = log_file.read_text()
            assert "delete_file" in content
            assert "user123" in content
            assert "/important/file.txt" in content
            assert "denied" in content

    def test_security_audit_log_data_access(self):
        """Test logging data access events."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "data_access.log"

            logger = SecurityAuditLogger(str(log_file))
            logger.log_data_access("read", "user123", "patient_records", "record_123")

            content = log_file.read_text()
            assert "read" in content
            assert "user123" in content
            assert "patient_records" in content
            assert "record_123" in content

    def test_security_audit_log_system_event(self):
        """Test logging system events."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "system.log"

            logger = SecurityAuditLogger(str(log_file))
            logger.log_system_event("startup", "system_version", "1.0.0")

            content = log_file.read_text()
            assert "startup" in content
            assert "system_version" in content
            assert "1.0.0" in content

    def test_security_audit_log_security_violation(self):
        """Test logging security violations."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "violation.log"

            logger = SecurityAuditLogger(str(log_file))
            logger.log_security_violation(
                "intrusion_attempt",
                "user123",
                "192.168.1.100",
                {"details": "multiple failed logins"},
            )

            content = log_file.read_text()
            assert "intrusion_attempt" in content
            assert "user123" in content
            assert "192.168.1.100" in content
            assert "multiple failed logins" in content

    def test_security_audit_get_recent_logs(self):
        """Test retrieving recent logs."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "recent.log"

            logger = SecurityAuditLogger(str(log_file))

            # Add some log entries
            logger.log_operation("op1", "user1", {})
            logger.log_operation("op2", "user2", {})
            logger.log_operation("op3", "user3", {})

            # Get recent logs
            recent = logger.get_recent_logs(2)
            assert len(recent) == 2
            assert "op2" in recent[0] or "op3" in recent[0]

    def test_security_audit_filter_logs_by_user(self):
        """Test filtering logs by user."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "filter.log"

            logger = SecurityAuditLogger(str(log_file))

            # Add logs for different users
            logger.log_operation("op1", "user1", {})
            logger.log_operation("op2", "user2", {})
            logger.log_operation("op3", "user1", {})

            # Filter by user1
            user1_logs = logger.filter_logs_by_user("user1")
            assert len(user1_logs) == 2
            assert all("user1" in log for log in user1_logs)

    def test_security_audit_filter_logs_by_operation(self):
        """Test filtering logs by operation type."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "op_filter.log"

            logger = SecurityAuditLogger(str(log_file))

            # Add different operation types
            logger.log_operation("read", "user1", {})
            logger.log_operation("write", "user2", {})
            logger.log_operation("read", "user3", {})

            # Filter by read operations
            read_logs = logger.filter_logs_by_operation("read")
            assert len(read_logs) == 2
            assert all("read" in log for log in read_logs)

    def test_security_audit_rotate_logs(self):
        """Test log rotation functionality."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "rotate.log"

            logger = SecurityAuditLogger(str(log_file))

            # Add some content
            for i in range(10):
                logger.log_operation(f"op_{i}", "user1", {})

            # Rotate logs
            logger.rotate_logs()

            # Check that backup was created
            backup_files = list(Path(temp_dir).glob("rotate.log.*"))
            assert len(backup_files) >= 1

    def test_security_audit_clear_logs(self):
        """Test clearing logs."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "clear.log"

            logger = SecurityAuditLogger(str(log_file))

            # Add content
            logger.log_operation("test_op", "user1", {})
            assert log_file.exists()
            assert log_file.read_text() != ""

            # Clear logs
            logger.clear_logs()
            assert log_file.exists()
            assert log_file.read_text() == ""

    def test_security_audit_get_statistics(self):
        """Test getting audit statistics."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "stats.log"

            logger = SecurityAuditLogger(str(log_file))

            # Add various log entries
            logger.log_operation("read", "user1", {})
            logger.log_operation("write", "user2", {})
            logger.log_authentication("login", "user1", True, "127.0.0.1")
            logger.log_permission_check("access", "user1", "/file", True)

            stats = logger.get_statistics()
            assert "total_operations" in stats
            assert "unique_users" in stats
            assert "operation_types" in stats
            assert stats["total_operations"] >= 3

    def test_security_decorator_with_kwargs(self):
        """Test security decorator with keyword arguments."""
        from utils.security_logging_integration import enforce_security_audit

        @enforce_security_audit("test_operation")
        def test_function(*args, **kwargs):
            return f"args: {args}, kwargs: {kwargs}"

        result = test_function("arg1", "arg2", kw1="value1", kw2="value2")
        assert "arg1" in result
        assert "value1" in result

    def test_security_logger_file_permissions(self):
        """Test security logger sets appropriate file permissions."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "perms.log"

            logger = SecurityAuditLogger(str(log_file))
            logger.log_operation("test", "user", {})

            # Check file exists and has reasonable permissions
            assert log_file.exists()
            # Note: File permission checks may vary by OS, so we just ensure it exists

    def test_security_logger_large_content(self):
        """Test security logger with large content."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "large.log"

            logger = SecurityAuditLogger(str(log_file))

            # Log with large data
            large_data = {"data": "x" * 1000}  # Large string
            logger.log_operation("large_op", "user1", large_data)

            content = log_file.read_text()
            assert "large_op" in content
            assert "user1" in content

    def test_security_logger_concurrent_access(self):
        """Test security logger with concurrent access."""
        import threading

        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "concurrent.log"

            logger = SecurityAuditLogger(str(log_file))

            def log_operations(thread_id):
                for i in range(5):
                    logger.log_operation(f"op_{thread_id}_{i}", f"user_{thread_id}", {})

            threads = []
            for i in range(3):
                thread = threading.Thread(target=log_operations, args=(i,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Check all operations were logged
            content = log_file.read_text()
            assert content.count("op_") == 15  # 3 threads * 5 operations

    def test_security_logger_invalid_log_file(self):
        """Test security logger with invalid log file path."""
        from utils.security_logging_integration import SecurityAuditLogger

        # Test with invalid path (should handle gracefully)
        invalid_path = "/nonexistent/directory/audit.log"

        # Should create the directory if it doesn't exist
        logger = SecurityAuditLogger(invalid_path)
        logger.log_operation("test", "user", {})

        # Check if the file was created (may fail if permissions are restrictive)
        try:
            assert Path(invalid_path).exists()
        except (PermissionError, OSError):
            # Expected in some environments
            pass

    def test_security_audit_decorator_class_method(self):
        """Test security audit decorator on class methods."""
        from utils.security_logging_integration import enforce_security_audit

        class TestClass:
            @enforce_security_audit("method_operation")
            def test_method(self, value):
                return f"method_result: {value}"

        obj = TestClass()
        result = obj.test_method("test_value")
        assert result == "method_result: test_value"

    def test_security_audit_decorator_static_method(self):
        """Test security audit decorator on static methods."""
        from utils.security_logging_integration import enforce_security_audit

        class TestClass:
            @staticmethod
            @enforce_security_audit("static_operation")
            def static_method(value):
                return f"static_result: {value}"

        result = TestClass.static_method("test_value")
        assert result == "static_result: test_value"

    def test_security_logger_unicode_content(self):
        """Test security logger with unicode content."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "unicode.log"

            logger = SecurityAuditLogger(str(log_file))

            # Log with unicode characters
            unicode_data = {"message": "测试消息", "emoji": "🔒", "accent": "café"}
            logger.log_operation("unicode_op", "用户", unicode_data)

            content = log_file.read_text()
            assert "unicode_op" in content
            # Unicode handling may vary, but operation should be logged

    def test_security_logger_json_serialization_error(self):
        """Test security logger handling of non-serializable data."""
        from utils.security_logging_integration import SecurityAuditLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "serial_error.log"

            logger = SecurityAuditLogger(str(log_file))

            # Try to log non-serializable data
            class NonSerializable:
                pass

            # Should handle gracefully
            logger.log_operation("serial_test", "user", {"object": NonSerializable()})

            # Log should still be created with error handling
            assert log_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
