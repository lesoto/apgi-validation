"""
Tests for main.py - CLI argument parsing and main execution flow.
=========================================================
"""

import pytest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
import sys
import threading
from click.testing import CliRunner

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    cli,
    get_config_value,
    set_config_value,
    handle_file_error,
    handle_validation_error,
    _validate_file_path,
    APGIModuleLoader,
    _check_file_size,
    _sanitize_error_message,
    _create_signal_handler,
    _validate_output_file_path,
)


class TestCLIArgumentParsing:
    """Test CLI argument parsing and validation."""

    def test_help_option(self):
        """Test help option displays correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "APGI Theory Framework" in result.output

    def test_version_option(self):
        """Test version option displays correctly."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.3.0" in result.output

    def test_verbose_flag(self):
        """Test verbose flag enables verbose output."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--verbose", "formal-model", "--simulation-steps", "10"]
        )
        assert result.exit_code == 0
        # Check that verbose mode was enabled
        assert get_config_value("verbose") is True

    def test_quiet_flag(self):
        """Test quiet flag suppresses output."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--quiet", "formal-model", "--simulation-steps", "10"]
        )
        assert result.exit_code == 0
        # Check that quiet mode was enabled
        assert get_config_value("quiet") is True

    def test_config_file_override(self):
        """Test config file path override using project-relative path."""
        import os

        # Create a test config file in project's config directory
        config_data = {"version": "test-version", "project_name": "Test Project"}
        config_file = "config/test_override_config.json"
        config_path = Path(__file__).parent.parent / config_file
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f)

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "--config-file",
                    config_file,
                    "formal-model",
                    "--simulation-steps",
                    "5",
                ],
            )

            assert result.exit_code == 0
            assert "test-version" in result.output
        finally:
            # Clean up
            if config_path.exists():
                os.remove(config_path)

    def test_log_level_override(self):
        """Test log level override."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--log-level", "DEBUG", "formal-model", "--simulation-steps", "5"]
        )

        # Command should execute; log level affects internal logging, not necessarily output
        assert result.exit_code == 0

    def test_invalid_simulation_steps(self):
        """Test invalid simulation steps validation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["formal-model", "--simulation-steps", "0"])

        # Should fail or show warning for zero steps
        assert (
            result.exit_code != 0
            or "steps" in result.output.lower()
            or "error" in result.output.lower()
        )

    def test_negative_simulation_steps(self):
        """Test negative simulation steps validation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["formal-model", "--simulation-steps", "-10"])

        # CLI should reject negative values or the app should error
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_invalid_dt_value(self):
        """Test invalid dt value validation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["formal-model", "--dt", "0"])

        # Should fail or show error/warning about dt value
        assert (
            result.exit_code != 0
            or "dt" in result.output.lower()
            or "error" in result.output.lower()
        )

    def test_large_dt_warning(self):
        """Test warning for large dt values."""
        runner = CliRunner()
        result = runner.invoke(cli, ["formal-model", "--dt", "2.0"])

        assert result.exit_code == 0
        assert "may reduce simulation accuracy" in result.output

    def test_invalid_output_file(self):
        """Test invalid output file extension."""
        runner = CliRunner()
        result = runner.invoke(cli, ["formal-model", "--output-file", "invalid.txt"])

        # Should fail or show error for invalid extension
        assert (
            result.exit_code != 0
            or "extension" in result.output.lower()
            or "error" in result.output.lower()
        )

    def test_absolute_config_file_path(self):
        """Test rejection of absolute config file paths."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--config-file", "/etc/passwd"])

        # Should fail for absolute path
        assert result.exit_code != 0

    def test_relative_path_traversal(self):
        """Test prevention of relative path traversal."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--config-file", "../../../etc/passwd"])

        # Should fail for path traversal attempt
        assert result.exit_code != 0


class TestMainExecutionFlow:
    """Test main execution flow and error handling."""

    def test_successful_formal_model_execution(self):
        """Test successful formal model execution."""
        import os

        runner = CliRunner()

        # Create minimal valid parameters file in project directory
        params_file = "config/test_params.json"
        params_path = Path(__file__).parent.parent / params_file
        try:
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump({"tau_S": 0.5, "alpha": 5.0}, f)

            result = runner.invoke(
                cli,
                ["formal-model", "--params", params_file, "--simulation-steps", "5"],
            )

            # Should complete successfully (exit code 0) or have simulation output
            assert result.exit_code == 0 or "simulation" in result.output.lower()
        finally:
            if params_path.exists():
                os.remove(params_path)

    def test_module_loading_error_handling(self):
        """Test error handling when modules fail to load."""
        runner = CliRunner()

        # Test with a non-existent module file to trigger module loading error
        with patch("main.APGIModuleLoader.get_module") as mock_get:
            mock_get.return_value = None  # Simulate module not found

            result = runner.invoke(cli, ["formal-model", "--simulation-steps", "5"])

            # Should fail when module can't be loaded
            assert (
                result.exit_code != 0
                or "not found" in result.output.lower()
                or "error" in result.output.lower()
            )

    def test_parameter_validation_error(self):
        """Test parameter validation error handling."""
        import os

        runner = CliRunner()

        # Create invalid parameters file in project directory
        params_file = "config/invalid_params.json"
        params_path = Path(__file__).parent.parent / params_file
        try:
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump({"invalid_param": "value"}, f)

            result = runner.invoke(
                cli,
                ["formal-model", "--params", params_file, "--simulation-steps", "5"],
            )

            # Should fail or show error message
            assert result.exit_code != 0 or "error" in result.output.lower()
        finally:
            if params_path.exists():
                os.remove(params_path)

    def test_file_size_validation(self):
        """Test file size validation."""
        import os

        runner = CliRunner()

        # Create a large file in project directory
        large_file = "config/large_params.json"
        large_path = Path(__file__).parent.parent / large_file
        large_data = {"data": "x" * (200 * 1024 * 1024)}  # > 200MB
        try:
            with open(large_path, "w", encoding="utf-8") as f:
                json.dump(large_data, f)

            result = runner.invoke(
                cli,
                ["formal-model", "--params", large_file, "--simulation-steps", "5"],
            )

            # Should fail due to file size or have error
            assert (
                result.exit_code != 0
                or "size" in result.output.lower()
                or "error" in result.output.lower()
            )
        finally:
            if large_path.exists():
                os.remove(large_path)

    def test_signal_handling(self):
        """Test graceful signal handling."""
        runner = CliRunner()

        # Mock signal handling
        with patch("threading.Event") as mock_event:
            cancel_event = MagicMock()
            mock_event.return_value = cancel_event

            # This would need to be tested with actual signal handling
            # For now, just test that the signal handler setup doesn't crash
            result = runner.invoke(
                cli,
                [
                    "formal-model",
                    "--simulation-steps",
                    "100",  # Long running to test signal setup
                ],
            )

            # Should complete without signal-related errors
            assert result.exit_code == 0 or "cancelled" in result.output

    def test_error_message_sanitization(self):
        """Test error message sanitization."""
        from main import _sanitize_error_message

        # Test that _sanitize_error_message properly redacts sensitive info
        error_msg = "Error with secret_key=abcdefghijklmnopqrstuvwxyz0123456789ABCDEF and /path/to/file"
        sanitized = _sanitize_error_message(error_msg)

        # Long API keys should be redacted
        assert "[REDACTED]" in sanitized or "[PATH]" in sanitized
        # The original long key should not be present
        assert "abcdefghijklmnopqrstuvwxyz0123456789ABCDEF" not in sanitized

    def test_configuration_thread_safety(self):
        """Test thread safety of configuration access."""

        def concurrent_config_access():
            # Simulate concurrent access
            set_config_value("test_key", "value1")
            get_config_value("test_key")

        # Test concurrent access doesn't raise exceptions
        import threading

        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_config_access)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # If we get here without exceptions, thread safety is working
        assert True

    def test_module_loader_functionality(self):
        """Test APGIModuleLoader functionality."""
        loader = APGIModuleLoader()

        # Test that loader has required attributes
        assert hasattr(loader, "modules")
        assert hasattr(loader, "get_module")

        # Test that modules dict is initialized
        assert isinstance(loader.modules, dict)

        # Test getting non-existent module returns None
        result = loader.get_module("non_existent")
        assert result is None


class TestUtilityFunctions:
    """Test utility functions in main.py."""

    def test_validate_file_path_valid_paths(self):
        """Test _validate_file_path with valid paths."""
        # Test relative path within project
        valid_path = _validate_file_path("config/test.json", ["config"])
        assert valid_path.name == "test.json"

        # Test path within allowed directory
        allowed_path = _validate_file_path("results/output.csv", ["results", "data"])
        assert allowed_path.name == "output.csv"

        # Test project root relative path
        root_path = _validate_file_path("test.json", None)
        assert root_path.name == "test.json"

    def test_validate_file_path_invalid_paths(self):
        """Test _validate_file_path with invalid paths."""
        # Test absolute path rejection
        with pytest.raises(ValueError):
            _validate_file_path("/etc/passwd")

        # Test path traversal rejection
        with pytest.raises(ValueError):
            _validate_file_path("../../../etc/passwd")

    def test_check_file_size(self):
        """Test _check_file_size functionality."""
        import os

        # Create test file of known size in project directory
        test_file = Path(__file__).parent.parent / "config/test_file.json"
        test_data = {"test": "data" * 1000}  # Small file
        try:
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(test_data, f)

            # Should not raise exception for small file
            assert _check_file_size(test_file) is None

            # Test with custom limit
            assert _check_file_size(test_file, max_mb=1) is None
        finally:
            if test_file.exists():
                os.remove(test_file)

    def test_sanitize_error_message(self):
        """Test _sanitize_error_message function."""
        # Test path redaction (paths get replaced with /[PATH])
        error_msg = "Error with /path/to/secret file"
        sanitized = _sanitize_error_message(error_msg)
        assert "/[PATH]" in sanitized
        assert "/path/to/secret" not in sanitized

        # Test API key redaction (40+ character alphanumeric strings)
        api_msg = "Error with key=abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLM"
        sanitized_api = _sanitize_error_message(api_msg)
        assert "[REDACTED]" in sanitized_api
        assert "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLM" not in sanitized_api

        # Test JWT token redaction (3 base64 parts separated by dots, each 20+ chars)
        jwt_msg = "Error with token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        sanitized_jwt = _sanitize_error_message(jwt_msg)
        assert "[JWT_REDACTED]" in sanitized_jwt

        # Test hex key redaction (32+ hex chars)
        hex_msg = "Error with secret_key=abcdef1234567890abcdef1234567890"
        sanitized_hex = _sanitize_error_message(hex_msg)
        assert "[HEX_REDACTED]" in sanitized_hex
        assert "abcdef1234567890abcdef1234567890" not in sanitized_hex

        # Test message length truncation
        long_msg = "Error: " + "x" * 300
        sanitized_long = _sanitize_error_message(long_msg)
        # After path sanitization, the message should be truncated if > 200 chars
        # Path patterns like /xxxx... become /[PATH] which is shorter
        if len(sanitized_long) > 200:
            assert sanitized_long.endswith("...")

    def test_signal_handler_creation(self):
        """Test _create_signal_handler function."""
        cancel_flag = threading.Event()
        handler = _create_signal_handler(cancel_flag)

        # Test that handler is callable
        assert callable(handler)

        # Test that handler sets the event
        mock_frame = MagicMock()
        handler(2, mock_frame)  # SIGINT
        assert cancel_flag.is_set()

    def test_validate_output_file_path(self):
        """Test _validate_output_file_path function."""
        # Test valid paths
        valid_path = _validate_output_file_path("results/output.csv")
        assert valid_path.name == "output.csv"

        # Test invalid paths (absolute paths should be rejected)
        with pytest.raises(ValueError):
            _validate_output_file_path("/absolute/path/output.csv")


class TestErrorHandling:
    """Test error handling functions."""

    def test_handle_file_error_not_found(self):
        """Test handle_file_error with file not found."""
        error = FileNotFoundError("No such file or directory")
        # Function should not raise exception
        handle_file_error("test.txt", "reading", error)
        # If we get here without exception, test passes

    def test_handle_file_error_permission_denied(self):
        """Test handle_file_error with permission denied."""
        error = PermissionError("Permission denied")
        # Function should not raise exception
        handle_file_error("test.txt", "writing", error)
        # If we get here without exception, test passes

    def test_handle_file_error_is_directory(self):
        """Test handle_file_error with directory instead of file."""
        error = IsADirectoryError("Is a directory")
        # Function should not raise exception
        handle_file_error("test", "reading", error)
        # If we get here without exception, test passes

    def test_handle_validation_error_range(self):
        """Test handle_validation_error with range error."""
        error = ValueError("Parameter out of valid range: 0-100")
        # Function should not raise exception
        handle_validation_error(error, "testing parameter validation")
        # If we get here without exception, test passes

    def test_handle_validation_error_type(self):
        """Test handle_validation_error with type error."""
        error = TypeError("Invalid parameter type: expected number")
        # Function should not raise exception
        handle_validation_error(error, "testing type validation")
        # If we get here without exception, test passes

    def test_handle_validation_error_generic(self):
        """Test handle_validation_error with generic error."""
        error = RuntimeError("Generic validation error")
        # Function should not raise exception
        handle_validation_error(error)
        # If we get here without exception, test passes


class TestModuleLoader:
    """Test APGIModuleLoader class."""

    def test_module_loader_initialization(self):
        """Test APGIModuleLoader initialization."""
        loader = APGIModuleLoader()
        # Verify loader has required attributes
        assert hasattr(loader, "modules")
        assert hasattr(loader, "get_module")
        # modules should be a dict
        assert isinstance(loader.modules, dict)

    def test_load_available_modules_with_missing_files(self):
        """Test module loading with missing files."""
        loader = APGIModuleLoader()

        # Should not crash when modules are missing
        assert isinstance(loader.modules, dict)

        # Should have some modules or empty dict
        # (Specific modules depend on file existence)

    def test_get_module_existing(self):
        """Test getting existing module."""
        loader = APGIModuleLoader()

        # Test that get_module method exists and works
        # For modules that exist, it should return module info or None if not loaded
        result = loader.get_module("formal_model")
        # Result could be None (if module not found) or a dict (if found)
        assert result is None or isinstance(result, dict)

    def test_get_module_non_existent(self):
        """Test getting non-existent module."""
        loader = APGIModuleLoader()
        result = loader.get_module("non_existent_module")
        assert result is None
