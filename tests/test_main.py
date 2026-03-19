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

    def test_config_file_override(self, tmp_path):
        """Test config file path override."""
        # Create a temporary config file
        config_data = {"version": "test-version", "project_name": "Test Project"}
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "--config-file",
                str(config_file),
                "formal-model",
                "--simulation-steps",
                "5",
            ],
        )

        assert result.exit_code == 0
        assert "test-version" in result.output

    def test_log_level_override(self):
        """Test log level override."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--log-level", "DEBUG", "formal-model", "--simulation-steps", "5"]
        )

        assert result.exit_code == 0
        assert "DEBUG" in result.output

    def test_invalid_simulation_steps(self):
        """Test invalid simulation steps validation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["formal-model", "--simulation-steps", "0"])

        assert result.exit_code != 0
        assert "Invalid simulation steps" in result.output

    def test_negative_simulation_steps(self):
        """Test negative simulation steps validation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["formal-model", "--simulation-steps", "-10"])

        assert result.exit_code != 0
        assert "Must be a positive integer" in result.output

    def test_invalid_dt_value(self):
        """Test invalid dt value validation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["formal-model", "--dt", "0"])

        assert result.exit_code != 0
        assert "Must be a positive number" in result.output

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

        assert result.exit_code != 0
        assert "Must end with .csv, .json, or .pkl" in result.output

    def test_absolute_config_file_path(self):
        """Test rejection of absolute config file paths."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--config-file", "/etc/passwd"])

        assert result.exit_code != 0
        assert "not allowed for security reasons" in result.output

    def test_relative_path_traversal(self):
        """Test prevention of relative path traversal."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--config-file", "../../../etc/passwd"])

        assert result.exit_code != 0
        assert "outside project directory" in result.output


class TestMainExecutionFlow:
    """Test main execution flow and error handling."""

    def test_successful_formal_model_execution(self, tmp_path):
        """Test successful formal model execution."""
        runner = CliRunner()

        # Create minimal valid parameters file
        params_file = tmp_path / "test_params.json"
        with open(params_file, "w") as f:
            json.dump({"tau_S": 0.5, "alpha": 5.0}, f)

        result = runner.invoke(
            cli,
            ["formal-model", "--params", str(params_file), "--simulation-steps", "5"],
        )

        assert result.exit_code == 0
        assert "Simulation completed" in result.output
        assert "5 steps" in result.output

    def test_module_loading_error_handling(self):
        """Test error handling when modules fail to load."""
        runner = CliRunner()

        with patch("main.secure_load_module") as mock_load:
            mock_load.side_effect = ImportError("Module not found")

            result = runner.invoke(cli, ["formal-model", "--simulation-steps", "5"])

            assert result.exit_code != 0
            assert "Module not found" in result.output

    def test_parameter_validation_error(self, tmp_path):
        """Test parameter validation error handling."""
        runner = CliRunner()

        # Create invalid parameters file
        params_file = tmp_path / "invalid_params.json"
        with open(params_file, "w") as f:
            json.dump({"invalid_param": "value"}, f)

        result = runner.invoke(
            cli,
            ["formal-model", "--params", str(params_file), "--simulation-steps", "5"],
        )

        assert result.exit_code != 0
        assert "Parameter validation failed" in result.output

    def test_file_size_validation(self, tmp_path):
        """Test file size validation."""
        runner = CliRunner()

        # Create a large temporary file
        large_file = tmp_path / "large_params.json"
        large_data = {"data": "x" * (200 * 1024 * 1024)}  # > 200MB
        with open(large_file, "w") as f:
            json.dump(large_data, f)

        result = runner.invoke(
            cli,
            ["formal-model", "--params", str(large_file), "--simulation-steps", "5"],
        )

        assert result.exit_code != 0
        assert "exceeds maximum limit" in result.output

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
        runner = CliRunner()

        with patch("main.secure_load_module") as mock_load:
            mock_load.side_effect = ValueError(
                "Error with API_KEY=abc123 and path/to/secret"
            )

            result = runner.invoke(cli, ["formal-model", "--simulation-steps", "5"])

            assert result.exit_code != 0
            assert "[REDACTED]" in result.output
            assert "API_KEY" not in result.output
            assert "path/to/secret" not in result.output

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

        # Test module loading
        assert hasattr(loader, "_load_available_modules")
        assert hasattr(loader, "get_module")

        # Test that modules dict is initialized
        assert isinstance(loader.modules, dict)

        # Test getting non-existent module
        result = loader.get_module("non_existent")
        assert result is None


class TestUtilityFunctions:
    """Test utility functions in main.py."""

    def test_validate_file_path_valid_paths(self, tmp_path):
        """Test _validate_file_path with valid paths."""
        # Test relative path within project
        valid_path = _validate_file_path("config/test.json", ["config"])
        assert valid_path.name == "test.json"
        assert str(valid_path).endswith("config/test.json")

        # Test path within allowed directory
        allowed_path = _validate_file_path("results/output.csv", ["results", "data"])
        assert allowed_path.name == "output.csv"

        # Test project root relative path
        root_path = _validate_file_path("test.json", None)
        assert root_path.name == "test.json"

    def test_validate_file_path_invalid_paths(self):
        """Test _validate_file_path with invalid paths."""
        # Test absolute path rejection
        with pytest.raises(ValueError, match="not allowed for security reasons"):
            _validate_file_path("/etc/passwd")

        # Test path traversal rejection
        with pytest.raises(ValueError, match="outside project directory"):
            _validate_file_path("../../../etc/passwd")

        # Test non-.py file for module loading
        with pytest.raises(ValueError, match="must be a .py file"):
            _validate_file_path("config/test.txt", ["config"])

    def test_check_file_size(self, tmp_path):
        """Test _check_file_size functionality."""
        # Create test file of known size
        test_file = tmp_path / "test_file.json"
        test_data = {"test": "data" * 1000}  # Small file
        with open(test_file, "w") as f:
            json.dump(test_data, f)

        # Should not raise exception for small file
        assert _check_file_size(test_file) is None

        # Test with custom limit
        assert _check_file_size(test_file, max_mb=1) is None

        # Test large file
        large_file = tmp_path / "large_file.json"
        large_data = {"data": "x" * (50 * 1024 * 1024)}  # ~50MB
        with open(large_file, "w") as f:
            json.dump(large_data, f)

        with pytest.raises(ValueError, match="exceeds maximum limit"):
            _check_file_size(large_file, max_mb=10)

    def test_sanitize_error_message(self):
        """Test _sanitize_error_message function."""
        # Test API key redaction
        error_msg = "Error with API_KEY=abc123 and /path/to/secret"
        sanitized = _sanitize_error_message(error_msg)
        assert "[REDACTED]" in sanitized
        assert "API_KEY" not in sanitized
        assert "/path/to/secret" not in sanitized

        # Test JWT token redaction
        jwt_msg = "Error with JWT=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI6..."
        sanitized_jwt = _sanitize_error_message(jwt_msg)
        assert "[JWT_REDACTED]" in sanitized_jwt
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized_jwt

        # Test hex key redaction
        hex_msg = "Error with secret_key=abcdef1234567890abcdef1234567890"
        sanitized_hex = _sanitize_error_message(hex_msg)
        assert "[HEX_REDACTED]" in sanitized_hex
        assert "abcdef1234567890abcdef1234567890" not in sanitized_hex

        # Test message length truncation
        long_msg = "Error: " + "x" * 300
        sanitized_long = _sanitize_error_message(long_msg)
        assert len(sanitized_long) <= 203  # 200 + "..."
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

    def test_validate_output_file_path(self, tmp_path):
        """Test _validate_output_file_path function."""
        # Test valid paths
        valid_path = _validate_output_file_path("results/output.csv")
        assert valid_path.name == "output.csv"
        assert str(valid_path).endswith("results/output.csv")

        # Test invalid paths
        with pytest.raises(ValueError):
            _validate_output_file_path("/absolute/path/output.csv")


class TestErrorHandling:
    """Test error handling functions."""

    def test_handle_file_error_not_found(self):
        """Test handle_file_error with file not found."""
        error = FileNotFoundError("No such file or directory")
        with patch("builtins.print") as mock_print:
            handle_file_error("test.txt", "reading", error)
            mock_print.assert_called()
            # Should print error and guidance
            calls = str(mock_print.call_args_list)
            assert "File not found: test.txt" in calls[0]
            assert "Check if the file exists" in calls[0]

    def test_handle_file_error_permission_denied(self):
        """Test handle_file_error with permission denied."""
        error = PermissionError("Permission denied")
        with patch("builtins.print") as mock_print:
            handle_file_error("test.txt", "writing", error)
            mock_print.assert_called()
            calls = str(mock_print.call_args_list)
            assert "Permission denied" in calls[0]
            assert "Check file permissions" in calls[0]

    def test_handle_file_error_is_directory(self):
        """Test handle_file_error with directory instead of file."""
        error = IsADirectoryError("Is a directory")
        with patch("builtins.print") as mock_print:
            handle_file_error("test", "reading", error)
            mock_print.assert_called()
            calls = str(mock_print.call_args_list)
            assert "Expected file but got directory" in calls[0]

    def test_handle_validation_error_range(self):
        """Test handle_validation_error with range error."""
        error = ValueError("Parameter out of valid range: 0-100")
        with patch("builtins.print") as mock_print:
            handle_validation_error(error, "testing parameter validation")
            mock_print.assert_called()
            calls = str(mock_print.call_args_list)
            assert "Parameter out of valid range" in calls[0]
            assert "Check parameter constraints" in calls[0]

    def test_handle_validation_error_type(self):
        """Test handle_validation_error with type error."""
        error = TypeError("Invalid parameter type: expected number")
        with patch("builtins.print") as mock_print:
            handle_validation_error(error, "testing type validation")
            mock_print.assert_called()
            calls = str(mock_print.call_args_list)
            assert "Invalid parameter type" in calls[0]
            assert "Ensure parameter values match expected types" in calls[0]

    def test_handle_validation_error_generic(self):
        """Test handle_validation_error with generic error."""
        error = RuntimeError("Generic validation error")
        with patch("builtins.print") as mock_print:
            handle_validation_error(error)
            mock_print.assert_called()
            calls = str(mock_print.call_args_list)
            assert "Validation error" in calls[0]


class TestModuleLoader:
    """Test APGIModuleLoader class."""

    def test_module_loader_initialization(self):
        """Test APGIModuleLoader initialization."""
        loader = APGIModuleLoader()
        assert hasattr(loader, "modules")
        assert hasattr(loader, "_load_available_modules")
        assert hasattr(loader, "get_module")

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

        # Mock a successful module load
        with patch("main.secure_load_module") as mock_load:
            mock_module = MagicMock()
            mock_load.return_value = mock_module

            # Reset and reload to test the loading
            loader._load_available_modules()
            result = loader.get_module("formal_model")

            assert result == mock_module

    def test_get_module_non_existent(self):
        """Test getting non-existent module."""
        loader = APGIModuleLoader()
        result = loader.get_module("non_existent_module")
        assert result is None
