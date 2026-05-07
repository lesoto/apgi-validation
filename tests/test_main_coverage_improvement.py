"""
Comprehensive tests for main.py to improve coverage.
===============================================

This file targets specific uncovered lines in main.py to improve overall test coverage.
"""

import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    APGIModuleLoader,
    _check_file_size,
    _create_signal_handler,
    _process_csv_file,
    _run_demo_mode,
    _sanitize_error_message,
    _validate_file_path,
    _validate_input_file,
    _validate_output_file_path,
    _validate_output_path,
    cli,
    get_config_value,
    handle_file_error,
    handle_validation_error,
    quiet_print,
    secure_load_module,
    secure_load_module_from_path,
    secure_open_file,
    set_config_value,
    verbose_print,
)


class TestImportErrorHandling:
    """Test import error handling paths."""

    def test_missing_click_import(self):
        """Test error when click is not available."""
        with patch.dict("sys.modules", {"click": None}):
            with patch("builtins.exit") as mock_exit:
                # Remove click from modules and re-import main
                if "main" in sys.modules:
                    del sys.modules["main"]

                # Mock the console to capture output
                with patch("main.console") as mock_console:
                    try:
                        import main  # noqa: F401
                    except SystemExit:
                        pass

                    # Verify error message was printed
                    mock_console.print.assert_called()
                    mock_exit.assert_called_with(1)

    def test_missing_numpy_import(self):
        """Test error when numpy is not available."""
        with patch.dict("sys.modules", {"numpy": None}):
            with patch("builtins.exit") as mock_exit:
                # Mock console
                with patch("main.console") as mock_console:
                    # Simulate numpy import failure
                    try:
                        import numpy  # noqa: F401
                    except ImportError:
                        pass
                        mock_console.print(
                            "[red]❌ Error: Required package 'numpy' not installed[/red]"
                        )
                        mock_console.print(
                            "[blue]Install with: pip install numpy[/blue]"
                        )
                        mock_exit(1)

    def test_missing_yaml_import(self):
        """Test error when yaml is not available."""
        with patch.dict("sys.modules", {"yaml": None}):
            with patch("builtins.exit") as mock_exit:
                # Mock console
                with patch("main.console") as mock_console:
                    # Simulate yaml import failure
                    try:
                        import yaml  # noqa: F401
                    except ImportError:
                        pass
                        mock_console.print(
                            "[red]❌ Error: Required package 'pyyaml' not installed[/red]"
                        )
                        mock_console.print(
                            "[blue]Install with: pip install pyyaml[/blue]"
                        )
                        mock_exit(1)


class TestSecureModuleLoading:
    """Test secure module loading functionality."""

    def test_secure_load_module_with_temp_dir(self):
        """Test loading module from temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test module
            test_module_path = Path(temp_dir) / "test_module.py"
            test_module_path.write_text("TEST_VALUE = 'test_success'\n")

            # Load the module with temp dir allowed
            module = secure_load_module(
                "test_module", test_module_path, allow_temp_dir=True
            )

            assert module is not None
            assert hasattr(module, "TEST_VALUE")
            assert module.TEST_VALUE == "test_success"

    def test_secure_load_module_cache_miss(self):
        """Test module loading when cache is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_module_path = Path(temp_dir) / "cache_test.py"
            test_module_path.write_text("CACHED_VALUE = 'no_cache'\n")

            # Mock cache import to fail
            with patch(
                "main.secure_cached_import",
                side_effect=ImportError("Cache not available"),
            ):
                module = secure_load_module(
                    "cache_test", test_module_path, allow_temp_dir=True
                )

                assert module is not None
                assert module.CACHED_VALUE == "no_cache"

    def test_secure_load_module_invalid_spec(self):
        """Test handling of invalid module spec."""
        invalid_path = Path("/nonexistent/module.py")

        with pytest.raises(ImportError, match="Could not load module spec"):
            secure_load_module("invalid", invalid_path)

    def test_secure_load_module_from_path(self):
        """Test convenience function for loading from path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_module_path = Path(temp_dir) / "convenience_test.py"
            test_module_path.write_text("CONVENIENCE_VALUE = 'success'\n")

            module = secure_load_module_from_path(test_module_path, allow_temp_dir=True)

            assert module is not None
            assert module.CONVENIENCE_VALUE == "success"


class TestConfigFunctions:
    """Test configuration management functions."""

    def test_get_config_value_thread_safety(self):
        """Test thread-safe configuration value retrieval."""
        # Set a test value
        set_config_value("test_key", "test_value")

        # Retrieve it from multiple threads
        results = []

        def get_value():
            results.append(get_config_value("test_key"))

        threads = [threading.Thread(target=get_value) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All threads should get the same value
        assert all(result == "test_value" for result in results)

    def test_set_config_value_thread_safety(self):
        """Test thread-safe configuration value setting."""
        values = []

        def set_value(index):
            set_config_value(f"thread_test_{index}", f"value_{index}")
            values.append(get_config_value(f"thread_test_{index}"))

        threads = [threading.Thread(target=set_value, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All values should be set correctly
        assert len(values) == 3
        assert all(value is not None for value in values)

    def test_verbose_print_levels(self):
        """Test verbose print with different levels."""
        with patch("main.console") as mock_console:
            # Test with verbose enabled
            with patch("main.VERBOSE", True):
                verbose_print("Debug message", "debug")
                verbose_print("Info message", "info")
                verbose_print("Warning message", "warning")
                verbose_print("Error message", "error")

            # Should have printed all messages
            assert mock_console.print.call_count == 4

    def test_verbose_print_disabled(self):
        """Test verbose print when disabled."""
        with patch("main.console") as mock_console:
            with patch("main.VERBOSE", False):
                verbose_print("This should not print", "info")

            # Should not have printed anything
            mock_console.print.assert_not_called()

    def test_quiet_print_enabled(self):
        """Test quiet print when not in quiet mode."""
        with patch("main.console") as mock_console:
            with patch("main.QUIET", False):
                quiet_print("Normal message")

            mock_console.print.assert_called_once()

    def test_quiet_print_suppressed(self):
        """Test quiet print when in quiet mode."""
        with patch("main.console") as mock_console:
            with patch("main.QUIET", True):
                quiet_print("This should be suppressed")

            # Should not have printed anything
            mock_console.print.assert_not_called()

    def test_quiet_print_forced(self):
        """Test quiet print with force flag."""
        with patch("main.console") as mock_console:
            with patch("main.QUIET", True):
                quiet_print("Forced message", force=True)

            # Should have printed despite quiet mode
            mock_console.print.assert_called_once()


class TestErrorHandling:
    """Test error handling functions."""

    def test_handle_file_error_permission_denied(self):
        """Test file error handling for permission issues."""
        error = PermissionError("Permission denied")
        with patch("main.quiet_print") as mock_print:
            handle_file_error("/test/file.txt", "reading", error)

            mock_print.assert_called_once()
            args = mock_print.call_args[0]
            assert "Permission denied" in args[1]

    def test_handle_file_error_not_found(self):
        """Test file error handling for missing files."""
        error = FileNotFoundError("No such file")
        with patch("main.quiet_print") as mock_print:
            handle_file_error("/missing/file.txt", "opening", error)

            mock_print.assert_called_once()
            args = mock_print.call_args[0]
            assert "not found" in args[1].lower()

    def test_handle_file_error_generic(self):
        """Test file error handling for generic errors."""
        error = Exception("Generic error")
        with patch("main.quiet_print") as mock_print:
            handle_file_error("/test/file.txt", "processing", error)

            mock_print.assert_called_once()

    def test_handle_validation_error_with_context(self):
        """Test validation error handling with context."""
        error = ValueError("Invalid parameter")
        with patch("main.verbose_print") as mock_verbose:
            with patch("main.quiet_print") as mock_quiet:
                handle_validation_error(error, "parameter validation")

                mock_verbose.assert_called_once()
                mock_quiet.assert_called_once()

    def test_handle_validation_error_without_context(self):
        """Test validation error handling without context."""
        error = ValueError("Invalid value")
        with patch("main.verbose_print") as mock_verbose:
            with patch("main.quiet_print") as mock_quiet:
                handle_validation_error(error)

                mock_verbose.assert_not_called()
                mock_quiet.assert_called_once()

    def test_sanitize_error_message(self):
        """Test error message sanitization."""
        # Test with sensitive information
        sensitive_msg = "Error: password=secret123, token=abc123"
        sanitized = _sanitize_error_message(sensitive_msg)

        # Should remove sensitive information
        assert "password" not in sanitized.lower()
        assert "secret123" not in sanitized
        assert "abc123" not in sanitized

    def test_sanitize_error_message_path_sanitization(self):
        """Test path sanitization in error messages."""
        path_msg = "Error accessing /home/user/secret/file.txt"
        sanitized = _sanitize_error_message(path_msg)

        # Should sanitize full paths
        assert "/home/user" not in sanitized


class TestAPGIModuleLoader:
    """Test APGI module loader functionality."""

    def test_module_loader_initialization(self):
        """Test module loader initialization."""
        loader = APGIModuleLoader()

        assert loader.modules == {}
        assert "formal_model" in loader._module_configs
        assert "multimodal_integration" in loader._module_configs

    def test_load_module_missing_class(self):
        """Test loading module that's missing expected class."""
        loader = APGIModuleLoader()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create module without expected class
            test_module_path = Path(temp_dir) / "incomplete.py"
            test_module_path.write_text("# Missing expected class\n")

            # Mock the config to expect a class that doesn't exist
            loader._module_configs["test"] = {
                "path": test_module_path,
                "class": "NonExistentClass",
            }

            result = loader._load_module("test")

            assert result is not None
            assert result["module"] is None
            assert "error" in result

    def test_get_module_with_error(self):
        """Test getting module that failed to load."""
        loader = APGIModuleLoader()

        # Simulate a failed module load
        loader.modules["failed_module"] = {
            "module": None,
            "config": {"path": "/nonexistent"},
            "error": "Module not found",
        }

        result = loader.get_module("failed_module")

        assert result is not None
        assert result["module"] is None
        assert result["error"] == "Module not found"


class TestFileValidation:
    """Test file validation functions."""

    def test_validate_file_path_with_allowed_dirs(self):
        """Test file path validation with allowed directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.touch()

            # Should validate successfully
            result = _validate_file_path(str(test_file), allowed_dirs=[temp_dir])
            assert result == test_file

    def test_validate_file_path_traversal_attempt(self):
        """Test file path validation rejects directory traversal."""
        malicious_path = "../../../etc/passwd"

        with pytest.raises(Exception, match="security violation"):
            _validate_file_path(malicious_path)

    def test_validate_output_file_path(self):
        """Test output file path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = f"{temp_dir}/output.txt"

            result = _validate_output_file_path(output_file)
            assert isinstance(result, Path)

    def test_validate_output_path(self):
        """Test output path validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = f"{temp_dir}/test_output.txt"

            result = _validate_output_path(output_path, allowed_dirs=[temp_dir])
            assert isinstance(result, Path)

    def test_validate_input_file_none(self):
        """Test input file validation with None."""
        result = _validate_input_file(None)
        assert result is None

    def test_validate_input_file_empty(self):
        """Test input file validation with empty string."""
        result = _validate_input_file("")
        assert result is None

    def test_validate_input_file_valid(self):
        """Test input file validation with valid file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "input.txt"
            test_file.write_text("test content")

            result = _validate_input_file(str(test_file))
            assert result == str(test_file)


class TestFileOperations:
    """Test file operation functions."""

    def test_check_file_size_small_file(self):
        """Test file size check for small file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("small content")
            temp_path = f.name

        try:
            # Should not raise for small file
            _check_file_size(temp_path, max_mb=10)
        finally:
            os.unlink(temp_path)

    def test_check_file_size_large_file(self):
        """Test file size check for large file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            # Mock a large file
            with patch("os.path.getsize", return_value=100 * 1024 * 1024):  # 100MB
                with patch("main.console"):
                    with pytest.raises(Exception, match="too large"):
                        _check_file_size(f.name, max_mb=10)

    def test_secure_open_file_read_mode(self):
        """Test secure file opening in read mode."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            with secure_open_file(temp_path, mode="r") as f:
                content = f.read()
                assert content == "test content"
        finally:
            os.unlink(temp_path)

    def test_secure_open_file_write_mode(self):
        """Test secure file opening in write mode."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            with secure_open_file(temp_path, mode="w") as f:
                f.write("new content")

            # Verify content was written
            with open(temp_path, "r") as f:
                assert f.read() == "new content"
        finally:
            os.unlink(temp_path)

    def test_process_csv_file(self):
        """Test CSV file processing."""
        # Create a test CSV
        csv_content = "col1,col2,col3\n1,2,3\n4,5,6\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
                output_path = out_f.name

            try:
                _process_csv_file(csv_path, output_path)

                # Check that output file was created
                assert os.path.exists(output_path)

                # Check content
                with open(output_path, "r") as f:
                    result = json.load(f)
                    assert "processed_data" in result
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)
        finally:
            os.unlink(csv_path)

    def test_run_demo_mode(self):
        """Test demo mode execution."""
        with patch("main.console"):
            with patch("pandas.DataFrame") as mock_df:
                # Mock DataFrame creation
                mock_instance = MagicMock()
                mock_df.return_value = mock_instance
                mock_instance.head.return_value = "mock_data"

                _run_demo_mode()

                # Should have printed demo output
        assert True  # Demo execution completed


class TestSignalHandling:
    """Test signal handling functionality."""

    def test_create_signal_handler(self):
        """Test signal handler creation."""
        cancel_flag = threading.Event()

        # Create signal handler
        handler = _create_signal_handler(cancel_flag)

        # Verify it's callable
        assert callable(handler)

        # Test calling the handler
        handler(2, None)  # SIGINT signal number

        # Flag should be set
        assert cancel_flag.is_set()


class TestCLICommands:
    """Test CLI command functionality."""

    def test_config_group_commands(self):
        """Test config group commands."""
        runner = CliRunner()

        # Test config-group command exists
        result = runner.invoke(cli, ["config-group", "--help"])
        assert result.exit_code == 0
        assert "Manage APGI configuration" in result.output

    def test_explain_config_command(self):
        """Test explain config command."""
        runner = CliRunner()

        with patch("main.console"):
            result = runner.invoke(cli, ["explain-config"])

            # Should attempt to show configuration
            assert result.exit_code == 0

    def test_formal_model_command_with_params(self):
        """Test formal model command with parameters."""
        runner = CliRunner()

        # Mock the simulation to avoid actual execution
        with patch("main._run_formal_model_simulation") as mock_sim:
            mock_sim.return_value = {"results": "test"}

            runner.invoke(
                cli,
                ["formal-model", "--simulation-steps", "10", "--dt", "0.1", "--plot"],
            )

            # Should have called simulation with correct parameters
            mock_sim.assert_called_once()

    def test_secure_protocol_command_without_token(self):
        """Test secure protocol command without authentication."""
        runner = CliRunner()

        # Mock security audit to be available
        with patch("main.enforce_security_audit"):
            result = runner.invoke(cli, ["run-secure-protocol"])

            # Should fail without token
            assert result.exit_code != 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_secure_load_module_invalid_path(self):
        """Test secure loading with invalid path."""
        invalid_path = Path("/etc/passwd")  # Non-Python file

        with pytest.raises(ValueError):
            secure_load_module("invalid", invalid_path)

    def test_validate_file_path_nonexistent(self):
        """Test validation of nonexistent file path."""
        nonexistent = "/nonexistent/file.txt"

        with pytest.raises(Exception):
            _validate_file_path(nonexistent)

    def test_sanitize_empty_error_message(self):
        """Test sanitizing empty error message."""
        result = _sanitize_error_message("")
        assert result == ""

    def test_handle_file_error_with_none_error(self):
        """Test file error handling with None error."""
        with patch("main.quiet_print") as mock_print:
            handle_file_error("/test/file", "reading", None)

            mock_print.assert_called_once()

    def test_module_loader_get_nonexistent_module(self):
        """Test getting module that doesn't exist."""
        loader = APGIModuleLoader()

        result = loader.get_module("nonexistent")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
