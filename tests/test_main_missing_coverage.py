"""
Targeted tests for main.py missing coverage areas.
===============================================

Focuses on specific functions and lines that are currently uncovered.
"""

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


class TestMissingCoverageFunctions:
    """Test functions with missing coverage."""

    def test_secure_load_module_temp_dir(self):
        """Test secure_load_module with temp directory allowed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_module_path = Path(temp_dir) / "test_mod.py"
            test_module_path.write_text("TEST_VAL = 'success'")

            module = secure_load_module(
                "test_mod", test_module_path, allow_temp_dir=True
            )
            assert module is not None
            assert module.TEST_VAL == "success"

    def test_secure_load_module_cache_failure(self):
        """Test secure_load_module when cache fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_module_path = Path(temp_dir) / "cache_fail.py"
            test_module_path.write_text("VALUE = 'no_cache'")

            with patch("main.secure_cached_import", side_effect=ImportError):
                module = secure_load_module(
                    "cache_fail", test_module_path, allow_temp_dir=True
                )
                assert module is not None
                assert module.VALUE == "no_cache"

    def test_secure_load_module_invalid_spec(self):
        """Test secure_load_module with invalid spec."""
        invalid_path = Path("/nonexistent/file.py")
        with pytest.raises(ImportError):
            secure_load_module("invalid", invalid_path)

    def test_secure_load_module_from_path(self):
        """Test secure_load_module_from_path function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_module_path = Path(temp_dir) / "from_path.py"
            test_module_path.write_text("FROM_PATH = 'success'")

            module = secure_load_module_from_path(test_module_path, allow_temp_dir=True)
            assert module is not None
            assert module.FROM_PATH == "success"

    def test_get_set_config_value(self):
        """Test config value getting and setting."""
        # Test setting and getting
        set_config_value("test_key", "test_value")
        assert get_config_value("test_key") == "test_value"

        # Test default value
        assert get_config_value("nonexistent", "default") == "default"

    def test_verbose_print_enabled(self):
        """Test verbose_print when enabled."""
        with patch("main.console") as mock_console:
            with patch("main.VERBOSE", True):
                verbose_print("Test message", "info")
                mock_console.print.assert_called_once()

    def test_verbose_print_disabled(self):
        """Test verbose_print when disabled."""
        with patch("main.console") as mock_console:
            with patch("main.VERBOSE", False):
                verbose_print("Test message", "info")
                mock_console.print.assert_not_called()

    def test_quiet_print_normal(self):
        """Test quiet_print in normal mode."""
        with patch("main.console") as mock_console:
            with patch("main.QUIET", False):
                quiet_print("Test message")
                mock_console.print.assert_called_once()

    def test_quiet_print_quiet_mode(self):
        """Test quiet_print in quiet mode."""
        with patch("main.console") as mock_console:
            with patch("main.QUIET", True):
                quiet_print("Test message")
                mock_console.print.assert_not_called()

    def test_quiet_print_forced(self):
        """Test quiet_print with force flag."""
        with patch("main.console") as mock_console:
            with patch("main.QUIET", True):
                quiet_print("Test message", force=True)
                mock_console.print.assert_called_once()

    def test_handle_file_error_permission(self):
        """Test handle_file_error with permission error."""
        error = PermissionError("Permission denied")
        with patch("main.quiet_print") as mock_print:
            handle_file_error("/test/file", "reading", error)
            mock_print.assert_called_once()

    def test_handle_file_error_not_found(self):
        """Test handle_file_error with file not found."""
        error = FileNotFoundError("No such file")
        with patch("main.quiet_print") as mock_print:
            handle_file_error("/missing/file", "opening", error)
            mock_print.assert_called_once()

    def test_handle_file_error_generic(self):
        """Test handle_file_error with generic error."""
        error = Exception("Generic error")
        with patch("main.quiet_print") as mock_print:
            handle_file_error("/test/file", "processing", error)
            mock_print.assert_called_once()

    def test_handle_validation_error_with_context(self):
        """Test handle_validation_error with context."""
        error = ValueError("Invalid value")
        with patch("main.verbose_print") as mock_verbose:
            with patch("main.quiet_print") as mock_quiet:
                handle_validation_error(error, "test context")
                mock_verbose.assert_called_once()
                mock_quiet.assert_called_once()

    def test_handle_validation_error_no_context(self):
        """Test handle_validation_error without context."""
        error = ValueError("Invalid value")
        with patch("main.verbose_print") as mock_verbose:
            with patch("main.quiet_print") as mock_quiet:
                handle_validation_error(error)
                mock_verbose.assert_not_called()
                mock_quiet.assert_called_once()

    def test_sanitize_error_message(self):
        """Test error message sanitization."""
        sensitive_msg = "Error: password=secret123, token=abc123"
        sanitized = _sanitize_error_message(sensitive_msg)

        assert "password" not in sanitized.lower()
        assert "secret123" not in sanitized
        assert "abc123" not in sanitized

    def test_sanitize_error_message_empty(self):
        """Test sanitizing empty message."""
        result = _sanitize_error_message("")
        assert result == ""

    def test_validate_file_path_valid(self):
        """Test _validate_file_path with valid path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.touch()

            result = _validate_file_path(str(test_file), allowed_dirs=[temp_dir])
            assert result == test_file

    def test_validate_file_path_traversal(self):
        """Test _validate_file_path rejects traversal."""
        with pytest.raises(Exception):
            _validate_file_path("../../../etc/passwd")

    def test_validate_output_file_path(self):
        """Test _validate_output_file_path function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = f"{temp_dir}/output.txt"
            result = _validate_output_file_path(output_file)
            assert isinstance(result, Path)

    def test_validate_output_path(self):
        """Test _validate_output_path function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = f"{temp_dir}/test.txt"
            result = _validate_output_path(output_path, allowed_dirs=[temp_dir])
            assert isinstance(result, Path)

    def test_validate_input_file_none(self):
        """Test _validate_input_file with None."""
        result = _validate_input_file(None)
        assert result is None

    def test_validate_input_file_empty(self):
        """Test _validate_input_file with empty string."""
        result = _validate_input_file("")
        assert result is None

    def test_validate_input_file_valid(self):
        """Test _validate_input_file with valid file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "input.txt"
            test_file.write_text("content")

            result = _validate_input_file(str(test_file))
            assert result == str(test_file)

    def test_check_file_size_small(self):
        """Test _check_file_size with small file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"small content")
            temp_path = f.name

        try:
            _check_file_size(temp_path, max_mb=10)
        finally:
            os.unlink(temp_path)

    def test_check_file_size_large(self):
        """Test _check_file_size with large file."""
        with tempfile.NamedTemporaryFile() as f:
            with patch("os.path.getsize", return_value=100 * 1024 * 1024):
                with patch("main.console"):
                    with pytest.raises(Exception):
                        _check_file_size(f.name, max_mb=10)

    def test_secure_open_file_read(self):
        """Test secure_open_file for reading."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            with secure_open_file(temp_path, mode="r") as f:
                content = f.read()
                assert content == "test content"
        finally:
            os.unlink(temp_path)

    def test_secure_open_file_write(self):
        """Test secure_open_file for writing."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            with secure_open_file(temp_path, mode="w") as f:
                f.write("new content")

            with open(temp_path, "r") as f:
                assert f.read() == "new content"
        finally:
            os.unlink(temp_path)

    def test_process_csv_file(self):
        """Test _process_csv_file function."""
        csv_content = "col1,col2\n1,2\n3,4\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            csv_path = f.name

        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_f:
                output_path = out_f.name

            try:
                _process_csv_file(csv_path, output_path)
                assert os.path.exists(output_path)
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)
        finally:
            os.unlink(csv_path)

    def test_run_demo_mode(self):
        """Test _run_demo_mode function."""
        with patch("main.console"):
            with patch("pandas.DataFrame") as mock_df:
                mock_instance = MagicMock()
                mock_df.return_value = mock_instance
                mock_instance.head.return_value = "data"

                _run_demo_mode()
                # Should complete without error

    def test_create_signal_handler(self):
        """Test _create_signal_handler function."""
        cancel_flag = threading.Event()
        handler = _create_signal_handler(cancel_flag)

        assert callable(handler)

        # Test the handler
        handler(2, None)
        assert cancel_flag.is_set()


class TestAPGIModuleLoaderMissingCoverage:
    """Test APGIModuleLoader missing coverage."""

    def test_module_loader_init(self):
        """Test APGIModuleLoader initialization."""
        loader = APGIModuleLoader()
        assert loader.modules == {}
        assert "formal_model" in loader._module_configs

    def test_load_module_missing_class(self):
        """Test _load_module when expected class is missing."""
        loader = APGIModuleLoader()

        with tempfile.TemporaryDirectory() as temp_dir:
            test_module_path = Path(temp_dir) / "incomplete.py"
            test_module_path.write_text("# Missing expected class")

            loader._module_configs["test"] = {
                "path": test_module_path,
                "class": "NonExistentClass",
            }

            result = loader._load_module("test")
            assert result is not None
            assert result["module"] is None
            assert "error" in result

    def test_get_module_failed_load(self):
        """Test get_module with failed load."""
        loader = APGIModuleLoader()

        loader.modules["failed"] = {
            "module": None,
            "config": {"path": "/nonexistent"},
            "error": "Load failed",
        }

        result = loader.get_module("failed")
        assert result is not None
        assert result["module"] is None
        assert result["error"] == "Load failed"

    def test_get_module_nonexistent(self):
        """Test get_module with nonexistent module."""
        loader = APGIModuleLoader()
        result = loader.get_module("nonexistent")
        assert result is None


class TestCLIMissingCoverage:
    """Test CLI commands with missing coverage."""

    def test_config_group_help(self):
        """Test config-group help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["config-group", "--help"])
        assert result.exit_code == 0
        assert "Manage APGI configuration" in result.output

    def test_explain_config(self):
        """Test explain-config command."""
        runner = CliRunner()
        with patch("main.console"):
            result = runner.invoke(cli, ["explain-config"])
            assert result.exit_code == 0

    def test_formal_model_with_params(self):
        """Test formal-model command with parameters."""
        runner = CliRunner()

        with patch("main._run_formal_model_simulation") as mock_sim:
            mock_sim.return_value = {"results": "test"}

            runner.invoke(
                cli,
                ["formal-model", "--simulation-steps", "10", "--dt", "0.1", "--plot"],
            )

            mock_sim.assert_called_once()

    def test_secure_protocol_no_token(self):
        """Test run-secure-protocol without token."""
        runner = CliRunner()

        with patch("main.enforce_security_audit"):
            result = runner.invoke(cli, ["run-secure-protocol"])
            assert result.exit_code != 0


class TestEdgeCases:
    """Test edge cases for missing coverage."""

    def test_validate_file_path_nonexistent(self):
        """Test _validate_file_path with nonexistent file."""
        with pytest.raises(Exception):
            _validate_file_path("/nonexistent/file.txt")

    def test_handle_file_error_none(self):
        """Test handle_file_error with None error."""
        with patch("main.quiet_print") as mock_print:
            handle_file_error("/test/file", "reading", None)
            mock_print.assert_called_once()

    def test_secure_load_module_invalid_path(self):
        """Test secure_load_module with invalid path."""
        invalid_path = Path("/etc/passwd")
        with pytest.raises(ValueError):
            secure_load_module("invalid", invalid_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
