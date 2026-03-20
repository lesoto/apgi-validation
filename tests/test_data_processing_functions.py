"""
Tests for untested data processing & protocol execution functions in main.py
=========================================================================
Comprehensive tests for data processing and protocol execution functions.
"""

from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    _process_csv_file,
    _run_demo_mode,
    _validate_input_file,
    _list_protocols,
    _run_parallel,
    _run_sequential,
    _show_config,
    _set_config,
    _reset_config,
)


class TestProcessCsvFile:
    """Test _process_csv_file function."""

    @patch("main.pd.read_csv")
    @patch("builtins.open", new_callable=mock_open)
    def test_process_csv_file_success(self, mock_file, mock_read_csv, tmp_path):
        """Test successful CSV file processing."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("col1,col2\n1,2\n3,4\n")

        output_file = tmp_path / "output.json"
        mock_df = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
        mock_read_csv.return_value = mock_df

        _process_csv_file(str(input_file), str(output_file))

        mock_read_csv.assert_called_once()

    @patch("main.pd.read_csv")
    def test_process_csv_file_error(self, mock_read_csv, tmp_path):
        """Test CSV file processing with error."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("invalid,csv\n")

        output_file = tmp_path / "output.json"
        mock_read_csv.side_effect = pd.errors.ParserError("Invalid CSV")

        _process_csv_file(str(input_file), str(output_file))

        # Should handle error gracefully


class TestRunDemoMode:
    """Test _run_demo_mode function."""

    @patch("main.console")
    def test_run_demo_mode(self, mock_console):
        """Test running demo mode."""
        _run_demo_mode()

        # Should execute without errors
        assert mock_console.print.called


class TestValidateInputFile:
    """Test _validate_input_file function."""

    def test_validate_input_file_none(self):
        """Test validating None input file."""
        result = _validate_input_file(None)
        assert result is None

    def test_validate_input_file_valid(self, tmp_path):
        """Test validating valid input file."""
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\n1,2\n")

        result = _validate_input_file(str(test_file))
        assert result == str(test_file)

    def test_validate_input_file_nonexistent(self, tmp_path):
        """Test validating non-existent input file."""
        test_file = tmp_path / "nonexistent.csv"

        result = _validate_input_file(str(test_file))
        assert result is None


class TestListProtocols:
    """Test _list_protocols function."""

    def test_list_protocols_with_files(self, tmp_path):
        """Test listing protocols with protocol files."""
        validation_dir = tmp_path / "Validation"
        validation_dir.mkdir()

        (validation_dir / "Validation-Protocol-1.py").write_text("# Protocol 1")
        (validation_dir / "Validation-Protocol-2.py").write_text("# Protocol 2")

        protocols = _list_protocols(validation_dir)

        assert len(protocols) == 2
        assert "Validation-Protocol-1.py" in protocols
        assert "Validation-Protocol-2.py" in protocols

    def test_list_protocols_empty_dir(self, tmp_path):
        """Test listing protocols with empty directory."""
        validation_dir = tmp_path / "Validation"
        validation_dir.mkdir()

        protocols = _list_protocols(validation_dir)

        assert len(protocols) == 0


class TestRunParallel:
    """Test _run_parallel function."""

    @patch("main.ThreadPoolExecutor")
    @patch("main._run_single_protocol")
    def test_run_parallel_success(self, mock_run_single, mock_executor, tmp_path):
        """Test running protocols in parallel successfully."""
        validation_dir = tmp_path / "Validation"
        validation_dir.mkdir()

        (validation_dir / "Validation-Protocol-1.py").write_text("# Protocol 1")
        (validation_dir / "Validation-Protocol-2.py").write_text("# Protocol 2")

        protocols = ["Validation-Protocol-1.py", "Validation-Protocol-2.py"]

        mock_run_single.return_value = ("1", "Success", None)
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        mock_future = MagicMock()
        mock_future.result.return_value = ("1", "Success", None)
        mock_executor_instance.submit.return_value = mock_future

        results = _run_parallel(protocols, validation_dir)

        assert "1" in results

    @patch("main.ThreadPoolExecutor")
    @patch("main._run_single_protocol")
    def test_run_parallel_with_errors(self, mock_run_single, mock_executor, tmp_path):
        """Test running protocols in parallel with errors."""
        validation_dir = tmp_path / "Validation"
        validation_dir.mkdir()

        (validation_dir / "Validation-Protocol-1.py").write_text("# Protocol 1")

        protocols = ["Validation-Protocol-1.py"]

        mock_run_single.return_value = ("1", None, "Error")
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance

        mock_future = MagicMock()
        mock_future.result.return_value = ("1", None, "Error")
        mock_executor_instance.submit.return_value = mock_future

        results = _run_parallel(protocols, validation_dir)

        assert "1" in results


class TestRunSequential:
    """Test _run_sequential function."""

    @patch("main._run_single_protocol")
    def test_run_sequential_success(self, mock_run_single, tmp_path):
        """Test running protocols sequentially successfully."""
        validation_dir = tmp_path / "Validation"
        validation_dir.mkdir()

        (validation_dir / "Validation-Protocol-1.py").write_text("# Protocol 1")
        (validation_dir / "Validation-Protocol-2.py").write_text("# Protocol 2")

        protocols = ["Validation-Protocol-1.py", "Validation-Protocol-2.py"]

        mock_run_single.return_value = ("1", "Success", None)

        results = _run_sequential(protocols, validation_dir)

        assert "1" in results

    @patch("main._run_single_protocol")
    def test_run_sequential_with_errors(self, mock_run_single, tmp_path):
        """Test running protocols sequentially with errors."""
        validation_dir = tmp_path / "Validation"
        validation_dir.mkdir()

        (validation_dir / "Validation-Protocol-1.py").write_text("# Protocol 1")

        protocols = ["Validation-Protocol-1.py"]

        mock_run_single.return_value = ("1", None, "Error")

        results = _run_sequential(protocols, validation_dir)

        assert "1" in results


class TestShowConfig:
    """Test _show_config function."""

    @patch("main.console")
    @patch("main.get_config_value")
    def test_show_config(self, mock_get_config, mock_console):
        """Test displaying configuration."""
        mock_get_config.side_effect = lambda key: {
            "version": "1.0.0",
            "project_name": "Test Project",
            "description": "Test Description",
        }.get(key, "")

        _show_config()

        assert mock_console.print.called


class TestSetConfig:
    """Test _set_config function."""

    @patch("main.set_config_value")
    @patch("main.console")
    def test_set_config_simple(self, mock_console, mock_set_config):
        """Test setting simple configuration value."""
        _set_config("test_key", "test_value")

        mock_set_config.assert_called_once()

    @patch("main.set_config_value")
    @patch("main.console")
    def test_set_config_dict_value(self, mock_console, mock_set_config):
        """Test setting dictionary configuration value."""
        _set_config("test_key", "key1=value1,key2=value2")

        mock_set_config.assert_called_once()


class TestResetConfig:
    """Test _reset_config function."""

    @patch("main.set_config_value")
    @patch("main.console")
    @patch("main.get_config_value")
    def test_reset_config(self, mock_get_config, mock_console, mock_set_config):
        """Test resetting configuration to defaults."""
        mock_get_config.side_effect = lambda key: {
            "version": "1.0.0",
            "project_name": "Test Project",
            "description": "Test Description",
        }.get(key, "")

        _reset_config()

        assert mock_console.print.called
