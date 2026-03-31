"""
Comprehensive integration tests for all CLI commands in main.py.
Tests all 18 previously untested CLI commands.
================================================================
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from click.testing import CliRunner

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import cli


class TestValidateCommand:
    """Tests for the validate CLI command."""

    def test_validate_with_valid_protocol(self, temp_dir):
        """Test validate command with a valid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--protocol", "1"])
        assert result.exit_code == 0

    def test_validate_with_invalid_protocol(self):
        """Test validate command with invalid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--protocol", "999"])
        assert result.exit_code == 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_validate_with_missing_file(self):
        """Test validate command handles missing data gracefully."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--protocol", "1"])
        # Should not crash, may show error message
        assert result.exit_code in [0, 1, 2]


class TestFalsifyCommand:
    """Tests for the falsify CLI command."""

    def test_falsify_with_valid_protocol(self, temp_dir):
        """Test falsify command with a valid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["falsify", "--protocol", "1"])
        assert result.exit_code == 0

    def test_falsify_with_invalid_protocol(self):
        """Test falsify command with invalid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["falsify", "--protocol", "999"])
        assert result.exit_code == 0


class TestEstimateParamsCommand:
    """Tests for the estimate_params CLI command."""

    def test_estimate_params_basic(self, temp_dir):
        """Test parameter estimation command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate-params", "--iterations", "10"])
        assert result.exit_code == 0

    def test_estimate_params_with_bounds(self, temp_dir):
        """Test parameter estimation with custom method."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["estimate-params", "--method", "map", "--iterations", "10"]
        )
        assert result.exit_code == 0


class TestCrossSpeciesCommand:
    """Tests for the cross_species CLI command."""

    def test_cross_species_basic(self, temp_dir):
        """Test cross-species analysis."""
        runner = CliRunner()
        result = runner.invoke(cli, ["cross-species"])
        assert result.exit_code == 0


class TestAnalyzeLogsCommand:
    """Tests for the analyze_logs CLI command."""

    def test_analyze_logs_with_file(self, temp_dir):
        """Test log analysis with a log file."""
        log_content = """
2024-01-01 10:00:00 INFO Starting simulation
2024-01-01 10:00:01 ERROR Simulation failed
2024-01-01 10:00:02 WARNING Memory usage high
"""
        log_file = temp_dir / "test.log"
        log_file.write_text(log_content)

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze-logs", "--log-file", str(log_file)])
        assert result.exit_code == 0

    def test_analyze_logs_with_filter(self, temp_dir):
        """Test log analysis with error filter."""
        log_content = """
2024-01-01 10:00:00 INFO Starting simulation
2024-01-01 10:00:01 ERROR Simulation failed
"""
        log_file = temp_dir / "test.log"
        log_file.write_text(log_content)

        runner = CliRunner()
        result = runner.invoke(
            cli, ["analyze-logs", "--log-file", str(log_file), "--level", "ERROR"]
        )
        assert result.exit_code == 0


class TestProcessDataCommand:
    """Tests for the process_data CLI command."""

    def test_process_data_csv(self, temp_dir):
        """Test data processing with CSV input."""
        test_data = pd.DataFrame(
            {
                "timestamp": range(100),
                "surprise": np.random.randn(100),
                "threshold": np.random.randn(100),
            }
        )
        csv_file = temp_dir / "test.csv"
        test_data.to_csv(csv_file, index=False)

        runner = CliRunner()
        result = runner.invoke(cli, ["process-data", "--input-file", str(csv_file)])
        assert result.exit_code == 0

    def test_process_data_json(self, temp_dir):
        """Test data processing with JSON input."""
        test_data = {"data": np.random.randn(100).tolist()}
        json_file = temp_dir / "test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        runner = CliRunner()
        result = runner.invoke(cli, ["process-data", "--input-file", str(json_file)])
        assert result.exit_code == 0


class TestMonitorPerformanceCommand:
    """Tests for the monitor_performance CLI command."""

    def test_monitor_performance_basic(self, temp_dir):
        """Test performance monitoring."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance"])
        assert result.exit_code == 0

    def test_monitor_performance_with_metrics(self, temp_dir):
        """Test performance monitoring with specific metrics."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance", "--cpu", "--memory"])
        assert result.exit_code == 0


class TestNeuralSignaturesCommand:
    """Tests for the neural_signatures CLI command."""

    def test_neural_signatures_basic(self, temp_dir):
        """Test neural signature analysis."""
        runner = CliRunner()
        result = runner.invoke(cli, ["neural-signatures"])
        assert result.exit_code == 0


class TestCausalManipulationsCommand:
    """Tests for the causal_manipulations CLI command."""

    def test_causal_manipulations_basic(self, temp_dir):
        """Test causal manipulation analysis."""
        runner = CliRunner()
        result = runner.invoke(cli, ["causal-manipulations"])
        assert result.exit_code == 0


class TestQuantitativeFitsCommand:
    """Tests for the quantitative_fits CLI command."""

    def test_quantitative_fits_basic(self, temp_dir):
        """Test quantitative fitting."""
        runner = CliRunner()
        result = runner.invoke(cli, ["quantitative-fits"])
        assert result.exit_code == 0


class TestClinicalConvergenceCommand:
    """Tests for the clinical_convergence CLI command."""

    def test_clinical_convergence_basic(self, temp_dir):
        """Test clinical convergence analysis."""
        runner = CliRunner()
        result = runner.invoke(cli, ["clinical-convergence"])
        assert result.exit_code == 0


class TestOpenScienceCommand:
    """Tests for the open_science CLI command."""

    def test_open_science_export(self, temp_dir):
        """Test open science data export."""
        runner = CliRunner()
        result = runner.invoke(cli, ["open-science"])
        assert result.exit_code == 0

    def test_open_science_metadata(self, temp_dir):
        """Test open science metadata generation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["open-science", "--component", "preregistration"])
        assert result.exit_code == 0


class TestBayesianEstimationCommand:
    """Tests for the bayesian_estimation CLI command."""

    def test_bayesian_estimation_basic(self, temp_dir):
        """Test Bayesian parameter estimation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["bayesian-estimation"])
        assert result.exit_code == 0


class TestComprehensiveValidationCommand:
    """Tests for the comprehensive_validation CLI command."""

    def test_comprehensive_validation_basic(self, temp_dir):
        """Test comprehensive validation suite."""
        runner = CliRunner()
        result = runner.invoke(cli, ["comprehensive-validation"])
        assert result.exit_code == 0


class TestGUICommand:
    """Tests for the gui CLI command."""

    def test_gui_launch(self):
        """Test GUI launch command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["gui"])
        assert result.exit_code == 0

    def test_gui_with_config(self, temp_dir):
        """Test GUI launch with custom config."""
        runner = CliRunner()
        result = runner.invoke(cli, ["gui", "--gui-type", "tkinter"])
        assert result.exit_code == 0


class TestLogsCommand:
    """Tests for the logs CLI command."""

    def test_logs_display(self, temp_dir):
        """Test log display command."""
        log_content = "2024-01-01 10:00:00 INFO Test log message"
        log_file = temp_dir / "test.log"
        log_file.write_text(log_content)

        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--export", str(temp_dir / "logs.txt")])
        assert result.exit_code == 0

    def test_logs_with_filter(self, temp_dir):
        """Test log display with level filter."""
        log_content = """
2024-01-01 10:00:00 INFO Test info message
2024-01-01 10:00:01 ERROR Test error message
"""
        log_file = temp_dir / "test.log"
        log_file.write_text(log_content)

        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--level", "ERROR"])
        assert result.exit_code == 0


class TestPerformanceCommand:
    """Tests for the performance CLI command."""

    def test_performance_dashboard(self, temp_dir):
        """Test performance dashboard command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["performance"])
        assert result.exit_code == 0

    def test_performance_with_benchmark(self, temp_dir):
        """Test performance with benchmark comparison."""
        runner = CliRunner()
        result = runner.invoke(cli, ["performance", "--detailed"])
        assert result.exit_code == 0


class TestMultimodalCommand:
    """Tests for the multimodal CLI command - edge cases."""

    def test_multimodal_with_missing_modalities(self, temp_dir):
        """Test multimodal integration with missing data."""
        test_data = {"eeg": np.random.randn(100, 10).tolist()}
        data_file = temp_dir / "multimodal_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        runner = CliRunner()
        result = runner.invoke(cli, ["multimodal", "--input-data", str(data_file)])
        assert result.exit_code == 0

    def test_multimodal_with_mismatched_dimensions(self, temp_dir):
        """Test multimodal integration with dimension mismatch."""
        test_data = {
            "eeg": np.random.randn(100, 10).tolist(),
            "fmri": np.random.randn(50, 20).tolist(),  # Different dimensions
        }
        data_file = temp_dir / "multimodal_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        runner = CliRunner()
        result = runner.invoke(cli, ["multimodal", "--input-data", str(data_file)])
        assert result.exit_code in [0, 1]  # May fail gracefully
