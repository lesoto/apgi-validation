"""
Tests for untested CLI commands in main.py.
=========================================
Comprehensive tests for 18 previously untested CLI commands.
"""

from click.testing import CliRunner
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import cli


class TestValidateCommand:
    """Test validate CLI command."""

    def test_validate_lists_protocols(self):
        """Test validate command lists available protocols."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate"])
        assert result.exit_code == 0
        assert "Validation Protocols" in result.output

    def test_validate_with_protocol(self):
        """Test validate command with specific protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--protocol", "1"])
        assert result.exit_code == 0

    def test_validate_all_protocols(self):
        """Test validate command with all-protocols flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--all-protocols"])
        assert result.exit_code == 0

    def test_validate_with_output_dir(self, tmp_path):
        """Test validate command with output directory."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["validate", "--protocol", "1", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0

    def test_validate_parallel_mode(self):
        """Test validate command in parallel mode."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--all-protocols", "--parallel"])
        assert result.exit_code == 0

    def test_validate_invalid_protocol(self):
        """Test validate command with invalid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--protocol", "999"])
        assert result.exit_code == 0
        assert "not found" in result.output


class TestFalsifyCommand:
    """Test falsify CLI command."""

    def test_falsify_lists_protocols(self):
        """Test falsify command lists available protocols."""
        runner = CliRunner()
        result = runner.invoke(cli, ["falsify"])
        assert result.exit_code == 0
        assert "Available Falsification Protocols" in result.output

    def test_falsify_with_protocol(self):
        """Test falsify command with specific protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["falsify", "--protocol", "1"])
        assert result.exit_code == 0

    def test_falsify_with_output_file(self, tmp_path):
        """Test falsify command with output file."""
        output_file = tmp_path / "falsification_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["falsify", "--protocol", "1", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0

    def test_falsify_invalid_protocol(self):
        """Test falsify command with invalid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["falsify", "--protocol", "999"])
        assert result.exit_code == 0
        assert "not found" in result.output


class TestEstimateParamsCommand:
    """Test estimate_params CLI command."""

    def test_estimate_params_basic(self):
        """Test estimate_params command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["estimate-params", "--method", "mcmc"])
        assert result.exit_code == 0

    def test_estimate_params_with_data_file(self, tmp_path):
        """Test estimate_params command with data file."""
        data_file = tmp_path / "test_data.json"
        data_file.write_text('{"data": [1, 2, 3]}')
        runner = CliRunner()
        result = runner.invoke(
            cli, ["estimate-params", "--data-file", str(data_file), "--method", "mle"]
        )
        assert result.exit_code == 0

    def test_estimate_params_with_iterations(self):
        """Test estimate_params command with custom iterations."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["estimate-params", "--method", "mcmc", "--iterations", "500"]
        )
        assert result.exit_code == 0

    def test_estimate_params_with_output_file(self, tmp_path):
        """Test estimate_params command with output file."""
        output_file = tmp_path / "params.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["estimate-params", "--method", "mle", "--output-file", str(output_file)],
        )
        assert result.exit_code == 0


class TestCrossSpeciesCommand:
    """Test cross_species CLI command."""

    def test_cross_species_basic(self):
        """Test cross_species command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["cross-species"])
        assert result.exit_code == 0

    def test_cross_species_with_species(self):
        """Test cross_species command with species parameter."""
        runner = CliRunner()
        result = runner.invoke(cli, ["cross-species", "--species", "human"])
        assert result.exit_code == 0

    def test_cross_species_with_output_file(self, tmp_path):
        """Test cross_species command with output file."""
        output_file = tmp_path / "cross_species.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["cross-species", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0

    def test_cross_species_with_plot(self):
        """Test cross_species command with plot flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["cross-species", "--plot"])
        assert result.exit_code == 0


class TestAnalyzeLogsCommand:
    """Test analyze_logs CLI command."""

    def test_analyze_logs_basic(self):
        """Test analyze_logs command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze-logs"])
        assert result.exit_code == 0

    def test_analyze_logs_with_file(self, tmp_path):
        """Test analyze_logs command with log file."""
        log_file = tmp_path / "test.log"
        log_file.write_text("INFO: Test log entry\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze-logs", "--log-file", str(log_file)])
        assert result.exit_code == 0

    def test_analyze_logs_with_level(self):
        """Test analyze_logs command with level filter."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze-logs", "--level", "ERROR"])
        assert result.exit_code == 0

    def test_analyze_logs_with_last_hours(self):
        """Test analyze_logs command with time filter."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze-logs", "--last-hours", "24"])
        assert result.exit_code == 0

    def test_analyze_logs_with_output_file(self, tmp_path):
        """Test analyze_logs command with output file."""
        output_file = tmp_path / "log_analysis.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze-logs", "--output-file", str(output_file)])
        assert result.exit_code == 0


class TestProcessDataCommand:
    """Test process_data CLI command."""

    def test_process_data_basic(self):
        """Test process_data command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process-data"])
        assert result.exit_code == 0

    def test_process_data_with_input_file(self, tmp_path):
        """Test process_data command with input file."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("col1,col2\n1,2\n3,4\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["process-data", "--input-file", str(input_file)])
        assert result.exit_code == 0

    def test_process_data_with_output_dir(self, tmp_path):
        """Test process_data command with output directory."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process-data", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0

    def test_process_data_with_config_file(self, tmp_path):
        """Test process_data command with config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("preprocessing:\n  normalize: true\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["process-data", "--config-file", str(config_file)])
        assert result.exit_code == 0


class TestMonitorPerformanceCommand:
    """Test monitor_performance CLI command."""

    def test_monitor_performance_basic(self):
        """Test monitor_performance command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance"])
        assert result.exit_code == 0

    def test_monitor_performance_with_command(self):
        """Test monitor_performance command with specific command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance", "--command", "validate"])
        assert result.exit_code == 0

    def test_monitor_performance_with_iterations(self):
        """Test monitor_performance command with iteration count."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance", "--iterations", "10"])
        assert result.exit_code == 0

    def test_monitor_performance_with_memory(self):
        """Test monitor_performance command with memory flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance", "--memory"])
        assert result.exit_code == 0

    def test_monitor_performance_with_cpu(self):
        """Test monitor_performance command with CPU flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance", "--cpu"])
        assert result.exit_code == 0


class TestNeuralSignaturesCommand:
    """Test neural_signatures CLI command."""

    def test_neural_signatures_basic(self):
        """Test neural_signatures command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["neural-signatures"])
        assert result.exit_code == 0

    def test_neural_signatures_with_priority(self):
        """Test neural_signatures command with priority level."""
        runner = CliRunner()
        result = runner.invoke(cli, ["neural-signatures", "--priority", "1"])
        assert result.exit_code == 0

    def test_neural_signatures_with_eeg_data(self, tmp_path):
        """Test neural_signatures command with EEG data."""
        eeg_file = tmp_path / "eeg_data.json"
        eeg_file.write_text('{"eeg": [1, 2, 3]}')
        runner = CliRunner()
        result = runner.invoke(cli, ["neural-signatures", "--eeg-data", str(eeg_file)])
        assert result.exit_code == 0

    def test_neural_signatures_with_fmri_data(self, tmp_path):
        """Test neural_signatures command with fMRI data."""
        fmri_file = tmp_path / "fmri_data.json"
        fmri_file.write_text('{"fmri": [1, 2, 3]}')
        runner = CliRunner()
        result = runner.invoke(
            cli, ["neural-signatures", "--fmri-data", str(fmri_file)]
        )
        assert result.exit_code == 0

    def test_neural_signatures_with_output_file(self, tmp_path):
        """Test neural_signatures command with output file."""
        output_file = tmp_path / "neural_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["neural-signatures", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0


class TestCausalManipulationsCommand:
    """Test causal_manipulations CLI command."""

    def test_causal_manipulations_basic(self):
        """Test causal_manipulations command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["causal-manipulations"])
        assert result.exit_code == 0

    def test_causal_manipulations_with_intervention(self):
        """Test causal_manipulations command with intervention type."""
        runner = CliRunner()
        result = runner.invoke(cli, ["causal-manipulations", "--intervention", "tms"])
        assert result.exit_code == 0

    def test_causal_manipulations_with_target(self):
        """Test causal_manipulations command with target."""
        runner = CliRunner()
        result = runner.invoke(cli, ["causal-manipulations", "--target", "prefrontal"])
        assert result.exit_code == 0

    def test_causal_manipulations_with_output_file(self, tmp_path):
        """Test causal_manipulations command with output file."""
        output_file = tmp_path / "causal_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["causal-manipulations", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0


class TestQuantitativeFitsCommand:
    """Test quantitative_fits CLI command."""

    def test_quantitative_fits_basic(self):
        """Test quantitative_fits command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["quantitative-fits"])
        assert result.exit_code == 0

    def test_quantitative_fits_with_model(self):
        """Test quantitative_fits command with model parameter."""
        runner = CliRunner()
        result = runner.invoke(cli, ["quantitative-fits", "--model", "apgi"])
        assert result.exit_code == 0

    def test_quantitative_fits_with_data_file(self, tmp_path):
        """Test quantitative_fits command with data file."""
        data_file = tmp_path / "fit_data.json"
        data_file.write_text('{"data": [1, 2, 3]}')
        runner = CliRunner()
        result = runner.invoke(
            cli, ["quantitative-fits", "--data-file", str(data_file)]
        )
        assert result.exit_code == 0

    def test_quantitative_fits_with_output_file(self, tmp_path):
        """Test quantitative_fits command with output file."""
        output_file = tmp_path / "fit_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["quantitative-fits", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0


class TestClinicalConvergenceCommand:
    """Test clinical_convergence CLI command."""

    def test_clinical_convergence_basic(self):
        """Test clinical_convergence command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["clinical-convergence"])
        assert result.exit_code == 0

    def test_clinical_convergence_with_population(self):
        """Test clinical_convergence command with population."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["clinical-convergence", "--population", "patients"]
        )
        assert result.exit_code == 0

    def test_clinical_convergence_with_condition(self):
        """Test clinical_convergence command with condition."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["clinical-convergence", "--condition", "schizophrenia"]
        )
        assert result.exit_code == 0

    def test_clinical_convergence_with_output_file(self, tmp_path):
        """Test clinical_convergence command with output file."""
        output_file = tmp_path / "clinical_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["clinical-convergence", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0


class TestOpenScienceCommand:
    """Test open_science CLI command."""

    def test_open_science_basic(self):
        """Test open_science command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["open-science"])
        assert result.exit_code == 0

    def test_open_science_with_component(self):
        """Test open_science command with component."""
        runner = CliRunner()
        result = runner.invoke(cli, ["open-science", "--component", "validation"])
        assert result.exit_code == 0

    def test_open_science_with_action(self):
        """Test open_science command with action."""
        runner = CliRunner()
        result = runner.invoke(cli, ["open-science", "--action", "preregister"])
        assert result.exit_code == 0

    def test_open_science_with_repository(self):
        """Test open_science command with repository URL."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["open-science", "--repository", "https://github.com/test/repo"]
        )
        assert result.exit_code == 0


class TestBayesianEstimationCommand:
    """Test bayesian_estimation CLI command."""

    def test_bayesian_estimation_basic(self):
        """Test bayesian_estimation command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["bayesian-estimation"])
        assert result.exit_code == 0

    def test_bayesian_estimation_with_method(self):
        """Test bayesian_estimation command with method."""
        runner = CliRunner()
        result = runner.invoke(cli, ["bayesian-estimation", "--method", "mcmc"])
        assert result.exit_code == 0

    def test_bayesian_estimation_with_data_file(self, tmp_path):
        """Test bayesian_estimation command with data file."""
        data_file = tmp_path / "bayesian_data.json"
        data_file.write_text('{"data": [1, 2, 3]}')
        runner = CliRunner()
        result = runner.invoke(
            cli, ["bayesian-estimation", "--data-file", str(data_file)]
        )
        assert result.exit_code == 0

    def test_bayesian_estimation_with_output_file(self, tmp_path):
        """Test bayesian_estimation command with output file."""
        output_file = tmp_path / "bayesian_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["bayesian-estimation", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0


class TestComprehensiveValidationCommand:
    """Test comprehensive_validation CLI command."""

    def test_comprehensive_validation_basic(self):
        """Test comprehensive_validation command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["comprehensive-validation"])
        assert result.exit_code == 0

    def test_comprehensive_validation_with_comprehensive_flag(self):
        """Test comprehensive_validation command with comprehensive flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["comprehensive-validation", "--comprehensive"])
        assert result.exit_code == 0

    def test_comprehensive_validation_with_output_file(self, tmp_path):
        """Test comprehensive_validation command with output file."""
        output_file = tmp_path / "comprehensive_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["comprehensive-validation", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0

    def test_comprehensive_validation_with_parallel(self):
        """Test comprehensive_validation command with parallel flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["comprehensive-validation", "--parallel"])
        assert result.exit_code == 0


class TestGUICommand:
    """Test gui CLI command."""

    def test_gui_basic(self):
        """Test gui command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["gui"])
        assert result.exit_code == 0

    def test_gui_with_type(self):
        """Test gui command with GUI type."""
        runner = CliRunner()
        result = runner.invoke(cli, ["gui", "--gui-type", "web"])
        assert result.exit_code == 0

    def test_gui_with_port(self):
        """Test gui command with custom port."""
        runner = CliRunner()
        result = runner.invoke(cli, ["gui", "--port", "8080"])
        assert result.exit_code == 0

    def test_gui_with_host(self):
        """Test gui command with custom host."""
        runner = CliRunner()
        result = runner.invoke(cli, ["gui", "--host", "0.0.0.0"])
        assert result.exit_code == 0

    def test_gui_with_debug(self):
        """Test gui command with debug flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["gui", "--debug"])
        assert result.exit_code == 0


class TestLogsCommand:
    """Test logs CLI command."""

    def test_logs_basic(self):
        """Test logs command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["logs"])
        assert result.exit_code == 0

    def test_logs_with_tail(self):
        """Test logs command with tail count."""
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--tail", "10"])
        assert result.exit_code == 0

    def test_logs_with_follow(self):
        """Test logs command with follow flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--follow"])
        assert result.exit_code == 0

    def test_logs_with_level(self):
        """Test logs command with level filter."""
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--level", "ERROR"])
        assert result.exit_code == 0

    def test_logs_with_export(self, tmp_path):
        """Test logs command with export."""
        export_file = tmp_path / "logs_export.txt"
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--export", str(export_file)])
        assert result.exit_code == 0


class TestPerformanceCommand:
    """Test performance CLI command."""

    def test_performance_basic(self):
        """Test performance command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["performance"])
        assert result.exit_code == 0

    def test_performance_detailed(self):
        """Test performance command with detailed flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["performance", "--detailed"])
        assert result.exit_code == 0


class TestMultimodalCommand:
    """Test multimodal CLI command."""

    def test_multimodal_basic(self):
        """Test multimodal command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["multimodal"])
        assert result.exit_code == 0

    def test_multimodal_with_input_data(self, tmp_path):
        """Test multimodal command with input data."""
        input_file = tmp_path / "multimodal_data.json"
        input_file.write_text('{"eeg": [1, 2], "fmri": [3, 4]}')
        runner = CliRunner()
        result = runner.invoke(cli, ["multimodal", "--input-data", str(input_file)])
        assert result.exit_code == 0

    def test_multimodal_with_output_file(self, tmp_path):
        """Test multimodal command with output file."""
        output_file = tmp_path / "multimodal_results.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["multimodal", "--output-file", str(output_file)])
        assert result.exit_code == 0

    def test_multimodal_with_modalities(self):
        """Test multimodal command with modalities."""
        runner = CliRunner()
        result = runner.invoke(cli, ["multimodal", "--modalities", "eeg,fmri"])
        assert result.exit_code == 0
