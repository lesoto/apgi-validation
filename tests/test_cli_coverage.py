"""
Tests for untested CLI commands in main.py.
=========================================
Comprehensive tests for 18 previously untested CLI commands.
"""

import pytest
from click.testing import CliRunner
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Theory"))
sys.path.insert(0, str(Path(__file__).parent / "Falsification"))


class TestValidateCommand:
    """Test validate CLI command."""

    def test_validate_lists_protocols(self, cli):
        """Test validate command lists available protocols."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate"])
        assert result.exit_code == 0
        assert "Validation Protocols" in result.output

    def test_validate_with_protocol(self, cli):
        """Test validate command with specific protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--protocol", "1"])
        assert result.exit_code == 0

    def test_validate_all_protocols(self, cli):
        """Test validate command with all-protocols flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--all-protocols"])
        assert result.exit_code == 0

    def test_validate_with_output_dir(self, cli, tmp_path):
        """Test validate command with output directory."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["validate", "--protocol", "1", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0

    def test_validate_parallel_mode(self, cli):
        """Test validate command in parallel mode."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--all-protocols", "--parallel"])
        assert result.exit_code == 0

    def test_validate_invalid_protocol(self, cli):
        """Test validate command with invalid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--protocol", "999"])
        assert result.exit_code == 0
        assert "not found" in result.output


class TestFalsifyCommand:
    """Test falsify CLI command."""

    def test_falsify_lists_protocols(self, cli):
        """Test falsify command lists available protocols."""
        runner = CliRunner()
        result = runner.invoke(cli, ["falsify"])
        assert result.exit_code == 0
        assert "Falsification Testing" in result.output

    def test_falsify_with_protocol(self, cli):
        """Test falsify command with specific protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["falsify", "--protocol", "1"])
        assert result.exit_code == 0

    def test_falsify_with_output_file(self, cli, tmp_path):
        """Test falsify command with output file."""
        output_file = tmp_path / "falsification_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["falsify", "--protocol", "1", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0

    def test_falsify_invalid_protocol(self, cli):
        """Test falsify command with invalid protocol."""
        runner = CliRunner()
        result = runner.invoke(cli, ["falsify", "--protocol", "999"])
        assert result.exit_code == 0
        assert "not found" in result.output


class TestEstimateParamsCommand:
    """Test estimate_params CLI command."""

    def test_estimate_params_basic(self, cli):
        """Test estimate_params command with basic parameters - just check CLI parsing."""
        runner = CliRunner()
        # Test with --help to avoid running expensive computation
        result = runner.invoke(cli, ["estimate-params", "--help"])
        assert result.exit_code == 0
        assert "method" in result.output

    def test_estimate_params_with_data_file(self, cli, tmp_path):
        """Test estimate_params command with data file."""
        data_file = tmp_path / "test_data.json"
        data_file.write_text('{"data": [1, 2, 3]}')
        runner = CliRunner()
        result = runner.invoke(
            cli, ["estimate-params", "--data-file", str(data_file), "--method", "mle"]
        )
        assert result.exit_code == 0

    @pytest.mark.timeout(300)  # 5 minutes timeout
    def test_estimate_params_with_iterations(self, cli):
        """Test estimate_params command with custom iterations."""
        from unittest.mock import patch

        with patch("main.module_loader.get_module") as mock_get_module:
            # Mock the module without a main() function to test the CLI path
            mock_module_info = {"module": type("MockModule", (), {})}
            mock_get_module.return_value = mock_module_info

            runner = CliRunner()
            result = runner.invoke(cli, ["estimate-params", "--iterations", "10"])
            assert result.exit_code == 0
            assert "Estimation method" in result.output

    def test_estimate_params_with_output_file(self, cli, tmp_path):
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

    def test_cross_species_basic(self, cli):
        """Test cross_species command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["cross-species"])
        assert result.exit_code == 0

    def test_cross_species_with_species(self, cli):
        """Test cross_species command with species parameter."""
        runner = CliRunner()
        result = runner.invoke(cli, ["cross-species", "--species", "human"])
        assert result.exit_code == 0

    def test_cross_species_with_output_file(self, cli, tmp_path):
        """Test cross_species command with output file."""
        output_file = tmp_path / "cross_species.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["cross-species", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0

    def test_cross_species_with_plot(self, cli):
        """Test cross_species command with plot flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["cross-species", "--plot"])
        assert result.exit_code == 0


class TestAnalyzeLogsCommand:
    """Test analyze_logs CLI command."""

    def test_analyze_logs_basic(self, cli):
        """Test analyze_logs command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze-logs"])
        assert result.exit_code == 0

    def test_analyze_logs_with_file(self, cli, tmp_path):
        """Test analyze_logs command with log file."""
        log_file = tmp_path / "test.log"
        log_file.write_text("INFO: Test log entry\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze-logs", "--log-file", str(log_file)])
        assert result.exit_code == 0

    def test_analyze_logs_with_level(self, cli):
        """Test analyze_logs command with level filter."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze-logs", "--level", "ERROR"])
        assert result.exit_code == 0

    def test_analyze_logs_with_last_hours(self, cli):
        """Test analyze_logs command with time filter."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze-logs", "--last-hours", "24"])
        assert result.exit_code == 0

    def test_analyze_logs_with_output_file(self, cli, tmp_path):
        """Test analyze_logs command with output file."""
        output_file = tmp_path / "log_analysis.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze-logs", "--output-file", str(output_file)])
        assert result.exit_code == 0


class TestProcessDataCommand:
    """Test process_data CLI command."""

    def test_process_data_basic(self, cli):
        """Test process_data command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process-data"])
        assert result.exit_code == 0

    def test_process_data_with_input_file(self, cli, tmp_path):
        """Test process_data command with input file."""
        input_file = tmp_path / "input.csv"
        input_file.write_text("col1,col2\n1,2\n3,4\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["process-data", "--input-file", str(input_file)])
        assert result.exit_code == 0

    def test_process_data_with_output_dir(self, cli, tmp_path):
        """Test process_data command with output directory."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process-data", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0

    def test_process_data_with_config_file(self, cli, tmp_path):
        """Test process_data command with config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("eeg_bandpass_low: 0.5\neeg_bandpass_high: 40.0\n")
        runner = CliRunner()
        result = runner.invoke(cli, ["process-data", "--config-file", str(config_file)])
        assert result.exit_code == 0


class TestMonitorPerformanceCommand:
    """Test monitor_performance CLI command."""

    def test_monitor_performance_basic(self, cli):
        """Test monitor_performance command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance"])
        assert result.exit_code == 0

    def test_monitor_performance_with_command(self, cli):
        """Test monitor_performance command with specific command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance", "--command", "validate"])
        assert result.exit_code == 0

    def test_monitor_performance_with_iterations(self, cli):
        """Test monitor_performance command with iteration count."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance", "--iterations", "10"])
        assert result.exit_code == 0

    def test_monitor_performance_with_memory(self, cli):
        """Test monitor_performance command with memory flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance", "--memory"])
        assert result.exit_code == 0

    def test_monitor_performance_with_cpu(self, cli):
        """Test monitor_performance command with CPU flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["monitor-performance", "--cpu"])
        assert result.exit_code == 0


class TestNeuralSignaturesCommand:
    """Test neural_signatures CLI command."""

    def test_neural_signatures_basic(self, cli):
        """Test neural_signatures command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["neural-signatures"])
        assert result.exit_code == 0

    def test_neural_signatures_with_priority(self, cli):
        """Test neural_signatures command with priority level."""
        runner = CliRunner()
        result = runner.invoke(cli, ["neural-signatures", "--priority", "1"])
        assert result.exit_code == 0

    def test_neural_signatures_with_eeg_data(self, cli, tmp_path):
        """Test neural_signatures command with EEG data."""
        eeg_file = tmp_path / "eeg_data.json"
        eeg_file.write_text('{"eeg": [1, 2, 3]}')
        runner = CliRunner()
        result = runner.invoke(
            cli, ["neural-signatures", "--neural-data", str(eeg_file)]
        )
        assert result.exit_code == 0

    def test_neural_signatures_with_fmri_data(self, cli, tmp_path):
        """Test neural_signatures command with fMRI data."""
        fmri_file = tmp_path / "fmri_data.json"
        fmri_file.write_text('{"fmri": [1, 2, 3]}')
        runner = CliRunner()
        result = runner.invoke(
            cli, ["neural-signatures", "--fmri-data", str(fmri_file)]
        )
        assert result.exit_code == 0

    def test_neural_signatures_with_output_file(self, cli, tmp_path):
        """Test neural_signatures command with output file."""
        output_file = tmp_path / "neural_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["neural-signatures", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0


class TestCausalManipulationsCommand:
    """Test causal_manipulations CLI command."""

    def test_causal_manipulations_basic(self, cli):
        """Test causal_manipulations command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["causal-manipulations"])
        assert result.exit_code == 0

    def test_causal_manipulations_with_intervention(self, cli):
        """Test causal_manipulations command with intervention type."""
        runner = CliRunner()
        result = runner.invoke(cli, ["causal-manipulations", "--intervention", "tms"])
        assert result.exit_code == 0

    def test_causal_manipulations_with_target(self, cli):
        """Test causal_manipulations command with target."""
        runner = CliRunner()
        result = runner.invoke(cli, ["causal-manipulations", "--target", "prefrontal"])
        assert result.exit_code == 0

    def test_causal_manipulations_with_output_file(self, cli, tmp_path):
        """Test causal_manipulations command with output file."""
        output_file = tmp_path / "causal_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["causal-manipulations", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0


class TestQuantitativeFitsCommand:
    """Test quantitative_fits CLI command."""

    @staticmethod
    def _make_quant_mocks():
        """Return (mock_module, mock_spec) pair for quantitative-fits patching."""
        from unittest.mock import MagicMock

        mock_module = MagicMock()
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_quantitative_fits.return_value = {
            "overall_quantitative_score": 0.85
        }
        mock_module.QuantitativeModelValidator.return_value = mock_validator_instance
        return mock_module, MagicMock()

    def test_quantitative_fits_basic(self, cli):
        """Test quantitative_fits command with basic parameters."""
        from unittest.mock import patch

        mock_module, mock_spec = self._make_quant_mocks()
        with patch(
            "importlib.util.spec_from_file_location", return_value=mock_spec
        ), patch("importlib.util.module_from_spec", return_value=mock_module):
            runner = CliRunner()
            result = runner.invoke(cli, ["quantitative-fits"])
            assert result.exit_code == 0

    def test_quantitative_fits_with_model(self, cli):
        """Test quantitative_fits command with model parameter."""
        from unittest.mock import patch

        mock_module, mock_spec = self._make_quant_mocks()
        with patch(
            "importlib.util.spec_from_file_location", return_value=mock_spec
        ), patch("importlib.util.module_from_spec", return_value=mock_module):
            runner = CliRunner()
            result = runner.invoke(cli, ["quantitative-fits", "--model", "bayesian"])
            assert result.exit_code == 0

    def test_quantitative_fits_with_data_file(self, cli, tmp_path):
        """Test quantitative_fits command with data file."""
        from unittest.mock import patch

        data_file = tmp_path / "fit_data.json"
        data_file.write_text('{"data": [1, 2, 3]}')
        mock_module, mock_spec = self._make_quant_mocks()
        with patch(
            "importlib.util.spec_from_file_location", return_value=mock_spec
        ), patch("importlib.util.module_from_spec", return_value=mock_module):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["quantitative-fits", "--data-file", str(data_file)]
            )
            assert result.exit_code == 0

    def test_quantitative_fits_with_output_file(self, cli, tmp_path):
        """Test quantitative_fits command with output file."""
        from unittest.mock import patch

        output_file = tmp_path / "fit_results.json"
        mock_module, mock_spec = self._make_quant_mocks()
        with patch(
            "importlib.util.spec_from_file_location", return_value=mock_spec
        ), patch("importlib.util.module_from_spec", return_value=mock_module):
            runner = CliRunner()
            result = runner.invoke(
                cli, ["quantitative-fits", "--output-file", str(output_file)]
            )
            assert result.exit_code == 0


class TestClinicalConvergenceCommand:
    """Test clinical_convergence CLI command."""

    def test_clinical_convergence_basic(self, cli):
        """Test clinical_convergence command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["clinical-convergence"])
        assert result.exit_code == 0

    def test_clinical_convergence_with_population(self, cli):
        """Test clinical_convergence command with population."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["clinical-convergence", "--population", "clinical"]
        )
        assert result.exit_code == 0

    def test_clinical_convergence_with_condition(self, cli):
        """Test clinical_convergence command with condition."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["clinical-convergence", "--condition", "schizophrenia"]
        )
        assert result.exit_code == 0

    def test_clinical_convergence_with_output_file(self, cli, tmp_path):
        """Test clinical_convergence command with output file."""
        output_file = tmp_path / "clinical_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["clinical-convergence", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0


class TestOpenScienceCommand:
    """Test open_science CLI command."""

    def test_open_science_basic(self, cli):
        """Test open_science command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["open-science"])
        assert result.exit_code == 0

    def test_open_science_with_component(self, cli):
        """Test open_science command with component."""
        runner = CliRunner()
        result = runner.invoke(cli, ["open-science", "--component", "preregistration"])
        assert result.exit_code == 0

    def test_open_science_with_action(self, cli):
        """Test open_science command with action."""
        runner = CliRunner()
        result = runner.invoke(cli, ["open-science", "--action", "preregister"])
        assert result.exit_code == 0

    def test_open_science_with_repository(self, cli):
        """Test open_science command with repository URL."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["open-science", "--data-repository", "https://github.com/test/repo"]
        )
        assert result.exit_code == 0


class TestBayesianEstimationCommand:
    """Test bayesian_estimation CLI command."""

    def test_bayesian_estimation_basic(self, cli):
        """Test bayesian_estimation command with basic parameters - just check CLI parsing."""
        runner = CliRunner()
        # Test with --help to avoid running expensive computation
        result = runner.invoke(cli, ["bayesian-estimation", "--help"])
        assert result.exit_code == 0
        assert "method" in result.output

    @pytest.mark.timeout(180)  # 3 minute timeout for MCMC computation
    def test_bayesian_estimation_with_method(self, cli):
        """Test bayesian_estimation command with method parameter."""
        from unittest.mock import patch, MagicMock

        # Mock the BayesianValidationFramework to avoid expensive MCMC
        mock_framework = MagicMock()
        mock_results = {
            "psychometric_estimation": {
                "beta_posterior_mean": 12.5,
                "theta_posterior_mean": 0.52,
                "phase_transition_posterior": True,
                "converged": True,
            },
            "overall_bayesian_score": 0.85,
        }
        mock_framework.comprehensive_bayesian_validation.return_value = mock_results

        with patch("main.importlib.util.spec_from_file_location") as mock_spec, patch(
            "main.importlib.util.module_from_spec"
        ) as mock_module_from_spec:
            mock_spec_instance = MagicMock()
            mock_spec.return_value = mock_spec_instance
            mock_module = MagicMock()
            mock_module.BayesianValidationFramework.return_value = mock_framework
            mock_module_from_spec.return_value = mock_module

            runner = CliRunner()
            result = runner.invoke(cli, ["bayesian-estimation", "--method", "mcmc"])
            assert result.exit_code == 0

    @pytest.mark.timeout(180)  # 3 minute timeout for Bayesian computation
    def test_bayesian_estimation_with_data_file(self, cli, tmp_path):
        """Test bayesian_estimation command with data file."""
        from unittest.mock import patch, MagicMock

        data_file = tmp_path / "test_data.json"
        data_file.write_text('{"data": [1, 2, 3]}')

        # Mock the BayesianValidationFramework to avoid expensive MCMC
        mock_framework = MagicMock()
        mock_results = {
            "psychometric_estimation": {
                "beta_posterior_mean": 12.5,
                "theta_posterior_mean": 0.52,
                "phase_transition_posterior": True,
                "converged": True,
            },
            "overall_bayesian_score": 0.85,
        }
        mock_framework.comprehensive_bayesian_validation.return_value = mock_results

        with patch("main.importlib.util.spec_from_file_location") as mock_spec, patch(
            "main.importlib.util.module_from_spec"
        ) as mock_module_from_spec:
            mock_spec_instance = MagicMock()
            mock_spec.return_value = mock_spec_instance
            mock_module = MagicMock()
            mock_module.BayesianValidationFramework.return_value = mock_framework
            mock_module_from_spec.return_value = mock_module

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "bayesian-estimation",
                    "--data-file",
                    str(data_file),
                    "--method",
                    "mcmc",
                ],
            )
            assert result.exit_code == 0

    @pytest.mark.timeout(180)  # 3 minute timeout for Bayesian computation
    def test_bayesian_estimation_with_output_file(self, cli, tmp_path):
        """Test bayesian_estimation command with output file."""
        from unittest.mock import patch, MagicMock

        output_file = tmp_path / "bayes_results.json"

        # Mock the BayesianValidationFramework to avoid expensive MCMC
        mock_framework = MagicMock()
        mock_results = {
            "psychometric_estimation": {
                "beta_posterior_mean": 12.5,
                "theta_posterior_mean": 0.52,
                "phase_transition_posterior": True,
                "converged": True,
            },
            "overall_bayesian_score": 0.85,
        }
        mock_framework.comprehensive_bayesian_validation.return_value = mock_results

        with patch("main.importlib.util.spec_from_file_location") as mock_spec, patch(
            "main.importlib.util.module_from_spec"
        ) as mock_module_from_spec:
            mock_spec_instance = MagicMock()
            mock_spec.return_value = mock_spec_instance
            mock_module = MagicMock()
            mock_module.BayesianValidationFramework.return_value = mock_framework
            mock_module_from_spec.return_value = mock_module

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "bayesian-estimation",
                    "--method",
                    "mcmc",
                    "--output-file",
                    str(output_file),
                ],
            )
            assert result.exit_code == 0


class TestComprehensiveValidationCommand:
    """Test comprehensive_validation CLI command."""

    def test_comprehensive_validation_basic(self, cli):
        """Test comprehensive_validation command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["comprehensive-validation"])
        assert result.exit_code == 0

    @pytest.mark.timeout(300)
    def test_comprehensive_validation_with_comprehensive_flag(self, cli):
        """Test comprehensive_validation command with comprehensive flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["comprehensive-validation", "--comprehensive"])
        assert result.exit_code == 0

    @pytest.mark.timeout(300)
    def test_comprehensive_validation_with_output_file(self, cli, tmp_path):
        """Test comprehensive_validation command with output file."""
        output_file = tmp_path / "comprehensive_results.json"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["comprehensive-validation", "--output-file", str(output_file)]
        )
        assert result.exit_code == 0

    @pytest.mark.timeout(300)
    def test_comprehensive_validation_with_parallel(self, cli):
        """Test comprehensive_validation command with parallel flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["comprehensive-validation", "--parallel"])
        assert result.exit_code == 0


class TestGUICommand:
    """Test gui CLI command."""

    @staticmethod
    def _gui_patches():
        """Context manager that blocks all real GUI launchers."""
        from unittest.mock import patch

        return (
            patch("main._launch_validation_gui", return_value=None),
            patch("main._launch_psychological_gui", return_value=None),
            patch("main._launch_analysis_gui", return_value=None),
        )

    def test_gui_basic(self, cli):
        """Test gui command with basic parameters (validation type, no options)."""
        from unittest.mock import patch

        with patch("main._launch_validation_gui") as mock_launch, patch(
            "main._launch_psychological_gui"
        ), patch("main._launch_analysis_gui"):
            runner = CliRunner()
            result = runner.invoke(cli, ["gui"])
            assert result.exit_code == 0
            mock_launch.assert_called_once_with(False)

    def test_gui_with_type(self, cli):
        """Test gui command with GUI type parameter."""
        from unittest.mock import patch

        with patch("main._launch_validation_gui"), patch(
            "main._launch_psychological_gui"
        ) as mock_psych, patch("main._launch_analysis_gui"):
            runner = CliRunner()
            result = runner.invoke(cli, ["gui", "--gui-type", "psychological"])
            assert result.exit_code == 0
            mock_psych.assert_called_once_with(False)

    def test_gui_with_port(self, cli):
        """Test gui command with custom port (port is accepted, passed to context)."""
        from unittest.mock import patch

        with patch("main._launch_validation_gui") as mock_launch, patch(
            "main._launch_psychological_gui"
        ), patch("main._launch_analysis_gui"):
            runner = CliRunner()
            result = runner.invoke(cli, ["gui", "--port", "9090"])
            assert result.exit_code == 0
            mock_launch.assert_called_once()

    def test_gui_with_host(self, cli):
        """Test gui command with custom host."""
        from unittest.mock import patch

        with patch("main._launch_validation_gui") as mock_launch, patch(
            "main._launch_psychological_gui"
        ), patch("main._launch_analysis_gui"):
            runner = CliRunner()
            result = runner.invoke(cli, ["gui", "--host", "0.0.0.0"])
            assert result.exit_code == 0
            mock_launch.assert_called_once()

    def test_gui_with_debug(self, cli):
        """Test gui command with debug flag."""
        from unittest.mock import patch

        with patch("main._launch_validation_gui") as mock_launch, patch(
            "main._launch_psychological_gui"
        ), patch("main._launch_analysis_gui"):
            runner = CliRunner()
            result = runner.invoke(cli, ["gui", "--debug"])
            assert result.exit_code == 0
            mock_launch.assert_called_once_with(True)


class TestLogsCommand:
    """Test logs CLI command."""

    def test_logs_basic(self, cli):
        """Test logs command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["logs"])
        assert result.exit_code == 0

    def test_logs_with_tail(self, cli):
        """Test logs command with tail count."""
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--tail", "10"])
        assert result.exit_code == 0

    def test_logs_with_follow(self, cli):
        """Test logs command with follow flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--follow"])
        assert result.exit_code == 0

    def test_logs_with_level(self, cli):
        """Test logs command with level filter."""
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--level", "ERROR"])
        assert result.exit_code == 0

    def test_logs_with_export(self, cli, tmp_path):
        """Test logs command with export."""
        export_file = tmp_path / "logs_export.txt"
        runner = CliRunner()
        result = runner.invoke(cli, ["logs", "--export", str(export_file)])
        assert result.exit_code == 0


class TestPerformanceCommand:
    """Test performance CLI command."""

    def test_performance_basic(self, cli):
        """Test performance command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["performance"])
        assert result.exit_code == 0

    def test_performance_detailed(self, cli):
        """Test performance command with detailed flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["performance", "--detailed"])
        assert result.exit_code == 0


class TestMultimodalCommand:
    """Test multimodal CLI command."""

    def test_multimodal_basic(self, cli):
        """Test multimodal command with basic parameters."""
        runner = CliRunner()
        result = runner.invoke(cli, ["multimodal"])
        assert result.exit_code == 0

    def test_multimodal_with_input_data(self, cli, tmp_path):
        """Test multimodal command with input data."""
        input_file = tmp_path / "multimodal_data.json"
        input_file.write_text('{"eeg": [1, 2], "fmri": [3, 4]}')
        runner = CliRunner()
        result = runner.invoke(cli, ["multimodal", "--input-data", str(input_file)])
        assert result.exit_code == 0

    def test_multimodal_with_output_file(self, cli, tmp_path):
        """Test multimodal command with output file."""
        output_file = tmp_path / "multimodal_results.json"
        runner = CliRunner()
        result = runner.invoke(cli, ["multimodal", "--output-file", str(output_file)])
        assert result.exit_code == 0

    def test_multimodal_with_modalities(self, cli):
        """Test multimodal command with modalities."""
        runner = CliRunner()
        result = runner.invoke(cli, ["multimodal", "--modalities", "eeg,fmri"])
        assert result.exit_code == 0
