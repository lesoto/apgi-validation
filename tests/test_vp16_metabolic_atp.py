"""
Tests for VP_16_Metabolic_ATP_GroundTruth.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Validation.VP_16_Metabolic_ATP_GroundTruth import (
    APGIMetabolicValidator,
    MetabolicGroundTruthSimulator,
    run_validation,
)


class TestMetabolicGroundTruthSimulator:
    """Test the ATP trace simulator."""

    def test_init_default(self):
        """Test simulator initialization with default parameters."""
        sim = MetabolicGroundTruthSimulator()
        assert sim.fs == 100.0
        assert sim.dt == 0.01

    def test_init_custom_fs(self):
        """Test simulator initialization with custom sampling rate."""
        sim = MetabolicGroundTruthSimulator(fs=200.0)
        assert sim.fs == 200.0
        assert sim.dt == 0.005

    def test_simulate_iatpsnfr_trace_basic(self):
        """Test basic iATPSnFR trace simulation."""
        sim = MetabolicGroundTruthSimulator()
        ignitions = np.array([0, 0, 1, 1, 0, 0, 1, 0])
        trace = sim.simulate_iatpsnfr_trace(ignitions)

        assert len(trace) == len(ignitions)
        assert isinstance(trace, np.ndarray)
        # Trace should be non-negative
        assert np.all(trace >= 0)

    def test_simulate_iatpsnfr_trace_custom_params(self):
        """Test iATPSnFR trace with custom cost coefficients."""
        sim = MetabolicGroundTruthSimulator()
        ignitions = np.array([0, 1, 0, 1, 0])
        trace = sim.simulate_iatpsnfr_trace(
            ignitions, c1_true=0.2, c2_true=0.03, tau_decay=0.2
        )

        assert len(trace) == len(ignitions)
        assert isinstance(trace, np.ndarray)

    def test_simulate_iatpsnfr_trace_all_ignitions(self):
        """Test trace with all ignitions active."""
        sim = MetabolicGroundTruthSimulator()
        ignitions = np.ones(100)
        trace = sim.simulate_iatpsnfr_trace(ignitions)

        assert len(trace) == 100
        # With all ignitions, trace should be higher
        assert np.mean(trace) > 0

    def test_simulate_iatpsnfr_trace_no_ignitions(self):
        """Test trace with no ignitions."""
        sim = MetabolicGroundTruthSimulator()
        ignitions = np.zeros(100)
        trace = sim.simulate_iatpsnfr_trace(ignitions)

        assert len(trace) == 100
        # With no ignitions, trace should be lower (only maintenance)
        # The maintenance cost accumulates, so we check it's relatively low
        assert np.mean(trace) < 0.5

    def test_simulate_pmrs_flux_basic(self):
        """Test P-MRS flux simulation."""
        sim = MetabolicGroundTruthSimulator()
        trace = np.random.randn(100)
        flux = sim.simulate_pmrs_flux(trace)

        assert len(flux) == len(trace)
        assert isinstance(flux, np.ndarray)

    def test_simulate_pmrs_flux_custom_window(self):
        """Test P-MRS flux with custom window size."""
        sim = MetabolicGroundTruthSimulator()
        trace = np.random.randn(100)
        flux = sim.simulate_pmrs_flux(trace, window_ms=100.0)

        assert len(flux) == len(trace)

    def test_simulate_pmrs_flux_small_window(self):
        """Test P-MRS flux with very small window."""
        sim = MetabolicGroundTruthSimulator()
        trace = np.random.randn(100)
        flux = sim.simulate_pmrs_flux(trace, window_ms=1.0)

        # Should handle small window gracefully
        assert len(flux) == len(trace)


class TestAPGIMetabolicValidator:
    """Test the APGI metabolic validator."""

    def test_init_default_config(self):
        """Test validator initialization with default config."""
        validator = APGIMetabolicValidator()
        assert validator.config is not None
        assert validator.simulator is not None

    def test_init_custom_config(self):
        """Test validator initialization with custom config."""
        custom_config = {"c1": 0.15, "c2": 0.025}
        validator = APGIMetabolicValidator(config=custom_config)
        assert validator.config == custom_config

    @patch("Validation.VP_16_Metabolic_ATP_GroundTruth.APGIModel")
    def test_validate_c1_c2_ground_truth_basic(self, mock_model):
        """Test basic validation execution."""
        # Mock the APGI model
        mock_instance = MagicMock()
        mock_instance.run.return_value = [
            {"metabolic_cost": 0.05, "ignited": True, "ignition_prob": 0.8}
            for _ in range(500)
        ]
        mock_model.return_value = mock_instance

        validator = APGIMetabolicValidator()
        results = validator.validate_c1_c2_ground_truth(n_trials=2)

        assert "v16_1_correlation" in results
        assert "v16_2_consistency" in results
        assert "v16_3_efficiency_gain" in results
        assert "c1_fitted" in results
        assert "c2_fitted" in results
        assert "passed" in results

    @patch("Validation.VP_16_Metabolic_ATP_GroundTruth.APGIModel")
    def test_validate_c1_c2_ground_truth_with_config(self, mock_model):
        """Test validation with custom config."""
        custom_config = {"c1": 0.12, "c2": 0.018}
        mock_instance = MagicMock()
        mock_instance.run.return_value = [
            {"metabolic_cost": 0.04, "ignited": True, "ignition_prob": 0.7}
            for _ in range(500)
        ]
        mock_model.return_value = mock_instance

        validator = APGIMetabolicValidator(config=custom_config)
        results = validator.validate_c1_c2_ground_truth(n_trials=2)

        assert results is not None

    @patch("Validation.VP_16_Metabolic_ATP_GroundTruth.APGIModel")
    def test_validate_c1_c2_ground_truth_curve_fit_failure(self, mock_model):
        """Test validation handles curve_fit failures gracefully."""
        mock_instance = MagicMock()
        # Return data that might cause fitting issues
        mock_instance.run.return_value = [
            {"metabolic_cost": 0.01, "ignited": False, "ignition_prob": 0.0}
            for _ in range(500)
        ]
        mock_model.return_value = mock_instance

        validator = APGIMetabolicValidator()
        results = validator.validate_c1_c2_ground_truth(n_trials=2)

        # Should still return results even with fitting failures
        assert results is not None
        assert "v16_2_consistency" in results


class TestRunValidation:
    """Test the run_validation entry point."""

    @patch("Validation.VP_16_Metabolic_ATP_GroundTruth.APGIMetabolicValidator")
    def test_run_validation_success(self, mock_validator):
        """Test run_validation with successful validation."""
        mock_instance = MagicMock()
        mock_instance.validate_c1_c2_ground_truth.return_value = {
            "v16_1_correlation": 0.8,
            "v16_2_consistency": True,
            "v16_3_efficiency_gain": 0.25,
            "c1_fitted": 0.12,
            "c2_fitted": 0.02,
            "passed": True,
        }
        mock_validator.return_value = mock_instance

        result = run_validation()

        assert result["protocol_id"] == "VP_16_Metabolic_ATP_GroundTruth"
        assert result["status"] == "success"
        assert result["passed"] is True
        assert "named_predictions" in result
        assert "metrics" in result
        assert "metadata" in result

    @patch("Validation.VP_16_Metabolic_ATP_GroundTruth.APGIMetabolicValidator")
    def test_run_validation_failure(self, mock_validator):
        """Test run_validation with failed validation."""
        mock_instance = MagicMock()
        mock_instance.validate_c1_c2_ground_truth.return_value = {
            "v16_1_correlation": 0.5,
            "v16_2_consistency": False,
            "v16_3_efficiency_gain": 0.1,
            "c1_fitted": 0.0,
            "c2_fitted": 0.0,
            "passed": False,
        }
        mock_validator.return_value = mock_instance

        result = run_validation()

        assert result["status"] == "failed"
        assert result["passed"] is False

    @patch("Validation.VP_16_Metabolic_ATP_GroundTruth.APGIMetabolicValidator")
    def test_run_validation_named_predictions(self, mock_validator):
        """Test named predictions structure."""
        mock_instance = MagicMock()
        mock_instance.validate_c1_c2_ground_truth.return_value = {
            "v16_1_correlation": 0.8,
            "v16_2_consistency": True,
            "v16_3_efficiency_gain": 0.25,
            "c1_fitted": 0.12,
            "c2_fitted": 0.02,
            "passed": True,
        }
        mock_validator.return_value = mock_instance

        result = run_validation()

        assert "V16.1" in result["named_predictions"]
        assert "V16.2" in result["named_predictions"]
        assert "V16.3" in result["named_predictions"]

        # Check V16.1 structure
        v16_1 = result["named_predictions"]["V16.1"]
        assert "passed" in v16_1
        assert "value" in v16_1
        assert "threshold" in v16_1
        assert "description" in v16_1

        # Check V16.3 structure
        v16_3 = result["named_predictions"]["V16.3"]
        assert "passed" in v16_3
        assert "value" in v16_3
        assert "threshold" in v16_3

    @patch("Validation.VP_16_Metabolic_ATP_GroundTruth.APGIMetabolicValidator")
    def test_run_validation_metrics(self, mock_validator):
        """Test metrics structure."""
        mock_instance = MagicMock()
        mock_instance.validate_c1_c2_ground_truth.return_value = {
            "v16_1_correlation": 0.8,
            "v16_2_consistency": True,
            "v16_3_efficiency_gain": 0.25,
            "c1_fitted": 0.12,
            "c2_fitted": 0.02,
            "passed": True,
        }
        mock_validator.return_value = mock_instance

        result = run_validation()

        metrics = result["metrics"]
        assert "c1_fitted" in metrics
        assert "c2_fitted" in metrics
        assert "correlation" in metrics
        assert "efficiency_gain" in metrics

    @patch("Validation.VP_16_Metabolic_ATP_GroundTruth.APGIMetabolicValidator")
    def test_run_validation_metadata(self, mock_validator):
        """Test metadata structure."""
        mock_instance = MagicMock()
        mock_instance.validate_c1_c2_ground_truth.return_value = {
            "v16_1_correlation": 0.8,
            "v16_2_consistency": True,
            "v16_3_efficiency_gain": 0.25,
            "c1_fitted": 0.12,
            "c2_fitted": 0.02,
            "passed": True,
        }
        mock_validator.return_value = mock_instance

        result = run_validation()

        metadata = result["metadata"]
        assert "temporal_resolution" in metadata
        assert "ground_truth_source" in metadata
        assert "timestamp" in metadata
