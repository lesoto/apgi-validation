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
        if sim.fs != 100.0:
            raise AssertionError(f"Expected fs=100.0, got {sim.fs}")
        if sim.dt != 0.01:
            raise AssertionError(f"Expected dt=0.01, got {sim.dt}")

    def test_init_custom_fs(self):
        """Test simulator initialization with custom sampling rate."""
        sim = MetabolicGroundTruthSimulator(fs=200.0)
        if sim.fs != 200.0:
            raise AssertionError(f"Expected fs=200.0, got {sim.fs}")
        if sim.dt != 0.005:
            raise AssertionError(f"Expected dt=0.005, got {sim.dt}")

    def test_simulate_iatpsnfr_trace_basic(self):
        """Test basic iATPSnFR trace simulation."""
        sim = MetabolicGroundTruthSimulator()
        ignitions = np.array([0, 0, 1, 1, 0, 0, 1, 0])
        trace = sim.simulate_iatpsnfr_trace(ignitions)

        if len(trace) != len(ignitions):
            raise AssertionError(f"Expected trace length {len(ignitions)}, got {len(trace)}")
        if not isinstance(trace, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(trace)}")
        # Trace should be non-negative
        if not np.all(trace >= 0):
            raise ValueError("Trace contains negative values")

    def test_simulate_iatpsnfr_trace_custom_params(self):
        """Test iATPSnFR trace with custom cost coefficients."""
        sim = MetabolicGroundTruthSimulator()
        ignitions = np.array([0, 1, 0, 1, 0])
        trace = sim.simulate_iatpsnfr_trace(ignitions, c1_true=0.2, c2_true=0.03, tau_decay=0.2)

        if len(trace) != len(ignitions):
            raise AssertionError(f"Expected trace length {len(ignitions)}, got {len(trace)}")
        if not isinstance(trace, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(trace)}")

    def test_simulate_iatpsnfr_trace_all_ignitions(self):
        """Test trace with all ignitions active."""
        sim = MetabolicGroundTruthSimulator()
        ignitions = np.ones(100)
        trace = sim.simulate_iatpsnfr_trace(ignitions)

        if len(trace) != 100:
            raise AssertionError(f"Expected trace length 100, got {len(trace)}")
        # With all ignitions, trace should be higher
        if not np.mean(trace) > 0:
            raise ValueError(f"Expected positive mean trace, got {np.mean(trace)}")

    def test_simulate_iatpsnfr_trace_no_ignitions(self):
        """Test trace with no ignitions."""
        sim = MetabolicGroundTruthSimulator()
        ignitions = np.zeros(100)
        trace = sim.simulate_iatpsnfr_trace(ignitions)

        if len(trace) != 100:
            raise AssertionError(f"Expected trace length 100, got {len(trace)}")
        # With no ignitions, trace should be lower (only maintenance)
        # The maintenance cost accumulates, so we check it's relatively low
        if not np.mean(trace) < 0.5:
            raise ValueError(f"Expected mean trace < 0.5, got {np.mean(trace)}")

    def test_simulate_pmrs_flux_basic(self):
        """Test P-MRS flux simulation."""
        sim = MetabolicGroundTruthSimulator()
        trace = np.random.randn(100)
        flux = sim.simulate_pmrs_flux(trace)

        if len(flux) != len(trace):
            raise AssertionError(f"Expected flux length {len(trace)}, got {len(flux)}")
        if not isinstance(flux, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(flux)}")

    def test_simulate_pmrs_flux_custom_window(self):
        """Test P-MRS flux with custom window size."""
        sim = MetabolicGroundTruthSimulator()
        trace = np.random.randn(100)
        flux = sim.simulate_pmrs_flux(trace, window_ms=100.0)

        if len(flux) != len(trace):
            raise AssertionError(f"Expected flux length {len(trace)}, got {len(flux)}")

    def test_simulate_pmrs_flux_small_window(self):
        """Test P-MRS flux with very small window."""
        sim = MetabolicGroundTruthSimulator()
        trace = np.random.randn(100)
        flux = sim.simulate_pmrs_flux(trace, window_ms=1.0)

        # Should handle small window gracefully
        if len(flux) != len(trace):
            raise AssertionError(f"Expected flux length {len(trace)}, got {len(flux)}")


class TestAPGIMetabolicValidator:
    """Test the APGI metabolic validator."""

    def test_init_default_config(self):
        """Test validator initialization with default config."""
        validator = APGIMetabolicValidator()
        if validator.config is None:
            raise ValueError("Validator config should not be None")
        if validator.simulator is None:
            raise ValueError("Validator simulator should not be None")

    def test_init_custom_config(self):
        """Test validator initialization with custom config."""
        custom_config = {"c1": 0.15, "c2": 0.025}
        validator = APGIMetabolicValidator(config=custom_config)
        if validator.config != custom_config:
            raise AssertionError(f"Expected config {custom_config}, got {validator.config}")

    @patch("Validation.VP_16_Metabolic_ATP_GroundTruth.APGIModel")
    def test_validate_c1_c2_ground_truth_basic(self, mock_model):
        """Test basic validation execution."""
        # Mock the APGI model
        mock_instance = MagicMock()
        mock_instance.run.return_value = [
            {"metabolic_cost": 0.05, "ignited": True, "ignition_prob": 0.8} for _ in range(500)
        ]
        mock_model.return_value = mock_instance

        validator = APGIMetabolicValidator()
        results = validator.validate_c1_c2_ground_truth(n_trials=2)

        required_keys = [
            "v16_1_correlation",
            "v16_2_consistency",
            "v16_3_efficiency_gain",
            "c1_fitted",
            "c2_fitted",
            "passed",
        ]
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            raise KeyError(f"Missing required keys in results: {missing_keys}")

    @patch("Validation.VP_16_Metabolic_ATP_GroundTruth.APGIModel")
    def test_validate_c1_c2_ground_truth_with_config(self, mock_model):
        """Test validation with custom config."""
        custom_config = {"c1": 0.12, "c2": 0.018}
        mock_instance = MagicMock()
        mock_instance.run.return_value = [
            {"metabolic_cost": 0.04, "ignited": True, "ignition_prob": 0.7} for _ in range(500)
        ]
        mock_model.return_value = mock_instance

        validator = APGIMetabolicValidator(config=custom_config)
        results = validator.validate_c1_c2_ground_truth(n_trials=2)

        if results is None:
            raise ValueError("Results should not be None")

    @patch("Validation.VP_16_Metabolic_ATP_GroundTruth.APGIModel")
    def test_validate_c1_c2_ground_truth_curve_fit_failure(self, mock_model):
        """Test validation handles curve_fit failures gracefully."""
        mock_instance = MagicMock()
        # Return data that might cause fitting issues
        mock_instance.run.return_value = [
            {"metabolic_cost": 0.01, "ignited": False, "ignition_prob": 0.0} for _ in range(500)
        ]
        mock_model.return_value = mock_instance

        validator = APGIMetabolicValidator()
        results = validator.validate_c1_c2_ground_truth(n_trials=2)

        # Should still return results even with fitting failures
        if results is None:
            raise ValueError("Results should not be None")
        if "v16_2_consistency" not in results:
            raise KeyError("Missing v16_2_consistency in results")


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

        if result["protocol_id"] != "VP_16_Metabolic_ATP_GroundTruth":
            raise AssertionError(f"Expected protocol_id VP_16_Metabolic_ATP_GroundTruth, got {result['protocol_id']}")
        if result["status"] != "success":
            raise AssertionError(f"Expected status success, got {result['status']}")
        if result["passed"] is not True:
            raise AssertionError(f"Expected passed True, got {result['passed']}")
        required_sections = ["named_predictions", "metrics", "metadata"]
        missing_sections = [section for section in required_sections if section not in result]
        if missing_sections:
            raise KeyError(f"Missing required sections in result: {missing_sections}")

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

        if result["status"] != "failed":
            raise AssertionError(f"Expected status failed, got {result['status']}")
        if result["passed"] is not False:
            raise AssertionError(f"Expected passed False, got {result['passed']}")

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

        required_predictions = ["V16.1", "V16.2", "V16.3"]
        missing_predictions = [pred for pred in required_predictions if pred not in result["named_predictions"]]
        if missing_predictions:
            raise KeyError(f"Missing required predictions: {missing_predictions}")

        # Check V16.1 structure
        v16_1 = result["named_predictions"]["V16.1"]
        required_fields = ["passed", "value", "threshold", "description"]
        missing_fields = [field for field in required_fields if field not in v16_1]
        if missing_fields:
            raise KeyError(f"V16.1 missing required fields: {missing_fields}")

        # Check V16.3 structure
        v16_3 = result["named_predictions"]["V16.3"]
        required_fields = ["passed", "value", "threshold"]
        missing_fields = [field for field in required_fields if field not in v16_3]
        if missing_fields:
            raise KeyError(f"V16.3 missing required fields: {missing_fields}")

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
        required_metrics = ["c1_fitted", "c2_fitted", "correlation", "efficiency_gain"]
        missing_metrics = [metric for metric in required_metrics if metric not in metrics]
        if missing_metrics:
            raise KeyError(f"Missing required metrics: {missing_metrics}")

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
        required_metadata = ["temporal_resolution", "ground_truth_source", "timestamp"]
        missing_metadata = [item for item in required_metadata if item not in metadata]
        if missing_metadata:
            raise KeyError(f"Missing required metadata: {missing_metadata}")
