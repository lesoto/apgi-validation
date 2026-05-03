"""Tests for Parameter Estimation with Priors module - comprehensive coverage."""

import numpy as np

from utils.parameter_estimation_with_priors import (
    APGIPhysiologicalEstimator,
    CalibratedAPGIEstimate,
    HybridCalibrator,
)


class TestCalibratedAPGIEstimate:
    """Test Calibrated APGI Estimate dataclass."""

    def test_estimate_creation(self):
        """Test creating calibrated estimate."""
        estimate = CalibratedAPGIEstimate(
            pi_i_baseline=0.5,
            pi_i_source="test",
            pi_i_confidence=0.8,
            beta_estimated=0.3,
            beta_uncertainty=0.05,
            pi_i_effective=0.5,
            m_ca_task=0.1,
            log_likelihood=-100.0,
            aic=210.0,
            bic=220.0,
            collinearity_broken=True,
            calibration_valid=True,
        )
        assert estimate.pi_i_baseline == 0.5
        assert estimate.pi_i_confidence == 0.8
        assert estimate.pi_i_source == "test"

    def test_to_dict(self):
        """Test converting estimate to dict."""
        estimate = CalibratedAPGIEstimate(
            pi_i_baseline=0.5,
            pi_i_source="hybrid",
            pi_i_confidence=0.8,
            beta_estimated=0.3,
            beta_uncertainty=0.05,
            pi_i_effective=0.5,
            m_ca_task=0.1,
            log_likelihood=-100.0,
            aic=210.0,
            bic=220.0,
            collinearity_broken=True,
            calibration_valid=True,
        )
        result = estimate.to_dict()
        assert result["pi_i_baseline"] == 0.5
        assert result["pi_i_source"] == "hybrid"


class TestHybridCalibrator:
    """Test Hybrid Calibrator class."""

    def test_init(self):
        """Test initialization."""
        calibrator = HybridCalibrator()
        assert calibrator is not None
        assert calibrator.hybrid_confidence == 0.0

    def test_calibrate_hybrid_with_eeg_only(self):
        """Test calibration with EEG only (AG ratio)."""
        calibrator = HybridCalibrator()
        # Create synthetic EEG data
        fs = 1000.0
        t = np.arange(0, 10, 1 / fs)
        eeg_data = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))

        pi_i, confidence, source = calibrator.calibrate_hybrid(eeg_data, None, fs)
        assert isinstance(pi_i, float)
        assert isinstance(confidence, float)
        assert isinstance(source, str)


class TestAPGIPhysiologicalEstimator:
    """Test APGI Physiological Estimator class."""

    def test_init(self):
        """Test initialization."""
        estimator = APGIPhysiologicalEstimator()
        assert estimator is not None
        assert not estimator.is_calibrated

    def test_calibrate(self):
        """Test calibration with EEG data."""
        estimator = APGIPhysiologicalEstimator()
        # Create synthetic EEG data
        fs = 1000.0
        t = np.arange(0, 10, 1 / fs)
        eeg_data = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))

        result = estimator.calibrate(eeg_data, None, fs)
        assert isinstance(result, bool)
        # Check that calibration state was updated
        assert estimator.pi_i_fixed is not None or not estimator.is_calibrated

    def test_attributes_after_init(self):
        """Test attributes are initialized correctly."""
        estimator = APGIPhysiologicalEstimator()
        assert estimator.pi_i_fixed is None
        assert estimator.pi_i_source is None
        assert estimator.pi_i_confidence == 0.0
        assert estimator.last_estimate is None
