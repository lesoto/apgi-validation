"""Tests for Physiological Priors module - comprehensive coverage."""

import numpy as np

from utils.physiological_priors import (
    AlphaGammaRatioPrior,
    CollinearityBreaker,
    HEPCalibrationPhase,
    HEPCalibrationResult,
    PhysiologicalPriorResult,
)


class TestPhysiologicalPriorResult:
    """Test Physiological Prior Result dataclass."""

    def test_result_creation(self):
        """Test creating physiological prior result."""
        result = PhysiologicalPriorResult(
            pi_estimate=0.5,
            confidence=0.8,
            alpha_gamma_ratio_db=-5.0,
            method="alpha_gamma",
        )
        assert result.pi_estimate == 0.5
        assert result.confidence == 0.8
        assert result.method == "alpha_gamma"

    def test_result_with_hep(self):
        """Test result with HEP component."""
        result = PhysiologicalPriorResult(
            pi_estimate=0.6,
            confidence=0.75,
            hep_amplitude_uv=8.0,
            method="hep",
        )
        assert result.hep_amplitude_uv == 8.0


class TestHEPCalibrationResult:
    """Test HEP Calibration Result dataclass."""

    def test_calibration_result_creation(self):
        """Test creating HEP calibration result."""
        result = HEPCalibrationResult(
            mean_hep_amplitude_uv=7.5,
            std_hep_uv=1.2,
            n_trials=100,
            confidence=0.85,
            pi_estimate=0.55,
        )
        assert result.mean_hep_amplitude_uv == 7.5
        assert result.n_trials == 100

    def test_calibration_with_metadata(self):
        """Test calibration result with metadata."""
        result = HEPCalibrationResult(
            mean_hep_amplitude_uv=7.5,
            std_hep_uv=1.2,
            n_trials=100,
            confidence=0.85,
            pi_estimate=0.55,
            cardiac_phase_distribution={"systole": 50, "diastole": 50},
        )
        assert "systole" in result.cardiac_phase_distribution


class TestAlphaGammaRatioPrior:
    """Test Alpha/Gamma Ratio Prior class."""

    def test_init(self):
        """Test initialization."""
        prior = AlphaGammaRatioPrior(
            alpha_band=(8.0, 12.0),
            gamma_band=(30.0, 80.0),
            target_ratio_db=-8.0,
        )
        assert prior.target_ratio_db == -8.0

    def test_compute_pi_from_ratio(self):
        """Test computing Pi from alpha/gamma ratio."""
        prior = AlphaGammaRatioPrior()
        # Typical ratio around -8 dB
        pi = prior._compute_pi_from_ratio(-8.0)
        assert isinstance(pi, float)
        assert 0 < pi < 2  # Pi should be in reasonable range

    def test_calculate_ratio_from_spectrum(self):
        """Test calculating ratio from power spectrum."""
        prior = AlphaGammaRatioPrior()
        freqs = np.linspace(1, 100, 1000)
        # Create spectrum with alpha and gamma peaks
        power = np.ones_like(freqs)
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        gamma_mask = (freqs >= 30) & (freqs <= 80)
        power[alpha_mask] = 10.0  # Higher alpha
        power[gamma_mask] = 2.0  # Lower gamma

        ratio_db = prior._calculate_ratio_from_spectrum(freqs, power)
        assert isinstance(ratio_db, float)


class TestHEPCalibrationPhase:
    """Test HEP Calibration Phase class."""

    def test_init(self):
        """Test initialization."""
        calibrator = HEPCalibrationPhase(
            n_calibration_trials=50,
            target_hep_amplitude_uv=8.0,
        )
        assert calibrator.n_calibration_trials == 50

    def test_run_calibration(self):
        """Test running calibration phase."""
        calibrator = HEPCalibrationPhase(n_calibration_trials=10)
        # Create synthetic HEP data
        hep_data = np.random.randn(10, 500)  # 10 trials, 500 samples
        result = calibrator.run_calibration(hep_data)
        assert isinstance(result, HEPCalibrationResult)

    def test_compute_hep_amplitude(self):
        """Test computing HEP amplitude."""
        calibrator = HEPCalibrationPhase()
        # Create synthetic single-trial HEP
        hep_segment = np.zeros(500)
        hep_segment[250:350] = 8.0  # HEP response

        amplitude = calibrator._compute_hep_amplitude(hep_segment)
        assert isinstance(amplitude, float)
        assert amplitude > 0


class TestCollinearityBreaker:
    """Test Collinearity Breaker class."""

    def test_init(self):
        """Test initialization."""
        breaker = CollinearityBreaker()
        assert breaker is not None

    def test_break_collinearity_with_hep(self):
        """Test breaking collinearity using HEP prior."""
        breaker = CollinearityBreaker()
        # Create synthetic correlated data
        beta_base = 0.5
        pi_i_base = 1.0
        hep_amplitude_uv = 8.0

        beta_estimated = breaker.break_collinearity_with_hep(
            beta_base, pi_i_base, hep_amplitude_uv
        )
        assert isinstance(beta_estimated, float)
        assert beta_estimated != beta_base  # Should be modified

    def test_validate_independence(self):
        """Test validating independence of estimates."""
        breaker = CollinearityBreaker()
        beta_estimate = 0.6
        pi_estimate = 1.2

        is_valid = breaker.validate_independence(beta_estimate, pi_estimate)
        assert isinstance(is_valid, bool)
