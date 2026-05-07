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
            alpha_power=10.0,
            gamma_power=5.0,
            ag_ratio=2.0,
            ag_ratio_z=1.5,
            pi_i_physiological=0.5,
            pi_i_confidence=0.8,
            calibration_valid=True,
            warnings=[],
        )
        assert result.pi_i_physiological == 0.5
        assert result.pi_i_confidence == 0.8
        assert result.calibration_valid

    def test_result_with_hep(self):
        """Test result with HEP component."""
        result = PhysiologicalPriorResult(
            alpha_power=12.0,
            gamma_power=4.0,
            ag_ratio=3.0,
            ag_ratio_z=2.0,
            pi_i_physiological=0.6,
            pi_i_confidence=0.75,
            calibration_valid=True,
            warnings=[],
        )
        assert result.pi_i_physiological == 0.6
        assert result.pi_i_confidence == 0.75


class TestHEPCalibrationResult:
    """Test HEP Calibration Result dataclass."""

    def test_calibration_result_creation(self):
        """Test creating HEP calibration result."""
        result = HEPCalibrationResult(
            hep_amplitude_baseline=7.5,
            hep_amplitude_std=1.2,
            hep_trials_used=100,
            pi_i_fixed=0.55,
            pi_i_uncertainty=0.1,
            calibration_duration_sec=180.0,
            heart_rate_bpm=65.0,
            signal_quality=0.85,
            calibration_success=True,
            warnings=[],
        )
        assert result.hep_amplitude_baseline == 7.5
        assert result.hep_trials_used == 100

    def test_calibration_with_metadata(self):
        """Test calibration result with metadata."""
        result = HEPCalibrationResult(
            hep_amplitude_baseline=7.5,
            hep_amplitude_std=1.2,
            hep_trials_used=100,
            pi_i_fixed=0.55,
            pi_i_uncertainty=0.1,
            calibration_duration_sec=180.0,
            heart_rate_bpm=65.0,
            signal_quality=0.85,
            calibration_success=True,
            warnings=[],
        )
        assert result.hep_amplitude_baseline == 7.5
        assert result.pi_i_fixed == 0.55


class TestAlphaGammaRatioPrior:
    """Test Alpha/Gamma Ratio Prior class."""

    def test_init(self):
        """Test initialization."""
        prior = AlphaGammaRatioPrior(
            coupling_constant=0.35,
            reference_mean=1.2,
            reference_std=0.4,
        )
        assert prior.coupling == 0.35
        assert prior.ref_mean == 1.2
        assert prior.ref_std == 0.4

    def test_compute_band_power(self):
        """Test computing band power."""
        prior = AlphaGammaRatioPrior()
        # Create synthetic EEG data
        fs = 1000.0
        duration = 5.0
        t = np.linspace(0, duration, int(fs * duration))
        # Create alpha signal at 9 Hz (within alpha band 8-12 Hz)
        alpha_signal = 2.0 * np.sin(2 * np.pi * 9 * t).reshape(1, -1)

        alpha_power = prior.compute_band_power(alpha_signal, fs, (8.0, 12.0))
        assert isinstance(alpha_power, float)
        assert alpha_power > 0

    def test_compute_alpha_gamma_ratio(self):
        """Test computing alpha/gamma ratio from EEG."""
        prior = AlphaGammaRatioPrior()
        # Create synthetic EEG with alpha and gamma
        fs = 1000.0
        duration = 5.0
        t = np.linspace(0, duration, int(fs * duration))
        alpha_signal = 2.0 * np.sin(2 * np.pi * 10 * t)
        gamma_signal = 0.5 * np.sin(2 * np.pi * 50 * t)
        eeg_data = (alpha_signal + gamma_signal).reshape(1, -1)

        result = prior.compute_alpha_gamma_ratio(eeg_data, fs)
        assert isinstance(result, PhysiologicalPriorResult)
        assert result.alpha_power > 0
        assert result.gamma_power > 0
        assert result.ag_ratio > 0


class TestHEPCalibrationPhase:
    """Test HEP Calibration Phase class."""

    def test_init(self):
        """Test initialization."""
        calibrator = HEPCalibrationPhase(
            hep_scale_factor=0.5,
            min_calibration_sec=120.0,
        )
        assert calibrator.hep_scale == 0.5
        assert calibrator.min_duration == 120.0

    def test_run_calibration(self):
        """Test running calibration phase."""
        calibrator = HEPCalibrationPhase()
        # Create synthetic EEG and ECG data
        fs = 1000.0
        duration = 5.0
        t = np.linspace(0, duration, int(fs * duration))

        # Synthetic EEG with HEP
        eeg_data = np.random.randn(1, len(t)) * 0.1

        # Synthetic ECG with R-peaks
        ecg_data = np.zeros(len(t))
        heart_rate = 80  # BPM
        rr_interval = int(fs * 60 / heart_rate)
        for i in range(0, len(t), rr_interval):
            if i + 10 < len(t):
                ecg_data[i : i + 10] = np.sin(np.linspace(0, np.pi, 10)) * 100

        result = calibrator.run_calibration(eeg_data, ecg_data, fs)
        assert isinstance(result, HEPCalibrationResult)

    def test_compute_hep_amplitude(self):
        """Test computing HEP amplitude."""
        calibrator = HEPCalibrationPhase()
        # Create synthetic epochs (n_epochs x n_channels x n_samples)
        epochs = np.random.randn(10, 1, 1000)  # 10 epochs, 1 channel, 1000 samples

        mean_amp, std_amp = calibrator.compute_hep_amplitude(epochs, fs=1000.0)
        assert isinstance(mean_amp, float)
        assert isinstance(std_amp, float)


class TestCollinearityBreaker:
    """Test Collinearity Breaker class."""

    def test_init(self):
        """Test initialization."""
        breaker = CollinearityBreaker()
        assert breaker is not None

    def test_calibrate_from_ag_ratio(self):
        """Test calibrating from alpha/gamma ratio."""
        breaker = CollinearityBreaker()
        # Create synthetic EEG data
        fs = 1000.0
        duration = 5.0
        t = np.linspace(0, duration, int(fs * duration))
        alpha_signal = 2.0 * np.sin(2 * np.pi * 10 * t)
        gamma_signal = 0.5 * np.sin(2 * np.pi * 50 * t)
        eeg_data = (alpha_signal + gamma_signal).reshape(1, -1)

        success = breaker.calibrate_from_ag_ratio(eeg_data, fs)
        assert isinstance(success, bool)

    def test_estimate_beta_independent(self):
        """Test estimating beta independently."""
        breaker = CollinearityBreaker()
        hep_task = 0.6  # μV during task
        pi_i_current = 1.2
        m_ca = 0.5  # Moderate somatic marker

        beta_estimated = breaker.estimate_beta_independent(hep_task, pi_i_current, m_ca)
        assert isinstance(beta_estimated, float)
        assert 0.3 <= beta_estimated <= 0.8  # Should be within bounds
