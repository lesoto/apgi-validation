"""Tests for FP_09 Neural Signatures P3b HEP - increase coverage from 15%."""

import numpy as np
import pytest

from Falsification.FP_09_NeuralSignatures_P3b_HEP import (
    EEGData,
    FalsificationThresholds,
    NeuralSignatureResult,
    NeuralSignatureValidator,
    compute_band_power,
    detect_gamma_oscillation,
    detect_hep_amplitude,
    detect_neural_signatures,
    detect_p3b_amplitude_from_eeg,
    detect_theta_gamma_pac,
)


class TestFalsificationThresholds:
    """Test Falsification Thresholds class."""

    def test_threshold_constants(self):
        """Test threshold constants are defined."""
        assert FalsificationThresholds.P3B_MIN_AMPLITUDE > 0
        assert FalsificationThresholds.HEP_MIN_AMPLITUDE > 0
        assert FalsificationThresholds.GAMMA_MIN_POWER > 0

    def test_get_threshold(self):
        """Test getting thresholds for specific metrics."""
        p3b_thresh = FalsificationThresholds.get_threshold("p3b_amplitude")
        hep_thresh = FalsificationThresholds.get_threshold("hep_amplitude")
        assert p3b_thresh > 0
        assert hep_thresh > 0

    def test_compute_bonferroni_alpha(self):
        """Test Bonferroni correction computation."""
        alpha = FalsificationThresholds.compute_bonferroni_alpha(4)
        assert alpha == 0.0125  # 0.05 / 4


class TestNeuralSignatureResult:
    """Test Neural Signature Result dataclass."""

    def test_result_creation(self):
        """Test creating a neural signature result."""
        result = NeuralSignatureResult(
            prediction_id="P1.1",
            metric_name="gamma_power",
            value=0.5,
            threshold=0.15,
            significant=True,
            effect_size=0.8,
            p_value=0.01,
            description="Test result",
            falsification_passed=True,
        )
        assert result.prediction_id == "P1.1"
        assert result.value == 0.5
        assert result.falsification_passed is True


class TestEEGData:
    """Test EEG Data dataclass."""

    def test_eeg_data_creation(self):
        """Test creating EEG data object."""
        data = np.random.randn(10, 64, 1000)  # epochs x channels x time
        eeg = EEGData(
            data=data,
            fs=1000.0,
            channels=[f"ch{i}" for i in range(64)],
            times=np.arange(1000) / 1000.0,
        )
        assert eeg.fs == 1000.0
        assert len(eeg.channels) == 64


class TestDetectGammaOscillation:
    """Test gamma oscillation detection."""

    def test_detect_gamma_with_sufficient_data(self):
        """Test gamma detection with sufficient data length."""
        # Create synthetic data with gamma component
        fs = 1000.0
        t = np.arange(2000) / fs
        gamma_signal = np.sin(2 * np.pi * 40 * t)  # 40 Hz gamma
        eeg_data = gamma_signal + 0.1 * np.random.randn(len(t))

        result = detect_gamma_oscillation(eeg_data, fs=fs)
        assert isinstance(result, NeuralSignatureResult)
        assert result.metric_name == "gamma_power"

    def test_detect_gamma_insufficient_data(self):
        """Test gamma detection raises error for insufficient data."""
        with pytest.raises(ValueError):
            detect_gamma_oscillation(np.zeros(100), fs=1000.0)


class TestDetectThetaGammaPAC:
    """Test theta-gamma phase amplitude coupling detection."""

    def test_detect_pac_with_sufficient_data(self):
        """Test PAC detection with sufficient data."""
        fs = 1000.0
        t = np.arange(2000) / fs
        # Create synthetic signal with theta-gamma coupling
        theta = np.sin(2 * np.pi * 6 * t)
        gamma = np.sin(2 * np.pi * 40 * t) * (1 + 0.5 * theta)
        eeg_data = gamma + 0.1 * np.random.randn(len(t))

        result = detect_theta_gamma_pac(eeg_data, fs=fs)
        assert isinstance(result, NeuralSignatureResult)
        assert result.metric_name == "theta_gamma_pac"

    def test_detect_pac_insufficient_data(self):
        """Test PAC detection with insufficient data."""
        result = detect_theta_gamma_pac(np.zeros(100), fs=1000.0)
        # Returns result with insufficient data description
        assert isinstance(result, NeuralSignatureResult)


class TestDetectP3bAmplitude:
    """Test P3b amplitude detection."""

    def test_detect_p3b_with_sufficient_data(self):
        """Test P3b detection with sufficient data."""
        fs = 1000.0
        t = np.arange(1000) / fs
        # Create synthetic P3b-like response around 350ms
        eeg_data = np.zeros_like(t)
        p3b_window = (t >= 0.3) & (t <= 0.5)
        eeg_data[p3b_window] = 15.0  # P3b amplitude

        result = detect_p3b_amplitude_from_eeg(eeg_data, fs=fs, stimulus_time=0.0)
        assert isinstance(result, NeuralSignatureResult)
        assert result.metric_name == "p3b_amplitude_eeg"


class TestDetectHepAmplitude:
    """Test HEP (Heartbeat Evoked Potential) amplitude detection."""

    def test_detect_hep_with_sufficient_data(self):
        """Test HEP detection with sufficient data."""
        fs = 1000.0
        t = np.arange(1000) / fs
        # Create synthetic HEP-like response
        eeg_data = np.zeros_like(t)
        hep_window = (t >= 0.05) & (t <= 0.25)
        eeg_data[hep_window] = 8.0  # HEP amplitude

        result = detect_hep_amplitude(eeg_data, fs=fs, stimulus_time=0.0)
        assert isinstance(result, NeuralSignatureResult)


class TestDetectP3bFromEEG:
    """Test P3b detection from EEG data."""

    def test_detect_p3b_from_eeg_basic(self):
        """Test basic P3b detection from EEG."""
        fs = 1000.0
        eeg_data = np.random.randn(1000)
        # Add P3b-like component
        eeg_data[350:450] = 15.0

        result = detect_p3b_amplitude_from_eeg(eeg_data, fs=fs)
        assert isinstance(result, NeuralSignatureResult)


class TestComputeBandPower:
    """Test band power computation."""

    def test_compute_gamma_band_power(self):
        """Test computing gamma band power."""
        fs = 1000.0
        t = np.arange(2000) / fs
        gamma_signal = np.sin(2 * np.pi * 40 * t)

        band_power, baseline, threshold = compute_band_power(
            gamma_signal, fs, (30.0, 80.0)
        )
        assert band_power > 0
        assert isinstance(threshold, float)


class TestDetectNeuralSignatures:
    """Test neural signature detection function."""

    def test_detect_with_valid_data(self):
        """Test detecting signatures from valid data."""
        fs = 1000.0
        eeg_data = np.random.randn(64, 2000)  # channels x samples

        result = detect_neural_signatures(eeg_data, markers=["gamma", "p3b"], fs=fs)
        assert isinstance(result, dict)

    def test_detect_with_single_channel(self):
        """Test detecting from single channel data."""
        fs = 1000.0
        eeg_data = np.random.randn(2000)

        result = detect_neural_signatures(eeg_data, markers=["gamma"], fs=fs)
        assert isinstance(result, dict)


class TestNeuralSignatureValidator:
    """Test Neural Signature Validator class."""

    def test_validator_init(self):
        """Test validator initialization."""
        validator = NeuralSignatureValidator()
        assert validator is not None
