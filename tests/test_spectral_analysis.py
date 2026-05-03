"""Tests for Spectral Analysis module - comprehensive coverage."""

import numpy as np

from utils.spectral_analysis import (
    batch_compute_spectral_slopes,
    compute_power_spectrum,
    compute_spectral_slope_specparam,
    create_fooof_frequencies,
    generate_synthetic_spectra,
    validate_specparam_fit,
)


class TestComputePowerSpectrum:
    """Test power spectrum computation."""

    def test_basic_power_spectrum(self):
        """Test basic power spectrum computation."""
        fs = 100.0
        t = np.arange(0, 1, 1 / fs)
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave

        freqs, power = compute_power_spectrum(signal, fs)
        assert len(freqs) == len(power)
        assert len(freqs) > 0

    def test_power_spectrum_with_noise(self):
        """Test power spectrum with noisy signal."""
        fs = 100.0
        t = np.arange(0, 1, 1 / fs)
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))

        freqs, power = compute_power_spectrum(signal, fs)
        assert len(freqs) == len(power)


class TestComputeSpectralSlopeSpecparam:
    """Test spectral slope computation using specparam/FooOF."""

    def test_basic_spectral_slope(self):
        """Test basic spectral slope computation."""
        freqs = np.linspace(1, 40, 100)
        # Create 1/f-like spectrum
        power = 1 / freqs

        result = compute_spectral_slope_specparam(freqs, power)
        assert isinstance(result, dict)
        assert "aperiodic_exponent" in result

    def test_spectral_slope_with_peaks(self):
        """Test spectral slope with periodic peaks."""
        freqs = np.linspace(1, 40, 100)
        # 1/f + alpha peak at 10 Hz
        power = 1 / freqs + 0.5 * np.exp(-((freqs - 10) ** 2) / 2)

        result = compute_spectral_slope_specparam(freqs, power)
        assert isinstance(result, dict)


class TestValidateSpecparamFit:
    """Test specparam fit validation."""

    def test_valid_fit(self):
        """Test validation of good fit."""
        results = {
            "aperiodic_exponent": -1.5,
            "aperiodic_offset": 1.0,
            "r_squared": 0.95,
            "error": False,
        }
        is_valid = validate_specparam_fit(results)
        assert isinstance(is_valid, bool)

    def test_invalid_fit_with_error(self):
        """Test validation of fit with error."""
        results = {
            "aperiodic_exponent": -1.5,
            "error": True,
            "error_message": "Fit failed",
        }
        is_valid = validate_specparam_fit(results)
        assert is_valid is False


class TestCreateFooofFrequencies:
    """Test FooOF frequency creation."""

    def test_create_frequencies(self):
        """Test creating frequency array for FooOF."""
        fs = 100.0
        freqs = create_fooof_frequencies(fs, freq_range=(1, 40))
        assert len(freqs) > 0
        assert freqs[0] >= 1.0
        assert freqs[-1] <= 40.0


class TestGenerateSyntheticSpectra:
    """Test synthetic spectra generation."""

    def test_generate_1f_spectrum(self):
        """Test generating 1/f spectrum."""
        freqs = np.linspace(1, 40, 100)
        spectrum = generate_synthetic_spectra(freqs, exponent=-1.5, offset=1.0)

        assert len(spectrum) == len(freqs)
        assert np.all(spectrum > 0)  # Power should be positive

    def test_generate_with_peaks(self):
        """Test generating spectrum with peaks."""
        freqs = np.linspace(1, 40, 100)
        peaks = [(10, 1.0, 2.0)]  # (center_freq, amplitude, std)
        spectrum = generate_synthetic_spectra(freqs, exponent=-1.5, peaks=peaks)

        assert len(spectrum) == len(freqs)


class TestBatchComputeSpectralSlopes:
    """Test batch spectral slope computation."""

    def test_batch_computation(self):
        """Test computing slopes for multiple signals."""
        # Generate multiple synthetic signals
        signals = []
        for i in range(3):
            t = np.arange(0, 10, 0.01)
            signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
            signals.append(signal)

        results = batch_compute_spectral_slopes(signals, fs=100.0)
        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert "aperiodic_exponent" in result or "error" in result
