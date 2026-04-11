"""
Comprehensive tests for eeg_simulator utility module.
===================================================

Tests all functions and classes in eeg_simulator.py including:
- EEGSimulator class
- Individual component generation
- Complete epoch generation
- Analysis functions
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.eeg_simulator import EEGSimulator, create_default_simulator
except ImportError as e:
    pytest.skip(f"Cannot import eeg_simulator: {e}", allow_module_level=True)


class TestEEGSimulatorInitialization:
    """Test EEGSimulator class initialization."""

    def test_default_initialization(self):
        """Test EEGSimulator with default parameters."""
        simulator = EEGSimulator()

        assert (
            simulator.sampling_rate == 1000.0
        ), "Default sampling rate should be 1000 Hz"
        assert simulator.eeg_duration == 1.5, "Default duration should be 1.5 seconds"
        assert (
            simulator.cardiac_cycle_duration == 0.9
        ), "Default cardiac cycle should be 0.9s"
        assert (
            simulator.r_wave_duration == 0.05
        ), "Default R-wave duration should be 0.05s"
        assert simulator.hep_window_start == 0.2, "HEP window should start at 0.2s"
        assert simulator.hep_window_end == 0.6, "HEP window should end at 0.6s"
        assert simulator.p3b_window_start == 0.3, "P3b window should start at 0.3s"
        assert simulator.p3b_window_end == 0.6, "P3b window should end at 0.6s"
        assert simulator.noise_level == 0.1, "Default noise level should be 0.1"

    def test_custom_initialization(self):
        """Test EEGSimulator with custom parameters."""
        simulator = EEGSimulator(
            sampling_rate=500.0,
            eeg_duration=2.0,
            cardiac_cycle_duration=1.0,
            r_wave_duration=0.1,
            hep_window_start=0.15,
            hep_window_end=0.65,
            p3b_window_start=0.25,
            p3b_window_end=0.65,
            noise_level=0.2,
        )

        assert simulator.sampling_rate == 500.0, "Should use custom sampling rate"
        assert simulator.eeg_duration == 2.0, "Should use custom duration"
        assert (
            simulator.cardiac_cycle_duration == 1.0
        ), "Should use custom cardiac cycle"
        assert simulator.r_wave_duration == 0.1, "Should use custom R-wave duration"
        assert simulator.hep_window_start == 0.15, "Should use custom HEP start"
        assert simulator.hep_window_end == 0.65, "Should use custom HEP end"
        assert simulator.p3b_window_start == 0.25, "Should use custom P3b start"
        assert simulator.p3b_window_end == 0.65, "Should use custom P3b end"
        assert simulator.noise_level == 0.2, "Should use custom noise level"

    def test_sample_calculations(self):
        """Test that sample calculations are correct."""
        simulator = EEGSimulator(sampling_rate=1000.0, eeg_duration=1.0)

        assert simulator.n_samples == 1000, "Should calculate correct number of samples"
        assert simulator.r_wave_samples == 50, "Should calculate correct R-wave samples"
        assert (
            simulator.hep_start_sample == 200
        ), "Should calculate correct HEP start sample"
        assert (
            simulator.hep_end_sample == 600
        ), "Should calculate correct HEP end sample"
        assert (
            simulator.p3b_start_sample == 300
        ), "Should calculate correct P3b start sample"
        assert (
            simulator.p3b_end_sample == 600
        ), "Should calculate correct P3b end sample"


class TestGenerateRWave:
    """Test R-wave generation."""

    def test_generate_r_wave_basic(self):
        """Test basic R-wave generation."""
        simulator = EEGSimulator()

        r_wave = simulator.generate_r_wave()

        assert isinstance(r_wave, np.ndarray), "Should return numpy array"
        assert len(r_wave) == simulator.n_samples, "Should have correct length"
        assert np.max(r_wave) > 0, "Should have positive peak"
        assert r_wave[0] > 0, "R-wave should start at time 0"

    def test_generate_r_wave_custom_amplitude(self):
        """Test R-wave generation with custom amplitude."""
        simulator = EEGSimulator()

        r_wave_default = simulator.generate_r_wave(amplitude=1.0)
        r_wave_large = simulator.generate_r_wave(amplitude=2.0)

        assert np.max(r_wave_large) > np.max(
            r_wave_default
        ), "Larger amplitude should produce larger peak"

    def test_generate_r_wave_timing_jitter(self):
        """Test R-wave generation with timing jitter."""
        simulator = EEGSimulator()

        # Generate multiple R-waves with jitter
        r_waves = []
        for _ in range(10):
            r_wave = simulator.generate_r_wave(timing_jitter=0.01)
            r_waves.append(r_wave)

        # Should have some variation due to jitter
        max_peaks = [np.max(rw) for rw in r_waves]
        assert len(set(max_peaks)) > 1, "Should have variation due to jitter"

    def test_generate_r_wave_zero_amplitude(self):
        """Test R-wave generation with zero amplitude."""
        simulator = EEGSimulator()

        r_wave = simulator.generate_r_wave(amplitude=0.0)

        assert np.max(r_wave) == 0.0, "Zero amplitude should produce zero signal"


class TestGenerateHEP:
    """Test HEP generation."""

    def test_generate_hep_basic(self):
        """Test basic HEP generation."""
        simulator = EEGSimulator()

        hep = simulator.generate_hep(r_wave_time=0.0)

        assert isinstance(hep, np.ndarray), "Should return numpy array"
        assert len(hep) == simulator.n_samples, "Should have correct length"
        assert np.max(hep) >= 0, "Should have non-negative values"

    def test_generate_hep_custom_amplitude(self):
        """Test HEP generation with custom amplitude."""
        simulator = EEGSimulator()

        hep_default = simulator.generate_hep(r_wave_time=0.0, amplitude=0.15)
        hep_large = simulator.generate_hep(r_wave_time=0.0, amplitude=0.3)

        assert np.max(hep_large) > np.max(
            hep_default
        ), "Larger amplitude should produce larger HEP"

    def test_generate_hep_modulation(self):
        """Test HEP generation with modulation."""
        simulator = EEGSimulator()

        hep_default = simulator.generate_hep(r_wave_time=0.0, modulation=1.0)
        hep_modulated = simulator.generate_hep(r_wave_time=0.0, modulation=2.0)

        assert np.max(hep_modulated) > np.max(
            hep_default
        ), "Modulation should affect HEP amplitude"

    def test_generate_hep_timing(self):
        """Test HEP timing relative to R-wave."""
        simulator = EEGSimulator()

        # HEP should appear after R-wave
        r_wave_time = 0.0
        hep = simulator.generate_hep(r_wave_time=r_wave_time)

        # Find peak location
        peak_idx = np.argmax(hep)
        peak_time = peak_idx / simulator.sampling_rate

        assert peak_time > r_wave_time + 0.1, "HEP peak should occur after R-wave"

    def test_generate_hep_zero_amplitude(self):
        """Test HEP generation with zero amplitude."""
        simulator = EEGSimulator()

        hep = simulator.generate_hep(r_wave_time=0.0, amplitude=0.0)

        assert np.max(hep) == 0.0, "Zero amplitude should produce zero signal"


class TestGenerateP3B:
    """Test P3b generation."""

    def test_generate_p3b_basic(self):
        """Test basic P3b generation."""
        simulator = EEGSimulator()

        p3b = simulator.generate_p3b(stimulus_time=0.5)

        assert isinstance(p3b, np.ndarray), "Should return numpy array"
        assert len(p3b) == simulator.n_samples, "Should have correct length"
        assert np.max(p3b) >= 0, "Should have non-negative values"

    def test_generate_p3b_custom_amplitude(self):
        """Test P3b generation with custom amplitude."""
        simulator = EEGSimulator()

        p3b_default = simulator.generate_p3b(stimulus_time=0.5, amplitude=0.3)
        p3b_large = simulator.generate_p3b(stimulus_time=0.5, amplitude=0.6)

        assert np.max(p3b_large) > np.max(
            p3b_default
        ), "Larger amplitude should produce larger P3b"

    def test_generate_p3b_custom_latency(self):
        """Test P3b generation with custom latency."""
        simulator = EEGSimulator()

        p3b_early = simulator.generate_p3b(stimulus_time=0.5, latency=0.3)
        p3b_late = simulator.generate_p3b(stimulus_time=0.5, latency=0.5)

        # Find peaks
        peak_early_idx = np.argmax(p3b_early)
        peak_late_idx = np.argmax(p3b_late)

        assert (
            peak_late_idx > peak_early_idx
        ), "Longer latency should produce later peak"

    def test_generate_p3b_modulation(self):
        """Test P3b generation with modulation."""
        simulator = EEGSimulator()

        p3b_default = simulator.generate_p3b(stimulus_time=0.5, modulation=1.0)
        p3b_modulated = simulator.generate_p3b(stimulus_time=0.5, modulation=2.0)

        assert np.max(p3b_modulated) > np.max(
            p3b_default
        ), "Modulation should affect P3b amplitude"

    def test_generate_p3b_timing(self):
        """Test P3b timing relative to stimulus."""
        simulator = EEGSimulator()

        stimulus_time = 0.5
        p3b = simulator.generate_p3b(stimulus_time=stimulus_time, latency=0.35)

        # Find peak location
        peak_idx = np.argmax(p3b)
        peak_time = peak_idx / simulator.sampling_rate

        expected_peak_time = stimulus_time + 0.35
        assert (
            abs(peak_time - expected_peak_time) < 0.1
        ), "P3b peak should occur at expected time"

    def test_generate_p3b_zero_amplitude(self):
        """Test P3b generation with zero amplitude."""
        simulator = EEGSimulator()

        p3b = simulator.generate_p3b(stimulus_time=0.5, amplitude=0.0)

        assert np.max(p3b) == 0.0, "Zero amplitude should produce zero signal"


class TestGenerateBackgroundEEG:
    """Test background EEG generation."""

    def test_generate_background_eeg_basic(self):
        """Test basic background EEG generation."""
        simulator = EEGSimulator()

        background = simulator.generate_background_eeg()

        assert isinstance(background, np.ndarray), "Should return numpy array"
        assert len(background) == simulator.n_samples, "Should have correct length"
        assert np.std(background) > 0, "Should have some variability"

    def test_generate_background_eeg_custom_powers(self):
        """Test background EEG with custom power settings."""
        simulator = EEGSimulator()

        background_low = simulator.generate_background_eeg(
            alpha_power=0.1, beta_power=0.1, theta_power=0.1
        )
        background_high = simulator.generate_background_eeg(
            alpha_power=1.0, beta_power=1.0, theta_power=1.0
        )

        assert np.std(background_high) > np.std(
            background_low
        ), "Higher power should produce more variability"

    def test_generate_background_eeg_components(self):
        """Test that background EEG contains expected frequency components."""
        simulator = EEGSimulator(sampling_rate=1000.0, eeg_duration=2.0)

        background = simulator.generate_background_eeg(
            alpha_power=1.0, beta_power=0.0, theta_power=0.0
        )

        # Should have dominant alpha component (10 Hz)
        from scipy import signal

        f, Pxx = signal.welch(background, fs=simulator.sampling_rate)

        # Find peak frequency
        peak_idx = np.argmax(Pxx)
        peak_freq = f[peak_idx]

        # Should be close to alpha frequency (10 Hz)
        assert (
            8 <= peak_freq <= 12
        ), f"Peak frequency {peak_freq} should be in alpha band"


class TestGenerateEpoch:
    """Test complete epoch generation."""

    def test_generate_epoch_basic(self):
        """Test basic epoch generation."""
        simulator = EEGSimulator()

        epoch = simulator.generate_epoch()

        assert isinstance(epoch, dict), "Should return dictionary"
        required_keys = ["eeg", "r_wave", "hep", "background"]
        for key in required_keys:
            assert key in epoch, f"Missing key: {key}"
            assert isinstance(epoch[key], np.ndarray), f"{key} should be numpy array"
            assert (
                len(epoch[key]) == simulator.n_samples
            ), f"{key} should have correct length"

    def test_generate_epoch_with_stimulus(self):
        """Test epoch generation with stimulus."""
        simulator = EEGSimulator()

        epoch = simulator.generate_epoch(stimulus_time=0.5)

        assert "p3b" in epoch, "Should include P3b when stimulus is provided"
        assert isinstance(epoch["p3b"], np.ndarray), "P3b should be numpy array"
        assert (
            len(epoch["p3b"]) == simulator.n_samples
        ), "P3b should have correct length"

    def test_generate_epoch_custom_parameters(self):
        """Test epoch generation with custom parameters."""
        simulator = EEGSimulator()

        epoch = simulator.generate_epoch(
            r_wave_amplitude=2.0,
            hep_amplitude=0.3,
            hep_modulation=1.5,
            stimulus_time=0.5,
            p3b_amplitude=0.6,
            p3b_modulation=2.0,
        )

        # Should have larger components due to increased amplitudes/modulations
        assert np.max(epoch["r_wave"]) > 1.0, "Should have larger R-wave"
        assert np.max(epoch["hep"]) > 0.2, "Should have larger HEP"
        assert "p3b" in epoch, "Should include P3b"
        assert np.max(epoch["p3b"]) > 0.5, "Should have larger P3b"

    def test_generate_epoch_signal_structure(self):
        """Test that generated epoch has expected signal structure."""
        simulator = EEGSimulator()

        epoch = simulator.generate_epoch()

        # EEG should be sum of components
        expected_eeg = epoch["r_wave"] + epoch["hep"] + epoch["background"]

        np.testing.assert_array_almost_equal(epoch["eeg"], expected_eeg, decimal=10)


class TestAnalysisFunctions:
    """Test analysis functions."""

    def test_compute_hep_amplitude(self):
        """Test HEP amplitude computation."""
        simulator = EEGSimulator()

        # Generate epoch with known HEP
        epoch = simulator.generate_epoch(hep_amplitude=0.3)

        hep_amplitude = simulator.compute_hep_amplitude(epoch["eeg"])

        assert isinstance(hep_amplitude, float), "Should return float"
        assert hep_amplitude > 0, "Should detect positive HEP amplitude"

    def test_compute_p3b_amplitude(self):
        """Test P3b amplitude computation."""
        simulator = EEGSimulator()

        # Generate epoch with P3b
        epoch = simulator.generate_epoch(stimulus_time=0.5, p3b_amplitude=0.4)

        p3b_amplitude = simulator.compute_p3b_amplitude(epoch["eeg"], 0.5)

        assert isinstance(p3b_amplitude, float), "Should return float"
        assert p3b_amplitude > 0, "Should detect positive P3b amplitude"

    def test_compute_hep_p3b_coupling(self):
        """Test HEP-P3b coupling computation."""
        simulator = EEGSimulator()

        # Test with known amplitudes
        hep_amp = 0.2
        p3b_amp = 0.4

        coupling = simulator.compute_hep_p3b_coupling(hep_amp, p3b_amp)

        assert isinstance(coupling, dict), "Should return dictionary"
        required_keys = [
            "hep_modulation",
            "p3b_modulation",
            "hep_p3b_ratio",
            "coupling_strength",
        ]
        for key in required_keys:
            assert key in coupling, f"Missing key: {key}"
            assert isinstance(coupling[key], float), f"{key} should be float"

        # Check calculations
        assert (
            coupling["hep_modulation"] == hep_amp / 0.15
        ), "HEP modulation should be ratio to baseline"
        assert (
            coupling["p3b_modulation"] == p3b_amp / 0.3
        ), "P3b modulation should be ratio to baseline"
        assert coupling["hep_p3b_ratio"] == hep_amp / p3b_amp, "Ratio should be HEP/P3b"
        assert (
            coupling["coupling_strength"]
            == coupling["hep_modulation"] * coupling["p3b_modulation"]
        ), "Coupling should be product"

    def test_simulate_cardiac_phase_alignment(self):
        """Test cardiac phase alignment simulation."""
        simulator = EEGSimulator()

        n_epochs = 50
        r_wave_times = simulator.simulate_cardiac_phase_alignment(n_epochs=n_epochs)

        assert isinstance(r_wave_times, np.ndarray), "Should return numpy array"
        assert len(r_wave_times) == n_epochs, "Should have correct number of times"
        assert np.all(r_wave_times > 0), "All times should be positive"
        assert np.all(
            np.diff(r_wave_times) > 0
        ), "Times should be monotonically increasing"

    def test_simulate_cardiac_phase_alignment_with_jitter(self):
        """Test cardiac phase alignment with jitter."""
        simulator = EEGSimulator()

        r_wave_times_no_jitter = simulator.simulate_cardiac_phase_alignment(
            n_epochs=20, cardiac_cycle_jitter=0.0
        )
        r_wave_times_jitter = simulator.simulate_cardiac_phase_alignment(
            n_epochs=20, cardiac_cycle_jitter=0.1
        )

        # With jitter, intervals should vary more
        intervals_no_jitter = np.diff(r_wave_times_no_jitter)
        intervals_jitter = np.diff(r_wave_times_jitter)

        assert np.std(intervals_jitter) > np.std(
            intervals_no_jitter
        ), "Jitter should increase variability"


class TestCreateDefaultSimulator:
    """Test default simulator creation function."""

    def test_create_default_simulator(self):
        """Test create_default_simulator function."""
        simulator = create_default_simulator()

        assert isinstance(
            simulator, EEGSimulator
        ), "Should return EEGSimulator instance"
        assert simulator.sampling_rate == 1000.0, "Should have default sampling rate"
        assert simulator.eeg_duration == 1.5, "Should have default duration"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_negative_parameters(self):
        """Test with negative parameters."""
        simulator = EEGSimulator()

        # Should handle negative amplitudes gracefully
        r_wave = simulator.generate_r_wave(amplitude=-1.0)
        assert isinstance(r_wave, np.ndarray), "Should handle negative amplitude"

    def test_zero_duration(self):
        """Test with zero duration."""
        simulator = EEGSimulator(eeg_duration=0.0)

        assert simulator.n_samples == 0, "Zero duration should produce zero samples"

        r_wave = simulator.generate_r_wave()
        assert len(r_wave) == 0, "Zero duration should produce empty signal"

    def test_very_large_parameters(self):
        """Test with very large parameters."""
        simulator = EEGSimulator(sampling_rate=10000.0, eeg_duration=10.0)

        r_wave = simulator.generate_r_wave()
        assert isinstance(r_wave, np.ndarray), "Should handle large parameters"
        assert len(r_wave) == simulator.n_samples, "Should produce correct length"

    def test_extreme_modulation(self):
        """Test with extreme modulation values."""
        simulator = EEGSimulator()

        # Very high modulation
        hep = simulator.generate_hep(r_wave_time=0.0, modulation=100.0)
        assert isinstance(hep, np.ndarray), "Should handle extreme modulation"

        # Zero modulation
        hep_zero = simulator.generate_hep(r_wave_time=0.0, modulation=0.0)
        assert np.max(hep_zero) == 0.0, "Zero modulation should produce zero signal"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
