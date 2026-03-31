"""
Real EEG Signal Processing
=========================

Utility functions for real EEG signal processing to replace synthetic generation
in VP-1. Implements gamma detection, theta-gamma PAC, and P3 amplitude analysis
following the specifications in TODO.md.

Functions:
- detect_gamma_band_power: Welch PSD with Simpson integration over 30-80 Hz
- compute_theta_gamma_pac: Modulation Index (Tort et al., 2010) for theta-gamma coupling
- detect_p3_amplitude: Bandpass filtering and peak detection for P3b component
- get_pac_bands: Load PAC band configuration from config file
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid  # Use trapezoid instead of deprecated simps
from typing import Dict, Any, Tuple
from pathlib import Path
import yaml


def get_pac_bands() -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Load PAC band configuration from default.yaml config file.

    Returns:
        Dictionary with PAC band specifications for different level boundaries
    """
    try:
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            pac_bands = config.get("pac_bands", {})

        # Convert lists to tuples for consistency
        for level, bands in pac_bands.items():
            if "phase" in bands and isinstance(bands["phase"], list):
                bands["phase"] = tuple(bands["phase"])
            if "amplitude" in bands and isinstance(bands["amplitude"], list):
                bands["amplitude"] = tuple(bands["amplitude"])

        return pac_bands
    except Exception:
        # Fallback configuration if file not found
        return {
            "L1_L2": {"phase": (4, 8), "amplitude": (30, 80)},
            "L2_L3": {"phase": (1, 4), "amplitude": (4, 8)},
            "L3_L4": {"phase": (1, 4), "amplitude": (4, 8)},
        }


def detect_gamma_band_power(
    eeg_data: np.ndarray,
    fs: float = 1000.0,
    gamma_band: Tuple[float, float] = (30.0, 80.0),
    n_permutations: int = 1000,
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """
    Detect gamma band power using Welch PSD with Simpson integration.

    Args:
        eeg_data: EEG data array (channels × timepoints or timepoints)
        fs: Sampling frequency in Hz
        gamma_band: Frequency range for gamma band (low, high) in Hz
        n_permutations: Number of permutations for significance testing
        alpha: Significance level for permutation test

    Returns:
        Dictionary with:
        - band_power: Total gamma band power
        - normalized_power: Power normalized by total power
        - p_value: Permutation test p-value
        - is_significant: Whether gamma power is significant
    """
    if fs <= 0:
        raise ValueError("Sampling rate must be positive")
    # Ensure 2D array
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)

    n_channels, n_samples = eeg_data.shape

    # Handle empty data gracefully
    if n_samples == 0:
        return {
            "band_power": 0.0,
            "normalized_power": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "gamma_band": gamma_band,
        }

    total_power = 0.0
    gamma_power = 0.0

    for ch in range(n_channels):
        # Compute PSD using Welch's method
        f, Pxx = signal.welch(eeg_data[ch], fs=fs, nperseg=min(256, n_samples // 4))

        # Integrate gamma band power using Simpson's rule
        gamma_idx = (f >= gamma_band[0]) & (f <= gamma_band[1])
        if np.any(gamma_idx):
            gamma_power_ch = trapezoid(Pxx[gamma_idx], f[gamma_idx])
            gamma_power += gamma_power_ch

        # Total power
        total_power_ch = trapezoid(Pxx, f)
        total_power += total_power_ch

    # Normalize
    normalized_power = gamma_power / max(total_power, 1e-10)

    # Permutation test for significance
    p_value = _permutation_test_gamma(
        eeg_data, fs, gamma_band, n_permutations, gamma_power
    )

    is_significant = p_value < alpha

    return {
        "band_power": gamma_power,
        "normalized_power": normalized_power,
        "p_value": p_value,
        "is_significant": is_significant,
        "gamma_band": gamma_band,
    }


def _permutation_test_gamma(
    eeg_data: np.ndarray,
    fs: float,
    gamma_band: Tuple[float, float],
    n_permutations: int,
    observed_power: float,
) -> float:
    """
    Permutation test for gamma band power significance.

    Randomly shuffles phase information while preserving amplitude spectrum
    to test if observed gamma power is greater than expected by chance.
    """
    n_channels, n_samples = eeg_data.shape
    permuted_powers = []

    for _ in range(n_permutations):
        # Randomize phases while preserving amplitudes
        fft_data = np.fft.rfft(eeg_data, axis=1)
        phases = np.angle(fft_data)
        np.random.shuffle(phases)
        fft_data_permuted = np.abs(fft_data) * np.exp(1j * phases)
        eeg_permuted = np.fft.irfft(fft_data_permuted, n=n_samples, axis=1)

        # Compute gamma power for permuted data
        gamma_power_perm = 0.0
        for ch in range(n_channels):
            f, Pxx = signal.welch(
                eeg_permuted[ch], fs=fs, nperseg=min(256, n_samples // 4)
            )
            gamma_idx = (f >= gamma_band[0]) & (f <= gamma_band[1])
            if np.any(gamma_idx):
                gamma_power_perm += trapezoid(Pxx[gamma_idx], f[gamma_idx])

        permuted_powers.append(gamma_power_perm)

    # Count how many permuted powers exceed observed
    n_greater = sum(p > observed_power for p in permuted_powers)
    p_value = (n_greater + 1) / (n_permutations + 1)

    return p_value


def compute_pac_with_bands(
    eeg_data: np.ndarray,
    fs: float = 1000.0,
    phase_band: Tuple[float, float] = (4.0, 8.0),
    amplitude_band: Tuple[float, float] = (30.0, 80.0),
    n_permutations: int = 1000,
    alpha: float = 0.01,
    level_boundary: str = "L1_L2",
) -> Dict[str, Any]:
    """
    Compute phase-amplitude coupling using configurable frequency bands.

    Args:
        eeg_data: EEG data array (channels × timepoints or timepoints)
        fs: Sampling frequency in Hz
        phase_band: Frequency range for phase band (low, high) in Hz
        amplitude_band: Frequency range for amplitude band (low, high) in Hz
        n_permutations: Number of permutations for significance testing
        alpha: Significance level for permutation test
        level_boundary: Which hierarchical level boundary this PAC represents

    Returns:
        Dictionary with PAC metrics and band information
    """
    # Ensure 2D array
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)

    n_channels, n_samples = eeg_data.shape

    # Filter phase and amplitude bands
    phase_filtered = _bandpass_filter(eeg_data, fs, phase_band)
    amplitude_filtered = _bandpass_filter(eeg_data, fs, amplitude_band)

    # Extract phase from phase band
    if phase_filtered.ndim == 1:
        phase = np.angle(signal.hilbert(phase_filtered))
    else:
        phase = np.angle(signal.hilbert(phase_filtered, axis=1))

    # Extract amplitude envelope from amplitude band
    amplitude_envelope = _amplitude_envelope(amplitude_filtered, fs)

    # Compute Modulation Index for each channel
    mi_values = []
    n_bins = 18  # 18 bins for 0-360 degrees
    phase_bins = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)

    for ch in range(n_channels):
        # Compute amplitude for each phase bin
        amp_by_phase = []
        for i in range(n_bins):
            next_bin_idx = (i + 1) % n_bins  # Wrap around for last bin
            in_bin = (phase[ch] >= phase_bins[i]) & (
                phase[ch] < phase_bins[next_bin_idx]
            )
            if np.any(in_bin):
                amp = amplitude_envelope[ch, in_bin]
                amp_by_phase.append(np.mean(amp) if len(amp) > 0 else 0.0)
            else:
                amp_by_phase.append(0.0)

        # Normalize amplitudes
        amp_by_phase = np.array(amp_by_phase)
        if np.sum(amp_by_phase) > 0:
            amp_by_phase = amp_by_phase / np.sum(amp_by_phase)

        # Compute MI using KL divergence from uniform distribution
        uniform_dist = np.ones(n_bins) / n_bins
        mi = np.sum(amp_by_phase * np.log(amp_by_phase / uniform_dist + 1e-10))
        mi_values.append(mi)

    # Mean MI across channels
    modulation_index = np.mean(mi_values)

    # Phase and amplitude amplitudes
    phase_amplitude = np.mean(np.abs(phase_filtered))
    amplitude_amplitude = np.mean(amplitude_envelope)

    # Permutation test (only if n_permutations > 1)
    if n_permutations > 1:
        p_value = _permutation_test_pac(
            eeg_data, fs, phase_band, amplitude_band, n_permutations, modulation_index
        )
    else:
        p_value = 1.0

    is_significant = p_value < alpha

    return {
        "modulation_index": modulation_index,
        "p_value": p_value,
        "is_significant": is_significant,
        "phase_amplitude": phase_amplitude,
        "amplitude_amplitude": amplitude_amplitude,
        "phase_band": phase_band,
        "amplitude_band": amplitude_band,
        "level_boundary": level_boundary,
        "description": f"{phase_band[0]}-{phase_band[1]}Hz phase × {amplitude_band[0]}-{amplitude_band[1]}Hz amplitude coupling",
    }


def compute_theta_gamma_pac(
    eeg_data: np.ndarray,
    fs: float = 1000.0,
    theta_band: Tuple[float, float] = (4.0, 8.0),
    gamma_band: Tuple[float, float] = (30.0, 80.0),
    n_permutations: int = 1000,
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """
    Compute theta-gamma phase-amplitude coupling using Modulation Index (Tort et al., 2010).

    Args:
        eeg_data: EEG data array (channels × timepoints or timepoints)
        fs: Sampling frequency in Hz
        theta_band: Frequency range for theta band (low, high) in Hz
        gamma_band: Frequency range for gamma band (low, high) in Hz
        n_permutations: Number of permutations for significance testing
        alpha: Significance level for permutation test

    Returns:
        Dictionary with:
        - modulation_index: Modulation Index value
        - p_value: Permutation test p-value
        - is_significant: Whether PAC is significant
        - theta_amplitude: Mean theta amplitude
        - gamma_amplitude: Mean gamma amplitude
    """
    # Ensure 2D array
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)

    n_channels, n_samples = eeg_data.shape

    # Filter theta and gamma bands
    theta_filtered = _bandpass_filter(eeg_data, fs, theta_band)
    gamma_filtered = _bandpass_filter(eeg_data, fs, gamma_band)

    # Extract amplitude envelopes
    theta_envelope = _amplitude_envelope(theta_filtered, fs)
    gamma_envelope = _amplitude_envelope(gamma_filtered, fs)

    # Compute phase of theta
    if theta_filtered.ndim == 1:
        theta_phase = np.angle(signal.hilbert(theta_filtered))
    else:
        theta_phase = np.angle(signal.hilbert(theta_filtered, axis=1))

    # Bin theta phases into bins
    n_bins = 18  # 18 bins for 0-360 degrees
    theta_bins = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)

    # Compute MI for each channel
    mi_values = []
    for ch in range(n_channels):
        # Compute gamma amplitude for each theta phase bin
        gamma_amp_by_phase = []
        for i in range(n_bins):
            # Find time points where theta phase is in this bin
            next_bin_idx = (i + 1) % n_bins  # Wrap around for last bin
            in_bin = (theta_phase[ch] >= theta_bins[i]) & (
                theta_phase[ch] < theta_bins[next_bin_idx]
            )
            if np.any(in_bin):
                gamma_amp = gamma_envelope[ch, in_bin]
                # Compute mean gamma amplitude in this phase bin
                gamma_amp_by_phase.append(
                    np.mean(gamma_amp) if len(gamma_amp) > 0 else 0.0
                )
            else:
                gamma_amp_by_phase.append(0.0)

        # Normalize gamma amplitudes
        gamma_amp_by_phase = np.array(gamma_amp_by_phase)
        if np.sum(gamma_amp_by_phase) > 0:
            gamma_amp_by_phase = gamma_amp_by_phase / np.sum(gamma_amp_by_phase)

        # Compute MI using KL divergence from uniform distribution
        uniform_dist = np.ones(n_bins) / n_bins
        mi = np.sum(
            gamma_amp_by_phase * np.log(gamma_amp_by_phase / uniform_dist + 1e-10)
        )

        mi_values.append(mi)

    # Mean MI across channels
    modulation_index = np.mean(mi_values)

    # Theta and gamma amplitudes
    theta_amplitude = np.mean(theta_envelope)
    gamma_amplitude = np.mean(gamma_envelope)

    # Permutation test (only if n_permutations > 1)
    if n_permutations > 1:
        p_value = _permutation_test_pac(
            eeg_data, fs, theta_band, gamma_band, n_permutations, modulation_index
        )
    else:
        # For single computation, return nominal p-value
        p_value = 1.0

    is_significant = p_value < alpha

    return {
        "modulation_index": modulation_index,
        "p_value": p_value,
        "is_significant": is_significant,
        "theta_amplitude": theta_amplitude,
        "gamma_amplitude": gamma_amplitude,
        "theta_band": theta_band,
        "gamma_band": gamma_band,
    }


def _bandpass_filter(
    data: pd.Series,
    fs: float,
    band: Tuple[float, float],
    order: int = 4,
) -> pd.Series:
    """Apply Butterworth bandpass filter to data."""
    # Convert Series to numpy array for scipy
    data_array = data.values if isinstance(data, pd.Series) else data

    nyquist = fs / 2
    low = band[0] / nyquist
    high = band[1] / nyquist
    b, a = signal.butter(order, [low, high], btype="band")

    # Handle both 1D and 2D data
    if data_array.ndim == 1:
        # 1D data - use axis=0
        filtered_data = signal.filtfilt(b, a, data_array, axis=0)
    else:
        # 2D data - use axis=1 (channels along rows)
        filtered_data = signal.filtfilt(b, a, data_array, axis=1)

    # Preserve original shape - don't flatten 2D data even if it has only one channel
    # Return appropriate type based on input
    if isinstance(data, pd.Series):
        return pd.Series(filtered_data, index=data.index, name=data.name)
    else:
        return filtered_data


def _amplitude_envelope(
    data: pd.Series,
    fs: float,
) -> pd.Series:
    """Extract amplitude envelope using Hilbert transform."""
    # Convert Series to numpy array for scipy
    data_array = data.values if isinstance(data, pd.Series) else data

    # Handle both 1D and 2D data
    if data_array.ndim == 1:
        # 1D data - use axis=0
        analytic = signal.hilbert(data_array, axis=0)
    else:
        # 2D data - use axis=1 (channels along rows)
        analytic = signal.hilbert(data_array, axis=1)

    envelope = np.abs(analytic)

    # Preserve original shape - don't flatten 2D data even if it has only one channel
    # Return appropriate type based on input
    if isinstance(data, pd.Series):
        return pd.Series(envelope, index=data.index, name=data.name)
    else:
        return envelope


def _permutation_test_pac(
    eeg_data: np.ndarray,
    fs: float,
    theta_band: Tuple[float, float],
    gamma_band: Tuple[float, float],
    n_permutations: int,
    observed_mi: float,
) -> float:
    """
    Permutation test for PAC significance.

    Randomly shuffles temporal structure while preserving spectral properties.
    """
    n_channels, n_samples = eeg_data.shape
    permuted_mi_values = []

    for _ in range(n_permutations):
        # Shuffle temporal structure across all channels
        eeg_permuted = eeg_data.copy()
        for ch in range(n_channels):
            np.random.shuffle(eeg_permuted[ch])

        # Compute PAC for permuted data
        pac_result = compute_theta_gamma_pac(
            eeg_permuted, fs, theta_band, gamma_band, n_permutations=1, alpha=1.0
        )
        permuted_mi_values.append(pac_result["modulation_index"])

    # Count how many permuted MIs exceed observed
    n_greater = sum(mi > observed_mi for mi in permuted_mi_values)
    p_value = (n_greater + 1) / (n_permutations + 1)

    return p_value


def detect_p3_amplitude(
    eeg_data: np.ndarray,
    fs: float = 1000.0,
    p3_window: Tuple[float, float] = (0.3, 0.8),
    baseline_window: Tuple[float, float] = (-0.2, 0.0),
    peak_window: Tuple[float, float] = (0.3, 0.6),
    prominence: float = 0.5,
    width: Tuple[int, int] = (1, 10),
    distance: int = 20,
) -> Dict[str, Any]:
    """
    Detect P3b amplitude using bandpass filtering and peak detection.

    Args:
        eeg_data: EEG data array (channels × timepoints)
        fs: Sampling frequency in Hz
        p3_window: Time window for P3b analysis in seconds (start, end)
        baseline_window: Baseline window for correction in seconds (start, end)
        peak_window: Window for peak detection in seconds (start, end)
        prominence: Minimum peak prominence
        width: Peak width range (min, max) in samples
        distance: Minimum distance between peaks in samples

    Returns:
        Dictionary with:
        - p3_amplitude: Mean P3b amplitude after baseline correction
        - peak_amplitudes: Amplitudes of detected peaks
        - peak_times: Time points of detected peaks (in seconds)
        - n_peaks: Number of detected peaks
        - is_significant: Whether P3b is significant (at least one peak detected)
    """
    # Ensure 2D array
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)

    n_channels, n_samples = eeg_data.shape

    # Convert time windows to samples
    baseline_start = int(baseline_window[0] * fs)
    baseline_end = int(baseline_window[1] * fs)
    p3_start = int(p3_window[0] * fs)
    p3_end = int(p3_window[1] * fs)
    peak_start = int((peak_window[0] - p3_window[0]) * fs)
    peak_end = int((peak_window[1] - p3_window[0]) * fs)

    # Bandpass filter (0.5-30 Hz)
    bandpass = (0.5, 30.0)
    eeg_filtered = _bandpass_filter(eeg_data, fs, bandpass)

    # Extract Pz channel (assuming channel 31 is Pz)
    if n_channels > 31:
        pz_data = eeg_filtered[31, :]
    else:
        pz_data = eeg_filtered[0, :]

    # Baseline correction
    baseline = np.mean(pz_data[baseline_start:baseline_end])
    p3_data = pz_data[p3_start:p3_end] - baseline

    # Peak detection in 300-600ms window
    peak_data = p3_data[peak_start:peak_end]

    # Find peaks
    peaks, properties = signal.find_peaks(
        peak_data,
        prominence=prominence,
        width=width,
        distance=distance,
    )

    # Extract peak amplitudes and times
    peak_amplitudes = properties["prominences"]
    peak_times = (
        properties["left_ips"] + properties["right_ips"]
    ) / 2 / fs + p3_window[0]

    # Mean P3b amplitude
    p3_amplitude = np.mean(p3_data) if len(p3_data) > 0 else 0.0

    # Determine significance (at least one peak detected)
    is_significant = len(peaks) > 0

    return {
        "p3_amplitude": p3_amplitude,
        "peak_amplitudes": peak_amplitudes,
        "peak_times": peak_times,
        "n_peaks": len(peaks),
        "is_significant": is_significant,
        "p3_window": p3_window,
        "baseline_window": baseline_window,
        "peak_window": peak_window,
    }


def compute_all_pac_bands(
    eeg_data: np.ndarray,
    fs: float = 1000.0,
    n_permutations: int = 1000,
    alpha: float = 0.01,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute phase-amplitude coupling for all configured level boundaries.

    Args:
        eeg_data: EEG data array (channels × timepoints or timepoints)
        fs: Sampling frequency in Hz
        n_permutations: Number of permutations for significance testing
        alpha: Significance level for permutation test

    Returns:
        Dictionary with PAC results for each level boundary
    """
    pac_bands = get_pac_bands()
    results = {}

    for level_boundary, bands in pac_bands.items():
        phase_band = tuple(bands["phase"])
        amplitude_band = tuple(bands["amplitude"])

        result = compute_pac_with_bands(
            eeg_data=eeg_data,
            fs=fs,
            phase_band=phase_band,
            amplitude_band=amplitude_band,
            n_permutations=n_permutations,
            alpha=alpha,
            level_boundary=level_boundary,
        )

        results[level_boundary] = result

    return results


def process_real_eeg(
    eeg_data: np.ndarray,
    fs: float = 1000.0,
) -> Dict[str, Any]:
    """
    Process real EEG data with all three analyses.

    Args:
        eeg_data: EEG data array (channels × timepoints)
        fs: Sampling frequency in Hz

    Returns:
        Dictionary with gamma, PAC, and P3 results
    """
    results = {
        "gamma": detect_gamma_band_power(eeg_data, fs),
        "pac": compute_theta_gamma_pac(eeg_data, fs),
        "pac_all": compute_all_pac_bands(eeg_data, fs),
        "p3": detect_p3_amplitude(eeg_data, fs),
    }

    return results


if __name__ == "__main__":
    # Test the functions with synthetic data

    # Generate synthetic EEG data for testing
    fs = 1000.0
    duration = 1.0
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)

    # Create synthetic EEG with gamma and theta components
    gamma_component = 5.0 * np.sin(2 * np.pi * 50 * t) * np.exp(-t / 0.5)
    theta_component = 3.0 * np.sin(2 * np.pi * 6 * t) * np.exp(-t / 0.3)
    delta_component = 2.0 * np.sin(2 * np.pi * 2 * t) * np.exp(-t / 0.4)
    p3_component = 8.0 * np.exp(-((t - 0.5) ** 2) / 0.1) * (t > 0.3) * (t < 0.8)
    noise = 0.5 * np.random.randn(n_samples)

    eeg_data = (
        gamma_component + theta_component + delta_component + p3_component + noise
    ).reshape(1, -1)

    # Process EEG
    results = process_real_eeg(eeg_data, fs)

    print("=" * 80)
    print("REAL EEG PROCESSING TEST")
    print("=" * 80)
    print("\nGamma Detection:")
    print(f"  Band power: {results['gamma']['band_power']:.4f}")
    print(f"  Normalized power: {results['gamma']['normalized_power']:.4f}")
    print(f"  P-value: {results['gamma']['p_value']:.4f}")
    print(f"  Significant: {results['gamma']['is_significant']}")

    print("\nTheta-Gamma PAC:")
    print(f"  Modulation Index: {results['pac']['modulation_index']:.4f}")
    print(f"  P-value: {results['pac']['p_value']:.4f}")
    print(f"  Significant: {results['pac']['is_significant']}")
    print(f"  Theta amplitude: {results['pac']['theta_amplitude']:.4f}")
    print(f"  Gamma amplitude: {results['pac']['gamma_amplitude']:.4f}")

    print("\nAll PAC Bands:")
    for level_boundary, pac_result in results["pac_all"].items():
        print(f"  {level_boundary}:")
        print(f"    Modulation Index: {pac_result['modulation_index']:.4f}")
        print(f"    Description: {pac_result['description']}")
        print(f"    Significant: {pac_result['is_significant']}")

    print("\nP3b Detection:")
    print(f"  P3b amplitude: {results['p3']['p3_amplitude']:.4f}")
    print(f"  Number of peaks: {results['p3']['n_peaks']}")
    print(f"  Significant: {results['p3']['is_significant']}")

    # Test PAC band configuration
    print("\nPAC Band Configuration:")
    pac_bands = get_pac_bands()
    for level, bands in pac_bands.items():
        print(
            f"  {level}: phase {bands['phase']} Hz, amplitude {bands['amplitude']} Hz"
        )

    print("=" * 80)
