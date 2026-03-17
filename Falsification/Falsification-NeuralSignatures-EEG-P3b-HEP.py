"""
Falsification Protocol 9: Neural Signatures Validation
====================================================

This protocol implements validation of neural signatures for consciousness markers.
Per Step 1.6 - Implement FP-9 real EEG signal processing.
"""

import logging
import numpy as np
from typing import Dict, Any, List
from scipy import signal
from scipy.integrate import simps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_gamma_oscillation(
    eeg_data: np.ndarray, fs: float = 1000.0
) -> Dict[str, Any]:
    """
    Detect gamma oscillation in EEG data using Welch's method.
    Per Step 1.6 - scipy.signal.welch PSD, band power via Simpson integration over 30–80 Hz,
    normalized by total power, permutation test for significance.
    """
    if len(eeg_data) < 100:
        return {"gamma_power": 0.0, "normalized_power": 0.0, "significant": False}

    try:
        # Compute power spectral density using Welch's method
        freqs, psd = signal.welch(eeg_data, fs=fs)

        # Extract gamma band power (30-80 Hz)
        gamma_mask = (freqs >= 30) & (freqs <= 80)
        gamma_freqs = freqs[gamma_mask]
        gamma_psd = psd[gamma_mask]

        # Integrate power over gamma band using Simpson's rule
        gamma_power = simps(gamma_psd, gamma_freqs)

        # Calculate total power
        total_power = simps(psd, freqs)

        # Normalize to total power
        normalized_gamma = gamma_power / total_power if total_power > 0 else 0

        # Permutation test for significance
        # Generate 1000 permutations of the data
        n_permutations = 1000
        perm_powers = []

        for _ in range(n_permutations):
            perm_data = np.random.permutation(eeg_data)
            _, perm_psd = signal.welch(perm_data, fs=fs)
            perm_gamma_psd = perm_psd[gamma_mask]
            perm_power = simps(perm_gamma_psd, gamma_freqs)
            perm_powers.append(perm_power)

        # Calculate p-value
        p_value = np.sum(perm_powers >= gamma_power) / n_permutations

        return {
            "gamma_power": float(gamma_power),
            "normalized_power": float(normalized_gamma),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n_permutations": n_permutations,
        }

    except Exception as e:
        logger.error(f"Error in gamma oscillation detection: {e}")
        return {
            "gamma_power": 0.0,
            "normalized_power": 0.0,
            "significant": False,
            "error": str(e),
        }


def detect_theta_gamma_pac(eeg_data: np.ndarray, fs: float = 1000.0) -> Dict[str, Any]:
    """
    Detect theta-gamma phase-amplitude coupling (PAC) using Modulation Index.
    Per Step 1.6 - filter theta (4–8 Hz) and gamma (30–80 Hz), compute phase-amplitude
    coupling via KL divergence from uniform phase distribution (Tort et al., 2010).
    """
    if len(eeg_data) < 200:
        return {"modulation_index": 0.0, "significant": False}

    try:
        # Filter theta band (4-8 Hz)
        theta_low, theta_high = 4.0, 8.0
        theta_b, theta_a = signal.butter(
            4, [theta_low, theta_high], btype="band", fs=fs
        )
        theta_filtered = signal.filtfilt(theta_b, theta_a, eeg_data)

        # Filter gamma band (30-80 Hz)
        gamma_low, gamma_high = 30.0, 80.0
        gamma_b, gamma_a = signal.butter(
            4, [gamma_low, gamma_high], btype="band", fs=fs
        )
        gamma_filtered = signal.filtfilt(gamma_b, gamma_a, eeg_data)

        # Extract phase from theta band using Hilbert transform
        theta_analytic = signal.hilbert(theta_filtered)
        theta_phase = np.angle(theta_analytic)

        # Extract amplitude envelope from gamma band
        gamma_envelope = np.abs(signal.hilbert(gamma_filtered))

        # Bin phase into bins for PAC calculation
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        phase_bin_indices = np.digitize(theta_phase, phase_bins)[:-1]

        # Calculate mean amplitude for each phase bin
        mean_amplitudes = []
        for i in range(n_bins):
            mask = phase_bin_indices == i
            if np.any(mask):
                mean_amp = np.mean(gamma_envelope[mask])
                mean_amplitudes.append(mean_amp)
            else:
                mean_amplitudes.append(0.0)

        mean_amplitudes = np.array(mean_amplitudes)

        # Normalize to sum to 1
        total = np.sum(mean_amplitudes)
        if total > 0:
            normalized_amplitudes = mean_amplitudes / total
        else:
            normalized_amplitudes = np.ones(n_bins) / n_bins

        # Calculate Kullback-Leibler divergence from uniform distribution
        uniform_dist = np.ones(n_bins) / n_bins
        kl_div = np.sum(
            normalized_amplitudes * np.log(normalized_amplitudes / uniform_dist)
        )

        # Calculate modulation index
        modulation_index = (kl_div - np.log(n_bins)) / np.log(n_bins)

        # Permutation test for significance
        n_permutations = 1000
        perm_mi_values = []

        for _ in range(n_permutations):
            perm_phase_indices = np.random.permutation(phase_bin_indices)
            perm_mean_amps = []
            for i in range(n_bins):
                mask = perm_phase_indices == i
                if np.any(mask):
                    perm_mean_amp = np.mean(gamma_envelope[mask])
                    perm_mean_amps.append(perm_mean_amp)
                else:
                    perm_mean_amps.append(0.0)

            perm_mean_amps = np.array(perm_mean_amps)
            perm_total = np.sum(perm_mean_amps)
            if perm_total > 0:
                perm_normalized = perm_mean_amps / perm_total
            else:
                perm_normalized = np.ones(n_bins) / n_bins

            perm_uniform = np.ones(n_bins) / n_bins
            perm_kl = np.sum(perm_normalized * np.log(perm_normalized / perm_uniform))
            perm_mi = (perm_kl - np.log(n_bins)) / np.log(n_bins)
            perm_mi_values.append(perm_mi)

        # Calculate p-value
        p_value = np.sum(perm_mi_values >= modulation_index) / n_permutations

        return {
            "modulation_index": float(modulation_index),
            "theta_power": float(np.mean(theta_filtered**2)),
            "gamma_power": float(np.mean(gamma_filtered**2)),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n_permutations": n_permutations,
        }

    except Exception as e:
        logger.error(f"Error in theta-gamma PAC detection: {e}")
        return {"modulation_index": 0.0, "significant": False, "error": str(e)}


def detect_p3_amplitude(
    eeg_data: np.ndarray,
    fs: float = 1000.0,
    stimulus_time: float = 0.0,
) -> Dict[str, Any]:
    """
    Detect P3 amplitude in EEG data.
    Per Step 1.6 - bandpass 0.5–30 Hz, epoch 0–800ms post-stimulus,
    baseline correct −200–0ms, scipy.signal.find_peaks in 300–600ms window with
    amplitude and prominence constraints.
    """
    if len(eeg_data) < 200:
        return {"p3_amplitude": 0.0, "peak_detected": False}

    try:
        # Bandpass filter 0.5-30 Hz
        low, high = 0.5, 30.0
        b, a = signal.butter(4, [low, high], btype="bandpass", fs=fs)
        filtered_eeg = signal.filtfilt(b, a, eeg_data)

        # Calculate time axis
        # time_axis = np.arange(len(filtered_eeg)) / fs  # Available if needed

        # Find stimulus onset (assume it's at the beginning)
        stimulus_onset_idx = int(stimulus_time * fs)

        # Baseline period: -200ms to 0ms before stimulus
        baseline_start = max(0, stimulus_onset_idx - int(0.2 * fs))
        baseline_end = stimulus_onset_idx
        baseline = filtered_eeg[baseline_start:baseline_end]
        baseline_mean = np.mean(baseline) if len(baseline) > 0 else 0

        # P3 window: 300-600ms post-stimulus
        p3_start = stimulus_onset_idx + int(0.3 * fs)
        p3_end = stimulus_onset_idx + int(0.6 * fs)

        if p3_end >= len(filtered_eeg):
            p3_end = len(filtered_eeg)

        p3_window = filtered_eeg[p3_start:p3_end]

        # Baseline correct
        p3_corrected = p3_window - baseline_mean

        # Find peaks in P3 window
        # prominence should be at least 2x the standard deviation of noise
        noise_std = np.std(baseline) if len(baseline) > 0 else 1.0
        peaks, properties = signal.find_peaks(
            p3_corrected, height=2 * noise_std, prominence=2 * noise_std
        )

        if len(peaks) > 0:
            # Find the peak in the middle of the P3 window (400-500ms)
            p3_center_start = int(0.4 * fs) + stimulus_onset_idx
            p3_center_end = int(0.5 * fs) + stimulus_onset_idx
            center_peaks = [
                (p, properties["peak_heights"][i])
                for i, p in enumerate(peaks)
                if p3_center_start <= p <= p3_center_end
            ]

            if center_peaks:
                # Use the highest peak in the center region
                max_peak_idx = np.argmax([h for _, h in center_peaks])
                p3_amplitude = center_peaks[max_peak_idx][1]
                peak_idx = center_peaks[max_peak_idx][0]
                peak_time = (peak_idx + p3_start) / fs
            else:
                # Use the overall maximum peak in P3 window
                max_idx = np.argmax(p3_corrected)
                p3_amplitude = p3_corrected[max_idx]
                peak_idx = max_idx
                peak_time = (peak_idx + p3_start) / fs

            return {
                "p3_amplitude": float(p3_amplitude),
                "peak_time": peak_time,
                "baseline_mean": float(baseline_mean),
                "peak_detected": True,
                "n_peaks": len(peaks),
            }
        else:
            return {
                "p3_amplitude": 0.0,
                "peak_time": 0.0,
                "baseline_mean": float(baseline_mean),
                "peak_detected": False,
                "n_peaks": 0,
            }

    except Exception as e:
        logger.error(f"Error in P3 amplitude detection: {e}")
        return {"p3_amplitude": 0.0, "peak_detected": False, "error": str(e)}


def detect_neural_signatures(
    eeg_data: np.ndarray,
    markers: List[str],
    fs: float = 1000.0,
    stimulus_time: float = 0.0,
) -> Dict[str, Any]:
    """
    Detect neural signatures in EEG data using real signal processing.
    Per Step 1.6.
    """
    signature_scores = {}

    for marker in markers:
        if marker == "gamma_oscillation":
            signature_scores[marker] = detect_gamma_oscillation(eeg_data, fs)
        elif marker == "theta_coupling":
            signature_scores[marker] = detect_theta_gamma_pac(eeg_data, fs)
        elif marker == "p3_amplitude":
            signature_scores[marker] = detect_p3_amplitude(eeg_data, fs, stimulus_time)
        else:
            signature_scores[marker] = {
                "score": 0.0,
                "significant": False,
                "error": f"Unknown marker: {marker}",
            }

    return signature_scores


def validate_consciousness_markers(
    signature_scores: Dict[str, Any], thresholds: Dict[str, float]
) -> Dict[str, bool]:
    """Validate consciousness markers against thresholds"""

    validation_results = {}

    for marker, results in signature_scores.items():
        if "error" in results:
            validation_results[marker] = False
            continue

        threshold = thresholds.get(marker, 0.5)

        if marker == "gamma_oscillation":
            # Use normalized power for validation
            score = results.get("normalized_power", 0.0)
            validation_results[marker] = score >= threshold
        elif marker == "theta_coupling":
            # Use modulation index for validation
            score = results.get("modulation_index", 0.0)
            validation_results[marker] = score >= threshold
        elif marker == "p3_amplitude":
            # Use p3_amplitude for validation
            score = results.get("p3_amplitude", 0.0)
            validation_results[marker] = score >= threshold
        else:
            validation_results[marker] = False

    return validation_results


def run_neural_signature_validation():
    """
    Run complete neural signature validation.
    Per Step 1.6 - Implement FP-9 real EEG signal processing.
    """
    logger.info("Running neural signature validation...")

    # Generate synthetic EEG data with realistic characteristics
    fs = 1000.0  # Sampling frequency
    n_samples = 2000
    time = np.arange(n_samples) / fs

    # Create synthetic EEG with gamma, theta, and P3 components
    # Gamma oscillation (30-80 Hz)
    gamma_signal = 0.5 * np.sin(2 * np.pi * 55 * time) * np.exp(-0.1 * time)

    # Theta oscillation (4-8 Hz)
    theta_signal = 0.3 * np.sin(2 * np.pi * 6 * time) * np.exp(-0.05 * time)

    # P3 component (peak at 400ms post-stimulus)
    p3_signal = np.zeros_like(time)
    p3_start = 0.4  # 400ms
    p3_width = 0.1  # 100ms width
    p3_center = p3_start + p3_width / 2
    p3_signal = (
        2.0
        * np.exp(-((time - p3_center) ** 2 / (2 * (0.05) ** 2)))
        * np.sin(2 * np.pi * 10 * time)
    )

    # Combine signals
    eeg_data = (
        gamma_signal + theta_signal + p3_signal + 0.1 * np.random.randn(len(time))
    )

    # Markers to detect
    markers = ["gamma_oscillation", "theta_coupling", "p3_amplitude"]

    # Detect signatures with real signal processing
    signature_scores = detect_neural_signatures(
        eeg_data, markers, fs, stimulus_time=0.0
    )

    # Validation thresholds (from falsification_thresholds.py)
    thresholds = {
        "gamma_oscillation": 0.15,  # Normalized gamma power threshold
        "theta_coupling": 0.05,  # Modulation index threshold
        "p3_amplitude": 0.8,  # P3 amplitude threshold
    }

    # Validate markers
    validation_results = validate_consciousness_markers(signature_scores, thresholds)

    return {
        "signature_scores": signature_scores,
        "validation_results": validation_results,
        "thresholds": thresholds,
        "eeg_data": eeg_data,
    }


if __name__ == "__main__":
    results = run_neural_signature_validation()
    print("Neural signature validation results:")
    print(f"Signature scores: {results['signature_scores']}")
    print(f"Validation results: {results['validation_results']}")
