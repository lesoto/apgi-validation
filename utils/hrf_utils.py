"""
hrf_utils.py
============

Shared hemodynamic response function (HRF) utilities for fMRI protocols.

This module provides the canonical double-gamma HRF function used across
VP-14 (fMRI Anticipation/Experience) and VP-15 (fMRI vmPFC Anticipation)
to ensure consistency and avoid code duplication.

The double-gamma HRF follows SPM/FSL canonical implementation based on:
Friston, K. J., Fletcher, P., Josephs, O., Holmes, A., Rugg, M. D.,
& Turner, R. (1998). Event-related fMRI: characterizing differential
responses. NeuroImage, 7(1), 30-40.
"""

import math

import numpy as np

# Import APGI constants for HRF parameters
try:
    from utils.constants import (HRF_DISPERSION, HRF_PEAK1_SECONDS,
                                 HRF_UNDERSHOOT_RATIO, HRF_UNDERSHOOT_SECONDS)
except ImportError:
    # Fallback defaults if constants not available
    HRF_PEAK1_SECONDS = 6.0
    HRF_UNDERSHOOT_SECONDS = 16.0
    HRF_DISPERSION = 1.0
    HRF_UNDERSHOOT_RATIO = 1.0 / 6.0


def double_gamma_hrf(
    t: np.ndarray,
    response_peak_delay_s: float = HRF_PEAK1_SECONDS,
    undershoot_delay_s: float = HRF_UNDERSHOOT_SECONDS,
    response_dispersion_s: float = HRF_DISPERSION,
    undershoot_dispersion_s: float = HRF_DISPERSION,
    undershoot_ratio: float = HRF_UNDERSHOOT_RATIO,
) -> np.ndarray:
    """
    Canonical SPM/FSL-style double-gamma hemodynamic response function.

    Parameters follow the standard canonical HRF settings:
    - peak1 = 6.0s (response peak delay, main gamma)
    - peak2 = 16.0s (undershoot peak delay)
    - dispersion = 1.0s (response/undershoot dispersion)
    - ratio = 6.0 (undershoot magnitude ratio = 1/6)

    Citation: Friston, K. J., Fletcher, P., Josephs, O., Holmes, A., Rugg, M. D.,
              & Turner, R. (1998). Event-related fMRI: characterizing differential
              responses. NeuroImage, 7(1), 30-40.

    Args:
        t: Time points in seconds (numpy array)
        response_peak_delay_s: Peak response delay (default: 6.0s)
        undershoot_delay_s: Undershoot peak delay (default: 16.0s)
        response_dispersion_s: Response dispersion (default: 1.0s)
        undershoot_dispersion_s: Undershoot dispersion (default: 1.0s)
        undershoot_ratio: Undershoot magnitude ratio (default: 1/6)

    Returns:
        Normalized HRF values at time points t (peak = 1.0)
    """
    # Avoid 0^0 by clipping t
    t_safe = np.clip(t, 1e-8, None)

    hrf = (
        t_safe ** (response_peak_delay_s - 1)
        * np.exp(-t_safe / response_dispersion_s)
        / (
            response_dispersion_s**response_peak_delay_s
            * math.factorial(int(response_peak_delay_s) - 1)
        )
    ) - undershoot_ratio * (
        t_safe ** (undershoot_delay_s - 1)
        * np.exp(-t_safe / undershoot_dispersion_s)
        / (
            undershoot_dispersion_s**undershoot_delay_s
            * math.factorial(int(undershoot_delay_s) - 1)
        )
    )
    hrf[t <= 0] = 0.0
    return hrf / np.max(hrf)


def compute_hrf_convolution(
    neural_signal: np.ndarray,
    dt: float,
    hrf_duration: float = 25.0,
) -> np.ndarray:
    """
    Convolve a neural signal with the double-gamma HRF.

    Args:
        neural_signal: Neural activity time series
        dt: Sampling interval in seconds
        hrf_duration: Duration of HRF kernel in seconds (default: 25.0)

    Returns:
        BOLD signal after HRF convolution
    """
    from scipy.signal import convolve

    hrf_t = np.arange(0, hrf_duration, dt)
    hrf = double_gamma_hrf(hrf_t)

    bold_signal = convolve(neural_signal, hrf, mode="full")[: len(neural_signal)]
    return bold_signal


def get_hrf_parameters() -> dict:
    """
    Get the current HRF parameters as a dictionary.

    Returns:
        Dictionary with HRF parameter names and values
    """
    return {
        "response_peak_delay_s": HRF_PEAK1_SECONDS,
        "undershoot_delay_s": HRF_UNDERSHOOT_SECONDS,
        "response_dispersion_s": HRF_DISPERSION,
        "undershoot_dispersion_s": HRF_DISPERSION,
        "undershoot_ratio": HRF_UNDERSHOOT_RATIO,
    }


def estimate_power_analysis_params(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Estimate required sample size for given effect size, alpha, and power.

    Uses standard power analysis formula for one-sample t-test.
    For Pearson correlation r, effect_size = r.

    Args:
        effect_size: Expected effect size (e.g., correlation r)
        alpha: Significance level (default: 0.05)
        power: Desired statistical power (default: 0.80)

    Returns:
        Required sample size N
    """
    from scipy import stats

    # Convert correlation r to Cohen's d approximation
    # For r: d ≈ 2*r / sqrt(1-r^2)
    if abs(effect_size) < 1.0:
        cohens_d = 2 * effect_size / np.sqrt(1 - effect_size**2)
    else:
        cohens_d = effect_size

    # Standard normal quantiles
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Sample size formula for one-sample test
    n = ((z_alpha + z_beta) / cohens_d) ** 2

    # Add 10% buffer and round up
    return int(np.ceil(n * 1.1))


__all__ = [
    "double_gamma_hrf",
    "compute_hrf_convolution",
    "get_hrf_parameters",
    "estimate_power_analysis_params",
]
