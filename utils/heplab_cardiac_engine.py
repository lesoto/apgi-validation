"""
LEVEL DESIGNATION: Level 2 (information-theoretic)

Bridge to Level 1

HEPLAB Cardiac Engine — Python port of HEPLAB's algorithmic core
=================================================================

Ports the three detection algorithms needed by APGI's interoceptive precision
parameters (Π^i_eff via HEP amplitude) and visceral context (M(c,a) via IBI).

Ported from:
  heplab_pan_tompkin.m  — Pan-Tompkins R-wave detection
  heplab_T_detect_MTEO.m — Multi-scale Teager Energy Operator T-wave detection

Discarded (not ported):
  GUI menus, EEGLAB structure wrappers, visualization scripts.

Algorithmic steps implemented:
  Step A: Pan-Tompkins R-wave detection (structural cardiac anchors)
  Step B: MTEO T-wave localization (ventricular repolarization boundary)
  Step C: HEP epoch extraction + amplitude (Π^i_eff proxy)
  Util:   OnlineVarianceEstimator (Welford) for running Z-score normalization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CardiacEvents:
    """Timestamped cardiac events extracted from a continuous ECG."""

    r_peak_indices: np.ndarray  # sample indices
    r_peak_times_s: np.ndarray  # seconds
    t_peak_indices: np.ndarray  # one per R-peak (0 if undetected)
    t_peak_times_s: np.ndarray  # seconds
    ibi_ms: np.ndarray  # inter-beat intervals in ms (len = n_peaks - 1)
    heart_rate_bpm: float
    detection_quality: float  # 0–1; 1 = perfectly regular
    srate: float  # sampling rate used

    @property
    def n_beats(self) -> int:
        return len(self.r_peak_indices)


@dataclass
class HEPResult:
    """Per-trial HEP amplitude and APGI parameter proxies."""

    # Raw epochs: shape (n_trials, n_times)
    epochs: np.ndarray
    times_ms: np.ndarray  # epoch time axis in ms relative to R-peak

    # APGI: Π^i_eff proxy — mean amplitude in 300–500 ms post-R window
    pi_interoceptive: np.ndarray  # one scalar per trial
    pi_interoceptive_mean: float

    # APGI: M(c,a) visceral context — IBI array (ms)
    ibi_ms: np.ndarray

    # Metadata
    n_trials: int
    baseline_window_ms: Tuple[float, float]
    hep_window_ms: Tuple[float, float]


# ──────────────────────────────────────────────────────────────────────────────
# Running statistics (Welford's online algorithm)
# ──────────────────────────────────────────────────────────────────────────────


class OnlineVarianceEstimator:
    """
    Welford's one-pass running mean and variance.

    Used to Z-score prediction errors in real time without storing history:
        z[t] = (x[t] - μ[t]) / σ[t]

    Safe to call before any data has arrived (returns 0 / 1 until n ≥ 2).
    """

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._M2: float = 0.0  # sum of squared deviations

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self._M2 / (self.n - 1) if self.n > 1 else 1.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def zscore(self, x: float) -> float:
        """Return Z-score of x relative to current running distribution."""
        s = self.std
        return (x - self.mean) / s if s > 0 else 0.0

    def batch_update(self, arr: np.ndarray) -> np.ndarray:
        """
        Update with each element of arr and return the per-element Z-score
        computed *before* each update (causal / streaming).
        """
        zscores = np.empty(len(arr))
        for i, v in enumerate(arr):
            zscores[i] = self.zscore(float(v))
            self.update(float(v))
        return zscores


# ──────────────────────────────────────────────────────────────────────────────
# Step A — Pan-Tompkins R-wave detection
# ──────────────────────────────────────────────────────────────────────────────


def _bandpass_ecg(ecg: np.ndarray, srate: float) -> np.ndarray:
    """2nd-order Butterworth 5–40 Hz — mirrors HEPLAB's preprocessing."""
    nyq = srate / 2.0
    low = min(5.0 / nyq, 0.99)
    high = min(40.0 / nyq, 0.99)
    if low >= high:
        return ecg.copy()
    b, a = signal.butter(2, [low, high], btype="bandpass")
    return signal.filtfilt(b, a, ecg)


def _derivative_filter(ecg: np.ndarray) -> np.ndarray:
    """5-point derivative per Pan-Tompkins (1985)."""
    out = np.zeros_like(ecg)
    # y[n] = (1/8)(−x[n−2]−2x[n−1]+2x[n+1]+x[n+2])
    for i in range(2, len(ecg) - 2):
        out[i] = (-ecg[i - 2] - 2 * ecg[i - 1] + 2 * ecg[i + 1] + ecg[i + 2]) / 8.0
    return out


def _moving_window_integration(squared: np.ndarray, srate: float, window_ms: float = 150.0) -> np.ndarray:
    """Uniform moving average over a physiological integration window."""
    win = max(1, int(round(window_ms * srate / 1000.0)))
    kernel = np.ones(win) / win
    return np.convolve(squared, kernel, mode="same")


def pan_tompkins_rwave_detect(
    ecg: np.ndarray,
    srate: float = 1000.0,
    min_rr_ms: float = 400.0,  # floor: ~150 BPM ceiling
    integration_window_ms: float = 150.0,
) -> np.ndarray:
    """
    Pan-Tompkins (1985) R-wave detection.

    Pipeline mirrors heplab_pan_tompkin.m:
      1. 2nd-order Butterworth bandpass 5–40 Hz
      2. 5-point derivative
      3. Element-wise squaring (nonlinear energy)
      4. Moving-window integration (150 ms)
      5. Adaptive threshold peak detection

    Parameters
    ----------
    ecg : 1-D float array — raw ECG voltage
    srate : sampling rate in Hz
    min_rr_ms : minimum physiologically valid RR interval (ms)
    integration_window_ms : moving-average window width (ms)

    Returns
    -------
    r_peak_indices : 1-D int array of R-peak sample positions
    """
    if len(ecg) < int(srate * 0.5):
        logger.warning("ECG too short for Pan-Tompkins (< 0.5 s).")
        return np.array([], dtype=int)

    # --- Preprocessing ---
    filtered = _bandpass_ecg(ecg, srate)
    differentiated = _derivative_filter(filtered)
    squared = differentiated**2
    integrated = _moving_window_integration(squared, srate, integration_window_ms)

    # --- Adaptive threshold initialisation (first 2-second learning window) ---
    learn_samples = min(int(2 * srate), len(integrated))
    spki = float(np.max(integrated[:learn_samples]))  # signal peak estimate
    npki = float(np.mean(integrated[:learn_samples]))  # noise peak estimate
    threshold_i1 = npki + 0.25 * (spki - npki)
    threshold_i2 = 0.5 * threshold_i1

    min_distance = max(1, int(min_rr_ms * srate / 1000.0))

    # --- First pass: peaks above threshold_i1 ---
    raw_peaks, _ = signal.find_peaks(integrated, height=threshold_i1, distance=min_distance)

    if len(raw_peaks) == 0:
        # Fall back to lower threshold
        raw_peaks, _ = signal.find_peaks(integrated, height=threshold_i2, distance=min_distance)

    if len(raw_peaks) == 0:
        logger.warning("Pan-Tompkins found no R-peaks (signal may be flat).")
        return np.array([], dtype=int)

    # --- Second pass: refine to raw ECG local maxima within ±search_ms ---
    search_ms = 40  # ms around integrated peak to find true ECG peak
    search_samp = max(1, int(search_ms * srate / 1000.0))
    r_peaks = []
    for p in raw_peaks:
        lo = max(0, p - search_samp)
        hi = min(len(ecg) - 1, p + search_samp)
        local_max = lo + int(np.argmax(ecg[lo : hi + 1]))
        r_peaks.append(local_max)

    r_peaks_arr = np.unique(np.array(r_peaks, dtype=int))

    # --- Enforce minimum distance after refinement ---
    if len(r_peaks_arr) > 1:
        keep = [r_peaks_arr[0]]
        for idx in r_peaks_arr[1:]:
            if idx - keep[-1] >= min_distance:
                keep.append(idx)
        r_peaks_arr = np.array(keep, dtype=int)

    logger.info("Pan-Tompkins: %d R-peaks detected at %.0f Hz", len(r_peaks_arr), srate)
    return r_peaks_arr


# ──────────────────────────────────────────────────────────────────────────────
# Step B — MTEO T-wave detection
# ──────────────────────────────────────────────────────────────────────────────


def _teager_energy_operator(x: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Generalised Teager Energy Operator at lag k.
    ψ_k[n] = x[n]^2 − x[n−k] · x[n+k]
    """
    n = len(x)
    out = np.zeros(n)
    for i in range(k, n - k):
        out[i] = x[i] ** 2 - x[i - k] * x[i + k]
    return out


def _mteo(x: np.ndarray, scales: Tuple[int, ...] = (1, 2, 3, 4)) -> np.ndarray:
    """
    Multi-scale Teager Energy Operator (MTEO).
    Returns the pointwise maximum across all lag scales — same as HEPLAB.
    """
    teo_stack = np.stack([_teager_energy_operator(x, k) for k in scales], axis=0)
    return np.max(teo_stack, axis=0)


def mteo_twave_detect(
    ecg: np.ndarray,
    r_peak_indices: np.ndarray,
    srate: float = 1000.0,
    search_start_ms: float = 200.0,  # earliest T-wave (ms post-R)
    search_end_ms: float = 500.0,  # latest T-wave (ms post-R)
    mteo_scales: Tuple[int, ...] = (1, 2, 3, 4),
) -> np.ndarray:
    """
    MTEO-based T-wave peak detection.

    For each R-peak, searches the physiological window [200, 500] ms post-R
    for the MTEO energy maximum — mirrors heplab_T_detect_MTEO.m.

    Parameters
    ----------
    ecg : raw ECG (1-D)
    r_peak_indices : output of pan_tompkins_rwave_detect()
    srate : sampling rate in Hz
    search_start_ms / search_end_ms : T-wave search window (ms post-R)
    mteo_scales : lag scales for MTEO

    Returns
    -------
    t_peak_indices : 1-D int array, same length as r_peak_indices.
                     Zero (0) when window falls outside signal bounds.
    """
    # Bandpass 1–30 Hz to isolate T-wave (removes QRS residual and noise)
    nyq = srate / 2.0
    low = min(1.0 / nyq, 0.99)
    high = min(30.0 / nyq, 0.99)
    if low < high:
        b, a = signal.butter(2, [low, high], btype="bandpass")
        ecg_filt = signal.filtfilt(b, a, ecg)
    else:
        ecg_filt = ecg.copy()

    mteo_signal = _mteo(ecg_filt, mteo_scales)

    start_samp = max(1, int(search_start_ms * srate / 1000.0))
    end_samp = int(search_end_ms * srate / 1000.0)

    t_peaks = np.zeros(len(r_peak_indices), dtype=int)
    for i, r in enumerate(r_peak_indices):
        lo = r + start_samp
        hi = min(len(ecg) - 1, r + end_samp)
        if lo >= len(ecg) or lo >= hi:
            continue
        local_max_rel = int(np.argmax(mteo_signal[lo : hi + 1]))
        t_peaks[i] = lo + local_max_rel

    return t_peaks


# ──────────────────────────────────────────────────────────────────────────────
# IBI computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_ibi(r_peak_indices: np.ndarray, srate: float) -> np.ndarray:
    """
    Inter-Beat Intervals in milliseconds.

    IBI[n] = (R[n+1] − R[n]) / srate × 1000

    Directly maps to APGI somatic marker context M(c, a).
    """
    if len(r_peak_indices) < 2:
        return np.array([], dtype=float)
    return np.diff(r_peak_indices.astype(float)) / srate * 1000.0


# ──────────────────────────────────────────────────────────────────────────────
# High-level: detect all cardiac events from raw ECG
# ──────────────────────────────────────────────────────────────────────────────


def detect_cardiac_events(
    ecg: np.ndarray,
    srate: float = 1000.0,
    min_rr_ms: float = 400.0,
    t_search_start_ms: float = 200.0,
    t_search_end_ms: float = 500.0,
) -> CardiacEvents:
    """
    Full HEPLAB pipeline: R-peaks → T-peaks → IBI.

    Parameters
    ----------
    ecg : raw 1-D ECG voltage array
    srate : sampling rate in Hz
    min_rr_ms : refractory period (ms); sets maximum detectable HR
    t_search_start_ms / t_search_end_ms : T-wave search window post-R (ms)

    Returns
    -------
    CardiacEvents dataclass with all timestamps and IBI array.
    """
    r_idx = pan_tompkins_rwave_detect(ecg, srate, min_rr_ms)

    if len(r_idx) == 0:
        return CardiacEvents(
            r_peak_indices=np.array([], dtype=int),
            r_peak_times_s=np.array([]),
            t_peak_indices=np.array([], dtype=int),
            t_peak_times_s=np.array([]),
            ibi_ms=np.array([]),
            heart_rate_bpm=0.0,
            detection_quality=0.0,
            srate=srate,
        )

    t_idx = mteo_twave_detect(ecg, r_idx, srate, t_search_start_ms, t_search_end_ms)
    ibi = compute_ibi(r_idx, srate)

    hr = 60000.0 / float(np.mean(ibi)) if len(ibi) > 0 else 0.0
    cv = float(np.std(ibi) / np.mean(ibi)) if len(ibi) > 0 and np.mean(ibi) > 0 else 1.0
    quality = float(np.clip(1.0 - cv, 0.0, 1.0))

    return CardiacEvents(
        r_peak_indices=r_idx,
        r_peak_times_s=r_idx.astype(float) / srate,
        t_peak_indices=t_idx,
        t_peak_times_s=np.where(t_idx > 0, t_idx.astype(float) / srate, 0.0),
        ibi_ms=ibi,
        heart_rate_bpm=hr,
        detection_quality=quality,
        srate=srate,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step C — HEP epoch extraction and Π^i amplitude
# ──────────────────────────────────────────────────────────────────────────────


def extract_hep_epochs(
    eeg_channel: np.ndarray,
    r_peak_indices: np.ndarray,
    srate: float = 1000.0,
    epoch_start_ms: float = -200.0,  # pre-R baseline start
    epoch_end_ms: float = 600.0,  # post-R epoch end
    baseline_start_ms: float = -100.0,
    baseline_end_ms: float = 0.0,
    hep_amplitude_start_ms: float = 250.0,  # Π^i canonical window 250–400 ms per APGI spec
    hep_amplitude_end_ms: float = 400.0,
) -> HEPResult:
    """
    Extract HEP epochs from a concurrent EEG channel (typically Cz or Fz).

    For each R-peak:
      1. Extract epoch [epoch_start_ms, epoch_end_ms] relative to R.
      2. Baseline-correct using mean of [baseline_start_ms, baseline_end_ms].
      3. Compute mean amplitude in [hep_amplitude_start_ms, hep_amplitude_end_ms]
         → this scalar is the trial-by-trial Π^i_baseline (interoceptive precision proxy).

    Parameters
    ----------
    eeg_channel : 1-D EEG signal (µV, single channel — Cz or Fz recommended)
    r_peak_indices : from detect_cardiac_events() or pan_tompkins_rwave_detect()
    srate : sampling rate in Hz (must match eeg_channel)
    epoch_start_ms ... hep_amplitude_end_ms : window boundaries in ms

    Returns
    -------
    HEPResult with epochs array, per-trial Π^i_baseline, and IBI array.
    """
    n_start = int(round(epoch_start_ms * srate / 1000.0))
    n_end = int(round(epoch_end_ms * srate / 1000.0))
    n_bl_start = int(round(baseline_start_ms * srate / 1000.0))
    n_bl_end = int(round(baseline_end_ms * srate / 1000.0))
    n_hep_start = int(round(hep_amplitude_start_ms * srate / 1000.0))
    n_hep_end = int(round(hep_amplitude_end_ms * srate / 1000.0))

    epoch_len = n_end - n_start
    times_ms = np.linspace(epoch_start_ms, epoch_end_ms, epoch_len, endpoint=False)

    good_epochs = []
    for r in r_peak_indices:
        lo = r + n_start
        hi = r + n_end
        if lo < 0 or hi > len(eeg_channel):
            continue
        epoch = eeg_channel[lo:hi].astype(float).copy()

        # Baseline correction
        bl_lo = -n_start + n_bl_start  # index within epoch array
        bl_hi = -n_start + n_bl_end
        bl_lo = max(0, bl_lo)
        bl_hi = min(epoch_len, bl_hi)
        if bl_hi > bl_lo:
            baseline_mean = float(np.mean(epoch[bl_lo:bl_hi]))
            epoch -= baseline_mean

        good_epochs.append(epoch)

    if not good_epochs:
        empty = np.zeros((0, epoch_len))
        return HEPResult(
            epochs=empty,
            times_ms=times_ms,
            pi_interoceptive=np.array([]),
            pi_interoceptive_mean=0.0,
            ibi_ms=compute_ibi(r_peak_indices, srate),
            n_trials=0,
            baseline_window_ms=(baseline_start_ms, baseline_end_ms),
            hep_window_ms=(hep_amplitude_start_ms, hep_amplitude_end_ms),
        )

    epochs_arr = np.array(good_epochs)  # (n_trials, epoch_len)

    # Π^i_eff: mean amplitude in HEP window per trial
    amp_lo = -n_start + n_hep_start
    amp_hi = -n_start + n_hep_end
    amp_lo = max(0, amp_lo)
    amp_hi = min(epoch_len, amp_hi)

    if amp_hi > amp_lo:
        pi_vals = np.mean(epochs_arr[:, amp_lo:amp_hi], axis=1)
    else:
        pi_vals = np.zeros(len(good_epochs))

    ibi = compute_ibi(r_peak_indices, srate)

    return HEPResult(
        epochs=epochs_arr,
        times_ms=times_ms,
        pi_interoceptive=pi_vals,
        pi_interoceptive_mean=float(np.mean(pi_vals)),
        ibi_ms=ibi,
        n_trials=len(good_epochs),
        baseline_window_ms=(baseline_start_ms, baseline_end_ms),
        hep_window_ms=(hep_amplitude_start_ms, hep_amplitude_end_ms),
    )
