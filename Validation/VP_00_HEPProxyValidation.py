"""
APGI Validation Protocol 0: HEP Proxy Validation (EP-0)
========================================================

Implements the foundational empirical grounding for interoceptive precision
weighting (Π^i_eff) and visceral context (M(c,a)) via HEPLAB's core algorithms.

Algorithmic components (ported from HEPLAB MATLAB toolbox):
  • Pan-Tompkins R-wave detection     → cardiac timestamps
  • MTEO T-wave detection             → ventricular repolarization boundary
  • HEP epoch extraction + amplitude  → Π^i_eff (interoceptive precision proxy)
  • IBI computation                   → M(c,a) (somatic marker context)
  • OnlineVarianceEstimator           → running Z-scores of prediction errors

FALSIFICATION CRITERIA
----------------------
  1. Detection quality < 0.6  → R-peaks unreliable; protocol aborted.
  2. |HEP mean amplitude| < 0.2 µV in 250–400 ms post-R window → Π^i falsified.
  3. IBI coefficient of variation > 0.3 (pathological variability) → M(c,a) unreliable.

APGI PARAMETER OUTPUTS
-----------------------
  Π^i_eff      : mean HEP amplitude in 250–400 ms post-R window (µV)
  M(c,a)       : IBI array (ms) summarised as mean ± SD
  Z-scored PE  : running normalisation via OnlineVarianceEstimator
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.heplab_cardiac_engine import (
    CardiacEvents,
    HEPResult,
    OnlineVarianceEstimator,
    detect_cardiac_events,
    extract_hep_epochs,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds (match FP_09 where overlapping)
# ──────────────────────────────────────────────────────────────────────────────

HEP_MIN_AMPLITUDE_UV = 0.2  # µV; paper prediction P2.a
DETECTION_QUALITY_MIN = 0.6  # below → abort
IBI_CV_MAX = 0.3  # IBI coefficient of variation ceiling
ALPHA = 0.05  # statistical significance level

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (for headless / no-data runs)
# ──────────────────────────────────────────────────────────────────────────────


def _generate_synthetic_ecg_eeg(
    duration_s: float = 60.0,
    srate: float = 1000.0,
    hr_bpm: float = 70.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic ECG + single EEG channel suitable for EP-0 testing.

    ECG: QRS complexes at the specified heart rate with Gaussian noise.
    EEG: White noise with a sinusoidal HEP component (300–500 ms post-R, ~0.5 µV).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * srate)

    # --- ECG ---
    rr_samp = int(srate * 60.0 / hr_bpm)
    r_positions = np.arange(rr_samp // 2, n, rr_samp)

    ecg = rng.normal(0, 0.05, n)  # baseline noise
    qrs_width = int(0.04 * srate)  # 40 ms QRS
    for r in r_positions:
        for offset in range(-qrs_width, qrs_width + 1):
            idx = r + offset
            if 0 <= idx < n:
                ecg[idx] += np.exp(-0.5 * (offset / (qrs_width / 3)) ** 2)

    # --- EEG (single channel, e.g. Cz) ---
    eeg = rng.normal(0, 0.1, n)
    hep_start_samp = int(0.250 * srate)  # 250 ms post-R
    hep_end_samp = int(0.400 * srate)  # 400 ms post-R
    hep_width = hep_end_samp - hep_start_samp
    hep_kernel = 0.5 * np.sin(np.linspace(0, np.pi, hep_width))
    for r in r_positions:
        lo = r + hep_start_samp
        hi = lo + hep_width
        if hi <= n:
            eeg[lo:hi] += hep_kernel

    return ecg, eeg


# ──────────────────────────────────────────────────────────────────────────────
# Core validation logic
# ──────────────────────────────────────────────────────────────────────────────


def _validate_cardiac_detection(events: CardiacEvents) -> Dict[str, Any]:
    """Sub-check 1: is R-peak detection reliable enough to proceed?"""
    passed = events.detection_quality >= DETECTION_QUALITY_MIN
    return {
        "check": "cardiac_detection_quality",
        "value": events.detection_quality,
        "threshold": DETECTION_QUALITY_MIN,
        "n_beats": events.n_beats,
        "heart_rate_bpm": events.heart_rate_bpm,
        "passed": passed,
        "message": (
            f"Detection quality {events.detection_quality:.3f} "
            f"({'OK' if passed else 'FAIL — below ' + str(DETECTION_QUALITY_MIN)})"
        ),
    }


def _validate_ibi(ibi_ms: np.ndarray) -> Dict[str, Any]:
    """Sub-check 2: IBI coefficient of variation within physiological bounds."""
    if len(ibi_ms) < 2:
        return {
            "check": "ibi_variability",
            "value": float("nan"),
            "threshold": IBI_CV_MAX,
            "passed": False,
            "message": "Insufficient beats for IBI analysis",
            "M_ca_mean_ms": float("nan"),
            "M_ca_std_ms": float("nan"),
        }

    mean_ibi = float(np.mean(ibi_ms))
    std_ibi = float(np.std(ibi_ms))
    cv = std_ibi / mean_ibi if mean_ibi > 0 else float("inf")
    passed = cv <= IBI_CV_MAX

    return {
        "check": "ibi_variability",
        "value": cv,
        "threshold": IBI_CV_MAX,
        "passed": passed,
        "M_ca_mean_ms": mean_ibi,  # APGI M(c,a) somatic marker context
        "M_ca_std_ms": std_ibi,
        "ibi_ms": ibi_ms.tolist(),
        "message": (
            f"IBI CV={cv:.3f} (mean={mean_ibi:.1f} ms, " f"std={std_ibi:.1f} ms) — {'OK' if passed else 'FAIL'}"
        ),
    }


def _validate_hep_amplitude(hep: HEPResult) -> Dict[str, Any]:
    """
    Sub-check 3: mean HEP amplitude in 300–500 ms post-R exceeds 0.2 µV.

    This is the APGI Π^i_eff extraction:
      Π^i_eff ≈ |mean HEP amplitude in 300–500 ms window|
    """
    if hep.n_trials == 0:
        return {
            "check": "hep_amplitude",
            "value": 0.0,
            "threshold": HEP_MIN_AMPLITUDE_UV,
            "passed": False,
            "pi_interoceptive_mean": 0.0,
            "n_trials": 0,
            "message": "No valid HEP epochs extracted",
        }

    pi_mean = float(np.mean(np.abs(hep.pi_interoceptive)))

    # One-sample t-test: is mean |Π^i| significantly different from 0?
    if hep.n_trials >= 2:
        t_stat, p_val = stats.ttest_1samp(np.abs(hep.pi_interoceptive), 0.0)
    else:
        t_stat, p_val = 0.0, 1.0

    passed = (pi_mean >= HEP_MIN_AMPLITUDE_UV) and (p_val < ALPHA)

    return {
        "check": "hep_amplitude",
        "value": pi_mean,
        "threshold": HEP_MIN_AMPLITUDE_UV,
        "passed": passed,
        "pi_interoceptive_mean": pi_mean,  # APGI Π^i_eff
        "pi_interoceptive_per_trial": hep.pi_interoceptive.tolist(),
        "n_trials": hep.n_trials,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "hep_window_ms": hep.hep_window_ms,
        "baseline_window_ms": hep.baseline_window_ms,
        "message": (
            f"HEP amplitude={pi_mean:.4f} µV "
            f"(threshold={HEP_MIN_AMPLITUDE_UV} µV, p={p_val:.4f}) — "
            f"{'PASS' if passed else 'FAIL'}"
        ),
    }


def _compute_running_zscores(pi_vals: np.ndarray) -> Dict[str, Any]:
    """
    Apply OnlineVarianceEstimator to Π^i values — produces running Z-scores
    that can be used for normalising prediction errors in real-time inference.
    """
    estimator = OnlineVarianceEstimator()
    zscores = estimator.batch_update(pi_vals.astype(float))
    return {
        "running_mean": estimator.mean,
        "running_std": estimator.std,
        "zscores": zscores.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def validate(
    ecg: Optional[np.ndarray] = None,
    eeg_channel: Optional[np.ndarray] = None,
    srate: float = 1000.0,
    use_synthetic: bool = True,
    synthetic_duration_s: float = 60.0,
    synthetic_hr_bpm: float = 70.0,
    synthetic_seed: int = 42,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    EP-0 HEP Proxy Validation.

    Parameters
    ----------
    ecg : 1-D array — raw ECG signal (µV or mV, consistent units).
          If None and use_synthetic=True, a synthetic signal is generated.
    eeg_channel : 1-D array — concurrent EEG channel (Cz / Fz recommended, µV).
                  Must have the same length and srate as ecg.
    srate : sampling rate in Hz
    use_synthetic : generate synthetic data when ecg/eeg_channel are absent
    synthetic_* : parameters for synthetic data generator

    Returns
    -------
    Dict with keys:
      status       — "passed" | "failed" | "aborted"
      canonical    — "EP-0"
      protocol     — "VP_00_HEPProxyValidation"
      checks       — list of sub-check result dicts
      apgi_params  — {"pi_interoceptive": ..., "M_ca_mean_ms": ..., "M_ca_std_ms": ...}
      running_stats— OnlineVarianceEstimator output
      message      — human-readable summary
    """
    result: Dict[str, Any] = {
        "canonical": "EP-0",
        "protocol": "VP_00_HEPProxyValidation",
        "checks": [],
        "apgi_params": {},
        "running_stats": {},
    }

    # ── Data acquisition ──────────────────────────────────────────────────────
    if ecg is None or eeg_channel is None:
        if not use_synthetic:
            result["status"] = "aborted"
            result["message"] = "No ECG/EEG data provided and use_synthetic=False."
            return result

        logger.info("VP-00: generating synthetic ECG+EEG (%.0f s, %.0f BPM)", synthetic_duration_s, synthetic_hr_bpm)
        ecg, eeg_channel = _generate_synthetic_ecg_eeg(
            duration_s=synthetic_duration_s,
            srate=srate,
            hr_bpm=synthetic_hr_bpm,
            seed=synthetic_seed,
        )
        result["data_source"] = "synthetic"
    else:
        ecg = np.asarray(ecg, dtype=float)
        eeg_channel = np.asarray(eeg_channel, dtype=float)
        result["data_source"] = "empirical"

    # ── Step A: R-wave and T-wave detection ──────────────────────────────────
    events: CardiacEvents = detect_cardiac_events(ecg, srate)

    check1 = _validate_cardiac_detection(events)
    result["checks"].append(check1)

    if not check1["passed"]:
        result["status"] = "aborted"
        result["message"] = (
            f"EP-0 aborted: cardiac detection quality {events.detection_quality:.3f} "
            f"< {DETECTION_QUALITY_MIN} (unreliable R-peaks)."
        )
        return result

    # ── Step B: IBI → M(c,a) ─────────────────────────────────────────────────
    check2 = _validate_ibi(events.ibi_ms)
    result["checks"].append(check2)

    # ── Step C: HEP epochs → Π^i_eff ─────────────────────────────────────────
    hep: HEPResult = extract_hep_epochs(eeg_channel, events.r_peak_indices, srate)

    check3 = _validate_hep_amplitude(hep)
    result["checks"].append(check3)

    # ── Running Z-score statistics ────────────────────────────────────────────
    if hep.n_trials > 0:
        result["running_stats"] = _compute_running_zscores(hep.pi_interoceptive)

    # ── APGI parameter pack ───────────────────────────────────────────────────
    result["apgi_params"] = {
        "pi_interoceptive": check3.get("pi_interoceptive_mean", 0.0),
        "pi_interoceptive_per_trial": check3.get("pi_interoceptive_per_trial", []),
        "M_ca_mean_ms": check2.get("M_ca_mean_ms", float("nan")),
        "M_ca_std_ms": check2.get("M_ca_std_ms", float("nan")),
        "ibi_ms": check2.get("ibi_ms", []),
        "n_beats": events.n_beats,
        "heart_rate_bpm": events.heart_rate_bpm,
        "detection_quality": events.detection_quality,
        "n_hep_trials": hep.n_trials,
    }

    # ── Overall status ────────────────────────────────────────────────────────
    all_passed = all(c.get("passed", False) for c in result["checks"])
    result["status"] = "passed" if all_passed else "failed"

    failed_checks: List[str] = [c["check"] for c in result["checks"] if not c.get("passed", False)]
    result["message"] = (
        "EP-0 PASSED: HEP amplitude validates Π^i proxy; IBI validates M(c,a)."
        if all_passed
        else f"EP-0 FAILED: {', '.join(failed_checks)} did not meet criteria."
    )

    return result


def run_validation(**kwargs: Any) -> Dict[str, Any]:
    """Alias for VP framework compatibility."""
    return validate(**kwargs)
