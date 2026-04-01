"""
APGI Protocol 14: fMRI Anticipation & Experience Validation
============================================================

Status: SIMULATION-VALIDATED, AWAITING EMPIRICAL CONFIRMATION
Note: This protocol uses double-gamma HRF convolution over synthetic S(t) and M(t)
time series to validate the computational pipeline. Final empirical confirmation
requires real fMRI data ingestion.

Simulates blood-oxygen-level-dependent (BOLD) tracking of APGI's
internal variables S(t) [Salience/AI] and M(t) [vmPFC] during a threat
anticipation paradigm.
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import convolve

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    nib = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def double_gamma_hrf(t: np.ndarray) -> np.ndarray:
    """
    Canonical SPM/FSL-style double-gamma hemodynamic response function.

    Parameters follow the standard named canonical HRF settings:
    - response_peak_delay_s = 6s
    - undershoot_delay_s = 16s
    - response_dispersion_s = 1s
    - undershoot_dispersion_s = 1s
    - undershoot_ratio = 1/6
    """
    response_peak_delay_s = 6.0
    undershoot_delay_s = 16.0
    response_dispersion_s = 1.0
    undershoot_dispersion_s = 1.0
    undershoot_ratio = 1.0 / 6.0
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


@dataclass
class fMRI_SimulationConfig:
    n_trials: int = 60
    threat_prob: float = 0.75
    fs: float = 10.0  # Sampling rate in Hz
    trial_duration: float = 12.0  # seconds
    cue_onset: float = 1.0
    shock_onset: float = 4.0
    tau_M: float = 1.52  # vmPFC somatic marker tau
    scanner_noise_pct_bold: float = 0.002  # 0.2% BOLD typical range 0.1-0.3%
    detectability_sample_size: int = 30


class APGI_fMRISimulator:
    """
    Simulates APGI variables mapping to fMRI BOLD signals:
    - M(t): Somatic Marker State -> vmPFC BOLD
    - S(t): Accumulated Surprise -> Anterior Insula (AI) BOLD
    """

    def __init__(self, config: fMRI_SimulationConfig):
        self.config = config
        self.dt = 1.0 / config.fs

    def simulate_trial(
        self, is_threat_cue: bool, receives_shock: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate neural APGI variables S(t) and M(t) for a single trial."""
        t = np.arange(0, self.config.trial_duration, self.dt)
        S = np.zeros_like(t)
        M = np.zeros_like(t)

        cue_idx = int(self.config.cue_onset / self.dt)
        shock_idx = int(self.config.shock_onset / self.dt)

        # M(t) maps to vmPFC (Anticipatory somatic marker)
        if is_threat_cue:
            # Anticipatory rise in vmPFC
            for i in range(cue_idx, shock_idx):
                time_since_cue = t[i] - self.config.cue_onset
                # Exponential rise to max
                M[i] = 1.0 * (1 - np.exp(-time_since_cue / self.config.tau_M))
                M[i] += np.random.normal(0, 0.01)

            # Decay after outcome
            outcome_M = M[shock_idx - 1]
            for i in range(shock_idx, len(t)):
                time_since_outcome = t[i] - self.config.shock_onset
                M[i] = outcome_M * np.exp(
                    -time_since_outcome / (self.config.tau_M * 0.5)
                )
                M[i] += np.random.normal(0, 0.01)

        # S(t) maps to Anterior Insula (AI) / Salience (Experience / Shock processing)
        if receives_shock:
            S[shock_idx : shock_idx + int(1.0 / self.dt)] = (
                1.0  # Sharp surprise / ignition
            )
        if is_threat_cue:
            # Slight S(t) during cue
            S[cue_idx : cue_idx + int(0.5 / self.dt)] = 0.3

        return t, S, M

    def run_experiment(self) -> Dict[str, Any]:
        """Run full block of trials and convolve with HRF to generate BOLD."""
        all_S_neural = []
        all_M_neural = []
        conditions = []

        for _ in range(self.config.n_trials):
            is_threat = np.random.random() < 0.66
            receives_shock = is_threat and (
                np.random.random() < self.config.threat_prob
            )

            t, S, M = self.simulate_trial(is_threat, receives_shock)
            all_S_neural.append(S)
            all_M_neural.append(M)
            conditions.append((is_threat, receives_shock))

        # Concatenate entire timeseries
        S_continuous = np.concatenate(all_S_neural)
        M_continuous = np.concatenate(all_M_neural)

        # Convolve with HRF
        hrf_t = np.arange(0, 25.0, self.dt)
        hrf = double_gamma_hrf(hrf_t)

        S_bold = convolve(S_continuous, hrf, mode="full")[: len(S_continuous)]
        M_bold = convolve(M_continuous, hrf, mode="full")[: len(M_continuous)]

        # Add scanner noise
        S_bold += np.random.normal(0, 0.05, len(S_bold))
        M_bold += np.random.normal(0, 0.05, len(M_bold))

        return {
            "S_bold": S_bold,
            "M_bold": M_bold,
            "conditions": conditions,
            "dt": self.dt,
            "trial_duration": self.config.trial_duration,
            "data_source": "synthetic",
            "scanner_noise_pct_bold": self.config.scanner_noise_pct_bold,
        }


def load_fmri_data(fmri_data_path: str) -> Dict[str, Any]:
    """
    Load real fMRI data from a NIfTI file or NPZ container.

    Supported inputs:
    - `.nii` / `.nii.gz` with voxel timeseries averaged across space
    - `.npz` containing `S_bold`, `M_bold`, optional `conditions`, `dt`, `trial_duration`
    """
    path = Path(fmri_data_path)
    if not path.exists():
        raise FileNotFoundError(f"fMRI data path not found: {fmri_data_path}")

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        conditions = data["conditions"].tolist() if "conditions" in data else []
        return {
            "S_bold": np.asarray(data["S_bold"], dtype=float),
            "M_bold": np.asarray(data["M_bold"], dtype=float),
            "conditions": conditions,
            "dt": float(data["dt"]) if "dt" in data else 1.0,
            "trial_duration": (
                float(data["trial_duration"]) if "trial_duration" in data else 12.0
            ),
            "data_source": "real_npz",
        }

    if path.suffix in {".nii", ".gz"}:
        if not HAS_NIBABEL:
            raise ImportError(
                "nibabel is required to load NIfTI fMRI data; install with pip install nibabel"
            )
        img: Any = nib.load(str(path))
        data = np.asarray(img.get_fdata(), dtype=float)
        if data.ndim < 4:
            raise ValueError("Expected 4D NIfTI data (x, y, z, time)")
        voxel_ts = data.reshape(-1, data.shape[-1])
        mean_ts = np.nanmean(voxel_ts, axis=0)
        dt = (
            float(img.header.get_zooms()[-1])  # type: ignore[attr-defined]
            if len(img.header.get_zooms()) >= 4  # type: ignore[attr-defined]
            else 1.0
        )
        sidecar_path = (
            Path(str(path)[:-7] + ".json")
            if str(path).endswith(".nii.gz")
            else path.with_suffix(".json")
        )
        sidecar = {}
        if sidecar_path.exists():
            with open(sidecar_path, "r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
        return {
            "S_bold": mean_ts.copy(),
            "M_bold": mean_ts.copy(),
            "conditions": sidecar.get("conditions", []),
            "dt": float(sidecar.get("dt", dt)),
            "trial_duration": float(sidecar.get("trial_duration", data.shape[-1] * dt)),
            "data_source": "real_nifti_mean_signal",
            "sidecar_path": str(sidecar_path) if sidecar_path.exists() else None,
        }

    raise ValueError(
        f"Unsupported fMRI data format for {fmri_data_path}. Use .npz, .nii, or .nii.gz."
    )


def compute_bold_detectability(
    sim_results: Dict[str, Any],
    threat_M_bolds: np.ndarray,
    safe_M_bolds: np.ndarray,
    scanner_noise_pct_bold: float = 0.002,
    n_participants: int = 30,
) -> Dict[str, Any]:
    """Estimate tSNR and whether the vmPFC BOLD difference is detectable at N=30."""
    M_bold = np.asarray(sim_results["M_bold"], dtype=float)
    bold_scale = max(np.max(np.abs(M_bold)), 1e-6)
    scanner_noise_amplitude = bold_scale * scanner_noise_pct_bold
    signal_difference = float(
        np.mean(np.max(threat_M_bolds, axis=1)) - np.mean(np.max(safe_M_bolds, axis=1))
    )
    tsnr = float(abs(signal_difference) / max(scanner_noise_amplitude, 1e-9))
    detectable = (tsnr >= 2.0) and (n_participants >= 30)

    return {
        "signal_difference": signal_difference,
        "scanner_noise_amplitude": float(scanner_noise_amplitude),
        "scanner_noise_pct_bold": float(scanner_noise_pct_bold),
        "tsnr": tsnr,
        "n_participants": int(n_participants),
        "detectable_at_n30": bool(detectable),
        "criterion": "tSNR >= 2.0 with N >= 30",
    }


def validate_fmri_predictions(sim_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate empirical APGI fMRI predictions."""
    S_bold = sim_results["S_bold"]
    M_bold = sim_results["M_bold"]
    conditions = sim_results["conditions"]
    dt = sim_results["dt"]
    pts_per_trial = int(sim_results["trial_duration"] / dt)

    # Extract threat vs safe trials
    threat_M_bolds_list: list[Any] = []
    safe_M_bolds_list: list[Any] = []

    for i, (is_threat, _) in enumerate(conditions):
        idx_start = i * pts_per_trial
        idx_end = idx_start + pts_per_trial
        trial_M = M_bold[idx_start:idx_end]
        if is_threat:
            threat_M_bolds_list.append(trial_M)
        else:
            safe_M_bolds_list.append(trial_M)

    if not threat_M_bolds_list or not safe_M_bolds_list:
        raise ValueError(
            "Threat/safe condition labels are required for anticipation/experience validation"
        )

    threat_M_bolds = np.asarray(threat_M_bolds_list, dtype=float)
    safe_M_bolds = np.asarray(safe_M_bolds_list, dtype=float)

    # We check if Threat completely diverges from Safe
    stat, p_val = stats.ttest_ind(
        np.max(threat_M_bolds, axis=1),
        np.max(safe_M_bolds, axis=1),
    )

    # P5.2: Functional connectivity between vmPFC (M) and AI (S) during threat
    # APGI predicts strong functional coupling between salience and somatic markers
    connectivity_r, p_conn = stats.pearsonr(S_bold, M_bold)
    detectability = compute_bold_detectability(
        sim_results,
        threat_M_bolds,
        safe_M_bolds,
        scanner_noise_pct_bold=sim_results.get("scanner_noise_pct_bold", 0.002),
        n_participants=sim_results.get("detectability_sample_size", 30),
    )

    results = {
        "F5.1_Anticipatory_vmPFC": {
            "passed": bool(
                p_val < 0.05
                and np.mean(np.max(threat_M_bolds, axis=1))
                > np.mean(np.max(safe_M_bolds, axis=1))
            ),
            "p_value": float(p_val),
            "threat_mean_peak": float(np.mean(np.max(threat_M_bolds, axis=1))),
            "safe_mean_peak": float(np.mean(np.max(safe_M_bolds, axis=1))),
        },
        "F5.2_vmPFC_AI_Connectivity": {
            "passed": bool(connectivity_r > 0.30 and p_conn < 0.01),
            "pearson_r": float(connectivity_r),
            "p_value": float(p_conn),
        },
        "V14_tSNR_Detectability": {
            "passed": detectability["detectable_at_n30"],
            **detectability,
        },
    }

    return results


def plot_fmri_results(results: Dict[str, Any]):
    """Provides a visualization of the BOLD signal validation."""
    S_bold = results["S_bold"]
    M_bold = results["M_bold"]
    t = np.arange(len(S_bold)) * results["dt"]

    # Plot a snippet of the first 5 trials
    display_t = 5 * results["trial_duration"]
    idx = int(display_t / results["dt"])

    plt.figure(figsize=(12, 6))
    plt.plot(t[:idx], S_bold[:idx], label="AI (Salience) BOLD", color="red")
    plt.plot(t[:idx], M_bold[:idx], label="vmPFC (Anticipation) BOLD", color="blue")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
    plt.title("Simulated APGI fMRI BOLD Timeseries (First 5 Trials)")
    plt.xlabel("Time (s)")
    plt.ylabel("BOLD Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("protocol5_fmri_timeseries.png")
    plt.close()


def main(fmri_data_path: Optional[str] = None):
    logger.info(
        "Initializing APGI fMRI Anticipation/Experience Simulation (VP-14 BOLD)..."
    )
    config = fMRI_SimulationConfig()
    if fmri_data_path:
        logger.info(f"Loading empirical fMRI data from: {fmri_data_path}")
        results = load_fmri_data(fmri_data_path)
        results.setdefault("scanner_noise_pct_bold", config.scanner_noise_pct_bold)
        results.setdefault(
            "detectability_sample_size", config.detectability_sample_size
        )
    else:
        simulator = APGI_fMRISimulator(config)
        results = simulator.run_experiment()
        results["detectability_sample_size"] = config.detectability_sample_size

    if results.get("conditions"):
        plot_fmri_results(results)
        logger.info(
            "Generated fMRI BOLD timeseries plot: protocol5_fmri_timeseries.png"
        )
    else:
        logger.info(
            "Skipping trial-wise plot because real-data conditions were not provided."
        )

    logger.info("Running Falsification Validation...")
    validation_report = validate_fmri_predictions(results)

    passed_count = sum(1 for v in validation_report.values() if v.get("passed", False))
    logger.info(
        f"Validation Summary: {passed_count}/{len(validation_report)} Checks Passed"
    )

    for k, v in validation_report.items():
        logger.info(
            f"{k}: {'PASS' if v.get('passed', False) else 'FAIL'} - Details: {v}"
        )

    output_payload = {
        "config": config.__dict__,
        "fmri_data_path": fmri_data_path,
        "data_source": results.get("data_source", "synthetic"),
        "validation_report": validation_report,
    }

    with open("protocol5_fmri_results.json", "w") as f:
        json.dump(output_payload, f, indent=4)

    logger.info("fMRI Validation Data Saved to protocol5_fmri_results.json")

    # Return proper validation result format for Master_Validation.py
    passed_count = sum(1 for v in validation_report.values() if v.get("passed", False))
    total_count = len(validation_report)
    all_passed = passed_count == total_count

    return {
        "passed": all_passed,
        "status": "success" if all_passed else "failed",
        "message": f"fMRI validation: {passed_count}/{total_count} checks passed",
        "criteria": validation_report,
        "summary": {
            "passed": passed_count,
            "total": total_count,
            "failed": total_count - passed_count,
        },
    }


def run_validation(**kwargs) -> Dict[str, Any]:
    """Run validation protocol for Master_Validation integration.

    This function provides the interface expected by Master_Validation.py
    """
    return main(fmri_data_path=kwargs.get("fmri_data_path"))


if __name__ == "__main__":
    main()
