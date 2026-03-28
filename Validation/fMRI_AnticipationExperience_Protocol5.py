"""
APGI Protocol 5 (Part B): fMRI Anticipation & Experience Validation
===================================================================

Simulates blood-oxygen-level-dependent (BOLD) tracking of APGI's
internal variables S(t) [Salience/AI] and M(t) [vmPFC] during a threat 
anticipation paradigm.

Tests neuroimaging hypotheses from the APGI Multi-Scale framework.
"""

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import convolve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def double_gamma_hrf(t: np.ndarray) -> np.ndarray:
    """Standard double-gamma Hemodynamic Response Function (HRF)"""
    a1, a2 = 6.0, 16.0
    b1, b2 = 1.0, 1.0
    c = 1.0 / 6.0
    # Avoid 0^0 by clipping t
    t_safe = np.clip(t, 1e-8, None)
    hrf = (
        t_safe ** (a1 - 1)
        * np.exp(-t_safe / b1)
        / (b1**a1 * math.factorial(int(a1) - 1))
    ) - c * (
        t_safe ** (a2 - 1)
        * np.exp(-t_safe / b2)
        / (b2**a2 * math.factorial(int(a2) - 1))
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
            S[
                shock_idx : shock_idx + int(1.0 / self.dt)
            ] = 1.0  # Sharp surprise / ignition
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
        }


def validate_fmri_predictions(sim_results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate empirical APGI fMRI predictions."""
    S_bold = sim_results["S_bold"]
    M_bold = sim_results["M_bold"]
    conditions = sim_results["conditions"]
    dt = sim_results["dt"]
    pts_per_trial = int(sim_results["trial_duration"] / dt)

    # Extract threat vs safe trials
    threat_M_bolds = []
    safe_M_bolds = []

    for i, (is_threat, _) in enumerate(conditions):
        idx_start = i * pts_per_trial
        idx_end = idx_start + pts_per_trial
        trial_M = M_bold[idx_start:idx_end]
        if is_threat:
            threat_M_bolds.append(trial_M)
        else:
            safe_M_bolds.append(trial_M)

    # We check if Threat completely diverges from Safe
    stat, p_val = stats.ttest_ind(
        np.max(threat_M_bolds, axis=1),
        np.max(safe_M_bolds, axis=1)
        if len(safe_M_bolds) > 0
        else [0] * len(threat_M_bolds),
    )

    # P5.2: Functional connectivity between vmPFC (M) and AI (S) during threat
    # APGI predicts strong functional coupling between salience and somatic markers
    connectivity_r, p_conn = stats.pearsonr(S_bold, M_bold)

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


def main():
    logger.info(
        "Initializing APGI fMRI Anticipation/Experience Simulation (VP-5 BOLD)..."
    )
    config = fMRI_SimulationConfig()
    simulator = APGI_fMRISimulator(config)
    results = simulator.run_experiment()

    plot_fmri_results(results)
    logger.info("Generated fMRI BOLD timeseries plot: protocol5_fmri_timeseries.png")

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

    output_payload = {"config": config.__dict__, "validation_report": validation_report}

    with open("protocol5_fmri_results.json", "w") as f:
        json.dump(output_payload, f, indent=4)

    logger.info("fMRI Validation Data Saved to protocol5_fmri_results.json")


if __name__ == "__main__":
    main()
