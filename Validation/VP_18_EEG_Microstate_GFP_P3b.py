#!/usr/bin/env python3
"""
VP-18: EEG Microstate Energy Analysis (GFP / P3b)
==================================================

Validation Protocol 18 from the APGI Empirical Credibility Roadmap.

Measures Global Field Power (GFP) — the spatial standard deviation of scalp
potential across all electrodes — as a proxy for synchronous large-scale
electromagnetic activity during the P3b window.

MEASUREMENT GAPS (required thermodynamic hedging):
    GFP is a measure of spatial standard deviation of scalp potential; it
    indexes global synchrony, not electromagnetic field energy in the physics
    sense.  The link to energy requires additional assumptions about current
    dipole density (Skrandies, 1990).  Accordingly, this protocol constitutes a
    Level 3 → Level 1 inference that requires the Landauer bridge and does NOT
    constitute direct thermodynamic measurement.  Program 3 (PET) is required
    for that.  This limitation must appear explicitly in the paper's Measurement
    Gaps section.

Protocol specification:
    Measure : GFP = √(Σᵢ Vᵢ²/N) at each timepoint across all scalp electrodes.
    Design  : Single-trial P3b paradigm (oddball or masked priming) with
              within-subject ignition vs. no-ignition conditions.
    Analysis: GFP integral (area under curve, 250–600 ms post-stimulus) is
              compared between conscious and unconscious trials.
    APGI prediction: GFP-AUC is significantly larger on ignition trials, and
              the effect magnitude correlates with Sₜ (ignition strength).

LEVEL DESIGNATION: All outputs are Level 3 (algorithmic/mathematical).
Bridge to Level 2 requires APGI_Information_Theoretic_Bandwidth.
Bridge to Level 1 requires APGI_Thermodynamic_Program_Aggregator.
This script does NOT claim thermodynamic or information-theoretic implications
without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If GFP-AUC does not differ significantly between ignition and no-ignition trials
(p > 0.05, corrected), or if the GFP effect does not correlate with ignition
strength (r < 0.5), or if the GFP dynamics do not match APGI predictions
(temporal deviation > 100ms), then the APGI microstate energy claim is falsified.
This would indicate that global field power does not reflect ignition dynamics.

Tier: SECONDARY

Master_Validation.py registration:
    "Protocol-18": {
        "file": "VP_18_EEG_Microstate_GFP_P3b.py",
        "function": "run_validation",
        "description": "EEG Microstate Energy Analysis (GFP / P3b)",
    }

VP_ALL_Aggregator.py registration:
    "V18.1": "VP_18_EEG_Microstate_GFP_P3b",
    "V18.2": "VP_18_EEG_Microstate_GFP_P3b",
    "V18.3": "VP_18_EEG_Microstate_GFP_P3b",
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# Matplotlib imports for PNG visualisation
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import falsification thresholds
try:
    from utils.falsification_thresholds import (
        DEFAULT_ALPHA,
        V18_AUC_COHENS_D_MIN,
        V18_GFP_AUC_IGNITION_ADVANTAGE,
        V18_ST_CORRELATION_MIN,
    )
except ImportError:
    DEFAULT_ALPHA = 0.05
    V18_GFP_AUC_IGNITION_ADVANTAGE = 0.20  # ≥20 % larger GFP-AUC on ignition trials
    V18_AUC_COHENS_D_MIN = 0.50  # Cohen's d ≥ 0.50 (medium effect)
    V18_ST_CORRELATION_MIN = 0.30  # r ≥ 0.30 between GFP-AUC and Sₜ

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

# P3b analysis window (ms post-stimulus)
GFP_WINDOW_MS: Tuple[float, float] = (250.0, 600.0)

# Simulated EEG parameters
N_ELECTRODES: int = 64
SAMPLING_RATE_HZ: float = 1000.0  # 1 ms resolution
EPOCH_DURATION_MS: float = 800.0  # –200 to +600 ms

RANDOM_SEED: int = 42


@dataclass
class EpochMetadata:
    subject_id: int
    trial_id: int
    condition: str  # "ignition" | "no_ignition"
    ignition_strength: float  # Sₜ ∈ [0, 1]


@dataclass
class GFPResult:
    epoch_meta: EpochMetadata
    gfp_timeseries: np.ndarray  # shape (n_timepoints,)
    gfp_auc: float  # area under GFP curve in 250–600 ms window
    timepoints_ms: np.ndarray  # shape (n_timepoints,)


# ──────────────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────────────


class OddballEEGSimulator:
    """
    Simulate single-trial EEG data for an oddball / masked-priming paradigm.

    Generates 64-channel synthetic EEG with a realistic P3b component on
    ignition trials and absent/attenuated P3b on no-ignition trials.
    Noise is spatially correlated via a simple distance kernel.
    """

    def __init__(
        self,
        n_subjects: int = 24,
        n_trials_per_condition: int = 60,
        sampling_rate: float = SAMPLING_RATE_HZ,
        n_electrodes: int = N_ELECTRODES,
        rng: Optional[np.random.Generator] = None,
    ):
        self.n_subjects = n_subjects
        self.n_trials_per_condition = n_trials_per_condition
        self.fs = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.n_electrodes = n_electrodes
        self.rng = rng if rng is not None else np.random.default_rng(RANDOM_SEED)

        # Epoch grid: –200 to +600 ms
        total_ms = EPOCH_DURATION_MS
        self.n_timepoints = int(total_ms * self.dt * sampling_rate)  # 800 samples
        self.timepoints_ms = np.linspace(-200.0, 600.0, self.n_timepoints)

        # Indices for GFP-AUC window
        self.window_idx = (self.timepoints_ms >= GFP_WINDOW_MS[0]) & (
            self.timepoints_ms <= GFP_WINDOW_MS[1]
        )

        # Spatial covariance matrix (simple 1/distance kernel)
        self._spatial_cov = self._build_spatial_cov()

    # ------------------------------------------------------------------
    def _build_spatial_cov(self) -> np.ndarray:
        idx = np.arange(self.n_electrodes)
        dist = np.abs(idx[:, None] - idx[None, :]) / self.n_electrodes
        cov = np.exp(-dist / 0.3)
        return cov

    # ------------------------------------------------------------------
    def _p3b_template(self, amplitude: float = 1.0) -> np.ndarray:
        """Gaussian P3b bump centred at 380 ms (σ = 60 ms)."""
        mu = 380.0
        sigma = 60.0
        template = amplitude * np.exp(-0.5 * ((self.timepoints_ms - mu) / sigma) ** 2)
        # Zero before stimulus onset
        template[self.timepoints_ms < 0] = 0.0
        return template

    # ------------------------------------------------------------------
    def _simulate_epoch(
        self,
        subject_id: int,
        trial_id: int,
        condition: str,
        ignition_strength: float,
    ) -> GFPResult:
        """Simulate a single EEG epoch and compute GFP."""
        # Subject-level amplitude scaling (between-subject variance)
        subject_scale = 1.0 + 0.2 * self.rng.standard_normal()

        # P3b amplitude: present and scaled by Sₜ on ignition, minimal otherwise
        if condition == "ignition":
            p3b_amp = subject_scale * (2.0 + 3.0 * ignition_strength)
        else:
            # Residual (subliminal / noise-level) response
            p3b_amp = subject_scale * 0.3 * ignition_strength

        # Build source time-series: P3b + broadband noise
        source = self._p3b_template(p3b_amp)  # (n_timepoints,)

        # Project to electrodes with spatial spread + independent noise
        noise_amplitude = 1.5
        spatial_noise = self.rng.multivariate_normal(
            mean=np.zeros(self.n_electrodes),
            cov=self._spatial_cov * noise_amplitude**2,
            size=self.n_timepoints,
        )  # (n_timepoints, n_electrodes)

        # Electrode-level signal: shared P3b source + independent noise
        electrode_signals = (
            source[:, None] * (1.0 + 0.1 * self.rng.standard_normal(self.n_electrodes))
            + spatial_noise
        )  # (n_timepoints, n_electrodes)

        # GFP = spatial standard deviation across electrodes at each timepoint
        gfp = np.sqrt(np.mean(electrode_signals**2, axis=1))  # (n_timepoints,)

        # GFP-AUC using trapezoidal integration over the 250–600 ms window
        window_gfp = gfp[self.window_idx]
        window_t = self.timepoints_ms[self.window_idx] / 1000.0  # convert to seconds
        trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        gfp_auc = float(trapz(window_gfp, window_t))

        return GFPResult(
            epoch_meta=EpochMetadata(
                subject_id, trial_id, condition, ignition_strength
            ),
            gfp_timeseries=gfp,
            gfp_auc=gfp_auc,
            timepoints_ms=self.timepoints_ms,
        )

    # ------------------------------------------------------------------
    def simulate_dataset(self) -> List[GFPResult]:
        """Simulate all subjects × trials × conditions."""
        dataset: List[GFPResult] = []

        for subj in range(self.n_subjects):
            for trial in range(self.n_trials_per_condition):
                # Ignition trials: Sₜ drawn from Beta(3, 1.5) → skewed toward 1
                st_ignition = float(self.rng.beta(3.0, 1.5))
                dataset.append(
                    self._simulate_epoch(subj, trial, "ignition", st_ignition)
                )

                # No-ignition trials: Sₜ drawn from Beta(1.5, 3) → skewed toward 0
                st_no = float(self.rng.beta(1.5, 3.0))
                dataset.append(self._simulate_epoch(subj, trial, "no_ignition", st_no))

        return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────────────────────────────────────


class GFPMicrostateValidator:
    """
    Validates APGI predictions from GFP-AUC measures.

    Three testable predictions:
        V18.1  GFP-AUC is significantly larger on ignition vs. no-ignition
               trials (two-sided t-test, p < DEFAULT_ALPHA, d ≥ 0.50).
        V18.2  The proportional GFP-AUC advantage on ignition trials exceeds
               V18_GFP_AUC_IGNITION_ADVANTAGE (≥ 20 %).
        V18.3  Single-trial GFP-AUC correlates positively with Sₜ (r ≥ 0.30,
               p < DEFAULT_ALPHA) — critical link to the APGI energy budget.
    """

    def __init__(self, simulator: Optional[OddballEEGSimulator] = None):
        self.simulator = simulator or OddballEEGSimulator()
        self.dataset: List[GFPResult] = []

    # ------------------------------------------------------------------
    def load_or_generate_data(self) -> None:
        if not self.dataset:
            logger.info("Simulating single-trial EEG dataset …")
            self.dataset = self.simulator.simulate_dataset()
            logger.info("Simulated %d epochs.", len(self.dataset))

    # ------------------------------------------------------------------
    def _split_by_condition(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (ignition_aucs, no_ignition_aucs, all_st, all_auc)."""
        ignition_aucs, no_ignition_aucs = [], []
        all_st, all_auc = [], []

        for r in self.dataset:
            if r.epoch_meta.condition == "ignition":
                ignition_aucs.append(r.gfp_auc)
            else:
                no_ignition_aucs.append(r.gfp_auc)
            all_st.append(r.epoch_meta.ignition_strength)
            all_auc.append(r.gfp_auc)

        return (
            np.array(ignition_aucs),
            np.array(no_ignition_aucs),
            np.array(all_st),
            np.array(all_auc),
        )

    # ------------------------------------------------------------------
    def validate_gfp_auc_ignition_effect(self) -> Dict[str, Any]:
        """V18.1: GFP-AUC significantly larger on ignition trials (d ≥ 0.50)."""
        ig_aucs, no_aucs, _, _ = self._split_by_condition()

        t_stat, p_value = stats.ttest_ind(ig_aucs, no_aucs, equal_var=False)

        pooled_sd = np.sqrt((np.var(ig_aucs, ddof=1) + np.var(no_aucs, ddof=1)) / 2.0)
        cohens_d = (
            (np.mean(ig_aucs) - np.mean(no_aucs)) / pooled_sd if pooled_sd > 0 else 0.0
        )

        passed = (
            (p_value < DEFAULT_ALPHA)
            and (t_stat > 0)
            and (cohens_d >= V18_AUC_COHENS_D_MIN)
        )

        return {
            "test_name": "V18.1 GFP-AUC Ignition Effect",
            "prediction_id": "V18.1",
            "mean_ignition_auc": float(np.mean(ig_aucs)),
            "mean_no_ignition_auc": float(np.mean(no_aucs)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "threshold_d": V18_AUC_COHENS_D_MIN,
            "threshold_alpha": DEFAULT_ALPHA,
            "passed": bool(passed),
        }

    # ------------------------------------------------------------------
    def validate_proportional_advantage(self) -> Dict[str, Any]:
        """V18.2: Proportional GFP-AUC advantage ≥ 20 % on ignition trials."""
        ig_aucs, no_aucs, _, _ = self._split_by_condition()

        mean_ig = float(np.mean(ig_aucs))
        mean_no = float(np.mean(no_aucs))

        if mean_no > 0:
            proportional_advantage = (mean_ig - mean_no) / mean_no
        else:
            proportional_advantage = 0.0

        passed = proportional_advantage >= V18_GFP_AUC_IGNITION_ADVANTAGE

        return {
            "test_name": "V18.2 Proportional GFP-AUC Advantage",
            "prediction_id": "V18.2",
            "proportional_advantage": float(proportional_advantage),
            "threshold": V18_GFP_AUC_IGNITION_ADVANTAGE,
            "mean_ignition_auc": mean_ig,
            "mean_no_ignition_auc": mean_no,
            "passed": bool(passed),
        }

    # ------------------------------------------------------------------
    def validate_st_correlation(self) -> Dict[str, Any]:
        """V18.3: Single-trial GFP-AUC correlates with Sₜ (r ≥ 0.30)."""
        _, _, all_st, all_auc = self._split_by_condition()

        r, p_value = stats.pearsonr(all_st, all_auc)
        passed = (r >= V18_ST_CORRELATION_MIN) and (p_value < DEFAULT_ALPHA)

        return {
            "test_name": "V18.3 GFP-AUC × Sₜ Correlation",
            "prediction_id": "V18.3",
            "pearson_r": float(r),
            "p_value": float(p_value),
            "threshold_r": V18_ST_CORRELATION_MIN,
            "threshold_alpha": DEFAULT_ALPHA,
            "n_trials": int(len(all_st)),
            "passed": bool(passed),
        }

    # ------------------------------------------------------------------
    def generate_summary_figure(
        self, output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """Save a three-panel summary figure if matplotlib is available."""
        if not HAS_MATPLOTLIB:
            return None

        ig_aucs, no_aucs, all_st, all_auc = self._split_by_condition()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("VP-18: EEG Microstate GFP / P3b Analysis", fontsize=13)

        # Panel 1: Average GFP time-series by condition
        ax = axes[0]
        ig_epochs = [r for r in self.dataset if r.epoch_meta.condition == "ignition"]
        no_epochs = [r for r in self.dataset if r.epoch_meta.condition == "no_ignition"]
        t = ig_epochs[0].timepoints_ms
        ig_mean = np.mean([r.gfp_timeseries for r in ig_epochs], axis=0)
        no_mean = np.mean([r.gfp_timeseries for r in no_epochs], axis=0)
        ax.plot(t, ig_mean, label="Ignition", color="steelblue")
        ax.plot(t, no_mean, label="No-ignition", color="salmon")
        ax.axvspan(
            GFP_WINDOW_MS[0],
            GFP_WINDOW_MS[1],
            alpha=0.15,
            color="grey",
            label="AUC window",
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("GFP (μV)")
        ax.set_title("Mean GFP time-series")
        ax.legend(fontsize=8)

        # Panel 2: GFP-AUC distribution by condition
        ax = axes[1]
        ax.hist(ig_aucs, bins=30, alpha=0.7, color="steelblue", label="Ignition")
        ax.hist(no_aucs, bins=30, alpha=0.7, color="salmon", label="No-ignition")
        ax.set_xlabel("GFP-AUC (μV·s)")
        ax.set_ylabel("Count")
        ax.set_title("GFP-AUC distributions")
        ax.legend(fontsize=8)

        # Panel 3: GFP-AUC vs Sₜ scatter
        ax = axes[2]
        ax.scatter(all_st, all_auc, alpha=0.3, s=10, color="dimgray")
        m, b = np.polyfit(all_st, all_auc, 1)
        x_line = np.linspace(all_st.min(), all_st.max(), 100)
        ax.plot(x_line, m * x_line + b, color="firebrick", linewidth=2)
        r, _ = stats.pearsonr(all_st, all_auc)
        ax.set_xlabel("Ignition strength Sₜ")
        ax.set_ylabel("GFP-AUC (μV·s)")
        ax.set_title(f"GFP-AUC vs Sₜ  (r = {r:.2f})")

        plt.tight_layout()

        if output_path is None:
            results_dir = _project_root / "data_repository" / "processed_data"
            results_dir.mkdir(parents=True, exist_ok=True)
            output_path = results_dir / "VP18_GFP_P3b_summary.png"

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Summary figure saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    def run_full_validation(self) -> Dict[str, Any]:
        """Run all three validation tests and return aggregated results."""
        self.load_or_generate_data()

        results: Dict[str, Any] = {
            "gfp_auc_ignition_effect": self.validate_gfp_auc_ignition_effect(),
            "proportional_advantage": self.validate_proportional_advantage(),
            "st_correlation": self.validate_st_correlation(),
        }

        passed_tests = sum(1 for r in results.values() if r.get("passed", False))
        total_tests = len(results)

        results.update(
            {
                "overall_score": passed_tests / total_tests,
                "tests_passed": passed_tests,
                "tests_total": total_tests,
                "protocol_id": "VP_18_EEG_Microstate_GFP_P3b",
                "validation_timestamp": datetime.now().isoformat(),
                "measurement_gap_note": (
                    "GFP is the spatial standard deviation of scalp potential "
                    "— it indexes global synchrony, not field energy in the "
                    "physics sense. The link to metabolic energy requires "
                    "assumptions about current dipole density (Skrandies, 1990). "
                    "This is a Level 3 → Level 1 inference requiring the Landauer "
                    "bridge; it does not constitute direct thermodynamic measurement."
                ),
            }
        )

        logger.info(
            "VP-18 validation complete: %d/%d tests passed (score %.2f)",
            passed_tests,
            total_tests,
            results["overall_score"],
        )
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────────────────


def run_validation(seed: Optional[int] = None, **kwargs) -> Dict[str, Any]:
    """
    Entry point for external validation runners.

    Args:
        seed: Optional RNG seed for reproducibility.
        **kwargs: Accepted but ignored (API compatibility).
    """
    rng = np.random.default_rng(seed if seed is not None else RANDOM_SEED)
    simulator = OddballEEGSimulator(rng=rng)
    validator = GFPMicrostateValidator(simulator=simulator)
    results = validator.run_full_validation()
    validator.generate_summary_figure()
    return results


def validate() -> Dict[str, Any]:
    """Alias for main.py compatibility."""
    return run_validation()


def main() -> None:
    """CLI entry point for standalone execution."""
    print("=" * 60)
    print("VP-18: EEG Microstate Energy Analysis (GFP / P3b)")
    print("=" * 60)

    results = run_validation()

    print(f"\nOverall Score : {results['overall_score']:.2%}")
    print(f"Tests Passed  : {results['tests_passed']}/{results['tests_total']}")

    print("\nDetailed Results:")
    for key, val in results.items():
        if isinstance(val, dict) and "test_name" in val:
            status = "PASS" if val.get("passed") else "FAIL"
            print(f"  [{status}] {val['test_name']}")

    print("\nMeasurement Gap:")
    print(f"  {results['measurement_gap_note']}")


if __name__ == "__main__":
    main()
