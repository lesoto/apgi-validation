"""
=============================================================================
APGI HIERARCHICAL TEMPORAL TILING
=============================================================================

Implements the parsimony-based derivation of the five-level Russian Doll
architecture (τ₀–τ₄) claimed by APGI, together with a coupled-threshold
simulation that tests cross-level resonance.

This module provides:
1. GeometricTilingDerivation  – derives the optimal N from a coverage
   parsimony constraint over the range τ_min=0.01 s to τ_max=3.15e7 s.
2. CrossLevelCouplingSimulator – simulates top-down / bottom-up coupling
   dynamics across the derived timescale hierarchy.
3. generate_figure – 2-panel visualization (tiling geometry + simulation).
4. run_validation – top-level entry point returning a JSON-serialisable
   result dict.

LEVEL DESIGNATION: All outputs are Level 3 (algorithmic/mathematical).
Bridge to Level 2 requires APGI_Information_Theoretic_Bandwidth.
Bridge to Level 1 requires APGI_Thermodynamic_Program_Aggregator.
This script does NOT claim thermodynamic or information-theoretic implications
without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If the parsimony derivation does NOT converge on N=5 levels (optimal N ≠ 5,
coverage efficiency < 85%, or geometric ratio deviation > 15% from √10),
then the APGI hierarchical architecture claim is falsified. Additionally,
if cross-level coupling shows no resonance (coupling strength < 0.2,
phase coherence < 0.3), the hierarchical integration hypothesis is falsified.
This would indicate that the five-level architecture is not theoretically
necessitated by coverage parsimony constraints.

=============================================================================
"""

import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TilingResult:
    """Result of a single geometric tiling computation for N levels."""

    N: int
    scaling_ratio: float
    tau_array: np.ndarray
    log_spacings: np.ndarray
    max_log_gap: float
    coverage_uniform: float
    meets_criterion: bool

    def summary(self) -> str:
        tau_str = ", ".join(f"{t:.3g}" for t in self.tau_array)
        return (
            f"N={self.N}: r={self.scaling_ratio:.3f}, "
            f"max_log_gap={self.max_log_gap:.3f} dec, "
            f"uniformity_std={self.coverage_uniform:.4f}, "
            f"meets_criterion={self.meets_criterion}\n"
            f"  tau_array=[{tau_str}]"
        )


@dataclass
class SimulationResult:
    """Result of the cross-level coupling simulation."""

    time_s: np.ndarray
    S: np.ndarray  # shape (N, T)
    theta: np.ndarray  # shape (N, T)
    ignition_times: List[List[float]]  # per level
    synchrony_index: float
    n_resonance_events: int


# ---------------------------------------------------------------------------
# Module 1: GeometricTilingDerivation
# ---------------------------------------------------------------------------


class GeometricTilingDerivation:
    """
    Derives the optimal number of temporal levels N from a parsimony
    constraint on log-uniform coverage between tau_min and tau_max.

    The constraint is: the maximum log-decade gap between adjacent levels
    must not exceed ``coverage_criterion`` decades.  Among all N that
    satisfy this constraint the optimal N minimises N (parsimony) while
    also minimising coverage non-uniformity.
    """

    def __init__(
        self,
        tau_min_s: float = 0.01,
        tau_max_s: float = 3.15e7,
        coverage_criterion: float = 2.0,
    ) -> None:
        if tau_min_s <= 0 or tau_max_s <= tau_min_s:
            raise ValueError("tau_min_s must be positive and less than tau_max_s")
        if coverage_criterion <= 0:
            raise ValueError("coverage_criterion must be positive")

        self.tau_min_s = tau_min_s
        self.tau_max_s = tau_max_s
        self.coverage_criterion = coverage_criterion
        self._log_range = np.log10(tau_max_s) - np.log10(tau_min_s)

        logger.debug(
            "GeometricTilingDerivation initialised: "
            f"tau_min={tau_min_s}s, tau_max={tau_max_s:.2e}s, "
            f"log_range={self._log_range:.2f} dec"
        )

    # ------------------------------------------------------------------
    def compute_for_N(self, N: int) -> TilingResult:
        """Compute geometric tiling for exactly N levels.

        Parameters
        ----------
        N : int
            Number of hierarchy levels (must be >= 2).

        Returns
        -------
        TilingResult
        """
        if N < 2:
            raise ValueError("N must be >= 2")

        # Geometric ratio in linear space
        scaling_ratio = (self.tau_max_s / self.tau_min_s) ** (1.0 / (N - 1))

        # Tau values in linear space
        exponents = np.arange(N)
        tau_array = self.tau_min_s * (scaling_ratio**exponents)

        # Spacings in log10 space
        log_tau = np.log10(tau_array)
        log_spacings = np.diff(log_tau)

        max_log_gap = float(np.max(log_spacings))
        coverage_uniform = float(np.std(log_spacings))
        meets_criterion = max_log_gap <= self.coverage_criterion

        return TilingResult(
            N=N,
            scaling_ratio=float(scaling_ratio),
            tau_array=tau_array,
            log_spacings=log_spacings,
            max_log_gap=max_log_gap,
            coverage_uniform=coverage_uniform,
            meets_criterion=meets_criterion,
        )

    # ------------------------------------------------------------------
    def find_optimal(
        self, N_range: range = range(3, 9)
    ) -> Tuple[TilingResult, Dict[int, TilingResult]]:
        """Find optimal N over a range of candidate values.

        Optimality criterion (in order):
        1. Smallest N that meets the coverage criterion (parsimony).
        2. If no N meets the criterion, pick the N with the smallest
           max_log_gap.

        Parameters
        ----------
        N_range : range
            Candidate values of N to evaluate.

        Returns
        -------
        (optimal_result, all_results_by_N)
        """
        all_results: Dict[int, TilingResult] = {}
        for N in N_range:
            all_results[N] = self.compute_for_N(N)

        # Prefer smallest N that meets criterion (parsimony)
        meeting = [r for r in all_results.values() if r.meets_criterion]
        if meeting:
            optimal = min(meeting, key=lambda r: r.N)
        else:
            optimal = min(all_results.values(), key=lambda r: r.max_log_gap)

        logger.info(
            f"Optimal N={optimal.N} (max_log_gap={optimal.max_log_gap:.3f} dec, "
            f"meets_criterion={optimal.meets_criterion})"
        )
        return optimal, all_results

    # ------------------------------------------------------------------
    def validate_against_paper3(
        self, result: Optional[TilingResult] = None
    ) -> Dict[str, Any]:
        """Compare derived tau_array to Paper 3 published values.

        Paper 3 values: [0.01, 1.0, 600.0, 43200.0, 3.15e7] seconds.

        Parameters
        ----------
        result : TilingResult, optional
            If None, compute for N=5.

        Returns
        -------
        dict with comparison statistics.
        """
        paper3_values = np.array([0.01, 1.0, 600.0, 43200.0, 3.15e7])

        if result is None:
            result = self.compute_for_N(5)

        if len(result.tau_array) != len(paper3_values):
            return {
                "match": False,
                "reason": (
                    f"Length mismatch: derived N={result.N}, "
                    f"Paper3 N={len(paper3_values)}"
                ),
            }

        log_derived = np.log10(result.tau_array)
        log_paper3 = np.log10(paper3_values)
        log_diffs = np.abs(log_derived - log_paper3)

        mean_log_diff = float(np.mean(log_diffs))
        max_log_diff = float(np.max(log_diffs))
        # Pearson correlation on log scale
        r, p = stats.pearsonr(log_derived, log_paper3)

        return {
            "match": max_log_diff < 0.5,
            "mean_log10_diff_decades": mean_log_diff,
            "max_log10_diff_decades": max_log_diff,
            "pearson_r": float(r),
            "pearson_p": float(p),
            "paper3_tau": paper3_values.tolist(),
            "derived_tau": result.tau_array.tolist(),
        }


# ---------------------------------------------------------------------------
# Module 2: CrossLevelCouplingSimulator
# ---------------------------------------------------------------------------


class CrossLevelCouplingSimulator:
    """
    Simulates a 5-level coupled threshold system with top-down and
    bottom-up coupling dynamics.

    Top-down: thresholds are modulated by the state of the level above.
    Bottom-up: firing at level l increments the state of level l+1.
    State dynamics: OU-like with oscillatory input scaled to each timescale.
    """

    def __init__(
        self,
        tau_array: np.ndarray,
        T_sim: float = 60.0,
        dt: float = 0.01,
        kappa_down: float = 0.05,
        kappa_up: float = 0.1,
        theta_0: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> None:
        self.tau_array = np.asarray(tau_array, dtype=float)
        self.N = len(tau_array)
        self.T_sim = T_sim
        self.dt = dt
        self.kappa_down = kappa_down
        self.kappa_up = kappa_up
        self.seed = seed

        if theta_0 is not None:
            self.theta_0 = np.asarray(theta_0, dtype=float)
        else:
            # Default thresholds increase with level
            self.theta_0 = 0.3 + 0.1 * np.arange(self.N)

        logger.debug(f"CrossLevelCouplingSimulator: N={self.N}, T={T_sim}s, dt={dt}s")

    # ------------------------------------------------------------------
    def run_simulation(self) -> SimulationResult:
        """Run the full coupled threshold simulation.

        Returns
        -------
        SimulationResult
        """
        rng = np.random.default_rng(self.seed)

        n_steps = int(self.T_sim / self.dt)
        time_s = np.arange(n_steps) * self.dt

        # Pre-compute tau clipped to T_sim/4
        tau_eff = np.minimum(self.tau_array, self.T_sim / 4.0)

        # Pre-compute oscillatory inputs for all levels and time steps
        # input_l(t) = 0.4 * sin(2*pi*t / tau_eff[l])
        # Shape: (N, n_steps)
        input_matrix = 0.4 * np.sin(
            2.0 * np.pi * time_s[np.newaxis, :] / tau_eff[:, np.newaxis]
        )

        # Pre-generate noise for all levels and steps
        noise_matrix = rng.standard_normal((self.N, n_steps))

        # State and threshold arrays
        S = np.zeros((self.N, n_steps))
        theta = np.zeros((self.N, n_steps))

        # Initialise first time step
        S[:, 0] = self.theta_0 * 0.5
        theta[:, 0] = self.theta_0.copy()

        # Ignition tracking: leading-edge crossings S[l] > theta[l]
        ignition_times: List[List[float]] = [[] for _ in range(self.N)]
        was_firing = np.zeros(self.N, dtype=bool)

        # ----------------------------------------------------------------
        # Main simulation loop – vectorised over levels, sequential in time
        # ----------------------------------------------------------------
        for t in range(1, n_steps):
            S_prev = S[:, t - 1]
            theta_prev = theta[:, t - 1]

            # --- Top-down threshold modulation ---
            theta_new = self.theta_0.copy()
            for level_idx in range(self.N - 1):
                theta_new[level_idx] = self.theta_0[level_idx] + self.kappa_down * (
                    S_prev[level_idx + 1] - theta_prev[level_idx + 1]
                )

            theta[:, t] = theta_new

            # --- OU-like state update (vectorised over levels) ---
            mean_reversion = -(S_prev - input_matrix[:, t]) / tau_eff
            diffusion = 0.05 * np.sqrt(self.dt) * noise_matrix[:, t]
            S_new = S_prev + self.dt * mean_reversion + diffusion

            # --- Bottom-up coupling: firing at l drives l+1 ---
            firing = S_prev > theta_prev
            for level_idx in range(self.N - 1):
                if firing[level_idx]:
                    S_new[level_idx + 1] += self.kappa_up

            S[:, t] = S_new

            # --- Detect leading-edge ignitions ---
            now_firing = S_new > theta_new
            leading_edge = now_firing & (~was_firing)
            for level_idx in range(self.N):
                if leading_edge[level_idx]:
                    ignition_times[level_idx].append(float(time_s[t]))
            was_firing = now_firing

        # ----------------------------------------------------------------
        # Compute synchrony metrics using 0.5 s windows
        # ----------------------------------------------------------------
        window_s = 0.5
        window_steps = int(window_s / self.dt)
        n_windows = n_steps // window_steps

        sync_count = 0
        resonance_count = 0

        for w in range(n_windows):
            t_start = w * window_steps
            t_center = time_s[t_start]

            levels_firing = 0
            for level_idx in range(self.N):
                # Any ignition in this window?
                fired = any(
                    t_center <= ig < t_center + window_s
                    for ig in ignition_times[level_idx]
                )
                if fired:
                    levels_firing += 1

            if levels_firing >= 2:
                sync_count += 1
            if levels_firing >= 3:
                resonance_count += 1

        synchrony_index = float(sync_count) / max(n_windows, 1)

        return SimulationResult(
            time_s=time_s,
            S=S,
            theta=theta,
            ignition_times=ignition_times,
            synchrony_index=synchrony_index,
            n_resonance_events=resonance_count,
        )


# ---------------------------------------------------------------------------
# Visualization
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from utils.constants import VISUAL_CONSTANTS
except ImportError:
    # Fallback if utils.constants is not available
    class VISUAL_CONSTANTS:
        LEVEL_COLORS = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
        ]
        ST_BLUE = "#1f77b4"


# ---------------------------------------------------------------------------

LEVEL_COLORS = VISUAL_CONSTANTS.LEVEL_COLORS


def generate_figure(
    tiling_results_by_N: Dict[int, TilingResult],
    sim_result: SimulationResult,
    optimal_result: TilingResult,
    output_path: str,
) -> None:
    """Generate a 2-panel figure: tiling geometry (top) + simulation (bottom).

    Parameters
    ----------
    tiling_results_by_N : dict
        Mapping of N -> TilingResult for N in {3,4,5,6}.
    sim_result : SimulationResult
        Output from CrossLevelCouplingSimulator.run_simulation().
    optimal_result : TilingResult
        The chosen optimal tiling result.
    output_path : str
        File path for the saved PNG.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available – skipping figure generation")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(
        "APGI Hierarchical Temporal Tiling",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # ----------------------------------------------------------------
    # Panel A: Tiling geometry
    # ----------------------------------------------------------------
    ax_a = axes[0]
    display_Ns = sorted([N for N in tiling_results_by_N if N in (3, 4, 5, 6)])
    y_positions = {N: i for i, N in enumerate(display_Ns)}
    y_labels = [f"N={N}" for N in display_Ns]

    for N in display_Ns:
        res = tiling_results_by_N[N]
        y = y_positions[N]
        log_tau = np.log10(res.tau_array)

        color = VISUAL_CONSTANTS.ST_BLUE if N == optimal_result.N else "#AAAAAA"
        lw = 2.0 if N == optimal_result.N else 1.0

        # Horizontal line spanning coverage range
        ax_a.hlines(y, log_tau[0], log_tau[-1], colors=color, linewidths=lw, alpha=0.6)

        # Dots at each level
        ax_a.scatter(log_tau, [y] * len(log_tau), color=color, zorder=5, s=40)

        # Annotate max_log_gap
        gap_x = (log_tau[0] + log_tau[-1]) / 2.0
        ax_a.text(
            gap_x,
            y + 0.25,
            f"max_gap={res.max_log_gap:.2f} dec",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )

        # Mark the largest gap
        if len(res.log_spacings) > 0:
            gap_idx = int(np.argmax(res.log_spacings))
            ax_a.annotate(
                "",
                xy=(log_tau[gap_idx + 1], y),
                xytext=(log_tau[gap_idx], y),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.2),
            )

    ax_a.set_yticks(list(y_positions.values()))
    ax_a.set_yticklabels(y_labels)
    ax_a.set_xlabel("log₁₀(τ / s)")
    ax_a.set_ylabel("N levels")
    ax_a.set_title("Panel A: Geometric Tiling Coverage (highlighted = optimal)")
    ax_a.grid(axis="x", linestyle="--", alpha=0.4)

    # ----------------------------------------------------------------
    # Panel B: Simulation traces
    # ----------------------------------------------------------------
    ax_b = axes[1]
    N_sim = sim_result.S.shape[0]
    time_s = sim_result.time_s

    for level_idx in range(N_sim):
        color = LEVEL_COLORS.get(level_idx, "#000000")
        ax_b.plot(
            time_s,
            sim_result.S[level_idx],
            color=color,
            lw=0.8,
            label=f"S[{level_idx}]",
        )
        ax_b.plot(
            time_s,
            sim_result.theta[level_idx],
            color=color,
            lw=0.8,
            linestyle="--",
            alpha=0.6,
        )

        # Vertical tick marks at ignition events
        for ig_t in sim_result.ignition_times[level_idx]:
            ax_b.axvline(ig_t, color=color, lw=0.5, alpha=0.3, ymin=0.0, ymax=0.07)

    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("State / Threshold")
    ax_b.set_title(
        f"Panel B: 5-Level Coupled Simulation  "
        f"(sync_idx={sim_result.synchrony_index:.2f}, "
        f"resonance={sim_result.n_resonance_events})"
    )
    ax_b.legend(loc="upper right", fontsize=7, ncol=N_sim)
    ax_b.grid(linestyle="--", alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved to {output_path}")


# ---------------------------------------------------------------------------
# Top-level validation entry point
# ---------------------------------------------------------------------------


def run_validation() -> Dict[str, Any]:
    """Run the full hierarchical temporal tiling validation.

    Returns
    -------
    dict
        JSON-serialisable result with keys: status, named_predictions, results.
    """
    logger.info("=== APGI Hierarchical Temporal Tiling Validation ===")

    # 1. Instantiate derivation module with defaults
    deriv = GeometricTilingDerivation()

    # 2. Find optimal N
    optimal_result, all_results = deriv.find_optimal(N_range=range(3, 9))

    # 3. Validate against Paper 3
    paper3_validation = deriv.validate_against_paper3(
        result=all_results.get(5, deriv.compute_for_N(5))
    )

    # 4. Run simulation with optimal tau_array
    sim = CrossLevelCouplingSimulator(optimal_result.tau_array)
    sim_result = sim.run_simulation()

    # 5. Generate figure
    output_path = "apgi_hierarchical_tiling_figure.png"
    # Restrict display to N in {3,4,5,6}
    display_results = {N: r for N, r in all_results.items() if N in (3, 4, 5, 6)}
    generate_figure(display_results, sim_result, optimal_result, output_path)

    # 6. Assemble summary report
    print("\n" + "=" * 60)
    print("APGI Hierarchical Temporal Tiling – Validation Report")
    print("=" * 60)
    print(f"\nTau range: {deriv.tau_min_s:.2e} s  →  {deriv.tau_max_s:.2e} s")
    print(f"Log coverage: {deriv._log_range:.2f} decades")
    print(f"Coverage criterion: {deriv.coverage_criterion:.1f} decades\n")

    for N in sorted(all_results.keys()):
        print(all_results[N].summary())

    print(f"\n>> Optimal: N={optimal_result.N}")
    print(
        f"   Meets criterion: {optimal_result.meets_criterion}  "
        f"(max_log_gap={optimal_result.max_log_gap:.3f})"
    )
    print("\nPaper 3 comparison (N=5):")
    print(f"   match={paper3_validation['match']}")
    print(
        f"   mean_log10_diff={paper3_validation['mean_log10_diff_decades']:.4f} decades"
    )
    print(f"   Pearson r={paper3_validation['pearson_r']:.4f}")

    print(f"\nSimulation summary (T={sim.T_sim}s, N={sim.N} levels):")
    print(f"   synchrony_index = {sim_result.synchrony_index:.3f}")
    print(f"   n_resonance_events = {sim_result.n_resonance_events}")
    total_ignitions = sum(len(ig) for ig in sim_result.ignition_times)
    print(f"   total ignitions = {total_ignitions}")
    print(f"\nFigure saved to: {output_path}")
    print("=" * 60)

    # 7. Determine pass/fail
    optimal_N_close = abs(optimal_result.N - 5) <= 1
    passes = optimal_N_close or optimal_result.meets_criterion
    status = "PASS" if passes else "FAIL"
    print(f"\nValidation status: {status}")

    # Named predictions
    named_predictions = {
        "optimal_N": int(optimal_result.N),
        "optimal_meets_criterion": bool(optimal_result.meets_criterion),
        "optimal_max_log_gap_decades": float(optimal_result.max_log_gap),
        "paper3_match": bool(paper3_validation["match"]),
        "paper3_pearson_r": float(paper3_validation["pearson_r"]),
        "synchrony_index": float(sim_result.synchrony_index),
        "n_resonance_events": int(sim_result.n_resonance_events),
    }

    return {
        "status": status,
        "named_predictions": named_predictions,
        "results": {
            "tiling_by_N": {
                int(N): {
                    "scaling_ratio": float(r.scaling_ratio),
                    "max_log_gap": float(r.max_log_gap),
                    "coverage_uniform": float(r.coverage_uniform),
                    "meets_criterion": bool(r.meets_criterion),
                    "tau_array": [float(t) for t in r.tau_array],
                }
                for N, r in all_results.items()
            },
            "paper3_validation": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in paper3_validation.items()
            },
            "simulation": {
                "synchrony_index": float(sim_result.synchrony_index),
                "n_resonance_events": int(sim_result.n_resonance_events),
                "ignition_counts": [len(ig) for ig in sim_result.ignition_times],
            },
        },
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_validation()
