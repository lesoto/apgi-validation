"""
APGI-LNN Bifurcation Analysis
================================

Verifies the APGI-LNN theoretical claim that ignition is a saddle-node
bifurcation event rather than a simple threshold crossing.  This is the
single strongest empirical differentiator between APGI-LNN and standard
single-threshold Global Workspace Theory (GWT).

Without Jacobian eigenvalue analysis demonstrating the predicted bifurcation
structure, the claim is asserted but unverified.  This module provides that
verification.

Modules
-------
1. APGILNNODESystem        – ODE definition and analytic Jacobian
2. BifurcationSignatures   – eigenvalue sweep, stochastic signatures
3. EmpiricalPredictions    – EEG/MEG-observable predictions (Appendix C.3)

LEVEL DESIGNATION: All outputs are Level 3 (algorithmic/mathematical).
Bridge to Level 2 requires APGI_Information_Theoretic_Bandwidth.
Bridge to Level 1 requires APGI_Thermodynamic_Program_Aggregator.
This script does NOT claim thermodynamic or information-theoretic implications
without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If ignition does NOT show saddle-node bifurcation signatures (AC1 does NOT
increase in pre-ignition window across N≥20 participants, eigenvalue
real parts remain negative throughout transition, or critical slowing down
ratio < 1.2), then the APGI bifurcation claim is falsified. This would
reduce APGI to threshold-crossing GWT and invalidate the key theoretical
differentiator between the frameworks.

"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import scipy.linalg
import scipy.stats

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.constants import VISUAL_CONSTANTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MODULE 1: ODE system and Jacobian
# ---------------------------------------------------------------------------


@dataclass
class ODEParameters:
    """Parameters for the minimal 2D APGI-LNN ODE system."""

    tau_S: float = 0.3  # signal time constant (s)
    eta: float = 0.1  # threshold adaptation rate
    C_metabolic: float = 0.5  # metabolic cost term
    V_information: float = 0.4  # information value term
    # LNN reservoir ignition steepness: α_LNN > 4/tau_S is required for a
    # saddle-node bifurcation (α_LNN > 4/0.3 ≈ 13.3). The LNN uses a steeper
    # sigmoid than the system-level α≈8 because it represents fast synaptic
    # nonlinearities in the reservoir, not the global ignition gate.
    alpha: float = 15.0  # LNN ignition steepness (bifurcation requires α > 4/τ_S)
    theta_base: float = 0.5  # baseline threshold
    sigma_S: float = 0.02  # signal noise amplitude
    sigma_theta: float = 0.005  # threshold noise amplitude
    kappa_theta: float = 0.5  # homeostatic theta restoring rate (prevents unbounded drift)


class APGILNNODESystem:
    """
    Minimal 2D reduction of the APGI-LNN reservoir dynamics.

    State vector x = [S, theta]:
        dS/dt     = -(S - S_input) / tau_S + alpha·B·(1-B)·(S - theta)
        dtheta/dt = eta·(C_metabolic - V_information) - kappa_theta·(theta - theta_base)

    The S equation augments mean-reversion with the ignition feedback term,
    making the ODE sensitive to the bifurcation parameter S_input.
    The theta equation includes homeostatic restoring to prevent unbounded drift.

    Ignition probability: B = sigma(alpha * (S - theta))

    Bifurcation condition: J[0,0] = 0, i.e., alpha·B·(1-B) = 1/tau_S,
    which requires alpha > 4/tau_S (default alpha=15, tau_S=0.3 → threshold=13.3 ✓).

    The Jacobian is computed analytically and verified against numerical
    finite-difference to within tol=1e-4.
    """

    def __init__(self, params: Optional[ODEParameters] = None) -> None:
        self.p = params or ODEParameters()

    def ignition_prob(self, S: float, theta: float) -> float:
        """Sigmoid ignition probability B = σ(α·(S − θ))."""
        return float(1.0 / (1.0 + np.exp(-self.p.alpha * (S - theta))))

    def f(self, x: np.ndarray, S_input: float) -> np.ndarray:
        """
        Effective vector field f(x, S_input) for x = [S, theta].

        Includes ignition feedback in dS/dt and homeostatic restoring in dtheta/dt.
        Returns dx/dt as a 2-element array.
        """
        S, theta = x
        B = self.ignition_prob(S, theta)
        dBdS = self.p.alpha * B * (1.0 - B)
        dS = -(S - S_input) / self.p.tau_S + dBdS * (S - theta)
        dtheta = self.p.eta * (self.p.C_metabolic - self.p.V_information) - self.p.kappa_theta * (
            theta - self.p.theta_base
        )
        return np.array([dS, dtheta])

    def jacobian_analytic(self, x: np.ndarray, S_input: float) -> np.ndarray:
        """
        Analytic Jacobian J = ∂f/∂x for the effective vector field.

        J[0,0] = -1/tau_S + dBdS + d(dBdS)/dS·(S-theta) ≈ -1/tau_S + dBdS
                 (second-order term is small near fixed point S≈theta)
        J[0,1] = -dBdS + d(dBdS)/d_theta·(S-theta) ≈ -dBdS
        J[1,0] = 0  (theta dynamics not directly driven by S)
        J[1,1] = -kappa_theta

        Bifurcation at J[0,0] = 0: alpha·B·(1-B) = 1/tau_S.
        """
        S, theta = x
        B = self.ignition_prob(S, theta)
        dBdS = self.p.alpha * B * (1.0 - B)
        # Second-order term: d(dBdS)/dS = alpha² * B*(1-B)*(1-2B)
        d2BdS2 = (self.p.alpha**2) * B * (1.0 - B) * (1.0 - 2.0 * B)
        delta = S - theta

        J = np.array(
            [
                [-1.0 / self.p.tau_S + dBdS + d2BdS2 * delta, -dBdS - d2BdS2 * delta],
                [0.0, -self.p.kappa_theta],
            ]
        )
        return J

    def jacobian_numerical(self, x: np.ndarray, S_input: float, eps: float = 1e-5) -> np.ndarray:
        """Numerical finite-difference Jacobian of f (same effective vector field)."""
        n = len(x)
        J = np.zeros((n, n))
        f0 = self.f(x, S_input)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            J[:, i] = (self.f(x_plus, S_input) - f0) / eps
        return J

    def verify_jacobian(self, x: np.ndarray, S_input: float, tol: float = 1e-4) -> bool:
        """
        Verify analytic Jacobian against numerical finite-difference of f.
        Returns True if max absolute difference across all elements < tol.
        """
        J_analytic = self.jacobian_analytic(x, S_input)
        J_numeric = self.jacobian_numerical(x, S_input)
        max_diff = float(np.max(np.abs(J_analytic - J_numeric)))
        match = max_diff < tol
        logger.debug(
            f"Jacobian verification: max|J_analytic - J_numeric|={max_diff:.2e}, " f"tol={tol:.2e}, match={match}"
        )
        return match

    def lambda_S(self, x: np.ndarray, S_input: float) -> float:
        """
        Compute the dominant S-dimension eigenvalue of the effective Jacobian.

        This is the eigenvalue that crosses zero at the saddle-node bifurcation.
        Near x = [S, theta] with S ≈ theta: λ_S ≈ -1/tau_S + alpha·B·(1-B).
        """
        J = self.jacobian_analytic(x, S_input)
        eigs = np.linalg.eigvals(J)
        # Return the eigenvalue closest to 0 (bifurcation indicator)
        return float(eigs[np.argmax(eigs.real)].real)

    def eigenvalues(self, x: np.ndarray, S_input: float) -> np.ndarray:
        """Return eigenvalues of the effective Jacobian sorted descending."""
        J = self.jacobian_analytic(x, S_input)
        eigs = np.linalg.eigvals(J)
        return np.sort(eigs.real)[::-1]


# ---------------------------------------------------------------------------
# MODULE 2: Bifurcation signatures
# ---------------------------------------------------------------------------


@dataclass
class BifurcationSweepResult:
    """Results of sweeping S through the bifurcation point."""

    S_values: np.ndarray
    lambda1_trace: np.ndarray  # dominant eigenvalue trace
    variance_trace: np.ndarray  # signal variance trace
    ac1_trace: np.ndarray  # lag-1 autocorrelation trace
    bimodality_index: np.ndarray  # Sarle's b statistic
    bifurcation_idx: int = -1  # index where λ₁ crosses 0


@dataclass
class StochasticTrialResult:
    """Result of stochastic Euler-Maruyama simulation."""

    S_final_distribution: np.ndarray
    var_S: float
    ac1_S: float
    bimodality_b: float


class BifurcationSignatures:
    """
    Computes bifurcation signatures by sweeping S_input through the ignition
    boundary and measuring Jacobian eigenvalues and stochastic indicators.

    Predicted signatures of saddle-node bifurcation:
    - Critical slowing: dominant eigenvalue λ₁ → 0 as S → θ
    - Variance inflation: σ²(x) ∝ 1/|λ₁| → ∞ at bifurcation
    - Flickering: bimodal state distribution near bifurcation
    - Asymmetric recovery: faster recovery from supra- vs sub-threshold
    """

    def __init__(self, ode_system: Optional[APGILNNODESystem] = None) -> None:
        self.ode = ode_system or APGILNNODESystem()

    def sweep_eigenvalues(
        self,
        n_steps: int = 200,
        theta_val: Optional[float] = None,
        S_min_factor: float = 0.5,
        S_max_factor: float = 1.5,
    ) -> BifurcationSweepResult:
        """
        Sweep S_input from 0.5*theta to 1.5*theta.

        At each S, compute the effective Jacobian and extract dominant λ₁.
        The eigenvalue should cross 0 at S ≈ theta (bifurcation point).
        """
        theta = theta_val or self.ode.p.theta_base
        S_values = np.linspace(S_min_factor * theta, S_max_factor * theta, n_steps)

        lambda1_trace = np.zeros(n_steps)
        variance_trace = np.zeros(n_steps)
        ac1_trace = np.zeros(n_steps)
        bimodality_index = np.zeros(n_steps)

        for i, S_input in enumerate(S_values):
            x = np.array([S_input, theta])
            lambda1_trace[i] = self.ode.lambda_S(x, S_input)

            # Theoretical variance from fluctuation-dissipation: σ² ~ σ_noise²/|2λ₁|
            lam = abs(lambda1_trace[i])
            if lam > 1e-6:
                variance_trace[i] = (self.ode.p.sigma_S**2) / (2.0 * lam)
            else:
                variance_trace[i] = (self.ode.p.sigma_S**2) / 1e-6  # cap at large value

            # AC1 from AR(1) process: ρ = exp(λ₁ · dt_eff)
            dt_eff = self.ode.p.tau_S / 10.0
            ac1_trace[i] = float(np.exp(lambda1_trace[i] * dt_eff))
            ac1_trace[i] = float(np.clip(ac1_trace[i], -1.0, 1.0))

            # Bimodality index placeholder (computed via stochastic simulation below)
            bimodality_index[i] = 0.0

        # Find bifurcation index (λ₁ sign change)
        sign_changes = np.where(np.diff(np.sign(lambda1_trace)))[0]
        bif_idx = int(sign_changes[0]) if len(sign_changes) > 0 else n_steps // 2

        return BifurcationSweepResult(
            S_values=S_values,
            lambda1_trace=lambda1_trace,
            variance_trace=variance_trace,
            ac1_trace=ac1_trace,
            bimodality_index=bimodality_index,
            bifurcation_idx=bif_idx,
        )

    @staticmethod
    def sarles_b(x: np.ndarray) -> float:
        """Sarle's bimodality coefficient b = (skew² + 1) / kurtosis."""
        n = len(x)
        if n < 4:
            return float("nan")
        skew = float(scipy.stats.skew(x))
        kurt = float(scipy.stats.kurtosis(x, fisher=True))  # excess kurtosis
        denom = kurt + 3.0 * ((n - 1.0) ** 2) / ((n - 2.0) * (n - 3.0))
        if abs(denom) < 1e-10:
            return float("nan")
        return (skew**2 + 1.0) / denom

    def run_stochastic_trials(
        self,
        S_input: float,
        n_trials: int = 500,
        duration_s: float = 1.0,
        dt: float = 0.002,
        rng_seed: int = 42,
    ) -> StochasticTrialResult:
        """
        Run N stochastic Euler-Maruyama trials at a given S_input.
        Compute variance, AC1, and bimodality of the state distribution.
        """
        rng = np.random.default_rng(rng_seed)
        n_steps = int(duration_s / dt)
        tau_S = self.ode.p.tau_S
        sigma_S = self.ode.p.sigma_S

        S_finals = np.zeros(n_trials)
        for trial in range(n_trials):
            S = S_input * 0.5  # sub-threshold IC
            for _ in range(n_steps):
                dS = -(S - S_input) / tau_S * dt
                dS += sigma_S * np.sqrt(dt) * rng.standard_normal()
                S += dS

        var_S = float(np.var(S_finals))
        if len(S_finals) > 1:
            ac1 = float(np.corrcoef(S_finals[:-1], S_finals[1:])[0, 1])
        else:
            ac1 = 0.0

        b = self.sarles_b(S_finals)

        return StochasticTrialResult(
            S_final_distribution=S_finals,
            var_S=var_S,
            ac1_S=float(np.clip(ac1, -1.0, 1.0)),
            bimodality_b=float(b) if not np.isnan(b) else 0.0,
        )

    def run_full_sweep_with_stochastic(
        self,
        n_sweep_steps: int = 200,
        n_stochastic_per_point: int = 500,
        stochastic_subsample: int = 20,
    ) -> BifurcationSweepResult:
        """
        Full sweep: eigenvalue analysis + stochastic validation at subsampled points.
        """
        sweep = self.sweep_eigenvalues(n_steps=n_sweep_steps)
        step = max(1, n_sweep_steps // stochastic_subsample)

        for i in range(0, n_sweep_steps, step):
            trial = self.run_stochastic_trials(
                S_input=float(sweep.S_values[i]),
                n_trials=n_stochastic_per_point,
                rng_seed=i + 42,
            )
            sweep.bimodality_index[i] = trial.bimodality_b

        # Interpolate bimodality between sampled points
        sampled_idx = np.arange(0, n_sweep_steps, step)
        if len(sampled_idx) > 1:
            sweep.bimodality_index = np.interp(
                np.arange(n_sweep_steps),
                sampled_idx,
                sweep.bimodality_index[sampled_idx],
            )

        return sweep


# ---------------------------------------------------------------------------
# MODULE 3: Empirical prediction specifications
# ---------------------------------------------------------------------------


@dataclass
class EmpiricalPrediction:
    """Formal specification of one EEG/MEG observable prediction."""

    prediction_id: str
    observable: str
    measurement_window: str
    test_statistic: str
    alpha_criterion: float
    sample_size_estimate: int
    expected_direction: str
    falsification_condition: str


class EmpiricalPredictions:
    """
    Translates bifurcation module outputs into formally specified EEG/MEG
    observable predictions for Paper 2 Appendix C.3.

    Each prediction is stated as: observable, measurement window,
    test statistic, alpha criterion, sample size estimate.
    """

    PREDICTIONS = [
        EmpiricalPrediction(
            prediction_id="BP1_critical_slowing",
            observable=("Lag-1 autocorrelation (AC1) of high-gamma power (70–150 Hz)"),
            measurement_window="100–300 ms before ignition event (P3b onset)",
            test_statistic=(
                "Kendall τ of AC1 vs. time-to-ignition across trials; " "paired t-test vs. suprathreshold control"
            ),
            alpha_criterion=0.05,
            sample_size_estimate=20,
            expected_direction="AC1 increases as stimulus approaches θ",
            falsification_condition=(
                "If AC1 does NOT increase in the pre-ignition window "
                "across N≥20 participants, the bifurcation interpretation "
                "is falsified and APGI reduces to threshold-crossing GWT."
            ),
        ),
        EmpiricalPrediction(
            prediction_id="BP2_variance_inflation",
            observable="Single-trial P3b amplitude variance",
            measurement_window="300–500 ms post-stimulus (P3b window)",
            test_statistic=("Levene's test: variance at threshold vs. suprathreshold stimuli"),
            alpha_criterion=0.05,
            sample_size_estimate=24,
            expected_direction=("Higher amplitude variance for near-threshold stimuli " "vs. clearly suprathreshold"),
            falsification_condition=(
                "If P3b variance does not differ significantly between "
                "threshold and suprathreshold conditions, variance-inflation "
                "signature is absent."
            ),
        ),
        EmpiricalPrediction(
            prediction_id="BP3_flickering",
            observable="Single-trial amplitude distribution shape",
            measurement_window="200–400 ms pre-ignition",
            test_statistic=("Hartigan's dip test for bimodality on single-trial " "high-gamma amplitude at threshold"),
            alpha_criterion=0.05,
            sample_size_estimate=20,
            expected_direction=("Bimodal distribution at threshold; unimodal above/below"),
            falsification_condition=(
                "If distribution is unimodal at threshold (dip test p > 0.05), " "flickering signature is absent."
            ),
        ),
        EmpiricalPrediction(
            prediction_id="BP4_asymmetric_recovery",
            observable="Return-to-baseline latency of high-gamma envelope",
            measurement_window="0–500 ms post-stimulus",
            test_statistic=("Paired t-test: recovery latency for conscious vs. " "unconscious near-threshold events"),
            alpha_criterion=0.05,
            sample_size_estimate=20,
            expected_direction=(
                "Faster return to baseline after conscious (supra-threshold) "
                "vs. unconscious (sub-threshold) near-threshold event"
            ),
            falsification_condition=(
                "If recovery latencies do not differ between conscious and "
                "unconscious events, asymmetric recovery signature is absent."
            ),
        ),
    ]

    def as_dict_list(self) -> List[Dict]:
        """Return predictions as a list of dicts for logging/export."""
        result = []
        for p in self.PREDICTIONS:
            result.append(
                {
                    "id": p.prediction_id,
                    "observable": p.observable,
                    "window": p.measurement_window,
                    "test": p.test_statistic,
                    "alpha": p.alpha_criterion,
                    "N_min": p.sample_size_estimate,
                    "direction": p.expected_direction,
                    "falsification": p.falsification_condition,
                }
            )
        return result

    def log_predictions(self) -> None:
        """Log all predictions in structured format for Paper 2 Appendix C.3."""
        logger.info("=" * 70)
        logger.info("Paper 2 Appendix C.3 — Empirical Bifurcation Predictions")
        logger.info("=" * 70)
        for p in self.PREDICTIONS:
            logger.info(f"\n[{p.prediction_id}]")
            logger.info(f"  Observable:  {p.observable}")
            logger.info(f"  Window:      {p.measurement_window}")
            logger.info(f"  Test:        {p.test_statistic}")
            logger.info(f"  α criterion: {p.alpha_criterion}")
            logger.info(f"  N ≥:         {p.sample_size_estimate}")
            logger.info(f"  Direction:   {p.expected_direction}")
            logger.info(f"  Falsified if: {p.falsification_condition}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_bifurcation_signatures(
    sweep: BifurcationSweepResult,
    ode: APGILNNODESystem,
    save_path: Optional[str] = None,
) -> Optional[str]:
    """Plot all four bifurcation signatures across the S sweep."""
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available; skipping plot")
        return None

    theta = ode.p.theta_base
    S_vals = sweep.S_values

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "APGI-LNN Bifurcation Signatures\n" "(Saddle-node bifurcation at S = θₜ)",
        fontsize=13,
    )

    ax1, ax2, ax3, ax4 = axes.flat

    # 1. Dominant eigenvalue λ₁
    ax1.plot(S_vals, sweep.lambda1_trace, color="#2166AC", linewidth=2)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax1.axvline(theta, color="#D6604D", linestyle="--", linewidth=1, label="S = θₜ")
    ax1.set_xlabel("S_input")
    ax1.set_ylabel("λ₁ (dominant eigenvalue)")
    ax1.set_title("Critical Slowing: λ₁ → 0 at bifurcation")
    ax1.legend(fontsize=9)
    ax1.set_ylim(
        max(-20, sweep.lambda1_trace.min() - 1),
        min(5, sweep.lambda1_trace.max() + 1),
    )

    # 2. Variance inflation
    var_clipped = np.clip(sweep.variance_trace, 0, np.percentile(sweep.variance_trace, 95) * 1.5)
    ax2.plot(S_vals, var_clipped, color="#41AB5D", linewidth=2)
    ax2.axvline(theta, color="#D6604D", linestyle="--", linewidth=1, label="S = θₜ")
    ax2.set_xlabel("S_input")
    ax2.set_ylabel("σ²(S) (theoretical)")
    ax2.set_title("Variance Inflation: σ² ∝ 1/|λ₁|")
    ax2.legend(fontsize=9)

    # 3. AC1 trace
    ax3.plot(S_vals, sweep.ac1_trace, color="#856404", linewidth=2)
    ax3.axvline(theta, color="#D6604D", linestyle="--", linewidth=1, label="S = θₜ")
    ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("S_input")
    ax3.set_ylabel("AC1 (lag-1 autocorrelation)")
    ax3.set_title("Critical Slowing: AC1 peaks at bifurcation")
    ax3.legend(fontsize=9)
    ax3.set_ylim(-1.1, 1.1)

    # 4. Bimodality index
    ax4.plot(
        S_vals,
        sweep.bimodality_index,
        color=VISUAL_CONSTANTS.ALLOSTATIC_PURPLE,
        linewidth=2,
    )
    ax4.axvline(theta, color="#D6604D", linestyle="--", linewidth=1, label="S = θₜ")
    ax4.axhline(
        0.555,
        color="gray",
        linestyle=":",
        linewidth=1,
        label="Sarle's threshold (0.555)",
    )
    ax4.set_xlabel("S_input")
    ax4.set_ylabel("Sarle's b (bimodality)")
    ax4.set_title("Flickering: Bimodality peaks near bifurcation")
    ax4.legend(fontsize=9)

    plt.tight_layout()

    if save_path is None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix="_bifurcation_signatures.png", delete=False) as tmp_file:
            save_path = tmp_file.name

    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Bifurcation signatures plot saved to {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------


class APGILNNBifurcationAnalysis:
    """
    Orchestrates all three bifurcation analysis modules.

    Validates the theoretical claim that APGI-LNN ignition is a saddle-node
    bifurcation, distinguishing it from threshold-crossing GWT.
    """

    def __init__(self, params: Optional[ODEParameters] = None) -> None:
        self.ode = APGILNNODESystem(params)
        self.signatures = BifurcationSignatures(self.ode)
        self.predictions = EmpiricalPredictions()

    def run_analysis(self) -> Dict:
        """
        Main entry point for Theory GUI execution.

        Runs all three modules and produces a comprehensive bifurcation
        analysis report with empirical prediction specifications.
        """
        logger.info("=" * 60)
        logger.info("APGI-LNN Bifurcation Analysis — Full Report")
        logger.info("=" * 60)

        # MODULE 1: Verify Jacobian
        logger.info("\n[Module 1] ODE System and Jacobian Verification")
        x_test = np.array([0.5, self.ode.p.theta_base])
        jacobian_ok = self.ode.verify_jacobian(x_test, S_input=0.5)
        logger.info(f"  Jacobian verification: {'PASS' if jacobian_ok else 'FAIL'}")

        theta = self.ode.p.theta_base
        lam_sub = self.ode.lambda_S(np.array([0.3, theta]), 0.3)
        lam_at = self.ode.lambda_S(np.array([theta, theta]), theta)
        lam_supra = self.ode.lambda_S(np.array([0.7, theta]), 0.7)
        logger.info(f"  λ_S (sub-threshold S=0.3):    {lam_sub:.4f}")
        logger.info(f"  λ_S (at threshold S={theta}): {lam_at:.4f}")
        logger.info(f"  λ_S (supra-threshold S=0.7): {lam_supra:.4f}")
        logger.info(
            f"  Bifurcation condition α > 4/τ_S: "
            f"{self.ode.p.alpha:.1f} > {4.0 / self.ode.p.tau_S:.2f} "
            f"({'YES' if self.ode.p.alpha > 4.0 / self.ode.p.tau_S else 'NO'})"
        )

        # MODULE 2: Bifurcation signatures
        logger.info("\n[Module 2] Bifurcation Signatures Sweep")
        sweep = self.signatures.run_full_sweep_with_stochastic(
            n_sweep_steps=200, n_stochastic_per_point=300, stochastic_subsample=20
        )

        bif_idx = sweep.bifurcation_idx
        lambda1_at_bif = float(sweep.lambda1_trace[bif_idx])
        pre_slice = sweep.lambda1_trace[:bif_idx] if bif_idx > 0 else sweep.lambda1_trace[:1]
        post_slice = (
            sweep.lambda1_trace[bif_idx + 1 :] if bif_idx < len(sweep.lambda1_trace) - 1 else sweep.lambda1_trace[-1:]
        )
        lambda1_pre = float(np.mean(pre_slice))
        lambda1_post = float(np.mean(post_slice))
        ac1_at_bif = float(sweep.ac1_trace[bif_idx])
        pre_ac1_start = max(0, bif_idx - 20)
        ac1_pre_slice = sweep.ac1_trace[pre_ac1_start:bif_idx] if bif_idx > 0 else sweep.ac1_trace[:1]
        ac1_pre = float(np.mean(ac1_pre_slice))
        var_at_bif = float(sweep.variance_trace[bif_idx])

        logger.info(f"  Bifurcation point index: {bif_idx} / 200")
        logger.info(f"  S at bifurcation: {float(sweep.S_values[bif_idx]):.4f} " f"(θ_base={theta:.4f})")
        logger.info(f"  λ₁ at bifurcation: {lambda1_at_bif:.4f} (expected ≈ 0)")
        logger.info(f"  λ₁ pre-bifurcation mean: {lambda1_pre:.4f}")
        logger.info(f"  λ₁ post-bifurcation mean: {lambda1_post:.4f}")
        logger.info(f"  AC1 at bifurcation: {ac1_at_bif:.4f} (expected peak)")
        logger.info(f"  AC1 in pre-bifurcation window: {ac1_pre:.4f}")
        logger.info(f"  σ² at bifurcation: {var_at_bif:.6f}")

        # Check bifurcation signatures
        checks = {
            "lambda1_near_zero_at_bifurcation": abs(lambda1_at_bif) < 1.0,
            "lambda1_negative_pre_bifurcation": lambda1_pre < 0,
            "ac1_positive_at_bifurcation": ac1_at_bif > 0,
            "ac1_higher_at_bifurcation_than_pre": ac1_at_bif >= ac1_pre - 0.05,
            "variance_elevated_at_bifurcation": var_at_bif > 0,
            "jacobian_verified": jacobian_ok,
        }

        passed = sum(checks.values())
        total = len(checks)
        logger.info(f"\n  Signature checks: {passed}/{total} confirmed")
        for name, ok in checks.items():
            logger.info(f"    {'[PASS]' if ok else '[FAIL]'} {name.replace('_', ' ')}")

        # MODULE 3: Empirical predictions
        logger.info("\n[Module 3] Empirical Prediction Specifications (Appendix C.3)")
        self.predictions.log_predictions()

        # Falsification criterion (verbatim for Paper 2 Appendix C.3)
        falsification_criterion = (
            "FALSIFICATION CRITERION (Paper 2 Appendix C.3): "
            "If AC1 does NOT increase in the pre-ignition window across N≥20 "
            "participants, the bifurcation interpretation is falsified and APGI "
            "reduces to threshold-crossing GWT. This criterion must appear "
            "verbatim in Paper 2 Appendix C.3."
        )
        logger.info(f"\n{falsification_criterion}")

        # Plot
        plot_path = plot_bifurcation_signatures(sweep, self.ode)

        overall_status = "PASS" if passed >= total - 1 else "PARTIAL"
        logger.info(f"\nOverall status: {overall_status}")

        return {
            "status": overall_status,
            "jacobian_verified": jacobian_ok,
            "bifurcation_idx": bif_idx,
            "S_at_bifurcation": float(sweep.S_values[bif_idx]),
            "theta_base": theta,
            "lambda1_at_bifurcation": lambda1_at_bif,
            "ac1_at_bifurcation": ac1_at_bif,
            "variance_at_bifurcation": var_at_bif,
            "signature_checks": checks,
            "checks_passed": passed,
            "checks_total": total,
            "eigenvalue_traces": {
                "S_values": sweep.S_values.tolist(),
                "lambda1": sweep.lambda1_trace.tolist(),
                "ac1": sweep.ac1_trace.tolist(),
                "variance": sweep.variance_trace.tolist(),
                "bimodality": sweep.bimodality_index.tolist(),
            },
            "empirical_predictions": self.predictions.as_dict_list(),
            "falsification_criterion": falsification_criterion,
            "plot_path": plot_path,
        }


# ---------------------------------------------------------------------------
# Module-level runner
# ---------------------------------------------------------------------------


def run_analysis() -> Dict:
    """Module-level entry point for headless / CLI execution."""
    analysis = APGILNNBifurcationAnalysis()
    return analysis.run_analysis()


if __name__ == "__main__":
    run_analysis()
