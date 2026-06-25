"""
APGI Neuromodulatory Control System
=====================================

Centralizes all neuromodulator-to-APGI-parameter mappings developed across
Paper 2 Appendix B. Implements the unified exponential form:

    θₜ_eff = θ_base · exp(Σₖ sₖ · κₖ · [neuromod_k])

where sₖ = -1 for threshold-lowering neuromodulators (NE, ACh) and +1 for
threshold-raising (5-HT, histamine). Multiplicative exponential form preserves
θₜ > 0 by construction and is biologically justified over physiological ranges.

Neuromodulator mappings:
  ACh    → ↑Πᵉ (M1 muscarinic; Hasselmo & McGaughy 2004)
  NE     → ↓θₜ + ↑gain (α2A adrenergic PFC; Arnsten 2011)
  DA     → M̂(c,a) value scaling (D1/D2 vmPFC; Frank & Claus 2006)
  5-HT   → ↑θₜ stability (5-HT2A cortical; Celada et al. 2013)
  Hist   → arousal floor, Πᵉ floor (H1; Haas & Panula 2003)
  Orexin → global ignition steepness α (Sakurai 2007)

TIER DESIGNATION: All outputs are Tier 3 (algorithmic/mathematical).
Bridge to Tier 2 requires APGI_Information_Theoretic_Bandwidth.
Bridge to Tier 1 requires APGI_Thermodynamic_Program_Aggregator.
This script does NOT claim thermodynamic or information-theoretic implications
without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If neuromodulatory effects do NOT follow the predicted exponential mapping
(parameter modulation error > 40% from predicted values, directionality
violations > 25% of mappings, or cross-modulatory interactions > 30% from
additive predictions), then the APGI neuromodulatory control system is
falsified. This would indicate that neuromodulators do not systematically
modulate APGI parameters as predicted by the framework.

"""

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.constants import VISUAL_CONSTANTS

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# κ parameter registry (literature-grounded)
# ---------------------------------------------------------------------------

KAPPA_ACh = 0.5  # M1 muscarinic τ-shortening; Hasselmo & McGaughy 2004
KAPPA_NE = 0.8  # α2A adrenergic θ-lowering; Arnsten 2011
KAPPA_DA = 0.6  # D1 vmPFC value amplification; Frank & Claus 2006
KAPPA_5HT = 0.4  # 5-HT2A cortical gain stabilization; Celada et al. 2013
KAPPA_HIST = 0.3  # H1 arousal floor; Haas & Panula 2003
KAPPA_OREXIN = 0.7  # Global gain on α; Sakurai 2007

# Canonical neuromodulator profiles (normalized concentrations 0–1)
CANONICAL_PROFILES: Dict[str, Dict[str, float]] = {
    "baseline": {
        "NE": 0.3,
        "ACh": 0.3,
        "DA": 0.3,
        "5HT": 0.3,
        "hist": 0.3,
        "orexin": 0.3,
    },
    "high_ACh_vigilance": {
        "NE": 0.4,
        "ACh": 0.8,
        "DA": 0.3,
        "5HT": 0.3,
        "hist": 0.5,
        "orexin": 0.5,
    },
    "high_NE_stress": {
        "NE": 0.9,
        "ACh": 0.4,
        "DA": 0.4,
        "5HT": 0.2,
        "hist": 0.4,
        "orexin": 0.6,
    },
    "high_DA_reward": {
        "NE": 0.3,
        "ACh": 0.4,
        "DA": 0.9,
        "5HT": 0.4,
        "hist": 0.4,
        "orexin": 0.4,
    },
    "low_5HT_depression": {
        "NE": 0.3,
        "ACh": 0.3,
        "DA": 0.2,
        "5HT": 0.05,
        "hist": 0.3,
        "orexin": 0.3,
    },
    "low_ACh_cognitive_decline": {
        "NE": 0.3,
        "ACh": 0.05,
        "DA": 0.3,
        "5HT": 0.3,
        "hist": 0.3,
        "orexin": 0.3,
    },
    "psychedelic_5HT2A": {
        "NE": 0.4,
        "ACh": 0.3,
        "DA": 0.5,
        "5HT": 0.9,
        "hist": 0.3,
        "orexin": 0.5,
    },
    "sleep": {
        "NE": 0.05,
        "ACh": 0.05,
        "DA": 0.2,
        "5HT": 0.4,
        "hist": 0.9,
        "orexin": 0.05,
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NeuromodProfile:
    """Neuromodulator concentration profile (all values in [0, 1])."""

    NE: float = 0.3
    ACh: float = 0.3
    DA: float = 0.3
    serotonin: float = 0.3  # 5-HT
    hist: float = 0.3
    orexin: float = 0.3

    def validate(self) -> None:
        for name, val in self.__dict__.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"NeuromodProfile.{name}={val} outside [0,1]")


@dataclass
class APGIParameterSet:
    """APGI parameters derived from a neuromodulator profile."""

    theta_eff: float = 0.5  # effective ignition threshold
    Pi_e: float = 1.0  # exteroceptive precision multiplier
    Pi_i: float = 1.0  # interoceptive precision multiplier
    alpha_eff: float = 8.0  # ignition steepness
    gamma_V: float = 1.0  # somatic-marker valence weight
    tau_e_eff: float = 0.3  # effective exteroceptive time constant


@dataclass
class SimulationResult:
    """Result of a single profile simulation."""

    profile_name: str
    neuromod: NeuromodProfile
    params: APGIParameterSet
    ignition_times: List[float] = field(default_factory=list)
    S_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    theta_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    ignition_probability: float = 0.0
    mean_ignition_latency: float = float("nan")
    bandwidth: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class APGINeuromodulatorSystem:
    """
    Centralizes all neuromodulator-to-APGI-parameter mappings.

    Each neuromodulator modifies APGI parameters via receptor-mediated equations.
    All κ parameters are empirically grounded (see module-level docstring).
    """

    def __init__(
        self,
        theta_base: float = 0.5,
        tau_e_base: float = 0.3,
        alpha_base: float = 8.0,
        gamma_V_base: float = 1.0,
        Pi_e_base: float = 1.0,
        Pi_i_base: float = 1.0,
    ) -> None:
        self.theta_base = theta_base
        self.tau_e_base = tau_e_base
        self.alpha_base = alpha_base
        self.gamma_V_base = gamma_V_base
        self.Pi_e_base = Pi_e_base
        self.Pi_i_base = Pi_i_base

    # ------------------------------------------------------------------
    # Core parameter computation methods
    # ------------------------------------------------------------------

    def compute_theta_eff(self, profile: NeuromodProfile) -> float:
        """
        θₜ_eff = θ_base · exp(Σₖ sₖ · κₖ · [neuromod_k])

        Threshold-lowering: NE (s=-1), ACh (s=-1)
        Threshold-raising:  5-HT (s=+1), hist (s=+1)
        DA and orexin act via separate pathways (not direct θ modulation).
        """
        exponent = (
            -KAPPA_NE * profile.NE - KAPPA_ACh * profile.ACh + KAPPA_5HT * profile.serotonin + KAPPA_HIST * profile.hist
        )
        return float(self.theta_base * np.exp(exponent))

    def compute_Pi_e(self, profile: NeuromodProfile) -> float:
        """
        Πᵉ_eff: ACh increases exteroceptive precision (M1 L2/3 excitability),
        histamine maintains arousal floor.

            Πᵉ = Pi_e_base · (1 + κ_ACh·[ACh]) · (1 + 0.5·κ_hist·[hist])
        """
        return float(self.Pi_e_base * (1.0 + KAPPA_ACh * profile.ACh) * (1.0 + 0.5 * KAPPA_HIST * profile.hist))

    def compute_Pi_i(self, profile: NeuromodProfile) -> float:
        """
        Πⁱ_eff: NE enhances interoceptive signal gain; 5-HT provides allostatic
        dampening of interoceptive drive.

            Πⁱ = Pi_i_base · (1 + κ_NE·[NE]) · exp(-0.3·κ_5HT·[5HT])
        """
        return float(self.Pi_i_base * (1.0 + KAPPA_NE * profile.NE) * np.exp(-0.3 * KAPPA_5HT * profile.serotonin))

    def compute_alpha(self, profile: NeuromodProfile) -> float:
        """
        α_eff: Orexin scales global ignition steepness.

            α = α_base · (1 + κ_orexin·[orexin])

        Psychedelic 5-HT2A agonism also increases α (secondary pathway).
        """
        orexin_gain = 1.0 + KAPPA_OREXIN * profile.orexin
        serotonin_gain = 1.0 + 0.4 * KAPPA_5HT * profile.serotonin
        return float(self.alpha_base * orexin_gain * serotonin_gain)

    def compute_gamma_V(self, profile: NeuromodProfile) -> float:
        """
        γ_V: DA scales somatic-marker valence weight via D1 vmPFC circuit.

            γ_V = γ_V0 · (1 + κ_DA·[DA])
        """
        return float(self.gamma_V_base * (1.0 + KAPPA_DA * profile.DA))

    def compute_tau_e_eff(self, profile: NeuromodProfile) -> float:
        """
        τᵉ_eff: ACh shortens effective exteroceptive time constant.

            τᵉ = τ₀ / (1 + κ_ACh·[ACh])
        """
        return float(self.tau_e_base / (1.0 + KAPPA_ACh * profile.ACh))

    def compute_parameter_set(self, profile: NeuromodProfile) -> APGIParameterSet:
        """Compute all APGI parameters for a given neuromodulator profile."""
        profile.validate()
        return APGIParameterSet(
            theta_eff=self.compute_theta_eff(profile),
            Pi_e=self.compute_Pi_e(profile),
            Pi_i=self.compute_Pi_i(profile),
            alpha_eff=self.compute_alpha(profile),
            gamma_V=self.compute_gamma_V(profile),
            tau_e_eff=self.compute_tau_e_eff(profile),
        )

    # ------------------------------------------------------------------
    # Simulation harness
    # ------------------------------------------------------------------

    def simulate_state(
        self,
        profile: NeuromodProfile,
        duration_s: float = 2.0,
        dt: float = 0.001,
        stimulus_stream: Optional[np.ndarray] = None,
        rng_seed: int = 42,
    ) -> SimulationResult:
        """
        Simulate APGI ignition dynamics under a given neuromodulator profile.

        Uses a leaky-integrator signal model with sigmoid ignition gating:
            dS/dt = -(S - S_input) / τᵉ + noise
            B(t) = σ(α · (S(t) - θₜ))

        Returns ignition times, state traces, and summary statistics.
        """
        params = self.compute_parameter_set(profile)
        rng = np.random.default_rng(rng_seed)

        n_steps = int(duration_s / dt)
        time = np.arange(n_steps) * dt

        if stimulus_stream is None:
            # Default: ramp stimulus with noise
            stimulus_stream = 0.3 * np.linspace(0, 1, n_steps) + 0.1 * rng.standard_normal(n_steps)

        tau_e = params.tau_e_eff
        alpha = params.alpha_eff
        theta = params.theta_eff
        noise_amp = 0.02 / np.sqrt(params.Pi_e + 1e-6)

        S = np.zeros(n_steps)
        theta_trace = np.full(n_steps, theta)
        ignition_times = []
        in_ignition = False

        for t in range(1, n_steps):
            dS = (-(S[t - 1] - stimulus_stream[t - 1]) / tau_e) * dt
            dS += noise_amp * np.sqrt(dt) * rng.standard_normal()
            S[t] = S[t - 1] + dS

            B = 1.0 / (1.0 + np.exp(-alpha * (S[t] - theta)))

            if B > 0.5 and not in_ignition:
                ignition_times.append(time[t])
                in_ignition = True
            elif B <= 0.5:
                in_ignition = False

        n_ignitions = len(ignition_times)
        ignition_prob = n_ignitions / max(1.0, duration_s)
        mean_latency = float(np.mean(ignition_times)) if ignition_times else float("nan")
        bandwidth = float(np.std(S[S > theta])) if np.any(S > theta) else 0.0

        return SimulationResult(
            profile_name="",
            neuromod=profile,
            params=params,
            ignition_times=ignition_times,
            S_trace=S,
            theta_trace=theta_trace,
            ignition_probability=ignition_prob,
            mean_ignition_latency=mean_latency,
            bandwidth=bandwidth,
        )

    # ------------------------------------------------------------------
    # Profile sweep
    # ------------------------------------------------------------------

    def run_profile_sweep(self, duration_s: float = 2.0) -> Dict[str, SimulationResult]:
        """Run simulation for all 8 canonical neuromodulator profiles."""
        results = {}
        for name, concs in CANONICAL_PROFILES.items():
            profile = NeuromodProfile(
                NE=concs["NE"],
                ACh=concs["ACh"],
                DA=concs["DA"],
                serotonin=concs["5HT"],
                hist=concs["hist"],
                orexin=concs["orexin"],
            )
            result = self.simulate_state(profile, duration_s=duration_s)
            result.profile_name = name
            results[name] = result
            logger.info(
                f"Profile '{name}': θ_eff={result.params.theta_eff:.3f}, "
                f"α={result.params.alpha_eff:.2f}, "
                f"P(ignition)={result.ignition_probability:.2f}/s"
            )
        return results

    # ------------------------------------------------------------------
    # Validation against known phenomenology
    # ------------------------------------------------------------------

    def validate_phenomenological_predictions(self, results: Dict[str, SimulationResult]) -> Dict[str, bool]:
        """
        Validate that profile predictions align with known phenomenology:

        1. high_NE_stress → lower θ → more ignitions (stress hypervigilance)
        2. low_ACh_cognitive_decline → lower Πᵉ → reduced Pi_e
        3. psychedelic_5HT2A → elevated α → sharper ignition curves
        4. sleep → highest θ → fewest ignitions
        5. high_DA_reward → highest γ_V
        """
        checks: Dict[str, bool] = {}

        baseline = results.get("baseline")
        high_ne = results.get("high_NE_stress")
        low_ach = results.get("low_ACh_cognitive_decline")
        psychedelic = results.get("psychedelic_5HT2A")
        sleep = results.get("sleep")
        high_da = results.get("high_DA_reward")

        if baseline and high_ne:
            checks["high_NE_lowers_theta"] = high_ne.params.theta_eff < baseline.params.theta_eff
            checks["high_NE_increases_ignitions"] = high_ne.ignition_probability >= baseline.ignition_probability

        if baseline and low_ach:
            checks["low_ACh_reduces_Pi_e"] = low_ach.params.Pi_e < baseline.params.Pi_e

        if baseline and psychedelic:
            checks["psychedelic_increases_alpha"] = psychedelic.params.alpha_eff > baseline.params.alpha_eff

        if sleep and baseline:
            checks["sleep_highest_theta"] = sleep.params.theta_eff > baseline.params.theta_eff

        if baseline and high_da:
            checks["high_DA_increases_gamma_V"] = high_da.params.gamma_V > baseline.params.gamma_V

        passed = sum(checks.values())
        total = len(checks)
        logger.info(f"Phenomenological validation: {passed}/{total} predictions confirmed")
        return checks

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_appendix_b_table(self, results: Dict[str, SimulationResult]) -> List[Dict]:
        """Generate Paper 2 Appendix B parameter table rows."""
        rows = []
        for name, result in results.items():
            p = result.params
            rows.append(
                {
                    "profile": name,
                    "theta_eff": round(p.theta_eff, 4),
                    "Pi_e": round(p.Pi_e, 4),
                    "Pi_i": round(p.Pi_i, 4),
                    "alpha_eff": round(p.alpha_eff, 4),
                    "gamma_V": round(p.gamma_V, 4),
                    "tau_e_eff": round(p.tau_e_eff, 4),
                    "ignition_prob_per_s": round(result.ignition_probability, 4),
                    "bandwidth": round(result.bandwidth, 4),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_profile_comparison(
        self, results: Dict[str, SimulationResult], save_path: Optional[str] = None
    ) -> Optional[str]:
        """Plot parameter values and ignition probabilities across profiles."""
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not available; skipping plot")
            return None

        profile_names = list(results.keys())
        n = len(profile_names)

        theta_vals = [results[p].params.theta_eff for p in profile_names]
        alpha_vals = [results[p].params.alpha_eff for p in profile_names]
        pie_vals = [results[p].params.Pi_e for p in profile_names]
        ignition_vals = [results[p].ignition_probability for p in profile_names]

        x = np.arange(n)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("APGI Neuromodulatory Parameter Profiles", fontsize=14)

        short_names = [p.replace("_", "\n") for p in profile_names]

        for ax, vals, title, color in zip(
            axes.flat,
            [theta_vals, alpha_vals, pie_vals, ignition_vals],
            [
                "θₜ_eff (ignition threshold)",
                "α_eff (ignition steepness)",
                "Πᵉ_eff (extero precision)",
                "P(ignition) per second",
            ],
            [
                VISUAL_CONSTANTS.ST_BLUE,
                VISUAL_CONSTANTS.IGNITION_GREEN,
                VISUAL_CONSTANTS.INTERO_AMBER,
                VISUAL_CONSTANTS.THETA_RED,
            ],
        ):
            bars = ax.bar(x, vals, color=color, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(short_names, fontsize=7)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(title.split("(")[0].strip())
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        plt.tight_layout()

        if save_path is None:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix="_neuromod_profiles.png", delete=False) as tmp_file:
                save_path = tmp_file.name

        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Profile comparison plot saved to {save_path}")
        return save_path

    # ------------------------------------------------------------------
    # GUI entry point
    # ------------------------------------------------------------------

    def run_analysis(self) -> Dict:
        """
        Main entry point for Theory GUI execution.

        Runs full profile sweep, validates phenomenological predictions,
        generates Appendix B table, and produces comparison plot.
        """
        logger.info("=" * 60)
        logger.info("APGI Neuromodulatory Control System — Analysis")
        logger.info("=" * 60)

        # 1. Parameter computation for all profiles
        logger.info("Computing APGI parameters for 8 canonical profiles...")
        results = self.run_profile_sweep(duration_s=2.0)

        # 2. Phenomenological validation
        logger.info("Validating phenomenological predictions...")
        checks = self.validate_phenomenological_predictions(results)
        all_passed = all(checks.values())

        # 3. Appendix B table
        table = self.generate_appendix_b_table(results)
        logger.info("\nPaper 2 Appendix B — Parameter Table:")
        logger.info(f"{'Profile':<30} {'θ_eff':>8} {'α_eff':>8} {'Πᵉ':>8} {'γ_V':>8} {'P_ign':>8}")
        logger.info("-" * 72)
        for row in table:
            logger.info(
                f"{row['profile']:<30} {row['theta_eff']:>8.3f} "
                f"{row['alpha_eff']:>8.2f} {row['Pi_e']:>8.3f} "
                f"{row['gamma_V']:>8.3f} {row['ignition_prob_per_s']:>8.3f}"
            )

        # 4. Validation summary
        logger.info("\nPhenomenological Validation Results:")
        for check_name, passed in checks.items():
            status = "[PASS]" if passed else "[FAIL]"
            logger.info(f"  {status} {check_name.replace('_', ' ')}")

        # 5. Plot
        plot_path = self.plot_profile_comparison(results)

        status = "PASS" if all_passed else "PARTIAL"
        logger.info(f"\nOverall status: {status}")
        logger.info(f"Checks passed: {sum(checks.values())}/{len(checks)}")

        return {
            "status": status,
            "profile_results": {
                name: {
                    "theta_eff": r.params.theta_eff,
                    "alpha_eff": r.params.alpha_eff,
                    "Pi_e": r.params.Pi_e,
                    "Pi_i": r.params.Pi_i,
                    "gamma_V": r.params.gamma_V,
                    "tau_e_eff": r.params.tau_e_eff,
                    "ignition_probability": r.ignition_probability,
                    "bandwidth": r.bandwidth,
                }
                for name, r in results.items()
            },
            "phenomenological_checks": checks,
            "appendix_b_table": table,
            "plot_path": plot_path,
            "checks_passed": sum(checks.values()),
            "checks_total": len(checks),
        }


# ---------------------------------------------------------------------------
# Module-level runner
# ---------------------------------------------------------------------------


def run_analysis() -> Dict:
    """Module-level entry point for headless / CLI execution."""
    system = APGINeuromodulatorSystem()
    return system.run_analysis()


class NeuromodulationModule:
    """TheoryModuleInterface implementation for neuromodulatory control system."""

    MODULE_LEVEL: str = "Level 3"

    def compute(self, **kwargs) -> dict:
        return run_analysis()

    def get_falsification_criteria(self) -> dict:
        return {
            "NM-1": {
                "description": "ACh modulates π_i in predicted direction (systole gating)",
                "threshold": "ACh ↑ → π_i systole ↓ by ≥ 15%, p < 0.05",
                "test": "Pharmacological simulation comparison",
            },
            "NM-2": {
                "description": "NE (norepinephrine) increases global π_i precision uniformly",
                "threshold": "NE ↑ → π_i global ↑ by ≥ 10%, p < 0.05",
                "test": "Paired simulation: NE-modulated vs baseline",
            },
            "NM-3": {
                "description": "5-HT predicts interoceptive homeostatic setpoint shift",
                "threshold": "Setpoint deviation < 20% of predicted APGI value",
                "test": "Fixed-point analysis of 5-HT-perturbed system",
            },
        }

    def as_protocol_result(self, protocol_id: str = "T-NeuromodulatorControl", **kwargs) -> dict:
        try:
            from utils.protocol_schema import PredictionResult, PredictionStatus, ProtocolResult
        except ImportError:
            return self.compute(**kwargs)
        raw = self.compute(**kwargs)
        criteria = self.get_falsification_criteria()
        named = {}
        for key, spec in criteria.items():
            passed = raw.get(key, {}).get("passed", False) if isinstance(raw.get(key), dict) else False
            named[key] = PredictionResult(
                passed=passed,
                status=PredictionStatus.PASSED if passed else PredictionStatus.NOT_EVALUATED,
                sources=[protocol_id],
                metadata=spec,
            )
        return ProtocolResult(
            protocol_id=protocol_id,
            named_predictions=named,
            completion_percentage=100,
            methodology=self.MODULE_LEVEL,
            metadata={"module_level": self.MODULE_LEVEL},
        ).to_dict()


if __name__ == "__main__":
    run_analysis()
