"""

=============================================================================
APGI THERMODYNAMIC PROGRAM AGGREGATOR
=============================================================================

Derives and audits the κ ≈ 100 ATP/bit claim that is central to APGI Level 1
(thermodynamic self-application).  The script makes the Double Bridge
(Level 3 → Level 2 → Level 1) computable rather than rhetorical.

CRITICAL DISTINCTION (enforced throughout):
  (a) Thermodynamic capacity  — Landauer minimum energy to erase one bit
                                 E_min = k_B · T · ln 2  [J/bit]
  (b) Shannon channel capacity — maximum bits per second transmissible
                                 C = B · log2(1 + SNR)   [bit/s]
  (c) Neural metabolic cost   — actual ATP consumed per ignition event
                                 κ = metabolic_ATP / I_bits  [ATP/bit]

These are three *different* quantities.  Conflating them is the primary
theoretical error the aggregator is designed to prevent.

Modules
-------
  1. LandauerMinimum          — thermodynamic floor (quantity a)
  2. NeuralMetabolicCost      — empirical firing cost (quantity c numerator)
  3. DoubleBridgeCalculator   — derives κ and efficiency ratio (a → c bridge)
  4. CrossSpeciesScaling      — validates APGI metabolic scaling exponent

Integration layer runs all four modules, generates a unified text report,
and produces a log-log figure of ATP cost per bit vs. brain mass with the
APGI prediction band overlaid on empirical species data.

LEVEL DESIGNATION: All outputs are Level 1 (thermodynamic).
Bridge to Level 2 requires APGI_Information_Theoretic_Bandwidth.
Bridge to Level 3 requires standard algorithmic implementations.
This script does NOT claim algorithmic or information-theoretic implications
without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If the κ ≈ 100 ATP/bit claim is NOT supported by empirical data (κ deviation
> 50% from 100 ATP/bit prediction, efficiency ratio < 0.1 of Landauer limit,
or cross-species scaling exponent deviation > 0.2 from b = 0.65-0.75), then
the APGI thermodynamic self-application claim is falsified. This would indicate
that conscious processing does not have the distinctive thermodynamic signature
predicted by the framework.

References
----------
  Attwell & Laughlin (2001) J Cereb Blood Flow Metab 21:1133–1145
  Karbowski (2024) PLOS Comput Biol (metabolic scaling)
  Street et al. (2020) J Neurosci (comparative neural energetics)
  Landauer (1961) IBM J Res Dev 5:183–191
  Shannon (1948) Bell Syst Tech J 27:379–423

=============================================================================
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats

try:
    import matplotlib

    matplotlib.use("Agg")
    # import matplotlib.patches as mpatches  # Available if needed for future patches
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

K_B: float = 1.380649e-23  # Boltzmann constant  [J/K]
T_BODY_K: float = 310.15  # 37 °C in Kelvin
ATP_HYDROLYSIS_KJ_MOL: float = 30.6  # Free energy of ATP hydrolysis [kJ/mol]
AVOGADRO: float = 6.02214076e23  # [mol⁻¹]
ATP_ENERGY_J: float = (ATP_HYDROLYSIS_KJ_MOL * 1e3) / AVOGADRO  # J per ATP molecule

# Attwell & Laughlin (2001) — energy cost of one cortical action potential
COST_PER_SPIKE_J: float = 4.7e-10  # [J/spike]


# ===========================================================================
# MODULE 1 — Landauer Minimum
# ===========================================================================


@dataclass
class LandauerResult:
    """Outputs of Module 1 (thermodynamic capacity — quantity a)."""

    temperature_K: float
    n_bits: float
    E_min_J: float  # Minimum energy to erase n_bits  [J]
    E_min_per_bit_J: float  # E_min / n_bits                  [J/bit]
    E_min_ATP: float  # Total erasure cost               [ATP molecules]
    E_min_per_bit_ATP: float  # Per-bit erasure cost             [ATP/bit]

    def summary(self) -> str:
        return (
            f"Landauer minimum @ T={self.temperature_K:.2f} K:\n"
            f"  E_min / bit = {self.E_min_per_bit_J:.4e} J/bit\n"
            f"  E_min / bit = {self.E_min_per_bit_ATP:.6f} ATP/bit\n"
            f"  (for {self.n_bits:.1f} bits total: "
            f"{self.E_min_J:.4e} J = {self.E_min_ATP:.4f} ATP)"
        )


class LandauerMinimum:
    """
    Module 1 — Thermodynamic floor computation.

    Computes the Landauer minimum energy required to erase n_bits of
    information at temperature T.  This is quantity (a): thermodynamic
    capacity, NOT the actual neural cost.

    E_min = n_bits · k_B · T · ln(2)
    """

    def __init__(self, temperature_K: float = T_BODY_K) -> None:
        if temperature_K <= 0:
            raise ValueError(f"Temperature must be positive; got {temperature_K}")
        self.temperature_K = temperature_K

    def compute(self, n_bits: float) -> LandauerResult:
        """
        Args:
            n_bits: Number of bits erased per ignition event.

        Returns:
            LandauerResult with energy in Joules and ATP equivalents.
        """
        if n_bits <= 0:
            raise ValueError(f"n_bits must be positive; got {n_bits}")

        E_min_J = n_bits * K_B * self.temperature_K * np.log(2)
        E_min_per_bit_J = K_B * self.temperature_K * np.log(2)
        E_min_ATP = E_min_J / ATP_ENERGY_J
        E_min_per_bit_ATP = E_min_per_bit_J / ATP_ENERGY_J

        return LandauerResult(
            temperature_K=self.temperature_K,
            n_bits=n_bits,
            E_min_J=E_min_J,
            E_min_per_bit_J=E_min_per_bit_J,
            E_min_ATP=E_min_ATP,
            E_min_per_bit_ATP=E_min_per_bit_ATP,
        )


# ===========================================================================
# MODULE 2 — Neural Metabolic Cost Estimator
# ===========================================================================


@dataclass
class MetabolicCostResult:
    """Outputs of Module 2 (actual neural metabolic cost — quantity c numerator)."""

    firing_rate_Hz: float
    n_neurons: int
    duration_ms: float
    cost_per_spike_J: float
    metabolic_cost_J: float
    metabolic_cost_ATP: float

    def summary(self) -> str:
        return (
            f"Neural metabolic cost:\n"
            f"  Neurons: {self.n_neurons}, rate: {self.firing_rate_Hz} Hz, "
            f"window: {self.duration_ms} ms\n"
            f"  Cost/spike: {self.cost_per_spike_J:.3e} J  "
            f"(Attwell & Laughlin 2001)\n"
            f"  Total: {self.metabolic_cost_J:.4e} J = "
            f"{self.metabolic_cost_ATP:.1f} ATP molecules"
        )


class NeuralMetabolicCost:
    """
    Module 2 — Empirical neural firing cost.

    Estimates the metabolic cost of a cortical ignition event from
    biophysical parameters.  This is the numerator of κ (quantity c).

    metabolic_cost_J = firing_rate · n_neurons · (duration_ms / 1000) · cost_per_spike_J

    Default cost_per_spike_J = 4.7 × 10⁻¹⁰ J from Attwell & Laughlin (2001).
    """

    def __init__(
        self,
        cost_per_spike_J: float = COST_PER_SPIKE_J,
    ) -> None:
        if cost_per_spike_J <= 0:
            raise ValueError(f"cost_per_spike_J must be positive; got {cost_per_spike_J}")
        self.cost_per_spike_J = cost_per_spike_J

    def compute(
        self,
        firing_rate_Hz: float,
        n_neurons: int,
        duration_ms: float,
    ) -> MetabolicCostResult:
        """
        Args:
            firing_rate_Hz: Mean firing rate of ignition-participating neurons [Hz].
            n_neurons: Number of neurons participating in the ignition event.
            duration_ms: Duration of the ignition window [ms].

        Returns:
            MetabolicCostResult with cost in Joules and ATP equivalents.
        """
        if firing_rate_Hz <= 0:
            raise ValueError(f"firing_rate_Hz must be positive; got {firing_rate_Hz}")
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be positive; got {n_neurons}")
        if duration_ms <= 0:
            raise ValueError(f"duration_ms must be positive; got {duration_ms}")

        duration_s = duration_ms / 1000.0
        metabolic_cost_J = firing_rate_Hz * n_neurons * duration_s * self.cost_per_spike_J
        metabolic_cost_ATP = metabolic_cost_J / ATP_ENERGY_J

        return MetabolicCostResult(
            firing_rate_Hz=firing_rate_Hz,
            n_neurons=n_neurons,
            duration_ms=duration_ms,
            cost_per_spike_J=self.cost_per_spike_J,
            metabolic_cost_J=metabolic_cost_J,
            metabolic_cost_ATP=metabolic_cost_ATP,
        )


# ===========================================================================
# MODULE 3 — Double Bridge Calculator
# ===========================================================================


@dataclass
class DoubleBridgeResult:
    """Outputs of Module 3 — the κ derivation (quantity c / bit)."""

    I_bits: float  # Information gained at ignition [bits]
    metabolic_cost_ATP: float  # From Module 2 [ATP]
    kappa: float  # ATP/bit (central APGI claim)
    landauer_per_bit_ATP: float  # Landauer floor for same I_bits [ATP/bit]
    efficiency_ratio: float  # kappa / landauer_per_bit_ATP  (biological overhead)
    kappa_in_range: bool  # True if kappa ∈ [50, 200] ATP/bit
    falsification_note: str  # Human-readable falsification criterion

    def summary(self) -> str:
        range_str = "IN RANGE" if self.kappa_in_range else "OUT OF RANGE"
        lines = [
            "Double Bridge (Level 3 → Level 2 → Level 1):",
            f"  I_bits = {self.I_bits:.2f} bits",
            f"  metabolic_cost = {self.metabolic_cost_ATP:.3e} ATP molecules",
            f"  κ = {self.kappa:.3e} ATP/bit  [{range_str}]",
            f"  Landauer floor = {self.landauer_per_bit_ATP:.6f} ATP/bit",
            f"  Neural overhead factor = {self.efficiency_ratio:.2e}×",
            "  (literature range: ~10⁴–10⁶× Landauer minimum)",
        ]
        if not self.kappa_in_range:
            expected_atp = 100.0 * self.I_bits
            gap_orders = np.log10(self.kappa / 100.0)
            lines.append(
                f"  DISCREPANCY NOTE: κ is {gap_orders:.1f} orders of magnitude above "
                f"the claimed ~100 ATP/bit. To reach κ=100 would require "
                f"{expected_atp:.0f} total ATP for {self.I_bits:.1f} bits — "
                f"inconsistent with Attwell & Laughlin (2001) spike costs. "
                f"The κ≈100 claim requires either a sub-spike normalization, "
                f"a different ATP unit, or derivation from the Landauer floor "
                f"(efficiency_ratio check above)."
            )
        lines.append(f"  Falsification: {self.falsification_note}")
        return "\n".join(lines)


class DoubleBridgeCalculator:
    """
    Module 3 — Derives κ and connects thermodynamic floor to neural cost.

    This is the Double Bridge:
      Level 3 (PCI / mutual information)  →  I_bits
      Level 2 (metabolic firing cost)     →  metabolic_cost_ATP
      Level 1 (Landauer floor)            →  efficiency_ratio

    κ = metabolic_cost_ATP / I_bits

    APGI prediction: κ ∈ [50, 200] ATP/bit for cortical ignition events.

    Falsification condition:
      κ_conscious NOT significantly greater than κ_unconscious
      (one-tailed t-test, p < 0.05, d > 0.5) would falsify the claim
      that ignition efficiency scales with conscious broadcast.
    """

    def __init__(self, landauer: LandauerMinimum) -> None:
        self.landauer = landauer

    def compute(
        self,
        I_bits: float,
        metabolic_result: MetabolicCostResult,
    ) -> DoubleBridgeResult:
        """
        Args:
            I_bits: Information gained at ignition from Level 2 mutual
                    information estimate [bits].
            metabolic_result: Output of NeuralMetabolicCost.compute().

        Returns:
            DoubleBridgeResult including κ and efficiency ratio.
        """
        if I_bits <= 0:
            raise ValueError(f"I_bits must be positive; got {I_bits}")

        kappa = metabolic_result.metabolic_cost_ATP / I_bits
        landauer_result = self.landauer.compute(n_bits=I_bits)
        efficiency_ratio = kappa / landauer_result.E_min_per_bit_ATP

        kappa_in_range = 50.0 <= kappa <= 200.0

        falsification_note = (
            "κ_conscious > κ_unconscious required (one-tailed t-test p<0.05, d>0.5). "
            "If κ does not differ significantly between conscious and unconscious "
            "ignition, the Level 1 thermodynamic claim is falsified."
        )

        return DoubleBridgeResult(
            I_bits=I_bits,
            metabolic_cost_ATP=metabolic_result.metabolic_cost_ATP,
            kappa=kappa,
            landauer_per_bit_ATP=landauer_result.E_min_per_bit_ATP,
            efficiency_ratio=efficiency_ratio,
            kappa_in_range=kappa_in_range,
            falsification_note=falsification_note,
        )

    def sensitivity_analysis(
        self,
        I_bits_range: Tuple[float, float],
        metabolic_ATP_range: Tuple[float, float],
        n_samples: int = 1000,
        rng_seed: int = 42,
    ) -> Dict[str, object]:
        """
        Monte Carlo sensitivity analysis over I_bits and metabolic_ATP.

        Samples uniformly from the provided ranges and reports the
        distribution of κ values to characterise parameter uncertainty.

        Args:
            I_bits_range: (min, max) for I_bits.
            metabolic_ATP_range: (min, max) for metabolic_cost_ATP.
            n_samples: Number of Monte Carlo draws.
            rng_seed: Random seed for reproducibility.

        Returns:
            Dict with kappa_samples, kappa_mean, kappa_std, fraction_in_range.
        """
        rng = np.random.default_rng(rng_seed)
        i_samples = rng.uniform(*I_bits_range, size=n_samples)
        m_samples = rng.uniform(*metabolic_ATP_range, size=n_samples)
        kappa_samples = m_samples / i_samples
        in_range = np.sum((kappa_samples >= 50) & (kappa_samples <= 200)) / n_samples

        return {
            "kappa_samples": kappa_samples,
            "kappa_mean": float(np.mean(kappa_samples)),
            "kappa_std": float(np.std(kappa_samples)),
            "kappa_median": float(np.median(kappa_samples)),
            "kappa_p5": float(np.percentile(kappa_samples, 5)),
            "kappa_p95": float(np.percentile(kappa_samples, 95)),
            "fraction_in_range": float(in_range),
        }


# ===========================================================================
# MODULE 4 — Cross-Species Scaling Validation
# ===========================================================================


@dataclass
class ScalingResult:
    """Outputs of Module 4 — cross-species metabolic scaling."""

    brain_masses_g: np.ndarray
    CMR_glucose: np.ndarray  # μmol / 100 g / min
    atp_cost_per_bit: np.ndarray  # Derived ATP/bit per species
    log_mass: np.ndarray
    log_atp: np.ndarray
    scaling_exponent: float  # slope of log-log regression
    intercept: float
    r_squared: float
    p_value: float
    apgi_predicted_exponent: float  # APGI prediction: ≈ 0.75
    exponent_matches_apgi: bool
    species_labels: List[str]

    def summary(self) -> str:
        match_str = "MATCHES" if self.exponent_matches_apgi else "DOES NOT MATCH"
        return (
            f"Cross-species metabolic scaling:\n"
            f"  Scaling exponent (empirical)  = {self.scaling_exponent:.3f}\n"
            f"  APGI predicted exponent       = {self.apgi_predicted_exponent:.2f}\n"
            f"  R² = {self.r_squared:.3f}  (p = {self.p_value:.4f})\n"
            f"  Exponent {match_str} APGI prediction "
            f"(tolerance ±0.15)\n"
            f"  Species: {', '.join(self.species_labels)}"
        )


class CrossSpeciesScalingValidator:
    """
    Module 4 — Validates APGI metabolic scaling predictions.

    Tests the prediction that ATP cost per bit scales with brain mass
    with exponent ≈ 0.75 (consistent with metabolic scaling literature:
    Karbowski 2024, Street et al. 2020).

    ATP/bit per species is derived from:
      - CMR_glucose → total metabolic power
      - Estimated information throughput ∝ neuron_count^0.5 (conservative)
    """

    # Empirical species data (Karbowski 2024 / Street 2020 combined)
    # brain_mass [g], CMR_glucose [μmol/100g/min], approx neuron count [millions]
    DEFAULT_SPECIES_DATA: List[Dict] = [
        {"name": "Mouse", "brain_mass_g": 0.42, "CMR_glucose": 65.0, "neurons_M": 71},
        {"name": "Rat", "brain_mass_g": 1.8, "CMR_glucose": 60.0, "neurons_M": 200},
        {
            "name": "Marmoset",
            "brain_mass_g": 7.8,
            "CMR_glucose": 52.0,
            "neurons_M": 635,
        },
        {
            "name": "Macaque",
            "brain_mass_g": 95.0,
            "CMR_glucose": 35.0,
            "neurons_M": 6376,
        },
        {
            "name": "Chimp",
            "brain_mass_g": 420.0,
            "CMR_glucose": 25.0,
            "neurons_M": 28000,
        },
        {
            "name": "Human",
            "brain_mass_g": 1400.0,
            "CMR_glucose": 22.0,
            "neurons_M": 86000,
        },
    ]

    APGI_PREDICTED_EXPONENT: float = 0.75
    EXPONENT_TOLERANCE: float = 0.15

    def __init__(
        self,
        species_data: Optional[List[Dict]] = None,
        apgi_predicted_exponent: float = APGI_PREDICTED_EXPONENT,
    ) -> None:
        self.species_data = species_data or self.DEFAULT_SPECIES_DATA
        self.apgi_predicted_exponent = apgi_predicted_exponent

    def _cmr_to_atp_per_bit(self, cmr_glucose: float, neurons_M: float, brain_mass_g: float) -> float:
        """
        Derive ATP/bit from CMR and neuron count.

        Total brain ATP rate (molecules/s) is computed from CMR × brain mass.
        Information throughput is approximated as ∝ sqrt(neuron count).
        Only the ratio is physically meaningful; the exponent of the resulting
        log-log regression is the quantity compared to the APGI prediction.

        Note: this uses total ATP (not per-gram) to give a dimensionally
        consistent ATP-per-bit estimate.  Mixing per-gram ATP with total-
        neuron information would produce the wrong scaling direction.
        """
        atp_per_glucose = 30.0
        # Total CMR: μmol/100g/min × g → μmol/min (whole brain)
        total_cmr_umol_min = cmr_glucose * brain_mass_g / 100.0
        # Convert to mol/s
        total_cmr_mol_s = (total_cmr_umol_min * 1e-6) / 60.0
        # Total ATP molecules per second (whole brain)
        total_atp_molecules_s = total_cmr_mol_s * atp_per_glucose * AVOGADRO
        # Approximate information throughput ∝ sqrt(total neurons)
        info_throughput = np.sqrt(neurons_M * 1e6)
        return total_atp_molecules_s / info_throughput

    def compute(
        self,
        brain_masses_g: Optional[np.ndarray] = None,
        CMR_glucose: Optional[np.ndarray] = None,
    ) -> ScalingResult:
        """
        Args:
            brain_masses_g: Optional override array of brain masses [g].
            CMR_glucose: Optional override array of CMR values [μmol/100g/min].
                         Must match brain_masses_g length if provided.

        Returns:
            ScalingResult with scaling exponent, R², and APGI comparison.
        """
        if brain_masses_g is not None and CMR_glucose is not None:
            if len(brain_masses_g) != len(CMR_glucose):
                raise ValueError("brain_masses_g and CMR_glucose must have the same length")
            masses = np.asarray(brain_masses_g, dtype=float)
            cmr = np.asarray(CMR_glucose, dtype=float)
            neurons_M = np.array([1.0] * len(masses))
            labels = [f"Species_{i}" for i in range(len(masses))]
        else:
            masses = np.array([d["brain_mass_g"] for d in self.species_data])
            cmr = np.array([d["CMR_glucose"] for d in self.species_data])
            neurons_M = np.array([d["neurons_M"] for d in self.species_data])
            labels = [d["name"] for d in self.species_data]

        atp_per_bit = np.array([self._cmr_to_atp_per_bit(cmr[i], neurons_M[i], masses[i]) for i in range(len(masses))])

        log_mass = np.log10(masses)
        log_atp = np.log10(atp_per_bit)

        slope, intercept, r_value, p_value, _ = stats.linregress(log_mass, log_atp)
        r_squared = r_value**2

        exponent_matches = abs(slope - self.apgi_predicted_exponent) <= self.EXPONENT_TOLERANCE

        return ScalingResult(
            brain_masses_g=masses,
            CMR_glucose=cmr,
            atp_cost_per_bit=atp_per_bit,
            log_mass=log_mass,
            log_atp=log_atp,
            scaling_exponent=float(slope),
            intercept=float(intercept),
            r_squared=float(r_squared),
            p_value=float(p_value),
            apgi_predicted_exponent=self.apgi_predicted_exponent,
            exponent_matches_apgi=bool(exponent_matches),
            species_labels=labels,
        )


# ===========================================================================
# INTEGRATION LAYER
# ===========================================================================


@dataclass
class AggregatorConfig:
    """Configuration for the full aggregator pipeline."""

    # Module 1 — Landauer
    temperature_K: float = T_BODY_K

    # Module 2 — Neural metabolic cost
    # Defaults represent one neuron firing once (minimal ignition unit).
    # Scale n_neurons and duration_ms upward for whole-ensemble estimates.
    # NOTE: With Attwell & Laughlin (2001) spike cost (~9.3e9 ATP/spike),
    # any physiologically realistic ignition event produces κ >> 100 ATP/bit.
    # The κ ≈ 100 claim is instead supported by the Landauer-floor argument
    # (Module 3 efficiency_ratio check).  Module 2 makes this gap explicit.
    firing_rate_Hz: float = 40.0  # Gamma-band rate [Hz]
    n_neurons: int = 50_000  # Cortical ignition ensemble size
    duration_ms: float = 100.0  # ~100 ms ignition window
    cost_per_spike_J: float = COST_PER_SPIKE_J

    # Module 3 — Double Bridge
    I_bits: float = 4.0  # Bits gained at ignition (from Level 2 MI)
    # Sensitivity analysis ranges
    I_bits_min: float = 1.0
    I_bits_max: float = 10.0
    metabolic_ATP_min_scale: float = 0.5  # fraction of nominal metabolic cost
    metabolic_ATP_max_scale: float = 2.0

    # Module 4 — Cross-species scaling
    brain_mass_kg: float = 1.4  # Human brain mass (~1400 g)
    apgi_predicted_exponent: float = 0.75

    # Output
    output_figure_path: str = "apgi_thermodynamic_aggregator_figure.png"
    run_sensitivity: bool = True
    sensitivity_n_samples: int = 1000


@dataclass
class AggregatorReport:
    """Unified output from the full aggregator pipeline."""

    landauer: LandauerResult
    metabolic: MetabolicCostResult
    double_bridge: DoubleBridgeResult
    scaling: ScalingResult
    sensitivity: Optional[Dict] = None
    figure_path: Optional[str] = None

    def print_report(self) -> None:
        separator = "=" * 70

        print(separator)
        print("APGI THERMODYNAMIC PROGRAM AGGREGATOR — UNIFIED REPORT")
        print(separator)

        print("\n[QUANTITY DISTINCTION]\n")
        print(
            "  (a) Thermodynamic capacity  — Landauer minimum (Module 1)\n"
            "  (b) Shannon channel capacity — max bits/s (NOT computed here;\n"
            "      see APGI_Entropy_Implementation.py for Shannon entropy)\n"
            "  (c) Neural metabolic cost   — actual ATP / ignition (Module 2+3)"
        )

        print(f"\n{'─' * 70}")
        print("MODULE 1 — LANDAUER MINIMUM (thermodynamic floor)")
        print(f"{'─' * 70}")
        print(self.landauer.summary())

        print(f"\n{'─' * 70}")
        print("MODULE 2 — NEURAL METABOLIC COST (empirical firing cost)")
        print(f"{'─' * 70}")
        print(self.metabolic.summary())

        print(f"\n{'─' * 70}")
        print("MODULE 3 — DOUBLE BRIDGE κ DERIVATION")
        print(f"{'─' * 70}")
        print(self.double_bridge.summary())

        if self.sensitivity is not None:
            s = self.sensitivity
            print(f"\n  Sensitivity analysis (Monte Carlo, n={s.get('n', '?')}):")
            print(f"    κ mean ± std  = {s['kappa_mean']:.1f} ± {s['kappa_std']:.1f} ATP/bit")
            print(f"    κ median      = {s['kappa_median']:.1f} ATP/bit")
            print(f"    κ [P5, P95]   = [{s['kappa_p5']:.1f}, {s['kappa_p95']:.1f}]")
            print(f"    Fraction in [50,200] = {s['fraction_in_range'] * 100:.1f}%")

        print(f"\n{'─' * 70}")
        print("MODULE 4 — CROSS-SPECIES SCALING VALIDATION")
        print(f"{'─' * 70}")
        print(self.scaling.summary())

        print(f"\n{'─' * 70}")
        print("LEVEL 1 AUDITABILITY CHECK")
        print(f"{'─' * 70}")
        kappa_ok = self.double_bridge.kappa_in_range
        exponent_ok = self.scaling.exponent_matches_apgi
        print(f"  κ ≈ 100 ATP/bit in [50, 200] range : {'PASS' if kappa_ok else 'FAIL'}")
        print(f"  Scaling exponent ≈ 0.75            : {'PASS' if exponent_ok else 'FAIL'}")
        print(f"  Neural overhead factor             : " f"{self.double_bridge.efficiency_ratio:.2e}× Landauer minimum")
        print("  (expected 10⁴–10⁶×; reflects noise, redundancy, metabolic overhead)")

        if self.figure_path:
            print(f"\n  Figure saved: {self.figure_path}")

        print(f"\n{separator}\n")


class ThermodynamicProgramAggregator:
    """
    Integration layer — runs all four modules sequentially and produces
    a unified report plus a publication-ready log-log figure.
    """

    def __init__(self, config: Optional[AggregatorConfig] = None) -> None:
        self.config = config or AggregatorConfig()
        cfg = self.config

        self.landauer_module = LandauerMinimum(temperature_K=cfg.temperature_K)
        self.metabolic_module = NeuralMetabolicCost(cost_per_spike_J=cfg.cost_per_spike_J)
        self.double_bridge_module = DoubleBridgeCalculator(landauer=self.landauer_module)
        self.scaling_module = CrossSpeciesScalingValidator(apgi_predicted_exponent=cfg.apgi_predicted_exponent)

    def run(self) -> AggregatorReport:
        """Execute the full pipeline and return an AggregatorReport."""
        cfg = self.config
        logger.info("Running APGI Thermodynamic Program Aggregator")

        # Module 1
        logger.info("Module 1: Landauer minimum")
        self.landauer_module.compute(n_bits=cfg.I_bits)

        # Module 4
        logger.info("Module 4: Scaling analysis")
        scaling_result = self.scaling_module.compute(
            brain_masses_g=np.array([cfg.brain_mass_kg * 1000]),  # Convert kg to g
            CMR_glucose=None,
        )

        # Module 2
        logger.info("Module 2: Neural metabolic cost")
        metabolic_result = self.metabolic_module.compute(
            firing_rate_Hz=cfg.firing_rate_Hz,
            n_neurons=cfg.n_neurons,
            duration_ms=cfg.duration_ms,
        )

        # Module 3
        logger.info("Module 3: Double bridge metabolism")
        double_bridge_result = self.double_bridge_module.compute(
            I_bits=cfg.I_bits,
            metabolic_result=metabolic_result,
        )

        # Sensitivity analysis
        sensitivity: Optional[Dict] = None
        if cfg.run_sensitivity:
            logger.info("Module 3 (sensitivity): Monte Carlo analysis")
            atp_nominal = metabolic_result.metabolic_cost_ATP
            sensitivity = self.double_bridge_module.sensitivity_analysis(
                I_bits_range=(cfg.I_bits_min, cfg.I_bits_max),
                metabolic_ATP_range=(
                    atp_nominal * cfg.metabolic_ATP_min_scale,
                    atp_nominal * cfg.metabolic_ATP_max_scale,
                ),
                n_samples=cfg.sensitivity_n_samples,
            )
            sensitivity["n"] = cfg.sensitivity_n_samples

        # Module 4
        logger.info("Module 4: Cross-species scaling")
        # scaling_result = self.scaling_module.compute()

        # Figure
        figure_path: Optional[str] = None
        if HAS_MATPLOTLIB:
            logger.info("Generating figure")
            figure_path = self._generate_figure(
                scaling_result=scaling_result,
                double_bridge_result=double_bridge_result,
                sensitivity=sensitivity,
                output_path=cfg.output_figure_path,
            )
        else:
            logger.warning("matplotlib not available; figure skipped")

        # Get landauer result
        landauer_result = self.landauer_module.compute(n_bits=cfg.I_bits)

        report = AggregatorReport(
            landauer=landauer_result,
            metabolic=metabolic_result,
            double_bridge=double_bridge_result,
            scaling=scaling_result,
            sensitivity=sensitivity,
            figure_path=figure_path,
        )

        report.print_report()
        return report

    def _generate_figure(
        self,
        scaling_result: ScalingResult,
        double_bridge_result: DoubleBridgeResult,
        sensitivity: Optional[Dict],
        output_path: str,
    ) -> str:
        """
        Generates a two-panel figure:
          Left  — log-log plot of ATP cost per bit vs. brain mass with
                  APGI prediction band overlaid on empirical species data.
          Right — κ distribution from sensitivity analysis (if available).
        """
        n_panels = 2 if sensitivity is not None else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        # ── Panel 1: cross-species scaling ──────────────────────────────────
        ax = axes[0]
        sr = scaling_result

        ax.scatter(
            sr.log_mass,
            sr.log_atp,
            color="steelblue",
            s=70,
            zorder=5,
            label="Empirical species",
        )

        for i, label in enumerate(sr.species_labels):
            ax.annotate(
                label,
                (sr.log_mass[i], sr.log_atp[i]),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=7,
            )

        # Regression line
        x_fit = np.linspace(sr.log_mass.min() - 0.2, sr.log_mass.max() + 0.2, 100)
        y_fit = sr.intercept + sr.scaling_exponent * x_fit
        ax.plot(
            x_fit,
            y_fit,
            "steelblue",
            lw=1.5,
            linestyle="--",
            label=f"Empirical fit (exp={sr.scaling_exponent:.2f})",
        )

        # APGI prediction band (± 0.05 around exponent 0.75)
        y_apgi_mid = sr.intercept + sr.apgi_predicted_exponent * x_fit
        y_apgi_lo = sr.intercept + (sr.apgi_predicted_exponent - 0.15) * x_fit
        y_apgi_hi = sr.intercept + (sr.apgi_predicted_exponent + 0.15) * x_fit
        ax.fill_between(
            x_fit,
            y_apgi_lo,
            y_apgi_hi,
            alpha=0.15,
            color="tomato",
            label="APGI prediction band (0.75 ± 0.15)",
        )
        ax.plot(
            x_fit,
            y_apgi_mid,
            "tomato",
            lw=1.5,
            linestyle=":",
            label=f"APGI predicted (exp={sr.apgi_predicted_exponent:.2f})",
        )

        ax.set_xlabel("log₁₀(Brain mass [g])", fontsize=11)
        ax.set_ylabel("log₁₀(ATP cost per bit [a.u.])", fontsize=11)
        ax.set_title("Cross-Species Metabolic Scaling\n(Module 4)", fontsize=11)
        ax.legend(fontsize=7, loc="upper left")
        ax.text(
            0.97,
            0.05,
            f"R²={sr.r_squared:.3f}  p={sr.p_value:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

        # ── Panel 2: κ sensitivity distribution ─────────────────────────────
        if sensitivity is not None and n_panels == 2:
            ax2 = axes[1]
            kappa_samples = sensitivity["kappa_samples"]
            ax2.hist(
                kappa_samples,
                bins=50,
                color="steelblue",
                alpha=0.7,
                edgecolor="white",
                linewidth=0.4,
            )
            ax2.axvspan(50, 200, alpha=0.15, color="green", label="APGI range [50, 200]")
            ax2.axvline(
                double_bridge_result.kappa,
                color="tomato",
                lw=2,
                label=f"κ (nominal) = {double_bridge_result.kappa:.1f}",
            )
            ax2.axvline(
                sensitivity["kappa_median"],
                color="navy",
                lw=1.5,
                linestyle="--",
                label=f"κ median = {sensitivity['kappa_median']:.1f}",
            )

            ax2.set_xlabel("κ [ATP / bit]", fontsize=11)
            ax2.set_ylabel("Count", fontsize=11)
            ax2.set_title("κ Sensitivity Analysis\n(Module 3 — Monte Carlo)", fontsize=11)
            ax2.legend(fontsize=8)
            ax2.text(
                0.97,
                0.95,
                f"{sensitivity['fraction_in_range'] * 100:.1f}% in range",
                transform=ax2.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path


# ===========================================================================
# CONVENIENCE FUNCTION
# ===========================================================================


def run_aggregator(config: Optional[AggregatorConfig] = None) -> AggregatorReport:
    """Run the full APGI Thermodynamic Program Aggregator pipeline."""
    aggregator = ThermodynamicProgramAggregator(config=config)
    return aggregator.run()


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_aggregator()
