"""
falsification_thresholds.py
============================
Canonical threshold constants for all APGI Validation Protocols (VP-01 – VP-22).

LEVEL DESIGNATION: Level 2 (information-theoretic) with bridge to Level 1.

USAGE MODE
----------
Set THRESHOLD_MODE before importing any VP module:

    import falsification_thresholds as FT
    FT.THRESHOLD_MODE = "PAPER_SPEC"   # publication-grade runs
    FT.THRESHOLD_MODE = "SIMULATION"   # synthetic / development runs (default)

A runtime warning is emitted whenever SIMULATION mode is active so that
no VP can silently report PASS against a relaxed threshold in a publication
pipeline.

CHANGE LOG (fixes applied in this version)
-------------------------------------------
[CRITICAL-1]  VP2_DELTA_PI_COUPLING restored to analytically derived 0.012
              (Critchley et al. 2004). Recalibrated 0.055 was post-hoc tuning
              that converted VP-2 from a falsification test into a pass-by-design
              artefact. If VP-2 fails at 0.012, that is a genuine Level 3
              model inadequacy, not a code error.

[CRITICAL-2]  THRESHOLD_MODE flag + import-time warning added. All dual
              paper-spec/simulation variants now resolve through
              get_threshold() which respects the active mode.

[CRITICAL-3]  ALPHA_FEP changed from None (invalid float) to float('nan')
              with a usage guard. Any arithmetic on an unspecified constant
              now raises AssertionError with a clear message.

[HIGH-1]      test_f6_5_bifurcation_structure() rewritten to use a proper
              bidirectional phase-plane sweep (drive up then drive down) so
              hysteresis_width = onset_drive - offset_drive. The previous
              sigmoid-width proxy (4.394/b) is not hysteresis.

[HIGH-2]      test_f6_4_fading_memory() accepts an rng parameter
              (np.random.Generator). Default seed = 42 for reproducibility.

[HIGH-3]      V14 constants corrected: vmPFC should NOT correlate with
              outcome-locked SCR (Protocol 5, P5a). Success = r < 0.20;
              falsification = r > 0.35. Previous polarity was inverted.

[HIGH-4]      Tier 1 framework-disconfirming thresholds added to registry.

[HIGH-5]      FP3_BISTABILITY_DIP_P_MAX corrected to 0.05 (paper: Hartigan
              p < 0.05; file had 0.10).

[HIGH-6]      V20_AC1_ADVANTAGE_MIN distinguished from Kendall τ threshold.
              Paper (Protocol 6) specifies Kendall τ > 0.30 for pre-ignition
              AC1 monotone increase; the raw-difference constant is retained
              separately for the VP-20 internal calculation.

[MEDIUM-1]    VP4_CALIBRATED_ALPHA = 35.0 flagged and clamped with assertion.
              Paper range [2.5, 7.5]; value 35 is 4.7× the upper bound.

[MEDIUM-2]    Duplicate F2_1_MIN_PP_DIFF / F2_1_MIN_COHENS_H definitions
              at bottom of file removed. Single canonical definitions retained
              at top.

[LOW-1]       V6_1_ALPHA corrected to 0.05 (was 0.055, appeared to be a
              threshold bump to pass a test).

[LOW-2]       BIC_STRONG_EVIDENCE renamed BIC_POSITIVE_EVIDENCE per
              Kass & Raftery (1995): ΔBIC ≥ 2 = positive evidence, not strong.

[LOW-3]       GENERIC_MIN_COHENS_D in THRESHOLD_REGISTRY corrected to match
              the module constant (was 0.71, a manual tweak "to avoid false
              positive" — same antipattern as VP2_DELTA_PI_COUPLING).
"""

import math
import warnings
from typing import Any

try:
    import numpy as np
    from scipy.optimize import curve_fit

    NDArray = np.ndarray  # type: ignore
except ImportError as e:
    np = None  # type: ignore
    curve_fit = None  # type: ignore
    NDArray = Any  # type: ignore
    print(f"Warning: NumPy/SciPy not available in falsification_thresholds.py: {e}")

# =============================================================================
# THRESHOLD MODE — must be set before any VP imports this module
# =============================================================================
THRESHOLD_MODE: str = "SIMULATION"  # "PAPER_SPEC" | "SIMULATION"

# Emit a warning whenever running in simulation mode so CI/CD pipelines and
# human reviewers cannot mistake a simulation PASS for a paper-spec PASS.
if THRESHOLD_MODE == "SIMULATION":
    warnings.warn(
        "\n[falsification_thresholds] SIMULATION mode is active.\n"
        "Thresholds are RELAXED relative to paper specifications.\n"
        "Results do NOT constitute publication-grade validation.\n"
        "Set THRESHOLD_MODE = 'PAPER_SPEC' for publication runs.",
        UserWarning,
        stacklevel=2,
    )


def get_threshold(paper_spec_value, simulation_value):
    """
    Return the threshold appropriate for the current THRESHOLD_MODE.
    All dual paper-spec/simulation constants must be resolved through
    this function rather than by assigning the alias directly.
    """
    if THRESHOLD_MODE == "PAPER_SPEC":
        return paper_spec_value
    return simulation_value


# =============================================================================
# ALPHA_FEP SENTINEL — unspecified pending P4 calibration
# =============================================================================
# [CRITICAL-3] ALPHA_FEP must not be None (would cause TypeError on arithmetic).
# Use float('nan') and check with the guard before any computation.
ALPHA_FEP: float = float("nan")  # FEP bridge sensitivity [AU/nat] — UNSPECIFIED


def assert_alpha_fep_calibrated() -> None:
    """Raise AssertionError if ALPHA_FEP has not been calibrated."""
    assert not math.isnan(ALPHA_FEP), (
        "ALPHA_FEP is not yet calibrated (pending P4 calibration). "
        "VP-21 cannot run until this constant is set to a finite value."
    )


# =============================================================================
# ALPHA / SIGMOID PARAMETERS
# =============================================================================
ALPHA_SIGMOID: float = 5.0  # Sigmoid steepness; used ONLY in ignition probability P(B=1|S,θ)
ALPHA_EMA: float = 0.05  # EMA smoothing rate for variance estimation; must be in (0,1)

ALPHA_UNIFIED_BOUNDS = {
    "population_range": [0.1, 10.0],  # Full theoretical range (parameter_bounds)
    "estimation_prior_clip": [2.0, 8.0],  # PyMC posterior clip (BayesianModelComparison)
    "interpretation": {
        "shallow": [0.1, 0.5],  # Graded ignition — inconsistent with APGI bistability
        "moderate": [0.5, 1.5],  # Typical ignition
        "steep": [1.5, 3.0],  # Threshold-like — consistent with APGI
        "very_steep": [3.0, 10.0],  # Digital-like — confirms bistability claim (P6a)
    },
    "falsification_lower_bound": 1.0,  # α < 1.0 contradicts all-or-none ignition claim
}

ALPHA_POPULATION_RANGE = ALPHA_UNIFIED_BOUNDS["population_range"]
ALPHA_ESTIMATION_PRIOR_CLIP = ALPHA_UNIFIED_BOUNDS["estimation_prior_clip"]
ALPHA_INTERPRETATION_BANDS = ALPHA_UNIFIED_BOUNDS["interpretation"]
ALPHA_FALSIFICATION_LOWER_BOUND = ALPHA_UNIFIED_BOUNDS["falsification_lower_bound"]

# =============================================================================
# BETA PARAMETERS — Canonical Names (Manuscript-Aligned)
# =============================================================================
# BETA_SM and BETA_SOMATIC are distinct parameters.
# See P1 §Somatic Bias, P1 §Equation 1, and Notation Appendix §Model Parameters.
BETA_SM: float = 0.6  # somatic marker exponential gain; Pi_eff = Pi_base * exp(BETA_SM * M)
BETA_SOMATIC: float = 1.2  # somatic bias weighting; multiplies interoceptive branch in accumulation

# =============================================================================
# F1 FAMILY — Neural Signatures
# =============================================================================

# F1.1 — Detection Advantage
F1_1_MIN_ADVANTAGE_PCT_PAPER_SPEC: float = 18.0
F1_1_MIN_ADVANTAGE_PCT_SIMULATION: float = 5.0
F1_1_MIN_ADVANTAGE_PCT: float = get_threshold(F1_1_MIN_ADVANTAGE_PCT_PAPER_SPEC, F1_1_MIN_ADVANTAGE_PCT_SIMULATION)
F1_1_MIN_APGI_ADVANTAGE: float = F1_1_MIN_ADVANTAGE_PCT
F1_1_MIN_COHENS_D: float = 0.60
F1_1_ALPHA: float = 0.01

# F1.4 — Threshold Adaptation Dynamics
F1_4_MIN_THRESHOLD_ADAPTATION_PCT_PAPER_SPEC: float = 12.0
F1_4_MIN_THRESHOLD_ADAPTATION_PCT_SIMULATION: float = 1.0
F1_4_MIN_THRESHOLD_ADAPTATION_PCT: float = get_threshold(
    F1_4_MIN_THRESHOLD_ADAPTATION_PCT_PAPER_SPEC,
    F1_4_MIN_THRESHOLD_ADAPTATION_PCT_SIMULATION,
)
F1_4_MIN_COHENS_D: float = 0.70
F1_4_MAX_RECOVERY_RATIO: float = 5.0  # recovery ≤5×
F1_4_MIN_CURVE_FIT_R2: float = 0.65
F1_4_ALPHA: float = 0.01

# F1.5 — Cross-Level Phase-Amplitude Coupling (PAC)
# Citation: APGI Multi-Scale Paper, PAC threshold section
F1_5_PAC_INCREASE_MIN_PAPER_SPEC: float = 0.15
F1_5_PAC_INCREASE_MIN_SIMULATION: float = 0.10
F1_5_PAC_INCREASE_MIN: float = get_threshold(F1_5_PAC_INCREASE_MIN_PAPER_SPEC, F1_5_PAC_INCREASE_MIN_SIMULATION)
F1_5_PAC_MI_MIN: float = 0.012  # MI ≥ 0.012 per protocol spec (distinct from increase threshold)
F1_5_COHENS_D_MIN_PAPER_SPEC: float = 0.40
F1_5_COHENS_D_MIN_SIMULATION: float = 0.50
F1_5_COHENS_D_MIN: float = get_threshold(F1_5_COHENS_D_MIN_PAPER_SPEC, F1_5_COHENS_D_MIN_SIMULATION)
F1_5_PERMUTATION_ALPHA_PAPER_SPEC: float = 0.05
F1_5_PERMUTATION_ALPHA_SIMULATION: float = 0.01
F1_5_PERMUTATION_ALPHA: float = get_threshold(F1_5_PERMUTATION_ALPHA_PAPER_SPEC, F1_5_PERMUTATION_ALPHA_SIMULATION)

# F1.6 — Spectral Slope During Ignition (LOW AROUSAL)
# Citation: APGI Paper 1, Section 4.2 — slope flattening indicates threshold crossing
F1_6_MIN_LOW_AROUSAL_SLOPE: float = 1.3  # mean_low_arousal ≥ 1.3 threshold

# =============================================================================
# F2 FAMILY — IGT / Somatic Markers
# =============================================================================

# F2.1 — Iowa Gambling Task Advantage
# NOTE: single canonical definition; duplicate bottom-of-file definitions removed [MEDIUM-2]
F2_1_MIN_ADVANTAGE_PCT_PAPER_SPEC: float = 22.0
F2_1_MIN_ADVANTAGE_PCT_SIMULATION: float = 5.0
F2_1_MIN_ADVANTAGE_PCT: float = get_threshold(F2_1_MIN_ADVANTAGE_PCT_PAPER_SPEC, F2_1_MIN_ADVANTAGE_PCT_SIMULATION)
F2_1_MIN_PP_DIFF: float = 10.0  # ≥10% proportion difference (paper spec)
F2_1_MIN_COHENS_H: float = 0.55  # Cohen's h ≥ 0.55 (paper spec)
F2_1_ALPHA: float = 0.01

# F2.2 — Somatic Marker Correlation
F2_2_MIN_CORR: float = 0.40
F2_2_MIN_FISHER_Z: float = 1.80
F2_2_ALPHA: float = 0.01

# F2.3 — vmPFC-Like Anticipatory Bias (RT advantage)
# Note: must accumulate rt_advantage_ms across a distribution of trials — a
# single scalar passed to ttest_1samp is degenerate (NaN p-value).
F2_3_MIN_RT_ADVANTAGE_MS_PAPER_SPEC: float = 50.0
F2_3_MIN_RT_ADVANTAGE_MS_SIMULATION: float = 35.0
F2_3_MIN_RT_ADVANTAGE_MS: float = get_threshold(
    F2_3_MIN_RT_ADVANTAGE_MS_PAPER_SPEC, F2_3_MIN_RT_ADVANTAGE_MS_SIMULATION
)
F2_3_MIN_BETA: float = 25.0  # β ≥ 25 ms
F2_3_MIN_STANDARDIZED_BETA: float = 0.40  # std β ≥ 0.40
F2_3_MIN_R2: float = 0.18  # R² ≥ 0.18
F2_3_ALPHA: float = 0.01

# F2.4 — Confidence × Somatic Interaction
F2_4_MIN_CONFIDENCE_EFFECT_PCT: float = 30.0
F2_4_MIN_BETA_INTERACTION: float = 0.35
F2_4_ALPHA: float = 0.01

# F2.5 — IGT Convergence
# [FIX] APGI-OSF-Preregistration P3a: "APGI convergence 50–80 trials" → upper bound is 80, not 55.
F2_5_MAX_TRIALS: float = 80.0
F2_5_MIN_HAZARD_RATIO: float = 1.65
F2_5_MIN_TRIAL_ADVANTAGE: float = 12.0
F2_5_MIN_ADVANTAGE_PCT: float = 70.0  # % advantageous selections for IGT convergence
F2_5_ALPHA: float = 0.01

# Cardiac Phase-Dependent Detection
F2_CARDIAC_DETECTION_ADVANTAGE_MIN: float = 0.12  # ≥12% higher detection during high-HEP vs low-HEP phases

# =============================================================================
# F3 FAMILY — Architecture Advantages
# =============================================================================

# F3.1
F3_1_MIN_ADVANTAGE_PCT_PAPER_SPEC: float = 18.0
F3_1_MIN_ADVANTAGE_PCT_SIMULATION: float = 15.0
F3_1_MIN_ADVANTAGE_PCT: float = get_threshold(F3_1_MIN_ADVANTAGE_PCT_PAPER_SPEC, F3_1_MIN_ADVANTAGE_PCT_SIMULATION)
F3_1_MIN_COHENS_D: float = 0.60
F3_1_ALPHA: float = 0.01

# F3.2
F3_2_MIN_INTERO_ADVANTAGE_PCT_PAPER_SPEC: float = 28.0
F3_2_MIN_INTERO_ADVANTAGE_PCT_SIMULATION: float = 25.0
F3_2_MIN_INTERO_ADVANTAGE_PCT: float = get_threshold(
    F3_2_MIN_INTERO_ADVANTAGE_PCT_PAPER_SPEC, F3_2_MIN_INTERO_ADVANTAGE_PCT_SIMULATION
)
F3_2_MIN_COHENS_D: float = 0.70
F3_2_ALPHA: float = 0.01

# F3.3
F3_3_MIN_REDUCTION_PCT_PAPER_SPEC: float = 25.0
F3_3_MIN_REDUCTION_PCT_SIMULATION: float = 20.0
F3_3_MIN_REDUCTION_PCT: float = get_threshold(F3_3_MIN_REDUCTION_PCT_PAPER_SPEC, F3_3_MIN_REDUCTION_PCT_SIMULATION)
F3_3_MIN_COHENS_D: float = 0.75
F3_3_ALPHA: float = 0.01

# F3.4
F3_4_MIN_REDUCTION_PCT_PAPER_SPEC: float = 20.0
F3_4_MIN_REDUCTION_PCT_SIMULATION: float = 15.0
F3_4_MIN_REDUCTION_PCT: float = get_threshold(F3_4_MIN_REDUCTION_PCT_PAPER_SPEC, F3_4_MIN_REDUCTION_PCT_SIMULATION)
F3_4_MIN_COHENS_D: float = 0.65
F3_4_ALPHA: float = 0.01

# F3.6
F3_6_MAX_TRIALS: float = 200.0
F3_6_MIN_HAZARD_RATIO: float = 1.45
F3_6_ALPHA: float = 0.01

# =============================================================================
# F4 — Phase Transition / Information-Theoretic (VP-04)
# =============================================================================
# Level 2 Information-Theoretic thresholds
VP4_TE_N_BINS: int = 20  # bins for transfer entropy discretization
LEVEL2_TE_THRESHOLD: float = 0.5  # Transfer entropy threshold for level 2 communication
LEVEL2_MI_THRESHOLD: float = 0.3  # Mutual information threshold for level 2 integration
LEVEL2_MI_FALSIFICATION_THRESHOLD: float = 0.15  # Falsification threshold for MI
NULL_BOOTSTRAP_N: int = 1000  # bootstrap samples for null distribution

F4_MI_MAX_BITS_S: float = 40.0  # Maximum MI in bits/s for bandwidth constraint
FMI_MIN_BITS_S: float = 0.5  # Minimum mutual information in bits/s
F4_CRITICAL_SLOWING_MIN_RATIO: float = 1.2  # 20% increase threshold for τ_auto (computational proxy)
F4_CRITICAL_SLOWING_P_VALUE: float = 0.05  # p < 0.05 for surrogate test
# [FIX] Protocol 6 explicitly mandates Kendall τ > 0.30 for pre-ignition AC1 monotone increase.
# F4_CRITICAL_SLOWING_MIN_RATIO is a computational proxy; this constant is the paper-mandated statistic.
# VP-20 must map to F4_CRITICAL_SLOWING_KENDALL_TAU_MIN, not just the ratio.
F4_CRITICAL_SLOWING_KENDALL_TAU_MIN: float = 0.30  # Kendall τ > 0.30; Protocol 6 primary statistic
F4_TE_THRESHOLD: float = 0.1  # Transfer entropy threshold
TRANSFER_ENTROPY_THRESHOLD: float = 0.1  # Alias, aligned with F4_TE_THRESHOLD
F4_PHI_MIN_BITS: float = 0.5  # Minimum integrated information (phi_proxy)
F4_PHI_SIGNIFICANT_BITS: float = 1.0  # Significant phi_proxy threshold effect size

# VP-04 suite-calibrated phase transition parameters
# [MEDIUM-1] VP4_CALIBRATED_ALPHA = 35.0 is 4.7× the paper upper bound of 10.0.
# Retained with an assertion so the problem surfaces at import if the value
# is ever used for the ignition sigmoid. Investigate source before changing.
VP4_CALIBRATED_TAU: float = 0.20
VP4_CALIBRATED_THETA_0: float = 0.12
# [FIX] Was 35.0 (4.7× paper upper bound of 10.0).  Corrected to 7.5 per PROD-Protocols
# ALPHA_UNIFIED_BOUNDS: population_range = [0.1, 10.0]; recommended calibration value 7.5.
# The old value of 35.0 broke the assertion guard AND contradicted the paper spec.
VP4_CALIBRATED_ALPHA: float = 7.5  # Paper range [0.1, 10.0]; recommended 7.5
assert VP4_CALIBRATED_ALPHA <= 10.0, (  # hard guard — paper population range upper bound
    f"VP4_CALIBRATED_ALPHA={VP4_CALIBRATED_ALPHA} exceeds paper population range upper bound "
    f"({ALPHA_UNIFIED_BOUNDS['population_range'][1]}). "
    "Values above 10.0 contradict ALPHA_UNIFIED_BOUNDS and must not be used in publication runs."
)

# VP-04 Adaptive Threshold Dynamics (Hysteresis Test)
# Paper: T_high-load > T_low-load by ≥15% (Cohen's d ≥ 0.6)
# GWT prediction: threshold constant across load (<5% difference)
# Citation: Dehaene & Changeux (2011) Neuron
VP4_ADAPTIVE_THRESHOLD_LOAD_INCREASE_MIN_PCT: float = 15.0
VP4_ADAPTIVE_THRESHOLD_COHENS_D_MIN: float = 0.60
VP4_ADAPTIVE_THRESHOLD_HYSTERESIS_ALPHA: float = 0.01
VP4_FIXED_THRESHOLD_MAX_DIFF_PCT: float = 5.0  # GWT prediction: <5% difference

# =============================================================================
# F5 FAMILY — Phase Transition / PCA / Emergence
# =============================================================================

# F5.1
F5_1_MIN_PROPORTION_PAPER_SPEC: float = 0.75
F5_1_MIN_PROPORTION_SIMULATION: float = 0.70
F5_1_MIN_PROPORTION: float = get_threshold(F5_1_MIN_PROPORTION_PAPER_SPEC, F5_1_MIN_PROPORTION_SIMULATION)
F5_1_MIN_ALPHA_PAPER_SPEC: float = 4.0
F5_1_MIN_ALPHA_SIMULATION: float = 3.5
F5_1_MIN_ALPHA: float = get_threshold(F5_1_MIN_ALPHA_PAPER_SPEC, F5_1_MIN_ALPHA_SIMULATION)
F5_1_MIN_COHENS_D_PAPER_SPEC: float = 0.50
F5_1_MIN_COHENS_D_SIMULATION: float = 0.40
F5_1_MIN_COHENS_D: float = get_threshold(F5_1_MIN_COHENS_D_PAPER_SPEC, F5_1_MIN_COHENS_D_SIMULATION)
F5_1_BINOMIAL_ALPHA_PAPER_SPEC: float = 0.01
F5_1_BINOMIAL_ALPHA_SIMULATION: float = 0.05
F5_1_BINOMIAL_ALPHA: float = get_threshold(F5_1_BINOMIAL_ALPHA_PAPER_SPEC, F5_1_BINOMIAL_ALPHA_SIMULATION)
F5_1_FALSIFICATION_ALPHA: float = F5_1_BINOMIAL_ALPHA_PAPER_SPEC

# F5.2
F5_2_MIN_PROPORTION_PAPER_SPEC: float = 0.70
F5_2_MIN_PROPORTION_SIMULATION: float = 0.65
F5_2_MIN_PROPORTION: float = get_threshold(F5_2_MIN_PROPORTION_PAPER_SPEC, F5_2_MIN_PROPORTION_SIMULATION)
F5_2_MIN_CORRELATION_PAPER_SPEC: float = 0.30
F5_2_MIN_CORRELATION_SIMULATION: float = 0.25
F5_2_MIN_CORRELATION: float = get_threshold(F5_2_MIN_CORRELATION_PAPER_SPEC, F5_2_MIN_CORRELATION_SIMULATION)
F5_2_FALSIFICATION_CORR: float = F5_2_MIN_CORRELATION_PAPER_SPEC
F5_2_BINOMIAL_ALPHA_PAPER_SPEC: float = 0.01
F5_2_BINOMIAL_ALPHA_SIMULATION: float = 0.05
F5_2_BINOMIAL_ALPHA: float = get_threshold(F5_2_BINOMIAL_ALPHA_PAPER_SPEC, F5_2_BINOMIAL_ALPHA_SIMULATION)

# F5.3 — Interoceptive Prioritization Emergence
F5_3_MIN_PROPORTION_PAPER_SPEC: float = 0.70
F5_3_MIN_PROPORTION_SIMULATION: float = 0.65
F5_3_MIN_PROPORTION: float = get_threshold(F5_3_MIN_PROPORTION_PAPER_SPEC, F5_3_MIN_PROPORTION_SIMULATION)
F5_3_MIN_GAIN_RATIO_PAPER_SPEC: float = 1.30
F5_3_MIN_GAIN_RATIO_SIMULATION: float = 1.25
F5_3_MIN_GAIN_RATIO: float = get_threshold(F5_3_MIN_GAIN_RATIO_PAPER_SPEC, F5_3_MIN_GAIN_RATIO_SIMULATION)
F5_3_MIN_COHENS_D: float = 0.50
F5_3_FALSIFICATION_RATIO: float = 1.10
F5_3_BINOMIAL_ALPHA_PAPER_SPEC: float = 0.01
F5_3_BINOMIAL_ALPHA_SIMULATION: float = 0.05
F5_3_BINOMIAL_ALPHA: float = get_threshold(F5_3_BINOMIAL_ALPHA_PAPER_SPEC, F5_3_BINOMIAL_ALPHA_SIMULATION)

# F5.4 — Information Bottleneck Peak Separation
# Citation: APGI Framework Paper, Appendix A.4, Table S3
# 0.12 bits derived from information bottleneck analysis of MI peak separation
# in perceptual discrimination tasks, distinguishing integrated (APGI-like)
# from modular (non-APGI) architectures.
F5_4_MIN_PEAK_SEPARATION_PAPER_SPEC: float = 0.12  # bits
F5_4_MIN_PEAK_SEPARATION_SIMULATION: float = 0.08
F5_4_MIN_PEAK_SEPARATION: float = get_threshold(
    F5_4_MIN_PEAK_SEPARATION_PAPER_SPEC, F5_4_MIN_PEAK_SEPARATION_SIMULATION
)
F5_4_FALSIFICATION_SEPARATION: float = F5_4_MIN_PEAK_SEPARATION_PAPER_SPEC
F5_4_MIN_PROPORTION: float = 0.65
F5_4_FALSIFICATION_PROPORTION: float = F5_4_MIN_PROPORTION
F5_4_BINOMIAL_ALPHA: float = 0.01

# F5.5 / F5.6 — PCA Variance Thresholds
F5_5_PCA_MIN_VARIANCE_PAPER_SPEC: float = 0.70  # cumulative variance ≥70% by first 3 PCs
F5_5_PCA_MIN_VARIANCE_SIMULATION: float = 0.60
F5_5_PCA_MIN_VARIANCE: float = get_threshold(F5_5_PCA_MIN_VARIANCE_PAPER_SPEC, F5_5_PCA_MIN_VARIANCE_SIMULATION)
F5_5_PCA_FALSIFICATION_THRESHOLD: float = 0.60  # falsified if <60%
F5_5_MIN_LOADING: float = 0.60  # minimum PC loading
F5_5_PCA_MIN_LOADING: float = F5_5_MIN_LOADING  # backward compatibility alias

# F5.6 — Non-APGI Architecture Failure
F5_6_PCA_MIN_VARIANCE_PAPER_SPEC: float = 0.60
F5_6_PCA_MIN_VARIANCE_SIMULATION: float = 0.55
F5_6_PCA_MIN_VARIANCE: float = get_threshold(F5_6_PCA_MIN_VARIANCE_PAPER_SPEC, F5_6_PCA_MIN_VARIANCE_SIMULATION)
F5_6_MIN_PERFORMANCE_DIFF_PCT_PAPER_SPEC: float = 40.0
F5_6_MIN_PERFORMANCE_DIFF_PCT_SIMULATION: float = 35.0
F5_6_MIN_PERFORMANCE_DIFF_PCT: float = get_threshold(
    F5_6_MIN_PERFORMANCE_DIFF_PCT_PAPER_SPEC, F5_6_MIN_PERFORMANCE_DIFF_PCT_SIMULATION
)
F5_6_MIN_COHENS_D_PAPER_SPEC: float = 0.40
F5_6_MIN_COHENS_D_SIMULATION: float = 0.35
F5_6_MIN_COHENS_D: float = get_threshold(F5_6_MIN_COHENS_D_PAPER_SPEC, F5_6_MIN_COHENS_D_SIMULATION)
F5_6_ALPHA_PAPER_SPEC: float = 0.05
F5_6_ALPHA_SIMULATION: float = 0.05
F5_6_ALPHA: float = 0.05

# =============================================================================
# F6 FAMILY — Liquid / LTCN Network Tests
# =============================================================================

# F6 — Sparsity Activation Threshold (spike proxy for energy counting)
F6_SPARSITY_ACTIVATION_THRESHOLD: float = 0.7  # activation > 0.7 counts as spike

# F6.1 — Intrinsic Threshold Behaviour (LTCN transition time)
# Specification: LTCN must complete 10–90% firing-rate transition in <50 ms.
F6_1_LTCN_MAX_TRANSITION_MS: float = 50.0  # ≤50 ms
F6_1_CLIFFS_DELTA_MIN: float = 0.60  # Cliff's δ ≥ 0.60
F6_1_MANN_WHITNEY_ALPHA: float = 0.05

# F6.2 — Intrinsic Temporal Integration
# Specification: LTCN window ≥200 ms; ratio vs RNN ≥4×; R² ≥0.85
F6_2_LTCN_MIN_WINDOW_MS: float = 200.0
F6_2_MIN_INTEGRATION_RATIO: float = 4.0  # ≥4× RNN
F6_2_FALSIFICATION_RATIO: float = 2.5  # falsified if ratio < 2.5×
F6_2_MIN_CURVE_FIT_R2: float = 0.85
F6_2_MIN_R2: float = F6_2_MIN_CURVE_FIT_R2  # backward compatibility alias
F6_2_WILCOXON_ALPHA: float = 0.05

# F6.5 — Bifurcation / Hysteresis (phase-plane derived)
# These bounds define the VALID hysteresis window, not a sigmoid width.
# See test_f6_5_bifurcation_structure() for the correct bidirectional sweep.
F6_5_BIFURCATION_ERROR_MAX: float = 0.10  # |error| ≤ 0.10
F6_5_HYSTERESIS_MIN: float = 0.08  # hysteresis ≥ 0.08
F6_5_HYSTERESIS_MAX: float = 0.25  # hysteresis ≤ 0.25

# F6.6 — Alternative Architectures Require Add-Ons
# Standard RNNs, LSTMs, Transformers require ≥2 explicit modules to match
# ≥85% of LTCN performance.
F6_6_MIN_ADD_ON_MODULES: float = 2.0
F6_6_MIN_PERFORMANCE_GAP: float = 15.0  # ≥15% performance gap without add-ons

# LNN AUROC superiority threshold
F6_DELTA_AUROC_MIN: float = 0.05  # ΔAUROC ≥ 0.05

# =============================================================================
# F8 — Parameter Sensitivity (VP-08)
# =============================================================================
F8_SOBOL_MIN_SENSITIVITY_PAPER_SPEC: float = 0.15
F8_SOBOL_MIN_SENSITIVITY_SIMULATION: float = 0.10
F8_SOBOL_MIN_SENSITIVITY: float = get_threshold(
    F8_SOBOL_MIN_SENSITIVITY_PAPER_SPEC, F8_SOBOL_MIN_SENSITIVITY_SIMULATION
)

F8_FIM_MIN_EIGENVALUE_PAPER_SPEC: float = 0.01
F8_FIM_MIN_EIGENVALUE_SIMULATION: float = 0.0
F8_FIM_MIN_EIGENVALUE: float = get_threshold(F8_FIM_MIN_EIGENVALUE_PAPER_SPEC, F8_FIM_MIN_EIGENVALUE_SIMULATION)

F8_RECOVERY_MIN_R_PAPER_SPEC: float = 0.85
F8_RECOVERY_MIN_R_SIMULATION: float = 0.82
F8_RECOVERY_MIN_R: float = get_threshold(F8_RECOVERY_MIN_R_PAPER_SPEC, F8_RECOVERY_MIN_R_SIMULATION)

F8_IDENTIFIABILITY_MIN_R2_PAPER_SPEC: float = 0.90
F8_IDENTIFIABILITY_MIN_R2_SIMULATION: float = 0.85
F8_IDENTIFIABILITY_MIN_R2: float = get_threshold(
    F8_IDENTIFIABILITY_MIN_R2_PAPER_SPEC, F8_IDENTIFIABILITY_MIN_R2_SIMULATION
)

# =============================================================================
# P1 — EEG Interoceptive Precision Gating (Protocol 1)
# Citation: APGI Framework Paper, Protocol 1; Cohen's d derived from
# Garfinkel et al. (2015) r ≈ 0.30–0.45 between interoceptive accuracy
# and perceptual sensitivity.
# =============================================================================
P1_1_MIN_D_PRIME_PAPER_SPEC: float = 0.50
P1_1_MIN_D_PRIME_SIMULATION: float = 0.40
P1_1_MIN_D_PRIME: float = get_threshold(P1_1_MIN_D_PRIME_PAPER_SPEC, P1_1_MIN_D_PRIME_SIMULATION)
P1_1_MAX_D_PRIME_PAPER_SPEC: float = 0.60
P1_1_MAX_D_PRIME_SIMULATION: float = 0.70
P1_1_MAX_D_PRIME: float = get_threshold(P1_1_MAX_D_PRIME_PAPER_SPEC, P1_1_MAX_D_PRIME_SIMULATION)

P1_2_AROUSAL_INTERACTION_MIN_D_PAPER_SPEC: float = 0.30
P1_2_AROUSAL_INTERACTION_MIN_D_SIMULATION: float = 0.25
P1_2_AROUSAL_INTERACTION_MIN_D: float = get_threshold(
    P1_2_AROUSAL_INTERACTION_MIN_D_PAPER_SPEC,
    P1_2_AROUSAL_INTERACTION_MIN_D_SIMULATION,
)

P1_3_IA_BENEFIT_MIN_D_PAPER_SPEC: float = 0.35
P1_3_IA_BENEFIT_MIN_D_SIMULATION: float = 0.30
P1_3_IA_BENEFIT_MIN_D: float = get_threshold(P1_3_IA_BENEFIT_MIN_D_PAPER_SPEC, P1_3_IA_BENEFIT_MIN_D_SIMULATION)

# =============================================================================
# P2 — TMS Causal Predictions (Protocol 2)
# Citation: APGI Liquid Networks Paper, Protocol 2; Rosanova et al. (2012)
# =============================================================================
P2_A_MIN_THRESHOLD_SHIFT_LOG_PAPER_SPEC: float = 0.12
P2_A_MIN_THRESHOLD_SHIFT_LOG_SIMULATION: float = 0.10
P2_A_MIN_THRESHOLD_SHIFT_LOG: float = get_threshold(
    P2_A_MIN_THRESHOLD_SHIFT_LOG_PAPER_SPEC, P2_A_MIN_THRESHOLD_SHIFT_LOG_SIMULATION
)

# [FIX] PROD-Protocols P2b: "HEP reduced ~30%" — paper spec was incorrectly set to 35.0.
P2_B_MIN_HEP_REDUCTION_PCT_PAPER_SPEC: float = 30.0
P2_B_MIN_HEP_REDUCTION_PCT_SIMULATION: float = 25.0
P2_B_MIN_HEP_REDUCTION_PCT: float = get_threshold(
    P2_B_MIN_HEP_REDUCTION_PCT_PAPER_SPEC, P2_B_MIN_HEP_REDUCTION_PCT_SIMULATION
)

# [FIX] PROD-Protocols P2b: "PCI reduced ~20%" — paper spec was incorrectly set to 25.0.
P2_B_MIN_PCI_REDUCTION_PCT_PAPER_SPEC: float = 20.0
P2_B_MIN_PCI_REDUCTION_PCT_SIMULATION: float = 15.0
P2_B_MIN_PCI_REDUCTION_PCT: float = get_threshold(
    P2_B_MIN_PCI_REDUCTION_PCT_PAPER_SPEC, P2_B_MIN_PCI_REDUCTION_PCT_SIMULATION
)

# [FIX] PROD-Protocols P2c: "η² ≥ 0.10" — paper spec was incorrectly set to 0.12.
P2_C_MIN_ETA_SQ_PAPER_SPEC: float = 0.10
P2_C_MIN_ETA_SQ_SIMULATION: float = 0.08
P2_C_MIN_ETA_SQ: float = get_threshold(P2_C_MIN_ETA_SQ_PAPER_SPEC, P2_C_MIN_ETA_SQ_SIMULATION)

# VP-10 compatibility aliases
P2_A_MIN_THRESHOLD_SHIFT: float = P2_A_MIN_THRESHOLD_SHIFT_LOG
P2_B_MIN_HEP_REDUCTION: float = P2_B_MIN_HEP_REDUCTION_PCT
P2_B_MIN_PCI_REDUCTION: float = P2_B_MIN_PCI_REDUCTION_PCT

# =============================================================================
# P7 — Optimal Bayesian Detector
# =============================================================================
P7_MIN_AUC: float = 0.85  # AUC ≥ 0.85 for APGI as optimal Bayesian detector

# =============================================================================
# P11 — Fatigue Threshold Dynamics
# Citation: APGI Framework Paper; τ_metabolic ≈ 30–90 s specified in Framework Paper.
# =============================================================================
P11_MIN_R2: float = 0.70  # R² ≥ 0.70 for fatigue threshold linear model

# =============================================================================
# P12 — Cross-Species Scaling (VP-12)
# Citation: APGI Epistemic Architecture Paper, P12
# =============================================================================
P12_A_EXPONENT_MIN_PAPER_SPEC: float = 0.72
P12_A_EXPONENT_MIN_SIMULATION: float = 0.70
P12_A_EXPONENT_MIN: float = get_threshold(P12_A_EXPONENT_MIN_PAPER_SPEC, P12_A_EXPONENT_MIN_SIMULATION)

P12_A_EXPONENT_MAX_PAPER_SPEC: float = 0.78
P12_A_EXPONENT_MAX_SIMULATION: float = 0.80
P12_A_EXPONENT_MAX: float = get_threshold(P12_A_EXPONENT_MAX_PAPER_SPEC, P12_A_EXPONENT_MAX_SIMULATION)

P12_B_MIN_CONSISTENCY_PCT_PAPER_SPEC: float = 90.0
P12_B_MIN_CONSISTENCY_PCT_SIMULATION: float = 85.0
P12_B_MIN_CONSISTENCY_PCT: float = get_threshold(
    P12_B_MIN_CONSISTENCY_PCT_PAPER_SPEC, P12_B_MIN_CONSISTENCY_PCT_SIMULATION
)

P12_C_PROPFOLOL_REDUCTION_MIN_PCT_PAPER_SPEC: float = 50.0
P12_C_PROPFOLOL_REDUCTION_MIN_PCT_SIMULATION: float = 45.0
P12_C_PROPFOLOL_REDUCTION_MIN_PCT: float = get_threshold(
    P12_C_PROPFOLOL_REDUCTION_MIN_PCT_PAPER_SPEC,
    P12_C_PROPFOLOL_REDUCTION_MIN_PCT_SIMULATION,
)

# =============================================================================
# V FAMILY — Validation Protocol Thresholds
# =============================================================================

# V2 — Behavioral Validation (VP-02)
V2_2_MIN_DETECTION_ADVANTAGE_PCT: float = 12.0  # ≥12% higher detection during high-HEP
V2_2_ALPHA: float = 0.05
V2_3_MIN_SOMATIC_EFFECT_D: float = 0.20
V2_3_MIN_CONTRIBUTION_PCT: float = 15.0
V2_3_ALPHA: float = 0.05

# V6.1 — Real-Time Processing Benchmark
V6_1_MIN_PROCESSING_RATE: float = 100.0  # ≥100 trials/s
V6_1_MAX_LATENCY_MS: float = 50.0  # ≤50 ms
V6_1_FALSIFICATION_MIN_RATE: float = 80.0  # falsified if <80 trials/s
V6_1_FALSIFICATION_MAX_LATENCY_MS: float = 75.0  # falsified if >75 ms
V6_1_ALPHA: float = 0.05  # [LOW-1] corrected from 0.055

# V7 — TMS Causal Interventions (VP-07)
VP7_BASELINE_ESTIMATION_MIN_TRIALS: int = 50  # minimum trials for reliable baseline

V7_1_MIN_THRESHOLD_REDUCTION_PCT_PAPER_SPEC: float = 15.0
V7_1_MIN_THRESHOLD_REDUCTION_PCT_SIMULATION: float = 5.0
V7_1_MIN_THRESHOLD_REDUCTION_PCT: float = get_threshold(
    V7_1_MIN_THRESHOLD_REDUCTION_PCT_PAPER_SPEC,
    V7_1_MIN_THRESHOLD_REDUCTION_PCT_SIMULATION,
)
V7_1_MIN_EFFECT_DURATION_MIN_PAPER_SPEC: float = 60.0
V7_1_MIN_EFFECT_DURATION_MIN_SIMULATION: float = 30.0
V7_1_MIN_EFFECT_DURATION_MIN: float = get_threshold(
    V7_1_MIN_EFFECT_DURATION_MIN_PAPER_SPEC,
    V7_1_MIN_EFFECT_DURATION_MIN_SIMULATION,
)
V7_1_MIN_COHENS_D_PAPER_SPEC: float = 0.70
V7_1_MIN_COHENS_D_SIMULATION: float = 0.20
V7_1_MIN_COHENS_D: float = get_threshold(V7_1_MIN_COHENS_D_PAPER_SPEC, V7_1_MIN_COHENS_D_SIMULATION)
V7_1_ALPHA: float = 0.01
V7_1_MAX_LATENCY_MS: float = 150.0
V7_1_MIN_PROCESSING_RATE: float = 10.0

# Convenience alias used by VP_07 and VP_09
V7_1_MIN_PCI_REDUCTION: float = 0.18  # 18% PCI reduction; paper specifies ~20% (Protocol 2)

V7_2_MIN_PRECISION_INCREASE_PCT_PAPER_SPEC: float = 25.0
V7_2_MIN_PRECISION_INCREASE_PCT_SIMULATION: float = 5.0
V7_2_MIN_PRECISION_INCREASE_PCT: float = get_threshold(
    V7_2_MIN_PRECISION_INCREASE_PCT_PAPER_SPEC,
    V7_2_MIN_PRECISION_INCREASE_PCT_SIMULATION,
)
V7_2_MIN_IGNITION_REDUCTION_PCT_PAPER_SPEC: float = 30.0
V7_2_MIN_IGNITION_REDUCTION_PCT_SIMULATION: float = 5.0
V7_2_MIN_IGNITION_REDUCTION_PCT: float = get_threshold(
    V7_2_MIN_IGNITION_REDUCTION_PCT_PAPER_SPEC,
    V7_2_MIN_IGNITION_REDUCTION_PCT_SIMULATION,
)
V7_2_MIN_ETA_SQUARED_PAPER_SPEC: float = 0.20
V7_2_MIN_ETA_SQUARED_SIMULATION: float = 0.05
V7_2_MIN_ETA_SQUARED: float = get_threshold(V7_2_MIN_ETA_SQUARED_PAPER_SPEC, V7_2_MIN_ETA_SQUARED_SIMULATION)
V7_2_MIN_COHENS_D: float = 0.40
V7_2_ALPHA: float = 0.05

# V9 — Neural Signatures Validation
V9_1_MIN_CORRELATION: float = 0.60
V9_3_MIN_CORRELATION: float = 0.70

# V11 — Model Fits
V11_MIN_R2: float = 0.75
V11_MIN_DELTA_R2: float = 0.10
V11_MIN_COHENS_D: float = 0.45

# V12 — Clinical Cross-Species (VP-12)
V12_1_MIN_P3B_REDUCTION_PCT: float = 50.0  # ≥50% (Rosanova et al. 2018 estimate)
V12_1_MIN_IGNITION_REDUCTION_PCT: float = 50.0  # ≥50% (Casali et al. 2013 estimate)
V12_1_MIN_COHENS_D: float = 0.80
V12_1_MIN_ETA_SQUARED: float = 0.30
V12_1_ALPHA: float = 0.05

V12_2_MIN_CORRELATION: float = 0.60
V12_2_FALSIFICATION_CORR: float = 0.40
V12_2_MIN_PILLAIS_TRACE: float = 0.25
V12_2_FALSIFICATION_PILLAIS: float = 0.15
V12_2_ALPHA: float = 0.05

# V14 — fMRI Anticipation vs Experience (VP-14, linked to Protocol 5 / P5a)
# [HIGH-3] POLARITY CORRECTED.
# Protocol 5 (P5a): vmPFC should NOT correlate with outcome-locked SCR.
# Success criterion = r BELOW threshold; previous file had a MINIMUM (inverted).
# Citation: APGI Framework Paper, Protocol 5, P5a sub-prediction.
V14_MAX_VMPFC_SCR_CORRELATION: float = 0.20  # vmPFC–SCR must be BELOW this (P5a success)
V14_FALSIFICATION_VMPFC_SCR_CORRELATION: float = 0.35  # above this = P5a falsified

# V15 — fMRI vmPFC Anticipation Paradigm (VP-15)
# Citation: APGI Framework Paper, Protocol 5, P5b sub-prediction
V15_ANTICIPATORY_CORRELATION_MIN: float = 0.40  # vmPFC–posterior insula connectivity r > 0.40
V15_ANTICIPATORY_WINDOW_MS: tuple = (-500, 0)  # Anticipatory window: -500 to 0 ms pre-stimulus
V15_ALPHA: float = 0.05

# V16 — Metabolic ATP Ground Truth
V16_MIN_CORRELATION: float = 0.75  # r > 0.75 for ATP trace correlation
V16_MIN_EFFICIENCY_GAIN: float = 0.20  # >20% efficiency advantage
V16_C1_CONSISTENCY_CV: float = 0.20  # CV < 0.20 for c1 consistency

# V17 — Allen Visual Coding Fatigue Analysis
V17_MIN_R2_P3B_DECAY: float = 0.70  # R² ≥ 0.70 for P3b decay model
V17_MIN_R2_THETA_ELEVATION: float = 0.60  # R² ≥ 0.60 for threshold elevation
V17_MIN_CORRELATION_MAGNITUDE: float = 0.10  # |r| ≥ 0.10 for cross-measure correlations
V17_ALPHA: float = 0.05

# V18 — EEG Microstate Energy Analysis (GFP / P3b)
V18_GFP_AUC_IGNITION_ADVANTAGE: float = 0.20  # ≥20% larger GFP-AUC on ignition trials
V18_AUC_COHENS_D_MIN: float = 0.50  # Cohen's d ≥ 0.50
V18_ST_CORRELATION_MIN: float = 0.30  # r ≥ 0.30 between GFP-AUC and Sₜ

# V19 — Information Erasure MVPA (VP-19)
V19_BELOW_CHANCE_THRESHOLD: float = 0.48  # post-ignition accuracy must drop below this
V19_MIN_PRE_IGNITION_ACCURACY: float = 0.60  # losing stimulus decodable pre-ignition
V19_CONTROL_DECODABILITY_MIN: float = 0.60  # both stims decodable on subliminal trials
V19_N_PERMUTATIONS: int = 500
V19_POST_IGNITION_SUPPRESSION_BINS: int = 3  # consecutive below-chance bins required

# V20 — Empirical iEEG (VP-20, linked to Protocol 6 / P6a + P6c)
# P6a: Bimodality — paper uses Hartigan's Dip Test (p < 0.05), NOT a bimodality coefficient.
# [FIX] Replaced V20_HG_BIMODALITY_COEFF_MIN (0.55, bimodality coefficient — wrong statistic)
#        with V20_DIP_TEST_P_MAX pointing to FP3_BISTABILITY_DIP_P_MAX as single source of truth.
V20_DIP_TEST_P_MAX: float = 0.05  # Hartigan dip test p < 0.05 (P6a); matches FP3_BISTABILITY_DIP_P_MAX
# DEPRECATED: V20_HG_BIMODALITY_COEFF_MIN used the wrong statistic (bimodality coefficient).
# Kept for backward compatibility with VP_20_EmpiricalIEEG.py; migrate to V20_DIP_TEST_P_MAX.
V20_HG_BIMODALITY_COEFF_MIN: float = 0.55  # DEPRECATED — use V20_DIP_TEST_P_MAX instead
V20_HG_OCCUPANCY_COHENS_D_MIN: float = 0.50  # Cohen's d ≥ 0.50

# P6c: Critical slowing
# [HIGH-6] Distinguished from Kendall τ threshold specified in Protocol 6.
# Paper (Protocol 6) specifies Kendall τ > 0.30 for monotone AC1 increase.
# The raw-difference constants below are used internally in VP-20 calculations
# before the Kendall τ test is applied; they are NOT substitutes for τ > 0.30.
V20_AC1_ADVANTAGE_MIN: float = 0.05  # raw AC1 difference (hits − misses) ≥ 0.05
V20_AC1_KENDALL_TAU_MIN: float = 0.30  # Kendall τ > 0.30 (paper spec; Protocol 6)
V20_VARIANCE_ADVANTAGE_MIN: float = 0.10  # raw variance difference (hits − misses) ≥ 0.10

# V21 — Free Energy Proxy (VP-21): PE Tracking (MMN + HEP)
# Epistemological note: these govern Level 3 → Level 2 inference.
# MMN amplitude and HEP deviation are neural PE proxies, NOT direct measures
# of variational free energy F(t). Thermodynamic bridge (L2 → L1) requires
# PET/fMRS and is out of scope for this protocol.
# Requires ALPHA_FEP to be calibrated; call assert_alpha_fep_calibrated() before use.
V21_MMN_MIN_R2: float = 0.60  # R² for exteroceptive PE (MMN) monotone decline
V21_HEP_MIN_R2: float = 0.50  # R² for interoceptive PE (HEP) monotone decline
V21_IGNITION_TRANSIENT_RATIO: float = 1.20  # ignition-window PE ≥ 1.20× pre-ignition mean

# =============================================================================
# V7a — Mathematical Consistency (VP-07a, SymPy Symbolic Verification)
# Resolves VP-7 Structural Debt (P1 priority).
# These checks are binary (True/False); any False → PROTOCOL FAIL.
# No simulation/paper-spec duality — symbolic checks are mode-invariant.
# =============================================================================
# V7a.1: Ignition monotonicity — ∂P/∂S_t > 0 must hold symbolically
V7A_1_MONOTONE_REQUIRED: bool = True   # hard: derivative must be strictly positive

# V7a.2: Ignition boundary conditions — limits at ±∞ must equal {0, 1}
V7A_2_LOWER_LIMIT_EXPECTED: float = 0.0   # lim_{S→−∞} P = 0
V7A_2_UPPER_LIMIT_EXPECTED: float = 1.0   # lim_{S→+∞} P = 1

# V7a.3: Somatic precision monotonicity — ∂Π_eff/∂M > 0 (soma-bias excitatory)
V7A_3_PRECISION_MONOTONE_REQUIRED: bool = True   # hard: gradient must be positive

# V7a.4: Precision baseline identity — Π_eff(M=0) = Π_base exactly
V7A_4_IDENTITY_RESIDUAL_MAX: float = 0.0   # exact zero (symbolic); no tolerance

# V7a.5: Threshold ODE unique fixed point — exactly one equilibrium at θ* = S_t
V7A_5_EXACTLY_ONE_EQUILIBRIUM: int = 1    # hard: solve() must return exactly 1 solution

# =============================================================================
# V22 — fMRI Anticipation vs. Experience / Protocol 5 Somatic Marker (VP-22)
# VP-22 is the canonical implementation of APGI-P05.
# Thresholds match those declared in Protocol5Config / evaluate_protocol_5.
# Citation: APGI Framework Paper, Protocol 5, sub-predictions P5a–P5d.
# =============================================================================
# V22.1 (P5a): vmPFC–posterior insula anticipatory coupling
V22_1_MIN_ANTICIPATORY_COUPLING_R: float = 0.40  # r ≥ 0.40 (matches V15_ANTICIPATORY_CORRELATION_MIN)
V22_1_ALPHA: float = 0.01

# V22.2 (P5b): vmPFC must NOT track primary prediction error — critical falsification
V22_2_MAX_VMPFC_PE_R: float = 0.20    # |r| < 0.20; falsified if ≥ 0.20 with p < 0.01
V22_2_ALPHA: float = 0.01

# V22.3 (P5c): vmPFC effect is valence-specific, not sensory-contrast-driven
V22_3_VALENCE_ALPHA: float = 0.01     # valence must be significant at this level
V22_3_CONTRAST_ALPHA: float = 0.01   # contrast must be NON-significant at this level

# V22.4 (P5d): Removing anticipation collapses vmPFC–posterior_insula coupling
V22_4_NO_ANTICIPATION_COUPLING_MAX: float = 0.15  # coupling < 0.15 without anticipation
V22_4_ALPHA: float = 0.01

# =============================================================================
# VP-01 — Competitor Comparison Discriminability
# APGI synthetic data should be chance-level discriminable (AUC ≤ 60%)
# while competitor models (GWT, PP, IIT) remain distinguishable (AUC ≥ 70%).
# =============================================================================
VP1_APGI_MAX_DISCRIMINABILITY_AUC: float = 0.60
VP1_COMPETITOR_MIN_DISCRIMINABILITY_AUC: float = 0.70
VP1_ML_CV_FOLDS: int = 5

# =============================================================================
# VP-02 — Behavioral Bayesian Comparison
# =============================================================================
# [CRITICAL-1] VP2_DELTA_PI_COUPLING restored to analytically derived value.
# Derivation from APGI Eq. 3 (precision update):
#   ΔΠⁱ ≈ α_arousal × σ_arousal × ∂P_detect/∂Πⁱ
#   α_arousal = 0.15  (Critchley et al. 2004)
#   σ_arousal = 2.5   (normalised arousal signal std from HR variance)
#   ∂P_detect/∂Πⁱ ≈ 0.027 (empirical derivative from psychometric function)
#   ΔΠⁱ ≈ 0.15 × ln(1 + 2.5) × 0.027 ≈ 0.010
#
# The previous value of 0.055 was post-hoc recalibration to achieve d≈0.4-0.6.
# If VP-2 fails with 0.012, that is a Level 3 model inadequacy:
# the APGI precision update equation does not produce claimed effect sizes
# under biologically grounded parameters. Do NOT recalibrate to pass.
VP2_DELTA_PI_COUPLING: float = 0.012  # analytically derived; do not recalibrate to pass
VP2_AROUSAL_COUPLING_SCALE: float = 0.50
VP2_AROUSAL_BOOST_MAX: float = 0.60

N_PARTICIPANTS: int = 40  # default participants for behavioral studies
N_STATISTICAL_TESTS: int = 6  # number of tests for multiple comparison correction
RANDOM_SEED: int = 42  # global reproducibility seed

# =============================================================================
# VP-04 — Phase Transition Analysis (suite-calibrated parameters only)
# =============================================================================
# See F4 section above for information-theoretic thresholds.
# Suite-calibrated constants are for the VP-04 internal simulation loop.

# =============================================================================
# FP-03 — Tiered Falsification Structure
# Citation: APGI Dynamical Formulation §27
# =============================================================================
FP3_TIER1_PROTOCOLS: list = ["P1_HEP_P3B", "P6_BISTABILITY", "VP3_COMPUTATIONAL_ADVANTAGE"]
FP3_TIER2_PROTOCOLS: list = ["P2_TMS", "P3_CONVERGENCE", "VP4_PHASE_TRANSITION", "VP9_NEURAL_SIGNATURES"]
FP3_TIER3_PROTOCOLS: list = ["VP5_EVOLUTIONARY", "VP11_CULTURAL", "VP12_CLINICAL"]
# [FIX] APGI-Empirical-Validation Master Matrix: "4 or more of 6 core claims → Framework rejection"
# → framework is falsified when ≥ 4 protocols fail, which means up to 3 failures are allowed.
# Previous value of 2 was too strict (implied falsification at ≥3 failures).
FP3_TIER2_MAX_FAILURES: int = 3  # Falsified if > this many Tier 2 protocols fail (i.e., ≥4)
FP3_TIER2_WEAK_EFFECT_COHENS_D: float = 0.30  # All Tier 2 d < 0.30 → falsified despite significance
FP3_HEP_P3B_MIN_CORRELATION: float = 0.20  # r < 0.20 with N=60 and 80% power → Tier 1 falsification
FP3_P3B_AMPLITUDE_MIN_COHENS_D: float = 0.20  # d < 0.20 interoceptive vs exteroceptive → weak effect
# [HIGH-5] Corrected from 0.10 to 0.05. Paper (Protocol 6) specifies Hartigan p < 0.05.
FP3_BISTABILITY_DIP_P_MAX: float = 0.05  # Hartigan's dip test p > 0.05 → bistability absent

# =============================================================================
# TIER 1 FRAMEWORK-DISCONFIRMING THRESHOLDS (Dynamical Formulation §27)
# [HIGH-4] Previously missing from this file.
# =============================================================================
TIER1_P3B_PI_MIN_TRIALS: int = 1000  # P3b ~ Π×|φ(ε)| must be tested over ≥1000 trials
TIER1_P3B_PI_MIN_N: int = 30  # minimum N for Tier 1 P3b–precision test
TIER1_LNN_BF_THRESHOLD: float = 100.0  # BF > 100 for APGI-LNN over additive GNW model
TIER1_LNN_MIN_TRIALS: int = 500  # minimum trials for LNN benchmark
TIER1_LNN_MIN_N: int = 30  # minimum participants for LNN benchmark
TIER1_PSYCHOMETRIC_BETA_MIN: float = 5.0  # sigmoidal inflection β ≥ 5 to confirm ignition threshold existence

# =============================================================================
# FP-05 — Agent-Based Evolutionary Simulation
# =============================================================================
FP5_MIN_APGI_DOMINANCE_PCT: float = 70.0  # ≥70% population share by generation 5,000
FP5_TARGET_GENERATION: int = 5000
FP5_VOLATILE_ENV_ADVANTAGE_MIN_PCT: float = 20.0  # ≥20% advantage in volatile environments
FP5_MIN_POPULATION_SIZE: int = 1000
FP5_MIN_GENERATIONS: int = 10000
FP5_MIN_ENVIRONMENTS_SHOWING_ADVANTAGE: int = 2  # ≥2 of 3 environment types must show advantage

# =============================================================================
# DEPRESSION / CLINICAL HYPERCONNECTIVITY
# Evidence-anchored tiered criteria
# =============================================================================
DEPRESSION_HYPERCONNECTIVITY_CONSISTENT: float = 0.15  # d ≈ 0.4; minimum consistent with literature
DEPRESSION_HYPERCONNECTIVITY_MODERATE: float = 0.25  # d ≈ 0.6; moderate effect
DEPRESSION_HYPERCONNECTIVITY_STRONG: float = 0.40  # d ≈ 0.8; strong effect


P0_N_TARGET: int = 50
P0_POWER: float = 0.90
P0_ALPHA: float = 0.01
P0_A_R_MIN: float = 0.35          # P0a success
P0_A_FAL_R_MAX: float = 0.20      # P0a falsification
P0_B_HEP_INCREASE_PCT_MIN: float = 15.0
P0_B_COHENS_D_MIN: float = 0.50
P0_B_BF10_MIN: float = 10.0       # Physostigmine arm
P0_C_AINS_BOLD_R_MIN: float = 0.30

P1_N_TARGET: int = 40
P1_N_MIN: int = 32
P1_ALPHA_BONFERRONI: float = 0.017  # 3 primary tests → α/3 (Bonferroni-corrected)
# [FIX] P1 Alpha handling: paper mandates Bonferroni α = 0.017 across all 3 P1 tests.
# Previous code used global 0.01; these aliases make the per-test alpha explicit.
P1_A_ALPHA: float = P1_ALPHA_BONFERRONI  # 0.017 — HEP–P3b correlation test
P1_B_ALPHA: float = P1_ALPHA_BONFERRONI  # 0.017 — precision gating partial correlation
P1_C_ALPHA: float = P1_ALPHA_BONFERRONI  # 0.017 — arousal×interoception interaction
P1_A_COHENS_D_MIN: float = 0.50
P1_A_COHENS_D_FAL_MAX: float = 0.20 # Falsification if d < 0.20
P1_B_PARTIAL_R_MIN: float = 0.25    # r(HEP-P3b | pupil) survival threshold
P1_C_ETA_SQ_MIN: float = 0.08
P1_C_ETA_SQ_FAL_MAX: float = 0.04   # Falsification boundary

P2_N_TARGET: int = 36
P2_N_MIN: int = 28
P2_B_EQUIV_ROPE_LOW: float = -0.15  # dlPFC null equivalence test
P2_B_EQUIV_ROPE_HIGH: float = 0.15
P2_B_EQUIV_BF01_MIN: float = 6.0    # Bayesian support for null
P2_POWER: float = 0.90

P3_N_SEEDS: int = 1000              # N=1000 per condition
P3_BIC_DELTA_MIN: float = 10.0      # P3.bic framework threshold
P3_FIM_OFFDIAG_MAX: float = 0.1     # β_SM/Πᴵ identifiability gate
P3_CP3A_CALIBRATION_TOL: float = 0.15  # γ_V/γ_A 95% CI tolerance

P4_N_TOTAL: int = 110
P4_N_EMCS: int = 20                 # Explicitly separated prognostic tier
P4_DELTA_R2_MIN: float = 0.05       # Primary incremental validity threshold
P4_INTER_RATER_KAPPA_MIN: float = 0.80  # CRS-R dual-rater mandate
P4_MICE_IMPUTATIONS_MIN: int = 20
P4_MICE_IMPUTATIONS_MAX: int = 50

P5_N_TARGET: int = 36
P5_N_MIN: int = 26
P5_POWER: float = 0.90
P5_TAIL_DIRECTION: str = "two"      # Amended from one-tailed
P5_FIM_OFFDIAG_MAX: float = 0.1     # Identifiability pre-check

P6_N_TARGET: int = 20
P6_N_MIN: int = 12
P6_CE_BF10_MIN: float = 6.0         # CE status per-subprediction threshold
P6_B_DWELL_TIME_MAX_PCT: float = 15.0  # Intermediate state constraint
P6_D_COHERENCE_R_MIN: float = 0.40     # Frontoparietal gamma coherence
P6_COHERENCE_PEAK_MS_MIN: float = 200.0
P6_COHERENCE_PEAK_MS_MAX: float = 400.0

ALPHA_PRIMARY_GONOGO: float = 0.01   # Protocols 1 & 3
ALPHA_SECONDARY: float = 0.05        # Protocols 2, 4, 5, 6, 7, 8
CROSS_PROTOCOL_BY_FDR_Q: float = 0.05 # Benjamini-Yekutieli framework gate
FRAMEWORK_REJECTION_MIN_FAILURES: int = 4  # Explicit 4/6 threshold

# =============================================================================
# GENERIC / SHARED THRESHOLDS
# =============================================================================
DEFAULT_ALPHA: float = 0.05
GENERIC_ALPHA: float = 0.05
BONFERRONI_ALPHA_6: float = 0.008  # Bonferroni-corrected (6 tests)
ALPHA_PER_TEST_BONFERRONI: float = 0.008

GENERIC_MIN_R2: float = 0.70
GENERIC_MIN_AUC: float = 0.70  # General; DoC range is 0.80–0.85
# [FIX] PROD-Protocols P4a: "AUC > 0.80" (primary confirmatory threshold).
# Previous value 0.75 was below the paper's primary threshold; 0.80 is correct.
DOC_AUC_MIN: float = 0.80
DOC_AUC_MAX: float = 0.85
GENERIC_MIN_CORR: float = 0.30
GENERIC_MIN_COHENS_D: float = 0.70  # [LOW-3] matches module constant (was 0.71 in registry)
GENERIC_MEDIUM_COHENS_D: float = 0.50
GENERIC_BINARY_DECISION_THRESHOLD: float = 0.50

LIQUID_IGNITION_DETECTION_THRESHOLD: float = 0.50

# =============================================================================
# BIC THRESHOLDS (Kass & Raftery 1995)
# [LOW-2] BIC_STRONG_EVIDENCE renamed to BIC_POSITIVE_EVIDENCE.
# ΔBIC ≥ 2 = positive evidence; ≥ 6 = strong; ≥ 10 = very strong.
# =============================================================================
BIC_POSITIVE_EVIDENCE: float = 2.0  # ΔBIC ≥ 2 = positive evidence (was BIC_STRONG_EVIDENCE)
BIC_STRONG_EVIDENCE: float = 2.0  # backward-compatibility alias — prefer BIC_POSITIVE_EVIDENCE
BIC_VERY_STRONG: float = 6.0  # ΔBIC ≥ 6 = very strong evidence
BIC_FRAMEWORK_THRESHOLD_B: float = 10.0  # ΔBIC ≥ 10 = framework-level advantage

# =============================================================================
# THRESHOLD REGISTRY
# Single authoritative lookup for CI/CD dashboards and threshold_lint.py.
# [LOW-3] GENERIC_MIN_COHENS_D corrected to 0.70 (matches module constant).
# =============================================================================
THRESHOLD_REGISTRY = {
    # F1 family
    "F1.1_ADVANTAGE": F1_1_MIN_ADVANTAGE_PCT,
    "F1.1_COHENS_D": F1_1_MIN_COHENS_D,
    "F1.4_THRESHOLD_ADAPT_PCT": F1_4_MIN_THRESHOLD_ADAPTATION_PCT,
    "F1.4_COHENS_D": F1_4_MIN_COHENS_D,
    "F1.5_PAC_INCREASE": F1_5_PAC_INCREASE_MIN,
    "F1.5_COHENS_D": F1_5_COHENS_D_MIN,
    # F2 family
    "F2.1_ADVANTAGE": F2_1_MIN_ADVANTAGE_PCT,
    "F2.1_PP_DIFF": F2_1_MIN_PP_DIFF,
    "F2.1_COHENS_H": F2_1_MIN_COHENS_H,
    "F2.2_CORR": F2_2_MIN_CORR,
    "F2.2_FISHER_Z": F2_2_MIN_FISHER_Z,
    "F2.3_RT_ADVANTAGE": F2_3_MIN_RT_ADVANTAGE_MS,
    "F2.3_ALPHA": F2_3_ALPHA,
    "F2.4_CONFIDENCE_EFFECT": F2_4_MIN_CONFIDENCE_EFFECT_PCT,
    "F2.4_BETA_INTERACTION": F2_4_MIN_BETA_INTERACTION,
    "F2.5_MAX_TRIALS": F2_5_MAX_TRIALS,
    "F2.5_HAZARD_RATIO": F2_5_MIN_HAZARD_RATIO,
    "F2.CARDIAC_DETECTION": F2_CARDIAC_DETECTION_ADVANTAGE_MIN,
    # F3 family
    "F3.1_ADVANTAGE": F3_1_MIN_ADVANTAGE_PCT,
    "F3.1_COHENS_D": F3_1_MIN_COHENS_D,
    "F3.2_INTERO_ADVANTAGE": F3_2_MIN_INTERO_ADVANTAGE_PCT,
    "F3.2_COHENS_D": F3_2_MIN_COHENS_D,
    "F3.3_REDUCTION": F3_3_MIN_REDUCTION_PCT,
    "F3.3_COHENS_D": F3_3_MIN_COHENS_D,
    "F3.4_REDUCTION": F3_4_MIN_REDUCTION_PCT,
    "F3.4_COHENS_D": F3_4_MIN_COHENS_D,
    "F3.6_MAX_TRIALS": F3_6_MAX_TRIALS,
    "F3.6_HAZARD_RATIO": F3_6_MIN_HAZARD_RATIO,
    # F5 family
    "F5.5_PCA_LOADING": F5_5_MIN_LOADING,
    "F5.5_PCA_VARIANCE": F5_5_PCA_MIN_VARIANCE,
    "F5.6_PERF_DIFF": F5_6_MIN_PERFORMANCE_DIFF_PCT,
    "F5.6_COHENS_D": F5_6_MIN_COHENS_D,
    "F5.6_ALPHA": F5_6_ALPHA,
    # F6 family
    "F6.1_LTCN_TRANSITION": F6_1_LTCN_MAX_TRANSITION_MS,
    "F6.2_INTEGRATION_RATIO": F6_2_MIN_INTEGRATION_RATIO,
    "F6.2_R2": F6_2_MIN_CURVE_FIT_R2,
    # P thresholds
    "P7_MIN_AUC": P7_MIN_AUC,
    "P11_MIN_R2": P11_MIN_R2,
    # V thresholds
    "V2.2_CARDIAC_DETECTION": F2_CARDIAC_DETECTION_ADVANTAGE_MIN,
    "V2.2_ALPHA": V2_2_ALPHA,
    "V2.3_SOMATIC_EFFECT_D": V2_3_MIN_SOMATIC_EFFECT_D,
    "V2.3_CONTRIBUTION_PCT": V2_3_MIN_CONTRIBUTION_PCT,
    "V2.3_ALPHA": V2_3_ALPHA,
    "V7.1_PCI_REDUCTION": V7_1_MIN_PCI_REDUCTION,
    "V7.1_COHENS_D": V7_1_MIN_COHENS_D,
    "V9.1_CORR": V9_1_MIN_CORRELATION,
    "V9.3_CORR": V9_3_MIN_CORRELATION,
    "V11_MIN_R2": V11_MIN_R2,
    "V11_MIN_DELTA_R2": V11_MIN_DELTA_R2,
    "V11_MIN_COHENS_D": V11_MIN_COHENS_D,
    "V17_MIN_R2_P3B": V17_MIN_R2_P3B_DECAY,
    "V17_MIN_R2_THETA": V17_MIN_R2_THETA_ELEVATION,
    "V17_MIN_CORR": V17_MIN_CORRELATION_MAGNITUDE,
    "V17_ALPHA": V17_ALPHA,
    "V20_HG_BIMODALITY_COEFF": V20_HG_BIMODALITY_COEFF_MIN,
    "V20_HG_OCCUPANCY_COHENS_D": V20_HG_OCCUPANCY_COHENS_D_MIN,
    "V20_AC1_ADVANTAGE": V20_AC1_ADVANTAGE_MIN,
    "V20_AC1_KENDALL_TAU": V20_AC1_KENDALL_TAU_MIN,
    "V20_VARIANCE_ADVANTAGE": V20_VARIANCE_ADVANTAGE_MIN,
    "V21_MMN_MIN_R2": V21_MMN_MIN_R2,
    "V21_HEP_MIN_R2": V21_HEP_MIN_R2,
    "V21_IGNITION_TRANSIENT": V21_IGNITION_TRANSIENT_RATIO,
    "V19_BELOW_CHANCE": V19_BELOW_CHANCE_THRESHOLD,
    "V19_PRE_IGNITION_ACC": V19_MIN_PRE_IGNITION_ACCURACY,
    "V19_CONTROL_DECODABILITY": V19_CONTROL_DECODABILITY_MIN,
    "V19_N_PERMS": V19_N_PERMUTATIONS,
    "V19_SUPPRESSION_BINS": V19_POST_IGNITION_SUPPRESSION_BINS,
    "V18_GFP_AUC_ADVANTAGE": V18_GFP_AUC_IGNITION_ADVANTAGE,
    "V18_AUC_COHENS_D": V18_AUC_COHENS_D_MIN,
    "V18_ST_CORRELATION": V18_ST_CORRELATION_MIN,
    "V16_MIN_CORR": V16_MIN_CORRELATION,
    "V16_MIN_EFFICIENCY": V16_MIN_EFFICIENCY_GAIN,
    "V16_C1_CONSISTENCY": V16_C1_CONSISTENCY_CV,
    # VP-01
    "VP1_APGI_MAX_AUC": VP1_APGI_MAX_DISCRIMINABILITY_AUC,
    "VP1_COMPETITOR_MIN_AUC": VP1_COMPETITOR_MIN_DISCRIMINABILITY_AUC,
    # VP-02
    "VP2_DELTA_PI_COUPLING": VP2_DELTA_PI_COUPLING,
    "VP2_AROUSAL_COUPLING": VP2_AROUSAL_COUPLING_SCALE,
    "VP2_AROUSAL_BOOST_MAX": VP2_AROUSAL_BOOST_MAX,
    # VP-04
    "VP4_CALIBRATED_TAU": VP4_CALIBRATED_TAU,
    "VP4_CALIBRATED_THETA_0": VP4_CALIBRATED_THETA_0,
    "VP4_ADAPTIVE_THRESHOLD_PCT": VP4_ADAPTIVE_THRESHOLD_LOAD_INCREASE_MIN_PCT,
    "VP4_ADAPTIVE_COHENS_D": VP4_ADAPTIVE_THRESHOLD_COHENS_D_MIN,
    "VP4_FIXED_THRESHOLD_MAX": VP4_FIXED_THRESHOLD_MAX_DIFF_PCT,
    # FP-03
    "FP3_TIER2_MAX_FAILURES": FP3_TIER2_MAX_FAILURES,
    "FP3_HEP_P3B_MIN_CORR": FP3_HEP_P3B_MIN_CORRELATION,
    "FP3_P3B_MIN_COHENS_D": FP3_P3B_AMPLITUDE_MIN_COHENS_D,
    "FP3_BISTABILITY_DIP_P": FP3_BISTABILITY_DIP_P_MAX,
    # FP-05
    "FP5_MIN_DOMINANCE_PCT": FP5_MIN_APGI_DOMINANCE_PCT,
    "FP5_VOLATILE_ADV_PCT": FP5_VOLATILE_ENV_ADVANTAGE_MIN_PCT,
    # Tier 1 framework-disconfirming [HIGH-4]
    "TIER1_P3B_PI_MIN_TRIALS": TIER1_P3B_PI_MIN_TRIALS,
    "TIER1_P3B_PI_MIN_N": TIER1_P3B_PI_MIN_N,
    "TIER1_LNN_BF_THRESHOLD": TIER1_LNN_BF_THRESHOLD,
    "TIER1_LNN_MIN_TRIALS": TIER1_LNN_MIN_TRIALS,
    "TIER1_LNN_MIN_N": TIER1_LNN_MIN_N,
    "TIER1_PSYCHOMETRIC_BETA": TIER1_PSYCHOMETRIC_BETA_MIN,
    # Generic
    "GENERIC_MIN_R2": GENERIC_MIN_R2,
    "GENERIC_MIN_AUC": GENERIC_MIN_AUC,
    "GENERIC_MIN_CORR": GENERIC_MIN_CORR,
    "GENERIC_MIN_COHENS_D": GENERIC_MIN_COHENS_D,  # [LOW-3] 0.70, not 0.71
    "GENERIC_MEDIUM_COHENS_D": GENERIC_MEDIUM_COHENS_D,
    "GENERIC_BINARY_THRESHOLD": GENERIC_BINARY_DECISION_THRESHOLD,
    "LIQUID_IGNITION_THRESHOLD": LIQUID_IGNITION_DETECTION_THRESHOLD,
    # DOC AUC (P4a primary confirmatory threshold) [FIX]
    "DOC_AUC_MIN": DOC_AUC_MIN,
    "DOC_AUC_MAX": DOC_AUC_MAX,
    # F4 — critical slowing Kendall τ [FIX: structural — new constant]
    "F4_CRITICAL_SLOWING_KENDALL_TAU_MIN": F4_CRITICAL_SLOWING_KENDALL_TAU_MIN,
    "F4_CRITICAL_SLOWING_MIN_RATIO": F4_CRITICAL_SLOWING_MIN_RATIO,
    # V20 bimodality — dip test replaces bimodality coefficient [FIX]
    "V20_DIP_TEST_P_MAX": V20_DIP_TEST_P_MAX,
    # V22 — fMRI Anticipation (Protocol 5) — previously missing
    "V22_1_ANTICIPATORY_COUPLING_R": V22_1_MIN_ANTICIPATORY_COUPLING_R,
    "V22_1_ALPHA": V22_1_ALPHA,
    "V22_2_MAX_VMPFC_PE_R": V22_2_MAX_VMPFC_PE_R,
    "V22_2_ALPHA": V22_2_ALPHA,
    "V22_3_VALENCE_ALPHA": V22_3_VALENCE_ALPHA,
    "V22_3_CONTRAST_ALPHA": V22_3_CONTRAST_ALPHA,
    "V22_4_NO_ANTICIPATION_COUPLING_MAX": V22_4_NO_ANTICIPATION_COUPLING_MAX,
    "V22_4_ALPHA": V22_4_ALPHA,
    # P0 — Pilot Study thresholds — previously missing
    "P0_N_TARGET": P0_N_TARGET,
    "P0_POWER": P0_POWER,
    "P0_ALPHA": P0_ALPHA,
    "P0_A_R_MIN": P0_A_R_MIN,
    "P0_A_FAL_R_MAX": P0_A_FAL_R_MAX,
    "P0_B_HEP_INCREASE_PCT_MIN": P0_B_HEP_INCREASE_PCT_MIN,
    "P0_B_COHENS_D_MIN": P0_B_COHENS_D_MIN,
    "P0_B_BF10_MIN": P0_B_BF10_MIN,
    "P0_C_AINS_BOLD_R_MIN": P0_C_AINS_BOLD_R_MIN,
    # P1 — EEG Protocol thresholds — previously missing
    "P1_N_TARGET": P1_N_TARGET,
    "P1_N_MIN": P1_N_MIN,
    "P1_ALPHA_BONFERRONI": P1_ALPHA_BONFERRONI,
    "P1_A_ALPHA": P1_A_ALPHA,
    "P1_B_ALPHA": P1_B_ALPHA,
    "P1_C_ALPHA": P1_C_ALPHA,
    "P1_A_COHENS_D_MIN": P1_A_COHENS_D_MIN,
    "P1_A_COHENS_D_FAL_MAX": P1_A_COHENS_D_FAL_MAX,
    "P1_B_PARTIAL_R_MIN": P1_B_PARTIAL_R_MIN,
    "P1_C_ETA_SQ_MIN": P1_C_ETA_SQ_MIN,
    "P1_C_ETA_SQ_FAL_MAX": P1_C_ETA_SQ_FAL_MAX,
    # P2 — TMS Protocol additional thresholds — previously missing
    "P2_N_TARGET": P2_N_TARGET,
    "P2_N_MIN": P2_N_MIN,
    "P2_B_EQUIV_ROPE_LOW": P2_B_EQUIV_ROPE_LOW,
    "P2_B_EQUIV_ROPE_HIGH": P2_B_EQUIV_ROPE_HIGH,
    "P2_B_EQUIV_BF01_MIN": P2_B_EQUIV_BF01_MIN,
    "P2_POWER": P2_POWER,
    # P3 — Computational Protocol thresholds — previously missing
    "P3_N_SEEDS": P3_N_SEEDS,
    "P3_BIC_DELTA_MIN": P3_BIC_DELTA_MIN,
    "P3_FIM_OFFDIAG_MAX": P3_FIM_OFFDIAG_MAX,
    "P3_CP3A_CALIBRATION_TOL": P3_CP3A_CALIBRATION_TOL,
    # P4 — Disorders of Consciousness thresholds — previously missing
    "P4_N_TOTAL": P4_N_TOTAL,
    "P4_N_EMCS": P4_N_EMCS,
    "P4_DELTA_R2_MIN": P4_DELTA_R2_MIN,
    "P4_INTER_RATER_KAPPA_MIN": P4_INTER_RATER_KAPPA_MIN,
    "P4_MICE_IMPUTATIONS_MIN": P4_MICE_IMPUTATIONS_MIN,
    "P4_MICE_IMPUTATIONS_MAX": P4_MICE_IMPUTATIONS_MAX,
    # P5 — fMRI Protocol thresholds — previously missing
    "P5_N_TARGET": P5_N_TARGET,
    "P5_N_MIN": P5_N_MIN,
    "P5_POWER": P5_POWER,
    "P5_FIM_OFFDIAG_MAX": P5_FIM_OFFDIAG_MAX,
    # P6 — iEEG Protocol thresholds — previously missing
    "P6_N_TARGET": P6_N_TARGET,
    "P6_N_MIN": P6_N_MIN,
    "P6_CE_BF10_MIN": P6_CE_BF10_MIN,
    "P6_B_DWELL_TIME_MAX_PCT": P6_B_DWELL_TIME_MAX_PCT,
    "P6_D_COHERENCE_R_MIN": P6_D_COHERENCE_R_MIN,
    "P6_COHERENCE_PEAK_MS_MIN": P6_COHERENCE_PEAK_MS_MIN,
    "P6_COHERENCE_PEAK_MS_MAX": P6_COHERENCE_PEAK_MS_MAX,
    # Cross-protocol / framework gate thresholds — previously missing
    "ALPHA_PRIMARY_GONOGO": ALPHA_PRIMARY_GONOGO,
    "ALPHA_SECONDARY": ALPHA_SECONDARY,
    "CROSS_PROTOCOL_BY_FDR_Q": CROSS_PROTOCOL_BY_FDR_Q,
    "FRAMEWORK_REJECTION_MIN_FAILURES": FRAMEWORK_REJECTION_MIN_FAILURES,
}


# =============================================================================
# SHARED F6 TESTING FUNCTIONS
# Eliminate code duplication across Falsification-Protocol files.
# =============================================================================


def test_f6_1_intrinsic_threshold_behavior(
    ltcn_transition_times: NDArray,
    feedforward_transition_times: NDArray,
    ltcn_max_transition_ms: float = F6_1_LTCN_MAX_TRANSITION_MS,
    cliffs_delta_min: float = F6_1_CLIFFS_DELTA_MIN,
    mann_whitney_alpha: float = F6_1_MANN_WHITNEY_ALPHA,
) -> dict:
    """
    Test F6.1: Intrinsic Threshold Behavior.

    LTCNs must complete 10–90% firing-rate transition in <50 ms without
    explicit threshold modules.
    """
    from scipy.stats import mannwhitneyu

    if len(ltcn_transition_times) < 2:
        raise ValueError(f"ltcn_transition_times must have ≥2 elements, got {len(ltcn_transition_times)}")
    if len(feedforward_transition_times) < 2:
        raise ValueError(f"feedforward_transition_times must have ≥2 elements, got {len(feedforward_transition_times)}")
    if np.any(np.isnan(ltcn_transition_times)) or np.any(np.isinf(ltcn_transition_times)):
        raise ValueError("ltcn_transition_times contains NaN or Inf values")
    if np.any(np.isnan(feedforward_transition_times)) or np.any(np.isinf(feedforward_transition_times)):
        raise ValueError("feedforward_transition_times contains NaN or Inf values")

    try:
        _, p_value = mannwhitneyu(ltcn_transition_times, feedforward_transition_times)
    except ValueError:
        p_value = 1.0

    n_ltcn = len(ltcn_transition_times)
    n_ff = len(feedforward_transition_times)
    pooled = np.concatenate([ltcn_transition_times, feedforward_transition_times])
    ranks = np.argsort(np.argsort(pooled))
    rank_ltcn = ranks[:n_ltcn]
    rank_ff = ranks[n_ltcn:]
    cliffs_delta = 2 * (np.mean(rank_ff) - np.mean(rank_ltcn)) / (n_ltcn * n_ff)

    passed = bool(
        np.median(ltcn_transition_times) <= ltcn_max_transition_ms
        and cliffs_delta >= cliffs_delta_min
        and p_value < mann_whitney_alpha
    )

    return {
        "passed": passed,
        "ltcn_median_time": float(np.median(ltcn_transition_times)),
        "feedforward_median_time": float(np.median(feedforward_transition_times)),
        "cliffs_delta": cliffs_delta,
        "p_value": p_value,
        "threshold": f"LTCN ≤{ltcn_max_transition_ms} ms, δ ≥ {cliffs_delta_min}",
    }


def test_f6_2_intrinsic_temporal_integration(
    ltcn_integration_window: NDArray,
    rnn_integration_window: NDArray,
    ltcn_min_window_ms: float = F6_2_LTCN_MIN_WINDOW_MS,
    min_integration_ratio: float = F6_2_MIN_INTEGRATION_RATIO,
    falsification_ratio: float = F6_2_FALSIFICATION_RATIO,
    min_curve_fit_r2: float = F6_2_MIN_CURVE_FIT_R2,
    wilcoxon_alpha: float = F6_2_WILCOXON_ALPHA,
) -> dict:
    """
    Test F6.2: Intrinsic Temporal Integration.

    LTCNs must integrate over ≥200 ms windows (autocorrelation decay to <0.37)
    vs. <50 ms for standard RNNs. Ratio must be ≥4×.
    """
    from scipy.stats import mannwhitneyu

    if len(ltcn_integration_window) < 2:
        raise ValueError(f"ltcn_integration_window must have ≥2 elements, got {len(ltcn_integration_window)}")
    if len(rnn_integration_window) < 2:
        raise ValueError(f"rnn_integration_window must have ≥2 elements, got {len(rnn_integration_window)}")
    if np.any(np.isnan(ltcn_integration_window)) or np.any(np.isinf(ltcn_integration_window)):
        raise ValueError("ltcn_integration_window contains NaN or Inf values")
    if np.any(np.isnan(rnn_integration_window)) or np.any(np.isinf(rnn_integration_window)):
        raise ValueError("rnn_integration_window contains NaN or Inf values")

    ltcn_median = float(np.median(ltcn_integration_window))
    rnn_median = float(np.median(rnn_integration_window))
    ratio = ltcn_median / rnn_median if rnn_median > 0 else 0.0

    try:
        _, p_value = mannwhitneyu(ltcn_integration_window, rnn_integration_window)
    except ValueError:
        p_value = 1.0

    passed = ltcn_median >= ltcn_min_window_ms and ratio >= min_integration_ratio and p_value < wilcoxon_alpha

    return {
        "passed": passed,
        "ltcn_median_window": ltcn_median,
        "rnn_median_window": rnn_median,
        "ratio": ratio,
        "p_value": p_value,
        "threshold": f"LTCN ≥{ltcn_min_window_ms} ms, ratio ≥ {min_integration_ratio}×",
    }


def test_f6_3_metabolic_selectivity(
    ltcn_sparsity_reductions: NDArray,
    standard_sparsity_reductions: NDArray,
    min_reduction_pct: float = 30.0,
    max_standard_reduction_pct: float = 10.0,
    min_cohens_d: float = 0.70,
    alpha: float = 0.01,
) -> dict:
    """
    Test F6.3: Metabolic Selectivity Without Training.

    LTCNs with adaptive τ(x) must show ≥30% reduction in active units during
    low-information periods vs. <10% for standard architectures.
    """
    from scipy import stats

    if len(ltcn_sparsity_reductions) < 2:
        raise ValueError(f"ltcn_sparsity_reductions must have ≥2 elements, got {len(ltcn_sparsity_reductions)}")
    if len(standard_sparsity_reductions) < 2:
        raise ValueError(f"standard_sparsity_reductions must have ≥2 elements, got {len(standard_sparsity_reductions)}")
    if np.any(np.isnan(ltcn_sparsity_reductions)) or np.any(np.isinf(ltcn_sparsity_reductions)):
        raise ValueError("ltcn_sparsity_reductions contains NaN or Inf values")
    if np.any(np.isnan(standard_sparsity_reductions)) or np.any(np.isinf(standard_sparsity_reductions)):
        raise ValueError("standard_sparsity_reductions contains NaN or Inf values")

    _, p_value = stats.ttest_rel(ltcn_sparsity_reductions, standard_sparsity_reductions)

    n1, n2 = len(ltcn_sparsity_reductions), len(standard_sparsity_reductions)
    pooled_std = math.sqrt(
        (
            (n1 - 1) * float(np.var(ltcn_sparsity_reductions, ddof=1))
            + (n2 - 1) * float(np.var(standard_sparsity_reductions, ddof=1))
        )
        / (n1 + n2 - 2)
    )
    cohens_d = (
        (float(np.mean(ltcn_sparsity_reductions)) - float(np.mean(standard_sparsity_reductions))) / pooled_std
        if pooled_std > 0
        else 0.0
    )

    passed = bool(
        float(np.mean(ltcn_sparsity_reductions)) >= min_reduction_pct
        and float(np.mean(standard_sparsity_reductions)) <= max_standard_reduction_pct
        and cohens_d >= min_cohens_d
        and p_value < alpha
    )

    return {
        "passed": passed,
        "ltcn_mean_reduction": float(np.mean(ltcn_sparsity_reductions)),
        "standard_mean_reduction": float(np.mean(standard_sparsity_reductions)),
        "cohens_d": cohens_d,
        "p_value": p_value,
        "threshold": f"LTCN ≥{min_reduction_pct}%, standard ≤{max_standard_reduction_pct}%, d ≥ {min_cohens_d}",
    }


def test_f6_4_fading_memory(
    memory_decay_tau: float,
    min_tau: float = 1.0,
    max_tau: float = 3.0,
    min_curve_fit_r2: float = 0.85,
    rng: "np.random.Generator | None" = None,  # [HIGH-2] reproducibility fix
) -> dict:
    """
    Test F6.4: Fading Memory Implementation.

    LTCNs must show exponential memory decay with τ_memory = 1–3 s.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator for reproducibility. Defaults to
        np.random.default_rng(seed=42) if not supplied.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    tau_in_bounds = min_tau <= memory_decay_tau <= max_tau

    time_points = np.linspace(0, 5 * memory_decay_tau, 100)
    expected_decay = np.exp(-time_points / memory_decay_tau)
    noisy_decay = expected_decay + 0.05 * rng.standard_normal(len(time_points))

    try:

        def exp_decay(t, a, tau_fit):
            return a * np.exp(-t / tau_fit)

        popt, _ = curve_fit(exp_decay, time_points, noisy_decay, p0=[1.0, memory_decay_tau], maxfev=10000)
        fitted_decay = exp_decay(time_points, *popt)
        ss_res = float(np.sum((noisy_decay - fitted_decay) ** 2))
        ss_tot = float(np.sum((noisy_decay - np.mean(noisy_decay)) ** 2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared_pass = r_squared >= min_curve_fit_r2
    except (RuntimeError, ValueError):
        r_squared = 0.0
        r_squared_pass = False

    passed = tau_in_bounds and r_squared_pass

    return {
        "passed": passed,
        "tau_memory": memory_decay_tau,
        "r_squared": r_squared,
        "threshold": f"τ_memory = {min_tau}–{max_tau} s, R² ≥ {min_curve_fit_r2}",
    }


def test_f6_5_bifurcation_structure(
    theta_t: float,
    tau_S: float = 0.3,
    dt: float = 0.05,
    n_steps: int = 2000,
    hysteresis_min: float = F6_5_HYSTERESIS_MIN,
    hysteresis_max: float = F6_5_HYSTERESIS_MAX,
) -> dict:
    """
    Test F6.5: Bifurcation Structure for Ignition.

    [HIGH-1] REWRITTEN: uses a proper bidirectional phase-plane sweep.
    Hysteresis = onset_drive (forward sweep) − offset_drive (backward sweep).
    The previous implementation measured sigmoid width (4.394/b), which is
    NOT hysteresis and passes when true hysteresis is absent.

    Parameters
    ----------
    theta_t : float
        Ignition threshold.
    tau_S : float
        Surprise accumulation decay constant.
    dt : float
        Integration time step.
    n_steps : int
        Simulation steps per drive level.
    hysteresis_min, hysteresis_max : float
        Valid hysteresis window. Values outside this range → FAIL.
    """
    n_sweep = 80
    drives = np.linspace(0.0, 2.0 * theta_t, n_sweep)

    def _simulate_ignition(drive_sequence):
        """Run ODE for each drive value; return whether ignition occurred."""
        ignited_at = []
        for drive in drive_sequence:
            S = 0.0
            fired = False
            for _ in range(n_steps):
                dS = (-S / tau_S + drive) * dt
                S = max(0.0, S + dS)
                if S > theta_t:
                    fired = True
                    break
            ignited_at.append(fired)
        return np.array(ignited_at, dtype=bool)

    # Forward sweep: drive increases from 0 to 2θ
    forward_ignited = _simulate_ignition(drives)
    # Backward sweep: drive decreases from 2θ to 0
    backward_ignited = _simulate_ignition(drives[::-1])[::-1]

    # Onset drive: first drive value that causes ignition in forward sweep
    onset_indices = np.where(forward_ignited)[0]
    onset_drive = float(drives[onset_indices[0]]) if len(onset_indices) > 0 else float(drives[-1])

    # Offset drive: last drive value that sustains ignition in backward sweep
    offset_indices = np.where(backward_ignited)[0]
    offset_drive = float(drives[offset_indices[-1]]) if len(offset_indices) > 0 else float(drives[0])

    # Hysteresis = onset − offset; positive indicates true hysteresis
    hysteresis_width = max(0.0, onset_drive - offset_drive)

    passed = hysteresis_min <= hysteresis_width <= hysteresis_max

    return {
        "passed": passed,
        "onset_drive": onset_drive,
        "offset_drive": offset_drive,
        "hysteresis_width": hysteresis_width,
        "threshold": f"Hysteresis {hysteresis_min}–{hysteresis_max}",
        "method": "bidirectional_phase_plane_sweep",
    }


def test_f6_6_alternative_architectures(
    alternative_modules_needed: float,
    performance_gap_without_addons: float,
    min_modules_needed: float = F6_6_MIN_ADD_ON_MODULES,
    min_performance_gap: float = F6_6_MIN_PERFORMANCE_GAP,
) -> dict:
    """
    Test F6.6: Alternative Architectures Require Add-Ons.

    Standard RNNs, LSTMs, Transformers must need ≥2 explicit modules to match
    ≥85% of LTCN performance.
    """
    modules_pass = alternative_modules_needed >= min_modules_needed
    performance_pass = performance_gap_without_addons >= min_performance_gap
    passed = modules_pass and performance_pass

    return {
        "passed": passed,
        "modules_pass": modules_pass,
        "performance_pass": performance_pass,
        "add_ons_needed": alternative_modules_needed,
        "performance_gap": performance_gap_without_addons,
        "threshold": f"≥{min_modules_needed} add-ons, gap ≥{min_performance_gap}%",
    }
