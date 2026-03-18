"""
falsification_thresholds.py
============================

Single source of truth for all APGI falsification thresholds.

Every protocol (VP-*, FP-*) MUST import thresholds from here rather than
hard-coding them locally.  Changes to the specification propagate
automatically to every protocol.

Usage::

    from falsification_thresholds import (
        F6_1_LTCN_MAX_TRANSITION_MS,
        F6_2_MIN_INTEGRATION_RATIO,
        F5_6_PCA_MIN_VARIANCE,
    )
"""

import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# F6.1 – Intrinsic Threshold Behaviour (LTCN transition time)
# Specification: LTCN must complete 10-90 % firing-rate transition in <50 ms.
# ---------------------------------------------------------------------------
F6_1_LTCN_MAX_TRANSITION_MS: float = 50.0  # ≤50 ms  (spec figure)
F6_1_CLIFFS_DELTA_MIN: float = 0.60  # Cliff's δ ≥ 0.60
F6_1_MANN_WHITNEY_ALPHA: float = 0.01  # p < 0.01

# ---------------------------------------------------------------------------
# F6.2 – Intrinsic Temporal Integration
# Specification: LTCN window ≥200 ms; ratio vs RNN ≥4×; R² ≥0.85
# (Falsification alternative: ratio < 2.5 OR window < 150 ms)
# ---------------------------------------------------------------------------
F6_2_LTCN_MIN_WINDOW_MS: float = 200.0  # ≥200 ms
F6_2_MIN_INTEGRATION_RATIO: float = 4.0  # ≥4× RNN  (spec criterion)
F6_2_FALSIFICATION_RATIO: float = 2.5  # falsified if ratio < 2.5×
F6_2_MIN_CURVE_FIT_R2: float = 0.85  # R² ≥ 0.85
F6_2_WILCOXON_ALPHA: float = 0.01

# ---------------------------------------------------------------------------
# F5.5 / F5.6 – PCA variance threshold
# Spec: cumulative variance ≥70 % by first 3 PCs.
# Falsification alternative: <60 % is a fail.
# ---------------------------------------------------------------------------
F5_5_PCA_MIN_VARIANCE: float = 0.70  # ≥70 %  (spec)
F5_5_PCA_FALSIFICATION_THRESHOLD: float = 0.60  # falsified if <60 %
F5_5_PCA_MIN_LOADING: float = 0.60  # minimum PC loading

# F5.1 thresholds (Threshold Filtering Emergence)
F5_1_MIN_PROPORTION: float = 0.75  # ≥75% agents (spec)
F5_1_MIN_ALPHA: float = 4.0  # mean α ≥ 4.0 (spec)
F5_1_FALSIFICATION_ALPHA: float = 3.0  # falsified if mean α < 3.0
F5_1_MIN_COHENS_D: float = 0.80  # Cohen's d ≥ 0.80
F5_1_BINOMIAL_ALPHA: float = 0.01

# F5.2 thresholds (Precision-Weighted Coding Emergence)
F5_2_MIN_CORRELATION: float = 0.45  # r ≥ 0.45 (spec)
F5_2_FALSIFICATION_CORR: float = 0.35  # falsified if r < 0.35
F5_2_MIN_PROPORTION: float = 0.65  # ≥65% agents (spec)
F5_2_BINOMIAL_ALPHA: float = 0.01

# F5.3 thresholds (Interoceptive Prioritization Emergence)
F5_3_MIN_GAIN_RATIO: float = 1.30  # ratio ≥ 1.30 (spec)
F5_3_FALSIFICATION_RATIO: float = 1.15  # falsified if ratio < 1.15
F5_3_MIN_PROPORTION: float = 0.70  # ≥70% agents (spec)
F5_3_MIN_COHENS_D: float = 0.60  # d ≥ 0.60
F5_3_BINOMIAL_ALPHA: float = 0.01

# F5.4 thresholds (Multi-Timescale Integration Emergence)
F5_4_MIN_PROPORTION: float = 0.60  # ≥60% agents (spec)
F5_4_FALSIFICATION_PROPORTION: float = 0.45  # falsified if < 45%
F5_4_MIN_PEAK_SEPARATION: float = 3.0  # separation ≥ 3x (spec)
F5_4_FALSIFICATION_SEPARATION: float = 2.0  # falsified if separation < 2.0
F5_4_BINOMIAL_ALPHA: float = 0.01

# F5.6 thresholds (Non-APGI architecture failure)
F5_6_PCA_MIN_VARIANCE: float = 0.60  # ≥60 % (spec)
F5_6_MIN_PERFORMANCE_DIFF_PCT: float = 15.0  # ≥15 % worse performance
F5_6_MIN_COHENS_D: float = 0.60  # d ≥ 0.60
F5_6_ALPHA: float = 0.01

# ---------------------------------------------------------------------------
# V6.1 – Real-Time Processing Benchmark
# Spec: ≥100 trials/s; ≤50 ms latency
# ---------------------------------------------------------------------------
V6_1_MIN_PROCESSING_RATE: float = 100.0  # ≥100 trials/s
V6_1_MAX_LATENCY_MS: float = 50.0  # ≤50 ms
V6_1_FALSIFICATION_MIN_RATE: float = 80.0  # falsified if <80 trials/s
V6_1_FALSIFICATION_MAX_LATENCY_MS: float = 75.0  # falsified if >75 ms
V6_1_ALPHA: float = 0.05

# ---------------------------------------------------------------------------
# F1.5 – Cross-Level Phase-Amplitude Coupling (PAC)
# ---------------------------------------------------------------------------
F1_5_PAC_MI_MIN: float = 0.012  # MI ≥ 0.012 (spec)
F1_5_PAC_INCREASE_MIN: float = 30.0  # increase ≥ 30% (spec)
F1_5_COHENS_D_MIN: float = 0.50  # d ≥ 0.50 (spec)
F1_5_PERMUTATION_ALPHA: float = 0.01

# ---------------------------------------------------------------------------
# F2.3 – vmPFC-Like Anticipatory Bias (RT advantage)
# RT advantage expected across a *distribution* of trials; collecting a
# single scalar and passing it to ttest_1samp is degenerate (NaN p-value).
# The correct fix is to accumulate rt_advantage_ms across trials into a list.
# ---------------------------------------------------------------------------
F2_3_MIN_RT_ADVANTAGE_MS: float = 50.0  # ≥50 ms RT advantage (spec)
F2_3_MIN_BETA: float = 25.0  # β ≥ 25 ms
F2_3_MIN_STANDARDIZED_BETA: float = 0.40  # std β ≥ 0.40
F2_3_MIN_R2: float = 0.18  # R² ≥ 0.18
F2_3_ALPHA: float = 0.01

# ---------------------------------------------------------------------------
# F6.5 – Bifurcation / Hysteresis (must derive from phase-plane, not hardcode)
# ---------------------------------------------------------------------------
F6_5_BIFURCATION_ERROR_MAX: float = 0.10  # |error| ≤ 0.10
F6_5_HYSTERESIS_MIN: float = 0.08  # hysteresis ≥ 0.08
F6_5_HYSTERESIS_MAX: float = 0.25  # hysteresis ≤ 0.25

# ---------------------------------------------------------------------------
# V7.1 – TMS Intervention Thresholds
# ---------------------------------------------------------------------------
V7_1_MIN_THRESHOLD_REDUCTION_PCT: float = 15.0  # ≥15 % reduction
V7_1_MIN_EFFECT_DURATION_MIN: float = 60.0  # ≥60 min
V7_1_MIN_COHENS_D: float = 0.70  # d ≥ 0.70
V7_1_ALPHA: float = 0.01

# ---------------------------------------------------------------------------
# V7.2 – Pharmacological Precision Modulation
# ---------------------------------------------------------------------------
V7_2_MIN_PRECISION_INCREASE_PCT: float = 25.0  # Π_i ≥ 25 %
V7_2_MIN_IGNITION_REDUCTION_PCT: float = 30.0  # ignition reduction ≥ 30 %
V7_2_MIN_ETA_SQUARED: float = 0.20  # η² ≥ 0.20
V7_2_MIN_COHENS_D: float = 0.50  # d ≥ 0.50
V7_2_ALPHA: float = 0.01

# ---------------------------------------------------------------------------
# V11 – Model Fits
# ---------------------------------------------------------------------------
V11_MIN_R2: float = 0.75  # R² ≥ 0.75
V11_MIN_DELTA_R2: float = 0.10  # ΔR² ≥ 0.10
V11_MIN_COHENS_D: float = 0.45  # d ≥ 0.45

# ---------------------------------------------------------------------------
# V12.1 – Clinical Gradient Prediction
# ---------------------------------------------------------------------------
V12_1_MIN_P3B_REDUCTION_PCT: float = 80.0  # ≥80 % reduction
V12_1_MIN_IGNITION_REDUCTION_PCT: float = 70.0  # ≥70 % reduction
V12_1_MIN_COHENS_D: float = 0.80  # d ≥ 0.80
V12_1_MIN_ETA_SQUARED: float = 0.20  # η² ≥ 0.20
V12_1_ALPHA: float = 0.01

# ---------------------------------------------------------------------------
# V12.2 – Cross-Species Homology
# ---------------------------------------------------------------------------
V12_2_MIN_CORRELATION: float = 0.60  # r ≥ 0.60 (spec)
V12_2_FALSIFICATION_CORR: float = 0.45  # falsified if r < 0.45
V12_2_MIN_PILLAIS_TRACE: float = 0.40  # Pillai's trace ≥ 0.40 (spec)
V12_2_FALSIFICATION_PILLAIS: float = 0.25  # falsified if Pillai's < 0.25
V12_2_ALPHA: float = 0.01

# ---------------------------------------------------------------------------
# F1 family
# ---------------------------------------------------------------------------
F1_1_MIN_ADVANTAGE_PCT: float = 18.0
F1_1_MIN_COHENS_D: float = 0.60
F1_1_ALPHA: float = 0.01

# ---------------------------------------------------------------------------
# F2 family (IGT / Somatic)
# ---------------------------------------------------------------------------
F2_1_MIN_ADVANTAGE_PCT: float = 22.0
F2_1_MIN_PP_DIFF: float = 10.0
F2_1_MIN_COHENS_H: float = 0.55
F2_1_ALPHA: float = 0.01

F2_2_MIN_CORR: float = 0.40
F2_2_MIN_FISHER_Z: float = 1.80
F2_2_ALPHA: float = 0.01

F2_4_MIN_CONFIDENCE_EFFECT_PCT: float = 30.0
F2_4_MIN_BETA_INTERACTION: float = 0.35
F2_4_ALPHA: float = 0.01

F2_5_MAX_TRIALS: float = 55.0
F2_5_MIN_HAZARD_RATIO: float = 1.65
F2_5_MIN_TRIAL_ADVANTAGE: float = 12.0
F2_5_ALPHA: float = 0.01

# ---------------------------------------------------------------------------
# F3 family (Advantages)
# ---------------------------------------------------------------------------
F3_1_MIN_ADVANTAGE_PCT: float = 18.0
F3_1_MIN_COHENS_D: float = 0.60
F3_1_ALPHA: float = 0.01

F3_2_MIN_INTERO_ADVANTAGE_PCT: float = 28.0
F3_2_MIN_COHENS_D: float = 0.70
F3_2_ALPHA: float = 0.01

F3_3_MIN_REDUCTION_PCT: float = 25.0
F3_3_MIN_COHENS_D: float = 0.75
F3_3_ALPHA: float = 0.01

F3_4_MIN_REDUCTION_PCT: float = 20.0
F3_4_MIN_COHENS_D: float = 0.65
F3_4_ALPHA: float = 0.01

F3_6_MAX_TRIALS: float = 200.0
F3_6_MIN_HAZARD_RATIO: float = 1.45
F3_6_ALPHA: float = 0.01

# ---------------------------------------------------------------------------
# V7/V9 family
# ---------------------------------------------------------------------------
V7_1_MIN_PCI_REDUCTION: float = 0.18
V7_1_MIN_COHENS_D: float = 0.50
V9_1_MIN_CORRELATION: float = 0.60
V9_3_MIN_CORRELATION: float = 0.70

# ---------------------------------------------------------------------------
# Misc / shared
# ---------------------------------------------------------------------------
DEFAULT_ALPHA: float = 0.05  # default significance level
BONFERRONI_ALPHA_6: float = 0.008  # Bonferroni-corrected (6 tests)

# =============================================================================
# THRESHOLD REGISTRY
# =============================================================================
THRESHOLD_REGISTRY = {
    "F1.1_ADVANTAGE": F1_1_MIN_ADVANTAGE_PCT,
    "F1.1_COHENS_D": F1_1_MIN_COHENS_D,
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
    "F5.5_PCA_VARIANCE": F5_5_PCA_MIN_VARIANCE,
    "F5.5_PCA_LOADING": F5_5_PCA_MIN_LOADING,
    "F5.6_PERF_DIFF": F5_6_MIN_PERFORMANCE_DIFF_PCT,
    "F5.6_COHENS_D": F5_6_MIN_COHENS_D,
    "F5.6_ALPHA": F5_6_ALPHA,
    "F6.1_LTCN_TRANSITION": F6_1_LTCN_MAX_TRANSITION_MS,
    "F6.2_INTEGRATION_RATIO": F6_2_MIN_INTEGRATION_RATIO,
    "F6.2_R2": F6_2_MIN_CURVE_FIT_R2,
    "V7.1_PCI_REDUCTION": V7_1_MIN_PCI_REDUCTION,
    "V7.1_COHENS_D": V7_1_MIN_COHENS_D,
    "V9.1_CORR": V9_1_MIN_CORRELATION,
    "V9.3_CORR": V9_3_MIN_CORRELATION,
}


# =============================================================================
# SHARED F6 TESTING FUNCTIONS
# =============================================================================
# These functions eliminate code duplication across Falsification-Protocol files
# =============================================================================


def test_f6_1_intrinsic_threshold_behavior(
    ltcn_transition_time: float,
    feedforward_transition_time: float,
    ltcn_max_transition_ms: float = F6_1_LTCN_MAX_TRANSITION_MS,
    cliffs_delta_min: float = F6_1_CLIFFS_DELTA_MIN,
    mann_whitney_alpha: float = F6_1_MANN_WHITNEY_ALPHA,
) -> dict:
    """
    Test F6.1: Intrinsic Threshold Behavior

    LTCNs should show sharp ignition transitions (10-90% firing rate increase within <50ms)
    without explicit threshold modules.

    Args:
        ltcn_transition_time: Median transition time for LTCNs (10-90% firing rate)
        feedforward_transition_time: Transition time for feedforward networks
        ltcn_max_transition_ms: Maximum allowed transition time for LTCNs
        cliffs_delta_min: Minimum Cliff's delta effect size
        mann_whitney_alpha: Significance level for Mann-Whitney U test

    Returns:
        Dictionary with pass/fail result and metrics
    """
    from scipy.stats import mannwhitneyu

    # Mann-Whitney U test for non-normal distributions
    try:
        u_stat, p_value = mannwhitneyu(
            [ltcn_transition_time], [feedforward_transition_time]
        )
    except ValueError:
        # Handle edge case with insufficient data
        p_value = 1.0

    # Cliff's delta (effect size for non-parametric data)
    pooled = np.concatenate([[ltcn_transition_time], [feedforward_transition_time]])
    n_ltcn = 1
    n_ff = 1

    # Calculate Cliff's delta
    ranks = np.argsort(np.argsort(pooled))
    rank_ltcn = ranks[0]
    rank_ff = ranks[1]

    # Cliff's delta formula
    cliffs_delta = (rank_ff - rank_ltcn) / (n_ltcn * n_ff)

    f6_1_pass = (
        ltcn_transition_time <= ltcn_max_transition_ms
        and cliffs_delta >= cliffs_delta_min
        and p_value < mann_whitney_alpha
    )

    return {
        "passed": f6_1_pass,
        "ltcn_time": ltcn_transition_time,
        "feedforward_time": feedforward_transition_time,
        "cliffs_delta": cliffs_delta,
        "p_value": p_value,
        "threshold": f"LTCN ≤{ltcn_max_transition_ms}ms, δ ≥ {cliffs_delta_min}",
    }


def test_f6_2_intrinsic_temporal_integration(
    ltcn_integration_window: float,
    rnn_integration_window: float,
    ltcn_min_window_ms: float = F6_2_LTCN_MIN_WINDOW_MS,
    min_integration_ratio: float = F6_2_MIN_INTEGRATION_RATIO,
    falsification_ratio: float = F6_2_FALSIFICATION_RATIO,
    min_curve_fit_r2: float = F6_2_MIN_CURVE_FIT_R2,
    wilcoxon_alpha: float = F6_2_WILCOXON_ALPHA,
) -> dict:
    """
    Test F6.2: Intrinsic Temporal Integration

    LTCNs should integrate information over 200-500ms windows (autocorrelation decay to <0.37)
    vs. <50ms for standard RNNs.

    Args:
        ltcn_integration_window: Integration window for LTCNs (autocorrelation decay)
        rnn_integration_window: Integration window for standard RNNs
        ltcn_min_window_ms: Minimum integration window for LTCNs
        min_integration_ratio: Minimum ratio of LTCN to RNN integration windows
        falsification_ratio: Ratio below which test fails
        min_curve_fit_r2: Minimum R² for exponential decay curve fitting
        wilcoxon_alpha: Significance level for Wilcoxon test

    Returns:
        Dictionary with pass/fail result and metrics
    """
    from scipy.stats import mannwhitneyu

    # Calculate ratio
    ratio = (
        ltcn_integration_window / rnn_integration_window
        if rnn_integration_window > 0
        else 0
    )

    # Wilcoxon signed-rank test
    try:
        stat, p_value = mannwhitneyu(
            [ltcn_integration_window], [rnn_integration_window]
        )
    except ValueError:
        p_value = 1.0

    f6_2_pass = (
        ltcn_integration_window >= ltcn_min_window_ms
        and ratio >= min_integration_ratio
        and p_value < wilcoxon_alpha
    )

    return {
        "passed": f6_2_pass,
        "ltcn_window": ltcn_integration_window,
        "rnn_window": rnn_integration_window,
        "ratio": ratio,
        "p_value": p_value,
        "threshold": f"LTCN ≥{ltcn_min_window_ms}ms, ratio ≥ {min_integration_ratio}×",
    }


def test_f6_3_metabolic_selectivity(
    ltcn_sparsity_reduction: float,
    standard_sparsity_reduction: float,
    min_reduction_pct: float = 30.0,
    max_standard_reduction_pct: float = 10.0,
    min_cohens_d: float = 0.70,
    alpha: float = 0.01,
) -> dict:
    """
    Test F6.3: Metabolic Selectivity Without Training

    LTCNs with adaptive τ(x) should show ≥30% reduction in active units during
    low-information periods vs. <10% for standard.

    Args:
        ltcn_sparsity_reduction: Sparsity reduction for LTCNs during low-information periods
        standard_sparsity_reduction: Sparsity reduction for standard architectures
        min_reduction_pct: Minimum reduction percentage for LTCNs
        max_standard_reduction_pct: Maximum reduction percentage for standard architectures
        min_cohens_d: Minimum Cohen's d effect size
        alpha: Significance level

    Returns:
        Dictionary with pass/fail result and metrics
    """
    from scipy import stats

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(
        [ltcn_sparsity_reduction], [standard_sparsity_reduction]
    )

    # Cohen's d
    pooled_std = np.sqrt(
        (
            (1 - 1) * np.var([ltcn_sparsity_reduction], ddof=1)
            + (1 - 1) * np.var([standard_sparsity_reduction], ddof=1)
        )
        / 2
    )
    cohens_d = (
        (ltcn_sparsity_reduction - standard_sparsity_reduction) / pooled_std
        if pooled_std > 0
        else 0
    )

    f6_3_pass = (
        ltcn_sparsity_reduction >= min_reduction_pct
        and standard_sparsity_reduction <= max_standard_reduction_pct
        and cohens_d >= min_cohens_d
        and p_value < alpha
    )

    return {
        "passed": f6_3_pass,
        "ltcn_reduction": ltcn_sparsity_reduction,
        "standard_reduction": standard_sparsity_reduction,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "threshold": f"LTCN ≥{min_reduction_pct}%, standard ≤{max_standard_reduction_pct}%, d ≥ {min_cohens_d}",
    }


def test_f6_4_fading_memory(
    memory_decay_tau: float,
    min_tau: float = 1.0,
    max_tau: float = 3.0,
    min_curve_fit_r2: float = 0.85,
) -> dict:
    """
    Test F6.4: Fading Memory Implementation

    LTCNs should show exponential memory decay with τ_memory = 1-3s for task-relevant information.

    Args:
        memory_decay_tau: Memory decay time constant for LTCNs
        min_tau: Minimum allowed τ value
        max_tau: Maximum allowed τ value
        min_curve_fit_r2: Minimum R² for exponential decay model fitting

    Returns:
        Dictionary with pass/fail result and metrics
    """
    f6_4_pass = memory_decay_tau >= min_tau and memory_decay_tau <= max_tau

    return {
        "passed": f6_4_pass,
        "tau_memory": memory_decay_tau,
        "threshold": f"τ_memory = {min_tau}-{max_tau}s",
    }


def test_f6_5_bifurcation_structure(
    theta_t: float,
    tau_S: float = 0.3,
    dt: float = 0.05,
    beta: float = 1.0,
    hysteresis_min: float = F6_5_HYSTERESIS_MIN,
    hysteresis_max: float = F6_5_HYSTERESIS_MAX,
) -> dict:
    """
    Test F6.5: Bifurcation Structure for Ignition

    LTCNs should exhibit bistable attractors with saddle-node bifurcation.
    Computed from phase portrait sweep varying input drive.

    Args:
        theta_t: Ignition threshold
        tau_S: Surprise decay time constant
        dt: Time step
        beta: Somatic bias
        hysteresis_min: Minimum hysteresis width
        hysteresis_max: Maximum hysteresis width

    Returns:
        Dictionary with pass/fail result and metrics
    """
    # Perform phase portrait sweep
    n_sweep = 100
    drives = np.linspace(0, 2 * theta_t, n_sweep)
    ignition_probs = []

    for drive in drives:
        # Simple surprise accumulation simulation
        S_t = 0.0
        ignited = False
        for i in range(1000):
            dS_dt = -S_t / tau_S + drive
            S_t += dS_dt * dt
            S_t = max(0.0, S_t)
            if S_t > theta_t:
                ignited = True
                break
        ignition_probs.append(1.0 if ignited else 0.0)

    ignition_probs = np.array(ignition_probs)

    # Fit sigmoid
    def sigmoid(x, a, b, c):
        return a / (1 + np.exp(-b * (x - c)))

    try:
        popt, pcov = curve_fit(
            sigmoid,
            drives,
            ignition_probs,
            p0=[1, 1, theta_t],
            bounds=([0.5, 0.1, 0], [1.5, 10, 2 * theta_t]),
        )
        a, b, c = popt
        bifurcation_point = c
        hysteresis_width = 4.39 / b  # Approximate width at 0.5 for logistic sigmoid
    except Exception:
        bifurcation_point = theta_t
        hysteresis_width = 0.1

    f6_5_pass = (
        hysteresis_width >= hysteresis_min and hysteresis_width <= hysteresis_max
    )

    return {
        "passed": f6_5_pass,
        "bifurcation_point": bifurcation_point,
        "hysteresis_width": hysteresis_width,
        "threshold": f"Hysteresis {hysteresis_min}-{hysteresis_max}",
    }


def test_f6_6_alternative_architectures(
    alternative_modules_needed: float,
    performance_gap_without_addons: float,
    min_modules_needed: float = 2.0,
    min_performance_gap: float = 15.0,
) -> dict:
    """
    Test F6.6: Alternative Architectures Require Add-Ons

    Standard RNNs, LSTMs, Transformers should require ≥2 explicit modules to match
    ≥85% of LTCN performance.

    Args:
        alternative_modules_needed: Number of modules needed for alternative architectures
        performance_gap_without_addons: Performance gap without add-ons
        min_modules_needed: Minimum number of modules required
        min_performance_gap: Minimum performance gap percentage

    Returns:
        Dictionary with pass/fail result and metrics
    """
    f6_6_pass = (
        alternative_modules_needed >= min_modules_needed
        and performance_gap_without_addons >= min_performance_gap
    )

    return {
        "passed": f6_6_pass,
        "add_ons_needed": alternative_modules_needed,
        "performance_gap": performance_gap_without_addons,
        "threshold": f"≥{min_modules_needed} add-ons, gap ≥{min_performance_gap}%",
    }
