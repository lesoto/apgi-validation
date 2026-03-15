"""
criteria_registry.py
======================

Single source of truth for all APGI falsification criteria definitions.

This module eliminates code duplication across Validation Protocols 6, 7, 8, 9, 11, and 12.
Every protocol MUST import criteria from here rather than hard-coding them locally.

Usage::

    from utils.criteria_registry import (
        get_falsification_criteria,
        get_protocol_specific_criteria,
    )
"""

from typing import Dict, Any


# =============================================================================
# CORE FALSIFICATION CRITERIA (F1.1–F6.2)
# =============================================================================

FALSIFICATION_CRITERIA: Dict[str, Dict[str, Any]] = {
    # F1.1: Threshold Ignition Emergence
    "F1.1": {
        "name": "Threshold Ignition Emergence",
        "description": "APGI agents should show discrete ignition events (Sₜ > θₜ) with sudden firing-rate increases",
        "test_type": "binomial",
        "threshold": "≥75% agents show ignition",
        "falsification_threshold": "<65% agents show ignition",
        "statistic": "binomtest",
        "min_proportion": 0.75,
        "falsification_proportion": 0.65,
        "alpha": 0.01,
    },
    # F1.2: Precision-Weighted Coding
    "F1.2": {
        "name": "Precision-Weighted Coding",
        "description": "Neural representations should be weighted by precision Π",
        "test_type": "correlation",
        "threshold": "r ≥ 0.45 between precision and neural gain",
        "falsification_threshold": "r < 0.35",
        "statistic": "pearsonr",
        "min_correlation": 0.45,
        "falsification_correlation": 0.35,
        "alpha": 0.01,
    },
    # F1.3: Interoceptive Prioritization
    "F1.3": {
        "name": "Interoceptive Prioritization",
        "description": "Interoceptive prediction errors should receive higher weighting than exteroceptive",
        "test_type": "ratio",
        "threshold": "Πⁱ/Πᵉ ≥ 1.30 during interoceptive salience",
        "falsification_threshold": "Πⁱ/Πᵉ < 1.15",
        "min_ratio": 1.30,
        "falsification_ratio": 1.15,
        "alpha": 0.01,
    },
    # F1.4: Threshold Adaptation Dynamics
    "F1.4": {
        "name": "Threshold Adaptation Dynamics",
        "description": "Ignition threshold θₜ should adapt based on metabolic cost",
        "test_type": "correlation",
        "threshold": "r ≥ 0.50 between θₜ and metabolic cost",
        "falsification_threshold": "r < 0.30",
        "min_correlation": 0.50,
        "falsification_correlation": 0.30,
        "alpha": 0.01,
    },
    # F1.5: Cross-Level Phase-Amplitude Coupling (PAC)
    "F1.5": {
        "name": "Cross-Level Phase-Amplitude Coupling",
        "description": "PAC between theta phase and gamma amplitude should increase during ignition",
        "test_type": "permutation",
        "threshold": "MI ≥ 0.012, increase ≥ 30%, d ≥ 0.50",
        "falsification_threshold": "MI < 0.008 OR increase < 15%",
        "min_mi": 0.012,
        "min_increase_pct": 30.0,
        "min_cohens_d": 0.50,
        "alpha": 0.01,
    },
    # F1.6: Spectral Slope During Ignition
    "F1.6": {
        "name": "Spectral Slope During Ignition",
        "description": "1/f spectral slope should flatten during ignition",
        "test_type": "ttest",
        "threshold": "slope difference ≥ 0.5, d ≥ 0.60",
        "falsification_threshold": "slope difference < 0.2",
        "min_slope_diff": 0.5,
        "min_cohens_d": 0.60,
        "alpha": 0.01,
    },
    # F2.1: Advantageous Deck Selection (Iowa Gambling Task)
    "F2.1": {
        "name": "Advantageous Deck Selection",
        "description": "APGI agents should prefer advantageous decks in IGT",
        "test_type": "binomial",
        "threshold": "≥70% selections from advantageous decks",
        "falsification_threshold": "<55% selections from advantageous decks",
        "min_proportion": 0.70,
        "falsification_proportion": 0.55,
        "alpha": 0.01,
    },
    # F2.2: Interoceptive Cost Correlation
    "F2.2": {
        "name": "Interoceptive Cost Correlation",
        "description": "Deck selection should correlate with interoceptive cost signals",
        "test_type": "correlation",
        "threshold": "r ≥ 0.40 between cost signals and avoidance",
        "falsification_threshold": "r < 0.25",
        "min_correlation": 0.40,
        "falsification_correlation": 0.25,
        "alpha": 0.01,
    },
    # F2.3: vmPFC-Like Anticipatory Bias (RT advantage)
    "F2.3": {
        "name": "vmPFC-Like Anticipatory Bias",
        "description": "Reaction time advantage for advantageous decks",
        "test_type": "ttest",
        "threshold": "RT advantage ≥ 50ms, β ≥ 25ms, R² ≥ 0.18",
        "falsification_threshold": "RT advantage < 25ms",
        "min_rt_advantage_ms": 50.0,
        "min_beta": 25.0,
        "min_r2": 0.18,
        "alpha": 0.01,
    },
    # F2.4: Precision vs Error Magnitude
    "F2.4": {
        "name": "Precision vs Error Magnitude",
        "description": "Neural precision should scale with prediction error magnitude",
        "test_type": "regression",
        "threshold": "β ≥ 0.40, R² ≥ 0.20",
        "falsification_threshold": "β < 0.20 OR R² < 0.10",
        "min_beta": 0.40,
        "min_r2": 0.20,
        "alpha": 0.01,
    },
    # F2.5: Somatic Marker Integration
    "F2.5": {
        "name": "Somatic Marker Integration",
        "description": "Somatic signals should modulate decision thresholds",
        "test_type": "hazard_ratio",
        "threshold": "HR ≥ 1.50 for advantageous decisions",
        "falsification_threshold": "HR < 1.20",
        "min_hazard_ratio": 1.50,
        "falsification_hazard_ratio": 1.20,
        "alpha": 0.01,
    },
    # F3.1: Phase Transition Detection
    "F3.1": {
        "name": "Phase Transition Detection",
        "description": "Abrupt changes in neural dynamics at critical points",
        "test_type": "susceptibility",
        "threshold": "susceptibility ratio ≥ 3.0",
        "falsification_threshold": "susceptibility ratio < 2.0",
        "min_susceptibility_ratio": 3.0,
        "falsification_ratio": 2.0,
        "alpha": 0.01,
    },
    # F3.2: Φ Spike During Ignition
    "F3.2": {
        "name": "Φ Spike During Ignition",
        "description": "Integrated information should spike during ignition",
        "test_type": "ttest",
        "threshold": "Φ increase ≥ 2.5×, d ≥ 0.80",
        "falsification_threshold": "Φ increase < 1.5×",
        "min_phi_increase": 2.5,
        "min_cohens_d": 0.80,
        "alpha": 0.01,
    },
    # F3.3: Critical Slowing Down
    "F3.3": {
        "name": "Critical Slowing Down",
        "description": "Autocorrelation should increase near critical points",
        "test_type": "correlation",
        "threshold": "autocorrelation increase ≥ 0.30",
        "falsification_threshold": "autocorrelation increase < 0.15",
        "min_ac_increase": 0.30,
        "falsification_ac_increase": 0.15,
        "alpha": 0.01,
    },
    # F3.4: Discontinuity Effect Size
    "F3.4": {
        "name": "Discontinuity Effect Size",
        "description": "Large discontinuities in neural metrics at phase transitions",
        "test_type": "ttest",
        "threshold": "Cohen's d ≥ 0.70",
        "falsification_threshold": "Cohen's d < 0.40",
        "min_cohens_d": 0.70,
        "falsification_cohens_d": 0.40,
        "alpha": 0.01,
    },
    # F3.5: Hurst Exponent Analysis
    "F3.5": {
        "name": "Hurst Exponent Analysis",
        "description": "Long-range temporal correlations in neural dynamics",
        "test_type": "equivalence",
        "threshold": "H ∈ [0.65, 0.85]",
        "falsification_threshold": "H < 0.50 OR H > 0.95",
        "min_hurst": 0.65,
        "max_hurst": 0.85,
        "alpha": 0.01,
    },
    # F3.6: Bistability Detection
    "F3.6": {
        "name": "Bistability Detection",
        "description": "Two stable states in neural dynamics",
        "test_type": "bimodality",
        "threshold": "Hartigan's dip test p < 0.01, dip statistic ≥ 0.05",
        "falsification_threshold": "p ≥ 0.05 OR dip < 0.03",
        "min_dip_statistic": 0.05,
        "falsification_dip_statistic": 0.03,
        "alpha": 0.01,
    },
    # F4.1: Multi-Scale Precision Hierarchy
    "F4.1": {
        "name": "Multi-Scale Precision Hierarchy",
        "description": "Precision should vary hierarchically across neural levels",
        "test_type": "anova",
        "threshold": "F ≥ 5.0, η² ≥ 0.15",
        "falsification_threshold": "F < 3.0 OR η² < 0.08",
        "min_f_statistic": 5.0,
        "min_eta_squared": 0.15,
        "alpha": 0.01,
    },
    # F4.2: Cross-Level Coherence
    "F4.2": {
        "name": "Cross-Level Coherence",
        "description": "Coherence between neural levels during ignition",
        "test_type": "correlation",
        "threshold": "coherence ≥ 0.40",
        "falsification_threshold": "coherence < 0.25",
        "min_coherence": 0.40,
        "falsification_coherence": 0.25,
        "alpha": 0.01,
    },
    # F4.3: Spectral Slope Hierarchy
    "F4.3": {
        "name": "Spectral Slope Hierarchy",
        "description": "1/f slope should vary systematically across levels",
        "test_type": "trend",
        "threshold": "monotonic decrease, R² ≥ 0.70",
        "falsification_threshold": "R² < 0.40",
        "min_r2": 0.70,
        "falsification_r2": 0.40,
        "alpha": 0.01,
    },
    # F4.4: Information Flow Direction
    "F4.4": {
        "name": "Information Flow Direction",
        "description": "Granger causality should flow bottom-up during ignition",
        "test_type": "granger",
        "threshold": "bottom-up GC ≥ 2× top-down",
        "falsification_threshold": "bottom-up GC < 1.5× top-down",
        "min_gc_ratio": 2.0,
        "falsification_gc_ratio": 1.5,
        "alpha": 0.01,
    },
    # F4.5: Cross-Scale Integration
    "F4.5": {
        "name": "Cross-Scale Integration",
        "description": "Integration across spatial and temporal scales",
        "test_type": "correlation",
        "threshold": "cross-scale correlation ≥ 0.50",
        "falsification_threshold": "cross-scale correlation < 0.30",
        "min_correlation": 0.50,
        "falsification_correlation": 0.30,
        "alpha": 0.01,
    },
    # F5.1: Threshold Filtering Emergence
    "F5.1": {
        "name": "Threshold Filtering Emergence",
        "description": "Evolved agents should develop threshold-like gating",
        "test_type": "binomial",
        "threshold": "≥75% agents with threshold filtering, mean α ≥ 4.0, d ≥ 0.80",
        "falsification_threshold": "<60% agents OR mean α < 3.0",
        "min_proportion": 0.75,
        "min_alpha": 4.0,
        "falsification_alpha": 3.0,
        "min_cohens_d": 0.80,
        "alpha": 0.01,
    },
    # F5.2: Precision-Weighted Coding Emergence
    "F5.2": {
        "name": "Precision-Weighted Coding Emergence",
        "description": "Evolved agents should develop precision weighting",
        "test_type": "correlation",
        "threshold": "r ≥ 0.45, ≥65% agents, d ≥ 0.70",
        "falsification_threshold": "r < 0.35",
        "min_correlation": 0.45,
        "falsification_correlation": 0.35,
        "min_proportion": 0.65,
        "min_cohens_d": 0.70,
        "alpha": 0.01,
    },
    # F5.3: Interoceptive Prioritization Emergence
    "F5.3": {
        "name": "Interoceptive Prioritization Emergence",
        "description": "Evolved agents should prioritize interoceptive signals",
        "test_type": "ratio",
        "threshold": "gain ratio ≥ 1.30, ≥70% agents, d ≥ 0.60",
        "falsification_threshold": "ratio < 1.15",
        "min_ratio": 1.30,
        "falsification_ratio": 1.15,
        "min_proportion": 0.70,
        "min_cohens_d": 0.60,
        "alpha": 0.01,
    },
    # F5.4: Multi-Timescale Integration Emergence
    "F5.4": {
        "name": "Multi-Timescale Integration Emergence",
        "description": "Evolved agents should integrate across timescales",
        "test_type": "binomial",
        "threshold": "≥60% agents, peak separation ≥ 3×",
        "falsification_threshold": "<45% agents OR separation < 2.0×",
        "min_proportion": 0.60,
        "falsification_proportion": 0.45,
        "min_peak_separation": 3.0,
        "falsification_separation": 2.0,
        "alpha": 0.01,
    },
    # F5.5: PCA Variance Structure
    "F5.5": {
        "name": "PCA Variance Structure",
        "description": "PCA should reveal predicted dimensional structure",
        "test_type": "pca",
        "threshold": "cumulative variance ≥ 70%, PC loading ≥ 0.60",
        "falsification_threshold": "cumulative variance < 60%",
        "min_cumulative_variance": 0.70,
        "falsification_variance": 0.60,
        "min_pc_loading": 0.60,
        "alpha": 0.01,
    },
    # F5.6: Non-APGI Architecture Failure
    "F5.6": {
        "name": "Non-APGI Architecture Failure",
        "description": "Architectures without APGI components should underperform",
        "test_type": "ttest",
        "threshold": "≥40% worse performance, d ≥ 0.85",
        "falsification_threshold": "<25% worse performance",
        "min_performance_diff_pct": 40.0,
        "falsification_performance_diff_pct": 25.0,
        "min_cohens_d": 0.85,
        "alpha": 0.01,
    },
    # F6.1: Intrinsic Threshold Behavior (LTCN transition)
    "F6.1": {
        "name": "Intrinsic Threshold Behavior",
        "description": "LTCNs should show sharp ignition transitions <50ms",
        "test_type": "mann_whitney",
        "threshold": "transition ≤50ms, Cliff's δ ≥ 0.60",
        "falsification_threshold": "transition >75ms OR δ < 0.40",
        "max_transition_ms": 50.0,
        "falsification_transition_ms": 75.0,
        "min_cliffs_delta": 0.60,
        "falsification_cliffs_delta": 0.40,
        "alpha": 0.01,
    },
    # F6.2: Intrinsic Temporal Integration
    "F6.2": {
        "name": "Intrinsic Temporal Integration",
        "description": "LTCNs should integrate over 200-500ms windows",
        "test_type": "ratio",
        "threshold": "window ≥200ms, ratio ≥4× RNN, R² ≥0.85",
        "falsification_threshold": "window <150ms OR ratio <2.5×",
        "min_window_ms": 200.0,
        "falsification_window_ms": 150.0,
        "min_integration_ratio": 4.0,
        "falsification_ratio": 2.5,
        "min_r2": 0.85,
        "alpha": 0.01,
    },
}


# =============================================================================
# PROTOCOL-SPECIFIC CRITERIA
# =============================================================================

PROTOCOL_SPECIFIC_CRITERIA: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Validation-Protocol-1": {
        "V1.1": {
            "name": "Heartbeat Discrimination Accuracy",
            "description": "APGI should improve heartbeat discrimination compared to StandardPP",
            "threshold": "d' improvement ≥ 0.30",
            "alpha": 0.05,
        },
        "V1.2": {
            "name": "Visual Detection Threshold Modulation",
            "description": "Detection threshold should vary with interoceptive precision",
            "threshold": "threshold shift ≥ 15%",
            "alpha": 0.05,
        },
        "V1.3": {
            "name": "Arousal Interaction",
            "description": "High arousal (100-120 bpm) should enhance interoceptive weighting",
            "threshold": "interaction d = 0.25–0.45",
            "alpha": 0.05,
        },
    },
    "Validation-Protocol-5": {
        "V5.1": {
            "name": "Algorithmic Falsification",
            "description": "APGI equations must produce correct numerical predictions",
            "threshold": "numerical accuracy ε ≤ 1e-6",
            "alpha": 0.001,
        },
    },
    "Validation-Protocol-9": {
        "V9.1": {
            "name": "Symptom Prediction",
            "description": "APGI metrics should predict clinical symptoms",
            "threshold": "r ≥ 0.60",
            "alpha": 0.01,
        },
        "V9.2": {
            "name": "Treatment Response Prediction",
            "description": "APGI metrics should predict treatment outcomes",
            "threshold": "r ≥ 0.50",
            "alpha": 0.01,
        },
        "V9.3": {
            "name": "Biomarker Prediction",
            "description": "APGI metrics should correlate with neural biomarkers",
            "threshold": "r ≥ 0.70",
            "alpha": 0.01,
        },
        "V9.4": {
            "name": "Cognitive Performance",
            "description": "APGI metrics should predict cognitive task performance",
            "threshold": "r ≥ 0.50",
            "alpha": 0.01,
        },
    },
    "Validation-Protocol-12": {
        "V12.1": {
            "name": "Clinical Gradient Prediction",
            "description": "APGI should distinguish VS, MCS, EMCS, healthy",
            "threshold": "P3b reduction ≥80%, ignition reduction ≥70%, d ≥ 0.80",
            "alpha": 0.01,
        },
        "V12.2": {
            "name": "Cross-Species Homology",
            "description": "APGI dynamics should be conserved across species",
            "threshold": "r ≥ 0.60, Pillai's trace ≥ 0.40",
            "falsification_threshold": "r < 0.45 OR Pillai's < 0.25",
            "alpha": 0.01,
        },
    },
}


def get_falsification_criteria() -> Dict[str, Dict[str, Any]]:
    """
    Get all core falsification criteria (F1.1–F6.2).

    Returns:
        Dictionary mapping criterion IDs to their specifications
    """
    return FALSIFICATION_CRITERIA.copy()


def get_protocol_specific_criteria(protocol_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Get protocol-specific validation criteria.

    Args:
        protocol_name: Name of the validation protocol

    Returns:
        Dictionary mapping criterion IDs to their specifications
    """
    return PROTOCOL_SPECIFIC_CRITERIA.get(protocol_name, {}).copy()


def get_all_criteria(protocol_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all criteria for a protocol (core + protocol-specific).

    Args:
        protocol_name: Name of the validation protocol

    Returns:
        Dictionary mapping all criterion IDs to their specifications
    """
    all_criteria = get_falsification_criteria()
    protocol_criteria = get_protocol_specific_criteria(protocol_name)
    all_criteria.update(protocol_criteria)
    return all_criteria


def get_criterion(criterion_id: str) -> Dict[str, Any]:
    """
    Get a specific criterion by ID.

    Args:
        criterion_id: ID of the criterion (e.g., "F1.1", "V5.1")

    Returns:
        Dictionary with criterion specification
    """
    # Check core criteria first
    if criterion_id in FALSIFICATION_CRITERIA:
        return FALSIFICATION_CRITERIA[criterion_id].copy()

    # Check protocol-specific criteria
    for protocol_criteria in PROTOCOL_SPECIFIC_CRITERIA.values():
        if criterion_id in protocol_criteria:
            return protocol_criteria[criterion_id].copy()

    raise ValueError(f"Criterion {criterion_id} not found in registry")
