"""
LEVEL DESIGNATION: Level 2 (information-theoretic)

Bridge to Level 1
"""

from typing import Any, Dict

# =============================================================================
# CORE FALSIFICATION CRITERIA (F1.1–F6.2)
# =============================================================================

# =============================================================================
# CANONICAL PROXY MAPPING: Theoretical constructs → Empirical measures
# This table is the authoritative cross-reference between APGI symbols and
# the experimental measures used in each protocol.
# =============================================================================

CONSTRUCT_PROXY_MAP = {
    "B_t": {
        "description": "Ignition event (binary): B_t = 1 iff S_t > θ_t",
        "computational": "Heaviside(S_t - theta_t); see algorithmic_verification.py",
        "proxies": {
            "P1_EEG": {"measure": "P3b amplitude", "threshold": "≥ 0.3 µV", "criterion": "F9.1"},
            "P2_TMS": {"measure": "PCI", "threshold": "≥ 0.31", "criterion": "P2.a"},
            "P4_DoC": {"measure": "PCI", "threshold": "≥ 0.31", "criterion": "P4.a"},
            "P6_iEEG": {"measure": "Hartigan dip p", "threshold": "< 0.05", "criterion": "P6a"},
        },
    },
    "Pi_i": {
        "description": "Interoceptive precision (baseline)",
        "proxies": {
            "P1_EEG": {
                "measure": "HEP amplitude 250–400 ms",
                "threshold": "trial-by-trial, r > 0.4 with P3b",
                "criterion": "P1a",
            },
            "P2_TMS": {
                "measure": "HEP amplitude 250–400 ms",
                "threshold": "≥ 30% reduction under insula TMS",
                "criterion": "P2.b",
            },
            "P4_DoC": {
                "measure": "HEP amplitude 250–400 ms",
                "threshold": "d > 0.5 MCS vs VS/UWS",
                "criterion": "P4b",
            },
        },
        "proxy_equation": "HEP_amplitude ∝ Pi_i_eff (monotonic, not linear; calibration required per session)",
    },
    "Pi_e": {
        "description": "Exteroceptive precision",
        "proxies": {
            "P3_sim": {"measure": "Gabor contrast level", "threshold": "0.02–0.32 cpd range", "criterion": "P3a"},
        },
        "note": "Pi_e is manipulated as stimulus contrast in P1/P6; it is not measured independently. "
        "Must be added to Protocol Table P1 hypothesis column.",
    },
    "S_t": {
        "description": "Accumulated surprise: dS/dt = -S/τ_S + Πᵉ|εᵉ| + β_s·Πⁱ|εⁱ|",
        "proxies": {
            "P1_EEG": {
                "measure": "N2 amplitude (200–300 ms) or pre-P3b negativity",
                "threshold": "not yet formally registered",
                "criterion": "MISSING — see TODO below",
            },
            "P6_iEEG": {
                "measure": "Pre-ignition high-gamma envelope",
                "threshold": "AC1 increase Kendall τ > 0.3",
                "criterion": "P6c",
            },
        },
        "todo": "P1 and P4 have no registered empirical proxy for S_t. "
        "Add N2/pre-P3b negativity as S_t proxy to NAMED_PREDICTIONS and F-criteria.",
    },
    "M_hat": {
        "description": "Somatic marker: M̂(c,a) = γ_V·V(c,a) + γ_A·A(c,a)",
        "scalar_alias": "M (used in sympy dict — scalar approximation only)",
        "full_form": "bilinear: M̂(c,a) = γ_V·V(c,a) + γ_A·A(c,a)",
        "proxies": {
            "P3_sim": {
                "measure": "Agent M̂ signal (simulated)",
                "threshold": "leads B_t by ≥1 trial in ≥75% events",
                "criterion": "P3c",
            },
            "P5_fMRI": {
                "measure": "vmPFC BOLD (anticipation phase)",
                "threshold": "PPI coupling r > 0.4 vs posterior insula",
                "criterion": "P5.a",
            },
        },
    },
}
# =============================================================================
# PROXY CALIBRATION SPECIFICATIONS
# Quantitative relationship between empirical measure and theoretical construct.
# All protocols MUST import from here; never assert proxy relationships inline.
# =============================================================================

PROXY_CALIBRATION = {
    "HEP_to_Pi_i": {
        "functional_form": "monotonic_positive",
        "specification": "HEP amplitude (µV, 250–400 ms, Cz) is a rank-order proxy for Πⁱ_eff. "
        "No linear scaling assumed. Operationalised as: high Πⁱ ↔ HEP > median "
        "within-session; low Πⁱ ↔ HEP ≤ median. Trial-by-trial mixed-effects "
        "regression (P3b ~ HEP_amplitude + RMSSD + random slopes) is the primary "
        "calibration procedure. Session-level z-scoring required before cross-session comparison.",
        "minimum_r_for_proxy_validity": 0.35,  # EP-0 CSV: HEP–Πⁱ r > 0.35 required for proxy validity
        "falsification": "HEP–P3b r < 0.35 invalidates HEP as Πⁱ proxy for that session (EP-0 criterion)",
        "references": ["Garfinkel et al. (2014)", "Canales-Johnson et al. (2020)"],
    },
    "PCI_to_B_t": {
        "functional_form": "threshold_dichotomous",
        "specification": "PCI ≥ 0.31 is treated as B_t = 1 (ignition present); PCI < 0.31 as B_t = 0. "
        "Threshold sourced from Casarotto et al. (2016) 100% specificity cutoff. "
        "Within-protocol comparisons use continuous PCI; the 0.31 cutoff applies "
        "only to clinical DoC classification (Protocol 4) and consciousness level "
        "assignment. Do NOT use 0.31 as a threshold in TMS or iEEG Protocol 2/6 "
        "comparisons — use within-condition delta-PCI instead.",
        "pci_consciousness_threshold": 0.31,
        "delta_pci_minimum_effect": 0.05,  # Minimum meaningful PCI change within condition
        "falsification": "PCI change < 0.05 following insula TMS does not constitute threshold shift (P2a)",
        "references": ["Casarotto et al. (2016)", "Comolatti et al. (2019)"],
    },
}

FALSIFICATION_CRITERIA: Dict[str, Dict[str, Any]] = {
    # F1.1: Threshold Ignition Emergence
    "F1.1": {
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
        "name": "vmPFC-Like Anticipatory Bias",
        "description": "vmPFC-Like Anticipatory Bias — operationalized via M̂(c,a) = γ_V·V(c,a) + γ_A·A(c,a), NOT scalar M. "
        "Test uses vmPFC BOLD as M̂ proxy during anticipation phase (P5a). "
        "Falsified if vmPFC correlates with outcome-phase εⁱ rather than anticipation-phase M̂.",
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
        "type": "FAL",
        "name": "Precision vs Error Magnitude",
        "description": "Precision-Weighted Integration via M̂ (Not Error Magnitude). "
        "M̂ modulates Πⁱ_eff = Πⁱ · exp(β_s · M̂), not raw |εⁱ|. "
        "Uses full bilinear form M̂(c,a); scalar M is insufficient for this criterion.",
        "test_type": "regression",
        "threshold": "β ≥ 0.40, R² ≥ 0.20",
        "falsification_threshold": "β < 0.20 OR R² < 0.10",
        "min_beta": 0.40,
        "min_r2": 0.20,
        "alpha": 0.01,
    },
    # F2.5: Somatic Marker Integration
    "F2.5": {
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
        "type": "FAL",
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
    # VP-08: Psychophysical Threshold Estimation (Paper 1 — Protocol 1)
    # V1.1–V1.3 here are the authoritative registry definitions; VP-01's local
    # criteria use the VP01.x prefix to avoid collision.
    "VP_08_PsychophysicalThresholdEstimation": {
        "V1.1": {
            "type": "VAL",
            "name": "Heartbeat Discrimination Accuracy",
            "description": "APGI should improve heartbeat discrimination compared to StandardPP",
            "threshold": "d' improvement ≥ 0.30",
            "alpha": 0.05,
        },
        "V1.2": {
            "type": "VAL",
            "name": "Visual Detection Threshold Modulation",
            "description": "Detection threshold should vary with interoceptive precision",
            "threshold": "threshold shift ≥ 15%",
            "alpha": 0.05,
        },
        "V1.3": {
            "type": "VAL",
            "name": "Arousal Interaction",
            "description": "High arousal (100-120 bpm) should enhance interoceptive weighting",
            "threshold": "interaction d = 0.25–0.45",
            "alpha": 0.05,
        },
    },
    "Validation_Protocol_5": {
        "V5.1": {
            "type": "VAL",
            "name": "Algorithmic Falsification",
            "description": "APGI equations must produce correct numerical predictions",
            "threshold": "numerical accuracy ε ≤ 1e-6",
            "alpha": 0.001,
        },
    },
    "Validation_Protocol_9": {
        "V9.1": {
            "type": "VAL",
            "name": "Symptom Prediction",
            "description": "APGI metrics should predict clinical symptoms",
            "threshold": "r ≥ 0.60",
            "alpha": 0.01,
        },
        "V9.2": {
            "type": "VAL",
            "name": "Treatment Response Prediction",
            "description": "APGI metrics should predict treatment outcomes",
            "threshold": "r ≥ 0.50",
            "alpha": 0.01,
        },
        "V9.3": {
            "type": "VAL",
            "name": "Biomarker Prediction",
            "description": "APGI metrics should correlate with neural biomarkers",
            "threshold": "r ≥ 0.70",
            "alpha": 0.01,
        },
        "V9.4": {
            "type": "VAL",
            "name": "Cognitive Performance",
            "description": "APGI metrics should predict cognitive task performance",
            "threshold": "r ≥ 0.50",
            "alpha": 0.01,
        },
    },
    "Validation_Protocol_12": {
        "V12.1": {
            "type": "VAL",
            "name": "Clinical Gradient Prediction",
            "description": "APGI should distinguish VS, MCS, EMCS, healthy",
            "threshold": "P3b reduction ≥80%, ignition reduction ≥70%, d ≥ 0.80",
            "alpha": 0.01,
        },
        "V12.2": {
            "type": "VAL",
            "name": "Cross-Species Homology",
            "description": "APGI dynamics should be conserved across species",
            "threshold": "r ≥ 0.60, Pillai's trace ≥ 0.40",
            "falsification_threshold": "r < 0.45 OR Pillai's < 0.25",
            "alpha": 0.01,
        },
    },
    # HIGH-05: VP-13 Epistemic Architecture Validation Protocol
    "VP_13_EpistemicArchitecture": {
        "V13.1": {
            "type": "VAL",
            "name": "Epistemic Hierarchy Validation",
            "description": "Multi-scale precision hierarchy should emerge across neural levels",
            "threshold": "susceptibility ratio ≥ 2.0, η² ≥ 0.15",
            "falsification_threshold": "susceptibility ratio < 1.5 OR η² < 0.08",
            "min_susceptibility_ratio": 2.0,
            "falsification_ratio": 1.5,
            "min_eta_squared": 0.15,
            "falsification_eta_squared": 0.08,
            "alpha": 0.01,
        },
        "V13.2": {
            "type": "VAL",
            "name": "Cross-Level Coherence",
            "description": "Coherence between neural levels during ignition events",
            "threshold": "mutual information ≥ 10.0 bits/s, coherence ≥ 0.40",
            "falsification_threshold": "MI < 6.0 bits/s OR coherence < 0.25",
            "min_mutual_info": 10.0,
            "falsification_mi": 6.0,
            "min_coherence": 0.40,
            "falsification_coherence": 0.25,
            "alpha": 0.01,
        },
        "V13.3": {
            "type": "VAL",
            "name": "Information Flow Directionality",
            "description": "Granger causality should flow bottom-up during ignition",
            "threshold": "TE ≥ 0.1 bits, bottom-up GC ≥ 2× top-down",
            "falsification_threshold": "TE < 0.05 OR GC ratio < 1.5",
            "min_transfer_entropy": 0.1,
            "falsification_te": 0.05,
            "min_gc_ratio": 2.0,
            "falsification_gc_ratio": 1.5,
            "alpha": 0.01,
        },
        "V13.4": {
            "type": "VAL",
            "name": "Phase Transition Detection",
            "description": "Critical slowing and susceptibility peaks at ignition",
            "threshold": "critical slowing ratio ≥ 1.2 (20% increase), p < 0.05",
            "falsification_threshold": "ratio < 1.1 OR p ≥ 0.05",
            "min_critical_slowing_ratio": 1.2,
            "falsification_ratio": 1.1,
            "alpha": 0.05,
        },
        "V13.5": {
            "type": "VAL",
            "name": "Cross-Scale Integration",
            "description": "Integrated information across spatial and temporal scales",
            "threshold": "Φ_proxy ≥ 0.5 bits, cross-scale correlation ≥ 0.50",
            "falsification_threshold": "Φ_proxy < 0.3 OR correlation < 0.30",
            "min_phi_proxy": 0.5,
            "falsification_phi": 0.3,
            "min_correlation": 0.50,
            "falsification_correlation": 0.30,
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
