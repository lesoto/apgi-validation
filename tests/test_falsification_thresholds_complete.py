"""
Comprehensive Tests for Falsification Thresholds Module
======================================================

Target: 100% coverage for utils/falsification_thresholds.py
Tests threshold constants and validation functions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import test functions (aliased for readability in test methods)
from utils.falsification_thresholds import (
    BIC_FRAMEWORK_THRESHOLD_B,
    BIC_STRONG_EVIDENCE,
    BIC_VERY_STRONG,
    DOC_AUC_MAX,
    DOC_AUC_MIN,
    F1_1_ALPHA,
    F1_1_MIN_ADVANTAGE_PCT,
    F1_1_MIN_COHENS_D,
    F2_1_ALPHA,
    F2_1_MIN_ADVANTAGE_PCT,
    F2_1_MIN_COHENS_H,
    F2_1_MIN_PP_DIFF,
    F2_2_ALPHA,
    F2_2_MIN_CORR,
    F2_3_ALPHA,
    F2_3_MIN_R2,
    F2_3_MIN_RT_ADVANTAGE_MS,
    F2_4_ALPHA,
    F2_5_ALPHA,
    F3_1_ALPHA,
    F3_1_MIN_ADVANTAGE_PCT,
    F3_1_MIN_COHENS_D,
    F3_2_ALPHA,
    F3_2_MIN_INTERO_ADVANTAGE_PCT,
    F3_3_ALPHA,
    F3_3_MIN_REDUCTION_PCT,
    F3_4_ALPHA,
    F3_6_ALPHA,
    F5_1_BINOMIAL_ALPHA,
    F5_1_MIN_ALPHA,
    F5_1_MIN_COHENS_D,
    F5_1_MIN_PROPORTION,
    F5_2_BINOMIAL_ALPHA,
    F5_2_MIN_CORRELATION,
    F5_2_MIN_PROPORTION,
    F5_4_BINOMIAL_ALPHA,
    F5_4_MIN_PEAK_SEPARATION,
    F5_4_MIN_PROPORTION,
    F5_5_MIN_LOADING,
    F5_5_PCA_FALSIFICATION_THRESHOLD,
    F5_5_PCA_MIN_VARIANCE,
    F6_1_CLIFFS_DELTA_MIN,
    F6_1_LTCN_MAX_TRANSITION_MS,
    F6_1_MANN_WHITNEY_ALPHA,
    F6_2_FALSIFICATION_RATIO,
    F6_2_LTCN_MIN_WINDOW_MS,
    F6_2_MIN_CURVE_FIT_R2,
    F6_2_MIN_INTEGRATION_RATIO,
    F6_2_MIN_R2,
    F6_2_WILCOXON_ALPHA,
    F6_DELTA_AUROC_MIN,
    F6_SPARSITY_ACTIVATION_THRESHOLD,
    F8_IDENTIFIABILITY_MIN_R2,
    F8_SOBOL_MIN_SENSITIVITY,
    GENERIC_ALPHA,
    GENERIC_BINARY_DECISION_THRESHOLD,
    GENERIC_MEDIUM_COHENS_D,
    GENERIC_MIN_AUC,
    GENERIC_MIN_COHENS_D,
    GENERIC_MIN_CORR,
    GENERIC_MIN_R2,
    LIQUID_IGNITION_DETECTION_THRESHOLD,
    P1_1_MAX_D_PRIME,
    P1_1_MIN_D_PRIME,
    P1_2_AROUSAL_INTERACTION_MIN_D,
    P1_3_IA_BENEFIT_MIN_D,
    P2_A_MIN_THRESHOLD_SHIFT,
    P2_B_MIN_HEP_REDUCTION,
    P2_B_MIN_PCI_REDUCTION,
    P2_C_MIN_ETA_SQ,
    P7_MIN_AUC,
    P11_MIN_R2,
    P12_A_EXPONENT_MAX,
    P12_A_EXPONENT_MIN,
    P12_B_MIN_CONSISTENCY_PCT,
    P12_C_PROPFOLOL_REDUCTION_MIN_PCT,
    THRESHOLD_REGISTRY,
    V7_1_ALPHA,
    V7_1_MIN_COHENS_D,
    V7_1_MIN_EFFECT_DURATION_MIN,
    V7_1_MIN_THRESHOLD_REDUCTION_PCT,
    V7_2_ALPHA,
    V11_MIN_COHENS_D,
    V11_MIN_DELTA_R2,
    V11_MIN_R2,
    V12_1_ALPHA,
    V12_1_MIN_COHENS_D,
    V12_1_MIN_ETA_SQUARED,
    V12_1_MIN_IGNITION_REDUCTION_PCT,
    V12_1_MIN_P3B_REDUCTION_PCT,
    V16_C1_CONSISTENCY_CV,
    V16_MIN_CORRELATION,
    V16_MIN_EFFICIENCY_GAIN,
    V17_ALPHA,
    V17_MIN_CORRELATION_MAGNITUDE,
    V17_MIN_R2_P3B_DECAY,
    V17_MIN_R2_THETA_ELEVATION,
    VP2_AROUSAL_BOOST_MAX,
    VP2_AROUSAL_COUPLING_SCALE,
    VP2_DELTA_PI_COUPLING,
    VP4_CALIBRATED_ALPHA,
    VP4_CALIBRATED_TAU,
    VP4_CALIBRATED_THETA_0,
)
from utils.falsification_thresholds import (
    test_f6_1_intrinsic_threshold_behavior as f6_1_validator,
)
from utils.falsification_thresholds import (
    test_f6_2_intrinsic_temporal_integration as f6_2_validator,
)
from utils.falsification_thresholds import (
    test_f6_3_metabolic_selectivity as f6_3_validator,
)
from utils.falsification_thresholds import test_f6_4_fading_memory as f6_4_validator
from utils.falsification_thresholds import (
    test_f6_5_bifurcation_structure as f6_5_validator,
)
from utils.falsification_thresholds import (
    test_f6_6_alternative_architectures as f6_6_validator,
)


class TestThresholdConstants:
    """Test that all threshold constants are defined and have valid values"""

    def test_f6_1_thresholds(self):
        """Test F6.1 LTCN intrinsic threshold behavior thresholds"""
        assert F6_1_LTCN_MAX_TRANSITION_MS == 50.0
        assert F6_1_CLIFFS_DELTA_MIN == 0.60
        assert F6_1_MANN_WHITNEY_ALPHA == 0.05

    def test_f6_2_thresholds(self):
        """Test F6.2 intrinsic temporal integration thresholds"""
        assert F6_2_LTCN_MIN_WINDOW_MS == 200.0
        assert F6_2_MIN_INTEGRATION_RATIO == 4.0
        assert F6_2_FALSIFICATION_RATIO == 2.5
        assert F6_2_MIN_CURVE_FIT_R2 == 0.85
        assert F6_2_MIN_R2 == 0.85
        assert F6_2_WILCOXON_ALPHA == 0.05

    def test_f5_pca_thresholds(self):
        """Test F5 PCA-related thresholds"""
        assert F5_5_PCA_MIN_VARIANCE == 0.70
        assert F5_5_PCA_FALSIFICATION_THRESHOLD == 0.60
        assert F5_5_MIN_LOADING == 0.60

    def test_f5_peak_separation_thresholds(self):
        """Test F5.4 peak separation thresholds"""
        assert F5_4_MIN_PEAK_SEPARATION == 0.12
        assert F5_4_MIN_PROPORTION == 0.65
        assert F5_4_BINOMIAL_ALPHA == 0.01

    def test_f5_proportion_thresholds(self):
        """Test F5 proportion thresholds"""
        assert F5_1_MIN_PROPORTION == 0.75
        assert F5_1_MIN_ALPHA == 4.0
        assert F5_1_MIN_COHENS_D == 0.50
        assert F5_1_BINOMIAL_ALPHA == 0.01

    def test_f2_family_thresholds(self):
        """Test F2 family (IGT/Somatic) thresholds"""
        assert F2_1_MIN_ADVANTAGE_PCT == 22.0
        assert F2_1_MIN_PP_DIFF == 10.0
        assert F2_1_MIN_COHENS_H == 0.55
        assert F2_1_ALPHA == 0.01

    def test_f3_family_thresholds(self):
        """Test F3 family (Advantages) thresholds"""
        assert F3_1_MIN_ADVANTAGE_PCT == 18.0
        assert F3_1_MIN_COHENS_D == 0.60
        assert F3_1_ALPHA == 0.01

    def test_v7_thresholds(self):
        """Test V7 TMS intervention thresholds"""
        assert V7_1_MIN_THRESHOLD_REDUCTION_PCT == 15.0
        assert V7_1_MIN_EFFECT_DURATION_MIN == 60.0
        assert V7_1_MIN_COHENS_D == 0.70
        assert V7_1_ALPHA == 0.01

    def test_v11_thresholds(self):
        """Test V11 model fit thresholds"""
        assert V11_MIN_R2 == 0.75
        assert V11_MIN_DELTA_R2 == 0.10
        assert V11_MIN_COHENS_D == 0.45

    def test_generic_thresholds(self):
        """Test generic validation thresholds"""
        assert GENERIC_MIN_R2 == 0.70
        assert GENERIC_MIN_AUC == 0.70
        assert GENERIC_MIN_CORR == 0.30
        assert GENERIC_MIN_COHENS_D == 0.70
        assert GENERIC_MEDIUM_COHENS_D == 0.50
        assert GENERIC_BINARY_DECISION_THRESHOLD == 0.50

    def test_p1_thresholds(self):
        """Test P1 primary detection thresholds"""
        assert P1_1_MIN_D_PRIME == 0.50
        assert P1_1_MAX_D_PRIME == 0.60
        assert P1_2_AROUSAL_INTERACTION_MIN_D == 0.30
        assert P1_3_IA_BENEFIT_MIN_D == 0.35

    def test_p2_thresholds(self):
        """Test P2 TMS causal prediction thresholds"""
        assert P2_A_MIN_THRESHOLD_SHIFT == 0.12
        assert P2_B_MIN_HEP_REDUCTION == 35.0
        assert P2_B_MIN_PCI_REDUCTION == 25.0
        assert P2_C_MIN_ETA_SQ == 0.12

    def test_p12_thresholds(self):
        """Test P12 cross-species scaling thresholds"""
        assert P12_A_EXPONENT_MIN == 0.72
        assert P12_A_EXPONENT_MAX == 0.78
        assert P12_B_MIN_CONSISTENCY_PCT == 90.0
        assert P12_C_PROPFOLOL_REDUCTION_MIN_PCT == 50.0

    def test_bic_thresholds(self):
        """Test BIC framework thresholds"""
        assert BIC_STRONG_EVIDENCE == 2.0
        assert BIC_VERY_STRONG == 6.0
        assert BIC_FRAMEWORK_THRESHOLD_B == 10.0

    def test_doc_auc_thresholds(self):
        """Test DoC AUC thresholds"""
        assert DOC_AUC_MIN == 0.75
        assert DOC_AUC_MAX == 0.85

    def test_vp2_coupling_thresholds(self):
        """Test VP2 coupling parameters"""
        assert VP2_DELTA_PI_COUPLING == 0.055
        assert VP2_AROUSAL_COUPLING_SCALE == 0.35
        assert VP2_AROUSAL_BOOST_MAX == 0.60

    def test_vp4_calibrated_parameters(self):
        """Test VP4 calibrated parameters"""
        assert VP4_CALIBRATED_TAU == 0.20
        assert VP4_CALIBRATED_THETA_0 == 0.12
        assert VP4_CALIBRATED_ALPHA == 35.0

    def test_v16_metabolic_thresholds(self):
        """Test V16 metabolic ATP thresholds"""
        assert V16_MIN_CORRELATION == 0.75
        assert V16_MIN_EFFICIENCY_GAIN == 0.20
        assert V16_C1_CONSISTENCY_CV == 0.20

    def test_v17_fatigue_thresholds(self):
        """Test V17 fatigue analysis thresholds"""
        assert V17_MIN_R2_P3B_DECAY == 0.70
        assert V17_MIN_R2_THETA_ELEVATION == 0.60
        assert V17_MIN_CORRELATION_MAGNITUDE == 0.10
        assert V17_ALPHA == 0.05

    def test_v12_clinical_thresholds(self):
        """Test V12 clinical gradient thresholds"""
        assert V12_1_MIN_P3B_REDUCTION_PCT == 50.0
        assert V12_1_MIN_IGNITION_REDUCTION_PCT == 50.0
        assert V12_1_MIN_COHENS_D == 0.80
        assert V12_1_MIN_ETA_SQUARED == 0.30
        assert V12_1_ALPHA == 0.05

    def test_p11_fatigue_thresholds(self):
        """Test P11 fatigue threshold dynamics"""
        assert P11_MIN_R2 == 0.70

    def test_f6_sparsity_threshold(self):
        """Test F6 sparsity activation threshold"""
        assert F6_SPARSITY_ACTIVATION_THRESHOLD == 0.7

    def test_f6_delta_auroc(self):
        """Test F6 delta AUROC threshold"""
        assert F6_DELTA_AUROC_MIN == 0.05

    def test_liquid_ignition_threshold(self):
        """Test liquid ignition detection threshold"""
        assert LIQUID_IGNITION_DETECTION_THRESHOLD == 0.50

    def test_generic_alpha(self):
        """Test generic significance level"""
        assert GENERIC_ALPHA == 0.05


class TestThresholdRegistry:
    """Test THRESHOLD_REGISTRY dictionary"""

    def test_registry_exists(self):
        """Test that threshold registry exists"""
        assert isinstance(THRESHOLD_REGISTRY, dict)
        assert len(THRESHOLD_REGISTRY) > 0

    def test_registry_keys(self):
        """Test that registry contains expected keys"""
        expected_keys = [
            "F1.1_ADVANTAGE",
            "F1.1_COHENS_D",
            "F2.1_ADVANTAGE",
            "F2.1_COHENS_H",
            "F3.1_ADVANTAGE",
            "F3.1_COHENS_D",
            "F6.1_LTCN_TRANSITION",
            "F6.2_INTEGRATION_RATIO",
            "V11_MIN_R2",
            "GENERIC_MIN_R2",
            "GENERIC_MIN_AUC",
            "P7_MIN_AUC",
        ]
        for key in expected_keys:
            assert key in THRESHOLD_REGISTRY, f"Missing key: {key}"

    def test_registry_values_positive(self):
        """Test that registry values are positive numbers"""
        for key, value in THRESHOLD_REGISTRY.items():
            assert isinstance(value, (int, float)), f"{key} is not a number"
            assert value > 0, f"{key} has non-positive value: {value}"

    def test_registry_f11_advantage(self):
        """Test F1.1 advantage threshold in registry"""
        assert THRESHOLD_REGISTRY["F1.1_ADVANTAGE"] == F1_1_MIN_ADVANTAGE_PCT

    def test_registry_f21_advantage(self):
        """Test F2.1 advantage threshold in registry"""
        assert THRESHOLD_REGISTRY["F2.1_ADVANTAGE"] == F2_1_MIN_ADVANTAGE_PCT

    def test_registry_p7_min_auc(self):
        """Test P7 min AUC threshold in registry"""
        assert THRESHOLD_REGISTRY["P7_MIN_AUC"] == P7_MIN_AUC


class TestF6ValidationFunctions:
    """Test F6 validation functions"""

    def test_f6_1_passing_case(self):
        """Test F6.1 with passing data"""
        ltcn_times = np.array([30.0, 35.0, 40.0, 32.0, 38.0])  # Fast transitions
        ff_times = np.array([100.0, 110.0, 120.0, 115.0, 105.0])  # Slow transitions

        result = f6_1_validator(ltcn_times, ff_times)
        assert "passed" in result
        assert "cliffs_delta" in result
        assert "p_value" in result

    def test_f6_1_insufficient_data(self):
        """Test F6.1 with insufficient data"""
        with pytest.raises(ValueError, match="at least 2 elements"):
            f6_1_validator(np.array([30.0]), np.array([100.0, 110.0]))

    def test_f6_1_nan_values(self):
        """Test F6.1 with NaN values"""
        with pytest.raises(ValueError, match="NaN"):
            f6_1_validator(
                np.array([30.0, np.nan, 40.0]), np.array([100.0, 110.0, 120.0])
            )

    def test_f6_2_passing_case(self):
        """Test F6.2 with passing data"""
        ltcn_windows = np.array([250.0, 300.0, 280.0, 320.0, 290.0])  # Long windows
        rnn_windows = np.array([50.0, 60.0, 55.0, 58.0, 52.0])  # Short windows

        result = f6_2_validator(ltcn_windows, rnn_windows)
        assert "passed" in result
        assert "ratio" in result
        assert "ltcn_median_window" in result

    def test_f6_2_insufficient_data(self):
        """Test F6.2 with insufficient data"""
        with pytest.raises(ValueError, match="at least 2 elements"):
            f6_2_validator(np.array([250.0]), np.array([50.0, 60.0]))

    def test_f6_3_passing_case(self):
        """Test F6.3 with passing data"""
        ltcn_reductions = np.array([35.0, 40.0, 38.0, 42.0, 36.0])  # High reduction
        standard_reductions = np.array([5.0, 8.0, 6.0, 7.0, 5.0])  # Low reduction

        result = f6_3_validator(ltcn_reductions, standard_reductions)
        assert "passed" in result
        assert "cohens_d" in result
        assert "ltcn_mean_reduction" in result

    def test_f6_3_insufficient_data(self):
        """Test F6.3 with insufficient data"""
        with pytest.raises(ValueError, match="at least 2 elements"):
            f6_3_validator(np.array([35.0]), np.array([5.0, 8.0]))

    def test_f6_4_passing_case(self):
        """Test F6.4 with passing memory decay"""
        result = f6_4_validator(memory_decay_tau=2.0)
        assert "passed" in result
        assert "tau_memory" in result
        assert "r_squared" in result

    def test_f6_4_tau_out_of_bounds(self):
        """Test F6.4 with tau outside valid range"""
        result = f6_4_validator(memory_decay_tau=5.0)  # Too high
        assert result["passed"] is False

    def test_f6_5_passing_case(self):
        """Test F6.5 with passing bifurcation structure"""
        result = f6_5_validator(theta_t=0.5)
        assert "passed" in result
        assert "bifurcation_point" in result
        assert "hysteresis_width" in result

    def test_f6_6_passing_case(self):
        """Test F6.6 with passing alternative architecture data"""
        result = f6_6_validator(
            alternative_modules_needed=3.0, performance_gap_without_addons=20.0
        )
        assert "passed" in result
        assert "modules_pass" in result
        assert "performance_pass" in result

    def test_f6_6_failing_case(self):
        """Test F6.6 with failing alternative architecture data"""
        result = f6_6_validator(
            alternative_modules_needed=1.0,  # Too few modules
            performance_gap_without_addons=10.0,  # Too small gap
        )
        assert result["passed"] is False


class TestThresholdRanges:
    """Test that thresholds are in reasonable ranges"""

    def test_all_f_thresholds_in_valid_range(self):
        """Test that F-prefixed thresholds are in valid ranges (0-100% or reasonable)"""
        f_thresholds = [
            F1_1_MIN_ADVANTAGE_PCT,
            F1_1_MIN_COHENS_D,
            F2_1_MIN_ADVANTAGE_PCT,
            F2_1_MIN_COHENS_H,
            F2_2_MIN_CORR,
            F2_3_MIN_RT_ADVANTAGE_MS,
            F2_3_MIN_R2,
            F3_1_MIN_ADVANTAGE_PCT,
            F3_1_MIN_COHENS_D,
            F3_2_MIN_INTERO_ADVANTAGE_PCT,
            F3_3_MIN_REDUCTION_PCT,
            F5_1_MIN_PROPORTION,
            F5_1_MIN_COHENS_D,
            F5_2_MIN_PROPORTION,
            F5_2_MIN_CORRELATION,
            F5_5_PCA_MIN_VARIANCE,
            F6_1_LTCN_MAX_TRANSITION_MS,
            F6_2_MIN_INTEGRATION_RATIO,
            F6_2_MIN_R2,
        ]

        for threshold in f_thresholds:
            assert threshold > 0, f"Threshold {threshold} is not positive"

    def test_alpha_values_in_valid_range(self):
        """Test that alpha values are in valid statistical range (0-1)"""
        alpha_values = [
            F1_1_ALPHA,
            F2_1_ALPHA,
            F2_2_ALPHA,
            F2_3_ALPHA,
            F2_4_ALPHA,
            F2_5_ALPHA,
            F3_1_ALPHA,
            F3_2_ALPHA,
            F3_3_ALPHA,
            F3_4_ALPHA,
            F3_6_ALPHA,
            F5_1_BINOMIAL_ALPHA,
            F5_2_BINOMIAL_ALPHA,
            V7_1_ALPHA,
            V7_2_ALPHA,
            V12_1_ALPHA,
            V17_ALPHA,
            GENERIC_ALPHA,
        ]

        for alpha in alpha_values:
            assert 0 < alpha < 1, f"Alpha value {alpha} is not in valid range (0, 1)"

    def test_r2_thresholds_in_valid_range(self):
        """Test that R² thresholds are in valid range (0-1)"""
        r2_thresholds = [
            F2_3_MIN_R2,
            F6_2_MIN_R2,
            V11_MIN_R2,
            GENERIC_MIN_R2,
            P11_MIN_R2,
            V17_MIN_R2_P3B_DECAY,
            V17_MIN_R2_THETA_ELEVATION,
            F8_IDENTIFIABILITY_MIN_R2,
        ]

        for r2 in r2_thresholds:
            assert 0 < r2 <= 1, f"R² threshold {r2} is not in valid range (0, 1]"

    def test_auc_thresholds_in_valid_range(self):
        """Test that AUC thresholds are in valid range (0.5-1)"""
        auc_thresholds = [
            P7_MIN_AUC,
            GENERIC_MIN_AUC,
            DOC_AUC_MIN,
            DOC_AUC_MAX,
        ]

        for auc in auc_thresholds:
            assert 0.5 < auc <= 1, f"AUC threshold {auc} is not in valid range (0.5, 1]"


class TestPaperVsSimulationVariants:
    """Test paper spec vs simulation variant threshold differences"""

    def test_f5_pca_variance_variants(self):
        """Test F5.5 PCA variance paper vs simulation variants exist"""
        from utils.falsification_thresholds import (
            F5_5_PCA_MIN_VARIANCE_PAPER_SPEC,
            F5_5_PCA_MIN_VARIANCE_SIMULATION,
        )

        assert F5_5_PCA_MIN_VARIANCE_PAPER_SPEC == 0.70
        assert F5_5_PCA_MIN_VARIANCE_SIMULATION == 0.60
        assert F5_5_PCA_MIN_VARIANCE == F5_5_PCA_MIN_VARIANCE_PAPER_SPEC

    def test_f5_1_proportion_variants(self):
        """Test F5.1 proportion paper vs simulation variants exist"""
        from utils.falsification_thresholds import (
            F5_1_MIN_PROPORTION_PAPER_SPEC,
            F5_1_MIN_PROPORTION_SIMULATION,
        )

        assert F5_1_MIN_PROPORTION_PAPER_SPEC == 0.75
        assert F5_1_MIN_PROPORTION_SIMULATION == 0.70
        assert F5_1_MIN_PROPORTION == F5_1_MIN_PROPORTION_PAPER_SPEC

    def test_p1_d_prime_variants(self):
        """Test P1 d' paper vs simulation variants exist"""
        from utils.falsification_thresholds import (
            P1_1_MIN_D_PRIME_PAPER_SPEC,
            P1_1_MIN_D_PRIME_SIMULATION,
        )

        assert P1_1_MIN_D_PRIME_PAPER_SPEC == 0.50
        assert P1_1_MIN_D_PRIME_SIMULATION == 0.40
        assert P1_1_MIN_D_PRIME == P1_1_MIN_D_PRIME_PAPER_SPEC

    def test_f8_sobol_variants(self):
        """Test F8 Sobol sensitivity paper vs simulation variants exist"""
        from utils.falsification_thresholds import (
            F8_SOBOL_MIN_SENSITIVITY_PAPER_SPEC,
            F8_SOBOL_MIN_SENSITIVITY_SIMULATION,
        )

        assert F8_SOBOL_MIN_SENSITIVITY_PAPER_SPEC == 0.15
        assert F8_SOBOL_MIN_SENSITIVITY_SIMULATION == 0.10
        assert F8_SOBOL_MIN_SENSITIVITY == F8_SOBOL_MIN_SENSITIVITY_PAPER_SPEC
