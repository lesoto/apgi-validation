#!/usr/bin/env python3
"""
Unit Test Suite for Threshold Consistency

Validates that falsification_thresholds.py constants match paper specifications.
"""

import pytest


def test_f5_1_constants():
    """Test F5.1 constants match paper specifications"""
    from utils.falsification_thresholds import (
        F5_1_MIN_PROPORTION,
        F5_1_MIN_ALPHA,
        F5_5_PCA_MIN_VARIANCE,
        F5_5_MIN_LOADING,
        F5_4_MIN_PEAK_SEPARATION,
    )

    # Test proportion threshold - should be 0.75 per paper spec
    assert (
        F5_1_MIN_PROPORTION == 0.75
    ), f"F5_1_MIN_PROPORTION should be 0.75, got {F5_1_MIN_PROPORTION}"

    # Test alpha threshold - should be 4.0 per paper spec
    assert F5_1_MIN_ALPHA == 4.0, f"F5_1_MIN_ALPHA should be 4.0, got {F5_1_MIN_ALPHA}"

    # Test PCA variance threshold - should be 0.70 per paper spec
    assert (
        F5_5_PCA_MIN_VARIANCE == 0.70
    ), f"F5_5_PCA_MIN_VARIANCE should be 0.70, got {F5_5_PCA_MIN_VARIANCE}"

    # Test loading threshold - should be 0.60 per paper spec
    assert (
        F5_5_MIN_LOADING == 0.60
    ), f"F5_5_MIN_LOADING should be 0.60, got {F5_5_MIN_LOADING}"

    # Test peak separation threshold - should be 3.0 per paper spec
    assert (
        F5_4_MIN_PEAK_SEPARATION == 3.0
    ), f"F5_4_MIN_PEAK_SEPARATION should be 3.0, got {F5_4_MIN_PEAK_SEPARATION}"


def test_f6_1_constants():
    """Test F6.1 constants match paper specifications"""
    from utils.falsification_thresholds import F6_1_LTCN_MAX_TRANSITION_MS

    # Test LTCN transition threshold - should be 50.0ms per paper spec
    assert (
        F6_1_LTCN_MAX_TRANSITION_MS == 50.0
    ), f"F6_1_LTCN_MAX_TRANSITION_MS should be 50.0, got {F6_1_LTCN_MAX_TRANSITION_MS}"


def test_v12_1_constants():
    """Test V12.1 constants match paper specifications"""
    from utils.falsification_thresholds import (
        V12_1_MIN_P3B_REDUCTION_PCT,
        V12_1_MIN_IGNITION_REDUCTION_PCT,
        V12_1_MIN_COHENS_D,
        V12_2_MIN_CORRELATION,
    )

    # Test P3b reduction threshold - should be 80.0% per paper spec
    assert (
        V12_1_MIN_P3B_REDUCTION_PCT == 80.0
    ), f"V12_1_MIN_P3B_REDUCTION_PCT should be 80.0, got {V12_1_MIN_P3B_REDUCTION_PCT}"

    # Test ignition reduction threshold - should be 70.0% per paper spec
    assert (
        V12_1_MIN_IGNITION_REDUCTION_PCT == 70.0
    ), f"V12_1_MIN_IGNITION_REDUCTION_PCT should be 70.0, got {V12_1_MIN_IGNITION_REDUCTION_PCT}"

    # Test coherence threshold - should be 0.40 per paper spec
    assert (
        V12_1_MIN_COHENS_D == 0.40
    ), f"V12_1_MIN_COHENS_D should be 0.40, got {V12_1_MIN_COHENS_D}"

    # Test correlation threshold - should be 0.60 per paper spec
    assert (
        V12_2_MIN_CORRELATION == 0.60
    ), f"V12_2_MIN_CORRELATION should be 0.60, got {V12_2_MIN_CORRELATION}"


def test_f1_f3_criteria_consistency():
    """Test that F1-F3 criteria are consistent across Protocols 1, 2, 3, 6, 9, and 12"""
    from utils.falsification_thresholds import (
        F1_1_MIN_ADVANTAGE_PCT,
        F1_1_MIN_COHENS_D,
        F1_1_ALPHA,
        F2_1_MIN_ADVANTAGE_PCT,
        F2_1_MIN_PP_DIFF,
        F2_1_MIN_COHENS_H,
        F2_1_ALPHA,
        F2_2_MIN_CORR,
        F2_2_MIN_FISHER_Z,
        F2_2_ALPHA,
        F2_3_MIN_RT_ADVANTAGE_MS,
        F2_3_MIN_BETA,
        F2_3_ALPHA,
        F2_4_MIN_CONFIDENCE_EFFECT_PCT,
        F2_4_MIN_BETA_INTERACTION,
        F2_4_ALPHA,
        F2_5_MAX_TRIALS,
        F2_5_MIN_HAZARD_RATIO,
        F2_5_MIN_TRIAL_ADVANTAGE,
        F2_5_ALPHA,
        F3_1_MIN_ADVANTAGE_PCT,
        F3_1_MIN_COHENS_D,
        F3_1_ALPHA,
        F3_2_MIN_INTERO_ADVANTAGE_PCT,
        F3_2_MIN_COHENS_D,
        F3_2_ALPHA,
        F3_3_MIN_REDUCTION_PCT,
        F3_3_MIN_COHENS_D,
        F3_3_ALPHA,
        F3_4_MIN_REDUCTION_PCT,
        F3_4_MIN_COHENS_D,
        F3_4_ALPHA,
        F3_6_MAX_TRIALS,
        F3_6_MIN_HAZARD_RATIO,
        F3_6_ALPHA,
    )

    # Test F1 family consistency
    assert F1_1_MIN_ADVANTAGE_PCT == 18.0, "F1.1_MIN_ADVANTAGE_PCT should be 18.0%"
    assert F1_1_MIN_COHENS_D == 0.60, "F1.1_MIN_COHENS_D should be 0.60"
    assert F1_1_ALPHA == 0.01, "F1.1_ALPHA should be 0.01"

    # Test F2 family consistency
    assert F2_1_MIN_ADVANTAGE_PCT == 22.0, "F2.1_MIN_ADVANTAGE_PCT should be 22.0%"
    assert F2_1_MIN_PP_DIFF == 10.0, "F2.1_MIN_PP_DIFF should be 10.0pp"
    assert F2_1_MIN_COHENS_H == 0.55, "F2.1_MIN_COHENS_H should be 0.55"
    assert F2_1_ALPHA == 0.01, "F2.1_ALPHA should be 0.01"

    assert F2_2_MIN_CORR == 0.40, "F2.2_MIN_CORR should be 0.40"
    assert F2_2_MIN_FISHER_Z == 1.80, "F2.2_MIN_FISHER_Z should be 1.80"
    assert F2_2_ALPHA == 0.01, "F2.2_ALPHA should be 0.01"

    assert F2_3_MIN_RT_ADVANTAGE_MS == 50.0, "F2.3_MIN_RT_ADVANTAGE_MS should be 50.0ms"
    assert F2_3_MIN_BETA == 25.0, "F2.3_MIN_BETA should be 25.0ms/unit"
    assert F2_3_ALPHA == 0.01, "F2.3_ALPHA should be 0.01"

    assert (
        F2_4_MIN_CONFIDENCE_EFFECT_PCT == 30.0
    ), "F2.4_MIN_CONFIDENCE_EFFECT_PCT should be 30.0%"
    assert F2_4_MIN_BETA_INTERACTION == 0.35, "F2.4_MIN_BETA_INTERACTION should be 0.35"
    assert F2_4_ALPHA == 0.01, "F2.4_ALPHA should be 0.01"

    assert F2_5_MAX_TRIALS == 55.0, "F2.5_MAX_TRIALS should be 55.0"
    assert F2_5_MIN_HAZARD_RATIO == 1.65, "F2.5_MIN_HAZARD_RATIO should be 1.65"
    assert F2_5_MIN_TRIAL_ADVANTAGE == 12.0, "F2.5_MIN_TRIAL_ADVANTAGE should be 12.0"
    assert F2_5_ALPHA == 0.01, "F2.5_ALPHA should be 0.01"

    # Test F3 family consistency
    assert F3_1_MIN_ADVANTAGE_PCT == 18.0, "F3.1_MIN_ADVANTAGE_PCT should be 18.0%"
    assert F3_1_MIN_COHENS_D == 0.60, "F3.1_MIN_COHENS_D should be 0.60"
    assert F3_1_ALPHA == 0.01, "F3.1_ALPHA should be 0.01"

    assert (
        F3_2_MIN_INTERO_ADVANTAGE_PCT == 28.0
    ), "F3.2_MIN_INTERO_ADVANTAGE_PCT should be 28.0%"
    assert F3_2_MIN_COHENS_D == 0.70, "F3.2_MIN_COHENS_D should be 0.70"
    assert F3_2_ALPHA == 0.01, "F3.2_ALPHA should be 0.01"

    assert F3_3_MIN_REDUCTION_PCT == 25.0, "F3.3_MIN_REDUCTION_PCT should be 25.0%"
    assert F3_3_MIN_COHENS_D == 0.75, "F3.3_MIN_COHENS_D should be 0.75"
    assert F3_3_ALPHA == 0.01, "F3.3_ALPHA should be 0.01"

    assert F3_4_MIN_REDUCTION_PCT == 20.0, "F3.4_MIN_REDUCTION_PCT should be 20.0%"
    assert F3_4_MIN_COHENS_D == 0.65, "F3.4_MIN_COHENS_D should be 0.65"
    assert F3_4_ALPHA == 0.01, "F3.4_ALPHA should be 0.01"

    assert F3_6_MAX_TRIALS == 200.0, "F3.6_MAX_TRIALS should be 200.0"
    assert F3_6_MIN_HAZARD_RATIO == 1.45, "F3.6_MIN_HAZARD_RATIO should be 1.45"
    assert F3_6_ALPHA == 0.01, "F3.6_ALPHA should be 0.01"


if __name__ == "__main__":
    pytest.main([__file__])
