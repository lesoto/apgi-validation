"""
Test that falsification_thresholds.py (root) and utils/falsification_thresholds.py
define identical values for the same constant names.
"""

import pytest


def test_cross_file_threshold_consistency():
    """
    Test that root falsification_thresholds.py and utils/falsification_thresholds.py
    define identical values for the same constant names.
    """
    # Import both threshold files
    import falsification_thresholds as root_thresholds
    from utils import falsification_thresholds as utils_thresholds

    # Get all constants from root file (excluding private/internal ones)
    root_constants = {
        name: value
        for name, value in vars(root_thresholds).items()
        if not name.startswith("_")
        and isinstance(value, (int, float, str))
        and not callable(value)
    }

    # Get all constants from utils file
    utils_constants = {
        name: value
        for name, value in vars(utils_thresholds).items()
        if not name.startswith("_")
        and isinstance(value, (int, float, str))
        and not callable(value)
    }

    # Find common constants between both files
    common_constants = set(root_constants.keys()) & set(utils_constants.keys())

    # Assert that all common constants have identical values
    mismatches = []
    for const_name in common_constants:
        root_value = root_constants[const_name]
        utils_value = utils_constants[const_name]

        if root_value != utils_value:
            mismatches.append(
                const_name + ": root=" + str(root_value) + ", utils=" + str(utils_value)
            )

    if mismatches:
        pytest.fail(
            "Threshold value mismatches between root and utils files:\n"
            + "\n".join(mismatches)
        )

    # Report number of common constants verified
    print(
        "Verified "
        + str(len(common_constants))
        + " common threshold constants are consistent"
    )  # noqa: E501


def test_all_threshold_constants_exported_from_utils():
    """Test that all important threshold constants are exported from utils/__init__.py."""
    from utils import (  # noqa: F401
        F1_1_MIN_ADVANTAGE_PCT,
        F1_1_MIN_COHENS_D,
        F1_1_ALPHA,
        F1_5_PAC_MI_MIN,
        F1_5_PAC_INCREASE_MIN,
        F1_5_COHENS_D_MIN,
        F1_5_PERMUTATION_ALPHA,
        F2_1_MIN_ADVANTAGE_PCT,
        F2_1_MIN_PP_DIFF,
        F2_1_MIN_COHENS_H,
        F2_1_ALPHA,
        F2_2_MIN_CORR,
        F2_2_MIN_FISHER_Z,
        F2_2_ALPHA,
        F2_3_MIN_RT_ADVANTAGE_MS,
        F2_3_MIN_BETA,
        F2_3_MIN_STANDARDIZED_BETA,
        F2_3_MIN_R2,
        F2_3_ALPHA,
        F2_4_MIN_CONFIDENCE_EFFECT_PCT,
        F2_4_MIN_BETA_INTERACTION,
        F2_4_ALPHA,
        F5_1_MIN_PROPORTION,
        F5_1_MIN_ALPHA,
        F5_1_MIN_COHENS_D,
        F5_2_MIN_PROPORTION,
        F5_2_MIN_CORRELATION,
        F5_3_MIN_PROPORTION,
        F5_3_MIN_GAIN_RATIO,
        F5_3_MIN_COHENS_D,
        F5_3_FALSIFICATION_RATIO,
        F5_4_MIN_PROPORTION,
        F5_4_MIN_PEAK_SEPARATION,
        F5_5_PCA_MIN_VARIANCE,
        F5_5_MIN_LOADING,
        F5_6_MIN_PERFORMANCE_DIFF_PCT,
        F5_6_MIN_COHENS_D,
        F5_6_ALPHA,
        F6_1_LTCN_MAX_TRANSITION_MS,
        F6_1_CLIFFS_DELTA_MIN,
        F6_1_MANN_WHITNEY_ALPHA,
        F6_2_LTCN_MIN_WINDOW_MS,
        F6_2_MIN_INTEGRATION_RATIO,
        F6_2_MIN_CURVE_FIT_R2,
        F6_2_MIN_R2,
        F6_2_WILCOXON_ALPHA,
        V12_1_MIN_P3B_REDUCTION_PCT,
        V12_1_MIN_IGNITION_REDUCTION_PCT,
        V12_1_MIN_COHENS_D,
        V12_1_MIN_ETA_SQUARED,
        V12_1_ALPHA,
        V12_2_MIN_CORRELATION,
        V12_2_FALSIFICATION_CORR,
        V12_2_MIN_PILLAIS_TRACE,
        V12_2_FALSIFICATION_PILLAIS,
        V12_2_ALPHA,
        F6_5_HYSTERESIS_MIN,
        F6_5_HYSTERESIS_MAX,
    )

    # Verify they're all accessible and have correct types
    assert isinstance(F1_1_MIN_ADVANTAGE_PCT, float)
    assert isinstance(F1_1_MIN_COHENS_D, float)
    assert isinstance(F1_1_ALPHA, float)
    assert isinstance(F6_2_MIN_R2, float)
    assert isinstance(F5_3_FALSIFICATION_RATIO, float)

    # Verify key values match root file
    from falsification_thresholds import (
        F6_2_MIN_R2 as root_F6_2_MIN_R2,
        F5_3_FALSIFICATION_RATIO as root_F5_3_FALSIFICATION_RATIO,
    )

    assert F6_2_MIN_R2 == root_F6_2_MIN_R2
    assert F5_3_FALSIFICATION_RATIO == root_F5_3_FALSIFICATION_RATIO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
