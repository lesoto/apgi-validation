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
    from utils import (
        F1_1_ALPHA,
        F1_1_MIN_ADVANTAGE_PCT,
        F1_1_MIN_COHENS_D,
        F5_3_FALSIFICATION_RATIO,
        F6_2_MIN_R2,
    )

    # Verify they're all accessible and have correct types
    assert isinstance(F1_1_MIN_ADVANTAGE_PCT, float)
    assert isinstance(F1_1_MIN_COHENS_D, float)
    assert isinstance(F1_1_ALPHA, float)
    assert isinstance(F6_2_MIN_R2, float)
    assert isinstance(F5_3_FALSIFICATION_RATIO, float)

    # Verify key values match root file
    from utils.falsification_thresholds import (
        F5_3_FALSIFICATION_RATIO as root_F5_3_FALSIFICATION_RATIO,
    )
    from utils.falsification_thresholds import F6_2_MIN_R2 as root_F6_2_MIN_R2

    assert F6_2_MIN_R2 == root_F6_2_MIN_R2
    assert F5_3_FALSIFICATION_RATIO == root_F5_3_FALSIFICATION_RATIO


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
