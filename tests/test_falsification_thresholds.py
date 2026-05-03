import numpy as np
import pytest

from utils.falsification_thresholds import (
    F6_1_LTCN_MAX_TRANSITION_MS,
    F6_2_LTCN_MIN_WINDOW_MS,
)
from utils.falsification_thresholds import (
    test_f6_1_intrinsic_threshold_behavior as _test_f6_1,
)
from utils.falsification_thresholds import (
    test_f6_2_intrinsic_temporal_integration as _test_f6_2,
)
from utils.falsification_thresholds import test_f6_3_metabolic_selectivity as _test_f6_3


def test_f6_1_intrinsic_threshold_behavior_pass():
    # Pass condition: LTCN <= 50ms, large difference
    ltcn = np.array([30.0, 35.0, 40.0, 32.0, 45.0])
    ff = np.array([100.0, 110.0, 120.0, 105.0, 115.0])

    result = _test_f6_1(ltcn, ff)

    assert "passed" in result
    assert result["ltcn_median_time"] <= F6_1_LTCN_MAX_TRANSITION_MS
    assert result["cliffs_delta"] > 0
    assert result["p_value"] < 0.05


def test_f6_1_intrinsic_threshold_behavior_fail_time():
    # Fail condition: LTCN > 50ms
    ltcn = np.array([60.0, 65.0, 70.0, 62.0, 75.0])
    ff = np.array([100.0, 110.0, 120.0, 105.0, 115.0])

    result = _test_f6_1(ltcn, ff)
    assert result["passed"] is False


def test_f6_1_validation_errors():
    with pytest.raises(ValueError, match="at least 2 elements"):
        _test_f6_1(np.array([1.0]), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="at least 2 elements"):
        _test_f6_1(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError, match="NaN or Inf"):
        _test_f6_1(np.array([1.0, np.nan]), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="NaN or Inf"):
        _test_f6_1(np.array([1.0, 2.0]), np.array([1.0, np.inf]))


def test_f6_2_intrinsic_temporal_integration_pass():
    # Pass condition: LTCN >= 200ms, Ratio >= 4.0
    ltcn = np.array([250.0, 260.0, 240.0, 270.0, 255.0])
    rnn = np.array([40.0, 45.0, 50.0, 35.0, 42.0])

    result = _test_f6_2(ltcn, rnn)

    assert bool(result["passed"]) is True
    assert result["ltcn_median_window"] >= F6_2_LTCN_MIN_WINDOW_MS
    assert result["ratio"] >= 4.0
    assert result["p_value"] < 0.05


def test_f6_2_intrinsic_temporal_integration_fail_ratio():
    # Fail condition: Ratio < 4.0
    ltcn = np.array([250.0, 260.0, 240.0, 270.0, 255.0])
    rnn = np.array([100.0, 110.0, 120.0, 105.0, 115.0])

    result = _test_f6_2(ltcn, rnn)
    assert result["passed"] is False


def test_f6_2_validation_errors():
    with pytest.raises(ValueError, match="at least 2 elements"):
        _test_f6_2(np.array([1.0]), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="at least 2 elements"):
        _test_f6_2(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError, match="NaN or Inf"):
        _test_f6_2(np.array([1.0, np.nan]), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="NaN or Inf"):
        _test_f6_2(np.array([1.0, 2.0]), np.array([1.0, np.inf]))


def test_f6_3_metabolic_selectivity_pass():
    # Pass: LTCN reduction >= 30%, Standard <= 10%
    ltcn = np.array([35.0, 40.0, 32.0, 38.0, 36.0])
    standard = np.array([5.0, 8.0, 4.0, 6.0, 7.0])

    result = _test_f6_3(ltcn, standard)

    assert result["passed"] is True
    assert result["p_value"] < 0.01


def test_f6_3_metabolic_selectivity_fail():
    # Fail: LTCN reduction < 30%
    ltcn = np.array([20.0, 25.0, 22.0, 28.0, 26.0])
    standard = np.array([5.0, 8.0, 4.0, 6.0, 7.0])

    result = _test_f6_3(ltcn, standard)
    assert result["passed"] is False


def test_f6_3_validation_errors():
    with pytest.raises(ValueError, match="at least 2 elements"):
        _test_f6_3(np.array([1.0]), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="at least 2 elements"):
        _test_f6_3(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError, match="NaN or Inf"):
        _test_f6_3(np.array([1.0, np.nan]), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="NaN or Inf"):
        _test_f6_3(np.array([1.0, 2.0]), np.array([1.0, np.inf]))
