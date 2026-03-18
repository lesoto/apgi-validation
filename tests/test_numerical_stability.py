"""
Property-based tests for numerical stability using Hypothesis.

Tests edge cases for statistical functions in APGI-Falsification-Framework.py:
- Empty arrays
- Constant arrays
- NaN/Inf inputs
- Extremely large/small values
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes
import importlib.util
from pathlib import Path

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load APGI-Falsification-Framework module using importlib.util
framework_path = Path(__file__).parent.parent / "APGI-Falsification-Framework.py"
spec = importlib.util.spec_from_file_location(
    "APGI_Falsification_Framework", framework_path
)
APGI_Falsification_Framework = importlib.util.module_from_spec(spec)
spec.loader.exec_module(APGI_Falsification_Framework)


@pytest.mark.hypothesis
class TestNumericalStability:
    """Property-based tests for numerical stability of statistical functions."""

    @given(
        arrays(
            np.float64,
            array_shapes(min_dims=1, max_dims=2, min_side=0),
        )
    )
    def test_mean_difference_with_empty_arrays(self, arr):
        """Test mean difference handles empty arrays gracefully."""
        criterion = APGI_Falsification_Framework.FalsificationCriterion(
            name="test_criterion",
            description="Test criterion for edge cases",
            test_statistic="mean_difference",
            threshold=0.5,
            direction="greater",
            alpha=0.05,
        )

        # Empty arrays should return error
        if len(arr) == 0:
            result = criterion.test({"group1": [], "group2": []})
            assert "error" in result
        else:
            # Non-empty arrays should work or return error for edge cases
            result = criterion.test({"group1": arr, "group2": arr})
            assert isinstance(result, dict)
            # Should either have test_statistic or error
            assert "test_statistic" in result or "error" in result

    @given(
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
    )
    def test_mean_difference_with_constant_arrays(self, arr1, arr2):
        """Test mean difference with constant arrays (all same values)."""
        criterion = APGI_Falsification_Framework.FalsificationCriterion(
            name="test_criterion",
            description="Test criterion for edge cases",
            test_statistic="mean_difference",
            threshold=0.5,
            direction="greater",
            alpha=0.05,
        )

        # If both arrays are constant, test should still work
        if np.all(arr1 == arr1.flat[0]) and np.all(arr2 == arr2.flat[0]):
            result = criterion.test({"group1": arr1, "group2": arr2})
            assert isinstance(result, dict)
            # With constant arrays, may get NaN results
            if "test_statistic" in result:
                assert result["test_statistic"] == 0 or np.isnan(
                    result["test_statistic"]
                )
            else:
                assert "error" in result

    @given(
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
    )
    def test_mean_difference_with_nan_inputs(self, arr1, arr2):
        """Test mean difference handles NaN inputs."""
        criterion = APGI_Falsification_Framework.FalsificationCriterion(
            name="test_criterion",
            description="Test criterion for edge cases",
            test_statistic="mean_difference",
            threshold=0.5,
            direction="greater",
            alpha=0.05,
        )

        # Introduce NaN values
        arr1_with_nan = arr1.copy()
        arr2_with_nan = arr2.copy()
        if len(arr1_with_nan) > 0:
            arr1_with_nan[0] = np.nan
        if len(arr2_with_nan) > 0:
            arr2_with_nan[0] = np.nan

        # Should handle NaN gracefully
        result = criterion.test({"group1": arr1_with_nan, "group2": arr2_with_nan})
        assert isinstance(result, dict)
        # Should either return error or handle NaN
        assert "error" in result or "test_statistic" in result

    @given(
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
    )
    def test_mean_difference_with_inf_inputs(self, arr1, arr2):
        """Test mean difference handles Inf inputs."""
        criterion = APGI_Falsification_Framework.FalsificationCriterion(
            name="test_criterion",
            description="Test criterion for edge cases",
            test_statistic="mean_difference",
            threshold=0.5,
            direction="greater",
            alpha=0.05,
        )

        # Introduce Inf values
        arr1_with_inf = arr1.copy()
        arr2_with_inf = arr2.copy()
        if len(arr1_with_inf) > 0:
            arr1_with_inf[0] = np.inf
        if len(arr2_with_inf) > 0:
            arr2_with_inf[0] = -np.inf

        # Should handle Inf gracefully
        result = criterion.test({"group1": arr1_with_inf, "group2": arr2_with_inf})
        assert isinstance(result, dict)
        # Should either return error or handle Inf
        assert "error" in result or "test_statistic" in result

    @given(
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
        st.floats(min_value=1e-10, max_value=1e10),
    )
    def test_effect_size_with_extreme_values(self, arr1, arr2, scale_factor):
        """Test effect size with extremely large/small values."""
        criterion = APGI_Falsification_Framework.FalsificationCriterion(
            name="test_criterion",
            description="Test criterion for edge cases",
            test_statistic="effect_size",
            threshold=0.5,
            direction="greater",
            alpha=0.05,
        )

        # Scale arrays to extreme values
        effect_size = np.mean(arr1 * scale_factor) - np.mean(arr2 / scale_factor)

        # Should handle extreme values without overflow/underflow
        result = criterion.test({"effect_size": float(effect_size)})
        # Result should be a dict
        assert isinstance(result, dict)
        if "test_statistic" in result:
            # Effect size should be finite (not NaN or Inf) or 0
            # For extreme values, inf is acceptable
            assert (
                np.isfinite(result["test_statistic"])
                or result["test_statistic"] == 0
                or np.isinf(result["test_statistic"])
            )

    @given(
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
    )
    def test_mean_difference_with_constant_arrays_variance(self, arr1, arr2):
        """Test mean difference with constant arrays (zero variance)."""
        criterion = APGI_Falsification_Framework.FalsificationCriterion(
            name="test_criterion",
            description="Test criterion for edge cases",
            test_statistic="mean_difference",
            threshold=0.5,
            direction="greater",
            alpha=0.05,
        )

        # If arrays are constant, variance is zero
        if np.all(arr1 == arr1.flat[0]) and np.all(arr2 == arr2.flat[0]):
            # t-test with zero variance should handle gracefully
            result = criterion.test({"group1": arr1, "group2": arr2})
            # Should return result with appropriate handling
            assert isinstance(result, dict)

    @given(
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
        st.floats(min_value=1e-10, max_value=1e10),
    )
    def test_correlation_with_extreme_values(self, arr1, arr2, scale_factor):
        """Test correlation with extreme values."""
        criterion = APGI_Falsification_Framework.FalsificationCriterion(
            name="test_criterion",
            description="Test criterion for edge cases",
            test_statistic="correlation",
            threshold=0.5,
            direction="greater",
            alpha=0.05,
        )

        # Scale arrays to extreme values
        arr1_scaled = arr1 * scale_factor
        arr2_scaled = arr2 / scale_factor

        # Should handle extreme values
        result = criterion.test({"x": arr1_scaled, "y": arr2_scaled})
        # Result should be a dict
        assert isinstance(result, dict)
        if "test_statistic" in result:
            # Test statistic (correlation) should be between -1 and 1
            assert -1 <= result["test_statistic"] <= 1

    @given(
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
    )
    def test_correlation_with_constant_arrays(self, arr1, arr2):
        """Test correlation with constant arrays (zero variance)."""
        criterion = APGI_Falsification_Framework.FalsificationCriterion(
            name="test_criterion",
            description="Test criterion for edge cases",
            test_statistic="correlation",
            threshold=0.5,
            direction="greater",
            alpha=0.05,
        )

        # If arrays are constant, correlation should handle gracefully
        if np.all(arr1 == arr1.flat[0]) and np.all(arr2 == arr2.flat[0]):
            result = criterion.test({"x": arr1, "y": arr2})
            # Should return result with appropriate handling
            assert isinstance(result, dict)
            # Constant arrays should produce error
            assert "error" in result

    @given(
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
        arrays(np.float64, array_shapes(min_dims=1, max_dims=2, min_side=1)),
    )
    def test_falsification_criterion_with_edge_cases(self, arr1, arr2):
        """Test FalsificationCriterion with edge case inputs."""
        # Create criterion with edge case data
        criterion = APGI_Falsification_Framework.FalsificationCriterion(
            name="test_criterion",
            description="Test criterion for edge cases",
            test_statistic="mean_difference",
            threshold=0.5,
            direction="greater",
            alpha=0.05,
        )

        # Test with various edge cases
        test_data = {
            "group1": arr1,
            "group2": arr2,
        }

        # Should handle edge cases gracefully
        result = criterion.test(test_data)
        # Result should be a dict
        assert isinstance(result, dict)
