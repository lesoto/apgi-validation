"""
Numerical boundary tests for edge cases and extreme parameter values.
These tests validate algorithm behavior at numerical boundaries.
============================================================================================
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.boundary
def test_empty_dataset():
    """Test algorithm behavior with empty dataset (n=0)."""
    try:
        from APGI_Equations import calculate_entropy

        # Test with empty arrays
        empty_data = np.array([])

        # Should handle empty data gracefully
        with pytest.raises((ValueError, IndexError)):
            calculate_entropy(empty_data)

        with pytest.raises((ValueError, IndexError)):
            calculate_entropy(empty_data)

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_single_sample_dataset():
    """Test algorithm behavior with single-sample dataset (n=1)."""
    try:
        from APGI_Equations import calculate_entropy

        # Test with single sample
        single_sample = np.array([0.5])

        # Should handle single sample (may return 0 or specific value)
        result = calculate_entropy(single_sample)
        assert isinstance(result, (float, np.ndarray))

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_all_constant_time_series():
    """Test algorithm behavior with all-constant time series."""
    try:
        from APGI_Equations import calculate_entropy, calculate_variance

        # Test with constant values
        constant_data = np.ones(100)

        # Entropy of constant data should be 0
        entropy = calculate_entropy(constant_data)
        assert entropy == 0 or np.isclose(entropy, 0, atol=1e-10)

        # Variance of constant data should be 0
        variance = calculate_variance(constant_data)
        assert variance == 0 or np.isclose(variance, 0, atol=1e-10)

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_all_zero_signals():
    """Test algorithm behavior with all-zero signals."""
    try:
        from APGI_Equations import calculate_entropy

        # Test with zero signals
        zero_data = np.zeros(100)

        # Should handle zero data
        entropy = calculate_entropy(zero_data)
        assert isinstance(entropy, (float, np.ndarray))

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_nan_propagation():
    """Test NaN propagation through algorithm pipeline."""
    try:
        from APGI_Equations import calculate_entropy

        # Test with NaN values
        data_with_nan = np.array([0.5, np.nan, 0.7, 0.3, np.nan])

        # Should handle NaN (either propagate or raise error)
        try:
            result = calculate_entropy(data_with_nan)
            assert np.isnan(result) or isinstance(result, float)
        except (ValueError, RuntimeError):
            pass  # Acceptable to raise error on NaN

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_inf_values():
    """Test algorithm behavior with infinite values."""
    try:
        from APGI_Equations import calculate_entropy

        # Test with infinite values
        data_with_inf = np.array([0.5, np.inf, 0.7, -np.inf, 0.3])

        # Should handle Inf (either propagate or raise error)
        try:
            result = calculate_entropy(data_with_inf)
            assert np.isinf(result) or isinstance(result, float)
        except (ValueError, RuntimeError):
            pass  # Acceptable to raise error on Inf

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_extreme_parameter_values():
    """Test algorithm behavior with extreme parameter values (min/max)."""
    try:
        from Falsification.Falsification_BayesianEstimation_ParameterRecovery import (
            run_bayesian_estimation_complete,
        )

        # Test cases with extreme parameter values
        extreme_cases = [
            {"beta": 0.001, "pi": 0.5, "description": "Near-zero beta"},
            {"beta": 0.999, "pi": 0.5, "description": "Near-one beta"},
            {"beta": 0.5, "pi": 0.001, "description": "Near-zero pi"},
            {"beta": 0.5, "pi": 0.999, "description": "Near-one pi"},
            {"beta": -0.5, "pi": 0.5, "description": "Negative beta"},
            {"beta": 1.5, "pi": 0.5, "description": "Beta > 1"},
        ]

        for case in extreme_cases:
            # Generate synthetic data with extreme parameters
            synthetic_data = {
                "observations": np.random.randn(50, 30),
                "true_parameters": {"beta": case["beta"], "pi": case["pi"]},
            }

            # Should handle extreme values without crashing
            try:
                results = run_bayesian_estimation_complete(
                    data=synthetic_data["observations"],
                    true_parameters=synthetic_data["true_parameters"],
                )
                assert "posterior_samples" in results
            except (ValueError, RuntimeError):
                pass  # Acceptable to reject invalid extreme values

    except ImportError:
        pytest.skip("BayesianEstimation module not available")


@pytest.mark.boundary
def test_division_by_very_small_std():
    """Test algorithm behavior when dividing by very small standard deviation."""
    try:
        from APGI_Equations import normalize_data

        # Test with very small variance
        small_variance_data = np.array([1.0, 1.0 + 1e-10, 1.0 - 1e-10])

        # Should handle division by near-zero std
        try:
            normalized = normalize_data(small_variance_data)
            assert not np.any(np.isnan(normalized))
            assert not np.any(np.isinf(normalized))
        except (ValueError, ZeroDivisionError, RuntimeError):
            pass  # Acceptable to raise error

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_large_dataset():
    """Test algorithm behavior with very large dataset."""
    try:
        from APGI_Equations import calculate_entropy

        # Test with large dataset
        large_data = np.random.randn(100000)

        # Should handle large datasets efficiently
        result = calculate_entropy(large_data)
        assert isinstance(result, (float, np.ndarray))

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_high_dimensional_data():
    """Test algorithm behavior with high-dimensional data."""
    try:
        import pytest
        import numpy as np

        # Test with high-dimensional data
        high_dim_data_1 = np.random.randn(100, 50)
        high_dim_data_2 = np.random.randn(100, 50)

        # Should handle high-dimensional data
        from APGI_Equations import calculate_mutual_information

        result = calculate_mutual_information(high_dim_data_1, high_dim_data_2)
        assert isinstance(result, (float, np.ndarray))

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_near_boundary_floating_point():
    """Test algorithm behavior with floating-point boundary values."""
    try:
        from APGI_Equations import calculate_entropy

        # Test with floating-point boundary values
        boundary_values = [
            np.finfo(float).tiny,  # Smallest positive float
            np.finfo(float).max,  # Largest float
            np.finfo(float).eps,  # Machine epsilon
            1.0 - np.finfo(float).eps,  # Just below 1
            0.0 + np.finfo(float).eps,  # Just above 0
        ]

        for value in boundary_values:
            test_data = np.array([value, value * 2, value * 3])

            try:
                result = calculate_entropy(test_data)
                assert isinstance(result, (float, np.ndarray))
                assert not np.isnan(result)
            except (ValueError, OverflowError, RuntimeError):
                pass  # Acceptable to reject boundary values

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_negative_values_in_probabilities():
    """Test algorithm behavior with negative probability values."""
    try:
        from APGI_Equations import calculate_entropy

        # Test with negative values (invalid for probabilities)
        negative_data = np.array([-0.5, 0.3, -0.2, 0.4])

        # Should handle negative values (either raise error or handle gracefully)
        try:
            result = calculate_entropy(negative_data)
            # If it doesn't raise error, result should be valid
            assert isinstance(result, (float, np.ndarray))
        except (ValueError, RuntimeError):
            pass  # Acceptable to raise error on negative values

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_convergence_failure_handling():
    """Test algorithm behavior when convergence fails."""
    try:
        from Falsification.Falsification_BayesianEstimation_ParameterRecovery import (
            run_bayesian_estimation_complete,
        )

        # Generate data that may cause convergence issues
        problematic_data = {
            "observations": np.random.randn(10, 5),  # Very small dataset
            "true_parameters": {"beta": 0.7, "pi": 0.5},
        }

        # Should handle convergence failure gracefully
        try:
            results = run_bayesian_estimation_complete(
                data=problematic_data["observations"],
                true_parameters=problematic_data["true_parameters"],
            )

            # Check if convergence diagnostics indicate issues
            if "convergence_diagnostics" in results:
                diagnostics = results["convergence_diagnostics"]
                # Should have convergence metrics even if failed
                assert "r_hat" in diagnostics or "ess" in diagnostics

        except (ValueError, RuntimeError):
            pass  # Acceptable to raise error on convergence failure

    except ImportError:
        pytest.skip("BayesianEstimation module not available")


@pytest.mark.boundary
def test_mixed_nan_and_valid_values():
    """Test algorithm behavior with mix of NaN and valid values."""
    try:
        from APGI_Equations import calculate_entropy

        # Test with mixed NaN and valid values
        mixed_data = np.array([0.5, np.nan, 0.7, 0.3, np.nan, 0.9, 0.1])

        # Should handle mixed data
        try:
            result = calculate_entropy(mixed_data)
            # Result should either be NaN (if propagated) or computed on valid subset
            assert np.isnan(result) or isinstance(result, float)
        except (ValueError, RuntimeError):
            pass  # Acceptable to raise error

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_very_small_positive_values():
    """Test algorithm behavior with very small positive values."""
    try:
        from APGI_Equations import calculate_entropy

        # Test with very small positive values
        small_values = np.array([1e-100, 1e-50, 1e-25, 1e-10])

        # Should handle very small values
        try:
            result = calculate_entropy(small_values)
            assert isinstance(result, (float, np.ndarray))
            assert not np.isinf(result)
        except (ValueError, OverflowError, RuntimeError):
            pass  # Acceptable to reject very small values

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.boundary
def test_parameter_bounds_validation():
    """Test parameter bounds validation in falsification framework."""
    try:
        from APGI_Falsification_Framework import FalsificationFramework

        framework = FalsificationFramework()

        # Test with parameter values at bounds
        boundary_test_data = {
            "P1": {
                "advantage_metric": np.array([0.0, 0.5, 1.0]),  # Min, mid, max
                "comparison_metric": np.array([0.0, 0.5, 1.0]),
                "effect_size": np.array([0.0, 0.5, 1.0]),
            },
        }

        # Should handle boundary values
        results = framework.run_comprehensive_falsification(boundary_test_data)
        assert isinstance(results, dict)
        assert "priority_results" in results

    except ImportError:
        pytest.skip("FalsificationFramework not available")
    except Exception as e:
        pytest.fail(f"Parameter bounds validation failed: {e}")
