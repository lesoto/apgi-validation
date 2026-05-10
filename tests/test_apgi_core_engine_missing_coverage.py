"""
Targeted tests for apgi_core/engine.py missing coverage areas.
============================================================

Focuses on specific functions and lines that are currently uncovered.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apgi_core.engine import APGIHierarchy, APGIRecovery, APGIValidationMetrics


class TestAPGIHierarchyMissingCoverage:
    """Test APGIHierarchy methods with missing coverage."""

    def test_level_count_overlap_le_1(self):
        """Test level_count when overlap <= 1."""
        result = APGIHierarchy.level_count(100.0, 1.0, 1.0)
        assert result == 1

        result = APGIHierarchy.level_count(100.0, 1.0, 0.5)
        assert result == 1

    def test_level_count_normal_case(self):
        """Test level_count with normal overlap > 1."""
        result = APGIHierarchy.level_count(100.0, 1.0, 2.0)
        assert result == int(np.ceil(np.log(100.0) / np.log(2.0)))
        assert result >= 1

    def test_level_count_edge_cases(self):
        """Test level_count edge cases."""
        # Test with tau_max == tau_min
        result = APGIHierarchy.level_count(10.0, 10.0, 2.0)
        assert result == 1  # log(1) = 0, ceil(0) = 0, but should return at least 1

        # Test with very large overlap
        result = APGIHierarchy.level_count(1000.0, 1.0, 10.0)
        assert result >= 1

    def test_cross_level_modulation(self):
        """Test cross_level_modulation function."""
        theta_0 = 1.0
        pi_next = 0.5
        phi_next = np.pi / 4
        kappa_down = 0.2

        result = APGIHierarchy.cross_level_modulation(theta_0, pi_next, phi_next, kappa_down)

        expected = theta_0 * (1 + kappa_down * pi_next * np.cos(phi_next))
        assert abs(result - expected) < 1e-10

    def test_cross_level_modulation_zero_values(self):
        """Test cross_level_modulation with zero values."""
        result = APGIHierarchy.cross_level_modulation(1.0, 0.0, 0.0, 0.0)
        assert result == 1.0

        result = APGIHierarchy.cross_level_modulation(0.0, 1.0, 0.0, 1.0)
        assert result == 0.0

    def test_bottom_up_cascade_active(self):
        """Test bottom_up_cascade when previous level is active."""
        theta_l = 1.0
        s_prev = 2.0
        theta_prev = 1.0  # s_prev - theta_prev = 1.0 > 0, so H = 1
        kappa_up = 0.3

        result = APGIHierarchy.bottom_up_cascade(theta_l, s_prev, theta_prev, kappa_up)
        expected = theta_l * (1 - kappa_up * 1.0)
        assert abs(result - expected) < 1e-10

    def test_bottom_up_cascade_inactive(self):
        """Test bottom_up_cascade when previous level is inactive."""
        theta_l = 1.0
        s_prev = 0.5
        theta_prev = 1.0  # s_prev - theta_prev = -0.5 < 0, so H = 0
        kappa_up = 0.3

        result = APGIHierarchy.bottom_up_cascade(theta_l, s_prev, theta_prev, kappa_up)
        expected = theta_l * (1 - kappa_up * 0.0)
        assert abs(result - expected) < 1e-10

    def test_bottom_up_cascade_edge_cases(self):
        """Test bottom_up_cascade edge cases."""
        # Test exactly at threshold
        result = APGIHierarchy.bottom_up_cascade(1.0, 1.0, 1.0, 0.5)
        expected = 1.0 * (1 - 0.5 * 0.0)  # H = 0 since s_prev - theta_prev = 0
        assert abs(result - expected) < 1e-10

        # Test with zero kappa_up
        result = APGIHierarchy.bottom_up_cascade(2.0, 3.0, 1.0, 0.0)
        assert result == 2.0

    def test_phase_signal(self):
        """Test phase_signal function."""
        omega = 2.0 * np.pi  # 1 Hz
        t = 0.5
        phi_0 = np.pi / 4

        result = APGIHierarchy.phase_signal(omega, t, phi_0)
        expected = omega * t + phi_0
        assert abs(result - expected) < 1e-10

    def test_phase_signal_zero_time(self):
        """Test phase_signal at t=0."""
        result = APGIHierarchy.phase_signal(1.0, 0.0, np.pi)
        assert result == np.pi

    def test_phase_signal_zero_frequency(self):
        """Test phase_signal with zero frequency."""
        result = APGIHierarchy.phase_signal(0.0, 10.0, 0.0)
        assert result == 0.0


class TestAPGIRecoveryMissingCoverage:
    """Test APGIRecovery methods with missing coverage."""

    def test_reset_rule(self):
        """Test reset_rule function."""
        s_t = 2.0
        theta_t = 1.5
        rho = 0.8
        delta = 0.1

        s_new, theta_new = APGIRecovery.reset_rule(s_t, theta_t, rho, delta)

        assert s_new == s_t * rho
        assert theta_new == theta_t + delta

    def test_reset_rule_zero_values(self):
        """Test reset_rule with zero values."""
        s_new, theta_new = APGIRecovery.reset_rule(0.0, 0.0, 0.0, 0.0)
        assert s_new == 0.0
        assert theta_new == 0.0

    def test_reset_rule_negative_values(self):
        """Test reset_rule with negative values."""
        s_t = -2.0
        theta_t = -1.0
        rho = -0.5
        delta = -0.2

        s_new, theta_new = APGIRecovery.reset_rule(s_t, theta_t, rho, delta)

        assert s_new == 1.0  # -2.0 * -0.5 = 1.0
        assert theta_new == -1.2  # -1.0 + -0.2 = -1.2


class TestAPGIValidationMetricsMissingCoverage:
    """Test APGIValidationMetrics methods with missing coverage."""

    def test_validation_metrics_initialization(self):
        """Test APGIValidationMetrics initialization."""
        metrics = APGIValidationMetrics()
        assert metrics is not None

    def test_power_spectrum(self):
        """Test power spectrum calculation."""
        f = np.array([1.0, 2.0, 3.0])
        sigma_l = np.array([0.5, 1.0, 1.5])
        tau_l = np.array([0.1, 0.2, 0.3])

        spectrum = APGIValidationMetrics.power_spectrum(f, sigma_l, tau_l)

        assert isinstance(spectrum, np.ndarray)
        assert spectrum.shape == f.shape

    def test_hurst_exponent(self):
        """Test Hurst exponent calculation."""
        beta_spec = 1.5
        hurst = APGIValidationMetrics.hurst_exponent(beta_spec)

        assert isinstance(hurst, float)
        assert hurst == (beta_spec + 1) / 2.0


class TestEngineEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_arrays(self):
        """Test functions with empty arrays."""
        metrics = APGIValidationMetrics()

        with pytest.raises((ValueError, IndexError)):
            metrics.bayesian_evidence(np.array([]), np.array([]))

    def test_single_value_arrays(self):
        """Test functions with single-value arrays."""
        metrics = APGIValidationMetrics()

        single_val = np.array([0.5])
        evidence = metrics.bayesian_evidence(single_val, single_val)
        assert isinstance(evidence, float)

    def test_negative_probabilities(self):
        """Test functions with negative probabilities."""
        metrics = APGIValidationMetrics()

        with pytest.raises(ValueError):
            metrics.kl_divergence(np.array([-0.1, 0.5]), np.array([0.3, 0.7]))

    def test_non_normalized_probabilities(self):
        """Test functions with non-normalized probabilities."""
        metrics = APGIValidationMetrics()

        # Should handle non-normalized probabilities gracefully
        p = np.array([2.0, 3.0])  # Sum = 5.0
        q = np.array([1.0, 4.0])  # Sum = 5.0

        kl_div = metrics.kl_divergence(p, q)
        assert isinstance(kl_div, float)

    def test_very_small_values(self):
        """Test functions with very small values."""
        metrics = APGIValidationMetrics()

        small_vals = np.array([1e-10, 1e-8, 1e-6])
        evidence = metrics.bayesian_evidence(small_vals, small_vals)
        assert isinstance(evidence, float)
        assert evidence > 0

    def test_very_large_values(self):
        """Test functions with very large values."""
        metrics = APGIValidationMetrics()

        large_vals = np.array([1e6, 1e7, 1e8])
        evidence = metrics.bayesian_evidence(large_vals, large_vals * 0.9)
        assert isinstance(evidence, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
