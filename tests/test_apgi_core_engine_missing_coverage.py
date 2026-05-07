"""
Targeted tests for apgi_core/engine.py missing coverage areas.
============================================================

Focuses on specific functions and lines that are currently uncovered.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apgi_core.engine import (
    APGIHierarchy,
    APGIRecovery,
    APGIValidationMetrics,
    APGIActiveInference,
    APGIMultimodalIntegration,
)


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

        result = APGIHierarchy.cross_level_modulation(
            theta_0, pi_next, phi_next, kappa_down
        )

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

    def test_bayesian_evidence(self):
        """Test bayesian_evidence calculation."""
        metrics = APGIValidationMetrics()

        # Mock data
        model_predictions = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        empirical_data = np.array([0.2, 0.4, 0.5, 0.7, 1.0])

        evidence = metrics.bayesian_evidence(model_predictions, empirical_data)
        assert isinstance(evidence, float)
        assert evidence > 0

    def test_bayesian_evidence_identical_data(self):
        """Test bayesian_evidence with identical data."""
        metrics = APGIValidationMetrics()

        data = np.array([0.1, 0.3, 0.6, 0.8, 0.9])
        evidence = metrics.bayesian_evidence(data, data)

        # Should be high for identical data
        assert evidence > 1.0

    def test_kl_divergence(self):
        """Test KL divergence calculation."""
        metrics = APGIValidationMetrics()

        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = np.array([0.25, 0.25, 0.25, 0.25])

        kl_div = metrics.kl_divergence(p, q)
        assert isinstance(kl_div, float)
        assert kl_div >= 0

    def test_kl_divergence_identical(self):
        """Test KL divergence with identical distributions."""
        metrics = APGIValidationMetrics()

        p = np.array([0.1, 0.2, 0.3, 0.4])
        kl_div = metrics.kl_divergence(p, p)

        # Should be zero for identical distributions
        assert abs(kl_div - 0.0) < 1e-10

    def test_correlation_analysis(self):
        """Test correlation analysis."""
        metrics = APGIValidationMetrics()

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # Perfect correlation

        corr = metrics.correlation_analysis(x, y)
        assert isinstance(corr, float)
        assert abs(corr - 1.0) < 1e-10

    def test_correlation_analysis_no_correlation(self):
        """Test correlation analysis with no correlation."""
        metrics = APGIValidationMetrics()

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 0, 1, 0, 1])  # No clear correlation

        corr = metrics.correlation_analysis(x, y)
        assert isinstance(corr, float)
        assert abs(corr) < 1.0

    def test_model_fit_metrics(self):
        """Test model fit metrics calculation."""
        metrics = APGIValidationMetrics()

        predicted = np.array([1.1, 1.9, 3.1, 4.0, 4.9])
        observed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        fit_metrics = metrics.model_fit_metrics(predicted, observed)
        assert isinstance(fit_metrics, dict)
        assert "rmse" in fit_metrics
        assert "r_squared" in fit_metrics
        assert "mae" in fit_metrics

    def test_model_fit_metrics_perfect_fit(self):
        """Test model fit metrics with perfect fit."""
        metrics = APGIValidationMetrics()

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fit_metrics = metrics.model_fit_metrics(data, data)

        # Perfect fit should have zero error
        assert fit_metrics["rmse"] < 1e-10
        assert fit_metrics["r_squared"] > 0.99  # Should be 1.0
        assert fit_metrics["mae"] < 1e-10


class TestAPGIActiveInferenceMissingCoverage:
    """Test APGIActiveInference methods with missing coverage."""

    def test_prediction_error(self):
        """Test prediction error calculation."""
        inference = APGIActiveInference()

        predicted = np.array([0.5, 0.3, 0.8])
        observed = np.array([0.6, 0.2, 0.9])

        error = inference.prediction_error(predicted, observed)
        assert isinstance(error, float)
        assert error >= 0

    def test_prediction_error_zero(self):
        """Test prediction error with identical predictions."""
        inference = APGIActiveInference()

        data = np.array([0.1, 0.2, 0.3])
        error = inference.prediction_error(data, data)

        assert abs(error - 0.0) < 1e-10

    def test_free_energy(self):
        """Test free energy calculation."""
        inference = APGIActiveInference()

        prediction = np.array([0.4, 0.6])
        sensory_input = np.array([0.3, 0.7])

        free_energy = inference.free_energy(prediction, sensory_input)
        assert isinstance(free_energy, float)

    def test_free_energy_perfect_match(self):
        """Test free energy with perfect prediction match."""
        inference = APGIActiveInference()

        data = np.array([0.5, 0.5])
        free_energy = inference.free_energy(data, data)

        # Should be minimal for perfect match
        assert free_energy >= 0

    def test_belief_update(self):
        """Test belief update process."""
        inference = APGIActiveInference()

        prior_belief = np.array([0.5, 0.5])
        prediction_error_signal = np.array([0.1, -0.1])
        learning_rate = 0.1

        updated_belief = inference.belief_update(
            prior_belief, prediction_error_signal, learning_rate
        )

        assert len(updated_belief) == len(prior_belief)
        assert not np.array_equal(updated_belief, prior_belief)

    def test_belief_update_zero_learning_rate(self):
        """Test belief update with zero learning rate."""
        inference = APGIActiveInference()

        prior = np.array([0.3, 0.7])
        error = np.array([0.1, -0.1])

        updated = inference.belief_update(prior, error, 0.0)

        # Should be unchanged with zero learning rate
        assert np.array_equal(updated, prior)


class TestAPGIMultimodalIntegrationMissingCoverage:
    """Test APGIMultimodalIntegration methods with missing coverage."""

    def test_multisensory_integration(self):
        """Test multisensory integration."""
        integration = APGIMultimodalIntegration()

        visual_input = np.array([0.8, 0.2])
        auditory_input = np.array([0.3, 0.7])

        integrated = integration.multisensory_integration(visual_input, auditory_input)

        assert len(integrated) == len(visual_input)
        assert isinstance(integrated, np.ndarray)

    def test_multisensory_integration_weighted(self):
        """Test multisensory integration with weights."""
        integration = APGIMultimodalIntegration()

        visual = np.array([0.6, 0.4])
        auditory = np.array([0.2, 0.8])
        visual_weight = 0.7
        auditory_weight = 0.3

        result = integration.multisensory_integration(
            visual, auditory, visual_weight, auditory_weight
        )

        assert len(result) == len(visual)

    def test_crossmodal_attention(self):
        """Test crossmodal attention."""
        integration = APGIMultimodalIntegration()

        primary_modality = np.array([0.9, 0.1])
        secondary_modality = np.array([0.4, 0.6])

        attention_weights = integration.crossmodal_attention(
            primary_modality, secondary_modality
        )

        assert len(attention_weights) == len(primary_modality)
        assert all(0 <= w <= 1 for w in attention_weights)

    def test_temporal_binding(self):
        """Test temporal binding across modalities."""
        integration = APGIMultimodalIntegration()

        visual_timeseries = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        auditory_timeseries = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        binding_strength = integration.temporal_binding(
            visual_timeseries, auditory_timeseries
        )

        assert isinstance(binding_strength, float)
        assert 0 <= binding_strength <= 1

    def test_temporal_binding_perfect_correlation(self):
        """Test temporal binding with perfectly correlated signals."""
        integration = APGIMultimodalIntegration()

        signal = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        binding = integration.temporal_binding(signal, signal)

        # Should be high for identical signals
        assert binding > 0.9

    def test_modality_fusion(self):
        """Test modality fusion."""
        integration = APGIMultimodalIntegration()

        modalities = {
            "visual": np.array([0.7, 0.3]),
            "auditory": np.array([0.4, 0.6]),
            "tactile": np.array([0.5, 0.5]),
        }

        fused = integration.modality_fusion(modalities)

        assert len(fused) == 2  # Should match dimensionality
        assert isinstance(fused, np.ndarray)

    def test_conflict_resolution(self):
        """Test conflict resolution between modalities."""
        integration = APGIMultimodalIntegration()

        conflicting_inputs = {
            "visual": np.array([0.9, 0.1]),
            "auditory": np.array([0.1, 0.9]),  # Conflicting with visual
        }

        resolved = integration.conflict_resolution(conflicting_inputs)

        assert len(resolved) == 2
        assert isinstance(resolved, np.ndarray)


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
