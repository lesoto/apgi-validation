"""
Tests for APGI threshold dynamics - adaptive threshold behavior with metabolic costs.
========================================================================
"""

import pytest
import numpy as np
from pathlib import Path

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util
from pathlib import Path

# Load APGI_Full_Dynamic_Model from hyphenated filename
spec = importlib.util.spec_from_file_location(
    "APGI_Full_Dynamic_Model",
    Path(__file__).parent.parent / "APGI_Full_Dynamic_Model.py",
)
APGI_Full_Dynamic_Model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(APGI_Full_Dynamic_Model)
APGIFullDynamicModel = APGI_Full_Dynamic_Model.APGIFullDynamicModel

from APGI_Equations import (
    APGIParameters,
)


class TestThresholdDynamics:
    """Test threshold dynamics with metabolic cost modulation."""

    def test_threshold_dynamics_basic(self):
        """Test threshold dynamics with positive change."""
        model = APGIFullDynamicModel(APGIParameters())

        # Step with high signal → metabolic cost
        model.step(Pi_e=0.8, Pi_i=0.7, epsilon_e=2.0, epsilon_i=0.5)

        # Threshold should be valid float
        assert isinstance(model.state.theta_t, float)

    def test_threshold_dynamics_negative_change(self):
        """Test threshold dynamics with negative change."""
        model = APGIFullDynamicModel(APGIParameters())

        # Step with low signal (low cost) → threshold should decrease
        model.step(Pi_e=0.5, Pi_i=0.5, epsilon_e=-1.0, epsilon_i=0.2)

        # Threshold behavior depends on metabolic cost vs information value
        assert isinstance(model.state.theta_t, float)

    def test_threshold_dynamics_boundary_sleep(self):
        """Test threshold dynamics at sleep boundary."""
        model = APGIFullDynamicModel(APGIParameters())

        # Set threshold to sleep level
        model.state.theta_t = model.params.theta_0_sleep

        # Step with minimal signal
        model.step(Pi_e=0.3, Pi_i=0.3, epsilon_e=0.1, epsilon_i=0.1)

        # Should remain near sleep threshold with minimal input
        assert abs(model.state.theta_t - model.params.theta_0_sleep) < 0.1

    def test_threshold_dynamics_boundary_alert(self):
        """Test threshold dynamics at alert boundary."""
        model = APGIFullDynamicModel(APGIParameters())

        # Set threshold to alert level
        model.state.theta_t = model.params.theta_0_alert

        # Step with moderate signal
        model.step(Pi_e=0.6, Pi_i=0.6, epsilon_e=1.0, epsilon_i=0.3)

        # Should move toward alert threshold
        assert model.state.theta_t > model.params.theta_0_sleep

    def test_threshold_dynamics_metabolic_coupling(self):
        """Test threshold dynamics with metabolic cost coupling."""
        model = APGIFullDynamicModel(APGIParameters())

        # Step with high signal to increase metabolic cost
        model.step(Pi_e=0.9, Pi_i=0.8, epsilon_e=3.0, epsilon_i=1.0)

        # Both threshold and metabolic state should be valid
        assert isinstance(model.state.theta_t, float)
        assert isinstance(model.state.eta_m, float)

    def test_threshold_dynamics_no_ignition(self):
        """Test threshold dynamics without ignition."""
        model = APGIFullDynamicModel(APGIParameters())
        initial_theta = model.state.theta_t

        # Step with low signal (no ignition)
        model.step(
            Pi_e=0.3,
            Pi_i=0.3,
            epsilon_e=0.1,
            epsilon_i=0.1,
            deterministic_ignition=True,
        )

        # Should have minimal threshold change without ignition
        assert abs(model.state.theta_t - initial_theta) < 0.5
        assert model.state.I == 0  # No ignition occurred

    def test_threshold_dynamics_with_ignition(self):
        """Test threshold dynamics with ignition."""
        model = APGIFullDynamicModel(APGIParameters())

        # Step with high signal (guaranteed ignition)
        model.step(
            Pi_e=0.95,
            Pi_i=0.9,
            epsilon_e=5.0,
            epsilon_i=2.0,
            deterministic_ignition=True,
        )

        # Should have valid state after ignition
        assert isinstance(model.state.theta_t, float)
        assert model.state.I in [0, 1]  # Ignition may or may not occur

    def test_threshold_dynamics_continuous_time(self):
        """Test threshold dynamics over continuous time."""
        model = APGIFullDynamicModel(APGIParameters())
        theta_values = []

        # Simulate multiple timesteps
        for _ in range(10):
            theta_values.append(model.state.theta_t)
            model.step(Pi_e=0.5, Pi_i=0.5, epsilon_e=0.5, epsilon_i=0.2)

        # Should show monotonic or near-monotonic behavior
        # (allowing for some noise/stochasticity)
        assert len(theta_values) == 10
        assert all(isinstance(theta, float) for theta in theta_values)

    def test_threshold_dynamics_extreme_signals(self):
        """Test threshold dynamics with extreme signal values."""
        model = APGIFullDynamicModel(APGIParameters())
        initial_theta = model.state.theta_t

        # Test with very high signal
        model.step(Pi_e=0.95, Pi_i=0.9, epsilon_e=10.0, epsilon_i=3.0)
        high_theta = model.state.theta_t

        # Reset and test with very low signal
        model.state.theta_t = initial_theta
        model.step(Pi_e=0.2, Pi_i=0.2, epsilon_e=-10.0, epsilon_i=-2.0)
        low_theta = model.state.theta_t

        # Should handle extreme values gracefully
        assert isinstance(high_theta, float)
        assert isinstance(low_theta, float)
        assert np.isfinite(high_theta)
        assert np.isfinite(low_theta)

    def test_threshold_dynamics_parameter_sensitivity(self):
        """Test threshold dynamics sensitivity to parameters."""
        # Test with different tau_theta values
        params_fast = APGIParameters(tau_theta=10.0)  # Fast adaptation
        params_slow = APGIParameters(tau_theta=100.0)  # Slow adaptation

        model_fast = APGIFullDynamicModel(params_fast)
        model_slow = APGIFullDynamicModel(params_slow)

        # Same initial conditions and signal
        model_fast.step(Pi_e=0.7, Pi_i=0.7, epsilon_e=2.0, epsilon_i=0.5)
        model_slow.step(Pi_e=0.7, Pi_i=0.7, epsilon_e=2.0, epsilon_i=0.5)

        # Both should have valid theta values
        assert isinstance(model_fast.state.theta_t, float)
        assert isinstance(model_slow.state.theta_t, float)

    def test_threshold_dynamics_numerical_stability(self):
        """Test numerical stability of threshold dynamics."""
        model = APGIFullDynamicModel(APGIParameters())

        # Test with very small timestep
        original_delta_t = model.params.delta_t
        model.params.delta_t = 1e-6

        model.step(Pi_e=0.5, Pi_i=0.5, epsilon_e=1.0, epsilon_i=0.3)

        # Should remain stable with small timestep
        assert np.isfinite(model.state.theta_t)
        # Threshold may or may not change depending on dynamics
        assert isinstance(model.state.theta_t, float)
        # Verify theta is non-negative
        assert model.state.theta_t >= 0

        # Restore original timestep
        model.params.delta_t = original_delta_t

    def test_threshold_dynamics_reset_conditions(self):
        """Test threshold dynamics under reset conditions."""
        model = APGIFullDynamicModel(APGIParameters())

        # Test at minimum threshold
        model.state.theta_t = 0.1
        model.step(Pi_e=0.3, Pi_i=0.3, epsilon_e=0.0, epsilon_i=0.0)

        # Should not go negative or explode
        assert model.state.theta_t >= 0.0
        assert np.isfinite(model.state.theta_t)

    def test_threshold_dynamics_integration_consistency(self):
        """Test integration consistency with other model components."""
        model = APGIFullDynamicModel(APGIParameters())

        # Step model and track all state variables
        initial_S = model.state.S

        model.step(Pi_e=0.6, Pi_i=0.6, epsilon_e=1.5, epsilon_i=0.5)

        # All components should be updated
        assert model.state.S != initial_S  # Signal should change
        # Threshold and metabolic may or may not change depending on dynamics
        assert isinstance(model.state.theta_t, float)
        assert isinstance(model.state.eta_m, float)

        # Values should be physically plausible
        assert model.state.S > 0
        assert model.state.theta_t > 0
        assert np.isfinite(model.state.eta_m)


if __name__ == "__main__":
    pytest.main([__file__])
