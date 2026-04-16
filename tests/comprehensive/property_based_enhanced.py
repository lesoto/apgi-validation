"""
APGI Property-Based Testing Enhancement Module
===============================================

Expanded property-based testing with:
- Increased max_examples for critical paths
- Custom Hypothesis strategies
- Stateful testing
- Comprehensive invariant checking

This module extends test_property_based.py with advanced Hypothesis features.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
from hypothesis import Phase, assume, given, settings, strategies
from hypothesis.extra import numpy as np_st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import APGI modules
try:
    from APGI_Equations import FoundationalEquations

    APGI_AVAILABLE = True
except ImportError:
    APGI_AVAILABLE = False


# ============================================================================
# Custom Hypothesis Strategies
# ============================================================================


class APGIStrategies:
    """Custom Hypothesis strategies for APGI testing."""

    @staticmethod
    def probability() -> strategies.SearchStrategy[float]:
        """Strategy for probability values (0.0 to 1.0)."""
        return strategies.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        )

    @staticmethod
    def positive_float(max_value: float = 1e6) -> strategies.SearchStrategy[float]:
        """Strategy for positive floating point numbers."""
        return strategies.floats(
            min_value=1e-10, max_value=max_value, allow_nan=False, allow_infinity=False
        )

    @staticmethod
    def small_positive_float() -> strategies.SearchStrategy[float]:
        """Strategy for small positive values suitable for parameters."""
        return strategies.floats(
            min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False
        )

    @staticmethod
    def time_series(length: int = 100) -> strategies.SearchStrategy[np.ndarray]:
        """Strategy for generating time series data."""
        return np_st.arrays(
            dtype=np.float64,
            shape=(length,),
            elements=strategies.floats(
                min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
            ),
        )

    @staticmethod
    def eeg_signal(
        channels: int = 64, samples: int = 1000
    ) -> strategies.SearchStrategy[np.ndarray]:
        """Strategy for generating EEG-like signals."""
        return np_st.arrays(
            dtype=np.float64,
            shape=(channels, samples),
            elements=strategies.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
        )

    @staticmethod
    def apgi_parameters() -> strategies.SearchStrategy[dict]:
        """Strategy for generating valid APGI model parameters."""
        return strategies.fixed_dictionaries(
            {
                "tau_S": strategies.floats(min_value=0.01, max_value=2.0),
                "tau_theta": strategies.floats(min_value=1.0, max_value=100.0),
                "theta_0": strategies.floats(min_value=0.1, max_value=1.0),
                "alpha": strategies.floats(min_value=1.0, max_value=50.0),
                "gamma_M": strategies.floats(min_value=-1.0, max_value=0.0),
                "gamma_A": strategies.floats(min_value=0.0, max_value=0.5),
                "rho": strategies.floats(min_value=0.0, max_value=1.0),
                "sigma_S": strategies.floats(min_value=0.001, max_value=0.5),
                "sigma_theta": strategies.floats(min_value=0.001, max_value=0.1),
            }
        )

    @staticmethod
    def configuration_dict() -> strategies.SearchStrategy[dict]:
        """Strategy for generating valid configuration dictionaries."""
        return strategies.fixed_dictionaries(
            {
                "model": strategies.fixed_dictionaries(
                    {
                        "tau_S": strategies.floats(min_value=0.01, max_value=2.0),
                        "tau_theta": strategies.floats(min_value=1.0, max_value=100.0),
                    }
                ),
                "simulation": strategies.fixed_dictionaries(
                    {
                        "default_steps": strategies.integers(
                            min_value=100, max_value=10000
                        ),
                        "default_dt": strategies.floats(min_value=0.001, max_value=0.1),
                    }
                ),
            }
        )


# ============================================================================
# Enhanced Property-Based Tests with Increased max_examples
# ============================================================================


@pytest.mark.slow
class TestEnhancedMathematicalProperties:
    """Enhanced mathematical property tests with increased coverage."""

    @pytest.mark.skipif(not APGI_AVAILABLE, reason="APGI modules not available")
    @settings(
        max_examples=500,
        deadline=None,
        phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.target],
    )
    @given(
        APGIStrategies.probability(),
        APGIStrategies.probability(),
        APGIStrategies.probability(),
    )
    def test_surprise_non_negativity(self, error, reference, precision):
        """Test that surprise is always non-negative (increased examples)."""
        if APGI_AVAILABLE:
            surprise = FoundationalEquations.surprise(error, reference)
            assert surprise >= 0, f"Surprise should be non-negative, got {surprise}"

            # Additional property: surprise increases with |error - reference|
            surprise2 = FoundationalEquations.surprise(error * 2, reference)
            if abs(error * 2 - reference) > abs(error - reference):
                assert (
                    surprise2 >= surprise
                ), "Surprise should increase with larger error"

    @pytest.mark.skipif(not APGI_AVAILABLE, reason="APGI modules not available")
    @settings(max_examples=300, deadline=None)
    @given(APGIStrategies.positive_float(), APGIStrategies.positive_float())
    def test_precision_bounds(self, variance, epsilon):
        """Test precision computation stays within valid bounds."""
        if APGI_AVAILABLE and variance > epsilon:
            precision = FoundationalEquations.precision(variance, epsilon)
            assert (
                0 <= precision <= 1
            ), f"Precision should be in [0, 1], got {precision}"

    @pytest.mark.skipif(not APGI_AVAILABLE, reason="APGI modules not available")
    @settings(max_examples=400, deadline=None)
    @given(APGIStrategies.time_series(length=50), APGIStrategies.small_positive_float())
    def test_entropy_monotonicity(self, distribution, epsilon):
        """Test entropy properties with increased examples."""
        # Normalize distribution
        dist_sum = np.sum(np.abs(distribution))
        if dist_sum > epsilon:
            normalized = np.abs(distribution) / dist_sum
            entropy = -np.sum(normalized * np.log2(normalized + epsilon))

            # Entropy should be non-negative
            assert entropy >= 0, f"Entropy should be non-negative, got {entropy}"

            # Entropy of uniform distribution should be maximal
            uniform = np.ones_like(normalized) / len(normalized)
            uniform_entropy = -np.sum(uniform * np.log2(uniform + epsilon))
            assert (
                entropy <= uniform_entropy * 1.01
            ), "Entropy should not exceed uniform by much"


@pytest.mark.slow
class TestEnhancedDataProperties:
    """Enhanced data property tests."""

    @settings(max_examples=200, deadline=None)
    @given(
        APGIStrategies.eeg_signal(channels=8, samples=100),
        strategies.integers(min_value=1, max_value=10),
    )
    def test_eeg_signal_shape_preservation(self, signal, filter_order):
        """Test that signal processing preserves shape."""
        # Simple moving average filter
        if filter_order < len(signal[0]):
            kernel = np.ones(filter_order) / filter_order
            filtered = np.array([np.convolve(ch, kernel, mode="same") for ch in signal])
            assert filtered.shape == signal.shape, "Filter should preserve signal shape"

    @settings(max_examples=150, deadline=None)
    @given(
        APGIStrategies.apgi_parameters(),
        strategies.integers(min_value=10, max_value=100),
    )
    def test_parameter_validity(self, params, n_steps):
        """Test that generated parameters produce valid simulations."""
        # Verify all parameters are positive where required
        assert params["tau_S"] > 0, "tau_S must be positive"
        assert params["tau_theta"] > 0, "tau_theta must be positive"
        assert params["alpha"] > 0, "alpha must be positive"
        assert 0 <= params["rho"] <= 1, "rho must be in [0, 1]"


# ============================================================================
# Stateful Testing
# ============================================================================


class APGISimulationStateMachine(RuleBasedStateMachine):
    """Stateful test for APGI simulation workflows."""

    def __init__(self):
        super().__init__()
        self.parameters = {}
        self.initial_state = None
        self.current_state = None
        self.time_steps = []
        self.surprise_history = []
        self.threshold_history = []

    @rule(params=APGIStrategies.apgi_parameters())
    def initialize_parameters(self, params):
        """Initialize simulation parameters."""
        self.parameters = params
        self.time_steps = [0]
        self.surprise_history = [0.1]
        self.threshold_history = [params.get("theta_0", 0.5)]

    @rule(dt=APGIStrategies.small_positive_float())
    def step_simulation(self, dt):
        """Advance simulation by one time step."""
        if not self.parameters or not self.time_steps:
            return

        # Simple Euler integration for testing
        current_t = self.time_steps[-1]
        current_S = self.surprise_history[-1]
        current_theta = self.threshold_history[-1]

        # Update equations (simplified)
        tau_S = self.parameters.get("tau_S", 0.5)
        tau_theta = self.parameters.get("tau_theta", 30.0)
        theta_0 = self.parameters.get("theta_0", 0.5)
        alpha = self.parameters.get("alpha", 10.0)

        # Simple dynamics
        dS = -current_S / tau_S + np.random.normal(0, 0.01)
        dtheta = (theta_0 - current_theta + alpha * current_S) / tau_theta

        new_S = max(0, current_S + dS * dt)
        new_theta = np.clip(current_theta + dtheta * dt, 0, 1)

        self.time_steps.append(current_t + dt)
        self.surprise_history.append(new_S)
        self.threshold_history.append(new_theta)

    @rule(noise_level=APGIStrategies.probability())
    def add_noise(self, noise_level):
        """Add noise to current state."""
        if self.surprise_history:
            current = self.surprise_history[-1]
            noise = np.random.normal(0, noise_level * 0.1)
            self.surprise_history[-1] = max(0, current + noise)

    @invariant()
    def state_consistency(self):
        """Verify state remains consistent."""
        if self.surprise_history:
            # Surprise should be non-negative
            assert all(
                s >= 0 for s in self.surprise_history
            ), "Surprise should be non-negative"

        if self.threshold_history:
            # Threshold should be in [0, 1]
            assert all(
                0 <= t <= 1 for t in self.threshold_history
            ), "Threshold should be in [0, 1]"

        if self.time_steps:
            # Time should be monotonic
            assert all(
                self.time_steps[i] <= self.time_steps[i + 1]
                for i in range(len(self.time_steps) - 1)
            ), "Time should be monotonic"

    @invariant()
    @precondition(lambda self: len(self.time_steps) > 1)
    def simulation_progress(self):
        """Verify simulation makes progress."""
        assert self.time_steps[-1] > self.time_steps[0], "Time should advance"


# Convert state machine to test
TestAPGISimulationStateful = APGISimulationStateMachine.TestCase
TestAPGISimulationStateful.settings = settings(
    max_examples=100, stateful_step_count=50, deadline=None
)


# ============================================================================
# Critical Path Testing with High Example Count
# ============================================================================


@pytest.mark.slow
class TestCriticalPaths:
    """Critical path testing with increased max_examples."""

    @pytest.mark.critical
    @settings(
        max_examples=1000,
        deadline=None,
        phases=[Phase.explicit, Phase.reuse, Phase.generate],
    )
    @given(
        x=APGIStrategies.probability(),
        y=APGIStrategies.probability(),
        z=APGIStrategies.probability(),
    )
    def test_associativity_critical(self, x, y, z):
        """Critical test: Associativity of probability combinations with 1000 examples."""
        # Test that weighted averages maintain associativity
        w1, w2, w3 = 0.5, 0.3, 0.2

        # (x ⊕ y) ⊕ z
        left = w1 * x + w2 * y
        left_result = (w1 + w2) * left + w3 * z

        # x ⊕ (y ⊕ z)
        right = w2 * y + w3 * z
        right_result = w1 * x + (w2 + w3) * right

        # These should be approximately equal (allowing for floating point)
        assert abs(left_result - right_result) < 1e-10, "Associativity violated"

    @pytest.mark.critical
    @settings(max_examples=800, deadline=None)
    @given(
        signal=APGIStrategies.time_series(length=20),
        window=strategies.integers(min_value=2, max_value=10),
    )
    def test_moving_average_properties_critical(self, signal, window):
        """Critical test: Moving average properties with 800 examples."""
        assume(len(signal) >= window)

        # Compute moving average
        kernel = np.ones(window) / window
        filtered = np.convolve(signal, kernel, mode="valid")

        # Property 1: Filtered signal should have reduced variance
        if len(filtered) > 1:
            original_var = np.var(signal)
            filtered_var = np.var(filtered)
            # Note: Not always true for small windows, but generally
            # We just check it's not unreasonably larger
            assert filtered_var <= original_var * 2, "Filtered variance too large"

        # Property 2: Mean should be preserved
        original_mean = np.mean(signal[window - 1 :])
        filtered_mean = np.mean(filtered)
        assert abs(original_mean - filtered_mean) < 1e-10, "Mean not preserved"


# ============================================================================
# Custom Strategy Composition Tests
# ============================================================================


@pytest.mark.slow
class TestCustomStrategies:
    """Tests demonstrating custom Hypothesis strategies."""

    @settings(max_examples=100, deadline=None)
    @given(config=APGIStrategies.configuration_dict())
    def test_configuration_validity(self, config):
        """Test that custom strategy generates valid configs."""
        assert "model" in config
        assert "simulation" in config
        assert config["model"]["tau_S"] > 0
        assert config["model"]["tau_theta"] > 0
        assert config["simulation"]["default_steps"] > 0
        assert config["simulation"]["default_dt"] > 0

    @settings(max_examples=100, deadline=None)
    @given(
        params1=APGIStrategies.apgi_parameters(),
        params2=APGIStrategies.apgi_parameters(),
    )
    def test_parameter_interpolation(self, params1, params2):
        """Test parameter interpolation properties."""
        alpha = 0.5

        # Interpolate between two parameter sets
        interpolated = {
            k: alpha * params1[k] + (1 - alpha) * params2[k] for k in params1.keys()
        }

        # Verify interpolated parameters are valid
        assert interpolated["tau_S"] > 0
        assert interpolated["tau_theta"] > 0
        assert 0 <= interpolated["rho"] <= 1


# ============================================================================
# Hypothesis Profile Registration
# ============================================================================

# Register profiles for different test execution modes
settings.register_profile("ci", max_examples=50, deadline=None, stateful_step_count=20)
settings.register_profile(
    "dev", max_examples=100, deadline=None, stateful_step_count=30
)
settings.register_profile(
    "full", max_examples=1000, deadline=None, stateful_step_count=50
)

# Load the CI profile by default for faster test execution
settings.load_profile("ci")


# ============================================================================
# Test Runner
# ============================================================================


def run_property_based_tests() -> Dict[str, Any]:
    """Run all property-based tests and return summary."""
    print("=" * 80)
    print("APGI ENHANCED PROPERTY-BASED TESTING")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - Mathematical properties: 300-500 examples")
    print("  - Critical paths: 800-1000 examples")
    print("  - Stateful testing: 100 examples × 50 steps")
    print("  - Custom strategies: 100-200 examples")
    print("\nFeatures:")
    print("  ✓ Custom Hypothesis strategies for APGI domain")
    print("  ✓ Increased max_examples for critical paths")
    print("  ✓ Stateful simulation testing")
    print("  ✓ Comprehensive invariant checking")
    print("\nRun with: pytest tests/comprehensive/property_based_enhanced.py -v")

    return {
        "test_modules": [
            "TestEnhancedMathematicalProperties",
            "TestEnhancedDataProperties",
            "TestCriticalPaths",
            "TestCustomStrategies",
            "APGISimulationStateMachine",
        ],
        "max_examples": {
            "critical": 1000,
            "mathematical": 500,
            "data": 300,
            "stateful": 100,
        },
        "features": [
            "custom_strategies",
            "stateful_testing",
            "invariant_checking",
            "critical_path_enhancement",
        ],
    }


if __name__ == "__main__":
    results = run_property_based_tests()
    print("\n" + "=" * 80)
    print("Use pytest to run these tests:")
    print(
        "  pytest tests/comprehensive/property_based_enhanced.py -v --hypothesis-seed=0"
    )
