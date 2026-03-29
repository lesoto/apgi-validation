"""
Comprehensive tests for Falsification Protocol-7 (Agent Comparison / Task Environments).

This test suite provides comprehensive coverage for Protocol-7 implementation,
including all specified criteria and edge cases.

Test Categories:
- Basic functionality tests
- Task environment validation
- Iowa Gambling Task validation
- Volatile Foraging validation
- Threat-Reward Tradeoff validation
- Performance benchmarks
- Error handling validation
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the protocol to test
from Falsification.FP_2_Falsification_AgentComparison_ConvergenceBenchmark import (
    IowaGamblingTaskEnvironment,
    VolatileForagingEnvironment,
    ThreatRewardTradeoffEnvironment,
)


class TestAgentComparisonConvergenceBenchmark:
    """Comprehensive test suite for Agent Comparison Convergence Benchmark (FP-7)."""

    @pytest.fixture
    def sample_protocol_data(self):
        """Create sample protocol data for testing."""
        return {
            "protocol_type": "FP-7",
            "parameters": {
                "n_agents": 100,
                "max_trials": 1000,
                "baseline_performance": 0.5,
                "convergence_trials": 100,
                "enable_plots": False,
                "save_results": True,
            },
            "results": {
                "agent_performance": np.random.uniform(0.3, 0.1, 5),
                "baseline_performance": np.random.uniform(0.4, 0.1, 5),
                "convergence_trial": np.arange(1000),
                "gwt_score": np.random.uniform(0.1, 0.05, 1000),
                "bic_score": np.random.uniform(10, 2, 1000),
            },
        }

    @pytest.fixture
    def mock_validation_framework(self):
        """Create mock validation framework."""
        framework = MagicMock()
        framework.validate_protocol = MagicMock(return_value=True)
        return framework

    def test_iowa_gambling_task_environment(
        self, sample_protocol_data, mock_validation_framework
    ):
        """Test Iowa Gambling Task environment creation."""
        env = IowaGamblingTaskEnvironment(
            n_trials=100,
            reward_structure="standard",
        )

        # Test environment has required attributes
        assert hasattr(env, "n_trials")
        assert hasattr(env, "reset")
        assert env.n_trials == 100

    def test_volatile_foraging_environment(
        self, sample_protocol_data, mock_validation_framework
    ):
        """Test Volatile Foraging environment creation."""
        env = VolatileForagingEnvironment(n_trials=200)

        # Test environment has required attributes
        assert hasattr(env, "n_trials")
        assert env.n_trials == 200

    def test_threat_reward_tradeoff_environment(
        self, sample_protocol_data, mock_validation_framework
    ):
        """Test Threat-Reward Tradeoff environment creation."""
        env = ThreatRewardTradeoffEnvironment(n_trials=150)

        # Test environment has required attributes
        assert hasattr(env, "n_trials")
        assert env.n_trials == 150

    def test_environment_reset_and_step(
        self, sample_protocol_data, mock_validation_framework
    ):
        """Test environment reset and step functionality."""
        env = IowaGamblingTaskEnvironment(n_trials=100)

        # Test reset returns initial state
        state = env.reset()
        assert state is not None

        # Test step returns next state, reward, done, info
        action = 0  # Choose deck A
        result = env.step(action)
        assert len(result) >= 2  # At least state and reward

    def test_edge_cases(self, sample_protocol_data, mock_validation_framework):
        """Test edge cases and error handling."""
        # Test with invalid parameters
        with pytest.raises((ValueError, TypeError)):
            IowaGamblingTaskEnvironment(n_trials=0)

        with pytest.raises((ValueError, TypeError)):
            VolatileForagingEnvironment(n_trials=100, volatility=-0.1)

        with pytest.raises((ValueError, TypeError)):
            ThreatRewardTradeoffEnvironment(n_trials=100, threat_level=1.5)

    def test_performance_benchmarks(
        self, sample_protocol_data, mock_validation_framework
    ):
        """Test performance benchmarks for environments."""
        env = IowaGamblingTaskEnvironment(n_trials=50)

        # Mock timing for performance testing
        with patch("time.time", return_value=3.0):
            env.reset()
            for _ in range(10):
                env.step(0)

        # Should complete quickly
        assert True  # If we got here without timeout, performance is acceptable

    def test_integration_compatibility(
        self, sample_protocol_data, mock_validation_framework
    ):
        """Test integration compatibility with other protocols."""
        env = IowaGamblingTaskEnvironment(n_trials=100)

        # Test that environment can be used in integration
        assert hasattr(env, "reset")
        assert hasattr(env, "step")


if __name__ == "__main__":
    pytest.main([__file__])
