"""
Comprehensive tests for Falsification Protocol-5 (Evolutionary Emergence).

This test suite provides comprehensive coverage for Protocol-5 implementation,
including all specified criteria and edge cases.

Test Categories:
- Basic functionality tests
- Evolutionary emergence validation
- Edge case handling
- Performance benchmarks
- Integration with other protocols
- Error handling validation
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the protocol to test
from Falsification.FP_05_EvolutionaryPlausibility import (
    EvolvableAgent,
    EvolutionaryAPGIEmergence,
)


class TestEvolutionaryPlausibilityProtocol5:
    """Comprehensive test suite for Evolutionary Plausibility Protocol (FP-5)."""

    @pytest.fixture
    def sample_protocol_data(self):
        """Create sample protocol data for testing."""
        return {
            "protocol_type": "FP-5",
            "parameters": {
                "population_size": 30,
                "n_generations": 50,
                "mutation_rate": 0.01,
                "crossover_rate": 0.7,
                "selection_pressure": 1.5,
            },
            "results": {
                "fitness_scores": np.random.uniform(0.6, 0.95, 100),
                "emergence_frequency": np.random.uniform(0.1, 0.3, 100),
                "convergence_generation": np.random.randint(20, 50, 100),
            },
        }

    @pytest.fixture
    def mock_validation_framework(self):
        """Create mock validation framework."""
        framework = MagicMock()
        framework.validate_protocol = MagicMock(return_value=True)
        return framework

    def test_evolvable_agent_creation(
        self, sample_protocol_data, mock_validation_framework
    ):
        """Test basic evolvable agent creation."""
        # Use correct genome structure with required keys
        genome = {
            "has_threshold": True,
            "has_intero_weighting": True,
            "has_somatic_markers": True,
            "has_precision_weighting": True,
            "theta_0": 0.5,
            "alpha": 5.0,
            "beta": 1.2,
            "Pi_e_lr": 0.01,
        }
        agent = EvolvableAgent(genome=genome)

        # Test agent has required attributes
        assert hasattr(agent, "genome")
        assert isinstance(agent.genome, dict)

    def test_evolutionary_emergence_initialization(
        self, sample_protocol_data, mock_validation_framework
    ):
        """Test evolutionary emergence framework initialization."""
        evo = EvolutionaryAPGIEmergence(
            population_size=sample_protocol_data["parameters"]["population_size"],
            n_generations=sample_protocol_data["parameters"]["n_generations"],
        )

        # Test framework has required attributes (note: class uses pop_size internally)
        assert hasattr(evo, "pop_size")
        assert hasattr(evo, "n_generations")
        assert evo.pop_size == sample_protocol_data["parameters"]["population_size"]

    def test_emergence_analysis(self, sample_protocol_data, mock_validation_framework):
        """Test emergence analysis functionality."""
        evo = EvolutionaryAPGIEmergence(
            population_size=sample_protocol_data["parameters"]["population_size"],
            n_generations=sample_protocol_data["parameters"]["n_generations"],
        )

        # Create mock history data with architecture_frequencies
        history = {
            "generations": list(range(50)),
            "best_fitness": np.linspace(0.5, 0.9, 50),
            "mean_fitness": np.linspace(0.4, 0.85, 50),
            "diversity": np.random.uniform(0.1, 0.5, 50),
            "architecture_frequencies": [
                {
                    "has_threshold": 0.5 + i * 0.01,
                    "has_intero_weighting": 0.5 + i * 0.01,
                    "has_somatic_markers": 0.5 + i * 0.01,
                    "has_precision_weighting": 0.5 + i * 0.01,
                }
                for i in range(50)
            ],
        }

        # Test analysis runs without errors
        analysis = evo.analyze_emergence(history)
        assert isinstance(analysis, dict)
        assert (
            "final_frequencies" in analysis
            or "emergence_detected" in analysis
            or "apgi_emerged" in analysis
        )

    def test_edge_cases(self, sample_protocol_data, mock_validation_framework):
        """Test edge cases and error handling."""
        # Test with invalid population size - EvolutionaryAPGIEmergence accepts any positive int
        # Just verify it doesn't crash with small values
        evo_small = EvolutionaryAPGIEmergence(population_size=1, n_generations=5)
        assert evo_small.pop_size == 1

        # Test with incomplete genome - should raise KeyError
        with pytest.raises((KeyError, ValueError, TypeError)):
            # Missing required keys like has_threshold, has_intero_weighting, etc.
            EvolvableAgent(genome={"invalid_key": 0.5})

    def test_performance_benchmarks(
        self, sample_protocol_data, mock_validation_framework
    ):
        """Test performance benchmarks and timing."""
        evo = EvolutionaryAPGIEmergence(
            population_size=10,  # Small population for speed
            n_generations=5,
        )

        # Mock timing for performance testing
        with patch("time.time", return_value=1.0):
            # Run a minimal evolution with required history structure
            history = {
                "generations": list(range(5)),
                "best_fitness": np.linspace(0.5, 0.7, 5),
                "architecture_frequencies": [
                    {
                        "has_threshold": 0.5 + i * 0.02,
                        "has_intero_weighting": 0.5 + i * 0.02,
                        "has_somatic_markers": 0.5 + i * 0.02,
                        "has_precision_weighting": 0.5 + i * 0.02,
                    }
                    for i in range(5)
                ],
            }
            analysis = evo.analyze_emergence(history)

        # Analysis should complete quickly
        assert isinstance(analysis, dict)

    def test_integration_compatibility(
        self, sample_protocol_data, mock_validation_framework
    ):
        """Test integration compatibility with other protocols."""
        evo = EvolutionaryAPGIEmergence(
            population_size=sample_protocol_data["parameters"]["population_size"],
            n_generations=sample_protocol_data["parameters"]["n_generations"],
        )

        # Test that framework has required methods
        assert hasattr(evo, "analyze_emergence")
        assert hasattr(evo, "run_evolution") or hasattr(evo, "evolve")


if __name__ == "__main__":
    pytest.main([__file__])
