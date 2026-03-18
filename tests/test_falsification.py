"""
Tests for falsification protocols.
==============================

"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.constants import DIM_CONSTANTS


def test_falsification_files_exist():
    """Test that falsification protocol files exist."""
    project_root = Path(__file__).parent.parent
    falsification_dir = project_root / "Falsification"

    # Check falsification protocol files - use actual filenames from directory
    falsification_files = [
        "APGI-Falsification-Aggregator.py",
        "APGI-Falsification-Protocols-GUI.py",
        "CausalManipulations-TMS-Pharmacological-Priority2.py",
        "Falsification-ActiveInferenceAgents-F1F2.py",
        "Falsification-AgentComparison-ConvergenceBenchmark.py",
        "Falsification-BayesianEstimation-MCMC.py",
        "Falsification-BayesianEstimation-ParameterRecovery.py",
        "Falsification-CrossSpeciesScaling-P12.py",
        "Falsification-EvolutionaryPlausibility-Standard6.py",
        "Falsification-FrameworkLevel-MultiProtocol.py",
        "Falsification-InformationTheoretic-PhaseTransition.py",
        "Falsification-LiquidNetworkDynamics-EchoState.py",
        "Falsification-MathematicalConsistency-Equations.py",
        "Falsification-NeuralNetwork-EnergyBenchmark.py",
        "Falsification-NeuralSignatures-EEG-P3b-HEP.py",
        "Falsification-ParameterSensitivity-Identifiability.py",
        "Falsification_AgentComparison_ConvergenceBenchmark.py",
    ]

    for file_name in falsification_files:
        file_path = falsification_dir / file_name
        assert file_path.exists(), f"Falsification file {file_name} missing"


def test_falsification_protocol_1_hierarchical_model():
    """Test HierarchicalGenerativeModel from Falsification-Protocol-1.py"""
    try:
        import importlib.util
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "falsification1",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-1.py",
        )
        falsification1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification1)
        HierarchicalGenerativeModel = falsification1.HierarchicalGenerativeModel
    except ImportError as e:
        pytest.skip(f"HierarchicalGenerativeModel import failed: {e}")

    # Test model initialization
    levels = [
        {"name": "bottom", "dim": 32},
        {"name": "middle", "dim": 16},
        {"name": "top", "dim": 8},
    ]

    model = HierarchicalGenerativeModel(levels, model_type="extero")

    # Test basic properties
    assert len(model.states) == 3
    assert len(model.weights) == 3

    # Test prediction
    prediction = model.predict()
    assert isinstance(prediction, np.ndarray)
    assert len(prediction) == 32  # EXTERO_DIM

    # Test update with error
    error = np.random.randn(32)
    model.update(error)

    # Test get_level
    level_state = model.get_level("bottom")
    assert isinstance(level_state, np.ndarray)
    assert len(level_state) == 32

    # Test get_all_levels
    all_states = model.get_all_levels()
    assert isinstance(all_states, np.ndarray)
    assert len(all_states) == 56  # 32 + 16 + 8


def test_falsification_protocol_1_somatic_marker_network():
    """Test SomaticMarkerNetwork from Falsification-Protocol-1.py"""
    try:
        import importlib.util
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "falsification1",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-1.py",
        )
        falsification1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification1)
        SomaticMarkerNetwork = falsification1.SomaticMarkerNetwork
    except ImportError as e:
        pytest.skip(f"SomaticMarkerNetwork import failed: {e}")

    # Test initialization
    network = SomaticMarkerNetwork(
        state_dim=DIM_CONSTANTS.EXTERO_DIM,
        action_dim=DIM_CONSTANTS.ACTION_DIM,
        hidden_dim=DIM_CONSTANTS.SOMATIC_HIDDEN_DIM,
    )

    # Test basic properties
    assert hasattr(network, "network")
    assert hasattr(network, "optimizer")

    # Test prediction
    state = np.random.randn(DIM_CONSTANTS.EXTERO_DIM)
    predictions = network.predict(state)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == DIM_CONSTANTS.ACTION_DIM

    # Test update
    network.update(state, action=0, prediction_error=0.1)

    # Verify predictions changed
    new_predictions = network.predict(state)
    assert not np.array_equal(predictions, new_predictions)


def test_falsification_protocol_5_evolvable_agent():
    """Test EvolvableAgent from Falsification-Protocol-5.py"""
    try:
        import importlib.util
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "falsification5",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-5.py",
        )
        falsification5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification5)
        EvolvableAgent = falsification5.EvolvableAgent
    except ImportError as e:
        pytest.skip(f"EvolvableAgent import failed: {e}")

    # Test initialization with different genomes
    genome_full = {
        "has_threshold": True,
        "has_intero_weighting": True,
        "has_somatic_markers": True,
        "has_precision_weighting": True,
        "theta_0": 0.5,
        "alpha": 5.0,
        "beta": 1.2,
        "Pi_e_lr": 0.01,  # Add missing precision learning rate
    }
    agent_full = EvolvableAgent(genome_full)
    assert agent_full.has_threshold is True
    assert agent_full.has_intero_weighting is True
    assert agent_full.has_somatic_markers is True
    assert agent_full.has_precision_weighting is True

    # Test basic evolution step
    observation = {"extero": np.random.randn(32), "intero": np.random.randn(16)}
    reward = agent_full.step(observation=observation, dt=0.05)
    assert isinstance(reward, (int, float))

    # Test genome with minimal features
    genome_minimal = {
        "has_threshold": False,
        "has_intero_weighting": False,
        "has_somatic_markers": False,
        "has_precision_weighting": False,
        "theta_0": 0.5,
        "alpha": 5.0,
        "beta": 1.2,
        "Pi_e_lr": 0.01,  # Add missing precision learning rate
    }
    agent_minimal = EvolvableAgent(genome_minimal)
    assert agent_minimal.has_threshold is False

    # Test threshold behavior
    agent_minimal.surprise = 1.0  # Above threshold
    conscious_access = agent_minimal.conscious_access
    assert conscious_access is True  # Always conscious when no threshold

    # Test policy network interaction
    state = np.random.randn(48)  # EXTERO (32) + INTERO (16) dimensions
    action = agent_minimal.get_action(state)
    assert isinstance(action, (int, np.integer))

    # Test learning dynamics
    old_surprise = agent_minimal.surprise
    agent_minimal.update_surprise(0.8)
    assert agent_minimal.surprise != old_surprise


def test_falsification_protocol_5_exists():
    """Test that Falsification-Protocol-5.py exists and can be imported"""
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-5.py"
    )
    assert protocol_path.exists()

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification5", protocol_path)
        falsification5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification5)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-5.py import failed: {e}")


def test_falsification_protocol_6_network_comparison():
    """Test NetworkComparisonExperiment from Falsification-Protocol-6.py"""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "falsification6",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-6.py",
        )
        falsification6 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification6)
        NetworkComparisonExperiment = falsification6.NetworkComparisonExperiment
    except ImportError as e:
        pytest.skip(f"NetworkComparisonExperiment import failed: {e}")

    # Test initialization with default config
    config = {
        "extero_dim": DIM_CONSTANTS.EXTERO_DIM,
        "intero_dim": DIM_CONSTANTS.INTERO_DIM,
        "action_dim": DIM_CONSTANTS.ACTION_DIM,
        "context_dim": DIM_CONSTANTS.CONTEXT_DIM,
        "n_episodes": 10,
    }
    experiment = NetworkComparisonExperiment(config)
    assert experiment.config is not None

    # Test that experiment can be run
    results = experiment.run_experiment()
    assert isinstance(results, dict)

    # Check for expected result structure
    if "error" not in results:
        # Results should contain task results with network accuracies
        assert len(results) > 0
        for task_name, task_results in results.items():
            assert isinstance(task_results, dict)
            for net_name, net_results in task_results.items():
                assert isinstance(net_results, dict)
                assert "accuracy" in net_results or "auc" in net_results


def test_falsification_protocol_6_exists():
    """Test that Falsification-Protocol-6.py exists and can be imported"""
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-6.py"
    )
    assert protocol_path.exists()

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification6", protocol_path)
        falsification6 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification6)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-6.py import failed: {e}")


# Tests for protocols 7-12 (placeholder files to be created)
def test_falsification_protocol_7_mathematical_consistency():
    """Test mathematical consistency checks from Falsification-Protocol-7.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-7.py"
    )
    if not protocol_path.exists():
        pytest.skip("Falsification-Protocol-7.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification7", protocol_path)
        falsification7 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification7)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-7.py import failed: {e}")

    # Test that module has mathematical consistency functions
    assert hasattr(falsification7, "check_parameter_bounds")
    assert hasattr(falsification7, "verify_equation_consistency")


def test_falsification_protocol_8_parameter_sensitivity():
    """Test parameter sensitivity analysis from Falsification-Protocol-8.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-8.py"
    )
    if not protocol_path.exists():
        pytest.skip("Falsification-Protocol-8.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification8", protocol_path)
        falsification8 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification8)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-8.py import failed: {e}")

    # Test that module has sensitivity analysis functions
    assert hasattr(falsification8, "analyze_parameter_sensitivity")
    assert hasattr(falsification8, "generate_sensitivity_report")


def test_falsification_protocol_9_neural_signatures():
    """Test neural signatures validation from Falsification-Protocol-9.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-9.py"
    )
    if not protocol_path.exists():
        pytest.skip("Falsification-Protocol-9.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification9", protocol_path)
        falsification9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification9)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-9.py import failed: {e}")

    # Test that module has neural signature functions
    assert hasattr(falsification9, "detect_neural_signatures")
    assert hasattr(falsification9, "validate_consciousness_markers")


def test_falsification_protocol_10_cross_species_scaling():
    """Test cross-species scaling from Falsification-Protocol-10.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-10.py"
    )
    if not protocol_path.exists():
        pytest.skip("Falsification-Protocol-10.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification10", protocol_path)
        falsification10 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification10)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-10.py import failed: {e}")

    # Test that module has scaling functions
    assert hasattr(falsification10, "apply_cross_species_scaling")
    assert hasattr(falsification10, "validate_scaling_laws")


def test_falsification_protocol_11_bayesian_estimation():
    """Test Bayesian estimation from Falsification-Protocol-11.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-11.py"
    )
    if not protocol_path.exists():
        pytest.skip("Falsification-Protocol-11.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification11", protocol_path)
        falsification11 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification11)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-11.py import failed: {e}")

    # Test that module has Bayesian estimation functions
    assert hasattr(falsification11, "run_bayesian_estimation")
    assert hasattr(falsification11, "compute_posterior_distributions")


def test_falsification_protocol_12_liquid_network():
    """Test liquid network validation from Falsification-Protocol-12.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-12.py"
    )
    if not protocol_path.exists():
        pytest.skip("Falsification-Protocol-12.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification12", protocol_path)
        falsification12 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification12)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-12.py import failed: {e}")

    # Test that module has liquid network functions
    assert hasattr(falsification12, "test_liquid_network_properties")
    assert hasattr(falsification12, "validate_network_topology")


def test_falsification_protocol_2_iowa_gambling_environment():
    """Test IowaGamblingTaskEnvironment from Falsification-Protocol-2.py"""
    try:
        import importlib.util
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "falsification2",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-2.py",
        )
        falsification2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification2)
        IowaGamblingTaskEnvironment = falsification2.IowaGamblingTaskEnvironment
    except ImportError as e:
        pytest.skip(f"IowaGamblingTaskEnvironment import failed: {e}")

    # Test initialization
    env = IowaGamblingTaskEnvironment(n_trials=10)

    # Test basic properties
    assert hasattr(env, "n_trials")
    assert hasattr(env, "trial")
    assert hasattr(env, "decks")
    assert env.n_trials == 10
    assert env.trial == 0
    assert len(env.decks) == 4  # A, B, C, D decks

    # Test reset
    initial_obs = env.reset()
    assert isinstance(initial_obs, dict)
    assert "extero" in initial_obs
    assert "intero" in initial_obs
    assert isinstance(initial_obs["extero"], np.ndarray)
    assert isinstance(initial_obs["intero"], np.ndarray)
    assert env.trial == 0

    # Test step with different actions
    for action in range(4):  # Test all decks A, B, C, D
        env.reset()  # Reset for each test
        reward, intero_cost, observation, done = env.step(action)

        # Verify return types
        assert isinstance(reward, (int, float))
        assert isinstance(intero_cost, (int, float))
        assert isinstance(observation, dict)
        assert isinstance(done, bool)

        # Verify observation structure
        assert "extero" in observation
        assert "intero" in observation
        assert isinstance(observation["extero"], np.ndarray)
        assert isinstance(observation["intero"], np.ndarray)

        # Verify reasonable ranges
        assert intero_cost >= 0  # Cost should be non-negative

        # Check if episode ends after n_trials
        assert not done  # Should not be done after single step
        assert env.trial == 1

    # Test multiple steps lead to completion
    env.reset()
    for i in range(9):  # 9 more steps to reach n_trials=10
        _, _, _, done = env.step(0)
        assert not done
    _, _, _, done = env.step(0)  # 10th step
    assert done

    # Test invalid action
    env.reset()
    with pytest.raises(ValueError):
        env.step(4)  # Invalid action


def test_falsification_protocol_2_exists():
    """Test that Falsification-Protocol-2.py exists and can be imported"""
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-2.py"
    )
    assert protocol_path.exists()

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification2", protocol_path)
        falsification2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification2)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-2.py import failed: {e}")


def test_falsification_protocol_3_exists():
    """Test that Falsification-Protocol-3.py exists and can be imported"""
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-3.py"
    )
    assert protocol_path.exists()

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification3", protocol_path)
        falsification3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification3)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-3.py import failed: {e}")


def test_falsification_protocol_4_exists():
    """Test that Falsification-Protocol-4.py exists and can be imported"""
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-4.py"
    )
    assert protocol_path.exists()

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification4", protocol_path)
        falsification4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification4)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-4.py import failed: {e}")


def test_falsification_protocol_5_evolutionary_emergence():
    """Test EvolvableAgent from Falsification-Protocol-5.py"""
    try:
        import importlib.util
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "falsification5",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-5.py",
        )
        falsification5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification5)
        EvolvableAgent = falsification5.EvolvableAgent
    except ImportError as e:
        pytest.skip(f"EvolvableAgent import failed: {e}")

    # Test initialization with different genomes
    genome_full = {
        "has_threshold": True,
        "has_intero_weighting": True,
        "has_somatic_markers": True,
        "has_precision_weighting": True,
        "theta_0": 0.5,
        "alpha": 5.0,
        "beta": 1.2,
    }
    agent_full = EvolvableAgent(genome_full)
    assert agent_full.has_threshold is True
    assert agent_full.has_intero_weighting is True

    # Test basic evolution step
    observation = {"extero": np.random.randn(32), "intero": np.random.randn(16)}
    reward = agent_full.step(observation=observation, dt=0.05)
    assert isinstance(reward, (int, float))

    # Test genome with minimal features
    genome_minimal = {
        "has_threshold": False,
        "has_intero_weighting": False,
        "has_somatic_markers": False,
        "has_precision_weighting": False,
        "theta_0": 0.5,
        "alpha": 5.0,
        "beta": 1.2,
    }
    agent_minimal = EvolvableAgent(genome_minimal)
    assert agent_minimal.has_threshold is False


def test_falsification_protocol_7_mathematical_consistency():
    """Test mathematical consistency checks from Falsification-Protocol-7.py"""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "falsification7",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-7.py",
        )
        falsification7 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification7)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-7.py import failed: {e}")

    # Test that the module has mathematical consistency functions
    assert hasattr(falsification7, "check_parameter_bounds")
    assert hasattr(falsification7, "verify_equation_consistency")


def test_falsification_protocol_8_parameter_sensitivity():
    """Test parameter sensitivity analysis from Falsification-Protocol-8.py"""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "falsification8",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-8.py",
        )
        falsification8 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification8)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-8.py import failed: {e}")

    # Test that the module has sensitivity analysis functions
    assert hasattr(falsification8, "analyze_parameter_sensitivity")
    assert hasattr(falsification8, "generate_sensitivity_report")


def test_falsification_protocol_9_neural_signatures():
    """Test neural signatures validation from Falsification-Protocol-9.py"""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "falsification9",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-9.py",
        )
        falsification9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification9)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-9.py import failed: {e}")

    # Test that the module has neural signature functions
    assert hasattr(falsification9, "detect_neural_signatures")
    assert hasattr(falsification9, "validate_consciousness_markers")


def test_falsification_protocol_10_cross_species_scaling():
    """Test cross-species scaling from Falsification-Protocol-10.py"""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "falsification10",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-10.py",
        )
        falsification10 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification10)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-10.py import failed: {e}")

    # Test that the module has scaling functions
    assert hasattr(falsification10, "apply_cross_species_scaling")
    assert hasattr(falsification10, "validate_scaling_laws")


def test_falsification_protocol_11_bayesian_estimation():
    """Test Bayesian estimation from Falsification-Protocol-11.py"""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "falsification11",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-11.py",
        )
        falsification11 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification11)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-11.py import failed: {e}")

    # Test that the module has Bayesian estimation functions
    assert hasattr(falsification11, "run_bayesian_estimation")
    assert hasattr(falsification11, "compute_posterior_distributions")


def test_falsification_protocol_12_liquid_network():
    """Test liquid network validation from Falsification-Protocol-12.py"""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "falsification12",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification-Protocol-12.py",
        )
        falsification12 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification12)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-12.py import failed: {e}")

    # Test that the module has liquid network functions
    assert hasattr(falsification12, "test_liquid_network_properties")
    assert hasattr(falsification12, "validate_network_topology")


def test_falsification_protocol_6_exists():
    """Test that Falsification-Protocol-6.py exists and can be imported"""
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "Falsification-Protocol-6.py"
    )
    assert protocol_path.exists()

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification6", protocol_path)
        falsification6 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification6)
    except ImportError as e:
        pytest.skip(f"Falsification-Protocol-6.py import failed: {e}")
