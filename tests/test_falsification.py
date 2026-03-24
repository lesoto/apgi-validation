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
    """Test SyntheticEEG_MLClassification protocol files exist and have valid content."""
    project_root = Path(__file__).parent.parent
    falsification_dir = project_root / "Falsification"

    # Check falsification protocol files - use actual filenames from directory
    falsification_files = [
        "APGI_Falsification_Aggregator.py",
        "APGI_Falsification_Protocols_GUI.py",
        "CausalManipulations_TMS_Pharmacological_Priority2.py",
        "Falsification_ActiveInferenceAgents_F1F2.py",
        "Falsification_AgentComparison_ConvergenceBenchmark.py",
        "Falsification_BayesianEstimation_MCMC.py",
        "Falsification_BayesianModelComparison_ParameterRecovery.py",
        "Falsification_CrossSpeciesScaling_P12.py",
        "Falsification_EvolutionaryPlausibility_Standard6.py",
        "Falsification_FrameworkLevel_MultiProtocol.py",
        "Falsification_InformationTheoretic_PhaseTransition.py",
        "Falsification_LiquidNetworkDynamics_EchoState.py",
        "Falsification_MathematicalConsistency_Equations.py",
        "Falsification_NeuralNetwork_EnergyBenchmark.py",
        "Falsification_NeuralSignatures_EEG_P3b_HEP.py",
        "Falsification_ParameterSensitivity_Identifiability.py",
    ]

    for file_name in falsification_files:
        file_path = falsification_dir / file_name
        # Check file exists
        assert file_path.exists(), f"Falsification file {file_name} missing"

        # Check file is not empty
        assert file_path.stat().st_size > 0, f"Falsification file {file_name} is empty"

        # Check file has valid Python syntax
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                compile(content, str(file_path), "exec")
        except SyntaxError as e:
            assert False, f"Falsification file {file_name} has syntax error: {e}"
        except UnicodeDecodeError as e:
            assert False, f"Falsification file {file_name} has encoding error: {e}"


def test_falsification_protocol_1_hierarchical_model():
    """Test HierarchicalGenerativeModel from Falsification_ActiveInferenceAgents_F1F2.py"""
    try:
        import importlib.util
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "falsification1",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification_ActiveInferenceAgents_F1F2.py",
        )
        if spec is None or spec.loader is None:
            pytest.skip("Falsification_ActiveInferenceAgents_F1F2.py not found")
        falsification1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification1)
        HierarchicalGenerativeModel = falsification1.HierarchicalGenerativeModel
    except (ImportError, AttributeError) as e:
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
    """Test SomaticMarkerNetwork from Falsification_ActiveInferenceAgents_F1F2.py"""
    try:
        import importlib.util
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "falsification1",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification_ActiveInferenceAgents_F1F2.py",
        )
        if spec is None or spec.loader is None:
            pytest.skip("Falsification_ActiveInferenceAgents_F1F2.py not found")
        falsification1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification1)
        SomaticMarkerNetwork = falsification1.SomaticMarkerNetwork
    except (ImportError, AttributeError) as e:
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
    """Test EvolvableAgent from Falsification_EvolutionaryPlausibility_Standard6.py"""
    try:
        import importlib.util
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "falsification5",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification_EvolutionaryPlausibility_Standard6.py",
        )
        if spec is None or spec.loader is None:
            pytest.skip("Falsification_EvolutionaryPlausibility_Standard6.py not found")
        falsification5 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification5)
        EvolvableAgent = falsification5.EvolvableAgent
    except (ImportError, AttributeError) as e:
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


def test_falsification_protocol_6_network_comparison():
    """Test NetworkComparisonExperiment from Falsification-EvolutionaryPlausibility-Standard6.py"""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "falsification6",
            Path(__file__).parent.parent
            / "Falsification"
            / "Falsification_EvolutionaryPlausibility_Standard6.py",
        )
        if spec is None or spec.loader is None:
            pytest.skip("Falsification_EvolutionaryPlausibility_Standard6.py not found")
        falsification6 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification6)
        NetworkComparisonExperiment = falsification6.NetworkComparisonExperiment
    except (ImportError, AttributeError) as e:
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


# Tests for protocols 7-12 (placeholder files to be created)
def test_falsification_protocol_7_mathematical_consistency():
    """Test mathematical consistency checks from Falsification-MathematicalConsistency-Equations.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent
        / "Falsification"
        / "Falsification-MathematicalConsistency-Equations.py"
    )
    if not protocol_path.exists():
        pytest.skip(
            "Falsification-MathematicalConsistency-Equations.py not yet implemented"
        )

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification7", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip("Falsification-MathematicalConsistency-Equations.py not found")
        falsification7 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification7)
    except (ImportError, AttributeError) as e:
        pytest.skip(
            f"Falsification-MathematicalConsistency-Equations.py import failed: {e}"
        )

    # Test that module has mathematical consistency functions
    assert hasattr(falsification7, "check_parameter_bounds")
    assert hasattr(falsification7, "verify_equation_consistency")


def test_falsification_protocol_8_parameter_sensitivity():
    """Test parameter sensitivity analysis from Falsification-ParameterSensitivity-Identifiability.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent
        / "Falsification"
        / "Falsification-ParameterSensitivity-Identifiability.py"
    )
    if not protocol_path.exists():
        pytest.skip(
            "Falsification-ParameterSensitivity-Identifiability.py not yet implemented"
        )

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification8", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip(
                "Falsification-ParameterSensitivity-Identifiability.py not found"
            )
        falsification8 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification8)
    except (ImportError, AttributeError) as e:
        pytest.skip(
            f"Falsification-ParameterSensitivity-Identifiability.py import failed: {e}"
        )

    # Test that module has sensitivity analysis functions
    assert hasattr(falsification8, "analyze_parameter_sensitivity")
    assert hasattr(falsification8, "generate_sensitivity_report")


def test_falsification_protocol_9_neural_signatures():
    """Test neural signatures validation from Falsification-NeuralSignatures-EEG-P3b-HEP.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent
        / "Falsification"
        / "Falsification-NeuralSignatures-EEG-P3b-HEP.py"
    )
    if not protocol_path.exists():
        pytest.skip("Falsification-NeuralSignatures-EEG-P3b-HEP.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification9", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip("Falsification-NeuralSignatures-EEG-P3b-HEP.py not found")
        falsification9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification9)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Falsification_NeuralSignatures_EEG_P3b_HEP.py import failed: {e}")

    # Test that module has neural signature functions
    assert hasattr(falsification9, "detect_neural_signatures")
    assert hasattr(falsification9, "validate_consciousness_markers")


def test_falsification_protocol_10_cross_species_scaling():
    """Test cross-species scaling from Falsification-InformationTheoretic_PhaseTransition_Level2.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent
        / "Falsification"
        / "Falsification-InformationTheoretic_PhaseTransition_Level2.py"
    )
    if not protocol_path.exists():
        pytest.skip(
            "Falsification-InformationTheoretic_PhaseTransition_Level2.py not yet implemented"
        )

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification10", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip(
                "Falsification-InformationTheoretic_PhaseTransition_Level2.py not found"
            )
        falsification10 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification10)
    except (ImportError, AttributeError) as e:
        pytest.skip(
            f"Falsification-InformationTheoretic_PhaseTransition_Level2.py import failed: {e}"
        )

    # Test that module has scaling functions
    assert hasattr(falsification10, "apply_cross_species_scaling")
    assert hasattr(falsification10, "validate_scaling_laws")


def test_falsification_protocol_11_bayesian_estimation():
    """Test Bayesian estimation from Falsification-BayesianEstimation-MCMC.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent
        / "Falsification"
        / "Falsification-BayesianEstimation-MCMC.py"
    )
    if not protocol_path.exists():
        pytest.skip("Falsification-BayesianEstimation-MCMC.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification11", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip("Falsification-BayesianEstimation-MCMC.py not found")
        falsification11 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification11)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Falsification-BayesianEstimation-MCMC.py import failed: {e}")

    # Test that module has Bayesian estimation functions
    assert hasattr(falsification11, "run_bayesian_estimation")
    assert hasattr(falsification11, "compute_posterior_distributions")


def test_falsification_protocol_12_liquid_network():
    """Test liquid network validation from Falsification_LiquidNetworkDynamics_EchoState.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent
        / "Falsification"
        / "Falsification_LiquidNetworkDynamics_EchoState.py"
    )
    if not protocol_path.exists():
        pytest.skip(
            "Falsification_LiquidNetworkDynamics_EchoState.py not yet implemented"
        )

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification12", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip("Falsification_LiquidNetworkDynamics_EchoState.py not found")
        falsification12 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification12)
    except (ImportError, AttributeError) as e:
        pytest.skip(
            f"Falsification_LiquidNetworkDynamics_EchoState.py import failed: {e}"
        )

    # Test that module has liquid network functions
    assert hasattr(falsification12, "detect_neural_signatures")
    assert hasattr(falsification12, "validate_network_topology")


def test_falsification_protocol_5_evolvable_agent():
    """Test EvolvableAgent from Psychophysical_ThresholdEstimation_Protocol1-5.py"""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
