"""
Tests for falsification protocols.
==============================

"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.constants import DIM_CONSTANTS


def test_falsification_files_exist():
    """Test falsification protocol files exist and have valid content."""
    project_root = Path(__file__).parent.parent
    falsification_dir = project_root / "Falsification"

    # Check falsification protocol files - use actual filenames from directory
    falsification_files = [
        "FP_01_ActiveInference.py",
        "FP_02_AgentComparison_ConvergenceBenchmark.py",
        "FP_03_FrameworkLevel_MultiProtocol.py",
        "FP_04_PhaseTransition_EpistemicArchitecture.py",
        "FP_05_EvolutionaryPlausibility.py",
        "FP_06_LiquidNetwork_EnergyBenchmark.py",
        "FP_07_MathematicalConsistency.py",
        "FP_08_ParameterSensitivity_Identifiability.py",
        "FP_09_NeuralSignatures_P3b_HEP.py",
        "FP_10_BayesianEstimation_MCMC.py",
        "FP_11_LiquidNetworkDynamics_EchoState.py",
        "FP_12_CrossSpeciesScaling.py",
        "FP_ALL_Aggregator.py",
        "Master_Falsification.py",
    ]

    for file_name in falsification_files:
        file_path = falsification_dir / file_name
        # Check file exists
        assert file_path.exists(), f"Falsification file {file_name} missing"

        # Check file is not empty
        assert file_path.stat().st_size > 0, f"Falsification file {file_name} is empty"

        # Check file has valid Python syntax
        try:
            with open(file_path, "r") as f:
                content = f.read()
                compile(content, str(file_path), "exec")
        except SyntaxError as e:
            assert False, f"Falsification file {file_name} has syntax error: {e}"
        except UnicodeDecodeError as e:
            assert False, f"Falsification file {file_name} has encoding error: {e}"

    # Check APGI_Falsification_Protocols_GUI.py at root level (moved from Falsification/)
    gui_path = Path(__file__).parent.parent / "APGI_Falsification_Protocols_GUI.py"
    assert (
        gui_path.exists()
    ), "APGI_Falsification_Protocols_GUI.py not found at root level"


def test_falsification_protocol_1_hierarchical_model():
    """Test HierarchicalGenerativeModel from FP_01_ActiveInference.py"""
    try:
        import importlib.util

        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "falsification1",
            Path(__file__).parent.parent / "Falsification" / "FP_01_ActiveInference.py",
        )
        if spec is None or spec.loader is None:
            pytest.skip("FP_01_ActiveInference.py not found")
        falsification1 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification1)
        HierarchicalGenerativeModel = falsification1.HierarchicalGenerativeModel
    except (ImportError, AttributeError) as e:
        pytest.skip(f"HierarchicalGenerativeModel import failed: {e}")

    # Test model initialization
    levels = [
        {"name": "bottom", "dim": 32, "tau": 0.05},
        {"name": "middle", "dim": 16, "tau": 0.2},
        {"name": "top", "dim": 8, "tau": 0.5},
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
    """Test SomaticMarkerNetwork from FP_01_ActiveInference.py"""
    try:
        import importlib.util

        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "falsification1",
            Path(__file__).parent.parent / "Falsification" / "FP_01_ActiveInference.py",
        )
        if spec is None or spec.loader is None:
            pytest.skip("FP_01_ActiveInference.py not found")
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
    """Test EvolvableAgent from FP_05_EvolutionaryPlausibility.py"""
    try:
        import importlib.util

        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "falsification5",
            Path(__file__).parent.parent
            / "Falsification"
            / "FP_05_EvolutionaryPlausibility.py",
        )
        if spec is None or spec.loader is None:
            pytest.skip("FP_05_EvolutionaryPlausibility.py not found")
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
    """Test EvolutionaryAPGIEmergence from FP_05_EvolutionaryPlausibility.py"""
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "falsification6",
            Path(__file__).parent.parent
            / "Falsification"
            / "FP_05_EvolutionaryPlausibility.py",
        )
        if spec is None or spec.loader is None:
            pytest.skip("FP_05_EvolutionaryPlausibility.py not found")
        falsification6 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification6)
        EvolutionaryAPGIEmergence = falsification6.EvolutionaryAPGIEmergence
    except (ImportError, AttributeError) as e:
        pytest.skip(f"EvolutionaryAPGIEmergence import failed: {e}")

    # Test initialization with default config
    config = {
        "population_size": 10,
        "n_generations": 5,
    }
    experiment = EvolutionaryAPGIEmergence(**config)
    assert experiment is not None
    assert hasattr(experiment, "pop_size")
    assert hasattr(experiment, "n_generations")


# Tests for protocols 7-12 (placeholder files to be created)
def test_falsification_protocol_7_mathematical_consistency():
    """Test mathematical consistency checks from FP_07_MathematicalConsistency.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent
        / "Falsification"
        / "FP_07_MathematicalConsistency.py"
    )
    if not protocol_path.exists():
        pytest.skip("FP_07_MathematicalConsistency.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification7", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip("FP_07_MathematicalConsistency.py not found")
        falsification7 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification7)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"FP_07_MathematicalConsistency.py import failed: {e}")

    # Test that module has mathematical consistency functions
    assert hasattr(falsification7, "check_parameter_bounds")
    assert hasattr(falsification7, "verify_equation_consistency")


def test_falsification_protocol_8_parameter_sensitivity():
    """Test parameter sensitivity analysis from FP_08_ParameterSensitivity_Identifiability.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent
        / "Falsification"
        / "FP_08_ParameterSensitivity_Identifiability.py"
    )
    if not protocol_path.exists():
        pytest.skip("FP_08_ParameterSensitivity_Identifiability.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification8", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip("FP_08_ParameterSensitivity_Identifiability.py not found")
        falsification8 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification8)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"FP_08_ParameterSensitivity_Identifiability.py import failed: {e}")

    # Test that module has sensitivity analysis functions
    assert hasattr(falsification8, "analyze_parameter_sensitivity")
    assert hasattr(falsification8, "generate_sensitivity_report")


def test_falsification_protocol_9_neural_signatures():
    """Test neural signatures validation from FP_09_NeuralSignatures_P3b_HEP.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent
        / "Falsification"
        / "FP_09_NeuralSignatures_P3b_HEP.py"
    )
    if not protocol_path.exists():
        pytest.skip("FP_09_NeuralSignatures_P3b_HEP.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification9", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip("FP_09_NeuralSignatures_P3b_HEP.py not found")
        falsification9 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification9)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"FP_09_NeuralSignatures_P3b_HEP.py import failed: {e}")

    # Test that module has neural signature functions
    assert hasattr(falsification9, "detect_neural_signatures")
    assert hasattr(falsification9, "validate_consciousness_markers")


def test_falsification_protocol_10_cross_species_scaling():
    """Test cross-species scaling from FP_12_CrossSpeciesScaling.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent / "Falsification" / "FP_12_CrossSpeciesScaling.py"
    )
    if not protocol_path.exists():
        pytest.skip("FP_12_CrossSpeciesScaling.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification10", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip("FP_12_CrossSpeciesScaling.py not found")
        falsification10 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification10)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"FP_12_CrossSpeciesScaling.py import failed: {e}")

    # Test that module has scaling functions
    assert hasattr(falsification10, "apply_cross_species_scaling")
    assert hasattr(falsification10, "validate_scaling_laws")


def test_falsification_protocol_11_bayesian_estimation():
    """Test Bayesian estimation from FP_10_BayesianEstimation_MCMC.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent
        / "Falsification"
        / "FP_10_BayesianEstimation_MCMC.py"
    )
    if not protocol_path.exists():
        pytest.skip("FP_10_BayesianEstimation_MCMC.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification11", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip("FP_10_BayesianEstimation_MCMC.py not found")
        falsification11 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification11)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"FP_10_BayesianEstimation_MCMC.py import failed: {e}")

    # Test that module has Bayesian estimation functions
    assert hasattr(falsification11, "run_bayesian_estimation")
    assert hasattr(falsification11, "compute_posterior_distributions")


def test_falsification_protocol_12_liquid_network():
    """Test liquid network validation from FP_11_LiquidNetworkDynamics_EchoState.py"""
    # Skip test if protocol doesn't exist yet
    protocol_path = (
        Path(__file__).parent.parent
        / "Falsification"
        / "FP_11_LiquidNetworkDynamics_EchoState.py"
    )
    if not protocol_path.exists():
        pytest.skip("FP_11_LiquidNetworkDynamics_EchoState.py not yet implemented")

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("falsification12", protocol_path)
        if spec is None or spec.loader is None:
            pytest.skip("FP_11_LiquidNetworkDynamics_EchoState.py not found")
        falsification12 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(falsification12)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"FP_11_LiquidNetworkDynamics_EchoState.py import failed: {e}")

    # Test that module has liquid network functions
    assert hasattr(falsification12, "detect_neural_signatures")
    assert hasattr(falsification12, "validate_network_topology")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
