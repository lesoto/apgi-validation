"""
Tests for falsification protocols.
==================================
"""

import pytest
from pathlib import Path


def test_falsification_files_exist():
    """Test that falsification protocol files exist."""
    project_root = Path(__file__).parent.parent
    falsification_dir = project_root / "Falsification"

    # Check falsification protocol files
    falsification_files = [
        "Falsification-Protocol-1.py",
        "Falsification-Protocol-2.py",
        "Falsification-Protocol-3.py",
        "Falsification-Protocol-4.py",
        "Falsification-Protocol-5.py",
        "Falsification-Protocol-6.py",
        "APGI-Falsification-Protocol-GUI.py",
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
    network = SomaticMarkerNetwork(state_dim=64, action_dim=4, hidden_dim=32)

    # Test basic properties
    assert hasattr(network, "network")
    assert hasattr(network, "optimizer")

    # Test prediction
    state = np.random.randn(64)
    predictions = network.predict(state)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == 4

    # Test update
    network.update(state, action=0, prediction_error=0.1)

    # Verify predictions changed
    new_predictions = network.predict(state)
    assert not np.array_equal(predictions, new_predictions)


# Placeholder tests for protocols 2-6 (to be expanded)
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
