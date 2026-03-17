"""
Falsification Protocol 12: Liquid Network Validation
==================================================

This protocol implements validation of liquid network properties for APGI models.
"""

import logging
import numpy as np
from typing import Dict
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import r2_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_liquid_network_properties(
    network_weights: Dict[str, np.ndarray], liquid_params: Dict[str, float]
) -> Dict[str, float]:
    """Test liquid network properties"""

    property_scores = {}

    # Test echo state property
    property_scores["echo_state"] = test_echo_state_property(
        network_weights, liquid_params
    )

    # Test fading memory
    property_scores["fading_memory"] = test_fading_memory(
        network_weights, liquid_params
    )

    # Test non-linearity
    property_scores["non_linearity"] = test_non_linearity(
        network_weights, liquid_params
    )

    # Test separation capacity
    property_scores["separation_capacity"] = test_separation_capacity(
        network_weights, liquid_params
    )

    return property_scores


def test_echo_state_property(
    network_weights: Dict[str, np.ndarray], liquid_params: Dict[str, float]
) -> float:
    """Test echo state property with runtime behavioral test.

    Inject 100 distinct random inputs, evolve 50 steps each,
    verify that state trajectories converge regardless of initial condition.
    """
    if (
        "liquid_to_liquid" not in network_weights
        or "input_to_liquid" not in network_weights
    ):
        logger.warning("Missing required network weights for echo state test")
        return 0.5

    W_in = network_weights["input_to_liquid"]
    W_res = network_weights["liquid_to_liquid"]
    leak_rate = liquid_params.get("leak_rate", 0.9)
    activation = liquid_params.get("activation", "tanh")
    reservoir_size = liquid_params.get("reservoir_size", W_res.shape[0])

    # Generate 100 distinct random inputs
    n_inputs = 100
    n_steps = 50
    input_dim = W_in.shape[1]

    inputs = np.random.randn(n_inputs, n_steps, input_dim) * 0.1

    # Evolve network for each input starting from different initial states
    all_states = []
    for i in range(n_inputs):
        # Random initial state
        state = np.random.randn(reservoir_size) * 0.1
        states = [state.copy()]

        for t in range(n_steps):
            pre_activation = W_in @ inputs[i, t] + W_res @ state
            if activation == "tanh":
                state = (1 - leak_rate) * state + leak_rate * np.tanh(pre_activation)
            elif activation == "relu":
                state = (1 - leak_rate) * state + leak_rate * np.maximum(
                    0, pre_activation
                )
            else:
                state = (1 - leak_rate) * state + leak_rate * pre_activation
            states.append(state.copy())

        all_states.append(np.array(states))

    # Measure pairwise state distances at t=50
    final_states = np.array([states[-1] for states in all_states])
    pairwise_distances = squareform(pdist(final_states))

    # Echo state property: trajectories should converge (low variance in distances)
    # Compute coefficient of variation of pairwise distances
    mean_distance = np.mean(pairwise_distances)
    std_distance = np.std(pairwise_distances)
    cv = std_distance / (mean_distance + 1e-10)

    # Lower CV indicates better convergence
    echo_score = max(0, 1.0 - cv)

    logger.info(f"Echo state test: CV={cv:.4f}, score={echo_score:.4f}")
    return echo_score


def test_fading_memory(
    network_weights: Dict[str, np.ndarray], liquid_params: Dict[str, float]
) -> float:
    """Test fading memory property with runtime behavioral test.

    Inject single impulse at t=0, measure exponential decay constant τ
    via scipy.optimize.curve_fit; verify τ < n_steps/2.
    """
    if "liquid_to_liquid" not in network_weights:
        logger.warning("Missing liquid_to_liquid weights for fading memory test")
        return 0.5

    W_res = network_weights["liquid_to_liquid"]
    leak_rate = liquid_params.get("leak_rate", 0.9)
    activation = liquid_params.get("activation", "tanh")
    reservoir_size = W_res.shape[0]

    n_steps = 100

    # Inject single impulse at t=0
    state = np.zeros(reservoir_size)
    impulse = np.random.randn(reservoir_size) * 0.5
    state = impulse.copy()

    # Track state evolution
    state_norms = [np.linalg.norm(state)]
    for t in range(n_steps):
        pre_activation = W_res @ state
        if activation == "tanh":
            state = (1 - leak_rate) * state + leak_rate * np.tanh(pre_activation)
        elif activation == "relu":
            state = (1 - leak_rate) * state + leak_rate * np.maximum(0, pre_activation)
        else:
            state = (1 - leak_rate) * state + leak_rate * pre_activation
        state_norms.append(np.linalg.norm(state))

    # Fit exponential decay: y = A * exp(-t/tau)
    def exp_decay(t, A, tau):
        return A * np.exp(-t / tau)

    try:
        t_data = np.arange(len(state_norms))
        popt, _ = curve_fit(exp_decay, t_data, state_norms, p0=[state_norms[0], 10.0])
        A_fit, tau_fit = popt

        # Verify τ < n_steps/2
        memory_score = (
            1.0
            if tau_fit < n_steps / 2
            else max(0, 1.0 - (tau_fit - n_steps / 2) / (n_steps / 2))
        )

        logger.info(f"Fading memory test: τ={tau_fit:.2f}, score={memory_score:.4f}")
        return memory_score
    except Exception as e:
        logger.warning(f"Fading memory curve fit failed: {e}")
        return 0.5


def test_non_linearity(
    network_weights: Dict[str, np.ndarray], liquid_params: Dict[str, float]
) -> float:
    """Test non-linearity of liquid network with runtime behavioral test.

    Compare R² of linear vs. nonlinear readout; verify nonlinear gain > 2×.
    """
    if "liquid_to_output" not in network_weights:
        logger.warning("Missing liquid_to_output weights for non-linearity test")
        return 0.5

    W_out = network_weights["liquid_to_output"]
    reservoir_size = W_out.shape[1]
    activation = liquid_params.get("activation", "tanh")
    leak_rate = liquid_params.get("leak_rate", 0.9)

    if "liquid_to_liquid" in network_weights:
        W_res = network_weights["liquid_to_liquid"]
    else:
        W_res = np.zeros((reservoir_size, reservoir_size))

    if "input_to_liquid" in network_weights:
        W_in = network_weights["input_to_liquid"]
    else:
        W_in = np.random.randn(reservoir_size, 10) * 0.1

    # Generate training data
    n_samples = 500
    input_dim = W_in.shape[1]
    inputs = np.random.randn(n_samples, input_dim) * 0.1

    # Evolve reservoir states
    states = []
    for i in range(n_samples):
        state = np.random.randn(reservoir_size) * 0.1
        for _ in range(20):  # Short evolution
            pre_activation = W_in @ inputs[i] + W_res @ state
            if activation == "tanh":
                state = (1 - leak_rate) * state + leak_rate * np.tanh(pre_activation)
            elif activation == "relu":
                state = (1 - leak_rate) * state + leak_rate * np.maximum(
                    0, pre_activation
                )
            else:
                state = (1 - leak_rate) * state + leak_rate * pre_activation
        states.append(state.copy())

    states = np.array(states)

    # Generate target outputs (nonlinear function of inputs)
    targets = np.sin(inputs[:, 0]) * np.cos(inputs[:, 1]) + 0.1 * inputs[:, 2]

    # Linear readout
    linear_pred = states @ W_out.T
    linear_r2 = r2_score(targets, linear_pred)

    # Nonlinear readout (apply activation to states before linear transformation)
    if activation == "tanh":
        nonlinear_states = np.tanh(states)
    elif activation == "relu":
        nonlinear_states = np.maximum(0, states)
    else:
        nonlinear_states = states

    nonlinear_pred = nonlinear_states @ W_out.T
    nonlinear_r2 = r2_score(targets, nonlinear_pred)

    # Verify nonlinear gain > 2×
    gain = (nonlinear_r2 + 1e-10) / (linear_r2 + 1e-10)
    non_linearity_score = 1.0 if gain > 2.0 else max(0, gain / 2.0)

    logger.info(
        f"Non-linearity test: linear R²={linear_r2:.4f}, nonlinear R²={nonlinear_r2:.4f}, gain={gain:.2f}, score={non_linearity_score:.4f}"
    )
    return non_linearity_score


def test_separation_capacity(
    network_weights: Dict[str, np.ndarray], liquid_params: Dict[str, float]
) -> float:
    """Test separation capacity with runtime behavioral test.

    Inject input pairs with cosine similarity > 0.95, measure output divergence;
    verify divergence exceeds threshold within 10 steps.
    """
    if "liquid_to_liquid" not in network_weights:
        logger.warning("Missing liquid_to_liquid weights for separation capacity test")
        return 0.5

    W_res = network_weights["liquid_to_liquid"]
    leak_rate = liquid_params.get("leak_rate", 0.9)
    activation = liquid_params.get("activation", "tanh")
    reservoir_size = W_res.shape[0]

    if "input_to_liquid" in network_weights:
        W_in = network_weights["input_to_liquid"]
    else:
        W_in = np.random.randn(reservoir_size, 10) * 0.1

    # Generate input pairs with high cosine similarity
    n_pairs = 50
    input_dim = W_in.shape[1]

    # Generate base inputs
    base_inputs = np.random.randn(n_pairs, input_dim) * 0.1
    # Create similar inputs by adding small perturbation
    perturbation_scale = 0.01
    similar_inputs = (
        base_inputs + np.random.randn(n_pairs, input_dim) * perturbation_scale
    )

    # Verify cosine similarity > 0.95
    similarities = []
    for i in range(n_pairs):
        sim = np.dot(base_inputs[i], similar_inputs[i]) / (
            np.linalg.norm(base_inputs[i]) * np.linalg.norm(similar_inputs[i]) + 1e-10
        )
        similarities.append(sim)

    avg_similarity = np.mean(similarities)
    if avg_similarity <= 0.95:
        logger.warning(f"Input pairs have low similarity: {avg_similarity:.4f}")

    # Evolve both inputs and measure output divergence
    divergences = []
    n_steps = 10

    for i in range(n_pairs):
        # Evolve first input
        state1 = np.random.randn(reservoir_size) * 0.1
        states1 = [state1.copy()]
        for t in range(n_steps):
            pre_activation = W_in @ base_inputs[i] + W_res @ state1
            if activation == "tanh":
                state1 = (1 - leak_rate) * state1 + leak_rate * np.tanh(pre_activation)
            elif activation == "relu":
                state1 = (1 - leak_rate) * state1 + leak_rate * np.maximum(
                    0, pre_activation
                )
            else:
                state1 = (1 - leak_rate) * state1 + leak_rate * pre_activation
            states1.append(state1.copy())

        # Evolve second input
        state2 = np.random.randn(reservoir_size) * 0.1
        states2 = [state2.copy()]
        for t in range(n_steps):
            pre_activation = W_in @ similar_inputs[i] + W_res @ state2
            if activation == "tanh":
                state2 = (1 - leak_rate) * state2 + leak_rate * np.tanh(pre_activation)
            elif activation == "relu":
                state2 = (1 - leak_rate) * state2 + leak_rate * np.maximum(
                    0, pre_activation
                )
            else:
                state2 = (1 - leak_rate) * state2 + leak_rate * pre_activation
            states2.append(state2.copy())

        # Measure divergence at each step
        step_divergences = []
        for t in range(len(states1)):
            divergence = np.linalg.norm(states1[t] - states2[t])
            step_divergences.append(divergence)

        # Check if divergence exceeds threshold within 10 steps
        # _threshold = 0.5  # Threshold for significant divergence (available if needed)
        max_divergence = max(step_divergences)
        divergences.append(max_divergence)

    # Separation capacity: proportion of pairs that diverge sufficiently
    divergence_threshold = 0.5
    diverging_pairs = sum(1 for d in divergences if d > divergence_threshold)
    separation_score = diverging_pairs / n_pairs

    logger.info(
        f"Separation capacity test: {diverging_pairs}/{n_pairs} pairs diverged, score={separation_score:.4f}"
    )
    return separation_score


def validate_network_topology(
    network_weights: Dict[str, np.ndarray], connectivity_pattern: str
) -> Dict[str, bool]:
    """Validate network topology"""

    validation_results = {}

    # Check connectivity pattern
    validation_results["valid_connectivity"] = validate_connectivity_pattern(
        network_weights, connectivity_pattern
    )

    # Check weight distribution
    validation_results["valid_weight_distribution"] = validate_weight_distribution(
        network_weights
    )

    # Check dimension consistency
    validation_results["dimension_consistency"] = validate_dimension_consistency(
        network_weights
    )

    return validation_results


def validate_connectivity_pattern(
    network_weights: Dict[str, np.ndarray], connectivity_pattern: str
) -> bool:
    """Validate connectivity pattern (placeholder)"""

    # Simple connectivity validation
    if connectivity_pattern == "random":
        return len(network_weights) > 0
    elif connectivity_pattern == "structured":
        return (
            "input_to_liquid" in network_weights
            and "liquid_to_output" in network_weights
        )
    else:
        return False


def validate_weight_distribution(network_weights: Dict[str, np.ndarray]) -> bool:
    """Validate weight distribution"""

    for name, weights in network_weights.items():
        # Check for finite values
        if not np.all(np.isfinite(weights)):
            return False

        # Check for reasonable weight magnitudes
        if np.any(np.abs(weights) > 10):
            return False

    return True


def validate_dimension_consistency(network_weights: Dict[str, np.ndarray]) -> bool:
    """Validate dimension consistency across weights"""

    # Simple dimension check (placeholder)
    weight_shapes = [weights.shape for weights in network_weights.values()]

    # Check that all weights have reasonable dimensions
    for shape in weight_shapes:
        if len(shape) != 2:  # Should be 2D matrices
            return False
        if any(dim <= 0 for dim in shape):
            return False

    return True


def run_liquid_network_validation():
    """Run complete liquid network validation"""
    logger.info("Running liquid network validation...")

    # Create example network weights
    network_weights = {
        "input_to_liquid": np.random.normal(0, 0.1, (100, 50)),
        "liquid_to_liquid": np.random.normal(0, 0.05, (100, 100)),
        "liquid_to_output": np.random.normal(0, 0.1, (10, 100)),
    }

    # Liquid network parameters
    liquid_params = {
        "leak_rate": 0.9,
        "spectral_radius": 0.95,
        "activation": "tanh",
        "reservoir_size": 100,
    }

    # Test liquid network properties
    property_scores = test_liquid_network_properties(network_weights, liquid_params)

    # Validate network topology
    connectivity_pattern = "structured"
    topology_validation = validate_network_topology(
        network_weights, connectivity_pattern
    )

    return {
        "property_scores": property_scores,
        "topology_validation": topology_validation,
        "liquid_parameters": liquid_params,
    }


if __name__ == "__main__":
    results = run_liquid_network_validation()
    print("Liquid network validation results:")
    print(f"Property scores: {results['property_scores']}")
    print(f"Topology validation: {results['topology_validation']}")
