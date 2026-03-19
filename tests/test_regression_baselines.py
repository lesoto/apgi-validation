"""
Regression test baselines with golden outputs for comparison.
These tests compare current outputs against stored golden outputs to detect regressions.
============================================================================================
"""

import pytest
import numpy as np
import json
from pathlib import Path
import sys
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))


# Directory for storing golden outputs
GOLDEN_OUTPUTS_DIR = Path(__file__).parent.parent / "tests" / "golden_outputs"
GOLDEN_OUTPUTS_DIR.mkdir(exist_ok=True)


def save_golden_output(test_name: str, output: Dict[str, Any]) -> None:
    """
    Save golden output for a test.

    Args:
        test_name: Name of the test
        output: Output data to save
    """
    output_file = GOLDEN_OUTPUTS_DIR / f"{test_name}.json"

    # Convert numpy arrays to lists for JSON serialization
    serializable_output = {}
    for key, value in output.items():
        if isinstance(value, np.ndarray):
            serializable_output[key] = {
                "type": "ndarray",
                "data": value.tolist(),
                "shape": value.shape,
                "dtype": str(value.dtype),
            }
        elif isinstance(value, (np.integer, np.floating)):
            serializable_output[key] = float(value)
        else:
            serializable_output[key] = value

    with open(output_file, "w") as f:
        json.dump(serializable_output, f, indent=2)


def load_golden_output(test_name: str) -> Dict[str, Any]:
    """
    Load golden output for a test.

    Args:
        test_name: Name of the test

    Returns:
        Golden output data
    """
    output_file = GOLDEN_OUTPUTS_DIR / f"{test_name}.json"

    if not output_file.exists():
        raise FileNotFoundError(f"Golden output not found: {output_file}")

    with open(output_file, "r") as f:
        loaded_data = json.load(f)

    # Convert back to numpy arrays
    output = {}
    for key, value in loaded_data.items():
        if isinstance(value, dict) and value.get("type") == "ndarray":
            output[key] = np.array(value["data"], dtype=value["dtype"])
        else:
            output[key] = value

    return output


def compare_outputs(
    current: Dict[str, Any],
    golden: Dict[str, Any],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compare current output with golden output.

    Args:
        current: Current test output
        golden: Golden output to compare against
        rtol: Relative tolerance for floating point comparison
        atol: Absolute tolerance for floating point comparison

    Returns:
        Comparison results
    """
    results = {"passed": True, "differences": [], "summary": {}}

    for key in golden.keys():
        if key not in current:
            results["passed"] = False
            results["differences"].append(f"Missing key: {key}")
            continue

        current_value = current[key]
        golden_value = golden[key]

        if isinstance(golden_value, np.ndarray):
            if not isinstance(current_value, np.ndarray):
                results["passed"] = False
                results["differences"].append(
                    f"Type mismatch for {key}: expected ndarray, got {type(current_value)}"
                )
            elif current_value.shape != golden_value.shape:
                results["passed"] = False
                results["differences"].append(
                    f"Shape mismatch for {key}: expected {golden_value.shape}, got {current_value.shape}"
                )
            elif not np.allclose(current_value, golden_value, rtol=rtol, atol=atol):
                results["passed"] = False
                max_diff = np.max(np.abs(current_value - golden_value))
                results["differences"].append(
                    f"Value mismatch for {key}: max diff = {max_diff}"
                )
        elif isinstance(golden_value, (float, np.floating)):
            if not np.isclose(current_value, golden_value, rtol=rtol, atol=atol):
                results["passed"] = False
                diff = abs(current_value - golden_value)
                results["differences"].append(
                    f"Value mismatch for {key}: {current_value} vs {golden_value} (diff={diff})"
                )
        else:
            if current_value != golden_value:
                results["passed"] = False
                results["differences"].append(
                    f"Value mismatch for {key}: {current_value} vs {golden_value}"
                )

    results["summary"] = {
        "total_keys": len(golden),
        "matched_keys": len(golden) - len(results["differences"]),
        "failed_keys": len(results["differences"]),
    }

    return results


@pytest.mark.regression
def test_entropy_calculation_regression():
    """Test entropy calculation against golden output."""
    try:
        from APGI_Equations import calculate_entropy

        # Generate reproducible test data
        np.random.seed(42)
        test_data = np.random.randn(100)

        # Calculate current output
        current_output = {"entropy": calculate_entropy(test_data)}

        # Load or create golden output
        try:
            golden_output = load_golden_output("entropy_calculation")
        except FileNotFoundError:
            # Create golden output on first run
            save_golden_output("entropy_calculation", current_output)
            pytest.skip("Golden output created for future runs")

        # Compare outputs
        comparison = compare_outputs(current_output, golden_output)

        if not comparison["passed"]:
            pytest.fail(
                f"Entropy calculation regression detected:\n"
                f"{chr(10).join(comparison['differences'])}"
            )

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.regression
def test_mutual_information_regression():
    """Test mutual information calculation against golden output."""
    try:
        from APGI_Equations import calculate_mutual_information

        # Generate reproducible test data
        np.random.seed(42)
        data1 = np.random.randn(100)
        data2 = np.random.randn(100)

        # Calculate current output
        current_output = {
            "mutual_information": calculate_mutual_information(data1, data2)
        }

        # Load or create golden output
        try:
            golden_output = load_golden_output("mutual_information")
        except FileNotFoundError:
            save_golden_output("mutual_information", current_output)
            pytest.skip("Golden output created for future runs")

        # Compare outputs
        comparison = compare_outputs(current_output, golden_output)

        if not comparison["passed"]:
            pytest.fail(
                f"Mutual information regression detected:\n"
                f"{chr(10).join(comparison['differences'])}"
            )

    except ImportError:
        pytest.skip("APGI_Equations module not available")


@pytest.mark.regression
def test_falsification_framework_regression():
    """Test falsification framework against golden output."""
    try:
        from APGI_Falsification_Framework import FalsificationFramework

        framework = FalsificationFramework()

        # Generate reproducible test data
        np.random.seed(42)
        test_data = {
            "P1": {
                "advantage_metric": np.random.randn(50),
                "comparison_metric": np.random.randn(50),
                "effect_size": np.random.randn(50),
            },
        }

        # Run falsification
        current_output = framework.run_comprehensive_falsification(test_data)

        # Extract key metrics for comparison
        comparison_output = {
            "theory_status": current_output.get("theory_status"),
            "overall_falsification": current_output.get("overall_falsification"),
        }

        # Load or create golden output
        try:
            golden_output = load_golden_output("falsification_framework")
        except FileNotFoundError:
            save_golden_output("falsification_framework", comparison_output)
            pytest.skip("Golden output created for future runs")

        # Compare outputs
        comparison = compare_outputs(comparison_output, golden_output)

        if not comparison["passed"]:
            pytest.fail(
                f"Falsification framework regression detected:\n"
                f"{chr(10).join(comparison['differences'])}"
            )

    except ImportError:
        pytest.skip("FalsificationFramework not available")


@pytest.mark.regression
def test_parameter_recovery_regression():
    """Test parameter recovery against golden output."""
    try:
        from Falsification.Falsification_BayesianEstimation_ParameterRecovery import (
            run_bayesian_estimation_complete,
        )

        # Generate reproducible test data
        np.random.seed(42)
        synthetic_data = {
            "observations": np.random.randn(50, 30),
            "true_parameters": {"beta": 0.7, "pi": 0.5},
        }

        # Run parameter recovery
        results = run_bayesian_estimation_complete(
            data=synthetic_data["observations"],
            true_parameters=synthetic_data["true_parameters"],
        )

        # Extract key metrics for comparison
        current_output = {
            "posterior_mean_beta": results["posterior_statistics"]["beta"]["mean"],
            "posterior_mean_pi": results["posterior_statistics"]["pi"]["mean"],
            "bayes_factor": results["bayes_factor"],
        }

        # Load or create golden output
        try:
            golden_output = load_golden_output("parameter_recovery")
        except FileNotFoundError:
            save_golden_output("parameter_recovery", current_output)
            pytest.skip("Golden output created for future runs")

        # Compare outputs (allow more tolerance for stochastic algorithms)
        comparison = compare_outputs(current_output, golden_output, rtol=0.1, atol=0.05)

        if not comparison["passed"]:
            pytest.fail(
                f"Parameter recovery regression detected:\n"
                f"{chr(10).join(comparison['differences'])}"
            )

    except ImportError:
        pytest.skip("BayesianEstimation module not available")


@pytest.mark.regression
def test_active_inference_simulation_regression():
    """Test active inference simulation against golden output."""
    try:
        from Falsification.Falsification_ActiveInferenceAgents_F1F2 import (
            ActiveInferenceAgents,
        )

        agents = ActiveInferenceAgents()

        # Generate reproducible test data
        np.random.seed(42)
        params = {"n_agents": 5, "n_steps": 50, "learning_rate": 0.01}

        # Run simulation
        results = agents.run_simulation(params)

        # Extract key metrics for comparison
        current_output = {
            "n_agents": params["n_agents"],
            "n_steps": params["n_steps"],
        }

        # Add simulation results if available
        if "agent_trajectories" in results:
            current_output["n_trajectories"] = len(results["agent_trajectories"])

        # Load or create golden output
        try:
            golden_output = load_golden_output("active_inference_simulation")
        except FileNotFoundError:
            save_golden_output("active_inference_simulation", current_output)
            pytest.skip("Golden output created for future runs")

        # Compare outputs
        comparison = compare_outputs(current_output, golden_output)

        if not comparison["passed"]:
            pytest.fail(
                f"Active inference simulation regression detected:\n"
                f"{chr(10).join(comparison['differences'])}"
            )

    except ImportError:
        pytest.skip("ActiveInferenceAgents module not available")


@pytest.mark.regression
def test_cross_species_scaling_regression():
    """Test cross-species scaling against golden output."""
    try:
        from Falsification.Falsification_CrossSpeciesScaling_P12 import (
            CrossSpeciesScaling,
        )

        scaling = CrossSpeciesScaling()

        # Generate reproducible test data
        np.random.seed(42)
        species_a_params = {"brain_size": 1000, "metabolic_rate": 0.5}

        # Scale parameters
        species_b_params = scaling.scale_parameters(
            species_a_params, from_species="A", to_species="B"
        )

        # Extract key metrics for comparison
        current_output = {
            "brain_size_scaled": species_b_params.get("brain_size"),
            "metabolic_rate_scaled": species_b_params.get("metabolic_rate"),
        }

        # Load or create golden output
        try:
            golden_output = load_golden_output("cross_species_scaling")
        except FileNotFoundError:
            save_golden_output("cross_species_scaling", current_output)
            pytest.skip("Golden output created for future runs")

        # Compare outputs
        comparison = compare_outputs(current_output, golden_output)

        if not comparison["passed"]:
            pytest.fail(
                f"Cross-species scaling regression detected:\n"
                f"{chr(10).join(comparison['differences'])}"
            )

    except ImportError:
        pytest.skip("CrossSpeciesScaling module not available")


@pytest.mark.regression
def test_bayesian_model_comparison_regression():
    """Test Bayesian model comparison against golden output."""
    try:
        from Falsification.Falsification_BayesianEstimation_ParameterRecovery import (
            run_bayesian_estimation_complete,
        )

        # Generate reproducible test data
        np.random.seed(42)
        synthetic_data = {
            "observations": np.random.randn(30, 20),
            "true_parameters": {"beta": 0.7, "pi": 0.5},
        }

        # Run estimation
        results = run_bayesian_estimation_complete(
            data=synthetic_data["observations"],
            true_parameters=synthetic_data["true_parameters"],
        )

        # Extract key metrics for comparison
        current_output = {
            "bayes_factor": results["bayes_factor"],
            "model_evidence": results.get("model_evidence", 0.0),
        }

        # Load or create golden output
        try:
            golden_output = load_golden_output("bayesian_model_comparison")
        except FileNotFoundError:
            save_golden_output("bayesian_model_comparison", current_output)
            pytest.skip("Golden output created for future runs")

        # Compare outputs (allow more tolerance for stochastic algorithms)
        comparison = compare_outputs(current_output, golden_output, rtol=0.15, atol=0.1)

        if not comparison["passed"]:
            pytest.fail(
                f"Bayesian model comparison regression detected:\n"
                f"{chr(10).join(comparison['differences'])}"
            )

    except ImportError:
        pytest.skip("BayesianEstimation module not available")


@pytest.mark.regression
def test_convergence_diagnostics_regression():
    """Test convergence diagnostics against golden output."""
    try:
        from Falsification.Falsification_BayesianEstimation_ParameterRecovery import (
            run_bayesian_estimation_complete,
        )

        # Generate reproducible test data
        np.random.seed(42)
        synthetic_data = {
            "observations": np.random.randn(40, 25),
            "true_parameters": {"beta": 0.7, "pi": 0.5},
        }

        # Run estimation
        results = run_bayesian_estimation_complete(
            data=synthetic_data["observations"],
            true_parameters=synthetic_data["true_parameters"],
        )

        # Extract convergence diagnostics
        convergence = results.get("convergence_diagnostics", {})
        current_output = {
            "r_hat_beta": convergence.get("r_hat", {}).get("beta", 1.0),
            "r_hat_pi": convergence.get("r_hat", {}).get("pi", 1.0),
            "ess_beta": convergence.get("ess", {}).get("beta", 0),
            "ess_pi": convergence.get("ess", {}).get("pi", 0),
        }

        # Load or create golden output
        try:
            golden_output = load_golden_output("convergence_diagnostics")
        except FileNotFoundError:
            save_golden_output("convergence_diagnostics", current_output)
            pytest.skip("Golden output created for future runs")

        # Compare outputs (allow more tolerance for MCMC)
        comparison = compare_outputs(current_output, golden_output, rtol=0.2, atol=0.1)

        if not comparison["passed"]:
            pytest.fail(
                f"Convergence diagnostics regression detected:\n"
                f"{chr(10).join(comparison['differences'])}"
            )

    except ImportError:
        pytest.skip("BayesianEstimation module not available")
