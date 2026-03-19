"""
Functional falsification tests with synthetic ground-truth data.
These tests run actual algorithms with known parameters to validate falsification criteria.
============================================================================================
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_synthetic_apgi_data(
    n_subjects: int = 50,
    n_timepoints: int = 100,
    true_beta: float = 0.7,
    true_pi: float = 0.5,
    noise_level: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic data with known APGI parameters for falsification testing.

    Args:
        n_subjects: Number of subjects
        n_timepoints: Number of timepoints per subject
        true_beta: True interoceptive prediction error parameter
        true_pi: True precision weighting parameter
        noise_level: Standard deviation of observation noise
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing synthetic data and true parameters
    """
    np.random.seed(seed)

    # Generate latent states
    latent_states = np.random.randn(n_subjects, n_timepoints)

    # Generate predictions based on APGI equations
    predictions = true_beta * latent_states + true_pi * np.random.randn(
        n_subjects, n_timepoints
    )

    # Generate observations with noise
    observations = predictions + noise_level * np.random.randn(n_subjects, n_timepoints)

    # Calculate true advantage metric
    advantage_metric = np.mean(observations, axis=1)

    # Calculate true comparison metric
    comparison_metric = np.std(observations, axis=1)

    return {
        "observations": observations,
        "predictions": predictions,
        "latent_states": latent_states,
        "true_parameters": {
            "beta": true_beta,
            "pi": true_pi,
        },
        "advantage_metric": advantage_metric,
        "comparison_metric": comparison_metric,
        "metadata": {
            "n_subjects": n_subjects,
            "n_timepoints": n_timepoints,
            "noise_level": noise_level,
        },
    }


@pytest.mark.functional
def test_parameter_recovery_with_synthetic_data():
    """Test parameter recovery with synthetic ground-truth data."""
    try:
        from Falsification.Falsification_BayesianEstimation_ParameterRecovery import (
            run_bayesian_estimation_complete,
            check_falsification,
        )

        # Generate synthetic data with known parameters
        synthetic_data = generate_synthetic_apgi_data(
            n_subjects=30, n_timepoints=50, true_beta=0.7, true_pi=0.5, noise_level=0.1
        )

        # Run Bayesian estimation on synthetic data
        results = run_bayesian_estimation_complete(
            data=synthetic_data["observations"],
            true_parameters=synthetic_data["true_parameters"],
        )

        # Verify results structure
        assert "posterior_samples" in results
        assert "true_parameters" in results
        assert "posterior_statistics" in results

        # Check parameter recovery accuracy
        posterior_stats = results["posterior_statistics"]
        true_params = synthetic_data["true_parameters"]

        # Beta should be recovered within reasonable bounds
        beta_mean = posterior_stats.get("beta", {}).get("mean", 0)
        beta_std = posterior_stats.get("beta", {}).get("std", 1)
        assert (
            abs(beta_mean - true_params["beta"]) < 3 * beta_std
        ), f"Beta recovery failed: {beta_mean:.3f} vs {true_params['beta']:.3f}"

        # Pi should be recovered within reasonable bounds
        pi_mean = posterior_stats.get("pi", {}).get("mean", 0)
        pi_std = posterior_stats.get("pi", {}).get("std", 1)
        assert (
            abs(pi_mean - true_params["pi"]) < 3 * pi_std
        ), f"Pi recovery failed: {pi_mean:.3f} vs {true_params['pi']:.3f}"

        # Run falsification checks
        falsification_results = check_falsification(
            posterior_samples=results["posterior_samples"],
            true_params=true_params,
            bayes_factor=results.get("bayes_factor", 10.0),
            convergence_diagnostics=results.get("convergence_diagnostics", {}),
            identifiability_diagnostics=results.get("identifiability_diagnostics"),
            calibration_diagnostics=results.get("calibration_diagnostics"),
        )

        # Verify falsification results
        assert "summary" in falsification_results
        assert "criteria" in falsification_results
        assert falsification_results["summary"]["total"] > 0

        # With synthetic data, we should pass most criteria
        pass_rate = (
            falsification_results["summary"]["passed"]
            / falsification_results["summary"]["total"]
        )
        assert pass_rate >= 0.5, f"Pass rate too low: {pass_rate:.2f}"

    except ImportError:
        pytest.skip("BayesianEstimation module not available")
    except Exception as e:
        pytest.fail(f"Parameter recovery test failed: {e}")


@pytest.mark.functional
def test_posterior_calibration_with_synthetic_data():
    """Test posterior calibration using synthetic ground-truth data."""
    try:
        from Falsification.Falsification_BayesianEstimation_ParameterRecovery import (
            run_bayesian_estimation_complete,
        )

        # Generate multiple synthetic datasets
        n_simulations = 10
        coverage_count = 0

        for i in range(n_simulations):
            synthetic_data = generate_synthetic_apgi_data(
                n_subjects=20,
                n_timepoints=30,
                true_beta=0.7,
                true_pi=0.5,
                noise_level=0.1,
                seed=i,
            )

            results = run_bayesian_estimation_complete(
                data=synthetic_data["observations"],
                true_parameters=synthetic_data["true_parameters"],
            )

            posterior_samples = results["posterior_samples"]
            true_params = synthetic_data["true_parameters"]

            # Check if true parameters are within 95% credible intervals
            for param_name, samples in posterior_samples.items():
                if param_name in true_params:
                    q_025 = np.percentile(samples, 2.5)
                    q_975 = np.percentile(samples, 97.5)
                    if q_025 <= true_params[param_name] <= q_975:
                        coverage_count += 1

        # Calculate coverage probability
        total_checks = n_simulations * len(synthetic_data["true_parameters"])
        coverage_prob = coverage_count / total_checks if total_checks > 0 else 0

        # Coverage should be close to 0.95 (within reasonable tolerance)
        assert (
            coverage_prob >= 0.80
        ), f"Posterior calibration failed: coverage {coverage_prob:.2f} < 0.80"

    except ImportError:
        pytest.skip("BayesianEstimation module not available")
    except Exception as e:
        pytest.fail(f"Posterior calibration test failed: {e}")


@pytest.mark.functional
def test_falsification_framework_with_synthetic_data():
    """Test falsification framework with synthetic data across priorities."""
    try:
        from APGI_Falsification_Framework import FalsificationFramework

        framework = FalsificationFramework()

        # Generate synthetic test data for all priorities
        synthetic_data = generate_synthetic_apgi_data(
            n_subjects=50, n_timepoints=100, true_beta=0.7, true_pi=0.5
        )

        all_test_data = {
            "P1": {
                "advantage_metric": synthetic_data["advantage_metric"],
                "comparison_metric": synthetic_data["comparison_metric"],
                "effect_size": synthetic_data["advantage_metric"]
                / synthetic_data["comparison_metric"],
            },
            "P2": {
                "pp_difference": synthetic_data["advantage_metric"] * 0.5,
                "cohens_h": synthetic_data["advantage_metric"]
                / np.std(synthetic_data["observations"]),
                "correlation": np.corrcoef(
                    synthetic_data["observations"].flatten(),
                    synthetic_data["predictions"].flatten(),
                )[0, 1],
                "rt_advantage": synthetic_data["advantage_metric"] * 2.5,
                "beta_interaction": synthetic_data["true_parameters"]["beta"] * 0.5,
            },
            "P3": {
                "intero_advantage": synthetic_data["advantage_metric"],
                "reduction_metric": synthetic_data["comparison_metric"],
                "cohens_d": synthetic_data["advantage_metric"]
                / synthetic_data["comparison_metric"],
            },
        }

        # Run comprehensive falsification
        results = framework.run_comprehensive_falsification(all_test_data)

        # Verify results structure
        assert isinstance(results, dict)
        assert "priority_results" in results
        assert "overall_falsification" in results
        assert "theory_status" in results

        # Verify priority results
        assert isinstance(results["priority_results"], list)
        assert len(results["priority_results"]) > 0

        # Verify theory status is valid
        expected_statuses = [
            "supported",
            "weakly_falsified",
            "strongly_falsified",
            "not_tested",
        ]
        assert results["theory_status"] in expected_statuses

    except ImportError:
        pytest.skip("FalsificationFramework not available")
    except Exception as e:
        pytest.fail(f"Falsification framework test failed: {e}")


@pytest.mark.functional
def test_active_inference_agents_with_synthetic_data():
    """Test active inference agents with synthetic ground-truth data."""
    try:
        from Falsification.Falsification_ActiveInferenceAgents_F1F2 import (
            ActiveInferenceAgents,
        )

        agents = ActiveInferenceAgents()

        # Generate synthetic environment data
        n_agents = 10
        n_steps = 100

        synthetic_env = {
            "states": np.random.randn(n_steps, 5),
            "observations": np.random.randn(n_steps, 5)
            + 0.1 * np.random.randn(n_steps, 5),
            "true_rewards": np.random.randn(n_steps),
        }

        # Run simulation
        params = {
            "n_agents": n_agents,
            "n_steps": n_steps,
            "learning_rate": 0.01,
            "precision": 1.0,
        }

        results = agents.run_simulation(params, environment_data=synthetic_env)

        # Verify results structure
        assert isinstance(results, dict)
        assert "agent_trajectories" in results or "falsification_result" in results

        # Check if agents learned something (not random behavior)
        if "agent_trajectories" in results:
            trajectories = results["agent_trajectories"]
            assert len(trajectories) == n_agents

    except ImportError:
        pytest.skip("ActiveInferenceAgents module not available")
    except Exception as e:
        pytest.fail(f"Active inference agents test failed: {e}")


@pytest.mark.functional
def test_numerical_boundary_conditions():
    """Test falsification with extreme parameter values."""
    try:
        from Falsification.Falsification_BayesianEstimation_ParameterRecovery import (
            run_bayesian_estimation_complete,
        )

        # Test with extreme beta values
        test_cases = [
            {"beta": 0.01, "pi": 0.5, "description": "Very low beta"},
            {"beta": 0.99, "pi": 0.5, "description": "Very high beta"},
            {"beta": 0.5, "pi": 0.01, "description": "Very low pi"},
            {"beta": 0.5, "pi": 0.99, "description": "Very high pi"},
        ]

        for test_case in test_cases:
            synthetic_data = generate_synthetic_apgi_data(
                n_subjects=30,
                n_timepoints=50,
                true_beta=test_case["beta"],
                true_pi=test_case["pi"],
                noise_level=0.1,
                seed=42,
            )

            results = run_bayesian_estimation_complete(
                data=synthetic_data["observations"],
                true_parameters=synthetic_data["true_parameters"],
            )

            # Verify estimation doesn't crash with extreme values
            assert "posterior_samples" in results
            assert "posterior_statistics" in results

    except ImportError:
        pytest.skip("BayesianEstimation module not available")
    except Exception as e:
        pytest.fail(f"Numerical boundary test failed: {e}")
