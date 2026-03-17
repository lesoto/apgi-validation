"""
Falsification Protocol 11: Bayesian Estimation
=============================================

This protocol implements Bayesian estimation for APGI model parameters.
Per Step 1.8 - Upgrade FP-11 from MH to NUTS with PyMC.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Any
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import pymc as pm
    import arviz as az

    HAS_PYMC = True
    HAS_ARVIZ = True
except ImportError:
    HAS_PYMC = False
    HAS_ARVIZ = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "PyMC not installed - Bayesian estimation will be limited to simple MH algorithm"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_bayesian_estimation_nuts(
    data: np.ndarray,
    prior_params: Dict[str, Tuple[float, float]],
    n_samples: int = 2000,
    tune_samples: int = 1000,
    n_chains: int = 4,
) -> Dict[str, Any]:
    """
    Run Bayesian estimation using NUTS (No-U-Turn Sampler) via PyMC.
    Per Step 1.8 - Replace MH with NUTS via pm.sample(init="nuts").
    """
    if not HAS_PYMC:
        logger.warning("PyMC not available - falling back to simple MH algorithm")
        return run_bayesian_estimation_mh(data, prior_params, n_iterations=n_samples)

    try:
        # Define PyMC model
        with pm.Model():
            # Priors for APGI parameters
            theta_0 = pm.Normal(
                "theta_0",
                mu=prior_params["theta_0"][0],
                sigma=prior_params["theta_0"][1],
            )
            alpha = pm.Normal(
                "alpha", mu=prior_params["alpha"][0], sigma=prior_params["alpha"][1]
            )
            beta = pm.Normal(
                "beta", mu=prior_params["beta"][0], sigma=prior_params["beta"][1]
            )

            # Likelihood: data ~ N(μ, σ²) where μ depends on APGI parameters
            # Simplified APGI generative model
            mu = theta_0 * alpha * beta
            sigma = pm.HalfNormal("sigma", sigma=1.0)

            # Likelihood
            pm.Normal("likelihood", mu=mu, sigma=sigma, observed=data)

            # Sample using NUTS
            trace = pm.sample(
                tune=tune_samples,
                draws=n_samples,
                chains=n_chains,
                init="nuts",
                return_inferencedata=False,
                progressbar=True,
            )

        # Extract posterior samples
        posterior_samples = {}
        for param in ["theta_0", "alpha", "beta"]:
            posterior_samples[param] = trace.posterior[param].values.flatten()

        # Calculate convergence diagnostics
        # R-hat should be < 1.1 for convergence
        r_hat = az.rhat(trace)
        mean_r_hat = np.mean([r_hat[param] for param in r_hat.keys()])

        # Effective sample size (ESS)
        ess = az.ess(trace)
        mean_ess = np.mean([ess[param] for param in ess.keys()])

        # Check convergence criteria
        convergence_pass = mean_r_hat < 1.1 and mean_ess > 400

        return {
            "posterior_samples": posterior_samples,
            "trace": trace,
            "r_hat": {k: float(v) for k, v in r_hat.items()},
            "ess": {k: float(v) for k, v in ess.items()},
            "mean_r_hat": float(mean_r_hat),
            "mean_ess": float(mean_ess),
            "convergence_pass": convergence_pass,
            "n_samples": n_samples,
            "tune_samples": tune_samples,
            "n_chains": n_chains,
        }

    except Exception as e:
        logger.error(f"Error in NUTS sampling: {e}")
        logger.warning("Falling back to simple MH algorithm")
        return run_bayesian_estimation_mh(data, prior_params, n_iterations=n_samples)


def run_bayesian_estimation_mh(
    data: np.ndarray,
    prior_params: Dict[str, Tuple[float, float]],
    n_iterations: int = 1000,
) -> Dict[str, Any]:
    """
    Run Bayesian estimation using simple Metropolis-Hastings sampling.
    Fallback when PyMC is not available.
    """

    # Initialize posterior samples
    posterior_samples = {}

    for param_name, (mean, std) in prior_params.items():
        # Simple MCMC sampling
        samples = metropolis_hastings_sampling(
            data, param_name, mean, std, n_iterations
        )
        posterior_samples[param_name] = samples

    return {
        "posterior_samples": posterior_samples,
        "convergence_pass": False,
        "mean_r_hat": 0.0,
        "mean_ess": 0.0,
        "n_samples": n_iterations,
    }


def metropolis_hastings_sampling(
    data: np.ndarray,
    param_name: str,
    initial_value: float,
    proposal_std: float,
    n_iterations: int,
) -> np.ndarray:
    """Simple Metropolis-Hastings sampling with proper acceptance ratio"""

    samples = np.zeros(n_iterations)
    current_value = initial_value

    # Improved likelihood function for APGI generative model
    def likelihood(value):
        # APGI generative model likelihood
        # Model: data ~ N(μ, σ²) where μ depends on APGI parameters
        # For now, use a more robust likelihood that avoids numerical issues
        if data.std() < 1e-10:
            # Handle edge case of constant data
            return 1.0 if abs(value - data.mean()) < 1e-6 else 1e-10

        # Gaussian likelihood with proper normalization
        diff = value - data.mean()
        log_likelihood = -0.5 * (diff / data.std()) ** 2 - np.log(
            data.std() * np.sqrt(2 * np.pi)
        )
        return np.exp(log_likelihood)

    for i in range(n_iterations):
        # Propose new value
        proposal = current_value + np.random.normal(0, proposal_std)

        # Calculate acceptance ratio (proper MH algorithm)
        likelihood_current = likelihood(current_value)
        likelihood_proposal = likelihood(proposal)

        # Avoid division by zero or numerical issues
        if likelihood_current < 1e-300:
            acceptance_ratio = (
                1.0  # Always accept if current likelihood is essentially zero
            )
        else:
            acceptance_ratio = likelihood_proposal / likelihood_current

        # Accept or reject
        if np.random.random() < min(acceptance_ratio, 1.0):
            current_value = proposal

        samples[i] = current_value

    return samples


def compute_posterior_distributions(
    posterior_samples: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """Compute posterior distribution statistics"""

    posterior_stats = {}

    for param_name, samples in posterior_samples.items():
        stats = {
            "mean": np.mean(samples),
            "std": np.std(samples),
            "median": np.median(samples),
            "q_025": np.percentile(samples, 2.5),
            "q_975": np.percentile(samples, 97.5),
            "effective_sample_size": len(samples),  # Placeholder
        }
        posterior_stats[param_name] = stats

    return posterior_stats


def calculate_bayesian_factor(
    posterior_samples_1: Dict[str, np.ndarray],
    posterior_samples_2: Dict[str, np.ndarray],
) -> float:
    """Calculate Bayesian factor between two models"""

    # Simple evidence calculation (placeholder)
    evidence_1 = 0.0
    evidence_2 = 0.0

    for param_name in posterior_samples_1.keys():
        if param_name in posterior_samples_2:
            # Approximate evidence using sample means
            samples_1 = posterior_samples_1[param_name]
            samples_2 = posterior_samples_2[param_name]

            # Simple evidence approximation
            evidence_1 += np.log(np.mean(np.exp(-0.5 * samples_1**2)))
            evidence_2 += np.log(np.mean(np.exp(-0.5 * samples_2**2)))

    bayes_factor = evidence_1 - evidence_2
    return bayes_factor


def run_bayesian_estimation_complete():
    """
    Run complete Bayesian estimation analysis.
    Per Step 1.8 - Upgrade FP-11 from MH to NUTS with PyMC.
    """
    logger.info("Running Bayesian estimation with NUTS...")

    # Generate synthetic data
    n_data_points = 100
    true_params = {"theta_0": 0.5, "alpha": 5.0, "beta": 1.2}
    data = np.random.normal(0, 1, n_data_points)

    # Prior distributions (mean, std)
    prior_params = {
        "theta_0": (0.5, 0.2),
        "alpha": (5.0, 2.0),
        "beta": (1.2, 0.5),
    }

    # Run Bayesian estimation with NUTS
    nuts_results = run_bayesian_estimation_nuts(
        data, prior_params, n_samples=2000, tune_samples=1000, n_chains=4
    )

    # Compute posterior statistics
    posterior_stats = compute_posterior_distributions(nuts_results["posterior_samples"])

    # Compare with alternative model (different priors)
    alternative_priors = {
        "theta_0": (0.3, 0.1),
        "alpha": (3.0, 1.0),
        "beta": (1.0, 0.3),
    }

    nuts_results_alt = run_bayesian_estimation_nuts(
        data, alternative_priors, n_samples=2000, tune_samples=1000, n_chains=4
    )
    bayes_factor = calculate_bayesian_factor(
        nuts_results["posterior_samples"],
        nuts_results_alt["posterior_samples"],
    )

    return {
        "posterior_samples": nuts_results["posterior_samples"],
        "posterior_statistics": posterior_stats,
        "bayes_factor": bayes_factor,
        "true_parameters": true_params,
        "convergence_diagnostics": {
            "r_hat": nuts_results["r_hat"],
            "ess": nuts_results["ess"],
            "mean_r_hat": nuts_results["mean_r_hat"],
            "mean_ess": nuts_results["mean_ess"],
            "convergence_pass": nuts_results["convergence_pass"],
        },
        "n_samples": nuts_results["n_samples"],
        "tune_samples": nuts_results["tune_samples"],
        "n_chains": nuts_results["n_chains"],
    }


def get_falsification_criteria() -> Dict[str, Dict[str, Any]]:
    """
    Return complete falsification specifications for Falsification-Protocol-11.

    Tests: Bayesian estimation quality, posterior calibration, model comparison

    Returns:
        Dictionary of falsification criteria with thresholds, tests, and effect sizes
    """
    return {
        "F11.1": {
            "description": "Posterior Convergence",
            "threshold": "ESS > 400 for all parameters, R-hat < 1.1 for convergence",
            "test": "NUTS convergence diagnostics, α=0.05",
            "effect_size": "R-hat < 1.1 indicates good mixing",
            "alternative": "Falsified if ESS < 200 OR R-hat > 1.2",
        },
        "F11.2": {
            "description": "Posterior Calibration",
            "threshold": "95% credible intervals contain true parameter value ≥ 90% of the time",
            "test": "Coverage probability test, α=0.05",
            "effect_size": "Coverage probability ≥ 0.90",
            "alternative": "Falsified if coverage < 0.80 OR bias > 0.20",
        },
        "F11.3": {
            "description": "Model Comparison",
            "threshold": "Bayes factor > 10 in favor of APGI model over null model",
            "test": "Bayes factor calculation, α=0.01",
            "effect_size": "Strong evidence (BF > 10)",
            "alternative": "Falsified if BF < 3 (weak evidence)",
        },
    }


def check_falsification(
    posterior_samples: Dict[str, np.ndarray],
    true_params: Dict[str, float],
    bayes_factor: float,
    convergence_diagnostics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Implement all statistical tests for Falsification-Protocol-11.

    Args:
        posterior_samples: Dictionary of posterior samples for each parameter
        true_params: Dictionary of true parameter values
        bayes_factor: Bayes factor comparing APGI model to null model
        convergence_diagnostics: Dictionary containing R-hat and ESS values

    Returns:
        Dictionary with pass/fail results, effect sizes, and test statistics
    """
    results = {
        "protocol": "Falsification-Protocol-11",
        "criteria": {},
        "summary": {"passed": 0, "failed": 0, "total": 3},
    }

    # F11.1: Posterior Convergence
    logger.info("Testing F11.1: Posterior Convergence")
    r_hat = convergence_diagnostics.get("r_hat", {})
    ess = convergence_diagnostics.get("ess", {})

    # Check ESS > 400 for all parameters
    ess_values = list(ess.values())
    min_ess = min(ess_values) if ess_values else 0
    f11_1_pass = min_ess > 400

    # Check R-hat < 1.1 for convergence
    r_hat_values = list(r_hat.values())
    max_r_hat = max(r_hat_values) if r_hat_values else 0
    f11_1_pass = f11_1_pass and max_r_hat < 1.1

    results["criteria"]["F11.1"] = {
        "passed": f11_1_pass,
        "min_ess": min_ess,
        "max_r_hat": max_r_hat,
        "threshold": "ESS > 400, R-hat < 1.1",
        "actual": f"Min ESS: {min_ess:.0f}, Max R-hat: {max_r_hat:.3f}",
    }
    if f11_1_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F11.1: {'PASS' if f11_1_pass else 'FAIL'} - Min ESS: {min_ess:.0f}, Max R-hat: {max_r_hat:.3f}"
    )

    # F11.2: Posterior Calibration
    logger.info("Testing F11.2: Posterior Calibration")
    coverage_count = 0
    for param_name, samples in posterior_samples.items():
        if param_name in true_params:
            q_025 = np.percentile(samples, 2.5)
            q_975 = np.percentile(samples, 97.5)
            if q_025 <= true_params[param_name] <= q_975:
                coverage_count += 1

    coverage_prob = coverage_count / len(true_params) if true_params else 0
    f11_2_pass = coverage_prob >= 0.90

    results["criteria"]["F11.2"] = {
        "passed": f11_2_pass,
        "coverage_probability": coverage_prob,
        "threshold": "Coverage ≥ 0.90",
        "actual": f"Coverage: {coverage_prob:.2f}",
    }
    if f11_2_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F11.2: {'PASS' if f11_2_pass else 'FAIL'} - Coverage: {coverage_prob:.2f}"
    )

    # F11.3: Model Comparison
    logger.info("Testing F11.3: Model Comparison")
    f11_3_pass = bayes_factor > 10.0

    results["criteria"]["F11.3"] = {
        "passed": f11_3_pass,
        "bayes_factor": bayes_factor,
        "threshold": "BF > 10",
        "actual": f"BF: {bayes_factor:.2f}",
    }
    if f11_3_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(f"F11.3: {'PASS' if f11_3_pass else 'FAIL'} - BF: {bayes_factor:.2f}")

    return results


if __name__ == "__main__":
    results = run_bayesian_estimation_complete()

    # Run falsification checks
    falsification_results = check_falsification(
        posterior_samples=results["posterior_samples"],
        true_params=results["true_parameters"],
        bayes_factor=results["bayes_factor"],
        convergence_diagnostics=results["convergence_diagnostics"],
    )

    print("Bayesian estimation results:")
    print(f"Posterior statistics: {results['posterior_statistics']}")
    print(f"Bayes factor: {results['bayes_factor']:.4f}")
    print(f"Convergence diagnostics: {results['convergence_diagnostics']}")
    print("\nFalsification results:")
    print(
        f"Passed: {falsification_results['summary']['passed']}/{falsification_results['summary']['total']}"
    )
    print(
        f"Failed: {falsification_results['summary']['failed']}/{falsification_results['summary']['total']}"
    )
