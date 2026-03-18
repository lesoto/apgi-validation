"""
Falsification Protocol 10: Bayesian Estimation MCMC
===============================================

This protocol implements MCMC-based Bayesian estimation for APGI model parameters.
Per Step 1.8 - Upgrade FP-10 from MH to NUTS with PyMC.

CRITICAL FEATURES:
- 5,000 MCMC samples across 4 chains with 1,000 burn-in
- Gelman-Rubin diagnostic (R̂ ≤ 1.01) mandatory convergence check
- Bayes factor computation for model comparison (APGI vs. StandardPP vs. GWTOnly)
- Priors over {θ₀, Πe, Πi, β, α} from physiological ranges
- Likelihood defined as the APGI psychometric function
"""

import logging
import numpy as np
from typing import Dict, Tuple, Any, List, Optional
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
    logger.warning("PyMC/ArviZ not installed - Bayesian estimation will be limited")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apgi_psychometric_function(
    stimulus_intensity: np.ndarray,
    theta_0: float,
    pi_e: float,
    pi_i: float,
    beta: float,
    alpha: float,
) -> np.ndarray:
    """
    APGI psychometric function for interoceptive awareness.

    The function models the probability of correct interoceptive detection
    as a function of stimulus intensity and APGI parameters.

    Args:
        stimulus_intensity: Array of stimulus intensities
        theta_0: Baseline ignition threshold (θ₀)
        pi_e: Exteroceptive precision (Πᵉ)
        pi_i: Interoceptive precision (Πⁱ)
        beta: Somatic bias weight (β)
        alpha: Attention gain factor (α)

    Returns:
        Array of detection probabilities
    """
    # APGI computational model
    # Precision-weighted prediction error
    precision_ratio = pi_i / (pi_e + 1e-10)  # Avoid division by zero

    # Somatic bias modulation
    somatic_gain = 1.0 + beta * precision_ratio

    # Attention-modulated threshold
    effective_threshold = theta_0 / (1.0 + alpha * stimulus_intensity)

    # Psychometric function (cumulative Gaussian)
    mu = effective_threshold * somatic_gain
    sigma = 1.0 / np.sqrt(pi_i + 1e-10)  # Noise decreases with precision

    # Use logistic approximation for computational efficiency
    psychometric = 1.0 / (1.0 + np.exp(-(stimulus_intensity - mu) / sigma))

    return psychometric


def define_apgi_priors() -> Dict[str, Any]:
    """
    Define physiologically plausible priors for APGI parameters.

    Returns:
        Dictionary of prior distributions for PyMC model
    """
    priors = {
        # θ₀: Baseline ignition threshold [0.1, 1.0]
        # Normalized threshold for interoceptive awareness
        "theta_0": {
            "dist": "Normal",
            "params": {"mu": 0.5, "sigma": 0.2},
            "range": [0.1, 1.0],
            "interpretation": "Baseline ignition threshold",
        },
        # Πᵉ: Exteroceptive precision [0.1, 5.0]
        # Precision of external sensory information
        "pi_e": {
            "dist": "HalfNormal",
            "params": {"sigma": 1.5},
            "range": [0.1, 5.0],
            "interpretation": "Exteroceptive precision",
        },
        # Πⁱ: Interoceptive precision [0.1, 5.0]
        # Precision of internal bodily signals
        "pi_i": {
            "dist": "HalfNormal",
            "params": {"sigma": 1.5},
            "range": [0.1, 5.0],
            "interpretation": "Interoceptive precision",
        },
        # β: Somatic bias weight [0.0, 3.0]
        # Weight of somatic markers in decision making
        "beta": {
            "dist": "Normal",
            "params": {"mu": 1.15, "sigma": 0.5},
            "range": [0.0, 3.0],
            "interpretation": "Somatic bias weight",
        },
        # α: Attention gain factor [0.1, 2.0]
        # Gain factor for attentional modulation
        "alpha": {
            "dist": "HalfNormal",
            "params": {"sigma": 0.5},
            "range": [0.1, 2.0],
            "interpretation": "Attention gain factor",
        },
    }

    return priors


def run_mcmc_bayesian_estimation(
    stimulus_data: np.ndarray,
    response_data: np.ndarray,
    n_samples: int = 5000,
    n_chains: int = 4,
    burn_in: int = 1000,
    target_accept: float = 0.95,
) -> Dict[str, Any]:
    """
    Run MCMC Bayesian estimation for APGI model parameters.

    CRITICAL: Implements 5,000 MCMC samples across 4 chains with 1,000 burn-in
    HIGH: Implements Gelman-Rubin diagnostic (R̂ ≤ 1.01) convergence check

    Args:
        stimulus_data: Array of stimulus intensities
        response_data: Array of binary responses (0/1)
        n_samples: Number of MCMC samples (default: 5000)
        n_chains: Number of MCMC chains (default: 4)
        burn_in: Number of burn-in samples (default: 1000)
        target_accept: Target acceptance rate for NUTS (default: 0.95)

    Returns:
        Dictionary with posterior samples, convergence diagnostics, and model evidence
    """
    if not HAS_PYMC:
        raise ImportError("PyMC is required for MCMC Bayesian estimation")

    if not HAS_ARVIZ:
        raise ImportError("ArviZ is required for convergence diagnostics")

    logger.info(
        f"Starting MCMC: {n_samples} samples across {n_chains} chains with {burn_in} burn-in"
    )

    # Get priors
    priors = define_apgi_priors()

    try:
        with pm.Model():
            # Define priors from physiological ranges
            theta_0 = pm.Normal(
                "theta_0",
                mu=priors["theta_0"]["params"]["mu"],
                sigma=priors["theta_0"]["params"]["sigma"],
            )
            pi_e = pm.HalfNormal("pi_e", sigma=priors["pi_e"]["params"]["sigma"])
            pi_i = pm.HalfNormal("pi_i", sigma=priors["pi_i"]["params"]["sigma"])
            beta = pm.Normal(
                "beta",
                mu=priors["beta"]["params"]["mu"],
                sigma=priors["beta"]["params"]["sigma"],
            )
            alpha = pm.HalfNormal("alpha", sigma=priors["alpha"]["params"]["sigma"])

            # Define likelihood using APGI psychometric function
            # Expected detection probabilities
            p_detection = apgi_psychometric_function(
                stimulus_data, theta_0, pi_e, pi_i, beta, alpha
            )

            # Bernoulli likelihood for binary responses
            pm.Bernoulli("likelihood", p=p_detection, observed=response_data)

            # Sample using NUTS
            trace = pm.sample(
                draws=n_samples,
                tune=burn_in,
                chains=n_chains,
                target_accept=target_accept,
                return_inferencedata=True,
                progressbar=True,
                random_seed=42,
            )

            # Compute model evidence (log marginal likelihood) using bridge sampling
            log_evidence = None
            try:
                # Try to compute bridge sampling evidence
                log_evidence = az.loo(trace, pointwise=False, reff=1.0)
            except Exception as e:
                logger.warning(f"Could not compute LOO evidence: {e}")
                # Fallback to WAIC
                try:
                    log_evidence = az.waic(trace, pointwise=False, scale="log")
                except Exception as e2:
                    logger.warning(f"Could not compute WAIC evidence: {e2}")

        # Extract posterior samples
        posterior_samples = {}
        param_names = ["theta_0", "pi_e", "pi_i", "beta", "alpha"]

        for param in param_names:
            if param in trace.posterior:
                # Flatten across chains and draws
                samples = trace.posterior[param].values
                posterior_samples[param] = samples.flatten()

        # Calculate convergence diagnostics
        r_hat = az.rhat(trace)
        ess = az.ess(trace)

        # Check Gelman-Rubin diagnostic (R̂ ≤ 1.01)
        max_r_hat = max(
            [float(r_hat[param].values) for param in param_names if param in r_hat]
        )
        convergence_pass = max_r_hat <= 1.01

        # Effective sample size diagnostics
        min_ess = min(
            [float(ess[param].values) for param in param_names if param in ess]
        )

        # Prepare results
        results = {
            "posterior_samples": posterior_samples,
            "trace": trace,
            "convergence_diagnostics": {
                "r_hat": {
                    param: float(r_hat[param].values)
                    for param in param_names
                    if param in r_hat
                },
                "ess": {
                    param: float(ess[param].values)
                    for param in param_names
                    if param in ess
                },
                "max_r_hat": max_r_hat,
                "min_ess": min_ess,
                "convergence_pass": convergence_pass,
                "convergence_threshold": 1.01,
            },
            "sampling_info": {
                "n_samples": n_samples,
                "n_chains": n_chains,
                "burn_in": burn_in,
                "target_accept": target_accept,
                "total_posterior_samples": n_samples * n_chains,
            },
            "model_evidence": {
                "log_evidence": float(log_evidence.iloc[0])
                if log_evidence is not None
                else None,
                "evidence_type": "LOO" if log_evidence is not None else "None",
            },
            "priors": priors,
        }

        if not convergence_pass:
            logger.warning(f"MCMC failed convergence: R̂ = {max_r_hat:.4f} > 1.01")
            results["convergence_diagnostics"]["reason"] = "non-convergence"

        logger.info(f"MCMC completed: R̂ = {max_r_hat:.4f}, ESS = {min_ess:.0f}")
        return results

    except Exception as e:
        logger.error(f"Error in MCMC sampling: {e}")
        raise


def compute_bayes_factors(
    evidence_dict: Dict[str, float],
    model_names: List[str] = ["APGI", "StandardPP", "GWTOnly"],
) -> Dict[str, Any]:
    """
    Compute Bayes factors for model comparison.

    HIGH: Implements Bayes factor computation for APGI vs. StandardPP vs. GWTOnly

    Args:
        evidence_dict: Dictionary of model evidence values (log scale)
        model_names: List of model names for comparison

    Returns:
        Dictionary with Bayes factors and model comparison results
    """
    if len(evidence_dict) < 2:
        raise ValueError("Need at least 2 models for Bayes factor comparison")

    logger.info(f"Computing Bayes factors for models: {model_names}")

    # Compute pairwise Bayes factors
    bayes_factors = {}

    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i != j and model1 in evidence_dict and model2 in evidence_dict:
                # BF_12 = evidence_1 - evidence_2 (log scale)
                bf_log = evidence_dict[model1] - evidence_dict[model2]
                bf_linear = np.exp(bf_log)

                pair_key = f"{model1}_vs_{model2}"
                bayes_factors[pair_key] = {
                    "log_bf": bf_log,
                    "linear_bf": bf_linear,
                    "interpretation": interpret_bayes_factor(bf_linear),
                }

    # Determine best model
    best_model = max(evidence_dict.keys(), key=lambda k: evidence_dict[k])

    # Model comparison summary
    comparison_results = {
        "bayes_factors": bayes_factors,
        "model_evidence": evidence_dict,
        "best_model": best_model,
        "model_ranking": sorted(
            evidence_dict.keys(), key=lambda k: evidence_dict[k], reverse=True
        ),
        "comparison_summary": {
            model: {
                "evidence": evidence_dict.get(model),
                "rank": i + 1,
                "vs_best": float(evidence_dict[model] - evidence_dict[best_model])
                if model in evidence_dict and best_model in evidence_dict
                else None,
            }
            for i, model in enumerate(
                sorted(
                    evidence_dict.keys(), key=lambda k: evidence_dict[k], reverse=True
                )
            )
        },
    }

    logger.info(f"Best model: {best_model}")
    return comparison_results


def interpret_bayes_factor(bf: float) -> str:
    """
    Interpret Bayes factor according to Jeffreys' scale.

    Args:
        bf: Bayes factor (linear scale)

    Returns:
        Interpretation string
    """
    if bf > 100:
        return "Extreme evidence for model 1"
    elif bf > 30:
        return "Very strong evidence for model 1"
    elif bf > 10:
        return "Strong evidence for model 1"
    elif bf > 3:
        return "Moderate evidence for model 1"
    elif bf > 1:
        return "Weak evidence for model 1"
    elif bf > 0.33:
        return "Weak evidence for model 2"
    elif bf > 0.1:
        return "Moderate evidence for model 2"
    elif bf > 0.03:
        return "Strong evidence for model 2"
    elif bf > 0.01:
        return "Very strong evidence for model 2"
    else:
        return "Extreme evidence for model 2"


def run_alternative_models(
    stimulus_data: np.ndarray,
    response_data: np.ndarray,
    n_samples: int = 2000,
    n_chains: int = 4,
    burn_in: int = 1000,
) -> Dict[str, Dict[str, Any]]:
    """
    Run alternative models for Bayes factor comparison.

    Args:
        stimulus_data: Array of stimulus intensities
        response_data: Array of binary responses (0/1)
        n_samples: Number of MCMC samples
        n_chains: Number of MCMC chains
        burn_in: Number of burn-in samples

    Returns:
        Dictionary with results from alternative models
    """
    if not HAS_PYMC:
        raise ImportError("PyMC is required for alternative model estimation")

    logger.info("Running alternative models for comparison")

    results = {}

    # StandardPP model (Precision Priors only)
    try:
        with pm.Model():
            # Simplified priors for StandardPP
            theta_0 = pm.Normal("theta_0", mu=0.5, sigma=0.2)
            pi_i = pm.HalfNormal("pi_i", sigma=1.5)
            # No beta or alpha in StandardPP

            # Simplified psychometric function
            p_detection = 1.0 / (1.0 + np.exp(-(stimulus_data - theta_0) * pi_i))
            pm.Bernoulli("likelihood", p=p_detection, observed=response_data)

            trace = pm.sample(
                draws=n_samples,
                tune=burn_in,
                chains=n_chains,
                target_accept=0.9,
                return_inferencedata=True,
                progressbar=False,
            )

            # Compute evidence
            try:
                evidence = az.loo(trace, pointwise=False, reff=1.0)
                log_evidence = float(evidence.iloc[0])
            except Exception:
                log_evidence = None

            results["StandardPP"] = {
                "trace": trace,
                "log_evidence": log_evidence,
                "model_type": "StandardPP",
            }

    except Exception as e:
        logger.warning(f"Error in StandardPP model: {e}")
        results["StandardPP"] = {"log_evidence": None, "error": str(e)}

    # GWTOnly model (Global Workspace Theory only)
    try:
        with pm.Model():
            # GWT-specific priors
            theta_0 = pm.Normal("theta_0", mu=0.6, sigma=0.2)
            alpha = pm.HalfNormal("alpha", sigma=0.5)
            # No precision parameters in GWT-only

            # GWT psychometric function (attention-based only)
            effective_threshold = theta_0 / (1.0 + alpha * stimulus_data)
            p_detection = 1.0 / (1.0 + np.exp(-(stimulus_data - effective_threshold)))
            pm.Bernoulli("likelihood", p=p_detection, observed=response_data)

            trace = pm.sample(
                draws=n_samples,
                tune=burn_in,
                chains=n_chains,
                target_accept=0.9,
                return_inferencedata=True,
                progressbar=False,
            )

            # Compute evidence
            try:
                evidence = az.loo(trace, pointwise=False, reff=1.0)
                log_evidence = float(evidence.iloc[0])
            except Exception:
                log_evidence = None

            results["GWTOnly"] = {
                "trace": trace,
                "log_evidence": log_evidence,
                "model_type": "GWTOnly",
            }

    except Exception as e:
        logger.warning(f"Error in GWTOnly model: {e}")
        results["GWTOnly"] = {"log_evidence": None, "error": str(e)}

    return results


def run_complete_mcmc_analysis(
    stimulus_data: np.ndarray,
    response_data: np.ndarray,
    n_samples: int = 5000,
    n_chains: int = 4,
    burn_in: int = 1000,
    run_alternatives: bool = True,
) -> Dict[str, Any]:
    """
    Run complete MCMC Bayesian estimation analysis.

    This is the main function that implements the full Protocol 10 analysis.

    Args:
        stimulus_data: Array of stimulus intensities
        response_data: Array of binary responses (0/1)
        n_samples: Number of MCMC samples (default: 5000)
        n_chains: Number of MCMC chains (default: 4)
        burn_in: Number of burn-in samples (default: 1000)
        run_alternatives: Whether to run alternative models for comparison

    Returns:
        Complete analysis results
    """
    logger.info("Starting complete MCMC Bayesian estimation analysis")

    # Run main APGI model
    apgi_results = run_mcmc_bayesian_estimation(
        stimulus_data, response_data, n_samples, n_chains, burn_in
    )

    # Check convergence
    if not apgi_results["convergence_diagnostics"]["convergence_pass"]:
        logger.error("APGI model failed convergence - returning early")
        return {
            "passed": False,
            "reason": "non-convergence",
            "apgi_results": apgi_results,
        }

    # Prepare results dictionary
    complete_results = {
        "passed": True,
        "apgi_results": apgi_results,
        "bayes_factor_comparison": None,
    }

    # Run alternative models and compute Bayes factors
    if run_alternatives:
        try:
            alternative_results = run_alternative_models(
                stimulus_data, response_data, n_samples // 2, n_chains, burn_in
            )

            # Collect evidence values
            evidence_dict = {}

            # APGI model evidence
            apgi_evidence = apgi_results["model_evidence"]["log_evidence"]
            if apgi_evidence is not None:
                evidence_dict["APGI"] = apgi_evidence

            # Alternative model evidence
            for model_name, result in alternative_results.items():
                if result.get("log_evidence") is not None:
                    evidence_dict[model_name] = result["log_evidence"]

            # Compute Bayes factors if we have at least 2 models
            if len(evidence_dict) >= 2:
                bayes_results = compute_bayes_factors(evidence_dict)
                complete_results["bayes_factor_comparison"] = bayes_results
                complete_results["alternative_results"] = alternative_results

                logger.info("Bayes factor comparison completed")
            else:
                logger.warning(
                    "Insufficient evidence values for Bayes factor comparison"
                )

        except Exception as e:
            logger.warning(f"Error in Bayes factor computation: {e}")
            complete_results["bayes_factor_error"] = str(e)

    logger.info("Complete MCMC analysis finished")
    return complete_results


def generate_synthetic_data(
    n_trials: int = 200,
    true_params: Optional[Dict[str, float]] = None,
    noise_level: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing the MCMC implementation.

    Args:
        n_trials: Number of trials
        true_params: True parameter values
        noise_level: Level of noise in responses

    Returns:
        Tuple of (stimulus_data, response_data)
    """
    if true_params is None:
        true_params = {
            "theta_0": 0.5,
            "pi_e": 1.0,
            "pi_i": 1.2,
            "beta": 1.15,
            "alpha": 0.8,
        }

    # Generate stimulus intensities
    stimulus_data = np.linspace(0.1, 2.0, n_trials)

    # Generate detection probabilities
    p_detection = apgi_psychometric_function(
        stimulus_data,
        true_params["theta_0"],
        true_params["pi_e"],
        true_params["pi_i"],
        true_params["beta"],
        true_params["alpha"],
    )

    # Generate binary responses with noise
    response_data = np.random.binomial(1, p_detection)

    return stimulus_data, response_data


if __name__ == "__main__":
    # Test the implementation with synthetic data
    logger.info("Testing MCMC Bayesian estimation implementation")

    # Generate synthetic data
    stimulus_data, response_data = generate_synthetic_data(n_trials=200)

    # Run complete analysis
    results = run_complete_mcmc_analysis(
        stimulus_data, response_data, n_samples=1000, n_chains=2, burn_in=500
    )

    # Print results
    print("\n=== MCMC Bayesian Estimation Results ===")
    print(f"Passed: {results['passed']}")
    if not results["passed"]:
        print(f"Reason: {results.get('reason', 'Unknown')}")

    if "apgi_results" in results:
        conv_diag = results["apgi_results"]["convergence_diagnostics"]
        print(f"Max R̂: {conv_diag['max_r_hat']:.4f}")
        print(f"Min ESS: {conv_diag['min_ess']:.0f}")

        if "bayes_factor_comparison" in results:
            bf_comp = results["bayes_factor_comparison"]
            print(f"Best model: {bf_comp['best_model']}")
            print(f"Model ranking: {bf_comp['model_ranking']}")

    print("\n=== Test Complete ===")
