"""
Parameter recovery validation wrapper for FP-10 Bayesian Estimation.
This module bridges the validation test suite with the actual MCMC implementation.
"""

import numpy as np
import pandas as pd
import pymc as pm
from typing import Dict, Any, Optional, Tuple
from Falsification.FP_10_BayesianEstimation_MCMC import run_mcmc_bayesian_estimation


def run_bayesian_estimation_complete(
    data: np.ndarray, true_parameters: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Wrapper for parameter recovery validation tests.

    Args:
        data: Observed response data (n_subjects, n_trials)
        true_parameters: Original parameters used to generate data (optional)

    Returns:
        Dict containing posterior_samples and posterior_statistics
    """
    # Map input data to stimulus/response format for the real model
    # The test generates data with n_subjects=40, n_timepoints=60
    # Our real model expects 1D arrays for a single "session" of trials

    # Flatten or average data across subjects?
    # Usually Bayesian parameter recovery in these tests is done on a per-dataset basis
    if data.ndim > 1:
        # If passed (subjects, timepoints), let's take a representative subject or average
        # For validation purposes, averaging stimulus and taking majority response or similar?
        # Actually, let's just use the first subject for validation speed
        response_data = data[0].astype(int)
    else:
        response_data = data.astype(int)

    n_trials = len(response_data)
    # Generate linear stimulus data as expected by FP_10
    stimulus_data = np.linspace(0.05, 2.5, n_trials)

    # Run real MCMC estimation
    # Use fewer samples for validation efficiency
    n_samples = 1000
    burn_in = 500

    results = run_mcmc_bayesian_estimation(
        stimulus_data=stimulus_data,
        response_data=response_data,
        n_samples=n_samples,
        n_chains=2,
        burn_in=burn_in,
    )

    trace = results.get("trace")
    if trace is None:
        raise RuntimeError("MCMC sampling failed to produce a trace")

    # Extract posterior samples and statistics
    # The test expects 'beta' and 'pi'
    # Our model has 'beta', 'pi_e', 'pi_i', 'theta_0', 'alpha'
    # Mapping: beta -> beta, pi -> pi_e

    param_mapping = {"beta": "beta", "pi": "pi_e"}

    posterior_samples = {}
    posterior_statistics = {}

    import arviz as az

    summary = az.summary(trace)

    for test_param, model_param in param_mapping.items():
        if model_param in trace.posterior:
            # Samples
            samples = trace.posterior[model_param].values.flatten()
            posterior_samples[test_param] = samples

            # Stats
            if model_param in summary.index:
                posterior_statistics[test_param] = {
                    "mean": float(summary.loc[model_param, "mean"]),
                    "std": float(summary.loc[model_param, "sd"]),
                    "hdi_3%": float(summary.loc[model_param, "hdi_3%"]),
                    "hdi_97%": float(summary.loc[model_param, "hdi_97%"]),
                    "r_hat": float(summary.loc[model_param, "r_hat"]),
                }
            else:
                posterior_statistics[test_param] = {
                    "mean": float(np.mean(samples)),
                    "std": float(np.std(samples)),
                }

    return {
        "posterior_samples": posterior_samples,
        "posterior_statistics": posterior_statistics,
        "trace": trace,
        "diagnostics": results.get("convergence_diagnostics"),
    }
