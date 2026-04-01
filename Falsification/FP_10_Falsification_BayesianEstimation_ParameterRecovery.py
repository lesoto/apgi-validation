"""
DEPRECATED: Parameter recovery validation wrapper for FP-10 Bayesian Estimation.

This wrapper file is DEPRECATED and will be removed in a future version.
All FP-10 calls should route through the canonical MCMC file:
    Falsification.FP_10_BayesianEstimation_MCMC

Migration:
    OLD: from Falsification.FP_10_Falsification_BayesianEstimation_ParameterRecovery import run_bayesian_estimation_complete
    NEW: from Falsification.FP_10_BayesianEstimation_MCMC import run_bayesian_estimation_complete

This module now simply re-exports from the canonical file for backward compatibility.
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "FP_10_Falsification_BayesianEstimation_ParameterRecovery is DEPRECATED. "
    "Use Falsification.FP_10_BayesianEstimation_MCMC instead. "
    "This wrapper will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and re-export from canonical MCMC file
from Falsification.FP_10_BayesianEstimation_MCMC import (
    run_bayesian_estimation_complete,
    run_mcmc_bayesian_estimation,
    run_complete_mcmc_analysis,
    generate_synthetic_data,
    BayesianParameterRecovery,
    FP10Dispatcher,
)

__all__ = [
    "run_bayesian_estimation_complete",
    "run_mcmc_bayesian_estimation",
    "run_complete_mcmc_analysis",
    "generate_synthetic_data",
    "BayesianParameterRecovery",
    "FP10Dispatcher",
]
