"""
Falsification Protocol 10b: Parameter Recovery Validation
======================================================

Independent implementation of parameter recovery as a distinct sub-protocol.
This exercises synthetic data generation + MCMC + parameter comparison to earn
its place as a distinct file from FP_10_BayesianEstimation_MCMC.py.

Implementation Strategy:
------------------------
1. Generate synthetic data with known ground truth parameters
2. Run MCMC Bayesian estimation to recover parameters
3. Compare recovered vs. true parameters with statistical tests
4. Validate parameter identifiability and recovery accuracy
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

# Import canonical MCMC functionality
from Falsification.FP_10_BayesianEstimation_MCMC import (
    generate_synthetic_data,
    run_mcmc_bayesian_estimation,
    BayesianParameterRecovery,
)

# Removed for GUI stability
logger = logging.getLogger(__name__)


@dataclass
class ParameterRecoveryResult:
    """Results for parameter recovery validation"""

    parameter_name: str
    true_value: float
    recovered_mean: float
    recovered_std: float
    credible_interval: Tuple[float, float]
    recovery_error: float
    relative_error: float
    recovered_successfully: bool
    identifiability_score: float


class FP10bParameterRecovery:
    """
    Independent parameter recovery validation protocol.

    This class implements a genuine parameter recovery sub-protocol that:
    1. Generates synthetic data with known parameters
    2. Applies MCMC Bayesian estimation
    3. Validates parameter recovery accuracy
    4. Tests parameter identifiability
    """

    def __init__(self):
        self.recovery_results: List[ParameterRecoveryResult] = []
        self.recovery_summary: Dict[str, Any] = {}

        # Parameter recovery tolerances per V5.1 criteria
        self.recovery_tolerance = 1e-6
        self.relative_error_threshold = 0.10  # 10% relative error threshold
        self.identifiability_threshold = 0.80  # 80% identifiability score threshold

    def run_parameter_recovery_validation(
        self,
        n_synthetic_datasets: int = 50,
        mcmc_samples: int = 10000,
        burn_in: int = 2000,
        true_params: Optional[Dict[str, float]] = None,
        noise_level: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run complete parameter recovery validation.

        Args:
            n_synthetic_datasets: Number of synthetic datasets to generate
            mcmc_samples: Number of MCMC samples per dataset
            burn_in: MCMC burn-in period
            true_params: Ground truth parameters (uses defaults if None)
            noise_level: Observation noise level for synthetic data

        Returns:
            Dictionary with recovery validation results
        """
        logger.info(
            f"Starting FP-10b parameter recovery validation with {n_synthetic_datasets} datasets"
        )

        # Use default true parameters if not provided
        if true_params is None:
            true_params = self._get_default_true_parameters()

        # Run parameter recovery for each synthetic dataset
        for dataset_idx in range(n_synthetic_datasets):
            try:
                dataset_result = self._recover_parameters_from_dataset(
                    true_params=true_params,
                    noise_level=noise_level,
                    mcmc_samples=mcmc_samples,
                    burn_in=burn_in,
                    dataset_idx=dataset_idx,
                )
                self.recovery_results.extend(dataset_result)

            except Exception as e:
                logger.error(f"Dataset {dataset_idx} failed: {e}")
                continue

        # Compute recovery summary statistics
        self.recovery_summary = self._compute_recovery_summary()

        logger.info("FP-10b parameter recovery validation completed")
        return self.recovery_summary

    def _get_default_true_parameters(self) -> Dict[str, float]:
        """Get default ground truth parameters for synthetic data generation"""
        return {
            "tau_S": 0.5,  # Surprise integration time constant
            "tau_theta": 2.0,  # Threshold adaptation time constant
            "theta_0": 1.0,  # Baseline threshold
            "alpha": 2.0,  # Ignition sharpness
            "beta": 1.2,  # Somatic bias weight
            "gamma_M": 0.3,  # Somatic marker influence
            "sigma_noise": 0.15,  # Observation noise
            "eta_theta": 0.1,  # Threshold adaptation rate
        }

    def _recover_parameters_from_dataset(
        self,
        true_params: Dict[str, float],
        noise_level: float,
        mcmc_samples: int,
        burn_in: int,
        dataset_idx: int,
    ) -> List[ParameterRecoveryResult]:
        """
        Recover parameters from a single synthetic dataset.

        Args:
            true_params: Ground truth parameters
            noise_level: Observation noise level
            mcmc_samples: MCMC sample count
            burn_in: MCMC burn-in period
            dataset_idx: Dataset identifier

        Returns:
            List of parameter recovery results for this dataset
        """
        # Generate synthetic data with known parameters
        synthetic_data = generate_synthetic_data(
            n_trials=100,
            true_params=true_params,
            noise_level=noise_level,
            random_seed=42 + dataset_idx,  # Reproducible but varied
        )

        # Run MCMC Bayesian estimation
        mcmc_results = run_mcmc_bayesian_estimation(
            data=synthetic_data,
            n_samples=mcmc_samples,
            burn_in=burn_in,
            random_seed=42 + dataset_idx,
        )

        # Compute recovery results for each parameter
        dataset_results = []
        for param_name, true_value in true_params.items():
            if param_name in mcmc_results["posterior_means"]:
                recovered_mean = mcmc_results["posterior_means"][param_name]
                recovered_std = mcmc_results["posterior_stds"][param_name]
                ci_lower = mcmc_results["credible_intervals"][param_name][0]
                ci_upper = mcmc_results["credible_intervals"][param_name][1]

                # Compute recovery metrics
                recovery_error = abs(recovered_mean - true_value)
                relative_error = (
                    recovery_error / abs(true_value)
                    if true_value != 0
                    else recovery_error
                )

                # Check if true value is within credible interval
                recovered_successfully = ci_lower <= true_value <= ci_upper

                # Compute identifiability score based on posterior concentration
                identifiability_score = self._compute_identifiability_score(
                    recovered_mean, recovered_std, true_value
                )

                result = ParameterRecoveryResult(
                    parameter_name=param_name,
                    true_value=true_value,
                    recovered_mean=recovered_mean,
                    recovered_std=recovered_std,
                    credible_interval=(ci_lower, ci_upper),
                    recovery_error=recovery_error,
                    relative_error=relative_error,
                    recovered_successfully=recovered_successfully,
                    identifiability_score=identifiability_score,
                )

                dataset_results.append(result)

        return dataset_results

    def _compute_identifiability_score(
        self, recovered_mean: float, recovered_std: float, true_value: float
    ) -> float:
        """
        Compute parameter identifiability score.

        Higher scores indicate better identifiability:
        - Low posterior uncertainty (small std)
        - Low recovery error
        - True value near posterior mean

        Args:
            recovered_mean: Posterior mean
            recovered_std: Posterior standard deviation
            true_value: Ground truth value

        Returns:
            Identifiability score (0-1)
        """
        # Normalize uncertainty (lower std = higher score)
        uncertainty_score = 1.0 / (1.0 + recovered_std)

        # Normalize accuracy (lower error = higher score)
        accuracy_score = 1.0 / (1.0 + abs(recovered_mean - true_value))

        # Combined identifiability score
        identifiability_score = 0.6 * accuracy_score + 0.4 * uncertainty_score

        return np.clip(identifiability_score, 0.0, 1.0)

    def _compute_recovery_summary(self) -> Dict[str, Any]:
        """Compute summary statistics across all recovery results"""
        if not self.recovery_results:
            return {"error": "No recovery results available"}

        # Group results by parameter
        param_results = {}
        for result in self.recovery_results:
            param_name = result.parameter_name
            if param_name not in param_results:
                param_results[param_name] = []
            param_results[param_name].append(result)

        # Compute summary statistics for each parameter
        summary = {
            "parameter_recovery_summary": {},
            "overall_recovery_metrics": {},
            "validation_passed": True,
            "recommendations": [],
        }

        all_relative_errors = []
        all_identifiability_scores = []
        successful_recoveries = 0
        total_recoveries = len(self.recovery_results)

        for param_name, results in param_results.items():
            relative_errors = [r.relative_error for r in results]
            identifiability_scores = [r.identifiability_score for r in results]
            success_rate = sum(1 for r in results if r.recovered_successfully) / len(
                results
            )

            param_summary = {
                "mean_relative_error": float(np.mean(relative_errors)),
                "std_relative_error": float(np.std(relative_errors)),
                "mean_identifiability_score": float(np.mean(identifiability_scores)),
                "success_rate": float(success_rate),
                "n_datasets": len(results),
                "passed_thresholds": (
                    np.mean(relative_errors) <= self.relative_error_threshold
                    and np.mean(identifiability_scores)
                    >= self.identifiability_threshold
                ),
            }

            summary["parameter_recovery_summary"][param_name] = param_summary

            all_relative_errors.extend(relative_errors)
            all_identifiability_scores.extend(identifiability_scores)
            successful_recoveries += sum(1 for r in results if r.recovered_successfully)

        # Overall recovery metrics
        summary["overall_recovery_metrics"] = {
            "mean_relative_error_all_params": float(np.mean(all_relative_errors)),
            "mean_identifiability_score_all_params": float(
                np.mean(all_identifiability_scores)
            ),
            "overall_success_rate": float(successful_recoveries / total_recoveries),
            "total_parameters_tested": total_recoveries,
            "parameters_passing_threshold": sum(
                1
                for ps in summary["parameter_recovery_summary"].values()
                if ps["passed_thresholds"]
            ),
        }

        # Determine if validation passed
        mean_error = summary["overall_recovery_metrics"][
            "mean_relative_error_all_params"
        ]
        mean_identifiability = summary["overall_recovery_metrics"][
            "mean_identifiability_score_all_params"
        ]

        summary["validation_passed"] = (
            mean_error <= self.relative_error_threshold
            and mean_identifiability >= self.identifiability_threshold
        )

        # Generate recommendations
        if not summary["validation_passed"]:
            if mean_error > self.relative_error_threshold:
                summary["recommendations"].append(
                    f"High recovery error ({mean_error:.3f} > {self.relative_error_threshold:.3f}). "
                    "Consider increasing data length or reducing observation noise."
                )
            if mean_identifiability < self.identifiability_threshold:
                summary["recommendations"].append(
                    f"Low identifiability ({mean_identifiability:.3f} < {self.identifiability_threshold:.3f}). "
                    "Parameters may be poorly identified; check model identifiability."
                )
        else:
            summary["recommendations"].append(
                "Parameter recovery validation passed successfully."
            )

        return summary


# Re-export canonical functions for backward compatibility
def run_bayesian_estimation_complete(*args, **kwargs):
    """Backward compatibility wrapper - delegates to canonical MCMC implementation"""
    warnings.warn(
        "run_bayesian_estimation_complete is deprecated. Use run_parameter_recovery_validation "
        "for parameter recovery or run_mcmc_bayesian_estimation for standard MCMC.",
        DeprecationWarning,
        stacklevel=2,
    )
    from Falsification.FP_10_BayesianEstimation_MCMC import (
        run_bayesian_estimation_complete,
    )

    return run_bayesian_estimation_complete(*args, **kwargs)


def run_mcmc_bayesian_estimation(*args, **kwargs):
    """Backward compatibility wrapper - delegates to canonical MCMC implementation"""
    from Falsification.FP_10_BayesianEstimation_MCMC import run_mcmc_bayesian_estimation

    return run_mcmc_bayesian_estimation(*args, **kwargs)


def run_complete_mcmc_analysis(*args, **kwargs):
    """Backward compatibility wrapper - delegates to canonical MCMC implementation"""
    from Falsification.FP_10_BayesianEstimation_MCMC import run_complete_mcmc_analysis

    return run_complete_mcmc_analysis(*args, **kwargs)


# Forward canonical class references
BayesianParameterRecovery = BayesianParameterRecovery
FP10Dispatcher = None  # Not used in this implementation


__all__ = [
    "run_bayesian_estimation_complete",
    "run_mcmc_bayesian_estimation",
    "run_complete_mcmc_analysis",
    "generate_synthetic_data",
    "BayesianParameterRecovery",
    "FP10bParameterRecovery",
    "FP10Dispatcher",
    "ParameterRecoveryResult",
]
