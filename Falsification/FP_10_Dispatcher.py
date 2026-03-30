"""
Falsification Protocol 10 Dispatcher (FP-10)
===========================================

FP-10: Bayesian Model Evidence + Cross-Species Scaling
Two sub-protocols, both required; either failure falsifies FP-10.

This dispatcher routes FP-10 to both sub-protocols and aggregates results:
- FP10a: Bayesian MCMC Estimation (BF₁₀ ≥ 3 vs. PP/GWT)
- FP10b: Cross-Species Scaling (allometric exponents ±2 SD)
"""

from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class FP10Dispatcher:
    """Routes FP-10 to both sub-protocols and aggregates results."""

    def __init__(self, n_samples: int = 1000, n_chains: int = 2, burn_in: int = 500):
        """Initialize the FP-10 dispatcher.

        Args:
            n_samples: Number of MCMC samples for Bayesian estimation
            n_chains: Number of MCMC chains
            burn_in: Number of burn-in samples
        """
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.burn_in = burn_in

    def run_falsification(self) -> Dict[str, Any]:
        """Run both FP-10 sub-protocols and aggregate results.

        Returns:
            Dict with fp10a_mcmc, fp10b_scaling results and overall falsified status
        """
        # Import sub-protocols
        try:
            from FP_10_BayesianEstimation_MCMC import run_falsification as run_mcmc
        except ImportError as e:
            logger.error(f"Failed to import FP_10_BayesianEstimation_MCMC: {e}")
            run_mcmc = None

        try:
            from FP_12_CrossSpeciesScaling import run_falsification as run_scaling
        except ImportError as e:
            logger.error(f"Failed to import FP_12_CrossSpeciesScaling: {e}")
            run_scaling = None

        # Run MCMC sub-protocol (FP10a)
        if run_mcmc:
            logger.info("Running FP10a: Bayesian MCMC Estimation")
            try:
                mcmc_result = run_mcmc(
                    n_samples=self.n_samples,
                    n_chains=self.n_chains,
                    burn_in=self.burn_in,
                )
            except Exception as e:
                logger.error(f"Error in FP10a MCMC: {e}")
                mcmc_result = {
                    "passed": False,
                    "falsified": True,
                    "error": str(e),
                    "named_predictions": {
                        "fp10a_mcmc": {"passed": False, "error": str(e)},
                    },
                }
        else:
            mcmc_result = {
                "passed": False,
                "falsified": True,
                "error": "Module not available",
                "named_predictions": {
                    "fp10a_mcmc": {"passed": False, "error": "Module not available"},
                },
            }

        # Run Cross-Species Scaling sub-protocol (FP10b)
        if run_scaling:
            logger.info("Running FP10b: Cross-Species Scaling")
            try:
                scaling_result = run_scaling()
            except Exception as e:
                logger.error(f"Error in FP10b Scaling: {e}")
                scaling_result = {
                    "passed": False,
                    "falsified": True,
                    "error": str(e),
                    "named_predictions": {
                        "fp10b_scaling": {"passed": False, "error": str(e)},
                    },
                }
        else:
            scaling_result = {
                "passed": False,
                "falsified": True,
                "error": "Module not available",
                "named_predictions": {
                    "fp10b_scaling": {"passed": False, "error": "Module not available"},
                },
            }

        # Extract named predictions from sub-results
        mcmc_predictions = mcmc_result.get("named_predictions", {})
        scaling_predictions = scaling_result.get("named_predictions", {})

        # Combine named predictions
        combined_predictions = {
            "fp10a_mcmc": mcmc_predictions.get("fp10a_mcmc", {"passed": False}),
            "fp10b_bf": mcmc_predictions.get("fp10b_bf", {"passed": False}),
            "fp10c_mae": mcmc_predictions.get("fp10c_mae", {"passed": False}),
        }

        # Add scaling predictions (may have P12.a, P12.b or fp10b_scaling)
        if "fp10b_scaling" in scaling_predictions:
            combined_predictions["fp10b_scaling"] = scaling_predictions["fp10b_scaling"]
        elif "P12.a" in scaling_predictions:
            # Map legacy P12 predictions to fp10b format
            combined_predictions["fp10b_scaling"] = {
                "passed": (
                    scaling_predictions.get("P12.a", {}).get("passed", False)
                    and scaling_predictions.get("P12.b", {}).get("passed", False)
                ),
                "predictions": {
                    "P12.a": scaling_predictions.get("P12.a"),
                    "P12.b": scaling_predictions.get("P12.b"),
                },
            }
        else:
            combined_predictions["fp10b_scaling"] = {"passed": False}

        # Overall falsified if either sub-protocol fails
        falsified = mcmc_result.get("falsified", True) or scaling_result.get(
            "falsified", True
        )

        return {
            "fp10a_mcmc": mcmc_result,
            "fp10b_scaling": scaling_result,
            "falsified": falsified,
            "passed": not falsified,
            "status": "falsified" if falsified else "passed",
            "named_predictions": combined_predictions,
        }

    def get_falsification_criteria(self) -> Dict[str, str]:
        """Return falsification criteria for FP-10.

        Returns:
            Dict mapping criteria IDs to their descriptions
        """
        return {
            "FP10a": "BF₁₀ < 3 for APGI vs. Standard PP or GWT → FALSIFIED",
            "FP10b": "Allometric exponents deviate >2 SD from neurobiological expectation → FALSIFIED",
        }

    def run_full_experiment(self) -> Dict[str, Any]:
        """GUI-compatible entry point that runs the full falsification protocol.

        Returns:
            Dict with complete falsification results
        """
        return self.run_falsification()


# Standalone function entry points for direct import
def run_falsification(
    n_samples: int = 1000, n_chains: int = 2, burn_in: int = 500
) -> Dict[str, Any]:
    """Standalone entry point for FP-10 falsification.

    Args:
        n_samples: Number of MCMC samples
        n_chains: Number of MCMC chains
        burn_in: Number of burn-in samples

    Returns:
        Dict with falsification results
    """
    dispatcher = FP10Dispatcher(n_samples=n_samples, n_chains=n_chains, burn_in=burn_in)
    return dispatcher.run_falsification()


def get_falsification_criteria() -> Dict[str, str]:
    """Return falsification criteria for FP-10.

    Returns:
        Dict mapping criteria IDs to their descriptions
    """
    return {
        "FP10a": "BF₁₀ < 3 for APGI vs. Standard PP or GWT → FALSIFIED",
        "FP10b": "Allometric exponents deviate >2 SD from neurobiological expectation → FALSIFIED",
    }


if __name__ == "__main__":
    print("=" * 60)
    print("FP-10 Dispatcher: Bayesian MCMC + Cross-Species Scaling")
    print("=" * 60)

    dispatcher = FP10Dispatcher(n_samples=1000, n_chains=2, burn_in=500)

    print("\nFalsification Criteria:")
    for criterion, description in dispatcher.get_falsification_criteria().items():
        print(f"  {criterion}: {description}")

    print("\nRunning FP-10...")
    results = dispatcher.run_falsification()

    print(f"\nOverall Status: {'PASS' if results['passed'] else 'FALSIFIED'}")
    print(
        f"FP10a (MCMC): {'PASS' if not results['fp10a_mcmc'].get('falsified', True) else 'FALSIFIED'}"
    )
    print(
        f"FP10b (Scaling): {'PASS' if not results['fp10b_scaling'].get('falsified', True) else 'FALSIFIED'}"
    )

    print("\nNamed Predictions:")
    for pred_id, pred_data in results.get("named_predictions", {}).items():
        status = "PASS" if pred_data.get("passed", False) else "FAIL"
        print(f"  {pred_id}: {status}")

    print("\n" + "=" * 60)
