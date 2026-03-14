"""
Falsification Protocol 8: Parameter Sensitivity Analysis
======================================================

This protocol implements parameter sensitivity analysis for APGI models.
"""

import logging
import numpy as np
from typing import Dict, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_parameter_sensitivity(
    base_params: Dict[str, float], param_ranges: Dict[str, Tuple[float, float]]
) -> Dict[str, Any]:
    """Analyze sensitivity of model performance to parameter changes"""

    sensitivity_results = {}

    for param_name, (min_val, max_val) in param_ranges.items():
        if param_name not in base_params:
            continue

        # Test parameter variations
        test_values = np.linspace(min_val, max_val, 5)
        param_sensitivity = []

        for test_value in test_values:
            # Create modified parameters
            test_params = base_params.copy()
            test_params[param_name] = test_value

            # Run simple test (placeholder - would normally run full model)
            performance = simulate_model_performance(test_params)
            param_sensitivity.append(performance)

        # Calculate sensitivity metric
        sensitivity = (
            np.std(param_sensitivity) / np.mean(param_sensitivity)
            if np.mean(param_sensitivity) > 0
            else 0
        )
        sensitivity_results[param_name] = {
            "sensitivity": sensitivity,
            "test_values": test_values.tolist(),
            "performances": param_sensitivity,
        }

    return sensitivity_results


def simulate_model_performance(params: Dict[str, float]) -> float:
    """Simulate model performance with given parameters (placeholder)"""
    # Simple performance simulation based on parameter values
    base_performance = 0.5

    # Add parameter effects
    if "theta_0" in params:
        base_performance += 0.1 * params["theta_0"]
    if "alpha" in params:
        base_performance += 0.05 * np.log(params["alpha"])
    if "beta" in params:
        base_performance -= 0.02 * params["beta"]

    # Add noise
    noise = np.random.normal(0, 0.05)
    performance = base_performance + noise

    return np.clip(performance, 0.0, 1.0)


def generate_sensitivity_report(sensitivity_results: Dict[str, Any]) -> str:
    """Generate a human-readable sensitivity report"""

    report = "Parameter Sensitivity Analysis Report\n"
    report += "=" * 40 + "\n\n"

    for param_name, results in sensitivity_results.items():
        sensitivity = results["sensitivity"]
        report += f"Parameter: {param_name}\n"
        report += f"Sensitivity: {sensitivity:.4f}\n"

        if sensitivity > 0.1:
            report += "Status: HIGH SENSITIVITY\n"
        elif sensitivity > 0.05:
            report += "Status: MODERATE SENSITIVITY\n"
        else:
            report += "Status: LOW SENSITIVITY\n"

        report += "\n"

    return report


def run_parameter_sensitivity_analysis():
    """Run complete parameter sensitivity analysis"""
    logger.info("Running parameter sensitivity analysis...")

    # Base parameters
    base_params = {
        "theta_0": 0.5,
        "alpha": 5.0,
        "beta": 1.2,
        "Pi_e_lr": 0.01,
    }

    # Parameter ranges to test
    param_ranges = {
        "theta_0": (0.1, 0.9),
        "alpha": (1.0, 10.0),
        "beta": (0.5, 3.0),
        "Pi_e_lr": (0.001, 0.05),
    }

    # Analyze sensitivity
    sensitivity_results = analyze_parameter_sensitivity(base_params, param_ranges)

    # Generate report
    report = generate_sensitivity_report(sensitivity_results)

    return {
        "sensitivity_results": sensitivity_results,
        "report": report,
    }


if __name__ == "__main__":
    results = run_parameter_sensitivity_analysis()
    print(results["report"])
