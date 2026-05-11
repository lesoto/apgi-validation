#!/usr/bin/env python3
"""
LEVEL DESIGNATION: Level 3 (algorithmic/mathematical)

Bridge to Level 2
Bridge to Level 1

FP-07 Mathematical Consistency - Validated Parameter Bounds
===========================================================

Canonical parameter bounds for FP-07 Mathematical Consistency protocol.
These bounds ensure that all parameter combinations tested in FP-07
remain within the valid domain of the APGI formal model.

Source: APGI Framework Specification, Falsification Protocol 07
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class ParameterBounds:
    """Validated parameter bounds"""

    # Time constants (seconds)
    tau_S_min: float = 0.1
    tau_S_max: float = 2.0
    tau_theta_min: float = 0.01
    tau_theta_max: float = 100.0
    tau_M_min: float = 0.01
    tau_M_max: float = 100.0

    # Gains and weights
    alpha_min: float = 0.1
    alpha_max: float = 10.0
    gamma_M_min: float = -1.0
    gamma_M_max: float = 1.0
    gamma_A_min: float = -1.0
    gamma_A_max: float = 1.0
    beta_min: float = -1.0
    beta_max: float = 1.0

    # Precision parameters
    Pi_e_min: float = 0.1
    Pi_e_max: float = 10.0
    Pi_i_baseline_min: float = 0.1
    Pi_i_baseline_max: float = 10.0
    beta_Pi_i_min: float = 0.0
    beta_Pi_i_max: float = 10.0

    # Threshold parameters
    theta_0_min: float = 0.0
    theta_0_max: float = 1.0
    rho_min: float = 0.0
    rho_max: float = 1.0

    # Noise parameters
    sigma_S_min: float = 0.0
    sigma_S_max: float = 1.0
    sigma_theta_min: float = 0.0
    sigma_theta_max: float = 1.0
    sigma_noise_min: float = 0.0
    sigma_noise_max: float = 1.0

    # Somatic parameters
    somatic_gain_min: float = 0.0
    somatic_gain_max: float = 1.0
    homeostatic_cost_weight_min: float = 0.0
    homeostatic_cost_weight_max: float = 1.0

    # Network dimensions
    hidden_dim_min: int = 8
    hidden_dim_max: int = 256
    somatic_hidden_dim_min: int = 4
    somatic_hidden_dim_max: int = 128

    # Learning rates
    learning_rate_min: float = 0.0001
    learning_rate_max: float = 0.1
    value_learning_rate_min: float = 0.0001
    value_learning_rate_max: float = 0.1

    # Simulation parameters
    dt_min: float = 0.001
    dt_max: float = 0.1
    duration_min: float = 0.1
    duration_max: float = 1000.0

    def get_bounds_dict(self) -> Dict[str, Tuple[float, float]]:
        """Return all bounds as a dictionary of (min, max) tuples."""
        bounds = {}
        for attr_name in dir(self):
            if attr_name.endswith("_min"):
                base_name = attr_name[:-4]  # Remove "_min"
                max_attr = f"{base_name}_max"
                if hasattr(self, max_attr):
                    min_val = getattr(self, attr_name)
                    max_val = getattr(self, max_attr)
                    bounds[base_name] = (min_val, max_val)
        return bounds

    def validate_parameter(self, param_name: str, value: float) -> Tuple[bool, str]:
        """
        Validate a single parameter against bounds.

        Returns:
            (is_valid, message)
        """
        bounds_dict = self.get_bounds_dict()

        if param_name not in bounds_dict:
            return False, f"Unknown parameter: {param_name}"

        min_val, max_val = bounds_dict[param_name]

        if value < min_val or value > max_val:
            return False, f"{param_name}={value} out of bounds [{min_val}, {max_val}]"

        return True, f"{param_name}={value} is valid"

    def validate_parameters(self, params: Dict[str, float]) -> Tuple[bool, list]:
        """
        Validate multiple parameters.

        Returns:
            (all_valid, list_of_messages)
        """
        messages = []
        all_valid = True

        for param_name, value in params.items():
            is_valid, message = self.validate_parameter(param_name, value)
            messages.append(message)
            if not is_valid:
                all_valid = False

        return all_valid, messages


# Global instance
FP07_BOUNDS = ParameterBounds()


# Canonical threshold registry for FP-07
FP07_THRESHOLD_REGISTRY = {
    "F1.1_ADVANTAGE": 0.60,  # Minimum advantage for F1.1
    "F1.1_COHENS_D": 0.50,  # Minimum Cohen's d for F1.1
    "F2.1_ADVANTAGE": 0.60,  # Minimum advantage for F2.1
    "F2.1_PP_DIFF": 0.15,  # Minimum PP difference for F2.1
    "F2.1_COHENS_H": 0.40,  # Minimum Cohen's h for F2.1
    "F2.1_ALPHA": 0.05,  # Alpha level for F2.1
    "F2.2_CORR": 0.30,  # Minimum correlation for F2.2
    "F2.2_FISHER_Z": 0.31,  # Minimum Fisher Z for F2.2
    "F2.2_ALPHA": 0.05,  # Alpha level for F2.2
    "F2.3_RT_ADVANTAGE": 50.0,  # Minimum RT advantage (ms) for F2.3
    "F2.3_BETA": 0.30,  # Minimum beta for F2.3
    "F2.3_ALPHA": 0.05,  # Alpha level for F2.3
    "F2.4_CONFIDENCE_EFFECT": 0.20,  # Minimum confidence effect for F2.4
    "F2.4_BETA_INTERACTION": 0.15,  # Minimum beta interaction for F2.4
    "F2.4_ALPHA": 0.05,  # Alpha level for F2.4
    "F2.5_MAX_TRIALS": 1000,  # Maximum trials for F2.5
    "F2.5_HAZARD_RATIO": 1.50,  # Minimum hazard ratio for F2.5
    "F2.5_ALPHA": 0.05,  # Alpha level for F2.5
    "F3.1_ADVANTAGE": 0.60,  # Minimum advantage for F3.1
    "F3.1_COHENS_D": 0.50,  # Minimum Cohen's d for F3.1
    "F3.1_ALPHA": 0.05,  # Alpha level for F3.1
    "F3.2_INTERO_ADVANTAGE": 0.60,  # Minimum interoceptive advantage for F3.2
    "F3.2_COHENS_D": 0.50,  # Minimum Cohen's d for F3.2
    "F3.2_ALPHA": 0.05,  # Alpha level for F3.2
    "F3.3_REDUCTION": 0.30,  # Minimum reduction for F3.3
    "F3.3_COHENS_D": 0.50,  # Minimum Cohen's d for F3.3
    "F3.3_ALPHA": 0.05,  # Alpha level for F3.3
    "F3.4_REDUCTION": 0.30,  # Minimum reduction for F3.4
    "F3.4_COHENS_D": 0.50,  # Minimum Cohen's d for F3.4
    "F3.4_ALPHA": 0.05,  # Alpha level for F3.4
    "F3.6_MAX_TRIALS": 1000,  # Maximum trials for F3.6
    "F3.6_HAZARD_RATIO": 1.50,  # Minimum hazard ratio for F3.6
    "F5.5_PCA_VARIANCE": 0.70,  # Minimum PCA variance for F5.5
    "F5.5_PCA_LOADING": 0.60,  # Minimum PC loading for F5.5
    "F5.6_PERF_DIFF": 0.40,  # Minimum performance difference for F5.6
    "F5.6_COHENS_D": 0.40,  # Minimum Cohen's d for F5.6
    "F5.6_ALPHA": 0.05,  # Alpha level for F5.6
    "F6.1_LTCN_TRANSITION": 50.0,  # Maximum LTCN transition time (ms) for F6.1
    "F6.2_INTEGRATION_RATIO": 4.0,  # Minimum integration ratio for F6.2
    "F6.2_R2": 0.85,  # Minimum R² for F6.2
    "V7.1_PCI_REDUCTION": 0.15,  # Minimum PCI reduction for V7.1
    "V7.1_COHENS_D": 0.70,  # Minimum Cohen's d for V7.1
    "V9.1_CORR": 0.30,  # Minimum correlation for V9.1
    "V9.3_CORR": 0.30,  # Minimum correlation for V9.3
}


def get_parameter_bounds() -> ParameterBounds:
    """Get the canonical parameter bounds for FP-07."""
    return FP07_BOUNDS


def validate_fp07_parameters(params: Dict[str, float]) -> Tuple[bool, list]:
    """
    Validate parameters for FP-07 protocol.

    Args:
        params: Dictionary of parameter names and values

    Returns:
        (all_valid, list_of_validation_messages)
    """
    return FP07_BOUNDS.validate_parameters(params)


if __name__ == "__main__":
    # Example usage
    bounds = get_parameter_bounds()
    print("FP-07 Parameter Bounds")
    print("=" * 50)

    bounds_dict = bounds.get_bounds_dict()
    for param_name, (min_val, max_val) in sorted(bounds_dict.items()):
        print(f"{param_name:30s}: [{min_val:10.4f}, {max_val:10.4f}]")

    print("\n" + "=" * 50)
    print("FP-07 Threshold Registry")
    print("=" * 50)

    for threshold_name, threshold_val in sorted(FP07_THRESHOLD_REGISTRY.items()):
        print(f"{threshold_name:30s}: {threshold_val}")
