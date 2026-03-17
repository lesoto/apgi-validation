"""
Falsification Protocol 10: Cross-Species Scaling
===============================================

This protocol implements cross-species scaling analysis for APGI models.
Per Step 1.7 - Implement FP-10 real allometric scaling.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_cross_species_scaling(
    base_params: Dict[str, float], scaling_factors: Dict[str, float]
) -> Dict[str, float]:
    """Apply cross-species scaling to parameters"""

    scaled_params = {}

    for param, value in base_params.items():
        if param in scaling_factors:
            scaling_factor = scaling_factors[param]
            scaled_params[param] = value * scaling_factor
        else:
            scaled_params[param] = value

    return scaled_params


def calculate_allometric_exponent(
    values_x: List[float], values_y: List[float]
) -> Tuple[float, float, float, float]:
    """
    Calculate allometric scaling exponent using log-log regression.
    Per Step 1.7 - compute scaling exponents via scipy.stats.linregress.
    Returns: (exponent, intercept, r_value, p_value)
    """
    if len(values_x) < 2 or len(values_y) < 2:
        return 0.0, 0.0, 0.0, 1.0

    # Log-transform both variables
    log_x = np.log10(np.array(values_x) + 1e-10)
    log_y = np.log10(np.array(values_y) + 1e-10)

    # Linear regression using scipy.stats.linregress
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

    return float(slope), float(intercept), float(r_value), float(p_value)


def validate_allometric_relationship(
    observed_exponent: float,
    expected_exponent: float,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Validate that observed exponent matches expected value within tolerance.
    Per Step 1.7 - validate that exponents match expected values within tolerance ±0.05.
    """
    difference = abs(observed_exponent - expected_exponent)
    within_tolerance = difference <= tolerance

    return {
        "observed_exponent": observed_exponent,
        "expected_exponent": expected_exponent,
        "difference": difference,
        "within_tolerance": within_tolerance,
        "tolerance": tolerance,
    }


def validate_scaling_laws(
    species_data: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Validate that scaling laws are maintained across species.
    Per Step 1.7 - Replace placeholder scaling laws with actual allometric relationships from literature.
    """
    validation_results = {}

    # Extract data from all species
    brain_masses = []
    body_masses = []
    neuronal_densities = []
    synapse_counts = []

    for species, data in species_data.items():
        brain_masses.append(data.get("brain_mass", 0.0))
        body_masses.append(data.get("body_mass", 0.0))
        neuronal_densities.append(data.get("neuronal_density", 0.0))
        synapse_counts.append(data.get("synapse_count", 0.0))

    # Expected allometric relationships from literature
    # Brain mass scaling: M_brain ∝ M_body^0.75 (Kleiber's law)
    # Neuronal density scaling: ρ_neuron ∝ M_brain^−0.33
    # Synapse count scaling: N_synapse ∝ M_brain^1.0

    # Brain-to-body scaling
    (
        brain_body_exp,
        brain_body_int,
        brain_body_r,
        brain_body_p,
    ) = calculate_allometric_exponent(body_masses, brain_masses)
    validation_results["brain_to_body"] = validate_allometric_relationship(
        brain_body_exp, 0.75, tolerance=0.05
    )
    validation_results["brain_to_body"]["r_value"] = brain_body_r
    validation_results["brain_to_body"]["p_value"] = brain_body_p

    # Neuronal density scaling
    (
        neuron_density_exp,
        neuron_density_int,
        neuron_density_r,
        neuron_density_p,
    ) = calculate_allometric_exponent(brain_masses, neuronal_densities)
    validation_results["neuronal_density"] = validate_allometric_relationship(
        neuron_density_exp, -0.33, tolerance=0.05
    )
    validation_results["neuronal_density"]["r_value"] = neuron_density_r
    validation_results["neuronal_density"]["p_value"] = neuron_density_p

    # Synapse count scaling
    synapse_exp, synapse_int, synapse_r, synapse_p = calculate_allometric_exponent(
        brain_masses, synapse_counts
    )
    validation_results["synapse_count"] = validate_allometric_relationship(
        synapse_exp, 1.0, tolerance=0.05
    )
    validation_results["synapse_count"]["r_value"] = synapse_r
    validation_results["synapse_count"]["p_value"] = synapse_p

    return validation_results


def run_cross_species_scaling():
    """
    Run complete cross-species scaling analysis.
    Per Step 1.7 - Implement FP-10 real allometric scaling.
    """
    logger.info("Running cross-species scaling analysis...")

    # Species data from literature (mass in grams)
    # Per Step 1.7 - Add cross-validation across species (human, macaque, mouse, rat)
    species_data = {
        "human": {
            "body_mass": 70000.0,
            "brain_mass": 1350.0,
            "neuronal_density": 10000.0,
            "synapse_count": 1.5e14,
        },
        "macaque": {
            "body_mass": 7000.0,
            "brain_mass": 95.0,
            "neuronal_density": 12000.0,
            "synapse_count": 1.2e13,
        },
        "mouse": {
            "body_mass": 25.0,
            "brain_mass": 0.4,
            "neuronal": 12000.0,
            "neuronal_density": 12000.0,
            "synapse_count": 1.0e11,
        },
        "rat": {
            "body_mass": 300.0,
            "brain_mass": 2.0,
            "neuronal_density": 11000.0,
            "synapse_count": 5.0e11,
        },
    }

    # Validate scaling laws
    validation_results = validate_scaling_laws(species_data)

    # Calculate allometric exponents
    brain_masses = [data["brain_mass"] for data in species_data.values()]
    body_masses = [data["body_mass"] for data in species_data.values()]
    neuronal_densities = [data["neuronal_density"] for data in species_data.values()]
    synapse_counts = [data["synapse_count"] for data in species_data.values()]

    brain_body_exp, _, brain_body_r, brain_body_p = calculate_allometric_exponent(
        body_masses, brain_masses
    )
    (
        neuron_density_exp,
        _,
        neuron_density_r,
        neuron_density_p,
    ) = calculate_allometric_exponent(brain_masses, neuronal_densities)
    synapse_exp, _, synapse_r, synapse_p = calculate_allometric_exponent(
        brain_masses, synapse_counts
    )

    allometric_exponents = {
        "brain_to_body": {
            "exponent": brain_body_exp,
            "expected": 0.75,
            "r_value": brain_body_r,
            "p_value": brain_body_p,
            "within_tolerance": abs(brain_body_exp - 0.75) <= 0.05,
        },
        "neuronal_density": {
            "exponent": neuron_density_exp,
            "expected": -0.33,
            "r_value": neuron_density_r,
            "p_value": neuron_density_p,
            "within_tolerance": abs(neuron_density_exp - (-0.33)) <= 0.05,
        },
        "synapse_count": {
            "exponent": synapse_exp,
            "expected": 1.0,
            "r_value": synapse_r,
            "p_value": synapse_p,
            "within_tolerance": abs(synapse_exp - 1.0) <= 0.05,
        },
    }

    return {
        "species_data": species_data,
        "validation_results": validation_results,
        "allometric_exponents": allometric_exponents,
    }


if __name__ == "__main__":
    results = run_cross_species_scaling()
    print("Cross-species scaling results:")
    print(f"Species data: {results['species_data']}")
    print(f"Validation results: {results['validation_results']}")
    print(f"Allometric exponents: {results['allometric_exponents']}")
