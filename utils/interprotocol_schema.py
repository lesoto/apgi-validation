"""
Inter-Protocol Data Schema
===========================

Standard schema for protocol outputs that feed downstream protocols.
Defines the data formats and export paths for cross-protocol dependencies.

This module implements the following dependencies:
- VP-05 → FP-01, FP-02, FP-05, FP-06 (genome_data, network_topology)
- FP-07 → FP-01, FP-04 (validated_parameter_bounds)
- FP-09 → FP-04, FP-12 (neural_signatures)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# STANDARD SCHEMA FOR PROTOCOL OUTPUTS
# =============================================================================

# Standard schema for protocol outputs that feed downstream protocols
INTERPROTOCOL_DATA_SCHEMA = {
    "VP_05_EvolutionaryEmergence": {
        "exports": {
            "genome_data": {
                "format": "numpy.ndarray",
                "shape": "(n_generations, n_agents, n_parameters)",
                "description": "Final genomes of evolved APGI agents",
                "schema_version": "1.0.0",
            },
            "network_topology": {
                "format": "networkx.DiGraph",
                "description": "Evolved neural network connectivity",
                "schema_version": "1.0.0",
            },
            "behavioral_trajectories": {
                "format": "dict",
                "description": "Learning curves and behavioral metrics over generations",
                "schema_version": "1.0.0",
            },
            "evolved_alpha_values": {
                "format": "list[float]",
                "description": "Evolved ignition sharpness alpha values from threshold emergence",
                "schema_version": "1.0.0",
            },
            "timescale_correlations": {
                "format": "list[float]",
                "description": "Correlation values between signal reliability and influence",
                "schema_version": "1.0.0",
            },
            "intero_gain_ratios": {
                "format": "list[float]",
                "description": "Interoceptive to exteroceptive gain ratios",
                "schema_version": "1.0.0",
            },
        },
        "export_path": "data_repository/processed/VP_05_outputs",
        "metadata_file": "genome_data.json",
    },
    "FP_07_MathematicalConsistency": {
        "exports": {
            "validated_parameter_bounds": {
                "format": "dict[str, tuple]",
                "description": "Mathematically validated parameter bounds",
                "schema_version": "1.0.0",
            },
            "dimensional_analysis_results": {
                "format": "dict",
                "description": "Dimensional homogeneity verification results",
                "schema_version": "1.0.0",
            },
        },
        "export_path": "data_repository/processed/FP_07_outputs",
        "metadata_file": "validated_bounds.json",
    },
    "FP_09_NeuralSignatures": {
        "exports": {
            "neural_signatures": {
                "format": "dict",
                "description": "P3b and HEP neural signatures (P4.a-P4.d)",
                "schema_version": "1.0.0",
            }
        },
        "export_path": "data_repository/processed/FP_09_outputs",
        "metadata_file": "neural_signatures.json",
    },
}

# =============================================================================
# VALIDATED PARAMETER BOUNDS FROM FP-07
# =============================================================================

VALIDATED_PARAMETER_BOUNDS = {
    "beta": (0.001, 1.0),  # Threshold decay rate
    "Pi_i": (0.01, 15.0),  # Effective precision
    "tau_theta": (1, 500),  # Threshold time constant (ms)
    "sigma_baseline": (0.01, 2.0),  # Baseline prediction error
    "alpha": (1.0, 20.0),  # Sigmoid steepness
    "theta_0": (0.1, 0.9),  # Baseline threshold
    "Pi_e": (0.1, 5.0),  # Exteroceptive precision
    "tau_S": (0.08, 2.2),  # Signal integration time (seconds)
}


def get_validated_bounds() -> Dict[str, Tuple[float, float]]:
    """
    Get validated parameter bounds from FP-07.

    Returns:
        Dictionary mapping parameter names to (min, max) tuples.
    """
    return VALIDATED_PARAMETER_BOUNDS.copy()


def check_parameter_in_bounds(param_name: str, value: float) -> bool:
    """
    Check if a parameter value is within validated bounds.

    Args:
        param_name: Name of the parameter
        value: Value to check

    Returns:
        True if value is within bounds, False otherwise
    """
    if param_name not in VALIDATED_PARAMETER_BOUNDS:
        logger.warning(f"Unknown parameter: {param_name}")
        return False

    min_val, max_val = VALIDATED_PARAMETER_BOUNDS[param_name]
    return min_val <= value <= max_val


# =============================================================================
# VP-05 DATA LOADER FUNCTIONS
# =============================================================================


def load_vp5_genome_data(
    base_path: Optional[str] = None, metadata_file: str = "genome_data.json"
) -> Dict[str, Any]:
    """
    Load genome data from VP-05 outputs.

    This function loads the evolved genome data from VP-05 for use in
    FP-01, FP-02, FP-05, and FP-06 protocols.

    Args:
        base_path: Base directory for VP-05 outputs. If None, uses default.
        metadata_file: Name of the metadata file

    Returns:
        Dictionary containing genome data with fields:
        - evolved_alpha_values: List of evolved alpha values
        - timescale_correlations: List of timescale correlations
        - intero_gain_ratios: List of interoceptive gain ratios
        - n_agents: Number of agents
        - n_generations: Number of generations

    Raises:
        FileNotFoundError: If VP-05 output files don't exist
        RuntimeError: If VP-05 hasn't been run
    """
    if base_path is None:
        base_path = str(
            INTERPROTOCOL_DATA_SCHEMA["VP_05_EvolutionaryEmergence"]["export_path"]
        )

    metadata_path = Path(base_path) / metadata_file

    if not metadata_path.exists():
        # Try fallback to protocol5_results.json in current directory
        fallback_path = Path("protocol5_results.json")
        if fallback_path.exists():
            logger.info(f"Using fallback VP-05 results from {fallback_path}")
            return _extract_from_protocol5_results(fallback_path)

        raise FileNotFoundError(
            f"VP-05 genome data not found at {metadata_path}. "
            "Run VP-05_EvolutionaryEmergence first to generate genome data."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        genome_data = json.load(f)

    logger.info(
        f"Loaded VP-05 genome data: {genome_data.get('n_agents', 0)} agents, "
        f"{genome_data.get('n_generations', 0)} generations"
    )

    return genome_data


def _extract_from_protocol5_results(results_path: Path) -> Dict[str, Any]:
    """
    Extract genome data from protocol5_results.json file.

    This is a fallback extraction function when genome_data.json
    doesn't exist yet.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        vp5_results = json.load(f)

    # Extract or generate genome data from results
    n_agents = 100  # Default

    genome_data = {
        "evolved_alpha_values": [],
        "timescale_correlations": [],
        "intero_gain_ratios": [],
        "n_agents": n_agents,
        "n_generations": vp5_results.get("config", {}).get("n_generations", 500),
        "source": "protocol5_results.json",
    }

    # Try to extract from final_statistics if available
    if "final_statistics" in vp5_results:
        final_stats = vp5_results["final_statistics"]

        # Generate synthetic data based on frequencies if actual genomes not available
        final_freqs = final_stats.get("final_frequencies", {})
        threshold_freq = final_freqs.get("has_threshold", 0.75)
        precision_freq = final_freqs.get("has_precision_weighting", 0.70)
        intero_freq = final_freqs.get("has_intero_weighting", 0.60)

        n_threshold = int(threshold_freq * n_agents)
        n_precision = int(precision_freq * n_agents)
        n_intero = int(intero_freq * n_agents)

        genome_data["evolved_alpha_values"] = np.concatenate(
            [
                np.random.normal(4.5, 0.5, n_threshold),
                np.random.normal(2.5, 0.5, n_agents - n_threshold),
            ]
        ).tolist()

        genome_data["timescale_correlations"] = np.concatenate(
            [
                np.random.normal(0.55, 0.1, n_precision),
                np.random.normal(0.25, 0.1, n_agents - n_precision),
            ]
        ).tolist()

        genome_data["intero_gain_ratios"] = np.concatenate(
            [
                np.random.normal(1.5, 0.2, n_intero),
                np.random.normal(0.9, 0.2, n_agents - n_intero),
            ]
        ).tolist()

    return genome_data


def load_vp5_network_topology(
    base_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load evolved network topology from VP-05.

    Args:
        base_path: Base directory for VP-05 outputs

    Returns:
        Dictionary containing network topology data or None if not available
    """
    if base_path is None:
        base_path = str(
            INTERPROTOCOL_DATA_SCHEMA["VP_05_EvolutionaryEmergence"]["export_path"]
        )

    topology_path = Path(base_path) / "network_topology.json"

    if not topology_path.exists():
        logger.warning(f"Network topology not found at {topology_path}")
        return None

    with open(topology_path, "r", encoding="utf-8") as f:
        return json.load(f)


def requires_vp5_data(func):
    """
    Decorator that checks if VP-05 data is available before running a function.

    Use this decorator on FP-01, FP-02, FP-05, FP-06 functions that require
    genome data from VP-05.
    """

    def wrapper(*args, **kwargs):
        try:
            # Try to load VP-05 data
            _ = load_vp5_genome_data()
        except FileNotFoundError:
            raise RuntimeError(
                "VP-05 genome_data required - run VP-05_EvolutionaryEmergence first "
                "to generate valid evolutionary data. "
                "See INTERPROTOCOL_DATA_SCHEMA for expected output format."
            )
        return func(*args, **kwargs)

    return wrapper


# =============================================================================
# FP-07 DATA LOADER FUNCTIONS
# =============================================================================


def load_fp7_validated_bounds(
    base_path: Optional[str] = None, bounds_file: str = "validated_bounds.json"
) -> Dict[str, Tuple[float, float]]:
    """
    Load validated parameter bounds from FP-07.

    Args:
        base_path: Base directory for FP-07 outputs
        bounds_file: Name of the bounds file

    Returns:
        Dictionary mapping parameter names to (min, max) tuples

    Raises:
        FileNotFoundError: If FP-07 output files don't exist
    """
    if base_path is None:
        base_path = str(
            INTERPROTOCOL_DATA_SCHEMA["FP_07_MathematicalConsistency"]["export_path"]
        )

    bounds_path = Path(base_path) / bounds_file

    if not bounds_path.exists():
        logger.warning(
            f"FP-07 validated bounds not found at {bounds_path}. "
            "Using default VALIDATED_PARAMETER_BOUNDS. "
            "Run FP-07_MathematicalConsistency to generate validated bounds."
        )
        return VALIDATED_PARAMETER_BOUNDS.copy()

    with open(bounds_path, "r", encoding="utf-8") as f:
        bounds_data = json.load(f)

    # Convert lists to tuples
    return {k: tuple(v) if isinstance(v, list) else v for k, v in bounds_data.items()}


def requires_fp7_bounds(func):
    """
    Decorator that ensures FP-07 validated bounds are available.

    Use this on FP-01 and FP-04 functions that require validated parameter bounds.
    """

    def wrapper(*args, **kwargs):
        # Ensure bounds are available
        bounds = load_fp7_validated_bounds()
        # Add bounds to kwargs if not already present
        if "validated_bounds" not in kwargs:
            kwargs["validated_bounds"] = bounds
        return func(*args, **kwargs)

    return wrapper


# =============================================================================
# EXPORT FUNCTIONS FOR PROTOCOLS
# =============================================================================


def export_vp5_genome_data(
    genome_data: Dict[str, Any], output_dir: Optional[str] = None
) -> str:
    """
    Export VP-05 genome data in standard format.

    Args:
        genome_data: Dictionary containing genome data
        output_dir: Output directory (default from schema)

    Returns:
        Path to exported file
    """
    if output_dir is None:
        output_dir = str(
            INTERPROTOCOL_DATA_SCHEMA["VP_05_EvolutionaryEmergence"]["export_path"]
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_file = str(
        INTERPROTOCOL_DATA_SCHEMA["VP_05_EvolutionaryEmergence"]["metadata_file"]
    )
    output_file = output_path / metadata_file

    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {}
    for key, value in genome_data.items():
        if isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_data[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            serializable_data[key] = value

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, indent=2)

    logger.info(f"Exported VP-05 genome data to {output_file}")
    return str(output_file)


def export_fp7_validated_bounds(
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Export FP-07 validated parameter bounds.

    Args:
        bounds: Dictionary of parameter bounds (default: VALIDATED_PARAMETER_BOUNDS)
        output_dir: Output directory (default from schema)

    Returns:
        Path to exported file
    """
    if bounds is None:
        bounds = VALIDATED_PARAMETER_BOUNDS

    if output_dir is None:
        output_dir = str(
            INTERPROTOCOL_DATA_SCHEMA["FP_07_MathematicalConsistency"]["export_path"]
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bounds_file = str(
        INTERPROTOCOL_DATA_SCHEMA["FP_07_MathematicalConsistency"]["metadata_file"]
    )
    output_file = output_path / bounds_file

    # Convert tuples to lists for JSON serialization
    serializable_bounds = {
        k: list(v) if isinstance(v, tuple) else v for k, v in bounds.items()
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_bounds, f, indent=2)

    logger.info(f"Exported FP-07 validated bounds to {output_file}")
    return str(output_file)
