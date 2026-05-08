#!/usr/bin/env python3
"""
Bridge Invocations for Cross-Level APGI Communications
===================================================

Provides explicit bridge functions for cross-level outputs between APGI analytical levels.
These functions implement the required bridge invocation pattern where
lower-level modules must explicitly call higher-level modules to make cross-level
claims.

Bridge Pattern:
- Level 3 (algorithmic/mathematical) → Level 2 (information-theoretic) → Level 1 (thermodynamic)
- Each bridge function validates level constraints and provides proper error handling
- Cross-level outputs require explicit bridge invocation, not implicit claims

Usage Example:
    from utils.bridge_invocations import bridge_to_thermodynamic

    # In a Level 2 module making thermodynamic claims
    thermodynamic_results = bridge_to_thermodynamic(
        bandwidth_data=bandwidth_results,
        module_name="APGI_Information_Theoretic_Bandwidth"
    )
"""

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class BridgeValidationError(Exception):
    """Raised when bridge validation fails."""

    pass


def bridge_to_thermodynamic(
    level2_data: Dict[str, Any],
    module_name: str,
    bridge_reason: str = "thermodynamic implications",
) -> Dict[str, Any]:
    """
    Bridge from Level 2 (information-theoretic) to Level 1 (thermodynamic).

    This function implements the explicit bridge invocation required for
    Level 2 modules to make thermodynamic claims. It calls the
    APGI_Thermodynamic_Program_Aggregator to convert information-theoretic
    outputs to thermodynamic quantities.

    Args:
        level2_data: Information-theoretic results from Level 2 module
        module_name: Name of the calling Level 2 module
        bridge_reason: Description of why bridge is needed

    Returns:
        Dictionary containing thermodynamic results

    Raises:
        BridgeValidationError: If bridge validation fails
    """
    try:
        # Import Level 1 module
        from Theory.APGI_Thermodynamic_Program_Aggregator import (
            run_thermodynamic_analysis,
        )

        logger.info(
            f"Bridge L2→L1: {module_name} → APGI_Thermodynamic_Program_Aggregator"
        )
        logger.info(f"Bridge reason: {bridge_reason}")

        # Validate Level 2 data structure
        if not isinstance(level2_data, dict):
            raise BridgeValidationError("Level 2 data must be a dictionary")

        # Extract key information-theoretic quantities
        bandwidth_bits = level2_data.get("bandwidth_bits_per_second")
        precision_weights = level2_data.get("precision_weights")
        channel_capacity = level2_data.get("channel_capacity")

        if bandwidth_bits is None:
            raise BridgeValidationError(
                "Missing required field: bandwidth_bits_per_second"
            )

        # Run thermodynamic analysis
        thermodynamic_results = run_thermodynamic_analysis(
            information_bits=bandwidth_bits,
            precision_weights=precision_weights,
            channel_capacity=channel_capacity,
        )

        # Add bridge metadata
        thermodynamic_results["_bridge_metadata"] = {
            "source_level": 2,
            "target_level": 1,
            "source_module": module_name,
            "bridge_reason": bridge_reason,
            "timestamp": level2_data.get("timestamp"),
        }

        return thermodynamic_results

    except ImportError as e:
        raise BridgeValidationError(f"Cannot import Level 1 module: {e}")
    except Exception as e:
        raise BridgeValidationError(f"Bridge execution failed: {e}")


def bridge_to_information_theoretic(
    level3_data: Dict[str, Any],
    module_name: str,
    bridge_reason: str = "information-theoretic analysis",
) -> Dict[str, Any]:
    """
    Bridge from Level 3 (algorithmic/mathematical) to Level 2 (information-theoretic).

    This function implements the explicit bridge invocation required for
    Level 3 modules to make information-theoretic claims. It calls the
    APGI_Information_Theoretic_Bandwidth module to convert algorithmic
    outputs to information-theoretic quantities.

    Args:
        level3_data: Algorithmic/mathematical results from Level 3 module
        module_name: Name of the calling Level 3 module
        bridge_reason: Description of why bridge is needed

    Returns:
        Dictionary containing information-theoretic results

    Raises:
        BridgeValidationError: If bridge validation fails
    """
    try:
        # Import Level 2 module
        from Theory.APGI_Information_Theoretic_Bandwidth import (
            analyze_precision_weighting_gain,
            calculate_bandwidth_from_precision,
        )

        logger.info(
            f"Bridge L3→L2: {module_name} → APGI_Information_Theoretic_Bandwidth"
        )
        logger.info(f"Bridge reason: {bridge_reason}")

        # Validate Level 3 data structure
        if not isinstance(level3_data, dict):
            raise BridgeValidationError("Level 3 data must be a dictionary")

        # Extract key algorithmic quantities
        precision_values = level3_data.get("precision_values")
        ignition_events = level3_data.get("ignition_events")

        if precision_values is None:
            raise BridgeValidationError("Missing required field: precision_values")

        # Run information-theoretic analysis
        info_theoretic_results = {
            "bandwidth_bits_per_second": calculate_bandwidth_from_precision(
                precision_values=precision_values, surprise_events=ignition_events
            ),
            "precision_weighting_gain": analyze_precision_weighting_gain(
                precision_values=precision_values
            ),
        }

        # Add bridge metadata
        info_theoretic_results["_bridge_metadata"] = {
            "source_level": 3,
            "target_level": 2,
            "source_module": module_name,
            "bridge_reason": bridge_reason,
            "timestamp": level3_data.get("timestamp"),
        }

        return info_theoretic_results

    except ImportError as e:
        raise BridgeValidationError(f"Cannot import Level 2 module: {e}")
    except Exception as e:
        raise BridgeValidationError(f"Bridge execution failed: {e}")


def validate_level_constraints(
    data: Dict[str, Any], expected_level: int, module_name: str
) -> bool:
    """
    Validate that data conforms to expected level constraints.

    Args:
        data: Data to validate
        expected_level: Expected analytical level (1, 2, or 3)
        module_name: Name of the module for error reporting

    Returns:
        True if validation passes

    Raises:
        BridgeValidationError: If validation fails
    """
    level_constraints = {
        1: {
            "allowed_fields": [
                "atp_cost_per_bit",
                "efficiency_ratio",
                "metabolic_scaling",
            ],
            "forbidden_fields": {
                "precision_weights": "precision_weights",
                "bandwidth_bits": "bandwidth_bits",
            },
            "description": "thermodynamic",
        },
        2: {
            "allowed_fields": [
                "bandwidth_bits",
                "precision_weights",
                "channel_capacity",
            ],
            "forbidden_fields": {
                "atp_cost_per_bit": "atp_cost_per_bit",
                "metabolic_scaling": "metabolic_scaling",
            },
            "description": "information-theoretic",
        },
        3: {
            "allowed_fields": [
                "precision_values",
                "surprise_values",
                "ignition_events",
                "algorithmic_metrics",
            ],
            "forbidden_fields": {
                "atp_cost_per_bit": "atp_cost_per_bit",
                "bandwidth_bits": "bandwidth_bits",
            },
            "description": "algorithmic/mathematical",
        },
    }

    if expected_level not in level_constraints:
        raise BridgeValidationError(f"Invalid level: {expected_level}")

    constraints = level_constraints[expected_level]

    # Check for forbidden fields
    for field in data.keys():
        if field in constraints["forbidden_fields"]:
            raise BridgeValidationError(
                f"Level {expected_level} module {module_name} contains forbidden field "
                f"'{field}' ({constraints['description']} level cannot claim {constraints['forbidden_fields'].get(field, 'unknown field')})"
            )

    logger.info(f"Level {expected_level} validation passed for {module_name}")
    return True


def create_bridge_function(
    source_level: int, target_level: int, source_module: str
) -> Callable:
    """
    Create a bridge function for specific level transition.

    Args:
        source_level: Source analytical level (1, 2, or 3)
        target_level: Target analytical level (1, 2, or 3)
        source_module: Name of the source module

    Returns:
        Bridge function for the specified level transition
    """
    if source_level == 3 and target_level == 2:
        return lambda data: bridge_to_information_theoretic(
            level3_data=data,
            module_name=source_module,
            bridge_reason=f"Level 3 {source_module} requires Level 2 analysis",
        )
    elif source_level == 2 and target_level == 1:
        return lambda data: bridge_to_thermodynamic(
            level2_data=data,
            module_name=source_module,
            bridge_reason=f"Level 2 {source_module} requires Level 1 thermodynamic analysis",
        )
    else:
        raise BridgeValidationError(
            f"Invalid bridge: Level {source_level} → Level {target_level} not supported"
        )


# Example bridge function for common use cases
def bridge_l2_to_l1_with_validation(
    level2_data: Dict[str, Any], module_name: str
) -> Dict[str, Any]:
    """
    Bridge from Level 2 to Level 1 with validation.

    This is a convenience function that combines validation and bridging
    for the most common L2→L1 transition.

    Args:
        level2_data: Information-theoretic results
        module_name: Name of the Level 2 module

    Returns:
        Thermodynamic results with validation metadata
    """
    # Validate Level 2 constraints
    validate_level_constraints(level2_data, 2, module_name)

    # Perform bridge
    return bridge_to_thermodynamic(level2_data, module_name)


def bridge_l3_to_l2_with_validation(
    level3_data: Dict[str, Any], module_name: str
) -> Dict[str, Any]:
    """
    Bridge from Level 3 to Level 2 with validation.

    This is a convenience function that combines validation and bridging
    for the most common L3→L2 transition.

    Args:
        level3_data: Algorithmic/mathematical results
        module_name: Name of the Level 3 module

    Returns:
        Information-theoretic results with validation metadata
    """
    # Validate Level 3 constraints
    validate_level_constraints(level3_data, 3, module_name)

    # Perform bridge
    return bridge_to_information_theoretic(level3_data, module_name)
