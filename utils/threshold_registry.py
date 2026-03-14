"""
APGI Theory Framework - Centralized Threshold Registry
===============================================

Centralized registry for managing falsification thresholds across all validation protocols.
Provides a single source of truth for threshold values that can be easily
maintained and updated without requiring changes across multiple protocol files.

Features:
- Centralized threshold definitions using constants from utils.constants
- Dynamic threshold loading from configuration with fallback to defaults
- Type-safe threshold access with validation
- Support for both individual thresholds and threshold sets
- Easy integration with existing validation protocols

Example:
    >>> from utils.threshold_registry import ThresholdRegistry
    >>> registry = ThresholdRegistry()
    >>> thresholds = registry.get_falsification_thresholds()
    >>> print(f"Cumulative reward advantage threshold: {thresholds.cumulative_reward_advantage_threshold}")
"""

import logging
from typing import Dict
from dataclasses import dataclass

try:
    from .constants import (
        ParameterBounds,
        IGNITION_THRESHOLD,
        MIN_SAMPLES_FOR_REGRESSION,
        DEFAULT_EPSILON,
    )
except ImportError:
    # Fallback constants if constants module not available
    @dataclass
    class ParameterBounds:
        """Fallback parameter bounds."""

        precision_min: float = 0.1
        precision_max: float = 10.0
        tau_min: float = 0.01
        tau_max: float = 100.0
        alpha_min: float = 0.0
        alpha_max: float = 15.0
        gamma_min: float = -1.0
        gamma_max: float = 1.0
        rho_min: float = 0.0
        rho_max: float = 1.0
        sigma_min: float = 0.0
        sigma_max: float = 2.0

    IGNITION_THRESHOLD = 0.5
    MIN_SAMPLES_FOR_REGRESSION = 10
    DEFAULT_EPSILON = 1e-8

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FalsificationThresholds:
    """Centralized falsification threshold configuration."""

    # Core falsification thresholds (from config/default.yaml)
    cumulative_reward_advantage_threshold: float = 18.0
    cohens_d_threshold: float = 0.60
    significance_level: float = 0.01
    threshold_reduction_min: float = 20.0
    cohens_d_adaptation_threshold: float = 0.70
    tau_theta_min: float = 10.0
    tau_theta_max: float = 100.0

    # Additional thresholds for extended protocols
    curve_fit_r2_min: float = 0.65
    performance_retention_min: float = 78.0
    efficiency_gain_min: float = 20.0
    time_to_criterion_max: float = 250.0
    hazard_ratio_min: float = 1.30
    sample_efficiency_min: float = 33.0

    def __post_init__(self):
        """Initialize with default values."""
        for field_name, field_def in self.__dataclass_fields__.items():
            if hasattr(field_def, "default"):
                setattr(self, field_name, field_def.default)

    @classmethod
    def from_config(cls, config_manager) -> "FalsificationThresholds":
        """Load thresholds from configuration manager."""
        thresholds = cls()

        try:
            falsif_config = config_manager.get_config().falsification

            # Update core thresholds from config
            thresholds.cumulative_reward_advantage_threshold = getattr(
                falsif_config,
                "cumulative_reward_advantage_threshold",
                thresholds.cumulative_reward_advantage_threshold,
            )
            thresholds.cohens_d_threshold = getattr(
                falsif_config, "cohens_d_threshold", thresholds.cohens_d_threshold
            )
            thresholds.significance_level = getattr(
                falsif_config, "significance_level", thresholds.significance_level
            )
            thresholds.threshold_reduction_min = getattr(
                falsif_config,
                "threshold_reduction_min",
                thresholds.threshold_reduction_min,
            )
            thresholds.cohens_d_adaptation_threshold = getattr(
                falsif_config,
                "cohens_d_adaptation_threshold",
                thresholds.cohens_d_adaptation_threshold,
            )
            thresholds.tau_theta_min = getattr(
                falsif_config, "tau_theta_min", thresholds.tau_theta_min
            )
            thresholds.tau_theta_max = getattr(
                falsif_config, "tau_theta_max", thresholds.tau_theta_max
            )

            logger.info("Loaded falsification thresholds from configuration")

        except Exception as e:
            logger.error(f"Failed to load thresholds from config: {e}")
            logger.info("Using default threshold values")

        return thresholds


class ThresholdRegistry:
    """Centralized registry for APGI falsification thresholds."""

    def __init__(self, config_manager=None):
        """Initialize threshold registry."""
        self.config_manager = config_manager
        self.thresholds = FalsificationThresholds.from_config(config_manager)
        self._validate_thresholds()

    def _validate_thresholds(self):
        """Validate all threshold values."""
        validation_errors = []

        # Check core thresholds
        if self.thresholds.cumulative_reward_advantage_threshold <= 0:
            validation_errors.append(
                "cumulative_reward_advantage_threshold must be > 0"
            )
        if self.thresholds.cohens_d_threshold <= 0:
            validation_errors.append("cohens_d_threshold must be > 0")
        if (
            self.thresholds.significance_level <= 0
            or self.thresholds.significance_level >= 1
        ):
            validation_errors.append("significance_level must be between 0 and 1")
        if self.thresholds.threshold_reduction_min <= 0:
            validation_errors.append("threshold_reduction_min must be > 0")
        if self.thresholds.cohens_d_adaptation_threshold <= 0:
            validation_errors.append("cohens_d_adaptation_threshold must be > 0")
        if self.thresholds.tau_theta_min <= 0:
            validation_errors.append("tau_theta_min must be > 0")
        if self.thresholds.tau_theta_max <= 0:
            validation_errors.append("tau_theta_max must be > 0")

        # Check extended thresholds
        if self.thresholds.curve_fit_r2_min <= 0:
            validation_errors.append("curve_fit_r2_min must be > 0")
        if self.thresholds.performance_retention_min <= 0:
            validation_errors.append("performance_retention_min must be > 0")
        if self.thresholds.efficiency_gain_min <= 0:
            validation_errors.append("efficiency_gain_min must be > 0")
        if self.thresholds.time_to_criterion_max <= 0:
            validation_errors.append("time_to_criterion_max must be > 0")
        if self.thresholds.hazard_ratio_min <= 0:
            validation_errors.append("hazard_ratio_min must be > 0")
        if self.thresholds.sample_efficiency_min <= 0:
            validation_errors.append("sample_efficiency_min must be > 0")

        if validation_errors:
            error_msg = f"Threshold validation errors: {', '.join(validation_errors)}"
            raise ValueError(error_msg)

        logger.info("All threshold values validated successfully")

    def get_falsification_thresholds(self) -> FalsificationThresholds:
        """Get the current falsification thresholds."""
        return self.thresholds

    def get_threshold(self, name: str) -> float:
        """Get a specific threshold by name."""
        if not hasattr(self.thresholds, name):
            raise ValueError(f"Unknown threshold: {name}")

        value = getattr(self.thresholds, name)
        if value <= 0:
            raise ValueError(f"Threshold {name} must be > 0, got {value}")

        return value

    def update_threshold(self, name: str, value: float) -> bool:
        """Update a specific threshold value."""
        if not hasattr(self.thresholds, name):
            raise ValueError(f"Unknown threshold: {name}")

        if value <= 0:
            raise ValueError(f"Threshold {name} must be > 0, got {value}")

        # Validate against bounds if available
        try:
            bounds = ParameterBounds()
            min_attr = f"{name}_min"
            max_attr = f"{name}_max"
            if hasattr(bounds, min_attr) and hasattr(bounds, max_attr):
                min_val = getattr(bounds, min_attr)
                max_val = getattr(bounds, max_attr)
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Threshold {name} must be between {min_val} and {max_val}, got {value}"
                    )
        except AttributeError:
            # Bounds not available, skip validation
            pass

        setattr(self.thresholds, name, value)
        logger.info(f"Updated threshold {name} to {value}")
        return True

    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all thresholds as a dictionary."""
        return {
            "cumulative_reward_advantage_threshold": self.thresholds.cumulative_reward_advantage_threshold,
            "cohens_d_threshold": self.thresholds.cohens_d_threshold,
            "significance_level": self.thresholds.significance_level,
            "threshold_reduction_min": self.thresholds.threshold_reduction_min,
            "cohens_d_adaptation_threshold": self.thresholds.cohens_d_adaptation_threshold,
            "tau_theta_min": self.thresholds.tau_theta_min,
            "tau_theta_max": self.thresholds.tau_theta_max,
            "curve_fit_r2_min": self.thresholds.curve_fit_r2_min,
            "performance_retention_min": self.thresholds.performance_retention_min,
            "efficiency_gain_min": self.thresholds.efficiency_gain_min,
            "time_to_criterion_max": self.thresholds.time_to_criterion_max,
            "hazard_ratio_min": self.thresholds.hazard_ratio_min,
            "sample_efficiency_min": self.thresholds.sample_efficiency_min,
        }

    def reset_to_defaults(self):
        """Reset all thresholds to default values."""
        self.thresholds = FalsificationThresholds()
        logger.info("Reset all thresholds to default values")
