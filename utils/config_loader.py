"""
APGI Theory Framework - Configuration Loader
==========================================

Simple utility for loading configuration values from config/default.yaml
for use in validation and falsification protocols.

This provides a single function that protocols can call to get
configuration values without needing to instantiate the full
ConfigManager class.

Example:
    >>> from utils.config_loader import load_config_value
    >>> tau_S = load_config_value("model.tau_S", 0.5)
    >>> threshold = load_config_value("falsification.cohens_d_threshold", 0.60)
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install with: pip install PyYAML")

# Configure logging
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "default.yaml"


def load_config_value(
    key: str, default: Any = None, config_file: Optional[Union[str, Path]] = None
) -> Any:
    """
    Load a configuration value from the default YAML config file.

    Args:
        key: Configuration key in dot notation (e.g., "model.tau_S", "falsification.cohens_d_threshold")
        default: Default value if key is not found in config
        config_file: Optional path to config file (defaults to config/default.yaml)

    Returns:
        Configuration value or default if not found

    Example:
        >>> tau_S = load_config_value("model.tau_S", 0.5)
        >>> threshold = load_config_value("falsification.cohens_d_threshold", 0.60)
    """
    config_path = Path(config_file) if config_file else DEFAULT_CONFIG_FILE

    try:
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return default

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            logger.warning(f"Config file is empty: {config_path}")
            return default

        # Navigate through the key using dot notation
        keys = key.split(".")
        current = config_data

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                logger.warning(f"Configuration key not found: {key}")
                return default

        logger.debug(f"Loaded config value {key} = {current}")
        return current

    except Exception as e:
        logger.error(f"Error loading config value {key}: {e}")
        return default


def load_model_config() -> dict:
    """
    Load all model configuration values as a dictionary.

    Returns:
        Dictionary containing all model configuration values
    """
    try:
        with open(DEFAULT_CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return config_data.get("model", {})
    except Exception as e:
        logger.error(f"Error loading model config: {e}")
        return {}


def load_falsification_config() -> dict:
    """
    Load all falsification configuration values as a dictionary.

    Returns:
        Dictionary containing all falsification configuration values
    """
    try:
        with open(DEFAULT_CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return config_data.get("falsification", {})
    except Exception as e:
        logger.error(f"Error loading falsification config: {e}")
        return {}


def load_network_config() -> dict:
    """
    Load all network configuration values as a dictionary.

    Returns:
        Dictionary containing all network configuration values
    """
    try:
        with open(DEFAULT_CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return config_data.get("network", {})
    except Exception as e:
        logger.error(f"Error loading network config: {e}")
        return {}


def get_config_section(section: str) -> dict:
    """
    Load a specific configuration section.

    Args:
        section: Section name (e.g., "model", "falsification", "network")

    Returns:
        Dictionary containing the requested section
    """
    return load_config_value(section, {})


# Convenience functions for commonly used values
def get_tau_S(default: float = 0.5) -> float:
    """Get tau_S configuration value."""
    return load_config_value("model.tau_S", default)


def get_tau_theta(default: float = 30.0) -> float:
    """Get tau_theta configuration value."""
    return load_config_value("model.tau_theta", default)


def get_theta_0(default: float = 0.5) -> float:
    """Get theta_0 configuration value."""
    return load_config_value("model.theta_0", default)


def get_alpha(default: float = 5.0) -> float:
    """Get alpha configuration value."""
    return load_config_value("model.alpha", default)


def get_gamma_M(default: float = 0.1) -> float:
    """Get gamma_M configuration value."""
    return load_config_value("model.gamma_M", default)


def get_gamma_A(default: float = 0.05) -> float:
    """Get gamma_A configuration value."""
    return load_config_value("model.gamma_A", default)


def get_rho(default: float = 0.7) -> float:
    """Get rho configuration value."""
    return load_config_value("model.rho", default)


def get_sigma_S(default: float = 0.1) -> float:
    """Get sigma_S configuration value."""
    return load_config_value("model.sigma_S", default)


def get_sigma_theta(default: float = 0.05) -> float:
    """Get sigma_theta configuration value."""
    return load_config_value("model.sigma_theta", default)


# Falsification threshold convenience functions
def get_cumulative_reward_advantage_threshold(default: float = 18.0) -> float:
    """Get cumulative reward advantage threshold."""
    return load_config_value(
        "falsification.cumulative_reward_advantage_threshold", default
    )


def get_cohens_d_threshold(default: float = 0.60) -> float:
    """Get Cohen's d threshold."""
    return load_config_value("falsification.cohens_d_threshold", default)


def get_significance_level(default: float = 0.01) -> float:
    """Get significance level."""
    return load_config_value("falsification.significance_level", default)


def get_threshold_reduction_min(default: float = 20.0) -> float:
    """Get minimum threshold reduction."""
    return load_config_value("falsification.threshold_reduction_min", default)


def get_cohens_d_adaptation_threshold(default: float = 0.70) -> float:
    """Get Cohen's d adaptation threshold."""
    return load_config_value("falsification.cohens_d_adaptation_threshold", default)


def get_tau_theta_min(default: float = 10.0) -> float:
    """Get minimum tau_theta."""
    return load_config_value("falsification.tau_theta_min", default)


def get_tau_theta_max(default: float = 100.0) -> float:
    """Get maximum tau_theta."""
    return load_config_value("falsification.tau_theta_max", default)
