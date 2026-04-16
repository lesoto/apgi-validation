"""
Centralized APGI Configuration

This module contains the single source of truth for APGI configuration parameters.
All other modules should import from here to avoid configuration divergence.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from .config_schema import APGISystemConfig

# Path to the default configuration file
DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent / "config" / "default_apgi_config.yaml"
)


class APGIConfig:
    """
    Centralized configuration for APGI network hyperparameters.
    Loads from declarative YAML schema with Pydantic enforcement.
    """

    # Network architecture
    input_size: int = 64
    hidden_size: int = 128
    num_levels: int = 3

    # Temporal dynamics
    dt_ms: float = 10.0
    max_window_ms: float = 500.0

    # Threshold parameters
    theta0: float = 0.03
    gamma: float = 0.1
    delta: float = 0.5
    lambda_urg: float = 0.2
    theta_min: float = 0.01
    theta_max: float = 5.0

    # Learning rates
    lr_extero: float = 0.01
    lr_intero: float = 0.01
    lr_somatic: float = 0.1
    lr_policy: float = 0.001
    lr_value: float = 0.001
    lr_precision: float = 0.05

    # Ignition parameters
    alpha_ignition: float = 8.0
    tau_S: float = 0.3
    tau_theta: float = 10.0
    eta_theta: float = 0.01

    # Initial values
    theta_init: float = 0.5
    theta_baseline: float = 0.5
    Pi_e_init: float = 1.0
    Pi_i_init: float = 1.5
    beta_somatic: float = 1.8

    # Stability
    eps: float = 1e-8

    # Refractory period
    max_refractory_ms: float = 50.0

    # Precision parameters
    precision_learning_rate: float = 0.05
    precision_min: float = 0.1
    precision_max: float = 10.0
    precision_history_max: int = 100

    # Neuromodulation (acetylcholine)
    neuromod_ach_baseline: float = 0.5
    neuromod_ach_scaling: float = 1.0

    # Energy dynamics
    energy_depletion_rate: float = 0.1
    energy_min: float = 0.0
    energy_max: float = 1.0

    # Allostatic regulation
    allostatic_increase_rate: float = 0.05
    allostatic_decrease_rate: float = 0.1
    allostatic_min: float = 0.0
    allostatic_max: float = 1.0

    # Monitoring flags
    gradient_monitoring_enabled: bool = True
    performance_tracking_enabled: bool = True
    cost_benefit_gating_enabled: bool = True

    # Physical temperature (thermodynamics)
    use_physical_temperature: bool = True
    boltzmann_constant: float = 1.380649e-23  # J/K
    temperature_kelvin: float = 310.0  # Body temperature
    temperature_normalized: float = 1.0
    energy_scale_factor: float = 1e21  # Scale J to neural units
    entropy_scale_factor: float = 1.0

    # Rigorous thermodynamics
    use_rigorous_thermodynamic_entropy: bool = True

    # Additional entropy/thermodynamics settings
    use_shannon_entropy: bool = True
    use_rigorous_variational_fe: bool = False
    cross_level_validation_enabled: bool = True
    cost_benefit_clamp_min: float = -10.0
    cost_benefit_clamp_max: float = 10.0
    cost_benefit_scaling: float = 1.0

    # Time constants
    tau_min: float = 0.01
    tau_max: float = 100.0
    tau_intero_baseline: float = 0.3
    tau_extero_baseline: float = 0.2

    # Refractory cost
    refractory_cost_baseline: float = 0.1
    refractory_cost_scaling: float = 1.0

    # Additional learning parameters
    alpha_broadcast: float = 1.0
    beta_maintenance: float = 0.5
    beta_transition: float = 0.8
    hysteresis: float = 0.1

    # History and memory
    volatility_history_max: int = 100
    entropy_calculation_interval: int = 10

    # Additional neuromodulation
    workspace_sustained_scaling: float = 1.0

    # Additional neural parameters
    gradient_warn_threshold: float = 1000.0
    gradient_clip_value: float = 1.0
    reservoir_scaling: float = 1.0
    reservoir_sparsity: float = 0.5

    def __init__(self, **kwargs):
        """Allow instantiation with keyword arguments for testing"""
        for key, value in kwargs.items():
            if hasattr(self.__class__, key):
                setattr(self, key, value)

    @classmethod
    def load_config(cls, path: Optional[Path] = None):
        """Load configuration from file and update class attributes"""
        if path is None:
            config_env = os.environ.get("APGI_CONFIG_PATH")
            path = Path(config_env) if config_env else DEFAULT_CONFIG_PATH

        if not path.exists():
            logging.warning(f"Config file not found: {path}. Using defaults.")
            return

        try:
            full_config = APGISystemConfig.load(path)

            # Update class attributes from nested models
            # Network
            cls.input_size = full_config.network.input_size
            cls.hidden_size = full_config.network.hidden_size
            cls.num_levels = full_config.network.num_levels

            # Temporal
            cls.dt_ms = full_config.temporal.dt_ms
            cls.max_window_ms = full_config.temporal.max_window_ms

            # Thresholds
            cls.theta0 = full_config.thresholds.theta0
            cls.gamma = full_config.thresholds.gamma
            cls.delta = full_config.thresholds.delta
            cls.lambda_urg = full_config.thresholds.lambda_urg
            cls.theta_min = full_config.thresholds.theta_min
            cls.theta_max = full_config.thresholds.theta_max

            # Ignition
            cls.alpha_ignition = full_config.ignition.alpha
            cls.tau_S = full_config.ignition.tau_S
            cls.tau_theta = full_config.ignition.tau_theta
            cls.eta_theta = full_config.ignition.eta_theta
            cls.theta_init = full_config.ignition.theta_init
            cls.theta_baseline = full_config.ignition.theta_baseline

            # Learning rates
            cls.lr_extero = full_config.learning_rates.get("extero", cls.lr_extero)
            cls.lr_intero = full_config.learning_rates.get("intero", cls.lr_intero)
            cls.lr_somatic = full_config.learning_rates.get("somatic", cls.lr_somatic)
            cls.lr_policy = full_config.learning_rates.get("policy", cls.lr_policy)
            cls.lr_value = full_config.learning_rates.get("value", cls.lr_value)
            cls.lr_precision = full_config.learning_rates.get(
                "precision", cls.lr_precision
            )

            logging.info(f"Successfully loaded declarative config from {path}")

        except Exception as e:
            logging.error(f"Failed to load config from {path}: {e}")


# Automatically load default config on import
APGIConfig.load_config()
