#!/usr/bin/env python3
"""
APGI Framework Constants
========================

Centralized constants for the APGI framework to avoid magic numbers
and ensure consistency across all modules.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelParameters:
    """Default model parameters for APGI simulations."""

    tau_S: float = 0.5
    tau_theta: float = 30.0
    theta_0: float = 0.0
    alpha: float = 0.1
    gamma_M: float = 0.5
    gamma_A: float = 0.3
    rho: float = 0.8
    sigma_S: float = 0.1
    sigma_theta: float = 0.05


@dataclass
class NeuralDataDefaults:
    """Default values for neural data processing."""

    sampling_rate_hz: float = 1000.0
    eeg_channels: int = 64
    fmri_tr_seconds: float = 2.0
    default_window_size: int = 1000
    default_overlap: float = 0.5


@dataclass
class SpeciesMetrics:
    """Species-specific metrics for cross-species scaling."""

    human: Dict = None
    monkey: Dict = None
    cat: Dict = None
    rat: Dict = None
    mouse: Dict = None

    def __post_init__(self):
        if self.human is None:
            self.human = {
                "name": "human",
                "cortical_volume_mm3": 120000,
                "cortical_thickness_mm": 3.0,
                "neuron_density_per_mm3": 25000,
                "synaptic_density_per_mm3": 500000,
                "conduction_velocity_m_s": 50.0,
                "body_mass_kg": 70.0,
                "brain_mass_g": 1400.0,
            }
        if self.monkey is None:
            self.monkey = {
                "name": "monkey",
                "cortical_volume_mm3": 80000,
                "cortical_thickness_mm": 2.5,
                "neuron_density_per_mm3": 30000,
                "synaptic_density_per_mm3": 600000,
                "conduction_velocity_m_s": 45.0,
                "body_mass_kg": 8.0,
                "brain_mass_g": 100.0,
            }
        if self.cat is None:
            self.cat = {
                "name": "cat",
                "cortical_volume_mm3": 15000,
                "cortical_thickness_mm": 2.0,
                "neuron_density_per_mm3": 35000,
                "synaptic_density_per_mm3": 700000,
                "conduction_velocity_m_s": 40.0,
                "body_mass_kg": 4.0,
                "brain_mass_g": 30.0,
            }
        if self.rat is None:
            self.rat = {
                "name": "rat",
                "cortical_volume_mm3": 500,
                "cortical_thickness_mm": 1.5,
                "neuron_density_per_mm3": 40000,
                "synaptic_density_per_mm3": 800000,
                "conduction_velocity_m_s": 35.0,
                "body_mass_kg": 0.3,
                "brain_mass_g": 2.5,
            }
        if self.mouse is None:
            self.mouse = {
                "name": "mouse",
                "cortical_volume_mm3": 100,
                "cortical_thickness_mm": 1.0,
                "neuron_density_per_mm3": 45000,
                "synaptic_density_per_mm3": 900000,
                "conduction_velocity_m_s": 30.0,
                "body_mass_kg": 0.02,
                "brain_mass_g": 0.4,
            }


@dataclass
class SystemDefaults:
    """System-wide default values."""

    thread_pool_size: int = 4
    plot_dpi: int = 100
    timeout_seconds: float = 300.0
    max_cache_size_mb: int = 1024
    batch_size: int = 1000
    validation_timeout_seconds: float = 300.0


@dataclass
class ParameterBounds:
    """Canonical parameter bounds for APGI models."""

    precision_min: float = 0.1
    precision_max: float = 10.0
    tau_min: float = 0.01
    tau_max: float = 100.0
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    gamma_min: float = 0.0
    gamma_max: float = 1.0
    rho_min: float = 0.0
    rho_max: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 1.0


@dataclass
class PCINormalization:
    """PCI normalization constants."""

    normalization_constant: float = 2.5


@dataclass
class DimensionConstants:
    """Dimension constants for APGI models."""

    # Exteroceptive pathway dimensions
    EXTERO_DIM: int = 32
    SENSORY_DIM: int = 32
    OBJECTS_DIM: int = 16
    CONTEXT_DIM: int = 8

    # Interoceptive pathway dimensions
    INTERO_DIM: int = 16
    VISCERAL_DIM: int = 16
    ORGAN_DIM: int = 8
    HOMEOSTATIC_DIM: int = 4

    # Combined dimensions
    WORKSPACE_DIM: int = 8
    STATE_DIMENSION: int = 48  # EXTERO_DIM + INTERO_DIM

    # Network dimensions
    HIDDEN_DIM_DEFAULT: int = 64
    SOMATIC_HIDDEN_DIM: int = 32

    # Action dimensions
    N_ACTIONS: int = 4
    ACTION_DIM: int = 4

    # Thresholds
    IGNITION_THRESHOLD: float = 0.5

    # Sample sizes
    MIN_SAMPLES_FOR_REGRESSION: int = 10
    MIN_BOOTSTRAP_SAMPLES: int = 100

    # Clipping values
    DEFAULT_EPSILON: float = 1e-8
    MAX_CLIP_VALUE: float = 10.0
    GRAD_CLIP_VALUE: float = 1.0
    WEIGHT_CLIP_VALUE: float = 2.0
    POLICY_GRAD_CLIP: float = 5.0

    # Extended dimensions for data generation (Validation Protocol 6)
    EXTERO_DIM_EXTENDED: int = 64  # 2 * EXTERO_DIM
    INTERO_DIM_EXTENDED: int = 32  # 2 * INTERO_DIM
    CONTEXT_DIM_EXTENDED: int = 8  # Same as CONTEXT_DIM


# Global instances
MODEL_PARAMS = ModelParameters()
NEURAL_DEFAULTS = NeuralDataDefaults()
SPECIES_METRICS = SpeciesMetrics()
SYSTEM_DEFAULTS = SystemDefaults()
PARAMETER_BOUNDS = ParameterBounds()
PCI_NORMALIZATION = PCINormalization()
DIM_CONSTANTS = DimensionConstants()
