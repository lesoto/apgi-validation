#!/usr/bin/env python3
"""
APGI Framework Constants
========================

Centralized constants for the APGI framework to avoid magic numbers
and ensure consistency across all modules.

Arousal coupling defaults follow the behavioral validation calibration used in
VP-02 and are anchored to interoceptive gain estimates discussed in:
Critchley HD, Wiens S, Rotshtein P, Ohman A, Dolan RJ. Neural systems
supporting interoceptive awareness. Nat Neurosci. 2004;7(2):189-195.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

try:
    from .falsification_thresholds import (
        TRANSFER_ENTROPY_THRESHOLD,
        VP4_CALIBRATED_ALPHA,
        VP4_CALIBRATED_TAU,
        VP4_CALIBRATED_THETA_0,
    )
except ImportError:
    try:
        from utils.falsification_thresholds import (
            TRANSFER_ENTROPY_THRESHOLD,
            VP4_CALIBRATED_ALPHA,
            VP4_CALIBRATED_TAU,
            VP4_CALIBRATED_THETA_0,
        )
    except ImportError:
        # Use fallback values if imports fail
        TRANSFER_ENTROPY_THRESHOLD = 0.05
        VP4_CALIBRATED_ALPHA = 0.3
        VP4_CALIBRATED_TAU = 20.0
        VP4_CALIBRATED_THETA_0 = 0.05

# VP-02 physiological arousal coupling defaults (Critchley et al. 2004-inspired)
ALPHA_AROUSAL: float = 0.15
SIGMA_AROUSAL: float = 2.5

# VP-14 fMRI HRF parameters (SPM canonical double-gamma HRF)
# Friston et al. 1998, NeuroImage 5:S66 - canonical hemodynamic response function
HRF_PEAK1_SECONDS: float = 6.0  # Peak response delay (main gamma)
HRF_UNDERSHOOT_SECONDS: float = 16.0  # Undershoot peak delay
HRF_DISPERSION: float = 1.0  # Dispersion parameter for both peaks
HRF_UNDERSHOOT_RATIO: float = 1.0 / 6.0  # Undershoot magnitude ratio

# BOLD detectability threshold for 3T fMRI
# Empirical threshold: tSNR >= 20-30 required for reliable BOLD signal detection
# Murphy et al. 2007, NeuroImage 37:912-920; Welvaert et al. 2013
BOLD_TSNR_MIN: float = 20.0

# VP-10 TMS defaults (Rossini et al. 2015 safety/technical guidance-inspired)
TMS_PULSE_WIDTH_MS: float = 0.3
TMS_MOTOR_THRESHOLD_ADJUST: float = 0.8
TMS_SIGMOID_STEEPNESS: float = 5.0


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
class ThermodynamicConfig:
    """Shared thermodynamic configuration for FP-04 and VP-04 consistency"""

    def __init__(self):
        self.tau = VP4_CALIBRATED_TAU
        self.theta_0 = VP4_CALIBRATED_THETA_0
        self.alpha = VP4_CALIBRATED_ALPHA
        self.transfer_entropy_threshold = TRANSFER_ENTROPY_THRESHOLD


# Global singleton instance for shared thermodynamic configuration
DEFAULT_THERMO_CONFIG = ThermodynamicConfig()


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
    validation_timeout_seconds: float = 600.0


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
class LevelTimescales:
    """
    Level-specific time constants (tau) for hierarchical generative models.

    VALIDATED: Document specifies τ ranges from 0.05s (sensory) to 2.0s (homeostatic).
    Source: APGI falsification criteria documentation, Innovations-Software-Implementation.md
    """

    # Level 1: Sensory (fastest timescale)
    TAU_SENSORY: float = 0.05  # seconds
    # Level 2: Organ/visceral
    TAU_ORGAN: float = 0.2  # seconds
    # Level 3: Cognitive/context
    TAU_COGNITIVE: float = 0.5  # seconds
    # Level 4: Homeostatic (slowest timescale)
    TAU_HOMEOSTATIC: float = 2.0  # seconds

    # Mapping by level number (1-indexed)
    LEVEL_TIMESCALES: Dict = None

    def __post_init__(self):
        if self.LEVEL_TIMESCALES is None:
            self.LEVEL_TIMESCALES = {
                1: self.TAU_SENSORY,
                2: self.TAU_ORGAN,
                3: self.TAU_COGNITIVE,
                4: self.TAU_HOMEOSTATIC,
            }


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
    IGNITION_THRESHOLD: float = 0.8

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


# F4 - Phase Transition & Epistemic Architecture thresholds
F4_CRITICAL_SLOWING_MIN_RATIO: float = 1.2  # 20% increase threshold for τ_auto
F4_CRITICAL_SLOWING_P_VALUE: float = 0.05  # p < 0.05 for surrogate test
F4_TE_THRESHOLD: float = 0.1  # Transfer entropy threshold
F4_PHI_MIN_BITS: float = 0.5  # Minimum integrated information (phi_proxy)
F4_PHI_SIGNIFICANT_BITS: float = 1.0  # Significant phi_proxy threshold effect size
F4_MIN_SENSITIVITY: float = 0.70  # Minimum sensitivity for clinical biomarkers
F4_MIN_SPECIFICITY: float = 0.70  # Minimum specificity for clinical biomarkers
F4_MIN_POWER: float = 0.80  # Minimum statistical power

# Global instances
MODEL_PARAMS = ModelParameters()
NEURAL_DEFAULTS = NeuralDataDefaults()
SPECIES_METRICS = SpeciesMetrics()
SYSTEM_DEFAULTS = SystemDefaults()
PARAMETER_BOUNDS = ParameterBounds()
PCI_NORMALIZATION = PCINormalization()
DIM_CONSTANTS = DimensionConstants()
LEVEL_TIMESCALES = LevelTimescales()

# Global random seed for reproducibility across all protocols
APGI_GLOBAL_SEED = 42

# Shuffle seed offset for cross-validation and resampling operations
SHUFFLE_SEED_OFFSET = 1000

# HIGH-02: FP-04 DoC Synthetic Classification Feature Weights
# Feature weights for biomarker-based DoC (Disorders of Consciousness) classification
# Used in FP_04_PhaseTransition_EpistemicArchitecture.py for ROC analysis
FP3_DOC_SYNTHETIC_FEATURE_WEIGHTS = np.array(
    [1.2, 0.8, 1.0, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
)

# Signal multipliers for APGI-specific biomarker signals in DoC classification
# These multipliers are applied to DoC samples during synthetic data generation
# Order: [entropy, temporal_pattern, integration_complexity, threshold_modulation]
FP3_DOC_SIGNAL_MULTIPLIERS = {
    "entropy": 1.2,
    "temporal_pattern": 0.8,
    "integration_complexity": 1.0,
    "threshold_modulation": 0.6,
}

# EEG frequency band definitions (Hz)
# Based on standard 10-20 international EEG electrode placement system
# Source: Jurcak, V. et al. (2007). International 10-20 system of EEG
# and standardized electrode placement. Revue d'Electroencéphalogie et
# Neurophysiologie Clinique 170(4):241-257.
EEG_DELTA_BAND_HZ = (0.5, 4.0)  # Delta band: 0.5-4 Hz
EEG_THETA_BAND_HZ = (4.0, 8.0)  # Theta band: 4-8 Hz
EEG_ALPHA_BAND_HZ = (8.0, 12.0)  # Alpha band: 8-12 Hz
EEG_BETA_BAND_HZ = (12.0, 30.0)  # Beta band: 12-30 Hz
EEG_GAMMA_BAND_HZ = (30.0, 80.0)  # Gamma band: 30-100 Hz

# Centralized citation dictionary for scientific references
# Used across validation protocols to ensure consistent citation management
CITATIONS = {
    "P3b_baseline": "Polich J. (2007). Updating P300: An integrative theory. Clin Neurophysiol, 118(10):2128-2148.",
    "HEP_baseline": "Nummenmaa L. et al. (2013). Bodily maps of emotions. PNAS, 111(2):646-651.",
}

# =============================================================================
# ECHO STATE NETWORK (ESN) PARAMETERS
# =============================================================================
# ESN hyperparameters control the echo state property and temporal integration.
# These values are derived from reservoir computing theory (Jaeger 2001) and
# calibrated for the critical edge-of-chaos regime where computational
# performance is maximized.
#
# Spectral radius controls the reservoir's memory capacity:
#   - ρ < 1.0: fading memory (echo state property guaranteed)
#   - ρ ≈ 1.0: critical edge-of-chaos (maximal computational power)
#   - ρ > 1.0: potential instability
#
# Leak rate controls the integration time constant:
#   - Small α: slow integration, longer memory
#   - Large α: fast integration, shorter memory
#   - α = 0.01 provides ~100ms effective time constant at 1000Hz sampling
#
# References:
#   Jaeger, H. (2001). The "echo state" approach to analysing and training
#   recurrent neural networks. GMD Report 148, German National Research Center
#   for Information Technology.
#
#   Lukoševičius, M., & Jaeger, H. (2009). Reservoir computing approaches to
#   recurrent neural network training. Computer Science Review, 3(3), 127-149.
# =============================================================================
ESN_SPECTRAL_RADIUS: float = 0.98  # Critical edge-of-chaos regime (ρ ≈ 1.0)
ESN_LEAK_RATE: float = 0.01  # Slow integration for 100ms effective window

# ESN sensitivity analysis parameter ranges
# Used for robustness testing across hyperparameter space
ESN_SPECTRAL_RADIUS_RANGE: tuple = (0.90, 0.95, 0.98, 1.00)
ESN_LEAK_RATE_RANGE: tuple = (0.01, 0.05, 0.10)
