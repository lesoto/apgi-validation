"""
Centralized APGI Configuration

This module contains the single source of truth for APGI configuration parameters.
All other modules should import from here to avoid configuration divergence.
"""

from dataclasses import dataclass


@dataclass
class APGIConfig:
    """
    Centralized configuration for APGI network hyperparameters.

    All magic numbers are parameterized here for easy experimentation
    and hyperparameter tuning.
    """

    # Network architecture
    input_size: int = 64
    hidden_size: int = 128
    num_levels: int = 3

    # Temporal dynamics
    dt_ms: float = 10.0  # Time step in milliseconds
    max_window_ms: float = 500.0  # Temporal integration window

    # Threshold parameters
    theta0: float = 0.03  # Baseline threshold (lowered for ignition)
    gamma: float = 0.1  # Homeostatic rate
    delta: float = 0.5  # Refractoriness strength
    lambda_urg: float = 0.2  # Urgency scaling
    theta_min: float = 0.01  # Minimum threshold
    theta_max: float = 5.0  # Maximum threshold

    # Phase transition parameters
    beta_transition: float = 10.0  # Steepness of phase transition
    hysteresis: float = 0.2  # Threshold difference for on/off

    # Precision parameters
    tau_min: float = 0.01  # Minimum time constant
    tau_max: float = 10.0  # Maximum time constant
    tau_intero_baseline: float = 0.1  # Baseline intero tau
    tau_extero_baseline: float = 0.05  # Baseline extero tau
    precision_min: float = 0.01  # Minimum precision
    precision_max: float = 10.0  # Maximum precision

    # Precision learning parameters
    precision_learning_baseline: float = 0.8  # Baseline weight for learned precision
    precision_learning_range: float = 0.4  # Range for learned adjustment
    precision_learning_rate: float = 0.01  # Meta-learning rate
    precision_history_max: int = 20  # Maximum precision history length
    precision_ema_steps: int = 5  # Steps for exponential moving average
    volatility_history_max: int = 20  # Maximum volatility history length

    # Metabolic parameters
    alpha_broadcast: float = 1.0  # Broadcast cost scaling
    beta_maintenance: float = 0.5  # Maintenance cost scaling
    energy_depletion_rate: float = 0.5  # Rate of energy depletion per cost unit
    energy_min: float = 0.0  # Minimum energy reserves
    energy_max: float = 1.0  # Maximum energy reserves

    # Allostatic parameters
    allostatic_increase_rate: float = 0.01  # Rate surprise increases allostatic load
    allostatic_decrease_rate: float = 0.5  # Rate ignition decreases allostatic load
    allostatic_min: float = 0.0  # Minimum allostatic load
    allostatic_max: float = 2.0  # Maximum allostatic load

    # Cost-benefit gating
    cost_benefit_gating_enabled: bool = True  # Enable cost-benefit threshold modulation
    cost_benefit_scaling: float = 0.1  # Scaling factor for cost-benefit adjustment
    cost_benefit_clamp_min: float = -1.0  # Minimum cost-benefit adjustment
    cost_benefit_clamp_max: float = 1.0  # Maximum cost-benefit adjustment

    # Refractory period
    max_refractory_ms: float = 200.0  # Maximum refractory period
    refractory_cost_baseline: float = 0.5  # Baseline refractory scaling
    refractory_cost_scaling: float = 0.5  # Cost-dependent refractory scaling

    # Reservoir computing
    reservoir_sparsity: float = 0.9  # Sparsity of recurrent connections
    reservoir_scaling: float = 0.1  # Scaling for recurrent weights

    # Neuromodulation
    neuromod_da_baseline: float = 0.5  # Baseline dopamine modulation
    neuromod_da_scaling: float = 0.5  # Dopamine scaling factor
    neuromod_ne_baseline: float = 0.5  # Baseline norepinephrine modulation
    neuromod_ne_scaling: float = 0.5  # Norepinephrine scaling factor
    neuromod_ach_baseline: float = 0.5  # Baseline ACh modulation
    neuromod_ach_scaling: float = 0.5  # ACh scaling factor

    # Entropy calculation interval
    entropy_calculation_interval: int = 10  # Steps between entropy calculations

    # Workspace dynamics
    workspace_sustained_scaling: float = 0.1  # Scaling for sustained activity

    # Gradient monitoring
    gradient_monitoring_enabled: bool = True  # Enable gradient monitoring
    gradient_clip_value: float = 100.0  # Gradient clipping threshold
    gradient_warn_threshold: float = 10.0  # Warning threshold for gradients

    # Performance tracking
    performance_tracking_enabled: bool = False  # Enable performance benchmarking

    # Cross-level validation
    cross_level_validation_enabled: bool = True  # Enable cross-level consistency checks

    # F6.2: Intrinsic Temporal Integration parameters
    F6_2_MIN_INTEGRATION_RATIO: float = 4.0  # Minimum LTCN/RNN integration ratio
    F6_2_MIN_CURVE_FIT_R2: float = 0.70  # Minimum curve fit R²
    F6_2_WILCOXON_ALPHA: float = 0.01  # Wilcoxon signed-rank test alpha
    F6_2_LTCN_MIN_WINDOW_MS: float = 200.0  # Minimum LTCN window in milliseconds

    # Numerical stability
    eps: float = 1e-8  # Small epsilon for numerical stability

    # Thermodynamic parameters
    use_physical_temperature: bool = True  # Enable physical temperature
    boltzmann_constant: float = 1.38e-23  # Boltzmann constant in J/K
    temperature_kelvin: float = 310.0  # Body temperature in Kelvin
    temperature_normalized: float = 1.0  # Normalized temperature (dimensionless)
    energy_scale_factor: float = 1.0  # Energy scale factor for physical units
    entropy_scale_factor: float = 1.0  # Entropy scale factor for physical units
    use_rigorous_thermodynamic_entropy: bool = (
        True  # Use rigorous thermodynamic entropy
    )
    use_shannon_entropy: bool = True  # Use Shannon entropy
    use_rigorous_variational_fe: bool = True  # Use rigorous variational free energy

    # VP-03: Agent Simulation Hyperparameters (HIGH-06)
    # Learning rates for hierarchical generative models
    lr_extero: float = 0.01  # Exteroceptive model learning rate
    lr_intero: float = 0.01  # Interoceptive model learning rate
    lr_somatic: float = 0.1  # Somatic policy network learning rate
    lr_policy: float = 0.001  # Policy network baseline learning rate
    lr_value: float = 0.001  # Value network learning rate
    lr_precision: float = 0.05  # Precision learning rate

    # Ignition parameters
    alpha_ignition: float = 8.0  # Steepness of ignition sigmoid
    tau_S: float = 0.3  # Surprise decay time constant
    tau_theta: float = 10.0  # Threshold adaptation time constant
    eta_theta: float = 0.01  # Threshold learning rate

    # Initial parameter values
    theta_init: float = 0.5  # Initial ignition threshold
    theta_baseline: float = 0.5  # Baseline ignition threshold
    Pi_e_init: float = 1.0  # Initial exteroceptive precision
    Pi_i_init: float = 1.5  # Initial interoceptive precision (calibrated higher)
    beta_somatic: float = 1.8  # Somatic bias strength (calibrated)

    # Workspace parameters
    workspace_update_rate: float = 0.1  # Rate of workspace content update

    def __post_init__(self):
        """Validate configuration parameters"""
        if not self.input_size > 0:
            raise ValueError("input_size must be positive")
        if not self.hidden_size > 0:
            raise ValueError("hidden_size must be positive")
        if not self.num_levels > 0:
            raise ValueError("num_levels must be positive")
        if not self.dt_ms > 0:
            raise ValueError("dt_ms must be positive")
        if not self.max_window_ms > 0:
            raise ValueError("max_window_ms must be positive")
        if not self.theta0 > 0:
            raise ValueError("theta0 must be positive")
        if not self.gamma > 0:
            raise ValueError("gamma must be positive")
        if not 0 <= self.delta <= 1:
            raise ValueError("delta must be in [0, 1]")
        if not 0 <= self.lambda_urg <= 1:
            raise ValueError("lambda_urg must be in [0, 1]")
        if not 0 < self.theta_min < self.theta_max:
            raise ValueError("theta_min must be < theta_max and both positive")
        if not 0 < self.reservoir_sparsity < 1:
            raise ValueError("reservoir_sparsity must be in (0, 1)")
        if not self.eps > 0:
            raise ValueError("eps must be positive")
