"""
=============================================================================
APGI APGI Full Dynamic Model Implementation
=============================================================================

Core Ignition Criterion:

    (Πe · |εe| + β_som · Πi · |εi|) > θt

Mathematical Formulation:

    1. Signal Accumulation: S(t+Δt) = exp(−Δt/τ)·S(t) + we·Πe(t)·|εe(t)| + wi·β_som(t)·Πi(t)·|εi(t)|
    ...
        S(t+Δt) = exp(−Δt/τ)·S(t) + we·Πe(t)·|εe(t)| + wi·β_som(t)·Πi(t)·|εi(t)|
    2. Threshold Dynamics: θt(t+Δt) = θt0 + ηm(t) + α·[θt(t) − (θt0 + ηm(t))] + φ·I(t)
    3. Ignition Probability: P(ignition|t) = 1 / (1 + exp(−k·(S(t)−θt(t))))
    4. Metabolic State: ηm(t+Δt) = ηm(t) + γc·I(t) − γr·(1−I(t))
    5. Signal Standardization: εnormalized = (εraw−µbaseline)/σbaseline

    where:
    - Δt: Timestep duration (typically 50-100ms)
    - τ: Signal decay time constant
    - φ: Post-ignition facilitation (positive increment)

=============================================================================
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class APGIParameters:
    """Parameters for APGI Full Dynamic Model with constraints."""

    # Temporal parameters
    delta_t: float = (
        0.05  # Δt: Timestep duration (seconds), typical 0.05-0.1 (50-100ms)
    )
    tau: float = 0.2  # τ: Signal decay time constant (seconds), typical 0.1-0.3

    # Signal accumulation parameters
    we: float = 0.5  # External domain weight (we + wi = 1)
    wi: float = 0.5  # Internal domain weight (we + wi = 1)

    # Threshold dynamics parameters
    theta_t0: float = 3.0  # Baseline threshold [1.0, 10.0], typical 2.0-4.0σ
    alpha: float = 0.9  # Threshold recovery rate (0,1), typical 0.8-0.95
    phi: float = 0.4  # Post-ignition facilitation [0.0, 1.0], typical 0.3-0.5σ

    # Metabolic parameters
    gamma_c: float = 0.2  # Metabolic cost per ignition [0.05, 0.5], typical 0.1-0.3σ
    gamma_r: float = 0.07  # Metabolic recovery rate [0.01, 0.2], typical 0.05-0.1σ

    # Sigmoid parameters
    k: float = 3.0  # Sigmoid steepness [1.0, 10.0], typical 2.0-5.0

    # Somatic bias
    beta: float = 1.0  # Somatic bias [0.0, 5.0], typical 0.5-2.0

    def __post_init__(self):
        """Validate parameter constraints."""
        if not (0.01 <= self.delta_t <= 0.2):
            raise ValueError(f"delta_t must be in [0.01, 0.2], got {self.delta_t}")
        if not (0.05 <= self.tau <= 1.0):
            raise ValueError(f"tau must be in [0.05, 1.0], got {self.tau}")
        if not np.isclose(self.we + self.wi, 1.0):
            raise ValueError(f"we + wi must equal 1.0, got {self.we + self.wi}")
        if not (1.0 <= self.theta_t0 <= 10.0):
            raise ValueError(f"theta_t0 must be in [1.0, 10.0], got {self.theta_t0}")
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha must be in (0,1), got {self.alpha}")
        if not (0.0 <= self.phi <= 1.0):
            raise ValueError(f"phi must be in [0.0, 1.0], got {self.phi}")
        if not (0.05 <= self.gamma_c <= 0.5):
            raise ValueError(f"gamma_c must be in [0.05, 0.5], got {self.gamma_c}")
        if not (0.01 <= self.gamma_r <= 0.2):
            raise ValueError(f"gamma_r must be in [0.01, 0.2], got {self.gamma_r}")
        if not (1.0 <= self.k <= 10.0):
            raise ValueError(f"k must be in [1.0, 10.0], got {self.k}")
        if not (0.0 <= self.beta <= 5.0):
            raise ValueError(f"beta must be in [0.0, 5.0], got {self.beta}")

    def get_circadian_modulation(self, time_of_day_hours: float) -> float:
        """
        Compute circadian modulation of threshold based on cortisol rhythm.

        Innovation #31: Cortisol-θ_t relationship formalized as inverted-U function.

        Theory:
            Δθ_circadian = -k_cort · [C(t) - C_optimal]²

            where:
            - C(t): Cortisol concentration at time t (approximated from circadian phase)
            - C_optimal ≈ 12.5 μg/dL (optimal for sustained attention)
            - k_cort: Inverted-U gain constant (default 0.08)

        Mechanism:
            - LOW cortisol (<8 μg/dL, evening): elevated θ_t (reduced ignition)
            - OPTIMAL cortisol (10-15 μg/dL, morning): minimal θ_t (enhanced access)
            - HIGH cortisol (>20 μg/dL, acute stress): elevated θ_t (narrowed focus)

        Args:
            time_of_day_hours: Time in hours (0-24), e.g., 9.5 for 9:30 AM

        Returns:
            Threshold modulation Δθ_circadian in σ units

        References:
            - Lupien et al. 2007: Morning cortisol facilitates memory encoding
            - McEwen 2007: Chronic elevation impairs hippocampal function
            - Dijk & Czeisler 2012: Evening nadir enables restorative processes
        """
        # Approximate cortisol concentration from circadian phase
        # Peak: 8-9 AM (~15 μg/dL), Nadir: 12 AM (~3 μg/dL)
        circadian_phase = (time_of_day_hours - 8.5) * 2 * np.pi / 24
        C_t = 9.0 + 6.0 * np.cos(circadian_phase)  # Range: 3-15 μg/dL

        # Optimal cortisol level
        C_optimal = 12.5  # μg/dL

        # Inverted-U gain constant
        k_cort = 0.08  # Scaled for threshold units

        # Compute threshold modulation
        delta_theta_circadian = -k_cort * (C_t - C_optimal) ** 2

        return delta_theta_circadian

    def get_ultradian_modulation(
        self, time_since_task_start_min: float, task_load: float = 0.7
    ) -> float:
        """
        Compute ultradian modulation of threshold (Innovation #32).

        Theory: ~90-minute oscillations in threshold due to neuromodulator depletion
        and metabolic accumulation under sustained cognitive load.

        Mechanism:
            Δθ_ultradian = A_ultradian · cos(ω_ultradian · t + φ) · task_load

            where:
            - A_ultradian: Oscillation amplitude (~20% of θ_0)
            - ω_ultradian: Angular frequency (2π/90 min)
            - φ: Phase offset (typically 0)
            - task_load: Cognitive load factor ∈ [0, 1]

        Args:
            time_since_task_start_min: Minutes since task/session began
            task_load: Cognitive load factor (0=rest, 1=maximal load)

        Returns:
            Threshold modulation Δθ_ultradian in σ units

        Note: This is APGI's contribution to BRAC. Full ultradian rhythm involves
        multiple neurotransmitter systems (NE, DA, ACh) beyond θ_t modulation alone.

        References:
            - Kleitman 1963: Basic rest-activity cycle (BRAC)
            - Dijk & Czeisler 1995: Two-process model (circadian × homeostatic)
        """
        # Ultradian parameters
        period_min = 90.0  # ~90-minute cycle
        amplitude = 0.2 * self.theta_t0  # 20% of baseline threshold
        omega = 2 * np.pi / period_min  # Angular frequency

        # Compute modulation (scaled by task load)
        delta_theta_ultradian = amplitude * np.cos(omega * time_since_task_start_min)
        delta_theta_ultradian *= task_load  # Only active during cognitive engagement

        return delta_theta_ultradian


@dataclass
class APGIState:
    """State variables for APGI model at a single timestep."""

    S: float = 0.0  # Accumulated signal (z-score units)
    theta_t: float = 3.0  # Current broadcast threshold (σ units)
    eta_m: float = 0.0  # Metabolic modulation (threshold elevation in σ)
    I: int = 0  # Binary ignition indicator
    ignition_probability: float = 0.0  # P(ignition|t)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "S": self.S,
            "theta_t": self.theta_t,
            "eta_m": self.eta_m,
            "I": self.I,
            "ignition_probability": self.ignition_probability,
        }


class APGIFullDynamicModel:
    """
    APGI Full Dynamic Model Implementation.

    Implements the complete dynamical system with:
    - Signal accumulation with decay
    - Adaptive threshold dynamics
    - Stochastic ignition probability
    - Metabolic state coupling
    - Signal standardization
    """

    def __init__(self, params: Optional[APGIParameters] = None):
        """
        Initialize APGI model.

        Args:
            params: Model parameters. If None, uses default parameters.
        """
        self.params = params or APGIParameters()
        self.state = APGIState(theta_t=self.params.theta_t0)
        self.timestep: int = 0

        # Baseline statistics for standardization
        self.baseline_stats: Dict[str, Tuple[float, float]] = {}

    def standardize_signal(
        self, signal: np.ndarray, baseline_window: int = 20
    ) -> np.ndarray:
        """
        Z-score normalize prediction errors.

        Args:
            signal: Raw signal values
            baseline_window: Number of samples for baseline calculation

        Returns:
            Normalized signal in z-score units
        """
        if len(signal) < baseline_window:
            raise ValueError(
                f"Signal length {len(signal)} < baseline_window {baseline_window}"
            )

        baseline = signal[:baseline_window]
        mu_baseline = np.mean(baseline)
        sigma_baseline = np.std(baseline)

        if sigma_baseline == 0:
            raise ValueError("Baseline std is zero, cannot standardize")

        return (signal - mu_baseline) / sigma_baseline

    def signal_accumulation(
        self,
        Pi_e: float,
        Pi_i: float,
        epsilon_e: float,
        epsilon_i: float,
        beta: Optional[float] = None,
    ) -> float:
        """
        Compute signal accumulation equation.

        S(t+Δt) = exp(−Δt/τ)·S(t) + we·Πe(t)·|εe(t)| + wi·β_som(t)·Πi(t)·|εi(t)|

        Args:
            Pi_e: External precision/reliability [0,1]
            Pi_i: Internal precision/reliability [0,1]
            epsilon_e: External prediction error magnitude (z-score)
            epsilon_i: Internal prediction error magnitude (z-score)
            beta: Somatic bias (overrides default if provided)

        Returns:
            New accumulated signal S(t+Δt)
        """
        beta_val = beta if beta is not None else self.params.beta

        decay_factor = np.exp(-self.params.delta_t / self.params.tau)
        external_contribution = self.params.we * Pi_e * abs(epsilon_e)
        internal_contribution = self.params.wi * beta_val * Pi_i * abs(epsilon_i)

        S_next = (
            decay_factor * self.state.S + external_contribution + internal_contribution
        )

        return S_next

    def threshold_dynamics(self, I_prev: int) -> float:
        """
        Compute threshold dynamics equation.

        θt(t+Δt) = θt0 + ηm(t) + α·[θt(t) − (θt0 + ηm(t))] + φ·I(t)

        Args:
            I_prev: Ignition indicator from previous timestep

        Returns:
            New threshold θt(t+Δt)
        """
        target_threshold = self.params.theta_t0 + self.state.eta_m
        theta_t_next = (
            target_threshold
            + self.params.alpha * (self.state.theta_t - target_threshold)
            + self.params.phi * I_prev
        )

        return theta_t_next

    def metabolic_dynamics(self, I_prev: int) -> float:
        """
        Compute metabolic state dynamics.

        ηm(t+1) = ηm(t) + γc·I(t) − γr·(1−I(t))

        Args:
            I_prev: Ignition indicator from previous timestep

        Returns:
            New metabolic modulation ηm(t+1)
        """
        eta_m_next = (
            self.state.eta_m
            + self.params.gamma_c * I_prev
            - self.params.gamma_r * (1 - I_prev)
        )

        # Ensure non-negative metabolic modulation
        eta_m_next = max(0.0, eta_m_next)

        return eta_m_next

    def ignition_probability(self, S: float, theta_t: float) -> float:
        """
        Compute ignition probability via sigmoid function.

        P(ignition|t) = 1 / (1 + exp(−k·(S(t)−θt(t))))

        Args:
            S: Accumulated signal
            theta_t: Current threshold

        Returns:
            Ignition probability [0,1]
        """
        logit = self.params.k * (S - theta_t)
        prob = 1.0 / (1.0 + np.exp(-logit))

        return prob

    def step(
        self,
        Pi_e: float,
        Pi_i: float,
        epsilon_e: float,
        epsilon_i: float,
        beta: Optional[float] = None,
        deterministic_ignition: bool = False,
    ) -> APGIState:
        """
        Execute one timestep of APGI dynamics.

        Args:
            Pi_e: External precision/reliability [0,1]
            Pi_i: Internal precision/reliability [0,1]
            epsilon_e: External prediction error magnitude (z-score)
            epsilon_i: Internal prediction error magnitude (z-score)
            beta: Somatic bias (optional override)
            deterministic_ignition: If True, ignition occurs when P > 0.5

        Returns:
            New APGIState after timestep
        """
        I_prev = self.state.I

        # 1. Compute new signal accumulation
        S_next = self.signal_accumulation(Pi_e, Pi_i, epsilon_e, epsilon_i, beta)

        # 2. Compute new threshold
        theta_t_next = self.threshold_dynamics(I_prev)

        # 3. Compute new metabolic state
        eta_m_next = self.metabolic_dynamics(I_prev)

        # 4. Compute ignition probability
        prob_ignition = self.ignition_probability(S_next, theta_t_next)

        # 5. Determine ignition (stochastic or deterministic)
        if deterministic_ignition:
            I_next = 1 if prob_ignition > 0.5 else 0
        else:
            I_next = 1 if np.random.random() < prob_ignition else 0

        # Update state
        self.state = APGIState(
            S=S_next,
            theta_t=theta_t_next,
            eta_m=eta_m_next,
            I=I_next,
            ignition_probability=prob_ignition,
        )

        self.timestep += 1

        return self.state

    def reset(self, theta_t0: Optional[float] = None):
        """
        Reset model to initial state.

        Args:
            theta_t0: Optional override for initial threshold
        """
        initial_theta = theta_t0 if theta_t0 is not None else self.params.theta_t0
        self.state = APGIState(theta_t=initial_theta)
        self.timestep = 0

    def simulate(
        self,
        Pi_e_sequence: np.ndarray,
        Pi_i_sequence: np.ndarray,
        epsilon_e_sequence: np.ndarray,
        epsilon_i_sequence: np.ndarray,
        beta_sequence: Optional[np.ndarray] = None,
        deterministic_ignition: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Run full simulation over sequence of inputs.

        Args:
            Pi_e_sequence: External precision values over time
            Pi_i_sequence: Internal precision values over time
            epsilon_e_sequence: External prediction errors over time
            epsilon_i_sequence: Internal prediction errors over time
            beta_sequence: Optional somatic bias values over time
            deterministic_ignition: If True, ignition occurs when P > 0.5

        Returns:
            Dictionary of time series for all state variables
        """
        n_steps = len(Pi_e_sequence)

        # Validate input lengths
        for arr, name in [
            (Pi_i_sequence, "Pi_i"),
            (epsilon_e_sequence, "epsilon_e"),
            (epsilon_i_sequence, "epsilon_i"),
        ]:
            if len(arr) != n_steps:
                raise ValueError(f"{name} length {len(arr)} != Pi_e length {n_steps}")

        if beta_sequence is not None and len(beta_sequence) != n_steps:
            raise ValueError(
                f"beta_sequence length {len(beta_sequence)} != Pi_e length {n_steps}"
            )

        # Initialize storage
        history = {
            "S": np.zeros(n_steps),
            "theta_t": np.zeros(n_steps),
            "eta_m": np.zeros(n_steps),
            "I": np.zeros(n_steps, dtype=int),
            "ignition_probability": np.zeros(n_steps),
        }

        # Reset model
        self.reset()

        # Run simulation
        for t in range(n_steps):
            beta_val = beta_sequence[t] if beta_sequence is not None else None

            state = self.step(
                Pi_e=Pi_e_sequence[t],
                Pi_i=Pi_i_sequence[t],
                epsilon_e=epsilon_e_sequence[t],
                epsilon_i=epsilon_i_sequence[t],
                beta=beta_val,
                deterministic_ignition=deterministic_ignition,
            )

            history["S"][t] = state.S
            history["theta_t"][t] = state.theta_t
            history["eta_m"][t] = state.eta_m
            history["I"][t] = state.I
            history["ignition_probability"][t] = state.ignition_probability

        return history

    def get_core_equations(self) -> Dict[str, str]:
        """
        Return core APGI equations as formatted strings.

        Returns:
            Dictionary mapping equation names to LaTeX-formatted strings
        """
        return {
            "signal_accumulation": r"$S(t+\Delta t) = \exp(-\Delta t/\tau) \cdot S(t) + w_e \cdot \Pi_e(t) \cdot |\varepsilon_e(t)| + w_i \cdot \beta(t) \cdot \Pi_i(t) \cdot |\varepsilon_i(t)|$",
            "threshold_dynamics": r"$\theta_t(t+\Delta t) = \theta_{t0} + \eta_m(t) + \alpha \cdot [\theta_t(t) - (\theta_{t0} + \eta_m(t))] + \phi \cdot I(t)$",
            "ignition_probability": r"$P(\text{ignition}|t) = \frac{1}{1 + \exp(-k \cdot (S(t) - \theta_t(t)))}$",
            "metabolic_dynamics": r"$\eta_m(t+\Delta t) = \eta_m(t) + \gamma_c \cdot I(t) - \gamma_r \cdot (1 - I(t))$",
            "signal_standardization": r"$\varepsilon_{\text{normalized}} = \frac{\varepsilon_{\text{raw}} - \mu_{\text{baseline}}}{\sigma_{\text{baseline}}}$",
            "core_criterion": r"$(\Pi_e \cdot |\varepsilon_e| + \beta \cdot \Pi_i \cdot |\varepsilon_i|) > \theta_t$",
        }

    def get_parameter_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Return summary of model parameters with constraints.

        Returns:
            Dictionary with parameter names, values, ranges, and descriptions
        """
        return {
            "delta_t": {
                "value": self.params.delta_t,
                "range": "[0.01, 0.2]",
                "typical": "0.05-0.1s",
                "description": "Timestep duration",
            },
            "tau": {
                "value": self.params.tau,
                "range": "[0.05, 1.0]",
                "typical": "0.1-0.3s",
                "description": "Signal decay time constant",
            },
            "theta_t0": {
                "value": self.params.theta_t0,
                "range": "[1.0, 10.0]",
                "typical": "2.0-4.0σ",
                "description": "Resting ignition threshold",
            },
            "alpha": {
                "value": self.params.alpha,
                "range": "(0, 1)",
                "typical": "0.8-0.95",
                "description": "Threshold return-to-baseline rate",
            },
            "phi": {
                "value": self.params.phi,
                "range": "[0.0, 1.0]",
                "typical": "0.3-0.5σ",
                "description": "Post-ignition facilitation",
            },
            "gamma_c": {
                "value": self.params.gamma_c,
                "range": "[0.05, 0.5]",
                "typical": "0.1-0.3σ/ignition",
                "description": "Metabolic cost per ignition event",
            },
            "gamma_r": {
                "value": self.params.gamma_r,
                "range": "[0.01, 0.2]",
                "typical": "0.05-0.1σ/timestep",
                "description": "Metabolic recovery during rest",
            },
            "k": {
                "value": self.params.k,
                "range": "[1.0, 10.0]",
                "typical": "2.0-5.0",
                "description": "Sharpness of threshold transition",
            },
            "beta": {
                "value": self.params.beta,
                "range": "[0.0, 5.0]",
                "typical": "0.5-2.0",
                "description": "Interoceptive signal amplification",
            },
            "we": {
                "value": self.params.we,
                "range": "[0, 1]",
                "typical": "0.5",
                "description": "External signal weight (we + wi = 1)",
            },
            "wi": {
                "value": self.params.wi,
                "range": "[0, 1]",
                "typical": "0.5",
                "description": "Internal signal weight (we + wi = 1)",
            },
        }


def create_default_model() -> APGIFullDynamicModel:
    """
    Factory function to create APGI model with default parameters.

    Returns:
        Initialized APGIFullDynamicModel instance
    """
    return APGIFullDynamicModel()


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("APGI Full Dynamic Model - Example Simulation")
    print("=" * 70)

    # Create model with default parameters
    model = create_default_model()

    # Print parameter summary
    print("\nParameter Summary:")
    print("-" * 70)
    params = model.get_parameter_summary()
    for name, info in params.items():
        print(
            f"{name:15s}: {info['value']:6.3f}  |  Range: {info['range']:12s}  |  Typical: {info['typical']:12s}"
        )

    # Print core equations
    print("\nCore Equations:")
    print("-" * 70)
    equations = model.get_core_equations()
    for name, eq in equations.items():
        print(f"{name:25s}: {eq}")

    # Run example simulation
    print("\nExample Simulation (100 timesteps):")
    print("-" * 70)

    np.random.seed(42)
    n_steps = 100

    # Generate synthetic inputs
    Pi_e_seq = np.random.uniform(0.5, 0.9, n_steps)
    Pi_i_seq = np.random.uniform(0.5, 0.9, n_steps)
    epsilon_e_seq = np.random.normal(0, 1, n_steps)
    epsilon_i_seq = np.random.normal(0, 1, n_steps)

    # Add a strong stimulus at t=50
    epsilon_e_seq[45:55] = np.random.normal(2.5, 0.5, 10)

    # Run simulation
    history = model.simulate(
        Pi_e_sequence=Pi_e_seq,
        Pi_i_sequence=Pi_i_seq,
        epsilon_e_sequence=epsilon_e_seq,
        epsilon_i_sequence=epsilon_i_seq,
        deterministic_ignition=True,
    )

    # Print summary statistics
    n_ignitions = np.sum(history["I"])
    print(f"Number of ignitions: {n_ignitions}")
    print(f"Peak signal: {np.max(history['S']):.3f}σ")
    print(f"Peak threshold: {np.max(history['theta_t']):.3f}σ")
    print(f"Peak metabolic modulation: {np.max(history['eta_m']):.3f}σ")
    print(f"Mean ignition probability: {np.mean(history['ignition_probability']):.3f}")

    print("\n" + "=" * 70)
    print("Simulation complete.")
    print("=" * 70)
