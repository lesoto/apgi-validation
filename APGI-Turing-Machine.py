"""
=============================================================================
APGI TURING MACHINE - Complete Implementation
=============================================================================

Implements Allostatic Precision-Gated Ignition as a discrete-state chemical
computation system with continuous dynamics.

Core equation: S_t = Π_e·|ε_e| + β_som·Π_i·|ε_i| → Ignition if S_t > θ_t
with θ_t dynamically regulated by neuromodulators and metabolic cost

Integrates:
- Tape-based Turing Machine architecture
- APGI precision-weighted multimodal integration
- Continuous surprise dynamics (SDE)
- Neuromodulatory state transitions
- Metabolic cost regulation
- Global workspace broadcast theory

=============================================================================
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# ENUMERATIONS - Discrete State Space
# =============================================================================


class PowerState(Enum):
    """System power state"""

    POWERED = "P"
    DEPOWERED = "X"


class IgnitionState(Enum):
    """Core computational state"""

    SUBTHRESHOLD = "SUBTHRESHOLD"
    IGNITED = "IGNITED"
    INACTIVE = "INACTIVE"
    REFRACTORY = "REFRACTORY"


class WorkspaceLevel(Enum):
    """Evidence accumulation levels (0-100% of threshold)"""

    W0 = 0.0  # Empty
    W1 = 0.2  # 20%
    W2 = 0.4  # 40%
    W3 = 0.6  # 60%
    W4 = 0.8  # 80%
    W5 = 1.0  # 100% - ready for ignition
    IGNITING = 1.2  # Actively igniting


class MetabolicState(Enum):
    """Energy availability"""

    METAB0 = "LOW"  # Sufficient energy
    METAB1 = "MEDIUM"  # Moderate depletion
    METAB2 = "CRITICAL"  # Emergency state


class SurpriseLevel(Enum):
    """Discrete surprise magnitude"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PrecisionState(Enum):
    """Signal reliability state"""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Neuromodulator(Enum):
    """Neuromodulatory systems"""

    DA = "DOPAMINE"  # Reward, motivation
    NE = "NOREPINEPHRINE"  # Threat, arousal
    AC = "ACETYLCHOLINE"  # Attention, learning
    SEROTONIN = "5HT"  # Homeostasis, mood


class SignalModality(Enum):
    """Sensory modality types"""

    # Exteroceptive
    VISUAL = "VISUAL"
    AUDITORY = "AUDITORY"
    TACTILE = "TACTILE"
    # Interoceptive
    HEARTBEAT = "HEARTBEAT"
    RESPIRATION = "RESPIRATION"
    PAIN = "PAIN"
    HUNGER = "HUNGER"
    # Internal
    NONE = "NONE"


class TaskContext(Enum):
    """Task-related context states"""

    THREAT = "THREAT"
    REWARD = "REWARD"
    ATTENTION = "ATTENTION"
    REST = "REST"
    SOCIAL = "SOCIAL"


class ReportabilityState(Enum):
    """Conscious access state"""

    REPORTABLE = "REPORTABLE"
    WORKSPACE_BROADCAST = "WORKSPACE_BROADCAST"
    CONSCIOUS_REPORT = "CONSCIOUS_REPORT"
    REFRACTORY = "REFRACTORY"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class APGIParameters:
    """APGI computational parameters"""

    # Precision weights
    Pi_e: float = 0.92  # Exteroceptive precision
    Pi_i: float = 0.85  # Interoceptive precision
    beta: float = 1.2  # Somatic bias

    # Threshold dynamics
    theta_base: float = 0.75  # Baseline threshold
    theta_adapt: float = 0.2  # Adaptation rate
    theta_current: float = 0.75  # Current dynamic threshold

    # Neuromodulatory gains (effect on threshold)
    kappa_DA: float = -0.1  # Dopamine lowers threshold
    kappa_NE: float = -0.15  # NE lowers more (vigilance)
    kappa_AC: float = -0.12  # ACh lowers (attention)
    kappa_5HT: float = 0.08  # Serotonin raises (homeostasis)

    # Metabolic parameters
    metabolic_suppression: float = 0.15  # Energy cost factor
    metabolic_recovery_rate: float = 0.1  # Recovery speed

    # Continuous dynamics parameters
    tau_S: float = 0.5  # Surprise timescale (s)
    tau_theta: float = 30.0  # Threshold timescale (s)
    alpha_sigmoid: float = 5.0  # Ignition sigmoid sharpness
    rho_reset: float = 0.7  # Post-ignition reset fraction

    # Noise strengths
    sigma_S: float = 0.05  # Surprise noise
    sigma_theta: float = 0.02  # Threshold noise

    # Individual differences (new)
    anxiety_level: float = 0.5  # [0,1] affects beta and kappa_NE
    alexithymia_level: float = 0.5  # [0,1] affects beta


@dataclass
class TapeState:
    """Complete Turing Machine tape state"""

    power: PowerState = PowerState.POWERED
    ignition: IgnitionState = IgnitionState.SUBTHRESHOLD
    workspace: WorkspaceLevel = WorkspaceLevel.W0
    metabolic: MetabolicState = MetabolicState.METAB0
    surprise: SurpriseLevel = SurpriseLevel.LOW
    precision: PrecisionState = PrecisionState.MEDIUM
    neuromod: Neuromodulator = Neuromodulator.SEROTONIN
    signal: SignalModality = SignalModality.NONE
    context: TaskContext = TaskContext.REST
    reportability: ReportabilityState = ReportabilityState.REPORTABLE

    # Continuous state variables (from dynamical system)
    S_continuous: float = 0.0  # Accumulated surprise (continuous)
    theta_continuous: float = 0.75  # Dynamic threshold (continuous)

    # Learning and history
    somatic_marker: float = 0.0  # Learned valence
    prediction_accuracy: str = "NEUTRAL"  # NOVEL, ACCURATE, INACCURATE

    # Energy tracking
    energy_consumed: float = 0.0
    total_ignitions: int = 0

    # Timing
    time: float = 0.0
    last_ignition_time: float = -np.inf


@dataclass
class APGIEvent:
    """Record of significant events"""

    time: float
    event_type: str  # 'IGNITION', 'THRESHOLD_CROSS', 'NEUROMOD_SWITCH', etc.
    state_snapshot: Dict[str, Any]
    energy_cost: float = 0.0


# =============================================================================
# CORE TURING MACHINE CLASS
# =============================================================================


class APGITuringMachine:
    """
    Complete APGI Turing Machine implementation

    Combines discrete state transitions with continuous dynamics
    """

    def __init__(self, params: Optional[APGIParameters] = None):
        """Initialize with parameters"""
        self.params = params if params is not None else APGIParameters()
        self.state = TapeState()
        self.history: List[TapeState] = []
        self.events: List[APGIEvent] = []

        # Initialize individual-difference parameters
        self._update_individual_parameters()

        # Running statistics for precision estimation
        self.extero_buffer = deque(maxlen=2500)  # ~10s at 250Hz
        self.intero_buffer = deque(maxlen=2500)

    def _update_individual_parameters(self):
        """Update parameters based on individual differences"""
        # Anxiety increases somatic bias and NE sensitivity
        anxiety_factor = (self.params.anxiety_level - 0.5) * 0.4
        self.params.beta += anxiety_factor
        self.params.kappa_NE -= anxiety_factor * 0.1

        # Alexithymia decreases somatic bias
        alexi_factor = (self.params.alexithymia_level - 0.5) * 0.4
        self.params.beta -= alexi_factor

        # Enforce bounds
        self.params.beta = np.clip(self.params.beta, 0.3, 2.0)
        self.params.kappa_NE = np.clip(self.params.kappa_NE, -0.3, 0.0)

    # =========================================================================
    # CLASSIFICATION HELPERS
    # =========================================================================

    @staticmethod
    def is_interoceptive(signal: SignalModality) -> bool:
        """Check if signal is interoceptive"""
        return signal in [
            SignalModality.HEARTBEAT,
            SignalModality.RESPIRATION,
            SignalModality.PAIN,
            SignalModality.HUNGER,
        ]

    @staticmethod
    def is_exteroceptive(signal: SignalModality) -> bool:
        """Check if signal is exteroceptive"""
        return signal in [
            SignalModality.VISUAL,
            SignalModality.AUDITORY,
            SignalModality.TACTILE,
        ]

    # =========================================================================
    # DYNAMIC THRESHOLD COMPUTATION
    # =========================================================================

    def compute_dynamic_threshold(self) -> float:
        """
        Compute current threshold based on neuromodulators and metabolism

        θ_t = θ_base + Σ(κ_i · [neuromod_i]) + metabolic_effect
        """
        # Neuromodulatory contribution
        neuromod_effect = {
            Neuromodulator.DA: self.params.kappa_DA,
            Neuromodulator.NE: self.params.kappa_NE,
            Neuromodulator.AC: self.params.kappa_AC,
            Neuromodulator.SEROTONIN: self.params.kappa_5HT,
        }.get(self.state.neuromod, 0.0)

        # Metabolic contribution (emergency threshold raise)
        metabolic_effect = {
            MetabolicState.METAB0: -0.05,  # Slight lowering when energy high
            MetabolicState.METAB1: 0.0,  # Neutral
            MetabolicState.METAB2: 0.15,  # Emergency raise
        }.get(self.state.metabolic, 0.0)

        threshold = self.params.theta_base + neuromod_effect + metabolic_effect
        return np.clip(threshold, 0.1, 0.95)

    # =========================================================================
    # PRECISION-WEIGHTED PREDICTION ERROR
    # =========================================================================

    def compute_weighted_prediction_error(
        self, signal: SignalModality, raw_error: float
    ) -> float:
        """
        Compute precision-weighted prediction error

        PE_weighted = Π · |ε| · β_som^(is_intero)

        """
        if self.is_interoceptive(signal):
            # Interoceptive: apply somatic bias
            precision = self.params.Pi_i
            weighted_pe = precision * np.abs(raw_error) * self.params.beta
        elif self.is_exteroceptive(signal):
            # Exteroceptive: standard weighting
            precision = self.params.Pi_e
            weighted_pe = precision * np.abs(raw_error)
        else:
            weighted_pe = np.abs(raw_error)

        return weighted_pe

    def estimate_precision_from_buffer(self, is_intero: bool) -> float:
        """
        Estimate precision as inverse variance of recent errors

        Π = 1 / (σ² + ε)
        """
        buffer = self.intero_buffer if is_intero else self.extero_buffer

        if len(buffer) < 10:
            return self.params.Pi_i if is_intero else self.params.Pi_e

        variance = np.var(list(buffer)) + 1e-8
        precision = 1.0 / variance

        # Clip to physiological bounds
        return np.clip(precision, 0.1, 10.0)

    # =========================================================================
    # DISCRETE STATE TRANSITIONS
    # =========================================================================

    def transition_surprise_accumulation(
        self, signal: SignalModality, raw_prediction_error: float, dt: float = 0.05
    ) -> None:
        """
        SUBTHRESHOLD → Accumulate surprise → Check for ignition

        Core APGI computation happens here
        """
        if self.state.ignition != IgnitionState.SUBTHRESHOLD:
            return

        # 1. Compute precision-weighted prediction error
        weighted_pe = self.compute_weighted_prediction_error(
            signal, raw_prediction_error
        )

        # 2. Update running buffers for precision estimation
        if self.is_interoceptive(signal):
            self.intero_buffer.append(raw_prediction_error)
        elif self.is_exteroceptive(signal):
            self.extero_buffer.append(raw_prediction_error)

        # 3. Update continuous surprise via Euler integration
        # dS = (-S/τ_S + weighted_PE) dt + σ_S dW
        noise = np.random.normal(0, 1) * np.sqrt(dt) * self.params.sigma_S
        dS = ((-self.state.S_continuous / self.params.tau_S) + weighted_pe) * dt + noise
        self.state.S_continuous = max(0, self.state.S_continuous + dS)

        # 4. Convert continuous surprise to discrete level
        if self.state.S_continuous > 0.6:
            self.state.surprise = SurpriseLevel.HIGH
        elif self.state.S_continuous > 0.3:
            self.state.surprise = SurpriseLevel.MEDIUM
        else:
            self.state.surprise = SurpriseLevel.LOW

        # 5. Update workspace accumulation (probabilistic)
        if self.state.surprise == SurpriseLevel.HIGH:
            if np.random.random() < self.params.Pi_e:  # Reliability-gated
                self._advance_workspace()

        # 6. Check for ignition threshold crossing
        current_threshold = self.compute_dynamic_threshold()
        self.state.theta_continuous = current_threshold

        workspace_level = self.state.workspace.value

        if (
            self.state.S_continuous >= current_threshold
            and workspace_level >= 0.6  # Sufficient evidence
            and self.state.surprise == SurpriseLevel.HIGH
            and self.state.reportability == ReportabilityState.REPORTABLE
        ):

            # IGNITION TRIGGER
            self._trigger_ignition(dt)

    def _advance_workspace(self) -> None:
        """Advance workspace evidence level"""
        level_sequence = [
            WorkspaceLevel.W0,
            WorkspaceLevel.W1,
            WorkspaceLevel.W2,
            WorkspaceLevel.W3,
            WorkspaceLevel.W4,
            WorkspaceLevel.W5,
        ]

        try:
            current_idx = level_sequence.index(self.state.workspace)
            if current_idx < len(level_sequence) - 1:
                self.state.workspace = level_sequence[current_idx + 1]
        except ValueError:
            pass  # Already at max or in special state

    def _trigger_ignition(self, dt: float) -> None:
        """
        Execute ignition transition

        SUBTHRESHOLD → IGNITED
        - Update metabolic cost
        - Set workspace to broadcasting
        - Record event
        - Apply post-ignition reset
        """
        # State transition
        self.state.ignition = IgnitionState.IGNITED
        self.state.workspace = WorkspaceLevel.IGNITING
        self.state.reportability = ReportabilityState.WORKSPACE_BROADCAST

        # Metabolic cost
        energy_cost = self._compute_ignition_energy_cost()
        self.state.energy_consumed += energy_cost
        self._update_metabolic_state()

        # Adaptive threshold increase (hebbian learning of ignition cost)
        metabolic_factor = {
            MetabolicState.METAB2: -0.1,  # Lower threshold when depleted (desperation)
            MetabolicState.METAB1: 0.1,  # Raise when moderate
            MetabolicState.METAB0: 0.05,  # Slight raise when abundant
        }.get(self.state.metabolic, 0.0)

        self.params.theta_base = np.clip(
            self.params.theta_base + self.params.theta_adapt * metabolic_factor,
            0.4,
            0.95,
        )

        # Record event
        self.state.total_ignitions += 1
        self.state.last_ignition_time = self.state.time

        event = APGIEvent(
            time=self.state.time,
            event_type="IGNITION",
            state_snapshot={
                "S": self.state.S_continuous,
                "theta": self.state.theta_continuous,
                "workspace": self.state.workspace.name,
                "neuromod": self.state.neuromod.name,
                "surprise": self.state.surprise.name,
            },
            energy_cost=energy_cost,
        )
        self.events.append(event)

    def _compute_ignition_energy_cost(self) -> float:
        """
        Compute energy cost of ignition event

        Cost scales with:
        - Workspace broadcasting extent
        - Current metabolic state
        - Neuromodulatory drive
        """
        base_cost = 75.0

        # Metabolic state multiplier
        metabolic_mult = {
            MetabolicState.METAB0: 1.0,
            MetabolicState.METAB1: 1.2,
            MetabolicState.METAB2: 1.5,
        }.get(self.state.metabolic, 1.0)

        # Neuromodulator efficiency
        neuromod_mult = {
            Neuromodulator.DA: 0.9,  # Dopamine improves efficiency
            Neuromodulator.NE: 1.1,  # NE increases cost (stress)
            Neuromodulator.AC: 1.0,  # ACh neutral
            Neuromodulator.SEROTONIN: 0.95,  # 5HT slightly efficient
        }.get(self.state.neuromod, 1.0)

        return base_cost * metabolic_mult * neuromod_mult

    def _update_metabolic_state(self) -> None:
        """Update metabolic state based on energy consumption"""
        if self.state.energy_consumed > 500:
            self.state.metabolic = MetabolicState.METAB2
        elif self.state.energy_consumed > 200:
            self.state.metabolic = MetabolicState.METAB1
        else:
            self.state.metabolic = MetabolicState.METAB0

    def transition_global_broadcast(self, dt: float) -> None:
        """
        IGNITED → Global workspace broadcast → Report

        Implements conscious access via global neuronal workspace
        """
        if self.state.ignition != IgnitionState.IGNITED:
            return

        if self.state.workspace == WorkspaceLevel.IGNITING:
            # Broadcast in progress
            self.state.reportability = ReportabilityState.CONSCIOUS_REPORT

            # Learn somatic marker based on outcome
            self.state.somatic_marker = self._compute_somatic_marker()

            # Reset workspace after broadcast
            self.state.workspace = WorkspaceLevel.W0

            # Apply post-ignition reset to surprise
            self.state.S_continuous *= 1.0 - self.params.rho_reset

            event = APGIEvent(
                time=self.state.time,
                event_type="GLOBAL_BROADCAST",
                state_snapshot={"somatic_marker": self.state.somatic_marker},
                energy_cost=25.0,  # Broadcasting cost
            )
            self.events.append(event)

        else:
            # Decay back to subthreshold
            self.state.ignition = IgnitionState.SUBTHRESHOLD
            self.state.reportability = ReportabilityState.REFRACTORY

            # Metabolic recovery begins
            if self.state.metabolic == MetabolicState.METAB2:
                self.state.metabolic = MetabolicState.METAB1

    def _compute_somatic_marker(self) -> float:
        """
        Learn somatic marker value based on outcome

        Implements Damasio's somatic marker hypothesis
        """
        # Outcome valence (simplified - would come from reward/punishment in real system)
        valence_probs = [
            (0.6, 1.0),
            (0.3, 0.0),
            (0.1, -1.0),
        ]  # Positive, Neutral, Negative
        valence = np.random.choice(
            [v for _, v in valence_probs], p=[p for p, _ in valence_probs]
        )

        # Update with learning rate
        learning_rate = 0.3
        new_marker = (
            1 - learning_rate
        ) * self.state.somatic_marker + learning_rate * valence

        return np.clip(new_marker, -2.0, 2.0)

    # =========================================================================
    # PRECISION DYNAMICS
    # =========================================================================

    def transition_precision_modulation(self, dt: float) -> None:
        """
        Update precision state based on prediction accuracy

        High prediction errors → Increase precision (signal is informative)
        High accuracy → Decrease precision (signal is redundant)
        """
        if self.state.prediction_accuracy == "NOVEL":
            # Novel stimuli increase precision
            if np.random.random() < 0.8:
                self.state.precision = PrecisionState.HIGH
                # Update continuous precision
                if self.is_interoceptive(self.state.signal):
                    self.params.Pi_i = min(10.0, self.params.Pi_i * 1.1)
                else:
                    self.params.Pi_e = min(10.0, self.params.Pi_e * 1.1)

        elif self.state.prediction_accuracy == "ACCURATE":
            # Accurate predictions reduce precision (precision decay)
            self.state.precision = PrecisionState.LOW
            if self.is_interoceptive(self.state.signal):
                self.params.Pi_i = max(0.1, self.params.Pi_i * 0.95)
            else:
                self.params.Pi_e = max(0.1, self.params.Pi_e * 0.95)

        elif self.state.prediction_accuracy == "INACCURATE":
            # Inaccurate predictions increase precision
            self.state.precision = PrecisionState.MEDIUM
            if self.is_interoceptive(self.state.signal):
                self.params.Pi_i = min(10.0, self.params.Pi_i * 1.05)
            else:
                self.params.Pi_e = min(10.0, self.params.Pi_e * 1.05)

    # =========================================================================
    # NEUROMODULATORY TRANSITIONS
    # =========================================================================

    def transition_neuromodulator(self, dt: float) -> None:
        """
        Context-dependent neuromodulator transitions

        Maps task contexts to appropriate neuromodulatory states
        """
        context = self.state.context
        current_neuromod = self.state.neuromod

        # Initialize with current neuromodulator
        new_neuromod = current_neuromod

        # Transition probabilities based on context
        if context == TaskContext.THREAT:
            if np.random.random() < 0.85:
                new_neuromod = Neuromodulator.NE

        elif context == TaskContext.REWARD:
            if np.random.random() < 0.7:
                new_neuromod = Neuromodulator.DA

        elif context == TaskContext.ATTENTION:
            if np.random.random() < 0.6:
                new_neuromod = Neuromodulator.AC

        elif context == TaskContext.REST:
            if np.random.random() < 0.5:
                new_neuromod = Neuromodulator.SEROTONIN

        if new_neuromod != current_neuromod:
            self.state.neuromod = new_neuromod

            event = APGIEvent(
                time=self.state.time,
                event_type="NEUROMOD_SWITCH",
                state_snapshot={
                    "from": current_neuromod.name,
                    "to": new_neuromod.name,
                    "context": context.name,
                },
                energy_cost=5.0,
            )
            self.events.append(event)

    # =========================================================================
    # METABOLIC REGULATION
    # =========================================================================

    def transition_metabolic_recovery(self, dt: float) -> None:
        """
        Metabolic recovery and emergency regulation
        """
        # Critical energy state
        if self.state.metabolic == MetabolicState.METAB2:
            # Emergency threshold increase
            self.params.theta_base = min(0.95, self.params.theta_base + 0.15)

            # Force workspace reset
            self.state.workspace = WorkspaceLevel.W0

            # Switch to homeostatic neuromodulation
            if np.random.random() < 0.7:
                self.state.neuromod = Neuromodulator.SEROTONIN

        # Passive recovery
        recovery_rate = self.params.metabolic_recovery_rate * dt
        self.state.energy_consumed = max(0, self.state.energy_consumed - recovery_rate)

        # Update metabolic state based on recovered energy
        self._update_metabolic_state()

    # =========================================================================
    # CONTINUOUS DYNAMICS INTEGRATION
    # =========================================================================

    def integrate_continuous_dynamics(self, dt: float) -> None:
        """
        Euler-Maruyama integration of continuous variables

        Integrates SDE for surprise and threshold dynamics
        """
        # Stochastic noise terms
        xi_S = np.random.normal(0, 1) * np.sqrt(dt)
        xi_theta = np.random.normal(0, 1) * np.sqrt(dt)

        # Surprise dynamics: dS = (-S/τ_S) dt + σ dW
        # (Input drive added in transition_surprise_accumulation)
        dS = (-self.state.S_continuous / self.params.tau_S) * dt
        dS += self.params.sigma_S * xi_S
        self.state.S_continuous = max(0, self.state.S_continuous + dS)

        # Threshold dynamics: dθ = ((θ_0 - θ)/τ_θ) dt + σ dW
        current_theta_target = self.compute_dynamic_threshold()
        dTheta = (
            (current_theta_target - self.state.theta_continuous) / self.params.tau_theta
        ) * dt
        dTheta += self.params.sigma_theta * xi_theta
        self.state.theta_continuous = max(0.1, self.state.theta_continuous + dTheta)

    # =========================================================================
    # MAIN STEP FUNCTION
    # =========================================================================

    def step(self, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one Turing Machine step

        Args:
            dt: Time step (seconds)
            inputs: Dictionary containing:
                - signal: SignalModality
                - prediction_error: float
                - context: TaskContext (optional)
                - accuracy: str (optional) - 'NOVEL', 'ACCURATE', 'INACCURATE'

        Returns:
            Current state summary
        """
        # Update time
        self.state.time += dt

        # Update inputs
        self.state.signal = inputs.get("signal", SignalModality.NONE)
        raw_pe = inputs.get("prediction_error", 0.0)
        self.state.context = inputs.get("context", self.state.context)
        self.state.prediction_accuracy = inputs.get("accuracy", "NEUTRAL")

        # Check power state
        if self.state.power != PowerState.POWERED:
            return self._get_state_summary()

        # Execute state-dependent transitions
        if self.state.ignition == IgnitionState.SUBTHRESHOLD:
            self.transition_surprise_accumulation(self.state.signal, raw_pe, dt)

        elif self.state.ignition == IgnitionState.IGNITED:
            self.transition_global_broadcast(dt)

        # Parallel transitions (always active)
        self.transition_precision_modulation(dt)
        self.transition_neuromodulator(dt)
        self.transition_metabolic_recovery(dt)
        self.integrate_continuous_dynamics(dt)

        # Refractory period handling
        if self.state.reportability == ReportabilityState.REFRACTORY:
            if self.state.time - self.state.last_ignition_time > 1.0:  # 1s refractory
                self.state.reportability = ReportabilityState.REPORTABLE

        # Store history
        self.history.append(self._copy_state())

        return self._get_state_summary()

    def _copy_state(self) -> TapeState:
        """Create deep copy of current state"""
        from copy import deepcopy

        return deepcopy(self.state)

    def _get_state_summary(self) -> Dict[str, Any]:
        """Extract current state summary"""
        return {
            "time": self.state.time,
            "ignition": self.state.ignition.name,
            "S": self.state.S_continuous,
            "theta": self.state.theta_continuous,
            "workspace": self.state.workspace.name,
            "surprise": self.state.surprise.name,
            "precision": self.state.precision.name,
            "neuromod": self.state.neuromod.name,
            "metabolic": self.state.metabolic.name,
            "reportability": self.state.reportability.name,
            "energy": self.state.energy_consumed,
            "ignitions": self.state.total_ignitions,
            "prob_ignition": self._compute_ignition_probability(),
        }

    def _compute_ignition_probability(self) -> float:
        """
        Compute instantaneous ignition probability

        P(ignite) = σ(α · (S - θ))
        """
        signal = self.state.S_continuous - self.state.theta_continuous
        prob = 1.0 / (1.0 + np.exp(-self.params.alpha_sigmoid * signal))
        return prob

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def reset(self) -> None:
        """Reset to initial state"""
        self.state = TapeState()
        self.history.clear()
        self.events.clear()
        self.extero_buffer.clear()
        self.intero_buffer.clear()

    def get_history_dataframe(self):
        """Convert history to pandas DataFrame"""
        import pandas as pd

        records = []
        for state in self.history:
            records.append(
                {
                    "time": state.time,
                    "ignition": state.ignition.name,
                    "S": state.S_continuous,
                    "theta": state.theta_continuous,
                    "workspace": state.workspace.value,
                    "surprise": state.surprise.name,
                    "precision": state.precision.name,
                    "neuromod": state.neuromod.name,
                    "metabolic": state.metabolic.name,
                    "energy": state.energy_consumed,
                    "ignitions": state.total_ignitions,
                }
            )

        return pd.DataFrame(records)


# =============================================================================
# VISUALIZATION
# =============================================================================


class APGIVisualizer:
    """Visualization tools for APGI Turing Machine"""

    @staticmethod
    def plot_simulation(
        machine: APGITuringMachine, title: str = "APGI Turing Machine Simulation"
    ):
        """
        Create comprehensive visualization of simulation
        """
        df = machine.get_history_dataframe()

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # Plot 1: Surprise and Threshold
        ax0 = axes[0]
        ax0.plot(
            df["time"], df["S"], label="$S_t$ (Surprise)", color="blue", linewidth=1.5
        )
        ax0.plot(
            df["time"],
            df["theta"],
            label=r"$\theta_t$ (Threshold)",
            color="orange",
            linestyle="--",
            linewidth=1.5,
        )

        # Mark ignitions
        ignited = df[df["ignition"] == "IGNITED"]
        if len(ignited) > 0:
            ax0.scatter(
                ignited["time"],
                ignited["S"],
                color="red",
                s=80,
                zorder=5,
                marker="*",
                label="Ignition",
            )

        ax0.set_ylabel("Magnitude")
        ax0.set_title(f"{title}: Dynamics")
        ax0.legend(loc="upper right")
        ax0.grid(True, alpha=0.3)

        # Plot 2: Workspace and Energy
        ax1 = axes[1]
        ax1_twin = ax1.twinx()

        ln1 = ax1.plot(
            df["time"],
            df["workspace"],
            label="Workspace Level",
            color="green",
            linewidth=1.5,
        )
        ln2 = ax1_twin.plot(
            df["time"],
            df["energy"],
            label="Energy Consumed",
            color="brown",
            linestyle=":",
            linewidth=1.5,
        )

        ax1.set_ylabel("Workspace Level")
        ax1_twin.set_ylabel("Energy (AU)")

        lns = ln1 + ln2
        labs = [line.get_label() for line in lns]
        ax1.legend(lns, labs, loc="upper right")
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Evidence Accumulation & Metabolic Cost")

        # Plot 3: Discrete States
        ax2 = axes[2]

        # Convert categorical variables to numeric for plotting
        neuromod_map = {
            "DOPAMINE": 0,
            "NOREPINEPHRINE": 1,
            "ACETYLCHOLINE": 2,
            "5HT": 3,
        }
        df.loc[:, "neuromod_numeric"] = df["neuromod"].map(neuromod_map)

        ax2.step(
            df["time"],
            df["neuromod_numeric"],
            where="post",
            label="Neuromodulator",
            linewidth=1.5,
        )
        ax2.set_ylabel("Neuromodulator State")
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(["DA", "NE", "ACh", "5HT"])
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Neuromodulatory State")

        # Plot 4: Ignition Events
        ax3 = axes[3]

        # Cumulative ignitions
        ax3.plot(
            df["time"],
            df["ignitions"],
            label="Cumulative Ignitions",
            color="purple",
            linewidth=2,
            marker="o",
            markersize=3,
        )

        ax3.set_xlabel("Time (seconds)")
        ax3.set_ylabel("Total Ignitions")
        ax3.set_title("Conscious Access Events")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_event_timeline(machine: APGITuringMachine):
        """Plot timeline of significant events"""
        if not machine.events:
            print("No events to plot")
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        event_types = {}
        for event in machine.events:
            if event.event_type not in event_types:
                event_types[event.event_type] = []
            event_types[event.event_type].append(event.time)

        colors = {
            "IGNITION": "red",
            "GLOBAL_BROADCAST": "blue",
            "NEUROMOD_SWITCH": "green",
        }

        for i, (event_type, times) in enumerate(event_types.items()):
            ax.scatter(
                times,
                [i] * len(times),
                color=colors.get(event_type, "gray"),
                s=100,
                label=event_type,
                alpha=0.7,
            )

        ax.set_yticks(range(len(event_types)))
        ax.set_yticklabels(list(event_types.keys()))
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Event Timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# =============================================================================
# SIMULATION SCENARIOS
# =============================================================================


def run_baseline_simulation(duration: float = 120.0, dt: float = 0.05):
    """
    Run baseline simulation with standard parameters
    """
    print("=" * 70)
    print("APGI TURING MACHINE - Baseline Simulation")
    print("=" * 70)

    # Initialize machine
    params = APGIParameters()
    machine = APGITuringMachine(params)

    steps = int(duration / dt)

    print(f"Duration: {duration}s | Time step: {dt}s | Steps: {steps}")
    print(
        f"Parameters: Π_e={params.Pi_e:.2f}, Π_i={params.Pi_i:.2f}, β_som={params.beta:.2f}"
    )
    print()

    # Generate synthetic input sequence
    print("Generating input sequence...")

    # Background noise
    eps_e = np.random.normal(0, 0.1, steps)
    eps_i = np.random.normal(0, 0.1, steps)

    # Event 1: External surprise burst (t=30s)
    event1_start = int(30.0 / dt)
    event1_end = int(35.0 / dt)
    eps_e[event1_start:event1_end] += np.random.normal(
        2.0, 0.5, event1_end - event1_start
    )

    # Event 2: Internal somatic burst (t=80s)
    event2_start = int(80.0 / dt)
    event2_end = int(85.0 / dt)
    eps_i[event2_start:event2_end] += np.random.normal(
        1.5, 0.5, event2_end - event2_start
    )

    # Context transitions
    contexts = [TaskContext.REST] * steps
    contexts[event1_start:event1_end] = [TaskContext.THREAT] * (
        event1_end - event1_start
    )
    contexts[event2_start:event2_end] = [TaskContext.ATTENTION] * (
        event2_end - event2_start
    )

    # Signal modalities
    signals = [
        SignalModality.VISUAL if t < steps // 2 else SignalModality.HEARTBEAT
        for t in range(steps)
    ]

    # Run simulation
    print("Running simulation...")

    for t_idx in range(steps):
        # Determine prediction error based on modality
        if machine.is_exteroceptive(signals[t_idx]):
            pe = eps_e[t_idx]
        else:
            pe = eps_i[t_idx]

        # Construct input
        inputs = {
            "signal": signals[t_idx],
            "prediction_error": pe,
            "context": contexts[t_idx],
            "accuracy": "NOVEL" if abs(pe) > 1.0 else "ACCURATE",
        }

        # Step machine
        state = machine.step(dt, inputs)

        # Progress indicator
        if t_idx % 500 == 0:
            print(
                f"  t={state['time']:.1f}s | "
                f"S={state['S']:.3f} | "
                f"θ={state['theta']:.3f} | "
                f"Ignitions={state['ignitions']}"
            )

    print()
    print("Simulation complete!")
    print(f"Total ignitions: {machine.state.total_ignitions}")
    print(f"Final energy consumed: {machine.state.energy_consumed:.1f}")
    print(f"Final threshold: {machine.params.theta_base:.3f}")
    print()

    return machine


def run_anxiety_comparison(duration: float = 60.0):
    """
    Compare neutral vs high-anxiety individual
    """
    print("=" * 70)
    print("APGI TURING MACHINE - Anxiety Comparison")
    print("=" * 70)

    dt = 0.05
    steps = int(duration / dt)

    # Neutral individual
    print("\nProfile 1: NEUTRAL (anxiety=0.5, alexithymia=0.5)")
    params_neutral = APGIParameters(anxiety_level=0.5, alexithymia_level=0.5)
    machine_neutral = APGITuringMachine(params_neutral)

    # High anxiety individual
    print("Profile 2: HIGH ANXIETY (anxiety=0.9, alexithymia=0.2)")
    params_anxiety = APGIParameters(anxiety_level=0.9, alexithymia_level=0.2)
    machine_anxiety = APGITuringMachine(params_anxiety)

    print(
        f"\nNeutral β_som={params_neutral.beta:.3f}, κ_NE={params_neutral.kappa_NE:.3f}"
    )
    print(
        f"Anxiety β_som={params_anxiety.beta:.3f}, κ_NE={params_anxiety.kappa_NE:.3f}"
    )
    print()

    # Same input sequence for both
    eps_threat = np.random.normal(0, 0.1, steps)
    threat_start = int(20.0 / dt)
    threat_end = int(25.0 / dt)
    eps_threat[threat_start:threat_end] += np.random.normal(
        1.2, 0.3, threat_end - threat_start
    )

    contexts = [TaskContext.REST] * steps
    contexts[threat_start:threat_end] = [TaskContext.THREAT] * (
        threat_end - threat_start
    )

    # Run both
    for t_idx in range(steps):
        inputs = {
            "signal": SignalModality.HEARTBEAT,
            "prediction_error": eps_threat[t_idx],
            "context": contexts[t_idx],
        }

        machine_neutral.step(dt, inputs)
        machine_anxiety.step(dt, inputs)

    print("Results:")
    print(
        f"  Neutral - Ignitions: {machine_neutral.state.total_ignitions}, "
        f"Energy: {machine_neutral.state.energy_consumed:.1f}"
    )
    print(
        f"  Anxiety - Ignitions: {machine_anxiety.state.total_ignitions}, "
        f"Energy: {machine_anxiety.state.energy_consumed:.1f}"
    )
    print()

    return machine_neutral, machine_anxiety


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run baseline simulation
    machine = run_baseline_simulation(duration=120.0, dt=0.05)

    # Visualize
    print("Generating visualizations...")
    fig1 = APGIVisualizer.plot_simulation(machine)
    fig2 = APGIVisualizer.plot_event_timeline(machine)

    plt.show()

    # Run anxiety comparison
    print("\n" + "=" * 70)
    machine_neutral, machine_anxiety = run_anxiety_comparison(duration=60.0)

    # Compare visualizations
    fig3 = APGIVisualizer.plot_simulation(
        machine_neutral, title="APGI - Neutral Individual"
    )
    fig4 = APGIVisualizer.plot_simulation(
        machine_anxiety, title="APGI - High Anxiety Individual"
    )

    plt.show()

    print("\n" + "=" * 70)
    print("APGI TURING MACHINE - Simulation Complete")
    print("=" * 70)
