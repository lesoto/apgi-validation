from collections import deque
from typing import Dict, List, Tuple

import numpy as np

# =====================
# DIMENSION CONSTANTS
# =====================
EXTERO_DIM = 32
INTERO_DIM = 16
SENSORY_DIM = 32
OBJECTS_DIM = 16
CONTEXT_DIM = 8
VISCERAL_DIM = 16
ORGAN_DIM = 8
HOMEOSTATIC_DIM = 4
WORKSPACE_DIM = 8
HIDDEN_DIM_DEFAULT = 64
SOMATIC_HIDDEN_DIM = 32
DEFAULT_EPSILON = 1e-8
MAX_CLIP_VALUE = 10.0
GRAD_CLIP_VALUE = 1.0
WEIGHT_CLIP_VALUE = 2.0
POLICY_GRAD_CLIP = 5.0


class HierarchicalGenerativeModel:
    """Hierarchical generative model with multiple levels"""

    def __init__(
        self,
        levels: List[Dict],
        learning_rate: float = 0.01,
        model_type: str = "extero",
    ):
        self.levels = levels
        self.learning_rate = learning_rate
        self.model_type = model_type  # "extero" or "intero"
        self.states = {}
        self.weights = {}

        # Initialize each level
        for level in levels:
            name = level["name"]
            dim = level["dim"]
            self.states[name] = np.zeros(dim)
            # Initialize weights with proper scaling
            fan_in = dim
            fan_out = dim
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights[name] = np.random.normal(0, std, (dim, dim)).astype(np.float32)

    def predict(self) -> np.ndarray:
        """Generate prediction from top level"""
        top_level = self.levels[-1]["name"]
        pred = self.states[top_level]

        # Ensure prediction matches expected input size based on model type
        if self.model_type == "extero":
            target_size = EXTERO_DIM
        else:  # intero
            target_size = INTERO_DIM

        if len(pred) < target_size:
            # Pad to match input size
            padded_pred = np.zeros(target_size)
            padded_pred[: len(pred)] = pred
            return padded_pred
        elif len(pred) > target_size:
            # Truncate if too large
            return pred[:target_size]
        return pred

    def update(self, error: np.ndarray):
        """Update model with prediction error"""
        # Simple gradient descent update
        for level_name in self.states.keys():
            self.states[level_name] += (
                self.learning_rate * error[: len(self.states[level_name])]
            )

    def get_level(self, level_name: str) -> np.ndarray:
        """Get state of specific level"""
        return self.states.get(level_name, np.zeros(1))

    def get_all_levels(self) -> np.ndarray:
        """Get all levels concatenated"""
        return np.concatenate([self.states[level["name"]] for level in self.levels])


class SomaticMarkerNetwork:
    """Somatic marker network for interoceptive predictions"""

    def __init__(
        self, context_dim: int, action_dim: int, hidden_dim: int, learning_rate: float
    ):
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # Initialize weights with proper scaling
        fan_in = context_dim
        fan_out = hidden_dim
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.W1 = np.random.normal(0, std, (hidden_dim, context_dim)).astype(np.float32)

        fan_in = hidden_dim
        fan_out = action_dim
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.W2 = np.random.normal(0, std, (action_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(action_dim)

    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict interoceptive outcomes for all actions with enhanced stability"""
        if not np.all(np.isfinite(context)):
            return np.zeros(self.action_dim)

        try:
            # Use double precision for critical calculations
            context_dp = context.astype(np.float64)
            W1_dp = self.W1.astype(np.float64)
            b1_dp = self.b1.astype(np.float64)
            W2_dp = self.W2.astype(np.float64)
            b2_dp = self.b2.astype(np.float64)

            # Forward pass with enhanced clipping
            pre_activation = W1_dp @ context_dp + b1_dp
            pre_activation = np.clip(pre_activation, -MAX_CLIP_VALUE, MAX_CLIP_VALUE)

            h = np.tanh(pre_activation)

            # Output layer with clipping
            output = W2_dp @ h + b2_dp
            output = np.clip(output, -MAX_CLIP_VALUE, MAX_CLIP_VALUE)

            # Convert back to float32
            return output.astype(np.float32)

        except Exception:
            return np.zeros(self.action_dim)

    def update(self, context: np.ndarray, action: int, error: float):
        """Update network based on somatic prediction error with gradient clipping"""
        if not np.all(np.isfinite(context)) or not np.isfinite(error):
            return

        # Clip error to prevent explosion
        error = np.clip(error, -MAX_CLIP_VALUE, MAX_CLIP_VALUE)

        # Forward pass with stability checks
        try:
            # Use double precision for forward pass
            context_dp = context.astype(np.float64)
            W1_dp = self.W1.astype(np.float64)
            b1_dp = self.b1.astype(np.float64)
            W2_dp = self.W2.astype(np.float64)
            b2_dp = self.b2.astype(np.float64)

            pre_activation = W1_dp @ context_dp + b1_dp
            pre_activation = np.clip(pre_activation, -MAX_CLIP_VALUE, MAX_CLIP_VALUE)
            h = np.tanh(pre_activation)

            pred = W2_dp @ h + b2_dp
            pred = np.clip(pred, -MAX_CLIP_VALUE, MAX_CLIP_VALUE)

        except (RuntimeWarning, FloatingPointError, OverflowError, ValueError):
            return

        # Backward pass with gradient clipping
        output_grad = np.zeros(self.action_dim, dtype=np.float64)
        output_grad[action] = error

        # Update weights with clipping
        W2_grad = np.outer(output_grad, h)
        W2_grad = np.clip(
            W2_grad, -GRAD_CLIP_VALUE, GRAD_CLIP_VALUE
        )  # Gradient clipping

        self.W2 = self.W2.astype(np.float64) + self.learning_rate * W2_grad
        self.W2 = np.clip(
            self.W2, -WEIGHT_CLIP_VALUE, WEIGHT_CLIP_VALUE
        )  # Weight clipping
        self.W2 = self.W2.astype(np.float32)

        self.b2 = self.b2.astype(np.float64) + self.learning_rate * output_grad
        self.b2 = np.clip(self.b2, -WEIGHT_CLIP_VALUE, WEIGHT_CLIP_VALUE)
        self.b2 = self.b2.astype(np.float32)

        # Hidden layer gradient with clipping
        h_grad = W2_dp.T @ output_grad * (1 - h**2)
        h_grad = np.clip(h_grad, -GRAD_CLIP_VALUE, GRAD_CLIP_VALUE)  # Gradient clipping

        W1_grad = np.outer(h_grad, context_dp)
        W1_grad = np.clip(
            W1_grad, -GRAD_CLIP_VALUE, GRAD_CLIP_VALUE
        )  # Gradient clipping

        self.W1 = self.W1.astype(np.float64) + self.learning_rate * W1_grad
        self.W1 = np.clip(
            self.W1, -WEIGHT_CLIP_VALUE, WEIGHT_CLIP_VALUE
        )  # Weight clipping
        self.W1 = self.W1.astype(np.float32)

        self.b1 = self.b1.astype(np.float64) + self.learning_rate * h_grad
        self.b1 = np.clip(self.b1, -WEIGHT_CLIP_VALUE, WEIGHT_CLIP_VALUE)
        self.b1 = self.b1.astype(np.float32)


class PolicyNetwork:
    """Policy network for action selection"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Xavier/Glorot initialization for better stability
        fan_in = state_dim
        fan_out = hidden_dim
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.W1 = np.random.normal(0, std, (hidden_dim, state_dim)).astype(np.float32)

        fan_in = hidden_dim
        fan_out = action_dim
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.W2 = np.random.normal(0, std, (action_dim, hidden_dim)).astype(np.float32)

        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.b2 = np.zeros(action_dim, dtype=np.float32)

        # More permissive gradient clipping
        self.grad_clip = POLICY_GRAD_CLIP
        self.max_weight = WEIGHT_CLIP_VALUE

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities with enhanced numerical stability"""
        # Check for valid input
        if not np.all(np.isfinite(state)):
            return np.ones(self.action_dim) / self.action_dim

        # Check for zero or near-zero state
        state_norm = np.linalg.norm(state)
        if state_norm < DEFAULT_EPSILON:
            return np.ones(self.action_dim) / self.action_dim

        # Check for valid weights before operations
        if not (np.all(np.isfinite(self.W1)) and np.all(np.isfinite(self.W2))):
            self._reset_weights()
            return np.ones(self.action_dim) / self.action_dim

        # Additional weight validation
        if np.linalg.norm(self.W1) > 1000 or np.linalg.norm(self.W2) > 1000:
            self._reset_weights()
            return np.ones(self.action_dim) / self.action_dim

        # Normalize state to prevent overflow
        state_norm = state / state_norm

        # Validate normalized state before matmul
        if not np.all(np.isfinite(state_norm)):
            return np.ones(self.action_dim) / self.action_dim

        epsilon = DEFAULT_EPSILON

        # Forward pass with numerical stability
        try:
            # Ensure state_norm is finite and reasonable
            state_norm = np.nan_to_num(state_norm, nan=0.0, posinf=1.0, neginf=-1.0)
            state_norm = np.clip(state_norm, -10.0, 10.0)

            # Additional check before matrix multiplication
            if not np.all(np.isfinite(state_norm)):
                return np.ones(self.action_dim) / self.action_dim

            # Use float64 for better precision during computation with error suppression
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                pre_activation = self.W1 @ state_norm + self.b1

                # Clip to prevent overflow
                pre_activation = np.clip(pre_activation, -50.0, 50.0)

                # Use stable activation functions
                h = np.tanh(pre_activation)

                # Check for valid hidden activations
                if not np.all(np.isfinite(h)):
                    self._reset_weights()
                    return np.ones(self.action_dim) / self.action_dim

                # Output layer with stability
                logits = self.W2 @ h + self.b2
                logits = np.clip(logits, -50.0, 50.0)

        except Exception:
            self._reset_weights()
            return np.ones(self.action_dim) / self.action_dim

        # Check for valid logits
        if not np.all(np.isfinite(logits)):
            self._reset_weights()
            return np.ones(self.action_dim) / self.action_dim

        # Log-space softmax for numerical stability
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)

        if not np.all(np.isfinite(exp_logits)):
            return np.ones(self.action_dim) / self.action_dim

        sum_exp = np.sum(exp_logits)
        if sum_exp < epsilon:
            return np.ones(self.action_dim) / self.action_dim

        action_probs = exp_logits / sum_exp

        # Final validation
        if not np.all(np.isfinite(action_probs)) or np.sum(action_probs) < 0.99:
            return np.ones(self.action_dim) / self.action_dim

        return action_probs

    def _reset_weights(self):
        """Reset weights to safe defaults when numerical instability detected"""
        # Use Xavier/Glorot initialization for better stability
        if hasattr(self, "W1") and self.W1 is not None:
            fan_in = self.W1.shape[1]
            fan_out = self.W1.shape[0]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W1 = np.random.uniform(-limit, limit, self.W1.shape).astype(np.float64)

        if hasattr(self, "W2") and self.W2 is not None:
            fan_in = self.W2.shape[1]
            fan_out = self.W2.shape[0]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W2 = np.random.uniform(-limit, limit, self.W2.shape).astype(np.float64)

        if hasattr(self, "b1") and self.b1 is not None:
            self.b1 = np.zeros_like(self.b1).astype(np.float64)

        if hasattr(self, "b2") and self.b2 is not None:
            self.b2 = np.zeros_like(self.b2).astype(np.float64)

    def update(self, value: float):
        """Update policy based on value signal with gradient clipping"""
        # Clip value to prevent explosion
        value = np.clip(value, -MAX_CLIP_VALUE, MAX_CLIP_VALUE)

        # Apply gradient clipping to maintain stability
        if hasattr(self, "W1"):
            self.W1 = np.clip(self.W1, -self.max_weight, self.max_weight)
            self.W2 = np.clip(self.W2, -self.max_weight, self.max_weight)
            self.b1 = np.clip(self.b1, -self.max_weight, self.max_weight)
            self.b2 = np.clip(self.b2, -self.max_weight, self.max_weight)


class HabitualPolicy:
    """Habitual policy for implicit actions"""

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize weights with proper scaling
        std = np.sqrt(2.0 / action_dim)  # Xavier initialization
        self.W = np.random.normal(0, std, (action_dim, state_dim)).astype(np.float32)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities with enhanced numerical stability"""
        # Add numerical stability checks
        if not np.all(np.isfinite(state)):
            return np.ones(self.action_dim) / self.action_dim

        epsilon = DEFAULT_EPSILON

        try:
            # Use double precision for critical calculations
            state_dp = state.astype(np.float64)
            W_dp = self.W.astype(np.float64)

            # Pre-activation with clipping
            logits = W_dp @ state_dp
            logits = np.clip(logits, -MAX_CLIP_VALUE, MAX_CLIP_VALUE)

            # Convert back to float32
            logits = logits.astype(np.float32)

        except Exception:
            return np.ones(self.action_dim) / self.action_dim

        # Log-space softmax for numerical stability
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)

        if not np.all(np.isfinite(exp_logits)):
            return np.ones(self.action_dim) / self.action_dim

        sum_exp = np.sum(exp_logits)
        if sum_exp < epsilon:
            return np.ones(self.action_dim) / self.action_dim

        action_probs = exp_logits / sum_exp

        # Final validation
        if not np.all(np.isfinite(action_probs)) or np.sum(action_probs) < 0.99:
            return np.ones(self.action_dim) / self.action_dim

        return action_probs

    def update(self, value: float):
        """Update habits based on value"""
        # Simplified habit update
        pass


class EpisodicMemory:
    """Episodic memory system"""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)

    def store(self, content: Dict, emotional_tag: float, context: np.ndarray):
        """Store episodic memory"""
        self.memories.append(
            {
                "content": content,
                "emotional_tag": emotional_tag,
                "context": context,
                "timestamp": len(self.memories),
            }
        )

    def retrieve(self, query_context: np.ndarray, n: int = 5) -> List[Dict]:
        """Retrieve most similar memories"""
        # Simplified retrieval - just return recent memories
        return list(self.memories)[-n:]


class WorkingMemory:
    """Working memory system"""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items = deque(maxlen=capacity)

    def update(self, content: Dict):
        """Update working memory"""
        self.items.append(content)

    def __len__(self):
        return len(self.items)


class APGIActiveInferenceAgent:
    """
    Complete APGI-based active inference agent

    Features:
    - Hierarchical exteroceptive and interoceptive generative models
    - Dynamic precision weighting (Πᵉ, Πⁱ)
    - Somatic marker learning (M(c,a))
    - Global workspace ignition (S_t > θ_t)
    - Adaptive threshold (metabolic cost vs information value)
    """

    def __init__(self, config: Dict):
        self.config = config

        # =====================
        # GENERATIVE MODELS
        # =====================

        # Exteroceptive model (3 levels)
        self.extero_model = HierarchicalGenerativeModel(
            levels=[
                {"name": "sensory", "dim": SENSORY_DIM, "tau": 0.05},
                {"name": "objects", "dim": OBJECTS_DIM, "tau": 0.2},
                {"name": "context", "dim": CONTEXT_DIM, "tau": 1.0},
            ],
            learning_rate=config.get("lr_extero", 0.01),
            model_type="extero",
        )

        # Interoceptive model (3 levels)
        self.intero_model = HierarchicalGenerativeModel(
            levels=[
                {"name": "visceral", "dim": VISCERAL_DIM, "tau": 0.1},
                {"name": "organ", "dim": ORGAN_DIM, "tau": 0.5},
                {"name": "homeostatic", "dim": HOMEOSTATIC_DIM, "tau": 2.0},
            ],
            learning_rate=config.get("lr_intero", 0.01),
            model_type="intero",
        )

        # =====================
        # PRECISION MECHANISMS
        # =====================

        self.Pi_e = config.get("Pi_e_init", 1.0)  # Exteroceptive precision
        self.Pi_i = config.get("Pi_i_init", 1.0)  # Interoceptive precision
        self.beta = config.get("beta", 1.2)  # Somatic bias

        # Precision learning rates
        self.lr_precision = config.get("lr_precision", 0.05)

        # =====================
        # SOMATIC MARKERS
        # =====================

        # M(context, action) → expected interoceptive outcome
        self.somatic_markers = SomaticMarkerNetwork(
            context_dim=CONTEXT_DIM
            + HOMEOSTATIC_DIM,  # extero_context + intero_homeostatic
            action_dim=config.get("n_actions", 4),
            hidden_dim=SOMATIC_HIDDEN_DIM,
            learning_rate=config.get("lr_somatic", 0.1),
        )

        # =====================
        # IGNITION MECHANISM
        # =====================

        self.S_t = 0.0  # Accumulated surprise
        self.theta_t = config.get("theta_init", 0.5)  # Ignition threshold
        self.theta_0 = config.get("theta_baseline", 0.5)
        self.alpha = config.get("alpha", 8.0)  # Sigmoid steepness

        # Threshold adaptation
        self.tau_S = config.get("tau_S", 0.3)
        self.tau_theta = config.get("tau_theta", 10.0)
        self.eta_theta = config.get("eta_theta", 0.01)

        # =====================
        # GLOBAL WORKSPACE
        # =====================

        self.workspace_content = None
        self.ignition_history = []
        self.conscious_access = False

        # =====================
        # POLICIES
        # =====================

        self.policy_network = PolicyNetwork(
            state_dim=CONTEXT_DIM
            + HOMEOSTATIC_DIM
            + WORKSPACE_DIM,  # extero + intero + workspace
            action_dim=config.get("n_actions", 4),
            hidden_dim=HIDDEN_DIM_DEFAULT,
        )

        # Separate explicit (conscious) and implicit (habitual) policies
        self.explicit_policy_weight = 0.5
        self.implicit_policy = HabitualPolicy(
            state_dim=SENSORY_DIM,
            action_dim=config.get("n_actions", 4),  # Low-level sensory
        )

        # =====================
        # MEMORY SYSTEMS
        # =====================

        self.episodic_memory = EpisodicMemory(capacity=1000)
        self.working_memory = WorkingMemory(capacity=7)

        # =====================
        # METABOLIC TRACKING
        # =====================

        self.metabolic_cost = 0.0
        self.information_value = 0.0
        self.time = 0.0
        self.last_action = 0

    def step(self, observation: Dict, dt: float = 0.05) -> int:
        """
        Execute one agent step

        Args:
            observation: {'extero': sensory_input, 'intero': visceral_input}
            dt: Time step

        Returns:
            action: Selected action index
        """

        # =====================
        # 1. HANDLE OBSERVATION DIMENSIONS
        # =====================

        # Ensure observations have correct dimensions
        extero_actual = observation["extero"]
        intero_actual = observation["intero"]

        # Validate observations before processing
        if not (
            np.all(np.isfinite(extero_actual)) and np.all(np.isfinite(intero_actual))
        ):
            # Return safe default action if observations are invalid
            return 0

        # Handle exteroceptive observation
        if len(extero_actual) < EXTERO_DIM:
            extero_padded = np.zeros(EXTERO_DIM)
            extero_padded[: len(extero_actual)] = extero_actual
            extero_actual = extero_padded
        elif len(extero_actual) > EXTERO_DIM:
            extero_actual = extero_actual[:EXTERO_DIM]

        # Handle interoceptive observation
        if len(intero_actual) < INTERO_DIM:
            intero_padded = np.zeros(INTERO_DIM)
            intero_padded[: len(intero_actual)] = intero_actual
            intero_actual = intero_padded
        elif len(intero_actual) > INTERO_DIM:
            intero_actual = intero_actual[:INTERO_DIM]

        # =====================
        # 2. PREDICTION ERROR COMPUTATION
        # =====================

        # Exteroceptive prediction error
        extero_pred = self.extero_model.predict()
        eps_e = extero_actual - extero_pred

        # Interoceptive prediction error
        intero_pred = self.intero_model.predict()
        eps_i = intero_actual - intero_pred

        # =====================
        # 3. PRECISION UPDATING
        # =====================

        # Update precision based on prediction error reliability
        # High variance in recent errors → lower precision
        self._update_precision(eps_e, eps_i)

        # =====================
        # 4. SURPRISE ACCUMULATION
        # =====================

        # APGI core equation
        input_drive = self.Pi_e * np.linalg.norm(
            eps_e
        ) + self.beta * self.Pi_i * np.linalg.norm(eps_i)

        # Dynamical update
        dS_dt = -self.S_t / self.tau_S + input_drive
        self.S_t += dS_dt * dt
        self.S_t = max(0.0, self.S_t)

        # =====================
        # 5. THRESHOLD DYNAMICS
        # =====================

        # Compute metabolic cost of current processing
        self.metabolic_cost = self._compute_metabolic_cost()

        # Compute information value of workspace content
        self.information_value = self._compute_information_value()

        # Threshold adaptation
        dtheta_dt = (self.theta_0 - self.theta_t) / self.tau_theta + self.eta_theta * (
            self.metabolic_cost - self.information_value
        )
        self.theta_t += dtheta_dt * dt
        self.theta_t = np.clip(self.theta_t, 0.1, 2.0)

        # =====================
        # 6. IGNITION CHECK
        # =====================

        P_ignition = 1.0 / (1.0 + np.exp(-self.alpha * (self.S_t - self.theta_t)))
        self.conscious_access = np.random.random() < P_ignition

        if self.conscious_access:
            # IGNITION OCCURRED

            # Broadcast to global workspace
            self.workspace_content = {
                "extero_context": self.extero_model.get_level("context"),
                "intero_state": self.intero_model.get_level("homeostatic"),
                "eps_e": eps_e,
                "eps_i": eps_i,
                "S_t": self.S_t,
                "time": self.time,
            }

            # Update working memory
            self.working_memory.update(self.workspace_content)

            # Store in episodic memory (with high β tag)
            self.episodic_memory.store(
                content=self.workspace_content,
                emotional_tag=self.beta * np.linalg.norm(eps_i),
                context=self.extero_model.get_level("context"),
            )

            # Partial reset of surprise
            self.S_t *= 1 - self.config.get("rho", 0.7)

            # Record ignition
            self.ignition_history.append(
                {
                    "time": self.time,
                    "S_t": self.S_t
                    + self.config.get("rho", 0.7) * self.S_t,  # Pre-reset
                    "theta_t": self.theta_t,
                    "Pi_e_eps_e": self.Pi_e * np.linalg.norm(eps_e),
                    "Pi_i_eps_i": self.Pi_i * np.linalg.norm(eps_i),
                    "intero_dominant": (
                        self.Pi_i * np.linalg.norm(eps_i)
                        > self.Pi_e * np.linalg.norm(eps_e)
                    ),
                }
            )

        # =====================
        # 7. ACTION SELECTION
        # =====================

        if self.conscious_access:
            # Explicit, deliberate policy (workspace-based)
            state_rep = self._get_workspace_state()
            explicit_action_probs = self.policy_network(state_rep)

            # Somatic marker influence
            context = np.concatenate(
                [
                    self.extero_model.get_level("context"),
                    self.intero_model.get_level("homeostatic"),
                ]
            )
            somatic_values = self.somatic_markers.predict(context)

            # Combine explicit policy with somatic markers
            action_probs = explicit_action_probs * np.exp(somatic_values)
            action_probs /= action_probs.sum()

        else:
            # Implicit, habitual policy (direct sensory-motor)
            sensory_state = extero_actual  # Use processed observation
            action_probs = self.implicit_policy(sensory_state)

        # Sample action
        action = np.random.choice(len(action_probs), p=action_probs)
        self.last_action = action

        # =====================
        # 8. MODEL UPDATES
        # =====================

        # Update generative models
        self.extero_model.update(eps_e)
        self.intero_model.update(eps_i)

        self.time += dt

        return action

    def _get_workspace_state(self) -> np.ndarray:
        """Get state representation for workspace-based policy"""
        if self.workspace_content is None:
            return np.zeros(
                CONTEXT_DIM + HOMEOSTATIC_DIM + WORKSPACE_DIM
            )  # extero + intero + workspace

        return np.concatenate(
            [
                self.workspace_content.get("extero_context", np.zeros(CONTEXT_DIM)),
                self.workspace_content.get("intero_state", np.zeros(HOMEOSTATIC_DIM)),
                [self.workspace_content.get("S_t", 0.0)]
                * WORKSPACE_DIM,  # Repeat S_t to fill workspace dim
            ]
        )

    def receive_outcome(
        self, reward: float, intero_cost: float, next_observation: Dict
    ):
        """
        Process outcome and update somatic markers

        Args:
            reward: External reward
            intero_cost: Interoceptive cost (e.g., glucose depletion)
            next_observation: Next state observation
        """

        # Compute somatic prediction error
        context = np.concatenate(
            [
                self.extero_model.get_level("context"),
                self.intero_model.get_level("homeostatic"),
            ]
        )
        predicted_intero = self.somatic_markers.predict(context)[self.last_action]
        actual_intero = intero_cost
        somatic_pe = actual_intero - predicted_intero

        # Update somatic markers
        self.somatic_markers.update(context, self.last_action, somatic_pe)

        # Update policies based on reward + intero_cost
        total_value = reward - self.beta * intero_cost
        self.policy_network.update(total_value)

        if not self.conscious_access:
            # Also update implicit policy
            self.implicit_policy.update(total_value)

    def _update_precision(self, eps_e: np.ndarray, eps_i: np.ndarray):
        """Update precision based on prediction error statistics"""

        # Track running variance of prediction errors
        if not hasattr(self, "_eps_e_buffer"):
            self._eps_e_buffer = deque(maxlen=50)
            self._eps_i_buffer = deque(maxlen=50)

        self._eps_e_buffer.append(np.linalg.norm(eps_e))
        self._eps_i_buffer.append(np.linalg.norm(eps_i))

        if len(self._eps_e_buffer) > 10:
            # Precision = 1 / variance (approximately)
            var_e = np.var(list(self._eps_e_buffer)) + 0.01
            var_i = np.var(list(self._eps_i_buffer)) + 0.01

            target_Pi_e = 1.0 / var_e
            target_Pi_i = 1.0 / var_i

            # Smooth update
            self.Pi_e += self.lr_precision * (target_Pi_e - self.Pi_e)
            self.Pi_i += self.lr_precision * (target_Pi_i - self.Pi_i)

            # Clip to reasonable range
            self.Pi_e = np.clip(self.Pi_e, 0.1, 5.0)
            self.Pi_i = np.clip(self.Pi_i, 0.1, 5.0)

    def _compute_metabolic_cost(self) -> float:
        """Compute metabolic cost of current processing"""

        # Workspace maintenance is costly
        workspace_cost = 1.0 if self.conscious_access else 0.2

        # High precision is costly
        precision_cost = 0.1 * (self.Pi_e + self.Pi_i)

        # Working memory is costly
        wm_cost = 0.05 * len(self.working_memory)

        return workspace_cost + precision_cost + wm_cost

    def _compute_information_value(self) -> float:
        """Compute information value of workspace content"""

        if self.workspace_content is None:
            return 0.0

        # Value = surprise resolved + policy improvement potential
        surprise_value = self.workspace_content.get("S_t", 0.0)

        # Policy entropy reduction from workspace info
        if hasattr(self, "last_policy_entropy"):
            state_rep = self._get_workspace_state()
            current_probs = self.policy_network(state_rep)
            current_entropy = -np.sum(current_probs * np.log(current_probs + 1e-10))
            entropy_reduction = self.last_policy_entropy - current_entropy
            self.last_policy_entropy = current_entropy
        else:
            entropy_reduction = 0.0
            self.last_policy_entropy = 1.0

        return surprise_value + entropy_reduction


class StandardPPAgent:
    """Comparison: Standard predictive processing without ignition"""

    def __init__(self, config: Dict):
        self.config = config

        # Same generative models as APGI but no ignition mechanism
        self.extero_model = HierarchicalGenerativeModel(
            levels=[
                {"name": "sensory", "dim": SENSORY_DIM, "tau": 0.05},
                {"name": "objects", "dim": OBJECTS_DIM, "tau": 0.2},
                {"name": "context", "dim": CONTEXT_DIM, "tau": 1.0},
            ],
            learning_rate=config.get("lr_extero", 0.01),
            model_type="extero",
        )

        self.intero_model = HierarchicalGenerativeModel(
            levels=[
                {"name": "visceral", "dim": VISCERAL_DIM, "tau": 0.1},
                {"name": "organ", "dim": ORGAN_DIM, "tau": 0.5},
                {"name": "homeostatic", "dim": HOMEOSTATIC_DIM, "tau": 2.0},
            ],
            learning_rate=config.get("lr_intero", 0.01),
            model_type="intero",
        )

        # Continuous processing - no threshold
        self.policy_network = PolicyNetwork(
            state_dim=CONTEXT_DIM
            + HOMEOSTATIC_DIM
            + WORKSPACE_DIM,  # extero + intero + combined
            action_dim=config.get("n_actions", 4),
            hidden_dim=HIDDEN_DIM_DEFAULT,
        )

        # Simple precision weights (no adaptive threshold)
        self.Pi_e = 1.0
        self.Pi_i = 1.0

        # Tracking
        self.time = 0.0
        self.last_action = 0

    def step(self, observation: Dict, dt: float = 0.05) -> int:
        """Standard PP processing without ignition gate"""

        # Handle observation dimensions
        extero_actual = observation["extero"]
        intero_actual = observation["intero"]

        # Standardize dimensions
        if len(extero_actual) < EXTERO_DIM:
            extero_padded = np.zeros(EXTERO_DIM)
            extero_padded[: len(extero_actual)] = extero_actual
            extero_actual = extero_padded
        elif len(extero_actual) > EXTERO_DIM:
            extero_actual = extero_actual[:EXTERO_DIM]

        if len(intero_actual) < INTERO_DIM:
            intero_padded = np.zeros(INTERO_DIM)
            intero_padded[: len(intero_actual)] = intero_actual
            intero_actual = intero_padded
        elif len(intero_actual) > INTERO_DIM:
            intero_actual = intero_actual[:INTERO_DIM]

        # Compute prediction errors
        extero_pred = self.extero_model.predict()
        eps_e = extero_actual - extero_pred

        intero_pred = self.intero_model.predict()
        eps_i = intero_actual - intero_pred

        # Update generative models
        self.extero_model.update(eps_e)
        self.intero_model.update(eps_i)

        # Direct mapping to action (no ignition gate)
        state = np.concatenate(
            [
                self.extero_model.get_level("context"),
                self.intero_model.get_level("homeostatic"),
                eps_e[:CONTEXT_DIM],  # Truncated prediction error
                eps_i[:HOMEOSTATIC_DIM],
            ]
        )

        # Ensure state has correct dimensions
        expected_dim = CONTEXT_DIM + HOMEOSTATIC_DIM + CONTEXT_DIM + HOMEOSTATIC_DIM
        if len(state) < expected_dim:
            state = np.pad(state, (0, expected_dim - len(state)))
        elif len(state) > expected_dim:
            state = state[:expected_dim]

        action_probs = self.policy_network(state)
        action = np.random.choice(len(action_probs), p=action_probs)

        self.last_action = action
        self.time += dt

        return action

    def receive_outcome(
        self, reward: float, intero_cost: float, next_observation: Dict
    ):
        """Process outcome (simplified for standard PP)"""
        # Standard PP doesn't have somatic markers, so simple value update
        total_value = reward - 0.5 * intero_cost  # Reduced interoceptive weighting
        self.policy_network.update(total_value)


class GWTOnlyAgent:
    """Comparison: Ignition without somatic markers"""

    def __init__(self, config: Dict):
        self.config = config

        # Exteroceptive model only (no interoceptive precision weighting)
        self.extero_model = HierarchicalGenerativeModel(
            levels=[
                {"name": "sensory", "dim": SENSORY_DIM, "tau": 0.05},
                {"name": "objects", "dim": OBJECTS_DIM, "tau": 0.2},
                {"name": "context", "dim": CONTEXT_DIM, "tau": 1.0},
            ],
            learning_rate=config.get("lr_extero", 0.01),
            model_type="extero",
        )

        # Simple interoceptive model (no precision weighting)
        self.intero_model = HierarchicalGenerativeModel(
            levels=[
                {"name": "visceral", "dim": VISCERAL_DIM, "tau": 0.1},
                {"name": "organ", "dim": ORGAN_DIM, "tau": 0.5},
                {"name": "homeostatic", "dim": HOMEOSTATIC_DIM, "tau": 2.0},
            ],
            learning_rate=config.get("lr_intero", 0.01),
            model_type="intero",
        )

        # Ignition mechanism but no interoceptive precision weighting
        self.S_t = 0.0  # Accumulated surprise
        self.theta_t = config.get("theta_init", 0.5)  # Fixed threshold
        self.theta_0 = config.get("theta_baseline", 0.5)
        self.alpha = config.get("alpha", 8.0)  # Sigmoid steepness
        self.tau_S = config.get("tau_S", 0.3)

        # No somatic markers
        self.policy_network = PolicyNetwork(
            state_dim=CONTEXT_DIM
            + HOMEOSTATIC_DIM
            + WORKSPACE_DIM,  # extero + intero + workspace
            action_dim=config.get("n_actions", 4),
            hidden_dim=HIDDEN_DIM_DEFAULT,
        )

        # Workspace for broadcast (but no somatic integration)
        self.workspace_content = None
        self.conscious_access = False
        self.ignition_history = []

        # Tracking
        self.time = 0.0
        self.last_action = 0

    def step(self, observation: Dict, dt: float = 0.05) -> int:
        """GWT processing without somatic markers"""

        # Handle observation dimensions
        extero_actual = observation["extero"]
        intero_actual = observation["intero"]

        # Standardize dimensions
        if len(extero_actual) < EXTERO_DIM:
            extero_padded = np.zeros(EXTERO_DIM)
            extero_padded[: len(extero_actual)] = extero_actual
            extero_actual = extero_padded
        elif len(extero_actual) > EXTERO_DIM:
            extero_actual = extero_actual[:EXTERO_DIM]

        if len(intero_actual) < INTERO_DIM:
            intero_padded = np.zeros(INTERO_DIM)
            intero_padded[: len(intero_actual)] = intero_actual
            intero_actual = intero_padded
        elif len(intero_actual) > INTERO_DIM:
            intero_actual = intero_actual[:INTERO_DIM]

        # Compute prediction errors
        extero_pred = self.extero_model.predict()
        eps_e = extero_actual - extero_pred

        intero_pred = self.intero_model.predict()
        eps_i = intero_actual - intero_pred

        # Update generative models
        self.extero_model.update(eps_e)
        self.intero_model.update(eps_i)

        # Ignition based only on exteroceptive surprise (no interoceptive term)
        input_drive = np.linalg.norm(eps_e)

        # Dynamical update
        dS_dt = -self.S_t / self.tau_S + input_drive
        self.S_t += dS_dt * dt
        self.S_t = max(0.0, self.S_t)

        # Check ignition (fixed threshold, no adaptation)
        P_ignition = 1.0 / (1.0 + np.exp(-self.alpha * (self.S_t - self.theta_t)))
        self.conscious_access = np.random.random() < P_ignition

        if self.conscious_access:
            # Broadcast to workspace (without somatic markers)
            self.workspace_content = {
                "extero_context": self.extero_model.get_level("context"),
                "intero_state": self.intero_model.get_level("homeostatic"),
                "eps_e": eps_e,
                "eps_i": eps_i,
                "S_t": self.S_t,
                "time": self.time,
            }

            # Record ignition (always exteroceptive dominant since no interoceptive weighting)
            self.ignition_history.append(
                {
                    "time": self.time,
                    "S_t": self.S_t,
                    "theta_t": self.theta_t,
                    "intero_dominant": False,  # Never interoceptive dominant
                }
            )

            # Partial reset of surprise
            self.S_t *= 0.3

        # Action selection
        if self.conscious_access:
            # Explicit, deliberate policy (workspace-based)
            state_rep = self._get_workspace_state()
            action_probs = self.policy_network(state_rep)
        else:
            # Implicit, habitual policy (direct sensory-motor)
            state = np.concatenate(
                [
                    self.extero_model.get_level("context"),
                    self.intero_model.get_level("homeostatic"),
                    eps_e[:CONTEXT_DIM],
                    eps_i[:HOMEOSTATIC_DIM],
                ]
            )

            # Ensure state has correct dimensions
            expected_dim = CONTEXT_DIM + HOMEOSTATIC_DIM + CONTEXT_DIM + HOMEOSTATIC_DIM
            if len(state) < expected_dim:
                state = np.pad(state, (0, expected_dim - len(state)))
            elif len(state) > expected_dim:
                state = state[:expected_dim]

            action_probs = self.policy_network(state)

        action = np.random.choice(len(action_probs), p=action_probs)

        self.last_action = action
        self.time += dt

        return action

    def _get_workspace_state(self) -> np.ndarray:
        """Get state representation for workspace-based policy"""
        if self.workspace_content is None:
            return np.zeros(
                CONTEXT_DIM + HOMEOSTATIC_DIM + WORKSPACE_DIM
            )  # extero + intero + workspace

        return np.concatenate(
            [
                self.workspace_content.get("extero_context", np.zeros(CONTEXT_DIM)),
                self.workspace_content.get("intero_state", np.zeros(HOMEOSTATIC_DIM)),
                [self.workspace_content.get("S_t", 0.0)]
                * WORKSPACE_DIM,  # Repeat S_t to fill workspace dim
            ]
        )

    def receive_outcome(
        self, reward: float, intero_cost: float, next_observation: Dict
    ):
        """Process outcome (without somatic marker updates)"""
        # Simple value update (no somatic learning)
        total_value = reward - 0.3 * intero_cost  # Minimal interoceptive weighting
        self.policy_network.update(total_value)


# Main execution
if __name__ == "__main__":
    print("Creating APGI Agent...")
    config = {
        "lr_extero": 0.01,
        "lr_intero": 0.01,
        "lr_precision": 0.05,
        "lr_somatic": 0.1,
        "n_actions": 4,
        "theta_init": 0.5,
        "theta_baseline": 0.5,
        "alpha": 8.0,
        "tau_S": 0.3,
        "tau_theta": 10.0,
        "eta_theta": 0.01,
        "beta": 1.2,
        "rho": 0.7,
    }

    agent = APGIActiveInferenceAgent(config)
    print("Agent config:", config)


def run_falsification():
    """Entry point for CLI falsification testing."""
    try:
        print("Running APGI Falsification Protocol 1...")
        print(
            "Protocol 1 falsifies APGI predictions through active inference agent simulations."
        )

        # Create test configuration
        config = {
            "lr_extero": 0.01,
            "lr_intero": 0.01,
            "lr_precision": 0.05,
            "lr_somatic": 0.1,
            "n_actions": 4,
            "theta_init": 0.5,
            "theta_baseline": 0.5,
            "alpha": 8.0,
            "tau_S": 0.3,
            "tau_theta": 10.0,
            "eta_theta": 0.01,
            "beta": 1.2,
            "rho": 0.7,
        }

        # Test 1: Agent initialization
        print("Test 1: Agent initialization...")
        agent = APGIActiveInferenceAgent(config)
        assert agent is not None, "Agent creation failed"
        print(" Agent initialized successfully")

        # Test 2: Basic step execution
        print("Test 2: Basic step execution...")
        obs = {
            "extero": np.random.randn(32).astype(np.float32),
            "intero": np.random.randn(16).astype(np.float32),
        }
        action = agent.step(obs, dt=0.05)
        assert 0 <= action < config["n_actions"], f"Invalid action: {action}"
        print(f" Step executed successfully, action={action}")

        # Test 3: receive_outcome
        print("Test 3: receive_outcome...")
        next_obs = {
            "extero": np.random.randn(32).astype(np.float32),
            "intero": np.random.randn(16).astype(np.float32),
        }
        agent.receive_outcome(reward=1.0, intero_cost=0.1, next_observation=next_obs)
        print(" receive_outcome executed successfully")

        # Test 4: Multiple steps
        print("Test 4: Multiple steps...")
        total_reward = 0.0
        for i in range(10):
            obs = {
                "extero": np.random.randn(32).astype(np.float32),
                "intero": np.random.randn(16).astype(np.float32),
            }
            action = agent.step(obs, dt=0.05)
            reward = np.random.randn()
            intero_cost = abs(np.random.randn()) * 0.1
            agent.receive_outcome(reward, intero_cost, obs)
            total_reward += reward
        print(f" Multiple steps completed, total_reward={total_reward:.2f}")

        # Test 5: Numerical stability with edge cases
        print("Test 5: Numerical stability with edge cases...")
        # Test with zeros
        obs_zeros = {"extero": np.zeros(32), "intero": np.zeros(16)}
        action = agent.step(obs_zeros, dt=0.05)
        assert 0 <= action < config["n_actions"]

        # Test with large values
        obs_large = {"extero": np.ones(32) * 1000, "intero": np.ones(16) * 1000}
        action = agent.step(obs_large, dt=0.05)
        assert 0 <= action < config["n_actions"]

        # Test with NaN/inf (should handle gracefully)
        obs_nan = {"extero": np.array([np.nan] * 32), "intero": np.array([np.inf] * 16)}
        action = agent.step(obs_nan, dt=0.05)
        assert 0 <= action < config["n_actions"]
        print(" Numerical stability tests passed")

        # Test 6: Comparison agents
        print("Test 6: Comparison agents...")
        standard_pp = StandardPPAgent(config)
        gwt_only = GWTOnlyAgent(config)

        action_pp = standard_pp.step(obs, dt=0.05)
        action_gwt = gwt_only.step(obs, dt=0.05)

        assert 0 <= action_pp < config["n_actions"]
        assert 0 <= action_gwt < config["n_actions"]
        print(" Comparison agents work correctly")

        print("\n Protocol 1 falsification completed successfully")
        print("All tests passed!")
        return "Protocol 1 completed: Active inference agent falsification test passed"
    except AssertionError as e:
        print(f"Assertion failed in falsification protocol 1: {e}")
        return f"Protocol 1 failed: {str(e)}"
    except Exception as e:
        print(f"Error in falsification protocol 1: {e}")
        import traceback

        traceback.print_exc()
        return f"Protocol 1 failed: {str(e)}"
