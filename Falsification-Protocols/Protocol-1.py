import numpy as np
from collections import deque
from typing import Dict, List, Tuple


class HierarchicalGenerativeModel:
    """Hierarchical generative model with multiple levels"""
    
    def __init__(self, levels: List[Dict], learning_rate: float = 0.01, model_type: str = "extero"):
        self.levels = levels
        self.learning_rate = learning_rate
        self.model_type = model_type  # "extero" or "intero"
        self.states = {}
        self.weights = {}
        
        # Initialize each level
        for level in levels:
            name = level['name']
            dim = level['dim']
            self.states[name] = np.zeros(dim)
            # Simple weight matrix for predictions
            self.weights[name] = np.random.normal(0, 0.1, (dim, dim))
    
    def predict(self) -> np.ndarray:
        """Generate prediction from top level"""
        top_level = self.levels[-1]['name']
        pred = self.states[top_level]
        
        # Ensure prediction matches expected input size based on model type
        if self.model_type == "extero":
            target_size = 32
        else:  # intero
            target_size = 16
            
        if len(pred) < target_size:
            # Pad to match input size
            padded_pred = np.zeros(target_size)
            padded_pred[:len(pred)] = pred
            return padded_pred
        elif len(pred) > target_size:
            # Truncate if too large
            return pred[:target_size]
        return pred
    
    def update(self, error: np.ndarray):
        """Update model with prediction error"""
        # Simple gradient descent update
        for level_name in self.states.keys():
            self.states[level_name] += self.learning_rate * error[:len(self.states[level_name])]
    
    def get_level(self, level_name: str) -> np.ndarray:
        """Get state of specific level"""
        return self.states.get(level_name, np.zeros(1))
    
    def get_all_levels(self) -> np.ndarray:
        """Get all levels concatenated"""
        return np.concatenate([self.states[level['name']] for level in self.levels])


class SomaticMarkerNetwork:
    """Somatic marker network for interoceptive predictions"""
    
    def __init__(self, context_dim: int, action_dim: int, hidden_dim: int, learning_rate: float):
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Simple neural network weights
        self.W1 = np.random.normal(0, 0.1, (hidden_dim, context_dim))
        self.W2 = np.random.normal(0, 0.1, (action_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(action_dim)
    
    def predict(self, context: np.ndarray) -> np.ndarray:
        """Predict interoceptive outcomes for all actions"""
        h = np.tanh(self.W1 @ context + self.b1)
        return self.W2 @ h + self.b2
    
    def update(self, context: np.ndarray, action: int, error: float):
        """Update network based on somatic prediction error"""
        # Forward pass
        h = np.tanh(self.W1 @ context + self.b1)
        pred = self.W2 @ h + self.b2
        
        # Backward pass (simplified)
        output_grad = np.zeros(self.action_dim)
        output_grad[action] = error
        
        # Update weights
        self.W2 += self.learning_rate * np.outer(output_grad, h)
        self.b2 += self.learning_rate * output_grad
        
        # Hidden layer gradient
        h_grad = self.W2.T @ output_grad * (1 - h**2)
        self.W1 += self.learning_rate * np.outer(h_grad, context)
        self.b1 += self.learning_rate * h_grad


class PolicyNetwork:
    """Policy network for action selection"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simple neural network weights
        self.W1 = np.random.normal(0, 0.1, (hidden_dim, state_dim))
        self.W2 = np.random.normal(0, 0.1, (action_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(action_dim)
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities"""
        h = np.tanh(self.W1 @ state + self.b1)
        logits = self.W2 @ h + self.b2
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def update(self, value: float):
        """Update policy based on value signal (simplified REINFORCE)"""
        # This is a placeholder - in practice would use more sophisticated RL
        pass


class HabitualPolicy:
    """Habitual policy for implicit actions"""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simple habit weights
        self.W = np.random.normal(0, 0.1, (action_dim, state_dim))
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities"""
        logits = self.W @ state
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
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
        self.memories.append({
            'content': content,
            'emotional_tag': emotional_tag,
            'context': context,
            'timestamp': len(self.memories)
        })
    
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
                {'name': 'sensory', 'dim': 32, 'tau': 0.05},
                {'name': 'objects', 'dim': 16, 'tau': 0.2},
                {'name': 'context', 'dim': 8, 'tau': 1.0}
            ],
            learning_rate=config.get('lr_extero', 0.01),
            model_type="extero"
        )
        
        # Interoceptive model (3 levels)
        self.intero_model = HierarchicalGenerativeModel(
            levels=[
                {'name': 'visceral', 'dim': 16, 'tau': 0.1},
                {'name': 'organ', 'dim': 8, 'tau': 0.5},
                {'name': 'homeostatic', 'dim': 4, 'tau': 2.0}
            ],
            learning_rate=config.get('lr_intero', 0.01),
            model_type="intero"
        )
        
        # =====================
        # PRECISION MECHANISMS
        # =====================
        
        self.Pi_e = config.get('Pi_e_init', 1.0)  # Exteroceptive precision
        self.Pi_i = config.get('Pi_i_init', 1.0)  # Interoceptive precision
        self.beta = config.get('beta', 1.2)       # Somatic bias
        
        # Precision learning rates
        self.lr_precision = config.get('lr_precision', 0.05)
        
        # =====================
        # SOMATIC MARKERS
        # =====================
        
        # M(context, action) → expected interoceptive outcome
        self.somatic_markers = SomaticMarkerNetwork(
            context_dim=8 + 4,  # extero_context + intero_homeostatic
            action_dim=config.get('n_actions', 4),
            hidden_dim=32,
            learning_rate=config.get('lr_somatic', 0.1)
        )
        
        # =====================
        # IGNITION MECHANISM
        # =====================
        
        self.S_t = 0.0          # Accumulated surprise
        self.theta_t = config.get('theta_init', 0.5)  # Ignition threshold
        self.theta_0 = config.get('theta_baseline', 0.5)
        self.alpha = config.get('alpha', 8.0)  # Sigmoid steepness
        
        # Threshold adaptation
        self.tau_S = config.get('tau_S', 0.3)
        self.tau_theta = config.get('tau_theta', 10.0)
        self.eta_theta = config.get('eta_theta', 0.01)
        
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
            state_dim=8 + 4 + 8,  # extero + intero + workspace
            action_dim=config.get('n_actions', 4),
            hidden_dim=64
        )
        
        # Separate explicit (conscious) and implicit (habitual) policies
        self.explicit_policy_weight = 0.5
        self.implicit_policy = HabitualPolicy(
            state_dim=32,  # Low-level sensory
            action_dim=config.get('n_actions', 4)
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
        extero_actual = observation['extero']
        intero_actual = observation['intero']
        
        # Handle exteroceptive observation
        if len(extero_actual) < 32:
            extero_padded = np.zeros(32)
            extero_padded[:len(extero_actual)] = extero_actual
            extero_actual = extero_padded
        elif len(extero_actual) > 32:
            extero_actual = extero_actual[:32]
            
        # Handle interoceptive observation
        if len(intero_actual) < 16:
            intero_padded = np.zeros(16)
            intero_padded[:len(intero_actual)] = intero_actual
            intero_actual = intero_padded
        elif len(intero_actual) > 16:
            intero_actual = intero_actual[:16]
        
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
        input_drive = (self.Pi_e * np.linalg.norm(eps_e) + 
                      self.beta * self.Pi_i * np.linalg.norm(eps_i))
        
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
        dtheta_dt = ((self.theta_0 - self.theta_t) / self.tau_theta +
                    self.eta_theta * (self.metabolic_cost - self.information_value))
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
                'extero_context': self.extero_model.get_level('context'),
                'intero_state': self.intero_model.get_level('homeostatic'),
                'eps_e': eps_e,
                'eps_i': eps_i,
                'S_t': self.S_t,
                'time': self.time
            }
            
            # Update working memory
            self.working_memory.update(self.workspace_content)
            
            # Store in episodic memory (with high β tag)
            self.episodic_memory.store(
                content=self.workspace_content,
                emotional_tag=self.beta * np.linalg.norm(eps_i),
                context=self.extero_model.get_level('context')
            )
            
            # Partial reset of surprise
            self.S_t *= (1 - self.config.get('rho', 0.7))
            
            # Record ignition
            self.ignition_history.append({
                'time': self.time,
                'S_t': self.S_t + self.config.get('rho', 0.7) * self.S_t,  # Pre-reset
                'theta_t': self.theta_t,
                'Pi_e_eps_e': self.Pi_e * np.linalg.norm(eps_e),
                'Pi_i_eps_i': self.Pi_i * np.linalg.norm(eps_i),
                'intero_dominant': (self.Pi_i * np.linalg.norm(eps_i) > 
                                   self.Pi_e * np.linalg.norm(eps_e))
            })
        
        # =====================
        # 7. ACTION SELECTION
        # =====================
        
        if self.conscious_access:
            # Explicit, deliberate policy (workspace-based)
            state_rep = self._get_workspace_state()
            explicit_action_probs = self.policy_network(state_rep)
            
            # Somatic marker influence
            context = np.concatenate([
                self.extero_model.get_level('context'),
                self.intero_model.get_level('homeostatic')
            ])
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
            return np.zeros(8 + 4 + 8)  # extero + intero + workspace
        
        return np.concatenate([
            self.workspace_content.get('extero_context', np.zeros(8)),
            self.workspace_content.get('intero_state', np.zeros(4)),
            [self.workspace_content.get('S_t', 0.0)] * 8  # Repeat S_t to fill workspace dim
        ])
    
    def receive_outcome(self, reward: float, intero_cost: float, 
                       next_observation: Dict):
        """
        Process outcome and update somatic markers
        
        Args:
            reward: External reward
            intero_cost: Interoceptive cost (e.g., glucose depletion)
            next_observation: Next state observation
        """
        
        # Compute somatic prediction error
        context = np.concatenate([
            self.extero_model.get_level('context'),
            self.intero_model.get_level('homeostatic')
        ])
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
        if not hasattr(self, '_eps_e_buffer'):
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
        surprise_value = self.workspace_content.get('S_t', 0.0)
        
        # Policy entropy reduction from workspace info
        if hasattr(self, 'last_policy_entropy'):
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
        # Same generative models, no ignition mechanism
        self.extero_model = HierarchicalGenerativeModel(...)
        self.intero_model = HierarchicalGenerativeModel(...)
        
        # Continuous processing - no threshold
        self.policy_network = PolicyNetwork(...)
        
    def step(self, observation: Dict, dt: float = 0.05) -> int:
        # Compute prediction errors
        eps_e = observation['extero'] - self.extero_model.predict()
        eps_i = observation['intero'] - self.intero_model.predict()
        
        # Direct mapping to action (no ignition gate)
        state = np.concatenate([
            self.extero_model.get_all_levels(),
            self.intero_model.get_all_levels(),
            eps_e[:8], eps_i[:4]
        ])
        
        action_probs = self.policy_network(state)
        return np.random.choice(len(action_probs), p=action_probs)


class GWTOnlyAgent:
    """Comparison: Ignition without somatic markers"""
    
    def __init__(self, config: Dict):
        self.extero_model = HierarchicalGenerativeModel(...)
        
        # Ignition but no interoceptive precision weighting
        self.S_t = 0.0
        self.theta_t = config.get('theta_init', 0.5)
        
        # No somatic markers
        self.policy_network = PolicyNetwork(...)
        
    def step(self, observation: Dict, dt: float = 0.05) -> int:
        eps_e = observation['extero'] - self.extero_model.predict()
        
        # Ignition based only on external surprise
        self.S_t = np.linalg.norm(eps_e)  # No interoceptive term
        
        if self.S_t > self.theta_t:
            # Broadcast and use explicit policy
            action = np.random.choice(self.config['n_actions'])
        else:
            # Use habitual policy
            action = np.random.choice(self.config['n_actions'])
        
        return action


# Main execution
if __name__ == "__main__":
    print("Creating APGI Agent...")
    config = {
        'lr_extero': 0.01,
        'lr_intero': 0.01,
        'lr_precision': 0.05,
        'lr_somatic': 0.1,
        'n_actions': 4,
        'theta_init': 0.5,
        'theta_baseline': 0.5,
        'alpha': 8.0,
        'tau_S': 0.3,
        'tau_theta': 10.0,
        'eta_theta': 0.01,
        'beta': 1.2,
        'rho': 0.7
    }
    
    agent = APGIActiveInferenceAgent(config)
    print("Agent config:", config)
    print("=== Protocol completed successfully ===")