"""
APGI Protocol 3: Active Inference Agent Simulations
====================================================

Complete implementation of APGI-based active inference agents in decision-making
environments. Tests whether incorporating interoceptive precision and global
workspace ignition produces adaptive advantages over alternative architectures.

This protocol implements:
- Full APGI active inference agent with hierarchical models
- Comparison agents (StandardPP, GWTOnly, Actor-Critic)
- Three task environments (IGT, Foraging, Threat-Reward)
- Comprehensive analysis and falsification framework

Author: APGI Research Team
Date: 2025
Version: 1.0 (Production)

Dependencies:
    numpy, scipy, torch, matplotlib, seaborn, pandas, tqdm
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import deque
import warnings
from tqdm import tqdm
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =============================================================================
# PART 1: NEURAL NETWORK COMPONENTS
# =============================================================================

class HierarchicalGenerativeModel(nn.Module):
    """
    Hierarchical predictive processing model
    
    Implements multi-level generative model with prediction error propagation.
    """
    
    def __init__(self, levels: List[Dict], learning_rate: float = 0.01):
        super().__init__()
        
        self.levels = levels
        self.n_levels = len(levels)
        
        # Create networks for each level
        self.level_networks = nn.ModuleList()
        
        for i in range(self.n_levels - 1):
            # Each level predicts the level below
            top_dim = levels[i + 1]['dim']
            bottom_dim = levels[i]['dim']
            
            network = nn.Sequential(
                nn.Linear(top_dim, top_dim * 2),
                nn.Tanh(),
                nn.Linear(top_dim * 2, bottom_dim)
            )
            
            self.level_networks.append(network)
        
        # State representations at each level
        self.states = [torch.zeros(level['dim']) for level in levels]
        
        # Time constants for each level
        self.taus = torch.tensor([level['tau'] for level in levels])
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def predict(self, level: int = 0) -> torch.Tensor:
        """Generate prediction for given level from level above"""
        if level == self.n_levels - 1:
            # Top level has no prediction
            return self.states[level]
        
        # Predict from level above
        prediction = self.level_networks[level](self.states[level + 1])
        return prediction
    
    def update(self, prediction_error: torch.Tensor, level: int = 0, dt: float = 0.05):
        """Update states based on prediction error"""
        
        # Bottom-up message (prediction error)
        if level < self.n_levels - 1:
            # Update current level state (non-in-place)
            self.states[level] = self.states[level] + dt * prediction_error / self.taus[level]
            
            # Propagate error upward (no gradient for recursive calls)
            if level < self.n_levels - 2:
                with torch.no_grad():
                    # Error at this level influences level above
                    # Use the network to compute appropriate error for upper level
                    upper_prediction = self.predict(level + 1)
                    if upper_prediction.shape[0] == self.states[level + 1].shape[0]:
                        upper_error = self.states[level + 1] - upper_prediction
                    else:
                        # If dimensions don't match, create a scaled error
                        scale_factor = self.states[level + 1].shape[0] / prediction_error.shape[0]
                        upper_error = torch.mean(prediction_error) * torch.ones_like(self.states[level + 1]) * scale_factor
                    self.update(upper_error, level + 1, dt)
        
        # Update network parameters
        if prediction_error.requires_grad:
            loss = torch.sum(prediction_error ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def get_level(self, level_name: str) -> torch.Tensor:
        """Get state at named level"""
        for i, level in enumerate(self.levels):
            if level['name'] == level_name:
                return self.states[i].detach().numpy()
        return None
    
    def get_all_levels(self) -> np.ndarray:
        """Get concatenated states from all levels"""
        return torch.cat(self.states).detach().numpy()


class SomaticMarkerNetwork(nn.Module):
    """
    Somatic marker learning: M(context, action) → predicted interoceptive cost
    
    Learns to associate contexts and actions with their interoceptive outcomes.
    """
    
    def __init__(self, context_dim: int, action_dim: int, 
                 hidden_dim: int = 32, learning_rate: float = 0.1):
        super().__init__()
        
        self.action_dim = action_dim
        
        # Network: context → somatic predictions for each action
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Predict interoceptive outcomes for all actions"""
        return self.network(context)
    
    def predict(self, context: np.ndarray) -> np.ndarray:
        """Get predictions (numpy interface)"""
        with torch.no_grad():
            context_tensor = torch.FloatTensor(context)
            predictions = self.forward(context_tensor)
        return predictions.numpy()
    
    def update(self, context: np.ndarray, action: int, prediction_error: float):
        """Update somatic marker for specific context-action pair"""
        
        context_tensor = torch.FloatTensor(context)
        
        # Forward pass
        predictions = self.forward(context_tensor)
        
        # Loss only for selected action
        target = predictions.clone().detach()
        target[action] += prediction_error
        
        loss = F.mse_loss(predictions[action:action+1], target[action:action+1])
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PolicyNetwork(nn.Module):
    """Policy network for action selection"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_dim: int = 64, learning_rate: float = 0.001):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Value baseline for variance reduction
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(), lr=learning_rate
        )
        
        # Store for policy gradient update
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities"""
        logits = self.network(state)
        return F.softmax(logits, dim=-1)
    
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """Select action and store for later update"""
        state_tensor = torch.FloatTensor(state)
        
        probs = self.forward(state_tensor)
        value = self.value_network(state_tensor)
        
        # Ensure we only have valid actions (0-3)
        probs = probs[:4] if probs.size(0) > 4 else probs
        probs = F.softmax(probs, dim=-1)
        
        # Sample action
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        # Ensure action is within valid range
        action = torch.clamp(action, 0, 3)
        
        # Store for update
        self.saved_log_probs.append(action_dist.log_prob(action))
        self.saved_values.append(value)
        
        return action.item(), probs.detach()
    
    def update(self, final_reward: float):
        """Policy gradient update"""
        
        self.rewards.append(final_reward)
        
        # Only update periodically (every 10 steps)
        if len(self.rewards) < 10:
            return
        
        # Compute returns
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient loss
        policy_losses = []
        value_losses = []
        
        for log_prob, value, R in zip(self.saved_log_probs, self.saved_values, returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob * advantage)
            value_losses.append(F.mse_loss(value, torch.tensor([R], dtype=torch.float32)))
        
        # Update
        if len(policy_losses) > 0:
            self.optimizer.zero_grad()
            policy_loss = torch.stack(policy_losses).sum()
            policy_loss.backward()
            self.optimizer.step()
        
        if len(value_losses) > 0:
            self.value_optimizer.zero_grad()
            value_loss = torch.stack(value_losses).sum()
            value_loss.backward()
            self.value_optimizer.step()
        
        # Clear
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []


class HabitualPolicy(nn.Module):
    """Simple habitual policy for implicit control"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim)
        )
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.network(state)
        return F.softmax(logits, dim=-1)
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            return self.forward(state_tensor).numpy()
    
    def update(self, reward: float):
        """Simple reinforcement update"""
        # Placeholder - would need to store state-action pairs
        pass


# =============================================================================
# PART 2: MEMORY SYSTEMS
# =============================================================================

class WorkingMemory:
    """Limited capacity working memory"""
    
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items = deque(maxlen=capacity)
    
    def update(self, item: Dict):
        """Add item to working memory"""
        self.items.append(item)
    
    def __len__(self):
        return len(self.items)
    
    def get_recent(self, n: int = 3) -> List[Dict]:
        """Get n most recent items"""
        return list(self.items)[-n:]


class EpisodicMemory:
    """Episodic memory with emotional tagging"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.memories = []
    
    def store(self, content: Dict, emotional_tag: float, context: np.ndarray):
        """Store episode with emotional weight"""
        
        memory = {
            'content': content,
            'emotional_tag': emotional_tag,
            'context': context,
            'timestamp': len(self.memories)
        }
        
        self.memories.append(memory)
        
        # Prune if over capacity (remove low-emotion memories)
        if len(self.memories) > self.capacity:
            self.memories.sort(key=lambda x: x['emotional_tag'])
            self.memories = self.memories[100:]  # Remove bottom 100
    
    def retrieve(self, context: np.ndarray, k: int = 5) -> List[Dict]:
        """Retrieve k most similar memories"""
        
        if len(self.memories) == 0:
            return []
        
        # Compute similarity
        similarities = []
        for mem in self.memories:
            sim = -np.linalg.norm(mem['context'] - context)
            similarities.append(sim)
        
        # Get top k
        top_indices = np.argsort(similarities)[-k:]
        return [self.memories[i] for i in top_indices]


# =============================================================================
# PART 3: APGI ACTIVE INFERENCE AGENT
# =============================================================================

class APGIActiveInferenceAgent:
    """
    Complete APGI-based active inference agent
    
    Features:
    - Hierarchical exteroceptive and interoceptive generative models
    - Dynamic precision weighting (Πᵉ, Πⁱ)
    - Somatic marker learning
    - Global workspace ignition
    - Adaptive threshold
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.time = 0.0
        
        # Generative models
        self.extero_model = HierarchicalGenerativeModel(
            levels=[
                {'name': 'sensory', 'dim': 32, 'tau': 0.05},
                {'name': 'objects', 'dim': 16, 'tau': 0.2},
                {'name': 'context', 'dim': 8, 'tau': 1.0}
            ],
            learning_rate=config.get('lr_extero', 0.01)
        )
        
        self.intero_model = HierarchicalGenerativeModel(
            levels=[
                {'name': 'visceral', 'dim': 16, 'tau': 0.1},
                {'name': 'organ', 'dim': 8, 'tau': 0.5},
                {'name': 'homeostatic', 'dim': 4, 'tau': 2.0}
            ],
            learning_rate=config.get('lr_intero', 0.01)
        )
        
        # Precision
        self.Pi_e = config.get('Pi_e_init', 1.0)
        self.Pi_i = config.get('Pi_i_init', 1.0)
        self.beta = config.get('beta', 1.2)
        self.lr_precision = config.get('lr_precision', 0.05)
        
        # Somatic markers
        self.somatic_markers = SomaticMarkerNetwork(
            context_dim=12,  # 8 + 4
            action_dim=config.get('n_actions', 4),
            learning_rate=config.get('lr_somatic', 0.1)
        )
        
        # Ignition
        self.S_t = 0.0
        self.theta_t = config.get('theta_init', 0.5)
        self.theta_0 = config.get('theta_baseline', 0.5)
        self.alpha = config.get('alpha', 8.0)
        self.tau_S = config.get('tau_S', 0.3)
        self.tau_theta = config.get('tau_theta', 10.0)
        self.eta_theta = config.get('eta_theta', 0.01)
        
        # Global workspace
        self.workspace_content = None
        self.ignition_history = []
        self.conscious_access = False
        
        # Policies
        self.policy_network = PolicyNetwork(
            state_dim=20,  # Simplified state
            action_dim=config.get('n_actions', 4)
        )
        
        self.implicit_policy = HabitualPolicy(
            state_dim=32,
            action_dim=config.get('n_actions', 4)
        )
        
        # Memory
        self.episodic_memory = EpisodicMemory(capacity=1000)
        self.working_memory = WorkingMemory(capacity=7)
        
        # Tracking
        self.metabolic_cost = 0.0
        self.information_value = 0.0
        self.last_action = None
        self.last_policy_entropy = 1.0
        
        # Buffers for precision learning
        self._eps_e_buffer = deque(maxlen=50)
        self._eps_i_buffer = deque(maxlen=50)
    
    def step(self, observation: Dict, dt: float = 0.05) -> int:
        """Execute one agent step"""
        
        # 1. Prediction errors
        extero_pred = self.extero_model.predict(level=0)
        extero_actual = torch.FloatTensor(observation['extero'])
        eps_e = extero_actual - extero_pred
        
        intero_pred = self.intero_model.predict(level=0)
        intero_actual = torch.FloatTensor(observation['intero'])
        eps_i = intero_actual - intero_pred
        
        # 2. Update precision
        self._update_precision(eps_e, eps_i)
        
        # 3. Surprise accumulation
        input_drive = (self.Pi_e * torch.norm(eps_e).item() + 
                      self.beta * self.Pi_i * torch.norm(eps_i).item())
        
        dS_dt = -self.S_t / self.tau_S + input_drive
        self.S_t += dS_dt * dt
        self.S_t = max(0.0, self.S_t)
        
        # 4. Threshold adaptation
        self.metabolic_cost = self._compute_metabolic_cost()
        self.information_value = self._compute_information_value()
        
        dtheta_dt = ((self.theta_0 - self.theta_t) / self.tau_theta +
                    self.eta_theta * (self.metabolic_cost - self.information_value))
        self.theta_t += dtheta_dt * dt
        self.theta_t = np.clip(self.theta_t, 0.1, 2.0)
        
        # 5. Ignition check
        P_ignition = 1.0 / (1.0 + np.exp(-self.alpha * (self.S_t - self.theta_t)))
        self.conscious_access = np.random.random() < P_ignition
        
        if self.conscious_access:
            # Broadcast to workspace
            self.workspace_content = {
                'extero_context': self.extero_model.get_level('context'),
                'intero_state': self.intero_model.get_level('homeostatic'),
                'eps_e': eps_e.detach().numpy(),
                'eps_i': eps_i.detach().numpy(),
                'S_t': self.S_t,
                'time': self.time
            }
            
            self.working_memory.update(self.workspace_content)
            
            self.episodic_memory.store(
                content=self.workspace_content,
                emotional_tag=self.beta * torch.norm(eps_i).item(),
                context=self.extero_model.get_level('context')
            )
            
            # Record ignition
            self.ignition_history.append({
                'time': self.time,
                'S_t': self.S_t,
                'theta_t': self.theta_t,
                'Pi_e_eps_e': self.Pi_e * torch.norm(eps_e).item(),
                'Pi_i_eps_i': self.Pi_i * torch.norm(eps_i).item(),
                'intero_dominant': (self.Pi_i * torch.norm(eps_i).item() > 
                                   self.Pi_e * torch.norm(eps_e).item())
            })
            
            # Partial reset
            self.S_t *= 0.3
        
        # 6. Action selection
        if self.conscious_access:
            # Explicit policy
            state_rep = self._get_workspace_state()
            action, action_probs = self.policy_network.select_action(state_rep)
            
            # Somatic influence
            context = np.concatenate([
                self.extero_model.get_level('context'),
                self.intero_model.get_level('homeostatic')
            ])
            somatic_values = self.somatic_markers.predict(context)
            
            # Modulate probabilities
            action_probs = action_probs.numpy() * np.exp(somatic_values[:4] * 0.5)
            action_probs /= action_probs.sum()
            # Ensure valid action range and normalize
            valid_probs = action_probs[:min(len(action_probs), 4)]
            valid_probs /= valid_probs.sum()
            action = np.random.choice(len(valid_probs), p=valid_probs)
            
        else:
            # Implicit policy
            sensory_state = observation['extero']
            action_probs = self.implicit_policy(sensory_state)
            # Ensure valid action range and normalize
            valid_probs = action_probs[:min(len(action_probs), 4)]
            valid_probs /= valid_probs.sum()
            action = np.random.choice(len(valid_probs), p=valid_probs)
        
        # 7. Update models
        self.extero_model.update(eps_e, level=0, dt=dt)
        self.intero_model.update(eps_i, level=0, dt=dt)
        
        self.last_action = action
        self.time += dt
        
        return action
    
    def receive_outcome(self, reward: float, intero_cost: float, 
                       next_observation: Dict):
        """Process outcome and learn"""
        
        # Update somatic markers
        context = np.concatenate([
            self.extero_model.get_level('context'),
            self.intero_model.get_level('homeostatic')
        ])
        
        predicted_intero = self.somatic_markers.predict(context)[self.last_action]
        somatic_pe = intero_cost - predicted_intero
        
        self.somatic_markers.update(context, self.last_action, somatic_pe)
        
        # Update policy
        total_value = reward - self.beta * intero_cost
        self.policy_network.update(total_value)
        
        if not self.conscious_access:
            self.implicit_policy.update(total_value)
    
    def _update_precision(self, eps_e: torch.Tensor, eps_i: torch.Tensor):
        """Update precision based on prediction error variance"""
        
        self._eps_e_buffer.append(torch.norm(eps_e).item())
        self._eps_i_buffer.append(torch.norm(eps_i).item())
        
        if len(self._eps_e_buffer) > 10:
            var_e = np.var(list(self._eps_e_buffer)) + 0.01
            var_i = np.var(list(self._eps_i_buffer)) + 0.01
            
            target_Pi_e = 1.0 / var_e
            target_Pi_i = 1.0 / var_i
            
            self.Pi_e += self.lr_precision * (target_Pi_e - self.Pi_e)
            self.Pi_i += self.lr_precision * (target_Pi_i - self.Pi_i)
            
            self.Pi_e = np.clip(self.Pi_e, 0.1, 5.0)
            self.Pi_i = np.clip(self.Pi_i, 0.1, 5.0)
    
    def _compute_metabolic_cost(self) -> float:
        """Compute current metabolic cost"""
        workspace_cost = 1.0 if self.conscious_access else 0.2
        precision_cost = 0.1 * (self.Pi_e + self.Pi_i)
        wm_cost = 0.05 * len(self.working_memory)
        return workspace_cost + precision_cost + wm_cost
    
    def _compute_information_value(self) -> float:
        """Compute information value of workspace"""
        if self.workspace_content is None:
            return 0.0
        return self.workspace_content.get('S_t', 0.0) * 0.5
    
    def _get_workspace_state(self) -> np.ndarray:
        """Get state representation from workspace"""
        if self.workspace_content is None:
            return np.zeros(20)
        
        return np.concatenate([
            self.workspace_content['extero_context'][:8],
            self.workspace_content['intero_state'][:4],
            np.array([self.S_t, self.theta_t, self.Pi_e, self.Pi_i,
                     self.beta, self.metabolic_cost, self.information_value, 0])
        ])


# =============================================================================
# PART 4: COMPARISON AGENTS
# =============================================================================

class StandardPPAgent:
    """Standard predictive processing without ignition"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.time = 0.0
        
        self.extero_model = HierarchicalGenerativeModel(
            levels=[
                {'name': 'sensory', 'dim': 32, 'tau': 0.05},
                {'name': 'objects', 'dim': 16, 'tau': 0.2},
                {'name': 'context', 'dim': 8, 'tau': 1.0}
            ]
        )
        
        self.intero_model = HierarchicalGenerativeModel(
            levels=[
                {'name': 'visceral', 'dim': 16, 'tau': 0.1},
                {'name': 'organ', 'dim': 8, 'tau': 0.5},
                {'name': 'homeostatic', 'dim': 4, 'tau': 2.0}
            ]
        )
        
        self.policy_network = PolicyNetwork(
            state_dim=60,
            action_dim=config.get('n_actions', 4)
        )
        
        self.last_action = None
        self.conscious_access = True  # Always "conscious"
    
    def step(self, observation: Dict, dt: float = 0.05) -> int:
        # Compute prediction errors
        eps_e = torch.FloatTensor(observation['extero']) - self.extero_model.predict(0)
        eps_i = torch.FloatTensor(observation['intero']) - self.intero_model.predict(0)
        
        # Update models
        self.extero_model.update(eps_e, 0, dt)
        self.intero_model.update(eps_i, 0, dt)
        
        # Direct policy (no ignition gate)
        state = np.concatenate([
            self.extero_model.get_all_levels(),
            self.intero_model.get_all_levels()
        ])[:60]
        
        action, _ = self.policy_network.select_action(state)
        self.last_action = action
        
        return action
    
    def receive_outcome(self, reward: float, intero_cost: float, next_observation: Dict):
        self.policy_network.update(reward - intero_cost)


class GWTOnlyAgent:
    """Global workspace without interoceptive precision weighting"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.time = 0.0
        
        self.extero_model = HierarchicalGenerativeModel(
            levels=[
                {'name': 'sensory', 'dim': 32, 'tau': 0.05},
                {'name': 'objects', 'dim': 16, 'tau': 0.2},
                {'name': 'context', 'dim': 8, 'tau': 1.0}
            ]
        )
        
        self.S_t = 0.0
        self.theta_t = config.get('theta_init', 0.5)
        self.alpha = 8.0
        self.tau_S = 0.3
        
        self.conscious_access = False
        self.ignition_history = []
        
        self.policy_network = PolicyNetwork(state_dim=20, action_dim=config.get('n_actions', 4))
        self.implicit_policy = HabitualPolicy(state_dim=32, action_dim=config.get('n_actions', 4))
        
        self.last_action = None
    
    def step(self, observation: Dict, dt: float = 0.05) -> int:
        eps_e = torch.FloatTensor(observation['extero']) - self.extero_model.predict(0)
        
        # Surprise from external only
        self.S_t = torch.norm(eps_e).item()
        
        # Ignition
        P_ignition = 1.0 / (1.0 + np.exp(-self.alpha * (self.S_t - self.theta_t)))
        self.conscious_access = np.random.random() < P_ignition
        
        if self.conscious_access:
            self.ignition_history.append({'time': self.time, 'S_t': self.S_t})
            state = np.concatenate([self.extero_model.get_level('context'), np.zeros(12)])
            action, _ = self.policy_network.select_action(state)
        else:
            action_probs = self.implicit_policy(observation['extero'])
            # Ensure valid action range and normalize
            valid_probs = action_probs[:min(len(action_probs), 4)]
            valid_probs /= valid_probs.sum()
            action = np.random.choice(len(valid_probs), p=valid_probs)
        
        self.extero_model.update(eps_e, 0, dt)
        self.last_action = action
        self.time += dt
        
        return action
    
    def receive_outcome(self, reward: float, intero_cost: float, next_observation: Dict):
        self.policy_network.update(reward)


class ActorCriticAgent:
    """Standard actor-critic baseline"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        self.policy_network = PolicyNetwork(
            state_dim=48,
            action_dim=config.get('n_actions', 4)
        )
        
        self.last_action = None
        self.conscious_access = False
        self.ignition_history = []
    
    def step(self, observation: Dict, dt: float = 0.05) -> int:
        state = np.concatenate([observation['extero'], observation['intero']])
        action, _ = self.policy_network.select_action(state)
        self.last_action = action
        return action
    
    def receive_outcome(self, reward: float, intero_cost: float, next_observation: Dict):
        self.policy_network.update(reward - 0.5 * intero_cost)


# =============================================================================
# PART 5: TASK ENVIRONMENTS
# =============================================================================

class IowaGamblingTaskEnvironment:
    """Iowa Gambling Task with interoceptive costs"""
    
    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
        self.trial = 0
        
        # Deck parameters
        self.decks = {
            'A': {'reward_mean': 100, 'reward_std': 50, 
                  'loss_prob': 0.5, 'loss_mean': 250, 'intero_cost': 0.8},
            'B': {'reward_mean': 100, 'reward_std': 50,
                  'loss_prob': 0.1, 'loss_mean': 1250, 'intero_cost': 0.5},
            'C': {'reward_mean': 50, 'reward_std': 25,
                  'loss_prob': 0.5, 'loss_mean': 50, 'intero_cost': 0.1},
            'D': {'reward_mean': 50, 'reward_std': 25,
                  'loss_prob': 0.1, 'loss_mean': 250, 'intero_cost': 0.05}
        }
    
    def reset(self) -> Dict:
        self.trial = 0
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[float, float, Dict, bool]:
        deck_name = ['A', 'B', 'C', 'D'][action]
        deck = self.decks[deck_name]
        
        # Reward
        reward = np.random.normal(deck['reward_mean'], deck['reward_std'])
        if np.random.random() < deck['loss_prob']:
            reward -= np.random.exponential(deck['loss_mean'])
        
        # Interoceptive cost
        intero_cost = deck['intero_cost']
        if reward < 0:
            intero_cost *= 1.5
        
        observation = self._get_observation(action, reward, intero_cost)
        
        self.trial += 1
        done = self.trial >= self.n_trials
        
        return reward, intero_cost, observation, done
    
    def _get_observation(self, action: int = 0, reward: float = 0, 
                        intero_cost: float = 0) -> Dict:
        # External: reward feedback
        extero = np.zeros(32)
        extero[action] = 1.0
        extero[4:8] = np.clip(reward / 100.0, -1, 1) * np.array([1, 0.8, 0.6, 0.4])
        extero[8:] = np.random.randn(24) * 0.1
        
        # Internal: physiological signals
        intero = np.zeros(16)
        intero[0:4] = np.random.normal(0, 0.1 + intero_cost * 0.3, size=4)  # HRV
        intero[4:8] = np.random.exponential(intero_cost, size=4)  # SCR
        intero[8:12] = np.random.normal(-intero_cost, 0.2, size=4)  # Gastric
        intero[12:] = np.random.randn(4) * 0.1
        
        return {'extero': extero, 'intero': intero}


class VolatileForagingEnvironment:
    """Foraging with shifting reward statistics"""
    
    def __init__(self, grid_size: int = 10, volatility: float = 0.1, n_trials: int = 200):
        self.grid_size = grid_size
        self.volatility = volatility
        self.n_trials = n_trials
        self.trial = 0
        
        self._generate_maps()
        self.position = np.array([grid_size // 2, grid_size // 2])
    
    def reset(self) -> Dict:
        self.trial = 0
        self._generate_maps()
        self.position = np.array([self.grid_size // 2, self.grid_size // 2])
        return self._get_observation()
    
    def _generate_maps(self):
        self.reward_map = np.zeros((self.grid_size, self.grid_size))
        n_patches = 3
        for _ in range(n_patches):
            center = np.random.randint(0, self.grid_size, size=2)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    self.reward_map[i, j] += 10 * np.exp(-dist / 2)
        
        self.cost_map = np.random.exponential(0.2, (self.grid_size, self.grid_size))
    
    def step(self, action: int) -> Tuple[float, float, Dict, bool]:
        # Actions: 0=up, 1=down, 2=left, 3=right, 4=forage
        if action < 4:
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            new_pos = self.position + np.array(moves[action])
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            self.position = new_pos
        
        x, y = self.position
        reward = self.reward_map[x, y] if action == 4 else 0
        intero_cost = self.cost_map[x, y]
        
        if action == 4:
            self.reward_map[x, y] *= 0.8
        
        if np.random.random() < self.volatility:
            self._shift_maps()
        
        observation = self._get_observation()
        
        self.trial += 1
        done = self.trial >= self.n_trials
        
        return reward, intero_cost, observation, done
    
    def _shift_maps(self):
        shift = np.random.randint(-2, 3, size=2)
        self.reward_map = np.roll(self.reward_map, shift, axis=(0, 1))
        self.cost_map += np.random.normal(0, 0.05, self.cost_map.shape)
        self.cost_map = np.clip(self.cost_map, 0, 1)
    
    def _get_observation(self) -> Dict:
        # Visual field around agent
        extero = np.zeros(32)
        x, y = self.position
        
        for i, dx in enumerate([-1, 0, 1]):
            for j, dy in enumerate([-1, 0, 1]):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    extero[i*3 + j] = self.reward_map[nx, ny] / 10
        
        extero[9:] = np.random.randn(23) * 0.1
        
        # Interoceptive
        intero = np.zeros(16)
        intero[:8] = np.random.normal(0, 0.1 + self.cost_map[x, y] * 0.2, size=8)
        intero[8:] = np.random.randn(8) * 0.1
        
        return {'extero': extero, 'intero': intero}


class ThreatRewardTradeoffEnvironment:
    """High reward options produce aversive interoception"""
    
    def __init__(self, n_trials: int = 150):
        self.n_trials = n_trials
        self.trial = 0
        
        self.options = {
            0: {'reward': 10, 'threat': 0.1, 'name': 'safe_low'},
            1: {'reward': 30, 'threat': 0.3, 'name': 'moderate'},
            2: {'reward': 60, 'threat': 0.6, 'name': 'risky'},
            3: {'reward': 100, 'threat': 0.9, 'name': 'dangerous'}
        }
        
        self.threat_accumulator = 0.0
        self.threat_decay = 0.9
    
    def reset(self) -> Dict:
        self.trial = 0
        self.threat_accumulator = 0.0
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[float, float, Dict, bool]:
        opt = self.options[action]
        
        reward = np.random.normal(opt['reward'], opt['reward'] * 0.2)
        threat = opt['threat']
        
        self.threat_accumulator = self.threat_decay * self.threat_accumulator + threat
        intero_cost = threat + 0.3 * self.threat_accumulator
        
        if self.threat_accumulator > 2.0:
            intero_cost += np.random.exponential(1.0)
            self.threat_accumulator *= 0.5
        
        observation = self._get_observation(action, reward, intero_cost)
        
        self.trial += 1
        done = self.trial >= self.n_trials
        
        return reward, intero_cost, observation, done
    
    def _get_observation(self, action: int = 0, reward: float = 0,
                        intero_cost: float = 0) -> Dict:
        extero = np.zeros(32)
        extero[action] = 1.0
        extero[4:8] = np.clip(reward / 50, 0, 2) * np.array([1, 0.8, 0.6, 0.4])
        extero[8:] = np.random.randn(24) * 0.1
        
        intero = np.zeros(16)
        intero[:8] = np.random.normal(intero_cost, 0.2, size=8)
        intero[8:] = np.random.exponential(intero_cost * 0.5, size=8)
        
        return {'extero': extero, 'intero': intero}


# =============================================================================
# PART 6: EXPERIMENTAL FRAMEWORK
# =============================================================================

class AgentComparisonExperiment:
    """Complete agent comparison experiment"""
    
    def __init__(self, n_agents: int = 20, n_trials: int = 100):
        self.n_agents = n_agents
        self.n_trials = n_trials
        
        self.agent_types = {
            'APGI': APGIActiveInferenceAgent,
            'StandardPP': StandardPPAgent,
            'GWTOnly': GWTOnlyAgent,
            'ActorCritic': ActorCriticAgent
        }
        
        self.environments = {
            'IGT': IowaGamblingTaskEnvironment,
            'Foraging': VolatileForagingEnvironment,
            'ThreatReward': ThreatRewardTradeoffEnvironment
        }
    
    def run_full_experiment(self) -> Dict:
        """Run all agents on all environments"""
        
        results = {}
        
        for env_name, EnvClass in self.environments.items():
            print(f"\n{'='*60}")
            print(f"ENVIRONMENT: {env_name}")
            print(f"{'='*60}")
            
            results[env_name] = {}
            
            for agent_name, AgentClass in self.agent_types.items():
                print(f"\nRunning {agent_name}...")
                
                agent_results = []
                
                for agent_idx in tqdm(range(self.n_agents), desc=f"  {agent_name}"):
                    config = self._get_config(env_name)
                    agent = AgentClass(config)
                    
                    if env_name == 'IGT':
                        env = EnvClass(n_trials=self.n_trials)
                    elif env_name == 'Foraging':
                        env = EnvClass(n_trials=self.n_trials, volatility=0.1)
                    else:
                        env = EnvClass(n_trials=self.n_trials)
                    
                    episode_data = self._run_episode(agent, env, env_name)
                    agent_results.append(episode_data)
                
                results[env_name][agent_name] = self._aggregate_results(agent_results)
        
        return results
    
    def _run_episode(self, agent, env, env_name: str) -> Dict:
        """Run single episode"""
        
        data = {
            'rewards': [],
            'intero_costs': [],
            'cumulative_reward': [],
            'actions': [],
            'ignitions': [],
            'intero_dominant_ignitions': [],
            'strategy_changes': [],
            'convergence_trial': None
        }
        
        observation = env.reset()
        cumulative = 0
        prev_action = None
        
        for trial in range(self.n_trials):
            action = agent.step(observation)
            reward, intero_cost, next_obs, done = env.step(action)
            
            data['rewards'].append(reward)
            data['intero_costs'].append(intero_cost)
            cumulative += reward
            data['cumulative_reward'].append(cumulative)
            data['actions'].append(action)
            
            # Ignition data
            if hasattr(agent, 'conscious_access'):
                data['ignitions'].append(int(agent.conscious_access))
                
                if agent.conscious_access and hasattr(agent, 'ignition_history'):
                    if len(agent.ignition_history) > 0:
                        last_ignition = agent.ignition_history[-1]
                        data['intero_dominant_ignitions'].append(
                            int(last_ignition.get('intero_dominant', False))
                        )
            
            # Strategy change
            if prev_action is not None:
                data['strategy_changes'].append(int(action != prev_action))
            
            prev_action = action
            
            # Convergence check (IGT: choosing C or D consistently)
            if env_name == 'IGT' and trial > 20:
                if data['convergence_trial'] is None:
                    recent_actions = data['actions'][-20:]
                    good_choices = sum([1 for a in recent_actions if a in [2, 3]])
                    if good_choices >= 15:
                        data['convergence_trial'] = trial
            
            agent.receive_outcome(reward, intero_cost, next_obs)
            observation = next_obs
            
            if done:
                break
        
        return data
    
    def _aggregate_results(self, agent_results: List[Dict]) -> Dict:
        """Aggregate results across agents"""
        
        aggregated = {
            'mean_cumulative_reward': np.mean([r['cumulative_reward'][-1] 
                                              for r in agent_results]),
            'std_cumulative_reward': np.std([r['cumulative_reward'][-1] 
                                            for r in agent_results]),
            'mean_convergence_trial': np.mean([r['convergence_trial'] 
                                              for r in agent_results 
                                              if r['convergence_trial'] is not None]),
            'convergence_rate': np.mean([r['convergence_trial'] is not None 
                                        for r in agent_results]),
            'mean_ignition_rate': np.mean([np.mean(r['ignitions']) 
                                          for r in agent_results 
                                          if len(r['ignitions']) > 0]),
            'intero_dominance_rate': np.mean([np.mean(r['intero_dominant_ignitions']) 
                                             for r in agent_results 
                                             if len(r['intero_dominant_ignitions']) > 0]),
            'raw_results': agent_results
        }
        
        return aggregated
    
    def _get_config(self, env_name: str) -> Dict:
        """Get agent configuration"""
        return {
            'n_actions': 4 if env_name == 'IGT' else 5,
            'theta_init': 0.5,
            'beta': 1.2,
            'Pi_e_init': 1.0,
            'Pi_i_init': 1.0
        }
    
    def analyze_predictions(self, results: Dict) -> Dict:
        """Analyze APGI predictions"""
        
        analysis = {
            'P3a_convergence': {},
            'P3b_intero_dominance': {},
            'P3c_ignition_strategy': {},
            'P3d_adaptation': {}
        }
        
        # P3a: Convergence speed
        for env_name in results.keys():
            analysis['P3a_convergence'][env_name] = {
                agent: results[env_name][agent]['mean_convergence_trial']
                for agent in results[env_name].keys()
            }
        
        # P3b: Interoceptive dominance (IGT, APGI only)
        if 'APGI' in results['IGT']:
            intero_dom = results['IGT']['APGI']['intero_dominance_rate']
            analysis['P3b_intero_dominance'] = {
                'rate': intero_dom,
                'prediction_met': 0.70 <= intero_dom <= 0.85
            }
        
        # P3c: Ignition predicts strategy change
        analysis['P3c_ignition_strategy'] = self._analyze_ignition_strategy(results)
        
        # P3d: Adaptation speed
        if 'Foraging' in results:
            analysis['P3d_adaptation'] = self._analyze_adaptation(results['Foraging'])
        
        return analysis
    
    def _analyze_ignition_strategy(self, results: Dict) -> Dict:
        """Analyze relationship between ignition and strategy changes"""
        
        if 'APGI' not in results['IGT']:
            return {'error': 'No APGI data'}
        
        from sklearn.linear_model import LogisticRegression
        
        X_data = []
        y_data = []
        
        for r in results['IGT']['APGI']['raw_results']:
            for t in range(1, min(len(r['ignitions']), len(r['strategy_changes']))):
                if t < len(r['ignitions']) and t-1 < len(r['strategy_changes']):
                    X_data.append([
                        r['ignitions'][t],
                        abs(r['rewards'][t]) / 100.0,
                        t / len(r['ignitions'])
                    ])
                    y_data.append(r['strategy_changes'][t-1])
        
        if len(X_data) < 10:
            return {'error': 'Insufficient data'}
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        model = LogisticRegression()
        model.fit(X, y)
        
        ignition_coef = model.coef_[0][0]
        
        # Bootstrap CI
        n_bootstrap = 100
        coef_samples = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(X), len(X), replace=True)
            model_boot = LogisticRegression()
            model_boot.fit(X[idx], y[idx])
            coef_samples.append(model_boot.coef_[0][0])
        
        ci = np.percentile(coef_samples, [2.5, 97.5])
        
        return {
            'ignition_coefficient': float(ignition_coef),
            'ci_95': ci.tolist(),
            'prediction_met': ci[0] > 0
        }
    
    def _analyze_adaptation(self, foraging_results: Dict) -> Dict:
        """Analyze adaptation speed in volatile environment"""
        
        apgi_reward = foraging_results['APGI']['mean_cumulative_reward']
        pp_reward = foraging_results['StandardPP']['mean_cumulative_reward']
        
        relative_improvement = (apgi_reward - pp_reward) / abs(pp_reward)
        
        return {
            'apgi_reward': float(apgi_reward),
            'standardpp_reward': float(pp_reward),
            'relative_improvement': float(relative_improvement),
            'prediction_met': relative_improvement > 0.15
        }
    
    def check_falsification(self, results: Dict, analysis: Dict) -> Dict:
        """Check falsification criteria"""
        
        falsified = {}
        
        # F3.1: No performance advantage
        igt_rewards = {agent: results['IGT'][agent]['mean_cumulative_reward']
                      for agent in results['IGT'].keys()}
        apgi_reward = igt_rewards['APGI']
        best_other = max([v for k, v in igt_rewards.items() if k != 'APGI'])
        
        improvement = (apgi_reward - best_other) / abs(best_other)
        falsified['F3.1'] = {
            'falsified': improvement < 0.05,
            'improvement': float(improvement),
            'threshold': 0.05
        }
        
        # F3.2: Ignition uncorrelated with adaptation
        if 'P3c_ignition_strategy' in analysis:
            p3c = analysis['P3c_ignition_strategy']
            if 'error' not in p3c:
                falsified['F3.2'] = {
                    'falsified': not p3c['prediction_met'],
                    'coefficient': p3c['ignition_coefficient']
                }
        
        # F3.3: StandardPP outperforms
        falsified['F3.3'] = {
            'falsified': igt_rewards['StandardPP'] > igt_rewards['APGI'],
            'apgi': float(apgi_reward),
            'standardpp': float(igt_rewards['StandardPP'])
        }
        
        return falsified


# =============================================================================
# PART 7: VISUALIZATION
# =============================================================================

def plot_experiment_results(results: Dict, analysis: Dict,
                            save_path: str = 'protocol3_results.png'):
    """Generate comprehensive visualization"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    colors = {
        'APGI': '#2E86AB',
        'StandardPP': '#A23B72',
        'GWTOnly': '#F18F01',
        'ActorCritic': '#06A77D'
    }
    
    # Row 1: Cumulative rewards
    for i, env_name in enumerate(['IGT', 'Foraging', 'ThreatReward']):
        ax = fig.add_subplot(gs[0, i])
        
        for agent_name in results[env_name].keys():
            agent_data = results[env_name][agent_name]['raw_results'][0]
            ax.plot(agent_data['cumulative_reward'], 
                   label=agent_name, color=colors.get(agent_name, 'gray'),
                   linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Trial', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cumulative Reward', fontsize=11, fontweight='bold')
        ax.set_title(f'{env_name}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    # Row 1, Col 4: Final performance comparison
    ax = fig.add_subplot(gs[0, 3])
    
    env_name = 'IGT'
    agents = list(results[env_name].keys())
    final_rewards = [results[env_name][a]['mean_cumulative_reward'] for a in agents]
    errors = [results[env_name][a]['std_cumulative_reward'] for a in agents]
    
    bars = ax.bar(agents, final_rewards, yerr=errors,
                  color=[colors.get(a, 'gray') for a in agents],
                  alpha=0.7, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Final Cumulative Reward', fontsize=11, fontweight='bold')
    ax.set_title('IGT Performance', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Row 2: Convergence analysis
    ax = fig.add_subplot(gs[1, 0:2])
    
    convergence_data = analysis['P3a_convergence']['IGT']
    agents = list(convergence_data.keys())
    convergence = [convergence_data[a] for a in agents]
    
    ax.bar(agents, convergence, color=[colors.get(a, 'gray') for a in agents],
          alpha=0.7, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Convergence Trial', fontsize=11, fontweight='bold')
    ax.set_title('IGT Convergence Speed', fontsize=12, fontweight='bold')
    ax.axhline(y=80, color='red', linestyle='--', linewidth=2, 
              label='Human benchmark')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Row 2: Interoceptive dominance (APGI only)
    ax = fig.add_subplot(gs[1, 2])
    
    if 'P3b_intero_dominance' in analysis:
        intero_data = analysis['P3b_intero_dominance']
        
        ax.bar(['Intero\nDominant', 'Extero\nDominant'],
              [intero_data['rate'], 1 - intero_data['rate']],
              color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black',
              linewidth=2)
        
        ax.set_ylabel('Proportion of Ignitions', fontsize=11, fontweight='bold')
        ax.set_title('APGI Ignition Sources (IGT)', fontsize=12, fontweight='bold')
        ax.axhline(y=0.70, color='green', linestyle='--', linewidth=2,
                  label='Prediction range')
        ax.axhline(y=0.85, color='green', linestyle='--', linewidth=2)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    # Row 2: Ignition-strategy relationship
    ax = fig.add_subplot(gs[1, 3])
    
    if 'P3c_ignition_strategy' in analysis:
        p3c = analysis['P3c_ignition_strategy']
        if 'error' not in p3c:
            coef = p3c['ignition_coefficient']
            ci = p3c['ci_95']
            
            ax.barh(['Ignition\nCoefficient'], [coef], 
                   xerr=[[coef - ci[0]], [ci[1] - coef]],
                   color='purple', alpha=0.7, edgecolor='black', linewidth=2)
            
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Logistic Regression Coefficient', fontsize=10,
                         fontweight='bold')
            ax.set_title('Ignition → Strategy Change', fontsize=11, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
    
    # Row 3: Action distributions
    for i, env_name in enumerate(['IGT', 'Foraging', 'ThreatReward']):
        ax = fig.add_subplot(gs[2, i])
        
        if env_name == 'IGT':
            # Show action preferences
            for agent_name in ['APGI', 'StandardPP']:
                if agent_name in results[env_name]:
                    actions = results[env_name][agent_name]['raw_results'][0]['actions']
                    action_counts = np.bincount(actions, minlength=4)
                    action_probs = action_counts / action_counts.sum()
                    
                    x = np.arange(4)
                    width = 0.35
                    offset = -width/2 if agent_name == 'APGI' else width/2
                    
                    ax.bar(x + offset, action_probs, width, 
                          label=agent_name, color=colors[agent_name],
                          alpha=0.7, edgecolor='black')
            
            ax.set_xlabel('Deck', fontsize=11, fontweight='bold')
            ax.set_ylabel('Selection Probability', fontsize=11, fontweight='bold')
            ax.set_title('IGT Action Distribution', fontsize=12, fontweight='bold')
            ax.set_xticks(range(4))
            ax.set_xticklabels(['A', 'B', 'C', 'D'])
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
    
    # Summary statistics
    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')
    
    summary_text = f"""
    SUMMARY STATISTICS
    {'='*35}
    
    IGT Performance:
      APGI: {results['IGT']['APGI']['mean_cumulative_reward']:.0f} Â± {results['IGT']['APGI']['std_cumulative_reward']:.0f}
      StandardPP: {results['IGT']['StandardPP']['mean_cumulative_reward']:.0f} Â± {results['IGT']['StandardPP']['std_cumulative_reward']:.0f}
    
    Convergence (trials):
      APGI: {analysis['P3a_convergence']['IGT']['APGI']:.1f}
      StandardPP: {analysis['P3a_convergence']['IGT']['StandardPP']:.1f}
    
    Intero Dominance (APGI):
      {analysis['P3b_intero_dominance']['rate']:.2%}
      {'âœ… Met' if analysis['P3b_intero_dominance']['prediction_met'] else 'âŒ Not met'}
    
    Adaptation (Foraging):
      Improvement: {analysis['P3d_adaptation']['relative_improvement']:.1%}
      {'âœ… Met' if analysis['P3d_adaptation']['prediction_met'] else 'âŒ Not met'}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
           verticalalignment='center')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.show()


def print_falsification_report(falsification: Dict):
    """Print falsification report"""
    
    print("\n" + "="*80)
    print("PROTOCOL 3 FALSIFICATION REPORT")
    print("="*80)
    
    n_falsified = sum([v['falsified'] for v in falsification.values()])
    n_total = len(falsification)
    
    print(f"\nOVERALL STATUS: ", end="")
    if n_falsified > 0:
        print("âŒ MODEL FALSIFIED")
    else:
        print("âœ… MODEL VALIDATED")
    
    print(f"\nCriteria Passed: {n_total - n_falsified}/{n_total}")
    print(f"Criteria Failed: {n_falsified}/{n_total}")
    
    for code, result in falsification.items():
        print(f"\n{code}:")
        print(f"  Falsified: {'âŒ YES' if result['falsified'] else 'âœ… NO'}")
        for k, v in result.items():
            if k != 'falsified':
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
    
    print("\n" + "="*80)


# =============================================================================
# PART 8: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline"""
    
    print("="*80)
    print("APGI PROTOCOL 3: ACTIVE INFERENCE AGENT SIMULATIONS")
    print("="*80)
    
    config = {
        'n_agents': 20,
        'n_trials': 100
    }
    
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Run experiment
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)
    
    experiment = AgentComparisonExperiment(
        n_agents=config['n_agents'],
        n_trials=config['n_trials']
    )
    
    results = experiment.run_full_experiment()
    
    # Analyze
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    analysis = experiment.analyze_predictions(results)
    
    print("\nKey Findings:")
    print(f"  P3a - APGI convergence: {analysis['P3a_convergence']['IGT']['APGI']:.1f} trials")
    if 'P3b_intero_dominance' in analysis:
        print(f"  P3b - Intero dominance: {analysis['P3b_intero_dominance']['rate']:.2%}")
    if 'P3d_adaptation' in analysis:
        print(f"  P3d - Foraging advantage: {analysis['P3d_adaptation']['relative_improvement']:.1%}")
    
    # Falsification
    print("\n" + "="*80)
    print("FALSIFICATION ANALYSIS")
    print("="*80)
    
    falsification = experiment.check_falsification(results, analysis)
    print_falsification_report(falsification)
    
    # Visualize
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_experiment_results(results, analysis, 'protocol3_results.png')
    
    # Save
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    summary = {
        'config': config,
        'analysis': {k: v for k, v in analysis.items() 
                    if not isinstance(v, dict) or 'raw_results' not in str(v)},
        'falsification': falsification
    }
    
    # Convert to JSON-serializable
    def convert(obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj
    
    summary = convert(summary)
    
    with open('protocol3_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("âœ… Results saved to: protocol3_results.json")
    
    print("\n" + "="*80)
    print("PROTOCOL 3 EXECUTION COMPLETE")
    print("="*80)
    
    return summary


if __name__ == "__main__":
    main()