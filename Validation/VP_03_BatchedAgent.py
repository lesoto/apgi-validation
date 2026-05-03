from typing import Dict, List

import torch
import torch.nn as nn

# Matplotlib imports for PNG visualization
try:
    import matplotlib

    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from utils.apgi_config import APGIConfig
from utils.constants import LEVEL_TIMESCALES
from utils.falsification_thresholds import THRESHOLD_REGISTRY  # noqa: F401


class BatchedHierarchicalGenerativeModel(nn.Module):
    """
    Vectorized Hierarchical Predictive Processing
    Handles many agents in parallel using tensor operations.
    """

    def __init__(
        self, batch_size: int, levels: List[Dict], learning_rate: float = 0.01
    ):
        super().__init__()
        self.batch_size = batch_size
        self.levels = levels
        self.n_levels = len(levels)

        self.level_networks = nn.ModuleList()
        self.states = nn.ParameterList()

        for i in range(self.n_levels):
            # Batch-indexable states: [Batch, Dim]
            state = nn.Parameter(torch.zeros(batch_size, levels[i]["dim"]))
            self.states.append(state)

            if i < self.n_levels - 1:
                top_dim = levels[i + 1]["dim"]
                bottom_dim = levels[i]["dim"]
                network = nn.Sequential(
                    nn.Linear(top_dim, top_dim * 2),
                    nn.Tanh(),
                    nn.Linear(top_dim * 2, bottom_dim),
                )
                self.level_networks.append(network)

        self.taus = torch.tensor([level["tau"] for level in levels]).view(1, -1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, observations: torch.Tensor, dt: float = 0.05):
        """Perform one parallelized prediction-error update for all agents"""
        # Predictions [Batch, Dim]
        predictions = []
        for i in range(self.n_levels - 1):
            predictions.append(self.level_networks[i](self.states[i + 1]))

        # Errors [Batch, Dim]
        errors = []
        errors.append(observations - predictions[0])
        for i in range(1, self.n_levels - 1):
            errors.append(self.states[i] - predictions[i])

        # Update states in parallel
        for i in range(self.n_levels - 1):
            # τ ∂s/∂t = ε -> s = s + dt * ε / τ
            self.states[i].data += dt * errors[i] / self.taus[0, i]

        # Global loss for parameter training
        loss = torch.stack([torch.sum(err**2) for err in errors]).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return errors[0]


class BatchedAPGIActiveInferenceAgent(nn.Module):
    """
    Vectorized APGI Agent for high-throughput simulations.
    Simulates thousands of agents simultaneously.
    """

    def __init__(self, batch_size: int, config: Dict):
        super().__init__()
        self.batch_size = batch_size

        # Vectorized models
        self.extero_model = BatchedHierarchicalGenerativeModel(
            batch_size,
            [
                {"name": "sensory", "dim": 32, "tau": LEVEL_TIMESCALES.TAU_SENSORY},
                {"name": "objects", "dim": 16, "tau": LEVEL_TIMESCALES.TAU_ORGAN},
                {"name": "context", "dim": 8, "tau": LEVEL_TIMESCALES.TAU_COGNITIVE},
            ],
        )

        # State vectors for ignition [Batch]
        self.S_t = torch.zeros(batch_size)
        self.theta_t = torch.full((batch_size,), APGIConfig.theta_init)

        # Constants [1] or [Batch]
        self.alpha = APGIConfig.alpha_ignition
        self.tau_S = APGIConfig.tau_S
        self.Pi_e = APGIConfig.Pi_e_init
        self.Pi_i = APGIConfig.Pi_i_init

    def step(self, observations: Dict[str, torch.Tensor], dt: float = 0.05):
        """Execute parallel step for all agents"""
        extero_obs = observations["extero"]  # [Batch, 32]

        # 1. Parallel prediction error update
        eps_e = self.extero_model(extero_obs, dt)

        # 2. Parallel surprise accumulation
        input_drive = self.Pi_e * torch.norm(eps_e, dim=1)
        dS_dt = -self.S_t / self.tau_S + input_drive
        self.S_t += dS_dt * dt

        # 3. Parallel ignition check
        z = self.alpha * (self.S_t - self.theta_t)
        P_ignition = torch.sigmoid(z)
        ignition = torch.rand(self.batch_size) < P_ignition

        # 4. Action selection (Simplified for batch)
        # In a real implementation, we'd use a batched policy network
        return ignition, P_ignition
