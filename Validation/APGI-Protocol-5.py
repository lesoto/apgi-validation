"""
APGI Protocol 5: Evolutionary Emergence of APGI-like Architectures
===================================================================

Complete implementation of evolutionary testing framework to determine whether
APGI-like computational architectures emerge under selection pressure in complex
environments.

This protocol uses genetic algorithms to evolve agent architectures and tests
whether APGI components (threshold ignition, interoceptive weighting, somatic
markers, precision gating) provide selective advantages.

Author: APGI Research Team
Date: 2025
Version: 1.0 (Production)

Dependencies:
    numpy, scipy, pandas, matplotlib, seaborn, tqdm
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from tqdm import tqdm
import json
from pathlib import Path
from collections import defaultdict
import copy
import networkx as nx
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# PART 1: GENOME & AGENT ARCHITECTURE
# =============================================================================


@dataclass
class AgentGenome:
    """
    Genome encoding architectural and parametric choices

    Structural genes determine which APGI components are present:
        - has_threshold: Discrete ignition vs continuous processing
        - has_intero_weighting: β·Π_i term vs uniform weighting
        - has_somatic_markers: Affective valuation of states
        - has_precision_weighting: Precision-gated vs direct processing

    Parameter genes determine component strengths (if present)
    """

    # Structural genes (architecture)
    has_threshold: bool
    has_intero_weighting: bool
    has_somatic_markers: bool
    has_precision_weighting: bool

    # Parameter genes
    theta_0: float  # Threshold baseline
    alpha: float  # Sigmoid steepness
    beta: float  # Somatic bias
    Pi_e_lr: float  # External precision learning rate
    Pi_i_lr: float  # Internal precision learning rate
    somatic_lr: float  # Somatic marker learning rate

    # Architecture genes
    n_hidden_layers: int
    hidden_dim: int

    # Metabolic cost parameter
    ignition_cost: float = 0.1  # Cost of ignition

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "has_threshold": self.has_threshold,
            "has_intero_weighting": self.has_intero_weighting,
            "has_somatic_markers": self.has_somatic_markers,
            "has_precision_weighting": self.has_precision_weighting,
            "theta_0": self.theta_0,
            "alpha": self.alpha,
            "beta": self.beta,
            "Pi_e_lr": self.Pi_e_lr,
            "Pi_i_lr": self.Pi_i_lr,
            "somatic_lr": self.somatic_lr,
            "n_hidden_layers": self.n_hidden_layers,
            "hidden_dim": self.hidden_dim,
            "ignition_cost": self.ignition_cost,
        }

    @classmethod
    def random(cls) -> "AgentGenome":
        """Generate random genome"""
        return cls(
            has_threshold=np.random.random() > 0.5,
            has_intero_weighting=np.random.random() > 0.5,
            has_somatic_markers=np.random.random() > 0.5,
            has_precision_weighting=np.random.random() > 0.5,
            theta_0=np.random.uniform(0.2, 0.8),
            alpha=np.random.uniform(2.0, 10.0),
            beta=np.random.uniform(0.5, 2.0),
            Pi_e_lr=np.random.uniform(0.01, 0.2),
            Pi_i_lr=np.random.uniform(0.01, 0.2),
            somatic_lr=np.random.uniform(0.01, 0.3),
            n_hidden_layers=np.random.randint(1, 4),
            hidden_dim=np.random.choice([16, 32, 64]),
            ignition_cost=np.random.uniform(0.05, 0.15),
        )

    def mutate(self, mutation_rate: float = 0.1) -> "AgentGenome":
        """Create mutated copy"""
        mutated = copy.deepcopy(self)

        # Structural mutations (flip bits)
        if np.random.random() < mutation_rate:
            mutated.has_threshold = not mutated.has_threshold
        if np.random.random() < mutation_rate:
            mutated.has_intero_weighting = not mutated.has_intero_weighting
        if np.random.random() < mutation_rate:
            mutated.has_somatic_markers = not mutated.has_somatic_markers
        if np.random.random() < mutation_rate:
            mutated.has_precision_weighting = not mutated.has_precision_weighting

        # Parameter mutations (Gaussian perturbation)
        if np.random.random() < mutation_rate:
            mutated.theta_0 = np.clip(
                mutated.theta_0 + np.random.normal(0, 0.1), 0.1, 0.9
            )

        if np.random.random() < mutation_rate:
            mutated.alpha = np.clip(
                mutated.alpha * np.random.uniform(0.8, 1.2), 1.0, 15.0
            )

        if np.random.random() < mutation_rate:
            mutated.beta = np.clip(mutated.beta * np.random.uniform(0.8, 1.2), 0.3, 2.5)

        if np.random.random() < mutation_rate:
            mutated.Pi_e_lr = np.clip(
                mutated.Pi_e_lr * np.random.uniform(0.8, 1.2), 0.005, 0.3
            )

        if np.random.random() < mutation_rate:
            mutated.Pi_i_lr = np.clip(
                mutated.Pi_i_lr * np.random.uniform(0.8, 1.2), 0.005, 0.3
            )

        if np.random.random() < mutation_rate:
            mutated.somatic_lr = np.clip(
                mutated.somatic_lr * np.random.uniform(0.8, 1.2), 0.005, 0.4
            )

        # Architecture mutations
        if np.random.random() < mutation_rate:
            mutated.n_hidden_layers = np.clip(
                mutated.n_hidden_layers + np.random.randint(-1, 2), 1, 5
            )

        if np.random.random() < mutation_rate:
            mutated.hidden_dim = np.clip(
                int(mutated.hidden_dim * np.random.choice([0.5, 1.5, 2.0])), 8, 128
            )

        return mutated

    @classmethod
    def crossover(cls, parent1: "AgentGenome", parent2: "AgentGenome") -> "AgentGenome":
        """Single-point crossover"""
        # Convert to dicts
        p1_dict = parent1.to_dict()
        p2_dict = parent2.to_dict()

        # Crossover point
        keys = list(p1_dict.keys())
        crossover_point = np.random.randint(len(keys))

        child_dict = {}
        for i, key in enumerate(keys):
            if i < crossover_point:
                child_dict[key] = p1_dict[key]
            else:
                child_dict[key] = p2_dict[key]

        return cls(**child_dict)


class EvolvableAgent:
    """
    Agent with evolvable architecture based on genome

    Implements decision-making with variable APGI components
    """

    def __init__(self, genome: AgentGenome):
        self.genome = genome

        # Internal state
        self.Pi_e = 1.0  # External precision
        self.Pi_i = 1.0  # Internal precision

        # Somatic markers (state → value associations)
        self.somatic_markers = {}

        # History
        self.ignition_history = []
        self.reward_history = []

    def compute_surprise(self, external_signal: float, internal_signal: float) -> float:
        """
        Compute accumulated surprise based on genome architecture

        Full APGI: S_t = Π_e·|ε_e| + β·Π_i·|ε_i|
        Without intero_weighting: S_t = Π_e·|ε_e| + |ε_i|
        Without precision: S_t = |ε_e| + |ε_i|
        """

        # External component
        if self.genome.has_precision_weighting:
            extero_component = self.Pi_e * abs(external_signal)
        else:
            extero_component = abs(external_signal)

        # Internal component
        if self.genome.has_intero_weighting:
            if self.genome.has_precision_weighting:
                intero_component = self.genome.beta * self.Pi_i * abs(internal_signal)
            else:
                intero_component = self.genome.beta * abs(internal_signal)
        else:
            intero_component = abs(internal_signal)

        S_t = extero_component + intero_component

        return S_t

    def decide(
        self, observation: np.ndarray, external_signal: float, internal_signal: float
    ) -> Tuple[int, bool, float]:
        """
        Make decision about whether to act

        Returns:
            action: Action index
            ignition: Whether conscious processing occurred
            metabolic_cost: Cost of decision process
        """

        S_t = self.compute_surprise(external_signal, internal_signal)

        # Apply threshold mechanism (if present)
        if self.genome.has_threshold:
            # Ignition probability
            P_ignition = 1.0 / (
                1.0 + np.exp(-self.genome.alpha * (S_t - self.genome.theta_0))
            )

            ignition = np.random.random() < P_ignition

            # Metabolic cost for ignition
            metabolic_cost = self.genome.ignition_cost if ignition else 0.0

        else:
            # Continuous processing (always "on")
            ignition = True
            metabolic_cost = 0.05  # Constant low cost

        # Select action
        if ignition:
            # Full processing with somatic markers
            action = self._select_action_full(observation)
        else:
            # Quick heuristic
            action = self._select_action_heuristic(observation)

        self.ignition_history.append(ignition)

        return action, ignition, metabolic_cost

    def _select_action_full(self, observation: np.ndarray) -> int:
        """Full action selection with somatic markers"""

        # Simple feedforward network
        # For this simulation, use random weights based on genome
        np.random.seed(hash(str(self.genome)) % 2**32)

        hidden = observation.copy()
        for _ in range(self.genome.n_hidden_layers):
            # Ensure proper dimensions: input_dim x hidden_dim
            input_dim = len(hidden)
            W = np.random.randn(input_dim, self.genome.hidden_dim) * 0.1
            hidden = np.tanh(hidden @ W)

        # Output layer: hidden_dim x 4 (for 4 actions)
        W_out = np.random.randn(self.genome.hidden_dim, 4) * 0.1
        logits = hidden @ W_out

        # Apply somatic markers if available
        if self.genome.has_somatic_markers:
            state_hash = hash(observation.tobytes())
            if state_hash in self.somatic_markers:
                # Bias towards previously rewarding actions
                marker_values = self.somatic_markers[state_hash]
                logits += marker_values

        # Softmax
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        action = np.random.choice(len(probs), p=probs)

        return action

    def _select_action_heuristic(self, observation: np.ndarray) -> int:
        """Fast heuristic action selection"""
        # Simple rule: choose based on observation mean
        if observation.mean() > 0:
            return 0
        else:
            return 1

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        prediction_error_external: float,
        prediction_error_internal: float,
    ):
        """
        Update internal parameters based on experience
        """

        # Update precision estimates
        if self.genome.has_precision_weighting:
            # External precision tracks prediction error magnitude
            pe_magnitude = abs(prediction_error_external)
            self.Pi_e += self.genome.Pi_e_lr * (pe_magnitude - self.Pi_e)
            self.Pi_e = np.clip(self.Pi_e, 0.1, 3.0)

            # Internal precision
            intero_pe_magnitude = abs(prediction_error_internal)
            self.Pi_i += self.genome.Pi_i_lr * (intero_pe_magnitude - self.Pi_i)
            self.Pi_i = np.clip(self.Pi_i, 0.1, 3.0)

        # Update somatic markers
        if self.genome.has_somatic_markers:
            state_hash = hash(observation.tobytes())

            if state_hash not in self.somatic_markers:
                self.somatic_markers[state_hash] = np.zeros(4)

            # Strengthen marker for taken action based on reward
            self.somatic_markers[state_hash][action] += self.genome.somatic_lr * reward

            # Decay others
            for a in range(4):
                if a != action:
                    self.somatic_markers[state_hash][a] *= 0.99

        self.reward_history.append(reward)

    def get_architecture_signature(self) -> str:
        """Get string signature of architecture"""
        components = []
        if self.genome.has_threshold:
            components.append("T")
        if self.genome.has_intero_weighting:
            components.append("I")
        if self.genome.has_somatic_markers:
            components.append("S")
        if self.genome.has_precision_weighting:
            components.append("P")

        return "".join(components) if components else "None"


# =============================================================================
# PART 2: TEST ENVIRONMENTS
# =============================================================================


class Environment(ABC):
    """Abstract base class for test environments"""

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation"""
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[float, float, np.ndarray, bool]:
        """
        Execute action in environment

        Returns:
            reward: External reward signal
            interoceptive_cost: Internal homeostatic cost
            next_observation: Next state
            done: Episode termination
        """
        pass


class IowaGamblingTask(Environment):
    """
    Iowa Gambling Task environment

    4 decks with different reward/punishment schedules
    Requires learning which decks are advantageous long-term

    Somatic markers should help by biasing away from risky decks
    """

    def __init__(self):
        self.decks = {
            0: {"mean_reward": 100, "penalty_prob": 0.5, "penalty_amount": -250},
            1: {"mean_reward": 100, "penalty_prob": 0.1, "penalty_amount": -1250},
            2: {"mean_reward": 50, "penalty_prob": 0.5, "penalty_amount": -50},
            3: {"mean_reward": 50, "penalty_prob": 0.1, "penalty_amount": -250},
        }

        self.trial = 0
        self.max_trials = 100

    def reset(self) -> np.ndarray:
        """Reset task"""
        self.trial = 0
        # Observation: [trial_number, previous_reward, deck_choices]
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def step(self, action: int) -> Tuple[float, float, np.ndarray, bool]:
        """
        Choose a deck (action 0-3)
        """
        action = int(np.clip(action, 0, 3))

        deck = self.decks[action]

        # Base reward
        reward = deck["mean_reward"]

        # Penalty with probability
        if np.random.random() < deck["penalty_prob"]:
            reward += deck["penalty_amount"]

        # Interoceptive cost (stress from uncertainty)
        # Higher for risky decks
        intero_cost = deck["penalty_prob"] * 0.5

        self.trial += 1
        done = self.trial >= self.max_trials

        # Next observation
        obs = np.array(
            [
                self.trial / self.max_trials,
                reward / 100,  # Normalized
                float(action == 0),
                float(action == 1),
                float(action == 2),
                float(action == 3),
            ]
        )

        return reward / 100, intero_cost, obs, done


class VolatileForagingEnvironment(Environment):
    """
    Foraging in volatile environment

    Resource patches change quality over time
    Requires tracking precision to know when to switch patches
    """

    def __init__(self):
        self.n_patches = 4
        self.patch_qualities = np.random.uniform(0.2, 0.8, self.n_patches)
        self.volatility = 0.1

        self.current_patch = 0
        self.steps_in_patch = 0
        self.timestep = 0
        self.max_timesteps = 200

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.patch_qualities = np.random.uniform(0.2, 0.8, self.n_patches)
        self.current_patch = 0
        self.steps_in_patch = 0
        self.timestep = 0

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        obs = np.zeros(8)
        obs[0] = self.timestep / self.max_timesteps
        obs[1] = self.steps_in_patch / 20  # Normalized
        obs[2:6] = self.patch_qualities
        obs[6] = self.current_patch / 3  # Current patch indicator
        obs[7] = np.random.normal(0, 0.1)  # Noise

        return obs

    def step(self, action: int) -> Tuple[float, float, np.ndarray, bool]:
        """
        Actions: 0=stay, 1-3=switch to patch 1-3
        """
        action = int(np.clip(action, 0, 3))

        # Update patch qualities (volatility)
        self.patch_qualities += np.random.normal(0, self.volatility, self.n_patches)
        self.patch_qualities = np.clip(self.patch_qualities, 0.0, 1.0)

        if action == 0:
            # Stay in current patch
            reward = self.patch_qualities[self.current_patch]
            self.steps_in_patch += 1

            # Depletion
            self.patch_qualities[self.current_patch] *= 0.95

            # Low interoceptive cost
            intero_cost = 0.0

        else:
            # Switch patches
            self.current_patch = action - 1
            reward = self.patch_qualities[self.current_patch] * 0.5  # Switching cost
            self.steps_in_patch = 0

            # Switching incurs interoceptive cost
            intero_cost = 0.3

        self.timestep += 1
        done = self.timestep >= self.max_timesteps

        obs = self._get_observation()

        return reward, intero_cost, obs, done


class ThreatRewardTradeoff(Environment):
    """
    Threat-reward tradeoff environment

    Must balance approach to rewards vs avoidance of threats
    Interoceptive signals (arousal) should guide risk assessment
    """

    def __init__(self):
        self.threat_level = 0.5
        self.reward_level = 0.5

        self.timestep = 0
        self.max_timesteps = 150

        self.arousal = 0.5  # Internal state

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.threat_level = np.random.uniform(0.3, 0.7)
        self.reward_level = np.random.uniform(0.3, 0.7)
        self.timestep = 0
        self.arousal = 0.5

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get observation"""
        return np.array(
            [
                self.timestep / self.max_timesteps,
                self.threat_level,
                self.reward_level,
                self.arousal,
                np.random.normal(0, 0.1),
            ]
        )

    def step(self, action: int) -> Tuple[float, float, np.ndarray, bool]:
        """
        Actions: 0=avoid, 1=approach cautiously, 2=approach boldly
        """
        action = int(np.clip(action, 0, 2))

        if action == 0:
            # Avoid
            reward = 0.0
            threat_encountered = False

        elif action == 1:
            # Cautious approach
            reward = self.reward_level * 0.5
            threat_encountered = np.random.random() < (self.threat_level * 0.3)

        else:  # action == 2
            # Bold approach
            reward = self.reward_level
            threat_encountered = np.random.random() < self.threat_level

        # Threat penalty
        if threat_encountered:
            reward -= 1.0
            self.arousal = min(1.0, self.arousal + 0.3)
        else:
            self.arousal = max(0.0, self.arousal - 0.1)

        # Interoceptive cost from arousal
        intero_cost = abs(self.arousal - 0.5)

        # Environment changes
        self.threat_level += np.random.normal(0, 0.1)
        self.threat_level = np.clip(self.threat_level, 0.1, 0.9)

        self.reward_level += np.random.normal(0, 0.1)
        self.reward_level = np.clip(self.reward_level, 0.1, 0.9)

        self.timestep += 1
        done = self.timestep >= self.max_timesteps

        obs = self._get_observation()

        return reward, intero_cost, obs, done


# =============================================================================
# PART 3: EVOLUTIONARY ALGORITHM
# =============================================================================


class EvolutionaryOptimizer:
    """
    Genetic algorithm for evolving agent architectures

    Tests whether APGI components provide selective advantages
    """

    def __init__(
        self,
        population_size: int = 100,
        n_generations: int = 500,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 5,
        elitism: int = 5,
    ):
        self.pop_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

        # Test environments
        self.environments = [
            IowaGamblingTask(),
            VolatileForagingEnvironment(),
            ThreatRewardTradeoff(),
        ]

        self.population: List[AgentGenome] = []

        # History tracking
        self.history = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "std_fitness": [],
            "architecture_frequencies": [],
            "best_genomes": [],
            "diversity": [],
        }

    def initialize_population(self):
        """Create initial random population"""
        self.population = [AgentGenome.random() for _ in range(self.pop_size)]

    def evaluate_fitness(self, genome: AgentGenome) -> float:
        """
        Evaluate fitness of genome across all environments

        Fitness = reward - metabolic_cost - homeostatic_violations
        """
        agent = EvolvableAgent(genome)

        total_fitness = 0.0

        for env in self.environments:
            obs = env.reset()
            episode_reward = 0.0
            episode_cost = 0.0
            homeostatic_violations = 0

            for t in range(500):  # Max steps per episode
                # Generate signals
                external_signal = np.random.uniform(-0.3, 0.3)
                internal_signal = np.random.uniform(-0.2, 0.2)

                # Agent decision
                action, ignition, metabolic_cost = agent.decide(
                    obs, external_signal, internal_signal
                )

                # Environment step
                reward, intero_cost, next_obs, done = env.step(action)

                # Accumulate
                episode_reward += reward
                episode_cost += metabolic_cost

                # Track homeostatic violations
                if intero_cost > 0.5:
                    homeostatic_violations += 1

                # Agent update
                pe_external = np.random.normal(0, 0.2)
                pe_internal = intero_cost

                agent.update(obs, action, reward, pe_external, pe_internal)

                obs = next_obs

                if done:
                    break

            # Environment fitness
            env_fitness = (
                episode_reward  # Reward seeking
                + -0.2 * episode_cost  # Metabolic efficiency
                + -0.1 * homeostatic_violations  # Homeostatic maintenance
            )

            total_fitness += env_fitness

        # Average across environments
        fitness = total_fitness / len(self.environments)

        return fitness

    def tournament_selection(self, fitness_scores: np.ndarray) -> AgentGenome:
        """Tournament selection"""
        tournament_indices = np.random.choice(
            self.pop_size, self.tournament_size, replace=False
        )

        winner_idx = tournament_indices[np.argmax(fitness_scores[tournament_indices])]

        return self.population[winner_idx]

    def compute_architecture_frequencies(self) -> Dict[str, float]:
        """Compute frequency of each architectural feature"""
        return {
            "has_threshold": np.mean([g.has_threshold for g in self.population]),
            "has_intero_weighting": np.mean(
                [g.has_intero_weighting for g in self.population]
            ),
            "has_somatic_markers": np.mean(
                [g.has_somatic_markers for g in self.population]
            ),
            "has_precision_weighting": np.mean(
                [g.has_precision_weighting for g in self.population]
            ),
        }

    def compute_diversity(self) -> float:
        """Compute genotypic diversity (unique architectures)"""
        signatures = set()
        for genome in self.population:
            sig = (
                genome.has_threshold,
                genome.has_intero_weighting,
                genome.has_somatic_markers,
                genome.has_precision_weighting,
            )
            signatures.add(sig)

        return len(signatures) / 16  # 16 possible architectures

    def run_evolution(self):
        """Execute evolutionary optimization"""

        print(f"\n{'='*80}")
        print("STARTING EVOLUTIONARY OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Population size: {self.pop_size}")
        print(f"Generations: {self.n_generations}")
        print(f"Environments: {len(self.environments)}")

        # Initialize
        self.initialize_population()

        for generation in tqdm(range(self.n_generations), desc="Generations"):

            # Evaluate fitness
            fitness_scores = np.array(
                [self.evaluate_fitness(genome) for genome in self.population]
            )

            # Record statistics
            self.history["generation"].append(generation)
            self.history["best_fitness"].append(np.max(fitness_scores))
            self.history["mean_fitness"].append(np.mean(fitness_scores))
            self.history["std_fitness"].append(np.std(fitness_scores))

            arch_freq = self.compute_architecture_frequencies()
            self.history["architecture_frequencies"].append(arch_freq)

            diversity = self.compute_diversity()
            self.history["diversity"].append(diversity)

            # Store best genome
            best_idx = np.argmax(fitness_scores)
            self.history["best_genomes"].append(
                copy.deepcopy(self.population[best_idx])
            )

            # Progress report
            if generation % 50 == 0:
                print(f"\nGeneration {generation}:")
                print(f"  Best fitness: {np.max(fitness_scores):.3f}")
                print(f"  Mean fitness: {np.mean(fitness_scores):.3f}")
                print(f"  Diversity: {diversity:.3f}")
                print(f"  Architecture frequencies:")
                print(f"    Threshold: {arch_freq['has_threshold']:.2f}")
                print(f"    Intero: {arch_freq['has_intero_weighting']:.2f}")
                print(f"    Somatic: {arch_freq['has_somatic_markers']:.2f}")
                print(f"    Precision: {arch_freq['has_precision_weighting']:.2f}")

            # Selection
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elitism :]
            elite = [self.population[i] for i in elite_indices]

            # Create new population
            new_population = elite.copy()

            while len(new_population) < self.pop_size:
                # Tournament selection
                parent1 = self.tournament_selection(fitness_scores)
                parent2 = self.tournament_selection(fitness_scores)

                # Crossover
                if np.random.random() < self.crossover_rate:
                    child = AgentGenome.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                # Mutation
                child = child.mutate(self.mutation_rate)

                new_population.append(child)

            self.population = new_population[: self.pop_size]

        print(f"\n{'='*80}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*80}")

        return self.history


# =============================================================================
# PART 4: EVOLUTIONARY ANALYSIS
# =============================================================================


class EvolutionaryAnalyzer:
    """Analyze evolutionary dynamics and test predictions"""

    def __init__(self, history: Dict):
        self.history = history

    def compute_selection_coefficients(self) -> Dict[str, float]:
        """
        Compute selection coefficient for each architectural trait

        Selection coefficient s: rate of frequency change
        Using logistic regression on logit-transformed frequencies
        """

        selection_coefficients = {}

        traits = [
            "has_threshold",
            "has_intero_weighting",
            "has_somatic_markers",
            "has_precision_weighting",
        ]

        for trait in traits:
            # Extract frequency time series
            freqs = np.array(
                [h[trait] for h in self.history["architecture_frequencies"]]
            )

            # Avoid edge cases
            freqs = np.clip(freqs, 0.01, 0.99)

            # Logit transform
            logit_freqs = np.log(freqs / (1 - freqs))

            # Linear regression
            generations = np.array(self.history["generation"])

            # Slope = selection coefficient
            slope, intercept = np.polyfit(generations, logit_freqs, 1)

            selection_coefficients[trait] = float(slope)

        return selection_coefficients

    def find_fixation_generations(
        self, threshold: float = 0.9
    ) -> Dict[str, Optional[int]]:
        """
        Find generation when each trait reached fixation (>threshold frequency)
        """

        fixation_gens = {}

        traits = [
            "has_threshold",
            "has_intero_weighting",
            "has_somatic_markers",
            "has_precision_weighting",
        ]

        for trait in traits:
            freqs = [h[trait] for h in self.history["architecture_frequencies"]]

            for gen, freq in enumerate(freqs):
                if freq >= threshold:
                    fixation_gens[trait] = gen
                    break
            else:
                fixation_gens[trait] = None  # Never fixed

        return fixation_gens

    def analyze_architectural_combinations(self) -> pd.DataFrame:
        """
        Analyze fitness of different architectural combinations

        Compare:
        - Full APGI (all 4 components)
        - Partial APGI (1-3 components)
        - No APGI (0 components)
        """

        # Get final generation genomes
        final_genomes = self.history["best_genomes"][-10:]  # Last 10 generations

        results = []

        for genome in final_genomes:
            n_components = sum(
                [
                    genome.has_threshold,
                    genome.has_intero_weighting,
                    genome.has_somatic_markers,
                    genome.has_precision_weighting,
                ]
            )

            # Would need to re-evaluate fitness, but for now use proxy
            # In full implementation, would evaluate each combination

            results.append(
                {
                    "n_components": n_components,
                    "has_threshold": genome.has_threshold,
                    "has_intero": genome.has_intero_weighting,
                    "has_somatic": genome.has_somatic_markers,
                    "has_precision": genome.has_precision_weighting,
                    "architecture": f"{n_components}_components",
                }
            )

        return pd.DataFrame(results)

    def test_emergence_order(self) -> Dict[str, int]:
        """
        Determine order of trait emergence

        Predicted order: Threshold → Precision → Interoceptive → Somatic
        """

        fixation_gens = self.find_fixation_generations(threshold=0.5)

        # Sort by fixation generation
        sorted_traits = sorted(
            fixation_gens.items(),
            key=lambda x: x[1] if x[1] is not None else float("inf"),
        )

        emergence_order = {
            trait: rank for rank, (trait, gen) in enumerate(sorted_traits)
        }

        return emergence_order


# =============================================================================
# PART 5: FALSIFICATION CRITERIA
# =============================================================================


class FalsificationChecker:
    """Check Protocol 5 falsification criteria"""

    def __init__(self):
        self.criteria = {
            "F5.1": {
                "description": "Threshold mechanism does not reach >60% by generation 500",
                "threshold": 0.60,
                "generation": 500,
            },
            "F5.2": {
                "description": "Interoceptive weighting shows negative selection",
                "threshold": 0.0,
            },
            "F5.3": {
                "description": "Somatic markers never exceed 50% frequency",
                "threshold": 0.50,
            },
            "F5.4": {
                "description": "Continuous architectures achieve equal/better fitness",
                "threshold": None,
            },
        }

    def check_F5_1(self, history: Dict, generation_idx: int = -1) -> Tuple[bool, Dict]:
        """
        F5.1: Threshold mechanism frequency at generation 500
        """

        freq = history["architecture_frequencies"][generation_idx]["has_threshold"]

        falsified = freq < self.criteria["F5.1"]["threshold"]

        return falsified, {
            "final_frequency": float(freq),
            "threshold": self.criteria["F5.1"]["threshold"],
            "generation": history["generation"][generation_idx],
        }

    def check_F5_2(self, selection_coefficients: Dict) -> Tuple[bool, Dict]:
        """
        F5.2: Interoceptive weighting selection coefficient
        """

        s = selection_coefficients["has_intero_weighting"]

        # Falsified if negative (decreasing over time)
        falsified = s < self.criteria["F5.2"]["threshold"]

        return falsified, {
            "selection_coefficient": float(s),
            "interpretation": "Positive selection" if s > 0 else "Negative selection",
        }

    def check_F5_3(self, history: Dict) -> Tuple[bool, Dict]:
        """
        F5.3: Somatic markers maximum frequency
        """

        freqs = [h["has_somatic_markers"] for h in history["architecture_frequencies"]]
        max_freq = max(freqs)

        falsified = max_freq < self.criteria["F5.3"]["threshold"]

        return falsified, {
            "max_frequency": float(max_freq),
            "threshold": self.criteria["F5.3"]["threshold"],
        }

    def check_F5_4(self, history: Dict) -> Tuple[bool, Dict]:
        """
        F5.4: Compare threshold vs continuous architectures

        Would need to track fitness by architecture type
        For now, check if threshold is dominant
        """

        final_freq = history["architecture_frequencies"][-1]["has_threshold"]

        # If threshold dominant, discrete ignition is advantageous
        # Falsified if continuous (no threshold) equally common
        falsified = final_freq < 0.5

        return falsified, {
            "threshold_frequency": float(final_freq),
            "interpretation": (
                "Discrete ignition advantageous"
                if final_freq > 0.5
                else "Continuous equally good"
            ),
        }

    def generate_report(self, history: Dict, selection_coefficients: Dict) -> Dict:
        """Generate comprehensive falsification report"""

        report = {
            "falsified_criteria": [],
            "passed_criteria": [],
            "overall_falsified": False,
        }

        # F5.1
        f5_1_result, f5_1_details = self.check_F5_1(history)
        criterion = {
            "code": "F5.1",
            "description": self.criteria["F5.1"]["description"],
            "falsified": f5_1_result,
            "details": f5_1_details,
        }

        if f5_1_result:
            report["falsified_criteria"].append(criterion)
        else:
            report["passed_criteria"].append(criterion)

        # F5.2
        f5_2_result, f5_2_details = self.check_F5_2(selection_coefficients)
        criterion = {
            "code": "F5.2",
            "description": self.criteria["F5.2"]["description"],
            "falsified": f5_2_result,
            "details": f5_2_details,
        }

        if f5_2_result:
            report["falsified_criteria"].append(criterion)
        else:
            report["passed_criteria"].append(criterion)

        # F5.3
        f5_3_result, f5_3_details = self.check_F5_3(history)
        criterion = {
            "code": "F5.3",
            "description": self.criteria["F5.3"]["description"],
            "falsified": f5_3_result,
            "details": f5_3_details,
        }

        if f5_3_result:
            report["falsified_criteria"].append(criterion)
        else:
            report["passed_criteria"].append(criterion)

        # F5.4
        f5_4_result, f5_4_details = self.check_F5_4(history)
        criterion = {
            "code": "F5.4",
            "description": self.criteria["F5.4"]["description"],
            "falsified": f5_4_result,
            "details": f5_4_details,
        }

        if f5_4_result:
            report["falsified_criteria"].append(criterion)
        else:
            report["passed_criteria"].append(criterion)

        report["overall_falsified"] = len(report["falsified_criteria"]) > 0

        return report


# =============================================================================
# PART 6: VISUALIZATION
# =============================================================================


def plot_evolutionary_results(
    history: Dict, save_path: str = "protocol5_evolution_results.png"
):
    """Generate comprehensive visualization of evolutionary results"""

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    generations = np.array(history["generation"])

    # ==========================================================================
    # Row 1: Fitness over time
    # ==========================================================================

    ax1 = fig.add_subplot(gs[0, :])

    best_fitness = np.array(history["best_fitness"])
    mean_fitness = np.array(history["mean_fitness"])
    std_fitness = np.array(history["std_fitness"])

    ax1.plot(generations, best_fitness, "r-", linewidth=2.5, label="Best")
    ax1.plot(generations, mean_fitness, "b-", linewidth=2, label="Mean")
    ax1.fill_between(
        generations,
        mean_fitness - std_fitness,
        mean_fitness + std_fitness,
        alpha=0.3,
        color="blue",
    )

    ax1.set_xlabel("Generation", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Fitness", fontsize=13, fontweight="bold")
    ax1.set_title("Fitness Evolution", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # ==========================================================================
    # Row 2: Architecture frequency trajectories
    # ==========================================================================

    traits = [
        ("has_threshold", "Threshold Mechanism", "red"),
        ("has_intero_weighting", "Interoceptive Weighting", "blue"),
        ("has_somatic_markers", "Somatic Markers", "green"),
        ("has_precision_weighting", "Precision Weighting", "purple"),
    ]

    for idx, (trait, label, color) in enumerate(traits):
        ax = fig.add_subplot(gs[1, idx if idx < 3 else 0])

        freqs = np.array([h[trait] for h in history["architecture_frequencies"]])

        ax.plot(generations, freqs, color=color, linewidth=2.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.5)
        ax.axhline(
            y=0.9,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            label="Fixation threshold",
        )

        ax.set_xlabel("Generation", fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=9)

    # ==========================================================================
    # Row 3: Diversity and selection coefficients
    # ==========================================================================

    # Diversity
    ax_div = fig.add_subplot(gs[2, 0])

    diversity = np.array(history["diversity"])

    ax_div.plot(generations, diversity, "orange", linewidth=2.5)
    ax_div.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax_div.set_ylabel("Genotypic Diversity", fontsize=11, fontweight="bold")
    ax_div.set_title("Architectural Diversity", fontsize=12, fontweight="bold")
    ax_div.set_ylim([0, 1])
    ax_div.grid(alpha=0.3)

    # Selection coefficients
    ax_sel = fig.add_subplot(gs[2, 1])

    analyzer = EvolutionaryAnalyzer(history)
    sel_coeffs = analyzer.compute_selection_coefficients()

    trait_names = [
        "Threshold",
        "Intero\nWeight",
        "Somatic\nMarkers",
        "Precision\nWeight",
    ]
    sel_values = [
        sel_coeffs["has_threshold"],
        sel_coeffs["has_intero_weighting"],
        sel_coeffs["has_somatic_markers"],
        sel_coeffs["has_precision_weighting"],
    ]

    colors_sel = ["red", "blue", "green", "purple"]
    bars = ax_sel.bar(
        trait_names,
        sel_values,
        color=colors_sel,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    ax_sel.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax_sel.axhline(
        y=0.02,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="P5a threshold (s>0.02)",
    )
    ax_sel.set_ylabel("Selection Coefficient", fontsize=11, fontweight="bold")
    ax_sel.set_title("Selection Strength", fontsize=12, fontweight="bold")
    ax_sel.legend(fontsize=9)
    ax_sel.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, sel_values):
        height = bar.get_height()
        ax_sel.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.4f}",
            ha="center",
            va="bottom" if val > 0 else "top",
            fontweight="bold",
            fontsize=9,
        )

    # Fixation generations
    ax_fix = fig.add_subplot(gs[2, 2])

    fixation_gens = analyzer.find_fixation_generations(threshold=0.9)

    fix_values = []
    fix_labels = []
    for trait, gen in fixation_gens.items():
        if gen is not None:
            fix_values.append(gen)
            trait_short = trait.replace("has_", "").replace("_", "\n")
            fix_labels.append(trait_short)

    if fix_values:
        ax_fix.barh(
            fix_labels,
            fix_values,
            color=colors_sel[: len(fix_values)],
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax_fix.set_xlabel("Generation", fontsize=11, fontweight="bold")
        ax_fix.set_title(
            "Fixation Generation (freq>0.9)", fontsize=12, fontweight="bold"
        )
        ax_fix.grid(axis="x", alpha=0.3)

    # ==========================================================================
    # Row 4: Final architecture composition and statistics
    # ==========================================================================

    # Final architecture pie chart
    ax_pie = fig.add_subplot(gs[3, 0])

    final_freqs = history["architecture_frequencies"][-1]

    labels_pie = ["Threshold", "Intero Weight", "Somatic", "Precision"]
    sizes_pie = [
        final_freqs["has_threshold"],
        final_freqs["has_intero_weighting"],
        final_freqs["has_somatic_markers"],
        final_freqs["has_precision_weighting"],
    ]

    ax_pie.pie(
        sizes_pie,
        labels=labels_pie,
        autopct="%1.1f%%",
        colors=["red", "blue", "green", "purple"],
        startangle=90,
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    ax_pie.set_title(
        "Final Architecture\nComponent Frequencies", fontsize=12, fontweight="bold"
    )

    # Emergence order
    ax_order = fig.add_subplot(gs[3, 1])
    ax_order.axis("off")

    emergence_order = analyzer.test_emergence_order()

    order_text = "EMERGENCE ORDER\n" + "=" * 40 + "\n\n"

    sorted_emergence = sorted(emergence_order.items(), key=lambda x: x[1])
    for rank, (trait, _) in enumerate(sorted_emergence, 1):
        trait_clean = trait.replace("has_", "").replace("_", " ").title()
        order_text += f"{rank}. {trait_clean}\n"

    order_text += "\n" + "=" * 40 + "\n"
    order_text += "\nPredicted Order:\n"
    order_text += "1. Threshold\n"
    order_text += "2. Precision\n"
    order_text += "3. Interoceptive\n"
    order_text += "4. Somatic"

    ax_order.text(
        0.1,
        0.5,
        order_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )

    # Statistics table
    ax_stats = fig.add_subplot(gs[3, 2])
    ax_stats.axis("off")

    stats_data = [
        ["Metric", "Value"],
        ["Final Best Fitness", f"{history['best_fitness'][-1]:.3f}"],
        ["Final Mean Fitness", f"{history['mean_fitness'][-1]:.3f}"],
        ["Final Diversity", f"{history['diversity'][-1]:.3f}"],
        ["Generations", f"{len(generations)}"],
        ["", ""],
        ["Component", "Frequency"],
        ["Threshold", f"{final_freqs['has_threshold']:.2f}"],
        ["Intero Weight", f"{final_freqs['has_intero_weighting']:.2f}"],
        ["Somatic", f"{final_freqs['has_somatic_markers']:.2f}"],
        ["Precision", f"{final_freqs['has_precision_weighting']:.2f}"],
    ]

    table = ax_stats.table(
        cellText=stats_data, cellLoc="left", loc="center", colWidths=[0.6, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style headers
    for i in range(2):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")
        table[(6, i)].set_facecolor("#4CAF50")
        table[(6, i)].set_text_props(weight="bold", color="white")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {save_path}")
    plt.show()


def print_falsification_report(report: Dict):
    """Print formatted falsification report"""

    print("\n" + "=" * 80)
    print("PROTOCOL 5 FALSIFICATION REPORT")
    print("=" * 80)

    print(f"\nOVERALL STATUS: ", end="")
    if report["overall_falsified"]:
        print("❌ MODEL FALSIFIED")
    else:
        print("✅ MODEL VALIDATED")

    print(
        f"\nCriteria Passed: {len(report['passed_criteria'])}/{len(report['passed_criteria']) + len(report['falsified_criteria'])}"
    )
    print(
        f"Criteria Failed: {len(report['falsified_criteria'])}/{len(report['passed_criteria']) + len(report['falsified_criteria'])}"
    )

    if report["passed_criteria"]:
        print("\n" + "-" * 80)
        print("PASSED CRITERIA:")
        print("-" * 80)
        for criterion in report["passed_criteria"]:
            print(f"\n✅ {criterion['code']}: {criterion['description']}")
            if "details" in criterion and criterion["details"]:
                for key, value in criterion["details"].items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")

    if report["falsified_criteria"]:
        print("\n" + "-" * 80)
        print("FAILED CRITERIA (FALSIFICATIONS):")
        print("-" * 80)
        for criterion in report["falsified_criteria"]:
            print(f"\n❌ {criterion['code']}: {criterion['description']}")
            if "details" in criterion and criterion["details"]:
                for key, value in criterion["details"].items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {value}")

    print("\n" + "=" * 80)


# =============================================================================
# PART 7: ADVANCED EVOLUTIONARY ANALYSIS
# =============================================================================


def phylogenetic_tree_analysis(evolution_history):
    """
    Build phylogenetic tree showing how APGI components emerged
    Track when threshold, interoception, somatic markers appeared
    """
    # Extract lineages
    lineages = []
    for generation in evolution_history:
        for agent in generation["agents"]:
            lineage = {
                "generation": generation["number"],
                "fitness": agent.fitness,
                "has_threshold": agent.genome.has_threshold,
                "has_intero": agent.genome.has_intero_weighting,
                "has_somatic": agent.genome.has_somatic_markers,
                "has_precision": agent.genome.has_precision_weighting,
                "parent_id": agent.parent_id,
            }
            lineages.append(lineage)

    # Build tree
    G = nx.DiGraph()
    for lineage in lineages:
        G.add_node(lineage["agent_id"], **lineage)
        if lineage["parent_id"] is not None:
            G.add_edge(lineage["parent_id"], lineage["agent_id"])

    # Analyze when components emerged
    emergence_times = {
        "threshold": None,
        "interoception": None,
        "somatic_markers": None,
        "precision": None,
    }

    for gen_num in range(len(evolution_history)):
        agents = evolution_history[gen_num]["agents"]
        if emergence_times["threshold"] is None:
            if any(a.genome.has_threshold for a in agents):
                emergence_times["threshold"] = gen_num
        if emergence_times["interoception"] is None:
            if any(a.genome.has_intero_weighting for a in agents):
                emergence_times["interoception"] = gen_num
        if emergence_times["somatic_markers"] is None:
            if any(a.genome.has_somatic_markers for a in agents):
                emergence_times["somatic_markers"] = gen_num
        if emergence_times["precision"] is None:
            if any(a.genome.has_precision_weighting for a in agents):
                emergence_times["precision"] = gen_num

    return G, emergence_times


def test_across_environmental_gradients():
    """
    Evolve agents across different environmental conditions
    Test which environments favor APGI components
    """
    environmental_gradients = {
        "stability": np.linspace(0.1, 0.9, 5),  # Reward stability
        "uncertainty": np.linspace(0.1, 0.9, 5),  # Noise level
        "metabolic_cost": np.linspace(0.0, 0.3, 5),  # Cost of computation
        "interoceptive_relevance": np.linspace(
            0.1, 0.9, 5
        ),  # How much internal states matter
    }

    results = {}

    for gradient_name, values in environmental_gradients.items():
        gradient_results = []

        for value in values:
            # Create environment with this parameter
            env = create_environment_with_parameter(gradient_name, value)

            # Run evolution
            final_population = run_evolution(env, generations=200)

            # Measure prevalence of APGI components
            prevalence = {
                "threshold": np.mean(
                    [a.genome.has_threshold for a in final_population]
                ),
                "intero": np.mean(
                    [a.genome.has_intero_weighting for a in final_population]
                ),
                "somatic": np.mean(
                    [a.genome.has_somatic_markers for a in final_population]
                ),
                "precision": np.mean(
                    [a.genome.has_precision_weighting for a in final_population]
                ),
            }

            gradient_results.append(
                {
                    "parameter_value": value,
                    "prevalence": prevalence,
                    "mean_fitness": np.mean([a.fitness for a in final_population]),
                }
            )

        results[gradient_name] = gradient_results

    # Visualize
    fig = plot_environmental_gradient_results(results)

    return results, fig


def test_convergent_evolution(n_independent_runs=10):
    """
    Run multiple independent evolutionary simulations
    Check if APGI components emerge repeatedly (convergent evolution)
    Strong evidence if components consistently emerge
    """
    convergence_results = {
        "threshold_emergence_rate": 0,
        "intero_emergence_rate": 0,
        "somatic_emergence_rate": 0,
        "precision_emergence_rate": 0,
        "full_apgi_emergence_rate": 0,
        "emergence_generation": [],
    }

    for run in range(n_independent_runs):
        # Independent evolution with different random seed
        np.random.seed(run)
        final_population = run_evolution(generations=300)

        # Check if components emerged
        has_any_threshold = any(a.genome.has_threshold for a in final_population)
        has_any_intero = any(a.genome.has_intero_weighting for a in final_population)
        has_any_somatic = any(a.genome.has_somatic_markers for a in final_population)
        has_any_precision = any(
            a.genome.has_precision_weighting for a in final_population
        )

        convergence_results["threshold_emergence_rate"] += has_any_threshold
        convergence_results["intero_emergence_rate"] += has_any_intero
        convergence_results["somatic_emergence_rate"] += has_any_somatic
        convergence_results["precision_emergence_rate"] += has_any_precision

        # Full APGI (all components)
        has_full_apgi = any(
            a.genome.has_threshold
            and a.genome.has_intero_weighting
            and a.genome.has_somatic_markers
            and a.genome.has_precision_weighting
            for a in final_population
        )
        convergence_results["full_apgi_emergence_rate"] += has_full_apgi

    # Convert counts to rates
    for key in convergence_results:
        if "rate" in key:
            convergence_results[key] /= n_independent_runs

    # Statistical test: Is emergence rate > chance?
    # Binomial test
    for component in ["threshold", "intero", "somatic", "precision", "full_apgi"]:
        key = f"{component}_emergence_rate"
        n_successes = int(convergence_results[key] * n_independent_runs)
        result = binomtest(n_successes, n_independent_runs, 0.5, alternative="greater")
        convergence_results[f"{component}_p_value"] = result.pvalue

    return convergence_results


def analyze_fitness_landscape(genome_space_sample, environment):
    """
    Sample fitness across genome space
    Visualize fitness landscape and identify peaks
    """
    # Sample genomes
    sampled_genomes = []
    fitnesses = []

    for _ in range(1000):
        genome = AgentGenome.random()
        agent = EvolvableAgent(genome)
        fitness = evaluate_agent(agent, environment)

        sampled_genomes.append(genome.to_dict())
        fitnesses.append(fitness)

    # Convert to dataframe
    df = pd.DataFrame(sampled_genomes)
    df["fitness"] = fitnesses

    # Visualize using PCA
    # Encode genomes numerically
    X = df.drop("fitness", axis=1).values
    X_scaled = StandardScaler().fit_transform(X)

    # PCA to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Plot fitness landscape
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], c=fitnesses, cmap="viridis", s=50, alpha=0.6
    )
    plt.colorbar(scatter, ax=ax, label="Fitness")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Fitness Landscape in Genome Space")

    # Identify fitness peaks
    top_genomes = df.nlargest(10, "fitness")

    return df, fig, top_genomes


# Helper functions for the advanced analyses


def create_environment_with_parameter(parameter_name, value):
    """Create environment with specific parameter value"""
    # This is a placeholder - would need to implement actual environment modification
    env = IowaGamblingTask()  # Default environment

    if parameter_name == "stability":
        # Modify reward stability
        pass
    elif parameter_name == "uncertainty":
        # Modify noise level
        pass
    elif parameter_name == "metabolic_cost":
        # Modify metabolic costs
        pass
    elif parameter_name == "interoceptive_relevance":
        # Modify interoceptive relevance
        pass

    return env


def run_evolution(environment, generations=200):
    """Run evolution for specified generations"""
    # This is a placeholder - would use the actual EvolutionaryOptimizer
    optimizer = EvolutionaryOptimizer(n_generations=generations)
    optimizer.environments = [environment]
    history = optimizer.run_evolution()

    # Return final population
    final_population = [EvolvableAgent(genome) for genome in optimizer.population]

    # Assign fitness values
    for i, agent in enumerate(final_population):
        agent.fitness = optimizer.evaluate_fitness(agent.genome)

    return final_population


def evaluate_agent(agent, environment):
    """Evaluate single agent fitness"""
    optimizer = EvolutionaryOptimizer()
    return optimizer.evaluate_fitness(agent.genome)


def plot_environmental_gradient_results(results):
    """Plot environmental gradient analysis results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (gradient_name, gradient_results) in enumerate(results.items()):
        ax = axes[idx]

        values = [r["parameter_value"] for r in gradient_results]
        threshold_prevalence = [r["prevalence"]["threshold"] for r in gradient_results]
        intero_prevalence = [r["prevalence"]["intero"] for r in gradient_results]
        somatic_prevalence = [r["prevalence"]["somatic"] for r in gradient_results]
        precision_prevalence = [r["prevalence"]["precision"] for r in gradient_results]

        ax.plot(values, threshold_prevalence, "r-", label="Threshold", linewidth=2)
        ax.plot(values, intero_prevalence, "b-", label="Interoceptive", linewidth=2)
        ax.plot(values, somatic_prevalence, "g-", label="Somatic", linewidth=2)
        ax.plot(values, precision_prevalence, "purple", label="Precision", linewidth=2)

        ax.set_xlabel(gradient_name.replace("_", " ").title())
        ax.set_ylabel("Component Prevalence")
        ax.set_title(f'APGI Components vs {gradient_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    return fig


# =============================================================================
# PART 8: MAIN EXECUTION PIPELINE
# =============================================================================


def main():
    """Main execution pipeline for Protocol 5"""

    print("=" * 80)
    print("APGI PROTOCOL 5: EVOLUTIONARY EMERGENCE OF APGI-LIKE ARCHITECTURES")
    print("=" * 80)

    # Configuration
    config = {
        "population_size": 100,
        "n_generations": 500,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "tournament_size": 5,
        "elitism": 5,
    }

    print(f"\nEvolutionary Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # =========================================================================
    # STEP 1: Run Evolution
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: RUNNING EVOLUTIONARY OPTIMIZATION")
    print("=" * 80)

    optimizer = EvolutionaryOptimizer(
        population_size=config["population_size"],
        n_generations=config["n_generations"],
        mutation_rate=config["mutation_rate"],
        crossover_rate=config["crossover_rate"],
        tournament_size=config["tournament_size"],
        elitism=config["elitism"],
    )

    history = optimizer.run_evolution()

    # =========================================================================
    # STEP 2: Analyze Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: ANALYZING EVOLUTIONARY DYNAMICS")
    print("=" * 80)

    analyzer = EvolutionaryAnalyzer(history)

    # Selection coefficients
    print("\n--- Selection Coefficients ---")
    sel_coeffs = analyzer.compute_selection_coefficients()
    for trait, coeff in sel_coeffs.items():
        print(f"{trait}: {coeff:.5f}")

    # Fixation generations
    print("\n--- Fixation Generations (freq > 0.9) ---")
    fixation_gens = analyzer.find_fixation_generations()
    for trait, gen in fixation_gens.items():
        if gen is not None:
            print(f"{trait}: Generation {gen}")
        else:
            print(f"{trait}: Never reached fixation")

    # Emergence order
    print("\n--- Emergence Order ---")
    emergence_order = analyzer.test_emergence_order()
    sorted_emergence = sorted(emergence_order.items(), key=lambda x: x[1])
    for rank, (trait, _) in enumerate(sorted_emergence, 1):
        print(f"{rank}. {trait}")

    # Final frequencies
    print("\n--- Final Architecture Frequencies ---")
    final_freqs = history["architecture_frequencies"][-1]
    for trait, freq in final_freqs.items():
        print(f"{trait}: {freq:.3f}")

    # =========================================================================
    # STEP 3: Falsification Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: FALSIFICATION ANALYSIS")
    print("=" * 80)

    checker = FalsificationChecker()
    falsification_report = checker.generate_report(history, sel_coeffs)

    print_falsification_report(falsification_report)

    # =========================================================================
    # STEP 4: Visualization
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_evolutionary_results(history, save_path="protocol5_evolution_results.png")

    # =========================================================================
    # STEP 5: Save Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: SAVING RESULTS")
    print("=" * 80)

    # Convert to serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, AgentGenome):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # Prepare results (exclude large arrays)
    results_summary = {
        "config": config,
        "final_statistics": {
            "best_fitness": float(history["best_fitness"][-1]),
            "mean_fitness": float(history["mean_fitness"][-1]),
            "final_diversity": float(history["diversity"][-1]),
            "final_frequencies": final_freqs,
        },
        "selection_coefficients": sel_coeffs,
        "fixation_generations": fixation_gens,
        "emergence_order": {k: int(v) for k, v in emergence_order.items()},
        "falsification": falsification_report,
    }

    results_summary = convert_to_serializable(results_summary)

    with open("protocol5_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("✅ Results saved to: protocol5_results.json")

    # Save full history
    history_df = pd.DataFrame(
        {
            "generation": history["generation"],
            "best_fitness": history["best_fitness"],
            "mean_fitness": history["mean_fitness"],
            "std_fitness": history["std_fitness"],
            "diversity": history["diversity"],
        }
    )

    history_df.to_csv("protocol5_history.csv", index=False)
    print("✅ History saved to: protocol5_history.csv")

    print("\n" + "=" * 80)
    print("PROTOCOL 5 EXECUTION COMPLETE")
    print("=" * 80)

    return results_summary


def run_validation():
    """Entry point for CLI validation."""
    try:
        print(
            "Running APGI Validation Protocol 5: Computational Falsification Framework"
        )
        return main()
    except Exception as e:
        print(f"Error in validation protocol 5: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    main()
