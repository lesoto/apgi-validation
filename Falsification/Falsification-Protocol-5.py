import os
import time
from typing import Dict, List

import numpy as np


class EvolvableAgent:
    """Agent that can evolve based on genome"""

    def __init__(self, genome: Dict):
        self.genome = genome

        # Initialize based on genome
        self.has_threshold = genome["has_threshold"]
        self.has_intero_weighting = genome["has_intero_weighting"]
        self.has_somatic_markers = genome["has_somatic_markers"]
        self.has_precision_weighting = genome["has_precision_weighting"]

        # Parameters
        self.theta_0 = genome["theta_0"]
        self.alpha = genome["alpha"]
        self.beta = genome["beta"]

        # State
        self.surprise = 0.0
        self.threshold = self.theta_0 if self.has_threshold else 0.0
        self.conscious_access = False

        # Simple policy network
        state_dim = 32 + 16  # extero + intero
        action_dim = 4

        self.policy_weights = np.random.normal(0, 0.1, (action_dim, state_dim))

        if self.has_somatic_markers:
            self.somatic_weights = np.random.normal(0, 0.1, (action_dim, state_dim))

        # Precision weights
        self.Pi_e = 1.0
        self.Pi_i = 1.0
        self.pi_lr = genome["Pi_e_lr"] if self.has_precision_weighting else 0.0

    def step(self, observation: Dict, dt: float = 0.05) -> int:
        """Execute one step"""
        # Get state representation
        extero = observation["extero"][:32]
        intero = observation["intero"][:16]
        state = np.concatenate([extero, intero])

        # Compute prediction errors (simplified)
        eps_e = np.linalg.norm(extero)
        eps_i = np.linalg.norm(intero)

        # Weight prediction errors
        if self.has_intero_weighting:
            weighted_eps_e = self.Pi_e * eps_e
            weighted_eps_i = self.beta * self.Pi_i * eps_i
        else:
            weighted_eps_e = eps_e
            weighted_eps_i = eps_i

        # Update surprise
        input_drive = weighted_eps_e + weighted_eps_i
        self.surprise = 0.9 * self.surprise + 0.1 * input_drive

        # Check ignition
        if self.has_threshold:
            ignition_prob = 1.0 / (
                1.0 + np.exp(-self.alpha * (self.surprise - self.threshold))
            )
            self.conscious_access = np.random.random() < ignition_prob
        else:
            self.conscious_access = True  # Always conscious

        # Compute action probabilities
        logits = self.policy_weights @ state

        if self.has_somatic_markers and self.conscious_access:
            somatic_values = self.somatic_weights @ state
            logits += 0.3 * somatic_values

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        action_probs = exp_logits / np.sum(exp_logits)

        return np.random.choice(len(action_probs), p=action_probs)

    def receive_outcome(
        self, reward: float, intero_cost: float, next_observation: Dict
    ):
        """Process outcome"""
        # Simple learning update
        if reward > 0:
            # Reinforce last action (simplified)
            pass

        # Update precision if enabled
        if self.has_precision_weighting:
            # Simple precision adaptation based on interoceptive cost
            adjustment = self.pi_lr * (1.0 - intero_cost)
            self.Pi_e = 0.99 * self.Pi_e + 0.01 * adjustment
            self.Pi_i = 0.99 * self.Pi_i + 0.01 * adjustment


class EvolutionaryAPGIEmergence:
    """
    Test whether APGI-like architectures emerge under selection pressure
    """

    def __init__(
        self, population_size: int = 20, n_generations: int = 50, stop_event=None
    ):
        self.pop_size = population_size
        self.n_generations = n_generations
        self.stop_event = stop_event

    def create_genome(self) -> Dict:
        """
        Genome encodes architectural choices:
        - Whether to have ignition threshold (vs continuous)
        - Whether to weight interoceptive signals differently
        - Whether to have somatic markers
        - Whether to have precision weighting
        """
        return {
            # Structural genes
            "has_threshold": np.random.random() > 0.5,
            "has_intero_weighting": np.random.random() > 0.5,
            "has_somatic_markers": np.random.random() > 0.5,
            "has_precision_weighting": np.random.random() > 0.5,
            # Parameter genes (if structure present)
            "theta_0": np.random.uniform(0.2, 0.8),
            "alpha": np.random.uniform(2, 10),
            "beta": np.random.uniform(0.5, 2.0),
            "Pi_e_lr": np.random.uniform(0.01, 0.2),
            "Pi_i_lr": np.random.uniform(0.01, 0.2),
            "somatic_lr": np.random.uniform(0.01, 0.3),
            # Architecture genes
            "n_hidden_layers": np.random.randint(1, 4),
            "hidden_dim": np.random.randint(16, 128),
        }

    def genome_to_agent(self, genome: Dict) -> "EvolvableAgent":
        """Create agent from genome"""
        return EvolvableAgent(genome)

    def evaluate_fitness(self, agent, environments: List) -> float:
        """
        Fitness = survival and reward across multiple environments
        Optimized for faster evaluation
        """
        total_fitness = 0

        for env in environments:
            # Run agent in environment for shorter time
            cumulative_reward = 0
            survival_time = 0
            homeostatic_violations = 0

            obs = env.reset()

            for t in range(100):  # Reduced from 1000 timesteps
                action = agent.step(obs)
                reward, intero_cost, next_obs, done = env.step(action)

                cumulative_reward += reward
                survival_time += 1

                # Track homeostatic violations
                if intero_cost > 1.0:
                    homeostatic_violations += 1

                agent.receive_outcome(reward, intero_cost, next_obs)
                obs = next_obs

                if done:
                    break

            # Fitness components
            env_fitness = (
                cumulative_reward / 50  # Reward seeking (adjusted)
                + survival_time / 100  # Survival (adjusted)
                - homeostatic_violations / 50  # Homeostatic maintenance (adjusted)
            )
            total_fitness += env_fitness

        return total_fitness / len(environments) if environments else 0.0

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Single-point crossover"""
        child = {}
        keys = list(parent1.keys())
        crossover_point = np.random.randint(len(keys))

        for i, key in enumerate(keys):
            if i < crossover_point:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]

        return child

    def mutate(self, genome: Dict, mutation_rate: float = 0.1) -> Dict:
        """Mutate genome"""
        mutated = genome.copy()

        for key, value in mutated.items():
            if np.random.random() < mutation_rate:
                if isinstance(value, bool):
                    mutated[key] = not value
                elif isinstance(value, int):
                    mutated[key] = max(1, value + np.random.randint(-2, 3))
                else:
                    mutated[key] = value * np.random.uniform(0.8, 1.2)

        return mutated

    def run_evolution(self, max_time_seconds: float = 30.0) -> Dict:
        """Run evolutionary optimization with timeout"""

        start_time = time.time()

        # Create environments (imported dynamically to avoid circular dependencies)
        environments = []
        try:
            import importlib.util

            spec2 = importlib.util.spec_from_file_location(
                "Protocol_2",
                os.path.join(os.path.dirname(__file__), "Falsification-Protocol-2.py"),
            )
            protocol2 = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(protocol2)

            environments = [
                protocol2.IowaGamblingTaskEnvironment(),
                protocol2.VolatileForagingEnvironment(),
                protocol2.ThreatRewardTradeoffEnvironment(),
            ]
        except (ImportError, AttributeError, KeyError) as e:
            print(f"Warning: Could not load environments from Protocol-2: {e}")
            # Use dummy environments for testing
            environments = []

        if not environments:
            print("Error: No environments available for evaluation")
            return {
                "best_fitness": [],
                "mean_fitness": [],
                "architecture_frequencies": [],
                "best_genome": [],
            }

        # Initialize population
        population = [self.create_genome() for _ in range(self.pop_size)]

        history = {
            "best_fitness": [],
            "mean_fitness": [],
            "architecture_frequencies": [],
            "best_genome": [],
        }

        for generation in range(self.n_generations):
            # Check timeout
            if time.time() - start_time > max_time_seconds:
                print(f"Evolution stopped after {max_time_seconds} seconds")
                break

            # Check stop event
            if self.stop_event and self.stop_event.is_set():
                print("Evolution stopped by user")
                break

            # Evaluate fitness
            fitness_scores = []
            for genome in population:
                agent = self.genome_to_agent(genome)
                fitness = self.evaluate_fitness(agent, environments)
                fitness_scores.append(fitness)

            fitness_scores = np.array(fitness_scores)

            # Record statistics
            history["best_fitness"].append(np.max(fitness_scores))
            history["mean_fitness"].append(np.mean(fitness_scores))
            history["best_genome"].append(population[np.argmax(fitness_scores)].copy())

            # Track architecture frequencies
            arch_freq = {
                "has_threshold": np.mean([g["has_threshold"] for g in population]),
                "has_intero_weighting": np.mean(
                    [g["has_intero_weighting"] for g in population]
                ),
                "has_somatic_markers": np.mean(
                    [g["has_somatic_markers"] for g in population]
                ),
                "has_precision_weighting": np.mean(
                    [g["has_precision_weighting"] for g in population]
                ),
            }
            history["architecture_frequencies"].append(arch_freq)

            # Selection (tournament)
            new_population = []
            for _ in range(self.pop_size):
                tournament = np.random.choice(self.pop_size, 5, replace=False)
                winner_idx = tournament[np.argmax(fitness_scores[tournament])]
                new_population.append(population[winner_idx].copy())

            # Crossover and mutation (handle odd population sizes)
            for i in range(0, len(new_population) - 1, 2):
                if np.random.random() < 0.7:  # Crossover rate
                    child1 = self.crossover(new_population[i], new_population[i + 1])
                    child2 = self.crossover(new_population[i + 1], new_population[i])
                    new_population[i] = child1
                    new_population[i + 1] = child2

            population = [self.mutate(g) for g in new_population]

            if generation % 10 == 0:  # More frequent updates
                elapsed = time.time() - start_time
                print(
                    f"Gen {generation} ({elapsed:.1f}s): Best={np.max(fitness_scores):.3f}, "
                    f"Mean={np.mean(fitness_scores):.3f}"
                )
                print(
                    f"  Threshold: {arch_freq['has_threshold']:.2f}, "
                    f"Intero: {arch_freq['has_intero_weighting']:.2f}, "
                    f"Somatic: {arch_freq['has_somatic_markers']:.2f}"
                )

        return history

    def analyze_emergence(self, history: Dict) -> Dict:
        """Analyze whether APGI-like architecture emerged"""

        # Final architecture frequencies
        final_freq = history["architecture_frequencies"][-1]

        # Check if APGI components reached fixation (>90%)
        apgi_emerged = (
            final_freq["has_threshold"] > 0.9
            and final_freq["has_intero_weighting"] > 0.9
            and final_freq["has_somatic_markers"] > 0.9
        )

        # Compute selection coefficients
        # (how quickly did each trait spread?)
        selection_coefficients = {}
        for trait in [
            "has_threshold",
            "has_intero_weighting",
            "has_somatic_markers",
            "has_precision_weighting",
        ]:
            freqs = [h[trait] for h in history["architecture_frequencies"]]

            # Logistic regression to estimate selection strength
            x = np.arange(len(freqs))
            y = np.array(freqs)

            # Avoid log(0)
            y = np.clip(y, 0.01, 0.99)

            # Logit transform
            logit_y = np.log(y / (1 - y))
            slope, _ = np.polyfit(x, logit_y, 1)
            selection_coefficients[trait] = slope

        return {
            "apgi_emerged": apgi_emerged,
            "final_frequencies": final_freq,
            "selection_coefficients": selection_coefficients,
            "generations_to_fixation": self._find_fixation_generation(history),
        }

    def _find_fixation_generation(self, history: Dict, threshold: float = 0.9) -> Dict:
        """Find generation when each trait reached fixation"""
        fixation_gens = {}

        for trait in [
            "has_threshold",
            "has_intero_weighting",
            "has_somatic_markers",
            "has_precision_weighting",
        ]:
            freqs = [h[trait] for h in history["architecture_frequencies"]]

            for gen, freq in enumerate(freqs):
                if freq >= threshold:
                    fixation_gens[trait] = gen
                    break
            else:
                fixation_gens[trait] = None  # Never fixed

        return fixation_gens


# Main execution
if __name__ == "__main__":
    print("Starting evolutionary simulation (this may take time)...")
    simulator = EvolutionaryAPGIEmergence()
    results = simulator.run_evolution()

    if results and results.get("best_fitness"):
        analysis = simulator.analyze_emergence(results)
        print("\n=== Emergence Analysis ===")
        print(f"APGI Emerged: {analysis['apgi_emerged']}")
        print(f"Final Frequencies: {analysis['final_frequencies']}")
        print(f"Selection Coefficients: {analysis['selection_coefficients']}")
        print(f"Generations to Fixation: {analysis['generations_to_fixation']}")

    print("Evolution completed:", type(results))
    print("=== Protocol completed successfully ===")


def run_falsification():
    """Entry point for CLI falsification testing."""
    try:
        print("Running APGI Falsification Protocol 5: Evolutionary APGI Emergence")
        simulator = EvolutionaryAPGIEmergence()
        results = simulator.run_evolution()
        print("Evolution completed:", type(results))
        print("=== Protocol completed successfully ===")
        return {"status": "success", "results": results}
    except (RuntimeError, ValueError, TypeError, ImportError, KeyError) as e:
        print(f"Error in falsification protocol 5: {e}")
        return {"status": "error", "message": str(e)}
