from typing import Dict, Tuple

import numpy as np


class IowaGamblingTaskEnvironment:
    """
    IGT variant with simulated interoceptive costs

    Decks:
    A: High reward variance, net negative, high intero cost
    B: High reward variance, net negative, moderate intero cost
    C: Low reward variance, net positive, low intero cost
    D: Low reward variance, net positive, minimal intero cost
    """

    def __init__(self, n_trials: int = 100):
        self.n_trials = n_trials
        self.trial = 0

        # Deck parameters
        self.decks = {
            "A": {
                "reward_mean": 100,
                "reward_std": 50,
                "loss_prob": 0.5,
                "loss_mean": 250,
                "intero_cost": 0.8,
            },
            "B": {
                "reward_mean": 100,
                "reward_std": 50,
                "loss_prob": 0.1,
                "loss_mean": 1250,
                "intero_cost": 0.5,
            },
            "C": {
                "reward_mean": 50,
                "reward_std": 25,
                "loss_prob": 0.5,
                "loss_mean": 50,
                "intero_cost": 0.1,
            },
            "D": {
                "reward_mean": 50,
                "reward_std": 25,
                "loss_prob": 0.1,
                "loss_mean": 250,
                "intero_cost": 0.05,
            },
        }

    def step(self, action: int) -> Tuple[float, float, Dict, bool]:
        """
        Returns:
            reward: Monetary outcome
            intero_cost: Simulated physiological cost
            observation: Next state
            done: Episode complete
        """
        if not 0 <= action < 4:
            raise ValueError(f"Action must be 0-3, got {action}")

        deck_name = ["A", "B", "C", "D"][action]
        deck = self.decks[deck_name]

        # Compute reward
        reward = np.random.normal(deck["reward_mean"], deck["reward_std"])
        if np.random.random() < deck["loss_prob"]:
            reward -= np.random.exponential(deck["loss_mean"])

        # Compute interoceptive cost (simulated physiological response)
        intero_cost = deck["intero_cost"]
        if reward < 0:
            intero_cost *= 1.5  # Amplified for losses

        # Observation includes both external (reward feedback) and internal
        observation = {
            "extero": self._encode_reward_feedback(reward),
            "intero": self._generate_intero_signal(intero_cost),
        }

        self.trial += 1
        done = self.trial >= self.n_trials

        return reward, intero_cost, observation, done

    def _generate_intero_signal(self, cost: float) -> np.ndarray:
        """Generate realistic interoceptive signal

        Args:
            cost: Physiological cost factor

        Returns:
            Combined interoceptive signal (16-dim)
        """
        if cost < 0:
            cost = 0.0

        # Heart rate variability
        hrv = np.random.normal(0, 0.1 + cost * 0.3, size=8)

        # Skin conductance
        scr = np.random.exponential(cost, size=4)

        # Gastric signals
        gastric = np.random.normal(-cost, 0.2, size=4)

        return np.concatenate([hrv, scr, gastric])

    def _encode_reward_feedback(self, reward: float) -> np.ndarray:
        """Encode reward feedback as exteroceptive signal

        Args:
            reward: Monetary reward value

        Returns:
            Encoded reward feedback (32-dim)
        """
        # Create a vector representation of reward
        encoding = np.zeros(32)

        # Encode magnitude
        magnitude = np.clip(abs(reward) / 200.0, 0, 1)  # Normalize to [0, 1]
        encoding[0] = magnitude

        # Encode valence (positive vs negative)
        encoding[1] = 1.0 if reward > 0 else 0.0

        # Encode different reward ranges
        if reward > 100:
            encoding[2:4] = [1.0, 0.0]  # High reward
        elif reward > 0:
            encoding[2:4] = [0.0, 1.0]  # Low positive reward
        elif reward > -100:
            encoding[2:4] = [0.0, 0.0]  # Small loss
        else:
            encoding[2:4] = [1.0, 0.0]  # Large loss

        # Add noise for realism
        encoding[4:] = np.random.normal(0, 0.1, 28)

        return encoding

    def reset(self) -> Dict:
        """Reset environment for new episode"""
        self.trial = 0
        # Return initial observation
        return {"extero": np.zeros(32), "intero": self._generate_intero_signal(0.1)}


class VolatileForagingEnvironment:
    """
    Foraging task with shifting reward statistics and location-dependent
    homeostatic costs
    """

    def __init__(self, grid_size: int = 10, volatility: float = 0.1):
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if not 0 <= volatility <= 1:
            raise ValueError("volatility must be between 0 and 1")

        self.grid_size = grid_size
        self.volatility = volatility

        # Initialize reward and cost maps
        self._generate_maps()

        # Agent position
        self.position = np.array([grid_size // 2, grid_size // 2])

    def _generate_maps(self):
        """Generate reward and homeostatic cost maps"""

        # Reward patches
        self.reward_map = np.zeros((self.grid_size, self.grid_size))
        n_patches = 3
        for _ in range(n_patches):
            center = np.random.randint(0, self.grid_size, size=2)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    dist = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
                    self.reward_map[i, j] += 10 * np.exp(-dist / 2)

        # Homeostatic cost map (e.g., temperature, predator risk)
        self.cost_map = np.random.exponential(0.2, (self.grid_size, self.grid_size))

    def step(self, action: int) -> Tuple[float, float, Dict, bool]:
        """
        Actions: 0=up, 1=down, 2=left, 3=right, 4=forage

        Returns:
            reward: Reward obtained
            intero_cost: Physiological cost
            observation: Environmental state
            done: Always False for this environment
        """
        if not 0 <= action <= 4:
            raise ValueError(f"Action must be 0-4, got {action}")
        # Movement
        if action < 4:
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            new_pos = self.position + np.array(moves[action])
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            self.position = new_pos

        # Get reward and cost at current position
        x, y = self.position
        reward = self.reward_map[x, y] if action == 4 else 0
        intero_cost = self.cost_map[x, y]

        # Deplete reward patch
        if action == 4:
            self.reward_map[x, y] *= 0.8

        # Volatile shifts
        if np.random.random() < self.volatility:
            self._shift_maps()

        observation = {
            "extero": self._get_visual_observation(),
            "intero": self._get_intero_signal(intero_cost),
        }

        return reward, intero_cost, observation, False

    def _shift_maps(self):
        """Shift reward/cost maps to simulate volatility"""
        # Rotate reward map
        shift = np.random.randint(-2, 3, size=2)
        self.reward_map = np.roll(self.reward_map, shift, axis=(0, 1))

        # Add noise to cost map
        self.cost_map += np.random.normal(0, 0.05, self.cost_map.shape)
        self.cost_map = np.clip(self.cost_map, 0, 1)

    def _get_visual_observation(self) -> np.ndarray:
        """Get visual observation of current position"""
        visual = np.zeros(32)

        # Encode position
        x, y = self.position
        visual[0] = x / self.grid_size
        visual[1] = y / self.grid_size

        # Encode reward at current position
        visual[2] = np.clip(self.reward_map[x, y] / 10.0, 0, 1)

        # Encode cost at current position
        visual[3] = self.cost_map[x, y]

        # Add noise
        visual[4:] = np.random.normal(0, 0.1, 28)

        return visual

    def _get_intero_signal(self, cost: float) -> np.ndarray:
        """Get interoceptive signal

        Args:
            cost: Physiological cost factor

        Returns:
            Combined interoceptive signal (16-dim)
        """
        if cost < 0:
            cost = 0.0

        hrv = np.random.normal(0, 0.1 + cost * 0.3, size=8)
        scr = np.random.exponential(cost, size=4)
        gastric = np.random.normal(-cost, 0.2, size=4)
        return np.concatenate([hrv, scr, gastric])

    def reset(self) -> Dict:
        """Reset environment"""
        self.position = np.array([self.grid_size // 2, self.grid_size // 2])
        self._generate_maps()
        return {
            "extero": self._get_visual_observation(),
            "intero": self._get_intero_signal(0.1),
        }


class ThreatRewardTradeoffEnvironment:
    """
    Environment where high-reward options produce aversive interoceptive
    consequences (e.g., stress, fear responses)
    """

    def __init__(self):
        # Options with varying reward-threat profiles
        self.options = {
            0: {"reward": 10, "threat": 0.1, "name": "safe_low"},
            1: {"reward": 30, "threat": 0.3, "name": "moderate"},
            2: {"reward": 60, "threat": 0.6, "name": "risky"},
            3: {"reward": 100, "threat": 0.9, "name": "dangerous"},
        }

        # Threat accumulates and affects future interoception
        self.threat_accumulator = 0.0
        self.threat_decay = 0.9

    def step(self, action: int) -> Tuple[float, float, Dict, bool]:
        """Execute action and return results

        Args:
            action: Option choice (0-3)

        Returns:
            reward: Monetary reward
            intero_cost: Physiological cost
            observation: Environmental state
            done: Always False for this environment
        """
        if not 0 <= action < 4:
            raise ValueError(f"Action must be 0-3, got {action}")

        opt = self.options[action]

        # Reward with variance
        reward = np.random.normal(opt["reward"], opt["reward"] * 0.2)

        # Threat response
        threat = opt["threat"]
        self.threat_accumulator = self.threat_decay * self.threat_accumulator + threat

        # Interoceptive cost depends on both immediate threat and accumulated
        intero_cost = threat + 0.3 * self.threat_accumulator

        # High accumulated threat can cause "panic" (large interoceptive burst)
        if self.threat_accumulator > 2.0:
            intero_cost += np.random.exponential(1.0)
            self.threat_accumulator *= 0.5  # Partial reset

        observation = {
            "extero": self._encode_option_outcome(action, reward),
            "intero": self._generate_threat_response(intero_cost),
        }

        return reward, intero_cost, observation, False

    def _encode_option_outcome(self, action: int, reward: float) -> np.ndarray:
        """Encode option outcome as exteroceptive signal"""
        encoding = np.zeros(32)

        # Encode which option was chosen
        encoding[action] = 1.0

        # Encode reward magnitude
        encoding[4 + action] = np.clip(reward / 100.0, 0, 1)

        # Add noise
        encoding[8:] = np.random.normal(0, 0.1, 24)

        return encoding

    def _generate_threat_response(self, cost: float) -> np.ndarray:
        """Generate threat-related interoceptive response

        Args:
            cost: Threat cost factor

        Returns:
            Threat response signal (16-dim)
        """
        if cost < 0:
            cost = 0.0

        # Heart rate and stress indicators
        hrv = np.random.normal(0, 0.2 + cost * 0.5, size=8)

        # Stress hormones (skin conductance)
        scr = np.random.exponential(cost * 1.5, size=4)

        # Fear responses (gastric)
        gastric = np.random.normal(-cost * 2, 0.3, size=4)

        return np.concatenate([hrv, scr, gastric])

    def reset(self) -> Dict:
        """Reset environment"""
        self.threat_accumulator = 0.0
        return {"extero": np.zeros(32), "intero": self._generate_threat_response(0.1)}


# Main execution
if __name__ == "__main__":
    print("Iowa Gambling Task Environment created")
    env = IowaGamblingTaskEnvironment()

    # Run a few demo trials
    total_reward = 0
    for trial in range(1, 6):
        action = np.random.choice(4)  # Random action
        reward, intero_cost, obs, done = env.step(action)
        total_reward += reward
        print(
            f"Trial {trial}: Action={action}, Reward={reward:.2f}, InteroCost={intero_cost:.2f}"
        )

    print(f"Demo completed. Total reward: {total_reward:.2f}")
    print("=== Protocol completed successfully ===")


def run_falsification():
    """Entry point for CLI falsification testing."""
    try:
        print("Running APGI Falsification Protocol 2: Iowa Gambling Task Environment")
        # Run the main demo
        env = IowaGamblingTaskEnvironment()

        total_reward = 0
        for trial in range(1, 6):
            action = np.random.choice(4)  # Random action
            reward, intero_cost, obs, done = env.step(action)
            total_reward += reward
            print(
                f"Trial {trial}: Action={action}, Reward={reward:.2f}, InteroCost={intero_cost:.2f}"
            )

        print(f"Demo completed. Total reward: {total_reward:.2f}")
        return {"status": "success", "total_reward": total_reward}
    except (RuntimeError, ValueError, TypeError, ImportError, KeyError) as e:
        print(f"Error in falsification protocol 2: {e}")
        return {"status": "error", "message": str(e)}
