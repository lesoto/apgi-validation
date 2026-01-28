# Import from other protocols
import importlib.util
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Constants for better maintainability
EXTEROCEPTIVE_DIM = 32
INTEROCEPTIVE_DIM = 16
STATE_DIMENSION = 48
N_ACTIONS = 4
IGNITION_THRESHOLD = 0.5
MIN_SAMPLES_FOR_REGRESSION = 10
MIN_BOOTSTRAP_SAMPLES = 100

# Suppress scipy deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Lazy imports to speed up module loading
def _get_protocol1():
    """Safely import Protocol 1 with error handling"""
    try:
        protocol1_path = os.path.join(
            os.path.dirname(__file__), "Falsification-Protocol-1.py"
        )
        if not os.path.exists(protocol1_path):
            raise ImportError(f"Protocol 1 file not found: {protocol1_path}")

        spec1 = importlib.util.spec_from_file_location("Protocol_1", protocol1_path)
        if spec1 is None or spec1.loader is None:
            raise ImportError(f"Failed to load spec for Protocol 1")

        protocol1 = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(protocol1)
        return protocol1
    except Exception as e:
        raise ImportError(f"Failed to import Protocol 1: {str(e)}")


def _get_protocol2():
    """Safely import Protocol 2 with error handling"""
    try:
        protocol2_path = os.path.join(
            os.path.dirname(__file__), "Falsification-Protocol-2.py"
        )
        if not os.path.exists(protocol2_path):
            raise ImportError(f"Protocol 2 file not found: {protocol2_path}")

        spec2 = importlib.util.spec_from_file_location("Protocol_2", protocol2_path)
        if spec2 is None or spec2.loader is None:
            raise ImportError(f"Failed to load spec for Protocol 2")

        protocol2 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(protocol2)
        return protocol2
    except Exception as e:
        raise ImportError(f"Failed to import Protocol 2: {str(e)}")


def _get_stats():
    """Safely import scipy stats"""
    try:
        from scipy import stats

        return stats
    except ImportError:
        raise ImportError("scipy is required for statistical analysis")


def _get_logistic_regression():
    """Safely import sklearn LogisticRegression"""
    try:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression
    except ImportError:
        raise ImportError("scikit-learn is required for logistic regression")


def _standardize_observation(observation: Dict) -> np.ndarray:
    """Standardize observation to fixed dimensions with proper validation"""
    try:
        if not isinstance(observation, dict):
            raise ValueError("Observation must be a dictionary")

        if "extero" not in observation or "intero" not in observation:
            raise ValueError("Observation must contain 'extero' and 'intero' keys")

        extero_obs = np.asarray(observation["extero"])
        intero_obs = np.asarray(observation["intero"])

        # Standardize exteroceptive to EXTEROCEPTIVE_DIM dimensions
        if extero_obs.size < EXTEROCEPTIVE_DIM:
            extero_standard = np.zeros(EXTEROCEPTIVE_DIM)
            extero_standard[: extero_obs.size] = extero_obs.flatten()
        else:
            extero_standard = extero_obs.flatten()[:EXTEROCEPTIVE_DIM]

        # Standardize interoceptive to INTEROCEPTIVE_DIM dimensions
        if intero_obs.size < INTEROCEPTIVE_DIM:
            intero_standard = np.zeros(INTEROCEPTIVE_DIM)
            intero_standard[: intero_obs.size] = intero_obs.flatten()
        else:
            intero_standard = intero_obs.flatten()[:INTEROCEPTIVE_DIM]

        return np.concatenate([extero_standard, intero_standard])

    except Exception as e:
        # Return zero array as fallback
        print(f"Warning: Observation standardization failed: {str(e)}")
        return np.zeros(STATE_DIMENSION)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax implementation"""
    if logits.size == 0:
        return np.array([0.25, 0.25, 0.25, 0.25])  # Uniform distribution

    logits_shifted = logits - np.max(logits)
    exp_logits = np.exp(logits_shifted)
    sum_exp = np.sum(exp_logits)

    if sum_exp == 0:
        return np.array([0.25, 0.25, 0.25, 0.25])  # Uniform distribution

    return exp_logits / sum_exp


class StandardPPAgent:
    """Standard predictive processing agent without ignition"""

    def __init__(self, config: Dict):
        self.config = config
        # Simple policy network with smaller variance for stability
        self.policy_weights = np.random.normal(
            0, 0.01, (N_ACTIONS, STATE_DIMENSION)
        )  # N_ACTIONS actions, STATE_DIMENSION state dims

    def step(self, observation: Dict, dt: float = 0.05) -> int:
        """Agent step with improved error handling"""
        try:
            state = _standardize_observation(observation)

            # Handle dimension mismatch with policy weights
            if state.shape[0] != self.policy_weights.shape[1]:
                if state.shape[0] < self.policy_weights.shape[1]:
                    state = np.pad(
                        state, (0, self.policy_weights.shape[1] - state.shape[0])
                    )
                else:
                    state = state[: self.policy_weights.shape[1]]

            logits = self.policy_weights @ state
            probs = _softmax(logits)
            return np.random.choice(N_ACTIONS, p=probs)

        except Exception as e:
            print(f"Warning: StandardPPAgent step failed: {str(e)}")
            return np.random.choice(N_ACTIONS)  # Random action as fallback

    def receive_outcome(
        self, reward: float, intero_cost: float, next_observation: Dict
    ) -> None:
        """Receive outcome with type hints"""
        pass  # Simple agent doesn't learn


class GWTOnlyAgent:
    """Global workspace theory agent without somatic markers"""

    def __init__(self, config: Dict):
        self.config = config
        self.policy_weights = np.random.normal(0, 0.01, (N_ACTIONS, STATE_DIMENSION))
        self.conscious_access = False
        self.ignition_history = []

    def step(self, observation: Dict, dt: float = 0.05) -> int:
        """Agent step with ignition detection"""
        try:
            state = _standardize_observation(observation)

            # Handle dimension mismatch with policy weights
            if state.shape[0] != self.policy_weights.shape[1]:
                if state.shape[0] < self.policy_weights.shape[1]:
                    state = np.pad(
                        state, (0, self.policy_weights.shape[1] - state.shape[0])
                    )
                else:
                    state = state[: self.policy_weights.shape[1]]

            # Simple ignition based on exteroceptive surprise
            extero_standard = state[:EXTEROCEPTIVE_DIM]
            surprise = np.linalg.norm(extero_standard)
            self.conscious_access = surprise > IGNITION_THRESHOLD

            if self.conscious_access:
                self.ignition_history.append({"intero_dominant": False})

            logits = self.policy_weights @ state
            probs = _softmax(logits)
            return np.random.choice(N_ACTIONS, p=probs)

        except Exception as e:
            print(f"Warning: GWTOnlyAgent step failed: {str(e)}")
            return np.random.choice(N_ACTIONS)  # Random action as fallback

    def receive_outcome(
        self, reward: float, intero_cost: float, next_observation: Dict
    ) -> None:
        """Receive outcome with type hints"""
        pass


class StandardActorCriticAgent:
    """Standard actor-critic agent"""

    def __init__(self, config: Dict):
        self.config = config
        self.actor_weights = np.random.normal(0, 0.01, (N_ACTIONS, STATE_DIMENSION))
        self.critic_weights = np.random.normal(0, 0.01, (STATE_DIMENSION,))

    def step(self, observation: Dict, dt: float = 0.05) -> int:
        """Agent step with improved error handling"""
        try:
            state = _standardize_observation(observation)

            # Handle dimension mismatch with actor weights
            if state.shape[0] != self.actor_weights.shape[1]:
                if state.shape[0] < self.actor_weights.shape[1]:
                    state = np.pad(
                        state, (0, self.actor_weights.shape[1] - state.shape[0])
                    )
                else:
                    state = state[: self.actor_weights.shape[1]]

            logits = self.actor_weights @ state
            probs = _softmax(logits)
            return np.random.choice(N_ACTIONS, p=probs)

        except Exception as e:
            print(f"Warning: StandardActorCriticAgent step failed: {str(e)}")
            return np.random.choice(N_ACTIONS)  # Random action as fallback

    def receive_outcome(
        self, reward: float, intero_cost: float, next_observation: Dict
    ) -> None:
        """Receive outcome with type hints"""
        pass


class AgentComparisonExperiment:
    """Run complete agent comparison experiment"""

    def __init__(self, n_agents: int = 100, n_trials: int = 200):
        self.n_agents = n_agents
        self.n_trials = n_trials

        # Lazy loading of agent types and environments
        self.agent_types = {
            "APGI": lambda config: _get_protocol1().APGIActiveInferenceAgent(config),
            "StandardPP": StandardPPAgent,
            "GWTOnly": GWTOnlyAgent,
            "ActorCritic": StandardActorCriticAgent,
        }

        self.environments = {
            "IGT": lambda: _get_protocol2().IowaGamblingTaskEnvironment(),
            "Foraging": lambda: _get_protocol2().VolatileForagingEnvironment(),
            "ThreatReward": lambda: _get_protocol2().ThreatRewardTradeoffEnvironment(),
        }

    def run_full_experiment(self) -> Dict[str, Any]:
        """Run all agent types on all environments"""

        results = {}

        for env_name, EnvFunc in self.environments.items():
            results[env_name] = {}

            for agent_name, AgentFunc in self.agent_types.items():
                print(f"Running {agent_name} on {env_name}...")

                agent_results = []

                for agent_idx in range(self.n_agents):
                    # Create fresh agent and environment using lazy loading
                    agent = (
                        AgentFunc(self._get_config())
                        if callable(AgentFunc)
                        else AgentFunc(self._get_config())
                    )
                    env = EnvFunc()  # Environment factory function

                    # Run episode
                    episode_data = self._run_episode(agent, env)
                    agent_results.append(episode_data)

                results[env_name][agent_name] = self._aggregate_results(agent_results)

        return results

    def _run_episode(self, agent, env) -> Dict[str, Any]:
        """Run single episode and collect data"""

        data = {
            "rewards": [],
            "intero_costs": [],
            "cumulative_reward": [],
            "ignitions": [],
            "intero_dominant_ignitions": [],
            "strategy_changes": [],
            "convergence_trial": None,
        }

        observation = env.reset()
        cumulative = 0

        for trial in range(self.n_trials):
            # Agent step
            action = agent.step(observation)

            # Environment step
            reward, intero_cost, next_obs, done = env.step(action)

            # Record data
            data["rewards"].append(reward)
            data["intero_costs"].append(intero_cost)
            cumulative += reward
            data["cumulative_reward"].append(cumulative)

            # Record ignition data (if applicable)
            if hasattr(agent, "conscious_access"):
                data["ignitions"].append(agent.conscious_access)

                if agent.conscious_access and hasattr(agent, "ignition_history"):
                    last_ignition = agent.ignition_history[-1]
                    data["intero_dominant_ignitions"].append(
                        last_ignition["intero_dominant"]
                    )

            # Detect strategy changes
            if trial > 0:
                strategy_change = self._detect_strategy_change(agent, action)
                data["strategy_changes"].append(strategy_change)

            # Check convergence (for IGT: consistent advantageous choices)
            if data["convergence_trial"] is None:
                if self._check_convergence(data, env):
                    data["convergence_trial"] = trial

            # Update agent
            agent.receive_outcome(reward, intero_cost, next_obs)
            observation = next_obs

            if done:
                break

        return data

    def _validate_analysis_input(self, data: any, context: str) -> bool:
        """Validate input data for analysis methods"""
        if data is None:
            print(f"Warning: {context} - None data provided")
            return False

        if isinstance(data, dict):
            if not data:
                print(f"Warning: {context} - Empty dictionary provided")
                return False
        elif isinstance(data, list):
            if not data:
                print(f"Warning: {context} - Empty list provided")
                return False
        elif isinstance(data, np.ndarray):
            if data.size == 0:
                print(f"Warning: {context} - Empty array provided")
                return False
            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                print(f"Warning: {context} - Array contains NaN or Inf values")
                return False

        return True

    def _safe_statistical_test(self, test_func, *args, **kwargs):
        """Safely execute statistical tests with error handling"""
        try:
            return test_func(*args, **kwargs)
        except Exception as e:
            print(f"Statistical test failed: {str(e)}")
            return None, 1.0  # Return null result with non-significant p-value

    def analyze_predictions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze predictions P3a-P3d with proper statistical testing"""

        # Validate input results
        if not self._validate_analysis_input(results, "Main analysis"):
            return {"error": "Invalid or empty results data provided"}

        analysis = {}

        # Statistical power analysis parameters
        alpha = 0.05  # Significance level
        power = 0.8  # Desired power
        effect_size_threshold = 0.5  # Cohen's d threshold for medium effect

        # Import stats for statistical tests
        stats = _get_stats()

        # P3a: Convergence trials with statistical testing
        analysis["P3a"] = {}
        if "IGT" in results and self._validate_analysis_input(
            results["IGT"], "P3a IGT data"
        ):
            analysis["P3a"]["IGT"] = {}
            for agent in results["IGT"].keys():
                if not self._validate_analysis_input(
                    results["IGT"][agent], f"P3a {agent} data"
                ):
                    analysis["P3a"]["IGT"][agent] = None
                    continue

                agent_data = results["IGT"][agent]
                if "convergence_trials" in agent_data:
                    convergence_data = agent_data["convergence_trials"]

                    # Validate convergence data
                    if not self._validate_analysis_input(
                        convergence_data, f"P3a {agent} convergence trials"
                    ):
                        analysis["P3a"]["IGT"][agent] = None
                        continue

                    # Statistical test comparing APGI vs other agents
                    if agent == "APGI":
                        other_agents = [a for a in results["IGT"].keys() if a != "APGI"]
                        other_convergence = []
                        for other in other_agents:
                            if "convergence_trials" in results["IGT"][
                                other
                            ] and self._validate_analysis_input(
                                results["IGT"][other]["convergence_trials"],
                                f"P3a {other} convergence trials",
                            ):
                                other_data = results["IGT"][other]["convergence_trials"]
                                if isinstance(other_data, list):
                                    other_convergence.extend(other_data)

                        if other_convergence and len(other_convergence) > 0:
                            # Perform safe t-test
                            t_stat, p_value = self._safe_statistical_test(
                                stats.ttest_ind, convergence_data, other_convergence
                            )

                            if t_stat is not None:
                                # Calculate effect size (Cohen's d) safely
                                try:
                                    pooled_std = np.sqrt(
                                        (
                                            (len(convergence_data) - 1)
                                            * np.var(convergence_data, ddof=1)
                                            + (len(other_convergence) - 1)
                                            * np.var(other_convergence, ddof=1)
                                        )
                                        / (
                                            len(convergence_data)
                                            + len(other_convergence)
                                            - 2
                                        )
                                    )
                                    if pooled_std > 0:
                                        cohens_d = (
                                            np.mean(convergence_data)
                                            - np.mean(other_convergence)
                                        ) / pooled_std
                                    else:
                                        cohens_d = 0.0
                                except Exception:
                                    cohens_d = 0.0

                                # Power analysis
                                power_calc = self._calculate_power(
                                    abs(cohens_d), alpha, len(convergence_data)
                                )

                                analysis["P3a"]["IGT"][agent] = {
                                    "mean": float(np.mean(convergence_data)),
                                    "std": float(np.std(convergence_data, ddof=1)),
                                    "t_statistic": float(t_stat),
                                    "p_value": float(p_value),
                                    "cohens_d": float(cohens_d),
                                    "power": float(power_calc),
                                    "significant": p_value < alpha,
                                    "effect_size_meaningful": abs(cohens_d)
                                    >= effect_size_threshold,
                                }
                            else:
                                analysis["P3a"]["IGT"][agent] = None
                        else:
                            analysis["P3a"]["IGT"][agent] = None
                    else:
                        analysis["P3a"]["IGT"][agent] = {
                            "mean": (
                                float(np.mean(convergence_data))
                                if convergence_data
                                else None
                            ),
                            "std": (
                                float(np.std(convergence_data, ddof=1))
                                if convergence_data
                                else None
                            ),
                        }
                else:
                    analysis["P3a"]["IGT"][agent] = None
        else:
            analysis["P3a"] = {"error": "IGT data not available or invalid"}

        # P3b: Interoceptive dominance in ignitions with proper testing
        if (
            "IGT" in results
            and "APGI" in results["IGT"]
            and self._validate_analysis_input(results["IGT"]["APGI"], "P3b APGI data")
        ):
            apgi_results = results["IGT"]["APGI"]

            # Handle both raw list and aggregated dict
            if isinstance(apgi_results, list):
                agent_results = apgi_results
            elif isinstance(apgi_results, dict) and "raw_results" in apgi_results:
                agent_results = apgi_results["raw_results"]
            else:
                analysis["P3b"] = {
                    "prediction_met": False,
                    "error": "Individual agent data not available",
                }
                return analysis  # Early return if no individual data

            intero_dominant_data = []

            for agent_result in agent_results:
                if not self._validate_analysis_input(
                    agent_result, "P3b individual agent"
                ):
                    continue
                if "intero_dominant_ignitions" in agent_result:
                    if self._validate_analysis_input(
                        agent_result["intero_dominant_ignitions"],
                        "P3b intero_dominant_ignitions",
                    ):
                        intero_dominant_data.extend(
                            agent_result["intero_dominant_ignitions"]
                        )

            if intero_dominant_data and len(intero_dominant_data) > 0:
                # Test if proportion > 0.5 (null hypothesis: p = 0.5)
                n_successes = sum(intero_dominant_data)
                n_total = len(intero_dominant_data)

                # Safe binomial test using binomtest (newer scipy) or fallback
                try:
                    # Try newer binomtest first
                    from scipy.stats import binomtest

                    binom_result = binomtest(
                        n_successes, n_total, p=0.5, alternative="greater"
                    )
                    binom_p = binom_result.pvalue
                except (ImportError, AttributeError):
                    # Fallback to deprecated binom_test if available
                    try:
                        binom_p = stats.binom_test(
                            n_successes, n_total, p=0.5, alternative="greater"
                        )
                    except (AttributeError, Exception):
                        # Manual binomial test calculation as last resort
                        from math import comb

                        p_value = 0.0
                        for k in range(n_successes, n_total + 1):
                            p_value += comb(n_total, k) * (0.5**n_total)
                        binom_p = p_value

                # Calculate effect size (proportion difference)
                prop_diff = (n_successes / n_total) - 0.5

                # Power analysis for proportion test
                power_prop = self._calculate_proportion_power(prop_diff, alpha, n_total)

                analysis["P3b"] = {
                    "intero_dominant_fraction": float(n_successes / n_total),
                    "n_ignitions": int(n_total),
                    "binomial_p_value": float(binom_p),
                    "proportion_difference": float(prop_diff),
                    "power": float(power_prop),
                    "significant": binom_p < alpha,
                    "prediction_met": binom_p < alpha and prop_diff > 0.1,
                }
            else:
                analysis["P3b"] = {
                    "prediction_met": False,
                    "error": "No valid intero_dominant data",
                }
        else:
            analysis["P3b"] = {
                "prediction_met": False,
                "error": "IGT or APGI data not available",
            }

        # P3c: Ignition predicts strategy change with proper regression analysis
        try:
            analysis["P3c"] = self._logistic_regression_analysis(results)
        except Exception as e:
            analysis["P3c"] = {
                "prediction_met": False,
                "error": f"Logistic regression failed: {str(e)}",
            }

        # P3d: Adaptation speed in volatile foraging with ANOVA
        if "Foraging" in results:
            try:
                analysis["P3d"] = self._analyze_adaptation_speed(
                    results.get("Foraging", {})
                )
            except Exception as e:
                analysis["P3d"] = {
                    "error": f"Adaptation speed analysis failed: {str(e)}"
                }
        else:
            analysis["P3d"] = {"error": "Foraging data not available"}

        # Add cumulative rewards with statistical comparisons
        analysis["cumulative_rewards"] = {}
        for env_name in results.keys():
            if not self._validate_analysis_input(
                results[env_name], f"Cumulative rewards {env_name}"
            ):
                continue

            analysis["cumulative_rewards"][env_name] = {}
            reward_data = {}

            for agent in results[env_name].keys():
                if not self._validate_analysis_input(
                    results[env_name][agent], f"Cumulative rewards {env_name} {agent}"
                ):
                    continue

                agent_data = results[env_name][agent]
                if "cumulative_rewards" in agent_data:
                    rewards = agent_data["cumulative_rewards"]
                    if self._validate_analysis_input(
                        rewards, f"Cumulative rewards list {agent}"
                    ):
                        reward_data[agent] = rewards
                        analysis["cumulative_rewards"][env_name][agent] = {
                            "mean": float(np.mean(rewards)),
                            "std": float(np.std(rewards, ddof=1)),
                            "n": int(len(rewards)),
                        }

            # Perform ANOVA if multiple agents
            if len(reward_data) > 2:
                try:
                    anova_result = self._perform_anova(reward_data)
                    analysis["cumulative_rewards"][env_name]["anova"] = anova_result
                except Exception as e:
                    analysis["cumulative_rewards"][env_name]["anova"] = {
                        "error": f"ANOVA failed: {str(e)}"
                    }

        return analysis

    def _logistic_regression_analysis(self, results: Dict) -> Dict:
        """
        P3c: Test if ignition predicts strategy change beyond |ε|

        Regression: P(strategy_change) ~ ignition + |ε| + controls
        """
        stats = _get_stats()
        LogisticRegression = _get_logistic_regression()

        # Collect data from APGI agents - need to access raw agent results
        X_data = []
        y_data = []

        # Check if we have raw agent results or aggregated results
        if "IGT" in results and "APGI" in results["IGT"]:
            apgi_data = results["IGT"]["APGI"]

            # Handle both raw list of agent results and aggregated dict
            if isinstance(apgi_data, list):
                # Raw agent results
                agent_results = apgi_data
            elif isinstance(apgi_data, dict) and "raw_results" in apgi_data:
                # Aggregated with raw results preserved
                agent_results = apgi_data["raw_results"]
            else:
                # Only aggregated data available - cannot perform individual trial analysis
                return {
                    "ignition_coefficient": None,
                    "ignition_95CI": [None, None],
                    "ignition_p_value": None,
                    "prediction_met": False,
                    "error": "Individual trial data not available for logistic regression",
                }

            # Process individual agent results
            for agent_result in agent_results:
                if not isinstance(agent_result, dict):
                    continue

                # Check required fields exist
                if not all(
                    key in agent_result
                    for key in ["ignitions", "rewards", "strategy_changes"]
                ):
                    continue

                ignitions = agent_result["ignitions"]
                rewards = agent_result["rewards"]
                strategy_changes = agent_result["strategy_changes"]

                # Ensure we have matching lengths
                min_length = min(
                    len(ignitions), len(rewards), len(strategy_changes) + 1
                )

                for t in range(1, min_length):
                    # Skip if any data is invalid
                    if (
                        t >= len(ignitions)
                        or t >= len(rewards)
                        or t - 1 >= len(strategy_changes)
                    ):
                        continue

                    try:
                        X_data.append(
                            [
                                int(
                                    bool(ignitions[t])
                                ),  # Ignition (ensure boolean/integer)
                                float(abs(rewards[t])),  # Prediction error proxy
                                t / len(ignitions),  # Time control
                            ]
                        )
                        y_data.append(int(bool(strategy_changes[t - 1])))
                    except (ValueError, TypeError, IndexError):
                        continue

        if len(X_data) < 10:  # Need minimum data for reliable regression
            return {
                "ignition_coefficient": None,
                "ignition_95CI": [None, None],
                "ignition_p_value": None,
                "prediction_met": False,
                "error": f"Insufficient data for logistic regression: {len(X_data)} samples",
            }

        X = np.array(X_data)
        y = np.array(y_data)

        # Check for valid data
        if np.any(np.isnan(X)) or np.any(np.isnan(y)) or np.any(np.isinf(X)):
            return {
                "ignition_coefficient": None,
                "ignition_95CI": [None, None],
                "ignition_p_value": None,
                "prediction_met": False,
                "error": "Invalid data (NaN/Inf) in regression inputs",
            }

        # Fit model with error handling
        try:
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X, y)
        except Exception as e:
            return {
                "ignition_coefficient": None,
                "ignition_95CI": [None, None],
                "ignition_p_value": None,
                "prediction_met": False,
                "error": f"Logistic regression fitting failed: {str(e)}",
            }

        # Get coefficients and p-values (bootstrap for CI)
        n_bootstrap = min(1000, len(X))  # Limit bootstrap samples
        coef_samples = []

        for _ in range(n_bootstrap):
            try:
                idx = np.random.choice(len(X), len(X), replace=True)
                model_boot = LogisticRegression(random_state=42, max_iter=1000)
                model_boot.fit(X[idx], y[idx])
                coef_samples.append(model_boot.coef_[0])
            except Exception:
                continue  # Skip failed bootstrap samples

        if len(coef_samples) < 100:  # Need sufficient bootstrap samples
            return {
                "ignition_coefficient": float(model.coef_[0][0]),
                "ignition_95CI": [None, None],
                "ignition_p_value": None,
                "prediction_met": False,
                "error": f"Insufficient successful bootstrap samples: {len(coef_samples)}",
            }

        coef_samples = np.array(coef_samples)

        # Ignition coefficient
        ignition_coef = float(model.coef_[0][0])
        ignition_ci = np.percentile(coef_samples[:, 0], [2.5, 97.5])
        ignition_ci = [float(ignition_ci[0]), float(ignition_ci[1])]
        ignition_significant = ignition_ci[0] > 0 or ignition_ci[1] < 0

        # Compute approximate p-value
        if np.std(coef_samples[:, 0]) > 0:
            ignition_z = ignition_coef / np.std(coef_samples[:, 0])
            ignition_p = 2 * (1 - stats.norm.cdf(abs(ignition_z)))
        else:
            ignition_p = 1.0

        return {
            "ignition_coefficient": ignition_coef,
            "ignition_95CI": ignition_ci,
            "ignition_p_value": ignition_p,
            "prediction_met": ignition_p < 0.01 and ignition_significant,
            "n_samples": len(X),
            "bootstrap_success": len(coef_samples),
        }

    def check_falsification(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Check all falsification criteria with proper statistical thresholds"""

        falsified = {}
        alpha = 0.05  # Family-wise error rate
        bonferroni_alpha = alpha / 6  # Adjust for multiple comparisons

        # F3.1: APGI shows no performance advantage (null: APGI <= others)
        if "cumulative_rewards" in analysis and "IGT" in analysis["cumulative_rewards"]:
            igt_rewards = analysis["cumulative_rewards"]["IGT"]
            if "APGI" in igt_rewards and any(k != "APGI" for k in igt_rewards.keys()):
                apgi_mean = igt_rewards["APGI"]["mean"]
                apgi_std = igt_rewards["APGI"]["std"]
                apgi_n = igt_rewards["APGI"]["n"]

                # Compare against best other agent
                other_means = []
                for k, v in igt_rewards.items():
                    if (
                        k != "APGI"
                        and isinstance(v, dict)
                        and "mean" in v
                        and v["mean"] > 0
                    ):
                        other_means.append(v["mean"])

                if other_means:
                    best_other_mean = max(other_means)

                    # Perform t-test
                    if apgi_n > 1:
                        from scipy import stats

                        # Find corresponding other agent data for proper t-test
                        valid_other_agents = [
                            k
                            for k in igt_rewards.keys()
                            if k != "APGI"
                            and isinstance(igt_rewards[k], dict)
                            and "mean" in igt_rewards[k]
                        ]
                        if valid_other_agents:
                            other_agent = max(
                                valid_other_agents, key=lambda k: igt_rewards[k]["mean"]
                            )
                            other_std = igt_rewards[other_agent].get("std", 0.0)
                            other_n = igt_rewards[other_agent].get("n", 1)

                            if other_n > 1:
                                # Pooled standard error
                                pooled_se = np.sqrt(
                                    (apgi_std**2 / apgi_n + other_std**2 / other_n)
                                )
                                if pooled_se > 0:
                                    t_stat = (apgi_mean - best_other_mean) / pooled_se
                                    df = apgi_n + other_n - 2
                                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

                                    # Effect size
                                    pooled_std = np.sqrt(
                                        (
                                            (apgi_n - 1) * apgi_std**2
                                            + (other_n - 1) * other_std**2
                                        )
                                        / (apgi_n + other_n - 2)
                                    )
                                    cohens_d = (
                                        (apgi_mean - best_other_mean) / pooled_std
                                        if pooled_std > 0
                                        else 0
                                    )

                                    # Falsify if no significant advantage AND small effect size
                                    falsified["F3.1"] = (
                                        p_value >= bonferroni_alpha
                                    ) and (abs(cohens_d) < 0.3)
                                else:
                                    falsified["F3.1"] = apgi_mean <= best_other_mean
                            else:
                                falsified["F3.1"] = apgi_mean <= best_other_mean
                        else:
                            falsified["F3.1"] = apgi_mean <= best_other_mean
                    else:
                        falsified["F3.1"] = apgi_mean <= best_other_mean
                else:
                    falsified["F3.1"] = True  # No comparison possible
            else:
                falsified["F3.1"] = True
        else:
            falsified["F3.1"] = True

        # F3.2: Ignition uncorrelated with adaptive behavior
        if "P3c" in analysis and analysis["P3c"]:
            ignition_p = analysis["P3c"].get("ignition_p_value", 1.0)
            ignition_coef = analysis["P3c"].get("ignition_coefficient", 0)

            # Falsify if no significant predictive relationship
            falsified["F3.2"] = (ignition_p >= bonferroni_alpha) or (
                abs(ignition_coef) < 0.1
            )
        else:
            falsified["F3.2"] = True  # No data available

        # F3.3: Pure PP outperforms APGI
        if "cumulative_rewards" in analysis and "IGT" in analysis["cumulative_rewards"]:
            igt_rewards = analysis["cumulative_rewards"]["IGT"]
            if "APGI" in igt_rewards and "StandardPP" in igt_rewards:
                apgi_mean = igt_rewards["APGI"]["mean"]
                pp_mean = igt_rewards["StandardPP"]["mean"]

                # Perform statistical test if possible
                if igt_rewards["APGI"]["n"] > 1 and igt_rewards["StandardPP"]["n"] > 1:
                    from scipy import stats

                    apgi_std = igt_rewards["APGI"]["std"]
                    pp_std = igt_rewards["StandardPP"]["std"]
                    apgi_n = igt_rewards["APGI"]["n"]
                    pp_n = igt_rewards["StandardPP"]["n"]

                    pooled_se = np.sqrt((apgi_std**2 / apgi_n + pp_std**2 / pp_n))
                    if pooled_se > 0:
                        t_stat = (pp_mean - apgi_mean) / pooled_se
                        df = apgi_n + pp_n - 2
                        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

                        # Falsify if PP significantly outperforms APGI
                        falsified["F3.3"] = (pp_mean > apgi_mean) and (
                            p_value < bonferroni_alpha
                        )
                    else:
                        falsified["F3.3"] = pp_mean > apgi_mean
                else:
                    falsified["F3.3"] = pp_mean > apgi_mean
            else:
                falsified["F3.3"] = False
        else:
            falsified["F3.3"] = False

        # Additional falsification criteria based on APGI-specific predictions

        # F3.4: No interoceptive dominance in ignitions
        if "P3b" in analysis and analysis["P3b"]:
            binom_p = analysis["P3b"].get("binomial_p_value", 1.0)
            prop_diff = analysis["P3b"].get("proportion_difference", 0)

            # Falsify if interoceptive signals are not dominant
            falsified["F3.4"] = (binom_p >= bonferroni_alpha) or (prop_diff < 0.1)
        else:
            falsified["F3.4"] = True

        # F3.5: No convergence advantage
        if "P3a" in analysis and "IGT" in analysis["P3a"]:
            if "APGI" in analysis["P3a"]["IGT"] and analysis["P3a"]["IGT"]["APGI"]:
                apgi_conv = analysis["P3a"]["IGT"]["APGI"]
                conv_p = apgi_conv.get("p_value", 1.0)
                conv_effect = apgi_conv.get("cohens_d", 0)

                # Falsify if no convergence advantage
                falsified["F3.5"] = (conv_p >= bonferroni_alpha) or (conv_effect < 0.3)
            else:
                falsified["F3.5"] = True
        else:
            falsified["F3.5"] = True

        # F3.6: No adaptation advantage
        if "P3d" in analysis and analysis["P3d"]:
            if "significant_difference" in analysis["P3d"]:
                # If there are differences but APGI is not best
                adaptation_scores = analysis["P3d"].get("adaptation_scores", {})
                if adaptation_scores and "APGI" in adaptation_scores:
                    apgi_score = adaptation_scores["APGI"]
                    other_scores = [
                        v for k, v in adaptation_scores.items() if k != "APGI"
                    ]
                    if other_scores:
                        best_other = max(other_scores)
                        falsified["F3.6"] = apgi_score <= best_other
                    else:
                        falsified["F3.6"] = False
                else:
                    falsified["F3.6"] = True
            else:
                falsified["F3.6"] = False
        else:
            falsified["F3.6"] = True

        # Overall falsification judgment
        falsified_count = sum(1 for v in falsified.values() if v)
        falsified["overall_falsified"] = falsified_count >= 3  # Majority of criteria
        falsified["falsification_summary"] = f"{falsified_count}/6 criteria met"

        return falsified

    def _get_config(self) -> Dict[str, Any]:
        """Get default configuration for agents"""
        return {
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

    def _detect_strategy_change(self, agent, action: int) -> bool:
        """Detect if agent changed strategy"""
        # Simplified: detect if action differs from previous action
        if not hasattr(agent, "_last_action"):
            agent._last_action = action
            return False

        changed = action != agent._last_action
        agent._last_action = action
        return changed

    def _check_convergence(self, data: Dict[str, Any], env) -> bool:
        """Check if agent has converged to good strategy"""
        # For IGT: check if consistently choosing advantageous decks (C, D)
        if len(data["rewards"]) < 20:
            return False

        # Check last 20 trials for positive rewards
        recent_rewards = data["rewards"][-20:]
        return np.mean(recent_rewards) > 0

    def _aggregate_results(self, agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple agents with data integrity checks"""

        # Validate input
        if not self._validate_analysis_input(
            agent_results, "Agent results aggregation"
        ):
            return {
                "cumulative_rewards": [],
                "total_ignitions": [],
                "convergence_trials": [],
                "strategy_changes": [],
                "error": "Invalid agent results data",
            }

        aggregated = {
            "cumulative_rewards": [],
            "total_ignitions": [],
            "convergence_trials": [],
            "strategy_changes": [],
            "raw_results": [],  # Preserve raw data for individual analysis
        }

        for i, result in enumerate(agent_results):
            if not isinstance(result, dict):
                print(f"Warning: Agent {i} result is not a dictionary, skipping")
                continue

            # Store raw result for individual analysis
            aggregated["raw_results"].append(result)

            # Process cumulative rewards
            if "cumulative_reward" in result and result["cumulative_reward"]:
                if self._validate_analysis_input(
                    result["cumulative_reward"], f"Cumulative reward agent {i}"
                ):
                    final_reward = result["cumulative_reward"][-1]
                    if isinstance(final_reward, (int, float)) and np.isfinite(
                        final_reward
                    ):
                        aggregated["cumulative_rewards"].append(float(final_reward))
                    else:
                        print(
                            f"Warning: Invalid final reward for agent {i}: {final_reward}"
                        )

            # Process ignitions
            if "ignitions" in result and result["ignitions"]:
                if self._validate_analysis_input(
                    result["ignitions"], f"Ignitions agent {i}"
                ):
                    total_ignitions = sum(1 for ign in result["ignitions"] if ign)
                    aggregated["total_ignitions"].append(int(total_ignitions))

            # Process convergence trials
            if (
                "convergence_trial" in result
                and result["convergence_trial"] is not None
            ):
                conv_trial = result["convergence_trial"]
                if (
                    isinstance(conv_trial, (int, float))
                    and np.isfinite(conv_trial)
                    and conv_trial >= 0
                ):
                    aggregated["convergence_trials"].append(float(conv_trial))

            # Process strategy changes
            if "strategy_changes" in result and result["strategy_changes"]:
                if self._validate_analysis_input(
                    result["strategy_changes"], f"Strategy changes agent {i}"
                ):
                    total_changes = sum(
                        1 for change in result["strategy_changes"] if change
                    )
                    aggregated["strategy_changes"].append(int(total_changes))

        # Compute statistics with error handling
        for key in ["cumulative_rewards", "total_ignitions", "strategy_changes"]:
            if aggregated[key]:
                try:
                    mean_val = float(np.mean(aggregated[key]))
                    std_val = float(np.std(aggregated[key], ddof=1))
                    aggregated[f"{key}_mean"] = mean_val
                    aggregated[f"{key}_std"] = std_val
                    aggregated[f"{key}_n"] = len(aggregated[key])
                    aggregated[f"{key}_min"] = float(np.min(aggregated[key]))
                    aggregated[f"{key}_max"] = float(np.max(aggregated[key]))
                except Exception as e:
                    print(f"Warning: Failed to compute statistics for {key}: {str(e)}")
                    aggregated[f"{key}_mean"] = 0.0
                    aggregated[f"{key}_std"] = 0.0
                    aggregated[f"{key}_n"] = 0
            else:
                aggregated[f"{key}_mean"] = 0.0
                aggregated[f"{key}_std"] = 0.0
                aggregated[f"{key}_n"] = 0

        # Special handling for convergence_trials (may have missing values)
        if aggregated["convergence_trials"]:
            try:
                conv_array = np.array(aggregated["convergence_trials"])
                aggregated["convergence_trials_mean"] = float(np.mean(conv_array))
                aggregated["convergence_trials_std"] = float(np.std(conv_array, ddof=1))
                aggregated["convergence_trials_n"] = len(conv_array)
                aggregated["convergence_rate"] = len(conv_array) / len(
                    agent_results
                )  # Proportion that converged

                # Additional convergence statistics
                aggregated["convergence_trials_median"] = float(np.median(conv_array))
                aggregated["convergence_trials_q25"] = float(
                    np.percentile(conv_array, 25)
                )
                aggregated["convergence_trials_q75"] = float(
                    np.percentile(conv_array, 75)
                )
            except Exception as e:
                print(f"Warning: Failed to compute convergence statistics: {str(e)}")
                aggregated["convergence_trials_mean"] = None
                aggregated["convergence_trials_std"] = None
                aggregated["convergence_trials_n"] = 0
                aggregated["convergence_rate"] = 0.0
        else:
            aggregated["convergence_trials_mean"] = None
            aggregated["convergence_trials_std"] = None
            aggregated["convergence_trials_n"] = 0
            aggregated["convergence_rate"] = 0.0

        # Add data quality metrics
        aggregated["data_quality"] = {
            "total_agents": len(agent_results),
            "valid_agents": len(aggregated["raw_results"]),
            "agents_with_rewards": len(aggregated["cumulative_rewards"]),
            "agents_with_convergence": len(aggregated["convergence_trials"]),
            "completion_rate": (
                len(aggregated["raw_results"]) / len(agent_results)
                if agent_results
                else 0.0
            ),
        }

        return aggregated

    def _calculate_power(self, effect_size: float, alpha: float, n: int) -> float:
        """Calculate statistical power for t-test"""
        from scipy import stats

        # Approximate power calculation for two-sample t-test
        df = 2 * n - 2
        t_critical = stats.t.ppf(1 - alpha / 2, df)

        # Non-central t-distribution parameter
        ncp = effect_size * np.sqrt(n / 2)

        # Power calculation
        power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)
        return power

    def _calculate_proportion_power(
        self, prop_diff: float, alpha: float, n: int
    ) -> float:
        """Calculate statistical power for proportion test"""
        from scipy import stats

        # Approximate power calculation for proportion test
        p1 = 0.5 + prop_diff
        p0 = 0.5

        # Z-critical for two-tailed test
        z_critical = stats.norm.ppf(1 - alpha / 2)

        # Standard error
        se = np.sqrt(p1 * (1 - p1) / n + p0 * (1 - p0) / n)

        # Z-statistic
        z_stat = prop_diff / se

        # Power
        power = (
            1
            - stats.norm.cdf(z_critical - z_stat)
            + stats.norm.cdf(-z_critical - z_stat)
        )
        return power

    def _perform_anova(self, reward_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one-way ANOVA on reward data with assumption testing"""
        from scipy import stats

        # Prepare data for ANOVA
        groups = []
        group_names = []

        for agent, rewards in reward_data.items():
            if rewards and len(rewards) > 1:
                # Validate data
                if not self._validate_analysis_input(rewards, f"ANOVA group {agent}"):
                    continue
                groups.append(np.array(rewards))
                group_names.append(agent)

        if len(groups) < 2:
            return {
                "test_performed": False,
                "error": "Need at least 2 valid groups for ANOVA",
            }

        # Test ANOVA assumptions
        assumptions_met = True
        assumption_details = {}

        # Test for normality (Shapiro-Wilk) for each group
        normality_p_values = []
        for i, group in enumerate(groups):
            if len(group) >= 3:  # Shapiro-Wilk requires at least 3 samples
                try:
                    _, p_normal = stats.shapiro(group)
                    normality_p_values.append(p_normal)
                    if p_normal < 0.05:
                        assumptions_met = False
                except Exception:
                    assumptions_met = False

        assumption_details["normality_test"] = {
            "p_values": normality_p_values,
            "assumption_met": (
                all(p >= 0.05 for p in normality_p_values)
                if normality_p_values
                else False
            ),
        }

        # Test for homogeneity of variances (Levene's test)
        if len(groups) >= 2:
            try:
                _, p_levene = stats.levene(*groups)
                assumption_details["levene_test"] = {
                    "p_value": float(p_levene),
                    "assumption_met": p_levene >= 0.05,
                }
                if p_levene < 0.05:
                    assumptions_met = False
            except Exception as e:
                assumption_details["levene_test"] = {"error": str(e)}
                assumptions_met = False

        # Perform ANOVA
        try:
            f_stat, p_value = stats.f_oneway(*groups)

            # Calculate effect size (eta-squared)
            all_data = np.concatenate(groups)
            grand_mean = np.mean(all_data)

            # Between-group sum of squares
            ss_between = sum(
                len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups
            )

            # Total sum of squares
            ss_total = np.sum((all_data - grand_mean) ** 2)

            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            # Post-hoc tests if ANOVA is significant
            post_hoc_results = None
            if p_value < 0.05 and len(groups) > 2:
                post_hoc_results = self._perform_post_hoc_tests(groups, group_names)

            return {
                "test_performed": True,
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "eta_squared": float(eta_squared),
                "groups": group_names,
                "significant": p_value < 0.05,
                "assumptions_met": assumptions_met,
                "assumption_details": assumption_details,
                "post_hoc_tests": post_hoc_results,
                "effect_size_interpretation": self._interpret_eta_squared(eta_squared),
            }
        except (ValueError, RuntimeWarning) as e:
            return {
                "test_performed": False,
                "error": f"ANOVA calculation failed: {str(e)}",
            }

    def _perform_post_hoc_tests(
        self, groups: list, group_names: list
    ) -> Dict[str, Any]:
        """Perform pairwise t-tests with Bonferroni correction"""
        from scipy import stats

        post_hoc_results = {}
        n_comparisons = 0
        significant_pairs = []

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                n_comparisons += 1
                group1_name = group_names[i]
                group2_name = group_names[j]

                # Perform t-test
                try:
                    t_stat, p_value = stats.ttest_ind(groups[i], groups[j])

                    # Bonferroni correction
                    corrected_p = p_value * n_comparisons
                    corrected_p = min(corrected_p, 1.0)  # Cap at 1.0

                    # Calculate effect size
                    pooled_std = np.sqrt(
                        (
                            (len(groups[i]) - 1) * np.var(groups[i], ddof=1)
                            + (len(groups[j]) - 1) * np.var(groups[j], ddof=1)
                        )
                        / (len(groups[i]) + len(groups[j]) - 2)
                    )
                    cohens_d = (
                        (np.mean(groups[i]) - np.mean(groups[j])) / pooled_std
                        if pooled_std > 0
                        else 0
                    )

                    pair_key = f"{group1_name}_vs_{group2_name}"
                    post_hoc_results[pair_key] = {
                        "t_statistic": float(t_stat),
                        "raw_p_value": float(p_value),
                        "corrected_p_value": float(corrected_p),
                        "significant": corrected_p < 0.05,
                        "cohens_d": float(cohens_d),
                        "group1_mean": float(np.mean(groups[i])),
                        "group2_mean": float(np.mean(groups[j])),
                    }

                    if corrected_p < 0.05:
                        significant_pairs.append(pair_key)

                except Exception:
                    continue

        return {
            "pairwise_comparisons": post_hoc_results,
            "n_comparisons": n_comparisons,
            "significant_pairs": significant_pairs,
            "bonferroni_alpha": 0.05 / n_comparisons,
        }

    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size according to Cohen's conventions"""
        if eta_squared < 0.01:
            return "negligible"
        elif eta_squared < 0.06:
            return "small"
        elif eta_squared < 0.14:
            return "medium"
        else:
            return "large"

    def _analyze_adaptation_speed(
        self, foraging_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze adaptation speed in volatile foraging with proper statistics"""
        adaptation_scores = {}

        for agent_name, results in foraging_results.items():
            # Calculate adaptation speed more rigorously
            if "cumulative_rewards" in results and results["cumulative_rewards"]:
                rewards = results["cumulative_rewards"]
                if len(rewards) > 10:
                    # Measure adaptation as slope of reward improvement
                    early_rewards = rewards[: len(rewards) // 2]
                    late_rewards = rewards[len(rewards) // 2 :]

                    early_mean = np.mean(early_rewards)
                    late_mean = np.mean(late_rewards)

                    # Adaptation score = improvement + consistency
                    improvement = late_mean - early_mean
                    consistency = 1 - (
                        np.std(late_rewards) / (np.mean(late_rewards) + 1e-10)
                    )

                    adaptation_scores[agent_name] = improvement * max(0, consistency)
                else:
                    adaptation_scores[agent_name] = 0.0
            else:
                adaptation_scores[agent_name] = 0.0

        # Statistical comparison if multiple agents
        if len(adaptation_scores) > 2:
            scores_list = [
                list(scores) if isinstance(scores, list) else [scores]
                for scores in adaptation_scores.values()
            ]
            try:
                f_stat, p_value = stats.f_oneway(*scores_list)
                return {
                    "adaptation_scores": adaptation_scores,
                    "anova_f": f_stat,
                    "anova_p": p_value,
                    "significant_difference": p_value < 0.05,
                }
            except (ValueError, RuntimeWarning):
                pass

        return {"adaptation_scores": adaptation_scores}


# Main execution
if __name__ == "__main__":
    print("Running Agent Comparison Experiment...")
    experiment = AgentComparisonExperiment(n_agents=10, n_trials=50)
    results = experiment.run_full_experiment()
    analysis = experiment.analyze_predictions(results)
    falsification = experiment.check_falsification(analysis)

    print("=== Protocol completed successfully ===")


def run_falsification():
    """Entry point for CLI falsification testing."""
    try:
        print("Running APGI Falsification Protocol 3: Agent Comparison Experiment")
        experiment = AgentComparisonExperiment(n_agents=10, n_trials=50)
        results = experiment.run_full_experiment()
        analysis = experiment.analyze_predictions(results)
        falsification = experiment.check_falsification(analysis)

        print("=== Protocol completed successfully ===")
        return {
            "status": "success",
            "results": results,
            "analysis": analysis,
            "falsification": falsification,
        }
    except (RuntimeError, ValueError, TypeError, ImportError, KeyError) as e:
        print(f"Error in falsification protocol 3: {e}")
        return {"status": "error", "message": str(e)}
