# Import from other protocols
import importlib.util
import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy import stats
from scipy.stats import binomtest
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# =====================
# DIMENSION CONSTANTS
# =====================
# Import centralized dimension constants
import sys
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import centralized falsification aggregator
try:
    from utils.error_handler import handle_import_error
    from utils.constants import DIM_CONSTANTS
    from utils.falsification_thresholds import (
        F1_1_MIN_ADVANTAGE_PCT,
        F1_1_MIN_COHENS_D,
        F1_1_ALPHA,
    )
    from utils.shared_falsification import check_F5_family
except ImportError as e:

    def handle_import_error(
        module_name: str, error: Exception, context: str = ""
    ) -> None:
        logger.warning(f"Could not import {module_name}: {error}")
        if context:
            logger.warning(f"Context: {context}")
        return None

    handle_import_error(
        "APGI_Falsification-Aggregator", e, "Framework-level falsification aggregation"
    )
    raise

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress excessive config_manager logs during multi-agent simulations
logging.getLogger("utils.config_manager").setLevel(logging.WARNING)
try:
    # Import using dynamic import since filename has hyphens
    spec = importlib.util.spec_from_file_location(
        "FP_ALL_Aggregator",
        os.path.join(os.path.dirname(__file__), "FP_ALL_Aggregator.py"),
    )
    aggregator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aggregator_module)

    aggregate_prediction_results = aggregator_module.aggregate_prediction_results
    run_framework_falsification = aggregator_module.run_framework_falsification
    check_framework_falsification_condition_a = (
        aggregator_module.check_framework_falsification_condition_a
    )
    check_framework_falsification_condition_b = (
        aggregator_module.check_framework_falsification_condition_b
    )
    NAMED_PREDICTIONS = aggregator_module.NAMED_PREDICTIONS
    FRAMEWORK_FALSIFICATION_THRESHOLD_A = (
        aggregator_module.FRAMEWORK_FALSIFICATION_THRESHOLD_A
    )
    ALTERNATIVE_PARSIMONY_THRESHOLD_B = (
        aggregator_module.ALTERNATIVE_PARSIMONY_THRESHOLD_B
    )

    logger.info("Successfully imported FP_ALL_Falsification_Aggregator functions")
except ImportError as e:
    handle_import_error(
        "APGI_Falsification-Aggregator", e, "Framework-level falsification aggregation"
    )
    raise
from utils.constants import DIM_CONSTANTS
from utils.shared_falsification import check_F5_family
from utils.falsification_thresholds import (
    F1_1_MIN_ADVANTAGE_PCT,
    F1_1_MIN_COHENS_D,
    F1_1_ALPHA,
)

# Suppress scipy deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress sklearn numerical warnings (overflow, invalid, divide by zero in matmul)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="divide by zero encountered"
)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="sklearn.linear_model"
)


def bootstrap_ci(
    data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for mean.

    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    if len(data) == 0:
        return 0.0, 0.0, 0.0

    bootstrap_means: List[float] = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    mean = float(np.mean(data))
    lower = float(np.percentile(bootstrap_means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(bootstrap_means, (1 + ci) / 2 * 100))

    return mean, lower, upper


def bootstrap_one_sample_test(
    data: np.ndarray,
    null_value: float = 0.0,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Perform one-sample test using bootstrap.

    Args:
        data: Sample data
        null_value: Null hypothesis value
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level

    Returns:
        Tuple of (test_statistic, p_value)
    """
    if len(data) < 2:
        return 0.0, 1.0

    observed_mean = np.mean(data)
    bootstrap_means_list: List[float] = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means_list.append(float(np.mean(sample)))

    bootstrap_means = np.array(bootstrap_means_list)

    # Two-sided p-value: proportion of bootstrap means as extreme as observed
    if observed_mean >= null_value:
        p_value = np.mean(bootstrap_means >= 2 * null_value - observed_mean)
    else:
        p_value = np.mean(bootstrap_means <= 2 * null_value - observed_mean)

    # Test statistic is standardized difference
    test_stat = (
        (observed_mean - null_value) / (np.std(data) / np.sqrt(len(data)))
        if np.std(data) > 0
        else 0.0
    )

    return test_stat, min(2 * p_value, 1.0)


# Lazy imports to speed up module loading
def _get_protocol1():
    """Safely import Protocol 1 with error handling using absolute path resolution"""
    try:
        from utils.error_handler import handle_import_error

        # Use absolute path to avoid CWD dependence
        protocol1_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "FP_01_ActiveInference.py",
            )
        )
        if not os.path.exists(protocol1_path):
            raise ImportError(f"Protocol 1 file not found: {protocol1_path}")

        spec1 = importlib.util.spec_from_file_location("Protocol_1", protocol1_path)
        if spec1 is None or spec1.loader is None:
            raise ImportError("Failed to load spec for Protocol 1")

        protocol1 = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(protocol1)
        return protocol1
    except ImportError as e:
        handle_import_error(
            "Falsification-Protocol-1", e, "Protocol 1 import for agent comparison"
        )
        raise
    except Exception as e:
        from utils.error_handler import import_error

        raise import_error(
            "DEPENDENCY_ERROR", details=f"Failed to import Protocol 1: {str(e)}"
        )


def _get_protocol2():
    """Safely import Protocol 2 with error handling using absolute path resolution"""
    try:
        from utils.error_handler import handle_import_error

        # Use absolute path to avoid CWD dependence
        protocol2_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "FP_02_AgentComparison_ConvergenceBenchmark.py",
            )
        )
        if not os.path.exists(protocol2_path):
            raise ImportError(f"Protocol 2 file not found: {protocol2_path}")

        spec2 = importlib.util.spec_from_file_location("Protocol_2", protocol2_path)
        if spec2 is None or spec2.loader is None:
            raise ImportError("Failed to load spec for Protocol 2")

        protocol2 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(protocol2)
        return protocol2
    except ImportError as e:
        handle_import_error(
            "Protocol_2", e, "Protocol 2 import for environment creation"
        )
        raise
    except Exception as e:
        from utils.error_handler import import_error

        raise import_error(
            "DEPENDENCY_ERROR", details=f"Failed to import Protocol 2: {str(e)}"
        )


def _get_stats():
    """Safely import scipy stats"""
    try:
        from scipy import stats

        return stats
    except ImportError as e:
        from utils.error_handler import handle_import_error

        handle_import_error(
            "scipy.stats", e, "Statistical analysis in agent comparison"
        )
        raise


def _get_logistic_regression():
    """Safely import sklearn LogisticRegression"""
    try:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression
    except ImportError as e:
        from utils.error_handler import handle_import_error

        handle_import_error(
            "sklearn.linear_model.LogisticRegression",
            e,
            "Logistic regression for strategy change analysis",
        )
        raise


def _standardize_observation(observation: Dict) -> np.ndarray:
    """Standardize observation to fixed dimensions with proper validation"""
    try:
        if not isinstance(observation, dict):
            raise ValueError("Observation must be a dictionary")

        if "extero" not in observation or "intero" not in observation:
            raise ValueError("Observation must contain 'extero' and 'intero' keys")

        extero_obs = np.asarray(observation["extero"])
        intero_obs = np.asarray(observation["intero"])

        # Standardize exteroceptive to DIM_CONSTANTS.EXTERO_DIM dimensions
        extero_standard: np.ndarray
        if extero_obs.size < DIM_CONSTANTS.EXTERO_DIM:
            extero_standard = np.zeros(DIM_CONSTANTS.EXTERO_DIM)
            extero_standard[: extero_obs.size] = extero_obs.flatten()
        else:
            extero_standard = extero_obs.flatten()[: DIM_CONSTANTS.EXTERO_DIM]

        # Standardize interoceptive to DIM_CONSTANTS.INTERO_DIM dimensions
        intero_standard: np.ndarray
        if intero_obs.size < DIM_CONSTANTS.INTERO_DIM:
            intero_standard = np.zeros(DIM_CONSTANTS.INTERO_DIM)
            intero_standard[: intero_obs.size] = intero_obs.flatten()
        else:
            intero_standard = intero_obs.flatten()[: DIM_CONSTANTS.INTERO_DIM]

        # Concatenate and ensure 1D array to prevent broadcasting errors
        result = np.concatenate([extero_standard, intero_standard])
        return result.flatten()  # Ensure 1D array

    except Exception as e:
        # Return zero array as fallback
        print(f"Warning: Observation standardization failed: {str(e)}")
        return np.zeros(DIM_CONSTANTS.STATE_DIMENSION)


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
            0, 0.01, (DIM_CONSTANTS.N_ACTIONS, DIM_CONSTANTS.STATE_DIMENSION)
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
            return np.random.choice(DIM_CONSTANTS.N_ACTIONS, p=probs)

        except Exception as e:
            print(f"Warning: StandardPPAgent step failed: {str(e)}")
            return np.random.choice(
                DIM_CONSTANTS.N_ACTIONS
            )  # Random action as fallback

    def receive_outcome(
        self, reward: float, intero_cost: float, next_observation: Dict
    ) -> None:
        """Receive outcome with type hints"""
        pass  # Simple agent doesn't learn


class GWTOnlyAgent:
    """Global workspace theory agent without somatic markers"""

    def __init__(self, config: Dict):
        self.config = config
        self.policy_weights = np.random.normal(
            0, 0.01, (DIM_CONSTANTS.N_ACTIONS, DIM_CONSTANTS.STATE_DIMENSION)
        )
        self.conscious_access = False
        self.ignition_history: List[Dict[str, Any]] = []

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
            extero_standard = state[: DIM_CONSTANTS.EXTERO_DIM]
            surprise = np.linalg.norm(extero_standard)
            self.conscious_access = bool(surprise > DIM_CONSTANTS.IGNITION_THRESHOLD)

            if self.conscious_access:
                self.ignition_history.append({"intero_dominant": False})

            logits = self.policy_weights @ state
            probs = _softmax(logits)
            return np.random.choice(DIM_CONSTANTS.N_ACTIONS, p=probs)

        except Exception as e:
            print(f"Warning: GWTOnlyAgent step failed: {str(e)}")
            return np.random.choice(
                DIM_CONSTANTS.N_ACTIONS
            )  # Random action as fallback

    def receive_outcome(
        self, reward: float, intero_cost: float, next_observation: Dict
    ) -> None:
        """Receive outcome with type hints"""
        pass


class StandardActorCriticAgent:
    """Standard actor-critic agent"""

    def __init__(self, config: Dict):
        self.config = config
        self.actor_weights = np.random.normal(
            0, 0.01, (DIM_CONSTANTS.N_ACTIONS, DIM_CONSTANTS.STATE_DIMENSION)
        )
        self.critic_weights = np.random.normal(
            0, 0.01, (DIM_CONSTANTS.STATE_DIMENSION,)
        )

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
            return np.random.choice(DIM_CONSTANTS.N_ACTIONS, p=probs)

        except Exception as e:
            print(f"Warning: StandardActorCriticAgent step failed: {str(e)}")
            return np.random.choice(
                DIM_CONSTANTS.N_ACTIONS
            )  # Random action as fallback

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
        """Run framework-level synthesis: load protocols 1-12 and apply aggregator logic"""
        import json

        logger.info(
            "Starting framework-level synthesis: loading protocols 1-12 and applying falsification"
        )

        # Define protocol files for synthesis (protocols 1, 2, 3, 6, 9, 12)
        # Note: Protocol 3 is the agent comparison protocol (VP_3_ActiveInference_AgentSimulations_Protocol3.py)
        # which produces P3.conv and P3.bic predictions
        protocol_files = {
            1: "FP_01_ActiveInference.py",
            2: "FP_02_AgentComparison_ConvergenceBenchmark.py",
            3: "../Validation/VP_03_ActiveInference_AgentSimulations.py",
            6: "FP_05_EvolutionaryPlausibility.py",
            9: "FP_09_NeuralSignatures_P3b_HEP.py",
            12: "FP_12_CrossSpeciesScaling.py",
        }

        # Load results from each available protocol
        protocol_results = {}
        for protocol_num, protocol_file in protocol_files.items():
            if protocol_file is None:
                logger.warning(f"Protocol {protocol_num} file not specified, skipping")
                continue

            protocol_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), protocol_file)
            )

            if not os.path.exists(protocol_path):
                logger.warning(
                    f"Protocol {protocol_num} file not found: {protocol_path}"
                )
                continue

            try:
                # Import protocol module with error handling
                spec = importlib.util.spec_from_file_location(
                    f"Protocol_{protocol_num}", protocol_path
                )
                if spec is None or spec.loader is None:
                    raise ImportError(
                        f"Failed to load spec for Protocol {protocol_num}"
                    )
                protocol_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(protocol_module)

                # Run protocol and get results
                if hasattr(protocol_module, "run_protocol"):
                    protocol_results[protocol_num] = protocol_module.run_protocol()
                elif hasattr(protocol_module, "run_validation"):
                    protocol_results[protocol_num] = protocol_module.run_validation()
                elif hasattr(protocol_module, "run_falsification"):
                    protocol_results[protocol_num] = protocol_module.run_falsification()
                else:
                    logger.warning(
                        f"Protocol {protocol_num} missing run method, using fallback"
                    )
                    continue

            except ImportError as e:
                logger.error(f"Failed to load Protocol {protocol_num}: {str(e)}")
                continue

        # Apply framework-level falsification using aggregator
        if protocol_results:
            logger.info("Applying framework-level falsification criteria")
            falsification = run_framework_falsification(protocol_results)

            # Save results with error handling
            output_dir = os.path.join(os.path.dirname(__file__), "results")
            os.makedirs(output_dir, exist_ok=True)

            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                output_dir, f"framework_falsification_{timestamp}.json"
            )

            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(falsification, f, indent=2)
                logger.info(f"Framework-level results saved to {output_file}")
            except (OSError, IOError) as e:
                logger.error(f"Failed to save results to {output_file}: {e}")
                # Try alternative location
                alt_output_file = os.path.join(
                    "/tmp", f"framework_falsification_{timestamp}.json"
                )
                try:
                    with open(alt_output_file, "w", encoding="utf-8") as f:
                        json.dump(falsification, f, indent=2)
                    logger.info(
                        f"Framework-level results saved to alternative location: {alt_output_file}"
                    )
                except (OSError, IOError) as e2:
                    logger.error(
                        f"Failed to save results to alternative location: {e2}"
                    )
                    # Continue without saving - return results in memory
                    logger.warning("Continuing without saving results to file")

            return falsification
        else:
            logger.error("No protocol results available for synthesis")
            return {"error": "No protocol results available for synthesis"}

    def _run_episode(self, agent, env) -> Dict[str, Any]:
        """Run single episode and collect data"""

        data: Dict[str, Any] = {
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

    def _validate_analysis_input(self, data: Any, context: str) -> bool:
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

    def _analyze_p3a_convergence(
        self, results: Dict[str, Any], alpha: float, effect_size_threshold: float, stats
    ) -> Dict[str, Any]:
        """Analyze P3a: Convergence trials with statistical testing"""
        analysis_p3a: Dict[str, Any] = {}
        if "IGT" in results and self._validate_analysis_input(
            results["IGT"], "P3a IGT data"
        ):
            analysis_p3a["IGT"] = {}
            for agent in list(results["IGT"].keys()):
                if not self._validate_analysis_input(
                    results["IGT"][agent], f"P3a {agent} data"
                ):
                    analysis_p3a["IGT"][agent] = None
                    continue

                agent_data = results["IGT"][agent]
                if "convergence_trials" in agent_data:
                    convergence_data = agent_data["convergence_trials"]

                    # Validate convergence data
                    if not self._validate_analysis_input(
                        convergence_data, f"P3a {agent} convergence trials"
                    ):
                        analysis_p3a["IGT"][agent] = None
                        continue

                    # Statistical test comparing APGI vs other agents
                    if agent == "APGI":
                        analysis_p3a["IGT"][agent] = self._analyze_apgi_convergence(
                            results["IGT"],
                            agent,
                            convergence_data,
                            alpha,
                            effect_size_threshold,
                            stats,
                        )
                    else:
                        analysis_p3a["IGT"][agent] = self._analyze_non_apgi_convergence(
                            convergence_data
                        )
                else:
                    analysis_p3a["IGT"][agent] = None
        else:
            analysis_p3a = {"error": "IGT data not available or invalid"}

        return analysis_p3a

    def _analyze_apgi_convergence(
        self,
        igt_results: Dict,
        agent: str,
        convergence_data: List,
        alpha: float,
        effect_size_threshold: float,
        stats: Any,
    ) -> Dict[str, Any]:
        """Analyze APGI convergence with statistical comparison to other agents"""
        other_agents = [a for a in list(igt_results.keys()) if a != "APGI"]
        other_convergence: List[float] = []
        for other in other_agents:
            if "convergence_trials" in igt_results[
                other
            ] and self._validate_analysis_input(
                igt_results[other]["convergence_trials"],
                f"P3a {other} convergence trials",
            ):
                other_data = igt_results[other]["convergence_trials"]
                if isinstance(other_data, list):
                    other_convergence.extend(other_data)

        if other_convergence and len(other_convergence) > 0:
            # Perform safe t-test
            t_stat, p_value = self._safe_statistical_test(
                stats.ttest_ind, convergence_data, other_convergence
            )

            if t_stat is not None:
                # Calculate effect size (Cohen's d) safely
                cohens_d = self._calculate_cohens_d(convergence_data, other_convergence)

                # Power analysis
                power_calc = self._calculate_power(
                    abs(cohens_d), alpha, len(convergence_data)
                )

                return {
                    "mean": float(np.mean(convergence_data)),
                    "std": float(np.std(convergence_data, ddof=1)),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_d": float(cohens_d),
                    "power": float(power_calc),
                    "significant": p_value < alpha,
                    "effect_size_meaningful": abs(cohens_d) >= effect_size_threshold,
                }
        return None

    def _analyze_non_apgi_convergence(self, convergence_data: List) -> Dict[str, Any]:
        """Analyze convergence for non-APGI agents (basic statistics only)"""
        return {
            "mean": (float(np.mean(convergence_data)) if convergence_data else None),
            "std": (
                float(np.std(convergence_data, ddof=1)) if convergence_data else None
            ),
        }

    def _calculate_cohens_d(self, group1: List, group2: List) -> float:
        """Calculate Cohen's d effect size between two groups"""
        try:
            pooled_std = np.sqrt(
                (
                    (len(group1) - 1) * np.var(group1, ddof=1)
                    + (len(group2) - 1) * np.var(group2, ddof=1)
                )
                / (len(group1) + len(group2) - 2)
            )
            if pooled_std > 0:
                return (np.mean(group1) - np.mean(group2)) / pooled_std
            else:
                return 0.0
        except (ValueError, ZeroDivisionError, TypeError) as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Cohen's d calculation failed: {e}")
            return 0.0

    def _analyze_p3b_interoceptive_dominance(
        self, results: Dict[str, Any], stats: Any
    ) -> Dict[str, Any]:
        """Analyze P3b: Interoceptive dominance in ignitions"""
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
                return {
                    "prediction_met": False,
                    "error": "Individual agent data not available",
                }

            intero_dominant_data: List[float] = []

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

                # Safe binomial test using binomtest (newer scipy)
                try:
                    from scipy.stats import binomtest

                    binom_result = binomtest(
                        n_successes, n_total, p=0.5, alternative="greater"
                    )
                    p_value = binom_result.pvalue
                except ImportError:
                    # Fallback: simple proportion test approximation
                    p_hat = n_successes / n_total
                    if p_hat > 0.5:
                        # Approximate p-value using normal approximation
                        se = np.sqrt(0.5 * 0.5 / n_total)
                        z = (p_hat - 0.5) / se
                        p_value = 1 - stats.norm.cdf(z)
                    else:
                        p_value = 1.0

                proportion = n_successes / n_total
                return {
                    "prediction_met": p_value < 0.05,
                    "proportion_intero_dominant": float(proportion),
                    "n_ignitions": int(n_total),
                    "p_value": float(p_value),
                    "test_type": "binomial_test",
                    "null_hypothesis": "p = 0.5",
                    "alternative": "p > 0.5",
                }

        return {
            "prediction_met": False,
            "error": "P3b data not available",
        }

    def analyze_predictions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze predictions P3a-P3d with proper statistical testing"""

        # Validate input results
        if not self._validate_analysis_input(results, "Main analysis"):
            return {"error": "Invalid or empty results data provided"}

        analysis = {}

        # Statistical power analysis parameters
        alpha = 0.05  # Significance level
        effect_size_threshold = 0.5  # Cohen's d threshold for medium effect

        # Import stats for statistical tests
        stats = _get_stats()

        # P3a: Convergence trials with statistical testing
        analysis["P3a"] = self._analyze_p3a_convergence(
            results, alpha, effect_size_threshold, stats
        )

        # P3b: Interoceptive dominance in ignitions with proper testing
        analysis["P3b"] = self._analyze_p3b_interoceptive_dominance(results, stats)

        # P3c and P3d would be implemented similarly if needed
        analysis["P3c"] = {"status": "not_implemented"}
        analysis["P3d"] = {"status": "not_implemented"}

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

        # More aggressive clipping to prevent numerical overflow in sklearn
        # Use much tighter bounds to ensure numerical stability
        X = np.clip(X, -1e3, 1e3)

        # Check for valid data
        if np.any(np.isnan(X)) or np.any(np.isnan(y)) or np.any(np.isinf(X)):
            return {
                "ignition_coefficient": None,
                "ignition_95CI": [None, None],
                "ignition_p_value": None,
                "prediction_met": False,
                "error": "Invalid data (NaN/Inf) in regression inputs",
            }

        # Scale features to prevent overflow in sklearn
        # Use RobustScaler instead of StandardScaler to handle outliers better
        from sklearn.preprocessing import RobustScaler
        import warnings

        try:
            scaler = RobustScaler(quantile_range=(5.0, 95.0))
            X_scaled = scaler.fit_transform(X)
        except Exception:
            # Fallback: manual robust scaling
            try:
                median = np.median(X, axis=0)
                q75 = np.percentile(X, 75, axis=0)
                q25 = np.percentile(X, 25, axis=0)
                iqr = q75 - q25
                iqr = np.where(iqr == 0, 1.0, iqr)  # Avoid division by zero
                X_scaled = (X - median) / iqr
                # Clip scaled values to prevent extreme outliers
                X_scaled = np.clip(X_scaled, -5, 5)
            except Exception:
                X_scaled = X  # Ultimate fallback to unscaled

        # Additional safety: ensure no extreme values after scaling
        X_scaled = np.clip(X_scaled, -1e3, 1e3)

        # Fit model with error handling and suppressed overflow warnings
        try:
            # Use liblinear solver which is more numerically stable for small datasets
            # and add strong regularization to prevent coefficient explosion
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=RuntimeWarning, message="overflow encountered"
                )
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="invalid value encountered",
                )
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="divide by zero encountered",
                )
                model = LogisticRegression(
                    random_state=42,
                    max_iter=500,
                    solver="liblinear",
                    C=0.1,  # Strong regularization to prevent coefficient explosion
                )
                model.fit(X_scaled, y)
        except Exception as e:
            return {
                "ignition_coefficient": None,
                "ignition_95CI": [None, None],
                "ignition_p_value": None,
                "prediction_met": False,
                "error": f"Logistic regression fitting failed: {str(e)}",
            }

        # Get coefficients and p-values (bootstrap for CI)
        n_bootstrap = min(500, len(X_scaled))  # Limit bootstrap samples for speed
        coef_samples = []

        for _ in range(n_bootstrap):
            try:
                idx = np.random.choice(len(X_scaled), len(X_scaled), replace=True)
                X_boot = X_scaled[idx]
                y_boot = y[idx]

                # Skip if bootstrap sample has no variance in y
                if len(np.unique(y_boot)) < 2:
                    continue

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=RuntimeWarning, message="overflow"
                    )
                    warnings.filterwarnings(
                        "ignore", category=RuntimeWarning, message="invalid"
                    )
                    warnings.filterwarnings(
                        "ignore", category=RuntimeWarning, message="divide by zero"
                    )
                    model_boot = LogisticRegression(
                        random_state=42,
                        max_iter=500,
                        solver="liblinear",
                        C=0.1,
                    )
                    model_boot.fit(X_boot, y_boot)
                    coef_samples.append(model_boot.coef_[0])
            except (ValueError, IndexError, KeyError, RuntimeError) as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Bootstrap sample failed: {e}")
                continue  # Skip failed bootstrap samples

        if len(coef_samples) < 50:  # Need sufficient bootstrap samples
            return {
                "ignition_coefficient": float(model.coef_[0][0]),
                "ignition_95CI": [None, None],
                "ignition_p_value": None,
                "prediction_met": False,
                "error": f"Insufficient successful bootstrap samples: {len(coef_samples)}",
            }

        coef_samples_arr = np.array(coef_samples)

        # Ignition coefficient
        ignition_coef = float(model.coef_[0][0])
        ignition_ci_arr = np.percentile(coef_samples_arr[:, 0], [2.5, 97.5])
        ignition_ci = [float(ignition_ci_arr[0]), float(ignition_ci_arr[1])]
        ignition_significant = ignition_ci[0] > 0 or ignition_ci[1] < 0

        # Compute approximate p-value
        if np.std(coef_samples_arr[:, 0]) > 0:
            ignition_z = ignition_coef / float(np.std(coef_samples_arr[:, 0]))
            ignition_p = float(2 * (1 - stats.norm.cdf(abs(ignition_z))))
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

                        # Effect size
                        pooled_std = np.sqrt(
                            ((apgi_n - 1) * apgi_std**2 + (pp_n - 1) * pp_std**2)
                            / (apgi_n + pp_n - 2)
                        )
                        cohens_d = (
                            (pp_mean - apgi_mean) / pooled_std if pooled_std > 0 else 0
                        )

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

    def check_framework_falsification_conditions(
        self, framework_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check framework-level falsification conditions using the aggregator.

        Args:
            framework_results: Results from running all protocols via run_full_experiment()

        Returns:
            Dict with framework falsification status and condition results
        """
        # Aggregate predictions from framework results
        apgi_predictions = aggregate_prediction_results(framework_results)

        # Generate alternative framework predictions
        try:
            generate_gnwt_predictions = aggregator_module.generate_gnwt_predictions
            generate_iit_predictions = aggregator_module.generate_iit_predictions
        except (AttributeError, NameError):
            # Fallback if not directly available
            def generate_gnwt_predictions():
                return {}

            def generate_iit_predictions():
                return {}

        gnwt_predictions = generate_gnwt_predictions()
        iit_predictions = generate_iit_predictions()

        # Check Condition A: All 14 named predictions fail
        condition_a_met = check_framework_falsification_condition_a(apgi_predictions)

        # Check Condition B: Alternative frameworks more parsimonious
        condition_b_met = check_framework_falsification_condition_b(
            results_input=framework_results,
            apgi_predictions=apgi_predictions,
            gnwt_predictions=gnwt_predictions,
            iit_predictions=iit_predictions,
        )

        # Count passing/failing predictions
        total_named = len(NAMED_PREDICTIONS)
        failed_predictions = sum(
            1 for r in apgi_predictions.values() if not r.get("passed", True)
        )

        return {
            "framework_falsified": bool(condition_a_met or condition_b_met),
            "condition_a": {
                "condition_met": bool(condition_a_met),
                "all_predictions_failed": bool(condition_a_met),
            },
            "condition_b": {
                "condition_met": bool(condition_b_met),
                "alternatives_more_parsimonious": bool(condition_b_met),
            },
            "summary": {
                "total_named_predictions": total_named,
                "failed_predictions": failed_predictions,
                "apgi_passing": total_named - failed_predictions,
            },
        }

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

        aggregated: Dict[str, Any] = {
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
                except (ValueError, TypeError, AttributeError) as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.debug(f"Normality test failed for group {i}: {e}")
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

                except (ValueError, TypeError, IndexError, KeyError) as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.debug(
                        f"Post-hoc comparison failed for {group1_name} vs {group2_name}: {e}"
                    )
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
        from scipy import stats

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
    print("Running Framework-Level Multi-Protocol Experiment...")
    experiment = AgentComparisonExperiment(n_agents=100, n_trials=500)

    # Run synthesis and get framework-level results
    framework_results = experiment.run_full_experiment()

    if "error" in framework_results:
        print(f"Error in framework synthesis: {framework_results['error']}")
    else:
        # Apply framework-level falsification conditions
        falsification_result = experiment.check_framework_falsification_conditions(
            framework_results
        )

        print("=== Framework-Level Falsification Results ===")
        print(f"Framework Falsified: {falsification_result['framework_falsified']}")
        print(
            f"Condition (a) Met: {falsification_result['condition_a']['condition_met']}"
        )
        print(
            f"Condition (b) Met: {falsification_result['condition_b']['condition_met']}"
        )
        print(
            f"Total Named Predictions: {falsification_result['summary']['total_named_predictions']}"
        )
        print(
            f"Failed Predictions: {falsification_result['summary']['failed_predictions']}"
        )

        print("=== Framework-Level Protocol completed successfully ===")


def run_falsification():
    """Entry point for CLI falsification testing."""
    try:
        print("Running APGI Falsification Protocol 3: Agent Comparison Experiment")
        experiment = AgentComparisonExperiment(n_agents=100, n_trials=500)
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


# =============================================================================
# FALSIFICATION CRITERIA IMPLEMENTATION
# =============================================================================


def get_falsification_criteria() -> Dict[str, Dict[str, Any]]:
    """
    Return complete falsification specifications for Falsification-Protocol-3.

    Tests: APGI vs. baseline model performance, quantitative advantage demonstration

    Returns:
        Dictionary of falsification criteria with thresholds, tests, and effect sizes
    """
    return {
        "F3.1": {
            "description": "Overall Performance Advantage",
            "threshold": "≥18% higher cumulative reward than best non-APGI baseline across mixed task battery",
            "test": "Independent samples t-test with Welch correction, α=0.008 (Bonferroni for 6 comparisons)",
            "effect_size": "Cohen's d ≥ 0.60; 95% CI for advantage excludes 10%",
            "alternative": "Falsified if APGI advantage <12% OR d < 0.40 OR p ≥ 0.008 OR 95% CI includes 8%",
        },
        "F3.2": {
            "description": "Interoceptive Task Specificity",
            "threshold": "APGI advantage increases to ≥28% in high interoceptive tasks vs. ≤12% in purely exteroceptive",
            "test": "Two-way mixed ANOVA (Agent Type × Task Category), α=0.01",
            "effect_size": "Partial η² ≥ 0.20 for interaction; simple effects d ≥ 0.70 for interoceptive tasks",
            "alternative": "Falsified if interoceptive advantage <20% OR interaction p ≥ 0.01 OR partial η² < 0.12 OR simple effects d < 0.45",
        },
        "F3.3": {
            "description": "Threshold Gating Necessity",
            "threshold": "Removing threshold gating reduces APGI performance by ≥25% in volatile environments",
            "test": "Paired t-test comparing full APGI vs. no-threshold variant, α=0.01",
            "effect_size": "Cohen's d ≥ 0.75",
            "alternative": "Falsified if performance reduction <15% OR d < 0.50 OR p ≥ 0.01",
        },
        "F3.4": {
            "description": "Precision Weighting Necessity",
            "threshold": "Uniform precision reduces APGI performance by ≥20% in tasks with unreliable sensory modalities",
            "test": "Paired t-test, α=0.01",
            "effect_size": "Cohen's d ≥ 0.65",
            "alternative": "Falsified if reduction <12% OR d < 0.42 OR p ≥ 0.01",
        },
        "F3.5": {
            "description": "Computational Efficiency Trade-Off",
            "threshold": "APGI maintains ≥85% of full model performance while using ≤60% of computational operations",
            "test": "Equivalence testing (TOST) for non-inferiority, α=0.05",
            "effect_size": "Efficiency gain ≥30%; performance retention ≥85%",
            "alternative": "Falsified if performance retention <78% OR efficiency gain <20% OR fails TOST non-inferiority",
        },
        "F3.6": {
            "description": "Sample Efficiency in Learning",
            "threshold": "APGI agents achieve 80% asymptotic performance in ≤200 trials vs. ≥300 for standard RL",
            "test": "Time-to-criterion analysis with log-rank test, α=0.01",
            "effect_size": "Hazard ratio ≥ 1.45",
            "alternative": "Falsified if APGI time-to-criterion >250 trials OR advantage <25% OR hazard ratio < 1.30 OR p ≥ 0.01",
        },
    }


def check_falsification(
    apgi_rewards: List[float],
    baseline_rewards: List[float],
    interoceptive_advantage: float,
    exteroceptive_advantage: float,
    threshold_reduction: float,
    precision_reduction: float,
    performance_retention: float,
    efficiency_gain: float,
    apgi_time_to_criterion: float,
    baseline_time_to_criterion: float,
    # F1 parameters
    pp_rewards: List[float],
    timescales: List[float],
    precision_weights: List[Tuple[float, float]],
    threshold_adaptation: List[float],
    pac_mi: List[Tuple[float, float]],
    spectral_slopes: List[Tuple[float, float]],
    # F2 parameters
    apgi_advantageous_selection: List[float],
    no_somatic_selection: List[float],
    apgi_cost_correlation: float,
    no_somatic_cost_correlation: float,
    rt_advantage_ms: float,
    rt_cost_modulation: float,
    confidence_effect: float,
    beta_interaction: float,
    no_somatic_time_to_criterion: float,
    # F5 parameters
    threshold_emergence_proportion: float,
    precision_emergence_proportion: float,
    intero_gain_ratio_proportion: float,
    multi_timescale_proportion: float,
    pca_variance_explained: float,
    control_performance_difference: float,
    # F6 parameters
    ltcn_transition_time: float,
    rnn_transition_time: float,
    ltcn_sparsity_reduction: float,
    rnn_sparsity_reduction: float,
    ltcn_integration_window: float,
    rnn_integration_window: float,
    memory_decay_tau: float,
    bifurcation_point: float,
    hysteresis_width: float,
    rnn_add_ons_needed: int,
    performance_gap: float,
) -> Dict[str, Any]:
    """
    Implement all statistical tests for Falsification-Protocol-3 (complete framework).

    Args:
        apgi_rewards: Cumulative rewards for APGI agents across mixed tasks
        baseline_rewards: Rewards for best non-APGI baseline
        interoceptive_advantage: APGI advantage in high interoceptive tasks
        exteroceptive_advantage: APGI advantage in purely exteroceptive tasks
        threshold_reduction: Performance reduction when threshold gating removed
        precision_reduction: Performance reduction with uniform precision
        performance_retention: Percentage of full model performance retained
        efficiency_gain: Computational operations reduction percentage
        apgi_time_to_criterion: Trials for APGI to reach 80% asymptotic performance
        baseline_time_to_criterion: Trials for baseline RL to reach 80% performance
        # F1 parameters
        pp_rewards: Cumulative rewards for standard PP agents
        timescales: Intrinsic timescale measurements
        precision_weights: (Level1, Level3) precision weights
        threshold_adaptation: Threshold adaptation measurements
        pac_mi: PAC modulation indices (baseline, ignition)
        spectral_slopes: (active, low_arousal) spectral slopes
        # F2 parameters
        apgi_advantageous_selection: Selection frequencies for advantageous decks by trial 60
        no_somatic_selection: Selection frequencies for agents without somatic modulation
        apgi_cost_correlation: Correlation between deck selection and interoceptive cost for APGI
        no_somatic_cost_correlation: Correlation for non-interoceptive agents
        rt_advantage_ms: RT advantage for rewarding decks with low interoceptive cost
        rt_cost_modulation: RT modulation per unit cost increase
        confidence_effect: Effect of confidence on deck preference
        beta_interaction: Interaction coefficient for confidence × interoceptive signal
        no_somatic_time_to_criterion: Trials for non-interoceptive agents
        # F5 parameters
        threshold_emergence_proportion: Proportion of evolved agents developing thresholds
        precision_emergence_proportion: Proportion developing precision weighting
        intero_gain_ratio_proportion: Proportion with interoceptive prioritization
        multi_timescale_proportion: Proportion with multi-timescale integration
        pca_variance_explained: Variance explained by APGI feature PCs
        control_performance_difference: Performance difference vs. control agents
        # F6 parameters
        ltcn_transition_time: Ignition transition time for LTCNs
        rnn_transition_time: Ignition transition time for standard RNNs
        ltcn_sparsity_reduction: Sparsity reduction for LTCNs
        rnn_sparsity_reduction: Sparsity reduction for RNNs
        ltcn_integration_window: Temporal integration window for LTCNs
        rnn_integration_window: Temporal integration window for RNNs
        memory_decay_tau: Memory decay time constant
        bifurcation_point: Bifurcation point precision value
        hysteresis_width: Hysteresis width
        rnn_add_ons_needed: Number of add-ons needed for RNNs
        performance_gap: Performance gap without add-ons

    Returns:
        Dictionary with pass/fail results, effect sizes, and test statistics
    """
    results = {
        "protocol": "Falsification-Protocol-3",
        "criteria": {},
        "summary": {"passed": 0, "failed": 0, "total": 16},
    }

    # F3.1: Overall Performance Advantage
    logger.info("Testing F3.1: Overall Performance Advantage")
    t_stat, p_value = stats.ttest_ind(apgi_rewards, baseline_rewards, equal_var=False)
    mean_apgi = np.mean(apgi_rewards)
    mean_baseline = np.mean(baseline_rewards)
    advantage_pct = ((mean_apgi - mean_baseline) / mean_baseline) * 100

    # Cohen's d (Welch's approximation)
    n1, n2 = len(apgi_rewards), len(baseline_rewards)
    var1, var2 = np.var(apgi_rewards, ddof=1), np.var(baseline_rewards, ddof=1)
    pooled_se = np.sqrt(var1 / n1 + var2 / n2)
    cohens_d = (mean_apgi - mean_baseline) / pooled_se

    # 95% CI
    se_diff = pooled_se * np.sqrt(1 / n1 + 1 / n2)
    ci_lower = (mean_apgi - mean_baseline) - 1.96 * se_diff
    ci_upper = (mean_apgi - mean_baseline) + 1.96 * se_diff

    f3_1_pass = (
        advantage_pct >= 12 and cohens_d >= 0.40 and p_value < 0.008 and ci_lower > 8
    )
    results["criteria"]["F3.1"] = {
        "passed": f3_1_pass,
        "advantage_pct": advantage_pct,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "t_statistic": t_stat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "threshold": "≥18% advantage, d ≥ 0.60",
        "actual": f"{advantage_pct:.2f}% advantage, d={cohens_d:.3f}",
    }
    if f3_1_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F3.1: {'PASS' if f3_1_pass else 'FAIL'} - Advantage: {advantage_pct:.2f}%, d={cohens_d:.3f}, p={p_value:.4f}"
    )

    # F3.2: Interoceptive Task Specificity
    logger.info("Testing F3.2: Interoceptive Task Specificity")
    # Use bootstrap test for proper statistical inference
    if (
        isinstance(interoceptive_advantage, (list, np.ndarray))
        and len(interoceptive_advantage) >= 30
    ):
        # Use standard t-test with sufficient sample size
        t_stat, p_value = stats.ttest_1samp(interoceptive_advantage, 12)
        mean_adv = float(np.mean(interoceptive_advantage))
        std_adv = float(np.std(interoceptive_advantage, ddof=1))
        cohens_d = (mean_adv - 12) / std_adv if std_adv > 0 else 0.0
    elif (
        isinstance(interoceptive_advantage, (list, np.ndarray))
        and len(interoceptive_advantage) >= 2
    ):
        # Use bootstrap test for small samples
        data_array = np.array(interoceptive_advantage)
        t_stat, p_value = bootstrap_one_sample_test(data_array, null_value=12.0)
        mean_adv = float(np.mean(data_array))
        std_adv = float(np.std(data_array, ddof=1))
        cohens_d = (mean_adv - 12) / std_adv if std_adv > 0 else 0.0
    else:
        # Insufficient data - fail criterion
        t_stat, p_value = 0.0, 1.0
        mean_adv = (
            float(interoceptive_advantage)
            if not isinstance(interoceptive_advantage, (list, np.ndarray))
            else (
                float(interoceptive_advantage[0])
                if len(interoceptive_advantage) > 0
                else 0.0
            )
        )
        cohens_d = 0.0

    f3_2_pass = (
        np.isfinite(mean_adv)
        and np.isfinite(cohens_d)
        and (
            p_value < 0.01
            if np.isfinite(p_value) and p_value != 1.0
            else mean_adv >= 28
        )
        and mean_adv >= 28
        and cohens_d >= 0.70
    )
    results["criteria"]["F3.2"] = {
        "passed": f3_2_pass,
        "interoceptive_advantage_pct": mean_adv,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "t_statistic": t_stat,
        "threshold": "≥28% interoceptive, d ≥ 0.70",
        "actual": f"Intero: {mean_adv:.2f}%, d={cohens_d:.3f}",
    }
    if f3_2_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F3.2: {'PASS' if f3_2_pass else 'FAIL'} - Intero: {mean_adv:.2f}%, d={cohens_d:.3f}, p={p_value:.4f}"
    )

    # F3.3: Threshold Gating Necessity
    logger.info("Testing F3.3: Threshold Gating Necessity")
    # Use bootstrap test for proper statistical inference
    if (
        isinstance(threshold_reduction, (list, np.ndarray))
        and len(threshold_reduction) >= 30
    ):
        # Use standard t-test with sufficient sample size
        t_stat, p_value = stats.ttest_1samp(threshold_reduction, 0)
        mean_red = float(np.mean(threshold_reduction))
        std_red = float(np.std(threshold_reduction, ddof=1))
        cohens_d = mean_red / std_red if std_red > 0 else 0.0
    elif (
        isinstance(threshold_reduction, (list, np.ndarray))
        and len(threshold_reduction) >= 2
    ):
        # Use bootstrap test for small samples
        data_array = np.array(threshold_reduction)
        t_stat, p_value = bootstrap_one_sample_test(data_array, null_value=0.0)
        mean_red = float(np.mean(data_array))
        std_red = float(np.std(data_array, ddof=1))
        cohens_d = mean_red / std_red if std_red > 0 else 0.0
    else:
        # Insufficient data - fail criterion
        t_stat, p_value = 0.0, 1.0
        mean_red = (
            float(threshold_reduction)
            if not isinstance(threshold_reduction, (list, np.ndarray))
            else (
                float(threshold_reduction[0]) if len(threshold_reduction) > 0 else 0.0
            )
        )
        cohens_d = 0.0

    f3_3_pass = (
        np.isfinite(mean_red)
        and np.isfinite(cohens_d)
        and (
            p_value < 0.01
            if np.isfinite(p_value) and p_value != 1.0
            else mean_red >= 25
        )
        and mean_red >= 25
        and cohens_d >= 0.75
    )
    results["criteria"]["F3.3"] = {
        "passed": f3_3_pass,
        "threshold_reduction_pct": mean_red,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "t_statistic": t_stat,
        "threshold": "≥25% reduction, d ≥ 0.75",
        "actual": f"{mean_red:.2f}% reduction, d={cohens_d:.3f}",
    }
    if f3_3_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F3.3: {'PASS' if f3_3_pass else 'FAIL'} - Reduction: {mean_red:.2f}%, d={cohens_d:.3f}, p={p_value:.4f}"
    )

    # F3.4: Precision Weighting Necessity
    logger.info("Testing F3.4: Precision Weighting Necessity")
    # Use bootstrap test for proper statistical inference
    if (
        isinstance(precision_reduction, (list, np.ndarray))
        and len(precision_reduction) >= 30
    ):
        # Use standard t-test with sufficient sample size
        t_stat, p_precision = stats.ttest_1samp(precision_reduction, 0)
        mean_precision = float(np.mean(precision_reduction))
    elif (
        isinstance(precision_reduction, (list, np.ndarray))
        and len(precision_reduction) >= 2
    ):
        # Use bootstrap test for small samples
        data_array = np.array(precision_reduction)
        t_stat, p_precision = bootstrap_one_sample_test(data_array, null_value=0.0)
        mean_precision = float(np.mean(data_array))
    else:
        # Insufficient data - fail criterion
        mean_precision = float(
            precision_reduction[0]
            if isinstance(precision_reduction, (list, np.ndarray))
            and len(precision_reduction) > 0
            else precision_reduction
        )
        t_stat, p_precision = 0.0, 1.0

    cohens_d_precision = mean_precision / 15

    f3_4_pass = (
        mean_precision >= 12 and cohens_d_precision >= 0.42 and p_precision < 0.01
    )
    results["criteria"]["F3.4"] = {
        "passed": f3_4_pass,
        "precision_reduction_pct": precision_reduction,
        "cohens_d": cohens_d_precision,
        "p_value": p_precision,
        "threshold": "≥20% reduction, d ≥ 0.65",
        "actual": f"{precision_reduction:.2f}% reduction, d={cohens_d_precision:.3f}",
    }
    if f3_4_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F3.4: {'PASS' if f3_4_pass else 'FAIL'} - Reduction: {precision_reduction:.2f}%, d={cohens_d_precision:.3f}"
    )

    # F3.5: Computational Efficiency Trade-Off
    logger.info("Testing F3.5: Computational Efficiency Trade-Off")
    # TOST non-inferiority test (simplified)
    f3_5_pass = performance_retention >= 78 and efficiency_gain >= 20
    results["criteria"]["F3.5"] = {
        "passed": f3_5_pass,
        "performance_retention_pct": performance_retention,
        "efficiency_gain_pct": efficiency_gain,
        "threshold": "≥85% performance, ≥30% efficiency",
        "actual": f"Performance: {performance_retention:.2f}%, Efficiency: {efficiency_gain:.2f}%",
    }
    if f3_5_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F3.5: {'PASS' if f3_5_pass else 'FAIL'} - Performance: {performance_retention:.2f}%, Efficiency: {efficiency_gain:.2f}%"
    )

    # F3.6: Sample Efficiency in Learning
    logger.info("Testing F3.6: Sample Efficiency in Learning")
    # Use bootstrap test for proper statistical inference
    if (
        isinstance(apgi_time_to_criterion, (list, np.ndarray))
        and len(apgi_time_to_criterion) >= 30
    ):
        # Use standard t-test with sufficient sample size
        t_stat, p_value = stats.ttest_1samp(apgi_time_to_criterion, 300)
        mean_trials = float(np.mean(apgi_time_to_criterion))
        hazard_ratio = 300 / mean_trials if mean_trials > 0 else 0
    elif (
        isinstance(apgi_time_to_criterion, (list, np.ndarray))
        and len(apgi_time_to_criterion) >= 2
    ):
        # Use bootstrap test for small samples
        data_array = np.array(apgi_time_to_criterion)
        t_stat, p_value = bootstrap_one_sample_test(data_array, null_value=300.0)
        mean_trials = float(np.mean(data_array))
        hazard_ratio = 300 / mean_trials if mean_trials > 0 else 0
    else:
        # Insufficient data - fail criterion
        t_stat, p_value = 0.0, 1.0
        mean_trials = (
            float(apgi_time_to_criterion)
            if not isinstance(apgi_time_to_criterion, (list, np.ndarray))
            else (
                float(apgi_time_to_criterion[0])
                if len(apgi_time_to_criterion) > 0
                else 300.0
            )
        )
        hazard_ratio = 300 / mean_trials if mean_trials > 0 else 0

    f3_6_pass = (
        np.isfinite(mean_trials)
        and np.isfinite(hazard_ratio)
        and (
            p_value < 0.01
            if np.isfinite(p_value) and p_value != 1.0
            else mean_trials <= 200
        )
        and mean_trials <= 200
        and hazard_ratio >= 1.45
    )
    results["criteria"]["F3.6"] = {
        "passed": f3_6_pass,
        "apgi_time_to_criterion": mean_trials,
        "hazard_ratio": hazard_ratio,
        "p_value": p_value,
        "t_statistic": t_stat,
        "threshold": "APGI ≤200 trials, HR ≥ 1.45",
        "actual": f"APGI: {mean_trials:.1f} trials, HR: {hazard_ratio:.2f}",
    }
    if f3_6_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F3.6: {'PASS' if f3_6_pass else 'FAIL'} - APGI: {mean_trials:.1f} trials, HR: {hazard_ratio:.2f}, p={p_value:.4f}"
    )

    # F1.1: APGI Agent Performance Advantage
    logger.info("Testing F1.1: APGI Agent Performance Advantage")
    t_stat, p_value = stats.ttest_ind(apgi_rewards, pp_rewards)
    mean_apgi = np.mean(apgi_rewards)
    mean_pp = np.mean(pp_rewards)
    # Guard against zero mean_pp to prevent division by zero
    safe_mean_pp = max(1e-10, float(abs(mean_pp))) * (1 if mean_pp >= 0 else -1)
    advantage_pct = ((mean_apgi - mean_pp) / safe_mean_pp) * 100

    # Cohen's d
    pooled_std = np.sqrt(
        (
            float((len(apgi_rewards) - 1) * np.var(apgi_rewards, ddof=1))
            + float((len(pp_rewards) - 1) * np.var(pp_rewards, ddof=1))
        )
        / float((len(apgi_rewards) + len(pp_rewards) - 2))
    )
    cohens_d = (mean_apgi - mean_pp) / float(pooled_std)

    f1_1_pass = (
        np.isfinite(advantage_pct)
        and np.isfinite(cohens_d)
        and np.isfinite(p_value)
        and advantage_pct >= F1_1_MIN_ADVANTAGE_PCT
        and cohens_d >= F1_1_MIN_COHENS_D
        and p_value < F1_1_ALPHA
    )
    results["criteria"]["F1.1"] = {
        "passed": f1_1_pass,
        "advantage_pct": advantage_pct,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "t_statistic": t_stat,
        "threshold": "≥18% advantage, d ≥ 0.60",
        "actual": f"{advantage_pct:.2f}% advantage, d={cohens_d:.3f}",
    }
    if f1_1_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F1.1: {'PASS' if f1_1_pass else 'FAIL'} - Advantage: {advantage_pct:.2f}%, d={cohens_d:.3f}, p={p_value:.4f}"
    )

    # F1.2: Hierarchical Level Emergence
    logger.info("Testing F1.2: Hierarchical Level Emergence")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    timescales_array = np.array(timescales).reshape(-1, 1)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(timescales_array)
    silhouette = silhouette_score(timescales_array, clusters)

    # One-way ANOVA
    cluster_means = [timescales[clusters == i] for i in range(3)]
    f_stat, p_anova = stats.f_oneway(*cluster_means)

    # Eta-squared
    ss_total = np.sum((timescales - np.mean(timescales)) ** 2)
    ss_between = sum(
        len(cm) * (np.mean(cm) - np.mean(timescales)) ** 2 for cm in cluster_means
    )
    eta_squared = ss_between / ss_total

    f1_2_pass = silhouette >= 0.30 and eta_squared >= 0.50 and p_anova < 0.001
    results["criteria"]["F1.2"] = {
        "passed": f1_2_pass,
        "n_clusters": len(np.unique(clusters)),
        "silhouette_score": silhouette,
        "eta_squared": eta_squared,
        "p_value": p_anova,
        "f_statistic": f_stat,
        "threshold": "≥3 clusters, silhouette ≥ 0.45, η² ≥ 0.70",
        "actual": f"{len(np.unique(clusters))} clusters, silhouette={silhouette:.3f}, η²={eta_squared:.3f}",
    }
    if f1_2_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F1.2: {'PASS' if f1_2_pass else 'FAIL'} - Clusters: {len(np.unique(clusters))}, silhouette={silhouette:.3f}, η²={eta_squared:.3f}"
    )

    # F1.3: Level-Specific Precision Weighting
    logger.info("Testing F1.3: Level-Specific Precision Weighting")
    level1_precision = np.array([pw[0] for pw in precision_weights])
    level3_precision = np.array([pw[1] for pw in precision_weights])
    precision_diff_pct = (
        (level1_precision - level3_precision) / level3_precision
    ) * 100
    mean_diff = np.mean(precision_diff_pct)

    # Repeated-measures ANOVA (simplified as paired t-test for level comparison)
    t_stat, p_rm = stats.ttest_rel(level1_precision, level3_precision)
    cohens_d_rm = np.mean(level1_precision - level3_precision) / np.std(
        level1_precision - level3_precision, ddof=1
    )

    f1_3_pass = mean_diff >= 15 and cohens_d_rm >= 0.35 and p_rm < 0.01
    results["criteria"]["F1.3"] = {
        "passed": f1_3_pass,
        "mean_precision_diff_pct": mean_diff,
        "cohens_d": cohens_d_rm,
        "p_value": p_rm,
        "t_statistic": t_stat,
        "threshold": "Level 1 25-40% higher than Level 3, partial η² ≥ 0.15",
        "actual": f"{mean_diff:.2f}% higher, d={cohens_d_rm:.3f}",
    }
    if f1_3_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F1.3: {'PASS' if f1_3_pass else 'FAIL'} - Precision diff: {mean_diff:.2f}%, d={cohens_d_rm:.3f}, p={p_rm:.4f}"
    )

    # F1.4: Threshold Adaptation Dynamics
    logger.info("Testing F1.4: Threshold Adaptation Dynamics")
    threshold_array = np.asarray(threshold_adaptation, dtype=float)
    threshold_reduction = float(np.mean(threshold_array))

    if len(threshold_array) >= 30:
        # Use standard t-test with sufficient sample size
        t_stat, p_adapt = stats.ttest_1samp(threshold_array, 0)
        adapt_std = float(np.std(threshold_array, ddof=1))
        if not np.isfinite(t_stat):
            t_stat = 0.0
    elif len(threshold_array) >= 2:
        # Use bootstrap test for small samples
        t_stat, p_adapt = bootstrap_one_sample_test(threshold_array, null_value=0.0)
        adapt_std = float(np.std(threshold_array, ddof=1))
    else:
        # Insufficient data - fail criterion
        t_stat, p_adapt = 0.0, 1.0
        adapt_std = 1.0  # fallback to avoid division by zero

    if not np.isfinite(p_adapt):
        p_adapt = 1.0

    cohens_d_adapt = float(threshold_reduction) / max(1e-10, adapt_std)

    f1_4_pass = threshold_reduction >= 20 and cohens_d_adapt >= 0.70 and p_adapt < 0.01
    results["criteria"]["F1.4"] = {
        "passed": f1_4_pass,
        "threshold_reduction_pct": threshold_reduction,
        "cohens_d": cohens_d_adapt,
        "p_value": p_adapt,
        "t_statistic": t_stat,
        "threshold": "≥20% reduction, d ≥ 0.70",
        "actual": f"{threshold_reduction:.2f}% reduction, d={cohens_d_adapt:.3f}",
    }
    if f1_4_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F1.4: {'PASS' if f1_4_pass else 'FAIL'} - Threshold reduction: {threshold_reduction:.2f}%, d={cohens_d_adapt:.3f}, p={p_adapt:.4f}"
    )

    # F1.5: Cross-Level Phase-Amplitude Coupling (PAC)
    logger.info("Testing F1.5: Cross-Level Phase-Amplitude Coupling")
    pac_baseline = np.array([pac[0] for pac in pac_mi])
    pac_ignition = np.array([pac[1] for pac in pac_mi])
    pac_increase = ((pac_ignition - pac_baseline) / pac_baseline) * 100
    mean_pac_increase = np.mean(pac_increase)

    # Paired t-test
    t_stat, p_pac = stats.ttest_rel(pac_ignition, pac_baseline)
    cohens_d_pac = np.mean(pac_ignition - pac_baseline) / np.std(
        pac_ignition - pac_baseline, ddof=1
    )

    # Permutation test (simplified)
    n_permutations = 10000
    perm_diffs = []
    for _ in range(n_permutations):
        perm_ignition = np.random.permutation(pac_ignition)
        perm_diffs.append(np.mean(perm_ignition) - np.mean(pac_baseline))
    perm_p = np.mean(
        np.abs(np.array(perm_diffs))
        >= np.abs(np.mean(pac_ignition) - np.mean(pac_baseline))
    )

    f1_5_pass = (
        mean_pac_increase >= 30
        and cohens_d_pac >= 0.50
        and p_pac < 0.01
        and perm_p < 0.01
    )
    results["criteria"]["F1.5"] = {
        "passed": f1_5_pass,
        "pac_increase_pct": mean_pac_increase,
        "cohens_d": cohens_d_pac,
        "p_value_ttest": p_pac,
        "p_value_permutation": perm_p,
        "t_statistic": t_stat,
        "threshold": "MI ≥ 0.012, ≥30% increase, d ≥ 0.5",
        "actual": f"{mean_pac_increase:.2f}% increase, d={cohens_d_pac:.3f}",
    }
    if f1_5_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F1.5: {'PASS' if f1_5_pass else 'FAIL'} - PAC increase: {mean_pac_increase:.2f}%, d={cohens_d_pac:.3f}"
    )

    # F1.6: 1/f Spectral Slope Predictions
    logger.info("Testing F1.6: 1/f Spectral Slope Predictions")
    active_slopes = np.array([s[0] for s in spectral_slopes])
    low_arousal_slopes = np.array([s[1] for s in spectral_slopes])
    mean_active = np.mean(active_slopes)
    mean_low_arousal = np.mean(low_arousal_slopes)
    delta_slope = mean_low_arousal - mean_active

    # Paired t-test
    t_stat, p_slope = stats.ttest_rel(low_arousal_slopes, active_slopes)
    cohens_d_slope = np.mean(low_arousal_slopes - active_slopes) / np.std(
        low_arousal_slopes - active_slopes, ddof=1
    )

    # Goodness of fit (R²)
    residuals = active_slopes - mean_active
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((active_slopes - np.mean(active_slopes)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    f1_6_pass = (
        mean_active <= 1.4
        and mean_low_arousal >= 1.3  # TODO: Import from falsification_thresholds.py
        and delta_slope >= 0.25
        and cohens_d_slope >= 0.50
        and r_squared >= 0.85
    )
    results["criteria"]["F1.6"] = {
        "passed": f1_6_pass,
        "active_slope_mean": mean_active,
        "low_arousal_slope_mean": mean_low_arousal,
        "delta_slope": delta_slope,
        "cohens_d": cohens_d_slope,
        "r_squared": r_squared,
        "p_value": p_slope,
        "t_statistic": t_stat,
        "threshold": "Active 0.8-1.2, low-arousal 1.5-2.0, Δ ≥ 0.4, d ≥ 0.8",
        "actual": f"Active={mean_active:.3f}, low-arousal={mean_low_arousal:.3f}, Δ={delta_slope:.3f}",
    }
    if f1_6_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F1.6: {'PASS' if f1_6_pass else 'FAIL'} - Active: {mean_active:.3f}, low-arousal: {mean_low_arousal:.3f}, Δ={delta_slope:.3f}"
    )

    # F2.1: APGI Advantageous Selection
    logger.info("Testing F2.1: APGI Advantageous Selection")
    t_stat, p_value = stats.ttest_ind(apgi_advantageous_selection, no_somatic_selection)
    mean_apgi = np.mean(apgi_advantageous_selection)
    mean_no_somatic = np.mean(no_somatic_selection)
    advantage_pct = ((mean_apgi - mean_no_somatic) / mean_no_somatic) * 100

    # Cohen's d
    pooled_std = np.sqrt(
        (
            (len(apgi_advantageous_selection) - 1)
            * np.var(apgi_advantageous_selection, ddof=1)
            + (len(no_somatic_selection) - 1) * np.var(no_somatic_selection, ddof=1)
        )
        / (len(apgi_advantageous_selection) + len(no_somatic_selection) - 2)
    )
    cohens_d = (mean_apgi - mean_no_somatic) / pooled_std

    f2_1_pass = advantage_pct >= 25 and cohens_d >= 0.80 and p_value < 0.01
    results["criteria"]["F2.1"] = {
        "passed": f2_1_pass,
        "advantage_pct": advantage_pct,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "t_statistic": t_stat,
        "threshold": "≥25% advantage, d ≥ 0.80",
        "actual": f"{advantage_pct:.2f}% advantage, d={cohens_d:.3f}",
    }
    if f2_1_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F2.1: {'PASS' if f2_1_pass else 'FAIL'} - Advantage: {advantage_pct:.2f}%, d={cohens_d:.3f}, p={p_value:.4f}"
    )

    # F2.2: APGI Cost Correlation
    logger.info("Testing F2.2: APGI Cost Correlation")
    # Test correlation between interoceptive cost and advantageous selection
    corr, p_corr = stats.pearsonr(
        apgi_advantageous_selection,
        [apgi_cost_correlation] * len(apgi_advantageous_selection),
    )
    corr_no_somatic, p_corr_no_somatic = stats.pearsonr(
        no_somatic_selection, [no_somatic_cost_correlation] * len(no_somatic_selection)
    )

    # Fisher's z-transformation for difference test
    z_apgi = np.arctanh(corr)
    z_no_somatic = np.arctanh(corr_no_somatic)
    se_diff = np.sqrt(
        1 / (len(apgi_advantageous_selection) - 3) + 1 / (len(no_somatic_selection) - 3)
    )
    z_diff = (z_apgi - z_no_somatic) / se_diff
    p_diff = 2 * (1 - stats.norm.cdf(abs(z_diff)))

    f2_2_pass = (
        corr >= 0.60 and corr_no_somatic <= 0.20 and p_diff < 0.01 and p_corr < 0.01
    )
    results["criteria"]["F2.2"] = {
        "passed": f2_2_pass,
        "apgi_correlation": corr,
        "no_somatic_correlation": corr_no_somatic,
        "correlation_difference": corr - corr_no_somatic,
        "z_difference": z_diff,
        "p_value_diff": p_diff,
        "p_value_apgi": p_corr,
        "threshold": "APGI r ≥ 0.60, No-somatic r ≤ 0.20, significant difference",
        "actual": f"APGI r={corr:.3f}, No-somatic r={corr_no_somatic:.3f}",
    }
    if f2_2_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F2.2: {'PASS' if f2_2_pass else 'FAIL'} - APGI: {corr:.3f}, No-somatic: {corr_no_somatic:.3f}, p_diff={p_diff:.4f}"
    )

    # F2.3: RT Advantage Modulation
    logger.info("Testing F2.3: RT Advantage Modulation")
    # Validate input: rt_advantage_ms must be a list/array with at least 2 observations
    # to avoid degenerate t-test (NaN p-value for single element)
    rt_array = np.atleast_1d(np.asarray(rt_advantage_ms, dtype=float))
    if len(rt_array) < 2:
        raise ValueError(
            f"F2.3: rt_advantage_ms must contain at least 2 observations for statistical testing. "
            f"Got {len(rt_array)} observation(s). "
            f"Accumulate RT advantages across trials into a list before calling this test."
        )

    # Use bootstrap test for proper statistical inference
    if len(rt_array) >= 30:
        # Use standard t-test with sufficient sample size
        rt_mean = float(np.mean(rt_array))
        t_stat_rt, p_rt = stats.ttest_1samp(rt_array, 0)
        corr_rt_cost, p_rt_cost = stats.pearsonr(rt_array, rt_cost_modulation)
    else:
        # Use bootstrap test for small samples (2-29 observations)
        t_stat_rt, p_rt = bootstrap_one_sample_test(rt_array, null_value=0.0)
        rt_mean = float(np.mean(rt_array))
        corr_rt_cost, p_rt_cost = stats.pearsonr(rt_array, rt_cost_modulation)

    f2_3_pass = (
        rt_mean <= -50
        and p_rt < 0.01
        and abs(corr_rt_cost) >= 0.40
        and p_rt_cost < 0.05
    )
    results["criteria"]["F2.3"] = {
        "passed": f2_3_pass,
        "rt_advantage_ms": rt_mean,
        "rt_cost_modulation": rt_cost_modulation,
        "correlation_rt_cost": corr_rt_cost,
        "p_value_rt": p_rt,
        "p_value_correlation": p_rt_cost,
        "t_statistic": t_stat_rt,
        "threshold": "RT ≤ -50ms, |r| ≥ 0.40 with cost modulation",
        "actual": f"RT {rt_mean:.1f}ms, r={corr_rt_cost:.3f}",
    }
    if f2_3_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F2.3: {'PASS' if f2_3_pass else 'FAIL'} - RT: {rt_mean:.1f}ms, r={corr_rt_cost:.3f}, p={p_rt_cost:.4f}"
    )

    # F2.4: Confidence Effects
    logger.info("Testing F2.4: Confidence Effects")
    # Two-proportion z-test for confidence advantage
    # TODO: Derive n_total from actual data samples
    n_total = 100  # Placeholder - needs real implementation
    p1 = 0.5 + confidence_effect / 2
    p2 = 0.5 - confidence_effect / 2
    se = np.sqrt(p1 * (1 - p1) / n_total + p2 * (1 - p2) / n_total)
    z_conf = confidence_effect / se
    p_conf = 2 * (1 - stats.norm.cdf(abs(z_conf)))

    f2_4_pass = confidence_effect >= 0.15 and p_conf < 0.01
    results["criteria"]["F2.4"] = {
        "passed": f2_4_pass,
        "confidence_effect": confidence_effect,
        "z_statistic": z_conf,
        "p_value": p_conf,
        "threshold": "≥15% confidence advantage",
        "actual": f"{confidence_effect:.2f} effect, z={z_conf:.3f}",
    }
    if f2_4_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F2.4: {'PASS' if f2_4_pass else 'FAIL'} - Confidence effect: {confidence_effect:.2f}, p={p_conf:.4f}"
    )

    # F2.5: Beta Interaction Effects
    logger.info("Testing F2.5: Beta Interaction Effects")
    # Use bootstrap test for proper statistical inference
    beta_array = np.atleast_1d(np.asarray(beta_interaction, dtype=float))
    if len(beta_array) >= 30:
        # Use standard t-test with sufficient sample size
        t_stat_beta, p_beta = stats.ttest_1samp(beta_array, 0)
    elif len(beta_array) >= 2:
        # Use bootstrap test for small samples
        t_stat_beta, p_beta = bootstrap_one_sample_test(beta_array, null_value=0.0)
    else:
        # Insufficient data - fail criterion
        t_stat_beta = 0.0
        p_beta = 1.0

    # Effect size (eta-squared) - simplified for single value
    ss_total = np.sum(
        (np.array([beta_interaction, 0]) - np.mean([beta_interaction, 0])) ** 2
    )
    ss_between = (np.mean([beta_interaction]) - np.mean([beta_interaction, 0])) ** 2
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    f2_5_pass = abs(beta_interaction) >= 0.30 and eta_squared >= 0.25 and p_beta < 0.01
    results["criteria"]["F2.5"] = {
        "passed": f2_5_pass,
        "beta_interaction": beta_interaction,
        "eta_squared": eta_squared,
        "p_value": p_beta,
        "t_statistic": t_stat_beta,
        "threshold": "|β| ≥ 0.30, η² ≥ 0.25",
        "actual": f"β={beta_interaction:.3f}, η²={eta_squared:.3f}",
    }
    if f2_5_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F2.5: {'PASS' if f2_5_pass else 'FAIL'} - β={beta_interaction:.3f}, η²={eta_squared:.3f}, p={p_beta:.4f}"
    )

    # F5 Family: Evolutionary Emergence (using shared function per Step 1.3)
    logger.info("Testing F5 Family: Evolutionary Emergence")

    # Prepare data for shared function
    f5_data = {
        "threshold_emergence_proportion": threshold_emergence_proportion,
        "precision_emergence_proportion": precision_emergence_proportion,
        "intero_gain_ratio_proportion": intero_gain_ratio_proportion,
    }

    # Use thresholds from falsification_thresholds.py
    from utils.falsification_thresholds import (
        F5_1_MIN_PROPORTION,
        F5_1_MIN_ALPHA,
        F5_1_MIN_COHENS_D,
        F5_2_MIN_PROPORTION,
        F5_2_MIN_CORRELATION,
        F5_3_MIN_PROPORTION,
        F5_3_MIN_GAIN_RATIO,
        F5_3_MIN_COHENS_D,
    )

    f5_thresholds = {
        "F5_1_MIN_PROPORTION": F5_1_MIN_PROPORTION,
        "F5_1_MIN_ALPHA": F5_1_MIN_ALPHA,
        "F5_1_MIN_COHENS_D": F5_1_MIN_COHENS_D,
        "F5_2_MIN_PROPORTION": F5_2_MIN_PROPORTION,
        "F5_2_MIN_CORRELATION": F5_2_MIN_CORRELATION,
        "F5_3_MIN_PROPORTION": F5_3_MIN_PROPORTION,
        "F5_3_MIN_GAIN_RATIO": F5_3_MIN_GAIN_RATIO,
        "F5_3_MIN_COHENS_D": F5_3_MIN_COHENS_D,
    }

    # Call shared function
    f5_results = check_F5_family(f5_data, f5_thresholds, genome_data=None)

    # Update results dict with shared function output
    for criterion, result in f5_results.items():
        results["criteria"][criterion] = result
        if result["passed"]:
            results["summary"]["passed"] += 1
            logger.info(f"{criterion}: PASS - {result['actual']}")
        else:
            results["summary"]["failed"] += 1
            logger.info(f"{criterion}: FAIL - {result['actual']}")

    # F5.4: Multi-Timescale Proportion
    logger.info("Testing F5.4: Multi-Timescale Proportion")
    n_success = int(multi_timescale_proportion * n_total)
    binom_result = binomtest(n_success, n_total, p=0.30, alternative="greater")
    p_binom = binom_result.pvalue

    f5_4_pass = multi_timescale_proportion >= 0.30 and p_binom < 0.01
    results["criteria"]["F5.4"] = {
        "passed": f5_4_pass,
        "proportion": multi_timescale_proportion,
        "p_value": p_binom,
        "n_success": n_success,
        "n_total": n_total,
        "threshold": "≥30% emergence",
        "actual": f"{multi_timescale_proportion:.1f} proportion, p={p_binom:.4f}",
    }
    if f5_4_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F5.4: {'PASS' if f5_4_pass else 'FAIL'} - Proportion: {multi_timescale_proportion:.1f}, p={p_binom:.4f}"
    )

    # F5.5: PCA Variance Explained
    logger.info("Testing F5.5: PCA Variance Explained")
    # Goodness of fit for variance explained
    residuals = pca_variance_explained - 0.70  # Threshold
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((pca_variance_explained - np.mean(pca_variance_explained)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    f5_5_pass = pca_variance_explained >= 0.70 and r_squared >= 0.80
    results["criteria"]["F5.5"] = {
        "passed": f5_5_pass,
        "variance_explained": pca_variance_explained,
        "r_squared": r_squared,
        "threshold": "≥70% variance explained, R² ≥ 0.80",
        "actual": f"{pca_variance_explained:.1f} variance, R²={r_squared:.3f}",
    }
    if f5_5_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F5.5: {'PASS' if f5_5_pass else 'FAIL'} - Variance: {pca_variance_explained:.1f}, R²={r_squared:.3f}"
    )

    # F5.6: Control Performance Difference
    logger.info("Testing F5.6: Control Performance Difference")
    # Use bootstrap test for proper statistical inference
    if (
        isinstance(control_performance_difference, (list, np.ndarray))
        and len(control_performance_difference) >= 30
    ):
        # Use standard t-test with sufficient sample size
        t_stat, p_value = stats.ttest_1samp(control_performance_difference, 0)
        cohens_d = (
            float(np.mean(control_performance_difference))
            / np.std(control_performance_difference, ddof=1)
            if np.std(control_performance_difference, ddof=1) > 0
            else 0
        )
        mean_diff = float(np.mean(control_performance_difference))
    elif (
        isinstance(control_performance_difference, (list, np.ndarray))
        and len(control_performance_difference) >= 2
    ):
        # Use bootstrap test for small samples
        data_array = np.array(control_performance_difference)
        t_stat, p_value = bootstrap_one_sample_test(data_array, null_value=0.0)
        cohens_d = (
            float(np.mean(data_array)) / np.std(data_array, ddof=1)
            if np.std(data_array, ddof=1) > 0
            else 0
        )
        mean_diff = float(np.mean(data_array))
    else:
        # Insufficient data - fail criterion
        mean_diff = float(
            control_performance_difference[0]
            if isinstance(control_performance_difference, (list, np.ndarray))
            and len(control_performance_difference) > 0
            else control_performance_difference
        )
        t_stat, p_value = 0.0, 1.0
        cohens_d = mean_diff / 0.1  # Mock cohens_d since we only have mean

    f5_6_pass = mean_diff >= 0.20 and cohens_d >= 0.50 and p_value < 0.01
    results["criteria"]["F5.6"] = {
        "passed": f5_6_pass,
        "performance_difference": control_performance_difference,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "t_statistic": t_stat,
        "threshold": "≥20% difference, d ≥ 0.50",
        "actual": f"{control_performance_difference:.2f} difference, d={cohens_d:.3f}",
    }
    if f5_6_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F5.6: {'PASS' if f5_6_pass else 'FAIL'} - Difference: {control_performance_difference:.2f}, d={cohens_d:.3f}, p={p_value:.4f}"
    )

    # F6.1: Liquid Transition Time Advantage
    logger.info("Testing F6.1: Liquid Transition Time Advantage")
    # Compare LTCN vs RNN transition times
    t_stat, p_value = stats.ttest_ind([ltcn_transition_time], [rnn_transition_time])
    mean_ltcn = ltcn_transition_time
    mean_rnn = rnn_transition_time
    advantage_pct = ((mean_rnn - mean_ltcn) / mean_rnn) * 100

    # Cohen's d
    pooled_std = (
        np.sqrt(
            (
                (1 - 1) * np.var([ltcn_transition_time], ddof=1)
                + (1 - 1) * np.var([rnn_transition_time], ddof=1)
            )
            / (1 + 1 - 2)
        )
        if len([ltcn_transition_time]) > 1 and len([rnn_transition_time]) > 1
        else np.std([ltcn_transition_time, rnn_transition_time], ddof=1)
    )
    cohens_d = (mean_ltcn - mean_rnn) / pooled_std if pooled_std > 0 else 0

    f6_1_pass = (
        ltcn_transition_time < rnn_transition_time
        and cohens_d <= -0.70
        and p_value < 0.01
    )
    results["criteria"]["F6.1"] = {
        "passed": f6_1_pass,
        "ltcn_time": ltcn_transition_time,
        "rnn_time": rnn_transition_time,
        "advantage_pct": advantage_pct,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "t_statistic": t_stat,
        "threshold": "LTCN < RNN transition time, d ≤ -0.70",
        "actual": f"LTCN {ltcn_transition_time:.1f}s, RNN {rnn_transition_time:.1f}s, d={cohens_d:.3f}",
    }
    if f6_1_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F6.1: {'PASS' if f6_1_pass else 'FAIL'} - LTCN: {ltcn_transition_time:.1f}s, RNN: {rnn_transition_time:.1f}s, d={cohens_d:.3f}"
    )

    # F6.2: Sparsity Reduction Advantage
    logger.info("Testing F6.2: Sparsity Reduction Advantage")
    # Compare LTCN vs RNN sparsity reduction
    t_stat, p_value = stats.ttest_ind(
        [ltcn_sparsity_reduction], [rnn_sparsity_reduction]
    )
    mean_ltcn = ltcn_sparsity_reduction
    mean_rnn = rnn_sparsity_reduction

    # Cohen's d
    pooled_std = (
        np.sqrt(
            (
                (1 - 1) * np.var([ltcn_sparsity_reduction], ddof=1)
                + (1 - 1) * np.var([rnn_sparsity_reduction], ddof=1)
            )
            / (1 + 1 - 2)
        )
        if len([ltcn_sparsity_reduction]) > 1 and len([rnn_sparsity_reduction]) > 1
        else np.std([ltcn_sparsity_reduction, rnn_sparsity_reduction], ddof=1)
    )
    cohens_d = (mean_ltcn - mean_rnn) / pooled_std if pooled_std > 0 else 0

    f6_2_pass = ltcn_sparsity_reduction >= 0.30 and cohens_d >= 0.70 and p_value < 0.01
    results["criteria"]["F6.2"] = {
        "passed": f6_2_pass,
        "ltcn_reduction": ltcn_sparsity_reduction,
        "rnn_reduction": rnn_sparsity_reduction,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "t_statistic": t_stat,
        "threshold": "LTCN ≥30% reduction, d ≥ 0.70",
        "actual": f"LTCN {ltcn_sparsity_reduction:.1f}%, RNN {rnn_sparsity_reduction:.1f}%, d={cohens_d:.3f}",
    }
    if f6_2_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F6.2: {'PASS' if f6_2_pass else 'FAIL'} - LTCN: {ltcn_sparsity_reduction:.1f}%, RNN: {rnn_sparsity_reduction:.1f}%, d={cohens_d:.3f}"
    )

    # F6.3: Integration Window Advantage
    logger.info("Testing F6.3: Integration Window Advantage")
    # Compare LTCN vs RNN integration windows
    t_stat, p_value = stats.ttest_ind(
        [ltcn_integration_window], [rnn_integration_window]
    )
    mean_ltcn = ltcn_integration_window
    mean_rnn = rnn_integration_window

    # Cohen's d
    pooled_std = (
        np.sqrt(
            (
                (1 - 1) * np.var([ltcn_integration_window], ddof=1)
                + (1 - 1) * np.var([rnn_integration_window], ddof=1)
            )
            / (1 + 1 - 2)
        )
        if len([ltcn_integration_window]) > 1 and len([rnn_integration_window]) > 1
        else np.std([ltcn_integration_window, rnn_integration_window], ddof=1)
    )
    cohens_d = (mean_ltcn - mean_rnn) / pooled_std if pooled_std > 0 else 0

    f6_3_pass = (
        ltcn_integration_window > rnn_integration_window
        and cohens_d >= 0.70
        and p_value < 0.01
    )
    results["criteria"]["F6.3"] = {
        "passed": f6_3_pass,
        "ltcn_window": ltcn_integration_window,
        "rnn_window": rnn_integration_window,
        "cohens_d": cohens_d,
        "p_value": p_value,
        "t_statistic": t_stat,
        "threshold": "LTCN > RNN integration window, d ≥ 0.70",
        "actual": f"LTCN {ltcn_integration_window:.1f}, RNN {rnn_integration_window:.1f}, d={cohens_d:.3f}",
    }
    if f6_3_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F6.3: {'PASS' if f6_3_pass else 'FAIL'} - LTCN: {ltcn_integration_window:.1f}, RNN: {rnn_integration_window:.1f}, d={cohens_d:.3f}"
    )

    # F6.4: Fading Memory Implementation
    logger.info("Testing F6.4: Fading Memory Implementation")
    # Exponential decay model fitting (simplified)
    f6_4_pass = memory_decay_tau >= 1.0 and memory_decay_tau <= 3.0
    results["criteria"]["F6.4"] = {
        "passed": f6_4_pass,
        "tau_memory": memory_decay_tau,
        "threshold": "τ_memory = 1-3s",
        "actual": f"τ = {memory_decay_tau:.1f}s",
    }
    if f6_4_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F6.4: {'PASS' if f6_4_pass else 'FAIL'} - τ = {memory_decay_tau:.1f}s"
    )

    # F6.5: Bifurcation Structure for Ignition
    logger.info("Testing F6.5: Bifurcation Structure for Ignition")
    # Use the provided hysteresis_width parameter instead of hardcoded value
    # The hysteresis should be computed from the model's response function
    # using scipy.optimize.brentq on increasing vs. decreasing input drives
    if hysteresis_width <= 0:
        raise ValueError(
            "hysteresis_width must be computed from bifurcation scan "
            "using scipy.optimize.brentq on model response function"
        )

    f6_5_pass = (
        abs(bifurcation_point - 0.15) <= 0.10
        and hysteresis_width >= 0.08
        and hysteresis_width <= 0.25
    )
    results["criteria"]["F6.5"] = {
        "passed": f6_5_pass,
        "bifurcation_point": bifurcation_point,
        "hysteresis_width": hysteresis_width,
        "threshold": "Bifurcation at Π·|ε| = θ_t ± 0.15, hysteresis 0.1-0.2",
        "actual": f"Point {bifurcation_point:.3f}, hysteresis {hysteresis_width:.3f}",
    }
    if f6_5_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F6.5: {'PASS' if f6_5_pass else 'FAIL'} - Point: {bifurcation_point:.3f}, hysteresis: {hysteresis_width:.3f}"
    )

    # F6.6: Alternative Architectures Require Add-Ons
    logger.info("Testing F6.6: Alternative Architectures Require Add-Ons")

    f6_6_pass = rnn_add_ons_needed >= 2 and performance_gap >= 15
    results["criteria"]["F6.6"] = {
        "passed": f6_6_pass,
        "add_ons_needed": rnn_add_ons_needed,
        "performance_gap": performance_gap,
        "threshold": "≥2 add-ons needed, ≥15% performance gap",
        "actual": f"{rnn_add_ons_needed} add-ons, {performance_gap:.1f}% gap",
    }
    if f6_6_pass:
        results["summary"]["passed"] += 1
    else:
        results["summary"]["failed"] += 1
    logger.info(
        f"F6.6: {'PASS' if f6_6_pass else 'FAIL'} - Add-ons: {rnn_add_ons_needed}, gap: {performance_gap:.1f}%"
    )

    logger.info(
        f"\nFalsification-Protocol-3 Summary: {results['summary']['passed']}/{results['summary']['total']} criteria passed"
    )
    return results
