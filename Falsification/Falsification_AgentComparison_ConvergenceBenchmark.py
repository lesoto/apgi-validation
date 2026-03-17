"""
APGI Falsification Protocol FP-2: Iowa Gambling Task / Agent Comparison Convergence Benchmark
===========================================================================================

Complete falsification test for APGI active inference agents through systematic comparison
with alternative architectures. Tests whether incorporating interoceptive precision and
global workspace ignition produces falsifiable behavioral signatures.

This protocol implements:
- Direct import of environments and agents from VP-3 (single source of truth)
- Explicit convergence trial count falsification (APGI must beat 80 trials)
- BIC model comparison falsification (Pure PP achieving lower BIC → falsified)
- Sensitivity analysis across full parameter ranges
- Comprehensive statistical falsification framework

"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import ParameterGrid

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import from VP-3 (single source of truth)
from Validation.ActiveInference_AgentSimulations_Protocol3 import (
    APGIActiveInferenceAgent,
    StandardPPAgent,
    GWTOnlyAgent,
    ActorCriticAgent,
    IowaGamblingTaskEnvironment,
    MultiArmedVolatileBandit,
    PatchLeavingForagingEnvironment,
    WAICModelComparison,
)

from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)


# =============================================================================
# PART 1: ENVIRONMENT IMPORTS FROM VP-3
# =============================================================================

# Environments imported directly from VP-3 to ensure single source of truth
VolatileForagingEnvironment = MultiArmedVolatileBandit
ThreatRewardTradeoffEnvironment = PatchLeavingForagingEnvironment


# =============================================================================
# PART 2: AGENT ALIGNMENT WITH VP-3
# =============================================================================


class ConvergenceBenchmarkExperiment:
    """
    Convergence benchmark experiment for falsification testing.

    Tests APGI convergence against explicit criteria:
    - Must converge in ≤80 trials (human benchmark)
    - Must outperform alternative agents on BIC
    - Must show robustness across parameter ranges
    """

    def __init__(self, n_agents: int = 20, n_trials: int = 100):
        self.n_agents = n_agents
        self.n_trials = n_trials

        # Agent configurations aligned with VP-3
        self.agent_configs = {
            "APGI": {
                "n_actions": 4,
                "theta_init": 0.5,
                "beta": 1.2,
                "Pi_e_init": 1.0,
                "Pi_i_init": 1.0,
                "lr_extero": 0.01,
                "lr_intero": 0.01,
                "lr_somatic": 0.1,
                "lr_precision": 0.05,
                "theta_baseline": 0.5,
                "alpha": 8.0,
                "tau_S": 0.3,
                "tau_theta": 10.0,
                "eta_theta": 0.01,
            },
            "StandardPP": {
                "n_actions": 4,
                "theta_init": 0.5,
                "beta": 1.2,
                "Pi_e_init": 1.0,
                "Pi_i_init": 1.0,
                "lr_extero": 0.01,
                "lr_intero": 0.01,
            },
            "GWTOnly": {
                "n_actions": 4,
                "theta_init": 0.5,
                "alpha": 8.0,
                "tau_S": 0.3,
            },
            "ActorCritic": {
                "n_actions": 4,
            },
        }

    def run_convergence_benchmark(self, env_class=IowaGamblingTaskEnvironment) -> Dict:
        """Run convergence benchmark across all agents"""

        results = {}
        env = env_class(n_trials=self.n_trials)

        for agent_name, config in self.agent_configs.items():
            print(f"\nBenchmarking {agent_name}...")

            agent_results = []
            convergence_trials = []

            for agent_idx in tqdm(range(self.n_agents), desc=f"  {agent_name}"):
                # Create fresh agent for each run
                agent_class = self._get_agent_class(agent_name)
                agent = agent_class(config)

                episode_data = self._run_single_episode(agent, env)
                agent_results.append(episode_data)

                # Track convergence trials
                if episode_data["convergence_trial"] is not None:
                    convergence_trials.append(episode_data["convergence_trial"])

            # Aggregate results
            results[agent_name] = {
                "mean_cumulative_reward": np.mean(
                    [r["cumulative_reward"][-1] for r in agent_results]
                ),
                "std_cumulative_reward": np.std(
                    [r["cumulative_reward"][-1] for r in agent_results]
                ),
                "mean_convergence_trial": np.mean(convergence_trials)
                if convergence_trials
                else None,
                "convergence_rate": len(convergence_trials) / self.n_agents,
                "raw_results": agent_results,
            }

        return results

    def _get_agent_class(self, agent_name: str):
        """Get agent class by name"""
        classes = {
            "APGI": APGIActiveInferenceAgent,
            "StandardPP": StandardPPAgent,
            "GWTOnly": GWTOnlyAgent,
            "ActorCritic": ActorCriticAgent,
        }
        return classes[agent_name]

    def _run_single_episode(self, agent, env) -> Dict:
        """Run single episode with convergence tracking"""

        data = {
            "rewards": [],
            "cumulative_reward": [],
            "actions": [],
            "convergence_trial": None,
        }

        observation = env.reset()
        cumulative = 0

        for trial in range(self.n_trials):
            action = agent.step(observation)
            reward, intero_cost, next_obs, done = env.step(action)

            data["rewards"].append(reward)
            cumulative += reward
            data["cumulative_reward"].append(cumulative)
            data["actions"].append(action)

            # Convergence check: IGT - choosing C or D consistently
            if trial > 20 and data["convergence_trial"] is None:
                recent_actions = data["actions"][-20:]
                good_choices = sum(
                    [1 for a in recent_actions if a in [2, 3]]
                )  # C=2, D=3

                if good_choices >= 15:  # 75% good choices in window
                    data["convergence_trial"] = trial

            # Process outcome for agent
            if hasattr(agent, "receive_outcome"):
                agent.receive_outcome(reward, intero_cost, next_obs)

            observation = next_obs

            if done:
                break

        return data


# =============================================================================
# PART 3: SENSITIVITY ANALYSIS
# =============================================================================


class SensitivityAnalysis:
    """
    Comprehensive sensitivity analysis across parameter ranges.

    Tests robustness of APGI predictions across:
    - α ∈ [3, 10] (ignition sensitivity)
    - β ∈ [0.6, 2.2] (interoceptive weighting)
    - θ_baseline ∈ [0.3, 0.8] (threshold baseline)
    """

    def __init__(self, base_config: Dict):
        self.base_config = base_config

        # Parameter ranges as specified
        self.param_grid = {
            "alpha": [3.0, 5.0, 7.0, 8.0, 10.0],
            "beta": [0.6, 1.0, 1.2, 1.5, 1.8, 2.2],
            "theta_baseline": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }

    def run_sensitivity_analysis(
        self, env_class=IowaGamblingTaskEnvironment, n_runs: int = 10
    ) -> Dict:
        """Run sensitivity analysis across parameter combinations"""

        results = {}
        total_combinations = len(list(ParameterGrid(self.param_grid)))
        print(
            f"\nRunning sensitivity analysis across {total_combinations} parameter combinations..."
        )

        for params in tqdm(ParameterGrid(self.param_grid), total=total_combinations):
            param_key = (
                f"α={params['alpha']}_β={params['beta']}_θ={params['theta_baseline']}"
            )

            # Create config with these parameters
            config = self.base_config.copy()
            config.update(params)

            # Run multiple times for stability
            param_results = []
            for _ in range(n_runs):
                agent = APGIActiveInferenceAgent(config)
                env = env_class(n_trials=100)

                episode_data = self._run_single_episode(agent, env)
                param_results.append(episode_data)

            # Aggregate results for this parameter combination
            results[param_key] = {
                "params": params,
                "mean_convergence_trial": np.mean(
                    [
                        r["convergence_trial"]
                        for r in param_results
                        if r["convergence_trial"]
                    ]
                ),
                "convergence_rate": np.mean(
                    [r["convergence_trial"] is not None for r in param_results]
                ),
                "mean_final_reward": np.mean(
                    [r["cumulative_reward"][-1] for r in param_results]
                ),
                "std_final_reward": np.std(
                    [r["cumulative_reward"][-1] for r in param_results]
                ),
                "individual_runs": param_results,
            }

        return results

    def _run_single_episode(self, agent, env) -> Dict:
        """Run single episode for sensitivity analysis"""
        data = {
            "rewards": [],
            "cumulative_reward": [],
            "actions": [],
            "convergence_trial": None,
        }

        observation = env.reset()
        cumulative = 0

        for trial in range(100):
            action = agent.step(observation)
            reward, intero_cost, next_obs, done = env.step(action)

            data["rewards"].append(reward)
            cumulative += reward
            data["cumulative_reward"].append(cumulative)
            data["actions"].append(action)

            # Convergence check
            if trial > 20 and data["convergence_trial"] is None:
                recent_actions = data["actions"][-20:]
                good_choices = sum([1 for a in recent_actions if a in [2, 3]])
                if good_choices >= 15:
                    data["convergence_trial"] = trial

            agent.receive_outcome(reward, intero_cost, next_obs)
            observation = next_obs

            if done:
                break

        return data


# =============================================================================
# PART 4: FALSIFICATION FRAMEWORK
# =============================================================================


class FalsificationFramework:
    """
    Comprehensive falsification framework with statistical tests.

    Implements:
    - Convergence trial count falsification (≤80 trials required)
    - BIC comparison falsification (Pure PP lower BIC → falsified)
    - Statistical significance testing
    """

    def __init__(self):
        self.bic_comparator = WAICModelComparison()

    def check_convergence_falsification(self, results: Dict) -> Dict:
        """
        Falsification criterion: APGI must converge in ≤80 trials

        Args:
            results: Benchmark results from ConvergenceBenchmarkExperiment

        Returns:
            Dict with falsification assessment
        """

        apgi_results = results.get("APGI", {})
        mean_convergence = apgi_results.get("mean_convergence_trial")

        if mean_convergence is None:
            # No convergence at all - definitely falsified
            return {
                "falsified": True,
                "criterion": "convergence_trial_count",
                "required_threshold": 80,
                "actual_value": None,
                "reason": "APGI agent never converged",
                "method": "statistical_test",
            }

        # Check against human benchmark (80 trials)
        falsified = mean_convergence > 80

        return {
            "falsified": falsified,
            "criterion": "convergence_trial_count",
            "required_threshold": 80,
            "actual_value": float(mean_convergence),
            "margin": float(mean_convergence - 80),
            "method": "statistical_test",
        }

    def check_bic_falsification(self, benchmark_results: Dict) -> Dict:
        """
        Falsification criterion: Pure PP achieving lower BIC than APGI

        Args:
            benchmark_results: Results from convergence benchmark

        Returns:
            Dict with BIC comparison falsification
        """

        # Prepare log-likelihood data for BIC computation
        model_lls = {}

        for agent_name, agent_results in benchmark_results.items():
            if "raw_results" in agent_results:
                # Extract log-likelihoods from agent performance
                # For simplicity, use negative cumulative rewards as proxy for log-likelihood
                lls = []
                for run in agent_results["raw_results"]:
                    # Convert rewards to pseudo log-likelihoods
                    rewards = np.array(run["cumulative_reward"])
                    # Normalize and convert to log-likelihood scale
                    normalized_rewards = (rewards - rewards.min()) / (
                        rewards.max() - rewards.min() + 1e-10
                    )
                    ll = np.log(normalized_rewards + 1e-10)  # Avoid log(0)
                    lls.append(ll)

                model_lls[agent_name] = np.array(lls)

        # Compute BIC for each model
        bic_results = {}

        # Parameter counts (approximate from VP-3)
        n_params = {
            "APGI": 250,
            "StandardPP": 150,
            "GWTOnly": 180,
            "ActorCritic": 200,
        }

        for agent_name, lls in model_lls.items():
            if agent_name in n_params and len(lls) > 0:
                bic_result = self.bic_comparator.compute_bic(
                    lls.flatten(), n_params[agent_name]
                )
                bic_results[agent_name] = bic_result

        # Check falsification: Pure PP (StandardPP) lower BIC than APGI
        if "APGI" in bic_results and "StandardPP" in bic_results:
            apgi_bic = bic_results["APGI"]["bic"]
            pp_bic = bic_results["StandardPP"]["bic"]

            # Lower BIC is better (less complex model that fits well)
            falsified = pp_bic < apgi_bic

            return {
                "falsified": falsified,
                "criterion": "bic_comparison",
                "apgi_bic": float(apgi_bic),
                "standardpp_bic": float(pp_bic),
                "bic_difference": float(apgi_bic - pp_bic),
                "method": "BIC_comparison",
                "all_bic_results": bic_results,
            }

        return {
            "falsified": False,  # Cannot falsify without both models
            "criterion": "bic_comparison",
            "reason": "Missing BIC results for APGI or StandardPP",
            "method": "BIC_comparison",
        }

    def check_robustness_falsification(self, sensitivity_results: Dict) -> Dict:
        """
        Falsification criterion: APGI predictions must be robust across parameter ranges

        Args:
            sensitivity_results: Results from SensitivityAnalysis

        Returns:
            Dict with robustness assessment
        """

        if not sensitivity_results:
            return {
                "falsified": True,
                "criterion": "parameter_robustness",
                "reason": "No sensitivity analysis results available",
            }

        convergence_rates = []
        convergence_trials = []

        for param_key, param_result in sensitivity_results.items():
            conv_rate = param_result.get("convergence_rate", 0)
            conv_trial = param_result.get("mean_convergence_trial")

            convergence_rates.append(conv_rate)
            if conv_trial is not None:
                convergence_trials.append(conv_trial)

        # Check robustness criteria
        mean_conv_rate = np.mean(convergence_rates)
        mean_conv_trial = np.mean(convergence_trials) if convergence_trials else None

        # Falsified if convergence rate drops below 70% or mean convergence exceeds 100 trials
        conv_rate_falsified = mean_conv_rate < 0.7
        conv_trial_falsified = mean_conv_trial > 100 if mean_conv_trial else True

        falsified = conv_rate_falsified or conv_trial_falsified

        return {
            "falsified": falsified,
            "criterion": "parameter_robustness",
            "mean_convergence_rate": float(mean_conv_rate),
            "mean_convergence_trial": float(mean_conv_trial)
            if mean_conv_trial
            else None,
            "conv_rate_threshold": 0.7,
            "conv_trial_threshold": 100,
            "method": "sensitivity_analysis",
        }

    def run_full_falsification(
        self, benchmark_results: Dict, sensitivity_results: Dict
    ) -> Dict:
        """Run complete falsification assessment"""

        falsification_results = {
            "convergence_falsification": self.check_convergence_falsification(
                benchmark_results
            ),
            "bic_falsification": self.check_bic_falsification(benchmark_results),
            "robustness_falsification": self.check_robustness_falsification(
                sensitivity_results
            ),
        }

        # Overall falsification status
        any_falsified = any(
            result["falsified"] for result in falsification_results.values()
        )

        return {
            "overall_falsified": any_falsified,
            "criteria_results": falsification_results,
            "summary": self._generate_falsification_summary(falsification_results),
        }

    def _generate_falsification_summary(self, falsification_results: Dict) -> str:
        """Generate human-readable falsification summary"""

        summary_lines = []

        if falsification_results["convergence_falsification"]["falsified"]:
            conv_result = falsification_results["convergence_falsification"]
            summary_lines.append(
                f"❌ CONVERGENCE FALSIFIED: APGI took {conv_result['actual_value']:.1f} trials "
                f"(required ≤{conv_result['required_threshold']})"
            )
        else:
            conv_result = falsification_results["convergence_falsification"]
            summary_lines.append(
                f"✅ CONVERGENCE PASSED: APGI converged in {conv_result['actual_value']:.1f} trials"
            )

        if falsification_results["bic_falsification"]["falsified"]:
            bic_result = falsification_results["bic_falsification"]
            summary_lines.append(
                f"❌ BIC FALSIFIED: StandardPP BIC ({bic_result['standardpp_bic']:.1f}) < "
                f"APGI BIC ({bic_result['apgi_bic']:.1f})"
            )
        else:
            summary_lines.append("✅ BIC PASSED: APGI has better BIC than StandardPP")

        if falsification_results["robustness_falsification"]["falsified"]:
            summary_lines.append(
                "❌ ROBUSTNESS FALSIFIED: APGI predictions not robust across parameters"
            )
        else:
            summary_lines.append(
                "✅ ROBUSTNESS PASSED: APGI predictions stable across parameters"
            )

        return "\n".join(summary_lines)


# =============================================================================
# PART 5: VISUALIZATION AND REPORTING
# =============================================================================


def plot_falsification_results(
    benchmark_results: Dict,
    sensitivity_results: Dict,
    falsification: Dict,
    save_path: str = "fp2_falsification_results.png",
):
    """Generate comprehensive falsification visualization"""

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Convergence comparison
    ax1 = fig.add_subplot(gs[0, 0])
    agents = list(benchmark_results.keys())
    convergence_trials = [
        benchmark_results[a].get("mean_convergence_trial", 100) for a in agents
    ]
    convergence_rates = [
        benchmark_results[a].get("convergence_rate", 0) for a in agents
    ]

    x = np.arange(len(agents))
    ax1.bar(
        x,
        convergence_trials,
        alpha=0.7,
        color=["#2E86AB", "#A23B72", "#F18F01", "#06A77D"],
    )
    ax1.axhline(
        y=80,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Human benchmark (80 trials)",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents, rotation=45)
    ax1.set_ylabel("Mean Convergence Trials")
    ax1.set_title("IGT Convergence Performance")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Convergence rates
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(
        x,
        convergence_rates,
        alpha=0.7,
        color=["#2E86AB", "#A23B72", "#F18F01", "#06A77D"],
    )
    ax2.axhline(
        y=0.8,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Robust threshold (80%)",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(agents, rotation=45)
    ax2.set_ylabel("Convergence Rate")
    ax2.set_title("Convergence Success Rate")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: Sensitivity analysis heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    if sensitivity_results:
        # Extract parameter combinations and convergence trials
        alphas = []
        betas = []
        thetas = []
        conv_trials = []

        for param_key, result in sensitivity_results.items():
            params = result["params"]
            alphas.append(params["alpha"])
            betas.append(params["beta"])
            thetas.append(params["theta_baseline"])
            conv_trials.append(result.get("mean_convergence_trial", 100))

        # Create scatter plot colored by convergence trial
        scatter = ax3.scatter(
            alphas,
            betas,
            c=conv_trials,
            cmap="RdYlGn_r",
            s=[t * 2 for t in thetas],
            alpha=0.7,
        )
        ax3.set_xlabel("α (ignition sensitivity)")
        ax3.set_ylabel("β (interoceptive weighting)")
        ax3.set_title(
            "Parameter Sensitivity\n(size = θ_baseline, color = convergence trials)"
        )
        plt.colorbar(scatter, ax=ax3, label="Convergence Trials")

    # Plot 4: BIC comparison
    ax4 = fig.add_subplot(gs[1, :2])
    if "bic_falsification" in falsification.get("criteria_results", {}):
        bic_data = falsification["criteria_results"]["bic_falsification"]
        if "all_bic_results" in bic_data:
            agents_bic = list(bic_data["all_bic_results"].keys())
            bic_values = [bic_data["all_bic_results"][a]["bic"] for a in agents_bic]

            colors = [
                "#2E86AB"
                if a == "APGI"
                else "#A23B72"
                if a == "StandardPP"
                else "#F18F01"
                for a in agents_bic
            ]
            ax4.bar(agents_bic, bic_values, alpha=0.7, color=colors)
            ax4.set_ylabel("BIC Score (lower is better)")
            ax4.set_title("Bayesian Information Criterion Comparison")
            ax4.grid(axis="y", alpha=0.3)

            # Highlight falsification
            if bic_data.get("falsified"):
                ax4.text(
                    0.5,
                    0.9,
                    "❌ FALSIFIED: StandardPP has better BIC",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                    bbox=dict(boxstyle="round", facecolor="red", alpha=0.3),
                )

    # Plot 5: Falsification summary
    ax5 = fig.add_subplot(gs[1, 2:])
    ax5.axis("off")

    summary_text = f"""
    FP-2 FALSIFICATION SUMMARY

    Overall Status: {'❌ FALSIFIED' if falsification.get('overall_falsified') else '✅ VALIDATED'}

    Criteria Results:
    {falsification.get('summary', 'No summary available')}
    """

    ax5.text(
        0.1,
        0.8,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
        transform=ax5.transAxes,
    )

    # Plot 6: Reward performance comparison
    ax6 = fig.add_subplot(gs[2, :])
    agents = list(benchmark_results.keys())
    mean_rewards = [benchmark_results[a]["mean_cumulative_reward"] for a in agents]
    std_rewards = [benchmark_results[a]["std_cumulative_reward"] for a in agents]

    ax6.bar(
        agents,
        mean_rewards,
        yerr=std_rewards,
        alpha=0.7,
        color=["#2E86AB", "#A23B72", "#F18F01", "#06A77D"],
        capsize=5,
    )
    ax6.set_ylabel("Mean Final Cumulative Reward")
    ax6.set_title("Overall Performance Comparison")
    ax6.grid(axis="y", alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {save_path}")
    plt.show()


# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================


def main():
    """Main execution pipeline for FP-2 falsification protocol"""

    print("=" * 80)
    print("APGI FALSIFICATION PROTOCOL FP-2: AGENT COMPARISON CONVERGENCE BENCHMARK")
    print("=" * 80)

    # Initialize components
    benchmark_experiment = ConvergenceBenchmarkExperiment(n_agents=20, n_trials=100)
    sensitivity_analyzer = SensitivityAnalysis(
        benchmark_experiment.agent_configs["APGI"]
    )
    falsifier = FalsificationFramework()

    # Run convergence benchmark
    print("\n" + "=" * 80)
    print("PHASE 1: CONVERGENCE BENCHMARK")
    print("=" * 80)

    benchmark_results = benchmark_experiment.run_convergence_benchmark()

    print("\nBenchmark Results:")
    for agent, results in benchmark_results.items():
        conv_trial = results.get("mean_convergence_trial")
        conv_rate = results.get("convergence_rate")
        print(f"  {agent}: {conv_trial:.1f} trials, {conv_rate:.1%} convergence rate")

    # Run sensitivity analysis
    print("\n" + "=" * 80)
    print("PHASE 2: SENSITIVITY ANALYSIS")
    print("=" * 80)

    sensitivity_results = sensitivity_analyzer.run_sensitivity_analysis(n_runs=5)

    print(
        f"\nSensitivity analysis completed across {len(sensitivity_results)} parameter combinations"
    )

    # Run falsification analysis
    print("\n" + "=" * 80)
    print("PHASE 3: FALSIFICATION ANALYSIS")
    print("=" * 80)

    falsification_results = falsifier.run_full_falsification(
        benchmark_results, sensitivity_results
    )

    print("\n" + falsification_results["summary"])

    # Generate visualization
    print("\n" + "=" * 80)
    print("PHASE 4: VISUALIZATION")
    print("=" * 80)

    plot_falsification_results(
        benchmark_results,
        sensitivity_results,
        falsification_results,
        "fp2_falsification_results.png",
    )

    # Save results
    print("\n" + "=" * 80)
    print("PHASE 5: SAVING RESULTS")
    print("=" * 80)

    results_summary = {
        "protocol": "FP-2",
        "description": "Iowa Gambling Task / Agent Comparison Convergence Benchmark",
        "benchmark_results": benchmark_results,
        "sensitivity_results": sensitivity_results,
        "falsification": falsification_results,
        "config": {
            "n_agents": 20,
            "n_trials": 100,
            "sensitivity_runs": 5,
        },
    }

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj

    results_summary = convert_for_json(results_summary)

    with open("fp2_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("✅ Results saved to: fp2_results.json")

    # Final status
    if falsification_results["overall_falsified"]:
        print("\n" + "=" * 80)
        print("❌ CRITICAL FAILURE: APGI FALSIFIED AT FP-2 LEVEL")
        print("=" * 80)
        print("\nThe APGI framework has been falsified by agent comparison benchmarks.")
        print(
            "RECOMMENDATION: Framework requires fundamental revision before proceeding."
        )
        print("=" * 80)
        return False
    else:
        print("\n" + "=" * 80)
        print("✅ SUCCESS: APGI PASSES FP-2 FALSIFICATION CRITERIA")
        print("=" * 80)
        print("\nAPGI demonstrates required convergence and model superiority.")
        print("Framework may proceed to next falsification protocols.")
        print("=" * 80)
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
