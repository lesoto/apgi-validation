import numpy as np
import logging
from scipy.optimize import brentq
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.statistical_tests import (
    safe_ttest_ind,
    safe_ttest_1samp,
    compute_cohens_d,
    safe_binomtest,
)

logger = logging.getLogger(__name__)


@dataclass
class FalsificationResult:
    passed: bool
    results: Dict[str, Dict[str, Any]]
    summary: Dict[str, int]


def calculate_hysteresis(theta_t: float, epsilon: float, precision: float) -> float:
    """
    Calculate hysteresis width using root finding (brentq) instead of hardcoding.
    Fixed per Step 1.2 of TODO.md.
    """

    def ignition_equation(th):
        # Simplified attractor equation: Π·|ε| - θ = 0
        return precision * abs(epsilon) - th

    try:
        # Find roots near the threshold
        root_upper = brentq(ignition_equation, 0.01, 1.0)
        # Hysteresis is typically a fraction of the threshold adaptation
        # In the context of APGI, it's the difference between ignition and extinction points
        return abs(root_upper - theta_t)
    except (ValueError, RuntimeError):
        # Fallback to a value derived from the dynamics if root finding fails
        return abs(precision * abs(epsilon) - theta_t)


def check_F1_family(data: dict, thresholds: dict) -> Dict[str, Any]:
    results = {}
    sig_level = thresholds.get("significance_level", 0.01)

    # F1.1: APGI Agent Performance Advantage
    if "apgi_rewards" in data and "pp_rewards" in data:
        apgi_rewards = data["apgi_rewards"]
        pp_rewards = data["pp_rewards"]
        t_stat, p_value, _ = safe_ttest_ind(
            apgi_rewards, pp_rewards, alpha=sig_level, min_n=2
        )
        mean_apgi = np.mean(apgi_rewards)
        mean_pp = np.mean(pp_rewards)
        safe_mean_pp = max(1e-10, abs(mean_pp)) * (1 if mean_pp >= 0 else -1)
        advantage_pct = ((mean_apgi - mean_pp) / safe_mean_pp) * 100
        cohens_d = compute_cohens_d(apgi_rewards, pp_rewards, min_n=2)

        min_advantage = thresholds.get("F1_1_MIN_ADVANTAGE_PCT", 18.0)
        min_cohens_d = thresholds.get("F1_1_MIN_COHENS_D", 0.60)

        f1_1_pass = (
            advantage_pct >= min_advantage
            and cohens_d >= min_cohens_d
            and p_value < sig_level
        )
        results["F1.1"] = {
            "passed": f1_1_pass,
            "advantage_pct": advantage_pct,
            "cohens_d": cohens_d,
            "p_value": p_value,
            "t_statistic": t_stat,
            "threshold": f"≥{min_advantage}% advantage, d ≥ {min_cohens_d}",
            "actual": f"{advantage_pct:.2f}% advantage, d={cohens_d:.3f}",
        }

    # F1.4: Threshold Adaptation Dynamics
    if "threshold_adaptation" in data:
        threshold_array = np.asarray(data["threshold_adaptation"], dtype=float)
        threshold_reduction = float(np.mean(threshold_array))
        t_stat, p_adapt, _ = safe_ttest_1samp(
            threshold_array, popmean=0.0, alpha=sig_level, min_n=2
        )
        adapt_std = float(np.std(threshold_array, ddof=1))
        cohens_d_adapt = threshold_reduction / max(1e-10, adapt_std)

        target_red = thresholds.get(
            "F3_3_MIN_REDUCTION_PCT", 20.0
        )  # F1.4 shares threshold logic with F3.3 in some specs
        min_d = thresholds.get("F3_3_MIN_COHENS_D", 0.70)

        f1_4_pass = (
            threshold_reduction >= target_red
            and cohens_d_adapt >= min_d
            and p_adapt < sig_level
        )
        results["F1.4"] = {
            "passed": f1_4_pass,
            "threshold_reduction_pct": threshold_reduction,
            "cohens_d": cohens_d_adapt,
            "p_value": p_adapt,
            "threshold": f"≥{target_red}% reduction, d ≥ {min_d}",
            "actual": f"{threshold_reduction:.2f}%, d={cohens_d_adapt:.3f}",
        }

    return results


def check_F5_family(
    data: dict, thresholds: dict, genome_data: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Check F5 family criteria (evolutionary emergence).
    Deduplicated from FP-1, FP-2, FP-3 per Step 1.3.
    """
    results = {}

    # F5.1: Threshold Filtering Emergence
    if "threshold_emergence_proportion" in data:
        prop = data["threshold_emergence_proportion"]
        # Require genome_data for evolutionary simulation results
        if not genome_data or "evolved_alpha_values" not in genome_data:
            raise ValueError("genome_data required — run VP-5 first")
        mean_alpha = np.mean(genome_data["evolved_alpha_values"])
        n_agents = len(genome_data["evolved_alpha_values"])

        # Cohen's d vs baseline alpha=3.0
        alpha_std = np.std(genome_data["evolved_alpha_values"]) if genome_data else 0.5
        cohens_d = (mean_alpha - 3.0) / max(1e-10, alpha_std)

        p_val, _ = safe_binomtest(int(prop * n_agents), n_agents, p=0.75)

        min_prop = thresholds.get("F5_1_MIN_PROPORTION", 0.75)
        min_alpha = thresholds.get("F5_1_MIN_ALPHA", 4.0)
        min_d = thresholds.get("F5_1_MIN_COHENS_D", 0.80)

        f5_1_pass = (
            np.isfinite(prop)
            and np.isfinite(mean_alpha)
            and np.isfinite(cohens_d)
            and prop >= min_prop
            and mean_alpha >= min_alpha
            and cohens_d >= min_d
            and p_val < 0.01
        )
        results["F5.1"] = {
            "passed": f5_1_pass,
            "proportion": prop,
            "mean_alpha": mean_alpha,
            "cohens_d": cohens_d,
            "p_value": p_val,
            "threshold": f"≥{min_prop * 100}% prop, α ≥ {min_alpha}, d ≥ {min_d}",
            "actual": f"{prop * 100:.1f}%, α={mean_alpha:.2f}, d={cohens_d:.3f}",
        }

    # F5.2: Precision-Weighted Coding Emergence
    if "precision_emergence_proportion" in data:
        prop = data["precision_emergence_proportion"]
        # Require genome_data for evolutionary simulation results
        if not genome_data or "timescale_correlations" not in genome_data:
            raise ValueError("genome_data required — run VP-5 first")
        mean_r = np.mean(genome_data["timescale_correlations"])
        n_agents = len(genome_data["timescale_correlations"])

        p_val, _ = safe_binomtest(int(prop * n_agents), n_agents, p=0.65)

        min_prop = thresholds.get("F5_2_MIN_PROPORTION", 0.65)
        min_r = thresholds.get("F5_2_MIN_CORRELATION", 0.45)

        f5_2_pass = (
            np.isfinite(prop)
            and np.isfinite(mean_r)
            and prop >= min_prop
            and mean_r >= min_r
            and p_val < 0.01
        )
        results["F5.2"] = {
            "passed": f5_2_pass,
            "proportion": prop,
            "mean_r": mean_r,
            "p_value": p_val,
            "threshold": f"≥{min_prop * 100}% develop weighting, r ≥ {min_r}",
            "actual": f"{prop * 100:.1f}%, r={mean_r:.3f}",
        }

    # F5.3: Interoceptive Prioritization Emergence
    if "intero_gain_ratio_proportion" in data:
        prop = data["intero_gain_ratio_proportion"]
        # Require genome_data for evolutionary simulation results
        if not genome_data or "intero_gain_ratios" not in genome_data:
            raise ValueError("genome_data required — run VP-5 first")
        mean_ratio = np.mean(genome_data["intero_gain_ratios"])
        n_agents = len(genome_data["intero_gain_ratios"])

        # Cohen's d vs baseline ratio=1.0
        ratio_std = np.std(genome_data["intero_gain_ratios"]) if genome_data else 0.5
        cohens_d = (mean_ratio - 1.0) / max(1e-10, ratio_std)

        p_val, _ = safe_binomtest(int(prop * n_agents), n_agents, p=0.55)

        min_prop = thresholds.get("F5_3_MIN_PROPORTION", 0.55)
        min_ratio = thresholds.get("F5_3_MIN_GAIN_RATIO", 1.3)
        min_d = thresholds.get("F5_3_MIN_COHENS_D", 0.60)

        f5_3_pass = (
            np.isfinite(prop)
            and np.isfinite(mean_ratio)
            and np.isfinite(cohens_d)
            and prop >= min_prop
            and mean_ratio >= min_ratio
            and cohens_d >= min_d
            and p_val < 0.01
        )
        results["F5.3"] = {
            "passed": f5_3_pass,
            "proportion": prop,
            "mean_ratio": mean_ratio,
            "cohens_d": cohens_d,
            "p_value": p_val,
            "threshold": f"≥{min_prop * 100}% prioritize, ratio ≥ {min_ratio}, d ≥ {min_d}",
            "actual": f"{prop * 100:.1f}%, ratio={mean_ratio:.2f}, d={cohens_d:.3f}",
        }

    return results


def check_F6_family(data: dict, thresholds: dict) -> Dict[str, Any]:
    results = {}

    # F6.5: Bifurcation Structure
    if "bifurcation_point" in data and "hysteresis_width" in data:
        bp = data["bifurcation_point"]
        hw = data["hysteresis_width"]

        # If hysteresis_width is -1, calculate it (Step 1.2 fix)
        if hw < 0:
            # We need epsilon and precision for this. If not provided, use defaults.
            eps = data.get("epsilon", 0.5)
            prec = data.get("precision", 2.0)
            hw = calculate_hysteresis(bp, eps, prec)

        bp_error_max = thresholds.get("F6_5_BIFURCATION_ERROR_MAX", 0.10)
        h_min = thresholds.get("F6_5_HYSTERESIS_MIN", 0.08)
        h_max = thresholds.get("F6_5_HYSTERESIS_MAX", 0.25)

        # Predicted is usually 0.15 in these protocols
        bp_error = abs(bp - 0.15)

        f6_5_pass = bp_error <= bp_error_max and hw >= h_min and hw <= h_max
        results["F6.5"] = {
            "passed": f6_5_pass,
            "bifurcation_point": bp,
            "bifurcation_error": bp_error,
            "hysteresis_width": hw,
            "threshold": f"Error ≤{bp_error_max}, hyst {h_min}-{h_max}",
            "actual": f"Point {bp:.3f}, error {bp_error:.3f}, hyst {hw:.3f}",
        }
    return results
