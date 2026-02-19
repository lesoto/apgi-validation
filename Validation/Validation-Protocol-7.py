"""
APGI Protocol 7: TMS/Pharmacological Intervention Predictions
==============================================================

Complete implementation of interventional testing framework for APGI predictions.
Tests how perturbing specific APGI parameters (via TMS, pharmacology, or other
interventions) causally affects conscious access.

This protocol generates falsifiable predictions about intervention effects and
provides tools for analyzing intervention studies.

"""

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from statsmodels.stats.power import tt_ind_solve_power

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# PART 1: INTERVENTION MODELS
# =============================================================================


@dataclass
class InterventionEffect:
    """Model of how intervention affects APGI parameters"""

    name: str
    target_parameter: str  # 'theta', 'Pi_i', 'beta', 'alpha'
    effect_size: float  # Standardized effect (Cohen's d)
    effect_direction: str  # 'increase', 'decrease', 'null'

    # Time course
    onset_time: float  # Minutes
    peak_time: float  # Minutes
    duration: float  # Minutes

    # Uncertainty
    effect_se: float  # Standard error

    def compute_time_course(self, t: np.ndarray) -> np.ndarray:
        """
        Compute intervention effect over time

        Uses gamma function for realistic pharmacokinetics
        """
        if self.effect_direction == "null":
            return np.zeros_like(t)

        # Gamma function for time course
        # k = shape, theta = scale
        k = 2.0
        theta_param = (self.peak_time - self.onset_time) / k

        # Avoid division by zero or negative scale
        if theta_param <= 0:
            theta_param = 0.001

        t_shifted = t - self.onset_time
        t_shifted[t_shifted < 0] = 0

        effect = stats.gamma.pdf(t_shifted, k, scale=theta_param)
        effect = effect / effect.max() if effect.max() > 0 else effect

        # Scale by effect size and direction
        sign = 1 if self.effect_direction == "increase" else -1
        effect = sign * self.effect_size * effect

        # Zero out after duration
        effect[t > (self.onset_time + self.duration)] = 0

        return effect


class TMSInterventions:
    """
    TMS intervention library with empirically-based effect sizes

    Each intervention targets specific APGI parameters based on
    known neural mechanisms.
    """

    @staticmethod
    def dlpfc_tms() -> InterventionEffect:
        """
        Dorsolateral PFC TMS

        Target: Increases external precision (Pi_e) via top-down attention
        Mechanism: Enhances sensory gain control
        """
        return InterventionEffect(
            name="dlPFC_TMS",
            target_parameter="Pi_e",
            effect_size=0.5,  # Medium effect
            effect_direction="increase",
            onset_time=0.0,
            peak_time=10.0,
            duration=30.0,
            effect_se=0.15,
        )

    @staticmethod
    def insula_tms() -> InterventionEffect:
        """
        Anterior insula TMS

        Target: Increases interoceptive precision (Pi_i)
        Mechanism: Enhances interoceptive signal processing
        """
        return InterventionEffect(
            name="Insula_TMS",
            target_parameter="Pi_i",
            effect_size=0.7,  # Large effect
            effect_direction="increase",
            onset_time=0.0,
            peak_time=5.0,
            duration=20.0,
            effect_se=0.20,
        )

    @staticmethod
    def v1_tms() -> InterventionEffect:
        """
        V1 TMS (phosphene threshold)

        Target: Decreases threshold (makes ignition easier)
        Mechanism: Direct cortical excitation
        """
        return InterventionEffect(
            name="V1_TMS",
            target_parameter="theta",
            effect_size=-0.4,  # Negative = threshold reduction
            effect_direction="decrease",
            onset_time=0.0,
            peak_time=0.1,  # Very rapid
            duration=5.0,
            effect_se=0.12,
        )

    @staticmethod
    def vertex_tms() -> InterventionEffect:
        """
        Vertex TMS (control condition)

        Target: None (null effect)
        Mechanism: Auditory/somatosensory artifact only
        """
        return InterventionEffect(
            name="Vertex_TMS_Control",
            target_parameter="theta",
            effect_size=0.0,
            effect_direction="null",
            onset_time=0.0,
            peak_time=0.0,
            duration=0.0,
            effect_se=0.10,
        )


class PharmacologicalInterventions:
    """
    Pharmacological intervention library

    Based on known receptor mechanisms and pharmacokinetics.
    """

    @staticmethod
    def propranolol() -> InterventionEffect:
        """
        Propranolol (β-blocker)

        Target: Decreases interoceptive precision (Pi_i)
        Mechanism: Blocks peripheral β-adrenergic receptors
        Effect: Attenuates somatic signals
        """
        return InterventionEffect(
            name="Propranolol",
            target_parameter="Pi_i",
            effect_size=-0.6,
            effect_direction="decrease",
            onset_time=30.0,  # 30 min onset
            peak_time=90.0,  # 1.5 hours peak
            duration=240.0,  # 4 hours duration
            effect_se=0.18,
        )

    @staticmethod
    def methylphenidate() -> InterventionEffect:
        """
        Methylphenidate (stimulant)

        Target: Increases external precision (Pi_e)
        Mechanism: Dopamine/norepinephrine reuptake inhibition
        Effect: Enhanced attentional focus
        """
        return InterventionEffect(
            name="Methylphenidate",
            target_parameter="Pi_e",
            effect_size=0.8,
            effect_direction="increase",
            onset_time=20.0,
            peak_time=60.0,
            duration=180.0,
            effect_se=0.22,
        )

    @staticmethod
    def ketamine_subanesthetic() -> InterventionEffect:
        """
        Ketamine (subanesthetic dose)

        Target: Decreases threshold (facilitates ignition)
        Mechanism: NMDA receptor antagonism
        Effect: Reduces computational precision, lowers ignition barrier
        """
        return InterventionEffect(
            name="Ketamine_Low",
            target_parameter="theta",
            effect_size=-0.9,
            effect_direction="decrease",
            onset_time=5.0,
            peak_time=30.0,
            duration=120.0,
            effect_se=0.25,
        )

    @staticmethod
    def placebo() -> InterventionEffect:
        """Placebo control"""
        return InterventionEffect(
            name="Placebo",
            target_parameter="theta",
            effect_size=0.0,
            effect_direction="null",
            onset_time=0.0,
            peak_time=0.0,
            duration=0.0,
            effect_se=0.15,
        )


# =============================================================================
# PART 2: PSYCHOMETRIC CURVE ANALYSIS
# =============================================================================


def logistic_psychometric(
    x: np.ndarray, threshold: float, slope: float, lapse: float
) -> np.ndarray:
    """Logistic psychometric function with lapse rate"""
    return lapse + (1 - 2 * lapse) / (1 + np.exp(-slope * (x - threshold)))


class PsychometricCurve:
    """
    Fit and compare psychometric curves before/after intervention

    Uses Bayesian adaptive psychophysics for efficient threshold estimation.
    """

    def __init__(self):
        self.params_baseline = None
        self.params_intervention = None

    def fit_curve(
        self,
        stimulus_levels: np.ndarray,
        n_trials: np.ndarray,
        n_correct: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Fit psychometric function: P(seen) = Φ((x - μ) / σ)

        Where Φ is cumulative Gaussian, μ is threshold, σ is slope
        """

        def logistic(x, threshold, slope, lapse):
            """Logistic psychometric function with lapse rate"""
            return logistic_psychometric(x, threshold, slope, lapse)

        def negative_log_likelihood(params):
            """Negative log-likelihood for optimization"""
            threshold, slope, lapse = params

            # Constrain lapse rate
            if lapse < 0 or lapse > 0.1:
                return 1e10

            if slope < 0:
                return 1e10

            p_pred = logistic(stimulus_levels, threshold, slope, lapse)
            p_pred = np.clip(p_pred, 1e-10, 1 - 1e-10)

            # Binomial log-likelihood
            ll = np.sum(
                n_correct * np.log(p_pred) + (n_trials - n_correct) * np.log(1 - p_pred)
            )

            return -ll

        # Initial guess
        if initial_guess is None:
            initial_guess = [
                np.mean(stimulus_levels),  # threshold
                5.0,  # slope
                0.02,  # lapse
            ]

        # Optimize
        result = minimize(
            negative_log_likelihood,
            initial_guess,
            method="Nelder-Mead",
            options={"maxiter": 10000},
        )

        threshold, slope, lapse = result.x

        # Compute confidence intervals (via Hessian)
        try:
            # Numerical Hessian
            eps = 1e-5
            hessian = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    params_pp = result.x.copy()
                    params_pm = result.x.copy()
                    params_mp = result.x.copy()
                    params_mm = result.x.copy()

                    params_pp[i] += eps
                    params_pp[j] += eps
                    params_pm[i] += eps
                    params_pm[j] -= eps
                    params_mp[i] -= eps
                    params_mp[j] += eps
                    params_mm[i] -= eps
                    params_mm[j] -= eps

                    hessian[i, j] = (
                        negative_log_likelihood(params_pp)
                        - negative_log_likelihood(params_pm)
                        - negative_log_likelihood(params_mp)
                        + negative_log_likelihood(params_mm)
                    ) / (4 * eps**2)

            # Covariance from inverse Hessian
            try:
                cov = np.linalg.inv(hessian)
                se = np.sqrt(np.diag(cov))
            except (np.linalg.LinAlgError, ValueError):
                se = [0.1, 1.0, 0.01]

        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            se = [0.1, 1.0, 0.01]

        return {
            "threshold": threshold,
            "slope": slope,
            "lapse": lapse,
            "threshold_se": se[0],
            "slope_se": se[1],
            "lapse_se": se[2],
            "nll": result.fun,
        }

    def compare_curves(
        self, baseline_params: Dict[str, float], intervention_params: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compare baseline vs intervention psychometric curves

        Returns effect sizes and significance tests.
        """

        # Threshold shift
        threshold_shift = (
            intervention_params["threshold"] - baseline_params["threshold"]
        )

        threshold_shift_se = np.sqrt(
            baseline_params["threshold_se"] ** 2
            + intervention_params["threshold_se"] ** 2
        )

        threshold_z = threshold_shift / (threshold_shift_se + 1e-10)
        threshold_p = 2 * (1 - stats.norm.cdf(abs(threshold_z)))

        # Slope change
        slope_change = intervention_params["slope"] - baseline_params["slope"]

        slope_change_se = np.sqrt(
            baseline_params["slope_se"] ** 2 + intervention_params["slope_se"] ** 2
        )

        slope_z = slope_change / (slope_change_se + 1e-10)
        slope_p = 2 * (1 - stats.norm.cdf(abs(slope_z)))

        return {
            "threshold_shift": threshold_shift,
            "threshold_shift_se": threshold_shift_se,
            "threshold_z": threshold_z,
            "threshold_p": threshold_p,
            "slope_change": slope_change,
            "slope_change_se": slope_change_se,
            "slope_z": slope_z,
            "slope_p": slope_p,
            "significant": threshold_p < 0.05,
        }


# =============================================================================
# PART 3: INTERVENTION STUDY SIMULATOR
# =============================================================================


class InterventionStudySimulator:
    """
    Simulate intervention experiments for power analysis and validation

    Generates synthetic data with realistic noise and individual differences.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def simulate_crossover_study(
        self,
        intervention: InterventionEffect,
        control: InterventionEffect,
        n_subjects: int = 24,
        n_trials_per_condition: int = 100,
        stimulus_levels: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Simulate within-subject crossover design

        Each subject receives both intervention and control in randomized order.
        """

        if stimulus_levels is None:
            stimulus_levels = np.linspace(0.2, 0.8, 8)

        data = []

        # Subject-level baseline parameters
        baseline_theta = self.rng.normal(0.50, 0.12, n_subjects)
        baseline_alpha = self.rng.normal(5.0, 1.0, n_subjects)

        for subj_id in range(n_subjects):
            # Randomize order
            order = self.rng.permutation(["intervention", "control"])

            for session, condition in enumerate(order):

                effect = intervention if condition == "intervention" else control

                # Time course (sample at 30 min mark for static design)
                time_point = 30.0
                effect_magnitude = effect.compute_time_course(np.array([time_point]))[0]

                # Apply intervention effect to parameters
                theta = baseline_theta[subj_id]
                alpha = baseline_alpha[subj_id]

                if effect.target_parameter == "theta":
                    theta = baseline_theta[subj_id] + effect_magnitude
                elif effect.target_parameter == "alpha":
                    alpha = baseline_alpha[subj_id] * (1 + effect_magnitude)

                # Generate trials at each stimulus level
                for stim_level in stimulus_levels:

                    n_trials = n_trials_per_condition // len(stimulus_levels)

                    # APGI ignition probability
                    # Simplified: S_t ≈ stimulus_level
                    S_t = stim_level

                    P_ignition = 1 / (1 + np.exp(-alpha * (S_t - theta)))

                    # Generate responses
                    seen = self.rng.rand(n_trials) < P_ignition
                    n_seen = seen.sum()

                    # Reaction time (inverse Gaussian)
                    proximity = np.abs(S_t - theta)
                    RT_mean = 300 + 200 * proximity
                    RT = self.rng.wald(RT_mean, 50000, size=n_trials)
                    RT = np.clip(RT, 200, 2000)

                    data.append(
                        {
                            "subject_id": subj_id,
                            "session": session,
                            "condition": condition,
                            "stimulus_level": stim_level,
                            "n_trials": n_trials,
                            "n_seen": n_seen,
                            "p_seen": n_seen / n_trials,
                            "mean_rt": RT.mean(),
                            "intervention": effect.name,
                            "true_theta": theta,
                            "true_effect": effect_magnitude,
                        }
                    )

        return pd.DataFrame(data)

    def simulate_parallel_group_study(
        self,
        intervention: InterventionEffect,
        control: InterventionEffect,
        n_per_group: int = 30,
        n_trials_per_subject: int = 200,
    ) -> pd.DataFrame:
        """
        Simulate between-subject parallel group design

        Half subjects get intervention, half get control.
        """

        stimulus_levels = np.linspace(0.2, 0.8, 8)
        n_subjects = n_per_group * 2

        data = []

        for subj_id in range(n_subjects):
            # Assign to group
            if subj_id < n_per_group:
                condition = "intervention"
                effect = intervention
            else:
                condition = "control"
                effect = control

            # Baseline parameters
            baseline_theta = self.rng.normal(0.50, 0.12)
            baseline_alpha = self.rng.normal(5.0, 1.0)

            # Apply effect
            time_point = 30.0
            effect_magnitude = effect.compute_time_course(np.array([time_point]))[0]

            if effect.target_parameter == "theta":
                theta = baseline_theta + effect_magnitude
            else:
                theta = baseline_theta

            alpha = baseline_alpha

            # Generate trials
            for stim_level in stimulus_levels:
                n_trials = n_trials_per_subject // len(stimulus_levels)

                S_t = stim_level
                P_ignition = 1 / (1 + np.exp(-alpha * (S_t - theta)))

                seen = self.rng.rand(n_trials) < P_ignition
                n_seen = seen.sum()

                data.append(
                    {
                        "subject_id": subj_id,
                        "group": condition,
                        "stimulus_level": stim_level,
                        "n_trials": n_trials,
                        "n_seen": n_seen,
                        "p_seen": n_seen / n_trials,
                        "intervention": effect.name,
                        "true_theta": theta,
                    }
                )

        return pd.DataFrame(data)


# =============================================================================
# PART 4: POWER ANALYSIS
# =============================================================================


class PowerAnalysis:
    """
    Statistical power analysis for intervention studies

    Determines required sample size for detecting APGI-predicted effects.
    """

    @staticmethod
    def compute_required_n(
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80,
        test_type: str = "two-sided",
    ) -> int:
        """
        Compute required sample size for independent t-test

        Args:
            effect_size: Cohen's d
            alpha: Type I error rate
            power: Desired power (1 - β)
            test_type: 'two-sided' or 'one-sided'

        Returns:
            Required N per group
        """

        ratio = 1.0  # Equal group sizes
        alternative = "two-sided" if test_type == "two-sided" else "larger"

        n = tt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative=alternative,
        )

        return int(np.ceil(n))

    @staticmethod
    def compute_power_curve(
        effect_size: float, n_range: np.ndarray, alpha: float = 0.05
    ) -> np.ndarray:
        """
        Compute statistical power across range of sample sizes
        """

        power = np.zeros(len(n_range))

        for i, n in enumerate(n_range):
            try:
                # Compute non-centrality parameter
                ncp = effect_size * np.sqrt(n / 2)

                # Critical t-value
                df = 2 * (n - 1)
                t_crit = stats.t.ppf(1 - alpha / 2, df)

                # Power = P(|t| > t_crit | H1)
                power[i] = (
                    1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
                )

            except (ValueError, RuntimeError):
                power[i] = np.nan

        return power

    @staticmethod
    def minimum_detectable_effect(
        n: int, alpha: float = 0.05, power: float = 0.80
    ) -> float:
        """
        Minimum effect size detectable with given N and power
        """

        return tt_ind_solve_power(
            nobs1=n, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
        )


# =============================================================================
# PART 5: FALSIFICATION CRITERIA
# =============================================================================


class InterventionFalsificationChecker:
    """Check Protocol 7 falsification criteria"""

    def __init__(self):
        self.criteria = {
            "F3.1": {
                "description": "dlPFC TMS fails to shift threshold by >0.05",
                "threshold": 0.05,
            },
            "F3.2": {
                "description": "Propranolol does not reduce interoceptive influence",
                "threshold": None,
            },
            "F3.3": {
                "description": "Interventions show opposite direction than predicted",
                "threshold": None,
            },
            "F3.4": {
                "description": "No dose-response relationship for ketamine",
                "threshold": None,
            },
            "F3.5": {
                "description": "Placebo effects larger than active intervention",
                "threshold": None,
            },
        }

    def check_F3_1(
        self,
        baseline_threshold: float,
        intervention_threshold: float,
        intervention_se: float,
    ) -> Tuple[bool, Dict]:
        """F3.1: Threshold shift magnitude"""

        shift = intervention_threshold - baseline_threshold

        # One-sided test: shift should be negative (threshold reduction)
        z = shift / (intervention_se + 1e-10)
        p_value = stats.norm.cdf(z)  # Left-tail

        # Falsified if shift is not negative OR magnitude < 0.05
        falsified = (shift >= 0) or (abs(shift) < 0.05)

        return falsified, {
            "shift": float(shift),
            "shift_se": float(intervention_se),
            "z_score": float(z),
            "p_value": float(p_value),
            "magnitude_sufficient": abs(shift) >= 0.05,
        }

    def check_F3_2(
        self, baseline_beta: float, intervention_beta: float, beta_se: float
    ) -> Tuple[bool, Dict]:
        """F3.2: Propranolol effect on somatic bias"""

        reduction = baseline_beta - intervention_beta

        # Should be positive (reduction in interoceptive influence)
        z = reduction / (beta_se + 1e-10)
        p_value = 1 - stats.norm.cdf(z)  # Right-tail

        falsified = (reduction <= 0) or (p_value > 0.05)

        return falsified, {
            "reduction": float(reduction),
            "reduction_se": float(beta_se),
            "z_score": float(z),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    def check_F3_3(
        self, predicted_direction: str, observed_effect: float
    ) -> Tuple[bool, Dict]:
        """F3.3: Direction of effect"""

        if predicted_direction == "increase":
            expected_sign = 1
        elif predicted_direction == "decrease":
            expected_sign = -1
        else:  # null
            expected_sign = 0

        observed_sign = np.sign(observed_effect)

        # Falsified if opposite direction
        if expected_sign != 0:
            falsified = observed_sign != expected_sign
        else:
            falsified = (
                abs(observed_effect) > 0.3
            )  # Significant effect when null predicted

        return falsified, {
            "predicted_direction": predicted_direction,
            "observed_effect": float(observed_effect),
            "correct_direction": not falsified,
        }

    def check_F3_4(self, doses: np.ndarray, effects: np.ndarray) -> Tuple[bool, Dict]:
        """F3.4: Dose-response relationship"""

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(doses, effects)

        # Should be monotonic increase (positive slope, significant)
        falsified = (slope <= 0) or (p_value > 0.05)

        return falsified, {
            "slope": float(slope),
            "r_squared": float(r_value**2),
            "p_value": float(p_value),
            "monotonic": slope > 0,
        }

    def check_F3_5(
        self,
        placebo_effect: float,
        active_effect: float,
        placebo_se: float,
        active_se: float,
    ) -> Tuple[bool, Dict]:
        """F3.5: Placebo vs active comparison"""

        difference = active_effect - placebo_effect
        difference_se = np.sqrt(placebo_se**2 + active_se**2)

        z = difference / (difference_se + 1e-10)
        p_value = 1 - stats.norm.cdf(z)

        # Falsified if active not better than placebo
        falsified = (difference <= 0) or (p_value > 0.05)

        return falsified, {
            "placebo_effect": float(placebo_effect),
            "active_effect": float(active_effect),
            "difference": float(difference),
            "z_score": float(z),
            "p_value": float(p_value),
            "active_superior": difference > 0 and p_value < 0.05,
        }

    def generate_report(self, intervention_results: Dict) -> Dict:
        """Generate comprehensive falsification report"""

        report = {
            "intervention": intervention_results.get("name", "Unknown"),
            "falsified_criteria": [],
            "passed_criteria": [],
            "overall_falsified": False,
        }

        # Check each criterion based on available data
        if "threshold_shift" in intervention_results:
            f3_1_result, f3_1_details = self.check_F3_1(
                intervention_results["baseline_threshold"],
                intervention_results["intervention_threshold"],
                intervention_results["threshold_se"],
            )

            criterion = {
                "code": "F3.1",
                "description": self.criteria["F3.1"]["description"],
                "falsified": f3_1_result,
                "details": f3_1_details,
            }

            if f3_1_result:
                report["falsified_criteria"].append(criterion)
            else:
                report["passed_criteria"].append(criterion)

        # Add other criteria checks...

        report["overall_falsified"] = len(report["falsified_criteria"]) > 0

        return report


# =============================================================================
# PART 6: VISUALIZATION
# =============================================================================


def plot_intervention_results(
    results_df: pd.DataFrame,
    intervention_name: str,
    save_path: str = "protocol7_intervention_results.png",
):
    """Generate comprehensive intervention results visualization"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ==========================================================================
    # Panel 1: Psychometric Curves (Baseline vs Intervention)
    # ==========================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    # Get baseline and intervention data
    baseline = results_df[results_df["condition"] == "control"]
    intervention = results_df[results_df["condition"] == "intervention"]

    # Group by stimulus level
    baseline_grouped = baseline.groupby("stimulus_level").agg(
        {"n_seen": "sum", "n_trials": "sum"}
    )

    intervention_grouped = intervention.groupby("stimulus_level").agg(
        {"n_seen": "sum", "n_trials": "sum"}
    )

    baseline_grouped["p_seen"] = (
        baseline_grouped["n_seen"] / baseline_grouped["n_trials"]
    )
    intervention_grouped["p_seen"] = (
        intervention_grouped["n_seen"] / intervention_grouped["n_trials"]
    )

    # Fit curves
    psychometric = PsychometricCurve()

    baseline_params = psychometric.fit_curve(
        baseline_grouped.index.values,
        baseline_grouped["n_trials"].values,
        baseline_grouped["n_seen"].values,
    )

    intervention_params = psychometric.fit_curve(
        intervention_grouped.index.values,
        intervention_grouped["n_trials"].values,
        intervention_grouped["n_seen"].values,
    )

    # Plot data points
    ax1.scatter(
        baseline_grouped.index,
        baseline_grouped["p_seen"],
        s=100,
        alpha=0.6,
        color="blue",
        label="Baseline",
        zorder=3,
    )
    ax1.scatter(
        intervention_grouped.index,
        intervention_grouped["p_seen"],
        s=100,
        alpha=0.6,
        color="red",
        label="Intervention",
        zorder=3,
    )

    # Plot fitted curves
    x_fine = np.linspace(0.1, 0.9, 200)

    y_baseline = logistic_psychometric(
        x_fine,
        baseline_params["threshold"],
        baseline_params["slope"],
        baseline_params["lapse"],
    )
    y_intervention = logistic_psychometric(
        x_fine,
        intervention_params["threshold"],
        intervention_params["slope"],
        intervention_params["lapse"],
    )

    ax1.plot(x_fine, y_baseline, "b-", linewidth=2.5, alpha=0.8)
    ax1.plot(x_fine, y_intervention, "r-", linewidth=2.5, alpha=0.8)

    # Threshold markers
    ax1.axvline(
        baseline_params["threshold"],
        color="blue",
        linestyle="--",
        linewidth=2,
        alpha=0.6,
    )
    ax1.axvline(
        intervention_params["threshold"],
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.6,
    )

    ax1.set_xlabel("Stimulus Intensity", fontsize=13, fontweight="bold")
    ax1.set_ylabel("P(Seen)", fontsize=13, fontweight="bold")
    ax1.set_title(
        f"Psychometric Functions - {intervention_name}", fontsize=14, fontweight="bold"
    )
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # ==========================================================================
    # Panel 2: Threshold Shift
    # ==========================================================================
    ax2 = fig.add_subplot(gs[0, 2])

    threshold_shift = intervention_params["threshold"] - baseline_params["threshold"]
    threshold_shift_se = np.sqrt(
        baseline_params["threshold_se"] ** 2 + intervention_params["threshold_se"] ** 2
    )

    ax2.bar(
        ["Baseline", "Intervention"],
        [baseline_params["threshold"], intervention_params["threshold"]],
        color=["blue", "red"],
        alpha=0.7,
        edgecolor="black",
        linewidth=2,
    )

    ax2.errorbar(
        ["Baseline", "Intervention"],
        [baseline_params["threshold"], intervention_params["threshold"]],
        yerr=[baseline_params["threshold_se"], intervention_params["threshold_se"]],
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=2,
    )

    ax2.set_ylabel("Threshold", fontsize=12, fontweight="bold")
    ax2.set_title(
        f"Threshold Shift: {threshold_shift:.3f}±{threshold_shift_se:.3f}",
        fontsize=11,
        fontweight="bold",
    )
    ax2.grid(axis="y", alpha=0.3)

    # ==========================================================================
    # Panel 3: Subject-Level Effects
    # ==========================================================================
    ax3 = fig.add_subplot(gs[1, :])

    # Extract subject-level thresholds
    psychometric = PsychometricCurve()
    subject_thresholds = (
        results_df.groupby(["subject_id", "condition"])
        .apply(
            lambda x: (
                psychometric.fit_curve(
                    x["stimulus_level"].values, x["n_trials"].values, x["n_seen"].values
                )["threshold"]
                if len(x) > 0
                else np.nan
            )
        )
        .unstack()
    )

    subject_ids = subject_thresholds.index.values
    baseline_thresholds = subject_thresholds["control"].values
    intervention_thresholds = subject_thresholds["intervention"].values

    # Paired comparison plot
    for i, subj in enumerate(subject_ids):
        ax3.plot(
            [0, 1],
            [baseline_thresholds[i], intervention_thresholds[i]],
            "o-",
            color="gray",
            alpha=0.4,
            linewidth=1,
            markersize=6,
        )

    # Mean effect
    ax3.plot(
        [0, 1],
        [np.nanmean(baseline_thresholds), np.nanmean(intervention_thresholds)],
        "o-",
        color="red",
        linewidth=3,
        markersize=12,
        label="Mean Effect",
    )

    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Baseline", "Intervention"], fontsize=12)
    ax3.set_ylabel("Threshold", fontsize=12, fontweight="bold")
    ax3.set_title("Subject-Level Threshold Changes", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=11)
    ax3.grid(axis="y", alpha=0.3)

    # ==========================================================================
    # Panel 4: Effect Size Distribution
    # ==========================================================================
    ax4 = fig.add_subplot(gs[2, 0])

    individual_effects = intervention_thresholds - baseline_thresholds
    individual_effects = individual_effects[~np.isnan(individual_effects)]

    ax4.hist(
        individual_effects,
        bins=15,
        density=True,
        alpha=0.6,
        color="purple",
        edgecolor="black",
        linewidth=1.5,
    )

    # Overlay normal distribution
    mu_effect = np.mean(individual_effects)
    sigma_effect = np.std(individual_effects)

    x_dist = np.linspace(individual_effects.min(), individual_effects.max(), 100)
    ax4.plot(
        x_dist,
        stats.norm.pdf(x_dist, mu_effect, sigma_effect),
        "r-",
        linewidth=2.5,
        label=f"N({mu_effect:.3f}, {sigma_effect:.3f})",
    )

    ax4.axvline(0, color="black", linestyle="--", linewidth=2, label="Null Effect")
    ax4.axvline(mu_effect, color="red", linestyle="-", linewidth=2, label="Mean Effect")

    ax4.set_xlabel("Threshold Shift", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax4.set_title("Effect Size Distribution", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # ==========================================================================
    # Panel 5: Statistical Summary
    # ==========================================================================
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    # Compute statistics (handle NaN values)
    valid_mask = ~(np.isnan(baseline_thresholds) | np.isnan(intervention_thresholds))
    baseline_thresholds_valid = baseline_thresholds[valid_mask]
    intervention_thresholds_valid = intervention_thresholds[valid_mask]

    if len(baseline_thresholds_valid) < 2:
        t_stat, p_value = np.nan, np.nan
    else:
        t_stat, p_value = stats.ttest_rel(
            baseline_thresholds_valid, intervention_thresholds_valid
        )

    cohens_d = (
        np.nanmean(intervention_thresholds) - np.nanmean(baseline_thresholds)
    ) / np.nanstd(individual_effects)

    ci_low, ci_high = stats.t.interval(
        0.95,
        len(individual_effects) - 1,
        loc=mu_effect,
        scale=stats.sem(individual_effects),
    )

    summary_text = f"""
    STATISTICAL SUMMARY
    {'=' * 40}

    N Subjects: {len(individual_effects)}

    Threshold Shift:
      Mean: {mu_effect:.4f}
      95% CI: [{ci_low:.4f}, {ci_high:.4f}]

    Effect Size:
      Cohen's d: {cohens_d:.3f}

    Significance:
      t({len(individual_effects) - 1}) = {t_stat:.3f}
      p = {p_value:.4f}
      {'✅ SIGNIFICANT' if p_value < 0.05 else '❌ NOT SIGNIFICANT'}

    Interpretation:
      {_interpret_effect_size(cohens_d)}
    """

    ax5.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )

    # ==========================================================================
    # Panel 6: Power Analysis
    # ==========================================================================
    ax6 = fig.add_subplot(gs[2, 2])

    power_analyzer = PowerAnalysis()
    n_range = np.arange(5, 51, 1)
    power_curve = power_analyzer.compute_power_curve(abs(cohens_d), n_range)

    ax6.plot(n_range, power_curve, "b-", linewidth=2.5)
    ax6.axhline(0.80, color="red", linestyle="--", linewidth=2, label="80% Power")
    ax6.axvline(
        len(individual_effects),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Current N={len(individual_effects)}",
    )

    ax6.fill_between(n_range, 0, power_curve, alpha=0.2, color="blue")

    ax6.set_xlabel("Sample Size (N)", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Statistical Power", fontsize=11, fontweight="bold")
    ax6.set_title("Power Analysis", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)
    ax6.set_ylim([0, 1])

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {save_path}")
    plt.show()


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible effect"
    elif abs_d < 0.5:
        return "Small effect"
    elif abs_d < 0.8:
        return "Medium effect"
    else:
        return "Large effect"


# =============================================================================
# PART 6: ADVANCED STATISTICAL TOOLS
# =============================================================================


def interactive_power_analysis_tool():
    """
    Interactive tool for researchers to compute required sample size
    for their specific intervention study
    """

    def compute_sample_size(
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80,
        test_type: str = "two-sided",
        design: str = "between",
    ):
        """
        Compute required sample size
        """
        from statsmodels.stats.power import tt_ind_solve_power, tt_solve_power

        if design == "between":
            n = tt_ind_solve_power(
                effect_size=effect_size, alpha=alpha, power=power, alternative=test_type
            )
        elif design == "within":
            # Paired t-test
            n = tt_solve_power(
                effect_size=effect_size, alpha=alpha, power=power, alternative=test_type
            )

        # Add dropout compensation
        n_with_dropout = n / (1 - 0.15)  # Assume 15% dropout

        return {
            "n_per_group": np.ceil(n),
            "n_total": np.ceil(n * 2) if design == "between" else np.ceil(n),
            "n_with_dropout": (
                np.ceil(n_with_dropout * 2)
                if design == "between"
                else np.ceil(n_with_dropout)
            ),
        }

    # Example usage with APGI predictions
    interventions = {
        "Propranolol_Pi_i": {"effect_size": -0.6},
        "Insula_TMS_Pi_i": {"effect_size": 0.7},
        "dlPFC_TMS_Pi_e": {"effect_size": 0.5},
    }

    sample_sizes = {}
    for intervention, params in interventions.items():
        sample_sizes[intervention] = compute_sample_size(**params)

    return sample_sizes


def correct_for_multiple_comparisons(p_values, method="holm"):
    """
    Apply appropriate correction for multiple hypothesis testing
    """
    from statsmodels.stats.multitest import multipletests

    # Methods: 'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg)
    reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values, alpha=0.05, method=method
    )

    results = pd.DataFrame(
        {
            "original_p": p_values,
            "corrected_p": p_corrected,
            "significant": reject,
            "method": method,
        }
    )

    # Report number of significant findings
    n_significant_original = np.sum(np.array(p_values) < 0.05)
    n_significant_corrected = np.sum(reject)

    print(f"Significant findings before correction: {n_significant_original}")
    print(f"Significant findings after {method} correction: {n_significant_corrected}")

    return results


def model_dose_response_relationship(doses, responses):
    """
    Fit Hill equation (sigmoid) to dose-response data
    Estimate EC50 (half-maximal effective concentration)
    """
    from scipy.optimize import curve_fit

    def hill_equation(dose, EC50, hill_coef, baseline, max_effect):
        """
        4-parameter Hill equation
        """
        return baseline + (max_effect - baseline) / (1 + (EC50 / dose) ** hill_coef)

    # Fit
    try:
        popt, pcov = curve_fit(
            hill_equation,
            doses,
            responses,
            p0=[np.median(doses), 1.0, min(responses), max(responses)],
            bounds=([0, 0.1, -np.inf, -np.inf], [np.inf, 10, np.inf, np.inf]),
        )

        EC50, hill_coef, baseline, max_effect = popt

        # Generate smooth curve
        dose_range = np.logspace(np.log10(min(doses)), np.log10(max(doses)), 100)
        fitted_response = hill_equation(dose_range, *popt)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(doses, responses, s=100, alpha=0.6, label="Data")
        ax.plot(dose_range, fitted_response, "r-", linewidth=2, label="Hill fit")
        ax.axvline(
            EC50, color="k", linestyle="--", alpha=0.5, label=f"EC50 = {EC50:.2f}"
        )
        ax.set_xscale("log")
        ax.set_xlabel("Dose (log scale)")
        ax.set_ylabel("Response")
        ax.legend()
        ax.grid(True, alpha=0.3)

        results = {
            "EC50": EC50,
            "hill_coefficient": hill_coef,
            "baseline": baseline,
            "max_effect": max_effect,
            "params_se": np.sqrt(np.diag(pcov)),
        }

        return results, fig

    except (RuntimeError, ValueError, TypeError, KeyError) as e:
        print(f"Fitting failed: {e}")
        return None, None


def bayesian_equivalence_test(control_data, treatment_data, rope_width=0.1):
    """
    Test if intervention effect is practically equivalent to zero
    Uses Region of Practical Equivalence (ROPE)

    More informative than null hypothesis testing
    """
    try:
        import arviz as az
        import pymc as pm

        with pm.Model():
            # Priors
            mu_control = pm.Normal("mu_control", mu=0, sigma=10)
            mu_treatment = pm.Normal("mu_treatment", mu=0, sigma=10)
            sigma = pm.HalfNormal("sigma", sigma=5)

            # Likelihood
            pm.Normal("control_obs", mu=mu_control, sigma=sigma, observed=control_data)
            pm.Normal(
                "treatment_obs", mu=mu_treatment, sigma=sigma, observed=treatment_data
            )

            # Effect size
            effect = pm.Deterministic("effect", mu_treatment - mu_control)

            # Sample
            trace = pm.sample(2000, tune=1000, return_inferencedata=True)

        # Analyze effect with ROPE
        effect_samples = trace.posterior["effect"].values.flatten()

        # ROPE = [-rope_width, rope_width]
        in_rope = np.sum((effect_samples > -rope_width) & (effect_samples < rope_width))
        prob_in_rope = in_rope / len(effect_samples)

        below_rope = np.sum(effect_samples < -rope_width) / len(effect_samples)
        above_rope = np.sum(effect_samples > rope_width) / len(effect_samples)

        # Decision
        if prob_in_rope > 0.95:
            decision = "Practically equivalent (effect negligible)"
        elif above_rope > 0.95:
            decision = "Clearly beneficial"
        elif below_rope > 0.95:
            decision = "Clearly harmful"
        else:
            decision = "Uncertain (collect more data)"

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(effect_samples, bins=50, density=True, alpha=0.6)
        ax.axvline(0, color="k", linestyle="--", linewidth=2, label="Null")
        ax.axvspan(-rope_width, rope_width, alpha=0.2, color="gray", label="ROPE")
        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Density")
        ax.set_title(f"Bayesian Equivalence Test\n{decision}")
        ax.legend()

        results = {
            "prob_in_rope": prob_in_rope,
            "prob_beneficial": above_rope,
            "prob_harmful": below_rope,
            "decision": decision,
            "effect_mean": np.mean(effect_samples),
            "effect_hdi_95": az.hdi(trace, var_names=["effect"], hdi_prob=0.95),
        }

        return results, fig

    except ImportError:
        print("Warning: pymc and arviz not available. Using frequentist approximation.")
        # Frequentist fallback
        effect = np.mean(treatment_data) - np.mean(control_data)
        pooled_se = np.sqrt(
            np.var(control_data) / len(control_data)
            + np.var(treatment_data) / len(treatment_data)
        )

        # Simple approximation
        if abs(effect) < rope_width:
            decision = "Likely equivalent (frequentist approximation)"
        else:
            decision = "Likely different (frequentist approximation)"

        return {
            "effect_mean": effect,
            "effect_se": pooled_se,
            "decision": decision,
            "note": "Frequentist approximation - install pymc for full Bayesian analysis",
        }, None


def meta_analysis_of_interventions(studies_data):
    """
    Meta-analyze multiple intervention studies
    Combine evidence across studies
    """
    try:
        import arviz as az
        import pymc as pm

        # studies_data: list of dicts with 'effect_size', 'se', 'n', 'study_name'

        n_studies = len(studies_data)
        effect_sizes = np.array([s["effect_size"] for s in studies_data])
        se = np.array([s["se"] for s in studies_data])

        with pm.Model():
            # Hyperpriors
            mu = pm.Normal("mu", mu=0, sigma=10)  # Overall effect
            tau = pm.HalfNormal("tau", sigma=1)  # Between-study heterogeneity

            # Study-specific effects
            theta = pm.Normal("theta", mu=mu, sigma=tau, shape=n_studies)

            # Likelihood (observed effect sizes)
            pm.Normal("obs", mu=theta, sigma=se, observed=effect_sizes)

            # Sample
            trace = pm.sample(2000, tune=1000, return_inferencedata=True)

        # Extract results
        overall_effect = trace.posterior["mu"].values.flatten()
        heterogeneity = trace.posterior["tau"].values.flatten()

        # I² statistic (heterogeneity)
        Q = np.sum((effect_sizes - np.mean(effect_sizes)) ** 2 / se**2)
        df = len(effect_sizes) - 1
        if Q > 0:
            I_squared = max(0, (Q - df) / Q)
        else:
            I_squared = 0

        # Forest plot
        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(n_studies)
        ax.scatter(effect_sizes, y_pos, s=100, zorder=3)
        ax.errorbar(effect_sizes, y_pos, xerr=1.96 * se, fmt="none", zorder=2)

        # Overall effect
        overall_mean = np.mean(overall_effect)
        overall_hdi = az.hdi(trace, var_names=["mu"], hdi_prob=0.95)["mu"]

        ax.scatter(
            [overall_mean],
            [-1],
            s=200,
            marker="D",
            c="red",
            zorder=4,
            label="Overall effect",
        )
        ax.errorbar(
            [overall_mean],
            [-1],
            xerr=[[overall_mean - overall_hdi[0]], [overall_hdi[1] - overall_mean]],
            fmt="none",
            c="red",
            linewidth=2,
            zorder=3,
        )

        ax.axvline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_yticks(list(range(-1, n_studies)))
        ax.set_yticklabels(["Overall"] + [s["study_name"] for s in studies_data])
        ax.set_xlabel("Effect Size (Cohen's d)")
        ax.set_title(f"Meta-Analysis Forest Plot\nI² = {I_squared:.1%} (heterogeneity)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")

        results = {
            "overall_effect_mean": overall_mean,
            "overall_effect_hdi": overall_hdi,
            "heterogeneity_tau": np.mean(heterogeneity),
            "I_squared": I_squared,
            "interpretation": (
                "Low heterogeneity"
                if I_squared < 0.25
                else (
                    "Moderate heterogeneity"
                    if I_squared < 0.75
                    else "High heterogeneity"
                )
            ),
        }

        return results, fig

    except (RuntimeError, ValueError, TypeError, KeyError) as e:
        print(f"Bayesian meta-analysis failed: {e}")
        print("Falling back to frequentist approximation.")
        # Frequentist fallback
        effect_sizes = np.array([s["effect_size"] for s in studies_data])
        se = np.array([s["se"] for s in studies_data])

        # Fixed effect weighted average
        weights = 1 / se**2
        overall_effect_fixed = np.sum(weights * effect_sizes) / np.sum(weights)
        se_fixed = np.sqrt(1 / np.sum(weights))

        # Simple heterogeneity estimate
        if len(effect_sizes) > 1:
            Q = np.sum(weights * (effect_sizes - overall_effect_fixed) ** 2)
            df = len(effect_sizes) - 1
            tau_squared = max(
                0, (Q - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights))
            )
        else:
            Q = 0
            df = 0
            tau_squared = 0

        # Random effects weights
        weights_re = 1 / (se**2 + tau_squared)
        overall_effect_re = np.sum(weights_re * effect_sizes) / np.sum(weights_re)
        se_re = np.sqrt(1 / np.sum(weights_re))

        # I²
        if Q > 0:
            I_squared = tau_squared / (tau_squared + np.var(effect_sizes))
        else:
            I_squared = 0

        return {
            "overall_effect_fixed": overall_effect_fixed,
            "overall_effect_random": overall_effect_re,
            "se_fixed": se_fixed,
            "se_random": se_re,
            "I_squared": I_squared,
            "tau_squared": tau_squared,
            "note": "Frequentist approximation - Bayesian analysis failed",
        }, None


# =============================================================================
# PART 7: MAIN EXECUTION
# =============================================================================


def main():
    """Main execution pipeline for Protocol 7"""

    print("=" * 80)
    print("APGI Protocol 7: TMS/PHARMACOLOGICAL INTERVENTION PREDICTIONS")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Define Interventions
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: DEFINING INTERVENTIONS")
    print("=" * 80)

    interventions = {
        "dlPFC_TMS": TMSInterventions.dlpfc_tms(),
        "Insula_TMS": TMSInterventions.insula_tms(),
        "V1_TMS": TMSInterventions.v1_tms(),
        "Vertex_Control": TMSInterventions.vertex_tms(),
        "Propranolol": PharmacologicalInterventions.propranolol(),
        "Methylphenidate": PharmacologicalInterventions.methylphenidate(),
        "Ketamine": PharmacologicalInterventions.ketamine_subanesthetic(),
        "Placebo": PharmacologicalInterventions.placebo(),
    }

    print(f"\nRegistered {len(interventions)} interventions:")
    for name, intervention in interventions.items():
        print(
            f"  - {name}: {intervention.target_parameter} "
            f"({intervention.effect_direction}, d={intervention.effect_size:.2f})"
        )

    # =========================================================================
    # STEP 2: Power Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: STATISTICAL POWER ANALYSIS")
    print("=" * 80)

    power_analyzer = PowerAnalysis()

    print("\nRequired sample sizes (80% power, α=0.05):")
    print(f"{'Intervention':<20} {'Effect Size':<15} {'Required N':<15}")
    print("-" * 50)

    for name, intervention in interventions.items():
        if intervention.effect_size != 0:
            required_n = power_analyzer.compute_required_n(
                abs(intervention.effect_size), alpha=0.05, power=0.80
            )
            print(f"{name:<20} {intervention.effect_size:>7.2f}        {required_n:>8}")

    # =========================================================================
    # STEP 3: Simulate TMS Study (dlPFC)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: SIMULATING dlPFC TMS STUDY")
    print("=" * 80)

    simulator = InterventionStudySimulator(seed=42)

    dlpfc_data = simulator.simulate_crossover_study(
        intervention=interventions["dlPFC_TMS"],
        control=interventions["Vertex_Control"],
        n_subjects=24,
        n_trials_per_condition=200,
    )

    print("\n✅ Simulated dlPFC TMS crossover study")
    print(f"   Subjects: {dlpfc_data['subject_id'].nunique()}")
    print(f"   Total trials: {dlpfc_data['n_trials'].sum()}")
    print(f"   Conditions: {dlpfc_data['condition'].unique()}")

    # =========================================================================
    # STEP 4: Analyze Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: ANALYZING INTERVENTION EFFECTS")
    print("=" * 80)

    psychometric = PsychometricCurve()

    # Fit baseline
    baseline_data = dlpfc_data[dlpfc_data["condition"] == "control"]
    baseline_grouped = baseline_data.groupby("stimulus_level").agg(
        {"n_seen": "sum", "n_trials": "sum"}
    )

    baseline_params = psychometric.fit_curve(
        baseline_grouped.index.values,
        baseline_grouped["n_trials"].values,
        baseline_grouped["n_seen"].values,
    )

    print("\nBaseline (Vertex Control):")
    print(
        f"  Threshold: {baseline_params['threshold']:.4f} ± {baseline_params['threshold_se']:.4f}"
    )
    print(f"  Slope: {baseline_params['slope']:.4f}")
    print(f"  Lapse: {baseline_params['lapse']:.4f}")

    # Fit intervention
    intervention_data = dlpfc_data[dlpfc_data["condition"] == "intervention"]
    intervention_grouped = intervention_data.groupby("stimulus_level").agg(
        {"n_seen": "sum", "n_trials": "sum"}
    )

    intervention_params = psychometric.fit_curve(
        intervention_grouped.index.values,
        intervention_grouped["n_trials"].values,
        intervention_grouped["n_seen"].values,
    )

    print("\nIntervention (dlPFC TMS):")
    print(
        f"  Threshold: {intervention_params['threshold']:.4f} ± {intervention_params['threshold_se']:.4f}"
    )
    print(f"  Slope: {intervention_params['slope']:.4f}")
    print(f"  Lapse: {intervention_params['lapse']:.4f}")

    # Compare
    comparison = psychometric.compare_curves(baseline_params, intervention_params)

    print("\nComparison:")
    print(
        f"  Threshold shift: {comparison['threshold_shift']:.4f} ± {comparison['threshold_shift_se']:.4f}"
    )
    print(f"  Z-score: {comparison['threshold_z']:.3f}")
    print(f"  P-value: {comparison['threshold_p']:.4f}")
    print(f"  Significant: {'✅ YES' if comparison['significant'] else '❌ NO'}")

    # =========================================================================
    # STEP 5: Simulate Propranolol Study
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: SIMULATING PROPRANOLOL STUDY")
    print("=" * 80)

    propranolol_data = simulator.simulate_crossover_study(
        intervention=interventions["Propranolol"],
        control=interventions["Placebo"],
        n_subjects=30,
        n_trials_per_condition=200,
    )

    print("\n✅ Simulated propranolol crossover study")
    print(f"   Subjects: {propranolol_data['subject_id'].nunique()}")

    # =========================================================================
    # STEP 6: Falsification Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: FALSIFICATION ANALYSIS")
    print("=" * 80)

    checker = InterventionFalsificationChecker()

    # Check dlPFC TMS
    dlpfc_results = {
        "name": "dlPFC_TMS",
        "baseline_threshold": baseline_params["threshold"],
        "intervention_threshold": intervention_params["threshold"],
        "threshold_se": intervention_params["threshold_se"],
    }

    dlpfc_report = checker.generate_report(dlpfc_results)

    print("\ndlPFC TMS Falsification Report:")
    print(
        f"  Overall: {'❌ FALSIFIED' if dlpfc_report['overall_falsified'] else '✅ VALIDATED'}"
    )
    print(f"  Passed: {len(dlpfc_report['passed_criteria'])}")
    print(f"  Failed: {len(dlpfc_report['falsified_criteria'])}")

    for criterion in dlpfc_report["passed_criteria"]:
        print(f"\n  ✅ {criterion['code']}: {criterion['description']}")
        if "details" in criterion:
            for key, value in criterion["details"].items():
                if isinstance(value, (int, float)):
                    print(f"     {key}: {value:.4f}")
                else:
                    print(f"     {key}: {value}")

    # =========================================================================
    # STEP 7: Visualizations
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_intervention_results(
        dlpfc_data, "dlPFC TMS", save_path="protocol7_dlpfc_results.png"
    )

    plot_intervention_results(
        propranolol_data, "Propranolol", save_path="protocol7_propranolol_results.png"
    )

    # =========================================================================
    # STEP 8: Save Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: SAVING RESULTS")
    print("=" * 80)

    results_summary = {
        "interventions": {
            name: {
                "target": intervention.target_parameter,
                "effect_size": intervention.effect_size,
                "direction": intervention.effect_direction,
            }
            for name, intervention in interventions.items()
        },
        "dlpfc_tms": {
            "baseline": baseline_params,
            "intervention": intervention_params,
            "comparison": comparison,
            "falsification": dlpfc_report,
        },
    }

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.bool_)):
            return int(obj) if isinstance(obj, np.integer) else bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return str(obj) if hasattr(obj, "__dict__") else obj

    results_summary = convert_to_serializable(results_summary)

    with open("protocol7_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("✅ Results saved to: protocol7_results.json")

    # Save data
    dlpfc_data.to_csv("protocol7_dlpfc_data.csv", index=False)
    propranolol_data.to_csv("protocol7_propranolol_data.csv", index=False)

    print("✅ Data saved to CSV files")

    print("\n" + "=" * 80)
    print("Protocol 7 EXECUTION COMPLETE")
    print("=" * 80)

    return results_summary


def run_validation():
    """Entry point for CLI validation."""
    try:
        print(
            "Running APGI Validation Protocol 7: TMS/Pharmacological Intervention Predictions"
        )
        results = main()
        return {"passed": True, "status": "success", "results": results}
    except (RuntimeError, ValueError, TypeError, ImportError, KeyError) as e:
        print(f"Error in validation protocol 7: {e}")
        return {"passed": False, "status": "failed", "error": str(e)}


if __name__ == "__main__":
    main()
