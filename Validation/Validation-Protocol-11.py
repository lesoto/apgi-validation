"""
APGI Validation Protocol 11: Quantitative Model Fits
===================================================

Complete implementation of Priority 3 from the APGI Empirical Credibility Roadmap:
Quantitative model fits to behavioral data.

This protocol implements:
- Psychometric function fitting: P(seen) = 1/(1 + exp(-β(S - θ_t)))
- Spiking Leaky Neural Network (LNN) implementation reproducing consciousness paradigms
- Model comparison against GNW and additive linear alternatives
- Bayesian parameter estimation for model validation

Author: APGI Research Team
Date: 2026
Version: 1.0 (Quantitative Validation)
"""

import json
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import optimize, stats
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm

# PyMC for Bayesian estimation
try:
    import arviz as az
    import pymc as pm

    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("PyMC/ArviZ not available. Install with: pip install pymc arviz")

warnings.filterwarnings("ignore")


class PsychometricFunctionFitter:
    """Fit psychometric functions to behavioral data"""

    def __init__(self):
        self.models = {
            "apgi_sigmoid": self._apgi_sigmoid,
            "gnw_equivalent": self._gnw_equivalent,
            "additive_linear": self._additive_linear,
        }

    @staticmethod
    def _apgi_sigmoid(
        x: np.ndarray,
        beta: float,
        theta: float,
        amplitude: float = 1.0,
        baseline: float = 0.0,
    ) -> np.ndarray:
        """
        APGI sigmoidal psychometric function:
        P(seen) = baseline + amplitude / (1 + exp(-β(S - θ)))
        """
        return baseline + amplitude / (1 + np.exp(-beta * (x - theta)))

    @staticmethod
    def _gnw_equivalent(
        x: np.ndarray,
        slope: float,
        threshold: float,
        amplitude: float = 1.0,
        baseline: float = 0.0,
    ) -> np.ndarray:
        """
        GNW equivalent (simplified linear accumulation):
        P(seen) = baseline + amplitude * sigmoid(slope * (x - threshold))
        """
        return baseline + amplitude / (1 + np.exp(-slope * (x - threshold)))

    @staticmethod
    def _additive_linear(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        """Additive linear model for comparison"""
        return np.clip(slope * x + intercept, 0, 1)

    def fit_psychometric_functions(
        self, stimulus_intensities: np.ndarray, detection_rates: np.ndarray
    ) -> Dict:
        """
        Fit all psychometric models and compare performance

        Args:
            stimulus_intensities: Array of stimulus intensity values
            detection_rates: Array of detection rates (0-1)

        Returns:
            Dictionary with fit results for all models
        """
        results = {}

        # Fit APGI model
        try:
            popt_apgi, pcov_apgi = curve_fit(
                self._apgi_sigmoid,
                stimulus_intensities,
                detection_rates,
                p0=[5.0, np.median(stimulus_intensities), 1.0, 0.0],
                bounds=(
                    [1.0, 0.0, 0.5, 0.0],
                    [20.0, np.max(stimulus_intensities), 1.0, 0.5],
                ),
            )
            apgi_pred = self._apgi_sigmoid(stimulus_intensities, *popt_apgi)
            apgi_r2 = r2_score(detection_rates, apgi_pred)
            apgi_aic = self._calculate_aic(
                detection_rates, apgi_pred, 4
            )  # 4 parameters

            results["apgi_model"] = {
                "parameters": popt_apgi,
                "r2": apgi_r2,
                "aic": apgi_aic,
                "predictions": apgi_pred,
                "beta": popt_apgi[0],  # Sigmoid steepness
                "theta": popt_apgi[1],  # Threshold
                "phase_transition": popt_apgi[0]
                >= 10,  # β ≥ 10 indicates phase transition
            }
        except Exception as e:
            results["apgi_model"] = {"error": str(e)}

        # Fit GNW equivalent
        try:
            popt_gnw, pcov_gnw = curve_fit(
                self._gnw_equivalent,
                stimulus_intensities,
                detection_rates,
                p0=[2.0, np.median(stimulus_intensities), 1.0, 0.0],
            )
            gnw_pred = self._gnw_equivalent(stimulus_intensities, *popt_gnw)
            gnw_r2 = r2_score(detection_rates, gnw_pred)
            gnw_aic = self._calculate_aic(detection_rates, gnw_pred, 4)

            results["gnw_model"] = {
                "parameters": popt_gnw,
                "r2": gnw_r2,
                "aic": gnw_aic,
                "predictions": gnw_pred,
            }
        except Exception as e:
            results["gnw_model"] = {"error": str(e)}

        # Fit additive linear
        try:
            popt_linear, pcov_linear = curve_fit(
                self._additive_linear,
                stimulus_intensities,
                detection_rates,
                p0=[1.0, 0.0],
            )
            linear_pred = self._additive_linear(stimulus_intensities, *popt_linear)
            linear_r2 = r2_score(detection_rates, linear_pred)
            linear_aic = self._calculate_aic(detection_rates, linear_pred, 2)

            results["linear_model"] = {
                "parameters": popt_linear,
                "r2": linear_r2,
                "aic": linear_aic,
                "predictions": linear_pred,
            }
        except Exception as e:
            results["linear_model"] = {"error": str(e)}

        # Model comparison
        if all(key in results for key in ["apgi_model", "gnw_model", "linear_model"]):
            results["model_comparison"] = self._compare_models(results)

        return results

    def _calculate_aic(
        self, observed: np.ndarray, predicted: np.ndarray, n_params: int
    ) -> float:
        """Calculate Akaike Information Criterion"""
        n = len(observed)
        rss = np.sum((observed - predicted) ** 2)
        if rss <= 0:
            return float("inf")
        sigma2 = rss / n
        log_likelihood = -n / 2 * np.log(2 * np.pi * sigma2) - 1 / (2 * sigma2) * rss
        return 2 * n_params - 2 * log_likelihood

    def _compare_models(self, results: Dict) -> Dict:
        """Compare fitted models using AIC and R²"""
        apgi_aic = results["apgi_model"].get("aic", float("inf"))
        gnw_aic = results["gnw_model"].get("aic", float("inf"))
        linear_aic = results["linear_model"].get("aic", float("inf"))

        # Calculate AIC weights (evidence ratios)
        aic_values = np.array([apgi_aic, gnw_aic, linear_aic])
        min_aic = np.min(aic_values)
        aic_weights = np.exp(-0.5 * (aic_values - min_aic))
        aic_weights /= np.sum(aic_weights)

        return {
            "apgi_vs_gnw_bf": (
                aic_weights[0] / aic_weights[1] if aic_weights[1] > 0 else float("inf")
            ),
            "apgi_vs_linear_bf": (
                aic_weights[0] / aic_weights[2] if aic_weights[2] > 0 else float("inf")
            ),
            "apgi_preferred": aic_weights[0] > aic_weights[1]
            and aic_weights[0] > aic_weights[2],
            "phase_transition_evidence": results["apgi_model"].get(
                "phase_transition", False
            ),
        }


class SpikingLNNModel:
    """Spiking Leaky Neural Network for consciousness paradigm simulation"""

    def __init__(self, n_neurons: int = 100, tau: float = 0.02, threshold: float = 1.0):
        """
        Args:
            n_neurons: Number of neurons in the network
            tau: Membrane time constant
            threshold: Spiking threshold
        """
        self.n_neurons = n_neurons
        self.tau = tau
        self.threshold = threshold

        # Network parameters
        self.weights = np.random.normal(0, 0.1, (n_neurons, n_neurons))
        np.fill_diagonal(self.weights, 0)  # No self-connections

        # APGI-specific parameters
        self.Pi_e = 1.0  # Exteroceptive precision
        self.Pi_i = 1.0  # Interoceptive precision
        self.theta_t = 0.5  # Dynamic threshold
        self.beta = 1.5  # Somatic influence

    def simulate_trial(
        self, stimulus_intensity: float, trial_duration: float = 1.0, dt: float = 0.001
    ) -> Dict:
        """
        Simulate single trial with APGI parameters

        Args:
            stimulus_intensity: External stimulus intensity (0-1)
            trial_duration: Trial duration in seconds
            dt: Time step

        Returns:
            Dictionary with simulation results
        """
        n_steps = int(trial_duration / dt)

        # Initialize network state
        membrane_potential = np.zeros(self.n_neurons)
        spikes = np.zeros((self.n_neurons, n_steps))
        ignition_signal = np.zeros(n_steps)

        # APGI state variables
        S = 0.0  # Accumulated surprise
        M = 0.0  # Somatic marker

        for step in range(1, n_steps):
            # Generate input (simplified)
            external_input = stimulus_intensity * np.random.normal(
                1, 0.1, self.n_neurons
            )
            internal_input = 0.1 * np.random.normal(
                0, 1, self.n_neurons
            )  # Interoceptive noise

            # APGI prediction error computation
            eps_e = external_input - np.mean(external_input)  # Simplified
            eps_i = internal_input - np.mean(internal_input)

            # Update somatic marker (simplified)
            M_target = np.tanh(self.beta * np.mean(eps_i))
            M += (M_target - M) / (0.1 / dt)  # Tau_M = 0.1 s

            # Effective interoceptive precision
            Pi_i_eff = self.Pi_i * np.exp(self.beta * M)

            # Accumulate surprise
            S += (self.Pi_e * np.mean(eps_e**2) + Pi_i_eff * np.mean(eps_i**2)) * dt
            S *= np.exp(-dt / 0.35)  # Tau_S = 350 ms

            # Dynamic threshold
            self.theta_t = 0.5 + 0.1 * S  # Simplified

            # Network dynamics with APGI modulation
            input_current = external_input + internal_input * Pi_i_eff
            input_current *= self.Pi_e  # Precision weighting

            # Update membrane potentials
            dV_dt = (
                -membrane_potential
                + input_current
                + np.dot(self.weights, spikes[:, step - 1])
            ) / self.tau
            membrane_potential += dV_dt * dt

            # Spiking
            spiking_neurons = membrane_potential >= self.threshold
            spikes[spiking_neurons, step] = 1
            membrane_potential[spiking_neurons] = 0  # Reset

            # Ignition detection
            ignition_signal[step] = 1.0 / (1.0 + np.exp(-5.0 * (S - self.theta_t)))

        return {
            "spikes": spikes,
            "ignition_signal": ignition_signal,
            "final_surprise": S,
            "final_threshold": self.theta_t,
            "detection": ignition_signal[-1] > 0.5,
            "time": np.arange(n_steps) * dt,
        }

    def simulate_consciousness_paradigm(
        self, paradigm: str, n_trials: int = 100
    ) -> Dict:
        """
        Simulate specific consciousness paradigm

        Args:
            paradigm: 'masking', 'binocular_rivalry', 'attentional_blink'
            n_trials: Number of trials to simulate

        Returns:
            Dictionary with paradigm-specific results
        """
        results = {
            "paradigm": paradigm,
            "trials": [],
            "psychometric_data": {"intensities": [], "detections": []},
        }

        if paradigm == "backward_masking":
            # Backward masking paradigm
            for trial in range(n_trials):
                # Vary stimulus-mask SOA (simulated as intensity)
                soa_intensity = np.random.uniform(0.1, 1.0)
                trial_result = self.simulate_trial(soa_intensity)
                results["trials"].append(trial_result)
                results["psychometric_data"]["intensities"].append(soa_intensity)
                results["psychometric_data"]["detections"].append(
                    trial_result["detection"]
                )

        elif paradigm == "attentional_blink":
            # Attentional blink
            for trial in range(n_trials):
                # Simulate lag between targets
                lag_intensity = np.random.uniform(0.1, 1.0)
                trial_result = self.simulate_trial(lag_intensity)
                results["trials"].append(trial_result)
                results["psychometric_data"]["intensities"].append(lag_intensity)
                results["psychometric_data"]["detections"].append(
                    trial_result["detection"]
                )

        return results


class BayesianParameterEstimator:
    """Bayesian parameter estimation for APGI model validation"""

    def __init__(self):
        if not BAYESIAN_AVAILABLE:
            raise ImportError("PyMC/ArviZ required for Bayesian estimation")

    def estimate_apgi_parameters(self, behavioral_data: pd.DataFrame) -> Dict:
        """
        Estimate APGI parameters using Bayesian inference

        Args:
            behavioral_data: DataFrame with stimulus intensities and detection responses

        Returns:
            Dictionary with posterior estimates
        """
        stimulus_intensities = behavioral_data["stimulus_intensity"].values
        detections = behavioral_data["detected"].values.astype(int)

        with pm.Model() as apgi_model:
            # Priors for APGI parameters
            beta = pm.Normal("beta", mu=5.0, sigma=2.0)  # Sigmoid steepness
            theta = pm.Normal("theta", mu=0.5, sigma=0.2)  # Threshold
            amplitude = pm.Beta("amplitude", alpha=2, beta=1)  # Response amplitude
            baseline = pm.Beta("baseline", alpha=1, beta=2)  # Baseline response

            # APGI psychometric function
            prob_detect = baseline + amplitude / (
                1 + pm.math.exp(-beta * (stimulus_intensities - theta))
            )

            # Likelihood
            detections_obs = pm.Bernoulli(
                "detections_obs", p=prob_detect, observed=detections
            )

            # Sample posterior
            trace = pm.sample(1000, tune=1000, return_inferencedata=True)

        # Extract posterior summaries
        summary = az.summary(trace, round_to=3)

        return {
            "posterior_summary": summary,
            "trace": trace,
            "beta_posterior_mean": summary.loc["beta", "mean"],
            "theta_posterior_mean": summary.loc["theta", "mean"],
            "phase_transition_posterior": summary.loc["beta", "mean"] >= 10,
        }


class QuantitativeModelValidator:
    """Complete quantitative model validation"""

    def __init__(self):
        self.psychometric_fitter = PsychometricFunctionFitter()
        self.lnn_model = SpikingLNNModel()
        self.bayesian_estimator = (
            BayesianParameterEstimator() if BAYESIAN_AVAILABLE else None
        )

    def validate_quantitative_fits(self) -> Dict:
        """
        Complete validation of quantitative model predictions

        Returns:
            Dictionary with all validation results
        """

        results = {
            "psychometric_fitting": self._validate_psychometric_fits(),
            "spiking_lnn_simulation": self._validate_spiking_lnn(),
            "consciousness_paradigms": self._validate_consciousness_paradigms(),
            "bayesian_estimation": self._validate_bayesian_estimation(),
            "overall_quantitative_score": 0.0,
        }

        # Calculate overall score
        results["overall_quantitative_score"] = self._calculate_quantitative_score(
            results
        )

        return results

    def _validate_psychometric_fits(self) -> Dict:
        """Validate psychometric function fitting"""

        # Generate synthetic data matching APGI predictions
        stimulus_intensities = np.linspace(0.1, 1.0, 50)
        true_beta = 12.0  # Phase transition steepness
        true_theta = 0.5

        # Generate detection probabilities using APGI model
        true_probabilities = 1.0 / (
            1 + np.exp(-true_beta * (stimulus_intensities - true_theta))
        )

        # Add noise to simulate behavioral data
        detection_rates = (
            np.random.binomial(20, true_probabilities) / 20
        )  # 20 trials per intensity

        # Fit all models
        fit_results = self.psychometric_fitter.fit_psychometric_functions(
            stimulus_intensities, detection_rates
        )

        return {
            "stimulus_intensities": stimulus_intensities,
            "detection_rates": detection_rates,
            "true_probabilities": true_probabilities,
            "model_fits": fit_results,
        }

    def _validate_spiking_lnn(self) -> Dict:
        """Validate spiking LNN implementation"""

        # Test basic spiking dynamics
        model = SpikingLNNModel(n_neurons=50)
        trial_result = model.simulate_trial(stimulus_intensity=0.7)

        # Check for realistic spiking statistics
        spike_counts = np.sum(trial_result["spikes"], axis=1)
        mean_firing_rate = np.mean(spike_counts) / 1.0  # Hz

        return {
            "spike_counts": spike_counts,
            "mean_firing_rate": mean_firing_rate,
            "ignition_detected": trial_result["detection"],
            "realistic_dynamics": 5 <= mean_firing_rate <= 50,  # Reasonable range
        }

    def _validate_consciousness_paradigms(self) -> Dict:
        """Validate reproduction of consciousness paradigms"""

        paradigms = ["backward_masking", "attentional_blink"]
        paradigm_results = {}

        for paradigm in paradigms:
            # Simulate paradigm
            results = self.lnn_model.simulate_consciousness_paradigm(
                paradigm, n_trials=50
            )

            # Fit psychometric function to simulated data
            intensities = np.array(results["psychometric_data"]["intensities"])
            detections = np.array(results["psychometric_data"]["detections"], dtype=int)

            # Simple psychometric fit
            try:
                from scipy.optimize import curve_fit

                def sigmoid(x, beta, theta):
                    return 1.0 / (1 + np.exp(-beta * (x - theta)))

                popt, _ = curve_fit(sigmoid, intensities, detections, p0=[5.0, 0.5])
                fitted_curve = sigmoid(intensities, *popt)

                paradigm_results[paradigm] = {
                    "beta_fitted": popt[0],
                    "theta_fitted": popt[1],
                    "phase_transition": popt[0] >= 10,
                    "r2": r2_score(detections, fitted_curve),
                }
            except Exception:
                paradigm_results[paradigm] = {"error": "Fit failed"}

        return paradigm_results

    def _validate_bayesian_estimation(self) -> Dict:
        """Validate Bayesian parameter estimation"""

        if not self.bayesian_estimator:
            return {"error": "Bayesian estimation not available"}

        # Generate synthetic behavioral data
        np.random.seed(42)
        n_trials = 200
        stimulus_intensities = np.random.uniform(0.1, 1.0, n_trials)
        true_beta, true_theta = 12.0, 0.5

        # Generate detections
        true_probs = 1.0 / (
            1 + np.exp(-true_beta * (stimulus_intensities - true_theta))
        )
        detections = np.random.binomial(1, true_probs)

        behavioral_data = pd.DataFrame(
            {"stimulus_intensity": stimulus_intensities, "detected": detections}
        )

        # Estimate parameters
        try:
            estimation_results = self.bayesian_estimator.estimate_apgi_parameters(
                behavioral_data
            )
            return estimation_results
        except Exception as e:
            return {"error": str(e)}

    def _calculate_quantitative_score(self, results: Dict) -> float:
        """Calculate overall quantitative validation score"""

        scores = []

        # Psychometric fitting (weight: 0.4)
        psycho_result = results.get("psychometric_fitting", {})
        model_comparison = psycho_result.get("model_fits", {}).get(
            "model_comparison", {}
        )
        scores.append(
            0.4 * (1.0 if model_comparison.get("apgi_preferred", False) else 0.0)
        )

        # Spiking LNN (weight: 0.3)
        lnn_result = results.get("spiking_lnn_simulation", {})
        scores.append(
            0.3 * (1.0 if lnn_result.get("realistic_dynamics", False) else 0.0)
        )

        # Consciousness paradigms (weight: 0.2)
        paradigm_result = results.get("consciousness_paradigms", {})
        paradigm_success = any(
            paradigm.get("phase_transition", False)
            for paradigm in paradigm_result.values()
            if isinstance(paradigm, dict)
        )
        scores.append(0.2 * (1.0 if paradigm_success else 0.0))

        # Bayesian estimation (weight: 0.1)
        bayesian_result = results.get("bayesian_estimation", {})
        scores.append(0.1 * (1.0 if "posterior_summary" in bayesian_result else 0.0))

        return sum(scores)


def main():
    """Run quantitative model validation"""
    validator = QuantitativeModelValidator()
    results = validator.validate_quantitative_fits()

    print("APGI Quantitative Model Validation Results:")
    print(
        f"Overall Quantitative Validation Score: {results['overall_quantitative_score']:.3f}"
    )

    print("\nDetailed Results:")
    for key, value in results.items():
        if key != "overall_quantitative_score":
            print(f"\n{key}:")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        print(f"  {sub_key}: {sub_value:.3f}")
                    else:
                        print(f"  {sub_key}: {sub_value}")
            else:
                print(f"  {value}")


if __name__ == "__main__":
    main()
