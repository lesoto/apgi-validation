"""
APGI Protocol 4: Information-Theoretic Phase Transition Analysis
==================================================================

Complete implementation of information-theoretic testing framework for APGI's
phase transition predictions. Tests whether conscious ignition represents a
genuine computational phase transition with distinctive information-theoretic
signatures.

This protocol implements:
- Transfer entropy analysis (information flow)
- Integrated information measures (Φ-like)
- Phase transition detection (discontinuities, susceptibility, critical slowing)
- Long-range correlation analysis (Hurst exponents)
- Comprehensive falsification testing

Author: APGI Research Team
Date: 2025
Version: 1.0 (Production)

Dependencies:
    numpy, scipy, pandas, matplotlib, seaborn, sklearn, tqdm
"""

import json
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal, stats
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# PART 1: APGI DYNAMICAL SYSTEM
# =============================================================================


class APGIDynamicalSystem:
    """
    Core APGI equations for surprise accumulation and ignition

    This is the system whose phase transition properties we're testing.
    """

    def __init__(
        self,
        tau: float = 0.2,
        theta_0: float = 0.55,
        alpha: float = 5.0,
        dt: float = 0.01,
    ):
        """
        Args:
            tau: Surprise decay time constant (seconds)
            theta_0: Baseline ignition threshold
            alpha: Sigmoid steepness for ignition probability
            dt: Integration timestep (seconds)
        """
        self.tau = tau
        self.theta_0 = theta_0
        self.alpha = alpha
        self.dt = dt

    def simulate(
        self, duration: float, input_generator: Callable, theta_noise_sd: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Simulate APGI dynamics with time-varying inputs

        Args:
            duration: Simulation duration (seconds)
            input_generator: Function(t) -> Dict with keys:
                - Pi_e, eps_e, beta, Pi_i, eps_i, M, c, a
            theta_noise_sd: Standard deviation of threshold noise

        Returns:
            Dictionary with time series:
                - time, S, theta, B, Pi_e, eps_e, Pi_i, eps_i, ignition_events
        """

        n_steps = int(duration / self.dt)

        # Initialize arrays
        time = np.arange(n_steps) * self.dt
        S = np.zeros(n_steps)
        theta = np.zeros(n_steps)
        B = np.zeros(n_steps)

        # Store inputs for analysis
        Pi_e_history = np.zeros(n_steps)
        eps_e_history = np.zeros(n_steps)
        Pi_i_history = np.zeros(n_steps)
        eps_i_history = np.zeros(n_steps)

        # Track ignition events
        ignition_events = []
        was_ignited = False

        for i in range(n_steps):
            # Get current inputs
            inputs = input_generator(time[i])

            Pi_e = inputs.get("Pi_e", 1.0)
            eps_e = inputs.get("eps_e", 0.0)
            beta = inputs.get("beta", 1.15)
            Pi_i = inputs.get("Pi_i", 1.2)
            eps_i = inputs.get("eps_i", 0.0)
            M = inputs.get("M", 1.0)
            c = inputs.get("c", 0.5)
            a = inputs.get("a", 0.3)

            # Store inputs
            Pi_e_history[i] = Pi_e
            eps_e_history[i] = eps_e
            Pi_i_history[i] = Pi_i
            eps_i_history[i] = eps_i

            # Dynamic threshold
            theta_noise = np.random.normal(0, theta_noise_sd) if theta_noise_sd > 0 else 0
            theta[i] = self.theta_0 + theta_noise

            # APGI core equation: dS/dt = -S/τ + Π_e·|ε_e| + β·Π_i(M,c,a)·|ε_i|
            if i > 0:
                extero_contrib = Pi_e * np.abs(eps_e)
                intero_contrib = beta * Pi_i * np.abs(eps_i)

                dS_dt = -S[i - 1] / self.tau + extero_contrib + intero_contrib
                S[i] = S[i - 1] + self.dt * dS_dt
                S[i] = np.clip(S[i], 0, 10)  # Physiological bounds

            # Ignition probability
            B[i] = self._sigmoid(S[i] - theta[i])

            # Detect ignition events (threshold crossing)
            is_ignited = S[i] > theta[i]
            if is_ignited and not was_ignited:
                ignition_events.append(i)
            was_ignited = is_ignited

        return {
            "time": time,
            "S": S,
            "theta": theta,
            "B": B,
            "Pi_e": Pi_e_history,
            "eps_e": eps_e_history,
            "Pi_i": Pi_i_history,
            "eps_i": eps_i_history,
            "ignition_events": np.array(ignition_events),
        }

    def _sigmoid(self, x: float) -> float:
        """Sigmoid with steepness α"""
        return 1.0 / (1.0 + np.exp(-self.alpha * x))


# =============================================================================
# PART 2: INFORMATION-THEORETIC MEASURES
# =============================================================================


class InformationTheoreticAnalysis:
    """
    Compute information-theoretic measures on APGI dynamics

    Measures:
        - Transfer Entropy (TE): Directed information flow
        - Integrated Information (Φ): Whole > sum of parts
        - Mutual Information (MI): Statistical dependencies
        - Entropy rates
    """

    def __init__(self, n_bins: int = 20):
        """
        Args:
            n_bins: Number of bins for discretization (for MI/TE estimation)
        """
        self.n_bins = n_bins

    def compute_transfer_entropy(
        self, source: np.ndarray, target: np.ndarray, lag: int = 1
    ) -> float:
        """
        Transfer entropy: Information flow from source to target

        TE(X→Y) = I(Y_t; X_{t-lag} | Y_{t-lag})
                = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})

        Prediction: TE increases dramatically at ignition

        Args:
            source: Source time series (X)
            target: Target time series (Y)
            lag: Time lag

        Returns:
            Transfer entropy value (nats)
        """

        if len(source) < lag + 10:
            return 0.0

        # Discretize
        source_binned = self._discretize(source)
        target_binned = self._discretize(target)

        # Create lagged versions
        Y_t = target_binned[lag:]
        Y_past = target_binned[:-lag]
        X_past = source_binned[:-lag]

        # Compute conditional entropies
        # H(Y_t | Y_{t-lag})
        H_Y_given_Ypast = self._conditional_entropy(Y_t, Y_past)

        # H(Y_t | Y_{t-lag}, X_{t-lag})
        H_Y_given_both = self._conditional_entropy_joint(Y_t, Y_past, X_past)

        # TE = difference
        te = H_Y_given_Ypast - H_Y_given_both

        return max(0, te)  # TE should be non-negative

    def compute_integrated_information(
        self, variables: List[np.ndarray], window_size: int = 100
    ) -> np.ndarray:
        """
        Φ-like measure: How much more information in whole than parts

        Φ ≈ MI(system) = H(X₁) + H(X₂) + ... - H(X₁, X₂, ...)

        Prediction: Φ spikes at ignition

        Args:
            variables: List of time series (e.g., [S, theta, Pi_i])
            window_size: Window for moving calculation

        Returns:
            Φ time series
        """

        n_vars = len(variables)
        n_timepoints = len(variables[0])

        phi_values = np.zeros(n_timepoints - window_size)

        for t in range(window_size, n_timepoints):
            # Extract windows
            windows = [var[t - window_size : t] for var in variables]

            # Discretize
            windows_binned = [self._discretize(w) for w in windows]

            # Marginal entropies
            H_marginals = sum(self._entropy(w) for w in windows_binned)

            # Joint entropy
            joint_data = np.column_stack(windows_binned)
            H_joint = self._entropy_multivariate(joint_data)

            # Φ ≈ mutual information
            phi_values[t - window_size] = H_marginals - H_joint

        return phi_values

    def compute_mutual_information(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Mutual information: I(X; Y) = H(X) + H(Y) - H(X, Y)

        Args:
            X: First variable
            Y: Second variable

        Returns:
            Mutual information (nats)
        """

        X_binned = self._discretize(X)
        Y_binned = self._discretize(Y)

        return mutual_info_score(X_binned, Y_binned)

    def compute_entropy_rate(self, series: np.ndarray, lag: int = 1) -> float:
        """
        Entropy rate: H_rate = H(X_t | X_{t-lag})

        Measures predictability of the process.

        Args:
            series: Time series
            lag: Conditioning lag

        Returns:
            Entropy rate (nats/sample)
        """

        if len(series) < lag + 10:
            return 0.0

        series_binned = self._discretize(series)

        X_t = series_binned[lag:]
        X_past = series_binned[:-lag]

        return self._conditional_entropy(X_t, X_past)

    def _discretize(self, data: np.ndarray) -> np.ndarray:
        """Discretize continuous data into bins"""
        data = data.reshape(-1, 1)
        discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="uniform")
        return discretizer.fit_transform(data).astype(int).flatten()

    def _entropy(self, binned_data: np.ndarray) -> float:
        """Compute entropy of discretized data"""
        counts = np.bincount(binned_data, minlength=self.n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Remove zeros
        return -np.sum(probs * np.log(probs))

    def _entropy_multivariate(self, joint_data: np.ndarray) -> float:
        """Compute joint entropy of multivariate data"""
        # Convert to tuple of values for counting
        tuples = [tuple(row) for row in joint_data]
        if len(tuples) == 0:
            return 0.0
        # Convert to numpy array for np.unique with axis parameter
        tuples_array = np.array(tuples)
        unique, counts = np.unique(tuples_array, return_counts=True, axis=0)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    def _conditional_entropy(self, Y: np.ndarray, X: np.ndarray) -> float:
        """H(Y | X) = H(X, Y) - H(X)"""
        H_X = self._entropy(X)
        joint_data = np.column_stack([X, Y])
        H_joint = self._entropy_multivariate(joint_data)
        return H_joint - H_X

    def _conditional_entropy_joint(self, Y: np.ndarray, X1: np.ndarray, X2: np.ndarray) -> float:
        """H(Y | X1, X2) = H(X1, X2, Y) - H(X1, X2)"""
        joint_X = np.column_stack([X1, X2])
        H_X = self._entropy_multivariate(joint_X)

        joint_all = np.column_stack([X1, X2, Y])
        H_joint = self._entropy_multivariate(joint_all)

        return H_joint - H_X


# =============================================================================
# PART 3: PHASE TRANSITION DETECTION
# =============================================================================


class PhaseTransitionDetector:
    """
    Detect signatures of phase transition in APGI dynamics

    Signatures tested:
        1. Discontinuity in derivatives at threshold crossing
        2. Diverging susceptibility (variance) near threshold
        3. Critical slowing down (autocorrelation increase)
        4. Long-range correlations (Hurst exponent)
        5. Finite-size scaling
    """

    def __init__(self):
        pass

    def detect_discontinuity(
        self,
        S: np.ndarray,
        theta: np.ndarray,
        time: np.ndarray,
        ignition_events: np.ndarray,
    ) -> Dict[str, float]:
        """
        Test for discontinuity in dS/dt at ignition

        P4a Prediction: Mean |dS/dt| discontinuity > 0.5

        Args:
            S: Surprise time series
            theta: Threshold time series
            time: Time points
            ignition_events: Indices of ignition events

        Returns:
            Dictionary with discontinuity measures
        """

        # Compute derivatives
        dS_dt = np.gradient(S, time)

        discontinuities = []
        pre_slopes = []
        post_slopes = []

        window = 10  # Points before/after to average

        for idx in ignition_events:
            if idx > window and idx < len(S) - window:
                # Pre-ignition slope
                pre_slope = np.mean(dS_dt[idx - window : idx])

                # Post-ignition slope
                post_slope = np.mean(dS_dt[idx : idx + window])

                # Discontinuity magnitude
                discontinuity = abs(post_slope - pre_slope)

                discontinuities.append(discontinuity)
                pre_slopes.append(pre_slope)
                post_slopes.append(post_slope)

        if len(discontinuities) == 0:
            return {"mean_discontinuity": 0.0, "max_discontinuity": 0.0, "n_events": 0}

        # Compare to random timepoints (control)
        random_indices = np.random.choice(
            range(window, len(S) - window),
            size=min(len(discontinuities) * 5, 100),
            replace=False,
        )

        random_discontinuities = []
        for idx in random_indices:
            pre = np.mean(dS_dt[idx - window : idx])
            post = np.mean(dS_dt[idx : idx + window])
            random_discontinuities.append(abs(post - pre))

        # Effect size (Cohen's d)
        if len(random_discontinuities) > 0:
            pooled_std = np.sqrt((np.var(discontinuities) + np.var(random_discontinuities)) / 2)
            cohens_d = (np.mean(discontinuities) - np.mean(random_discontinuities)) / (
                pooled_std + 1e-10
            )
        else:
            cohens_d = 0.0

        return {
            "mean_discontinuity": float(np.mean(discontinuities)),
            "std_discontinuity": float(np.std(discontinuities)),
            "max_discontinuity": float(np.max(discontinuities)),
            "n_events": len(discontinuities),
            "random_mean": (
                float(np.mean(random_discontinuities)) if random_discontinuities else 0.0
            ),
            "cohens_d": float(cohens_d),
        }

    def compute_susceptibility(self, S: np.ndarray, theta: np.ndarray) -> Dict[str, float]:
        """
        Test for diverging susceptibility near threshold

        P4b Prediction: Variance ratio > 2.0

        Susceptibility χ ∝ variance of order parameter near critical point

        Args:
            S: Surprise time series
            theta: Threshold time series

        Returns:
            Dictionary with susceptibility measures
        """

        proximity = np.abs(S - theta)

        # Near threshold: proximity < 0.15
        near_mask = proximity < 0.15

        # Far from threshold: proximity > 0.4
        far_mask = proximity > 0.4

        if np.sum(near_mask) < 10 or np.sum(far_mask) < 10:
            return {
                "susceptibility_ratio": 1.0,
                "variance_near": 0.0,
                "variance_far": 0.0,
            }

        # Variance (susceptibility proxy)
        var_near = np.var(S[near_mask])
        var_far = np.var(S[far_mask])

        ratio = var_near / (var_far + 1e-10)

        return {
            "susceptibility_ratio": float(ratio),
            "variance_near": float(var_near),
            "variance_far": float(var_far),
            "n_near": int(np.sum(near_mask)),
            "n_far": int(np.sum(far_mask)),
        }

    def detect_critical_slowing(
        self, S: np.ndarray, theta: np.ndarray, max_lag: int = 20
    ) -> Dict[str, float]:
        """
        Test for critical slowing down near threshold

        P4c Prediction: Autocorrelation ratio > 1.5

        Near phase transition, relaxation time diverges → increased autocorrelation

        Args:
            S: Surprise time series
            theta: Threshold time series
            max_lag: Maximum lag for autocorrelation

        Returns:
            Dictionary with autocorrelation measures
        """

        proximity = np.abs(S - theta)

        near_mask = proximity < 0.15
        far_mask = proximity > 0.4

        if np.sum(near_mask) < max_lag * 3 or np.sum(far_mask) < max_lag * 3:
            return {"critical_slowing_ratio": 1.0, "acf_near": 0.0, "acf_far": 0.0}

        # Compute autocorrelation at lag
        lag = min(max_lag, min(np.sum(near_mask), np.sum(far_mask)) // 3)

        acf_near = self._autocorrelation(S[near_mask], lag)
        acf_far = self._autocorrelation(S[far_mask], lag)

        ratio = acf_near / (acf_far + 1e-10)

        return {
            "critical_slowing_ratio": float(ratio),
            "acf_near": float(acf_near),
            "acf_far": float(acf_far),
            "lag": lag,
        }

    def compute_hurst_exponent(self, S: np.ndarray, theta: np.ndarray) -> Dict[str, float]:
        """
        Test for long-range correlations via Hurst exponent

        P4e Prediction:
            - H near threshold > 0.6 (long-range correlations)
            - H far from threshold ≈ 0.5 (random walk)

        Hurst exponent H:
            - H = 0.5: Random walk (no correlations)
            - H > 0.5: Persistent (positive autocorrelations)
            - H < 0.5: Anti-persistent

        Args:
            S: Surprise time series
            theta: Threshold time series

        Returns:
            Dictionary with Hurst exponents
        """

        proximity = np.abs(S - theta)

        near_mask = proximity < 0.15
        far_mask = proximity > 0.4

        results = {}

        if np.sum(near_mask) > 50:
            H_near = self._hurst_exponent(S[near_mask])
            results["hurst_near"] = float(H_near)
        else:
            results["hurst_near"] = 0.5

        if np.sum(far_mask) > 50:
            H_far = self._hurst_exponent(S[far_mask])
            results["hurst_far"] = float(H_far)
        else:
            results["hurst_far"] = 0.5

        results["hurst_difference"] = results["hurst_near"] - results["hurst_far"]

        return results

    def _autocorrelation(self, series: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag"""
        if len(series) < lag + 2:
            return 0.0

        # Normalize
        series_norm = (series - np.mean(series)) / (np.std(series) + 1e-10)

        # Autocorrelation
        n = len(series_norm)
        acf = np.correlate(series_norm[:-lag], series_norm[lag:], mode="valid")[0]
        acf /= n - lag

        return acf

    def _hurst_exponent(self, series: np.ndarray) -> float:
        """
        Estimate Hurst exponent via R/S analysis

        R/S = (R/S)_expected * n^H
        where H is Hurst exponent
        """

        if len(series) < 20:
            return 0.5

        n = len(series)
        max_lag = n // 4

        if max_lag < 10:
            return 0.5

        lags = np.logspace(1, np.log10(max_lag), num=10, dtype=int)
        lags = np.unique(lags)

        rs_values = []

        for lag in lags:
            if lag >= n:
                continue

            # Number of subseries
            n_subseries = n // lag

            if n_subseries < 1:
                continue

            rs_sub = []

            for i in range(n_subseries):
                subseries = series[i * lag : (i + 1) * lag]

                if len(subseries) < 2:
                    continue

                # Mean-adjusted cumulative sum
                mean = np.mean(subseries)
                cumsum = np.cumsum(subseries - mean)

                # Range
                R = np.max(cumsum) - np.min(cumsum)

                # Standard deviation
                S = np.std(subseries)

                if S > 1e-10:
                    rs_sub.append(R / S)

            if len(rs_sub) > 0:
                rs_values.append(np.mean(rs_sub))

        if len(rs_values) < 3:
            return 0.5

        # Fit log(R/S) vs log(n)
        log_lags = np.log(lags[: len(rs_values)])
        log_rs = np.log(rs_values)

        # Linear regression
        slope, _ = np.polyfit(log_lags, log_rs, 1)

        # Hurst exponent is the slope
        return np.clip(slope, 0.0, 1.0)


# =============================================================================
# PART 3.5: FINITE-SIZE SCALING AND CRITICAL EXPONENT ANALYSIS
# =============================================================================


class FiniteSizeScalingAnalysis:
    """
    Finite-size scaling analysis to identify critical points and estimate exponents

    Based on Binder cumulant method and critical exponent estimation
    """

    def __init__(self):
        pass

    def finite_size_scaling_analysis(
        self,
        system_sizes: List[int],
        parameter_range: np.ndarray,
        parameter_name: str = "theta_0",
    ) -> Dict:
        """
        Proper finite-size scaling to identify critical point
        Essential for phase transition claims

        Based on Binder cumulant method (Binder, 1981)

        Args:
            system_sizes: List of system sizes (durations)
            parameter_range: Range of parameter values to test
            parameter_name: Name of parameter to vary

        Returns:
            Dictionary with scaling analysis results
        """
        results = {
            "sizes": system_sizes,
            "order_parameters": {},
            "susceptibilities": {},
            "binder_cumulants": {},
            "critical_point_estimate": None,
            "parameter_values": parameter_range,
        }

        for size in system_sizes:
            order_params = []
            suscepts = []
            binder_vals = []

            for param in parameter_range:
                # Create system with this parameter value
                system_params = {"tau": 0.2, "theta_0": 0.55, "alpha": 5.0, "dt": 0.01}
                system_params[parameter_name] = param

                system = APGIDynamicalSystem(**system_params)

                # Input generator for this simulation
                def input_gen(t):
                    return {
                        "Pi_e": 1.0 + 0.3 * np.sin(2 * np.pi * t / 15),
                        "eps_e": np.random.normal(0.4, 0.2),
                        "beta": 1.15,
                        "Pi_i": 1.0 + 0.5 * np.sin(2 * np.pi * t / 25),
                        "eps_i": np.random.normal(0.2, 0.15),
                        "M": 1.0,
                        "c": 0.5,
                        "a": 0.3,
                    }

                # Simulate system at this size and parameter value
                timeseries = system.simulate(
                    duration=size, input_generator=input_gen, theta_noise_sd=0.05
                )

                # Order parameter (mean ignition rate)
                m = np.mean(timeseries["B"])
                order_params.append(m)

                # Susceptibility (variance of order parameter)
                chi = np.var(timeseries["B"]) * size
                suscepts.append(chi)

                # Binder cumulant: U = 1 - <m^4>/(3<m^2>^2)
                m2 = np.mean(timeseries["B"] ** 2)
                m4 = np.mean(timeseries["B"] ** 4)
                U = 1 - m4 / (3 * m2**2) if m2 > 0 else 0
                binder_vals.append(U)

            results["order_parameters"][size] = order_params
            results["susceptibilities"][size] = suscepts
            results["binder_cumulants"][size] = binder_vals

        # Find crossing point of Binder cumulants (identifies critical point)
        # At criticality, Binder cumulant becomes size-independent
        crossing_point = self._find_binder_crossing(results["binder_cumulants"], parameter_range)
        results["critical_point_estimate"] = crossing_point

        return results

    def estimate_critical_exponents(self, data: Dict, critical_point: float) -> Dict:
        """
        Estimate critical exponents β, γ, ν
        These characterize the universality class of the transition

        Near criticality:
        - Order parameter: m ~ |T - Tc|^β
        - Susceptibility: χ ~ |T - Tc|^(-γ)
        - Correlation length: ξ ~ |T - Tc|^(-ν)

        Args:
            data: Data from finite-size scaling analysis
            critical_point: Estimated critical parameter value

        Returns:
            Dictionary with critical exponents and universality class
        """
        # Extract data near critical point
        parameter_values = data["parameter_values"]

        # Use largest system size for exponent estimation
        largest_size = max(data["order_parameters"].keys())
        order_param = np.array(data["order_parameters"][largest_size])
        susceptibility = np.array(data["susceptibilities"][largest_size])

        # Focus on region near critical point
        mask = np.abs(parameter_values - critical_point) < 0.1
        if np.sum(mask) < 5:
            mask = np.abs(parameter_values - critical_point) < 0.2

        param_near_crit = parameter_values[mask]
        order_param_near = order_param[mask]
        susceptibility_near = susceptibility[mask]

        exponents = {}

        # Log-log fit for exponent β
        # log(m) = β * log(|T - Tc|) + const
        if len(param_near_crit) > 3:
            reduced_param = np.abs(param_near_crit - critical_point)
            # Avoid log(0)
            reduced_param = np.maximum(reduced_param, 1e-10)
            order_param_safe = np.maximum(order_param_near, 1e-10)

            log_reduced = np.log(reduced_param)
            log_order = np.log(order_param_safe)

            # Linear fit in log-log space
            try:
                beta_fit = np.polyfit(log_reduced, log_order, 1)
                beta = beta_fit[0]
                exponents["beta"] = beta
            except (ValueError, np.linalg.LinAlgError):
                exponents["beta"] = np.nan
        else:
            exponents["beta"] = np.nan

        # Same for susceptibility exponent γ
        if len(param_near_crit) > 3:
            susceptibility_safe = np.maximum(susceptibility_near, 1e-10)
            log_suscept = np.log(susceptibility_safe)

            try:
                gamma_fit = np.polyfit(log_reduced, log_suscept, 1)
                gamma = -gamma_fit[0]  # Negative slope
                exponents["gamma"] = gamma
            except (ValueError, np.linalg.LinAlgError):
                exponents["gamma"] = np.nan
        else:
            exponents["gamma"] = np.nan

        # Add confidence intervals (simplified bootstrap)
        exponents["beta_ci"] = self._bootstrap_exponent(
            param_near_crit, order_param_near, critical_point, "beta"
        )
        exponents["gamma_ci"] = self._bootstrap_exponent(
            param_near_crit, susceptibility_near, critical_point, "gamma"
        )

        # Check against known universality classes
        universality_class = self._classify_universality_class(exponents)

        return {
            "exponents": exponents,
            "universality_class": universality_class,
            "critical_point": critical_point,
        }

    def analyze_autocorrelation_functions(self, timeseries: Dict, max_lag: int = 100) -> Dict:
        """
        Compute autocorrelation functions to measure temporal correlations
        At criticality, should show power-law decay

        Args:
            timeseries: Time series data
            max_lag: Maximum lag to compute

        Returns:
            Dictionary with ACF analysis results
        """
        acf_results = {}

        for variable in ["S", "B", "Pi_i"]:
            if variable in timeseries:
                # Compute ACF
                data = timeseries[variable]
                acf_data = []
                for lag in range(max_lag):
                    if lag < len(data):
                        corr = (
                            np.corrcoef(data[: -lag if lag > 0 else None], data[lag:])[0, 1]
                            if lag < len(data) // 2
                            else 0
                        )
                        acf_data.append(corr)

                acf_data = np.array(acf_data)
                lags = np.arange(len(acf_data))

                # Fit to exponential decay: A(τ) ~ exp(-τ/τ_corr)
                # At criticality, becomes power-law: A(τ) ~ τ^(-α)

                # Fit exponential (only to positive correlations)
                positive_mask = acf_data > 0.01
                if np.sum(positive_mask) > 3:
                    try:
                        popt_exp, _ = curve_fit(
                            lambda t, tau: np.exp(-t / tau),
                            lags[positive_mask],
                            acf_data[positive_mask],
                            p0=[1.0],
                            maxfev=1000,
                        )
                        tau_corr = popt_exp[0]
                    except (ValueError, RuntimeError, OptimizeWarning):
                        tau_corr = np.nan
                else:
                    tau_corr = np.nan

                # Fit power-law (in log-log space, exclude lag=0)
                if len(lags) > 2:
                    log_lags = np.log(lags[1:])
                    log_acf = np.log(np.abs(acf_data[1:]) + 1e-10)

                    try:
                        alpha_fit = np.polyfit(log_lags, log_acf, 1)
                        alpha = -alpha_fit[0]
                    except (ValueError, np.linalg.LinAlgError):
                        alpha = np.nan
                else:
                    alpha = np.nan

                acf_results[variable] = {
                    "lags": lags,
                    "acf": acf_data,
                    "correlation_time": tau_corr,
                    "power_law_exponent": alpha,
                    "is_power_law": alpha > 0.5 if not np.isnan(alpha) else False,
                }

        return acf_results

    def compute_mutual_information_matrix(
        self, timeseries: Dict, n_bins: int = 20
    ) -> Tuple[np.ndarray, plt.Figure]:
        """
        Compute MI between all variable pairs
        Shows information integration structure

        Args:
            timeseries: Time series data
            n_bins: Number of bins for discretization

        Returns:
            MI matrix and figure
        """
        variables = ["S", "theta", "B", "Pi_e", "Pi_i", "eps_e", "eps_i"]
        available_vars = [v for v in variables if v in timeseries]

        n_vars = len(available_vars)
        mi_matrix = np.zeros((n_vars, n_vars))

        for i, var1 in enumerate(available_vars):
            for j, var2 in enumerate(available_vars):
                if i <= j:
                    # Discretize for MI estimation
                    x = self._discretize(timeseries[var1], n_bins)
                    y = self._discretize(timeseries[var2], n_bins)

                    mi = mutual_info_score(x, y)
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi

        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(mi_matrix, cmap="viridis")
        ax.set_xticks(range(n_vars))
        ax.set_yticks(range(n_vars))
        ax.set_xticklabels(available_vars, rotation=45)
        ax.set_yticklabels(available_vars)

        # Add values
        for i in range(n_vars):
            for j in range(n_vars):
                text = ax.text(j, i, f"{mi_matrix[i, j]:.2f}", ha="center", va="center", color="w")

        ax.set_title("Mutual Information Matrix")
        plt.colorbar(im, ax=ax)

        return mi_matrix, fig

    def compute_permutation_entropy(
        self, timeseries: Dict, embedding_dim: int = 3, delay: int = 1
    ) -> Dict:
        """
        Permutation entropy: model-free complexity measure
        Captures ordinal patterns in time series

        Reference: Bandt & Pompe (2002), PRL

        Args:
            timeseries: Time series data
            embedding_dim: Embedding dimension
            delay: Time delay

        Returns:
            Dictionary with PE results
        """
        results = {}

        for variable in ["S", "B"]:
            if variable in timeseries:
                # Embed
                embedded = self._embed_sequence(timeseries[variable], embedding_dim, delay)

                # Get patterns
                patterns = self._ordinal_patterns(embedded)

                # Count pattern frequencies
                pattern_counts = Counter(patterns)
                total = len(patterns)
                probabilities = np.array([count / total for count in pattern_counts.values()])

                # Compute entropy
                H = -np.sum(probabilities * np.log2(probabilities + 1e-10))

                # Normalized entropy (0-1)
                H_max = np.log2(math.factorial(embedding_dim))
                H_normalized = H / H_max

                results[variable] = {
                    "entropy": H,
                    "normalized_entropy": H_normalized,
                    "n_patterns": len(pattern_counts),
                    "max_patterns": math.factorial(embedding_dim),
                }

        return results

    def detrended_fluctuation_analysis(
        self, timeseries: Dict, min_window: int = 10, max_window: Optional[int] = None
    ) -> Tuple[Dict, plt.Figure]:
        """
        DFA to quantify long-range correlations
        α > 0.5 indicates long-range correlations (critical-like behavior)
        α = 0.5 is uncorrelated (white noise)
        α = 1.0 is 1/f noise (scale-free)

        Reference: Peng et al. (1994)

        Args:
            timeseries: Time series data
            min_window: Minimum window size
            max_window: Maximum window size

        Returns:
            Dictionary with DFA results and figure
        """
        data = timeseries["B"]  # Analyze ignition probability
        n = len(data)

        # Integrate the signal
        y = np.cumsum(data - np.mean(data))

        if max_window is None:
            max_window = n // 4

        # Window sizes (logarithmically spaced)
        windows = np.unique(
            np.logspace(np.log10(min_window), np.log10(max_window), num=20).astype(int)
        )

        fluctuations = []

        for window in windows:
            # Divide into non-overlapping segments
            n_segments = n // window

            if n_segments < 2:
                continue

            F_sum = 0
            for segment in range(n_segments):
                # Extract segment
                start = segment * window
                end = start + window
                segment_data = y[start:end]

                # Fit polynomial trend
                x = np.arange(len(segment_data))
                poly = np.polyfit(x, segment_data, 1)  # Linear detrending
                trend = np.polyval(poly, x)

                # Detrend and compute fluctuation
                detrended = segment_data - trend
                F_sum += np.mean(detrended**2)

            # Average fluctuation for this window size
            F = np.sqrt(F_sum / n_segments)
            fluctuations.append(F)

        # Log-log plot should be linear with slope = α
        if len(fluctuations) > 2:
            log_windows = np.log10(windows[: len(fluctuations)])
            log_fluctuations = np.log10(fluctuations)

            # Fit to get scaling exponent
            alpha = np.polyfit(log_windows, log_fluctuations, 1)[0]
        else:
            alpha = 0.5

        # Interpret
        interpretation = {
            "alpha": alpha,
            "correlation_type": (
                "Anti-correlated"
                if alpha < 0.5
                else (
                    "Uncorrelated (white noise)"
                    if abs(alpha - 0.5) < 0.05
                    else (
                        "Long-range correlated"
                        if alpha < 1.0
                        else "1/f noise" if abs(alpha - 1.0) < 0.1 else "Non-stationary"
                    )
                )
            ),
        }

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(windows[: len(fluctuations)], fluctuations, "o-", label=f"α = {alpha:.3f}")
        ax.set_xlabel("Window size")
        ax.set_ylabel("Fluctuation F(n)")
        ax.set_title(f'Detrended Fluctuation Analysis\n{interpretation["correlation_type"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return interpretation, fig

    # =====================================================================
    # Helper methods
    # =====================================================================

    def _find_binder_crossing(
        self, binder_data: Dict[int, List[float]], parameter_values: np.ndarray
    ) -> Optional[float]:
        """Find crossing point of Binder cumulants for different system sizes"""
        sizes = sorted(binder_data.keys())

        if len(sizes) < 2:
            return None

        # Find crossing between largest and second-largest systems
        size1, size2 = sizes[-2], sizes[-1]
        binder1 = np.array(binder_data[size1])
        binder2 = np.array(binder_data[size2])

        # Find minimum difference
        diff = np.abs(binder1 - binder2)
        min_idx = np.argmin(diff)

        return float(parameter_values[min_idx])

    def _bootstrap_exponent(
        self,
        params: np.ndarray,
        data: np.ndarray,
        critical_point: float,
        exponent_type: str,
        n_bootstrap: int = 100,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for exponent estimation"""
        exponents = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(params), len(params), replace=True)
            params_boot = params[indices]
            data_boot = data[indices]

            # Estimate exponent
            reduced_param = np.abs(params_boot - critical_point)
            reduced_param = np.maximum(reduced_param, 1e-10)
            data_safe = np.maximum(data_boot, 1e-10)

            try:
                if exponent_type == "beta":
                    fit = np.polyfit(np.log(reduced_param), np.log(data_safe), 1)
                    exponents.append(fit[0])
                elif exponent_type == "gamma":
                    fit = np.polyfit(np.log(reduced_param), np.log(data_safe), 1)
                    exponents.append(-fit[0])
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue

        if len(exponents) > 0:
            return (np.percentile(exponents, 2.5), np.percentile(exponents, 97.5))
        else:
            return (np.nan, np.nan)

    def _classify_universality_class(self, exponents: Dict) -> str:
        """Classify universality class based on critical exponents"""
        beta = exponents.get("beta", np.nan)
        gamma = exponents.get("gamma", np.nan)

        if np.isnan(beta) or np.isnan(gamma):
            return "Unknown"

        # Check against known universality classes
        # Mean-field: β=0.5, γ=1.0
        # 2D Ising: β=0.125, γ=1.75
        # 3D Ising: β=0.326, γ=1.237

        tolerance = 0.1

        if abs(beta - 0.5) < tolerance and abs(gamma - 1.0) < tolerance:
            return "Mean-field"
        elif abs(beta - 0.125) < tolerance and abs(gamma - 1.75) < tolerance:
            return "2D Ising"
        elif abs(beta - 0.326) < tolerance and abs(gamma - 1.237) < tolerance:
            return "3D Ising"
        else:
            return f"Custom (β={beta:.3f}, γ={gamma:.3f})"

    def _discretize(self, data: np.ndarray, n_bins: int) -> np.ndarray:
        """Discretize continuous data into bins"""
        data = data.reshape(-1, 1)
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
        return discretizer.fit_transform(data).astype(int).flatten()

    def _embed_sequence(self, data: np.ndarray, dim: int, tau: int) -> np.ndarray:
        """Create delay embedding"""
        n = len(data)
        embedded = np.array([data[i : i + dim * tau : tau] for i in range(n - dim * tau + 1)])
        return embedded

    def _ordinal_patterns(self, embedded: np.ndarray) -> List[Tuple]:
        """Extract ordinal patterns (permutations)"""
        patterns = []
        for row in embedded:
            pattern = tuple(np.argsort(row))
            patterns.append(pattern)
        return patterns


class ComprehensivePhaseTransitionAnalysis:
    """
    Integrate all analyses to test APGI phase transition predictions
    """

    def __init__(self):
        self.info_analyzer = InformationTheoreticAnalysis(n_bins=20)
        self.phase_detector = PhaseTransitionDetector()
        self.scaling_analyzer = FiniteSizeScalingAnalysis()

    def analyze_simulation(self, history: Dict) -> Dict:
        """
        Comprehensive analysis of single simulation

        Args:
            history: Simulation history from APGIDynamicalSystem

        Returns:
            Dictionary with all measures
        """

        results = {}

        S = history["S"]
        theta = history["theta"]
        B = history["B"]
        time = history["time"]
        ignition_events = history["ignition_events"]

        # =====================================================================
        # Information-Theoretic Measures
        # =====================================================================

        # Transfer entropy: external input → surprise
        if len(history["eps_e"]) > 20:
            te_input_to_S = self.info_analyzer.compute_transfer_entropy(history["eps_e"], S, lag=2)
            results["te_input_to_S"] = float(te_input_to_S)

        # Transfer entropy: surprise → ignition probability
        if len(S) > 20:
            te_S_to_B = self.info_analyzer.compute_transfer_entropy(S, B, lag=1)
            results["te_S_to_B"] = float(te_S_to_B)

        # Integrated information
        if len(S) > 100:
            phi = self.info_analyzer.compute_integrated_information(
                [S, theta, history["Pi_i"]], window_size=50
            )

            # Φ at ignition vs baseline
            if len(ignition_events) > 0 and len(phi) > 0:
                ignition_phi = []
                for idx in ignition_events:
                    phi_idx = idx - 50
                    if 0 <= phi_idx < len(phi):
                        ignition_phi.append(phi[phi_idx])

                if len(ignition_phi) > 0:
                    results["phi_at_ignition"] = float(np.mean(ignition_phi))

                    # Baseline (non-ignition)
                    non_ignition_mask = B < 0.3
                    baseline_indices = np.where(non_ignition_mask)[0]
                    baseline_indices = baseline_indices[baseline_indices < len(phi)]

                    if len(baseline_indices) > 0:
                        results["phi_baseline"] = float(np.mean(phi[baseline_indices]))
                        results["phi_ratio"] = results["phi_at_ignition"] / (
                            results["phi_baseline"] + 1e-10
                        )

        # Mutual information: S and theta
        if len(S) > 50:
            mi_S_theta = self.info_analyzer.compute_mutual_information(S, theta)
            results["mi_S_theta"] = float(mi_S_theta)

        # Entropy rate
        if len(S) > 30:
            entropy_rate = self.info_analyzer.compute_entropy_rate(S, lag=3)
            results["entropy_rate_S"] = float(entropy_rate)

        # =====================================================================
        # Phase Transition Signatures
        # =====================================================================

        # Discontinuity
        if len(ignition_events) > 0:
            disc_results = self.phase_detector.detect_discontinuity(S, theta, time, ignition_events)
            results.update({f"discontinuity_{k}": v for k, v in disc_results.items()})

        # Susceptibility
        susc_results = self.phase_detector.compute_susceptibility(S, theta)
        results.update({f"susceptibility_{k}": v for k, v in susc_results.items()})

        # Critical slowing
        crit_slow_results = self.phase_detector.detect_critical_slowing(S, theta)
        results.update({f"critical_slowing_{k}": v for k, v in crit_slow_results.items()})

        # Hurst exponent
        hurst_results = self.phase_detector.compute_hurst_exponent(S, theta)
        results.update({f"hurst_{k}": v for k, v in hurst_results.items()})

        return results

    def run_monte_carlo(
        self,
        n_simulations: int = 100,
        duration: float = 100.0,
        save_results: bool = True,
    ) -> pd.DataFrame:
        """
        Run Monte Carlo analysis across many simulations

        Args:
            n_simulations: Number of simulations
            duration: Duration per simulation (seconds)
            save_results: Whether to save results to file

        Returns:
            DataFrame with all results
        """

        print(f"\n{'=' * 80}")
        print("RUNNING MONTE CARLO PHASE TRANSITION ANALYSIS")
        print(f"{'=' * 80}")
        print(f"Simulations: {n_simulations}")
        print(f"Duration per simulation: {duration}s")

        all_results = []

        for sim_idx in tqdm(range(n_simulations), desc="Simulations"):
            # Create system
            system = APGIDynamicalSystem(
                tau=np.random.uniform(0.15, 0.25),
                theta_0=np.random.uniform(0.45, 0.65),
                alpha=np.random.uniform(4.0, 6.0),
                dt=0.01,
            )

            # Time-varying input generator
            def input_gen(t):
                # Oscillating external input
                Pi_e = 1.0 + 0.3 * np.sin(2 * np.pi * t / 15)
                eps_e = np.random.normal(0.4, 0.2)

                # Varying interoceptive signals
                Pi_i = 1.0 + 0.5 * np.sin(2 * np.pi * t / 25)
                eps_i = np.random.normal(0.2, 0.15)

                beta = np.random.normal(1.15, 0.15)

                return {
                    "Pi_e": max(0.5, Pi_e),
                    "eps_e": eps_e,
                    "beta": max(0.7, min(1.8, beta)),
                    "Pi_i": max(0.5, Pi_i),
                    "eps_i": eps_i,
                    "M": 1.0,
                    "c": 0.5,
                    "a": 0.3,
                }

            # Run simulation
            history = system.simulate(duration, input_gen, theta_noise_sd=0.08)

            # Analyze
            results = self.analyze_simulation(history)
            results["simulation_id"] = sim_idx
            results["n_ignition_events"] = len(history["ignition_events"])

            all_results.append(results)

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        print("\n✅ Analysis complete")
        print(f"   Total simulations: {len(df)}")
        print(f"   Mean ignition events: {df['n_ignition_events'].mean():.1f}")

        if save_results:
            df.to_csv("protocol4_monte_carlo_results.csv", index=False)
            print("   Results saved to: protocol4_monte_carlo_results.csv")

        return df


# =============================================================================
# PART 5: FALSIFICATION CRITERIA
# =============================================================================


class FalsificationChecker:
    """Check Protocol 4 falsification criteria"""

    def __init__(self):
        self.criteria = {
            "F4.1": {
                "description": "Susceptibility ratio < 1.2 (no phase transition)",
                "threshold": 1.2,
                "comparison": "less_than",
            },
            "F4.2": {
                "description": "Φ at ignition < 1.3× baseline (not informationally distinct)",
                "threshold": 1.3,
                "comparison": "less_than",
            },
            "F4.3": {
                "description": "Critical slowing ratio < 1.2 (continuous, not discrete)",
                "threshold": 1.2,
                "comparison": "less_than",
            },
            "F4.4": {
                "description": "Discontinuity effect size d < 0.5 (no sharp transition)",
                "threshold": 0.5,
                "comparison": "less_than",
            },
            "F4.5": {
                "description": "Hurst near threshold < 0.55 (no long-range correlations)",
                "threshold": 0.55,
                "comparison": "less_than",
            },
        }

    def check_all_criteria(self, results_df: pd.DataFrame) -> Dict:
        """
        Check all falsification criteria

        Args:
            results_df: DataFrame from Monte Carlo analysis

        Returns:
            Comprehensive falsification report
        """

        report = {
            "falsified_criteria": [],
            "passed_criteria": [],
            "overall_falsified": False,
        }

        # F4.1: Susceptibility
        susc_ratio = results_df["susceptibility_susceptibility_ratio"].mean()
        susc_se = results_df["susceptibility_susceptibility_ratio"].sem()

        f4_1_falsified = susc_ratio < self.criteria["F4.1"]["threshold"]

        criterion = {
            "code": "F4.1",
            "description": self.criteria["F4.1"]["description"],
            "falsified": f4_1_falsified,
            "value": float(susc_ratio),
            "se": float(susc_se),
            "threshold": self.criteria["F4.1"]["threshold"],
            "n": len(results_df),
        }

        if f4_1_falsified:
            report["falsified_criteria"].append(criterion)
        else:
            report["passed_criteria"].append(criterion)

        # F4.2: Integrated information
        if "phi_ratio" in results_df.columns:
            phi_ratio = results_df["phi_ratio"].dropna().mean()
            phi_se = results_df["phi_ratio"].dropna().sem()

            f4_2_falsified = phi_ratio < self.criteria["F4.2"]["threshold"]

            criterion = {
                "code": "F4.2",
                "description": self.criteria["F4.2"]["description"],
                "falsified": f4_2_falsified,
                "value": float(phi_ratio),
                "se": float(phi_se),
                "threshold": self.criteria["F4.2"]["threshold"],
            }

            if f4_2_falsified:
                report["falsified_criteria"].append(criterion)
            else:
                report["passed_criteria"].append(criterion)

        # F4.3: Critical slowing
        crit_slow_ratio = results_df["critical_slowing_critical_slowing_ratio"].mean()
        crit_slow_se = results_df["critical_slowing_critical_slowing_ratio"].sem()

        f4_3_falsified = crit_slow_ratio < self.criteria["F4.3"]["threshold"]

        criterion = {
            "code": "F4.3",
            "description": self.criteria["F4.3"]["description"],
            "falsified": f4_3_falsified,
            "value": float(crit_slow_ratio),
            "se": float(crit_slow_se),
            "threshold": self.criteria["F4.3"]["threshold"],
        }

        if f4_3_falsified:
            report["falsified_criteria"].append(criterion)
        else:
            report["passed_criteria"].append(criterion)

        # F4.4: Discontinuity
        if "discontinuity_cohens_d" in results_df.columns:
            disc_d = results_df["discontinuity_cohens_d"].dropna().mean()
            disc_se = results_df["discontinuity_cohens_d"].dropna().sem()

            f4_4_falsified = disc_d < self.criteria["F4.4"]["threshold"]

            criterion = {
                "code": "F4.4",
                "description": self.criteria["F4.4"]["description"],
                "falsified": f4_4_falsified,
                "value": float(disc_d),
                "se": float(disc_se),
                "threshold": self.criteria["F4.4"]["threshold"],
            }

            if f4_4_falsified:
                report["falsified_criteria"].append(criterion)
            else:
                report["passed_criteria"].append(criterion)

        # F4.5: Hurst exponent
        hurst_near = results_df["hurst_hurst_near"].mean()
        hurst_se = results_df["hurst_hurst_near"].sem()

        f4_5_falsified = hurst_near < self.criteria["F4.5"]["threshold"]

        criterion = {
            "code": "F4.5",
            "description": self.criteria["F4.5"]["description"],
            "falsified": f4_5_falsified,
            "value": float(hurst_near),
            "se": float(hurst_se),
            "threshold": self.criteria["F4.5"]["threshold"],
        }

        if f4_5_falsified:
            report["falsified_criteria"].append(criterion)
        else:
            report["passed_criteria"].append(criterion)

        # Overall verdict
        report["overall_falsified"] = len(report["falsified_criteria"]) > 0

        return report


def print_falsification_report(report: Dict):
    """Print formatted falsification report"""

    print("\n" + "=" * 80)
    print("PROTOCOL 4 FALSIFICATION REPORT")
    print("=" * 80)

    print("\nOVERALL STATUS: ", end="")
    if report["overall_falsified"]:
        print("❌ MODEL FALSIFIED")
    else:
        print("✅ MODEL VALIDATED")

    total = len(report["passed_criteria"]) + len(report["falsified_criteria"])
    print(f"\nCriteria Passed: {len(report['passed_criteria'])}/{total}")
    print(f"Criteria Failed: {len(report['falsified_criteria'])}/{total}")

    if report["passed_criteria"]:
        print("\n" + "-" * 80)
        print("PASSED CRITERIA:")
        print("-" * 80)
        for criterion in report["passed_criteria"]:
            print(f"\n✅ {criterion['code']}: {criterion['description']}")
            if "value" in criterion:
                print(f"   Value: {criterion['value']:.4f} ± {criterion.get('se', 0):.4f}")
                print(f"   Threshold: {criterion['threshold']}")

    if report["falsified_criteria"]:
        print("\n" + "-" * 80)
        print("FAILED CRITERIA (FALSIFICATIONS):")
        print("-" * 80)
        for criterion in report["falsified_criteria"]:
            print(f"\n❌ {criterion['code']}: {criterion['description']}")
            if "value" in criterion:
                print(f"   Value: {criterion['value']:.4f} ± {criterion.get('se', 0):.4f}")
                print(f"   Threshold: {criterion['threshold']}")

    print("\n" + "=" * 80)


# =============================================================================
# PART 6: VISUALIZATION
# =============================================================================


def plot_phase_transition_results(
    results_df: pd.DataFrame,
    example_history: Optional[Dict] = None,
    save_path: str = "protocol4_results.png",
):
    """Generate comprehensive visualization of phase transition analysis"""

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)

    # ==========================================================================
    # Row 1: Example Time Series (if provided)
    # ==========================================================================

    if example_history is not None:
        time = example_history["time"]
        S = example_history["S"]
        theta = example_history["theta"]
        B = example_history["B"]
        ignition_events = example_history["ignition_events"]

        ax1 = fig.add_subplot(gs[0, :3])

        ax1.plot(time, S, "b-", linewidth=2, label="Surprise (S)", alpha=0.8)
        ax1.plot(time, theta, "r--", linewidth=2, label="Threshold (θ)", alpha=0.8)
        ax1.fill_between(time, 0, B * 10, color="green", alpha=0.2, label="Ignition (B)")

        # Mark ignition events
        for idx in ignition_events:
            if idx < len(time):
                ax1.axvline(time[idx], color="orange", linestyle=":", linewidth=1.5, alpha=0.6)

        ax1.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Magnitude", fontsize=12, fontweight="bold")
        ax1.set_title(
            "Example APGI Dynamics: Surprise Accumulation and Ignition",
            fontsize=14,
            fontweight="bold",
        )
        ax1.legend(fontsize=11, loc="upper right")
        ax1.grid(alpha=0.3)

        # Zoom inset
        ax1_inset = fig.add_subplot(gs[0, 3])

        if len(ignition_events) > 0:
            zoom_idx = ignition_events[0]
            zoom_start = max(0, zoom_idx - 50)
            zoom_end = min(len(time), zoom_idx + 50)

            ax1_inset.plot(time[zoom_start:zoom_end], S[zoom_start:zoom_end], "b-", linewidth=2)
            ax1_inset.plot(
                time[zoom_start:zoom_end],
                theta[zoom_start:zoom_end],
                "r--",
                linewidth=2,
            )
            ax1_inset.axvline(time[zoom_idx], color="orange", linestyle=":", linewidth=2)
            ax1_inset.set_title("Ignition Event (Zoomed)", fontsize=10, fontweight="bold")
            ax1_inset.grid(alpha=0.3)

    # ==========================================================================
    # Row 2: Information-Theoretic Measures
    # ==========================================================================

    # Integrated Information (Φ)
    ax2 = fig.add_subplot(gs[1, 0])

    phi_at_ignition = results_df["phi_at_ignition"].dropna()
    phi_baseline = results_df["phi_baseline"].dropna()

    if len(phi_at_ignition) > 0 and len(phi_baseline) > 0:
        positions = [1, 2]
        data = [phi_baseline, phi_at_ignition]

        bp = ax2.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showmeans=True)

        for patch, color in zip(bp["boxes"], ["lightblue", "salmon"]):
            patch.set_facecolor(color)

        ax2.set_xticks(positions)
        ax2.set_xticklabels(["Baseline", "At Ignition"])
        ax2.set_ylabel("Φ (Integrated Information)", fontsize=11, fontweight="bold")
        ax2.set_title("P4d: Φ Spike at Ignition", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        # Add significance
        mean_baseline = phi_baseline.mean()
        mean_ignition = phi_at_ignition.mean()
        ratio = mean_ignition / mean_baseline

        ax2.text(
            0.5,
            0.95,
            f"Ratio: {ratio:.2f}×",
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Transfer Entropy
    ax3 = fig.add_subplot(gs[1, 1])

    if "te_S_to_B" in results_df.columns:
        te_values = results_df["te_S_to_B"].dropna()

        ax3.hist(
            te_values,
            bins=30,
            density=True,
            alpha=0.7,
            color="purple",
            edgecolor="black",
        )

        mean_te = te_values.mean()
        ax3.axvline(
            mean_te,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_te:.3f}",
        )

        ax3.set_xlabel("Transfer Entropy (S → B)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax3.set_title("Information Flow: Surprise → Ignition", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

    # Mutual Information
    ax4 = fig.add_subplot(gs[1, 2])

    if "mi_S_theta" in results_df.columns:
        mi_values = results_df["mi_S_theta"].dropna()

        ax4.hist(mi_values, bins=30, density=True, alpha=0.7, color="teal", edgecolor="black")

        mean_mi = mi_values.mean()
        ax4.axvline(
            mean_mi,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_mi:.3f}",
        )

        ax4.set_xlabel("Mutual Information I(S; θ)", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax4.set_title("Statistical Dependence", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)

    # Summary statistics
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.axis("off")

    summary_text = f"""
    INFORMATION-THEORETIC SUMMARY
    {'=' * 35}

    Integrated Information (Φ):
      Baseline: {phi_baseline.mean():.3f} ± {phi_baseline.std():.3f}
      At Ignition: {phi_at_ignition.mean():.3f} ± {phi_at_ignition.std():.3f}
      Ratio: {(phi_at_ignition.mean() / phi_baseline.mean()):.2f}×

    Transfer Entropy (S→B):
      Mean: {results_df['te_S_to_B'].mean():.4f}

    Mutual Information (S;θ):
      Mean: {results_df['mi_S_theta'].mean():.4f}
    """

    ax5.text(
        0.1,
        0.5,
        summary_text,
        fontsize=9,
        family="monospace",
        verticalalignment="center",
    )

    # ==========================================================================
    # Row 3: Phase Transition Signatures
    # ==========================================================================

    # Discontinuity
    ax6 = fig.add_subplot(gs[2, 0])

    if "discontinuity_cohens_d" in results_df.columns:
        disc_d = results_df["discontinuity_cohens_d"].dropna()

        ax6.hist(disc_d, bins=30, density=True, alpha=0.7, color="orange", edgecolor="black")

        mean_d = disc_d.mean()
        ax6.axvline(
            mean_d,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean d: {mean_d:.2f}",
        )
        ax6.axvline(0.8, color="green", linestyle=":", linewidth=2, label="Large effect (d=0.8)")

        ax6.set_xlabel("Cohen's d (Discontinuity)", fontsize=11, fontweight="bold")
        ax6.set_ylabel("Density", fontsize=11, fontweight="bold")
        ax6.set_title("P4a: Discontinuity at Ignition", fontsize=12, fontweight="bold")
        ax6.legend(fontsize=9)
        ax6.grid(alpha=0.3)

    # Susceptibility
    ax7 = fig.add_subplot(gs[2, 1])

    susc_ratio = results_df["susceptibility_susceptibility_ratio"].dropna()

    ax7.hist(susc_ratio, bins=30, density=True, alpha=0.7, color="red", edgecolor="black")

    mean_susc = susc_ratio.mean()
    ax7.axvline(
        mean_susc,
        color="darkred",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_susc:.2f}",
    )
    ax7.axvline(2.0, color="green", linestyle=":", linewidth=2, label="Prediction: >2.0")

    ax7.set_xlabel("Susceptibility Ratio (near/far)", fontsize=11, fontweight="bold")
    ax7.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax7.set_title("P4b: Diverging Susceptibility", fontsize=12, fontweight="bold")
    ax7.legend(fontsize=9)
    ax7.grid(alpha=0.3)

    # Critical Slowing
    ax8 = fig.add_subplot(gs[2, 2])

    crit_slow = results_df["critical_slowing_critical_slowing_ratio"].dropna()

    ax8.hist(crit_slow, bins=30, density=True, alpha=0.7, color="blue", edgecolor="black")

    mean_cs = crit_slow.mean()
    ax8.axvline(
        mean_cs,
        color="darkblue",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_cs:.2f}",
    )
    ax8.axvline(1.5, color="green", linestyle=":", linewidth=2, label="Prediction: >1.5")

    ax8.set_xlabel("Critical Slowing Ratio", fontsize=11, fontweight="bold")
    ax8.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax8.set_title("P4c: Critical Slowing Down", fontsize=12, fontweight="bold")
    ax8.legend(fontsize=9)
    ax8.grid(alpha=0.3)

    # Hurst Exponent
    ax9 = fig.add_subplot(gs[2, 3])

    hurst_near = results_df["hurst_hurst_near"].dropna()
    hurst_far = results_df["hurst_hurst_far"].dropna()

    positions = [1, 2]
    data = [hurst_far, hurst_near]

    bp = ax9.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showmeans=True)

    for patch, color in zip(bp["boxes"], ["lightblue", "salmon"]):
        patch.set_facecolor(color)

    ax9.axhline(0.5, color="black", linestyle="--", linewidth=1.5, label="Random walk (H=0.5)")
    ax9.axhline(0.6, color="green", linestyle=":", linewidth=2, label="Prediction: H>0.6")

    ax9.set_xticks(positions)
    ax9.set_xticklabels(["Far", "Near Threshold"])
    ax9.set_ylabel("Hurst Exponent", fontsize=11, fontweight="bold")
    ax9.set_title("P4e: Long-Range Correlations", fontsize=12, fontweight="bold")
    ax9.legend(fontsize=8)
    ax9.grid(axis="y", alpha=0.3)

    # ==========================================================================
    # Row 4: Summary & Predictions
    # ==========================================================================

    # Prediction summary table
    ax10 = fig.add_subplot(gs[3, :2])
    ax10.axis("off")

    predictions = [
        ["Prediction", "Criterion", "Observed", "Met?"],
        [
            "P4a: Discontinuity",
            "d > 0.8",
            f"{results_df['discontinuity_cohens_d'].mean():.2f}",
            "✅" if results_df["discontinuity_cohens_d"].mean() > 0.8 else "❌",
        ],
        [
            "P4b: Susceptibility",
            "ratio > 2.0",
            f"{results_df['susceptibility_susceptibility_ratio'].mean():.2f}",
            ("✅" if results_df["susceptibility_susceptibility_ratio"].mean() > 2.0 else "❌"),
        ],
        [
            "P4c: Crit. Slowing",
            "ratio > 1.5",
            f"{results_df['critical_slowing_critical_slowing_ratio'].mean():.2f}",
            ("✅" if results_df["critical_slowing_critical_slowing_ratio"].mean() > 1.5 else "❌"),
        ],
        [
            "P4d: Φ Spike",
            "ratio > 2.0×",
            f"{(phi_at_ignition.mean() / phi_baseline.mean()):.2f}×",
            "✅" if (phi_at_ignition.mean() / phi_baseline.mean()) > 2.0 else "❌",
        ],
        [
            "P4e: Hurst Near",
            "H > 0.6",
            f"{results_df['hurst_hurst_near'].mean():.2f}",
            "✅" if results_df["hurst_hurst_near"].mean() > 0.6 else "❌",
        ],
    ]

    table = ax10.table(
        cellText=predictions,
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.2, 0.2, 0.1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax10.set_title("Prediction Summary", fontsize=13, fontweight="bold", pad=20)

    # Correlation matrix
    ax11 = fig.add_subplot(gs[3, 2:])

    measures = [
        "susceptibility_susceptibility_ratio",
        "critical_slowing_critical_slowing_ratio",
        "hurst_hurst_near",
        "discontinuity_cohens_d",
    ]

    measure_labels = ["Suscept.", "Crit.Slow", "Hurst", "Discont."]

    corr_data = results_df[measures].dropna()

    if len(corr_data) > 10:
        corr_matrix = corr_data.corr()

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax11,
            xticklabels=measure_labels,
            yticklabels=measure_labels,
            cbar_kws={"label": "Correlation"},
        )

        ax11.set_title("Phase Transition Measure Correlations", fontsize=12, fontweight="bold")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Visualization saved to: {save_path}")
    plt.show()


# =============================================================================
# PART 7: MAIN EXECUTION PIPELINE
# =============================================================================


def main():
    """Main execution pipeline for Protocol 4"""

    print("=" * 80)
    print("APGI PROTOCOL 4: INFORMATION-THEORETIC PHASE TRANSITION ANALYSIS")
    print("=" * 80)

    # Configuration
    config = {"n_simulations": 100, "duration": 100.0, "dt": 0.01}

    print("\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # =========================================================================
    # STEP 1: Run Single Example Simulation
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING EXAMPLE SIMULATION")
    print("=" * 80)

    system = APGIDynamicalSystem(tau=0.2, theta_0=0.55, alpha=5.0, dt=config["dt"])

    def example_input_gen(t):
        return {
            "Pi_e": 1.0 + 0.3 * np.sin(2 * np.pi * t / 15),
            "eps_e": np.random.normal(0.4, 0.2),
            "beta": 1.15,
            "Pi_i": 1.0 + 0.5 * np.sin(2 * np.pi * t / 25),
            "eps_i": np.random.normal(0.2, 0.15),
            "M": 1.0,
            "c": 0.5,
            "a": 0.3,
        }

    example_history = system.simulate(config["duration"], example_input_gen, theta_noise_sd=0.08)

    print("\n✅ Example simulation complete")
    print(f"   Duration: {config['duration']}s")
    print(f"   Ignition events: {len(example_history['ignition_events'])}")

    # =========================================================================
    # STEP 2: Analyze Example
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: ANALYZING EXAMPLE SIMULATION")
    print("=" * 80)

    analyzer = ComprehensivePhaseTransitionAnalysis()
    example_results = analyzer.analyze_simulation(example_history)

    print("\nKey Results:")
    if "phi_ratio" in example_results:
        print(f"  Φ ratio (ignition/baseline): {example_results['phi_ratio']:.2f}×")
    if "susceptibility_susceptibility_ratio" in example_results:
        print(
            f"  Susceptibility ratio: {example_results['susceptibility_susceptibility_ratio']:.2f}"
        )
    if "critical_slowing_critical_slowing_ratio" in example_results:
        print(
            f"  Critical slowing ratio: {example_results['critical_slowing_critical_slowing_ratio']:.2f}"
        )
    if "hurst_hurst_near" in example_results:
        print(f"  Hurst exponent (near threshold): {example_results['hurst_hurst_near']:.2f}")

    # =========================================================================
    # STEP 3: Monte Carlo Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: MONTE CARLO ANALYSIS")
    print("=" * 80)

    results_df = analyzer.run_monte_carlo(
        n_simulations=config["n_simulations"],
        duration=config["duration"],
        save_results=True,
    )

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 80)

    summary_measures = {
        "Φ Ratio": "phi_ratio",
        "Susceptibility Ratio": "susceptibility_susceptibility_ratio",
        "Critical Slowing Ratio": "critical_slowing_critical_slowing_ratio",
        "Hurst (Near)": "hurst_hurst_near",
        "Hurst (Far)": "hurst_hurst_far",
        "Discontinuity (Cohen's d)": "discontinuity_cohens_d",
    }

    for name, col in summary_measures.items():
        if col in results_df.columns:
            data = results_df[col].dropna()
            if len(data) > 0:
                print(f"{name:30s}: {data.mean():7.3f} ± {data.std():6.3f}")

    # =========================================================================
    # STEP 4: Falsification Testing
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: FALSIFICATION TESTING")
    print("=" * 80)

    checker = FalsificationChecker()
    falsification_report = checker.check_all_criteria(results_df)

    print_falsification_report(falsification_report)

    # =========================================================================
    # STEP 5: Visualization
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_phase_transition_results(
        results_df, example_history=example_history, save_path="protocol4_results.png"
    )

    # =========================================================================
    # STEP 6: Save Results
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: SAVING RESULTS")
    print("=" * 80)

    # Prepare summary
    results_summary = {
        "config": config,
        "n_simulations": len(results_df),
        "summary_statistics": {
            name: {
                "mean": float(results_df[col].dropna().mean()),
                "std": float(results_df[col].dropna().std()),
                "median": float(results_df[col].dropna().median()),
            }
            for name, col in summary_measures.items()
            if col in results_df.columns and len(results_df[col].dropna()) > 0
        },
        "falsification": falsification_report,
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

    with open("protocol4_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("✅ Results saved to: protocol4_results.json")

    print("\n" + "=" * 80)
    print("PROTOCOL 4 EXECUTION COMPLETE")
    print("=" * 80)

    return results_summary


def run_validation():
    """Entry point for CLI validation."""
    try:
        print("Running APGI Validation Protocol 4: Cross-Modal Replication and Meta-Analysis")
        return main()
    except (RuntimeError, ValueError, TypeError, ImportError, KeyError) as e:
        print(f"Error in validation protocol 4: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    main()
