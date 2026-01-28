import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import entropy

# Constants for better maintainability
DEFAULT_SURPRISE_THRESHOLD = 0.5
DEFAULT_ALPHA = 8.0
DEFAULT_TAU_S = 0.3
DEFAULT_TAU_THETA = 10.0
DEFAULT_ETA_THETA = 0.01
DEFAULT_BETA = 1.2
DEFAULT_SURPRISE_RESET_FACTOR = 0.3
DEFAULT_N_BINS = 20
DEFAULT_WINDOW_SIZE = 50
DEFAULT_MIN_LAG = 10
DEFAULT_MIN_SAMPLES = 20
DEFAULT_MIN_SUBSERIES = 10
DEFAULT_HURST_LAG_MULTIPLIER = 4
DEFAULT_SIMULATION_DURATION = 50.0
DEFAULT_DT = 0.1
DEFAULT_N_SIMULATIONS = 10
DEFAULT_PRINT_INTERVAL = 5
DEFAULT_AC_LAG = 5
DEFAULT_NEAR_THRESHOLD = 0.2
DEFAULT_FAR_THRESHOLD = 0.5
DEFAULT_IGNITION_THRESHOLD = 0.5
DEFAULT_DISCONTINUITY_WINDOW = 10
DEFAULT_PHI_LOOKBACK = 50
DEFAULT_HISTOGRAM_BINS = 10
DEFAULT_EPSILON = 1e-10

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SurpriseIgnitionSystem:
    """Surprise accumulation and ignition system for APGI"""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        tau_S: float = DEFAULT_TAU_S,
        tau_theta: float = DEFAULT_TAU_THETA,
        eta_theta: float = DEFAULT_ETA_THETA,
        beta: float = DEFAULT_BETA,
        theta_0: float = DEFAULT_SURPRISE_THRESHOLD,
        random_seed: Optional[int] = None,
    ):
        # System parameters
        self.S_t = 0.0  # Current surprise level
        self.theta_t = theta_0  # Current threshold
        self.theta_0 = theta_0  # Baseline threshold
        self.alpha = alpha  # Sigmoid steepness
        self.tau_S = tau_S  # Surprise time constant
        self.tau_theta = tau_theta  # Threshold time constant
        self.eta_theta = eta_theta  # Threshold adaptation rate
        self.beta = beta  # Somatic bias

        # State tracking
        self.time = 0.0
        self.ignition_states = []

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate(
        self,
        duration: float = DEFAULT_SIMULATION_DURATION,
        dt: float = DEFAULT_DT,
        input_generator=None,
    ) -> Dict[str, np.ndarray]:
        """Simulate the system for given duration

        Args:
            duration: Simulation duration in time units
            dt: Time step size
            input_generator: Function that returns dict with Pi_e, Pi_i, eps_e, eps_i, beta

        Returns:
            Dictionary containing time series of system variables
        """
        if input_generator is None:
            raise ValueError("input_generator must be provided")

        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}")

        n_steps = int(duration / dt)
        if n_steps < 1:
            raise ValueError(f"duration ({duration}) / dt ({dt}) results in < 1 step")

        history = {
            "time": [],
            "S": [],
            "theta": [],
            "B": [],  # Ignition states
            "Pi_e": [],
            "Pi_i": [],
            "eps_e": [],
            "eps_i": [],
        }

        for step in range(n_steps):
            t = step * dt

            try:
                # Get inputs
                inputs = input_generator(t)
                Pi_e = inputs["Pi_e"]
                Pi_i = inputs["Pi_i"]
                eps_e = inputs["eps_e"]
                eps_i = inputs["eps_i"]
                beta = inputs["beta"]
            except (KeyError, TypeError) as e:
                raise ValueError(f"input_generator returned invalid data: {e}")

            # Compute input drive
            input_drive = Pi_e * eps_e + beta * Pi_i * eps_i

            # Update surprise
            dS_dt = -self.S_t / self.tau_S + input_drive
            self.S_t += dS_dt * dt
            self.S_t = max(0.0, self.S_t)

            # Update threshold (simplified)
            dtheta_dt = (self.theta_0 - self.theta_t) / self.tau_theta
            self.theta_t += dtheta_dt * dt
            self.theta_t = np.clip(self.theta_t, 0.1, 2.0)

            # Check ignition
            ignition_prob = 1.0 / (1.0 + np.exp(-self.alpha * (self.S_t - self.theta_t)))
            ignition = np.random.random() < ignition_prob

            # Store history
            history["time"].append(t)
            history["S"].append(self.S_t)
            history["theta"].append(self.theta_t)
            history["B"].append(1.0 if ignition else 0.0)
            history["Pi_e"].append(Pi_e)
            history["Pi_i"].append(Pi_i)
            history["eps_e"].append(eps_e)
            history["eps_i"].append(eps_i)

            if ignition:
                # Partial reset of surprise
                self.S_t *= DEFAULT_SURPRISE_RESET_FACTOR

        # Convert to arrays
        for key in history.keys():
            history[key] = np.array(history[key])

        return history


class InformationTheoreticAnalysis:
    """
    Test whether APGI ignition exhibits phase transition signatures
    """

    def __init__(self, apgi_system: SurpriseIgnitionSystem):
        self.system = apgi_system

    def compute_transfer_entropy(
        self, history: Dict[str, np.ndarray], source: str, target: str, lag: int = 1
    ) -> np.ndarray:
        """
        Transfer entropy: Information flow from source to target

        TE(X→Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})

        Prediction: TE increases dramatically at ignition

        Args:
            history: Dictionary containing time series data
            source: Source variable name
            target: Target variable name
            lag: Time lag for transfer entropy calculation

        Returns:
            Array of transfer entropy values
        """
        if source not in history or target not in history:
            raise KeyError(f"Source '{source}' or target '{target}' not in history")

        if lag < 1:
            raise ValueError(f"lag must be >= 1, got {lag}")

        X = history[source]
        Y = history[target]
        B = history["B"]  # Ignition states

        if len(X) < lag + 1:
            raise ValueError(f"Data length ({len(X)}) < lag + 1 ({lag + 1})")

        # Discretize for MI estimation
        n_bins = DEFAULT_N_BINS
        X_min, X_max = X.min(), X.max()
        Y_min, Y_max = Y.min(), Y.max()

        # Handle constant data
        if X_max == X_min or Y_max == Y_min:
            return np.zeros(len(X) - lag)

        X_binned = np.digitize(X, np.linspace(X_min, X_max, n_bins))
        Y_binned = np.digitize(Y, np.linspace(Y_min, Y_max, n_bins))

        te_values = np.zeros(len(X) - lag)

        for t in range(lag, len(X)):
            # H(Y_t | Y_{t-lag})
            p_Y_given_Ypast = self._conditional_prob(Y_binned[t], Y_binned[t - lag], n_bins)
            H_Y_given_Ypast = entropy(p_Y_given_Ypast)

            # H(Y_t | Y_{t-lag}, X_{t-lag})
            p_Y_given_both = self._conditional_prob_joint(
                Y_binned[t], Y_binned[t - lag], X_binned[t - lag], n_bins
            )
            H_Y_given_both = entropy(p_Y_given_both)

            te_values[t - lag] = H_Y_given_Ypast - H_Y_given_both

        return te_values

    def compute_integrated_information(
        self, history: Dict[str, np.ndarray], window_size: int = DEFAULT_WINDOW_SIZE
    ) -> np.ndarray:
        """
        Φ-like measure: How much more information is in the whole
        than in the parts

        Prediction: Φ spikes at ignition

        Args:
            history: Dictionary containing time series data
            window_size: Size of sliding window for analysis

        Returns:
            Array of phi values
        """
        S = history["S"]
        theta = history["theta"]
        B = history["B"]

        if len(S) < window_size:
            raise ValueError(f"Data length ({len(S)}) < window_size ({window_size})")

        phi_values = np.zeros(len(S) - window_size)

        for t in range(window_size, len(S)):
            window_S = S[t - window_size : t]
            window_theta = theta[t - window_size : t]

            # Joint entropy
            joint_data = np.column_stack([window_S, window_theta])
            H_joint = self._estimate_entropy(joint_data)

            # Sum of marginal entropies
            H_S = self._estimate_entropy(window_S)
            H_theta = self._estimate_entropy(window_theta)

            # Φ ≈ sum of marginals - joint (mutual information)
            phi_values[t - window_size] = H_S + H_theta - H_joint

        return phi_values

    def detect_phase_transition(self, history: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect signatures of phase transition at ignition

        Signatures:
        1. Discontinuity in first derivative of S
        2. Diverging susceptibility (variance)
        3. Critical slowing down before transition
        4. Long-range correlations at criticality

        Args:
            history: Dictionary containing time series data

        Returns:
            Dictionary containing phase transition metrics
        """
        S = history["S"]
        theta = history["theta"]
        B = history["B"]
        time = history["time"]

        results = {}

        # 1. Discontinuity analysis
        dS_dt = np.gradient(S, time)
        # Note: d2S_dt2 is computed but not used - keeping for potential future use
        # d2S_dt2 = np.gradient(dS_dt, time)

        # Find ignition events
        ignition_indices = np.where(B > DEFAULT_IGNITION_THRESHOLD)[0]

        # Compare derivatives before/after ignition
        discontinuities = []
        window = DEFAULT_DISCONTINUITY_WINDOW
        for idx in ignition_indices:
            if idx > window and idx < len(S) - window:
                pre_deriv = np.mean(dS_dt[idx - window : idx])
                post_deriv = np.mean(dS_dt[idx : idx + window])
                discontinuities.append(abs(post_deriv - pre_deriv))

        results["mean_discontinuity"] = np.mean(discontinuities) if discontinuities else 0

        # 2. Susceptibility (variance near threshold)
        near_threshold = np.abs(S - theta) < DEFAULT_NEAR_THRESHOLD
        far_from_threshold = np.abs(S - theta) > DEFAULT_FAR_THRESHOLD

        if (
            np.sum(near_threshold) > DEFAULT_MIN_SAMPLES
            and np.sum(far_from_threshold) > DEFAULT_MIN_SAMPLES
        ):
            susceptibility_near = np.var(S[near_threshold])
            susceptibility_far = np.var(S[far_from_threshold])
            results["susceptibility_ratio"] = susceptibility_near / (
                susceptibility_far + DEFAULT_EPSILON
            )
        else:
            results["susceptibility_ratio"] = 1.0

        # 3. Critical slowing down
        # Autocorrelation should increase near threshold
        if np.sum(near_threshold) > DEFAULT_MIN_SAMPLES:
            acf_near = self._autocorrelation(S[near_threshold], lag=DEFAULT_AC_LAG)
            acf_far = self._autocorrelation(S[far_from_threshold], lag=DEFAULT_AC_LAG)
            results["critical_slowing"] = acf_near / (acf_far + DEFAULT_EPSILON)
        else:
            results["critical_slowing"] = 1.0

        # 4. Long-range correlations (Hurst exponent)
        if np.sum(near_threshold) > DEFAULT_MIN_SAMPLES:
            H_near = self._hurst_exponent(S[near_threshold])
            H_far = self._hurst_exponent(S[far_from_threshold])
            results["hurst_near"] = H_near
            results["hurst_far"] = H_far
        else:
            results["hurst_near"] = 0.5
            results["hurst_far"] = 0.5

        return results

    def _hurst_exponent(self, series: np.ndarray) -> float:
        """Estimate Hurst exponent via R/S analysis

        Args:
            series: Time series data

        Returns:
            Hurst exponent (0.5 = random walk, > 0.5 = persistent, < 0.5 = anti-persistent)
        """
        if len(series) < DEFAULT_MIN_SAMPLES:
            return 0.5

        n = len(series)
        max_lag = n // DEFAULT_HURST_LAG_MULTIPLIER

        if max_lag < DEFAULT_MIN_LAG:
            return 0.5

        rs_values = []
        lag_values = []

        for lag in range(DEFAULT_MIN_LAG, max_lag):
            # Divide into subseries
            n_subseries = n // lag
            if n_subseries < DEFAULT_MIN_SUBSERIES:
                continue

            rs_subseries = []

            for i in range(n_subseries):
                subseries = series[i * lag : (i + 1) * lag]

                # Mean-adjusted cumulative sum
                mean = np.mean(subseries)
                cumsum = np.cumsum(subseries - mean)

                # Range
                R = np.max(cumsum) - np.min(cumsum)

                # Standard deviation
                std_dev = np.std(subseries)

                if std_dev > 0:
                    rs_subseries.append(R / std_dev)

            if rs_subseries:
                rs_values.append(np.mean(rs_subseries))
                lag_values.append(lag)

        if len(rs_values) > 2:
            # Fit log(R/S) vs log(lag)
            log_lag = np.log(lag_values)
            log_rs = np.log(rs_values)
            try:
                slope, _ = np.polyfit(log_lag, log_rs, 1)
                return slope
            except (ValueError, np.linalg.LinAlgError):
                return 0.5

        return 0.5

    def run_phase_transition_analysis(
        self, n_simulations: int = DEFAULT_N_SIMULATIONS
    ) -> Dict[str, Any]:
        """
        Run comprehensive phase transition analysis

        Args:
            n_simulations: Number of simulation runs

        Returns:
            Dictionary containing aggregated analysis results
        """
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")

        results = {
            "discontinuities": [],
            "susceptibility_ratios": [],
            "critical_slowing": [],
            "hurst_near": [],
            "hurst_far": [],
            "phi_at_ignition": [],
            "phi_baseline": [],
        }

        for i in range(n_simulations):
            if i % DEFAULT_PRINT_INTERVAL == 0:
                print(f"Running simulation {i+1}/{n_simulations}...")

            # Generate simulation with varying inputs
            def input_gen(t):
                return {
                    "Pi_e": 1.0 + 0.3 * np.sin(2 * np.pi * t / 10),
                    "eps_e": np.random.normal(0.5, 0.3),
                    "beta": DEFAULT_BETA,
                    "Pi_i": 1.0,
                    "eps_i": np.random.normal(0.2, 0.2),
                    "M": 1.0,
                    "A": 0.5,
                }

            try:
                history = self.system.simulate(DEFAULT_SIMULATION_DURATION, DEFAULT_DT, input_gen)

                # Phase transition analysis
                pt_results = self.detect_phase_transition(history)
                results["discontinuities"].append(pt_results["mean_discontinuity"])
                results["susceptibility_ratios"].append(pt_results.get("susceptibility_ratio", 1.0))
                results["critical_slowing"].append(pt_results.get("critical_slowing", 1.0))
                results["hurst_near"].append(pt_results.get("hurst_near", 0.5))
                results["hurst_far"].append(pt_results.get("hurst_far", 0.5))

                # Integrated information
                phi = self.compute_integrated_information(history)
                ignition_indices = np.where(history["B"] > DEFAULT_IGNITION_THRESHOLD)[0]

                if len(ignition_indices) > 0:
                    # Φ at ignition
                    ignition_phi = [
                        phi[max(0, idx - DEFAULT_PHI_LOOKBACK)]
                        for idx in ignition_indices
                        if idx >= DEFAULT_PHI_LOOKBACK and idx - DEFAULT_PHI_LOOKBACK < len(phi)
                    ]
                    if ignition_phi:
                        results["phi_at_ignition"].append(np.mean(ignition_phi))

                # Baseline Φ
                non_ignition = np.where(history["B"] < DEFAULT_IGNITION_THRESHOLD)[0]
                baseline_phi = phi[non_ignition[non_ignition < len(phi)]]
                if len(baseline_phi) > 0:
                    results["phi_baseline"].append(np.mean(baseline_phi))

            except Exception as e:
                print(f"Warning: Simulation {i+1} failed: {str(e)}")
                continue

        return {k: np.array(v) for k, v in results.items()}

    def _conditional_prob(self, y_t: int, y_past: int, n_bins: int) -> np.ndarray:
        """Compute conditional probability P(y_t | y_past)

        Note: This is a simplified placeholder implementation.
        A proper implementation would estimate the conditional probability
        from empirical data.

        Args:
            y_t: Current value
            y_past: Past value
            n_bins: Number of bins for discretization

        Returns:
            Probability distribution
        """
        # Simplified - return uniform distribution
        # In a full implementation, this would estimate from data
        return np.ones(n_bins) / n_bins

    def _conditional_prob_joint(
        self, y_t: int, y_past: int, x_past: int, n_bins: int
    ) -> np.ndarray:
        """Compute conditional probability P(y_t | y_past, x_past)

        Note: This is a simplified placeholder implementation.
        A proper implementation would estimate the joint conditional probability
        from empirical data.

        Args:
            y_t: Current value
            y_past: Past Y value
            x_past: Past X value
            n_bins: Number of bins for discretization

        Returns:
            Probability distribution
        """
        # Simplified - return uniform distribution
        # In a full implementation, this would estimate from data
        return np.ones(n_bins) / n_bins

    def _estimate_entropy(self, data: np.ndarray) -> float:
        """Estimate entropy of data

        Args:
            data: Input data (can be univariate or multivariate)

        Returns:
            Estimated entropy value
        """
        # Handle edge cases
        if len(data) == 0:
            return 0.0

        # Check for constant data
        if np.all(data == data[0]):
            return 0.0

        # Simplified entropy estimation
        if len(data.shape) > 1:
            # Multivariate case
            try:
                hist, _ = np.histogramdd(data, bins=DEFAULT_HISTOGRAM_BINS, density=False)
            except (ValueError, np.linalg.LinAlgError):
                # Fallback for problematic multivariate data
                return 0.0
        else:
            # Univariate case - handle edge cases for histogram
            data_std = np.std(data)
            if data_std < DEFAULT_EPSILON:
                return 0.0

            try:
                # Use more robust binning
                data_range = (data.min(), data.max())
                if data_range[0] == data_range[1]:
                    return 0.0
                hist, _ = np.histogram(
                    data, bins=DEFAULT_HISTOGRAM_BINS, range=data_range, density=False
                )
            except (ValueError, np.linalg.LinAlgError):
                # Fallback for problematic data
                return 0.0

        hist = hist.flatten()
        hist = hist[hist > 0]  # Remove zero probabilities

        if len(hist) == 0:
            return 0.0

        # Normalize to get probabilities
        hist = hist / np.sum(hist)
        return -np.sum(hist * np.log(hist + DEFAULT_EPSILON))

    def _autocorrelation(self, series: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag

        Args:
            series: Time series data
            lag: Lag for autocorrelation

        Returns:
            Autocorrelation value at given lag
        """
        if len(series) <= lag:
            return 0.0

        series = series - np.mean(series)
        autocorr = np.correlate(series, series, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        if autocorr[0] == 0:
            return 0.0

        return autocorr[lag] / (autocorr[0] + DEFAULT_EPSILON)


# Main execution
if __name__ == "__main__":
    print("Running phase transition analysis...")
    apgi_system = SurpriseIgnitionSystem()
    analyzer = InformationTheoreticAnalysis(apgi_system)
    results = analyzer.run_phase_transition_analysis()
    print("Phase transition analysis completed:", type(results))
    print("=== Protocol completed successfully ===")


def run_falsification():
    """Entry point for CLI falsification testing."""
    try:
        print("Running APGI Falsification Protocol 4: Phase Transition Analysis")
        apgi_system = SurpriseIgnitionSystem()
        analyzer = InformationTheoreticAnalysis(apgi_system)
        results = analyzer.run_phase_transition_analysis()
        print("Phase transition analysis completed:", type(results))
        print("=== Protocol completed successfully ===")
        return {"status": "success", "results": results}
    except (RuntimeError, ValueError, TypeError, ImportError, KeyError) as e:
        print(f"Error in falsification protocol 4: {e}")
        return {"status": "error", "message": str(e)}
