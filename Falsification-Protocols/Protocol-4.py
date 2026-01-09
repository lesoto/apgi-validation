import numpy as np
from scipy.stats import entropy
from typing import Dict


class SurpriseIgnitionSystem:
    """Surprise accumulation and ignition system for APGI"""

    def __init__(self):
        # System parameters
        self.S_t = 0.0  # Current surprise level
        self.theta_t = 0.5  # Current threshold
        self.theta_0 = 0.5  # Baseline threshold
        self.alpha = 8.0  # Sigmoid steepness
        self.tau_S = 0.3  # Surprise time constant
        self.tau_theta = 10.0  # Threshold time constant
        self.eta_theta = 0.01  # Threshold adaptation rate
        self.beta = 1.2  # Somatic bias

        # State tracking
        self.time = 0.0
        self.ignition_states = []

    def simulate(self, duration: float, dt: float, input_generator) -> Dict:
        """Simulate the system for given duration"""
        n_steps = int(duration / dt)

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

            # Get inputs
            inputs = input_generator(t)
            Pi_e = inputs["Pi_e"]
            Pi_i = inputs["Pi_i"]
            eps_e = inputs["eps_e"]
            eps_i = inputs["eps_i"]
            beta = inputs["beta"]

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
            ignition_prob = 1.0 / (
                1.0 + np.exp(-self.alpha * (self.S_t - self.theta_t))
            )
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
                self.S_t *= 0.3

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
        self, history: Dict, source: str, target: str, lag: int = 1
    ) -> np.ndarray:
        """
        Transfer entropy: Information flow from source to target

        TE(X→Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})

        Prediction: TE increases dramatically at ignition
        """
        from scipy.stats import entropy

        X = history[source]
        Y = history[target]
        B = history["B"]  # Ignition states

        # Discretize for MI estimation
        n_bins = 20
        X_binned = np.digitize(X, np.linspace(X.min(), X.max(), n_bins))
        Y_binned = np.digitize(Y, np.linspace(Y.min(), Y.max(), n_bins))

        te_values = np.zeros(len(X) - lag)

        for t in range(lag, len(X)):
            # H(Y_t | Y_{t-lag})
            p_Y_given_Ypast = self._conditional_prob(
                Y_binned[t], Y_binned[t - lag], n_bins
            )
            H_Y_given_Ypast = entropy(p_Y_given_Ypast)

            # H(Y_t | Y_{t-lag}, X_{t-lag})
            p_Y_given_both = self._conditional_prob_joint(
                Y_binned[t], Y_binned[t - lag], X_binned[t - lag], n_bins
            )
            H_Y_given_both = entropy(p_Y_given_both)

            te_values[t - lag] = H_Y_given_Ypast - H_Y_given_both

        return te_values

    def compute_integrated_information(
        self, history: Dict, window_size: int = 50
    ) -> np.ndarray:
        """
        Φ-like measure: How much more information is in the whole
        than in the parts

        Prediction: Φ spikes at ignition
        """
        S = history["S"]
        theta = history["theta"]
        B = history["B"]

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

    def detect_phase_transition(self, history: Dict) -> Dict:
        """
        Detect signatures of phase transition at ignition

        Signatures:
        1. Discontinuity in first derivative of S
        2. Diverging susceptibility (variance)
        3. Critical slowing down before transition
        4. Long-range correlations at criticality
        """
        S = history["S"]
        theta = history["theta"]
        B = history["B"]
        time = history["time"]

        results = {}

        # 1. Discontinuity analysis
        dS_dt = np.gradient(S, time)
        d2S_dt2 = np.gradient(dS_dt, time)

        # Find ignition events
        ignition_indices = np.where(B > 0.5)[0]

        # Compare derivatives before/after ignition
        discontinuities = []
        for idx in ignition_indices:
            if idx > 10 and idx < len(S) - 10:
                pre_deriv = np.mean(dS_dt[idx - 10 : idx])
                post_deriv = np.mean(dS_dt[idx : idx + 10])
                discontinuities.append(abs(post_deriv - pre_deriv))

        results["mean_discontinuity"] = (
            np.mean(discontinuities) if discontinuities else 0
        )

        # 2. Susceptibility (variance near threshold)
        near_threshold = np.abs(S - theta) < 0.2
        far_from_threshold = np.abs(S - theta) > 0.5

        if np.sum(near_threshold) > 10 and np.sum(far_from_threshold) > 10:
            susceptibility_near = np.var(S[near_threshold])
            susceptibility_far = np.var(S[far_from_threshold])
            results["susceptibility_ratio"] = susceptibility_near / (
                susceptibility_far + 1e-10
            )

        # 3. Critical slowing down
        # Autocorrelation should increase near threshold
        acf_near = self._autocorrelation(S[near_threshold], lag=5)
        acf_far = self._autocorrelation(S[far_from_threshold], lag=5)
        results["critical_slowing"] = acf_near / (acf_far + 1e-10)

        # 4. Long-range correlations (Hurst exponent)
        H_near = self._hurst_exponent(S[near_threshold])
        H_far = self._hurst_exponent(S[far_from_threshold])
        results["hurst_near"] = H_near
        results["hurst_far"] = H_far

        return results

    def _hurst_exponent(self, series: np.ndarray) -> float:
        """Estimate Hurst exponent via R/S analysis"""
        if len(series) < 20:
            return 0.5

        n = len(series)
        max_lag = n // 4

        rs_values = []
        lag_values = []

        for lag in range(10, max_lag):
            # Divide into subseries
            n_subseries = n // lag
            rs_subseries = []

            for i in range(n_subseries):
                subseries = series[i * lag : (i + 1) * lag]

                # Mean-adjusted cumulative sum
                mean = np.mean(subseries)
                cumsum = np.cumsum(subseries - mean)

                # Range
                R = np.max(cumsum) - np.min(cumsum)

                # Standard deviation
                S = np.std(subseries)

                if S > 0:
                    rs_subseries.append(R / S)

            if rs_subseries:
                rs_values.append(np.mean(rs_subseries))
                lag_values.append(lag)

        if len(rs_values) > 2:
            # Fit log(R/S) vs log(lag)
            log_lag = np.log(lag_values)
            log_rs = np.log(rs_values)
            slope, _ = np.polyfit(log_lag, log_rs, 1)
            return slope

        return 0.5

    def run_phase_transition_analysis(self, n_simulations: int = 100) -> Dict:
        """
        Run comprehensive phase transition analysis
        """
        results = {
            "discontinuities": [],
            "susceptibility_ratios": [],
            "critical_slowing": [],
            "hurst_near": [],
            "hurst_far": [],
            "phi_at_ignition": [],
            "phi_baseline": [],
        }

        for _ in range(n_simulations):
            # Generate simulation with varying inputs
            def input_gen(t):
                return {
                    "Pi_e": 1.0 + 0.3 * np.sin(2 * np.pi * t / 10),
                    "eps_e": np.random.normal(0.5, 0.3),
                    "beta": 1.2,
                    "Pi_i": 1.0,
                    "eps_i": np.random.normal(0.2, 0.2),
                    "M": 1.0,
                    "A": 0.5,
                }

            history = self.system.simulate(100.0, 0.05, input_gen)

            # Phase transition analysis
            pt_results = self.detect_phase_transition(history)
            results["discontinuities"].append(pt_results["mean_discontinuity"])
            results["susceptibility_ratios"].append(
                pt_results.get("susceptibility_ratio", 1.0)
            )
            results["critical_slowing"].append(pt_results.get("critical_slowing", 1.0))
            results["hurst_near"].append(pt_results.get("hurst_near", 0.5))
            results["hurst_far"].append(pt_results.get("hurst_far", 0.5))

            # Integrated information
            phi = self.compute_integrated_information(history)
            ignition_indices = np.where(history["B"] > 0.5)[0]

            if len(ignition_indices) > 0:
                # Φ at ignition
                ignition_phi = [
                    phi[max(0, idx - 50)]
                    for idx in ignition_indices
                    if idx >= 50 and idx - 50 < len(phi)
                ]
                if ignition_phi:
                    results["phi_at_ignition"].append(np.mean(ignition_phi))

            # Baseline Φ
            non_ignition = np.where(history["B"] < 0.5)[0]
            baseline_phi = phi[non_ignition[non_ignition < len(phi)]]
            if len(baseline_phi) > 0:
                results["phi_baseline"].append(np.mean(baseline_phi))

        return {k: np.array(v) for k, v in results.items()}

    def _conditional_prob(self, y_t: int, y_past: int, n_bins: int) -> np.ndarray:
        """Compute conditional probability P(y_t | y_past)"""
        # Simplified - return uniform distribution
        return np.ones(n_bins) / n_bins

    def _conditional_prob_joint(
        self, y_t: int, y_past: int, x_past: int, n_bins: int
    ) -> np.ndarray:
        """Compute conditional probability P(y_t | y_past, x_past)"""
        # Simplified - return uniform distribution
        return np.ones(n_bins) / n_bins

    def _estimate_entropy(self, data: np.ndarray) -> float:
        """Estimate entropy of data"""
        # Simplified entropy estimation
        if len(data.shape) > 1:
            # Multivariate case
            hist, _ = np.histogramdd(data, bins=10)
        else:
            # Univariate case
            hist, _ = np.histogram(data, bins=10)

        hist = hist.flatten()
        hist = hist[hist > 0]  # Remove zero probabilities
        return -np.sum(hist * np.log(hist + 1e-10))

    def _autocorrelation(self, series: np.ndarray, lag: int) -> float:
        """Compute autocorrelation at given lag"""
        if len(series) <= lag:
            return 0.0

        series = series - np.mean(series)
        autocorr = np.correlate(series, series, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        if autocorr[0] == 0:
            return 0.0

        return autocorr[lag] / autocorr[0]


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
    except Exception as e:
        print(f"Error in falsification protocol 4: {e}")
        return {"status": "error", "message": str(e)}


# Main execution
if __name__ == "__main__":
    print("Running phase transition analysis...")
    apgi_system = SurpriseIgnitionSystem()
    analyzer = InformationTheoreticAnalysis(apgi_system)
    results = analyzer.run_phase_transition_analysis()
    print("Phase transition analysis completed:", type(results))
    print("=== Protocol completed successfully ===")
