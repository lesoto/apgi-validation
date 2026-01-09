import numpy as np
import matplotlib.pyplot as plt
import hashlib
import pickle
from pathlib import Path
from functools import wraps

# Cache directory for simulation results
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def cache_simulation(func):
    """Decorator to cache simulation results."""

    @wraps(func)
    def wrapper(self, steps, dt=0.01, inputs=None, use_cache=True):
        if not use_cache:
            return func(self, steps, dt, inputs)

        # Create cache key from parameters and inputs
        cache_key = {"params": self.p, "steps": steps, "dt": dt, "inputs": inputs or {}}
        cache_hash = hashlib.md5(str(cache_key).encode()).hexdigest()
        cache_file = CACHE_DIR / f"simulation_{cache_hash}.pkl"

        # Check if cached result exists
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_result = pickle.load(f)
                print(f"Loaded simulation from cache: {cache_file.name}")
                return cached_result
            except Exception:
                # Cache corrupted, regenerate
                cache_file.unlink(missing_ok=True)

        # Run simulation and cache result
        result = func(self, steps, dt, inputs)

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            print(f"Cached simulation result: {cache_file.name}")
        except Exception as e:
            print(f"Warning: Could not cache simulation result: {e}")

        return result

    return wrapper


class SurpriseIgnitionSystem:
    """
    Implements the Dynamical System for Accumulated Surprise and Ignition
    as specified in the mathematical model.
    """

    def __init__(self, params=None):
        # Default parameters based on Section A.2 Constraints
        default_params = {
            # Timescales (converted to seconds for consistency)
            "tau_S": 0.5,  # 500 ms (Range: 100-1000ms)
            "tau_theta": 30.0,  # 30 s   (Range: 5-60s)
            # Threshold parameters
            "theta_0": 0.5,  # Baseline threshold (Range: 0.1-1.0 AU)
            # Sigmoid parameters
            "alpha": 10.0,  # Sharpness (Range: 1-15)
            # Sensitivities
            "gamma_M": -0.3,  # Metabolic sensitivity (Range: -0.5 to 0.5)
            "gamma_A": 0.1,  # Arousal sensitivity (Range: -0.3 to 0.3)
            # Reset dynamics
            "rho": 0.7,  # Reset fraction (Range: 0.3-0.9)
            # Noise strengths (Assumed values for simulation stability)
            "sigma_S": 0.05,
            "sigma_theta": 0.02,
        }

        if params:
            default_params.update(params)
        self.p = default_params

        # Initial State
        self.S = 0.0
        self.theta = self.p["theta_0"]
        self.B = 0

        # Baselines for modulators
        self.M0 = 1.0
        self.A0 = 0.5

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def step(self, dt, inputs):
        """
        Performs a single Euler-Maruyama integration step.

        Args:
            dt (float): Time step in seconds.
            inputs (dict): Dictionary containing current values for:
                - Pi_e: External precision/gain
                - eps_e: External prediction error
                - beta: Somatic focus
                - Pi_i: Internal precision
                - eps_i: Internal prediction error
                - M: Current metabolic state
                - A: Current arousal state

        Returns:
            dict: The new state of the system (S, theta, B, prob_ignition)
        """
        # 1. Unpack Inputs
        Pi_e = inputs.get("Pi_e", 1.0)
        eps_e = inputs.get("eps_e", 0.0)
        beta = inputs.get("beta", 1.0)
        Pi_i = inputs.get("Pi_i", 1.0)
        eps_i = inputs.get("eps_i", 0.0)
        M_t = inputs.get("M", self.M0)
        A_t = inputs.get("A", self.A0)

        # 2. Stochastic Noise Terms (Wiener processes)
        # Scaled by sqrt(dt) for correct SDE integration
        xi_S = np.random.normal(0, 1) * np.sqrt(dt)
        xi_theta = np.random.normal(0, 1) * np.sqrt(dt)

        # 3. Update Accumulated Surprise (S_t)
        # dS = (-S/tau_S + Inputs)dt + sigma * dW
        input_drive = (Pi_e * np.abs(eps_e)) + (beta * Pi_i * np.abs(eps_i))
        dS = (-self.S / self.p["tau_S"] + input_drive) * dt + self.p["sigma_S"] * xi_S

        # Tentative new S (before ignition check)
        S_new = max(0, self.S + dS)  # Enforce S >= 0 constraint

        # 4. Update Threshold (theta_t)
        # dTheta = ((theta_0 - theta)/tau_theta + Modulators)dt + sigma * dW
        modulation = self.p["gamma_M"] * (M_t - self.M0) + self.p["gamma_A"] * (
            A_t - self.A0
        )
        dTheta = (
            (self.p["theta_0"] - self.theta) / self.p["tau_theta"] + modulation
        ) * dt + self.p["sigma_theta"] * xi_theta

        theta_new = max(0.01, self.theta + dTheta)  # Enforce theta > 0 constraint

        # 5. Determine Ignition (B_t)
        # P(B_t = 1) = Sigmoid(alpha * (S - theta))
        ignition_prob = self.sigmoid(self.p["alpha"] * (S_new - theta_new))

        # Bernoulli trial for ignition
        is_ignited = 1 if np.random.random() < ignition_prob else 0

        # 6. Post-ignition reset
        if is_ignited:
            S_final = S_new * (1.0 - self.p["rho"])
        else:
            S_final = S_new

        # Update internal state
        self.S = S_final
        self.theta = theta_new
        self.B = is_ignited

        return {"S": self.S, "theta": self.theta, "B": self.B, "prob": ignition_prob}

    @cache_simulation
    def simulate(self, steps, dt=0.01, inputs=None):
        """
        Run simulation for specified number of steps with caching support.

        Args:
            steps (int): Number of simulation steps
            dt (float): Time step size in seconds
            inputs (dict or list): Input parameters. Can be:
                - dict: Single set of inputs for all steps
                - list: List of input dictionaries, one per step

        Returns:
            dict: Simulation results containing time series data
        """
        # Reset state for new simulation
        self.reset_state()

        # Prepare inputs
        if inputs is None:
            inputs = {}
        if isinstance(inputs, dict):
            # Repeat same inputs for all steps
            input_sequence = [inputs] * steps
        else:
            # Assume list of inputs
            input_sequence = inputs
            if len(input_sequence) != steps:
                raise ValueError(
                    f"Length of inputs ({len(input_sequence)}) must match steps ({steps})"
                )

        # Storage for results
        results = {
            "time": np.arange(steps) * dt,
            "S": np.zeros(steps),
            "theta": np.zeros(steps),
            "B": np.zeros(steps),
            "prob": np.zeros(steps),
        }

        # Run simulation
        for i in range(steps):
            state = self.step(dt, input_sequence[i])
            results["S"][i] = state["S"]
            results["theta"][i] = state["theta"]
            results["B"][i] = state["B"]
            results["prob"][i] = state["prob"]

        return results

    def reset_state(self):
        """Reset system to initial conditions."""
        self.S = 0.0
        self.theta = self.p["theta_0"]
        self.B = 0

    def clear_cache(self):
        """Clear all cached simulation results."""
        import glob

        cache_files = glob.glob(str(CACHE_DIR / "simulation_*.pkl"))
        for cache_file in cache_files:
            Path(cache_file).unlink(missing_ok=True)
        print(f"Cleared {len(cache_files)} cached simulation files")


def run_simulation():
    # --- Simulation Configuration ---
    duration = 120.0  # seconds
    dt = 0.05  # Time step
    steps = int(duration / dt)
    time = np.linspace(0, duration, steps)

    # Initialize System
    sys = SurpriseIgnitionSystem()

    # --- Generate Synthetic Inputs ---
    # 1. Background noise for errors
    eps_e = np.random.normal(0, 0.1, steps)
    eps_i = np.random.normal(0, 0.1, steps)

    # 2. Event: A sudden burst of external surprise at t=30s
    event_start = int(30.0 / dt)
    event_end = int(35.0 / dt)
    eps_e[event_start:event_end] += np.random.normal(2.0, 0.5, event_end - event_start)

    # 3. Event: A burst of internal somatic error at t=80s
    somatic_start = int(80.0 / dt)
    somatic_end = int(85.0 / dt)
    eps_i[somatic_start:somatic_end] += np.random.normal(
        1.5, 0.5, somatic_end - somatic_start
    )

    # 4. Metabolic Oscillation (slow sine wave)
    # This affects the threshold
    M_signal = 1.0 + 0.5 * np.sin(2 * np.pi * time / 50.0)

    # --- Run Loop ---
    history = {"S": [], "theta": [], "B": [], "prob": [], "input_drive": []}

    print(f"Starting simulation for {duration} seconds...")

    for t_idx in range(steps):
        # Construct input vector for this step
        current_inputs = {
            "Pi_e": 1.0,
            "eps_e": eps_e[t_idx],
            "beta": 1.2,  # Slightly heightened somatic focus
            "Pi_i": 0.8,
            "eps_i": eps_i[t_idx],
            "M": M_signal[t_idx],
            "A": 0.5,  # Constant arousal for this test
        }

        # Log input drive for visualization
        drive = (current_inputs["Pi_e"] * abs(current_inputs["eps_e"])) + (
            current_inputs["beta"]
            * current_inputs["Pi_i"]
            * abs(current_inputs["eps_i"])
        )
        history["input_drive"].append(drive)

        # Step system
        state = sys.step(dt, current_inputs)

        # Store results
        history["S"].append(state["S"])
        history["theta"].append(state["theta"])
        history["B"].append(state["B"])
        history["prob"].append(state["prob"])

    # --- Visualization ---
    print("Simulation complete. Generating plots...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Dynamics of S and Theta
    ax0 = axes[0]
    ax0.plot(time, history["S"], label="$S_t$ (Surprise)", color="blue", linewidth=1.5)
    ax0.plot(
        time,
        history["theta"],
        label=r"$\theta_t$ (Threshold)",
        color="orange",
        linestyle="--",
        linewidth=1.5,
    )

    # Mark Ignitions
    ignitions = np.array(history["B"])
    ignition_times = time[ignitions == 1]
    ignition_vals = np.array(history["S"])[ignitions == 1]

    if len(ignition_times) > 0:
        ax0.scatter(
            ignition_times,
            ignition_vals,
            color="red",
            s=50,
            zorder=5,
            label="Ignition ($B_t=1$)",
        )
        # Draw vertical lines for context
        for it in ignition_times:
            ax0.axvline(x=it, color="red", alpha=0.1)

    ax0.set_ylabel("Magnitude")
    ax0.set_title("A.1 Full Dynamical System: State Variables")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    # Plot 2: Input Drive (External vs Internal Errors)
    ax1 = axes[1]
    ax1.plot(
        time,
        history["input_drive"],
        color="green",
        alpha=0.6,
        label="Total Input Drive",
    )
    ax1.fill_between(time, 0, history["input_drive"], color="green", alpha=0.1)
    ax1.set_ylabel("Input Drive")
    ax1.set_title("Driving Forces (Prediction Errors)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot 3: Ignition Probability and Modulators
    ax2 = axes[2]
    ax2_twin = ax2.twinx()

    ln1 = ax2.plot(time, history["prob"], color="purple", label="$P(Ignition)$")
    ln2 = ax2_twin.plot(
        time, M_signal, color="brown", linestyle=":", label="Metabolism ($M_t$)"
    )

    ax2.set_ylabel("Probability")
    ax2.set_ylim(-0.1, 1.1)
    ax2_twin.set_ylabel("Metabolic State")

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc="upper right")

    ax2.set_xlabel("Time (seconds)")
    ax2.set_title("Ignition Probability & Physiological Modulation")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
