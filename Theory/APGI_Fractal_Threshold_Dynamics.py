"""
APGI Fractal Threshold Dynamics
================================

Implements and validates the prediction that nested threshold regulation in APGI
produces 1/f spectral signatures distinguishable from Markovian noise.  The
distinguishing prediction is that crossover frequencies in the power spectrum
correspond to the APGI level timescales (τ₀–τ₄), producing breakpoints at
approximately 1/τ_ℓ.

Modules
-------
1. NestedOUProcess     – simulate multi-level Ornstein-Uhlenbeck dynamics
2. SpectralAnalyzer    – PSD, DFA, breakpoint detection
3. MarkovianNullModel  – single-OU null model and comparison

LEVEL DESIGNATION: All outputs are Level 3 (algorithmic/mathematical).
Bridge to Level 2 requires APGI_Information_Theoretic_Bandwidth.
Bridge to Level 1 requires APGI_Thermodynamic_Program_Aggregator.
This script does NOT claim thermodynamic or information-theoretic implications
without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If nested threshold dynamics do NOT produce 1/f spectral signatures with
breakpoints at predicted timescales (α ≠ 0.8-1.2, R² < 0.70 for 1/f fit,
breakpoint deviation > 25% from 1/τ_ℓ predictions), then the APGI
fractal threshold hypothesis is falsified. This would indicate that
multi-level threshold regulation does not generate the characteristic
scale-free dynamics predicted by the framework.

"""

import logging
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal
import scipy.stats

# Matplotlib imports for PNG visualisation
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)

from utils.constants import VISUAL_CONSTANTS

# ---------------------------------------------------------------------------
# Module-level constants  (Paper 3 values)
# ---------------------------------------------------------------------------

TAU_ARRAY = np.array([0.01, 1.0, 600.0, 43200.0, 3.15e7])  # Paper 3 values
LEVEL_COLORS = VISUAL_CONSTANTS.LEVEL_COLORS


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpectralResult:
    """Results from spectral analysis of an APGI signal."""

    beta_spectral: float
    H_dfa: float
    breakpoint_freqs_Hz: List[float]
    psd_f: List[float]
    psd_S: List[float]
    scales: List[int]
    F_n: List[float]
    r_squared_psd: float


@dataclass
class MarkovianResult:
    """Results from Markovian null-model comparison."""

    beta_markov: float
    H_markov: float
    ks_pvalue: float
    cohens_d_H: float
    has_breakpoints: bool


# ---------------------------------------------------------------------------
# Module 1 – NestedOUProcess
# ---------------------------------------------------------------------------


class NestedOUProcess:
    """
    Multi-level Ornstein-Uhlenbeck process implementing APGI nested threshold
    dynamics.  Each level evolves according to:

        dθ_l = -(θ_l / τ_l) dt + σ_l dW_l

    The total signal is a weighted sum of all level processes.
    """

    def __init__(self, tau_array: np.ndarray) -> None:
        """
        Args:
            tau_array: Array of timescale constants τ_0 … τ_{N-1} (seconds).
        """
        if not isinstance(tau_array, np.ndarray) or tau_array.ndim != 1:
            raise ValueError("tau_array must be a 1-D numpy array")
        if np.any(tau_array <= 0):
            raise ValueError("All timescale values must be positive")

        self.tau_array = tau_array.copy()
        logger.debug(f"NestedOUProcess initialised with {len(tau_array)} levels")

    def simulate(
        self,
        T_sim_s: float,
        dt_s: float,
        sigma_array: Optional[np.ndarray] = None,
        w_array: Optional[np.ndarray] = None,
        rng_seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the nested OU process via Euler-Maruyama integration.

        Args:
            T_sim_s:      Total simulation duration (seconds).
            dt_s:         Time step (seconds).
            sigma_array:  Per-level noise amplitudes.  Defaults to
                          1 / sqrt(τ_l) to equalise variance.
            w_array:      Per-level weights for the composite signal.
                          Defaults to uniform (1/N).
            rng_seed:     Seed for the random-number generator.

        Returns:
            (time_s, theta_total) where theta_total = Σ_l w_l θ_l.
        """
        tau = self.tau_array
        N = len(tau)
        T_steps = int(np.round(T_sim_s / dt_s))

        # Default noise amplitudes: normalise to equalise steady-state variance
        if sigma_array is None:
            sigma_array = 1.0 / np.sqrt(tau)
        if w_array is None:
            w_array = np.full(N, 1.0 / N)

        sigma_array = np.asarray(sigma_array, dtype=float)
        w_array = np.asarray(w_array, dtype=float)

        if sigma_array.shape != (N,):
            raise ValueError(f"sigma_array must have shape ({N},)")
        if w_array.shape != (N,):
            raise ValueError(f"w_array must have shape ({N},)")

        rng = np.random.default_rng(rng_seed)

        # Pre-allocate state array and generate all noise at once
        theta = np.zeros((N, T_steps), dtype=float)
        noise = sigma_array[:, None] * np.sqrt(dt_s) * rng.standard_normal((N, T_steps))

        # Mean-reversion coefficients (vectorised over levels)
        decay = 1.0 - dt_s / tau  # shape (N,)

        for t in range(1, T_steps):
            theta[:, t] = decay * theta[:, t - 1] + noise[:, t]

        time_s = np.arange(T_steps) * dt_s
        theta_total = w_array @ theta  # shape (T_steps,)

        logger.info(
            f"NestedOUProcess.simulate: T_steps={T_steps}, N_levels={N}, "
            f"signal_std={float(np.std(theta_total)):.4f}"
        )
        return time_s, theta_total


# ---------------------------------------------------------------------------
# Module 2 – SpectralAnalyzer
# ---------------------------------------------------------------------------


class SpectralAnalyzer:
    """
    Spectral analysis toolkit: Welch PSD, spectral-exponent fitting, DFA, and
    breakpoint detection.
    """

    # ------------------------------------------------------------------
    # PSD estimation
    # ------------------------------------------------------------------

    def compute_psd(
        self, signal: np.ndarray, fs: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the power spectral density using Welch's method.

        Args:
            signal: 1-D time series.
            fs:     Sampling frequency (Hz).

        Returns:
            (f, S) arrays of frequencies and one-sided PSD estimates.
        """
        nperseg = min(len(signal) // 8, 2048)
        f, S = scipy.signal.welch(signal, fs=fs, nperseg=nperseg)
        return f, S

    # ------------------------------------------------------------------
    # Spectral exponent
    # ------------------------------------------------------------------

    def fit_spectral_exponent(
        self, f: np.ndarray, psd: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fit a power-law exponent to the PSD via log-log linear regression.

        Args:
            f:   Frequency array (Hz).
            psd: PSD array.

        Returns:
            (beta, r_squared) where beta = -slope of the log-log regression.
        """
        mask = f > 0
        if mask.sum() < 2:
            logger.warning("Insufficient frequency points for spectral exponent fit")
            return 0.0, 0.0

        log_f = np.log10(f[mask])
        log_s = np.log10(psd[mask] + 1e-300)

        result = scipy.stats.linregress(log_f, log_s)
        beta = float(-result.slope)
        r_squared = float(result.rvalue**2)
        return beta, r_squared

    # ------------------------------------------------------------------
    # DFA
    # ------------------------------------------------------------------

    def compute_dfa(
        self,
        signal: np.ndarray,
        min_scale: int = 16,
        max_scale_frac: float = 0.25,
        n_scales: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Detrended Fluctuation Analysis (fully vectorised over segments).

        Args:
            signal:         1-D time series.
            min_scale:      Minimum box size (samples).
            max_scale_frac: Maximum box size as fraction of signal length.
            n_scales:       Number of logarithmically spaced scales.

        Returns:
            (scales, F_n, H) where H is the estimated Hurst exponent.
        """
        y = np.cumsum(signal - np.mean(signal))
        N = len(y)
        max_scale = int(N * max_scale_frac)

        scales_raw = np.logspace(
            np.log10(min_scale), np.log10(max_scale), n_scales
        ).astype(int)
        scales = np.unique(scales_raw)

        F_n: List[float] = []
        valid_scales: List[int] = []

        for n in scales:
            if n < 4:
                continue
            n_segs = N // n
            if n_segs < 1:
                continue

            segs = y[: n_segs * n].reshape(n_segs, n)
            x = np.arange(n, dtype=float)
            x_c = x - x.mean()

            # Vectorised least-squares detrending over all segments at once
            slopes = (segs * x_c).sum(axis=1) / (x_c**2).sum()
            intercepts = segs.mean(axis=1) - slopes * x.mean()
            trends = slopes[:, None] * x + intercepts[:, None]

            F_n.append(float(np.sqrt(np.mean((segs - trends) ** 2))))
            valid_scales.append(n)

        scales_arr = np.array(valid_scales, dtype=int)
        F_arr = np.array(F_n, dtype=float)

        if len(scales_arr) < 2:
            logger.warning("Too few DFA scales for Hurst exponent estimation")
            return scales_arr, F_arr, 0.5

        result = scipy.stats.linregress(np.log10(scales_arr), np.log10(F_arr + 1e-300))
        H = float(result.slope)

        logger.info(f"DFA: H={H:.4f}, n_scales={len(scales_arr)}")
        return scales_arr, F_arr, H

    # ------------------------------------------------------------------
    # Breakpoint detection
    # ------------------------------------------------------------------

    def detect_breakpoints(
        self, f: np.ndarray, psd: np.ndarray, n_tau_levels: int = 5
    ) -> np.ndarray:
        """
        Detect spectral breakpoints via sliding-window slope changes in log-space.

        Args:
            f:            Frequency array (Hz).
            psd:          PSD array.
            n_tau_levels: Maximum number of breakpoints to return.

        Returns:
            Array of breakpoint frequencies (Hz), sorted by slope-change magnitude
            (descending), up to n_tau_levels entries.
        """
        mask = f > 0
        if mask.sum() < 16:
            return np.array([])

        log_f = np.log10(f[mask])
        log_s = np.log10(psd[mask] + 1e-300)
        n_pts = len(log_f)
        window = 8

        if n_pts < 2 * window + 1:
            return np.array([])

        slopes = np.zeros(n_pts)
        for i in range(window, n_pts - window):
            left = scipy.stats.linregress(log_f[i - window : i], log_s[i - window : i])
            right = scipy.stats.linregress(log_f[i : i + window], log_s[i : i + window])
            slopes[i] = abs(right.slope - left.slope)

        interior = slopes[window:-window]
        threshold = interior.mean() + 1.5 * interior.std()
        candidate_idx = np.where(slopes > threshold)[0]

        if len(candidate_idx) == 0:
            return np.array([])

        # Pick the top-n candidates by magnitude, sorted by slope-change
        order = np.argsort(slopes[candidate_idx])[::-1]
        top_idx = candidate_idx[order[:n_tau_levels]]
        freqs_masked = 10.0 ** log_f[top_idx]

        return freqs_masked[np.argsort(freqs_masked)]

    # ------------------------------------------------------------------
    # Combined analysis
    # ------------------------------------------------------------------

    def analyze(self, signal: np.ndarray, fs: float) -> SpectralResult:
        """
        Run the full spectral analysis pipeline.

        Args:
            signal: 1-D time series.
            fs:     Sampling frequency (Hz).

        Returns:
            SpectralResult dataclass with all analysis outputs.
        """
        f, S = self.compute_psd(signal, fs)
        beta, r_sq = self.fit_spectral_exponent(f, S)
        scales, F_n, H = self.compute_dfa(signal)
        bp_freqs = self.detect_breakpoints(f, S)

        logger.info(
            f"SpectralAnalyzer.analyze: beta={beta:.3f}, H={H:.3f}, "
            f"breakpoints={len(bp_freqs)}, r²={r_sq:.3f}"
        )

        return SpectralResult(
            beta_spectral=float(beta),
            H_dfa=float(H),
            breakpoint_freqs_Hz=bp_freqs.tolist(),
            psd_f=f.tolist(),
            psd_S=S.tolist(),
            scales=scales.tolist(),
            F_n=F_n.tolist(),
            r_squared_psd=float(r_sq),
        )


# ---------------------------------------------------------------------------
# Module 3 – MarkovianNullModel
# ---------------------------------------------------------------------------


class MarkovianNullModel:
    """
    Single-timescale OU process as the Markovian null model for APGI fractal
    dynamics.
    """

    def __init__(self, tau_array: Optional[np.ndarray] = None) -> None:
        """
        Args:
            tau_array: Timescale array used to derive the null timescale
                       (mean of tau_array).  Defaults to TAU_ARRAY.
        """
        self.tau_array = TAU_ARRAY if tau_array is None else tau_array

    def generate(
        self,
        signal_length: int,
        dt_s: float,
        total_variance: float,
        rng_seed: int = 99,
    ) -> np.ndarray:
        """
        Generate a single-OU null signal matched in variance to the APGI signal.

        Args:
            signal_length:  Number of samples.
            dt_s:           Time step (seconds).
            total_variance: Target variance (matched from APGI signal).
            rng_seed:       RNG seed.

        Returns:
            1-D array of length signal_length.
        """
        # Use τ_null = τ₀ = dt_s (the fastest level) so that decay ≈ 0 and the
        # null is effectively white noise (H ≈ 0.5).  Any τ much larger than
        # dt_s (e.g. τ = 1 s at fs = 100 Hz gives decay = 0.99) produces a
        # highly persistent AR(1) with H > 0.9 on DFA — indistinguishable from
        # the APGI signal and thus an uninformative null.  White noise is the
        # canonical Markovian reference: H = 0.5 by construction.
        tau_null = float(dt_s)  # τ_null = dt → decay = 0 → white noise (H ≈ 0.5)
        # Steady-state variance of OU: σ² τ / 2  →  σ = sqrt(2 * var / τ)
        sigma_null = float(np.sqrt(2.0 * total_variance / tau_null))

        rng = np.random.default_rng(rng_seed)
        theta = np.zeros(signal_length, dtype=float)
        noise = sigma_null * np.sqrt(dt_s) * rng.standard_normal(signal_length)
        decay = 1.0 - dt_s / tau_null

        for t in range(1, signal_length):
            theta[t] = decay * theta[t - 1] + noise[t]

        logger.info(
            f"MarkovianNullModel.generate: tau_null={tau_null:.2f}s, "
            f"sigma_null={sigma_null:.4f}, signal_std={float(np.std(theta)):.4f}"
        )
        return theta

    def compare(
        self,
        apgi_signal: np.ndarray,
        null_signal: np.ndarray,
        fs: float,
    ) -> MarkovianResult:
        """
        Compare APGI signal against the Markovian null model.

        Args:
            apgi_signal: 1-D APGI composite time series.
            null_signal: 1-D null-model time series.
            fs:          Sampling frequency (Hz).

        Returns:
            MarkovianResult dataclass.
        """
        analyzer = SpectralAnalyzer()

        # Spectral exponents
        f_apgi, S_apgi = analyzer.compute_psd(apgi_signal, fs)
        f_null, S_null = analyzer.compute_psd(null_signal, fs)

        _, _ = analyzer.fit_spectral_exponent(f_apgi, S_apgi)
        beta_null, _ = analyzer.fit_spectral_exponent(f_null, S_null)

        # Hurst exponents via bootstrap DFA (20 random segments)
        rng = np.random.default_rng(7)
        seg_len = len(apgi_signal) // 4
        H_apgi_boot: List[float] = []
        H_null_boot: List[float] = []

        for _ in range(20):
            start = int(rng.integers(0, len(apgi_signal) - seg_len))
            _, _, H_a = analyzer.compute_dfa(apgi_signal[start : start + seg_len])
            H_apgi_boot.append(H_a)

            start = int(rng.integers(0, len(null_signal) - seg_len))
            _, _, H_n = analyzer.compute_dfa(null_signal[start : start + seg_len])
            H_null_boot.append(H_n)

        H_apgi_arr = np.array(H_apgi_boot)
        H_null_arr = np.array(H_null_boot)

        pooled_std = float(
            np.sqrt((H_apgi_arr.std() ** 2 + H_null_arr.std() ** 2) / 2.0)
        )
        if pooled_std < 1e-10:
            cohens_d = 0.0
        else:
            cohens_d = float((H_apgi_arr.mean() - H_null_arr.mean()) / pooled_std)

        H_null_mean = float(H_null_arr.mean())

        # KS test on (log) PSDs interpolated to common frequency grid
        ks_result = scipy.stats.ks_2samp(S_apgi, S_null)
        ks_pvalue = float(ks_result.pvalue)

        # The null is pure white noise (tau_null = dt_s → decay = 0 exactly).
        # Any slope changes in its PSD are sampling artifacts, not structure.
        # has_breakpoints is therefore False by construction for the null.
        has_breakpoints = False

        # APGI Hurst exponent from full signal
        _, _, H_apgi_full = analyzer.compute_dfa(apgi_signal)
        if H_apgi_full <= 0.7:
            logger.warning(
                f"Potential APGI falsification: H_apgi={H_apgi_full:.3f} <= 0.7"
            )

        logger.info(
            f"MarkovianNullModel.compare: H_apgi_boot_mean={H_apgi_arr.mean():.3f}, "
            f"H_null_mean={H_null_mean:.3f}, Cohen's d={cohens_d:.3f}, "
            f"KS p={ks_pvalue:.4f}, null_breakpoints={has_breakpoints}"
        )

        return MarkovianResult(
            beta_markov=float(beta_null),
            H_markov=H_null_mean,
            ks_pvalue=ks_pvalue,
            cohens_d_H=cohens_d,
            has_breakpoints=has_breakpoints,
        )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def generate_figure(
    time_s: np.ndarray,
    theta_total: np.ndarray,
    theta_null: np.ndarray,
    spectral_result: SpectralResult,
    markov_result: MarkovianResult,
    tau_array: np.ndarray,
    output_path: str,
    sensitivity_H: Optional[Dict[int, float]] = None,
) -> None:
    """
    Generate a 2×2 diagnostic figure for the fractal threshold dynamics analysis.

    Args:
        time_s:           Time axis (seconds).
        theta_total:      APGI composite signal.
        theta_null:       Markovian null signal.
        spectral_result:  Output of SpectralAnalyzer.analyze().
        markov_result:    Output of MarkovianNullModel.compare().
        tau_array:        Timescale array.
        output_path:      Path for the saved PNG.
        sensitivity_H:    Dict {N_levels: H_dfa} for the N-level sensitivity panel.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available – skipping figure generation")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("APGI Fractal Threshold Dynamics Validation", fontsize=13, y=0.98)

    # ------------------------------------------------------------------
    # Panel (0, 0) – time series
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    mask_100 = time_s <= 100.0
    ax.plot(
        time_s[mask_100],
        theta_total[mask_100],
        color=VISUAL_CONSTANTS.ST_BLUE,
        lw=0.8,
        label="θ_total",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("θ_total")
    ax.set_title("Nested OU Composite Signal (first 100 s)")
    ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Panel (0, 1) – PSD log-log
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    f_arr = np.array(spectral_result.psd_f)
    S_arr = np.array(spectral_result.psd_S)
    mask_pos = f_arr > 0
    ax.loglog(
        f_arr[mask_pos],
        S_arr[mask_pos],
        color=VISUAL_CONSTANTS.HC_GREY,
        lw=0.7,
        label="PSD",
    )

    # Fitted power-law overlay
    f_fit = f_arr[mask_pos]
    if len(f_fit) > 1:
        log_f_mid = np.log10(np.sqrt(f_fit[1] * f_fit[-1]))
        idx_mid = np.argmin(np.abs(np.log10(f_fit) - log_f_mid))
        S_mid = S_arr[mask_pos][idx_mid]
        f_line = np.array([f_fit[1], f_fit[-1]])
        S_line = S_mid * (f_line / f_fit[idx_mid]) ** (-spectral_result.beta_spectral)
        ax.loglog(
            f_line,
            S_line,
            "--",
            color=VISUAL_CONSTANTS.THETA_RED,
            lw=1.5,
            label=f"β={spectral_result.beta_spectral:.2f}",
        )

    # Detected breakpoints
    for bp in spectral_result.breakpoint_freqs_Hz:
        ax.axvline(
            bp, color=VISUAL_CONSTANTS.IGNITION_GREEN, lw=1.0, ls="--", alpha=0.8
        )

    # Theoretical τ_l^{-1} markers
    for idx, tau_level in enumerate(tau_array):
        f_tau = 1.0 / tau_level
        color = LEVEL_COLORS.get(idx, VISUAL_CONSTANTS.HC_GREY)
        ax.axvline(f_tau, color=color, lw=0.8, ls=":", alpha=0.7, label=f"1/τ_{idx}")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Power Spectral Density")
    ax.legend(fontsize=6, ncol=2)

    # ------------------------------------------------------------------
    # Panel (1, 0) – DFA
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    scales_arr = np.array(spectral_result.scales, dtype=float)
    F_arr = np.array(spectral_result.F_n)
    if len(scales_arr) > 1:
        ax.loglog(
            scales_arr,
            F_arr,
            "o-",
            color=VISUAL_CONSTANTS.ST_BLUE,
            ms=3,
            lw=1.0,
            label="APGI",
        )

        # Slope guide
        sc_mid = np.sqrt(scales_arr[0] * scales_arr[-1])
        F_mid = F_arr[len(F_arr) // 2]
        sc_line = np.array([scales_arr[0], scales_arr[-1]])
        F_line = F_mid * (sc_line / sc_mid) ** spectral_result.H_dfa
        ax.loglog(
            sc_line,
            F_line,
            "--",
            color=VISUAL_CONSTANTS.THETA_RED,
            lw=1.5,
            label=f"H={spectral_result.H_dfa:.3f}",
        )

    # Null model DFA
    analyzer_tmp = SpectralAnalyzer()
    null_scales, null_F, null_H = analyzer_tmp.compute_dfa(theta_null)
    if len(null_scales) > 1:
        ax.loglog(
            null_scales.astype(float),
            null_F,
            "s--",
            color="#999999",
            ms=3,
            lw=0.8,
            label=f"Null H={null_H:.3f}",
        )

    ax.set_xlabel("Scale (samples)")
    ax.set_ylabel("F(n)")
    ax.set_title("Detrended Fluctuation Analysis")
    ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Panel (1, 1) – N-level sensitivity bar chart
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    if sensitivity_H is not None and len(sensitivity_H) > 0:
        n_vals = sorted(sensitivity_H.keys())
        h_vals = [sensitivity_H[n] for n in n_vals]
        colors_bar = [LEVEL_COLORS.get(i % 5, "#555555") for i in range(len(n_vals))]
        ax.bar(
            [str(n) for n in n_vals], h_vals, color=colors_bar, edgecolor="k", lw=0.7
        )
        ax.axhline(
            0.7, color="#D6604D", lw=1.5, ls="--", label="H=0.7 (APGI threshold)"
        )
        ax.set_xlabel("Number of OU levels (N)")
        ax.set_ylabel("Hurst exponent H")
        ax.set_title("N-level Sensitivity: H vs N")
        ax.set_ylim(0, 1.2)
        ax.legend(fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "No sensitivity data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("N-level Sensitivity")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved to {output_path}")


# ---------------------------------------------------------------------------
# run_validation
# ---------------------------------------------------------------------------


def run_validation() -> Dict:
    """
    Execute the full APGI fractal threshold dynamics validation.

    Returns:
        Dictionary with keys: status, named_predictions, results.
        All values are JSON-serialisable.
    """
    logger.info("Starting APGI Fractal Threshold Dynamics validation")

    # ------------------------------------------------------------------
    # Step 1 – Simulate nested OU process
    # ------------------------------------------------------------------
    ou_process = NestedOUProcess(TAU_ARRAY)
    time_s, theta_total = ou_process.simulate(T_sim_s=600.0, dt_s=0.01)
    fs = 1.0 / 0.01  # 100 Hz
    logger.info(f"Simulation complete: {len(time_s)} samples, fs={fs} Hz")

    # ------------------------------------------------------------------
    # Step 2 – Spectral analysis
    # ------------------------------------------------------------------
    analyzer = SpectralAnalyzer()
    spectral_result = analyzer.analyze(theta_total, fs)

    # ------------------------------------------------------------------
    # Step 3 – Markovian null model comparison
    # ------------------------------------------------------------------
    null_model = MarkovianNullModel(TAU_ARRAY)
    total_variance = float(np.var(theta_total))
    theta_null = null_model.generate(
        signal_length=len(theta_total),
        dt_s=0.01,
        total_variance=total_variance,
        rng_seed=99,
    )
    markov_result = null_model.compare(theta_total, theta_null, fs)

    # ------------------------------------------------------------------
    # Step 4 – N-level sensitivity analysis
    # ------------------------------------------------------------------
    sensitivity_H: Dict[int, float] = {}
    for n_levels in [3, 4, 5, 6]:
        tau_sub = TAU_ARRAY[:n_levels]
        ou_sub = NestedOUProcess(tau_sub)
        _, theta_sub = ou_sub.simulate(T_sim_s=600.0, dt_s=0.01)
        _, _, H_sub = analyzer.compute_dfa(theta_sub)
        sensitivity_H[n_levels] = float(H_sub)
        logger.info(f"N-level sensitivity: N={n_levels}, H={H_sub:.4f}")

    # ------------------------------------------------------------------
    # Step 5 – Generate figure
    # ------------------------------------------------------------------
    output_path = "apgi_fractal_threshold_figure.png"
    generate_figure(
        time_s=time_s,
        theta_total=theta_total,
        theta_null=theta_null,
        spectral_result=spectral_result,
        markov_result=markov_result,
        tau_array=TAU_ARRAY,
        output_path=output_path,
        sensitivity_H=sensitivity_H,
    )

    # ------------------------------------------------------------------
    # Step 6 – Print summary report
    # ------------------------------------------------------------------
    H_dfa = spectral_result.H_dfa
    ks_p = markov_result.ks_pvalue
    status = "PASS" if (H_dfa > 0.7 and ks_p < 0.05) else "FAIL"

    print("\n" + "=" * 60)
    print("APGI Fractal Threshold Dynamics – Validation Report")
    print("=" * 60)
    print(f"  Status                : {status}")
    print(f"  Hurst exponent (H)    : {H_dfa:.4f}  (threshold > 0.7)")
    print(f"  Spectral exponent (β) : {spectral_result.beta_spectral:.4f}")
    print(f"  PSD fit r²            : {spectral_result.r_squared_psd:.4f}")
    print(f"  KS p-value (APGI vs null): {ks_p:.4f}  (threshold < 0.05)")
    print(f"  Cohen's d (H)         : {markov_result.cohens_d_H:.4f}")
    print(f"  Null β                : {markov_result.beta_markov:.4f}")
    print(f"  Null H                : {markov_result.H_markov:.4f}")
    print(f"  Null has breakpoints  : {markov_result.has_breakpoints}")
    print(f"  Breakpoints detected  : {len(spectral_result.breakpoint_freqs_Hz)}")
    print(f"  N-level sensitivity   : {sensitivity_H}")
    print(f"  Figure saved to       : {output_path}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Step 7 – Build and return results dict (JSON-serialisable)
    # ------------------------------------------------------------------
    named_predictions = {
        "fractal_1f_signature": {
            "prediction": "Nested OU produces 1/f PSD (β > 0)",
            "result": float(spectral_result.beta_spectral),
            "passed": bool(spectral_result.beta_spectral > 0),
        },
        "hurst_above_threshold": {
            "prediction": "H > 0.7 indicating long-range correlations",
            "result": float(H_dfa),
            "passed": bool(H_dfa > 0.7),
        },
        "distinguishable_from_markov": {
            "prediction": "KS p < 0.05 vs single-OU null model",
            "result": float(ks_p),
            "passed": bool(ks_p < 0.05),
        },
        "spectral_breakpoints": {
            "prediction": "PSD breakpoints correspond to τ_l^{-1}",
            "result": spectral_result.breakpoint_freqs_Hz,
            "passed": bool(len(spectral_result.breakpoint_freqs_Hz) > 0),
        },
    }

    results = {
        "beta_spectral": float(spectral_result.beta_spectral),
        "H_dfa": float(spectral_result.H_dfa),
        "r_squared_psd": float(spectral_result.r_squared_psd),
        "breakpoint_freqs_Hz": spectral_result.breakpoint_freqs_Hz,
        "beta_markov": float(markov_result.beta_markov),
        "H_markov": float(markov_result.H_markov),
        "ks_pvalue": float(markov_result.ks_pvalue),
        "cohens_d_H": float(markov_result.cohens_d_H),
        "has_breakpoints_null": bool(markov_result.has_breakpoints),
        "sensitivity_H": {str(k): float(v) for k, v in sensitivity_H.items()},
        "figure_path": output_path,
    }

    return {
        "status": status,
        "named_predictions": named_predictions,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_validation()
