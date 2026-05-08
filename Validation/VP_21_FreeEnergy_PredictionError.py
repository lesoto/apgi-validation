#!/usr/bin/env python3
"""
VP-FE-Proxy: Free Energy Approximation via Dual-Channel PE Tracking
====================================================================

Implements proximal prediction-error (PE) tracking as an operational
approximation of variational free energy minimisation (APGI FEP framework).

EPISTEMOLOGICAL FRAMING — required for paper integrity
-------------------------------------------------------
Variational free energy F in the FEP sense is a mathematical bound on
surprise (negative log model evidence); it is NOT directly observable in
neural data.  What is tractable is a three-level proxy chain:

    Level 1 (thermodynamic):  ΔF → Landauer energy dissipation
    Level 2 (computational):  PE magnitude → F proxy
    Level 3 (neural correlate): MMN amplitude + HEP deviation → PE proxies

This protocol operates at Level 3 → Level 2.  The paper MUST state this
inference chain explicitly and must NOT claim to "track F(t) directly."
See Friston (2010) for the FEP derivation; Näätänen et al. (2007) for
MMN; Park et al. (2014) for interoceptive PE / HEP.

What is measured
----------------
Exteroceptive PE proxy — Mismatch Negativity (MMN):
    Auditory oddball paradigm (80 % standard / 20 % deviant tones).
    MMN = deviant-minus-standard ERP difference, peak in −100–+250 ms
    window after deviant onset.  Amplitude tracks exteroceptive PE
    magnitude (larger MMN → higher surprise / free energy).

LEVEL DESIGNATION: All outputs are Level 3 (algorithmic/mathematical).
Bridge to Level 2 requires APGI_Information_Theoretic_Bandwidth.
Bridge to Level 1 requires APGI_Thermodynamic_Program_Aggregator.
This script does NOT claim thermodynamic or information-theoretic implications
without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If prediction-error proxies do not correlate with free energy approximations
(r < 0.7), or if the dual-channel PE tracking fails to capture expected
deviation patterns (prediction accuracy < 80%), or if the PE-to-F mapping
does not follow theoretical predictions (error > 20%), then the APGI
free energy approximation claim is falsified. This would indicate that
PE tracking cannot serve as a reliable proxy for variational free energy.

Interoceptive PE proxy — Heartbeat-Evoked Potential (HEP) deviation:
    EEG epochs time-locked to ECG R-peaks.  HEP deviation = mean
    amplitude in the 200–500 ms post-R-peak window relative to a
    pre-task baseline.  Interoceptive prediction error increases when
    the cardiac model is imprecise; a declining HEP deviation indexes
    improved interoceptive inference.

APGI predictions (V21.1 – V21.3)
---------------------------------
V21.1 — MMN amplitude declines monotonically across a 30-minute
        sustained task (linear regression: slope < 0, R² ≥ 0.60,
        p < 0.05).  Interpretation: exteroceptive model converges,
        reducing free energy through predictive inference.

V21.2 — HEP deviation declines monotonically across the same task
        (slope < 0, R² ≥ 0.50, p < 0.05).  Interpretation:
        interoceptive model converges in parallel.

V21.3 — Ignition events produce transient PE spikes (MMN or HEP
        amplitude ≥ 1.20 × pre-ignition mean) that resolve to
        below-baseline within the subsequent 2-minute epoch — the
        "catch-and-return" signature predicted by APGI ignition theory.

Protocol design
---------------
    Task            : 30-minute auditory oddball with continuous ECG
    Block structure : Six 5-minute blocks; one block-mean PE value per signal
    Ignition probes : 10 near-threshold visual probes per block; EEG + ECG
                      both time-locked; ignition defined by P3b > 2 µV
    Participants    : N = 24 simulated (real data via BIDS-EEG root)
    Analysis        : Linear regression over 6 time points per PE channel;
                      ignition transient test on block-mean amplitudes

Tier: SECONDARY

VP_ALL_Aggregator.py registration
----------------------------------
    "V21.1": "VP_20_FreeEnergy_PredictionError",
    "V21.2": "VP_20_FreeEnergy_PredictionError",
    "V21.3": "VP_20_FreeEnergy_PredictionError",
"""

import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import mne

    HAS_MNE = True
except Exception:
    HAS_MNE = False
    mne = None  # type: ignore[assignment]

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Falsification thresholds
DEFAULT_ALPHA: float = 0.05
V21_MMN_MIN_R2: float = 0.60  # R² for exteroceptive PE decline
V21_HEP_MIN_R2: float = 0.50  # R² for interoceptive PE decline
V21_IGNITION_TRANSIENT_RATIO: float = 1.20  # spike must be ≥1.20× pre-ignition mean

# Try to override with centralized values if available
try:
    from utils.falsification_thresholds import DEFAULT_ALPHA as _DEFAULT_ALPHA
    from utils.falsification_thresholds import V21_HEP_MIN_R2 as _V21_HEP_MIN_R2
    from utils.falsification_thresholds import (
        V21_IGNITION_TRANSIENT_RATIO as _V21_IGNITION_TRANSIENT_RATIO,
    )
    from utils.falsification_thresholds import V21_MMN_MIN_R2 as _V21_MMN_MIN_R2

    DEFAULT_ALPHA = _DEFAULT_ALPHA
    V21_MMN_MIN_R2 = _V21_MMN_MIN_R2
    V21_HEP_MIN_R2 = _V21_HEP_MIN_R2
    V21_IGNITION_TRANSIENT_RATIO = _V21_IGNITION_TRANSIENT_RATIO
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
SFREQ: float = 1000.0  # Hz
N_BLOCKS: int = 6  # 5-minute blocks in a 30-minute task
BLOCK_DURATION_S: float = 300.0
N_SUBJECTS: int = 24

# MMN window: −100 to +250 ms post-deviant
MMN_WIN_LOW_MS: float = -100.0
MMN_WIN_HIGH_MS: float = 250.0
MMN_PEAK_WINDOW_MS: Tuple[float, float] = (100.0, 200.0)  # canonical MMN peak

# HEP window: 200–500 ms post R-peak
HEP_WIN_MS: Tuple[float, float] = (200.0, 500.0)

# Oddball ratio
STANDARD_PROB: float = 0.80
DEVIANT_PROB: float = 0.20
TONE_SOA_MS: float = 600.0  # stimulus onset asynchrony

# Ignition probe: P3b threshold (µV)
P3B_IGNITION_THRESHOLD_UV: float = 2.0

# Proxy chain levels for provenance labelling
PROXY_CHAIN = (
    "Level 3 (neural correlate: MMN / HEP)"
    " → Level 2 (computational: PE magnitude)"
    " → Level 1 (thermodynamic: ΔF via Landauer bridge — NOT measured here)"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BlockPE:
    """Block-level prediction-error summary for one subject."""

    subject_id: str
    block_idx: int  # 0-based, 0 = first 5 min
    mmn_amplitude_uv: float  # mean MMN peak amplitude (µV; negative = larger PE)
    hep_deviation_uv: float  # HEP deviation from pre-task baseline (µV)
    n_deviants: int
    n_heartbeats: int
    has_ignition: bool  # whether an ignition probe fell in this block
    ignition_mmn_uv: Optional[float] = None  # PE amplitude at ignition onset
    ignition_hep_uv: Optional[float] = None


@dataclass
class SubjectTrajectory:
    """Full 6-block PE trajectory for one subject."""

    subject_id: str
    blocks: List[BlockPE] = field(default_factory=list)

    def mmn_series(self) -> np.ndarray:
        return np.array([b.mmn_amplitude_uv for b in self.blocks])

    def hep_series(self) -> np.ndarray:
        return np.array([b.hep_deviation_uv for b in self.blocks])


# ---------------------------------------------------------------------------
# Signal utilities
# ---------------------------------------------------------------------------


def _bandpass(
    data: np.ndarray, low: float, high: float, fs: float, order: int = 4
) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)


def _epoch_mean(
    data: np.ndarray, fs: float, t_start_ms: float, t_end_ms: float
) -> float:
    """Mean amplitude in a time window of a 1-D ERP (t=0 at epoch centre)."""
    n = len(data)
    epoch_start_ms = -n / (2.0 * fs) * 1000.0
    i_start = max(0, int((t_start_ms - epoch_start_ms) * fs / 1000.0))
    i_end = min(n, int((t_end_ms - epoch_start_ms) * fs / 1000.0))
    if i_end <= i_start:
        return 0.0
    return float(np.mean(data[i_start:i_end]))


# ---------------------------------------------------------------------------
# MMN extractor
# ---------------------------------------------------------------------------


class MMNExtractor:
    """
    Compute the Mismatch Negativity difference wave from an auditory-oddball
    ERP dataset.

    In the synthetic path this class generates surrogate ERP waveforms with
    the expected MMN characteristics; in the empirical path it wraps an
    MNE Epochs object.

    MMN = deviant ERP − standard ERP, amplitude read in the 100–200 ms window.
    """

    def __init__(
        self, fs: float = SFREQ, rng: Optional[np.random.Generator] = None
    ) -> None:
        self.fs = fs
        self.rng = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
        self._dt = 1.0 / fs
        # Epoch length: MMN_WIN_LOW_MS to MMN_WIN_HIGH_MS
        n_total_ms = MMN_WIN_HIGH_MS - MMN_WIN_LOW_MS
        self.n_samples = int(n_total_ms * fs / 1000.0)
        self.t_ms = np.linspace(MMN_WIN_LOW_MS, MMN_WIN_HIGH_MS, self.n_samples)

    def _mmn_template(self, amplitude: float) -> np.ndarray:
        """Gaussian MMN component centred at 150 ms, σ=25 ms."""
        mu = 150.0
        sigma = 25.0
        wave = -amplitude * np.exp(-0.5 * ((self.t_ms - mu) / sigma) ** 2)
        wave[self.t_ms < 0] = 0.0
        return wave

    def simulate_block_mmn(
        self,
        n_deviants: int,
        mmn_amplitude: float,
        noise_level: float = 0.3,
    ) -> float:
        """
        Simulate trial-averaged MMN and return peak amplitude (µV).

        Parameters
        ----------
        n_deviants : int
        mmn_amplitude : float   true underlying amplitude (µV)
        noise_level : float     within-subject variability (σ of Gaussian noise)

        Returns
        -------
        float — measured peak amplitude (µV, negative)
        """
        template = self._mmn_template(mmn_amplitude)
        # Simulate trial-level noise then average
        trials = np.stack(
            [
                template
                + self.rng.normal(0.0, noise_level * mmn_amplitude, self.n_samples)
                for _ in range(n_deviants)
            ]
        )
        erp = np.mean(trials, axis=0)
        # MMN peak = most negative point in 100–200 ms window
        peak_idx = (self.t_ms >= MMN_PEAK_WINDOW_MS[0]) & (
            self.t_ms <= MMN_PEAK_WINDOW_MS[1]
        )
        return float(np.min(erp[peak_idx]))

    def extract_from_epochs(
        self,
        standard_epochs: "mne.Epochs",
        deviant_epochs: "mne.Epochs",
    ) -> float:
        """
        Extract MMN peak from real MNE Epochs objects.

        Returns the mean amplitude in the canonical MMN window (µV).
        """
        std_mean = np.mean(standard_epochs.get_data(units="uV"), axis=0)  # (ch, time)
        dev_mean = np.mean(deviant_epochs.get_data(units="uV"), axis=0)
        mmn_wave = np.mean(dev_mean - std_mean, axis=0)  # channel-averaged difference
        times_ms = standard_epochs.times * 1000.0
        mask = (times_ms >= MMN_PEAK_WINDOW_MS[0]) & (times_ms <= MMN_PEAK_WINDOW_MS[1])
        return float(np.min(mmn_wave[mask])) if np.any(mask) else 0.0


# ---------------------------------------------------------------------------
# HEP extractor
# ---------------------------------------------------------------------------


class HEPExtractor:
    """
    Compute the Heartbeat-Evoked Potential deviation from a pre-task baseline.

    In the synthetic path generates surrogate HEP waveforms.
    In the empirical path wraps MNE Epochs time-locked to ECG R-peaks.

    HEP deviation = mean amplitude in 200–500 ms post-R-peak minus baseline.
    Larger positive deviation → higher interoceptive PE.
    """

    def __init__(
        self, fs: float = SFREQ, rng: Optional[np.random.Generator] = None
    ) -> None:
        self.fs = fs
        self.rng = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
        # Epoch: −200 to +700 ms around R-peak
        self.t_ms = np.arange(-200, 701, 1000.0 / fs)
        self.n_samples = len(self.t_ms)

    def _hep_template(self, amplitude: float) -> np.ndarray:
        """Sinusoidal HEP component in 200–500 ms window."""
        wave = np.zeros(self.n_samples)
        mask = (self.t_ms >= HEP_WIN_MS[0]) & (self.t_ms <= HEP_WIN_MS[1])
        wave[mask] = amplitude * np.sin(
            np.pi * (self.t_ms[mask] - HEP_WIN_MS[0]) / (HEP_WIN_MS[1] - HEP_WIN_MS[0])
        )
        return wave

    def simulate_block_hep(
        self,
        n_heartbeats: int,
        hep_amplitude: float,
        noise_level: float = 0.25,
    ) -> float:
        """
        Simulate trial-averaged HEP and return window-mean amplitude (µV).
        """
        template = self._hep_template(hep_amplitude)
        trials = np.stack(
            [
                template
                + self.rng.normal(
                    0.0, noise_level * abs(hep_amplitude) + 0.1, self.n_samples
                )
                for _ in range(n_heartbeats)
            ]
        )
        erp = np.mean(trials, axis=0)
        mask = (self.t_ms >= HEP_WIN_MS[0]) & (self.t_ms <= HEP_WIN_MS[1])
        return float(np.mean(erp[mask]))

    def extract_from_epochs(self, r_peak_epochs: "mne.Epochs") -> float:
        """
        Extract HEP window mean from MNE Epochs (ECG R-peak locked).

        Returns mean amplitude in 200–500 ms window (µV).
        """
        data = r_peak_epochs.get_data(units="uV")  # (n_epochs, n_ch, n_times)
        mean_wave = np.mean(np.mean(data, axis=0), axis=0)  # channel+epoch average
        times_ms = r_peak_epochs.times * 1000.0
        mask = (times_ms >= HEP_WIN_MS[0]) & (times_ms <= HEP_WIN_MS[1])
        return float(np.mean(mean_wave[mask])) if np.any(mask) else 0.0


# ---------------------------------------------------------------------------
# Synthetic task simulator
# ---------------------------------------------------------------------------


class FEProxySimulator:
    """
    Generate synthetic 30-minute oddball + HEP data that instantiates the
    APGI free energy minimisation prediction.

    Trajectory design:
      - MMN amplitude starts large (high PE) and decays exponentially across
        blocks as the auditory model converges.
      - HEP deviation follows the same decay on a slightly slower timescale.
      - Ignition events (2 per task) insert a transient PE spike 1.3× the
        local pre-ignition level that resolves in the next block.

    Parameters
    ----------
    n_subjects : int
    rng : np.random.Generator or None
    """

    def __init__(
        self,
        n_subjects: int = N_SUBJECTS,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n_subjects = n_subjects
        self.rng = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
        self._mmn_ext = MMNExtractor(rng=self.rng)
        self._hep_ext = HEPExtractor(rng=self.rng)

    def _decay_trajectory(
        self,
        start: float,
        end: float,
        n_blocks: int,
        tau: float = 2.5,
        noise_sd: float = 0.05,
    ) -> np.ndarray:
        """Exponential decay from start to end with Gaussian noise."""
        t = np.arange(n_blocks, dtype=float)
        traj = end + (start - end) * np.exp(-t / tau)
        traj += self.rng.normal(0.0, noise_sd * abs(start - end), n_blocks)
        return traj

    def simulate(self) -> List[SubjectTrajectory]:
        trajectories: List[SubjectTrajectory] = []
        # Ignition blocks for all subjects: blocks 2 and 4 (0-indexed)
        ignition_blocks = {2, 4}

        for s in range(self.n_subjects):
            subject_id = f"sub-sim{s + 1:02d}"
            # Between-subject variability on starting amplitude
            scale = 1.0 + self.rng.normal(0.0, 0.15)

            # MMN: starts at ~3.5 µV (negative), converges to ~1.0 µV
            mmn_start = 3.5 * scale
            mmn_end = 1.0 * scale
            mmn_traj = self._decay_trajectory(mmn_start, mmn_end, N_BLOCKS, tau=2.5)

            # HEP: starts at ~2.0 µV deviation, converges to ~0.4 µV
            hep_start = 2.0 * scale
            hep_end = 0.4 * scale
            hep_traj = self._decay_trajectory(hep_start, hep_end, N_BLOCKS, tau=3.0)

            traj = SubjectTrajectory(subject_id=subject_id)
            for b in range(N_BLOCKS):
                n_deviants = int(
                    DEVIANT_PROB * (BLOCK_DURATION_S * 1000.0 / TONE_SOA_MS)
                )
                n_heartbeats = int(BLOCK_DURATION_S * 1.1)  # ~66 bpm

                has_ignition = b in ignition_blocks
                mmn_amp = float(mmn_traj[b])
                hep_amp = float(hep_traj[b])

                # Ignition transient: spike up then resolve
                ig_mmn: Optional[float] = None
                ig_hep: Optional[float] = None
                if has_ignition:
                    ig_mmn = (
                        mmn_amp
                        * V21_IGNITION_TRANSIENT_RATIO
                        * (1.0 + self.rng.normal(0.0, 0.05))
                    )
                    ig_hep = (
                        hep_amp
                        * V21_IGNITION_TRANSIENT_RATIO
                        * (1.0 + self.rng.normal(0.0, 0.05))
                    )

                measured_mmn = self._mmn_ext.simulate_block_mmn(n_deviants, mmn_amp)
                measured_hep = self._hep_ext.simulate_block_hep(n_heartbeats, hep_amp)

                traj.blocks.append(
                    BlockPE(
                        subject_id=subject_id,
                        block_idx=b,
                        mmn_amplitude_uv=measured_mmn,
                        hep_deviation_uv=measured_hep,
                        n_deviants=n_deviants,
                        n_heartbeats=n_heartbeats,
                        has_ignition=has_ignition,
                        ignition_mmn_uv=ig_mmn,
                        ignition_hep_uv=ig_hep,
                    )
                )
            trajectories.append(traj)

        logger.info(
            "Simulated %d subject trajectories (%d blocks each).",
            self.n_subjects,
            N_BLOCKS,
        )
        return trajectories


# ---------------------------------------------------------------------------
# Empirical BIDS loader
# ---------------------------------------------------------------------------


class BIDSFELoader:
    """
    Load auditory oddball + ECG data from a BIDS-EEG root and compute
    block-level MMN and HEP values.

    Expects:
        sub-*/eeg/*_task-oddball_eeg.(edf|set|fif|vhdr)
        sub-*/eeg/*_task-oddball_events.tsv   columns: onset, trial_type
            trial_type values: "standard" | "deviant" | "r_peak"
    """

    def __init__(
        self,
        bids_root: Union[str, Path],
        sfreq_target: float = SFREQ,
    ) -> None:
        if not HAS_MNE:
            raise ImportError("MNE-Python is required for BIDS loading.")
        self.bids_root = Path(bids_root)
        self.sfreq_target = sfreq_target
        if not self.bids_root.exists():
            raise FileNotFoundError(f"BIDS root not found: {self.bids_root}")

    def _read_raw(self, path: Path) -> "mne.io.BaseRaw":
        suffix = path.suffix.lower()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dispatch = {
                ".edf": mne.io.read_raw_edf,
                ".set": mne.io.read_raw_eeglab,
                ".fif": mne.io.read_raw_fif,
                ".vhdr": mne.io.read_raw_brainvision,
            }
            if suffix not in dispatch:
                raise ValueError(f"Unsupported format: {suffix}")
            raw = dispatch[suffix](str(path), preload=True, verbose=False)
        if raw.info["sfreq"] != self.sfreq_target:
            raw.resample(self.sfreq_target, verbose=False)
        return raw

    def _load_events(
        self, raw: "mne.io.BaseRaw", events_path: Path
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        import pandas as pd

        if not events_path.exists():
            return np.empty((0, 3), dtype=int), {}
        df = pd.read_csv(events_path, sep="\t")
        event_id = {"standard": 1, "deviant": 2, "r_peak": 3}
        rows = []
        for _, row in df.iterrows():
            tt = str(row.get("trial_type", "")).lower()
            code = event_id.get(tt, 0)
            if code == 0:
                continue
            sample = int(float(row["onset"]) * raw.info["sfreq"])
            rows.append([sample, 0, code])
        events = np.array(rows, dtype=int) if rows else np.empty((0, 3), dtype=int)
        return events, event_id

    def load(self, max_subjects: Optional[int] = None) -> List[SubjectTrajectory]:

        patterns = [
            "**/*_task-oddball_eeg.edf",
            "**/*_task-oddball_eeg.set",
            "**/*_task-oddball_eeg.fif",
            "**/*_task-oddball_eeg.vhdr",
        ]
        files: List[Path] = []
        for pat in patterns:
            files.extend(sorted(self.bids_root.glob(pat)))

        subjects: Dict[str, List[Path]] = {}
        for f in files:
            sub = next((p for p in f.parts if p.startswith("sub-")), "unknown")
            subjects.setdefault(sub, []).append(f)

        if max_subjects is not None:
            subject_keys = list(subjects.keys())[:max_subjects]
            subjects = {k: subjects[k] for k in subject_keys}

        mmn_ext = MMNExtractor()
        hep_ext = HEPExtractor()
        trajectories: List[SubjectTrajectory] = []

        for sub_id, sub_files in subjects.items():
            traj = SubjectTrajectory(subject_id=sub_id)
            for f in sub_files:
                try:
                    raw = self._read_raw(f)
                    events_path = Path(str(f).rsplit("_eeg.", 1)[0] + "_events.tsv")
                    events, event_id = self._load_events(raw, events_path)
                    if len(events) == 0:
                        continue
                    # Split into N_BLOCKS equal time windows
                    total_s = raw.times[-1]
                    block_s = total_s / N_BLOCKS
                    for b in range(N_BLOCKS):
                        t_start = b * block_s
                        t_end = (b + 1) * block_s
                        block_mask = (events[:, 0] / raw.info["sfreq"] >= t_start) & (
                            events[:, 0] / raw.info["sfreq"] < t_end
                        )
                        block_events = events[block_mask]

                        std_ev = block_events[block_events[:, 2] == 1]
                        dev_ev = block_events[block_events[:, 2] == 2]
                        rp_ev = block_events[block_events[:, 2] == 3]

                        mmn_val = 0.0
                        if len(std_ev) >= 5 and len(dev_ev) >= 5:
                            ep_std = mne.Epochs(
                                raw,
                                std_ev,
                                tmin=-0.1,
                                tmax=0.25,
                                baseline=(-0.1, 0),
                                preload=True,
                                verbose=False,
                            )
                            ep_dev = mne.Epochs(
                                raw,
                                dev_ev,
                                tmin=-0.1,
                                tmax=0.25,
                                baseline=(-0.1, 0),
                                preload=True,
                                verbose=False,
                            )
                            mmn_val = mmn_ext.extract_from_epochs(ep_std, ep_dev)

                        hep_val = 0.0
                        if len(rp_ev) >= 5:
                            ep_rp = mne.Epochs(
                                raw,
                                rp_ev,
                                tmin=-0.2,
                                tmax=0.7,
                                baseline=(-0.2, 0),
                                preload=True,
                                verbose=False,
                            )
                            hep_val = hep_ext.extract_from_epochs(ep_rp)

                        traj.blocks.append(
                            BlockPE(
                                subject_id=sub_id,
                                block_idx=b,
                                mmn_amplitude_uv=mmn_val,
                                hep_deviation_uv=hep_val,
                                n_deviants=len(dev_ev),
                                n_heartbeats=len(rp_ev),
                                has_ignition=False,
                            )
                        )
                except Exception as exc:
                    logger.warning("Could not process %s block %d: %s", sub_id, b, exc)
            if traj.blocks:
                trajectories.append(traj)

        logger.info("Loaded %d subject trajectories from BIDS root.", len(trajectories))
        return trajectories


# ---------------------------------------------------------------------------
# Temporal-regression PE-decline tests
# ---------------------------------------------------------------------------


def _linear_regression_stats(
    y: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    OLS regression of y on block index [0, N-1].

    Returns
    -------
    slope, intercept, r_squared, p_value (one-sided: slope < 0)
    """
    x = np.arange(len(y), dtype=float)
    result = stats.linregress(x, y)
    slope = float(result.slope)
    intercept = float(result.intercept)
    r2 = float(result.rvalue**2)
    # One-sided p-value: slope < 0
    p_one = float(result.pvalue / 2.0) if slope < 0 else 1.0
    return slope, intercept, r2, p_one


def test_monotone_decline(
    trajectories: List[SubjectTrajectory],
    signal: str,  # "mmn" | "hep"
    min_r2: float,
) -> Dict[str, Any]:
    """
    Test V21.1 (MMN) or V21.2 (HEP): group-mean PE declines across blocks.

    Strategy: compute group-mean trajectory (N_BLOCKS values), then run
    OLS regression against block index.  Individual-subject slopes are also
    computed for a one-sample t-test against zero.

    Parameters
    ----------
    signal : "mmn" | "hep"
    min_r2 : minimum R² for the group-mean regression to pass

    Returns
    -------
    dict with slope, r2, pvalue, subject_slopes, passed
    """
    per_subject = []
    for traj in trajectories:
        if len(traj.blocks) < N_BLOCKS:
            continue
        vals = traj.mmn_series() if signal == "mmn" else traj.hep_series()
        per_subject.append(vals)

    if not per_subject:
        return {
            "signal": signal,
            "slope": 0.0,
            "r2": 0.0,
            "pvalue": 1.0,
            "subject_slopes": [],
            "passed": False,
        }

    mat = np.vstack(per_subject)  # (n_subjects, N_BLOCKS)
    group_mean = np.mean(mat, axis=0)  # (N_BLOCKS,)

    slope, intercept, r2, p_group = _linear_regression_stats(group_mean)

    # Per-subject slopes for secondary t-test
    sub_slopes: List[float] = []
    for row in mat:
        s, *_ = _linear_regression_stats(row)
        sub_slopes.append(s)
    arr_slopes = np.array(sub_slopes)
    t_stat, p_t = stats.ttest_1samp(arr_slopes, 0.0)
    p_t_one = float(p_t / 2.0) if t_stat < 0 else 1.0

    # Pass if group regression meets R² and both group + subject tests significant
    passed = (
        slope < 0
        and r2 >= min_r2
        and p_group < DEFAULT_ALPHA
        and p_t_one < DEFAULT_ALPHA
    )

    label = "V21.1 (MMN)" if signal == "mmn" else "V21.2 (HEP)"
    logger.info(
        "%s: slope=%.4f, R²=%.3f, p_group=%.4f, p_subj=%.4f — %s",
        label,
        slope,
        r2,
        p_group,
        p_t_one,
        "PASS" if passed else "FAIL",
    )
    return {
        "signal": signal,
        "group_mean_trajectory": group_mean.tolist(),
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "p_group": p_group,
        "p_subjects": p_t_one,
        "subject_slopes": arr_slopes.tolist(),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Ignition transient test (V21.3)
# ---------------------------------------------------------------------------


def test_ignition_transient(
    trajectories: List[SubjectTrajectory],
    ratio_threshold: float = V21_IGNITION_TRANSIENT_RATIO,
) -> Dict[str, Any]:
    """
    V21.3: Ignition events produce transient PE spikes that resolve in the
    subsequent block.

    Spike condition: ignition-block PE ≥ ratio_threshold × pre-ignition mean.
    Resolution condition: post-ignition block PE < pre-ignition mean.

    Both conditions must hold for a majority of ignition events (> 50 %).
    """
    mmn_spike_ratios: List[float] = []
    hep_spike_ratios: List[float] = []
    n_resolved_mmn = 0
    n_resolved_hep = 0
    n_ignitions = 0

    for traj in trajectories:
        blocks = traj.blocks
        for b_idx, block in enumerate(blocks):
            if not block.has_ignition:
                continue
            if b_idx == 0 or b_idx >= len(blocks) - 1:
                continue  # need pre and post block

            pre_mmn = abs(blocks[b_idx - 1].mmn_amplitude_uv)
            pre_hep = abs(blocks[b_idx - 1].hep_deviation_uv)

            ig_mmn = (
                abs(block.ignition_mmn_uv)
                if block.ignition_mmn_uv is not None
                else abs(block.mmn_amplitude_uv)
            )
            ig_hep = (
                abs(block.ignition_hep_uv)
                if block.ignition_hep_uv is not None
                else abs(block.hep_deviation_uv)
            )

            post_mmn = abs(blocks[b_idx + 1].mmn_amplitude_uv)
            post_hep = abs(blocks[b_idx + 1].hep_deviation_uv)

            if pre_mmn > 0:
                mmn_spike_ratios.append(ig_mmn / pre_mmn)
                if post_mmn < pre_mmn:
                    n_resolved_mmn += 1

            if pre_hep > 0:
                hep_spike_ratios.append(ig_hep / pre_hep)
                if post_hep < pre_hep:
                    n_resolved_hep += 1

            n_ignitions += 1

    if n_ignitions == 0:
        logger.warning("V21.3: no ignition events found.")
        return {
            "n_ignitions": 0,
            "mmn_spike_ratio_mean": 0.0,
            "hep_spike_ratio_mean": 0.0,
            "mmn_resolution_rate": 0.0,
            "hep_resolution_rate": 0.0,
            "passed": False,
        }

    mmn_ratio_mean = float(np.mean(mmn_spike_ratios)) if mmn_spike_ratios else 0.0
    hep_ratio_mean = float(np.mean(hep_spike_ratios)) if hep_spike_ratios else 0.0
    mmn_res_rate = n_resolved_mmn / n_ignitions
    hep_res_rate = n_resolved_hep / n_ignitions

    passed = (
        mmn_ratio_mean >= ratio_threshold
        and hep_ratio_mean >= ratio_threshold
        and mmn_res_rate > 0.50
        and hep_res_rate > 0.50
    )

    logger.info(
        "V21.3: MMN spike=%.2f× (res=%.0f%%), HEP spike=%.2f× (res=%.0f%%) — %s",
        mmn_ratio_mean,
        mmn_res_rate * 100,
        hep_ratio_mean,
        hep_res_rate * 100,
        "PASS" if passed else "FAIL",
    )
    return {
        "n_ignitions": n_ignitions,
        "mmn_spike_ratio_mean": mmn_ratio_mean,
        "hep_spike_ratio_mean": hep_ratio_mean,
        "mmn_resolution_rate": mmn_res_rate,
        "hep_resolution_rate": hep_res_rate,
        "ratio_threshold": ratio_threshold,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def generate_figures(
    mmn_result: Dict[str, Any],
    hep_result: Dict[str, Any],
    ig_result: Dict[str, Any],
    data_source: str,
) -> None:
    if not HAS_MATPLOTLIB:
        return

    out_dir = _project_root / "outputs"
    out_dir.mkdir(exist_ok=True)
    block_x = np.arange(1, N_BLOCKS + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # V21.1: MMN trajectory
    ax = axes[0]
    mmn_traj = mmn_result.get("group_mean_trajectory", [])
    if mmn_traj:
        ax.plot(block_x, mmn_traj, "o-", color="steelblue", label="Group mean")
        slope = mmn_result["slope"]
        interc = mmn_result["intercept"]
        fit_x = np.array([0, N_BLOCKS - 1], dtype=float)
        ax.plot(
            block_x[[0, -1]],
            slope * fit_x + interc,
            "--",
            color="steelblue",
            alpha=0.6,
            label=f"OLS slope={slope:.3f}",
        )
    ax.set_xlabel("Block (5 min each)")
    ax.set_ylabel("MMN amplitude (µV)")
    ax.set_title(
        f"V21.1 — Exteroceptive PE (MMN)\n"
        f"R²={mmn_result.get('r2', 0):.3f}, "
        f"p={mmn_result.get('p_group', 1):.4f} "
        f"{'✓' if mmn_result.get('passed') else '✗'}"
    )
    ax.legend(fontsize=8)
    ax.invert_yaxis()  # more negative = larger MMN = more PE

    # V21.2: HEP trajectory
    ax2 = axes[1]
    hep_traj = hep_result.get("group_mean_trajectory", [])
    if hep_traj:
        ax2.plot(block_x, hep_traj, "o-", color="tomato", label="Group mean")
        slope2 = hep_result["slope"]
        interc2 = hep_result["intercept"]
        ax2.plot(
            block_x[[0, -1]],
            slope2 * fit_x + interc2,
            "--",
            color="tomato",
            alpha=0.6,
            label=f"OLS slope={slope2:.3f}",
        )
    ax2.set_xlabel("Block (5 min each)")
    ax2.set_ylabel("HEP deviation (µV)")
    ax2.set_title(
        f"V21.2 — Interoceptive PE (HEP)\n"
        f"R²={hep_result.get('r2', 0):.3f}, "
        f"p={hep_result.get('p_group', 1):.4f} "
        f"{'✓' if hep_result.get('passed') else '✗'}"
    )
    ax2.legend(fontsize=8)
    ax2.invert_yaxis()

    # V21.3: Ignition transients
    ax3 = axes[2]
    mmn_r = ig_result.get("mmn_spike_ratio_mean", 0)
    hep_r = ig_result.get("hep_spike_ratio_mean", 0)
    ax3.bar(["MMN spike", "HEP spike"], [mmn_r, hep_r], color=["steelblue", "tomato"])
    ax3.axhline(
        V21_IGNITION_TRANSIENT_RATIO,
        color="red",
        linestyle="--",
        label=f"Threshold ({V21_IGNITION_TRANSIENT_RATIO:.2f}×)",
    )
    ax3.set_ylabel("Spike ratio (ignition / pre-ignition)")
    ax3.set_title(
        f"V21.3 — Ignition PE transient\n"
        f"MMN res={ig_result.get('mmn_resolution_rate', 0):.0%}, "
        f"HEP res={ig_result.get('hep_resolution_rate', 0):.0%} "
        f"{'✓' if ig_result.get('passed') else '✗'}"
    )
    ax3.legend(fontsize=8)

    proxy_note = "3-level proxy chain: MMN/HEP (L3) → PE (L2) → F(t) (L1, not measured)"
    fig.suptitle(
        f"VP-FE-Proxy: Free Energy Approximation via PE Tracking — {data_source.upper()} data\n"
        f"{proxy_note}",
        fontsize=10,
    )
    fig.tight_layout()
    path = out_dir / "VP_20_FreeEnergy_PredictionError.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


# ---------------------------------------------------------------------------
# Top-level validator
# ---------------------------------------------------------------------------


class FEProxyValidator:
    """
    Orchestrates the full free-energy proxy pipeline.

    When bids_root is supplied and MNE-Python is available, real EEG data
    are loaded.  Otherwise the synthetic simulator is used and all results
    are flagged 'synthetic'.

    Parameters
    ----------
    bids_root : path-like or None
    seed : int
    max_subjects : int or None
    """

    def __init__(
        self,
        bids_root: Optional[Union[str, Path]] = None,
        seed: int = RANDOM_SEED,
        max_subjects: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.bids_root = Path(bids_root) if bids_root is not None else None
        self.seed = seed
        self.max_subjects = max_subjects
        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self._trajectories: Optional[List[SubjectTrajectory]] = None
        self.data_source: str = "synthetic"

    def _load(self) -> List[SubjectTrajectory]:
        if self._trajectories is not None:
            return self._trajectories

        if self.bids_root is not None and HAS_MNE:
            try:
                loader = BIDSFELoader(self.bids_root)
                trajs = loader.load(max_subjects=self.max_subjects)
                if trajs:
                    self.data_source = "empirical"
                    self._trajectories = trajs
                    return trajs
            except FileNotFoundError as exc:
                logger.warning("BIDS load failed (%s); using synthetic data.", exc)
            except Exception as exc:
                logger.warning("BIDS load error (%s); using synthetic data.", exc)
        elif self.bids_root is not None and not HAS_MNE:
            logger.warning("MNE-Python unavailable; using synthetic data.")

        sim = FEProxySimulator(rng=self.rng)
        self._trajectories = sim.simulate()
        self.data_source = "synthetic"
        return self._trajectories

    def run_full_validation(self) -> Dict[str, Any]:
        """
        Run V21.1, V21.2, V21.3 analyses and return results dict.

        Returns
        -------
        dict with keys: overall_score, tests_passed, tests_total,
            data_source, proxy_chain, V21.1, V21.2, V21.3,
            measurement_gap_note
        """
        trajectories = self._load()
        n_subjects = len(trajectories)

        mmn_result = test_monotone_decline(trajectories, "mmn", V21_MMN_MIN_R2)
        hep_result = test_monotone_decline(trajectories, "hep", V21_HEP_MIN_R2)
        ig_result = test_ignition_transient(trajectories)

        generate_figures(mmn_result, hep_result, ig_result, self.data_source)

        v21_1_passed = bool(mmn_result["passed"])
        v21_2_passed = bool(hep_result["passed"])
        v21_3_passed = bool(ig_result["passed"])

        tests_passed = sum([v21_1_passed, v21_2_passed, v21_3_passed])
        tests_total = 3

        measurement_gap = (
            "PROXY CHAIN (three levels): "
            "(L3 → L2) MMN amplitude and HEP deviation are neural correlates of "
            "extero- and interoceptive PE respectively — they do NOT measure "
            "variational free energy F(t) directly. "
            "(L2 → L1) F(t) → Landauer thermodynamic dissipation requires the "
            "Landauer bridge and PET/fMRS metabolic data — NOT provided here. "
            "The paper MUST NOT claim to 'track F(t) directly'; the correct "
            "framing is 'proximal PE tracking as an operational Level-3 "
            "approximation of free energy minimisation'. "
            f"Data source: {self.data_source.upper()}."
        )

        results: Dict[str, Any] = {
            "protocol": "VP_20_FreeEnergy_PredictionError",
            "data_source": self.data_source,
            "n_subjects": n_subjects,
            "n_blocks": N_BLOCKS,
            "proxy_chain": PROXY_CHAIN,
            "overall_score": tests_passed / tests_total,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "V21.1": {
                "test_name": "MMN amplitude declines monotonically (exteroceptive PE, slope < 0)",
                "passed": v21_1_passed,
                "slope": mmn_result.get("slope", 0.0),
                "r2": mmn_result.get("r2", 0.0),
                "p_group": mmn_result.get("p_group", 1.0),
                "p_subjects": mmn_result.get("p_subjects", 1.0),
                "threshold_r2": V21_MMN_MIN_R2,
                "group_mean_trajectory": mmn_result.get("group_mean_trajectory", []),
            },
            "V21.2": {
                "test_name": "HEP deviation declines monotonically (interoceptive PE, slope < 0)",
                "passed": v21_2_passed,
                "slope": hep_result.get("slope", 0.0),
                "r2": hep_result.get("r2", 0.0),
                "p_group": hep_result.get("p_group", 1.0),
                "p_subjects": hep_result.get("p_subjects", 1.0),
                "threshold_r2": V21_HEP_MIN_R2,
                "group_mean_trajectory": hep_result.get("group_mean_trajectory", []),
            },
            "V21.3": {
                "test_name": "Ignition events produce transient PE spikes that resolve post-ignition",
                "passed": v21_3_passed,
                "mmn_spike_ratio": ig_result.get("mmn_spike_ratio_mean", 0.0),
                "hep_spike_ratio": ig_result.get("hep_spike_ratio_mean", 0.0),
                "mmn_resolution_rate": ig_result.get("mmn_resolution_rate", 0.0),
                "hep_resolution_rate": ig_result.get("hep_resolution_rate", 0.0),
                "threshold_ratio": V21_IGNITION_TRANSIENT_RATIO,
                "n_ignitions": ig_result.get("n_ignitions", 0),
            },
            "measurement_gap_note": measurement_gap,
        }

        logger.info(
            "VP-FE-Proxy complete: %d/%d passed (%.0f%%) [%s data]",
            tests_passed,
            tests_total,
            results["overall_score"] * 100,
            self.data_source,
        )
        return results


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_validation(
    bids_root: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
    max_subjects: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Entry point for VP_ALL_Aggregator and external runners.

    Parameters
    ----------
    bids_root : path to BIDS-EEG root, or None for synthetic fallback
    seed : RNG seed
    max_subjects : cap subjects (useful for fast CI tests)
    """
    _seed = seed if seed is not None else RANDOM_SEED
    validator = FEProxyValidator(
        bids_root=bids_root, seed=_seed, max_subjects=max_subjects
    )
    return validator.run_full_validation()


def validate(**kwargs: Any) -> Dict[str, Any]:
    """Alias for main.py compatibility."""
    return run_validation(**kwargs)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="VP-FE-Proxy: PE tracking as F(t) approximation"
    )
    parser.add_argument("--bids-root", default=None, help="BIDS EEG dataset root")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("VP-FE-Proxy: Free Energy Approximation via Dual-Channel PE Tracking")
    print("=" * 70)
    print(f"\nProxy chain: {PROXY_CHAIN}\n")

    results = run_validation(
        bids_root=args.bids_root,
        seed=args.seed,
        max_subjects=args.max_subjects,
    )

    print(f"Data source   : {results['data_source'].upper()}")
    print(f"Subjects      : {results['n_subjects']}")
    print(f"Overall Score : {results['overall_score']:.2%}")
    print(f"Tests Passed  : {results['tests_passed']}/{results['tests_total']}")

    print("\nNamed Predictions:")
    for key in ("V21.1", "V21.2", "V21.3"):
        v = results[key]
        status = "PASS" if v["passed"] else "FAIL"
        print(f"  [{status}] {v['test_name']}")

    print(f"\nMeasurement Gap:\n  {results['measurement_gap_note']}")


if __name__ == "__main__":
    main()
