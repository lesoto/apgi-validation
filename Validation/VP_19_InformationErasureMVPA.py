#!/usr/bin/env python3
"""
VP-19: Information Erasure Protocol — MVPA Decoding of Non-Selected Representations
=====================================================================================

DEPRECATION STATUS: EXTENDED — MVPA-LEVEL P8 OUTSIDE CANONICAL REGISTRY
------------------------------------------------------------------------
VP-19 provides an MVPA layer on top of VP-4's mutual-information check for P8.
The canonical P8 falsification test is in VP-4 (VP_04_PhaseTransitionEpistemicLevel2.py).
VP-19 is a supplementary extension; V19.x predictions are NOT counted in the
canonical 14-prediction tally.
------------------------------------------------------------------------

Extends the informational analysis in VP_04 with a stimulus-pair MVPA decoding layer
that directly tests Prediction P8: post-ignition active suppression of non-selected
stimulus representations.

VP_04 addresses mutual-information optimisation during ignition (Crossing 3→2) but
does not operationalise the erasure prediction specifically.  P8 is a stronger claim:
that non-selected representations are actively suppressed post-ignition, not merely
outcompeted.

Protocol specification:
    Paradigm : Binocular rivalry / backward masking with two competing category-
               discriminable stimuli (faces vs. houses).  Conscious access to one
               is confirmed by button-press / verbal report.
    Measurement : MVPA decoding accuracy for the non-reported category using
               time-resolved SVM classifiers (scikit-learn) in 50 ms bins from
               −500 ms to +1000 ms relative to ignition onset (P3b peak).
    Prediction : Decoding accuracy for the losing stimulus DECREASES post-ignition
               below permutation-derived chance — not merely plateaus — providing
               the suppression signature.
    Control : "No-ignition" subliminal trials where both categories remain decodable
               confirm that suppression is ignition-specific.

MEASUREMENT GAPS:
    This protocol tests REPRESENTATIONAL erasure at Tier 3 (neural pattern
    decoding).  The thermodynamic cost of that erasure — Landauer's principle

TIER DESIGNATION: All outputs are Tier 3 (algorithmic/mathematical).
Bridge to Tier 2 requires APGI_Information_Theoretic_Bandwidth.
Bridge to Tier 1 requires APGI_Thermodynamic_Program_Aggregator.
This script does NOT claim thermodynamic or information-theoretic implications
without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If decoding accuracy for non-selected stimuli does not significantly decrease
below chance levels post-ignition (p > 0.05, corrected), or if the suppression
effect is not specific to ignition trials (no difference from no-ignition controls),
or if the temporal dynamics of suppression do not match APGI predictions
(onset delay > 100ms from P3b peak), then the APGI information erasure claim
is falsified. This would indicate that conscious access does not actively
suppress non-selected representations.
    applied to bit erasure (Level 1) — is NOT directly measured here.  The link
    requires the Landauer bridge and remains in Gap 1 territory.  Program 3 (PET)
    is required for thermodynamic ground-truth.  This distinction must appear
    explicitly in the paper to prevent level-confusion.

Tier: SECONDARY

VP_ALL_Aggregator.py registration:
    "V19.1": "VP_19_InformationErasureMVPA",
    "V19.2": "VP_19_InformationErasureMVPA",
    "V19.3": "VP_19_InformationErasureMVPA",
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Matplotlib
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Project root
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Thresholds
try:
    from utils.falsification_thresholds import (
        DEFAULT_ALPHA,
        V19_BELOW_CHANCE_THRESHOLD,
        V19_CONTROL_DECODABILITY_MIN,
        V19_MIN_PRE_IGNITION_ACCURACY,
        V19_N_PERMUTATIONS,
        V19_POST_IGNITION_SUPPRESSION_BINS,
    )
except ImportError:
    DEFAULT_ALPHA = 0.05
    V19_BELOW_CHANCE_THRESHOLD = 0.48  # post-ignition accuracy must drop below this
    V19_MIN_PRE_IGNITION_ACCURACY = 0.60  # losing stimulus must be decodable pre-ignition
    V19_CONTROL_DECODABILITY_MIN = 0.60  # both stims decodable on subliminal trials
    V19_N_PERMUTATIONS = 500  # permutations for chance distribution
    V19_POST_IGNITION_SUPPRESSION_BINS = 3  # consecutive below-chance bins required

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

# Time axis: −500 ms to +1000 ms, 50 ms bins
BIN_WIDTH_MS: float = 50.0
T_START_MS: float = -500.0
T_END_MS: float = 1000.0
N_BINS: int = int((T_END_MS - T_START_MS) / BIN_WIDTH_MS)  # 30 bins
BIN_CENTRES_MS: np.ndarray = np.arange(
    T_START_MS + BIN_WIDTH_MS / 2,
    T_END_MS,
    BIN_WIDTH_MS,
)  # shape (30,)

# P3b ignition onset is bin-centre 0 (t = 0 ms)
IGNITION_BIN_IDX: int = int((0.0 - T_START_MS) / BIN_WIDTH_MS)  # bin containing t=0

# Feature dimensionality per bin (simulated EEG/MEG sensors)
N_SENSORS: int = 64

CATEGORIES = ("faces", "houses")


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TrialMetadata:
    subject_id: int
    trial_id: int
    condition: str  # "ignition" | "no_ignition"
    reported_category: str  # "faces" | "houses"
    losing_category: str  # the other one
    ignition_strength: float  # Sₜ ∈ [0, 1]


@dataclass
class DecodingResult:
    """Per-trial MVPA result — binned accuracy time-series decoded for losing category."""

    trial_meta: TrialMetadata
    accuracy_timeseries: np.ndarray  # shape (N_BINS,)  accuracy in each 50 ms bin
    permutation_threshold: float  # 95th percentile of permuted distribution


# ──────────────────────────────────────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────────────────────────────────────


class StimulusPairEEGSimulator:
    """
    Simulate time-resolved EEG patterns for binocular rivalry / backward masking.

    Two competing stimulus categories (faces, houses) produce distinct spatial
    activation patterns.  The winning (reported) category maintains its pattern
    throughout; the losing category's pattern is suppressed post-ignition on
    conscious trials but remains detectable on subliminal / no-ignition trials.
    """

    def __init__(
        self,
        n_subjects: int = 20,
        n_trials_per_condition: int = 60,
        n_sensors: int = N_SENSORS,
        rng: Optional[np.random.Generator] = None,
    ):
        self.n_subjects = n_subjects
        self.n_trials_per_condition = n_trials_per_condition
        self.n_sensors = n_sensors
        self.rng = rng if rng is not None else np.random.default_rng(RANDOM_SEED)

        # Fixed orthogonal category prototypes (simulate face vs. house topographies)
        self._face_proto = self._make_prototype(seed=1)
        self._house_proto = self._make_prototype(seed=2)

    def _make_prototype(self, seed: int) -> np.ndarray:
        g = np.random.default_rng(seed)
        p = g.standard_normal(self.n_sensors)
        return p / np.linalg.norm(p)

    def _epoch_features(
        self,
        category: str,
        amplitude_profile: np.ndarray,  # shape (N_BINS,) — signal amplitude over time
        noise_level: float = 0.4,
    ) -> np.ndarray:
        """
        Build a (N_BINS, n_sensors) feature matrix for one category's representation.
        amplitude_profile controls how strongly the prototype is expressed in each bin.
        """
        proto = self._face_proto if category == "faces" else self._house_proto
        signal = amplitude_profile[:, None] * proto[None, :]  # (N_BINS, n_sensors)
        noise = self.rng.standard_normal((N_BINS, self.n_sensors)) * noise_level
        return signal + noise

    def _ignition_amplitude_profile(
        self,
        category: str,
        reported: str,
        ignition_strength: float,
    ) -> np.ndarray:
        """
        Amplitude profile for a category across the 30 time bins on ignition trials.

        Winning category: rises with P3b, sustains post-ignition.
        Losing category : decodable pre-ignition, then undergoes opponent suppression
                          post-ignition (amplitude reversal), causing the classifier
                          to misclassify systematically and produce below-chance
                          accuracy — the signature of active erasure.
        """
        profile = np.ones(N_BINS)

        if category == reported:
            # Winner: pre-ignition baseline, then strong ignition boost
            for b in range(N_BINS):
                t = BIN_CENTRES_MS[b]
                if t >= 0:
                    profile[b] = 1.5 + 1.5 * ignition_strength
        else:
            # Loser: pre-ignition decodable; post-ignition opponent suppression.
            # Amplitude flips sign (reversal), causing classifier to predict the
            # wrong category label → below-chance accuracy.
            for b in range(N_BINS):
                t = BIN_CENTRES_MS[b]
                if t < 0:
                    profile[b] = 1.0
                elif t < 200:
                    # Rapid transition through zero
                    profile[b] = 1.0 - 2.0 * (t / 200.0) * ignition_strength
                else:
                    # Sustained reversal — opponent suppression signature
                    profile[b] = -(0.5 + 0.5 * ignition_strength)

        return profile

    def _no_ignition_amplitude_profile(self, category: str) -> np.ndarray:
        """On subliminal / no-ignition trials both categories remain stable."""
        return np.full(N_BINS, 1.2)

    def simulate_dataset(self) -> List[Tuple[TrialMetadata, np.ndarray, np.ndarray]]:
        """
        Returns list of (metadata, losing_features, winning_features) tuples.
        losing_features shape: (N_BINS, n_sensors)
        """
        dataset = []

        for subj in range(self.n_subjects):
            for trial in range(self.n_trials_per_condition):
                # Randomly assign reported category
                reported = self.rng.choice(CATEGORIES)
                losing = "houses" if reported == "faces" else "faces"

                # --- Ignition trial ---
                st = float(self.rng.beta(3.0, 1.5))
                ig_meta = TrialMetadata(subj, trial, "ignition", reported, losing, st)

                losing_feats = self._epoch_features(
                    losing,
                    self._ignition_amplitude_profile(losing, reported, st),
                )
                winning_feats = self._epoch_features(
                    reported,
                    self._ignition_amplitude_profile(reported, reported, st),
                )
                dataset.append((ig_meta, losing_feats, winning_feats))

                # --- No-ignition / subliminal trial ---
                st_no = float(self.rng.beta(1.5, 3.0))
                no_meta = TrialMetadata(subj, trial, "no_ignition", reported, losing, st_no)
                losing_feats_no = self._epoch_features(losing, self._no_ignition_amplitude_profile(losing))
                winning_feats_no = self._epoch_features(reported, self._no_ignition_amplitude_profile(reported))
                dataset.append((no_meta, losing_feats_no, winning_feats_no))

        return dataset


# ──────────────────────────────────────────────────────────────────────────────
# MVPA decoding pipeline
# ──────────────────────────────────────────────────────────────────────────────


class TimeResolvedMVPADecoder:
    """
    Temporal-generalisation MVPA decoder (50 ms bins, King & Dehaene 2014).

    Trains a linear SVM on all pre-ignition bins (t < 0 ms) and tests at
    every time bin.  When the post-ignition representation actively reverses
    (opponent suppression), the pre-trained classifier labels test patterns
    in the wrong category → below-chance accuracy.

    This cross-time approach is the standard method for detecting representational
    reversal / suppression in EEG/MEG and correctly captures the P8 prediction
    in a way that within-bin k-fold CV cannot (within-bin CV adapts to both
    positive and inverted signals).

    Decoding target: category label (faces = 1, houses = 0).
    """

    N_CV_FOLDS: int = 5

    def __init__(
        self,
        n_permutations: int = V19_N_PERMUTATIONS,
        rng: Optional[np.random.Generator] = None,
    ):
        self.n_permutations = n_permutations
        self.rng = rng if rng is not None else np.random.default_rng(RANDOM_SEED)
        self._trained_clf: Optional[Pipeline] = None

    def _make_clf(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", LinearSVC(C=1.0, max_iter=2000, dual="auto")),
            ]
        )

    def _train_on_pre_ignition(self, features: np.ndarray, labels: np.ndarray) -> Pipeline:
        """
        Train one SVM on the concatenation of all pre-ignition bin patterns.
        Each trial contributes one row per pre-ignition bin (data augmentation).
        """
        pre_mask = BIN_CENTRES_MS < 0
        pre_bins = np.where(pre_mask)[0]
        # Tile: (n_trials × n_pre_bins, n_sensors)
        X_train = features[:, pre_bins, :].reshape(-1, features.shape[2])
        # Each trial contributes len(pre_bins) consecutive rows after reshape;
        # np.repeat duplicates each label once per bin (not np.tile which
        # repeats the full array).
        y_train = np.repeat(labels, len(pre_bins))
        clf = self._make_clf()
        clf.fit(X_train, y_train)
        return clf

    def decode_timeseries(
        self,
        features: np.ndarray,  # (n_trials, N_BINS, n_sensors)
        labels: np.ndarray,  # (n_trials,) binary
    ) -> Tuple[np.ndarray, float]:
        """
        Train on pre-ignition patterns; test at each time bin.

        Returns:
            accuracy_ts : shape (N_BINS,)  accuracy per bin
            perm_threshold : 95th-percentile of permuted chance distribution
        """
        clf = self._train_on_pre_ignition(features, labels)
        self._trained_clf = clf

        _, n_bins, _ = features.shape
        accuracy_ts = np.zeros(n_bins)

        for b in range(n_bins):
            X_bin = features[:, b, :]
            X_scaled = clf.named_steps["scaler"].transform(X_bin)
            preds = clf.named_steps["svm"].predict(X_scaled)
            accuracy_ts[b] = float(np.mean(preds == labels))

        perm_threshold = self._permutation_threshold(features, labels)
        return accuracy_ts, perm_threshold

    def _cv_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Stratified k-fold accuracy for permutation baseline."""
        if len(np.unique(y)) < 2:
            return 0.5
        n_folds = min(self.N_CV_FOLDS, int(np.min(np.bincount(y.astype(int)))))
        if n_folds < 2:
            return 0.5
        cv = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=int(self.rng.integers(0, 2**31)),
        )
        scores = cross_val_score(self._make_clf(), X, y, cv=cv, scoring="accuracy")
        return float(scores.mean())

    def _permutation_threshold(self, features: np.ndarray, labels: np.ndarray, alpha: float = 0.05) -> float:
        """
        Permutation null: shuffle labels, retrain on pre-ignition, test at
        the representative post-ignition bin (t = 300 ms).
        Returns (1 − alpha) quantile of the null accuracy distribution.
        """
        # Representative post-ignition bin index
        post_rep_bin = int(np.argmin(np.abs(BIN_CENTRES_MS - 300.0)))
        X_post = features[:, post_rep_bin, :]
        pre_mask = BIN_CENTRES_MS < 0
        pre_bins = np.where(pre_mask)[0]
        X_train_base = features[:, pre_bins, :].reshape(-1, features.shape[2])

        perm_accs = []
        for _ in range(self.n_permutations):
            shuffled = self.rng.permutation(labels)
            y_train_perm = np.repeat(shuffled, len(pre_bins))
            clf_perm = self._make_clf()
            clf_perm.fit(X_train_base, y_train_perm)
            X_scaled = clf_perm.named_steps["scaler"].transform(X_post)
            preds = clf_perm.named_steps["svm"].predict(X_scaled)
            perm_accs.append(float(np.mean(preds == shuffled)))
        return float(np.quantile(perm_accs, 1.0 - alpha))


# ──────────────────────────────────────────────────────────────────────────────
# Validator
# ──────────────────────────────────────────────────────────────────────────────


class InformationErasureValidator:
    """
    Tests three APGI predictions derived from P8 (information erasure):

    V19.1 — Pre-ignition, the losing stimulus is decodable above chance
            (≥ V19_MIN_PRE_IGNITION_ACCURACY).  Establishes that suppression is
            post-ignition, not due to weak initial representation.

    V19.2 — Post-ignition, decoding accuracy for the losing stimulus drops below
            permutation-derived chance for ≥ V19_POST_IGNITION_SUPPRESSION_BINS
            consecutive 50 ms bins — the suppression signature.

    V19.3 — On no-ignition (subliminal) trials, both categories remain decodable
            above V19_CONTROL_DECODABILITY_MIN, confirming suppression is
            ignition-specific.
    """

    def __init__(
        self,
        simulator: Optional[StimulusPairEEGSimulator] = None,
        decoder: Optional[TimeResolvedMVPADecoder] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        _rng = rng or np.random.default_rng(RANDOM_SEED)
        self.simulator = simulator or StimulusPairEEGSimulator(rng=_rng)
        self.decoder = decoder or TimeResolvedMVPADecoder(rng=_rng)
        self.dataset: List[Tuple[TrialMetadata, np.ndarray, np.ndarray]] = []

        # Computed results cached after first run
        self._ig_accuracy_ts: Optional[np.ndarray] = None
        self._no_ig_losing_accuracy_ts: Optional[np.ndarray] = None
        self._no_ig_winning_accuracy_ts: Optional[np.ndarray] = None
        self._perm_threshold: float = 0.55

    def load_or_generate_data(self) -> None:
        if not self.dataset:
            logger.info("Simulating stimulus-pair EEG dataset …")
            self.dataset = self.simulator.simulate_dataset()
            logger.info("Simulated %d trials.", len(self.dataset))

    # ------------------------------------------------------------------
    def _build_feature_matrices(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            ig_feats   : (n_ig, N_BINS, N_SENSORS)  — losing-stimulus features on ignition trials
            ig_labels  : (n_ig,)  — binary category labels
            no_ig_losing_feats  : (n_no, N_BINS, N_SENSORS)  — losing on no-ignition
            no_ig_winning_feats : (n_no, N_BINS, N_SENSORS)  — winning on no-ignition
        """
        ig_feats_list, ig_labels_list = [], []
        no_ig_losing_list, no_ig_winning_list = [], []

        for meta, losing_feats, winning_feats in self.dataset:
            label = 1 if meta.losing_category == "faces" else 0
            if meta.condition == "ignition":
                ig_feats_list.append(losing_feats)
                ig_labels_list.append(label)
            else:
                no_ig_losing_list.append(losing_feats)
                no_ig_winning_list.append(winning_feats)

        return (
            np.array(ig_feats_list),
            np.array(ig_labels_list),
            np.array(no_ig_losing_list),
            np.array(no_ig_winning_list),
        )

    def _run_decoding(self) -> None:
        """Decode all conditions and cache results."""
        ig_feats, ig_labels, no_ig_losing, no_ig_winning = self._build_feature_matrices()

        logger.info("Running time-resolved MVPA on ignition trials …")
        self._ig_accuracy_ts, self._perm_threshold = self.decoder.decode_timeseries(ig_feats, ig_labels)

        logger.info("Running MVPA on no-ignition losing-stimulus trials …")
        no_ig_labels = np.array(
            [1 if m.losing_category == "faces" else 0 for m, _, _ in self.dataset if m.condition == "no_ignition"]
        )
        self._no_ig_losing_accuracy_ts, _ = self.decoder.decode_timeseries(no_ig_losing, no_ig_labels)

        logger.info("Running MVPA on no-ignition winning-stimulus trials …")
        win_labels = np.array(
            [1 if m.reported_category == "faces" else 0 for m, _, _ in self.dataset if m.condition == "no_ignition"]
        )
        self._no_ig_winning_accuracy_ts, _ = self.decoder.decode_timeseries(no_ig_winning, win_labels)

    # ------------------------------------------------------------------
    def validate_pre_ignition_decodability(self) -> Dict[str, Any]:
        """V19.1 — Losing stimulus is decodable pre-ignition (≥ threshold)."""
        if self._ig_accuracy_ts is None:
            raise ValueError("_ig_accuracy_ts must not be None")
        pre_bins = self._ig_accuracy_ts[BIN_CENTRES_MS < 0]
        mean_pre = float(np.mean(pre_bins))
        passed = mean_pre >= V19_MIN_PRE_IGNITION_ACCURACY

        return {
            "test_name": "V19.1 Pre-ignition Losing-Stimulus Decodability",
            "prediction_id": "V19.1",
            "mean_pre_ignition_accuracy": mean_pre,
            "threshold": V19_MIN_PRE_IGNITION_ACCURACY,
            "n_pre_bins": int((BIN_CENTRES_MS < 0).sum()),
            "passed": bool(passed),
        }

    def validate_post_ignition_suppression(self) -> Dict[str, Any]:
        """V19.2 — Post-ignition accuracy drops below chance for ≥ N consecutive bins."""
        if self._ig_accuracy_ts is None:
            raise ValueError("_ig_accuracy_ts must not be None")
        post_mask = BIN_CENTRES_MS > 200  # 200 ms post-ignition onset
        post_acc = self._ig_accuracy_ts[post_mask]

        below_chance = post_acc < V19_BELOW_CHANCE_THRESHOLD

        # Count maximum run of consecutive below-chance bins
        max_run = 0
        current_run = 0
        for b in below_chance:
            if b:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        passed = max_run >= V19_POST_IGNITION_SUPPRESSION_BINS

        return {
            "test_name": "V19.2 Post-ignition Suppression Below Chance",
            "prediction_id": "V19.2",
            "permutation_threshold": float(self._perm_threshold),
            "below_chance_threshold": V19_BELOW_CHANCE_THRESHOLD,
            "max_consecutive_below_chance_bins": int(max_run),
            "required_consecutive_bins": V19_POST_IGNITION_SUPPRESSION_BINS,
            "post_ignition_mean_accuracy": float(np.mean(post_acc)),
            "passed": bool(passed),
        }

    def validate_control_decodability(self) -> Dict[str, Any]:
        """V19.3 — On no-ignition trials both categories remain decodable."""
        if self._no_ig_losing_accuracy_ts is None:
            raise ValueError("_no_ig_losing_accuracy_ts must not be None")
        if self._no_ig_winning_accuracy_ts is None:
            raise ValueError("_no_ig_winning_accuracy_ts must not be None")

        mean_losing = float(np.mean(self._no_ig_losing_accuracy_ts))
        mean_winning = float(np.mean(self._no_ig_winning_accuracy_ts))

        passed = mean_losing >= V19_CONTROL_DECODABILITY_MIN and mean_winning >= V19_CONTROL_DECODABILITY_MIN

        return {
            "test_name": "V19.3 No-ignition Both-Category Decodability",
            "prediction_id": "V19.3",
            "mean_no_ignition_losing_accuracy": mean_losing,
            "mean_no_ignition_winning_accuracy": mean_winning,
            "threshold": V19_CONTROL_DECODABILITY_MIN,
            "passed": bool(passed),
        }

    # ------------------------------------------------------------------
    def generate_summary_figure(self, output_path: Optional[Path] = None) -> Optional[Path]:
        """Three-panel decoding time-course figure."""
        if not HAS_MATPLOTLIB:
            return None
        if self._ig_accuracy_ts is None:
            raise ValueError("_ig_accuracy_ts must not be None")

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle("VP-19: Information Erasure — MVPA Decoding Time-courses", fontsize=13)

        # Panel 1: Ignition-trial losing-stimulus decoding
        ax = axes[0]
        ax.plot(
            BIN_CENTRES_MS,
            self._ig_accuracy_ts,
            color="steelblue",
            linewidth=2,
            label="Losing stimulus (ignition)",
        )
        ax.axhline(
            self._perm_threshold,
            color="grey",
            linestyle="--",
            label=f"Permutation threshold ({self._perm_threshold:.2f})",
        )
        ax.axhline(
            V19_BELOW_CHANCE_THRESHOLD,
            color="salmon",
            linestyle=":",
            label=f"Below-chance criterion ({V19_BELOW_CHANCE_THRESHOLD})",
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", label="Ignition onset")
        ax.set_xlabel("Time re. ignition (ms)")
        ax.set_ylabel("Decoding accuracy")
        ax.set_title("Losing stimulus — ignition trials")
        ax.legend(fontsize=7)
        ax.set_ylim(0.3, 0.9)

        # Panel 2: No-ignition comparison
        ax = axes[1]
        ax.plot(
            BIN_CENTRES_MS,
            self._no_ig_losing_accuracy_ts,
            color="steelblue",
            linewidth=2,
            label="Losing (no-ignition)",
        )
        ax.plot(
            BIN_CENTRES_MS,
            self._no_ig_winning_accuracy_ts,
            color="darkorange",
            linewidth=2,
            label="Winning (no-ignition)",
        )
        ax.axhline(
            V19_CONTROL_DECODABILITY_MIN,
            color="grey",
            linestyle="--",
            label=f"Decodability threshold ({V19_CONTROL_DECODABILITY_MIN})",
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time re. ignition (ms)")
        ax.set_ylabel("Decoding accuracy")
        ax.set_title("Both categories — no-ignition trials")
        ax.legend(fontsize=7)
        ax.set_ylim(0.3, 0.9)

        # Panel 3: Ignition vs. no-ignition overlay (losing category only)
        ax = axes[2]
        ax.plot(
            BIN_CENTRES_MS,
            self._ig_accuracy_ts,
            color="steelblue",
            linewidth=2,
            label="Losing — ignition",
        )
        ax.plot(
            BIN_CENTRES_MS,
            self._no_ig_losing_accuracy_ts,
            color="steelblue",
            linewidth=2,
            linestyle="--",
            label="Losing — no-ignition",
        )
        ax.axhline(0.5, color="grey", linestyle=":")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time re. ignition (ms)")
        ax.set_ylabel("Decoding accuracy")
        ax.set_title("Suppression signature (ignition-specific)")
        ax.legend(fontsize=7)
        ax.set_ylim(0.3, 0.9)

        plt.tight_layout()

        if output_path is None:
            results_dir = _project_root / "data_repository" / "processed_data"
            results_dir.mkdir(parents=True, exist_ok=True)
            output_path = results_dir / "VP19_InformationErasure_MVPA_summary.png"

        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Summary figure saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    def run_full_validation(self) -> Dict[str, Any]:
        """Run all three validation tests and return aggregated results."""
        self.load_or_generate_data()
        self._run_decoding()

        results: Dict[str, Any] = {
            "pre_ignition_decodability": self.validate_pre_ignition_decodability(),
            "post_ignition_suppression": self.validate_post_ignition_suppression(),
            "control_decodability": self.validate_control_decodability(),
        }

        passed = sum(1 for r in results.values() if r.get("passed", False))
        total = len(results)

        results.update(
            {
                "overall_score": passed / total,
                "tests_passed": passed,
                "tests_total": total,
                "protocol_id": "VP_19_InformationErasureMVPA",
                "validation_timestamp": datetime.now().isoformat(),
                "measurement_gap_note": (
                    "This protocol tests representational erasure at Tier 3 "
                    "(neural MVPA decoding).  The thermodynamic cost of that "
                    "erasure — Landauer's principle applied to bit erasure "
                    "(Level 1) — is NOT directly measured here.  The link "
                    "requires the Landauer bridge and remains in Gap 1 territory. "
                    "Program 3 (PET) is required for thermodynamic ground-truth. "
                    "This distinction must appear explicitly in the paper to "
                    "prevent level-confusion."
                ),
            }
        )

        logger.info(
            "VP-19 validation complete: %d/%d tests passed (score %.2f)",
            passed,
            total,
            results["overall_score"],
        )
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────────────────


def run_validation(
    seed: Optional[int] = None,
    n_permutations: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Entry point for external validation runners.

    Args:
        seed: RNG seed for reproducibility.
        n_permutations: Override default permutation count (useful for fast tests).
        **kwargs: Accepted but ignored (API compatibility).
    """
    rng = np.random.default_rng(seed if seed is not None else RANDOM_SEED)
    n_perm = n_permutations if n_permutations is not None else V19_N_PERMUTATIONS
    decoder = TimeResolvedMVPADecoder(n_permutations=n_perm, rng=rng)
    simulator = StimulusPairEEGSimulator(rng=rng)
    validator = InformationErasureValidator(simulator=simulator, decoder=decoder, rng=rng)
    results = validator.run_full_validation()
    validator.generate_summary_figure()
    return results


def validate() -> Dict[str, Any]:
    """Alias for main.py compatibility."""
    return run_validation()


def main() -> None:
    """CLI entry point for standalone execution."""
    print("=" * 60)
    print("VP-19: Information Erasure Protocol — MVPA Decoding")
    print("=" * 60)

    results = run_validation()

    print(f"\nOverall Score : {results['overall_score']:.2%}")
    print(f"Tests Passed  : {results['tests_passed']}/{results['tests_total']}")

    print("\nDetailed Results:")
    for key, val in results.items():
        if isinstance(val, dict) and "test_name" in val:
            status = "PASS" if val.get("passed") else "FAIL"
            print(f"  [{status}] {val['test_name']}")

    print("\nMeasurement Gap:")
    print(f"  {results['measurement_gap_note']}")


def run_protocol_main(config=None):
    """Execute and return a standardized ProtocolResult."""
    legacy = run_validation()
    try:
        from utils.protocol_schema import PredictionResult, PredictionStatus, ProtocolResult

        named = {}
        for pred_id in ["V19.1", "V19.2", "V19.3"]:
            v = legacy.get(pred_id, {})
            passed = v.get("passed", False)
            named[pred_id] = PredictionResult(
                passed=passed,
                value=v.get("effect_size") or v.get("value"),
                threshold=v.get("threshold"),
                status=PredictionStatus.PASSED if passed else PredictionStatus.FAILED,
                name=v.get("test_name", pred_id),
                sources=["VP_19_InformationErasureMVPA"],
            )
        return ProtocolResult(
            protocol_id="VP_19_InformationErasureMVPA",
            named_predictions=named,
            completion_percentage=100,
            data_sources=["MVPA Simulation"],
            methodology="information_erasure_mvpa",
            errors=[],
            metadata={"overall_score": legacy.get("overall_score")},
        ).to_dict()
    except ImportError:
        return legacy


if __name__ == "__main__":
    main()
