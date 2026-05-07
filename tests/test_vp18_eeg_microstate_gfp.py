"""
Tests for VP_18_EEG_Microstate_GFP_P3b.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from Validation.VP_18_EEG_Microstate_GFP_P3b import (
    GFP_WINDOW_MS,
    GFPMicrostateValidator,
    GFPResult,
    OddballEEGSimulator,
    run_validation,
    validate,
)

# ---------------------------------------------------------------------------
# OddballEEGSimulator tests
# ---------------------------------------------------------------------------


class TestOddballEEGSimulator:

    def test_default_init(self):
        sim = OddballEEGSimulator(n_subjects=2, n_trials_per_condition=4)
        assert sim.fs == 1000.0
        assert sim.n_electrodes == 64
        assert len(sim.timepoints_ms) == sim.n_timepoints

    def test_gfp_window_indices_nonempty(self):
        sim = OddballEEGSimulator()
        assert sim.window_idx.sum() > 0

    def test_gfp_window_covers_correct_range(self):
        sim = OddballEEGSimulator()
        t_in_window = sim.timepoints_ms[sim.window_idx]
        assert t_in_window.min() >= GFP_WINDOW_MS[0]
        assert t_in_window.max() <= GFP_WINDOW_MS[1]

    def test_simulate_ignition_epoch(self):
        sim = OddballEEGSimulator(n_subjects=1, n_trials_per_condition=1)
        result = sim._simulate_epoch(0, 0, "ignition", 0.9)
        assert isinstance(result, GFPResult)
        assert result.epoch_meta.condition == "ignition"
        assert result.gfp_auc > 0
        assert len(result.gfp_timeseries) == sim.n_timepoints

    def test_simulate_no_ignition_epoch(self):
        sim = OddballEEGSimulator(n_subjects=1, n_trials_per_condition=1)
        result = sim._simulate_epoch(0, 0, "no_ignition", 0.1)
        assert isinstance(result, GFPResult)
        assert result.epoch_meta.condition == "no_ignition"
        assert result.gfp_auc > 0

    def test_ignition_gfp_auc_greater_than_no_ignition_on_average(self):
        """Ignition trials should produce systematically larger GFP-AUC."""
        rng = np.random.default_rng(0)
        sim = OddballEEGSimulator(n_subjects=10, n_trials_per_condition=20, rng=rng)
        dataset = sim.simulate_dataset()
        ig = [r.gfp_auc for r in dataset if r.epoch_meta.condition == "ignition"]
        no = [r.gfp_auc for r in dataset if r.epoch_meta.condition == "no_ignition"]
        assert np.mean(ig) > np.mean(no)

    def test_simulate_dataset_length(self):
        sim = OddballEEGSimulator(n_subjects=3, n_trials_per_condition=5)
        dataset = sim.simulate_dataset()
        # 3 subjects × 5 trials × 2 conditions
        assert len(dataset) == 3 * 5 * 2

    def test_spatial_cov_shape(self):
        sim = OddballEEGSimulator()
        cov = sim._build_spatial_cov()
        assert cov.shape == (64, 64)
        # Diagonal should be 1 (distance 0)
        assert np.allclose(np.diag(cov), 1.0)

    def test_p3b_template_zero_before_stimulus(self):
        sim = OddballEEGSimulator()
        template = sim._p3b_template(amplitude=1.0)
        pre_stim = template[sim.timepoints_ms < 0]
        assert np.all(pre_stim == 0.0)

    def test_p3b_template_peak_near_380ms(self):
        sim = OddballEEGSimulator()
        template = sim._p3b_template(amplitude=1.0)
        peak_t = sim.timepoints_ms[np.argmax(template)]
        assert 360.0 <= peak_t <= 400.0

    def test_gfp_timeseries_nonnegative(self):
        sim = OddballEEGSimulator(n_subjects=1, n_trials_per_condition=1)
        result = sim._simulate_epoch(0, 0, "ignition", 0.8)
        assert np.all(result.gfp_timeseries >= 0)


# ---------------------------------------------------------------------------
# GFPMicrostateValidator tests
# ---------------------------------------------------------------------------


class TestGFPMicrostateValidator:

    @pytest.fixture
    def small_validator(self):
        rng = np.random.default_rng(42)
        sim = OddballEEGSimulator(n_subjects=10, n_trials_per_condition=30, rng=rng)
        v = GFPMicrostateValidator(simulator=sim)
        v.load_or_generate_data()
        return v

    def test_load_or_generate_data_populates_dataset(self, small_validator):
        assert len(small_validator.dataset) > 0

    def test_load_or_generate_data_idempotent(self, small_validator):
        original_len = len(small_validator.dataset)
        small_validator.load_or_generate_data()
        assert len(small_validator.dataset) == original_len

    def test_split_by_condition_balanced(self, small_validator):
        ig, no, st, auc = small_validator._split_by_condition()
        assert len(ig) == len(no)
        assert len(st) == len(ig) + len(no)
        assert len(auc) == len(st)

    def test_validate_gfp_auc_ignition_effect_keys(self, small_validator):
        result = small_validator.validate_gfp_auc_ignition_effect()
        for key in (
            "test_name",
            "prediction_id",
            "t_statistic",
            "p_value",
            "cohens_d",
            "passed",
        ):
            assert key in result
        assert result["prediction_id"] == "V18.1"

    def test_validate_proportional_advantage_keys(self, small_validator):
        result = small_validator.validate_proportional_advantage()
        for key in (
            "test_name",
            "prediction_id",
            "proportional_advantage",
            "threshold",
            "passed",
        ):
            assert key in result
        assert result["prediction_id"] == "V18.2"

    def test_validate_st_correlation_keys(self, small_validator):
        result = small_validator.validate_st_correlation()
        for key in (
            "test_name",
            "prediction_id",
            "pearson_r",
            "p_value",
            "threshold_r",
            "passed",
        ):
            assert key in result
        assert result["prediction_id"] == "V18.3"

    def test_run_full_validation_structure(self, small_validator):
        results = small_validator.run_full_validation()
        assert "overall_score" in results
        assert "tests_passed" in results
        assert "tests_total" in results
        assert results["tests_total"] == 3
        assert results["protocol_id"] == "VP_18_EEG_Microstate_GFP_P3b"
        assert "measurement_gap_note" in results

    def test_run_full_validation_score_range(self, small_validator):
        results = small_validator.run_full_validation()
        assert 0.0 <= results["overall_score"] <= 1.0

    def test_run_full_validation_passes_all_tests(self):
        """With default parameters the simulation should pass all three tests."""
        rng = np.random.default_rng(42)
        sim = OddballEEGSimulator(n_subjects=24, n_trials_per_condition=60, rng=rng)
        v = GFPMicrostateValidator(simulator=sim)
        results = v.run_full_validation()
        assert results["tests_passed"] == 3

    def test_measurement_gap_note_content(self, small_validator):
        results = small_validator.run_full_validation()
        note = results["measurement_gap_note"]
        assert "Level 3" in note
        assert "Landauer" in note
        assert "thermodynamic" in note

    @patch("Validation.VP_18_EEG_Microstate_GFP_P3b.HAS_MATPLOTLIB", False)
    def test_generate_summary_figure_no_matplotlib(self, small_validator):
        result = small_validator.generate_summary_figure()
        assert result is None

    def test_generate_summary_figure_with_matplotlib(self, small_validator, tmp_path):
        output = tmp_path / "test_figure.png"
        result = small_validator.generate_summary_figure(output_path=output)
        if result is not None:
            assert output.exists()

    def test_proportional_advantage_zero_denominator(self):
        """Edge case: no-ignition AUC is zero."""
        rng = np.random.default_rng(0)
        sim = OddballEEGSimulator(n_subjects=2, n_trials_per_condition=5, rng=rng)
        v = GFPMicrostateValidator(simulator=sim)
        v.load_or_generate_data()
        # Monkey-patch all no-ignition AUCs to 0
        for r in v.dataset:
            if r.epoch_meta.condition == "no_ignition":
                object.__setattr__(r, "gfp_auc", 0.0)
        result = v.validate_proportional_advantage()
        assert result["proportional_advantage"] == 0.0

    def test_cohens_d_calculation_positive(self, small_validator):
        result = small_validator.validate_gfp_auc_ignition_effect()
        # Ignition AUC mean should exceed no-ignition AUC mean → d > 0
        assert result["cohens_d"] > 0

    def test_st_correlation_sign_positive(self, small_validator):
        result = small_validator.validate_st_correlation()
        # Higher Sₜ should yield higher GFP-AUC → positive correlation
        assert result["pearson_r"] > 0


# ---------------------------------------------------------------------------
# Public entry-point tests
# ---------------------------------------------------------------------------


class TestRunValidation:

    def test_run_validation_returns_dict(self):
        result = run_validation(seed=99)
        assert isinstance(result, dict)

    def test_run_validation_has_protocol_id(self):
        result = run_validation(seed=0)
        assert result["protocol_id"] == "VP_18_EEG_Microstate_GFP_P3b"

    def test_run_validation_accepts_kwargs(self):
        result = run_validation(seed=7, extra_param="ignored")
        assert "overall_score" in result

    def test_validate_alias(self):
        result = validate()
        assert isinstance(result, dict)
        assert "overall_score" in result

    def test_run_validation_deterministic(self):
        r1 = run_validation(seed=42)
        r2 = run_validation(seed=42)
        assert r1["overall_score"] == r2["overall_score"]
        assert r1["tests_passed"] == r2["tests_passed"]
