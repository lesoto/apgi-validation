"""
Targeted tests for APGIParameters and PsychologicalState dataclasses
===================================================================

Focuses on dataclass validation, methods, and properties that are currently uncovered.
"""

import pytest

from apgi_core.equations import (
    APGIParameters,
    PsychologicalState,
    StateCategory,
)


def test_apgi_parameters_defaults():
    """Test APGIParameters with default values."""
    params = APGIParameters()
    assert params.tau_S == 0.35
    assert params.theta_0 == 0.5
    assert params.alpha == 5.5
    assert params.beta == 1.5


def test_apgi_parameters_property_aliases():
    """Test property aliases for compatibility."""
    params = APGIParameters(tau_S=0.4, theta_0=0.6)
    assert params.theta_t0 == 0.6
    assert params.tau == 0.4


def test_apgi_parameters_validate_valid():
    """Test validation with valid parameters."""
    params = APGIParameters(
        tau_S=0.35,
        tau_theta=30.0,
        theta_0=0.5,
        alpha=5.5,
        beta=1.5,
        rho=0.7,
        theta_survival=0.3,
        theta_neutral=0.7,
    )
    violations = params.validate()
    assert len(violations) == 0


def test_apgi_parameters_validate_invalid_tau_S():
    """Test validation with invalid tau_S."""
    params = APGIParameters(tau_S=1.0)  # Outside [0.2, 0.5]
    violations = params.validate()
    assert len(violations) > 0
    # Check for either tau_S or τ_S in the message
    assert any("tau" in v.lower() or "τ" in v for v in violations)


def test_apgi_parameters_validate_invalid_theta_0():
    """Test validation with invalid theta_0."""
    params = APGIParameters(theta_0=2.0)  # Outside [0.1, 1.0]
    violations = params.validate()
    assert len(violations) > 0
    assert any("theta_0" in v for v in violations)


def test_apgi_parameters_validate_invalid_alpha():
    """Test validation with invalid alpha."""
    params = APGIParameters(alpha=10.0)  # Outside [3.0, 8.0]
    violations = params.validate()
    assert len(violations) > 0
    assert any("α" in v or "alpha" in v for v in violations)


def test_apgi_parameters_validate_invalid_beta():
    """Test validation with invalid beta."""
    params = APGIParameters(beta=3.0)  # Outside [0.5, 2.5]
    violations = params.validate()
    assert len(violations) > 0
    assert any("β" in v or "beta" in v for v in violations)


def test_apgi_parameters_validate_invalid_rho():
    """Test validation with invalid rho."""
    params = APGIParameters(rho=1.0)  # Outside [0.3, 0.9]
    violations = params.validate()
    assert len(violations) > 0
    assert any("rho" in v for v in violations)


def test_apgi_parameters_validate_invalid_domain_thresholds():
    """Test validation with invalid domain thresholds."""
    params = APGIParameters(theta_survival=1.0, theta_neutral=0.3)
    violations = params.validate()
    assert len(violations) > 0


def test_apgi_parameters_get_domain_threshold_survival():
    """Test getting domain-specific threshold for survival."""
    params = APGIParameters(theta_survival=0.3, theta_neutral=0.7)
    threshold = params.get_domain_threshold("survival")
    assert threshold == 0.3


def test_apgi_parameters_get_domain_threshold_neutral():
    """Test getting domain-specific threshold for neutral."""
    params = APGIParameters(theta_survival=0.3, theta_neutral=0.7)
    threshold = params.get_domain_threshold("neutral")
    assert threshold == 0.7


def test_apgi_parameters_get_domain_threshold_default():
    """Test getting domain-specific threshold for unknown domain."""
    params = APGIParameters(theta_survival=0.3, theta_neutral=0.7, theta_0=0.5)
    threshold = params.get_domain_threshold("unknown")
    assert threshold == 0.5


def test_apgi_parameters_apply_neuromodulator_effects():
    """Test applying neuromodulator effects."""
    params = APGIParameters(ACh=1.0, NE=1.0, DA=1.0, HT5=1.0)
    effects = params.apply_neuromodulator_effects()
    assert "Pi_e_mod" in effects
    assert "theta_mod" in effects
    assert "beta_mod" in effects
    assert "Pi_i_mod" in effects
    # Check that values are reasonable
    assert effects["Pi_e_mod"] > 0
    assert effects["theta_mod"] > 0


def test_apgi_parameters_compute_precision_expectation_gap():
    """Test computing precision expectation gap."""
    params = APGIParameters(ACh=1.0, NE=1.0)
    gap = params.compute_precision_expectation_gap(Pi_e_actual=1.0, Pi_i_actual=1.0)
    # Should return a float
    assert isinstance(gap, float)


def test_psychological_state_creation():
    """Test creating a PsychologicalState."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1", "feature2"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.0,
        beta=1.0,
        z_e=0.0,
        z_i=0.0,
        theta_t=0.5,
    )
    assert state.name == "test_state"
    assert state.category == StateCategory.OPTIMAL_FUNCTIONING
    assert state.Pi_e_actual == 1.0


def test_psychological_state_post_init_beta_clipping():
    """Test that beta is clipped to valid range in __post_init__."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        state = PsychologicalState(
            name="test_state",
            category=StateCategory.OPTIMAL_FUNCTIONING,
            description="Test state",
            phenomenology=["feature1"],
            Pi_e_actual=1.0,
            Pi_i_baseline_actual=1.0,
            M_ca=0.0,
            beta=3.0,  # Outside [0.5, 2.5]
            z_e=0.0,
            z_i=0.0,
            theta_t=0.5,
        )
        # Should be clipped to 2.5
        assert state.beta == 2.5
        # Should have issued a warning
        assert len(w) > 0


def test_psychological_state_post_init_expected_precision():
    """Test that expected precision defaults to actual if not provided."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.0,
        beta=1.0,
        z_e=0.0,
        z_i=0.0,
        theta_t=0.5,
        # Pi_e_expected and Pi_i_expected not provided
    )
    assert state.Pi_e_expected == 1.0
    assert state.Pi_i_expected == 1.0


def test_psychological_state_post_init_effective_precision():
    """Test computation of effective interoceptive precision."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.5,
        beta=1.0,
        z_e=0.0,
        z_i=0.0,
        theta_t=0.5,
    )
    # Pi_i_eff should be computed
    assert state.Pi_i_eff_actual is not None
    assert state.Pi_i_eff_expected is not None
    # Should be greater than baseline due to positive M_ca and beta
    assert state.Pi_i_eff_actual > state.Pi_i_baseline_actual


def test_psychological_state_post_init_accumulated_surprise():
    """Test computation of accumulated surprise."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.0,
        beta=1.0,
        z_e=0.5,
        z_i=0.5,
        theta_t=0.5,
    )
    # S_t should be computed
    assert state.S_t is not None
    assert state.S_t > 0


def test_psychological_state_post_init_precision_gap():
    """Test computation of precision expectation gap."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.0,
        beta=1.0,
        z_e=0.0,
        z_i=0.0,
        theta_t=0.5,
        Pi_e_expected=2.0,  # Higher than actual
        Pi_i_expected=2.0,
    )
    # Should have positive gap (anxiety-like)
    assert state.precision_expectation_gap > 0


def test_psychological_state_compute_ignition_probability():
    """Test ignition probability computation."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.0,
        beta=1.0,
        z_e=0.0,
        z_i=0.0,
        theta_t=0.5,
    )
    prob = state.compute_ignition_probability()
    assert 0.0 <= prob <= 1.0


def test_psychological_state_compute_ignition_probability_survival():
    """Test ignition probability with survival domain."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.0,
        beta=1.0,
        z_e=0.0,
        z_i=0.0,
        theta_t=0.5,
        content_domain="survival",
    )
    prob_survival = state.compute_ignition_probability(domain_aware=True)
    prob_neutral = state.compute_ignition_probability(domain_aware=False)
    # Survival should have higher probability (lower threshold)
    assert prob_survival >= prob_neutral


def test_psychological_state_get_anxiety_index():
    """Test anxiety index computation."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.0,
        beta=1.0,
        z_e=0.0,
        z_i=0.0,
        theta_t=0.5,
        Pi_e_expected=2.0,  # Higher than actual
        Pi_i_expected=2.0,
    )
    anxiety = state.get_anxiety_index()
    assert anxiety >= 0


def test_psychological_state_get_anxiety_index_no_gap():
    """Test anxiety index with no precision gap."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.0,
        beta=1.0,
        z_e=0.0,
        z_i=0.0,
        theta_t=0.5,
        Pi_e_expected=1.0,  # Same as actual
        Pi_i_expected=1.0,
    )
    anxiety = state.get_anxiety_index()
    # Should be 0 when no gap
    assert anxiety == 0


def test_psychological_state_to_dynamical_inputs_actual():
    """Test converting to dynamical inputs with actual precision."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.5,
        beta=1.0,
        z_e=0.0,
        z_i=0.0,
        theta_t=0.5,
    )
    inputs = state.to_dynamical_inputs(include_expectation=False)
    assert "Pi_e" in inputs
    assert "eps_e" in inputs
    assert "beta" in inputs
    assert "Pi_i" in inputs
    assert "eps_i" in inputs
    assert "M" in inputs
    assert "A" in inputs


def test_psychological_state_to_dynamical_inputs_expected():
    """Test converting to dynamical inputs with expected precision."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.5,
        beta=1.0,
        z_e=0.0,
        z_i=0.0,
        theta_t=0.5,
        Pi_e_expected=2.0,
        Pi_i_expected=2.0,
    )
    inputs = state.to_dynamical_inputs(include_expectation=True)
    # Should use expected precision
    assert inputs["Pi_e"] > 1.0
    assert inputs["Pi_i"] > 1.0


def test_psychological_state_to_dynamical_inputs_time_dependent():
    """Test time-dependent oscillations in dynamical inputs."""
    state = PsychologicalState(
        name="test_state",
        category=StateCategory.OPTIMAL_FUNCTIONING,
        description="Test state",
        phenomenology=["feature1"],
        Pi_e_actual=1.0,
        Pi_i_baseline_actual=1.0,
        M_ca=0.0,
        beta=1.0,
        z_e=0.0,
        z_i=0.0,
        theta_t=0.5,
    )
    inputs_t0 = state.to_dynamical_inputs(time=0.0)
    inputs_t1 = state.to_dynamical_inputs(time=1.0)
    # Should be different due to oscillations
    assert inputs_t0["Pi_e"] != inputs_t1["Pi_e"]


def test_state_category_enum():
    """Test StateCategory enum values."""
    assert StateCategory.OPTIMAL_FUNCTIONING.value[0] == "#2166AC"
    assert StateCategory.OPTIMAL_FUNCTIONING.value[1] == "Optimal Functioning"
    assert StateCategory.PATHOLOGICAL_EXTREME.value[1] == "Pathological/Extreme"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
