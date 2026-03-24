"""
Comprehensive unit tests for APGI_Equations.py covering missing classes and methods.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from importlib import import_module

APGI_Equations = import_module("APGI_Equations")
APGIParameters = APGI_Equations.APGIParameters
PsychologicalState = APGI_Equations.PsychologicalState
StateCategory = APGI_Equations.StateCategory
APGIStateLibrary = APGI_Equations.APGIStateLibrary
MeasurementEquations = APGI_Equations.MeasurementEquations
NeuromodulatorSystem = APGI_Equations.NeuromodulatorSystem
EnhancedSurpriseIgnitionSystem = APGI_Equations.EnhancedSurpriseIgnitionSystem


class TestAPGIParametersExtended:
    def test_parameter_validation_ranges(self):
        """Test validation of all parameter groups in APGIParameters."""
        # Valid parameters
        params = APGIParameters()
        assert len(params.validate()) == 0

        # Invalid tau_S (0.2-0.5)
        params.tau_S = 0.1
        params.tau_theta = 4.0  # (5-60)
        violations = params.validate()
        assert any("τ_S" in v for v in violations)
        assert any("tau_theta" in v for v in violations)

        # Invalid alpha (3.0-8.0)
        params = APGIParameters(alpha=2.0)
        assert any("α" in v for v in params.validate())

        # Invalid beta (0.5-2.5)
        params = APGIParameters(beta=0.4)
        assert any("β_som" in v for v in params.validate())

        # Invalid rho (0.3-0.9)
        params = APGIParameters(rho=1.0)
        assert any("rho" in v for v in params.validate())

        # Invalid domain thresholds
        params = APGIParameters(theta_survival=0.0, theta_neutral=2.0)
        violations = params.validate()
        assert any("theta_survival" in v for v in violations)
        assert any("theta_neutral" in v for v in violations)

    def test_get_domain_threshold(self):
        params = APGIParameters(theta_survival=0.35, theta_neutral=0.75, theta_0=0.5)
        assert params.get_domain_threshold("survival") == 0.35
        assert params.get_domain_threshold("neutral") == 0.75
        assert params.get_domain_threshold("unknown") == 0.5

    def test_apply_neuromodulator_effects(self):
        params = APGIParameters(ACh=2.0, NE=1.5, DA=1.2, HT5=0.8)
        effects = params.apply_neuromodulator_effects()
        assert effects["Pi_e_mod"] == 2.0 * 0.3
        assert effects["theta_mod"] == 1.5 * 0.2
        assert effects["Pi_i_mod"] == 0.8 * 0.25

    def test_compute_precision_expectation_gap(self):
        params = APGIParameters(ACh=1.0, NE=1.0)
        # expected = 1.0 * 0.5 + 1.0 * 0.3 = 0.8
        gap = params.compute_precision_expectation_gap(Pi_e_actual=0.5, Pi_i_actual=0.5)
        # actual = (0.5 + 0.5) / 2 = 0.5
        # gap = 0.8 - 0.5 = 0.3
        assert abs(gap - 0.3) < 1e-10


class TestPsychologicalStateExtended:
    def test_state_post_init_clipping(self):
        # Test beta clipping
        state = PsychologicalState(
            name="test",
            category=StateCategory.OPTIMAL_FUNCTIONING,
            description="",
            phenomenology=[],
            Pi_e_actual=1.0,
            Pi_i_baseline_actual=1.0,
            M_ca=0.0,
            beta=3.0,
            z_e=0.0,
            z_i=0.0,
            theta_t=1.0,
        )
        assert state.beta == 2.5  # Clipped from 3.0

        state2 = PsychologicalState(
            name="test2",
            category=StateCategory.OPTIMAL_FUNCTIONING,
            description="",
            phenomenology=[],
            Pi_e_actual=1.0,
            Pi_i_baseline_actual=1.0,
            M_ca=0.0,
            beta=0.1,
            z_e=0.0,
            z_i=0.0,
            theta_t=1.0,
        )
        assert state2.beta == 0.5  # Clipped from 0.1

    def test_compute_ignition_probability(self):
        state = PsychologicalState(
            name="test",
            category=StateCategory.OPTIMAL_FUNCTIONING,
            description="",
            phenomenology=[],
            Pi_e_actual=1.0,
            Pi_i_baseline_actual=1.0,
            M_ca=0.0,
            beta=1.0,
            z_e=1.0,
            z_i=1.0,
            theta_t=1.0,
            content_domain="neutral",
        )
        # S_t = Pi_e * |z_e| + Pi_i_eff * |z_i|
        # Pi_i_eff = 1.0 * exp(1.0 * 0) = 1.0
        # S_t = 1.0 * 1.0 + 1.0 * 1.0 = 2.0
        # Prob = 1 / (1 + exp(-5.5 * (2.0 - 1.0))) = 1 / (1 + exp(-5.5 * 1.0))
        prob = state.compute_ignition_probability()
        assert prob > 0.99

        # Domain aware survival
        state.content_domain = "survival"
        # effective_theta = 1.0 * 0.7 = 0.7
        # Prob = 1 / (1 + exp(-5.5 * (2.5 - 0.7)))
        prob_surv = state.compute_ignition_probability()
        assert prob_surv > prob  # Lower threshold should mean higher probability

    def test_to_dynamical_inputs(self):
        state = PsychologicalState(
            name="test",
            category=StateCategory.OPTIMAL_FUNCTIONING,
            description="",
            phenomenology=[],
            Pi_e_actual=2.0,
            Pi_i_baseline_actual=1.0,
            M_ca=0.5,
            beta=1.0,
            z_e=0.3,
            z_i=0.2,
            theta_t=1.0,
            arousal_level=0.6,
            content_domain="survival",
            Pi_e_expected=3.0,
            Pi_i_expected=2.0,
        )
        inputs = state.to_dynamical_inputs(time=0.0, include_expectation=False)
        assert inputs["Pi_e"] == 2.0
        assert inputs["eps_e"] == 0.3
        assert inputs["beta"] == 1.0

        inputs_exp = state.to_dynamical_inputs(time=0.0, include_expectation=True)
        assert inputs_exp["Pi_e"] == 3.0
        # Pi_i_eff_expected = Pi_i_expected * exp(beta * M_ca)
        # exp(0.5) = 1.6487
        # Pi_i_eff_expected = 2.0 * 1.6487 = 3.2974
        assert abs(inputs_exp["Pi_i"] - 3.2974) < 0.01


class TestAPGIStateLibraryExtended:
    def test_library_initialization(self):
        lib = APGIStateLibrary()
        assert len(lib.states) == 52
        assert "flow" in lib.states
        assert "panic" in lib.states

        state = lib.get_state("flow")
        assert state.name == "flow"

        with pytest.raises(ValueError):
            lib.get_state("nonexistent")

    def test_apply_psychiatric_profile(self):
        lib = APGIStateLibrary()
        # Apply GAD to "neutral"
        gad_neutral = lib.apply_psychiatric_profile("neutral", "GAD")
        assert gad_neutral.name == "neutral_GAD"
        # Since theta_survival is NOT in PsychologicalState, it's NOT set
        assert gad_neutral.beta == 1.8

        # Apply MDD to "joy"
        mdd_joy = lib.apply_psychiatric_profile("joy", "MDD")
        assert mdd_joy.arousal_level == 0.3
        assert mdd_joy.M_ca == -0.5

        with pytest.raises(ValueError):
            lib.apply_psychiatric_profile("nonexistent", "GAD")
        with pytest.raises(ValueError):
            lib.apply_psychiatric_profile("neutral", "UNKNOWN")


class TestMeasurementEquationsExtended:
    def test_compute_all_measurements(self):
        lib = APGIStateLibrary()
        state = lib.get_state("flow")
        measurements = MeasurementEquations.compute_all_measurements(state)

        assert "HEP_amplitude" in measurements
        assert "P3b_latency" in measurements
        assert "detection_threshold" in measurements
        assert "ignition_probability" in measurements
        assert "ignition_duration" in measurements
        assert "anxiety_index" in measurements
        assert "precision_expectation_gap" in measurements

    def test_compute_P3b_latency_no_ignition(self):
        # Low surprise, high threshold
        latency = MeasurementEquations.compute_P3b_latency(
            S_t=0.5, theta_t=2.0, Pi_e=1.0
        )
        assert latency >= 500  # Should be high because S_t < theta_t


class TestNeuromodulatorSystemExtended:
    def test_parameter_mapping(self):
        # ACh effects
        ach_map = NeuromodulatorSystem.PARAMETER_MAPPINGS["ACh"]
        assert ach_map["Pi_e"] == 0.3

        # NE effects
        ne_map = NeuromodulatorSystem.PARAMETER_MAPPINGS["NE"]
        assert ne_map["theta_t"] == 0.4


class TestEnhancedAPGIIntegration:
    """Integration tests for the enhanced system components."""

    def test_neuromodulator_system_basic(self):
        """Test NeuromodulatorSystem basic operation."""
        nms = NeuromodulatorSystem()
        assert nms.levels["NE"] == 1.0

        # Test update
        nms.update_dynamically(S_t=10.0, B_t=0, time=10.0)
        assert nms.levels["NE"] >= 1.0  # Should not decay below 1.0 if S_t is high

        # Test DA increase
        nms.update_dynamically(S_t=5.0, B_t=1, time=20.0)
        assert nms.levels["DA"] > 1.0

    def test_measurement_equations_d_prime(self):
        """Test d' computation in MeasurementEquations."""
        me = MeasurementEquations()
        d_prime = me.compute_detection_threshold(theta_t=0.5, content_domain="survival")
        assert d_prime > 0

        d_prime_neutral = me.compute_detection_threshold(
            theta_t=0.5, content_domain="neutral"
        )
        assert d_prime > d_prime_neutral  # Survival multiplier should be higher

    def test_enhanced_system_reset_and_step(self):
        """Test EnhancedSurpriseIgnitionSystem reset and step."""
        system = EnhancedSurpriseIgnitionSystem()
        system.reset()
        assert system.S == 0.0
        assert system.time == 0.0

        # Test step
        inputs = {"u_e": 1.0, "up_e": 0.5, "u_i": 1.0, "up_i": 0.9, "g_stim": 0.1}
        state = system.step(inputs, dt=0.01)

        assert "S" in state
        assert "B" in state
        assert "detection_threshold" in state
        assert system.time == 0.01

    def test_enhanced_system_neuromod_integration(self):
        """Test system with neuromodulator integration."""
        nms = NeuromodulatorSystem()
        nms.set_levels(NE=2.0)  # High excitement
        system = EnhancedSurpriseIgnitionSystem(neuromodulator_system=nms)

        inputs = {"u_e": 0.1, "up_e": 0.1, "u_i": 0.1, "up_i": 0.1, "g_stim": 0.0}
        system.step(inputs, dt=0.01)

        # NE should affect precision via apply_neuromodulator_effects
        # We check if it ran without error
        assert system.neuromodulator_system.levels["NE"] >= 1.0
