"""
Tests for uncovered Theory/ module paths
========================================

Tests for Theory modules that need additional coverage:
- APGI_Cross_Species_Scaling
- APGI_Open_Science_Framework
- APGI_Cultural_Neuroscience
- APGI_Turing_Machine
- APGI_Psychological_States
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCrossSpeciesScaling:
    """Test Cross-Species Scaling module paths"""

    def test_cross_species_scaling_init(self):
        """Test CrossSpeciesScaling initialization"""
        from Theory.APGI_Cross_Species_Scaling import CrossSpeciesScaling

        model = CrossSpeciesScaling()
        assert hasattr(model, "species_data")
        assert "Human" in model.species_data
        assert "Mouse" in model.species_data

    def test_generate_scaling_laws(self):
        """Test scaling laws generation"""
        from Theory.APGI_Cross_Species_Scaling import CrossSpeciesScaling

        model = CrossSpeciesScaling()
        laws = model.generate_scaling_laws()

        assert "pci_vs_brain_size" in laws
        assert "levels_vs_brain_size" in laws
        assert "timescale_vs_brain_size" in laws

    def test_compute_predicted_pci(self):
        """Test PCI prediction for various brain masses"""
        from Theory.APGI_Cross_Species_Scaling import CrossSpeciesScaling

        model = CrossSpeciesScaling()

        # Test with human brain mass
        human_pci = model.compute_predicted_pci(1500.0)
        assert 0.6 < human_pci < 0.9

        # Test with mouse brain mass (theoretical prediction ~0.054)
        mouse_pci = model.compute_predicted_pci(0.4)
        assert 0.05 < mouse_pci < 0.1

        # Test with very small brain
        small_pci = model.compute_predicted_pci(0.001)
        assert small_pci > 0

    def test_predict_species_consciousness(self):
        """Test species consciousness prediction"""
        from Theory.APGI_Cross_Species_Scaling import (
            SpeciesParameters, predict_species_consciousness)

        params = SpeciesParameters(
            name="TestSpecies",
            cortical_volume_mm3=1000.0,
            cortical_thickness_mm=2.5,
            neuron_density_per_mm3=80000.0,
            synaptic_density_per_mm3=1000000.0,
            conduction_velocity_m_s=2.0,
            body_mass_kg=70.0,
            brain_mass_g=1500.0,
        )

        result = predict_species_consciousness(params)

        assert "predicted_pci" in result
        assert "hierarchical_levels" in result
        assert "intrinsic_timescale" in result
        assert "encephalization_quotient" in result

    def test_predict_species_zero_body_mass(self):
        """Test species consciousness prediction with zero body mass"""
        from Theory.APGI_Cross_Species_Scaling import (
            SpeciesParameters, predict_species_consciousness)

        params = SpeciesParameters(
            name="TestSpecies",
            cortical_volume_mm3=100.0,
            cortical_thickness_mm=1.0,
            neuron_density_per_mm3=80000.0,
            synaptic_density_per_mm3=1000000.0,
            conduction_velocity_m_s=1.0,
            body_mass_kg=0.0,
            brain_mass_g=0.4,
        )

        result = predict_species_consciousness(params)
        assert result["encephalization_quotient"] == 1.0

    def test_generate_species_comparison_report(self):
        """Test species comparison report generation"""
        from Theory.APGI_Cross_Species_Scaling import \
            generate_species_comparison_report

        report = generate_species_comparison_report()

        assert "APGI CROSS-SPECIES SCALING VALIDATION REPORT" in report
        assert "Human" in report
        assert "SCALING LAWS:" in report

    def test_get_implementation_metadata(self):
        """Test implementation metadata retrieval"""
        from Theory.APGI_Cross_Species_Scaling import \
            get_implementation_metadata

        metadata = get_implementation_metadata()

        assert "protocol_id" in metadata
        assert "quality_rating" in metadata
        assert "innovation_alignment" in metadata

    def test_plot_scaling_relationships(self, tmp_path):
        """Test plotting functionality"""
        from Theory.APGI_Cross_Species_Scaling import CrossSpeciesScaling

        model = CrossSpeciesScaling()
        save_path = tmp_path / "test_scaling.png"

        with patch("matplotlib.pyplot.savefig") as mock_savefig:
            with patch("matplotlib.pyplot.close"):
                model.plot_scaling_relationships(str(save_path))
                mock_savefig.assert_called_once()


class TestOpenScienceFramework:
    """Test Open Science Framework module paths"""

    def test_preregistration_template_init(self):
        """Test PreregistrationTemplate initialization"""
        from Theory.APGI_Open_Science_Framework import PreregistrationTemplate

        template = PreregistrationTemplate(
            title="Test Study",
            authors=["Author 1", "Author 2"],
            date_created="2024-01-01",
            predicted_completion="2024-12-31",
            research_questions=["Q1", "Q2"],
            hypotheses=["H1", "H2"],
            theoretical_background="Background info",
            design_type="behavioral",
            paradigm="masking",
            sample_size=100,
            power_analysis={"alpha": 0.05, "power": 0.8},
            apgi_predictions={"tau_S": "increase"},
            falsification_criteria=["Criterion 1"],
            primary_analyses=["Analysis 1"],
            secondary_analyses=["Analysis 2"],
            exclusion_criteria=["Criterion 1"],
            data_repository="https://osf.io/test",
            code_repository="https://github.com/test",
            open_materials=True,
            open_data=True,
        )

        assert template.title == "Test Study"
        assert template.sample_size == 100

    def test_preregistration_to_json(self):
        """Test preregistration JSON export"""
        from Theory.APGI_Open_Science_Framework import PreregistrationTemplate

        template = PreregistrationTemplate(
            title="Test Study",
            authors=["Author 1"],
            date_created="2024-01-01",
            predicted_completion="2024-12-31",
            research_questions=["Q1"],
            hypotheses=["H1"],
            theoretical_background="Background",
            design_type="behavioral",
            paradigm="test",
            sample_size=50,
            power_analysis={},
            apgi_predictions={},
            falsification_criteria=[],
            primary_analyses=[],
            secondary_analyses=[],
            exclusion_criteria=[],
            data_repository="",
            code_repository="",
            open_materials=False,
            open_data=False,
        )

        json_str = template.to_json()
        data = json.loads(json_str)
        assert data["title"] == "Test Study"
        assert data["sample_size"] == 50

    def test_preregistration_from_json(self):
        """Test preregistration JSON import"""
        from Theory.APGI_Open_Science_Framework import PreregistrationTemplate

        json_str = json.dumps(
            {
                "title": "Test Study",
                "authors": ["Author 1"],
                "date_created": "2024-01-01",
                "predicted_completion": "2024-12-31",
                "research_questions": ["Q1"],
                "hypotheses": ["H1"],
                "theoretical_background": "Background",
                "design_type": "behavioral",
                "paradigm": "test",
                "sample_size": 50,
                "power_analysis": {},
                "apgi_predictions": {},
                "falsification_criteria": [],
                "primary_analyses": [],
                "secondary_analyses": [],
                "exclusion_criteria": [],
                "data_repository": "",
                "code_repository": "",
                "open_materials": False,
                "open_data": False,
            }
        )

        template = PreregistrationTemplate.from_json(json_str)
        assert template.title == "Test Study"

    def test_serialize_numpy_array(self):
        """Test numpy array serialization"""
        from Theory.APGI_Open_Science_Framework import serialize

        arr = np.array([1, 2, 3])
        result = serialize(arr)
        assert result == [1, 2, 3]

    def test_serialize_datetime(self):
        """Test datetime serialization"""
        from datetime import datetime

        from Theory.APGI_Open_Science_Framework import serialize

        dt = datetime.now()
        result = serialize(dt)
        assert isinstance(result, str)


class TestCulturalNeuroscience:
    """Test Cultural Neuroscience module paths"""

    def test_cultural_parameter_modulator_init(self):
        """Test CulturalParameterModulator initialization"""
        from Theory.APGI_Cultural_Neuroscience import \
            CulturalParameterModulator

        modulator = CulturalParameterModulator()
        assert hasattr(modulator, "modulate_cultural_dimensions")

    def test_cultural_baseline_shift(self):
        """Test cultural baseline shift computation"""
        from Theory.APGI_Cultural_Neuroscience import (
            ContemplativeParameters, CulturalContext,
            CulturalParameterModulator, LinguisticParameters)

        modulator = CulturalParameterModulator()
        # Create a Japanese-like cultural context
        language = LinguisticParameters(
            language_name="Japanese",
            culture="East Asian",
            embedding_depth=4.0,
            morphological_complexity=0.7,
            word_order_flexibility=0.3,
            semantic_density=0.75,
            polysemy_index=2.5,
            abstraction_level=0.6,
            working_memory_load=0.7,
            temporal_sequencing=0.8,
        )
        contemplative = ContemplativeParameters(
            practice_name="Zen",
            culture="Japanese",
            tradition="Buddhist",
            duration_years=5.0,
            session_duration_minutes=30.0,
            frequency_per_week=7.0,
            attention_stability=0.85,
            emotional_regulation=0.8,
            self_referential_processing=-0.6,
            decentering_ability=0.75,
            default_mode_reduction=0.3,
            salience_network_enhancement=0.25,
            frontal_parietal_integration=0.55,
        )
        context = CulturalContext(
            culture_name="Japan",
            primary_language=language,
            dominant_contemplative_practice=contemplative,
            individualism_score=20.0,  # Collectivist
            power_distance_score=50.0,
            uncertainty_avoidance=85.0,
            long_term_orientation=80.0,
            education_years=12.0,
            urbanization_rate=0.92,
            social_complexity=0.8,
        )
        params = modulator.compute_cultural_apgi_parameters(context)

        assert isinstance(params, dict)
        assert "theta" in params

    def test_all_cultural_regions(self):
        """Test that all cultural regions have defined parameters"""
        from Theory.APGI_Cultural_Neuroscience import (
            CulturalContext, CulturalParameterModulator, LinguisticParameters)

        modulator = CulturalParameterModulator()

        # Create minimal contexts for different region types
        contexts = [
            CulturalContext(
                culture_name="USA",
                primary_language=LinguisticParameters(
                    language_name="English",
                    culture="Western",
                    embedding_depth=3.0,
                    morphological_complexity=0.4,
                    word_order_flexibility=0.8,
                    semantic_density=0.6,
                    polysemy_index=3.0,
                    abstraction_level=0.7,
                    working_memory_load=0.5,
                    temporal_sequencing=0.6,
                ),
                dominant_contemplative_practice=None,
                individualism_score=80.0,  # Individualist
                power_distance_score=40.0,
                uncertainty_avoidance=45.0,
                long_term_orientation=25.0,
                education_years=13.0,
                urbanization_rate=0.83,
                social_complexity=0.75,
            ),
            CulturalContext(
                culture_name="Japan",
                primary_language=LinguisticParameters(
                    language_name="Japanese",
                    culture="East Asian",
                    embedding_depth=4.0,
                    morphological_complexity=0.7,
                    word_order_flexibility=0.3,
                    semantic_density=0.75,
                    polysemy_index=2.5,
                    abstraction_level=0.6,
                    working_memory_load=0.7,
                    temporal_sequencing=0.8,
                ),
                dominant_contemplative_practice=None,
                individualism_score=20.0,  # Collectivist
                power_distance_score=50.0,
                uncertainty_avoidance=85.0,
                long_term_orientation=80.0,
                education_years=12.0,
                urbanization_rate=0.92,
                social_complexity=0.8,
            ),
        ]

        for context in contexts:
            params = modulator.compute_cultural_apgi_parameters(context)
            assert isinstance(params, dict)
            assert "Pi_e" in params
            assert "theta" in params


class TestTuringMachine:
    """Test Turing Machine module paths"""

    def test_turing_machine_init(self):
        """Test APGITuringMachine initialization"""
        from Theory.APGI_Turing_Machine import APGITuringMachine

        tm = APGITuringMachine()
        assert hasattr(tm, "state")
        assert hasattr(tm, "params")
        assert hasattr(tm, "history")

    def test_turing_machine_step(self):
        """Test Turing machine step execution"""
        from Theory.APGI_Turing_Machine import APGITuringMachine

        tm = APGITuringMachine()

        # step() requires dt and inputs parameters
        result = tm.step(dt=0.01, inputs={"extero": 0.5, "intero": 0.3})
        assert isinstance(result, dict)

    def test_turing_machine_reset(self):
        """Test Turing machine reset"""
        from Theory.APGI_Turing_Machine import APGITuringMachine

        tm = APGITuringMachine()
        # Run a few steps first
        tm.step(dt=0.01, inputs={"extero": 0.5})
        tm.step(dt=0.01, inputs={"extero": 0.6})
        tm.reset()

        # After reset, state should be fresh
        assert tm.state.S_continuous == 0.0
        assert len(tm.history) == 0


class TestPsychologicalStates:
    """Test Psychological States module paths"""

    def test_psychological_state_init(self):
        """Test PsychologicalState initialization"""
        from Theory.APGI_Psychological_States import (APGIParameters,
                                                      PsychologicalState)

        params = APGIParameters(
            Pi_e=3.0,
            Pi_i_baseline=2.5,
            Pi_i_eff=2.5,
            theta_t=0.5,
            S_t=1.0,
            M_ca=0.0,
            beta=0.5,
            z_e=0.5,
            z_i=0.8,
        )
        state = PsychologicalState(
            name="TestState",
            parameters=params,
            category="TestCategory",
            description="Test description",
            phenomenology=["feature1", "feature2"],
            distinguishing_features={"key": "value"},
        )

        assert state.name == "TestState"
        assert state.category == "TestCategory"
        assert isinstance(state.parameters, APGIParameters)

    def test_psychological_state_attributes(self):
        """Test PsychologicalState attributes"""
        from Theory.APGI_Psychological_States import (APGIParameters,
                                                      PsychologicalState)

        params = APGIParameters(
            Pi_e=3.0,
            Pi_i_baseline=2.5,
            Pi_i_eff=2.5,
            theta_t=0.5,
            S_t=1.0,
            M_ca=0.0,
            beta=0.5,
            z_e=0.5,
            z_i=0.8,
        )
        state = PsychologicalState(
            name="TestState",
            parameters=params,
            category="TestCategory",
            description="Test description",
            phenomenology=["feature1"],
            distinguishing_features={"key": "value"},
            pathological_variant="None",
        )

        assert hasattr(state, "name")
        assert hasattr(state, "parameters")
        assert hasattr(state, "pathological_variant")
        assert state.pathological_variant == "None"

    def test_state_category_enum(self):
        """Test StateCategory enum exists"""
        from Theory.APGI_Psychological_States import StateCategory

        # Verify enum values exist
        assert hasattr(StateCategory, "OPTIMAL_FUNCTIONING")
        assert hasattr(StateCategory, "PATHOLOGICAL_EXTREME")


class TestTheoryModuleImports:
    """Test Theory module import paths"""

    def test_theory_package_import(self):
        """Test Theory package imports"""
        from Theory import (APGITuringMachine, CrossSpeciesScaling,
                            CulturalParameterModulator, OpenScienceValidator,
                            PsychState)

        assert CrossSpeciesScaling is not None
        assert OpenScienceValidator is not None
        assert CulturalParameterModulator is not None
        assert APGITuringMachine is not None
        assert PsychState is not None

    def test_theory_version(self):
        """Test Theory package version"""
        from Theory import __version__

        assert __version__ == "1.0.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
