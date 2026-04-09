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

import pytest
import numpy as np
from pathlib import Path
import sys
import json
from unittest.mock import patch

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

        # Test with mouse brain mass
        mouse_pci = model.compute_predicted_pci(0.4)
        assert 0.1 < mouse_pci < 0.5

        # Test with very small brain
        small_pci = model.compute_predicted_pci(0.001)
        assert small_pci > 0

    def test_predict_species_consciousness(self):
        """Test species consciousness prediction"""
        from Theory.APGI_Cross_Species_Scaling import (
            predict_species_consciousness,
            SpeciesParameters,
        )

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
            predict_species_consciousness,
            SpeciesParameters,
        )

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
        from Theory.APGI_Cross_Species_Scaling import (
            generate_species_comparison_report,
        )

        report = generate_species_comparison_report()

        assert "APGI CROSS-SPECIES SCALING VALIDATION REPORT" in report
        assert "Human" in report
        assert "SCALING LAWS:" in report

    def test_get_implementation_metadata(self):
        """Test implementation metadata retrieval"""
        from Theory.APGI_Cross_Species_Scaling import get_implementation_metadata

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
        from Theory.APGI_Open_Science_Framework import serialize
        from datetime import datetime

        dt = datetime.now()
        result = serialize(dt)
        assert isinstance(result, str)


class TestCulturalNeuroscience:
    """Test Cultural Neuroscience module paths"""

    def test_cultural_parameter_modulator_init(self):
        """Test CulturalParameterModulator initialization"""
        from Theory.APGI_Cultural_Neuroscience import CulturalParameterModulator

        modulator = CulturalParameterModulator()
        assert hasattr(modulator, "cultural_dimensions")

    def test_cultural_baseline_shift(self):
        """Test cultural baseline shift computation"""
        from Theory.APGI_Cultural_Neuroscience import CulturalParameterModulator

        modulator = CulturalParameterModulator()
        shift = modulator.get_cultural_baseline_shift("Japan")

        assert isinstance(shift, dict)
        assert "tau_theta" in shift or "collectivism_score" in shift

    def test_all_cultural_regions(self):
        """Test that all cultural regions have defined parameters"""
        from Theory.APGI_Cultural_Neuroscience import CulturalParameterModulator

        modulator = CulturalParameterModulator()
        regions = ["USA", "Japan", "Germany", "Brazil", "India", "Nordic"]

        for region in regions:
            shift = modulator.get_cultural_baseline_shift(region)
            assert isinstance(shift, dict)


class TestTuringMachine:
    """Test Turing Machine module paths"""

    def test_turing_machine_init(self):
        """Test APGITuringMachine initialization"""
        from Theory.APGI_Turing_Machine import APGITuringMachine

        tm = APGITuringMachine()
        assert hasattr(tm, "tape")
        assert hasattr(tm, "head_position")

    def test_turing_machine_step(self):
        """Test Turing machine step execution"""
        from Theory.APGI_Turing_Machine import APGITuringMachine

        tm = APGITuringMachine()
        initial_pos = tm.head_position

        tm.step()
        assert tm.head_position != initial_pos or tm.steps > 0

    def test_turing_machine_run(self):
        """Test Turing machine run execution"""
        from Theory.APGI_Turing_Machine import APGITuringMachine

        tm = APGITuringMachine()
        tm.run(max_steps=10)

        assert tm.steps <= 10

    def test_turing_machine_reset(self):
        """Test Turing machine reset"""
        from Theory.APGI_Turing_Machine import APGITuringMachine

        tm = APGITuringMachine()
        tm.run(max_steps=5)
        tm.reset()

        assert tm.head_position == 0
        assert tm.steps == 0


class TestPsychologicalStates:
    """Test Psychological States module paths"""

    def test_psychological_state_init(self):
        """Test PsychologicalState initialization"""
        from Theory.APGI_Psychological_States import PsychologicalState

        state = PsychologicalState(
            arousal=0.5,
            valence=0.3,
            surprise=0.2,
            theta=0.4,
        )

        assert state.arousal == 0.5
        assert state.valence == 0.3

    def test_psychological_state_from_dict(self):
        """Test PsychologicalState from dict"""
        from Theory.APGI_Psychological_States import PsychologicalState

        data = {"arousal": 0.6, "valence": 0.4, "surprise": 0.3, "theta": 0.5}
        state = PsychologicalState.from_dict(data)

        assert state.arousal == 0.6
        assert state.valence == 0.4

    def test_psychological_state_to_dict(self):
        """Test PsychologicalState to dict"""
        from Theory.APGI_Psychological_States import PsychologicalState

        state = PsychologicalState(
            arousal=0.5,
            valence=0.3,
            surprise=0.2,
            theta=0.4,
        )

        data = state.to_dict()
        assert data["arousal"] == 0.5
        assert data["valence"] == 0.3


class TestTheoryModuleImports:
    """Test Theory module import paths"""

    def test_theory_package_import(self):
        """Test Theory package imports"""
        from Theory import CrossSpeciesScaling
        from Theory import OpenScienceValidator
        from Theory import CulturalParameterModulator
        from Theory import APGITuringMachine
        from Theory import PsychState

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
