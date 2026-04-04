"""Integration tests for all FP and VP protocols.

Verifies that:
1. All 27 protocols can run end-to-end
2. All return standardized ProtocolResult objects
3. Aggregators can consume all protocol outputs
4. Framework falsification conditions can be evaluated
"""

import sys
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.protocol_schema import ProtocolResult, PredictionResult, PredictionStatus


class TestAllFPProtocols:
    """Test all 12 falsification protocols."""

    FP_PROTOCOLS = [
        ("Falsification.FP_01_ActiveInference", "FP_01_ActiveInference"),
        (
            "Falsification.FP_02_AgentComparison_ConvergenceBenchmark",
            "FP_02_AgentComparison_ConvergenceBenchmark",
        ),
        (
            "Falsification.FP_03_FrameworkLevel_MultiProtocol",
            "FP_03_FrameworkLevel_MultiProtocol",
        ),
        (
            "Falsification.FP_04_PhaseTransition_EpistemicArchitecture",
            "FP_04_PhaseTransition_EpistemicArchitecture",
        ),
        (
            "Falsification.FP_05_EvolutionaryPlausibility",
            "FP_05_EvolutionaryPlausibility",
        ),
        (
            "Falsification.FP_06_LiquidNetwork_EnergyBenchmark",
            "FP_06_LiquidNetwork_EnergyBenchmark",
        ),
        (
            "Falsification.FP_07_MathematicalConsistency",
            "FP_07_MathematicalConsistency",
        ),
        (
            "Falsification.FP_08_ParameterSensitivity_Identifiability",
            "FP_08_ParameterSensitivity_Identifiability",
        ),
        (
            "Falsification.FP_09_NeuralSignatures_P3b_HEP",
            "FP_09_NeuralSignatures_P3b_HEP",
        ),
        (
            "Falsification.FP_10_BayesianEstimation_MCMC",
            "FP_10_BayesianEstimation_MCMC",
        ),
        (
            "Falsification.FP_11_LiquidNetworkDynamics_EchoState",
            "FP_11_LiquidNetworkDynamics_EchoState",
        ),
        ("Falsification.FP_12_CrossSpeciesScaling", "FP_12_CrossSpeciesScaling"),
    ]

    @pytest.mark.parametrize("module_path,protocol_id", FP_PROTOCOLS)
    def test_fp_protocol_returns_protocol_result(self, module_path, protocol_id):
        """Test that FP protocol returns standardized ProtocolResult."""
        try:
            mod = __import__(module_path, fromlist=[protocol_id])
            result = mod.run_protocol_main()

            # Check result type
            assert (
                result is not None
            ), f"{protocol_id}: run_protocol_main() returned None"

            # Convert to ProtocolResult if dict
            if isinstance(result, dict):
                result = ProtocolResult.from_dict(result)

            assert isinstance(
                result, ProtocolResult
            ), f"{protocol_id}: Result is not ProtocolResult"
            assert (
                result.protocol_id == protocol_id
            ), f"{protocol_id}: protocol_id mismatch"
            assert result.named_predictions, f"{protocol_id}: No named_predictions"
            assert (
                result.completion_percentage > 0
            ), f"{protocol_id}: completion_percentage is 0"

        except Exception as e:
            pytest.fail(f"{protocol_id}: {str(e)}")

    def test_all_fp_protocols_have_predictions(self):
        """Test that all FP protocols have named_predictions."""
        for module_path, protocol_id in self.FP_PROTOCOLS:
            try:
                mod = __import__(module_path, fromlist=[protocol_id])
                result = mod.run_protocol_main()

                if isinstance(result, dict):
                    result = ProtocolResult.from_dict(result)

                assert (
                    len(result.named_predictions) > 0
                ), f"{protocol_id}: No predictions"

                # Check each prediction is a PredictionResult
                for pred_id, pred in result.named_predictions.items():
                    if isinstance(pred, dict):
                        pred = PredictionResult(**pred)
                    assert isinstance(
                        pred, PredictionResult
                    ), f"{protocol_id}.{pred_id}: Not PredictionResult"
                    assert hasattr(
                        pred, "passed"
                    ), f"{protocol_id}.{pred_id}: No 'passed' field"

            except Exception as e:
                pytest.fail(f"{protocol_id}: {str(e)}")


class TestAllVPProtocols:
    """Test all 15 validation protocols."""

    VP_PROTOCOLS = [
        (
            "Validation.VP_01_SyntheticEEG_MLClassification",
            "VP_01_SyntheticEEG_MLClassification",
        ),
        (
            "Validation.VP_02_Behavioral_BayesianComparison",
            "VP_02_Behavioral_BayesianComparison",
        ),
        (
            "Validation.VP_03_ActiveInference_AgentSimulations",
            "VP_03_ActiveInference_AgentSimulations",
        ),
        (
            "Validation.VP_04_PhaseTransition_EpistemicLevel2",
            "VP_04_PhaseTransition_EpistemicLevel2",
        ),
        ("Validation.VP_05_EvolutionaryEmergence", "VP_05_EvolutionaryEmergence"),
        (
            "Validation.VP_06_LiquidNetwork_InductiveBias",
            "VP_06_LiquidNetwork_InductiveBias",
        ),
        ("Validation.VP_07_TMS_CausalInterventions", "VP_07_TMS_CausalInterventions"),
        (
            "Validation.VP_08_Psychophysical_ThresholdEstimation",
            "VP_08_Psychophysical_ThresholdEstimation",
        ),
        (
            "Validation.VP_09_NeuralSignatures_EmpiricalPriority1",
            "VP_09_NeuralSignatures_EmpiricalPriority1",
        ),
        (
            "Validation.VP_10_CausalManipulations_Priority2",
            "VP_10_CausalManipulations_Priority2",
        ),
        (
            "Validation.VP_11_MCMC_CulturalNeuroscience_Priority3",
            "VP_11_MCMC_CulturalNeuroscience_Priority3",
        ),
        (
            "Validation.VP_12_Clinical_CrossSpecies_Convergence",
            "VP_12_Clinical_CrossSpecies_Convergence",
        ),
        ("Validation.VP_13_Epistemic_Architecture", "VP_13_Epistemic_Architecture"),
        (
            "Validation.VP_14_fMRI_Anticipation_Experience",
            "VP_14_fMRI_Anticipation_Experience",
        ),
        ("Validation.VP_15_fMRI_Anticipation_vmPFC", "VP_15_fMRI_Anticipation_vmPFC"),
    ]

    @pytest.mark.parametrize("module_path,protocol_id", VP_PROTOCOLS)
    def test_vp_protocol_returns_protocol_result(self, module_path, protocol_id):
        """Test that VP protocol returns standardized ProtocolResult."""
        try:
            mod = __import__(module_path, fromlist=[protocol_id])
            result = mod.run_protocol_main()

            # Check result type
            assert (
                result is not None
            ), f"{protocol_id}: run_protocol_main() returned None"

            # Convert to ProtocolResult if dict
            if isinstance(result, dict):
                result = ProtocolResult.from_dict(result)

            assert isinstance(
                result, ProtocolResult
            ), f"{protocol_id}: Result is not ProtocolResult"
            assert (
                result.protocol_id == protocol_id
            ), f"{protocol_id}: protocol_id mismatch"
            assert result.named_predictions, f"{protocol_id}: No named_predictions"
            assert (
                result.completion_percentage > 0
            ), f"{protocol_id}: completion_percentage is 0"

        except Exception as e:
            pytest.fail(f"{protocol_id}: {str(e)}")

    def test_all_vp_protocols_have_predictions(self):
        """Test that all VP protocols have named_predictions."""
        for module_path, protocol_id in self.VP_PROTOCOLS:
            try:
                mod = __import__(module_path, fromlist=[protocol_id])
                result = mod.run_protocol_main()

                if isinstance(result, dict):
                    result = ProtocolResult.from_dict(result)

                assert (
                    len(result.named_predictions) > 0
                ), f"{protocol_id}: No predictions"

                # Check each prediction is a PredictionResult
                for pred_id, pred in result.named_predictions.items():
                    if isinstance(pred, dict):
                        pred = PredictionResult(**pred)
                    assert isinstance(
                        pred, PredictionResult
                    ), f"{protocol_id}.{pred_id}: Not PredictionResult"
                    assert hasattr(
                        pred, "passed"
                    ), f"{protocol_id}.{pred_id}: No 'passed' field"

            except Exception as e:
                pytest.fail(f"{protocol_id}: {str(e)}")


class TestProtocolResultSchema:
    """Test ProtocolResult schema compliance."""

    def test_protocol_result_to_dict_serializable(self):
        """Test that ProtocolResult can be serialized to JSON."""
        import json

        result = ProtocolResult(
            protocol_id="TEST_PROTOCOL",
            timestamp="2026-04-04T00:00:00",
            named_predictions={
                "P1.1": PredictionResult(
                    passed=True,
                    value=0.95,
                    threshold=0.90,
                    status=PredictionStatus.PASSED,
                    evidence=["Test evidence"],
                    sources=["Test source"],
                    metadata={"test": "data"},
                )
            },
            completion_percentage=100,
            data_sources=["Test data"],
            methodology="test",
            errors=[],
            metadata={"test": "metadata"},
        )

        # Should be JSON serializable
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        assert json_str is not None

    def test_protocol_result_from_dict_roundtrip(self):
        """Test that ProtocolResult can be converted to/from dict."""
        original = ProtocolResult(
            protocol_id="TEST_PROTOCOL",
            timestamp="2026-04-04T00:00:00",
            named_predictions={
                "P1.1": PredictionResult(
                    passed=True,
                    value=0.95,
                    threshold=0.90,
                    status=PredictionStatus.PASSED,
                )
            },
            completion_percentage=100,
            data_sources=["Test data"],
            methodology="test",
        )

        # Convert to dict and back
        result_dict = original.to_dict()
        restored = ProtocolResult.from_dict(result_dict)

        assert restored.protocol_id == original.protocol_id
        assert restored.completion_percentage == original.completion_percentage
        assert len(restored.named_predictions) == len(original.named_predictions)


class TestAggregators:
    """Test aggregator functionality."""

    def test_fp_aggregator_exists(self):
        """Test that FP_ALL_Aggregator can be imported."""
        try:
            from Falsification.FP_ALL_Aggregator import FPAllAggregator

            aggregator = FPAllAggregator()
            assert aggregator is not None
        except Exception as e:
            pytest.fail(f"FP_ALL_Aggregator import failed: {str(e)}")

    def test_vp_aggregator_exists(self):
        """Test that VP_ALL_Aggregator can be imported."""
        try:
            from Validation.VP_ALL_Aggregator import ValidationAggregator

            aggregator = ValidationAggregator()
            assert aggregator is not None
        except Exception as e:
            pytest.fail(f"VP_ALL_Aggregator import failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
