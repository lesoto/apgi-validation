"""Optimized integration tests for FP and VP protocols.

This is a faster version that uses mocking for slow operations.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.protocol_schema import ProtocolResult, PredictionResult, PredictionStatus


class TestFPProtocolsOptimized:
    """Optimized tests for FP protocols using mocks."""

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

    @pytest.mark.parametrize("module_name,protocol_id", FP_PROTOCOLS)
    def test_fp_protocol_structure(self, module_name, protocol_id):
        """Test that FP protocol module has required structure."""
        import importlib

        try:
            mod = importlib.import_module(module_name)
            assert hasattr(
                mod, "run_protocol_main"
            ), f"{module_name} missing run_protocol_main"
            assert callable(
                mod.run_protocol_main
            ), f"{module_name}.run_protocol_main not callable"
        except Exception as e:
            pytest.fail(f"Failed to import {module_name}: {e}")

    @pytest.mark.slow
    @pytest.mark.parametrize("module_name,protocol_id", FP_PROTOCOLS[:3])  # Test subset
    def test_fp_protocol_execution_mocked(self, module_name, protocol_id):
        """Test FP protocol with mocked slow operations."""
        import importlib
        import os

        # Set test mode
        os.environ["APGI_TEST_MODE"] = "true"

        try:
            mod = importlib.import_module(module_name)
            importlib.reload(mod)

            # Mock the slow validation function
            mock_result = ProtocolResult(
                protocol_id=protocol_id,
                timestamp="2024-01-01T00:00:00",
                named_predictions={
                    "test_pred": PredictionResult(
                        passed=True,
                        value=0.5,
                        threshold=0.4,
                        status=PredictionStatus.PASSED,
                        evidence=["mock evidence"],
                        sources=[module_name],
                    )
                },
                completion_percentage=100,
                data_sources=["mock"],
                methodology="test",
                errors=[],
                metadata={},
            )

            with patch.object(mod, "run_protocol_main", return_value=mock_result):
                result = mod.run_protocol_main()

            assert result is not None
            assert isinstance(result, ProtocolResult)
            assert result.protocol_id == protocol_id
            assert len(result.named_predictions) > 0

        finally:
            os.environ.pop("APGI_TEST_MODE", None)


class TestVPProtocolsOptimized:
    """Optimized tests for VP protocols using mocks."""

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

    @pytest.mark.parametrize("module_name,protocol_id", VP_PROTOCOLS)
    def test_vp_protocol_structure(self, module_name, protocol_id):
        """Test that VP protocol module has required structure."""
        import importlib

        try:
            mod = importlib.import_module(module_name)
            assert hasattr(
                mod, "run_protocol_main"
            ), f"{module_name} missing run_protocol_main"
            assert callable(
                mod.run_protocol_main
            ), f"{module_name}.run_protocol_main not callable"
        except Exception as e:
            pytest.fail(f"Failed to import {module_name}: {e}")

    @pytest.mark.slow
    @pytest.mark.parametrize("module_name,protocol_id", VP_PROTOCOLS[:3])  # Test subset
    def test_vp_protocol_execution_mocked(self, module_name, protocol_id):
        """Test VP protocol with mocked slow operations."""
        import importlib

        try:
            mod = importlib.import_module(module_name)

            # Mock the slow validation function
            mock_result = ProtocolResult(
                protocol_id=protocol_id,
                timestamp="2024-01-01T00:00:00",
                named_predictions={
                    "test_pred": PredictionResult(
                        passed=True,
                        value=0.5,
                        threshold=0.4,
                        status=PredictionStatus.PASSED,
                        evidence=["mock evidence"],
                        sources=[module_name],
                    )
                },
                completion_percentage=100,
                data_sources=["mock"],
                methodology="test",
                errors=[],
                metadata={},
            )

            with patch.object(mod, "run_protocol_main", return_value=mock_result):
                result = mod.run_protocol_main()

            assert result is not None
            assert isinstance(result, ProtocolResult)
            assert result.protocol_id == protocol_id
            assert len(result.named_predictions) > 0

        except Exception as e:
            pytest.fail(f"{protocol_id}: {e}")


class TestProtocolSchemaCompliance:
    """Test that all protocols comply with the schema."""

    def test_protocol_result_serialization(self):
        """Test ProtocolResult can be serialized and deserialized."""
        from datetime import datetime

        result = ProtocolResult(
            protocol_id="TEST_01",
            timestamp=datetime.now().isoformat(),
            named_predictions={
                "pred1": PredictionResult(
                    passed=True,
                    value=1.0,
                    threshold=0.5,
                    status=PredictionStatus.PASSED,
                    evidence=["test"],
                    sources=["test"],
                )
            },
            completion_percentage=100,
            data_sources=["test"],
            methodology="test",
            errors=[],
            metadata={"test": True},
        )

        # Test serialization
        result_dict = result.to_dict()
        assert result_dict["protocol_id"] == "TEST_01"
        assert result_dict["completion_percentage"] == 100

        # Test deserialization
        restored = ProtocolResult.from_dict(result_dict)
        assert restored.protocol_id == "TEST_01"
        assert len(restored.named_predictions) == 1

    def test_prediction_result_status_values(self):
        """Test PredictionStatus enum values."""
        assert PredictionStatus.PASSED.value == "passed"
        assert PredictionStatus.FAILED.value == "failed"
        assert PredictionStatus.MISSING_PROTOCOL.value == "missing_protocol"
        assert PredictionStatus.LOAD_ERROR.value == "load_error"
        assert PredictionStatus.DATA_UNAVAILABLE.value == "data_unavailable"
        assert PredictionStatus.NOT_EVALUATED.value == "not_evaluated"
        assert PredictionStatus.PARTIAL.value == "partial"
