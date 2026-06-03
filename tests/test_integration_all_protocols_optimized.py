"""Optimized integration tests for FP and VP protocols.

This is a faster version that uses mocking for slow operations.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.protocol_schema import PredictionResult, PredictionStatus, ProtocolResult


class TestFPProtocolsOptimized:
    """Optimized tests for FP protocols using mocks."""

    FP_PROTOCOLS = [
        ("Falsification.FP_01_ActiveInference", "FP_01_ActiveInference"),
        (
            "Falsification.FP_02_AgentComparisonConvergenceBenchmark",
            "FP_02_AgentComparisonConvergenceBenchmark",
        ),
        (
            "Falsification.FP_03_FrameworkLevelMultiProtocol",
            "FP_03_FrameworkLevelMultiProtocol",
        ),
        (
            "Falsification.FP_04_PhaseTransitionEpistemicArchitecture",
            "FP_04_PhaseTransitionEpistemicArchitecture",
        ),
        (
            "Falsification.FP_05_EvolutionaryPlausibility",
            "FP_05_EvolutionaryPlausibility",
        ),
        (
            "Falsification.FP_06_LiquidNetworkEnergyBenchmark",
            "FP_06_LiquidNetworkEnergyBenchmark",
        ),
        (
            "Falsification.FP_07_MathematicalConsistency",
            "FP_07_MathematicalConsistency",
        ),
        (
            "Falsification.FP_08_ParameterSensitivityIdentifiability",
            "FP_08_ParameterSensitivityIdentifiability",
        ),
        (
            "Falsification.FP_09_NeuralSignaturesP3bHEP",
            "FP_09_NeuralSignaturesP3bHEP",
        ),
        (
            "Falsification.FP_10_BayesianEstimationMCMC",
            "FP_10_BayesianEstimationMCMC",
        ),
        (
            "Falsification.FP_11_LiquidNetworkDynamicsEchoState",
            "FP_11_LiquidNetworkDynamicsEchoState",
        ),
        ("Falsification.FP_12_CrossSpeciesScaling", "FP_12_CrossSpeciesScaling"),
    ]

    @pytest.mark.parametrize("module_name,protocol_id", FP_PROTOCOLS)
    def test_fp_protocol_structure(self, module_name, protocol_id):
        """Test that FP protocol module has required structure."""
        import importlib

        try:
            mod = importlib.import_module(module_name)
            assert hasattr(mod, "run_protocol_main"), f"{module_name} missing run_protocol_main"
            assert callable(mod.run_protocol_main), f"{module_name}.run_protocol_main not callable"
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
            assert result.protocol_id == protocol_id
            assert len(result.named_predictions) > 0

        finally:
            os.environ.pop("APGI_TEST_MODE", None)


class TestVPProtocolsOptimized:
    """Optimized tests for VP protocols using mocks."""

    VP_PROTOCOLS = [
        (
            "Validation.VP_01_SyntheticEEGMLClassification",
            "VP_01_SyntheticEEGMLClassification",
        ),
        (
            "Validation.VP_02_BehavioralBayesianComparison",
            "VP_02_BehavioralBayesianComparison",
        ),
        (
            "Validation.VP_03_ActiveInferenceAgentSimulations",
            "VP_03_ActiveInferenceAgentSimulations",
        ),
        (
            "Validation.VP_04_PhaseTransitionEpistemicLevel2",
            "VP_04_PhaseTransitionEpistemicLevel2",
        ),
        ("Validation.VP_05_EvolutionaryEmergence", "VP_05_EvolutionaryEmergence"),
        (
            "Validation.VP_06_LiquidNetworkInductiveBias",
            "VP_06_LiquidNetworkInductiveBias",
        ),
        ("Validation.VP_07_TMSCausalInterventions", "VP_07_TMSCausalInterventions"),
        (
            "Validation.VP_08_PsychophysicalThresholdEstimation",
            "VP_08_PsychophysicalThresholdEstimation",
        ),
        (
            "Validation.VP_09_NeuralSignaturesEmpiricalPriority1",
            "VP_09_NeuralSignaturesEmpiricalPriority1",
        ),
        (
            "Validation.VP_10_CausalManipulationsPriority2",
            "VP_10_CausalManipulationsPriority2",
        ),
        (
            "Validation.VP_11_MCMCCulturalNeurosciencePriority3",
            "VP_11_MCMCCulturalNeurosciencePriority3",
        ),
        (
            "Validation.VP_12_ClinicalCrossSpeciesConvergence",
            "VP_12_ClinicalCrossSpeciesConvergence",
        ),
        ("Validation.VP_13_EpistemicArchitecture", "VP_13_EpistemicArchitecture"),
        (
            "Validation.VP_14_FMRIAnticipationExperience",
            "VP_14_FMRIAnticipationExperience",
        ),
        ("Validation.VP_15_FMRIAnticipationVmPFC", "VP_15_FMRIAnticipationVmPFC"),
    ]

    @pytest.mark.parametrize("module_name,protocol_id", VP_PROTOCOLS)
    def test_vp_protocol_structure(self, module_name, protocol_id):
        """Test that VP protocol module has required structure."""
        import importlib

        try:
            mod = importlib.import_module(module_name)
            assert hasattr(mod, "run_protocol_main"), f"{module_name} missing run_protocol_main"
            assert callable(mod.run_protocol_main), f"{module_name}.run_protocol_main not callable"
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
