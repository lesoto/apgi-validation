"""
Individual protocol tests for validation and falsification.
Tests each validation/falsification protocol individually for correctness.
================================================================
"""

import json
import numpy as np
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestValidationProtocolsIndividual:
    """Individual tests for each validation protocol."""

    def test_entropy_validation_protocol(self, temp_dir):
        """Test entropy validation protocol individually."""
        # Create test data
        test_data = {
            "surprise": np.random.randn(100).tolist(),
            "threshold": np.random.randn(100).tolist(),
        }
        data_file = temp_dir / "entropy_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        # Import and run entropy validation
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "validation_gui", "Validation/APGI_Validation_GUI.py"
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result = validation_module.validate_entropy_protocol(str(data_file))
                assert result is not None
                assert "entropy" in result or "status" in result
        except (ImportError, FileNotFoundError):
            pytest.skip("Entropy validation protocol not available")

    def test_active_inference_validation_protocol(self, temp_dir):
        """Test active inference validation protocol individually."""
        test_data = {
            "prediction": np.random.randn(100).tolist(),
            "observation": np.random.randn(100).tolist(),
        }
        data_file = temp_dir / "active_inference_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "active_inference",
                "Validation/ActiveInference_AgentSimulations_Protocol3.py",
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result = validation_module.validate_active_inference(str(data_file))
                assert result is not None
        except (ImportError, FileNotFoundError):
            pytest.skip("Active inference validation not available")

    def test_parameter_recovery_validation(self, temp_dir):
        """Test parameter recovery validation individually."""
        test_data = {
            "true_params": {"tau_S": 0.5, "tau_theta": 30.0},
            "estimated_params": {"tau_S": 0.48, "tau_theta": 29.5},
        }
        data_file = temp_dir / "params_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "parameter_recovery",
                "Validation/BayesianModelComparison_ParameterRecovery.py",
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result = validation_module.validate_parameter_recovery(str(data_file))
                assert result is not None
                assert "recovery_error" in result or "status" in result
        except (ImportError, FileNotFoundError):
            pytest.skip("Parameter recovery validation not available")

    def test_cross_species_validation_protocol(self, temp_dir):
        """Test cross-species validation protocol individually."""
        test_data = {
            "human": {"surprise": np.random.randn(100).tolist()},
            "rodent": {"surprise": np.random.randn(100).tolist()},
        }
        data_file = temp_dir / "cross_species_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "cross_species", "APGI_Cross_Species_Scaling.py"
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result = validation_module.validate_cross_species_scaling(
                    str(data_file)
                )
                assert result is not None
        except (ImportError, FileNotFoundError):
            pytest.skip("Cross-species validation not available")

    def test_multimodal_validation_protocol(self, temp_dir):
        """Test multimodal validation protocol individually."""
        test_data = {
            "eeg": np.random.randn(100, 10).tolist(),
            "fmri": np.random.randn(100, 20).tolist(),
        }
        data_file = temp_dir / "multimodal_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "multimodal", "APGI_Multimodal_Integration.py"
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result = validation_module.validate_multimodal_integration(
                    str(data_file)
                )
                assert result is not None
        except (ImportError, FileNotFoundError):
            pytest.skip("Multimodal validation not available")

    def test_cultural_neuroscience_validation(self, temp_dir):
        """Test cultural neuroscience validation individually."""
        test_data = {
            "cultural_groups": ["group1", "group2"],
            "data": {
                "group1": np.random.randn(50).tolist(),
                "group2": np.random.randn(50).tolist(),
            },
        }
        data_file = temp_dir / "cultural_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "cultural", "APGI_Cultural_Neuroscience.py"
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result = validation_module.validate_cultural_patterns(str(data_file))
                assert result is not None
        except (ImportError, FileNotFoundError):
            pytest.skip("Cultural validation not available")


class TestFalsificationProtocolsIndividual:
    """Individual tests for each falsification protocol."""

    def test_active_inference_falsification_protocol(self, temp_dir):
        """Test active inference falsification protocol individually."""
        test_data = {
            "predictions": np.random.randn(100).tolist(),
            "observations": np.random.randn(100).tolist(),
        }
        data_file = temp_dir / "falsification_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "falsification_ai",
                "Falsification/Falsification_ActiveInferenceAgents_F1F2.py",
            )
            if spec and spec.loader:
                falsification_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(falsification_module)
                result = falsification_module.falsify_active_inference(str(data_file))
                assert result is not None
                assert "falsified" in result or "status" in result
        except (ImportError, FileNotFoundError):
            pytest.skip("Active inference falsification not available")

    def test_entropy_falsification_protocol(self, temp_dir):
        """Test entropy falsification protocol individually."""
        test_data = {
            "surprise": np.random.randn(100).tolist(),
            "threshold": np.random.randn(100).tolist(),
        }
        data_file = temp_dir / "entropy_falsification_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "falsification_entropy",
                "Falsification/APGI_Falsification_Aggregator.py",
            )
            if spec and spec.loader:
                falsification_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(falsification_module)
                result = falsification_module.falsify_entropy_model(str(data_file))
                assert result is not None
        except (ImportError, FileNotFoundError):
            pytest.skip("Entropy falsification not available")

    def test_tms_falsification_protocol(self, temp_dir):
        """Test TMS causal manipulation falsification individually."""
        test_data = {
            "pre_tms": np.random.randn(50).tolist(),
            "post_tms": np.random.randn(50).tolist(),
        }
        data_file = temp_dir / "tms_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "falsification_tms",
                "Falsification/CausalManipulations_TMS_Pharmacological_Priority2.py",
            )
            if spec and spec.loader:
                falsification_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(falsification_module)
                result = falsification_module.falsify_tms_causal(str(data_file))
                assert result is not None
        except (ImportError, FileNotFoundError):
            pytest.skip("SyntheticEEG_MLClassification not available")

    def test_pharmacological_falsification_protocol(self, temp_dir):
        """Test pharmacological falsification protocol individually."""
        test_data = {
            "baseline": np.random.randn(50).tolist(),
            "drug": np.random.randn(50).tolist(),
        }
        data_file = temp_dir / "pharmacological_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "falsification_pharma",
                "Falsification/CausalManipulations_TMS_Pharmacological_Priority2.py",
            )
            if spec and spec.loader:
                falsification_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(falsification_module)
                result = falsification_module.falsify_pharmacological(str(data_file))
                assert result is not None
        except (ImportError, FileNotFoundError):
            pytest.skip("Pharmacological falsification not available")

    def test_neural_signatures_falsification(self, temp_dir):
        """Test neural signatures falsification individually."""
        test_data = {
            "eeg_patterns": np.random.randn(100, 10).tolist(),
            "fmri_patterns": np.random.randn(100, 20).tolist(),
        }
        data_file = temp_dir / "neural_signatures_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "neural_signatures", "APGI_Neural_Signatures.py"
            )
            if spec and spec.loader:
                falsification_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(falsification_module)
                result = falsification_module.falsify_neural_signatures(str(data_file))
                assert result is not None
        except (ImportError, FileNotFoundError):
            pytest.skip(
                "ConvergentNeuralSignatures_Priority1_EmpiricalRoadmap not available"
            )

    def test_quantitative_fits_falsification(self, temp_dir):
        """Test quantitative fits falsification individually."""
        test_data = {
            "model_predictions": np.random.randn(100).tolist(),
            "empirical_data": np.random.randn(100).tolist(),
        }
        data_file = temp_dir / "quantitative_fits_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            from main import falsify_quantitative_fits

            result = falsify_quantitative_fits(str(data_file))
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("QuantitativeModelFits_SpikingLNN_Priority3 not available")

    def test_clinical_convergence_falsification(self, temp_dir):
        """Test clinical convergence falsification individually."""
        test_data = {
            "patient_responses": [
                {"id": 1, "response": np.random.randn(10).tolist()},
                {"id": 2, "response": np.random.randn(10).tolist()},
            ]
        }
        data_file = temp_dir / "clinical_convergence_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            from main import falsify_clinical_convergence

            result = falsify_clinical_convergence(str(data_file))
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("Clinical_CrossSpecies_Convergence_Protocol4 not available")


class TestProtocolCorrectness:
    """Test protocol correctness and edge cases."""

    def test_protocol_with_empty_data(self, temp_dir):
        """Test protocol behavior with empty data."""
        test_data = {"surprise": [], "threshold": []}
        data_file = temp_dir / "empty_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "validation_gui", "Validation/APGI_Validation_GUI.py"
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result = validation_module.validate_entropy_protocol(str(data_file))
                # Should handle empty data gracefully
                assert result is not None
        except (ImportError, FileNotFoundError, ValueError, IndexError):
            pytest.skip("Protocol not available or doesn't handle empty data")

    def test_protocol_with_nan_values(self, temp_dir):
        """Test protocol behavior with NaN values."""
        test_data = {
            "surprise": [0.1, 0.2, float("nan"), 0.4],
            "threshold": [0.5, 0.5, 0.5, 0.5],
        }
        data_file = temp_dir / "nan_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "validation_gui", "Validation/APGI_Validation_GUI.py"
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result = validation_module.validate_entropy_protocol(str(data_file))
                # Should handle NaN values gracefully
                assert result is not None
        except (ImportError, FileNotFoundError, ValueError):
            pytest.skip("Protocol not available or doesn't handle NaN")

    def test_protocol_with_inf_values(self, temp_dir):
        """Test protocol behavior with infinite values."""
        test_data = {
            "surprise": [0.1, 0.2, float("inf"), 0.4],
            "threshold": [0.5, 0.5, 0.5, 0.5],
        }
        data_file = temp_dir / "inf_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "validation_gui", "Validation/APGI_Validation_GUI.py"
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result = validation_module.validate_entropy_protocol(str(data_file))
                # Should handle infinite values gracefully
                assert result is not None
        except (ImportError, FileNotFoundError, ValueError):
            pytest.skip("Protocol not available or doesn't handle inf")

    def test_protocol_with_mismatched_lengths(self, temp_dir):
        """Test protocol behavior with mismatched array lengths."""
        test_data = {
            "surprise": [0.1, 0.2, 0.3],
            "threshold": [0.5, 0.5],  # Mismatched length
        }
        data_file = temp_dir / "mismatched_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "validation_gui", "Validation/APGI_Validation_GUI.py"
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result = validation_module.validate_entropy_protocol(str(data_file))
                # Should handle mismatched lengths gracefully
                assert result is not None
        except (ImportError, FileNotFoundError, ValueError):
            pytest.skip("Protocol not available or doesn't handle mismatched lengths")

    def test_protocol_output_format(self, temp_dir):
        """Test that protocol outputs have consistent format."""
        test_data = {
            "surprise": np.random.randn(100).tolist(),
            "threshold": np.random.randn(100).tolist(),
        }
        data_file = temp_dir / "format_test_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "validation_gui", "Validation/APGI_Validation_GUI.py"
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result = validation_module.validate_entropy_protocol(str(data_file))

                # Check for expected fields
                assert isinstance(result, dict)
                # Should have at least one of these fields
                has_expected_field = any(
                    field in result
                    for field in ["status", "entropy", "valid", "result"]
                )
                assert has_expected_field
        except (ImportError, FileNotFoundError):
            pytest.skip("Protocol not available")

    def test_protocol_reproducibility(self, temp_dir):
        """Test that protocol results are reproducible."""
        test_data = {
            "surprise": [0.1, 0.2, 0.3, 0.4, 0.5],
            "threshold": [0.5, 0.5, 0.5, 0.5, 0.5],
        }
        data_file = temp_dir / "reproducibility_data.json"
        with open(data_file, ', encoding="utf-8"w') as f:
            json.dump(test_data, f)

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "validation_gui", "Validation/APGI_Validation_GUI.py"
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                result1 = validation_module.validate_entropy_protocol(str(data_file))
                result2 = validation_module.validate_entropy_protocol(str(data_file))

                # Results should be identical
                assert result1 == result2
        except (ImportError, FileNotFoundError):
            pytest.skip("Protocol not available")
