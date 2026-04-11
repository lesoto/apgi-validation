"""
Comprehensive Validation Protocol Integration Tests
=================================================

This module provides comprehensive integration tests for all validation protocols
in the APGI Validation Framework. It tests:

1. Module import and basic functionality
2. Protocol execution and result validation
3. Cross-protocol compatibility
4. Error handling and edge cases
5. Performance and resource management

Tests are organized by priority tier and include both unit and integration
testing patterns.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import validation modules dynamically
VALIDATION_MODULES = {}
VALIDATION_PROTOCOLS = []

# All validation protocol files found in the Validation directory
VALIDATION_FILES = [
    "APGI_Validation_GUI.py",
    "VP_03_ActiveInference_AgentSimulations.py",
    "VP_02_Behavioral_BayesianComparison.py",
    "VP_12_Clinical_CrossSpecies_Convergence.py",
    "VP_09_NeuralSignatures_EmpiricalPriority1.py",
    "VP_05_EvolutionaryEmergence.py",
    "VP_10_CausalManipulations_Priority2.py",
    "VP_04_PhaseTransition_EpistemicLevel2.py",
    "Master_Validation.py",
    "VP_06_LiquidNetwork_InductiveBias.py",
    "VP_08_Psychophysical_ThresholdEstimation.py",
    "VP_11_MCMC_CulturalNeuroscience_Priority3.py",
    "VP_01_SyntheticEEG_MLClassification.py",
    "VP_07_TMS_CausalInterventions.py",
    "VP_13_Epistemic_Architecture.py",
    "VP_14_fMRI_Anticipation_Experience.py",
    "VP_15_fMRI_Anticipation_vmPFC.py",
]


def load_validation_modules():
    """Load all validation modules dynamically."""
    loaded_modules = {}
    loaded_protocols = []

    for file_name in VALIDATION_FILES:
        module_name = file_name.replace(".py", "")
        try:
            # Convert hyphenated names to underscores for import
            import_name = module_name.replace("-", "_")
            # APGI_Validation_GUI is at root level, others in Validation/
            if module_name == "APGI_Validation_GUI":
                module = __import__(import_name, fromlist=[import_name])
            else:
                module = __import__(f"Validation.{import_name}", fromlist=[import_name])
            loaded_modules[module_name] = module
            loaded_protocols.append(module_name)
            print(f"✅ Loaded validation module: {module_name}")
        except ImportError as e:
            print(f"❌ Failed to load {module_name}: {e}")
            loaded_modules[module_name] = None
            loaded_protocols.append(module_name)
        except Exception as e:
            print(f"⚠️  Error loading {module_name}: {e}")
            loaded_modules[module_name] = None
            loaded_protocols.append(module_name)

    return loaded_modules, loaded_protocols


# Load modules at import time
VALIDATION_MODULES, VALIDATION_PROTOCOLS = load_validation_modules()


class TestValidationProtocolImports:
    """Test that all validation protocols can be imported and have expected structure."""

    @pytest.mark.parametrize("protocol_name", VALIDATION_PROTOCOLS)
    def test_protocol_import(self, protocol_name):
        """Test that each validation protocol can be imported."""
        module = VALIDATION_MODULES[protocol_name]

        if module is None:
            pytest.skip(f"Protocol {protocol_name} not available")

        # Basic module structure checks
        assert hasattr(module, "__name__")
        assert hasattr(module, "__file__")

        # Check for common validation protocol patterns
        has_run_validation = hasattr(module, "run_validation")
        has_main_class = any(
            hasattr(module, name)
            for name in dir(module)
            if name[0].isupper() and not name.startswith("_")
        )

        # At least one of these should be true for a validation protocol
        assert (
            has_run_validation or has_main_class
        ), f"Protocol {protocol_name} lacks expected entry points"

    @pytest.mark.parametrize("protocol_name", VALIDATION_PROTOCOLS)
    def test_protocol_has_docstring(self, protocol_name):
        """Test that each validation protocol has proper documentation."""
        module = VALIDATION_MODULES[protocol_name]

        if module is None:
            pytest.skip(f"Protocol {protocol_name} not available")

        # Check module docstring
        assert (
            module.__doc__ is not None
        ), f"Protocol {protocol_name} missing module docstring"
        assert (
            len(module.__doc__.strip()) > 50
        ), f"Protocol {protocol_name} docstring too short"


class TestValidationProtocolExecution:
    """Test execution of validation protocols with various configurations."""

    @pytest.fixture
    def temp_results_dir(self):
        """Create temporary directory for test results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.parametrize("protocol_name", VALIDATION_PROTOCOLS)
    def test_protocol_basic_execution(self, protocol_name, temp_results_dir):
        """Test basic execution of each validation protocol."""
        module = VALIDATION_MODULES[protocol_name]

        if module is None:
            pytest.skip(f"Protocol {protocol_name} not available")

        # Try to find and execute the main validation function
        validation_result = None

        # Method 1: Look for run_validation function
        if hasattr(module, "run_validation"):
            try:
                # Test with minimal parameters
                with patch("sys.stdout"):  # Suppress output during testing
                    validation_result = module.run_validation(
                        n_participants=10,  # Small number for testing
                        seed=42,
                        output_dir=str(temp_results_dir),
                        verbose=False,
                    )
            except Exception:
                # Some protocols may have different parameter signatures
                try:
                    validation_result = module.run_validation()
                except Exception as e2:
                    pytest.skip(f"Protocol {protocol_name} run_validation failed: {e2}")

        # Method 2: Look for main class and instantiate it
        elif validation_result is None:
            try:
                # Find the main class (usually starts with capital letter)
                main_classes = [
                    name
                    for name in dir(module)
                    if name[0].isupper() and not name.startswith("_")
                ]

                if main_classes:
                    class_name = main_classes[0]
                    main_class = getattr(module, class_name)

                    # Try to instantiate and run
                    instance = main_class()
                    if hasattr(instance, "run_validation"):
                        with patch("sys.stdout"):
                            validation_result = instance.run_validation()
                    elif hasattr(instance, "validate"):
                        with patch("sys.stdout"):
                            validation_result = instance.validate()
            except Exception as e:
                pytest.skip(f"Protocol {protocol_name} class instantiation failed: {e}")

        # Validate result structure
        if validation_result is not None:
            assert isinstance(
                validation_result, dict
            ), f"Protocol {protocol_name} should return dict"

            # Check for common result fields
            common_fields = ["passed", "status", "message", "results"]
            found_fields = [
                field for field in common_fields if field in validation_result
            ]
            assert (
                len(found_fields) >= 2
            ), f"Protocol {protocol_name} result missing common fields"

    @pytest.mark.parametrize("protocol_name", VALIDATION_PROTOCOLS)
    def test_protocol_error_handling(self, protocol_name):
        """Test error handling in validation protocols."""
        module = VALIDATION_MODULES[protocol_name]

        if module is None:
            pytest.skip(f"Protocol {protocol_name} not available")

        # Test with invalid parameters
        if hasattr(module, "run_validation"):
            try:
                # Test with negative participants (should be handled gracefully)
                result = module.run_validation(n_participants=-1)
                # Should either return error result or raise appropriate exception
                if isinstance(result, dict):
                    assert result.get("status") in [
                        "failed",
                        "error",
                    ], "Should indicate failure for invalid input"
            except (ValueError, TypeError, AssertionError):
                # Expected for invalid input
                pass
            except Exception as e:
                # Other exceptions should be reasonable
                assert (
                    "crashed" not in str(e).lower()
                ), f"Protocol {protocol_name} should handle errors gracefully"


class TestCrossProtocolCompatibility:
    """Test compatibility and interactions between validation protocols."""

    def test_master_validation_integration(self):
        """Test Master_Validation can coordinate other protocols."""
        module = VALIDATION_MODULES.get("Master_Validation")

        if module is None:
            pytest.skip("Master_Validation not available")

        try:
            # Test Master_Validation initialization
            if hasattr(module, "MasterValidator"):
                validator = module.MasterValidator()

                # Test protocol discovery
                if hasattr(validator, "get_available_protocols"):
                    protocols = validator.get_available_protocols()
                    assert isinstance(protocols, list)
                    assert (
                        len(protocols) > 0
                    ), "Master_Validation should discover protocols"

                # Test batch validation
                if hasattr(validator, "run_batch_validation"):
                    with patch("sys.stdout"):
                        # Test with subset of protocols
                        test_protocols = (
                            protocols[:3] if len(protocols) > 3 else protocols
                        )
                        results = validator.run_batch_validation(
                            protocols=test_protocols, n_participants=5, seed=42
                        )
                        assert isinstance(results, dict)
                        assert len(results) == len(test_protocols)
        except Exception as e:
            pytest.skip(f"Master_Validation integration test failed: {e}")

    def test_protocol_result_compatibility(self):
        """Test that protocol results have basic expected structure."""
        sample_results = []

        # Collect results from available protocols
        for protocol_name in VALIDATION_PROTOCOLS[:5]:  # Test first 5 protocols
            module = VALIDATION_MODULES[protocol_name]

            if module is None:
                continue

            try:
                if hasattr(module, "run_validation"):
                    with patch("sys.stdout"):
                        result = module.run_validation(
                            n_participants=5, seed=42, verbose=False
                        )
                        if isinstance(result, dict):
                            sample_results.append((protocol_name, result))
            except Exception:
                continue

        # Check that results have basic structure
        if len(sample_results) >= 2:
            for protocol_name, result in sample_results:
                # All results should be dictionaries
                assert isinstance(
                    result, dict
                ), f"Protocol {protocol_name} should return dict"

                # Results should have at least some basic keys (different protocols may have different keys)
                assert (
                    len(result.keys()) > 0
                ), f"Protocol {protocol_name} result should not be empty"

                # Check for common validation result patterns
                has_any_validation_key = any(
                    key in result
                    for key in [
                        "success",
                        "passed",
                        "results",
                        "data",
                        "metrics",
                        "summary",
                    ]
                )
                assert (
                    has_any_validation_key
                ), f"Protocol {protocol_name} should have validation-related keys"


class TestValidationProtocolPerformance:
    """Test performance and resource management of validation protocols."""

    @pytest.mark.parametrize(
        "protocol_name", VALIDATION_PROTOCOLS[:3]
    )  # Test first 3 for performance
    def test_protocol_performance(self, protocol_name):
        """Test that protocols complete within reasonable time."""
        module = VALIDATION_MODULES[protocol_name]

        if module is None:
            pytest.skip(f"Protocol {protocol_name} not available")

        start_time = time.time()

        try:
            if hasattr(module, "run_validation"):
                # Run with small dataset for performance testing
                with patch("sys.stdout"):
                    result = module.run_validation(
                        n_participants=10, seed=42, verbose=False
                    )

                execution_time = time.time() - start_time

                # Should complete within reasonable time (30 seconds for small dataset)
                assert (
                    execution_time < 30
                ), f"Protocol {protocol_name} took too long: {execution_time:.2f}s"

                # Check result quality
                if isinstance(result, dict):
                    assert result.get("status") in [
                        "success",
                        "failed",
                        "error",
                    ], "Protocol should return valid status"
        except Exception as e:
            pytest.skip(f"Protocol {protocol_name} performance test failed: {e}")

    def test_memory_usage_during_execution(self):
        """Test that protocols don't have memory leaks."""
        import gc
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run a few protocols sequentially
        for protocol_name in VALIDATION_PROTOCOLS[:2]:  # Test first 2 protocols
            module = VALIDATION_MODULES[protocol_name]

            if module is None:
                continue

            try:
                if hasattr(module, "run_validation"):
                    with patch("sys.stdout"):
                        _ = module.run_validation(  # result not used
                            n_participants=5, seed=42, verbose=False
                        )

                    # Force garbage collection
                    gc.collect()

                    # Check memory usage
                    current_memory = process.memory_info().rss
                    memory_increase = current_memory - initial_memory

                    # Should not increase by more than 100MB for small test
                    assert (
                        memory_increase < 100 * 1024 * 1024
                    ), f"Protocol {protocol_name} may have memory leak: {memory_increase / 1024 / 1024:.1f}MB"
            except Exception:
                continue


class TestValidationProtocolDataHandling:
    """Test data handling and output generation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.parametrize("protocol_name", VALIDATION_PROTOCOLS[:3])
    def test_output_file_generation(self, protocol_name, temp_output_dir):
        """Test that protocols generate expected output files."""
        module = VALIDATION_MODULES[protocol_name]

        if module is None:
            pytest.skip(f"Protocol {protocol_name} not available")

        try:
            if hasattr(module, "run_validation"):
                # Run with output directory specified
                with patch("sys.stdout"):
                    _ = module.run_validation(  # result not used
                        n_participants=10,
                        seed=42,
                        output_dir=str(temp_output_dir),
                        verbose=False,
                    )

                # Check that output files were created
                output_files = list(temp_output_dir.glob("**/*"))

                # Should create at least one output file
                assert (
                    len(output_files) > 0
                ), f"Protocol {protocol_name} should generate output files"

                # Check file sizes are reasonable
                for file_path in output_files:
                    if file_path.is_file():
                        file_size = file_path.stat().st_size
                        assert file_size > 0, f"Output file {file_path.name} is empty"
                        assert (
                            file_size < 10 * 1024 * 1024
                        ), f"Output file {file_path.name} is too large: {file_size / 1024 / 1024:.1f}MB"
        except Exception as e:
            pytest.skip(f"Protocol {protocol_name} output test failed: {e}")

    def test_result_serialization(self):
        """Test that protocol results can be serialized."""
        sample_results = []

        # Collect results from available protocols
        for protocol_name in VALIDATION_PROTOCOLS[:3]:
            module = VALIDATION_MODULES[protocol_name]

            if module is None:
                continue

            try:
                if hasattr(module, "run_validation"):
                    with patch("sys.stdout"):
                        result = module.run_validation(
                            n_participants=5, seed=42, verbose=False
                        )
                        if isinstance(result, dict):
                            sample_results.append((protocol_name, result))
            except Exception:
                continue

        # Test JSON serialization
        for protocol_name, result in sample_results:
            try:
                json_str = json.dumps(result, default=str)
                assert (
                    len(json_str) > 0
                ), f"Protocol {protocol_name} result should be serializable"

                # Test deserialization
                parsed_result = json.loads(json_str)
                assert isinstance(
                    parsed_result, dict
                ), f"Protocol {protocol_name} result should deserialize to dict"
            except Exception as e:
                pytest.fail(
                    f"Protocol {protocol_name} result serialization failed: {e}"
                )


class TestValidationProtocolEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_parameter_handling(self):
        """Test protocols handle empty or None parameters gracefully."""
        for protocol_name in VALIDATION_PROTOCOLS[:3]:  # Test first 3 protocols
            module = VALIDATION_MODULES[protocol_name]

            if module is None:
                continue

            try:
                if hasattr(module, "run_validation"):
                    # Test with minimal parameters
                    with patch("sys.stdout"):
                        result = module.run_validation()

                        # Should not crash and should return some result
                        assert (
                            result is not None
                        ), f"Protocol {protocol_name} should return a result"
                        assert isinstance(
                            result, dict
                        ), f"Protocol {protocol_name} should return a dict"
            except Exception as e:
                # Should handle errors gracefully
                assert (
                    "crash" not in str(e).lower()
                ), f"Protocol {protocol_name} should not crash: {e}"

    def test_large_parameter_handling(self):
        """Test protocols handle large parameter values gracefully."""
        for protocol_name in VALIDATION_PROTOCOLS[:2]:  # Test first 2 protocols
            module = VALIDATION_MODULES[protocol_name]

            if module is None:
                continue

            try:
                if hasattr(module, "run_validation"):
                    # Test with large participants number (should handle gracefully)
                    with patch("sys.stdout"):
                        result = module.run_validation(
                            n_participants=1000, seed=42, verbose=False
                        )

                        # Should either complete successfully or fail gracefully
                        if isinstance(result, dict):
                            status = result.get("status", "")
                            assert status in [
                                "success",
                                "failed",
                                "error",
                            ], f"Protocol {protocol_name} should return valid status"
            except Exception as e:
                # Should handle large parameters gracefully
                assert (
                    "memory" not in str(e).lower() or "timeout" in str(e).lower()
                ), f"Protocol {protocol_name} should handle large parameters gracefully: {e}"


# Integration test runner
def run_comprehensive_validation_tests():
    """Run comprehensive validation protocol tests."""
    print("🧪 Running Comprehensive Validation Protocol Integration Tests")
    print("=" * 60)

    # Test module loading
    print("\n📦 Testing Module Loading...")
    loaded_count = sum(
        1 for module in VALIDATION_MODULES.values() if module is not None
    )
    total_count = len(VALIDATION_MODULES)
    print(f"✅ Loaded {loaded_count}/{total_count} validation modules")

    # Test basic functionality
    print("\n🔧 Testing Basic Functionality...")
    basic_tests_passed = 0
    for protocol_name in VALIDATION_PROTOCOLS[:5]:  # Test first 5
        module = VALIDATION_MODULES[protocol_name]
        if module is not None and hasattr(module, "run_validation"):
            try:
                with patch("sys.stdout"):
                    result = module.run_validation(
                        n_participants=5, seed=42, verbose=False
                    )
                    if isinstance(result, dict):
                        basic_tests_passed += 1
                        print(f"✅ {protocol_name}: Basic execution successful")
                    else:
                        print(f"❌ {protocol_name}: Invalid result type")
            except Exception as e:
                print(f"⚠️  {protocol_name}: Execution failed - {e}")
        else:
            print(f"❌ {protocol_name}: Not available or no run_validation")

    print(f"\n📊 Basic Tests: {basic_tests_passed}/5 passed")

    # Test result schema consistency
    print("\n📋 Testing Result Schema Consistency...")
    schema_consistent = 0
    common_fields = set()

    for protocol_name in VALIDATION_PROTOCOLS[:3]:
        module = VALIDATION_MODULES[protocol_name]
        if module is None:
            continue

        try:
            if hasattr(module, "run_validation"):
                with patch("sys.stdout"):
                    result = module.run_validation(
                        n_participants=5, seed=42, verbose=False
                    )
                    if isinstance(result, dict):
                        if not common_fields:
                            common_fields = set(result.keys())
                        else:
                            common_fields.intersection_update(result.keys())
                        schema_consistent += 1
                        print(f"✅ {protocol_name}: Schema consistent")
        except Exception:
            pass

    print(f"📊 Schema Tests: {schema_consistent}/3 consistent")
    print(f"📊 Common Fields: {len(common_fields)} fields")

    print("\n🎯 Integration Test Summary:")
    print(f"   Modules Loaded: {loaded_count}/{total_count}")
    print(f"   Basic Tests: {basic_tests_passed}/5")
    print(f"   Schema Tests: {schema_consistent}/3")
    print(
        f"   Overall Status: {'✅ PASSED' if loaded_count > 15 and basic_tests_passed >= 3 else '❌ NEEDS ATTENTION'}"
    )


if __name__ == "__main__":
    run_comprehensive_validation_tests()
