#!/usr/bin/env python3
"""
APGI Theory Framework Test Suite
=================================

Comprehensive testing framework for the APGI Theory Framework.
Tests all major components including:
- Formal model simulations
- Multimodal integration
- Parameter estimation
- Validation protocols
- Configuration management
- Logging system
- GUI components integration testing
"""

import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import APGI framework components
try:
    from config_manager import config_manager, get_config, set_parameter
    from logging_config import apgi_logger

    print("✓ Successfully imported core components")
except ImportError as e:
    print(f"✗ Failed to import core components: {e}")
    sys.exit(1)


class TestConfiguration(unittest.TestCase):
    """Test configuration management system."""

    def setUp(self):
        """Set up test configuration."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = Path(self.temp_dir) / "test_config.yaml"

    def tearDown(self):
        """Clean up test configuration."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_loading(self):
        """Test configuration loading."""
        try:
            config = get_config()
            self.assertIsNotNone(config)
            print("✓ Configuration loading test passed")
        except Exception as e:
            self.fail(f"Configuration loading failed: {e}")

    def test_parameter_setting(self):
        """Test parameter setting functionality."""
        try:
            # Test setting a valid parameter (convert to int)
            set_parameter("simulation", "default_steps", 500)

            # Verify the parameter was set by checking the config
            config = get_config()
            self.assertEqual(config.simulation.default_steps, 500)
            print("✓ Parameter setting test passed")
        except Exception as e:
            self.fail(f"Parameter setting failed: {e}")


class TestLogging(unittest.TestCase):
    """Test logging system."""

    def test_logger_initialization(self):
        """Test logger initialization."""
        try:
            self.assertIsNotNone(apgi_logger)
            self.assertTrue(hasattr(apgi_logger, "logger"))
            print("✓ Logger initialization test passed")
        except Exception as e:
            self.fail(f"Logger initialization failed: {e}")

    def test_log_functions(self):
        """Test logging functions."""
        try:
            # Test different log levels
            apgi_logger.logger.info("Test info message")
            apgi_logger.logger.warning("Test warning message")
            apgi_logger.logger.error("Test error message")
            print("✓ Log functions test passed")
        except Exception as e:
            self.fail(f"Log functions failed: {e}")


class TestModuleLoading(unittest.TestCase):
    """Test module loading functionality."""

    def test_formal_model_loading(self):
        """Test formal model module loading."""
        try:
            formal_model_path = PROJECT_ROOT / "APGI-Formal-Model.py"
            if formal_model_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "formal_model", formal_model_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertTrue(hasattr(module, "SurpriseIgnitionSystem"))
                print("✓ Formal model loading test passed")
            else:
                self.skipTest("Formal model file not found")
        except Exception as e:
            self.fail(f"Formal model loading failed: {e}")

    def test_multimodal_loading(self):
        """Test multimodal integration module loading."""
        try:
            multimodal_path = PROJECT_ROOT / "APGI-Multimodal-Integration.py"
            if multimodal_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "multimodal", multimodal_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertTrue(hasattr(module, "APGINormalizer"))
                print("✓ Multimodal integration loading test passed")
            else:
                self.skipTest("Multimodal integration file not found")
        except Exception as e:
            self.fail(f"Multimodal integration loading failed: {e}")

    def test_parameter_estimation_loading(self):
        """Test parameter estimation module loading."""
        try:
            param_est_path = PROJECT_ROOT / "APGI-Parameter-Estimation-Protocol.py"
            if param_est_path.exists():
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "parameter_estimation", param_est_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertTrue(hasattr(module, "NeuralSignalGenerator"))
                print("✓ Parameter estimation loading test passed")
            else:
                self.skipTest("Parameter estimation file not found")
        except Exception as e:
            self.fail(f"Parameter estimation loading failed: {e}")


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of components."""

    def test_numpy_functionality(self):
        """Test basic numpy functionality."""
        try:
            # Test basic array operations
            arr = np.array([1, 2, 3, 4, 5])
            self.assertEqual(len(arr), 5)
            self.assertEqual(arr.mean(), 3.0)
            print("✓ NumPy functionality test passed")
        except Exception as e:
            self.fail(f"NumPy functionality failed: {e}")

    def test_pandas_functionality(self):
        """Test basic pandas functionality."""
        try:
            # Test basic DataFrame operations
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            self.assertEqual(len(df), 3)
            self.assertEqual(list(df.columns), ["A", "B"])
            print("✓ Pandas functionality test passed")
        except Exception as e:
            self.fail(f"Pandas functionality failed: {e}")

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        try:
            # Generate synthetic data similar to what modules use
            n_samples = 100
            data = pd.DataFrame(
                {
                    "signal": np.random.normal(0, 1, n_samples),
                    "time": np.linspace(0, 1, n_samples),
                }
            )
            self.assertEqual(len(data), n_samples)
            self.assertTrue("signal" in data.columns)
            self.assertTrue("time" in data.columns)
            print("✓ Synthetic data generation test passed")
        except Exception as e:
            self.fail(f"Synthetic data generation failed: {e}")


class TestDependencies(unittest.TestCase):
    """Test that all required dependencies are available."""

    def test_scientific_computing(self):
        """Test scientific computing dependencies."""
        try:
            import numpy
            import scipy
            import pandas

            print(f"✓ NumPy version: {numpy.__version__}")
            print(f"✓ SciPy version: {scipy.__version__}")
            print(f"✓ Pandas version: {pandas.__version__}")
        except ImportError as e:
            self.fail(f"Scientific computing dependency missing: {e}")

    def test_machine_learning(self):
        """Test machine learning dependencies."""
        try:
            import sklearn

            print(f"✓ Scikit-learn version: {sklearn.__version__}")
        except ImportError as e:
            self.fail(f"Machine learning dependency missing: {e}")

    def test_bayesian_modeling(self):
        """Test Bayesian modeling dependencies."""
        try:
            import pymc
            import arviz

            print(f"✓ PyMC version: {pymc.__version__}")
            print(f"✓ ArviZ version: {arviz.__version__}")
        except ImportError as e:
            self.fail(f"Bayesian modeling dependency missing: {e}")

    def test_visualization(self):
        """Test visualization dependencies."""
        try:
            import matplotlib
            import seaborn

            print(f"✓ Matplotlib version: {matplotlib.__version__}")
            print(f"✓ Seaborn version: {seaborn.__version__}")
        except ImportError as e:
            self.fail(f"Visualization dependency missing: {e}")

    def test_cli_dependencies(self):
        """Test CLI dependencies."""
        try:
            import click
            import rich

            print(f"✓ Click version: {click.__version__}")
            # Rich doesn't have __version__ in all versions, handle gracefully
            try:
                print(f"✓ Rich version: {rich.__version__}")
            except AttributeError:
                print("✓ Rich version: available")
        except ImportError as e:
            self.fail(f"CLI dependency missing: {e}")


def run_integration_tests():
    """Run integration tests for the complete framework."""
    print("\n" + "=" * 60)
    print("Running Integration Tests")
    print("=" * 60)

    try:
        # Test CLI functionality
        print("Testing CLI functionality...")
        import subprocess

        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            print("✓ CLI help command works")
        else:
            print("✗ CLI help command failed")
            return False

        # Test configuration
        print("Testing configuration system...")
        try:
            config = get_config()
            print("✓ Configuration system works")
        except Exception as e:
            print(f"✗ Configuration system failed: {e}")
            return False

        # Test logging
        print("Testing logging system...")
        try:
            apgi_logger.logger.info("Integration test log message")
            print("✓ Logging system works")
        except Exception as e:
            print(f"✗ Logging system failed: {e}")
            return False

        print("✓ All integration tests passed")
        return True

    except Exception as e:
        print(f"✗ Integration tests failed: {e}")
        return False


class TestGUIIntegration(unittest.TestCase):
    """Test GUI components integration."""

    def setUp(self):
        """Set up GUI test environment."""
        self.temp_dir = tempfile.mkdtemp()
        # Set environment variable to prevent actual GUI display during tests
        os.environ["DISPLAY"] = ":99"  # Virtual display

    def tearDown(self):
        """Clean up GUI test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validation_gui_import(self):
        """Test that Validation GUI can be imported without errors."""
        try:
            # Test import without actually launching GUI
            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import sys
import importlib.util
sys.path.insert(0, '.')
try:
    spec = importlib.util.spec_from_file_location('validation_gui', 'Validation/APGI-Validation-GUI.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print('SUCCESS: Validation GUI imported successfully')
except ImportError as e:
    print(f'FAILED: Could not import Validation GUI: {e}')
    sys.exit(1)
except Exception as e:
    print(f'FAILED: Error importing Validation GUI: {e}')
    sys.exit(1)
                """,
                ],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )

            self.assertEqual(result.returncode, 0, "Validation GUI import failed")
            self.assertIn("SUCCESS", result.stdout)

        except Exception as e:
            self.fail(f"Validation GUI import test failed: {e}")

    def test_psychological_gui_import(self):
        """Test that Psychological States GUI can be imported without errors."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import sys
import importlib.util
sys.path.insert(0, '.')
try:
    spec = importlib.util.spec_from_file_location('psychological_gui', 'APGI-Psychological-States-GUI.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print('SUCCESS: Psychological States GUI imported successfully')
except ImportError as e:
    print(f'FAILED: Could not import Psychological States GUI: {e}')
    sys.exit(1)
except Exception as e:
    print(f'FAILED: Error importing Psychological States GUI: {e}')
    sys.exit(1)
                """,
                ],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )

            self.assertEqual(
                result.returncode, 0, "Psychological States GUI import failed"
            )
            self.assertIn("SUCCESS", result.stdout)

        except Exception as e:
            self.fail(f"Psychological States GUI import test failed: {e}")

    def test_gui_dependencies(self):
        """Test that GUI dependencies are available."""
        required_modules = ["tkinter", "matplotlib", "numpy"]

        for module_name in required_modules:
            try:
                if module_name == "tkinter":
                    import tkinter
                elif module_name == "matplotlib":
                    import matplotlib
                elif module_name == "numpy":
                    import numpy
            except ImportError:
                self.fail(f"Required GUI dependency '{module_name}' not available")

    def test_gui_file_paths(self):
        """Test that GUI files exist and are accessible."""
        gui_files = [
            PROJECT_ROOT / "Validation" / "APGI-Validation-GUI.py",
            PROJECT_ROOT / "APGI-Psychological-States-GUI.py",
        ]

        for gui_file in gui_files:
            self.assertTrue(gui_file.exists(), f"GUI file not found: {gui_file}")
            self.assertGreater(
                gui_file.stat().st_size, 0, f"GUI file is empty: {gui_file}"
            )


def main():
    """Main test runner."""
    print("APGI Theory Framework Test Suite")
    print("=" * 60)

    # Run unit tests
    print("\nRunning Unit Tests...")
    print("-" * 40)

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestConfiguration,
        TestLogging,
        TestModuleLoading,
        TestBasicFunctionality,
        TestDependencies,
        TestGUIIntegration,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 60)
    print("Unit Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    # Run integration tests
    integration_success = run_integration_tests()

    # Final result
    print("\n" + "=" * 60)
    print("Final Test Result")
    print("=" * 60)

    unit_test_success = len(result.failures) == 0 and len(result.errors) == 0

    if unit_test_success and integration_success:
        print("✓ ALL TESTS PASSED")
        print("The APGI Theory Framework is ready to use!")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the errors above before using the framework.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
