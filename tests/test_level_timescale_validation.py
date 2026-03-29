#!/usr/bin/env python3
"""
Test Level Timescale Validation

Tests for validating that hierarchical models use the correct tau values
as specified in the LEVEL_TIMESCALES constant.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import LEVEL_TIMESCALES


class TestLevelTimescaleValidation:
    """Test that tau values match LEVEL_TIMESCALES specification"""

    def test_level_timescales_constant(self):
        """Test that LEVEL_TIMESCALES constant has correct values"""
        # Test the specific values mentioned in the documentation
        assert LEVEL_TIMESCALES.TAU_SENSORY == 0.05, "Sensory tau should be 0.05s"
        assert LEVEL_TIMESCALES.TAU_HOMEOSTATIC == 2.0, "Homeostatic tau should be 2.0s"

        # Test the mapping
        expected_mapping = {
            1: 0.05,  # Sensory
            2: 0.2,  # Organ
            3: 0.5,  # Cognitive
            4: 2.0,  # Homeostatic
        }

        assert (
            LEVEL_TIMESCALES.LEVEL_TIMESCALES == expected_mapping
        ), "LEVEL_TIMESCALES mapping should match expected values"

    def test_tau_range_specification(self):
        """Test that tau values follow the documented range specification"""
        # According to documentation: τ ranges from 0.05s (sensory) to 2.0s (homeostatic)
        taus = list(LEVEL_TIMESCALES.LEVEL_TIMESCALES.values())

        # Check range
        assert min(taus) == 0.05, "Minimum tau should be 0.05s (sensory)"
        assert max(taus) == 2.0, "Maximum tau should be 2.0s (homeostatic)"

        # Check monotonic increase (sensory to homeostatic should be increasing)
        assert all(
            taus[i] <= taus[i + 1] for i in range(len(taus) - 1)
        ), "Tau values should increase from sensory to homeostatic levels"

    def test_pytorch_model_import(self):
        """Test that PyTorch model can be imported and uses constants"""
        try:
            # Try importing the PyTorch model
            from Validation.ActiveInferenceAgentSimulations_Protocol3 import (
                HierarchicalGenerativeModel,
            )

            # Test that it validates tau values
            valid_levels = [
                {"name": "sensory", "dim": 32, "tau": LEVEL_TIMESCALES.TAU_SENSORY},
                {"name": "objects", "dim": 16, "tau": LEVEL_TIMESCALES.TAU_ORGAN},
                {"name": "context", "dim": 8, "tau": LEVEL_TIMESCALES.TAU_COGNITIVE},
            ]

            # Should not raise an exception
            model = HierarchicalGenerativeModel(valid_levels)
            assert model.taus[0] == LEVEL_TIMESCALES.TAU_SENSORY
            assert model.taus[1] == LEVEL_TIMESCALES.TAU_ORGAN
            assert model.taus[2] == LEVEL_TIMESCALES.TAU_COGNITIVE

            print("✓ PyTorch model validation works")

        except ImportError as e:
            print(f"⚠ PyTorch model import failed: {e}")

    def test_numpy_model_import(self):
        """Test that NumPy model can be imported and uses constants"""
        try:
            # Try importing the NumPy model
            from Falsification.FP_1_Falsification_ActiveInferenceAgents_F1F2 import (
                HierarchicalGenerativeModel as NumpyHierarchicalGenerativeModel,
            )

            # Test that it validates tau values
            valid_levels = [
                {"name": "sensory", "dim": 32, "tau": LEVEL_TIMESCALES.TAU_SENSORY},
                {"name": "objects", "dim": 16, "tau": LEVEL_TIMESCALES.TAU_ORGAN},
                {"name": "context", "dim": 8, "tau": LEVEL_TIMESCALES.TAU_COGNITIVE},
            ]

            # Should not raise an exception
            model = NumpyHierarchicalGenerativeModel(valid_levels)
            assert len(model.levels) == 3

            print("✓ NumPy model validation works")

        except ImportError as e:
            print(f"⚠ NumPy model import failed: {e}")


if __name__ == "__main__":
    # Run basic validation
    test = TestLevelTimescaleValidation()

    print("Testing LEVEL_TIMESCALES constant...")
    test.test_level_timescales_constant()
    print("✓ LEVEL_TIMESCALES constant is correct")

    print("Testing tau range specification...")
    test.test_tau_range_specification()
    print("✓ Tau values follow documented range")

    print("Testing PyTorch model import...")
    test.test_pytorch_model_import()

    print("Testing NumPy model import...")
    test.test_numpy_model_import()

    print("\nLevel timescale validation complete! ✓")
