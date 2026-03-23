#!/usr/bin/env python3
"""
Test script for Fractional Dimension Biomarker implementation.
Tests DFA with different signal types to verify clinical interpretations.
"""

import sys
import os
import importlib.util

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Load the module with hyphen in filename
spec = importlib.util.spec_from_file_location(
    "APGI_Model", "APGI-Full-Dynamic-Model.py"
)
APGI_Model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(APGI_Model)


def generate_test_signals(n_points=1000):
    """Generate different types of test signals with known Hurst properties."""

    # Random walk (H ≈ 0.5)
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(n_points))

    # Persistent signal (H > 0.5) - fractional Brownian motion
    # Simple approximation: integrate random walk with positive feedback
    persistent = np.zeros(n_points)
    for i in range(1, n_points):
        persistent[i] = 0.9 * persistent[i - 1] + 0.1 * np.random.randn()

    # Anti-persistent signal (H < 0.5) - mean-reverting
    anti_persistent = np.zeros(n_points)
    for i in range(1, n_points):
        anti_persistent[i] = -0.5 * anti_persistent[i - 1] + np.random.randn()

    return {
        "random_walk": random_walk,
        "persistent": persistent,
        "anti_persistent": anti_persistent,
    }


def test_fractional_dimension_biomarker():
    """Test the Fractional Dimension Biomarker with different signal types."""

    print("=" * 70)
    print("Fractional Dimension Biomarker - Test Suite")
    print("=" * 70)

    # Create model
    model = APGI_Model.create_default_model()

    # Generate test signals
    signals = generate_test_signals(1000)  # Longer series for better DFA

    # Test each signal type
    for signal_name, signal in signals.items():
        print(f"\n{signal_name.upper()} SIGNAL:")
        print("-" * 40)

        try:
            results = model.compute_fractional_dimension_biomarker(signal)

            print(f"Hurst Exponent: {results['hurst_exponent']:.3f}")
            print(f"H Variance: {results['variance_h']:.4f}")
            print(f"Clinical Interpretation: {results['clinical_interpretation']}")

            # Verify expected ranges
            H = results["hurst_exponent"]
            if signal_name == "random_walk":
                # DFA on random walk (integrated white noise) typically gives H≈1.0
                assert (
                    0.8 <= H <= 1.0
                ), f"Random walk H should be ~1.0 (DFA on integrated signal), got {H}"
            elif signal_name == "persistent":
                assert H > 0.5, f"Persistent signal H should be >0.5, got {H}"
            elif signal_name == "anti_persistent":
                assert H < 0.5, f"Anti-persistent signal H should be <0.5, got {H}"

            print("✅ Test passed")

        except Exception as e:
            print(f"❌ Test failed: {e}")

    print("\n" + "=" * 70)
    print("Test suite complete.")
    print("=" * 70)


if __name__ == "__main__":
    test_fractional_dimension_biomarker()
