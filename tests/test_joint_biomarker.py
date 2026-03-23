#!/usr/bin/env python3
"""
Test script for Joint HEP + PCI Biomarker validation.
Tests the ΔR² > 0.05 criterion with different scenarios.
"""

import sys
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the joint biomarker validation function
import importlib.util

spec = importlib.util.spec_from_file_location(
    "multimodal_module", "APGI-Multimodal-Integration.py"
)
multimodal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multimodal_module)


def test_joint_biomarker_scenarios():
    """Test joint biomarker validation with different scenarios."""

    print("=" * 70)
    print("JOINT HEP + PCI BIOMARKER - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    # Test 1: Data with clear joint advantage (should PASS)
    print("\n📊 Test 1: Clear Joint Advantage (Expected: PASS)")
    print("-" * 50)

    (
        HEP_feat,
        PCI_feat,
        joint_feat,
        target,
    ) = multimodal_module.create_joint_biomarker_test_data(
        n_samples=100, effect_size=0.5, noise_level=0.3, random_seed=42
    )

    results1 = multimodal_module.validate_joint_biomarker_advantage(
        HEP_features=HEP_feat,
        PCI_features=PCI_feat,
        joint_features=joint_feat,
        target=target,
        delta_r2_threshold=0.05,
    )

    print(f"Result: {'PASS' if results1['passed'] else 'FAIL'}")
    print(f"ΔR²: {results1['delta_r2']:.4f}")

    # Test 2: Data with minimal joint advantage (should FAIL)
    print("\n📊 Test 2: Minimal Joint Advantage (Expected: FAIL)")
    print("-" * 50)

    (
        HEP_feat2,
        PCI_feat2,
        joint_feat2,
        target2,
    ) = multimodal_module.create_joint_biomarker_test_data(
        n_samples=100, effect_size=0.05, noise_level=0.8, random_seed=123
    )

    results2 = multimodal_module.validate_joint_biomarker_advantage(
        HEP_features=HEP_feat2,
        PCI_features=PCI_feat2,
        joint_features=joint_feat2,
        target=target2,
        delta_r2_threshold=0.05,
    )

    print(f"Result: {'PASS' if results2['passed'] else 'FAIL'}")
    print(f"ΔR²: {results2['delta_r2']:.4f}")

    # Test 3: Edge case - very small sample (should handle gracefully)
    print("\n📊 Test 3: Small Sample Size (Edge Case)")
    print("-" * 50)

    try:
        (
            HEP_feat3,
            PCI_feat3,
            joint_feat3,
            target3,
        ) = multimodal_module.create_joint_biomarker_test_data(
            n_samples=12, effect_size=0.3, noise_level=0.4, random_seed=456
        )

        results3 = multimodal_module.validate_joint_biomarker_advantage(
            HEP_features=HEP_feat3,
            PCI_features=PCI_feat3,
            joint_features=joint_feat3,
            target=target3,
            delta_r2_threshold=0.05,
        )

        print(f"Result: {'PASS' if results3['passed'] else 'FAIL'}")
        print(f"ΔR²: {results3['delta_r2']:.4f}")

    except Exception as e:
        print(f"Expected error for small sample: {e}")

    # Test 4: Different threshold values
    print("\n📊 Test 4: Different Threshold Values")
    print("-" * 50)

    (
        HEP_feat4,
        PCI_feat4,
        joint_feat4,
        target4,
    ) = multimodal_module.create_joint_biomarker_test_data(
        n_samples=80, effect_size=0.3, noise_level=0.4, random_seed=789
    )

    thresholds = [0.01, 0.05, 0.10, 0.20]
    for threshold in thresholds:
        results4 = multimodal_module.validate_joint_biomarker_advantage(
            HEP_features=HEP_feat4,
            PCI_features=PCI_feat4,
            joint_features=joint_feat4,
            target=target4,
            delta_r2_threshold=threshold,
        )
        print(
            f"Threshold {threshold:.2f}: {'PASS' if results4['passed'] else 'FAIL'} (ΔR² = {results4['delta_r2']:.4f})"
        )

    # Test 5: Verify ΔR² calculation manually
    print("\n📊 Test 5: Manual ΔR² Verification")
    print("-" * 50)

    # Use same data as Test 1
    # Manual calculation
    hep_model = LinearRegression()
    pci_model = LinearRegression()
    joint_model = LinearRegression()

    hep_model.fit(HEP_feat, target)
    pci_model.fit(PCI_feat, target)
    joint_model.fit(joint_feat, target)

    r2_hep_manual = r2_score(target, hep_model.predict(HEP_feat))
    r2_pci_manual = r2_score(target, pci_model.predict(PCI_feat))
    r2_joint_manual = r2_score(target, joint_model.predict(joint_feat))

    delta_r2_manual = r2_joint_manual - max(r2_hep_manual, r2_pci_manual)

    print("Manual calculation:")
    print(f"  HEP R²: {r2_hep_manual:.4f}")
    print(f"  PCI R²: {r2_pci_manual:.4f}")
    print(f"  Joint R²: {r2_joint_manual:.4f}")
    print(f"  ΔR²: {delta_r2_manual:.4f}")

    print(f"Function result ΔR²: {results1['delta_r2']:.4f}")
    print(
        f"Match: {'YES' if abs(delta_r2_manual - results1['delta_r2']) < 1e-6 else 'NO'}"
    )

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)

    tests = [
        ("Clear advantage", results1["passed"]),
        ("Minimal advantage", not results2["passed"]),
        ("Manual verification", abs(delta_r2_manual - results1["delta_r2"]) < 1e-6),
    ]

    passed_tests = sum(1 for _, result in tests if result)
    total_tests = len(tests)

    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 All tests passed! Joint biomarker validation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the implementation.")


if __name__ == "__main__":
    test_joint_biomarker_scenarios()
