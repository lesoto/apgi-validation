#!/usr/bin/env python3
"""
LEVEL DESIGNATION: Level 2 (information-theoretic)

Bridge to Level 1
"""

import sys
import importlib.util
from pathlib import Path
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def test_protocol_import(protocol_num: int) -> Tuple[bool, str]:
    """Test if a protocol can be imported without errors."""
    try:
        matches = list(project_root.glob(f"Falsification/FP_{protocol_num:02d}_*.py"))

        if not matches:
            return False, f"Protocol {protocol_num} file not found"

        protocol_path = matches[0]
        spec = importlib.util.spec_from_file_location(f"protocol_{protocol_num}", protocol_path)

        if spec is None or spec.loader is None:
            return False, "Failed to create module spec"

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return True, f"✓ Imported successfully from {protocol_path.name}"

    except Exception as e:
        return False, f"Import error: {str(e)[:100]}"


def check_pac_variance_fix() -> Tuple[bool, str]:
    """Check if FP_02 PAC variance has been increased."""
    try:
        fp02_path = project_root / "Falsification" / "FP_02_AgentComparison_ConvergenceBenchmark.py"
        with open(fp02_path, "r") as f:
            content = f.read()

        # Check for the fixed variance values
        checks = [
            ("0.100" in content, "PAC baseline std increased to 0.100"),
            ("0.025" in content, "Individual noise std increased to 0.025"),
            ("0.20" in content, "PAC baseline clipping range increased to 0.20"),
            ("warning_std_threshold=0.05" in content, "Warning threshold updated to 0.05"),
        ]

        all_passed = all(check[0] for check in checks)
        details = "\n  ".join([f"{'✓' if check[0] else '✗'} {check[1]}" for check in checks])

        return all_passed, details

    except Exception as e:
        return False, f"Error checking PAC variance: {str(e)}"


def check_fp07_imports() -> Tuple[bool, str]:
    """Check if FP_07 imports have been fixed."""
    try:
        fp07_path = project_root / "Falsification" / "FP_07_MathematicalConsistency.py"
        with open(fp07_path, "r") as f:
            content = f.read()

        # Check for incorrect imports
        bad_imports = [
            "from apgi_validation.utils",
            "from .utils.constants",
        ]

        has_bad_imports = any(bad_import in content for bad_import in bad_imports)

        # Check for correct imports
        good_imports = [
            "from utils.constants import VISUAL_CONSTANTS",
            "from utils.protocol_schema import",
            "from utils.analytical_solutions import",
        ]

        has_good_imports = all(good_import in content for good_import in good_imports)

        if has_bad_imports:
            return False, "Found incorrect import paths (apgi_validation or relative imports)"

        if not has_good_imports:
            return False, "Missing expected correct imports"

        return True, "✓ All imports use correct absolute paths"

    except Exception as e:
        return False, f"Error checking FP_07 imports: {str(e)}"


def main():
    """Run all validation tests."""
    print(f"\n{BLUE}{'=' * 60}")
    print("FALSIFICATION PROTOCOL VALIDATION TEST")
    print(f"{'=' * 60}{RESET}\n")

    # Test 1: Check all 12 protocols can be imported
    print(f"{BLUE}Test 1: Protocol Import Validation{RESET}")
    print("-" * 60)

    import_results = {}
    for protocol_num in range(1, 13):
        success, message = test_protocol_import(protocol_num)
        import_results[protocol_num] = success
        status = f"{GREEN}✓{RESET}" if success else f"{RED}✗{RESET}"
        print(f"  FP_{protocol_num:02d}: {status} {message}")

    import_success = all(import_results.values())
    print(f"\n  Result: {GREEN if import_success else RED}{'PASS' if import_success else 'FAIL'}{RESET}")

    # Test 2: Check PAC variance fix in FP_02
    print(f"\n{BLUE}Test 2: PAC Variance Fix (FP_02){RESET}")
    print("-" * 60)

    pac_success, pac_details = check_pac_variance_fix()
    print(f"  {pac_details}")
    print(f"\n  Result: {GREEN if pac_success else RED}{'PASS' if pac_success else 'FAIL'}{RESET}")

    # Test 3: Check FP_07 import fixes
    print(f"\n{BLUE}Test 3: Import Path Fixes (FP_07){RESET}")
    print("-" * 60)

    fp07_success, fp07_details = check_fp07_imports()
    print(f"  {fp07_details}")
    print(f"\n  Result: {GREEN if fp07_success else RED}{'PASS' if fp07_success else 'FAIL'}{RESET}")

    # Summary
    print(f"\n{BLUE}{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}{RESET}")

    all_tests_passed = import_success and pac_success and fp07_success

    print(f"  Protocol Imports:     {GREEN if import_success else RED}{'PASS' if import_success else 'FAIL'}{RESET}")
    print(f"  PAC Variance Fix:     {GREEN if pac_success else RED}{'PASS' if pac_success else 'FAIL'}{RESET}")
    print(f"  FP_07 Import Fixes:   {GREEN if fp07_success else RED}{'PASS' if fp07_success else 'FAIL'}{RESET}")

    print(
        f"\n  Overall: {GREEN if all_tests_passed else RED}{'ALL TESTS PASSED ✓' if all_tests_passed else 'SOME TESTS FAILED ✗'}{RESET}\n"
    )

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())
