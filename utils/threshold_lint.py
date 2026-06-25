#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import ast
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# Core thresholds that should be defined in the canonical file
# This is a subset of all defined thresholds - the most important ones
EXPECTED_THRESHOLDS = {
    # F1 family - Core thresholds
    "F1_5_PAC_MI_MIN",
    "F1_5_PERMUTATION_ALPHA",
    "F1_6_MIN_LOW_AROUSAL_SLOPE",
    # F2 family - Core thresholds
    "F2_1_MIN_ADVANTAGE_PCT",
    "F2_1_MIN_PP_DIFF",
    "F2_1_MIN_COHENS_H",
    "F2_1_ALPHA",
    "F2_2_MIN_CORR",
    "F2_2_MIN_FISHER_Z",
    "F2_2_ALPHA",
    "F2_3_MIN_RT_ADVANTAGE_MS",
    "F2_3_MIN_BETA",
    "F2_3_ALPHA",
    "F2_4_MIN_CONFIDENCE_EFFECT_PCT",
    "F2_4_MIN_BETA_INTERACTION",
    "F2_4_ALPHA",
    "F2_5_MAX_TRIALS",
    "F2_5_MIN_HAZARD_RATIO",
    "F2_5_ALPHA",
    # F3 family - Core thresholds
    "F3_1_MIN_ADVANTAGE_PCT",
    "F3_1_MIN_COHENS_D",
    "F3_1_ALPHA",
    "F3_2_MIN_COHENS_D",
    "F3_2_ALPHA",
    "F3_3_MIN_REDUCTION_PCT",
    "F3_3_MIN_COHENS_D",
    "F3_3_ALPHA",
    "F3_4_MIN_REDUCTION_PCT",
    "F3_4_MIN_COHENS_D",
    "F3_4_ALPHA",
    # F5 family - Core thresholds
    "F5_1_MIN_PROPORTION",
    "F5_1_MIN_COHENS_D",
    "F5_1_BINOMIAL_ALPHA",
    "F5_2_MIN_CORRELATION",
    "F5_2_MIN_PROPORTION",
    "F5_2_BINOMIAL_ALPHA",
    "F5_3_MIN_COHENS_D",
    "F5_3_BINOMIAL_ALPHA",
    "F5_4_MIN_PROPORTION",
    "F5_4_MIN_PEAK_SEPARATION",
    "F5_4_BINOMIAL_ALPHA",
    "F5_5_PCA_MIN_VARIANCE",
    "F5_5_PCA_FALSIFICATION_THRESHOLD",
    "F5_6_MIN_COHENS_D",
    "F5_6_ALPHA",
    # F6 family - Core thresholds
    "F6_DELTA_AUROC_MIN",
    "F6_1_LTCN_MAX_TRANSITION_MS",
    "F6_1_CLIFFS_DELTA_MIN",
    "F6_1_MANN_WHITNEY_ALPHA",
    "F6_2_LTCN_MIN_WINDOW_MS",
    "F6_2_MIN_INTEGRATION_RATIO",
    "F6_2_MIN_CURVE_FIT_R2",
    "F6_2_WILCOXON_ALPHA",
    "F6_5_BIFURCATION_ERROR_MAX",
    "F6_5_HYSTERESIS_MIN",
    "F6_5_HYSTERESIS_MAX",
    "F6_SPARSITY_ACTIVATION_THRESHOLD",
    # V6 family - Core thresholds
    "V6_1_MIN_PROCESSING_RATE",
    "V6_1_MAX_LATENCY_MS",
    "V6_1_ALPHA",
    # V7 family - Core thresholds
    "V7_1_MIN_THRESHOLD_REDUCTION_PCT",
    "V7_1_MIN_COHENS_D",
    "V7_1_ALPHA",
    "V7_2_MIN_COHENS_D",
    "V7_2_ALPHA",
    # V9 family - Core thresholds
    "V9_1_MIN_CORRELATION",
    "V9_3_MIN_CORRELATION",
    # V11 family - Core thresholds
    "V11_MIN_R2",
    "V11_MIN_DELTA_R2",
    "V11_MIN_COHENS_D",
    # V12 family - Core thresholds
    "V12_1_MIN_COHENS_D",
    "V12_1_ALPHA",
    "V12_2_ALPHA",
    # Generic thresholds
    "GENERIC_ALPHA",
    "GENERIC_MIN_COHENS_D",
    "GENERIC_MIN_CORR",
    "GENERIC_MIN_R2",
}

# Pattern for hard-coded threshold values (e.g., 0.05, 50.0, 0.70)
HARD_CODE_PATTERNS = [
    r"\b0\.\d{2,}\b",  # Decimal values like 0.05, 0.70
    r"\b[1-9]\d{1,2}\.0\b",  # Values like 50.0, 200.0
]


def get_thresholds_from_canonical_file() -> Set[str]:
    """Extract threshold constants defined in utils/falsification_thresholds.py."""
    canonical_file = Path("utils/falsification_thresholds.py")
    if not canonical_file.exists():
        print(f"ERROR: Canonical file not found: {canonical_file}")
        return set()

    try:
        content = canonical_file.read_text(encoding="utf-8")
        tree = ast.parse(content)

        defined_thresholds = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):  # Type annotated assignment
                if hasattr(node.target, "id"):
                    defined_thresholds.add(node.target.id)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if hasattr(target, "id"):
                        defined_thresholds.add(target.id)

        return defined_thresholds
    except Exception as e:
        print(f"ERROR: Failed to parse {canonical_file}: {e}")
        return set()


def check_canonical_definitions() -> Tuple[bool, List[str]]:
    """Check that all expected thresholds are defined in canonical file."""
    defined = get_thresholds_from_canonical_file()

    # Only check for thresholds that are actually expected to be defined
    # Filter EXPECTED_THRESHOLDS to only include those that should exist
    expected_filtered = EXPECTED_THRESHOLDS
    missing = expected_filtered - defined

    errors = []
    if missing:
        errors.append("Missing thresholds in utils/falsification_thresholds.py:")
        for threshold in sorted(missing):
            errors.append(f"  - {threshold}")

    # Report extra defined thresholds (not necessarily an error)
    extra = defined - expected_filtered
    if extra:
        errors.append("Note: Additional thresholds found (not in expected list):")
        for threshold in sorted(extra)[:10]:  # Limit to first 10 to avoid spam
            errors.append(f"  + {threshold}")
        if len(extra) > 10:
            errors.append(f"  ... and {len(extra) - 10} more")

    return len(missing) == 0, errors


def check_imports_in_file(file_path: Path) -> Tuple[bool, List[str]]:
    """Check that file uses correct import path."""
    errors = []
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Skip files that can't be decoded as UTF-8
        return True, []

    # Skip this script itself (it contains the pattern as a string literal)
    if "scripts/threshold_lint.py" in str(file_path) or file_path.name == "threshold_lint.py":
        return True, []

    # Check for old import pattern
    if "from falsification_thresholds import" in content:
        errors.append(
            "  Old import: 'from falsification_thresholds import' -> use 'from utils.falsification_thresholds import'"
        )

    # Check for direct import (should use utils. prefix)
    if re.search(r"^from falsification_thresholds import", content, re.MULTILINE):
        errors.append("  Direct import without utils prefix")

    return len(errors) == 0, errors


def find_hardcoded_thresholds(file_path: Path) -> Tuple[bool, List[str]]:
    """Find potentially hard-coded threshold values in file."""
    errors = []
    content = file_path.read_text()

    # Skip the canonical file itself
    if "utils/falsification_thresholds.py" in str(file_path):
        return True, []

    # Look for suspicious patterns (float literals that could be thresholds)
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        # Skip comments and docstrings
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue

        # Look for float literals in comparisons - more specific pattern to avoid false positives
        if re.search(r"[=<>!]+\s*\d+\.\d+", line):
            # Exclude common patterns that are not thresholds
            if any(
                skip in line
                for skip in [
                    "version",
                    "timestamp",
                    "epoch",
                    "step",
                    "idx",
                    "index",
                    "size",
                    "count",
                    "length",
                    "range(",
                    "enumerate",
                    "len(",
                    "#",
                ]
            ):
                continue
            errors.append(f"  Line {i}: Potential hard-coded value: {line.strip()}")

    return len(errors) == 0, errors


def main() -> int:
    """Run threshold consistency checks."""
    print("=" * 70)
    print("APGI Threshold Consistency Checker")
    print("=" * 70)
    print()

    all_passed = True
    all_errors: List[str] = []

    # Check 1: Canonical definitions
    print("[1/3] Checking canonical threshold definitions...")
    passed, errors = check_canonical_definitions()
    if not passed:
        all_passed = False
        all_errors.extend(errors)
        for error in errors:
            print(f"  FAIL: {error}")
    else:
        print("  OK: All expected thresholds defined in canonical file")
    print()

    # Check 2: Imports across all Python files
    print("[2/3] Checking threshold imports across codebase...")
    import_errors = []
    py_files = list(Path(".").rglob("*.py"))
    for py_file in py_files:
        # Skip hidden directories and __pycache__
        if any(part.startswith(".") or part == "__pycache__" for part in py_file.parts):
            continue

        passed, errors = check_imports_in_file(py_file)
        if not passed:
            all_passed = False
            for error in errors:
                import_errors.append(f"{py_file}: {error}")

    if import_errors:
        all_errors.extend(import_errors)
        for error in import_errors:
            print(f"  FAIL: {error}")
    else:
        print("  OK: All imports use canonical path")
    print()

    # Check 3: Registry completeness
    print("[3/3] Checking THRESHOLD_REGISTRY completeness...")
    try:
        from utils.falsification_thresholds import THRESHOLD_REGISTRY

        registry_keys = set(THRESHOLD_REGISTRY.keys())
        expected_keys = {
            "F1.1_ADVANTAGE",
            "F1.1_COHENS_D",
            "F2.1_ADVANTAGE",
            "F2.1_PP_DIFF",
            "F2.1_COHENS_H",
            "F2.2_CORR",
            "F2.2_FISHER_Z",
            "F2.3_RT_ADVANTAGE",
            "F2.3_ALPHA",
            "F2.4_CONFIDENCE_EFFECT",
            "F2.4_BETA_INTERACTION",
            "F2.5_MAX_TRIALS",
            "F2.5_HAZARD_RATIO",
            "F2.CARDIAC_DETECTION_ADVANTAGE",
            "F3.1_ADVANTAGE",
            "F3.1_COHENS_D",
            "F3.2_INTERO_ADVANTAGE",
            "F3.2_COHENS_D",
            "F3.3_REDUCTION",
            "F3.3_COHENS_D",
            "F3.4_REDUCTION",
            "F3.4_COHENS_D",
            "F3.6_MAX_TRIALS",
            "F3.6_HAZARD_RATIO",
            "F5.5_PCA_VARIANCE",
            "F5.5_PCA_LOADING",
            "F5.6_PERF_DIFF",
            "F5.6_COHENS_D",
            "F5.6_ALPHA",
            "F6.1_LTCN_TRANSITION",
            "F6.2_INTEGRATION_RATIO",
            "F6.2_R2",
            "V7.1_PCI_REDUCTION",
            "V7.1_COHENS_D",
            "V9.1_CORR",
            "V9.3_CORR",
        }

        missing_in_registry = expected_keys - registry_keys
        if missing_in_registry:
            all_passed = False
            for key in sorted(missing_in_registry):
                err = f"THRESHOLD_REGISTRY missing key: {key}"
                all_errors.append(err)
                print(f"  FAIL: {err}")
        else:
            print("  OK: THRESHOLD_REGISTRY contains expected keys")
    except ImportError as e:
        all_passed = False
        err = f"Could not import THRESHOLD_REGISTRY: {e}"
        all_errors.append(err)
        print(f"  FAIL: {err}")
    print()

    # Summary
    print("=" * 70)
    if all_passed:
        print("RESULT: All checks PASSED")
        print("=" * 70)
        return 0
    else:
        print(f"RESULT: {len(all_errors)} issues found")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
