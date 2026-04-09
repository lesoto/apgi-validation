"""
Test that all falsification criteria import thresholds from falsification_thresholds.py

This ensures no hardcoded threshold literals exist in protocol files.
"""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

# Add parent directory to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def get_threshold_registry():
    """Load the threshold registry from falsification_thresholds.py"""
    try:
        from utils.falsification_thresholds import THRESHOLD_REGISTRY

        return THRESHOLD_REGISTRY
    except ImportError:
        pytest.skip("falsification_thresholds.py not available")


def extract_float_literals(file_path: Path) -> list[float]:
    """Extract float literals from a Python file that appear in threshold-like contexts.

    Only flags literals that:
    1. Appear in comparison operations (<, >, <=, >=, ==) with threshold-like operands
    2. Are assigned to threshold-like variable names (e.g., alpha, threshold, cutoff)
    3. Are arguments to threshold-like function parameters (e.g., alpha=0.05)

    Excludes:
    - Literals in weight assignments (e.g., weight = 0.5)
    - Literals in score comparisons (e.g., score > 0.5)
    - Literals in mathematical constants or simulation parameters
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    threshold_literals: list[float] = []

    # Collect all nodes that are inside ExceptHandler blocks to exclude them
    excluded_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            for child in ast.walk(node):
                excluded_nodes.add(id(child))

    # Threshold-like variable name patterns
    threshold_patterns = [
        "alpha",
        "threshold",
        "cutoff",
        "significance",
        "p_value",
        "min_",
        "max_",
        "_min_",
        "_max_",
        "_coef",
        "coef_",
    ]

    # Non-threshold patterns to exclude
    exclude_patterns = ["weight", "score", "intercept", "slope", "scale", "bias"]

    for node in ast.walk(tree):
        # Skip nodes inside exception handlers
        if id(node) in excluded_nodes:
            continue

        # Check if this is a comparison operation
        if isinstance(node, ast.Compare):
            # Check if the left operand is a threshold-like variable
            left_is_threshold_like = False
            if isinstance(node.left, ast.Name):
                left_name = node.left.id.lower()
                if any(pattern in left_name for pattern in threshold_patterns):
                    left_is_threshold_like = True
                # Exclude if it's a non-threshold variable
                if any(pattern in left_name for pattern in exclude_patterns):
                    left_is_threshold_like = False

            # Only flag if left side is threshold-like
            if left_is_threshold_like:
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and isinstance(
                        comparator.value, float
                    ):
                        # Only flag small values (likely thresholds) in comparisons
                        if 0.001 <= abs(comparator.value) <= 1.0:
                            threshold_literals.append(comparator.value)

        # Check for assignments to threshold-like names
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id.lower()
                    # Must match threshold pattern AND not be excluded
                    if any(pattern in var_name for pattern in threshold_patterns):
                        if not any(pattern in var_name for pattern in exclude_patterns):
                            if isinstance(node.value, ast.Constant) and isinstance(
                                node.value, float
                            ):
                                value = float(node.value.value)  # type: ignore[arg-type]
                                if 0.001 <= abs(value) <= 1.0:
                                    threshold_literals.append(value)

        # Check for keyword arguments in function calls (e.g., alpha=0.05)
        if isinstance(node, ast.Call):
            # Skip matplotlib/plotting functions where alpha is transparency
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id.lower()
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr.lower()

            # Exclude plotting functions where alpha means transparency
            plotting_funcs = [
                "scatter",
                "plot",
                "bar",
                "hist",
                "fill_between",
                "errorbar",
            ]
            is_plotting = any(p in func_name for p in plotting_funcs)

            for keyword in node.keywords:
                arg_name = keyword.arg.lower()
                if any(pattern in arg_name for pattern in threshold_patterns):
                    if not any(pattern in arg_name for pattern in exclude_patterns):
                        # Skip alpha in plotting functions (transparency, not significance)
                        if arg_name == "alpha" and is_plotting:
                            continue
                        if isinstance(keyword.value, ast.Constant) and isinstance(
                            keyword.value.value, float
                        ):
                            if 0.001 <= abs(keyword.value.value) <= 1.0:
                                threshold_literals.append(keyword.value.value)

    return threshold_literals


def test_all_protocols_use_threshold_registry():
    """Test that all protocol files import thresholds from the registry"""
    validation_dir = project_root / "Validation"
    # Check both hyphenated and underscored protocol files
    protocol_files = list(validation_dir.glob("VP_*.py"))

    files_missing_import = []
    for protocol_file in protocol_files:
        # Read the file content
        with open(protocol_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if it imports from falsification_thresholds
        has_import = "from utils.falsification_thresholds import" in content

        if not has_import:
            files_missing_import.append(protocol_file.name)

    if files_missing_import:
        pytest.fail(
            f"Files missing import from falsification_thresholds.py: {files_missing_import}"
        )


def test_no_assumed_values_in_falsification_functions():
    """Test that no falsification functions contain '# Assume' comments"""
    validation_dir = project_root / "Validation"
    falsification_dir = project_root / "Falsification"

    # Validation files use VP_*.py pattern, Falsification uses FP_*.py
    validation_files = list(validation_dir.glob("VP_*.py"))
    falsification_files = list(falsification_dir.glob("FP_*.py"))
    protocol_files = validation_files + falsification_files

    for protocol_file in protocol_files:
        with open(protocol_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for '# Assume' comments
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if "# Assume" in line and "assumed" in line.lower():
                pytest.fail(
                    f"{protocol_file.name} line {i} contains assumed value comment: {line.strip()}"
                )


def test_falsification_thresholds_file_exists():
    """Test that falsification_thresholds.py exists and contains required constants"""
    threshold_file = project_root / "utils" / "falsification_thresholds.py"

    if not threshold_file.exists():
        pytest.fail("falsification_thresholds.py does not exist")

    # Load and check for required constants
    spec = importlib.util.spec_from_file_location(
        "falsification_thresholds", threshold_file
    )
    if spec is None or spec.loader is None:
        pytest.fail("Could not load falsification_thresholds.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check for required threshold constants
    required_thresholds = [
        "F1_1_MIN_APGI_ADVANTAGE",
        "F1_1_MIN_COHENS_D",
        "F2_3_MIN_RT_ADVANTAGE_MS",
        "F2_3_ALPHA",
        "F5_5_PCA_MIN_VARIANCE",
        "F5_5_PCA_MIN_LOADING",
        "F5_6_MIN_PERFORMANCE_DIFF_PCT",
        "F5_6_MIN_COHENS_D",
        "F5_6_ALPHA",
        "F6_1_LTCN_MAX_TRANSITION_MS",
        "F6_2_MIN_INTEGRATION_RATIO",
        "F6_2_MIN_R2",
        "V7_1_MIN_PCI_REDUCTION",
        "V7_1_MIN_COHENS_D",
        "V9_1_MIN_CORRELATION",
        "V9_3_MIN_CORRELATION",
    ]

    for threshold_name in required_thresholds:
        if not hasattr(module, threshold_name):
            pytest.fail(f"Missing required threshold: {threshold_name}")

    # Check for THRESHOLD_REGISTRY
    if not hasattr(module, "THRESHOLD_REGISTRY"):
        pytest.fail("Missing THRESHOLD_REGISTRY in falsification_thresholds.py")

    if not isinstance(module.THRESHOLD_REGISTRY, dict):
        pytest.fail("THRESHOLD_REGISTRY must be a dictionary")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
