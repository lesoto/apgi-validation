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
        from falsification_thresholds import THRESHOLD_REGISTRY

        return THRESHOLD_REGISTRY
    except ImportError:
        pytest.skip("falsification_thresholds.py not available")


def extract_float_literals(file_path: Path) -> list:
    """Extract all float literals from a Python file"""
    with open(file_path, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    literals = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            literals.append(node.value)
        elif isinstance(node, ast.Num):  # For older Python versions
            literals.append(node.n)

    return literals


def test_all_protocols_use_threshold_registry():
    """Test that all protocol files import thresholds from the registry"""
    threshold_registry = get_threshold_registry()
    validation_dir = project_root / "Validation"
    protocol_files = list(validation_dir.glob("Validation-Protocol-*.py"))

    for protocol_file in protocol_files:
        # Read the file content
        with open(protocol_file, "r") as f:
            content = f.read()

        # Check if it imports from falsification_thresholds
        has_import = "from falsification_thresholds import" in content

        if not has_import:
            pytest.fail(
                f"{protocol_file.name} does not import from falsification_thresholds.py"
            )

        # Check for hardcoded threshold literals (within 1% of registry values)
        float_literals = extract_float_literals(protocol_file)
        for literal in float_literals:
            for threshold_name, threshold_value in threshold_registry.items():
                # Check if literal is within 1% of threshold value
                if abs(literal - threshold_value) / threshold_value < 0.01:
                    pytest.fail(
                        f"{protocol_file.name} contains hardcoded threshold literal {literal} "
                        f"matching {threshold_name}={threshold_value}"
                    )


def test_no_assumed_values_in_falsification_functions():
    """Test that no falsification functions contain '# Assume' comments"""
    validation_dir = project_root / "Validation"
    falsification_dir = project_root / "Falsification"

    for directory in [validation_dir, falsification_dir]:
        protocol_files = list(directory.glob("*Protocol*.py"))

        for protocol_file in protocol_files:
            with open(protocol_file, "r") as f:
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
    threshold_file = project_root / "falsification_thresholds.py"

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
