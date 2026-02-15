"""
Tests for validation protocols.
===============================
"""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_validation_files_exist():
    """Test that validation protocol files exist."""
    project_root = Path(__file__).parent.parent
    validation_dir = project_root / "Validation"

    # Check validation protocol files
    validation_files = [
        "Validation-Protocol-1.py",
        "Validation-Protocol-2.py",
        "APGI-Master-Validation.py",
        "APGI-Validation-GUI.py",
    ]

    for file_name in validation_files:
        file_path = validation_dir / file_name
        assert file_path.exists(), f"Validation file {file_name} missing"


def test_validation_config_structure(sample_config):
    """Test validation configuration structure."""
    validation_config = sample_config.get("validation", {})

    # Check required validation settings
    required_settings = [
        "enable_cross_validation",
        "cv_folds",
        "enable_sensitivity_analysis",
        "sensitivity_samples",
        "enable_robustness_tests",
        "significance_level",
    ]

    for setting in required_settings:
        assert setting in validation_config, f"Validation setting {setting} missing"

    # Check data types
    assert isinstance(validation_config["cv_folds"], int)
    assert isinstance(validation_config["sensitivity_samples"], int)
    assert isinstance(validation_config["significance_level"], (int, float))
    assert validation_config["cv_folds"] > 0
    assert validation_config["sensitivity_samples"] > 0
    assert 0 <= validation_config["significance_level"] <= 1
