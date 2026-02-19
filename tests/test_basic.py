"""
Basic tests for APGI validation framework.
=========================================
"""

import pytest
from pathlib import Path


def test_import_main():
    """Test that main module can be imported."""
    try:
        import main

        assert main is not None
    except ImportError as e:
        if "click" in str(e):
            pytest.skip(f"main module requires click, which is not installed: {e}")
        else:
            pytest.fail(f"Failed to import main module: {e}")


def test_import_validation():
    """Test that validation module can be imported."""
    try:
        import Validation

        assert Validation is not None
    except ImportError as e:
        pytest.fail(f"Failed to import Validation module: {e}")


def test_project_structure():
    """Test that essential project directories exist."""
    project_root = Path(__file__).parent.parent

    # Check essential directories
    assert (project_root / "Validation").exists(), "Validation directory missing"
    assert (project_root / "utils").exists(), "utils directory missing"
    assert (project_root / "config").exists(), "config directory missing"
    assert (project_root / "docs").exists(), "docs directory missing"


def test_config_files_exist():
    """Test that configuration files exist."""
    project_root = Path(__file__).parent.parent

    # Check essential config files
    assert (project_root / "config" / "default.yaml").exists(), "default.yaml missing"
    assert (project_root / "requirements.txt").exists(), "requirements.txt missing"
    assert (project_root / "pytest.ini").exists(), "pytest.ini missing"


def test_sample_config_fixture(sample_config):
    """Test that sample_config fixture provides expected data."""
    assert isinstance(sample_config, dict)
    assert "model" in sample_config
    assert "simulation" in sample_config
    assert "validation" in sample_config

    # Check model parameters
    model_params = sample_config["model"]
    assert "tau_S" in model_params
    assert "tau_theta" in model_params
    assert isinstance(model_params["tau_S"], (int, float))
    assert isinstance(model_params["tau_theta"], (int, float))


def test_temp_dir_fixture(temp_dir):
    """Test that temp_dir fixture provides a valid temporary directory."""
    assert temp_dir.exists()
    assert temp_dir.is_dir()
