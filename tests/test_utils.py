"""
Tests for utility modules.
===========================

This module provides comprehensive functional tests for APGI utility modules,
going beyond simple directory structure checks to test actual functionality.
"""

import pytest

# Import utility modules to test
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.config_loader import (
        load_config_value,
        load_model_config,
        load_falsification_config,
    )
    from utils.threshold_registry import ThresholdRegistry, FalsificationThresholds
    from utils.constants import DIM_CONSTANTS
    from utils.error_handler import APGIError
    from utils.logging_config import apgi_logger
except ImportError as e:
    pytest.skip(f"Cannot import utility modules: {e}", allow_module_level=True)


def test_utils_directory_structure():
    """Test that utils directory has expected structure."""
    project_root = Path(__file__).parent.parent
    utils_dir = project_root / "utils"

    assert utils_dir.exists(), "utils directory missing"
    assert utils_dir.is_dir(), "utils should be a directory"

    # Check subdirectories (optional - may not exist in all environments)
    subdirs = ["config", "data"]
    for subdir in subdirs:
        subdir_path = utils_dir / subdir
        if subdir_path.exists():
            assert subdir_path.is_dir(), f"utils/{subdir} should be a directory"


def test_utility_files_exist():
    """Test that essential utility files exist."""
    project_root = Path(__file__).parent.parent
    utils_dir = project_root / "utils"

    # Check essential utility files
    utility_files = [
        "config_manager.py",
        "backup_manager.py",
        "batch_processor.py",
        "data_quality_assessment.py",
    ]

    for file_name in utility_files:
        file_path = utils_dir / file_name
        assert file_path.exists(), f"Utility file {file_name} missing"


def test_sample_data_fixture_structure(sample_data):
    """Test that sample_data fixture has expected structure."""
    assert isinstance(sample_data, dict)

    # Check required data keys
    required_keys = ["timestamps", "surprise", "threshold", "metabolic", "arousal"]
    for key in required_keys:
        assert key in sample_data, f"Sample data missing key: {key}"
        assert isinstance(sample_data[key], list), f"Sample data {key} should be a list"
        assert len(sample_data[key]) > 0, f"Sample data {key} should not be empty"

    # Check data consistency
    data_length = len(sample_data["timestamps"])
    for key in required_keys:
        assert (
            len(sample_data[key]) == data_length
        ), f"Sample data {key} length mismatch"


class TestConfigLoader:
    """Test the config_loader utility module."""

    def test_load_config_value_existing_key(self):
        """Test loading an existing configuration value."""
        # Test with a known key from default.yaml
        tau_S = load_config_value("model.tau_S", 0.5)
        assert isinstance(tau_S, (int, float)), "tau_S should be numeric"
        assert tau_S > 0, "tau_S should be positive"

    def test_load_config_value_missing_key(self):
        """Test loading a missing configuration key returns default."""
        # Test with a non-existent key
        value = load_config_value("nonexistent.key", "default_value")
        assert value == "default_value", "Should return default value for missing key"

    def test_load_model_config(self):
        """Test loading model configuration section."""
        model_config = load_model_config()
        assert isinstance(model_config, dict), "Model config should be a dictionary"

        # Check for expected model parameters
        expected_keys = [
            "tau_S",
            "tau_theta",
            "theta_0",
            "alpha",
            "gamma_M",
            "gamma_A",
            "rho",
        ]
        for key in expected_keys:
            assert key in model_config, f"Model config should contain {key}"

    def test_load_falsification_config(self):
        """Test loading falsification configuration section."""
        falsif_config = load_falsification_config()
        assert isinstance(
            falsif_config, dict
        ), "Falsification config should be a dictionary"

        # Check for expected falsification parameters
        expected_keys = [
            "cumulative_reward_advantage_threshold",
            "cohens_d_threshold",
            "significance_level",
        ]
        for key in expected_keys:
            assert key in falsif_config, f"Falsification config should contain {key}"


class TestThresholdRegistry:
    """Test the threshold_registry utility module."""

    def test_falsification_thresholds_defaults(self):
        """Test FalsificationThresholds default values."""
        thresholds = FalsificationThresholds()

        assert thresholds.cumulative_reward_advantage_threshold == 18.0
        assert thresholds.cohens_d_threshold == 0.60
        assert thresholds.significance_level == 0.01
        assert thresholds.threshold_reduction_min == 20.0

    def test_threshold_registry_initialization(self):
        """Test ThresholdRegistry initialization."""
        registry = ThresholdRegistry()
        assert isinstance(registry.thresholds, FalsificationThresholds)

        # Test getting individual thresholds
        advantage_threshold = registry.get_threshold(
            "cumulative_reward_advantage_threshold"
        )
        assert advantage_threshold == 18.0

        cohens_d = registry.get_threshold("cohens_d_threshold")
        assert cohens_d == 0.60

    def test_threshold_registry_validation(self):
        """Test threshold validation."""
        registry = ThresholdRegistry()

        # Valid threshold update
        assert registry.update_threshold("cumulative_reward_advantage_threshold", 20.0)

        # Invalid threshold (negative)
        with pytest.raises(ValueError):
            registry.update_threshold("cohens_d_threshold", -1.0)

    def test_get_all_thresholds(self):
        """Test getting all thresholds as dictionary."""
        registry = ThresholdRegistry()
        all_thresholds = registry.get_all_thresholds()

        assert isinstance(all_thresholds, dict)
        assert "cumulative_reward_advantage_threshold" in all_thresholds
        assert "cohens_d_threshold" in all_thresholds
        assert "significance_level" in all_thresholds


class TestConstants:
    """Test the constants utility module."""

    def test_dimension_constants(self):
        """Test that dimension constants are defined and positive."""
        assert hasattr(DIM_CONSTANTS, "EXTERO_DIM")
        assert hasattr(DIM_CONSTANTS, "INTERO_DIM")
        assert hasattr(DIM_CONSTANTS, "SENSORY_DIM")

        # Check that dimensions are positive integers
        assert isinstance(DIM_CONSTANTS.EXTERO_DIM, int)
        assert isinstance(DIM_CONSTANTS.INTERO_DIM, int)
        assert DIM_CONSTANTS.EXTERO_DIM > 0
        assert DIM_CONSTANTS.INTERO_DIM > 0

    def test_constant_values(self):
        """Test specific constant values."""
        # Test known constant values
        assert DIM_CONSTANTS.EXTERO_DIM == 32
        assert DIM_CONSTANTS.INTERO_DIM == 16
        assert DIM_CONSTANTS.SENSORY_DIM == 32


class TestErrorHandling:
    """Test the error_handler utility module."""

    def test_apgi_error_creation(self):
        """Test APGIError exception creation."""
        error = APGIError("Test error message")
        assert str(error) == "Test error message"

        # Test with error code
        error_with_code = APGIError("Test error", error_code=123)
        assert error_with_code.error_code == 123


class TestLoggingConfig:
    """Test the logging_config utility module."""

    def test_logger_exists(self):
        """Test that APGI logger is properly configured."""
        assert apgi_logger is not None
        assert hasattr(apgi_logger, "logger")
        assert hasattr(apgi_logger, "log_error_with_context")


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "timestamps": [0.0, 0.1, 0.2, 0.3, 0.4],
        "surprise": [1.0, 1.5, 2.0, 1.8, 1.2],
        "threshold": [0.5, 0.5, 0.5, 0.5, 0.5],
        "metabolic": [0.1, 0.15, 0.2, 0.18, 0.12],
        "arousal": [0.8, 0.9, 1.0, 0.95, 0.85],
    }
