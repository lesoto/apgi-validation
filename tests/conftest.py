"""
Pytest configuration and fixtures for APGI test suite
======================================================

Provides common test fixtures and configuration:
- Temporary directories
- Mock objects
- Test data
- Test configuration
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Provide sample configuration data."""
    return {
        "model": {
            "tau_S": 0.5,
            "tau_theta": 30.0,
            "theta_0": 0.5,
            "alpha": 10.0,
            "gamma_M": -0.3,
            "gamma_A": 0.1,
            "rho": 0.7,
            "sigma_S": 0.05,
            "sigma_theta": 0.02,
        },
        "simulation": {
            "default_steps": 1000,
            "default_dt": 0.01,
            "max_steps": 100000,
            "enable_plots": True,
            "plot_format": "png",
            "plot_dpi": 150,
            "save_results": True,
            "results_format": "csv",
        },
        "logging": {
            "level": "INFO",
            "enable_console": True,
            "log_rotation": "10 MB",
            "log_retention": "30 days",
            "enable_performance_logging": True,
            "enable_structured_logging": True,
        },
        "data": {
            "default_data_dir": "data",
            "supported_formats": ["csv", "json", "xlsx", "pkl"],
            "max_file_size_mb": 100,
            "enable_caching": True,
            "cache_dir": "cache",
        },
        "validation": {
            "enable_cross_validation": True,
            "cv_folds": 5,
            "enable_sensitivity_analysis": True,
            "sensitivity_samples": 100,
            "enable_robustness_tests": True,
            "significance_level": 0.05,
        },
    }


@pytest.fixture
def config_file(temp_dir, sample_config):
    """Provide a temporary configuration file."""
    config_path = temp_dir / "test_config.yaml"

    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)

    return config_path


@pytest.fixture
def invalid_config_file(temp_dir):
    """Provide an invalid configuration file."""
    config_path = temp_dir / "invalid_config.yaml"

    with open(config_path, "w") as f:
        f.write("invalid: yaml: content: [")

    return config_path


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        "timestamps": [0.0, 0.1, 0.2, 0.3, 0.4],
        "surprise": [0.1, 0.2, 0.15, 0.25, 0.3],
        "threshold": [0.5, 0.52, 0.51, 0.53, 0.54],
        "metabolic": [1.0, 1.1, 1.05, 1.15, 1.2],
        "arousal": [0.8, 0.85, 0.82, 0.88, 0.9],
    }


@pytest.fixture
def mock_logger():
    """Provide a mock logger."""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    return logger


@pytest.fixture
def cache_file(temp_dir):
    """Provide a temporary cache file."""
    return temp_dir / "test_cache.json"


@pytest.fixture
def log_file(temp_dir):
    """Provide a temporary log file."""
    return temp_dir / "test.log"


@pytest.fixture
def profile_data():
    """Provide sample profile data."""
    return {
        "name": "test_profile",
        "description": "Test configuration profile",
        "category": "test",
        "version": "1.0",
        "author": "test",
        "tags": ["test", "sample"],
        "created_at": "2024-01-01T00:00:00",
        "parameters": {"model": {"tau_S": 0.8, "tau_theta": 25.0}},
        "metadata": {"purpose": "testing"},
    }


@pytest.fixture
def performance_data():
    """Provide sample performance data."""
    return {
        "execution_time": 1.23,
        "memory_usage": 1024,
        "cpu_usage": 0.75,
        "operations_per_second": 1000,
        "cache_hit_rate": 0.85,
    }


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data after each test."""
    yield

    # Clean up any test files that might have been created
    import glob

    test_files = glob.glob("**/test_*", recursive=True)
    for file_path in test_files:
        try:
            Path(file_path).unlink()
        except (FileNotFoundError, PermissionError):
            pass


@pytest.fixture
def mock_environment():
    """Provide mock environment variables."""
    return {
        "APGI_LOG_LEVEL": "DEBUG",
        "APGI_ENABLE_PLOTS": "true",
        "APGI_DATA_DIR": "/tmp/test_data",
        "APGI_TAU_S": "0.7",
        "APGI_TAU_THETA": "25.0",
    }


@pytest.fixture
def error_context_data():
    """Provide sample error context data."""
    return {
        "operation": "test_operation",
        "component": "test_component",
        "user_data": {"param1": "value1", "param2": "value2"},
    }


@pytest.fixture
def validation_errors():
    """Provide sample validation errors."""
    return [
        {
            "field": "tau_S",
            "value": 2.0,
            "constraint": "range",
            "expected": (0.1, 1.0),
            "message": "Value out of valid range",
        },
        {
            "field": "alpha",
            "value": "invalid",
            "constraint": "type",
            "expected": "number",
            "message": "Invalid type",
        },
    ]


@pytest.fixture
def benchmark_data():
    """Provide benchmark data for performance testing."""
    return {
        "small_dataset": {"size": 100, "expected_time": 0.1, "tolerance": 0.05},
        "medium_dataset": {"size": 10000, "expected_time": 1.0, "tolerance": 0.2},
        "large_dataset": {"size": 1000000, "expected_time": 10.0, "tolerance": 2.0},
    }


# Test markers
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add unit marker to tests in test files that don't have integration markers
        if "integration" not in item.keywords and "performance" not in item.keywords:
            item.add_marker(pytest.mark.unit)

        # Add slow marker to performance tests
        if "performance" in item.keywords:
            item.add_marker(pytest.mark.slow)


# Test utilities
def create_test_file(file_path: Path, content: str):
    """Create a test file with given content."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)


def create_test_json(file_path: Path, data: dict):
    """Create a test JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def create_test_yaml(file_path: Path, data: dict):
    """Create a test YAML file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        yaml.dump(data, f)


def assert_performance_within_tolerance(actual_time: float, expected_time: float, tolerance: float):
    """Assert that performance is within acceptable tolerance."""
    lower_bound = expected_time * (1 - tolerance)
    upper_bound = expected_time * (1 + tolerance)
    assert (
        lower_bound <= actual_time <= upper_bound
    ), f"Performance {actual_time:.3f}s not within tolerance of {expected_time:.3f}s ± {tolerance*100}%"
