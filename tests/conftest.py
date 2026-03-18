"""
Pytest configuration and fixtures for APGI test suite.
======================================================

Provides common test fixtures and configuration:
- Temporary directories
- Mock objects
- Test data
- Test configuration
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

# Configure Hypothesis profiles
from hypothesis import settings, HealthCheck

# Define Hypothesis profiles for different test environments
# Use individual health checks to avoid version compatibility issues
hypothesis_profiles = {
    "dev": settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
            HealthCheck.large_base_example,
            HealthCheck.data_too_large,
        ],
    ),
    "ci": settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
            HealthCheck.large_base_example,
            HealthCheck.data_too_large,
        ],
        derandomize=True,  # Deterministic for CI
    ),
    "thorough": settings(
        max_examples=100,
        deadline=None,
    ),
}

# Apply settings based on environment variable or auto-detect CI environment
hypothesis_profile = os.getenv("HYPOTHESIS_PROFILE", "dev")

# Auto-detect CI environments if not explicitly set
if hypothesis_profile == "dev":
    # Check for common CI environment variables
    ci_indicators = [
        "CI",  # GitHub Actions, GitLab CI, Jenkins
        "GITHUB_ACTIONS",  # GitHub Actions
        "GITLAB_CI",  # GitLab CI
        "JENKINS_URL",  # Jenkins
        "BUILDKITE",  # Buildkite
        "TRAVIS",  # Travis CI
        "CIRCLECI",  # CircleCI
        "CODEBUILD_ID",  # AWS CodeBuild
        "GOOGLE_CLOUD_BUILD",  # Google Cloud Build
        "VERCEL",  # Vercel
        "NETLIFY",  # Netlify
    ]
    is_ci_environment = any(os.getenv(indicator) for indicator in ci_indicators)
    if is_ci_environment:
        hypothesis_profile = "ci"

if hypothesis_profile in hypothesis_profiles:
    settings.register_profile(
        hypothesis_profile, hypothesis_profiles[hypothesis_profile]
    )
    settings.load_profile(hypothesis_profile)

sys.path.insert(0, str(Path(__file__).parent.parent))


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
def sample_data():
    """Provide sample data for testing."""
    import numpy as np

    # Generate sufficient data points for time-series validation
    n_samples = 1000
    dt = 0.01
    time = np.arange(0, n_samples * dt, dt)

    return {
        "timestamps": time.tolist(),
        "surprise": (
            0.2
            + 0.1 * np.sin(2 * np.pi * 0.1 * time)
            + 0.05 * np.random.randn(n_samples)
        ).tolist(),
        "threshold": (
            0.5
            + 0.02 * np.sin(2 * np.pi * 0.05 * time)
            + 0.01 * np.random.randn(n_samples)
        ).tolist(),
        "metabolic": (
            1.0
            + 0.1 * np.sin(2 * np.pi * 0.2 * time)
            + 0.05 * np.random.randn(n_samples)
        ).tolist(),
        "arousal": (
            0.8
            + 0.1 * np.sin(2 * np.pi * 0.15 * time)
            + 0.03 * np.random.randn(n_samples)
        ).tolist(),
    }


@pytest.fixture
def raises_fixture():
    """Fixture that provides a context manager for testing exceptions."""

    class RaisesContext:
        def __init__(self, expected_exception=Exception):
            self.expected_exception = expected_exception
            self.exception_raised = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                pytest.fail(
                    f"Expected {self.expected_exception.__name__} to be raised, but no exception was raised"
                )
            if not isinstance(exc_val, self.expected_exception):
                pytest.fail(
                    f"Expected {self.expected_exception.__name__} to be raised, but got {exc_type.__name__}: {exc_val}"
                )
            self.exception_raised = exc_val
            return True  # Suppress the exception

    return RaisesContext


@pytest.fixture
def oom_fixture():
    """Fixture for testing out-of-memory conditions."""

    class OOMContext:
        def __init__(self):
            self.original_memory_limit = None

        def __enter__(self):
            # Try to simulate OOM by setting a very low memory limit
            # This is a best-effort simulation since actual OOM is hard to trigger safely
            try:
                import resource

                self.original_memory_limit = resource.getrlimit(resource.RLIMIT_AS)
                # Set memory limit to 10MB for testing
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (10 * 1024 * 1024, self.original_memory_limit[1]),
                )
            except ImportError:
                # resource module not available on Windows
                pass
            except Exception:
                # If we can't set limits, we'll use a mock approach
                pass
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original memory limit
            if self.original_memory_limit is not None:
                try:
                    import resource

                    resource.setrlimit(resource.RLIMIT_AS, self.original_memory_limit)
                except Exception:
                    pass
            return False  # Don't suppress exceptions

    return OOMContext


@pytest.fixture
def mock_memory_error():
    """Fixture that mocks memory allocation to raise MemoryError."""

    class MemoryErrorMocker:
        def __init__(self):
            self.patches = []

        def patch_numpy_zeros(self):
            """Patch numpy.zeros to raise MemoryError after a few calls."""
            call_count = [0]
            original_zeros = __import__("numpy").zeros

            def mock_zeros(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] > 3:  # Allow a few calls before failing
                    raise MemoryError("Simulated out of memory")
                return original_zeros(*args, **kwargs)

            patch_obj = patch("numpy.zeros", side_effect=mock_zeros)
            patch_obj.start()
            self.patches.append(patch_obj)
            return patch_obj

        def patch_torch_tensor(self):
            """Patch torch tensor creation to raise MemoryError."""
            try:
                torch = __import__("torch")
                call_count = [0]
                original_tensor = torch.tensor

                def mock_tensor(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] > 2:
                        raise RuntimeError("CUDA out of memory")  # Simulates GPU OOM
                    return original_tensor(*args, **kwargs)

                patch_obj = patch("torch.tensor", side_effect=mock_tensor)
                patch_obj.start()
                self.patches.append(patch_obj)
                return patch_obj
            except ImportError:
                return None

        def cleanup(self):
            """Clean up all patches."""
            for patch_obj in self.patches:
                patch_obj.stop()
            self.patches.clear()

    mocker = MemoryErrorMocker()
    yield mocker
    mocker.cleanup()


@pytest.fixture
def exception_test_cases():
    """Provide common exception test cases."""
    return {
        "value_error": ValueError("Invalid value"),
        "type_error": TypeError("Invalid type"),
        "key_error": KeyError("Missing key"),
        "attribute_error": AttributeError("'NoneType' object has no attribute"),
        "io_error": IOError("File operation failed"),
        "memory_error": MemoryError("Out of memory"),
        "runtime_error": RuntimeError("Runtime error"),
        "assertion_error": AssertionError("Assertion failed"),
    }


@pytest.fixture
def random_seed():
    """Provide a fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def seeded_rng(random_seed):
    """Provide a numpy RandomState with a fixed seed for reproducible tests."""
    return np.random.RandomState(random_seed)


@pytest.fixture(autouse=True)
def reset_random_state_before_each_test():
    """Reset random state before each test for reproducibility."""
    original_state = np.random.get_state()
    yield
    np.random.set_state(original_state)


@pytest.fixture
def flaky_operation():
    """Fixture that simulates a flaky operation that may fail intermittently."""
    import random

    def operation(success_rate=0.7):
        """Simulate an operation that succeeds with given probability."""
        if random.random() < success_rate:
            return "success"
        else:
            raise RuntimeError("Flaky operation failed")

    return operation


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


def assert_performance_within_tolerance(
    actual_time: float, expected_time: float, tolerance: float
):
    """Assert that performance is within acceptable tolerance."""
    lower_bound = expected_time * (1 - tolerance)
    upper_bound = expected_time * (1 + tolerance)
    assert (
        lower_bound <= actual_time <= upper_bound
    ), f"Performance {actual_time:.3f}s not within tolerance of {expected_time:.3f}s ± {tolerance * 100}%"


if __name__ == "__main__":
    print("conftest.py is a pytest configuration file and should not be run directly.")
    print("Use 'pytest' to run the test suite.")
    exit(0)
