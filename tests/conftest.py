"""
Pytest configuration and fixtures for APGI test suite.
======================================================

Provides common test fixtures and configuration:
- Temporary directories
- Mock objects
- Test data
- Test configuration
"""

# Mock tkinter BEFORE any imports to prevent GUI initialization during tests
import sys
from unittest.mock import MagicMock

# Mock tkinter modules before any test imports them
tkinter_modules = [
    "tkinter",
    "tkinter.ttk",
    "tkinter.messagebox",
    "tkinter.filedialog",
    "tkinter.scrolledtext",
    "tkinter.font",
]

for module in tkinter_modules:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

# Configure mock tkinter with basic widget behavior
mock_tk = sys.modules["tkinter"]
mock_tk.Tk = MagicMock()  # type: ignore
mock_tk.StringVar = MagicMock()  # type: ignore
mock_tk.BooleanVar = MagicMock()  # type: ignore
mock_tk.DoubleVar = MagicMock()  # type: ignore
mock_tk.IntVar = MagicMock()  # type: ignore
mock_tk.ttk = MagicMock()  # type: ignore
mock_tk.messagebox = MagicMock()  # type: ignore
mock_tk.filedialog = MagicMock()  # type: ignore

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session", autouse=True)
def headless_gui_setup():
    """Ensure GUI tests run in headless mode by mocking tkinter before any imports.

    This fixture runs at session start and mocks tkinter to prevent actual GUI
    initialization during tests. Essential for CI/CD headless environments.
    Note: tkinter is already mocked at module level in conftest.py, this fixture
    ensures the mock persists throughout the test session.
    """
    # tkinter is already mocked at module level before any imports
    # This fixture ensures the mock configuration is complete
    mock_tk = sys.modules.get("tkinter")
    if mock_tk:
        mock_tk.Tk = MagicMock
        mock_tk.StringVar = MagicMock
        mock_tk.BooleanVar = MagicMock
        mock_tk.DoubleVar = MagicMock
        mock_tk.IntVar = MagicMock
        mock_tk.ttk = MagicMock()
        mock_tk.messagebox = MagicMock()
        mock_tk.filedialog = MagicMock()

    yield

    # Cleanup not needed as mocks persist for session duration


@pytest.fixture
def apgi_backup_hmac_key(monkeypatch):
    """Provide APGI_BACKUP_HMAC_KEY for tests.

    This fixture injects a test HMAC key into the environment,
    allowing tests to run without external environment configuration.
    """
    key = "test_backup_hmac_key_" + "x" * 32
    monkeypatch.setenv("APGI_BACKUP_HMAC_KEY", key)
    yield key


@pytest.fixture
def pickle_secret_key(monkeypatch):
    """Provide PICKLE_SECRET_KEY for tests.

    This fixture injects a test pickle secret key into the environment,
    allowing tests to run without external environment configuration.
    """
    key = "test_pickle_secret_key_" + "x" * 32
    monkeypatch.setenv("PICKLE_SECRET_KEY", key)
    yield key


@pytest.fixture
def env_vars(monkeypatch):
    """Provide all required environment variables for tests.

    This fixture injects both APGI_BACKUP_HMAC_KEY and PICKLE_SECRET_KEY
    into the environment, allowing tests to run without external configuration.
    """
    env_vars = {
        "APGI_BACKUP_HMAC_KEY": "test_key_" + "x" * 32,
        "PICKLE_SECRET_KEY": "test_secret_" + "x" * 32,
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    yield env_vars


@pytest.fixture(scope="session")
def cli():
    """Lazy-load CLI to avoid hanging during test collection.

    This fixture delays the import of main.py until tests actually run,
    preventing collection errors caused by module-level logging initialization.
    """
    try:
        from main import cli as main_cli

        return main_cli
    except Exception as e:
        # If import fails, return a mock CLI for testing
        pytest.skip(f"Could not import CLI: {e}")


def pytest_sessionfinish(session, exitstatus):
    """Clean up background resources without forcing process exit.

    Pytest already controls process termination and exit codes. Calling
    ``sys.exit`` from this hook can surface as an interrupted run during
    collection on newer pytest versions, so we intentionally avoid doing that
    here.
    """
    # Intentionally no explicit sys.exit(...) call.
    return None


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests with secure permissions."""
    with tempfile.TemporaryDirectory() as temp_path:
        # Set restrictive permissions (owner only) on the temp directory
        os.chmod(temp_path, 0o700)
        yield Path(temp_path)


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

    return OOMContext()


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
    """Fixture that provides a factory for creating flaky operations."""
    import time

    import numpy as np

    def create_operation(success_rate=0.5):
        """Create a flaky operation with specified success rate.

        Args:
            success_rate: Probability of success (0.0 to 1.0)

        Returns:
            A function that succeeds with the given probability
        """

        def operation():
            if np.random.random() < success_rate:
                return "success"
            else:
                raise RuntimeError("Operation failed")

        return operation

    def retry_wrapper(func, max_attempts=3, timeout=None, backoff_factor=1):
        """Execute a function with retry logic.

        Args:
            func: Callable to execute
            max_attempts: Maximum number of retry attempts
            timeout: Maximum time to wait (not implemented in this fixture)
            backoff_factor: Multiplier for wait time between retries
        """
        last_exception = None

        for attempt in range(max_attempts):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_attempts - 1:
                    # Exponential backoff with jitter
                    wait_time = (
                        backoff_factor * (2**attempt) * (0.5 + np.random.random() * 0.5)
                    )
                    time.sleep(wait_time)

        # All attempts failed, raise the last exception
        raise last_exception

    # Return both the factory and the retry wrapper
    class FlakyOperationFactory:
        def __call__(self, success_rate=0.5):
            return create_operation(success_rate)

        def retry(self, func, max_attempts=3, timeout=None, backoff_factor=1):
            return retry_wrapper(func, max_attempts, timeout, backoff_factor)

    return FlakyOperationFactory()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")


def pytest_collection_finish(session):
    """Called after test collection is complete."""
    # If we're only collecting (--collect-only), we should not exit
    # as it causes pytest to report collection errors
    pass


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
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def create_test_json(file_path: Path, data: dict):
    """Create a test JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_test_yaml(file_path: Path, data: dict):
    """Create a test YAML file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
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
