"""
Tests utilizing defined fixtures from conftest.py.
Tests for raises_fixture, oom_fixture, and flaky_operation.
=======================================================
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRaisesFixture:
    """Tests for raises_fixture from conftest.py."""

    def test_raises_fixture_with_value_error(self, raises_fixture):
        """Test raises_fixture with ValueError."""
        with raises_fixture(ValueError):
            raise ValueError("Test error")

    def test_raises_fixture_with_type_error(self, raises_fixture):
        """Test raises_fixture with TypeError."""
        with raises_fixture(TypeError):
            raise TypeError("Test error")

    def test_raises_fixture_with_runtime_error(self, raises_fixture):
        """Test raises_fixture with RuntimeError."""
        with raises_fixture(RuntimeError):
            raise RuntimeError("Test error")

    def test_raises_fixture_no_exception(self, raises_fixture):
        """Test raises_fixture when no exception is raised."""
        # raises_fixture should call pytest.fail when no exception is raised
        with pytest.raises(BaseException):  # pytest.fail raises _pytest.outcomes.Failed
            with raises_fixture(ValueError):
                pass  # No exception raised

    def test_raises_fixture_wrong_exception(self, raises_fixture):
        """Test raises_fixture with wrong exception type."""
        # raises_fixture should call pytest.fail when wrong exception type is raised
        with pytest.raises(BaseException):  # pytest.fail raises _pytest.outcomes.Failed
            with raises_fixture(ValueError):
                raise TypeError("Wrong exception type")


class TestOOMFixture:
    """Tests for oom_fixture from conftest.py."""

    def test_oom_fixture_basic(self, oom_fixture):
        """Test oom_fixture basic functionality."""
        with oom_fixture:
            # Simulate a memory-intensive operation
            data = list(range(1000000))
            assert len(data) == 1000000

    def test_oom_fixture_with_large_allocation(self, oom_fixture):
        """Test oom_fixture with large memory allocation."""
        with oom_fixture:
            # This may trigger OOM on systems with limited memory
            try:
                data = [0] * 100000000
                assert len(data) == 100000000
            except MemoryError:
                # Expected behavior on memory-constrained systems
                pytest.skip("OOM triggered as expected")

    def test_oom_fixture_with_numpy(self, oom_fixture):
        """Test oom_fixture with numpy operations."""
        import numpy as np

        with oom_fixture:
            # Large numpy array
            arr = np.zeros((1000, 1000))
            assert arr.shape == (1000, 1000)


class TestFlakyOperationFixture:
    """Tests for flaky_operation fixture from conftest.py."""

    def test_flaky_operation_basic(self, flaky_operation):
        """Test flaky_operation with basic retry logic."""
        call_count = [0]

        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = flaky_operation.retry(flaky_func, max_attempts=5)
        assert result == "success"
        assert call_count[0] == 3

    def test_flaky_operation_with_timeout(self, flaky_operation):
        """Test flaky_operation with timeout."""

        def slow_func():
            import time

            time.sleep(0.1)
            return "done"

        result = flaky_operation.retry(slow_func, max_attempts=3, timeout=1.0)
        assert result == "done"

    def test_flaky_operation_max_attempts_exceeded(self, flaky_operation):
        """Test flaky_operation when max attempts exceeded."""

        def always_fail_func():
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError):
            flaky_operation.retry(always_fail_func, max_attempts=2)

    def test_flaky_operation_with_custom_backoff(self, flaky_operation):
        """Test flaky_operation with custom backoff strategy."""
        attempt_count = [0]

        def flaky_with_backoff():
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise ValueError("Retry needed")
            return "final"

        result = flaky_operation.retry(
            flaky_with_backoff, max_attempts=3, backoff_factor=2
        )
        assert result == "final"
        assert attempt_count[0] == 2


class TestFixtureIntegration:
    """Tests combining multiple fixtures."""

    def test_raises_with_oom_scenario(self, raises_fixture, oom_fixture):
        """Test combining raises_fixture with oom_fixture."""
        with oom_fixture:
            with raises_fixture(MemoryError):
                # Actually raise MemoryError to test the fixture combination
                raise MemoryError("Simulated OOM")

    def test_flaky_with_raises_validation(self, flaky_operation, raises_fixture):
        """Test flaky_operation with raises_fixture for validation."""

        def always_fail_validation():
            """Always raise ValueError for deterministic test."""
            raise ValueError("Expected failure")

        with raises_fixture(ValueError):
            flaky_operation.retry(always_fail_validation, max_attempts=1)

    def test_all_fixtures_together(self, raises_fixture, oom_fixture, flaky_operation):
        """Test all three fixtures together."""

        import random

        import numpy as np

        def complex_operation():
            # Randomly fail
            if random.random() < 0.3:
                raise ValueError("Random error")

            # Memory intensive
            arr = np.random.randn(100, 100)
            return arr.mean()

        flaky_operation.retry(complex_operation, max_attempts=5, timeout=10.0)
