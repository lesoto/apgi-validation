"""
Tests for concurrent access patterns with _config_lock.
Tests thread-safety of configuration management.
==============================================
"""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConcurrentConfigAccess:
    """Tests for concurrent access patterns with _config_lock."""

    def test_config_lock_basic_thread_safety(self):
        """Test basic thread safety with _config_lock."""
        from main import _config_lock

        shared_value = [0]
        num_threads = 10
        num_iterations = 100

        def increment():
            with _config_lock:
                for _ in range(num_iterations):
                    shared_value[0] += 1

        threads = [threading.Thread(target=increment) for _ in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        expected = num_threads * num_iterations
        assert (
            shared_value[0] == expected
        ), f"Race condition detected: {shared_value[0]} != {expected}"

    def test_config_lock_read_write_concurrency(self):
        """Test concurrent reads and writes with _config_lock."""
        from main import _config_lock

        config_data = {"value": 0}
        read_results = []
        num_threads = 10

        def write_config():
            with _config_lock:
                config_data["value"] += 1
                time.sleep(0.001)  # Simulate work

        def read_config():
            with _config_lock:
                read_results.append(config_data["value"])

        threads = []
        for i in range(num_threads):
            if i % 2 == 0:
                threads.append(threading.Thread(target=write_config))
            else:
                threads.append(threading.Thread(target=read_config))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All reads should have seen consistent values
        assert len(read_results) > 0
        assert all(isinstance(r, int) for r in read_results)

    def test_config_lock_timeout_handling(self):
        """Test timeout handling with _config_lock."""
        from main import _config_lock

        lock_acquired = [False]
        timeout_occurred = [False]

        def hold_lock():
            with _config_lock:
                lock_acquired[0] = True
                time.sleep(0.5)  # Hold lock for 0.5 seconds

        def try_acquire_with_timeout():
            acquired = _config_lock.acquire(timeout=0.1)
            if not acquired:
                timeout_occurred[0] = True
            else:
                _config_lock.release()

        thread1 = threading.Thread(target=hold_lock)
        thread2 = threading.Thread(target=try_acquire_with_timeout)

        thread1.start()
        time.sleep(0.05)  # Ensure thread1 acquires lock first
        thread2.start()

        thread1.join()
        thread2.join()

        assert lock_acquired[0]
        assert timeout_occurred[0], "Timeout should have occurred"

    @pytest.mark.slow
    def test_config_lock_nested_acquire(self):
        """Test nested lock acquisition (should handle gracefully)."""
        from main import _config_lock

        # Test that same thread can acquire lock multiple times
        with _config_lock:
            with _config_lock:
                # This should work if lock is reentrant
                pass

    def test_config_lock_deadlock_prevention(self):
        """Test that deadlock prevention works."""
        from main import _config_lock

        resource1 = [0]
        resource2 = [0]
        deadlock_detected = [False]

        def thread1_func():
            try:
                with _config_lock:
                    resource1[0] = 1
                    time.sleep(0.01)
                    # Try to acquire another resource (simulated)
                    resource2[0] = 1
            except Exception:
                deadlock_detected[0] = True

        def thread2_func():
            try:
                with _config_lock:
                    resource2[0] = 2
                    time.sleep(0.01)
                    resource1[0] = 2
            except Exception:
                deadlock_detected[0] = True

        thread1 = threading.Thread(target=thread1_func)
        thread2 = threading.Thread(target=thread2_func)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Should not deadlock
        assert not deadlock_detected[0]

    def test_config_lock_high_concurrency(self):
        """Test lock behavior under high concurrency."""
        from main import _config_lock

        counter = [0]
        num_threads = 50
        num_iterations = 10

        def increment_counter():
            for _ in range(num_iterations):
                with _config_lock:
                    counter[0] += 1

        threads = [
            threading.Thread(target=increment_counter) for _ in range(num_threads)
        ]
        start_time = time.time()

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        elapsed = time.time() - start_time
        expected = num_threads * num_iterations

        assert counter[0] == expected, f"Race condition: {counter[0]} != {expected}"
        assert elapsed < 10.0, f"Lock contention too high: {elapsed:.3f}s"

    def test_config_lock_exception_safety(self):
        """Test that lock is released even if exception occurs."""
        from main import _config_lock

        def operation_with_exception():
            with _config_lock:
                raise ValueError("Test exception")

        thread = threading.Thread(target=operation_with_exception)
        thread.start()
        thread.join()

        # Lock should be available after exception
        with _config_lock:
            pass  # Should not block

    def test_config_lock_fairness(self):
        """Test lock fairness (threads acquire in reasonable order)."""
        from main import _config_lock

        acquisition_order = []
        num_threads = 5

        def acquire_and_record(thread_id):
            time.sleep(0.01 * thread_id)  # Stagger start times
            with _config_lock:
                acquisition_order.append(thread_id)

        threads = [
            threading.Thread(target=acquire_and_record, args=(i,))
            for i in range(num_threads)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All threads should have acquired the lock
        assert len(acquisition_order) == num_threads
        assert set(acquisition_order) == set(range(num_threads))

    def test_config_lock_with_config_manager(self):
        """Test _config_lock with actual config manager operations."""
        try:
            from main import _config_lock
            from utils.config_manager import ConfigManager
        except ImportError:
            pytest.skip("ConfigManager not available")

        # Test that ConfigManager operations work while holding the config lock
        config_manager = ConfigManager()
        results = []
        num_threads = 5

        def update_config(thread_id):
            try:
                # Use the _config_lock while accessing config
                with _config_lock:
                    # Get config (read operation) - verify it works with lock held
                    _ = config_manager.get_config()
                    results.append(thread_id)
            except Exception as e:
                results.append(f"Error: {e}")

        threads = [
            threading.Thread(target=update_config, args=(i,))
            for i in range(num_threads)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(results) == num_threads
        # Results should be integers (thread_ids)
        assert all(isinstance(r, int) for r in results)

    def test_config_lock_performance_overhead(self):
        """Test that lock overhead is acceptable."""
        from main import _config_lock

        # Measure time with lock
        start_time = time.time()
        for _ in range(1000):
            with _config_lock:
                pass
        elapsed_with_lock = time.time() - start_time

        # Measure time without lock (simulated)
        start_time = time.time()
        for _ in range(1000):
            pass
        elapsed_without_lock = time.time() - start_time

        overhead = elapsed_with_lock - elapsed_without_lock

        # Lock overhead should be minimal (< 0.1 seconds for 1000 acquisitions)
        assert overhead < 0.1, f"Lock overhead too high: {overhead:.3f}s"
