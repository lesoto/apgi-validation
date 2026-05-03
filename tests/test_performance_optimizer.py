"""
Comprehensive Tests for Performance Optimizer Module
=====================================================

Target: 100% coverage for utils/performance_optimizer.py
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.performance_optimizer import (
    MemoizationCache,
    PerformanceMetrics,
    memoize,
    parallel_map,
    timed_execution,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass"""

    def test_metrics_creation(self):
        """Test metrics creation"""
        metrics = PerformanceMetrics(
            execution_time=1.5,
            memory_used_mb=100.0,
            cpu_percent=50.0,
        )
        assert metrics.execution_time == 1.5
        assert metrics.memory_used_mb == 100.0
        assert metrics.cpu_percent == 50.0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0

    def test_metrics_with_cache(self):
        """Test metrics with cache stats"""
        metrics = PerformanceMetrics(
            execution_time=1.0,
            memory_used_mb=50.0,
            cpu_percent=25.0,
            cache_hits=10,
            cache_misses=5,
        )
        assert metrics.cache_hits == 10
        assert metrics.cache_misses == 5


class TestMemoizationCache:
    """Test MemoizationCache class"""

    def test_cache_creation(self):
        """Test cache creation"""
        cache = MemoizationCache(max_size=100, ttl_seconds=60)
        assert cache.max_size == 100
        assert cache.ttl_seconds == 60
        assert cache._hits == 0
        assert cache._misses == 0

    def test_cache_get_set(self):
        """Test cache get and set operations"""
        cache = MemoizationCache()
        cache.set("key1", "value1")

        result = cache.get("key1")
        assert result == "value1"
        assert cache._hits == 1

    def test_cache_miss(self):
        """Test cache miss"""
        cache = MemoizationCache()
        result = cache.get("nonexistent")
        assert result is None
        assert cache._misses == 1

    def test_cache_expiration(self):
        """Test cache entry expiration"""
        cache = MemoizationCache(ttl_seconds=0.1)
        cache.set("key1", "value1")

        # Wait for expiration
        time.sleep(0.2)

        result = cache.get("key1")
        assert result is None

    def test_cache_size_limit(self):
        """Test cache size limit enforcement"""
        cache = MemoizationCache(max_size=5)

        # Add more entries than max_size
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")

        # Cache should only have 5 entries
        assert len(cache._cache) == 5

    def test_cache_lru_eviction(self):
        """Test LRU eviction policy"""
        cache = MemoizationCache(max_size=3)

        cache.set("key1", "value1")
        time.sleep(0.05)
        cache.set("key2", "value2")
        time.sleep(0.05)
        cache.set("key3", "value3")

        # Access key1 to make it recently used
        cache.get("key1")
        time.sleep(0.05)

        # Add key4, should evict key2 (least recently used)
        cache.set("key4", "value4")

        assert "key1" in cache._cache
        assert "key2" not in cache._cache
        assert "key3" in cache._cache
        assert "key4" in cache._cache

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = MemoizationCache()

        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key1")  # hit
        cache.get("nonexistent")  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2 / 3
        assert stats["size"] == 1

    def test_cache_clear(self):
        """Test cache clear"""
        cache = MemoizationCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert len(cache._cache) == 0
        assert cache._hits == 0
        assert cache._misses == 0


class TestMemoizeDecorator:
    """Test memoize decorator"""

    def test_memoize_basic(self):
        """Test basic memoization"""
        call_count = 0

        @memoize(max_size=100, ttl_seconds=3600)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call should execute
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same args should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

    def test_memoize_with_kwargs(self):
        """Test memoization with kwargs"""
        call_count = 0

        @memoize()
        def func_with_kwargs(x, y=10):
            nonlocal call_count
            call_count += 1
            return x + y

        func_with_kwargs(5, y=20)
        func_with_kwargs(5, y=20)

        assert call_count == 1

    def test_memoize_with_different_args(self):
        """Test memoization with different arguments"""
        call_count = 0

        @memoize()
        def func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        func(5)
        func(10)

        assert call_count == 2


class TestTimedExecution:
    """Test timed_execution decorator"""

    def test_timed_execution_basic(self):
        """Test basic timed execution"""

        @timed_execution
        def slow_function():
            time.sleep(0.1)
            return 42

        result = slow_function()
        assert result == 42

    def test_timed_execution_with_args(self):
        """Test timed execution with arguments"""

        @timed_execution
        def add_function(x, y):
            return x + y

        result = add_function(3, 5)
        assert result == 8


class TestParallelMap:
    """Test parallel_map function"""

    def test_parallel_map_basic(self):
        """Test basic parallel map"""

        def square(x):
            return x * x

        inputs = [1, 2, 3, 4, 5]
        results = parallel_map(square, inputs, max_workers=2)

        assert results == [1, 4, 9, 16, 25]

    def test_parallel_map_empty_list(self):
        """Test parallel map with empty list"""

        def identity(x):
            return x

        results = parallel_map(identity, [])
        assert results == []

    def test_parallel_map_with_numpy(self):
        """Test parallel map with numpy operations"""

        def process_array(x):
            return np.mean(x)

        inputs = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        results = parallel_map(process_array, inputs, max_workers=2)

        assert len(results) == 3
        assert all(isinstance(r, (float, np.floating)) for r in results)


class TestPerformanceOptimizerEdgeCases:
    """Test edge cases"""

    def test_cache_thread_safety(self):
        """Test cache thread safety"""
        cache = MemoizationCache()
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f"key_{i}", f"value_{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"key_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_cache_with_unhashable_types(self):
        """Test cache handling of unhashable types"""

        # Should handle list arguments by converting to string representation
        @memoize()
        def func_with_list(lst):
            return sum(lst)

        result = func_with_list([1, 2, 3])
        assert result == 6
