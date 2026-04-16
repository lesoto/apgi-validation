"""
APGI Performance Optimizer
==========================

Protocol execution speed improvements with caching,
memoization, and parallel execution capabilities.
"""

import functools
import hashlib
import json
import multiprocessing
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from utils.logging_config import apgi_logger
except ImportError:
    apgi_logger = None


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    execution_time: float
    memory_used_mb: float
    cpu_percent: float
    cache_hits: int = 0
    cache_misses: int = 0


class MemoizationCache:
    """Thread-safe memoization cache with size limits."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from function arguments."""
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": kwargs,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    self._hits += 1
                    return value
                else:
                    # Expired
                    del self._cache[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any):
        """Set value in cache with LRU eviction."""
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (value, time.time())

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "hit_rate": hit_rate,
            }

    def clear(self):
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# Global cache instance
_global_cache = MemoizationCache()


def memoize(
    max_size: int = 1000, ttl_seconds: float = 3600, cache_key: Optional[str] = None
):
    """
    Decorator for memoizing function results.

    Args:
        max_size: Maximum cache entries
        ttl_seconds: Time-to-live for cached entries
        cache_key: Optional specific cache key to use

    Returns:
        Decorated function with caching
    """

    def decorator(func: Callable) -> Callable:
        cache = (
            _global_cache
            if cache_key is None
            else MemoizationCache(max_size, ttl_seconds)
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = cache._make_key(func.__name__, args, kwargs)
            cached_result = cache.get(key)

            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        # Attach cache reference for stats access
        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator


class ProtocolOptimizer:
    """Optimizer for protocol execution performance."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._performance_log: List[Dict] = []

    def optimize_numpy_operations(self, data: np.ndarray) -> np.ndarray:
        """
        Optimize NumPy operations with memory alignment and vectorization.

        Args:
            data: Input numpy array

        Returns:
            Optimized result
        """
        # Ensure C-contiguous memory layout for faster operations
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)

        # Use in-place operations where possible
        return data

    @memoize(max_size=500, ttl_seconds=1800)
    def cached_statistical_computation(
        self, data: Tuple, computation_type: str
    ) -> Dict[str, float]:
        """
        Cached statistical computations to avoid recomputation.

        Args:
            data: Tuple of data points (hashable)
            computation_type: Type of computation to perform

        Returns:
            Dictionary of computed statistics
        """
        arr = np.array(data)

        if computation_type == "basic":
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        elif computation_type == "advanced":
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "percentile_25": float(np.percentile(arr, 25)),
                "percentile_75": float(np.percentile(arr, 75)),
            }
        else:
            return {"mean": float(np.mean(arr))}

    def parallel_protocol_execution(
        self, protocols: List[Callable], *args, **kwargs
    ) -> List[Any]:
        """
        Execute multiple protocols in parallel using thread pool.

        Args:
            protocols: List of protocol functions to execute
            *args: Arguments to pass to each protocol
            **kwargs: Keyword arguments to pass to each protocol

        Returns:
            List of results from each protocol
        """

        def execute_protocol(protocol):
            try:
                start_time = time.time()
                result = protocol(*args, **kwargs)
                execution_time = time.time() - start_time

                self._performance_log.append(
                    {
                        "protocol": protocol.__name__,
                        "execution_time": execution_time,
                        "timestamp": time.time(),
                    }
                )
                return result
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(
                        f"Protocol {protocol.__name__} failed: {e}"
                    )
                return None

        # Submit all protocols to thread pool
        futures = [self._thread_pool.submit(execute_protocol, p) for p in protocols]
        results = [f.result() for f in futures]

        return results

    def batch_process_with_chunking(
        self,
        data: List[Any],
        process_func: Callable,
        chunk_size: int = 100,
        parallel: bool = True,
        cpu_bound: bool = False,
    ) -> List[Any]:
        """
        Process large datasets in chunks for memory efficiency.

        Args:
            data: Large dataset to process
            process_func: Function to apply to each chunk
            chunk_size: Size of each chunk
            parallel: Whether to process chunks in parallel
            cpu_bound: Use process pool for CPU-bound workloads

        Returns:
            Combined results from all chunks
        """
        adaptive_chunk_size = max(
            25, min(chunk_size, max(1, len(data) // max(1, self.max_workers)))
        )
        chunks = [
            data[i : i + adaptive_chunk_size]
            for i in range(0, len(data), adaptive_chunk_size)
        ]

        if parallel and len(chunks) > 1:
            if cpu_bound:
                if self._process_pool is None:
                    self._process_pool = ProcessPoolExecutor(
                        max_workers=self.max_workers
                    )
                futures = [
                    self._process_pool.submit(
                        self._process_chunk_static, chunk, process_func
                    )
                    for chunk in chunks
                ]
            else:
                futures = [
                    self._thread_pool.submit(process_func, chunk) for chunk in chunks
                ]
            results = [f.result() for f in futures]
        else:
            # Process sequentially
            results = [process_func(chunk) for chunk in chunks]

        # Flatten results if they're lists
        if results and isinstance(results[0], list):
            return [item for sublist in results for item in sublist]
        return results

    @staticmethod
    def _process_chunk_static(chunk: List[Any], process_func: Callable) -> List[Any]:
        """Static helper for process-pool compatibility."""
        return process_func(chunk)

    def lazy_evaluation(
        self, computation_chain: List[Callable], initial_data: Any
    ) -> Any:
        """
        Execute computation chain with lazy evaluation.

        Args:
            computation_chain: List of functions to apply sequentially
            initial_data: Starting data

        Returns:
            Final computed result
        """
        result = initial_data
        for func in computation_chain:
            result = func(result)
        return result

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate performance report from logged metrics.

        Returns:
            Dictionary with performance statistics
        """
        if not self._performance_log:
            return {"status": "no_data"}

        execution_times = [p["execution_time"] for p in self._performance_log]

        return {
            "total_executions": len(self._performance_log),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "cache_stats": _global_cache.get_stats(),
        }

    def shutdown(self):
        """Clean up resources."""
        self._thread_pool.shutdown(wait=True)
        if self._process_pool:
            self._process_pool.shutdown(wait=True)


class FastProtocolRunner:
    """Fast runner for APGI validation protocols with optimizations."""

    def __init__(self):
        self.optimizer = ProtocolOptimizer()
        self._warmup_complete = False

    def warmup(self):
        """Warm up caches and JIT compilation where applicable."""
        if self._warmup_complete:
            return

        # Warm up NumPy
        _ = np.random.rand(100, 100)
        _ = np.dot(_, _.T)

        # Warm up cache with common computations
        sample_data = tuple(np.random.randn(100))
        self.optimizer.cached_statistical_computation(sample_data, "basic")
        self.optimizer.cached_statistical_computation(sample_data, "advanced")

        self._warmup_complete = True
        if apgi_logger:
            apgi_logger.logger.info("Protocol runner warmup complete")

    def run_optimized_protocol(
        self,
        protocol_func: Callable,
        *args,
        use_cache: bool = True,
        parallel_chunks: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Any, PerformanceMetrics]:
        """
        Run a protocol with full optimizations.

        Args:
            protocol_func: Protocol function to execute
            *args: Arguments for the protocol
            use_cache: Whether to use memoization
            parallel_chunks: Chunk size for parallel processing
            **kwargs: Keyword arguments for the protocol

        Returns:
            Tuple of (result, performance_metrics)
        """
        import psutil

        # Warm up on first run
        if not self._warmup_complete:
            self.warmup()

        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.time()
        cache_hits_before = _global_cache._hits

        try:
            if use_cache:
                # Try cache first
                key = _global_cache._make_key(protocol_func.__name__, args, kwargs)
                cached = _global_cache.get(key)
                if cached is not None:
                    execution_time = time.time() - start_time
                    mem_after = process.memory_info().rss / (1024 * 1024)
                    return cached, PerformanceMetrics(
                        execution_time=execution_time,
                        memory_used_mb=mem_after - mem_before,
                        cpu_percent=process.cpu_percent(),
                        cache_hits=1,
                        cache_misses=0,
                    )

            # Execute protocol
            result = protocol_func(*args, **kwargs)

            # Cache result
            if use_cache:
                _global_cache.set(key, result)

            execution_time = time.time() - start_time
            mem_after = process.memory_info().rss / (1024 * 1024)
            cache_hits_after = _global_cache._hits

            return result, PerformanceMetrics(
                execution_time=execution_time,
                memory_used_mb=mem_after - mem_before,
                cpu_percent=process.cpu_percent(),
                cache_hits=cache_hits_after - cache_hits_before,
                cache_misses=1 if use_cache else 0,
            )

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Protocol execution failed: {e}")
            raise


# Convenience functions
def optimized_run(func: Callable, *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
    """Quick optimized execution of a function."""
    runner = FastProtocolRunner()
    return runner.run_optimized_protocol(func, *args, **kwargs)


def clear_performance_cache():
    """Clear global performance cache."""
    _global_cache.clear()


def get_cache_statistics() -> Dict[str, Any]:
    """Get current cache statistics."""
    return _global_cache.get_stats()


if __name__ == "__main__":
    # Demo performance optimization
    print("APGI Performance Optimizer Demo")
    print("=" * 40)

    optimizer = ProtocolOptimizer()

    # Sample data
    data = list(np.random.randn(1000))

    # Cached computation
    print("\n1. Cached Statistical Computation:")
    result1 = optimizer.cached_statistical_computation(tuple(data), "basic")
    print(f"   Result: {result1}")

    # Same computation - should be cached
    result2 = optimizer.cached_statistical_computation(tuple(data), "basic")
    print(f"   Cache stats: {get_cache_statistics()}")

    # Batch processing
    print("\n2. Batch Processing:")
    large_data = list(range(10000))

    def process_chunk(chunk):
        return [x * 2 for x in chunk]

    results = optimizer.batch_process_with_chunking(
        large_data, process_chunk, chunk_size=1000, parallel=True
    )
    print(f"   Processed {len(results)} items")

    # Performance report
    print("\n3. Performance Report:")
    report = optimizer.get_performance_report()
    print(f"   {report}")

    print("\nPerformance optimization module ready!")
