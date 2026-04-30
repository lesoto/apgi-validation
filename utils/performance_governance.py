"""Performance SLO governance and regression checks for APGI protocols."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Optional

import psutil


@dataclass(frozen=True)
class ProtocolSLO:
    """Service Level Objective for a specific protocol or operation."""

    protocol_id: str
    max_p95_latency_ms: float
    min_throughput_ops_per_sec: float
    max_memory_mb: float = 1024.0  # Default 1GB
    critical: bool = True


DEFAULT_SLOS: Dict[str, ProtocolSLO] = {
    "VP_01_SyntheticEEG": ProtocolSLO(
        protocol_id="VP_01_SyntheticEEG",
        max_p95_latency_ms=60000.0,  # 60s
        min_throughput_ops_per_sec=0.01,
        max_memory_mb=4096.0,  # 4GB
    ),
    "VP_02_Behavioral": ProtocolSLO(
        protocol_id="VP_02_Behavioral",
        max_p95_latency_ms=30000.0,  # 30s
        min_throughput_ops_per_sec=0.03,
        max_memory_mb=2048.0,  # 2GB
    ),
    "validation_pipeline_prepare": ProtocolSLO(
        protocol_id="validation_pipeline_prepare",
        max_p95_latency_ms=2000.0,
        min_throughput_ops_per_sec=0.5,
    ),
    "cache_get_set": ProtocolSLO(
        protocol_id="cache_get_set",
        max_p95_latency_ms=50.0,
        min_throughput_ops_per_sec=500.0,
    ),
}


class PerformanceRegressionError(RuntimeError):
    """Raised when benchmark metrics violate declared SLOs."""


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark run."""

    protocol_id: str
    latencies_ms: List[float]
    throughput_ops_per_sec: float
    peak_memory_mb: float
    timestamp: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_vals = sorted(self.latencies_ms)
        idx = min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1)))
        return sorted_vals[idx]

    @property
    def mean_latency_ms(self) -> float:
        return mean(self.latencies_ms) if self.latencies_ms else 0.0


def benchmark_callable(
    protocol_id: str, fn: Callable[[], object], iterations: int = 5
) -> BenchmarkResult:
    """Benchmark a callable and return metrics."""
    latencies_ms: List[float] = []
    process = psutil.Process(os.getpid())
    peak_memory = 0.0

    start_total = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()

        # Sample memory after
        mem_after = process.memory_info().rss / (1024 * 1024)
        peak_memory = max(peak_memory, mem_after)

        latencies_ms.append((t1 - t0) * 1000)

    elapsed_total = time.perf_counter() - start_total
    throughput = iterations / elapsed_total if elapsed_total > 0 else 0.0

    return BenchmarkResult(
        protocol_id=protocol_id,
        latencies_ms=latencies_ms,
        throughput_ops_per_sec=throughput,
        peak_memory_mb=peak_memory,
    )


def assert_slo(result: BenchmarkResult, slo: Optional[ProtocolSLO] = None) -> None:
    """Assert that a benchmark result meets its SLO."""
    if slo is None:
        slo = DEFAULT_SLOS.get(result.protocol_id)

    if slo is None:
        return  # No SLO defined for this protocol

    errors = []
    if result.p95_latency_ms > slo.max_p95_latency_ms:
        errors.append(
            f"p95 latency {result.p95_latency_ms:.2f}ms exceeds {slo.max_p95_latency_ms}ms"
        )

    if result.throughput_ops_per_sec < slo.min_throughput_ops_per_sec:
        errors.append(
            f"throughput {result.throughput_ops_per_sec:.2f} < {slo.min_throughput_ops_per_sec}"
        )

    if result.peak_memory_mb > slo.max_memory_mb:
        errors.append(
            f"peak memory {result.peak_memory_mb:.2f}MB exceeds {slo.max_memory_mb}MB"
        )

    if errors:
        msg = f"Performance regression in {result.protocol_id}: " + "; ".join(errors)
        if slo.critical:
            raise PerformanceRegressionError(msg)
        else:
            print(f"WARNING: {msg}")


def check_all_slos(results: List[BenchmarkResult]) -> bool:
    """Check all results against DEFAULT_SLOS. Returns False if any critical fails."""
    all_passed = True
    for result in results:
        try:
            assert_slo(result)
        except PerformanceRegressionError as e:
            print(f"CRITICAL SLO FAILURE: {e}")
            all_passed = False
    return all_passed


def export_benchmark_report(
    results: Iterable[BenchmarkResult], output_file: Path
) -> None:
    """Export benchmark results to a JSON report."""
    payload = []
    for result in results:
        payload.append(
            {
                "protocol_id": result.protocol_id,
                "timestamp": result.timestamp,
                "mean_latency_ms": result.mean_latency_ms,
                "p95_latency_ms": result.p95_latency_ms,
                "throughput_ops_per_sec": result.throughput_ops_per_sec,
                "peak_memory_mb": result.peak_memory_mb,
                "iterations": len(result.latencies_ms),
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
