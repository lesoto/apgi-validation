"""Performance SLO governance and regression checks for APGI protocols."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List


@dataclass(frozen=True)
class ProtocolSLO:
    protocol_id: str
    max_p95_latency_ms: float
    min_throughput_ops_per_sec: float


DEFAULT_SLOS: Dict[str, ProtocolSLO] = {
    "validation_pipeline_prepare": ProtocolSLO(
        protocol_id="validation_pipeline_prepare",
        max_p95_latency_ms=2000.0,
        min_throughput_ops_per_sec=5.0,
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
    protocol_id: str
    latencies_ms: List[float]
    throughput_ops_per_sec: float

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_vals = sorted(self.latencies_ms)
        idx = min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1)))
        return sorted_vals[idx]


def benchmark_callable(
    protocol_id: str, fn: Callable[[], object], iterations: int = 25
) -> BenchmarkResult:
    latencies_ms: List[float] = []
    start = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - start
    throughput = iterations / elapsed if elapsed > 0 else 0.0
    return BenchmarkResult(
        protocol_id=protocol_id,
        latencies_ms=latencies_ms,
        throughput_ops_per_sec=throughput,
    )


def assert_slo(result: BenchmarkResult, slo: ProtocolSLO) -> None:
    if result.p95_latency_ms > slo.max_p95_latency_ms:
        raise PerformanceRegressionError(
            f"{result.protocol_id} p95 latency {result.p95_latency_ms:.2f}ms exceeds {slo.max_p95_latency_ms}ms"
        )
    if result.throughput_ops_per_sec < slo.min_throughput_ops_per_sec:
        raise PerformanceRegressionError(
            f"{result.protocol_id} throughput {result.throughput_ops_per_sec:.2f} < {slo.min_throughput_ops_per_sec}"
        )


def export_benchmark_report(
    results: Iterable[BenchmarkResult], output_file: Path
) -> None:
    payload = []
    for result in results:
        payload.append(
            {
                "protocol_id": result.protocol_id,
                "mean_latency_ms": (
                    mean(result.latencies_ms) if result.latencies_ms else 0.0
                ),
                "p95_latency_ms": result.p95_latency_ms,
                "throughput_ops_per_sec": result.throughput_ops_per_sec,
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
