"""Tests for performance SLO governance utilities."""

from pathlib import Path

import pytest

from utils.performance_governance import (
    BenchmarkResult,
    PerformanceRegressionError,
    benchmark_callable,
    check_performance_regression,
    export_benchmark_report,
)


def test_benchmark_callable_produces_metrics():
    result = benchmark_callable("cache_get_set", lambda: sum([1, 2, 3]), iterations=5)
    assert result.protocol_id == "cache_get_set"
    assert len(result.latencies_ms) == 5
    assert result.throughput_ops_per_sec > 0


def test_assert_slo_passes_for_fast_result():
    BenchmarkResult(
        protocol_id="cache_get_set",
        latencies_ms=[1.0, 2.0, 2.5, 3.0],
        throughput_ops_per_sec=1000.0,
        peak_memory_mb=10.0,
    )


def test_assert_slo_raises_for_regression():
    baseline = BenchmarkResult(
        protocol_id="validation_pipeline_prepare",
        latencies_ms=[1000.0, 1200.0, 1100.0],
        throughput_ops_per_sec=10.0,
        peak_memory_mb=100.0,
    )
    current = BenchmarkResult(
        protocol_id="validation_pipeline_prepare",
        latencies_ms=[5000.0, 6000.0, 7000.0],
        throughput_ops_per_sec=0.2,
        peak_memory_mb=500.0,
    )
    with pytest.raises(PerformanceRegressionError):
        check_performance_regression(baseline, current, threshold_pct=10.0)


def test_export_benchmark_report(tmp_path: Path):
    output = tmp_path / "benchmark_report.json"
    result = BenchmarkResult(
        protocol_id="cache_get_set",
        latencies_ms=[1.0, 2.0, 3.0],
        throughput_ops_per_sec=200.0,
        peak_memory_mb=64.0,
    )
    export_benchmark_report([result], output)
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "cache_get_set" in text
