"""Tests for performance SLO governance utilities."""

from pathlib import Path

import pytest

from utils.performance_governance import (
    DEFAULT_SLOS,
    BenchmarkResult,
    PerformanceRegressionError,
    assert_slo,
    benchmark_callable,
    export_benchmark_report,
)


def test_benchmark_callable_produces_metrics():
    result = benchmark_callable("cache_get_set", lambda: sum([1, 2, 3]), iterations=5)
    assert result.protocol_id == "cache_get_set"
    assert len(result.latencies_ms) == 5
    assert result.throughput_ops_per_sec > 0


def test_assert_slo_passes_for_fast_result():
    result = BenchmarkResult(
        protocol_id="cache_get_set",
        latencies_ms=[1.0, 2.0, 2.5, 3.0],
        throughput_ops_per_sec=1000.0,
        peak_memory_mb=10.0,
    )
    assert_slo(result, DEFAULT_SLOS["cache_get_set"])


def test_assert_slo_raises_for_regression():
    result = BenchmarkResult(
        protocol_id="validation_pipeline_prepare",
        latencies_ms=[5000.0, 6000.0, 7000.0],
        throughput_ops_per_sec=0.2,
        peak_memory_mb=500.0,
    )
    with pytest.raises(PerformanceRegressionError):
        assert_slo(result, DEFAULT_SLOS["validation_pipeline_prepare"])


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
