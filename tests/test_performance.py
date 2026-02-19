"""
Performance tests for APGI validation framework.
===============================================

Benchmarks for simulation, validation, and data processing performance.
"""

import pytest

from utils import data_validation, sample_data_generator


def test_simulation_performance():
    """Benchmark simulation performance."""
    # This would benchmark simulation execution time
    # For now, just a placeholder

    def run_simulation():
        generator = sample_data_generator.SampleDataGenerator(
            sampling_rate=1000, duration=1
        )
        eeg_signal, p300_events = generator.generate_eeg_data()
        return len(eeg_signal), len(p300_events)

    signal_length, event_count = run_simulation()

    assert signal_length == 1000  # 1000 Hz * 1 second
    assert event_count > 0


def test_data_generation_performance():
    """Benchmark data generation performance."""

    def generate_data():
        return sample_data_generator.generate_sample_multimodal_data(
            n_samples=1000, sampling_rate=100.0, duration_minutes=0.1
        )

    df = generate_data()

    assert len(df) == 1000
    assert "EEG_Cz" in df.columns
    assert "pupil_diameter" in df.columns


def test_data_validation_performance():
    """Benchmark data validation performance."""
    # Generate test data first
    df = sample_data_generator.generate_sample_multimodal_data(
        n_samples=1000, sampling_rate=100.0, duration_minutes=0.1
    )

    validator = data_validation.DataValidator()

    def validate_data():
        return validator.validate_data_quality(df)

    quality_report = validate_data()

    assert "overall_score" in quality_report
    assert quality_report["overall_score"] > 0


def test_batch_processing_performance():
    """Benchmark batch processing performance."""
    from utils.batch_processor import BatchProcessor

    def run_batch():
        processor = BatchProcessor(max_workers=1, use_processes=False)

        # Add multiple simulation jobs
        for i in range(3):
            processor.add_simulation_job(
                job_id=f"perf_sim_{i}",
                params={"tau_S": 0.5, "alpha": 5.0},
                steps=50,  # Smaller for performance test
                dt=0.1,
                output_file=None,
            )

        results = processor.run_batch(show_progress=False)
        return results

    results = run_batch()

    assert results["total_jobs"] == 3
    assert results["completed"] == 3


def test_cache_performance():
    """Benchmark caching performance."""
    from utils.cache_manager import CacheManager

    cache = CacheManager(max_size_mb=50)

    def cache_operation():
        # Store and retrieve data
        test_data = {"data": list(range(1000)), "metadata": {"size": 1000}}

        cache.set("perf_test", test_data)
        retrieved = cache.get("perf_test")

        return retrieved == test_data

    success = cache_operation()
    assert success


def test_memory_usage():
    """Test memory usage of key operations."""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Perform memory-intensive operation
    data_df = sample_data_generator.generate_sample_multimodal_data(
        n_samples=10000, sampling_rate=100.0, duration_minutes=1
    )

    from utils import data_validation

    validator = data_validation.DataValidator()
    quality_report = validator.validate_data_quality(data_df)

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory

    # Memory usage should be reasonable (< 500 MB increase)
    assert memory_used < 500, f"Memory usage too high: {memory_used:.1f} MB"
    assert "overall_score" in quality_report


@pytest.mark.slow
def test_large_scale_simulation():
    """Test performance with larger simulation."""
    import time

    start_time = time.time()

    # Large simulation
    generator = sample_data_generator.SampleDataGenerator(
        sampling_rate=1000, duration=30
    )
    eeg_signal, p300_events = generator.generate_eeg_data()

    duration = time.time() - start_time

    # Should complete in reasonable time (< 10 seconds)
    assert duration < 10.0, f"Simulation too slow: {duration:.2f} seconds"
    assert len(eeg_signal) == 30000
    assert len(p300_events) > 5
