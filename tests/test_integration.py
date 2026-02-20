"""
Integration tests for APGI validation framework.
===============================================

Tests full protocol execution and end-to-end workflows.
"""

import warnings
from pathlib import Path

import pytest

# Suppress scipy overflow warnings in tests
warnings.filterwarnings(
    "ignore", message="overflow encountered in vecdot", category=RuntimeWarning
)

# Add the project root to Python path

from utils import sample_data_generator


@pytest.mark.integration
def test_validation_protocol_9_integration():
    """Test full execution of Validation Protocol 9."""
    try:
        import importlib.util
        from pathlib import Path

        # Load the module with hyphen in filename
        module_path = (
            Path(__file__).parent.parent / "Validation" / "Validation-Protocol-9.py"
        )
        spec = importlib.util.spec_from_file_location(
            "validation_protocol_9", module_path
        )
        if spec and spec.loader:
            validation_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(validation_module)
            APGINeuralSignaturesValidator = (
                validation_module.APGINeuralSignaturesValidator
            )
        else:
            raise ImportError("Could not load Validation Protocol 9")

        validator = APGINeuralSignaturesValidator()

        # Suppress scipy warnings during validation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # Test with None paths (should load from data_repository or fail gracefully)
            results = validator.validate_convergent_signatures()

        # Should return results dict even if data not available
        assert isinstance(results, dict)
        assert "overall_validation_score" in results

    except ImportError:
        pytest.skip("Validation Protocol 9 not available")


@pytest.mark.integration
def test_data_repository_integration(temp_dir):
    """Test data loading from data_repository."""
    PROJECT_ROOT = Path(__file__).parent.parent

    # Data repository paths
    DATA_REPO = PROJECT_ROOT / "data_repository"
    RAW_DATA_DIR = DATA_REPO / "raw_data"
    PROCESSED_DATA_DIR = DATA_REPO / "processed_data"

    # Check that data directories exist
    assert RAW_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()

    # Test creating a sample data file in temp directory
    sample_file = temp_dir / "test_behavioral.csv"
    # Create sample behavioral data
    import pandas as pd

    sample_data = pd.DataFrame(
        {
            "stimulus_surprisal": [0.1, 0.2, 0.3],
            "precision_e": [1.0, 1.1, 0.9],
            "error_e": [0.0, 0.1, -0.1],
            "precision_i": [1.0, 0.9, 1.2],
            "error_i": [0.0, -0.1, 0.1],
            "threshold": [0.5, 0.5, 0.5],
            "alpha": [5.0, 5.0, 5.0],
        }
    )
    sample_data.to_csv(sample_file, index=False)

    assert sample_file.exists()


@pytest.mark.integration
def test_full_validation_pipeline():
    """Test complete validation protocol execution."""
    # This is a placeholder for a full integration test
    # In a real implementation, this would:
    # 1. Load a validation protocol
    # 2. Generate sample data
    # 3. Run the protocol
    # 4. Verify results

    # For now, just test that we can generate sample data
    generator = sample_data_generator.SampleDataGenerator(sampling_rate=100, duration=5)
    eeg_signal, p300_events = generator.generate_eeg_data()

    assert len(eeg_signal) == 500  # 100 Hz * 5 seconds
    assert isinstance(p300_events, list)
    assert len(p300_events) > 0  # Should have some P300 events


@pytest.mark.integration
def test_batch_processing_integration():
    """Test batch processing with multiple jobs."""
    from utils.batch_processor import BatchProcessor

    # Create batch processor with single worker and threads to avoid hanging
    processor = BatchProcessor(max_workers=1, use_processes=False)

    # Add a simple simulation job
    processor.add_simulation_job(
        job_id="test_batch_job", params={"tau_S": 0.5, "alpha": 5.0}, steps=10, dt=0.1
    )

    # Run batch
    results = processor.run_batch(show_progress=False)

    # Verify results
    assert results["total_jobs"] == 1
    assert results["completed"] == 1
    assert results["failed"] == 0
    assert len(results["jobs"]) == 1

    job_result = results["jobs"][0]
    assert job_result["job_id"] == "test_batch_job"
    assert job_result["status"] == "completed"
    assert "result" in job_result
    assert "summary" in job_result["result"]


@pytest.mark.integration
def test_data_processing_pipeline():
    """Test complete data processing pipeline."""
    from utils import data_validation, sample_data_generator

    # Generate sample data
    data_df = sample_data_generator.generate_sample_multimodal_data(
        n_samples=1000, sampling_rate=100.0, duration_minutes=1
    )

    # Create validator
    validator = data_validation.DataValidator()

    # Validate file format (using a temporary CSV)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
        data_df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)

        # Validate format
        format_result = validator.validate_file_format(tmp_path)
        assert format_result["is_readable"] is True
        assert format_result["format_valid"] is True

        # Assess quality
        quality_result = validator.validate_data_quality(data_df)
        assert "overall_score" in quality_result
        assert quality_result["overall_score"] >= 0

        # Clean up
        tmp_path.unlink()


@pytest.mark.integration
def test_config_management_integration():
    """Test configuration management end-to-end."""
    from utils.config_manager import ConfigManager

    # Create config manager
    manager = ConfigManager()

    # Test getting config
    config = manager.get_config()
    assert config is not None
    assert hasattr(config, "model")
    assert hasattr(config, "simulation")

    # Test setting parameter
    original_value = config.model.tau_S
    manager.set_parameter("model", "tau_S", 0.8)
    assert config.model.tau_S == 0.8

    # Reset to defaults
    manager.reset_to_defaults("model")
    assert config.model.tau_S == original_value


@pytest.mark.integration
def test_cache_integration():
    """Test caching system with data processing."""
    import tempfile

    from utils.cache_manager import CacheManager

    # Create cache manager with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = CacheManager(cache_dir=temp_dir, max_size_mb=10)

        # Test basic caching
        test_key = "test_key"
        test_value = {"data": [1, 2, 3], "metadata": {"type": "test"}}

        # Store value
        cache.set(test_key, test_value)

        # Retrieve value
        retrieved = cache.get(test_key)
        assert retrieved == test_value

        # Test cache statistics
        stats = cache.get_stats()
        assert stats["total_requests"] > 0
        assert stats["hits"] > 0


@pytest.mark.integration
def test_end_to_end_workflow():
    """Test complete end-to-end workflow from data generation to validation."""
    # Generate sample data
    data_df = sample_data_generator.generate_sample_multimodal_data(
        n_samples=500, sampling_rate=50.0, duration_minutes=0.5
    )

    # Process data
    from utils import data_validation

    validator = data_validation.DataValidator()

    # Validate and assess quality
    quality_report = validator.validate_data_quality(data_df)

    # Verify report structure
    assert "overall_score" in quality_report
    assert "missing_data" in quality_report
    assert "outliers" in quality_report
    assert "signal_quality" in quality_report

    # Generate validation report
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
        data_df.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)

        validation_report = validator.generate_validation_report(tmp_path)

        assert "file_info" in validation_report
        assert "data_quality" in validation_report
        assert "recommendations" in validation_report

        # Clean up
        tmp_path.unlink()
