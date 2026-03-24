"""
Integration tests for APGI validation framework.
===============================================

Tests full protocol execution and end-to-end workflows.
"""

import warnings
from pathlib import Path

import numpy as np
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
            Path(__file__).parent.parent / "Validation" / "Validation_Protocol_9.py"
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

        # Generate sample data for testing
        import numpy as np

        sample_data = {
            "eeg_signals": np.random.randn(100, 1000),  # 100 trials, 1000 time points
            "behavioral_data": {
                "response_times": np.random.uniform(0.2, 1.0, 100),
                "accuracy": np.random.choice([0, 1], 100),
            },
            "physiological_data": {
                "pupil_diameter": np.random.normal(3.0, 0.5, 100),
                "heart_rate": np.random.normal(70, 10, 100),
            },
        }

        # Test with sample data instead of None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            results = validator.validate_convergent_signatures(sample_data)

        # Should return results dict
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
    try:
        import importlib.util
        from pathlib import Path

        # Load Validation Protocol 1 module
        module_path = (
            Path(__file__).parent.parent / "Validation" / "Validation_Protocol_1.py"
        )
        spec = importlib.util.spec_from_file_location(
            "validation_protocol_1", module_path
        )
        if spec and spec.loader:
            validation_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(validation_module)
            APGIDatasetGenerator = validation_module.APGIDatasetGenerator
        else:
            raise ImportError("Could not load Validation Protocol 1")

        # Generate small dataset for integration testing
        generator = APGIDatasetGenerator(fs=100)  # Lower sampling rate for faster test
        dataset = generator.generate_dataset(n_trials_per_model=10, save_path=None)

        # Verify dataset structure
        assert "eeg" in dataset
        assert "hep" in dataset
        assert "pupil" in dataset
        assert "ignition_labels" in dataset
        assert "S_values" in dataset
        assert "model_labels" in dataset
        assert "model_names" in dataset

        # Verify shapes
        expected_total_trials = 10 * 4  # 10 per model × 4 models
        assert dataset["eeg"].shape[0] == expected_total_trials
        assert dataset["hep"].shape[0] == expected_total_trials
        assert dataset["pupil"].shape[0] == expected_total_trials
        assert len(dataset["ignition_labels"]) == expected_total_trials
        assert len(dataset["S_values"]) == expected_total_trials
        assert len(dataset["model_labels"]) == expected_total_trials
        assert len(dataset["model_names"]) == expected_total_trials

        # Verify model diversity
        unique_models = set(dataset["model_names"])
        assert len(unique_models) == 4  # All 4 models should be present
        expected_models = {"APGI", "StandardPP", "GWTOnly", "Continuous"}
        assert unique_models == expected_models

        # Verify data types
        assert dataset["eeg"].dtype == np.float32 or dataset["eeg"].dtype == np.float64
        assert (
            dataset["ignition_labels"].dtype == np.int64
            or dataset["ignition_labels"].dtype == np.int32
        )

    except ImportError:
        pytest.skip("Validation Protocol 1 not available")


@pytest.mark.integration
def test_batch_processing_integration():
    """Test batch processing with multiple jobs."""
    from utils.batch_processor import BatchProcessor

    # Create batch processor with single worker and threads to avoid hanging
    processor = BatchProcessor(max_workers=1, use_processes=False)

    # Add multiple simulation jobs to test batch processing
    jobs = [
        {
            "job_id": "test_batch_job_1",
            "params": {"tau_S": 0.5, "alpha": 5.0},
            "steps": 10,
            "dt": 0.1,
        },
        {
            "job_id": "test_batch_job_2",
            "params": {"tau_S": 0.6, "alpha": 6.0},
            "steps": 15,
            "dt": 0.1,
        },
        {
            "job_id": "test_batch_job_3",
            "params": {"tau_S": 0.7, "alpha": 7.0},
            "steps": 20,
            "dt": 0.1,
        },
    ]

    for job in jobs:
        processor.add_simulation_job(**job)

    # Run batch
    results = processor.run_batch(show_progress=False)

    # Verify results for multiple jobs
    assert results["total_jobs"] == len(jobs)
    assert results["completed"] == len(jobs)
    assert results["failed"] == 0
    assert len(results["jobs"]) == len(jobs)

    for i, job_result in enumerate(results["jobs"]):
        expected_job = jobs[i]
        assert job_result["job_id"] == expected_job["job_id"]
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


@pytest.fixture
def isolated_config_manager():
    """Provide an isolated config manager for each test."""
    from utils.config_manager import ConfigManager
    import tempfile
    import os

    # Create a temporary config file for this test
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_config_path = f.name

    # Create manager with isolated config
    manager = ConfigManager(config_file=temp_config_path)

    yield manager

    # Cleanup
    try:
        os.unlink(temp_config_path)
    except FileNotFoundError:
        pass


@pytest.mark.integration
def test_config_management_integration(isolated_config_manager):
    """Test configuration management end-to-end."""

    # Use the isolated config manager
    manager = isolated_config_manager

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

    # Reset all configurations to ensure test isolation
    manager.reset_to_defaults()


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


@pytest.mark.integration
def test_run_comprehensive_falsification_pipeline():
    """Test full run_comprehensive_falsification pipeline across all priorities."""
    import sys
    from pathlib import Path

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from APGI_Falsification_Framework import FalsificationFramework

    # Create falsification framework instance
    framework = FalsificationFramework()

    # Generate test data for all priorities
    # Create sample data that matches the expected structure
    all_test_data = {
        "P1": {
            "advantage_metric": np.array([25.0, 22.0, 28.0, 24.0, 26.0]),
            "comparison_metric": np.array([5.0, 4.0, 6.0, 5.0, 5.5]),
            "effect_size": np.array([0.75, 0.68, 0.82, 0.70, 0.78]),
        },
        "P2": {
            "pp_difference": np.array([12.0, 11.0, 13.0, 12.5, 11.5]),
            "cohens_h": np.array([0.60, 0.55, 0.65, 0.58, 0.62]),
            "correlation": np.array([0.45, 0.42, 0.48, 0.44, 0.46]),
            "rt_advantage": np.array([60.0, 55.0, 65.0, 58.0, 62.0]),
            "beta_interaction": np.array([0.38, 0.35, 0.41, 0.36, 0.39]),
        },
        "P3": {
            "intero_advantage": np.array([30.0, 28.0, 32.0, 29.0, 31.0]),
            "reduction_metric": np.array([27.0, 25.0, 29.0, 26.0, 28.0]),
            "cohens_d": np.array([0.72, 0.68, 0.76, 0.70, 0.74]),
        },
    }

    # Run comprehensive falsification
    results = framework.run_comprehensive_falsification(all_test_data)

    # Verify results structure
    assert isinstance(results, dict)
    assert "priority_results" in results
    assert "overall_falsification" in results
    assert "falsification_summary" in results
    assert "theory_status" in results

    # Verify priority results
    assert isinstance(results["priority_results"], list)
    assert len(results["priority_results"]) > 0

    # Verify overall falsification metrics
    if "total_criteria" in results:
        assert "total_falsified_criteria" in results
        assert "overall_falsification_rate" in results
        assert 0 <= results["overall_falsification_rate"] <= 1

    # Verify theory status is one of expected values
    expected_statuses = [
        "supported",
        "weakly_falsified",
        "strongly_falsified",
        "not_tested",
    ]
    assert results["theory_status"] in expected_statuses

    # Verify each priority result has expected structure
    for priority_result in results["priority_results"]:
        if "error" not in priority_result:
            assert "priority" in priority_result
            assert "falsified_criteria" in priority_result
            assert "total_criteria" in priority_result
            assert isinstance(priority_result["falsified_criteria"], int)
            assert isinstance(priority_result["total_criteria"], int)
            assert (
                priority_result["falsified_criteria"]
                <= priority_result["total_criteria"]
            )
