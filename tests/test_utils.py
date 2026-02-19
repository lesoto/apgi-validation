"""
Tests for utility modules.
===========================
"""

from pathlib import Path


def test_utils_directory_structure():
    """Test that utils directory has expected structure."""
    project_root = Path(__file__).parent.parent
    utils_dir = project_root / "utils"

    assert utils_dir.exists(), "utils directory missing"
    assert utils_dir.is_dir(), "utils should be a directory"

    # Check subdirectories
    subdirs = ["config", "data"]
    for subdir in subdirs:
        subdir_path = utils_dir / subdir
        assert subdir_path.exists(), f"utils/{subdir} directory missing"
        assert subdir_path.is_dir(), f"utils/{subdir} should be a directory"


def test_utility_files_exist():
    """Test that essential utility files exist."""
    project_root = Path(__file__).parent.parent
    utils_dir = project_root / "utils"

    # Check essential utility files
    utility_files = [
        "config_manager.py",
        "backup_manager.py",
        "batch_processor.py",
        "data_quality_assessment.py",
    ]

    for file_name in utility_files:
        file_path = utils_dir / file_name
        assert file_path.exists(), f"Utility file {file_name} missing"


def test_sample_data_fixture_structure(sample_data):
    """Test that sample_data fixture has expected structure."""
    assert isinstance(sample_data, dict)

    # Check required data keys
    required_keys = ["timestamps", "surprise", "threshold", "metabolic", "arousal"]
    for key in required_keys:
        assert key in sample_data, f"Sample data missing key: {key}"
        assert isinstance(sample_data[key], list), f"Sample data {key} should be a list"
        assert len(sample_data[key]) > 0, f"Sample data {key} should not be empty"

    # Check data consistency
    data_length = len(sample_data["timestamps"])
    for key in required_keys:
        assert (
            len(sample_data[key]) == data_length
        ), f"Sample data {key} length mismatch"
