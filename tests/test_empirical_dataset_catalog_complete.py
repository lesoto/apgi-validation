"""
Comprehensive tests for utils/empirical_dataset_catalog.py - 100% coverage target.

This file tests:
- Dataset catalog initialization and management
- Dataset metadata handling
- Search and filtering functionality
- Data validation and integrity
- Import/export operations
- Error handling and recovery
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.empirical_dataset_catalog import (
        DataQuality,
        DatasetMetadata,
        DatasetType,
        EmpiricalDatasetCatalog,
        SearchCriteria,
    )

    EMPIRICAL_CATALOG_AVAILABLE = True
except ImportError as e:
    EMPIRICAL_CATALOG_AVAILABLE = False
    print(f"Warning: empirical_dataset_catalog not available for testing: {e}")


class TestEmpiricalDatasetCatalogComplete:
    """Comprehensive tests for EmpiricalDatasetCatalog functionality."""

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_catalog_initialization(self):
        """Test catalog initialization with default settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            assert catalog.data_dir == Path(temp_dir)
            assert catalog.datasets == {}
            assert catalog.metadata_file.exists()

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_add_dataset_with_metadata(self):
        """Test adding dataset with complete metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            # Create test dataset file
            dataset_file = Path(temp_dir) / "test_dataset.csv"
            dataset_file.write_text("subject,accuracy,rt\n1,0.85,500\n2,0.90,450")

            # Create metadata
            metadata = DatasetMetadata(
                name="test_dataset",
                dataset_type=DatasetType.BEHAVIORAL,
                description="Test behavioral dataset",
                source="test_lab",
                date_created=datetime.now(),
                file_path=str(dataset_file),
                file_size=dataset_file.stat().st_size,
                format="csv",
                quality=DataQuality.HIGH,
                subjects=2,
                sessions=1,
                duration_minutes=30,
                tags=["test", "behavioral"],
                doi=None,
                license="CC-BY-4.0",
            )

            # Add dataset
            catalog.add_dataset(metadata)

            assert "test_dataset" in catalog.datasets
            assert catalog.datasets["test_dataset"].name == "test_dataset"

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_search_by_criteria(self):
        """Test dataset search functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            # Add multiple datasets
            for i in range(3):
                dataset_file = Path(temp_dir) / f"dataset_{i}.csv"
                dataset_file.write_text(f"data_{i}")

                metadata = DatasetMetadata(
                    name=f"dataset_{i}",
                    dataset_type=(
                        DatasetType.BEHAVIORAL
                        if i % 2 == 0
                        else DatasetType.NEUROIMAGING
                    ),
                    description=f"Test dataset {i}",
                    source="test_lab",
                    date_created=datetime.now(),
                    file_path=str(dataset_file),
                    file_size=dataset_file.stat().st_size,
                    format="csv",
                    quality=DataQuality.HIGH,
                    subjects=10 + i,
                    sessions=1,
                    duration_minutes=30,
                    tags=["test", f"type_{i}"],
                    doi=None,
                    license="CC-BY-4.0",
                )

                catalog.add_dataset(metadata)

            # Search by type
            criteria = SearchCriteria(dataset_types=[DatasetType.BEHAVIORAL])
            results = catalog.search(criteria)

            assert len(results) == 2  # dataset_0 and dataset_2

            # Search by tags
            criteria = SearchCriteria(tags=["type_1"])
            results = catalog.search(criteria)

            assert len(results) == 1
            assert results[0].name == "dataset_1"

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_filter_by_subject_count(self):
        """Test filtering by subject count range."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            # Add datasets with different subject counts
            subject_counts = [5, 15, 25, 35]
            for i, count in enumerate(subject_counts):
                dataset_file = Path(temp_dir) / f"dataset_{i}.csv"
                dataset_file.write_text(f"data_{i}")

                metadata = DatasetMetadata(
                    name=f"dataset_{i}",
                    dataset_type=DatasetType.BEHAVIORAL,
                    description=f"Test dataset {i}",
                    source="test_lab",
                    date_created=datetime.now(),
                    file_path=str(dataset_file),
                    file_size=dataset_file.stat().st_size,
                    format="csv",
                    quality=DataQuality.HIGH,
                    subjects=count,
                    sessions=1,
                    duration_minutes=30,
                    tags=["test"],
                    doi=None,
                    license="CC-BY-4.0",
                )

                catalog.add_dataset(metadata)

            # Filter by subject range
            criteria = SearchCriteria(min_subjects=10, max_subjects=30)
            results = catalog.search(criteria)

            assert len(results) == 2  # datasets with 15 and 25 subjects

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_dataset_validation(self):
        """Test dataset file validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            # Create valid dataset
            valid_file = Path(temp_dir) / "valid.csv"
            valid_file.write_text("col1,col2\n1,2\n3,4")

            # Test validation
            is_valid = catalog.validate_dataset(str(valid_file))
            assert is_valid is True

            # Create invalid dataset (empty)
            invalid_file = Path(temp_dir) / "invalid.csv"
            invalid_file.write_text("")

            is_valid = catalog.validate_dataset(str(invalid_file))
            assert is_valid is False

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_remove_dataset(self):
        """Test dataset removal functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            # Add dataset
            dataset_file = Path(temp_dir) / "test.csv"
            dataset_file.write_text("data")

            metadata = DatasetMetadata(
                name="test_dataset",
                dataset_type=DatasetType.BEHAVIORAL,
                description="Test dataset",
                source="test_lab",
                date_created=datetime.now(),
                file_path=str(dataset_file),
                file_size=dataset_file.stat().st_size,
                format="csv",
                quality=DataQuality.HIGH,
                subjects=10,
                sessions=1,
                duration_minutes=30,
                tags=["test"],
                doi=None,
                license="CC-BY-4.0",
            )

            catalog.add_dataset(metadata)
            assert "test_dataset" in catalog.datasets

            # Remove dataset
            removed = catalog.remove_dataset("test_dataset")
            assert removed is True
            assert "test_dataset" not in catalog.datasets

            # Try to remove non-existent dataset
            removed = catalog.remove_dataset("non_existent")
            assert removed is False

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_export_catalog(self):
        """Test catalog export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            # Add test dataset
            dataset_file = Path(temp_dir) / "test.csv"
            dataset_file.write_text("data")

            metadata = DatasetMetadata(
                name="test_dataset",
                dataset_type=DatasetType.BEHAVIORAL,
                description="Test dataset",
                source="test_lab",
                date_created=datetime.now(),
                file_path=str(dataset_file),
                file_size=dataset_file.stat().st_size,
                format="csv",
                quality=DataQuality.HIGH,
                subjects=10,
                sessions=1,
                duration_minutes=30,
                tags=["test"],
                doi=None,
                license="CC-BY-4.0",
            )

            catalog.add_dataset(metadata)

            # Export catalog
            export_file = Path(temp_dir) / "catalog_export.json"
            catalog.export_catalog(str(export_file))

            assert export_file.exists()

            # Verify export content
            with open(export_file, "r") as f:
                exported_data = json.load(f)

            assert "datasets" in exported_data
            assert len(exported_data["datasets"]) == 1
            assert exported_data["datasets"][0]["name"] == "test_dataset"

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_import_catalog(self):
        """Test catalog import functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create export data
            export_data = {
                "datasets": [
                    {
                        "name": "imported_dataset",
                        "dataset_type": "BEHAVIORAL",
                        "description": "Imported test dataset",
                        "source": "import_lab",
                        "date_created": datetime.now().isoformat(),
                        "file_path": "/path/to/imported.csv",
                        "file_size": 1024,
                        "format": "csv",
                        "quality": "HIGH",
                        "subjects": 20,
                        "sessions": 2,
                        "duration_minutes": 45,
                        "tags": ["imported", "test"],
                        "doi": None,
                        "license": "CC-BY-4.0",
                    }
                ]
            }

            # Save export file
            export_file = Path(temp_dir) / "catalog_to_import.json"
            with open(export_file, "w") as f:
                json.dump(export_data, f)

            # Import catalog
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)
            catalog.import_catalog(str(export_file))

            assert "imported_dataset" in catalog.datasets
            assert catalog.datasets["imported_dataset"].name == "imported_dataset"

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_get_statistics(self):
        """Test catalog statistics generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            # Add datasets of different types
            types = [
                DatasetType.BEHAVIORAL,
                DatasetType.NEUROIMAGING,
                DatasetType.PHYSIOLOGICAL,
            ]
            for i, dataset_type in enumerate(types):
                dataset_file = Path(temp_dir) / f"dataset_{i}.csv"
                dataset_file.write_text(f"data_{i}")

                metadata = DatasetMetadata(
                    name=f"dataset_{i}",
                    dataset_type=dataset_type,
                    description=f"Test dataset {i}",
                    source="test_lab",
                    date_created=datetime.now(),
                    file_path=str(dataset_file),
                    file_size=dataset_file.stat().st_size,
                    format="csv",
                    quality=DataQuality.HIGH,
                    subjects=10 + i * 5,
                    sessions=1,
                    duration_minutes=30,
                    tags=["test"],
                    doi=None,
                    license="CC-BY-4.0",
                )

                catalog.add_dataset(metadata)

            # Get statistics
            stats = catalog.get_statistics()

            assert stats["total_datasets"] == 3
            assert stats["total_subjects"] == 45  # 10 + 15 + 20
            assert "by_type" in stats
            assert stats["by_type"]["BEHAVIORAL"] == 1
            assert stats["by_type"]["NEUROIMAGING"] == 1
            assert stats["by_type"]["PHYSIOLOGICAL"] == 1

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_update_dataset_metadata(self):
        """Test updating existing dataset metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            # Add initial dataset
            dataset_file = Path(temp_dir) / "test.csv"
            dataset_file.write_text("data")

            metadata = DatasetMetadata(
                name="test_dataset",
                dataset_type=DatasetType.BEHAVIORAL,
                description="Original description",
                source="test_lab",
                date_created=datetime.now(),
                file_path=str(dataset_file),
                file_size=dataset_file.stat().st_size,
                format="csv",
                quality=DataQuality.MEDIUM,
                subjects=10,
                sessions=1,
                duration_minutes=30,
                tags=["original"],
                doi=None,
                license="CC-BY-4.0",
            )

            catalog.add_dataset(metadata)

            # Update metadata
            updated_metadata = catalog.datasets["test_dataset"]
            updated_metadata.description = "Updated description"
            updated_metadata.quality = DataQuality.HIGH
            updated_metadata.tags.append("updated")

            catalog.update_dataset("test_dataset", updated_metadata)

            # Verify update
            assert catalog.datasets["test_dataset"].description == "Updated description"
            assert catalog.datasets["test_dataset"].quality == DataQuality.HIGH
            assert "updated" in catalog.datasets["test_dataset"].tags

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_error_handling_invalid_file(self):
        """Test error handling for invalid file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            # Try to add dataset with non-existent file
            with pytest.raises(FileNotFoundError):
                metadata = DatasetMetadata(
                    name="invalid_dataset",
                    dataset_type=DatasetType.BEHAVIORAL,
                    description="Invalid dataset",
                    source="test_lab",
                    date_created=datetime.now(),
                    file_path="/non/existent/file.csv",
                    file_size=0,
                    format="csv",
                    quality=DataQuality.HIGH,
                    subjects=10,
                    sessions=1,
                    duration_minutes=30,
                    tags=["test"],
                    doi=None,
                    license="CC-BY-4.0",
                )

                catalog.add_dataset(metadata)

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_duplicate_dataset_handling(self):
        """Test handling of duplicate dataset names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            # Add first dataset
            dataset_file = Path(temp_dir) / "test.csv"
            dataset_file.write_text("data")

            metadata = DatasetMetadata(
                name="duplicate_test",
                dataset_type=DatasetType.BEHAVIORAL,
                description="First dataset",
                source="test_lab",
                date_created=datetime.now(),
                file_path=str(dataset_file),
                file_size=dataset_file.stat().st_size,
                format="csv",
                quality=DataQuality.HIGH,
                subjects=10,
                sessions=1,
                duration_minutes=30,
                tags=["test"],
                doi=None,
                license="CC-BY-4.0",
            )

            catalog.add_dataset(metadata)

            # Try to add duplicate
            with pytest.raises(ValueError):
                catalog.add_dataset(metadata)

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_search_with_date_range(self):
        """Test searching with date range criteria."""
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog = EmpiricalDatasetCatalog(data_dir=temp_dir)

            # Add datasets with different dates
            base_date = datetime.now()
            for i in range(3):
                dataset_file = Path(temp_dir) / f"dataset_{i}.csv"
                dataset_file.write_text(f"data_{i}")

                metadata = DatasetMetadata(
                    name=f"dataset_{i}",
                    dataset_type=DatasetType.BEHAVIORAL,
                    description=f"Test dataset {i}",
                    source="test_lab",
                    date_created=base_date - timedelta(days=i * 10),
                    file_path=str(dataset_file),
                    file_size=dataset_file.stat().st_size,
                    format="csv",
                    quality=DataQuality.HIGH,
                    subjects=10,
                    sessions=1,
                    duration_minutes=30,
                    tags=["test"],
                    doi=None,
                    license="CC-BY-4.0",
                )

                catalog.add_dataset(metadata)

            # Search by date range
            start_date = base_date - timedelta(days=15)
            end_date = base_date - timedelta(days=5)

            criteria = SearchCriteria(start_date=start_date, end_date=end_date)
            results = catalog.search(criteria)

            assert len(results) == 1  # Only dataset_1 falls in this range
