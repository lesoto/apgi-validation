"""Tests for BIDS Data Loaders module - comprehensive coverage for 0% coverage file."""

import json

import pandas as pd
import pytest

from utils.bids_data_loaders import (
    discover_bids_dataset,
    load_carhart_harris_fmri,
    load_cogitate_ieeg_data,
    load_openneuro_depression_eeg,
)


class TestDiscoverBidsDataset:
    """Test BIDS dataset discovery."""

    def test_discover_nonexistent_path(self):
        """Test discovery with non-existent path."""
        with pytest.raises(FileNotFoundError):
            discover_bids_dataset("/nonexistent/path")

    def test_discover_valid_dataset(self, tmp_path):
        """Test discovery with valid BIDS structure."""
        # Create BIDS structure
        bids_root = tmp_path / "bids_dataset"
        bids_root.mkdir()

        # Create dataset_description.json
        desc = {"Name": "Test Dataset", "BIDSVersion": "1.0.0"}
        with open(bids_root / "dataset_description.json", "w") as f:
            json.dump(desc, f)

        # Create participants.tsv
        participants = pd.DataFrame({"participant_id": ["sub-01", "sub-02"]})
        participants.to_csv(bids_root / "participants.tsv", sep="\t", index=False)

        # Create modality directories
        (bids_root / "eeg").mkdir()
        (bids_root / "func").mkdir()

        result = discover_bids_dataset(bids_root)

        assert result["root"] == str(bids_root)
        assert result["description"]["Name"] == "Test Dataset"
        assert result["n_participants"] == 2
        assert "eeg" in result["modalities"]
        assert "func" in result["modalities"]

    def test_discover_minimal_dataset(self, tmp_path):
        """Test discovery with minimal structure."""
        bids_root = tmp_path / "minimal"
        bids_root.mkdir()

        result = discover_bids_dataset(bids_root)

        assert result["root"] == str(bids_root)
        assert "description" not in result
        assert "participants" not in result


class TestLoadCogitateIEEGData:
    """Test Cogitate iEEG data loading."""

    def test_load_nonexistent_dataset(self):
        """Test loading from non-existent dataset."""
        with pytest.raises(FileNotFoundError):
            load_cogitate_ieeg_data("/nonexistent/path")

    def test_load_without_ieeg_directory(self, tmp_path):
        """Test loading without iEEG directory."""
        bids_root = tmp_path / "bids"
        bids_root.mkdir()
        # Create valid JSON file (not empty)
        desc = {"Name": "Test Dataset", "BIDSVersion": "1.0.0"}
        with open(bids_root / "dataset_description.json", "w") as f:
            json.dump(desc, f)

        with pytest.raises(FileNotFoundError):
            load_cogitate_ieeg_data(bids_root)


class TestLoadDepressionEEGData:
    """Test Depression EEG data loading."""

    def test_load_nonexistent_dataset(self):
        """Test loading from non-existent dataset."""
        with pytest.raises(FileNotFoundError):
            load_openneuro_depression_eeg("/nonexistent/path")


class TestLoadCarhartHarrisFMRIData:
    """Test Carhart-Harris fMRI data loading."""

    def test_load_nonexistent_dataset(self):
        """Test loading from non-existent dataset."""
        with pytest.raises(FileNotFoundError):
            load_carhart_harris_fmri("/nonexistent/path")


class TestIntegrationMockData:
    """Integration tests with mock BIDS data."""

    @pytest.fixture
    def mock_bids_dataset(self, tmp_path):
        """Create a mock BIDS dataset."""
        bids_root = tmp_path / "mock_bids"
        bids_root.mkdir()

        # Dataset description
        desc = {"Name": "Mock Dataset", "BIDSVersion": "1.6.0", "DatasetType": "raw"}
        with open(bids_root / "dataset_description.json", "w") as f:
            json.dump(desc, f)

        # Participants
        participants = pd.DataFrame(
            {
                "participant_id": ["sub-01", "sub-02", "sub-03"],
                "age": [25, 30, 35],
                "sex": ["M", "F", "M"],
            }
        )
        participants.to_csv(bids_root / "participants.tsv", sep="\t", index=False)

        # Create modality dirs
        (bids_root / "eeg").mkdir()
        (bids_root / "ieeg").mkdir()

        return bids_root

    def test_full_discovery_workflow(self, mock_bids_dataset):
        """Test complete discovery workflow."""
        result = discover_bids_dataset(mock_bids_dataset)

        assert result["n_participants"] == 3
        assert "eeg" in result["modalities"]
        assert "ieeg" in result["modalities"]
        assert result["description"]["DatasetType"] == "raw"
