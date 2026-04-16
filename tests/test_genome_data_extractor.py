"""
Tests for genome_data_extractor module.
"""

import json

import pytest

from utils.genome_data_extractor import (
    extract_genome_data_from_vp5,
    load_genome_data,
    save_genome_data,
)


@pytest.fixture
def sample_vp5_results(tmp_path):
    """Create sample VP-5 results for testing."""
    results = {
        "config": {"n_generations": 500},
        "final_statistics": {
            "final_frequencies": {
                "has_threshold": 0.3,
                "has_precision_weighting": 0.4,
                "has_intero_weighting": 0.5,
            }
        },
    }
    results_path = tmp_path / "protocol5_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    return str(results_path)


@pytest.fixture
def sample_genome_data():
    """Create sample genome data for testing."""
    return {
        "evolved_alpha_values": [4.5, 4.0, 2.5, 2.0],
        "timescale_correlations": [0.55, 0.5, 0.25, 0.2],
        "intero_gain_ratios": [1.5, 1.4, 0.9, 0.8],
        "n_agents": 100,
        "n_generations": 500,
    }


class TestExtractGenomeData:
    """Tests for extract_genome_data_from_vp5 function."""

    def test_extract_genome_data_basic(self, sample_vp5_results):
        """Test basic genome data extraction."""
        genome_data = extract_genome_data_from_vp5(sample_vp5_results)

        assert isinstance(genome_data, dict)
        assert "evolved_alpha_values" in genome_data
        assert "timescale_correlations" in genome_data
        assert "intero_gain_ratios" in genome_data
        assert "n_agents" in genome_data
        assert "n_generations" in genome_data

    def test_extract_genome_data_structure(self, sample_vp5_results):
        """Test structure of extracted genome data."""
        genome_data = extract_genome_data_from_vp5(sample_vp5_results)

        assert len(genome_data["evolved_alpha_values"]) == 100
        assert len(genome_data["timescale_correlations"]) == 100
        assert len(genome_data["intero_gain_ratios"]) == 100
        assert genome_data["n_agents"] == 100
        assert genome_data["n_generations"] == 500

    def test_extract_genome_data_no_final_statistics(self, tmp_path):
        """Test extraction when VP-5 results have no final_statistics."""
        results = {"config": {"n_generations": 300}}
        results_path = tmp_path / "protocol5_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f)

        genome_data = extract_genome_data_from_vp5(str(results_path))

        # When no final_statistics, arrays are empty lists
        assert genome_data["evolved_alpha_values"] == []
        assert genome_data["timescale_correlations"] == []
        assert genome_data["intero_gain_ratios"] == []
        assert genome_data["n_agents"] == 100
        assert genome_data["n_generations"] == 300

    def test_extract_genome_data_zero_frequencies(self, tmp_path):
        """Test extraction when all frequencies are zero."""
        results = {
            "config": {"n_generations": 500},
            "final_statistics": {
                "final_frequencies": {
                    "has_threshold": 0.0,
                    "has_precision_weighting": 0.0,
                    "has_intero_weighting": 0.0,
                }
            },
        }
        results_path = tmp_path / "protocol5_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f)

        genome_data = extract_genome_data_from_vp5(str(results_path))

        assert len(genome_data["evolved_alpha_values"]) == 100
        assert len(genome_data["timescale_correlations"]) == 100
        assert len(genome_data["intero_gain_ratios"]) == 100

    def test_extract_genome_data_full_frequencies(self, tmp_path):
        """Test extraction when all frequencies are 1.0."""
        results = {
            "config": {"n_generations": 500},
            "final_statistics": {
                "final_frequencies": {
                    "has_threshold": 1.0,
                    "has_precision_weighting": 1.0,
                    "has_intero_weighting": 1.0,
                }
            },
        }
        results_path = tmp_path / "protocol5_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f)

        genome_data = extract_genome_data_from_vp5(str(results_path))

        assert len(genome_data["evolved_alpha_values"]) == 100
        assert len(genome_data["timescale_correlations"]) == 100
        assert len(genome_data["intero_gain_ratios"]) == 100


class TestSaveGenomeData:
    """Tests for save_genome_data function."""

    def test_save_genome_data_basic(self, sample_genome_data, tmp_path):
        """Test basic genome data saving."""
        output_path = tmp_path / "genome_data.json"
        save_genome_data(sample_genome_data, str(output_path))

        assert output_path.exists()
        with open(output_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        assert loaded_data == sample_genome_data

    def test_save_genome_data_creates_file(self, sample_genome_data, tmp_path):
        """Test that save creates the file."""
        output_path = tmp_path / "new_genome_data.json"
        assert not output_path.exists()

        save_genome_data(sample_genome_data, str(output_path))

        assert output_path.exists()

    def test_save_genome_data_format(self, sample_genome_data, tmp_path):
        """Test that saved data is properly formatted JSON."""
        output_path = tmp_path / "genome_data.json"
        save_genome_data(sample_genome_data, str(output_path))

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert content  # File not empty
        assert "evolved_alpha_values" in content


class TestLoadGenomeData:
    """Tests for load_genome_data function."""

    def test_load_genome_data_basic(self, sample_genome_data, tmp_path):
        """Test basic genome data loading."""
        genome_path = tmp_path / "genome_data.json"
        with open(genome_path, "w", encoding="utf-8") as f:
            json.dump(sample_genome_data, f)

        loaded_data = load_genome_data(str(genome_path))

        assert loaded_data == sample_genome_data

    def test_load_genome_data_structure(self, sample_genome_data, tmp_path):
        """Test that loaded data has correct structure."""
        genome_path = tmp_path / "genome_data.json"
        with open(genome_path, "w", encoding="utf-8") as f:
            json.dump(sample_genome_data, f)

        loaded_data = load_genome_data(str(genome_path))

        assert isinstance(loaded_data, dict)
        assert all(key in loaded_data for key in sample_genome_data.keys())


class TestIntegration:
    """Integration tests for genome data workflow."""

    def test_extract_save_load_workflow(self, sample_vp5_results, tmp_path):
        """Test complete extract-save-load workflow."""
        # Extract
        genome_data = extract_genome_data_from_vp5(sample_vp5_results)

        # Save
        output_path = tmp_path / "genome_data.json"
        save_genome_data(genome_data, str(output_path))

        # Load
        loaded_data = load_genome_data(str(output_path))

        assert loaded_data == genome_data

    def test_multiple_extracts_same_results(self, sample_vp5_results):
        """Test that multiple extractions produce consistent structure."""
        genome_data1 = extract_genome_data_from_vp5(sample_vp5_results)
        genome_data2 = extract_genome_data_from_vp5(sample_vp5_results)

        assert genome_data1.keys() == genome_data2.keys()
        assert genome_data1["n_agents"] == genome_data2["n_agents"]
        assert genome_data1["n_generations"] == genome_data2["n_generations"]
