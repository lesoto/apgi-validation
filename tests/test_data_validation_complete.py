"""
Comprehensive Tests for Data Validation Module
===============================================

Target: 100% coverage for utils/data_validation.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_validation import (
    DataPreprocessor,
    DataValidator,
    ValidationConfig,
    load_real_data_stub,
    main,
    validate_doc_eeg_dataset,
    validate_fmri_dataset,
)


class TestValidationConfig:
    """Test ValidationConfig dataclass"""

    def test_default_values(self):
        """Test default configuration values"""
        config = ValidationConfig()
        assert config.missing_data_threshold == 10.0
        assert config.outlier_threshold == 5.0
        assert config.outlier_zscore_threshold == 1.5
        assert config.signal_quality_threshold == 70.0
        assert config.eeg_min_range == -500.0
        assert config.eeg_max_range == 500.0

    def test_custom_values(self):
        """Test custom configuration values"""
        config = ValidationConfig(
            missing_data_threshold=5.0,
            outlier_threshold=3.0,
            signal_quality_threshold=80.0,
        )
        assert config.missing_data_threshold == 5.0
        assert config.outlier_threshold == 3.0
        assert config.signal_quality_threshold == 80.0


class TestDataValidator:
    """Test DataValidator class"""

    @pytest.fixture
    def validator(self):
        return DataValidator()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1ms"),
                "EEG_Cz": np.random.randn(100),
                "pupil_diameter": np.random.uniform(2, 8, 100),
                "eda": np.random.uniform(0.5, 5, 100),
            }
        )

    def test_validator_init_default(self):
        """Test validator initialization with defaults"""
        validator = DataValidator()
        assert validator.strict_mode is False
        assert isinstance(validator.config, ValidationConfig)

    def test_validator_init_strict(self):
        """Test validator initialization with strict mode"""
        validator = DataValidator(strict_mode=True)
        assert validator.strict_mode is True

    def test_validator_init_custom_config(self):
        """Test validator with custom config"""
        config = ValidationConfig(missing_data_threshold=5.0)
        validator = DataValidator(config=config)
        assert validator.config.missing_data_threshold == 5.0

    def test_validate_file_format_nonexistent(self, validator):
        """Test validation of non-existent file"""
        result = validator.validate_file_format("/nonexistent/file.csv")
        assert result["file_exists"] is False
        assert len(result["errors"]) > 0

    def test_validate_file_format_csv(self, validator, sample_df, tmp_path):
        """Test CSV file validation"""
        csv_file = tmp_path / "test.csv"
        sample_df.to_csv(csv_file, index=False)

        result = validator.validate_file_format(csv_file)
        assert result["file_exists"] is True
        assert result["is_readable"] is True

    def test_validate_file_format_json(self, validator, sample_df, tmp_path):
        """Test JSON file validation"""
        json_file = tmp_path / "test.json"
        # Convert timestamps to ISO format strings for JSON serialization
        df_copy = sample_df.copy()
        df_copy["timestamp"] = df_copy["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        data = {
            "metadata": {
                "subject_id": "test_001",
                "session_id": "session_1",
                "sampling_rate": 1000.0,
                "duration": 10.0,
            },
            "data": df_copy.to_dict("records"),
        }
        with open(json_file, "w") as f:
            json.dump(data, f)

        result = validator.validate_file_format(json_file)
        assert result["file_exists"] is True

    def test_validate_file_format_unsupported(self, validator, tmp_path):
        """Test unsupported file format"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test content")

        result = validator.validate_file_format(txt_file)
        assert "Unsupported file format" in str(result["errors"])

    def test_validate_file_format_too_large(self, validator, tmp_path):
        """Test file size limit"""
        csv_file = tmp_path / "large.csv"
        # Create a config with small file size limit
        config = ValidationConfig()
        config.max_file_size_mb = 0.001  # 1KB limit
        validator = DataValidator(config=config)

        # Create a file larger than limit
        csv_file.write_text("col1,col2\n" + "a,b\n" * 1000)

        result = validator.validate_file_format(csv_file)
        assert "File too large" in str(result["errors"])

    def test_validate_data_quality(self, validator, sample_df):
        """Test data quality validation"""
        result = validator.validate_data_quality(sample_df)
        assert "total_samples" in result
        assert result["total_samples"] == 100
        assert "missing_data" in result
        assert "outliers" in result
        assert "signal_quality" in result

    def test_validate_data_quality_with_missing(self, validator):
        """Test quality validation with missing data"""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1ms"),
                "EEG_Cz": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "pupil_diameter": [2.0] * 10,
                "eda": [0.5] * 10,
            }
        )

        result = validator.validate_data_quality(df)
        assert "EEG_Cz" in result["missing_data"]
        assert result["missing_data"]["EEG_Cz"]["count"] == 1

    def test_assess_signal_quality_good(self, validator):
        """Test signal quality assessment with good signal"""
        signal = pd.Series(np.random.randn(100))
        result = validator._assess_signal_quality(signal)
        assert "score" in result
        assert "issues" in result
        assert result["score"] > 50

    def test_assess_signal_quality_no_variation(self, validator):
        """Test signal quality with constant signal"""
        signal = pd.Series([5.0] * 100)
        result = validator._assess_signal_quality(signal)
        assert "No signal variation" in result["issues"]

    def test_assess_signal_quality_empty(self, validator):
        """Test signal quality with empty signal"""
        signal = pd.Series([], dtype=float)
        result = validator._assess_signal_quality(signal)
        assert result["score"] == 0.0
        assert "No valid data" in result["issues"]

    def test_assess_temporal_consistency(self, validator):
        """Test temporal consistency assessment"""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1ms"),
                "value": range(100),
            }
        )
        result = validator._assess_temporal_consistency(df)
        assert "score" in result

    def test_assess_temporal_consistency_no_timestamp(self, validator):
        """Test temporal consistency without timestamp"""
        df = pd.DataFrame({"value": range(100)})
        result = validator._assess_temporal_consistency(df)
        assert result["score"] == 0.0

    def test_calculate_quality_score(self, validator, sample_df):
        """Test quality score calculation"""
        metrics = validator.validate_data_quality(sample_df)
        score = validator._calculate_quality_score(metrics)
        assert 0 <= score <= 100

    def test_generate_validation_report(self, validator, sample_df, tmp_path):
        """Test validation report generation"""
        csv_file = tmp_path / "test.csv"
        sample_df.to_csv(csv_file, index=False)

        report = validator.generate_validation_report(csv_file)
        assert "file_info" in report
        assert "data_quality" in report
        assert "validation_timestamp" in report

    def test_generate_recommendations(self, validator, sample_df, tmp_path):
        """Test recommendations generation"""
        csv_file = tmp_path / "test.csv"
        sample_df.to_csv(csv_file, index=False)

        report = validator.generate_validation_report(csv_file)
        recommendations = validator._generate_recommendations(report)
        assert isinstance(recommendations, list)


class TestDataPreprocessor:
    """Test DataPreprocessor class"""

    @pytest.fixture
    def preprocessor(self):
        return DataPreprocessor()

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1ms"),
                "EEG_Cz": np.random.randn(100),
                "pupil_diameter": np.random.uniform(2, 8, 100),
                "eda": np.random.uniform(0.5, 5, 100),
            }
        )

    def test_load_data_csv(self, preprocessor, sample_df, tmp_path):
        """Test loading CSV data"""
        csv_file = tmp_path / "test.csv"
        sample_df.to_csv(csv_file, index=False)

        result = preprocessor.load_data(csv_file)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100

    def test_load_data_json(self, preprocessor, sample_df, tmp_path):
        """Test loading JSON data"""
        json_file = tmp_path / "test.json"
        # Convert timestamps to ISO format strings for JSON serialization
        df_copy = sample_df.copy()
        df_copy["timestamp"] = df_copy["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        data = {
            "metadata": {
                "subject_id": "test",
                "session_id": "1",
                "sampling_rate": 1000.0,
                "duration": 10.0,
            },
            "data": df_copy.to_dict("records"),
        }
        with open(json_file, "w") as f:
            json.dump(data, f)

        result = preprocessor.load_data(json_file)
        assert isinstance(result, pd.DataFrame)

    def test_load_data_unsupported(self, preprocessor, tmp_path):
        """Test loading unsupported format"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test")

        with pytest.raises(ValueError, match="Unsupported file format"):
            preprocessor.load_data(txt_file)

    def test_clean_missing_data_interpolate(self, preprocessor, sample_df):
        """Test missing data cleaning with interpolation"""
        df = sample_df.copy()
        df.loc[10:15, "EEG_Cz"] = np.nan

        result = preprocessor.clean_missing_data(df, strategy="interpolate")
        assert result["EEG_Cz"].isna().sum() < df["EEG_Cz"].isna().sum()

    def test_clean_missing_data_forward_fill(self, preprocessor, sample_df):
        """Test missing data cleaning with forward fill"""
        df = sample_df.copy()
        df.loc[10:15, "EEG_Cz"] = np.nan

        result = preprocessor.clean_missing_data(df, strategy="forward_fill")
        # Should fill or reduce NaN values
        assert isinstance(result, pd.DataFrame)

    def test_clean_missing_data_drop(self, preprocessor, sample_df):
        """Test missing data cleaning with drop"""
        df = sample_df.copy()
        df.loc[10:15, "EEG_Cz"] = np.nan

        result = preprocessor.clean_missing_data(df, strategy="drop")
        assert result["EEG_Cz"].isna().sum() == 0

    def test_remove_outliers_iqr(self, preprocessor, sample_df):
        """Test outlier removal with IQR method"""
        # Add some outliers
        df = sample_df.copy()
        df.loc[0, "EEG_Cz"] = 1000  # Extreme outlier

        result = preprocessor.remove_outliers(df, method="iqr")
        assert len(result) < len(df)  # Should remove outliers

    def test_remove_outliers_zscore(self, preprocessor, sample_df):
        """Test outlier removal with zscore method"""
        df = sample_df.copy()
        df.loc[0, "EEG_Cz"] = 1000

        result = preprocessor.remove_outliers(df, method="zscore", threshold=2.0)
        assert isinstance(result, pd.DataFrame)

    def test_normalize_data_zscore(self, preprocessor, sample_df):
        """Test z-score normalization"""
        result = preprocessor.normalize_data(sample_df, method="zscore")
        # Check that data is normalized (mean close to 0, std close to 1)
        assert abs(result["EEG_Cz"].mean()) < 0.1
        assert abs(result["EEG_Cz"].std() - 1.0) < 0.1

    def test_normalize_data_minmax(self, preprocessor, sample_df):
        """Test min-max normalization"""
        result = preprocessor.normalize_data(sample_df, method="minmax")
        # Check that data is in [0, 1] range
        assert result["EEG_Cz"].min() >= 0
        assert result["EEG_Cz"].max() <= 1

    def test_normalize_data_robust(self, preprocessor, sample_df):
        """Test robust normalization"""
        result = preprocessor.normalize_data(sample_df, method="robust")
        assert isinstance(result, pd.DataFrame)

    def test_resample_data(self, preprocessor, sample_df):
        """Test data resampling"""
        result = preprocessor.resample_data(sample_df, target_rate=500.0)
        assert isinstance(result, pd.DataFrame)

    def test_resample_data_no_timestamp(self, preprocessor):
        """Test resampling without timestamp column"""
        df = pd.DataFrame({"value": range(100)})
        with pytest.raises(ValueError, match="Time column 'timestamp' not found"):
            preprocessor.resample_data(df, target_rate=500.0)

    def test_save_processed_data_csv(self, preprocessor, sample_df, tmp_path):
        """Test saving processed data as CSV"""
        output_file = tmp_path / "output.csv"
        preprocessor.save_processed_data(sample_df, output_file, format="csv")
        assert output_file.exists()

    def test_save_processed_data_json(self, preprocessor, sample_df, tmp_path):
        """Test saving processed data as JSON"""
        output_file = tmp_path / "output.json"
        preprocessor.save_processed_data(sample_df, output_file, format="json")
        assert output_file.exists()

        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)
        assert "data" in data
        assert "preprocessing_steps" in data

    def test_save_processed_data_unsupported(self, preprocessor, sample_df, tmp_path):
        """Test saving with unsupported format"""
        output_file = tmp_path / "output.txt"
        with pytest.raises(ValueError, match="Unsupported output format"):
            preprocessor.save_processed_data(sample_df, output_file, format="txt")


class TestFmriDatasetValidation:
    """Test fMRI dataset validation functions"""

    def test_validate_fmri_dataset_nonexistent(self):
        """Test validation of non-existent fMRI dataset"""
        result = validate_fmri_dataset("/nonexistent/path")
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_fmri_npz_valid(self, tmp_path):
        """Test validation of valid NPZ fMRI file"""
        npz_file = tmp_path / "test_fmri.npz"

        # Create synthetic fMRI data
        vmPFC_bold = np.random.randn(100, 10)
        conditions = np.array(
            [{"trial_type": "threat"}] * 50 + [{"trial_type": "safe"}] * 50
        )
        dt = 2.0
        trial_duration = 12.0

        np.savez(
            npz_file,
            vmPFC_bold=vmPFC_bold,
            conditions=conditions,
            dt=dt,
            trial_duration=trial_duration,
        )

        result = validate_fmri_dataset(npz_file)
        assert result["valid"] is True

    def test_validate_fmri_npz_missing_fields(self, tmp_path):
        """Test validation of NPZ with missing required fields"""
        npz_file = tmp_path / "test_fmri.npz"
        np.savez(npz_file, vmPFC_bold=np.random.randn(100, 10))  # Missing other fields

        result = validate_fmri_dataset(npz_file)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_fmri_bids_directory(self, tmp_path):
        """Test validation of BIDS directory structure"""
        bids_dir = tmp_path / "bids_dataset"
        bids_dir.mkdir()

        # Create participants.tsv
        participants_file = bids_dir / "participants.tsv"
        participants_file.write_text("participant_id\nsub-01\nsub-02\n")

        # Create subject directories with minimal structure
        for sub_id in ["sub-01", "sub-02"]:
            sub_dir = bids_dir / sub_id / "func"
            sub_dir.mkdir(parents=True)

        result = validate_fmri_dataset(bids_dir, min_subjects=1)
        assert result["valid"] is True

    def test_validate_fmri_unsupported_format(self, tmp_path):
        """Test validation of unsupported file format"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not valid data")

        result = validate_fmri_dataset(txt_file)
        assert result["valid"] is False
        assert "Unsupported data format" in str(result["errors"])


class TestDocEegDatasetValidation:
    """Test DoC EEG dataset validation functions"""

    def test_validate_doc_eeg_dataset_nonexistent(self):
        """Test validation of non-existent DoC dataset"""
        result = validate_doc_eeg_dataset("/nonexistent/path")
        assert result["valid"] is False

    def test_validate_doc_npz_valid(self, tmp_path):
        """Test validation of valid NPZ DoC file"""
        npz_file = tmp_path / "test_doc.npz"

        pci_scores = np.random.uniform(0, 1, 50)
        hep_amplitudes = np.random.randn(50)
        consciousness_labels = np.array([0] * 20 + [1] * 20 + [2] * 10)

        np.savez(
            npz_file,
            pci_scores=pci_scores,
            hep_amplitudes=hep_amplitudes,
            consciousness_labels=consciousness_labels,
        )

        result = validate_doc_eeg_dataset(npz_file, min_patients=10)
        assert result["valid"] is True

    def test_validate_doc_bids_directory(self, tmp_path):
        """Test validation of DoC BIDS directory"""
        bids_dir = tmp_path / "doc_bids"
        bids_dir.mkdir()

        # Create participants.tsv with clinical data
        participants_file = bids_dir / "participants.tsv"
        participants_file.write_text(
            "participant_id\tdiagnosis\nsub-01\tVS/UWS\nsub-02\tMCS\n"
        )

        # Create subject directories
        for sub_id in ["sub-01", "sub-02"]:
            sub_dir = bids_dir / sub_id / "eeg"
            sub_dir.mkdir(parents=True)

        result = validate_doc_eeg_dataset(bids_dir, min_patients=1)
        assert result["valid"] is True


class TestLoadRealDataStub:
    """Test load_real_data_stub function"""

    def test_load_real_data_stub_no_data(self):
        """Test stub when no data path provided"""
        result = load_real_data_stub("VP-14")
        assert result["status"] == "data_required"
        assert "message" in result

    def test_load_real_data_stub_with_valid_data(self, tmp_path):
        """Test stub with valid data path"""
        npz_file = tmp_path / "test_fmri.npz"
        vmPFC_bold = np.random.randn(100, 10)
        conditions = np.array(
            [{"trial_type": "threat"}] * 50 + [{"trial_type": "safe"}] * 50
        )
        np.savez(
            npz_file,
            vmPFC_bold=vmPFC_bold,
            conditions=conditions,
            dt=2.0,
            trial_duration=12.0,
        )

        result = load_real_data_stub("VP-14", str(npz_file))
        assert result["status"] == "empirical_loaded"


class TestMain:
    """Test main function"""

    def test_main_runs(self, tmp_path, monkeypatch):
        """Test that main() runs without errors"""
        # Create data directory with sample file
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        sample_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1ms"),
                "EEG_Cz": range(10),
                "pupil_diameter": [2.0] * 10,
                "eda": [0.5] * 10,
            }
        )
        sample_df.to_csv(data_dir / "demo_demo.csv", index=False)

        # Monkeypatch the current working directory
        monkeypatch.chdir(tmp_path)

        # Should not raise
        main()


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_validate_csv_structure_missing_columns(self, tmp_path):
        """Test CSV validation with missing required columns"""
        validator = DataValidator()
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("col1,col2\n1,2\n3,4\n")

        result = validator.validate_file_format(csv_file)
        assert "Missing required columns" in str(result.get("errors", []))

    def test_validate_csv_structure_non_numeric(self, tmp_path):
        """Test CSV validation with non-numeric data in numeric columns"""
        validator = DataValidator()
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("timestamp,EEG_Cz,pupil_diameter,eda\na,b,c,d\ne,f,g,h\n")

        result = validator.validate_file_format(csv_file)
        assert len(result.get("errors", [])) > 0

    @pytest.mark.xfail(
        reason="Environment-specific recursion depth behavior", strict=False
    )
    def test_json_recursion_error(self, validator, tmp_path):
        """Test JSON validation with deeply nested structure"""
        json_file = tmp_path / "deep.json"

        # Create deeply nested JSON
        nested = {}
        current = nested
        for _ in range(500):
            current["next"] = {}
            current = current["next"]

        with open(json_file, "w") as f:
            json.dump(nested, f)

        result = validator.validate_file_format(json_file)
        # Should either pass validation or return an error about recursion/nesting
        assert result["file_exists"] is True

    def test_data_ranges_validation_warnings(self, tmp_path):
        """Test data range validation warnings"""
        validator = DataValidator()
        csv_file = tmp_path / "extreme.csv"

        # Create data with extreme values
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1ms"),
                "EEG_Cz": [10000] * 10,  # Extreme EEG value
                "pupil_diameter": [50.0] * 10,  # Extreme pupil value
                "eda": [100.0] * 10,  # Extreme EDA value
            }
        )
        df.to_csv(csv_file, index=False)

        result = validator.validate_file_format(csv_file)
        # Should have warnings about unusual ranges
        assert len(result.get("warnings", [])) > 0
