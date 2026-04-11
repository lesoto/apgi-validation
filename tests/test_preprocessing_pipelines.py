"""
Comprehensive tests for preprocessing_pipelines utility module.
============================================================

Tests all classes and functions in preprocessing_pipelines.py including:
- PreprocessingConfig
- EEGPreprocessor
- PupilPreprocessor
- EDAPreprocessor
- HeartRatePreprocessor
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.preprocessing_pipelines import (EDAPreprocessor,
                                               EEGPreprocessor,
                                               HeartRatePreprocessor,
                                               PreprocessingConfig,
                                               PupilPreprocessor)
except ImportError as e:
    pytest.skip(f"Cannot import preprocessing_pipelines: {e}", allow_module_level=True)


class TestPreprocessingConfig:
    """Test PreprocessingConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PreprocessingConfig()

        # EEG preprocessing defaults
        assert config.eeg_bandpass_low == 0.5, "Default EEG low cutoff should be 0.5 Hz"
        assert (
            config.eeg_bandpass_high == 40.0
        ), "Default EEG high cutoff should be 40 Hz"
        assert config.eeg_notch_freq == 50.0, "Default notch frequency should be 50 Hz"
        assert config.eeg_notch_width == 2.0, "Default notch width should be 2 Hz"
        assert (
            config.eeg_artifact_threshold == 5.0
        ), "Default artifact threshold should be 5.0"

        # Pupil preprocessing defaults
        assert (
            config.pupil_blink_detection
        ), "Blink detection should be enabled by default"
        assert (
            config.pupil_interpolation_method == "linear"
        ), "Default interpolation should be linear"
        assert (
            config.pupil_smoothing_window == 5
        ), "Default smoothing window should be 5"

        # EDA preprocessing defaults
        assert config.eda_lowpass_cutoff == 5.0, "Default EDA cutoff should be 5 Hz"
        assert (
            config.eda_smoothing_window == 10
        ), "Default EDA smoothing window should be 10"

        # Heart rate preprocessing defaults
        assert (
            config.hr_outlier_threshold == 3.0
        ), "Default HR outlier threshold should be 3.0"
        assert (
            config.hr_interpolation_method == "cubic"
        ), "Default HR interpolation should be cubic"

        # General preprocessing defaults
        assert (
            config.missing_data_strategy == "interpolate"
        ), "Default missing strategy should be interpolate"
        assert config.handle_missing, "Missing data handling should be enabled"
        assert config.remove_outliers, "Outlier removal should be enabled"
        assert config.normalize_data, "Normalization should be enabled"
        assert config.outlier_method == "iqr", "Default outlier method should be iqr"
        assert (
            config.outlier_threshold == 1.5
        ), "Default outlier threshold should be 1.5"
        assert (
            config.normalization_method == "zscore"
        ), "Default normalization should be zscore"

        # Resampling defaults
        assert (
            config.target_sampling_rate == 250.0
        ), "Default target rate should be 250 Hz"
        assert (
            config.resample_method == "interpolate"
        ), "Default resampling should be interpolate"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PreprocessingConfig(
            eeg_bandpass_low=1.0,
            eeg_bandpass_high=30.0,
            eeg_artifact_threshold=3.0,
            pupil_smoothing_window=10,
            eda_lowpass_cutoff=3.0,
            hr_outlier_threshold=2.5,
            outlier_method="zscore",
            normalization_method="minmax",
            target_sampling_rate=500.0,
        )

        assert config.eeg_bandpass_low == 1.0, "Should use custom EEG low cutoff"
        assert config.eeg_bandpass_high == 30.0, "Should use custom EEG high cutoff"
        assert (
            config.eeg_artifact_threshold == 3.0
        ), "Should use custom artifact threshold"
        assert config.pupil_smoothing_window == 10, "Should use custom smoothing window"
        assert config.eda_lowpass_cutoff == 3.0, "Should use custom EDA cutoff"
        assert config.hr_outlier_threshold == 2.5, "Should use custom HR threshold"
        assert config.outlier_method == "zscore", "Should use custom outlier method"
        assert (
            config.normalization_method == "minmax"
        ), "Should use custom normalization"
        assert config.target_sampling_rate == 500.0, "Should use custom target rate"


class TestEEGPreprocessor:
    """Test EEGPreprocessor class."""

    @pytest.fixture
    def sample_eeg_data(self):
        """Create sample EEG data for testing."""
        n_samples = 1000
        n_channels = 3
        fs = 1000.0
        t = np.linspace(0, n_samples / fs, n_samples)

        # Create multi-channel EEG data
        data = {}
        for i in range(n_channels):
            # Each channel has different frequency components
            signal = (
                2.0 * np.sin(2 * np.pi * 10 * t)
                + 1.0 * np.sin(2 * np.pi * 20 * t)  # Alpha
                + 0.5 * np.sin(2 * np.pi * 50 * t)  # Beta
                + 0.2 * np.random.randn(n_samples)  # Gamma  # Noise
            )
            data[f"eeg_{i + 1}"] = signal

        df = pd.DataFrame(data)
        return df, fs

    def test_eeg_preprocessor_initialization(self):
        """Test EEGPreprocessor initialization."""
        config = PreprocessingConfig()
        preprocessor = EEGPreprocessor(config)

        assert preprocessor.config == config, "Should store config"
        assert isinstance(
            preprocessor.preprocessing_log, list
        ), "Should initialize log list"
        assert len(preprocessor.preprocessing_log) == 0, "Log should be empty initially"

    def test_preprocess_eeg_basic(self, sample_eeg_data):
        """Test basic EEG preprocessing."""
        df, fs = sample_eeg_data
        config = PreprocessingConfig()
        preprocessor = EEGPreprocessor(config)

        result = preprocessor.preprocess_eeg(df, sampling_rate=fs, show_progress=False)

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) == len(df), "Should preserve data length"
        assert all(
            col in result.columns for col in df.columns
        ), "Should preserve all columns"
        assert len(preprocessor.preprocessing_log) > 0, "Should log preprocessing steps"

    def test_preprocess_eeg_custom_columns(self, sample_eeg_data):
        """Test EEG preprocessing with custom column selection."""
        df, fs = sample_eeg_data
        config = PreprocessingConfig()
        preprocessor = EEGPreprocessor(config)

        # Process only first two channels
        eeg_columns = ["eeg_1", "eeg_2"]
        result = preprocessor.preprocess_eeg(
            df, eeg_columns=eeg_columns, sampling_rate=fs, show_progress=False
        )

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert all(
            col in result.columns for col in eeg_columns
        ), "Should process specified columns"

    def test_preprocess_eeg_missing_sampling_rate(self, sample_eeg_data):
        """Test EEG preprocessing without sampling rate (should raise error)."""
        df, _ = sample_eeg_data
        config = PreprocessingConfig()
        preprocessor = EEGPreprocessor(config)

        with pytest.raises(ValueError, match="sampling_rate parameter is required"):
            preprocessor.preprocess_eeg(df, show_progress=False)

    def test_preprocess_eeg_no_eeg_columns(self):
        """Test EEG preprocessing with no EEG columns."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        config = PreprocessingConfig()
        preprocessor = EEGPreprocessor(config)

        result = preprocessor.preprocess_eeg(
            df, sampling_rate=1000.0, show_progress=False
        )

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) == len(df), "Should preserve original data"

    def test_apply_bandpass_filter(self):
        """Test bandpass filter application."""
        config = PreprocessingConfig(eeg_bandpass_low=8.0, eeg_bandpass_high=12.0)
        preprocessor = EEGPreprocessor(config)

        # Create test signal with known frequencies
        fs = 1000.0
        n_samples = 1000
        t = np.linspace(0, n_samples / fs, n_samples)

        # Signal with alpha (10 Hz) and gamma (50 Hz) components
        signal_data = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
        signal_series = pd.Series(signal_data)

        filtered = preprocessor._apply_bandpass_filter(signal_series, fs)

        assert isinstance(filtered, pd.Series), "Should return Series"
        assert len(filtered) == len(signal_series), "Should preserve length"
        assert len(preprocessor.preprocessing_log) > 0, "Should log filtering step"

    def test_apply_notch_filter(self):
        """Test notch filter application."""
        config = PreprocessingConfig(eeg_notch_freq=50.0, eeg_notch_width=2.0)
        preprocessor = EEGPreprocessor(config)

        # Create test signal with power line noise
        fs = 1000.0
        n_samples = 1000
        t = np.linspace(0, n_samples / fs, n_samples)

        # Signal with power line noise (50 Hz)
        signal_data = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
        signal_series = pd.Series(signal_data)

        filtered = preprocessor._apply_notch_filter(signal_series, fs)

        assert isinstance(filtered, pd.Series), "Should return Series"
        assert len(filtered) == len(signal_series), "Should preserve length"

    def test_detect_and_correct_artifacts(self):
        """Test artifact detection and correction."""
        config = PreprocessingConfig(eeg_artifact_threshold=2.0)
        preprocessor = EEGPreprocessor(config)

        # Create signal with artifacts (extreme values)
        signal_data = np.random.randn(1000)
        signal_data[100] = 10.0  # Add artifact
        signal_data[200] = -8.0  # Add artifact
        signal_series = pd.Series(signal_data)

        corrected = preprocessor._detect_and_correct_artifacts(signal_series)

        assert isinstance(corrected, pd.Series), "Should return Series"
        assert len(corrected) == len(signal_series), "Should preserve length"
        # Artifacts should be reduced
        assert abs(corrected[100]) < abs(signal_data[100]), "Artifact should be reduced"

    def test_ica_artifact_removal_insufficient_data(self):
        """Test ICA with insufficient data."""
        config = PreprocessingConfig()
        preprocessor = EEGPreprocessor(config)

        # Very short signal
        signal_data = np.random.randn(100)
        signal_series = pd.Series(signal_data)

        result = preprocessor._apply_ica_artifact_removal(signal_series)

        # Should return original signal due to insufficient data
        assert len(result) == len(signal_series), "Should return original signal"

    def test_estimate_sampling_rate(self):
        """Test sampling rate estimation."""
        config = PreprocessingConfig()
        preprocessor = EEGPreprocessor(config)

        # Test with regular time index
        time_index = pd.timedelta_range(start="0s", periods=1000, freq="1ms")
        signal_data = np.random.randn(1000)
        signal_series = pd.Series(signal_data, index=time_index)

        estimated_fs = preprocessor._estimate_sampling_rate(signal_series)

        assert isinstance(estimated_fs, float), "Should return float"
        assert estimated_fs > 0, "Should be positive"


class TestPupilPreprocessor:
    """Test PupilPreprocessor class."""

    @pytest.fixture
    def sample_pupil_data(self):
        """Create sample pupil data for testing."""
        n_samples = 1000
        t = np.linspace(0, 10, n_samples)  # 10 seconds

        # Create realistic pupil diameter data with blinks
        baseline = 4.0  # 4mm baseline
        variation = 0.5 * np.sin(2 * np.pi * 0.5 * t)  # Slow variation
        noise = 0.1 * np.random.randn(n_samples)

        pupil_data = baseline + variation + noise

        # Add blinks (drops to near zero)
        blink_indices = [200, 500, 800]
        for idx in blink_indices:
            pupil_data[idx - 2 : idx + 3] = 0.1  # Blink period

        df = pd.DataFrame({"pupil_diameter": pupil_data})
        return df

    def test_pupil_preprocessor_initialization(self):
        """Test PupilPreprocessor initialization."""
        config = PreprocessingConfig()
        preprocessor = PupilPreprocessor(config)

        assert preprocessor.config == config, "Should store config"
        assert isinstance(
            preprocessor.preprocessing_log, list
        ), "Should initialize log list"

    def test_preprocess_pupil_basic(self, sample_pupil_data):
        """Test basic pupil preprocessing."""
        df = sample_pupil_data
        config = PreprocessingConfig()
        preprocessor = PupilPreprocessor(config)

        result = preprocessor.preprocess_pupil(df, show_progress=False)

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert "pupil_diameter" in result.columns, "Should preserve pupil column"
        assert len(result) == len(df), "Should preserve data length"
        assert len(preprocessor.preprocessing_log) > 0, "Should log preprocessing steps"

    def test_preprocess_pupil_missing_column(self):
        """Test pupil preprocessing with missing column."""
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        config = PreprocessingConfig()
        preprocessor = PupilPreprocessor(config)

        result = preprocessor.preprocess_pupil(df, show_progress=False)

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) == len(df), "Should preserve original data"

    def test_detect_blinks(self, sample_pupil_data):
        """Test blink detection."""
        config = PreprocessingConfig()
        preprocessor = PupilPreprocessor(config)

        result = preprocessor._detect_blinks(sample_pupil_data, "pupil_diameter")

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert (
            "pupil_diameter_blink" in result.columns
        ), "Should add blink detection column"
        assert (
            result["pupil_diameter_blink"].dtype == bool
        ), "Blink column should be boolean"

    def test_interpolate_blinks(self, sample_pupil_data):
        """Test blink interpolation."""
        config = PreprocessingConfig()
        preprocessor = PupilPreprocessor(config)

        # First detect blinks
        df_with_blinks = preprocessor._detect_blinks(
            sample_pupil_data, "pupil_diameter"
        )

        # Then interpolate
        result = preprocessor._interpolate_blinks(df_with_blinks, "pupil_diameter")

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) == len(df_with_blinks), "Should preserve data length"

    def test_detect_and_handle_blinks(self, sample_pupil_data):
        """Test combined blink detection and handling."""
        config = PreprocessingConfig()
        preprocessor = PupilPreprocessor(config)

        pupil_series = sample_pupil_data["pupil_diameter"]
        result = preprocessor._detect_and_handle_blinks(pupil_series)

        assert isinstance(result, pd.Series), "Should return Series"
        assert len(result) == len(pupil_series), "Should preserve length"

    def test_normalize_pupil_diameter(self, sample_pupil_data):
        """Test pupil diameter normalization."""
        config = PreprocessingConfig()
        preprocessor = PupilPreprocessor(config)

        pupil_series = sample_pupil_data["pupil_diameter"]
        result = preprocessor._normalize_pupil_diameter(pupil_series)

        assert isinstance(result, pd.Series), "Should return Series"
        assert len(result) == len(pupil_series), "Should preserve length"
        # Normalized data should have zero mean (approximately)
        assert abs(result.mean()) < 1.0

    def test_smooth_pupil_signal(self, sample_pupil_data):
        """Test pupil signal smoothing."""
        config = PreprocessingConfig(pupil_smoothing_window=10)
        preprocessor = PupilPreprocessor(config)

        pupil_series = sample_pupil_data["pupil_diameter"]
        result = preprocessor._smooth_pupil_signal(pupil_series)

        assert isinstance(result, pd.Series), "Should return Series"
        assert len(result) == len(pupil_series), "Should preserve length"


class TestEDAPreprocessor:
    """Test EDAPreprocessor class."""

    @pytest.fixture
    def sample_eda_data(self):
        """Create sample EDA data for testing."""
        n_samples = 1000
        t = np.linspace(0, 10, n_samples)  # 10 seconds

        # Create realistic EDA data with tonic and phasic components
        tonic = 1.0 + 0.2 * np.sin(2 * np.pi * 0.1 * t)  # Slow tonic component
        phasic = 0.3 * np.sin(2 * np.pi * 0.5 * t)  # Faster phasic component
        noise = 0.05 * np.random.randn(n_samples)

        eda_data = tonic + phasic + noise

        df = pd.DataFrame({"eda": eda_data})
        return df

    def test_eda_preprocessor_initialization(self):
        """Test EDAPreprocessor initialization."""
        config = PreprocessingConfig()
        preprocessor = EDAPreprocessor(config)

        assert preprocessor.config == config, "Should store config"
        assert isinstance(
            preprocessor.preprocessing_log, list
        ), "Should initialize log list"

    def test_preprocess_eda_basic(self, sample_eda_data):
        """Test basic EDA preprocessing."""
        df = sample_eda_data
        config = PreprocessingConfig()
        preprocessor = EDAPreprocessor(config)

        result = preprocessor.preprocess_eda(df, show_progress=False)

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert "eda" in result.columns, "Should preserve EDA column"
        assert len(result) == len(df), "Should preserve data length"
        assert len(preprocessor.preprocessing_log) > 0, "Should log preprocessing steps"

    def test_preprocess_eda_missing_column(self):
        """Test EDA preprocessing with missing column."""
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        config = PreprocessingConfig()
        preprocessor = EDAPreprocessor(config)

        result = preprocessor.preprocess_eda(df, show_progress=False)

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) == len(df), "Should preserve original data"

    def test_apply_lowpass_filter(self, sample_eda_data):
        """Test lowpass filter application."""
        config = PreprocessingConfig(eda_lowpass_cutoff=3.0)
        preprocessor = EDAPreprocessor(config)

        eda_series = sample_eda_data["eda"]
        result = preprocessor._apply_lowpass_filter(eda_series)

        assert isinstance(result, pd.Series), "Should return Series"
        assert len(result) == len(eda_series), "Should preserve length"

    def test_smooth_eda_signal(self, sample_eda_data):
        """Test EDA signal smoothing."""
        config = PreprocessingConfig(eda_smoothing_window=15)
        preprocessor = EDAPreprocessor(config)

        eda_series = sample_eda_data["eda"]
        result = preprocessor._smooth_eda_signal(eda_series)

        assert isinstance(result, pd.Series), "Should return Series"
        assert len(result) == len(eda_series), "Should preserve length"

    def test_extract_phasic_tonic(self, sample_eda_data):
        """Test phasic and tonic component extraction."""
        config = PreprocessingConfig()
        preprocessor = EDAPreprocessor(config)

        result = preprocessor._extract_phasic_tonic(sample_eda_data, "eda")

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert "eda_tonic" in result.columns, "Should add tonic component"
        assert "eda_phasic" in result.columns, "Should add phasic component"
        assert len(result) == len(sample_eda_data), "Should preserve data length"

    def test_estimate_sampling_rate_eda(self, sample_eda_data):
        """Test sampling rate estimation for EDA."""
        config = PreprocessingConfig()
        preprocessor = EDAPreprocessor(config)

        eda_series = sample_eda_data["eda"]
        estimated_fs = preprocessor._estimate_sampling_rate(eda_series)

        assert isinstance(
            estimated_fs, (float, type(None))
        ), "Should return float or None"


class TestHeartRatePreprocessor:
    """Test HeartRatePreprocessor class."""

    @pytest.fixture
    def sample_hr_data(self):
        """Create sample heart rate data for testing."""
        n_samples = 1000
        t = np.linspace(0, 10, n_samples)  # 10 seconds

        # Create realistic heart rate data with outliers
        baseline_hr = 70  # 70 bpm baseline
        variation = 5 * np.sin(2 * np.pi * 0.1 * t)  # Natural variation
        noise = 2 * np.random.randn(n_samples)

        hr_data = baseline_hr + variation + noise

        # Add some outliers
        hr_data[100] = 150  # Unrealistic high value
        hr_data[500] = 30  # Unrealistic low value

        df = pd.DataFrame({"heart_rate": hr_data})
        return df

    def test_heart_rate_preprocessor_initialization(self):
        """Test HeartRatePreprocessor initialization."""
        config = PreprocessingConfig()
        preprocessor = HeartRatePreprocessor(config)

        assert preprocessor.config == config, "Should store config"
        assert isinstance(
            preprocessor.preprocessing_log, list
        ), "Should initialize log list"

    def test_preprocess_heart_rate_basic(self, sample_hr_data):
        """Test basic heart rate preprocessing."""
        df = sample_hr_data
        config = PreprocessingConfig()
        preprocessor = HeartRatePreprocessor(config)

        result = preprocessor.preprocess_heart_rate(df, show_progress=False)

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert "heart_rate" in result.columns, "Should preserve HR column"
        assert len(result) == len(df), "Should preserve data length"
        assert len(preprocessor.preprocessing_log) > 0, "Should log preprocessing steps"

    def test_preprocess_heart_rate_missing_column(self):
        """Test heart rate preprocessing with missing column."""
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        config = PreprocessingConfig()
        preprocessor = HeartRatePreprocessor(config)

        result = preprocessor.preprocess_heart_rate(df, show_progress=False)

        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
        assert len(result) == len(df), "Should preserve original data"

    def test_detect_and_handle_outliers(self, sample_hr_data):
        """Test outlier detection and handling."""
        config = PreprocessingConfig(hr_outlier_threshold=2.0)
        preprocessor = HeartRatePreprocessor(config)

        hr_series = sample_hr_data["heart_rate"]
        result = preprocessor._detect_and_handle_outliers(hr_series)

        assert isinstance(result, pd.Series), "Should return Series"
        assert len(result) == len(hr_series), "Should preserve length"

    def test_interpolate_missing_values(self, sample_hr_data):
        """Test missing value interpolation."""
        config = PreprocessingConfig()
        preprocessor = HeartRatePreprocessor(config)

        # Create data with missing values
        hr_series = sample_hr_data["heart_rate"].copy()
        hr_series[100:105] = np.nan  # Insert missing values

        result = preprocessor._interpolate_missing_values(hr_series)

        assert isinstance(result, pd.Series), "Should return Series"
        assert len(result) == len(hr_series), "Should preserve length"
        assert not result.isna().any(), "Should fill missing values"

    def test_smooth_heart_rate(self, sample_hr_data):
        """Test heart rate smoothing."""
        config = PreprocessingConfig()
        preprocessor = HeartRatePreprocessor(config)

        hr_series = sample_hr_data["heart_rate"]
        result = preprocessor._smooth_heart_rate(hr_series)

        assert isinstance(result, pd.Series), "Should return Series"
        assert len(result) == len(hr_series), "Should preserve length"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_dataframe(self):
        """Test preprocessing with empty DataFrame."""
        df = pd.DataFrame()
        config = PreprocessingConfig()

        # EEG preprocessing
        eeg_preprocessor = EEGPreprocessor(config)
        result = eeg_preprocessor.preprocess_eeg(
            df, sampling_rate=1000.0, show_progress=False
        )
        assert isinstance(result, pd.DataFrame), "Should handle empty DataFrame"

        # Pupil preprocessing
        pupil_preprocessor = PupilPreprocessor(config)
        result = pupil_preprocessor.preprocess_pupil(df, show_progress=False)
        assert isinstance(result, pd.DataFrame), "Should handle empty DataFrame"

    def test_single_sample(self):
        """Test preprocessing with single sample."""
        df = pd.DataFrame(
            {
                "eeg_1": [1.0],
                "pupil_diameter": [4.0],
                "eda": [1.0],
                "heart_rate": [70.0],
            }
        )
        config = PreprocessingConfig()

        # Should handle single sample gracefully
        eeg_preprocessor = EEGPreprocessor(config)
        result = eeg_preprocessor.preprocess_eeg(
            df, sampling_rate=1000.0, show_progress=False
        )
        assert isinstance(result, pd.DataFrame), "Should handle single sample"

    def test_nan_data(self):
        """Test preprocessing with NaN values."""
        df = pd.DataFrame(
            {
                "eeg_1": [1.0, np.nan, 3.0],
                "pupil_diameter": [4.0, 4.1, np.nan],
                "eda": [1.0, 1.1, 1.2],
                "heart_rate": [70.0, np.nan, 72.0],
            }
        )
        config = PreprocessingConfig()

        # Should handle NaN values gracefully
        eeg_preprocessor = EEGPreprocessor(config)
        result = eeg_preprocessor.preprocess_eeg(
            df, sampling_rate=1000.0, show_progress=False
        )
        assert isinstance(result, pd.DataFrame), "Should handle NaN values"

    def test_infinite_values(self):
        """Test preprocessing with infinite values."""
        df = pd.DataFrame(
            {
                "eeg_1": [1.0, np.inf, 3.0],
                "pupil_diameter": [4.0, -np.inf, 4.2],
            }
        )
        config = PreprocessingConfig()

        # Should handle infinite values gracefully
        eeg_preprocessor = EEGPreprocessor(config)
        result = eeg_preprocessor.preprocess_eeg(
            df, sampling_rate=1000.0, show_progress=False
        )
        assert isinstance(result, pd.DataFrame), "Should handle infinite values"

    def test_zero_sampling_rate(self):
        """Test with zero sampling rate."""
        df = pd.DataFrame({"eeg_1": [1.0, 2.0, 3.0]})
        config = PreprocessingConfig()
        preprocessor = EEGPreprocessor(config)

        # Should handle zero sampling rate gracefully
        result = preprocessor._apply_bandpass_filter(df["eeg_1"], 0.0)
        assert isinstance(result, pd.Series), "Should handle zero sampling rate"

    def test_invalid_frequency_bands(self):
        """Test with invalid frequency bands."""
        config = PreprocessingConfig(
            eeg_bandpass_low=50.0, eeg_bandpass_high=10.0
        )  # Low > High
        preprocessor = EEGPreprocessor(config)

        df = pd.DataFrame({"eeg_1": [1.0, 2.0, 3.0]})
        result = preprocessor._apply_bandpass_filter(df["eeg_1"], 1000.0)

        # Should handle invalid bands gracefully
        assert isinstance(result, pd.Series), "Should handle invalid frequency bands"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
