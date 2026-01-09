#!/usr/bin/env python3
"""
Data Preprocessing Pipelines for APGI Framework
==============================================

Standardized preprocessing pipelines for different types of multimodal data.
Includes EEG preprocessing, physiological signal processing, and data integration.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass
from scipy import signal, stats
import warnings

from data_validation import DataPreprocessor, DataValidator


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipelines."""

    # EEG preprocessing
    eeg_bandpass_low: float = 0.5
    eeg_bandpass_high: float = 40.0
    eeg_notch_freq: float = 50.0  # Power line noise
    eeg_notch_width: float = 2.0
    eeg_artifact_threshold: float = 5.0  # Z-score threshold

    # Pupil preprocessing
    pupil_blink_detection: bool = True
    pupil_interpolation_method: str = "linear"
    pupil_smoothing_window: int = 5

    # EDA preprocessing
    eda_lowpass_cutoff: float = 5.0
    eda_smoothing_window: int = 10

    # Heart rate preprocessing
    hr_outlier_threshold: float = 3.0  # Z-score threshold
    hr_interpolation_method: str = "cubic"

    # General preprocessing
    missing_data_strategy: str = "interpolate"
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    normalization_method: str = "zscore"

    # Resampling
    target_sampling_rate: float = 250.0  # Hz
    resample_method: str = "interpolate"


class EEGPreprocessor:
    """Specialized preprocessing for EEG data."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.preprocessing_log = []

    def preprocess_eeg(
        self, df: pd.DataFrame, eeg_columns: List[str] = None
    ) -> pd.DataFrame:
        """Apply comprehensive EEG preprocessing pipeline."""
        if eeg_columns is None:
            eeg_columns = [col for col in df.columns if col.startswith("eeg")]

        df_processed = df.copy()

        for col in eeg_columns:
            if col in df_processed.columns:
                self.preprocessing_log.append(f"Processing EEG channel: {col}")

                # Step 1: Handle missing values
                df_processed[col] = df_processed[col].interpolate(method="linear")

                # Step 2: Apply bandpass filter
                df_processed[col] = self._apply_bandpass_filter(df_processed[col])

                # Step 3: Apply notch filter for power line noise
                df_processed[col] = self._apply_notch_filter(df_processed[col])

                # Step 4: Detect and correct artifacts
                df_processed[col] = self._detect_and_correct_artifacts(
                    df_processed[col]
                )

        return df_processed

    def _apply_bandpass_filter(self, signal_data: pd.Series) -> pd.Series:
        """Apply bandpass filter to EEG signal."""
        # Estimate sampling rate from time column if available
        fs = self._estimate_sampling_rate(signal_data)

        if fs is None:
            self.preprocessing_log.append(
                "Could not estimate sampling rate, skipping bandpass filter"
            )
            return signal_data

        # Design bandpass filter
        nyquist = fs / 2
        low = self.config.eeg_bandpass_low / nyquist
        high = self.config.eeg_bandpass_high / nyquist

        if low >= high:
            self.preprocessing_log.append(
                f"Invalid filter frequencies: {low} >= {high}"
            )
            return signal_data

        try:
            b, a = signal.butter(4, [low, high], btype="band")
            filtered_data = signal.filtfilt(b, a, signal_data.dropna())

            result = signal_data.copy()
            result.loc[signal_data.dropna().index] = filtered_data
            self.preprocessing_log.append(
                f"Applied bandpass filter ({self.config.eeg_bandpass_low}-{self.config.eeg_bandpass_high} Hz)"
            )

            return result

        except Exception as e:
            self.preprocessing_log.append(f"Error applying bandpass filter: {e}")
            return signal_data

    def _apply_notch_filter(self, signal_data: pd.Series) -> pd.Series:
        """Apply notch filter to remove power line noise."""
        fs = self._estimate_sampling_rate(signal_data)

        if fs is None:
            return signal_data

        try:
            # Design notch filter
            nyquist = fs / 2
            notch_freq = self.config.eeg_notch_freq / nyquist
            notch_width = self.config.eeg_notch_width / nyquist

            low = notch_freq - notch_width / 2
            high = notch_freq + notch_width / 2

            b, a = signal.butter(2, [low, high], btype="bandstop")
            filtered_data = signal.filtfilt(b, a, signal_data.dropna())

            result = signal_data.copy()
            result.loc[signal_data.dropna().index] = filtered_data
            self.preprocessing_log.append(
                f"Applied notch filter at {self.config.eeg_notch_freq} Hz"
            )

            return result

        except Exception as e:
            self.preprocessing_log.append(f"Error applying notch filter: {e}")
            return signal_data

    def _detect_and_correct_artifacts(self, signal_data: pd.Series) -> pd.Series:
        """Detect and correct artifacts in EEG signal."""
        # Calculate Z-scores
        signal_clean = signal_data.dropna()
        if len(signal_clean) == 0:
            return signal_data

        z_scores = np.abs(stats.zscore(signal_clean))
        artifact_mask = z_scores > self.config.eeg_artifact_threshold

        if artifact_mask.any():
            # Replace artifacts with interpolated values
            result = signal_data.copy()
            artifact_indices = signal_clean.index[artifact_mask]

            # Interpolate over artifacts
            result.loc[artifact_indices] = np.nan
            result = result.interpolate(method="linear")

            self.preprocessing_log.append(
                f"Corrected {artifact_mask.sum()} artifacts in {signal_data.name}"
            )
            return result

        return signal_data

    def _estimate_sampling_rate(self, signal_data: pd.Series) -> Optional[float]:
        """Estimate sampling rate from data index."""
        if len(signal_data) < 2:
            return None

        # Try to infer from time column if available
        if hasattr(signal_data, "index"):
            try:
                time_diff = signal_data.index[1] - signal_data.index[0]
                if hasattr(time_diff, "total_seconds"):
                    return 1.0 / time_diff.total_seconds()
            except (IndexError, AttributeError, TypeError):
                pass

        # Default assumption if we can't determine
        return 1000.0  # 1 kHz default for EEG


class PupilPreprocessor:
    """Specialized preprocessing for pupil diameter data."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.preprocessing_log = []

    def preprocess_pupil(
        self, df: pd.DataFrame, pupil_column: str = "pupil_diameter"
    ) -> pd.DataFrame:
        """Apply comprehensive pupil preprocessing pipeline."""
        if pupil_column not in df.columns:
            self.preprocessing_log.append(f"Pupil column '{pupil_column}' not found")
            return df

        df_processed = df.copy()
        self.preprocessing_log.append(f"Processing pupil data: {pupil_column}")

        # Step 1: Detect and handle blinks
        if self.config.pupil_blink_detection:
            df_processed[pupil_column] = self._detect_and_handle_blinks(
                df_processed[pupil_column]
            )

        # Step 2: Smooth the signal
        df_processed[pupil_column] = self._smooth_pupil_signal(
            df_processed[pupil_column]
        )

        # Step 3: Normalize pupil diameter
        df_processed[pupil_column] = self._normalize_pupil_diameter(
            df_processed[pupil_column]
        )

        return df_processed

    def _detect_and_handle_blinks(self, pupil_data: pd.Series) -> pd.Series:
        """Detect blinks and interpolate over them."""
        # Blinks typically cause pupil diameter to drop to near zero
        blink_threshold = 1.0  # mm
        blink_mask = pupil_data < blink_threshold

        if blink_mask.any():
            result = pupil_data.copy()

            # Mark blink periods
            result.loc[blink_mask] = np.nan

            # Interpolate over blinks
            result = result.interpolate(method=self.config.pupil_interpolation_method)

            self.preprocessing_log.append(
                f"Detected and interpolated {blink_mask.sum()} blinks"
            )
            return result

        return pupil_data

    def _smooth_pupil_signal(self, pupil_data: pd.Series) -> pd.Series:
        """Apply smoothing to pupil signal."""
        try:
            # Use moving average smoothing
            window_size = self.config.pupil_smoothing_window
            smoothed_data = pupil_data.rolling(window=window_size, center=True).mean()

            # Fill NaN values at edges with original data
            smoothed_data = smoothed_data.fillna(pupil_data)

            self.preprocessing_log.append(
                f"Applied smoothing with window size {window_size}"
            )
            return smoothed_data

        except Exception as e:
            self.preprocessing_log.append(f"Error smoothing pupil signal: {e}")
            return pupil_data

    def _normalize_pupil_diameter(self, pupil_data: pd.Series) -> pd.Series:
        """Normalize pupil diameter relative to baseline."""
        # Calculate baseline (median of signal)
        baseline = pupil_data.median()

        if baseline > 0:
            # Convert to percentage change from baseline
            normalized_data = ((pupil_data - baseline) / baseline) * 100
            self.preprocessing_log.append(
                f"Normalized pupil diameter (baseline: {baseline:.2f} mm)"
            )
            return normalized_data

        return pupil_data


class EDAPreprocessor:
    """Specialized preprocessing for electrodermal activity data."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.preprocessing_log = []

    def preprocess_eda(self, df: pd.DataFrame, eda_column: str = "eda") -> pd.DataFrame:
        """Apply comprehensive EDA preprocessing pipeline."""
        if eda_column not in df.columns:
            self.preprocessing_log.append(f"EDA column '{eda_column}' not found")
            return df

        df_processed = df.copy()
        self.preprocessing_log.append(f"Processing EDA data: {eda_column}")

        # Step 1: Apply lowpass filter
        df_processed[eda_column] = self._apply_lowpass_filter(df_processed[eda_column])

        # Step 2: Smooth the signal
        df_processed[eda_column] = self._smooth_eda_signal(df_processed[eda_column])

        # Step 3: Extract phasic and tonic components
        df_processed = self._extract_phasic_tonic(df_processed, eda_column)

        return df_processed

    def _apply_lowpass_filter(self, eda_data: pd.Series) -> pd.Series:
        """Apply lowpass filter to EDA signal."""
        fs = self._estimate_sampling_rate(eda_data)

        if fs is None:
            return eda_data

        try:
            nyquist = fs / 2
            cutoff = self.config.eda_lowpass_cutoff / nyquist

            b, a = signal.butter(4, cutoff, btype="low")
            filtered_data = signal.filtfilt(b, a, eda_data.dropna())

            result = eda_data.copy()
            result.loc[eda_data.dropna().index] = filtered_data

            self.preprocessing_log.append(
                f"Applied lowpass filter ({self.config.eda_lowpass_cutoff} Hz cutoff)"
            )
            return result

        except Exception as e:
            self.preprocessing_log.append(f"Error applying lowpass filter: {e}")
            return eda_data

    def _smooth_eda_signal(self, eda_data: pd.Series) -> pd.Series:
        """Apply smoothing to EDA signal."""
        try:
            window_size = self.config.eda_smoothing_window
            smoothed_data = eda_data.rolling(window=window_size, center=True).mean()
            smoothed_data = smoothed_data.fillna(eda_data)

            self.preprocessing_log.append(
                f"Applied EDA smoothing with window size {window_size}"
            )
            return smoothed_data

        except Exception as e:
            self.preprocessing_log.append(f"Error smoothing EDA signal: {e}")
            return eda_data

    def _extract_phasic_tonic(self, df: pd.DataFrame, eda_column: str) -> pd.DataFrame:
        """Extract phasic and tonic components of EDA."""
        try:
            eda_data = df[eda_column]

            # Tonic component (slow varying)
            tonic_window = 30  # 30 second window
            tonic = eda_data.rolling(
                window=int(
                    tonic_window * self._estimate_sampling_rate(eda_data) or 1000
                ),
                center=True,
                min_periods=1,
            ).mean()

            # Phasic component (fast varying)
            phasic = eda_data - tonic

            df[f"{eda_column}_tonic"] = tonic
            df[f"{eda_column}_phasic"] = phasic

            self.preprocessing_log.append("Extracted phasic and tonic EDA components")
            return df

        except Exception as e:
            self.preprocessing_log.append(
                f"Error extracting phasic/tonic components: {e}"
            )
            return df

    def _estimate_sampling_rate(self, signal_data: pd.Series) -> Optional[float]:
        """Estimate sampling rate from data."""
        if len(signal_data) < 2:
            return None

        try:
            time_diff = signal_data.index[1] - signal_data.index[0]
            if hasattr(time_diff, "total_seconds"):
                return 1.0 / time_diff.total_seconds()
        except (IndexError, AttributeError, TypeError):
            pass

        return 1000.0  # Default assumption


class HeartRatePreprocessor:
    """Specialized preprocessing for heart rate data."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.preprocessing_log = []

    def preprocess_heart_rate(
        self, df: pd.DataFrame, hr_column: str = "heart_rate"
    ) -> pd.DataFrame:
        """Apply comprehensive heart rate preprocessing pipeline."""
        if hr_column not in df.columns:
            self.preprocessing_log.append(f"Heart rate column '{hr_column}' not found")
            return df

        df_processed = df.copy()
        self.preprocessing_log.append(f"Processing heart rate data: {hr_column}")

        # Step 1: Detect and handle outliers
        df_processed[hr_column] = self._detect_and_handle_outliers(
            df_processed[hr_column]
        )

        # Step 2: Interpolate missing values
        df_processed[hr_column] = df_processed[hr_column].interpolate(
            method=self.config.hr_interpolation_method
        )

        # Step 3: Smooth the signal
        df_processed[hr_column] = self._smooth_heart_rate(df_processed[hr_column])

        return df_processed

    def _detect_and_handle_outliers(self, hr_data: pd.Series) -> pd.Series:
        """Detect and handle outliers in heart rate data."""
        # Calculate Z-scores
        hr_clean = hr_data.dropna()
        if len(hr_clean) == 0:
            return hr_data

        z_scores = np.abs(stats.zscore(hr_clean))
        outlier_mask = z_scores > self.config.hr_outlier_threshold

        if outlier_mask.any():
            result = hr_data.copy()
            outlier_indices = hr_clean.index[outlier_mask]

            # Replace outliers with NaN for interpolation
            result.loc[outlier_indices] = np.nan

            self.preprocessing_log.append(
                f"Detected and marked {outlier_mask.sum()} heart rate outliers"
            )
            return result

        return hr_data

    def _smooth_heart_rate(self, hr_data: pd.Series) -> pd.Series:
        """Apply smoothing to heart rate signal."""
        try:
            # Use Savitzky-Golay filter for heart rate
            window_length = 11  # Should be odd
            polyorder = 3

            if len(hr_data.dropna()) > window_length:
                smoothed_data = signal.savgol_filter(
                    hr_data.dropna(), window_length, polyorder
                )
                result = hr_data.copy()
                result.loc[hr_data.dropna().index] = smoothed_data

                self.preprocessing_log.append(
                    f"Applied Savitzky-Golay smoothing (window={window_length})"
                )
                return result

        except Exception as e:
            self.preprocessing_log.append(f"Error smoothing heart rate: {e}")

        return hr_data


class MultimodalPreprocessingPipeline:
    """Complete preprocessing pipeline for multimodal data."""

    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        self.preprocessor = DataPreprocessor()
        self.validator = DataValidator()

        # Initialize specialized preprocessors
        self.eeg_processor = EEGPreprocessor(self.config)
        self.pupil_processor = PupilPreprocessor(self.config)
        self.eda_processor = EDAPreprocessor(self.config)
        self.hr_processor = HeartRatePreprocessor(self.config)

        self.pipeline_log = []

    def run_complete_pipeline(
        self,
        input_file: Union[str, Path],
        output_dir: Union[str, Path] = "data/processed",
    ) -> Dict[str, Any]:
        """Run complete preprocessing pipeline."""
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.pipeline_log.append(f"Starting complete pipeline for {input_path.name}")

        # Step 1: Validate input data
        validation_report = self.validator.generate_validation_report(input_path)
        self.pipeline_log.append(
            f"Validation complete - Quality score: {validation_report['data_quality'].get('overall_score', 'N/A')}"
        )

        # Step 2: Load data
        try:
            df = self.preprocessor.load_data(input_path)
            self.pipeline_log.append(f"Loaded data: {df.shape}")
        except Exception as e:
            self.pipeline_log.append(f"Error loading data: {e}")
            return {"error": str(e), "log": self.pipeline_log}

        # Step 3: Apply specialized preprocessing
        df_processed = self._apply_specialized_preprocessing(df)

        # Step 4: General preprocessing
        df_processed = self._apply_general_preprocessing(df_processed)

        # Step 5: Resample if needed
        if "timestamp" in df_processed.columns and len(df_processed) > 100:
            df_processed = self.preprocessor.resample_data(
                df_processed, self.config.target_sampling_rate
            )
            self.pipeline_log.append(
                f"Resampled to {self.config.target_sampling_rate} Hz"
            )

        # Step 6: Final validation
        final_quality = self.validator.validate_data_quality(df_processed)
        self.pipeline_log.append(
            f"Final quality score: {final_quality['overall_score']:.1f}"
        )

        # Step 7: Save processed data
        output_file = output_path / f"{input_path.stem}_processed.json"
        self.preprocessor.save_processed_data(df_processed, output_file, format="json")
        self.pipeline_log.append(f"Saved processed data to {output_file}")

        # Step 8: Save processing report
        report_file = output_path / f"{input_path.stem}_processing_report.json"
        self._save_processing_report(validation_report, final_quality, report_file)

        return {
            "input_file": str(input_path),
            "output_file": str(output_file),
            "report_file": str(report_file),
            "original_shape": df.shape,
            "processed_shape": df_processed.shape,
            "initial_quality": validation_report["data_quality"].get(
                "overall_score", 0
            ),
            "final_quality": final_quality["overall_score"],
            "quality_improvement": final_quality["overall_score"]
            - validation_report["data_quality"].get("overall_score", 0),
            "pipeline_log": self.pipeline_log,
            "preprocessing_steps": self.preprocessor.preprocessing_steps,
        }

    def _apply_specialized_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply specialized preprocessing for each modality."""
        df_processed = df.copy()

        # EEG preprocessing
        eeg_columns = [col for col in df_processed.columns if col.startswith("eeg")]
        if eeg_columns:
            df_processed = self.eeg_processor.preprocess_eeg(df_processed, eeg_columns)
            self.pipeline_log.extend(self.eeg_processor.preprocessing_log)

        # Pupil preprocessing
        if "pupil_diameter" in df_processed.columns:
            df_processed = self.pupil_processor.preprocess_pupil(df_processed)
            self.pipeline_log.extend(self.pupil_processor.preprocessing_log)

        # EDA preprocessing
        if "eda" in df_processed.columns:
            df_processed = self.eda_processor.preprocess_eda(df_processed)
            self.pipeline_log.extend(self.eda_processor.preprocessing_log)

        # Heart rate preprocessing
        if "heart_rate" in df_processed.columns:
            df_processed = self.hr_processor.preprocess_heart_rate(df_processed)
            self.pipeline_log.extend(self.hr_processor.preprocessing_log)

        return df_processed

    def _apply_general_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply general preprocessing steps."""
        df_processed = df.copy()

        # Handle missing data
        df_processed = self.preprocessor.clean_missing_data(
            df_processed, strategy=self.config.missing_data_strategy
        )

        # Remove outliers
        df_processed = self.preprocessor.remove_outliers(
            df_processed,
            method=self.config.outlier_method,
            threshold=self.config.outlier_threshold,
        )

        # Normalize data
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed = self.preprocessor.normalize_data(
            df_processed,
            method=self.config.normalization_method,
            columns=list(numeric_cols),
        )

        return df_processed

    def _save_processing_report(
        self, initial_validation: Dict, final_quality: Dict, report_file: Path
    ):
        """Save comprehensive processing report."""
        report = {
            "processing_timestamp": datetime.now().isoformat(),
            "pipeline_config": self.config.__dict__,
            "initial_validation": initial_validation,
            "final_quality": final_quality,
            "pipeline_log": self.pipeline_log,
            "preprocessing_steps": self.preprocessor.preprocessing_steps,
            "quality_improvement": final_quality["overall_score"]
            - initial_validation["data_quality"].get("overall_score", 0),
        }

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)


def main():
    """Demonstrate preprocessing pipelines."""
    print("APGI Framework - Data Preprocessing Pipelines")
    print("=" * 50)

    # Initialize pipeline
    config = PreprocessingConfig()
    pipeline = MultimodalPreprocessingPipeline(config)

    # Process demo data
    demo_file = Path("data/demo_demo.csv")
    if demo_file.exists():
        print(f"Processing demo file: {demo_file.name}")

        result = pipeline.run_complete_pipeline(demo_file)

        print(f"  Original shape: {result['original_shape']}")
        print(f"  Processed shape: {result['processed_shape']}")
        print(f"  Quality improvement: {result['quality_improvement']:.1f} points")
        print(f"  Output file: {result['output_file']}")
        print(f"  Report file: {result['report_file']}")

        print("\nProcessing steps:")
        for step in result["preprocessing_steps"][:5]:
            print(f"  - {step}")

        print("\nPipeline log:")
        for log_entry in result["pipeline_log"][:5]:
            print(f"  - {log_entry}")

    else:
        print(f"Demo file not found: {demo_file}")
        print("Please run the sample data generator first.")

    print("\nPreprocessing pipelines ready!")


if __name__ == "__main__":
    main()
