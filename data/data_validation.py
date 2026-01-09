#!/usr/bin/env python3
"""
Data Validation Utilities for APGI Framework
============================================

Comprehensive data validation and quality assessment tools
for multimodal physiological data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import warnings


class DataValidator:
    """Comprehensive data validation for APGI framework."""

    def __init__(self, strict_mode=False):
        self.strict_mode = strict_mode
        self.validation_results = {}

    def validate_file_format(self, file_path: Union[str, Path]) -> Dict:
        """Validate file format and structure."""
        file_path = Path(file_path)
        results = {
            "file_path": str(file_path),
            "file_exists": file_path.exists(),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "file_extension": file_path.suffix.lower(),
            "is_readable": False,
            "format_valid": False,
            "errors": [],
            "warnings": [],
        }

        if not file_path.exists():
            results["errors"].append(f"File does not exist: {file_path}")
            return results

        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
                results["is_readable"] = True
                results["format_valid"] = self._validate_csv_structure(df, results)
            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                results["is_readable"] = True
                results["format_valid"] = self._validate_json_structure(data, results)
            else:
                results["errors"].append(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            results["errors"].append(f"Error reading file: {e}")

        return results

    def _validate_csv_structure(self, df: pd.DataFrame, results: Dict) -> bool:
        """Validate CSV DataFrame structure."""
        required_columns = ["timestamp", "eeg_fz", "pupil_diameter", "eda"]
        optional_columns = [
            "eeg_pz",
            "heart_rate",
            "event_marker",
            "subject_id",
            "session_id",
        ]

        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            results["errors"].append(f"Missing required columns: {missing_columns}")
            return False

        # Check data types
        for col in required_columns:
            if col == "timestamp":
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        pd.to_datetime(df[col])
                    except (ValueError, TypeError):
                        results["errors"].append(
                            f"Column {col} cannot be converted to datetime"
                        )
                        return False
            else:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        pd.to_numeric(df[col], errors="coerce")
                    except (ValueError, TypeError):
                        results["errors"].append(
                            f"Column {col} cannot be converted to numeric"
                        )
                        return False

        # Check for missing values
        missing_counts = df[required_columns].isnull().sum()
        if missing_counts.any():
            results["warnings"].append(
                f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}"
            )

        # Check data ranges
        self._validate_data_ranges(df, results)

        return len(results["errors"]) == 0

    def _validate_json_structure(self, data: Dict, results: Dict) -> bool:
        """Validate JSON data structure."""
        if not isinstance(data, dict):
            results["errors"].append("JSON root must be an object")
            return False

        # Check for required sections
        if "metadata" not in data:
            results["errors"].append("Missing 'metadata' section")
            return False

        if "data" not in data:
            results["errors"].append("Missing 'data' section")
            return False

        # Validate metadata
        metadata = data["metadata"]
        required_metadata = ["subject_id", "session_id", "sampling_rate", "duration"]
        for field in required_metadata:
            if field not in metadata:
                results["errors"].append(f"Missing metadata field: {field}")

        # Validate data records
        data_records = data["data"]
        if not isinstance(data_records, list):
            results["errors"].append("'data' section must be a list")
            return False

        if len(data_records) == 0:
            results["errors"].append("'data' section is empty")
            return False

        # Check first record structure
        first_record = data_records[0]
        required_fields = ["timestamp", "eeg_fz", "pupil_diameter", "eda"]
        for field in required_fields:
            if field not in first_record:
                results["errors"].append(f"Missing data field: {field}")

        return len(results["errors"]) == 0

    def _validate_data_ranges(self, df: pd.DataFrame, results: Dict):
        """Validate physiological data ranges."""
        # EEG ranges (microvolts)
        eeg_cols = [col for col in df.columns if col.startswith("eeg")]
        for col in eeg_cols:
            if col in df.columns:
                eeg_data = df[col].dropna()
                if len(eeg_data) > 0:
                    eeg_min, eeg_max = eeg_data.min(), eeg_data.max()
                    if eeg_min < -500 or eeg_max > 500:
                        results["warnings"].append(
                            f"{col}: Unusual EEG range ({eeg_min:.2f} to {eeg_max:.2f} μV)"
                        )

        # Pupil diameter (mm)
        if "pupil_diameter" in df.columns:
            pupil_data = df["pupil_diameter"].dropna()
            if len(pupil_data) > 0:
                pupil_min, pupil_max = pupil_data.min(), pupil_data.max()
                if pupil_min < 1 or pupil_max > 10:
                    results["warnings"].append(
                        f"Pupil diameter: Unusual range ({pupil_min:.2f} to {pupil_max:.2f} mm)"
                    )

        # EDA (microsiemens)
        if "eda" in df.columns:
            eda_data = df["eda"].dropna()
            if len(eda_data) > 0:
                eda_min, eda_max = eda_data.min(), eda_data.max()
                if eda_min < 0 or eda_max > 10:
                    results["warnings"].append(
                        f"EDA: Unusual range ({eda_min:.3f} to {eda_max:.3f} μS)"
                    )

        # Heart rate (BPM)
        if "heart_rate" in df.columns:
            hr_data = df["heart_rate"].dropna()
            if len(hr_data) > 0:
                hr_min, hr_max = hr_data.min(), hr_data.max()
                if hr_min < 30 or hr_max > 200:
                    results["warnings"].append(
                        f"Heart rate: Unusual range ({hr_min:.1f} to {hr_max:.1f} BPM)"
                    )

    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Assess data quality metrics."""
        quality_metrics = {
            "total_samples": len(df),
            "missing_data": {},
            "outliers": {},
            "signal_quality": {},
            "temporal_consistency": {},
            "overall_score": 0.0,
        }

        # Missing data analysis
        for col in df.columns:
            if df[col].dtype in ["float64", "int64"]:
                missing_count = df[col].isnull().sum()
                missing_percent = (missing_count / len(df)) * 100
                quality_metrics["missing_data"][col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_percent),
                }

        # Outlier detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                    quality_metrics["outliers"][col] = {
                        "count": len(outliers),
                        "percentage": (len(outliers) / len(data)) * 100,
                    }

        # Signal quality metrics
        if "eeg_fz" in df.columns:
            quality_metrics["signal_quality"]["eeg_fz"] = self._assess_signal_quality(
                df["eeg_fz"]
            )

        if "pupil_diameter" in df.columns:
            quality_metrics["signal_quality"]["pupil_diameter"] = (
                self._assess_signal_quality(df["pupil_diameter"])
            )

        if "eda" in df.columns:
            quality_metrics["signal_quality"]["eda"] = self._assess_signal_quality(
                df["eda"]
            )

        # Temporal consistency
        if "timestamp" in df.columns:
            quality_metrics["temporal_consistency"] = self._assess_temporal_consistency(
                df
            )

        # Calculate overall quality score
        quality_metrics["overall_score"] = self._calculate_quality_score(
            quality_metrics
        )

        return quality_metrics

    def _assess_signal_quality(self, signal: pd.Series) -> Dict:
        """Assess quality of a physiological signal."""
        signal_clean = signal.dropna()
        if len(signal_clean) == 0:
            return {"score": 0.0, "issues": ["No valid data"]}

        issues = []
        score = 100.0

        # Check for flat segments
        if len(signal_clean.unique()) < len(signal_clean) * 0.1:
            issues.append("Many repeated values")
            score -= 20

        # Check for extreme values
        signal_mean = signal_clean.mean()
        signal_std = signal_clean.std()

        if signal_std == 0:
            issues.append("No signal variation")
            score -= 30
        else:
            z_scores = np.abs((signal_clean - signal_mean) / signal_std)
            extreme_count = (z_scores > 5).sum()
            if extreme_count > len(signal_clean) * 0.01:
                issues.append(f"Many extreme values ({extreme_count})")
                score -= 15

        # Check for sudden jumps
        if len(signal_clean) > 1:
            diff = np.diff(signal_clean)
            jump_threshold = signal_std * 5
            jumps = np.abs(diff) > jump_threshold
            if jumps.sum() > len(diff) * 0.001:
                issues.append("Sudden jumps detected")
                score -= 10

        return {
            "score": max(0, score),
            "issues": issues,
            "mean": float(signal_mean),
            "std": float(signal_std),
            "samples": len(signal_clean),
        }

    def _assess_temporal_consistency(self, df: pd.DataFrame) -> Dict:
        """Assess temporal consistency of the data."""
        if "timestamp" not in df.columns:
            return {"score": 0.0, "issues": ["No timestamp column"]}

        try:
            timestamps = pd.to_datetime(df["timestamp"])
            time_diffs = timestamps.diff().dropna()

            issues = []
            score = 100.0

            # Check for irregular sampling
            if len(time_diffs) > 0:
                expected_interval = time_diffs.mode().iloc[0]  # Most common interval
                irregular_diffs = time_diffs[time_diffs != expected_interval]

                if len(irregular_diffs) > len(time_diffs) * 0.05:
                    issues.append(
                        f"Irregular sampling: {len(irregular_diffs)} irregular intervals"
                    )
                    score -= 20

                # Check for gaps
                large_gaps = time_diffs[time_diffs > expected_interval * 2]
                if len(large_gaps) > 0:
                    issues.append(f"Data gaps: {len(large_gaps)} large gaps detected")
                    score -= 15

            return {
                "score": max(0, score),
                "issues": issues,
                "expected_interval": (
                    str(expected_interval) if len(time_diffs) > 0 else "unknown"
                ),
                "total_duration": (
                    str(timestamps.iloc[-1] - timestamps.iloc[0])
                    if len(timestamps) > 1
                    else "unknown"
                ),
            }

        except Exception as e:
            return {"score": 0.0, "issues": [f"Error processing timestamps: {e}"]}

    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall data quality score."""
        score = 100.0

        # Penalize missing data
        total_missing = sum(
            info["percentage"] for info in metrics["missing_data"].values()
        )
        if total_missing > 10:
            score -= min(30, total_missing / 2)

        # Penalize outliers
        total_outliers = sum(
            info["percentage"] for info in metrics["outliers"].values()
        )
        if total_outliers > 5:
            score -= min(20, total_outliers)

        # Penalize poor signal quality
        signal_scores = [info["score"] for info in metrics["signal_quality"].values()]
        if signal_scores:
            avg_signal_score = sum(signal_scores) / len(signal_scores)
            score -= (100 - avg_signal_score) * 0.3

        # Penalize temporal inconsistency
        if "temporal_consistency" in metrics:
            temporal_score = metrics["temporal_consistency"]["score"]
            score -= (100 - temporal_score) * 0.2

        return max(0, score)

    def generate_validation_report(self, file_path: Union[str, Path]) -> Dict:
        """Generate comprehensive validation report."""
        file_path = Path(file_path)

        report = {
            "file_info": self.validate_file_format(file_path),
            "data_quality": {},
            "recommendations": [],
            "validation_timestamp": datetime.now().isoformat(),
            "validator_version": "1.0.0",
        }

        # Load data for quality assessment
        if report["file_info"]["is_readable"]:
            try:
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path)
                elif file_path.suffix.lower() == ".json":
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    df = pd.DataFrame(data["data"])

                report["data_quality"] = self.validate_data_quality(df)
                report["recommendations"] = self._generate_recommendations(report)

            except Exception as e:
                report["data_quality"]["error"] = str(e)

        return report

    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate data improvement recommendations."""
        recommendations = []

        # File format recommendations
        if report["file_info"]["errors"]:
            recommendations.append("Fix file format errors before processing")

        # Missing data recommendations
        missing_data = report["data_quality"].get("missing_data", {})
        for col, info in missing_data.items():
            if info["percentage"] > 10:
                recommendations.append(
                    f"Consider imputation or removal of missing data in {col}"
                )

        # Outlier recommendations
        outliers = report["data_quality"].get("outliers", {})
        for col, info in outliers.items():
            if info["percentage"] > 5:
                recommendations.append(
                    f"Review outliers in {col} ({info['percentage']:.1f}%)"
                )

        # Signal quality recommendations
        signal_quality = report["data_quality"].get("signal_quality", {})
        for col, info in signal_quality.items():
            if info["score"] < 70:
                recommendations.append(
                    f"Improve signal quality for {col}: {', '.join(info['issues'])}"
                )

        # Temporal consistency recommendations
        temporal = report["data_quality"].get("temporal_consistency", {})
        if temporal.get("score", 100) < 80:
            recommendations.append(
                f"Address temporal consistency issues: {', '.join(temporal.get('issues', []))}"
            )

        # Overall quality recommendations
        overall_score = report["data_quality"].get("overall_score", 100)
        if overall_score < 70:
            recommendations.append(
                "Overall data quality is poor - consider data cleaning and preprocessing"
            )
        elif overall_score < 85:
            recommendations.append(
                "Data quality is acceptable but could be improved with preprocessing"
            )

        return recommendations


class DataPreprocessor:
    """Data preprocessing utilities for APGI framework."""

    def __init__(self):
        self.preprocessing_steps = []

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from various formats."""
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() == ".json":
            with open(file_path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data["data"])
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Convert timestamp if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def clean_missing_data(
        self, df: pd.DataFrame, strategy: str = "interpolate"
    ) -> pd.DataFrame:
        """Clean missing data using various strategies."""
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        if strategy == "interpolate":
            for col in numeric_cols:
                df_clean[col] = df_clean[col].interpolate(method="linear")
        elif strategy == "forward_fill":
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method="ffill")
        elif strategy == "backward_fill":
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(method="bfill")
        elif strategy == "mean":
            for col in numeric_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == "median":
            for col in numeric_cols:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == "drop":
            df_clean = df_clean.dropna()

        self.preprocessing_steps.append(
            f"Missing data cleaned using {strategy} strategy"
        )
        return df_clean

    def remove_outliers(
        self, df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5
    ) -> pd.DataFrame:
        """Remove outliers from numeric columns."""
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        outliers_removed = 0

        for col in numeric_cols:
            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outlier_mask = (df_clean[col] < lower_bound) | (
                    df_clean[col] > upper_bound
                )
                outliers_removed += outlier_mask.sum()
                df_clean = df_clean[~outlier_mask]

            elif method == "zscore":
                z_scores = np.abs(
                    (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
                )
                outlier_mask = z_scores > threshold
                outliers_removed += outlier_mask.sum()
                df_clean = df_clean[~outlier_mask]

        self.preprocessing_steps.append(
            f"Removed {outliers_removed} outliers using {method} method"
        )
        return df_clean

    def filter_signals(
        self,
        df: pd.DataFrame,
        filter_type: str = "bandpass",
        columns: List[str] = None,
        **filter_params,
    ) -> pd.DataFrame:
        """Apply signal filtering to specified columns."""
        from scipy import signal

        df_filtered = df.copy()

        if columns is None:
            columns = [
                col
                for col in df.columns
                if col.startswith("eeg") or col.startswith("pupil") or col == "eda"
            ]

        for col in columns:
            if col in df_filtered.columns:
                data = df_filtered[col].dropna()

                if filter_type == "bandpass":
                    low_freq = filter_params.get("low_freq", 0.5)
                    high_freq = filter_params.get("high_freq", 40.0)
                    fs = filter_params.get("sampling_rate", 1000.0)

                    nyquist = fs / 2
                    low = low_freq / nyquist
                    high = high_freq / nyquist

                    b, a = signal.butter(4, [low, high], btype="band")
                    filtered_data = signal.filtfilt(b, a, data)

                    df_filtered.loc[data.index, col] = filtered_data

                elif filter_type == "lowpass":
                    cutoff_freq = filter_params.get("cutoff_freq", 40.0)
                    fs = filter_params.get("sampling_rate", 1000.0)

                    nyquist = fs / 2
                    cutoff = cutoff_freq / nyquist

                    b, a = signal.butter(4, cutoff, btype="low")
                    filtered_data = signal.filtfilt(b, a, data)

                    df_filtered.loc[data.index, col] = filtered_data

        self.preprocessing_steps.append(f"Applied {filter_type} filter to {columns}")
        return df_filtered

    def normalize_data(
        self, df: pd.DataFrame, method: str = "zscore", columns: List[str] = None
    ) -> pd.DataFrame:
        """Normalize data using various methods."""
        df_normalized = df.copy()

        if columns is None:
            columns = df_normalized.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col in df_normalized.columns:
                data = df_normalized[col]

                if method == "zscore":
                    df_normalized[col] = (data - data.mean()) / data.std()
                elif method == "minmax":
                    df_normalized[col] = (data - data.min()) / (data.max() - data.min())
                elif method == "robust":
                    median = data.median()
                    mad = np.median(np.abs(data - median))
                    df_normalized[col] = (data - median) / mad

        self.preprocessing_steps.append(f"Normalized {columns} using {method} method")
        return df_normalized

    def resample_data(
        self, df: pd.DataFrame, target_rate: float, time_column: str = "timestamp"
    ) -> pd.DataFrame:
        """Resample data to target sampling rate."""
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found")

        df_resampled = df.copy()
        df_resampled = df_resampled.set_index(time_column)

        # Calculate target frequency
        target_freq = f"{int(target_rate)}S"

        # Resample numeric columns
        numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns
        df_resampled = df_resampled[numeric_cols].resample(target_freq).mean()

        # Forward fill non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            df_non_numeric = df[non_numeric_cols].copy()
            df_non_numeric[time_column] = pd.to_datetime(df[time_column])
            df_non_numeric = df_non_numeric.set_index(time_column)
            df_non_numeric = df_non_numeric.resample(target_freq).ffill()
            df_resampled = pd.concat([df_resampled, df_non_numeric], axis=1)

        self.preprocessing_steps.append(f"Resampled data to {target_rate} Hz")
        return df_resampled.reset_index()

    def save_processed_data(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        format: str = "csv",
        include_metadata: bool = True,
    ):
        """Save processed data with metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            data_dict = {
                "data": df.to_dict("records"),
                "preprocessing_steps": self.preprocessing_steps,
                "processing_timestamp": datetime.now().isoformat(),
            }

            if include_metadata:
                data_dict["metadata"] = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                }

            with open(output_path, "w") as f:
                json.dump(data_dict, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported output format: {format}")


def main():
    """Demonstrate data validation and preprocessing."""
    print("APGI Framework - Data Validation & Preprocessing")
    print("=" * 50)

    # Initialize validator
    validator = DataValidator(strict_mode=True)

    # Validate sample data files
    data_dir = Path("data")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))

        for csv_file in csv_files[:2]:  # Validate first 2 files
            print(f"\nValidating: {csv_file.name}")
            report = validator.generate_validation_report(csv_file)

            print(f"  File readable: {report['file_info']['is_readable']}")
            print(f"  Format valid: {report['file_info']['format_valid']}")
            print(
                f"  Overall quality: {report['data_quality'].get('overall_score', 'N/A'):.1f}"
            )

            if report["recommendations"]:
                print("  Recommendations:")
                for rec in report["recommendations"][:3]:
                    print(f"    - {rec}")

    # Demonstrate preprocessing
    print(f"\nDemonstrating preprocessing pipeline...")

    try:
        preprocessor = DataPreprocessor()

        # Load demo data
        demo_file = data_dir / "demo_demo.csv"
        if demo_file.exists():
            df = preprocessor.load_data(demo_file)
            print(f"  Loaded data: {df.shape}")

            # Clean missing data
            df_clean = preprocessor.clean_missing_data(df, strategy="interpolate")
            print(f"  Cleaned missing data: {df_clean.shape}")

            # Remove outliers
            df_no_outliers = preprocessor.remove_outliers(df_clean, method="iqr")
            print(f"  Removed outliers: {df_no_outliers.shape}")

            # Normalize data
            df_normalized = preprocessor.normalize_data(df_no_outliers, method="zscore")
            print(f"  Normalized data: {df_normalized.shape}")

            # Save processed data
            output_file = data_dir / "demo_demo_processed.json"
            preprocessor.save_processed_data(df_normalized, output_file, format="json")
            print(f"  Saved processed data: {output_file}")

    except Exception as e:
        print(f"  Error in preprocessing: {e}")

    print("\nData validation and preprocessing utilities ready!")


if __name__ == "__main__":
    main()
