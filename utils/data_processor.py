"""
APGI Data Processing Utilities
=============================

Basic data processing utilities for APGI framework data handling.
Provides data loading, preprocessing, and basic analysis functions.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
# warnings.filterwarnings("ignore")  # Removed: Global warning suppression affects entire framework


class DataProcessor:
    """Basic data processing utilities for APGI multimodal data."""

    def __init__(self):
        self.supported_formats = ["csv", "json", "xlsx", "parquet"]
        self.required_columns = ["timestamp", "EEG_Cz", "pupil_diameter"]

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from various file formats.

        Args:
            file_path: Path to the data file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        file_extension = file_path.suffix.lower().lstrip(".")

        if file_extension == "csv":
            data = pd.read_csv(file_path)
        elif file_extension == "json":
            data = pd.read_json(file_path)
        elif file_extension in ["xlsx", "xls"]:
            data = pd.read_excel(file_path)
        elif file_extension == "parquet":
            data = pd.read_parquet(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {self.supported_formats}"
            )

        # Convert timestamp if it's string
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        return data

    def validate_data_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data structure and return validation results.

        Args:
            data: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "summary": {},
        }

        # Check required columns
        missing_columns = []
        for col in self.required_columns:
            if col not in data.columns:
                missing_columns.append(col)
                validation_results["issues"].append(f"Missing required column: {col}")
                validation_results["is_valid"] = False

        # Check data types
        for col in data.columns:
            if col == "timestamp":
                if not pd.api.types.is_datetime64_any_dtype(data[col]):
                    validation_results["warnings"].append(
                        f"Column '{col}' should be datetime type"
                    )
            elif col in ["eeg_fz", "pupil_diameter", "eda", "heart_rate"]:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    validation_results["issues"].append(
                        f"Column '{col}' should be numeric"
                    )
                    validation_results["is_valid"] = False

        # Check for missing values
        null_counts = data.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            null_percentage = (total_nulls / (len(data) * len(data.columns))) * 100
            validation_results["warnings"].append(
                f"{null_percentage:.1f}% of data contains null values"
            )

        # Summary statistics
        validation_results["summary"] = {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "columns": list(data.columns),
            "data_types": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "null_counts": null_counts.to_dict(),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024**2),
        }

        return validation_results

    def clean_data(
        self,
        data: pd.DataFrame,
        remove_outliers: bool = True,
        interpolate_missing: bool = True,
        outlier_method: str = "iqr",
    ) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.

        Args:
            data: Input DataFrame
            remove_outliers: Whether to remove outliers
            interpolate_missing: Whether to interpolate missing values
            outlier_method: Method for outlier detection ('iqr' or 'zscore')

        Returns:
            Cleaned DataFrame
        """
        cleaned_data = data.copy()

        # Handle missing values
        if interpolate_missing:
            # Interpolate numeric columns
            numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
            cleaned_data[numeric_cols] = cleaned_data[numeric_cols].interpolate(
                method="linear"
            )

            # Forward fill remaining NaNs
            cleaned_data = cleaned_data.ffill()

            # Backward fill any remaining NaNs at the beginning
            cleaned_data = cleaned_data.bfill()

        # Remove outliers
        if remove_outliers:
            for col in cleaned_data.select_dtypes(include=[np.number]).columns:
                if col == "timestamp":
                    continue

                if outlier_method == "iqr":
                    # IQR method
                    Q1 = cleaned_data[col].quantile(0.25)
                    Q3 = cleaned_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    cleaned_data = cleaned_data[
                        (cleaned_data[col] >= lower_bound)
                        & (cleaned_data[col] <= upper_bound)
                    ]
                elif outlier_method == "zscore":
                    # Z-score method
                    z_scores = np.abs(
                        (cleaned_data[col] - cleaned_data[col].mean())
                        / cleaned_data[col].std()
                    )
                    cleaned_data = cleaned_data[z_scores < 3]

        return cleaned_data.reset_index(drop=True)

    def normalize_data(
        self,
        data: pd.DataFrame,
        method: str = "zscore",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Normalize specified columns in the data.

        Args:
            data: Input DataFrame
            method: Normalization method ('zscore', 'minmax', 'robust')
            columns: Columns to normalize (default: all numeric columns)

        Returns:
            Normalized DataFrame
        """
        normalized_data = data.copy()

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
            # Remove timestamp from normalization
            if "timestamp" in columns:
                columns.remove("timestamp")

        for col in columns:
            if col not in normalized_data.columns:
                continue

            if method == "zscore":
                # Standard z-score normalization
                mean_val = normalized_data[col].mean()
                std_val = normalized_data[col].std()
                if std_val > 0:
                    normalized_data[col] = (normalized_data[col] - mean_val) / std_val

            elif method == "minmax":
                # Min-max normalization
                min_val = normalized_data[col].min()
                max_val = normalized_data[col].max()
                if max_val > min_val:
                    normalized_data[col] = (normalized_data[col] - min_val) / (
                        max_val - min_val
                    )

            elif method == "robust":
                # Robust normalization using median and IQR
                median_val = normalized_data[col].median()
                mad_val = (normalized_data[col] - median_val).abs().median()
                if mad_val > 0:
                    normalized_data[col] = (normalized_data[col] - median_val) / mad_val

        return normalized_data

    def extract_features(
        self, data: pd.DataFrame, window_size: int = 100, overlap: float = 0.5
    ) -> pd.DataFrame:
        """
        Extract statistical features from sliding windows.

        Args:
            data: Input DataFrame
            window_size: Size of sliding window
            overlap: Overlap between windows (0-1)

        Returns:
            DataFrame with extracted features
        """
        features_list = []
        step_size = int(window_size * (1 - overlap))

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if "timestamp" in numeric_cols:
            numeric_cols.remove("timestamp")

        for start_idx in range(0, len(data) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = data.iloc[start_idx:end_idx]

            feature_row = {
                "window_start": start_idx,
                "window_end": end_idx,
                "timestamp_start": (
                    data.iloc[start_idx]["timestamp"]
                    if "timestamp" in data.columns
                    else None
                ),
                "timestamp_end": (
                    data.iloc[end_idx - 1]["timestamp"]
                    if "timestamp" in data.columns
                    else None
                ),
            }

            # Extract features for each numeric column
            for col in numeric_cols:
                values = window_data[col].values
                feature_row.update(
                    {
                        f"{col}_mean": np.mean(values),
                        f"{col}_std": np.std(values),
                        f"{col}_min": np.min(values),
                        f"{col}_max": np.max(values),
                        f"{col}_median": np.median(values),
                        f"{col}_skew": pd.Series(values).skew(),
                        f"{col}_kurtosis": pd.Series(values).kurtosis(),
                    }
                )

            features_list.append(feature_row)

        return pd.DataFrame(features_list)

    def save_processed_data(
        self, data: pd.DataFrame, output_path: Union[str, Path], format: str = "auto"
    ) -> Path:
        """
        Save processed data to file.

        Args:
            data: DataFrame to save
            output_path: Output file path
            format: Output format ('auto', 'csv', 'json', 'parquet')

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)

        if format == "auto":
            if output_path.suffix.lower() == ".csv":
                format = "csv"
            elif output_path.suffix.lower() == ".json":
                format = "json"
            elif output_path.suffix.lower() == ".parquet":
                format = "parquet"
            else:
                format = "csv"

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            data.to_csv(output_path, index=False)
        elif format == "json":
            # Convert datetime to string for JSON
            json_data = data.copy()
            if "timestamp" in json_data.columns:
                json_data["timestamp"] = json_data["timestamp"].astype(str)
            json_data.to_json(output_path, orient="records", indent=2)
        elif format == "parquet":
            data.to_parquet(output_path, index=False)

        return output_path


def create_data_summary(
    data: pd.DataFrame, output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive data summary.

    Args:
        data: Input DataFrame
        output_path: Optional path to save summary as JSON

    Returns:
        Dictionary containing data summary
    """
    summary = {
        "dataset_info": {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "columns": list(data.columns),
            "data_types": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "memory_usage_mb": data.memory_usage(deep=True).sum() / (1024**2),
        },
        "statistics": {},
        "quality_metrics": {},
    }

    # Basic statistics for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_data = data[col].dropna()
        summary["statistics"][col] = {
            "count": len(col_data),
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "median": float(col_data.median()),
            "skewness": float(col_data.skew()),
            "kurtosis": float(col_data.kurtosis()),
        }

    # Quality metrics
    summary["quality_metrics"] = {
        "completeness": 1
        - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
        "duplicate_rows": data.duplicated().sum(),
        "duplicate_rate": data.duplicated().sum() / len(data),
    }

    # Categorical columns summary
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    summary["categorical_summary"] = {}
    for col in categorical_cols:
        value_counts = data[col].value_counts()
        summary["categorical_summary"][col] = {
            "n_unique": data[col].nunique(),
            "top_values": value_counts.head(5).to_dict(),
            "null_count": data[col].isnull().sum(),
        }

    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    print("APGI Data Processing Utilities")
    print("=" * 40)
    print("Available functions:")
    print("- DataProcessor: Main data processing class")
    print("- create_data_summary: Generate data summary")
    print("\nImport this module to use the utilities.")
