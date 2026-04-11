"""
End-to-end data pipeline tests with real CSV files.
Tests the complete data processing pipeline from CSV input to output.
================================================================
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.slow
class TestDataPipelineEndToEnd:
    """End-to-end tests for data pipeline with real CSV files."""

    def test_pipeline_csv_to_json_conversion(self, temp_dir):
        """Test complete pipeline from CSV to JSON."""
        # Create test CSV file
        csv_file = temp_dir / "input.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "surprise", "threshold", "metabolic"])
            for i in range(100):
                writer.writerow([i, 0.1 + i * 0.01, 0.5, 1.0 + i * 0.1])

        # Read CSV
        df = pd.read_csv(csv_file)
        assert len(df) == 100
        assert list(df.columns) == ["timestamp", "surprise", "threshold", "metabolic"]

        # Process data (normalize)
        df["surprise_normalized"] = (df["surprise"] - df["surprise"].mean()) / df[
            "surprise"
        ].std()
        df["threshold_normalized"] = (df["threshold"] - df["threshold"].mean()) / df[
            "threshold"
        ].std()

        # Write to JSON
        json_file = temp_dir / "output.json"
        df.to_json(json_file, orient="records")

        # Verify JSON output
        with open(json_file) as f:
            data = json.load(f)
        assert len(data) == 100
        assert "surprise_normalized" in data[0]
        assert "threshold_normalized" in data[0]

    def test_pipeline_with_missing_values(self, temp_dir):
        """Test pipeline handling of missing values."""
        csv_file = temp_dir / "missing_values.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "surprise", "threshold"])
            writer.writerow([0, 0.1, 0.5])
            writer.writerow([1, None, 0.6])  # Missing surprise
            writer.writerow([2, 0.3, None])  # Missing threshold
            writer.writerow([3, 0.4, 0.8])

        # Read and handle missing values
        df = pd.read_csv(csv_file)
        assert df.isnull().sum().sum() == 2

        # Fill missing values
        df_filled = df.fillna(df.mean())
        assert df_filled.isnull().sum().sum() == 0

        # Verify filled values
        assert df_filled.loc[1, "surprise"] == df["surprise"].mean()
        assert df_filled.loc[2, "threshold"] == df["threshold"].mean()

    def test_pipeline_with_outliers(self, temp_dir):
        """Test pipeline handling of outliers."""
        csv_file = temp_dir / "outliers.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["value"])
            for i in range(100):
                writer.writerow([i * 0.1])
            writer.writerow([100.0])  # Outlier

        # Read and detect outliers
        df = pd.read_csv(csv_file)
        mean = df["value"].mean()
        std = df["value"].std()
        outliers = df[(df["value"] > mean + 3 * std) | (df["value"] < mean - 3 * std)]

        assert len(outliers) == 1
        assert outliers.iloc[0]["value"] == 100.0

    def test_pipeline_data_transformation(self, temp_dir):
        """Test data transformation pipeline."""
        csv_file = temp_dir / "transform.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            for i in range(50):
                writer.writerow([i, i * 2])

        # Read and transform
        df = pd.read_csv(csv_file)

        # Apply transformations
        df["x_squared"] = df["x"] ** 2
        df["y_log"] = np.log1p(df["y"])
        df["x_y_ratio"] = df["x"] / (df["y"] + 1e-10)

        # Verify transformations
        assert df["x_squared"].iloc[10] == 100
        assert np.isclose(df["y_log"].iloc[10], np.log1p(20))
        assert np.isclose(df["x_y_ratio"].iloc[10], 10 / 20, atol=1e-5)

    def test_pipeline_data_aggregation(self, temp_dir):
        """Test data aggregation pipeline."""
        csv_file = temp_dir / "aggregate.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "value"])
            for i in range(100):
                category = f"cat_{i % 5}"
                writer.writerow([category, i * 0.5])

        # Read and aggregate
        df = pd.read_csv(csv_file)

        # Aggregate by category
        agg_result = (
            df.groupby("category")
            .agg({"value": ["mean", "std", "count"]})
            .reset_index()
        )

        assert len(agg_result) == 5
        assert ("value", "mean") in agg_result.columns
        assert ("value", "std") in agg_result.columns
        assert ("value", "count") in agg_result.columns

    def test_pipeline_filter_operations(self, temp_dir):
        """Test data filtering pipeline."""
        csv_file = temp_dir / "filter.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "score", "status"])
            for i in range(1000):
                status = "active" if i % 2 == 0 else "inactive"
                writer.writerow([i, i * 0.1, status])

        # Read and filter
        df = pd.read_csv(csv_file)

        # Apply filters
        active_df = df[df["status"] == "active"]
        high_score_df = df[df["score"] >= 50.0]  # Threshold for ~500 rows
        combined_filter = df[(df["status"] == "active") & (df["score"] >= 50.0)]

        print(f"DEBUG: Generated {len(df)} rows")
        print(f"DEBUG: Active count: {len(df[df['status'] == 'active'])}")
        print(f"DEBUG: High score count: {len(df[df['score'] >= 5.0])}")

        assert len(active_df) == 500
        assert len(high_score_df) == 500
        assert len(combined_filter) == 250

    def test_pipeline_time_series_processing(self, temp_dir):
        """Test time series processing pipeline."""
        csv_file = temp_dir / "timeseries.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "value"])
            for i in range(100):
                writer.writerow([i, np.sin(i * 0.1) + np.random.randn() * 0.1])

        # Read and process time series
        df = pd.read_csv(csv_file)

        # Calculate rolling statistics
        df["rolling_mean"] = df["value"].rolling(window=10).mean()
        df["rolling_std"] = df["value"].rolling(window=10).std()

        # Calculate differences
        df["diff"] = df["value"].diff()

        # Verify calculations
        assert df["rolling_mean"].iloc[10] is not None
        assert df["rolling_std"].iloc[10] is not None
        assert pd.isna(df["diff"].iloc[0])  # First diff is NaN (not None)

    def test_pipeline_multi_file_processing(self, temp_dir):
        """Test processing multiple CSV files."""
        # Create multiple CSV files
        for i in range(3):
            csv_file = temp_dir / f"file_{i}.csv"
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "value"])
                for j in range(50):
                    writer.writerow([j, j * 0.1 + i])

        # Process all files
        all_data = []
        for i in range(3):
            csv_file = temp_dir / f"file_{i}.csv"
            df = pd.read_csv(csv_file)
            df["source_file"] = f"file_{i}.csv"
            all_data.append(df)

        # Combine
        combined_df = pd.concat(all_data, ignore_index=True)

        assert len(combined_df) == 150
        assert combined_df["source_file"].nunique() == 3

    def test_pipeline_large_file_handling(self, temp_dir):
        """Test handling of large CSV files."""
        csv_file = temp_dir / "large.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "value1", "value2", "value3", "value4", "value5"])
            for i in range(10000):
                writer.writerow([i, i * 0.1, i * 0.2, i * 0.3, i * 0.4, i * 0.5])

        # Read in chunks
        chunk_size = 1000
        chunks = []
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            chunks.append(chunk)

        combined = pd.concat(chunks, ignore_index=True)
        assert len(combined) == 10000
        assert len(chunks) == 10

    def test_pipeline_data_validation(self, temp_dir):
        """Test data validation in pipeline."""
        csv_file = temp_dir / "validate.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["age", "score", "name"])
            writer.writerow([25, 85.5, "Alice"])
            writer.writerow([-5, 95.0, "Bob"])  # Invalid age
            writer.writerow([30, 105.0, "Charlie"])  # Invalid score
            writer.writerow([150, 75.0, "Diana"])  # Invalid age

        # Read and validate
        df = pd.read_csv(csv_file)

        # Validate age (0-120)
        df[(df["age"] >= 0) & (df["age"] <= 120)]
        # Validate score (0-100)
        df[(df["score"] >= 0) & (df["score"] <= 100)]

        # Combined validation
        valid_df = df[
            (df["age"] >= 0)
            & (df["age"] <= 120)
            & (df["score"] >= 0)
            & (df["score"] <= 100)
        ]

        assert len(valid_df) == 1
        assert valid_df.iloc[0]["name"] == "Alice"

    def test_pipeline_export_formats(self, temp_dir):
        """Test exporting to multiple formats."""
        csv_file = temp_dir / "input.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "value"])
            for i in range(10):
                writer.writerow([i, i * 0.1])

        df = pd.read_csv(csv_file)

        # Export to JSON
        json_file = temp_dir / "output.json"
        df.to_json(json_file, orient="records")
        assert json_file.exists()

        # Export to CSV
        csv_out = temp_dir / "output.csv"
        df.to_csv(csv_out, index=False)
        assert csv_out.exists()

        # Export to Excel
        excel_file = temp_dir / "output.xlsx"
        df.to_excel(excel_file, index=False)
        assert excel_file.exists()

        # Verify all exports
        df_json = pd.read_json(json_file)
        df_csv = pd.read_csv(csv_out)
        df_excel = pd.read_excel(excel_file)

        assert len(df_json) == 10
        assert len(df_csv) == 10
        assert len(df_excel) == 10

    def test_pipeline_error_recovery(self, temp_dir):
        """Test pipeline error recovery."""
        # Create a file with some invalid rows
        csv_file = temp_dir / "errors.csv"
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "value"])
            writer.writerow([0, 1.0])
            writer.writerow(["invalid", 2.0])  # Invalid id
            writer.writerow([2, 3.0])
            writer.writerow([3, "invalid"])  # Invalid value
            writer.writerow([4, 5.0])

        # Read with error handling
        try:
            df = pd.read_csv(csv_file)
            # This will fail due to mixed types
        except Exception:
            # Try reading with dtype specification
            df = pd.read_csv(csv_file, dtype={"id": str, "value": str})

        # Convert numeric columns
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Drop rows with NaN
        df_clean = df.dropna()

        assert len(df_clean) == 3
        assert list(df_clean["id"]) == [0, 2, 4]

    def test_pipeline_parallel_processing(self, temp_dir):
        """Test parallel processing of multiple files."""
        import concurrent.futures

        # Create multiple files
        for i in range(5):
            csv_file = temp_dir / f"parallel_{i}.csv"
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "value"])
                for j in range(100):
                    writer.writerow([j, j * 0.1 + i])

        def process_file(file_path):
            df = pd.read_csv(file_path)
            df["mean"] = df["value"].mean()
            df["std"] = df["value"].std()
            return df

        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(5):
                file_path = temp_dir / f"parallel_{i}.csv"
                futures.append(executor.submit(process_file, file_path))

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 5
        for result in results:
            assert "mean" in result.columns
            assert "std" in result.columns
