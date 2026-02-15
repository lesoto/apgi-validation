"""
APGI Data Quality Metrics and Anomaly Detection
===============================================

Comprehensive data quality assessment and anomaly detection system for APGI framework.
Provides quality scoring, anomaly detection, and data validation insights.
"""

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# APGI imports
from logging_config import apgi_logger
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class DataQualityMetric:
    """Individual data quality metric."""

    name: str
    value: float
    threshold: float
    status: str  # 'good', 'warning', 'critical'
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""

    type: str
    severity: str  # 'low', 'medium', 'high'
    description: str
    affected_features: List[str]
    confidence: float
    timestamp: datetime
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""

    dataset_name: str
    overall_score: float
    grade: str  # 'A', 'B', 'C', 'D', 'F'
    metrics: List[DataQualityMetric]
    anomalies: List[AnomalyDetection]
    summary: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime


class DataQualityAssessment:
    """Advanced data quality assessment and anomaly detection."""

    def __init__(self):
        self.quality_thresholds = {
            "completeness": {"good": 0.95, "warning": 0.80, "critical": 0.60},
            "uniqueness": {"good": 0.95, "warning": 0.85, "critical": 0.70},
            "validity": {"good": 0.95, "warning": 0.85, "critical": 0.70},
            "consistency": {"good": 0.90, "warning": 0.75, "critical": 0.60},
            "accuracy": {"good": 0.90, "warning": 0.80, "critical": 0.65},
        }

    def assess_completeness(self, data: pd.DataFrame) -> DataQualityMetric:
        """Assess data completeness (missing values)."""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)

        # Missing values by column
        missing_by_column = (data.isnull().sum() / len(data)).to_dict()

        # Determine status
        thresholds = self.quality_thresholds["completeness"]
        if completeness >= thresholds["good"]:
            status = "good"
        elif completeness >= thresholds["warning"]:
            status = "warning"
        else:
            status = "critical"

        return DataQualityMetric(
            name="completeness",
            value=completeness,
            threshold=thresholds["warning"],
            status=status,
            description=f"Data completeness: {completeness:.1%} of values present",
            details={
                "missing_cells": missing_cells,
                "total_cells": total_cells,
                "missing_by_column": missing_by_column,
                "columns_with_missing": [
                    col for col, pct in missing_by_column.items() if pct > 0
                ],
            },
        )

    def assess_uniqueness(self, data: pd.DataFrame) -> DataQualityMetric:
        """Assess data uniqueness (duplicate records)."""
        total_records = len(data)
        duplicate_records = data.duplicated().sum()
        unique_records = total_records - duplicate_records
        uniqueness = unique_records / total_records

        # Duplicates by key columns (if any)
        duplicate_details = {}
        for col in data.columns:
            if data[col].dtype in ["object", "int64", "float64"]:
                col_duplicates = data[col].duplicated().sum()
                if col_duplicates > 0:
                    duplicate_details[col] = {
                        "duplicates": col_duplicates,
                        "unique_values": data[col].nunique(),
                        "duplicate_rate": col_duplicates / total_records,
                    }

        # Determine status
        thresholds = self.quality_thresholds["uniqueness"]
        if uniqueness >= thresholds["good"]:
            status = "good"
        elif uniqueness >= thresholds["warning"]:
            status = "warning"
        else:
            status = "critical"

        return DataQualityMetric(
            name="uniqueness",
            value=uniqueness,
            threshold=thresholds["warning"],
            status=status,
            description=f"Data uniqueness: {uniqueness:.1%} of records are unique",
            details={
                "total_records": total_records,
                "duplicate_records": duplicate_records,
                "unique_records": unique_records,
                "duplicate_details": duplicate_details,
            },
        )

    def assess_validity(
        self, data: pd.DataFrame, schema: Optional[Dict[str, Dict]] = None
    ) -> DataQualityMetric:
        """Assess data validity (data type and range validation)."""
        validity_issues = 0
        total_checks = 0
        validity_details = {}

        # Check data types
        for col in data.columns:
            col_data = data[col]

            # Numeric range checks
            if pd.api.types.is_numeric_dtype(col_data):
                total_checks += 1

                # Check for negative values where inappropriate
                if col.lower().startswith(
                    ("count", "amount", "size", "duration", "age")
                ):
                    negative_count = (col_data < 0).sum()
                    if negative_count > 0:
                        validity_issues += negative_count
                        validity_details[f"{col}_negative"] = negative_count

                # Check for extreme outliers (beyond 5 standard deviations)
                if col_data.std() > 0:
                    z_scores = np.abs(stats.zscore(col_data.dropna()))
                    outliers = (z_scores > 5).sum()
                    if outliers > 0:
                        validity_issues += outliers
                        validity_details[f"{col}_outliers"] = outliers

            # String validation
            elif pd.api.types.is_string_dtype(col_data) or col_data.dtype == "object":
                total_checks += 1

                # Check for empty strings
                empty_strings = (col_data == "").sum()
                if empty_strings > 0:
                    validity_issues += empty_strings
                    validity_details[f"{col}_empty"] = empty_strings

                # Check for whitespace-only strings
                whitespace_only = col_data.str.strip().eq("").sum()
                if whitespace_only > 0:
                    validity_issues += whitespace_only
                    validity_details[f"{col}_whitespace"] = whitespace_only

        # Calculate validity score
        if total_checks > 0:
            validity = 1 - (validity_issues / (len(data) * total_checks))
        else:
            validity = 1.0

        # Determine status
        thresholds = self.quality_thresholds["validity"]
        if validity >= thresholds["good"]:
            status = "good"
        elif validity >= thresholds["warning"]:
            status = "warning"
        else:
            status = "critical"

        return DataQualityMetric(
            name="validity",
            value=validity,
            threshold=thresholds["warning"],
            status=status,
            description=f"Data validity: {validity:.1%} of values pass validation checks",
            details={
                "validity_issues": validity_issues,
                "total_checks": total_checks,
                "validity_details": validity_details,
            },
        )

    def assess_consistency(self, data: pd.DataFrame) -> DataQualityMetric:
        """Assess data consistency (format and value consistency)."""
        consistency_issues = 0
        total_checks = 0
        consistency_details = {}

        for col in data.columns:
            col_data = data[col]

            # Skip if all values are null
            if col_data.isnull().all():
                continue

            total_checks += 1

            # Check for inconsistent case in string columns
            if pd.api.types.is_string_dtype(col_data) or col_data.dtype == "object":
                non_null_values = col_data.dropna()
                if len(non_null_values) > 1:
                    # Check for mixed case
                    unique_cases = set()
                    for val in non_null_values:
                        if isinstance(val, str):
                            if val.isupper():
                                unique_cases.add("upper")
                            elif val.islower():
                                unique_cases.add("lower")
                            elif val[0].isupper() and val[1:].islower():
                                unique_cases.add("title")
                            else:
                                unique_cases.add("mixed")

                    if len(unique_cases) > 1:
                        consistency_issues += len(non_null_values)
                        consistency_details[f"{col}_case_inconsistency"] = list(
                            unique_cases
                        )

            # Check for inconsistent date formats
            elif "date" in col.lower() or "time" in col.lower():
                try:
                    # Try to parse as datetime
                    pd.to_datetime(col_data, errors="raise")
                except (ValueError, TypeError):
                    consistency_issues += col_data.notna().sum()
                    consistency_details[f"{col}_date_format"] = (
                        "Inconsistent date formats"
                    )

        # Calculate consistency score
        if total_checks > 0:
            consistency = 1 - (consistency_issues / (len(data) * total_checks))
        else:
            consistency = 1.0

        # Determine status
        thresholds = self.quality_thresholds["consistency"]
        if consistency >= thresholds["good"]:
            status = "good"
        elif consistency >= thresholds["warning"]:
            status = "warning"
        else:
            status = "critical"

        return DataQualityMetric(
            name="consistency",
            value=consistency,
            threshold=thresholds["warning"],
            status=status,
            description=f"Data consistency: {consistency:.1%} of values are consistent",
            details={
                "consistency_issues": consistency_issues,
                "total_checks": total_checks,
                "consistency_details": consistency_details,
            },
        )

    def assess_accuracy(
        self, data: pd.DataFrame, reference_data: Optional[pd.DataFrame] = None
    ) -> DataQualityMetric:
        """Assess data accuracy (statistical plausibility)."""
        accuracy_score = 1.0  # Start with perfect score
        accuracy_details = {}

        for col in data.columns:
            col_data = data[col].dropna()

            if len(col_data) == 0:
                continue

            # Statistical accuracy checks for numeric data
            if pd.api.types.is_numeric_dtype(col_data):
                # Check for impossible values
                if col.lower().startswith(("age", "years")):
                    # Age should be reasonable (0-120)
                    invalid_ages = ((col_data < 0) | (col_data > 120)).sum()
                    if invalid_ages > 0:
                        accuracy_score -= (invalid_ages / len(col_data)) * 0.1
                        accuracy_details[f"{col}_invalid_age"] = invalid_ages

                elif col.lower().startswith(("percentage", "percent", "rate")):
                    # Percentages should be 0-100
                    invalid_percentages = ((col_data < 0) | (col_data > 100)).sum()
                    if invalid_percentages > 0:
                        accuracy_score -= (invalid_percentages / len(col_data)) * 0.1
                        accuracy_details[f"{col}_invalid_percentage"] = (
                            invalid_percentages
                        )

                # Check for statistical outliers
                if len(col_data) > 10 and col_data.std() > 0:
                    z_scores = np.abs(stats.zscore(col_data))
                    extreme_outliers = (z_scores > 4).sum()
                    if extreme_outliers > 0:
                        accuracy_score -= (extreme_outliers / len(col_data)) * 0.05
                        accuracy_details[f"{col}_extreme_outliers"] = extreme_outliers

        # Ensure score doesn't go below 0
        accuracy_score = max(0, accuracy_score)

        # Determine status
        thresholds = self.quality_thresholds["accuracy"]
        if accuracy_score >= thresholds["good"]:
            status = "good"
        elif accuracy_score >= thresholds["warning"]:
            status = "warning"
        else:
            status = "critical"

        return DataQualityMetric(
            name="accuracy",
            value=accuracy_score,
            threshold=thresholds["warning"],
            status=status,
            description=f"Data accuracy: {accuracy_score:.1%} of values appear accurate",
            details=accuracy_details,
        )

    def detect_statistical_anomalies(
        self, data: pd.DataFrame
    ) -> List[AnomalyDetection]:
        """Detect statistical anomalies using multiple methods."""
        anomalies = []

        # Focus on numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) == 0:
            return anomalies

        # Prepare data for anomaly detection
        X = data[numeric_cols].fillna(data[numeric_cols].median())

        if len(X) < 10:  # Need sufficient data
            return anomalies

        try:
            # Method 1: Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X)
            anomaly_scores = iso_forest.decision_function(X)

            # Identify anomalies
            anomaly_indices = np.where(anomaly_labels == -1)[0]

            if len(anomaly_indices) > 0:
                # Determine affected features
                affected_features = []
                for idx in anomaly_indices:
                    # Find features that contribute most to the anomaly
                    z_scores = np.abs(stats.zscore(X.iloc[idx]))
                    high_z_features = numeric_cols[z_scores > 2].tolist()
                    affected_features.extend(high_z_features)

                anomalies.append(
                    AnomalyDetection(
                        type="statistical_outlier",
                        severity="medium",
                        description=f"Detected {len(anomaly_indices)} statistical outliers using Isolation Forest",
                        affected_features=list(set(affected_features)),
                        confidence=0.85,
                        timestamp=datetime.now(),
                        recommendations=[
                            "Review outlier records for data entry errors",
                            "Consider whether outliers represent legitimate edge cases",
                            "Validate data collection procedures",
                        ],
                    )
                )

            # Method 2: Distribution-based anomalies
            for col in numeric_cols:
                col_data = X[col].dropna()

                if len(col_data) < 20:
                    continue

                # Check for multimodal distributions
                try:
                    # Normality test
                    _, p_value = stats.normaltest(col_data)

                    if p_value < 0.001:  # Not normally distributed
                        # Check for multimodality using Hartigan's dip test
                        try:
                            from scipy.stats import kstest

                            # Simple multimodality check using kernel density
                            values = col_data.values
                            if len(values) > 50:
                                # Look for multiple peaks in distribution
                                hist, bins = np.histogram(values, bins=20)
                                peaks = 0
                                for i in range(1, len(hist) - 1):
                                    if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                                        peaks += 1

                                if peaks > 2:
                                    anomalies.append(
                                        AnomalyDetection(
                                            type="multimodal_distribution",
                                            severity="low",
                                            description=f"Column '{col}' shows multimodal distribution with {peaks} peaks",
                                            affected_features=[col],
                                            confidence=0.70,
                                            timestamp=datetime.now(),
                                            recommendations=[
                                                "Investigate potential subpopulations in data",
                                                "Consider data stratification for analysis",
                                                "Review data collection sources",
                                            ],
                                        )
                                    )
                        except Exception:
                            pass

                except Exception:
                    pass

        except Exception as e:
            apgi_logger.logger.warning(f"Error in statistical anomaly detection: {e}")

        return anomalies

    def detect_temporal_anomalies(
        self, data: pd.DataFrame, timestamp_col: Optional[str] = None
    ) -> List[AnomalyDetection]:
        """Detect temporal anomalies in time series data."""
        anomalies = []

        if timestamp_col is None:
            # Try to find timestamp column
            timestamp_col = None
            for col in data.columns:
                if any(
                    keyword in col.lower() for keyword in ["time", "date", "timestamp"]
                ):
                    timestamp_col = col
                    break

        if timestamp_col is None or timestamp_col not in data.columns:
            return anomalies

        try:
            # Convert timestamp column
            data.loc[:, timestamp_col] = pd.to_datetime(data[timestamp_col])
            data_sorted = data.sort_values(timestamp_col)

            # Check for gaps in time series
            time_diffs = data_sorted[timestamp_col].diff()
            median_diff = time_diffs.median()

            # Look for gaps larger than 5x median
            large_gaps = time_diffs > (median_diff * 5)
            gap_count = large_gaps.sum()

            if gap_count > 0:
                anomalies.append(
                    AnomalyDetection(
                        type="temporal_gap",
                        severity="medium",
                        description=f"Found {gap_count} gaps in time series data",
                        affected_features=[timestamp_col],
                        confidence=0.80,
                        timestamp=datetime.now(),
                        recommendations=[
                            "Investigate missing time periods",
                            "Check data collection continuity",
                            "Consider interpolation for small gaps",
                        ],
                    )
                )

            # Check for duplicate timestamps
            duplicate_timestamps = data_sorted[timestamp_col].duplicated().sum()
            if duplicate_timestamps > 0:
                anomalies.append(
                    AnomalyDetection(
                        type="duplicate_timestamps",
                        severity="medium",
                        description=f"Found {duplicate_timestamps} duplicate timestamps",
                        affected_features=[timestamp_col],
                        confidence=0.90,
                        timestamp=datetime.now(),
                        recommendations=[
                            "Review data collection process",
                            "Aggregate duplicate records",
                            "Check for sensor or logging issues",
                        ],
                    )
                )

        except Exception as e:
            apgi_logger.logger.warning(f"Error in temporal anomaly detection: {e}")

        return anomalies

    def calculate_overall_score(
        self, metrics: List[DataQualityMetric]
    ) -> Tuple[float, str]:
        """Calculate overall data quality score and grade."""
        if not metrics:
            return 0.0, "F"

        # Weight different metrics
        weights = {
            "completeness": 0.25,
            "uniqueness": 0.20,
            "validity": 0.25,
            "consistency": 0.15,
            "accuracy": 0.15,
        }

        weighted_score = 0.0
        total_weight = 0.0

        for metric in metrics:
            weight = weights.get(metric.name, 0.1)
            weighted_score += metric.value * weight
            total_weight += weight

        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 0.0

        # Assign grade
        if overall_score >= 0.90:
            grade = "A"
        elif overall_score >= 0.80:
            grade = "B"
        elif overall_score >= 0.70:
            grade = "C"
        elif overall_score >= 0.60:
            grade = "D"
        else:
            grade = "F"

        return overall_score, grade

    def generate_quality_report(
        self,
        data: pd.DataFrame,
        dataset_name: str = "dataset",
        timestamp_col: Optional[str] = None,
        schema: Optional[Dict[str, Dict]] = None,
    ) -> DataQualityReport:
        """Generate comprehensive data quality report."""

        apgi_logger.logger.info(f"Generating data quality report for {dataset_name}")

        # Assess all quality dimensions
        metrics = [
            self.assess_completeness(data),
            self.assess_uniqueness(data),
            self.assess_validity(data, schema),
            self.assess_consistency(data),
            self.assess_accuracy(data),
        ]

        # Detect anomalies
        statistical_anomalies = self.detect_statistical_anomalies(data)
        temporal_anomalies = self.detect_temporal_anomalies(data, timestamp_col)
        all_anomalies = statistical_anomalies + temporal_anomalies

        # Calculate overall score
        overall_score, grade = self.calculate_overall_score(metrics)

        # Generate summary
        summary = {
            "dataset_shape": data.shape,
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "total_anomalies": len(all_anomalies),
            "critical_metrics": [m.name for m in metrics if m.status == "critical"],
            "warning_metrics": [m.name for m in metrics if m.status == "warning"],
            "good_metrics": [m.name for m in metrics if m.status == "good"],
        }

        # Generate recommendations
        recommendations = []

        # Metric-based recommendations
        for metric in metrics:
            if metric.status == "critical":
                recommendations.append(
                    f"URGENT: Address {metric.name} issues - {metric.description}"
                )
            elif metric.status == "warning":
                recommendations.append(f"Review {metric.name} - {metric.description}")

        # Anomaly-based recommendations
        high_severity_anomalies = [a for a in all_anomalies if a.severity == "high"]
        if high_severity_anomalies:
            recommendations.append(
                f"URGENT: {len(high_severity_anomalies)} high-severity anomalies detected"
            )

        # General recommendations
        if overall_score < 0.70:
            recommendations.append(
                "Consider comprehensive data cleaning and validation procedures"
            )
        elif overall_score < 0.85:
            recommendations.append("Implement automated data quality monitoring")

        return DataQualityReport(
            dataset_name=dataset_name,
            overall_score=overall_score,
            grade=grade,
            metrics=metrics,
            anomalies=all_anomalies,
            summary=summary,
            recommendations=recommendations,
            timestamp=datetime.now(),
        )

    def create_quality_visualization(
        self, report: DataQualityReport, save_path: Optional[Path] = None
    ) -> Path:
        """Create data quality visualization dashboard."""

        if save_path is None:
            save_path = Path(
                f"data_quality_{report.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Data Quality Report - {report.dataset_name}",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Overall score gauge
        ax1 = axes[0, 0]
        colors = ["red", "orange", "yellow", "lightgreen", "green"]
        score_ranges = [0, 0.6, 0.7, 0.8, 0.9, 1.0]

        for i in range(len(score_ranges) - 1):
            ax1.barh(
                0,
                score_ranges[i + 1] - score_ranges[i],
                left=score_ranges[i],
                height=0.5,
                color=colors[i],
                alpha=0.7,
            )

        ax1.barh(0, report.overall_score, height=0.5, color="blue", alpha=0.8)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("Quality Score")
        ax1.set_title(f"Overall Score: {report.overall_score:.2f} ({report.grade})")
        ax1.set_yticks([])
        ax1.axvline(
            x=report.overall_score, color="darkblue", linestyle="--", linewidth=2
        )

        # 2. Metrics breakdown
        ax2 = axes[0, 1]
        metric_names = [m.name.title() for m in report.metrics]
        metric_values = [m.value for m in report.metrics]
        metric_colors = [
            (
                "green"
                if m.status == "good"
                else "orange" if m.status == "warning" else "red"
            )
            for m in report.metrics
        ]

        bars = ax2.bar(metric_names, metric_values, color=metric_colors, alpha=0.7)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Score")
        ax2.set_title("Quality Metrics Breakdown")
        ax2.tick_params(axis="x", rotation=45)

        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # 3. Anomaly severity distribution
        ax3 = axes[0, 2]
        if report.anomalies:
            severity_counts = {"low": 0, "medium": 0, "high": 0}
            for anomaly in report.anomalies:
                severity_counts[anomaly.severity] += 1

            colors_severity = ["green", "orange", "red"]
            ax3.pie(
                severity_counts.values(),
                labels=list(severity_counts.keys()),
                colors=colors_severity,
                autopct="%1.0f%%",
                startangle=90,
            )
            ax3.set_title("Anomaly Severity Distribution")
        else:
            ax3.text(
                0.5,
                0.5,
                "No Anomalies Detected",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=14,
            )
            ax3.set_title("Anomaly Severity Distribution")

        # 4. Data types distribution
        ax4 = axes[1, 0]
        dtype_counts = report.summary["data_types"]
        if dtype_counts:
            # Convert pandas dtypes to strings for counting
            str_dtypes = [str(dtype) for dtype in dtype_counts.values()]
            unique_dtypes = list(set(str_dtypes))
            dtype_counts_list = [str_dtypes.count(dtype) for dtype in unique_dtypes]

            ax4.pie(
                dtype_counts_list,
                labels=unique_dtypes,
                autopct="%1.0f%%",
                startangle=90,
            )
            ax4.set_title("Data Types Distribution")
        else:
            ax4.text(
                0.5,
                0.5,
                "No Data Type Information",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=14,
            )
            ax4.set_title("Data Types Distribution")

        # 5. Missing values heatmap (if applicable)
        ax5 = axes[1, 1]
        completeness_metric = next(
            (m for m in report.metrics if m.name == "completeness"), None
        )
        if completeness_metric and completeness_metric.details.get("missing_by_column"):
            missing_data = completeness_metric.details["missing_by_column"]
            if missing_data:
                columns = list(missing_data.keys())[:10]  # Limit to 10 columns
                missing_percentages = [missing_data[col] * 100 for col in columns]

                bars = ax5.barh(columns, missing_percentages, color="red", alpha=0.7)
                ax5.set_xlabel("Missing Percentage (%)")
                ax5.set_title("Missing Values by Column")
                ax5.tick_params(axis="y", rotation=45)

                # Add value labels
                for bar, value in zip(bars, missing_percentages):
                    width = bar.get_width()
                    ax5.text(
                        width + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.1f}%",
                        ha="left",
                        va="center",
                    )
            else:
                ax5.text(
                    0.5,
                    0.5,
                    "No Missing Values",
                    ha="center",
                    va="center",
                    transform=ax5.transAxes,
                    fontsize=14,
                )
                ax5.set_title("Missing Values by Column")
        else:
            ax5.text(
                0.5,
                0.5,
                "No Missing Data",
                ha="center",
                va="center",
                transform=ax5.transAxes,
                fontsize=14,
            )
            ax5.set_title("Missing Values by Column")

        # 6. Recommendations summary
        ax6 = axes[1, 2]
        ax6.axis("off")

        # Count recommendations by priority
        urgent_count = len([r for r in report.recommendations if "URGENT" in r])
        review_count = len([r for r in report.recommendations if "Review" in r])
        other_count = len(report.recommendations) - urgent_count - review_count

        recommendation_text = f"""Recommendations Summary:

Urgent Actions: {urgent_count}
Review Needed: {review_count}
Other: {other_count}

Total: {len(report.recommendations)} recommendations"""

        ax6.text(
            0.1,
            0.9,
            recommendation_text,
            transform=ax6.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax6.set_title("Recommendations Summary")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        apgi_logger.logger.info(f"Data quality visualization saved to {save_path}")
        return save_path


# Global data quality assessment instance
data_quality_assessor = DataQualityAssessment()


# Convenience functions
def assess_data_quality(
    data: pd.DataFrame, dataset_name: str = "dataset"
) -> DataQualityReport:
    """Quick data quality assessment."""
    return data_quality_assessor.generate_quality_report(data, dataset_name)


def detect_anomalies(
    data: pd.DataFrame, timestamp_col: Optional[str] = None
) -> List[AnomalyDetection]:
    """Quick anomaly detection."""
    statistical = data_quality_assessor.detect_statistical_anomalies(data)
    temporal = data_quality_assessor.detect_temporal_anomalies(data, timestamp_col)
    return statistical + temporal
