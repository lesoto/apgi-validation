"""
Comprehensive tests for ordinal_logistic_regression utility module.
================================================================

Tests all functions and classes in ordinal_logistic_regression.py including:
- OrdinalLogisticRegression class
- analyze_clinical_gradient_ordinal function
- compare_ordinal_vs_anova function
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.ordinal_logistic_regression import (
        OrdinalLogisticRegression,
        analyze_clinical_gradient_ordinal,
        compare_ordinal_vs_anova,
    )
except ImportError as e:
    pytest.skip(
        f"Cannot import ordinal_logistic_regression: {e}", allow_module_level=True
    )


class TestOrdinalLogisticRegression:
    """Test OrdinalLogisticRegression class."""

    def test_initialization_default(self):
        """Test default initialization."""
        model = OrdinalLogisticRegression()

        assert model.n_categories == 4, "Default should be 4 categories"
        assert model.n_thresholds == 3, "Should have n_categories-1 thresholds"
        assert model.thresholds is None, "Thresholds should be None initially"
        assert model.coefficients is None, "Coefficients should be None initially"
        assert not model.fitted, "Model should not be fitted initially"

    def test_initialization_custom_categories(self):
        """Test initialization with custom number of categories."""
        model = OrdinalLogisticRegression(n_categories=5)

        assert model.n_categories == 5, "Should use custom number of categories"
        assert model.n_thresholds == 4, "Should have n_categories-1 thresholds"

    def test_initialization_invalid_categories(self):
        """Test initialization with invalid number of categories."""
        with pytest.raises(ValueError):
            OrdinalLogisticRegression(n_categories=1)

        with pytest.raises(ValueError):
            OrdinalLogisticRegression(n_categories=0)

    @pytest.fixture
    def sample_data(self):
        """Create sample ordinal data for testing."""
        np.random.seed(42)
        n_samples = 200
        n_features = 3

        # Generate features
        X = np.random.randn(n_samples, n_features)

        # Generate ordinal outcomes with some relationship to features
        # Create latent variable
        latent = X @ [0.5, -0.3, 0.2] + np.random.randn(n_samples) * 0.5

        # Convert to ordinal categories
        thresholds = [-1.0, 0.0, 1.0]  # 3 thresholds for 4 categories
        y = np.zeros(n_samples, dtype=int)
        for i, threshold in enumerate(thresholds):
            y[latent > threshold] = i + 1

        return X, y

    def test_negative_log_likelihood(self, sample_data):
        """Test negative log likelihood computation."""
        X, y = sample_data
        model = OrdinalLogisticRegression(n_categories=4)

        # Create some test parameters
        thresholds = np.array([-1.0, 0.0, 1.0])
        coefficients = np.array([0.1, -0.1, 0.1])
        params = np.concatenate([thresholds, coefficients])

        nll = model._negative_log_likelihood(params, X, y)

        assert isinstance(nll, float), "Should return float"
        assert nll > 0, "Negative log likelihood should be positive"
        assert np.isfinite(nll), "Should be finite"

    def test_negative_log_likelihood_unordered_thresholds(self, sample_data):
        """Test negative log likelihood with unordered thresholds."""
        X, y = sample_data
        model = OrdinalLogisticRegression(n_categories=4)

        # Create unordered thresholds
        thresholds = np.array([1.0, -1.0, 0.0])  # Not ordered
        coefficients = np.array([0.1, -0.1, 0.1])
        params = np.concatenate([thresholds, coefficients])

        nll = model._negative_log_likelihood(params, X, y)

        # Should return large penalty for unordered thresholds
        assert nll == 1e10, "Should return large penalty for unordered thresholds"

    def test_fit_basic(self, sample_data):
        """Test basic model fitting."""
        X, y = sample_data
        model = OrdinalLogisticRegression(n_categories=4)

        result = model.fit(X, y)

        assert isinstance(result, dict), "Should return dictionary"
        required_keys = [
            "success",
            "message",
            "n_iterations",
            "thresholds",
            "coefficients",
            "accuracy",
            "null_deviance",
            "residual_deviance",
            "deviance_reduction",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        assert isinstance(result["success"], bool), "Success should be boolean"
        assert isinstance(result["n_iterations"], int), "Iterations should be int"
        assert (
            len(result["thresholds"]) == model.n_thresholds
        ), "Should have correct number of thresholds"
        assert (
            len(result["coefficients"]) == X.shape[1]
        ), "Should have correct number of coefficients"
        assert 0 <= result["accuracy"] <= 1, "Accuracy should be between 0 and 1"
        assert result["null_deviance"] > 0, "Null deviance should be positive"
        assert (
            result["residual_deviance"] >= 0
        ), "Residual deviance should be non-negative"
        assert isinstance(
            result["deviance_reduction"], float
        ), "Deviance reduction should be float"

    def test_fit_with_initial_params(self, sample_data):
        """Test fitting with initial parameters."""
        X, y = sample_data
        model = OrdinalLogisticRegression(n_categories=4)

        # Provide initial parameters
        init_params = np.array([-0.5, 0.0, 0.5, 0.1, -0.1, 0.1])

        result = model.fit(X, y, init_params=init_params)

        assert isinstance(result, dict), "Should return dictionary"
        assert result["success"], "Should succeed with good initial params"

    def test_fit_convergence_failure(self):
        """Test fitting with convergence failure."""
        # Create challenging data
        np.random.seed(42)
        n_samples = 50
        n_features = 10  # Many features, few samples
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 4, n_samples)

        model = OrdinalLogisticRegression(n_categories=4)

        result = model.fit(X, y)

        # Should still return result even if not converged
        assert isinstance(result, dict), "Should return dictionary"
        assert "success" in result, "Should indicate success status"

    def test_predict_proba_before_fit(self):
        """Test prediction before fitting (should raise error)."""
        model = OrdinalLogisticRegression()
        X = np.random.randn(10, 3)

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba(X)

    def test_predict_before_fit(self):
        """Test class prediction before fitting (should raise error)."""
        model = OrdinalLogisticRegression()
        X = np.random.randn(10, 3)

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(X)

    def test_predict_proba_after_fit(self, sample_data):
        """Test probability prediction after fitting."""
        X, y = sample_data
        model = OrdinalLogisticRegression(n_categories=4)
        model.fit(X, y)

        probs = model.predict_proba(X)

        assert isinstance(probs, np.ndarray), "Should return numpy array"
        assert probs.shape == (
            X.shape[0],
            model.n_categories,
        ), "Should have correct shape"
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities should sum to 1"
        assert np.all(probs >= 0), "Probabilities should be non-negative"
        assert np.all(probs <= 1), "Probabilities should not exceed 1"

    def test_predict_after_fit(self, sample_data):
        """Test class prediction after fitting."""
        X, y = sample_data
        model = OrdinalLogisticRegression(n_categories=4)
        model.fit(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray), "Should return numpy array"
        assert predictions.shape == (X.shape[0],), "Should have correct shape"
        assert np.all(predictions >= 0), "Predictions should be non-negative"
        assert np.all(
            predictions < model.n_categories
        ), "Predictions should be valid categories"

    def test_compute_null_deviance(self, sample_data):
        """Test null deviance computation."""
        X, y = sample_data
        model = OrdinalLogisticRegression(n_categories=4)

        null_dev = model._compute_null_deviance(y)

        assert isinstance(null_dev, float), "Should return float"
        assert null_dev > 0, "Null deviance should be positive"
        assert np.isfinite(null_dev), "Should be finite"

    def test_get_coefficient_significance_before_fit(self):
        """Test significance testing before fitting (should raise error)."""
        model = OrdinalLogisticRegression()
        X = np.random.randn(10, 3)
        y = np.random.randint(0, 4, 10)

        with pytest.raises(ValueError, match="Model must be fitted"):
            model.get_coefficient_significance(X, y)

    def test_get_coefficient_significance_after_fit(self, sample_data):
        """Test coefficient significance testing after fitting."""
        X, y = sample_data
        model = OrdinalLogisticRegression(n_categories=4)
        model.fit(X, y)

        significance = model.get_coefficient_significance(X, y)

        assert isinstance(significance, dict), "Should return dictionary"
        required_keys = ["p_values", "std_errors", "significant"]
        for key in required_keys:
            assert key in significance, f"Missing key: {key}"

        assert (
            len(significance["p_values"]) == X.shape[1]
        ), "Should have p-value for each coefficient"
        assert (
            len(significance["std_errors"]) == X.shape[1]
        ), "Should have std error for each coefficient"
        assert (
            len(significance["significant"]) == X.shape[1]
        ), "Should have significance for each coefficient"
        assert np.all(significance["p_values"] >= 0), "P-values should be non-negative"
        assert np.all(significance["p_values"] <= 1), "P-values should not exceed 1"
        assert np.all(
            significance["std_errors"] >= 0
        ), "Std errors should be non-negative"

    def test_single_category_data(self):
        """Test with data from single category."""
        X = np.random.randn(100, 3)
        y = np.zeros(100, dtype=int)  # All in category 0

        model = OrdinalLogisticRegression(n_categories=4)

        # Should handle single category data
        result = model.fit(X, y)
        assert isinstance(result, dict), "Should handle single category data"

    def test_edge_case_extreme_values(self):
        """Test with extreme feature values."""
        X = np.array([[1e6, -1e6, 1e-6], [-1e6, 1e6, -1e-6]])
        y = np.array([0, 3])

        model = OrdinalLogisticRegression(n_categories=4)

        # Should handle extreme values
        result = model.fit(X, y)
        assert isinstance(result, dict), "Should handle extreme values"


class TestAnalyzeClinicalGradientOrdinal:
    """Test analyze_clinical_gradient_ordinal function."""

    @pytest.fixture
    def sample_patient_data(self):
        """Create sample patient data for testing."""
        np.random.seed(42)
        n_patients = 200

        # Create consciousness states
        states = ["VS", "MCS", "EMCS", "Healthy"]
        consciousness_state = np.random.choice(states, n_patients)

        # Create features that vary by consciousness state
        p3b_reduction = []
        ignition_reduction = []

        for state in consciousness_state:
            if state == "VS":
                p3b_reduction.append(np.random.normal(0.6, 0.1))
                ignition_reduction.append(np.random.normal(0.5, 0.1))
            elif state == "MCS":
                p3b_reduction.append(np.random.normal(0.7, 0.1))
                ignition_reduction.append(np.random.normal(0.6, 0.1))
            elif state == "EMCS":
                p3b_reduction.append(np.random.normal(0.8, 0.1))
                ignition_reduction.append(np.random.normal(0.7, 0.1))
            else:  # Healthy
                p3b_reduction.append(np.random.normal(0.9, 0.1))
                ignition_reduction.append(np.random.normal(0.85, 0.1))

        df = pd.DataFrame(
            {
                "consciousness_state": consciousness_state,
                "p3b_reduction": p3b_reduction,
                "ignition_reduction": ignition_reduction,
            }
        )

        return df

    def test_analyze_clinical_gradient_basic(self, sample_patient_data):
        """Test basic clinical gradient analysis."""
        df = sample_patient_data
        feature_columns = ["p3b_reduction", "ignition_reduction"]

        result = analyze_clinical_gradient_ordinal(df, feature_columns)

        assert isinstance(result, dict), "Should return dictionary"
        required_keys = [
            "model_fitted",
            "thresholds",
            "coefficients",
            "feature_names",
            "accuracy",
            "per_class_accuracy",
            "category_names",
            "confusion_matrix",
            "null_deviance",
            "residual_deviance",
            "deviance_reduction_pct",
            "coefficient_p_values",
            "coefficient_std_errors",
            "significant_coefficients",
            "proportional_odds_assumption",
            "predicted_probabilities",
            "predicted_classes",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        assert isinstance(
            result["model_fitted"], bool
        ), "Model fitted should be boolean"
        assert (
            len(result["thresholds"]) == 3
        ), "Should have 3 thresholds for 4 categories"
        assert len(result["coefficients"]) == len(
            feature_columns
        ), "Should have coefficient for each feature"
        assert (
            result["feature_names"] == feature_columns
        ), "Should preserve feature names"
        assert 0 <= result["accuracy"] <= 1, "Accuracy should be between 0 and 1"
        assert (
            len(result["per_class_accuracy"]) == 4
        ), "Should have accuracy for each class"
        assert result["category_names"] == [
            "VS",
            "MCS",
            "EMCS",
            "Healthy",
        ], "Should have correct category names"
        assert result["confusion_matrix"].shape == (
            4,
            4,
        ), "Confusion matrix should be 4x4"
        assert result["null_deviance"] > 0, "Null deviance should be positive"
        assert (
            result["residual_deviance"] >= 0
        ), "Residual deviance should be non-negative"
        assert isinstance(
            result["deviance_reduction_pct"], float
        ), "Deviance reduction should be float"

    def test_analyze_clinical_gradient_custom_category_order(self):
        """Test with custom category order."""
        df = pd.DataFrame(
            {
                "consciousness_state": ["Healthy", "EMCS", "MCS", "VS"] * 50,
                "feature1": np.random.randn(200),
                "feature2": np.random.randn(200),
            }
        )

        custom_order = ["Healthy", "EMCS", "MCS", "VS"]
        feature_columns = ["feature1", "feature2"]

        result = analyze_clinical_gradient_ordinal(
            df, feature_columns, category_order=custom_order
        )

        assert (
            result["category_names"] == custom_order
        ), "Should use custom category order"

    def test_analyze_clinical_gradient_missing_states(self):
        """Test with missing consciousness states."""
        df = pd.DataFrame(
            {
                "consciousness_state": ["VS", "MCS"] * 100,  # Only VS and MCS
                "feature1": np.random.randn(200),
                "feature2": np.random.randn(200),
            }
        )

        feature_columns = ["feature1", "feature2"]

        result = analyze_clinical_gradient_ordinal(df, feature_columns)

        assert isinstance(result, dict), "Should handle missing states"
        assert result["category_names"] == [
            "VS",
            "MCS",
            "EMCS",
            "Healthy",
        ], "Should still have all categories"

    def test_analyze_clinical_gradient_single_feature(self, sample_patient_data):
        """Test with single feature."""
        df = sample_patient_data
        feature_columns = ["p3b_reduction"]

        result = analyze_clinical_gradient_ordinal(df, feature_columns)

        assert isinstance(result, dict), "Should handle single feature"
        assert len(result["coefficients"]) == 1, "Should have single coefficient"

    def test_analyze_clinical_gradient_no_features(self):
        """Test with no features (should raise error)."""
        df = pd.DataFrame(
            {
                "consciousness_state": ["VS", "MCS", "EMCS", "Healthy"],
                "other_column": [1, 2, 3, 4],
            }
        )

        with pytest.raises(ValueError):
            analyze_clinical_gradient_ordinal(df, [])

    def test_analyze_clinical_gradient_perfect_separation(self):
        """Test with perfectly separable data."""
        df = pd.DataFrame(
            {
                "consciousness_state": ["VS"] * 50 + ["Healthy"] * 50,
                "feature1": np.concatenate(
                    [np.random.randn(50) - 2, np.random.randn(50) + 2]
                ),
            }
        )

        result = analyze_clinical_gradient_ordinal(df, ["feature1"])

        assert isinstance(result, dict), "Should handle perfect separation"
        assert (
            result["accuracy"] >= 0.9
        ), "Should achieve high accuracy with perfect separation"


class TestCompareOrdinalVsAnova:
    """Test compare_ordinal_vs_anova function."""

    @pytest.fixture
    def sample_comparison_data(self):
        """Create sample data for comparison."""
        np.random.seed(42)
        n_patients = 200

        states = ["VS", "MCS", "EMCS", "Healthy"]
        consciousness_state = np.random.choice(states, n_patients)

        # Create features with clear group differences
        p3b_reduction = []
        for state in consciousness_state:
            if state == "VS":
                p3b_reduction.append(np.random.normal(0.6, 0.1))
            elif state == "MCS":
                p3b_reduction.append(np.random.normal(0.7, 0.1))
            elif state == "EMCS":
                p3b_reduction.append(np.random.normal(0.8, 0.1))
            else:  # Healthy
                p3b_reduction.append(np.random.normal(0.9, 0.1))

        df = pd.DataFrame(
            {
                "consciousness_state": consciousness_state,
                "p3b_reduction": p3b_reduction,
            }
        )

        return df

    def test_compare_ordinal_vs_anova_basic(self, sample_comparison_data):
        """Test basic comparison."""
        df = sample_comparison_data
        feature_columns = ["p3b_reduction"]

        result = compare_ordinal_vs_anova(df, feature_columns)

        assert isinstance(result, dict), "Should return dictionary"
        required_keys = ["ordinal_logistic", "anova_cohens_d", "recommendation"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # Check ordinal logistic results
        ordinal_result = result["ordinal_logistic"]
        assert isinstance(ordinal_result, dict), "Ordinal result should be dictionary"
        assert "accuracy" in ordinal_result, "Should have accuracy"
        assert (
            "deviance_reduction_pct" in ordinal_result
        ), "Should have deviance reduction"
        assert (
            "n_significant" in ordinal_result
        ), "Should have number of significant coefficients"

        # Check ANOVA results
        anova_result = result["anova_cohens_d"]
        assert isinstance(anova_result, dict), "ANOVA result should be dictionary"
        assert "f_statistic" in anova_result, "Should have F statistic"
        assert "p_value" in anova_result, "Should have p-value"
        assert "cohens_d_values" in anova_result, "Should have Cohen's d values"

        # Check recommendation
        assert isinstance(
            result["recommendation"], str
        ), "Recommendation should be string"
        assert len(result["recommendation"]) > 0, "Recommendation should not be empty"

    def test_compare_ordinal_vs_anova_multiple_features(self, sample_comparison_data):
        """Test comparison with multiple features."""
        df = sample_comparison_data
        df["ignition_reduction"] = np.random.randn(len(df))
        feature_columns = ["p3b_reduction", "ignition_reduction"]

        result = compare_ordinal_vs_anova(df, feature_columns)

        assert isinstance(result, dict), "Should handle multiple features"
        assert (
            result["ordinal_logistic"]["n_significant"] >= 0
        ), "Should handle multiple features"

    def test_compare_ordinal_vs_anova_missing_states(self):
        """Test comparison with missing states."""
        df = pd.DataFrame(
            {
                "consciousness_state": ["VS", "MCS"] * 100,  # Only two states
                "feature1": np.random.randn(200),
            }
        )

        result = compare_ordinal_vs_anova(df, ["feature1"])

        assert isinstance(result, dict), "Should handle missing states"
        assert isinstance(
            result["anova_cohens_d"]["f_statistic"], float
        ), "Should compute F statistic"

    def test_compare_ordinal_vs_anova_single_state(self):
        """Test comparison with single state (degenerate case)."""
        df = pd.DataFrame(
            {
                "consciousness_state": ["VS"] * 100,
                "feature1": np.random.randn(100),
            }
        )

        result = compare_ordinal_vs_anova(df, ["feature1"])

        assert isinstance(result, dict), "Should handle single state"
        assert (
            result["anova_cohens_d"]["f_statistic"] == 0
        ), "F statistic should be 0 for single group"

    def test_compare_ordinal_vs_anova_no_variance(self):
        """Test comparison with no variance in data."""
        df = pd.DataFrame(
            {
                "consciousness_state": ["VS", "MCS", "EMCS", "Healthy"] * 25,
                "feature1": [1.0] * 100,  # No variance
            }
        )

        result = compare_ordinal_vs_anova(df, ["feature1"])

        assert isinstance(result, dict), "Should handle no variance"
        assert (
            result["anova_cohens_d"]["f_statistic"] == 0
        ), "F statistic should be 0 with no variance"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_n_categories(self):
        """Test with invalid number of categories."""
        with pytest.raises(ValueError):
            OrdinalLogisticRegression(n_categories=0)

        with pytest.raises(ValueError):
            OrdinalLogisticRegression(n_categories=1)

    def test_mismatched_dimensions(self):
        """Test with mismatched X and y dimensions."""
        model = OrdinalLogisticRegression()
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 4, 50)  # Different length

        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_invalid_y_values(self):
        """Test with invalid y values."""
        model = OrdinalLogisticRegression(n_categories=4)
        X = np.random.randn(100, 3)
        y = np.random.randint(-2, 6, 100)  # Values outside expected range

        # Should handle invalid y values gracefully
        result = model.fit(X, y)
        assert isinstance(result, dict), "Should handle invalid y values"

    def test_empty_data(self):
        """Test with empty data."""
        model = OrdinalLogisticRegression()
        X = np.array([]).reshape(0, 3)
        y = np.array([])

        # Should handle empty data gracefully
        result = model.fit(X, y)
        assert isinstance(result, dict), "Should handle empty data"

    def test_single_sample(self):
        """Test with single sample."""
        model = OrdinalLogisticRegression()
        X = np.random.randn(1, 3)
        y = np.array([1])

        # Should handle single sample
        result = model.fit(X, y)
        assert isinstance(result, dict), "Should handle single sample"

    def test_nan_data(self):
        """Test with NaN values in data."""
        model = OrdinalLogisticRegression()
        X = np.random.randn(100, 3)
        X[50, :] = np.nan  # Insert NaN values
        y = np.random.randint(0, 4, 100)

        # Should handle NaN values
        result = model.fit(X, y)
        assert isinstance(result, dict), "Should handle NaN values"

    def test_infinite_data(self):
        """Test with infinite values in data."""
        model = OrdinalLogisticRegression()
        X = np.random.randn(100, 3)
        X[50, 0] = np.inf  # Insert infinite value
        y = np.random.randint(0, 4, 100)

        # Should handle infinite values
        result = model.fit(X, y)
        assert isinstance(result, dict), "Should handle infinite values"

    def test_very_large_data(self):
        """Test with very large data values."""
        model = OrdinalLogisticRegression()
        X = np.random.randn(100, 3) * 1e10  # Very large values
        y = np.random.randint(0, 4, 100)

        # Should handle large values
        result = model.fit(X, y)
        assert isinstance(result, dict), "Should handle large values"

    def test_very_small_data(self):
        """Test with very small data values."""
        model = OrdinalLogisticRegression()
        X = np.random.randn(100, 3) * 1e-10  # Very small values
        y = np.random.randint(0, 4, 100)

        # Should handle small values
        result = model.fit(X, y)
        assert isinstance(result, dict), "Should handle small values"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
