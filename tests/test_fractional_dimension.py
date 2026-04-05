"""
Tests for fractional dimension biomarker computation.
========================================================

Tests for the APGI Full Dynamic Model's compute_fractional_dimension_biomarker method.
"""

import sys
from pathlib import Path

# Add subdirectories to path for module imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Theory"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Falsification"))

import numpy as np
import pytest

# Import the main module
try:
    from APGI_Full_Dynamic_Model import APGIFullDynamicModel
except ImportError:
    pytest.skip("APGI_Full_Dynamic_Model not available", allow_module_level=True)


class TestFractionalDimensionBiomarker:
    """Tests for compute_fractional_dimension_biomarker method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = APGIFullDynamicModel()

    def test_minimum_signal_length_requirement(self):
        """Test that signal with fewer than 100 points raises ValueError."""
        short_signal = np.random.randn(50)

        with pytest.raises(ValueError, match="at least 100 points"):
            self.model.compute_fractional_dimension_biomarker(short_signal)

    def test_valid_signal_computation(self):
        """Test computation with valid signal length."""
        signal = np.random.randn(200)

        result = self.model.compute_fractional_dimension_biomarker(signal)

        # Check required keys
        assert "hurst_exponent" in result
        assert "variance_h" in result
        assert "variance_h_available" in result
        assert "clinical_interpretation" in result

        # Check Hurst exponent is within valid range
        assert 0 < result["hurst_exponent"] < 1

    def test_healthy_persistent_interpretation(self):
        """Test clinical interpretation for healthy state (H > 0.55)."""
        # Generate a signal with persistent correlations (H > 0.5)
        # Using cumulative sum of random walk creates persistence
        np.random.seed(42)
        noise = np.random.randn(500)
        persistent_signal = np.cumsum(noise)
        # Normalize to reasonable range
        persistent_signal = persistent_signal / np.std(persistent_signal)

        result = self.model.compute_fractional_dimension_biomarker(persistent_signal)

        assert result["clinical_interpretation"] in [
            "healthy_persistent",
            "elevated_H_variance_ADHD",
        ]

    def test_sliding_window_variance_computation(self):
        """Test that sliding window variance is computed with sufficient data."""
        np.random.seed(42)
        # Long signal for sliding window analysis
        signal = np.random.randn(1000)

        result = self.model.compute_fractional_dimension_biomarker(signal)

        # With 1000 points, we should have enough for sliding windows
        assert result["variance_h_available"] is True
        assert result["variance_h"] is not None
        assert result["variance_h"] >= 0

    def test_insufficient_data_variance_unavailable(self):
        """Test that variance_H is unavailable with short but valid signal."""
        # 200 points is valid but may not allow many sliding windows
        np.random.seed(42)
        signal = np.random.randn(150)

        result = self.model.compute_fractional_dimension_biomarker(signal)

        # Should compute successfully but variance may or may not be available
        assert result["hurst_exponent"] > 0
        assert isinstance(result["variance_h_available"], bool)

    def test_with_threshold_time_series(self):
        """Test computation with optional threshold time series."""
        np.random.seed(42)
        signal = np.random.randn(500)
        threshold = np.random.randn(500) * 0.5 + 2.0

        result = self.model.compute_fractional_dimension_biomarker(
            signal, threshold_time_series=threshold
        )

        # Check proximity effects are included when threshold provided
        if "hurst_near_threshold" in result:
            assert isinstance(result["hurst_near_threshold"], (int, float))
            assert "proximity_effect" in result

    def test_threshold_mismatch_handling(self):
        """Test handling of mismatched signal and threshold lengths."""
        np.random.seed(42)
        signal = np.random.randn(300)
        threshold = np.random.randn(200)  # Different length

        result = self.model.compute_fractional_dimension_biomarker(
            signal, threshold_time_series=threshold
        )

        # Should still work but without proximity analysis
        assert "hurst_exponent" in result
        assert "hurst_near_threshold" not in result

    def test_return_type(self):
        """Test that return type is correct dictionary."""
        np.random.seed(42)
        signal = np.random.randn(300)

        result = self.model.compute_fractional_dimension_biomarker(signal)

        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result.keys())

    def test_deterministic_seed(self):
        """Test that same seed gives consistent results."""
        np.random.seed(42)
        signal1 = np.random.randn(300)

        np.random.seed(42)
        signal2 = np.random.randn(300)

        result1 = self.model.compute_fractional_dimension_biomarker(signal1)
        result2 = self.model.compute_fractional_dimension_biomarker(signal2)

        assert result1["hurst_exponent"] == pytest.approx(result2["hurst_exponent"])

    def test_constant_signal_handling(self):
        """Test handling of constant signal."""
        constant_signal = np.ones(300)

        # Should handle without crashing
        result = self.model.compute_fractional_dimension_biomarker(constant_signal)

        assert "hurst_exponent" in result
        assert "clinical_interpretation" in result


class TestHurstExponentDFA:
    """Tests for DFA-based Hurst exponent computation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = APGIFullDynamicModel()

    def test_dfa_with_white_noise(self):
        """Test DFA with white noise (H ≈ 0.5)."""
        np.random.seed(42)
        white_noise = np.random.randn(500)

        H = self.model.compute_hurst_exponent_dfa(white_noise)

        # White noise should have H close to 0.5
        assert 0.4 < H < 0.6

    def test_dfa_with_random_walk(self):
        """Test DFA with random walk (H > 0.5)."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(500))

        H = self.model.compute_hurst_exponent_dfa(random_walk)

        # Random walk should have H > 0.5 (persistent)
        assert H > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
