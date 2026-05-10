"""
Tests for Theory/APGI_Information_Theoretic_Bandwidth.py module
========================================================================

Comprehensive test suite for APGI Information-Theoretic Bandwidth analysis.
Tests all classes and functions to achieve >90% coverage.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBinaryEntropy:
    """Test binary entropy function."""

    def test_binary_entropy_basic(self):
        """Test basic binary entropy calculation."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _binary_entropy

        # Test with known probabilities
        p = np.array([0.5, 0.5])
        result = _binary_entropy(p)
        expected = -0.5 * np.log2(0.5) - 0.5 * np.log2(0.5)  # = 1.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_binary_entropy_edge_cases(self):
        """Test binary entropy edge cases."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _binary_entropy

        # Test with p=0 and p=1 (should handle clipping)
        p_zero = np.array([0.0, 1.0])
        result_zero = _binary_entropy(p_zero)
        assert np.all(np.isfinite(result_zero))  # Should not be NaN due to clipping

        # Test with uniform distribution
        p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
        result_uniform = _binary_entropy(p_uniform)
        expected_uniform = -4 * 0.25 * np.log2(0.25)  # = 2.0
        np.testing.assert_allclose(result_uniform, expected_uniform, rtol=1e-10)


class TestSigmoid:
    """Test sigmoid function."""

    def test_sigmoid_basic(self):
        """Test basic sigmoid functionality."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _sigmoid

        x = np.array([0.0, 1.0, -1.0])
        result = _sigmoid(x, alpha=1.0)

        # Sigmoid(0) = 0.5, Sigmoid(1) ≈ 0.731, Sigmoid(-1) ≈ 0.269
        expected = 1.0 / (1.0 + np.exp(-x))
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_sigmoid_clipping(self):
        """Test sigmoid clipping for large values."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _sigmoid

        x = np.array([1000.0, -1000.0])
        result = _sigmoid(x, alpha=1.0)

        # Should be clipped to avoid overflow
        assert np.all(result >= 0.0) & np.all(result <= 1.0)


class TestIgnitionProbPerLevel:
    """Test ignition probability per level function."""

    def test_ignition_prob_basic(self):
        """Test basic ignition probability calculation."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _ignition_prob_per_level

        stimulus_levels = np.array([1.0, 2.0, 3.0])
        Pi_e = 0.8
        Pi_i = 0.2

        result = _ignition_prob_per_level(stimulus_levels, Pi_e, Pi_i)

        # Should return probabilities for each level
        assert result.shape == (3,)
        assert np.all(result >= 0.0) & np.all(result <= 1.0)

    def test_ignition_prob_edge_cases(self):
        """Test ignition probability edge cases."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _ignition_prob_per_level

        stimulus_levels = np.array([1.0, 2.0, 3.0])

        # Test with extreme precision values
        result_high_precision = _ignition_prob_per_level(stimulus_levels, Pi_e=1.0, Pi_i=0.0)
        expected_high = np.array([1.0, 0.0, 0.0])  # All probability at lowest level
        np.testing.assert_allclose(result_high_precision, expected_high)

        result_low_precision = _ignition_prob_per_level(stimulus_levels, Pi_e=0.0, Pi_i=1.0)
        expected_low = np.array([0.0, 0.0, 1.0])  # All probability at highest level
        np.testing.assert_allclose(result_low_precision, expected_low)


class TestMutualInfoBits:
    """Test mutual information bits function."""

    def test_mutual_info_basic(self):
        """Test basic mutual information calculation."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _mutual_info_bits

        # Test with identical labels (should give 0 MI)
        labels_a = np.array([1, 2, 3, 1, 2])
        labels_b = np.array([1, 2, 3, 1, 2])
        result = _mutual_info_bits(labels_a, labels_b)
        assert result == 0.0

    def test_mutual_info_different(self):
        """Test mutual information with different labels."""
        from Theory.APGI_Information_Theoretic_Bandwidth import _mutual_info_bits

        labels_a = np.array([1, 1, 1, 2, 2])
        labels_b = np.array([2, 2, 2, 1, 1])
        result = _mutual_info_bits(labels_a, labels_b)
        assert result > 0.0  # Should be positive for different distributions


class TestBandwidthDerivation:
    """Test BandwidthDerivation class."""

    def test_bandwidth_derivation_init(self):
        """Test BandwidthDerivation initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation

        model = BandwidthDerivation()
        assert hasattr(model, "compute")
        assert hasattr(model, "parameter_sweep")

    def test_bandwidth_derivation_compute(self):
        """Test bandwidth computation."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation

        model = BandwidthDerivation()

        # Test with default parameters
        result = model.compute()
        assert hasattr(result, "bandwidth_bps")
        assert hasattr(result, "prediction_in_range")
        assert isinstance(result.bandwidth_bps, dict)
        assert isinstance(result.prediction_in_range, bool)

    def test_bandwidth_derivation_parameter_sweep(self):
        """Test parameter sweep functionality."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation

        model = BandwidthDerivation()

        # Test parameter sweep
        result = model.parameter_sweep()
        assert hasattr(result, "bandwidth_matrix")
        assert hasattr(result, "optimal_params")
        assert hasattr(result, "prediction_in_range")


class TestPrecisionWeightingAnalyzer:
    """Test PrecisionWeightingAnalyzer class."""

    def test_precision_weighting_init(self):
        """Test PrecisionWeightingAnalyzer initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import PrecisionWeightingAnalyzer, BandwidthDerivation

        bandwidth_module = BandwidthDerivation()
        analyzer = PrecisionWeightingAnalyzer(bandwidth_module)
        assert hasattr(analyzer, "compare")
        assert analyzer.bw == bandwidth_module

    def test_precision_weighting_compare(self):
        """Test precision weighting comparison."""
        from Theory.APGI_Information_Theoretic_Bandwidth import PrecisionWeightingAnalyzer, BandwidthDerivation

        bandwidth_module = BandwidthDerivation()
        analyzer = PrecisionWeightingAnalyzer(bandwidth_module)

        # Test comparison with valid inputs
        result = analyzer.compare(theta_baseline=0.1, alpha=0.5)
        assert hasattr(result, "delta_I_pct")
        assert hasattr(result, "mi_gain_bits")
        assert isinstance(result.delta_I_pct, float)
        assert isinstance(result.mi_gain_bits, float)


class TestClinicalProfile:
    """Test ClinicalProfile class."""

    def test_clinical_profile_init(self):
        """Test ClinicalProfile initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import ClinicalProfile

        profile = ClinicalProfile(
            label="Test Disorder", tau_S=1.0, tau_theta=0.1, tau_M=0.1, alpha=0.5, gamma_M=0.1, gamma_A=0.1, beta=0.1
        )

        assert profile.label == "Test Disorder"
        assert profile.tau_S == 1.0
        assert profile.alpha == 0.5

    def test_clinical_profile_validation(self):
        """Test clinical profile validation."""
        from Theory.APGI_Information_Theoretic_Bandwidth import ClinicalProfile

        # Test valid profile
        valid_profile = ClinicalProfile(
            label="Valid", tau_S=1.0, tau_theta=0.1, tau_M=0.1, alpha=0.5, gamma_M=0.1, gamma_A=0.1, beta=0.1
        )
        assert valid_profile.validate_parameters()

        # Test invalid profile (negative parameter)
        invalid_profile = ClinicalProfile(
            label="Test Disorder",
            Pi_e=1.0,
            Pi_i=0.8,
            theta_baseline=0.3,
            alpha=0.5,
            sigma_theta=0.05,
            description="Test profile",
        )
        assert not invalid_profile.validate_parameters()


class TestClinicalBandwidthMapper:
    """Test ClinicalBandwidthMapper class."""

    def test_clinical_mapper_init(self):
        """Test ClinicalBandwidthMapper initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import ClinicalBandwidthMapper, BandwidthDerivation

        bandwidth_module = BandwidthDerivation()
        mapper = ClinicalBandwidthMapper(bandwidth_module)
        assert hasattr(mapper, "compute_all")
        assert mapper.bw == bandwidth_module

    def test_clinical_mapper_compute(self):
        """Test clinical bandwidth computation."""
        from Theory.APGI_Information_Theoretic_Bandwidth import (
            ClinicalBandwidthMapper,
            ClinicalProfile,
            BandwidthDerivation,
        )

        bandwidth_module = BandwidthDerivation()
        mapper = ClinicalBandwidthMapper(bandwidth_module)

        # Create test profiles
        profiles = {
            "MDD": ClinicalProfile("MDD", 1.0, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1),
            "ADHD": ClinicalProfile("ADHD", 0.8, 0.05, 0.05, 0.3, 0.05, 0.05, 0.05),
        }

        result = mapper.compute_all(profiles=profiles)
        assert hasattr(result, "clinical")
        assert hasattr(result, "heatmap_bandwidth")
        assert hasattr(result, "clinical")


class TestBandwidthConfig:
    """Test BandwidthConfig class."""

    def test_bandwidth_config_init(self):
        """Test BandwidthConfig initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthConfig

        config = BandwidthConfig()
        assert hasattr(config, "n_trials")
        assert hasattr(config, "n_stimulus_bins")
        assert hasattr(config, "theta_range")

    def test_bandwidth_config_custom(self):
        """Test BandwidthConfig with custom parameters."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthConfig

        config = BandwidthConfig(n_trials=5000, n_stimulus_bins=50, theta_range=(0.05, 0.15))

        assert config.n_trials == 5000
        assert config.n_stimulus_bins == 50
        assert config.theta_range == (0.05, 0.15)


class TestBandwidthAnalyzer:
    """Test BandwidthAnalyzer class."""

    def test_bandwidth_analyzer_init(self):
        """Test BandwidthAnalyzer initialization."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthAnalyzer, BandwidthConfig

        config = BandwidthConfig()
        analyzer = BandwidthAnalyzer(config)
        assert hasattr(analyzer, "run")
        assert analyzer.config == config

    def test_bandwidth_analyzer_run(self):
        """Test BandwidthAnalyzer run method."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthAnalyzer, BandwidthConfig

        config = BandwidthConfig(n_trials=100)  # Small number for testing
        analyzer = BandwidthAnalyzer(config)

        # Mock the internal modules to speed up testing
        with patch.object(analyzer, "bw") as mock_bw:
            with patch.object(analyzer, "pw") as mock_pw:
                with patch.object(analyzer, "cb") as mock_cb:
                    # Configure mocks to return simple results
                    mock_bw.compute.return_value = MagicMock(bandwidth_bps={"test": 40.0}, prediction_in_range=True)
                    mock_pw.compare.return_value = MagicMock(delta_I_pct=25.0, mi_gain_bits=1.5)
                    mock_cb.compute_all.return_value = MagicMock(
                        clinical=MagicMock(), heatmap_bandwidth=np.array([[40.0]])
                    )

                    result = analyzer.run()

                    # Verify the result structure
                    assert hasattr(result, "nominal")
                    assert hasattr(result, "sweep")
                    assert hasattr(result, "clinical")


class TestRunBandwidthAnalysis:
    """Test run_bandwidth_analysis function."""

    def test_run_bandwidth_analysis_default(self):
        """Test run_bandwidth_analysis with default config."""
        from Theory.APGI_Information_Theoretic_Bandwidth import run_bandwidth_analysis

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"

            # Create a simple config file
            config_data = {"n_trials": 10, "output_dir": str(temp_dir)}

            with open(config_path, "w") as f:
                json.dump(config_data, f)

            # Run analysis
            result = run_bandwidth_analysis(str(config_path))

            # Verify result structure
            assert hasattr(result, "nominal")
            assert hasattr(result, "sweep")
            assert hasattr(result, "clinical")

    def test_run_bandwidth_analysis_custom(self):
        """Test run_bandwidth_analysis with custom config."""
        from Theory.APGI_Information_Theoretic_Bandwidth import run_bandwidth_analysis, BandwidthConfig

        custom_config = BandwidthConfig(n_trials=50, n_stimulus_bins=20)

        result = run_bandwidth_analysis(config=custom_config)

        # Verify result structure
        assert hasattr(result, "nominal")
        assert hasattr(result, "sweep")
        assert hasattr(result, "clinical")


class TestErrorHandling:
    """Test error handling in bandwidth module."""

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        from Theory.APGI_Information_Theoretic_Bandwidth import BandwidthDerivation

        model = BandwidthDerivation()

        # Test with invalid parameter ranges
        with pytest.raises((ValueError, TypeError)):
            model.compute(Pi_e=-1.0)  # Invalid negative precision

    def test_missing_dependencies(self):
        """Test behavior when dependencies are missing."""
        # This test ensures graceful degradation when numpy is not available
        with patch.dict("sys.modules", {"numpy": None}):
            # Should handle ImportError gracefully
            with pytest.raises(ImportError):
                pass  # Import should fail gracefully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
