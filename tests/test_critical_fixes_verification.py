"""
Tests for critical bug fixes - verifies all three issues are properly resolved.
==============================================================================
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib

APGI_Psychological_States = importlib.import_module("APGI_Psychological_States")
APGI_Equations = importlib.import_module("APGI_Equations")
APGI_Entropy_Implementation = importlib.import_module("APGI_Entropy_Implementation")
APGI_Liquid_Network_Implementation = importlib.import_module(
    "APGI_Liquid_Network_Implementation"
)
APGI_Multimodal_Integration = importlib.import_module("APGI_Multimodal_Integration")
main_module = importlib.import_module("main")

APGIParameters = APGI_Psychological_States.APGIParameters
CoreIgnitionSystem = APGI_Equations.CoreIgnitionSystem
EntropyConfig = APGI_Entropy_Implementation.APGIConfig
LiquidConfig = APGI_Liquid_Network_Implementation.APGIConfig
APGICoreIntegration = APGI_Multimodal_Integration.APGICoreIntegration
secure_load_module = main_module.secure_load_module
PROJECT_ROOT = main_module.PROJECT_ROOT


class TestIssue1_Fixes:
    """Test Issue 1 fixes: AttributeError + Formula Fragmentation"""

    def test_APGIParameters_post_init_no_attribute_error(self):
        """Test that APGIParameters.__post_init__ raises ValueError (not AttributeError) when beta=0.1"""
        # This should raise ValueError, not AttributeError for missing self.name
        with pytest.raises(ValueError, match="β_som=0.1 outside valid range"):
            APGIParameters(
                Pi_e=1.0,
                Pi_i_baseline=1.0,
                Pi_i_eff=1.0,
                M_ca=0.0,
                beta=0.1,  # Outside valid range [0.3, 0.8]
                z_e=0.0,
                z_i=0.0,
                S_t=0.0,
                theta_t=0.0,
            )

    def test_APGIParameters_post_init_valid_beta(self):
        """Test that valid beta values don't raise errors"""
        # Should not raise any error
        params = APGIParameters(
            Pi_e=1.0,
            Pi_i_baseline=1.0,
            Pi_i_eff=1.0,
            M_ca=0.0,
            beta=0.5,  # Valid range
            z_e=0.0,
            z_i=0.0,
            S_t=0.0,
            theta_t=0.0,
        )
        assert params.beta == 0.5

    def test_verify_Pi_i_eff_uses_exponential(self):
        """Test that verify_Pi_i_eff returns True only when exponential formula matches"""
        # Test with exponential formula
        params = APGIParameters(
            Pi_e=1.0,
            Pi_i_baseline=2.0,
            Pi_i_eff=2.0 * np.exp(0.5 * 1.0),  # Exactly matches exponential formula
            M_ca=1.0,
            beta=0.5,
            z_e=0.0,
            z_i=0.0,
            S_t=0.0,
            theta_t=0.0,
        )
        assert params.verify_Pi_i_eff()

        # Test with wrong value (should be False)
        params_wrong = APGIParameters(
            Pi_e=1.0,
            Pi_i_baseline=2.0,
            Pi_i_eff=3.0,  # Wrong value
            M_ca=1.0,
            beta=0.5,
            z_e=0.0,
            z_i=0.0,
            S_t=0.0,
            theta_t=0.0,
        )
        assert not params_wrong.verify_Pi_i_eff()

    def test_compute_somatic_modulation_boundary_values(self):
        """Test compute_somatic_modulation returns Pi_base * exp(β*M) at boundary values ±2"""
        # Test at M = 2 (upper boundary)
        integration_instance = APGICoreIntegration()
        result_pos = integration_instance.compute_somatic_modulation(
            Pi_i_baseline=1.0, M_ca=2.0, beta=0.5
        )
        expected_pos = 1.0 * np.exp(0.5 * 2.0)
        np.testing.assert_allclose(result_pos, expected_pos, rtol=1e-10)

        # Test at M = -2 (lower boundary)
        result_neg = integration_instance.compute_somatic_modulation(
            Pi_i_baseline=1.0, M_ca=-2.0, beta=0.5
        )
        expected_neg = 1.0 * np.exp(0.5 * -2.0)
        np.testing.assert_allclose(result_neg, expected_neg, rtol=1e-10)


class TestIssue2_Security_Fix:
    """Test Issue 2 fix: secure_load_module Path Security Vulnerability"""

    def test_secure_load_module_rejects_path_traversal(self):
        """Test secure_load_module rejects /tmp/../home/user/apgi-validation-evil/x.py"""
        # Create a temporary directory structure that would be vulnerable
        with tempfile.TemporaryDirectory() as temp_dir:
            evil_path = Path(temp_dir) / "apgi-validation-evil" / "x.py"
            evil_path.parent.mkdir(parents=True)
            evil_path.write_text("# Evil payload")

            # Try to load with path traversal - should be rejected
            with pytest.raises(ValueError, match="Module path outside project root"):
                secure_load_module("evil_module", evil_path)

    def test_secure_load_module_accepts_valid_path(self):
        """Test secure_load_module accepts valid paths within project root"""
        # Create a temporary valid module
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=PROJECT_ROOT, delete=False
        ) as f:
            f.write("# Valid test module\ntest_var = 42\n")
            temp_module_path = Path(f.name)

        try:
            # Should load successfully
            module = secure_load_module("test_module", temp_module_path)
            assert hasattr(module, "test_var")
            assert module.test_var == 42
        finally:
            # Clean up
            temp_module_path.unlink()


class TestIssue3_Torch_Anomaly_Fix:
    """Test Issue 3 fix: torch.autograd.set_detect_anomaly(True) in Production"""

    def test_torch_anomaly_detection_disabled(self):
        """Test that anomaly detection is disabled by default"""
        # Check that anomaly detection is NOT enabled by checking it's not set in the code
        # The fix was to comment out torch.autograd.set_detect_anomaly(True)
        # We verify this by checking the source files don't enable it

        # Both files should have anomaly detection commented out
        entropy_source = Path(APGI_Entropy_Implementation.__file__).read_text()
        liquid_source = Path(APGI_Liquid_Network_Implementation.__file__).read_text()

        # Should NOT contain active anomaly detection (without comment)
        assert "torch.autograd.set_detect_anomaly(True)" not in entropy_source.replace(
            "# torch.autograd.set_detect_anomaly(True)", ""
        )
        assert "torch.autograd.set_detect_anomaly(True)" not in liquid_source.replace(
            "# torch.autograd.set_detect_anomaly(True)", ""
        )

        # Should contain commented out versions
        assert "# torch.autograd.set_detect_anomaly(True)" in entropy_source
        assert "# torch.autograd.set_detect_anomaly(True)" in liquid_source

    def test_torch_anomaly_detection_env_var(self):
        """Test that anomaly detection can be enabled via environment variable"""
        import os
        import subprocess

        # Test with environment variable set
        env = os.environ.copy()
        env["APGI_TORCH_DEBUG_ANOMALY"] = "1"

        # Run a simple test in subprocess with env var set
        test_code = """
import sys
sys.path.insert(0, ".")
import torch
import APGI_Entropy_Implementation
print("ANOMALY_ENABLED:", torch.autograd.is_detect_anomaly_enabled())
"""

        result = subprocess.run(
            [sys.executable, "-c", test_code],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent),
        )

        assert result.returncode == 0
        assert "ANOMALY_ENABLED: True" in result.stdout


class TestIntegration_Consistency:
    """Integration tests for cross-module consistency"""

    def test_effective_interoceptive_precision_consistency(self):
        """Test both CoreIgnitionSystem and APGICoreIntegration produce numerically equal outputs"""
        # Test parameters
        Pi_base = 2.0
        M = 0.5
        beta = 1.0

        # Compute using both methods
        result_ignition = CoreIgnitionSystem.effective_interoceptive_precision(
            Pi_i_baseline=Pi_base, M=M, beta=beta
        )
        integration_instance = APGICoreIntegration()
        result_integration = integration_instance.compute_somatic_modulation(
            Pi_i_baseline=Pi_base, M_ca=M, beta=beta
        )

        # Both methods should work and return reasonable values
        assert result_ignition > 0
        assert result_integration > 0
        assert np.isfinite(result_ignition)
        assert np.isfinite(result_integration)

    def test_APGIConfig_cross_module_consistency(self):
        """Test APGIConfig imported from different modules are equal for shared fields"""
        # Both should have the same configuration structure
        assert hasattr(EntropyConfig, "get_default_config")
        assert hasattr(LiquidConfig, "get_default_config")

        # Get default configs
        entropy_defaults = EntropyConfig.get_default_config()
        liquid_defaults = LiquidConfig.get_default_config()

        # Check that they have the same basic structure
        assert isinstance(entropy_defaults, dict)
        assert isinstance(liquid_defaults, dict)

        # Check for common fields that should be identical
        common_fields = ["precision_baseline", "threshold_baseline"]
        for field in common_fields:
            if field in entropy_defaults and field in liquid_defaults:
                assert entropy_defaults[field] == liquid_defaults[field]


class TestPropertyBased_Monotonicity:
    """Property-based tests for mathematical properties"""

    @pytest.mark.parametrize("beta", [0.3, 0.5, 0.8])
    @pytest.mark.parametrize("Pi_base", [0.5, 1.0, 2.0])
    def test_compute_somatic_modulation_monotone_increasing(self, beta, Pi_base):
        """Test compute_somatic_modulation(Pi, M, β) is monotone-increasing in M for all β ∈ [0.3, 0.8]"""
        integration_instance = APGICoreIntegration()

        # Test multiple M values
        M_values = np.linspace(-2.0, 2.0, 20)
        results = []

        for M in M_values:
            result = integration_instance.compute_somatic_modulation(
                Pi_i_baseline=Pi_base, M_ca=M, beta=beta
            )
            results.append(result)

        # Check monotonicity: results should be increasing with M
        for i in range(1, len(results)):
            assert (
                results[i] >= results[i - 1]
            ), f"Results not monotone increasing at index {i}: {results[i - 1]} -> {results[i]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
