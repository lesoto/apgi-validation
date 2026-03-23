"""
Simple test to verify threshold dynamics functionality works.
"""

import pytest
from pathlib import Path

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# Test the core threshold dynamics function directly
def test_threshold_dynamics_core():
    """Test threshold dynamics function directly."""
    from APGI_Equations.DynamicalSystemEquations import threshold_dynamics

    # Test basic functionality
    theta_new = threshold_dynamics(
        theta=3.0,
        theta_0_sleep=2.0,
        theta_0_alert=4.0,
        A=0.5,
        gamma_M=0.1,
        M=1.0,
        lambda_S=0.01,
        S=1.0,
        tau_theta=50.0,
        sigma_theta=0.2,
        dt=0.1,
    )

    # Should return a valid float
    assert isinstance(theta_new, float)
    assert theta_new > 0

    print("SUCCESS: threshold_dynamics function works correctly")


if __name__ == "__main__":
    pytest.main([__file__])
