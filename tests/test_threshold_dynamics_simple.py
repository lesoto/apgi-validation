# Use minimal imports to avoid F401 errors
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import from the implementation file directly
from APGI_Liquid_Network_Implementation import APGILiquidNetwork


# Import wrapper function from test_apgi_equations.py
def wrapped_threshold_dynamics(
    theta: float, theta_0_sleep: float, theta_0_alert: float
) -> float:
    """Wrapper for threshold_dynamics with default parameters."""
    return APGILiquidNetwork.threshold_dynamics(
        theta=theta,
        theta_0_sleep=theta_0_sleep,
        theta_0_alert=theta_0_alert,
    )
