# Use minimal imports to avoid F401 errors
import sys
import os
import importlib.util
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Load APGI_Liquid_Network_Implementation from hyphenated filename
spec = importlib.util.spec_from_file_location(
    "APGI_Liquid_Network_Implementation",
    Path(__file__).parent.parent / "APGI_Liquid_Network_Implementation.py",
)
APGI_Liquid_Network_Implementation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(APGI_Liquid_Network_Implementation)
APGILiquidNetwork = APGI_Liquid_Network_Implementation.APGILiquidNetwork


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
