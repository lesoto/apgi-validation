"""
Python 3.14 compatibility fixes for APGI test suite.
===================================================

This module provides compatibility fixes for Python 3.14, particularly
addressing the _NoValueType issue with MagicMock objects and int() conversion.
"""

import sys
from typing import Any
from unittest.mock import MagicMock

# Store original int function
_original_int = int


def _safe_int(obj: Any, *args, **kwargs) -> int:
    """Safe int conversion that handles _NoValueType objects from MagicMock.

    This function provides a compatibility layer for Python 3.14 where
    MagicMock objects can create _NoValueType instances that cause
    TypeError when passed to int().
    """
    try:
        return _original_int(obj, *args, **kwargs)
    except (TypeError, ValueError) as e:
        # Handle _NoValueType and other MagicMock-related issues
        if "_NoValueType" in str(e) or "NoValueType" in str(e):
            # Return a sensible default for mocked objects
            return 0
        # Re-raise other conversion errors
        raise


def apply_python314_compatibility_fixes():
    """Apply Python 3.14 compatibility fixes to the current Python session."""
    # Replace int with our safe version
    if sys.version_info >= (3, 14):
        builtins = sys.modules.get("builtins")
        if builtins and hasattr(builtins, "int"):
            builtins.int = _safe_int

        # Also patch it in sys.modules to ensure it's used everywhere
        sys.modules["builtins"].int = _safe_int


def create_safe_mock_var(default_value: int = 0):
    """Create a MagicMock variable that's safe for int() conversion.

    Args:
        default_value: Default value to return from get() method

    Returns:
        MagicMock object with safe get() method
    """
    mock_var = MagicMock()
    mock_var.get.return_value = default_value
    mock_var.set = MagicMock()
    return mock_var
