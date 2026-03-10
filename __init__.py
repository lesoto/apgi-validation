"""
APGI Theory Package
===================

A comprehensive framework for testing and validating the APGI theory of consciousness.

This package contains protocols, validation tools, and implementations for
testing APGI predictions against empirical data.
"""

__version__ = "1.0.0"
__author__ = "APGI Framework"

# Import main classes for convenience
try:
    from Validation.Master_Validation import APGIMasterValidator

    __all__ = ["APGIMasterValidator"]
except ImportError as e:
    # Re-raise import error with helpful message instead of silently failing
    raise ImportError(
        f"Failed to import APGIMasterValidator from Validation.Master_Validation. "
        f"This usually means the Validation module is missing or corrupted. "
        f"Original error: {e}"
    ) from e
