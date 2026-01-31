"""
APGI Theory Package
===================

A comprehensive framework for testing and validating the APGI (Attention,
Phenomenal consciousness, Global workspace, Ignition) theory of consciousness.

This package contains protocols, validation tools, and implementations for
testing APGI predictions against empirical data.
"""

__version__ = "1.3.0"
__author__ = "APGI Research Team"

# Import main classes for convenience
try:
    from .Validation.APGI_Master_Validation import APGIMasterValidator

    __all__ = ["APGIMasterValidator"]
except ImportError:
    # If imports fail, provide minimal package
    __all__ = []
