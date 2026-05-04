"""
APGI Theory Package
===================

A comprehensive framework for testing and validating the APGI theory of consciousness.

This package contains protocols, validation tools, and implementations for
testing APGI predictions against empirical data.
"""

__version__ = "1.0.0"
__author__ = "APGI Framework"

__all__ = []

# NOTE:
# Keep package import side-effect free. Importing the full validation stack can
# pull in large optional scientific dependencies and may fail in minimal
# environments (or during unit test collection). Consumers should import
# `Validation.Master_Validation.APGIMasterValidator` directly when needed.
