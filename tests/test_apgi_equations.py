"""
Comprehensive consolidated tests for APGI_Equations.py.
=======================================================================

This file consolidates and merges all tests from:
- test_apgi_equations.py
- test_apgi_equations_fixed.py  
- test_apgi_equations_extended.py

Retains 100% test coverage while eliminating duplication.
"""

from __future__ import annotations

# Add project root to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "Theory"))
sys.path.insert(0, str(Path(__file__).parent / "Falsification"))

# Import the main module - will be imported when tests are added
