"""
APGI Validation Package.
======================

Contains validation protocols and GUI tools for testing APGI theory predictions.
"""

import importlib.util
import os
import sys
from pathlib import Path

# Import APGI-Master-Validation.py using importlib (needed for hyphenated filename)
validation_dir = Path(__file__).parent
master_validation_path = validation_dir / "APGI-Master-Validation.py"

spec = importlib.util.spec_from_file_location(
    "APGI_Master_Validation", master_validation_path
)
master_validation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_validation_module)

# Extract the class
APGIMasterValidator = master_validation_module.APGIMasterValidator

__all__ = ["APGIMasterValidator"]
