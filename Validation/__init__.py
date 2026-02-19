"""
APGI Validation Package.
======================

Contains validation protocols and GUI tools for testing APGI theory predictions.
"""

import importlib.util
from pathlib import Path

# Import Master-Validation.py using importlib (needed for hyphenated filename)
validation_dir = Path(__file__).parent
master_validation_path = validation_dir / "Master-Validation.py"

try:
    spec = importlib.util.spec_from_file_location(
        "APGI_Master_Validation", master_validation_path
    )
    if spec and spec.loader:
        master_validation_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(master_validation_module)

        # Extract the class
        APGIMasterValidator = master_validation_module.APGIMasterValidator
    else:
        raise ImportError("Could not create module spec for Master-Validation.py")
except Exception as e:
    # Log the error but don't crash the import
    import warnings

    warnings.warn(f"Failed to import Master-Validation.py: {e}", ImportWarning)
    APGIMasterValidator = None

__all__ = ["APGIMasterValidator"]
