"""
APGI Validation Package.
=======================

Contains validation protocols and GUI tools for testing APGI theory predictions.
"""

import importlib.util
from pathlib import Path

# Import Master_Validation.py using importlib (needed for hyphenated filename)
validation_dir = Path(__file__).parent
master_validation_path = validation_dir / "Master_Validation.py"

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

# Import GUI module using importlib (needed for hyphenated filename)
gui_path = validation_dir.parent / "APGI_Validation_GUI.py"

try:
    gui_spec = importlib.util.spec_from_file_location("APGI_Validation_GUI", gui_path)
    if gui_spec and gui_spec.loader:
        gui_module = importlib.util.module_from_spec(gui_spec)
        gui_spec.loader.exec_module(gui_module)

        # Extract GUI classes/functions
        if hasattr(gui_module, "APGIValidationGUI"):
            APGIValidationGUI = gui_module.APGIValidationGUI
        if hasattr(gui_module, "safe_import_module"):
            safe_import_module = gui_module.safe_import_module
    else:
        raise ImportError("Could not create module spec for APGI_Validation_GUI.py")
except Exception as e:
    # Log the error but don't crash the import
    import warnings

    warnings.warn(f"Failed to import APGI_Validation_GUI.py: {e}", ImportWarning)
    APGIValidationGUI = None
    safe_import_module = None

__all__ = ["APGIMasterValidator", "APGIValidationGUI", "safe_import_module"]
