"""
APGI Validation Package.
=======================

Contains validation protocols and GUI tools for testing APGI theory predictions.

LEVEL DESIGNATION: All outputs are Level 2 (information-theoretic).
Bridge to Level 1 requires APGI_Thermodynamic_Program_Aggregator.
This script does NOT claim thermodynamic implications without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If the validation package fails to coordinate protocols effectively,
or if the validation framework cannot load required modules (error rate > 5%),
or if the validation tools cannot produce consistent results across protocols
(variability > 15%), then the APGI validation framework claim is falsified.
This would indicate that the validation infrastructure is not robust.
"""

import importlib.util

# Apply NumPy compatibility fixes for Python 3.14
import sys
from pathlib import Path

if sys.version_info >= (3, 14):
    try:
        import numpy as np

        # Store original functions to avoid recursion
        _original_wrapreduction = None
        if hasattr(np, "_core") and hasattr(np._core, "fromnumeric"):
            _original_wrapreduction = np._core.fromnumeric._wrapreduction

        # Fix zero-size array reduction operations
        def _safe_wrapreduction(obj, ufunc, method, *args, **kwargs):
            """Safe _wrapreduction that handles zero-size arrays."""
            try:
                if _original_wrapreduction is not None:
                    return _original_wrapreduction(obj, ufunc, method, *args, **kwargs)
                else:
                    # Fallback if original not available
                    return ufunc.reduce(obj, axis=None, out=None, keepdims=False)
            except ValueError as e:
                if "zero-size array to reduction operation" in str(e):
                    # Handle zero-size arrays by returning appropriate result
                    if hasattr(obj, "size") and obj.size == 0:
                        if method == "prod":
                            return np.array([], dtype=getattr(obj, "dtype", float))
                        elif method == "sum":
                            return np.array([], dtype=getattr(obj, "dtype", float))
                        elif method == "multiply":
                            return np.array([], dtype=getattr(obj, "dtype", float))
                        else:
                            return np.array([], dtype=getattr(obj, "dtype", float))
                raise

        # Patch the _wrapreduction function
        if hasattr(np, "_core") and hasattr(np._core, "fromnumeric"):
            np._core.fromnumeric._wrapreduction = _safe_wrapreduction

        # Wrap the multiply ufunc
        class SafeUFunc:
            def __init__(self, ufunc):
                self.ufunc = ufunc

            def __call__(self, *args, **kwargs):
                try:
                    return self.ufunc(*args, **kwargs)
                except ValueError as e:
                    if "zero-size array to reduction operation" in str(e):
                        # Handle zero-size arrays by returning appropriate result
                        for arg in args:
                            if hasattr(arg, "size") and arg.size == 0:
                                return np.array([], dtype=getattr(arg, "dtype", float))
                    raise

            def reduce(self, *args, **kwargs):
                try:
                    return self.ufunc.reduce(*args, **kwargs)
                except ValueError as e:
                    if "zero-size array to reduction operation" in str(e):
                        # Handle zero-size arrays by returning appropriate result
                        for arg in args:
                            if hasattr(arg, "size") and arg.size == 0:
                                return np.array([], dtype=getattr(arg, "dtype", float))
                    raise

        # Wrap the multiply ufunc
        _original_multiply = np.multiply
        np.multiply = SafeUFunc(_original_multiply)

    except ImportError:
        pass  # NumPy not available

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
