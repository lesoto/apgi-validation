"""
APGI Framework Utils Package
============================

Utility modules for the APGI framework.
"""

import os
import secrets
import sys
import types

import numpy as np

# Fix for missing numpy.lib.array_utils in some NumPy versions (e.g. 1.26.4 with Python 3.14)
# This module is expected by PyMC 5.27.0 and ArviZ.
if not hasattr(np.lib, "array_utils"):
    try:
        # Create module
        array_utils = types.ModuleType("numpy.lib.array_utils")
        # Link it to numpy.lib
        np.lib.array_utils = array_utils
        # Add to sys.modules
        sys.modules["numpy.lib.array_utils"] = array_utils

        # Import needed functions from where they actually are in NumPy 1.26
        try:
            from numpy.core.multiarray import normalize_axis_index

            array_utils.normalize_axis_index = normalize_axis_index  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            pass

        try:
            from numpy.core.numeric import normalize_axis_tuple

            array_utils.normalize_axis_tuple = normalize_axis_tuple  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            pass
    except Exception:
        # If anything fails during monkey-patching, we still want to continue
        pass

# Note: Standard dependency-handling logic follows

import warnings
from typing import Any, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, continue with system environment variables


# Security: Check for required environment variables at import time
def _check_required_env_vars():
    """Check for required security environment variables and raise error if missing."""
    missing_vars = []

    # Check for PICKLE_SECRET_KEY
    if not os.environ.get("PICKLE_SECRET_KEY"):
        missing_vars.append("PICKLE_SECRET_KEY")

    # Check for APGI_BACKUP_HMAC_KEY
    if not os.environ.get("APGI_BACKUP_HMAC_KEY"):
        missing_vars.append("APGI_BACKUP_HMAC_KEY")

    if missing_vars:
        # In production, raise an error rather than generating insecure ephemeral keys
        if os.environ.get("APGI_ENV") == "production":
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                "Set these in your environment or .env file before running in production."
            )
        else:
            # In development, warn but allow continued execution
            warnings.warn(
                f"Missing environment variables: {', '.join(missing_vars)}. "
                "Pickle signing and backup HMAC verification are disabled. "
                "Set these for production use.",
                RuntimeWarning,
                stacklevel=2,
            )


_check_required_env_vars()

from .constants import (
    DIM_CONSTANTS,
    MODEL_PARAMS,
    NEURAL_DEFAULTS,
    PARAMETER_BOUNDS,
    PCI_NORMALIZATION,
    SPECIES_METRICS,
    SYSTEM_DEFAULTS,
)
from .data_validation import DataValidator

# Import falsification thresholds from local utils directory
from .falsification_thresholds import (  # Validation Protocol 12 constants
    F1_1_ALPHA,
    F1_1_MIN_ADVANTAGE_PCT,
    F1_1_MIN_COHENS_D,
    F1_5_COHENS_D_MIN,
    F1_5_PAC_INCREASE_MIN,
    F1_5_PAC_MI_MIN,
    F1_5_PERMUTATION_ALPHA,
    F2_1_ALPHA,
    F2_1_MIN_ADVANTAGE_PCT,
    F2_1_MIN_COHENS_H,
    F2_1_MIN_PP_DIFF,
    F2_2_ALPHA,
    F2_2_MIN_CORR,
    F2_2_MIN_FISHER_Z,
    F2_3_ALPHA,
    F2_3_MIN_BETA,
    F2_3_MIN_R2,
    F2_3_MIN_RT_ADVANTAGE_MS,
    F2_3_MIN_STANDARDIZED_BETA,
    F2_4_ALPHA,
    F2_4_MIN_BETA_INTERACTION,
    F2_4_MIN_CONFIDENCE_EFFECT_PCT,
    F5_1_MIN_ALPHA,
    F5_1_MIN_COHENS_D,
    F5_1_MIN_PROPORTION,
    F5_2_MIN_CORRELATION,
    F5_2_MIN_PROPORTION,
    F5_3_FALSIFICATION_RATIO,
    F5_3_MIN_COHENS_D,
    F5_3_MIN_GAIN_RATIO,
    F5_3_MIN_PROPORTION,
    F5_4_MIN_PEAK_SEPARATION,
    F5_4_MIN_PROPORTION,
    F5_5_MIN_LOADING,
    F5_5_PCA_MIN_VARIANCE,
    F5_6_ALPHA,
    F5_6_MIN_COHENS_D,
    F5_6_MIN_PERFORMANCE_DIFF_PCT,
    F6_1_CLIFFS_DELTA_MIN,
    F6_1_LTCN_MAX_TRANSITION_MS,
    F6_1_MANN_WHITNEY_ALPHA,
    F6_2_LTCN_MIN_WINDOW_MS,
    F6_2_MIN_CURVE_FIT_R2,
    F6_2_MIN_INTEGRATION_RATIO,
    F6_2_MIN_R2,
    F6_2_WILCOXON_ALPHA,
    F6_5_HYSTERESIS_MAX,
    F6_5_HYSTERESIS_MIN,
    V12_1_ALPHA,
    V12_1_MIN_COHENS_D,
    V12_1_MIN_ETA_SQUARED,
    V12_1_MIN_IGNITION_REDUCTION_PCT,
    V12_1_MIN_P3B_REDUCTION_PCT,
    V12_2_ALPHA,
    V12_2_FALSIFICATION_CORR,
    V12_2_FALSIFICATION_PILLAIS,
    V12_2_MIN_CORRELATION,
    V12_2_MIN_PILLAIS_TRACE,
)

# Import lightweight utilities for easy access.
# Keep this package import as side-effect free as possible: avoid importing large
# protocol runners or scientific stacks at import time.
from .sample_data_generator import SampleDataGenerator, generate_sample_multimodal_data

# Batch processing is optional and may pull in large dependency graphs.
# Provide it lazily via __getattr__ instead of importing here.
BATCH_PROCESSOR_AVAILABLE = False
BatchProcessor: Any = None  # type: ignore[misc,assignment]

# Check optional protocol dependencies and emit an import-level warning if absent.
_OPTIONAL_DEPS = {
    "pymc": ("pymc", "5.0.0"),  # Bayesian modeling (VP-11, FP-10)
    "arviz": ("arviz", "0.16.0"),  # Bayesian visualization
    "mne": ("mne", "1.6.0"),  # EEG processing (VP-01, VP-07, VP-09)
    "lifelines": ("lifelines", "0.27.0"),  # Survival analysis (VP-02, FP-02)
    "captum": ("captum", "0.7.0"),  # Deep learning explainability (VP-09)
    "salib": ("SALib", "1.4.0"),  # Sensitivity analysis (FP-08)
    "sklearn": ("sklearn", "1.5.0"),  # Machine learning (VP-01, VP-02, FP-03)
    "sympy": ("sympy", "1.12"),  # Symbolic mathematics (FP-07, FP-08)
}

import importlib.util

_MISSING_OPTIONAL_DEPS = []

for dep_name, (import_name, min_version) in _OPTIONAL_DEPS.items():
    try:
        if importlib.util.find_spec(import_name) is None:
            _MISSING_OPTIONAL_DEPS.append(dep_name)
    except Exception:
        _MISSING_OPTIONAL_DEPS.append(dep_name)

if _MISSING_OPTIONAL_DEPS:
    warnings.warn(
        f"Optional protocol dependencies missing: {', '.join(_MISSING_OPTIONAL_DEPS)}. "
        f"Install with: pip install -r requirements-protocols.txt. "
        f"Some validation/falsification protocols will be unavailable.",
        ImportWarning,
        stacklevel=2,
    )

from .cache_manager import CacheManager
from .config_manager import ConfigManager


def __getattr__(name: str) -> Any:
    if name == "BatchProcessor":
        try:
            from .batch_processor import (
                BatchProcessor as _BatchProcessor,
            )  # type: ignore

            globals()["BatchProcessor"] = _BatchProcessor
            globals()["BATCH_PROCESSOR_AVAILABLE"] = True
            return _BatchProcessor
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"BatchProcessor unavailable: {exc}",
                ImportWarning,
                stacklevel=2,
            )
            globals()["BATCH_PROCESSOR_AVAILABLE"] = False
            return None
    raise AttributeError(name)


__all__ = [
    "SampleDataGenerator",
    "generate_sample_multimodal_data",
    "DataValidator",
    "BatchProcessor",
    "ConfigManager",
    "CacheManager",
    "MODEL_PARAMS",
    "NEURAL_DEFAULTS",
    "SPECIES_METRICS",
    "SYSTEM_DEFAULTS",
    "PARAMETER_BOUNDS",
    "PCI_NORMALIZATION",
    "DIM_CONSTANTS",
    # Falsification threshold constants
    "F1_1_MIN_ADVANTAGE_PCT",
    "F1_1_MIN_COHENS_D",
    "F1_1_ALPHA",
    "F1_5_PAC_MI_MIN",
    "F1_5_PAC_INCREASE_MIN",
    "F1_5_COHENS_D_MIN",
    "F1_5_PERMUTATION_ALPHA",
    "F2_1_MIN_ADVANTAGE_PCT",
    "F2_1_MIN_PP_DIFF",
    "F2_1_MIN_COHENS_H",
    "F2_1_ALPHA",
    "F2_2_MIN_CORR",
    "F2_2_MIN_FISHER_Z",
    "F2_2_ALPHA",
    "F2_3_MIN_RT_ADVANTAGE_MS",
    "F2_3_MIN_BETA",
    "F2_3_MIN_STANDARDIZED_BETA",
    "F2_3_MIN_R2",
    "F2_3_ALPHA",
    "F2_4_MIN_CONFIDENCE_EFFECT_PCT",
    "F2_4_MIN_BETA_INTERACTION",
    "F2_4_ALPHA",
    "F5_1_MIN_PROPORTION",
    "F5_1_MIN_ALPHA",
    "F5_1_MIN_COHENS_D",
    "F5_2_MIN_PROPORTION",
    "F5_2_MIN_CORRELATION",
    "F5_3_MIN_PROPORTION",
    "F5_3_MIN_GAIN_RATIO",
    "F5_3_MIN_COHENS_D",
    "F5_3_FALSIFICATION_RATIO",
    "F5_4_MIN_PROPORTION",
    "F5_4_MIN_PEAK_SEPARATION",
    "F5_5_PCA_MIN_VARIANCE",
    "F5_5_MIN_LOADING",
    "F5_6_MIN_PERFORMANCE_DIFF_PCT",
    "F5_6_MIN_COHENS_D",
    "F5_6_ALPHA",
    "F6_2_MIN_INTEGRATION_RATIO",
    "F6_2_MIN_CURVE_FIT_R2",
    "F6_1_LTCN_MAX_TRANSITION_MS",
    "F6_1_CLIFFS_DELTA_MIN",
    "F6_1_MANN_WHITNEY_ALPHA",
    "F6_2_LTCN_MIN_WINDOW_MS",
    "F6_5_HYSTERESIS_MIN",
    "F6_5_HYSTERESIS_MAX",
    "F6_2_WILCOXON_ALPHA",
    "V12_1_MIN_P3B_REDUCTION_PCT",
    "V12_1_MIN_IGNITION_REDUCTION_PCT",
    "V12_1_MIN_COHENS_D",
    "V12_1_MIN_ETA_SQUARED",
    "V12_1_ALPHA",
    "V12_2_MIN_CORRELATION",
    "V12_2_FALSIFICATION_CORR",
    "V12_2_MIN_PILLAIS_TRACE",
    "V12_2_FALSIFICATION_PILLAIS",
    "V12_2_ALPHA",
]
