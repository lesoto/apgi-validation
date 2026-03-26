"""
APGI Framework Utils Package
============================

Utility modules for the APGI framework.
"""

import os
import sys
import secrets
import warnings
from typing import Optional

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

# Import key classes for easy access
from .sample_data_generator import SampleDataGenerator, generate_sample_multimodal_data
from .data_validation import DataValidator
from .constants import (
    MODEL_PARAMS,
    NEURAL_DEFAULTS,
    SPECIES_METRICS,
    SYSTEM_DEFAULTS,
    PARAMETER_BOUNDS,
    PCI_NORMALIZATION,
    DIM_CONSTANTS,
)

# Import falsification thresholds from local utils directory
from .falsification_thresholds import (
    F1_1_MIN_ADVANTAGE_PCT,
    F1_1_MIN_COHENS_D,
    F1_1_ALPHA,
    F1_5_PAC_MI_MIN,
    F1_5_PAC_INCREASE_MIN,
    F1_5_COHENS_D_MIN,
    F1_5_PERMUTATION_ALPHA,
    F2_1_MIN_ADVANTAGE_PCT,
    F2_1_MIN_PP_DIFF,
    F2_1_MIN_COHENS_H,
    F2_1_ALPHA,
    F2_2_MIN_CORR,
    F2_2_MIN_FISHER_Z,
    F2_2_ALPHA,
    F2_3_MIN_RT_ADVANTAGE_MS,
    F2_3_MIN_BETA,
    F2_3_MIN_STANDARDIZED_BETA,
    F2_3_MIN_R2,
    F2_3_ALPHA,
    F2_4_MIN_CONFIDENCE_EFFECT_PCT,
    F2_4_MIN_BETA_INTERACTION,
    F2_4_ALPHA,
    F5_1_MIN_PROPORTION,
    F5_1_MIN_ALPHA,
    F5_1_MIN_COHENS_D,
    F5_2_MIN_PROPORTION,
    F5_2_MIN_CORRELATION,
    F5_3_MIN_PROPORTION,
    F5_3_MIN_GAIN_RATIO,
    F5_3_MIN_COHENS_D,
    F5_3_FALSIFICATION_RATIO,
    F5_4_MIN_PROPORTION,
    F5_4_MIN_PEAK_SEPARATION,
    F5_5_PCA_MIN_VARIANCE,
    F5_5_MIN_LOADING,
    F5_6_MIN_PERFORMANCE_DIFF_PCT,
    F5_6_MIN_COHENS_D,
    F5_6_ALPHA,
    F6_1_LTCN_MAX_TRANSITION_MS,
    F6_1_CLIFFS_DELTA_MIN,
    F6_1_MANN_WHITNEY_ALPHA,
    F6_2_LTCN_MIN_WINDOW_MS,
    F6_2_MIN_INTEGRATION_RATIO,
    F6_2_MIN_CURVE_FIT_R2,
    F6_2_WILCOXON_ALPHA,
    # Validation Protocol 12 constants
    V12_1_MIN_P3B_REDUCTION_PCT,
    V12_1_MIN_IGNITION_REDUCTION_PCT,
    V12_1_MIN_COHENS_D,
    V12_1_MIN_ETA_SQUARED,
    V12_1_ALPHA,
    V12_2_MIN_CORRELATION,
    V12_2_FALSIFICATION_CORR,
    V12_2_MIN_PILLAIS_TRACE,
    V12_2_FALSIFICATION_PILLAIS,
    V12_2_ALPHA,
    F6_5_HYSTERESIS_MIN,
    F6_5_HYSTERESIS_MAX,
)

try:
    from .batch_processor import BatchProcessor

    BATCH_PROCESSOR_AVAILABLE = True
except ImportError as e:
    if "tqdm" in str(e):
        BATCH_PROCESSOR_AVAILABLE = False
        BatchProcessor = None
        warnings.warn(
            "Warning: BatchProcessor unavailable due to missing tqdm dependency. Install tqdm to enable batch processing.",
            ImportWarning,
        )
    else:
        raise

from .config_manager import ConfigManager
from .cache_manager import CacheManager

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
