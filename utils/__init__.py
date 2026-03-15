"""
APGI Framework Utils Package
============================

Utility modules for the APGI framework.
"""

import os
import sys

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, continue with system environment variables


# Security: Check for required environment variables at import time
def _check_required_env_vars():
    """Check for required security environment variables."""
    missing_vars = []

    # Check for PICKLE_SECRET_KEY
    if not os.environ.get("PICKLE_SECRET_KEY"):
        missing_vars.append("PICKLE_SECRET_KEY")

    # Check for APGI_BACKUP_HMAC_KEY
    if not os.environ.get("APGI_BACKUP_HMAC_KEY"):
        missing_vars.append("APGI_BACKUP_HMAC_KEY")

    if missing_vars:
        print("ERROR: Missing required environment variables:", file=sys.stderr)
        for var in missing_vars:
            if var == "PICKLE_SECRET_KEY":
                print(
                    f"  - {var}: Generate with: openssl rand -hex 32", file=sys.stderr
                )
            elif var == "APGI_BACKUP_HMAC_KEY":
                print(
                    f"  - {var}: Generate with: openssl rand -hex 32", file=sys.stderr
                )
        print("Set these variables before importing APGI modules.", file=sys.stderr)
        # Don't raise error here, just warn - let individual modules handle the requirement


_check_required_env_vars()

# Import key classes for easy access
from .sample_data_generator import SampleDataGenerator, generate_sample_multimodal_data
from .data_validation import DataValidator

try:
    from .batch_processor import BatchProcessor

    BATCH_PROCESSOR_AVAILABLE = True
except ImportError as e:
    if "tqdm" in str(e):
        BATCH_PROCESSOR_AVAILABLE = False
        BatchProcessor = None
        print(
            "Warning: BatchProcessor unavailable due to missing tqdm dependency. Install tqdm to enable batch processing."
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
]
