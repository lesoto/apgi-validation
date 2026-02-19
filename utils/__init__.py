"""
APGI Framework Utils Package
============================

Utility modules for the APGI framework.
"""

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
