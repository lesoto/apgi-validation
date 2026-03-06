#!/usr/bin/env python3
"""
Batch Processor Configuration
===========================

Configuration file for parallel processing settings in the APGI framework.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


# Default configuration
DEFAULT_CONFIG = {
    "max_workers": {
        "default": 4,
        "min": 1,
        "max": os.cpu_count() or 8,
        "description": "Maximum number of parallel workers",
    },
    "use_processes": {
        "default": False,
        "description": "Use processes instead of threads for CPU-bound tasks",
    },
    "chunk_size": {
        "default": 100,
        "min": 10,
        "max": 1000,
        "description": "Chunk size for batch processing",
    },
    "timeout": {
        "default": 300,  # 5 minutes
        "min": 60,  # 1 minute
        "max": 3600,  # 1 hour
        "description": "Timeout for individual jobs in seconds",
    },
    "auto_scale": {
        "default": True,
        "description": "Automatically scale workers based on CPU load",
    },
}


class BatchProcessorConfig:
    """Configuration manager for batch processing."""

    def __init__(self, config_file: Optional[Path] = None):
        """Initialize configuration manager.

        Args:
            config_file: Path to configuration file. If None, uses default config.
        """
        self.config_file = config_file or Path("config/batch_config.json")
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()

    def load_config(self):
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                import json

                with open(self.config_file, "r") as f:
                    file_config = json.load(f)

                # Merge with defaults, validating ranges
                for key, value in file_config.items():
                    if key in self.config:
                        if (
                            isinstance(self.config[key], dict)
                            and "default" in self.config[key]
                        ):
                            # Validate range if min/max specified
                            if (
                                "min" in self.config[key]
                                and value < self.config[key]["min"]
                            ):
                                value = self.config[key]["min"]
                            if (
                                "max" in self.config[key]
                                and value > self.config[key]["max"]
                            ):
                                value = self.config[key]["max"]

                        self.config[key]["default"] = value
                    else:
                        self.config[key] = {"default": value}

            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load batch config: {e}")

    def save_config(self):
        """Save current configuration to file."""
        try:
            import json

            # Ensure config directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            # Extract default values for saving
            save_config = {}
            for key, value in self.config.items():
                if isinstance(value, dict) and "default" in value:
                    save_config[key] = value["default"]
                else:
                    save_config[key] = value

            with open(self.config_file, "w") as f:
                json.dump(save_config, f, indent=2)

        except IOError as e:
            print(f"Warning: Could not save batch config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if key in self.config:
            if isinstance(self.config[key], dict) and "default" in self.config[key]:
                return self.config[key]["default"]
            return self.config[key]
        return default

    def set(self, key: str, value: Any):
        """Set configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        if key in self.config:
            if isinstance(self.config[key], dict):
                # Validate range if min/max specified
                if "min" in self.config[key] and value < self.config[key]["min"]:
                    raise ValueError(
                        f"Value {value} is below minimum {self.config[key]['min']}"
                    )
                if "max" in self.config[key] and value > self.config[key]["max"]:
                    raise ValueError(
                        f"Value {value} is above maximum {self.config[key]['max']}"
                    )

                self.config[key]["default"] = value
            else:
                self.config[key] = value
        else:
            self.config[key] = {"default": value}

    def get_max_workers(self) -> int:
        """Get optimal number of workers based on configuration and system."""
        max_workers = self.get("max_workers", 4)

        if self.get("auto_scale", True):
            # Scale based on CPU count
            cpu_count = os.cpu_count() or 4
            max_workers = min(max_workers, cpu_count)

            # Reduce workers if system is under load (simple heuristic)
            try:
                import psutil

                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > 80:
                    max_workers = max(1, max_workers // 2)
            except ImportError:
                pass  # psutil not available, use configured value

        return max_workers

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary."""
        result = {}
        for key, value in self.config.items():
            if isinstance(value, dict) and "default" in value:
                result[key] = value["default"]
                # Add metadata
                for meta_key, meta_value in value.items():
                    if meta_key != "default":
                        result[f"{key}_{meta_key}"] = meta_value
            else:
                result[key] = value
        return result
