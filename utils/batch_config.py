"""
batch_config.py
===============

Configuration management for the batch processing system.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class BatchProcessorConfig:
    """Configuration manager for batch processing."""

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration.

        Args:
            config_file: Path to YAML configuration file
        """
        self._config: Dict[str, Any] = {
            "max_workers": self._get_default_workers(),
            "use_processes": False,
            "default_output_dir": "results",
            "retry_attempts": 3,
            "timeout": 3600,
        }

        if config_file and config_file.exists():
            self._load_from_file(config_file)

    def _get_default_workers(self) -> int:
        """Get default number of workers based on CPU cores."""
        try:
            return min(os.cpu_count() or 4, 8)
        except Exception:
            return 4

    def _load_from_file(self, config_file: Path) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_file, "r") as f:
                file_config = yaml.safe_load(f)
                if file_config and isinstance(file_config, dict):
                    self._config.update(file_config)
        except (IOError, yaml.YAMLError):
            # Use defaults if file can't be loaded
            pass

    def get_max_workers(self) -> int:
        """Get maximum number of parallel workers."""
        return self._config.get("max_workers", self._get_default_workers())

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
