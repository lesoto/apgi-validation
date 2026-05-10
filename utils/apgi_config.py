"""
LEVEL DESIGNATION: Level 2 (information-theoretic)

Bridge to Level 1
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.apgi_settings import APGISettings, resolve_apgi_config_path  # noqa: E402

_settings_lock = threading.Lock()
_settings_singleton: Optional[APGISettings] = None


def get_apgi_settings(path: Optional[Path] = None, *, reload: bool = False) -> APGISettings:
    """Return the process-wide APGI settings instance.

    Args:
        path: Optional YAML path to load settings from.
        reload: When True, forces a reload from disk/env.
    """
    global _settings_singleton
    with _settings_lock:
        if _settings_singleton is None or reload:
            resolved = resolve_apgi_config_path(path)
            _settings_singleton = APGISettings.from_yaml(resolved)
        return _settings_singleton


# Backwards compatibility: allow `APGIConfig()` while migrating call sites.
APGIConfig = APGISettings
