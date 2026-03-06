#!/usr/bin/env python3
"""
APGI Theme Manager
=================

Manages UI themes for APGI applications.
"""

from typing import Dict, List, Optional
from pathlib import Path


class ThemeManager:
    """Manages UI themes for APGI applications"""

    def __init__(
        self, themes_dir: Optional[Path] = None, initial_theme: str = "default"
    ):
        self.themes_dir = themes_dir or Path(__file__).parent / "themes"
        self._themes = self._load_themes()
        self.current_theme = (
            initial_theme if initial_theme in self._themes else "default"
        )

    def _load_themes(self) -> Dict[str, Dict]:
        """Load available themes from themes directory"""
        themes = {}

        # Default theme
        themes["default"] = {
            "background": "#ffffff",
            "foreground": "#000000",
            "button": "#e0e0e0",
            "highlight": "#007acc",
            "text": "#333333",
        }

        # Normal theme (alias for default)
        themes["normal"] = themes["default"]

        # Dark theme
        themes["dark"] = {
            "background": "#2b2b2b",
            "foreground": "#ffffff",
            "button": "#404040",
            "highlight": "#0055aa",
            "text": "#cccccc",
        }

        return themes

    def get_available_themes(self) -> List[str]:
        """Get list of available theme names"""
        return list(self._themes.keys())

    def get_theme(self, theme_name: str) -> Dict:
        """Get theme configuration by name"""
        return self._themes.get(theme_name, self._themes["default"])

    def set_theme(self, theme_name: str) -> bool:
        """Set current theme"""
        if theme_name in self._themes:
            self.current_theme = theme_name
            return True
        return False

    def get_current_theme(self) -> Dict:
        """Get current theme configuration"""
        return self._themes[self.current_theme]
