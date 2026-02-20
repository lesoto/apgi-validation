"""
Theme Manager for APGI GUI Applications
=====================================

Provides theme management functionality for tkinter-based GUI applications with support for multiple color schemes including Normal, Dark, and High Contrast themes.

"""

from typing import Dict, List
import tkinter as tk


class ThemeManager:
    """
    Theme manager for APGI GUI applications.

    Supports multiple predefined themes with configurable color schemes
    for background, foreground, and other UI elements.
    """

    def __init__(self, initial_theme: str = "normal"):
        """
        Initialize the theme manager.

        Args:
            initial_theme: Name of the initial theme to apply
        """
        self.current_theme = initial_theme.lower()
        self.themes = self._define_themes()

        # Validate initial theme
        if self.current_theme not in self.themes:
            self.current_theme = "normal"

    def _define_themes(self) -> Dict[str, Dict[str, str]]:
        """
        Define available themes and their color schemes.

        Returns:
            Dictionary mapping theme names to color dictionaries
        """
        return {
            "normal": {
                "bg": "#ffffff",  # White background
                "fg": "#000000",  # Black text
                "select_bg": "#0078d4",  # Windows blue selection
                "select_fg": "#ffffff",  # White text on selection
                "button_bg": "#f0f0f0",  # Light gray button
                "button_fg": "#000000",  # Black button text
                "border": "#cccccc",  # Light gray border
            },
            "dark": {
                "bg": "#2d2d2d",  # Dark gray background
                "fg": "#ffffff",  # White text
                "select_bg": "#404040",  # Darker selection
                "select_fg": "#ffffff",  # White text on selection
                "button_bg": "#404040",  # Dark gray button
                "button_fg": "#ffffff",  # White button text
                "border": "#555555",  # Medium gray border
            },
            "high_contrast": {
                "bg": "#000000",  # Black background
                "fg": "#ffffff",  # White text
                "select_bg": "#ffffff",  # White selection
                "select_fg": "#000000",  # Black text on selection
                "button_bg": "#ffffff",  # White button
                "button_fg": "#000000",  # Black button text
                "border": "#ffffff",  # White border
            },
        }

    def get_available_themes(self) -> List[str]:
        """
        Get list of available theme names.

        Returns:
            List of theme names in lowercase
        """
        return list(self.themes.keys())

    def set_theme(self, theme_name: str) -> bool:
        """
        Set the current theme.

        Args:
            theme_name: Name of the theme to apply

        Returns:
            True if theme was set successfully, False if theme doesn't exist
        """
        theme_name = theme_name.lower()
        if theme_name in self.themes:
            self.current_theme = theme_name
            return True
        return False

    def get_theme_color(self, color_type: str) -> str:
        """
        Get a color value from the current theme.

        Args:
            color_type: Type of color (e.g., 'bg', 'fg', 'select_bg')

        Returns:
            Color hex string for the requested color type
        """
        return self.themes.get(self.current_theme, {}).get(color_type, "#000000")

    def get_current_theme(self) -> str:
        """
        Get the name of the current theme.

        Returns:
            Current theme name
        """
        return self.current_theme

    def apply_theme_to_widget(self, widget: tk.Widget, **overrides):
        """
        Apply the current theme to a tkinter widget.

        Args:
            widget: The tkinter widget to theme
            **overrides: Optional color overrides
        """
        theme_colors = self.themes.get(self.current_theme, {})

        # Build configuration dictionary with theme colors and overrides
        config = {}

        # Map common widget options to theme colors
        color_mappings = {
            "bg": "bg",
            "foreground": "fg",
            "fg": "fg",
            "background": "bg",
            "selectbackground": "select_bg",
            "selectforeground": "select_fg",
            "buttonbackground": "button_bg",
            "activebackground": "select_bg",
            "activeforeground": "select_fg",
        }

        for widget_option, theme_key in color_mappings.items():
            if theme_key in theme_colors:
                config[widget_option] = theme_colors[theme_key]

        # Apply any overrides
        config.update(overrides)

        try:
            widget.config(**config)
        except tk.TclError:
            # Widget doesn't support some configuration options
            pass
