"""APGI Design System — shared theme constants, styles, and UI components."""

import logging
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)

# ── Color Palette (Lab System) ────────────────────────────────────────────────
COLORS = {
    "primary": "#2166AC",
    "success": "#41AB5D",
    "alert": "#D6604D",
    "background": "#f8f9fa",
    "surface": "#ffffff",
    "border": "#dee2e6",
    "text_primary": "#212529",
    "text_secondary": "#6c757d",
    "text_muted": "#adb5bd",
    "soft_gray": "#e9ecef",
    "info": "#856404",
}


def _resolve_font(candidates: list, fallback: str) -> str:
    """Return the first available font from candidates, else fallback."""
    try:
        import tkinter as _tk
        import tkinter.font as _tkfont

        _root = _tk.Tk()
        _root.withdraw()
        available = set(_tkfont.families())
        _root.destroy()
        for name in candidates:
            if name in available:
                return name
    except (tk.TclError, RuntimeError, ImportError) as exc:
        logger.debug("Font resolution failed: %s", exc)
    return fallback


FONTS = {
    "primary": _resolve_font(["Noto Sans", "Segoe UI", "Helvetica Neue", "Arial"], "TkDefaultFont"),
    "monospace": _resolve_font(["Noto Sans Mono", "Consolas", "Menlo", "Courier New"], "TkFixedFont"),
    "academic": _resolve_font(["Noto Serif", "Georgia", "Times New Roman"], "TkTextFont"),
}


def apply_apgi_theme(root):
    """Apply unified APGI theme to tkinter application."""
    style = ttk.Style()
    style.theme_use("clam")

    bg_color = COLORS["background"]
    fg_color = COLORS["text_primary"]

    style.configure("TFrame", background=bg_color)
    style.configure("TLabel", background=bg_color, foreground=fg_color, font=(FONTS["primary"], 10))
    style.configure("Header.TLabel", font=(FONTS["primary"], 12, "bold"))
    style.configure("Title.TLabel", font=(FONTS["primary"], 16, "bold"))
    style.configure(
        "Subtitle.TLabel",
        font=(FONTS["primary"], 10),
        foreground=COLORS["text_secondary"],
    )
    style.configure(
        "Monospace.TLabel",
        font=(FONTS["monospace"], 11),
        foreground=COLORS["text_primary"],
    )
    style.configure(
        "Card.TFrame",
        background=COLORS["surface"],
        borderwidth=1,
        relief="solid",
    )
    style.configure(
        "Metric.TLabelframe",
        background=COLORS["surface"],
        borderwidth=1,
        relief="solid",
    )
    style.configure(
        "Metric.TLabelframe.Label",
        background=COLORS["surface"],
        foreground=COLORS["text_secondary"],
        font=(FONTS["primary"], 9),
    )
    style.configure("TButton", padding=6, background=COLORS["soft_gray"])
    style.map(
        "TButton",
        background=[("active", COLORS["border"])],
    )
    style.configure(
        "Primary.TButton",
        background=COLORS["success"],
        foreground="white",
        font=(FONTS["primary"], 10, "bold"),
        padding=8,
    )
    style.map(
        "Primary.TButton",
        background=[("active", "#2d7a3d")],
        foreground=[("active", "white")],
    )
    style.configure(
        "Secondary.TButton",
        background=COLORS["primary"],
        foreground="white",
        font=(FONTS["primary"], 10),
        padding=6,
    )
    style.map(
        "Secondary.TButton",
        background=[("active", "#1f5a82")],
        foreground=[("active", "white")],
    )
    style.configure(
        "Danger.TButton",
        background=COLORS["alert"],
        foreground="white",
        font=(FONTS["primary"], 10, "bold"),
        padding=8,
    )
    style.map(
        "Danger.TButton",
        background=[("active", "#5a161d")],
        foreground=[("active", "white")],
    )
    style.configure("Card.TCheckbutton", background=COLORS["surface"])
    style.configure(
        "TNotebook",
        background=bg_color,
        tabmargins=[2, 5, 2, 0],
    )
    style.configure(
        "TNotebook.Tab",
        font=(FONTS["primary"], 10),
        padding=[10, 5],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", COLORS["surface"])],
        expand=[("selected", [1, 1, 1, 0])],
    )
    style.configure(
        "TProgressbar",
        background=COLORS["primary"],
        troughcolor=COLORS["soft_gray"],
        borderwidth=0,
    )
    style.configure(
        "Status.TFrame",
        background=COLORS["surface"],
        borderwidth=1,
        relief="solid",
    )
    root.configure(background=bg_color)
    return style


class APGICard(ttk.Frame):
    """Standardized information card for all APGI apps."""

    def __init__(self, parent, title, value, intervention="", **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)
        container = ttk.Frame(self, padding=15, style="Card.TFrame")
        container.pack(fill="both", expand=True)

        self.lbl_title = ttk.Label(container, text=title.upper(), style="Header.TLabel")
        self.lbl_title.pack(anchor="w")

        self.lbl_value = ttk.Label(container, text=value, font=(FONTS["monospace"], 14))
        self.lbl_value.pack(anchor="w", pady=(5, 10))

        if intervention:
            ttk.Separator(container, orient="horizontal").pack(fill="x", pady=5)
            self.lbl_hint = ttk.Label(
                container,
                text=f"Intervention: {intervention}",
                wraplength=250,
                foreground="#495057",
                font=(FONTS["primary"], 9, "italic"),
            )
            self.lbl_hint.pack(anchor="w")


class APGIButtons:
    """Standard button configurations for APGI applications."""

    @staticmethod
    def primary(parent, text, command):
        return ttk.Button(parent, text=text, command=command, style="Primary.TButton", cursor="hand2")

    @staticmethod
    def danger(parent, text, command):
        return ttk.Button(parent, text=text, command=command, style="Danger.TButton", cursor="hand2")

    @staticmethod
    def secondary(parent, text, command):
        return ttk.Button(
            parent,
            text=text,
            command=command,
            style="Secondary.TButton",
            cursor="hand2",
        )

    @staticmethod
    def standard(parent, text, command):
        return ttk.Button(parent, text=text, command=command, cursor="hand2")


def show_status(parent, status_type, message):
    """Create a status indicator with icon and color (WCAG compliant)."""
    icons = {"success": "[OK]", "error": "[X]", "warning": "[!]", "info": "[i]"}
    colors = {
        "success": COLORS["success"],
        "error": COLORS["alert"],
        "warning": COLORS["info"],
        "info": COLORS["primary"],
    }
    return ttk.Label(
        parent,
        text=f"{icons[status_type]} {message}",
        foreground=colors[status_type],
        font=(FONTS["primary"], 10, "bold"),
    )


def create_empty_state(parent, message):
    """Create empty state placeholder for data views."""
    frame = ttk.Frame(parent, padding=40)
    frame.pack(expand=True)
    canvas = tk.Canvas(
        frame,
        width=200,
        height=120,
        bg=COLORS["background"],
        highlightbackground=COLORS["soft_gray"],
        highlightthickness=2,
    )
    canvas.pack()
    ttk.Label(
        frame,
        text=message,
        wraplength=300,
        font=(FONTS["primary"], 11),
        foreground=COLORS["text_secondary"],
    ).pack(pady=(20, 0))
    return frame
