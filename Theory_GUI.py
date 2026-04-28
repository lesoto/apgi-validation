#!/usr/bin/env python3
"""
APGI Theory GUI - Redesigned with APGI Design Guide

Scientific Instrument aesthetic with standardized APGI components.
"""

import os
import sys

os.environ["MPLBACKEND"] = "Agg"
os.environ["MATPLOTLIB_BACKEND"] = "Agg"
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import matplotlib

matplotlib.use("Agg", force=True)

_original_use = matplotlib.use
_original_switch_backend = getattr(matplotlib, "switch_backend", None)


def _locked_use(backend, *args, **kwargs):
    """Prevent backend switching to GUI backends"""
    if backend.lower() not in ("agg", "svg", "pdf", "ps", "pgf", "cairo", "inline"):
        import warnings

        warnings.warn(f"Blocking backend switch to {backend}")
        return
    return _original_use(backend, *args, **kwargs)


matplotlib.use = _locked_use

import matplotlib.pyplot as plt

plt.switch_backend("Agg")

import importlib.util
import logging
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import List

try:
    import torch

    TORCH_AVAILABLE = True
    _ = torch.__version__

except ImportError:
    TORCH_AVAILABLE = False

import tempfile

os.environ["TMPDIR"] = tempfile.gettempdir()
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

cache_dirs = [
    os.path.join(tempfile.gettempdir(), "matplotlib_cache"),
    os.path.expanduser("~/.cache/matplotlib"),
]
for cache_dir in cache_dirs:
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        pass

# Suppress lifelines and pandas FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="lifelines")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# APGI Design System - Theme and Components
# ============================================================================

# Color Palette (Lab System)
COLORS = {
    "primary": "#2874a6",
    "success": "#155724",
    "alert": "#721c24",
    "background": "#f8f9fa",
    "surface": "#ffffff",
    "border": "#dee2e6",
    "text_primary": "#212529",
    "text_secondary": "#6c757d",
    "text_muted": "#adb5bd",
    "soft_gray": "#e9ecef",
    "info": "#856404",
}

# Typography
FONTS = {
    "primary": "Noto Sans",
    "monospace": "Noto Sans Mono",
    "academic": "Noto Serif",
}


def apply_apgi_theme(root):
    """Apply unified APGI theme to tkinter application."""
    style = ttk.Style()
    style.theme_use("clam")

    # Core Palette
    bg_color = COLORS["background"]
    fg_color = COLORS["text_primary"]

    # Configure Global Elements
    style.configure("TFrame", background=bg_color)
    style.configure(
        "TLabel", background=bg_color, foreground=fg_color, font=(FONTS["primary"], 10)
    )
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

    # Custom Card Style
    style.configure(
        "Card.TFrame",
        background=COLORS["surface"],
        borderwidth=1,
        relief="solid",
    )

    # Labeled Frame (Metric Card Style)
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

    # Button Styling
    style.configure("TButton", padding=6, background=COLORS["soft_gray"])
    style.map(
        "TButton",
        background=[("active", "#dee2e6"), ("disabled", "#f1f3f5")],
        foreground=[("disabled", "#adb5bd")],
    )

    # Primary Button (Success - Green)
    style.configure(
        "Primary.TButton",
        background=COLORS["success"],
        foreground="white",
        font=(FONTS["primary"], 10, "bold"),
        padding=8,
    )
    style.map(
        "Primary.TButton",
        background=[("active", "#0f3d1a")],
        foreground=[("active", "white")],
    )

    # Secondary Button (Primary Blue)
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

    # Danger Button (Alert - Red)
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

    # Checkbutton
    style.configure("Card.TCheckbutton", background=COLORS["surface"])

    # Notebook (Tab) Styling
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

    # Progress Bar
    style.configure(
        "TProgressbar",
        background=COLORS["primary"],
        troughcolor=COLORS["soft_gray"],
        borderwidth=0,
    )

    # Status Bar
    style.configure(
        "Status.TFrame",
        background=COLORS["surface"],
        borderwidth=1,
        relief="solid",
    )

    # Configure root window
    root.configure(background=bg_color)

    return style


class APGICard(ttk.Frame):
    """Standardized information card for all APGI apps.

    Features:
    - Title with uppercase styling
    - Monospace value display
    - Optional intervention hint (The Intervention Rule)
    - Consistent padding and borders
    """

    def __init__(self, parent, title, value, intervention="", **kwargs):
        # Use custom style 'Card.TFrame' defined in apply_apgi_theme
        super().__init__(parent, style="Card.TFrame", **kwargs)

        # Internal padding via sub-frame
        container = ttk.Frame(self, padding=15, style="Card.TFrame")
        container.pack(fill="both", expand=True)

        # Title (uppercase per lab convention)
        self.lbl_title = ttk.Label(container, text=title.upper(), style="Header.TLabel")
        self.lbl_title.pack(anchor="w")

        # Value (monospace for scientific precision)
        self.lbl_value = ttk.Label(container, text=value, font=(FONTS["monospace"], 14))
        self.lbl_value.pack(anchor="w", pady=(5, 10))

        # Intervention Rule (MANDATORY if displaying deficits)
        if intervention:
            separator = ttk.Separator(container, orient="horizontal")
            separator.pack(fill="x", pady=5)

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
        """Primary action button (green)."""
        return ttk.Button(
            parent,
            text=text,
            command=command,
            style="Primary.TButton",
            cursor="hand2",
        )

    @staticmethod
    def danger(parent, text, command):
        """Danger/Stop button (red)."""
        return ttk.Button(
            parent,
            text=text,
            command=command,
            style="Danger.TButton",
            cursor="hand2",
        )

    @staticmethod
    def secondary(parent, text, command):
        """Secondary action button (blue)."""
        return ttk.Button(
            parent,
            text=text,
            command=command,
            style="Secondary.TButton",
            cursor="hand2",
        )

    @staticmethod
    def standard(parent, text, command):
        """Standard action button."""
        return ttk.Button(parent, text=text, command=command, cursor="hand2")


def show_status(parent, status_type, message):
    """Create a status indicator with icon and color (WCAG compliant)."""
    icons = {
        "success": "[OK]",
        "error": "[X]",
        "warning": "[!]",
        "info": "[i]",
    }
    colors = {
        "success": COLORS["success"],
        "error": COLORS["alert"],
        "warning": COLORS["info"],
        "info": COLORS["primary"],
    }

    label = ttk.Label(
        parent,
        text=f"{icons[status_type]} {message}",
        foreground=colors[status_type],
        font=(FONTS["primary"], 10, "bold"),
    )
    return label


def create_empty_state(parent, message):
    """Create empty state placeholder for data views."""
    frame = ttk.Frame(parent, padding=40)
    frame.pack(expand=True)

    # Soft gray outline
    canvas = tk.Canvas(
        frame,
        width=200,
        height=120,
        bg=COLORS["background"],
        highlightbackground=COLORS["soft_gray"],
        highlightthickness=2,
    )
    canvas.pack()

    # Message
    label = ttk.Label(
        frame,
        text=message,
        wraplength=300,
        font=(FONTS["primary"], 11),
        foreground=COLORS["text_secondary"],
    )
    label.pack(pady=(20, 0))

    return frame


# ============================================================================
# Main GUI Class
# ============================================================================


class ScriptRunnerGUI:
    """APGI Theory Framework Runner - Scientific Instrument Interface."""

    def __init__(self, root):
        self.root = root
        self.root.title("APGI Theory Framework - Scientific Instrument Interface")
        self.root.geometry("1200x850")
        self.root.minsize(900, 650)

        # Apply APGI Theme
        self.style = apply_apgi_theme(root)

        # Add project root to Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Discover all scripts from Theory folder
        theory_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Theory")
        self.protocols = self._discover_protocols(theory_dir)

        # Thread management
        self.stop_event = threading.Event()
        self.running_thread = None

        # Parameter storage
        self.parameter_values = {}
        self.load_default_parameters()

        # Initialize selected protocol
        self.selected_protocol = None

        # Add window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Build UI
        self.setup_ui()
        self.log_message(
            f"Instrument initialized. Loaded {len(self.protocols)} theory scripts."
        )

    def _discover_protocols(self, theory_dir):
        """Discover all Python scripts in Theory folder and introspect their capabilities."""
        protocols = {}

        if not os.path.exists(theory_dir):
            self.log_message(f"Theory directory not found: {theory_dir}")
            return protocols

        for file in sorted(os.listdir(theory_dir)):
            if (
                not file.endswith(".py")
                or file.startswith("__")
                or file == os.path.basename(__file__)
            ):
                continue

            protocol_name = file.replace(".py", "")
            file_path = os.path.join(theory_dir, file)

            try:
                # Load module spec without executing
                spec = importlib.util.spec_from_file_location(protocol_name, file_path)
                if spec is None or spec.loader is None:
                    continue

                # Analyze module without full execution (check source for key patterns)
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()

                # Find classes and their methods
                import ast

                tree = ast.parse(source)

                runnable_classes = []
                module_level_runners = []
                has_main_block = False

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = node.name
                        methods = [
                            n.name for n in node.body if isinstance(n, ast.FunctionDef)
                        ]

                        # Skip GUI classes - they require tkinter root and can't be
                        # instantiated by the Theory GUI
                        if class_name.endswith("GUI"):
                            continue

                        # Check for runnable methods
                        if any(
                            m in methods
                            for m in [
                                "run_validation",
                                "run_falsification",
                                "run_full_experiment",
                                "run_analysis",
                            ]
                        ):
                            runnable_classes.append(
                                {"name": class_name, "methods": methods}
                            )

                    elif isinstance(node, ast.FunctionDef) and node.name in [
                        "run_validation",
                        "run_falsification",
                        "validate_cross_species_model",
                        "validate_cultural_modulation_effects",
                        "generate_cross_cultural_comparison",
                        "run_complete_demo",
                        "run_baseline_simulation",
                        "run_anxiety_comparison",
                    ]:
                        module_level_runners.append(node.name)
                    elif isinstance(node, ast.FunctionDef) and node.name == "main":
                        # Check if main() launches a GUI (tkinter) - if so, skip it
                        func_source = ast.get_source_segment(source, node)
                        if func_source and not (
                            "tkinter" in func_source
                            and ("Tk()" in func_source or "tk.Tk()" in func_source)
                        ):
                            module_level_runners.append(node.name)
                    elif isinstance(node, ast.FunctionDef):
                        # Catch any top-level run_* or validate_* function
                        # EXCEPT those that require specific arguments that GUI can't provide
                        if node.name.startswith(
                            ("run_", "validate_")
                        ) and not node.name.startswith("_"):
                            # Skip functions that require specific data arguments
                            skip_functions = [
                                "validate_joint_biomarker_advantage",
                                "validate_transition_plausibility",
                                "validate_ode_integration",
                                "validate_cross_level_consistency",
                                "validate_precision_surprise_relationship",
                                "validate_three_level_entropy",
                                "validate_information_gain_positive",
                                "validate_phase_transition",
                                "validate_metabolic_cost_scales",
                                "validate_cost_benefit_gating",
                                "validate_gradient_flow",
                                "validate_dataset_compliance",
                                "validate_replication_attempt",
                                "validate_open_science_compliance",
                                "validate_measurement",
                                "validate_implementation",
                                "run_analysis",
                                "stop_analysis",
                            ]
                            if node.name not in skip_functions:
                                module_level_runners.append(node.name)
                    elif isinstance(node, ast.If):
                        # Detect if __name__ == "__main__" blocks
                        test = node.test
                        if isinstance(test, ast.Compare):
                            if (
                                isinstance(test.left, ast.Name)
                                and test.left.id == "__name__"
                            ):
                                for comparator in test.comparators:
                                    if (
                                        isinstance(comparator, ast.Constant)
                                        and comparator.value == "__main__"
                                    ):
                                        has_main_block = True

                # Determine execution strategy
                exec_info = self._determine_execution_strategy(
                    protocol_name,
                    runnable_classes,
                    module_level_runners,
                    has_main_block,
                )

                if exec_info:
                    # Get docstring for description
                    try:
                        docstring = (
                            ast.get_docstring(tree) or f"Theory script: {protocol_name}"
                        )
                        description = docstring.split("\n")[0][:100]
                    except Exception:
                        description = f"Theory script: {protocol_name}"

                    # Generate display name
                    display_name = self._format_display_name(protocol_name)

                    protocols[display_name] = {
                        "file": file,
                        "file_path": file_path,
                        "module_name": protocol_name,
                        "description": description,
                        "execution_info": exec_info,
                        "parameters": self._infer_parameters(source, exec_info),
                    }

            except Exception as e:
                logger.warning(f"Error analyzing {file}: {e}")
                continue

        return protocols

    def _determine_execution_strategy(
        self,
        protocol_name,
        runnable_classes,
        module_level_runners,
        has_main_block=False,
    ):
        """Determine how to execute a protocol based on its structure."""

        # Priority: module-level runners
        for func in ["run_falsification", "run_validation", "main"]:
            if func in module_level_runners:
                return {"type": "module_function", "function": func}

        # Other module-level runners
        for func in module_level_runners:
            return {"type": "module_function", "function": func}

        # Then class-based runners
        for cls in runnable_classes:
            for method in [
                "run_falsification",
                "run_validation",
                "run_full_experiment",
                "run_analysis",
            ]:
                if method in cls["methods"]:
                    return {
                        "type": "class_method",
                        "class": cls["name"],
                        "method": method,
                    }

        # Default: just run the module (if it has executable code at module level)
        # Note: if __name__ == "__main__" blocks won't execute when loaded via importlib
        return {"type": "exec_module", "has_main_block": has_main_block}

    def _format_display_name(self, protocol_name):
        """Convert filename to human-readable display name."""
        # Remove APGI_ prefix and convert underscores to spaces
        name = protocol_name.replace("APGI_", "").replace("_", " ")
        # Title case
        return name.title()

    def _infer_parameters(self, source, exec_info):
        """Infer configurable parameters from source code."""
        parameters = {}

        # Look for common parameter patterns in source
        import re

        # Pattern for n_samples, n_trials, etc.
        int_params = re.findall(r"(n_\w+|\w+_size|\w+_count)\s*=\s*(\d+)", source)
        for param_name, default_val in int_params:
            if param_name not in parameters:
                parameters[param_name] = {
                    "default": int(default_val),
                    "min": 1,
                    "max": max(int(default_val) * 10, 10000),
                    "type": "int",
                    "description": f"Number of {param_name.replace('n_', '').replace('_', ' ')}",
                }

        # Pattern for learning rates
        lr_params = re.findall(r"(lr_\w+|learning_rate)\s*=\s*([\d.]+)", source)
        for param_name, default_val in lr_params:
            if param_name not in parameters:
                parameters[param_name] = {
                    "default": float(default_val),
                    "min": 0.001,
                    "max": 1.0,
                    "type": "float",
                    "description": f"Learning rate for {param_name.replace('lr_', '').replace('_', ' ')}",
                }

        # Pattern for thresholds
        theta_params = re.findall(
            r"(theta_\w+|threshold_\w+|alpha|beta|rho)\s*=\s*([\d.]+)", source
        )
        for param_name, default_val in theta_params:
            if param_name not in parameters:
                parameters[param_name] = {
                    "default": float(default_val),
                    "min": 0.01,
                    "max": max(float(default_val) * 5, 100.0),
                    "type": "float",
                    "description": f"Parameter {param_name}",
                }

        # Pattern for dimensions
        dim_params = re.findall(r"(\w+_dim)\s*=\s*(\d+)", source)
        for param_name, default_val in dim_params:
            if param_name not in parameters:
                parameters[param_name] = {
                    "default": int(default_val),
                    "min": 1,
                    "max": max(int(default_val) * 4, 512),
                    "type": "int",
                    "description": f"Dimension for {param_name.replace('_dim', '').replace('_', ' ')}",
                }

        return parameters

    def _on_closing(self):
        """Handle window close event with proper thread cleanup."""
        # Stop the running thread if it exists
        if self.running_thread and self.running_thread.is_alive():
            self.stop_event.set()
            # Join thread with timeout to prevent blocking
            self.running_thread.join(timeout=2.0)
            if self.running_thread.is_alive():
                logging.warning("Thread did not stop cleanly on exit")
        self.root.destroy()

    # ============================================================================
    # UI Layout - Scientific Instrument Architecture
    # ============================================================================

    def setup_ui(self):
        """Setup the scientific instrument interface."""

        # Configure grid weights for main layout
        self.root.grid_columnconfigure(0, weight=0, minsize=220)  # Sidebar
        self.root.grid_columnconfigure(1, weight=1)  # Workspace
        self.root.grid_rowconfigure(0, weight=0)  # Top metric bar
        self.root.grid_rowconfigure(1, weight=1)  # Main content
        self.root.grid_rowconfigure(2, weight=0, minsize=180)  # Console

        # Top Metric Bar ($B_t$, epoch, connection status)
        self._create_metric_bar()

        # Left Sidebar (Script Selection)
        self._create_sidebar()

        # Main Workspace (Tabs: Scripts, Parameters)
        self._create_workspace()

        # Bottom Console (Log output)
        self._create_console()

        # Configure tab order for keyboard navigation
        self._setup_tab_order()

    def _create_metric_bar(self):
        """Create top metric bar with system status indicators."""
        metric_bar = ttk.Frame(self.root, padding=(15, 8))
        metric_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        metric_bar.configure(style="TFrame")

        # System Status Card
        status_card = ttk.LabelFrame(
            metric_bar,
            text="SYSTEM STATUS",
            padding=(10, 5),
            style="Metric.TLabelframe",
        )
        status_card.pack(side=tk.LEFT, padx=(0, 15))

        self.system_status_label = ttk.Label(
            status_card,
            text="[OK] Ready",
            font=(FONTS["monospace"], 11),
            foreground=COLORS["success"],
        )
        self.system_status_label.pack(side=tk.LEFT)

        # Scripts Loaded Card
        scripts_card = ttk.LabelFrame(
            metric_bar,
            text="ACTIVE SCRIPTS",
            padding=(10, 5),
            style="Metric.TLabelframe",
        )
        scripts_card.pack(side=tk.LEFT, padx=(0, 15))

        self.scripts_count_label = ttk.Label(
            scripts_card,
            text=str(len(self.protocols)),
            font=(FONTS["monospace"], 11),
            foreground=COLORS["primary"],
        )
        self.scripts_count_label.pack(side=tk.LEFT)

        # Platform Card
        platform_card = ttk.LabelFrame(
            metric_bar, text="PLATFORM", padding=(10, 5), style="Metric.TLabelframe"
        )
        platform_card.pack(side=tk.LEFT)

        import platform

        platform_text = f"{platform.system()} {platform.machine()}"
        platform_label = ttk.Label(
            platform_card,
            text=platform_text,
            font=(FONTS["monospace"], 10),
            foreground=COLORS["text_secondary"],
        )
        platform_label.pack(side=tk.LEFT)

    def _create_sidebar(self):
        """Create left sidebar with script selection."""
        sidebar = ttk.Frame(self.root, padding=15)
        sidebar.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        sidebar.configure(style="TFrame")

        # Sidebar Title
        sidebar_title = ttk.Label(sidebar, text="SCRIPT LIBRARY", style="Header.TLabel")
        sidebar_title.pack(anchor="w", pady=(0, 15))

        # Script count subtitle
        sidebar_subtitle = ttk.Label(
            sidebar,
            text=f"{len(self.protocols)} Theory Modules Available",
            style="Subtitle.TLabel",
        )
        sidebar_subtitle.pack(anchor="w", pady=(0, 10))

        # Scrollable script list
        script_canvas = tk.Canvas(
            sidebar, background=COLORS["background"], highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(
            sidebar, orient="vertical", command=script_canvas.yview
        )
        self.script_list_frame = ttk.Frame(script_canvas, style="TFrame")

        self.script_list_frame.bind(
            "<Configure>",
            lambda e: script_canvas.configure(scrollregion=script_canvas.bbox("all")),
        )

        script_canvas.create_window((0, 0), window=self.script_list_frame, anchor="nw")
        script_canvas.configure(yscrollcommand=scrollbar.set)

        # Populate script buttons
        self.script_buttons = {}
        for i, (protocol_name, protocol_info) in enumerate(self.protocols.items()):
            btn_frame = ttk.Frame(self.script_list_frame, style="Card.TFrame")
            btn_frame.pack(fill="x", pady=(0, 8), padx=2)

            btn = ttk.Button(
                btn_frame,
                text=protocol_name,
                command=lambda info=protocol_info, name=protocol_name: self.select_protocol(
                    name, info
                ),
            )
            btn.pack(fill="x", padx=8, pady=8)

            self.script_buttons[protocol_name] = btn

            # Add tooltip
            self.create_tooltip(btn, protocol_info["description"])

        script_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _create_workspace(self):
        """Create main workspace area with notebook tabs."""
        workspace = ttk.Frame(self.root, padding=15)
        workspace.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.W, tk.E))
        workspace.configure(style="TFrame")

        # Notebook for tabs
        self.notebook = ttk.Notebook(workspace)
        self.notebook.pack(fill="both", expand=True)

        # Scripts Tab
        scripts_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(scripts_tab, text="Protocols")
        self._setup_protocols_tab(scripts_tab)

        # Parameters Tab
        params_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(params_tab, text="Parameters")
        self._setup_parameters_tab(params_tab)

    def _setup_protocols_tab(self, parent_frame):
        """Setup the protocols selection tab."""
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(1, weight=1)

        # Selected Protocol Display Card
        selected_card = ttk.LabelFrame(
            parent_frame,
            text="SELECTED PROTOCOL",
            padding=15,
            style="Metric.TLabelframe",
        )
        selected_card.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        self.selected_protocol_label = ttk.Label(
            selected_card,
            text="No protocol selected",
            font=(FONTS["primary"], 12, "bold"),
            foreground=COLORS["text_secondary"],
        )
        self.selected_protocol_label.pack(anchor="w")

        # Script description label
        self.selected_protocol_desc = ttk.Label(
            selected_card,
            text="Select a theory script from the sidebar to begin",
            wraplength=500,
            foreground=COLORS["text_secondary"],
        )
        self.selected_protocol_desc.pack(anchor="w", pady=(5, 0))

        # Control Buttons Frame
        control_frame = ttk.Frame(parent_frame)
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        # Run selected button
        self.run_selected_button = APGIButtons.primary(
            control_frame, "Run Selected", self.run_selected_protocol
        )
        self.run_selected_button.pack(side=tk.LEFT, padx=(0, 10))
        self.run_selected_button.config(state=tk.DISABLED)

        # Run all button
        self.run_all_button = APGIButtons.secondary(
            control_frame, "Run All Scripts", self.run_all_protocols
        )
        self.run_all_button.pack(side=tk.LEFT)

        # Quick Stats Frame
        stats_frame = ttk.LabelFrame(
            parent_frame,
            text="QUICK STATISTICS",
            padding=15,
            style="Metric.TLabelframe",
        )
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create stat cards in grid
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(1, weight=1)
        stats_frame.columnconfigure(2, weight=1)

        # Total Scripts
        self.stat_total_card = APGICard(
            stats_frame, "Total Scripts", str(len(self.protocols))
        )
        self.stat_total_card.grid(
            row=0, column=0, padx=(0, 10), pady=(0, 10), sticky="nsew"
        )

        # Configurable Scripts (those with parameters)
        configurable_count = sum(
            1 for p in self.protocols.values() if p.get("parameters")
        )
        self.stat_config_card = APGICard(
            stats_frame, "Configurable", str(configurable_count)
        )
        self.stat_config_card.grid(
            row=0, column=1, padx=(0, 10), pady=(0, 10), sticky="nsew"
        )

        # Ready Status
        self.stat_ready_card = APGICard(stats_frame, "Status", "Ready")
        self.stat_ready_card.grid(row=0, column=2, pady=(0, 10), sticky="nsew")

    def _setup_parameters_tab(self, parent_frame):
        """Setup the parameters configuration tab."""
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(1, weight=1)

        # Protocol Selector
        selector_card = ttk.LabelFrame(
            parent_frame,
            text="SCRIPT SELECTION",
            padding=15,
            style="Metric.TLabelframe",
        )
        selector_card.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

        selector_inner = ttk.Frame(selector_card)
        selector_inner.pack(fill="x")

        ttk.Label(selector_inner, text="Configure parameters for:").pack(side=tk.LEFT)
        self.protocol_selector = ttk.Combobox(
            selector_inner,
            values=list(self.protocols.keys()),
            state="readonly",
            width=40,
        )
        self.protocol_selector.pack(side=tk.LEFT, padx=(10, 0))
        self.protocol_selector.bind("<<ComboboxSelected>>", self.on_protocol_selected)

        # Parameters Frame
        params_card = ttk.LabelFrame(
            parent_frame,
            text="PARAMETER CONFIGURATION",
            padding=15,
            style="Metric.TLabelframe",
        )
        params_card.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollable frame for parameters
        self.params_canvas = tk.Canvas(params_card, background=COLORS["surface"])
        self.params_scrollbar = ttk.Scrollbar(
            params_card, orient="vertical", command=self.params_canvas.yview
        )
        self.params_scrollable_frame = ttk.Frame(self.params_canvas)

        self.params_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.params_canvas.configure(
                scrollregion=self.params_canvas.bbox("all")
            ),
        )

        self.params_canvas.create_window(
            (0, 0), window=self.params_scrollable_frame, anchor="nw"
        )
        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)

        self.params_canvas.pack(side="left", fill="both", expand=True)
        self.params_scrollbar.pack(side="right", fill="y")

        # Control Buttons
        control_frame = ttk.Frame(parent_frame)
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        APGIButtons.standard(
            control_frame, "Load Defaults", self.load_default_parameters
        ).pack(side=tk.LEFT, padx=(0, 10))
        APGIButtons.standard(
            control_frame, "Save Parameters", self.save_parameters
        ).pack(side=tk.LEFT)

        # Empty state message
        self._show_empty_params_state()

    def _show_empty_params_state(self):
        """Show empty state for parameters tab."""
        for widget in self.params_scrollable_frame.winfo_children():
            widget.destroy()

        create_empty_state(
            self.params_scrollable_frame,
            "Select a script from the dropdown above to view configurable parameters",
        )

    def _create_console(self):
        """Create bottom console (log output)."""
        console_frame = ttk.LabelFrame(
            self.root,
            text="INSTRUMENT CONSOLE - Real-time Data Stream",
            padding=10,
            style="Metric.TLabelframe",
        )
        console_frame.grid(
            row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

        # Console toolbar
        toolbar = ttk.Frame(console_frame)
        toolbar.pack(fill="x", pady=(0, 5))

        # Status indicator
        self.console_status = ttk.Label(
            toolbar,
            text="Idle",
            foreground=COLORS["text_secondary"],
            font=(FONTS["monospace"], 9),
        )
        self.console_status.pack(side=tk.LEFT)

        # Clear button
        APGIButtons.standard(toolbar, "Clear Console", self.clear_console).pack(
            side=tk.RIGHT, padx=(5, 0)
        )

        # Stop button
        self.stop_btn = APGIButtons.danger(toolbar, "Stop", self.stop_protocol)
        self.stop_btn.pack(side=tk.RIGHT)
        self.stop_btn.config(state=tk.DISABLED)

        # Output console
        console_container = ttk.Frame(console_frame)
        console_container.pack(fill="both", expand=True)

        self.output_console = scrolledtext.ScrolledText(
            console_container,
            height=10,
            font=(FONTS["monospace"], 10),
            background=COLORS["surface"],
            foreground=COLORS["text_primary"],
            insertbackground=COLORS["primary"],
            relief="solid",
            borderwidth=1,
        )
        self.output_console.pack(fill="both", expand=True)

        # Progress bar at bottom
        progress_frame = ttk.Frame(console_frame)
        progress_frame.pack(fill="x", pady=(5, 0))

        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
            length=400,
        )
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 0), fill="x", expand=True)

        self.progress_label = ttk.Label(
            progress_frame, text="0%", font=(FONTS["monospace"], 9)
        )
        self.progress_label.pack(side=tk.LEFT, padx=(5, 0))

    # ============================================================================
    # Protocol Selection and Parameter Management
    # ============================================================================

    def select_protocol(self, protocol_name, protocol_info):
        """Select a protocol for running."""
        self.selected_protocol = protocol_name
        self.selected_protocol_label.config(
            text=f"[OK] {protocol_name}",
            foreground=COLORS["success"],
        )
        self.selected_protocol_desc.config(text=protocol_info["description"])
        self.run_selected_button.config(state=tk.NORMAL)

        # Update status card
        self.stat_ready_card.lbl_value.config(text="Selected")

        # Also update combobox in parameters tab
        self.protocol_selector.set(protocol_name)
        self.display_protocol_parameters(protocol_name)

        self.log_message(f"Selected protocol: {protocol_name}")
        self.log_message(f"  Description: {protocol_info['description']}")

    def on_protocol_selected(self, event):
        """Handle protocol selection for parameter configuration."""
        selected = self.protocol_selector.get()
        if selected:
            self.display_protocol_parameters(selected)

    def display_protocol_parameters(self, protocol_name):
        """Display parameter controls for the selected protocol."""
        # Clear existing parameters
        for widget in self.params_scrollable_frame.winfo_children():
            widget.destroy()

        protocol_info = self.protocols.get(protocol_name)
        if not protocol_info or "parameters" not in protocol_info:
            ttk.Label(
                self.params_scrollable_frame,
                text="No configurable parameters for this protocol",
                foreground=COLORS["text_secondary"],
            ).pack(pady=20)
            return

        # Create parameter controls
        row = 0
        self.parameter_widgets = {}

        # Parameter header
        header = ttk.Label(
            self.params_scrollable_frame,
            text=f"Parameters for {protocol_name}",
            font=(FONTS["primary"], 11, "bold"),
        )
        header.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 15))
        row += 1

        for param_name, param_config in protocol_info["parameters"].items():
            # Parameter label with scientific notation styling
            label_text = param_name.replace("_", " ").title()
            label = ttk.Label(
                self.params_scrollable_frame,
                text=f"{label_text}:",
                font=(FONTS["monospace"], 10),
            )
            label.grid(row=row, column=0, sticky=tk.W, padx=(0, 15), pady=5)

            # Create appropriate widget based on type
            if param_config["type"] == "float":
                var = tk.DoubleVar(value=param_config["default"])
                widget = tk.Spinbox(
                    self.params_scrollable_frame,
                    from_=param_config["min"],
                    to=param_config["max"],
                    increment=(param_config["max"] - param_config["min"]) / 100,
                    textvariable=var,
                    width=15,
                    font=(FONTS["monospace"], 10),
                    validate="key",
                    validatecommand=(
                        self.root.register(self._validate_spinbox_float),
                        "%P",
                        param_config["min"],
                        param_config["max"],
                    ),
                )
            elif param_config["type"] == "int":
                var = tk.IntVar(value=param_config["default"])
                widget = tk.Spinbox(
                    self.params_scrollable_frame,
                    from_=param_config["min"],
                    to=param_config["max"],
                    increment=1,
                    textvariable=var,
                    width=15,
                    font=(FONTS["monospace"], 10),
                    validate="key",
                    validatecommand=(
                        self.root.register(self._validate_spinbox_int),
                        "%P",
                        param_config["min"],
                        param_config["max"],
                    ),
                )
            else:  # string
                var = tk.StringVar(value=param_config["default"])
                widget = ttk.Entry(
                    self.params_scrollable_frame,
                    textvariable=var,
                    width=17,
                    font=(FONTS["monospace"], 10),
                )

            widget.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)

            # Description tooltip label
            desc_label = ttk.Label(
                self.params_scrollable_frame,
                text=param_config.get("description", ""),
                foreground=COLORS["text_secondary"],
                font=(FONTS["primary"], 9),
                wraplength=300,
            )
            desc_label.grid(row=row, column=2, sticky=tk.W, padx=(10, 0), pady=5)

            # Store widget reference
            self.parameter_widgets[param_name] = {"var": var, "widget": widget}

            row += 1

        # Configure grid weights
        self.params_scrollable_frame.columnconfigure(1, weight=0)
        self.params_scrollable_frame.columnconfigure(2, weight=1)

    def load_default_parameters(self):
        """Load default parameter values."""
        for protocol_name, protocol_info in self.protocols.items():
            if "parameters" in protocol_info:
                self.parameter_values[protocol_name] = {}
                for param_name, param_config in protocol_info["parameters"].items():
                    self.parameter_values[protocol_name][param_name] = param_config[
                        "default"
                    ]

        self.log_message("Default parameters loaded for all scripts")

    def save_parameters(self):
        """Save current parameter values with validation."""
        selected = self.protocol_selector.get()
        if not selected:
            messagebox.showwarning(
                "No Selection", "Please select a protocol from the dropdown first."
            )
            return

        # First, update parameter values from widgets if they exist
        if hasattr(self, "parameter_widgets") and self.parameter_widgets:
            # Get current values from widgets
            for param_name, widget_info in self.parameter_widgets.items():
                try:
                    value = widget_info["var"].get()
                    self.parameter_values[selected][param_name] = value
                except Exception as e:
                    messagebox.showerror(
                        "Update Error", f"Error updating {param_name}: {e}"
                    )
                    return

        # Validate parameter values
        validation_errors = []
        protocol_info = self.protocols[selected]

        if "parameters" in protocol_info:
            for param_name, param_config in protocol_info["parameters"].items():
                current_value = self.parameter_values[selected].get(param_name)

                # Type-specific validation
                if param_config["type"] == "float":
                    try:
                        value = float(current_value)
                        if value < param_config["min"] or value > param_config["max"]:
                            validation_errors.append(
                                f"{param_name}: {value} is out of range [{param_config['min']}, {param_config['max']}]"
                            )
                    except (ValueError, TypeError):
                        validation_errors.append(
                            f"{param_name}: Invalid float value '{current_value}'"
                        )

                elif param_config["type"] == "int":
                    try:
                        value = int(current_value)
                        if value < param_config["min"] or value > param_config["max"]:
                            validation_errors.append(
                                f"{param_name}: {value} is out of range [{param_config['min']}, {param_config['max']}]"
                            )
                    except (ValueError, TypeError):
                        validation_errors.append(
                            f"{param_name}: Invalid integer value '{current_value}'"
                        )

                elif param_config["type"] == "str":
                    # String validation - check if it's a valid format
                    if param_name == "surprise_range":
                        try:
                            # Parse [min, max] format using JSON
                            import json

                            value_list = json.loads(current_value)
                            if not isinstance(value_list, list) or len(value_list) != 2:
                                validation_errors.append(
                                    f"{param_name}: Must be in format [min, max]"
                                )
                        except (json.JSONDecodeError, TypeError):
                            validation_errors.append(
                                f"{param_name}: Invalid range format '{current_value}'"
                            )

        if validation_errors:
            messagebox.showerror(
                "Validation Errors",
                "Please fix the following errors:\n\n" + "\n".join(validation_errors),
            )
            return

        # Save to file
        try:
            import json

            # Use absolute path based on project root
            project_root = Path(__file__).parent
            config_dir = project_root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)

            config_file = (
                config_dir / f"{selected.lower().replace(' ', '_')}_params.json"
            )
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self.parameter_values[selected], f, indent=2)

            messagebox.showinfo("Success", f"Parameters saved to {config_file}")
            self.log_message(f"Parameters saved for {selected}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save parameters: {e}")
            self.log_message(f"Error saving parameters: {e}")
            return

    def refresh_parameter_list(self):
        """Refresh the parameter display for the currently selected protocol."""
        selected = self.protocol_selector.get()
        if selected:
            self.display_protocol_parameters(selected)

    # ============================================================================
    # Protocol Execution
    # ============================================================================

    def run_all_protocols(self):
        """Run all protocols sequentially with default parameters."""
        # Confirm with user
        protocol_count = len(self.protocols)
        if not messagebox.askyesno(
            "Confirm Run All",
            f"This will execute all {protocol_count} theory scripts sequentially.\n\n"
            "This may take a significant amount of time.\n\n"
            "Continue?",
        ):
            return

        # Force dialog dismissal and UI update before long-running operation
        self.root.update()

        # Update console status
        self.console_status.config(text="Running", foreground=COLORS["primary"])
        self.system_status_label.config(text="Processing", foreground=COLORS["primary"])
        self.stat_ready_card.lbl_value.config(text="Running All")

        # On macOS, run synchronously; on other platforms use thread
        import platform

        is_macos = platform.system() == "Darwin"

        if is_macos:
            self.stop_event.clear()
            self.run_all_button.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.set_status("Running all protocols...")

            total = len(self.protocols)
            for idx, (protocol_name, protocol_info) in enumerate(
                self.protocols.items(), 1
            ):
                if self.stop_event.is_set():
                    self.log_message("=== Run All stopped by user ===")
                    break

                self.log_message(f"\n{'=' * 60}")
                self.log_message(f"Running {protocol_name} ({idx}/{total})")
                self.log_message(f"{'=' * 60}")

                # Update progress
                progress = (idx / total) * 100
                self.progress_var.set(progress)
                self.progress_label.config(text=f"{progress:.0f}%")
                self.root.update()

                # Use default parameters
                protocol_info_with_params = protocol_info.copy()
                protocol_info_with_params["configured_params"] = {}

                # Run synchronously on main thread for macOS
                self._run_single_protocol(protocol_info_with_params)

            self.log_message(f"\n{'=' * 60}")
            self.log_message("All protocols completed!")
            self.log_message(f"{'=' * 60}")
            self.set_status("Ready")
            self.run_all_button.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

            # Reset status indicators
            self.console_status.config(text="Idle", foreground=COLORS["text_secondary"])
            self.system_status_label.config(
                text="[OK] Ready", foreground=COLORS["success"]
            )
            self.stat_ready_card.lbl_value.config(text="Ready")
            self.progress_var.set(0)
            self.progress_label.config(text="0%")
            return

        # Run protocols in a separate thread on non-macOS platforms
        def run_all_thread():
            total = len(self.protocols)
            for idx, (protocol_name, protocol_info) in enumerate(
                self.protocols.items(), 1
            ):
                if self.stop_event.is_set():
                    self.log_message("=== Run All stopped by user ===")
                    break

                self.log_message(f"\n{'=' * 60}")
                self.log_message(f"Running {protocol_name} ({idx}/{total})")
                self.log_message(f"{'=' * 60}")

                # Update progress
                progress = (idx / total) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(
                    0, lambda p=progress: self.progress_label.config(text=f"{p:.0f}%")
                )

                # Use default parameters
                protocol_info_with_params = protocol_info.copy()
                protocol_info_with_params["configured_params"] = {}

                # Run synchronously in this thread
                self._run_single_protocol(protocol_info_with_params)

            self.log_message(f"\n{'=' * 60}")
            self.log_message("All protocols completed!")
            self.log_message(f"{'=' * 60}")
            self.set_status("Ready")
            self.root.after(0, lambda: self.run_all_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
            self.root.after(
                0,
                lambda: self.console_status.config(
                    text="Idle", foreground=COLORS["text_secondary"]
                ),
            )
            self.root.after(
                0,
                lambda: self.system_status_label.config(
                    text="[OK] Ready", foreground=COLORS["success"]
                ),
            )
            self.root.after(
                0, lambda: self.stat_ready_card.lbl_value.config(text="Ready")
            )
            self.root.after(0, lambda: self.progress_var.set(0))
            self.root.after(0, lambda: self.progress_label.config(text="0%"))

        self.stop_event.clear()
        self.run_all_button.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.set_status("Running all protocols...")

        thread = threading.Thread(target=run_all_thread)
        thread.daemon = True
        thread.start()

    def _run_single_protocol(self, protocol_info):
        """Run a single protocol synchronously (for use in run_all)."""
        try:
            # Ensure project root is in sys.path for imports
            project_root = os.path.dirname(os.path.abspath(__file__))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # Load the protocol module
            file_path = protocol_info.get("file_path")
            if not file_path or not os.path.exists(file_path):
                self.log_message(f"Error: Script file not found: {file_path}")
                return {"status": "ERROR", "message": "Script file not found"}

            # Use consistent module name (without .py extension) for both spec and sys.modules
            _mod_key = protocol_info.get(
                "module_name", protocol_info["file"].replace(".py", "")
            )
            spec = importlib.util.spec_from_file_location(_mod_key, file_path)
            module = importlib.util.module_from_spec(spec)
            # Pre-register so modules that do sys.modules[__name__] self-reference work
            sys.modules[_mod_key] = module
            try:
                spec.loader.exec_module(module)
            finally:
                # Clean up temporary sys.modules entry to avoid stale state
                sys.modules.pop(_mod_key, None)

            # Get configured parameters and execution info
            configured_params = protocol_info.get("configured_params", {})
            exec_info = protocol_info.get("execution_info", {})

            # Execute based on execution_info type
            exec_type = exec_info.get("type", "exec_module")
            result = {}

            if exec_type == "module_function":
                func_name = exec_info.get("function", "main")
                run_func = getattr(module, func_name, None)
                if run_func:
                    self.log_message(f"  Running {func_name}()...")
                    try:
                        # Try with configured parameters first
                        result = run_func(**configured_params)
                    except TypeError as e:
                        # If that fails, try without parameters
                        if "required positional argument" in str(e):
                            self.log_message(
                                f"  Function {func_name} requires specific arguments - skipping execution"
                            )
                            result = {
                                "status": "SKIPPED",
                                "message": f"Function {func_name} requires data arguments not available in GUI",
                            }
                        else:
                            # Try calling without arguments
                            try:
                                result = run_func()
                            except Exception as e2:
                                self.log_message(
                                    f"  ERROR calling {func_name}: {str(e2)}"
                                )
                                result = {
                                    "status": "ERROR",
                                    "message": f"Function {func_name} failed: {str(e2)}",
                                }
                    except Exception as e:
                        self.log_message(f"  ERROR calling {func_name}: {str(e)}")
                        result = {
                            "status": "ERROR",
                            "message": f"Function {func_name} failed: {str(e)}",
                        }
                else:
                    self.log_message(f"  Function {func_name} not found")
                    result = {
                        "status": "ERROR",
                        "message": f"Function {func_name} not found",
                    }

            elif exec_type == "class_method":
                class_name = exec_info.get("class")
                method_name = exec_info.get("method", "run_validation")

                if class_name:
                    cls = getattr(module, class_name)
                    self.log_message(f"  Instantiating {class_name}...")

                    try:
                        instance = cls(**configured_params)
                    except (TypeError, ValueError):
                        instance = cls()

                    method = getattr(instance, method_name)
                    self.log_message(f"  Calling {method_name}()...")
                    result = method()
                else:
                    result = {"status": "ERROR", "message": "No class specified"}

            else:
                # Module-level execution
                self.log_message("  Executing module...")
                result = {
                    "status": "Executed",
                    "module": protocol_info.get("module_name"),
                }

            # Save results
            self._save_results(result, protocol_info["file"])
            self.log_message("  Script completed successfully")
            return result

        except Exception as e:
            self.log_message(f"  ERROR: {str(e)}")
            logger.error(f"Error in {protocol_info.get('file', 'unknown')}: {e}")
            return {"status": "ERROR", "message": str(e)}

    def run_selected_protocol(self):
        """Run the currently selected protocol with configured parameters."""
        if not self.selected_protocol:
            messagebox.showwarning("Warning", "No protocol selected")
            return

        protocol_info = self.protocols[self.selected_protocol]

        # Get configured parameters from widgets
        params = {}
        if hasattr(self, "parameter_widgets"):
            for param_name, widget_info in self.parameter_widgets.items():
                try:
                    value = widget_info["var"].get()
                    params[param_name] = value
                except Exception:
                    # Use default value if widget access fails
                    pass

        # Merge with protocol info
        protocol_info_with_params = protocol_info.copy()
        protocol_info_with_params["configured_params"] = params

        self.run_protocol(protocol_info_with_params)

    # ============================================================================
    # Core Protocol Execution
    # ============================================================================

    def run_protocol(self, protocol_info):
        """Run selected protocol - on macOS run synchronously to avoid threading issues"""
        import platform

        is_macos = platform.system() == "Darwin"

        def protocol_execution():
            """Actual protocol execution code"""
            try:
                self.set_status(f"Running {protocol_info['file']}...")
                self.log_message(f"=== Running {protocol_info['file']} ===")

                # Ensure project root is in sys.path for imports
                project_root = os.path.dirname(os.path.abspath(__file__))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                # Use the stored file_path from discovery
                file_path = protocol_info.get("file_path")
                if not file_path or not os.path.exists(file_path):
                    # Fallback for manual calls
                    file_path = os.path.join(
                        os.path.dirname(__file__), "Theory", protocol_info["file"]
                    )

                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Script file not found: {file_path}")

                # Use consistent module name (without .py extension) for both spec and sys.modules
                _mod_key = protocol_info.get(
                    "module_name", protocol_info["file"].replace(".py", "")
                )
                spec = importlib.util.spec_from_file_location(_mod_key, file_path)
                module = importlib.util.module_from_spec(spec)
                # Pre-register so modules that do sys.modules[__name__] self-reference work
                sys.modules[_mod_key] = module
                try:
                    spec.loader.exec_module(module)
                finally:
                    sys.modules.pop(_mod_key, None)
                self.module = module

                # Handle protocol based on execution_info with configured parameters
                configured_params = protocol_info.get("configured_params", {})
                exec_info = protocol_info.get("execution_info", {})

                # Validate numeric parameters
                validated_params = {}
                for key, value in configured_params.items():
                    try:
                        num_value = float(value)
                        if "dim" in key or "n_" in key:
                            if num_value <= 0 or num_value != int(num_value):
                                raise ValueError(f"{key} must be a positive integer")
                            validated_params[key] = int(num_value)
                        elif "lr" in key:
                            if num_value <= 0 or num_value > 1.0:
                                raise ValueError(f"{key} must be between 0 and 1")
                            validated_params[key] = num_value
                        else:
                            validated_params[key] = num_value
                    except (ValueError, TypeError) as e:
                        self.log_message(
                            f"Warning: Invalid value for {key}: {value}. Error: {e}. Using default."
                        )

                # Execute based on execution_info type
                exec_type = exec_info.get("type", "exec_module")

                if exec_type == "module_function":
                    # Call module-level function
                    func_name = exec_info.get("function", "main")
                    run_func = getattr(module, func_name, None)
                    if run_func:
                        self.log_message(f"Running module function: {func_name}()")
                        # Try to pass params if function accepts them
                        try:
                            results = run_func(**validated_params)
                        except TypeError:
                            results = run_func()
                    else:
                        raise AttributeError(
                            f"Function {func_name} not found in module"
                        )

                elif exec_type == "class_method":
                    # Instantiate class and call method
                    class_name = exec_info.get("class")
                    method_name = exec_info.get("method", "run_validation")

                    if not class_name:
                        raise ValueError(
                            "No class specified for class_method execution"
                        )

                    cls = getattr(module, class_name)
                    self.log_message(f"Instantiating {class_name}...")

                    # Try to create instance with params, fall back to defaults
                    try:
                        instance = cls(**validated_params)
                    except (TypeError, ValueError):
                        instance = cls()
                        if validated_params:
                            self.log_message(
                                "Using default parameters (custom params not supported)"
                            )

                    # Call the method
                    method = getattr(instance, method_name)
                    self.log_message(f"Calling {class_name}.{method_name}()...")
                    results = method()

                else:
                    # Just execute the module (runs any top-level code)
                    self.log_message("Executing module-level code...")
                    results = {
                        "status": "Module executed",
                        "module": protocol_info.get("module_name"),
                    }

                # Save results to file
                self._save_results(results, protocol_info["file"])

                self.log_message("=== Script completed successfully ===")
                self.set_status("Ready")

                # Update UI
                if is_macos:
                    self.stop_btn.config(state=tk.DISABLED)
                    self.console_status.config(
                        text="Idle", foreground=COLORS["text_secondary"]
                    )
                    self.stat_ready_card.lbl_value.config(text="Ready")
                else:
                    self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
                    self.root.after(
                        0,
                        lambda: self.console_status.config(
                            text="Idle", foreground=COLORS["text_secondary"]
                        ),
                    )
                    self.root.after(
                        0, lambda: self.stat_ready_card.lbl_value.config(text="Ready")
                    )

            except (
                ImportError,
                ModuleNotFoundError,
                AttributeError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
            ) as e:
                error_msg = f"Error running {protocol_info['file']}: {str(e)}"
                logger.warning(error_msg)
                if is_macos:
                    messagebox.showerror("Error", error_msg)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.console_status.config(
                        text="[X] Error", foreground=COLORS["alert"]
                    )
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
                    self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
                    self.root.after(
                        0,
                        lambda: self.console_status.config(
                            text="[X] Error", foreground=COLORS["alert"]
                        ),
                    )
                self.set_status("Error")
            except Exception as e:
                logger.error(f"Unexpected error in {protocol_info['file']}: {str(e)}")
                error_message = f"Unexpected error: {str(e)}"
                if is_macos:
                    messagebox.showerror("Error", error_message)
                    self.stop_btn.config(state=tk.DISABLED)
                    self.console_status.config(
                        text="[X] Error", foreground=COLORS["alert"]
                    )
                else:
                    self.root.after(
                        0, lambda: messagebox.showerror("Error", error_message)
                    )
                    self.root.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
                    self.root.after(
                        0,
                        lambda: self.console_status.config(
                            text="[X] Error", foreground=COLORS["alert"]
                        ),
                    )
                self.set_status("Error")

        # On macOS, run synchronously on main thread to avoid NSWindow threading issues
        if is_macos:
            self.stop_event.clear()
            self.stop_btn.config(state=tk.NORMAL)
            self.console_status.config(text="Running", foreground=COLORS["primary"])
            self.stat_ready_card.lbl_value.config(text="Running")
            protocol_execution()
        else:
            # Run in separate thread to avoid blocking GUI on other platforms
            self.stop_event.clear()
            self.running_thread = None
            thread = threading.Thread(target=protocol_execution)
            thread.daemon = True
            self.running_thread = thread
            self.stop_btn.config(state=tk.NORMAL)
            self.root.after(
                0,
                lambda: self.console_status.config(
                    text="Running", foreground=COLORS["primary"]
                ),
            )
            self.root.after(
                0, lambda: self.stat_ready_card.lbl_value.config(text="Running")
            )
            thread.start()

    # ============================================================================
    # Specialized Handlers
    # ============================================================================

    def _handle_framework_aggregator(self, module):
        """Handle FP_ALL_Aggregator: collect saved result files and run aggregation."""
        project_root = Path(__file__).parent
        validation_dir = project_root / "validation_results"

        result_files = []
        if validation_dir.exists():
            result_files = [str(f) for f in sorted(validation_dir.glob("*.json"))]

        self.log_message(
            f"Framework Aggregator: found {len(result_files)} result file(s) in {validation_dir}"
        )

        run_fn = getattr(module, "run_framework_falsification", None)
        if run_fn is None:
            # Fall back to class method
            aggregator_cls = getattr(module, "FalsificationAggregator")
            agg = aggregator_cls()
            if result_files:
                result = agg.run_full_analysis(result_files)
            else:
                result = {
                    "status": "NO_RESULTS",
                    "message": "No previous validation results found. Run individual protocols first.",
                }
        else:
            if result_files:
                result = run_fn(result_files)
            else:
                result = {
                    "status": "NO_RESULTS",
                    "message": "No previous validation results found. Run individual protocols first.",
                }

        status = result.get("status", "Done") if isinstance(result, dict) else "Done"
        self.log_message(f"Framework Aggregator completed: {status}")
        return result

    def _handle_network_comparison(self, cls, params):
        """Handle NetworkComparisonExperiment with configured parameters."""
        config = {
            "extero_dim": params.get("extero_dim", 32),
            "intero_dim": params.get("intero_dim", 16),
            "action_dim": params.get("action_dim", 4),
            "context_dim": params.get("context_dim", 8),
            "n_episodes": params.get("n_episodes", 50),
        }
        instance = cls(config)
        result = instance.run_full_experiment()
        self.log_message("Network Comparison Experiment completed")
        self.log_message(f"Results: {type(result)}")
        return {"result": result, "type": "NetworkComparisonExperiment"}

    def _handle_apgi_agent(self, cls, module, params):
        """Run the full falsification protocol for APGI agent with configured parameters."""
        try:
            # Import the run_falsification function
            run_func = getattr(module, "run_falsification", None)
            if run_func:
                # Run the full protocol
                result = run_func()
                self.log_message(f"APGI Agent falsification completed: {result}")
                return {"result": result, "type": "APGIActiveInferenceAgent"}
            else:
                # Fallback to old behavior with configured parameters
                config = {
                    "lr_extero": params.get("lr_extero", 0.01),
                    "lr_intero": params.get("lr_intero", 0.01),
                    "lr_precision": params.get("lr_precision", 0.05),
                    "lr_somatic": params.get("lr_somatic", 0.1),
                    "n_actions": params.get("n_actions", 4),
                    "theta_init": params.get("theta_init", 0.5),
                    "theta_baseline": params.get("theta_baseline", 0.5),
                    "alpha": params.get("alpha", 8.0),
                    "tau_S": params.get("tau_S", 0.3),
                    "tau_theta": params.get("tau_theta", 10.0),
                    "eta_theta": params.get("eta_theta", 0.01),
                    "beta": params.get("beta", 1.2),
                    "rho": params.get("rho", 0.7),
                }
                _ = cls(config)
                self.log_message("APGI Agent created successfully (fallback)")
                self.log_message(f"Agent config: {config}")
                return {
                    "result": {"config": config},
                    "type": "APGIActiveInferenceAgent",
                }
        except Exception as e:
            self.log_message(f"Error in APGI agent protocol: {str(e)}")
            raise

    def _handle_iowa_gambling(self, cls, module, params):
        """Run the full falsification protocol for Iowa Gambling Task with configured parameters."""
        try:
            # Import the run_falsification function
            run_func = getattr(module, "run_falsification", None)
            if run_func:
                # Run the full protocol
                result = run_func()
                self.log_message(
                    f"Iowa Gambling Task falsification completed: {result}"
                )
                return {"result": result, "type": "IowaGamblingTaskEnvironment"}
            else:
                # Fallback to old behavior with configured parameters
                n_trials = params.get("n_trials", 100)
                env = cls(n_trials=n_trials)
                self.log_message("Iowa Gambling Task Environment created (fallback)")

                # Run demo trials
                n_demo_trials = min(5, n_trials)
                total_reward = 0
                for trial in range(n_demo_trials):
                    action = 0  # Always pick deck A for demo
                    reward, intero_cost, obs, done = env.step(action)
                    total_reward += reward
                    self.log_message(
                        f"Trial {trial + 1}: Action={action}, Reward={reward:.2f}, "
                        f"InteroCost={intero_cost:.2f}"
                    )

                self.log_message(f"Demo completed. Total reward: {total_reward:.2f}")
                return {
                    "total_reward": total_reward,
                    "type": "IowaGamblingTaskEnvironment",
                }
        except Exception as e:
            self.log_message(f"Error in Iowa Gambling protocol: {str(e)}")
            raise

    def _handle_run_full_experiment(self, cls, params):
        """Handle protocols with run_full_experiment method using configured parameters."""
        result = {}
        try:
            # Try to pass parameters to constructor if they match expected signature
            try:
                instance = cls(**params)
            except (TypeError, ValueError) as e:
                # These are expected when parameters don't match constructor signature
                logger.warning(f"Parameter mismatch for {cls.__name__}: {e}")
                instance = cls()  # Create instance with defaults
                self.log_message(
                    "Warning: Could not apply configured parameters, using defaults"
                )
            except Exception as e:
                # Log unexpected errors but don't suppress them
                logger.warning(f"Unexpected error in application: {e}")
                instance = cls()  # Create instance with defaults
                self.log_message(
                    "Warning: Could not apply configured parameters, using defaults"
                )

            result = instance.run_full_experiment()
            self.log_message(f"Experiment completed: {type(result)}")
        except ValueError as ve:
            if "broadcast" in str(ve):
                self.log_message(
                    "Broadcasting error in Script-3 - this is expected due to observation size mismatches"
                )
                self.log_message("Script-3 needs observation size alignment")
            else:
                raise
        return result

    def _handle_phase_transition(self, cls, module, params):
        """Handle phase transition analysis with configured parameters."""
        surprise_system = module.SurpriseIgnitionSystem()

        # Apply configured parameters to surprise system if possible
        for param_name, value in params.items():
            if hasattr(surprise_system, param_name):
                try:
                    setattr(surprise_system, param_name, value)
                except Exception as _:  # noqa: F841
                    pass  # Skip parameters that can't be set

        instance = cls(surprise_system)
        result = instance.run_phase_transition_analysis()
        self.log_message(f"Phase transition analysis completed: {type(result)}")
        return result

    def _handle_evolution(self, cls, params):
        """Handle evolutionary protocols with configured parameters."""
        try:
            config = {
                "population_size": params.get("population_size", 50),
                "n_generations": params.get("n_generations", 100),
                "mutation_rate": params.get("mutation_rate", 0.1),
                "selection_pressure": params.get("selection_pressure", 2.0),
            }

            instance = cls(stop_event=self.stop_event, **config)
            self.log_message("Starting evolutionary simulation (this may take time)...")
            result = instance.run_evolution()
            self.log_message(f"Evolution completed: {type(result)}")
        except KeyboardInterrupt:
            self.log_message("Evolution interrupted by user")
        except (
            RuntimeError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
        ) as e:
            self.log_message(f"Error in evolution: {str(e)}")
            self.log_message("EvolutionaryAPGIEmergence instance creation failed")

    def _handle_default(self, cls, module, params):
        """Handle default protocol execution with configured parameters."""
        try:
            instance = cls(**params)  # Create instance with parameters
            self.log_message(f"Created {cls.__name__} instance with parameters")
        except TypeError:
            instance = cls()  # Create instance with defaults
            self.log_message(
                "Warning: Could not apply configured parameters, using defaults"
            )

        # Try to find any standard run method
        self.log_message(f"Attempting to run {cls.__name__}...")
        if hasattr(instance, "run_validation"):
            return instance.run_validation()
        elif hasattr(module, "run_validation"):
            return module.run_validation()
        elif hasattr(instance, "run_falsification"):
            return instance.run_falsification()
        else:
            self.log_message(f"No standard run method found for {cls.__name__}")
            return {"status": "Completed (No Test)", "instance": str(instance)}

    # ============================================================================
    # Utilities
    # ============================================================================

    def _save_results(self, results, protocol_file):
        """Save validation results to a JSON file."""
        try:
            import json
            import uuid
            from datetime import datetime

            # Create validation results directory if it doesn't exist
            project_root = Path(__file__).parent
            validation_dir = project_root / "validation_results"
            validation_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            unique_id = str(uuid.uuid4())[:8]
            filename = f"validation_results_{unique_id}.json"
            filepath = validation_dir / filename

            # Extract named_predictions from results if present
            named_predictions = {}
            if isinstance(results, dict):
                # Try to get named_predictions from top level
                if "named_predictions" in results:
                    named_predictions = results["named_predictions"]
                # Also try nested in results["results"]
                elif "results" in results and isinstance(results["results"], dict):
                    if "named_predictions" in results["results"]:
                        named_predictions = results["results"]["named_predictions"]

            # Extract bic_values from results for aggregator Condition B
            bic_values = {}
            if isinstance(results, dict):
                # Try top level
                if "bic_values" in results:
                    bic_values = results["bic_values"]
                elif "model_comparison" in results:
                    # Convert model_comparison to bic_values format
                    mc = results["model_comparison"]
                    if isinstance(mc, dict) and "apgi" in mc:
                        bic_values = {"IGT": mc}  # Use environment name as key

            # Build final output structure
            results_with_metadata = {
                "metadata": {
                    "protocol_file": protocol_file,
                    "timestamp": datetime.now().isoformat(),
                    "id": unique_id,
                },
                "named_predictions": named_predictions,
                "bic_values": bic_values,
                "results": results,
            }

            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(results_with_metadata, f, indent=2, default=str)

            self.log_message(f"Results saved to {filepath}")

        except Exception as e:
            self.log_message(f"Error saving results: {e}")

    def _validate_spinbox_float(self, value, min_val, max_val):
        """Validate float spinbox input."""
        if value == "":
            return True
        try:
            val = float(value)
            return float(min_val) <= val <= float(max_val)
        except ValueError:
            return False

    def _validate_spinbox_int(self, value, min_val, max_val):
        """Validate integer spinbox input."""
        if value == "":
            return True
        try:
            val = int(float(value))  # Allow decimal input but convert to int
            return int(min_val) <= val <= int(max_val)
        except ValueError:
            return False

    def create_tooltip(self, widget, text):
        """Create tooltip for widget (macOS thread-safe)"""

        def on_enter(event):
            # Use after() to ensure GUI operations happen on main thread
            def show_tooltip():
                try:
                    tooltip = tk.Toplevel()
                    tooltip.wm_overrideredirect(True)
                    tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
                    label = tk.Label(
                        tooltip,
                        text=text,
                        background=COLORS["soft_gray"],
                        relief=tk.SOLID,
                        borderwidth=1,
                        font=(FONTS["primary"], 9),
                        foreground=COLORS["text_primary"],
                    )
                    label.pack()
                    widget.tooltip = tooltip
                except Exception:
                    pass  # Silently fail if tooltip can't be created

            self.root.after(0, show_tooltip)

        def on_leave(event):
            def hide_tooltip():
                if hasattr(widget, "tooltip"):
                    try:
                        widget.tooltip.destroy()
                        del widget.tooltip
                    except Exception:
                        pass

            self.root.after(0, hide_tooltip)

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def clear_console(self):
        """Clear the console output."""
        if hasattr(self, "output_console"):
            self.output_console.delete(1.0, tk.END)

    def _setup_tab_order(self) -> None:
        """Configure tab order for consistent keyboard navigation."""
        self._tab_widgets: List[tk.Widget] = []

    def stop_protocol(self):
        """Stop the currently running protocol if any"""
        if self.stop_event:
            self.stop_event.set()
            self.log_message("Stop signal sent to protocol...")
            self.set_status("Stopping...")
            self.console_status.config(text="Stopping", foreground=COLORS["info"])

    def set_status(self, message):
        """Set status bar message (thread-safe)"""

        def _update():
            self.system_status_label.config(
                text=f"{message}", foreground=COLORS["primary"]
            )

        self.root.after(0, _update)

    def log_message(self, message):
        """Add message to output console (thread-safe)"""

        def _update():
            self.output_console.insert(tk.END, message + "\n")
            self.output_console.see(tk.END)

            # Limit the number of lines to prevent memory issues
            max_lines = 1000
            lines = self.output_console.get("1.0", tk.END).splitlines()
            if len(lines) > max_lines:
                # Keep only the last max_lines lines
                self.output_console.delete("1.0", tk.END)
                self.output_console.insert(tk.END, "\n".join(lines[-max_lines:]) + "\n")

        self.root.after(0, _update)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    import platform

    if platform.system() == "Darwin":
        # macOS requires tkinter on main thread
        pass

    root = tk.Tk()
    ScriptRunnerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
