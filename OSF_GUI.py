#!/usr/bin/env python3
"""
APGI Open Science Framework (OSF) Protocol Management GUI

Manages the 7 Empirical Protocols (EP-0 through EP-6):
pre-registration tracking, dependency visualization, and report export.
"""

import json
import logging
import os
import queue
import sys
import tempfile
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import TclError, filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

# ── APGI Design System ────────────────────────────────────────────────────────

_BG = "#f8f9fa"
_FG = "#212529"
_BLUE = "#2874a6"
_GREEN = "#155724"
_RED = "#721c24"
_BORDER = "#dee2e6"
_SURFACE = "#ffffff"
_MUTED = "#6c757d"
_SOFT = "#e9ecef"


def apply_apgi_theme(root: tk.Tk) -> ttk.Style:
    """Apply unified APGI theme to tkinter application."""
    style = ttk.Style()
    style.theme_use("clam")

    style.configure("TFrame", background=_BG)
    style.configure("TLabel", background=_BG, foreground=_FG, font=("Noto Sans", 10))
    style.configure("Header.TLabel", font=("Noto Sans", 12, "bold"), background=_BG, foreground=_FG)
    style.configure("Title.TLabel", font=("Noto Sans", 14, "bold"), background=_SURFACE, foreground=_FG)
    style.configure("Subtitle.TLabel", font=("Noto Sans", 10), background=_SURFACE, foreground=_MUTED)
    style.configure("Monospace.TLabel", font=("Noto Sans Mono", 11), background=_SURFACE, foreground=_BLUE)
    style.configure("TLabelframe", background=_BG, bordercolor=_BORDER)
    style.configure("TLabelframe.Label", background=_BG, foreground=_MUTED, font=("Noto Sans", 9, "bold"))
    style.configure("Card.TFrame", background=_SURFACE, borderwidth=1, relief="solid")
    style.configure("Card.TCheckbutton", background=_SURFACE)
    style.configure("TButton", padding=6, background=_SOFT, font=("Noto Sans", 10))
    style.map(
        "TButton", background=[("active", _BORDER), ("disabled", "#f1f3f5")], foreground=[("disabled", "#adb5bd")]
    )
    style.configure("Primary.TButton", background=_GREEN, foreground="white", font=("Noto Sans", 10, "bold"), padding=8)
    style.map(
        "Primary.TButton",
        background=[("active", "#0f3d1a"), ("disabled", _MUTED)],
        foreground=[("active", "white"), ("disabled", _BORDER)],
    )
    style.configure("Secondary.TButton", background=_BLUE, foreground="white", font=("Noto Sans", 10), padding=6)
    style.map(
        "Secondary.TButton",
        background=[("active", "#1f5a82"), ("disabled", _MUTED)],
        foreground=[("active", "white"), ("disabled", _BORDER)],
    )
    style.configure("Danger.TButton", background=_RED, foreground="white", font=("Noto Sans", 10, "bold"), padding=8)
    style.map(
        "Danger.TButton",
        background=[("active", "#5a161d"), ("disabled", _MUTED)],
        foreground=[("active", "white"), ("disabled", _BORDER)],
    )
    style.configure("Horizontal.TProgressbar", background=_BLUE, troughcolor=_BORDER, borderwidth=0)
    style.configure("TCombobox", font=("Noto Sans", 10))
    style.configure("TNotebook", background=_BG, tabmargins=[2, 5, 2, 0])
    style.configure("TNotebook.Tab", font=("Noto Sans", 10), padding=[10, 5])
    style.map("TNotebook.Tab", background=[("selected", _SURFACE)], expand=[("selected", [1, 1, 1, 0])])

    root.configure(background=_BG)
    return style


class APGICard(ttk.Frame):
    """Standardized information card for APGI applications."""

    def __init__(self, parent: tk.Widget, title: str, value: str, subtitle: str = "", **kwargs: Any) -> None:
        super().__init__(parent, style="Card.TFrame", **kwargs)
        container = ttk.Frame(self, padding=15, style="Card.TFrame")
        container.pack(fill="both", expand=True)

        ttk.Label(
            container, text=title.upper(), font=("Noto Sans", 11, "bold"), background=_SURFACE, foreground=_FG
        ).pack(anchor="w")

        ttk.Label(
            container, text=value, font=("Noto Sans Mono", 11), background=_SURFACE, foreground=_BLUE, wraplength=580
        ).pack(anchor="w", pady=(4, 8))

        if subtitle:
            ttk.Separator(container, orient="horizontal").pack(fill="x", pady=(0, 6))
            ttk.Label(
                container,
                text=subtitle,
                font=("Noto Sans", 9, "italic"),
                background=_SURFACE,
                foreground=_MUTED,
                wraplength=580,
            ).pack(anchor="w")


# ── Protocol Definitions ──────────────────────────────────────────────────────

EP_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "EP-0: HEP Proxy Validation": {
        "id": "EP-0",
        "title": "HEP Proxy Validation",
        "priority": 1,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": [],
        "status": "Not started",
        "description": (
            "Heartbeat-evoked potential (HEP) proxy validation for interoceptive precision gating. "
            "Establishes baseline cardiac-neural coupling signatures required by all downstream EP protocols."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 30 (power analysis required)",
        "measures": ["HEP amplitude", "Cardiac cycle phase", "Interoceptive accuracy (heartbeat detection)"],
        "analysis": "Mixed-effects models, time-frequency decomposition, HEP source localization",
        "falsification_criterion": (
            "HEP modulation < 0.1 μV between high/low interoceptive precision conditions "
            "falsifies cardiac-interoceptive coupling hypothesis."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-1: EEG Interoceptive Precision Gating": {
        "id": "EP-1",
        "title": "EEG Interoceptive Precision Gating",
        "priority": 1,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "Tests APGI prediction that interoceptive precision gates sensory processing via EEG microstate "
            "dynamics and global field power (GFP) modulation. Requires EP-0 baseline signatures."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 40",
        "measures": ["EEG microstates", "GFP", "P3b amplitude", "Interoceptive precision estimates"],
        "analysis": "Microstate segmentation, GFP analysis, Bayesian model comparison (APGI vs. null)",
        "falsification_criterion": (
            "No GFP modulation by interoceptive precision prior (Bayes factor B < 1/3) "
            "falsifies the APGI precision-gating hypothesis."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-2: Causal TMS Insula/dlPFC": {
        "id": "EP-2",
        "title": "Causal TMS Insula/dlPFC",
        "priority": 2,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "Causal intervention using TMS over insular cortex and dorsolateral prefrontal cortex "
            "to test APGI causal architecture predictions. Requires EP-0 HEP baselines."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 24",
        "measures": ["TMS-evoked potentials (TEP)", "Behavioral accuracy", "Precision-weighted prediction errors"],
        "analysis": "Repeated-measures ANOVA, TEP component analysis, causal inference modelling",
        "falsification_criterion": (
            "Insula TMS fails to modulate precision-weighted prediction errors (p > 0.05, d < 0.3) "
            "falsifies the causal role of insula in APGI gating."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-3: Active Inference Simulations": {
        "id": "EP-3",
        "title": "Active Inference Simulations",
        "priority": 1,
        "type": "Computational",
        "prereg_required": True,
        "depends_on": [],
        "status": "Not started",
        "description": (
            "Computational validation of the APGI active inference architecture via simulation benchmarks. "
            "Establishes formal quantitative predictions for all empirical protocols (EP-1, EP-5, EP-6)."
        ),
        "platform": "OSF",
        "sample_size": "N/A (computational — 10 000 simulation runs minimum)",
        "measures": ["Free energy minimisation", "Precision convergence", "Policy selection accuracy"],
        "analysis": "Simulation benchmarking, parameter recovery, model comparison (BIC/AIC/WAIC)",
        "falsification_criterion": (
            "Active inference agent fails to minimise free energy below baseline "
            "(>5 % residual prediction error) falsifies the APGI computational architecture."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-4: Disorders of Consciousness": {
        "id": "EP-4",
        "title": "Disorders of Consciousness",
        "priority": 3,
        "type": "Clinical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "Clinical validation of the APGI interoceptive precision model in patients with disorders of "
            "consciousness (VS/UWS and MCS). Requires EP-0 HEP signatures as biomarker reference."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 20 DOC patients + N ≥ 20 healthy controls",
        "measures": ["CRS-R scores", "HEP modulation", "Neural complexity (LZC, PCI)", "EEG connectivity"],
        "analysis": "Group comparisons, ROC analysis, Bayesian classifier, permutation testing",
        "falsification_criterion": (
            "HEP signatures fail to discriminate VS/UWS from MCS (AUC ≤ 0.70) "
            "falsifies clinical validity of APGI interoceptive precision biomarker."
        ),
        "prereg_status": "Pending",
        "notes": "Ethics approval and patient consent protocols required before data collection.",
    },
    "EP-5: fMRI Anticipation vs. Experience": {
        "id": "EP-5",
        "title": "fMRI Anticipation vs. Experience",
        "priority": 1,
        "type": "Empirical",
        "prereg_required": True,
        "depends_on": ["EP-0"],
        "status": "Not started",
        "description": (
            "fMRI study testing the APGI prediction that anticipatory interoceptive precision differs from "
            "experience-based interoceptive processing in vmPFC/insula circuitry. Requires EP-0 baselines."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 32",
        "measures": ["BOLD vmPFC/insula", "Anticipatory vs. reactive contrasts", "Functional connectivity"],
        "analysis": "GLM, psychophysiological interaction (PPI), dynamic causal modelling (DCM)",
        "falsification_criterion": (
            "No vmPFC/insula dissociation between anticipation and experience conditions (p > 0.05 FWE) "
            "falsifies the predictive interoception model."
        ),
        "prereg_status": "Pending",
        "notes": "",
    },
    "EP-6: iEEG All-or-None Dynamics": {
        "id": "EP-6",
        "title": "iEEG All-or-None Dynamics",
        "priority": 2,
        "type": "Clinical-Empirical",
        "prereg_required": True,
        "depends_on": ["EP-3"],
        "status": "Not started",
        "description": (
            "Intracranial EEG study in epilepsy patients testing the APGI prediction of all-or-none ignition "
            "dynamics in insular/anterior cingulate circuits. Predictions supplied by EP-3 simulations."
        ),
        "platform": "OSF",
        "sample_size": "N ≥ 10 iEEG patients",
        "measures": ["iEEG local field potentials", "High-gamma activity (70–150 Hz)", "Phase-amplitude coupling"],
        "analysis": "Single-trial ignition detection, phase transition analysis, Lempel-Ziv complexity",
        "falsification_criterion": (
            "Graded (not threshold/all-or-none) responses in insula/ACC circuits "
            "falsify the APGI ignition hypothesis."
        ),
        "prereg_status": "Pending",
        "notes": "Ethics approval required; patients recruited via epilepsy-monitoring unit.",
    },
}

# Status options for dropdown
_STATUS_OPTIONS = [
    "Not started",
    "In preparation",
    "Pre-registered",
    "Data collection",
    "Analysis",
    "Completed",
    "On hold",
]
_PREREG_OPTIONS = ["Pending", "Draft ready", "Submitted", "Registered", "Not applicable"]

# Type badge colours
_TYPE_COLORS: Dict[str, str] = {
    "Empirical": "#2874a6",
    "Computational": "#155724",
    "Clinical": "#721c24",
    "Clinical-Empirical": "#5a0080",
}

# Priority labels
_PRIORITY_LABELS = {1: "Priority 1 — High", 2: "Priority 2 — Medium", 3: "Priority 3 — Long-term"}


# ── Main Application ──────────────────────────────────────────────────────────


class OSFProtocolGUI:
    """GUI for APGI Open Science Framework — EP-0 through EP-6 protocol management."""

    def __init__(self, root: tk.Tk, headless: bool = False) -> None:
        self.root = root
        self.headless = headless
        self.gui_queue: queue.Queue[Any] = queue.Queue()
        self._protocol_status: Dict[str, Dict[str, str]] = {
            key: {"status": meta["status"], "prereg_status": meta["prereg_status"], "notes": meta["notes"]}
            for key, meta in EP_PROTOCOLS.items()
        }
        self._selected_protocol: Optional[str] = None
        self._console_visible = True
        self._card_widgets: List[tk.Widget] = []
        self._detail_vars: Dict[str, tk.StringVar] = {}
        self.stop_event = threading.Event()
        self.running_thread: Optional[threading.Thread] = None

        if not self.headless:
            apply_apgi_theme(self.root)
            self._process_gui_queue()
            self.root.title("APGI Open Science Framework — Empirical Protocols (EP-0 – EP-6)")
            self.root.geometry("1020x740")
            self.root.minsize(800, 560)
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self._create_menu_bar()
            self._bind_shortcuts()
            self.setup_ui()

        _proj_root = os.path.dirname(os.path.abspath(__file__))
        if _proj_root not in sys.path:
            sys.path.insert(0, _proj_root)

    # ── Queue ─────────────────────────────────────────────────────────────────

    def _process_gui_queue(self) -> None:
        try:
            while True:
                fn = self.gui_queue.get_nowait()
                if callable(fn):
                    try:
                        fn()
                    except Exception as exc:
                        logger.warning("GUI queue update failed: %s", exc)
        except queue.Empty:
            pass
        try:
            self.root.after(100, self._process_gui_queue)
        except (KeyboardInterrupt, TclError):
            logger.info("GUI processing interrupted.")
            self.stop_event.set()

    # ── Menu & shortcuts ──────────────────────────────────────────────────────

    def _create_menu_bar(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Protocol Report", command=self.export_protocol_report, accelerator="Ctrl+E")
        file_menu.add_command(
            label="Generate Pre-reg Template", command=self.generate_prereg_template, accelerator="Ctrl+G"
        )
        file_menu.add_separator()
        file_menu.add_command(label="Save Status Snapshot", command=self.save_status_snapshot, accelerator="Ctrl+S")
        file_menu.add_command(label="Load Status Snapshot", command=self.load_status_snapshot)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._on_closing, accelerator="Ctrl+Q")

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Clear Console", command=self.clear_console, accelerator="Ctrl+L")
        view_menu.add_command(label="Toggle Console", command=self._toggle_console)
        view_menu.add_command(label="Dependency Overview", command=self._show_dependency_overview)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-q>", lambda e: self._on_closing())
        self.root.bind("<Command-q>", lambda e: self._on_closing())
        self.root.bind("<Control-e>", lambda e: self.export_protocol_report())
        self.root.bind("<Control-g>", lambda e: self.generate_prereg_template())
        self.root.bind("<Control-s>", lambda e: self.save_status_snapshot())
        self.root.bind("<Control-l>", lambda e: self.clear_console())

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "APGI Open Science Framework GUI\nEmpirical Protocol Manager (EP-0 – EP-6)\nVersion 1.0.0",
        )

    # ── UI Construction ───────────────────────────────────────────────────────

    def setup_ui(self) -> None:
        """Build the application layout per APGI design spec."""
        # ── Row 0: Metric bar ─────────────────────────────────────────────────
        metric_bar = tk.Frame(self.root, bg=_BLUE, pady=8, padx=15)
        metric_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        metric_bar.columnconfigure(1, weight=1)

        tk.Label(
            metric_bar,
            text="APGI OPEN SCIENCE FRAMEWORK  —  EMPIRICAL PROTOCOLS",
            bg=_BLUE,
            fg="white",
            font=("Noto Sans", 13, "bold"),
        ).grid(row=0, column=0, sticky="w")

        # Summary counters
        self._counter_var = tk.StringVar(value=self._build_counter_text())
        tk.Label(metric_bar, textvariable=self._counter_var, bg=_BLUE, fg="white", font=("Noto Sans", 10)).grid(
            row=0, column=1, padx=(20, 15), sticky="e"
        )

        # OSF status indicator
        self._osf_status_var = tk.StringVar(value="ℹ  OSF: Ready")
        tk.Label(metric_bar, textvariable=self._osf_status_var, bg=_BLUE, fg="white", font=("Noto Sans", 10)).grid(
            row=0, column=2, sticky="e"
        )

        # ── Grid weights ──────────────────────────────────────────────────────
        self.root.columnconfigure(0, weight=0, minsize=220)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=0)
        self.root.rowconfigure(3, weight=0, minsize=160)

        # ── Row 1, Col 0: Sidebar ─────────────────────────────────────────────
        sidebar = tk.Frame(self.root, bg=_SURFACE, width=220, bd=0)
        sidebar.grid(row=1, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)

        tk.Label(
            sidebar, text="EMPIRICAL PROTOCOLS", bg=_BORDER, fg=_MUTED, font=("Noto Sans", 9, "bold"), pady=5
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        lb_frame = tk.Frame(sidebar, bg=_SURFACE)
        lb_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")
        lb_frame.columnconfigure(0, weight=1)
        lb_frame.rowconfigure(0, weight=1)

        self.protocol_listbox = tk.Listbox(
            lb_frame,
            font=("Noto Sans", 9),
            bg=_SURFACE,
            fg=_FG,
            selectbackground=_BLUE,
            selectforeground="white",
            borderwidth=0,
            highlightthickness=1,
            highlightcolor=_BORDER,
            highlightbackground=_BORDER,
            activestyle="none",
            cursor="hand2",
        )
        lb_scroll = ttk.Scrollbar(lb_frame, orient="vertical", command=self.protocol_listbox.yview)
        self.protocol_listbox.configure(yscrollcommand=lb_scroll.set)
        self.protocol_listbox.grid(row=0, column=0, sticky="nsew")
        lb_scroll.grid(row=0, column=1, sticky="ns")

        for key in EP_PROTOCOLS:
            ep_id = EP_PROTOCOLS[key]["id"]
            short_title = EP_PROTOCOLS[key]["title"]
            self.protocol_listbox.insert(tk.END, f"  {ep_id}  {short_title}")

        self.protocol_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        # Sidebar action buttons
        btn_area = tk.Frame(sidebar, bg=_SURFACE, pady=8, padx=8)
        btn_area.grid(row=2, column=0, columnspan=2, sticky="ew")
        btn_area.columnconfigure(0, weight=1)

        self._gen_prereg_btn = ttk.Button(
            btn_area,
            text="✎  Generate Pre-reg",
            command=self.generate_prereg_template,
            style="Primary.TButton",
            state=tk.DISABLED,
        )
        self._gen_prereg_btn.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        self._export_btn = ttk.Button(
            btn_area,
            text="Export Report",
            command=self.export_protocol_report,
            style="Secondary.TButton",
        )
        self._export_btn.grid(row=1, column=0, sticky="ew", pady=(0, 4))

        ttk.Separator(btn_area, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=(0, 6))

        ttk.Button(btn_area, text="Dependency Overview", command=self._show_dependency_overview).grid(
            row=3, column=0, sticky="ew", pady=(0, 4)
        )
        ttk.Button(btn_area, text="Clear Console", command=self.clear_console).grid(row=4, column=0, sticky="ew")

        # Right border divider
        tk.Frame(self.root, bg=_BORDER, width=1).grid(row=1, column=0, sticky="nse")

        # ── Row 1, Col 1: Workspace ───────────────────────────────────────────
        workspace = ttk.Frame(self.root, padding=(15, 10, 15, 10))
        workspace.grid(row=1, column=1, sticky="nsew")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=0)  # cards row
        workspace.rowconfigure(1, weight=1)  # detail/notes row

        self._card_area = ttk.Frame(workspace)
        self._card_area.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self._card_area.columnconfigure(0, weight=1)
        self._show_workspace_empty_state()

        # Detail / status panel
        self._detail_frame = ttk.LabelFrame(workspace, text="PROTOCOL DETAILS & STATUS", padding=(10, 6))
        self._detail_frame.grid(row=1, column=0, sticky="nsew")
        self._detail_frame.columnconfigure(0, weight=1)
        self._detail_frame.rowconfigure(0, weight=1)

        self._detail_inner = ttk.Frame(self._detail_frame)
        self._detail_inner.grid(row=0, column=0, sticky="nsew")
        self._detail_inner.columnconfigure(0, weight=1)
        self._detail_inner.rowconfigure(0, weight=1)

        ttk.Label(
            self._detail_inner,
            text="Select a protocol from the sidebar to view details and manage pre-registration.",
            foreground=_MUTED,
            font=("Noto Sans", 10),
        ).grid(row=0, column=0, padx=20, pady=20)

        # ── Row 2: Console toggle header ──────────────────────────────────────
        console_header = tk.Frame(self.root, bg=_BORDER, pady=3)
        console_header.grid(row=2, column=0, columnspan=2, sticky="ew")

        self._console_toggle_btn = tk.Button(
            console_header,
            text="▼  OUTPUT CONSOLE",
            bg=_BORDER,
            fg=_MUTED,
            font=("Noto Sans", 9, "bold"),
            relief="flat",
            cursor="hand2",
            command=self._toggle_console,
            activebackground=_BORDER,
            activeforeground="#495057",
            bd=0,
        )
        self._console_toggle_btn.pack(side="left", padx=10)

        # ── Row 3: Console body ───────────────────────────────────────────────
        self.root.rowconfigure(3, weight=0, minsize=160)
        self._console_frame = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        self._console_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self._console_frame.columnconfigure(0, weight=1)
        self._console_frame.rowconfigure(0, weight=1)

        self.console = scrolledtext.ScrolledText(
            self._console_frame,
            height=8,
            font=("Noto Sans Mono", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.console.grid(row=0, column=0, sticky="nsew")

        self._log("OSF Protocol Manager initialised. Select a protocol to begin.")
        self._log("7 protocols loaded  |  All status: Not started  |  Pre-reg: Pending")

    # ── Workspace helpers ─────────────────────────────────────────────────────

    def _show_workspace_empty_state(self) -> None:
        for w in self._card_area.winfo_children():
            w.destroy()

        frame = ttk.Frame(self._card_area, padding=30)
        frame.grid(row=0, column=0)
        tk.Canvas(frame, width=200, height=80, bg=_BG, highlightbackground=_SOFT, highlightthickness=2).pack()
        ttk.Label(
            frame,
            text="No protocol selected.  Choose one from the sidebar to view metadata and manage pre-registration.",
            wraplength=400,
            font=("Noto Sans", 11),
            foreground=_MUTED,
        ).pack(pady=(15, 0))

    def _show_protocol_cards(self, key: str) -> None:
        """Populate the card area with summary cards for the selected protocol."""
        for w in self._card_area.winfo_children():
            w.destroy()

        meta = EP_PROTOCOLS[key]
        st = self._protocol_status[key]

        cards_row = ttk.Frame(self._card_area)
        cards_row.grid(row=0, column=0, sticky="ew")
        for i in range(4):
            cards_row.columnconfigure(i, weight=1)

        APGICard(cards_row, "Protocol", meta["id"], f"Type: {meta['type']}  |  Priority: {meta['priority']}").grid(
            row=0, column=0, padx=(0, 6), pady=4, sticky="ew"
        )

        prereq_text = ", ".join(meta["depends_on"]) if meta["depends_on"] else "None"
        APGICard(cards_row, "Dependencies", prereq_text, "Must be completed before this protocol").grid(
            row=0, column=1, padx=(0, 6), pady=4, sticky="ew"
        )

        APGICard(cards_row, "Study Status", st["status"], "Update via the status panel below").grid(
            row=0, column=2, padx=(0, 6), pady=4, sticky="ew"
        )

        APGICard(
            cards_row,
            "Pre-registration",
            st["prereg_status"],
            f"Platform: {meta['platform']}  |  Required: {'Yes' if meta['prereg_required'] else 'No'}",
        ).grid(row=0, column=3, padx=0, pady=4, sticky="ew")

    def _populate_detail_panel(self, key: str) -> None:
        """Build the scrollable detail/status editor for the selected protocol."""
        for w in self._detail_inner.winfo_children():
            w.destroy()

        meta = EP_PROTOCOLS[key]
        st = self._protocol_status[key]

        # Scrollable canvas
        canvas = tk.Canvas(self._detail_inner, bg=_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self._detail_inner, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self._detail_inner.rowconfigure(0, weight=1)
        self._detail_inner.columnconfigure(0, weight=1)

        scrollable = ttk.Frame(canvas, padding=(10, 6))
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        scrollable.columnconfigure(1, weight=1)

        def _row(r: int, label: str, value: str, mono: bool = False) -> None:
            ttk.Label(scrollable, text=label + ":", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
                row=r, column=0, sticky="nw", padx=(0, 12), pady=2
            )
            fnt = ("Noto Sans Mono", 9) if mono else ("Noto Sans", 10)
            ttk.Label(scrollable, text=value, font=fnt, foreground=_FG, wraplength=560, justify="left").grid(
                row=r, column=1, sticky="nw", pady=2
            )

        row_idx = 0

        # Title
        ttk.Label(scrollable, text=meta["title"], font=("Noto Sans", 13, "bold"), foreground=_FG).grid(
            row=row_idx, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        row_idx += 1

        _row(row_idx, "Description", meta["description"])
        row_idx += 1
        _row(row_idx, "Study Type", meta["type"])
        row_idx += 1
        _row(row_idx, "Priority", _PRIORITY_LABELS.get(meta["priority"], str(meta["priority"])))
        row_idx += 1
        _row(row_idx, "Sample Size", meta["sample_size"])
        row_idx += 1
        _row(row_idx, "Key Measures", "  •  " + "\n  •  ".join(meta["measures"]))
        row_idx += 1
        _row(row_idx, "Analysis Plan", meta["analysis"])
        row_idx += 1
        _row(row_idx, "Falsification Criterion", meta["falsification_criterion"], mono=True)
        row_idx += 1

        if meta["notes"]:
            _row(row_idx, "Notes", meta["notes"])
            row_idx += 1

        ttk.Separator(scrollable, orient="horizontal").grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=8)
        row_idx += 1

        # ── Editable status fields ────────────────────────────────────────────
        ttk.Label(scrollable, text="UPDATE STATUS", font=("Noto Sans", 10, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        row_idx += 1

        # Study status
        ttk.Label(scrollable, text="Study Status:", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, sticky="w", pady=2
        )
        status_var = tk.StringVar(value=st["status"])
        self._detail_vars["status"] = status_var
        status_cb = ttk.Combobox(
            scrollable, textvariable=status_var, values=_STATUS_OPTIONS, state="readonly", width=28
        )
        status_cb.grid(row=row_idx, column=1, sticky="w", pady=2)
        row_idx += 1

        # Pre-reg status
        ttk.Label(scrollable, text="Pre-reg Status:", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, sticky="w", pady=2
        )
        prereg_var = tk.StringVar(value=st["prereg_status"])
        self._detail_vars["prereg_status"] = prereg_var
        prereg_cb = ttk.Combobox(
            scrollable, textvariable=prereg_var, values=_PREREG_OPTIONS, state="readonly", width=28
        )
        prereg_cb.grid(row=row_idx, column=1, sticky="w", pady=2)
        row_idx += 1

        # Notes
        ttk.Label(scrollable, text="Notes:", font=("Noto Sans", 9, "bold"), foreground=_MUTED).grid(
            row=row_idx, column=0, sticky="nw", pady=2
        )
        notes_text = tk.Text(
            scrollable,
            height=3,
            width=55,
            font=("Noto Sans", 10),
            wrap=tk.WORD,
            bg=_SURFACE,
            fg=_FG,
            relief="solid",
            bd=1,
        )
        notes_text.insert("1.0", st["notes"])
        notes_text.grid(row=row_idx, column=1, sticky="ew", pady=2)
        self._notes_widget = notes_text
        row_idx += 1

        # Save button
        def _save() -> None:
            self._protocol_status[key]["status"] = status_var.get()
            self._protocol_status[key]["prereg_status"] = prereg_var.get()
            self._protocol_status[key]["notes"] = notes_text.get("1.0", tk.END).strip()
            self._show_protocol_cards(key)
            self._counter_var.set(self._build_counter_text())
            self._log(f"[{meta['id']}] Status saved: {status_var.get()} | Pre-reg: {prereg_var.get()}")
            messagebox.showinfo("Saved", f"{meta['id']} status updated.")

        ttk.Button(scrollable, text="Save Changes", command=_save, style="Primary.TButton").grid(
            row=row_idx, column=1, sticky="w", pady=8
        )

    # ── Sidebar selection ─────────────────────────────────────────────────────

    def _on_listbox_select(self, _event: Any = None) -> None:
        sel = self.protocol_listbox.curselection()
        if not sel:
            return
        key = list(EP_PROTOCOLS.keys())[sel[0]]
        self._selected_protocol = key
        self._gen_prereg_btn.configure(state=tk.NORMAL)
        meta = EP_PROTOCOLS[key]
        self._log(f"Selected: {meta['id']} — {meta['title']}")
        self._show_protocol_cards(key)
        self._populate_detail_panel(key)

    # ── Counter text ──────────────────────────────────────────────────────────

    def _build_counter_text(self) -> str:
        total = len(EP_PROTOCOLS)
        registered = sum(1 for s in self._protocol_status.values() if s["prereg_status"] == "Registered")
        completed = sum(1 for s in self._protocol_status.values() if s["status"] == "Completed")
        return f"Protocols: {total}  |  Pre-registered: {registered}/{total}  |  Completed: {completed}/{total}"

    # ── Dependency Overview ───────────────────────────────────────────────────

    def _show_dependency_overview(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Dependency Overview — APGI Empirical Protocols")
        win.geometry("560x480")
        win.configure(bg=_BG)
        win.resizable(True, True)

        tk.Label(
            win, text="PROTOCOL DEPENDENCY GRAPH", bg=_BLUE, fg="white", font=("Noto Sans", 12, "bold"), pady=8
        ).pack(fill="x", padx=0)

        frame = ttk.Frame(win, padding=15)
        frame.pack(fill="both", expand=True)

        lines: List[str] = []
        lines.append("Execution Order & Dependencies\n")
        lines.append("=" * 48)
        for key, meta in EP_PROTOCOLS.items():
            dep_str = " ← requires: " + ", ".join(meta["depends_on"]) if meta["depends_on"] else " (no dependencies)"
            st = self._protocol_status[key]["status"]
            prereg = self._protocol_status[key]["prereg_status"]
            lines.append(
                f"\n  {meta['id']:6s}  {meta['title']}"
                f"\n          Type: {meta['type']}  |  Priority: {meta['priority']}"
                f"\n          Depends on:{dep_str}"
                f"\n          Status: {st}  |  Pre-reg: {prereg}"
            )
        lines.append("\n" + "=" * 48)

        text = scrolledtext.ScrolledText(
            frame, font=("Noto Sans Mono", 9), bg=_SURFACE, fg=_FG, state=tk.DISABLED, wrap=tk.WORD
        )
        text.pack(fill="both", expand=True)
        text.configure(state=tk.NORMAL)
        text.insert(tk.END, "\n".join(lines))
        text.configure(state=tk.DISABLED)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(8, 0))

    # ── Pre-registration template ─────────────────────────────────────────────

    def generate_prereg_template(self) -> None:
        key = self._selected_protocol
        if not key:
            messagebox.showwarning("No Protocol Selected", "Please select a protocol from the sidebar first.")
            return

        meta = EP_PROTOCOLS[key]
        st = self._protocol_status[key]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        lines: List[str] = [
            "=" * 64,
            "APGI PRE-REGISTRATION TEMPLATE",
            f"Generated: {timestamp}",
            "=" * 64,
            "",
            f"Protocol ID:        {meta['id']}",
            f"Title:              {meta['title']}",
            f"Study Type:         {meta['type']}",
            f"Priority:           {_PRIORITY_LABELS.get(meta['priority'], str(meta['priority']))}",
            f"Platform:           {meta['platform']}",
            f"Current Status:     {st['status']}",
            f"Pre-reg Status:     {st['prereg_status']}",
            "",
            "─" * 64,
            "1. RESEARCH QUESTION & HYPOTHESES",
            "─" * 64,
            "",
            meta["description"],
            "",
            "─" * 64,
            "2. EXPERIMENTAL DESIGN",
            "─" * 64,
            "",
            f"Sample Size:  {meta['sample_size']}",
            "",
            "Key Measures:",
        ]
        for m in meta["measures"]:
            lines.append(f"  • {m}")

        lines += [
            "",
            "─" * 64,
            "3. ANALYSIS PLAN",
            "─" * 64,
            "",
            meta["analysis"],
            "",
            "─" * 64,
            "4. FALSIFICATION CRITERIA",
            "─" * 64,
            "",
            meta["falsification_criterion"],
            "",
            "─" * 64,
            "5. DEPENDENCIES",
            "─" * 64,
            "",
        ]

        if meta["depends_on"]:
            for dep in meta["depends_on"]:
                dep_key = next((k for k in EP_PROTOCOLS if EP_PROTOCOLS[k]["id"] == dep), None)
                dep_title = EP_PROTOCOLS[dep_key]["title"] if dep_key else dep
                lines.append(f"  Requires: {dep} — {dep_title}")
        else:
            lines.append("  None — this protocol has no upstream dependencies.")

        lines += [
            "",
            "─" * 64,
            "6. NOTES",
            "─" * 64,
            "",
            st["notes"] if st["notes"] else "[No notes recorded]",
            "",
            "=" * 64,
            "END OF PRE-REGISTRATION TEMPLATE",
            "=" * 64,
        ]

        template_text = "\n".join(lines)

        # Show in a new window and offer save
        win = tk.Toplevel(self.root)
        win.title(f"Pre-registration Template — {meta['id']}")
        win.geometry("680x560")
        win.configure(bg=_BG)

        tk.Label(
            win,
            text=f"PRE-REGISTRATION TEMPLATE  —  {meta['id']}",
            bg=_BLUE,
            fg="white",
            font=("Noto Sans", 12, "bold"),
            pady=8,
        ).pack(fill="x")

        txt = scrolledtext.ScrolledText(win, font=("Noto Sans Mono", 9), bg=_SURFACE, fg=_FG, wrap=tk.WORD)
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        txt.insert(tk.END, template_text)

        def _save_file() -> None:
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"prereg_{meta['id'].replace('-', '_')}.txt",
            )
            if path:
                Path(path).write_text(template_text, encoding="utf-8")
                self._log(f"Pre-reg template saved: {path}")
                messagebox.showinfo("Saved", f"Template saved to:\n{path}")

        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Save to File…", command=_save_file, style="Primary.TButton").pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side="left")

        self._log(f"[{meta['id']}] Pre-registration template generated.")

    # ── Export report ─────────────────────────────────────────────────────────

    def export_protocol_report(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines: List[str] = [
            "=" * 68,
            "APGI OPEN SCIENCE FRAMEWORK — EMPIRICAL PROTOCOL STATUS REPORT",
            f"Generated: {timestamp}",
            "=" * 68,
            "",
        ]

        for key, meta in EP_PROTOCOLS.items():
            st = self._protocol_status[key]
            dep_str = ", ".join(meta["depends_on"]) if meta["depends_on"] else "None"
            lines += [
                f"{'─' * 68}",
                f"  {meta['id']}  |  {meta['title']}",
                f"  Type: {meta['type']}   Priority: {meta['priority']}   Platform: {meta['platform']}",
                f"  Depends on: {dep_str}",
                f"  Study status:    {st['status']}",
                f"  Pre-reg status:  {st['prereg_status']}",
                f"  Notes: {st['notes'] if st['notes'] else '—'}",
                "",
            ]

        lines += [
            "=" * 68,
            "END OF REPORT",
            "=" * 68,
        ]

        report_text = "\n".join(lines)
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="apgi_osf_protocol_report.txt",
        )
        if not path:
            return

        if path.endswith(".json"):
            data = {
                "generated": timestamp,
                "protocols": {key: {**EP_PROTOCOLS[key], **self._protocol_status[key]} for key in EP_PROTOCOLS},
            }
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            Path(path).write_text(report_text, encoding="utf-8")

        self._log(f"Protocol report exported: {path}")
        messagebox.showinfo("Exported", f"Report saved to:\n{path}")

    # ── Status snapshot (persistence) ────────────────────────────────────────

    def save_status_snapshot(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="osf_status_snapshot.json",
        )
        if not path:
            return
        Path(path).write_text(json.dumps(self._protocol_status, indent=2), encoding="utf-8")
        self._log(f"Status snapshot saved: {path}")

    def load_status_snapshot(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            for key in EP_PROTOCOLS:
                if key in data:
                    for field in ("status", "prereg_status", "notes"):
                        if field in data[key]:
                            self._protocol_status[key][field] = data[key][field]
            self._counter_var.set(self._build_counter_text())
            self._log(f"Status snapshot loaded: {path}")
            # Refresh workspace if a protocol is selected
            if self._selected_protocol:
                self._show_protocol_cards(self._selected_protocol)
                self._populate_detail_panel(self._selected_protocol)
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            messagebox.showerror("Load Failed", f"Could not load snapshot:\n{exc}")

    # ── Console ───────────────────────────────────────────────────────────────

    def _log(self, message: str) -> None:
        if self.headless:
            logger.info(message)
            return

        def _do() -> None:
            self.console.configure(state=tk.NORMAL)
            ts = datetime.now().strftime("%H:%M:%S")
            self.console.insert(tk.END, f"[{ts}]  {message}\n")
            self.console.see(tk.END)
            self.console.configure(state=tk.DISABLED)

        try:
            self.gui_queue.put_nowait(_do)
        except queue.Full:
            pass

    def clear_console(self) -> None:
        self.console.configure(state=tk.NORMAL)
        self.console.delete("1.0", tk.END)
        self.console.configure(state=tk.DISABLED)

    def _toggle_console(self) -> None:
        if self._console_visible:
            self._console_frame.grid_remove()
            self.root.rowconfigure(3, minsize=0)
            self._console_toggle_btn.configure(text="▶  OUTPUT CONSOLE")
            self._console_visible = False
        else:
            self._console_frame.grid()
            self.root.rowconfigure(3, minsize=160)
            self._console_toggle_btn.configure(text="▼  OUTPUT CONSOLE")
            self._console_visible = True

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _on_closing(self) -> None:
        self.stop_event.set()
        if self.running_thread and self.running_thread.is_alive():
            self.running_thread.join(timeout=2.0)
        self.root.destroy()


# ── Entry Point ───────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="APGI Open Science Framework GUI")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (for testing)")
    args = parser.parse_args()

    if args.headless:
        _ = OSFProtocolGUI(tk.Tk(), headless=True)
        print("OSF GUI initialised in headless mode.")
        print(f"Protocols loaded: {list(EP_PROTOCOLS.keys())}")
        return

    root = tk.Tk()
    _ = OSFProtocolGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
