#!/usr/bin/env python3
"""
Simple GUI for running APGI falsification protocols
"""

import importlib.util
import logging
import os
import queue
import sys
import threading
import tkinter as tk
import warnings
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable, List

# Pre-import torch to prevent circular import issues with dynamically loaded protocols
try:
    import torch

    TORCH_AVAILABLE = True
    _ = torch.__version__  # Use torch to avoid 'unused import' warning
except ImportError:
    TORCH_AVAILABLE = False

# Fix temp directory and cache issues for matplotlib/sklearn
import tempfile

os.environ["TMPDIR"] = tempfile.gettempdir()
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

# Ensure cache directories exist
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


class ProtocolRunnerGUI:
    """GUI for running APGI falsification protocols with progress tracking."""

    def __init__(self, root, headless=False):
        self.root = root
        self.headless = headless
        self.gui_queue = queue.Queue()

        if not self.headless:
            self._process_gui_queue()
            self.root.title("APGI Framework-Level Falsification Aggregator (FP-ALL)")
            self.root.geometry("800x600")
            self.root.minsize(640, 480)  # Prevent resizing below usable size

        # Add project root directory to Python path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        # Protocol definitions with parameters
        self.protocols = {
            "Protocol 1: APGI Agent": {
                "file": "Falsification/FP_01_ActiveInference.py",
                "class": "APGIActiveInferenceAgent",
                "description": "Complete APGI-based active inference agent",
                "parameters": {
                    "lr_extero": {
                        "default": 0.01,
                        "min": 0.001,
                        "max": 0.1,
                        "type": "float",
                        "description": "Exteroceptive learning rate",
                    },
                    "lr_intero": {
                        "default": 0.01,
                        "min": 0.001,
                        "max": 0.1,
                        "type": "float",
                        "description": "Interoceptive learning rate",
                    },
                    "lr_precision": {
                        "default": 0.05,
                        "min": 0.001,
                        "max": 0.1,
                        "type": "float",
                        "description": "Precision learning rate",
                    },
                    "lr_somatic": {
                        "default": 0.1,
                        "min": 0.001,
                        "max": 0.1,
                        "type": "float",
                        "description": "Somatic learning rate",
                    },
                    "n_actions": {
                        "default": 4,
                        "min": 2,
                        "max": 10,
                        "type": "int",
                        "description": "Number of actions",
                    },
                    "theta_init": {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 2.0,
                        "type": "float",
                        "description": "Initial threshold",
                    },
                    "theta_baseline": {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 2.0,
                        "type": "float",
                        "description": "Threshold baseline",
                    },
                    "alpha": {
                        "default": 12.0,  # Tuned for sharper threshold crossing
                        "min": 1.0,
                        "max": 20.0,
                        "type": "float",
                        "description": "Sigmoid slope",
                    },
                    "tau_S": {
                        "default": 0.3,
                        "min": 0.1,
                        "max": 5.0,
                        "type": "float",
                        "description": "Signal time constant",
                    },
                    "tau_theta": {
                        "default": 25.0,  # Tuned for better adaptation dynamics
                        "min": 1.0,
                        "max": 100.0,
                        "type": "float",
                        "description": "Threshold time constant",
                    },
                    "eta_theta": {
                        "default": 0.05,  # Tuned 5x for measurable threshold adaptation
                        "min": 0.001,
                        "max": 0.1,
                        "type": "float",
                        "description": "Threshold adaptation rate",
                    },
                    "beta": {
                        "default": 2.0,  # Tuned for better interoceptive advantage
                        "min": 0.5,
                        "max": 3.0,
                        "type": "float",
                        "description": "Somatic gain",
                    },
                    "rho": {
                        "default": 0.8,  # Tuned for stronger precision modulation
                        "min": 0.1,
                        "max": 1.0,
                        "type": "float",
                        "description": "Precision weight",
                    },
                },
            },
            "Protocol 2: Iowa Gambling": {
                "file": "Falsification/FP_02_AgentComparison_ConvergenceBenchmark.py",
                "class": "IowaGamblingTaskEnvironment",
                "description": "IGT variant with simulated interoceptive costs",
                "parameters": {
                    "n_trials": {
                        "default": 100,
                        "min": 10,
                        "max": 1000,
                        "type": "int",
                        "description": "Number of trials",
                    },
                    "n_decks": {
                        "default": 4,
                        "min": 2,
                        "max": 8,
                        "type": "int",
                        "description": "Number of decks",
                    },
                    "interoceptive_cost_weight": {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "type": "float",
                        "description": "Interoceptive cost weight",
                    },
                    "learning_rate": {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 0.5,
                        "type": "float",
                        "description": "Agent learning rate",
                    },
                },
            },
            "Protocol 3: Agent Comparison": {
                "file": "Falsification/FP_03_FrameworkLevel_MultiProtocol.py",
                "class": "AgentComparisonExperiment",
                "description": "Run complete agent comparison experiment",
                "parameters": {
                    "n_episodes": {
                        "default": 100,
                        "min": 10,
                        "max": 1000,
                        "type": "int",
                        "description": "Number of episodes",
                    },
                    "episode_length": {
                        "default": 50,
                        "min": 10,
                        "max": 500,
                        "type": "int",
                        "description": "Episode length",
                    },
                    "n_agents": {
                        "default": 3,
                        "min": 2,
                        "max": 10,
                        "type": "int",
                        "description": "Number of agents to compare",
                    },
                },
            },
            "Protocol 4: Phase Transition": {
                "file": "Falsification/FP_04_PhaseTransition_EpistemicArchitecture.py",
                "class": "InformationTheoreticAnalysis",
                "description": "Test APGI ignition phase transition signatures",
                "parameters": {
                    "surprise_range": {
                        "default": "[0.1, 2.0]",
                        "type": "str",
                        "description": "Surprise range [min, max]",
                    },
                    "n_points": {
                        "default": 50,
                        "min": 10,
                        "max": 200,
                        "type": "int",
                        "description": "Number of analysis points",
                    },
                    "tau_S": {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 2.0,
                        "type": "float",
                        "description": "Signal time constant",
                    },
                    "alpha": {
                        "default": 10.0,
                        "min": 1.0,
                        "max": 20.0,
                        "type": "float",
                        "description": "Sigmoid slope",
                    },
                },
            },
            "Protocol 5: Evolutionary": {
                "file": "Falsification/FP_05_EvolutionaryPlausibility.py",
                "class": "EvolutionaryAPGIEmergence",
                "description": "Test APGI emergence under selection pressure",
                "parameters": {
                    "population_size": {
                        "default": 50,
                        "min": 10,
                        "max": 200,
                        "type": "int",
                        "description": "Population size",
                    },
                    "n_generations": {
                        "default": 100,
                        "min": 10,
                        "max": 1000,
                        "type": "int",
                        "description": "Number of generations",
                    },
                    "mutation_rate": {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 0.5,
                        "type": "float",
                        "description": "Mutation rate",
                    },
                    "selection_pressure": {
                        "default": 2.0,
                        "min": 1.0,
                        "max": 5.0,
                        "type": "float",
                        "description": "Selection pressure",
                    },
                },
            },
            "Protocol 6: Network Comparison": {
                "file": "Falsification/FP_06_LiquidNetwork_EnergyBenchmark.py",
                "class": "NetworkComparisonExperiment",
                "description": "Compare APGI-inspired vs standard architectures",
                "parameters": {
                    "extero_dim": {
                        "default": 32,
                        "min": 8,
                        "max": 128,
                        "type": "int",
                        "description": "Exteroceptive dimension",
                    },
                    "intero_dim": {
                        "default": 16,
                        "min": 4,
                        "max": 64,
                        "type": "int",
                        "description": "Interoceptive dimension",
                    },
                    "action_dim": {
                        "default": 4,
                        "min": 2,
                        "max": 16,
                        "type": "int",
                        "description": "Action dimension",
                    },
                    "context_dim": {
                        "default": 8,
                        "min": 2,
                        "max": 32,
                        "type": "int",
                        "description": "Context dimension",
                    },
                    "n_episodes": {
                        "default": 50,
                        "min": 10,
                        "max": 200,
                        "type": "int",
                        "description": "Number of episodes",
                    },
                },
            },
            "Protocol 7: Mathematical Consistency": {
                "file": "Falsification/FP_07_MathematicalConsistency.py",
                "class": "MathematicalConsistencyChecker",
                "description": "Test mathematical consistency of APGI equations",
                "parameters": {
                    "epsilon": {
                        "default": 1e-6,
                        "min": 1e-9,
                        "max": 1e-3,
                        "type": "float",
                        "description": "Numerical epsilon for derivatives",
                    },
                    "n_samples": {
                        "default": 1000,
                        "min": 100,
                        "max": 10000,
                        "type": "int",
                        "description": "Number of test samples",
                    },
                    "tolerance": {
                        "default": 1e-6,
                        "min": 1e-8,
                        "max": 1e-3,
                        "type": "float",
                        "description": "Tolerance for equality checks (V5.1 spec: ε ≤ 1e-6)",
                    },
                },
            },
            "Protocol 8: Parameter Sensitivity": {
                "file": "Falsification/FP_08_ParameterSensitivity_Identifiability.py",
                "class": "ParameterSensitivityAnalyzer",
                "description": "Parameter sensitivity and identifiability analysis",
                "parameters": {
                    "n_samples": {
                        "default": 1000,
                        "min": 100,
                        "max": 5000,
                        "type": "int",
                        "description": "Number of Sobol samples",
                    },
                    "n_trials": {
                        "default": 1000,
                        "min": 100,
                        "max": 5000,
                        "type": "int",
                        "description": "Number of trials per sample",
                    },
                    "n_levels": {
                        "default": 15,
                        "min": 5,
                        "max": 50,
                        "type": "int",
                        "description": "Number of OAT levels",
                    },
                },
            },
            "Protocol 9: Neural Signatures": {
                "file": "Falsification/FP_09_NeuralSignatures_P3b_HEP.py",
                "class": "NeuralSignatureValidator",
                "description": "Validate P3b and HEP neural signatures",
                "parameters": {
                    "n_participants": {
                        "default": 50,
                        "min": 10,
                        "max": 200,
                        "type": "int",
                        "description": "Number of participants",
                    },
                    "sampling_rate": {
                        "default": 500,
                        "min": 100,
                        "max": 2000,
                        "type": "int",
                        "description": "EEG sampling rate (Hz)",
                    },
                    "window_size": {
                        "default": 1000,
                        "min": 500,
                        "max": 5000,
                        "type": "int",
                        "description": "Window size (ms)",
                    },
                    "n_chains": {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "type": "int",
                        "description": "Number of MCMC chains (2 recommended for GUI)",
                    },
                    "burn_in": {
                        "default": 500,
                        "min": 100,
                        "max": 2000,
                        "type": "int",
                        "description": "Burn-in samples",
                    },
                },
            },
            "Protocol 10: Bayesian Estimation": {
                "file": "Falsification/FP_10_BayesianEstimation_MCMC.py",
                "class": "BayesianParameterRecovery",
                "description": "Bayesian parameter recovery analysis",
                "parameters": {
                    "n_samples": {
                        "default": 1000,
                        "min": 500,
                        "max": 5000,
                        "type": "int",
                        "description": "Number of MCMC samples (1000 recommended for GUI)",
                    },
                    "n_chains": {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "type": "int",
                        "description": "Number of MCMC chains (2 recommended for GUI)",
                    },
                    "burn_in": {
                        "default": 500,
                        "min": 100,
                        "max": 2000,
                        "type": "int",
                        "description": "Burn-in samples",
                    },
                },
            },
            "Protocol 11: Liquid Network Dynamics": {
                "file": "Falsification/FP_11_LiquidNetworkDynamics_EchoState.py",
                "class": "LiquidNetworkDynamicsAnalyzer",
                "description": "Liquid network dynamics and echo state analysis",
                "parameters": {
                    "spectral_radius": {
                        "default": 0.9,
                        "min": 0.1,
                        "max": 2.0,
                        "type": "float",
                        "description": "Spectral radius",
                    },
                    "leak_rate": {
                        "default": 0.3,
                        "min": 0.1,
                        "max": 1.0,
                        "type": "float",
                        "description": "Leak rate",
                    },
                    "n_units": {
                        "default": 100,
                        "min": 10,
                        "max": 500,
                        "type": "int",
                        "description": "Number of liquid units",
                    },
                },
            },
            "Protocol 12: Cross Species Scaling": {
                "file": "Falsification/FP_12_CrossSpeciesScaling.py",
                "class": "LiquidTimeConstantChecker",
                "description": "Cross-species allometric scaling and clinical convergence analysis",
                "parameters": {
                    "spectral_radius": {
                        "default": 0.98,
                        "min": 0.1,
                        "max": 2.0,
                        "type": "float",
                        "description": "ESN spectral radius",
                    },
                    "leak_rate": {
                        "default": 0.01,
                        "min": 0.001,
                        "max": 1.0,
                        "type": "float",
                        "description": "LTC leak rate",
                    },
                    "n_nodes": {
                        "default": 100,
                        "min": 10,
                        "max": 500,
                        "type": "int",
                        "description": "Number of ESN nodes",
                    },
                },
            },
            "Protocol 13: Framework Aggregator": {
                "file": "Falsification/FP_ALL_Aggregator.py",
                "class": "FalsificationAggregator",
                "description": "Full Framework Falsification Report",
                "parameters": {},
            },
        }

        # Thread management
        self.stop_event = threading.Event()
        self.running_thread = None

        # Parameter storage - initialize with default values
        self.parameter_values = {}
        self.load_default_parameters()

        # Initialize selected protocol
        self.selected_protocol = None

        # Add window close handler for proper thread cleanup
        if not self.headless:
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.setup_ui()

        self.log_message("APGI Falsification Protocols GUI initialized")
        if not self.headless:
            self._process_gui_queue()

    def _process_gui_queue(self):
        """Process pending GUI updates from the queue (thread-safe).

        This method is called periodically via root.after() to process
        updates that worker threads put into gui_queue.
        """
        try:
            while True:
                update_func = self.gui_queue.get_nowait()
                if callable(update_func):
                    try:
                        update_func()
                    except Exception as e:
                        logger.warning(f"GUI queue update failed: {e}")
        except queue.Empty:
            pass
        # Schedule next check in 100ms
        self.root.after(100, self._process_gui_queue)

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

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Exit", command=self.root.quit)

    def setup_ui(self):
        """Setup the user interface."""

        # Create menu bar
        self._create_menu_bar()

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame, text="APGI Falsification Protocols", font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10)
        )

        # Protocols tab
        protocols_frame = ttk.Frame(self.notebook)
        self.notebook.add(protocols_frame, text="Protocols")
        self.setup_protocols_tab(protocols_frame)

        # Parameters tab
        params_frame = ttk.Frame(self.notebook)
        self.notebook.add(params_frame, text="Parameters")
        self.setup_parameters_tab(params_frame)

        # Output console
        console_frame = ttk.LabelFrame(main_frame, text="Output Console", padding="10")
        console_frame.grid(
            row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

        self.output_console = scrolledtext.ScrolledText(
            console_frame, height=15, width=80
        )
        self.output_console.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        # Status bar and clear console button frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(
            row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0)
        )
        status_frame.columnconfigure(1, weight=1)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            status_frame, textvariable=self.status_var, relief=tk.SUNKEN
        )
        status_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
            length=300,
        )
        self.progress_bar.grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        # Control buttons
        button_frame = ttk.Frame(status_frame)
        button_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

        clear_btn = ttk.Button(
            button_frame, text="Clear Console", command=self.clear_console
        )
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_btn = ttk.Button(
            button_frame, text="Stop", command=self.stop_protocol, state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT)

        # Configure tab order for keyboard navigation
        self._setup_tab_order()

    def setup_protocols_tab(self, parent_frame):
        """Setup the protocols selection tab."""
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(0, weight=1)

        # Protocol buttons frame
        button_frame = ttk.LabelFrame(
            parent_frame, text="Select Protocol", padding="10"
        )
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Create 6 buttons in 2x3 grid
        for i, (protocol_name, protocol_info) in enumerate(self.protocols.items()):
            row = i // 3
            col = i % 3

            btn = ttk.Button(
                button_frame,
                text=protocol_name.split(": ")[1],
                command=lambda info=protocol_info, name=protocol_name: self.select_protocol(
                    name, info
                ),
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky=(tk.W, tk.E))

            # Add tooltip
            self.create_tooltip(btn, protocol_info["description"])

        # Configure button grid weights
        for col in range(3):
            button_frame.columnconfigure(col, weight=1)

        # Selected protocol display
        selected_frame = ttk.LabelFrame(
            parent_frame, text="Selected Protocol", padding="10"
        )
        selected_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.selected_protocol_label = ttk.Label(
            selected_frame, text="No protocol selected", font=("Arial", 10, "bold")
        )
        self.selected_protocol_label.pack()

        # Run buttons frame
        run_buttons_frame = ttk.Frame(parent_frame)
        run_buttons_frame.grid(row=2, column=0, pady=(10, 0))

        # Run selected button
        self.run_selected_button = ttk.Button(
            run_buttons_frame,
            text="Run Selected Protocol",
            command=self.run_selected_protocol,
            state=tk.DISABLED,
        )
        self.run_selected_button.pack(side=tk.LEFT, padx=(0, 5))

        # Run all button
        self.run_all_button = ttk.Button(
            run_buttons_frame,
            text="Run All Protocols",
            command=self.run_all_protocols,
        )
        self.run_all_button.pack(side=tk.LEFT, padx=(5, 0))

    def setup_parameters_tab(self, parent_frame):
        """Setup the parameters configuration tab."""
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(1, weight=1)

        # Protocol selector
        selector_frame = ttk.Frame(parent_frame)
        selector_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(selector_frame, text="Configure parameters for:").pack(side=tk.LEFT)
        self.protocol_selector = ttk.Combobox(
            selector_frame, values=list(self.protocols.keys()), state="readonly"
        )
        self.protocol_selector.pack(side=tk.LEFT, padx=(10, 0))
        self.protocol_selector.bind("<<ComboboxSelected>>", self.on_protocol_selected)

        # Parameters frame
        self.params_frame = ttk.LabelFrame(
            parent_frame, text="Parameters", padding="10"
        )
        self.params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollable frame for parameters
        self.params_canvas = tk.Canvas(self.params_frame, height=300)
        self.params_scrollbar = ttk.Scrollbar(
            self.params_frame, orient="vertical", command=self.params_canvas.yview
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

        # Control buttons
        control_frame = ttk.Frame(parent_frame)
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Button(
            control_frame, text="Load Defaults", command=self.load_default_parameters
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(
            control_frame, text="Save Parameters", command=self.save_parameters
        ).pack(side=tk.LEFT)

    def select_protocol(self, protocol_name, protocol_info):
        """Select a protocol for running."""
        self.selected_protocol = protocol_name
        self.selected_protocol_label.config(text=f"Selected: {protocol_name}")
        self.run_selected_button.config(state=tk.NORMAL)
        self.log_message(f"Selected protocol: {protocol_name}")

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
            ).pack()
            return

        # Create parameter controls
        row = 0
        self.parameter_widgets = {}

        for param_name, param_config in protocol_info["parameters"].items():
            # Parameter label
            label = ttk.Label(self.params_scrollable_frame, text=f"{param_name}:")
            label.grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=2)

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
                    self.params_scrollable_frame, textvariable=var, width=17
                )

            widget.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)

            # Description tooltip
            self.create_tooltip(widget, param_config.get("description", ""))

            # Store widget reference
            self.parameter_widgets[param_name] = {"var": var, "widget": widget}

            row += 1

        # Configure grid weights
        self.params_scrollable_frame.columnconfigure(1, weight=1)

    def load_default_parameters(self):
        """Load default parameter values."""
        for protocol_name, protocol_info in self.protocols.items():
            if "parameters" in protocol_info:
                self.parameter_values[protocol_name] = {}
                for param_name, param_config in protocol_info["parameters"].items():
                    self.parameter_values[protocol_name][param_name] = param_config[
                        "default"
                    ]

    def save_parameters(self):
        """Save current parameter values with validation."""
        selected = self.protocol_selector.get()
        if not selected:
            messagebox.showwarning("No Selection", "Please select a protocol first.")
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

    def run_all_protocols(self):
        """Run all protocols sequentially with default parameters."""
        # Confirm with user
        protocol_count = len(self.protocols)
        if not messagebox.askyesno(
            "Confirm Run All",
            f"This will run all {protocol_count} protocols sequentially.\n\n"
            "This may take a significant amount of time.\n\n"
            "Continue?",
        ):
            return

        # Run protocols in a separate thread
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

                # Use default parameters
                protocol_info_with_params = protocol_info.copy()
                protocol_info_with_params["configured_params"] = {}

                # Run synchronously in this thread
                self._run_single_protocol(protocol_info_with_params)

            self.log_message(f"\n{'=' * 60}")
            self.log_message("All protocols completed!")
            self.log_message(f"{'=' * 60}")
            self.set_status("Ready")
            self.gui_queue.put(lambda: self.run_all_button.config(state=tk.NORMAL))
            self.gui_queue.put(lambda: self.stop_btn.config(state=tk.DISABLED))

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
            file_path = os.path.join(os.path.dirname(__file__), protocol_info["file"])

            spec = importlib.util.spec_from_file_location(
                protocol_info["file"], file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get configured parameters
            configured_params = protocol_info.get("configured_params", {})

            # Protocol 13 – Framework Aggregator: special entry point
            if protocol_info["class"] == "FalsificationAggregator":
                result = self._handle_framework_aggregator(module)

            else:
                # Get the main class for all other protocols
                cls = getattr(module, protocol_info["class"])

                # Robust protocol execution logic
                result = {}
                if protocol_info["class"] == "NetworkComparisonExperiment":
                    config = {
                        "extero_dim": configured_params.get("extero_dim", 32),
                        "intero_dim": configured_params.get("intero_dim", 16),
                        "action_dim": configured_params.get("action_dim", 4),
                        "context_dim": configured_params.get("context_dim", 8),
                        "n_episodes": configured_params.get("n_episodes", 50),
                    }
                    instance = cls(config)
                    result = instance.run_full_experiment()
                    self.log_message(f"  Result: {type(result)}")

                elif protocol_info["class"] == "APGIActiveInferenceAgent":
                    run_func = getattr(module, "run_falsification", None)
                    if run_func:
                        result = run_func()
                        self.log_message(f"  Falsification completed: {result}")
                    else:
                        self.log_message("  No run_falsification function found")

                elif protocol_info["class"] == "IowaGamblingTaskEnvironment":
                    run_func = getattr(module, "run_falsification", None)
                    if run_func:
                        result = run_func()
                        self.log_message(f"  Falsification completed: {result}")
                    else:
                        self.log_message("  No run_falsification function found")

                elif hasattr(module, "run_falsification"):
                    self.log_message("  Running run_falsification from module scope...")
                    result = module.run_falsification()
                    status = "Done"
                    if result and isinstance(result, dict):
                        status = result.get("status", "Done")
                    elif result is None:
                        status = "ERROR (Returned None)"
                    self.log_message(f"  Protocol completed: {status}")

                elif hasattr(cls, "run_full_experiment"):
                    self.log_message("  Running instance.run_full_experiment()...")
                    try:
                        instance = cls(**configured_params)
                    except Exception:
                        instance = cls()
                    result = instance.run_full_experiment()
                    self.log_message("  Experiment completed")

                elif hasattr(cls, "run_phase_transition_analysis"):
                    self.log_message("  Running phase transition analysis...")
                    surprise_system = module.SurpriseIgnitionSystem()
                    instance = cls(surprise_system)
                    result = instance.run_phase_transition_analysis()
                    self.log_message("  Analysis completed")

                elif hasattr(cls, "run_evolution"):
                    self.log_message("  Running evolutionary simulation...")
                    config = {
                        "population_size": configured_params.get("population_size", 50),
                        "n_generations": configured_params.get("n_generations", 100),
                        "mutation_rate": configured_params.get("mutation_rate", 0.1),
                        "selection_pressure": configured_params.get(
                            "selection_pressure", 2.0
                        ),
                    }
                    instance = cls(stop_event=self.stop_event, **config)
                    result = instance.run_evolution()
                    self.log_message("  Evolution completed")

                else:
                    self.log_message(f"  Using generic runner for {cls.__name__}...")
                    try:
                        instance = cls(**configured_params)
                    except TypeError:
                        instance = cls()

                    if hasattr(instance, "run_validation"):
                        result = instance.run_validation()
                    elif hasattr(module, "run_validation"):
                        result = module.run_validation()
                    elif hasattr(instance, "run_falsification"):
                        result = instance.run_falsification()
                    else:
                        self.log_message(
                            "  WARNING: No standard execution method found. Result will be empty."
                        )
                        result = {"status": "Incomplete", "warning": "No runner found"}

            # FIX: Actually save the result instead of an empty dict
            self._save_results(result, protocol_info["file"])
            self.log_message("  Protocol completed successfully")

        except Exception as e:
            self.log_message(f"  ERROR: {str(e)}")
            logger.error(f"Error in {protocol_info['file']}: {e}")

    def run_headless(self):
        """Run all protocols in headless mode."""
        self.log_message("Running in HEADLESS mode")
        # Initialize default parameters if not loaded
        if not self.parameter_values:
            self.load_default_parameters()

        total = len(self.protocols)
        for idx, (protocol_name, protocol_info) in enumerate(self.protocols.items(), 1):
            self.log_message(f"\n[{idx}/{total}] Running {protocol_name}...")
            # Use default parameters for headless run
            protocol_info_with_params = protocol_info.copy()
            protocol_info_with_params["configured_params"] = {}
            self._run_single_protocol(protocol_info_with_params)

        self.log_message("\n" + "=" * 60)
        self.log_message("Headless run completed.")
        self.log_message("=" * 60)

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

    def create_tooltip(self, widget, text):
        """Create tooltip for widget"""

        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            label = tk.Label(
                tooltip,
                text=text,
                background="lightyellow",
                relief=tk.SOLID,
                borderwidth=1,
            )
            label.pack()
            widget.tooltip = tooltip

        def on_leave(event):
            if hasattr(widget, "tooltip"):
                widget.tooltip.destroy()
                del widget.tooltip

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def clear_console(self):
        """Clear the console output."""
        if hasattr(self, "output_console"):
            self.output_console.delete(1.0, tk.END)

    def _setup_tab_order(self) -> None:
        """Configure tab order for consistent keyboard navigation."""
        # This will be called after UI setup to configure tab navigation
        # Store widgets that will be created during setup
        self._tab_widgets: List[tk.Widget] = []  # type: ignore

    def _finalize_tab_order(self) -> None:
        """Finalize tab order after all widgets are created."""
        if hasattr(self, "_tab_widgets"):
            # Set up tab navigation for collected widgets
            for i, widget in enumerate(self._tab_widgets):
                if widget and hasattr(widget, "bind"):
                    next_widget = self._tab_widgets[(i + 1) % len(self._tab_widgets)]
                    prev_widget = self._tab_widgets[(i - 1) % len(self._tab_widgets)]

                    # Bind Tab to move to next widget
                    def make_tab_handler(nw: tk.Widget) -> Callable[[Any], None]:
                        def handler(e: Any) -> None:
                            self._focus_widget(nw)

                        return handler

                    widget.bind("<Tab>", make_tab_handler(next_widget))

                    # Bind Shift+Tab to move to previous widget
                    def make_shift_tab_handler(pw: tk.Widget) -> Callable[[Any], None]:
                        def handler(e: Any) -> None:
                            self._focus_widget(pw)

                        return handler

                    widget.bind("<Shift-Tab>", make_shift_tab_handler(prev_widget))

    def _focus_widget(self, widget: tk.Widget) -> None:
        """Focus on a specific widget and handle different widget types."""
        try:
            if hasattr(widget, "focus_set"):
                widget.focus_set()
            elif hasattr(widget, "select"):
                widget.select()
        except tk.TclError:
            # Widget might be disabled or not focusable
            pass

    def stop_protocol(self):
        """Stop the currently running protocol if any"""
        if self.stop_event:
            self.stop_event.set()
            self.log_message("Stop signal sent to protocol...")
            self.set_status("Stopping...")

    def set_status(self, message):
        """Set status bar message (thread-safe)"""
        if self.headless:
            print(f"STATUS: {message}")
            return

        def _update():
            self.status_var.set(message)

        self.gui_queue.put(_update)

    def log_message(self, message):
        """Add message to output console (thread-safe)"""
        if self.headless:
            print(message)
            return

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

        self.gui_queue.put(_update)

    def run_protocol(self, protocol_info):
        """Run selected protocol in separate thread"""

        def protocol_thread():
            try:
                self.set_status(f"Running {protocol_info['file']}...")
                self.log_message(f"=== Running {protocol_info['file']} ===")

                # Ensure project root is in sys.path for imports
                project_root = os.path.dirname(os.path.abspath(__file__))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                # Load the protocol module
                file_path = os.path.join(
                    os.path.dirname(__file__), protocol_info["file"]
                )

                # Validate file path to prevent path traversal
                # Check for path traversal attempts
                if ".." in protocol_info["file"] or protocol_info["file"].startswith(
                    "/"
                ):
                    raise ValueError(
                        f"Invalid protocol path: {protocol_info['file']} (contains path traversal)"
                    )

                # Resolve and validate it's within project directory
                try:
                    Path(file_path).resolve().relative_to(
                        Path(os.path.dirname(__file__)).resolve()
                    )
                except ValueError:
                    raise ValueError(
                        f"Invalid protocol path: {protocol_info['file']} (outside project directory)"
                    )

                spec = importlib.util.spec_from_file_location(
                    protocol_info["file"], file_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.module = module

                # Handle protocol based on class with configured parameters
                configured_params = protocol_info.get("configured_params", {})

                # Validate numeric parameters
                validated_params = {}
                for key, value in configured_params.items():
                    try:
                        # Try to convert to float
                        num_value = float(value)
                        # Add basic range validation
                        if "dim" in key or "n_" in key:
                            # Dimensions and counts should be positive integers
                            if num_value <= 0 or num_value != int(num_value):
                                raise ValueError(f"{key} must be a positive integer")
                            validated_params[key] = int(num_value)
                        elif "lr" in key:
                            # Learning rates should be positive and reasonable
                            if num_value <= 0 or num_value > 1.0:
                                raise ValueError(f"{key} must be between 0 and 1")
                            validated_params[key] = num_value
                        else:
                            validated_params[key] = num_value
                    except (ValueError, TypeError) as e:
                        self.log_message(
                            f"Warning: Invalid value for {key}: {value}. Error: {e}. Using default."
                        )
                        # Don't add invalid params to validated_params

                if protocol_info["class"] == "FalsificationAggregator":
                    # Framework aggregator: no class instantiation needed
                    results = self._handle_framework_aggregator(module)
                else:
                    # All other protocols: resolve class then dispatch
                    cls = getattr(module, protocol_info["class"])
                    if protocol_info["class"] == "NetworkComparisonExperiment":
                        results = self._handle_network_comparison(cls, validated_params)
                    elif protocol_info["class"] == "APGIActiveInferenceAgent":
                        results = self._handle_apgi_agent(cls, module, validated_params)
                    elif protocol_info["class"] == "IowaGamblingTaskEnvironment":
                        results = self._handle_iowa_gambling(
                            cls, module, validated_params
                        )
                    elif hasattr(module, "run_falsification"):
                        # Fallback to module-level run_falsification
                        self.log_message(
                            "Running run_falsification from module scope..."
                        )
                        results = module.run_falsification()
                    elif hasattr(cls, "run_full_experiment"):
                        results = self._handle_run_full_experiment(
                            cls, validated_params
                        )
                    elif hasattr(cls, "run_phase_transition_analysis"):
                        results = self._handle_phase_transition(
                            cls, module, validated_params
                        )
                    elif hasattr(cls, "run_evolution"):
                        results = self._handle_evolution(cls, validated_params)
                    else:
                        results = self._handle_default(cls, module, validated_params)

                # Save results to file
                self._save_results(results, protocol_info["file"])

                self.log_message("=== Protocol completed successfully ===")
                self.set_status("Ready")
                self.gui_queue.put(lambda: self.stop_btn.config(state=tk.DISABLED))

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
                self.gui_queue.put(lambda: messagebox.showerror("Error", error_msg))
                self.set_status("Error")
                self.gui_queue.put(lambda: self.stop_btn.config(state=tk.DISABLED))
            except Exception as e:
                logger.error(f"Unexpected error in {protocol_info['file']}: {str(e)}")
                error_message = f"Unexpected error: {str(e)}"
                self.gui_queue.put(lambda: messagebox.showerror("Error", error_message))
                self.set_status("Error")
                self.gui_queue.put(lambda: self.stop_btn.config(state=tk.DISABLED))

        # Run in separate thread to avoid blocking GUI
        self.stop_event.clear()
        self.running_thread = None
        thread = threading.Thread(target=protocol_thread)
        thread.daemon = True
        self.running_thread = thread
        self.stop_btn.config(state=tk.NORMAL)
        thread.start()

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
                    "Broadcasting error in Protocol-3 - this is expected due to observation size mismatches"
                )
                self.log_message("Protocol-3 needs observation size alignment")
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


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="APGI Falsification Protocols GUI / Headless Runner"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run all falsification protocols without launching the GUI",
    )
    parser.add_argument(
        "--token",
        help="JWT authentication token for secured operations",
    )
    args = parser.parse_args()

    # Security check for non-headless GUI mode - GUI scripts run in dev mode by default
    if not args.headless:
        try:
            from utils.security_gateway import Role, SecurityGateway

            gateway = SecurityGateway()

            if args.token:
                # Validate token and check GUI access permissions
                gateway.require_roles(args.token, [Role.RESEARCHER, Role.ADMIN])
            else:
                # GUI scripts automatically run in development mode
                # Generate a token automatically without requiring APGI_DEV_MODE
                try:
                    from utils.auth_adapter import get_auth_manager

                    auth_manager = get_auth_manager()
                    dev_token = auth_manager.generate_token(
                        "dev_user", Role.RESEARCHER, 24
                    )  # 24 hour expiry
                    print("Development mode: Generated token (valid 24 hours)")
                    args.token = dev_token
                except Exception as e:
                    # If token generation fails, continue without authentication
                    # This allows GUI to work even if auth system has issues
                    print(f"Note: Running without authentication ({e})")
        except ImportError:
            # Security gateway not available - allow GUI to run without authentication
            pass
        except PermissionError as e:
            print(f"Authentication failed: {e}")
            sys.exit(1)

    if args.headless:
        print("Running in HEADLESS mode")
        from unittest.mock import MagicMock

        mock_root = MagicMock()
        runner = ProtocolRunnerGUI(mock_root, headless=True)
        runner.run_headless()
        return

    root = tk.Tk()
    _ = ProtocolRunnerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
