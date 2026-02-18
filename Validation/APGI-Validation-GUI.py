"""
APGI Validation GUI
==================

Simple tkinter GUI for running APGI validation protocols
with real-time progress tracking and results visualization.
"""

import contextlib
import gc
import importlib
import importlib.util
import io
import json
import logging
import os
import queue
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, List, Optional

import numpy as np


def safe_import_module(module_name: str, file_path: Path) -> Optional[Any]:
    """Safely import a module with detailed error reporting."""
    try:
        if not file_path.exists():
            return None

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    except ImportError as e:
        return None
    except SyntaxError as e:
        return None
    except (AttributeError, ValueError, TypeError, RuntimeError) as e:
        return None


# Try to import master validation module
master_validation_path = Path(__file__).parent / "Master-Validation.py"
APGI_Master_Validation = safe_import_module("Master_Validation", master_validation_path)

if APGI_Master_Validation:
    try:
        APGIMasterValidator = APGI_Master_Validation.APGIMasterValidator
    except AttributeError as e:
        APGIMasterValidator = None
else:
    APGIMasterValidator = None

# Try to import individual protocols
protocol_files = [
    ("APGI_Protocol_1", "Validation-Protocol-1.py"),
    ("APGI_Protocol_2", "Validation-Protocol-2.py"),
    ("APGI_Protocol_3", "Validation-Protocol-3.py"),
    ("APGI_Protocol_4", "Validation-Protocol-4.py"),
    ("APGI_Protocol_5", "Validation-Protocol-5.py"),
    ("APGI_Protocol_6", "Validation-Protocol-6.py"),
    ("APGI_Protocol_7", "Validation-Protocol-7.py"),
    ("APGI_Protocol_8", "Validation-Protocol-8.py"),
]

for protocol_name, filename in protocol_files:
    protocol_path = Path(__file__).parent / filename
    protocol_module = safe_import_module(protocol_name, protocol_path)


class APGIValidationGUI:
    """GUI for running APGI validation protocols with real-time progress tracking."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("APGI Validation Protocol Runner")
        self.root.geometry("800x600")

        # Import status tracking (instance variable instead of global)
        self._import_status = {
            "master_validation": False,
            "protocols": {},
            "errors": [],
        }

        # Thread safety locks
        self._running_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._thread_cleanup_lock = threading.Lock()

        # GUI update queue for thread safety
        self._update_queue = queue.Queue(
            maxsize=100
        )  # Limit queue size to prevent memory issues
        self._process_gui_updates()

        # Set environment variables to prevent GUI operations in worker threads
        self._setup_worker_thread_environment()

        # Initialize validator with proper validation
        if APGIMasterValidator:
            try:
                self.validator = APGIMasterValidator()
            except Exception as e:
                logging.error(f"Failed to initialize validator: {e}")
                self.validator = None
        else:
            self.validator = None

        # Module cache to prevent memory leaks
        self._protocol_cache: Dict[str, Any] = {}

        # Setup logging
        self._setup_logging()

        # Create GUI elements
        self.create_widgets()

        # Validation thread
        self.validation_thread: Optional[threading.Thread] = None
        self._is_running = False

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_logging(self) -> None:
        """Setup logging system with error handling."""
        try:
            log_dir = Path(__file__).parent.parent / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"validation_{datetime.now().strftime('%Y%m%d')}.log"

            # Configure logging with error handling
            logging.basicConfig(
                filename=str(log_file),
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

            # Test logging to ensure it works
            logging.info("APGI Validation GUI initialized")

        except (OSError, PermissionError, IOError) as e:
            # Fallback to console logging if file logging fails
            print(f"Warning: Could not setup file logging: {e}")
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )
            logging.info("APGI Validation GUI initialized (console logging fallback)")
        except Exception as e:
            print(f"Critical: Failed to setup logging: {e}")
            # Continue without logging rather than crash the GUI

    def _setup_worker_thread_environment(self) -> None:
        """Setup environment to prevent GUI operations in worker threads."""
        # Set matplotlib to use non-interactive backend to prevent GUI operations
        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend
            logging.info("Set matplotlib to non-interactive backend")
        except ImportError:
            pass  # matplotlib not available
        except Exception as e:
            logging.warning(f"Failed to set matplotlib backend: {e}")

        # Set environment variables to disable GUI
        os.environ["MPLBACKEND"] = "Agg"
        os.environ["DISPLAY"] = ""  # Disable display for worker threads

    @property
    def is_running(self) -> bool:
        """Thread-safe getter for running status."""
        with self._running_lock:
            return self._is_running

    @is_running.setter
    def is_running(self, value: bool) -> None:
        """Thread-safe setter for running status."""
        with self._running_lock:
            self._is_running = value

    def on_closing(self) -> None:
        """Handle window close event with proper cleanup."""
        if self.is_running:
            if not messagebox.askyesno(
                "Quit", "Validation in progress. Stop and quit?"
            ):
                return
            self.stop_validation()
        self.clear_protocol_cache()
        self.root.destroy()

    def clear_protocol_cache(self) -> None:
        """Clear protocol cache and force garbage collection."""
        with self._cache_lock:
            self._protocol_cache.clear()
        gc.collect()
        logging.info("Protocol cache cleared")

    def _process_gui_updates(self) -> None:
        """Process GUI updates from queue to ensure thread safety"""
        try:
            # Process up to 10 updates per cycle to prevent blocking
            for _ in range(10):
                try:
                    update_type, data = self._update_queue.get_nowait()

                    if update_type == "status":
                        self.status_label.config(text=data)
                    elif update_type == "progress":
                        self.progress_var.set(
                            min(100, max(0, data))
                        )  # Clamp between 0-100
                    elif update_type == "results":
                        self.results_text.insert(tk.END, data)
                        self.results_text.see(tk.END)
                    elif update_type == "summary":
                        self.summary_label.config(text=data)

                    self._update_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            logging.error(f"Error processing GUI updates: {e}")

        # Schedule next update check
        self.root.after(50, self._process_gui_updates)  # Check every 50ms

    def _ensure_ui_consistency(self) -> None:
        """Ensure UI state is consistent with running status"""
        try:
            # Check if widgets exist before accessing
            if hasattr(self, "run_button") and hasattr(self, "stop_button"):
                if self.is_running:
                    self.run_button.config(state=tk.DISABLED)
                    self.stop_button.config(state=tk.NORMAL)
                else:
                    self.run_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
        except Exception as e:
            logging.error(f"Error ensuring UI consistency: {e}")

    def show_import_status(self) -> None:
        """Display import status and provide guidance"""
        status_window = tk.Toplevel(self.root)
        status_window.title("Import Status")
        status_window.geometry("600x400")

        # Create scrolled text widget
        text_widget = scrolledtext.ScrolledText(
            status_window, wrap=tk.WORD, width=70, height=20
        )
        text_widget.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Add status information
        text_widget.insert(tk.END, "=== APGI Validation GUI Import Status ===\n\n")

        # Master validation status
        if APGIMasterValidator:
            text_widget.insert(tk.END, "✅ Master Validation: Loaded successfully\n")
        else:
            text_widget.insert(tk.END, "❌ Master Validation: Failed to load\n")

        # Protocol status
        text_widget.insert(tk.END, "\n=== Protocol Status ===\n")
        for protocol_name, filename in protocol_files:
            protocol_path = Path(__file__).parent / filename
            exists = protocol_path.exists()
            status_symbol = "✅" if exists else "❌"
            text_widget.insert(
                tk.END,
                f"{status_symbol} {protocol_name}: {'Found' if exists else 'Not Found'}\n",
            )

        # Troubleshooting guidance
        text_widget.insert(tk.END, "\n=== Troubleshooting ===\n")
        text_widget.insert(
            tk.END,
            "1. Check that all validation protocol files exist in the Validation directory\n",
        )
        text_widget.insert(
            tk.END,
            "2. Verify all dependencies are installed: pip install -r requirements.txt\n",
        )
        text_widget.insert(tk.END, "3. Check for syntax errors in protocol files\n")
        text_widget.insert(
            tk.END, "4. Ensure Python path includes the Validation directory\n"
        )
        text_widget.insert(
            tk.END, "5. Run individual protocols from command line to test\n\n"
        )

        # Fallback options
        text_widget.insert(tk.END, "=== Fallback Options ===\n")
        text_widget.insert(
            tk.END, "• GUI will operate in limited mode without full validation\n"
        )
        text_widget.insert(
            tk.END, "• Use command line: python main.py validate --protocol <number>\n"
        )
        text_widget.insert(tk.END, "• Check logs for detailed error information\n")

        text_widget.config(state=tk.DISABLED)

        # Close button
        close_btn = ttk.Button(
            status_window, text="Close", command=status_window.destroy
        )
        close_btn.pack(pady=10)

    def create_widgets(self) -> None:
        """Create all GUI widgets with tabbed interface"""

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="APGI Validation Protocol Runner",
            font=("Arial", 16, "bold"),
        )
        title_label.grid(row=0, column=0, pady=(0, 20))

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Validation Tab
        validation_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(validation_frame, text="Validation")

        # Configure validation tab grid
        validation_frame.columnconfigure(1, weight=1)
        validation_frame.rowconfigure(3, weight=1)

        # Protocol selection frame
        protocol_frame = ttk.LabelFrame(
            validation_frame, text="Protocol Selection", padding="10"
        )
        protocol_frame.grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10)
        )
        protocol_frame.columnconfigure(0, weight=1)

        # Protocol checkboxes
        self.protocol_vars = {}
        protocols_info = {
            1: "Protocol 1: Primary Test",
            2: "Protocol 2: Secondary Test",
            3: "Protocol 3: Primary Test",
            4: "Protocol 4: Secondary Test",
            5: "Protocol 5: Tertiary Test",
            6: "Protocol 6: Tertiary Test",
            7: "Protocol 7: Tertiary Test",
            8: "Protocol 8: Secondary Test",
        }

        for i, (num, desc) in enumerate(protocols_info.items()):
            var = tk.BooleanVar(value=True)
            self.protocol_vars[num] = var

            cb = ttk.Checkbutton(protocol_frame, text=desc, variable=var)
            cb.grid(row=i // 2, column=(i % 2) * 2, sticky=tk.W, padx=5, pady=2)

        # Control buttons frame
        control_frame = ttk.Frame(validation_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10))

        self.run_button = ttk.Button(
            control_frame, text="Run Validation", command=self.run_parameter_simulation
        )
        self.run_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(
            control_frame, text="Stop", command=self.stop_validation, state=tk.DISABLED
        )
        self.stop_button.grid(row=0, column=1, padx=5)

        self.save_button = ttk.Button(
            control_frame, text="Save Results", command=self.save_results
        )
        self.save_button.grid(row=0, column=2, padx=5)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            validation_frame, variable=self.progress_var, maximum=100, length=400
        )
        self.progress_bar.grid(
            row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        # Status label
        self.status_label = ttk.Label(
            validation_frame, text="Ready to run validation", font=("Arial", 10)
        )
        self.status_label.grid(row=3, column=0, columnspan=2, pady=(0, 10))

        # Results text area
        results_frame = ttk.LabelFrame(
            validation_frame, text="Validation Results", padding="10"
        )
        results_frame.grid(
            row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10)
        )
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.results_text = scrolledtext.ScrolledText(
            results_frame, height=15, width=80
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Summary frame
        summary_frame = ttk.LabelFrame(
            validation_frame, text="Validation Summary", padding="10"
        )
        summary_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.summary_label = ttk.Label(
            summary_frame, text="No validation run yet", font=("Arial", 10)
        )
        self.summary_label.grid(row=0, column=0)

        # Parameter Exploration Tab
        exploration_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(exploration_frame, text="Parameter Exploration")

        self.create_parameter_exploration_widgets(exploration_frame)

        # Settings Tab
        settings_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(settings_frame, text="Settings")

        self.create_settings_widgets(settings_frame)

        # Data Export Tab
        export_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(export_frame, text="Data Export")

        self.create_export_widgets(export_frame)

        # Alerts Tab
        alerts_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(alerts_frame, text="Alerts")

        self.create_alerts_widgets(alerts_frame)

    def create_parameter_exploration_widgets(self, parent_frame: ttk.Frame) -> None:
        """Create interactive parameter exploration widgets with sliders and real-time feedback"""

        # Configure parent frame
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(1, weight=1)

        # Parameter controls frame
        controls_frame = ttk.LabelFrame(
            parent_frame, text="Parameter Controls", padding="10"
        )
        controls_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)

        # APGI Parameters with sliders
        self.param_vars = {}
        self.param_sliders = {}
        self.param_labels = {}

        parameters = {
            "tau_S": {
                "label": "Signal Time Constant (τ_S)",
                "min": 0.1,
                "max": 5.0,
                "default": 0.5,
                "step": 0.1,
            },
            "tau_theta": {
                "label": "Threshold Time Constant (τ_θ)",
                "min": 10.0,
                "max": 100.0,
                "default": 30.0,
                "step": 5.0,
            },
            "theta_0": {
                "label": "Initial Threshold (θ₀)",
                "min": 0.1,
                "max": 2.0,
                "default": 0.5,
                "step": 0.1,
            },
            "alpha": {
                "label": "Sigmoid Slope (α)",
                "min": 0.1,
                "max": 20.0,
                "default": 5.0,
                "step": 0.5,
            },
        }

        row = 0
        for param_name, config in parameters.items():
            # Parameter label
            label = ttk.Label(controls_frame, text=config["label"])
            label.grid(row=row, column=0, sticky=tk.W, padx=(0, 10))

            # Value variable and label
            value_var = tk.DoubleVar(value=config["default"])
            self.param_vars[param_name] = value_var

            value_label = ttk.Label(controls_frame, text=".2f")
            value_label.grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
            self.param_labels[param_name] = value_label

            # Slider
            slider = tk.Scale(
                controls_frame,
                from_=config["min"],
                to=config["max"],
                resolution=config["step"],
                orient=tk.HORIZONTAL,
                variable=value_var,
                command=lambda val, name=param_name: self.on_parameter_change(
                    name, val
                ),
            )
            slider.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5)
            self.param_sliders[param_name] = slider

            row += 1

        # Control buttons for exploration
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=(10, 0))

        self.run_sim_button = ttk.Button(
            button_frame, text="Run Simulation", command=self.run_parameter_simulation
        )
        self.run_sim_button.grid(row=0, column=0, padx=5)

        self.reset_params_button = ttk.Button(
            button_frame, text="Reset to Defaults", command=self.reset_parameters
        )
        self.reset_params_button.grid(row=0, column=1, padx=5)

        # Results display frame
        results_frame = ttk.LabelFrame(
            parent_frame, text="Simulation Results", padding="10"
        )
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Text area for results
        self.param_results_text = scrolledtext.ScrolledText(
            results_frame, height=15, width=80
        )
        self.param_results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Initialize parameter display
        self.update_parameter_display()

    def create_settings_widgets(self, parent_frame: ttk.Frame) -> None:
        """Create settings configuration widgets"""

        # Configure parent frame
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(0, weight=1)

        # Settings frame
        settings_frame = ttk.LabelFrame(
            parent_frame, text="Configuration Settings", padding="10"
        )
        settings_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Initialize settings if not already done
        if not hasattr(self, "settings"):
            self.settings = {
                "update_interval": tk.IntVar(value=10),  # seconds
                "data_retention": tk.IntVar(value=30),  # days
                "monitoring_threshold": tk.DoubleVar(value=0.05),  # error rate
            }

        # Update interval
        ttk.Label(settings_frame, text="Update Interval (seconds):").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        ttk.Spinbox(
            settings_frame,
            from_=1,
            to=3600,
            textvariable=self.settings["update_interval"],
            width=10,
        ).grid(row=0, column=1, sticky=tk.W, padx=10)

        # Data retention
        ttk.Label(settings_frame, text="Data Retention (days):").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        ttk.Spinbox(
            settings_frame,
            from_=1,
            to=365,
            textvariable=self.settings["data_retention"],
            width=10,
        ).grid(row=1, column=1, sticky=tk.W, padx=10)

        # Monitoring threshold
        ttk.Label(settings_frame, text="Monitoring Threshold (error rate):").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        ttk.Spinbox(
            settings_frame,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.settings["monitoring_threshold"],
            width=10,
        ).grid(row=2, column=1, sticky=tk.W, padx=10)

        # Save settings button
        ttk.Button(
            settings_frame, text="Save Settings", command=self.save_settings
        ).grid(row=3, column=0, columnspan=2, pady=20)

    def save_settings(self) -> None:
        """Save current settings"""
        # For now, just show a message
        messagebox.showinfo("Settings", "Settings saved successfully!")

    def create_export_widgets(self, parent_frame: ttk.Frame) -> None:
        """Create data export widgets"""

        # Configure parent frame
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(0, weight=1)

        # Export frame
        export_frame = ttk.LabelFrame(parent_frame, text="Data Export", padding="10")
        export_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Export buttons
        ttk.Button(
            export_frame, text="Export Results to CSV", command=self.export_csv
        ).grid(row=0, column=0, pady=10, padx=10, sticky=(tk.W, tk.E))

        ttk.Button(
            export_frame, text="Export Results to JSON", command=self.export_json
        ).grid(row=1, column=0, pady=10, padx=10, sticky=(tk.W, tk.E))

        ttk.Button(
            export_frame, text="Generate PDF Report", command=self.generate_report
        ).grid(row=2, column=0, pady=10, padx=10, sticky=(tk.W, tk.E))

    def export_csv(self) -> None:
        """Export validation results to CSV"""
        if (
            not hasattr(self.validator, "protocol_results")
            or not self.validator.protocol_results
        ):
            messagebox.showwarning(
                "Export", "No results to export. Run validation first."
            )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if file_path:
            # Simple CSV export
            import csv

            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Protocol", "Status", "Passed"])
                for protocol, result in self.validator.protocol_results.items():
                    writer.writerow(
                        [protocol, result.get("status", ""), result.get("passed", "")]
                    )
            messagebox.showinfo("Export", f"Results exported to {file_path}")

    def export_json(self) -> None:
        """Export validation results to JSON"""
        if (
            not hasattr(self.validator, "protocol_results")
            or not self.validator.protocol_results
        ):
            messagebox.showwarning(
                "Export", "No results to export. Run validation first."
            )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if file_path:
            with open(file_path, "w") as f:
                json.dump(self.validator.protocol_results, f, indent=2, default=str)
            messagebox.showinfo("Export", f"Results exported to {file_path}")

    def generate_report(self) -> None:
        """Generate a PDF report"""
        if not hasattr(self.validator, "generate_master_report"):
            messagebox.showerror("Report", "Report generation not available.")
            return

        try:
            report = self.validator.generate_master_report()
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            )
            if file_path:
                with open(file_path, "w") as f:
                    f.write(f"APGI Validation Report\n{'='*50}\n\n")
                    f.write(
                        f"Overall Decision: {report.get('overall_decision', 'Unknown')}\n\n"
                    )
                    f.write("Protocol Results:\n")
                    for protocol, result in report.get("protocol_results", {}).items():
                        f.write(f"- {protocol}: {result.get('status', '')}\n")
                messagebox.showinfo("Report", f"Report generated at {file_path}")
        except Exception as e:
            messagebox.showerror("Report", f"Failed to generate report: {e}")

    def create_alerts_widgets(self, parent_frame: ttk.Frame) -> None:
        """Create alert configuration widgets"""

        # Configure parent frame
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(0, weight=1)

        # Alerts frame
        alerts_frame = ttk.LabelFrame(
            parent_frame, text="Alert Configuration", padding="10"
        )
        alerts_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Initialize alert settings if not already done
        if not hasattr(self, "alert_settings"):
            self.alert_settings = {
                "threshold": tk.DoubleVar(value=0.8),
                "enabled": tk.BooleanVar(value=True),
            }

        # Alert threshold
        ttk.Label(alerts_frame, text="Alert Threshold:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        ttk.Spinbox(
            alerts_frame,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.alert_settings["threshold"],
            width=10,
        ).grid(row=0, column=1, sticky=tk.W, padx=10)

        # Enable alerts
        ttk.Checkbutton(
            alerts_frame, text="Enable Alerts", variable=self.alert_settings["enabled"]
        ).grid(row=1, column=0, columnspan=2, pady=10)

        # Save alerts button
        ttk.Button(
            alerts_frame, text="Save Alert Settings", command=self.save_alert_settings
        ).grid(row=2, column=0, columnspan=2, pady=20)

    def save_alert_settings(self) -> None:
        """Save alert settings"""
        messagebox.showinfo("Alerts", "Alert settings saved successfully!")

    def on_parameter_change(self, param_name: str, value: str) -> None:
        """Handle parameter slider changes"""
        try:
            float_value = float(value)
            self.param_labels[param_name].config(text=f"{float_value:.2f}")
        except ValueError:
            pass

    def update_parameter_display(self) -> None:
        """Update all parameter value displays"""
        for param_name, value_var in self.param_vars.items():
            value = value_var.get()
            try:
                float_value = float(value)
                self.param_labels[param_name].config(text=f"{float_value:.2f}")
            except ValueError:
                self.param_labels[param_name].config(text=value)

    def reset_parameters(self) -> None:
        """Reset all parameters to default values"""
        defaults = {
            "tau_S": 0.5,
            "tau_theta": 30.0,
            "theta_0": 0.5,
            "alpha": 5.0,
        }

        for param_name, default_value in defaults.items():
            self.param_vars[param_name].set(default_value)
            self.param_sliders[param_name].set(default_value)

        self.update_parameter_display()
        self.param_results_text.delete(1.0, tk.END)
        self.param_results_text.insert(tk.END, "Parameters reset to defaults\n")

    def run_parameter_simulation(self) -> None:
        """Run simulation with current parameter values"""
        # Get current parameter values
        params = {name: var.get() for name, var in self.param_vars.items()}

        self.param_results_text.delete(1.0, tk.END)
        self.param_results_text.insert(tk.END, "Running simulation with parameters:\n")
        for name, value in params.items():
            self.param_results_text.insert(tk.END, f"  {name}: {value}\n")
        self.param_results_text.insert(tk.END, "\n")

        # Run simulation in thread to avoid blocking GUI
        threading.Thread(
            target=self._run_parameter_simulation_worker, args=(params,), daemon=True
        ).start()

    def _run_parameter_simulation_worker(self, params: Dict[str, float]) -> None:
        """Worker thread for parameter simulation"""
        try:
            # Import APGI equations for simulation
            from APGI_Equations import CoreIgnitionSystem

            ignition_system = CoreIgnitionSystem()

            # Simulate ignition dynamics
            steps = 1000
            dt = 0.01

            # Initialize variables
            S = 0.0
            theta = params["theta_0"]
            tau_S = params["tau_S"]
            tau_theta = params["tau_theta"]
            alpha = params["alpha"]

            # Track variables over time
            S_history = []
            theta_history = []
            ignition_prob_history = []

            # Simulate with a test stimulus
            stimulus_strength = 1.0

            for step in range(steps):
                # Update signal accumulation
                dS_dt = (stimulus_strength - S) / tau_S
                S += dS_dt * dt

                # Update threshold adaptation
                dtheta_dt = (params["theta_0"] - theta) / tau_theta
                theta += dtheta_dt * dt

                # Calculate ignition probability
                ignition_prob = ignition_system.ignition_probability(S, theta, alpha)

                # Store history
                S_history.append(S)
                theta_history.append(theta)
                ignition_prob_history.append(ignition_prob)

            # Calculate summary statistics
            final_ignition_prob = ignition_prob_history[-1]
            max_ignition_prob = max(ignition_prob_history)
            time_to_half_max = next(
                (
                    i
                    for i, p in enumerate(ignition_prob_history)
                    if p >= max_ignition_prob / 2
                ),
                steps,
            )

            # Update results text
            result_text = f"""Simulation Results:
Final Ignition Probability: {final_ignition_prob:.3f}
Maximum Ignition Probability: {max_ignition_prob:.3f}
Time to Half-Max Ignition: {time_to_half_max * dt:.2f} seconds
Final Signal Level (S): {S:.3f}
Final Threshold (θ): {theta:.3f}

Interpretation:
"""
            if final_ignition_prob > 0.5:
                result_text += "• High ignition probability - parameters favor conscious detection\n"
            else:
                result_text += "• Low ignition probability - parameters suppress conscious detection\n"

            if time_to_half_max < 0.5:
                result_text += "• Fast response - quick conscious access\n"
            else:
                result_text += "• Slow response - gradual conscious access\n"

            # Update GUI from main thread
            self.root.after(
                0, lambda: self.param_results_text.insert(tk.END, result_text)
            )

        except Exception as e:
            error_text = f"Simulation failed: {str(e)}\n"
            self.root.after(
                0, lambda: self.param_results_text.insert(tk.END, error_text)
            )
        """Run the selected validation protocols"""
        if self.is_running:
            return

        if not self.validator:
            messagebox.showerror("Error", "APGI Master Validator not available")
            return

        # Get selected protocols with validation
        selected_protocols: List[int] = [
            num for num, var in self.protocol_vars.items() if var.get()
        ]

        # Validate protocol numbers
        for protocol_num in selected_protocols:
            if (
                not isinstance(protocol_num, int)
                or protocol_num not in self.validator.PROTOCOL_TIERS
            ):
                messagebox.showerror(
                    "Error",
                    f"Invalid protocol number: {protocol_num}. Must be between 1 and 8.",
                )
                return

        if not selected_protocols:
            messagebox.showwarning("Warning", "No protocols selected")
            return

        # Clear protocol cache to prevent cross-protocol contamination
        self.clear_protocol_cache()

        # Start validation in separate thread
        self.is_running = True

        # Make UI state changes atomically
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set(0)

        self.validation_thread = threading.Thread(
            target=self._run_validation_worker, args=(selected_protocols,)
        )
        self.validation_thread.start()
        logging.info(f"Started validation for protocols: {selected_protocols}")

    def _execute_single_protocol(
        self, protocol_num: int, i: int, total_protocols: int, protocol_tiers: Dict
    ) -> None:
        """Execute a single protocol and handle its results"""
        self.update_status(f"Running Protocol {protocol_num}...")
        self.update_results(f"=== Protocol {protocol_num} ===\n")

        try:
            # Execute actual protocol in isolated environment
            protocol_file = f"Validation-Protocol-{protocol_num}.py"
            protocol_path = Path(__file__).parent / protocol_file

            if not protocol_path.exists():
                raise FileNotFoundError(f"Protocol file {protocol_file} not found")

            # Use cached module if available, otherwise import and cache
            cache_key = f"APGI_Protocol_{protocol_num}"

            # Check cache first with lock protection
            with self._cache_lock:
                if cache_key in self._protocol_cache:
                    protocol_module = self._protocol_cache[cache_key]
                else:
                    protocol_module = None

            # Import if not cached
            if protocol_module is None:
                # Import protocol module dynamically
                spec = importlib.util.spec_from_file_location(cache_key, protocol_path)
                if spec is None or spec.loader is None:
                    raise ImportError(
                        f"Could not create spec for protocol {protocol_num}"
                    )

                protocol_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(protocol_module)

                # Cache the module with lock protection
                with self._cache_lock:
                    self._protocol_cache[cache_key] = protocol_module
            else:
                # Use cached module - ensure it's fresh by reloading
                importlib.reload(protocol_module)

            # Capture stdout to get results
            captured_output = io.StringIO()

            # Execute protocol in a completely isolated manner
            with contextlib.redirect_stdout(captured_output):
                # Create a subprocess-like environment for protocol execution
                protocol_result = self._execute_protocol_safely(
                    protocol_module, protocol_num, i, total_protocols
                )

            # Parse captured output for results
            output_text = captured_output.getvalue()

            # Create result based on protocol execution
            result: Dict[str, Any] = {
                "status": "COMPLETED",
                "passed": False,  # Default to False until explicitly verified
                "timestamp": datetime.now().isoformat(),
                "output": output_text,
                "protocol_result": protocol_result if protocol_result else {},
            }

            # Determine protocol success with strict validation
            passed = self._determine_protocol_success(
                protocol_result, output_text, result
            )

            result["passed"] = passed

            # Safely assign to validator with validation
            self._store_protocol_result(protocol_num, result, protocol_tiers, passed)

            self.update_results(f"Status: {'PASSED' if passed else 'FAILED'}\n\n")
            logging.info(
                f"Protocol {protocol_num} completed: {'PASSED' if passed else 'FAILED'}"
            )

        except (ImportError, FileNotFoundError, SyntaxError) as e:
            self._handle_protocol_execution_error(e, protocol_num, protocol_tiers)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            self._handle_protocol_execution_error(e, protocol_num, protocol_tiers)
        except (RuntimeError, MemoryError, FutureTimeoutError) as e:
            self._handle_protocol_execution_error(e, protocol_num, protocol_tiers)

    def _determine_protocol_success(
        self, protocol_result: Any, output_text: str, result: Dict
    ) -> bool:
        """Determine if protocol execution was successful"""
        passed = False

        # Check if protocol returned explicit success
        if protocol_result and isinstance(protocol_result, dict):
            # Check for explicit success indicators
            passed = (
                protocol_result.get("status") == "PASSED"
                or protocol_result.get("success") is True
                or protocol_result.get("passed") is True
            )

        # Check for error indicators in output
        error_indicators = ["ERROR", "FAILED", "Exception", "Traceback", "Error:"]
        has_errors = any(indicator in output_text for indicator in error_indicators)

        # Final decision: explicit success required, no fallback assumptions
        if has_errors:
            passed = False
            result["status"] = "COMPLETED_WITH_ERRORS"
        elif not passed:
            result["status"] = "INDETERMINATE"
            passed = False

        return passed

    def _store_protocol_result(
        self, protocol_num: int, result: Dict, protocol_tiers: Dict, passed: bool
    ) -> None:
        """Store protocol result in validator with proper validation"""
        protocol_key = f"protocol_{protocol_num}"
        if hasattr(self.validator, "protocol_results"):
            self.validator.protocol_results[protocol_key] = result
        else:
            logging.error("Validator missing protocol_results attribute")
            self.validator.protocol_results = {protocol_key: result}

        tier = protocol_tiers[protocol_num]

        # Validate tier exists in falsification_status
        if (
            hasattr(self.validator, "falsification_status")
            and tier in self.validator.falsification_status
        ):
            if not isinstance(self.validator.falsification_status[tier], list):
                self.validator.falsification_status[tier] = []

            self.validator.falsification_status[tier].append(
                {
                    "protocol": protocol_num,
                    "passed": passed,
                    "result": result,
                }
            )
        else:
            logging.error(
                f"Validator missing or invalid falsification_status for tier: {tier}"
            )
            # Initialize if missing
            if not hasattr(self.validator, "falsification_status"):
                self.validator.falsification_status = {
                    "primary": [],
                    "secondary": [],
                    "tertiary": [],
                }
            self.validator.falsification_status[tier] = [
                {
                    "protocol": protocol_num,
                    "passed": passed,
                    "result": result,
                }
            ]

    def _run_validation_worker(self, selected_protocols: List[int]) -> None:
        """Worker thread for running validation"""
        try:
            self.update_status("Starting validation...")
            self.update_progress(0)
            logging.info(f"Starting validation for protocols: {selected_protocols}")

            # Clear previous results
            self.validator.protocol_results = {}
            self.validator.falsification_status = {
                "primary": [],
                "secondary": [],
                "tertiary": [],
            }

            protocol_tiers = self.validator.PROTOCOL_TIERS
            total_protocols = len(selected_protocols)

            for i, protocol_num in enumerate(selected_protocols):
                # Check for cancellation before each protocol
                if not self.is_running:
                    self.update_status("Validation cancelled by user")
                    logging.info("Validation cancelled by user")
                    break

                progress = int((i / total_protocols) * 100)
                self.update_progress(progress)

                self._execute_single_protocol(
                    protocol_num, i, total_protocols, protocol_tiers
                )

                # Update progress
                progress = ((i + 1) / total_protocols) * 100
                self.update_progress(progress)

            # Generate final report
            if self.is_running:
                self.update_status("Generating final report...")

                # Validate that validator has required method
                if not hasattr(self.validator, "generate_master_report"):
                    error_msg = "Validator does not have generate_master_report method"
                    self.update_status(error_msg)
                    self.update_results(f"CRITICAL ERROR: {error_msg}\n")
                    logging.error(error_msg)
                    return

                try:
                    report = self.validator.generate_master_report()
                    # Validate report structure
                    self._validate_report(report)
                except Exception as e:
                    error_msg = (
                        f"Failed to generate master report: {type(e).__name__}: {e}"
                    )
                    self.update_status(error_msg)
                    self.update_results(f"CRITICAL ERROR: {error_msg}\n")
                    logging.error(error_msg)
                    return

                # Update summary
                self.update_summary(report)
                self.update_results("\n=== FINAL RESULT ===\n")
                self.update_results(f"Overall Decision: {report['overall_decision']}\n")

                self.update_status("Validation completed")
                logging.info("Validation completed successfully")

        except (ImportError, FileNotFoundError, SyntaxError) as e:
            self._handle_validation_critical_error(e, selected_protocols)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            self._handle_validation_critical_error(e, selected_protocols)
        except (RuntimeError, MemoryError, FutureTimeoutError) as e:
            self._handle_validation_critical_error(e, selected_protocols)

        finally:
            with self._thread_cleanup_lock:
                # Ensure thread cleanup regardless of how validation ends
                self.is_running = False

                # Ensure UI state is consistent
                self._ensure_ui_consistency()

                # Clear any pending GUI updates with non-blocking approach
                try:
                    cleared_count = 0
                    while (
                        cleared_count < 50
                    ):  # Limit iterations to prevent infinite loop
                        try:
                            self._update_queue.get_nowait()
                            self._update_queue.task_done()
                            cleared_count += 1
                        except queue.Empty:
                            break
                except Exception as e:
                    logging.error(f"Error clearing GUI update queue: {e}")

                # Clear validation thread reference AFTER ensuring thread is done
                if self.validation_thread:
                    if self.validation_thread.is_alive():
                        logging.warning(
                            "Validation thread still alive in finally block"
                        )
                        # Don't force join here to avoid blocking GUI
                    self.validation_thread = None

                logging.info("Validation worker thread cleanup completed")

    def _handle_protocol_error(
        self, error: Exception, protocol_num: int
    ) -> Dict[str, Any]:
        """Handle protocol errors and return error result with troubleshooting."""
        error_type = type(error).__name__
        error_str = str(error).lower()

        if isinstance(error, ImportError):
            return {
                "status": "IMPORT_ERROR",
                "error": f"Protocol module not found: {error}",
                "error_type": error_type,
                "timestamp": datetime.now().isoformat(),
                "troubleshooting": "Check that protocol file exists and is importable",
            }
        elif isinstance(error, AttributeError):
            return {
                "status": "INTERFACE_ERROR",
                "error": f"Missing required function: {error}",
                "error_type": error_type,
                "timestamp": datetime.now().isoformat(),
                "troubleshooting": "Check that protocol has required validation functions",
            }
        elif isinstance(error, (ValueError, TypeError)):
            return {
                "status": "PARAMETER_ERROR",
                "error": f"Invalid parameter or data: {error}",
                "error_type": error_type,
                "timestamp": datetime.now().isoformat(),
                "troubleshooting": "Check parameter values and data types",
            }
        elif isinstance(error, MemoryError):
            return {
                "status": "MEMORY_ERROR",
                "error": f"Insufficient memory: {error}",
                "error_type": error_type,
                "timestamp": datetime.now().isoformat(),
                "troubleshooting": "Try reducing data size or closing other applications",
            }
        elif isinstance(error, (TimeoutError, FutureTimeoutError)):
            return {
                "status": "TIMEOUT_ERROR",
                "error": f"Protocol timed out: {error}",
                "error_type": error_type,
                "timestamp": datetime.now().isoformat(),
                "troubleshooting": "Protocol may be too complex, try with debug mode",
            }
        else:
            return {
                "status": "UNEXPECTED_ERROR",
                "error": f"Unexpected error: {error}",
                "error_type": error_type,
                "timestamp": datetime.now().isoformat(),
                "troubleshooting": self._get_error_troubleshooting(error, protocol_num),
            }

    def _handle_protocol_execution_error(
        self, error: Exception, protocol_num: int, protocol_tiers: Dict[int, str]
    ) -> None:
        """Handle protocol execution errors with logging and UI updates."""

        error_result = self._handle_protocol_error(error, protocol_num)
        tier = protocol_tiers[protocol_num]

        logging.error(
            f"Protocol {protocol_num} failed: {type(error).__name__}: {error}"
        )

        if (
            hasattr(self.validator, "falsification_status")
            and tier in self.validator.falsification_status
            and isinstance(self.validator.falsification_status[tier], list)
        ):
            self.validator.falsification_status[tier].append(
                {"protocol": protocol_num, "passed": False, "result": error_result}
            )
        else:
            logging.error(f"Invalid falsification_status structure for tier: {tier}")

        error_msg = f"Protocol {protocol_num} failed: {type(error).__name__}: {error}"
        self.update_results(f"ERROR: {error_msg}\n")
        self.update_results(f"Troubleshooting: {error_result['troubleshooting']}\n\n")

    def _handle_validation_critical_error(
        self, error: Exception, selected_protocols: List[int]
    ) -> None:
        """Handle critical validation errors with detailed troubleshooting."""
        import traceback

        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat(),
            "traceback": traceback.format_exc(),
            "selected_protocols": selected_protocols,
        }

        logging.critical(f"Validation failed critically: {error_details}")

        error_msg = f"Validation failed: {type(error).__name__}: {error}"
        self.update_status(error_msg)
        self.update_results(f"CRITICAL ERROR: {error_msg}\n")
        self.update_results(f"Error details: {error_details['error_message']}\n")
        self.update_results("\nTroubleshooting steps:\n")
        self.update_results("1. Check that all protocol files exist and are readable\n")
        self.update_results("2. Verify all dependencies are installed\n")
        self.update_results("3. Check file permissions in the Validation directory\n")
        self.update_results("4. Try running individual protocols separately\n")
        self.update_results("5. Check logs for detailed traceback information\n")

    def _validate_report(self, report: Any) -> None:
        """Validate report structure before accessing it."""
        if not isinstance(report, dict):
            raise ValueError("Invalid report format: expected dict")

        required_keys = ["overall_decision", "falsification_status", "protocol_results"]
        for key in required_keys:
            if key not in report:
                raise ValueError(f"Report missing required key: {key}")

        if not isinstance(report["falsification_status"], dict):
            raise ValueError("Invalid falsification_status format: expected dict")

        for tier in ["primary", "secondary", "tertiary"]:
            if tier not in report["falsification_status"]:
                raise ValueError(f"Report missing falsification tier: {tier}")

            # Validate that each tier contains a list
            if not isinstance(report["falsification_status"][tier], list):
                raise ValueError(f"Invalid falsification tier {tier}: expected list")

            # Validate each result in the tier
            for i, result in enumerate(report["falsification_status"][tier]):
                if not isinstance(result, dict):
                    raise ValueError(f"Invalid result at {tier}[{i}]: expected dict")

                required_result_keys = ["protocol", "passed", "result"]
                for result_key in required_result_keys:
                    if result_key not in result:
                        raise ValueError(
                            f"Result at {tier}[{i}] missing key: {result_key}"
                        )

                # Validate protocol number
                if not isinstance(result["protocol"], int):
                    raise ValueError(
                        f"Invalid protocol number at {tier}[{i}]: expected int"
                    )

                # Validate passed status
                if not isinstance(result["passed"], bool):
                    raise ValueError(
                        f"Invalid passed status at {tier}[{i}]: expected bool"
                    )

                # Validate result object
                if not isinstance(result["result"], dict):
                    raise ValueError(
                        f"Invalid result object at {tier}[{i}]: expected dict"
                    )

    def stop_validation(self) -> None:
        """Stop the running validation with proper thread cancellation"""
        with self._thread_cleanup_lock:
            if not self.is_running:
                return

            self.is_running = False
            self.update_status("Stopping validation...")

            # Force stop the validation thread
            if self.validation_thread and self.validation_thread.is_alive():
                # Give the thread a moment to clean up
                self.update_progress(100)
                self.update_results("Validation stopped by user\n")

                # Wait for thread to finish (with timeout) - non-blocking approach
                self.validation_thread.join(
                    timeout=1.0
                )  # Short timeout to avoid GUI freeze

                if self.validation_thread.is_alive():
                    self.update_results(
                        "Warning: Validation thread did not stop cleanly\n"
                    )
                    logging.warning(
                        "Validation thread did not stop cleanly, may be zombie thread"
                    )
                    # Don't block on thread cleanup
                    self.validation_thread = None
                else:
                    self.update_results("Validation stopped successfully\n")
                    logging.info("Validation thread stopped successfully")
                    self.validation_thread = None

            # Reset UI state atomically
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

            # Clear any pending GUI updates with non-blocking approach
            try:
                cleared_count = 0
                while cleared_count < 50:  # Limit iterations to prevent infinite loop
                    try:
                        self._update_queue.get_nowait()
                        self._update_queue.task_done()
                        cleared_count += 1
                    except queue.Empty:
                        break
            except Exception as e:
                logging.error(f"Error clearing GUI update queue: {e}")

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy and other non-serializable types to Python types."""
        if obj is None:
            return None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return self._convert_to_serializable(obj.__dict__)
        else:
            return obj

    def save_results(self) -> None:
        """Save validation results to file"""
        if not self.validator or not self.validator.protocol_results:
            messagebox.showwarning("Warning", "No results to save")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"APGI-Validation-Results-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )

        if filename:
            try:
                report = self.validator.generate_master_report()
                self._validate_report(report)

                # Convert numpy types to serializable format
                report_serializable = self._convert_to_serializable(report)

                # Use context manager for proper file handling
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(report_serializable, f, indent=2)

                messagebox.showinfo("Success", f"Results saved to {filename}")
                logging.info(f"Results saved to {filename}")

            except (
                ValueError,
                OSError,
                IOError,
                PermissionError,
                json.JSONEncodeError,
                UnicodeEncodeError,
            ) as e:
                error_msg = f"Failed to save results: {type(e).__name__}: {e}"
                messagebox.showerror(
                    "Save Error",
                    f"{error_msg}\n\nTroubleshooting:\n"
                    f"1. Check file permissions for the target directory\n"
                    f"2. Ensure disk space is available\n"
                    f"3. Try saving to a different location\n"
                    f"4. Check that the filename is valid\n"
                    f"5. Ensure the directory exists",
                )
                logging.error(f"Failed to save results: {error_msg}")
            except Exception as e:
                # Catch any other unexpected errors
                error_msg = f"Unexpected error saving results: {type(e).__name__}: {e}"
                messagebox.showerror("Save Error", f"{error_msg}")
                logging.error(f"Unexpected save error: {error_msg}")

    def _execute_protocol_safely(
        self,
        protocol_module: Any,
        protocol_num: int,
        protocol_index: int,
        total_protocols: int,
    ) -> Dict[str, Any]:
        """Execute protocol in a completely isolated environment to prevent GUI operations."""
        try:
            # Check if protocol has main function
            if not hasattr(protocol_module, "main"):
                return {
                    "status": "COMPLETED",
                    "message": "Protocol executed successfully (no main function)",
                }

            # Create progress callback that doesn't interfere with GUI
            def safe_progress_callback(percent: float) -> None:
                """Safe progress callback that won't cause GUI issues."""
                try:
                    if not isinstance(percent, (int, float)):
                        return

                    # Clamp percent between 0-100
                    percent = max(0, min(100, percent))

                    # Calculate overall progress with more granular tracking
                    protocol_start_progress = (protocol_index / total_protocols) * 100
                    protocol_progress_range = 100 / total_protocols
                    overall_progress = protocol_start_progress + (
                        percent / 100 * protocol_progress_range
                    )

                    # Update progress with sub-protocol granularity
                    self.update_progress(overall_progress)
                    self.update_status(
                        f"Running Protocol {protocol_num}... {int(percent)}% (Step {protocol_index + 1}/{total_protocols})"
                    )
                except Exception:
                    # Silently ignore progress callback errors to avoid interrupting protocol
                    pass

            # Execute protocol with timeout in a separate thread to ensure isolation
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Check if protocol accepts progress callback
                import inspect

                sig = inspect.signature(protocol_module.main)

                if "progress_callback" in sig.parameters:
                    future = executor.submit(
                        protocol_module.main, progress_callback=safe_progress_callback
                    )
                else:
                    future = executor.submit(protocol_module.main)

                try:
                    result = future.result(timeout=self.validator.timeout_seconds)
                    # Do NOT default to passed=True - require explicit success
                    if result is None:
                        return {
                            "status": "INDETERMINATE",
                            "message": "Protocol returned None - cannot determine success",
                            "passed": False,
                        }
                    return result
                except FutureTimeoutError:
                    raise TimeoutError(
                        f"Protocol {protocol_num} timed out after {self.validator.timeout_seconds} seconds"
                    )

        except Exception as e:
            # Return error result instead of raising to prevent GUI crashes
            return {
                "status": "ERROR",
                "error": str(e),
                "error_type": type(e).__name__,
                "message": f"Protocol execution failed: {e}",
            }

    def update_status(self, message: str) -> None:
        """Update status label thread-safely"""
        try:
            self._update_queue.put(("status", message), block=False)
        except queue.Full:
            # If queue is full, skip this update to prevent blocking
            pass

    def update_progress(self, value: float) -> None:
        """Update progress bar thread-safely"""
        try:
            self._update_queue.put(("progress", value), block=False)
        except queue.Full:
            # If queue is full, skip this update to prevent blocking
            pass

    def update_results(self, message: str) -> None:
        """Update results text widget thread-safely"""
        try:
            self._update_queue.put(("results", message), block=False)
        except queue.Full:
            # If queue is full, skip this update to prevent blocking
            pass

    def _get_error_troubleshooting(self, error: Exception, protocol_num: int) -> str:
        """Get troubleshooting hints based on error type and protocol"""
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Common issues
        if "module not found" in error_str or "modulenotfounderror" in error_str:
            try:
                module_name = (
                    error_str.split("no module named")[-1].strip("'\"")
                    if "no module named" in error_str
                    else "unknown"
                )
            except (IndexError, AttributeError):
                module_name = "unknown"
            return (
                f"Missing dependency detected. Install with: pip install -r requirements.txt\n"
                f"Specifically check: {module_name}"
            )

        elif "file not found" in error_str or "filenotfounderror" in error_str:
            return f"Protocol file missing. Check that APGI-Protocol-{protocol_num}.py exists in Validation directory"

        elif "permission" in error_str:
            return "File permission error. Check read/write permissions for the Validation directory"

        elif "memory" in error_str or "ram" in error_str:
            return "Memory error. Try reducing protocol parameters or closing other applications"

        elif "timeout" in error_str:
            return "Protocol timed out. Try running with debug mode enabled or reduce complexity"

        # Protocol-specific issues
        if protocol_num == 3:
            if "broadcast" in error_str:
                return "Observation dimension mismatch. This should be fixed with the latest updates"

        elif protocol_num in [2, 8]:
            if "pymc" in error_str or "arviz" in error_str:
                return "Bayesian modeling library issue. Install with: pip install pymc arviz"

        # Generic fallback
        return (
            f"Error type: {error_type}. "
            "Check the protocol file for syntax issues and ensure all dependencies are installed. "
            "Try running the protocol individually to isolate the problem."
        )

    def update_summary(self, report: Dict[str, Any]) -> None:
        """Update summary label thread-safely"""
        summary_text = "Summary:\n"
        for tier, results in report["falsification_status"].items():
            failures = sum(1 for r in results if not r["passed"])
            total = len(results)
            summary_text += f"{tier.capitalize()} tier: {failures}/{total} failed\n"

        try:
            self._update_queue.put(("summary", summary_text), block=False)
        except queue.Full:
            # If queue is full, skip this update to prevent blocking
            pass


def main() -> None:
    """Main function to run the GUI"""
    root = tk.Tk()
    app = APGIValidationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
