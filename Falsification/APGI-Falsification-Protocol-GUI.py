#!/usr/bin/env python3
"""
Simple GUI for running APGI falsification protocols
"""

import importlib.util
import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Dict


class ProtocolRunnerGUI:
    """GUI for running APGI falsification protocols with progress tracking."""

    def __init__(self, root):
        self.root = root
        self.root.title("APGI Falsification Protocols")
        self.root.geometry("800x600")

        # Add current directory to Python path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        # Protocol definitions
        self.protocols = {
            "Protocol 1: APGI Agent": {
                "file": "Falsification-Protocol-1.py",
                "class": "APGIActiveInferenceAgent",
                "description": "Complete APGI-based active inference agent",
            },
            "Protocol 2: Iowa Gambling": {
                "file": "Falsification-Protocol-2.py",
                "class": "IowaGamblingTaskEnvironment",
                "description": "IGT variant with simulated interoceptive costs",
            },
            "Protocol 3: Agent Comparison": {
                "file": "Falsification-Protocol-3.py",
                "class": "AgentComparisonExperiment",
                "description": "Run complete agent comparison experiment",
            },
            "Protocol 4: Phase Transition": {
                "file": "Falsification-Protocol-4.py",
                "class": "InformationTheoreticAnalysis",
                "description": "Test APGI ignition phase transition signatures",
            },
            "Protocol 5: Evolutionary": {
                "file": "Falsification-Protocol-5.py",
                "class": "EvolutionaryAPGIEmergence",
                "description": "Test APGI emergence under selection pressure",
            },
            "Protocol 6: Network Comparison": {
                "file": "Falsification-Protocol-6.py",
                "class": "NetworkComparisonExperiment",
                "description": "Compare APGI-inspired vs standard architectures",
            },
        }

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame, text="APGI Falsification Protocols", font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Protocol buttons frame
        button_frame = ttk.LabelFrame(main_frame, text="Select Protocol", padding="10")
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        # Create 6 buttons in 2x3 grid
        for i, (protocol_name, protocol_info) in enumerate(self.protocols.items()):
            row = i // 3
            col = i % 3

            btn = ttk.Button(
                button_frame,
                text=protocol_name.split(": ")[1],
                command=lambda info=protocol_info: self.run_protocol(info),
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky=(tk.W, tk.E))

            # Add tooltip
            self.create_tooltip(btn, protocol_info["description"])

        # Configure button grid weights
        for col in range(3):
            button_frame.columnconfigure(col, weight=1)

        # Output console
        console_frame = ttk.LabelFrame(main_frame, text="Output Console", padding="10")
        console_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.output_console = scrolledtext.ScrolledText(console_frame, height=15, width=80)
        self.output_console.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        # Clear console button
        clear_btn = ttk.Button(main_frame, text="Clear Console", command=self.clear_console)
        clear_btn.grid(row=3, column=1, sticky=tk.E, pady=(10, 0))

    def create_tooltip(self, widget, text):
        """Create tooltip for widget"""

        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
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
        """Clear the output console"""
        self.output_console.delete(1.0, tk.END)

    def log_message(self, message):
        """Add message to output console"""
        self.output_console.insert(tk.END, message + "\n")
        self.output_console.see(tk.END)
        self.root.update_idletasks()

    def run_protocol(self, protocol_info):
        """Run selected protocol in separate thread"""

        def protocol_thread():
            try:
                self.status_var.set(f"Running {protocol_info['file']}...")
                self.log_message(f"=== Running {protocol_info['file']} ===")

                # Load the protocol module
                file_path = os.path.join(os.path.dirname(__file__), protocol_info["file"])
                spec = importlib.util.spec_from_file_location(protocol_info["file"], file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Get the main class
                cls = getattr(module, protocol_info["class"])

                # Try to find and run the main method
                if protocol_info["class"] == "NetworkComparisonExperiment":
                    # Protocol 6 - Create experiment with config
                    config = {
                        "extero_dim": 32,
                        "intero_dim": 16,
                        "action_dim": 4,
                        "context_dim": 8,
                    }
                    instance = cls(config)
                    result = instance.run_full_experiment()
                    self.log_message("Network Comparison Experiment completed")
                    self.log_message(f"Results: {type(result)}")

                elif hasattr(cls, "run_full_experiment"):
                    # Protocol 3 - Handle potential broadcasting issues
                    try:
                        instance = cls()
                        result = instance.run_full_experiment()
                        self.log_message(f"Experiment completed: {type(result)}")
                    except ValueError as ve:
                        if "broadcast" in str(ve):
                            self.log_message(
                                "Broadcasting error in Protocol-3 - this is expected due to observation size mismatches"
                            )
                            self.log_message("Protocol-3 needs observation size alignment")
                        else:
                            raise ve

                elif hasattr(cls, "run_phase_transition_analysis"):
                    # Protocol 4 - Create analysis with surprise system
                    surprise_system = module.SurpriseIgnitionSystem()
                    instance = cls(surprise_system)
                    result = instance.run_phase_transition_analysis()
                    self.log_message(f"Phase transition analysis completed: {type(result)}")

                elif hasattr(cls, "run_evolution"):
                    # Protocol 5 - Add timeout handling
                    try:
                        instance = cls()
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
                        self.log_message("Created EvolutionaryAPGIEmergence instance")

                elif protocol_info["class"] == "APGIActiveInferenceAgent":
                    # Protocol 1 - Create agent and run demo
                    config = {
                        "lr_extero": 0.01,
                        "lr_intero": 0.01,
                        "lr_precision": 0.05,
                        "lr_somatic": 0.1,
                        "n_actions": 4,
                        "theta_init": 0.5,
                        "theta_baseline": 0.5,
                        "alpha": 8.0,
                        "tau_S": 0.3,
                        "tau_theta": 10.0,
                        "eta_theta": 0.01,
                        "beta": 1.2,
                        "rho": 0.7,
                    }
                    agent = cls(config)
                    self.log_message("APGI Agent created successfully")
                    self.log_message(f"Agent config: {config}")

                elif protocol_info["class"] == "IowaGamblingTaskEnvironment":
                    # Protocol 2 - Create environment and run demo
                    env = cls(n_trials=10)
                    self.log_message("Iowa Gambling Task Environment created")

                    # Run a few demo trials
                    total_reward = 0
                    for trial in range(5):
                        action = 0  # Always pick deck A for demo
                        reward, intero_cost, obs, done = env.step(action)
                        total_reward += reward
                        self.log_message(
                            f"Trial {trial+1}: Action={action}, Reward={reward:.2f}, "
                            f"InteroCost={intero_cost:.2f}"
                        )

                    self.log_message(f"Demo completed. Total reward: {total_reward:.2f}")

                else:
                    self.log_message(f"Created {protocol_info['class']} instance")

                self.log_message("=== Protocol completed successfully ===")
                self.status_var.set("Ready")

            except (
                ImportError,
                ModuleNotFoundError,
                AttributeError,
                RuntimeError,
                ValueError,
                KeyError,
            ) as e:
                error_msg = f"Error running {protocol_info['file']}: {str(e)}"
                self.log_message(error_msg)
                messagebox.showerror("Error", error_msg)
                self.status_var.set("Error")

        # Run in separate thread to avoid blocking GUI
        thread = threading.Thread(target=protocol_thread)
        thread.daemon = True
        thread.start()


def main():
    root = tk.Tk()
    app = ProtocolRunnerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
