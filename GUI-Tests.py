#!/usr/bin/env python3
"""
Tests GUI for APGI validation framework
======================================

A simple GUI for running and displaying test results.
"""

import subprocess
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any, Dict
from pathlib import Path


class TestsGUI:
    """Simple GUI for running tests.
    ======================================
    Provides a tkinter-based interface to run tests with real-time
    output display and results visualization.
    """

    def __init__(self, root: tk.Tk = None):
        """Initialize the Tests GUI.

        Args:
            root: Tkinter root window (creates new one if None)
        """
        self.root = root or tk.Tk()
        self.root.title("APGI Tests GUI")
        self.root.geometry("800x600")

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Test configuration frame
        config_frame = ttk.LabelFrame(
            main_frame, text="Test Configuration", padding="5"
        )
        config_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        # Test type selection
        ttk.Label(config_frame, text="Test Type:").grid(row=0, column=0, sticky=tk.W)
        self.test_type_var = tk.StringVar(value="unit")
        test_types = ["unit", "integration", "performance", "all"]
        test_type_combo = ttk.Combobox(
            config_frame, textvariable=self.test_type_var, values=test_types
        )
        test_type_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)

        # Coverage checkbox
        self.coverage_var = tk.BooleanVar(value=True)
        coverage_check = ttk.Checkbutton(
            config_frame, text="Run with Coverage", variable=self.coverage_var
        )
        coverage_check.grid(row=0, column=2, padx=5)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        self.run_button = ttk.Button(
            button_frame, text="Run Tests", command=self.run_tests_gui
        )
        self.run_button.grid(row=0, column=0, padx=5)

        self.clear_button = ttk.Button(
            button_frame, text="Clear Output", command=self.clear_output
        )
        self.clear_button.grid(row=0, column=1, padx=5)

        # Output display
        output_frame = ttk.LabelFrame(main_frame, text="Test Output", padding="5")
        output_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.output_text = scrolledtext.ScrolledText(output_frame, height=20, width=80)
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def run_tests_gui(self):
        """Run tests from GUI."""
        config = {
            "test_type": self.test_type_var.get(),
            "coverage": self.coverage_var.get(),
        }

        results = self.run_tests(config)
        self.display_results(results)

    def clear_output(self):
        """Clear the output display."""
        self.output_text.delete(1.0, tk.END)

    def run_tests(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests based on configuration.

        Args:
            config: Test configuration dictionary

        Returns:
            Dictionary containing test results
        """
        test_type = config.get("test_type", "unit")
        coverage = config.get("coverage", False)

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", "tests/"]

        # Only filter by marker if tests have markers (currently they don't)
        if test_type != "all":
            # TODO: Add markers to tests or remove this filtering
            pass

        if coverage:
            cmd.append("--cov=.")
            cmd.append("--cov-report=term-missing")

        # Add verbose output
        cmd.append("-v")

        # Run tests
        try:
            cwd = Path(__file__).parent
            self.output_text.insert(tk.END, f"Running: {' '.join(cmd)}\n")
            self.output_text.insert(tk.END, f"Working directory: {cwd}\n")
            self.output_text.update()

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)

            return {
                "test_results": {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                },
                "config": config,
            }

        except Exception as e:
            return {
                "test_results": {
                    "success": False,
                    "error": str(e),
                    "stdout": "",
                    "stderr": "",
                    "returncode": -1,
                },
                "config": config,
            }

    def display_results(self, results: Dict[str, Any]):
        """Display test results in the GUI.

        Args:
            results: Test results dictionary
        """
        self.output_text.delete(1.0, tk.END)

        test_results = results.get("test_results", {})
        config = results.get("config", {})

        # Display configuration
        self.output_text.insert(tk.END, "Test Configuration:\n")
        self.output_text.insert(
            tk.END, f"Test Type: {config.get('test_type', 'unknown')}\n"
        )
        self.output_text.insert(
            tk.END, f"  Coverage: {config.get('coverage', False)}\n"
        )
        self.output_text.insert(tk.END, "\n")

        # Display results
        if test_results.get("success", False):
            self.output_text.insert(tk.END, "✅ Tests PASSED\n", "success")
        else:
            self.output_text.insert(tk.END, "❌ Tests FAILED\n", "error")

        # Display stdout
        stdout = test_results.get("stdout", "")
        if stdout:
            self.output_text.insert(tk.END, "\nStandard Output:\n")
            self.output_text.insert(tk.END, stdout)

        # Display stderr
        stderr = test_results.get("stderr", "")
        if stderr:
            self.output_text.insert(tk.END, "\nStandard Error:\n")
            self.output_text.insert(tk.END, stderr)

        # Display error if any
        error = test_results.get("error")
        if error:
            self.output_text.insert(tk.END, "\nError:\n")
            self.output_text.insert(tk.END, error, "error")

        # Configure text tags for coloring
        self.output_text.tag_configure("success", foreground="green")
        self.output_text.tag_configure("error", foreground="red")

        # Scroll to bottom
        self.output_text.see(tk.END)

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    # Create and run the GUI
    gui = TestsGUI()
    gui.run()
