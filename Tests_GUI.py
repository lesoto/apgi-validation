#!/usr/bin/env python3
"""
Tests GUI for APGI validation framework
======================================

A simple GUI for running and displaying test results.
"""

import os
import subprocess
import sys

# Add current directory to path
sys.path.insert(0, os.getcwd())
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk
from typing import Any, Dict


def apply_apgi_theme(root):
    """Apply unified APGI theme to tkinter application."""
    style = ttk.Style()
    style.theme_use("clam")

    # Core Palette
    bg_color = "#f8f9fa"
    fg_color = "#212529"

    # Configure Global Elements
    style.configure("TFrame", background=bg_color)
    style.configure(
        "TLabel", background=bg_color, foreground=fg_color, font=("Noto Sans", 10)
    )
    style.configure("Header.TLabel", font=("Noto Sans", 12, "bold"))

    # Custom Card Style
    style.configure("Card.TFrame", background="#ffffff", borderwidth=1, relief="solid")

    # Button Styling
    style.configure("TButton", padding=6, background="#e9ecef")
    style.map(
        "TButton",
        background=[("active", "#dee2e6"), ("disabled", "#f1f3f5")],
        foreground=[("disabled", "#adb5bd")],
    )

    # Primary Button Styling
    style.configure(
        "Primary.TButton",
        background="#155724",
        foreground="white",
        font=("Noto Sans", 10, "bold"),
        padding=8,
    )
    style.map(
        "Primary.TButton",
        background=[("active", "#0f3d1a")],
        foreground=[("active", "white")],
    )

    # Danger Button Styling
    style.configure(
        "Danger.TButton",
        background="#721c24",
        foreground="white",
        font=("Noto Sans", 10, "bold"),
        padding=8,
    )
    style.map(
        "Danger.TButton",
        background=[("active", "#5a161d")],
        foreground=[("active", "white")],
    )

    # Secondary Button Styling
    style.configure(
        "Secondary.TButton",
        background="#2874a6",
        foreground="white",
        font=("Noto Sans", 10),
        padding=6,
    )
    style.map(
        "Secondary.TButton",
        background=[("active", "#1f5a82")],
        foreground=[("active", "white")],
    )

    # Checkbutton Styling
    style.configure("Card.TCheckbutton", background="#ffffff")

    # Status styling
    style.configure(
        "Success.TLabel", foreground="#155724", font=("Noto Sans", 10, "bold")
    )
    style.configure(
        "Error.TLabel", foreground="#721c24", font=("Noto Sans", 10, "bold")
    )

    # Configure root window
    root.configure(background=bg_color)

    return style


class APGIButtons:
    """Standard button configurations for APGI applications."""

    @staticmethod
    def primary(parent, text, command):
        """Primary action button (green)."""
        return ttk.Button(
            parent, text=text, command=command, style="Primary.TButton", cursor="hand2"
        )

    @staticmethod
    def danger(parent, text, command):
        """Danger/Stop button (red)."""
        return ttk.Button(
            parent, text=text, command=command, style="Danger.TButton", cursor="hand2"
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


class APGICard(ttk.Frame):
    """Standardized information card for all APGI apps."""

    def __init__(self, parent, title, **kwargs):
        super().__init__(parent, style="Card.TFrame", **kwargs)
        self.container = ttk.Frame(self, padding=15, style="Card.TFrame")
        self.container.pack(fill="both", expand=True)

        if title:
            self.lbl_title = ttk.Label(
                self.container, text=title.upper(), style="Header.TLabel"
            )
            self.lbl_title.pack(anchor="w", pady=(0, 10))


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
        self.root.geometry("900x700")
        apply_apgi_theme(self.root)

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        # Top metric bar / header
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.pack(fill="x")
        ttk.Label(
            header_frame, text="APGI TEST ENVIRONMENT", style="Header.TLabel"
        ).pack(side="left")

        # Main content
        main_content = ttk.Frame(self.root, padding="10")
        main_content.pack(fill="both", expand=True)

        # Grid layout
        main_content.columnconfigure(0, weight=0, minsize=250)
        main_content.columnconfigure(1, weight=1)
        main_content.rowconfigure(0, weight=1)

        # Left Sidebar (Controls)
        sidebar = ttk.Frame(main_content)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        config_card = APGICard(sidebar, "Test Configuration")
        config_card.pack(fill="x", pady=(0, 15))

        ttk.Label(config_card.container, text="Test Type:").pack(
            anchor="w", pady=(0, 5)
        )
        self.test_type_var = tk.StringVar(value="all")
        test_type_combo = ttk.Combobox(
            config_card.container,
            textvariable=self.test_type_var,
            values=["unit", "integration", "performance", "all"],
            state="readonly",
        )
        test_type_combo.pack(fill="x", pady=(0, 15))

        self.coverage_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            config_card.container,
            text="Run with Coverage",
            variable=self.coverage_var,
            style="Card.TCheckbutton",
        ).pack(anchor="w", pady=(0, 20))

        # Buttons Frame for alignment
        btn_frame = ttk.Frame(sidebar)
        btn_frame.pack(fill="x")

        APGIButtons.primary(btn_frame, "RUN TESTS", self.run_tests_gui).pack(
            fill="x", pady=(0, 10)
        )
        APGIButtons.secondary(btn_frame, "CLEAR OUTPUT", self.clear_output).pack(
            fill="x"
        )

        # Workspace (Output)
        workspace = APGICard(main_content, "Test Output")
        workspace.grid(row=0, column=1, sticky="nsew")

        self.status_label = ttk.Label(workspace.container, text="Ready", style="TLabel")
        self.status_label.pack(anchor="w", pady=(0, 5))

        self.output_text = scrolledtext.ScrolledText(
            workspace.container,
            bg="#212529",
            fg="#f8f9fa",
            font=("Noto Sans Mono", 10),
            borderwidth=0,
            highlightthickness=0,
        )
        self.output_text.pack(fill="both", expand=True)

    def show_status(self, status_type, message):
        """Update status label."""
        if status_type == "success":
            self.status_label.config(text=f"✔ {message}", style="Success.TLabel")
        elif status_type == "error":
            self.status_label.config(text=f"✖ {message}", style="Error.TLabel")
        else:
            self.status_label.config(text=f"ℹ {message}", style="TLabel")

    def run_tests_gui(self):
        """Run tests from GUI."""
        self.show_status("info", "Running tests...")
        self.root.update()

        config = {
            "test_type": self.test_type_var.get(),
            "coverage": self.coverage_var.get(),
        }

        results = self.run_tests(config)
        self.display_results(results)

    def clear_output(self):
        """Clear the output display."""
        self.output_text.delete(1.0, tk.END)
        self.show_status("info", "Ready")

    def run_tests(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run tests based on configuration.

        Args:
            config: Test configuration dictionary

        Returns:
            Dictionary containing test results
        """
        test_type = config.get("test_type", "all")
        coverage = config.get("coverage", False)

        # Build pytest command
        cmd = [sys.executable, "-m", "pytest", "tests/"]

        # If it's a specific type of test, add marker logic or path logic
        # For APGI validation, usually we just run the tests
        if test_type != "all":
            # If tests have markers
            cmd.extend(["-m", test_type])

        if coverage:
            cmd.append("--cov=.")
            cmd.append("--cov-report=term-missing")

        # Add verbose output
        cmd.append("-v")

        # Run tests
        try:
            cwd = Path(__file__).parent
            self.output_text.insert(tk.END, f"> {' '.join(cmd)}\n\n")
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

        error = test_results.get("error")
        if error:
            self.show_status("error", "Tests Failed - Exception occurred")
            self.output_text.insert(tk.END, f"Exception:\n{error}\n")
            return

        if test_results.get("success", False):
            self.show_status("success", "Tests Passed Successfully")
        else:
            self.show_status("error", "Tests Failed")

        # Display stdout
        stdout = test_results.get("stdout", "")
        if stdout:
            self.output_text.insert(tk.END, stdout)

        # Display stderr
        stderr = test_results.get("stderr", "")
        if stderr:
            self.output_text.insert(tk.END, "\nErrors:\n" + stderr)

        # Scroll to bottom
        self.output_text.see(tk.END)

    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    # Create and run the GUI
    gui = TestsGUI()
    gui.run()
