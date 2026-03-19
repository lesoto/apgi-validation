#!/usr/bin/env python3
"""
GUI to run all tests folder scripts and complete test suite
===========================================================

A tkinter-based GUI that allows running all test scripts in the tests folder
individually, running all scripts sequentially, or running the complete test
suite using pytest with real-time output display and error handling.
"""

import logging
import subprocess
import sys
import threading
import tkinter as tk
import os
from collections import deque
from pathlib import Path
from tkinter import scrolledtext, ttk
from typing import Any, Dict, List, Optional, Tuple

# Set matplotlib backend before importing any matplotlib modules
import matplotlib

# Try to set matplotlib backend with fallback for headless environments
try:
    # Try to use TkAgg for interactive GUI, fallback to Agg if not available
    if (
        "DISPLAY" not in os.environ
        and sys.platform != "win32"
        and sys.platform != "darwin"
    ):
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("TkAgg")
        except (ImportError, RuntimeError):
            matplotlib.use("Agg")
except ImportError:
    pass

# Import matplotlib for visualization
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Advanced visualization disabled.")


class ToolTip:
    """Consistent tooltip implementation for tkinter widgets"""

    def __init__(self, widget: tk.Widget, text: str = "") -> None:
        """Initialize tooltip

        Args:
            widget: The widget to attach tooltip to
            text: Tooltip text to display
        """
        self.widget = widget
        self.text = text
        self.tipwindow: Optional[tk.Toplevel] = None
        self.id: Optional[str] = None
        self._delay = 500

        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<ButtonPress>", self.on_leave)

    def on_enter(self, event: Optional[tk.Event[Any]] = None) -> None:
        """Show tooltip when mouse enters widget"""
        self.schedule()

    def on_leave(self, event: Optional[tk.Event[Any]] = None) -> None:
        """Hide tooltip when mouse leaves widget"""
        self.unschedule()
        self.hidetip()

    def schedule(self) -> None:
        """Schedule tooltip display"""
        self.unschedule()
        self.id = self.widget.after(self._delay, self.showtip)

    def unschedule(self) -> None:
        """Cancel scheduled tooltip display"""
        id = self.id
        self.id = None
        if id is not None:
            self.widget.after_cancel(id)

    def showtip(self) -> None:
        """Display the tooltip"""
        if self.tipwindow or not self.text:
            return
        try:
            bbox = self.widget.bbox("insert")  # type: ignore
            if bbox is None:
                return
            x, y, _, _ = bbox
            x = x + self.widget.winfo_rootx() + 25
            y = y + self.widget.winfo_rooty() + 20
        except tk.TclError:
            # For widgets that don't support bbox("insert") like buttons
            x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("tahoma", 8, "normal"),
        )
        label.pack(padx=1, pady=1)

    def hidetip(self) -> None:
        """Hide the tooltip"""
        tw = self.tipwindow
        self.tipwindow = None
        if tw is not None:
            tw.destroy()

    def update_text(self, new_text: str) -> None:
        """Update tooltip text"""
        self.text = new_text


class TestsRunnerGUI:
    """GUI for running tests scripts and complete test suite.

    Provides a tkinter-based interface to run test scripts from the tests folder
    individually, sequentially, or run the complete test suite using pytest
    with real-time output display, error handling, and process management.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("APGI Tests Scripts Runner")
        self.root.geometry("800x600")
        self.root.minsize(640, 480)  # Prevent resizing below usable size

        # Get tests directory
        self.tests_dir = Path(__file__).parent / "tests"
        self.scripts = self.get_script_list()

        # Store running processes
        self.running_processes: Dict[str, subprocess.Popen[str]] = {}

        # Track running threads for proper cleanup
        self.running_threads: List[threading.Thread] = []

        # Cancellation event for run_all operation
        self.run_all_cancel_event = threading.Event()
        self.run_all_running = False

        # Output tag constants
        self.TAG_INFO = "info"
        self.TAG_ERROR = "error"
        self.TAG_SUCCESS = "success"
        self.TAG_WARNING = "warning"

        # Bounded output buffer to prevent memory leaks
        self.output_buffer_size = 1000  # Maximum number of output lines to keep (reduced from 10,000 for better memory efficiency)
        self.output_buffer: deque[Tuple[str, str]] = deque(
            maxlen=self.output_buffer_size
        )

        self.setup_ui()

        # Configure tab order for keyboard navigation
        self._setup_tab_order()

        # Add keyboard shortcut for quitting (Ctrl+Q or Cmd+Q)
        self.root.bind("<Control-q>", lambda e: self.quit_application())
        self.root.bind("<Command-q>", lambda e: self.quit_application())

        # Handle window close button
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)

    def get_script_list(self) -> List[Path]:
        """Get all Python scripts in tests directory recursively.

        Returns:
            List of Path objects for executable Python scripts.
        """
        scripts = []
        if self.tests_dir.exists() and self.tests_dir.is_dir():
            for file_path in self.tests_dir.rglob("*.py"):
                if file_path.name not in ["__init__.py", "conftest.py"]:
                    scripts.append(file_path)
        return sorted(scripts)

    def _create_menu_bar(self) -> None:
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Quit", command=self.quit_application)

    def setup_ui(self) -> None:
        """Setup the user interface.

        Creates the main layout with script list, control buttons,
        status display, and output area.
        """
        # Create menu bar
        self._create_menu_bar()

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame, text="APGI Tests Scripts Runner", font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Scripts list frame
        list_frame = ttk.LabelFrame(
            main_frame, text="Available Test Scripts", padding="5"
        )
        list_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        # Scripts listbox with scrollbar
        list_scrollbar = ttk.Scrollbar(list_frame)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.scripts_listbox = tk.Listbox(
            list_frame, yscrollcommand=list_scrollbar.set, height=15, width=40
        )
        self.scripts_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.config(command=self.scripts_listbox.yview)

        # Populate scripts list with relative paths
        for script in self.scripts:
            relative_path = script.relative_to(self.tests_dir)
            self.scripts_listbox.insert(tk.END, str(relative_path))

        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=1, sticky="nsew", padx=(0, 10))

        # Buttons
        self.run_button = ttk.Button(
            control_frame, text="Run Selected", command=self.run_selected_script
        )
        self.run_button.pack(pady=5, fill=tk.X)
        ToolTip(self.run_button, "Run the currently selected test script")

        self.run_all_button = ttk.Button(
            control_frame, text="Run All Scripts", command=self.run_all_scripts
        )
        self.run_all_button.pack(pady=5, fill=tk.X)
        ToolTip(self.run_all_button, "Run all test scripts in sequence")

        self.run_all_tests_button = ttk.Button(
            control_frame,
            text="Run All Tests (pytest)",
            command=self.run_all_tests_pytest,
        )
        self.run_all_tests_button.pack(pady=5, fill=tk.X)
        ToolTip(self.run_all_tests_button, "Run complete test suite using pytest")

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Process",
            command=self.stop_selected_script,
            state=tk.DISABLED,
        )
        self.stop_button.pack(pady=5, fill=tk.X)
        ToolTip(self.stop_button, "Stop the currently running test script or pytest")

        self.clear_button = ttk.Button(
            control_frame, text="Clear Output", command=self.clear_output
        )
        self.clear_button.pack(pady=5, fill=tk.X)
        ToolTip(self.clear_button, "Clear the output text area")

        self.quit_button = ttk.Button(
            control_frame, text="Quit", command=self.quit_application
        )
        self.quit_button.pack(pady=5, fill=tk.X)
        ToolTip(self.quit_button, "Exit the application")

        # Status frame
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding="5")
        status_frame.pack(pady=20, fill=tk.X)

        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack()

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode="indeterminate", length=200)
        self.progress.pack(pady=10, fill=tk.X)

        # Output frame with tabs
        output_notebook = ttk.Notebook(main_frame)
        output_notebook.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(10, 0))

        # Output tab
        output_frame = ttk.Frame(output_notebook)
        output_notebook.add(output_frame, text="Output")

        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            output_frame, height=15, wrap=tk.WORD, font=("Courier", 9)
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Visualization tab
        viz_frame = ttk.Frame(output_notebook)
        output_notebook.add(viz_frame, text="Visualization")
        self.setup_visualization_tab(viz_frame)

        # Bind tab selection event to refresh visualization when tab is selected
        output_notebook.bind(
            "<<NotebookTabChanged>>", lambda e: self._on_visualization_tab_selected()
        )

    def setup_visualization_tab(self, parent_frame: ttk.Frame) -> None:
        """Setup the visualization tab with test result displays."""
        parent_frame.columnconfigure(0, weight=1)
        parent_frame.rowconfigure(1, weight=1)

        # Summary frame
        summary_frame = ttk.LabelFrame(parent_frame, text="Test Summary", padding="10")
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Summary labels
        self.summary_vars = {
            "total_tests": tk.StringVar(value="0"),
            "passed": tk.StringVar(value="0"),
            "failed": tk.StringVar(value="0"),
            "errors": tk.StringVar(value="0"),
            "skipped": tk.StringVar(value="0"),
            "duration": tk.StringVar(value="0.0s"),
            "coverage": tk.StringVar(value="0%"),
        }

        summary_items = [
            ("Total Tests:", self.summary_vars["total_tests"]),
            ("Passed:", self.summary_vars["passed"]),
            ("Failed:", self.summary_vars["failed"]),
            ("Errors:", self.summary_vars["errors"]),
            ("Skipped:", self.summary_vars["skipped"]),
            ("Duration:", self.summary_vars["duration"]),
            ("Coverage:", self.summary_vars["coverage"]),
        ]

        for i, (label, var) in enumerate(summary_items):
            ttk.Label(summary_frame, text=label).grid(
                row=i // 2, column=(i % 2) * 2, sticky=tk.W, padx=(0, 5)
            )
            ttk.Label(summary_frame, textvariable=var, font=("Arial", 10, "bold")).grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky=tk.W
            )

        # Visualization frame with charts
        viz_container = ttk.Frame(parent_frame)
        viz_container.grid(
            row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10)
        )

        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib figure for charts
            self.fig = Figure(figsize=(12, 4), dpi=80)
            self.fig.patch.set_facecolor("white")

            # Create subplots
            self.ax_pie = self.fig.add_subplot(131)
            self.ax_bar = self.fig.add_subplot(132)
            self.ax_timeline = self.fig.add_subplot(133)

            # Embed matplotlib in tkinter
            self.canvas = FigureCanvasTkAgg(self.fig, master=viz_container)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Initial draw to ensure canvas is visible
            self.root.update_idletasks()
        else:
            # Fallback text display if matplotlib not available
            no_viz_label = ttk.Label(
                viz_container,
                text="Advanced visualization requires matplotlib.\nInstall with: pip install matplotlib",
                font=("Arial", 12),
            )
            no_viz_label.pack(expand=True)

        # Results display frame
        results_frame = ttk.LabelFrame(parent_frame, text="Test Results", padding="10")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        # Results treeview
        columns = ("Test", "Status", "Duration", "Details")
        self.results_tree = ttk.Treeview(
            results_frame, columns=columns, show="headings", height=15
        )

        # Configure columns
        self.results_tree.heading("Test", text="Test Name")
        self.results_tree.heading("Status", text="Status")
        self.results_tree.heading("Duration", text="Duration")
        self.results_tree.heading("Details", text="Details")

        self.results_tree.column("Test", width=200)
        self.results_tree.column("Status", width=80)
        self.results_tree.column("Duration", width=80)
        self.results_tree.column("Details", width=300)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            results_frame, orient=tk.VERTICAL, command=self.results_tree.yview
        )
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        # Pack treeview and scrollbar
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Initialize test results storage
        self.test_results = []

    def _on_visualization_tab_selected(self) -> None:
        """Handle visualization tab selection event."""
        # Refresh visualization when tab is selected to ensure it displays properly
        if MATPLOTLIB_AVAILABLE and hasattr(self, "fig"):
            self.root.after_idle(self.update_visualization)

    def _setup_tab_order(self) -> None:
        """Configure tab order for consistent keyboard navigation."""
        # Define tab order: listbox -> buttons -> output tabs
        tab_order = [
            self.scripts_listbox,
            self.run_button,
            self.run_all_button,
            self.run_all_tests_button,
            self.stop_button,
            self.clear_button,
            self.quit_button,
            self.output_text,
            self.results_tree,
        ]

        # Set up tab navigation
        for i, widget in enumerate(tab_order):
            if widget:  # Check if widget exists
                next_widget = tab_order[(i + 1) % len(tab_order)]
                prev_widget = tab_order[(i - 1) % len(tab_order)]

                # Bind Tab to move to next widget
                widget.bind("<Tab>", lambda e, nw=next_widget: self._focus_widget(nw))
                # Bind Shift+Tab to move to previous widget
                widget.bind(
                    "<Shift-Tab>", lambda e, pw=prev_widget: self._focus_widget(pw)
                )

        # Ensure listbox is focused initially
        self.scripts_listbox.focus_set()

    def _focus_widget(self, widget: tk.Widget) -> None:
        """Focus on a specific widget and handle different widget types."""
        try:
            if isinstance(widget, tk.Listbox):
                widget.focus_set()
                # Select first item if nothing is selected
                if widget.curselection() == ():
                    widget.selection_set(0)
            elif isinstance(widget, ttk.Treeview):
                widget.focus_set()
                # Select first item if nothing is selected
                if widget.selection() == ():
                    first_item = widget.get_children()
                    if first_item:
                        widget.selection_set(first_item[0])
            else:
                widget.focus_set()
        except tk.TclError:
            # Widget might be disabled or not focusable
            pass

    def update_visualization(self) -> None:
        """Update the visualization with current test results."""
        try:
            # Clear existing results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            # Parse test results from output buffer
            self._parse_test_results()

            # Update summary statistics
            self._update_summary_stats()

            # Update charts if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                self._update_charts()

            # Populate results tree
            for result in self.test_results[-100:]:  # Show last 100 results
                status = result.get("status", "unknown")
                tags = ()
                if status == "PASSED":
                    tags = ("passed",)
                elif status in ["FAILED", "ERROR"]:
                    tags = ("failed",)

                self.results_tree.insert(
                    "",
                    tk.END,
                    values=(
                        result.get("test", "unknown"),
                        status,
                        result.get("duration", "0.0s"),
                        result.get("details", ""),
                    ),
                    tags=tags,
                )

        except Exception as e:
            self.log_output(f"Error updating visualization: {e}", self.TAG_ERROR)

    def _update_charts(self) -> None:
        """Update matplotlib charts with current test results."""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, "fig"):
            return

        try:
            # Clear previous plots
            self.ax_pie.clear()
            self.ax_bar.clear()
            self.ax_timeline.clear()

            if not self.test_results:
                # Show empty state with proper layout
                for ax in [self.ax_pie, self.ax_bar, self.ax_timeline]:
                    ax.text(
                        0.5,
                        0.5,
                        "No test data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=14,
                    )
                # Ensure the canvas is properly sized and visible
                self.fig.tight_layout()
                if hasattr(self, "canvas") and self.canvas is not None:
                    self.canvas.draw()
                self.root.update_idletasks()
                return

            # Count test results
            status_counts = {}
            for result in self.test_results:
                status = result.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

            # Pie chart of test results
            if status_counts:
                labels = list(status_counts.keys())
                sizes = list(status_counts.values())
                colors = {
                    "PASSED": "#2ecc71",  # green
                    "FAILED": "#e74c3c",  # red
                    "ERROR": "#e67e22",  # orange
                    "SKIPPED": "#95a5a6",  # gray
                }
                pie_colors = [colors.get(label, "#3498db") for label in labels]

                self.ax_pie.pie(
                    sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%"
                )
                self.ax_pie.set_title("Test Results Distribution")

            # Bar chart of test results
            if status_counts:
                labels = list(status_counts.keys())
                values = list(status_counts.values())
                bar_colors = [colors.get(label, "#3498db") for label in labels]

                self.ax_bar.bar(labels, values, color=bar_colors)
                self.ax_bar.set_title("Test Results Count")
                self.ax_bar.set_ylabel("Number of Tests")

            # Timeline of recent test results
            recent_results = self.test_results[-20:]  # Last 20 results
            if recent_results:
                status_values = []
                status_colors = []

                for result in recent_results:
                    status = result.get("status", "unknown")
                    if status == "PASSED":
                        status_values.append(1)
                        status_colors.append("#2ecc71")
                    elif status == "FAILED":
                        status_values.append(0)
                        status_colors.append("#e74c3c")
                    elif status == "ERROR":
                        status_values.append(-1)
                        status_colors.append("#e67e22")
                    else:
                        status_values.append(0.5)
                        status_colors.append("#95a5a6")

                self.ax_timeline.bar(
                    range(len(recent_results)), status_values, color=status_colors
                )
                self.ax_timeline.set_title("Recent Test Timeline")
                self.ax_timeline.set_xlabel("Test Sequence")
                self.ax_timeline.set_ylabel("Status")
                self.ax_timeline.set_ylim(-1.5, 1.5)
                # Ensure the canvas is properly sized and visible
                self.fig.tight_layout()
                if hasattr(self, "canvas") and self.canvas is not None:
                    self.canvas.draw()
                self.root.update_idletasks()
                return

        except Exception as e:
            self.log_output(f"Error updating charts: {e}", self.TAG_ERROR)

    def _parse_test_results(self) -> None:
        """Parse test results from the output buffer."""
        self.test_results = []

        # Parse pytest output for test results
        current_test = None
        for message, tag in self.output_buffer:
            line = message.strip()

            # Detect test start
            if line.startswith("tests/") and "::" in line:
                current_test = line.split()[0] if line else "unknown"
            elif "PASSED" in line and current_test:
                self.test_results.append(
                    {
                        "test": current_test,
                        "status": "PASSED",
                        "duration": self._extract_duration(line),
                        "details": "",
                    }
                )
                current_test = None
            elif "FAILED" in line and current_test:
                self.test_results.append(
                    {
                        "test": current_test,
                        "status": "FAILED",
                        "duration": self._extract_duration(line),
                        "details": "Test failed",
                    }
                )
                current_test = None
            elif "ERROR" in line and current_test:
                self.test_results.append(
                    {
                        "test": current_test,
                        "status": "ERROR",
                        "duration": self._extract_duration(line),
                        "details": "Test error",
                    }
                )
                current_test = None
            elif "SKIPPED" in line and current_test:
                self.test_results.append(
                    {
                        "test": current_test,
                        "status": "SKIPPED",
                        "duration": "0.0s",
                        "details": "Test skipped",
                    }
                )
                current_test = None

    def _extract_duration(self, line: str) -> str:
        """Extract duration from pytest output line."""
        try:
            # Look for duration in format like "0.12s"
            import re

            duration_match = re.search(r"(\d+\.\d+)s", line)
            if duration_match:
                return f"{duration_match.group(1)}s"
        except Exception:
            pass
        return "0.0s"

    def _update_summary_stats(self) -> None:
        """Update summary statistics from test results."""
        if not self.test_results:
            return

        total = len(self.test_results)
        passed = sum(1 for r in self.test_results if r["status"] == "PASSED")
        failed = sum(1 for r in self.test_results if r["status"] == "FAILED")
        errors = sum(1 for r in self.test_results if r["status"] == "ERROR")
        skipped = sum(1 for r in self.test_results if r["status"] == "SKIPPED")

        # Calculate total duration
        total_duration = 0.0
        for result in self.test_results:
            try:
                duration_str = result.get("duration", "0.0s").rstrip("s")
                total_duration += float(duration_str)
            except (ValueError, AttributeError):
                pass

        # Update summary variables
        self.summary_vars["total_tests"].set(str(total))
        self.summary_vars["passed"].set(str(passed))
        self.summary_vars["failed"].set(str(failed))
        self.summary_vars["errors"].set(str(errors))
        self.summary_vars["skipped"].set(str(skipped))
        self.summary_vars["duration"].set(f"{total_duration:.2f}s")

        # Try to extract coverage from recent output
        coverage = self._extract_coverage()
        self.summary_vars["coverage"].set(coverage)

    def _extract_coverage(self) -> str:
        """Extract coverage percentage from recent output."""
        # Look for coverage in recent output lines
        recent_lines = [msg for msg, tag in list(self.output_buffer)[-20:]]
        for line in reversed(recent_lines):
            if "TOTAL" in line and "%" in line:
                try:
                    # Extract percentage from line like "TOTAL                     85      0   100%"
                    parts = line.split()
                    for part in parts:
                        if "%" in part:
                            return part
                except Exception:
                    pass
        return "0%"

    def log_output(self, message: str, tag: Optional[str] = None) -> None:
        """Add message to output text area.

        Args:
            message: The message to log.
            tag: The color tag to apply (info, error, success, warning).
        """
        if tag is None:
            tag = self.TAG_INFO
        # Store in bounded buffer to prevent memory leaks
        self.output_buffer.append((message, tag))
        self.root.after_idle(lambda: self._safe_log_output(message, tag))

    def _safe_log_output(self, message: str, tag: str) -> None:
        """Thread-safe output logging with bounded buffer management."""
        # Clear and rebuild output if buffer is getting large
        if len(self.output_buffer) > 5000:
            self.output_text.delete(1.0, tk.END)
            # Re-add recent entries from buffer
            for msg, t in list(self.output_buffer)[-1000:]:
                self.output_text.insert(tk.END, msg + "\n", t)
        else:
            self.output_text.insert(tk.END, message + "\n", tag)
        self.output_text.see(tk.END)

    def update_status(self, message: str) -> None:
        """Update status label.

        Args:
            message: The status message to display.
        """
        self.root.after_idle(lambda: self._safe_update_status(message))

    def _safe_update_status(self, message: str) -> None:
        """Thread-safe status update."""
        self.status_label.config(text=message)

    def get_selected_script(self) -> Optional[Path]:
        """Get the currently selected script.

        Returns:
            Path to selected script or None if no selection.
        """
        selection = tuple(self.scripts_listbox.curselection())  # type: ignore
        if selection:
            index = selection[0]
            return self.scripts[index]
        return None

    def run_selected_script(self) -> None:
        """Run the selected script in a separate thread."""
        script = self.get_selected_script()
        if not script:
            self.log_output("No test script selected", self.TAG_ERROR)
            return

        self.run_script(script)

    def run_all_scripts(self) -> None:
        """Run all scripts sequentially."""
        if not self.scripts:
            self.log_output("No test scripts found", self.TAG_ERROR)
            return

        def run_all() -> None:
            self.run_all_running = True
            self.run_all_cancel_event.clear()

            for i, script in enumerate(self.scripts):
                if self.run_all_cancel_event.is_set():
                    self.log_output("Run All cancelled by user", self.TAG_WARNING)
                    break

                relative_path = script.relative_to(self.tests_dir)
                self.log_output(
                    f"Running test {i + 1}/{len(self.scripts)}: {relative_path}",
                    self.TAG_INFO,
                )
                self.root.after(
                    0, lambda: self.scripts_listbox.selection_clear(0, tk.END)
                )
                self.root.after(
                    0, lambda idx=i: self.scripts_listbox.selection_set(idx)
                )
                self.root.after(0, lambda idx=i: self.scripts_listbox.see(idx))

                success = self.run_script(script, wait=True)
                if not success:
                    self.log_output(
                        f"Test {relative_path} failed, stopping execution",
                        self.TAG_ERROR,
                    )
                    break

            self.log_output("All tests execution completed", self.TAG_SUCCESS)
            self.update_status("Ready")
            self.progress.stop()
            self.run_all_running = False

        # Run in separate thread to avoid blocking GUI
        thread = threading.Thread(target=run_all, daemon=False)
        self.running_threads.append(thread)
        thread.start()

    def run_all_tests_pytest(self) -> None:
        """Run all tests using pytest."""
        try:
            # Check if pytest is available
            try:
                subprocess.run(
                    [sys.executable, "-m", "pytest", "--version"],
                    capture_output=True,
                    check=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.log_output(
                    "pytest not found. Please install pytest manually.",
                    self.TAG_WARNING,
                )
                # Ask for user consent before installing
                try:
                    import tkinter.messagebox as messagebox

                    consent = messagebox.askyesno(
                        "Install pytest?",
                        "pytest is required to run the test suite. Would you like to install it now?",
                        parent=self.root,
                        icon=messagebox.QUESTION,
                    )
                    if consent:
                        try:
                            install_process = subprocess.Popen(
                                [sys.executable, "-m", "pip", "install", "pytest"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                text=True,
                                cwd=self.tests_dir.parent,
                            )
                            # Wait for installation to complete
                            stdout, stderr = install_process.communicate()
                            if install_process.returncode == 0:
                                self.log_output(
                                    "pytest installed successfully.", self.TAG_SUCCESS
                                )
                            else:
                                self.log_output(
                                    f"pytest installation failed with code {install_process.returncode}: {stderr}",
                                    self.TAG_ERROR,
                                )
                        except Exception as e:
                            self.log_output(
                                f"Error during pytest installation: {e}", self.TAG_ERROR
                            )
                            return False
                except Exception as e:
                    self.log_output(
                        f"Error showing consent dialog: {e}", self.TAG_WARNING
                    )
                    return False
                else:
                    return False
            except Exception as e:
                self.log_output(
                    f"Error checking pytest availability: {e}", self.TAG_ERROR
                )
                return False

            # Run pytest with coverage and detailed output
            env = os.environ.copy()
            parent_path = str(Path(__file__).parent)
            if not parent_path or len(parent_path) > 1024:
                self.log_output("Error: Invalid PYTHONPATH path", self.TAG_ERROR)
                return False
            env["PYTHONPATH"] = parent_path
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-v",  # verbose output
                    "--tb=short",  # shorter traceback format
                    "--color=yes",  # colored output
                    "tests/",  # run tests in tests directory
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=self.tests_dir.parent,
                env=env,
            )

            self.running_processes["pytest_all"] = process

            def read_pytest_output() -> None:
                if process.stdout is None:
                    return
                try:
                    while True:
                        output = process.stdout.readline()
                        if output == "" and process.poll() is not None:
                            break
                        if output:
                            # Color-code pytest output
                            line = output.strip()
                            if line.startswith("FAILED") or "ERROR" in line:
                                self.log_output(line, self.TAG_ERROR)
                            elif line.startswith("PASSED") or "passed" in line.lower():
                                self.log_output(line, self.TAG_SUCCESS)
                            elif "WARNING" in line or "warning" in line.lower():
                                self.log_output(line, self.TAG_WARNING)
                            else:
                                self.log_output(line)
                except Exception as e:
                    logging.warning(f"Error reading pytest output: {e}")

            # Start output reading thread before waiting for process
            output_thread = threading.Thread(target=read_pytest_output, daemon=False)
            self.running_threads.append(output_thread)
            output_thread.start()

            # Wait for process to complete (with timeout to avoid hanging)
            try:
                return_code = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                self.log_output(
                    "Test process did not complete within 5 minutes, forcing termination",
                    self.TAG_WARNING,
                )
                process.kill()
                return_code = -1

            if return_code == 0:
                self.log_output(" All tests passed successfully!", self.TAG_SUCCESS)
            else:
                self.log_output(
                    f"Test suite failed with return code {return_code}",
                    self.TAG_ERROR,
                )

            # Clean up
            if "pytest_all" in self.running_processes:
                del self.running_processes["pytest_all"]

            self.progress.stop()
            self.update_status("Ready")

            # Update visualization statistics and charts after tests complete
            self.root.after_idle(self.update_visualization)
        except Exception as e:
            self.log_output(f"Error running pytest: {e}", self.TAG_ERROR)
            if "pytest_all" in self.running_processes:
                try:
                    self.running_processes["pytest_all"].kill()
                except Exception:
                    pass
                del self.running_processes["pytest_all"]
            self.progress.stop()
            self.update_status("Ready")
            return False

    def run_script(self, script: Path, wait: bool = False) -> bool:
        """Run a single script.

        Args:
            script: Path to the script to run.
            wait: If True, block until script completes and return success status.
                 If False, run asynchronously and return True immediately.

        Returns:
            True if script started successfully, False if error occurred.
            When wait=True, returns True if script completed with return code 0.
        """
        try:
            # Validate script path is within tests directory to prevent directory traversal
            try:
                relative_path = script.resolve().relative_to(self.tests_dir.resolve())
            except ValueError:
                self.log_output(
                    f"Error: Script {script} is outside tests directory", self.TAG_ERROR
                )
                return False

            if not script.exists() or not script.is_file():
                self.log_output(f"Error: Script {script} not found", self.TAG_ERROR)
                return False

            self.log_output(f"Starting: {relative_path}", self.TAG_INFO)
            self.update_status(f"Running: {relative_path}")
            self.progress.start()

            # Enable stop button when script starts
            self.root.after_idle(lambda: self.stop_button.config(state=tk.NORMAL))

            # Run the script as a module to handle imports correctly
            module_path = str(
                script.relative_to(self.tests_dir.parent).with_suffix("")
            ).replace(os.sep, ".")

            # Create minimal environment for test isolation
            import secrets

            # Validate and set PYTHONPATH
            parent_path = str(Path(__file__).parent)
            if not parent_path or len(parent_path) > 1024:  # Reasonable length limit
                self.log_output("Error: Invalid PYTHONPATH path", self.TAG_ERROR)
                return False

            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": parent_path,
                "HOME": os.environ.get("HOME", ""),
                "USER": os.environ.get("USER", ""),
            }

            # Set required environment variables for secure operations
            if "PICKLE_SECRET_KEY" not in env:
                env["PICKLE_SECRET_KEY"] = secrets.token_hex(32)
            if "APGI_BACKUP_HMAC_KEY" not in env:
                env["APGI_BACKUP_HMAC_KEY"] = secrets.token_hex(32)

            process = subprocess.Popen(
                [sys.executable, "-m", module_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=self.tests_dir.parent,
                env=env,
            )

            self.running_processes[script.name] = process

            def read_output() -> None:
                try:
                    if process.stdout is None:
                        return
                    while True:
                        output = process.stdout.readline()
                        if output == "" and process.poll() is not None:
                            break
                        if output:
                            # Color-code pytest output
                            line = output.strip()
                            if line.startswith("FAILED") or "ERROR" in line:
                                self.log_output(line, self.TAG_ERROR)
                            elif line.startswith("PASSED") or "passed" in line.lower():
                                self.log_output(line, self.TAG_SUCCESS)
                            elif "WARNING" in line or "warning" in line.lower():
                                self.log_output(line, self.TAG_WARNING)
                            else:
                                self.log_output(line)

                    # Wait for process to complete (with timeout to avoid hanging)
                    try:
                        return_code = process.wait(timeout=300)
                    except subprocess.TimeoutExpired:
                        self.log_output(
                            "Test process did not complete within 5 minutes, forcing termination",
                            self.TAG_WARNING,
                        )
                        process.kill()
                        return_code = -1

                    if return_code == 0:
                        self.log_output(
                            "🎉 All tests passed successfully!", self.TAG_SUCCESS
                        )
                    else:
                        self.log_output(
                            f"❌ Test suite failed with return code {return_code}",
                            self.TAG_ERROR,
                        )

                    # Clean up
                    if script.name in self.running_processes:
                        del self.running_processes[script.name]

                    self.progress.stop()
                    self.update_status("Ready")

                    # Update visualization statistics and charts after tests complete
                    self.root.after_idle(self.update_visualization)
                except Exception as e:
                    self.log_output(
                        f"Error reading test output: {str(e)}", self.TAG_ERROR
                    )
                    self.progress.stop()
                    self.update_status("Error")

            if wait:
                # Wait for output thread to complete and get return code
                output_thread = threading.Thread(target=read_output, daemon=False)
                self.running_threads.append(output_thread)
                output_thread.start()
                output_thread.join(timeout=305)
                # Check process return code after thread completes
                return_code = process.poll()
                return return_code == 0 if return_code is not None else False
            else:
                # Run asynchronously
                output_thread = threading.Thread(target=read_output, daemon=False)
                self.running_threads.append(output_thread)
                output_thread.start()
                return True

        except Exception as e:
            self.log_output(f"Error running pytest: {str(e)}", self.TAG_ERROR)
            self.progress.stop()
            self.update_status("Error")
            return False

    def _update_stop_button_state(self) -> None:
        """Update stop button state based on running processes."""
        if self.running_processes:
            self.stop_button.config(state=tk.NORMAL)
        else:
            self.stop_button.config(state=tk.DISABLED)

    def stop_selected_script(self) -> None:
        """Stop the selected script if it's running, or cancel run_all operation."""
        # First, check if run_all is running and cancel it
        if self.run_all_running:
            self.run_all_cancel_event.set()
            self.run_all_running = False
            self.log_output("Cancelled Run All operation", self.TAG_WARNING)
            self._update_stop_button_state()
            self.progress.stop()
            self.update_status("Ready")
            return

        script = self.get_selected_script()
        if not script:
            # Check if pytest is running
            if "pytest_all" in self.running_processes:
                try:
                    process = self.running_processes["pytest_all"]
                    process.terminate()
                    self.log_output("Stopped pytest test suite", self.TAG_WARNING)
                    del self.running_processes["pytest_all"]
                    self._update_stop_button_state()
                    self.progress.stop()
                    self.update_status("Ready")
                except Exception as e:
                    self.log_output(f"Error stopping pytest: {str(e)}", self.TAG_ERROR)
            else:
                self.log_output("No test script selected", self.TAG_ERROR)
            return

        if script.name in self.running_processes:
            try:
                process = self.running_processes[script.name]
                process.terminate()
                relative_path = script.relative_to(self.tests_dir)
                self.log_output(f"Stopped: {relative_path}", self.TAG_WARNING)
                del self.running_processes[script.name]
                # Disable stop button if no processes are running
                self._update_stop_button_state()
                self.progress.stop()
                self.update_status("Ready")
            except Exception as e:
                relative_path = script.relative_to(self.tests_dir)
                self.log_output(
                    f"Error stopping {relative_path}: {str(e)}", self.TAG_ERROR
                )
        else:
            relative_path = script.relative_to(self.tests_dir)
            self.log_output(f"Test {relative_path} is not running", self.TAG_WARNING)

    def clear_output(self) -> None:
        """Clear the output text area."""
        self.output_text.delete(1.0, tk.END)
        self.log_output("Output cleared", self.TAG_INFO)

    def quit_application(self) -> None:
        """Quit the application safely.

        Stops all running processes and closes the application.
        """
        # Stop all running processes
        for script_name, process in list(self.running_processes.items()):
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    self.log_output(
                        f"Terminated process: {script_name}", self.TAG_WARNING
                    )
            except Exception as e:
                self.log_output(
                    f"Error terminating process {script_name}: {e}", self.TAG_ERROR
                )

        # Clear running processes
        self.running_processes.clear()

        # Cleanup environment variables
        for key in ["PICKLE_SECRET_KEY", "APGI_BACKUP_HMAC_KEY"]:
            if key in os.environ:
                del os.environ[key]

        # Cleanup matplotlib figures
        if MATPLOTLIB_AVAILABLE and hasattr(self, "fig"):
            try:
                import matplotlib.pyplot as plt

                plt.close(self.fig)
            except (RuntimeError, ValueError) as e:
                logging.warning(f"Error closing matplotlib figure: {e}")

        # Log quit message
        self.log_output("Quitting application...", self.TAG_INFO)

        # Wait a moment for messages to be processed
        self.root.update_idletasks()

        # Clean up running threads with timeout to avoid hanging
        for thread in self.running_threads:
            if thread.is_alive():
                thread.join(timeout=2)

        # Quit the application
        self.root.quit()
        self.root.destroy()


def main() -> None:
    """Launch the tests runner GUI."""
    try:
        # Set required environment variables before importing APGI modules
        import os
        import secrets

        if "PICKLE_SECRET_KEY" not in os.environ:
            os.environ["PICKLE_SECRET_KEY"] = secrets.token_hex(32)
        if "APGI_BACKUP_HMAC_KEY" not in os.environ:
            os.environ["APGI_BACKUP_HMAC_KEY"] = secrets.token_hex(32)

        # Create and run the GUI
        root = tk.Tk()
        app = TestsRunnerGUI(root)

        # Signal handling for graceful termination
        import signal

        def signal_handler(signum, frame):
            print(f"Received signal {signum}, shutting down...")
            root.after(0, app.quit_application)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")

        root.mainloop()

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("This script requires tkinter, which should come with Python.")
        sys.exit(1)
    except tk.TclError as e:
        print(f"❌ Tkinter Error: {e}")
        print("There was an error initializing the GUI.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
