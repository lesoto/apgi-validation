#!/usr/bin/env python3
"""
Simple GUI to run all utils folder scripts
==========================================

A tkinter-based GUI that allows running all scripts in the utils folder
with output display and error handling.
"""

import importlib.util
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk
from typing import Dict, List


class UtilsRunnerGUI:
    """Simple GUI for running utils scripts."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("APGI Utils Scripts Runner")
        self.root.geometry("800x600")

        # Get utils directory
        self.utils_dir = Path(__file__).parent / "utils"
        self.scripts = self.get_script_list()

        # Store running processes
        self.running_processes: Dict[str, subprocess.Popen] = {}

        self.setup_ui()

        # Add keyboard shortcut for quitting (Ctrl+Q or Cmd+Q)
        self.root.bind("<Control-q>", lambda e: self.quit_application())
        self.root.bind("<Command-q>", lambda e: self.quit_application())

    def get_script_list(self) -> List[Path]:
        """Get all Python scripts in utils directory."""
        scripts = []
        if self.utils_dir.exists():
            for file_path in self.utils_dir.glob("*.py"):
                if file_path.name != "__init__.py":
                    scripts.append(file_path)
        return sorted(scripts)

    def setup_ui(self):
        """Setup the user interface."""
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
            main_frame, text="APGI Utils Scripts Runner", font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Scripts list frame
        list_frame = ttk.LabelFrame(main_frame, text="Available Scripts", padding="5")
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Scripts listbox with scrollbar
        list_scrollbar = ttk.Scrollbar(list_frame)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.scripts_listbox = tk.Listbox(
            list_frame, yscrollcommand=list_scrollbar.set, height=15, width=40
        )
        self.scripts_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.config(command=self.scripts_listbox.yview)

        # Populate scripts list
        for script in self.scripts:
            self.scripts_listbox.insert(tk.END, script.name)

        # Bind selection change to update stop button state
        self.scripts_listbox.bind(
            "<<ListboxSelect>>", lambda e: self.update_stop_button_state()
        )

        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))

        # Buttons
        self.run_button = ttk.Button(
            control_frame, text="Run Selected", command=self.run_selected_script
        )
        self.run_button.pack(pady=5, fill=tk.X)

        self.run_all_button = ttk.Button(
            control_frame, text="Run All Scripts", command=self.run_all_scripts
        )
        self.run_all_button.pack(pady=5, fill=tk.X)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Selected",
            command=self.stop_selected_script,
            state=tk.DISABLED,
        )
        self.stop_button.pack(pady=5, fill=tk.X)

        self.clear_button = ttk.Button(
            control_frame, text="Clear Output", command=self.clear_output
        )
        self.clear_button.pack(pady=5, fill=tk.X)

        self.quit_button = ttk.Button(
            control_frame, text="Quit", command=self.quit_application
        )
        self.quit_button.pack(pady=5, fill=tk.X)

        # Status frame
        status_frame = ttk.LabelFrame(control_frame, text="Status", padding="5")
        status_frame.pack(pady=20, fill=tk.X)

        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack()

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode="indeterminate", length=200)
        self.progress.pack(pady=10, fill=tk.X)

        # Output frame
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0)
        )

        # Output text area
        self.output_text = scrolledtext.ScrolledText(
            output_frame, height=15, wrap=tk.WORD, font=("Courier", 9)
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for different output types
        self.output_text.tag_config("info", foreground="black")
        self.output_text.tag_config("error", foreground="red")
        self.output_text.tag_config("success", foreground="green")
        self.output_text.tag_config("warning", foreground="orange")

    def log_output(self, message: str, tag: str = "info"):
        """Add message to output text area."""
        self.output_text.insert(tk.END, message + "\n", tag)
        self.output_text.see(tk.END)
        self.root.update_idletasks()

    def update_status(self, message: str):
        """Update status label."""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def get_selected_script(self) -> Path:
        """Get the currently selected script."""
        selection = self.scripts_listbox.curselection()
        if selection:
            index = selection[0]
            return self.scripts[index]
        return None

    def update_stop_button_state(self):
        """Update stop button state based on selected script running status."""
        script = self.get_selected_script()
        if script and script.name in self.running_processes:
            self.stop_button.config(state=tk.NORMAL)
        else:
            self.stop_button.config(state=tk.DISABLED)

    def run_selected_script(self):
        """Run the selected script in a separate thread."""
        script = self.get_selected_script()
        if not script:
            self.log_output("No script selected", "error")
            return

        self.run_script(script)

    def run_all_scripts(self):
        """Run all scripts sequentially."""
        if not self.scripts:
            self.log_output("No scripts found", "error")
            return

        def run_all():
            for i, script in enumerate(self.scripts):
                self.log_output(
                    f"Running script {i + 1}/{len(self.scripts)}: {script.name}", "info"
                )
                self.scripts_listbox.selection_clear(0, tk.END)
                self.scripts_listbox.selection_set(i)
                self.scripts_listbox.see(i)

                success = self.run_script(script, wait=True)
                if not success:
                    self.log_output(
                        f"Script {script.name} failed, stopping execution", "error"
                    )
                    break

            self.log_output("All scripts execution completed", "success")
            self.update_status("Ready")
            self.progress.stop()

        # Run in separate thread to avoid blocking GUI
        thread = threading.Thread(target=run_all, daemon=True)
        thread.start()

    def run_script(self, script: Path, wait: bool = False) -> bool:
        """Run a single script."""
        try:
            self.log_output(f"Starting: {script.name}", "info")
            self.update_status(f"Running: {script.name}")
            self.progress.start()

            # Run the script
            process = subprocess.Popen(
                [sys.executable, str(script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=self.utils_dir.parent,
            )

            self.running_processes[script.name] = process
            self.update_stop_button_state()

            # Read output in real-time
            def read_output():
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        self.log_output(output.strip())

                # Get return code
                return_code = process.poll()
                if return_code == 0:
                    self.log_output(
                        f"✅ {script.name} completed successfully", "success"
                    )
                else:
                    self.log_output(
                        f"❌ {script.name} failed with return code {return_code}",
                        "error",
                    )

                # Clean up
                if script.name in self.running_processes:
                    del self.running_processes[script.name]

                self.progress.stop()
                self.update_status("Ready")
                self.update_stop_button_state()

            # Start output reading thread
            output_thread = threading.Thread(target=read_output, daemon=True)
            output_thread.start()

            if wait:
                output_thread.join()
                return process.returncode == 0
            else:
                return True

        except Exception as e:
            self.log_output(f"Error running {script.name}: {str(e)}", "error")
            self.progress.stop()
            self.update_status("Error")
            return False

    def stop_selected_script(self):
        """Stop the selected script if it's running."""
        script = self.get_selected_script()
        if not script:
            self.log_output("No script selected", "error")
            return

        if script.name in self.running_processes:
            try:
                process = self.running_processes[script.name]
                process.terminate()
                self.log_output(f"Stopped: {script.name}", "warning")
                del self.running_processes[script.name]
                self.progress.stop()
                self.update_status("Ready")
                self.update_stop_button_state()
            except Exception as e:
                self.log_output(f"Error stopping {script.name}: {str(e)}", "error")
        else:
            self.log_output(f"Script {script.name} is not running", "warning")

    def clear_output(self):
        """Clear the output text area."""
        self.output_text.delete(1.0, tk.END)
        self.log_output("Output cleared", "info")

    def quit_application(self):
        """Quit the application safely."""
        # Stop all running processes
        for script_name, process in self.running_processes.items():
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    self.log_output(f"Terminated process: {script_name}", "warning")
            except Exception as e:
                self.log_output(
                    f"Error terminating process {script_name}: {e}", "error"
                )

        # Clear running processes
        self.running_processes.clear()

        # Log quit message
        self.log_output("Quitting application...", "info")

        # Wait a moment for messages to be processed
        self.root.update_idletasks()

        # Quit the application
        self.root.quit()
        self.root.destroy()


def _get_protocol1():
    """Safely import Protocol 1 with error handling"""
    import os

    try:
        protocol1_path = os.path.join(
            os.path.dirname(__file__), "Falsification-Protocol-1.py"
        )
        if not os.path.exists(protocol1_path):
            raise ImportError(f"Protocol 1 file not found: {protocol1_path}")

        spec1 = importlib.util.spec_from_file_location("Protocol_1", protocol1_path)
        if spec1 is None or spec1.loader is None:
            raise ImportError("Failed to load spec for Protocol 1")

        protocol1 = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(protocol1)
        return protocol1
    except Exception as e:
        raise ImportError(f"Failed to import Protocol 1: {str(e)}")


def main() -> None:
    """Launch the utils runner GUI."""
    try:
        # Import tkinter
        import tkinter as tk

        # Create and run the GUI
        root = tk.Tk()
        app = UtilsRunnerGUI(root)

        # Center window on screen - wait for window to be properly rendered
        root.update_idletasks()
        root.withdraw()  # Hide window during positioning
        root.update_idletasks()

        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")

        root.deiconify()  # Show window after positioning
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
