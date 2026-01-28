#!/usr/bin/env python3
"""
Quick launcher for the Utils Runner GUI
======================================

Simple script to launch the utils runner GUI with error handling.
"""

import sys


def main():
    """Launch the utils runner GUI."""
    try:
        # Import the GUI module
        import tkinter as tk

        from utils_runner_gui import UtilsRunnerGUI

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
