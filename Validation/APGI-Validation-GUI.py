"""
APGI Validation GUI
==================

Simple tkinter GUI for running APGI validation protocols
with real-time progress tracking and results visualization.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import json
import threading
import sys
from pathlib import Path
from datetime import datetime

# Add Validation directory to path
sys.path.append(str(Path(__file__).parent / "Validation"))

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("APGI_Master_Validation", 
                                                 Path(__file__).parent / "APGI-Master-Validation.py")
    APGI_Master_Validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(APGI_Master_Validation)
    APGIMasterValidator = APGI_Master_Validation.APGIMasterValidator
except ImportError as e:
    # Fallback if import fails
    print(f"Warning: Could not import APGI_Master_Validation: {e}")
    print("Troubleshooting:")
    print("1. Ensure APGI-Master-Validation.py exists in the Validation directory")
    print("2. Check that the file has proper syntax and no import errors")
    print("3. Verify all dependencies are installed (pip install -r requirements.txt)")
    APGIMasterValidator = None

class APGIValidationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("APGI Validation Protocol Runner")
        self.root.geometry("800x600")
        
        # Initialize validator
        self.validator = APGIMasterValidator() if APGIMasterValidator else None
        
        # Create GUI elements
        self.create_widgets()
        
        # Validation thread
        self.validation_thread = None
        self.is_running = False
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="APGI Validation Protocol Runner", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Protocol selection frame
        protocol_frame = ttk.LabelFrame(main_frame, text="Protocol Selection", padding="10")
        protocol_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
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
            8: "Protocol 8: Secondary Test"
        }
        
        for i, (num, desc) in enumerate(protocols_info.items()):
            var = tk.BooleanVar(value=True)
            self.protocol_vars[num] = var
            
            cb = ttk.Checkbutton(protocol_frame, text=desc, variable=var)
            cb.grid(row=i//2, column=(i%2)*2, sticky=tk.W, padx=5, pady=2)
        
        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        
        self.run_button = ttk.Button(control_frame, text="Run Validation", 
                                    command=self.run_validation)
        self.run_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop", 
                                     command=self.stop_validation, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.save_button = ttk.Button(control_frame, text="Save Results", 
                                     command=self.save_results)
        self.save_button.grid(row=0, column=2, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to run validation", 
                                     font=('Arial', 10))
        self.status_label.grid(row=4, column=0, columnspan=2, pady=(0, 10))
        
        # Results text area
        results_frame = ttk.LabelFrame(main_frame, text="Validation Results", padding="10")
        results_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Summary frame
        summary_frame = ttk.LabelFrame(main_frame, text="Validation Summary", padding="10")
        summary_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.summary_label = ttk.Label(summary_frame, text="No validation run yet", 
                                     font=('Arial', 10))
        self.summary_label.grid(row=0, column=0)
        
    def run_validation(self):
        """Run the selected validation protocols"""
        if self.is_running:
            return
            
        if not self.validator:
            messagebox.showerror("Error", "APGI Master Validator not available")
            return
            
        # Get selected protocols
        selected_protocols = [num for num, var in self.protocol_vars.items() if var.get()]
        
        if not selected_protocols:
            messagebox.showwarning("Warning", "No protocols selected")
            return
            
        # Start validation in separate thread
        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.progress_var.set(0)
        
        self.validation_thread = threading.Thread(target=self._run_validation_worker, 
                                                 args=(selected_protocols,))
        self.validation_thread.start()
        
    def _run_validation_worker(self, selected_protocols):
        """Worker thread for running validation"""
        try:
            self.update_status("Starting validation...")
            self.update_progress(0)
            
            # Clear previous results
            self.validator.protocol_results = {}
            self.validator.falsification_status = {
                'primary': [],
                'secondary': [],
                'tertiary': []
            }
            
            # Protocol tier classification
            protocol_tiers = {
                1: 'primary', 2: 'secondary', 3: 'primary', 4: 'secondary',
                5: 'tertiary', 6: 'tertiary', 7: 'tertiary', 8: 'secondary'
            }
            
            total_protocols = len(selected_protocols)
            
            for i, protocol_num in enumerate(selected_protocols):
                if not self.is_running:
                    break
                    
                self.update_status(f"Running Protocol {protocol_num}...")
                self.update_results(f"=== Protocol {protocol_num} ===\n")
                
                try:
                    # Execute actual protocol
                    protocol_file = f"APGI-Protocol-{protocol_num}.py"
                    protocol_path = Path(__file__).parent / protocol_file
                    
                    if not protocol_path.exists():
                        raise FileNotFoundError(f"Protocol file {protocol_file} not found")
                    
                    # Import protocol module dynamically
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(f"APGI_Protocol_{protocol_num}", protocol_path)
                    protocol_module = importlib.util.module_from_spec(spec)
                    
                    # Capture stdout to get results
                    import io
                    import contextlib
                    
                    captured_output = io.StringIO()
                    with contextlib.redirect_stdout(captured_output):
                        spec.loader.exec_module(protocol_module)
                        
                        # Run the protocol's main function if it exists
                        if hasattr(protocol_module, 'main'):
                            protocol_result = protocol_module.main()
                        else:
                            # If no main function, consider it completed
                            protocol_result = {'status': 'COMPLETED', 'message': 'Protocol executed successfully'}
                    
                    # Parse captured output for results
                    output_text = captured_output.getvalue()
                    
                    # Create result based on protocol execution
                    result = {
                        'status': 'COMPLETED',
                        'passed': True,  # Default to passed unless error occurs
                        'timestamp': datetime.now().isoformat(),
                        'output': output_text,
                        'protocol_result': protocol_result if protocol_result else {}
                    }
                    
                    # Check for error indicators in output
                    error_indicators = ['ERROR', 'FAILED', 'Exception', 'Traceback', 'Error:']
                    if any(indicator in output_text for indicator in error_indicators):
                        result['passed'] = False
                        result['status'] = 'COMPLETED_WITH_ERRORS'
                    
                    self.validator.protocol_results[f'protocol_{protocol_num}'] = result
                    
                    passed = result.get('passed', True)
                    tier = protocol_tiers[protocol_num]
                    
                    self.validator.falsification_status[tier].append({
                        'protocol': protocol_num,
                        'passed': passed,
                        'result': result
                    })
                    
                    self.update_results(f"Status: {'PASSED' if passed else 'FAILED'}\n\n")
                    
                except Exception as e:
                    error_result = {
                        'status': 'EXECUTION_ERROR',
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'timestamp': datetime.now().isoformat(),
                        'troubleshooting': self._get_error_troubleshooting(e, protocol_num)
                    }
                    
                    self.validator.protocol_results[f'protocol_{protocol_num}'] = error_result
                    tier = protocol_tiers[protocol_num]
                    self.validator.falsification_status[tier].append({
                        'protocol': protocol_num,
                        'passed': False,
                        'result': error_result
                    })
                    
                    error_msg = f"Protocol {protocol_num} failed: {type(e).__name__}: {e}"
                    self.update_results(f"ERROR: {error_msg}\n")
                    self.update_results(f"Troubleshooting: {error_result['troubleshooting']}\n\n")
                    
                    self.update_results(f"ERROR: {e}\n\n")
                
                # Update progress
                progress = ((i + 1) / total_protocols) * 100
                self.update_progress(progress)
            
            # Generate final report
            if self.is_running:
                self.update_status("Generating final report...")
                report = self.validator.generate_master_report()
                
                # Update summary
                self.update_summary(report)
                self.update_results(f"\n=== FINAL RESULT ===\n")
                self.update_results(f"Overall Decision: {report['overall_decision']}\n")
                
                self.update_status("Validation completed")
                
        except Exception as e:
            error_msg = f"Validation failed: {type(e).__name__}: {e}"
            self.update_status(error_msg)
            self.update_results(f"CRITICAL ERROR: {error_msg}\n")
            self.update_results(f"\nTroubleshooting steps:\n")
            self.update_results(f"1. Check that all protocol files exist and are readable\n")
            self.update_results(f"2. Verify all dependencies are installed\n")
            self.update_results(f"3. Check file permissions in the Validation directory\n")
            self.update_results(f"4. Try running individual protocols separately\n")
            
        finally:
            self.is_running = False
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
    def stop_validation(self):
        """Stop the running validation"""
        self.is_running = False
        self.update_status("Stopping validation...")
        
    def save_results(self):
        """Save validation results to file"""
        if not self.validator or not self.validator.protocol_results:
            messagebox.showwarning("Warning", "No results to save")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"APGI-Validation-Results-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if filename:
            try:
                report = self.validator.generate_master_report()
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                messagebox.showinfo("Success", f"Results saved to {filename}")
            except Exception as e:
                error_msg = f"Failed to save results: {type(e).__name__}: {e}"
                messagebox.showerror("Save Error", f"{error_msg}\n\nTroubleshooting:\n"
                                   f"1. Check file permissions for the target directory\n"
                                   f"2. Ensure disk space is available\n"
                                   f"3. Try saving to a different location\n"
                                   f"4. Check that the filename is valid")
                
    def update_status(self, message):
        """Update status label"""
        self.root.after(0, lambda: self.status_label.config(text=message))
        
    def update_progress(self, value):
        """Update progress bar"""
        self.root.after(0, lambda: self.progress_var.set(value))
        
    def update_results(self, message):
        """Update results text widget"""
        self.root.after(0, lambda: self.results_text.insert(tk.END, message))
        self.root.after(0, lambda: self.results_text.see(tk.END))
        
    def _get_error_troubleshooting(self, error: Exception, protocol_num: int) -> str:
        """Get troubleshooting hints based on error type and protocol"""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Common issues
        if "module not found" in error_str or "modulenotfounderror" in error_str:
            return ("Missing dependency detected. Install with: pip install -r requirements.txt\n"
                   f"Specifically check: {error_str.split('no module named')[-1] if 'no module named' in error_str else 'unknown module'}")
        
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
        return (f"Error type: {error_type}. "
                "Check the protocol file for syntax issues and ensure all dependencies are installed. "
                "Try running the protocol individually to isolate the problem.")
        
    def update_summary(self, report):
        """Update summary label"""
        summary_text = "Summary:\n"
        for tier, results in report['falsification_status'].items():
            failures = sum(1 for r in results if not r['passed'])
            total = len(results)
            summary_text += f"{tier.capitalize()} tier: {failures}/{total} failed\n"
            
        self.root.after(0, lambda: self.summary_label.config(text=summary_text))

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = APGIValidationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
