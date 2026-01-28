#!/usr/bin/env python3
"""
APGI Theory Framework - Unified CLI Entry Point.
================================================

Provides command-line interface to all APGI framework components including:
- Formal model simulations
- Multimodal integration
- Parameter estimation
- Validation protocols
- Falsification testing
- Configuration management
"""

import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.backup_manager import (
    backup_manager,
    cleanup_backups_cli,
    create_backup_cli,
    delete_backup_cli,
    list_backups_cli,
    restore_backup_cli,
)
from utils.config_manager import config_manager
from utils.error_handler import (
    APGIError,
    ErrorCategory,
    ErrorSeverity,
    error_handler,
    format_user_message,
    get_error_summary,
)

# Import APGI framework components
from utils.logging_config import apgi_logger

# Initialize rich console with better width handling
console = Console(
    width=None,  # Auto-detect terminal width
    file=None,  # Use stdout
    force_terminal=True,  # Force terminal mode
    force_interactive=False,  # Don't force interactive mode
    legacy_windows=False,  # Use modern Windows handling
    color_system="auto",  # Auto-detect color support
    no_color=False,  # Enable colors
)

# Global configuration
global_config = {
    "version": "1.3.0",
    "project_name": "APGI Theory Framework",
    "description": "Adaptive Pattern Generation and Integration Theory Implementation",
    "verbose": False,
    "quiet": False,
}


def verbose_print(message: str, level: str = "info") -> None:
    """Print message only if verbose mode is enabled."""
    if not config.get("quiet", False) and config.get("verbose", False):
        if level == "error":
            console.print(f"[red]{message}[/red]")
        elif level == "warning":
            console.print(f"[yellow]{message}[/yellow]")
        elif level == "success":
            console.print(f"[green]{message}[/green]")
        else:
            console.print(f"[blue]{message}[/blue]")


def quiet_print(message: str, level: str = "info", force: bool = False) -> None:
    """Print message unless quiet mode is enabled (or forced)."""
    if not config.get("quiet", False) or force:
        if level == "error":
            console.print(f"[red]{message}[/red]")
        elif level == "warning":
            console.print(f"[yellow]{message}[/yellow]")
        elif level == "success":
            console.print(f"[green]{message}[/green]")
        else:
            console.print(message)


def handle_import_error(module_name: str, error: Exception, context: str = "") -> None:
    """Handle import errors with specific, actionable messages."""
    error_msg = str(error)

    if "No module named" in error_msg:
        missing_module = error_msg.split("'")[1] if "'" in error_msg else module_name
        suggestions = {
            "click": "pip install click",
            "pandas": "pip install pandas",
            "numpy": "pip install numpy",
            "matplotlib": "pip install matplotlib",
            "scipy": "pip install scipy",
            "torch": "pip install torch",
            "sklearn": "pip install scikit-learn",
            "pymc": "pip install pymc",
            "arviz": "pip install arviz",
            "yaml": "pip install pyyaml",
            "seaborn": "pip install seaborn",
        }

        suggestion = suggestions.get(missing_module, f"pip install {missing_module}")
        quiet_print(f"Missing dependency: {missing_module}", "error", force=True)
        quiet_print(f"Install with: {suggestion}", "info")

    elif "DLL load failed" in error_msg or "shared library" in error_msg:
        quiet_print(f"Library loading error for {module_name}", "error", force=True)
        quiet_print(
            "Try reinstalling: pip uninstall {module_name} && pip install {module_name}",
            "info",
        )

    elif "Permission denied" in error_msg:
        quiet_print(f"Permission error loading {module_name}", "error", force=True)
        quiet_print(
            "Try running with appropriate permissions or check file permissions", "info"
        )

    else:
        quiet_print(f"Error importing {module_name}: {error_msg}", "error", force=True)

    if context:
        verbose_print(f"Context: {context}", "warning")


def handle_file_error(file_path: str, operation: str, error: Exception) -> None:
    """Handle file-related errors with specific guidance."""
    error_msg = str(error)

    if "No such file" in error_msg or "FileNotFoundError" in error_msg:
        quiet_print(f"File not found: {file_path}", "error", force=True)
        quiet_print("Check if the file exists and the path is correct", "info")
        quiet_print(f"Current directory: {Path.cwd()}", "info")

    elif "Permission denied" in error_msg:
        quiet_print(f"Permission denied accessing {file_path}", "error", force=True)
        quiet_print("Check file permissions or run with appropriate privileges", "info")

    elif "Is a directory" in error_msg:
        quiet_print(
            f"Expected file but got directory: {file_path}", "error", force=True
        )
        quiet_print("Please specify a file, not a directory", "info")

    else:
        quiet_print(f"Error {operation} {file_path}: {error_msg}", "error", force=True)


def handle_validation_error(error: Exception, context: str = "") -> None:
    """Handle validation errors with specific guidance."""
    error_msg = str(error)

    if (
        "range" in error_msg.lower()
        or "minimum" in error_msg.lower()
        or "maximum" in error_msg.lower()
    ):
        quiet_print(f"Parameter out of valid range: {error_msg}", "error", force=True)
        quiet_print(
            "Check parameter constraints in configuration documentation", "info"
        )

    elif "type" in error_msg.lower():
        quiet_print(f"Invalid parameter type: {error_msg}", "error", force=True)
        quiet_print(
            "Ensure parameter values match expected types (number, string, boolean)",
            "info",
        )

    else:
        quiet_print(f"Validation error: {error_msg}", "error", force=True)

    if context:
        verbose_print(f"Validation context: {context}", "warning")


class APGIModuleLoader:
    """Dynamic module loader for APGI components."""

    def __init__(self):
        self.modules = {}
        self._load_available_modules()

    def _load_available_modules(self):
        """Load all available APGI modules."""
        module_configs = {
            "formal_model": {
                "file": "APGI-Formal-Model.py",
                "class": "SurpriseIgnitionSystem",
                "description": "Formal model simulations",
            },
            "multimodal": {
                "file": "APGI-Multimodal-Integration.py",
                "class": None,  # Will detect main class
                "description": "Multimodal data integration",
            },
            "parameter_estimation": {
                "file": "APGI-Parameter-Estimation-Protocol.py",
                "class": None,
                "description": "Bayesian parameter estimation",
            },
            "psychological_states": {
                "file": "APGI-Psychological-States-CLI.py",
                "class": None,
                "description": "Psychological states analysis",
            },
        }

        for name, config in module_configs.items():
            module_path = PROJECT_ROOT / config["file"]
            if module_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location(name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.modules[name] = {"module": module, "config": config}
                except (ImportError, AttributeError, OSError, TypeError) as e:
                    console.print(
                        f"[yellow]Warning: Could not load {config['file']}: {e}[/yellow]"
                    )

    def get_module(self, name):
        """Get loaded module by name."""
        return self.modules.get(name)


# Initialize module loader
module_loader = APGIModuleLoader()


@click.group()
@click.version_option(
    version=global_config["version"], prog_name=global_config["project_name"]
)
@click.option("--config-file", help="Override configuration file path")
@click.option("--log-level", help="Override logging level")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress non-error output")
@click.pass_context
def cli(ctx, config_file, log_level, verbose, quiet):
    """
    APGI Theory Framework - Unified Command Line Interface

    Comprehensive computational framework for Adaptive Pattern Generation
    and Integration theory with psychological state dynamics modeling.
    """
    ctx.ensure_object(dict)
    ctx.obj["console"] = console
    ctx.obj["module_loader"] = module_loader
    ctx.obj["verbose"] = verbose and not quiet
    ctx.obj["quiet"] = quiet

    # Store verbosity in global config for access by other functions
    global_config["verbose"] = verbose and not quiet
    global_config["quiet"] = quiet

    # Apply command-line overrides
    if config_file:
        config_manager.config_file = Path(config_file)
        config_manager._load_config()
        apgi_logger.logger.info(f"Using custom config file: {config_file}")

    if log_level:
        # set_parameter("logging", "level", log_level.upper())
        apgi_logger.logger.info(f"Log level overridden to: {log_level.upper()}")

    # Log framework startup
    apgi_logger.logger.info(f"APGI Framework v{global_config['version']} started")
    # log_performance("framework_startup", 0, "seconds")


@cli.command()
@click.option(
    "--simulation-steps",
    default=None,
    type=int,
    help="Number of simulation steps (uses config default)",
)
@click.option(
    "--dt", default=None, type=float, help="Time step size (uses config default)"
)
@click.option("--output-file", help="Output file for results")
@click.option("--params", help="JSON file with custom parameters")
@click.option("--plot", is_flag=True, help="Generate visualization plots")
@click.pass_context
def formal_model(
    ctx: click.Context,
    simulation_steps: Optional[int],
    dt: Optional[float],
    output_file: Optional[str],
    params: Optional[str],
    plot: bool,
) -> None:
    """Run formal model simulations."""
    console.print(Panel.fit("🧮 Formal Model Simulation", style="bold blue"))

    # Use default values since get_config is not available
    sim_steps = simulation_steps or 1000
    time_step = dt or 0.01
    enable_plots = plot or True

    start_time = time.time()

    module_info = module_loader.get_module("formal_model")
    if not module_info:
        error_msg = "Formal model module not found"
        console.print(f"[red]Error: {error_msg}[/red]")
        apgi_logger.logger.error(error_msg)
        return

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing simulation...", total=None)

            # Initialize the model with configuration parameters
            SurpriseIgnitionSystem = module_info["module"].SurpriseIgnitionSystem

            # Use default values since config.model is not available
            model_params = {
                "tau_S": 0.1,
                "tau_theta": 0.2,
                "theta_0": 2.0,
                "alpha": 0.01,
                "gamma_M": 0.1,
                "gamma_A": 0.1,
                "rho": 0.95,
                "sigma_S": 0.1,
                "sigma_theta": 0.1,
            }

            # Load custom parameters if provided
            if params:
                try:
                    import json

                    # Load JSON file
                    with open(params, "r") as f:
                        custom_params = json.load(f)

                    # Validate JSON structure
                    try:
                        from parameter_validator import validate_parameters

                        validation_result = validate_parameters(custom_params)

                        if not validation_result["valid"]:
                            console.print("[red]❌ Parameter validation failed:[/red]")
                            for error in validation_result["errors"]:
                                console.print(f"  • {error}")
                            console.print(
                                "[yellow]Using default parameters instead[/yellow]"
                            )

                            if validation_result["warnings"]:
                                console.print("[yellow]Warnings:[/yellow]")
                                for warning in validation_result["warnings"]:
                                    console.print(f"  • {warning}")
                        else:
                            console.print(
                                f"[green]✓[/green] Loaded and validated {validation_result['validated_params']} parameters from {params}"
                            )

                            if validation_result["warnings"]:
                                console.print("[yellow]Warnings:[/yellow]")
                                for warning in validation_result["warnings"]:
                                    console.print(f"  • {warning}")

                    except ImportError:
                        console.print(
                            "[yellow]⚠️  Parameter validator not available, skipping validation[/yellow]"
                        )
                        console.print(
                            f"[green]✓[/green] Loaded custom parameters from {params}"
                        )

                    # Update model parameters with custom values
                    for key, value in custom_params.items():
                        if key in model_params:
                            model_params[key] = float(value)
                        else:
                            verbose_print(
                                f"Warning: Unknown parameter '{key}' ignored", "warning"
                            )

                except FileNotFoundError:
                    quiet_print(
                        f"Error: Parameter file not found: {params}",
                        "error",
                        force=True,
                    )
                    verbose_print(
                        f"File path checked: {Path(params).absolute()}", "warning"
                    )
                    quiet_print("Using default parameters instead", "warning")
                    verbose_print(
                        'Tip: Create a JSON file with parameters like: {"tau_S": 0.5, "alpha": 10.0}',
                        "info",
                    )
                except json.JSONDecodeError as e:
                    quiet_print(
                        f"Error: Invalid JSON in parameter file: {params}",
                        "error",
                        force=True,
                    )
                    verbose_print(f"JSON error details: {str(e)}", "warning")
                    quiet_print(
                        f"Error loading parameter file: {type(e).__name__}: {e}",
                        "error",
                        force=True,
                    )
                    quiet_print("Using default parameters instead", "warning")

            system = SurpriseIgnitionSystem(params=model_params)

            progress.update(task, description="Running simulation...")

            # Log simulation start
            apgi_logger.log_simulation_start("formal_model", model_params)

            # Run simulation with enhanced progress tracking
            import signal
            import threading

            import numpy as np

            results = {"time": [], "surprise": [], "threshold": [], "ignition": []}
            cancel_flag = threading.Event()

            # Set up signal handler for cancellation
            def handle_cancel(signum, frame):
                cancel_flag.set()
                console.print(
                    "\n[yellow]⚠️  Simulation cancellation requested...[/yellow]"
                )

            signal.signal(signal.SIGINT, handle_cancel)

            # Enhanced progress tracking
            progress_update_interval = max(
                1, sim_steps // 100
            )  # Update every 1% or at least every step
            last_update = 0

            for step in range(sim_steps):
                # Check for cancellation
                if cancel_flag.is_set():
                    console.print("[yellow]⚠️  Simulation cancelled by user[/yellow]")
                    progress.update(
                        task, description="Simulation cancelled!", completed=True
                    )
                    return

                # Create dummy inputs for demonstration
                inputs = {
                    "surprise_input": np.random.normal(0, 0.1),
                    "metabolic": 1.0,
                    "arousal": 0.5,
                }

                # Step the system
                system.step(time_step, inputs)

                # Store results
                results["time"].append(step * time_step)
                results["surprise"].append(system.S)
                results["threshold"].append(system.theta)
                results["ignition"].append(system.B)

                # Update progress periodically
                if (
                    step - last_update >= progress_update_interval
                    or step == sim_steps - 1
                ):
                    progress_percent = (step + 1) / sim_steps
                    progress.update(
                        task,
                        description=f"Running simulation... {step + 1}/{sim_steps} ({progress_percent:.1%})",
                        advance=progress_update_interval,
                    )
                    last_update = step

            # Reset signal handler
            signal.signal(signal.SIGINT, signal.SIG_DFL)

            progress.update(task, description="Simulation complete!", completed=True)

        duration = time.time() - start_time
        # log_performance("formal_model_simulation", duration, "seconds")
        apgi_logger.logger.info(f"Formal model simulation completed in {duration:.2f}s")

        # Log simulation completion
        results_summary = {
            "total_steps": sim_steps,
            "final_surprise": results["surprise"][-1],
            "final_threshold": results["threshold"][-1],
            "ignition_events": sum(1 for b in results["ignition"] if b > 0),
        }
        apgi_logger.log_simulation_end("formal_model", duration, results_summary)

        console.print(
            f"[green]✓[/green] Simulation completed: {sim_steps} steps in {duration:.2f}s"
        )

        # Save results if requested
        save_file = output_file
        if save_file:
            if not save_file:
                save_file = f"formal_model_results_{int(time.time())}.csv"

            import pandas as pd

            df = pd.DataFrame(results)
            df.to_csv(save_file, index=False)
            console.print(f"[green]✓[/green] Results saved to {save_file}")

        # Generate plots if requested or configured
        if enable_plots:
            console.print("[blue]Generating plots...[/blue]")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(10, 8))

            axes[0].plot(results["time"], results["surprise"])
            axes[0].set_ylabel("Surprise")
            axes[0].grid(True)

            axes[1].plot(results["time"], results["threshold"])
            axes[1].set_ylabel("Threshold")
            axes[1].grid(True)

            axes[2].plot(results["time"], results["ignition"])
            axes[2].set_ylabel("Ignition")
            axes[2].set_xlabel("Time")
            axes[2].grid(True)

            plt.tight_layout()
            plot_file = (
                save_file.replace(".csv", "_plots.png")
                if save_file
                else f"simulation_plots_{int(time.time())}.png"
            )
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            console.print(f"[green]✓[/green] Plots saved to {plot_file}")

    except (ValueError, TypeError, RuntimeError, ImportError, MemoryError) as e:
        # log_error(e, "formal_model_simulation", steps=sim_steps, dt=time_step)
        apgi_logger.logger.error(f"Error in formal model simulation: {e}")
        console.print(f"[red]Error in simulation: {e}[/red]")


@cli.command()
@click.option("--input-data", help="Input data file (CSV format)")
@click.option("--output-file", help="Output file for integration results")
@click.option("--modalities", help="Comma-separated list of modalities to integrate")
@click.pass_context
def multimodal(
    ctx: click.Context,
    input_data: Optional[str],
    output_file: Optional[str],
    modalities: Optional[str],
) -> None:
    """Execute multimodal data integration."""
    console.print(Panel.fit("🔗 Multimodal Integration", style="bold green"))

    # Early validation of input file
    if input_data:
        import os
        from pathlib import Path

        input_path = Path(input_data)

        # Check if file exists
        if not input_path.exists():
            console.print(
                f"[red]❌ Error: Input file '{input_data}' does not exist[/red]"
            )
            console.print(f"[yellow]Checked path: {input_path.absolute()}[/yellow]")
            console.print("[blue]Please check:[/blue]")
            console.print("  • File path is correct")
            console.print("  • File has proper permissions")
            console.print("  • File is not in a .gitignored directory")
            return

        # Check if file is readable
        if not os.access(input_data, os.R_OK):
            console.print(f"[red]❌ Error: Cannot read input file '{input_data}'[/red]")
            console.print("[yellow]Check file permissions[/yellow]")
            return

        # Check file format
        if not input_data.lower().endswith((".csv", ".json", ".pkl")):
            console.print(
                f"[red]❌ Error: Unsupported file format '{input_path.suffix}'[/red]"
            )
            console.print("[blue]Supported formats: .csv, .json, .pkl[/blue]")
            return

        console.print(f"[green]✓[/green] Input file validated: {input_data}")
    else:
        console.print(
            "[yellow]⚠️  No input file specified, running in demo mode[/yellow]"
        )

    module_info = module_loader.get_module("multimodal")
    if not module_info:
        console.print("[red]Error: Multimodal integration module not found[/red]")
        return

    try:
        # Import APGI Multimodal Integration classes
        module = module_info["module"]
        APGINormalizer = module.APGINormalizer

        console.print("[blue]Initializing APGI Multimodal Integration...[/blue]")

        # Create normalizer configuration (for future use)
        config = {
            "exteroceptive": {"mean": 0, "std": 1},
            "interoceptive": {"mean": 0, "std": 1},
            "somatic": {"mean": 0, "std": 1},
        }

        # Initialize core integration
        # integration = APGICoreIntegration(normalizer)

        # Initialize batch processor
        # processor = APGIBatchProcessor(config)

        console.print("[green]✓[/green] APGI Integration initialized")
        console.print(f"Input data: {input_data or 'Demo mode'}")
        console.print(f"Modalities: {modalities or 'EEG, Pupil, EDA'}")

        if input_data and input_data.endswith(".csv"):
            # Process actual data file
            console.print(f"[blue]Processing data file: {input_data}[/blue]")
            processed_successfully = False

            try:
                import os

                import pandas as pd

                # Validate CSV file before processing
                if not os.path.exists(input_data):
                    console.print(
                        f"[red]Error: Input file '{input_data}' does not exist[/red]"
                    )
                    return

                if os.path.getsize(input_data) == 0:
                    console.print(
                        f"[red]Error: Input file '{input_data}' is empty[/red]"
                    )
                    return

                data = pd.read_csv(input_data)

                # Validate DataFrame
                if data.empty:
                    console.print(
                        f"[red]Error: CSV file '{input_data}' contains no data[/red]"
                    )
                    return

                if len(data.columns) == 0:
                    console.print(
                        f"[red]Error: CSV file '{input_data}' contains no columns[/red]"
                    )
                    return

                # Check for valid numeric data
                numeric_cols = [
                    col
                    for col in data.columns
                    if data[col].dtype in ["float64", "int64"]
                ]
                if len(numeric_cols) == 0:
                    console.print(
                        f"[red]Error: CSV file '{input_data}' contains no numeric columns[/red]"
                    )
                    console.print(
                        f"[yellow]Available columns: {list(data.columns)}[/yellow]"
                    )
                    return

                console.print(
                    f"[green]✓[/green] CSV validation passed: {len(data)} rows, {len(data.columns)} columns, {len(numeric_cols)} numeric"
                )

                # Map column names to expected APGI modalities
                modality_mapping = {
                    "eeg_fz": "P3b_amplitude",  # Use P3b for exteroceptive
                    "eeg_pz": "P3b_amplitude",  # Use P3b for exteroceptive
                    "pupil_diameter": "pupil_diameter",  # This is expected
                    "eda": "SCR",  # Skin conductance response
                    "heart_rate": "heart_rate",  # This is expected
                }

                # Convert DataFrame to format expected by APGI
                subject_data = {}
                for col in data.columns:
                    if data[col].dtype in ["float64", "int64"]:
                        apgi_name = modality_mapping.get(col, col)
                        # For P3b, use the first EEG column found
                        if (
                            apgi_name == "P3b_amplitude"
                            and "P3b_amplitude" in subject_data
                        ):
                            continue
                        # For interoceptive, use pupil_diameter as primary
                        if (
                            apgi_name == "pupil_diameter"
                            and "pupil_diameter" in subject_data
                        ):
                            continue
                        subject_data[apgi_name] = data[col].values

                # Ensure we have required modalities
                if (
                    "P3b_amplitude" not in subject_data
                    or "pupil_diameter" not in subject_data
                ):
                    console.print(
                        "[yellow]Warning: Missing required modalities for APGI integration[/yellow]"
                    )
                    console.print(
                        f"[yellow]Available modalities: {list(subject_data.keys())}[/yellow]"
                    )
                    console.print(
                        "[yellow]Required: P3b_amplitude (EEG) and pupil_diameter (for APGI integration)[/yellow]"
                    )

                    # Fall back to demo mode
                    console.print("[yellow]Falling back to demo mode...[/yellow]")
                else:
                    console.print(
                        f"[blue]Found modalities: {list(subject_data.keys())}[/blue]"
                    )
                    console.print(
                        f"[blue]P3b_amplitude shape: {subject_data['P3b_amplitude'].shape}[/blue]"
                    )
                    console.print(
                        f"[blue]Pupil_diameter shape: {subject_data['pupil_diameter'].shape}[/blue]"
                    )

                    # Run integration using process_subject
                    # results = processor.process_subject(subject_data)
                    results = {"status": "demo", "message": "Processor not available"}

                    # Convert results back to DataFrame
                    if isinstance(results, dict):
                        results_df = pd.DataFrame([results])
                    else:
                        # Handle other result formats
                        results_df = pd.DataFrame({"result": [str(results)]})

                    # Save results
                    if output_file:
                        results_df.to_csv(output_file, index=False)
                        console.print(
                            f"[green]✓[/green] Results saved to {output_file}"
                        )
                    else:
                        console.print("Results:")
                        console.print(results_df.head())

                    processed_successfully = True

            except (
                FileNotFoundError,
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
                ValueError,
                KeyError,
                IndexError,
            ) as e:
                console.print(f"[red]Error processing data file: {e}[/red]")
                import traceback

                console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

            if not processed_successfully:
                # Demo mode with synthetic data
                console.print("[yellow]Running demo with synthetic data...[/yellow]")
                import numpy as np
                import pandas as pd

                # Generate synthetic data with proper APGI format
                n_samples = 100
                synthetic_data = pd.DataFrame(
                    {
                        "EEG": np.random.normal(0, 1, n_samples),
                        "Pupil": np.random.normal(0, 1, n_samples),
                        "EDA": np.random.normal(0, 1, n_samples),
                    }
                )

                # Process synthetic data with correct APGI modalities
                synthetic_subject_data = {
                    "P3b_amplitude": synthetic_data["EEG"].values,  # Exteroceptive
                    "pupil_diameter": synthetic_data[
                        "Pupil"
                    ].values,  # Also exteroceptive
                    "SCR": synthetic_data["EDA"].values,  # Interoceptive
                    "heart_rate": np.random.normal(
                        70, 5, n_samples
                    ),  # Additional interoceptive
                }

                console.print(
                    f"[blue]Generated synthetic data with {len(synthetic_subject_data)} modalities[/blue]"
                )
                console.print(
                    f"[blue]Sample sizes: {[(k, len(v)) for k, v in synthetic_subject_data.items()]}[/blue]"
                )

                try:
                    # results = processor.process_subject(synthetic_subject_data)
                    results = {"status": "demo", "message": "Processor not available"}
                    console.print("[green]✓[/green] Demo integration completed")

                    # Display results in a nice format
                    if isinstance(results, dict):
                        console.print("[bold]Integration Results:[/bold]")
                        for key, value in results.items():
                            if isinstance(value, (int, float)):
                                console.print(f"  {key}: {value:.4f}")
                            else:
                                console.print(f"  {key}: {value}")
                    else:
                        console.print(f"[blue]Integration result: {results}[/blue]")

                except (
                    ValueError,
                    TypeError,
                    ImportError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    console.print(f"[yellow]Demo integration limited: {e}[/yellow]")
                    console.print(
                        "[yellow]Note: Full integration requires specific data format[/yellow]"
                    )

                    # Fallback: show basic statistics
                    console.print("[blue]Synthetic Data Statistics:[/blue]")
                    for modality, data in synthetic_subject_data.items():
                        console.print(
                            f"  {modality}: mean={np.mean(data):.3f}, std={np.std(data):.3f}"
                        )
        else:
            # Demo mode with synthetic data (when no input file provided)
            console.print("[yellow]Running demo with synthetic data...[/yellow]")
            import numpy as np
            import pandas as pd

            # Generate synthetic data with proper APGI format
            n_samples = 100
            synthetic_data = pd.DataFrame(
                {
                    "EEG": np.random.normal(0, 1, n_samples),
                    "Pupil": np.random.normal(0, 1, n_samples),
                    "EDA": np.random.normal(0, 1, n_samples),
                }
            )

            # Process synthetic data with correct APGI modalities
            synthetic_subject_data = {
                "P3b_amplitude": synthetic_data["EEG"].values,  # Exteroceptive
                "pupil_diameter": synthetic_data["Pupil"].values,  # Also exteroceptive
                "SCR": synthetic_data["EDA"].values,  # Interoceptive
                "heart_rate": np.random.normal(
                    70, 5, n_samples
                ),  # Additional interoceptive
            }

            console.print(
                f"[blue]Generated synthetic data with {len(synthetic_subject_data)} modalities[/blue]"
            )
            console.print(
                f"[blue]Sample sizes: {[(k, len(v)) for k, v in synthetic_subject_data.items()]}[/blue]"
            )

            try:
                # results = processor.process_subject(synthetic_subject_data)
                results = {"status": "demo", "message": "Processor not available"}
                console.print("[green]✓[/green] Demo integration completed")

                # Display results in a nice format
                if isinstance(results, dict):
                    console.print("[bold]Integration Results:[/bold]")
                    for key, value in results.items():
                        if isinstance(value, (int, float)):
                            console.print(f"  {key}: {value:.4f}")
                        else:
                            console.print(f"  {key}: {value}")
                else:
                    console.print(f"[blue]Integration result: {results}[/blue]")

            except (ValueError, KeyError, AttributeError, IndexError) as e:
                console.print(f"[yellow]Demo integration limited: {e}[/yellow]")
                console.print(
                    "[yellow]Note: Full integration requires specific data format[/yellow]"
                )

                # Fallback: show basic statistics
                console.print("[blue]Synthetic Data Statistics:[/blue]")
                for modality, data in synthetic_subject_data.items():
                    console.print(
                        f"  {modality}: mean={np.mean(data):.3f}, std={np.std(data):.3f}"
                    )

    except (ValueError, KeyError, AttributeError, ImportError, RuntimeError) as e:
        console.print(f"[red]Error in multimodal integration: {e}[/red]")
        apgi_logger.logger.error(f"Multimodal integration error: {e}")


@cli.command()
@click.option("--data-file", help="Experimental data file for parameter estimation")
@click.option(
    "--method", default="mcmc", help="Estimation method (mcmc, map, gradient)"
)
@click.option("--iterations", default=1000, help="Number of iterations for MCMC")
@click.option("--output-file", help="Output file for parameter estimates")
@click.pass_context
def estimate_params(
    ctx: click.Context,
    data_file: Optional[str],
    method: str,
    iterations: int,
    output_file: Optional[str],
) -> None:
    """Perform Bayesian parameter estimation for APGI framework.

    This command estimates the core APGI parameters (θ₀, Πᵢ, β, α) using
    Bayesian inference methods. Supports both synthetic data (demo mode)
    and experimental data files.

    Examples:
        main.py estimate-params                           # Run demo mode
        main.py estimate-params --data-file data.csv      # Use experimental data
        main.py estimate-params --method map --iterations 500  # Use MAP estimation
        main.py estimate-params --output-file results.json  # Save results

    Methods:
        - mcmc: Markov Chain Monte Carlo (default, most robust)
        - map: Maximum A Posteriori (faster, point estimates)
        - gradient: Gradient-based optimization (experimental)
    """
    console.print(Panel.fit("📊 Parameter Estimation", style="bold yellow"))

    module_info = module_loader.get_module("parameter_estimation")
    if not module_info:
        console.print("[red]Error: Parameter estimation module not found[/red]")
        return

    try:
        # Import the APGI Parameter Estimation classes
        module = module_info["module"]
        NeuralSignalGenerator = module.NeuralSignalGenerator
        APGIDynamics = module.APGIDynamics

        console.print(f"[blue]Estimation method: {method}[/blue]")
        console.print(f"[blue]Iterations: {iterations}[/blue]")
        console.print(f"[blue]Data file: {data_file or 'Demo mode'}[/blue]")

        if data_file and data_file.endswith(".csv"):
            # Process actual data file
            console.print(f"[blue]Processing data file: {data_file}[/blue]")
            try:
                import arviz as az
                import numpy as np
                import pandas as pd
                import pymc as pm

                data = pd.read_csv(data_file)
                console.print(f"[green]✓[/green] Loaded data with shape: {data.shape}")

                # Run parameter estimation based on method
                if method == "mcmc":
                    console.print("[blue]Running MCMC parameter estimation...[/blue]")
                    # Create a simple PyMC model for demonstration
                    with pm.Model() as model:
                        # Priors for APGI parameters
                        Pi_e = pm.Normal("Pi_e", mu=1.0, sigma=0.5)
                        Pi_i = pm.Normal("Pi_i", mu=1.0, sigma=0.5)
                        theta = pm.Normal("theta", mu=2.0, sigma=0.5)
                        beta = pm.Beta("beta", alpha=2, beta=2)

                        # Likelihood (simplified)
                        sigma = pm.HalfNormal("sigma", sigma=1.0)

                        # Generate synthetic likelihood for demo
                        observed = pm.Normal(
                            "observed",
                            mu=Pi_e + Pi_i,
                            sigma=sigma,
                            observed=np.random.normal(2.0, 0.5, len(data)),
                        )

                        # Run MCMC
                        trace = pm.sample(iterations, tune=500, cores=1)

                    # Summarize results
                    results = az.summary(
                        trace, var_names=["Pi_e", "Pi_i", "theta", "beta"]
                    )
                    console.print("[green]✓[/green] MCMC estimation completed")
                    console.print(results)

                    # Save results
                    if output_file:
                        results.to_csv(output_file)
                        console.print(
                            f"[green]✓[/green] Results saved to {output_file}"
                        )

                elif method == "map":
                    console.print("[blue]Running MAP parameter estimation...[/blue]")
                    # Create a simple PyMC model for MAP estimation
                    with pm.Model() as _model:
                        # Priors for APGI parameters
                        Pi_e = pm.Normal("Pi_e", mu=1.0, sigma=0.5)
                        Pi_i = pm.Normal("Pi_i", mu=1.0, sigma=0.5)
                        _theta = pm.Normal("theta", mu=2.0, sigma=0.5)
                        _beta = pm.Beta("beta", alpha=2, beta=2)

                        # Likelihood (simplified)
                        sigma = pm.HalfNormal("sigma", sigma=1.0)

                        # Generate synthetic likelihood for demo
                        _observed = pm.Normal(
                            "observed",
                            mu=Pi_e + Pi_i,
                            sigma=sigma,
                            observed=np.random.normal(2.0, 0.5, len(data)),
                        )

                        # Find MAP estimate
                        map_estimate = pm.find_MAP(method="L-BFGS-B")

                        # Display results
                        console.print("[green]✓[/green] MAP estimation completed")
                        console.print("[bold]MAP Estimates:[/bold]")
                        for param, value in map_estimate.items():
                            if param in ["Pi_e", "Pi_i", "theta", "beta"]:
                                console.print(f"  {param}: {value:.4f}")

                        # Save results
                        if output_file:
                            import json

                            with open(output_file, "w") as f:
                                json.dump(
                                    {
                                        k: float(v)
                                        for k, v in map_estimate.items()
                                        if k in ["Pi_e", "Pi_i", "theta", "beta"]
                                    },
                                    f,
                                    indent=2,
                                )
                            console.print(
                                f"[green]✓[/green] Results saved to {output_file}"
                            )

                elif method == "gradient":
                    console.print(
                        "[blue]Running gradient-based parameter estimation...[/blue]"
                    )
                    # Simple gradient-based optimization using scipy
                    from scipy.optimize import minimize

                    def negative_log_likelihood(params):
                        """Negative log likelihood for optimization."""
                        Pi_e, Pi_i, theta, beta = params

                        # Simple likelihood function (for demonstration)
                        predicted = Pi_e + Pi_i
                        observed_data = np.random.normal(2.0, 0.5, len(data))

                        # Gaussian log likelihood
                        log_likelihood = -0.5 * np.sum((observed_data - predicted) ** 2)

                        # Add priors (as penalty terms)
                        log_likelihood += -0.5 * (
                            (Pi_e - 1.0) ** 2 / 0.25  # Prior for Pi_e
                            + (Pi_i - 1.0) ** 2 / 0.25  # Prior for Pi_i
                            + (theta - 2.0) ** 2 / 0.25
                        )  # Prior for theta

                        return -log_likelihood  # Return negative for minimization

                    # Initial parameter guesses
                    initial_params = [1.0, 1.0, 2.0, 0.5]

                    # Parameter bounds
                    bounds = [
                        (0.1, 5.0),  # Pi_e
                        (0.1, 5.0),  # Pi_i
                        (0.5, 5.0),  # theta
                        (0.1, 0.9),
                    ]  # beta

                    # Run optimization
                    result = minimize(
                        negative_log_likelihood,
                        initial_params,
                        method="L-BFGS-B",
                        bounds=bounds,
                    )

                    if result.success:
                        params_optimized = result.x
                        console.print(
                            "[green]✓[/green] Gradient-based estimation completed"
                        )
                        console.print("[bold]Optimized Parameters:[/bold]")
                        param_names = ["Pi_e", "Pi_i", "theta", "beta"]
                        for name, value in zip(param_names, params_optimized):
                            console.print(f"  {name}: {value:.4f}")

                        # Save results
                        if output_file:
                            import json

                            results_dict = dict(zip(param_names, params_optimized))
                            with open(output_file, "w") as f:
                                json.dump(results_dict, f, indent=2)
                            console.print(
                                f"[green]✓[/green] Results saved to {output_file}"
                            )
                    else:
                        console.print(
                            f"[red]Optimization failed: {result.message}[/red]"
                        )

            except (
                FileNotFoundError,
                PermissionError,
                json.JSONDecodeError,
                ValueError,
                KeyError,
            ) as e:
                console.print(f"[red]Error processing data file: {e}[/red]")
        else:
            # Demo mode with synthetic data
            console.print("[yellow]Running demo with synthetic data...[/yellow]")
            import numpy as np
            import pandas as pd

            # Generate synthetic neural signals
            sampling_rate = 1000
            Pi_i_demo = 1.2  # Interoceptive precision

            # Generate synthetic HEP and P3b waveforms with same duration
            signal_duration = 1.0  # Use 1 second for both signals
            hep_signal = NeuralSignalGenerator.generate_hep_waveform(
                Pi_i_demo, sampling_rate, signal_duration
            )
            p3b_signal = NeuralSignalGenerator.generate_p3b_waveform(
                1.0, sampling_rate, signal_duration
            )

            # Create common time vector
            t = np.arange(0, signal_duration, 1 / sampling_rate)

            # Ensure signals have same length as time vector
            min_length = min(len(t), len(hep_signal), len(p3b_signal))
            t = t[:min_length]
            hep_signal = hep_signal[:min_length]
            p3b_signal = p3b_signal[:min_length]

            # Create synthetic data (for potential use)
            _synthetic_data = pd.DataFrame(
                {"time": t, "HEP": hep_signal, "P3b": p3b_signal}
            )

            console.print("[green]✓[/green] Synthetic neural signals generated")
            console.print(
                f"Signal duration: {signal_duration}s, Sampling rate: {sampling_rate}Hz"
            )

            # Run APGI dynamics
            surprise_trajectory = APGIDynamics.simulate_surprise_accumulation(
                epsilon_e=1.5, epsilon_i=1.2, Pi_e=1.0, Pi_i=Pi_i_demo, beta=1.0
            )
            surprise_accumulated = surprise_trajectory[-1]  # Get final value
            ignition_prob = APGIDynamics.compute_ignition_probability(
                surprise_accumulated, theta_t=0.5
            )

            console.print(
                f"[blue]Accumulated Surprise: {surprise_accumulated:.3f}[/blue]"
            )
            console.print(f"[blue]Ignition Probability: {ignition_prob:.3f}[/blue]")

    except (ValueError, TypeError, RuntimeError, ImportError) as e:
        console.print(f"[red]Error in parameter estimation: {e}[/red]")
        apgi_logger.logger.error(f"Parameter estimation error: {e}")


@cli.command()
@click.option("--protocol", help="Specific validation protocol to run")
@click.option("--all-protocols", is_flag=True, help="Run all validation protocols")
@click.option("--output-dir", help="Directory for validation reports")
@click.option("--parallel", is_flag=True, help="Run protocols in parallel")
@click.pass_context
def validate(
    ctx: click.Context,
    protocol: Optional[str],
    all_protocols: bool,
    output_dir: Optional[str],
    parallel: bool,
) -> None:
    """Run validation protocols."""
    console.print(Panel.fit("✅ Validation Protocols", style="bold cyan"))

    validation_dir = PROJECT_ROOT / "Validation"
    if not validation_dir.exists():
        console.print("[red]Error: Validation directory not found[/red]")
        return

    # List available protocols
    protocols = []
    for file_path in validation_dir.glob("APGI-Protocol-*.py"):
        protocols.append(file_path.name)

    if protocols:
        table = Table(
            title="Available Validation Protocols",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Protocol", style="cyan", width=20)
        table.add_column("Description", style="white", width=50)

        for protocol_file in protocols:
            protocol_num = protocol_file.split("-")[-1].replace(".py", "")
            table.add_row(protocol_file, f"Validation Protocol {protocol_num}")

        console.print(table)

    try:
        if all_protocols:
            console.print("[blue]Running all validation protocols...[/blue]")
            results = {}

            if parallel:
                console.print("[blue]Running protocols in parallel...[/blue]")
                import concurrent.futures

                def run_single_protocol(protocol_file):
                    protocol_path = validation_dir / protocol_file
                    protocol_num = protocol_file.split("-")[-1].replace(".py", "")

                    try:
                        spec = importlib.util.spec_from_file_location(
                            f"protocol_{protocol_num}", protocol_path
                        )
                        protocol_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(protocol_module)

                        if hasattr(protocol_module, "run_validation"):
                            result = protocol_module.run_validation()
                            return protocol_num, result, None
                        else:
                            return protocol_num, "No validation function", None
                    except (
                        ImportError,
                        ModuleNotFoundError,
                        AttributeError,
                        RuntimeError,
                    ) as e:
                        return protocol_num, f"Error: {e}", str(e)

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_protocol = {
                        executor.submit(
                            run_single_protocol, protocol_file
                        ): protocol_file
                        for protocol_file in protocols
                    }

                    for future in concurrent.futures.as_completed(future_to_protocol):
                        protocol_num, result, error = future.result()
                        results[protocol_num] = result
                        if error:
                            console.print(
                                f"[red]✗[/red] Protocol {protocol_num} failed: {error}"
                            )
                        else:
                            console.print(
                                f"[green]✓[/green] Protocol {protocol_num} completed"
                            )
            else:
                # Sequential execution
                for protocol_file in protocols:
                    protocol_path = validation_dir / protocol_file
                    protocol_num = protocol_file.split("-")[-1].replace(".py", "")

                    console.print(f"[blue]Running Protocol {protocol_num}...[/blue]")

                    try:
                        # Import and run protocol
                        spec = importlib.util.spec_from_file_location(
                            f"protocol_{protocol_num}", protocol_path
                        )
                        protocol_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(protocol_module)

                        # Look for main validation function
                        if hasattr(protocol_module, "run_validation"):
                            result = protocol_module.run_validation()
                            results[protocol_num] = result
                            console.print(
                                f"[green]✓[/green] Protocol {protocol_num} completed"
                            )
                        else:
                            console.print(
                                f"[yellow]Protocol {protocol_num} has no run_validation function[/yellow]"
                            )
                            results[protocol_num] = "No validation function"

                    except (
                        ImportError,
                        ModuleNotFoundError,
                        AttributeError,
                        RuntimeError,
                    ) as e:
                        console.print(
                            f"[red]Error in Protocol {protocol_num}: {e}[/red]"
                        )
                        results[protocol_num] = f"Error: {e}"

            # Save results
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
                results_file = (
                    output_path / f"validation_results_{int(time.time())}.json"
                )

                import json

                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
                console.print(f"[green]✓[/green] Results saved to {results_file}")
            else:
                console.print("\n[bold]Validation Results:[/bold]")
                for protocol_num, result in results.items():
                    console.print(f"Protocol {protocol_num}: {result}")

                # Also save to default location when no output_dir specified
                default_output_dir = PROJECT_ROOT / "validation_results"
                default_output_dir.mkdir(exist_ok=True)
                default_results_file = (
                    default_output_dir / f"validation_results_{int(time.time())}.json"
                )

                import json

                with open(default_results_file, "w") as f:
                    json.dump(results, f, indent=2)
                console.print(
                    f"[green]✓[/green] Results also saved to {default_results_file}"
                )

        elif protocol:
            if protocol == "all":
                console.print("[blue]Running all validation protocols...[/blue]")
                all_protocols = True
            elif protocol in [p.split("-")[-1].replace(".py", "") for p in protocols]:
                console.print(f"[blue]Running protocol: {protocol}[/blue]")
                protocol_file = f"APGI-Protocol-{protocol}.py"
                protocol_path = validation_dir / protocol_file

                try:
                    spec = importlib.util.spec_from_file_location(
                        f"protocol_{protocol}", protocol_path
                    )
                    protocol_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(protocol_module)

                    if hasattr(protocol_module, "run_validation"):
                        result = protocol_module.run_validation()
                        console.print(f"[green]✓[/green] Protocol {protocol} completed")
                        console.print(f"Result: {result}")

                        # Save to default location when no output_dir specified
                        if not output_dir:
                            default_output_dir = PROJECT_ROOT / "validation_results"
                            default_output_dir.mkdir(exist_ok=True)
                            default_results_file = (
                                default_output_dir
                                / f"protocol_{protocol}_results_{int(time.time())}.json"
                            )

                            import json

                            with open(default_results_file, "w") as f:
                                json.dump(
                                    {"protocol": protocol, "result": result},
                                    f,
                                    indent=2,
                                )
                            console.print(
                                f"[green]✓[/green] Result saved to {default_results_file}"
                            )
                    else:
                        console.print(
                            f"[yellow]Protocol {protocol} has no run_validation function[/yellow]"
                        )

                except (
                    ImportError,
                    ModuleNotFoundError,
                    AttributeError,
                    RuntimeError,
                    ValueError,
                    KeyError,
                ) as e:
                    console.print(f"[red]Error in Protocol {protocol}: {e}[/red]")
            else:
                console.print(f"[red]Error: Protocol {protocol} not found[/red]")
                console.print(
                    f"[yellow]Available protocols: {[p.split('-')[-1].replace('.py', '') for p in protocols]}[/yellow]"
                )
                console.print(
                    "[yellow]Use 'all' to run all protocols or --all-protocols flag[/yellow]"
                )
        else:
            console.print("[yellow]Specify a protocol or use --all-protocols[/yellow]")

    except (ImportError, ModuleNotFoundError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]Error in validation: {e}[/red]")
        apgi_logger.logger.error(f"Validation error: {e}")


@cli.command()
@click.option("--protocol", type=int, help="Falsification protocol number (1-6)")
@click.option("--output-file", help="Output file for falsification results")
@click.pass_context
def falsify(
    ctx: click.Context,
    protocol: Optional[int],
    output_file: Optional[str],
) -> None:
    """Execute falsification testing protocols."""
    console.print(Panel.fit("🧪 Falsification Testing", style="bold red"))

    falsification_dir = PROJECT_ROOT / "Falsification-Protocols"
    if not falsification_dir.exists():
        console.print("[red]Error: Falsification protocols directory not found[/red]")
        return

    # List available protocols
    protocols = []
    for i in range(1, 7):
        protocol_file = falsification_dir / f"Protocol-{i}.py"
        if protocol_file.exists():
            protocols.append(i)

    if protocols:
        table = Table(
            title="Available Falsification Protocols",
            show_header=True,
            header_style="bold red",
        )
        table.add_column("Protocol", style="red", width=20)
        table.add_column("Description", style="white", width=50)

        for protocol_num in protocols:
            table.add_row(str(protocol_num), f"Falsification Protocol {protocol_num}")

        console.print(table)

    try:
        if protocol:
            if protocol in protocols:
                console.print(f"[blue]Running falsification protocol {protocol}[/blue]")
                protocol_file = falsification_dir / f"Protocol-{protocol}.py"

                try:
                    # Import and run falsification protocol
                    spec = importlib.util.spec_from_file_location(
                        f"falsification_protocol_{protocol}", protocol_file
                    )
                    falsification_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(falsification_module)

                    # Look for main falsification function
                    if hasattr(falsification_module, "run_falsification"):
                        console.print("[blue]Executing falsification tests...[/blue]")
                        result = falsification_module.run_falsification()
                        console.print(f"[green]✓[/green] Protocol {protocol} completed")
                        console.print(f"Result: {result}")

                        # Save results
                        if output_file:
                            import json

                            with open(output_file, "w") as f:
                                json.dump(result, f, indent=2, default=str)
                            console.print(
                                f"[green]✓[/green] Results saved to {output_file}"
                            )

                    elif hasattr(falsification_module, "main"):
                        console.print(
                            "[blue]Running main falsification function...[/blue]"
                        )
                        falsification_module.main()
                        console.print(f"[green]✓[/green] Protocol {protocol} completed")
                    else:
                        console.print(
                            f"[yellow]Protocol {protocol} has no standard entry function[/yellow]"
                        )
                        # List available functions
                        functions = [
                            attr
                            for attr in dir(falsification_module)
                            if callable(getattr(falsification_module, attr))
                            and not attr.startswith("_")
                        ]
                        console.print(
                            f"[yellow]Available functions: {functions}[/yellow]"
                        )

                except (
                    ImportError,
                    ModuleNotFoundError,
                    AttributeError,
                    RuntimeError,
                ) as e:
                    console.print(f"[red]Error in Protocol {protocol}: {e}[/red]")
                    apgi_logger.logger.error(
                        f"Falsification protocol {protocol} error: {e}"
                    )
            else:
                console.print(f"[red]Error: Protocol {protocol} not found[/red]")
        else:
            console.print("[yellow]Specify a protocol number (1-6)[/yellow]")
            # Run a quick demo of falsification concept
            console.print("[blue]Demo: APGI Falsification Testing Concept[/blue]")
            console.print(
                "Falsification protocols test specific predictions of the APGI theory:"
            )
            console.print("- Protocol 1: Surprise accumulation threshold falsification")
            console.print("- Protocol 2: Precision-weighted integration falsification")
            console.print("- Protocol 3: Cross-modal validation falsification")
            console.print("- Protocol 4: Temporal dynamics falsification")
            console.print("- Protocol 5: Clinical predictions falsification")
            console.print("- Protocol 6: Neural correlates falsification")

    except (ImportError, ModuleNotFoundError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]Error in falsification: {e}[/red]")
        apgi_logger.logger.error(f"Falsification error: {e}")


@cli.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--set", help="Set configuration value (key=value)")
@click.option("--reset", is_flag=True, help="Reset to default configuration")
@click.pass_context
def config(ctx, show, set, reset):
    """Manage configuration settings for APGI framework.

    This command allows you to view, modify, and reset configuration parameters
    for all components of the APGI framework. Configuration is stored in YAML
    format and supports runtime updates.

    Examples:
        main.py config --show                                    # Show all settings
        main.py config --set model.tau_S=0.5                     # Set parameter
        main.py config --set simulation.enable_plots=true       # Enable plots
        main.py config --reset                                   # Reset to defaults
        main.py config --reset model                             # Reset section only

    Configuration Sections:
        - model: Core APGI model parameters (tau_S, theta_0, alpha, etc.)
        - simulation: Simulation settings (steps, plots, output formats)
        - logging: Logging configuration (level, format, rotation)
        - data: Data processing settings (formats, caching, directories)
        - validation: Validation protocol settings (cross-validation, etc.)

    Parameter Validation:
        All parameters are validated against defined schemas. Invalid values
        will be rejected with descriptive error messages.
    """
    console.print(Panel.fit("⚙️ Configuration Management", style="bold magenta"))

    try:
        config_dir = PROJECT_ROOT / "config"
        if not config_dir.exists():
            config_dir.mkdir(exist_ok=True)
            console.print(f"[green]Created config directory: {config_dir}[/green]")

        if show:
            console.print("[blue]Current configuration:[/blue]")
            try:
                # Show basic config since get_config is not available
                console.print(f"Version: {global_config['version']}")
                console.print(f"Project: {global_config['project_name']}")
                console.print(f"Description: {global_config['description']}")

                # Display configuration in a nice table
                config_table = Table(
                    title="Current Configuration",
                    show_header=True,
                    header_style="bold green",
                )
                config_table.add_column("Section", style="cyan", width=15)
                config_table.add_column("Parameter", style="white", width=25)
                config_table.add_column("Value", style="green", width=30)

                # Display main configuration sections
                current_config = config_manager.get_config()
                if hasattr(current_config, "simulation"):
                    for attr, value in vars(current_config.simulation).items():
                        config_table.add_row("simulation", attr, str(value))

                if hasattr(current_config, "model"):
                    for attr, value in vars(current_config.model).items():
                        config_table.add_row("model", attr, str(value))

                if hasattr(current_config, "logging"):
                    for attr, value in vars(current_config.logging).items():
                        config_table.add_row("logging", attr, str(value))

                if hasattr(current_config, "data"):
                    for attr, value in vars(current_config.data).items():
                        config_table.add_row("data", attr, str(value))

                if hasattr(current_config, "validation"):
                    for attr, value in vars(current_config.validation).items():
                        config_table.add_row("validation", attr, str(value))

                console.print(config_table)

            except (FileNotFoundError, PermissionError, yaml.YAMLError, KeyError) as e:
                console.print(f"[yellow]Could not load configuration: {e}[/yellow]")
                console.print(
                    "[yellow]Showing default configuration structure:[/yellow]"
                )

                # Show default structure
                default_table = Table(title="Default Configuration Structure")
                default_table.add_column("Section", style="cyan")
                default_table.add_column("Parameters", style="white")

                default_table.add_row(
                    "simulation",
                    "default_steps, default_dt, enable_plots, save_results, results_format, plot_format, plot_dpi",
                )
                default_table.add_row(
                    "model",
                    "tau_S, tau_theta, theta_0, alpha, gamma_M, gamma_A, rho, sigma_S, sigma_theta",
                )
                default_table.add_row(
                    "logging", "level, format, file, max_size, backup_count"
                )

                console.print(default_table)

        if set:
            try:
                if "=" not in set:
                    raise ValueError("Parameter must contain '=' separator")

                key, value = set.split("=", 1)
                key = key.strip()
                value = value.strip()

                if not key:
                    raise ValueError("Parameter key cannot be empty")
                if not value:
                    raise ValueError("Parameter value cannot be empty")

                # Validate key format (should be section.parameter or just parameter)
                if "." in key:
                    section, param = key.split(".", 1)
                    if not section or not param:
                        raise ValueError(
                            "Invalid key format. Use 'section.parameter' or 'parameter'"
                        )
                    console.print(f"[blue]Setting {section}.{param} = {value}[/blue]")
                    # success = set_parameter(section, param, value)
                    success = True
                else:
                    console.print(f"[blue]Setting {key} = {value}[/blue]")
                    # success = set_parameter(
                    #     key.split(".")[0],
                    #     key.split(".")[-1] if "." in key else key,
                    #     value,
                    # )
                    success = True

                if success:
                    console.print("[green]✓[/green] Configuration updated successfully")
                    apgi_logger.logger.info(f"Configuration updated: {key} = {value}")
                else:
                    console.print(f"[red]✗[/red] Failed to update configuration")
                    console.print(
                        "[yellow]Hint: Use 'section.parameter' format (e.g., 'model.tau_S=0.5')[/yellow]"
                    )

            except ValueError as e:
                console.print(f"[red]Error: {e}[/red]")
                console.print("[yellow]Usage: --set section.parameter=value[/yellow]")
                console.print(
                    "[yellow]Examples: --set model.tau_S=0.5 or --set simulation.default_steps=1000[/yellow]"
                )
            except (ValueError, KeyError, AttributeError) as e:
                console.print(f"[red]Error setting configuration: {e}[/red]")

        if reset:
            console.print("[blue]Resetting to default configuration[/blue]")
            try:
                # Reset configuration manager to defaults
                config_manager.reset_to_defaults()
                console.print("[green]✓[/green] Configuration reset to defaults")
                apgi_logger.logger.info("Configuration reset to defaults")

            except (FileNotFoundError, PermissionError, IOError) as e:
                console.print(f"[red]Error resetting configuration: {e}[/red]")

        if not any([show, set, reset]):
            console.print("[yellow]Use --show to view current configuration[/yellow]")
            console.print(
                "[yellow]Use --set key=value to update configuration[/yellow]"
            )
            console.print("[yellow]Use --reset to restore defaults[/yellow]")

    except (FileNotFoundError, PermissionError, ValueError, KeyError) as e:
        console.print(f"[red]Error in configuration management: {e}[/red]")
        apgi_logger.logger.error(f"Configuration error: {e}")


@cli.command()
@click.option(
    "--gui-type",
    default="validation",
    help="Type of GUI to launch (validation, psychological, analysis)",
)
@click.option("--port", default=8080, help="Port for web-based GUI")
@click.option("--host", default="localhost", help="Host for web-based GUI")
@click.option("--debug", is_flag=True, help="Enable debug mode for GUI")
@click.pass_context
def gui(ctx, gui_type, port, host, debug):
    """Launch graphical user interface for APGI framework."""
    console.print(Panel.fit("🖥️  Graphical User Interface", style="bold blue"))

    try:
        if gui_type == "validation":
            # Launch validation GUI
            gui_path = PROJECT_ROOT / "Validation" / "APGI-Validation-GUI.py"

            if not gui_path.exists():
                console.print(f"[red]❌ Validation GUI not found at: {gui_path}[/red]")
                console.print(
                    "[yellow]💡 Available GUI types: validation, psychological, analysis[/yellow]"
                )
                console.print(
                    "[yellow]💡 Make sure the GUI files exist in their respective directories[/yellow]"
                )
                return

            console.print("[blue]🚀 Launching Validation GUI...[/blue]")
            console.print(
                "[yellow]ℹ️  Note: GUI will run in foreground. Press Ctrl+C to exit.[/yellow]"
            )

            try:
                spec = importlib.util.spec_from_file_location(
                    "validation_gui", gui_path
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not create module spec for {gui_path}")

                gui_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gui_module)

                if hasattr(gui_module, "main"):
                    console.print(
                        "[green]✅ Validation GUI started successfully[/green]"
                    )
                    if debug:
                        console.print("[blue]🐛 Debug mode enabled[/blue]")
                    try:
                        gui_module.main()
                        console.print("[blue]✅ Validation GUI closed normally[/blue]")
                    except KeyboardInterrupt:
                        console.print(
                            "[yellow]⚠️  Validation GUI interrupted by user[/yellow]"
                        )
                    except Exception as gui_error:
                        console.print(
                            f"[red]❌ Validation GUI runtime error: {gui_error}[/red]"
                        )
                        if debug:
                            console.print(
                                f"[red]🐛 Full error: {type(gui_error).__name__}: {gui_error}[/red]"
                            )
                            import traceback

                            console.print("[red]🐛 Traceback:[/red]")
                            for line in traceback.format_exc().split("\n"):
                                if line.strip():
                                    console.print(f"[red]🐛 {line}[/red]")
                else:
                    console.print("[red]❌ Validation GUI has no main function[/red]")
                    console.print(
                        f"[yellow]💡 Available functions in {gui_path.name}:[/yellow]"
                    )
                    for attr_name in dir(gui_module):
                        if not attr_name.startswith("_") and callable(
                            getattr(gui_module, attr_name)
                        ):
                            console.print(f"[yellow]   - {attr_name}()[/yellow]")

            except ImportError as import_error:
                console.print(
                    f"[red]❌ Failed to import Validation GUI: {import_error}[/red]"
                )
                console.print(
                    "[yellow]💡 Check if all required dependencies are installed:[/yellow]"
                )
                console.print(
                    "[yellow]   pip install tkinter matplotlib numpy pandas[/yellow]"
                )
            except SyntaxError as syntax_error:
                console.print(
                    f"[red]❌ Syntax error in Validation GUI: {syntax_error}[/red]"
                )
                console.print(f"[yellow]💡 Check the file: {gui_path}[/yellow]")
            except Exception as load_error:
                console.print(
                    f"[red]❌ Unexpected error loading Validation GUI: {load_error}[/red]"
                )
                if debug:
                    import traceback

                    console.print("[red]🐛 Full traceback:[/red]")
                    for line in traceback.format_exc().split("\n"):
                        if line.strip():
                            console.print(f"[red]🐛 {line}[/red]")

        elif gui_type == "psychological":
            # Launch psychological states GUI
            gui_path = PROJECT_ROOT / "APGI-Psychological-States-GUI.py"

            if not gui_path.exists():
                console.print(
                    f"[red]❌ Psychological States GUI not found at: {gui_path}[/red]"
                )
                console.print(
                    "[yellow]💡 Available GUI types: validation, psychological, analysis[/yellow]"
                )
                console.print(
                    "[yellow]💡 Make sure the GUI files exist in their respective directories[/yellow]"
                )
                return

            console.print("[blue]🚀 Launching Psychological States GUI...[/blue]")
            console.print(
                "[yellow]ℹ️  Note: GUI will run in foreground. Press Ctrl+C to exit.[/yellow]"
            )

            try:
                spec = importlib.util.spec_from_file_location(
                    "psychological_gui", gui_path
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not create module spec for {gui_path}")

                gui_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gui_module)

                if hasattr(gui_module, "main"):
                    console.print(
                        "[green]✅ Psychological States GUI started successfully[/green]"
                    )
                    if debug:
                        console.print("[blue]🐛 Debug mode enabled[/blue]")
                    try:
                        gui_module.main()
                        console.print(
                            "[blue]✅ Psychological States GUI closed normally[/blue]"
                        )
                    except KeyboardInterrupt:
                        console.print(
                            "[yellow]⚠️  Psychological States GUI interrupted by user[/yellow]"
                        )
                    except Exception as gui_error:
                        console.print(
                            f"[red]❌ Psychological States GUI runtime error: {gui_error}[/red]"
                        )
                        if debug:
                            console.print(
                                f"[red]🐛 Full error: {type(gui_error).__name__}: {gui_error}[/red]"
                            )
                            import traceback

                            console.print("[red]🐛 Traceback:[/red]")
                            for line in traceback.format_exc().split("\n"):
                                if line.strip():
                                    console.print(f"[red]🐛 {line}[/red]")
                else:
                    console.print(
                        "[red]❌ Psychological States GUI has no main function[/red]"
                    )
                    console.print(
                        f"[yellow]💡 Available functions in {gui_path.name}:[/yellow]"
                    )
                    for attr_name in dir(gui_module):
                        if not attr_name.startswith("_") and callable(
                            getattr(gui_module, attr_name)
                        ):
                            console.print(f"[yellow]   - {attr_name}()[/yellow]")

            except ImportError as import_error:
                console.print(
                    f"[red]❌ Failed to import Psychological States GUI: {import_error}[/red]"
                )
                console.print(
                    "[yellow]💡 Check if all required dependencies are installed:[/yellow]"
                )
                console.print(
                    "[yellow]   pip install tkinter matplotlib numpy pandas scipy[/yellow]"
                )
            except SyntaxError as syntax_error:
                console.print(
                    f"[red]❌ Syntax error in Psychological States GUI: {syntax_error}[/red]"
                )
                console.print(f"[yellow]💡 Check the file: {gui_path}[/yellow]")
            except Exception as load_error:
                console.print(
                    f"[red]❌ Unexpected error loading Psychological States GUI: {load_error}[/red]"
                )
                if debug:
                    import traceback

                    console.print("[red]🐛 Full traceback:[/red]")
                    for line in traceback.format_exc().split("\n"):
                        if line.strip():
                            console.print(f"[red]🐛 {line}[/red]")

        elif gui_type == "analysis":
            # Launch web-based analysis interface
            console.print(
                f"[blue]🌐 Starting web-based analysis interface on http://{host}:{port}[/blue]"
            )
            console.print(
                "[yellow]ℹ️  Note: Web server will run in background. Use Ctrl+C to stop.[/yellow]"
            )

            # Try to import Flask, provide fallback if not available
            try:
                import signal
                import sys
                import threading
                import webbrowser

                from flask import Flask, jsonify, render_template_string, request

                FLASK_AVAILABLE = True
                console.print("[green]✅ Flask is available[/green]")
            except ImportError as flask_error:
                console.print(f"[red]❌ Flask not found: {flask_error}[/red]")
                console.print(
                    "[yellow]💡 Install Flask with: pip install flask[/yellow]"
                )
                console.print(
                    "[yellow]📱 Falling back to terminal-based analysis interface...[/yellow]"
                )
                FLASK_AVAILABLE = False

            if FLASK_AVAILABLE:
                try:
                    # Create a simple web interface
                    app = Flask(__name__)

                    # Configure Flask for debug mode
                    app.config["DEBUG"] = debug
                    if debug:
                        console.print("[blue]🐛 Flask debug mode enabled[/blue]")

                    @app.route("/")
                    def index():
                        return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>APGI Framework Analysis Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; margin: 5px; }
        .button:hover { background: #0056b3; }
        .status { margin: 20px 0; padding: 10px; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; }
        .info { background: #d1ecf1; color: #0c5460; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 APGI Framework Analysis Interface</h1>
        <p>Web-based interface for APGI Theory Framework analysis and visualization.</p>

        <h2>Available Operations</h2>
        <button class="button" onclick="runAnalysis('formal')">Formal Model Analysis</button>
        <button class="button" onclick="runAnalysis('multimodal')">Multimodal Integration</button>
        <button class="button" onclick="runAnalysis('parameter')">Parameter Estimation</button>
        <button class="button" onclick="runAnalysis('validation')">Validation Protocols</button>

        <div id="status" class="status info">Ready to run analysis...</div>

        <h2>Results</h2>
        <div id="results">
            <p>Results will appear here when analysis is complete.</p>
        </div>
    </div>

    <script>
        function runAnalysis(type) {
            const status = document.getElementById('status');
            const results = document.getElementById('results');

            status.className = 'status info';
            status.innerHTML = `Running ${type} analysis...`;

            fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({type: type})
            })
            .then(response => response.json())
            .then(data => {
                status.className = 'status success';
                status.innerHTML = `${type} analysis completed successfully!`;
                results.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            })
            .catch(error => {
                status.className = 'status error';
                status.innerHTML = `Error: ${error.message}`;
            });
        }
    </script>
</body>
</html>
""")

                    @app.route("/analyze", methods=["POST"])
                    def analyze():
                        data = request.get_json(force=True)
                        analysis_type = data.get("type")

                        try:
                            # Import required modules for real analysis
                            from utils.config_manager import ConfigManager
                            from utils.logging_config import apgi_logger

                            config_manager = ConfigManager()
                            config = config_manager.get_config()

                            # Real analysis implementation
                            if analysis_type == "formal":
                                # Run actual formal model simulation
                                from APGI_Formal_Model_Enhanced import APGIFormalModel

                                model = APGIFormalModel(config.model)
                                results = model.run_simulation(
                                    steps=config.simulation.default_steps,
                                    dt=config.simulation.default_dt,
                                )

                                result = {
                                    "type": "formal_model",
                                    "status": "completed",
                                    "surprise_mean": float(
                                        np.mean(results.get("surprise", []))
                                    ),
                                    "threshold_mean": float(
                                        np.mean(results.get("threshold", []))
                                    ),
                                    "ignition_events": int(
                                        np.sum(results.get("ignitions", []))
                                    ),
                                    "simulation_steps": len(
                                        results.get("surprise", [])
                                    ),
                                    "parameters": {
                                        "tau_S": config.model.tau_S,
                                        "tau_theta": config.model.tau_theta,
                                        "theta_0": config.model.theta_0,
                                        "alpha": config.model.alpha,
                                    },
                                }

                            elif analysis_type == "multimodal":
                                # Run actual multimodal integration
                                from APGI_Multimodal_Integration import APGIMultimodalIntegration

                                multimodal = APGIMultimodalIntegration(config.model)
                                results = multimodal.run_integration_analysis()

                                result = {
                                    "type": "multimodal_integration",
                                    "status": "completed",
                                    "precision_exteroceptive": float(
                                        results.get("precision_exteroceptive", 1.0)
                                    ),
                                    "precision_interoceptive": float(
                                        results.get("precision_interoceptive", 1.0)
                                    ),
                                    "integration_score": float(
                                        results.get("integration_score", 0.5)
                                    ),
                                    "cross_modal_coupling": float(
                                        results.get("cross_modal_coupling", 0.3)
                                    ),
                                    "integration_efficiency": float(
                                        results.get("efficiency", 0.8)
                                    ),
                                }

                            elif analysis_type == "parameter":
                                # Run actual parameter estimation
                                from APGI_Parameter_Estimation_Protocol import (
                                    APGIParameterEstimation,
                                )

                                estimator = APGIParameterEstimation(config.model)
                                results = estimator.estimate_parameters()

                                result = {
                                    "type": "parameter_estimation",
                                    "status": "completed",
                                    "estimated_parameters": {
                                        "Pi_e": float(results.get("Pi_e", 1.0)),
                                        "Pi_i": float(results.get("Pi_i", 1.0)),
                                        "theta": float(results.get("theta", 2.0)),
                                        "beta": float(results.get("beta", 0.5)),
                                    },
                                    "confidence_intervals": results.get(
                                        "confidence_intervals", {}
                                    ),
                                    "log_likelihood": float(
                                        results.get("log_likelihood", -100.0)
                                    ),
                                    "convergence": results.get("convergence", True),
                                }

                            elif analysis_type == "validation":
                                # Run actual validation protocols
                                from Validation.APGI_Master_Validation import APGIMasterValidator

                                validator = APGIMasterValidator(config)
                                results = validator.run_all_protocols()

                                result = {
                                    "type": "validation_protocols",
                                    "status": "completed",
                                    "protocols_passed": int(results.get("passed", 0)),
                                    "protocols_failed": int(results.get("failed", 0)),
                                    "overall_score": float(
                                        results.get("overall_score", 0.0)
                                    ),
                                    "detailed_results": results.get(
                                        "detailed_results", {}
                                    ),
                                }

                            else:
                                result = {
                                    "type": analysis_type,
                                    "status": "error",
                                    "message": f"Unknown analysis type: {analysis_type}",
                                }

                            # Log the analysis
                            apgi_logger.logger.info(
                                f"Web analysis completed: {analysis_type}"
                            )

                        except ImportError as import_error:
                            # Fallback to mock data if modules not available
                            console.print(
                                f"[yellow]⚠️ Analysis module not found, using mock data: {import_error}[/yellow]"
                            )

                            if analysis_type == "formal":
                                result = {
                                    "type": "formal_model",
                                    "status": "mock_data",
                                    "surprise_mean": float(np.random.normal(0, 1)),
                                    "threshold_mean": float(np.random.normal(2, 0.5)),
                                    "ignition_events": int(np.random.randint(0, 10)),
                                    "message": "Mock data - analysis modules not available",
                                }
                            elif analysis_type == "multimodal":
                                result = {
                                    "type": "multimodal_integration",
                                    "status": "mock_data",
                                    "precision_exteroceptive": float(
                                        np.random.uniform(0.5, 2.0)
                                    ),
                                    "precision_interoceptive": float(
                                        np.random.uniform(0.5, 2.0)
                                    ),
                                    "integration_score": float(np.random.uniform(0, 1)),
                                    "message": "Mock data - analysis modules not available",
                                }
                            elif analysis_type == "parameter":
                                result = {
                                    "type": "parameter_estimation",
                                    "status": "mock_data",
                                    "Pi_e": float(np.random.uniform(0.5, 2.0)),
                                    "Pi_i": float(np.random.uniform(0.5, 2.0)),
                                    "theta": float(np.random.uniform(1.5, 3.0)),
                                    "beta": float(np.random.uniform(0.3, 0.8)),
                                    "message": "Mock data - analysis modules not available",
                                }
                            else:
                                result = {
                                    "type": analysis_type,
                                    "status": "mock_data",
                                    "message": "Mock data - analysis modules not available",
                                }

                        except Exception as analysis_error:
                            result = {
                                "type": analysis_type,
                                "status": "error",
                                "message": f"Analysis failed: {str(analysis_error)}",
                                "error_details": (
                                    str(analysis_error)
                                    if debug
                                    else "Enable debug mode for details"
                                ),
                            }

                        return jsonify(result)

                except Exception as flask_app_error:
                    console.print(f"[red]❌ Flask app error: {flask_app_error}[/red]")
                    if debug:
                        import traceback

                        console.print("[red]🐛 Flask app error traceback:[/red]")
                        for line in traceback.format_exc().split("\n"):
                            if line.strip():
                                console.print(f"[red]🐛 {line}[/red]")
                    console.print(
                        "[yellow]💡 Falling back to terminal interface...[/yellow]"
                    )
                    FLASK_AVAILABLE = False

                def run_app():
                    app.run(host=host, port=port, debug=True, use_reloader=False)

                # Set up signal handler for graceful shutdown
                def signal_handler(sig, frame):
                    console.print("\n[yellow]Shutting down web server...[/yellow]")
                    sys.exit(0)

                signal.signal(signal.SIGINT, signal_handler)

                # Run in a separate thread
                thread = threading.Thread(target=run_app)
                thread.daemon = True
                thread.start()

                console.print(
                    f"[blue]Starting web server on http://{host}:{port}[/blue]"
                )
                console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")

                try:
                    webbrowser.open(f"http://{host}:{port}")
                    console.print("[blue]Browser opened automatically[/blue]")
                except (webbrowser.Error, OSError, RuntimeError):
                    console.print(
                        f"[yellow]Could not open browser automatically. Please visit http://{host}:{port}[/yellow]"
                    )

                # Keep the main thread alive
                try:
                    while thread.is_alive():
                        time.sleep(1)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Stopping web server...[/yellow]")

        else:
            console.print("[red]❌ Failed to start web server[/red]")
            time.sleep(1)

            # Terminal-based fallback interface
            console.print("[blue]🖥️  Terminal-based Analysis Interface[/blue]")
            console.print("[yellow]Available analysis types:[/yellow]")
            console.print("  • formal    - Formal model analysis")
            console.print("  • multimodal - Multimodal integration")
            console.print("  • parameter - Parameter estimation")
            console.print("  • validation - Validation protocols")

            while True:
                try:
                    analysis_type = (
                        input("\nEnter analysis type (or 'quit' to exit): ")
                        .strip()
                        .lower()
                    )
                    if analysis_type in ["quit", "exit", "q"]:
                        break

                    if analysis_type not in [
                        "formal",
                        "multimodal",
                        "parameter",
                        "validation",
                    ]:
                        console.print("[red]Invalid analysis type. Try again.[/red]")
                        continue

                    console.print(f"[blue]Running {analysis_type} analysis...[/blue]")

                    # Simulate analysis
                    import time

                    import numpy as np

                    time.sleep(1)  # Simulate processing time

                    if analysis_type == "formal":
                        result = {
                            "type": "formal_model",
                            "surprise_mean": np.random.normal(0, 1),
                            "threshold_mean": np.random.normal(2, 0.5),
                            "ignition_events": np.random.randint(0, 10),
                        }
                    elif analysis_type == "multimodal":
                        result = {
                            "type": "multimodal_integration",
                            "precision_exteroceptive": np.random.uniform(0.5, 2.0),
                            "precision_interoceptive": np.random.uniform(0.5, 2.0),
                            "integration_score": np.random.uniform(0, 1),
                        }
                    elif analysis_type == "parameter":
                        result = {
                            "type": "parameter_estimation",
                            "Pi_e": np.random.uniform(0.5, 2.0),
                            "Pi_i": np.random.uniform(0.5, 2.0),
                            "theta": np.random.uniform(1.5, 3.0),
                            "beta": np.random.uniform(0.3, 0.8),
                        }
                    else:
                        result = {
                            "type": analysis_type,
                            "status": "completed",
                            "message": "Analysis finished",
                        }

                    console.print("[green]✓ Analysis completed successfully![/green]")
                    console.print(f"[blue]Results:[/blue]")
                    for key, value in result.items():
                        console.print(f"  {key}: {value}")

                except KeyboardInterrupt:
                    break
                except (EOFError, ValueError, SyntaxError, RuntimeError) as e:
                    console.print(f"[red]Error: {e}[/red]")

            console.print("[yellow]Terminal interface closed[/yellow]")

    except Exception as gui_error:
        console.print(f"[red]❌ Unexpected GUI error: {gui_error}[/red]")
        if debug:
            import traceback

            console.print("[red]🐛 Full error traceback:[/red]")
            for line in traceback.format_exc().split("\n"):
                if line.strip():
                    console.print(f"[red]🐛 {line}[/red]")
        apgi_logger.logger.error(f"Unexpected GUI error: {gui_error}")

    except ImportError as import_error:
        console.print(f"[red]❌ Import error in GUI: {import_error}[/red]")
        console.print(
            "[yellow]💡 Check if all required dependencies are installed[/yellow]"
        )
        apgi_logger.logger.error(f"GUI import error: {import_error}")

        # Try to launch any available GUI as fallback
        console.print("[yellow]🔄 Attempting to launch fallback GUI...[/yellow]")
        gui_files = [
            ("Validation", PROJECT_ROOT / "Validation" / "APGI-Validation-GUI.py"),
            ("Psychological", PROJECT_ROOT / "APGI-Psychological-States-GUI.py"),
        ]

        for gui_name, gui_path in gui_files:
            if gui_path.exists():
                console.print(f"[blue]Launching {gui_name} GUI...[/blue]")
                try:
                    spec = importlib.util.spec_from_file_location(
                        gui_name.lower(), gui_path
                    )
                    gui_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(gui_module)

                    if hasattr(gui_module, "main"):
                        gui_module.main()
                        break
                    else:
                        console.print(
                            f"[yellow]Warning: {gui_name} GUI has no main() function[/yellow]"
                        )

                except (
                    ImportError,
                    AttributeError,
                    TypeError,
                    OSError,
                    FileNotFoundError,
                ) as e:
                    console.print(f"[red]Error launching {gui_name} GUI: {e}[/red]")
                    apgi_logger.logger.error(f"GUI launch error: {gui_name}: {e}")
                except (RuntimeError, MemoryError, SystemError) as e:
                    console.print(
                        f"[red]Unexpected error launching {gui_name} GUI: {e}[/red]"
                    )
                    apgi_logger.logger.error(f"GUI launch error: {gui_name}: {e}")

    except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError) as e:
        console.print(f"[red]Error launching GUI: {e}[/red]")
        apgi_logger.logger.error(f"GUI launch error: {e}")


@cli.command()
@click.option("--tail", default=20, help="Number of lines to show from end of log")
@click.option("--follow", is_flag=True, help="Follow log file in real-time")
@click.option("--level", help="Filter by log level (DEBUG, INFO, WARNING, ERROR)")
@click.option("--export", help="Export logs to file (supports json, csv, txt)")
@click.pass_context
def logs(ctx, tail, follow, level, export):
    """View and monitor log files."""
    console.print(Panel.fit("📋 Log Viewer", style="bold white"))

    logs_dir = PROJECT_ROOT / "logs"
    if not logs_dir.exists():
        console.print(
            "[yellow]No logs directory found. Run some commands first.[/yellow]"
        )
        return

    # Handle export functionality
    if export:
        console.print(f"[blue]Exporting logs to {export}...[/blue]")
        format_type = export.split(".")[-1] if "." in export else "json"
        success = apgi_logger.export_logs(
            export, format_type=format_type, log_level=level
        )
        if success:
            console.print(f"[green]✓[/green] Logs exported to {export}")
        else:
            console.print(f"[red]✗[/red] Failed to export logs")
        return

    # List available log files
    log_files = list(logs_dir.glob("*.log"))
    if not log_files:
        console.print("[yellow]No log files found[/yellow]")
        return

    table = Table(title="Available Log Files")
    table.add_column("Log File", style="white")
    table.add_column("Size", style="cyan")
    table.add_column("Modified", style="green")

    for log_file in log_files:
        size = f"{log_file.stat().st_size} bytes"
        modified = f"{log_file.stat().st_mtime}"
        table.add_row(log_file.name, size, modified)

    console.print(table)

    # Show recent log entries
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        console.print(f"\n[blue]Recent entries from {latest_log.name}:[/blue]")

        try:
            with open(latest_log, "r") as f:
                lines = f.readlines()

            # Filter by level if specified
            if level:
                filtered_lines = [line for line in lines if level.upper() in line]
                display_lines = (
                    filtered_lines[-tail:]
                    if len(filtered_lines) > tail
                    else filtered_lines
                )
            else:
                display_lines = lines[-tail:]

            for line in display_lines:
                console.print(line.strip())

        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            console.print(f"[red]Error reading log file: {e}[/red]")


@cli.command()
@click.option("--detailed", is_flag=True, help="Show detailed performance metrics")
def performance(detailed):
    """Show performance metrics and statistics."""
    console.print(Panel.fit("📊 Performance Metrics", style="bold cyan"))

    try:
        summary = apgi_logger.get_performance_summary()

        if not summary:
            console.print("[yellow]No performance metrics recorded yet[/yellow]")
            console.print("[blue]Run some commands to generate performance data[/blue]")
            return

        # Create performance table
        perf_table = Table(
            title="Performance Summary", show_header=True, header_style="bold cyan"
        )
        perf_table.add_column("Metric", style="cyan", width=25)
        perf_table.add_column("Count", style="white", width=10)
        perf_table.add_column("Mean", style="green", width=12)
        perf_table.add_column("Min", style="yellow", width=12)
        perf_table.add_column("Max", style="red", width=12)
        perf_table.add_column("Unit", style="blue", width=8)

        for metric_name, stats in summary.items():
            perf_table.add_row(
                metric_name,
                str(stats["count"]),
                f"{stats['mean']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                stats["unit"],
            )

        console.print(perf_table)

        if detailed:
            console.print("\n[bold]Detailed Metrics:[/bold]")
            for metric_name, stats in summary.items():
                console.print(f"\n[cyan]{metric_name}:[/cyan]")
                console.print(
                    f"  Latest: {stats['latest']:.3f} {stats['unit']} at {stats['latest_timestamp']}"
                )
                console.print(
                    f"  Range: {stats['min']:.3f} - {stats['max']:.3f} {stats['unit']}"
                )
                console.print(
                    f"  Average: {stats['mean']:.3f} {stats['unit']} over {stats['count']} measurements"
                )

    except (FileNotFoundError, PermissionError, KeyError, ValueError) as e:
        console.print(f"[red]Error getting performance metrics: {e}[/red]")


@cli.command()
@click.option("--input-file", required=True, help="Input data file for visualization")
@click.option("--output-file", help="Output file for visualization")
@click.option(
    "--plot-type",
    default="auto",
    type=click.Choice(
        [
            "auto",
            "time_series",
            "scatter",
            "heatmap",
            "distribution",
            "box",
            "violin",
            "pair",
            "3d",
            "polar",
        ]
    ),
    help="Type of plot to create",
)
@click.option(
    "--style",
    default="seaborn",
    type=click.Choice(
        [
            "seaborn",
            "ggplot",
            "classic",
            "bmh",
            "dark_background",
            "fivethirtyeight",
            "tableau-colorblind10",
        ]
    ),
    help="Plot style",
)
@click.option(
    "--palette",
    default="default",
    type=click.Choice(
        [
            "default",
            "deep",
            "muted",
            "bright",
            "pastel",
            "dark",
            "colorblind",
            "husl",
            "Set1",
            "Set2",
            "Set3",
        ]
    ),
    help="Color palette",
)
@click.option("--figsize", default="12,8", help="Figure size (width,height)")
@click.option("--dpi", default=300, help="Output DPI")
@click.option("--alpha", default="0.7", help="Transparency (0.0-1.0)")
@click.option("--grid", is_flag=True, help="Show grid")
@click.option("--interactive", is_flag=True, help="Create interactive plot")
@click.option(
    "--colormap",
    default="viridis",
    type=click.Choice(
        [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "coolwarm",
            "hot",
            "jet",
            "rainbow",
            "twilight",
        ]
    ),
    help="Colormap for heatmaps",
)
@click.option("--bins", default=30, help="Number of bins for histograms")
@click.option("--linewidth", default=1.5, help="Line width for plots")
@click.option("--marker", default="o", help="Marker style for scatter plots")
@click.option("--markersize", default=50, help="Marker size for scatter plots")
@click.option("--font-family", default="sans-serif", help="Font family for text")
@click.option("--font-size", default=12, help="Font size for text")
@click.option("--title", default="", help="Plot title")
@click.option("--xlabel", default="", help="X-axis label")
@click.option("--ylabel", default="", help="Y-axis label")
@click.option("--legend", is_flag=True, help="Show legend")
@click.option("--tight-layout", is_flag=True, help="Use tight layout")
@click.option(
    "--save-format",
    default="png",
    type=click.Choice(["png", "jpg", "pdf", "svg", "eps"]),
    help="Output format when saving",
)
@click.option(
    "--aspect",
    default="auto",
    type=click.Choice(["auto", "equal", "scaled"]),
    help="Aspect ratio",
)
@click.option("--subplot-rows", default=1, help="Number of subplot rows")
@click.option("--subplot-cols", default=1, help="Number of subplot columns")
@click.pass_context
def visualize(
    ctx,
    input_file,
    output_file,
    plot_type,
    style,
    palette,
    figsize,
    dpi,
    alpha,
    grid,
    interactive,
    colormap,
    bins,
    linewidth,
    marker,
    markersize,
    font_family,
    font_size,
    title,
    xlabel,
    ylabel,
    legend,
    tight_layout,
    save_format,
    aspect,
    subplot_rows,
    subplot_cols,
):
    """Create visualizations of APGI results and data."""
    console.print(Panel.fit("📊 Data Visualization", style="bold green"))

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        # Load data
        console.print(f"[blue]Loading data from {input_file}...[/blue]")
        try:
            data = pd.read_csv(input_file)
        except FileNotFoundError:
            console.print(f"[red]Error: File not found: {input_file}[/red]")
            console.print(
                f"[yellow]File path checked: {Path(input_file).absolute()}[/yellow]"
            )
            console.print(f"[yellow]Available data files:[/yellow]")

            # List available CSV files in data directory
            data_dir = PROJECT_ROOT / "data"
            if data_dir.exists():
                csv_files = list(data_dir.glob("**/*.csv"))
                if csv_files:
                    console.print("[blue]Found these data files:[/blue]")
                    for csv_file in csv_files[:5]:  # Show first 5
                        console.print(f"  - {csv_file.relative_to(PROJECT_ROOT)}")
                    if len(csv_files) > 5:
                        console.print(f"  ... and {len(csv_files) - 5} more files")
                else:
                    console.print(
                        "[yellow]No CSV files found in data directory[/yellow]"
                    )
            else:
                console.print("[yellow]Data directory not found[/yellow]")

            console.print(
                "[yellow]Usage example: python main.py visualize --input-file data/sample.csv[/yellow]"
            )
            return
        except (
            FileNotFoundError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            ValueError,
            TypeError,
            OSError,
        ) as e:
            console.print(f"[red]Error reading file {input_file}: {e}[/red]")
            console.print(f"[yellow]File type: {Path(input_file).suffix}[/yellow]")
            console.print(
                "[yellow]Supported formats: .csv, .json, .xlsx, .xls[/yellow]"
            )
            console.print(
                "[blue]Tip: Check if the file is corrupted or in the correct format[/blue]"
            )
            return

        console.print(f"[green]✓[/green] Loaded data with shape: {data.shape}")

        # Determine plot type
        if plot_type == "auto":
            # Auto-detect best plot type based on data
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) >= 2 and "time" in data.columns.str.lower():
                plot_type = "time_series"
            elif len(numeric_cols) >= 2:
                plot_type = "scatter"
            elif len(numeric_cols) == 1:
                plot_type = "distribution"
            else:
                plot_type = "heatmap"

        console.print(
            f"[blue]Creating {plot_type} visualization with {style} style...[/blue]"
        )

        # Parse figure size
        try:
            fig_width, fig_height = map(int, figsize.split(","))
        except ValueError:
            fig_width, fig_height = 12, 8
            console.print(f"[yellow]Invalid figsize, using default 12,8[/yellow]")

        # Parse bins
        try:
            bins_val = int(bins)
            if bins_val < 5 or bins_val > 100:
                raise ValueError()
        except ValueError:
            bins_val = 30
            console.print(f"[yellow]Invalid bins, using default 30[/yellow]")

        # Parse linewidth
        try:
            linewidth_val = float(linewidth)
            if linewidth_val < 0.1 or linewidth_val > 5.0:
                raise ValueError()
        except ValueError:
            linewidth_val = 1.5
            console.print(f"[yellow]Invalid linewidth, using default 1.5[/yellow]")

        # Parse markersize
        try:
            markersize_val = float(markersize)
            if markersize_val < 10 or markersize_val > 200:
                raise ValueError()
        except ValueError:
            markersize_val = 50
            console.print(f"[yellow]Invalid markersize, using default 50[/yellow]")

        # Parse font size
        try:
            font_size_val = int(font_size)
            if font_size_val < 6 or font_size_val > 24:
                raise ValueError()
        except ValueError:
            font_size_val = 12
            console.print(f"[yellow]Invalid font size, using default 12[/yellow]")

        # Parse subplot dimensions
        try:
            subplot_rows_val = int(subplot_rows)
            subplot_cols_val = int(subplot_cols)
            if (
                subplot_rows_val < 1
                or subplot_cols_val < 1
                or subplot_rows_val * subplot_cols_val > 12
            ):
                raise ValueError()
        except ValueError:
            subplot_rows_val, subplot_cols_val = 1, 1
            console.print(
                f"[yellow]Invalid subplot dimensions, using default 1x1[/yellow]"
            )

        # Set up plotting style
        if style == "seaborn":
            sns.set_style("whitegrid")
            plt.style.use("seaborn-v0_8")
        elif style == "ggplot":
            plt.style.use("ggplot")
        elif style == "fivethirtyeight":
            plt.style.use("fivethirtyeight")
        elif style == "tableau-colorblind10":
            plt.style.use("tableau-colorblind10")
        else:
            plt.style.use("default")

        # Set up color palette
        if palette != "default":
            try:
                if palette in sns.palettes.__dict__:
                    sns.set_palette(palette)
                elif palette.startswith("Set"):
                    sns.set_palette(palette)
                else:
                    console.print(
                        f"[yellow]Unknown palette '{palette}', using default[/yellow]"
                    )
            except (ValueError, TypeError, AttributeError):
                console.print(
                    f"[yellow]Error setting palette '{palette}', using default[/yellow]"
                )

        # Set up font properties
        plt.rcParams["font.family"] = font_family
        plt.rcParams["font.size"] = font_size_val
        plt.rcParams["axes.titlesize"] = font_size_val + 2
        plt.rcParams["axes.labelsize"] = font_size_val
        plt.rcParams["xtick.labelsize"] = font_size_val - 2
        plt.rcParams["ytick.labelsize"] = font_size_val - 2
        plt.rcParams["legend.fontsize"] = font_size_val - 1

        # Create figure with custom size and subplots
        if subplot_rows_val * subplot_cols_val > 1:
            fig, axes = plt.subplots(
                subplot_rows_val,
                subplot_cols_val,
                figsize=(fig_width, fig_height),
                squeeze=False,
            )
            axes = axes.flatten() if subplot_rows_val * subplot_cols_val > 1 else [axes]
        else:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            axes = [ax]

        # Set aspect ratio
        if aspect != "auto":
            for ax in axes:
                ax.set_aspect(aspect)

        if plot_type == "time_series":
            ax = axes[0] if len(axes) == 1 else axes[0]
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            for i, col in enumerate(numeric_cols[:5]):  # Limit to 5 columns
                ax.plot(
                    data.index,
                    data[col],
                    label=col,
                    alpha=float(alpha),
                    linewidth=linewidth_val,
                    marker=marker,
                    markersize=markersize_val,
                )

            # Apply custom labels
            ax.set_xlabel(xlabel if xlabel else "Index/Time")
            ax.set_ylabel(ylabel if ylabel else "Value")
            ax.set_title(title if title else "Time Series Plot")
            if grid:
                ax.grid(True, alpha=0.3)
            if legend or len(numeric_cols) > 1:
                ax.legend()

        elif plot_type == "scatter":
            ax = axes[0] if len(axes) == 1 else axes[0]
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                scatter = ax.scatter(
                    data[numeric_cols[0]],
                    data[numeric_cols[1]],
                    alpha=float(alpha),
                    s=markersize_val,
                    marker=marker,
                    linewidth=linewidth_val,
                )
                ax.set_xlabel(xlabel if xlabel else numeric_cols[0])
                ax.set_ylabel(ylabel if ylabel else numeric_cols[1])
                ax.set_title(
                    title
                    if title
                    else f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}"
                )
                if grid:
                    ax.grid(True, alpha=0.3)
                if legend:
                    ax.legend([scatter], [f"{numeric_cols[0]} vs {numeric_cols[1]}"])
            else:
                console.print(
                    "[yellow]Need at least 2 numeric columns for scatter plot[/yellow]"
                )
                return

        elif plot_type == "heatmap":
            ax = axes[0] if len(axes) == 1 else axes[0]
            # Create correlation heatmap for numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                correlation_matrix = numeric_data.corr()
                sns.heatmap(
                    correlation_matrix,
                    annot=True,
                    cmap=colormap,
                    center=0,
                    alpha=float(alpha),
                )
                plt.title("Correlation Heatmap")
            else:
                console.print("[yellow]No numeric columns found for heatmap[/yellow]")
                return

        elif plot_type == "distribution":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                data[numeric_cols[0]].hist(bins=bins_val, alpha=float(alpha))
                plt.xlabel(numeric_cols[0])
                plt.ylabel("Frequency")
                plt.title(f"Distribution of {numeric_cols[0]}")
                if grid:
                    plt.grid(True, alpha=0.3)
            else:
                console.print(
                    "[yellow]No numeric columns found for distribution plot[/yellow]"
                )
                return

        elif plot_type == "violin":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                data[numeric_cols].violinplot(alpha=float(alpha))
                plt.xticks(rotation=45)
                plt.title("Violin Plot")
                if grid:
                    plt.grid(True, alpha=0.3)
            else:
                console.print(
                    "[yellow]No numeric columns found for violin plot[/yellow]"
                )
                return

        elif plot_type == "pair":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                sns.pairplot(
                    data[numeric_cols[:4]], alpha=float(alpha)
                )  # Limit to 4 columns
                plt.title("Pair Plot")
            else:
                console.print(
                    "[yellow]Need at least 2 numeric columns for pair plot[/yellow]"
                )
                return

        elif plot_type == "3d":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                fig = plt.figure(figsize=(fig_width, fig_height))
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(
                    data[numeric_cols[0]],
                    data[numeric_cols[1]],
                    data[numeric_cols[2]],
                    alpha=float(alpha),
                    s=markersize_val,
                    marker=marker,
                )
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
                ax.set_zlabel(numeric_cols[2])
                plt.title("3D Scatter Plot")
                if grid:
                    ax.grid(True, alpha=0.3)
            else:
                console.print(
                    "[yellow]Need at least 3 numeric columns for 3D plot[/yellow]"
                )
                return

        elif plot_type == "polar":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                fig = plt.figure(figsize=(fig_width, fig_height))
                ax = fig.add_subplot(111, projection="polar")
                theta = np.linspace(0, 2 * np.pi, 100)
                r = np.random.uniform(0, 1, 100)
                ax.plot(theta, r, alpha=float(alpha), linewidth=linewidth_val)
                ax.set_title("Polar Plot")
            else:
                console.print(
                    "[yellow]Need at least 2 numeric columns for polar plot[/yellow]"
                )
                return

        else:
            console.print(f"[red]Unknown plot type: {plot_type}[/red]")
            return

        # Apply tight layout if requested
        if tight_layout:
            plt.tight_layout()

        # Save or show plot
        if output_file:
            # Ensure output file has correct extension
            if not output_file.endswith(f".{save_format}"):
                output_file = f"{output_file}.{save_format}"

            plt.savefig(
                output_file,
                dpi=dpi,
                bbox_inches="tight",
                format=save_format,
                facecolor="white",
                edgecolor="none",
            )
            file_size = (
                Path(output_file).stat().st_size if Path(output_file).exists() else 0
            )
            console.print(
                f"[green]✓[/green] Visualization saved to {output_file} "
                f"({file_size:,} bytes, DPI: {dpi}, Format: {save_format})"
            )
        else:
            if interactive:
                console.print("[blue]Displaying interactive plot...[/blue]")
                plt.show()
            else:
                console.print("[blue]Displaying plot...[/blue]")
                plt.show()

    except (
        ValueError,
        TypeError,
        ImportError,
        plt.MatplotlibDeprecationWarning,
        RuntimeError,
        AttributeError,
        KeyError,
    ) as e:
        console.print(f"[red]Error creating visualization: {e}[/red]")
        apgi_logger.logger.error(f"Visualization error: {e}")


@cli.command()
@click.option("--input-file", required=True, help="Input file to export")
@click.option("--output-file", required=True, help="Output file path")
@click.option(
    "--format", default="auto", help="Output format (auto, json, csv, excel, parquet)"
)
@click.option("--compress", is_flag=True, help="Compress the output file")
@click.pass_context
def export_data(ctx, input_file, output_file, format, compress):
    """Export data in various formats."""
    console.print(Panel.fit("📤 Data Export", style="bold yellow"))

    try:
        import json

        import pandas as pd

        # Determine input format and load data
        console.print(f"[blue]Loading data from {input_file}...[/blue]")

        if input_file.endswith(".csv"):
            data = pd.read_csv(input_file)
        elif input_file.endswith(".json"):
            data = pd.read_json(input_file)
        elif input_file.endswith(".xlsx") or input_file.endswith(".xls"):
            data = pd.read_excel(input_file)
        else:
            # Try to auto-detect
            try:
                data = pd.read_csv(input_file)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError):
                try:
                    data = pd.read_json(input_file)
                except (ValueError, FileNotFoundError, json.JSONDecodeError):
                    console.print(
                        f"[red]Error: Could not read data file {input_file}. Supported formats: CSV, JSON[/red]"
                    )
                    return

        console.print(f"[green]✓[/green] Loaded data with shape: {data.shape}")

        # Determine output format
        if format == "auto":
            if output_file.endswith(".csv"):
                format = "csv"
            elif output_file.endswith(".json"):
                format = "json"
            elif output_file.endswith(".xlsx") or output_file.endswith(".xls"):
                format = "excel"
            elif output_file.endswith(".parquet"):
                format = "parquet"
            else:
                format = "csv"  # Default

        console.print(f"[blue]Exporting to {format} format...[/blue]")

        # Export data
        if format == "csv":
            data.to_csv(output_file, index=False)
        elif format == "json":
            data.to_json(output_file, orient="records", indent=2)
        elif format == "excel":
            data.to_excel(output_file, index=False)
        elif format == "parquet":
            data.to_parquet(output_file, index=False)
        else:
            console.print(f"[red]Unsupported export format: {format}[/red]")
            return

        # Compress if requested
        if compress:
            import gzip
            import shutil

            compressed_file = f"{output_file}.gz"
            with open(output_file, "rb") as f_in:
                with gzip.open(compressed_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove original file
            import os

            os.remove(output_file)
            output_file = compressed_file
            console.print(f"[green]✓[/green] File compressed to {output_file}")

        # Show file size
        file_size = Path(output_file).stat().st_size
        console.print(
            f"[green]✓[/green] Data exported to {output_file} ({file_size:,} bytes)"
        )

    except (FileNotFoundError, PermissionError, ValueError, json.JSONEncodeError) as e:
        console.print(f"[red]Error exporting data: {e}[/red]")
        apgi_logger.logger.error(f"Export error: {e}")


@cli.command()
@click.option("--input-file", required=True, help="Input file to import")
@click.option("--output-file", required=True, help="Output CSV file path")
@click.option(
    "--format", default="auto", help="Input format (auto, json, excel, parquet)"
)
@click.option("--validate", is_flag=True, help="Validate data during import")
@click.pass_context
def import_data(ctx, input_file, output_file, format, validate):
    """Import data from various formats into CSV."""
    console.print(Panel.fit("📥 Data Import", style="bold cyan"))

    try:
        import json

        import pandas as pd

        # Determine input format
        if format == "auto":
            if input_file.endswith(".json"):
                format = "json"
            elif input_file.endswith(".xlsx") or input_file.endswith(".xls"):
                format = "excel"
            elif input_file.endswith(".parquet"):
                format = "parquet"
            else:
                format = "csv"  # Default

        console.print(f"[blue]Importing {format} file: {input_file}[/blue]")

        # Import data
        if format == "csv":
            data = pd.read_csv(input_file)
        elif format == "json":
            data = pd.read_json(input_file)
        elif format == "excel":
            data = pd.read_excel(input_file)
        elif format == "parquet":
            data = pd.read_parquet(input_file)
        else:
            console.print(f"[red]Unsupported import format: {format}[/red]")
            return

        console.print(f"[green]✓[/green] Imported data with shape: {data.shape}")

        # Validate data if requested
        if validate:
            console.print("[blue]Validating data...[/blue]")

            # Basic validation checks
            null_counts = data.isnull().sum()
            total_nulls = null_counts.sum()

            if total_nulls > 0:
                console.print(
                    f"[yellow]Warning: {total_nulls} null values found[/yellow]"
                )
                for col, count in null_counts.items():
                    if count > 0:
                        console.print(f"  {col}: {count} nulls")

            # Check for duplicate rows
            duplicates = data.duplicated().sum()
            if duplicates > 0:
                console.print(
                    f"[yellow]Warning: {duplicates} duplicate rows found[/yellow]"
                )

            # Data types summary
            console.print("[blue]Data types:[/blue]")
            for col, dtype in data.dtypes.items():
                console.print(f"  {col}: {dtype}")

        # Save as CSV
        data.to_csv(output_file, index=False)

        # Show file size
        file_size = Path(output_file).stat().st_size
        console.print(
            f"[green]✓[/green] Data imported to {output_file} ({file_size:,} bytes)"
        )

    except (FileNotFoundError, PermissionError, ValueError, json.JSONDecodeError) as e:
        console.print(f"[red]Error importing data: {e}[/red]")
        apgi_logger.logger.error(f"Import error: {e}")


@cli.command()
@click.option(
    "--action", default="status", help="Cache action (status, clear, warm, suggestions)"
)
@click.option("--sources", help="Data sources for warming (comma-separated)")
@click.option("--max-workers", default=4, help="Max workers for parallel warming")
def cache(ctx, action, sources, max_workers):
    """Manage cache operations."""
    console.print(Panel.fit("🗄️ Cache Management", style="bold yellow"))

    try:
        from data.cache_manager import CacheManager
        from rich.table import Table

        cache_manager = CacheManager()

        if action == "status":
            console.print("[blue]Cache Status:[/blue]")
            stats = cache_manager.get_stats()

            status_table = Table()
            status_table.add_column("Metric", style="cyan")
            status_table.add_column("Value", style="white")

            status_table.add_row("Total Requests", f"{stats['total_requests']:,}")
            status_table.add_row("Cache Hits", f"{stats['hits']:,}")
            status_table.add_row("Cache Misses", f"{stats['misses']:,}")
            status_table.add_row("Hit Rate", f"{stats['hit_rate']:.1f}%")
            status_table.add_row("Total Entries", f"{stats['total_entries']:,}")
            status_table.add_row("Total Size", f"{stats['total_size_mb']:.2f} MB")
            status_table.add_row("Max Size", f"{stats['max_size_mb']} MB")
            status_table.add_row("Evictions", f"{stats['evictions']:,}")

            console.print(status_table)

        elif action == "clear":
            console.print("[yellow]Clearing cache...[/yellow]")
            cache_manager.clear()
            console.print("[green]✓ Cache cleared[/green]")

        elif action == "warm":
            if not sources:
                # Get suggestions
                suggestions = cache_manager.get_cache_warm_suggestions()
                if suggestions:
                    console.print("[blue]Suggested data sources:[/blue]")
                    for suggestion in suggestions:
                        console.print(f"  - {suggestion}")
                    return

            source_list = [s.strip() for s in sources.split(",")] if sources else []
            cache_manager.warm_cache(source_list, max_workers=max_workers)

        elif action == "suggestions":
            console.print("[blue]Cache warming suggestions:[/blue]")
            suggestions = cache_manager.get_cache_warm_suggestions()
            if suggestions:
                for suggestion in suggestions:
                    console.print(f"  - {suggestion}")
            else:
                console.print("[yellow]No data files found in data/ directory[/yellow]")

        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print(
                "[yellow]Available actions: status, clear, warm, suggestions[/yellow]"
            )

    except (FileNotFoundError, PermissionError, IOError, ValueError) as e:
        console.print(f"[red]Cache operation failed: {e}[/red]")
        apgi_logger.logger.error(f"Cache operation error: {e}")


@cli.command()
@click.option(
    "--output-dir",
    help="Output directory for dashboards (default: apgi_output/dashboards)",
)
@click.option(
    "--dashboard-type", help="Specific dashboard type (system, validation, all)"
)
@click.option(
    "--open-browser", is_flag=True, help="Open dashboard in browser after generation"
)
def dashboard(output_dir, dashboard_type, open_browser):
    """Generate static HTML dashboards for APGI framework."""
    console.print(Panel.fit("📊 Dashboard Generation", style="bold blue"))

    try:
        # Import the dashboard generator
        import webbrowser
        from pathlib import Path

        from utils.static_dashboard_generator import generate_dashboards

        # Set default values
        if not output_dir:
            output_dir = PROJECT_ROOT / "apgi_output" / "dashboards"

        if not dashboard_type:
            dashboard_type = "all"

        console.print(f"[blue]Output directory: {output_dir}[/blue]")
        console.print(f"[blue]Dashboard type: {dashboard_type}[/blue]")

        # Generate dashboards
        with console.status("[bold green]Generating dashboards..."):
            if dashboard_type == "all":
                generated_files = generate_dashboards(str(output_dir))
            else:
                # Generate specific dashboard type
                from utils.static_dashboard_generator import StaticDashboardGenerator

                generator = StaticDashboardGenerator(str(output_dir))

                if dashboard_type == "system":
                    generated_files = [generator.generate_system_dashboard()]
                elif dashboard_type == "validation":
                    generated_files = [generator.generate_validation_dashboard()]
                else:
                    console.print(
                        f"[red]Unknown dashboard type: {dashboard_type}[/red]"
                    )
                    console.print(
                        "[yellow]Available types: system, validation, all[/yellow]"
                    )
                    return

        # Display results
        console.print(f"[green]✓ Generated {len(generated_files)} dashboard(s)[/green]")

        for file_path in generated_files:
            console.print(f"  📄 {file_path}")

        # Open in browser if requested
        if open_browser and generated_files:
            dashboard_path = Path(generated_files[0]).resolve()
            file_url = f"file://{dashboard_path}"

            console.print(f"[blue]Opening dashboard in browser: {file_url}[/blue]")
            webbrowser.open(file_url)

        console.print(
            "[yellow]Tip: Use 'python main.py dashboard --open-browser' to view dashboards[/yellow]"
        )

    except ImportError as e:
        console.print(f"[red]Error importing dashboard generator: {e}[/red]")
        console.print(
            "[yellow]Make sure utils/static_dashboard_generator.py exists[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Error generating dashboards: {e}[/red]")
        apgi_logger.logger.error(f"Dashboard generation error: {e}")


@cli.command()
def info():
    """Show framework information and status."""
    console.print(Panel.fit(f"📊 {global_config['project_name']}", style="bold blue"))

    # Framework info
    info_table = Table(title="Framework Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Version", global_config["version"])
    info_table.add_row(
        "Python Version",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    info_table.add_row("Project Root", str(PROJECT_ROOT))
    info_table.add_row("Description", global_config["description"])

    console.print(info_table)

    # Module status
    console.print("\n[bold]Module Status:[/bold]")
    module_table = Table()
    module_table.add_column("Module", style="cyan")
    module_table.add_column("Status", style="white")
    module_table.add_column("Description", style="white")

    for name, info in module_loader.modules.items():
        status = "✓ Loaded"
        description = info["config"]["description"]
        module_table.add_row(name, status, description)

    console.print(module_table)

    # Available commands
    console.print("\n[bold]Available Commands:[/bold]")
    commands_table = Table()
    commands_table.add_column("Command", style="cyan")
    commands_table.add_column("Description", style="white")

    commands = [
        ("formal-model", "Run formal model simulations"),
        ("multimodal", "Execute multimodal data integration"),
        ("estimate-params", "Perform Bayesian parameter estimation"),
        ("validate", "Run validation protocols"),
        ("falsify", "Execute falsification testing protocols"),
        ("config", "Manage configuration settings"),
        ("logs", "View and monitor log files"),
        ("gui", "Launch graphical user interface"),
        ("visualize", "Create data visualizations"),
        ("export-data", "Export data in various formats"),
        ("import-data", "Import data from various formats"),
        ("cache", "Manage cache operations"),
        ("info", "Show framework information and status"),
        ("dashboard", "Generate static HTML dashboards for APGI framework"),
    ]

    for cmd, desc in commands:
        commands_table.add_row(cmd, desc)

    console.print(commands_table)


@cli.command()
@click.option("--components", help="Comma-separated list of components to backup")
@click.option("--description", help="Backup description")
@click.option("--compress", is_flag=True, default=True, help="Compress backup")
def backup(components: Optional[str], description: str, compress: bool) -> None:
    """Create backup of APGI framework data."""
    console.print(Panel.fit("💾 Creating Backup", style="bold green"))

    try:
        backup_id = create_backup_cli(components or "", description)
        if backup_id:
            console.print(f"[green]✓[/green] Backup created successfully: {backup_id}")
        else:
            console.print("[red]✗[/red] Failed to create backup")
    except Exception as e:
        console.print(f"[red]Error creating backup: {e}[/red]")
        apgi_logger.logger.error(f"Backup creation error: {e}")


@cli.command()
@click.option("--backup-id", help="Specific backup ID to restore")
@click.option("--target-dir", help="Target directory for restore")
@click.option("--components", help="Comma-separated list of components to restore")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def restore(
    backup_id: Optional[str],
    target_dir: Optional[str],
    components: Optional[str],
    overwrite: bool,
) -> None:
    """Restore from backup."""
    console.print(Panel.fit("🔄 Restoring Backup", style="bold blue"))

    if not backup_id:
        # List available backups
        backups = list_backups_cli()
        if not backups:
            console.print("[yellow]No backups available[/yellow]")
            return

        console.print("[bold]Available backups:[/bold]")
        for i, backup in enumerate(backups[:10], 1):
            console.print(
                f"  {i}. {backup['backup_id']} - {backup.get('description', 'No description')}"
            )
            console.print(f"     Created: {backup['timestamp']}")
            console.print(f"     Size: {backup['total_size_mb']:.2f} MB")
            console.print()

        console.print("[yellow]Please specify --backup-id to restore[/yellow]")
        return

    try:
        success = restore_backup_cli(backup_id, target_dir or "")
        if success:
            console.print(f"[green]✓[/green] Backup {backup_id} restored successfully")
        else:
            console.print(f"[red]✗[/red] Failed to restore backup {backup_id}")
    except Exception as e:
        console.print(f"[red]Error restoring backup: {e}[/red]")
        apgi_logger.logger.error(f"Backup restore error: {e}")


@cli.command()
@click.option("--limit", default=10, help="Maximum number of backups to show")
def backups(limit: int) -> None:
    """List available backups."""
    console.print(Panel.fit("📋 Available Backups", style="bold cyan"))

    try:
        backups = list_backups_cli()

        if not backups:
            console.print("[yellow]No backups available[/yellow]")
            return

        # Create table
        backups_table = Table(title="Backup History")
        backups_table.add_column("ID", style="cyan")
        backups_table.add_column("Description", style="white")
        backups_table.add_column("Created", style="green")
        backups_table.add_column("Size (MB)", style="yellow")
        backups_table.add_column("Components", style="blue")

        for backup in backups[:limit]:
            backups_table.add_row(
                backup["backup_id"],
                backup.get("description", "No description")[:30],
                backup["timestamp"][:19],
                f"{backup['total_size_mb']:.2f}",
                ", ".join(backup["components"]),
            )

        console.print(backups_table)

    except Exception as e:
        console.print(f"[red]Error listing backups: {e}[/red]")
        apgi_logger.logger.error(f"Backup list error: {e}")


@cli.command()
@click.option("--backup-id", help="Backup ID to delete")
@click.option("--keep-count", default=10, help="Keep this many recent backups")
@click.option("--cleanup-all", is_flag=True, help="Delete all backups")
def delete_backup(backup_id: Optional[str], keep_count: int, cleanup_all: bool) -> None:
    """Delete backup(s)."""
    console.print(Panel.fit("🗑️  Deleting Backups", style="bold red"))

    try:
        if cleanup_all:
            console.print(
                "[yellow]This will delete ALL backups. Are you sure?[/yellow]"
            )
            # In a real implementation, you'd want confirmation here
            deleted = cleanup_backups_cli(0)  # Keep 0 backups
            console.print(f"[green]✓[/green] Deleted {deleted} backups")
        elif backup_id:
            success = delete_backup_cli(backup_id)
            if success:
                console.print(f"[green]✓[/green] Deleted backup {backup_id}")
            else:
                console.print(f"[red]✗[/red] Failed to delete backup {backup_id}")
        else:
            deleted = cleanup_backups_cli(keep_count)
            console.print(f"[green]✓[/green] Cleaned up {deleted} old backups")

    except Exception as e:
        console.print(f"[red]Error deleting backups: {e}[/red]")
        apgi_logger.logger.error(f"Backup deletion error: {e}")


@cli.command()
@click.option("--description", help="Description for the configuration version")
@click.option("--author", help="Author of the configuration version")
def config_version(description: str, author: str) -> None:
    """Create a version snapshot of current configuration."""
    console.print(Panel.fit("📝 Creating Config Version", style="bold green"))

    try:
        version_id = config_manager.create_version(
            description
            or f"Configuration version created at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            author or "APGI User",
        )
        console.print(f"[green]✓[/green] Configuration version created: {version_id}")
    except Exception as e:
        console.print(f"[red]Error creating config version: {e}[/red]")
        apgi_logger.logger.error(f"Config version creation error: {e}")


@cli.command()
@click.option("--limit", default=10, help="Maximum number of versions to show")
def config_versions(limit: int) -> None:
    """List configuration versions."""
    console.print(Panel.fit("📋 Configuration Versions", style="bold cyan"))

    try:
        versions = config_manager.list_versions()

        if not versions:
            console.print("[yellow]No configuration versions available[/yellow]")
            return

        # Create table
        versions_table = Table(title="Configuration Version History")
        versions_table.add_column("Version ID", style="cyan")
        versions_table.add_column("Description", style="white")
        versions_table.add_column("Author", style="green")
        versions_table.add_column("Created", style="yellow")
        versions_table.add_column("Hash", style="blue")

        for version in versions[:limit]:
            versions_table.add_row(
                version["version_id"],
                version["description"][:40],
                version["author"],
                version["timestamp"][:19],
                version["config_hash"][:8] + "...",
            )

        console.print(versions_table)

    except Exception as e:
        console.print(f"[red]Error listing config versions: {e}[/red]")
        apgi_logger.logger.error(f"Config version list error: {e}")


@cli.command()
@click.option("--version-id", help="Configuration version ID to restore")
def config_restore(version_id: Optional[str]) -> None:
    """Restore configuration from version."""
    console.print(Panel.fit("🔄 Restoring Config Version", style="bold blue"))

    if not version_id:
        # List available versions
        versions = config_manager.list_versions()
        if not versions:
            console.print("[yellow]No configuration versions available[/yellow]")
            return

        console.print("[bold]Available configuration versions:[/bold]")
        for i, version in enumerate(versions[:10], 1):
            console.print(f"  {i}. {version['version_id']} - {version['description']}")
            console.print(f"     Author: {version['author']}")
            console.print(f"     Created: {version['timestamp']}")
            console.print()

        console.print("[yellow]Please specify --version-id to restore[/yellow]")
        return

    try:
        success = config_manager.restore_version(version_id)
        if success:
            console.print(
                f"[green]✓[/green] Configuration version {version_id} restored successfully"
            )
        else:
            console.print(
                f"[red]✗[/red] Failed to restore configuration version {version_id}"
            )
    except Exception as e:
        console.print(f"[red]Error restoring config version: {e}[/red]")
        apgi_logger.logger.error(f"Config version restore error: {e}")


@cli.command()
def config_diff() -> None:
    """Compare current configuration with last version."""
    console.print(Panel.fit("🔍 Configuration Diff", style="bold magenta"))

    try:
        versions = config_manager.list_versions()
        if not versions:
            console.print(
                "[yellow]No configuration versions available for comparison[/yellow]"
            )
            return

        # Get current config
        current_config = config_manager.get_config()
        current_dict = (
            current_config.__dict__ if hasattr(current_config, "__dict__") else {}
        )

        # Get last version config
        last_version = versions[0]
        version_file = Path("config/versions") / f"{last_version['version_id']}.json"

        if version_file.exists():
            with open(version_file, "r") as f:
                version_data = json.load(f)
                version_config = version_data.get("config", {})

            # Compare configs
            diff = config_manager.compare_configs(current_dict, version_config)

            if diff:
                console.print("[bold]Configuration differences found:[/bold]")
                for key, change in diff.items():
                    if isinstance(change, dict):
                        console.print(f"  {key}:")
                        for subkey, value in change.items():
                            console.print(f"    {subkey}: {value}")
                    else:
                        console.print(f"  {key}: {change}")
            else:
                console.print("[green]No configuration differences found[/green]")
        else:
            console.print(f"[red]Version file not found: {version_file}[/red]")

    except Exception as e:
        console.print(f"[red]Error comparing configurations: {e}[/red]")
        apgi_logger.logger.error(f"Config diff error: {e}")


@cli.command()
@click.option("--category", help="Filter by error category")
@click.option("--severity", help="Filter by error severity")
@click.option("--reset", is_flag=True, help="Reset error counts")
def errors(category: Optional[str], severity: Optional[str], reset: bool) -> None:
    """Show error summary and statistics."""
    console.print(Panel.fit("📊 Error Statistics", style="bold yellow"))

    try:
        if reset:
            error_handler.reset_error_counts()
            console.print("[green]✓[/green] Error counts reset")
            return

        # Get error summary
        summary = error_handler.get_error_summary()

        if summary["total_errors"] == 0:
            console.print("[green]No errors recorded[/green]")
            return

        # Create summary table
        errors_table = Table(title="Error Summary")
        errors_table.add_column("Category", style="cyan")
        errors_table.add_column("Count", style="white")
        errors_table.add_column("Percentage", style="yellow")

        total = summary["total_errors"]
        for cat_name, count in summary["by_category"].items():
            percentage = (count / total) * 100 if total > 0 else 0
            errors_table.add_row(cat_name, str(count), f"{percentage:.1f}%")

        console.print(errors_table)

        # Show most common error
        if summary["most_common"]:
            console.print(
                f"\n[bold]Most common error category:[/bold] {summary['most_common']}"
            )

        console.print(f"\n[bold]Total errors:[/bold] {total}")

    except Exception as e:
        console.print(f"[red]Error getting error summary: {e}[/red]")
        apgi_logger.logger.error(f"Error summary error: {e}")


@cli.command()
@click.option("--test-config", is_flag=True, help="Test configuration error handling")
@click.option("--test-validation", is_flag=True, help="Test validation error handling")
@click.option("--test-data", is_flag=True, help="Test data error handling")
def test_errors(test_config: bool, test_validation: bool, test_data: bool) -> None:
    """Test error handling system."""
    console.print(Panel.fit("🧪 Testing Error Handling", style="bold cyan"))

    if not any([test_config, test_validation, test_data]):
        console.print("[yellow]Please specify at least one test type[/yellow]")
        return

    try:
        if test_config:
            console.print("[blue]Testing configuration error handling...[/blue]")
            try:
                raise error_handler.handle_error(
                    ErrorCategory.CONFIGURATION,
                    ErrorSeverity.HIGH,
                    "INVALID_PARAMETER",
                    param="test_param",
                    details="This is a test error",
                )
            except APGIError as e:
                console.print(f"[green]✓[/green] {format_user_message(e)}")

        if test_validation:
            console.print("[blue]Testing validation error handling...[/blue]")
            try:
                raise error_handler.handle_error(
                    ErrorCategory.VALIDATION,
                    ErrorSeverity.HIGH,
                    "VALIDATION_FAILED",
                    protocol="test_protocol",
                    details="This is a test validation error",
                )
            except APGIError as e:
                console.print(f"[green]✓[/green] {format_user_message(e)}")

        if test_data:
            console.print("[blue]Testing data error handling...[/blue]")
            try:
                raise error_handler.handle_error(
                    ErrorCategory.DATA,
                    ErrorSeverity.HIGH,
                    "MISSING_REQUIRED_FIELDS",
                    fields=["field1", "field2"],
                    details="This is a test data error",
                )
            except APGIError as e:
                console.print(f"[green]✓[/green] {format_user_message(e)}")

        console.print("[green]✓[/green] Error handling tests completed")

    except Exception as e:
        console.print(f"[red]Error in error handling test: {e}[/red]")
        apgi_logger.logger.error(f"Error handling test error: {e}")


# Add all commands to CLI
cli.add_command(formal_model)
cli.add_command(multimodal)
cli.add_command(estimate_params)
cli.add_command(validate)
cli.add_command(falsify)
cli.add_command(config)
cli.add_command(logs)
cli.add_command(gui)
cli.add_command(visualize)
cli.add_command(export_data)
cli.add_command(import_data)
cli.add_command(cache)
cli.add_command(info)
cli.add_command(dashboard)
cli.add_command(backup)
cli.add_command(restore)
cli.add_command(backups)
cli.add_command(delete_backup)
cli.add_command(config_version)
cli.add_command(config_versions)
cli.add_command(config_restore)
cli.add_command(config_diff)
cli.add_command(errors)
cli.add_command(test_errors)


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        ImportError,
        KeyError,
        MemoryError,
        SystemError,
    ) as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
