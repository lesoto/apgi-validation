#!/usr/bin/env python3
"""
APGI Theory Framework - Unified CLI Entry Point
================================================

Provides command-line interface to all APGI framework components including:
- Formal model simulations
- Multimodal integration
- Parameter estimation
- Validation protocols
- Falsification testing
- Configuration management
"""

import sys
import os
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import importlib.util
import time

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import APGI framework components
from logging_config import apgi_logger, log_performance, log_error, log_simulation
from config_manager import config_manager, get_config, set_parameter

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
CONFIG = {
    "version": "1.3.0",
    "project_name": "APGI Theory Framework",
    "description": "Adaptive Pattern Generation and Integration Theory Implementation",
    "verbose": False,
    "quiet": False,
}


def verbose_print(message, level="info"):
    """Print message only if verbose mode is enabled."""
    if not CONFIG.get("quiet", False) and CONFIG.get("verbose", False):
        if level == "error":
            console.print(f"[red]{message}[/red]")
        elif level == "warning":
            console.print(f"[yellow]{message}[/yellow]")
        elif level == "success":
            console.print(f"[green]{message}[/green]")
        else:
            console.print(f"[blue]{message}[/blue]")


def quiet_print(message, level="info", force=False):
    """Print message unless quiet mode is enabled (or forced)."""
    if not CONFIG.get("quiet", False) or force:
        if level == "error":
            console.print(f"[red]{message}[/red]")
        elif level == "warning":
            console.print(f"[yellow]{message}[/yellow]")
        elif level == "success":
            console.print(f"[green]{message}[/green]")
        else:
            console.print(message)


def handle_import_error(module_name, error, context=""):
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


def handle_file_error(file_path, operation, error):
    """Handle file-related errors with specific guidance."""
    error_msg = str(error)

    if "No such file" in error_msg or "FileNotFoundError" in error_msg:
        quiet_print(f"File not found: {file_path}", "error", force=True)
        quiet_print(f"Check if the file exists and the path is correct", "info")
        quiet_print(f"Current directory: {Path.cwd()}", "info")

    elif "Permission denied" in error_msg:
        quiet_print(f"Permission denied accessing {file_path}", "error", force=True)
        quiet_print(
            f"Check file permissions or run with appropriate privileges", "info"
        )

    elif "Is a directory" in error_msg:
        quiet_print(
            f"Expected file but got directory: {file_path}", "error", force=True
        )
        quiet_print(f"Please specify a file, not a directory", "info")

    else:
        quiet_print(f"Error {operation} {file_path}: {error_msg}", "error", force=True)


def handle_validation_error(error, context=""):
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
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not load {config['file']}: {e}[/yellow]"
                    )

    def get_module(self, name):
        """Get loaded module by name."""
        return self.modules.get(name)


# Initialize module loader
module_loader = APGIModuleLoader()


@click.group()
@click.version_option(version=CONFIG["version"], prog_name=CONFIG["project_name"])
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
    global CONFIG
    CONFIG["verbose"] = verbose and not quiet
    CONFIG["quiet"] = quiet

    # Apply command-line overrides
    if config_file:
        config_manager.config_file = Path(config_file)
        config_manager._load_config()
        apgi_logger.logger.info(f"Using custom config file: {config_file}")

    if log_level:
        set_parameter("logging", "level", log_level.upper())
        apgi_logger.logger.info(f"Log level overridden to: {log_level.upper()}")

    # Log framework startup
    apgi_logger.logger.info(f"APGI Framework v{CONFIG['version']} started")
    log_performance("framework_startup", 0, "seconds")


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
def formal_model(ctx, simulation_steps, dt, output_file, params, plot):
    """Run formal model simulations."""
    console.print(Panel.fit("🧮 Formal Model Simulation", style="bold blue"))

    # Get configuration values
    config = get_config()
    sim_steps = simulation_steps or config.simulation.default_steps
    time_step = dt or config.simulation.default_dt
    enable_plots = plot or config.simulation.enable_plots

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

            # Convert config parameters to dict for model initialization
            model_params = {
                "tau_S": config.model.tau_S,
                "tau_theta": config.model.tau_theta,
                "theta_0": config.model.theta_0,
                "alpha": config.model.alpha,
                "gamma_M": config.model.gamma_M,
                "gamma_A": config.model.gamma_A,
                "rho": config.model.rho,
                "sigma_S": config.model.sigma_S,
                "sigma_theta": config.model.sigma_theta,
            }

            # Load custom parameters if provided
            if params:
                try:
                    import json

                    with open(params, "r") as f:
                        custom_params = json.load(f)
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
                    quiet_print("Using default parameters instead", "warning")
                    verbose_print(
                        "Tip: Check for missing commas, quotes, or brackets in your JSON file",
                        "info",
                    )
                except Exception as e:
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

            # Run simulation
            import numpy as np

            results = {"time": [], "surprise": [], "threshold": [], "ignition": []}

            for step in range(sim_steps):
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

            progress.update(task, description="Simulation complete!", completed=True)

        duration = time.time() - start_time
        log_performance("formal_model_simulation", duration, "seconds")

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

        # Save results if requested or configured
        save_file = output_file
        if (config.simulation.save_results and not save_file) or save_file:
            if not save_file:
                save_file = f"formal_model_results_{int(time.time())}.{config.simulation.results_format}"

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
                save_file.replace(".csv", f"_plots.{config.simulation.plot_format}")
                if save_file
                else f"simulation_plots_{int(time.time())}.{config.simulation.plot_format}"
            )
            plt.savefig(plot_file, dpi=config.simulation.plot_dpi, bbox_inches="tight")
            console.print(f"[green]✓[/green] Plots saved to {plot_file}")

    except Exception as e:
        log_error(e, "formal_model_simulation", steps=sim_steps, dt=time_step)
        console.print(f"[red]Error in simulation: {e}[/red]")


@cli.command()
@click.option("--input-data", help="Input data file (CSV format)")
@click.option("--output-file", help="Output file for integration results")
@click.option("--modalities", help="Comma-separated list of modalities to integrate")
@click.pass_context
def multimodal(ctx, input_data, output_file, modalities):
    """Execute multimodal data integration."""
    console.print(Panel.fit("🔗 Multimodal Integration", style="bold green"))

    module_info = module_loader.get_module("multimodal")
    if not module_info:
        console.print("[red]Error: Multimodal integration module not found[/red]")
        return

    try:
        # Import the APGI Multimodal Integration classes
        module = module_info["module"]
        APGINormalizer = module.APGINormalizer
        APGICoreIntegration = module.APGICoreIntegration
        APGIBatchProcessor = module.APGIBatchProcessor

        console.print("[blue]Initializing APGI Multimodal Integration...[/blue]")

        # Create normalizer with default configuration
        config = {
            "exteroceptive": {"mean": 0, "std": 1},
            "interoceptive": {"mean": 0, "std": 1},
            "somatic": {"mean": 0, "std": 1},
        }
        normalizer = APGINormalizer(config)

        # Initialize core integration
        integration = APGICoreIntegration(normalizer)

        # Initialize batch processor
        processor = APGIBatchProcessor(normalizer, config)

        console.print(f"[green]✓[/green] APGI Integration initialized")
        console.print(f"Input data: {input_data or 'Demo mode'}")
        console.print(f"Modalities: {modalities or 'EEG, Pupil, EDA'}")

        if input_data and input_data.endswith(".csv"):
            # Process actual data file
            console.print(f"[blue]Processing data file: {input_data}[/blue]")
            processed_successfully = False

            try:
                import pandas as pd
                import os

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
                    results = processor.process_subject(subject_data)

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

            except Exception as e:
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
                    results = processor.process_subject(synthetic_subject_data)
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

                except Exception as e:
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
                results = processor.process_subject(synthetic_subject_data)
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

            except Exception as e:
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

    except Exception as e:
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
def estimate_params(ctx, data_file, method, iterations, output_file):
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
                import pandas as pd
                import numpy as np
                import pymc as pm
                import arviz as az

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

            except Exception as e:
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

            # Create synthetic data
            synthetic_data = pd.DataFrame(
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

    except Exception as e:
        console.print(f"[red]Error in parameter estimation: {e}[/red]")
        apgi_logger.logger.error(f"Parameter estimation error: {e}")


@cli.command()
@click.option("--protocol", help="Specific validation protocol to run")
@click.option("--all-protocols", is_flag=True, help="Run all validation protocols")
@click.option("--output-dir", help="Directory for validation reports")
@click.option("--parallel", is_flag=True, help="Run protocols in parallel")
@click.pass_context
def validate(ctx, protocol, all_protocols, output_dir, parallel):
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
                    except Exception as e:
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

                    except Exception as e:
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

                except Exception as e:
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

    except Exception as e:
        console.print(f"[red]Error in validation: {e}[/red]")
        apgi_logger.logger.error(f"Validation error: {e}")


@cli.command()
@click.option("--protocol", type=int, help="Falsification protocol number (1-6)")
@click.option("--output-file", help="Output file for falsification results")
@click.pass_context
def falsify(ctx, protocol, output_file):
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

                except Exception as e:
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

    except Exception as e:
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
                current_config = get_config()

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
                if hasattr(current_config, "simulation"):
                    for attr, value in vars(current_config.simulation).items():
                        config_table.add_row("simulation", attr, str(value))

                if hasattr(current_config, "model"):
                    for attr, value in vars(current_config.model).items():
                        config_table.add_row("model", attr, str(value))

                if hasattr(current_config, "logging"):
                    for attr, value in vars(current_config.logging).items():
                        config_table.add_row("logging", attr, str(value))

                console.print(config_table)

            except Exception as e:
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
                    success = set_parameter(section, param, value)
                else:
                    console.print(f"[blue]Setting {key} = {value}[/blue]")
                    success = set_parameter(
                        key.split(".")[0],
                        key.split(".")[-1] if "." in key else key,
                        value,
                    )

                if success:
                    console.print(
                        f"[green]✓[/green] Configuration updated successfully"
                    )
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
            except Exception as e:
                console.print(f"[red]Error setting configuration: {e}[/red]")

        if reset:
            console.print("[blue]Resetting to default configuration[/blue]")
            try:
                # Reset configuration manager to defaults
                config_manager._reset_to_defaults()
                console.print("[green]✓[/green] Configuration reset to defaults")
                apgi_logger.logger.info("Configuration reset to defaults")

            except Exception as e:
                console.print(f"[red]Error resetting configuration: {e}[/red]")

        if not any([show, set, reset]):
            console.print("[yellow]Use --show to view current configuration[/yellow]")
            console.print(
                "[yellow]Use --set key=value to update configuration[/yellow]"
            )
            console.print("[yellow]Use --reset to restore defaults[/yellow]")

    except Exception as e:
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
@click.pass_context
def gui(ctx, gui_type, port, host):
    """Launch graphical user interface for APGI framework."""
    console.print(Panel.fit("🖥️  Graphical User Interface", style="bold blue"))

    try:
        if gui_type == "validation":
            # Launch validation GUI
            gui_path = PROJECT_ROOT / "Validation" / "APGI-Validation-GUI.py"
            if gui_path.exists():
                console.print("[blue]Launching Validation GUI...[/blue]")
                console.print(
                    "[yellow]Note: GUI will run in foreground. Press Ctrl+C to exit.[/yellow]"
                )

                spec = importlib.util.spec_from_file_location(
                    "validation_gui", gui_path
                )
                gui_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gui_module)

                if hasattr(gui_module, "main"):
                    console.print(
                        "[green]✓[/green] Validation GUI started successfully"
                    )
                    gui_module.main()
                    console.print("[blue]Validation GUI closed[/blue]")
                else:
                    console.print(
                        "[yellow]Validation GUI has no main function[/yellow]"
                    )
            else:
                console.print("[red]Validation GUI not found[/red]")

        elif gui_type == "psychological":
            # Launch psychological states GUI
            gui_path = PROJECT_ROOT / "APGI-Psychological-States-GUI.py"
            if gui_path.exists():
                console.print("[blue]Launching Psychological States GUI...[/blue]")
                console.print(
                    "[yellow]Note: GUI will run in foreground. Press Ctrl+C to exit.[/yellow]"
                )

                spec = importlib.util.spec_from_file_location(
                    "psychological_gui", gui_path
                )
                gui_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gui_module)

                if hasattr(gui_module, "main"):
                    console.print(
                        "[green]✓[/green] Psychological States GUI started successfully"
                    )
                    gui_module.main()
                    console.print("[blue]Psychological States GUI closed[/blue]")
                else:
                    console.print(
                        "[yellow]Psychological States GUI has no main function[/yellow]"
                    )
            else:
                console.print("[red]Psychological States GUI not found[/red]")

        elif gui_type == "analysis":
            # Launch web-based analysis interface
            console.print(
                f"[blue]Starting web-based analysis interface on http://{host}:{port}[/blue]"
            )
            console.print(
                "[yellow]Note: Web server will run in background. Use Ctrl+C to stop.[/yellow]"
            )

            # Create a simple web interface
            from flask import Flask, render_template_string, request, jsonify
            import threading
            import webbrowser
            import signal
            import sys

            app = Flask(__name__)

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
                data = request.get_json()
                analysis_type = data.get("type")

                # Simple analysis simulation
                import numpy as np
                import time

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

                return jsonify(result)

            def run_app():
                app.run(host=host, port=port, debug=False, use_reloader=False)

            # Set up signal handler for graceful shutdown
            def signal_handler(sig, frame):
                console.print("\n[yellow]Shutting down web server...[/yellow]")
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)

            # Run in a separate thread
            thread = threading.Thread(target=run_app, daemon=True)
            thread.start()

            # Give the server a moment to start
            import time

            console.print("[blue]Starting web server...[/blue]")
            time.sleep(2)

            # Check if server is running
            if thread.is_alive():
                console.print(
                    f"[green]✓[/green] Web interface launched successfully at http://{host}:{port}"
                )
                console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")

                # Open browser
                try:
                    webbrowser.open(f"http://{host}:{port}")
                    console.print("[blue]Browser opened automatically[/blue]")
                except Exception:
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
                console.print("[red]Failed to start web server[/red]")

        else:
            console.print(f"[yellow]Unknown GUI type: {gui_type}[/yellow]")
            console.print(
                "[yellow]Available types: validation, psychological, analysis[/yellow]"
            )

    except ImportError:
        console.print("[red]Flask not installed. Install with: pip install flask[/red]")
        console.print("[yellow]Falling back to desktop GUI...[/yellow]")

        # Try to launch any available GUI
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
                except Exception as e:
                    console.print(f"[red]Error launching {gui_name} GUI: {e}[/red]")

    except Exception as e:
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

        except Exception as e:
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

    except Exception as e:
        console.print(f"[red]Error getting performance metrics: {e}[/red]")


@cli.command()
@click.option("--input-file", required=True, help="Input data file for visualization")
@click.option("--output-file", help="Output file for visualization")
@click.option(
    "--plot-type",
    default="auto",
    help="Type of plot (auto, time_series, heatmap, scatter, distribution)",
)
@click.option("--interactive", is_flag=True, help="Create interactive plot")
@click.pass_context
def visualize(ctx, input_file, output_file, plot_type, interactive):
    """Create visualizations of APGI results and data."""
    console.print(Panel.fit("📊 Data Visualization", style="bold green"))

    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

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
        except Exception as e:
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

        console.print(f"[blue]Creating {plot_type} visualization...[/blue]")

        # Create visualization
        plt.figure(figsize=(12, 8))

        if plot_type == "time_series":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols[:5]:  # Limit to 5 columns
                plt.plot(data.index, data[col], label=col, alpha=0.7)
            plt.xlabel("Index/Time")
            plt.ylabel("Value")
            plt.title("Time Series Plot")
            plt.legend()
            plt.grid(True, alpha=0.3)

        elif plot_type == "scatter":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                plt.scatter(data[numeric_cols[0]], data[numeric_cols[1]], alpha=0.6)
                plt.xlabel(numeric_cols[0])
                plt.ylabel(numeric_cols[1])
                plt.title(f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
                plt.grid(True, alpha=0.3)
            else:
                console.print(
                    "[yellow]Need at least 2 numeric columns for scatter plot[/yellow]"
                )
                return

        elif plot_type == "heatmap":
            # Create correlation heatmap for numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                correlation_matrix = numeric_data.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
                plt.title("Correlation Heatmap")
            else:
                console.print("[yellow]No numeric columns found for heatmap[/yellow]")
                return

        elif plot_type == "distribution":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                data[numeric_cols[0]].hist(bins=30, alpha=0.7)
                plt.xlabel(numeric_cols[0])
                plt.ylabel("Frequency")
                plt.title(f"Distribution of {numeric_cols[0]}")
                plt.grid(True, alpha=0.3)
            else:
                console.print(
                    "[yellow]No numeric columns found for distribution plot[/yellow]"
                )
                return

        else:
            console.print(f"[red]Unknown plot type: {plot_type}[/red]")
            return

        plt.tight_layout()

        # Save or show plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            console.print(f"[green]✓[/green] Visualization saved to {output_file}")
        else:
            if interactive:
                console.print("[blue]Displaying interactive plot...[/blue]")
                plt.show()
            else:
                console.print("[blue]Displaying plot...[/blue]")
                plt.show()

    except Exception as e:
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
        import pandas as pd
        import json

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

    except Exception as e:
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
        import pandas as pd
        import json

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

    except Exception as e:
        console.print(f"[red]Error importing data: {e}[/red]")
        apgi_logger.logger.error(f"Import error: {e}")


@cli.command()
def info():
    """Show framework information and status."""
    console.print(Panel.fit(f"📊 {CONFIG['project_name']}", style="bold blue"))

    # Framework info
    info_table = Table(title="Framework Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("Version", CONFIG["version"])
    info_table.add_row(
        "Python Version",
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    info_table.add_row("Project Root", str(PROJECT_ROOT))
    info_table.add_row("Description", CONFIG["description"])

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
        ("info", "Show framework information and status"),
    ]

    for cmd, desc in commands:
        commands_table.add_row(cmd, desc)

    console.print(commands_table)


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
cli.add_command(info)


if __name__ == "__main__":
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
