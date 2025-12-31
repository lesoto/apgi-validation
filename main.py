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

# Initialize rich console
console = Console()

# Global configuration
CONFIG = {
    'version': '1.3.0',
    'project_name': 'APGI Theory Framework',
    'description': 'Adaptive Pattern Generation and Integration Theory Implementation'
}


class APGIModuleLoader:
    """Dynamic module loader for APGI components."""
    
    def __init__(self):
        self.modules = {}
        self._load_available_modules()
    
    def _load_available_modules(self):
        """Load all available APGI modules."""
        module_configs = {
            'formal_model': {
                'file': 'APGI-Formal-Model.py',
                'class': 'SurpriseIgnitionSystem',
                'description': 'Formal model simulations'
            },
            'multimodal': {
                'file': 'APGI-Multimodal-Integration.py',
                'class': None,  # Will detect main class
                'description': 'Multimodal data integration'
            },
            'parameter_estimation': {
                'file': 'APGI-Parameter-Estimation-Protocol.py',
                'class': None,
                'description': 'Bayesian parameter estimation'
            },
            'psychological_states': {
                'file': 'APGI-Psychological-States-CLI.py',
                'class': None,
                'description': 'Psychological states analysis'
            }
        }
        
        for name, config in module_configs.items():
            module_path = PROJECT_ROOT / config['file']
            if module_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location(name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.modules[name] = {
                        'module': module,
                        'config': config
                    }
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load {config['file']}: {e}[/yellow]")
    
    def get_module(self, name):
        """Get loaded module by name."""
        return self.modules.get(name)


# Initialize module loader
module_loader = APGIModuleLoader()


@click.group()
@click.version_option(version=CONFIG['version'], prog_name=CONFIG['project_name'])
@click.option('--config-file', help='Override configuration file path')
@click.option('--log-level', help='Override logging level')
@click.pass_context
def cli(ctx, config_file, log_level):
    """
    APGI Theory Framework - Unified Command Line Interface
    
    Comprehensive computational framework for Adaptive Pattern Generation 
    and Integration theory with psychological state dynamics modeling.
    """
    ctx.ensure_object(dict)
    ctx.obj['console'] = console
    ctx.obj['module_loader'] = module_loader
    
    # Apply command-line overrides
    if config_file:
        config_manager.config_file = Path(config_file)
        config_manager._load_config()
        apgi_logger.logger.info(f"Using custom config file: {config_file}")
    
    if log_level:
        set_parameter('logging', 'level', log_level.upper())
        apgi_logger.logger.info(f"Log level overridden to: {log_level.upper()}")
    
    # Log framework startup
    apgi_logger.logger.info(f"APGI Framework v{CONFIG['version']} started")
    log_performance("framework_startup", 0, "seconds")


@cli.command()
@click.option('--simulation-steps', default=None, type=int, help='Number of simulation steps (uses config default)')
@click.option('--dt', default=None, type=float, help='Time step size (uses config default)')
@click.option('--output-file', help='Output file for results')
@click.option('--params', help='JSON file with custom parameters')
@click.option('--plot', is_flag=True, help='Generate visualization plots')
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
    
    module_info = module_loader.get_module('formal_model')
    if not module_info:
        error_msg = "Formal model module not found"
        console.print(f"[red]Error: {error_msg}[/red]")
        apgi_logger.logger.error(error_msg)
        return
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing simulation...", total=None)
            
            # Initialize the model with configuration parameters
            SurpriseIgnitionSystem = module_info['module'].SurpriseIgnitionSystem
            
            # Convert config parameters to dict for model initialization
            model_params = {
                'tau_S': config.model.tau_S,
                'tau_theta': config.model.tau_theta,
                'theta_0': config.model.theta_0,
                'alpha': config.model.alpha,
                'gamma_M': config.model.gamma_M,
                'gamma_A': config.model.gamma_A,
                'rho': config.model.rho,
                'sigma_S': config.model.sigma_S,
                'sigma_theta': config.model.sigma_theta
            }
            
            system = SurpriseIgnitionSystem(params=model_params)
            
            progress.update(task, description="Running simulation...")
            
            # Log simulation start
            apgi_logger.log_simulation_start("formal_model", model_params)
            
            # Run simulation
            import numpy as np
            results = {
                'time': [],
                'surprise': [],
                'threshold': [],
                'ignition': []
            }
            
            for step in range(sim_steps):
                # Create dummy inputs for demonstration
                inputs = {
                    'surprise_input': np.random.normal(0, 0.1),
                    'metabolic': 1.0,
                    'arousal': 0.5
                }
                
                # Step the system
                system.step(time_step, inputs)
                
                # Store results
                results['time'].append(step * time_step)
                results['surprise'].append(system.S)
                results['threshold'].append(system.theta)
                results['ignition'].append(system.B)
            
            progress.update(task, description="Simulation complete!", completed=True)
        
        duration = time.time() - start_time
        log_performance("formal_model_simulation", duration, "seconds")
        
        # Log simulation completion
        results_summary = {
            'total_steps': sim_steps,
            'final_surprise': results['surprise'][-1],
            'final_threshold': results['threshold'][-1],
            'ignition_events': sum(1 for b in results['ignition'] if b > 0)
        }
        apgi_logger.log_simulation_end("formal_model", duration, results_summary)
        
        console.print(f"[green]✓[/green] Simulation completed: {sim_steps} steps in {duration:.2f}s")
        
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
            
            axes[0].plot(results['time'], results['surprise'])
            axes[0].set_ylabel('Surprise')
            axes[0].grid(True)
            
            axes[1].plot(results['time'], results['threshold'])
            axes[1].set_ylabel('Threshold')
            axes[1].grid(True)
            
            axes[2].plot(results['time'], results['ignition'])
            axes[2].set_ylabel('Ignition')
            axes[2].set_xlabel('Time')
            axes[2].grid(True)
            
            plt.tight_layout()
            plot_file = save_file.replace('.csv', f'_plots.{config.simulation.plot_format}') if save_file else f'simulation_plots_{int(time.time())}.{config.simulation.plot_format}'
            plt.savefig(plot_file, dpi=config.simulation.plot_dpi, bbox_inches='tight')
            console.print(f"[green]✓[/green] Plots saved to {plot_file}")
            
    except Exception as e:
        log_error(e, "formal_model_simulation", steps=sim_steps, dt=time_step)
        console.print(f"[red]Error in simulation: {e}[/red]")


@cli.command()
@click.option('--input-data', help='Input data file (CSV format)')
@click.option('--output-file', help='Output file for integration results')
@click.option('--modalities', help='Comma-separated list of modalities to integrate')
@click.pass_context
def multimodal(ctx, input_data, output_file, modalities):
    """Execute multimodal data integration."""
    console.print(Panel.fit("🔗 Multimodal Integration", style="bold green"))
    
    module_info = module_loader.get_module('multimodal')
    if not module_info:
        console.print("[red]Error: Multimodal integration module not found[/red]")
        return
    
    console.print("[blue]Multimodal integration functionality available[/blue]")
    console.print(f"Input data: {input_data or 'Not specified'}")
    console.print(f"Modalities: {modalities or 'All available'}")
    
    # Placeholder for actual integration logic
    console.print("[yellow]Note: Full integration logic requires specific data format[/yellow]")


@cli.command()
@click.option('--data-file', help='Experimental data file for parameter estimation')
@click.option('--method', default='mcmc', help='Estimation method (mcmc, map, gradient)')
@click.option('--iterations', default=1000, help='Number of iterations for MCMC')
@click.option('--output-file', help='Output file for parameter estimates')
@click.pass_context
def estimate_params(ctx, data_file, method, iterations, output_file):
    """Perform Bayesian parameter estimation."""
    console.print(Panel.fit("📊 Parameter Estimation", style="bold yellow"))
    
    module_info = module_loader.get_module('parameter_estimation')
    if not module_info:
        console.print("[red]Error: Parameter estimation module not found[/red]")
        return
    
    console.print(f"[blue]Estimation method: {method}[/blue]")
    console.print(f"[blue]Iterations: {iterations}[/blue]")
    console.print(f"[blue]Data file: {data_file or 'Not specified'}[/blue]")
    
    # Placeholder for actual estimation logic
    console.print("[yellow]Note: Full estimation requires experimental data[/yellow]")


@cli.command()
@click.option('--protocol', help='Specific validation protocol to run')
@click.option('--all-protocols', is_flag=True, help='Run all validation protocols')
@click.option('--output-dir', help='Directory for validation reports')
@click.pass_context
def validate(ctx, protocol, all_protocols, output_dir):
    """Run validation protocols."""
    console.print(Panel.fit("✅ Validation Protocols", style="bold cyan"))
    
    validation_dir = PROJECT_ROOT / 'Validation'
    if not validation_dir.exists():
        console.print("[red]Error: Validation directory not found[/red]")
        return
    
    # List available protocols
    protocols = []
    for file_path in validation_dir.glob('APGI-Protocol-*.py'):
        protocols.append(file_path.name)
    
    if protocols:
        table = Table(title="Available Validation Protocols")
        table.add_column("Protocol", style="cyan")
        table.add_column("Description", style="white")
        
        for protocol_file in protocols:
            table.add_row(protocol_file, "Validation protocol")
        
        console.print(table)
    
    if all_protocols:
        console.print("[blue]Running all validation protocols...[/blue]")
        # Placeholder for running all protocols
    elif protocol:
        console.print(f"[blue]Running protocol: {protocol}[/blue]")
        # Placeholder for specific protocol
    else:
        console.print("[yellow]Specify a protocol or use --all-protocols[/yellow]")


@cli.command()
@click.option('--protocol', type=int, help='Falsification protocol number (1-6)')
@click.option('--output-file', help='Output file for falsification results')
@click.pass_context
def falsify(ctx, protocol, output_file):
    """Execute falsification testing protocols."""
    console.print(Panel.fit("🧪 Falsification Testing", style="bold red"))
    
    falsification_dir = PROJECT_ROOT / 'Falsification-Protocols'
    if not falsification_dir.exists():
        console.print("[red]Error: Falsification protocols directory not found[/red]")
        return
    
    # List available protocols
    protocols = []
    for i in range(1, 7):
        protocol_file = falsification_dir / f'Protocol-{i}.py'
        if protocol_file.exists():
            protocols.append(i)
    
    if protocols:
        table = Table(title="Available Falsification Protocols")
        table.add_column("Protocol", style="red")
        table.add_column("Description", style="white")
        
        for protocol_num in protocols:
            table.add_row(str(protocol_num), f"Falsification Protocol {protocol_num}")
        
        console.print(table)
    
    if protocol:
        if protocol in protocols:
            console.print(f"[blue]Running falsification protocol {protocol}[/blue]")
            # Placeholder for running specific protocol
        else:
            console.print(f"[red]Error: Protocol {protocol} not found[/red]")
    else:
        console.print("[yellow]Specify a protocol number (1-6)[/yellow]")


@cli.command()
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--set', help='Set configuration value (key=value)')
@click.option('--reset', is_flag=True, help='Reset to default configuration')
@click.pass_context
def config(ctx, show, set, reset):
    """Manage configuration settings."""
    console.print(Panel.fit("⚙️ Configuration Management", style="bold magenta"))
    
    config_dir = PROJECT_ROOT / 'config'
    if not config_dir.exists():
        config_dir.mkdir(exist_ok=True)
        console.print(f"[green]Created config directory: {config_dir}[/green]")
    
    if show:
        console.print("[blue]Current configuration:[/blue]")
        # Placeholder for showing config
        console.print("Configuration system not yet implemented")
    
    if set:
        try:
            key, value = set.split('=', 1)
            console.print(f"[blue]Setting {key} = {value}[/blue]")
            # Placeholder for setting config
        except ValueError:
            console.print("[red]Error: Use format key=value[/red]")
    
    if reset:
        console.print("[blue]Resetting to default configuration[/blue]")
        # Placeholder for resetting config


@cli.command()
@click.option('--tail', default=20, help='Number of lines to show from end of log')
@click.option('--follow', is_flag=True, help='Follow log file in real-time')
@click.option('--level', help='Filter by log level (DEBUG, INFO, WARNING, ERROR)')
@click.pass_context
def logs(ctx, tail, follow, level):
    """View and monitor log files."""
    console.print(Panel.fit("📋 Log Viewer", style="bold white"))
    
    logs_dir = PROJECT_ROOT / 'logs'
    if not logs_dir.exists():
        console.print("[yellow]No logs directory found. Run some commands first.[/yellow]")
        return
    
    # List available log files
    log_files = list(logs_dir.glob('*.log'))
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
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                for line in lines[-tail:]:
                    console.print(line.strip())
        except Exception as e:
            console.print(f"[red]Error reading log file: {e}[/red]")


@cli.command()
def info():
    """Show framework information and status."""
    console.print(Panel.fit(f"📊 {CONFIG['project_name']}", style="bold blue"))
    
    # Framework info
    info_table = Table(title="Framework Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Version", CONFIG['version'])
    info_table.add_row("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    info_table.add_row("Project Root", str(PROJECT_ROOT))
    info_table.add_row("Description", CONFIG['description'])
    
    console.print(info_table)
    
    # Module status
    console.print("\n[bold]Module Status:[/bold]")
    module_table = Table()
    module_table.add_column("Module", style="cyan")
    module_table.add_column("Status", style="white")
    module_table.add_column("Description", style="white")
    
    for name, info in module_loader.modules.items():
        status = "✓ Loaded"
        description = info['config']['description']
        module_table.add_row(name, status, description)
    
    console.print(module_table)


if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
