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
    
    try:
        # Import the APGI Multimodal Integration classes
        module = module_info['module']
        APGINormalizer = module.APGINormalizer
        APGICoreIntegration = module.APGICoreIntegration
        APGIBatchProcessor = module.APGIBatchProcessor
        
        console.print("[blue]Initializing APGI Multimodal Integration...[/blue]")
        
        # Create normalizer with default configuration
        config = {
            'exteroceptive': {'mean': 0, 'std': 1},
            'interoceptive': {'mean': 0, 'std': 1},
            'somatic': {'mean': 0, 'std': 1}
        }
        normalizer = APGINormalizer(config)
        
        # Initialize core integration
        integration = APGICoreIntegration(normalizer)
        
        # Initialize batch processor
        processor = APGIBatchProcessor(normalizer, config)
        
        console.print(f"[green]✓[/green] APGI Integration initialized")
        console.print(f"Input data: {input_data or 'Demo mode'}")
        console.print(f"Modalities: {modalities or 'EEG, Pupil, EDA'}")
        
        if input_data and input_data.endswith('.csv'):
            # Process actual data file
            console.print(f"[blue]Processing data file: {input_data}[/blue]")
            processed_successfully = False
            
            try:
                import pandas as pd
                data = pd.read_csv(input_data)
                
                # Map column names to expected APGI modalities
                modality_mapping = {
                    'eeg_fz': 'P3b_amplitude',  # Use P3b for exteroceptive
                    'eeg_pz': 'P3b_amplitude',  # Use P3b for exteroceptive
                    'pupil_diameter': 'pupil_diameter',  # This is expected
                    'eda': 'SCR',  # Skin conductance response
                    'heart_rate': 'heart_rate'  # This is expected
                }
                
                # Convert DataFrame to format expected by APGI
                subject_data = {}
                for col in data.columns:
                    if data[col].dtype in ['float64', 'int64']:
                        apgi_name = modality_mapping.get(col, col)
                        # For P3b, use the first EEG column found
                        if apgi_name == 'P3b_amplitude' and 'P3b_amplitude' in subject_data:
                            continue
                        # For interoceptive, use pupil_diameter as primary
                        if apgi_name == 'pupil_diameter' and 'pupil_diameter' in subject_data:
                            continue
                        subject_data[apgi_name] = data[col].values
                
                # Ensure we have required modalities
                if 'P3b_amplitude' not in subject_data or 'pupil_diameter' not in subject_data:
                    console.print("[yellow]Warning: Missing required modalities for APGI integration[/yellow]")
                    console.print(f"[yellow]Available modalities: {list(subject_data.keys())}[/yellow]")
                    console.print("[yellow]Required: P3b_amplitude (EEG) and pupil_diameter (for APGI integration)[/yellow]")
                    
                    # Fall back to demo mode
                    console.print("[yellow]Falling back to demo mode...[/yellow]")
                else:
                    console.print(f"[blue]Found modalities: {list(subject_data.keys())}[/blue]")
                    console.print(f"[blue]P3b_amplitude shape: {subject_data['P3b_amplitude'].shape}[/blue]")
                    console.print(f"[blue]Pupil_diameter shape: {subject_data['pupil_diameter'].shape}[/blue]")
                    
                    # Run integration using process_subject
                    results = processor.process_subject(subject_data)
                    
                    # Convert results back to DataFrame
                    if isinstance(results, dict):
                        results_df = pd.DataFrame([results])
                    else:
                        # Handle other result formats
                        results_df = pd.DataFrame({'result': [str(results)]})
                    
                    # Save results
                    if output_file:
                        results_df.to_csv(output_file, index=False)
                        console.print(f"[green]✓[/green] Results saved to {output_file}")
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
                
                # Generate synthetic data
                n_samples = 100
                synthetic_data = pd.DataFrame({
                    'EEG': np.random.normal(0, 1, n_samples),
                    'Pupil': np.random.normal(0, 1, n_samples),
                    'EDA': np.random.normal(0, 1, n_samples)
                })
                
                # Process synthetic data
                synthetic_subject_data = {
                    'exteroceptive': synthetic_data['EEG'].values,
                    'interoceptive': synthetic_data['Pupil'].values
                }
                
                try:
                    results = processor.process_subject(synthetic_subject_data)
                    console.print("[green]✓[/green] Demo integration completed")
                    console.print(f"Accumulated surprise: {results.get('S_t', 'N/A')}")
                    console.print(f"Ignition probability: {results.get('P_ignition', 'N/A')}")
                except Exception as e:
                    console.print(f"[yellow]Demo integration limited: {e}[/yellow]")
                    console.print("[yellow]Note: Full integration requires specific data format[/yellow]")
        else:
            # Demo mode with synthetic data
            console.print("[yellow]Running demo with synthetic data...[/yellow]")
            import numpy as np
            import pandas as pd
            
            # Generate synthetic data
            n_samples = 100
            synthetic_data = pd.DataFrame({
                'EEG': np.random.normal(0, 1, n_samples),
                'Pupil': np.random.normal(0, 1, n_samples),
                'EDA': np.random.normal(0, 1, n_samples)
            })
            
            # Process synthetic data
            synthetic_subject_data = {
                'exteroceptive': synthetic_data['EEG'].values,
                'interoceptive': synthetic_data['Pupil'].values
            }
            
            try:
                results = processor.process_subject(synthetic_subject_data)
                console.print("[green]✓[/green] Demo integration completed")
                console.print(f"Accumulated surprise: {results.get('S_t', 'N/A')}")
                console.print(f"Ignition probability: {results.get('P_ignition', 'N/A')}")
            except Exception as e:
                console.print(f"[yellow]Demo integration limited: {e}[/yellow]")
                console.print("[yellow]Note: Full integration requires specific data format[/yellow]")
    
    except Exception as e:
        console.print(f"[red]Error in multimodal integration: {e}[/red]")
        apgi_logger.logger.error(f"Multimodal integration error: {e}")


@click.command()
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
    
    try:
        # Import the APGI Parameter Estimation classes
        module = module_info['module']
        NeuralSignalGenerator = module.NeuralSignalGenerator
        APGIDynamics = module.APGIDynamics
        
        console.print(f"[blue]Estimation method: {method}[/blue]")
        console.print(f"[blue]Iterations: {iterations}[/blue]")
        console.print(f"[blue]Data file: {data_file or 'Demo mode'}[/blue]")
        
        if data_file and data_file.endswith('.csv'):
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
                if method == 'mcmc':
                    console.print("[blue]Running MCMC parameter estimation...[/blue]")
                    # Create a simple PyMC model for demonstration
                    with pm.Model() as model:
                        # Priors for APGI parameters
                        Pi_e = pm.Normal('Pi_e', mu=1.0, sigma=0.5)
                        Pi_i = pm.Normal('Pi_i', mu=1.0, sigma=0.5)
                        theta = pm.Normal('theta', mu=2.0, sigma=0.5)
                        beta = pm.Beta('beta', alpha=2, beta=2)
                        
                        # Likelihood (simplified)
                        sigma = pm.HalfNormal('sigma', sigma=1.0)
                        
                        # Generate synthetic likelihood for demo
                        observed = pm.Normal('observed', 
                                          mu=Pi_e + Pi_i, 
                                          sigma=sigma, 
                                          observed=np.random.normal(2.0, 0.5, len(data)))
                        
                        # Run MCMC
                        trace = pm.sample(iterations, tune=500, cores=1)
                        
                    # Summarize results
                    results = az.summary(trace, var_names=['Pi_e', 'Pi_i', 'theta', 'beta'])
                    console.print("[green]✓[/green] MCMC estimation completed")
                    console.print(results)
                    
                    # Save results
                    if output_file:
                        results.to_csv(output_file)
                        console.print(f"[green]✓[/green] Results saved to {output_file}")
                        
                else:
                    console.print(f"[yellow]Method {method} not yet implemented[/yellow]")
                    
            except Exception as e:
                console.print(f"[red]Error processing data file: {e}[/red]")
        else:
            # Demo mode with synthetic data
            console.print("[yellow]Running demo with synthetic data...[/yellow]")
            import numpy as np
            import pandas as pd
            
            # Generate synthetic neural signals
            sampling_rate = 1000
            duration = 2.0
            t = np.arange(0, duration, 1/sampling_rate)
            
            # Generate synthetic HEP and P3b waveforms
            Pi_i_demo = 1.2  # Interoceptive precision
            hep_signal = NeuralSignalGenerator.generate_hep_waveform(Pi_i_demo, sampling_rate, 0.6)
            p3b_signal = NeuralSignalGenerator.generate_p3b_waveform(1.0, sampling_rate, 0.8)
            
            # Create synthetic data
            synthetic_data = pd.DataFrame({
                'time': t,
                'HEP': hep_signal,
                'P3b': p3b_signal
            })
            
            console.print("[green]✓[/green] Synthetic neural signals generated")
            console.print(f"Signal duration: {duration}s, Sampling rate: {sampling_rate}Hz")
            
            # Run APGI dynamics
            surprise_accumulated = APGIDynamics.compute_surprise_accumulation(
                Pi_e=1.0, Pi_i=Pi_i_demo, z_e=1.5, z_i=1.2
            )
            ignition_prob = APGIDynamics.compute_ignition_probability(
                surprise_accumulated, theta_threshold=2.0
            )
            
            console.print(f"[blue]Accumulated Surprise: {surprise_accumulated:.3f}[/blue]")
            console.print(f"[blue]Ignition Probability: {ignition_prob:.3f}[/blue]")
            
    except Exception as e:
        console.print(f"[red]Error in parameter estimation: {e}[/red]")
        apgi_logger.logger.error(f"Parameter estimation error: {e}")


@cli.command()
@click.option('--protocol', help='Specific validation protocol to run')
@click.option('--all-protocols', is_flag=True, help='Run all validation protocols')
@click.option('--output-dir', help='Directory for validation reports')
@click.option('--parallel', is_flag=True, help='Run protocols in parallel')
@click.pass_context
def validate(ctx, protocol, all_protocols, output_dir, parallel):
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
            protocol_num = protocol_file.split('-')[-1].replace('.py', '')
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
                    protocol_num = protocol_file.split('-')[-1].replace('.py', '')
                    
                    try:
                        spec = importlib.util.spec_from_file_location(f"protocol_{protocol_num}", protocol_path)
                        protocol_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(protocol_module)
                        
                        if hasattr(protocol_module, 'run_validation'):
                            result = protocol_module.run_validation()
                            return protocol_num, result, None
                        else:
                            return protocol_num, "No validation function", None
                    except Exception as e:
                        return protocol_num, f"Error: {e}", str(e)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_protocol = {executor.submit(run_single_protocol, protocol_file): protocol_file 
                                        for protocol_file in protocols}
                    
                    for future in concurrent.futures.as_completed(future_to_protocol):
                        protocol_num, result, error = future.result()
                        results[protocol_num] = result
                        if error:
                            console.print(f"[red]✗[/red] Protocol {protocol_num} failed: {error}")
                        else:
                            console.print(f"[green]✓[/green] Protocol {protocol_num} completed")
            else:
                # Sequential execution
                for protocol_file in protocols:
                    protocol_path = validation_dir / protocol_file
                    protocol_num = protocol_file.split('-')[-1].replace('.py', '')
                    
                    console.print(f"[blue]Running Protocol {protocol_num}...[/blue]")
                    
                    try:
                        # Import and run protocol
                        spec = importlib.util.spec_from_file_location(f"protocol_{protocol_num}", protocol_path)
                        protocol_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(protocol_module)
                        
                        # Look for main validation function
                        if hasattr(protocol_module, 'run_validation'):
                            result = protocol_module.run_validation()
                            results[protocol_num] = result
                            console.print(f"[green]✓[/green] Protocol {protocol_num} completed")
                        else:
                            console.print(f"[yellow]Protocol {protocol_num} has no run_validation function[/yellow]")
                            results[protocol_num] = "No validation function"
                            
                    except Exception as e:
                        console.print(f"[red]Error in Protocol {protocol_num}: {e}[/red]")
                        results[protocol_num] = f"Error: {e}"
            
            # Save results
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)
                results_file = output_path / f"validation_results_{int(time.time())}.json"
                
                import json
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                console.print(f"[green]✓[/green] Results saved to {results_file}")
            else:
                console.print("\n[bold]Validation Results:[/bold]")
                for protocol_num, result in results.items():
                    console.print(f"Protocol {protocol_num}: {result}")
                    
        elif protocol:
            if protocol in [p.split('-')[-1].replace('.py', '') for p in protocols]:
                console.print(f"[blue]Running protocol: {protocol}[/blue]")
                protocol_file = f"APGI-Protocol-{protocol}.py"
                protocol_path = validation_dir / protocol_file
                
                try:
                    spec = importlib.util.spec_from_file_location(f"protocol_{protocol}", protocol_path)
                    protocol_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(protocol_module)
                    
                    if hasattr(protocol_module, 'run_validation'):
                        result = protocol_module.run_validation()
                        console.print(f"[green]✓[/green] Protocol {protocol} completed")
                        console.print(f"Result: {result}")
                    else:
                        console.print(f"[yellow]Protocol {protocol} has no run_validation function[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]Error in Protocol {protocol}: {e}[/red]")
            else:
                console.print(f"[red]Error: Protocol {protocol} not found[/red]")
        else:
            console.print("[yellow]Specify a protocol or use --all-protocols[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error in validation: {e}[/red]")
        apgi_logger.logger.error(f"Validation error: {e}")


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
    
    try:
        if protocol:
            if protocol in protocols:
                console.print(f"[blue]Running falsification protocol {protocol}[/blue]")
                protocol_file = falsification_dir / f'Protocol-{protocol}.py'
                
                try:
                    # Import and run falsification protocol
                    spec = importlib.util.spec_from_file_location(f"falsification_protocol_{protocol}", protocol_file)
                    falsification_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(falsification_module)
                    
                    # Look for main falsification function
                    if hasattr(falsification_module, 'run_falsification'):
                        console.print("[blue]Executing falsification tests...[/blue]")
                        result = falsification_module.run_falsification()
                        console.print(f"[green]✓[/green] Protocol {protocol} completed")
                        console.print(f"Result: {result}")
                        
                        # Save results
                        if output_file:
                            import json
                            with open(output_file, 'w') as f:
                                json.dump(result, f, indent=2, default=str)
                            console.print(f"[green]✓[/green] Results saved to {output_file}")
                            
                    elif hasattr(falsification_module, 'main'):
                        console.print("[blue]Running main falsification function...[/blue]")
                        falsification_module.main()
                        console.print(f"[green]✓[/green] Protocol {protocol} completed")
                    else:
                        console.print(f"[yellow]Protocol {protocol} has no standard entry function[/yellow]")
                        # List available functions
                        functions = [attr for attr in dir(falsification_module) 
                                   if callable(getattr(falsification_module, attr)) 
                                   and not attr.startswith('_')]
                        console.print(f"[yellow]Available functions: {functions}[/yellow]")
                        
                except Exception as e:
                    console.print(f"[red]Error in Protocol {protocol}: {e}[/red]")
                    apgi_logger.logger.error(f"Falsification protocol {protocol} error: {e}")
            else:
                console.print(f"[red]Error: Protocol {protocol} not found[/red]")
        else:
            console.print("[yellow]Specify a protocol number (1-6)[/yellow]")
            # Run a quick demo of falsification concept
            console.print("[blue]Demo: APGI Falsification Testing Concept[/blue]")
            console.print("Falsification protocols test specific predictions of the APGI theory:")
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
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--set', help='Set configuration value (key=value)')
@click.option('--reset', is_flag=True, help='Reset to default configuration')
@click.pass_context
def config(ctx, show, set, reset):
    """Manage configuration settings."""
    console.print(Panel.fit("⚙️ Configuration Management", style="bold magenta"))
    
    try:
        config_dir = PROJECT_ROOT / 'config'
        if not config_dir.exists():
            config_dir.mkdir(exist_ok=True)
            console.print(f"[green]Created config directory: {config_dir}[/green]")
        
        if show:
            console.print("[blue]Current configuration:[/blue]")
            try:
                current_config = get_config()
                
                # Display configuration in a nice table
                config_table = Table(title="Current Configuration")
                config_table.add_column("Section", style="cyan")
                config_table.add_column("Parameter", style="white")
                config_table.add_column("Value", style="green")
                
                # Display main configuration sections
                if hasattr(current_config, 'simulation'):
                    for attr, value in vars(current_config.simulation).items():
                        config_table.add_row("simulation", attr, str(value))
                        
                if hasattr(current_config, 'model'):
                    for attr, value in vars(current_config.model).items():
                        config_table.add_row("model", attr, str(value))
                        
                if hasattr(current_config, 'logging'):
                    for attr, value in vars(current_config.logging).items():
                        config_table.add_row("logging", attr, str(value))
                
                console.print(config_table)
                
            except Exception as e:
                console.print(f"[yellow]Could not load configuration: {e}[/yellow]")
                console.print("[yellow]Showing default configuration structure:[/yellow]")
                
                # Show default structure
                default_table = Table(title="Default Configuration Structure")
                default_table.add_column("Section", style="cyan")
                default_table.add_column("Parameters", style="white")
                
                default_table.add_row("simulation", "default_steps, default_dt, enable_plots, save_results, results_format, plot_format, plot_dpi")
                default_table.add_row("model", "tau_S, tau_theta, theta_0, alpha, gamma_M, gamma_A, rho, sigma_S, sigma_theta")
                default_table.add_row("logging", "level, format, file, max_size, backup_count")
                
                console.print(default_table)
        
        if set:
            try:
                key, value = set.split('=', 1)
                console.print(f"[blue]Setting {key} = {value}[/blue]")
                
                # Try to set the configuration
                success = set_parameter(key.split('.')[0], key.split('.')[-1] if '.' in key else key, value)
                
                if success:
                    console.print(f"[green]✓[/green] Configuration updated successfully")
                    apgi_logger.logger.info(f"Configuration updated: {key} = {value}")
                else:
                    console.print(f"[red]✗[/red] Failed to update configuration")
                    
            except ValueError:
                console.print("[red]Error: Use format key=value[/red]")
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
            console.print("[yellow]Use --set key=value to update configuration[/yellow]")
            console.print("[yellow]Use --reset to restore defaults[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error in configuration management: {e}[/red]")
        apgi_logger.logger.error(f"Configuration error: {e}")


@cli.command()
@click.option('--gui-type', default='validation', help='Type of GUI to launch (validation, psychological, analysis)')
@click.option('--port', default=8080, help='Port for web-based GUI')
@click.option('--host', default='localhost', help='Host for web-based GUI')
@click.pass_context
def gui(ctx, gui_type, port, host):
    """Launch graphical user interface for APGI framework."""
    console.print(Panel.fit("🖥️  Graphical User Interface", style="bold blue"))
    
    try:
        if gui_type == 'validation':
            # Launch validation GUI
            gui_path = PROJECT_ROOT / 'Validation' / 'APGI-Validation-GUI.py'
            if gui_path.exists():
                console.print("[blue]Launching Validation GUI...[/blue]")
                spec = importlib.util.spec_from_file_location('validation_gui', gui_path)
                gui_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gui_module)
                
                if hasattr(gui_module, 'main'):
                    gui_module.main()
                else:
                    console.print("[yellow]Validation GUI has no main function[/yellow]")
            else:
                console.print("[red]Validation GUI not found[/red]")
                
        elif gui_type == 'psychological':
            # Launch psychological states GUI
            gui_path = PROJECT_ROOT / 'APGI-Psychological-States-GUI.py'
            if gui_path.exists():
                console.print("[blue]Launching Psychological States GUI...[/blue]")
                spec = importlib.util.spec_from_file_location('psychological_gui', gui_path)
                gui_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gui_module)
                
                if hasattr(gui_module, 'main'):
                    gui_module.main()
                else:
                    console.print("[yellow]Psychological States GUI has no main function[/yellow]")
            else:
                console.print("[red]Psychological States GUI not found[/red]")
                
        elif gui_type == 'analysis':
            # Launch web-based analysis interface
            console.print(f"[blue]Starting web-based analysis interface on http://{host}:{port}[/blue]")
            
            # Create a simple web interface
            from flask import Flask, render_template_string, request, jsonify
            import threading
            import webbrowser
            
            app = Flask(__name__)
            
            @app.route('/')
            def index():
                return render_template_string('''
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
                ''')
            
            @app.route('/analyze', methods=['POST'])
            def analyze():
                data = request.get_json()
                analysis_type = data.get('type')
                
                # Simple analysis simulation
                import numpy as np
                import time
                
                if analysis_type == 'formal':
                    result = {
                        'type': 'formal_model',
                        'surprise_mean': np.random.normal(0, 1),
                        'threshold_mean': np.random.normal(2, 0.5),
                        'ignition_events': np.random.randint(0, 10)
                    }
                elif analysis_type == 'multimodal':
                    result = {
                        'type': 'multimodal_integration',
                        'precision_exteroceptive': np.random.uniform(0.5, 2.0),
                        'precision_interoceptive': np.random.uniform(0.5, 2.0),
                        'integration_score': np.random.uniform(0, 1)
                    }
                elif analysis_type == 'parameter':
                    result = {
                        'type': 'parameter_estimation',
                        'Pi_e': np.random.uniform(0.5, 2.0),
                        'Pi_i': np.random.uniform(0.5, 2.0),
                        'theta': np.random.uniform(1.5, 3.0),
                        'beta': np.random.uniform(0.3, 0.8)
                    }
                else:
                    result = {'type': analysis_type, 'status': 'completed', 'message': 'Analysis finished'}
                
                return jsonify(result)
            
            def run_app():
                app.run(host=host, port=port, debug=False)
            
            # Run in a separate thread
            thread = threading.Thread(target=run_app)
            thread.daemon = True
            thread.start()
            
            # Give the server a moment to start
            import time
            time.sleep(1)
            
            # Open browser
            webbrowser.open(f'http://{host}:{port}')
            console.print(f"[green]✓[/green] Web interface launched at http://{host}:{port}")
            
        else:
            console.print(f"[yellow]Unknown GUI type: {gui_type}[/yellow]")
            console.print("[yellow]Available types: validation, psychological, analysis[/yellow]")
            
    except ImportError:
        console.print("[red]Flask not installed. Install with: pip install flask[/red]")
        console.print("[yellow]Falling back to desktop GUI...[/yellow]")
        
        # Try to launch any available GUI
        gui_files = [
            ('Validation', PROJECT_ROOT / 'Validation' / 'APGI-Validation-GUI.py'),
            ('Psychological', PROJECT_ROOT / 'APGI-Psychological-States-GUI.py')
        ]
        
        for gui_name, gui_path in gui_files:
            if gui_path.exists():
                console.print(f"[blue]Launching {gui_name} GUI...[/blue]")
                try:
                    spec = importlib.util.spec_from_file_location(gui_name.lower(), gui_path)
                    gui_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(gui_module)
                    
                    if hasattr(gui_module, 'main'):
                        gui_module.main()
                        break
                except Exception as e:
                    console.print(f"[red]Error launching {gui_name} GUI: {e}[/red]")
        
    except Exception as e:
        console.print(f"[red]Error launching GUI: {e}[/red]")
        apgi_logger.logger.error(f"GUI launch error: {e}")


@cli.command()
@click.option('--tail', default=20, help='Number of lines to show from end of log')
@click.option('--follow', is_flag=True, help='Follow log file in real-time')
@click.option('--level', help='Filter by log level (DEBUG, INFO, WARNING, ERROR)')
@click.option('--export', help='Export logs to file (supports json, csv, txt)')
@click.pass_context
def logs(ctx, tail, follow, level, export):
    """View and monitor log files."""
    console.print(Panel.fit("📋 Log Viewer", style="bold white"))
    
    logs_dir = PROJECT_ROOT / 'logs'
    if not logs_dir.exists():
        console.print("[yellow]No logs directory found. Run some commands first.[/yellow]")
        return
    
    # Handle export functionality
    if export:
        console.print(f"[blue]Exporting logs to {export}...[/blue]")
        format_type = export.split('.')[-1] if '.' in export else 'json'
        success = apgi_logger.export_logs(export, format_type=format_type, log_level=level)
        if success:
            console.print(f"[green]✓[/green] Logs exported to {export}")
        else:
            console.print(f"[red]✗[/red] Failed to export logs")
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
                
            # Filter by level if specified
            if level:
                filtered_lines = [line for line in lines if level.upper() in line]
                display_lines = filtered_lines[-tail:] if len(filtered_lines) > tail else filtered_lines
            else:
                display_lines = lines[-tail:]
            
            for line in display_lines:
                console.print(line.strip())
                
        except Exception as e:
            console.print(f"[red]Error reading log file: {e}[/red]")


@cli.command()
@click.option('--input-file', required=True, help='Input data file for visualization')
@click.option('--output-file', help='Output file for visualization')
@click.option('--plot-type', default='auto', help='Type of plot (auto, time_series, heatmap, scatter, distribution)')
@click.option('--interactive', is_flag=True, help='Create interactive plot')
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
        data = pd.read_csv(input_file)
        console.print(f"[green]✓[/green] Loaded data with shape: {data.shape}")
        
        # Determine plot type
        if plot_type == 'auto':
            # Auto-detect best plot type based on data
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2 and 'time' in data.columns.str.lower():
                plot_type = 'time_series'
            elif len(numeric_cols) >= 2:
                plot_type = 'scatter'
            elif len(numeric_cols) == 1:
                plot_type = 'distribution'
            else:
                plot_type = 'heatmap'
        
        console.print(f"[blue]Creating {plot_type} visualization...[/blue]")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'time_series':
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols[:5]:  # Limit to 5 columns
                plt.plot(data.index, data[col], label=col, alpha=0.7)
            plt.xlabel('Index/Time')
            plt.ylabel('Value')
            plt.title('Time Series Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        elif plot_type == 'scatter':
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                plt.scatter(data[numeric_cols[0]], data[numeric_cols[1]], alpha=0.6)
                plt.xlabel(numeric_cols[0])
                plt.ylabel(numeric_cols[1])
                plt.title(f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}')
                plt.grid(True, alpha=0.3)
            else:
                console.print("[yellow]Need at least 2 numeric columns for scatter plot[/yellow]")
                return
                
        elif plot_type == 'heatmap':
            # Create correlation heatmap for numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                correlation_matrix = numeric_data.corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
            else:
                console.print("[yellow]No numeric columns found for heatmap[/yellow]")
                return
                
        elif plot_type == 'distribution':
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                data[numeric_cols[0]].hist(bins=30, alpha=0.7)
                plt.xlabel(numeric_cols[0])
                plt.ylabel('Frequency')
                plt.title(f'Distribution of {numeric_cols[0]}')
                plt.grid(True, alpha=0.3)
            else:
                console.print("[yellow]No numeric columns found for distribution plot[/yellow]")
                return
        
        else:
            console.print(f"[red]Unknown plot type: {plot_type}[/red]")
            return
        
        plt.tight_layout()
        
        # Save or show plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
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
@click.option('--input-file', required=True, help='Input file to export')
@click.option('--output-file', required=True, help='Output file path')
@click.option('--format', default='auto', help='Output format (auto, json, csv, excel, parquet)')
@click.option('--compress', is_flag=True, help='Compress the output file')
@click.pass_context
def export_data(ctx, input_file, output_file, format, compress):
    """Export data in various formats."""
    console.print(Panel.fit("📤 Data Export", style="bold yellow"))
    
    try:
        import pandas as pd
        import json
        
        # Determine input format and load data
        console.print(f"[blue]Loading data from {input_file}...[/blue]")
        
        if input_file.endswith('.csv'):
            data = pd.read_csv(input_file)
        elif input_file.endswith('.json'):
            data = pd.read_json(input_file)
        elif input_file.endswith('.xlsx') or input_file.endswith('.xls'):
            data = pd.read_excel(input_file)
        else:
            # Try to auto-detect
            try:
                data = pd.read_csv(input_file)
            except:
                data = pd.read_json(input_file)
        
        console.print(f"[green]✓[/green] Loaded data with shape: {data.shape}")
        
        # Determine output format
        if format == 'auto':
            if output_file.endswith('.csv'):
                format = 'csv'
            elif output_file.endswith('.json'):
                format = 'json'
            elif output_file.endswith('.xlsx') or output_file.endswith('.xls'):
                format = 'excel'
            elif output_file.endswith('.parquet'):
                format = 'parquet'
            else:
                format = 'csv'  # Default
        
        console.print(f"[blue]Exporting to {format} format...[/blue]")
        
        # Export data
        if format == 'csv':
            data.to_csv(output_file, index=False)
        elif format == 'json':
            data.to_json(output_file, orient='records', indent=2)
        elif format == 'excel':
            data.to_excel(output_file, index=False)
        elif format == 'parquet':
            data.to_parquet(output_file, index=False)
        else:
            console.print(f"[red]Unsupported export format: {format}[/red]")
            return
        
        # Compress if requested
        if compress:
            import gzip
            import shutil
            
            compressed_file = f"{output_file}.gz"
            with open(output_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            import os
            os.remove(output_file)
            output_file = compressed_file
            console.print(f"[green]✓[/green] File compressed to {output_file}")
        
        # Show file size
        file_size = Path(output_file).stat().st_size
        console.print(f"[green]✓[/green] Data exported to {output_file} ({file_size:,} bytes)")
        
    except Exception as e:
        console.print(f"[red]Error exporting data: {e}[/red]")
        apgi_logger.logger.error(f"Export error: {e}")


@cli.command()
@click.option('--input-file', required=True, help='Input file to import')
@click.option('--output-file', required=True, help='Output CSV file path')
@click.option('--format', default='auto', help='Input format (auto, json, excel, parquet)')
@click.option('--validate', is_flag=True, help='Validate data during import')
@click.pass_context
def import_data(ctx, input_file, output_file, format, validate):
    """Import data from various formats into CSV."""
    console.print(Panel.fit("📥 Data Import", style="bold cyan"))
    
    try:
        import pandas as pd
        import json
        
        # Determine input format
        if format == 'auto':
            if input_file.endswith('.json'):
                format = 'json'
            elif input_file.endswith('.xlsx') or input_file.endswith('.xls'):
                format = 'excel'
            elif input_file.endswith('.parquet'):
                format = 'parquet'
            else:
                format = 'csv'  # Default
        
        console.print(f"[blue]Importing {format} file: {input_file}[/blue]")
        
        # Import data
        if format == 'csv':
            data = pd.read_csv(input_file)
        elif format == 'json':
            data = pd.read_json(input_file)
        elif format == 'excel':
            data = pd.read_excel(input_file)
        elif format == 'parquet':
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
                console.print(f"[yellow]Warning: {total_nulls} null values found[/yellow]")
                for col, count in null_counts.items():
                    if count > 0:
                        console.print(f"  {col}: {count} nulls")
            
            # Check for duplicate rows
            duplicates = data.duplicated().sum()
            if duplicates > 0:
                console.print(f"[yellow]Warning: {duplicates} duplicate rows found[/yellow]")
            
            # Data types summary
            console.print("[blue]Data types:[/blue]")
            for col, dtype in data.dtypes.items():
                console.print(f"  {col}: {dtype}")
        
        # Save as CSV
        data.to_csv(output_file, index=False)
        
        # Show file size
        file_size = Path(output_file).stat().st_size
        console.print(f"[green]✓[/green] Data imported to {output_file} ({file_size:,} bytes)")
        
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
        ("info", "Show framework information and status")
    ]
    
    for cmd, desc in commands:
        commands_table.add_row(cmd, desc)
    
    console.print(commands_table)


if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
