# APGI Theory Framework - Tutorial with Real Data Examples

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Formal Model Simulation](#formal-model-simulation)
4. [Multimodal Integration](#multimodal-integration)
5. [Parameter Estimation](#parameter-estimation)
6. [Validation Protocols](#validation-protocols)
7. [Data Visualization](#data-visualization)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Required dependencies (see requirements.txt)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd apgi-theory

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### Quick Start

```bash
# Show framework information
python main.py info

# Run a simple simulation
python main.py formal-model --simulation-steps 100

# Launch GUI
python main.py gui --gui-type analysis
```

## Basic Usage

### Commands Tested Successfully

```text
✓ python main.py --help
✓ python main.py info
✗ python main.py formal-model --simulation-steps 10 --plot (CRITICAL BUG)
✓ python main.py multimodal --help
✓ python main.py config --help
✓ python main.py validate --help
✓ python main.py dashboard --help
✓ python main.py visualize --help
✓ python main.py logs --help
✓ python main.py cache --help
✓ python main.py export-data --help
✓ python main.py falsify --help
✓ python main.py estimate-params --help
✓ python main.py import-data --help
✓ python main.py performance --help
✓ python main.py backup --help
✓ python main.py backups --help
✓ python main.py restore --help
✓ python main.py delete-backup --help
✓ python main.py errors --help
✓ python main.py test-errors --help
✓ python main.py test-errors --test-config
✓ python main.py config-diff --help
✓ python main.py config-restore --help
✓ python main.py config-version --help
✓ python main.py config-versions --help
✓ python Utils-GUI.py
```

### Configuration Management

```python
# Python API
from config_manager import get_config, set_parameter

# View current configuration
config = get_config()
print(f"Default steps: {config.simulation.default_steps}")

# Modify configuration
set_parameter('simulation', 'default_steps', 1000)

# CLI
python main.py config --show
python main.py config --set simulation.default_steps=1000
```

### Logging

```python
# Python API
from logging_config import apgi_logger

# Basic logging
apgi_logger.logger.info("Starting analysis")
apgi_logger.logger.warning("Unexpected data format")

# Export logs
apgi_logger.export_logs("analysis_logs.json", format_type="json")

# CLI
python main.py logs --tail 20 --level INFO
python main.py logs --export logs.json
```

## Formal Model Simulation

### Basic Simulation

```python
# Python API
from APGI_Equations import SurpriseIgnitionSystem

# Initialize model with default parameters
system = SurpriseIgnitionSystem()

# Run simulation
import numpy as np
dt = 0.01
steps = 1000

# Define input generator function
def input_generator(t):
    return {
        'Pi_e': np.random.normal(0, 0.1),  # Exteroceptive input
        'Pi_i': 1.0,                       # Interoceptive metabolic
        'eps_e': 1.0,                       # Exteroceptive precision
        'eps_i': 0.5,                       # Interoceptive arousal
        'beta': 1.2                         # Somatic bias
    }

# Run simulation using simulate method
results = system.simulate(duration=steps*dt, dt=dt, input_generator=input_generator)

# Extract results
time = results['time']
surprise = results['S']
threshold = results['theta']
ignition = results['B']

# CLI
python main.py formal-model --simulation-steps 1000 --plot --output-file results.csv
```

### Custom Parameters

```python
# Custom model parameters
custom_params = {
    'tau_S': 0.05,      # Faster surprise dynamics
    'tau_theta': 0.1,   # Faster threshold adaptation
    'theta_0': 1.5,     # Lower initial threshold
    'alpha': 0.8,       # Stronger coupling
    'gamma_M': 1.2,     # Higher metabolic gain
    'gamma_A': 0.6,     # Lower arousal gain
    'rho': 0.05,        # Less noise
    'sigma_S': 0.05,    # Less surprise noise
    'sigma_theta': 0.05 # Less threshold noise
}

system = SurpriseIgnitionSystem(params=custom_params)

# CLI with custom parameters
python main.py formal-model --simulation-steps 1000 --params custom_params.json
```

### Real Data Input

```python
# Load real EEG data
import pandas as pd

# Assume you have a CSV file with EEG features
eeg_data = pd.read_csv('real_eeg_data.csv')

# Extract surprise-related features
# (This depends on your specific data preprocessing)
surprise_inputs = eeg_data['p300_amplitude'].values

# Run simulation with real data
system = SurpriseIgnitionSystem()

# Define input generator for real data
def real_data_input_generator(t):
    step_idx = int(t / 0.01)  # Assuming 0.01s dt
    if step_idx < len(surprise_inputs):
        return {
            'Pi_e': surprise_inputs[step_idx],
            'Pi_i': eeg_data['heart_rate'].iloc[step_idx] if step_idx < len(eeg_data) else 1.0,
            'eps_e': 1.0,
            'eps_i': eeg_data['arousal'].iloc[step_idx] if step_idx < len(eeg_data) and 'arousal' in eeg_data.columns else 0.5,
            'beta': 1.2
        }
    else:
        return {
            'Pi_e': 0.0,
            'Pi_i': 1.0,
            'eps_e': 1.0,
            'eps_i': 0.5,
            'beta': 1.2
        }

# Run simulation using simulate method
results = system.simulate(duration=len(surprise_inputs)*0.01, dt=0.01, input_generator=real_data_input_generator)

# Extract results
time = results['time']
surprise = results['S']
threshold = results['theta']
ignition = results['B']

# Save results
pd.DataFrame(results).to_csv('simulation_results.csv', index=False)
```

## Multimodal Integration

### Basic Integration

```python
# Python API
from APGI_Multimodal_Integration import APGINormalizer, APGICoreIntegration

# Sample multimodal data
data = {
    'EEG': [1.2, 0.8, 1.5, 0.9, 1.1],      # Exteroceptive
    'Pupil': [2.1, 1.9, 2.3, 2.0, 2.2],    # Interoceptive
    'EDA': [0.5, 0.7, 0.4, 0.6, 0.5]       # Somatic marker
}

# Initialize normalizer
config = {
    'exteroceptive': {'mean': 0, 'std': 1},
    'interoceptive': {'mean': 0, 'std': 1},
    'somatic': {'mean': 0, 'std': 1}
}
normalizer = APGINormalizer(config)

# Normalize data
normalized_data = normalizer.normalize(data)

# Initialize integration
integration = APGICoreIntegration(normalizer)

# Perform integration
result = integration.integrate_multimodal({
    'exteroceptive': normalized_data['EEG'][0],
    'interoceptive': normalized_data['Pupil'][0],
    'somatic': normalized_data['EDA'][0]
})

print(f"Accumulated surprise: {result['S_t']}")
print(f"Ignition probability: {result['P_ignition']}")

# CLI
python main.py multimodal --input-data multimodal_data.csv --output-file integration_results.csv
```

### Real Multimodal Data

```python
# Load real multimodal dataset
import pandas as pd

# Assume you have synchronized EEG, pupil, and EDA data
multimodal_data = pd.read_csv('real_multimodal_data.csv')

# Preprocess each modality
from scipy import stats

# Z-score normalization
def normalize_modality(data):
    return stats.zscore(data)

eeg_normalized = normalize_modality(multimodal_data['EEG'])
pupil_normalized = normalize_modality(multimodal_data['Pupil'])
eda_normalized = normalize_modality(multimodal_data['EDA'])

# Initialize APGI components
normalizer = APGINormalizer({
    'exteroceptive': {'mean': 0, 'std': 1},
    'interoceptive': {'mean': 0, 'std': 1},
    'somatic': {'mean': 0, 'std': 1}
})

integration = APGICoreIntegration(normalizer)

# Process all time points
results = []
for i in range(len(multimodal_data)):
    result = integration.integrate_multimodal({
        'exteroceptive': eeg_normalized[i],
        'interoceptive': pupil_normalized[i],
        'somatic': eda_normalized[i]
    })

    results.append({
        'time': multimodal_data['time'].iloc[i],
        'surprise_total': result['S_t'],
        'ignition_prob': result['P_ignition'],
        'Pi_e_eff': result.get('Pi_e_eff', 1.0),
        'Pi_i_eff': result.get('Pi_i_eff', 1.0)
    })

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv('multimodal_integration_results.csv', index=False)
```

### Batch Processing

```python
# Python API
from APGI_Multimodal_Integration import APGIBatchProcessor

# Load large dataset
large_dataset = pd.read_csv('large_multimodal_dataset.csv')

# Initialize batch processor
processor = APGIBatchProcessor(normalizer, config)

# Process in batches
batch_results = processor.process_batch(large_dataset)

# CLI
python main.py multimodal --input-data large_dataset.csv --output-file batch_results.csv
```

## Parameter Estimation

### Bayesian Parameter Estimation

```python
# Python API
from APGI_Parameter_Estimation-Protocol import NeuralSignalGenerator, APGIDynamics
import pymc as pm
import arviz as az

# Generate synthetic neural signals for demonstration
Pi_i_true = 1.2  # True interoceptive precision
hep_signal = NeuralSignalGenerator.generate_hep_waveform(Pi_i_true, 1000, 0.6)
p3b_signal = NeuralSignalGenerator.generate_p3b_waveform(1.0, 1000, 0.8)

# Create PyMC model for parameter estimation
with pm.Model() as param_model:
    # Priors for APGI parameters
    Pi_e = pm.Normal('Pi_e', mu=1.0, sigma=0.5)
    Pi_i = pm.Normal('Pi_i', mu=1.0, sigma=0.5)
    theta = pm.Normal('theta', mu=2.0, sigma=0.5)
    beta = pm.Beta('beta', alpha=2, beta=2)

    # Likelihood based on observed signals
    sigma = pm.HalfNormal('sigma', sigma=1.0)

    # Simplified likelihood (replace with actual signal model)
    predicted_hep = NeuralSignalGenerator.generate_hep_waveform(Pi_i, 1000, 0.6)
    observed_hep = pm.Normal('observed_hep', mu=predicted_hep, sigma=sigma, observed=hep_signal)

    # Run MCMC
    trace = pm.sample(1000, tune=500, cores=1)

# Analyze results
results = az.summary(trace, var_names=['Pi_e', 'Pi_i', 'theta', 'beta'])
print(results)

# CLI
python main.py estimate-params --data-file neural_data.csv --method mcmc --iterations 1000
```

### Real Data Parameter Estimation

```python
# Load real experimental data
experimental_data = pd.read_csv('experiment_data.csv')

# Extract relevant features
eeg_features = experimental_data[['P300_amplitude', 'N400_amplitude']]
pupil_features = experimental_data[['pupil_diameter', 'pupil_latency']]
physiological_features = experimental_data[['heart_rate', 'skin_conductance']]

# Build parameter estimation model
with pm.Model() as real_data_model:
    # Priors
    Pi_e = pm.Normal('Pi_e', mu=1.0, sigma=0.5)
    Pi_i = pm.Normal('Pi_i', mu=1.0, sigma=0.5)
    theta = pm.Normal('theta', mu=2.0, sigma=0.5)
    beta = pm.Beta('beta', alpha=2, beta=2)

    # Link parameters to observables
    # This is a simplified example - actual model would be more complex
    eeg_pred = Pi_e * theta  # Simplified relationship
    pupil_pred = Pi_i * theta * beta  # Simplified relationship

    # Likelihood
    sigma_eeg = pm.HalfNormal('sigma_eeg', sigma=1.0)
    sigma_pupil = pm.HalfNormal('sigma_pupil', sigma=1.0)

    obs_eeg = pm.Normal('obs_eeg', mu=eeg_pred, sigma=sigma_eeg, observed=eeg_features['P300_amplitude'])
    obs_pupil = pm.Normal('obs_pupil', mu=pupil_pred, sigma=sigma_pupil, observed=pupil_features['pupil_diameter'])

    # Sample
    trace = pm.sample(2000, tune=1000, cores=1)

# Save results
az.to_netcdf(trace, 'parameter_estimation_results.nc')
```

## Validation Protocols

### Running Validation Protocols

```python
# Python API
import importlib.util
from pathlib import Path

# Load and run a specific validation protocol
validation_dir = Path('Validation')
protocol_file = validation_dir / 'APGI_Protocol_1.py'

spec = importlib.util.spec_from_file_location('protocol_1', protocol_file)
protocol_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol_module)

# Run validation
results = protocol_module.run_validation()
print(f"Validation results: {results}")

# CLI
python main.py validate --protocol 1 --output-dir validation_results/
python main.py validate --all-protocols --parallel
```

### Custom Validation Protocol

```python
# Create a custom validation protocol
def run_validation():
    """Custom validation protocol for specific hypothesis."""
    import numpy as np
    from APGI_Equations import SurpriseIgnitionSystem

    # Test hypothesis: Higher precision leads to faster ignition
    precision_values = [0.5, 1.0, 1.5, 2.0]
    ignition_times = []

    for Pi in precision_values:
        system = SurpriseIgnitionSystem()
        system.theta_0 = 2.0  # Fixed threshold

        # Run simulation until ignition
        dt = 0.01
        max_steps = 10000

        for step in range(max_steps):
            inputs = {
                'surprise_input': np.random.normal(0, 1/Pi),  # Precision affects input
                'metabolic': 1.0,
                'arousal': 0.5
            }

            system.step(dt, inputs)

            if system.B > 0.5:  # Ignition threshold
                ignition_times.append(step * dt)
                break
        else:
            ignition_times.append(None)  # No ignition

    # Analyze results
    valid_results = [t for t in ignition_times if t is not None]

    return {
        'precision_values': precision_values,
        'ignition_times': ignition_times,
        'correlation': np.corrcoef(precision_values[:len(valid_results)], valid_results)[0, 1],
        'hypothesis_supported': len(valid_results) > 0 and np.corrcoef(precision_values[:len(valid_results)], valid_results)[0, 1] < 0
    }

if __name__ == '__main__':
    results = run_validation()
    print(f"Validation Results: {results}")
```

## Data Visualization

### Basic Visualization

```python
# Python API
import matplotlib.pyplot as plt
import seaborn as sns

# Load simulation results
results = pd.read_csv('simulation_results.csv')

# Create time series plot
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(results['time'], results['surprise'])
plt.ylabel('Surprise')
plt.title('APGI Simulation Results')

plt.subplot(3, 1, 2)
plt.plot(results['time'], results['threshold'])
plt.ylabel('Threshold')

plt.subplot(3, 1, 3)
plt.plot(results['time'], results['ignition'])
plt.ylabel('Ignition')
plt.xlabel('Time')

plt.tight_layout()
plt.savefig('simulation_plots.png', dpi=300, bbox_inches='tight')

# CLI
python main.py visualize --input-file simulation_results.csv --plot-type time_series --output-file plots.png
```

### Advanced Visualization

```python
# Multimodal integration visualization
integration_results = pd.read_csv('multimodal_integration_results.csv')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Surprise accumulation
axes[0, 0].plot(integration_results['time'], integration_results['surprise_total'])
axes[0, 0].set_title('Accumulated Surprise')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('S_t')

# Ignition probability
axes[0, 1].plot(integration_results['time'], integration_results['ignition_prob'])
axes[0, 1].set_title('Ignition Probability')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('P_ignition')

# Precision effectiveness
axes[1, 0].plot(integration_results['time'], integration_results['Pi_e_eff'], label='Exteroceptive')
axes[1, 0].plot(integration_results['time'], integration_results['Pi_i_eff'], label='Interoceptive')
axes[1, 0].set_title('Effective Precision')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Pi_eff')
axes[1, 0].legend()

# Correlation heatmap
numeric_cols = integration_results.select_dtypes(include=[np.number]).columns
correlation_matrix = integration_results[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation Matrix')

plt.tight_layout()
plt.savefig('multimodal_analysis.png', dpi=300, bbox_inches='tight')

# CLI
python main.py visualize --input-file multimodal_results.csv --plot-type heatmap --output-file correlation.png
```

### Interactive Visualization

```python
# Using Plotly for interactive plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create interactive subplot
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Surprise', 'Threshold', 'Ignition'),
    shared_xaxes=True
)

fig.add_trace(
    go.Scatter(x=results['time'], y=results['surprise'], name='Surprise'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x=results['time'], y=results['threshold'], name='Threshold'),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=results['time'], y=results['ignition'], name='Ignition'),
    row=3, col=1
)

fig.update_layout(height=600, title_text="Interactive APGI Simulation")
fig.write_html('interactive_simulation.html')

# CLI
python main.py visualize --input-file results.csv --interactive
```

## Advanced Usage

### Custom Model Extension

```python
# Create custom model extending SurpriseIgnitionSystem
from APGI_Equations import SurpriseIgnitionSystem

class CustomSurpriseSystem(SurpriseIgnitionSystem):
    def __init__(self, params=None):
        super().__init__(params)
        self.custom_parameter = params.get('custom_param', 1.0) if params else 1.0

    def step(self, dt, inputs):
        # Custom dynamics
        dS_dt = (-self.S + self.custom_parameter * inputs['surprise_input']) / self.tau_S
        dtheta_dt = (self.theta_0 - self.theta + self.alpha * self.S) / self.tau_theta

        # Update state
        self.S += dS_dt * dt
        self.theta += dtheta_dt * dt

        # Custom ignition rule
        self.B = 1.0 / (1.0 + np.exp(-(self.S - self.theta) * self.custom_parameter))

        return self.get_state()
```

### Performance Optimization

```python
# Optimized batch processing
from multiprocessing import Pool
import numpy as np

def process_batch_chunk(args):
    """Process a chunk of data in parallel."""
    data_chunk, params = args
    results = []

    system = SurpriseIgnitionSystem(params)

    for _, row in data_chunk.iterrows():
        inputs = {
            'surprise_input': row['surprise'],
            'metabolic': row['metabolic'],
            'arousal': row['arousal']
        }

        system.step(0.01, inputs)
        results.append(system.get_state())

    return results

# Parallel processing
def parallel_process(data, params, n_processes=4):
    chunk_size = len(data) // n_processes
    chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    with Pool(n_processes) as pool:
        results = pool.map(process_batch_chunk, [(chunk, params) for chunk in chunks])

    # Flatten results
    return [item for sublist in results for item in sublist]
```

### Integration with External Tools

```python
# Integration with Jupyter notebooks
%matplotlib inline
from IPython.display import HTML, display

# Create interactive dashboard
def create_dashboard(data):
    import ipywidgets as widgets
    from IPython.display import display

    # Create widgets
    time_slider = widgets.IntSlider(min=0, max=len(data)-1, value=0, description='Time:')

    def update_plot(time_idx):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(data['surprise'][:time_idx+1])
        plt.title('Surprise')

        plt.subplot(1, 3, 2)
        plt.plot(data['threshold'][:time_idx+1])
        plt.title('Threshold')

        plt.subplot(1, 3, 3)
        plt.plot(data['ignition'][:time_idx+1])
        plt.title('Ignition')

        plt.tight_layout()
        plt.show()

    # Link widgets
    widgets.interactive(update_plot, time_idx=time_slider)
    display(time_slider)

# Use in Jupyter
# create_dashboard(results_df)
```

## Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Check virtual environment
   which python
   pip list | grep -i apgi

   # Reinstall if needed
   pip install -e .
   ```

2. **Configuration Issues**

   ```python
   # Reset configuration
   python main.py config --reset

   # Check configuration
   python main.py config --show
   ```

3. **Memory Issues**

   ```python
   # Use batch processing for large datasets
   processor = APGIBatchProcessor(normalizer, config, batch_size=1000)

   # Clean up resources
   import gc
   gc.collect()
   ```

4. **Performance Issues**

   ```python
   # Enable parallel processing
   python main.py validate --all-protocols --parallel

   # Use progress tracking
   from rich.progress import Progress
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug info
python main.py --log-level DEBUG formal-model --simulation-steps 100
```

### Getting Help

```bash
# Check logs
python main.py logs --tail 50 --level ERROR

# Run validation
python main.py validate --all-protocols

# Check system info
python main.py info
```

### Performance Monitoring

```python
# Monitor performance
from logging_config import log_performance
import time

start_time = time.time()
# Your code here
duration = time.time() - start_time

log_performance("operation_name", duration, "seconds")
```

This tutorial provides comprehensive examples for using the APGI Theory Framework with real data. For more specific use cases, consult the API reference and validation protocols.
