# APGI Theory Framework - API Reference

## Overview

The APGI Framework provides a comprehensive computational framework for modeling conscious access, multimodal integration, and psychological state dynamics.

## Documentation Structure

- **Core Components**: Configuration, logging, formal models
- **Error Handling**: Standardized error management system
- **Falsification Protocols**: Comprehensive testing suite
- **Validation System**: Master validation framework

## Core Components

### 1. Configuration Management

#### `config_manager`

The configuration management system provides centralized control over all framework parameters.

```python
from config_manager import config_manager, get_config, set_parameter

# Get current configuration
config = get_config()

# Set a parameter
set_parameter('simulation', 'default_steps', 1000)

# Access specific sections
simulation_config = config.simulation
model_config = config.model
```

**Key Classes:**

- `ConfigManager`: Main configuration manager
- `Config`: Configuration data container

**Configuration Sections:**

- `simulation`: Simulation parameters (steps, dt, plots, etc.)
- `model`: Model parameters (tau_S, tau_theta, alpha, etc.)
- `logging`: Logging configuration (level, format, file, etc.)

### 2. Error Handling System

#### `utils.error_handling`

Comprehensive error handling with standardized messages and exception types.

```python
from utils.error_handling import (
    APGIError,
    ValidationError,
    ConfigurationError,
    ProtocolError,
    handle_error,
    safe_execute
)

# Raise standardized errors
raise ValidationError(
    message="Invalid parameter value",
    data_field="threshold",
    context={"value": 1.5, "valid_range": [0.0, 1.0]},
    suggestion="Use a value between 0.0 and 1.0"
)

# Safe execution
result = safe_execute(
    func=risky_function,
    error_message="Operation failed",
    default_return=None,
    logger=logger
)
```

**Key Classes:**

- `APGIError`: Base exception class
- `ValidationError`: Data validation errors
- `ConfigurationError`: Configuration errors
- `ProtocolError`: Protocol execution errors
- `DataError`: Data loading/processing errors

**Documentation:** See [Error Handling Reference](Error-Handling-Reference.md)

### 3. Falsification Protocols

#### `utils.falsification_protocols`

Comprehensive suite of protocols for testing APGI theory predictions.

```python
import importlib.util
from pathlib import Path

# Load the package
spec = importlib.util.spec_from_file_location(
    "falsification_protocols",
    Path("Falsification/__init__.py")
)
falsification_protocols = importlib.util.module_from_spec(spec)
spec.loader.exec_module(falsification_protocols)

# Import specific components
from falsification_protocols import (
    HierarchicalGenerativeModel,
    IowaGamblingTaskEnvironment,
    APGIActiveInferenceAgent,
    run_falsification_protocol_1
)

# Run protocols
results = run_falsification_protocol_1()
```

**Available Protocols:**

1. **Protocol 1**: Active Inference Agent Testing
2. **Protocol 2**: Iowa Gambling Task Environment
3. **Protocol 3**: Agent Comparison Experiment
4. **Protocol 4**: Phase Transition Analysis
5. **Protocol 5**: Evolutionary APGI Emergence
6. **Protocol 6**: Network Comparison Experiment

**Documentation:** See [Falsification Protocols Reference](Falsification-Protocols-Reference.md)

### 4. Logging System

#### `apgi_logger`

Advanced logging system with performance tracking and export capabilities.

```python
from logging_config import apgi_logger

# Basic logging
apgi_logger.logger.info("Information message")
apgi_logger.logger.warning("Warning message")
apgi_logger.logger.error("Error message")

# Performance logging
from logging_config import log_performance, log_error, log_simulation
log_performance("operation_name", duration, "seconds")
log_error(exception, "context", additional_info)
log_simulation("model_type", parameters)

# Export logs
apgi_logger.export_logs("output.json", format_type="json", log_level="INFO")
```

**Key Classes:**

- `APGILogger`: Main logger class
- `PerformanceTracker`: Performance metrics tracking

**Log Levels:**

- DEBUG: Detailed debugging information
- INFO: General information messages
- WARNING: Warning messages
- ERROR: Error messages

### 5. Formal Model

#### `SurpriseIgnitionSystem`

Core formal model for surprise accumulation and ignition dynamics.

```python
from APGI-Formal-Model import SurpriseIgnitionSystem

# Initialize model
system = SurpriseIgnitionSystem(params={
    'tau_S': 0.1,      # Surprise time constant
    'tau_theta': 0.2,  # Threshold time constant
    'theta_0': 2.0,     # Initial threshold
    'alpha': 0.5,       # Coupling strength
    'gamma_M': 1.0,     # Metabolic gain
    'gamma_A': 0.8,     # Arousal gain
    'rho': 0.1,         # Noise strength
    'sigma_S': 0.1,     # Surprise noise
    'sigma_theta': 0.1  # Threshold noise
})

# Step the simulation
inputs = {
    'surprise_input': 1.5,
    'metabolic': 1.0,
    'arousal': 0.5
}
system.step(dt=0.01, inputs=inputs)

# Access state variables
surprise = system.S      # Current surprise
threshold = system.theta # Current threshold
ignition = system.B      # Ignition state
```

**Key Methods:**

- `step(dt, inputs)`: Advance simulation by one time step
- `reset()`: Reset model to initial conditions
- `get_state()`: Get current state dictionary

**State Variables:**

- `S`: Surprise accumulation
- `theta`: Ignition threshold
- `B`: Ignition state (binary)

### 4. Multimodal Integration

#### `APGINormalizer`

Data normalization for multimodal integration.

```python
from APGI-Multimodal-Integration import APGINormalizer

# Initialize normalizer
config = {
    'exteroceptive': {'mean': 0, 'std': 1},
    'interoceptive': {'mean': 0, 'std': 1},
    'somatic': {'mean': 0, 'std': 1}
}
normalizer = APGINormalizer(config)

# Normalize data
data = {
    'EEG': [1.2, 0.8, 1.5],
    'Pupil': [2.1, 1.9, 2.3],
    'EDA': [0.5, 0.7, 0.4]
}
normalized_data = normalizer.normalize(data)
```

#### `APGICoreIntegration`

Core precision-weighted integration algorithms.

```python
from APGI-Multimodal-Integration import APGICoreIntegration

# Initialize integration
integration = APGICoreIntegration(normalizer)

# Perform integration
result = integration.integrate_multimodal({
    'exteroceptive': z_e,
    'interoceptive': z_i,
    'somatic': M_ca
})

# Access results
surprise_total = result['S_t']
ignition_prob = result['P_ignition']
```

#### `APGIBatchProcessor`

Batch processing for large datasets.

```python
from APGI-Multimodal-Integration import APGIBatchProcessor

# Initialize processor
processor = APGIBatchProcessor(normalizer, config)

# Process batch data
results = processor.process_batch(data_frame)
```

### 5. Parameter Estimation

#### `NeuralSignalGenerator`

Biophysically realistic neural signal synthesis.

```python
from APGI-Parameter-Estimation-Protocol import NeuralSignalGenerator

# Generate HEP waveform
hep_signal = NeuralSignalGenerator.generate_hep_waveform(
    Pi_i=1.2,           # Interoceptive precision
    sampling_rate=1000,
    duration=0.6
)

# Generate P3b waveform
p3b_signal = NeuralSignalGenerator.generate_p3b_waveform(
    amplitude_factor=1.0,
    sampling_rate=1000,
    duration=0.8
)
```

#### `APGIDynamics`

Core APGI computational equations.

```python
from APGI-Parameter-Estimation-Protocol import APGIDynamics

# Compute surprise accumulation
S_t = APGIDynamics.compute_surprise_accumulation(
    Pi_e=1.0, Pi_i=1.2, z_e=1.5, z_i=1.2
)

# Compute ignition probability
P_ignition = APGIDynamics.compute_ignition_probability(
    S_t, theta_threshold=2.0
)
```

### 6. Psychological States

#### `APGI-Psychological-States-CLI`

Command-line interface for psychological state analysis.

```python
from APGI-Psychological-States-CLI import PsychologicalStatesAnalyzer

# Initialize analyzer
analyzer = PsychologicalStatesAnalyzer()

# Analyze psychological states
results = analyzer.analyze_states(
    data_file='psychological_data.csv',
    model_type='apgi'
)
```

#### `APGI-Psychological-States-GUI`

Graphical interface for psychological state visualization and analysis.

```python
from APGI-Psychological-States-GUI import main as gui_main

# Launch GUI (blocking call)
gui_main()
```

### 7. GUI Components

#### `APGIValidationGUI`

Validation protocol GUI with real-time progress tracking.

```python
from Validation.APGI-Validation-GUI import APGIValidationGUI

# Initialize GUI
gui = APGIValidationGUI()

# Run specific protocols
gui.run_protocols(['protocol_1', 'protocol_2'])
```

### 8. Validation Protocols

Validation protocols are located in the `Validation/` directory and follow a standard interface:

```python
# Each protocol should implement:
def run_validation():
    """Run the validation protocol and return results."""
    # Implementation here
    return results

def main():
    """Main entry point for standalone execution."""
    results = run_validation()
    print(results)
```

**Available Protocols:**

- Protocol 1: Basic validation
- Protocol 2: Cross-modal validation
- Protocol 3: Temporal validation
- Protocol 4: Clinical validation
- Protocol 5: Statistical validation
- Protocol 6: Robustness validation
- Protocol 7: Performance validation
- Protocol 8: Integration validation

### 7. Falsification Protocols

Falsification protocols are located in the `Falsification-Protocols/` directory:

```python
# Each protocol should implement:
def run_falsification():
    """Run falsification tests and return results."""
    # Implementation here
    return results

def main():
    """Main entry point for standalone execution."""
    results = run_falsification()
    print(results)
```

**Available Protocols:**

- Protocol 1: Surprise accumulation falsification
- Protocol 2: Precision-weighted integration falsification
- Protocol 3: Cross-modal validation falsification
- Protocol 4: Temporal dynamics falsification
- Protocol 5: Clinical predictions falsification
- Protocol 6: Neural correlates falsification

## CLI Interface

The framework provides a comprehensive command-line interface:

```bash
# Show help
python main.py --help

# Run formal model simulation
python main.py formal-model --simulation-steps 1000 --plot

# Multimodal integration
python main.py multimodal --input-data data.csv --output-file results.csv

# Parameter estimation
python main.py estimate-params --data-file experiment.csv --method mcmc

# Validation protocols
python main.py validate --protocol 1 --output-dir results/

# Falsification testing
python main.py falsify --protocol 1 --output-file falsification_results.json

# Configuration management
python main.py config --show
python main.py config --set simulation.default_steps=2000

# Log management
python main.py logs --tail 50 --level ERROR
python main.py logs --export logs.json

# GUI interface
python main.py gui --gui-type validation
python main.py gui --gui-type analysis --port 8080

# Data visualization
python main.py visualize --input-file results.csv --plot-type time_series

# Data export/import
python main.py export-data --input-file data.csv --output-file data.json --format json
python main.py import-data --input-file data.json --output-file data.csv --validate
```

## Error Handling

The framework provides comprehensive error handling:

```python
try:
    # Your code here
    result = some_function()
except APGIError as e:
    # Handle APGI-specific errors
    logger.error(f"APGI Error: {e}")
except ValueError as e:
    # Handle value errors
    logger.error(f"Value Error: {e}")
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected Error: {e}")
    raise
```

**Custom Exceptions:**

- `APGIError`: Base exception for APGI framework
- `ConfigurationError`: Configuration-related errors
- `ValidationError`: Validation protocol errors
- `IntegrationError`: Multimodal integration errors

## Performance Optimization

### Memory Management

```python
# Use context managers for resource management
with APGIContext() as ctx:
    results = ctx.run_analysis()

# Clean up resources
apgi_logger.cleanup_old_logs(days_to_keep=30)
```

### Parallel Processing

```python
# Use parallel execution for validation
python main.py validate --all-protocols --parallel

# Parallel batch processing
processor = APGIBatchProcessor(normalizer, config, parallel=True)
```

### Caching

```python
# Enable caching for expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(params):
    # Expensive computation here
    return result
```

## Best Practices

1. **Configuration Management**

   - Always use the configuration manager for parameters
   - Validate configuration values before use
   - Document custom configuration options

2. **Logging**

   - Use appropriate log levels
   - Include context in log messages
   - Export logs for analysis

3. **Error Handling**

   - Handle exceptions gracefully
   - Provide meaningful error messages
   - Log errors with context

4. **Performance**

   - Use batch processing for large datasets
   - Enable parallel processing when possible
   - Monitor performance metrics

5. **Testing**

   - Run the test suite before deployment
   - Validate results with known datasets
   - Use validation protocols regularly

## Extension Points

The framework is designed to be extensible:

1. **Custom Models**

   - Inherit from `SurpriseIgnitionSystem`
   - Implement required methods
   - Register with module loader

2. **Custom Protocols**

   - Follow the standard interface
   - Place in appropriate directory
   - Update CLI commands if needed

3. **Custom GUI Components**

   - Follow the GUI interface patterns
   - Integrate with main CLI
   - Document usage

## Dependencies

Core dependencies:

- Python >= 3.8
- NumPy, SciPy, Pandas
- PyMC, ArviZ (Bayesian modeling)
- Matplotlib, Seaborn (Visualization)
- Click, Rich (CLI)
- Loguru (Logging)

Optional dependencies:

- Flask (Web interface)
- Jupyter (Notebook support)
- PyTorch (Neural networks)

## License

This framework is provided under the MIT License. See LICENSE file for details.

## Support

For support and questions:

1. Check the troubleshooting guide
2. Run validation protocols
3. Check log files for errors
4. Consult the API documentation
