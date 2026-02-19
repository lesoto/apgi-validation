# APGI Theory Framework

A comprehensive computational framework for implementing the **Adaptive Pattern Generation and Integration (APGI)** theory, which models psychological state dynamics through surprise accumulation, ignition cascades, and multimodal integration.

## 🎯 Overview

The APGI Theory Framework provides:

- **Formal mathematical models** of surprise accumulation and ignition dynamics
- **Multimodal integration** protocols for psychological state assessment
- **Parameter estimation** tools using Bayesian inference
- **Validation protocols** for model verification
- **CLI interfaces** for interactive exploration

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd apgi-theory

# Install dependencies
pip install -r requirements.txt

# Run the main CLI
python main.py --help
```

### Basic Usage

```bash
# Run formal model simulation
python main.py formal_model --simulation-steps 1000

# Execute multimodal integration
python main.py multimodal --input-data data/sample.csv

# Estimate parameters from data
python main.py estimate-params --data-file data/experimental.csv

# Run validation protocols
python main.py validate --protocol 3

# Run falsification protocols
python main.py falsify --protocol 1

# CLI integration via `validate-pipeline` command
python main.py validate-pipeline --protocol 1 --use-synthetic
python main.py validate-pipeline --protocol 2 --input-data data.csv
python main.py validate-pipeline --protocol 3 --use-synthetic --samples 2000

# View framework information
python main.py info
```

## CLI Commands Reference

### Core Commands

| Command | Description | Status |
| :--- | :--- | :--- |
| `python main.py formal_model` | Run formal model simulations | WORKING |
| `python main.py multimodal` | Execute multimodal integration | WORKING |
| `python main.py estimate-params` | Perform parameter estimation | WORKING |
| `python main.py validate` | Run validation protocols | WORKING |
| `python main.py falsify` | Execute falsification tests | WORKING |

### Protocol Commands

| Command | Description | Status |
| :--- | :--- | :--- |
| `python main.py validate --protocol N` | Run specific validation protocol (1-8) | WORKING |
| `python main.py validate --all-protocols` | Run all validation protocols | WORKING |
| `python main.py falsify --protocol N` | Run specific falsification protocol (1-6) | WORKING |

### Utility Commands

| Command | Description | Status |
| :--- | :--- | :--- |
| `python main.py info` | Show framework information | WORKING |
| `python main.py config --show` | Show current configuration | WORKING |
| `python main.py config --set key=value` | Set configuration value | WORKING |
| `python main.py logs --tail 20` | View recent log entries | WORKING |
| `python main.py logs --follow` | Follow logs in real-time | WORKING |
| `python main.py performance` | Show performance metrics | WORKING |

### Data Management

| Command | Description | Status |
| :--- | :--- | :--- |
| `python main.py export --input-file X --output-file Y` | Export data | WORKING |
| `python main.py import --input-file X --output-file Y` | Import data | WORKING |
| `python main.py visualize --input-file X` | Generate visualizations | WORKING |
| `python main.py cache --action clear` | Clear cache | WORKING |
| `python main.py dashboard` | Generate dashboards | WORKING |

### Backup & Restore

| Command | Description | Status |
| :--- | :--- | :--- |
| `python main.py backup --components "config,logs"` | Create backup | WORKING |
| `python main.py restore --backup-id X` | Restore from backup | WORKING |
| `python main.py backups --limit 10` | List backups | WORKING |
| `python main.py delete-backup --backup-id X` | Delete backup | WORKING |

### Configuration Versioning

| Command | Description | Status |
| :--- | :--- | :--- |
| `python main.py config-version --description "X"` | Create config version | WORKING |
| `python main.py config-versions --limit 10` | List config versions | WORKING |
| `python main.py config-restore --version-id X` | Restore config version | WORKING |
| `python main.py config-diff` | Compare config with last version | WORKING |

### Error Management

| Command | Description | Status |
| :--- | :--- | :--- |
| `python main.py errors --category CONFIGURATION` | View errors | WORKING |
| `python main.py test-errors --test-config` | Test error handling | WORKING |

### GUI

| Command | Description | Status |
| :--- | :--- | :--- |
| `python main.py gui --gui-type validation` | Launch validation GUI | WORKING |
| `python main.py gui --gui-type falsification` | Launch falsification GUI | WORKING |

## 🔬 Core Components

### 1. Formal Model (`APGI-Equations.py`)

Implements dynamical system for accumulated surprise and ignition:

  • Surprise accumulation with timescale τ_S
  • Threshold adaptation with timescale τ_θ
  • Ignition cascades and reset dynamics
  • Metabolic and arousal modulation

### 2. Multimodal Integration (`APGI-Multimodal-Integration.py`)

Integrates multiple data modalities:

  • Physiological signals (HRV, GSR, EEG)
  • Behavioral metrics
  • Self-report measures
  • Real-time state estimation
  • Cross-modal fusion algorithms

### 3. Parameter Estimation (`APGI-Parameter-Estimation-Protocol.py`)

Bayesian parameter inference:

  • Markov Chain Monte Carlo (MCMC) sampling
  • Posterior distribution analysis
  • Model comparison metrics
  • Uncertainty quantification

### 4. Validation Protocols (`Validation/`)

Comprehensive testing suite:

- Cross-validation procedures
- Predictive validity tests
- Sensitivity analysis
- Robustness checks

### System Requirements

- Python 3.8+
- 8GB+ RAM recommended
- 2GB+ disk space

See `requirements.txt` for complete list. Key dependencies include:

- `numpy`, `scipy`, `pandas` - Scientific computing
- `torch`, `scikit-learn` - Machine learning
- `pymc`, `arviz` - Bayesian modeling
- `matplotlib`, `seaborn` - Visualization

## 📖 Detailed Usage

### Command Line Interface

The unified CLI provides access to all framework components:

```bash
python main.py [COMMAND] [OPTIONS]

Commands:
  formal_model      Run formal model simulations
  multimodal        Execute multimodal integration
  estimate-params   Perform parameter estimation
  validate          Run validation protocols
  falsify          Execute falsification tests
  config           Manage configuration settings
  logs             View log files
  info             Show framework information
  backup           Create backups
  restore          Restore from backups
  gui              Launch GUI interfaces
```

### Configuration Management

Configuration files in `config/` allow customization of:

- Model parameters
- Simulation settings
- Data paths
- Logging preferences
- Visualization options

### Logging Framework

Comprehensive logging system with:

- Multiple log levels (DEBUG, INFO, WARNING, ERROR)
- Rotating log files
- Structured output formats
- Performance metrics tracking

## 🧪 Validation and Testing

### Running Validation

```bash
# Run all validation protocols
python main.py validate --all-protocols

# Run specific validation protocol (1-8)
python main.py validate --protocol 3

# Run validation with output directory
python main.py validate --protocol 1 --output-dir results/
```

### Falsification Testing

```bash
# List available falsification protocols
python main.py falsify

# Execute specific falsification protocol (1-6)
python main.py falsify --protocol 1

# Run falsification with output file
python main.py falsify --protocol 2 --output-file results.json
```

## 📊 Example Workflows

### 1. Basic Simulation

```python
from APGIFormalModel import SurpriseIgnitionSystem

# Initialize system
system = SurpriseIgnitionSystem()

# Run simulation
results = system.simulate(steps=1000, dt=0.01)
```

### 2. Parameter Estimation

```python
from APGIParameterEstimation import ParameterEstimator

# Load data
data = load_experimental_data('data/experiment.csv')

# Estimate parameters
estimator = ParameterEstimator()
results = estimator.estimate_parameters(data)
```

### 3. Multimodal Integration

```python
from APGIMultimodalIntegration import MultimodalIntegrator

# Initialize integrator
integrator = MultimodalIntegrator()

# Process multimodal data
state_estimates = integrator.integrate_data(
    physiological=physio_data,
    behavioral=behavior_data,
    self_report=report_data
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all validation protocols pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the details below:

```text
MIT License

Copyright (c) 2026 APGI Framework

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📚 References

- [Primary APGI Theory Paper]
- [Mathematical Foundations]
- [Validation Studies]

## 🆘 Support

For issues and questions:

1. Check the logs directory for error messages
2. Review validation protocol outputs
3. Consult the troubleshooting section in documentation
4. Create an issue with detailed error information

## 🔄 Version History

- **v1.0.0** - Initial framework release
- **v1.1.0** - Added multimodal integration
- **v1.2.0** - Enhanced parameter estimation
- **v1.3.0** - Unified CLI and configuration system
