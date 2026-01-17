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
python main.py formal-model --simulation-steps 1000

# Execute multimodal integration
python main.py multimodal --input-data data/sample.csv

# Estimate parameters from data
python main.py estimate-parameters --data-file data/experimental.csv

# Run validation protocols
python main.py validate --protocol all
```

## Backup and Restore

```bash
python main.py backup --components "config,logs" --description "My backup"
python main.py restore --backup-id backup_20240117_120000
python main.py backups --limit 10
python main.py delete-backup --cleanup-all
```

## Configuration Versioning

```bash
python main.py config-version --description "Updated parameters" --author "User"
python main.py config-versions --limit 15
python main.py config-restore --version-id v20240117_120000
python main.py config-diff
```

## Error Management

```bash
python main.py errors --category CONFIGURATION --severity HIGH
python main.py test-errors --test-config --test-validation
```

## 🔬 Core Components

### 1. Formal Model (`APGI-Formal-Model.py`)

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
  formal-model      Run formal model simulations
  multimodal        Execute multimodal integration
  estimate-params   Perform parameter estimation
  validate          Run validation protocols
  falsify          Execute falsification tests
  config          Manage configuration settings
  logs            View log files
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
python main.py validate --protocol all

# Run specific validation
python main.py validate --protocol cross-validation
```

### Falsification Testing

```bash
# Execute falsification protocols
python main.py falsify --protocol 1
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

Copyright (c) 2024 APGI Theory Framework

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
