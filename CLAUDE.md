# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APGI (Active Perception and Generative Inference) Validation Framework - A CLI-based scientific computing framework for validating and falsifying active inference models through computational experiments.

## Development Commands

### Run the CLI application

```bash
python main.py --help                    # Show all available commands
python main.py validate --data data.csv     # Run validation protocol
python main.py falsify --model model.pkl    # Run falsification protocol
python main.py visualize --output plot.png  # Generate visualizations
python main.py benchmark --iterations 100
```

### Install dependencies

```bash
pip install -r requirements.txt              # Core dependencies
pip install -r requirements-dev.txt          # Development dependencies (includes test/lint tools)
```

### Testing

```bash
pytest                                              # All tests
pytest tests/test_cli_integration_comprehensive.py    # CLI integration tests
pytest tests/test_utility_modules_comprehensive.py   # Utility module tests
pytest tests/test_file_io_real.py                  # Real file I/O tests
pytest tests/test_data_pipeline_end_to_end.py      # Data pipeline tests
pytest tests/test_property_based_comprehensive.py   # Property-based tests (Hypothesis)
pytest tests/test_validation_falsification_protocols_individual.py  # Protocol tests
pytest tests/test_fixtures_utilization.py          # Fixture tests
pytest tests/test_performance_regression.py        # Performance tests
pytest tests/test_concurrent_config_access.py      # Concurrency tests
pytest tests/test_persistent_audit_logger.py       # Audit logger tests
pytest --hypothesis-profile=ci                      # Full Hypothesis runs (100 examples)
```

### Linting and formatting

```bash
black .                                            # Format code
isort .                                           # Sort imports
flake8 .                                          # Lint code (Config in .flake8)
```

## Architecture

### Main entry point

`main.py` defines the unified CLI interface using `click` with commands for validation, falsification, visualization, and benchmarking. It includes secure module loading (`secure_load_module`, `secure_load_module_from_path`) with path validation to prevent directory traversal attacks.

### Configuration

`config/` directory contains YAML configuration files:

- `default.yaml` - Default configuration settings
- `config_schema.json` - Configuration schema validation
- `profiles/` - Environment-specific profiles (adhd.yaml, anxiety-disorder.yaml, etc.)
- `versions/` - Version-specific configurations

### Key directories

- `Validation/` - Validation protocols (APGI_Validation_GUI.py, ActiveInference_AgentSimulations_Protocol3.py, BayesianModelComparison_ParameterRecovery.py, etc.)
- `Falsification/` - Falsification protocols (APGI_Falsification_Aggregator.py, CausalManipulations_TMS_Pharmacological_Priority2.py, etc.)
- `utils/` - Utility modules (dependency_scanner.py, security_audit_logger.py, backup_manager.py, batch_processor.py, etc.)
- `data/` - Data directory for input/output files
- `tests/` - Comprehensive test suite with property-based testing (Hypothesis)
- `docs/` - Documentation (APGI_Equations.md, APGI-Parameter-Specifications.md, etc.)

### Security features

- Secure module loading with path validation
- Audit logging via `SecurityAuditLogger` and persistent audit logging
- Environment variable validation for required keys (`PICKLE_SECRET_KEY`, `APGI_BACKUP_HMAC_KEY`)
- Dependency vulnerability scanning via `DependencyScanner` (pip-audit)
- Thread-safe configuration access with `_config_lock`

### Key Python modules

- `APGI_Equations.py` - Core APGI equations (entropy, KL divergence, free energy, etc.)
- `APGI_Entropy_Implementation.py` - Entropy implementation
- `APGI_Full_Dynamic_Model.py` - Full dynamic model
- `APGI_Liquid_Network_Implementation.py` - Liquid network implementation
- `APGI_Multimodal_Classifier.py` - Multimodal classifier
- `APGI_Multimodal_Integration.py` - Multimodal integration

### Test setup

Unit tests use pytest with Hypothesis for property-based testing. The `conftest.py` provides fixtures including `temp_dir`, `sample_config`, `sample_data`, `raises_fixture`, `oom_fixture`, and `flaky_operation`. Hypothesis profiles: `dev` (20 examples, default locally), `ci` (100 examples), `thorough` (1000 examples).

## Required Environment Variables

Minimum for development:

```bash
PICKLE_SECRET_KEY=<random 64-character hex string>
APGI_BACKUP_HMAC_KEY=<random 64-character hex string>
```

Generate keys: `python -c "import os; print(os.urandom(32).hex())"`

For production, additional security keys and settings may be required.

## CLI Commands

### Validation commands

- `validate` - Run validation protocols on data
- `validate-entropy` - Validate entropy calculations
- `validate-active-inference` - Validate active inference models

### Falsification commands

- `falsify` - Run falsification protocols
- `falsify-entropy` - Falsify entropy models
- `falsify-active-inference` - Falsify active inference models
- `falsify-causal` - Falsify causal manipulation models

### Visualization commands

- `visualize` - Generate visualizations from data
- `visualize-time-series` - Create time series plots
- `visualize-scatter` - Create scatter plots
- `visualize-heatmap` - Create heatmaps
- `visualize-distribution` - Create distribution plots

### Utility commands

- `benchmark` - Run computational benchmarks
- `audit-dependencies` - Scan for dependency vulnerabilities
- `backup` - Create backups of data and configuration
- `restore` - Restore from backups

## Data formats

Input data formats:

- JSON files for structured data
- CSV files for time series and tabular data
- Pickle files for serialized Python objects

Output formats:

- JSON reports for validation/falsification results
- PNG/SVG plots for visualizations
- Pickle files for model checkpoints
