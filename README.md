# APGI Validation Framework

A computational framework for validating and falsifying Active Inference and Predictive Processing models through systematic experimentation.

This repository contains all code needed to reproduce the validation protocols, falsification tests, and computational benchmarks presented in our work on the Active Perception and Global Ignition (APGI) theory framework.

---

## Data Availability

All empirical data used for validation (synthetic EEG, fMRI vmPFC recordings, cross-cultural datasets) is organized in the `data_repository/` directory:

- `empirical_data/` - Raw and processed experimental data
- `metadata/` - Dataset documentation and codebooks
- `dashboard_data/` - Pre-computed results for dashboard visualization
- `codebooks/` - Variable dictionaries and data specifications

Pre-computed validation results and performance profiles are available in `apgi_outputs/` for rapid reproduction of figures and statistical analyses.

---

## Usage

### Installation

First, clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/apgi-validation.git
cd apgi-validation
```

We recommend creating a virtual environment for this project:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

For protocol-specific dependencies:

```bash
pip install -r requirements-protocols.txt
```

### Required Environment Variables

Set the minimum required security keys:

```bash
export PICKLE_SECRET_KEY=$(python -c "import os; print(os.urandom(32).hex())")
export APGI_BACKUP_HMAC_KEY=$(python -c "import os; print(os.urandom(32).hex())")
```

### Quick Start

Run the CLI to see available commands:

```bash
python main.py --help
```

Run a formal model simulation with default parameters:

```bash
python main.py formal-model --plot
```

Run with custom parameters:

```bash
python main.py formal-model --simulation-steps 500 --dt 0.01 --output-file results.csv
```

---

## Theory Modules

The theoretical foundation of APGI is implemented in the `Theory/` directory:

### Core Components

- **`APGI_Bayesian_Estimation_Framework.py`** - Bayesian parameter estimation with precision-weighted prediction errors
- **`APGI_Computational_Benchmarking.py`** - Performance profiling and scaling analysis
- **`APGI_Cross_Species_Scaling.py`** - Cross-species neural dynamics comparison
- **`APGI_Cultural_Neuroscience.py`** - Cultural variation in predictive processing
- **`APGI_DynamicalSystems_Formalization.py`** - Dynamical systems analysis of ignition thresholds
- **`APGI_Ignition_Dynamics_Simulator.py`** - Core ignition dynamics simulation engine
- **`APGI_Mathematical_Formalization.py`** - Mathematical formalization of APGI equations
- **`APGI_Neural_Mass_Models.py`** - Neural mass model implementations
- **`APGI_Physics_Integration.py`** - Physics-informed neural dynamics
- **`APGI_Precision_Dynamics.py`** - Precision-weighting mechanisms
- **`APGI_Rational_Pathway_Model.py`** - Rational pathway modeling
- **`APGI_Urgency_Gating.py`** - Urgency-signal gating mechanisms

All theory modules can be run independently or through the unified CLI in `main.py`.

---

## Validation Protocols

Systematic validation protocols are implemented in the `Validation/` directory:

### Protocol 1: Synthetic EEG Classification

**File:** `VP_01_SyntheticEEG_MLClassification.py`

Validates APGI predictions using machine learning classification of synthetic EEG signals. Generates synthetic P3b waveforms with varying latency and amplitude, then tests classification accuracy.

Run:

```bash
python Validation/VP_01_SyntheticEEG_MLClassification.py
```

### Protocol 2: Behavioral Bayesian Comparison

**File:** `VP_02_Behavioral_BayesianComparison.py`

Compares APGI model predictions against behavioral data using Bayesian model comparison metrics.

### Protocol 3: Active Inference Agent Simulations

**File:** `VP_03_ActiveInference_AgentSimulations.py`

Runs agent-based simulations to validate active inference predictions in controlled environments.

### Additional Protocols

- **`BayesianModelComparison_ParameterRecovery.py`** - Parameter recovery validation
- **`BayesianModelComparison_PredictiveAccuracy.py`** - Predictive accuracy assessment
- **`BayesianModelComparison_RobustnessChecks.py`** - Robustness validation
- **`CrossValidation_Empirical_Generalizability.py`** - Cross-validation protocols
- **`CrossValidation_ParameterStability.py`** - Parameter stability analysis
- **`Master_Validation.py`** - Orchestrates all validation protocols
- **`Multiverse_Analysis_Robustness.py`** - Multiverse robustness analysis
- **`VP_03_ActiveInference_AgentSimulations_Protocol3.py`** - Extended agent simulations
- **`VP_04_CausalIntervention_TMS.py`** - TMS intervention validation

Run all validations:

```bash
python Validation/Master_Validation.py
```

---

## Falsification Protocols

Rigorous falsification tests are implemented in the `Falsification/` directory:

### Core Falsification Tests

- **`FP_01_ActiveInference.py`** - Active inference model falsification
- **`FP_02_AgentComparison_ConvergenceBenchmark.py`** - Multi-agent convergence benchmarks
- **`FP_03_FrameworkLevel_MultiProtocol.py`** - Framework-level multi-protocol testing
- **`FP_04_PhaseTransition_EpistemicArchitecture.py`** - Phase transition analysis
- **`FP_05_SurvivalAnalysis_TimeToIgnition.py`** - Survival analysis of ignition timing
- **`FP_06_Perturbation_Resilience.py`** - Perturbation resilience testing
- **`FP_07_Diversity_BehavioralSpaceCoverage.py`** - Behavioral space coverage analysis
- **`FP_08_MetaLearning_FewShotAdaptation.py`** - Meta-learning adaptation tests
- **`FP_09_TemporalDynamics_SequenceSensitivity.py`** - Temporal dynamics validation
- **`FP_10_Adversarial_AttackResistance.py`** - Adversarial robustness testing
- **`FP_11_ResourceEfficiency_ComputationalCost.py`** - Resource efficiency benchmarks
- **`FP_12_Falsification_Aggregator.py`** - Aggregates falsification results

Run the falsification aggregator:

```bash
python Falsification/FP_12_Falsification_Aggregator.py
```

---

## Testing

Comprehensive test suite with property-based testing:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_cli_integration_comprehensive.py      # CLI integration
pytest tests/test_utility_modules_comprehensive.py      # Utility modules
pytest tests/test_file_io_real.py                       # File I/O
pytest tests/test_data_pipeline_end_to_end.py           # Data pipeline
pytest tests/test_property_based_comprehensive.py       # Property-based tests
pytest tests/test_validation_falsification_protocols_individual.py  # Protocols
pytest tests/test_performance_regression.py             # Performance
pytest tests/test_concurrent_config_access.py           # Concurrency
```

Run with coverage:

```bash
pytest --cov=utils --cov=Theory --cov=Validation --cov=Falsification
```

---

## GUI Applications

Launch graphical interfaces for interactive exploration:

```bash
python Theory_GUI.py         # Theory module GUI
python Validation_GUI.py     # Validation protocols GUI
python Falsification_Protocols_GUI.py  # Falsification tests GUI
python Tests_GUI.py          # Test runner GUI
python Utils_GUI.py          # Utilities GUI
```

---

## Configuration

Configuration files are in the `config/` directory:

- **`default.yaml`** - Default configuration settings
- **`config_schema.json`** - Configuration validation schema
- **`profiles/`** - Environment-specific profiles (ADHD, anxiety-disorder, research-default)
- **`versions/`** - Version-specific configurations

Load a specific profile:

```bash
python main.py formal-model --params config/profiles/adhd.yaml
```

---

## Project Structure

```text
apgi-validation/
├── Theory/                    # Theoretical framework implementations
├── Validation/                # Validation protocols
├── Falsification/             # Falsification test protocols
├── utils/                     # Utility modules
│   ├── errors.py              # Typed error taxonomy
│   ├── security_audit_logger.py  # Security auditing
│   ├── cache_manager.py       # Cache governance
│   ├── data_protection.py     # Data protection workflows
│   └── ...
├── config/                    # Configuration files
├── data_repository/           # Data storage
│   ├── empirical_data/
│   ├── metadata/
│   └── codebooks/
├── tests/                     # Comprehensive test suite
├── docs/                      # Documentation
└── apgi_outputs/              # Output directory
    └── performance_profiles/
```

---

## Security Features

- Secure module loading with path validation
- Persistent audit logging (`SecurityAuditLogger`)
- Dependency vulnerability scanning (pip-audit integration)
- Thread-safe configuration access
- Deny-by-default security context
- Automated SAST/DAST in CI pipeline (Bandit, Semgrep, Safety)
- Typed error taxonomy with domain-specific exceptions
- Data protection workflows with secure deletion

---

## Documentation

Additional documentation is available in `docs/`:

- **`APGI-Design-Guide.md`** - UI/UX design specifications
- **`APGI-Equations.md`** - Complete mathematical formalization
- **`APGI-Parameter-Specifications.md`** - Parameter specifications
- **`APGI-Empirical-Credibility-Roadmap.md`** - Empirical validation roadmap
- **`Compliance-Matrix.md`** - Regulatory compliance matrix
- **`APGI-Falsification-Criteria.md`** - Falsification criteria

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{apgi_validation_framework,
  title={APGI Validation Framework: A Computational Framework for Validating Active Inference Models},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/apgi-validation}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## About

The APGI Framework provides a systematic, scientifically rigorous approach to validating and falsifying computational models of active inference and predictive processing. It combines theoretical formalization, computational benchmarking, and empirical validation through a unified CLI interface with comprehensive testing and security features.
