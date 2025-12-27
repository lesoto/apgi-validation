# APGI Parameter Estimation Protocol

## Overview
This script implements a hierarchical Bayesian framework for estimating parameters of the Active Predictive Global Ignition (APGI) model, which integrates interoceptive and exteroceptive processing. The protocol synthesizes realistic neural and behavioral data, fits a hierarchical Bayesian model, and validates parameter recovery across multiple modalities.

## Core Components

### 1. Neural Signal Generator (`NeuralSignalGenerator` Class)
Generates biologically plausible neural and physiological signals:

- **Heartbeat-Evoked Potential (HEP)**: Simulates EEG waveforms following heartbeats
  - Adds realistic noise (1/f pink noise) and 50Hz line noise
  - Amplitude modulated by interoceptive precision (Pi_i)

- **P3b Component**: Simulates the P3b event-related potential
  - Models both P3b and earlier P2 components
  - Includes physiological noise

- **Pupil Response**: Generates pupil dilation responses
  - Includes blink artifacts and slow drifts
  - Modulated by interoceptive precision

### 2. APGI Dynamics Engine (`APGIDynamics` Class)
Implements the core computational model:

- **Surprise Accumulation**: Integrates interoceptive and exteroceptive prediction errors
  - Implements: dS/dt = -S/τ + f(Πₑ·|εₑ|, β·Πᵢ·|εᵢ|)
  - Tracks surprise (S) over time

- **Ignition Probability**: Computes probability of conscious perception
  - Implements: B(t) = σ(α(S(t) - θ(t)))
  - Sigmoidal function with gain parameter α

### 3. Synthetic Dataset Generation (`generate_synthetic_dataset` Function)
Creates realistic multimodal datasets with known ground truth parameters:

- **Task 1: Detection Psychometric Curve**
  - Varies stimulus intensity
  - Simulates binary detection responses
  
- **Task 2: Heartbeat Detection**
  - Generates HEP waveforms and pupil responses
  - Simulates behavioral d-prime for heartbeat detection
  
- **Task 3: Oddball P3b**
  - Simulates ERP responses to interoceptive and exteroceptive deviants
  - Captures neural signatures of prediction error processing

### 4. Hierarchical Bayesian Model (`build_apgi_model` Function)
Implements a probabilistic model in PyMC with:

- **Hierarchical Structure**:
  - Group-level hyperpriors for population parameters
  - Subject-level parameters with non-centered parameterization
  - Parameter constraints for biological plausibility

- **Likelihood Functions**:
  1. Detection psychometric curve (binomial likelihood)
  2. Heartbeat detection d-prime (normal likelihood)
  3. HEP amplitude (neural prior, normal likelihood)
  4. Pupil dilation response (normal likelihood)
  5. P3b ratio between intero/extero conditions (normal likelihood)

### 5. Validation Framework

#### Parameter Recovery (`validate_parameter_recovery`)
- Computes correlation between true and recovered parameters
- Calculates RMSE and credible interval coverage
- Implements threshold-based passing criteria for each parameter

#### Predictive Validity (`assess_predictive_validity`)
- Tests model predictions on held-out behavioral tasks:
  - Emotional Stroop interference
  - Continuous Performance Test (CPT) lapses
  - Body Vigilance Scale scores

#### Test-Retest Reliability (`assess_test_retest_reliability`)
- Simulates test-retest reliability over time
- Computes intraclass correlation coefficients (ICCs)
- Assesses stability of parameter estimates

## Key Features

1. **Multimodal Integration**: Combines behavioral, EEG, and pupillometry data
2. **Hierarchical Bayesian Modeling**: Accounts for both within- and between-subject variability
3. **Parameter Recovery**: Validates ability to accurately recover ground truth parameters
4. **Predictive Validity**: Tests generalizability to novel tasks
5. **Test-Retest Reliability**: Assesses stability of parameter estimates
6. **Computational Efficiency**: Vectorized operations for performance

## Dependencies

- Python 3.7+
- NumPy
- PyMC
- ArviZ
- SciPy
- Matplotlib
- scikit-learn

## Usage

1. Generate synthetic dataset:
   ```python
   subjects, true_params = generate_synthetic_dataset(n_subjects=100)
   ```

2. Build and sample from the model:
   ```python
   model = build_apgi_model(subjects)
   with model:
       trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)
   ```

3. Validate parameter recovery:
   ```python
   recovery_results, falsified = validate_parameter_recovery(true_params, trace, n_subjects=100)
   ```

4. Assess predictive validity:
   ```python
   validity_results = assess_predictive_validity(subjects, trace)
   ```

## Model Parameters

| Parameter | Description | Constraints |
|-----------|-------------|-------------|
| θ₀        | Perceptual threshold | [0.25, 0.85] |
| Πᵢ        | Interoceptive precision | [0.4, 2.8] |
| β         | Interoceptive bias | [0.6, 2.3] |
| α         | Gain parameter | [2.5, 7.5] |

## Output

The script generates several outputs:
1. Parameter recovery plots (true vs. recovered values)
2. Posterior distributions for all parameters
3. Convergence diagnostics (R-hat, effective sample size)
4. Predictive validity metrics
5. Test-retest reliability statistics
