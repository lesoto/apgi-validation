
# APGI Protocol 4: Information-Theoretic Phase Transition Analysis

## Overview

Protocol 4 tests one of APGI's most distinctive predictions: that **conscious ignition represents a genuine computational phase transition**, not merely a continuous increase in processing. This protocol uses information-theoretic measures and critical phenomena analysis to test whether ignition exhibits the characteristic signatures of phase transitions found in physical and computational systems.

## Theoretical Background

### The Phase Transition Hypothesis

The APGI framework proposes that when accumulated surprise S(t) crosses the threshold θ(t), the system undergoes a qualitative state change analogous to physical phase transitions (e.g., water → ice, paramagnet → ferromagnet). This makes specific, falsifiable predictions about the **information-theoretic signatures** that should appear at ignition.

### Why This Matters

If conscious access were merely a **continuous amplification** of sensory signals (as some theories propose), we would NOT expect:

- Discontinuous jumps in information flow
- Diverging variance near threshold
- Critical slowing down
- Long-range temporal correlations

But APGI predicts ALL of these should occur at ignition, providing a **computational fingerprint** of the phase transition.

## What Protocol 4 Tests

### Five Core Predictions

#### P4a: Discontinuity in Surprise Dynamics

**Prediction:** The first derivative of surprise S(t) should show a sharp discontinuity at ignition, with effect size **d > 0.8** compared to random timepoints.

**Why:** Phase transitions involve sudden reorganization of system dynamics, creating observable jumps in state variables.

#### P4b: Diverging Susceptibility

**Prediction:** Variance of S near threshold should be **>2× higher** than variance far from threshold.

**Why:** At critical points, systems become highly sensitive to perturbations (susceptibility χ → ∞), manifesting as increased variance.

#### P4c: Critical Slowing Down

**Prediction:** Autocorrelation should be **>1.5× higher** near threshold than far from it.

**Why:** Near phase transitions, relaxation time diverges, causing the system to "slow down" (increased temporal correlations).

#### P4d: Integrated Information Spike

**Prediction:** Φ (a measure of integrated information) should be **>2× higher** at ignition than baseline.

**Why:** Phase transitions create system-wide integration; information in the whole exceeds sum of parts.

#### P4e: Long-Range Correlations

**Prediction:** Hurst exponent H near threshold should be **>0.6** (indicating long-range correlations), while H far from threshold ≈ 0.5 (random walk).

**Why:** Critical systems exhibit power-law correlations across timescales, detectable via Hurst analysis.

### Falsification Criteria

The framework is **falsified** if:

- **F4.1:** Susceptibility ratio < 1.2 (no divergence)
- **F4.2:** Φ ratio < 1.3 (no information integration spike)
- **F4.3:** Critical slowing ratio < 1.2 (continuous dynamics)
- **F4.4:** Discontinuity effect size d < 0.5 (gradual transition)
- **F4.5:** Hurst exponent near threshold < 0.55 (no long-range correlations)

## Implementation Structure

### Part 1: APGI Dynamical System

```python
class APGIDynamicalSystem
```

Simulates the core APGI equations with time-varying inputs, generating realistic surprise accumulation and ignition events.

### Part 2: Information-Theoretic Measures

```python
class InformationTheoreticAnalysis
```text
Computes:
- **Transfer Entropy (TE):** Directed information flow between variables
- **Integrated Information (Φ):** System-level integration measure
- **Mutual Information (MI):** Statistical dependencies
- **Entropy Rates:** Predictability of dynamics


### Part 3: Phase Transition Detection


```python
class PhaseTransitionDetector
```text
Analyzes:
- **Discontinuities:** Jumps in derivatives at threshold crossing
- **Susceptibility:** Variance scaling near critical points
- **Critical Slowing:** Autocorrelation increase
- **Hurst Exponents:** Long-range temporal correlations via R/S analysis


### Part 4: Comprehensive Analysis


```python
class ComprehensivePhaseTransitionAnalysis
```text
Integrates all measures, runs Monte Carlo simulations, and generates aggregate statistics.


### Part 5: Falsification Testing


```python
class FalsificationChecker
```text
Systematically tests all five criteria against thresholds, generates comprehensive report.


### Part 6: Visualization


Multi-panel plots showing:
- Example time series with ignition events
- Information-theoretic measure distributions
- Phase transition signature comparisons
- Prediction summary tables


## Expected Baseline Values for Innovation #33 (Cross-Species)

To support Innovation #33 (Cross-Species Translation), the following baseline values for Φ (Integrated Information) and Transfer Entropy are provided across different species:

### Integrated Information (Φ) Baselines

| Species | Conscious State | Φ Range (bits) | Experimental Paradigm | Key Reference |
|---------|-----------------|----------------|----------------------|---------------|
| **Human** | Awake/Conscious | 0.3 – 0.8 | Masked detection (P3b trials) | Oizumi et al. (2014) |
| **Human** | Deep Sleep (N3) | 0.05 – 0.15 | Sleep EEG during slow waves | Tononi et al. (2016) |
| **Human** | Anesthesia (Propofol) | 0.01 – 0.08 | Loss of consciousness | Casali et al. (2013) |
| **Macaque** | Awake/Fixation | 0.2 – 0.6 | Visual discrimination task | Tegmark (2016) |
| **Macaque** | Natural Sleep | 0.03 – 0.12 | Multi-unit recordings during sleep | N/A |
| **Mouse** | Awake/Exploring | 0.1 – 0.4 | Open field behavior | Gao et al. (2017) |
| **Mouse** | Anesthesia | 0.005 – 0.05 | Isoflurane administration | Gao et al. (2017) |
| **Rat** | Awake/Active | 0.15 – 0.5 | Navigation/Exploration | Barrett et al. (2020) |
| **C. elegans** | Active Movement | 0.001 – 0.01 | Nose touch response | Kim et al. (2013) |

**APGI Prediction**: Φ should spike to >2× baseline at ignition threshold crossing for all mammalian species (human, macaque, mouse, rat).

### Transfer Entropy (TE) Baselines

| Direction | Awake TE (nats) | Sleep TE (nats) | Paradigm |
|-----------|-----------------|-----------------|----------|
| **Frontal → Parietal** | 0.15 – 0.35 | 0.03 – 0.10 | Resting state EEG/fMRI |
| **Parietal → Frontal** | 0.20 – 0.45 | 0.05 – 0.12 | Visual task fMRI |
| **Insula → Frontal** | 0.10 – 0.25 | 0.02 – 0.08 | Interoceptive attention |
| **Frontal → Insula** | 0.08 – 0.20 | 0.02 – 0.06 | Cognitive control |
| **Sensory → Association** | 0.25 – 0.50 | 0.08 – 0.15 | Oddball ERP |

**APGI Prediction**: TE should diverge near ignition threshold (critical slowing signature), with TE_peak / TE_baseline > 1.5× for all species with measurable cortical dynamics.

### Cross-Species Scaling Law

Φ scales with cortical neuron count (N) following:

```text
Φ_species ≈ Φ_human × (N_species / N_human)^0.75
```

Approximate scaling:

- Human (16B neurons): 1.0× reference
- Macaque (6B neurons): 0.55× human Φ
- Mouse (70M neurons): 0.11× human Φ
- Rat (200M neurons): 0.18× human Φ

### Falsification Criteria for Innovation #33

- **F33.1**: If Φ_human < 0.15 at conscious ignition → Φ underestimation hypothesis
- **F33.2**: If TE divergence < 1.2× near threshold → No critical slowing; continuous dynamics
- **F33.3**: If cross-species Φ doesn't follow scaling law (r² < 0.6) → Species-specific mechanisms

## Usage

### Basic Execution

```python
python APGI_Protocol_4.py
```text

This runs the full pipeline:
1. Example simulation (100s duration)
2. Information-theoretic analysis
3. Monte Carlo analysis (100 simulations)
4. Falsification testing
5. Comprehensive visualization

### Custom Parameters

```python
from APGI_Protocol_4 import *


# Create system with custom parameters


system = APGIDynamicalSystem(
    tau=0.2,           # Decay time constant (s)
    theta_0=0.55,      # Baseline threshold
    alpha=5.0,         # Sigmoid steepness
    dt=0.01            # Integration timestep (s)
)


# Define time-varying inputs


def input_generator(t):
    return {
        'Pi_e': 1.0 + 0.3 * np.sin(2 * np.pi * t / 15),
        'eps_e': np.random.normal(0.4, 0.2),
        'beta': 1.15,
        'Pi_i': 1.0 + 0.5 * np.sin(2 * np.pi * t / 25),
        'eps_i': np.random.normal(0.2, 0.15)
    }


# Run simulation


history = system.simulate(
    duration=100.0,
    input_generator=input_generator,
    theta_noise_sd=0.08
)


# Analyze


analyzer = ComprehensivePhaseTransitionAnalysis()
results = analyzer.analyze_simulation(history)
```text


### Monte Carlo Analysis


```python

# Run multiple simulations for robust statistics


analyzer = ComprehensivePhaseTransitionAnalysis()

results_df = analyzer.run_monte_carlo(
    n_simulations=200,
    duration=100.0,
    save_results=True
)


# Check falsification criteria


checker = FalsificationChecker()
report = checker.check_all_criteria(results_df)
print_falsification_report(report)
```text


## Output Files


### 1. `protocol4_monte_carlo_results.csv`


Complete results from all simulations:
- Information-theoretic measures (TE, Φ, MI)
- Phase transition signatures (discontinuity, susceptibility, critical slowing, Hurst)
- Per-simulation metadata


### 2. `protocol4_results.json`


Summary statistics and falsification report:
```json
{
  "config": {...},
  "n_simulations": 100,
  "summary_statistics": {
    "Φ Ratio": {"mean": 2.34, "std": 0.45},
    "Susceptibility Ratio": {"mean": 2.78, "std": 0.62},
    ...
  },
  "falsification": {
    "overall_falsified": false,
    "passed_criteria": [...],
    "falsified_criteria": []
  }
}
```text


### 3. `protocol4_results.png`


Comprehensive 16-panel visualization:
- **Row 1:** Example dynamics, ignition events (zoomed)
- **Row 2:** Information-theoretic measures (Φ, TE, MI)
- **Row 3:** Phase transition signatures (discontinuity, susceptibility, critical slowing, Hurst)
- **Row 4:** Prediction summary table, correlation matrix


## Interpreting Results


### Evidence for Phase Transition


The framework is **strongly supported** if:

✅ **Φ ratio > 2.0:** Ignition creates genuine system-wide integration
✅ **Susceptibility ratio > 2.0:** Variance diverges near threshold
✅ **Critical slowing > 1.5:** Temporal correlations increase at criticality
✅ **Discontinuity d > 0.8:** Sharp transition in dynamics
✅ **Hurst > 0.6 near threshold:** Long-range temporal dependencies emerge


### Evidence Against Phase Transition


The framework is **challenged** if:

❌ **Φ ratio < 1.3:** No integration spike (continuous processing)
❌ **Susceptibility ratio < 1.2:** Variance doesn't diverge
❌ **Critical slowing < 1.2:** No temporal slowing
❌ **Discontinuity d < 0.5:** Gradual transition
❌ **Hurst ≈ 0.5 everywhere:** No long-range correlations


### Typical Expected Results


Based on APGI's theoretical predictions, you should observe:

```text
Information-Theoretic:
  Φ Ratio:              2.2-2.8× (strong integration)
  TE (S→B):             0.15-0.25 (directed flow)
  MI (S;θ):             0.8-1.2 (strong coupling)

Phase Transition Signatures:
  Discontinuity d:      0.9-1.3 (large effect)
  Susceptibility:       2.5-3.5× (diverging variance)
  Critical Slowing:     1.8-2.4× (slowing down)
  Hurst (near):         0.62-0.72 (long-range)
  Hurst (far):          0.48-0.52 (random walk)
```text


## Technical Details


### Transfer Entropy Calculation


Transfer entropy quantifies **directed information flow**:

```text
TE(X→Y) = I(Y_t; X_{t-τ} | Y_{t-τ})
 = H(Y_t | Y_{t-τ}) - H(Y_t | Y_{t-τ}, X_{t-τ})
```text

Implementation uses:
- K-bins discretization (n_bins=20)
- Lag τ optimized per time series
- Conditional entropy via joint probability estimation


### Integrated Information (Φ)


Φ-like measure based on **mutual information**:

```text
Φ ≈ Σ H(Xᵢ) - H(X₁, X₂, ..., Xₙ)
```text

Where:
- H(Xᵢ) = marginal entropy of variable i
- H(X₁,...,Xₙ) = joint entropy of all variables
- Moving window (50 samples) for temporal resolution


### Hurst Exponent via R/S Analysis


Rescaled range (R/S) analysis:

```text
R/S ∝ n^H
```text

Where:
- R = Range of cumulative deviations
- S = Standard deviation
- H = Hurst exponent (slope in log-log plot)

Interpretation:
- H = 0.5: Uncorrelated (random walk)
- H > 0.5: Persistent (long memory)
- H < 0.5: Anti-persistent


## Computational Requirements


### Memory


- Single simulation: ~50 MB
- Monte Carlo (100 sims): ~500 MB
- Recommended: 2+ GB RAM


### Time


- Single simulation (100s): ~2 seconds
- Monte Carlo (100 sims): ~5 minutes
- 1000 simulations: ~45 minutes


### Dependencies


```text
numpy >= 1.20.0
scipy >= 1.7.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
tqdm >= 4.62.0
```text

Install via:
```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn tqdm
```text


## Relation to Other Protocols


### Protocol 1 (Synthetic Neural Data)


- **Protocol 1** tests whether machine learning can distinguish APGI from competing models
- **Protocol 4** tests whether APGI dynamics exhibit phase transition signatures

Both provide **orthogonal evidence**: ML classification tests predictive accuracy, information theory tests mechanistic properties.


### Protocol 2 (Bayesian Model Comparison)


- **Protocol 2** uses empirical data fitting to compare models
- **Protocol 4** uses dynamical analysis to test theoretical predictions

Protocol 4 can be applied to **fitted parameters** from Protocol 2 to test whether real neural dynamics show phase transition signatures.


### Protocols 7-8 (Interventions & Individual Differences)


- **Protocols 7-8** test behavioral/physiological predictions
- **Protocol 4** tests computational/information-theoretic predictions

Together, these span multiple **levels of analysis** (Marr's hierarchy): computational theory, algorithmic implementation, and neural substrate.


## Advanced Usage


### Custom Information Measures


```python

# Add custom measure


class CustomInfoAnalyzer(InformationTheoreticAnalysis):
    def compute_renyi_entropy(self, series, alpha=2.0):
        """Rényi entropy of order α"""
        binned = self._discretize(series)
        counts = np.bincount(binned)
        probs = counts / counts.sum()
        probs = probs[probs > 0]

        if alpha == 1.0:
            return -np.sum(probs * np.log(probs))  # Shannon
        else:
            return (1/(1-alpha)) * np.log(np.sum(probs**alpha))
```text


### Real Data Application


```python

# Load empirical EEG/MEG data


eeg_data = load_eeg_data('subject01_task.fif')


# Extract surprise proxy (e.g., P3 amplitude)


S_empirical = extract_p3_amplitude(eeg_data)


# Extract threshold proxy (e.g., baseline + noise)


theta_empirical = estimate_threshold(eeg_data)


# Detect ignition events (e.g., P3b peaks)


ignition_events = detect_p3b_peaks(eeg_data)


# Apply phase transition analysis


detector = PhaseTransitionDetector()
results = detector.detect_discontinuity(
    S_empirical, theta_empirical, time, ignition_events
)
```text


### Sensitivity Analysis


```python

# Test robustness to parameter variation


param_ranges = {
    'tau': np.linspace(0.1, 0.3, 10),
    'theta_0': np.linspace(0.4, 0.7, 10),
    'alpha': np.linspace(3.0, 7.0, 10)
}

sensitivity_results = {}

for param_name, param_values in param_ranges.items():
    sensitivity_results[param_name] = []

    for param_value in param_values:
        # Create system with varied parameter
        kwargs = {'tau': 0.2, 'theta_0': 0.55, 'alpha': 5.0}
        kwargs[param_name] = param_value

        system = APGIDynamicalSystem(**kwargs)
        history = system.simulate(100.0, input_generator)

        analyzer = ComprehensivePhaseTransitionAnalysis()
        results = analyzer.analyze_simulation(history)

        sensitivity_results[param_name].append(results)
```text


## Troubleshooting


### Issue: Low ignition rate


**Symptom:** Few or no ignition events in simulations
### Solution:
- Increase input magnitude (higher ε_e, ε_i)
- Decrease threshold θ₀
- Increase β (somatic bias)


### Issue: Constant ignition


**Symptom:** System always above threshold
### Solution:
- Decrease input magnitude
- Increase threshold θ₀
- Increase decay rate (lower τ)


### Issue: NaN in Hurst exponent


**Symptom:** Hurst calculation returns NaN
### Solution:
- Ensure sufficient data points (>50)
- Check for constant segments (add noise)
- Increase simulation duration


### Issue: Low Φ values


**Symptom:** Integrated information near zero
### Solution:
- Increase coupling between variables
- Use longer integration windows
- Verify variables aren't independent


## Citation


If you use Protocol 4 in your research, please cite:

```text
APGI Framework (2026). Protocol 4: Information-Theoretic
Phase Transition Analysis. APGI Framework Testing Suite.
```text


## References


### Information Theory


- Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*
- Cover, T.M. & Thomas, J.A. (2006). *Elements of Information Theory*
- Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*


### Phase Transitions


- Stanley, H.E. (1971). *Introduction to Phase Transitions and Critical Phenomena*
- Sornette, D. (2006). *Critical Phenomena in Natural Sciences*
- Scheffer, M. et al. (2009). Early-warning signals for critical transitions. *Nature*


### Consciousness & Information


- Tononi, G. (2008). Consciousness as integrated information. *Biological Bulletin*
- Dehaene, S. & Changeux, J.P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron*
- Seth, A.K. (2021). Being You: A New Science of Consciousness


### Hurst Analysis


- Hurst, H.E. (1951). Long-term storage capacity of reservoirs. *Transactions of the American Society of Civil Engineers*
- Mandelbrot, B.B. & Wallis, J.R. (1969). Robustness of the rescaled range R/S


## Support


For questions, issues, or contributions:
- GitHub: [APGI Framework Repository]
- Email: apgi-research@anthropic.com
- Documentation: https://apgi-framework.readthedocs.io


## License


MIT License - see LICENSE file for details.

---

**Protocol 4 provides the computational fingerprint of conscious ignition as a phase transition. Use it to test whether awareness emerges through discrete state change or continuous amplification.**
