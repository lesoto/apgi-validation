# APGI Protocol 8: Psychophysical Threshold Estimation & Individual Differences

## Overview

Protocol 8 implements **adaptive psychophysical methods** and **individual differences analysis** to validate APGI framework's predictions about individual variability in conscious perception. This protocol bridges synthetic validation (Protocol 1) and real-data Bayesian comparison (Protocol 2) by providing tools for:

1. **Efficient threshold estimation** using Bayesian adaptive psychophysics (Psi method)
2. **APGI parameter estimation** from psychometric curve data
3. **Individual differences analysis** across physiological, cognitive, and clinical measures
4. **Test-retest reliability** assessment
5. **Factor analysis** of parameter structure
6. **Falsifiable predictions** about parameter relationships

## Key Innovation

Unlike standard psychophysics that only estimates threshold and slope, Protocol 8 extracts **all four APGI parameters** (θ₀, Π_i, β, α) from behavioral and physiological data, enabling systematic investigation of individual differences in conscious access mechanisms.

## Fatigue Guard Protocol

When subjects' θₜ rises too rapidly due to metabolic depletion during extended testing sessions, implement the following Fatigue Guard protocol to adjust trial counts and maintain data quality.

### Fatigue Detection Criteria

| Indicator | Threshold | Interpretation | Action |
| :--- | :--- | :--- | :--- |
| **Rapid θₜ Increase** | Δθₜ > +0.15 over 20 trials | Fatigue-induced threshold elevation | Trigger Fatigue Guard |
| **Response Time Increase** | RT > 150% of baseline | Cognitive slowing | Consider break |
| **Error Rate Spike** | Errors > 30% in last 10 trials | Performance degradation | Immediate break |
| **Pupil Dilation Drop** | Pupil < 80% of initial | Reduced arousal | End session |

### Trial Count Adjustment Protocol

```python
class FatigueGuard:
    """Implements fatigue-aware trial count adjustment."""

    BASELINE_TRIALS = 50  # Standard trial count

    def __init__(self, initial_theta_0):
        self.initial_theta = initial_theta_0
        self.trial_history = []
        self.theta_history = []

    def check_fatigue(self, current_theta, trial_number):
        """Detect fatigue based on threshold trajectory."""
        self.theta_history.append(current_theta)

        if len(self.theta_history) < 20:
            return False, self.BASELINE_TRIALS

        # Calculate recent trend (last 20 trials)
        recent_theta = np.mean(self.theta_history[-20:])
        baseline_theta = np.mean(self.theta_history[:20])
        theta_change = recent_theta - baseline_theta

        # Fatigue threshold: θₜ rises >0.15 units
        if theta_change > 0.15:
            # Calculate adjusted trial count
            excess_fatigue = (theta_change - 0.15) / 0.05  # per 0.05 excess
            reduced_trials = max(30, self.BASELINE_TRIALS - int(excess_fatigue * 10))

            return True, reduced_trials

        return False, self.BASELINE_TRIALS

    def get_recommended_break_interval(self, fatigue_detected):
        """Recommend break schedule based on fatigue status."""
        if fatigue_detected:
            return {
                'break_interval_trials': 15,  # Break every 15 trials
                'break_duration_sec': 60,     # 1 minute rest
                'max_total_trials': 40        # Cap at 40 trials
            }
        return {
            'break_interval_trials': 50,      # Standard: break at end
            'break_duration_sec': 120,
            'max_total_trials': 50
        }
```

### Implementation Guide

#### Step 1: Monitor θₜ Trajectory

```python
fatigue_guard = FatigueGuard(initial_theta_0=0.55)

for trial in range(max_trials):
    # Run trial and get current θₜ estimate
    current_theta = run_psi_trial()

    # Check for fatigue
    is_fatigued, adjusted_trials = fatigue_guard.check_fatigue(
        current_theta, trial
    )

    if is_fatigued:
        print(f"⚠️  FATIGUE DETECTED at trial {trial}")
        print(f"   Reducing trial count to {adjusted_trials}")
        print(f"   Inserting mandatory break")
        insert_break(duration=60)

    if trial >= adjusted_trials:
        print("Session complete (adjusted for fatigue)")
        break
```

#### Step 2: Post-Session Fatigue Correction

If fatigue was detected, apply correction to estimated parameters:

```python
def correct_for_fatigue(raw_theta_0, fatigue_detected, fatigue_severity):
    """
    Adjust estimated θ₀ to compensate for fatigue-induced elevation.

    Args:
        raw_theta_0: Estimated threshold from fatigued session
        fatigue_detected: Boolean from FatigueGuard
        fatigue_severity: θₜ elevation magnitude (Δθ)

    Returns:
        corrected_theta_0: Fatigue-corrected threshold estimate
    """
    if not fatigue_detected:
        return raw_theta_0

    # Correction factor: Assume fatigue elevates measured threshold
    # by ~60% of the observed θₜ increase
    correction = 0.6 * fatigue_severity
    corrected = raw_theta_0 - correction

    return max(0.25, min(0.85, corrected))  # Keep in valid range
```

### Reporting Fatigue Adjustments

When reporting results with Fatigue Guard intervention:

```text
Results (with Fatigue Guard):
- Initial trial target: 50
- Fatigue detected at trial 32 (Δθₜ = +0.18)
- Adjusted trial count: 40
- Breaks inserted: 2 × 60 seconds
- Fatigue-corrected θ₀: 0.52 (raw: 0.58)
- Reliability flag: Fatigue-adjusted data
```

---

## Installation

### Dependencies

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn statsmodels
```text

### Versions Tested

- Python 3.8+
- NumPy 1.21+
- SciPy 1.7+
- Pandas 1.3+
- Matplotlib 3.4+
- Seaborn 0.11+
- scikit-learn 0.24+
- statsmodels 0.13+

---

## Quick Start

### Run Complete Protocol

```bash
python APGI_Protocol_3.py
```text

This executes the full pipeline:
1. Simulates 50 participants with individual differences
2. Estimates psychometric functions using Psi method (50 trials each)
3. Extracts APGI parameters from curves
4. Analyzes correlations and relationships
5. Tests falsification criteria
6. Generates comprehensive visualizations

**Runtime:** ~5-10 minutes on standard laptop


### Expected Output Files

1. **protocol3_participant_data.csv** - Individual participant parameters and measures
2. **protocol3_results.json** - Complete analysis results and statistics
3. **protocol3_individual_differences.png** - Comprehensive visualization (20×14 inches)

---

## Theoretical Framework

### APGI Parameters

The framework models conscious access as threshold-crossing in a dynamical system:

```text
 S_t = Π_e· | ε_e | + β·Π_i· | ε_i | (Surprise accumulation)

P(conscious) = σ(α·(S_t - θ_t))  (Ignition probability)
```text

### Four core parameters:

| Parameter | Description | Typical Range | Meaning |
| --- | --- | --- | --- |
| **θ₀** | Baseline threshold | 0.25-0.75 | How much surprise needed for ignition |
| **Π_i** | Interoceptive precision | 0.5-2.5 | Weight on bodily signals |
| **β** | Somatic bias | 0.7-1.8 | Facilitation by interoception |
| **α** | Sigmoid steepness | 2.0-15.0 | Sharpness of ignition transition |

### Parameter Estimation Logic

### Direct mappings:
- θ₀ ≈ psychometric threshold (50% detection point)
- α ≈ psychometric slope (steepness)

### Inferred mappings:
- Π_i ← interoceptive measures (HEP amplitude, heartbeat detection, HRV)
- β ← relationship between Π_i and threshold modulation

---


## Key Predictions & Falsification Criteria


### Prediction P3a: Interoceptive Precision Correlates


**Claim:** Π_i should correlate with physiological interoceptive measures

| Measure | Predicted r | Threshold |
| --------- | ------------- | ----------- |
| HEP amplitude | > 0.40 | p < 0.01 |
| Heartbeat detection accuracy | > 0.35 | p < 0.01 |
| Heart rate variability (RMSSD) | > 0.30 | p < 0.05 |

**Falsification F3.1:** If r < 0.30 or p > 0.05 for HEP, interoceptive precision is not a valid construct.

---

### Prediction P3b: Threshold-Somatic Bias Relationship

**Claim:** Higher somatic bias (β) should predict lower threshold (θ₀)

```text
r(θ₀, β) < -0.25, p < 0.05
```text

**Rationale:** Somatic facilitation lowers the barrier for conscious access

**Falsification F3.2:** If r > 0 (positive correlation), the somatic bias mechanism is falsified.

---


### Prediction P3c: Test-Retest Reliability


**Claim:** Parameters should show good temporal stability

| Parameter | Minimum ICC |
| ----------- | ------------- |
| Threshold (θ₀) | 0.75 |
| Sigmoid steepness (α) | 0.70 |
| Interoceptive precision (Π_i) | 0.65 |

**Falsification F3.3:** If ICC < 0.60, parameters are too unstable to be meaningful.

---


### Prediction P3d: Parameter Independence


**Claim:** APGI parameters measure distinct constructs

 - Correlation between different parameters: | r | < 0.6
- At least 2 factors emerge in factor analysis
- Parameters don't collapse into single dimension

 **Falsification F3.4:** If only 1 factor or all | r | > 0.7, parameters are redundant.

---


### Prediction P3e: Factor Structure


### Expected structure:
- **Factor 1 (Threshold):** θ₀, α
- **Factor 2 (Interoceptive):** Π_i, β
- **Factor 3 (Cognitive):** attention, working memory

**Cumulative variance explained:** > 70%

---


### Prediction P3f: Clinical Correlates


### Anxiety & Interoception:

```text
r(Π_i, anxiety) < -0.25, p < 0.05
```text

*Higher anxiety → reduced interoceptive precision*

### Depression & Threshold:

```text
r(θ₀, depression) > 0.25, p < 0.05
```text

*Higher depression → elevated conscious access threshold*

 **Falsification F3.5:** If all | r | < 0.20, APGI parameters have no clinical relevance.

---


## Usage Examples


### Example 1: Simulate Single Participant


```python
from APGI_Protocol_3 import PsychophysicsStudySimulator

simulator = PsychophysicsStudySimulator(seed=42)

participant = simulator.simulate_participant(
    participant_id='P001',
    n_trials=50,
    use_psi_method=True
)

print(f"Estimated threshold: {participant.threshold:.3f}")
print(f"Interoceptive precision: {participant.Pi_i:.3f}")
print(f"Somatic bias: {participant.beta:.3f}")
```text


### Example 2: Adaptive Staircase (Psi Method)


```python
from APGI_Protocol_3 import PsiMethod


# Initialize Psi method


psi = PsiMethod(
    stimulus_range=(0.0, 1.0),
    threshold_range=(0.2, 0.8),
    slope_range=(1.0, 20.0)
)


# Run adaptive experiment


for trial in range(50):
    # Get next optimal stimulus
    stimulus = psi.select_next_stimulus()

    # Present stimulus and get response
    response = present_stimulus_and_get_response(stimulus)

    # Update posterior
    psi.update_posterior(stimulus, response)


# Get parameter estimates


estimates = psi.get_parameter_estimates()
print(f"Threshold: {estimates['threshold_mean']:.3f}")
print(f"95% CI: [{estimates['threshold_ci_low']:.3f}, {estimates['threshold_ci_high']:.3f}]")
```text


### Example 3: Estimate APGI Parameters


```python
from APGI_Protocol_3 import APGIParameterEstimator

estimator = APGIParameterEstimator()


# From psychometric curve


apgi_params = estimator.estimate_from_psychometric(
    threshold=0.52,
    slope=6.8,
    interoceptive_measure=0.75,  # Normalized HEP amplitude
    trial_variability=0.08
)

print(f"θ₀: {apgi_params['theta_0']:.3f}")
print(f"Π_i: {apgi_params['Pi_i']:.3f}")
print(f"β: {apgi_params['beta']:.3f}")
print(f"α: {apgi_params['alpha']:.3f}")
```text


### Example 4: Individual Differences Analysis


```python
from APGI_Protocol_3 import IndividualDifferencesAnalysis


# Load participant data


participants = load_participant_data()


# Create analysis object


analysis = IndividualDifferencesAnalysis(participants)


# Test interoceptive correlates


interoceptive_results = analysis.test_interoceptive_precision_correlates()


# Test threshold-beta relationship


threshold_beta = analysis.test_threshold_somatic_bias_relationship()


# Test-retest reliability


reliability = analysis.test_retest_reliability()


# Factor analysis


factors = analysis.factor_analysis()


# Clinical correlates


clinical = analysis.clinical_correlates()
```text

---


## Interpreting Results


### Participant Data CSV


### Columns:
- `participant_id`: Unique identifier
- `age`, `sex`: Demographics
- `threshold`, `slope`, `lapse`: Psychometric parameters
- `theta_0`, `Pi_i`, `beta`, `alpha`: APGI parameters
- `hrv_rmssd`: Heart rate variability (ms)
- `hep_amplitude`: Heartbeat-evoked potential (μV)
- `heartbeat_detection_accuracy`: Proportion correct (0-1)
- `attention_score`, `working_memory`: Cognitive measures
- `anxiety_score`, `depression_score`: Clinical measures (0-100)
- `threshold_session2`: Retest threshold


### Results JSON Structure


```json
{
  "config": {...},
  "n_participants": 50,
  "interoceptive_correlates": {
    "Pi_i_vs_HEP": {
      "r": 0.58,
      "p": 0.0001,
      "n": 50,
      "prediction_met": true
    }
  },
  "threshold_beta_relationship": {
    "r": -0.41,
    "p": 0.003,
    "prediction_met": true
  },
  "test_retest_reliability": {
    "icc": 0.82,
    "prediction_met": true
  },
  "falsification": {
    "overall_falsified": false,
    "passed_criteria": [...],
    "falsified_criteria": []
  }
}
```text


### Visualization Panels


### Row 1: Parameter Distributions
- Histograms with overlaid normal distributions
- Mean and SD for each parameter

### Row 2: Key Correlations
- Π_i vs HEP amplitude
- Π_i vs Heartbeat detection
- θ₀ vs β (somatic bias)
- Test-retest scatter

### Row 3: Clinical Correlates
- Anxiety vs Π_i
- Depression vs θ₀

### Row 4: Summary
- Parameter correlation heatmap
- Descriptive statistics table

---


## Statistical Power


### Required Sample Sizes (80% power, α=0.05)


| Effect | Expected r | Required N |
| -------- | ----------- | ------------ |
| Π_i vs HEP | 0.45 | 36 |
| θ₀ vs β | -0.30 | 64 |
| Test-retest | 0.75 ICC | 15 |

**Recommendation:** N ≥ 50 for comprehensive individual differences analysis

---


## Extending Protocol 8


### Add New Interoceptive Measures


```python

# In ParticipantData dataclass


@dataclass
class ParticipantData:
    # ... existing fields ...

    # Add new measure
    respiratory_sinus_arrhythmia: Optional[float] = None
    skin_conductance: Optional[float] = None
```text


### Implement Custom Psychometric Function


```python
class CustomPsychometric(PsiMethod):
    def psychometric_function(self, x, threshold, slope, lapse):
        # Implement your function
        # E.g., Weibull instead of logistic
        return ...
```text


### Add New Factor Analysis Variables


```python

# In IndividualDifferencesAnalysis.factor_analysis()


var_cols = ['theta_0', 'Pi_i', 'beta', 'alpha']


# Add your variables


var_cols.extend(['your_measure_1', 'your_measure_2'])
```text

---


## Troubleshooting


### Common Issues


### 1. "Insufficient data" errors
- **Cause:** Too few participants or missing measures
- **Solution:** Increase n_participants or reduce analysis scope

### 2. Psi method convergence slow
- **Cause:** Poor prior specification or noisy responses
- **Solution:** Adjust threshold_range based on pilot data

### 3. Factor analysis fails
- **Cause:** Multicollinearity or insufficient variance
- **Solution:** Check parameter correlations, remove redundant variables

### 4. Negative ICC values
- **Cause:** Session differences > individual differences
- **Solution:** Control experimental conditions better, increase trials

---


## Citation


If you use this protocol, please cite:

```bibtex
@software{apgi_protocol3_2026,
  title={APGI Protocol 8: Psychophysical Threshold Estimation and Individual Differences},
  author={APGI Framework},
  year={2026},
  url={https://github.com/apgi-framework/protocols}
}
```text

### Related Publications:
- Kontsevich & Tyler (1999). "Bayesian adaptive estimation of psychometric slope and threshold." *Vision Research*, 39(16), 2729-2737.
- Prins (2012). "The psi-marginal adaptive method: How to give nuisance parameters the attention they deserve." *Journal of Vision*, 12(6):3.

---


## FAQ


### Q: Can I use Protocol 8 with real human data?

A: Yes! Replace `simulate_participant()` with actual psychophysical data collection. The Psi method can be implemented in PsychoPy or MATLAB.

### Q: What's the difference between θ₀ and psychometric threshold?

A: They're approximately equal in APGI. θ₀ is the theoretical ignition threshold; psychometric threshold is its behavioral manifestation.

### Q: Why estimate Π_i from HEP instead of measuring directly?

A: Π_i is a computational construct (precision weight). HEP amplitude is its neural correlate. We use convergent validity across multiple measures.

### Q: Can parameters change over time?

A: Yes, especially Π_i (interoceptive precision) which fluctuates with arousal, attention, and clinical state. θ₀ and α are more trait-like.

### Q: What if test-retest reliability is low?

A: Could indicate: (1) true state changes, (2) measurement noise, or (3) context dependence. Increase trials per session or control experimental conditions.

---


## Future Directions


### Planned Enhancements


1. **Hierarchical Bayesian Estimation**
   - Full posterior distribution over parameters
   - Uncertainty quantification

2. **Longitudinal Analysis**
   - Parameter change over time
   - Intervention effects

3. **Multi-Context Estimation**
   - Parameters under different attention loads
   - Interoceptive vs. exteroceptive blocks

4. **Machine Learning Integration**
   - Predict parameters from multimodal data
   - Classification of clinical groups

5. **Real-Time Adaptive Testing**
   - Online parameter updates
   - Closed-loop threshold tracking

---


## License


MIT License - See LICENSE file for details

---


## Contact


For questions, issues, or contributions:
- GitHub Issues: [APGI Framework Repository]
- Email: apgi-framework@example.com

---


## Acknowledgments


- **Psi Method:** Kontsevich & Tyler (1999)
- **Psychophysics Theory:** Gescheider (1997), Kingdom & Prins (2016)
- **Individual Differences:** Underwood & Everatt (1992), Hedge et al. (2018)

