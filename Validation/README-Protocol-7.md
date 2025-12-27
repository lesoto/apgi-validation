# APGI Protocol 7: Complete Implementation Guide

## Overview

This is a complete implementation of Protocol 7 from the APGI framework. The protocol tests **causal predictions** about how perturbing APGI parameters affects conscious access through TMS, pharmacological interventions, or other experimental manipulations.

## Core Approach

Protocol 7 tests interventional predictions:
1. Defines how each intervention (TMS target, drug) affects specific APGI parameters
2. Simulates intervention studies with realistic noise and individual differences
3. Analyzes psychometric curves before/after intervention
4. Tests 5 falsification criteria about predicted effects

## What This Script Does

### 1. Intervention Models

**TMS Interventions:**
- `dlPFC_TMS`: Increases external precision (Π_e) via attentional modulation
- `Insula_TMS`: Increases interoceptive precision (Π_i)
- `V1_TMS`: Decreases threshold (direct cortical excitation)
- `Vertex_TMS`: Control condition (null effect)

**Pharmacological Interventions:**
- `Propranolol`: Decreases Π_i (blocks peripheral β-receptors)
- `Methylphenidate`: Increases Π_e (dopamine/NE enhancement)
- `Ketamine`: Decreases threshold (NMDA antagonism)
- `Placebo`: Control condition

### 2. Time Course Modeling

Each intervention has realistic pharmacokinetics/pharmacodynamics:

```python
effect(t) = Gamma_pdf(t - onset, k, θ) × effect_size × direction
```

Parameters:
- **onset_time**: When effect begins (min)
- **peak_time**: When effect is maximal (min)
- **duration**: How long effect lasts (min)

### 3. Study Designs

**Crossover Design** (within-subject):
- Each subject receives both intervention and control
- Order randomized
- Eliminates between-subject variance
- Requires smaller N

**Parallel Groups** (between-subject):
- Half get intervention, half get control
- Simpler but requires larger N
- Useful when carryover effects possible

### 4. Psychometric Analysis

Fits logistic curves to detection data:

```
P(seen) = λ + (1 - 2λ) / (1 + exp(-β(x - α)))
```

Where:
- α = threshold (50% detection point)
- β = slope (sensitivity)
- λ = lapse rate (error rate)

### 5. Power Analysis

Computes required sample size for detecting predicted effects:

```
N = f(effect_size, α, power)
```

Helps determine feasibility before running expensive experiments.

## Installation

```bash
# Core dependencies
pip install numpy pandas scipy matplotlib seaborn statsmodels

# For advanced power analysis
pip install pingouin
```

## Quick Start

```python
# Run the complete pipeline
python APGI-Protocol-3-Complete.py
```

The script will:
1. Define 8 interventions (4 TMS, 4 pharmacological)
2. Run power analysis for each
3. Simulate dlPFC TMS crossover study (N=24)
4. Simulate propranolol study (N=30)
5. Analyze psychometric curves
6. Test falsification criteria
7. Generate comprehensive visualizations

**Expected Runtime:**
- Intervention definition: <1 second
- Power analysis: ~5 seconds
- Study simulation: ~10 seconds per study
- Psychometric fitting: ~5 seconds
- Visualization: ~10 seconds
- **Total**: ~1 minute

## Defining Custom Interventions

### Example: rTMS to Dorsal ACC

```python
from scipy import stats

class CustomIntervention:
    @staticmethod
    def dacc_rtms() -> InterventionEffect:
        """
        Dorsal ACC rTMS
        
        Target: Decreases threshold (error monitoring enhancement)
        Mechanism: Increases conflict detection sensitivity
        """
        return InterventionEffect(
            name="dACC_rTMS",
            target_parameter="theta",
            effect_size=-0.35,  # Medium threshold reduction
            effect_direction="decrease",
            onset_time=0.0,
            peak_time=8.0,      # Peaks at 8 minutes
            duration=25.0,      # Lasts 25 minutes
            effect_se=0.18
        )

# Use in simulation
simulator = InterventionStudySimulator()
data = simulator.simulate_crossover_study(
    intervention=CustomIntervention.dacc_rtms(),
    control=TMSInterventions.vertex_tms(),
    n_subjects=20
)
```

### Example: Oxytocin Nasal Spray

```python
def oxytocin_intervention() -> InterventionEffect:
    """
    Intranasal oxytocin
    
    Target: Increases somatic bias (β) toward social signals
    Mechanism: Modulates interoceptive weighting in social contexts
    """
    return InterventionEffect(
        name="Oxytocin",
        target_parameter="beta",
        effect_size=0.45,
        effect_direction="increase",
        onset_time=40.0,      # 40 min onset (intranasal)
        peak_time=80.0,       # 80 min peak
        duration=180.0,       # 3 hours
        effect_se=0.20
    )
```

## Analyzing Real Data

### Data Format

Your intervention data should have this structure:

```python
import pandas as pd

real_data = pd.DataFrame({
    'subject_id': [0, 0, 0, 1, 1, 1, ...],
    'condition': ['baseline', 'baseline', 'intervention', ...],
    'stimulus_level': [0.3, 0.5, 0.7, ...],
    'n_trials': [25, 25, 25, ...],
    'n_seen': [8, 15, 22, ...]  # Number detected
})
```

### Fit Psychometric Curves

```python
psychometric = PsychometricCurve()

# Separate baseline and intervention
baseline = real_data[real_data['condition'] == 'baseline']
intervention = real_data[real_data['condition'] == 'intervention']

# Group by stimulus level
baseline_grouped = baseline.groupby('stimulus_level').agg({
    'n_seen': 'sum',
    'n_trials': 'sum'
})

# Fit
baseline_params = psychometric.fit_curve(
    stimulus_levels=baseline_grouped.index.values,
    n_trials=baseline_grouped['n_trials'].values,
    n_correct=baseline_grouped['n_seen'].values
)

# Same for intervention...

# Compare
comparison = psychometric.compare_curves(baseline_params, intervention_params)

print(f"Threshold shift: {comparison['threshold_shift']:.3f}")
print(f"P-value: {comparison['threshold_p']:.4f}")
```

## Expected Results

### Predicted Effect Sizes (Cohen's d)

| Intervention | Parameter | Effect | Predicted d |
|--------------|-----------|--------|-------------|
| dlPFC TMS | Π_e | Increase | +0.5 |
| Insula TMS | Π_i | Increase | +0.7 |
| V1 TMS | θ | Decrease | -0.4 |
| Propranolol | Π_i | Decrease | -0.6 |
| Methylphenidate | Π_e | Increase | +0.8 |
| Ketamine (low) | θ | Decrease | -0.9 |

### Threshold Shifts

Based on APGI theory:

**Threshold-Reducing Interventions:**
- V1 TMS: -0.08 to -0.12 (threshold units)
- Ketamine: -0.15 to -0.25

**Precision-Enhancing Interventions:**
- dlPFC TMS: Threshold shift -0.05 to -0.10 (indirect via Π_e)
- Methylphenidate: Threshold shift -0.08 to -0.15

**Precision-Reducing Interventions:**
- Propranolol: Threshold shift +0.06 to +0.12 (raises threshold)

### Statistical Power

Required sample sizes for 80% power (α = 0.05, two-tailed):

```
Small effect (d = 0.3):   N ≈ 90 per group
Medium effect (d = 0.5):  N ≈ 34 per group
Large effect (d = 0.8):   N ≈ 15 per group
```

For crossover designs (within-subject), divide by ~2.

## Output Files

### 1. Results JSON

**protocol3_results.json**
```json
{
  "interventions": {
    "dlPFC_TMS": {
      "target": "Pi_e",
      "effect_size": 0.5,
      "direction": "increase"
    }
  },
  "dlpfc_tms": {
    "baseline": {
      "threshold": 0.498,
      "slope": 5.23,
      "lapse": 0.019
    },
    "intervention": {
      "threshold": 0.437,
      "slope": 5.89,
      "lapse": 0.021
    },
    "comparison": {
      "threshold_shift": -0.061,
      "threshold_p": 0.0023
    }
  }
}
```

### 2. Data CSVs

**protocol3_dlpfc_data.csv**
```
subject_id,session,condition,stimulus_level,n_trials,n_seen,p_seen
0,0,control,0.3,12,3,0.25
0,0,control,0.5,13,8,0.62
0,1,intervention,0.3,12,5,0.42
...
```

### 3. Visualizations

**protocol3_dlpfc_results.png**

6-panel figure showing:
1. **Psychometric curves**: Baseline vs intervention
2. **Threshold shift**: Bar chart with error bars
3. **Subject-level effects**: Paired comparison plot
4. **Effect size distribution**: Histogram of individual effects
5. **Statistical summary**: Text table with t-test results
6. **Power analysis**: Power curve across sample sizes

## Console Output

```
================================================================================
APGI Protocol 7: TMS/PHARMACOLOGICAL INTERVENTION PREDICTIONS
================================================================================

STEP 1: DEFINING INTERVENTIONS
Registered 8 interventions:
  - dlPFC_TMS: Pi_e (increase, d=0.50)
  - Insula_TMS: Pi_i (increase, d=0.70)
  - Propranolol: Pi_i (decrease, d=-0.60)
  ...

STEP 2: STATISTICAL POWER ANALYSIS
Required sample sizes (80% power, α=0.05):
Intervention         Effect Size    Required N
--------------------------------------------------
dlPFC_TMS                  0.50            34
Insula_TMS                 0.70            22
Propranolol               -0.60            24

STEP 3: SIMULATING dlPFC TMS STUDY
✅ Simulated dlPFC TMS crossover study
   Subjects: 24
   Total trials: 38400

STEP 4: ANALYZING INTERVENTION EFFECTS

Baseline (Vertex Control):
  Threshold: 0.4983 ± 0.0142
  Slope: 5.234

Intervention (dlPFC TMS):
  Threshold: 0.4371 ± 0.0138
  Slope: 5.891

Comparison:
  Threshold shift: -0.0612 ± 0.0198
  Z-score: -3.09
  P-value: 0.0020
  Significant: ✅ YES

STEP 6: FALSIFICATION ANALYSIS
dlPFC TMS Falsification Report:
  Overall: ✅ VALIDATED
  Passed: 1
  Failed: 0

  ✅ F3.1: dlPFC TMS fails to shift threshold by >0.05
     shift: -0.0612
     magnitude_sufficient: True

Protocol 7 EXECUTION COMPLETE
```

## Advanced Usage

### Time-Resolved Analysis

Track intervention effects over time:

```python
intervention = PharmacologicalInterventions.methylphenidate()

# Time points: 0, 30, 60, 90, 120 minutes
time_points = np.array([0, 30, 60, 90, 120])

# Compute effect at each time
effects = intervention.compute_time_course(time_points)

# Simulate at each time point
results = []
for t, effect in zip(time_points, effects):
    # Modify simulator to use specific effect magnitude
    data = simulator.simulate_crossover_study(...)
    results.append(data)

# Plot time course
plt.plot(time_points, threshold_shifts)
plt.xlabel('Time (min)')
plt.ylabel('Threshold Shift')
```

### Dose-Response Curves

Test multiple doses:

```python
doses = [0.0, 0.5, 1.0, 2.0]  # mg/kg
effects = []

for dose in doses:
    # Scale effect by dose
    intervention = InterventionEffect(
        name=f"Propranolol_{dose}mg",
        target_parameter="Pi_i",
        effect_size=-0.6 * (dose / 2.0),  # Linear scaling
        ...
    )
    
    data = simulator.simulate_crossover_study(intervention, ...)
    effects.append(fit_and_extract_effect(data))

# Fit dose-response curve
slope, intercept, r, p, se = stats.linregress(doses, effects)
print(f"Dose-response slope: {slope:.3f} (r²={r**2:.3f}, p={p:.4f})")
```

### Interaction Effects

Test if interventions interact:

```python
# Simulate combined intervention
combined_effect = InterventionEffect(
    name="Combined",
    target_parameter="theta",
    effect_size=tms_effect.effect_size + drug_effect.effect_size,  # Additive
    # OR
    effect_size=tms_effect.effect_size * 1.5,  # Synergistic
    ...
)

data_combined = simulator.simulate_crossover_study(combined_effect, ...)

# Compare to sum of individual effects
if observed_combined > (effect_tms + effect_drug):
    print("Synergistic interaction detected")
elif observed_combined < (effect_tms + effect_drug):
    print("Antagonistic interaction detected")
```

### Individual Differences Analysis

Correlate intervention effects with baseline traits:

```python
# Simulate with varying baseline thresholds
simulator = InterventionStudySimulator()

baseline_thresholds = np.random.normal(0.5, 0.15, 30)
intervention_effects = []

for baseline in baseline_thresholds:
    # Modify simulator to use specific baseline
    # ... simulate and extract effect
    intervention_effects.append(effect)

# Test correlation
r, p = stats.pearsonr(baseline_thresholds, intervention_effects)

if p < 0.05 and r < -0.3:
    print("Subjects with higher baseline thresholds show larger reductions")
```

## Troubleshooting

### Psychometric Curve Won't Fit

**Problem**: Optimization fails or returns implausible parameters

**Causes**:
- Insufficient data (too few trials)
- Flat psychometric function (floor/ceiling effects)
- Bad initial guess

**Solutions**:

```python
# Provide better initial guess
initial_guess = [
    0.5,   # threshold (center of stimulus range)
    3.0,   # slope (steeper if data is steep)
    0.01   # lapse (small)
]

params = psychometric.fit_curve(
    stimulus_levels,
    n_trials,
    n_correct,
    initial_guess=initial_guess
)

# Or use constrained optimization
def constrained_fit(x, n_trials, n_correct):
    threshold, slope = x  # Fix lapse at 0.02
    ...
    
result = minimize(nll, [0.5, 5.0], bounds=[(0.1, 0.9), (0.5, 20)])
```

### Effect Not Significant

**Problem**: Predicted effect not detected (p > 0.05)

**Causes**:
- Underpowered study (N too small)
- High inter-subject variability
- Weak intervention effect

**Solutions**:

```python
# Run post-hoc power analysis
observed_d = (mean_intervention - mean_baseline) / pooled_sd

from statsmodels.stats.power import ttest_power

achieved_power = ttest_power(
    effect_size=observed_d,
    nobs=n_subjects,
    alpha=0.05,
    alternative='two-sided'
)

print(f"Achieved power: {achieved_power:.2f}")

if achieved_power < 0.50:
    print("Study severely underpowered")
    required_n = power_analyzer.compute_required_n(observed_d, power=0.80)
    print(f"Need N={required_n} for adequate power")
```

### Simulated Data Too Clean

**Problem**: Simulated effects are unrealistically large/consistent

**Solution**: Add more realistic noise sources

```python
class RealisticSimulator(InterventionStudySimulator):
    
    def simulate_crossover_study(self, ...):
        # Base simulation
        data = super().simulate_crossover_study(...)
        
        # Add session effects
        session_effect = self.rng.normal(0, 0.03, n_subjects)
        
        # Add practice effects
        practice_effect = np.linspace(0, -0.05, n_sessions)
        
        # Add fatigue
        if session > 60_minutes:
            fatigue_effect = 0.02 * (session - 60) / 60
        
        # Combine
        total_effect = intervention_effect + session_effect + practice_effect + fatigue_effect
        
        return data
```

## Scientific Interpretation

### If F3.1 Falsified (Threshold shift too small)

**Implication**: TMS to dlPFC doesn't causally reduce ignition threshold as predicted.

**Possible explanations**:
1. dlPFC not critically involved in threshold setting
2. TMS parameters suboptimal (wrong frequency/intensity)
3. Threshold is set by distributed network, not focal region
4. Timing mismatch (TMS effect has dissipated before testing)

**Action**: 
- Test alternative TMS targets (TPJ, ACC, FEF)
- Optimize TMS parameters via parameter space search
- Use concurrent TMS-EEG to verify target engagement

### If F3.2 Falsified (Propranolol doesn't reduce interoceptive influence)

**Implication**: Peripheral β-blockade doesn't affect interoceptive precision weighting.

**Possible explanations**:
1. Interoceptive precision is centrally determined
2. Propranolol doesn't cross BBB sufficiently
3. Β-receptors not critical for interoceptive signals
4. Compensatory mechanisms mask effect

**Action**:
- Test central β-blocker (e.g., propranolol with longer duration)
- Use direct interoceptive manipulation (heartbeat tracking task)
- Test on populations with heightened interoception (anxiety patients)

### If F3.3 Falsified (Wrong direction)

**Implication**: Intervention has opposite effect than predicted.

**Possible explanations**:
1. Theoretical mapping incorrect (parameter assignment wrong)
2. Unexpected compensatory mechanisms
3. Dose in paradoxical range
4. Measurement error in outcome

**Action**:
- Replicate with independent measure
- Test dose-response curve (check for inverted-U)
- Verify intervention actually engages target (manipulation check)

### If F3.4 Falsified (No dose-response)

**Implication**: Effect doesn't scale with intervention intensity.

**Possible explanations**:
1. Ceiling effect (maximal effect at lowest dose)
2. Floor effect (threshold mechanism saturated)
3. Non-linear dose-response (sigmoid)
4. Optimal dose window (inverted-U)

**Action**:
- Expand dose range (test lower and higher)
- Fit non-linear dose-response curves
- Use biomarkers to verify dose-dependent target engagement

### If F3.5 Falsified (Placebo = Active)

**Implication**: Observed effects are placebo-driven, not mechanism-specific.

**Possible explanations**:
1. Strong expectancy effects in conscious perception
2. Active control inadequate (not matched for expectancy)
3. True effect small relative to placebo
4. Inadequate blinding

**Action**:
- Use active placebo (mimics side effects)
- Measure expectancy explicitly and control statistically
- Use implicit measures less susceptible to demand
- Pharmacological manipulation checks (measure plasma levels)

## Computational Requirements

### Memory
- Minimal: ~100 MB for simulations
- Analysis: ~500 MB with full psychometric fitting

### Time
- Single simulation: ~1 second
- 100 simulations (power analysis): ~2 minutes
- Full protocol (8 interventions): ~5 minutes

### Parallelization

For large parameter sweeps:

```python
from multiprocessing import Pool

def simulate_one_condition(params):
    intervention, n_subjects, seed = params
    simulator = InterventionStudySimulator(seed=seed)
    return simulator.simulate_crossover_study(intervention, ...)

# Parallel execution
with Pool(8) as pool:
    results = pool.map(simulate_one_condition, parameter_combinations)
```

## Integration with Other Protocols

Protocol 7 complements other protocols:

**Protocol 1 → Protocol 7:**
- Use Protocol 1 to generate neural signatures
- Use Protocol 7 to predict how interventions alter signatures

**Protocol 2 → Protocol 7:**
- Use Protocol 2 to estimate baseline APGI parameters
- Use Protocol 7 to predict intervention effects on those parameters

**Protocol 7 → Clinical Translation:**
- Use Protocol 7 to identify effective interventions
- Design clinical trials based on power analyses
- Optimize dosing and timing

## Citation

If you use this implementation, please cite:

```
APGI Framework: Allostatic Precision-Gated Ignition
Protocol 7: TMS/Pharmacological Intervention Predictions
Implementation Version 1.0
2025
```

## References

**TMS Literature:**
- Thut & Pascual-Leone (2010). "A review of combined TMS-EEG studies to characterize lasting effects of repetitive TMS." *Brain Topography*.
- Romei et al. (2016). "Information-based approaches of noninvasive transcranial brain stimulation." *Trends in Neurosciences*.

**Pharmacology:**
- Lane & Green (2014). "Systematic review of propranolol." *Anxiety Disorders Review*.
- Faraone (2018). "The pharmacology of amphetamine and methylphenidate." *Neuropharmacology*.

**Psychophysics:**
- Wichmann & Hill (2001). "The psychometric function: I. Fitting, sampling, and goodness of fit." *Perception & Psychophysics*.
- Treutwein (1995). "Adaptive psychophysical procedures." *Vision Research*.

## License

This implementation is provided for scientific research purposes.

---

**Note**: This protocol requires careful experimental design. Always validate 
predictions with pilot data before large-scale studies.
