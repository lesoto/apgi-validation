# APGI Protocol 1: Complete Implementation Guide

This protocol tests falsifiable predictions about consciousness through synthetic neural data generation and machine learning classification.

## Core Functionality

1. **Generates Biophysically Realistic Neural Signals**
   - Multi-channel EEG (64 channels × 1000 timepoints)
   - Heartbeat-Evoked Potentials (HEP)
   - Pupil dilation responses
   - All signals generated from APGI dynamical equations

2. **Implements 4 Competing Theoretical Models**
   - **APGI**: Full framework with ignition threshold + interoceptive precision
   - **StandardPP**: Continuous predictive processing (no ignition)
   - **GWTOnly**: Global Workspace ignition without embodiment
   - **Continuous**: Graded consciousness without phase transition

3. **Trains Deep Learning Classifiers**
   - **Task 1A**: Binary ignition classification (conscious vs unconscious)
   - **Task 1B**: 4-class model identification (which theory generated this?)

4. **Tests Falsification Criteria**
   - F1.1: APGI ignition classification accuracy < 75% → Falsified
   - F1.2: APGI-GWT confusion > 40% → Falsified
   - F1.3: Generalization to real data < 55% → Falsified
   - F1.4: StandardPP ≥ APGI accuracy → Falsified

## Installation

```bash
pip install numpy scipy torch scikit-learn matplotlib seaborn tqdm
```

## Quick Start

```python
# Run the complete pipeline
python APGI_Protocol_1-Complete.py
```

The script will:

- Generate 20,000 synthetic trials (5,000 per model)
- Train and evaluate classifiers
- Check falsification criteria
- Generate comprehensive visualizations
- Save all results

## Expected Outputs

### Files Created

- **apgi_protocol1_dataset.npz**
  - Compressed dataset with all synthetic neural signals
  - ~500 MB for 20,000 trials

- **protocol1_results.json**
  - Quantitative metrics for all tasks
  - Falsification criterion values
  - Per-model performance

- **protocol1_results.png**
  - Comprehensive visualization dashboard
  - Accuracy comparisons
  - Confusion matrices
  - ROC curves

- **best_apgi_model_epoch*.pth**
  - Trained model checkpoints
  - Can be loaded for inference

### Console Output

```text
================================================================================
APGI PROTOCOL 1: SYNTHETIC DATA GENERATION & ML CLASSIFICATION
================================================================================

STEP 1: GENERATING SYNTHETIC DATASET
  - Detection: 8 intensity levels × 25 trials per subject
  - Heartbeat: 60 trials per subject with HEP/pupil signals
  - Oddball: 30 intero + 30 extero deviants per subject

STEP 2: TASK 1A - BINARY IGNITION CLASSIFICATION
  Training classifier for APGI...
  Epoch 50/50 | Train Acc: 91.2% | Val Acc: 89.8% | Val AUC: 0.9234

STEP 3: TASK 1B - MULTI-MODAL MODEL IDENTIFICATION
  Overall Accuracy: 0.823

STEP 4: FALSIFICATION ANALYSIS
  F1.1 PASSED: APGI accuracy = 0.898 (threshold: 0.75)
  F1.2 PASSED: APGI-GWT confusion = 0.14 (threshold: 0.40)

OVERALL STATUS: MODEL VALIDATED
```

## Architecture Details

### APGI Dynamical System

Core equation implemented:

```python
dS/dt = -S/τ + Π_e·|ε_e| + β·Π_i·|ε_i|
```

Where:

- S(t): Accumulated surprise
- Π_e, Π_i: Extero/interoceptive precision
- ε_e, ε_i: Prediction errors
- β: Somatic bias parameter
- τ = 0.2s: Decay constant

### Neural Network Architectures

#### Ignition Classifier (Task 1A)

- Input: 64 × 1000 EEG
- Multi-scale 1D convolutions (kernels: 25, 15, 7)
- Temporal attention mechanism
- Output: Binary (ignition vs no ignition)
- Parameters: ~2.5M

#### Multi-Modal Fusion (Task 1B)

- Inputs: EEG (64×1000) + HEP (600) + Pupil (3000)
- Separate encoders per modality
- Late fusion with learned weights
- Output: 4-class (model identification)
- Parameters: ~1.8M

## Predicted Results

Based on APGI theory, we predict:

| Metric | APGI | StandardPP | GWTOnly | Continuous |
| --- | --- | --- | --- | --- |
| Accuracy (Task 1A) | 85-92% | 60-70% | 75-82% | 55-65% |
| F1 Score | 0.85-0.90 | 0.58-0.68 | 0.73-0.80 | 0.50-0.62 |
| AUC-ROC | 0.90-0.95 | 0.65-0.75 | 0.80-0.87 | 0.55-0.68 |

### Confusion Matrix (Task 1B)

- APGI correctly identified: 80-88%
- APGI ↔ GWT confusion: 12-18% (both have ignition)
- APGI ↔ StandardPP: < 5% (clearly distinct)

## Customization

### Adjust Dataset Size

```python
# In main(), modify:
config = {
    'n_trials_per_model': 10000,  # Increase for better statistics
    ...
}
```

### Change Training Parameters

```python
config = {
    'epochs_task_1a': 100,  # More epochs
    'learning_rate': 5e-5,  # Lower learning rate
    'batch_size': 64,       # Larger batches
    ...
}
```

### Generate Specific Model Data Only

```python
generator = APGIDatasetGenerator(fs=1000)

# Generate only APGI trials
params = generator.sample_physiological_parameters()
trial = generator._generate_apgi_trial(params)
```

## Advanced Usage

### Load and Analyze Saved Dataset

```python
import numpy as np

# Load dataset
data = np.load('apgi_protocol1_dataset.npz')

print(f"EEG shape: {data['eeg'].shape}")
print(f"Models: {np.unique(data['model_names'])}")
print(f"Ignition rate per model:")
for i, model in enumerate(['APGI', 'StandardPP', 'GWTOnly', 'Continuous']):
    mask = data['model_labels'] == i
    ignition_rate = data['ignition_labels'][mask].mean()
    print(f"  {model}: {ignition_rate:.1%}")
```

### Visualize Single Trial

```python
import matplotlib.pyplot as plt

# Load first APGI trial
apgi_mask = data['model_labels'] == 0
eeg_trial = data['eeg'][apgi_mask][0]  # First APGI trial
hep_trial = data['hep'][apgi_mask][0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# Plot EEG at Pz (channel 31)
ax1.plot(eeg_trial[31], linewidth=0.5)
ax1.set_title('EEG at Pz')
ax1.set_ylabel('Amplitude (μV)')

# Plot HEP
ax2.plot(hep_trial, linewidth=1.5, color='red')
ax2.set_title('Heartbeat-Evoked Potential')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Amplitude (μV)')

plt.tight_layout()
plt.show()
```

### Test on Custom Parameters

```python
from APGI_Protocol_1_Complete import APGIDynamicalSystem, APGISyntheticSignalGenerator

# Create custom trial
system = APGIDynamicalSystem()
generator = APGISyntheticSignalGenerator(fs=1000)

# High precision, low threshold → should ignite
S_traj, B_traj, ignited = system.simulate_surprise_accumulation(
    epsilon_e=0.3,      # Moderate prediction error
    epsilon_i=0.2,
    Pi_e=2.5,           # High precision
    Pi_i=2.0,
    beta=1.2,           # Moderate somatic bias
    theta_t=0.4         # Low threshold (easy to ignite)
)

print(f"Ignition occurred: {ignited}")
print(f"Max surprise: {S_traj.max():.3f}")

# Generate corresponding EEG
eeg = generator.generate_multi_channel_eeg(
    S_t=S_traj.max(),
    theta_t=0.4,
    ignition=ignited,
    n_channels=64
)
```

## Computational Requirements

### Memory

- **Dataset generation**: ~4 GB RAM
- **Training Task 1A**: ~6 GB GPU / 12 GB CPU
- **Training Task 1B**: ~8 GB GPU / 16 GB CPU

### Time

On NVIDIA RTX 3090:

- Dataset generation (20K trials): ~15 minutes
- Task 1A training (all models): ~2 hours
- Task 1B training: ~1.5 hours
- **Total**: ~4 hours

On CPU (16-core):

- Dataset generation: ~25 minutes
- Task 1A training: ~8 hours
- Task 1B training: ~6 hours
- **Total**: ~15 hours

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
config['batch_size'] = 16

# Or reduce dataset size
config['n_trials_per_model'] = 2500  # 10K total
```

### Poor Convergence

```python
# Increase training epochs
config['epochs_task_1a'] = 100

# Lower learning rate
config['learning_rate'] = 5e-5

# Reduce dropout
classifier = IgnitionClassifier(dropout=0.3)
```

### NaN in Pupil Data

Blinks create NaN values - this is intentional and realistic:

```text
# Already handled in network with:
pupil = torch.nan_to_num(pupil, nan=0.0)
```

## Scientific Interpretation

### If F1.1 Falsified (APGI accuracy < 75%)

**Implication**: The P3b measurement equation doesn't capture ignition signatures.

**Action**: Revise how EEG relates to surprise accumulation.

### If F1.2 Falsified (APGI-GWT confusion > 40%)

**Implication**: Interoceptive precision doesn't distinguish APGI from GWT.

**Action**: HEP amplitude may not be a valid proxy for Π_i.

### If F1.3 Falsified (Real data accuracy < 55%)

**Implication**: Synthetic signals don't match real neural dynamics.

**Action**: Measurement equations need empirical calibration.

### If F1.4 Falsified (StandardPP ≥ APGI)

**Implication**: Ignition threshold provides no computational advantage.

**Action**: Core APGI mechanism may be incorrect.

## Integration with Other Protocols

This implementation provides the foundation for:

- **Protocol 2**: Psychometric parameter estimation (see `APGI_Parameter_Estimation-Protocol.py`)
- **Protocol 3**: Clinical diagnostic markers
- **Protocol 4**: Pharmacological interventions
- **Protocol 5**: Cross-species validation

## Citation

If you use this code, please cite:

```text
APGI Framework: Allostatic Precision-Gated Ignition
Protocol 1: Synthetic Neural Data Generation and Machine Learning Classification
```

## License

This implementation is provided for scientific research purposes.

## Contact

For questions about implementation details or theoretical aspects:

- Check the inline documentation

- Review the comprehensive comments

- Examine the example usage sections

---

### Note: This is synthetic data for testing theoretical predictions

Always validate with empirical data before drawing neuroscientific conclusions.
