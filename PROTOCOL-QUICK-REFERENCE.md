# Protocol Quick Reference

**Quick lookup table for all APGI protocols: new canonical naming, legacy files, and Python implementations**

---

## Protocol Lookup Table (All 47 Scripts)

### Canonical Protocols (APGI-P##)

| APGI ID | Title | New File | Legacy VP File | Legacy FP File | Python VP | Python FP | Status |
|---------|-------|----------|----------------|----------------|-----------|-----------|--------|
| **APGI-P00** | HEP Proxy Validation | `protocol_0_hep_proxy_validation.json` | `protocol_vp_00_hep_proxy_validation.json` | — | `VP_00_HEPProxyValidation.py` | — | ✅ Foundational |
| **APGI-P01** | Cardiac EEG | `protocol_1_cardiac_eeg.json` | `protocol_vp_01_synthetic_eeg_ml_classification.json` | `protocol_fp_09_neural_signatures_eeg_p3b_hep_apgi_p01_linked.json` | `VP_01_SyntheticEEGMLClassification.py`, `VP_09_NeuralSignaturesEmpiricalPriority1.py` | `FP_09_NeuralSignaturesP3bHEP.py` | ✅ Primary |
| **APGI-P02** | Somatic Agent Sim | `protocol_2_somatic_agent_sim.json` | `protocol_vp_03_active_inference_agent_simulations_apgi_p03_linked.json` | `protocol_fp_01_active_inference_agents_f1_f2.json` | `VP_03_ActiveInferenceAgentSimulations.py`, `VP_10_CausalManipulationsPriority2.py` | `FP_01_ActiveInference.py` | ✅ Primary |
| **APGI-P03** | fMRI Anticipation | `protocol_3_anticipation_fmri.json` | `protocol_vp_14_fmri_anticipation_vs_experience_apgi_p05_linked.json`, `protocol_vp_15_fmri_anticipation_vmpfc.json` | `protocol_fp_14_fmri_anticipation_vmpfc.json` | `VP_14_FMRIAnticipationExperience.py`, `VP_15_FMRIAnticipationVmPFC.py` | `FP_14_fMRI_Anticipation_vmPFC.py` | ✅ Secondary |
| **APGI-P04** | Metabolic Crossover | `protocol_4_metabolic_crossover.json` | `protocol_vp_12_clinical_cross_species_convergence_apgi_p04_linked.json`, `protocol_vp_16_metabolic_atp_ground_truth.json` | `protocol_fp_12_cross_species_scaling.json` | `VP_12_ClinicalCrossSpeciesConvergence.py`, `VP_16_MetabolicATPGroundTruth.py` | `FP_12_CrossSpeciesScaling.py` | ✅ Secondary |
| **APGI-P05** | Causal TMS | `protocol_5_causal_tms.json` | `protocol_vp_07_tms_causal_interventions_apgi_p02_linked.json`, `protocol_vp_22_enhanced_fmri_anticipation_vs_experience_apgi_p05_linked.json` | `protocol_fp_03_framework_level_multi_protocol.json` | `VP_07_TMSCausalInterventions.py`, `VP_22_FMRIAnticipationExperience.py` | `FP_03_FrameworkLevelMultiProtocol.py` | ✅ Secondary |
| **APGI-P06** | Ignition iEEG | `protocol_6_ignition_ieeg.json` | `protocol_vp_20_empirical_intracranial_eeg_apgi_p06_linked.json` | `protocol_fp_04_phase_transition_bistability_apgi_p06_linked.json`, `protocol_fp_11_liquid_network_dynamics_echo_state.json` | `VP_20_EmpiricalIEEG.py` | `FP_04_PhaseTransitionEpistemicArchitecture.py`, `FP_11_LiquidNetworkDynamicsEchoState.py` | ✅ Tertiary |
| **APGI-P07** | DOC Biomarker | `protocol_7_doc_biomarker.json` | `protocol_vp_13_epistemic_architecture.json` | `protocol_fp_13_clinical_cross_species_convergence.json` | `VP_13_EpistemicArchitecture.py` | `FP_13_Clinical_CrossSpecies_Convergence.py` | ✅ Tertiary |

### Non-Canonical Validation Protocols (VP-##)

| VP ID | Python File | Legacy File | Title | Notes |
|-------|-------------|------------|-------|-------|
| **VP-02** | `VP_02_BehavioralBayesianComparison.py` | `protocol_vp_02_behavioral_bayesian_model_comparison.json` | Behavioral Bayesian Comparison | No APGI-P equivalent |
| **VP-04** | `VP_04_PhaseTransitionEpistemicLevel2.py` | `protocol_vp_04_phase_transition_epistemic_level_2.json` | Phase Transition Epistemic Level 2 | No APGI-P equivalent |
| **VP-05** | `VP_05_EvolutionaryEmergence.py` | `protocol_vp_05_evolutionary_emergence.json` | Evolutionary Emergence | No APGI-P equivalent |
| **VP-06** | `VP_06_LiquidNetworkInductiveBias.py` | `protocol_vp_06_liquid_network_inductive_bias.json` | Liquid Network Inductive Bias | No APGI-P equivalent |
| **VP-07a** | `VP_07a_MathematicalConsistency.py` | `protocol_vp_07a_mathematical_consistency.json` | Mathematical Consistency (variant) | Variant of VP-07 |
| **VP-08** | `VP_08_PsychophysicalThresholdEstimation.py` | `protocol_vp_08_psychophysical_threshold_estimation.json` | Psychophysical Threshold Estimation | No APGI-P equivalent |
| **VP-11** | `VP_11_MCMCCulturalNeurosciencePriority3.py` | `protocol_vp_11_mcmc_cultural_neuroscience_priority_3.json` | MCMC Cultural Neuroscience Priority 3 | No APGI-P equivalent |
| **VP-17** | `VP_17_AllenVisualCodingFatigue.py` | `protocol_vp_17_allen_visual_coding_fatigue.json` | Allen Visual Coding Fatigue | No APGI-P equivalent |
| **VP-18** | `VP_18_EEGMicrostateGFPP3b.py` | `protocol_vp_18_eeg_microstate_gfp_p3b.json` | EEG Microstate GFP/P3b | Deprecated (overlaps VP-09) |
| **VP-19** | `VP_19_InformationErasureMVPA.py` | `protocol_vp_19_information_erasure_mvpa.json` | Information Erasure MVPA | No APGI-P equivalent |
| **VP-21** | `VP_21_FreeEnergyPredictionError.py` | `protocol_vp_21_free_energy_prediction_error.json` | Free Energy Prediction Error | No APGI-P equivalent |

### Non-Canonical Falsification Protocols (FP-##)

| FP ID | Python File | Legacy File | Title | Notes |
|-------|-------------|------------|-------|-------|
| **FP-02** | `FP_02_AgentComparisonConvergenceBenchmark.py` | `protocol_fp_02_agent_comparison_convergence_benchmark.json` | Agent Comparison Convergence | No APGI-P equivalent |
| **FP-05** | `FP_05_EvolutionaryPlausibility.py` | `protocol_fp_05_evolutionary_plausibility.json` | Evolutionary Plausibility | No APGI-P equivalent |
| **FP-06** | `FP_06_LiquidNetworkEnergyBenchmark.py` | `protocol_fp_06_neural_network_energy_benchmark.json` | Liquid Network Energy Benchmark | No APGI-P equivalent |
| **FP-07** | `FP_07_MathematicalConsistency.py` | `protocol_fp_07_mathematical_consistency_of_equations.json` | Mathematical Consistency | No APGI-P equivalent |
| **FP-08** | `FP_08_ParameterSensitivityIdentifiability.py` | `protocol_fp_08_parameter_sensitivity_identifiability.json` | Parameter Sensitivity | No APGI-P equivalent |
| **FP-10** | `FP_10_BayesianEstimationMCMC.py` | `protocol_fp_10_bayesian_estimation_with_mcmc.json` | Bayesian Estimation MCMC | No APGI-P equivalent |
| **FP-15** | `FP_15_AllenVisualCoding_Fatigue.py` | `protocol_fp_15_allen_visual_coding_fatigue.json` | Allen Visual Coding Fatigue | No APGI-P equivalent |

---

## Master Aggregators & Runners

| File | Purpose | Type |
|------|---------|------|
| `Validation/Master_Validation.py` | Compatibility wrapper for APGIMasterFalsifier | Aggregator |
| `Validation/VP_ALL_Aggregator.py` | Aggregates 23 VP results, tracks named predictions | Aggregator |
| `Falsification/Master_Falsification.py` | Orchestrates 15 FP protocols, central registry | Orchestrator |
| `Falsification/FP_ALL_Aggregator.py` | Framework-level falsification aggregation | Aggregator |
| `Validation_GUI.py` | GUI for validation protocol execution | Entry point |
| `Falsification_GUI.py` | GUI for falsification protocol execution | Entry point |
| `Protocols_GUI.py` | Protocol management interface | Entry point |
| `Theory_GUI.py` | Theory module runner (21 modules) | Entry point |
| `Tests_GUI.py` | Test execution interface | Entry point |
| `Utils_GUI.py` | Utility functions GUI | Entry point |
| `OSF_GUI.py` | Open Science Framework integration | Entry point |
| `gui/script_runner_gui.py` | Dynamic protocol discovery (AST-based) | Discovery |
| `gui/headless_runner.py` | CI/headless execution without tkinter | Runner |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `/config/protocol_manifest.json` | SHA256 integrity hashes (21 Theory + 23 VP + 17 FP scripts) |
| `/config/protocol_config.yaml` | Runtime parameters for all protocols |
| `/config/default_apgi_config.yaml` | Core framework configuration |
| `/config/default.yaml` | Default configuration template |
| `/config/gui_config.yaml` | GUI theme and layout settings |
| `/config/gui_alert_config.yaml` | Alert thresholds for protocol execution |
| `/config/profiles/*.yaml` | Clinical condition profiles (GAD, MDD, psychosis, etc.) |

---

## Protocol Tier Classification

### Primary (Core APGI Predictions)
- **APGI-P00**: HEP Proxy Validation (foundational)
- **APGI-P01**: Cardiac EEG (interoceptive precision)
- **APGI-P02**: Somatic Agent Simulation (active inference)

### Secondary (Extended Validation)
- **APGI-P03**: fMRI Anticipation
- **APGI-P04**: Metabolic Crossover
- **APGI-P05**: Causal TMS

### Tertiary (Specialized Tests)
- **APGI-P06**: Ignition Dynamics (iEEG)
- **APGI-P07**: DOC Biomarker

---

## Protocol Load Methods

### Method 1: Using New APGI-P## Format (Recommended)
```python
from utils.protocol_loader import load_protocol
spec = load_protocol("APGI-P01")  # Returns ProtocolSpec
```

### Method 2: Load All Protocols
```python
from utils.protocol_loader import load_all_protocols
all_specs = load_all_protocols()  # Dict[protocol_id, ProtocolSpec]
```

### Method 3: Get APGI Parameters
```python
from utils.protocol_loader import get_apgi_parameters
params = get_apgi_parameters("APGI-P01")  # Returns Dict
```

### Method 4: Load Template
```python
from utils.protocol_loader import get_template_protocol
template = get_template_protocol("P01")  # From template file
```

---

## Named Predictions (14 Core)

| Prediction | Description | Protocol | Threshold |
|------------|-------------|----------|-----------|
| **V1.1** | ML Classification accuracy | APGI-P01, VP-01 | > 85% |
| **V1.2** | Hierarchical levels correct | APGI-P02, VP-03 | All 3 levels |
| **V1.3** | PAC signatures present | APGI-P01, VP-09 | p < 0.05 |
| **V2.1** | Somatic marker advantage | APGI-P02, VP-03 | d > 0.5 |
| **V2.2** | Interoceptive precision effect | APGI-P01, VP-09 | r > 0.35 |
| **V3.1** | Task performance > baseline | APGI-P02, VP-10 | p < 0.05 |
| **V3.2** | APGI-specific networks | Multiple | p < 0.05 |
| **V3.3** | Threshold necessity | APGI-P01, VP-01 | Ablation study |
| **V3.4** | Efficiency gain | Multiple | Learning curve |
| **V3.5** | vmPFC anticipatory | APGI-P03, VP-15 | r > 0.3 |
| **V3.6** | Adaptation effects | Multiple | Time series |
| **P3.conv** | Framework convergence | Multiple | > 2 modalities |
| **P3.bic** | Model comparison | Multiple | ΔBIC > 10 |
| **P4/P5** | Neural signatures | APGI-P01, FP-09 | Multiple markers |

---

## File Locations

### Python Scripts
- **Validation**: `/Validation/VP_*.py` (23 files)
- **Falsification**: `/Falsification/FP_*.py` (15 files + Master)
- **Theory**: `/Theory/APGI_*.py` (21 files)
- **GUIs**: Root level `*_GUI.py` (7 files)
- **Runners**: `/gui/*.py` (2 files)

### Protocol Definition Files
- **New Canonical**: `/protocols/protocol_[0-7]_*.json` (8 files)
- **Legacy VP**: `/protocols/protocol_vp_*.json` (23 files)
- **Legacy FP**: `/protocols/protocol_fp_*.json` (15 files)
- **Schema**: `/protocols/schemas/protocol.schema.json` (1 file)
- **Template**: `/protocols/apgi_protocol_template.json` (1 file)

### Configuration
- **Config**: `/config/*.yaml`, `/config/*.json` (7 files)
- **Profiles**: `/config/profiles/*.yaml` (multiple clinical profiles)
- **Documentation**: `/docs/*.md` (12+ files)

---

## Protocol Execution Checklist

### Before Running Protocol
- [ ] Verify APGI-P00 (HEP Proxy) passed first
- [ ] Check protocol dependencies are available
- [ ] Confirm config file exists at `/config/protocol_config.yaml`
- [ ] Ensure data repository available at `/data_repository/`

### During Execution
- [ ] Monitor progress via GUI or log files
- [ ] Check for missing data or configuration errors
- [ ] Verify result format (ProtocolResult or dict)
- [ ] Collect named predictions for aggregation

### After Execution
- [ ] Validate results against falsification thresholds
- [ ] Compare to baseline/reference values
- [ ] Update VP_ALL_Aggregator with VP results
- [ ] Update FP_ALL_Aggregator with FP results
- [ ] Generate report with Master_Falsification

---

## Quick Command Reference

```bash
# Run all validation protocols
python -m Validation.Master_Validation

# Run all falsification protocols
python -m Falsification.Master_Falsification

# Run specific protocol
python Validation/VP_01_SyntheticEEGMLClassification.py

# Run GUI
python Validation_GUI.py        # Validation protocols
python Falsification_GUI.py     # Falsification protocols
python Protocols_GUI.py         # Protocol management
python Theory_GUI.py            # Theory modules

# Test protocol loader
python -c "from utils.protocol_loader import load_protocol; print(load_protocol('APGI-P01'))"

# Verify manifest
python -c "from utils.protocol_manifest import verify_protocol_file; \
  verify_protocol_file(Path('protocols/protocol_1_cardiac_eeg.json'), 'Validation')"

# List all protocols
python -c "from utils.protocol_loader import load_all_protocols; \
  [print(f'{id}: {spec.title}') for id, spec in load_all_protocols().items()]"
```

---

## Status Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **New canonical files** | ✅ Complete | 8 APGI-P## files created |
| **Legacy files** | ✅ Coexist | No breaking changes |
| **Protocol loader** | ✅ Both schemes | Auto-detection by number |
| **Python scripts** | ✅ Unchanged | Independent of JSON naming |
| **Aggregators** | ✅ Working | Use protocol IDs, not filenames |
| **Migration path** | ✅ Ready | Can proceed gradually |
| **Breaking changes** | ❌ None | Safe for immediate use |

---

## See Also

- **Full Mapping**: `/docs/PROTOCOL-LEGACY-MAPPING.md`
- **Migration Guide**: `/PROTOCOL-MIGRATION-GUIDE.md`
- **Discovery Results**: `/PROTOCOL-DISCOVERY-RESULTS.md`
- **Protocol Loader**: `/utils/protocol_loader.py`
- **Master Falsification**: `/Falsification/Master_Falsification.py`
- **VP All Aggregator**: `/Validation/VP_ALL_Aggregator.py`
