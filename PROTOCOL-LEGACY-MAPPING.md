# Protocol Legacy-to-New Naming Mapping

**Document Status**: Discovery Complete | Migration Ready  
**Last Updated**: 2026-06-24  
**Scope**: Maps all legacy protocol naming (`protocol_fp_XX`, `protocol_vp_ZZ`) to new canonical naming (`protocol_X`)

---

## Executive Summary

The APGI validation framework contains **47 protocol-related scripts** spread across three categories:
- **22 Validation Protocols (VP-00 to VP-22)** in `/Validation/`
- **15 Falsification Protocols (FP-01 to FP-15)** in `/Falsification/`
- **4 Master aggregators & GUI runners**

### Current State
- ✅ New canonical protocol files exist: `protocol_0_hep_proxy_validation.json` through `protocol_7_doc_biomarker.json`
- ✅ Legacy protocol files exist: `protocol_fp_01.json` through `protocol_fp_15.json` and `protocol_vp_00.json` through `protocol_vp_22.json`
- ⚠️ Python scripts still reference legacy filenames in some contexts
- ⚠️ Both naming conventions coexist in codebase

---

## Protocol Reference Mapping

### Canonical Protocol File Mapping

| New File | Protocol ID | Title | Linked VP | Linked FP | Status |
|----------|------------|-------|-----------|-----------|--------|
| `protocol_0_hep_proxy_validation.json` | APGI-P00 | HEP Proxy Validation (Empirical Prerequisite) | VP-00 | — | ✅ New |
| `protocol_1_cardiac_eeg.json` | APGI-P01 | EEG Interoceptive Gating (Cardiac EEG) | VP-01 | FP-09 | ✅ New |
| `protocol_2_somatic_agent_sim.json` | APGI-P02 | Somatic Agent Simulation (Active Inference) | VP-03 | FP-01 | ✅ New |
| `protocol_3_anticipation_fmri.json` | APGI-P03 | fMRI Anticipation (vmPFC) | VP-14, VP-15 | FP-14 | ✅ New |
| `protocol_4_metabolic_crossover.json` | APGI-P04 | Metabolic Crossover | VP-16 | FP-12 | ✅ New |
| `protocol_5_causal_tms.json` | APGI-P05 | Causal TMS & Neuromodulation | VP-07 | FP-03 | ✅ New |
| `protocol_6_ignition_ieeg.json` | APGI-P06 | Ignition Dynamics (iEEG) | VP-20 | FP-04, FP-11 | ✅ New |
| `protocol_7_doc_biomarker.json` | APGI-P07 | DOC Biomarker | VP-13 | FP-13 | ✅ New |

### Legacy Protocol Files (To Be Deprecated)

#### Validation Protocols (protocol_vp_XX.json)
| File | Protocol ID | Maps To | Python Script |
|------|------------|---------|----------------|
| `protocol_vp_00_hep_proxy_validation.json` | VP-00 | APGI-P00 | VP_00_HEPProxyValidation.py |
| `protocol_vp_01_synthetic_eeg_ml_classification.json` | VP-01 | APGI-P01 | VP_01_SyntheticEEGMLClassification.py |
| `protocol_vp_02_behavioral_bayesian_model_comparison.json` | VP-02 | (no direct map) | VP_02_BehavioralBayesianComparison.py |
| `protocol_vp_03_active_inference_agent_simulations_apgi_p03_linked.json` | VP-03 | APGI-P02 | VP_03_ActiveInferenceAgentSimulations.py |
| `protocol_vp_04_phase_transition_epistemic_level_2.json` | VP-04 | (no direct map) | VP_04_PhaseTransitionEpistemicLevel2.py |
| `protocol_vp_05_evolutionary_emergence.json` | VP-05 | (no direct map) | VP_05_EvolutionaryEmergence.py |
| `protocol_vp_06_liquid_network_inductive_bias.json` | VP-06 | (no direct map) | VP_06_LiquidNetworkInductiveBias.py |
| `protocol_vp_07_tms_causal_interventions_apgi_p02_linked.json` | VP-07 | APGI-P05 | VP_07_TMSCausalInterventions.py |
| `protocol_vp_07a_mathematical_consistency.json` | VP-07a | (variant) | VP_07a_MathematicalConsistency.py |
| `protocol_vp_08_psychophysical_threshold_estimation.json` | VP-08 | (no direct map) | VP_08_PsychophysicalThresholdEstimation.py |
| `protocol_vp_09_convergent_neural_signatures_priority_1_apgi_p01_linked.json` | VP-09 | APGI-P01 | VP_09_NeuralSignaturesEmpiricalPriority1.py |
| `protocol_vp_10_causal_manipulations_priority_2_apgi_p02_linked.json` | VP-10 | APGI-P02 | VP_10_CausalManipulationsPriority2.py |
| `protocol_vp_11_mcmc_cultural_neuroscience_priority_3.json` | VP-11 | (no direct map) | VP_11_MCMCCulturalNeurosciencePriority3.py |
| `protocol_vp_12_clinical_cross_species_convergence_apgi_p04_linked.json` | VP-12 | APGI-P04 | VP_12_ClinicalCrossSpeciesConvergence.py |
| `protocol_vp_13_epistemic_architecture.json` | VP-13 | APGI-P07 | VP_13_EpistemicArchitecture.py |
| `protocol_vp_14_fmri_anticipation_vs_experience_apgi_p05_linked.json` | VP-14 | APGI-P03 | VP_14_FMRIAnticipationExperience.py |
| `protocol_vp_15_fmri_anticipation_vmpfc.json` | VP-15 | APGI-P03 | VP_15_FMRIAnticipationVmPFC.py |
| `protocol_vp_16_metabolic_atp_ground_truth.json` | VP-16 | APGI-P04 | VP_16_MetabolicATPGroundTruth.py |
| `protocol_vp_17_allen_visual_coding_fatigue.json` | VP-17 | (no direct map) | VP_17_AllenVisualCodingFatigue.py |
| `protocol_vp_18_eeg_microstate_gfp_p3b.json` | VP-18 | (no direct map) | VP_18_EEGMicrostateGFPP3b.py |
| `protocol_vp_19_information_erasure_mvpa.json` | VP-19 | (no direct map) | VP_19_InformationErasureMVPA.py |
| `protocol_vp_20_empirical_intracranial_eeg_apgi_p06_linked.json` | VP-20 | APGI-P06 | VP_20_EmpiricalIEEG.py |
| `protocol_vp_21_free_energy_prediction_error.json` | VP-21 | (no direct map) | VP_21_FreeEnergyPredictionError.py |
| `protocol_vp_22_enhanced_fmri_anticipation_vs_experience_apgi_p05_linked.json` | VP-22 | APGI-P05 | VP_22_FMRIAnticipationExperience.py |

#### Falsification Protocols (protocol_fp_XX.json)
| File | Protocol ID | Maps To | Python Script |
|------|------------|---------|----------------|
| `protocol_fp_01_active_inference_agents_f1_f2.json` | FP-01 | APGI-P02 | FP_01_ActiveInference.py |
| `protocol_fp_02_agent_comparison_convergence_benchmark.json` | FP-02 | (no direct map) | FP_02_AgentComparisonConvergenceBenchmark.py |
| `protocol_fp_03_framework_level_multi_protocol.json` | FP-03 | (no direct map) | FP_03_FrameworkLevelMultiProtocol.py |
| `protocol_fp_04_phase_transition_bistability_apgi_p06_linked.json` | FP-04 | APGI-P06 | FP_04_PhaseTransitionEpistemicArchitecture.py |
| `protocol_fp_05_evolutionary_plausibility.json` | FP-05 | (no direct map) | FP_05_EvolutionaryPlausibility.py |
| `protocol_fp_06_neural_network_energy_benchmark.json` | FP-06 | (no direct map) | FP_06_LiquidNetworkEnergyBenchmark.py |
| `protocol_fp_07_mathematical_consistency_of_equations.json` | FP-07 | (no direct map) | FP_07_MathematicalConsistency.py |
| `protocol_fp_08_parameter_sensitivity_identifiability.json` | FP-08 | (no direct map) | FP_08_ParameterSensitivityIdentifiability.py |
| `protocol_fp_09_neural_signatures_eeg_p3b_hep_apgi_p01_linked.json` | FP-09 | APGI-P01 | FP_09_NeuralSignaturesP3bHEP.py |
| `protocol_fp_10_bayesian_estimation_with_mcmc.json` | FP-10 | (no direct map) | FP_10_BayesianEstimationMCMC.py |
| `protocol_fp_11_liquid_network_dynamics_echo_state.json` | FP-11 | APGI-P06 | FP_11_LiquidNetworkDynamicsEchoState.py |
| `protocol_fp_12_cross_species_scaling.json` | FP-12 | APGI-P04 | FP_12_CrossSpeciesScaling.py |
| `protocol_fp_13_clinical_cross_species_convergence.json` | FP-13 | APGI-P07 | FP_13_Clinical_CrossSpecies_Convergence.py |
| `protocol_fp_14_fmri_anticipation_vmpfc.json` | FP-14 | APGI-P03 | FP_14_fMRI_Anticipation_vmPFC.py |
| `protocol_fp_15_allen_visual_coding_fatigue.json` | FP-15 | (no direct map) | FP_15_AllenVisualCoding_Fatigue.py |

---

## Scripts Using Protocols

### Master Aggregators & Orchestrators
- **Validation/Master_Validation.py** - Wraps APGIMasterFalsifier for validation interface
- **Validation/VP_ALL_Aggregator.py** - Aggregates VP-00 through VP-22 results
- **Falsification/Master_Falsification.py** - Orchestrates FP-01 through FP-15, tracks named predictions
- **Falsification/FP_ALL_Aggregator.py** - Framework-level falsification aggregation

### GUI Entry Points
- **Validation_GUI.py** - Loads protocols via `safe_import_module()`, protocol list hardcoded
- **Falsification_GUI.py** - Dynamic protocol discovery via Master_Falsification
- **Protocols_GUI.py** - Protocol management interface
- **Theory_GUI.py** - Theory module runner
- **Tests_GUI.py** - Test execution

### Headless/CI Runners
- **gui/script_runner_gui.py** - ScriptRunnerGUI with `_discover_protocols()` method
- **gui/headless_runner.py** - HeadlessRunner for CI/headless execution

---

## Protocol Loading Patterns

### Pattern 1: Direct File Reference (LEGACY - NEEDS UPDATE)
**Location**: `Validation_GUI.py` line 119+

```python
protocol_files = [
    ("APGI_Protocol_0_HEPProxy", "VP_00_HEPProxyValidation.py"),
    ("APGI_Protocol_1", "VP_01_SyntheticEEGMLClassification.py"),
    # ... hardcoded list of 22+ protocols
]
```

**Impact**: GUI labels still reference old naming; no direct JSON file refs here.

### Pattern 2: Dynamic Module Loading via importlib
**Location**: `Master_Falsification.py`, GUI runners

```python
spec = importlib.util.spec_from_file_location(protocol_id, file_path)
module = importlib.util.module_from_spec(spec)
sys.modules[protocol_id] = module
spec.loader.exec_module(module)
```

**Impact**: Python scripts are loaded dynamically; JSON protocol files are independent.

### Pattern 3: Protocol Registry Lookup
**Location**: `utils/protocol_loader.py`

```python
def _find_protocol_file(protocol_id: str) -> Optional[Path]:
    # Fast path for APGI-P## convention
    if protocol_id.startswith("APGI-P"):
        num = int(protocol_id.split("P")[-1])
        for path in _iter_protocol_files():
            if int(path.stem.split("_")[1]) == num:
                return path
    # Generic path: match embedded protocol_id
    for path in _iter_protocol_files():
        data = json.load(path)
        if data.get("protocol_id") == normalized_id:
            return path
```

**Impact**: Already supports new APGI-P## naming; looks for numeric match in filename.

---

## Migration Status

### ✅ Complete (No Action Needed)
1. **New canonical JSON files created** (protocol_0 through protocol_7)
2. **Protocol loader supports both naming schemes**
3. **Python VP/FP scripts are independent of JSON naming**

### ⚠️ Partial (Attention Needed)
1. **Legacy JSON files still present** (protocol_vp_00 through protocol_vp_22, protocol_fp_01 through protocol_fp_15)
2. **Config files reference old structure** (if any)
3. **Tests may expect legacy filenames**

### ❌ TODO (After Deprecation Period)
1. **Retire legacy JSON files** (after 2-3 release cycles)
2. **Update any hardcoded protocol file path references**
3. **Update test fixtures** (if they reference old filenames)
4. **Update documentation** (PROTOCOL references)

---

## Detailed Protocol Linking

### APGI-P00: HEP Proxy Validation (Empirical Prerequisite)
- **New File**: `protocol_0_hep_proxy_validation.json`
- **Legacy File**: `protocol_vp_00_hep_proxy_validation.json`
- **Python Scripts**: `VP_00_HEPProxyValidation.py` (validation module)
- **Dependencies**: None (foundational protocol)
- **Predictions**: Pred 0.A, 0.B, 0.C (HEP-interoceptive correlation, physostigmine effect, aINS tracking)
- **Status**: ✅ Canonicalized

### APGI-P01: Cardiac EEG (EEG Interoceptive Gating)
- **New File**: `protocol_1_cardiac_eeg.json`
- **Legacy Files**: 
  - `protocol_vp_01_synthetic_eeg_ml_classification.json`
  - `protocol_fp_09_neural_signatures_eeg_p3b_hep_apgi_p01_linked.json`
- **Python Scripts**: 
  - `VP_01_SyntheticEEGMLClassification.py` (validation)
  - `FP_09_NeuralSignaturesP3bHEP.py` (falsification)
  - `VP_09_NeuralSignaturesEmpiricalPriority1.py` (validation priority 1)
- **Dependencies**: APGI-P00 (must pass HEP proxy validation first)
- **Tier**: Primary validation
- **Status**: ✅ Canonicalized

### APGI-P02: Somatic Agent Simulation (Active Inference)
- **New File**: `protocol_2_somatic_agent_sim.json`
- **Legacy Files**:
  - `protocol_vp_03_active_inference_agent_simulations_apgi_p03_linked.json` (wrong numbering)
  - `protocol_fp_01_active_inference_agents_f1_f2.json`
- **Python Scripts**:
  - `VP_03_ActiveInferenceAgentSimulations.py` (validation)
  - `FP_01_ActiveInference.py` (falsification)
  - `VP_10_CausalManipulationsPriority2.py` (Priority 2 validation)
- **Dependencies**: APGI-P01 (cardiac EEG measurements)
- **Tier**: Primary validation
- **Falsification Criteria**: F1.x, F2.x (active inference agent predictions)
- **Status**: ✅ Canonicalized

### APGI-P03: fMRI Anticipation (vmPFC)
- **New File**: `protocol_3_anticipation_fmri.json`
- **Legacy Files**:
  - `protocol_vp_14_fmri_anticipation_vs_experience_apgi_p05_linked.json` (wrong numbering)
  - `protocol_vp_15_fmri_anticipation_vmpfc.json` (wrong numbering)
  - `protocol_fp_14_fmri_anticipation_vmpfc.json`
- **Python Scripts**:
  - `VP_14_FMRIAnticipationExperience.py`
  - `VP_15_FMRIAnticipationVmPFC.py`
  - `FP_14_fMRI_Anticipation_vmPFC.py`
- **Dependencies**: APGI-P01 (interoceptive precision measures)
- **Tier**: Extended validation
- **Status**: ✅ Canonicalized

### APGI-P04: Metabolic Crossover
- **New File**: `protocol_4_metabolic_crossover.json`
- **Legacy Files**:
  - `protocol_vp_12_clinical_cross_species_convergence_apgi_p04_linked.json`
  - `protocol_vp_16_metabolic_atp_ground_truth.json`
  - `protocol_fp_12_cross_species_scaling.json`
- **Python Scripts**:
  - `VP_12_ClinicalCrossSpeciesConvergence.py`
  - `VP_16_MetabolicATPGroundTruth.py`
  - `FP_12_CrossSpeciesScaling.py`
- **Dependencies**: APGI-P01 (cardinal interoceptive precision)
- **Tier**: Secondary validation
- **Status**: ✅ Canonicalized

### APGI-P05: Causal TMS & Neuromodulation
- **New File**: `protocol_5_causal_tms.json`
- **Legacy Files**:
  - `protocol_vp_07_tms_causal_interventions_apgi_p02_linked.json` (wrong numbering)
  - `protocol_vp_22_enhanced_fmri_anticipation_vs_experience_apgi_p05_linked.json` (wrong numbering)
  - `protocol_fp_03_framework_level_multi_protocol.json` (wrong linking)
- **Python Scripts**:
  - `VP_07_TMSCausalInterventions.py`
  - `VP_22_FMRIAnticipationExperience.py`
  - `FP_03_FrameworkLevelMultiProtocol.py`
- **Dependencies**: APGI-P01, APGI-P03 (fMRI baseline)
- **Tier**: Secondary validation
- **Status**: ✅ Canonicalized

### APGI-P06: Ignition Dynamics (iEEG)
- **New File**: `protocol_6_ignition_ieeg.json`
- **Legacy Files**:
  - `protocol_vp_20_empirical_intracranial_eeg_apgi_p06_linked.json`
  - `protocol_fp_04_phase_transition_bistability_apgi_p06_linked.json`
  - `protocol_fp_11_liquid_network_dynamics_echo_state.json`
- **Python Scripts**:
  - `VP_20_EmpiricalIEEG.py`
  - `FP_04_PhaseTransitionEpistemicArchitecture.py`
  - `FP_11_LiquidNetworkDynamicsEchoState.py`
- **Dependencies**: APGI-P01 (HEP/P3b signatures), APGI-P03 (anticipatory dynamics)
- **Tier**: Tertiary/specialized
- **Status**: ✅ Canonicalized

### APGI-P07: DOC Biomarker (Disorders of Consciousness)
- **New File**: `protocol_7_doc_biomarker.json`
- **Legacy Files**:
  - `protocol_vp_13_epistemic_architecture.json`
  - `protocol_fp_13_clinical_cross_species_convergence.json`
- **Python Scripts**:
  - `VP_13_EpistemicArchitecture.py`
  - `FP_13_Clinical_CrossSpecies_Convergence.py`
- **Dependencies**: APGI-P01 (HEP as consciousness biomarker)
- **Tier**: Tertiary/specialized
- **Status**: ✅ Canonicalized

---

## Scripts with Legacy References (Audit)

### Direct File Path References
- ✅ **utils/protocol_loader.py**: Already supports both naming schemes via numeric matching
- ✅ **config/protocol_manifest.json**: Contains both old and new files in SHA256 registry
- ⚠️ **Validation_GUI.py** line 119+: Hardcoded protocol display names (not file refs, OK to keep)
- ⚠️ **Falsification_GUI.py**: Dynamic discovery via Master_Falsification (OK, independent)

### Protocol ID String References (Audit)
Most Python scripts use protocol_id strings like:
- `protocol_id = "VP_01_SyntheticEEGMLClassification"` (long form, internal use)
- `protocol_id = "VP-01"` (short form in results)
- `protocol_id = "APGI-P01"` (canonical form in new files)

**Status**: No required changes; strings don't depend on JSON filename.

---

## Recommended Deprecation Timeline

### Phase 1: Current (✅ Complete)
- New canonical files created (protocol_0 through protocol_7)
- Protocol loader supports both schemes
- Documentation updated

### Phase 2: Next Release (1-2 months)
- Add deprecation warnings if legacy protocol files are accessed directly
- Update any remaining hardcoded paths in tests/CI
- Publish migration guide to users

### Phase 3: Following Release (3-4 months)
- Archive legacy JSON files (keep in git history)
- Remove from active protocols/ directory
- Update test fixtures

### Phase 4: Final Release (6+ months)
- Full removal of legacy files
- Update documentation to reference new files only

---

## Key Files for Review

### Protocol Definition Files
- Validation: `/Validation/VP_*.py` (23 files)
- Falsification: `/Falsification/FP_*.py` (15 files)
- Theory: `/Theory/APGI_*.py` (21 files)

### Aggregation & Orchestration
- `/Validation/Master_Validation.py`
- `/Validation/VP_ALL_Aggregator.py`
- `/Falsification/Master_Falsification.py`
- `/Falsification/FP_ALL_Aggregator.py`

### Configuration & Loading
- `/config/protocol_manifest.json`
- `/config/protocol_config.yaml`
- `/utils/protocol_loader.py`
- `/utils/protocol_registry.py`

### GUI & Entry Points
- `/Validation_GUI.py`
- `/Falsification_GUI.py`
- `/gui/script_runner_gui.py`
- `/gui/headless_runner.py`

---

## Migration Checklist

### Before Deprecation
- [ ] Verify all new protocol_N.json files have correct protocol_id fields
- [ ] Test protocol_loader.py with both old and new filenames
- [ ] Update unit tests to use new APGI-P## naming
- [ ] Run full test suite with legacy files present

### Deprecation Notice (In Code)
- [ ] Add comments in protocol_loader.py about deprecation timeline
- [ ] Update config/protocol_config.yaml with deprecation note
- [ ] Add deprecation warning when old filenames are accessed

### Archive Phase
- [ ] Create legacy-protocols/ archive folder
- [ ] Move old protocol_vp_* and protocol_fp_* files there
- [ ] Update .gitignore if needed
- [ ] Document in CHANGELOG.md

### Final Cleanup
- [ ] Update all documentation to reference APGI-P## only
- [ ] Remove any remaining hardcoded path references
- [ ] Clean up protocol_manifest.json

---

## Notes

1. **Python script naming** (VP_NN, FP_NN) is independent of JSON protocol files and requires no change
2. **Protocol IDs** in Python code strings (VP-01, FP-01, APGI-P01) are separate from filenames
3. **New canonical files** use APGI-P## convention for unambiguous linking
4. **Legacy files should be archived gradually**, not deleted abruptly
5. **Protocol loader already handles both conventions** intelligently
