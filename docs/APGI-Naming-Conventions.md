# APGI Naming Conventions

**Scope:** All 35 protocol scripts across `Falsification/` and `Validation/` folders, plus orchestrators  
**Source of truth for file names:** `ls Falsification/*.py` and `ls Validation/*.py` 

---

## UPID Design Reference

Every protocol entry carries a **Unique Protocol ID (UPID)** with the format:

```text
APGI-[TIER][SEQ]-[DOM3]
```

| Segment | Values | Meaning |
| --------- | -------- | --------- |
| `TIER` | `E` | **Empirical Primary** — paper-specified experimental protocol |
| | `C` | **Computational** — cross-cutting theoretical / simulation protocol |
| | `X` | **Extended** — auxiliary protocols beyond the original six-paper matrix |
| | `G` | **Global Aggregator** — framework-level orchestrators |
| `SEQ` | `01` … `99` | Zero-padded sequence number within tier |
| `DOM3` | 3-char code | Domain shorthand |

### Domain Codes

| Code | Domain |
| ------ | -------- |
| `EEG` | EEG / Electrophysiology / Interoceptive |
| `TMS` | Transcranial Magnetic Stimulation / Causal Neuromodulation |
| `AIN` | Active Inference / Agent Simulation |
| `DOC` | Disorders of Consciousness / Clinical Biomarker |
| `MRI` | fMRI / BOLD / vmPFC Imaging |
| `IEG` | Intracranial EEG / All-or-None Ignition |
| `PHT` | Phase Transition (Information-Theoretic) |
| `EVL` | Evolutionary Plausibility / Genetic Gating |
| `ENE` | Energy Efficiency / Thermodynamic Cost |
| `MTH` | Mathematical Consistency / Equation Verification |
| `PRM` | Parameter Identifiability / Sensitivity |
| `BAY` | Bayesian Estimation / MCMC |
| `BNC` | Benchmark / Agent Convergence Comparison |
| `FWK` | Framework-Level Multi-Protocol |
| `LNN` | Liquid Neural Network / Echo-State Dynamics |
| `XSP` | Cross-Species Allometric Scaling |
| `EPS` | Epistemic Architecture |
| `MET` | Metabolic / ATP Ground Truth |
| `VIS` | Visual Coding / Allen Brain Observatory |
| `GFP` | Global Field Power / EEG Microstate |
| `MVA` | Multi-Voxel Pattern Analysis / MVPA |
| `FEN` | Free Energy / Prediction Error |

---

## Section 1 — Primary Empirical Protocols (Tier E)

*Directly mapped to numbered paper protocols.*

| UPID | Legacy ID | Experimental Concept | Paper Protocol | Roadmap Priority | FP IDs | VP IDs | Falsification File(s) | Validation File(s) | Notes |
| ------ | ----------- | ---------------------- | --------------- | ----------------- | -------- | -------- | ---------------------- | -------------------- | ------- |
| **APGI-E01-EEG** | APGI-P1 | EEG-Heart-Evoked Potential (Interoceptive Gating) | Protocol 1 | Priority 1 | FP-9 | VP-9 | `Falsification/FP_09_NeuralSignaturesP3bHEP.py` | `Validation/VP_09_NeuralSignaturesEmpiricalPriority1.py` | ⚠ Old matrix named FP file as `FP_09_TemporalDynamics_SequenceSensitivity.py` — **stale, file does not exist** |
| **APGI-E02-TMS** | APGI-P2 | Causal TMS Study (Insula / Thalamic Gating) | Protocol 2 | Priority 2 | FP-9 shared | VP-7, VP-10 | `Falsification/FP_09_NeuralSignaturesP3bHEP.py` | `Validation/VP_07_TMSCausalInterventions.py` · `Validation/VP_10_CausalManipulationsPriority2.py` | ⚠ Old matrix listed `CausalManipulations_TMS_Pharmacological_...py` — **stale partial name; two VP files serve this protocol** |
| **APGI-E03-AIN** | APGI-P3 | Active Inference Simulations (Agent Performance) | Protocol 3 | Comp. Protocol 3 | FP-1 | VP-3 | `Falsification/FP_01_ActiveInference.py` | `Validation/VP_03_ActiveInferenceAgentSimulations.py` | ✅ File names correct in original matrix |
| **APGI-E04-DOC** | APGI-P4 | Disorders of Consciousness (Clinical Biomarkers) | Protocol 4 | Priority 4 | — | VP-12 | *(no dedicated FP; shares FP-3 framework)* | `Validation/VP_12_ClinicalCrossSpeciesConvergence.py` | ⚠ Old matrix named VP file as `Clinical_CrossSpecies_Convergence_Protocol4.py` — **stale; missing `VP_12_` prefix** |
| **APGI-E05-MRI** | APGI-P5 | fMRI Anticipation vs. Experience (vmPFC) | Protocol 5 | N/A | — | VP-14, VP-15 | *(no dedicated FP)* | `Validation/VP_14_FMRIAnticipationExperience.py` · `Validation/VP_15_FMRIAnticipationVmPFC.py` | ⚠ Old matrix marked VP as **MISSING** and file as "To be developed" — **both VP-14 and VP-15 exist on disk** |
| **APGI-E06-IEG** | APGI-P6 | Intracranial EEG (All-or-None Ignition) | Protocol 6 | N/A | — | VP-1, VP-2 | *(no dedicated FP; shares FP-ALL)* | `Validation/VP_01_SyntheticEEGMLClassification.py` · `Validation/VP_02_BehavioralBayesianComparison.py` | ⚠ Old matrix named VP file as `SyntheticEEG_MLClassification.py` — **missing `VP_01_` prefix** |

---

## Section 2 — Computational Cross-Cutting Protocols (Tier C)

*Not tied to a single paper protocol; validate theoretical / mathematical claims.*

| UPID | Legacy ID | Experimental Concept | Paper Protocol | Roadmap Priority | FP IDs | VP IDs | Falsification File(s) | Validation File(s) | Notes |
| ------ | ----------- | ---------------------- | --------------- | ----------------- | -------- | -------- | ---------------------- | -------------------- | ------- |
| **APGI-C01-PHT** | APGI-C1 | Phase Transition Analysis (Information-Theoretic) | N/A | N/A | FP-4 | VP-4 | `Falsification/FP_04_PhaseTransitionEpistemicArchitecture.py` | `Validation/VP_04_PhaseTransitionEpistemicLevel2.py` | ✅ File names correct in original matrix |
| **APGI-C02-EVL** | APGI-C2 | Evolutionary Emergence (Genetic Gating) | N/A | N/A | FP-5 | VP-5 | `Falsification/FP_05_EvolutionaryPlausibility.py` | `Validation/VP_05_EvolutionaryEmergence.py` | ⚠ Old matrix named FP file as `FP_05_SurvivalAnalysis_TimeToIgnition.py` — **stale; file does not exist** |
| **APGI-C03-ENE** | APGI-C3 | Energy Efficiency (Thermodynamic Cost) | Program 1–4 | N/A | FP-6 | VP-6 | `Falsification/FP_06_LiquidNetworkEnergyBenchmark.py` | `Validation/VP_06_LiquidNetworkInductiveBias.py` | ⚠ Old matrix named FP file as `Falsification-NeuralNetwork-EnergyBenchmark.py` — **legacy hyphen-prefixed name; does not exist** |
| **APGI-C04-MTH** | APGI-C4 | Mathematical Consistency (Equation Verification) | N/A | N/A | FP-7 | VP-7 | `Falsification/FP_07_MathematicalConsistency.py` | `Validation/VP_07_TMSCausalInterventions.py` | ⚠ Old matrix named FP file as `Falsification-MathematicalConsistency-Equations.py` — **legacy hyphen-prefixed name; does not exist**; VP-7 is TMS, not math — consider adding VP-13 for epistemic |
| **APGI-C05-PRM** | APGI-C5 | Parameter Identifiability (Reduced Model) | N/A | N/A | FP-8 | VP-8 | `Falsification/FP_08_ParameterSensitivityIdentifiability.py` | `Validation/VP_08_PsychophysicalThresholdEstimation.py` | ⚠ Old matrix named FP file as `Falsification-ParameterSensitivity-Identifiability.py` — **legacy hyphen-prefixed name; does not exist** |
| **APGI-C06-BAY** | APGI-C6 | Bayesian Parameter Estimation (MCMC) | N/A | Priority 3 | FP-10 | VP-11 | `Falsification/FP_10_BayesianEstimationMCMC.py` | `Validation/VP_11_MCMCCulturalNeurosciencePriority3.py` | ⚠ Old matrix named FP file as `Falsification-BayesianEstimation-MCMC.py` — **legacy hyphen-prefixed name; does not exist** |

---

## Section 3 — Extended / Auxiliary Protocols (Tier X)

*Protocols implemented in the repository but absent from the original six-column matrix. Assigned new UPIDs.*

| UPID | FP ID | VP ID | Concept | Falsification File | Validation File | Notes |
| ------ | ----- | ----- | ------- | ------------------- | ---------------- | ------- |
| **APGI-X01-BNC** | FP-2 | VP-2 | Agent Comparison / Convergence Benchmark | `Falsification/FP_02_AgentComparisonConvergenceBenchmark.py` | `Validation/VP_02_BehavioralBayesianComparison.py` | Not in original matrix; FP-2 is the agent BIC/AIC benchmark |
| **APGI-X02-FWK** | FP-3 | — | Framework-Level Multi-Protocol Synthesis | `Falsification/FP_03_FrameworkLevelMultiProtocol.py` | *(covered by VP-ALL aggregator)* | Not in original matrix; cross-protocol synthesis |
| **APGI-X03-LNN** | FP-11 | VP-6 shared | Liquid Neural Network / Echo-State Dynamics | `Falsification/FP_11_LiquidNetworkDynamicsEchoState.py` | `Validation/VP_06_LiquidNetworkInductiveBias.py` | Not in original matrix; LTCN echo-state reservoir |
| **APGI-X04-XSP** | FP-12 | VP-12 shared | Cross-Species Allometric Scaling | `Falsification/FP_12_CrossSpeciesScaling.py` | `Validation/VP_12_ClinicalCrossSpeciesConvergence.py` | ⚠ Old matrix incorrectly assigned FP-12 label to the **aggregator** (`FP_ALL_Aggregator.py`); FP-12 is this cross-species scaling file |
| **APGI-X05-EPS** | — | VP-13 | Epistemic Architecture (Self-Audit) | *(no dedicated FP)* | `Validation/VP_13_EpistemicArchitecture.py` | Predictions P5–P12 from Paper 4 three-tier model |
| **APGI-X06-MRI** | — | VP-15 | fMRI vmPFC Anticipation (Detailed Focus) | *(no dedicated FP)* | `Validation/VP_15_FMRIAnticipationVmPFC.py` | Companion to APGI-E05-MRI / VP-14; vmPFC-specific |
| **APGI-X07-MET** | — | VP-16 | Metabolic ATP Ground Truth (³¹P-MRS) | *(no dedicated FP)* | `Validation/VP_16_MetabolicATPGroundTruth.py` | Epistemic Tier L1 metabolic calibration |
| **APGI-X08-VIS** | — | VP-17 | Allen Visual Coding / Neural Fatigue | *(no dedicated FP)* | `Validation/VP_17_AllenVisualCodingFatigue.py` | Prediction P11 — stimulus-repetition fatigue |
| **APGI-X09-GFP** | — | VP-18 | EEG Microstate / Global Field Power P3b | *(no dedicated FP)* | `Validation/VP_18_EEGMicrostateGFPP3b.py` | GFP-based microstate classification |
| **APGI-X10-MVA** | — | VP-19 | Information Erasure / MVPA | *(no dedicated FP)* | `Validation/VP_19_InformationErasureMVPA.py` | Multi-voxel decoding of ignition events |
| **APGI-X11-IEG** | — | VP-20 | Empirical iEEG Analysis | *(no dedicated FP)* | `Validation/VP_20_EmpiricalIEEG.py` | Real intracranial EEG; complements APGI-E06-IEG |
| **APGI-X12-FEN** | — | VP-21 | Free Energy / Prediction Error | *(no dedicated FP)* | `Validation/VP_21_FreeEnergyPredictionError.py` | Variational free-energy prediction-error analysis |

---

## Section 4 — Global Aggregators (Tier G)

| UPID | Legacy ID | Role | File | Notes |
| ------ | ----------- | ---- | ---- | ------- |
| **APGI-G01-FWK** | APGI-AGG (FP) | Terminal falsification aggregator — combines all FP results, evaluates Conditions A & B | `Falsification/FP_ALL_Aggregator.py` | ⚠ Old matrix labelled this `FP_12_Falsification_Aggregator.py` — **file does not exist**; `FP_12_CrossSpeciesScaling.py` is a *different* protocol (see APGI-X04-XSP) |
| **APGI-G02-FWK** | — (VP) | Framework-level validation aggregator — collects V1.1–V21.x predictions | `Validation/VP_ALL_Aggregator.py` | No corresponding entry in original matrix |
| **APGI-G03-FWK** | — | FP orchestrator — runs all 12 FP scripts sequentially | `Falsification/Master_Falsification.py` | CLI entry point for full falsification run |
| **APGI-G04-FWK** | — | VP orchestrator — runs all VP scripts | `Validation/Master_Validation.py` | CLI entry point for full validation run |

---

## Section 5 — Error Log: Original Matrix vs. Repository Audit

The following discrepancies were found between the user-supplied matrix and the actual file system state (May 2026):

| Row | Original File Name in Matrix | Actual File on Disk | Error Type |
| ---- | ------------------------------ | ------------------- | ---------- |
| APGI-P1 | `FP_09_TemporalDynamics_SequenceSensitivity.py` | `Falsification/FP_09_NeuralSignaturesP3bHEP.py` | Wrong name — file does not exist |
| APGI-P2 | `CausalManipulations_TMS_Pharmacological_...py` | `Validation/VP_10_CausalManipulationsPriority2.py` | Truncated legacy name — missing `VP_10_` prefix |
| APGI-P4 | `Clinical_CrossSpecies_Convergence_Protocol4.py` | `Validation/VP_12_ClinicalCrossSpeciesConvergence.py` | Missing `VP_12_` prefix |
| APGI-P5 | "To be developed" (VP listed as MISSING) | `Validation/VP_14_FMRIAnticipationExperience.py` + `VP_15_FMRIAnticipationVmPFC.py` | Both files exist — **false negative** |
| APGI-P6 | `SyntheticEEG_MLClassification.py` | `Validation/VP_01_SyntheticEEGMLClassification.py` | Missing `VP_01_` prefix |
| APGI-C2 | `FP_05_SurvivalAnalysis_TimeToIgnition.py` | `Falsification/FP_05_EvolutionaryPlausibility.py` | Wrong name — file does not exist |
| APGI-C3 | `Falsification-NeuralNetwork-EnergyBenchmark.py` | `Falsification/FP_06_LiquidNetworkEnergyBenchmark.py` | Legacy hyphen convention — file does not exist |
| APGI-C4 | `Falsification-MathematicalConsistency-Equations.py` | `Falsification/FP_07_MathematicalConsistency.py` | Legacy hyphen convention — file does not exist |
| APGI-C5 | `Falsification-ParameterSensitivity-Identifiability.py` | `Falsification/FP_08_ParameterSensitivityIdentifiability.py` | Legacy hyphen convention — file does not exist |
| APGI-C6 | `Falsification-BayesianEstimation-MCMC.py` | `Falsification/FP_10_BayesianEstimationMCMC.py` | Legacy hyphen convention — file does not exist |
| APGI-AGG | `FP_12_Falsification_Aggregator.py` | `Falsification/FP_ALL_Aggregator.py` | FP-12 is a different protocol (cross-species); aggregator uses `FP_ALL_` prefix |

**Summary:** 11 of 13 original rows contained incorrect file names (85% error rate). 10 VP protocols (VP-13 → VP-22) and 3 FP protocols (FP-2, FP-3, FP-11) were completely absent from the original matrix.

---

## Section 6 — Suggested Improvements

### 6.1 File Naming Convention

Adopt a single, enforced convention across the repository:

```text
[FP|VP]_[NN]_[CompoundCamelCase].py
```

The concept segment uses **pure CamelCase with no internal underscores** — every word is capitalised and concatenated directly.

```text
✅  FP_09_NeuralSignaturesP3bHEP.py
✅  VP_12_ClinicalCrossSpeciesConvergence.py
✗   FP_09_NeuralSignatures_P3b_HEP.py   ← underscores inside concept
```

All current Falsification and Validation scripts already follow this pattern. The old `Falsification-*.py` hyphen-prefixed names are entirely absent from disk and should be removed from all documentation.

### 6.2 Aggregator Naming Ambiguity

`FP_ALL_Aggregator.py` and `VP_ALL_Aggregator.py` do not follow the `NN` numbering convention. Consider renaming to:

```text
FP_00_ALL_Aggregator.py   # prefix 00 = meta/aggregator tier
VP_00_ALL_Aggregator.py
```

This keeps them sortable with ls and avoids confusion with `FP_12_*.py`.

### 6.3 Protocol Coverage Gaps

| Gap | Recommendation |
| ---- | --------------- |
| APGI-E04-DOC has no dedicated FP | Add `FP_13_ClinicalBiomarkers_DoC.py` to complete the falsification loop |
| APGI-E05-MRI has no FP | Add `FP_14_fMRI_Anticipation_vmPFC.py` for vmPFC-specific falsification criteria |
| APGI-E06-IEG has no FP | Add `FP_15_Ignition_AllOrNone_iEEG.py` to falsify all-or-none ignition threshold |
| VP-7 (TMS) and APGI-C04-MTH share a VP slot | VP-7 is TMS; mathematical consistency VP is missing — add `VP_22_MathematicalConsistency_Validation.py` |

### 6.4 UPID Adoption in Code

Add the UPID as a module-level constant in each protocol script:

```python
PROTOCOL_UPID = "APGI-E01-EEG"  # EEG-Heart-Evoked Potential — Interoceptive Gating
```

This allows the aggregators (`FP_ALL_Aggregator.py`, `VP_ALL_Aggregator.py`) to tag results by UPID rather than by positional FP/VP number, making cross-paper result tables stable.

### 6.5 Matrix Maintenance

Keep this matrix as the **single source of truth** for protocol-to-file mapping. The following files currently duplicate portions of it and should either reference this matrix or be deprecated:

- `docs/Files-Protocols.md` — partial FP/VP table
- `docs/PROTOCOLS.md` — completion percentages overlap
- `docs/Status-Protocols.md` — status overlap

---

## Section 7 — Quick-Reference UPID Lookup Table

| UPID | Concept (Short) | FP | VP | FP File | VP File |
| ------ | --------------- | -- | -- | ------- | ------- |
| APGI-E01-EEG | EEG / HEP Interoceptive | FP-9 | VP-9 | `FP_09_NeuralSignaturesP3bHEP.py` | `VP_09_NeuralSignaturesEmpiricalPriority1.py` |
| APGI-E02-TMS | Causal TMS / Insula | FP-9¹ | VP-7, VP-10 | `FP_09_NeuralSignaturesP3bHEP.py` | `VP_07_TMSCausalInterventions.py` · `VP_10_CausalManipulationsPriority2.py` |
| APGI-E03-AIN | Active Inference Agents | FP-1 | VP-3 | `FP_01_ActiveInference.py` | `VP_03_ActiveInferenceAgentSimulations.py` |
| APGI-E04-DOC | Disorders of Consciousness | — | VP-12 | — | `VP_12_ClinicalCrossSpeciesConvergence.py` |
| APGI-E05-MRI | fMRI vmPFC Anticipation | — | VP-14, VP-15 | — | `VP_14_FMRIAnticipationExperience.py` · `VP_15_FMRIAnticipationVmPFC.py` |
| APGI-E06-IEG | Intracranial EEG Ignition | — | VP-1, VP-2 | — | `VP_01_SyntheticEEGMLClassification.py` · `VP_02_BehavioralBayesianComparison.py` |
| APGI-C01-PHT | Phase Transition Info-Theoretic | FP-4 | VP-4 | `FP_04_PhaseTransitionEpistemicArchitecture.py` | `VP_04_PhaseTransitionEpistemicLevel2.py` |
| APGI-C02-EVL | Evolutionary / Genetic Gating | FP-5 | VP-5 | `FP_05_EvolutionaryPlausibility.py` | `VP_05_EvolutionaryEmergence.py` |
| APGI-C03-ENE | Energy / Thermodynamic Cost | FP-6 | VP-6 | `FP_06_LiquidNetworkEnergyBenchmark.py` | `VP_06_LiquidNetworkInductiveBias.py` |
| APGI-C04-MTH | Mathematical Consistency | FP-7 | VP-7² | `FP_07_MathematicalConsistency.py` | `VP_07_TMSCausalInterventions.py`² |
| APGI-C05-PRM | Parameter Identifiability | FP-8 | VP-8 | `FP_08_ParameterSensitivityIdentifiability.py` | `VP_08_PsychophysicalThresholdEstimation.py` |
| APGI-C06-BAY | Bayesian MCMC Estimation | FP-10 | VP-11 | `FP_10_BayesianEstimationMCMC.py` | `VP_11_MCMCCulturalNeurosciencePriority3.py` |
| APGI-X01-BNC | Agent Convergence Benchmark | FP-2 | VP-2 | `FP_02_AgentComparisonConvergenceBenchmark.py` | `VP_02_BehavioralBayesianComparison.py` |
| APGI-X02-FWK | Framework Multi-Protocol | FP-3 | — | `FP_03_FrameworkLevelMultiProtocol.py` | — |
| APGI-X03-LNN | Liquid Network Echo-State | FP-11 | VP-6 | `FP_11_LiquidNetworkDynamicsEchoState.py` | `VP_06_LiquidNetworkInductiveBias.py` |
| APGI-X04-XSP | Cross-Species Scaling | FP-12 | VP-12 | `FP_12_CrossSpeciesScaling.py` | `VP_12_ClinicalCrossSpeciesConvergence.py` |
| APGI-X05-EPS | Epistemic Architecture | — | VP-13 | — | `VP_13_EpistemicArchitecture.py` |
| APGI-X06-MRI | fMRI vmPFC Detailed | — | VP-15 | — | `VP_15_FMRIAnticipationVmPFC.py` |
| APGI-X07-MET | Metabolic ATP (³¹P-MRS) | — | VP-16 | — | `VP_16_MetabolicATPGroundTruth.py` |
| APGI-X08-VIS | Allen Visual / Fatigue | — | VP-17 | — | `VP_17_AllenVisualCodingFatigue.py` |
| APGI-X09-GFP | EEG Microstate GFP P3b | — | VP-18 | — | `VP_18_EEGMicrostateGFPP3b.py` |
| APGI-X10-MVA | Info Erasure / MVPA | — | VP-19 | — | `VP_19_InformationErasureMVPA.py` |
| APGI-X11-IEG | Empirical iEEG | — | VP-20 | — | `VP_20_EmpiricalIEEG.py` |
| APGI-X12-FEN | Free Energy / Prediction Error | — | VP-21 | — | `VP_21_FreeEnergyPredictionError.py` |
| APGI-G01-FWK | FP Aggregator (all FPs) | FP-ALL | — | `FP_ALL_Aggregator.py` | — |
| APGI-G02-FWK | VP Aggregator (all VPs) | — | VP-ALL | — | `VP_ALL_Aggregator.py` |
| APGI-G03-FWK | FP Orchestrator (Master) | — | — | `Master_Falsification.py` | — |
| APGI-G04-FWK | VP Orchestrator (Master) | — | — | — | `Master_Validation.py` |

> ¹ FP-9 serves both E01 (EEG signatures) and E02 (TMS causal) via shared neural-signature criteria.  
> ² VP-7 is the TMS validation file; a dedicated mathematical-consistency VP is absent (see improvement §6.3).

---

---

## Section 8 — Paper Protocol to VP/FP Mapping

*Sourced from codebase audit (April 2026). Maps the six numbered paper protocols and four epistemic roadmap priorities to their implementing VP/FP files. The **UPID** column cross-references the canonical entry in Sections 1–3.*

| Paper Protocol | UPID | VP/FP | Status | Notes |
| ---------------- | ----- | ------- | -------- | ------- |
| Paper Protocol 1 — EEG / Interoceptive Gating | APGI-E01-EEG | VP-9 | ✅ Full | HEP + P3b signatures; power-analysis gating |
| Paper Protocol 2 — Causal TMS / Insula | APGI-E02-TMS | VP-7, VP-10 | ✅ Full | Dual-file structure; VP-7 intervention, VP-10 causal priority |
| Paper Protocol 3 — Active Inference Agents | APGI-E03-AIN | VP-3 | ✅ Full | BIC/AIC and Mann-Whitney U |
| Paper Protocol 4 — Disorders of Consciousness | APGI-E04-DOC | VP-12 | ✅ Full | Liquid network included |
| Paper Protocol 5 — fMRI Anticipation / vmPFC | APGI-E05-MRI | VP-14, VP-15, VP-22 | ✅ Full | VP-14 primary; VP-15 vmPFC focus; VP-22 somatic marker emphasis |
| Paper Protocol 6 — Intracranial EEG / All-or-None Ignition | APGI-E06-IEG | VP-1, VP-2 | ✅ Split | VP-1 EEG/ML classification; VP-2 Behavioral Bayesian |
| Epistemic Paper P5–P8 (Phase Transition) | APGI-C01-PHT | VP-4 | ✅ Full | Metabolic cost driver patched |
| Epistemic Roadmap Priority 1 | APGI-E01-EEG | VP-9 | ✅ Full | Shared with Paper Protocol 1 |
| Epistemic Roadmap Priority 2 | APGI-E02-TMS | VP-10 | ✅ Full | Overlaps VP-7 |
| Epistemic Roadmap Priority 3 | APGI-C06-BAY | VP-11 | ✅ Full | Gelman-Rubin R̂ ≤ 1.01 |
| Epistemic Roadmap Priority 4 | APGI-E04-DOC | VP-12 | ✅ Shared | Shared with Paper Protocol 4 |

---

## Section 9 — Named Predictions Cross-Reference

*18 predictions tracked by `FP_ALL_Aggregator.py` via `PREDICTION_TO_PROTOCOL` dict. Each prediction maps to a source protocol and has a defined falsification criterion.*

| Prediction ID | Description | Source Protocol | Falsified if… |
| --------------- | ------------- | ----------------- | --------------- |
| P1.1 | Interoceptive precision modulates detection threshold (d=0.40–0.60) | FP-1 / `FP_01_ActiveInference.py` | d <0.35 or p≥0.01 |
| P1.2 | Arousal amplifies Πⁱ–threshold relationship | FP-1 / `FP_01_ActiveInference.py` | Interaction p≥0.01 |
| P1.3 | High-IA individuals show stronger arousal benefit | FP-1 / `FP_01_ActiveInference.py` | Effect absent or reversed |
| P2.a | dlPFC TMS shifts threshold >0.1 log units | VP-10 / `VP_10_CausalManipulationsPriority2.py` | Shift <0.05 log units |
| P2.b | Insula TMS reduces HEP ~30% AND PCI ~20% (double dissociation) | VP-10 / `VP_10_CausalManipulationsPriority2.py` | Either reduction absent |
| P2.c | High-IA × insula TMS interaction | VP-10 / `VP_10_CausalManipulationsPriority2.py` | Interaction p≥0.05 |
| P3.conv | APGI converges in 50–80 trials (beats baselines) | FP-2 / `FP_02_AgentComparisonConvergenceBenchmark.py` | APGI >100 trials to criterion |
| P3.bic | APGI BIC lower than StandardPP and GWTOnly | FP-2 / `FP_02_AgentComparisonConvergenceBenchmark.py` | ΔBIC <10 vs alternatives |
| P4.a | PCI+HEP joint AUC >0.80 for DoC classification | FP-9 / `FP_09_NeuralSignaturesP3bHEP.py` | AUC <0.75 |
| P4.b | DMN to PCI r>0.50; DMN to HEP r<0.20 | FP-9 / `FP_09_NeuralSignaturesP3bHEP.py` | Either correlation outside range |
| P4.c | Cold pressor increases PCI >10% in MCS, not VS | FP-9 / `FP_09_NeuralSignaturesP3bHEP.py` | No differential response |
| P4.d | Baseline PCI+HEP predicts 6-month recovery ΔR²>0.10 | FP-9 / `FP_09_NeuralSignaturesP3bHEP.py` | ΔR² <0.05 |
| P5.a | vmPFC to SCR anticipatory correlation r>0.40 | FP-5 / `FP_05_EvolutionaryPlausibility.py` | r <0.25 |
| P5.b | vmPFC uncorrelated with posterior insula (r<0.20) | FP-5 / `FP_05_EvolutionaryPlausibility.py` | r >0.30 |
| fp10a_mcmc | Bayesian MCMC: Gelman–Rubin R̂ ≤1.01 (convergence) | FP-10 / `FP_10_BayesianEstimationMCMC.py` | R̂ >1.01 |
| fp10b_bf | BF₁₀ ≥3 for APGI vs StandardPP / GWT | FP-10 / `FP_10_BayesianEstimationMCMC.py` | BF₁₀ <3 |
| fp10c_mae | APGI ≥20% lower MAE than alternatives | FP-10 / `FP_10_BayesianEstimationMCMC.py` | MAE reduction <20% |
| fp10_scaling | Cross-species scaling: allometric exponents within ±2 SD | FP-12 / `FP_12_CrossSpeciesScaling.py` | Exponents >2 SD from expectation |

**FP-10 note:** FP-10 is internally split into FP10a (Bayesian MCMC) and FP10b (Cross-Species Scaling via FP-12). Both must pass; either failure falsifies FP-10.

---
