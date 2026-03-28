# APGI Protocol Mapping — Python Files ↔ Protocols

---

## Table of Contents

1. [File Inventory](#1-file-inventory)
2. [Master Validation Orchestrator](#2-master-validation-orchestrator)
3. [Validation Protocol Mapping (VP-1 → VP-12)](#3-validation-protocol-mapping)
4. [Falsification Protocol Mapping (FP-1 → FP-12)](#4-falsification-protocol-mapping)
5. [Named Predictions → Protocol Routing](#5-named-predictions--protocol-routing)
6. [Criterion Code Cross-Reference](#6-criterion-code-cross-reference)
7. [Architecture & Shared Utilities](#7-architecture--shared-utilities)
8. [Implementation Gaps & Notes](#8-implementation-gaps--notes)

---

## 1. File Inventory

### 1.1 Validation Files (16 total)

 | #  Filename  VP ID  Paper Protocol  Tier  Completion  |
 | ---  ---  ---  ---  ---  ---  ---  |
 | 1  `ActiveInference_AgentSimulations_Protocol3.py`  VP-3  Paper Protocol 3  Secondary  82%  |
 | 2  `BayesianModelComparison_ParameterRecovery.py`  VP-1 (support)  Paper Protocol 6 (partial)  Primary  62%  |
 | 3  `Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py`  VP-10  Roadmap Priority 2  Tertiary  70%  |
 | 4  `Clinical_CrossSpecies_Convergence_Protocol4.py`  VP-12  Paper Protocol 4 / Roadmap Priority 4  Secondary  76%  |
 | 5  `ConvergentNeuralSignatures_Priority1_EmpiricalRoadmap.py`  VP-9  Roadmap Priority 1  Tertiary  73%  |
 | 6  `EvolutionaryEmergence_AnalyticalValidation.py`  VP-5  — (Paper Protocol 5 MISSING)  Tertiary  66%  |
 | 7  `InformationTheoretic_PhaseTransition_Level2.py`  VP-4  Epistemic Paper P5–P8 (Level 2)  Secondary  74%  |
 | 8  `Master_Validation.py`  —  Orchestrator (not a protocol)  —  Active  |
 | 9  `NeuralNetwork_InductiveBias_ComputationalBenchmark.py`  VP-6  Computational Architecture  Tertiary  68%  |
 | 10  `Psychophysical_ThresholdEstimation_Protocol1.py`  VP-8  Paper Protocol 1  Secondary  77%  |
 | 11  `QuantitativeModelFits_SpikingLNN_Priority3.py`  VP-11  Roadmap Priority 3  Secondary  75%  |
 | 12  `SyntheticEEG_MLClassification.py`  VP-1 (support)  Paper Protocol 6 (partial)  Primary  62%  |
 | 13  `TMS_Pharmacological_CausalIntervention_Protocol2.py`  VP-7  Paper Protocol 2  Secondary  79%  |
 | 14  `Validation_Protocol_2.py`  VP-2  Paper Protocol 6 (behavioral)  Primary  71%  |
 | 15  `Validation_Protocol_11.py`  VP-11 (canonical)  Roadmap Priority 3 / Cultural Neuro  Secondary  75%  |
 | 16  `Validation_Protocol_P4_Epistemic.py`  VP-4 (support)  Epistemic Paper P5–P12  Secondary  74%  |

### 1.2 Falsification Files (22 total)

 | #  Filename  FP ID  Role  Completion  |
 | ---  ---  ---  ---  ---  |
 | 1  `APGI_Falsification_Aggregator.py`  FP-12  Framework-level aggregator  76%  |
 | 2  `APGI_Falsification_Protocols_GUI.py`  —  GUI runner (not a protocol)  Active  |
 | 3  `Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py`  FP-2 (support)  Causal manipulation engine  72%  |
 | 10  `Falsification_ActiveInferenceAgents_F1F2.py`  FP-1 (full)  Primary F1/F2 engine  74%  |
 | 11  `Falsification_AgentComparison_ConvergenceBenchmark.py`  FP-2 (full)  Convergence benchmark  72%  |
 | 12  `Falsification_BayesianEstimation_MCMC.py`  FP-10  NUTS/PyMC MCMC estimation  55%  |
 | 13  `Falsification_BayesianEstimation_ParameterRecovery.py`  FP-11  Parameter recovery  67%  |
 | 14  `Falsification_CrossSpeciesScaling_P12.py`  FP-10 (cross-species)  Allometric scaling  55%  |
 | 15  `Falsification_EvolutionaryPlausibility_Standard6.py`  FP-5 (full)  Evolutionary analysis  70%  |
 | 16  `Falsification_FrameworkLevel_MultiProtocol.py`  FP-3 (full)  Multi-protocol synthesis  65%  |
 | 17  `Falsification_InformationTheoretic_PhaseTransition.py`  FP-4 (full)  Phase transition analysis  73%  |
 | 18  `Falsification_LiquidNetworkDynamics_EchoState.py`  FP-11 / FP-12  Echo-state / liquid network  67%  |
 | 19  `Falsification_MathematicalConsistency_Equations.py`  FP-7  Sympy equation verification  63%  |
 | 20  `Falsification_NeuralNetwork_EnergyBenchmark.py`  FP-6 (full)  Energy / ATP benchmark  68%  |
 | 21  `Falsification_NeuralSignatures_EEG_P3b_HEP.py`  FP-9  EEG / P3b / HEP neural sigs  62%  |
 | 22  `Falsification_ParameterSensitivity_Identifiability.py`  FP-8  Sobol / FIM / profile likelihood  60%  |

---

## 2. Master Validation Orchestrator

**File:** `Master_Validation.py`
**Class:** `APGIMasterValidator`

Coordinates all 12 validation protocols. Tier classification and protocol registration are defined here — **this is the authoritative source for VP numbering**.

### 2.1 Tier Registration (from `PROTOCOL_TIERS` dict, lines 46–59)

 | VP Number  Tier  Weight  |
 | ---  ---  ---  | |
 | 1, 2  Primary  50% combined  |
 | 3, 4, 8, 11, 12  Secondary  30% combined  |
 | 5, 6, 7, 9, 10  Tertiary  20% combined  |

> **Note:** The tier rationale document assigns VP-7 and VP-8 to secondary/tertiary differently than the Master file. The `Master_Validation.py` values are ground truth for runtime scoring.

### 2.2 Protocol File Registration (lines 83–140)

The Master registers protocol *names* as `Validation_Protocol_N.py`, but the actual implementation files in the ZIP often differ. The table below resolves this:

 | VP Registered Name  Actual Implementation File  Notes  |
 | ---  ---  ---  | |
 | `Validation_Protocol_1.py`  `SyntheticEEG_MLClassification.py`  Self-documents as "NOT Protocol 1" — supports VP-1 predictions  |
 | `Validation_Protocol_2.py`  `Validation_Protocol_2.py`  ✅ Matches  |
 | `Validation_Protocol_3.py`  `ActiveInference_AgentSimulations_Protocol3.py`  ✅ Matches  |
 | `Validation_Protocol_P4_Epistemic.py`  `Validation_Protocol_P4_Epistemic.py` + `InformationTheoretic_PhaseTransition_Level2.py`  Two files share VP-4  |
 | `Validation_Protocol_5.py`  `EvolutionaryEmergence_AnalyticalValidation.py`  ✅ Matches (fMRI Paper Protocol 5 still MISSING)  |
 | `Validation_Protocol_6.py`  `NeuralNetwork_InductiveBias_ComputationalBenchmark.py`  ✅ Matches  |
 | `Validation_Protocol_7.py`  `TMS_Pharmacological_CausalIntervention_Protocol2.py`  ✅ Matches  |
 | `Validation_Protocol_8.py`  `Psychophysical_ThresholdEstimation_Protocol1.py`  ✅ Matches  |
 | `Validation_Protocol_9.py`  `ConvergentNeuralSignatures_Priority1_EmpiricalRoadmap.py`  ✅ Matches  |
 | `Validation_Protocol_10.py`  `Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py`  ✅ Matches  |
 | `Validation_Protocol_11.py`  `Validation_Protocol_11.py` + `QuantitativeModelFits_SpikingLNN_Priority3.py`  Two files share VP-11  |
 | `Validation_Protocol_12.py`  `Clinical_CrossSpecies_Convergence_Protocol4.py`  ✅ Matches  |

---

## 3. Validation Protocol Mapping

---

### VP-1 · Synthetic Neural Data / ML Classification

**Primary files:**

- `SyntheticEEG_MLClassification.py` — main computational engine
- `BayesianModelComparison_ParameterRecovery.py` — Bayesian model comparison support

**Paper protocol:** Paper Protocol 6 (Parameter Estimation) — partial
**Tier:** Primary | **Completion:** 62%

**Key classes:**

 | Class  File  Purpose  |
 | ---  ---  ---  |
 | `APGIDynamicalSystem`  SyntheticEEG_MLClassification.py  Simulates APGI ODE dynamics  |
 | `APGISyntheticSignalGenerator`  SyntheticEEG_MLClassification.py  Generates synthetic EEG-like signals  |
 | `StandardPredictiveProcessingGenerator`  SyntheticEEG_MLClassification.py  Comparison generator (standard PP)  |
 | `GlobalWorkspaceOnlyGenerator`  SyntheticEEG_MLClassification.py  Comparison generator (GWT-only)  |
 | `IgnitionClassifier`  SyntheticEEG_MLClassification.py  CNN classifier for ignition detection  |
 | `MultiModalFusionNetwork`  SyntheticEEG_MLClassification.py  Multimodal fusion for model ID  |
 | `FalsificationChecker`  SyntheticEEG_MLClassification.py  F1.x / F2.x / F3.x / F5.x criteria gate  |
 | `Protocol1Psychophysics`  SyntheticEEG_MLClassification.py  Near-threshold psychophysics simulation  |
 | `APGIGenerativeModel`  BayesianModelComparison_ParameterRecovery.py  PyMC APGI generative model  |
 | `BayesianModelComparison`  BayesianModelComparison_ParameterRecovery.py  LOO-CV / WAIC comparison  |
 | `FalsificationChecker`  BayesianModelComparison_ParameterRecovery.py  F2.1–F2.5 Bayesian criteria  |

**Key functions:**

 | Function  File  Line  Purpose  |
 | | ------  ------  ---------  |
 | `compare_models_with_statistics()`  SyntheticEEG_MLClassification.py  3618  McNemar + DeLong AUC (≥1,000 permutations)  |
 | `train_ignition_classifier()`  SyntheticEEG_MLClassification.py  1613  Trains CNN for ignition detection  |
 | `validate_parameter_recovery()`  BayesianModelComparison_ParameterRecovery.py  805  n=100 simulations, r≥0.82 core params  |
 | `compute_fisher_information_matrix()`  BayesianModelComparison_ParameterRecovery.py  861  FIM identifiability  |
 | `analyze_beta_pi_identifiability()`  BayesianModelComparison_ParameterRecovery.py  1047  β / Πⁱ collinearity check

**Criterion codes implemented:** F1.1, F1.2, F1.3, F1.4, F2.1, F2.1A, F2.1B, F2.2, F2.3, F2.4, F2.5, F3.1, F3.1A, F3.1B, F3.2, F3.3, F3.4, F3.5, F3.6, F5.1, F5.2, F5.3 (+ F5.4, F5.5, F5.6, F6.1, F6.2 via FalsificationChecker), V12.1

**Important self-documentation note:**
`SyntheticEEG_MLClassification.py` explicitly states at line 13: *"This is NOT Protocol 1. The actual Protocol 1 is 'Interoceptive Precision Modulates Detection Threshold'."* It provides computational support for VP-1 predictions, not the human-participant psychophysics paradigm itself.

---

### VP-2 · Behavioral Validation

**Primary file:** `Validation_Protocol_2.py`
**Paper protocol:** Paper Protocol 6 (behavioral arm) — partial
**Tier:** Primary | **Completion:** 71%

**Key classes:**

 | Class  File  Purpose  |
 | ---  ---  ---  |
 | `SyntheticSubject`  Validation_Protocol_2.py  Simulates participant behavior  |
 | `APGIValidationProtocol11` (inner)  Validation_Protocol_2.py  Behavioral simulation driver  |

**Key functions:**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `apgi_detection_probability()`  125  Logistic psychometric function: P = σ(β(S−θ))  |
 | `_null_detection_probability()`  157  Baseline (chance-level) detection  |
 | `_extero_only_probability()`  165  Exteroceptive-only comparison model  |
 | `generate_synthetic_dataset()`  189  Generates behavioral datasets with IA-split  |
 | `run_mh_sampler()`  330  Metropolis-Hastings MCMC fallback  |
 | `_compute_rhat()`  435  Gelman-Rubin R̂ diagnostic  |
 | `_compute_ess()`  469  Effective sample size  |
 | `run_nuts_sampler()`  493  PyMC NUTS (primary MCMC method)  |
 | `test_parameter_recovery()`  582  Recovers {θ₀, Πⁱ, β, α}  |
 | `compute_model_comparison()`  674  LOO-CV / WAIC model comparison  |
 | `test_individual_differences()`  787  IA-split group analysis  |

**Criterion codes implemented:** P1.1, P1.2, P1.3 (via `get_falsification_criteria()`), V11.1–V11.5 (via inner Bayesian loop)

---

### VP-3 · Agent Comparison Experiment (Active Inference)

**Primary file:** `ActiveInference_AgentSimulations_Protocol3.py`
**Paper protocol:** Paper Protocol 3 — Active Inference Agents
**Tier:** Secondary | **Completion:** 82%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `HierarchicalGenerativeModel`  70  3-level hierarchical generative model (τ₁/τ₂/τ₃)  |
 | `SomaticMarkerNetwork`  192  Somatic marker prediction for interoception  |
 | `PolicyNetwork`  252  Action selection with precision-weighted softmax  |
 | `HabitualPolicy`  367  Habitual/cached action policy  |
 | `WorkingMemory`  399  Capacity-7 working memory buffer  |
 | `EpisodicMemory`  418  Context-indexed episodic store  |
 | `APGIActiveInferenceAgent`  464  Full APGI agent (main implementation)  |
 | `StandardPPAgent`  780  Comparison: standard predictive processing  |
 | `GWTOnlyAgent`  844  Comparison: global workspace theory only  |
 | `ActorCriticAgent`  921  Comparison: actor-critic RL baseline  |
 | `IowaGamblingTaskEnvironment`  949  IGT task environment  |
 | `MultiArmedVolatileBandit`  1035  Volatile bandit (uncertainty manipulation)  |
 | `PatchLeavingForagingEnvironment`  1110  Foraging task with metabolic costs  |
 | `WAICModelComparison`  1274  WAIC / LOO-CV model selection  |
 | `AgentComparisonExperiment`  1431  Orchestrates 3-way agent comparison  |
 | `SystematicAblationStudy`  1201, 2747  Ablation of APGI components  |

**Key functions:**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `APGIActiveInferenceAgent.step()`  554  One step: prediction → ignition → action  |
 | `APGIActiveInferenceAgent._update_precision()`  714  Precision update: Πⁱ_eff = Πⁱ · exp(β·M)  |
 | `APGIActiveInferenceAgent._compute_metabolic_cost()`  737  ATP cost per step  |

**Criterion codes implemented:** V3.1, V3.2, F3.1, F3.2, F3.3, F5.1–F5.6, F6.1, F6.2
**Named predictions tested:** P3.conv (50–80 trial convergence), P3.bic (APGI BIC < StandardPP BIC < GWTonly BIC)

---

### VP-4 · Phase Transition Analysis (Information-Theoretic)

**Primary files:**

- `InformationTheoretic_PhaseTransition_Level2.py` — Level 2 epistemic tests
- `Validation_Protocol_P4_Epistemic.py` — P5–P12 wrapper

**Paper protocol:** Epistemic Architecture Paper — Level 2 (P5–P8) and Level 1 (P9–P12)
**Tier:** Secondary | **Completion:** 74%

**Key classes (Level 2 file):**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `APGIDynamicalSystem`  90  ODE solver for surprise accumulation  |
 | `InformationTheoreticAnalysis`  228  Transfer entropy, mutual information, Φ  |
 | `PhaseTransitionDetector`  446  Detects discontinuities at ignition boundary  |
 | `ClinicalPowerAnalysis`  795  Sample size / power for clinical studies  |
 | `FiniteSizeScalingAnalysis`  935  Finite-size scaling (critical exponents)  |
 | `IntracranialRecordingPipeline`  1690  iEEG-compatible analysis pipeline  |
 | `ComprehensivePhaseTransitionAnalysis`  1967  Orchestrator for all phase transition tests  |
 | `FalsificationChecker`  2176  F3.x, F4.x, F5.x, F6.x criteria gates  |
 | `ClinicalDoCBiomarkerValidation`  2829  DoC biomarker (MCS/VS classification)  |

**Key classes (Epistemic wrapper):**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `EpistemicArchitectureValidator`  88  Routes P5–P12 to correct sub-analyzers  |

**Key functions:**

 | Function  File  Line  Purpose  |
 | | ------  ------  ---------  |
 | `run_validation()`  Level2.py  3330  Entry point for VP-4
 | `get_falsification_criteria()`  Level2.py  3348  Returns F3.x/F4.x criteria dict
 | `run_validation()`  P4_Epistemic.py  650  Entry point for P5–P12

**Criterion codes implemented:** F3.1–F3.6, F4.1–F4.5, F5.1–F5.6, F6.1, F6.2, V4.1

**Epistemic predictions tested:**

- P5: MI increase ≥1 bit with precision cueing
- P6: Bandwidth asymptotes at ~40 bits/s (biological ceiling, corrected from 100 bits/s)

- P7: Optimal Bayesian detector (Neyman-Pearson, ≤2 SD deviation)

- P8: Information erasure in backward masking

- P9: Metabolic cost — Attwell & Laughlin (2001) ~10⁹ ATP/spike

- P10: Energy efficiency advantage

- P11: Fatigue threshold dynamics

- P12: Cross-species scaling consistency

---

### VP-5 · Evolutionary Emergence

**Primary file:** `EvolutionaryEmergence_AnalyticalValidation.py`
**Paper protocol:** MISSING — Paper Protocol 5 (fMRI Anticipation/Experience) has no corresponding implementation
**Tier:** Tertiary | **Completion:** 66%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `EvolvableAgent`  80  Agent with mutable APGI genome  |
 | `ContinuousUpdateAgent`  269  Continuous parameter update variant  |
 | `EvolutionaryAPGIEmergence`  446  Evolutionary simulation engine  |

**Key functions:**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `run_falsification()`  848  Entry point  |
 | `get_falsification_criteria()`  867  Returns F5.x criteria  |
 | `check_falsification()`  923  Applies falsification gate  |

**Criterion codes implemented:** F5.1 (≥75% agents develop multi-timescale threshold), F5.5 (PCA variance ≥70%, loading ≥0.60)
**Critical fix:** Cross-modal check now uses `abs(intero_error − extero_error) < 1e-6` tolerance

---

### VP-6 · Network Comparison (Neural Network Inductive Bias)

**Primary file:** `NeuralNetwork_InductiveBias_ComputationalBenchmark.py`
**Paper protocol:** Computational Architecture Energy Comparison
**Tier:** Tertiary | **Completion:** 68%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `SpikeEnergyMonitor`  127  ATP spike counting (Attwell & Laughlin)  |
 | `APGIInspiredNetwork`  380  Dual-pathway network with precision + ignition  |
 | `StandardMLPNetwork`  616  Baseline MLP comparison  |
 | `LSTMNetwork`  656  LSTM comparison  |
 | `AttentionNetwork`  702  Attention-mechanism comparison  |
 | `ConsciousClassificationDataset`  743  Conscious/unconscious trial data  |
 | `NetworkTrainer`  1067  Training loop with energy monitoring  |
 | `NetworkComparison`  1349  Runs and compares all architectures  |
 | `FalsificationChecker`  1525  F1.x–F6.x criteria gate  |
 | `APGIValidationProtocol6`  3705  VP-6 entry point class  |
 | `TemporalDynamicsValidator`  3726  LTCN transition time validation  |
 | `AdaptiveThresholdChecker`  3740  Threshold adaptation dynamics  |

**Key functions:**

 | Function  Line  Purpose  |
 | ----------  ------  ---------  |
 | `run_validation()`  2639  Entry point  |
 | `get_falsification_criteria()`  2657  Returns F1.x–F6.x criteria  |
 | `check_falsification()`  2852  Applies gate  |

**Criterion codes implemented:** F1.1–F1.6, F2.1–F2.5, F3.1–F3.6, F5.1–F5.6 (same shared F-criteria as other protocols)
**Protocol-specific criteria:** F6.1 (LTCN ≤50ms), F6.2 (window 200–500ms, ratio ≥4×, R² ≥0.85)

---

### VP-7 · TMS / Pharmacological Causal Interventions

**Primary file:** `TMS_Pharmacological_CausalIntervention_Protocol2.py`
**Paper protocol:** Paper Protocol 2 — TMS Causal Manipulations
**Tier:** Secondary | **Completion:** 79%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `InterventionEffect`  61  Data container: effect size, CI, p-value  |
 | `TMSInterventions`  111  dlPFC / insula TMS simulation  |
 | `PharmacologicalInterventions`  196  Propranolol, atomoxetine, propofol, ketamine, psilocybin  |
 | `PsychometricCurve`  290  Logistic psychometric fitting  |
 | `InterventionStudySimulator`  464  Full intervention study simulation  |
 | `PowerAnalysis`  628  Sample size calculation for each prediction  |
 | `InterventionFalsificationChecker`  715  F3.x, V7.x, P7.x criteria gate  |
 | `APGIValidationProtocol7`  2719  VP-7 entry point class  |
 | `HierarchicalProcessingValidator`  2740  Hierarchical level validation  |
 | `LevelEmergenceChecker`  2754  Level emergence criteria  |

**Key functions:**

 | Function  Line  Purpose  |
 | ----------  ------  ---------  |
 | `logistic_psychometric()`  283  σ(β(S−θ)) — APGI psychometric function  |
 | `run_validation()`  2458  Entry point  |
 | `get_falsification_criteria()`  2476  Returns V7.x, P7.x, F3.x criteria  |
 | `check_falsification()`  2524  Applies gate  |
 | `correct_for_multiple_comparisons()`  1871  Bonferroni / Holm correction  |
 | `bayesian_equivalence_test()`  1958  ROPE-based Bayesian equivalence  |

**Criterion codes implemented:** F3.1–F3.6, V7.1, V7.2, P7.1, P7.2, P7.3

**Predictions tested:**

- V7.1: TMS → θ_t reduction ≥15%, lasting ≥60min
- V7.2: Propranolol → Π_i increase ≥25%, ignition reduction ≥30%

- P7.1: Propofol → P3b:MMN ratio ≥1.5:1

- P7.2: Ketamine → MMN ≥20% suppression, P3b <50% suppression

- P7.3: Psilocybin → P3b ≥10% increase for low-salience; HEP-embodiment r≥0.20

---

### VP-8 · Psychophysical Threshold Estimation

**Primary file:** `Psychophysical_ThresholdEstimation_Protocol1.py`
**Paper protocol:** Paper Protocol 1 — Interoceptive Precision Modulates Detection Threshold
**Tier:** Secondary | **Completion:** 77%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `APGIParameters`  40  Parameter container {θ₀, Πⁱ, β, α, τ_θ}  |
 | `ParticipantData`  58  Trial-level participant data  |
 | `PsiMethod`  115  Bayesian adaptive psychophysics (Psi method)  |
 | `APGIPsychophysicalEstimator`  197  Estimates APGI params from psychometric curves  |
 | `APGIValidationProtocol8`  2443  VP-8 entry point class  |
 | `PrecisionWeightingValidator`  2464  Interoceptive vs. exteroceptive weighting  |
 | `InteroceptiveBiasChecker`  2549  IA-group bias analysis  |

**Key functions:**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `run_validation()`  1196  Entry point  |
 | `get_falsification_criteria()`  1301  Returns F1.x–F3.x criteria + V8.x  |
 | `check_falsification()`  1503  Applies gate (Bonferroni: α=0.008 per 6 tests)  |
 | `validate_disorder_parameters()`  2629  θ_t, Πⁱ, arousal vs. disorder table ±10%  |

**Criterion codes implemented:** F1.1–F1.6, F2.1–F2.5, F3.1–F3.6, F5.1–F5.3, V8.1–V8.4

**Named predictions tested:**

- P1.1: d = 0.40–0.60 interoceptive precision modulation
- P1.2: Arousal × Πⁱ interaction d = 0.25–0.45

- P1.3: High-IA group benefit d > 0.30

**Key implementation details:**

- Bonferroni correction: 6-test battery, α_per_test = 0.008
- F3.1 inline threshold: 0.18

- F3.6 inline threshold: 200 trials

- Two-pathway pharmacological disambiguation: β-blockade (25–40% reduction) vs. cardiac feedback (15–25% Πⁱ reduction)

---

### VP-9 · Convergent Neural Signatures (Roadmap Priority 1)

**Primary file:** `ConvergentNeuralSignatures_Priority1_EmpiricalRoadmap.py`
**Paper protocol:** Empirical Credibility Roadmap Priority 1
**Tier:** Tertiary | **Completion:** 73%

**Key functions:**

 | Function  Purpose  |
 | ---  ---  |
 | `run_validation()`  Entry point  |
 | `get_falsification_criteria()`  Returns F6.x and P5/P6/P7 criteria  |

**Criterion codes implemented:** F6.1 (standardised to ≤50.0ms), F6.2 (≥200.0ms, ≥4.0×, R²≥0.85), P5, P6, P7

**Key implementation details:**

- Power analysis gating: criteria flagged "UNDERPOWERED" when estimated power < 0.80
- P7 uses Neyman-Pearson criterion with bootstrap CI (deviation ≤2 SD from optimal threshold)

---

### VP-10 · Causal Manipulations (Roadmap Priority 2)

**Primary file:** `Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py` (Validation folder)
**Also:** `Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py` (Falsification folder) — note: same filename, different folder, different content

**Paper protocol:** Empirical Credibility Roadmap Priority 2
**Tier:** Tertiary | **Completion:** 70%

**Key classes (Validation file):**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `TMSIntervention`  278  TMS pulse simulation  |
 | `TACSIntervention`  158  tACS oscillatory intervention  |
 | `PharmacologicalIntervention`  191  Drug administration simulation  |
 | `MetabolicIntervention`  246  Glucose / metabolic challenge  |
 | `CausalManipulationsValidator`  291  Validates all causal predictions  |
 | `SubliminalPrimingMeasure`  982  Subliminal priming dissociation test  |
 | `APGIValidationProtocol10`  1569  VP-10 entry point class  |
 | `FeatureClusteringValidator`  1590  Feature clustering (F5.5 gate)  |

**Key classes (Falsification file):**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `TMSIntervention`  71  Extended TMS model  |
 | `PharmacologicalIntervention`  139  Pharmacological simulation  |
 | `MetabolicIntervention`  225  Metabolic challenge  |
 | `ColdPressorTest`  299  Cold pressor (Protocol 4c)  |
 | `CausalManipulationsValidator`  388  Core validation engine  |
 | `MNEDataInterface`  1143  Real EEG data input via MNE  |

**Named predictions tested:** P2.a (>0.1 log units threshold shift), P2.b (≥30% HEP AND ≥20% PCI reduction — double dissociation), P2.c (high-IA × insula TMS interaction η² ≥ 0.10), P5.a (vmPFC-SCR r > 0.40), P5.b (vmPFC uncorrelated with posterior insula r < 0.20)

**APGI-derived confidence:** σ(Πⁱ·| εᵢ |−θₜ) — replaces heuristic confidence measure

---

### VP-11 · Bayesian Estimation / Cultural Neuroscience (Roadmap Priority 3)

**Primary files:**

- `Validation_Protocol_11.py` — canonical Bayesian parameter recovery
- `QuantitativeModelFits_SpikingLNN_Priority3.py` — spiking LNN model fits

**Paper protocol:** Empirical Credibility Roadmap Priority 3
**Tier:** Secondary | **Completion:** 75%

**Key classes (Validation_Protocol_11.py):**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `SyntheticSubject`  178  Generates individual-difference data  |
 | `APGIValidationProtocol11`  1308  VP-11 entry point class  |

**Key functions (Validation_Protocol_11.py):**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `apgi_detection_probability()`  125  APGI psychometric function  |
 | `_null_detection_probability()`  157  Null model  |
 | `_extero_only_probability()`  165  Exteroceptive-only model  |
 | `run_mh_sampler()`  330  Metropolis-Hastings fallback  |
 | `_compute_rhat()`  435  Gelman-Rubin R̂ (R̂ ≤ 1.01 gate)  |
 | `_compute_ess()`  469  Effective sample size  |
 | `run_nuts_sampler()`  493  PyMC NUTS: 4 chains, 5,000 samples, 1,000 burn-in  |
 | `test_parameter_recovery()`  582  Recovers {θ₀, Πⁱ, β, α} — r≥0.82 core  |
 | `compute_model_comparison()`  674  LOO-CV / WAIC: BF₁₀ ≥ 10  |
 | `test_individual_differences()`  787  Cultural / IA-group parameter variation  |

**Criterion codes implemented:** V11.1–V11.5

**Key classes (QuantitativeModelFits_SpikingLNN_Priority3.py):**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `PsychometricFunctionFitter`  54  Psychometric curve fitting|
 | `SpikingLNNModel`  232  Leaky neural network (spiking dynamics)|
 | `BayesianParameterEstimator`  467  Bayesian estimation for spiking LNN|
 | `ConvergenceBenchmark`  863  Convergence speed analysis|
 | `ModelComparisonTable`  1028  GNW / additive linear vs. APGI|
 | `QuantitativeModelValidator`  1306  Orchestrator|
 | `APGIValidationProtocol11`  2872  Entry point class (mirrors main file)|
 | `NonAPGIComparisonValidator`  2893  F5.6-style ablation|
 | `ArchitectureFailureChecker`  3028  Non-APGI architecture failure|

---

### VP-12 · Clinical / Cross-Species Convergence (Roadmap Priority 4)

**Primary file:** `Clinical_CrossSpecies_Convergence_Protocol4.py`
**Paper protocol:** Paper Protocol 4 (DoC) / Roadmap Priority 4
**Tier:** Secondary | **Completion:** 76%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `ClinicalDataAnalyzer`  75  P3b / frontoparietal activation in DoC  |
 | `PsychiatricProfileAnalyzer`  234  APGI params in GAD, MDD, Psychosis  |
 | `CrossSpeciesHomologyAnalyzer`  468  Ignition signatures across species  |
 | `IITConvergenceAnalyzer`  670  APGI ↔ IIT convergence (Φ correlation)  |
 | `LongitudinalOutcomePredictor`  769  6-month recovery prediction  |
 | `AutonomicPerturbationAnalyzer`  960  Cold pressor / breathlessness (P4.c)  |
 | `ClinicalPowerAnalyzer`  1206  Sample size for clinical studies  |
 | `ClinicalConvergenceValidator`  1396  Orchestrator  |
 | `APGIValidationProtocol12`  2897  VP-12 entry point class  |
 | `IntrinsicBehaviorValidator`  2918  Behavioral autonomy (O'Reilly & Frank 2006)  |
 | `LiquidTimeConstantChecker`  3013  Echo-state network: F6.2 validation  |

**Key functions:**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `run_validation()`  1781  Entry point
 | `get_falsification_criteria()`  1808  Returns V12.x and F-criteria
 | `check_falsification()`  2010  Applies gate

**Criterion codes implemented:** F1.1–F1.6, F2.1–F2.5, F3.1–F3.6, F5.1–F5.3, V12.1 (P3b ≥80%, ignition ≥70%), V12.2 (r ≥ 0.60 cross-species)

**Named predictions tested:** P4.a–P4.d (PCI+HEP AUC, DMN connectivity, cold pressor, 6-month recovery)

---

## 4. Falsification Protocol Mapping

---

### FP-1 · Active Inference Agents (F1/F2)

**Primary file:** `Falsification_ActiveInferenceAgents_F1F2.py`
**Stub file:** `Falsification-Protocol-1.py` (thin wrapper, delegates to full file)
**Tier:** Primary | **Completion:** 74%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `HierarchicalGenerativeModel`  440  3-level hierarchical model  |
 | `SomaticMarkerNetwork`  522  Somatic marker anticipation  |
 | `PolicyNetwork`  649  Precision-weighted policy  |
 | `HabitualPolicy`  795  Habitual action selection  |
 | `EpisodicMemory`  854  Context-indexed memory  |
 | `WorkingMemory`  995  7-item capacity buffer  |
 | `APGIActiveInferenceAgent`  1010  Full APGI agent  |
 | `StandardPPAgent`  1447  Standard PP comparison  |
 | `GWTOnlyAgent`  1558  GWT-only comparison  |

**Key functions:**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `bootstrap_ci()`  143  Bootstrap confidence intervals
 | `bootstrap_one_sample_test()`  173  Bootstrap one-sample significance
 | `check_F5_family()`  233  F5.1–F5.6 criteria check (shared)
 | `analyze_bifurcation_structure()`  916  Phase portrait sweep for F6.5
 | `run_falsification()`  1772  Entry point
 | `get_falsification_criteria()`  1890  Full criteria dict
 | `check_falsification()`  1975  Gate function

**Dependencies:** `utils.constants.LEVEL_TIMESCALES`, `specparam.FOOOF` (fallback: `fooof.FOOOF`)

---

### FP-2 · Agent Comparison / Convergence Benchmark

**Primary file:** `Falsification_AgentComparison_ConvergenceBenchmark.py`
**Stub file:** `Falsification-Protocol-2.py` (Iowa Gambling Task wrapper)
**Support file:** `Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py` (Falsification folder)
**Tier:** Primary | **Completion:** 72%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `IowaGamblingTaskEnvironment`  156  IGT environment  |
 | `VolatileForagingEnvironment`  302  Volatile foraging task  |
 | `ThreatRewardTradeoffEnvironment`  435  Threat/reward trade-off  |

**Key functions:**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `bootstrap_ci()`  80  Bootstrap CI
 | `bootstrap_one_sample_test()`  110  Bootstrap test |
 | `run_falsification()`  556  Entry point
 | `get_falsification_criteria()`  584  Criteria dict (shared F1–F3 + F5)
 | `check_falsification()`  633  Gate function

**Shared criteria with FP-1:** F5.1–F5.6 via imported `check_F5_family()` from `utils.shared_falsification`
**FP-2 specific:** Survival analysis (log-rank test) for F2.5 using `lifelines.KaplanMeierFitter` (scipy fallback)

---

### FP-3 · Framework-Level Multi-Protocol

**Primary file:** `Falsification_FrameworkLevel_MultiProtocol.py`
**Stub file:** `Falsification-Protocol-3.py` (neural network analysis wrapper)
**Tier:** Secondary | **Completion:** 65%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `StandardPPAgent`  317  PP comparison agent  |
 | `GWTOnlyAgent`  358  GWT comparison agent  |
 | `StandardActorCriticAgent`  408  AC comparison agent  |
 | `AgentComparisonExperiment`  451  Orchestrates comparison  |

**Key functions:**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `_get_protocol1()`  165  Dynamic import of FP-1
 | `_get_protocol2()`  199  Dynamic import of FP-2
 | `check_framework_falsification_conditions()`  1695  Conditions A and B
 | `run_falsification()`  1797  Entry point
 | `get_falsification_criteria()`  1823  Full criteria dict
 | `check_falsification()`  1878  Gate function

**Path fix:** Protocol 3 routes to `../Validation/ActiveInference_AgentSimulations_Protocol3.py`
**Framework conditions delegated to:** `APGI_Falsification_Aggregator.py`

---

### FP-4 · Information-Theoretic Phase Transition

**Primary file:** `Falsification_InformationTheoretic_PhaseTransition.py`
**Stub file:** `Falsification-Protocol-4.py` (information-theoretic wrapper)
**Tier:** Secondary | **Completion:** 73%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `ThermodynamicConfig`  86  Configuration for thermodynamic analysis  |
 | `SurpriseIgnitionSystem`  249  Surprise accumulation + ignition detection  |
 | `InformationTheoreticAnalysis`  431  Transfer entropy, MI, Φ  |
 | `ClinicalBiomarkerFalsification`  1624  Clinical DoC biomarker falsification  |

**Key constants:**

 | Constant  Value  Purpose
 | ----------  -------  ---------  |
 | `DEFAULT_SURPRISE_THRESHOLD`  0.5  Ignition threshold
 | `DEFAULT_ALPHA`  8.0  Sigmoid steepness
 | `DEFAULT_TAU_S`  0.3  Surprise time constant
 | `DEFAULT_TAU_THETA`  10.0  Threshold adaptation time constant
 | `DEFAULT_ETA_THETA`  0.01  Threshold learning rate
 | `DEFAULT_BETA`  1.2  Interoceptive weighting
 | `DEFAULT_HURST_LAG_MULTIPLIER`  4  DFA lag multiplier

**Key functions:**

 | Function  Line  Purpose  |
 | --------------------  -------  ----------------  |
 | `run_falsification()`  1976  Entry point  |

**Implementation details:**

- Hurst exponent via DFA at lines 479–525

- Level 1 metabolic cost: Attwell & Laughlin ATP budget (10⁹ ATP/spike), prefrontal ~0.5W at lines 1242+

---

### FP-5 · Evolutionary Plausibility

**Primary file:** `Falsification_EvolutionaryPlausibility_Standard6.py`
**Stub file:** `Falsification-Protocol-5.py` (evolutionary wrapper)
**Tier:** Tertiary | **Completion:** 70%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `EvolvableAgent`  80  Genome-based APGI agent  |
 | `ContinuousUpdateAgent`  269  Continuous parameter update variant  |
 | `EvolutionaryAPGIEmergence`  446  Evolution simulation engine  |

**Key functions:**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `run_falsification()`  848  Entry point  |
 | `get_falsification_criteria()`  867  F5.x / F6.x criteria  |
 | `check_falsification()`  923  Gate function  |

**Threshold constants (in `falsification_thresholds.py`):**

- F5_1_MIN_PROPORTION = 0.75
- F5_1_MIN_ALPHA = 4.0
- F5_4_MIN_PEAK_SEPARATION = 3.0
- F5_5_PCA_MIN_VARIANCE = 0.70
- F5_5_MIN_LOADING = 0.60
- F6_5_HYSTERESIS MIN = 0.08, MAX = 0.25

---

### FP-6 · Neural Network Energy Benchmark

**Primary file:** `Falsification_NeuralNetwork_EnergyBenchmark.py`
**Stub file:** `Falsification-Protocol-6.py` (APGI-inspired network wrapper)
**Tier:** Tertiary | **Completion:** 68%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `APGIInspiredNetwork`  437  APGI-structured network  |
 | `ComparisonNetworks`  685  MLP / LSTM / Attention baselines  |
 | `NetworkComparisonExperiment`  849  Orchestrates energy + performance comparison  |

**Key functions:**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `calculate_atp_cost()`  175  ATP per spike (Attwell & Laughlin)
 | `compare_atp_cost_with_literature()`  200  vs. biological ceiling
 | `calculate_bic_aic_comparison()`  287  BIC/AIC between APGI and baselines
 | `get_model_parameter_counts()`  400  k (param count) for BIC
 | `calculate_energy_per_correct_detection()`  417  Core energy metric
 | `run_falsification()`  1351  Entry point
 | `get_falsification_criteria()`  1377  Returns F-criteria
 | `check_falsification()`  1432  Gate function

---

### FP-7 · Mathematical Consistency / Equations

**Primary file:** `Falsification_MathematicalConsistency_Equations.py`
**Tier:** Secondary | **Completion:** 63%

**Key classes:**

 | Class  Line  Purpose  |
 | | ------  ---------  |
 | `EquationType`  69  Enum: ODE, SIGMOID, PRECISION, THRESHOLD  |
 | `ParameterBounds`  82  Physiological parameter ranges  |
 | `EquationTest`  93  Test result container  |
 | `MathematicalConsistencyChecker`  104  Orchestrates all 4 equation tests  |

**Key functions:**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `check_parameter_bounds()`  261  Validates {θ₀, Πⁱ, β, α, τ_S, τ_θ} in range
 | `verify_dimensional_homogeneity()`  281  Units consistent in ODE
 | `verify_surprise_derivatives()`  351  Sign/magnitude of ∂S/∂t
 | `verify_analytical_jacobian()`  434  Jacobian stability analysis
 | `verify_asymptotic_behavior()`  592  S → steady state as t→∞
 | `verify_threshold_stability()`  658  θ converges to fixed point
 | `verify_effective_precision()`  818  Πⁱ_eff = Πⁱ·exp(β·M) monotonicity
 | `verify_jacobian_stability()`  927  Eigenvalue stability check
 | `verify_paper_predictions()`  1009  All 4 equations vs. paper
 | `verify_four_core_equations_comprehensive()`  1312  Master verification (1,000+ draws)
 | `test_surprise_ode_comprehensive()`  1411  Eq.1: dS/dt = −S/τ_S + Πᵉ  εᵉ  + βΠⁱ  εⁱ |
 | `test_ignition_sigmoid_comprehensive()`  1524  Eq.2: P_ign = σ(α(S−θ))
 | `test_threshold_update_comprehensive()`  1639  Eq.4: dθ/dt = (θ₀−θ)/τ_θ + η_θ(C−V)
 | `test_free_energy_comprehensive()`  1752  Eq.3: Πⁱ_eff = Πⁱ·exp(β·M)

**Tolerance:** 1e-6 at lines 101, 1543, 1752, 1960, 2213

---

### FP-8 · Parameter Sensitivity & Identifiability

**Primary file:** `Falsification_ParameterSensitivity_Identifiability.py`
**Tier:** Tertiary | **Completion:** 60%

**Key class:**

 | Class  Line  Purpose  |
 | ---  ---  ---  |
 | `ParameterSensitivityAnalyzer`  2022  Full sensitivity analysis orchestrator  |

**Key functions:**

 | Function  Line  Purpose
 | ---  ---  ---  |
 | `simulate_model_performance_with_agent()`  47  Model performance for SA |
 | `analyze_oat_sensitivity()`  168  One-at-a-time sensitivity |
 | `analyze_beta_pi_collinearity()`  224  β / Πⁱ collinearity |
 | `analyze_parameter_recovery()`  333  Recovery accuracy |
 | `analyze_profile_likelihood()`  451  Profile likelihood (lines 393–560) |
 | `analyze_fisher_information_matrix()`  634  FIM (lines 578–670) |
 | `analyze_sobol_sensitivity()`  732  Sobol indices (SALib) |
 | `analyze_parameter_interactions()`  1338  Pairwise interactions |
 | `analyze_parameter_robustness()`  1442  Robustness under perturbation |
 | `run_comprehensive_parameter_sensitivity_analysis()`  1178  Master analysis |

**Dependencies:** `SALib.analyze.sobol`, `SALib.sample.saltelli`
**n_samples:** 1024 (power-of-2 compliant, lines 676, 1184, 1191)

---

### FP-9 · Neural Signatures EEG / P3b / HEP

**Primary file:** `Falsification_NeuralSignatures_EEG_P3b_HEP.py`
**Tier:** Secondary | **Completion:** 62%

**Key classes:**

 | Class  Line  Purpose  |
 | ---  ---  ---  |
 | `PaperPrediction`  37  Enum: P1.1, P1.2, P1.3, P2.a–P2.c, P3.x, P4.a–P4.d  |
 | `NeuralSignatureResult`  58  Result container (passed, evidence, stats)  |
 | `EEGData`  74  EEG epoch data container  |
 | `FalsificationThresholds`  101  All FP-9 threshold constants  |
 | `NeuralSignatureValidator`  2144  Master validator  |

**Key functions:**

 | Function  Line  Purpose
 | ---  ---  ---  |
 | `detect_gamma_oscillation()`  154  40Hz gamma power during ignition |
 | `detect_theta_gamma_pac()`  271  Phase-amplitude coupling (MI ≥ 0.012) |
 | `detect_hep_amplitude()`  497  Heartbeat-evoked potential amplitude |
 | `detect_p3b_amplitude()`  653  P3b ERP component (300–600ms) |
 | `tms_double_dissociation_test()`  797  P2.b: dlPFC vs. insula TMS dissociation |
 | `frequency_specific_power_analysis()`  1017  Band-specific power |
 | `mne_compatible_analysis()`  1164  MNE-Python EEG pipeline |
 | `pci_hep_joint_auc_classification()`  1502  P4.a: AUC > 0.80 |
 | `dmn_connectivity_specificity()`  1694  P4.b: DMN↔PCI r > 0.50; DMN↔HEP r < 0.20 |
 | `cold_pressor_pci_response()`  1817  P4.c: PCI increase >10% in MCS |
 | `baseline_recovery_prediction()`  1984  P4.d: 6-month recovery ΔR² > 0.10 |
 | `comprehensive_validation_framework()`  2437  Orchestrates P1–P4 predictions |
 | `run_neural_signature_validation()`  2698  Entry point |

**Criterion codes implemented:** P1.1, P1.2, P1.3, P2.a, P2.b, P2.c, P3.1, P3.2, P4.1, P4.2, P4.a, P4.b, P4.c, P4.d

---

### FP-10 · Bayesian Estimation MCMC / Cross-Species Scaling

**Primary files:**

- `Falsification_BayesianEstimation_MCMC.py` — NUTS/PyMC estimation

- `Falsification_CrossSpeciesScaling_P12.py` — allometric scaling

**Tier:** Tertiary | **Completion:** 55%

**Key functions (MCMC file):**

 | Function  Line  Purpose
 | ---  ---  ---  |
 | `apgi_psychometric_function()`  43  σ(β(S−θ)) psychometric |
 | `define_apgi_priors()`  65  Priors over {θ₀, Πe, Πi, β, α} |
 | `run_mcmc_bayesian_estimation()`  118  NUTS: 5,000 samples, 4 chains, 1,000 burn-in |
 | `compute_bayes_factors()`  284  BF₁₀ via ArviZ LOO-CV / WAIC |
 | `interpret_bayes_factor()`  353  Jeffreys scale interpretation |
 | `run_alternative_models()`  385  Standard PP / GWT alternatives |
 | `run_complete_mcmc_analysis()`  492  Master analysis |
 | `generate_synthetic_data()`  579  Synthetic data for validation |

**Key classes (Scaling file):**

 | Class  Line  Purpose  |
 | ---  ---  ---  |
 | `ScalingLawType`  44  Enum: LINEAR, POWER, EXPONENTIAL |
 | `SpeciesData`  54  Brain mass, neuron count, τ_S, θ₀ per species |
 | `APGIParameters`  89  APGI parameter set for scaling |
 | `CrossSpeciesScalingAnalyzer`  127  Allometric exponent analysis |

**Key functions (Scaling file):**

 | Function  Line  Purpose
 | ---  ---  ---  |
 | `apply_cross_species_scaling()`  788  Scale APGI params across species |
 | `calculate_allometric_exponent()`  802  Fit power law exponent |
 | `validate_allometric_relationship()`  823  Check exponent within ±2 SD |
 | `validate_scaling_laws()`  844  Master scaling validation |
 | `run_cross_species_scaling()`  909  Entry point |

**P12 falsification criterion:** Allometric exponents deviate >2 SD from expected neurobiological scaling → FALSIFIED

---

### FP-11 · Bayesian Estimation Parameter Recovery

**Primary file:** `Falsification_BayesianEstimation_ParameterRecovery.py`
**Tier:** Secondary | **Completion:** 67%

**Key class:**

 | Class  Line  Purpose  |
 | ---  ---  ---  |
 | `BayesianParameterRecovery`  1201  Full Bayesian recovery orchestrator  |

**Key functions:**

 | Function  Line  Purpose
 | ---  ---  ---  |
 | `run_bayesian_estimation_nuts()`  38  PyMC NUTS (primary)|
 | `run_bayesian_estimation_mh()`  138  MH sampler (fallback)|
 | `metropolis_hastings_sampling()`  173  MH implementation|
 | `compute_posterior_distributions()`  226  Posterior summaries|
 | `calculate_bayesian_factor()`  247  BF computation|
 | `test_parameter_identifiability()`  271  Structural identifiability check|
 | `map_bayesian_factor_to_predictions()`  379  BF → named prediction|
 | `run_bayesian_estimation_hierarchical()`  485  Hierarchical Bayes|
 | `check_posterior_calibration()`  657  Posterior calibration|
 | `get_falsification_criteria()`  906  Criteria dict|
 | `check_falsification()`  968  Gate function|

---

### FP-12 · Framework-Level Aggregator / Liquid Network Dynamics

**Primary files:**

- `APGI_Falsification_Aggregator.py` — framework-level conditions A and B
- `Falsification_LiquidNetworkDynamics_EchoState.py` — echo-state network

**Tier:** Primary | **Completion:** 76%

**Key functions (Aggregator):**

 | Function  Line  Purpose
 | ---  ---  ---  |
 | `aggregate_prediction_results()`  46  Loads JSON from all protocols, tallies pass/fail  |
 | `check_framework_falsification_condition_a()`  63  All 14 predictions fail → FALSIFIED  |
 | `check_framework_falsification_condition_b()`  75  Alt framework (GWT/IIT) ≥90% predictions  |
 | `generate_gnwt_predictions()`  92  GNWT prediction set  |
 | `generate_iit_predictions()`  117  IIT prediction set  |
 | `run_framework_falsification()`  134  Master falsification run  |

**Constants (Aggregator):**

- `FRAMEWORK_FALSIFICATION_THRESHOLD_A` = 14 (all named predictions must fail)
- `ALTERNATIVE_FRAMEWORK_PARSIMONY_THRESHOLD` = 0.90

**Key classes (Echo-State file):**

 | Class  Line  Purpose  |
 | ---  ---  ---  |
 | `NetworkType`  49  Enum: LIQUID, APGI, STANDARD  |
 | `LiquidTimeConstantNeuron`  58  LTC neuron (τ-gated dynamics)  |
 | `PhaseTransitionMetrics`  89  Metrics container  |
 | `SeparationResult`  101  Echo-state separation capacity  |
 | `LiquidNetworkDynamicsAnalyzer`  1730  Orchestrates all F6.x tests  |

**Key functions (Echo-State file):**

 | Function  Line  Purpose  |
 | ---  ---  ---  |
 | `test_liquid_network_properties()`  112  Full F6 suite  |
 | `test_echo_state_property()`  193  Echo state property check  |
 | `test_f6_3_sparsity()`  281  F6.3: ≥30% sparsity reduction  |
 | `test_f6_4_fading_memory_detailed()`  378  F6.4: fading memory property  |
 | `test_f6_5_bifurcation_sweep()`  472  F6.5: input gain sweep, hysteresis 0.08–0.25  |
 | `test_liquid_time_constant_dynamics()`  1146  LTC dynamics test  |
 | `test_phase_transition()`  1220  Phase transition detection  |
 | `run_liquid_network_validation()`  909  Entry point  |

**Spectral radius guard:** Lines 215–225 — scales ρ to 0.98 if ≥1.0 to maintain echo-state property

---

## 5. Named Predictions → Protocol Routing

From `APGI_Falsification_Aggregator.py` (`PREDICTION_TO_PROTOCOL` dict) and `NAMED_PREDICTIONS`:

 | Prediction ID  Description  Routed To  |
 | ---  ---  ---  |
 | P1.1  Interoceptive precision modulates detection threshold (d=0.40–0.60)  `Validation_Protocol_8` → VP-8
 | P1.2  Arousal amplifies the Πⁱ–threshold relationship  `Validation_Protocol_8` → VP-8
 | P1.3  High-IA individuals show stronger arousal benefit  `Validation_Protocol_8` → VP-8
 | P2.a  dlPFC TMS shifts threshold >0.1 log units  `Validation_Protocol_10` → VP-10
 | P2.b  Insula TMS reduces HEP ~30% AND PCI ~20% (double dissociation)  `Validation_Protocol_10` → VP-10
 | P2.c  High-IA × insula TMS interaction  `Validation_Protocol_10` → VP-10
 | P3.conv  APGI converges in 50–80 trials (beats baselines)  `Validation_Protocol_3` → VP-3
 | P3.bic  APGI BIC lower than StandardPP and GWTonly  `Validation_Protocol_3` → VP-3
 | P4.a  PCI+HEP joint AUC > 0.80 for DoC classification  `Falsification-Protocol-9` → FP-9
 | P4.b  DMN↔PCI r > 0.50; DMN↔HEP r < 0.20  `Falsification-Protocol-9` → FP-9
 | P4.c  Cold pressor increases PCI >10% in MCS, not VS  `Falsification-Protocol-9` → FP-9
 | P4.d  Baseline PCI+HEP predicts 6-month recovery ΔR² > 0.10  `Falsification-Protocol-9` → FP-9
 | P5.a  vmPFC–SCR anticipatory correlation r > 0.40  `Validation_Protocol_10` → VP-10
 | P5.b  vmPFC uncorrelated with posterior insula (r < 0.20)  `Validation_Protocol_10` → VP-10

---

## 6. Criterion Code Cross-Reference

The table below shows which Python files implement each criterion family and what test is applied.

### F1.x — Hierarchical Properties

 | Code  Threshold  Test  Files  |
 | ---  ---  ---  ---  |
 | F1.1  APGI advantage <10% OR d<0.35 → FALSIFIED  Independent t-test  FP-1 (full), VP-1, VP-6, VP-8, VP-9, VP-11, VP-12
 | F1.2  <3 clusters OR silhouette<0.30 → FALSIFIED  K-means + silhouette + ANOVA  Same
 | F1.3  Difference <15% OR η²<0.08 → FALSIFIED  Repeated-measures ANOVA  Same
 | F1.4  Adaptation <12% OR τ<5s or >150s → FALSIFIED  Exponential decay fit  Same
 | F1.5  MI<0.008 OR increase<15% → FALSIFIED  Permutation test  Same
 | F1.6  Active α>1.4 OR low α<1.3 → FALSIFIED  Paired t-test + spectral fit  Same

### F2.x — Somatic Marker / Bayesian

 | Code  Threshold  Test  Files  |
 | ---  ---  ---  ---  |
 | F2.1  LOO worse than competitors by >10  WAIC/LOO-CV  VP-1 (BayesianComparison)
 | F2.2  Interoceptive precision CI includes zero  HDI credible interval  VP-2, VP-11
 | F2.3  RT advantage <35ms  Paired t-test  VP-1
 | F2.4  Confidence effect <30%  Paired t-test  VP-1
 | F2.5  BF APGI vs GWT < 3 (UPDATED: WAIC/LOO primary)  ArviZ LOO-CV  VP-1, VP-11

### F3.x — Performance Advantage

 | Code  Threshold  Test  Files  |
 | ---  ---  ---  ---  |
 | F3.1  Advantage <18% OR d<0.60 → FALSIFIED  Independent t-test  FP-1, VP-4, VP-7, VP-12
 | F3.2  Advantage <28% OR η²<0.20 → FALSIFIED  Two-way mixed ANOVA  Same
 | F3.3  Reduction <25% OR d<0.75 → FALSIFIED  Paired t-test  Same
 | F3.4  Reduction <20% OR d<0.65 → FALSIFIED  Paired t-test  Same
 | F3.5  Retention <85%, gain <30% → FALSIFIED  TOST + efficiency t-test  Same
 | F3.6  Time >200 trials, HR<1.45 → FALSIFIED  Log-rank test  Same

### F4.x — Phase Transition

 | Code  Threshold  Test  Files  |
 | ---  ---  ---  ---  |
 | F4.1  Bandwidth >40 bits/s → FALSIFIED  Curve fitting + bootstrap  VP-4 (Level2 file)
 | F4.2–F4.4  Phase transition metrics  Transfer entropy, MI, Φ  VP-4 (Level2 file)
 | F4.5  Hurst H ≤ 0.5 → FALSIFIED  DFA  FP-4, VP-4

### F5.x — Evolutionary Emergence

 | Code  Threshold  Test  Files  |
 | ---  ---  ---  ---  |
 | F5.1  <75% develop multi-timescale OR α<4.0 OR sep<3× → FALSIFIED  Binomial test  FP-1, FP-2, FP-5, VP-3
 | F5.2  Mean r<0.5 → FALSIFIED  Correlation  Same
 | F5.3  Gain ratio <0.8 → FALSIFIED  t-test  Same
 | F5.4  <75% multi-timescale OR sep<3× → FALSIFIED  Binomial test  Same
 | F5.5  Variance <70% OR loading <0.60 → FALSIFIED  PCA + scree  FP-1, FP-5
 | F5.6  Difference <40% OR d<0.85 → FALSIFIED  t-test  Same

### F6.x — Network / Liquid Dynamics

 | Code  Threshold  Test  Files  |
 | ---  ---  ---  ---  |
 | F6.1  LTCN transition >50ms → FALSIFIED  Mann-Whitney U  FP-6, VP-6, VP-9
 | F6.2  Window <200ms OR ratio <4× OR R²<0.85 → FALSIFIED  Wilcoxon signed-rank  FP-6, VP-6, VP-9, VP-12
 | F6.3  Sparsity reduction <30% → FALSIFIED  Connectivity comparison  FP-11 (LiquidNetwork)
 | F6.5  Hysteresis outside [0.08, 0.25] → FALSIFIED  Phase portrait sweep  FP-1, FP-5, FP-11

### V11.x — Cultural Neuroscience / Individual Differences

 | Code  Threshold  Test  File
 | ------  -----------  ------  ------  |
 | V11.1  Cultural group θ₀ difference  BF comparison  VP-11
 | V11.2  Πⁱ credible interval excludes 0  HDI  VP-11
 | V11.3  β and α cross-cultural universality  HDI comparison  VP-11
 | V11.4  R̂ ≤ 1.01 for all parameters  Gelman-Rubin  VP-11
 | V11.5  Posterior predictive check p > 0.05  PPC  VP-11

### V12.x — Clinical / Cross-Species

 | Code  Threshold  Test  File
 | ------  -----------  ------  ------  |
 | V12.1  P3b reduction ≥80%, ignition ≥70%  Paired t-test + permutation  VP-1, VP-12
 | V12.2  Cross-species r ≥ 0.60  Pearson correlation  VP-12

---

## 7. Architecture & Shared Utilities

### 7.1 Shared utility modules (referenced by multiple protocol files)

 | Module  Used by  Purpose  |
 | ---  ---  ---  |
 | `utils.constants.LEVEL_TIMESCALES`  FP-1, FP-3  τ₁/τ₂/τ₃ time constants
 | `utils.constants.DIM_CONSTANTS`  FP-1, FP-5, VP-10  Dimension constants (EXTERO_DIM etc.)
 | `utils.config_manager.ConfigManager`  FP-5  Configuration management
 | `utils.falsification_thresholds`  FP-6 (imports 8 constants)  Centralised threshold values
 | `utils.shared_falsification.check_F5_family`  FP-2  Shared F5.1–F5.6 implementation
 | `utils.statistical_tests`  FP-4  Power analysis functions
 | `utils.logging_config.apgi_logger`  Master_Validation.py  Structured logging

### 7.2 GUI runner (not a protocol)

**File:** `APGI_Falsification_Protocols_GUI.py`
**Class:** `ProtocolRunnerGUI` (tkinter)
**Role:** Provides a GUI for selecting and running any of the 12 falsification protocols. Maps human-readable protocol names to implementation files. Not part of the falsification logic itself.

**Protocol-to-file mapping in GUI (ground truth for protocol numbering):**

 | GUI Name  File  |
 | ---  ---  |
 | Protocol 1: APGI Agent  `Falsification_ActiveInferenceAgents_F1F2.py`
 | Protocol 2: Iowa Gambling  `Falsification_AgentComparison_ConvergenceBenchmark.py`
 | Protocol 3: Agent Comparison  `Falsification_FrameworkLevel_MultiProtocol.py`
 | Protocol 4: Phase Transition  `Falsification_InformationTheoretic_PhaseTransition.py`
 | Protocol 5: Evolutionary  `Falsification_EvolutionaryPlausibility_Standard6.py`
 | Protocol 6: Network Comparison  `Falsification_NeuralNetwork_EnergyBenchmark.py`
 | Protocol 7: Mathematical Consistency  `Falsification_MathematicalConsistency_Equations.py`
 | Protocol 8: Parameter Sensitivity  `Falsification_ParameterSensitivity_Identifiability.py`
 | Protocol 9: Neural Signatures  `Falsification_NeuralSignatures_EEG_P3b_HEP.py`
 | Protocol 10: Cross-Species Scaling  `Falsification_CrossSpeciesScaling_P12.py`
 | Protocol 11: Bayesian Estimation  `Falsification_BayesianEstimation_ParameterRecovery.py`
 | Protocol 12: Liquid Network Dynamics  `Falsification_LiquidNetworkDynamics_EchoState.py`

### 7.3 Protocol File Mapping (Direct GUI Implementation)

All 12 falsification protocols work identically: the GUI maps directly to full implementation files without intermediate wrappers. The mapping is consistent across Protocols 1–12.

 | Protocol  GUI Name  Full Implementation File  Domain  |
 | ---  ---  ---  ---  |
 | FP-1  Protocol 1: APGI Agent  `Falsification_ActiveInferenceAgents_F1F2.py`  Active inference agents
 | FP-2  Protocol 2: Iowa Gambling  `Falsification_AgentComparison_ConvergenceBenchmark.py`  Iowa Gambling Task / convergence
 | FP-3  Protocol 3: Agent Comparison  `Falsification_FrameworkLevel_MultiProtocol.py`  Neural network analysis
 | FP-4  Protocol 4: Phase Transition  `Falsification_InformationTheoretic_PhaseTransition.py`  Information-theoretic
 | FP-5  Protocol 5: Evolutionary  `Falsification_EvolutionaryPlausibility_Standard6.py`  Evolutionary plausibility
 | FP-6  Protocol 6: Network Comparison  `Falsification_NeuralNetwork_EnergyBenchmark.py`  APGI-inspired network
 | FP-7  Protocol 7: Mathematical Consistency  `Falsification_MathematicalConsistency_Equations.py`  Equation verification
 | FP-8  Protocol 8: Parameter Sensitivity  `Falsification_ParameterSensitivity_Identifiability.py`  Sensitivity analysis
 | FP-9  Protocol 9: Neural Signatures  `Falsification_NeuralSignatures_EEG_P3b_HEP.py`  EEG / P3b / HEP signatures
 | FP-10  Protocol 10: Cross-Species Scaling  `Falsification_CrossSpeciesScaling_P12.py`  Allometric scaling
 | FP-11  Protocol 11: Bayesian Estimation  `Falsification_BayesianEstimation_ParameterRecovery.py`  Parameter recovery
 | FP-12  Protocol 12: Liquid Network Dynamics  `Falsification_LiquidNetworkDynamics_EchoState.py`  Echo-state networks

**Note:** Six placeholder files (`Falsification-Protocol-1.py` through `-6.py`) exist but are not used by the GUI. They contain only stub implementations with random data generation and can be safely deleted.

---

## 8. Implementation Gaps & Notes

### 8.1 Missing implementation

 | Gap  Description  Priority  |
 | ---  ---  ---  |
 | **Paper Protocol 5 (fMRI Anticipation/Experience)**  No Python file exists. `EvolutionaryEmergence_AnalyticalValidation.py` currently covers VP-5's slot but is an evolutionary simulation, not the fMRI anticipation paradigm. Must create.  HIGH
 | **Master_Validation.py protocol name mismatch**  Registers `Validation_Protocol_1.py` but actual file is `SyntheticEEG_MLClassification.py`. File explicitly notes it is NOT Protocol 1. Symlink or rename required for orchestrator to load correctly.  MEDIUM
 | **FP-10 split across two files**  `Falsification_BayesianEstimation_MCMC.py` and `Falsification_CrossSpeciesScaling_P12.py` both claim FP-10. GUI maps Protocol 10 only to `Falsification_CrossSpeciesScaling_P12.py`.  MEDIUM
 | **VP-11 split across two files**  `Validation_Protocol_11.py` and `QuantitativeModelFits_SpikingLNN_Priority3.py` both implement VP-11. Master_Validation.py registers only `Validation_Protocol_11.py`.  LOW
 | **VP-4 split across two files**  `InformationTheoretic_PhaseTransition_Level2.py` (Level 2, F-criteria) and `Validation_Protocol_P4_Epistemic.py` (P5–P12 wrapper) both serve VP-4. Master registers only the epistemic file.  LOW

### 8.2 Filename conflicts

 | Conflict  Validation folder  Falsification folder  Resolution  |
 | ---  ---  ---  ---  |
 | `Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py`  VP-10 entry point  FP-2 causal engine  Both exist. Validation copy has `APGIValidationProtocol10` class; Falsification copy adds `ColdPressorTest` and `MNEDataInterface`.

### 8.3 Recent critical fixes (confirmed in source files)

 | Fix  File  Impact  |
 | ---  ---  ---  |
 | P6 bandwidth ceiling corrected to 40 bits/s (was 100 bits/s)  `InformationTheoretic_PhaseTransition_Level2.py`  V4.1 / P6 criterion now biologically correct  |
 | F5_4_MIN_PEAK_SEPARATION = 3.0  `utils/falsification_thresholds.py`  All F5.4 checks now use correct paper value  |
 | F5_1_MIN_PROPORTION = 0.75, F5_1_MIN_ALPHA = 4.0  `utils/falsification_thresholds.py`  F5.1 threshold corrected  |
 | F5_5_PCA_MIN_VARIANCE = 0.70, MIN_LOADING = 0.60  `utils/falsification_thresholds.py`  F5.5 PCA threshold corrected  |
 | V12_1_MIN_P3B_REDUCTION_PCT = 80.0  `utils/falsification_thresholds.py`  V12.1 P3b threshold corrected  |
 | V12_1_MIN_IGNITION_REDUCTION_PCT = 70.0  `utils/falsification_thresholds.py`  V12.1 ignition threshold corrected  |
 | F6_1_LTCN_MAX_TRANSITION_MS = 50.0  `utils/falsification_thresholds.py`  F6.1 standardised to 50ms  |
 | F6_2_LTCN_MIN_WINDOW_MS = 200.0, ratio = 4.0, R² = 0.85  `utils/falsification_thresholds.py`  F6.2 standardised to paper values  |
 | F6_5_HYSTERESIS MIN = 0.08, MAX = 0.25  `utils/falsification_thresholds.py`  F6.5 bounds corrected from generic range  |
 | Spectral radius guard: ρ → 0.98 if ≥1.0  `Falsification_LiquidNetworkDynamics_EchoState.py` lines 215–225  Echo-state property enforced  |
 | Cross-modal falsification: abs(intero−extero) < 1e-6  `Falsification_EvolutionaryPlausibility_Standard6.py`  Float equality bug fixed  |
 | MCMC: R̂ ≤ 1.01 gate enforced  `Validation_Protocol_11.py` lines 435–447, 849  Convergence criterion enforced  |
 | n_samples = 1024 (power-of-2)  `Falsification_ParameterSensitivity_Identifiability.py` lines 676, 1184, 1191  Sobol sampling corrected  |
 | Bonferroni correction: α_per_test = 0.008 (6 tests)  `Psychophysical_ThresholdEstimation_Protocol1.py`  Multiple comparison correction added  |

---

*Document generated from direct inspection of 38 Python source files. All class names, function names, line numbers, and criterion codes are sourced from the uploaded ZIP archives.*
