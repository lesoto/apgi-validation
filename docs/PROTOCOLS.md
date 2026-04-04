# APGI Paper Protocol

## File Inventory — All 35 Source Files (Falsification/ + Validation/)

Verified against direct ls of Falsification/ (16 files) and Validation/ (19 files) folders, March 2026. GUI runners, `__init__.py`, and orchestrators listed separately.

| # | Filename | Folder | Protocol ID | Role | Completion |
| --- | ---------- | -------- | ------------- | ------ | ------------ |
| **Validation Files — Protocol Scripts (VP_01–VP_15)** | | | | | |
| 1 | VP_01_SyntheticEEG_MLClassification.py | Validation | VP-1 | Synthetic EEG / ML Classification support | 62% |
| 2 | VP_02_Behavioral_BayesianComparison.py | Validation | VP-2 | Behavioral Bayesian Comparison (P1.1–P1.3) | 71% |
| 3 | VP_03_ActiveInference_AgentSimulations.py | Validation | VP-3 | Active Inference Agent Comparison | 82% |
| 4 | VP_04_PhaseTransition_EpistemicLevel2.py | Validation | VP-4 | Phase Transition / Level 2 Epistemic | 74% |
| 5 | VP_05_EvolutionaryEmergence.py | Validation | VP-5 | Evolutionary Emergence (simulation only) | 66% |
| 6 | VP_06_LiquidNetwork_InductiveBias.py | Validation | VP-6 | Liquid Network Inductive Bias Benchmark | 68% |
| 7 | VP_07_TMS_CausalInterventions.py | Validation | VP-7 | TMS / Pharmacological Causal Interventions | 79% |
| 8 | VP_08_Psychophysical_ThresholdEstimation.py | Validation | VP-8 | Psychophysical Threshold Estimation | 77% |
| 9 | VP_09_NeuralSignatures_EmpiricalPriority1.py | Validation | VP-9 | Convergent Neural Signatures (Roadmap P1) | 73% |
| 10 | VP_10_CausalManipulations_Priority2.py | Validation | VP-10 | Causal Manipulations (Roadmap P2) | 70% |
| 11 | VP_11_MCMC_CulturalNeuroscience_Priority3.py | Validation | VP-11 | MCMC Cultural Neuroscience (Roadmap P3) | 75% |
| 12 | VP_12_Clinical_CrossSpecies_Convergence.py | Validation | VP-12 | Clinical / Cross-Species Convergence (P4) | 76% |
| 13 | VP_13_Epistemic_Architecture.py | Validation | VP-13 | Epistemic Architecture Predictions (P5–P12) | 74% |
| 14 | VP_14_fMRI_Anticipation_Experience.py | Validation | VP-14 | fMRI BOLD Simulation (Anticipation/Experience) | 70% |
| 15 | VP_15_fMRI_Anticipation_vmPFC.py | Validation | VP-15 | fMRI vmPFC Anticipation (awaiting real data) | 5% |
| **Validation Files — Orchestrators & Infrastructure** | | | | | |
| 16 | VP_ALL_Aggregator.py | Validation | — | Framework-level validation aggregator (V1.1–V14.x) | Active |
| 17 | Master_Validation.py | Validation | — | Orchestrator (APGIMasterValidator) | Active |
| 18 | APGI_Validation_GUI.py | Validation | — | GUI runner (tkinter, not protocol logic) | Active |
| 19 | **init**.py | Validation | — | Python package marker | — |
| **Falsification Files — Protocol Scripts (FP_01–FP_12)** | | | | | |
| 20 | FP_01_ActiveInference.py | Falsification | FP-1 | Active Inference & Hierarchical Dynamics | 74% |
| 21 | FP_02_AgentComparison_ConvergenceBenchmark.py | Falsification | FP-2 | Agent Comparison & Convergence Benchmark | 72% |
| 22 | FP_03_FrameworkLevel_MultiProtocol.py | Falsification | FP-3 | Framework-Level Multi-Protocol Synthesis | 65% |
| 23 | FP_04_PhaseTransition_EpistemicArchitecture.py | Falsification | FP-4 | Phase Transition & Epistemic Architecture | 73% |
| 24 | FP_05_EvolutionaryPlausibility.py | Falsification | FP-5 | Evolutionary Plausibility (Standard 6) | 70% |
| 25 | FP_06_LiquidNetwork_EnergyBenchmark.py | Falsification | FP-6 | Liquid Network Energy Benchmark (ATP) | 68% |
| 26 | FP_07_MathematicalConsistency.py | Falsification | FP-7 | Mathematical Consistency (Sympy, Eq. 1–4) | 63% |
| 27 | FP_08_ParameterSensitivity_Identifiability.py | Falsification | FP-8 | Parameter Sensitivity & Identifiability | 60% |
| 28 | FP_09_NeuralSignatures_EEG_P3b_HEP.py | Falsification | FP-9 | Neural Signatures — EEG / P3b / HEP | 62% |
| 29 | FP_10_BayesianEstimation_MCMC.py | Falsification | FP-10 | Bayesian Model Evidence (NUTS/PyMC MCMC) | 55% |
| 30 | FP_11_LiquidNetworkDynamics_EchoState.py | Falsification | FP-11 | Liquid Network Dynamics & Echo State (LTCN) | 67% |
| 31 | FP_12_CrossSpeciesScaling.py | Falsification | FP-12 | Cross-Species Allometric Scaling (P12) | 55% |
| **Falsification Files — Orchestrators & Infrastructure** | | | | | |
| 32 | FP_ALL_Aggregator.py | Falsification | FP-AGG | Terminal framework falsification (Conditions A & B) | Active |
| 33 | Master_Falsification.py | Falsification | — | Orchestrator for all 12 FP runs [NEW] | Active |
| 34 | APGI_Falsification_Protocols_GUI.py | Falsification | — | GUI runner (tkinter, not protocol logic) | Active |
| 35 | **init**.py | Falsification | — | Python package marker | — |

| GUI Protocol | Maps To |
| ------------ | ------- |
| Protocol 1: APGI Agent | FP_1_FP_1_Falsification_ActiveInferenceAgents_F1F2.py |
| Protocol 2: Iowa Gambling | FP_2_FP_2_Falsification_AgentComparison_ConvergenceBenchmark.py |
| Protocol 3: Agent Comparison | FP_3_FP_3_Falsification_FrameworkLevel_MultiProtocol.py |
| Protocol 4: Phase Transition | FP_4_FP_4_Falsification_InformationTheoretic_PhaseTransition.py |
| Protocol 5: Evolutionary | FP_5_FP_5_Falsification_EvolutionaryPlausibility_Standard6.py |
| Protocol 6: Network Comparison | FP_6_FP_6_Falsification_NeuralNetwork_EnergyBenchmark.py |

## 1. File Inventory

### 1.1 Validation Files (16 total)

| # | Filename | VP ID | Paper Protocol |
| --- | --------- | ------- | --------------- |
| 1 | `VP_3_ActiveInference_AgentSimulations_Protocol3.py` | VP-3 | Paper Protocol 3 |
| 2 | `BayesianModelComparison_ParameterRecovery.py` | VP-1 (support) | Paper Protocol 6 (partial) |
| 3 | `VP_10_Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py` | VP-10 | Roadmap Priority 2 |
| 4 | `VP_12_Clinical_CrossSpecies_Convergence_Protocol4.py` | VP-12 | Paper Protocol 4 / Roadmap Priority 4 |
| 5 | `VP_9_ConvergentNeuralSignatures_Priority1_EmpiricalRoadmap.py` | VP-9 | Roadmap Priority 1 |
| 6 | `VP_5_EvolutionaryEmergence_AnalyticalValidation.py` | VP-5 | — (Paper Protocol 5 MISSING) |
| 7 | `VP_4_InformationTheoretic_PhaseTransition_Level2.py` | VP-4 | Epistemic Paper P5–P8 (Level 2) |
| 8 | `Master_Validation.py` | — | Orchestrator (not a protocol) |
| 9 | `VP_6_NeuralNetwork_InductiveBias_ComputationalBenchmark.py` | VP-6 | Computational Architecture |
| 10 | `VP_8_Psychophysical_ThresholdEstimation_Protocol1.py` | VP-8 | Paper Protocol 1 |
| 11 | `VP_11_QuantitativeModelFits_SpikingLNN_Priority3.py` | VP-11 | Roadmap Priority 3 |
| 12 | `VP_1_SyntheticEEG_MLClassification.py` | VP-1 (support) | Paper Protocol 6 (partial) |
| 13 | `VP_7_TMS_Pharmacological_CausalIntervention_Protocol2.py` | VP-7 | Paper Protocol 2 |
| 14 | `VP_2_Validation_Protocol_2.py` | VP-2 | Paper Protocol 6 (behavioral) |
| 15 | `VP_11_Validation_Protocol_11.py` | VP-11 (canonical) | Roadmap Priority 3 / Cultural Neuro |
| 16 | `Validation_Protocol_P4_Epistemic.py` | VP-4 (support) | Epistemic Paper P5–P12 |

### 1.2 Falsification Files (22 total)

| # | Filename | FP ID | Description |
| --- | --------- | ------- | ----------- |
| 1 | `FP_12_Falsification_Aggregator.py` | FP-AGG | Framework-level aggregator |
| 2 | `APGI_Falsification_Protocols_GUI.py` | — | GUI runner (not a protocol) |
| 3 | `VP_10_Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py` | FP-2 (support) | Causal manipulation engine |
| 10 | `FP_1_FP_1_Falsification_ActiveInferenceAgents_F1F2.py` | FP-1 (full) | Primary F1/F2 engine |
| 11 | `FP_2_FP_2_Falsification_AgentComparison_ConvergenceBenchmark.py` | FP-2 (full) | Convergence benchmark |
| 12 | `FP_10_FP_10_Falsification_BayesianEstimation_MCMC.py` | FP-10 | NUTS/PyMC MCMC estimation |
| 13 | `FP_10_FP_10_Falsification_BayesianEstimation_ParameterRecovery.py` | FP-11 | Parameter recovery |
| 14 | `FP_12_CrossSpeciesScaling.py` | FP-12 | Allometric scaling |
| 15 | `FP_5_FP_5_Falsification_EvolutionaryPlausibility_Standard6.py` | FP-5 (full) | Evolutionary analysis |
| 16 | `FP_3_FP_3_Falsification_FrameworkLevel_MultiProtocol.py` | FP-3 (full) | Multi-protocol synthesis |
| 17 | `FP_4_FP_4_Falsification_InformationTheoretic_PhaseTransition.py` | FP-4 (full) | Phase transition analysis |
| 18 | `FP_11_FP_11_Falsification_LiquidNetworkDynamics_EchoState.py` | FP-11 | Echo-state / liquid network |
| 19 | `FP_7_Falsification_MathematicalConsistency_Equations.py` | FP-7 | Sympy equation verification |
| 20 | `FP_6_FP_6_Falsification_NeuralNetwork_EnergyBenchmark.py` | FP-6 (full) | Energy / ATP benchmark |
| 21 | `FP_9_FP_9_Falsification_NeuralSignatures_EEG_P3b_HEP.py` | FP-9 | EEG / P3b / HEP neural sigs |
| 22 | `FP_8_FP_8_Falsification_ParameterSensitivity_Identifiability.py` | FP-8 | Sobol / FIM / profile likelihood |
| 23 | `FP_12_CrossSpeciesScaling.py` | FP-12 (scaling sub-protocol) | Allometric scaling |

---

### Full cross-reference of all FP (Falsification Protocol) and VP (Validation Protocol) assignments across the four-paper APGI series. Compiled from source code audit of all 14 protocol files (FP_01–FP_12, FP_10 Dispatcher, FP_ALL_Aggregator)

---

### Badge Key

- **Primary** = Protocol's primary home / introduces the criterion
- **Shared / Also** = Protocol is re-invoked or cross-referenced in this paper
- **Level 1–3** = Epistemic tier from Paper 4's three-tier architecture (thermodynamic / information-theoretic / computational)
- **Gap 1–4** = Measurement gap category (Paper 2 & 4 taxonomy)
- **Std 1–7** = Methodological Standard from Paper 4 / Paper 1 Appendix D
- **Terminal** = Protocol is a framework-level aggregator
- **Roadmap** = Cited in empirical roadmap / future prediction
- **—** = Not referenced in this paper

---

### 2.1 Falsification Protocols (FP) by Paper

| Protocol | Name | Key Criteria Summary | Paper 1 — Framework | Paper 2 — Liquid Networks | Paper 3 — Multi-Scale | Paper 4 — Epistemic Architecture |
| ---------- | ------ | --------------------- | --------------------- | ------------------------- | ---------------------- | ---------------------------------- |
| **FP-1** | Active Inference & Hierarchical Dynamics (F1/F2) | F1.1–F1.6: Performance advantage ≥18%, hierarchical timescale emergence, precision weighting, threshold adaptation τ_θ=10–100 s, PAC, 1/f slope. F2.1–F2.5: Somatic marker advantage ≥22%, interoceptive cost correlation r=−0.45–−0.65, vmPFC-like RT bias ≥35 ms, precision-weighted integration, learning trajectory discrimination HR≥1.65. | **Primary** — Appendix A.4 / P1–P6 preds | — | — | Std 3 & Std 4 (Quantitative Benchmarks & Falsification Conditions) |
| **FP-2** | Agent Comparison & Convergence Benchmark | P3.conv: APGI converges in 50–80 trials (beats baselines). P3.bic: APGI BIC lower than StandardPP and GWTOnly. F1.1–F1.6 + F2.1–F2.5 cross-run via shared check_falsification(). F5.1–F5.3 via VP-5 genome data integration. | **Primary** — P3 predictions | — | — | Std 5 (Alternative Comparison) |
| **FP-3** | Framework-Level Multi-Protocol Synthesis | F3.1–F3.6: Overall performance advantage ≥18%, interoceptive task specificity ≥28%, threshold gating necessity ≥25% reduction, precision weighting necessity ≥20%, computational efficiency ≥85% / ≤60% ops, sample efficiency ≤200 trials vs ≥300 standard RL. Aggregates FP-1 through FP-12 via FP_ALL_Aggregator. | **Primary** — F3.1–F3.6 & synthesis | — | — | Cross-framework synthesis |
| **FP-4** | Phase Transition & Epistemic Architecture | Level 2: Transfer entropy >0.1 bits, MI ≤40 bits/s, integrated information above baseline, τ_auto >20% increase (critical slowing). Level 1: Thermodynamic entropy via PyTorch ThermodynamicConfig. Clinical (P4.a–P4.d): AUC 0.75–0.85 for DoC classification, DMN to PCI r>0.50, cold pressor PCI, 6-month recovery prediction. | — | **Level 1** (thermodynamic stubs), **Level 2** (TE, MI, Φ, τ_auto) | — | **Level 1** Falsification scenario, **Level 2** Falsification scenario, Std 3 & Std 4 (Benchmarks & Falsification) |
| **FP-5** | Evolutionary Plausibility | F5.1–F5.6: ≥75% agents develop threshold gating by gen. 500, ≥65% precision weighting by gen. 400, ≥70% interoceptive prioritization β_intero ≥1.3×, ≥60% multi-timescale windows, PCA ≥70% variance, control agents ≥40% worse. P5.a–P5.b: vmPFC to SCR r>0.40, vmPFC to posterior insula r<0.20. | **Primary** — F5.1–F5.6; P5.a, P5.b | Gap 4 (Comparative Neuroscience) | — | Std 6 (Evolutionary Plausibility) |
| **FP-6** | Liquid Network Energy Benchmark | F6.1–F6.6: LTCN intrinsic threshold transition ≤50 ms (Cliff's δ ≥0.60), temporal integration 200–500 ms window ≥4× RNN, metabolic selectivity ≥30% sparsity reduction, memory decay τ_memory=1–3 s, saddle-node bifurcation ±0.15, alternatives need ≥2 add-ons. VP-6 topology tests (spectral radius, LNN substrate). | — | **Level 1** (metabolic / energy), Gap 2 (Alt. Architecture Modeling) | — | **Level 1** Falsification scenario, **Level 3** Computational Claims |
| **FP-7** | Mathematical Consistency | E1.1–E1.3: Surprise ODE dimensional consistency, derivative signs, asymptotic behavior. E2.1–E2.2: Ignition sigmoid monotonicity, threshold effect. E3.1–E3.2: Effective precision boundedness [0.01–15.0], exponential modulation. E4.1–E4.2: Threshold dynamics stability (eigenvalue <0), fixed-point analysis. E5.1–E5.2: Analytical–numerical Jacobian agreement ΔJ<1e-4, parameter bounds compliance. ε≤1e-6 throughout. | **Primary** — Appendix A / Eq. 1–4 | — | — | Std 2 & Std 3 (Bridge Principles & Benchmarks) |
| **FP-8** | Parameter Sensitivity & Identifiability | F8.SA: β + Πⁱ must account for >50% total Sobol ST sensitivity. F8.PL: Profile likelihood CI finite for all core params (HIGH / MODERATE identifiability). F8.FIM: FIM positive definite (all eigenvalues >0). APGI hierarchy: S_total(θ_t) > S_total(β) > S_total(Πⁱ) > S_total(Πᵉ). OAT, Sobol, collinearity (β/Πⁱ), parameter recovery (N=200 sims × 1000 trials), profile likelihood, FIM — 1500+ line comprehensive analysis. | **Primary** — Appendix A.4 / Param. Recovery | — | — | Std 3 (Quantitative Benchmarks) |
| **FP-9** | Neural Signatures — P3b & HEP | P1.1–P1.3: Gamma oscillation power, theta-gamma coupling, P3b amplitude >0.3 µV. P2.a–P2.c: HEP >0.2 µV, TMS double dissociation, cross-freq coupling specificity. P3.1–P3.2: Consciousness marker integration, neural complexity. P4.a–P4.d: PCI+HEP AUC >0.80, DMN to PCI r>0.50, cold pressor MCS vs VS, 6-month recovery ΔR²>0.10. Real EEG MNE pipeline (FP-9 Step 1.6). | **Primary** — P4 preds; Appendix A.3 HEP proxy | — | H1, H4 (Level-specific signatures & clinical biomarkers) | — |
| **FP-10** | Bayesian Model Evidence + Cross-Species Scaling (Dispatcher routes to FP10a MCMC + FP10b Scaling) | FP10a (MCMC): Gelman–Rubin R̂ ≤1.01; BF₁₀ ≥3 vs StandardPP/GWT; APGI MAE ≥20% lower. FP10b (Scaling): P12.a — allometric exponent 0.70–0.80; P12.b — >85% cross-species consistency; allometric exponents within ±2 SD of neurobiological expectation. Either sub-protocol failure falsifies FP-10. | — | Gap 4 (Comparative Neuroscience) | Cross-species predictions | P12 (Level 2 & Level 3 predictions) |
| **FP-11** | Liquid Network Dynamics & Echo State (LTCN) | F6.1–F6.6 re-implemented via echo state / liquid time-constant networks. VP-6 topology tests: spectral radius <1, Gaussian weight distribution, LNN substrate topology. Phase transition detection (bifurcation score). Connectivity density. Echo state property (F6.3 sparsity). LTCN threshold transition V6.1, temporal integration window V6.2. LNN substrate from Paper 2. | **Primary** — Appendix A.4 recovery | — | — | — |
| **FP-12** | Framework Aggregator (FP_ALL_Aggregator) | Condition A (FA): All 14 named predictions (P1.1–P1.3, P2.a–P2.c, P3.conv, P3.bic, P4.a–P4.d, P5.a–P5.b, fp10a_mcmc, fp10b_bf, fp10c_mae, fp10b_scaling) fail simultaneously. Condition B (FB): ΔBIC <10.0 — alternative framework (GWT/IIT) strictly more parsimonious. Either condition = framework falsified. Routes via PREDICTION_TO_PROTOCOL mapping table. | **Primary** — Synthesis | **Primary** — Framework Falsification | **Primary** — Framework Falsification | **Terminal** — Framework Falsification |

---

### 2.2 Validation Protocols (VP) by Paper

| Protocol | Name | Key Criteria Summary | Paper 1 — Framework | Paper 2 — Liquid Networks | Paper 3 — Multi-Scale | Paper 4 — Epistemic Architecture |
| ---------- | ------ | --------------------- | --------------------- | ------------------------- | ---------------------- | ---------------------------------- |
| **VP-1** | Synthetic ML Validation | Validates APGI parameter recovery in controlled synthetic environments. Confirms parameter identifiability via simulation before committing to empirical studies. Cited as P6 prediction in Paper 1 (simulation-based parameter validation section). | P6 (Simulation-Based Param. Validation) | Computational Claims validation | — | — |
| **VP-2** | Behavioral Validation | Tests APGI behavioral predictions: deck-task advantageous selection, interoceptive cost sensitivity, RT modulation. P6 behavioral predictions in Paper 1. Cross-validates F2.1–F2.5 somatic marker criteria from FP-2 in real agent/participant data. | P6 behavioral (Iowa Gambling Task predictions) | — | — | — |
| **VP-3** | Agent Comparison | Full multi-agent comparison: APGI vs StandardPP, GWTOnly, RandomAgent. Validates P3.conv and P3.bic named predictions. FP-8 uses VP-3 APGIAgent directly for parameter sensitivity simulation. Cited as P3 in Paper 1. | P3 (Agent Comparison section) | Liquid vs non-liquid architecture comparison | — | — |
| **VP-4** | Phase Transition Validation | Validates bistable dynamics, ignition transitions, and thermodynamic phase structure. Tied to FP-4 Level 2 falsification criteria. Paper 4 cites VP-4 coverage across P5–P12 predictions (Level 1–3 predictions table). | — | **Level 2** (bifurcation validation) | — | P5–P12 (Level 1–3 prediction coverage) |
| **VP-5** | Evolutionary Emergence Validation | Validates that APGI-like features (threshold gating, precision weighting, interoceptive prioritization) emerge naturally under metabolic constraint via evolutionary simulation. FP-1, FP-2, FP-3, FP-5, FP-6 all accept genome_data from VP-5. Std 6 (Evolutionary Plausibility) in Paper 4. | **Primary** — Evolutionary Origins section | — | — | Std 6 (Evolutionary Plausibility) |
| **VP-6** | Network Comparison | Validates LTCN vs standard RNN / LSTM / Transformer architectures. Tests V6.1 (LTCN threshold transition), V6.2 (temporal integration window), spectral radius, LNN substrate topology, connectivity density, echo state property. Cited in FP-11 directly as "VP-6 specific tests for LTCN." | — | **Primary** — Liquid vs. alt. architectures | — | **Level 3** (Computational Claims) |
| **VP-7** | TMS / Pharmacological Causal Manipulation | P2.a: dlPFC TMS shifts threshold >0.1 log units. P2.b: Insula TMS reduces HEP ~30% AND PCI ~20% (double dissociation). P2.c: High-IA × insula TMS interaction. Routes to VP_10_Falsification_CausalManipulations_TMS_Pharmacological_Priority2 in aggregator. Std 7 (Causal Roadmap) in Paper 4. | P2 (Causal prediction; Appendix A.3) | — | H4 (Clinical biomarker coupling dysregulation) | **Level 3** (Priority 2 causal manipulation) |
| **VP-8** | Psychophysical Threshold Detection | P1.1–P1.3 (primary prediction set): Interoceptive precision modulates detection threshold (d=0.40–0.60), arousal amplifies Πⁱ–threshold relationship, high-IA individuals show stronger arousal benefit. Primary empirical entry point for Paper 1's core detection threshold predictions. | **P1 (Primary)** — Core threshold detection preds | — | — | — |
| **VP-9** | Neural Signatures Empirical Validation | Empirical EEG / HEP / PCI validation pipeline (builds on FP-9). Validates level-specific neural signatures (H1) and clinical biomarker coupling (H4) from Paper 3. Includes consciousness marker integration (P3.1–P3.2) and neural complexity measures. MNE pipeline for real EEG signal processing. | — | — | H1, Roadmap (Level-specific signatures; empirical roadmap) | — |
| **VP-10** | Causal Manipulations | Full TMS/pharmacological causal protocol (VP_10_Falsification_CausalManipulations_TMS_Pharmacological_Priority2). Validates P2.a–P2.c named predictions in aggregator. Paper 3 clinical causal validation. Paper 4 Priority 2 listing in causal roadmap (Std 7). | P2 (TMS/pharmacological; Appendix A.3) | — | Clinical (H4; disorder-specific causal tests) | **Priority 2** (Std 7 Causal Roadmap) |
| **VP-11** | Cultural Neuroscience | Validates cross-cultural generalizability of APGI precision profiles and threshold parameters. Paper 3 includes disorder-specific precision profiles across DSM-5 conditions (60+ parameter table). Paper 4 cites P1–P4 predictions (Level 1 thermodynamic tier) including cross-population biological constraints. | — | — | Disorder-specific precision profiles | P1–P4 (Level 1 thermodynamic preds) |
| **VP-12** | Clinical / Cross-Species Validation | Combines DoC clinical validation (P4.a–P4.d) with cross-species allometric scaling (P12.a, P12.b). Paper 3 covers DoC clinical protocols and cross-species predictions. Paper 4 Priority 4 in causal roadmap; P12 named predictions in framework aggregator (FP-12). Allometric exponent 0.70–0.80, >85% cross-species consistency. | — | — | DoC / Clinical (Disorders of Consciousness table) | **Priority 4** (Std 7 Causal Roadmap; P12) |

---

### 2.3 Named Predictions Cross-Reference (FP_ALL_Aggregator — 18 predictions tracked)

| Prediction ID | Description | Source Protocol | Falsified if… |
| --------------- | ------------- | ----------------- | --------------- |
| P1.1 | Interoceptive precision modulates detection threshold (d=0.40–0.60) | FP-1 / FP_01_ActiveInference | d <0.35 or p≥0.01 |
| P1.2 | Arousal amplifies Πⁱ–threshold relationship | FP-1 / FP_01_ActiveInference | Interaction p≥0.01 |
| P1.3 | High-IA individuals show stronger arousal benefit | FP-1 / FP_01_ActiveInference | Effect absent or reversed |
| P2.a | dlPFC TMS shifts threshold >0.1 log units | VP-10 (TMS/Pharmacological) | Shift <0.05 log units |
| P2.b | Insula TMS reduces HEP ~30% AND PCI ~20% (double dissociation) | VP-10 (TMS/Pharmacological) | Either reduction absent |
| P2.c | High-IA × insula TMS interaction | VP-10 (TMS/Pharmacological) | Interaction p≥0.05 |
| P3.conv | APGI converges in 50–80 trials (beats baselines) | FP-2 / FP_02_AgentComparison | APGI >100 trials to criterion |
| P3.bic | APGI BIC lower than StandardPP and GWTOnly | FP-2 / FP_02_AgentComparison | ΔBIC <10 vs alternatives |
| P4.a | PCI+HEP joint AUC >0.80 for DoC classification | FP-9 / FP_09_NeuralSignatures | AUC <0.75 |
| P4.b | DMN to PCI r>0.50; DMN to HEP r<0.20 | FP-9 / FP_09_NeuralSignatures | Either correlation outside range |
| P4.c | Cold pressor increases PCI >10% in MCS, not VS | FP-9 / FP_09_NeuralSignatures | No differential response |
| P4.d | Baseline PCI+HEP predicts 6-month recovery ΔR²>0.10 | FP-9 / FP_09_NeuralSignatures | ΔR² <0.05 |
| P5.a | vmPFC to SCR anticipatory correlation r>0.40 | FP-5 / FP_05_EvolutionaryPlausibility | r <0.25 |
| P5.b | vmPFC uncorrelated with posterior insula (r<0.20) | FP-5 / FP_05_EvolutionaryPlausibility | r >0.30 |
| fp10a_mcmc | Bayesian MCMC: Gelman–Rubin R̂ ≤1.01 (convergence) | FP-10_BayesianEstimation_MCMC | R̂ >1.01 |
| fp10b_bf | BF₁₀ ≥3 for APGI vs StandardPP / GWT | FP-10_BayesianEstimation_MCMC | BF₁₀ <3 |
| fp10c_mae | APGI ≥20% lower MAE than alternatives | FP-10_BayesianEstimation_MCMC | MAE reduction <20% |
| fp10_scaling | Cross-species scaling: allometric exponents within ±2 SD | FP-12_CrossSpeciesScaling | Exponents >2 SD from expectation |

**Source:** Compiled from direct audit of FP_01–FP_09, FP_10_BayesianEstimation_MCMC, FP_11_LiquidNetworkDynamics_EchoState, FP_12_CrossSpeciesScaling, FP_ALL_Aggregator, and knowledge-base documents *APGI Series Structure, All Four Papers* and *APGI_Series_TOC_All_Papers*.

**FP-10 note:** FP-10 is internally split into FP10a (Bayesian MCMC — *FP_10_BayesianEstimation_MCMC.py*) and FP10b (Cross-Species Scaling — *FP_12_CrossSpeciesScaling.py*). Both must pass; either failure falsifies FP-10.

**FP-11 note:** *FP_11_LiquidNetworkDynamics_EchoState.py* re-implements and extends FP-6 (F6.1–F6.6) using echo state / liquid time-constant network methods. It explicitly runs "VP-6 specific tests for LTCN" (V6.1 threshold transition, V6.2 integration window) and LNN substrate topology tests described as "from Paper 2."

**Paper 3 (Multi-Scale) note:** Paper 3's empirical protocols are organized around four hypotheses (H1–H4). H1 = level-specific neural signatures (→ FP-9 / VP-9), H2 = cross-frequency coupling, H3 = developmental trajectories, H4 = clinical biomarker coupling dysregulation (→ VP-7 / VP-10). The DoC and cross-species clinical predictions map to VP-12.

**Paper 4 (Epistemic Architecture) note:** Paper 4's seven Methodological Standards (Std 1–7) and three-tier epistemic levels (Level 1 thermodynamic, Level 2 information-theoretic, Level 3 computational) provide the primary organizational framework for how protocols are classified. Level and Std labels in tables reflect these Paper 4 categories.
