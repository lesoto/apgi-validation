# APGI Paper Protocol

The GUI directly maps to full implementations, completely bypassing stubs:

| GUI Protocol | Maps To |
| ------------ | ------- |
| Protocol 1: APGI Agent | Falsification_ActiveInferenceAgents_F1F2.py |
| Protocol 2: Iowa Gambling | Falsification_AgentComparison_ConvergenceBenchmark.py |
| Protocol 3: Agent Comparison | Falsification_FrameworkLevel_MultiProtocol.py |
| Protocol 4: Phase Transition | Falsification_InformationTheoretic_PhaseTransition.py |
| Protocol 5: Evolutionary | Falsification_EvolutionaryPlausibility_Standard6.py |
| Protocol 6: Network Comparison | Falsification_NeuralNetwork_EnergyBenchmark.py |

## 1. File Inventory

### 1.1 Validation Files (16 total)

| # | Filename | VP ID | Paper Protocol |
| --- | --------- | ------- | --------------- |
| 1 | `ActiveInference_AgentSimulations_Protocol3.py` | VP-3 | Paper Protocol 3 |
| 2 | `BayesianModelComparison_ParameterRecovery.py` | VP-1 (support) | Paper Protocol 6 (partial) |
| 3 | `Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py` | VP-10 | Roadmap Priority 2 |
| 4 | `Clinical_CrossSpecies_Convergence_Protocol4.py` | VP-12 | Paper Protocol 4 / Roadmap Priority 4 |
| 5 | `ConvergentNeuralSignatures_Priority1_EmpiricalRoadmap.py` | VP-9 | Roadmap Priority 1 |
| 6 | `EvolutionaryEmergence_AnalyticalValidation.py` | VP-5 | — (Paper Protocol 5 MISSING) |
| 7 | `InformationTheoretic_PhaseTransition_Level2.py` | VP-4 | Epistemic Paper P5–P8 (Level 2) |
| 8 | `Master_Validation.py` | — | Orchestrator (not a protocol) |
| 9 | `NeuralNetwork_InductiveBias_ComputationalBenchmark.py` | VP-6 | Computational Architecture |
| 10 | `Psychophysical_ThresholdEstimation_Protocol1.py` | VP-8 | Paper Protocol 1 |
| 11 | `QuantitativeModelFits_SpikingLNN_Priority3.py` | VP-11 | Roadmap Priority 3 |
| 12 | `SyntheticEEG_MLClassification.py` | VP-1 (support) | Paper Protocol 6 (partial) |
| 13 | `TMS_Pharmacological_CausalIntervention_Protocol2.py` | VP-7 | Paper Protocol 2 |
| 14 | `Validation_Protocol_2.py` | VP-2 | Paper Protocol 6 (behavioral) |
| 15 | `Validation_Protocol_11.py` | VP-11 (canonical) | Roadmap Priority 3 / Cultural Neuro |
| 16 | `Validation_Protocol_P4_Epistemic.py` | VP-4 (support) | Epistemic Paper P5–P12 |

### 1.2 Falsification Files (22 total)

| # | Filename | FP ID | Description |
| --- | --------- | ------- | ----------- |
| 1 | `APGI_Falsification_Aggregator.py` | FP-12 | Framework-level aggregator |
| 2 | `APGI_Falsification_Protocols_GUI.py` | — | GUI runner (not a protocol) |
| 3 | `Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py` | FP-2 (support) | Causal manipulation engine |
| 10 | `Falsification_ActiveInferenceAgents_F1F2.py` | FP-1 (full) | Primary F1/F2 engine |
| 11 | `Falsification_AgentComparison_ConvergenceBenchmark.py` | FP-2 (full) | Convergence benchmark |
| 12 | `Falsification_BayesianEstimation_MCMC.py` | FP-10 | NUTS/PyMC MCMC estimation |
| 13 | `Falsification_BayesianEstimation_ParameterRecovery.py` | FP-11 | Parameter recovery |
| 14 | `Falsification_CrossSpeciesScaling_P12.py` | FP-10 (cross-species) | Allometric scaling |
| 15 | `Falsification_EvolutionaryPlausibility_Standard6.py` | FP-5 (full) | Evolutionary analysis |
| 16 | `Falsification_FrameworkLevel_MultiProtocol.py` | FP-3 (full) | Multi-protocol synthesis |
| 17 | `Falsification_InformationTheoretic_PhaseTransition.py` | FP-4 (full) | Phase transition analysis |
| 18 | `Falsification_LiquidNetworkDynamics_EchoState.py` | FP-11 / FP-12 | Echo-state / liquid network |
| 19 | `Falsification_MathematicalConsistency_Equations.py` | FP-7 | Sympy equation verification |
| 20 | `Falsification_NeuralNetwork_EnergyBenchmark.py` | FP-6 (full) | Energy / ATP benchmark |
| 21 | `Falsification_NeuralSignatures_EEG_P3b_HEP.py` | FP-9 | EEG / P3b / HEP neural sigs |
| 22 | `Falsification_ParameterSensitivity_Identifiability.py` | FP-8 | Sobol / FIM / profile likelihood |

---

### VALIDATION PROTOCOLS (12 Protocols)

## Validation Protocol 1 — Synthetic Neural Data Generation & ML Classification SyntheticEEG_MLClassification.py

McNemar's test and DeLong's AUC test are now fully implemented with permutation testing (minimum 1,000 permutations, α=0.05). The compare_models_with_statistics() function includes complete statistical testing with proper p-value computation and significance determination.
V12.1 thresholds corrected to paper specifications: V12_1_MIN_P3B_REDUCTION_PCT = 80.0 and V12_1_MIN_IGNITION_REDUCTION_PCT = 70.0 in falsification_thresholds.py.
F5.5 thresholds corrected to match paper: F5_5_PCA_MIN_VARIANCE = 0.70 and F5_5_MIN_LOADING = 0.60.
check_F6_2() now properly returns report["overall_falsified"] after appending to report["passed_criteria"].

## Validation Protocol 2 — Behavioral Validation (Validation_Protocol_2.py)

Full behavioral simulation implemented with complete psychometric function fitting (logistic), d-prime calculation, arousal condition branching, and Garfinkel SD-split IA group categorization.
Complete get_falsification_criteria() implementation returning P1.1, P1.2, P1.3 with paper-specified thresholds (d=0.40–0.60 range, arousal interaction d=0.25–0.45, IA group benefit d>0.30).
Protocol 2 tier classification corrected to "primary" with proper description alignment. Master_Validation.py now correctly registers this as the Behavioral Validation Protocol.

## Validation Protocol 3 — Agent Comparison Experiment (ActiveInference_AgentSimulations_Protocol3.py)

BIC/AIC model comparison fully implemented with softmax action-selection log-likelihoods. The compute_bic() function calculates BIC = −2·ln(L) + k·ln(n) and model comparison asserts APGI BIC < StandardPP BIC and < GWTOnly BIC.
Convergence aggregation fixed to treat non-convergent agents as n_max_trials (worst case) rather than excluding them, preventing upward bias in mean calculations.
Statistical test implemented: Mann-Whitney U test comparing APGI convergence trials vs. alternatives with α=0.01 for proper significance testing.
Convergence criterion properly anchored to paper-specified 50–80 trial range with robust trial counting.

## Validation Protocol 4 — Phase Transition Analysis (InformationTheoretic_PhaseTransition_Level2.py)

P6 passing condition corrected to bandwidth ≤ 40.0 bits/s (not 100.0). The implementation now properly uses the biological ceiling rather than the falsification threshold.
Metabolic cost model implemented using Attwell & Laughlin (2001) ~10⁹ ATP per spike with proper energy-per-correct-detection computation and biological ceiling comparison (≤20% above baseline).
Hurst exponent implemented via detrended fluctuation analysis (DFA) for proper long-range correlation analysis before checking F4.5.

## Validation Protocol 5 — Evolutionary Emergence (EvolutionaryEmergence_AnalyticalValidation.py)

All F5 thresholds in falsification_thresholds.py audited and corrected to match paper-cited values: F5_4_MIN_PEAK_SEPARATION = 3.0, F5_1_MIN_PROPORTION = 0.75, F5_1_MIN_ALPHA = 4.0, F5_5_PCA_MIN_VARIANCE = 0.70, F5_5_MIN_LOADING = 0.60.
Cross-modal falsification condition fixed with tolerance check: abs(intero_error - extero_error) < 1e-6 to properly detect near-identical precision weighting instead of degenerate float equality.

## Validation Protocol 6 — Network Comparison (NeuralNetwork_InductiveBias_ComputationalBenchmark.py)

F6.1 LTCN transition threshold corrected to 50.0ms (paper: <50ms).
F6.2 LTCN window threshold corrected to 200.0ms (paper: 200-500ms).
F6.2 integration ratio corrected to 4.0 (paper: ≥4× standard RNN).
F6.2 curve fit R2 corrected to 0.85 (paper: ≥0.85).
V6 latency/processing rate thresholds updated with biological citations.

## Validation Protocol 7 — Mathematical Consistency (Falsification-MathematicalConsistency-Equations.py)

get_falsification_criteria() is implemented with all four canonical APGI equations tested.
Tolerance tightened to 1e-6 matching V5.1 criteria_registry specification.
All four equations (surprise accumulation ODE, ignition sigmoid, precision update, threshold dynamics) are covered with comprehensive sympy-based verification.

## Validation Protocol 8 — Parameter Sensitivity (Psychophysical_ThresholdEstimation_Protocol1.py)

Two-pathway pharmacological disambiguation implemented: β-blockade (25-40% β reduction) vs cardiac feedback perturbation (15-25% Πⁱ reduction) with proper interaction testing.
Inline thresholds synchronized to paper specifications: F3.1: 0.18, F3.6: 200 trials.
Bonferroni correction implemented for 6-test battery (per-test α=0.008).
F6.1 threshold consistency fixed (50ms across all files).

## Validation Protocol 9 — Neural Signatures Validation (ConvergentNeuralSignatures_Priority1_EmpiricalRoadmap.py)

F6.1 standardized to ltcn_transition_time <= 50.0 across all files.
F6.2 standardized to paper specifications: >= 200.0ms, >= 4.0×, R² >= 0.85.
Power analysis gating implemented: criteria marked "UNDERPOWERED" when estimated power < 0.80.

## Validation Protocol 10 — Cross-Species Scaling / Causal Manipulations (CausalManipulations_TMS_Pharmacological_CausalIntervention_Protocol2.py)

Named prediction logic implemented: P2.a (>0.1 log units), P2.b (≥30% HEP AND ≥20% PCI reduction), P2.c (interaction η² ≥ 0.10).
APGI-derived confidence implemented as σ(Πⁱ · |εᵢ| − θₜ) replacing heuristic.
run_validation() returns structured {"P2.a": {...}, "P2.b": {...}, "P2.c": {...}} for Aggregator consumption.

## Validation Protocol 11 — Cultural Neuroscience / Bayesian Estimation

Cultural neuroscience motivation documented in lines 6-32 of Validation_Protocol_11.py, mapping to APGI parameters {θ₀, Πⁱ, β, α} with quantitative predictions from APGI-MULTI-SCALE-CONSCIOUSNESS-Paper.
PyMC3 MCMC implemented with 4 chains, 1,000 burn-in, 5,000 post-warmup samples (lines 99-102).
get_falsification_criteria() fully implemented with all 5 criteria (V11.1-V11.5) including Bayes-factor thresholds.
Gelman-Rubin convergence diagnostics (R̂ ≤ 1.01) fully implemented: _compute_rhat() function (lines 416-447) calculates R̂ for all parameters; assert_convergence() gate function (line 849) enforces R̂ ≤ 1.01; convergence check integrated in run_validation() (line 1080).
Credible interval calculations validated via np.percentile (lines 390-396) and ArviZ HDI (lines 537-540).

## Validation Protocol 12 — Liquid Network / Clinical-Cross-Species (Clinical_CrossSpecies_Convergence_Protocol4.py)

Framework-Level Aggregator (APGI_Falsification_Aggregator.py) has complete implementation:

- aggregate_prediction_results() loads JSON results from all protocols
- check_framework_falsification_condition_a() checks if all 14 predictions fail
- check_framework_falsification_condition_b() checks alternative framework distinctiveness
- generate_gnwt_predictions() and generate_iit_predictions() create alternative framework predictions
- FRAMEWORK_FALSIFICATION_THRESHOLD_A properly used in condition (a)

V12.1 thresholds corrected: V12_1_MIN_P3B_REDUCTION_PCT = 80.0 and V12_1_MIN_IGNITION_REDUCTION_PCT = 70.0 in falsification_thresholds.py.
V12.2 uses V12_2_MIN_CORRELATION = 0.60 (validation target) instead of falsification boundary.
LiquidTimeConstantChecker.check_ltc() implemented with echo state network simulation and F6.2 threshold checking.
Behavioral autonomy thresholds have paper citation: O'Reilly & Frank (2006) on exploratory behavior in reinforcement learning (lines 2978-2981).

## Falsification Protocols (12 Protocols)

### Falsification Protocol 1 — Active Inference Agents / F1 & F2

Falsification_ActiveInferenceAgents_F1F2.py

simulate_surprise_accumulation() implemented with ignition detection logic.
F6.5 bifurcation test now uses proper phase portrait sweep via analyze_bifurcation_structure() function.
F5_4_MIN_PEAK_SEPARATION = 3.0 in falsification_thresholds.py (matches paper spec and inline constants).

### Falsification Protocol 2 — Agent Comparison / Convergence Benchmark (Falsification_AgentComparison_ConvergenceBenchmark.py)

Duplicate F1–F3 criteria checking architecturally correct (shared criteria).
apgi_time_to_criterion type consistency: declared as float across all protocols (Python handles float/int comparisons correctly).
Survival analysis (log-rank test) implemented for F2.5 at lines 839-868 using lifelines.KaplanMeierFitter with scipy.stats.logrank fallback.

### Falsification Protocol 3 — Framework-Level Multi-Protocol (Falsification-FrameworkLevel-MultiProtocol.py)

Framework-level criteria (conditions a and b, the 14 named predictions) live in APGI_Falsification_Aggregator.py.
Naming/routing collision fixed: Protocol 3 path corrected to "../Validation/ActiveInference_AgentSimulations_Protocol3.py".
Synthesis loop implemented in run_full_experiment() method (lines 444-531) to load protocols and apply Aggregator logic.

## Falsification Protocol 4 — Information-Theoretic Phase Transition (Falsification-InformationTheoretic-PhaseTransition.py)

Level 1 metabolic cost uses Attwell & Laughlin ATP budget (10⁹ ATP/spike) at line 1242 with proper prefrontal cortex power budget comparison (~0.5W, ~10¹⁸ ATP/s).
Hurst exponent implemented via DFA (Detrended Fluctuation Analysis) at lines 479-525 for long-range correlations analysis.

## Falsification Protocol 5 — Evolutionary Plausibility (Falsification-EvolutionaryPlausibility-Standard6.py)

F6.5: Dynamic hysteresis computation from simulation data (no hardcoded constants)
F5.5: Correct import paths using paper-correct thresholds (0.60, 0.70)
F6.6: Validation added for alternative_modules_needed and performance_gap_without_addons parameters

## Falsification Protocol 6 — Neural Network Energy Benchmark (Falsification-NeuralNetwork-EnergyBenchmark.py)

F6.1/F6.2 thresholds corrected (50.0ms, 200.0ms, ratio=4.0×)
ATP cost per correct detection implemented using Attwell & Laughlin constraints
BIC/AIC model comparison implemented between APGI-network and baseline architectures

## Falsification Protocol 7 — Mathematical Consistency (Falsification-MathematicalConsistency-Equations.py)

Tolerance set to 1e-6 (lines 101, 1543, 1752, 1960, 2213).
All four core equations explicitly enumerated and tested with 1,000+ parameter draws.
Analytical cross-validation implemented via AnalyticalAPGISolutions import.

## Falsification Protocol 8 — Parameter Sensitivity & Identifiability (Falsification-ParameterSensitivity-Identifiability.py)

n_samples set to 1024 (power-of-2 compliant) at lines 676, 1184, 1191.
Profile likelihood analysis implemented (lines 393-560) for practical identifiability.
Fisher Information Matrix analysis implemented (lines 578-670).
Identifiability analysis with falsification criterion gate implemented.

### Falsification Protocol 9 — Neural Signatures EEG P3b HEP (Falsification-NeuralSignatures-EEG-P3b-HEP.py)

- P4.a (PCI+HEP joint AUC > 0.80) at lines 1460-1480
- P4.b (DMN↔PCI r > 0.50; DMN↔HEP r < 0.20) at lines 1650-1669
- P4.c (cold pressor increases PCI >10% in MCS) at lines 1773-1794
- P4.d (baseline PCI+HEP predicts 6-month recovery ΔR² > 0.10) at lines 1940-1961

run_validation() returns structured dict with all P4 predictions at lines 2140-2260.

### Falsification Protocol 10 — Bayesian Estimation MCMC (Falsification-BayesianEstimation-MCMC.py)

MCMC sampler implemented using PyMC NUTS (lines 206-214) with 5,000 samples, 4 chains, 1,000 burn-in.
Gelman-Rubin diagnostic (R̂ ≤ 1.01) implemented at lines 240-247.
Bayes factor computation via ArviZ LOO-CV and WAIC implemented at lines 216-228.
Priors over {θ₀, Πe, Πi, β, α} defined from physiological ranges at lines 88-100+.

#### Falsification Protocol 11 — Liquid Network Dynamics Echo State (Falsification-LiquidNetworkDynamics-EchoState.py)

Spectral radius guard implemented at lines 215-225 (scales to 0.98 if ≥ 1.0).
F6.3 (sparsity) implemented at lines 281-375 with ≥30% reduction test.
F6.5 (bifurcation sweep) implemented at lines 472-567 with input gain sweep and hysteresis detection.
F6_5_HYSTERESIS thresholds already correct in falsification_thresholds.py: MIN = 0.08, MAX = 0.25 (matches paper spec).

#### Falsification Protocol 12 — Framework-Level Aggregator (APGI_Falsification_Aggregator.py)

aggregate_prediction_results() fully implemented at lines 46-60 (loads JSON results, tallies predictions).
check_framework_falsification_condition_a() fully implemented at lines 63-72 with FRAMEWORK_FALSIFICATION_THRESHOLD_A used.
check_framework_falsification_condition_b() fully implemented at lines 75-89 with ALTERNATIVE_FRAMEWORK_PARSIMONY_THRESHOLD.
generate_gnwt_predictions() implemented at lines 92-110.
generate_iit_predictions() implemented at lines 113-131.
run_framework_falsification() implemented at lines 134-149+.

## APGI Protocol Tier Classification Rationale

## Overview

The APGI validation framework organizes validation protocols into three tiers based on their importance, scope, and relationship to core APGI theory. This classification enables weighted scoring and prioritized validation efforts.

## Tier Definitions

### Primary Protocols (Protocols 1-2)

**Rationale:** Core validation protocols that test fundamental APGI properties. These are essential for establishing basic APGI validity and must pass for the framework to be considered viable.

### Protocol 1: Basic Equation Validation

- Tests fundamental APGI mathematical equations
- Validates surprise accumulation, threshold adaptation, and ignition dynamics
- Checks parameter bounds and numerical stability
- **Why Primary:** Without passing Protocol 1, the APGI model itself is fundamentally broken

### Protocol 2: Parameter Consistency Checks

- Validates parameter interactions and constraints
- Tests parameter estimation accuracy
- Checks for parameter identifiability
- **Why Primary:** Parameter consistency is foundational to all other validations

**Weight in Scoring:** 50% (0.5)

### Secondary Protocols (Protocols 3-4, 8, 11-12)

**Rationale:** Extended validation covering specific aspects of APGI. These test important but not absolutely foundational properties. They provide deeper insights into APGI's behavior and falsifiability.

### Protocol 3: Behavioral Pattern Validation

- Tests whether APGI agents produce expected behavioral patterns
- Validates decision-making and action selection
- Compares APGI predictions to empirical behavioral data
- **Why Secondary:** Important for real-world applicability but builds on core equations

### Protocol 4: State Transition Verification

- Tests phase transition properties of ignition
- Validates information-theoretic signatures
- Checks for critical phenomena in APGI dynamics
- **Why Secondary:** Tests specific predictions about consciousness-related phenomena

### Protocol 8: Cross-Species Scaling Validation

- Tests whether APGI scales appropriately across species
- Validates scaling laws and allometric relationships
- Checks species-specific parameter adjustments
- **Why Secondary:** Important for generalizability but not core to APGI theory

### Protocol 11: Cultural Neuroscience Validation

- Tests cultural and contextual influences on APGI
- Validates cross-cultural parameter variations
- Checks for cultural universals vs. specifics
- **Why Secondary:** Important for ecological validity but not foundational

### Protocol 12: Liquid Network Validation

- Tests liquid network properties and dynamics
- Validates network topology and connectivity effects
- Checks for phase transitions in network states
- **Why Secondary:** Tests advanced network properties but not core requirements

**Weight in Scoring:** 30% (0.3)

### Tertiary Protocols (Protocols 5-7, 9-10)

**Rationale:** Specialized and experimental protocols that test niche aspects, computational implementations, or exploratory hypotheses. These provide supplementary evidence but are not critical for core validation.

### Protocol 5: Computational Benchmarking

- Tests computational efficiency and performance
- Validates implementation quality
- Benchmarks against alternative architectures
- **Why Tertiary:** Important for practical use but not theoretical validity

### Protocol 6: Bayesian Estimation Framework

- Tests parameter estimation using Bayesian methods
- Validates uncertainty quantification
- Checks posterior predictive validity
- **Why Tertiary:** Advanced statistical methods but not required for basic validation

### Protocol 7: Multimodal Integration

- Tests integration of multiple data modalities
- Validates cross-modal consistency
- Checks for modality-specific parameter adjustments
- **Why Tertiary:** Important for real-world applications but not core theory

### Protocol 9: Psychological States Validation

- Tests APGI predictions about psychological states
- Validates state-dependent parameter changes
- Checks for emotion and motivation effects
- **Why Tertiary:** Tests specific predictions but not fundamental

### Protocol 10: Turing Machine Validation

- Tests computational universality properties
- Validates APGI as a computational model
- Checks for algorithmic completeness
- **Why Tertiary:** Theoretical curiosity but not required for validation

**Weight in Scoring:** 20% (0.2)

## Weighted Scoring Rationale

The tier weights (Primary: 0.5, Secondary: 0.3, Tertiary: 0.2) reflect:

1. **Criticality:** Primary protocols must pass for APGI to be viable
2. **Scope:** Secondary protocols cover important but not essential aspects
3. **Supplementary:** Tertiary protocols provide additional evidence

This weighting ensures that:

- Core APGI validity dominates the overall assessment
- Important but not critical aspects contribute meaningfully
- Supplementary evidence provides context without overwhelming core results

## Decision Thresholds

Based on weighted scores:

- **PASS (≥0.85):** Strong validation support - APGI is well-validated
- **MARGINAL (0.60-0.84):** Moderate validation support - APGI has issues but may be viable
- **FAIL (<0.60):** Insufficient validation support - APGI requires major revisions
