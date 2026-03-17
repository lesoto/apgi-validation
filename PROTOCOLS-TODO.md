# VALIDATION PROTOCOLS (12 Protocols)

## Validation Protocol 1 — Synthetic Neural Data Generation & ML Classification SyntheticEEG-MLClassification.py

CRITICAL — McNemar's test is scaffolded but never completed. compare_models_with_statistics() sets "p_value": None and "significant": None with a comment "compute via permutation" — the permutation loop is missing entirely. This is a named primary statistical test in the paper.
HIGH — Threshold string/numeric mismatch. criteria_registry.py records V12.1 threshold text as "P3b reduction ≥80%, ignition reduction ≥70%, d ≥ 0.80" but falsification_thresholds.py implements V12_1_MIN_P3B_REDUCTION_PCT = 10.0 and V12_1_MIN_IGNITION_REDUCTION_PCT = 15.0. The threshold is 8× too low for P3b and 4.7× too low for ignition. The criteria_registry is the paper-faithful reference; the constants file is wrong.
HIGH — F5.5 threshold inconsistency across files. falsification_thresholds.py sets F5_5_PCA_MIN_VARIANCE = 0.60 / F5_5_MIN_LOADING = 0.40, while the paper-faithful description in NeuralNetwork-InductiveBias-ComputationalBenchmark.py states "Cumulative variance ≥70%, min loading ≥0.60". The canonical constants file is operating at 60% vs. the paper's 70%.
MEDIUM — DeLong's AUC test referenced in compare_models_with_statistics() but not implemented.

TODO:

Implement the permutation loop in compare_models_with_statistics() for both McNemar's and DeLong's AUC tests (minimum 1,000 permutations, α=0.05).
Fix V12_1_MIN_P3B_REDUCTION_PCT to 80.0 and V12_1_MIN_IGNITION_REDUCTION_PCT to 70.0 in falsification_thresholds.py.
Fix F5_5_PCA_MIN_VARIANCE to 0.70 and F5_5_MIN_LOADING to 0.60.
Add return to check_F6_2() — the current version appends to report["passed_criteria"] but never sets report["overall_falsified"] before returning.


## Validation Protocol 2 — Behavioral Validation (Validation-Protocol-2.py)

CRITICAL — File content not indexed in project knowledge, suggesting it may be a minimal stub. The Master declares "Behavioral Validation Protocol" but no behavioral simulation logic (reaction times, accuracy curves, psychometric functions) appears in any indexed file under this filename.
HIGH — No falsification criteria function get_falsification_criteria() is visible for this protocol — the paper requires P1.1 (d=0.40–0.60 interoceptive advantage), P1.2 (arousal interaction), P1.3 (IA group interaction).
**HIGH — Protocol 2 tier classification in Master_Validation.py says "primary" but the internal rationale comment says "Parameter consistency checks" — contradicting the description "Behavioral Validation Protocol". These must align.

TODO:

Implement full behavioral simulation: psychometric function fitting (Weibull or logistic), d-prime calculation, arousal condition branching, IA group split (>1 SD per Garfinkel).
Add get_falsification_criteria() returning P1.1, P1.2, P1.3 with paper thresholds (d=0.40–0.60 range check, r=−0.30 to −0.50, arousal effect size 0.40–0.60).
Reconcile the tier comment with the actual protocol description.


## Validation Protocol 3 — Agent Comparison Experiment (ActiveInference-AgentSimulations-Protocol3.py)

HIGH — Convergence criterion uses 70% by trial 45 in the get_falsification_criteria() description string, but the paper (P3.conv in the Aggregator) specifies 50–80 trials. The implementation checks a numerical threshold but trial counting is not robustly anchored to either boundary.
HIGH — BIC comparison (P3.bic) for model comparison is declared in NAMED_PREDICTIONS ("APGI BIC lower than StandardPP and GWTonly") but no BIC computation is present in the agent simulation file — only reward curves.
MEDIUM — _aggregate_results() uses np.mean() of convergence trials that are None (unconverged agents are filtered), silently biasing the mean upward. The denominator should be total agents, not converged agents.

TODO:

Implement BIC/AIC model comparison: fit softmax action-selection log-likelihoods per agent type over trial sequences, compute BIC = −2·ln(L) + k·ln(n), assert APGI BIC < StandardPP BIC and < GWTOnly BIC.
Fix convergence aggregation: treat non-convergent agents as n_max_trials (worst case), not excluded.
Add statistical test (Mann-Whitney U) comparing APGI convergence trials vs. alternatives, with α=0.01.


## Validation Protocol 4 — Phase Transition Analysis (InformationTheoretic-PhaseTransition-Level2.py)

HIGH — P6 threshold direction is inverted in implementation. The criteria dict sets "threshold": 100.0 (falsification threshold) and "ceiling": 40.0 (expected), but the check logic in the broader codebase compares bandwidth ≤ 100 to pass — which means a model outputting 99 bits/s passes. The paper says the ceiling is 40 bits/s; the falsification threshold (>100) is an extreme outlier test. The passing condition should be bandwidth ≤ 40.
HIGH — Level 1 falsification stubs are explicitly unimplemented. run_level1_falsification_stubs() contains three # TODO: Implement actual metabolic cost calculation blocks with placeholder > 10.0 cost-benefit threshold and no actual ATP or metabolic model — yet Level 1 (P9–P12) predictions are declared complete in Validation-Protocol-P4-Epistemic.py.
MEDIUM — Hurst exponent (F4.5) is declared in the criteria dict (threshold: 0.55) but no DFA or R/S analysis is implemented — the value must be computed, not assumed.

TODO:

Fix P6 passing condition to bandwidth ≤ 40.0 bits/s (not 100.0).
Implement metabolic cost model: use Attwell & Laughlin (2001) ~10⁹ ATP per spike; compute energy-per-correct-detection; compare to biological ceiling (≤20% above baseline).
Implement Hurst exponent via detrended fluctuation analysis (DFA) using numpy or nolds library before checking F4.5.


## Validation Protocol 5 — Evolutionary Emergence (EvolutionaryEmergence-AnalyticalValidation.py)

HIGH — F5 thresholds in falsification_thresholds.py are systematically lower than in Falsification-ActiveInferenceAgents-F1F2.py. Specifically: F5_1_MIN_PROPORTION = 0.70 in the canonical file but the inline comment in F1F2.py says 0.75 (validated against paper). F5_1_MIN_ALPHA = 3.0 in the constants file but 4.0 inline. F5_5_PCA_MIN_VARIANCE = 0.60 vs. paper 0.70. The canonical constants file (falsification_thresholds.py) consistently uses relaxed values.
HIGH — F5_4_MIN_PEAK_SEPARATION = 0.20 in falsification_thresholds.py but the inline-validated constant in F1F2.py is 3.0 (separation ≥ 3×). This is a 15× discrepancy. Any file importing from falsification_thresholds.py will apply an essentially non-restrictive criterion.
MEDIUM — Cross-modal falsification condition (combined_error >= extero_error) uses a degenerate equality check — floating point equality on intero_error == extero_error is essentially never triggered, making this criterion trivially pass.

TODO:

Audit and correct every constant in falsification_thresholds.py against the inline "VALIDATED" constants in Falsification-ActiveInferenceAgents-F1F2.py. Use the inline values as ground truth (they bear paper citations).
Specifically set F5_4_MIN_PEAK_SEPARATION = 3.0, F5_1_MIN_PROPORTION = 0.75, F5_1_MIN_ALPHA = 4.0, F5_5_PCA_MIN_VARIANCE = 0.70, F5_5_MIN_LOADING = 0.60.
Replace equality check with tolerance: abs(intero_error - extero_error) < 1e-6 to catch near-identical precision weighting.


Validation Protocol 6 — Network Comparison (NeuralNetwork-InductiveBias-ComputationalBenchmark.py)
Rating: 71/100
What is correctly implemented:

APGI-inspired dual-pathway network vs. MLP, LSTM, Attention baselines
Comprehensive evaluation: accuracy, AUC, energy per spike (ATP model)
F5.1–F6.2 criteria imported and formally checked
Spike-based energy logging with Attwell & Laughlin constant

Critical deficiencies:

HIGH — F6.1 imports F6_1_LTCN_MAX_TRANSITION_MS = 200.0 from the constants file, but the paper spec in the get_falsification_criteria() dict within this same file says "<50ms" for LTCN transition. The constant is 4× too permissive.
HIGH — F6.2 imports F6_2_LTCN_MIN_WINDOW_MS = 50.0 (constants file) but the paper spec in the same file says "200-500ms" window. The constant is 4× too small.
HIGH — F6_2_MIN_INTEGRATION_RATIO = 1.50 in constants, but paper spec says "≥4× standard RNN". The constant allows passing at 1.5×, one-third of the required ratio.
MEDIUM — V6 latency/processing rate checks (V6_1_MAX_LATENCY_MS = 100.0, V6_1_MIN_PROCESSING_RATE = 50.0) have no paper citation and appear to be placeholder values.

TODO:

Set F6_1_LTCN_MAX_TRANSITION_MS = 50.0 (paper: <50ms).
Set F6_2_LTCN_MIN_WINDOW_MS = 200.0 (paper: 200-500ms).
Set F6_2_MIN_INTEGRATION_RATIO = 4.0 (paper: ≥4×).
Set F6_2_MIN_CURVE_FIT_R2 = 0.85 (paper says ≥0.85, constant currently 0.80).
Derive V6 latency/rate thresholds from the biological ATP/spike literature or remove them pending paper specification.


Validation Protocol 7 — Mathematical Consistency (Falsification-MathematicalConsistency-Equations.py)
Rating: 66/100
What is correctly implemented (inferred from GUI registration):

MathematicalConsistencyChecker class registered with epsilon, n_samples, tolerance parameters
Numerical derivative checking infrastructure

Critical deficiencies (inferred from GUI and partial indexing):

HIGH — No get_falsification_criteria() implementation appears in indexed files. Without it, the Master Validator cannot register falsification status.
HIGH — Tolerance 1e-5 used in GUI default but the criteria_registry V5.1 specifies ε ≤ 1e-6. The tolerance is 10× too loose.
MEDIUM — Mathematical consistency tests should verify all core APGI equations (surprise accumulation ODE, ignition sigmoid, precision update, threshold dynamics) against closed-form solutions — it is unclear whether all four equations are covered.

TODO:

Add get_falsification_criteria() returning all equation-level checks with paper thresholds.
Tighten numerical tolerance to 1e-6 (matching V5.1 in criteria_registry).
Enumerate and test all four canonical APGI equations; document which paper equation each test maps to.


Validation Protocol 8 — Parameter Sensitivity (Psychophysical-ThresholdEstimation-Protocol1.py)
Rating: 70/100
What is correctly implemented:

Bayesian adaptive psychophysics (Psi method) for threshold estimation
Full individual differences analysis: HEP amplitude, HRV RMSSD, IA group splits
6 falsification TODOs explicitly coded (pharmacological β-blockade, Garfinkel SD-split, Khalsa benchmark)
Factor analysis of parameter structure
Test-retest reliability scaffold

Critical deficiencies:

CRITICAL — All 6 TODOs are scaffolded but some contain placeholder logic. E.g., TODO_6_beta_disambiguation simulates β-blockade by reducing pi_i by 30% — a made-up model, not a validated pharmacological effect size. The paper requires a dissociation test between β (somatic bias) and Πⁱ (interoceptive precision), not a uniform reduction.
HIGH — check_falsification() in this file uses looser inline thresholds than the paper. F3.1 checks apgi_advantage_f3 >= 0.12 but the display string says "Advantage ≥18%". F3.6 checks time_to_criterion <= 250 but displays "Time ≤200 trials".
HIGH — F6.1 inline check uses ltcn_transition_time <= 50 (consistent with paper) but falsification_thresholds.py exports F6_1_LTCN_MAX_TRANSITION_MS = 200.0 — so files that import from the constants file are silently using a 4× relaxed threshold.

TODO:

Implement pharmacological disambiguation using a two-pathway model: β-blockade reduces somatic bias coefficient β by 25–40% (literature range), while a separate Πⁱ manipulation (e.g., cardiac feedback perturbation) reduces interoceptive precision independently. The dissociation test is a significant β × pathway interaction.
Sync inline thresholds to match the paper descriptions in all display strings (F3.1: 0.18, F3.6: 200 trials).
Add Bonferroni correction for the 6-test battery (family-wise α=0.05 → per-test α=0.008).


Validation Protocol 9 — Neural Signatures Validation (ConvergentNeuralSignatures-Priority1-EmpiricalRoadmap.py)
Rating: 73/100
What is correctly implemented:

F1.1–F6.2 full criteria check suite covering 24 named criteria
Comprehensive parameter signature through check_falsification()
F6.1 inline check uses the paper-correct ≤50ms threshold

Critical deficiencies:

HIGH — F6.1 in this file uses ltcn_transition_time <= 80 (line in ConvergentNeuralSignatures), not 50 — inconsistent with the same file's display string "LTCN time ≤50ms". Three different values (50, 80, 200) exist across three files for the same criterion.
HIGH — F6.2 in this file uses ltcn_integration_window >= 150 and ratio >= 2.5 and curve_fit_r2 >= 0.70 vs. paper: ≥200ms, ≥4×, R²≥0.85. All three sub-criteria are relaxed.
MEDIUM — Power analysis (N=80 for primary tests) is called but not linked to protocol-level go/no-go decisions; it should gate the overall protocol pass/fail when sample size is insufficient.

TODO:

Standardize F6.1: ltcn_transition_time <= 50.0 across all files. Run a global text search for 80 and 200 in F6.1 contexts and correct them.
Standardize F6.2: >= 200.0ms, >= 4.0×, R² >= 0.85.
Make power analysis gating explicit: if estimated power < 0.80 for any primary test, mark that criterion as "UNDERPOWERED" rather than pass/fail.


Validation Protocol 10 — Cross-Species Scaling / Causal Manipulations (CausalManipulations-TMS-Pharmacological-Priority2.py)
Rating: 58/100
What is correctly implemented:

TMS (dlPFC, insula) simulation with threshold and HEP effect modeling
Passing threshold uses overall_causal_validation_score > 0.5 (simple majority)
get_falsification_criteria() present

Critical deficiencies:

CRITICAL — The passing criterion (> 0.5) is not paper-specified. The paper's named predictions P2.a, P2.b, P2.c each have quantitative thresholds (dlPFC shifts >0.1 log units; insula reduces HEP ~30% AND PCI ~20%; High-IA × insula interaction). A majority score ignores whether any specific mandatory prediction passes.
HIGH — Confidence calculation (_calculate_confidence()) uses stimulus intensity, response accuracy, and RT — a generic placeholder not connected to the actual APGI confidence model (which should derive from precision-weighted prediction error).
HIGH — No run_validation() in Protocol 10 returns structured pass/fail per named prediction (P2.a, P2.b, P2.c) for propagation to the Aggregator's NAMED_PREDICTIONS dictionary — the Aggregator will therefore have no data for these three predictions.

TODO:

Replace scalar score with named-prediction logic: P2.a passes if dlpfc_threshold_shift > 0.1 log units (t-test, p<0.01); P2.b passes if hep_reduction >= 0.30 AND pci_reduction >= 0.20; P2.c passes if interaction F > threshold (partial η² ≥ 0.10).
Return {"P2.a": {...}, "P2.b": {...}, "P2.c": {...}} from run_validation() for Aggregator consumption.
Implement APGI-derived confidence as σ(Πⁱ · |εᵢ| − θₜ) rather than the current heuristic.


Validation Protocol 11 — Cultural Neuroscience / Bayesian Estimation
Rating: 48/100
What is correctly implemented:

Registered in Master Validator as "Bayesian Estimation"
MCMC parameters (n_chains, burn_in, n_samples) accessible via GUI

Critical deficiencies:

CRITICAL — No implementation is indexed. The file Validation-Protocol-11.py appears to be a stub run_validation() shell only. No MCMC sampler, no prior/posterior specification, no Gelman-Rubin convergence diagnostics, no credible interval calculation.
CRITICAL — Cultural neuroscience claims (the "secondary" tier rationale from Master_Validation) have no connection to APGI's mathematical model — it is unclear which paper section motivates this protocol.
HIGH — Missing get_falsification_criteria() means falsification_status cannot be populated.

TODO:

Identify the paper section that specifies Protocol 11's theoretical motivation and quantitative predictions; map to specific APGI parameters.
Implement PyMC3 or NumPyro MCMC with at least 4 chains, 1,000 burn-in, 5,000 post-warmup samples.
Report R̂ (Gelman-Rubin) ≤ 1.01 for all parameters as a convergence gate.
Add get_falsification_criteria() with Bayes-factor thresholds (BF > 10 for decisive evidence).


Validation Protocol 12 — Liquid Network / Clinical-Cross-Species (Clinical-CrossSpecies-Convergence-Protocol4.py)
Rating: 67/100
What is correctly implemented:

V12.1 (clinical gradient) and V12.2 (cross-species homology) formally checked
Imports from falsification_thresholds.py for V12 constants
LiquidTimeConstantChecker class present (stub)
IIT-APGI convergence logic present

Critical deficiencies:

CRITICAL — V12.1 check imports V12_1_MIN_P3B_REDUCTION_PCT = 10.0 and V12_1_MIN_IGNITION_REDUCTION_PCT = 15.0 — but the paper spec (get_falsification_criteria() in this same file) says "≥80% reduction" and "≥70% reduction". The protocol checks against 10% when it should check against 80%.
HIGH — V12.2 check uses V12_2_FALSIFICATION_CORR = 0.50 (the minimum to avoid falsification) rather than the paper's target r ≥ 0.60. The falsification boundary and the validation target are being used interchangeably.
HIGH — LiquidTimeConstantChecker.check_ltc() returns {"status": "implemented"} with no actual computation — it is a documented stub.
MEDIUM — Behavioral autonomy checks use fixed 0.55 spontaneous ratio and 0.50 novelty mean thresholds with no paper citation.

TODO:

Fix V12_1_MIN_P3B_REDUCTION_PCT = 80.0 and V12_1_MIN_IGNITION_REDUCTION_PCT = 70.0 in falsification_thresholds.py.
Use V12_2_MIN_CORRELATION = 0.60 (the validation target), not the falsification boundary, as the passing threshold.
Implement check_ltc(): simulate an echo state network with configurable spectral radius and leak rate; measure autocorrelation decay and integration window; check against F6.2 thresholds.






# FALSIFICATION PROTOCOLS (12 Protocols)

## Falsification_ActiveInferenceAgents_F1F2.py

Falsification Protocol 1 — Active Inference Agents / F1 & F2 

HIGH — simulate_surprise_accumulation() uses a placeholder sigmoid (B_traj[i]) that does not connect to the actual ignition boolean — the ignition_times list is populated but the appending logic (if B_traj[i] > threshold) is truncated in the indexed code. Ignition detection is incomplete.
HIGH — F6.5 bifurcation test in Falsification-EvolutionaryPlausibility-Standard6.py uses a hardcoded bifurcation_point = 0.15 and hysteresis = abs(0.15 - 0.05) rather than computing these from the actual simulated phase portrait. This is fabricated data.
MEDIUM — F5_4_MIN_PEAK_SEPARATION = 0.20 in falsification_thresholds.py contradicts 3.0 inline — any file that imports from the constants file gets a 15× relaxed criterion.

TODO:

Complete the ignition detection loop: append i * dt to ignition_times when S_t > theta_t.
Replace hardcoded bifurcation analysis with a proper phase portrait sweep: vary input drive from 0 to 2×θ_t in 100 steps, record stable ignition states, fit sigmoid to find the bifurcation point and hysteresis width.
Fix F5_4_MIN_PEAK_SEPARATION = 3.0 in falsification_thresholds.py.


Falsification Protocol 2 — Agent Comparison / Convergence Benchmark (Falsification-AgentComparison-ConvergenceBenchmark.py)
Rating: 72/100
What is correctly implemented:

Same comprehensive check_falsification() signature as Protocol 1 (F1–F6 full suite)
F2-specific parameters: Iowa Gambling Task, somatic marker advantage, RT advantage, confidence effect, learning trajectory

Critical deficiencies:

HIGH — Duplicate F1–F3 criteria checking across Protocols 1, 2, 3, 6, 9, and 12 means any threshold error in the constants file is inherited by all six. This is architecturally correct (shared criteria), but the inconsistency in falsification_thresholds.py propagates to all.
HIGH — apgi_time_to_criterion is typed as float here (in Protocol 2's function signature) but as int in Protocol 1's and in the paper (trials are integers). Type inconsistency could cause silent float comparison errors (<= on trial counts).
MEDIUM — No survival analysis (log-rank test) is present for F2.5 (learning trajectory discrimination — "APGI reaches 70% by trial 45"), only a scalar apgi_time_to_criterion comparison.

TODO:

Fix apgi_time_to_criterion type to int consistently across all function signatures.
Implement Kaplan-Meier log-rank test for F2.5: treat each agent's convergence trial as a survival event; compare APGI vs. no-somatic curves with scipy.stats.logrank_test or lifelines.
Add an explicit cross-file threshold consistency test (a unit test that asserts falsification_thresholds.py values match the inline paper-validated values in F1F2.py).


Falsification Protocol 3 — Framework-Level Multi-Protocol (Falsification-FrameworkLevel-MultiProtocol.py)
Rating: 65/100
What is correctly implemented (from GUI registration):

AgentComparisonExperiment class
Episode/agent-count parameterization

Critical deficiencies:

HIGH — GUI registers this file for "Protocol 3: Agent Comparison" but Master_Validation.py registers Protocol-3 as "Agent Comparison Experiment" pointing to Validation-Protocol-3.py. There is a naming/routing collision between falsification and validation Protocol 3.
HIGH — No indexed get_falsification_criteria() visible for this file — the framework-level criteria (conditions a and b, the 14 named predictions) live in APGI-Falsification-Aggregator.py, not here.
MEDIUM — "Framework-Level" suggests this should synthesize results from Protocols 1–12 and apply the Aggregator logic — but no such synthesis loop is visible.

TODO:

Rename the file to avoid collision with Validation-Protocol-3.py; update GUI and Master registrations.
Import and call APGI-Falsification-Aggregator functions from here to close the framework-level assessment loop.
Implement synthesis: load each protocol's JSON results, call aggregate_prediction_results(), then check_framework_falsification_condition_a() and _condition_b().


Falsification Protocol 4 — Information-Theoretic Phase Transition (Falsification-InformationTheoretic-PhaseTransition.py)
Rating: 63/100
What is correctly implemented:

Transfer entropy falsification (F4.1 approximation)
Mutual information falsification
Integrated information with surrogate baseline
Level 2 phase transition criteria (F4.1–F4.5)

Critical deficiencies:

CRITICAL — Level 1 falsification (metabolic cost, energy efficiency, thermodynamic plausibility) is explicitly a stub with three # TODO: Implement actual comments. All three stubs use placeholder thresholds (> 10.0, < 0.1, < 0.01) with no paper citation.
HIGH — run_level1_falsification_stubs() always returns metabolic_cost_falsified: False when the cost-benefit ratio is ≤ 10.0 — but the placeholder formula is not grounded in biology. The Attwell & Laughlin ATP budget (∼10⁹ ATP/spike) must be the basis.
HIGH — overall_falsified logic (transfer OR mutual OR integrated) is too permissive — any one of three criteria falsifies the entire protocol. The paper specifies that all three must fail for falsification, not any one.

TODO:

Implement metabolic cost: measure spike count per ignition event; multiply by 1e9 ATP; compare to energetic budget of prefrontal cortex (~0.5W, ~10¹⁸ ATP/s).
Fix overall_falsified logic to require transfer_entropy_falsified AND mutual_info_falsified AND integrated_info_falsified (conjunction, not disjunction) for Level 2; Level 1 should be a separate gate.
Implement Hurst exponent via DFA for F4.5.


Falsification Protocol 5 — Evolutionary Plausibility (Falsification-EvolutionaryPlausibility-Standard6.py)
Rating: 69/100
What is correctly implemented:

F5.1–F6.6 criteria all formally checked
Genome-based analysis with evolutionary selection pressure
F6.5 bifurcation and hysteresis check present
Binomial tests for emergence proportions

Critical deficiencies:

CRITICAL — F6.5 uses hardcoded values: hysteresis = abs(0.15 - 0.05) = 0.10, bifurcation_point = 0.15. These are not computed from a simulation — they are literal constants that will always produce the same result regardless of model parameters.
HIGH — F5.5 import path: from falsification_thresholds import F5_5_PCA_MIN_LOADING, F5_5_PCA_MIN_VARIANCE — these are the relaxed values (0.40, 0.60), not the paper-correct values (0.60, 0.70).
MEDIUM — alternative_modules_needed and performance_gap_without_addons for F6.6 are passed in as parameters from the caller — meaning correctness depends entirely on how the caller generates these values. No defensive validation of their derivation exists.

TODO:

Replace F6.5 hardcoded values with a bifurcation sweep: iterate Πⁱ·|εᵢ| from 0 to 2 in 200 steps; record B_t; fit a threshold sigmoid; extract the 50% crossing point and hysteresis width from ascending vs. descending sweeps.
Import from the corrected falsification_thresholds.py after fixing F5.5 constants.
Add caller-side validation: assert alternative_modules_needed >= 0 and performance_gap_without_addons >= 0 before passing to check_falsification().


Falsification Protocol 6 — Neural Network Energy Benchmark (Falsification-NeuralNetwork-EnergyBenchmark.py / NetworkComparisonExperiment)
Rating: 70/100
What is correctly implemented (inferred from GUI):

NetworkComparisonExperiment class with extero/intero/action/context dimensions
Episode-based comparison

Critical deficiencies (same as Validation Protocol 6 — they share threshold infrastructure):

HIGH — Shares all threshold discrepancies with Validation-Protocol-6 (F6.1=200ms, F6.2=50ms, ratio=1.5×).
HIGH — No paper-grounded ATP cost per correct detection comparison is visible in the indexed content.
MEDIUM — BIC/AIC model comparison must be implemented as specified in the paper.

TODO: Same as Validation Protocol 6, plus: implement formal BIC comparison between APGI-network and baseline architectures fitted to the same task data.

Falsification Protocol 7 — Mathematical Consistency (Falsification-MathematicalConsistency-Equations.py)
Rating: 64/100
What is correctly implemented:

Numerical derivative checking with configurable epsilon and tolerance
MathematicalConsistencyChecker class
Sample count parameterization

Critical deficiencies:

HIGH — Tolerance 1e-5 (GUI default) vs. paper 1e-6 (V5.1 in criteria_registry). The implementation is 10× less stringent.
HIGH — Equations tested are not enumerated in indexed content. The paper requires testing at minimum: (a) surprise ODE dS/dt = −S/τS + Πe·|εe| + β·Πi·|εi|, (b) ignition sigmoid B = σ(α(S−θt)), (c) allostatic threshold update dθt/dt, (d) free energy gradient. It's unknown if all four are covered.
MEDIUM — No analytical cross-validation against closed-form solutions (as done elegantly in EvolutionaryEmergence-AnalyticalValidation.py).

TODO:

Set default tolerance to 1e-6.
Enumerate all four core equations explicitly; test each with at least 1,000 random parameter draws within physiological ranges.
Add closed-form cross-validation for steady-state solutions wherever analytical solutions exist (import from AnalyticalAPGISolutions).


Falsification Protocol 8 — Parameter Sensitivity & Identifiability (Falsification-ParameterSensitivity-Identifiability.py)
Rating: 67/100
What is correctly implemented (from GUI):

Sobol sensitivity analysis (n_samples, n_trials parameters)
One-at-a-time (OAT) sensitivity with n_levels
ParameterSensitivityAnalyzer class

Critical deficiencies:

HIGH — Sobol indices require n_samples to be a power of 2 for SALib's Saltelli sampler — the default 1000 is not a power of 2 (nearest valid values: 512, 1024). This will cause silent approximation errors or runtime warnings.
HIGH — No identifiability analysis is visible — Sobol sensitivity tells you which parameters matter, but identifiability (whether parameters are uniquely recoverable from data) requires profile likelihood or MCMC posterior geometry analysis. The paper standard requires both.
MEDIUM — No falsification criterion gate: sensitivity analysis should declare falsification if S_total(θt) < S_total(Πi) (threshold less sensitive than precision — contradicting APGI's theoretical hierarchy).

TODO:

Fix n_samples to 1024 (default) and enforce power-of-2 in the parameterization.
Add profile likelihood for each of the four APGI parameters to test practical identifiability (flat profiles = non-identifiable).
Add a falsification gate based on sensitivity rank ordering: APGI predicts S_total(θt) > S_total(β) > S_total(Πi) > S_total(Πe) — any inversion should register as a falsification signal.


Falsification Protocol 9 — Neural Signatures EEG P3b HEP (Falsification-NeuralSignatures-EEG-P3b-HEP.py)
Rating: 66/100
What is correctly implemented (from GUI):

NeuralSignatureValidator class
P3b and HEP signatures

Critical deficiencies:

CRITICAL — Named predictions P4.a (PCI+HEP joint AUC > 0.80), P4.b (DMN↔PCI r > 0.50; DMN↔HEP r < 0.20), P4.c (cold pressor increases PCI >10% in MCS), P4.d (baseline PCI+HEP predicts 6-month recovery ΔR² > 0.10) must each map to a specific falsification check. No indexed content confirms these four are implemented.
HIGH — NeuralSignatureValidator must interface with the Aggregator so P4.a–P4.d can be tallied in NAMED_PREDICTIONS. Without this interface, four of the fourteen named predictions will always show as unknown.
MEDIUM — HEP amplitude simulation must use a physiologically valid range (10–50 µV) and cardiac cycle timing (R-peak ± 600ms) — it is unclear if the synthetic data generation respects these constraints.

TODO:

Implement all four P4 named-prediction checks with paper thresholds; return them in a dict that matches the NAMED_PREDICTIONS key format.
Add run_validation() that returns {"P4.a": ..., "P4.b": ..., "P4.c": ..., "P4.d": ...} for Aggregator consumption.
Validate HEP amplitude range against literature (Candia-Rivera et al., 2021).


Falsification Protocol 10 — Bayesian Estimation MCMC (Falsification-BayesianEstimation-MCMC.py)
Rating: 55/100
What is correctly implemented (from GUI):

BayesianEstimation class with MCMC parameters (n_samples, n_chains, burn-in)

Critical deficiencies:

CRITICAL — No sampler implementation visible. 5,000 MCMC samples across 4 chains with 1,000 burn-in is specified in the GUI but the actual sampling logic is unindexed.
HIGH — Gelman-Rubin diagnostic (R̂ ≤ 1.01) is a mandatory convergence check not visible in indexed content.
HIGH — Bayes factor computation for model comparison (APGI vs. StandardPP vs. GWTOnly) is not implemented.

TODO:

Implement MCMC using PyMC or NumPyro: define priors over {θ₀, Πe, Πi, β, α} from physiological ranges; define likelihood as the APGI psychometric function; sample posterior.
Add R̂ gate: if any parameter's R̂ > 1.01, return {"passed": False, "reason": "non-convergence"}.
Compute Bayes factors via ArviZ LOO-CV or bridge sampling.


Falsification Protocol 11 — Liquid Network Dynamics Echo State (Falsification-LiquidNetworkDynamics-EchoState.py)
Rating: 61/100
What is correctly implemented (from GUI):

LiquidNetworkDynamicsAnalyzer class
Spectral radius, leak rate, n_units parameterization

Critical deficiencies:

HIGH — Echo state network implementation: spectral radius < 1.0 is required for the echo state property — the GUI allows values up to 2.0 without a guard, which would produce an unstable reservoir.
HIGH — F6.1–F6.4 liquid-specific criteria (transition time, integration window, fading memory decay τ=1–3s, bifurcation structure) must all be tested against the ESN dynamics — it is unclear if all four are implemented.
MEDIUM — F6_5_HYSTERESIS_MIN = 0.05 and _MAX = 0.30 in constants but paper spec and inline validation in F1F2.py say 0.08–0.25. Both boundary values are wrong.

TODO:

Add guard: spectral_radius = min(spectral_radius, 0.98) before reservoir initialization, or raise ValueError for values ≥ 1.0.
Implement F6.3 (sparsity), F6.4 (fading memory: fit exponential decay to impulse response, assert τ ∈ [1.0, 3.0]s), F6.5 (bifurcation sweep on ESN input gain).
Fix F6_5_HYSTERESIS_MIN = 0.08 and F6_5_HYSTERESIS_MAX = 0.25.


Falsification Protocol 12 — Framework-Level Aggregator (APGI-Falsification-Aggregator.py)
Rating: 38/100
This is the single most critical gap in the entire system.
What is correctly implemented:

NAMED_PREDICTIONS dict with all 14 predictions correctly specified
FRAMEWORK_FALSIFICATION_THRESHOLD_A and ALTERNATIVE_FRAMEWORK_PARSIMONY_THRESHOLD constants
Correct logic for check_framework_falsification_condition_a() (zero-predictions-pass check)
check_framework_falsification_condition_b() stub with correct docstring

Critical deficiencies:

CRITICAL — aggregate_prediction_results() is a literal stub (...). The function body is ... — Python's ellipsis. It is completely unimplemented. This means the aggregator can never be called in practice.
CRITICAL — check_framework_falsification_condition_b() has no body. The docstring describes the logic but the function contains no code — it will return None.
CRITICAL — No JSON result file loading, no result tallying, no mapping of protocol outputs to named predictions. The entire framework-level falsification pipeline is structurally declared but computationally empty.
HIGH — Condition (a) logic is correct (passing == 0) but FRAMEWORK_FALSIFICATION_THRESHOLD_A = len(NAMED_PREDICTIONS) = 14 is set but never used in the condition check — the variable exists but the check doesn't reference it, making the constant meaningless.
HIGH — GNWT and IIT prediction dicts are required as parameters to both condition functions but no code anywhere generates these dicts — comparative framework analysis is entirely absent.

TODO:

Implement aggregate_prediction_results(result_files):

python   import json
   def aggregate_prediction_results(result_files):
       tallies = {k: {"passed": False, "evidence": []} for k in NAMED_PREDICTIONS}
       for path in result_files:
           with open(path) as f:
               data = json.load(f)
           for pred_id, result in data.get("named_predictions", {}).items():
               if pred_id in tallies:
                   tallies[pred_id]["passed"] |= result.get("passed", False)
                   tallies[pred_id]["evidence"].append(path)
       return tallies

Implement check_framework_falsification_condition_b():

python   def check_framework_falsification_condition_b(apgi_predictions, gnwt_predictions, iit_predictions):
       apgi_passing = {k for k, v in apgi_predictions.items() if v.get("passed")}
       for alt_preds in [gnwt_predictions, iit_predictions]:
           alt_passing = {k for k, v in alt_preds.items() if v.get("passed")}
           overlap = len(apgi_passing & alt_passing) / max(len(apgi_passing), 1)
           if overlap >= ALTERNATIVE_FRAMEWORK_PARSIMONY_THRESHOLD:
               return True  # APGI loses distinctiveness
       return False

Use FRAMEWORK_FALSIFICATION_THRESHOLD_A in condition (a): change the check to return passing < FRAMEWORK_FALSIFICATION_THRESHOLD_A — the framework is not falsified if any predictions pass (the constant captures the "all must fail" requirement).
Add a generate_gnwt_predictions() and generate_iit_predictions() function that applies the same 14 prediction tests using alternative-framework models.


# ISSUES

(1) fix falsification_thresholds.py constants, and (2) implement APGI-Falsification-Aggregator.py. These two changes unblock every other protocol's scoring and close the framework-level falsification gap that currently makes the system scientifically inert at the aggregation layer.

1. falsification_thresholds.py — The Canonical Source of Truth Is Wrong (Impact: ALL protocols)
The constants file is the single most impactful deficiency. A corrected version must set:
ConstantCurrent (Wrong)Correct (Paper)F5_1_MIN_PROPORTION0.700.75F5_1_MIN_ALPHA3.04.0F5_1_MIN_COHENS_D0.500.80F5_2_MIN_PROPORTION0.700.65F5_2_MIN_CORRELATION0.300.45F5_4_MIN_PEAK_SEPARATION0.203.0F5_5_PCA_MIN_VARIANCE0.600.70F5_5_MIN_LOADING0.400.60F5_6_MIN_PERFORMANCE_DIFF_PCT5.025.0 (or 40.0)F5_6_MIN_COHENS_D0.400.55F6_1_LTCN_MAX_TRANSITION_MS200.050.0F6_1_CLIFFS_DELTA_MIN0.300.60F6_1_MANN_WHITNEY_ALPHA0.050.01F6_2_LTCN_MIN_WINDOW_MS50.0200.0F6_2_MIN_INTEGRATION_RATIO1.504.0F6_2_MIN_CURVE_FIT_R20.800.85F6_2_WILCOXON_ALPHA0.050.01V12_1_MIN_P3B_REDUCTION_PCT10.080.0V12_1_MIN_IGNITION_REDUCTION_PCT15.070.0V12_1_MIN_COHENS_D0.400.80V12_1_MIN_ETA_SQUARED0.100.30V12_2_MIN_CORRELATION0.300.60V12_2_MIN_PILLAIS_TRACE0.150.40F6_5_HYSTERESIS_MIN0.050.08F6_5_HYSTERESIS_MAX0.300.25
2. Master_Validation.py — Equal Tier Weighting Contradicts Paper's Evidential Hierarchy
The master uses tier_weights = {"primary": 1.0, "secondary": 1.0, "tertiary": 1.0} with an inline comment defending this as preventing underweighting. However, the paper's epistemic roadmap explicitly assigns evidential priority: Level 1 predictions (metabolic/energy) are hardest to fake and should receive higher weight. The weighting should reflect the paper's 3-tier evidential hierarchy rather than treating all tiers equally.
3. Named Prediction → Protocol Routing Table Is Missing
The Aggregator's 14 NAMED_PREDICTIONS (P1.1, P1.2, P1.3, P2.a–c, P3.conv, P3.bic, P4.a–d, P5.a–b) are not connected to any protocol's run_validation() output by a formal routing table. Add a dict:
pythonPREDICTION_TO_PROTOCOL = {
    "P1.1": "Validation-Protocol-8",
    "P1.2": "Validation-Protocol-8",
    "P1.3": "Validation-Protocol-8",
    "P2.a": "Validation-Protocol-10",
    "P2.b": "Validation-Protocol-10",
    "P2.c": "Validation-Protocol-10",
    "P3.conv": "Validation-Protocol-3",
    "P3.bic": "Validation-Protocol-3",
    "P4.a": "Falsification-Protocol-9",
    "P4.b": "Falsification-Protocol-9",
    "P4.c": "Falsification-Protocol-9",
    "P4.d": "Falsification-Protocol-9",
    "P5.a": "Validation-Protocol-10",
    "P5.b": "Validation-Protocol-10",
}
This routing table then drives aggregate_prediction_results().
4. Unit Test Suite for Threshold Consistency Is Absent
Add tests/test_threshold_consistency.py that imports both falsification_thresholds.py and the inline-validated constants from Falsification-ActiveInferenceAgents-F1F2.py and asserts equality for every shared constant. This prevents future regressions.

Overall Ratings Summary
ProtocolScorePrimary GapValidation 1 (SyntheticEEG)62/100McNemar stub, threshold constants wrongValidation 2 (Behavioral)55/100Near-stub implementationValidation 3 (Agent Comparison)72/100Missing BIC, convergence aggregation biasValidation 4 (Phase Transition)68/100Level 1 stubs, P6 threshold invertedValidation 5 (Evolutionary)74/100Threshold constants systematically relaxedValidation 6 (Network)71/100F6.1/F6.2 constants 4× offValidation 7 (Math Consistency)66/100Tolerance 10× loose, equations not enumeratedValidation 8 (Parameter Sensitivity)70/100β-blockade model fabricated, threshold mismatchesValidation 9 (Neural Signatures)73/100F6.1 value = 80ms not 50msValidation 10 (Causal/TMS)58/100Named prediction P2.a–c not returnedValidation 11 (Bayesian)48/100Near-stub, MCMC not implementedValidation 12 (Clinical/Cross-Species)67/100V12.1 threshold 8× too smallFalsification 1 (Active Inference)78/100Ignition detection incompleteFalsification 2 (Agent Comparison)72/100Type inconsistency, no log-rank testFalsification 3 (Framework Multi)65/100Routing collision with validationFalsification 4 (Info-Theoretic)63/100Level 1 fully unimplementedFalsification 5 (Evolutionary)69/100Bifurcation values hardcodedFalsification 6 (Network Energy)70/100Same as Validation 6Falsification 7 (Math Consistency)64/100Tolerance, equation coverageFalsification 8 (Parameter Sensitivity)67/100n_samples not power-of-2, no identifiabilityFalsification 9 (Neural Sig. P3b/HEP)66/100P4.a–P4.d not wired to AggregatorFalsification 10 (Bayesian MCMC)55/100Sampler unimplementedFalsification 11 (Liquid ESN)61/100Spectral radius guard missingFalsification 12 (Aggregator)38/100Entire body is ... stubs
