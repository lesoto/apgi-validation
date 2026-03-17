# VALIDATION PROTOCOLS

## SyntheticEEG_MLClassification.py

VP-1: "Synthetic Neural Data Generation and Machine Learning Classification"
4,360 lines

Diagnostic:

CRITICAL: The paper's Protocol 1 is Interoceptive Precision Modulates Detection Threshold — a psychophysics paradigm with human participants using heartbeat discrimination and near-threshold visual stimuli. VP-1 instead implements synthetic EEG generation + deep learning classification. This is a complete protocol mapping mismatch. VP-1 does not test what Paper Protocol 1 specifies.
HIGH: The APGIDynamicalSystem implements simulate_surprise_accumulation correctly using the Π·|ε| > θₜ ignition equation, but the threshold adaptation equation θₜ₊₁ = θₜ + η(C_metabolic - V_information) is absent from the core generator.
HIGH: The FalsificationChecker references F1_1 through F6_2 but these are internally defined rather than mapped to paper predictions P1.1–P1.3.
MEDIUM: Comparison agents (StandardPP, GWTOnly, ContinuousIntegration) are implemented and distinguish themselves correctly. The multi-modal fusion network architecture is sound.
LOW: The APGISyntheticSignalGenerator generates plausible P3b, HEP, and gamma waveforms, but the parameter ranges don't match the paper's cited empirical effect sizes (d=0.40–0.60).


TODO:
Rename and reframe as a supplementary computational tool, not Protocol 1
Implement the actual Protocol 1 psychophysics paradigm: heartbeat discrimination → detection threshold task
Add the Garfinkel et al. (2015) SD-split group comparison (>1 SD vs <1 SD)
Implement the arousal interaction design (HR 100–120 bpm via exercise)
Implement the P1.1–P1.3 sub-predictions with the exact effect size tests (Cohen's d = 0.40–0.60)
Add the Bonferroni-corrected significance threshold (already defined but not used in threshold generator)
Ensure the threshold adaptation equation appears in the ignition trigger logic


## APGI_BayesianModelComparison_ParameterRecovery.py

VP-2: "Bayesian Model Comparison on Existing Consciousness Datasets"

Diagnostic:

CRITICAL: Paper Protocol 2 is the TMS Causal Intervention protocol — testing that anterior insula TMS reduces both PCI (by ~20%) and HEP (by ~30%), and that dlPFC TMS reduces PCI but not HEP. VP-2 implements Bayesian model comparison using PyMC3, which is more relevant to the computational validation tier, not the TMS causal tier.
HIGH: APGIGenerativeModel.build_model() correctly implements the Bayesian APGI model with Πⁱ, πᵉ, and the threshold ignition as a PyMC3 model. The bridge sampling approach for model evidence is architecturally correct.
HIGH: The validate_parameter_recovery function and analyze_beta_pi_identifiability directly address the paper's identifiability challenge (β/Πⁱ collinearity). This is excellent and paper-aligned, but belongs in a parameter estimation section, not Protocol 2.
MEDIUM: SyntheticConsciousnessDataGenerator generates Melloni-style and Canales-Johnson-style datasets, which is methodologically sound.
LOW: The BIC/WAIC comparison structure is good; BF thresholds (BF > 10) are consistent with Bayesian convention.

TODO:
Reframe as a computational supplement; create a separate VP implementing Paper Protocol 2 (TMS causal)
Add the P2a prediction: TMS delivered to dlPFC during near-threshold trials shifts detection threshold by >0.1 log units
Add the P2b critical test: insula TMS reduces HEP (30%) and PCI (20%) with double dissociation from dlPFC
Add the P2c interaction: high baseline IA subjects show stronger TMS effects
Implement the neuronavigation confirmation control
Add safety/ethics protocol stubs for TMS screening criteria
Implement full Fisher Information Matrix analysis for parameter identifiability (currently partial)

## ActiveInference_AgentSimulations_Protocol3.py

VP-3: "Active Inference Agent Simulations"
3,216 lines 

Diagnostic:

HIGH: This is the best-aligned VP — it directly implements Paper Protocol 3. HierarchicalGenerativeModel, SomaticMarkerNetwork, PolicyNetwork correctly implement the 3-level hierarchy with separate interoceptive/exteroceptive generative models and precision-weighting.
HIGH: Iowa Gambling Task, VolatileForaging, and ThreatReward environments are present and match the three task environments specified in the paper.
MEDIUM: The APGI-full agent convergence benchmark target (50–80 trials) is present but the specific comparison agents' convergence baselines (Actor-Critic ~100, Pure PE ~150, GNWT-only ~100–120) need explicit verification checks, not just logged output.
MEDIUM: BIC model comparison is mentioned but not fully implemented; cross-validation on held-out trials is missing.
LOW: The threshold adaptation equation θₜ₊₁ = θₜ + η(C_metabolic - V_information) appears in the agent update loop — correct. Parameter sensitivity across (α, β, θ_baseline) ranges needs systematic grid-search output.

TODO:

Add formal BIC computation for each agent type against human Iowa Gambling Task data
Implement systematic cross-validation (k-fold) on trial-level predictions
Add explicit convergence trial count comparisons with statistical tests (not just print statements)
Implement sensitivity analysis grid across α ∈ [3, 10], β ∈ [0.6, 2.2], θ_baseline ∈ [0.3, 0.8]
Add the falsification condition: if APGI shows <5% performance advantage, flag as falsified
Add the "ignition uncorrelated with adaptive behavior" falsification check (p > 0.3 criterion)

## InformationTheoretic_PhaseTransition_Level2.py

VP-4: "Information-Theoretic Phase Transition Analysis"
4,025 lines 

Diagnostic:

HIGH: This protocol has no direct mapping to a numbered paper protocol — it is a computational elaboration of information-theoretic claims from the Level 2 falsification tier. The InformationTheoreticAnalysis class (transfer entropy, integrated information, mutual information, entropy rate) is technically solid.
HIGH: PhaseTransitionDetector.detect_discontinuity() and compute_susceptibility() implement the phase transition detection machinery correctly for the Π·|ε| > θₜ threshold.
MEDIUM: detect_critical_slowing() (Hurst exponent, autocorrelation) is present but the critical slowing down criterion is not explicitly connected to the APGI ignition threshold crossing.
MEDIUM: FiniteSizeScalingAnalysis is theoretically sound for the phase transition claim but the paper does not specify this test — it is an enhancement.
LOW: ClinicalPowerAnalysis.analyze_clinical_group_power() with AUC targets (0.75–0.85) aligns with P4a from the paper.

TODO:

Explicitly label this as a Level 2 (information-theoretic) falsification test per the epistemic paper
Connect critical slowing detection output to the phase boundary equation from the paper
Add the P5 mutual information increase check (>30%) as a primary validated prediction
Add the P6 bandwidth asymptote check (~40 bits/s ceiling) with falsification threshold (>100 bits/s)
Implement the formal bridge principle from Level 3 (neural) to Level 2 (information-theoretic) with explicit justification text
Add the integrated information comparison against IIT's Φ metric explicitly

## EvolutionaryEmergence_AnalyticalValidation.py

VP-5: "Evolutionary Emergence of APGI-like Architectures"
3,471 lines 

Diagnostic:

CRITICAL: No numbered paper protocol specifies evolutionary simulation as a validation method. This is a theoretical elaboration of the evolutionary plausibility standard (Standard 6 from the epistemic paper), not a primary empirical protocol. While scientifically interesting, it is misclassified as a core validation protocol.
HIGH: AnalyticalAPGISolutions with steady-state, ignition time, phase boundary, and critical point calculations is mathematically rigorous and correctly derives from the dynamical equations. This is the most technically precise section.
HIGH: AgentGenome with mutation and crossover operators is well-implemented. The EvolvableAgent with the APGI ignition equation is correctly structured.
MEDIUM: The analytical validation test cases in AnalyticalValidationTestCases are high-quality unit tests for the mathematical model.
LOW: The three task environments (Iowa Gambling, Volatile Foraging, Threat-Reward) duplicate VP-3 environments.

TODO:

Reframe as "Evolutionary Plausibility + Mathematical Validation Supplement"
Add explicit Standard 6 (evolutionary plausibility) compliance scoring per the epistemic paper rubric
Eliminate environment duplication — import from VP-3/FP-2 instead
Add quantitative fitness landscape visualization: show APGI-like architectures occupy fitness peaks
Add a formal theorem/proof structure for the analytical solutions (currently just functions)
Connect evolutionary fitness advantage to the metabolic efficiency claim (≥20% energy saving)

## NeuralNetwork_InductiveBias_ComputationalBenchmark.py

VP-6: "Recurrent Neural Network Architectures with APGI Inductive Biases"
3,071 lines 

Diagnostic:

HIGH: No paper protocol maps to RNN architectures. This is an engineering supplement inspired by the computational validation tier. The APGIInspiredNetwork with threshold gating, precision-weighting, and somatic marker integration is architecturally faithful.
HIGH: Comparison networks (MLP, LSTM, Attention) are properly implemented for performance benchmarking.
MEDIUM: Three specialized datasets (ConsciousClassificationDataset, MaskingThresholdDataset, AttentionalBlinkDataset) map loosely to consciousness paradigms but are synthetic without grounding in empirical parameter ranges from the paper.
MEDIUM: The attentional blink dataset is particularly relevant to the ignition threshold dynamics but the T1-T2 lag-dependent suppression curve isn't formally tested against the APGI threshold equation.
LOW: InteroceptiveAccuracyDataset has the right feature set but the precision-accuracy relationship lacks the paper's beta coefficient range.

TODO:

Explicitly frame as Program 4 (Computational Architecture Energy Comparison) from the epistemic paper
Implement the ATP spike-cost budget (~10⁹ ATP/spike from Attwell & Laughlin) for energy comparison
Add the ≥20% energy-per-correct-detection falsification criterion
Ground AttentionalBlinkDataset T1-T2 parameters in empirical attentional blink literature values
Add formal BIC/AIC model comparison output (not just accuracy)
Implement the Brian2/NEST-style spike count logging for energy measurement

## TMS_Pharmacological_CausalIntervention_Protocol2.py

VP-7: "TMS/Pharmacological Intervention Predictions"
2,730 lines 

Diagnostic:

HIGH: This correctly implements Paper Protocol 2 (TMS causal intervention). TMSInterventions with dlPFC, insula, V1, and vertex conditions matches the paper's P2a–P2c predictions.
HIGH: PharmacologicalInterventions with propranolol, methylphenidate, ketamine, and placebo conditions maps to the paper's neuromodulator manipulation predictions.
MEDIUM: The InterventionStudySimulator.simulate_crossover_study() correctly implements the crossover design but the insula TMS effect size (HEP reduction 30%, PCI reduction 20%) is referenced but not enforced as a falsification criterion in the checker.
MEDIUM: PsychometricCurve fitting is sound but the 3-parameter logistic (threshold, slope, lapse) should also include a γ (guess rate) parameter for near-threshold paradigms.
LOW: Power analysis via PowerAnalysis.compute_required_n() is present but doesn't match the specific n=30 per group target from Paper Protocol 2.

TODO:

Add the double-dissociation falsification check: if insula TMS reduces PCI but not HEP → falsified
Fix psychometric curve to 4-parameter (add γ guess rate)
Enforce n=30 per group as the minimum power-adequate sample
Add the P2c interaction check: high IA × insula TMS interaction (strongest effect for high-Πⁱ individuals)
Add vertex TMS control as a formal falsification guard (vertex TMS should not affect threshold)
Connect TMS effect sizes to the threshold adaptation equation output

## Psychophysical_ThresholdEstimation_Protocol1.py

VP-8: "Psychophysical Threshold Estimation & Individual Differences"
2,229 lines 

Diagnostic:

HIGH: This is the best match for Paper Protocol 1. PsiMethod (adaptive psychophysics), APGIPsychophysicalEstimator, and heartbeat detection integration are all paper-aligned.
HIGH: The Π·|ε| > θₜ detection threshold model is correctly implemented in estimate_apgi_parameters.
MEDIUM: The arousal interaction (HR 100–120 bpm from exercise) is not implemented — this is a named sub-prediction (P1.2) in the paper that is missing.
MEDIUM: The Cohen's d threshold checks (0.40–0.60 range) reference paper values but use a single-point threshold rather than the paper's range test.
LOW: ICC (intraclass correlation) for test-retest reliability is implemented — this is appropriate for the parameter estimation protocol.

TODO:

Add the exercise arousal condition (HR 100–120 bpm) as a formal experimental arm
Implement P1.2: arousal interaction test (Cohen's d = 0.25–0.45 for interaction term)
Add P1.3: high-IA individuals show greater arousal benefit (specific contrast test)
Implement the Garfinkel et al. (2015) SD-split criterion (>1 SD vs <1 SD on heartbeat discrimination)
Add the Khalsa et al. (2018) meta-analytic benchmark (r = 0.43 as criterion validity check)
Implement the β/Πⁱ disambiguation protocol: pharmacological β-blockade arm

## ConvergentNeuralSignatures_Priority1_EmpiricalRoadmap.py

VP-9: "Convergent Neural Signatures"
2,461 lines | Rating: 73/100
Diagnostic:

HIGH: Maps correctly to Priority 1 of the Empirical Credibility Roadmap. APGIP3bAnalyzer.fit_sigmoidal_apgi_model() implements the key prediction that P3b scales sigmoidally (not linearly) with Π × |ε|.
HIGH: APGIFMRIAnalyzer.analyze_frontoparietal_coactivation() correctly tests the contingency prediction: frontoparietal BOLD only when S(t) > θₜ.
MEDIUM: `_analyze_theta_gamma_coupling()` implements PAC analysis for theta-gamma coupling at threshold crossing — but the paper specifies this as an ECoG/LFP prediction (not EEG), which is a measurement validity concern.
MEDIUM: `check_P5_mutual_information()` and `check_P6_bandwidth_constraint()` are present and implement the Level 2 predictions correctly.
LOW: The `_calculate_validation_score()` aggregation is arbitrary weighting; the epistemic paper's scoring rubric (0–5 per standard) should be used.

TODO:

Clarify the theta-gamma PAC prediction as requiring intracranial recordings
Replace arbitrary validation scoring with the epistemic paper's 5-standard rubric
Add the subthreshold activation falsification check: sensory cortex activates, frontoparietal does NOT (AUC < 0.6 criterion for classification)
Implement real MNE pipeline hooks rather than stub loaders (currently load_eeg_data is a stub)
Add the P3b latency criterion: ignition should occur within 200–400ms post-stimulus
Connect sigmoid fit parameters (α, θ) back to the APGI parameter space

## CausalManipulations_TMS_Pharmacological_Priority2.py

VP-10: "Causal Manipulations"
1,385 lines 

Diagnostic:

CRITICAL: At 1,385 lines, this is the shortest validation protocol — yet it covers Priority 2 of the roadmap which is arguably the most important causal tier. Significant underdevelopment compared to peer protocols.
HIGH: TMSIntervention, TACSIntervention, PharmacologicalIntervention, and MetabolicIntervention classes are correctly structured. The metabolic intervention (glucose/fasting) is a novel addition not in other protocols.
HIGH: CausalManipulationsValidator._validate_tms_ignition_disruption() correctly tests the causal directionality claim but uses simulated data only — no hooks for real EEG input.
MEDIUM: `_validate_erp_invariance()` checks that early ERP components (N1/P1) are unaffected by TMS while P3b is disrupted — this is the key dissociation test and is correctly framed.
LOW: `_validate_tacs_effects()` is implemented but tACS is not specified in the paper's core protocols — this is an enhancement.

TODO:

Expand to at least 2,500 lines matching peer protocols in depth
Add cold pressor test and breathlessness induction from Paper Protocol 4c (MCS interventions)
Add the P2b double-dissociation statistical test (χ² or mixed ANOVA)
Implement the neuronavigation MRI-guided targeting confirmation
Add adverse event stopping criteria stubs (clinically appropriate)
Add the Ethics/DSMB monitoring requirement as a documented protocol gate
Implement real EEG data input hooks using MNE compatibility layer

## QuantitativeModelFits_SpikingLNN_Priority3.py

VP-11: "Quantitative Model Fits"
2,324 lines | Rating: 75/100
Diagnostic:

HIGH: PsychometricFunctionFitter correctly compares APGI sigmoid, GNW equivalent, and linear models with AIC/BIC — this directly implements the model comparison prediction.
HIGH: SpikingLNNModel with liquid time-constant neurons implementing Π-gated threshold dynamics is architecturally correct and unique among the protocols.
MEDIUM: BayesianParameterEstimator correctly uses MCMC but the prior distributions are not justified against the paper's parameter ranges (Πⁱ ∈ [0.1, 2.0], β ∈ [0.6, 2.2]).
MEDIUM: `_validate_consciousness_paradigms()` tests masking and attentional blink but the specific trial counts and effect sizes from the paper are not enforced.
LOW: The `_calculate_quantitative_score()` provides a composite score but without the 5-standard epistemic rubric weighting.

TODO:

Align prior distributions with paper-specified parameter ranges explicitly
Add Fisher Information Matrix analysis for parameter identifiability (β/Πⁱ collinearity test)
Implement the paper's convergence benchmark comparison: APGI < 80 trials vs. Actor-Critic ~100 trials
Add hierarchical Bayesian pooling across subjects (currently single-subject estimation)
Expand spiking LNN to include liquid time-constant dynamics (τᵢ as learnable per neuron)
Add the formal model comparison table format from the paper (APGI vs. StandardPP vs. GWTonly vs. Continuous)

## Clinical_CrossSpecies_Convergence_Protocol4.py

VP-12: "Clinical and Cross-Species Convergence"
2,240 lines 

Diagnostic:

HIGH: ClinicalDataAnalyzer with VS/UWS, MCS, EMCS, and control groups directly implements Paper Protocol 4. PCI and HEP as joint predictors are correctly operationalized.
HIGH: PsychiatricProfileAnalyzer with anxiety, depression, PTSD, and autism profiles is paper-aligned for the APGI-CAB (Clinical Assessment Battery) roadmap item.
HIGH: CrossSpeciesHomologyAnalyzer with rat, macaque, and human data implements Standard 6 (evolutionary plausibility) and cross-species scaling.
MEDIUM: IITConvergenceAnalyzer comparing Φ with APGI parameters is a novel addition not in the core paper but appropriate for the alternative comparison standard.
MEDIUM: The P4d longitudinal prediction (6-month outcome, ΔR² ≈ 0.10–0.15) is not explicitly implemented — only the cross-sectional P4a is tested.

TODO:

Add P4d longitudinal model: baseline (PCI + HEP) → 6-month CRS-R outcome regression
Implement the target ΔR² ≈ 0.10–0.15 improvement test against CRS-R + structural imaging baseline
Add the autonomic perturbation intervention (cold pressor, breathlessness) with the specific temporal parameters (pre, +30s, +5min, +30min)
Add the sham control comparison (tactile vs. cold pressor, auditory vs. breathlessness)
Expand psychiatric profiles to include β parameter abnormality signatures (panic disorder β ≈ 1.5–2.2)
Add the power analysis for the clinical protocol (n=30 per group minimum)

FALSIFICATION PROTOCOLS

## Falsification_ActiveInferenceAgents_F1F2.py

FP-1: Active Inference Agent Falsification
3,114 lines 

Diagnostic:

HIGH: Implements F1.1–F1.6 and F2.1–F5.x falsification criteria with explicit threshold registries. The FalsificationChecker architecture is the strongest feature.
HIGH: HierarchicalGenerativeModel and SomaticMarkerNetwork correctly implement the APGI dynamics. The ThresholdRegistry with configurable thresholds is excellent engineering.
MEDIUM: The dependency on utils.shared_falsification and falsification_thresholds modules creates coupling without those modules being in-project — runtime failure risk.
MEDIUM: F5 family (precision-gating family) falsification is imported rather than implemented — reduces auditability.
LOW: The bootstrap CI functions at the top are correct but use a non-standard one-sample bootstrap (should use pivotal bootstrap for mean comparisons).


TODO:

Bundle falsification_thresholds constants inline (or include the file in the project)
Implement F5 family directly rather than importing from shared module
Replace one-sample bootstrap with pivotal bootstrap method
Add the cumulative reward advantage threshold test against the paper's 18-trial benchmark
Add the "ignition uncorrelated with behavior" falsification (p > 0.3 criterion from paper)
Validate all threshold constants against paper-specified values explicitly

## Falsification_AgentComparison_ConvergenceBenchmark.py

FP-2: Iowa Gambling Task / Agent Comparison Falsification
1,768 lines | Rating: 72/100
Diagnostic:

HIGH: IowaGamblingTaskEnvironment, VolatileForagingEnvironment, and ThreatRewardTradeoffEnvironment are well-implemented and paper-aligned.
MEDIUM: StandardPPAgent, GWTOnlyAgent, and StandardActorCriticAgent comparison agents are correctly structured but their internal dynamics differ subtly from VP-3 implementations — creating inconsistency risk between validation and falsification protocols.
MEDIUM: The convergence trial count falsification (APGI must beat 80 trials) is not explicitly checked — only performance is.
LOW: Duplicates environment implementations from FP-1 and VP-3.

TODO:

Import environments from VP-3 rather than reimplementing (single source of truth)
Add explicit convergence trial count falsification criterion
Add the BIC comparison falsification: if Pure PP achieves lower BIC → falsified
Align agent implementations precisely with VP-3 counterparts
Add sensitivity analysis across the full parameter range simultaneously

## Falsification_FrameworkLevel_MultiProtocol.py

FP-3: Multi-Protocol Integration Falsification
2,879 lines

Diagnostic:

CRITICAL: FP-3 uses runtime dynamic imports (importlib.util, _get_protocol1(), _get_protocol2()) which is fragile architecture. If VP-1 or VP-2 are not present at runtime, the protocol silently falls back to stubs.
HIGH: The integration of multiple protocols into a unified falsification decision is conceptually correct (framework-level falsification requires all 14+ predictions to fail).
MEDIUM: StandardPPAgent, GWTOnlyAgent, StandardActorCriticAgent are reimplemented a third time — identical to FP-2 and VP-3.
LOW: `_softmax` and `_standardize_observation` utility functions should be shared utilities.

TODO:

Replace dynamic imports with direct imports; add proper dependency documentation
Eliminate agent reimplementation — import from shared module
Add the 14+ prediction tally counter: framework falsification triggers when fewer than X pass
Implement the framework-level falsification criterion: if all 14+ predictions fail, framework_falsified = True
Add the alternative framework parsimony check (condition b): can GNWT/IIT account for same predictions?
Add logging of which specific predictions pass/fail in the aggregate

## Falsification_InformationTheoretic_PhaseTransition.py

FP-4: Information-Theoretic Falsification
1,047 lines

Diagnostic:

HIGH: SurpriseIgnitionSystem.simulate() with the correct ODE dynamics (dS/dt = -S/τS + Π·|ε|) is the most accurate dynamical implementation across all protocols.
HIGH: InformationTheoreticAnalysis.detect_phase_transition() correctly identifies the discontinuity signature at threshold crossing.
MEDIUM: Transfer entropy and integrated information calculations are sound but computationally expensive without vectorization.
LOW: At 1,047 lines, this protocol is significantly shorter than its VP-4 counterpart (4,025 lines) — the falsification side is underspecified relative to validation.

TODO:

Add Level 2 falsification criteria explicitly: if transfer entropy < 0.1 bits → falsified
Add the integrated information comparison against null (shuffled) baseline
Implement vectorized transfer entropy for computational efficiency
Add the bandwidth falsification: if mutual information > 100 bits/s → falsified
Expand to include Level 1 falsification stubs (metabolic cost measurement protocols)
Double the implementation depth to match VP-4 coverage

## Falsification_EvolutionaryPlausibility_Standard6.py

FP-5: Evolutionary Emergence Falsification
2,039 lines
Diagnostic:

HIGH: EvolvableAgent with the APGI ignition equation and ContinuousUpdateAgent comparison are correctly implemented.
MEDIUM: EvolutionaryAPGIEmergence tracks fitness landscape correctly but doesn't implement the formal fitness advantage quantification the paper requires.
LOW: Duplicates VP-5 evolutionary machinery with slight parameter variations.

TODO:

Add formal fitness advantage quantification (APGI vs continuous; ≥20% criterion)
Add Standard 6 compliance score (0–5 per criterion)
Eliminate duplication with VP-5; share genome/agent classes
Add convergent evolution test: do multiple independent runs converge to similar architectures?
Add the simplicity constraint: does a simpler architecture without precision-gating achieve similar fitness?

## Falsification_NeuralNetwork_EnergyBenchmark.py

FP-6: Neural Network Architecture Falsification
1,743 lines 

Diagnostic:

HIGH: APGIInspiredNetwork is faithfully reimplemented with threshold gating, matching VP-6.
MEDIUM: NetworkComparisonExperiment performs the comparison correctly but lacks the energy-per-correct-detection metric.
LOW: Duplicates VP-6 substantially. The falsification-specific logic (when does the network fail to outperform?) is underspecified.

TODO:

Add energy-per-correct-detection as primary falsification metric (≥20% advantage criterion)
Add the spike count / ATP budget calculation
Add explicit falsification condition: if MLP/LSTM achieves equivalent accuracy within 5% → falsified
Eliminate network architecture duplication from VP-6
Add statistical test for performance difference (not just point estimates)

## Falsification_MathematicalConsistency_Equations.py

FP-7: Mathematical Consistency Checks
368 lines | Rating: 63/100
Diagnostic:

CRITICAL: At 368 lines, this is severely underdeveloped. Mathematical consistency is a foundational falsification requirement (Standard III, mechanistic specificity).
HIGH: verify_dimensional_homogeneity(), verify_surprise_derivatives(), verify_asymptotic_behavior(), verify_jacobian_stability() are the right functions — but each is only ~20–40 lines.
HIGH: The Jacobian stability analysis (verify_jacobian_stability()) uses numerical differentiation rather than the analytical Jacobian — less rigorous.
MEDIUM: No tests for the threshold adaptation equation stability (θₜ₊₁ dynamics).
LOW: No tests for the effective interoceptive precision formula (Πⁱ_eff = Πⁱ_baseline · exp(β·M)).

TODO:

Expand to 1,500+ lines with comprehensive equation tests
Add analytical Jacobian computation and compare with numerical
Add stability analysis for θₜ₊₁ = θₜ + η(C_metabolic - V_information) dynamics
Add verification of Πⁱ_eff = Πⁱ_baseline · exp(β·M) formula with parameter bounds
Add unit tests for all 14+ paper predictions' mathematical form
Add dimensional analysis for all equation parameters (units consistency)
Add formal proof stubs: show parameter space bounds where ignition is guaranteed/impossible

## Falsification_ParameterSensitivity_Identifiability.py

FP-8: Parameter Sensitivity Analysis
345 lines 

Diagnostic:

CRITICAL: At 345 lines, this is the second-shortest protocol. Parameter sensitivity is critical for APGI's identifiability claims.
HIGH: analyze_oat_sensitivity() (One-At-a-Time) and analyze_sobol_sensitivity() are the right methods but Sobol implementation is a placeholder — it doesn't actually compute Sobol indices.
HIGH: The OAT analysis function signature references simulate_model_performance_with_agent which imports from a non-bundled module.

TODO:

Implement full Sobol sensitivity indices using SALib or manual implementation
Add the β/Πⁱ collinearity sensitivity test explicitly
Add parameter recovery analysis (simulate data → estimate parameters → compare)
Add the Fisher Information Matrix analysis for identifiability
Expand to 1,500+ lines with systematic parameter space exploration
Add falsification criterion: if Sobol total index for any core parameter < 0.05 → parameter is redundant

## Falsification_NeuralSignatures_EEG_P3b_HEP.py

FP-9: Neural Signatures Validation
410 lines 

Diagnostic:

CRITICAL: At 410 lines, severely underdeveloped for what should be the core neural falsification protocol.
HIGH: detect_gamma_oscillation(), detect_theta_gamma_pac(), detect_p3_amplitude() are correctly framed but are very thin wrappers.
HIGH: P3b falsification threshold (< 0.3 µV or non-significant) is correct but the range (0.40–0.60 Cohen's d) from the paper is not enforced.
MEDIUM: validate_consciousness_markers() provides a framework-level pass/fail but the individual prediction mapping (P1.1, P2.b, etc.) is absent.

TODO:

Expand to 2,000+ lines with comprehensive EEG analysis functions
Add HEP amplitude falsification (< 0.2 µV criterion)
Add the double-dissociation test for TMS: insula vs dlPFC differential effects
Connect each function output to a specific named paper prediction
Add real MNE pipeline compatibility
Add frequency-specific power analysis (gamma 30–80Hz, theta 4–8Hz) with paper-specified bands


## Falsification_CrossSpeciesScaling_P12.py

FP-10: Cross-Species Scaling
237 lines
Diagnostic:

CRITICAL: At 237 lines, this is the shortest and most underdeveloped file in the entire codebase. Cross-species scaling (Standard 6, Level 1 P12) is a genuine scientific contribution that deserves full treatment.
HIGH: apply_cross_species_scaling() and calculate_allometric_exponent() are mathematically correct but trivially thin.
HIGH: No actual species data, no brain mass scaling, no reference to the allometric relationships the paper implies.

TODO:

Expand from 237 to 1,500+ lines — this is critically incomplete
Add actual species data: rat (1.5g brain), macaque (70g), human (1,350g), dolphin (~1,500g), elephant (~4,780g)
Implement the brain mass allometric scaling for Πⁱ, θₜ, and τS parameters
Add the phylogenetic conservation test: is the precision-gating mechanism homologous?
Add falsification criterion: P12 is falsified if allometric exponent deviates >2 SD from expected
Connect to the evolutionary plausibility claim with quantitative cross-species benchmarks

## Falsification_BayesianEstimation_ParameterRecovery.py

FP-11: Bayesian Estimation
465 lines 

Diagnostic:

HIGH: Both NUTS (run_bayesian_estimation_nuts()) and Metropolis-Hastings (run_bayesian_estimation_mh()) samplers are implemented.
HIGH: compute_posterior_distributions() and calculate_bayesian_factor() are correctly framed.
MEDIUM: The NUTS implementation uses PyMC3 but the priors are not aligned with paper-specified parameter ranges.
MEDIUM: get_falsification_criteria() is implemented but the criteria don't map to specific named paper predictions.

TODO:

Align all priors with paper-specified ranges: Πⁱ ~ HalfNormal(1.0), β ~ Normal(1.15, 0.3), θ₀ ~ Normal(0.5, 0.1)
Add convergence diagnostics (R-hat, effective sample size)
Add the formal identifiability test for β/Πⁱ collinearity
Map each Bayesian factor to a specific paper prediction
Add the hierarchical model version (pooling across subjects)
Add calibration checks: does posterior coverage match credible interval nominal rates?

## Falsification_LiquidNetworkDynamics_EchoState.py

FP-12: Liquid Network Validation
466 lines 

Diagnostic:

HIGH: test_echo_state_property(), test_fading_memory(), test_non_linearity(), test_separation_capacity() are the correct reservoir computing validation tests for liquid time-constant networks.
MEDIUM: test_fading_memory() uses exponential decay fitting which is correct but the decay time constant is not connected to the paper's τ parameter ranges.
MEDIUM: validate_network_topology() and validate_connectivity_pattern() check structural properties but don't test the specific liquid network dynamics from Paper 2 (LNN substrate).
LOW: At 466 lines, this deserves 1,500+ lines given Paper 2's focus on liquid networks.

TODO:

Expand to 1,500+ lines
Connect fading memory time constant to APGI's τS (0.3–0.5s from paper)
Add the separation capacity falsification: if liquid network cannot separate conscious/unconscious trials → falsified
Add the echo state property test with the paper's connectivity density requirements
Add the liquid time-constant (LTC) neuron model explicitly (learnable τ per neuron)
Add the phase transition test: does the network exhibit critical dynamics near the ignition threshold?


# ISSUES

1. Protocol Mapping Mismatch — VP-1 and VP-2 do not implement the paper's Protocol 1 (psychophysics) and Protocol 2 (TMS), respectively. VP-7 and VP-8 actually implement those. A protocol numbering reference table should be added to a README.md documenting which file implements which paper section.
2. Systemic Code Duplication — The Iowa Gambling Task environment, agent classes (StandardPP, GWTOnly, ActorCritic), and bootstrap CI functions are reimplemented identically across 4–5 files each. A shared_environments.py and shared_agents.py module would eliminate this and reduce maintenance burden.
3. Missing Shared Dependencies — falsification_thresholds.py, utils/shared_falsification.py, utils/logging_config.py, utils/config_loader.py, and utils/statistical_tests.py are imported but not in the project. Every protocol fails at runtime without them. These must be bundled or made self-contained.
4. Falsification Protocols FP-7, FP-8, FP-9, FP-10 Are Critically Thin — Four of twelve falsification protocols are under 470 lines. Given the paper's emphasis on rigorous falsification criteria, these represent the most urgent development gap. FP-10 at 237 lines is particularly alarming for a cross-species claim that spans Level 1 (thermodynamic) analysis.
5. Framework-Level Falsification Counter Is Absent — The paper specifies that APGI is framework-falsified if all 14+ predictions fail. No protocol currently counts how many predictions have passed/failed across the suite and triggers a global framework-falsification flag. This is the most important single missing feature.
Finding 1: Core Equation Inconsistency Across Protocols — CRITICAL
The threshold adaptation equation exists in three distinct forms across the 24 files, which means the dynamical system is not canonically identical in all protocols:

Form A (Correct — FP-1, FP-1 agent):

```python
dtheta_dt = (theta_0 - theta_t) / tau_theta + eta_theta * (metabolic_cost - information_value)
theta_t += dtheta_dt * dt
```

This correctly implements the paper's θₜ₊₁ = θₜ + η(C_metabolic - V_information) with a restoring force toward baseline.

Form B (Incomplete — FP-4, FP-4 simulate loop):

```python
dtheta_dt = (theta_0 - theta_t) / tau_theta   # eta_theta term ABSENT
theta_t += dtheta_dt * dt
```

This omits the metabolic cost driver entirely. The threshold in FP-4 decays passively to baseline but never rises in response to metabolic load — the central claim of the APGI framework goes untested in this protocol.
Form C (Missing entirely — VP-1 through VP-6 data generators):
The synthetic data generators in VP-1 do not include threshold adaptation at all in their trial generation logic — only in the APGIDynamicalSystem.simulate_surprise_accumulation() method, which is not called during trial generation.
Remediation (applies to all 24 files):
Create a canonical APGIDynamics class in shared_dynamics.py with the complete, correct equations. Every protocol must import and use this single source of truth. The canonical class must implement:

dS/dt = -S/τS + Πᵉ·|εᵉ| + β·Πⁱ·|εⁱ|
dθ/dt = (θ₀ - θ)/τθ + η(C_metabolic - V_information)
Πⁱ_eff = Πⁱ_baseline · exp(β · M(c,a))
P(ignition) = σ(α · (S - θ))
Finding 2: The Πⁱ_eff Formula — Present in VP-5 Only
The signature equation of the APGI framework — Πⁱ_eff = Πⁱ_baseline · exp(β · M(c,a)) — appears in its correct mathematical form in only one file: Validation-Protocol-5.py (line 192: return Pi_i_baseline  *np.exp(beta* M)).
In all other protocols, the somatic marker modulation of interoceptive precision is implemented as a scalar multiplication, a neural network output, or simply omitted. Specifically:

FP-1 uses SomaticMarkerNetwork.predict() (a neural net output substituted for the exponential formula)
FP-2 through FP-6 use agent step functions without the explicit exponential term
VP-7, VP-8 parameterize β as a constant without the M(c,a) somatic marker context-action function

Remediation: The Πⁱ_eff formula must be explicitly implemented in every protocol that claims to test precision-gated access. The somatic marker M(c,a) must be a context-and-action-indexed lookup or neural approximator — not a fixed scalar — in all agent implementations.

Finding 3: Validation-Protocol-P4-Epistemic.py Is Circular and Ungrounded
This file validates predictions P5–P12 entirely through self-generated random data. For example:

P9 (metabolic cost): Generates random exponential variates with pre-set means (1.0 vs 1.2) and tests whether the 20% difference is significant. Since the data is generated with the answer embedded, this test will always pass with sufficient n. This is circular validation, not empirical testing.
P10 (energy efficiency): Same pattern — generates Normal(10, 1) vs Normal(12, 1) and computes advantage. Guaranteed to pass.
P8 (information erasure): Generates SOA curves with hardcoded break at 50ms — the falsification criterion is baked into the data generator.

Rating adjustment: VP-P4-Epistemic drops from an apparent pass rate to 42/100 on account of this structural circularity. The simulations are syntactically valid Python but scientifically self-confirming.
Suggested name: APGI_EpistemicPredictions_P5_P12_PLACEHOLDER.py (pending real data integration)
TODO: for VP-P4-Epistemic:

Replace all random data generation with real-data pipelines or at minimum parameter-blind simulations where the analyst does not control the generative parameters
For P9: implement the ATP spike-budget calculation from Attwell & Laughlin (2001) using spiking network output — not exponential variates
For P10: implement the ≥20% energy-per-correct-detection test against a matched continuous-processing baseline agent
For P6: implement an actual information-transmission rate measurement over a signal channel, not a hardcoded asymptote curve
For P7: implement a proper Bayes-optimal detector and compare APGI ignition probability against the theoretical optimal ROC — not a simulated comparison with preset means
For P8: implement a genuine backward masking simulation with a mask generator that does not embed the 50ms cutoff as a control parameter

Finding 5: DIM_CONSTANTS Usage Creates Dimension Mismatch Risk
DIM_CONSTANTS from utils/constants.py sets the observation space dimensions (EXTERO_DIM, INTERO_DIM, CONTEXT_DIM, etc.) used by agents across FP-1, FP-2, FP-3, FP-5, FP-6. Since this module is missing, each file falls back to hardcoded defaults that may differ. FP-1 falls back to inline constants while FP-3 uses dynamic imports. If these defaults are inconsistent, agents built in one protocol cannot be evaluated against environments built in another — silently producing incorrect cross-protocol results.
Remediation: Create utils/constants.py with:
pythonfrom dataclasses import dataclass

```python
@dataclass(frozen=True)
class _DimConstants:
    EXTERO_DIM: int = 10
    INTERO_DIM: int = 5
    SENSORY_DIM: int = 15
    OBJECTS_DIM: int = 8
    CONTEXT_DIM: int = 12
    VISCERAL_DIM: int = 4
    ORGAN_DIM: int = 3
    HOMEOSTATIC_DIM: int = 6

DIM_CONSTANTS = _DimConstants()
```

Then verify every agent's input dimension matches every environment's output dimension across all 24 files.

Finding 6: Framework-Level Falsification Counter Is Absent
The paper specifies two framework-level falsification conditions:

(a) All 14+ predictions fail to show significant relationships
(b) A competing framework accounts for all confirmed predictions with equal parsimony

No file currently implements a cross-protocol prediction counter. FP-3 is supposed to be the framework-level aggregator but its _aggregate_results() only aggregates within-environment agent results — it does not count pass/fail across the 14 named predictions (P1.1, P1.2, P1.3, P2.a, P2.b, P2.c, P3 benchmarks, P4a–P4d, P5a–P5d).
Remediation — create APGI_FrameworkFalsification_Aggregator.py:
python"""
APGI Framework-Level Falsification Aggregator
Implements conditions (a) and (b) from the framework falsification specification.
Requires all 24 protocol files to have run and produced JSON result files.
"""

NAMED_PREDICTIONS = {
    "P1.1": "Interoceptive precision modulates detection threshold (d=0.40–0.60)",
    "P1.2": "Arousal amplifies the Πⁱ–threshold relationship",
    "P1.3": "High-IA individuals show stronger arousal benefit",
    "P2.a": "dlPFC TMS shifts threshold >0.1 log units",
    "P2.b": "Insula TMS reduces HEP ~30% AND PCI ~20% (double dissociation)",
    "P2.c": "High-IA × insula TMS interaction",
    "P3.conv": "APGI converges in 50–80 trials (beats baselines)",
    "P3.bic": "APGI BIC lower than StandardPP and GWTonly",
    "P4.a": "PCI+HEP joint AUC > 0.80 for DoC classification",
    "P4.b": "DMN↔PCI r > 0.50; DMN↔HEP r < 0.20",
    "P4.c": "Cold pressor increases PCI >10% in MCS, not VS",
    "P4.d": "Baseline PCI+HEP predicts 6-month recovery ΔR² > 0.10",
    "P5.a": "vmPFC–SCR anticipatory correlation r > 0.40",
    "P5.b": "vmPFC uncorrelated with posterior insula (r < 0.20)",
}

FRAMEWORK_FALSIFICATION_THRESHOLD_A = len(NAMED_PREDICTIONS)  # All 14 must fail
ALTERNATIVE_FRAMEWORK_PARSIMONY_THRESHOLD = 0.90  # 90% of same predictions

def aggregate_prediction_results(result_files: list) -> dict:
    """Load JSON results from all protocols and tally prediction pass/fail."""
    ...

def check_framework_falsification_condition_a(prediction_results: dict) -> bool:
    """
    Condition (a): Framework falsified if ALL 14+ predictions fail.
    Returns True if framework is falsified.
    """
    passing = sum(1 for r in prediction_results.values() if r.get("passed"))
    return passing == 0  # Falsified only if zero predictions pass

def check_framework_falsification_condition_b(
    apgi_predictions: dict,
    gnwt_predictions: dict,
    iit_predictions: dict
) -> bool:
    """
    Condition (b): Framework loses distinctiveness if alternative
    accounts for same predictions with equal parsimony.
    """
    ...

```python

---

## CONSOLIDATED PATH TO 100/100 — PRIORITY ORDERING

### TIER 1: Infrastructure (Must Complete First — Blocks Everything Else)

**Step 1.1 — Create `falsification_thresholds.py`**
All 30+ threshold constants from the paper, documented with paper citations. Estimated: 150 lines.

**Step 1.2 — Create `utils/constants.py`**
`DIM_CONSTANTS` dataclass with all agent/environment dimension constants. Estimated: 40 lines.

**Step 1.3 — Create `utils/shared_falsification.py`**
`check_F5_family()` function with the complete F5.1–F5.6 checks. Estimated: 200 lines.

**Step 1.4 — Create `utils/shared_dynamics.py`**
Canonical `APGIDynamics` class implementing all four core equations with unit tests. Estimated: 300 lines.

**Step 1.5 — Create `utils/shared_environments.py`**
Single canonical implementations of `IowaGamblingTask`, `VolatileForagingEnvironment`, `ThreatRewardTradeoffEnvironment` used by VP-3, VP-5, FP-1, FP-2, FP-3, FP-5. Eliminates five-way duplication.

**Step 1.6 — Create `utils/shared_agents.py`**
Single canonical implementations of `StandardPPAgent`, `GWTOnlyAgent`, `StandardActorCriticAgent`. Eliminates four-way duplication.

**Step 1.7 — Create `utils/statistical_tests.py`**
`safe_ttest_1samp`, `compute_power_analysis`, `compute_required_n`, `pivotal_bootstrap_ci`. Estimated: 250 lines.

---

### TIER 2: Equation Canonicalization (Must Complete Second)

**Step 2.1 — Refactor FP-4 threshold adaptation**
Replace `dtheta_dt = (theta_0 - theta_t) / tau_theta` with the full Form A equation including `eta_theta * (C_metabolic - V_information)`.

**Step 2.2 — Add Πⁱ_eff formula to all agent step() methods**
FP-1, FP-2, FP-3, FP-5, FP-6 must all call `Pi_i_eff = Pi_i_baseline  *np.exp(beta* M_ca)` explicitly in their precision computation, not use a neural net surrogate.

**Step 2.3 — Remove circular data generation from VP-P4-Epistemic**
Replace self-confirming random generators with parameter-blind simulation or real-data hooks for P5–P12.

**Step 2.4 — Enforce paper-specified parameter ranges as assertions**
Every protocol that instantiates an APGI agent should assert: `0.6 <= beta <= 2.2`, `0.1 <= Pi_i <= 2.0`, `0.3 <= theta_0 <= 0.8`, `0.3 <= tau_S <= 0.5`.

---

### TIER 3: Protocol Mapping Correction (High Value)

**Step 3.1 — Rename protocols with a mapping index**
Create `PROTOCOL_REGISTRY.md`:

```markdown
Paper Protocol 1 (Psychophysics/Threshold) → VP-8
Paper Protocol 2 (TMS Causal)             → VP-7
Paper Protocol 3 (Active Inference Agents) → VP-3
Paper Protocol 4 (Disorders of Consciousness) → VP-12
Paper Protocol 5 (fMRI Anticipation/Experience) → [MISSING — must create]
Paper Protocol 6 (Parameter Estimation)   → VP-2 (partial)

Empirical Credibility Roadmap Priority 1  → VP-9
Empirical Credibility Roadmap Priority 2  → VP-10
Empirical Credibility Roadmap Priority 3  → VP-11
Empirical Credibility Roadmap Priority 4  → VP-12

Epistemic Paper P5–P8 (Level 2)           → VP-P4-Epistemic (needs rework)
Epistemic Paper P9–P12 (Level 1)          → VP-P4-Epistemic (needs real data)
```

Step 3.2 — Create the Missing Protocol: Paper Protocol 5 (fMRI Anticipation vs Experience)
No current file implements P5a–P5d (vmPFC/amygdala anticipatory correlation, posterior insula dissociation, PPI analysis, learning curves). This is a complete gap. Estimated: 2,000 lines.
Step 3.3 — Create the Missing Protocol: Paper Protocol 6 (Parameter Estimation)
No file implements the full parameter estimation protocol with double-dissociation pharmacological design. Estimated: 1,500 lines.

TIER 4: Underdeveloped Protocol Expansion
In order of urgency (shortest files first):
Step 4.1 — FP-10 (237 lines → 1,500 lines)
Add species data, allometric exponents, phylogenetic conservation test, P12 falsification criterion.
Step 4.2 — FP-8 (345 lines → 1,500 lines)
Implement true Sobol indices using SALib, add Fisher Information Matrix, add β/Πⁱ collinearity test.
Step 4.3 — FP-7 (368 lines → 1,500 lines)
Add analytical Jacobian, θₜ₊₁ stability analysis, Πⁱ_eff formula verification, dimensional analysis for all parameters.
Step 4.4 — FP-9 (410 lines → 2,000 lines)
Add HEP amplitude falsification, double-dissociation TMS test, per-prediction mapping, MNE pipeline hooks.
Step 4.5 — FP-12 (466 lines → 1,500 lines)
Connect fading memory τ to paper's τS, add LTC neuron model, add phase transition test, add separation capacity criterion.
Step 4.6 — VP-P4-Epistemic (644 lines → 2,000 lines, non-circular)
Rebuild all P5–P12 validators without self-confirming data generation.
Step 4.7 — VP-10 (1,385 lines → 2,500 lines)
Expand causal manipulations: add cold pressor, breathlessness, ethics stubs, MNE hooks, double-dissociation ANOVA.

TIER 5: Framework-Level Aggregation
Step 5.1 — Create APGI_FrameworkFalsification_Aggregator.py
Implement the 14-prediction counter, condition (a) and (b) falsification checks, and the competing framework parsimony comparison (GNWT, IIT). This is the capstone file that gives the project scientific completeness. Estimated: 800 lines.
Step 5.2 — Add JSON result export to every protocol
Every main() function must write results to results/{protocol_name}_results.json. The aggregator reads these files. Add a run_all_protocols.py orchestrator.
Step 5.3 — Add a README.md with protocol registry
Document which file implements which paper section, which predictions are tested, and what the falsification thresholds are.


# APGI Protocol Status Report

| File | Rating | Suggested Name |
| ---- | ------ | ---------------- |
| Validation-Protocol-1.py | 62/100 | SyntheticEEG_MLClassification.py |
| Validation-Protocol-2.py | 71/100 | BayesianModelComparison_ParameterRecovery.py |
| Validation-Protocol-3.py | 82/100 | ActiveInference_AgentSimulations_Protocol3.py |
| Validation-Protocol-4.py | 74/100 | InformationTheoretic_PhaseTransition_Level2.py |
| Validation-Protocol-5.py | 66/100 | EvolutionaryEmergence_AnalyticalValidation.py |
| Validation-Protocol-6.py | 68/100 | NeuralNetwork_InductiveBias_ComputationalBenchmark.py |
| Validation-Protocol-7.py | 79/100 | TMS_Pharmacological_CausalIntervention_Protocol2.py |
| Validation-Protocol-8.py | 77/100 | Psychophysical_ThresholdEstimation_Protocol1.py |
| Validation-Protocol-9.py | 73/100 | ConvergentNeuralSignatures_Priority1_EmpiricalRoadmap.py |
| Validation-Protocol-10.py | 70/100 | CausalManipulations_TMS_Pharmacological_Priority2.py |
| Validation-Protocol-11.py | 75/100 | QuantitativeModelFits_SpikingLNN_Priority3.py |
| Validation-Protocol-12.py | 76/100 | Clinical_CrossSpecies_Convergence_Protocol4.py |
| Falsification-Protocol-1.py | 74/100 | Falsification_ActiveInferenceAgents_F1F2.py |
| Falsification-Protocol-2.py | 72/100 | Falsification_AgentComparison_ConvergenceBenchmark.py |
| Falsification-Protocol-3.py | 65/100 | Falsification_FrameworkLevel_MultiProtocol.py |
| Falsification-Protocol-4.py | 73/100 | Falsification_InformationTheoretic_PhaseTransition.py |
| Falsification-Protocol-5.py | 70/100 | Falsification_EvolutionaryPlausibility_Standard6.py |
| Falsification-Protocol-6.py | 68/100 | Falsification_NeuralNetwork_EnergyBenchmark.py |
| Falsification-Protocol-7.py | 63/100 | Falsification_MathematicalConsistency_Equations.py |
| Falsification-Protocol-8.py | 60/100 | Falsification_ParameterSensitivity_Identifiability.py |
| Falsification-Protocol-9.py | 62/100 | Falsification_NeuralSignatures_EEG_P3b_HEP.py |
| Falsification-Protocol-10.py | 55/100 | Falsification_CrossSpeciesScaling_P12.py |
| Falsification-Protocol-11.py | 67/100 | Falsification_BayesianEstimation_ParameterRecovery.py |
| Falsification-Protocol-12.py | 65/100 | Falsification_LiquidNetworkDynamics_EchoState.py |
