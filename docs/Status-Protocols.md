# VALIDATION PROTOCOLS (VP)

| Protocol | Score | Assessment |
| -------- | ----- | ---------- |
| VP-01 SyntheticEEG_ML | 78 | Verified: Clean PyTorch implementation with proper ML classification. Good fallback handling for missing dependencies. Runtime output shows expected statistical variance warnings (not bugs). |
| VP-02 Behavioral_Bayesian | 80 | Verified: Solid Bayesian comparison framework with proper statistical tests. Good documentation of thresholds. |
| VP-03 ActiveInference | 85 | Verified: Strong implementation. Clean ABC pattern for AgentInterface, proper hierarchical generative model with PyTorch. Well-structured agent comparison framework. |
| VP-04 PhaseTransition | 75 | Verified: Good phase transition implementation with epistemic architecture level 2. Proper mathematical foundations. |
| VP-05 EvolutionaryEmergence | 72 | Verified: Evolutionary optimizer with proper genome encoding. VP-5→FP-5 dependency chain correctly enforced. |
| VP-06 LiquidNetwork | 76 | Verified: Liquid time-constant network (LTCN) implementation with proper inductive bias testing. |
| VP-07 TMS_Causal | 70 | Verified: TMS/pharmacological intervention simulation. Adequate but less comprehensive than other protocols. |
| VP-08 Psychophysical | 78 | Verified: Proper psychometric function implementation with threshold estimation. Good parameter recovery testing. |
| VP-09 NeuralSignatures | 75 | Verified: Neural signature validation (P3b, HEP) with empirical priority 1 markers. |
| VP-10 CausalManipulations | 72 | Verified: Causal manipulation protocols for priority 2 testing. |
| VP-11 MCMC_Cultural | 68 | Verified: SYNTHETIC_PENDING_EMPIRICAL status correctly flagged. Proper PyMC/NUTS fallback to Metropolis-Hastings. Parameter recovery tests recover synthetic data generation, not real cross-cultural data. Results correctly marked as SIMULATION_ONLY. Mathematical implementation is sound but empirical validation missing. |
| VP-12 Clinical_CrossSpecies | 71 | Verified: Cross-species convergence testing with clinical markers. |
| VP-13 Epistemic_Architecture | 76 | Verified: Epistemic architecture predictions P5-P12 implementation. |
| VP-14 fMRI_Anticipation_Exp | 74 | Verified: fMRI anticipation experience with proper HRF convolution. |
| VP-15 fMRI_vmPFC | 45 | Verified: STUB/SIMULATION ONLY. Properly marked as simulation_validated_only. Power analysis implemented. Awaiting empirical fMRI data. Cannot validate P5.a/P5.b without real data. |

## FALSIFICATION PROTOCOLS (FP)

| Protocol | Score | Assessment |
| -------- | ----- | ---------- |
| FP-01 ActiveInference | 82 | Verified: Strong implementation. F1.x-F2.x criteria properly implemented with bootstrap CI, spectral analysis with FOOOF fallback. Good falsification threshold enforcement. Runtime shows expected statistical variance warnings (not bugs) - low variance in PAC measures is expected degenerate behavior in edge cases. F2.1/F2.5 FAIL results are valid falsification outcomes. |
| FP-02 AgentComparison | 80 | Verified: F3.x convergence benchmarks with proper Cohen's d calculations and statistical rigor. Runtime behavior shows valid statistical test results. |
| FP-03 FrameworkLevel | 76 | Verified: Multi-protocol synthesis with named prediction tracking. Successfully orchestrates FP-04 through FP-07. FP-08 dependency issue isolated to that module only. |
| FP-04 PhaseTransition | 77 | Verified: Information-theoretic phase transition with proper hysteresis width testing. Transfer entropy and integrated information computed correctly. Runtime verified working. |
| FP-05 EvolutionaryPlausibility | 74 | Verified: Evolutionary plausibility with proper genome frequency analysis. Genome encoding and evolutionary optimizer functional. CRIT-04 agent creation warnings are expected during VP-05→FP-05 dependency chain execution. |
| FP-06 LiquidNetwork_Energy | 76 | Verified: Energy benchmark with LTCN fast transition testing. Runtime verified working correctly. |
| FP-07 MathematicalConsistency | 81 | Verified: Strong mathematical validation. ODE consistency checks for surprise accumulation, ignition sigmoid, precision update rules. Runtime verified. |
| FP-08 ParameterSensitivity | 70 | Verified: Fixed KeyError: 'parameter_ranking' bug in zero-variance edge case (lines 1623-1644). Report generation now defensive against missing keys. Mathematical correctness is sound (F8.FIM, F8.PL, F8.SA criteria properly implemented). CRIT-04 APGIAgent dependency injection working. SALib Sobol analysis with fallback to RF importance when unavailable. |
| FP-09 NeuralSignatures_P3b_HEP | 75 | Verified: P3b and HEP neural signature validation. Runtime verified. |
| FP-10 BayesianEstimation_MCMC | 79 | Verified: Proper NUTS sampler with PyMC, Metropolis-Hastings fallback. Gelman-Rubin R̂ ≤ 1.01 check. Bayes factor computation. VP-11 Fix 1 implemented with data source flagging. |
| FP-11 LiquidNetworkDynamics | 77 | Verified: Echo state property validation with spectral radius guards. Runtime verified. |
| FP-12 CrossSpeciesScaling | 73 | Verified: Allometric scaling exponent validation. Runtime verified. |
