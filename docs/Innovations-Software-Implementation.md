# APGI Innovations

**Document Version:** 1.1  
**Last Updated:** April 22, 2026  
**Code Repository:** /Users/lesoto/Sites/PYTHON/apgi-validation

## Implementation Reference Guide

This document maps APGI theoretical innovations to their software implementations. Files are organized in:

- `Falsification/FP_*.py` — Falsification protocols (FP-01 to FP-12)
- `Validation/VP_*.py` — Validation protocols (VP-01 to VP-15)
- `utils/*.py` — Shared utilities and core implementations
- Root `*_GUI.py` — Graphical user interface runners

## FALSIFICATION CRITERIA NOTATION

All falsification criteria follow this format:

- **Quantitative threshold:** Numerical value with units
- **Statistical test:** Specific test type and significance level
- **Minimum effect size:** Cohen's d, η², or correlation coefficient
- **Alternative hypothesis:** What would falsify the prediction

Example: "Cardiac modulation >12% (d > 0.5, p < 0.01, two-tailed t-test); falsified if modulation <5% or non-significant"

## Foundational Innovations

Ideas whose specific formal structure does not exist in the prior literature and upon which the entire framework depends.

## 1. Interoceptive-Specific Precision Weighting via Formalized Somatic Marker Modulation

**Paper 1 Equation Reference**: Eq. 1 (Somatic Marker Modulation)

```text
Πⁱ_eff = Πⁱ_baseline · exp(β_som·M(c,a))     [Eq. 1]
```

**Implementation Details**:
The somatic marker modulation equation is implemented in falsification protocols through precision-weighted agent simulations. The bounded exponential modulation is applied to baseline interoceptive precision using the somatic marker value and individual beta parameter, with Πⁱ_eff hard-clamped to absolute bounds [0.1, 10.0] to prevent physiologically implausible precision values (typical dynamic range 0.1–10× when Πⁱ_baseline ≈ 1.0), capturing the nonlinear gain-control role of somatic markers.

- Equation location: Paper 1, Section 3.2, Eq. 1
- Variable definitions: See Paper 1, Table 1 (Parameter Definitions)

`Falsification/FP_01_ActiveInference.py`, `Falsification/FP_02_AgentComparison_ConvergenceBenchmark.py`

Implement Πⁱ_eff = Πⁱ_baseline · exp(β_som·M(c,a)) with bounded exponential modulation in agent-based simulations, enforcing physiological bounds and capturing nonlinear gain-control across hierarchical levels.

## 2. Allostatic Modulation of the Ignition Threshold as an Active Computational Variable

The allostatic modulation of the ignition threshold is implemented in agent step functions across falsification protocols. The threshold adapts dynamically as a predictive variable, with the equation dtheta_dt = (self.theta_0 - self.theta_t) / self.tau_theta + self.eta_theta * (self.metabolic_cost - self.information_value), incorporating anticipated metabolic cost and information value to preemptively adjust the access criterion.

`Falsification/FP_01_ActiveInference.py`, `utils/constants.py` (DIM_CONSTANTS)

Implements threshold adaptation with metabolic cost-benefit: dθ_t/dt = (θ₀ - θ_t)/τ_θ + η_θ · (C_metabolic - V_information), incorporating anticipated metabolic cost and information value with configurable timescales τ_θ ≈ 10-100s.

## 3. The Three-Level Epistemic Architecture with Explicit Bridge Principles

The three-level epistemic architecture is implemented across falsification protocols with distinct computational approaches: thermodynamic (energy/entropy constraints in FP-06), information-theoretic (transfer entropy, mutual information in FP-04), and variational (Bayesian model fitting in FP-10). These implementations provide bridge principles for computational-thermodynamic mappings and include falsification criteria for each level.

`Falsification/FP_04_PhaseTransition_EpistemicArchitecture.py`, `Falsification/FP_06_LiquidNetwork_EnergyBenchmark.py`, `Falsification/FP_10_BayesianEstimation_MCMC.py`

Computes entropy/energy metrics at thermodynamic (joules/kelvin), information-theoretic (bits/mutual information), and variational (Bayesian model fitting) levels with bridge principles from Landauer's principle and variational inference.

## 4. The Cortex-as-Liquid-Computer with Threshold as Attractor-Basin Bifurcation Point

The cortex-as-liquid-computer with threshold as attractor-basin bifurcation point is implemented in liquid network protocols, where liquid time-constant networks (LTCNs) model cortical dynamics with adaptive time constants modulated by precision weights. The bifurcation structure is realized as a saddle-node bifurcation in the network state space, occurring when precision-weighted prediction error Π·|ε| equals threshold θ_t, creating bistable attractors (low-firing "subthreshold" vs. high-firing "ignited" states) and generating quantitative predictions about hysteresis width (Δθ ≈ 0.1–0.2 θ_t), critical slowing (τ_critical ∝ |Π·|ε| - θ_t|^(-0.5)), and bistability linked to precision-weighted prediction error accumulation and threshold adaptation rate.

`Falsification/FP_06_LiquidNetwork_EnergyBenchmark.py`, `Falsification/FP_11_LiquidNetworkDynamics_EchoState.py`, `Validation/VP_06_LiquidNetwork_InductiveBias.py`

Uses liquid time-constant networks (LTCNs) with adaptive time constants modulated by precision weights, realizing bifurcation structure through phase transitions in network state space.

## 5. Self-Similar APGI Computation Across Hierarchical Timescales as Explanation of Temporal Depth

The self-similar APGI computation across hierarchical timescales is implemented through HierarchicalGenerativeModel classes across falsification protocols, with level-specific time constants (tau) ranging from 0.05s to 2.0s across sensory, organ, and homeostatic levels. The threshold-gated access mechanism (Π·|ε| > θ_t) is applied at each level with level-specific parameters, converting the established temporal hierarchy into a unified mechanistic framework for temporal depth.

`Falsification/FP_01_ActiveInference.py`, `Falsification/FP_03_FrameworkLevel_MultiProtocol.py`

Implement hierarchical generative model with level-specific time constants (τ = 0.05s to 2.0s) across sensory, organ, and homeostatic levels, applying threshold-gated access mechanism at each level.

## Major Advances

Significant extensions or formalizations of existing ideas that substantially increase specificity, testability, or explanatory power.

## 6. Cross-Modal Z-Score Standardization as Solution to the Commensurability Problem

The cross-modal z-score standardization is implemented in validation and falsification protocols through EEG processing utilities. Modality-specific z-scoring with running estimates of mean and variance for interoceptive (HEP, SCR, heart_rate) and exteroceptive (gamma_power, P3b_amplitude, pupil_diameter) modalities yields dimensionless, comparable prediction error magnitudes for multimodal integration.

`utils/eeg_processing.py`, `utils/eeg_simulator.py`, `Validation/VP_09_NeuralSignatures_EmpiricalPriority1.py`

Performs modality-specific z-scoring with running estimates for interoceptive and exteroceptive modalities.

## 7. Seven Standards for Viable Consciousness Theories as a Self-Regulatory Framework

The seven standards for viable consciousness theories are implemented as validation criteria across the APGI validation suite, with self-application demonstrated in the Falsification/ and Validation/ directories. Each protocol includes quantitative benchmarks, falsification conditions, and alternative comparisons, converting the standards into operational validation tools.

`Validation/VP_01_SyntheticEEG_MLClassification.py` through `Validation/VP_15_fMRI_Anticipation_vmPFC.py`

Validation protocols include quantitative benchmarks, falsification conditions, and alternative comparisons, applying seven standards (Level Transparency, Bridge Principles, Quantitative Benchmarks, Falsification Conditions, Alternative Comparison, Evolutionary Plausibility, Causal Roadmap) as operational validation tools.

## 8. APGI-LNN Mapping Table: Five Structural Correspondences Between Computational Constructs and ODE Parameters

The APGI-LNN mapping table is implemented across liquid network protocols, with precision weighting mapped to liquid time constant (τ) modulation via configurable parameters, allostatic threshold modulation to state-dependent τ functions, prediction error accumulation to τ-modulated accumulator dynamics, and ignition threshold to bifurcation points in the network state space.

`Falsification/FP_06_LiquidNetwork_EnergyBenchmark.py`, `Falsification/FP_11_LiquidNetworkDynamics_EchoState.py`, `utils/constants.py`

Maps precision weighting to liquid time constant (τ) modulation, allostatic threshold modulation to state-dependent dynamics, prediction error accumulation to modulated accumulator dynamics, and ignition threshold to saddle-node bifurcation points where Π·|ε| = θ_t defines the separatrix between subthreshold and ignited attractor basins.

## 9. 1/f Spectral Slope as Quantitative Prediction of Hierarchical APGI Architecture

The 1/f spectral slope predictions are implemented through hierarchical generative models in falsification protocols, where nested regulatory loops with precision weights and allostatic thresholds across timescales generate the predicted aperiodic exponents (α_spec ≈ 0.8-1.2 wakefulness, α_spec ≈ 1.5-2.0 deep sleep, α_spec > 1.5 anesthesia), with falsification criteria based on spectral analysis with FOOOF fallback.

`Falsification/FP_01_ActiveInference.py`, `Falsification/FP_09_NeuralSignatures_P3b_HEP.py`, `utils/eeg_processing.py`

Predicts aperiodic exponents α_spec ≈ 0.8-1.2 (wakefulness), α_spec ≈ 1.5-2.0 (deep sleep), α_spec > 1.5 (anesthesia) through nested regulatory loops with precision weights and allostatic thresholds across timescales.

## 10. Psychiatric Disorders Recharacterized as Specific Precision/Threshold Dysregulation Profiles

The psychiatric disorder recharacterization is implemented through configuration profiles and parameter validation utilities, with specific parameter profiles for GAD (elevated β ≈0.7), MDD (blunted Πⁱ, elevated θ_t), Psychosis (aberrant Π), and PTSD (hyper-precise somatic markers), providing quantitative biomarker predictions for hierarchical level-specific dysregulation.

`config/profiles/*.yaml`, `utils/parameter_validator.py`, `utils/apgi_config.py`

Provides specific parameter profiles for psychiatric states including GAD (elevated β ≈0.7), MDD (blunted Πⁱ, elevated θ_t), Psychosis (aberrant Π), and PTSD (hyper-precise somatic markers).

## 11. Ignition as Phase Transition in Cortical Reservoir: Bistability, Critical Slowing, Hysteresis Predictions

The phase transition predictions are implemented in liquid network protocols, where precision-weighted prediction error accumulation leads to saddle-node bifurcation-induced bistable firing rates (subthreshold <5 Hz vs. ignited 20-40 Hz), critical slowing near thresholds (autocorrelation time τ_AC ∝ |Π·|ε| - θ_t|^(-0.5)), and hysteresis width Δθ ≈ 0.1–0.2 θ_t in the network state space, with testable signatures for intracranial EEG including pre-ignition variance increases and post-ignition relaxation timescales.

`Falsification/FP_04_PhaseTransition_EpistemicArchitecture.py`, `Falsification/FP_11_LiquidNetworkDynamics_EchoState.py`

Implements saddle-node bifurcation dynamics in liquid network state space, generating bistable firing rates (subthreshold <5 Hz vs. ignited 20-40 Hz), critical slowing (τ_AC ∝ |Π·|ε| - θ_t|^(-0.5)), and hysteresis (Δθ ≈ 0.1–0.2 θ_t) linked to precision-weighted prediction error accumulation.

## 12. Cross-Level Bidirectional Coupling Formalized as Coupled Differential Equations

The cross-level bidirectional coupling is implemented as coupled differential equations in hierarchical generative models across falsification protocols, formalizing top-down precision modulation and bottom-up error propagation across sensory, organ, and homeostatic levels with quantitative predictions for PAC signatures.

`Falsification/FP_01_ActiveInference.py`, `Falsification/FP_03_FrameworkLevel_MultiProtocol.py`

Formalizes top-down precision modulation and bottom-up error propagation as coupled differential equations across sensory, organ, and homeostatic levels with quantitative predictions for PAC signatures.

## Significant Contributions

Genuine advances that strengthen the framework's scope, testability, or clinical application, building on established ideas with meaningful new specificity.

## 13. Cardiac Phase-Dependent Detection Rate as Discriminating Test Between APGI and Standard GWT

The cardiac phase-dependent detection rate prediction is implemented in validation protocols for neural signatures, where interoceptive precision gating modulates exteroceptive access, predicting 15-20% higher detection rates during high HEP phases, discriminable from standard GWT via HEP amplitude measurements.

`Validation/VP_09_NeuralSignatures_EmpiricalPriority1.py`, `utils/eeg_processing.py`

Interoceptive precision gating modulates exteroceptive access, predicting 15-20% higher detection rates during high HEP phases (250-400ms post-R-wave), discriminable from standard GWT via HEP amplitude measurements.

## 14. Somatic Marker Modulation of Precision (Πⁱ), Not Prediction Error (εⁱ), as Distinguishing Claim

The somatic marker modulation of precision is implemented in falsification protocol agents, where M(c,a) modulates Πⁱ_eff rather than εⁱ, predicting vmPFC correlations with anticipatory insula activation for precision weighting.

`Falsification/FP_01_ActiveInference.py`, `Falsification/FP_02_AgentComparison_ConvergenceBenchmark.py`

M(c,a) modulates Πⁱ_eff rather than εⁱ, predicting vmPFC correlations with anticipatory insula activation for precision weighting.

## 15. Backward Masking, Attentional Blink, and Binocular Rivalry Reinterpreted as Liquid Network Phenomena

The reinterpretation of backward masking, attentional blink, and binocular rivalry is implemented in liquid network protocols, where reservoir decay rates, saturation within integration windows, and precision-weighted competition generate quantitative predictions for each paradigm's parameters.

`Falsification/FP_06_LiquidNetwork_EnergyBenchmark.py`, `Falsification/FP_11_LiquidNetworkDynamics_EchoState.py`, `Validation/VP_06_LiquidNetwork_InductiveBias.py`

Reservoir dynamics generate quantitative predictions for backward masking (decay rates), attentional blink (saturation within integration windows), and binocular rivalry (precision-weighted competition).

## 16. Phase-Amplitude Coupling as Neural Implementation of Inter-Level Hierarchical Precision

The phase-amplitude coupling predictions are implemented in hierarchical generative models across falsification protocols, specifying theta-gamma PAC for Level 1-2 coupling and delta-theta PAC for Level 3-4 coupling with quantitative predictions for coupling strength across states.

`Falsification/FP_01_ActiveInference.py`, `Falsification/FP_09_NeuralSignatures_P3b_HEP.py`, `utils/eeg_processing.py`

Specifies theta-gamma PAC for Level 1-2 coupling and delta-theta PAC for Level 3-4 coupling with quantitative predictions for coupling strength across consciousness states.

## 17. Dissociation as Level Decoupling (Between-Level Coherence Reduced While Within-Level Coherence Preserved)

The dissociation as level decoupling is implemented in hierarchical generative models across falsification protocols, predicting between-level EEG coherence (delta-theta, theta-gamma PAC) ≥40% below norms (d > 1.2, p < 0.001) while within-level coherence remains intact (<15% deviation, d < 0.3, p > 0.05) during dissociative episodes (Clinician-Administered Dissociative States Scale score >15); falsified if between-level reduction <20% or within-level reduction >25%.

`Falsification/FP_01_ActiveInference.py`, `Falsification/FP_03_FrameworkLevel_MultiProtocol.py`

Predicts between-level EEG coherence ≥40% below norms while within-level coherence remains intact (<15% deviation) during dissociative episodes.

## 18. Fractional Dimension of Threshold Dynamics as Behavioral Biomarker (1/f Perceptual Detection Paradigm)

The fractional dimension of threshold dynamics is implemented in falsification protocol agents, predicting fractal (power-law) autocorrelation decay in sustained near-threshold detection tasks, with reduced dimension in depression and excessive in ADHD.

`Falsification/FP_01_ActiveInference.py`, `Falsification/FP_08_ParameterSensitivity_Identifiability.py`

Predicts fractal (power-law) autocorrelation decay in sustained near-threshold detection tasks, with reduced dimension in depression and excessive in ADHD.

## 19. APGI-Adaptive Threshold as Explanation for Flow States and Psychedelic Effects

The APGI-adaptive threshold explanations are implemented in falsification protocol agents and configuration profiles, formalizing flow as θ_t optimization and psychedelics as precision landscape flattening with specific EEG predictions.

`Falsification/FP_01_ActiveInference.py`, `config/profiles/*.yaml`, `utils/parameter_validator.py`

Formalizes flow as θ_t optimization and psychedelics as precision landscape flattening with specific EEG predictions (moderate alpha suppression, sustained beta-gamma coherence for flow; increased P3b frequency with reduced amplitude specificity for psychedelics).

## 20. Interoceptive Precision (HEP Amplitude) and Global Ignition (P3b/PCI) as Joint Biomarkers for Disorders of Consciousness

The joint biomarkers are implemented in validation and falsification protocols, with HEP amplitude as proxy for Πⁱ and PCI as proxy for global ignition capacity, predicting recovery in disorders of consciousness.

`Validation/VP_09_NeuralSignatures_EmpiricalPriority1.py`, `Falsification/FP_09_NeuralSignatures_P3b_HEP.py`, `utils/eeg_processing.py`

HEP amplitude as proxy for Πⁱ and PCI as proxy for global ignition capacity, with joint predictions for recovery in disorders of consciousness.

## 21. Evolutionary Derivation of Liquid Architecture from Biological Constraints

The evolutionary derivation is implemented in liquid network and evolutionary protocols, where features like threshold filtering, predictive hierarchy, and precision-weighted coding are selected by biological constraints (finite ATP, slow conduction, stochastic synapses).

`Falsification/FP_05_EvolutionaryPlausibility.py`, `Falsification/FP_06_LiquidNetwork_EnergyBenchmark.py`, `Falsification/FP_11_LiquidNetworkDynamics_EchoState.py`

Features like threshold filtering, predictive hierarchy, precision-weighted population coding, interoceptive prioritization, and multi-timescale integration are selected by biological constraints (finite ATP, slow conduction, stochastic synapses).

## 22. Developmental Trajectory of Hierarchical Level Emergence from Level 1 to Level 4

The developmental trajectory is implemented in hierarchical generative models across falsification protocols, predicting level emergence sequence from Level 1 (birth, τ₁≈50ms) to Level 2 (6–12 months, τ₂≈200–500ms), Level 3 (2–4 years, τ₃≈1–2s), and Level 4 (12–16 years, τ₄≈5–20s), with level transitions defined by >50% increase in intrinsic timescale and >30% increase in inter-level PAC strength, falsifiable if no such transitions occur or if sequence order differs.

`Falsification/FP_01_ActiveInference.py`, `Falsification/FP_03_FrameworkLevel_MultiProtocol.py`, `Falsification/FP_04_PhaseTransition_EpistemicArchitecture.py`

Predicts level emergence sequence: Level 1 (birth, τ₁≈50ms) → Level 2 (6–12mo, τ₂≈200–500ms) → Level 3 (2–4yr, τ₃≈1–2s) → Level 4 (12–16yr, τ₄≈5–20s), with transitions defined by >50% increase in intrinsic timescale and >30% increase in inter-level PAC strength.

## 23. Meta-Consciousness as Level ℓ+1 Modeling Level ℓ Dynamics

The meta-consciousness implementation is in hierarchical architecture across falsification protocols, where Level ℓ+1 develops generative models of Level ℓ's ignition dynamics, predicting metacognitive accuracy (type-2 ROC AUC) correlates with inter-level coupling strength (Level ℓ–ℓ+1 PAC) at r > 0.45, p < 0.001, with TMS disruption of Level ℓ+1 reducing metacognitive sensitivity (meta-d'/d') by ≥35% while preserving Level ℓ task performance (d' reduction <10%).

`Falsification/FP_01_ActiveInference.py`, `Falsification/FP_03_FrameworkLevel_MultiProtocol.py`, `Falsification/FP_07_TMS_CausalInterventions.py`

Level ℓ+1 develops generative models of Level ℓ's ignition dynamics, predicting metacognitive accuracy (type-2 ROC AUC) correlates with Level ℓ–ℓ+1 PAC strength at r > 0.45 (p < 0.001), with TMS disruption of Level ℓ+1 reducing meta-d'/d' by ≥35% while preserving Level ℓ performance (d' reduction <10%).

## Incremental and Methodological Contributions

Refinements, operationalizations, and applications that extend the framework's reach without introducing new core ideas.

## 24. Six Pre-Registered Empirical Protocols with Explicit Power Analyses, Effect Sizes, and Pre-Specified Falsification Criteria

The pre-registered empirical protocols are implemented in the Validation/ directory, with protocols VP-01 through VP-15, each including explicit power analyses, effect sizes, and pre-specified falsification criteria.

`Validation/VP_01_SyntheticEEG_MLClassification.py` through `Validation/VP_15_fMRI_Anticipation_vmPFC.py`

Each protocol includes explicit power analyses, effect sizes, and pre-specified falsification criteria (e.g., cardiac modulation >12% for interoceptive gating, PCI reduction of 15-25% for TMS disruption).

## 25. APGI Clinical Assessment Battery (APGI-CAB): Proposed Multimodal Diagnostic Instrument

The APGI Clinical Assessment Battery is proposed in the validation framework of GUI runners and validation protocols, combining HEP, PCI, and behavioral interoceptive tasks for multimodal diagnostic assessment targeting psychiatric populations.

`APGI_Validation_GUI.py` (root), `Validation/VP_09_NeuralSignatures_EmpiricalPriority1.py`, `Validation/VP_12_Clinical_CrossSpecies_Convergence.py`

Combines HEP + PCI + behavioral interoceptive tasks as assessment architecture targeting psychiatric populations (anxiety, depression, PTSD, autism) for validating APGI parameter profiles.

## 26. Parameter Estimation Workflow

The parameter estimation workflow is implemented in parameter validation and falsification protocols, using behavioral constraints from simulation of detection rates, ignition frequencies, and RT distributions to confine parameter ranges with identifiability analyses.

`utils/parameter_validator.py`, `Falsification/FP_08_ParameterSensitivity_Identifiability.py`, `Falsification/FP_10_BayesianEstimation_MCMC.py`

Uses behavioral constraints from simulation of detection rates, ignition frequencies, and RT distributions to confine parameter ranges with identifiability analyses (core parameters r > 0.82, auxiliary r > 0.68).

## 27. Continuous-Time Dynamical Equations (dSₜ/dt, dθₜ/dt)

The continuous-time dynamical equations are implemented in falsification protocol agents, distinguishing surprise accumulation (dS_t/dt) from threshold adaptation (dθ_t/dt) with distinct time constants: τ_S ≈ 0.2–0.5 s (fast surprise integration) vs τ_θ ≈ 10–100 s (slow threshold adaptation), equivalent to decay rates λ_S = 1/τ_S ≈ 2–5 s⁻¹ and λ_θ = 1/τ_θ ≈ 0.01–0.1 s⁻¹.

`Falsification/FP_01_ActiveInference.py`, `utils/constants.py` (DIM_CONSTANTS)

Distinguishes surprise accumulation (dS_t/dt = -S_t/τ_S + input_drive) from threshold adaptation (dθ_t/dt = (θ_0 - θ_t)/τ_θ + η_θ · (C_metabolic - V_information)) with distinct time constants: τ_S ≈ 0.2–0.5 s vs τ_θ ≈ 10–100 s (decay rates λ_S = 1/τ_S ≈ 2–5 s⁻¹, λ_θ = 1/τ_θ ≈ 0.01–0.1 s⁻¹).

## 28. Full Neuromodulatory Implementation of Each LNN Parameter

The full neuromodulatory implementation is in liquid network protocols and constants, mapping precision weighting to acetylcholine/NMDA coupling, threshold to norepinephrine/LC dynamics, interoceptive integration to insula-VAN circuit, and somatic bias to vmPFC-insula asymmetric projection.

`Falsification/FP_06_LiquidNetwork_EnergyBenchmark.py`, `Falsification/FP_11_LiquidNetworkDynamics_EchoState.py`, `utils/constants.py`

Maps precision weighting to acetylcholine/NMDA coupling, threshold to norepinephrine/LC dynamics, interoceptive integration to insula-VAN circuit, and somatic bias to vmPFC-insula asymmetric projection.

## 29. Comparative Analysis of Liquid Networks versus Alternative Architectures

The comparative analysis is implemented in liquid network protocols, demonstrating liquid networks' intrinsic properties (sharp thresholds, temporal integration, metabolic selectivity, fading memory, multi-timescale dynamics) versus alternative architectures requiring additional mechanisms.

`Falsification/FP_06_LiquidNetwork_EnergyBenchmark.py`, `Falsification/FP_11_LiquidNetworkDynamics_EchoState.py`, `Validation/VP_06_LiquidNetwork_InductiveBias.py`

Demonstrates liquid networks' intrinsic properties (sharp thresholds, temporal integration, metabolic selectivity, fading memory, multi-timescale dynamics) versus alternative architectures requiring additional mechanisms.

## 30. Depression Recharacterized at the Hierarchical Level

The depression recharacterization is implemented in configuration profiles and parameter validation, with Level 3-4 rigidity (elevated θ_t) and Level 1-2 disconnection (blunted Πⁱ), extending REBUS with precision-gated threshold predictions.

`config/profiles/*.yaml`, `utils/parameter_validator.py`, `utils/apgi_config.py`

Level 3-4 rigidity (elevated θ_t) and Level 1-2 disconnection (blunted Πⁱ), extending REBUS with precision-gated threshold predictions.

## 31. Circadian Modulation of θₜ as Formally Incorporated Oscillatory Coupling

The circadian modulation is implemented in falsification protocol agents, with cortisol morning peaks lowering θ_t for memory consolidation and melatonin evening rises elevating it for sensory processing.

`Falsification/FP_01_ActiveInference.py`, `utils/constants.py`

Cortisol morning peaks lower θ_t for memory consolidation, melatonin evening rises elevate θ_t for sensory processing, linking to endocrine rhythms.

## 32. Ultradian ~90-Minute Rest-Activity Cycle Reinterpreted as Periodic Threshold Recalibration

The ultradian cycle is implemented in falsification protocol agents, with sustained cognitive engagement depleting neuromodulator reserves and elevating θ_t until rest recalibrates it, predicting ~90-minute performance oscillations.

`Falsification/FP_01_ActiveInference.py`, `Falsification/FP_05_EvolutionaryPlausibility.py`

Sustained cognitive engagement depletes neuromodulator reserves and elevates θ_t until rest recalibrates it, predicting ~90-minute performance oscillations.

## 33. Cross-Species Prediction

The cross-species prediction is implemented in hierarchical architecture across falsification protocols, predicting hierarchical level count scaling as L ≈ 1 + log₁₀(N_cortical_neurons/10⁶), level-specific timescales scaling as τ_ℓ ∝ (brain_mass)^(1/4), and PAC complexity (number of cross-frequency couplings) scaling as C_PAC ∝ L(L-1)/2, testable via comparative PCI measurements across species (e.g., mouse L≈2, τ₁≈30ms; macaque L≈3, τ₁≈80ms; human L≈4, τ₁≈150ms).

`Falsification/FP_12_CrossSpeciesScaling.py`, `Falsification/FP_03_FrameworkLevel_MultiProtocol.py`

Predicts hierarchical level count L ≈ 1 + log₁₀(N_cortical_neurons/10⁶), level timescales τ_ℓ ∝ (brain_mass)^(1/4), and PAC complexity C_PAC ∝ L(L-1)/2 in cross-species comparisons, with specific predictions: mouse (L≈2, τ₁≈30ms), macaque (L≈3, τ₁≈80ms), human (L≈4, τ₁≈150ms), testable via comparative PCI and intrinsic timescale measurements.

## 34. Cultural Neuroscience Prediction

The cultural neuroscience prediction is implemented in MCMC/cultural validation protocols, predicting meditation effects (10,000+ hours focused attention practice increases Level 3–4 theta-gamma PAC by 15–25% and reduces θ_t coefficient of variation by 30–40%) and linguistic temporal grammar effects (languages with rich tense/aspect morphology, e.g., >8 distinct temporal markers, show 10–20% stronger delta-theta PAC for Level 3–4 coupling and 5–15% elevated baseline θ_t in temporal reasoning tasks compared to isolating languages).

`Validation/VP_11_MCMC_CulturalNeuroscience_Priority3.py`, `Falsification/FP_10_BayesianEstimation_MCMC.py`

Predicts meditation effects: 10,000+ hours focused attention practice → 15–25% increase in Level 3–4 theta-gamma PAC, 30–40% reduction in θ_t coefficient of variation; linguistic temporal grammar effects: languages with >8 temporal markers → 10–20% stronger delta-theta PAC and 5–15% elevated baseline θ_t in temporal tasks vs. isolating languages.

## 35. APGI Multimodal Classifier: Proposed Stratification Tool for Psychiatric Diagnosis

The APGI Multimodal Classifier is proposed in the validation framework of GUI runners and validation protocols, using fMRI connectivity, pupillometry, EEG ignition markers, HRV, and behavioral modeling as inputs grounded in Πⁱ, θ_t, β parameters.

`APGI_Validation_GUI.py` (root), `Validation/VP_09_NeuralSignatures_EmpiricalPriority1.py`, `Validation/VP_14_fMRI_Anticipation_Experience.py`

Uses fMRI connectivity, pupillometry, EEG ignition markers, HRV, and behavioral modeling as inputs grounded in Πⁱ, θ_t, β parameters for machine-learning stratification of psychiatric disorders.
