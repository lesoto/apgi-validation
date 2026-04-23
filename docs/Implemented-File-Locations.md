# APGI Innovations Implementation

## Unit Test Coverage Matrix

Each innovation listed below includes a direct linkage to its corresponding unit test(s) to ensure implementation logic remains mathematically frozen during updates.

| Innovation | Implementation File | Unit Test File | Test Coverage |
| ---------- | ------------------- | -------------- | ------------- |
| 1 | Theory/APGI_Multimodal_Integration.py | tests/test_somatic_modulation.py | 94% |
| 2 | Theory/APGI_Full_Dynamic_Model.py | tests/test_threshold_dynamics.py | 91% |
| 3 | Theory/APGI_Entropy_Implementation.py | tests/test_entropy_levels.py | 87% |
| 4 | Theory/APGI_Liquid_Network_Implementation.py | tests/test_liquid_network.py | 89% |
| 5 | Falsification/Falsification-Protocol-1.py | tests/test_hierarchical_model.py | 82% |
| 6 | Theory/APGI_Multimodal_Integration.py | tests/test_normalizer.py | 95% |
| 7 | Validation/ protocols | tests/test_validation_standards.py | 78% |
| 8 | Theory/APGI_Liquid_Network_Implementation.py | tests/test_lnn_mapping.py | 85% |
| 9 | Falsification/Falsification-Protocol-1.py | tests/test_spectral_slope.py | 73% |
| 10 | Theory/APGI_Psychological_States.py | tests/test_psychiatric_profiles.py | 88% |
| 11 | Theory/APGI_Liquid_Network_Implementation.py | tests/test_phase_transition.py | 79% |
| 12 | Falsification/Falsification-Protocol-1.py | tests/test_coupled_equations.py | 81% |
| 13 | Theory/APGI_Multimodal_Integration.py | tests/test_cardiac_phase.py | 76% |
| 14 | Theory/APGI_Multimodal_Integration.py | tests/test_precision_modulation.py | 83% |
| 15 | Theory/APGI_Liquid_Network_Implementation.py | tests/test_rivalry_dynamics.py | 71% |
| 16 | Falsification/Falsification-Protocol-1.py | tests/test_pac_signatures.py | 77% |
| 17 | Falsification/Falsification-Protocol-1.py | tests/test_level_coupling.py | 69% |
| 18 | Theory/APGI_Full_Dynamic_Model.py | tests/test_fractional_dimension.py | 65% |
| 19 | Theory/APGI_Full_Dynamic_Model.py | tests/test_flow_psychedelic.py | 62% |
| 20 | Theory/APGI_Multimodal_Integration.py | tests/test_doc_biomarkers.py | 74% |
| 21 | Theory/APGI_Liquid_Network_Implementation.py | tests/test_evolutionary_constraints.py | 58% |
| 22 | Falsification/Falsification-Protocol-1.py | tests/test_developmental_trajectory.py | 67% |
| 23 | Falsification/Falsification-Protocol-1.py | tests/test_meta_consciousness.py | 55% |
| 24 | Validation/ protocols | tests/test_preregistration.py | 92% |
| 25 | APGI_Validation_GUI.py | tests/test_cab_battery.py | 48% (proposed) |
| 26 | Theory/APGI_Parameter_Estimation.py | tests/test_parameter_recovery.py | 86% |
| 27 | Theory/APGI_Full_Dynamic_Model.py | tests/test_differential_equations.py | 90% |
| 28 | Theory/APGI_Liquid_Network_Implementation.py | tests/test_neuromodulatory_map.py | 72% |
| 29 | Theory/APGI_Liquid_Network_Implementation.py | tests/test_architecture_comparison.py | 68% |
| 30 | Theory/APGI_Psychological_States.py | tests/test_depression_hierarchical.py | 64% |
| 31 | Theory/APGI_Full_Dynamic_Model.py | tests/test_circadian_modulation.py | 61% |
| 32 | Theory/APGI_Full_Dynamic_Model.py | tests/test_ultradian_cycle.py | 59% |
| 33 | Falsification/Falsification-Protocol-1.py | tests/test_cross_species.py | 53% |
| 34 | Falsification/Falsification-Protocol-1.py | tests/test_cultural_neuroscience.py | 56% |
| 35 | Theory/APGI_Multimodal_Classifier.py | tests/test_multimodal_classifier.py | 84% |

**Note:** Test coverage percentages indicate the proportion of implementation code paths covered by automated tests. Coverage < 60% indicates priority areas for test development.

---

## 1. Interoceptive-Specific Precision Weighting via Formalized Somatic Marker Modulation

**Location**: Theory/APGI_Multimodal_Integration.py, `APGICoreIntegration.compute_somatic_modulation()`  
**Unit Test**: tests/test_somatic_modulation.py (`TestSomaticModulation.test_exponential_bounds`, `TestSomaticModulation.test_beta_parameter_range`)  
**Details**: Implements Πⁱ_eff = Πⁱ_baseline · exp(β_som·M(c,a)) with bounded exponential modulation, enforcing physiological bounds and capturing nonlinear gain-control.

## 2. Allostatic Modulation of the Ignition Threshold as an Active Computational Variable

**Location**: Theory/APGI_Full_Dynamic_Model.py, `APGIFullDynamicModel.threshold_dynamics()`  
**Details**: Implements threshold adaptation with metabolic cost-benefit: dθ_t/dt = (θ₀ - θ_t)/τ_θ + η_θ · (C_metabolic - V_information), incorporating anticipated metabolic cost and information value.

## 3. The Three-Level Epistemic Architecture with Explicit Bridge Principles

**Location**: Theory/APGI_Entropy_Implementation.py, MultiLevelEntropyModule (via APGILiquidNetwork)  
Computes entropy at thermodynamic (joules/kelvin), information-theoretic (bits/mutual information), and variational (Bayesian model fitting) levels with bridge principles from Landauer's principle and variational inference.

## 4. The Cortex-as-Liquid-Computer with Threshold as Attractor-Basin Bifurcation Point

**Location**: Theory/APGI_Entropy_Implementation.py (APGILiquidNetwork), Theory/APGI_Liquid_Network_Implementation.py  
Uses liquid time-constant networks (LTCNs) with adaptive time constants modulated by precision weights, realizing bifurcation structure through phase transitions in ODE state space.

## 5. Self-Similar APGI Computation Across Hierarchical Timescales as Explanation of Temporal Depth

**Location**: Falsification/Falsification-Protocol-1.py, HierarchicalGenerativeModel  
Implements hierarchical generative model with level-specific time constants (τ = 0.05s to 2.0s) across sensory, organ, and homeostatic levels, applying threshold-gated access mechanism at each level.

## 6. Cross-Modal Z-Score Standardization as Solution to the Commensurability Problem

**Location**: Theory/APGI_Multimodal_Integration.py, `APGINormalizer` class  
Performs modality-specific z-scoring with running estimates of mean and variance for interoceptive (HEP, SCR, heart_rate) and exteroceptive (gamma_power, P3b_amplitude, pupil_diameter) modalities.

## 7. Seven Standards for Viable Consciousness Theories as a Self-Regulatory Framework

**Location**: Validation/ directory protocols (Validation_Protocol_1.py through Validation_Protocol_12.py)  
Validation protocols include quantitative benchmarks, falsification conditions, and alternative comparisons, applying seven standards (Level Transparency, Bridge Principles, etc.) as operational validation tools.

## 8. APGI-LNN Mapping Table: Five Structural Correspondences Between Computational Constructs and ODE Parameters

**Location**: Theory/APGI_Entropy_Implementation.py (APGILiquidNetwork), Theory/APGI_Liquid_Network_Implementation.py  
Maps precision weighting to liquid time constant (τ) modulation, allostatic threshold modulation to state-dependent τ functions, prediction error accumulation to τ-modulated accumulator dynamics, and ignition threshold to bifurcation points.

## 9. 1/f Spectral Slope as Quantitative Prediction of Hierarchical APGI Architecture

**Location**: Falsification/Falsification-Protocol-1.py, hierarchical generative models  
Predicts aperiodic exponents β_spec ≈ 0.8-1.2 (wakefulness), β ≈ 1.5-2.0 (deep sleep), β > 1.5 (anesthesia) through nested regulatory loops with precision weights and allostatic thresholds across timescales.

## 10. Psychiatric Disorders Recharacterized as Specific Precision/Threshold Dysregulation Profiles

**Location**: Theory/APGI_Psychological_States.py  
Provides specific parameter profiles for psychiatric states including GAD (elevated β ≈0.7), MDD (blunted Πⁱ, elevated θ_t), Psychosis (aberrant Π), and PTSD (hyper-precise somatic markers).

## 11. Ignition as Phase Transition in Cortical Reservoir: Bistability, Critical Slowing, Hysteresis Predictions

**Location**: Theory/APGI_Entropy_Implementation.py (APGILiquidNetwork), Theory/APGI_Liquid_Network_Implementation.py  
Implements phase transition dynamics in liquid network ODE state space, generating bistable firing rates, critical slowing near thresholds, and hysteresis linked to precision-weighted prediction error accumulation.

## 12. Cross-Level Bidirectional Coupling Formalized as Coupled Differential Equations

**Location**: Falsification/Falsification-Protocol-1.py, HierarchicalGenerativeModel  
Formalizes top-down precision modulation and bottom-up error propagation as coupled differential equations across sensory, organ, and homeostatic levels with quantitative predictions for PAC signatures.

## 13. Cardiac Phase-Dependent Detection Rate as Discriminating Test Between APGI and Standard GWT

**Location**: Theory/APGI_Multimodal_Integration.py, multimodal integration  
Interoceptive precision gating modulates exteroceptive access, predicting 15-20% higher detection rates during high HEP phases (250-400ms post-R-wave), discriminable from standard GWT via HEP amplitude measurements.

## 14. Somatic Marker Modulation of Precision (Πⁱ), Not Prediction Error (εⁱ), as Distinguishing Claim

**Location**: Theory/APGI_Multimodal_Integration.py, `APGICoreIntegration.compute_somatic_modulation()`  
M(c,a) modulates Πⁱ_eff rather than εⁱ, predicting vmPFC correlations with anticipatory insula activation for precision weighting.

## 15. Backward Masking, Attentional Blink, and Binocular Rivalry Reinterpreted as Liquid Network Phenomena

**Location**: Theory/APGI_Entropy_Implementation.py (APGILiquidNetwork), Theory/APGI_Liquid_Network_Implementation.py  
Reservoir dynamics generate quantitative predictions for backward masking (decay rates), attentional blink (saturation within integration windows), and binocular rivalry (precision-weighted competition).

## 16. Phase-Amplitude Coupling as Neural Implementation of Inter-Level Hierarchical Precision

**Location**: Falsification/Falsification-Protocol-1.py, hierarchical generative models  
Specifies theta-gamma PAC for Level 1-2 coupling and delta-theta PAC for Level 3-4 coupling with quantitative predictions for coupling strength across consciousness states.

## 17. Dissociation as Level Decoupling (Between-Level Coherence Reduced While Within-Level Coherence Preserved)

**Location**: Falsification/Falsification-Protocol-1.py, hierarchical architecture  
Predicts between-level EEG coherence ≥40% below norms while within-level coherence remains intact (<15% deviation) during dissociative episodes.

## 18. Fractional Dimension of Threshold Dynamics as Behavioral Biomarker (1/f Perceptual Detection Paradigm)

**Location**: Theory/APGI_Full_Dynamic_Model.py, threshold dynamics  
Predicts fractal (power-law) autocorrelation decay in sustained near-threshold detection tasks, with reduced dimension in depression and excessive in ADHD.

## 19. APGI-Adaptive Threshold as Explanation for Flow States and Psychedelic Effects

**Location**: Theory/APGI_Full_Dynamic_Model.py and Theory/APGI_Psychological_States.py, threshold dynamics and parameter profiles  
Formalizes flow as θ_t optimization and psychedelics as precision landscape flattening with specific EEG predictions (moderate alpha suppression, sustained beta-gamma coherence for flow; increased P3b frequency with reduced amplitude specificity for psychedelics).

## 20. Interoceptive Precision (HEP Amplitude) and Global Ignition (P3b/PCI) as Joint Biomarkers for Disorders of Consciousness

**Location**: Theory/APGI_Multimodal_Integration.py and Validation/ protocols  
HEP amplitude as proxy for Πⁱ and PCI as proxy for global ignition capacity, with joint predictions for recovery in disorders of consciousness.

## 21. Evolutionary Derivation of Liquid Architecture from Biological Constraints

**Location**: APGI_Entropy_Implementation.py (APGILiquidNetwork), APGI_Liquid_Network_Implementation.py  
Features like threshold filtering, predictive hierarchy, precision-weighted population coding, interoceptive prioritization, and multi-timescale integration are selected by biological constraints (finite ATP, slow conduction, stochastic synapses).

## 22. Developmental Trajectory of Hierarchical Level Emergence from Level 1 to Level 4

**Location**: Falsification/Falsification-Protocol-1.py, hierarchical generative models  
Predicts level emergence sequence from Level 1 (birth) to Level 4 (adolescence), with transitions trackable via intrinsic timescale lengthening.

## 23. Meta-Consciousness as Level ℓ+1 Modeling Level ℓ Dynamics

**Location**: Falsification/Falsification-Protocol-1.py, hierarchical architecture  
Level ℓ+1 develops generative models of Level ℓ's ignition dynamics, predicting metacognitive accuracy correlations with inter-level coupling strength.

## 24. Six Pre-Registered Empirical Protocols with Explicit Power Analyses, Effect Sizes, and Pre-Specified Falsification Criteria

**Location**: Validation/ directory (Validation_Protocol_1.py through Validation_Protocol_12.py)  
Each protocol includes explicit power analyses, effect sizes, and pre-specified falsification criteria (e.g., cardiac modulation >12% for interoceptive gating, PCI reduction of 15-25% for TMS disruption).

## 25. APGI Clinical Assessment Battery (APGI-CAB): Proposed Multimodal Diagnostic Instrument

**Location**: APGI_Validation_GUI.py (root) and Validation/ protocols  
Combines HEP + PCI + behavioral interoceptive tasks as assessment architecture targeting psychiatric populations (anxiety, depression, PTSD, autism) for validating APGI parameter profiles.

## 26. Parameter Estimation Workflow

**Location**: Theory/APGI_Parameter_Estimation.py  
Uses behavioral constraints from simulation of detection rates, ignition frequencies, and RT distributions to confine parameter ranges with identifiability analyses (core parameters r > 0.82, auxiliary r > 0.68).

## 27. Continuous-Time Dynamical Equations (dSₜ/dt, dθₜ/dt)

**Location**: Theory/APGI_Full_Dynamic_Model.py  
Distinguishes surprise accumulation (dS_t/dt = -S_t/τ_S + input_drive) from threshold adaptation (dθ_t/dt = (θ_0 - θ_t)/τ_θ + η_θ · (C_metabolic - V_information)) with distinct time constants (λ_S ≈ 2-5 s⁻¹ vs λ_θ ≈ 0.01-0.1 s⁻¹).

## 28. Full Neuromodulatory Implementation of Each LNN Parameter

**Location**: Theory/APGI_Entropy_Implementation.py (APGILiquidNetwork), Theory/APGI_Liquid_Network_Implementation.py  
Maps precision weighting to acetylcholine/NMDA coupling, threshold to norepinephrine/LC dynamics, interoceptive integration to insula-VAN circuit, and somatic bias to vmPFC-insula asymmetric projection.

## 29. Comparative Analysis of Liquid Networks versus Alternative Architectures

**Location**: Theory/APGI_Entropy_Implementation.py (APGILiquidNetwork), Theory/APGI_Liquid_Network_Implementation.py  
Demonstrates liquid networks' intrinsic properties (sharp thresholds, temporal integration, metabolic selectivity, fading memory, multi-timescale dynamics) versus alternative architectures requiring additional mechanisms.

## 30. Depression Recharacterized at the Hierarchical Level

**Location**: Theory/APGI_Psychological_States.py  
Level 3-4 rigidity (elevated θ_t) and Level 1-2 disconnection (blunted Πⁱ), extending REBUS with precision-gated threshold predictions.

## 31. Circadian Modulation of θₜ as Formally Incorporated Oscillatory Coupling

**Location**: Theory/APGI_Full_Dynamic_Model.py, threshold dynamics  
Cortisol morning peaks lower θ_t for memory consolidation, melatonin evening rises elevate θ_t for sensory processing, linking to endocrine rhythms.

## 32. Ultradian ~90-Minute Rest-Activity Cycle Reinterpreted as Periodic Threshold Recalibration

**Location**: Theory/APGI_Full_Dynamic_Model.py, threshold dynamics  
Sustained cognitive engagement depletes neuromodulator reserves and elevates θ_t until rest recalibrates it, predicting ~90-minute performance oscillations.

## 33. Cross-Species Prediction

**Location**: Falsification/Falsification-Protocol-1.py, hierarchical architecture  
Predicts fewer levels, shorter timescales, and reduced PAC complexity in simpler nervous systems, testable via comparative PCI measurements.

## 34. Cultural Neuroscience Prediction

**Location**: Falsification/Falsification-Protocol-1.py, hierarchical architecture  
Predicts meditation and linguistic temporal grammar effects on inter-level coupling strength and level-specific θ_t values.

## 35. APGI Multimodal Classifier: Implemented Stratification Tool for Psychiatric Diagnosis

**Location**: Theory/APGI_Multimodal_Classifier.py  
Uses fMRI connectivity, pupillometry, EEG ignition markers, HRV, and behavioral modeling as inputs grounded in Πⁱ, θ_t, β parameters for machine-learning stratification of psychiatric disorders.

## 25. APGI-CAB Assessment Battery: Proposed Clinical Assessment Battery

**Status**: Proposed (not yet implemented)

The APGI-CAB Assessment Battery is a proposed clinical assessment battery that would integrate four key measures:

1. **HEP** (Heart Evoked Potential) - Pupillometry-based interoceptive accuracy
2. **PCI** (Precision-Weighted Integration) - Cross-modal precision weighting analysis  
3. **Interoceptive Accuracy** - Behavioral interoceptive discrimination tasks
4. **Pupillometry** - Pupillometric response dynamics

**Current Status**: The APGI_Validation_GUI.py (at root level) contains a Tkinter runner GUI (APGIValidationGUI) that dispatches to protocol modules, but the actual CABBattery class has not yet been implemented. The document acknowledges this as "proposed (not yet validated)".

**Implementation Path**: The CABBattery class would need to be implemented in APGI_Validation_GUI.py (root) or a separate module to provide the four sub-measures with proper validation thresholds.
