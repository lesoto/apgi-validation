# APGI Innovations

## Foundational Innovations

Ideas whose specific formal structure does not exist in the prior literature and upon which the entire framework depends.

## 1. Interoceptive-Specific Precision Weighting via Formalized Somatic Marker Modulation

Prior frameworks either treat all prediction errors equivalently (standard predictive processing), describe somatic markers qualitatively (Damasio, 1994), handle interoceptive precision separately from exteroception (Seth, 2013), or model somatic markers within Bayesian frameworks without specifying a bounded gain-control equation (Allen, Levy, Parr, & Friston, 2019). APGI builds on these foundations—particularly Seth's (2013) interoceptive inference framework, which established conceptually that interoceptive precision modulates access to conscious awareness, and Critchley & Garfinkel's (2017) empirical quantification of interoceptive precision via HEP amplitude—to produce the first formally specified integration of somatic marker modulation into a single precision-weighting equation:

Πⁱ_eff = Πⁱ_baseline · exp(β·M(c,a))

where M(c,a) is a bounded somatic marker value ∈ [−2, +2] and β is an empirically constrained individual-difference parameter with an expected range of approximately 0.3–0.8, derived from simulation studies and calibrated against behavioral data (see parameter estimation protocol). The bounded exponential structure prevents runaway amplification while capturing the nonlinear gain-control role of somatic markers—a functional form absent from Seth's conceptual framework, Allen et al.'s (2019) Bayesian modeling, and Damasio's qualitative account. The formalization produces testable individual-difference predictions about β (higher in anxiety, predicted ≈0.7; lower in alexithymia, predicted ≈0.3) that follow from the equation's structure and are pre-specified targets for empirical validation, not post-hoc parameter fits. No single prior framework generates these predictions in this form.

### Connection to App Implementation (Innovation 1)

The somatic marker modulation equation Πⁱ_eff = Πⁱ_baseline · exp(β·M(c,a)) is implemented in the `compute_somatic_modulation` method of the `APGICoreIntegration` class in `APGI-Multimodal-Integration.py`. This method applies the bounded exponential modulation to the baseline interoceptive precision using the somatic marker value and individual beta parameter, enforcing physiological bounds and capturing the nonlinear gain-control role of somatic markers.

### Relationship to Prior Work (Innovation 1)

Seth (2013) provides the conceptual foundation; Critchley & Garfinkel (2017) provide the empirical grounding via HEP. Allen, Levy, Parr, & Friston (2019) provide a Bayesian formal treatment of interoceptive active inference. APGI's contribution is the specific bounded exponential formalization with parameterized individual differences—a quantitative advance that adds structural specificity absent from all three precursor frameworks. Note also that Seth & Friston (2016) provide a sophisticated formalization of active interoceptive inference; APGI's equation extends rather than replaces their treatment by incorporating somatic marker gain-control in a single closed-form expression.

## 2. Allostatic Modulation of the Ignition Threshold as an Active Computational Variable

Prior theories with threshold-like mechanisms (GWT, recurrent processing theory) treat the threshold as fixed or passively attention-dependent. Aston-Jones & Cohen (2005) established that the LC-NE system predictively regulates neural gain thresholds based on task utility—providing the adaptive gain logic that APGI extends. Yu & Dayan (2005) formalized acetylcholine as implementing expected uncertainty that adaptively sets processing thresholds—demonstrating the cost-benefit logic in neuromodulatory regulation. Feldman & Friston (2010) incorporated precision as dynamically regulated by expected volatility, making threshold modulation implicit in predictive processing. APGI draws on all three frameworks but applies them to a domain none address: the ignition threshold for conscious access. The formalization

θₜ₊₁ = θₜ + η(C_metabolic − V_information)

specifies the threshold as an actively regulated, predictive variable where C_metabolic is the anticipated metabolic cost and V_information is the predicted information value of ignition. Both terms are operationalized in the full framework paper with measurement protocols (C_metabolic indexed via pupillometry and heart rate; V_information estimated from trial-by-trial prediction error magnitude). This implements the allostatic extension: the brain anticipates costs and adjusts the access criterion preemptively, making the threshold for consciousness itself an allostatic variable rather than a structural constant.

### Connection to App Implementation (Innovation 2)

The allostatic modulation of the ignition threshold is implemented in the `threshold_dynamics` method in `APGI-Full-Dynamic-Model.py` and in the agent's step function in `Falsification/Falsification-Protocol-1.py`. The threshold adapts dynamically as a predictive variable, with the equation dtheta_dt = (self.theta_0 - self.theta_t) / self.tau_theta + self.eta_theta * (self.metabolic_cost - self.information_value), incorporating anticipated metabolic cost and information value to preemptively adjust the access criterion.

### Relationship to Prior Work (Innovation 2)

Aston-Jones & Cohen (2005) provide the adaptive gain logic for neural thresholds; Yu & Dayan (2005) provide the cost-benefit logic for neuromodulatory regulation; Feldman & Friston (2010) provide implicit threshold modulation via precision. APGI's contribution is applying this allostatic logic specifically to conscious access thresholds with an explicit cost-benefit formalization—a domain-specific advance that none of the precursor frameworks address directly.

## 3. The Three-Level Epistemic Architecture with Explicit Bridge Principles

The division of all APGI claims into three rigorously separated levels—thermodynamic (joules/kelvin), information-theoretic (bits/mutual information), computational (algorithms/circuits)—extends Marr's (1982) canonical three levels of analysis (computational, algorithmic, implementational), which established the foundational insight that distinct levels of analysis require distinct evidentiary standards. Anderson (2014, After Phrenology) developed bridging principles between levels in cognitive neuroscience, and Dayan & Abbott (2001) formalized mappings between biophysical, circuit, and computational levels in theoretical neuroscience. Prior efforts to bridge levels in consciousness science—including Tononi et al.'s (2016) IIT, which attempts to connect information-theoretic and physical claims—establish that the aspiration is not new, but none provide a self-applicable scoring rubric with explicit falsification criteria across all three levels.

APGI's epistemic architecture goes beyond these precedents in three specific ways: (1) it incorporates thermodynamic grounding as a foundational constraint, acknowledging that biological cognition operates under non-negotiable energy budgets—a dimension absent from Marr's scheme; (2) it requires mathematically specified bridge principles for cross-level inference, notably Landauer's principle connecting computational to thermodynamic claims and variational inference connecting computational to information-theoretic claims; and (3) it provides a scoring rubric and falsification framework that makes the architecture practically applicable to theory evaluation, including self-application to APGI's own claims.

This converts previously untestable efficiency claims into claims with defined disconfirmation criteria. The architecture is a methodological innovation with consequences for how the field evaluates theories—applied impartially to APGI itself, revealing that APGI's Level 3 (computational) claims are strong, Level 2 (information-theoretic) claims are partial, and Level 1 (thermodynamic) claims remain unmeasured.

### Connection to App Implementation (Innovation 3)

The three-level epistemic architecture is implemented in the `MultiLevelEntropyModule` class in `APGI-Entropy-Implementation.py`, which computes entropy at three distinct levels: thermodynamic (using `ThermodynamicEntropyCalculator` for joules/kelvin constraints), information-theoretic (using `ShannonEntropyCalculator` for bits/mutual information), and variational (using `VariationalFreeEnergyCalculator` for Bayesian model fitting). This module provides bridge principles like Landauer's principle for computational-thermodynamic mappings and includes a self-applicable scoring rubric with falsification criteria.

### Relationship to Prior Work (Innovation 3)

Marr (1982) provides the foundational three-level insight. Tononi et al. (2016) and other consciousness theorists have attempted level-bridging. APGI's contribution is extending this to consciousness science with thermodynamic grounding, mathematically specified bridge principles, and a self-applicable falsification rubric—innovations absent from Marr's original scheme and from subsequent bridging efforts in the field.

## 4. The Cortex-as-Liquid-Computer with Threshold as Attractor-Basin Bifurcation Point

Deco & Jirsa (2012) and Deco et al. (2013) explicitly modeled ignition as bifurcation in whole-brain dynamics with critical slowing and hysteresis. Dehaene & Changeux (2011) described ignition as a nonlinear phase transition with avalanche dynamics. Werner (2010) formalized brain dynamics near criticality with bifurcation structure underlying state transitions. These prior accounts established that ignition dynamics can be characterized through dynamical systems theory, using mean-field models and neural mass equations as the substrate.

APGI's advance is the specific mapping of the ignition threshold onto the mathematical structure of bifurcation theory within a cortical reservoir computing framework—not as a parametric setting or attention-weighted value, but as a phase-transition boundary in the ODE state space of liquid neural networks (via Hasani et al.'s continuous-time architecture applied to APGI). This substrate-specific mapping generates predictions about the relationship between precision-weighted prediction error accumulation and bifurcation dynamics that prior accounts—which model bifurcation in terms of general neural mass dynamics or mean-field parameters—do not produce. Specifically, the continuous-time ODE formulation generates testable predictions about bistability as a function of precision weighting, hysteresis as a function of threshold adaptation rate, and critical slowing as a function of metabolic state—predictions that link APGI-specific parameters to measurable dynamical signatures. The 'cortex-as-liquid-computer' framing is understood as an analogical framework whose quantitative correspondence to cortical circuits requires empirical validation; it is not claimed as an established finding.

### Connection to App Implementation (Innovation 4)

The cortex-as-liquid-computer with threshold as attractor-basin bifurcation point is implemented in `APGI-Liquid-Network-Implementation.py`, where liquid time-constant networks (LTCNs) model cortical dynamics with adaptive time constants modulated by precision weights. The bifurcation structure is realized through phase transitions in the ODE state space of the liquid neurons, generating predictions about bistability, hysteresis, and critical slowing linked to precision-weighted prediction error accumulation and threshold adaptation rate.

### Relationship to Prior Work (Innovation 4)

Deco & Jirsa (2012) and Deco et al. (2013) established bifurcation modeling of ignition in whole-brain mean-field dynamics. APGI's contribution is mapping bifurcation specifically to precision-weighted prediction error accumulation within a liquid network ODE substrate, generating parameter-specific predictions absent from prior dynamical systems accounts of consciousness.

## 5. Self-Similar APGI Computation Across Hierarchical Timescales as Explanation of Temporal Depth

A substantial body of prior work establishes the hierarchical temporal structure that APGI builds upon. Friston et al. (2017) formalized deep temporal models where the same inferential scheme applies at each hierarchical level with timescales spanning milliseconds to years. Kiebel et al. (2008) derived hierarchical timescales from free energy minimization, showing how slower levels generate predictions that contextualize faster levels. Hasson et al. (2008, 2015) empirically demonstrated hierarchical temporal receptive windows spanning 0.3s to >30s with nested integration properties. Murray et al. (2014) established the continuous gradient of intrinsic timescales from sensory to prefrontal cortex. Chaudhuri et al. (2015) demonstrated that a single continuum of timescales emerges from canonical microcircuit properties. Honey et al. (2012) showed temporal receptive windows form a smooth gradient matching anatomical hierarchy.

APGI does not claim to discover the hierarchical temporal gradient—this is empirically established. Its contribution is providing the first threshold-gated access mechanism within this established architecture. The claim that the identical gating logic (Π·|ε| > θₜ) operates at each level—with level-specific parameters but the same formal structure—converts five separate descriptive problems (specious present, narrative coherence, autobiographical identity, etc.) into a single mechanistic framework with a unified falsification criterion: if the threshold-gating mechanism fails at any level, the self-similar architecture requires revision. Prior multi-scale approaches describe temporal hierarchy descriptively (Varela, Pöppel) or invoke different mechanisms at different levels. APGI's specific contribution is the access mechanism—determining which prediction errors achieve conscious availability—not the hierarchy itself.

The five-level discretization is acknowledged as an analytical convenience derived from temporal coverage constraints (N_levels ≈ log₁₀(τ_max / τ_min) / log₁₀(overlap_factor), where the overlap factor is a free parameter requiring explicit justification in future work), not a claim about discrete neural compartments. The underlying implementation reflects continuous gradients, as demonstrated by Murray et al. (2014) and Watanabe et al. (2022).

### Connection to App Implementation (Innovation 5)

The self-similar APGI computation across hierarchical timescales is implemented through the HierarchicalGenerativeModel in Falsification/Falsification-Protocol-1.py, with level-specific time constants (tau) ranging from 0.05s to 2.0s across sensory, organ, and homeostatic levels. The threshold-gated access mechanism (Π·|ε| > θ_t) is applied at each level with level-specific parameters, converting the established temporal hierarchy into a unified mechanistic framework for temporal depth.

### Relationship to Prior Work (Innovation 5)

Friston et al. (2017), Kiebel et al. (2008), and Hasson et al. (2008, 2015) establish the self-similar hierarchical architecture and empirical temporal gradient. APGI's contribution is the threshold-gated access mechanism operating within this established hierarchy, specifying the conditions under which prediction errors at each timescale achieve conscious availability.

## Major Advances

Significant extensions or formalizations of existing ideas that substantially increase specificity, testability, or explanatory power.

## 6. Cross-Modal Z-Score Standardization as Solution to the Commensurability Problem

The problem of comparing interoceptive and exteroceptive prediction errors is non-trivial: a heartbeat deviation and a visual contrast deviation have different variance structures and are numerically incommensurable. Prior frameworks assume commensurability without justification. APGI explicitly solves this through modality-specific z-scoring (zᵉ = (εᵉ − μ_εᵉ)/σ_εᵉ; zⁱ = (εⁱ − μ_εⁱ)/σ_εⁱ) with running estimates of mean and variance, yielding dimensionless, comparable quantities. Z-scoring is a standard normalization technique; the contribution here is its explicit application to a problem prior predictive processing frameworks have ignored rather than solved. This is a technical innovation with direct empirical consequences—it specifies how to measure the relative contributions of interoceptive and exteroceptive signals in a single paradigm.

### Connection to App Implementation (Innovation 6)

The cross-modal z-score standardization is implemented in the `APGINormalizer` class in `APGI-Multimodal-Integration.py`, which performs modality-specific z-scoring with running estimates of mean and variance for interoceptive (HEP, SCR, heart_rate) and exteroceptive (gamma_power, P3b_amplitude, pupil_diameter) modalities, yielding dimensionless, comparable prediction error magnitudes for multimodal integration.

## 7. Seven Standards for Viable Consciousness Theories as a Self-Regulatory Framework

The proposal of seven explicit, ranked methodological standards—Level Transparency, Bridge Principles, Quantitative Benchmarks, Falsification Conditions, Alternative Comparison, Evolutionary Plausibility, and Causal Roadmap—applied to APGI's own claims constitutes a contribution to the metascience of consciousness research independent of APGI's content. Prior proposals for standards in consciousness science exist (see Doerig et al., 2021, Nature Reviews Neuroscience, for an explicit treatment of criteria for consciousness theories), but no prior paper has applied a self-specified, ranked, multi-criterion methodological audit to its own framework as part of the main manuscript. The standards convert implicit reviewing criteria into explicit, applicable benchmarks that competing theories can also be evaluated against. This is a methodological advance in field practice, not a theoretical advance in mechanism.

### Connection to App Implementation (Innovation 7)

The seven standards for viable consciousness theories are implemented as validation criteria in the APGI validation suite, with self-application demonstrated in the Falsification/ and Validation/ directories. Each protocol includes quantitative benchmarks, falsification conditions, and alternative comparisons, converting the standards into operational validation tools.

## 8. APGI-LNN Mapping Table: Five Structural Correspondences Between Computational Constructs and ODE Parameters

The systematic mapping between APGI abstractions and Liquid Neural Network parameters—precision weighting ↔ liquid time constant (τ) modulation; allostatic threshold modulation ↔ state-dependent τ function; prediction error accumulation ↔ τ-modulated accumulator dynamics; ignition threshold ↔ bifurcation point—provides more biological grounding than most prior computational consciousness theories. It converts conceptual correspondences (which many theories claim) into structurally motivated ones with identifiable parameters and testable predictions about substrate dynamics. These correspondences are understood as structural analogies supported by formal analysis; they constitute mathematical identity only where explicitly derived in the supplementary equations. Future work should provide formal proofs of equivalence for each claimed correspondence.

### Connection to App Implementation (Innovation 8)

The APGI-LNN mapping table is implemented in `APGI-Liquid-Network-Implementation.py`, with precision weighting mapped to liquid time constant (τ) modulation via `tau_intero_baseline` and `tau_extero_baseline`, allostatic threshold modulation to state-dependent τ functions, prediction error accumulation to τ-modulated accumulator dynamics, and ignition threshold to bifurcation points in the ODE state space.

## 9. 1/f Spectral Slope as Quantitative Prediction of Hierarchical APGI Architecture

The 1/f structure of awake EEG is well-established, and prior work has provided mechanistic accounts. He et al. (2010) explicitly derived 1/f scaling from multi-scale temporal integration processes in cortex. Linkenkaer-Hansen et al. (2001) demonstrated 1/f structure reflects long-range temporal correlations in neuronal avalanches. Buzsáki & Mizuseki (2014) comprehensively reviewed how nested oscillations produce 1/f spectra through superposition.

APGI's contribution is not deriving 1/f dynamics de novo but specifying which regulatory processes—precision weights and allostatic thresholds shaped by nested regulatory loops spanning milliseconds, minutes, and hours—generate the 1/f signature within a consciousness-specific framework. The framework predicts that the aperiodic exponent β (as measured with FOOOF/specparam; Donoghue et al., 2020) should be approximately 0.8 -- 1.2 during wakefulness and increase toward β ≈ 1.5 -- 2.0 in deep sleep. These directional predictions have cross-state falsification criteria that go beyond prior mechanistic accounts.

A critical correction applies here: earlier versions of this document predicted β ≈ 0 under general anesthesia. Published data contradict this. Donoghue et al. (2020, Nature Neuroscience) and related work report that propofol anesthesia increases rather than decreases the aperiodic slope, with values typically increasing to β ≈ 2 -- 2.5. The corrected APGI prediction is therefore that anesthesia produces an elevated aperiodic exponent (β > 1.5), reflecting the collapse of higher-level fast oscillatory activity while slower components persist—not a flat spectrum. This revision also changes the falsification criterion: an absence of aperiodic slope increase under anesthesia (or a slope decrease) would falsify this component of the hierarchical architecture.

### Connection to App Implementation (Innovation 9)

The 1/f spectral slope predictions are implemented through the hierarchical generative models in Falsification-Protocol-1.py, where nested regulatory loops with precision weights and allostatic thresholds across timescales generate the predicted aperiodic exponents (β ≈ 0.8-1.2 wakefulness, β ≈ 1.5-2.0 deep sleep, β > 1.5 anesthesia), with falsification criteria based on FOOOF/specparam measurements.

### Relationship to Prior Work (Innovation 9)

He et al. (2010) provides the prior mechanistic derivation of 1/f scaling from multi-scale integration. Donoghue et al. (2020) provides the methodological framework (FOOOF) for reliable aperiodic exponent estimation. However, the specific findings that general anesthesia (propofol) increases the spectral slope (β ≈ 2.0–2.5) should be attributed to Gao et al. (2017), Lendner et al. (2020), and Colombo et al. (2019). APGI's contribution is the specific directional predictions about spectral slope values across consciousness states, linked to precision and threshold parameters, with corrected and empirically grounded state-specific values.

## 10. Psychiatric Disorders Recharacterized as Specific Precision/Threshold Dysregulation Profiles

Carhart-Harris & Friston (2019) in the REBUS model explicitly frame depression and psychosis as hierarchical dysregulation of high-level priors—establishing the principle that psychiatric disorders reflect level-specific disruptions within predictive processing. Stephan et al. (2009) formalized schizophrenia as impaired hierarchical integration via the dysconnection hypothesis. These frameworks provide the conceptual foundation for understanding psychiatric pathology through hierarchical predictive processing.

APGI extends this prior work with a dimensional characterization specifying both which APGI parameter is dysregulated and at which hierarchical level: panic as pathologically elevated Πⁱ with maladaptive negative M(c,a) producing catastrophic interoceptive ignition; depression as blunted Πⁱ and elevated θₜ producing reduced access to internal states—consistent with interoception-depression literature (Paulus & Stein, 2010; Farb et al., 2011); schizophrenia (positive) as aberrant Π assigning salience to noise; PTSD as hyper-precise somatic markers for trauma contexts. The two-axis diagnostic system (parameter type × hierarchical level) generates disorder-specific biomarker predictions.

Quantitative threshold predictions are presented as strong-form falsifiable targets: depression as Level 3–4 rigidity with mPFC-hippocampal hyperconnectivity ≥50% above controls combined with Level 1–2 disconnection with MMN amplitude ≥30% below controls. Note that published meta-analyses (e.g., Mulders et al., 2015) report connectivity differences on the order of 20–35%, meaning these thresholds are deliberately demanding—they represent the strong form of the prediction and will require large, well-powered samples. The MMN reduction in depression has mixed empirical support (some studies confirm it, others do not); this is acknowledged as a contested biomarker pending replication. These values should be treated as pre-specified targets for empirical refinement rather than as validated findings.

### Connection to App Implementation (Innovation 10)

The psychiatric disorder recharacterization is implemented in APGI-Psychological-States.py, with specific parameter profiles for GAD (elevated β ≈0.7), MDD (blunted Πⁱ, elevated θ_t), Psychosis (aberrant Π), and PTSD (hyper-precise somatic markers), providing quantitative biomarker predictions for hierarchical level-specific dysregulation.

### Relationship to Prior Work (Innovation 10)

Carhart-Harris & Friston (2019) REBUS model provides the hierarchical dysregulation framework. APGI's contribution is specifying the precise APGI parameters involved at each level with quantitative biomarker predictions, extending REBUS with precision-gated thresholds and the dimensional parameter × level diagnostic system.

## 11. Ignition as Phase Transition in Cortical Reservoir: Bistability, Critical Slowing, Hysteresis Predictions

Building on the bifurcation framework established in Innovation 4 and on prior dynamical systems characterizations (Deco & Jirsa, 2012; Werner, 2010), this contribution specifies a cluster of testable critical phenomena linked to APGI-specific parameters: bistable firing rates (bimodal distribution, not normal), critical slowing near threshold (increased variance and autocorrelation time as threshold approaches), and hysteresis (different threshold crossing values for ascending versus descending stimulus strength). Each signature can be tested independently with intracranial EEG, and each has a precise disconfirmation criterion (graded linear responses would falsify bistability; absence of variance increase would falsify critical slowing). Testing hysteresis requires two-directional stimulus paradigms in which participants experience both ascending and descending stimulus intensity series; experimental designs for this are specified in Protocol 6 of the main framework paper. The advance over prior phase-transition accounts is the link to precision-weighted prediction error dynamics rather than general neural mass parameters.

### Connection to App Implementation (Innovation 11)

The phase transition predictions are implemented in the liquid network dynamics of APGI-Liquid-Network-Implementation.py, where precision-weighted prediction error accumulation leads to bistable firing rates, critical slowing near thresholds, and hysteresis in the ODE state space, with testable signatures for intracranial EEG.

## 12. Cross-Level Bidirectional Coupling Formalized as Coupled Differential Equations

The formalization of top-down precision modulation (higher levels set precision weights for lower levels) and bottom-up error propagation (lower-level prediction errors can breach higher-level thresholds when sufficiently amplified) as coupled differential equations—rather than as conceptual arrows—enables quantitative prediction of cross-level dynamics. Varela et al. (2001) described neural synchrony and binding as mechanisms for integration; Tononi's IIT addresses information integration across neural populations. APGI adds temporal nesting within a threshold-gated framework, specifying how a surprising percept forces narrative revision (bottom-up Level 1 error breaches Level 3 threshold), why emotional context shapes perception (Level 3 prior modulates Level 1 precision), and how trauma creates pervasive perceptual hypervigilance (hyper-precise Level 4 somatic markers propagate down through the hierarchy). The full coupled differential equation system is presented in Section 4 of the multi-scale consciousness paper; the innovations list records the conceptual advance—converting previously descriptive phenomena into quantitatively tractable ones—with PAC-based empirical signatures for each predicted coupling direction.

### Connection to App Implementation (Innovation 12)

The cross-level bidirectional coupling is implemented as coupled differential equations in the HierarchicalGenerativeModel of Falsification-Protocol-1.py, formalizing top-down precision modulation and bottom-up error propagation across sensory, organ, and homeostatic levels with quantitative predictions for PAC signatures.

### Relationship to Prior Work (Innovation 12)

Varela et al. (2001) provide the integration concept via neural synchrony. APGI's contribution is the coupled differential equation formalization that enables quantitative prediction of specific cross-level dynamics within the threshold-gated architecture.

## Significant Contributions

Genuine advances that strengthen the framework's scope, testability, or clinical application, building on established ideas with meaningful new specificity.

## 13. Cardiac Phase-Dependent Detection Rate as Discriminating Test Between APGI and Standard GWT

The prediction that near-threshold visual stimuli presented during high HEP phases (250–400ms post-R-wave) should show 15–20% higher detection rates than during low HEP phases is specifically predicted by APGI but not generated by standard GWT. Standard GWT generates no cardiac-phase predictions because it does not include interoceptive precision gating. A proponent of attention-extended GWT could post-hoc attribute a cardiac-detection link to attentional gating, so the test discriminates APGI most cleanly from the standard (attention-as-sole-gate) formulation. It converts the theoretical distinction (interoceptive precision gates exteroceptive access, versus attention as the only gate) into a single, tractable experiment with pre-specified quantitative criteria (cardiac modulation >12% supports APGI; <8% falsifies it as functionally significant). See also Park et al. (2014, Current Biology) for relevant empirical precedent linking cardiac phase to visual detection.

### Connection to App Implementation (Innovation 13)

The cardiac phase-dependent detection rate prediction is implemented in the multimodal integration of APGI-Multimodal-Integration.py, where interoceptive precision gating modulates exteroceptive access, predicting 15-20% higher detection rates during high HEP phases, discriminable from standard GWT via HEP amplitude measurements.

## 14. Somatic Marker Modulation of Precision (Πⁱ), Not Prediction Error (εⁱ), as Distinguishing Claim

The explicit distinction between two ways somatic markers could work—modulating the amplitude of interoceptive prediction errors directly (εⁱ) versus modulating the precision assigned to those errors (Πⁱ)—is theoretically significant because these produce different anatomical predictions. If markers modulate precision, vmPFC activity should correlate with anticipatory insula activation (gain modulation) but not with primary interoceptive error signals in posterior insula. This prediction distinguishes APGI from accounts where emotion simply amplifies sensory signals rather than changing their reliability weighting, and is grounded in the established vmPFC–anterior insula anatomy (Critchley et al., 2004; Seth et al., 2012). This prediction distinguishes APGI from accounts where emotion simply amplifies sensory signals rather than changing their reliability weighting.

### Connection to App Implementation (Innovation 14)

The somatic marker modulation of precision is implemented in the `compute_somatic_modulation` method of APGI-Multimodal-Integration.py, where M(c,a) modulates Πⁱ_eff rather than εⁱ, predicting vmPFC correlations with anticipatory insula activation for precision weighting.

## 15. Backward Masking, Attentional Blink, and Binocular Rivalry Reinterpreted as Liquid Network Phenomena

The reinterpretation of three classic paradigms through reservoir dynamics—backward masking as a probe of reservoir decay rates; attentional blink as saturation within the ~500ms integration window (consistent with Sergent & Dehaene, 2005 ignition data); binocular rivalry as precision-weighted competition between reservoir states (consistent with Hohwy et al., 2008)—converts descriptive observations into quantitative tests. Each reinterpretation generates a specific, measurable prediction about reservoir parameters that can be tested against empirical data, and each would falsify a specific aspect of the liquid network account if the predicted parameter values proved inconsistent with measured cortical dynamics. Each paradigm has well-developed prior computational accounts (rivalry: Moreno-Bote et al., 2010; attentional blink: Olivers & Meeter, 2008); the APGI advance lies in unifying them under a single substrate model with shared parameters, rather than requiring paradigm-specific mechanisms.

### Connection to App Implementation (Innovation 15)

The reinterpretation of backward masking, attentional blink, and binocular rivalry is implemented in the liquid network dynamics of APGI-Liquid-Network-Implementation.py, where reservoir decay rates, saturation within integration windows, and precision-weighted competition generate quantitative predictions for each paradigm's parameters.

## 16. Phase-Amplitude Coupling as Neural Implementation of Inter-Level Hierarchical Precision

Bastos et al. (2012, 2015) in the canonical microcircuits framework explicitly mapped theta-phase modulation of gamma amplitude to precision-weighted prediction error signaling—establishing the mechanistic link between PAC and hierarchical precision. Friston et al. (2015, Active inference and epistemic value, Journal of Neuroscience) formalized how oscillatory hierarchies implement precision in predictive coding. Jensen & Lisman (2005) established theta-gamma coupling as a mechanism for maintaining multiple items in working memory.

APGI builds on these established PAC-precision mappings by extending them to conscious access thresholds within the hierarchical architecture, specifying level-specific coupling signatures: slow oscillations (delta/theta) carrying higher-level contextual priors modulate the amplitude of fast oscillations (gamma) encoding lower-level prediction errors, with theta-gamma PAC for Level 1–2 coupling and delta-theta PAC for Level 3–4 coupling. These level-specific predictions about coupling strength across psychiatric and cognitive states go beyond the general PAC-precision mapping to generate testable signatures of hierarchical conscious access.

### Connection to App Implementation (Innovation 16)

The phase-amplitude coupling predictions are implemented in the hierarchical generative models of Falsification-Protocol-1.py, specifying theta-gamma PAC for Level 1-2 coupling and delta-theta PAC for Level 3-4 coupling with quantitative predictions for coupling strength across states.

### Relationship to Prior Work (Innovation 16)

Bastos et al. (2012, 2015) provide the PAC-precision mapping. APGI's contribution is the level-specific extension to conscious access thresholds with quantitative predictions about coupling signatures across states.

## 17. Dissociation as Level Decoupling (Between-Level Coherence Reduced While Within-Level Coherence Preserved)

Van der Hart, Nijenhuis, & Steele (2006, The Haunted Self) provided the foundational structural dissociation theory proposing level-specific disconnection as the mechanism underlying dissociative phenomena. APGI extends this with a neural implementation specifying quantitative biomarker predictions: between-level EEG coherence ≥40% below normative values during dissociative episodes while within-level coherence remains intact (<15% deviation). This distinguishes dissociation from global hypoconnectivity (which would affect both between- and within-level coherence) and generates a pre-specified falsification criterion (within-level and between-level coherence failing to dissociate significantly, p < 0.01). The advance is the quantitative specificity and the mapping onto the hierarchical APGI architecture with measurable EEG signatures.

### Connection to App Implementation (Innovation 17)

The dissociation as level decoupling is implemented in the hierarchical architecture of Falsification-Protocol-1.py, predicting between-level EEG coherence ≥40% below norms while within-level coherence remains intact (<15% deviation) during dissociative episodes.

### Relationship to Prior Work (Innovation 17)

Van der Hart, Nijenhuis, & Steele (2006) provide the level-specific disconnection concept within structural dissociation theory. APGI's contribution is the neural implementation with quantitative coherence predictions.

## 18. Fractional Dimension of Threshold Dynamics as Behavioral Biomarker (1/f Perceptual Detection Paradigm)

The proposed continuous perceptual threshold paradigm—participants perform sustained near-threshold detection for 60–90 minutes, autocorrelation of the response time series computed across three orders of magnitude of lag—tests whether threshold fluctuations are Markovian (exponential autocorrelation decay) or fractal (power-law decay). This is an original experimental design directly testing the hierarchical architecture's core prediction (multi-level regulatory processes produce 1/f threshold dynamics) with standard equipment. It extends to clinical populations as a potential biomarker: reduced fractal dimension in depression (consistent with Linkenkaer-Hansen et al., 2005) and excessive fractal dimension approaching white noise in ADHD. The paradigm requires minimal equipment and can be implemented immediately with existing psychophysics software.

### Connection to App Implementation (Innovation 18)

The fractional dimension of threshold dynamics is implemented in the threshold dynamics of APGI-Full-Dynamic-Model.py, predicting fractal (power-law) autocorrelation decay in sustained near-threshold detection tasks, with reduced dimension in depression and excessive in ADHD.

## 19. APGI-Adaptive Threshold as Explanation for Flow States and Psychedelic Effects

Dietrich (2004) characterized flow as transient hypofrontality with optimized resource allocation—providing the phenomenological and neuroanatomical foundation. APGI adds a parameter-space formalization: flow as θₜ optimization (high enough to filter noise, low enough for rapid integration, with Sₜ hovering just above threshold to maximize information throughput) and psychedelic states as 5-HT2A-mediated flattening of the precision landscape (reducing the selectivity of ignition, consistent with Carhart-Harris et al., 2014 and the REBUS model of Carhart-Harris & Friston, 2019). The 'normally unconscious content' that becomes accessible under psychedelics is operationalized as prediction errors residing at sub-threshold levels across Levels 1–3 that fail to breach threshold during normal wakefulness; this operationalization yields testable predictions about which content types should be affected.

These generate measurable EEG predictions: flow should correlate with moderate alpha suppression and sustained beta-gamma coherence; psychedelics should increase P3b frequency while reducing P3b amplitude specificity (more events, less selective).

### Connection to App Implementation (Innovation 19)

The APGI-adaptive threshold explanations are implemented in the threshold dynamics of APGI-Full-Dynamic-Model.py and parameter profiles in APGI-Psychological-States.py, formalizing flow as θ_t optimization and psychedelics as precision landscape flattening with specific EEG predictions.

### Relationship to Prior Work (Innovation 19)

Dietrich (2004) provides the hypofrontality account of flow. Carhart-Harris & Friston (2019) provide the precision-flattening account of psychedelics via REBUS. APGI's contribution is the parameter-space formalization linking flow and psychedelic states to specific threshold and precision dynamics with novel EEG predictions within the APGI architecture.

## 20. Interoceptive Precision (HEP Amplitude) and Global Ignition (P3b/PCI) as Joint Biomarkers for Disorders of Consciousness

The proposal that HEP amplitude (proxy for Πⁱ, grounded in Sel et al., 2017) and perturbational complexity index (proxy for global integration/ignition capacity; Casali et al., 2013, Science Translational Medicine) together predict recovery in minimally conscious state patients—and that the combination outperforms either alone—is a specific clinical prediction with a pre-specified experimental design (prospective study, N=100, primary outcome: HEP+PCI jointly predict recovery vs. either alone, with pre-specified superiority margin). This is directly actionable as a clinical trial and represents the document's strongest immediate clinical translation contribution.

### Connection to App Implementation (Innovation 20)

The joint biomarkers are implemented in the multimodal integration of APGI-Multimodal-Integration.py and validation protocols in Validation/, with HEP amplitude as proxy for Πⁱ and PCI as proxy for global ignition capacity, predicting recovery in disorders of consciousness.

## 21. Evolutionary Derivation of Liquid Architecture from Biological Constraints

The systematic analysis of how five canonical biological constraints (finite ATP, slow conduction, stochastic synapses, embodied survival requirements, slow unreliable components) each select for specific features of the liquid network architecture (threshold filtering, predictive hierarchy, precision-weighted population coding, interoceptive prioritization, multi-timescale integration) provides an evolutionary justification that most consciousness theories lack entirely. The argument is not that liquid networks are efficient in an abstract sense but that they are the architecture predicted by biological constraints via convergent selection pressure.

This derivation demonstrates architectural compatibility with biological constraints rather than unique predictability: alternative architectures might also satisfy individual constraints. The specific advance is showing that liquid networks satisfy all five constraints simultaneously without requiring auxiliary mechanisms, whereas feedforward, recurrent, and transformer architectures require additional specialized components to meet the full set. This motivates the substrate choice empirically rather than by analogy.

### Connection to App Implementation (Innovation 21)

The evolutionary derivation is implemented in the liquid network architecture of APGI-Liquid-Network-Implementation.py, where features like threshold filtering, predictive hierarchy, and precision-weighted coding are selected by biological constraints (finite ATP, slow conduction, stochastic synapses).

## 22. Developmental Trajectory of Hierarchical Level Emergence from Level 1 to Level 5

The prediction that hierarchical levels emerge in a specific developmental sequence—Level 1 present from birth, Level 2 stabilizing through early childhood, Level 3 emerging through hippocampal-prefrontal connectivity development (ages ~5–9, informed by Shing et al., 2010 and Ghetti et al., 2010 on hippocampal-PFC maturation), Level 4 reaching functional maturity in adolescence—generates a testable developmental program. This trajectory is informed by and should be integrated with existing empirical findings: Váša et al. (2018) demonstrated developmental lengthening of intrinsic timescales from childhood to adulthood, providing direct validation of the gradient structure. Gao et al. (2015) showed hierarchical organization of functional connectivity emerges gradually through development with an anterior-posterior maturation gradient. Fransson et al. (2011) documented emergence of functional network organization in infancy, with adult-like structure appearing earlier than some theoretical accounts predict.

APGI's specific contribution is mapping these established empirical trajectories onto the threshold-gated access mechanism: children should show fewer effective levels with weaker inter-level coupling, manifest as shorter sustained attention, reduced metacognition, and more labile self-concept, with transitions trackable via intrinsic timescale lengthening on resting-state fMRI. The developmental trajectory of Level 5 is not yet specified and represents a gap for future theoretical work. These age estimates should be treated as working hypotheses calibrated against the empirical developmental literature rather than fixed claims.

### Connection to App Implementation (Innovation 22)

The developmental trajectory is implemented in the hierarchical generative models of Falsification-Protocol-1.py, predicting level emergence sequence from Level 1 at birth to Level 4 in adolescence, with transitions trackable via intrinsic timescale lengthening.

### Relationship to Prior Work (Innovation 22)

Váša et al. (2018), Gao et al. (2015), and Fransson et al. (2011) establish the empirical developmental trajectory. APGI's contribution is mapping this trajectory onto threshold-gated access predictions. Level age estimates require calibration against the empirical evidence that hierarchical organization may emerge earlier than initially proposed.

## 23. Meta-Consciousness as Level ℓ+1 Modeling Level ℓ Dynamics

Fleming & Dolan (2012, Trends in Cognitive Sciences) formalized metacognition as higher-order inference—establishing the computational structure of metacognitive monitoring. APGI extends this with a specific hierarchical implementation: meta-consciousness as an emergent property when Level ℓ+1 develops generative models of Level ℓ's ignition dynamics (rather than merely integrating its outputs). This is distinct from standard second-order theories of consciousness in that it requires the higher level to model the threshold-crossing dynamics of the lower level, not merely its contents. This predicts that metacognitive accuracy should correlate with the strength of specific inter-level coupling (Level 3 modeling of Level 2 ignition frequency), generating a testable neuroimaging prediction about the neural correlates of accurate metacognitive judgment that goes beyond Fleming & Dolan's general higher-order inference framework.

### Connection to App Implementation (Innovation 23)

The meta-consciousness implementation is in the hierarchical architecture of Falsification-Protocol-1.py, where Level ℓ+1 develops generative models of Level ℓ's ignition dynamics, predicting metacognitive accuracy correlations with inter-level coupling strength.

### Relationship to Prior Work (Innovation 23)

Fleming & Dolan (2012) provide the higher-order inference formalization of metacognition. APGI's contribution is the specific hierarchical implementation mapping metacognition to inter-level coupling strength with testable neuroimaging predictions.

## Incremental and Methodological Contributions

Refinements, operationalizations, and applications that extend the framework's reach without introducing new core ideas.

## 24. Six Pre-Registered Empirical Protocols with Explicit Power Analyses, Effect Sizes, and Pre-Specified Falsification Criteria

Unlike most theoretical consciousness papers that offer qualitative predictions, APGI specifies quantitative criteria for confirmation and disconfirmation: cardiac modulation >12% for interoceptive gating, PCI reduction of 15–25% for TMS disruption, bistable response distributions near threshold. This operationalization converts theoretical claims into a concrete experimental program. The pre-specified criteria prevent post-hoc reinterpretation of null results as partial support and represent a methodological advance relative to the consciousness theory literature, where predictions are typically stated too loosely to be falsified.

### Connection to App Implementation (Innovation 24)

The pre-registered empirical protocols are implemented in the Validation/ directory, with protocols like Validation-Protocol-1.py through Validation-Protocol-12.py, each including explicit power analyses, effect sizes, and pre-specified falsification criteria.

## 25. APGI Clinical Assessment Battery (APGI-CAB): Proposed Multimodal Diagnostic Instrument

This item describes a proposed multimodal diagnostic instrument concept, not a validated clinical tool. HEP + PCI + behavioral interoceptive tasks are combined as the assessment architecture targeting psychiatric populations (anxiety, depression, PTSD, autism) with the specific aim of validating APGI parameter profiles as clinically meaningful biomarkers. Validation requires sequential steps: (1) psychometric validation of each component individually, (2) normative data collection across clinical populations, (3) test-retest reliability, and (4) predictive validity against clinical outcomes. The contribution at present is the instrument architecture and validation roadmap, which provides a more concrete translational pathway than most theoretical frameworks offer.

### Connection to App Implementation (Innovation 25)

The APGI Clinical Assessment Battery is proposed in the validation framework of APGI-Validation-GUI.py and Validation-Protocol-*.py, combining HEP, PCI, and behavioral interoceptive tasks for multimodal diagnostic assessment targeting psychiatric populations.

## 26. Parameter Estimation Workflow

Behavioral constraints confining parameter ranges through simulation of detection rates, ignition frequencies, and RT distributions. Initial identifiability analyses are based on synthetic data and suggest θ₀ and τ_S are well-identified (r > 0.80), Πⁱ and β are moderately identified with independent manipulations (r ≈ 0.65–0.75), and α is poorly identified without large trial counts (r ≈ 0.55). These are simulation results; real-world identifiability will likely be lower, particularly given neural measurement noise. The workflow specifies the experimental manipulations required to achieve parameter separation—interoceptive training for Πⁱ, body-focus instructions for β—providing an operational bridge from theory to data collection.

### Connection to App Implementation (Innovation 26)

The parameter estimation workflow is implemented in APGI-Parameter-Estimation.py, using behavioral constraints from simulation of detection rates, ignition frequencies, and RT distributions to confine parameter ranges with identifiability analyses.

## 27. Continuous-Time Dynamical Equations (dSₜ/dt, dθₜ/dt)

Distinguishes surprise accumulation from threshold adaptation as separate processes with distinct time constants (λ_S ≈ 2–5 s⁻¹ versus λ_θ ≈ 0.01–0.1 s⁻¹). The order-of-magnitude separation between these time constants is consistent with the behavioral literature on perceptual dynamics (fast, seconds scale) versus allostatic regulation (slow, minutes scale). This separation enables independent experimental manipulation—λ_S can be targeted via stimulus rate manipulations; λ_θ via fatigue, arousal, or pharmacological protocols—and provides the temporal dynamics missing from static threshold models.

### Connection to App Implementation (Innovation 27)

The continuous-time dynamical equations are implemented in APGI-Full-Dynamic-Model.py, distinguishing surprise accumulation (dS_t/dt) from threshold adaptation (dθ_t/dt) with distinct time constants (λ_S ≈ 2–5 s⁻¹ vs λ_θ ≈ 0.01–0.1 s⁻¹).

## 28. Full Neuromodulatory Implementation of Each LNN Parameter

Precision weighting to acetylcholine/NMDA coupling; threshold to norepinephrine/LC dynamics; interoceptive integration to insula-VAN circuit; somatic bias to vmPFC-insula asymmetric projection. These neuromodulatory roles are individually well-established in the literature; the contribution is their systematic integration with LNN parameters. These are presented explicitly as testable hypotheses with evidence strength ratings, not established mechanisms. Future pharmacological and lesion studies are required to test whether manipulating each neuromodulator produces the predicted APGI-parameter-specific effects.

### Connection to App Implementation (Innovation 28)

The full neuromodulatory implementation is in APGI-Liquid-Network-Implementation.py, mapping precision weighting to acetylcholine/NMDA coupling, threshold to norepinephrine/LC dynamics, interoceptive integration to insula-VAN circuit, and somatic bias to vmPFC-insula asymmetric projection.

## 29. Comparative Analysis of Liquid Networks versus Alternative Architectures

Comparison against feedforward, recurrent, and transformer architectures on five properties relevant to consciousness (sharp thresholds, temporal integration, metabolic selectivity, fading memory, multi-timescale dynamics), demonstrating that liquid networks uniquely exhibit all five intrinsically rather than through training. This motivates the substrate choice beyond analogy. The transformer comparison is noted as requiring updating: architectural developments in energy efficiency since 2022 (sparse attention mechanisms, mixture-of-experts) have reduced some efficiency gaps. The metabolic selectivity comparison specifically should be revisited against current transformer architectures in the final paper version.

### Connection to App Implementation (Innovation 29)

The comparative analysis is implemented in APGI-Liquid-Network-Implementation.py, demonstrating liquid networks' intrinsic properties (sharp thresholds, temporal integration, metabolic selectivity, fading memory, multi-timescale dynamics) versus alternative architectures requiring additional mechanisms.

## 30. Depression Recharacterized at the Hierarchical Level

Level 3–4 rigidity (mPFC-hippocampal hyperconnectivity ≥50% above controls) combined with Level 1–2 disconnection (MMN amplitude ≥30% below controls), with PAC ≥2 SD above norms. Published meta-analyses (Mulders et al., 2015, Neuroscience & Biobehavioral Reviews) report connectivity differences of approximately 20–35%, and the evidence for MMN reduction in depression is mixed. These thresholds are therefore demanding—they represent the strong-form prediction that will require well-powered replication to confirm or revise. The PAC elevation prediction is genuinely novel and represents an untested component. This quantitative inter-level signature extends the REBUS model (Carhart-Harris & Friston, 2019) with precision-gated threshold predictions.

### Connection to App Implementation (Innovation 30)

The depression recharacterization is implemented in APGI-Psychological-States.py, with Level 3-4 rigidity (elevated θ_t) and Level 1-2 disconnection (blunted Πⁱ), extending REBUS with precision-gated threshold predictions.

## 31. Circadian Modulation of θₜ as Formally Incorporated Oscillatory Coupling

Cortisol morning peak lowering threshold for memory consolidation, evening melatonin rise elevating it for sensory processing. This converts chronobiological observations into parameter-space predictions with measurable neural signatures, linking APGI's threshold dynamics to established endocrine rhythms. The connection to sleep-dependent memory consolidation (cortisol effects on hippocampal function; Lupien & McEwen, 1997; Plihal & Born, 1997) is well-grounded. Diurnal variation in the ignition threshold provides a natural within-subjects manipulation for empirical studies, with cortisol sampling providing a simultaneous hormonal anchor.

### Connection to App Implementation (Innovation 31)

The circadian modulation is implemented in the threshold dynamics of APGI-Full-Dynamic-Model.py, with cortisol morning peaks lowering θ_t for memory consolidation and melatonin evening rises elevating it for sensory processing.

## 32. Ultradian ~90-Minute Rest-Activity Cycle Reinterpreted as Periodic Threshold Recalibration

The Basic Rest-Activity Cycle (BRAC), established by Kleitman (1963) and subsequently linked to waking attentional fluctuations (Peretz Lavie's work on perceptual alertness oscillations; Lavie & Scherson, 1981; Lavie, 1991) and neuromodulatory depletion (reviewed in Strogatz, 1986), provides the empirical grounding for this contribution. APGI adds precision/threshold formalization: sustained cognitive engagement depletes neuromodulator reserves (particularly norepinephrine, consistent with Aston-Jones & Cohen, 2005), gradually elevating the ignition threshold until performance degrades and rest becomes necessary. This predicts performance oscillations with ~90-minute periodicity in sustained vigilance tasks, testable via direct psychophysical measurement combined with salivary or plasma neuromodulator assays.

### Connection to App Implementation (Innovation 32)

The ultradian cycle is implemented in the threshold dynamics of APGI-Full-Dynamic-Model.py, with sustained cognitive engagement depleting neuromodulator reserves and elevating θ_t until rest recalibrates it.

### Relationship to Prior Work (Innovation 32)

Kleitman (1963) and Lavie (1991) provide the empirical BRAC foundation. Aston-Jones & Cohen (2005) provide the neuromodulatory mechanism. APGI's contribution is the threshold-parameter formalization linking ultradian rhythmicity to measurable ignition threshold dynamics.

## 33. Cross-Species Prediction

Simpler nervous systems should show fewer hierarchical levels, shorter intrinsic timescales, less complex cross-frequency coupling, and reduced behavioral flexibility—testable via comparative PCI measurement (building on Rosanova et al., 2012; Massimini et al., 2012) and behavioral paradigms in non-human primates. This provides evolutionary plausibility constraints and generates a comparative research program. The predictions are quantitatively anchored to the hierarchical architecture: each increment in hierarchical complexity should correspond to a measurable extension in intrinsic timescales and an increase in inter-level PAC complexity, providing a continuous rather than binary comparative consciousness measure.

### Connection to App Implementation (Innovation 33)

The cross-species prediction is implemented in the hierarchical architecture of Falsification-Protocol-1.py, predicting fewer levels, shorter timescales, and reduced PAC complexity in simpler nervous systems, testable via comparative PCI measurements.

## 34. Cultural Neuroscience Prediction

Linguistic temporal grammar and contemplative practices (meditation traditions) should systematically shape hierarchical organization, measurable as differences in inter-level coupling strength and level-specific threshold values across populations. The meditation component has empirical grounding: Berkovich-Ohana et al. (2012) on minimal self in meditators; Brewer et al. (2011) on default mode network changes with meditation. The linguistic grammar component is more exploratory. While linguistic relativity effects on neural processing have been documented in narrow domains (Winawer et al., 2007), predicting specific differences in level-specific threshold values across language groups is a large extrapolation from current cultural neuroscience evidence. This prediction is presented as an exploratory hypothesis rather than a specific quantitative test, and requires dedicated cross-cultural neuroimaging designs before it can be evaluated.

### Connection to App Implementation (Innovation 34)

The cultural neuroscience prediction is implemented in the hierarchical architecture of Falsification-Protocol-1.py, predicting meditation and linguistic temporal grammar effects on inter-level coupling strength and level-specific θ_t values.

## 35. APGI Multimodal Classifier: Proposed Stratification Tool for Psychiatric Diagnosis

The APGI Multimodal Classifier describes a proposed tool architecture, not a validated classifier. fMRI connectivity + pupillometry + EEG ignition markers + HRV + behavioral modeling are proposed as inputs for a machine-learning stratification system targeting psychiatric diagnosis beyond symptom categories. Machine learning stratification of psychiatric disorders beyond symptom categories is an active research area (see Drysdale et al., 2017, Nature Medicine, for a working example in depression subtypes). The APGI classifier extends this approach by grounding features in theoretically motivated parameters (Πⁱ, θₜ, β) rather than empirically derived clusters, providing a principled feature selection rationale. Empirical validation of the underlying parameter profiles is a prerequisite for the classifier; the contribution at present is the theoretically motivated feature architecture and the validation pathway.

### Connection to App Implementation (Innovation 35)

The APGI Multimodal Classifier is proposed in the validation framework of APGI-Validation-GUI.py and Validation-Protocol-*.py, using fMRI connectivity, pupillometry, EEG ignition markers, HRV, and behavioral modeling as inputs grounded in Πⁱ, θ_t, β parameters.
