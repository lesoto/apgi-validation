# APGI EMPIRICAL CREDIBILITY ROADMAP

Establishing the Allostatic Precision-Gated Ignition (APGI) framework as a credible scientific theory demands a multi-layered empirical program that satisfies the methodological standards of top-tier consciousness science — convergent neural signatures, causal interventions, quantitative model fits, open science infrastructure, and explicit falsification commitments. This roadmap organizes that program into four foundational empirical priorities, followed by eleven strategic credibility pillars.

APGI occupies a specific and deliberate position within the existing theoretical landscape. It is not a competitor to Friston's Free Energy Principle (FEP) and active inference framework — it is a constrained implementation of it. The FEP establishes that biological systems minimize variational free energy — the gap between predicted and actual sensory states — through hierarchical predictive processing, with precision-weighting (Π) as the mechanism by which the system modulates confidence in its own predictions (Friston, 2010). APGI accepts this machinery wholesale and adds two empirical commitments the FEP leaves underspecified: first, that conscious access corresponds specifically to the discrete event of suprathreshold ignition — the moment precision-weighted prediction error exceeds a dynamically adjusted threshold θₜ — rather than to free energy minimization per se; second, that θₜ is not a free parameter but is allosterically regulated by metabolic state, interoceptive afference, and circadian dynamics. In Fristonian terms, APGI is a theory of the conditions under which active inference crosses into global broadcast. This framing transforms Pillars 7 and 8 of this roadmap from departures into extensions: computational benchmarking tests whether APGI adds explanatory precision beyond the FEP's predictions, while pharmacological and metabolic dissections test the allostatic grounding of θₜ that the FEP itself does not specify.
APGI's relationship to Integrated Information Theory (IIT) is structurally different. IIT (Tononi, 2004; Tononi et al., 2016) locates consciousness in the substrate — specifically in the degree of irreducible causal integration (Φ) of a physical system — rather than in the computational dynamics of ignition. APGI makes no claim about substrate-level Φ. The two frameworks are not competing answers to the same question; they operate at different levels of Marr's hierarchy. IIT is an implementational theory; APGI is an algorithmic one. The empirically productive question is not which framework is correct, but whether suprathreshold ignition events in APGI correspond to high-Φ network states in IIT — a testable, falsifiable question addressed directly in Pillar 7. If they do not co-occur, one or both frameworks require revision; if they do, the convergence constitutes evidence that the algorithmic and implementational descriptions pick out the same underlying phenomenon from complementary angles.

---

## PART I — FOUR CORE EMPIRICAL PRIORITIES

### Priority 1: Convergent Neural Signatures Across Paradigms

Global Neuronal Workspace (GNW) theory is anchored to a reproducible neural signature: the late P3b event-related potential (~300–500 ms post-stimulus), frontoparietal BOLD coactivation, and long-range gamma-band synchronization during conscious access — each absent under matched unconscious conditions such as subliminal masking or inattentional blindness. APGI must establish an equivalent convergent signature, now parameterized by precision (Π), prediction error magnitude (|ε|), and dynamic threshold (θₜ).

Specifically, the framework must demonstrate that suprathreshold ignition events reliably produce:

- A P3b-like potential whose amplitude scales sigmoidally with Π × |ε|, not linearly with stimulus intensity
- Frontoparietal BOLD coactivation contingent on S(t) > θₜ, not merely on stimulus presence
- Theta-gamma phase-amplitude coupling that emerges specifically at threshold crossing, not before

Critically, subthreshold trials (S(t) < θₜ) must activate early sensory cortex while failing to engage frontoparietal networks — replicating GNW's local-vs-global dissociation but now with APGI's three-parameter specification.

**Actionable step:** Re-analyze published GNW masking datasets (e.g., Dehaene's backward-masking paradigms) using trial-wise APGI variables — pupil diameter as a proxy for Π, stimulus surprisal as a proxy for ε — and test whether P3b amplitude fits a sigmoidal function of Π × |ε| − θₜ significantly better than additive linear alternatives.

---

### Priority 2: Causal Manipulations That Selectively Disrupt Ignition Parameters

GNW achieved causal credibility through targeted interventions: transcranial magnetic stimulation (TMS) to frontal cortex at 200–300 ms post-stimulus abolishes the P3b and conscious report; anesthesia disrupts frontoparietal connectivity while leaving local sensory processing intact. APGI requires a parallel set of parametric causal tests that selectively perturb Π and θₜ independently.

The critical predictions are:

- TMS or transcranial alternating current stimulation (tACS) to dlPFC or posterior parietal cortex disrupts ignition specifically when applied near the threshold-crossing window (~200–300 ms), not at earlier or later intervals
- Pharmacological precision modulation — propranolol reducing interoceptive precision, atomoxetine augmenting norepinephrine-mediated gain — shifts psychometric detection thresholds predictably without altering early sensory ERPs (N1/P2)
- Metabolic challenge (glucose depletion, caloric restriction) elevates θₜ, raising detection thresholds, without affecting pre-conscious sensory encoding

The null prediction is as diagnostically important as the positive one: early ERPs should be invariant to precision and metabolic manipulations. Preregistering this null provides a powerful credibility signal.

**Actionable step:** Conduct a 2×2×2 factorial study crossing precision manipulation (atomoxetine vs. placebo), error magnitude (stimulus contrast), and TMS timing (threshold-crossing window vs. control). APGI predicts a three-way interaction on P3b amplitude and reportability; GNW predicts only a two-way interaction.

---

### Priority 3: Quantitative Model Fits to Behavioral Data

GNW explains the all-or-none phenomenology of conscious access, the ~200 ms masking window, and attentional modulation of threshold — but largely at the qualitative level. APGI's distinctive contribution is a quantitative, algebraically specified model that generates numerical predictions across paradigms.

The core behavioral fit is the psychometric function:

$$P(\text{seen}) = \frac{1}{1 + e^{-\beta(\Pi \cdot |\varepsilon| - \theta_t)}}$$

where β ≥ 10 indicates phase-transition dynamics rather than graded accumulation. The framework makes three additional quantitative commitments:

First, individual differences in θₜ estimated from psychometric functions should correlate with resting-state frontoparietal excitability — measurable via GABA concentrations (magnetic resonance spectroscopy) or TMS motor threshold. Second, the sigmoid steepness parameter β should be higher in experienced meditators (predicted enhanced precision modulation) than controls. Third, a single spiking leaky neural network (LNN) model with fixed architecture but tunable Π and θₜ should reproduce attentional blink, backward masking, and binocular rivalry within the same parameter space — without paradigm-specific re-fitting.

**Actionable step:** Publish an open-source APGI-LNN simulator capable of reproducing at least five canonical consciousness paradigms with fewer than five free parameters, and demonstrate superior fit over additive linear models using Bayesian model comparison (target: BF > 100 for the full APGI model over GNW-equivalent).

---

### Priority 4: Clinical and Cross-Species Convergence

GNW draws validation from three convergent domains: loss of the P3b and frontoparietal activation in vegetative state patients, preserved local processing in blindsight and anesthesia, and evolutionary plausibility demonstrated through primate single-unit electrophysiology. APGI must match and extend this clinical-comparative profile.

In disorders of consciousness (DoC), elevated θₜ or attenuated Π modulation — measurable as reduced trial-by-trial variability in late ERP amplitude — should distinguish vegetative state from minimally conscious state more sensitively than categorical P3b presence or absence. In anxiety disorders, hyperactive interoceptive precision weighting (reflected in insula hyperactivation) should manifest as a selectively lowered θₜ for somatic signals, producing measurable asymmetry in the psychometric functions for interoceptive versus exteroceptive stimuli.

In non-human primates and corvids — species with demonstrated capacity for sustained frontoparietal firing on reportable stimuli — ignition-like activity should be modulated by arousal state in the manner APGI specifies: metabolic depletion raising threshold, precision enhancement lowering it.

**Actionable step:** Partner with clinical neurology groups to apply APGI-derived EEG biomarkers — ignition slope, precision-weighted P3b variance, θₜ estimate from trial-by-trial psychometric fitting — as prognostic indices in DoC recovery trajectories, testing whether APGI parameters predict outcome more accurately than standard P3b presence-or-absence scoring.

---

## PART II — STRATEGIC CREDIBILITY PILLARS

### Pillar 1: Open Science and Preregistration

This is the single most asymmetric credibility investment available to APGI. Consciousness science has been damaged by replication failures, and any framework without transparent data infrastructure faces preemptive dismissal by reviewers at Nature, PNAS, and Neuron.

Before collecting data for any key prediction, submit the hypothesis, analysis pipeline, sample size with power calculation, and decision criteria to OSF or AsPredicted. The preregistration must specify the exact formula under test — e.g., P(seen) = 1/(1 + e^−β(Π·|ε|−θₜ)) — and the numerical thresholds that constitute confirmation versus disconfirmation. This converts APGI from a post-hoc fitting exercise into genuine predictive science.

Submit papers as Registered Reports to journals including eLife, Cortex, or NeuroImage — formats in which peer review and in-principle acceptance occur before data collection, eliminating publication bias and ensuring that null results are publishable. Deposit all datasets, preprocessing pipelines, model-fitting scripts, and simulation code in public repositories (OSF, GitHub, Zenodo) with each publication, enabling independent replication and direct computational comparison with GNW and IIT on identical datasets.

Preregistered null predictions carry exceptional credibility: specify conditions under which APGI explicitly predicts null effects — precision modulation should not affect early sensory ERPs (N1/P2) — and confirm them.

---

### Pillar 2: Adversarial Collaboration and Red-Team Testing

The 2020 Cogitate adversarial collaboration between GNW and IIT proponents is the current gold standard for consciousness theory testing. APGI requires an equivalent.

Invite one or two skeptical researchers — ideally proponents of IIT or higher-order theories — to co-design a study in which each framework makes divergent, pre-agreed predictions. Data should be collected and analyzed by a neutral third party, and results published jointly regardless of outcome. Internally, before submitting any major APGI paper, convene a structured red-team session with the explicit mandate to falsify the framework. Assign team members to argue that every positive result is explainable by GNW without APGI's additional parameters. Document which objections could not be answered and publish them as a limitations section or companion commentary.

The framework's most powerful credibility demonstration would be a set of crossover predictions — conditions under which APGI, GNW, IIT, higher-order theories, and recurrent processing theories make divergent numerical predictions about the same dependent variable, and only APGI's formalism yields the correct quantitative answer. These are the predictions that establish the framework's necessity rather than its compatibility with existing accounts.

Stress-testing with anomalous cases is equally important. APGI must generate explicit, quantitative predictions for split-brain patients (where Π and θₜ may differ between hemispheres), locked-in syndrome (motor report absent, conscious experience potentially intact), post-cardiac-arrest recovery trajectories (measurable metabolic threshold disruption), and psychedelic states (pharmacologically well-characterized precision perturbation) — without parameter re-fitting between populations.

---

### Pillar 3: Single-Unit and Laminar Electrophysiology

The current empirical roadmap operates primarily at EEG, fMRI, and behavioral resolution. This constrains the framework's ability to resolve the computational architecture questions that differentiate APGI from GNW, because those questions concern laminar and cellular-level mechanisms inaccessible to noninvasive measurement.

APGI's precision-gating mechanism implies specific laminar predictions: precision signals should be encoded in superficial-layer feedback projections consistent with predictive processing hierarchies, while prediction errors should emerge in granular input layers. These are testable with laminar probes in neurosurgical patients or non-human primates.

At the single-unit level, neurons in frontoparietal cortex should cross a firing-rate threshold sigmoidally as stimulus precision increases, with the inflection point shifting as a function of metabolic state (e.g., circadian phase, fed vs. fasted). Simultaneously recorded local field potentials should show theta-gamma phase-amplitude coupling linked to the spiking of individual precision-encoding and error-encoding neurons.

If the threshold θₜ is implemented by inhibitory interneuron populations — as the framework implies — selective optogenetic or pharmacological modulation of PV+ versus SST+ interneurons should produce dissociable effects on threshold versus gain of the ignition function, providing a mechanistic falsification test currently achievable in rodent models.

---

### Pillar 4: Falsification Hierarchy and Theoretical Commitments

A theory without an explicit falsification hierarchy is a research program, not a theory. APGI must publicly commit to a three-tier structure:

**Disconfirming results** — findings that require abandonment of APGI's core claim. Example: if P3b amplitude shows no relationship to Π × |ε| across 1,000 trials with reliable precision manipulation and N ≥ 30 participants, the framework is falsified at the neural-correlates level.

**Revision-requiring results** — findings that necessitate modification of specific parameters without abandoning the core. Example: if θₜ dynamics prove non-metabolic in origin, the interoceptive weighting component requires revision while the precision-gating mechanism remains intact.

**Peripheral findings** — null results on secondary predictions that leave the core untouched.

Quantitative benchmarks must accompany these commitments. Rather than treating any above-chance result as support, APGI should specify minimum effect sizes, explained-variance thresholds, and Bayes factors that constitute genuine confirmation. A concrete example: "APGI is confirmed at the behavioral level if and only if the sigmoidal model achieves BF > 100 over an additive linear model in a preregistered dataset of N ≥ 500 trials per participant across N ≥ 30 participants."

Every APGI paper should include a formal nested model comparison: full APGI, APGI without dynamic θₜ, APGI without precision weighting, standard GNW models, and linear baseline. Publishing this hierarchy — rather than showing only that APGI fits the data — is the methodological standard now required at top-tier journals.

---

### Pillar 5: Multimodal Biomarker Convergence

Relying on any single measure renders APGI vulnerable to paradigm-specific confounds. Convergent evidence across independent modalities is required.

Simultaneous EEG-fMRI allows APGI to demonstrate that P3b amplitude predicts frontoparietal BOLD on the same trial, and that both measures jointly track Π × |ε| − θₜ. Pupillometry provides a continuous, non-invasive proxy for Π on every trial via locus coeruleus–norepinephrine coupling, enabling trial-by-trial model fitting without relying on group-level precision manipulations.

Given APGI's emphasis on interoceptive precision, the heartbeat-evoked potential (HEP) (the scalp electrical response time-locked to one's own heartbeat) and its attentional modulation should track θₜ dynamics — a prediction that distinguishes APGI from purely cortical theories and is testable with standard EEG. Near-infrared spectroscopy (NIRS) provides metabolic state indices independent of BOLD fMRI; APGI predicts that pre-stimulus NIRS signals in frontoparietal cortex predict subsequent ignition probability, testable in populations where fMRI is contraindicated.

EEG microstates — brief periods of quasi-stable scalp electrical topography lasting 60–120 ms — may correspond to attractor states in APGI's dynamical system. APGI should predict that the subthreshold-to-suprathreshold microstate transition precedes the P3b, and that this transition probability scales with Π × |ε|.

---

### Pillar 6: Developmental and Lifespan Validation

Consciousness theories gain substantial credibility when they correctly predict the developmental trajectory of conscious access — a domain neither GNW nor IIT has addressed quantitatively.

If θₜ is metabolically determined, neonates should show broad, graded ignition dynamics reflecting immature precision computation, transitioning to sharp sigmoidal ignition by late infancy as myelination and cortical development mature. This is testable with infant EEG adapted from the adult GNW literature. The protracted development of prefrontal cortex into the mid-twenties should produce measurably lower frontoparietal coupling efficiency and higher θₜ for emotionally neutral stimuli in adolescents versus adults.

In healthy aging, locus coeruleus degeneration reduces norepinephrine signaling, which APGI predicts should manifest as elevated θₜ and reduced Π dynamic range — a measurable narrowing of conscious access bandwidth testable in lifespan datasets. In sleep, slow-wave stages should produce maximal θₜ elevation (consistent with the near-elimination of conscious report in deep sleep), while REM should restore Π-mediated ignition capacity (consistent with dream phenomenology), testable with polysomnography combined with EEG connectivity.

---

### Pillar 7: Computational and AI Benchmarking

Because APGI inherits the precision-weighting architecture of the Free Energy Principle while adding the ignition threshold and allostatic dynamics the FEP leaves underspecified, its computational benchmarking program has a sharper mandate than simple model comparison: it must demonstrate that APGI adds quantitative predictive precision to the FEP baseline — not merely that it fits data the FEP also fits.

The APGI-LNN simulator should be run against the same benchmark paradigms used to validate GNW (Shanahan's global workspace model), IIT (PyPhi), and recurrent processing theories (Lamme's models), with a systematic comparison table showing which paradigms each framework fits, at what parameter cost, and with what predictive precision. Implementation on neuromorphic hardware (Intel Loihi, SpiNNaker) would constrain APGI to biologically plausible computational primitives and provide proof of physical implementability.

Computing integrated information (Φ) for APGI network states during simulated ignition versus non-ignition — and showing that Φ tracks threshold crossing — would establish whether APGI and IIT are mathematically compatible or genuinely divergent. APGI's ignition dynamics imply operation near a phase transition, demonstrable via branching ratio approaching 1.0 (a measure of whether neural activity neither dies out nor explodes, but propagates at the edge of stability), power-law scaling in avalanche size distributions, and maximum perturbation susceptibility near θₜ. Whether analogous dynamics emerge in trained artificial neural networks under surprise-weighted gating conditions is a generativity question worth examining — provided functional analogy is carefully distinguished from mechanistic homology.

---

### Pillar 8: Pharmacological and Metabolic Dissections

No existing consciousness theory quantitatively links ignition to cellular metabolism. Pillar 8 is where APGI either earns or forfeits its most original claim.

APGI must demonstrate that θₜ elevation following metabolic challenge reflects a central rather than peripheral mechanism — that equivalent metabolic perturbation not affecting cerebral glucose metabolism does not elevate θₜ, requiring pharmacological dissociation via insulin clamp protocols. A striking positive test: exogenous ketone supplementation (beta-hydroxybutyrate) should buffer against θₜ elevation during caloric restriction by providing an alternative cerebral fuel source.

Mechanistic independence of the two core parameters requires pharmacological dissociation: atomoxetine (norepinephrine transporter reuptake inhibitor) should raise exteroceptive Π without affecting θₜ; propranolol (beta-blocker) should reduce interoceptive Π specifically without affecting exteroceptive precision. Scopolamine (muscarinic antagonist) and physostigmine (cholinesterase inhibitor) should parametrically shift precision weighting in opposite directions, consistent with acetylcholine's established role in precision modulation (Yu & Dayan, 2005). Combining these with psychometric fitting from the behavioral validation program would provide direct pharmacological validation of Π's biological substrate.

Psychedelics offer a particularly stringent cross-domain test. Psilocybin, LSD, and DMT differ in receptor binding profiles and phenomenological signature. APGI should generate specific predictions for how each compound shifts Π, θₜ, and the ignition sigmoid, derivable from receptor pharmacology without post-hoc parameter adjustment. Correctly predicting the direction and magnitude of these shifts would constitute striking convergent validation.

---

### Pillar 9: Network Neuroscience and Graph-Theoretic Validation

APGI's ignition mechanism implies testable network-level properties accessible through graph-theoretic analysis of functional and structural connectivity data.

The frontoparietal network's anatomical rich-club structure (a network architecture in which the most highly connected hub nodes are disproportionately interconnected with each other) — its densely interconnected hub nodes — is the proposed structural scaffold for global ignition. APGI predicts that individual differences in θₜ should correlate inversely with rich-club connectivity strength measured by diffusion tractography: stronger hub organization predicting lower threshold and sharper ignition slope. This is a novel, currently untested prediction derived from APGI's architecture rather than from existing rich-club–cognition correlates, and should be understood as a prospective empirical commitment.

Dynamic functional connectivity should show a specific temporal sequence: subthreshold stimuli produce no change in frontoparietal synchrony, while suprathreshold stimuli produce rapid (<50 ms) hub-node synchronization, testable via sliding-window or point-process connectivity analyses in existing EEG-fMRI datasets. Default mode network (DMN) suppression onset, frontoparietal ignition, and the P3b should follow a specific temporal ordering predictable from Π × |ε| − θₜ — a three-way relationship testable with simultaneous EEG-fMRI. Across lifespan datasets, gray matter covariance within the frontoparietal ignition network should predict individual θₜ estimates derived from behavioral psychometric functions, providing structural neuroimaging validation of the behavioral parameter.

---

### Pillar 10: Philosophical and Conceptual Fortification

Scientific credibility in consciousness research is inseparable from philosophical rigor, because the field's core concepts — phenomenal consciousness, access consciousness, reportability — are philosophically contested.

APGI must state clearly whether it is a theory of access consciousness (the functional availability of information for reasoning and report), phenomenal consciousness (the qualitative character of experience), or the relationship between them. The current framework conflates these levels, inviting the standard objection that any functional account leaves the hard problem untouched. The appropriate response is not to claim that APGI solves the hard problem, but to delineate precisely what it explains and what it does not.

A concise section acknowledging that APGI provides a computational-level account that does not entail phenomenal properties — while arguing that computational precision is a necessary precondition for empirical progress — would preempt the most common philosophical objections. Explicitly positioning the framework at Marr's computational, algorithmic, and implementational levels, with clear statements about which predictions operate at which level, prevents the level-crossing confusions that have undermined previous frameworks.

Finally, APGI must engage the no-report paradigm debate: whether behavioral report paradigms measure consciousness or merely the access required for report. The framework must generate predictions for no-report paradigms — where neural signatures of conscious access are measured without verbal or motor response — and demonstrate that the predicted ignition dynamics occur independently of the report mechanism.

---

### Pillar 11: Metascience and Cumulative Evidence Standards

Before data collection begins, specify a prospective meta-analysis protocol: the effect sizes, moderators, and sample sizes required for cumulative evidence to constitute a scientific consensus. This establishes a public benchmark that prevents post-hoc cherry-picking of positive results.

Actively seek and support independent replication of APGI's core findings by groups with no stake in the framework. Provide analysis code, stimuli, and detailed protocols to replicating laboratories. Publish replication failures alongside successes — this single behavior most distinguishes credible from non-credible research programs in contemporary science. Maintain a publicly accessible, continuously updated systematic review of all evidence bearing on APGI's core predictions — positive, negative, and ambiguous — establishing the framework as a transparent, self-correcting scientific enterprise rather than an advocacy program.

Every APGI paper should report effect sizes with confidence intervals and, where Bayesian methods are used, Bayes factors alongside posterior distributions over key parameters. This enables meta-analysts to include APGI results in future syntheses without requesting raw data.

## Effect Size Benchmarks for Strategic Pillars

To match the quantitative rigor of the Falsification-Criteria specifications, each strategic pillar includes explicit effect size benchmarks with target Bayes Factors:

| Pillar | Primary Effect Size Metric | Target Cohen's d | Target Bayes Factor (BF₁₀) | Falsification Threshold |
| -------- | ---------------------------- | ------------------ | ---------------------------- | ------------------------- |
| **Pillar 1** (Open Science) | Preregistered prediction accuracy | d > 0.5 | BF₁₀ > 3 | d < 0.3 or BF₁₀ < 1 |
| **Pillar 2** (Adversarial Collaboration) | Crossover prediction resolution | d > 0.8 | BF₁₀ > 10 | d < 0.4 or BF₁₀ < 3 |
| **Pillar 3** (Single-Unit/Laminar) | Spike-rate threshold modulation | d > 1.2 | BF₁₀ > 30 | d < 0.6 or BF₁₀ < 3 |
| **Pillar 4** (Falsification Hierarchy) | Nested model comparison | d > 0.6 | BF₁₀ > 10 | d < 0.3 or BF₁₀ < 3 |
| **Pillar 5** (Multimodal Biomarkers) | Cross-modal convergence | d > 0.7 | BF₁₀ > 10 | d < 0.4 or BF₁₀ < 3 |
| **Pillar 6** (Developmental/Lifespan) | Age-related threshold change | d > 0.5 per decade | BF₁₀ > 3 per comparison | d < 0.3 or nonsignificant trend |
| **Pillar 7** (Computational/AI) | APGI-FEP discriminability | d > 0.6 | BF₁₀ > 10 | d < 0.3 or BF₁₀ < 3 |
| **Pillar 8** (Pharmacological) | Drug-induced threshold shift | d > 0.8 | BF₁₀ > 10 | d < 0.4 or BF₁₀ < 3 |
| **Pillar 9** (Network Neuroscience) | Rich-club connectivity correlation | d > 0.5 | BF₁₀ > 3 | d < 0.3 or BF₁₀ < 1 |
| **Pillar 10** (Philosophical) | No-report paradigm ignition | d > 0.6 | BF₁₀ > 10 | d < 0.3 or BF₁₀ < 3 |
| **Pillar 11** (Metascience) | Meta-analytic effect size | d > 0.5 | BF₁₀ > 10 (cumulative) | d < 0.3 or heterogeneity I² > 75% |

**Interpretation Guidelines:**

- BF₁₀ > 3: Weak evidence (minimum publishable)
- BF₁₀ > 10: Moderate evidence (standard for single studies)
- BF₁₀ > 30: Strong evidence (required for extraordinary claims)
- BF₁₀ > 100: Very strong evidence (required for paradigm shift claims)

All Bayes factor calculations use default JZS priors with scale = √2/2. Alternative prior specifications must be reported as sensitivity analyses.

---

The above program does not exhaust the possible empirical directions for APGI, but represents the minimum portfolio required for the framework to achieve credibility parity with GNW and IIT at top-tier publication venues. Priorities 1–4 represent the critical path; Pillars 1–11 represent the infrastructure within which those priorities must be embedded.
