# APGI FRAMEWORK FALSIFICATION CRITERIA SPECIFICATIONS

## FALSIFICATION NOTATION STANDARD

All criteria follow this mandatory format:

```text
**[Criterion ID]: [Brief Description]**

- **Quantitative threshold:** [Specific numerical value with units]
- **Statistical test:** [Test type (e.g., two-tailed t-test, ANOVA, Mann-Whitney U), significance level α]
- **Minimum effect size:** [Cohen's d, η², Pearson's r, or appropriate metric with threshold]
- **Alternative hypothesis:** Falsified if [specific contradictory outcome with numerical bounds]
```

---

## PRIORITY 1: FOUNDATIONAL FALSIFICATION PROTOCOLS

### Falsification-Protocol-1: Active Inference Agent Simulations

**Tests:** Hierarchical generative models (#5), self-similar APGI computation, level-specific timescales

#### F1.1: APGI Agent Performance Advantage

- **Quantitative threshold:** APGI agents achieve ≥18% higher cumulative reward than standard predictive processing agents over 1000 trials in multi-level decision tasks
- **Statistical test:** Independent samples t-test, two-tailed, α = 0.01 (Bonferroni-corrected for 6 comparisons, family-wise α = 0.05)
- **Minimum effect size:** Cohen's d ≥ 0.6 (medium-to-large effect)
- **Alternative hypothesis:** Falsified if APGI advantage <10% OR d < 0.35 OR p ≥ 0.01

#### F1.2: Hierarchical Level Emergence

- **Quantitative threshold:** Intrinsic timescale measurements show ≥3 distinct temporal clusters corresponding to Levels 1-3 (τ₁ ≈ 50-150ms, τ₂ ≈ 200-800ms, τ₃ ≈ 1-3s), with between-cluster separation >2× within-cluster standard deviation
- **Statistical test:** K-means clustering (k=3) with silhouette score validation; one-way ANOVA comparing cluster means, α = 0.001
- **Minimum effect size:** η² ≥ 0.70 (large effect), silhouette score ≥ 0.45
- **Alternative hypothesis:** Falsified if <3 clusters emerge OR silhouette score < 0.30 OR between-cluster separation < 1.5× within-cluster SD OR η² < 0.50

#### F1.3: Level-Specific Precision Weighting

- **Quantitative threshold:** Precision weights (Πⁱ, Πᵉ) show differential modulation across hierarchical levels, with Level 1 interoceptive precision 25-40% higher than Level 3 during interoceptive salience tasks
- **Statistical test:** Repeated-measures ANOVA (Level × Precision Type), α = 0.001; post-hoc Tukey HSD
- **Minimum effect size:** Partial η² ≥ 0.15 for Level × Type interaction
- **Alternative hypothesis:** Falsified if Level 1-3 interoceptive precision difference <15% OR interaction p ≥ 0.01 OR partial η² < 0.08

#### F1.4: Threshold Adaptation Dynamics

- **Quantitative threshold:** Allostatic threshold θ_t adapts with time constant τ_θ = 10-100s, showing >20% reduction after sustained high prediction error exposure (>5min), with recovery time constant within 2-3× τ_θ
- **Statistical test:** Exponential decay curve fitting (R² ≥ 0.80); paired t-test comparing pre/post-exposure thresholds, α = 0.01
- **Minimum effect size:** Cohen's d ≥ 0.7 for pre/post comparison; θ_t reduction ≥20%
- **Alternative hypothesis:** Falsified if threshold adaptation <12% OR τ_θ < 5s or >150s OR curve fit R² < 0.65 OR recovery time >5× τ_θ

#### F1.5: Cross-Level Phase-Amplitude Coupling (PAC)

- **Quantitative threshold:** Theta-gamma PAC (Level 1-2 coupling) shows modulation index MI ≥ 0.012, with ≥30% increase during ignition events vs. baseline
- **Statistical test:** Permutation test (10,000 iterations) for PAC significance, α = 0.001; paired t-test for ignition vs. baseline, α = 0.01
- **Minimum effect size:** Cohen's d ≥ 0.5 for ignition effect
- **Alternative hypothesis:** Falsified if MI < 0.008 OR ignition increase <15% OR permutation p ≥ 0.01 OR d < 0.30

#### F1.6: 1/f Spectral Slope Predictions

- **Quantitative threshold:** Aperiodic exponent α_spec = 0.8-1.2 during active task engagement, increasing to α_spec = 1.5-2.0 during low-arousal states (using FOOOF/specparam algorithm)
- **Statistical test:** Paired t-test comparing active vs. low-arousal states, α = 0.001; goodness-of-fit for spectral parameterization R² ≥ 0.90
- **Minimum effect size:** Cohen's d ≥ 0.8 for state difference; Δα_spec ≥ 0.4
- **Alternative hypothesis:** Falsified if active α_spec > 1.4 OR low-arousal α_spec < 1.3 OR Δα_spec < 0.25 OR d < 0.50 OR spectral fit R² < 0.85

---

### Falsification-Protocol-2: Iowa Gambling Task (IGT) Environment

**Tests:** Somatic marker modulation (#1, #14), interoceptive precision weighting, vmPFC-like decision bias

#### F2.1: Somatic Marker Advantage Quantification

- **Quantitative threshold:** APGI agents show ≥22% higher selection frequency for advantageous decks (C+D) vs. disadvantageous (A+B) by trial 60, compared to ≤12% for agents without somatic modulation
- **Statistical test:** Two-proportion z-test comparing APGI vs. no-somatic agents, α = 0.01; repeated-measures ANOVA for learning trajectory
- **Minimum effect size:** Cohen's h ≥ 0.55 (medium-large effect for proportions); between-group difference ≥10 percentage points
- **Alternative hypothesis:** Falsified if APGI advantageous selection <18% by trial 60 OR advantage over no-somatic agents <8 percentage points OR h < 0.35 OR p ≥ 0.01

#### F2.2: Interoceptive Cost Sensitivity

- **Quantitative threshold:** Deck selection correlates with simulated interoceptive cost at r = -0.45 to -0.65 for APGI agents (i.e., higher cost → lower selection), vs. r = -0.15 to +0.05 for non-interoceptive agents
- **Statistical test:** Pearson correlation with Fisher's z-transformation for group comparison, α = 0.01
- **Minimum effect size:** APGI |r| ≥ 0.40; Fisher's z for group difference ≥ 1.80 (p < 0.05)
- **Alternative hypothesis:** Falsified if APGI |r| < 0.30 OR group difference z < 1.50 (p ≥ 0.07) OR non-interoceptive |r| > 0.20

#### F2.3: vmPFC-Like Anticipatory Bias

- **Quantitative threshold:** APGI agents show ≥35ms faster reaction times for selections from previously rewarding decks with low interoceptive cost, with RT modulation β_cost ≥ 25ms per unit cost increase
- **Statistical test:** Linear mixed-effects model (LMM) with random intercepts for agents; F-test for cost effect, α = 0.01
- **Minimum effect size:** Standardized β ≥ 0.40; marginal R² ≥ 0.18
- **Alternative hypothesis:** Falsified if RT advantage <20ms OR β_cost < 15ms/unit OR standardized β < 0.25 OR marginal R² < 0.10

#### F2.4: Precision-Weighted Integration (Not Error Magnitude)

- **Quantitative threshold:** Somatic marker modulation targets precision (Πⁱ_eff) as demonstrated by ≥30% greater influence of high-confidence interoceptive signals vs. low-confidence signals, independent of prediction error magnitude
- **Statistical test:** Multiple regression: Deck preference ~ Intero_Signal × Confidence + PE_Magnitude; test Confidence interaction, α = 0.01
- **Minimum effect size:** Standardized β_interaction ≥ 0.35; semi-partial R² ≥ 0.12
- **Alternative hypothesis:** Falsified if confidence effect <18% OR β_interaction < 0.22 OR p ≥ 0.01 OR semi-partial R² < 0.08

#### F2.5: Learning Trajectory Discrimination

- **Quantitative threshold:** APGI agents reach 70% advantageous selection criterion by trial 45 ± 10, whereas non-interoceptive agents require >65 trials (≥20 trial advantage)
- **Statistical test:** Log-rank test for survival analysis (time-to-criterion), α = 0.01; Cox proportional hazards model
- **Minimum effect size:** Hazard ratio ≥ 1.65 (APGI learns 65% faster)
- **Alternative hypothesis:** Falsified if APGI time-to-criterion >55 trials OR hazard ratio < 1.35 OR log-rank p ≥ 0.01 OR trial advantage <12

---

### Falsification-Protocol-3: Agent Comparison Experiment

**Tests:** APGI vs. baseline model performance, quantitative advantage demonstration

#### F3.1: Overall Performance Advantage

- **Quantitative threshold:** APGI agents achieve ≥18% higher cumulative reward than the best non-APGI baseline (Standard PP, GWT-only, or Q-learning) across mixed task battery (n ≥ 100 trials per task, 3+ task types)
- **Statistical test:** Independent samples t-test with Welch correction for unequal variances, two-tailed, α = 0.008 (Bonferroni for 6 comparisons)
- **Minimum effect size:** Cohen's d ≥ 0.60; 95% CI for advantage excludes 10%
- **Alternative hypothesis:** Falsified if APGI advantage <12% OR d < 0.40 OR p ≥ 0.008 OR 95% CI includes 8%

#### F3.2: Interoceptive Task Specificity

- **Quantitative threshold:** APGI advantage increases to ≥28% in tasks with high interoceptive relevance (e.g., IGT, threat detection, effort allocation) vs. ≤12% in purely exteroceptive tasks
- **Statistical test:** Two-way mixed ANOVA (Agent Type × Task Category); test interaction, α = 0.01
- **Minimum effect size:** Partial η² ≥ 0.20 for interaction; simple effects d ≥ 0.70 for interoceptive tasks
- **Alternative hypothesis:** Falsified if interoceptive advantage <20% OR interaction p ≥ 0.01 OR partial η² < 0.12 OR simple effects d < 0.45

#### F3.3: Threshold Gating Necessity

- **Quantitative threshold:** Removing threshold gating (θ_t → 0) reduces APGI performance by ≥25% in volatile environments, demonstrating non-redundancy of ignition mechanism
- **Statistical test:** Paired t-test comparing full APGI vs. no-threshold variant, α = 0.01
- **Minimum effect size:** Cohen's d ≥ 0.75
- **Alternative hypothesis:** Falsified if performance reduction <15% OR d < 0.50 OR p ≥ 0.01

#### F3.4: Precision Weighting Necessity

- **Quantitative threshold:** Uniform precision (Πⁱ = Πᵉ = constant) reduces APGI performance by ≥20% in tasks with unreliable sensory modalities
- **Statistical test:** Paired t-test, α = 0.01
- **Minimum effect size:** Cohen's d ≥ 0.65
- **Alternative hypothesis:** Falsified if reduction <12% OR d < 0.42 OR p ≥ 0.01

#### F3.5: Computational Efficiency Trade-Off

- **Quantitative threshold:** APGI maintains ≥85% of full model performance while using ≤60% of computational operations (measured by floating-point operations per decision)
- **Statistical test:** Equivalence testing (TOST procedure) for non-inferiority in performance, with efficiency ratio t-test, α = 0.05
- **Minimum effect size:** Efficiency gain ≥30%; performance retention ≥85%
- **Alternative hypothesis:** Falsified if performance retention <78% OR efficiency gain <20% OR fails TOST non-inferiority bounds

#### F3.6: Sample Efficiency in Learning

- **Quantitative threshold:** APGI agents achieve 80% asymptotic performance in ≤200 trials, vs. ≥300 trials for standard RL baselines (≥33% sample efficiency advantage)
- **Statistical test:** Time-to-criterion analysis with log-rank test, α = 0.01
- **Minimum effect size:** Hazard ratio ≥ 1.45
- **Alternative hypothesis:** Falsified if APGI time-to-criterion >250 trials OR advantage <25% OR hazard ratio < 1.30 OR p ≥ 0.01

---

## PRIORITY 2: CRITICAL FALSIFICATION PROTOCOLS

### Falsification-Protocol-5: Evolutionary APGI Emergence

**Tests:** Evolutionary derivation from biological constraints (#21), selection pressure for APGI features

#### F5.1: Threshold Filtering Emergence

- **Quantitative threshold:** ≥75% of evolved agents under metabolic constraint (energy budget <80% baseline) develop threshold-like gating with ignition sharpness α ≥ 4.0 by generation 500
- **Statistical test:** Binomial test against 50% null rate, α = 0.01; one-sample t-test for α values
- **Minimum effect size:** Proportion difference ≥ 0.25 (75% vs. 50%); mean α ≥ 4.0 with Cohen's d ≥ 0.80 vs. unconstrained control
- **Alternative hypothesis:** Falsified if <60% develop thresholds OR mean α < 3.0 OR d < 0.50 OR binomial p ≥ 0.01

#### F5.2: Precision-Weighted Coding Emergence

- **Quantitative threshold:** ≥65% of evolved agents under noisy signaling constraints develop precision-like weighting (correlation between signal reliability and influence ≥0.45) by generation 400
- **Statistical test:** Binomial test, α = 0.01; Pearson correlation test
- **Minimum effect size:** r ≥ 0.45; proportion difference ≥ 0.15 vs. no-noise control
- **Alternative hypothesis:** Falsified if <50% develop weighting OR mean r < 0.35 OR binomial p ≥ 0.01

#### F5.3: Interoceptive Prioritization Emergence

- **Quantitative threshold:** Under survival pressure (resources tied to homeostasis), ≥70% of agents evolve interoceptive signal gain β_intero ≥ 1.3× exteroceptive gain by generation 600
- **Statistical test:** Binomial test, α = 0.01; paired t-test comparing β_intero vs. β_extero
- **Minimum effect size:** Mean gain ratio ≥ 1.3; Cohen's d ≥ 0.60 for paired comparison
- **Alternative hypothesis:** Falsified if <55% show prioritization OR mean ratio < 1.15 OR d < 0.40 OR binomial p ≥ 0.01

#### F5.4: Multi-Timescale Integration Emergence

- **Quantitative threshold:** ≥60% of evolved agents develop ≥2 distinct temporal integration windows (fast: 50-200ms, slow: 500ms-2s) under multi-level environmental dynamics
- **Statistical test:** Autocorrelation function analysis with peak detection; binomial test for proportion, α = 0.01
- **Minimum effect size:** Peak separation ≥3× fast window duration; proportion difference ≥ 0.10
- **Alternative hypothesis:** Falsified if <45% develop multi-timescale OR peak separation < 2× fast window OR binomial p ≥ 0.01

#### F5.5: APGI-Like Feature Clustering

- **Quantitative threshold:** Principal component analysis on evolved agent parameters shows ≥70% of variance captured by first 3 PCs corresponding to threshold gating, precision weighting, and interoceptive bias dimensions
- **Statistical test:** Scree plot analysis; varimax rotation for interpretability; loadings ≥0.60 on predicted dimensions
- **Minimum effect size:** Cumulative variance ≥70%; minimum loading ≥0.60
- **Alternative hypothesis:** Falsified if cumulative variance <60% OR loadings <0.45 OR PCs don't align with predicted dimensions (cosine similarity <0.65)

#### F5.6: Non-APGI Architecture Failure

- **Quantitative threshold:** Control agents without evolved APGI features (threshold, precision, interoceptive bias) show ≥40% worse performance under combined metabolic + noise + survival constraints
- **Statistical test:** Independent samples t-test, α = 0.01
- **Minimum effect size:** Cohen's d ≥ 0.85
- **Alternative hypothesis:** Falsified if performance difference <25% OR d < 0.55 OR p ≥ 0.01

---

### Falsification-Protocol-6: Network Comparison Experiment

**Tests:** Liquid networks vs. alternative architectures (#4, #29), intrinsic vs. add-on mechanisms

#### F6.1: Intrinsic Threshold Behavior

- **Quantitative threshold:** Liquid time-constant networks (LTCNs) show sharp ignition transitions (10-90% firing rate increase within <50ms) without explicit threshold modules, whereas feedforward networks require added sigmoidal gates
- **Statistical test:** Transition time comparison (Mann-Whitney U test for non-normal distributions), α = 0.01
- **Minimum effect size:** LTCN median transition time ≤50ms vs. >150ms for feedforward without gates; Cliff's delta ≥ 0.60
- **Alternative hypothesis:** Falsified if LTCN transition time >80ms OR Cliff's delta < 0.45 OR Mann-Whitney p ≥ 0.01

#### F6.2: Intrinsic Temporal Integration

- **Quantitative threshold:** LTCNs naturally integrate information over 200-500ms windows (measured by autocorrelation decay to <0.37) without recurrent add-ons, vs. <50ms for standard RNNs
- **Statistical test:** Exponential decay curve fitting; Wilcoxon signed-rank test comparing integration windows, α = 0.01
- **Minimum effect size:** LTCN integration window ≥4× standard RNN; curve fit R² ≥ 0.85
- **Alternative hypothesis:** Falsified if LTCN window <150ms OR ratio < 2.5× OR R² < 0.70 OR p ≥ 0.01

#### F6.3: Metabolic Selectivity Without Training

- **Quantitative threshold:** LTCNs with adaptive time constants τ(x) show ≥30% reduction in active units during low-information periods without explicit sparsity training, vs. <10% for standard architectures
- **Statistical test:** Paired t-test comparing low vs. high information periods, α = 0.01; between-architecture independent t-test
- **Minimum effect size:** Cohen's d ≥ 0.70 for LTCN sparsity effect; d ≥ 0.60 between architectures
- **Alternative hypothesis:** Falsified if LTCN sparsity <20% OR d < 0.45 OR between-architecture d < 0.40 OR p ≥ 0.01

#### F6.4: Fading Memory Implementation

- **Quantitative threshold:** LTCNs show exponential memory decay with time constant τ_memory = 1-3s for task-relevant information, measurable through delayed match-to-sample accuracy decay
- **Statistical test:** Exponential decay model fitting (R² ≥ 0.85); goodness-of-fit χ²
- **Minimum effect size:** τ_memory within predicted 1-3s range; 95% CI excludes <0.5s and >5s
- **Alternative hypothesis:** Falsified if τ_memory < 0.5s OR > 5s OR R² < 0.75 OR 95% CI includes physiologically implausible values

#### F6.5: Bifurcation Structure for Ignition

- **Quantitative threshold:** LTCNs exhibit bistable attractors with saddle-node bifurcation at precision-weighted prediction error Π·|ε| = θ_t ± 0.15, with hysteresis width Δθ = 0.1-0.2 θ_t
- **Statistical test:** Phase plane analysis; bifurcation detection via eigenvalue sign changes; hysteresis loop area calculation
- **Minimum effect size:** Bifurcation point within ±0.20 of predicted value; hysteresis width 0.08-0.25 θ_t
- **Alternative hypothesis:** Falsified if no bistability detected OR bifurcation point error >0.30 OR hysteresis width <0.05θ_t or >0.30θ_t

#### F6.6: Alternative Architectures Require Add-Ons

- **Quantitative threshold:** Standard RNNs, LSTMs, and Transformers require ≥2 explicit modules (e.g., attention gates + sparsity constraints + threshold nonlinearities) to match ≥85% of LTCN performance on APGI-relevant tasks
- **Statistical test:** Performance equivalence testing (TOST), α = 0.05; module count comparison
- **Minimum effect size:** Alternative architectures need ≥2 add-ons; performance gap ≥15% without add-ons
- **Alternative hypothesis:** Falsified if alternatives match LTCN with ≤1 add-on OR performance gap <10% with native architecture

---

## PRIORITY 3: VALIDATION PROTOCOLS

### Validation-Protocol-1: Synthetic Neural Data Generation

**Tests:** Core APGI dynamics (Innovations #1, #2, #4), ignition thresholds, surprise accumulation

#### V1.1: Synthetic Data Discriminability

- **Quantitative threshold:** Machine learning classifiers achieve ≥85% accuracy (AUC-ROC ≥ 0.90) in discriminating APGI-generated conscious vs. unconscious trials based on simulated neural features
- **Statistical test:** Cross-validated classifier performance with 95% confidence intervals via bootstrapping (10,000 iterations)
- **Minimum effect size:** Cohen's d ≥ 0.90 for conscious vs. unconscious feature distributions
- **Alternative hypothesis:** Falsified if accuracy <78% OR AUC-ROC < 0.83 OR d < 0.65 OR 95% CI includes 72%

#### V1.2: Parameter Sensitivity Analysis

- **Quantitative threshold:** Classification performance degrades by ≥35% when APGI core parameters (Πⁱ, θ_t, β_som) are randomized, demonstrating non-arbitrary parameter structure
- **Statistical test:** Paired t-test comparing true vs. randomized parameters, α = 0.01
- **Minimum effect size:** Cohen's d ≥ 0.80
- **Alternative hypothesis:** Falsified if degradation <22% OR d < 0.52 OR p ≥ 0.01

#### V1.3: Temporal Dynamics Signature

- **Quantitative threshold:** Simulated ignition events show characteristic temporal profile: pre-ignition buildup (200-400ms), sharp transition (<50ms), sustained plateau (≥300ms), with ≥75% of trials matching this pattern
- **Statistical test:** Template matching with cross-correlation ≥0.70; binomial test for proportion, α = 0.01
- **Minimum effect size:** Median cross-correlation ≥0.70; proportion ≥75%
- **Alternative hypothesis:** Falsified if median correlation <0.60 OR proportion <65% OR binomial p ≥ 0.01

#### V1.4: Cross-Modal Integration Verification

- **Quantitative threshold:** Simulated multimodal trials (intero + extero) show ≥30% higher ignition probability than unimodal trials when Πⁱ and Πᵉ both elevated
- **Statistical test:** Logistic regression with interaction term; likelihood ratio test for interaction, α = 0.01
- **Minimum effect size:** Odds ratio ≥ 1.8 for multimodal advantage
- **Alternative hypothesis:** Falsified if advantage <18% OR OR < 1.5 OR interaction p ≥ 0.01

---

### Validation-Protocol-4: Information-Theoretic Phase Transition Analysis

**Tests:** Phase transitions (#11), bifurcation dynamics, entropy measures (#3)

#### V4.1: Critical Slowing Detection

- **Quantitative threshold:** Autocorrelation time τ_AC increases according to τ_AC ∝ |Π·|ε| - θ_t|^(-α) with exponent α = 0.45-0.55 as system approaches ignition threshold
- **Statistical test:** Power-law fitting with log-log regression (R² ≥ 0.85); 95% CI for α excludes 0.30 and 0.70
- **Minimum effect size:** R² ≥ 0.85; RMSE < 15% of range
- **Alternative hypothesis:** Falsified if α < 0.35 OR α > 0.65 OR R² < 0.75 OR 95% CI includes non-critical exponents

#### V4.2: Bistability Regime Identification

- **Quantitative threshold:** System exhibits two stable fixed points (subthreshold firing rate <5 Hz, ignited firing rate 20-40 Hz) within parameter region Π·|ε| ∈ [0.9θ_t, 1.2θ_t], with basin boundary at θ_t ± 0.10
- **Statistical test:** Fixed point stability analysis via Jacobian eigenvalues; basin of attraction estimation via Monte Carlo (10,000 initial conditions)
- **Minimum effect size:** Firing rate separation ≥15 Hz; basin boundary precision ±0.15θ_t
- **Alternative hypothesis:** Falsified if no bistability OR separation <10 Hz OR boundary error >0.20θ_t OR bistable region <0.15θ_t wide

#### V4.3: Hysteresis Loop Quantification

- **Quantitative threshold:** Forward (increasing Π·|ε|) and backward (decreasing Π·|ε|) ignition thresholds differ by Δθ = 0.12 ± 0.05 θ_t, forming measurable hysteresis loop
- **Statistical test:** Paired t-test comparing forward vs. backward thresholds, α = 0.01; hysteresis width within predicted range
- **Minimum effect size:** Cohen's d ≥ 0.75 for hysteresis; Δθ = 0.08-0.20 θ_t
- **Alternative hypothesis:** Falsified if Δθ < 0.06θ_t OR Δθ > 0.25θ_t OR d < 0.50 OR p ≥ 0.01

#### V4.4: Entropy Production Signatures

- **Quantitative threshold:** Thermodynamic entropy production rate increases by ≥45% during ignition transitions, with information-theoretic mutual information I(intero;extero) increasing by ≥0.8 bits
- **Statistical test:** Paired t-test for entropy change, α = 0.01; mutual information permutation test
- **Minimum effect size:** Cohen's d ≥ 0.70 for entropy; ΔI ≥ 0.8 bits
- **Alternative hypothesis:** Falsified if entropy increase <30% OR ΔI < 0.5 bits OR d < 0.48 OR permutation p ≥ 0.01

#### V4.5: Three-Level Entropy Consistency

- **Quantitative threshold:** Thermodynamic (S_thermo, J/K), information-theoretic (S_info, bits), and variational free energy (F, nats) show consistent ordering and bridge relations via Landauer's principle (ΔS_thermo ≥ k_B ln(2) ΔS_info) with ≤20% deviation
- **Statistical test:** Correlation between entropy measures r ≥ 0.75; Landauer relation residuals t-test, α = 0.05
- **Minimum effect size:** r ≥ 0.75; Landauer deviation <20%
- **Alternative hypothesis:** Falsified if r < 0.60 OR Landauer deviation >30% OR measures show contradictory trends

---

### Validation-Protocol-2: Bayesian Model Comparison

**Tests:** Model comparison framework, epistemic architecture (#3), empirical validation

#### V2.1: Bayes Factor Threshold

- **Quantitative threshold:** APGI model shows Bayes Factor BF₁₀ ≥ 10 (strong evidence) over standard predictive processing models on empirical consciousness datasets
- **Statistical test:** Bayesian model comparison via bridge sampling or thermodynamic integration; BIC approximation cross-validation
- **Minimum effect size:** BF₁₀ ≥ 10; ΔBIC ≥ 10 (strong support)
- **Alternative hypothesis:** Falsified if BF₁₀ < 3 (weak evidence) OR ΔBIC < 6

#### V2.2: Posterior Predictive Accuracy

- **Quantitative threshold:** APGI model achieves ≥20% lower mean absolute error (MAE) in posterior predictive checks for ignition timing compared to alternative models
- **Statistical test:** Cross-validated MAE with 95% HDI; Bayesian estimation
- **Minimum effect size:** ΔMAE ≥ 20%; 95% HDI excludes 10%
- **Alternative hypothesis:** Falsified if ΔMAE < 12% OR 95% HDI includes 8%

#### V2.3: Parameter Recovery

- **Quantitative threshold:** Synthetic data generated from known parameters recovers parameter estimates with r ≥ 0.82 (core parameters Πⁱ, θ_t, β_som) and r ≥ 0.68 (auxiliary parameters)
- **Statistical test:** Pearson correlation between true and recovered parameters; regression slope 0.85-1.15
- **Minimum effect size:** Core r ≥ 0.82; auxiliary r ≥ 0.68
- **Alternative hypothesis:** Falsified if core r < 0.75 OR auxiliary r < 0.60 OR slope outside [0.80, 1.20]

---

### Validation-Protocol-3: Active Inference Agent Simulations

**Tests:** Hierarchical architecture (#5), active inference, temporal depth

#### V3.1: Hierarchical Policy Emergence

- **Quantitative threshold:** Agents develop ≥3 hierarchical policy levels (reactive, tactical, strategic) with characteristic timescales τ₁ < 0.5s, τ₂ = 1-3s, τ₃ = 5-20s, evidenced by multi-timescale autocorrelation peaks
- **Statistical test:** Autocorrelation function peak detection with statistical significance via surrogate data; ANOVA comparing timescales
- **Minimum effect size:** Peak separation ≥2× lower timescale; η² ≥ 0.60 for timescale ANOVA
- **Alternative hypothesis:** Falsified if <3 levels emerge OR timescale separation < 1.5× OR η² < 0.45

#### V3.2: Active Inference Convergence

- **Quantitative threshold:** Agents using active inference reach ≥25% higher reward than passive perception agents by 500 trials, with policy entropy decreasing by ≥40%
- **Statistical test:** Independent samples t-test, α = 0.01; paired t-test for entropy change
- **Minimum effect size:** Cohen's d ≥ 0.70 for reward difference; d ≥ 0.65 for entropy reduction
- **Alternative hypothesis:** Falsified if reward advantage <15% OR entropy reduction <28% OR d < 0.48 for either

---

### Validation-Protocol-5: Evolutionary Emergence

**Tests:** Evolutionary derivation (#21), biological constraints

#### V5.1: Constraint-Driven Selection

- **Quantitative threshold:** APGI-like features emerge in ≥70% of populations under combined constraints (metabolic, noise, survival) by generation 600, vs. <30% under no constraints
- **Statistical test:** Fisher's exact test comparing proportions, α = 0.01
- **Minimum effect size:** Proportion difference ≥ 0.40; odds ratio ≥ 5.0
- **Alternative hypothesis:** Falsified if constrained proportion <55% OR proportion difference <0.30 OR OR < 3.5

---

### Validation-Protocol-6: RNN Architectures

**Tests:** Liquid networks (#4, #8), reservoir computing, APGI-LNN mapping

#### V6.1: Liquid Network Superiority

- **Quantitative threshold:** Liquid time-constant networks achieve ≥15% higher accuracy than standard RNNs on temporal integration tasks requiring 200-500ms windows
- **Statistical test:** Independent samples t-test, α = 0.01; cross-validated accuracy
- **Minimum effect size:** Cohen's d ≥ 0.55
- **Alternative hypothesis:** Falsified if accuracy advantage <10% OR d < 0.38 OR p ≥ 0.01

---

### Validation-Protocol-7: TMS/Pharmacological Predictions

**Tests:** Causal manipulations, neuromodulation (#28)

#### V7.1: TMS Disruption Prediction

- **Quantitative threshold:** TMS to anterior insula reduces ignition probability by 15-25% and increases ignition threshold θ_t by 18-30%, measurable via PCI reduction
- **Statistical test:** Paired t-test pre/post TMS, α = 0.01; within-subjects repeated measures
- **Minimum effect size:** Cohen's d ≥ 0.65 for both measures
- **Alternative hypothesis:** Falsified if ignition probability reduction <10% OR θ_t increase <12% OR d < 0.45 OR p ≥ 0.01

#### V7.2: Pharmacological Modulation

- **Quantitative threshold:** Cholinergic enhancement (e.g., nicotine) increases precision weighting Πⁱ by 20-35% and lowers θ_t by 12-20%
- **Statistical test:** Paired t-test drug vs. placebo, α = 0.01
- **Minimum effect size:** Cohen's d ≥ 0.60 for both
- **Alternative hypothesis:** Falsified if Πⁱ increase <15% OR θ_t reduction <8% OR d < 0.42

---

### Validation-Protocol-8: Psychophysical Thresholds

**Tests:** Parameter estimation (#26), behavioral biomarkers (#18)

#### V8.1: Individual Differences Correlation

- **Quantitative threshold:** Estimated θ_t from psychophysical detection tasks correlates with trait anxiety scores at r = 0.40-0.60
- **Statistical test:** Pearson correlation with 95% CI; permutation test for significance, α = 0.01
- **Minimum effect size:** r ≥ 0.40; 95% CI excludes 0.25
- **Alternative hypothesis:** Falsified if r < 0.30 OR 95% CI includes 0.20 OR permutation p ≥ 0.01

#### V8.2: Test-Retest Reliability

- **Quantitative threshold:** Parameter estimates show ICC ≥ 0.75 across sessions (1-week interval)
- **Statistical test:** Intraclass correlation coefficient (ICC 2,1), α = 0.05
- **Minimum effect size:** ICC ≥ 0.75; 95% CI lower bound >0.60
- **Alternative hypothesis:** Falsified if ICC < 0.65 OR 95% CI lower bound <0.50

---

### Validation-Protocol-9: Convergent Neural Signatures

**Tests:** HEP, P3b, PCI, multimodal integration (#6, #13)

(Already has complete falsification format - see original document)

---

### Validation-Protocol-10: Causal Manipulations

**Tests:** Network perturbations

#### V10.1: Perturbation Specificity

- **Quantitative threshold:** TMS to predicted nodes (anterior insula, ACC) disrupts ignition by ≥20%, while control regions show <8% disruption
- **Statistical test:** Two-way ANOVA (Region × Time), α = 0.01; post-hoc contrasts
- **Minimum effect size:** Partial η² ≥ 0.18 for Region effect; d ≥ 0.70 for target vs. control
- **Alternative hypothesis:** Falsified if target disruption <15% OR control disruption >12% OR η² < 0.12

---

### Validation-Protocol-11: Quantitative Model Fits

**Tests:** Parameter fitting, model comparison

#### V11.1: Model Fit Quality

- **Quantitative threshold:** APGI model achieves R² ≥ 0.75 for behavioral data (RT, accuracy), BIC at least 10 points better than next-best model
- **Statistical test:** Cross-validated R²; BIC comparison
- **Minimum effect size:** R² ≥ 0.75; ΔBIC ≥ 10
- **Alternative hypothesis:** Falsified if R² < 0.65 OR ΔBIC < 6

---

### Validation-Protocol-12: Clinical & Cross-Species Convergence

**Tests:** Psychiatric disorders (#10), cross-species (#33), clinical assessment

(Partial specifications exist in original document - see Innovation #17)

#### V12.1: Cross-Species PCI Scaling

- **Quantitative threshold:** PCI scales as predicted: mouse = 0.25-0.35, rat = 0.30-0.40, macaque = 0.42-0.55, human = 0.50-0.70
- **Statistical test:** Kruskal-Wallis H test, α = 0.01; post-hoc Dunn's test with Bonferroni correction
- **Minimum effect size:** η² ≥ 0.60; all pairwise differences p < 0.01
- **Alternative hypothesis:** Falsified if species overlap >30% OR ordering violated OR η² < 0.45

#### V12.2: Psychiatric Profile Discrimination

- **Quantitative threshold:** APGI parameter profiles classify psychiatric conditions (GAD, MDD, PTSD, psychosis) with ≥70% accuracy (multi-class AUC ≥ 0.82)
- **Statistical test:** Cross-validated classifier performance; confusion matrix analysis
- **Minimum effect size:** Accuracy ≥ 70%; per-class sensitivity ≥ 0.65
- **Alternative hypothesis:** Falsified if accuracy <62% OR multi-class AUC < 0.75 OR any class sensitivity <0.55

---

## SUMMARY STATISTICS

**Effect Size Distribution:**

- Cohen's d: 41 specifications (range 0.35-0.90)
- Correlation r: 8 specifications (range 0.40-0.82)
- Eta-squared η²: 7 specifications (range 0.15-0.70)
- Other (OR, hazard ratio, R², Bayes Factor): 11 specifications

**Statistical Tests Used:**

- t-tests: 28 specifications
- ANOVA: 8 specifications
- Correlation/regression: 10 specifications
- Non-parametric: 6 specifications
- Model comparison: 8 specifications
- Bayesian: 3 specifications

### Statistical Test Distribution

Parametric t-tests/ANOVA: 36 specifications (69%)
Correlation/regression: 10 specifications (19%)
Non-parametric: 6 specifications (12%)
Bayesian methods: 3 specifications (6%)
Model comparison: 8 specifications (15%)

### Significance Levels

- α = 0.01: 38 criteria (73%) - conservative threshold for main claims
- α = 0.008-0.05: 14 criteria (27%) - Bonferroni-corrected or less critical tests
