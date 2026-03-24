# APGI Paper Protocol

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

| Criterion Code | Protocol | Paper Prediction | Threshold Value | Statistical Test | Implementation Date | Status |
| -------------- | --------- | ------------------ | -------------- | --------------- | -------------------- | ------ |
| V1.1 | Validation_Protocol_1 | APGI agents achieve ≥18% higher cumulative reward than standard PP agents | Advantage ≥18%, d ≥ 0.60 | Independent samples t-test, α = 0.01 | 2024-01-15 | 62% complete |
| V1.2 | Validation_Protocol_1 | Intrinsic timescale measurements show ≥3 distinct temporal clusters (τ₁ ≈ 50-150ms, τ₂ ≈ 200-800ms, τ₃ ≈ 1-3s) | ≥3 clusters, separation ≥2×, η² ≥ 0.70 | K-means clustering, silhouette score validation, ANOVA, α = 0.001 | 2024-01-15 | 62% complete |
| V1.3 | Validation_Protocol_1 | Level 1 interoceptive precision 25-40% higher than Level 3 during interoceptive salience tasks | Difference ≥15%, η² ≥ 0.15 | Repeated-measures ANOVA, α = 0.001 | 2024-01-15 | 62% complete |
| V1.4 | Validation_Protocol_1 | Allostatic threshold θ_t adapts with time constant τ_θ = 10-100s, showing >20% reduction after sustained high prediction error | Adaptation ≥20%, d ≥ 0.7, recovery ≤5× | Exponential decay fitting, t-test, α = 0.01 | 2024-01-15 | 62% complete |
| V1.5 | Validation_Protocol_1 | Theta-gamma PAC (Level 1-2 coupling) shows MI ≥ 0.012, with ≥30% increase during ignition events | MI ≥ 0.012, increase ≥30%, d ≥ 0.50 | Permutation test (10,000 iterations), t-test, α = 0.001 | 2024-01-15 | 62% complete |
| V1.6 | Validation_Protocol_1 | Aperiodic exponent α_spec = 0.8-1.2 during active task, increasing to 1.5-2.0 during low-arousal states | Active ≤1.2, low ≥1.3, Δα ≥0.4, d ≥ 0.8 | Paired t-test, spectral fit R² ≥ 0.90, α = 0.001 | 2024-01-15 | 62% complete |
| V2.1 | Validation_Protocol_1 | APGI model shows Bayes Factor BF₁₀ ≥ 10 over standard PP models | BF₁₀ ≥ 10, ΔBIC ≥ 10 | Bayesian model comparison, bridge sampling/thermodynamic integration | 2024-01-15 | 62% complete |
| V2.2 | Validation_Protocol_1 | APGI model achieves ≥20% lower MAE in posterior predictive checks for ignition timing | ΔMAE ≥ 20%, 95% HDI excludes 10% | Cross-validated MAE with 95% HDI | 2024-01-15 | 62% complete |
| V2.3 | Validation_Protocol_1 | Synthetic data recovers parameter estimates with r ≥ 0.82 (core) and r ≥ 0.68 (auxiliary) | Core r ≥ 0.82, aux r ≥ 0.68, slope in [0.85, 1.15] | Pearson correlation, regression slope | 2024-01-15 | 62% complete |
| V7.1 | Validation_Protocol_7 | TMS over prefrontal cortex reduces ignition threshold θ_t by ≥15% within 30min, effect lasts ≥60min | Reduction ≥15%, duration ≥60min, d ≥ 0.70 | Paired t-test pre vs. post TMS, α=0.01 | 2024-01-15 | 79% complete |
| V7.2 | Validation_Protocol_7 | Propranolol increases interoceptive precision Π_i by ≥25% within 20min, decreases ignition probability by ≥30% | Π_i ≥25%, ignition ≥30% reduction, η² ≥ 0.20, d ≥ 0.65 | Repeated-measures ANOVA, paired t-test, α = 0.01 | 2024-01-15 | 79% complete |
| P7.1 | Validation_Protocol_7 | Propofol increases P3b:MMN amplitude ratio to ≥ 1.5:1, demonstrating thalamic gating mechanism | Ratio ≥ 1.5:1, d ≥ 0.60 | Paired t-test comparing P3b:MMN ratio pre vs. post propofol, α = 0.01 | 2024-01-15 | 79% complete |
| P7.2 | Validation_Protocol_7 | Ketamine suppresses MMN amplitude by ≥ 20% while P3b suppression < 50%, preserving conscious ignition | MMN ≥20% suppression, P3b <50% suppression, η² ≥ 0.15 | Repeated-measures ANOVA (drug × component), α = 0.01 | 2024-01-15 | 79% complete |
| P7.3 | Validation_Protocol_7 | Psilocybin increases P3b amplitude by ≥ 10% for low-salience stimuli, with HEP-embodiment correlation r ≥ 0.20 | P3b ≥10% increase, HEP-embodiment r ≥ 0.20, d ≥ 0.45 | Paired t-test for P3b increase; Pearson correlation for HEP-embodiment, α = 0.01 | 2024-01-15 | 79% complete |
| P5 | Validation_Protocol_9 | Mutual information increases ≥1 bit with precision cueing | MI increase ≥1 bit, permutation p < 0.01 | Permutation test for significance | 2024-01-15 | 73% complete |
| P6 | Validation_Protocol_9 | Information transmission rate asymptotes at ~40 bits/s | Asymptote ~40 bits/s, rate < 100 bits/s | Curve fitting, statistical tests | 2024-01-15 | 73% complete |
| P7 | Validation_Protocol_9 | Neyman-Pearson optimal threshold test - empirical threshold deviates from optimal by >2 SD | Deviation SD ≤ 2.0 | Neyman-Pearson criterion, bootstrap CI | 2024-01-15 | 73% complete |
| F1.1 | Falsification-Protocol-1 | APGI agent performance advantage falsification | Advantage <10% OR d < 0.35 OR p ≥ 0.01 | Independent samples t-test | 2024-01-15 | Implemented |
| F1.2 | Falsification-Protocol-1 | Hierarchical level emergence falsification | <3 clusters OR silhouette < 0.30 OR separation < 1.5× OR η² < 0.50 | K-means clustering, silhouette score, ANOVA | 2024-01-15 | Implemented |
| F1.3 | Falsification-Protocol-1 | Level-specific precision weighting falsification | Difference <15% OR interaction p ≥ 0.01 OR η² < 0.08 | Repeated-measures ANOVA | 2024-01-15 | Implemented |
| F1.4 | Falsification-Protocol-1 | Threshold adaptation dynamics falsification | Adaptation <12% OR τ_θ < 5s or >150s OR R² < 0.65 | Exponential decay fitting, t-test | 2024-01-15 | Implemented |
| F1.5 | Falsification-Protocol-1 | Cross-level PAC falsification | MI < 0.008 OR increase <15% OR permutation p ≥ 0.01 OR d < 0.30 | Permutation test, t-test | 2024-01-15 | Implemented |
| F1.6 | Falsification-Protocol-1 | 1/f spectral slope falsification | Active α_spec > 1.4 OR low α_spec < 1.3 OR Δα < 0.25 OR d < 0.50 OR R² < 0.85 | Paired t-test, spectral fit | 2024-01-15 | Implemented |
| F2.1 | Falsification-Protocol-1 | Somatic marker advantage falsification | Mean advantage ≥22 (supports model) | Paired t-test | 2024-01-15 | Implemented |
| F2.2 | Falsification-Protocol-1 | Interoceptive cost sensitivity falsification | Correlation in range [-0.65, -0.45] (supports model) | Pearson correlation | 2024-01-15 | Implemented |
| F2.3 | Falsification-Protocol-1 | vmPFC-like anticipatory bias falsification | RT advantage ≥35ms (supports model) | Paired t-test | 2024-01-15 | Implemented |
| F2.4 | Falsification-Protocol-1 | Precision-weighted integration falsification | Confidence effect ≥30% (supports model) | Paired t-test | 2024-01-15 | Implemented |
| F2.5 | Falsification-Protocol-1 | Learning trajectory discrimination falsification | Time to criterion ≤55 trials (supports model) | Paired t-test | 2024-01-15 | Implemented |
| F3.1 | Falsification-Protocol-1 | Overall performance advantage falsification | Advantage ≥18% OR d ≥ 0.60 | Independent samples t-test | 2024-01-15 | Implemented |
| F3.2 | Falsification-Protocol-1 | Interoceptive task specificity falsification | Advantage ≥28% OR η² ≥ 0.20 | Two-way mixed ANOVA | 2024-01-15 | Implemented |
| F3.3 | Falsification-Protocol-1 | Threshold gating necessity falsification | Reduction ≥25% OR d ≥ 0.75 | Paired t-test | 2024-01-15 | Implemented |
| F3.4 | Falsification-Protocol-1 | Precision weighting necessity falsification | Reduction ≥20% OR d ≥ 0.65 | Paired t-test | 2024-01-15 | Implemented |
| F3.5 | Falsification-Protocol-1 | Computational efficiency trade-off falsification | Retention ≥85%, gain ≥30% | TOST non-inferiority, efficiency ratio t-test | 2024-01-15 | Implemented |
| F3.6 | Falsification-Protocol-1 | Sample efficiency in learning falsification | Time ≤200 trials, HR ≥ 1.45 | Log-rank test | 2024-01-15 | Implemented |
| F5.1 | Falsification-Protocol-1 | Threshold emergence falsification | ≥60% develop multi-timescale, separation ≥3×, α ≥ 0.8 | Binomial test | 2024-01-15 | Implemented |
| F5.2 | Falsification-Protocol-1 | Precision emergence falsification | Mean r ≥ 0.5 | Correlation test | 2024-01-15 | Implemented |
| F5.3 | Falsification-Protocol-1 | Interoceptive emergence falsification | Gain ratio ≥ 0.8 | t-test | 2024-01-15 | Implemented |
| F5.4 | Falsification-Protocol-1 | Multi-timescale integration emergence falsification | ≥60% develop multi-timescale, separation ≥3× | Binomial test | 2024-01-15 | Implemented |
| F5.5 | Falsification-Protocol-1 | APGI-like feature clustering falsification | Cumulative variance ≥70%, min loading ≥0.60 | PCA, scree plot | 2024-01-15 | Implemented |
| F5.6 | Falsification-Protocol-1 | Non-APGI architecture failure falsification | Difference ≥40%, d ≥ 0.85 | t-test | 2024-01-15 | Implemented |
| F6.1 | Falsification-Protocol-1 | Intrinsic threshold behavior falsification | LTCN transition ≤50ms, delta ≥ 0.60 | Mann-Whitney U test | 2024-01-15 | Implemented |
| F6.2 | Falsification-Protocol-1 | Intrinsic temporal integration falsification | LTCN window ≥150ms, ratio ≥4×, R² ≥0.85 | Wilcoxon signed-rank test | 2024-01-15 | Implemented |
| F6.5 | Falsification-Protocol-1 | Hysteresis behavior falsification | Hysteresis 0.05-0.30 | Hysteresis range test | 2024-01-15 | Implemented |
| V8.1 | Validation_Protocol_8 | Interoceptive-exteroceptive bias falsification | Mean difference ≥10% | Paired t-test | 2024-01-15 | Implemented |
| V8.2 | Validation_Protocol_8 | Parameter reliability falsification | Test-retest reliability r ≥ 0.70 | Pearson correlation | 2024-01-15 | Implemented |
| V8.3 | Validation_Protocol_8 | Factor structure falsification | 4-factor solution accounts for ≥70% variance | Factor analysis | 2024-01-15 | Implemented |
| V8.4 | Validation_Protocol_8 | Disorder parameter validation | All parameters within paper range ±10% | Cross-check with disorder table | 2024-01-15 | Implemented |

## Implementation Notes

### Validation Protocol 1 (V1.x)

- All criteria V1.1-V1.6, V2.1-V2.3, F1.x, F2.x, F3.x, F5.x, F6.x implemented
- F2.x methods (F2.1-F2.5) now require parameters instead of using placeholder values
- F3.x methods (F3.1-F3.6) now require parameters instead of using placeholder values
- **Completion: 62%**

### Validation Protocol 2 (V2.x)

- Bayesian model comparison and parameter recovery validation
- **Completion: 71%**

### Validation Protocol 3 (V3.x)

- Active Inference agent simulations
- **Completion: 82%**

### Validation Protocol 4 (V4.x)

- Information theoretic phase transition analysis
- **Completion: 74%**

### Validation Protocol 5 (V5.x)

- Evolutionary emergence analytical validation
- **Completion: 66%**

### Validation Protocol 6 (V6.x)

- Neural network inductive bias computational benchmark
- **Completion: 68%**

### Validation Protocol 7 (V7.x and P7.x)

- Original V7.1 and V7.2 criteria retained (TMS and pharmacological tests)
- Replaced F1.x-F6.x criteria with Protocol 2 paper criteria:
  - P7.1: Propofol P3b:MMN ratio test
  - P7.2: Ketamine MMN suppression test
  - P7.3: Psilocybin P3b enhancement test
- **Completion: 79%**

### Validation Protocol 8 (V8.x)

- Psychophysical threshold estimation protocol
- **Completion: 77%**

### Validation Protocol 9 (V9.x)

- Convergent neural signatures priority 1 empirical roadmap
- P5 (mutual information) and P6 (bandwidth constraint) already implemented
- P7 (optimal threshold test) newly implemented using Neyman-Pearson criterion
- **Completion: 73%**

### Validation Protocol 10 (V10.x)

- Causal manipulations TMS pharmacological priority 2
- Subliminal priming dissociation test (SubliminalPrimingMeasure class) already implemented
- **Completion: 70%**

### Validation Protocol 11 (V11.x)

- Quantitative model fits spiking LNN priority 3
- **Completion: 75%**

### Validation Protocol 12 (V12.x)

- Clinical cross-species convergence protocol 4
- **Completion: 76%**

- Disorder parameter validation cross-check function newly implemented
- Validates θₜ offset, Πⁱ modification, and arousal level against paper ranges

### Falsification Protocols

#### Falsification Protocol 1 (F1.x)

- APGI falsification active inference agents F1F2
- **Completion: 74%**

#### Falsification Protocol 2 (F2.x)

- Agent comparison convergence benchmark
- **Completion: 72%**

#### Falsification Protocol 3 (F3.x)

- Framework level multi-protocol
- **Completion: 65%**

#### Falsification Protocol 4 (F4.x)

- Information theoretic phase transition
- **Completion: 73%**

#### Falsification Protocol 5 (F5.x)

- Evolutionary plausibility standard 6
- **Completion: 70%**

#### Falsification Protocol 6 (F6.x)

- Neural network energy benchmark
- **Completion: 68%**

#### Falsification Protocol 7 (F7.x)

- Mathematical consistency equations
- **Completion: 63%**

#### Falsification Protocol 8 (F8.x)

- Parameter sensitivity identifiability
- **Completion: 60%**

#### Falsification Protocol 9 (F9.x)

- Neural signatures EEG P3b HEP
- **Completion: 62%**

#### Falsification Protocol 10 (F10.x)

- Cross-species scaling P12
- **Completion: 55%**

#### Falsification Protocol 11 (F11.x)

- Bayesian estimation parameter recovery
- **Completion: 67%**

#### Falsification Protocol 12 (F12.x)

- Clinical cross-species convergence protocol 4
- **Completion: 76%**
