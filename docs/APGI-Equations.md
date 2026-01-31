# APGI MATHEMATICAL FORMALIZATION

## Complete Equations with Dimensional Corrections

## PART 1: FOUNDATIONAL CONCEPTS & DEFINITIONS

### 1.1 Core Variables and Their Units

| Symbol | Name | Units | Interpretation | Measurable Via |
| --- | --- | --- | --- | --- |
| $S(t)$ | Accumulated Surprise | [nats] = [dimensionless] | Information content competing for access | EEG: P3b amplitude (μV); fMRI: BOLD signal |
| $\theta_t(t)$ | Dynamic Ignition Threshold | [nats] = [dimensionless] | Energy/information barrier to broadcast | Behavioral: detection threshold (contrast) |
| $\varepsilon^e(t)$ | Exteroceptive Prediction Error | [sensory units] | Discrepancy between input and prediction | EEG: prediction-related negativity |
| $\varepsilon^i(t)$ | Interoceptive Prediction Error | [AU] = [arbitrary internal units] | Mismatch between expected and actual body state | Physiological: heart rate, respiration variance |
| $\Pi^e(t)$ | Exteroceptive Precision | [1/error²] | Inverse variance of exteroceptive errors | Neural: gain of afferent pathway |
| $\Pi^i(t)$ | Interoceptive Precision | [1/error²] | Inverse variance of interoceptive errors | Physiological: interoceptive sensitivity (HEP) |
| $M(t)$ | Somatic Marker State | [dimensionless] | vmPFC representation of predicted homeostatic cost | fMRI: vmPFC BOLD activity |
| $A(t)$ | Arousal Level | [0,1] | Global state of neural readiness | Behavioral: vigilance; Physiological: pupil diameter |
| $B(t)$ | Broadcast Probability | [0,1] | Probability global ignition occurs at time t | EEG: P3b latency distribution |

---

### 1.2 Prediction Error (Foundational)

**Definition:**

$$\varepsilon(t) = x(t) - \hat{x}(t)$$

where:

- $x(t)$ = observed input (sensory or interoceptive)
- $\hat{x}(t)$ = predicted input from generative model

**Dimensions:** $[\text{sensory units}]$ for exteroceptive; $[\text{arbitrary units}]$ for interoceptive

---

### 1.3 Precision (Foundational)

**Definition (Inverse Variance):**
$$\Pi = \frac{1}{\sigma_\varepsilon^2}$$

**Dimensions:** $[1/(\text{error}^2)]$

**Biological interpretation:**

- High precision: prediction errors are tightly clustered (low variance) → high confidence
- Low precision: prediction errors are scattered (high variance) → low confidence
- Precision weighting prioritizes reliable signals

**Implementation:**

- Implemented via gain modulation: $\Pi \propto$ neuron gain / locus coeruleus firing rate / LC-derived norepinephrine
- Can be estimated from neural population variance
- Varies across individuals (COMT polymorphisms, developmental history)

---

### 1.4 Standardized Prediction Errors (z-scores)

**Definition for z-score representation:**

$$z(t) = \frac{\varepsilon(t) - \mu_\varepsilon(t)}{\sigma_\varepsilon(t)}$$

where:

- $\mu_\varepsilon(t)$ = running mean of prediction errors
- $\sigma_\varepsilon(t)$ = running standard deviation of prediction errors

**Dimensions:** [dimensionless]

**Note:** Z-scores are used for *visualization and understanding* but NOT for mathematical formulation. All mathematical equations use raw errors or squared errors for dimensional consistency.

---

## PART 2: CORE IGNITION SYSTEM (REVISED)

### 2.1 Accumulated Signal (Dimensionally Correct)

**BEFORE (Dimensionally Inconsistent):**
$$S_t = \Pi^e \cdot |z^e| + \Pi^i_{\text{eff}} \cdot |z^i|$$
Problem: $\Pi \cdot [\text{dimensionless}] = [1/\text{error}^2]$ ≠ signal strength

---

### AFTER (Dimensionally Correct - Option A: Information Theoretic)

$$S(t) = \frac{1}{2}\Pi^e(t) \cdot (\varepsilon^e(t))^2 + \frac{1}{2}\Pi^i_{\text{eff}}(t) \cdot (\varepsilon^i(t))^2$$

**Dimensional verification:**

- $\Pi^e \cdot (\varepsilon^e)^2 = [1/\text{error}^2] \times [\text{error}^2] = [\text{dimensionless}]$ ✓
- $S(t)$ now has units of [nats] (information-theoretic surprise)
- Comparable across sensory modalities (both dimensionless)
- Comparable to $\theta_t$ (also dimensionless) ✓

**Interpretation:**
$$S(t) = \underbrace{\frac{1}{2}\Pi^e(\varepsilon^e)^2}_{\text{Exteroceptive surprise}} + \underbrace{\frac{1}{2}\Pi^i_{\text{eff}}(\varepsilon^i)^2}_{\text{Interoceptive surprise, embodied}}$$

$S(t)$ represents the total **surprise** (information content of observations relative to predictions) weighted by **precision** (confidence in each stream).

**Key property:** When both $\varepsilon^e$ and $\varepsilon^i$ are near zero (predictions accurate), $S(t) \approx 0$. When either error is large relative to precision, $S(t)$ increases.

---

### 2.2 Effective Interoceptive Precision (With Somatic Modulation)

**BEFORE (Problematic):**
$$\Pi^i_{\text{eff}} = \Pi^i_{\text{baseline}} \cdot \exp(\beta \cdot M(c,a))$$
Problem: Exponential can be unbounded; no biological justification

---

**AFTER (Biologically Bounded):**

#### Option A: Sigmoid Modulation (Recommended for high flexibility)

$$\Pi^i_{\text{eff}}(t) = \Pi^i_{\text{baseline}} \cdot \left[1 + \beta \cdot \sigma(M(t) - M_0)\right]$$

where:

- $\sigma(x) = \frac{1}{1+e^{-x}}$ is the logistic sigmoid
- $M_0$ is the reference somatic marker level
- $\beta \in [0, 2]$ controls modulation strength

**Dimensional verification:**

- $\sigma(\cdot) = [\text{dimensionless}] \in (0,1)$
- $1 + \beta \sigma = [\text{dimensionless}] \in (1, 1+\beta)$
- $\Pi^i_{\text{eff}} = \Pi^i_{\text{baseline}} \times [\text{dimensionless}] = [1/\text{error}^2]$ ✓

**Biological justification:**

- Sigmoid bounded between 0 and 1 (plausible for neuromodulation)
- $\beta > 0$: positive somatic markers increase interoceptive precision (safe conditions)
- $\beta < 0$: negative somatic markers increase interoceptive precision (threat/salience)
- Flexible: can explain both "calm precision" and "anxious hypervigilance"

**Implementation:**

- $M(t)$ = vmPFC activity (homeostatic cost prediction)
- $\sigma(M)$ = output of insular gain-modulation circuitry
- Norepinephrine/dopamine tune $\beta$ (neuromodulatory state)

---

#### Option B: Linear Clipped Modulation (More parsimonious)

$$\Pi^i_{\text{eff}}(t) = \Pi^i_{\text{baseline}} \cdot \max(0.5, \min(2.0, 1 + \beta M(t)))$$

where $\beta \in [-1, 1]$ and clipping ensures $\Pi^i_{\text{eff}} \in [0.5, 2.0] \times \Pi^i_{\text{baseline}}$

**Advantage:** Computationally simpler, easier to interpret

**Disadvantage:** Non-smooth (kink at boundaries)

---

**Recommendation for papers:** Use **Option A (Sigmoid)** for main equations; mention Option B as alternative.

---

### 2.3 Ignition Criterion (Unified Continuous Formulation)

**BEFORE (Discrete-Continuous Mismatch):**

- "If $S_t > \theta_t$ then Ignition" (discrete)
- "Ignition probability = $\sigma(\alpha(S_t - \theta_t))$" (continuous)
- Incompatible formulations

---

### AFTER (Unified Stochastic Formulation)

### Definition 1: Deterministic Threshold Crossing (Low-Noise Limit)

$$\text{Ignition occurs when: } S(t) = \theta(t)$$

with dynamics:

$$\frac{dS}{dt} = -\tau_S^{-1} S(t) + \text{[input terms]}$$

$$\frac{d\theta}{dt} = \text{[dynamics]}$$

When noise is negligible, ignition is **all-or-none**: occurs at the moment when $S$ first crosses $\theta$ from below.

---

### Definition 2: Probabilistic Model (Realistic, Includes Noise)

$$P(\text{broadcast at time } t \mid S(t), \theta(t)) = \Phi\left(\frac{S(t) - \theta(t)}{\sigma_{\text{noise}}}\right)$$

where:

- $\Phi$ is the standard normal CDF (cumulative distribution function)
- $\sigma_{\text{noise}}$ is the effective noise standard deviation

**Interpretation:**

- When $S(t) \gg \theta(t)$: $P(\text{broadcast}) \approx 1$ (ignition nearly certain; effectively all-or-none)
- When $S(t) \approx \theta(t)$: $P(\text{broadcast}) \approx 0.5$ (ignition uncertain; probabilistic)
- When $S(t) \ll \theta(t)$: $P(\text{broadcast}) \approx 0$ (ignition impossible)

**Equivalence to Logistic Function:**
$$P(\text{broadcast}) = \frac{1}{1 + \exp(-\alpha(S(t) - \theta(t)))}$$

where $\alpha = 1/\sigma_{\text{noise}}$ relates noise standard deviation to ignition sharpness.

---

**Recommended formulation for papers:**

"Ignition is a probabilistic event whose probability increases sigmoidally as the accumulated surprise $S(t)$ exceeds the dynamic threshold $\theta(t)$:

$$P(\text{Global Ignition at time } t) = \sigma(\alpha(S(t) - \theta(t)))$$

where $\sigma$ is the logistic function and $\alpha \in (0, \infty)$ controls ignition sharpness. For $\alpha \to \infty$ (low noise), ignition becomes approximately all-or-none. For $\alpha \to 0$ (high noise), ignition becomes gradual. We estimate $\alpha$ from behavioral data (detection threshold statistics)."

---

## PART 3: COMPLETE DYNAMICAL SYSTEM (ALL EQUATIONS)

### 3.1 Accumulated Signal Dynamics

**Continuous-time ODE for surprise accumulation:**

$$\frac{dS}{dt} = -\tau_S^{-1} S(t) + \frac{1}{2}\Pi^e(t)(\varepsilon^e(t))^2 + \frac{1}{2}\Pi^i_{\text{eff}}(t)(\varepsilon^i(t))^2 + \sigma_S \xi_S(t)$$

**Component-by-component explanation:**

| Term | Meaning | Dynamics |
| --- | --- | --- |
| $-\tau_S^{-1} S(t)$ | Surprise decay | Leaky integration; information fades with time constant $\tau_S$ |
| $\frac{1}{2}\Pi^e(\varepsilon^e)^2$ | Exteroceptive surprise input | Sensory error weighted by sensory precision |
| $\frac{1}{2}\Pi^i_{\text{eff}}(\varepsilon^i)^2$ | Interoceptive surprise input | Body state error weighted by embodied precision |
| $\sigma_S \xi_S(t)$ | Stochastic noise | White noise with amplitude $\sigma_S$; $\xi_S \sim \mathcal{N}(0,1)$ |

**Biological implementation:**

- $\tau_S$ ≈ 500 ms (from LSM fading memory; Paper 1)
- Cortical pyramidal cells implement leaky integration
- Precision terms arise from gain modulation (LC-NA system)

**Dimensional verification:**

- $[-\tau_S^{-1} S] = [1/\text{time}] \times [\text{nats}] = [\text{nats/time}]$ ✓
- $[\Pi(\varepsilon)^2] = [1/\text{error}^2] \times [\text{error}^2] = [\text{dimensionless}] = [\text{nats/time}]$ ✓

**Note:** The $1/2$ factor appears in information theory (relating free energy to squared prediction error). It's conventional and dimensionally irrelevant but aids interpretation.

---

### 3.2 Dynamic Threshold (With Metabolic Coupling)

**BEFORE (Missing feedback from accumulated signal):**
$$\frac{d\theta}{dt} = \frac{\theta_0 - \theta_t}{\tau_\theta} + \gamma_M \cdot M(t) + \gamma_A \cdot A(t) + \eta_\theta(t)$$
Problem: Threshold doesn't respond to accumulated metabolic cost (S)

---

**AFTER (Complete Allostatic Regulation):**

$$\frac{d\theta}{dt} = \tau_\theta^{-1}(\theta_0(A(t)) - \theta(t)) + \gamma_M M(t) + \lambda S(t) + \sigma_\theta \xi_\theta(t)$$

**Component-by-component:**

| Term | Meaning | Function |
| --- | --- | --- |
| $\tau_\theta^{-1}(\theta_0(A) - \theta)$ | Homeostatic restoration | Threshold relaxes toward arousal-dependent baseline |
| $\gamma_M M(t)$ | Somatic marker influence | Anticipated homeostatic cost; lowers threshold if safety predicted |
| $\lambda S(t)$ | Metabolic cost feedback | **NEW**: As S increases, metabolic cost increases → threshold rises |
| $\sigma_\theta \xi_\theta(t)$ | Threshold noise | Stochasticity in neuromodulatory state |

**Key addition: $\lambda S(t)$ term**

This implements **negative feedback**: as information accumulates (S ↑), metabolic cost increases, threshold rises → prevents runaway ignitions from wasteful continuous broadcasting.

**Biological implementation:**

- $\theta_0(A)$ = baseline threshold set by arousal state (locus coeruleus)
- $M(t)$ = vmPFC interoceptive prediction (from anterior insula)
- $\lambda$ = metabolic coupling strength (tuned by metabolic state)
- Dopamine/acetylcholine modulate $\lambda$

**Arousal-dependent baseline:**

$$\theta_0(A(t)) = \theta_0^{\text{sleep}} + (1 - A(t))(\theta_0^{\text{alert}} - \theta_0^{\text{sleep}})$$

where:

- $A(t) = 0$ → deep sleep → $\theta_0 = \theta_0^{\text{sleep}}$ (low threshold, easy ignition for survival threats)
- $A(t) = 1$ → alert vigilance → $\theta_0 = \theta_0^{\text{alert}}$ (high threshold, conservative broadcast)

**Dimensional verification:**

- $[\tau_\theta^{-1}(\theta_0 - \theta)] = [1/\text{time}] \times [\text{dimensionless}] = [\text{nats/time}]$ ✓
- $[\gamma_M M] = [\text{dimensionless}] = [\text{nats/time}]$ ✓
- $[\lambda S] = [\text{dimensionless}] \times [\text{dimensionless}] = [\text{nats/time}]$ ✓

---

### 3.3 Somatic Marker Dynamics (NEW - Previously Undefined)

**BEFORE:** $M(c,a)$ appeared in equations without dynamics

---

### AFTER: Complete ODE for somatic marker state

$$\frac{dM}{dt} = \tau_M^{-1}(M^*(\varepsilon^i(t)) - M(t)) + \gamma_{\text{context}} C(t) + \sigma_M \xi_M(t)$$

**Component-by-component:**

| Term | Meaning | Implementation |
| --- | --- | --- |
| $\tau_M^{-1}(M^* - M)$ | Somatic state dynamics | M tracks predicted homeostatic cost; exponential approach |
| $M^*(\varepsilon^i)$ | Predicted somatic cost | Function of actual interoceptive state |
| $\gamma_{\text{context}} C(t)$ | Context modulation | Learned associations with environments |
| $\sigma_M \xi_M(t)$ | Fluctuations | Prediction uncertainty |

**Target somatic marker (predicted homeostatic cost):**

$$M^*(\varepsilon^i(t)) = \tanh(\beta_M \varepsilon^i(t))$$

**Rationale:**

- $\tanh$ maps unbounded interoceptive errors to bounded somatic cost $\in (-1, 1)$
- Large positive $\varepsilon^i$ (elevated heart rate, etc.) → $M^* \to 1$ (high predicted cost)
- Large negative $\varepsilon^i$ (suppressed physiology, etc.) → $M^* \to -1$ (cost of constraint)
- $\beta_M$ is sensitivity: how much does body state mismatch matter?

**Context modulation:**
$$C(t) = \sum_k \omega_k \phi_k(t)$$

where $\phi_k(t)$ are learned context features (e.g., environmental safety cues) with weights $\omega_k$ learned from experience.

**Biological implementation:**

- vmPFC encodes M (interoceptive self-model)
- Insula provides $\varepsilon^i$ input to vmPFC
- Amygdala/basolateral inputs provide context $C(t)$
- Dopamine modulates learning rates for context weights

**Dimensional verification:**

- $[\tau_M^{-1}(M^* - M)] = [1/\text{time}] \times [\text{dimensionless}] = [\text{nats/time}]$ ✓
- $[\tanh(\cdot)] = [\text{dimensionless}]$ ✓

**Timescale:** $\tau_M \approx 1-2$ seconds (vmPFC integration of visceral signals)

---

### 3.4 Arousal Dynamics

**BEFORE:** $A(t)$ appeared in threshold equation without dynamics

---

### AFTER: Complete ODE for arousal state

$$\frac{dA}{dt} = \tau_A^{-1}(A_{\text{target}}(t) - A(t)) + \sigma_A \xi_A(t)$$

where the target arousal depends on multiple factors:

$$A_{\text{target}}(t) = A_{\text{circ}}(t) + g_{\text{stim}}(\max \varepsilon) + \int_0^t K(t-s) \varepsilon^i(s) ds + \text{[learning]}$$

**Components:**

| Term | Meaning | Range | Source |
| --- | --- | --- | --- |
| $A_{\text{circ}}(t)$ | Circadian rhythm | [0,1] | Suprachiasmatic nucleus |
| $g_{\text{stim}}(\max \varepsilon)$ | Stimulus-driven arousal | [0,1] | Salience of current input |
| $\int K(t-s) \varepsilon^i(s) ds$ | Interoceptive arousal | [0,1] | Recent body state history |
| Learning term | Experience-dependent baseline | [0,1] | Threat learning from amygdala |

**Circadian component:**
$$A_{\text{circ}}(t) = 0.5 + 0.5 \cos(2\pi(t - t_{\text{peak}})/24h)$$

where $t_{\text{peak}}$ ≈ 10 AM (peak human alertness).

**Stimulus-driven component:**
$$g_{\text{stim}}(e) = \min(1, 0.1 + 0.5 \cdot \sigma(5(e - e_{\text{mean}})))$$

Response increases sigmoidally with stimulus surprise relative to baseline.

**Interoceptive component:**
$$\int_0^t K(t-s) \varepsilon^i(s) ds$$

with memory kernel (e.g., exponential decay):
$$K(t) = \frac{1}{\tau_{\text{int}}} e^{-t/\tau_{\text{int}}}$$

Recent body state perturbations increase arousal; effect decays over $\tau_{\text{int}} \approx 5-10$ min.

**Biological implementation:**

- Locus coeruleus (LC) firing rate encodes $A(t)$
- LC receives inputs from: SCN (circadian), thalamus (stimulus), vagus nerve (interoceptive), amygdala (learned threat)
- Orexin/hypocretin in lateral hypothalamus coordinates these inputs

**Dimensional verification:**

- $[\tau_A^{-1}(A_{\text{target}} - A)] = [1/\text{time}] \times [\text{dimensionless}] = [1/\text{time}]$ ✓

**Timescales:**

- $\tau_A^{\text{fast}} \approx 100-500$ ms (stimulus-driven phasic response)
- $\tau_A^{\text{slow}} \approx 5-30$ min (homeostatic baseline adjustment)

---

### 3.5 Precision Dynamics

**Exteroceptive precision:**

$$\frac{d\Pi^e}{dt} = \alpha_\Pi^e (\Pi^{e*}(\text{task}) - \Pi^e) + \sigma_{\Pi^e} \xi_{\Pi^e}$$

where target precision depends on task demands:

$$\Pi^{e*}(\text{task}) = \Pi^{e,\text{baseline}} \cdot \begin{cases}

1.0 & \text{if task-irrelevant} \\
1.5 & \text{if task-relevant} \\
2.5 & \text{if task-critical (near threshold)}
\end{cases}$$

**Interoceptive precision baseline:**

$$\frac{d\Pi^i_{\text{baseline}}}{dt} = \alpha_\Pi^i (\Pi^{i*}(\text{threat}) - \Pi^i_{\text{baseline}}) + \sigma_{\Pi^i} \xi_{\Pi^i}$$

where:

$$\Pi^{i*}(\text{threat}) = \Pi^{i,0} \cdot \left(1 + \rho \cdot M(t)\right)^+$$

Threat-related somatic markers increase interoceptive precision (gain tuning for body monitoring).

**Biological implementation:**
- Posterior insula (primary interoceptive cortex) processes precision based on interoceptive salience
- Anterior insula (emotion integration) modulates this based on M
- Learning rate $\alpha_\Pi$ controlled by dopamine (reward prediction error)

---

### 3.6 Running Statistics (Z-Score Normalization)

**For normalization of prediction errors to z-scores, maintain running moments:**

**Running mean (exponential moving average):**
$$\frac{d\mu_\varepsilon^e}{dt} = \alpha_\mu (\varepsilon^e(t) - \mu_\varepsilon^e(t))$$

with typical learning rate $\alpha_\mu = 0.01$ (slow adaptation to baseline shift).

**Running variance (second moment):**
$$\frac{d(\sigma_\varepsilon^e)^2}{dt} = \alpha_\sigma ((\varepsilon^e(t) - \mu_\varepsilon^e)^2 - (\sigma_\varepsilon^e)^2)$$

with typical learning rate $\alpha_\sigma = 0.005$ (even slower adaptation to variance change).

**Interpretation:**
- System maintains exponential-moving-average statistics of prediction error distributions
- These adapt slowly to non-stationary input distributions
- Allow comparison across modalities with different scales
- Z-scores computed at each time: $z^e(t) = (\varepsilon^e(t) - \mu_\varepsilon^e(t))/\sigma_\varepsilon^e(t)$

**Dimensional verification:**
- $[d\mu/dt] = [\text{error/time}]$ ✓
- $[d(\sigma)^2/dt] = [\text{error}^2/\text{time}]$ ✓

---

## PART 4: IGNITION PROBABILITY & BROADCAST

### 4.1 Broadcast Probability (Probabilistic Formulation)

$$P(\text{Global Ignition at time } t) = B(t) = \sigma(\alpha(S(t) - \theta(t)))$$

where $\sigma(x) = \frac{1}{1+e^{-x}}$ is the logistic sigmoid.

**Interpretation of $\alpha$ (steepness parameter):**

$$\alpha = \frac{1}{\sigma_{\text{noise}}}$$

where $\sigma_{\text{noise}}$ is the effective noise standard deviation.

**Biological values:**
- $\alpha \approx 0.1$ (high noise, gradual transition) → Large olfactory/gustatory thresholds (noisy signals)
- $\alpha \approx 1.0$ (moderate noise, sharp transition) → Typical visual/auditory (well-processed)
- $\alpha \approx 5.0$ (low noise, abrupt transition) → Pain/threat (critical for survival)

Estimate $\alpha$ from behavioral data: fit logistic to detection probability curve.

---

### 4.2 Deterministic (All-or-None) Approximation

In the **low-noise limit** ($\alpha \to \infty$):

$$B(t) \to \begin{cases}

1 & \text{if } S(t) > \theta(t) \\
0 & \text{if } S(t) < \theta(t) \\
\text{undefined} & \text{if } S(t) = \theta(t)
\end{cases}$$

Use indicator function:
$$B_{\text{deterministic}}(t) = \mathbb{1}(S(t) > \theta(t))$$

**When to use:**
- Theoretical analysis (when noise negligible)
- Bifurcation analysis (finding critical points)
- Low-cost approximation for large-scale simulations

---

### 4.3 Reporting/Consciousness Definition

**Definition: Global Ignition Event**

A global ignition occurs when:
1. $S(t) > \theta(t)$ (signal exceeds threshold)
2. $B(t) > 0.5$ (probability exceeds 50%)
3. Ignition persists for $\geq 50$ ms (minimum duration for reportability)

**Operational definition (behavioral):**

Conscious access to content $C$ is inferred when:
- Subject reports $C$ (verbal/behavioral)
- Timing coincides with predicted ignition ($\pm 100$ ms)
- Report is influenced by stimulus characteristics in predicted direction

**Operational definition (neural):**

Neural correlate of consciousness (NCC) identified as:
- Broadband increase (40-200 Hz) in frontoparietal regions
- P3b component (EEG, 200-600 ms latency, >2 μV)
- Increased long-range coherence (distance >5 cm, <5 Hz phase lag)

---

## PART 5: ADDITIONAL DERIVED QUANTITIES

### 5.1 Latency to Ignition

**Context:** Given initial conditions $S(0) = S_0, \theta(0) = \theta_0$, and constant input $I = \frac{1}{2}\Pi^e (\varepsilon^e)^2$, when does $S$ first reach threshold?

**Deterministic equation (ignoring noise):**
$$\frac{dS}{dt} = -\tau_S^{-1} S + I$$

**Steady state:** $S_\infty = I \tau_S$

**Solution (linear ODE):**
$$S(t) = (S_0 - I\tau_S) e^{-t/\tau_S} + I\tau_S$$

**Time to reach threshold $\theta$:**

Set $S(t^*) = \theta$ and solve for $t^*$:

$$\theta = (S_0 - I\tau_S) e^{-t^*/\tau_S} + I\tau_S$$

$$(S_0 - I\tau_S) e^{-t^*/\tau_S} = \theta - I\tau_S$$

$$e^{-t^*/\tau_S} = \frac{\theta - I\tau_S}{S_0 - I\tau_S}$$

$$t^* = \tau_S \ln\left(\frac{S_0 - I\tau_S}{\theta - I\tau_S}\right)$$

**Requirements for ignition:** $S_0 > \theta$ (already above) OR $I\tau_S > \theta$ (input drives above threshold)

**Interpretation:**
- If $I = 0$ (no input), $t^* = \tau_S \ln(S_0/\theta)$ (decay to threshold)
- If $I$ large (strong signal), $t^*$ small (rapid ignition)
- If $\theta$ high (alert state), $t^*$ large (slower ignition)

---

### 5.2 Information Cost of Ignition

**Free energy cost of global broadcast (rough estimate):**

$$\text{Metabolic cost} \propto \int_0^{T_{\text{ignition}}} S(t) \, dt$$

where $T_{\text{ignition}}$ is the duration of the ignition event.

This is a rough approximation; full metabolic cost calculation requires:
- Detailed neural implementation (how many neurons, firing rates)
- ATP consumption (19.1 ATP per spike in cerebral cortex)
- Metabolic rate increase during broadcast

**Formal approach:** Use thermodynamic dissipation formula:
$$J = T \cdot \dot{S}_{\text{entropy}} = T \cdot \frac{dF}{dt}$$

where $F$ is free energy and $J$ is heat dissipation.

---

### 5.3 Level-Specific Dynamics

For hierarchical extension, replicate core equations at each level $\ell$:

**At level $\ell$:**

$$S_\ell(t) = \frac{1}{2}\Pi_\ell^e(t)(\varepsilon_\ell^e(t))^2 + \frac{1}{2}\Pi_\ell^i(t)(\varepsilon_\ell^i(t))^2$$

$$\frac{dS_\ell}{dt} = -\tau_\ell^{-1} S_\ell + \text{[input]} + \sigma_{S_\ell} \xi_{S_\ell}$$

$$\frac{d\theta_\ell}{dt} = -\tau_{\theta,\ell}^{-1}(\theta_\ell - \theta_{0,\ell}) + \lambda_\ell S_\ell + \text{[cross-level inputs]}$$

**Cross-level coupling (top-down precision modulation):**

$$\Pi_{\ell-1}^e(t) \leftarrow \Pi_{\ell-1}^{e,\text{baseline}} \cdot (1 + \beta_{\text{cross}} B_\ell(t))$$

where $B_\ell(t) = \sigma(\alpha_\ell(S_\ell - \theta_\ell))$ is the ignition probability at higher level $\ell$.

**Interpretation:** Higher-level ignition (broad context) increases precision weighting at lower levels (selective attention).

---

## PART 6: SUMMARY TABLE - ALL EQUATIONS

| System Component | Equation | Units | Key Parameters |
| --- | --- | --- | --- |
| **Accumulated Signal** | $S = \frac{1}{2}\Pi^e(\varepsilon^e)^2 + \frac{1}{2}\Pi^i_{\text{eff}}(\varepsilon^i)^2$ | [nats] | $\Pi^e, \Pi^i$ |
| **Signal Dynamics** | $\frac{dS}{dt} = -\tau_S^{-1}S + \frac{1}{2}\Pi^e(\varepsilon^e)^2 + \frac{1}{2}\Pi^i_{\text{eff}}(\varepsilon^i)^2$ | [nats/s] | $\tau_S \approx 0.5$ s |
| **Effective Precision** | $\Pi^i_{\text{eff}} = \Pi^i_{\text{baseline}} [1 + \beta \sigma(M-M_0)]$ | [1/error²] | $\beta \in [0,2]$ |
| **Ignition Threshold** | $\frac{d\theta}{dt} = -\tau_\theta^{-1}(\theta-\theta_0(A)) + \gamma_M M + \lambda S$ | [nats/s] | $\tau_\theta \approx 1$ s, $\lambda > 0$ |
| **Somatic Marker** | $\frac{dM}{dt} = -\tau_M^{-1}(M-\tanh(\beta_M\varepsilon^i)) + \gamma_C C$ | [1/s] | $\tau_M \approx 1-2$ s |
| **Arousal** | $\frac{dA}{dt} = -\tau_A^{-1}(A-A_{\text{target}})$ | [1/s] | $\tau_A^{\text{fast}} \approx 0.2$ s |
| **Broadcast Probability** | $B = \sigma(\alpha(S-\theta)) = \frac{1}{1+e^{-\alpha(S-\theta)}}$ | [0,1] | $\alpha \approx 0.5-2$ |
| **Arousal Baseline** | $\theta_0(A) = \theta^{\text{sleep}} + (1-A)(\theta^{\text{alert}} - \theta^{\text{sleep}})$ | [nats] | $\theta^{\text{sleep}}, \theta^{\text{alert}}$ |

---
