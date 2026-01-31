
# APGI Protocol 3: Active Inference Agent Simulations

## Overview

Protocol 3 implements complete active inference agents to test whether APGI's integration of interoceptive precision weighting and global workspace ignition produces measurable adaptive advantages in decision-making tasks.

**Core Question**: Does the APGI architecture (hierarchical generative models + precision-weighted interoception + ignition gate) outperform alternative architectures in tasks requiring integration of external and internal information?

## Architecture

### APGI Active Inference Agent

The complete APGI agent implements:

```text
┌─────────────────────────────────────────────────────────────┐
│                    APGI AGENT ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Hierarchical Generative Models                              │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │  Exteroceptive  │        │  Interoceptive  │            │
│  │   (3 levels)    │        │   (3 levels)    │            │
│  │                 │        │                 │            │
│  │  Context        │        │  Homeostatic    │            │
│  │  Objects        │        │  Organs         │            │
│  │  Sensory        │        │  Visceral       │            │
│  └────────┬────────┘        └────────┬────────┘            │
│           │                          │                      │
│           └──────────┬───────────────┘                      │
│                      ▼                                      │
│           Precision Weighting                               │
│           Πᵉ(t) | Πⁱ(t)                                     │
│                      │                                      │
│                      ▼                                      │
│           Surprise Accumulator                              │
│           S_t = Πᵉ | εᵉ | + β·Πⁱ | εⁱ | │
│                      │                                      │
│                      ▼                                      │
│           Global Workspace Ignition                         │
│           if S_t > θ_t: BROADCAST                           │
│                      │                                      │
│           ┌──────────┴──────────┐                          │
│           ▼                     ▼                           │
│    Explicit Policy      Somatic Markers                     │
│    (Conscious)          M(context,action)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Hierarchical Generative Models**
   - Exteroceptive: Sensory → Objects → Context
   - Interoceptive: Visceral → Organs → Homeostatic
   - 3-level predictive processing architecture

2. **Dynamic Precision Weighting**
   - Πᵉ(t): Exteroceptive precision (learned from prediction error variance)
   - Πⁱ(t): Interoceptive precision (learned from interoceptive error variance)
   - Adaptive adjustment based on reliability

3. **Somatic Marker System**

   - Learns M(context, action) → predicted interoceptive outcome
   - Influences action selection via somatic bias β = 1.2
   - Updated via prediction error: actual_intero - predicted_intero

4. **Global Workspace Ignition**

   - Surprise accumulation: dS/dt = -S/τ + Πᵉ | εᵉ | + β·Πⁱ | εⁱ |
   - Ignition probability: P(ignition) = σ(α(S_t - θ_t))
   - Threshold adaptation: balances metabolic cost vs information value

5. **Dual Policy System**

   - Explicit policy: Workspace-based deliberation (when ignited)
   - Implicit policy: Habitual sensorimotor mapping (default)

6. **Memory Systems**

   - Working memory: Limited capacity (7 items)
   - Episodic memory: Emotional tagging for retrieval

## Installation

### Dependencies

```bash
pip install numpy scipy torch matplotlib seaborn pandas tqdm
```

### Required Versions

- Python 3.8+
- PyTorch 1.9+
- NumPy 1.20+
- SciPy 1.7+

## Usage

### Quick Start

```python
from APGI_Protocol_3 import main

# Run complete experiment (default: 20 agents × 4 types × 3 environments)

results = main()
```

### Custom Configuration

```python
from APGI_Protocol_3 import AgentComparisonExperiment


# Custom experiment setup


experiment = AgentComparisonExperiment(
    n_agents=50,        # More agents for robustness
    n_trials=200        # Longer episodes
)


# Run experiment


results = experiment.run_full_experiment()


# Analyze specific predictions


analysis = experiment.analyze_predictions(results)


# Check falsification


falsification = experiment.check_falsification(results, analysis)
```text


### Individual Agent Testing


```python
from APGI_Protocol_3 import (
    APGIActiveInferenceAgent,
    IowaGamblingTaskEnvironment
)


# Create agent


agent = APGIActiveInferenceAgent({
    'beta': 1.2,
    'theta_init': 0.5,
    'alpha': 8.0,
    'n_actions': 4
})


# Create environment


env = IowaGamblingTaskEnvironment(n_trials=100)


# Run episode


observation = env.reset()
for trial in range(100):
    action = agent.step(observation)
    reward, intero_cost, next_obs, done = env.step(action)
    agent.receive_outcome(reward, intero_cost, next_obs)
    observation = next_obs
```text


## Components


### Agent Types


1. **APGIActiveInferenceAgent**
   - Full APGI architecture
   - Interoceptive precision weighting
   - Global workspace ignition
   - Somatic markers

2. **StandardPPAgent**
   - Predictive processing without ignition
   - Continuous conscious access
   - No threshold mechanism

3. **GWTOnlyAgent**

   - Ignition based only on external surprise
   - No interoceptive precision weighting
   - No somatic markers

4. **ActorCriticAgent**

   - Standard reinforcement learning baseline
   - No predictive processing
   - Simple policy gradient


### Task Environments


#### 1. Iowa Gambling Task (IGT)


Simulates Bechara et al.'s classic decision-making paradigm with interoceptive costs:

- **Deck A**: High variance, net negative, high interoceptive cost (0.8)
- **Deck B**: High variance, net negative, moderate interoceptive cost (0.5)
- **Deck C**: Low variance, net positive, low interoceptive cost (0.1)
- **Deck D**: Low variance, net positive, minimal interoceptive cost (0.05)

**Optimal strategy**: Prefer decks C & D (advantageous decks)

```python
from APGI_Protocol_3 import IowaGamblingTaskEnvironment

env = IowaGamblingTaskEnvironment(n_trials=100)
```text


#### 2. Volatile Foraging Environment


10×10 grid world with:
- Shifting reward patches (volatility = 0.1)
- Location-dependent homeostatic costs
- Requires adaptation to changing statistics

```python
from APGI_Protocol_3 import VolatileForagingEnvironment

env = VolatileForagingEnvironment(grid_size=10, volatility=0.1)
```text


#### 3. Threat-Reward Tradeoff


Four options with varying risk profiles:
- High reward options produce aversive interoceptive responses
- Threat accumulates and can trigger "panic" responses
- Tests somatic marker learning

```python
from APGI_Protocol_3 import ThreatRewardTradeoffEnvironment

env = ThreatRewardTradeoffEnvironment()
```text


## Predictions


### P3a: Convergence Speed in IGT


**Prediction**: APGI converges to advantageous deck selection in 50-80 trials (matching human performance), significantly faster than StandardPP (150+ trials).

**Mechanism**: Somatic markers provide early guidance before explicit reward learning.


### P3b: Interoceptive Dominance in Ignitions


 **Prediction**: 70-85% of ignition events are dominated by interoceptive precision-weighted signals (β·Πⁱ | εⁱ | > Πᵉ | εᵉ | ).

**Mechanism**: Body-based surprise is inherently more alarming/behaviorally relevant.


### P3c: Ignition Predicts Strategy Change


**Prediction**: Logistic regression shows ignition significantly predicts strategy changes beyond external prediction error alone (p < 0.01).

**Mechanism**: Workspace broadcast enables flexible policy updating.


### P3d: Adaptation Speed in Volatile Foraging


**Prediction**: APGI adapts 20-30% faster than StandardPP when reward statistics shift.

**Mechanism**: Selective ignition allocates computational resources efficiently.


## Falsification Criteria


### F3.1: No Performance Advantage


**Falsification**: APGI shows <5% performance advantage over best alternative in IGT.

**Implication**: Ignition mechanism provides no adaptive benefit.


### F3.2: Ignition Uncorrelated with Adaptive Behavior


**Falsification**: Ignition coefficient in logistic regression not significantly positive (p > 0.30).

**Implication**: Conscious access is epiphenomenal, not functional.


### F3.3: StandardPP Outperforms APGI


**Falsification**: Pure predictive processing (always conscious) achieves higher cumulative reward than APGI in IGT.

**Implication**: Selective ignition is counterproductive.


## Expected Outputs


### Console Output


```text
================================================================================
PROTOCOL 3: ACTIVE INFERENCE AGENT SIMULATIONS
================================================================================

Configuration:
  n_agents: 20
  n_trials: 100
  environments: ['IGT', 'Foraging', 'ThreatReward']

================================================================================
RUNNING EXPERIMENTS
================================================================================

 IGT - APGI: 100% | ██████████ | 20/20
 IGT - StandardPP: 100% | ██████████ | 20/20
...

================================================================================
ANALYZING RESULTS
================================================================================

Key Findings:
  P3a - APGI convergence: 65.3 trials
  P3b - Intero dominance: 78.4%
  P3d - Foraging advantage: 24.7%

================================================================================
FALSIFICATION ANALYSIS
================================================================================

✅ F3.1: APGI shows 18.3% advantage (PASSED)
✅ F3.2: Ignition coefficient = 0.42, p = 0.003 (PASSED)
✅ F3.3: APGI outperforms StandardPP (PASSED)

Overall: ✅ MODEL VALIDATED
```text


### Generated Files


1. **protocol3_results.png**
   - 3×4 subplot grid showing:
     - Cumulative rewards per environment
     - Final performance comparison
     - Convergence speed analysis
     - Interoceptive dominance in ignitions
     - Ignition-strategy relationship
     - Action distribution preferences
     - Summary statistics

2. **protocol3_results.json**
   ```json
   {
     "config": {...},
     "analysis": {
       "P3a_convergence": {...},
       "P3b_intero_dominance": {...},
       "P3c_ignition_strategy": {...},
       "P3d_adaptation": {...}
     },
     "falsification": {
       "F3.1": false,
       "F3.2": false,
       "F3.3": false,
       "overall_falsified": false
     }
   }
   ```text


## Performance Benchmarks


Expected performance metrics (based on specification):

| Metric | APGI | StandardPP | GWTOnly | ActorCritic | Human |
| --- | --- | --- | --- | --- | --- |
| IGT Convergence (trials) | 50-80 | 150+ | 100-120 | ~100 | 60-80 |
| Final Performance (% optimal) | 75-85% | 50-60% | 65-75% | 60-70% | 70-80% |
| Volatile Foraging Adaptation | 20-30% faster | baseline | 10-15% faster | baseline | - |
| Threat-Reward Avoidance | 85-92% | 60-70% | 70-80% | 65-75% | - |


## Advanced Usage


### Analyzing Ignition Patterns


```python

# Extract ignition data from APGI agents


ignition_data = []
for agent_result in results['IGT']['APGI']:
    ignition_data.extend(agent_result['ignition_events'])


# Analyze temporal patterns


import matplotlib.pyplot as plt

ignition_times = [event['trial'] for event in ignition_data]
plt.hist(ignition_times, bins=20)
plt.xlabel('Trial Number')
plt.ylabel('Ignition Frequency')
plt.title('Temporal Distribution of Ignitions')
plt.show()
```text


### Custom Environment Creation


```python
class CustomEnvironment:
    """Template for custom environments"""

    def __init__(self):
        self.state = None

    def reset(self) -> Dict:
        """Initialize environment"""
        self.state = self._initial_state()
        return self._get_observation()

    def step(self, action: int) -> Tuple[float, float, Dict, bool]:
        """
        Execute action

        Returns:
            reward: External reward
            intero_cost: Interoceptive cost (0-1)
            observation: {'extero': sensory, 'intero': visceral}
            done: Episode complete
        """
        # Update state
        self.state = self._update_state(action)

        # Compute outcomes
        reward = self._compute_reward()
        intero_cost = self._compute_intero_cost()

        # Generate observation
        observation = self._get_observation()

        done = self._check_done()

        return reward, intero_cost, observation, done

    def _get_observation(self) -> Dict:
        """Generate sensory and interoceptive observations"""
        return {
            'extero': np.random.randn(32),  # Sensory input
            'intero': np.random.randn(16)   # Visceral input
        }
```text


### Parameter Sensitivity Analysis


```python

# Test different somatic bias values


beta_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
results_by_beta = {}

for beta in beta_values:
    config = {'beta': beta, 'n_actions': 4}
    agent = APGIActiveInferenceAgent(config)
    # Run trials...
    results_by_beta[beta] = performance_metric


# Plot sensitivity


plt.plot(beta_values, [results_by_beta[b] for b in beta_values])
plt.xlabel('Somatic Bias (β)')
plt.ylabel('Performance')
plt.title('Sensitivity to Somatic Bias Parameter')
```text


## Computational Requirements


- **Memory**: ~2GB RAM (for default configuration)
- **Runtime**: ~10-15 minutes (20 agents, 100 trials, 3 environments, 4 agent types)
- **GPU**: Optional (speeds up neural network training)


### Optimization Tips


1. **Reduce agent count** for quick testing:
   ```python
   experiment = AgentComparisonExperiment(n_agents=5, n_trials=50)
   ```text

2. **Parallel execution** (not implemented but possible):
   ```python
   from multiprocessing import Pool
   # Run agents in parallel
   ```text

3. **Use GPU** if available:
   ```python
   # PyTorch automatically uses CUDA if available
   # No code changes needed
   ```text


## Theoretical Background


### APGI Framework


The APGI framework proposes that consciousness functions as an "emergency broadcast system" triggered when precision-weighted surprise exceeds an adaptive threshold:

**Core Equation**:
```text
 S_t = Πᵉ· | εᵉ | + β·Πⁱ· | εⁱ |

where:
  S_t: Accumulated surprise
  Πᵉ: Exteroceptive precision
  Πⁱ: Interoceptive precision
  εᵉ: Exteroceptive prediction error
  εⁱ: Interoceptive prediction error
  β: Somatic bias (typically 1.2)
```text

**Ignition Criterion**:
```text
if S_t > θ_t:
    Broadcast to global workspace
    Enable conscious access
    Update explicit policies
```text


### Key Innovations


1. **Precision Weighting**: Not all surprise is equal; reliability determines influence
2. **Somatic Bias**: Body-based signals amplified (β > 1) for survival relevance
3. **Adaptive Threshold**: Balances metabolic cost against information value
4. **Hierarchical Models**: Multi-level abstraction for efficient prediction


## Troubleshooting


### Common Issues


**Issue**: `RuntimeError: CUDA out of memory`
- **Solution**: Use CPU by setting `torch.device('cpu')` or reduce batch sizes

**Issue**: Agents converge too slowly
- **Solution**: Increase learning rates or reduce task difficulty

**Issue**: Ignitions never occur
- **Solution**: Check threshold initialization (θ_init should be ~0.5)

**Issue**: Poor performance across all agents
- **Solution**: Verify environment reward structure is appropriate


## Citation


If using this implementation in research:

```bibtex
@software{apgi_protocol3,
  title = {APGI Protocol 3: Active Inference Agent Simulations},
  author = {APGI Research Team},
  year = {2025},
  version = {1.0},
  url = {https://github.com/apgi-framework/protocols}
}
```text


## License


MIT License - See LICENSE file for details


## Contact


For questions, bug reports, or contributions:
- GitHub Issues: [github.com/apgi-framework/protocols/issues]
- Email: research@apgi-framework.org


## Acknowledgments


This implementation builds on:
- Free Energy Principle (Friston et al.)
- Global Workspace Theory (Baars, Dehaene)
- Somatic Marker Hypothesis (Damasio)
- Active Inference Framework (Friston, Parr, et al.)

---

**Version**: 1.0 (Production)
**Last Updated**: December 2025
**Status**: Ready for empirical validation
