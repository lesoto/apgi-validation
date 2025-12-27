# APGI Protocol 5: Evolutionary Emergence of APGI-like Architectures

## Overview

Protocol 5 tests whether APGI components emerge under evolutionary selection pressure. If APGI describes a computationally advantageous architecture, genetic algorithms optimizing for survival in complex environments should converge on APGI-like structures.

This protocol implements a complete evolutionary framework where:
1. Agent architectures are encoded as evolvable genomes
2. Agents compete across multiple challenging environments
3. Selection favors architectures that maximize fitness
4. APGI components either emerge or fail to spread

## Scientific Rationale

**Central Question**: Is APGI an evolutionarily optimal architecture for conscious access?

**Logic**: 
- If APGI components (threshold ignition, interoceptive weighting, somatic markers, precision gating) provide computational advantages, they should be positively selected
- Evolution is an unbiased test: it only cares about survival and reproduction
- Emergence of APGI-like features would support the framework's computational necessity
- Failure to emerge would suggest APGI is not functionally advantageous

**Why This Matters**:
- Distinguishes "descriptions" from "explanations" - APGI must explain why these features exist
- Tests computational efficiency claims
- Validates that APGI's complexity provides real benefits over simpler alternatives

## Installation

### Requirements

```bash
pip install numpy scipy pandas matplotlib seaborn tqdm
```

### Files

- `APGI-Protocol-5.py` - Main implementation
- This README

## Quick Start

```python
python APGI-Protocol-5.py
```

Default configuration:
- Population: 100 agents
- Generations: 500
- Mutation rate: 0.1
- Crossover rate: 0.7

**Expected runtime**: ~30-60 minutes (depending on hardware)

## Protocol Structure

### 1. Genome Architecture

Each agent has a genome encoding:

**Structural Genes** (Boolean - present or absent):
- `has_threshold`: Discrete ignition threshold vs continuous processing
- `has_intero_weighting`: β·Π_i term vs uniform signal weighting
- `has_somatic_markers`: Affective valuation vs neutral processing
- `has_precision_weighting`: Precision-gated vs direct processing

**Parameter Genes** (Continuous values):
- `theta_0`: Threshold baseline (0.2-0.8)
- `alpha`: Sigmoid steepness (2.0-10.0)
- `beta`: Somatic bias (0.5-2.0)
- `Pi_e_lr`: External precision learning rate (0.01-0.2)
- `Pi_i_lr`: Internal precision learning rate (0.01-0.2)
- `somatic_lr`: Somatic marker learning rate (0.01-0.3)

**Architecture Genes**:
- `n_hidden_layers`: Neural network depth (1-5)
- `hidden_dim`: Layer width (8-128)
- `ignition_cost`: Metabolic cost of ignition (0.05-0.15)

### 2. Test Environments

Agents are evaluated across three environments designed to test different APGI components:

#### Iowa Gambling Task
- **Tests**: Somatic markers (affective guidance)
- **Challenge**: 4 decks with different risk/reward profiles
- **Optimal strategy**: Learn to avoid risky options despite immediate rewards
- **APGI prediction**: Somatic markers provide valuation bias

#### Volatile Foraging Environment
- **Tests**: Precision weighting (tracking uncertainty)
- **Challenge**: Resource patches with changing quality
- **Optimal strategy**: Track environmental volatility and switch appropriately
- **APGI prediction**: Precision weighting enables adaptive foraging

#### Threat-Reward Tradeoff
- **Tests**: Interoceptive weighting (arousal as information)
- **Challenge**: Balance approach to rewards vs threat avoidance
- **Optimal strategy**: Use internal arousal to guide risk assessment
- **APGI prediction**: Interoceptive signals inform decisions

### 3. Fitness Function

```
Fitness = Σ(rewards - 0.2·metabolic_cost - 0.1·homeostatic_violations)
```

Components:
- **Rewards**: External reinforcement from environments
- **Metabolic cost**: Energy expenditure (ignition is expensive)
- **Homeostatic violations**: Interoceptive dysregulation

This multi-objective fitness ensures evolution balances multiple constraints.

### 4. Genetic Algorithm

**Selection**: Tournament selection (size 5) with elitism (top 5 preserved)

**Crossover**: Single-point crossover at 70% rate
- Combines structural and parametric genes from two parents
- Creates novel architectural combinations

**Mutation**: 
- Structural: Bit-flip mutations (toggle component presence)
- Parametric: Gaussian perturbations (±20%)
- Rate: 10% per gene per generation

## Predictions & Falsification

### Core Predictions

**P5a: Positive Selection for APGI Components**
- Threshold mechanism: s > 0.02
- Interoceptive weighting: s > 0.02  
- Somatic markers: s > 0.015
- Precision weighting: s > 0.015

Where s = selection coefficient (rate of frequency increase)

**P5b: Fixation by Generation 300**
- All components reach >80% frequency by generation 300
- Indicates strong selective advantage

**P5c: Combined Architecture Superiority**
- Full APGI (4 components) > Partial (1-3) > None (0)
- Synergistic benefits from integration

**P5d: Emergence Order**
- Predicted sequence: Threshold → Precision → Interoceptive → Somatic
- Rationale: Basic gating before embodied refinement

### Falsification Criteria

**F5.1**: Threshold mechanism fails to reach >60% by generation 500
- **Interpretation**: Ignition provides no selective advantage
- **Implication**: APGI's discrete threshold is not computationally necessary

**F5.2**: Interoceptive weighting shows negative selection (s < 0)
- **Interpretation**: Embodied priority is maladaptive
- **Implication**: Body-to-mind influence harms performance

**F5.3**: Somatic markers never exceed 50% frequency
- **Interpretation**: Affective valuation provides no advantage
- **Implication**: Emotion-free decision-making is superior

**F5.4**: Continuous architectures achieve equal/better fitness than threshold
- **Interpretation**: Discrete ignition hypothesis falsified
- **Implication**: Graded processing is equally or more efficient

## Understanding Results

### Output Files

1. **protocol5_results.json**
   - Final statistics
   - Selection coefficients
   - Fixation generations
   - Falsification report

2. **protocol5_history.csv**
   - Fitness trajectory over generations
   - Architecture frequencies over time
   - Diversity metrics

3. **protocol5_evolution_results.png**
   - Comprehensive visualization
   - 12-panel figure showing all key metrics

### Interpreting Visualizations

**Panel 1: Fitness Evolution**
- Rising best/mean fitness = successful adaptation
- Plateau = local optimum or convergence
- Sustained increase = ongoing innovation

**Panels 2-5: Architecture Frequencies**
- Upward trajectory = positive selection
- Downward = negative selection  
- Plateau at high frequency = fixation
- Plateau at low frequency = elimination

**Panel 6: Diversity**
- High diversity = exploration phase
- Decreasing diversity = convergence
- Persistent diversity = stable polymorphism

**Panel 7: Selection Coefficients**
- Positive bars = beneficial traits
- Negative bars = deleterious traits
- Magnitude = strength of selection

**Panel 8: Fixation Generations**
- Earlier fixation = stronger selection
- Order reveals evolutionary sequence
- Never fixing = neutral or context-dependent

**Panel 9: Final Composition**
- Pie chart shows equilibrium state
- Balanced = multiple strategies viable
- Skewed = strong selective sweep

## Advanced Usage

### Customizing Evolution

```python
# Longer evolution
optimizer = EvolutionaryOptimizer(
    population_size=200,
    n_generations=1000,
    mutation_rate=0.05,  # Lower for fine-tuning
    crossover_rate=0.8
)

# Different environments
optimizer.environments = [
    IowaGamblingTask(),
    YourCustomEnvironment()
]

history = optimizer.run_evolution()
```

### Analyzing Specific Architectures

```python
# Extract best genome
best_genome = history['best_genomes'][-1]

# Create agent
agent = EvolvableAgent(best_genome)

# Test on specific environment
env = IowaGamblingTask()
obs = env.reset()

for t in range(100):
    action, ignition, cost = agent.decide(
        obs, 
        external_signal=0.1,
        internal_signal=0.05
    )
    reward, intero_cost, next_obs, done = env.step(action)
    agent.update(obs, action, reward, 0.1, intero_cost)
    obs = next_obs
    if done:
        break

print(f"Total reward: {sum(agent.reward_history)}")
print(f"Ignition rate: {np.mean(agent.ignition_history)}")
```

### Comparing Architectures

```python
# Create specific architectures
full_apgi = AgentGenome(
    has_threshold=True,
    has_intero_weighting=True,
    has_somatic_markers=True,
    has_precision_weighting=True,
    # ... other params
)

no_threshold = AgentGenome(
    has_threshold=False,
    has_intero_weighting=True,
    has_somatic_markers=True,
    has_precision_weighting=True,
    # ... other params
)

# Evaluate
optimizer = EvolutionaryOptimizer(population_size=2)
fitness_full = optimizer.evaluate_fitness(full_apgi)
fitness_no_thresh = optimizer.evaluate_fitness(no_threshold)

print(f"Full APGI: {fitness_full:.3f}")
print(f"No threshold: {fitness_no_thresh:.3f}")
```

## Expected Results (Example)

### Successful Validation

If APGI is correct, expect:

```
Final Architecture Frequencies:
  has_threshold: 0.95 ✅
  has_intero_weighting: 0.88 ✅
  has_somatic_markers: 0.82 ✅
  has_precision_weighting: 0.91 ✅

Selection Coefficients:
  has_threshold: 0.0234 ✅ (>0.02)
  has_intero_weighting: 0.0287 ✅ (>0.02)
  has_somatic_markers: 0.0189 ✅ (>0.015)
  has_precision_weighting: 0.0245 ✅ (>0.015)

Fixation Generations:
  has_threshold: 78
  has_precision_weighting: 112
  has_intero_weighting: 203
  has_somatic_markers: 267

FALSIFICATION: ✅ ALL CRITERIA PASSED
```

### Falsification Example

If APGI is incorrect, might see:

```
Final Architecture Frequencies:
  has_threshold: 0.43 ❌ (<0.60)
  has_intero_weighting: 0.38 ❌
  has_somatic_markers: 0.29 ❌
  has_precision_weighting: 0.67

Selection Coefficients:
  has_threshold: -0.0015 ❌ (negative!)
  has_intero_weighting: -0.0089 ❌ (negative!)
  has_somatic_markers: 0.0021 ❌ (<0.015)
  has_precision_weighting: 0.0198 ✅

FALSIFICATION: ❌ F5.1, F5.2, F5.3 FAILED
Interpretation: Continuous processing without threshold
outperforms APGI's discrete ignition. Interoceptive
weighting is maladaptive. Somatic markers provide
minimal benefit.
```

## Troubleshooting

### Low Final Fitness
- **Cause**: Environments too difficult or fitness function miscalibrated
- **Solution**: Adjust reward scaling or simplify environments
- **Check**: Are any architectures achieving positive fitness?

### No Convergence
- **Cause**: Population too small or mutation rate too high
- **Solution**: Increase population to 200, reduce mutation to 0.05
- **Check**: Is diversity staying constant (stuck in exploration)?

### Premature Convergence
- **Cause**: Population too small or selection too strong
- **Solution**: Increase diversity via larger population or higher mutation
- **Check**: Did fitness plateau early with low diversity?

### Memory Issues
- **Cause**: Large populations with long histories
- **Solution**: Reduce population or use shorter evaluation periods
- **Check**: Monitor RAM usage during evolution

## Interpreting for APGI Framework

### Strong Support Scenarios

1. **All components fix rapidly** (by gen 300)
   - APGI is computationally necessary
   - Components work synergistically
   
2. **Predicted emergence order holds**
   - Threshold → Precision → Intero → Somatic
   - Matches theoretical dependencies
   
3. **Full > Partial > None**
   - Integration provides benefits
   - Not just sum of parts

### Moderate Support

1. **Most components fix, one doesn't**
   - Core APGI valid
   - One component context-dependent
   - Example: Somatic markers only in some environments

2. **Slower fixation than predicted**
   - Components beneficial but not essential
   - Weaker selection than expected

### Weak Support / Falsification

1. **Continuous outperforms threshold**
   - Discrete ignition not necessary
   - Major blow to APGI

2. **Interoceptive weighting disfavored**
   - Body-to-mind influence harmful
   - Contradicts embodied priority

3. **Random fluctuation, no convergence**
   - Components neutral
   - APGI architectural choices don't matter

## Scientific Context

### Why Evolutionary Testing?

Evolution is blind to our theories. It only cares about fitness. If APGI components emerge:
- Not due to confirmation bias
- Not due to fitting parameters
- Result from genuine computational advantages

This is stronger evidence than:
- Fitting existing data (can always fit with enough parameters)
- Expert intuition (can be wrong)
- Mathematical elegance (nature doesn't care)

### Limitations

1. **Simplified environments**
   - Real evolution vastly more complex
   - Results are proof-of-concept

2. **Short timescales**
   - 500 generations << biological evolution
   - May miss long-term dynamics

3. **Fitness approximation**
   - Real fitness includes reproduction, not just survival
   - Social/cultural factors absent

4. **No co-evolution**
   - Environments fixed
   - Real evolution involves arms races

### Relation to Other Protocols

- **Protocol 1**: Tests APGI predictions on data
- **Protocol 2**: Tests APGI fits to empirical patterns
- **Protocol 5**: Tests APGI computational necessity
- **Protocol 7**: Tests APGI causal mechanisms
- **Protocol 8**: Tests APGI individual differences

Protocol 5 is unique in testing *why* APGI architecture exists, not just whether it describes data.

## Citation

If using this protocol:

```
APGI Research Team (2025). Protocol 5: Evolutionary Emergence of 
APGI-like Architectures. APGI Falsification Framework v1.0.
```

## References

### Genetic Algorithms
- Mitchell, M. (1998). *An Introduction to Genetic Algorithms*
- Goldberg, D. E. (1989). *Genetic Algorithms in Search*

### Evolutionary Robotics  
- Nolfi & Floreano (2000). *Evolutionary Robotics*
- Cliff et al. (1993). From Animals to Animats

### Neural Architecture Search
- Stanley & Miikkulainen (2002). Evolving Neural Networks through NEAT
- Real et al. (2019). Regularized Evolution for Image Classifier Architecture Search

### Consciousness Evolution
- Feinberg & Mallatt (2016). The Ancient Origins of Consciousness
- Ginsburg & Jablonka (2019). The Evolution of the Sensitive Soul

## Contact

For questions, issues, or contributions:
- Open an issue on the repository
- Contact: APGI Research Team

## License

MIT License - See LICENSE file

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Status**: Production Ready
