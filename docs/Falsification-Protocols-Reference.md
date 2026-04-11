# Falsification Protocols API Reference

## Overview

The falsification protocols package provides a comprehensive suite of testing protocols designed to potentially falsify APGI theory predictions. Each protocol targets specific aspects of the theory with rigorous empirical tests.

## Parameter Range Validation Table

To prevent researchers from entering non-biological values, the following tables specify valid parameter ranges for each class constructor:

### HierarchicalGenerativeModel Parameter Ranges

| Parameter | Valid Range | Biological Constraint | Error if Outside Range |
| --------- | ----------- | -------------------- | ---------------------- |
| `learning_rate` | [0.001, 0.1] | Synaptic plasticity bounds | `ValueError: learning_rate must be in [0.001, 0.1]` |
| `model_type` | {"extero", "intero"} | Modal specificity | `ValueError: model_type must be 'extero' or 'intero'` |
| Level dimensions | [8, 256] | Neural population sizes | `ValueError: dimension must be in [8, 256]` |

### APGIActiveInferenceAgent Parameter Ranges

| Parameter | Valid Range | Biological Constraint | Error if Outside Range |
| --------- | ----------- | -------------------- | ---------------------- |
| `state_dim` | [16, 512] | Cortical column approximations | `ValueError: state_dim must be in [16, 512]` |
| `action_dim` | [2, 16] | Effector degrees of freedom | `ValueError: action_dim must be in [2, 16]` |
| `hidden_dim` | [32, 256] | Layer 2/3 pyramidale sizes | `ValueError: hidden_dim must be in [32, 256]` |
| `learning_rate` | [0.001, 0.1] | NMDA receptor timescales | `ValueError: learning_rate must be in [0.001, 0.1]` |

### IowaGamblingTaskEnvironment Parameter Ranges

| Parameter | Valid Range | Biological Constraint | Error if Outside Range |
| --------- | ----------- | -------------------- | ---------------------- |
| `n_trials` | [50, 500] | Attention span limitations | `ValueError: n_trials must be in [50, 500]` |
| Deck reward variance | [0.1, 0.9] | Realistic risk profiles | Automatically clamped with warning |

### Validation Enforcement

```python
# Example validation in constructor
if not (0.001 <= learning_rate <= 0.1):
    raise ValidationError(
        message=f"learning_rate {learning_rate} outside biological range [0.001, 0.1]",
        data_field="learning_rate",
        context={"value": learning_rate, "valid_range": [0.001, 0.1]},
        suggestion="Use learning rate compatible with NMDA receptor kinetics"
    )
```

---

## Package Import

```python
# Import the package
import importlib.util
from pathlib import Path

# Load the falsification protocols package
spec = importlib.util.spec_from_file_location(
    "falsification_protocols",
    Path("Falsification/__init__.py")
)
falsification_protocols = importlib.util.module_from_spec(spec)
spec.loader.exec_module(falsification_protocols)

# Import specific components
from falsification_protocols import (
    HierarchicalGenerativeModel,
    IowaGamblingTaskEnvironment,
    ProtocolRunnerGUI
)
```

## Protocol 1: Active Inference Agent Testing

### Core Classes

#### `HierarchicalGenerativeModel`

Multi-level hierarchical generative model for active inference.

```python
from falsification_protocols import HierarchicalGenerativeModel

# Define model levels
levels = [
    {"name": "intero", "dim": 16},
    {"name": "extero", "dim": 32},
    {"name": "policy", "dim": 8}
]

model = HierarchicalGenerativeModel(
    levels=levels,
    learning_rate=0.01,
    model_type="extero"  # or "intero"
)

# Generate prediction
prediction = model.predict()

# Update with error
model.update(error_signal)
```

**Parameters:**

- `levels` (List[Dict]): Hierarchical levels with name and dimension
- `learning_rate` (float): Learning rate for weight updates (default: 0.01)
- `model_type` (str): Model type - "extero" or "intero" (default: "extero")

#### `APGIActiveInferenceAgent`

Complete APGI-based active inference agent.

```python
from falsification_protocols import APGIActiveInferenceAgent

config = {
    "state_dim": 48,
    "action_dim": 4,
    "hidden_dim": 64,
    "learning_rate": 0.01
}

agent = APGIActiveInferenceAgent(config)

# Process observation
action, surprise = agent.process_observation(observation)

# Get agent state
state = agent.get_state()
```

#### `SomaticMarkerNetwork`

Somatic marker network for interoceptive predictions.

```python
from falsification_protocols import SomaticMarkerNetwork

somatic = SomaticMarkerNetwork(
    input_dim=16,
    hidden_dim=32,
    output_dim=8,
    learning_rate=0.01
)

# Process interoceptive input
somatic_signal = somatic.forward(interoceptive_input)

# Update network
somatic.update(error_signal)
```

### Protocol Execution

```python
from falsification_protocols import run_falsification_protocol_1

# Run the complete protocol
results = run_falsification_protocol_1()
```

## Protocol 2: Iowa Gambling Task Environment

### Environment Classes

#### `IowaGamblingTaskEnvironment`

Iowa Gambling Task with simulated interoceptive costs.

```python
from falsification_protocols import IowaGamblingTaskEnvironment

env = IowaGamblingTaskEnvironment(n_trials=100)

# Reset environment
observation = env.reset()

# Take action
obs, reward, done, info = env.step(action)

# Environment details
print(f"Available decks: {list(env.decks.keys())}")
# Output: ['A', 'B', 'C', 'D']
```

**Deck Characteristics:**

- Deck A: High reward variance, net negative, high interoceptive cost
- Deck B: High reward variance, net negative, moderate interoceptive cost
- Deck C: Low reward variance, net positive, low interoceptive cost
- Deck D: Low reward variance, net positive, minimal interoceptive cost

#### `VolatileForagingEnvironment`

Foraging task with shifting reward statistics.

```python
from falsification_protocols import VolatileForagingEnvironment

env = VolatileForagingEnvironment(n_trials=200)

# Environment features
obs = env.reset()
action, reward, done, info = env.step(1)  # Choose location 1
```

#### `ThreatRewardTradeoffEnvironment`

Environment with threat-reward tradeoffs.

```python
from falsification_protocols import ThreatRewardTradeoffEnvironment

env = ThreatRewardTradeoffEnvironment(n_trials=150)

# Test different threat levels
for threat_level in [0.1, 0.5, 0.9]:
    obs = env.reset(threat_level=threat_level)
    # Run trials with this threat level
```

### Protocol Execution - Protocol 2

```python
from falsification_protocols import run_falsification_protocol_2

# Run the complete protocol
results = run_falsification_protocol_2()
```

## Protocol 3: Agent Comparison Experiment

### Agent Classes

#### `StandardPPAgent`

Standard predictive processing agent without ignition.

```python
from falsification_protocols import StandardPPAgent_P3

config = {
    "state_dim": 32,
    "action_dim": 4,
    "learning_rate": 0.01
}

agent = StandardPPAgent_P3(config)

# Agent interaction
action = agent.select_action(observation)
agent.update(observation, reward, next_observation)
```

#### `GWTOnlyAgent`

Global workspace theory agent without somatic markers.

```python
from falsification_protocols import GWTOnlyAgent_P3

agent = GWTOnlyAgent_P3(config)

# This agent has global workspace but no somatic markers
action = agent.select_action(observation)
```

#### `StandardActorCriticAgent`

Standard actor-critic agent for comparison.

```python
from falsification_protocols import StandardActorCriticAgent

agent = StandardActorCriticAgent(config)

# Standard RL agent
action = agent.select_action(observation)
agent.update(observation, reward, next_observation, done)
```

#### `AgentComparisonExperiment`

Run complete agent comparison experiment.

```python
from falsification_protocols import AgentComparisonExperiment

experiment = AgentComparisonExperiment(
    n_agents=100,
    n_trials=200
)

# Run comparison
results = experiment.run_comparison()

# Get falsification report
report = experiment.get_falsification_report()
```

### Protocol Execution - Protocol 3

```python
from falsification_protocols import run_falsification_protocol_3

# Run the complete protocol
results = run_falsification_protocol_3()
```

## Protocol 4: Phase Transition Analysis

### System Components

#### `SurpriseIgnitionSystem`

Surprise accumulation and ignition system.

```python
from falsification_protocols import SurpriseIgnitionSystem

ignition = SurpriseIgnitionSystem()

# Add surprise values
ignition.add_surprise(0.1)
ignition.add_surprise(0.3)
ignition.add_surprise(0.8)

# Check for ignition
if ignition.check_ignition():
    print("Ignition occurred!")

# Get ignition history
history = ignition.get_history()
```

#### `InformationTheoreticAnalysis`

Test whether APGI ignition exhibits phase transition signatures.

```python
from falsification_protocols import InformationTheoreticAnalysis

analysis = InformationTheoreticAnalysis()

# Analyze surprise time series
results = analysis.analyze_surprise_series(surprise_data)

# Check for phase transitions
phase_transition_detected = results['phase_transition_detected']
critical_point = results['critical_point']
```

### Protocol Execution - Protocol 4

```python
from falsification_protocols import run_falsification_protocol_4

# Run the complete protocol
results = run_falsification_protocol_4()
```

## Protocol 5: Evolutionary APGI Emergence

### Evolution Classes

#### `EvolvableAgent`

Agent that can evolve based on genome.

```python
from falsification_protocols import EvolvableAgent

# Create agent with genome
genome = {
    "has_threshold": True,
    "has_interoception": True,
    "has_somatic_markers": True,
    "threshold_value": 0.5,
    "learning_rate": 0.01
}

agent = EvolvableAgent(genome)

# Agent interaction
action, intero_cost = agent.act(observation)
fitness = agent.evaluate_fitness()
```

#### `EvolutionaryAPGIEmergence`

Test whether APGI-like architectures emerge under selection pressure.

```python
from falsification_protocols import EvolutionaryAPGIEmergence

evolution = EvolutionaryAPGIEmergence(
    population_size=100,
    n_generations=200
)

# Run evolution
results = evolution.run_evolution()

# Analyze results
emergence_report = evolution.analyze_emergence()
```

### Protocol Execution - Protocol 5

```python
from falsification_protocols import run_falsification_protocol_5

# Run the complete protocol
results = run_falsification_protocol_5()
```

## Protocol 6: Network Comparison Experiment

### Network Classes

#### `APGIInspiredNetwork`

Neural network with APGI architectural constraints.

```python
from falsification_protocols import APGIInspiredNetwork

config = {
    "input_dim": 32,
    "hidden_dim": 64,
    "output_dim": 4,
    "n_layers": 3,
    "ignition_threshold": 0.5
}

network = APGIInspiredNetwork(config)

# Forward pass
output, ignition_prob = network(input_data)

# Get network state
state = network.get_network_state()
```

#### `ComparisonNetworks`

Comparison architectures without APGI constraints.

```python
from falsification_protocols import ComparisonNetworks

# Get standard networks
mlp = ComparisonNetworks.create_mlp(config)
lstm = ComparisonNetworks.create_lstm(config)
attention = ComparisonNetworks.create_attention(config)

# Compare performance
apgi_performance = network.evaluate(test_data)
mlp_performance = mlp.evaluate(test_data)
```

#### `NetworkComparisonExperiment`

Compare APGI-inspired vs standard architectures.

```python
from falsification_protocols import NetworkComparisonExperiment

experiment = NetworkComparisonExperiment(config)

# Run comparison
results = experiment.run_comparison()

# Get performance metrics
metrics = experiment.get_performance_metrics()
```

### Protocol Execution - Protocol 6

```python
from falsification_protocols import run_falsification_protocol_6

# Run the complete protocol
results = run_falsification_protocol_6()
```

## GUI Components

### `ProtocolRunnerGUI`

GUI for running APGI falsification protocols with progress tracking.

```python
from falsification_protocols import ProtocolRunnerGUI
import tkinter as tk

# Create GUI
root = tk.Tk()
app = ProtocolRunnerGUI(root)

# Start GUI
root.mainloop()
```

**GUI Features:**

- Protocol selection and configuration
- Progress tracking with real-time updates
- Result visualization
- Export functionality

## Running All Protocols

```python
# Run all protocols sequentially
protocols = [
    run_falsification_protocol_1,
    run_falsification_protocol_2,
    run_falsification_protocol_3,
    run_falsification_protocol_4,
    run_falsification_protocol_5,
    run_falsification_protocol_6
]

results = {}
for i, protocol_func in enumerate(protocols, 1):
    print(f"Running Protocol {i}...")
    results[f"protocol_{i}"] = protocol_func()
    print(f"Protocol {i} completed.")

# Analyze overall falsification status
falsification_summary = analyze_falsification_results(results)
```

## Best Practices

### 1. Protocol Configuration

```python
# Use consistent configuration across protocols
base_config = {
    "n_trials": 100,
    "learning_rate": 0.01,
    "random_seed": 42
}

# Pass to all protocols
results = run_protocol_with_config(base_config)
```

### 2. Result Analysis

```python
# Standard result format
result = {
    "status": "passed" or "failed",
    "metrics": {...},
    "falsification_criteria": {...},
    "detailed_results": {...}
}

# Check falsification status
if result["status"] == "failed":
    print(f"Protocol falsified APGI prediction: {result['falsification_criteria']}")
```

### 3. Error Handling

```python
from utils.error_handler import ProtocolError, safe_execute

def run_protocol_safely(protocol_func):
    return safe_execute(
        func=protocol_func,
        error_message="Protocol execution failed",
        error_type=ProtocolError,
        default_return={"status": "error", "message": "Execution failed"}
    )
```

## Integration with Validation System

```python
from validation.APGI_Master_Validation import APGIMasterValidator

validator = APGIMasterValidator()

# Run falsification protocols through validator
for protocol_name in ["protocol_1", "protocol_2", ...]:
    result = validator.run_falsification_protocol(protocol_name)
    validator.store_result(protocol_name, result)

# Get overall falsification status
falsification_report = validator.get_falsification_report()
```
