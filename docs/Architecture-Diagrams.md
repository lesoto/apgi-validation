# APGI Validation Architecture

## System Architecture Overview

```mermaid
graph TB
    subgraph "Core APGI Framework"
        APGI[APGI Formal Model]
        APGI_DYN[APGI Dynamics Engine]
        APGI_PARAMS[Parameter Estimation]
        APGI_STATES[Psychological States]
    end

    subgraph "Validation Protocols"
        VAL1[Protocol 1: Synthetic Data]
        VAL2[Protocol 2: Bayesian Comparison]
        VAL3[Protocol 3: Agent Simulations]
        VAL4[Protocol 4: Cross-Modal]
        VAL5[Protocol 5: Falsification]
        VAL6[Protocol 6: Real-Time]
        VAL7[Protocol 7: Clinical Translation]
        VAL8[Protocol 8: Psychometric]
    end

    subgraph "Falsification Protocols"
        FALS1[Protocol 1: APGI Agent]
        FALS2[Protocol 2: Iowa Gambling]
        FALS3[Protocol 3: Agent Comparison]
        FALS4[Protocol 4: Phase Transition]
        FALS5[Protocol 5: Evolutionary]
        FALS6[Protocol 6: Deep Learning]
    end

    subgraph "Data Processing"
        PREP[Preprocessing Pipelines]
        EEG[EEG Processor]
        PUPIL[Pupil Processor]
        EDA[EDA Processor]
        HR[Heart Rate Processor]
        MULTI[Multimodal Integration]
    end

    subgraph "User Interfaces"
        GUI[Validation GUI]
        CLI[Command Line Interface]
        FALS_GUI[Falsification GUI]
        PSYCH_GUI[Psychological States GUI]
    end

    subgraph "Output & Visualization"
        REPORTS[Validation Reports]
        VIZ[3D Networks]
        DASH[Interactive Dashboard]
        SCREENSHOTS[Screenshots]
    end

    APGI --> VAL1
    APGI_DYN --> VAL2
    APGI_PARAMS --> VAL3
    APGI_STATES --> VAL4

    PREP --> VAL1
    PREP --> VAL2
    PREP --> VAL3

    GUI --> VAL1
    GUI --> VAL2
    CLI --> VAL3

    FALS1 --> APGI
    FALS2 --> APGI_DYN
    FALS3 --> APGI_PARAMS

    FALS_GUI --> FALS1
    FALS_GUI --> FALS2

    VAL1 --> REPORTS
    VAL2 --> VIZ
    VAL3 --> DASH

    EEG --> PREP
    PUPIL --> PREP
    EDA --> PREP
    HR --> PREP
    PREP --> MULTI
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph "Input Data"
        RAW[Raw Multimodal Data]
        CONFIG[Configuration Files]
        PARAMS[Parameters]
    end

    subgraph "Processing Pipeline"
        CLEAN[Data Cleaning]
        FILTER[Filtering & Artifact Removal]
        FEATURE[Feature Extraction]
        INTEGRATION[Multimodal Integration]
    end

    subgraph "APGI Core"
        SURPRISE[Surprise Accumulation]
        IGNITION[Ignition Detection]
        DYNAMICS[Temporal Dynamics]
        STATES[State Estimation]
    end

    subgraph "Validation"
        HYPOTHESIS[Hypothesis Testing]
        FALSIFICATION[Falsification Tests]
        COMPARISON[Model Comparison]
        CLINICAL[Clinical Validation]
    end

    subgraph "Outputs"
        RESULTS[Validation Results]
        REPORTS[Final Reports]
        VIZUAL[Visualizations]
        METRICS[Performance Metrics]
    end

    RAW --> CLEAN
    CONFIG --> CLEAN
    PARAMS --> CLEAN

    CLEAN --> FILTER
    FILTER --> FEATURE
    FEATURE --> INTEGRATION

    INTEGRATION --> SURPRISE
    SURPRISE --> IGNITION
    IGNITION --> DYNAMICS
    DYNAMICS --> STATES

    STATES --> HYPOTHESIS
    STATES --> FALSIFICATION
    STATES --> COMPARISON
    STATES --> CLINICAL

    HYPOTHESIS --> RESULTS
    FALSIFICATION --> REPORTS
    COMPARISON --> VIZUAL
    CLINICAL --> METRICS
```

## Component Interactions

```mermaid
graph TB
    subgraph "Main Application"
        MAIN[main.py]
        CONFIG_MGR[config_manager.py]
        LOGGER[logging_config.py]
    end

    subgraph "Core Modules"
        FORMAL[APGI-Formal-Model.py]
        LIQUID[APGI-Liquid-Network.py]
        MULTIMODAL[APGI-Multimodal-Integration.py]
        TURING[APGI-Turing-Machine.py]
    end

    subgraph "Validation Layer"
        MASTER_VAL[APGI-Master-Validation.py]
        VAL_GUI[APGI-Validation-GUI.py]
        PARAM_VAL[parameter_validator.py]
    end

    subgraph "Data Layer"
        DATA_VAL[data/data_validation.py]
        PREPROC[data/preprocessing_pipelines.py]
        CACHE[data/cache_manager.py]
    end

    subgraph "Testing Framework"
        TESTS[test_framework.py]
        BATCH[batch_processor.py]
        SCREEN[take_screenshots.py]
    end

    MAIN --> CONFIG_MGR
    MAIN --> LOGGER
    MAIN --> FORMAL

    FORMAL --> LIQUID
    FORMAL --> MULTIMODAL
    FORMAL --> TURING

    MASTER_VAL --> VAL_GUI
    MASTER_VAL --> PARAM_VAL

    DATA_VAL --> PREPROC
    PREPROC --> CACHE

    TESTS --> BATCH
    BATCH --> SCREEN
```

## Technology Stack

```mermaid
graph LR
    subgraph "Languages"
        PYTHON[Python 3.8+]
        MD[Markdown]
        YAML[YAML]
    end

    subgraph "Core Libraries"
        NUMPY[NumPy]
        SCIPY[SciPy]
        PANDAS[Pandas]
        SKLEARN[Scikit-learn]
    end

    subgraph "Deep Learning"
        TORCH[PyTorch]
        TF[TensorFlow]
        KERAS[Keras]
    end

    subgraph "Visualization"
        MATPLOTLIB[Matplotlib]
        PLOTLY[Plotly]
        SEABORN[Seaborn]
    end

    subgraph "GUI Frameworks"
        TKINTER[Tkinter]
        RICH[Rich CLI]
    end

    subgraph "Testing & Quality"
        PYTEST[Pytest]
        BLACK[Black Formatter]
        FLAKE8[Flake8]
    end

    PYTHON --> NUMPY
    PYTHON --> SCIPY
    PYTHON --> PANDAS

    NUMPY --> TORCH
    SCIPY --> SKLEARN

    TORCH --> MATPLOTLIB
    SKLEARN --> PLOTLY

    MATPLOTLIB --> TKINTER
    PLOTLY --> RICH

    TKINTER --> PYTEST
    RICH --> BLACK
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development Environment"
        DEV[Local Development]
        VENV[Virtual Environment]
        DEPS[Dependencies]
    end

    subgraph "Configuration"
        YAML_CONFIG[YAML Config]
        ENV_VARS[Environment Variables]
        PARAM_FILES[Parameter Files]
    end

    subgraph "Execution Modes"
        CLI_MODE[CLI Mode]
        GUI_MODE[GUI Mode]
        BATCH_MODE[Batch Mode]
        TEST_MODE[Test Mode]
    end

    subgraph "Output Management"
        REPORTS_DIR[Reports/]
        CACHE_DIR[cache/]
        VIZ_DIR[apgi_visualizations/]
        SCREEN_DIR[docs/screenshots/]
    end

    subgraph "Monitoring"
        LOGS[Logging]
        PROGRESS[Progress Tracking]
        ERRORS[Error Handling]
    end

    DEV --> VENV
    VENV --> DEPS

    YAML_CONFIG --> CLI_MODE
    ENV_VARS --> GUI_MODE
    PARAM_FILES --> BATCH_MODE

    CLI_MODE --> REPORTS_DIR
    GUI_MODE --> VIZ_DIR
    BATCH_MODE --> CACHE_DIR
    TEST_MODE --> SCREEN_DIR

    LOGS --> PROGRESS
    PROGRESS --> ERRORS
```

## Key Design Patterns

### 1. **Strategy Pattern**: Different validation protocols
### 2. **Factory Pattern**: Agent and environment creation
### 3. **Observer Pattern**: GUI progress updates
### 4. **Command Pattern**: CLI command structure
### 5. **Template Method**: Common validation workflow
### 6. **Decorator Pattern**: Logging and caching
### 7. **State Pattern**: Psychological state transitions

## Security & Error Handling

```mermaid
graph LR
    subgraph "Input Validation"
        SCHEMA[Schema Validation]
        TYPE_CHECK[Type Checking]
        RANGE_CHECK[Range Validation]
    end

    subgraph "Error Handling"
        SPECIFIC[Specific Exceptions]
        LOGGING[Error Logging]
        RECOVERY[Graceful Recovery]
    end

    subgraph "Data Security"
        SANITIZATION[Input Sanitization]
        PERMISSIONS[File Permissions]
        BACKUP[Data Backup]
    end

    SCHEMA --> SPECIFIC
    TYPE_CHECK --> LOGGING
    RANGE_CHECK --> RECOVERY

    SPECIFIC --> SANITIZATION
    LOGGING --> PERMISSIONS
    RECOVERY --> BACKUP
```

This architecture documentation provides a comprehensive overview of the APGI validation system's structure, data flow, component interactions, and technology stack.
