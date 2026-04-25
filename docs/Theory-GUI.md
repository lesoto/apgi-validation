# APGI Theory GUI - Quick Start Guide

## Overview

The APGI Theory GUI is a scientific instrument interface for discovering, configuring, and executing theory scripts. It provides real-time monitoring, parameter management, and result tracking.

## Starting the GUI

```bash
python Theory_GUI.py
```

The GUI will open in a new window with the following layout:

```text
┌─────────────────────────────────────────────────────────────┐
│ SYSTEM STATUS: [OK] Ready | ACTIVE SCRIPTS: 15 | PLATFORM   │
├──────────────┬──────────────────────────────────────────────┤
│ SCRIPT       │ PROTOCOLS TAB / PARAMETERS TAB               │
│ LIBRARY      │                                              │
│              │ [Selected Protocol Display]                  │
│ • Script 1   │ [Run Selected] [Run All Scripts]             │
│ • Script 2   │                                              │
│ • Script 3   │ [Quick Statistics Cards]                     │
│ ...          │                                              │
├──────────────┴──────────────────────────────────────────────┤
│ INSTRUMENT CONSOLE - Real-time Data Stream                  │
│ [Stop] [Clear Console]                                      │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ Console output appears here...                         │  │
│ │ [Scrollable text area]                                 │  │
│ └────────────────────────────────────────────────────────┘  │
│ Progress: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 40%       │
└──────────────────────────────────────────────────────────────┘
```

## Main Tabs

### 1. Protocols Tab

- **Selected Protocol Display:** Shows the currently selected script
- **Run Selected:** Execute the selected protocol with configured parameters
- **Run All Scripts:** Execute all 15 theory scripts sequentially
- **Quick Statistics:** View total scripts, configurable scripts, and status

### 2. Parameters Tab

- **Script Selection:** Choose which script to configure
- **Parameter Configuration:** Adjust numeric and string parameters
- **Load Defaults:** Reset all parameters to defaults
- **Save Parameters:** Save configuration to JSON file

## Workflow

### Step 1: Select a Protocol

Click on any script name in the left sidebar. The script will be highlighted and its description will appear in the main area.

### Step 2: Configure Parameters (Optional)

1. Go to the "Parameters" tab
2. Select the same script from the dropdown
3. Adjust parameters using the spinboxes or text fields
4. Click "Save Parameters" to persist your configuration

### Step 3: Run the Protocol

- **Single Script:** Click "Run Selected" to execute the selected protocol
- **All Scripts:** Click "Run All Scripts" to run all 15 protocols sequentially

### Step 4: Monitor Execution

- Watch the console output for real-time feedback
- Check the progress bar for completion percentage
- Click "Stop" to interrupt execution at any time

### Step 5: Review Results

- Results are automatically saved to `validation_results/` directory
- Each result file includes metadata, predictions, and BIC values
- Console output is preserved for reference

## Available Scripts

| # | Script Name | Purpose |
| --- | --- | --- |
| 1 | Bayesian Estimation Framework | Parameter estimation using Bayesian methods |
| 2 | Computational Benchmarking | Performance benchmarking of algorithms |
| 3 | Cross Species Scaling | Cross-species model validation |
| 4 | Cultural Neuroscience | Cultural modulation effects analysis |
| 5 | Entropy Implementation | Information-theoretic entropy calculations |
| 6 | Equations | Core APGI mathematical equations |
| 7 | Falsification Framework | Falsification protocol implementation |
| 8 | Full Dynamic Model | Complete dynamic system model |
| 9 | Liquid Network Implementation | Liquid state machine networks |
| 10 | Multimodal Classifier | Multi-modal classification system |
| 11 | Multimodal Integration | Multi-modal data integration |
| 12 | Open Science Framework | Open science compliance tools |
| 13 | Parameter Estimation | Advanced parameter estimation |
| 14 | Psychological States | Psychological state modeling |
| 15 | Turing Machine | Turing machine implementation |

## Common Tasks

### Run a Single Script

1. Click the script name in the sidebar
2. Click "Run Selected"
3. Monitor the console output

### Run All Scripts

1. Click "Run All Scripts"
2. Confirm the dialog
3. Wait for all scripts to complete (may take significant time)

### Configure Parameters

1. Go to "Parameters" tab
2. Select a script from the dropdown
3. Adjust parameters
4. Click "Save Parameters"

### Clear Console

1. Click "Clear Console" button in the console toolbar
2. Console output will be cleared

### Stop Execution

1. Click "Stop" button during execution
2. The running script will be interrupted
3. Status will update to "Stopping"

## Parameter Types

### Integer Parameters

- **Format:** Whole numbers only
- **Range:** Specified min/max values
- **Examples:** `n_samples`, `n_trials`, `population_size`

### Float Parameters

- **Format:** Decimal numbers
- **Range:** Specified min/max values
- **Examples:** `learning_rate`, `alpha`, `beta`, `rho`

### String Parameters

- **Format:** Text or JSON arrays
- **Examples:** `surprise_range` (format: `[min, max]`)

## Output Files

### Console Output

- Displayed in real-time in the GUI console
- Limited to 1000 lines to prevent memory issues
- Can be cleared using "Clear Console" button

### Result Files

- Saved to `validation_results/` directory
- Format: JSON with metadata, predictions, and BIC values
- Filename: `validation_results_<unique_id>.json`

### Parameter Files

- Saved to `config/` directory
- Format: JSON with parameter values
- Filename: `<script_name>_params.json`

## Keyboard Shortcuts

| Shortcut | Action |
| -------- | ------ |
| Tab | Navigate between UI elements |
| Enter | Activate focused button |
| Escape | Close tooltips |

## Tips & Tricks

1. **Batch Processing:** Use "Run All Scripts" to execute all protocols overnight
2. **Parameter Tuning:** Save different parameter configurations for comparison
3. **Result Analysis:** Check `validation_results/` directory for detailed results
4. **Console Monitoring:** Watch the console for warnings and errors
5. **Progress Tracking:** Use the progress bar to estimate completion time

## Troubleshooting

### GUI Won't Start

```bash
# Check Python version
python --version

# Verify tkinter is installed
python -c "import tkinter; print('OK')"

# Run with verbose output
python -u New_Theory_GUI.py
```

### Scripts Fail to Run

1. Check the console output for error messages
2. Verify all dependencies are installed
3. Check that Theory scripts are in the `Theory/` directory
4. Review parameter values for validity

### Memory Issues

1. Click "Clear Console" to free memory
2. Avoid running all scripts simultaneously
3. Monitor system resources during execution

### Parameters Not Applying

1. Some scripts may not accept custom parameters
2. The GUI will use defaults in these cases
3. Check the console for parameter-related warnings

## Advanced Usage

### Running from Command Line

```bash
# Run with custom matplotlib backend
MPLBACKEND=Agg python New_Theory_GUI.py

# Run with debug logging
python -u New_Theory_GUI.py 2>&1 | tee gui.log
```

### Accessing Results Programmatically

```python
import json
from pathlib import Path

results_dir = Path('validation_results')
for result_file in results_dir.glob('*.json'):
    with open(result_file) as f:
        data = json.load(f)
        print(f"Protocol: {data['metadata']['protocol_file']}")
        print(f"Timestamp: {data['metadata']['timestamp']}")
```
