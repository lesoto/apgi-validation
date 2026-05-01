# APGI Validation GUI - Comprehensive Test Checklist

> **Note**: This checklist is based on the actual implementation in `Validation_GUI.py`.
> Last updated: April 2026

## GUI Launch Verification

- [ ] GUI window opens without errors (run `python Validation_GUI.py`)
- [ ] Window title displays "APGI Validation Protocol Runner" (line 112)
- [ ] Window size is 800x600 (line 113)
- [ ] Window can be resized (minimum 800x600 enforced via minsize, line 114)
- [ ] Log file created at `logs/validation_YYYYMMDD.log`
- [ ] Status label shows "Ready to run validation" (line 522)
- [ ] Progress bar is at 0%

## Keyboard Shortcuts

- [ ] Ctrl+Q quits the application (line 117)
- [ ] Cmd+Q quits the application (macOS, line 118)
- [ ] Ctrl+R runs selected protocols (line 121-126)
- [ ] Ctrl+S stops running validation (line 127)
- [ ] Ctrl+E saves results (line 128)
- [ ] Ctrl+L clears output (line 129)

## Tab Navigation (5 Tabs)

- [ ] "Validation" tab exists and is accessible (line 436)
- [ ] "Parameter Exploration" tab exists and is accessible (line 552)
- [ ] "Settings" tab exists and is accessible (line 557)
- [ ] "Data Export" tab exists and is accessible (line 563)
- [ ] "Alerts" tab exists and is accessible (line 569)
- [ ] Tabs switch correctly when clicked
- [ ] Tab content displays properly for each tab

## Validation Tab Testing

### Protocol Selection (17 Protocols via Checkboxes)

- [ ] All 17 protocol checkboxes display (lines 451-469):
  - Protocol 1: Primary Test (Synthetic EEG ML)
  - Protocol 2: Secondary Test (Behavioral Bayesian)
  - Protocol 3: Primary Test (Active Inference Agent)
  - Protocol 4: Secondary Test (Phase Transition)
  - Protocol 5: Tertiary Test (Evolutionary Emergence)
  - Protocol 6: Tertiary Test (Liquid Network)
  - Protocol 7: Tertiary Test (TMS Causal)
  - Protocol 8: Secondary Test (Psychophysical Threshold)
  - Protocol 9: Primary Test (Neural Signatures)
  - Protocol 10: Priority 2 (Causal Manipulations)
  - Protocol 11: Priority 3 (MCMC Cultural Neuroscience)
  - Protocol 12: Clinical/Cross-Species Convergence
  - Protocol 13: P5-P12 Epistemic Architecture
  - Protocol 14: Priority 1 (fMRI Anticipation Experience)
  - Protocol 15: Priority 1 (fMRI Anticipation vmPFC)
  - Protocol 16: Metabolic ATP Ground-Truth (iATPSnFR2)
  - Protocol ALL: Master Aggregator (All Protocols)

### Protocol Selection Controls

- [ ] All checkboxes are checked by default (line 472)
- [ ] "Select All" button selects all protocols (line 487-489)
- [ ] "Deselect All" button deselects all protocols (line 490-492)
- [ ] Individual checkboxes can be toggled

### Control Buttons

- [ ] "Run Validation" button exists and is enabled when protocols selected (line 498-501)
- [ ] "Stop" button exists but is disabled initially (line 503-506)
- [ ] "Save Results" button exists (line 508-511)
- [ ] Run button disables and Stop button enables during validation
- [ ] Stop button stops the running validation (line 2676-2710)

### Progress and Status

- [ ] Progress bar updates during validation (0-100%)
- [ ] Status label updates with current protocol status
- [ ] Results text area displays protocol output
- [ ] Summary label shows validation summary after completion

## Parameter Exploration Tab Testing

### Parameter Controls (4 Parameters)

- [ ] Parameter "Surprise Time Constant (τ_S)" displays:
  - Type: float
  - Min: 0.1, Max: 2.0
  - Default: 0.5
  - Step: 0.1
- [ ] Parameter "Threshold Time Constant (τ_θ)" displays:
  - Type: float
  - Min: 5.0, Max: 60.0
  - Default: 30.0
  - Step: 5.0
- [ ] Parameter "Baseline Threshold (θ₀)" displays:
  - Type: float
  - Min: 0.1, Max: 1.0
  - Default: 0.5
  - Step: 0.05
- [ ] Parameter "Sigmoid Slope (α)" displays:
  - Type: float
  - Min: 2.0, Max: 20.0
  - Default: 5.0
  - Step: 0.5

### Parameter Interaction

- [ ] Slider movement updates value label in real-time (line 637-641, 688-693)
- [ ] Values are clamped to min/max bounds (line 1952-1956)
- [ ] "Run Simulation" button runs parameter simulation (line 661-664)
- [ ] "Reset to Defaults" button resets all parameters (line 666-669)
- [ ] Simulation results display in text area

## Settings Tab Testing

### Configuration Settings

- [ ] "Update Interval (seconds)" spinbox exists:
  - Type: int
  - Min: 1, Max: 3600
  - Default: 10
- [ ] "Data Retention (days)" spinbox exists:
  - Type: int
  - Min: 1, Max: 365
  - Default: 30
- [ ] "Monitoring Threshold (error rate)" spinbox exists:
  - Type: float
  - Min: 0.0, Max: 1.0
  - Default: 0.05
- [ ] "Save Settings" button saves to `config/gui_config.yaml` (line 778-803)
- [ ] Settings load from config file on startup (line 805-833)

## Data Export Tab Testing

### Export Options

- [ ] "Export Type" radio buttons exist:
  - Current Results
  - Historical Data
  - Comprehensive Report
- [ ] Date Range fields exist (From/To) with default last 30 days

### Export Actions

- [ ] "Export to JSON" button works (line 950)
- [ ] "Export to CSV" button works (line 953)
- [ ] "Generate PDF Report" button works (line 955-959)
- [ ] "Email Report" button shows placeholder message (line 961-962)

### Historical Analysis

- [ ] "Analyze Trends" button works (line 972-973)
- [ ] "Generate Analytics" button works (line 975-978)
- [ ] "Launch Historical Dashboard" button works (line 980-983)
- [ ] "View Summary" button works (line 985-986)

### Data Collection

- [ ] Status label shows "Stopped" or "Running"
- [ ] "Start Collection" button starts data collection (line 1003-1006)
- [ ] "Stop Collection" button stops data collection (line 1008-1014)
- [ ] Export results text area displays output

## Alerts Tab Testing

### Alert Configuration

- [ ] "Alert Threshold" spinbox exists:
  - Type: float
  - Min: 0.0, Max: 1.0
  - Default: 0.8
- [ ] "Enable Alerts" checkbox exists and is checked by default
- [ ] "Save Alert Settings" button saves to `config/gui_alert_config.yaml` (line 1906-1944)
- [ ] Alert settings load from config file on startup (line 835-889)

## Validation Execution Testing

### Single Protocol Execution

- [ ] Select Protocol 1 only, click Run Validation
- [ ] Progress bar updates from 0% to 100%
- [ ] Results text shows protocol output
- [ ] Status label updates during execution
- [ ] Summary shows protocol 1 results

### Multiple Protocol Execution

- [ ] Select Protocols 1-3, click Run Validation
- [ ] Progress bar updates incrementally
- [ ] Each protocol output appears in sequence
- [ ] Final summary shows aggregate results
- [ ] Stop button interrupts execution

### Error Handling

- [ ] Invalid protocol files show appropriate error messages
- [ ] Import errors show troubleshooting hints
- [ ] Timeout errors are handled gracefully
- [ ] Critical errors show detailed troubleshooting steps

## Save Results Testing

- [ ] Save Results button prompts for file location
- [ ] JSON export creates valid JSON file
- [ ] CSV export creates valid CSV file
- [ ] PDF report generates with reportlab (if installed)
- [ ] File path validation prevents directory traversal

## Window Management

- [ ] Window close button (X) prompts if validation running
- [ ] "Stop and quit?" dialog appears when closing during validation
- [ ] Graceful cleanup on exit (cache cleared, threads stopped)

## Headless Mode

- [ ] Running with `--headless` flag runs without GUI (line 3168)
- [ ] Running with `-h` flag runs without GUI
- [ ] All protocols execute in sequence
- [ ] Results logged to console
- [ ] Master report generated with overall decision

## Summary

Total items to verify: 100+

Test each item systematically and mark as complete when verified.
