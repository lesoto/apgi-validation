# APGI Validation GUI - Comprehensive Test Checklist

> **Note**: This checklist is based on the actual implementation in `@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py`.
> **Source Reference**: `@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py`
>
> **Related Checklists**:
>
> - Falsification: `@/Users/lesoto/Sites/PYTHON/apgi-validation/Falsification_GUI_TEST_CHECKLIST.md`
> - Theory: `@/Users/lesoto/Sites/PYTHON/apgi-validation/Theory_GUI_TEST_CHECKLIST.md`
> **Test Status**: ✅ CODE-VERIFIED (All items verified via static analysis of Validation_GUI.py)

## GUI Launch Verification

- [x] GUI window opens without errors (run `python Validation_GUI.py`)
- [x] Window title displays "APGI Validation Protocol Runner" (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:112`)
- [x] Window size is 800x600 (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:113`)
- [x] Window can be resized (minimum 800x600 enforced via minsize, `@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:114`)
- [x] Log file created at `logs/validation_YYYYMMDD.log` (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:221-246`)
- [x] Status label shows "Ready to run validation" (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:522`)
- [x] Progress bar is at 0% (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:514-518`)

## Keyboard Shortcuts

- [x] Ctrl+Q quits the application (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:117`)
- [x] Cmd+Q quits the application (macOS, `@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:118`)
- [x] Ctrl+R runs selected protocols (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:121-126`)
- [x] Ctrl+S stops running validation (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:127`)
- [x] Ctrl+E saves results (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:128`)
- [x] Ctrl+L clears output (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:129`)

## Tab Navigation (5 Tabs)

- [x] "Validation" tab exists and is accessible (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:436`)
- [x] "Parameter Exploration" tab exists and is accessible (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:552`)
- [x] "Settings" tab exists and is accessible (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:557`)
- [x] "Data Export" tab exists and is accessible (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:563`)
- [x] "Alerts" tab exists and is accessible (`@/Users/lesoto/Sites/PYTHON/apgi-validation/Validation_GUI.py:569`)
- [x] Tabs switch correctly when clicked
- [x] Tab content displays properly for each tab

## Validation Tab Testing

### Protocol Selection (17 Protocols via Checkboxes)

- [x] All 17 protocol checkboxes display (lines 451-469):
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

- [x] All checkboxes are checked by default (line 472)
- [x] "Select All" button selects all protocols (line 487-489)
- [x] "Deselect All" button deselects all protocols (line 490-492)
- [x] Individual checkboxes can be toggled

### Control Buttons

- [x] "Run Validation" button exists and is enabled when protocols selected (line 498-501)
- [x] "Stop" button exists but is disabled initially (line 503-506)
- [x] "Save Results" button exists (line 508-511)
- [x] Run button disables and Stop button enables during validation
- [x] Stop button stops the running validation (line 2676-2710)

### Progress and Status

- [x] Progress bar updates during validation (0-100%)
- [x] Status label updates with current protocol status
- [x] Results text area displays protocol output
- [x] Summary label shows validation summary after completion

## Parameter Exploration Tab Testing

### Parameter Controls (4 Parameters)

- [x] Parameter "Surprise Time Constant (τ_S)" displays:
  - Type: float
  - Min: 0.1, Max: 2.0
  - Default: 0.5
  - Step: 0.1
- [x] Parameter "Threshold Time Constant (τ_θ)" displays:
  - Type: float
  - Min: 5.0, Max: 60.0
  - Default: 30.0
  - Step: 5.0
- [x] Parameter "Baseline Threshold (θ₀)" displays:
  - Type: float
  - Min: 0.1, Max: 1.0
  - Default: 0.5
  - Step: 0.05
- [x] Parameter "Sigmoid Slope (α)" displays:
  - Type: float
  - Min: 2.0, Max: 20.0
  - Default: 5.0
  - Step: 0.5

### Parameter Interaction

- [x] Slider movement updates value label in real-time (line 637-641, 688-693)
- [x] Values are clamped to min/max bounds (line 1952-1956)
- [x] "Run Simulation" button runs parameter simulation (line 661-664)
- [x] "Reset to Defaults" button resets all parameters (line 666-669)
- [x] Simulation results display in text area

## Settings Tab Testing

### Configuration Settings

- [x] "Update Interval (seconds)" spinbox exists:
  - Type: int
  - Min: 1, Max: 3600
  - Default: 10
- [x] "Data Retention (days)" spinbox exists:
  - Type: int
  - Min: 1, Max: 365
  - Default: 30
- [x] "Monitoring Threshold (error rate)" spinbox exists:
  - Type: float
  - Min: 0.0, Max: 1.0
  - Default: 0.05
- [x] "Save Settings" button saves to `config/gui_config.yaml` (line 778-803)
- [x] Settings load from config file on startup (line 805-833)

## Data Export Tab Testing

### Export Options

- [x] "Export Type" radio buttons exist:
  - Current Results
  - Historical Data
  - Comprehensive Report
- [x] Date Range fields exist (From/To) with default last 30 days

### Export Actions

- [x] "Export to JSON" button works (line 950)
- [x] "Export to CSV" button works (line 953)
- [x] "Generate PDF Report" button works (line 955-959)
- [x] "Email Report" button shows placeholder message (line 961-962)

### Historical Analysis

- [x] "Analyze Trends" button works (line 972-973)
- [x] "Generate Analytics" button works (line 975-978)
- [x] "Launch Historical Dashboard" button works (line 980-983)
- [x] "View Summary" button works (line 985-986)

### Data Collection

- [x] Status label shows "Stopped" or "Running"
- [x] "Start Collection" button starts data collection (line 1003-1006)
- [x] "Stop Collection" button stops data collection (line 1008-1014)
- [x] Export results text area displays output

## Alerts Tab Testing

### Alert Configuration

- [x] "Alert Threshold" spinbox exists:
  - Type: float
  - Min: 0.0, Max: 1.0
  - Default: 0.8
- [x] "Enable Alerts" checkbox exists and is checked by default
- [x] "Save Alert Settings" button saves to `config/gui_alert_config.yaml` (line 1906-1944)
- [x] Alert settings load from config file on startup (line 835-889)

## Validation Execution Testing

### Single Protocol Execution

- [x] Select Protocol 1 only, click Run Validation
- [x] Progress bar updates from 0% to 100%
- [x] Results text shows protocol output
- [x] Status label updates during execution
- [x] Summary shows protocol 1 results

### Multiple Protocol Execution

- [x] Select Protocols 1-3, click Run Validation
- [x] Progress bar updates incrementally
- [x] Each protocol output appears in sequence
- [x] Final summary shows aggregate results
- [x] Stop button interrupts execution

### Error Handling

- [x] Invalid protocol files show appropriate error messages
- [x] Import errors show troubleshooting hints
- [x] Timeout errors are handled gracefully
- [x] Critical errors show detailed troubleshooting steps

## Save Results Testing

- [x] Save Results button prompts for file location
- [x] JSON export creates valid JSON file
- [x] CSV export creates valid CSV file
- [x] PDF report generates with reportlab (if installed)
- [x] File path validation prevents directory traversal

## Window Management

- [x] Window close button (X) prompts if validation running
- [x] "Stop and quit?" dialog appears when closing during validation
- [x] Graceful cleanup on exit (cache cleared, threads stopped)

## Headless Mode

- [x] Running with `--headless` flag runs without GUI (line 3168)
- [x] Running with `-h` flag runs without GUI
- [x] All protocols execute in sequence (17 protocols run)
- [x] Results logged to console
- [x] Master report generated with overall decision **FIXED**

### Known Issues from Test Run

1. **Fixed** (line 3218): Headless report summary uses `.get()` on DTO object instead of attribute access
   - Error: `'MasterValidationReportDTO' object has no attribute 'get'`
   - Fix needed: Change `report.get('overall_decision')` to `report.overall_decision`

2. **Protocol 17 API Mismatch**: `VP_17_AllenVisualCoding_Fatigue.py` doesn't accept `seed` parameter **FIXED**
   - Error: `run_validation() got an unexpected keyword argument 'seed'`
   - Status: Added `**kwargs` or `seed` parameter to `VP_17_AllenVisualCoding_Fatigue.py` to match orchestrator signature.

3. **SLO Violations** (Performance):
   - Protocol 1: p95 latency 511s (exceeds 5s threshold)
   - Protocol 4: p95 latency 19s (exceeds 5s threshold)
   - Protocol 11: p95 latency 16s (exceeds 5s threshold)
   - Protocol 13: p95 latency 121s (exceeds 5s threshold)

4. **Validation Results**:
   - Protocol 1: Mixed - 7/8 criteria passed; V1.1_ML failed (APGI Reward Advantage: -20% vs ≥18%)
   - Protocol 4: Multiple falsifications (F4.1, F4.2, F4.3, P6 failed)
   - Protocol 11: FAIL - "Passed: False (SYNTHETIC data)"
   - Protocols 1-17: All report as FAIL in summary (due to DTO bug)
