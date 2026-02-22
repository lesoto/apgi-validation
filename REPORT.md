# APGI Validation Framework — Comprehensive Application Audit Report

**Audit Date:** 2026-02-22
**Auditor:** Automated Code Audit (Claude Sonnet 4.6)
**Framework Version:** 1.3.0
**Branch:** `claude/app-audit-testing-GPhpI`
**Python Version:** 3.11.14
**Pytest Version:** 9.0.2

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [KPI Scores](#kpi-scores)
3. [Audit Scope](#audit-scope)
4. [Bug Inventory](#bug-inventory)
5. [Missing Features & Incomplete Implementations](#missing-features--incomplete-implementations)
6. [Detailed Findings by Component](#detailed-findings-by-component)
7. [Actionable Recommendations](#actionable-recommendations)

---

## Executive Summary

The APGI (Adaptive Pattern Generation and Integration) Theory Framework is a scientific Python application providing a unified CLI, multiple tkinter-based GUIs, 12 validation protocols, 6 falsification protocols, and a comprehensive utility layer. The framework targets neuroscience researchers and data scientists running formal model simulations, parameter estimation, multimodal integration, and statistical falsification workflows.

This audit conducted a full end-to-end code review of 63 Python files, all configuration YAML files, documentation markdown files, test suites (14 collected tests across 5 test modules), and all GUI entry points.

**Key findings:**

- **Critical blocking defect:** The `jsonschema` package is absent from the runtime environment. This causes `utils/__init__.py` to fail at import, which cascades and blocks the main CLI entry point (`main.py`), both integration test files, and the `test_import_main` basic test. **The application cannot be started from the CLI**.
- **Missing dependency `matplotlib`** prevents Validation Protocol 1 (and potentially others) from running in the test harness.
- **Test pass rate is 71% (10/14 collected tests pass; 2 additional test files fail to collect)**, meaning only ~63% of all intended tests execute successfully.
- **A critical NameError bug** exists in `Falsification/APGI-Falsification-Protocol-GUI.py`: `_handle_apgi_agent()` and `_handle_iowa_gambling()` both reference `self.module`, which is never defined on the class, causing an `AttributeError` at runtime.
- **Missing file `utils/data_quality_assessment.py`** is expected by `tests/test_utils.py` but was never created.
- **`utils/theme_manager.py` does not exist**, causing a soft import warning at every `Tests-GUI.py` startup.
- **`Validation/Master_Validation.py` does not exist**, so `APGIMasterValidator` is always `None` in the Validation GUI — the "Run Validation" button always fails immediately.
- Several GUI usability issues exist: generic protocol labels, sparse Data Export tab, no CSV export button wired into the export widget, and no confirmation dialog on destructive actions.
- Documentation generation target in the Makefile prints "not yet implemented".
- Several performance-logging calls in `main.py` are commented out, representing dead code from an incomplete feature.

---

## KPI Scores

| # | KPI | Score (1–100) | Rationale |
|---|-----|--------------|-----------|
| 1 | **Functional Completeness** | **48** | CLI cannot be imported due to missing `jsonschema`. ~37% of tests fail or cannot be collected. `Master_Validation.py` is missing, blocking GUI validation entirely. Key files expected by tests are absent. Several GUI handlers contain broken `self.module` references. Core exports (CSV export) are unconnected. |
| 2 | **UI/UX Consistency** | **62** | GUI layouts are generally consistent across tkinter windows; tabbed interface in the Validation GUI is clear. However, protocol labels are non-descriptive (e.g., "Protocol 1: Primary Test"), the Data Export tab is sparse with a missing CSV button, tooltips are inconsistent across GUIs, and the Falsification GUI has no progress bar or overall status indicator. |
| 3 | **Responsiveness & Performance** | **70** | Worker threads are used correctly throughout all GUIs. Queue-based thread-safe UI update pattern is implemented in the Validation GUI. Output buffer bounding exists in Tests-GUI. However, the output limit in Utils-GUI uses a less efficient full-buffer-delete approach, the Validation GUI update queue caps at 100 items (could silently drop status updates), and performance-logging calls are commented out. |
| 4 | **Error Handling & Resilience** | **61** | Error handler utility (`utils/error_handler.py`) and structured logging (`utils/logging_config.py`) are present. Broad exception categories are caught in the CLI. GUIs handle import failures gracefully with `safe_import_module()`. However, the `self.module` bug in Falsification GUI causes an unhandled `AttributeError`; `pd.read_csv()` at `main.py:655` is uncaught; and several commented-out `log_error()` calls leave error events untracked. |
| 5 | **Overall Implementation Quality** | **55** | Architecture is sound and reasonably modular. Docstrings and type hints are present throughout. The 4,998-line `main.py` violates single-responsibility; a signal handler is defined but never registered (`main.py:480`); config values are duplicated between `config/` and `utils/config/`; multiple redundant local imports shadow global imports; documentation generation is unimplemented. |

**Composite Score: 59 / 100**

---

## Audit Scope

| Category | Files Audited |
|----------|--------------|
| CLI entry point | `main.py` (4,998 lines) |
| GUI modules | `Utils-GUI.py`, `Tests-GUI.py`, `Validation/APGI-Validation-GUI.py`, `Falsification/APGI-Falsification-Protocol-GUI.py` |
| Validation protocols | `Validation/Validation-Protocol-1.py` through `Validation-Protocol-12.py` (12 files) |
| Falsification protocols | `Falsification/Falsification-Protocol-1.py` through `Falsification-Protocol-6.py` (6 files) |
| Utility modules | All 13 files in `utils/` |
| Configuration | `config/default.yaml`, `config/gui_config.yaml`, `config/gui_alert_config.yaml`, `config/profiles/*.yaml` |
| Test suite | `tests/test_basic.py`, `tests/test_utils.py`, `tests/test_validation.py`, `tests/test_integration.py`, `tests/test_performance.py` |
| Build / tooling | `Makefile`, `pytest.ini`, `mypy.ini`, `.flake8`, `requirements.txt` |
| Documentation | 18 markdown files in `docs/` |

---

## Bug Inventory

### Critical Severity

| ID | Title | File | Line(s) | Reproduction Steps | Expected Behavior | Actual Behavior |
|----|-------|------|---------|-------------------|-------------------|-----------------|
| BUG-001 | `jsonschema` package missing — blocks entire CLI | `utils/config_manager.py` | 37 | `python3 main.py --help` | CLI launches and shows help | `ModuleNotFoundError: No module named 'jsonschema'` |
| BUG-002 | `self.module` NameError in Falsification GUI | `Falsification/APGI-Falsification-Protocol-GUI.py` | 279, 315 | Launch Falsification GUI → click "APGI Agent" or "Iowa Gambling" button | Protocol runs in background thread | `AttributeError: 'ProtocolRunnerGUI' object has no attribute 'module'` — protocol crashes silently (error only shows in GUI console) |
| BUG-003 | `utils/__init__.py` import chain blocks test collection | `utils/__init__.py` | 26 | `python3 -m pytest tests/test_integration.py tests/test_performance.py` | All tests collected and run | `ModuleNotFoundError: No module named 'jsonschema'` — 2 test files cannot be collected |
| BUG-004 | `Master_Validation.py` missing — Validation GUI cannot run | `Validation/APGI-Validation-GUI.py` | 55–64 | Launch Validation GUI → click "Run Validation" | Validation protocols run | `APGIMasterValidator` is always `None`; clicking "Run Validation" immediately shows "APGI Master Validator not available" |

### High Severity

| ID | Title | File | Line(s) | Reproduction Steps | Expected Behavior | Actual Behavior |
|----|-------|------|---------|-------------------|-------------------|-----------------|
| BUG-005 | Missing `utils/data_quality_assessment.py` | `tests/test_utils.py` | 35 | `python3 -m pytest tests/test_utils.py::test_utility_files_exist` | File exists, test passes | `AssertionError: Utility file data_quality_assessment.py missing` |
| BUG-006 | `matplotlib` package missing — blocks Validation-Protocol-1 | `Validation/Validation-Protocol-1.py` | 14 | `python3 -m pytest tests/test_validation.py::test_apgi_dynamical_system_simulate_surprise_accumulation` | Protocol module loads, test passes | `ModuleNotFoundError: No module named 'matplotlib'` |
| BUG-007 | `export_csv()` method defined but not wired to any button | `Validation/APGI-Validation-GUI.py` | 700–735, 701–707 | Launch Validation GUI → Data Export tab | "Export CSV" button is visible | No CSV export button exists; `export_csv()` is dead code |
| BUG-008 | Signal handler defined but never registered | `main.py` | 480–483 | Run `python3 main.py formal-model` → press Ctrl+C | Graceful simulation cancellation | `handle_cancel` is defined inside `formal_model()` but never passed to `signal.signal()`; Ctrl+C causes an unhandled interrupt |
| BUG-009 | `pd.read_csv()` at CLI preprocessing not in try/except | `main.py` | 655 | `python3 main.py process-data --input-file /nonexistent.csv` | Informative error message | Unhandled `FileNotFoundError` stack trace shown to user |
| BUG-010 | `_run_parameter_simulation_worker` import uses wrong module name | `Validation/APGI-Validation-GUI.py` | 950 | Click "Run Simulation" in Parameter Exploration tab | Simulation runs | `from APGI_Equations import CoreIgnitionSystem` fails — actual file is `APGI-Equations.py` (hyphen); `ModuleNotFoundError` |

### Medium Severity

| ID | Title | File | Line(s) | Reproduction Steps | Expected Behavior | Actual Behavior |
|----|-------|------|---------|-------------------|-------------------|-----------------|
| BUG-011 | `utils/theme_manager.py` missing — generates startup warning | `Tests-GUI.py` | 22–27 | Launch `Tests-GUI.py` | Silent import, Theme menu available | `Warning: Theme manager not available. Theme support disabled.` printed; Theme menu absent |
| BUG-012 | Duplicate config directories — `utils/config/` shadows `config/` | `utils/config/`, `config/` | — | Run any command using `config_manager.py` | Single authoritative config source | Two parallel config trees; unclear which takes precedence |
| BUG-013 | Validation GUI status label at row=3 can be hidden on resize | `Validation/APGI-Validation-GUI.py` | 413–417 | Launch Validation GUI → resize window smaller | Status label always visible | Results frame expands, pushing status label out of view |
| BUG-014 | `run_all` in Utils-GUI.py uses lambda capture-by-reference in loop | `Utils-GUI.py` | 382–386 | Launch Utils-GUI → "Run All Scripts" | Each script highlighted in turn | Lambda closures over loop variable `i`; all lambdas resolve to last value — last script always highlighted |
| BUG-015 | Validation protocol descriptions are generic and uninformative | `Validation/APGI-Validation-GUI.py` | 367–376 | Launch Validation GUI | Checkboxes show meaningful protocol names | Labels read "Protocol 1: Primary Test" etc. — no scientific context |
| BUG-016 | Performance logging calls commented out — dead code | `main.py` | 290, 295, 539, 597 | — | Performance metrics tracked | `log_performance()` calls are commented out; metrics are never recorded |
| BUG-017 | `config-diff` CLI command cannot compare against a specific version | `main.py` | 4649–4701 | `python3 main.py config-diff` | Compare current vs a prior version | No `--base-version` option; always diffs current config against itself |
| BUG-018 | `messagebox.showerror()` called from daemon thread in Falsification GUI | `Falsification/APGI-Falsification-Protocol-GUI.py` | 250 | Trigger a protocol error | Error dialog shown | tkinter dialog is called from non-main thread; can cause `RuntimeError: main thread is not in main loop` |

### Low Severity

| ID | Title | File | Line(s) | Reproduction Steps | Expected Behavior | Actual Behavior |
|----|-------|------|---------|-------------------|-------------------|-----------------|
| BUG-019 | Redundant local imports shadow module-level imports | `main.py` | 474, 561, 743, 1102 | — | Single import at module level | `numpy` and `pandas` are re-imported locally inside functions that already have them globally |
| BUG-020 | Makefile `docs` target prints "not yet implemented" | `Makefile` | last target | `make docs` | Generate documentation | Prints "Documentation generation not yet implemented" |
| BUG-021 | `_force_kill_process()` escalation not invoked on timeout | `Utils-GUI.py` | 534–540 | Run a script that hangs past timeout | Force-killed via SIGKILL after timeout | Only `process.terminate()` (SIGTERM) is called; no escalation to `process.kill()` (SIGKILL) |
| BUG-022 | `test_config_manager_load_save_cycle` passes silently on failure | `tests/test_validation.py` | 105–141 | `python3 -m pytest tests/test_validation.py::test_config_manager_load_save_cycle` | Test correctly validates round-trip | Exception in `set_parameter` is caught and printed; test continues and passes regardless of correctness |
| BUG-023 | No `__init__.py` in Falsification directory | `Falsification/` | — | `import Falsification` | Module importable as a package | `Falsification` directory cannot be imported as a Python package |

---

## Missing Features & Incomplete Implementations

| ID | Feature / Component | Status | Notes |
|----|---------------------|--------|-------|
| MF-001 | **Documentation generation** (`make docs`) | Not implemented | Makefile target prints placeholder; no doc generator configured |
| MF-002 | **`utils/data_quality_assessment.py`** | File missing | Expected by test suite; asserted as an essential utility file |
| MF-003 | **`utils/theme_manager.py`** | File missing | Imported by `Tests-GUI.py`; absence removes theme switching UI |
| MF-004 | **`Validation/Master_Validation.py`** | File missing | Required by `APGI-Validation-GUI.py`; without it `APGIMasterValidator = None` and validation cannot run via GUI |
| MF-005 | **Validation Protocols 9–12 not selectable in GUI** | Partial | GUI hard-codes 8 protocol checkboxes; Protocols 9–12 exist as files but are not accessible from the UI |
| MF-006 | **CSV export button in Data Export tab** | Button missing | `export_csv()` method is implemented but no button is added in `create_export_widgets()` |
| MF-007 | **PDF report generation** | Partial | `generate_report()` saves a plain-text `.txt` file; button label says "Generate PDF Report" |
| MF-008 | **Performance logging framework** | Commented out | `log_performance()` calls exist but are commented in 4 locations in `main.py`; metrics are never persisted |
| MF-009 | **Signal handling for `formal_model` cancellation** | Incomplete | Handler function is defined but `signal.signal()` is never called to register it |
| MF-010 | **`config-diff` base-version selection** | Incomplete | `config_diff` CLI command has no `--base-version` option; always diffs against self |
| MF-011 | **Integration and performance tests cannot run** | Blocked | `tests/test_integration.py` and `tests/test_performance.py` fail during collection due to missing `jsonschema` |
| MF-012 | **`Falsification/__init__.py`** | Missing | Falsification directory cannot be imported as a Python package |
| MF-013 | **`dashboard` CLI command** | Partial | Described as generating "static HTML dashboards" but produces minimal placeholder output |
| MF-014 | **Cross-browser / web GUI (Dash/Flask)** | Not testable | `dash` and `flask` are in requirements but not installed; `gui --gui-type=analysis` and `performance-dashboard` commands cannot be verified |
| MF-015 | **All 39 CLI commands exercised by tests** | Not covered | No test exercises any CLI command end-to-end; test coverage of the CLI layer is 0% |

---

## Detailed Findings by Component

### 1. CLI Entry Point (`main.py`)

**Lines:** 4,998 | **Commands:** 39

**Strengths:**
- Comprehensive command set covering simulation, validation, falsification, data I/O, caching, backup, and performance.
- Consistent Rich-formatted output with color-coded severity levels.
- Verbose/quiet mode correctly propagated.
- Broad exception type coverage in most commands.
- `APGIModuleLoader` class provides lazy module loading with graceful failure.

**Issues:**
- Cannot be imported or executed due to missing `jsonschema` (BUG-001).
- Extremely large single file (4,998 lines) violates single-responsibility principle.
- 4 commented-out `log_performance()` / `set_parameter()` calls (BUG-016).
- Unregistered signal handler in `formal_model()` (BUG-008).
- Unguarded `pd.read_csv()` in `process_data` (BUG-009).
- 16 hardcoded default values that should be sourced from config (ports, paths, counts).
- Redundant local imports of `numpy` and `pandas` in 4 functions (BUG-019).
- 0% of CLI commands are covered by automated tests.

---

### 2. Validation GUI (`Validation/APGI-Validation-GUI.py`)

**Lines:** 1,771 | **Tabs:** Validation, Parameter Exploration, Settings, Data Export, Alerts

**Strengths:**
- Thread-safe GUI update pattern via bounded queue (maxsize=100).
- Protocol module caching with lock protection prevents memory leaks.
- `is_running` property with threading lock for thread-safe access.
- Comprehensive error classification in `_handle_protocol_error()`.
- Settings and alert configurations persist to YAML files.
- `_convert_to_serializable()` correctly handles numpy types for JSON export.
- `_validate_report()` enforces strict structure before accessing report keys.

**Issues:**
- `Master_Validation.py` is missing → `APGIMasterValidator = None` → validation always fails (BUG-004 / MF-004).
- Only 8 protocols shown in GUI; Protocols 9–12 are unreachable via the UI (MF-005).
- Protocol labels are generic: "Primary Test", "Secondary Test", "Tertiary Test" (BUG-015).
- `export_csv()` is implemented but no button exists in the Data Export tab (BUG-007 / MF-006).
- PDF generation saves `.txt` not `.pdf` (MF-007).
- `_run_parameter_simulation_worker` uses a module import path that fails due to hyphenated filename (BUG-010).
- Status label can be hidden when window is resized (BUG-013).
- Update queue hard cap of 100: rapid protocol logging could silently drop status messages.

---

### 3. Falsification GUI (`Falsification/APGI-Falsification-Protocol-GUI.py`)

**Lines:** 389 | **Protocols:** 6

**Strengths:**
- Clean 2×3 grid button layout with tooltips.
- Separate stop mechanism via `threading.Event`.
- Protocol dispatch through dedicated handler methods per protocol type.

**Issues:**
- `_handle_apgi_agent()` (line 279) and `_handle_iowa_gambling()` (line 315) both reference `self.module` which is never assigned on the class — `AttributeError` at runtime (BUG-002).
- `messagebox.showerror()` called from a daemon thread (line 250) — not thread-safe (BUG-018).
- No `__init__.py` in Falsification directory (BUG-023 / MF-012).
- No progress bar in the UI despite protocols potentially running for minutes.
- `_handle_default()` (line 378–379) creates a class instance and logs its name but performs no meaningful action.
- Stop button cannot interrupt `run_full_experiment()` type calls.

---

### 4. Utils GUI (`Utils-GUI.py`)

**Lines:** 693

**Strengths:**
- Timeout mechanism per-script via `select`-based polling (Unix) with Windows fallback.
- Retry logic for failed scripts (up to 2 retries with 1-second delay).
- Configuration via external JSON file with automatic default-file creation.
- Graceful process termination sequence with `_force_kill_process()`.

**Issues:**
- Lambda closure capture bug in `run_all_scripts()` (BUG-014).
- `_force_kill_process()` is not called from `_read_output()` timeout path — only `process.terminate()` is used (BUG-021).
- `load_config()` calls `self.log_output()` before `setup_ui()` has created the output widget; exception caught broadly but warning is silently lost.

---

### 5. Tests GUI (`Tests-GUI.py`)

**Lines:** 782

**Strengths:**
- Bounded output buffer (`deque` with `maxlen=10000`) prevents memory leaks.
- Correct pytest installation check before running full suite.
- Proper PYTHONPATH injection for module resolution.
- Cancellation event for `run_all` operation.
- `ToolTip` helper class with delay scheduling and proper cleanup on leave/press.

**Issues:**
- `utils/theme_manager.py` missing → Theme menu absent, warning printed on startup (BUG-011).
- `_safe_log_output()` clears entire output area when buffer exceeds 5,000 entries, causing a jarring UI flash.
- Stop button cancels `run_all` but does not kill the currently running script process.

---

### 6. Test Suite (`tests/`)

**Collected:** 14 tests from 3 files (2 files failed to collect)
**Pass:** 10 | **Fail:** 4 | **Collection Error:** 2 files

| Test | File | Result | Root Cause |
|------|------|--------|-----------|
| `test_import_main` | `test_basic.py` | FAIL | Missing `jsonschema` |
| `test_import_validation` | `test_basic.py` | PASS | — |
| `test_project_structure` | `test_basic.py` | PASS | — |
| `test_config_files_exist` | `test_basic.py` | PASS | — |
| `test_sample_config_fixture` | `test_basic.py` | PASS | — |
| `test_temp_dir_fixture` | `test_basic.py` | PASS | — |
| `test_utils_directory_structure` | `test_utils.py` | PASS | — |
| `test_utility_files_exist` | `test_utils.py` | FAIL | Missing `data_quality_assessment.py` |
| `test_sample_data_fixture_structure` | `test_utils.py` | PASS | — |
| `test_validation_files_exist` | `test_validation.py` | PASS | — |
| `test_validation_config_structure` | `test_validation.py` | PASS | — |
| `test_apgi_dynamical_system_simulate_surprise_accumulation` | `test_validation.py` | FAIL | Missing `matplotlib` |
| `test_config_manager_load_save_cycle` | `test_validation.py` | FAIL | Missing `jsonschema` |
| `test_data_validator_validate_data_quality` | `test_validation.py` | PASS | — |
| All 6 tests in `test_integration.py` | `test_integration.py` | COLLECTION ERROR | Missing `jsonschema` |
| All tests in `test_performance.py` | `test_performance.py` | COLLECTION ERROR | Missing `jsonschema` |

**Observed test pass rate:** 71% of collected (10/14); ~63% of all intended tests if collection errors are counted as failures.

---

### 7. Configuration Files

**Files:** `config/default.yaml`, `config/gui_config.yaml`, `config/gui_alert_config.yaml`, `config/profiles/*.yaml`

**Issues:**
- Duplicate config directory: `config/` at project root AND `utils/config/` — both contain `default.yaml` and profile subdirectories. Unclear which takes precedence (BUG-012).
- `config/gui_config.yaml` is minimal (3 keys); full schema is implicitly defined only by GUI source code.
- No JSON Schema validation for config files (ironic given the missing `jsonschema` dependency).

---

### 8. Dependencies (`requirements.txt`)

**Installed packages relevant to this project:**
`click 8.3.1`, `numpy 2.4.2`, `pandas 3.0.1`, `psutil 7.2.2`, `rich 14.3.3`, `scipy 1.17.0`

**Required but missing from runtime environment:**
`jsonschema`, `matplotlib`, `seaborn`, `plotly`, `torch`, `torchvision`, `pymc`, `arviz`, `nilearn`, `tqdm`, `flask`, `dash`, `pydantic`, `sqlalchemy`, `alembic`, `numba`, `loguru`, `black`, `jupyter`, `python-dotenv`, `slowapi`

**Impact:** The majority of scientific computation and visualization features are non-functional in the current environment. `pip install -r requirements.txt` must be executed before any meaningful testing can occur.

---

## Actionable Recommendations

### Priority 1 — Fix Now (Blocking Issues)

**R-01: Install all required dependencies**
```bash
pip install -r requirements.txt
```
This alone will unblock the CLI, the integration tests, and the performance tests. Minimum immediate fix: `pip install jsonschema matplotlib`.

**R-02: Fix `self.module` AttributeError in Falsification GUI**
`Falsification/APGI-Falsification-Protocol-GUI.py` lines 279 and 315.
The `module` variable is local to `protocol_thread()`. Pass it as a parameter to `_handle_apgi_agent(module, ...)` and `_handle_iowa_gambling(module, ...)`, or assign `self.module = module` before calling those methods.

**R-03: Create `Validation/Master_Validation.py`**
Define a minimal `APGIMasterValidator` class exposing at minimum: `PROTOCOL_TIERS` dict, `protocol_results` dict, `falsification_status` dict, `timeout_seconds` attribute, and `generate_master_report()` method. Without this file the Validation GUI is completely non-functional.

**R-04: Create `utils/data_quality_assessment.py`**
Implement or stub the data quality assessment utility to satisfy both the test assertion and any code that calls it.

---

### Priority 2 — Fix Soon (High-Impact Defects)

**R-05: Wire CSV export button in Data Export tab**
In `Validation/APGI-Validation-GUI.py`, `create_export_widgets()`, add:
```python
ttk.Button(parent_frame, text="Export Results to CSV",
           command=self.export_csv).grid(row=0, column=0, pady=10, padx=10, sticky=(tk.W, tk.E))
```

**R-06: Register signal handler in `main.py` `formal_model()`**
After defining `handle_cancel`, add:
```python
import signal
signal.signal(signal.SIGINT, handle_cancel)
```

**R-07: Wrap `pd.read_csv()` at `main.py:655` in error handling**
```python
try:
    data = pd.read_csv(input_data)
except (FileNotFoundError, pd.errors.ParserError, PermissionError) as e:
    handle_file_error(input_data, "read", e)
    return
```

**R-08: Add Protocols 9–12 to the Validation GUI**
Extend `protocols_info` in `create_widgets()` to include entries for protocols 9–12, or add a numeric spin-box / range input.

**R-09: Fix lambda closure bug in `Utils-GUI.py`**
```python
# Change:
self.root.after(0, lambda: self.scripts_listbox.selection_set(i))
# To:
self.root.after(0, lambda idx=i: self.scripts_listbox.selection_set(idx))
```

**R-10: Fix `_run_parameter_simulation_worker` import path**
Replace `from APGI_Equations import CoreIgnitionSystem` with a `importlib.util.spec_from_file_location` call using the absolute path to `APGI-Equations.py`.

**R-11: Fix thread-safety of `messagebox` in Falsification GUI**
Replace `messagebox.showerror("Error", error_msg)` in `protocol_thread()` with a thread-safe alternative using `self.root.after(0, lambda: messagebox.showerror(...))`.

---

### Priority 3 — Improve (Medium-Impact)

**R-12: Replace generic protocol labels in Validation GUI**
Update `protocols_info` dict to use descriptive names sourced from the actual protocol files' docstrings (e.g., "Protocol 1: Surprise Accumulation Dynamics", "Protocol 2: Bayesian Parameter Estimation").

**R-13: Add `Falsification/__init__.py`**
Create an empty `__init__.py` to make the Falsification directory a proper Python package.

**R-14: Create `utils/theme_manager.py`** or remove the import and conditional Theme menu code from `Tests-GUI.py` to eliminate the startup warning.

**R-15: Consolidate duplicate config directories**
Remove `utils/config/` and update `config_manager.py` to reference `config/` at the project root as the single authoritative configuration location.

**R-16: Add a progress bar to the Falsification GUI** for long-running protocols (add a `ttk.Progressbar` in `setup_ui()` and call it indeterminate when a protocol thread is running).

**R-17: Rename "Generate PDF Report" button** to "Export Report (Text)" or implement actual PDF generation using `reportlab` or `fpdf2`.

**R-18: Remove or activate commented-out `log_performance()` calls** in `main.py`.

**R-19: Implement `make docs`**: Integrate `pdoc`, `sphinx`, or `mkdocs` and update the Makefile target accordingly.

---

### Priority 4 — Refactor / Improve Quality

**R-20: Split `main.py` (4,998 lines) into sub-modules**: Group commands by domain (e.g., `cli/simulation.py`, `cli/validation.py`, `cli/config.py`, `cli/data.py`) and compose a thin `main.py` that just assembles the Click group.

**R-21: Strengthen test coverage**: Add unit tests for individual CLI commands (use Click's `CliRunner`), GUI widget interactions (using `unittest.mock`), and all 12 validation protocol classes. Aim for ≥80% line coverage.

**R-22: Enable `mypy` strict mode** for the `utils/` package and gradually extend to all modules.

**R-23: Remove redundant local imports** of `numpy` and `pandas` in `main.py` at lines 474, 561, 743, and 1102.

**R-24: Fix `test_config_manager_load_save_cycle`** to assert correctness rather than silently passing on exception (remove the bare `except Exception: print(...)` pattern at line 138).

**R-25: Move hardcoded constants** (default port 8050, default host 127.0.0.1, max line counts, buffer sizes) into the config system.

---

*End of Report — APGI Validation Framework Audit v2.0 (2026-02-22)*
