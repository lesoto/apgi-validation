# APGI Theory Framework — Comprehensive Application Audit Report

**Report Date:** 2026-02-21
**Auditor:** Claude (Sonnet 4.6) — Automated Static + Structural Audit
**Repository Branch:** `claude/app-audit-testing-76F4X`
**Framework Version:** 1.3.0
**Scope:** Full codebase, CLI interface, GUI components, test infrastructure, configuration system, documentation

---

## Executive Summary

The APGI (Adaptive Pattern Generation and Integration) Theory Framework is a scientific Python CLI application providing computational modeling, multimodal data integration, Bayesian parameter estimation, validation protocols, and falsification testing for consciousness research. The framework is substantial (~5,000-line `main.py`, 63 Python files, 39 CLI commands, 3 GUI applications, 12+ validation protocols) but contains **critical unresolved issues** that prevent several commands from functioning correctly in production.

The most severe finding is that **four commands (`open-science`, `falsification`, `bayesian-estimation`, and `comprehensive-validation`) attempt to import files that do not exist** under the names they reference. The actual files are present but use hyphenated names (`APGI-Open-Science-Framework.py`) while the code references them without the `APGI-` prefix and with underscores (`Open_Science_Framework.py`). These commands will fail at runtime with `FileNotFoundError` or a module-spec error. Additionally, the configuration `--set` action is a **no-op** (the write path is permanently commented out), the `cache` command has a **wrong import path**, and the `visualize` command passes **wrong argument counts** to two helper functions (`_create_heatmap_plot` and `_create_distribution_plot`), causing `TypeError` failures at runtime.

Core infrastructure (logging, config loading, error handling, backup manager, data validation, batch processor, preprocessing pipelines) is well-engineered. The test suite structure is solid with clear pytest fixtures and markers, though it cannot be executed in this environment because none of the required dependencies (`pytest`, `click`, `numpy`, etc.) are installed at the system level.

---

## KPI Scores

| KPI | Score | Rationale |
|-----|-------|-----------|
| **1. Functional Completeness** | **52 / 100** | 39 CLI commands defined; ≥4 are broken at runtime due to wrong file references; 1 is a silent no-op (`--set`); 2 have argument-count bugs; `--follow` log option is declared but not implemented; `--port`/`--host` GUI options are accepted but silently ignored; `theme_manager` dependency missing for Tests-GUI; docs misrepresent several command signatures. |
| **2. UI/UX Consistency** | **60 / 100** | Rich console output is consistently styled across most commands. However, ~83 raw `print()` calls bypass Rich formatting in newer commands (`bayesian-estimation`, `falsification`, `open-science`, etc.), breaking visual consistency. Error messages inconsistently use Rich markup vs. plain text. GUI User Guide documents commands as positional subcommands (`python main.py gui validation`) when they are actually flags (`python main.py gui --gui-type validation`). |
| **3. Responsiveness & Performance** | **68 / 100** | `formal_model` simulation has progress tracking with cancellation support. `process_data` uses `Progress` with `BarColumn`. Batch processing uses `ThreadPoolExecutor`. Caching layer is implemented. However, large modules are fully imported at startup via `APGIModuleLoader`, which may significantly slow startup for users without all dependencies. No async I/O; web-based analysis interface (`--host`/`--port`) is accepted but not implemented in the `gui` command body. |
| **4. Error Handling & Resilience** | **63 / 100** | A central `error_handler.py` with typed error categories and severity levels exists. Most commands catch specific exception types. However: (a) several newer commands use bare `except Exception` masking the actual failure cause; (b) the duplicate `except ValueError` in `_set_config` will silently swallow `KeyError`/`AttributeError` after the first handler has already matched; (c) four commands will raise uncaught errors outside their try-blocks when module spec construction fails; (d) `_create_heatmap_plot` and `_create_distribution_plot` are called with missing positional arguments — these `TypeErrors` will propagate to the broad `visualize` exception handler but with no user-friendly guidance. |
| **5. Overall Implementation Quality** | **58 / 100** | Core scientific modules (APGI equations, liquid networks, entropy, cross-species, Bayesian framework) are extensive and well-commented. Infrastructure utilities are modular and testable. However, `main.py` at 4,998 lines is monolithic and mixes concerns. Multiple dead-code patterns exist (unreachable save block at L558, permanently-commented `set_parameter` calls). Wrong import paths for 4 major features indicate the integration layer was not fully tested. 63 Python files with 0 syntax errors is a positive sign. |

---

## Bug Inventory

### Critical Severity

---

**BUG-001 — Four commands fail at runtime: wrong file references**

- **Severity:** Critical
- **Affected Commands:** `open-science`, `falsification`, `bayesian-estimation`, `comprehensive-validation`
- **Affected File:** `main.py`
- **Line Numbers:** 2619, 2701, 2775, 3004
- **Description:** These commands use `importlib.util.spec_from_file_location()` to load modules by constructing `PROJECT_ROOT / "<name>.py"` paths. The actual files on disk are named with the `APGI-` prefix and hyphens, not the bare underscored names expected:

| Command | Expected path (in code) | Actual file on disk |
|---------|------------------------|---------------------|
| `open-science` | `Open_Science_Framework.py` | `APGI-Open-Science-Framework.py` |
| `falsification` | `Falsification_Framework.py` | `APGI-Falsification-Framework.py` |
| `bayesian-estimation` | `Bayesian_Estimation_Framework.py` | `APGI-Bayesian-Estimation-Framework.py` |
| `comprehensive-validation` | `Falsification_Framework.py` | `APGI-Falsification-Framework.py` |

- **Reproduction Steps:**
  1. `python main.py open-science --component compliance`
  2. `python main.py falsification --comprehensive`
  3. `python main.py bayesian-estimation --method mcmc`
  4. `python main.py comprehensive-validation`
- **Expected Behavior:** Commands execute against their respective framework files.
- **Actual Behavior:** `spec_from_file_location` returns `None` or raises `FileNotFoundError`; module execution silently fails or raises an unhandled `AttributeError` on `None.loader`.

---

**BUG-002 — `cache` command has wrong import path**

- **Severity:** Critical
- **Affected Command:** `cache`
- **Affected File:** `main.py:4216`
- **Description:** The `cache` command imports `from data.cache_manager import CacheManager`. The actual module is located at `utils/cache_manager.py`, not `data/cache_manager.py`. No `data/` package exists at the project root.
- **Reproduction Steps:** `python main.py cache --action status`
- **Expected Behavior:** Displays cache status table.
- **Actual Behavior:** `ModuleNotFoundError: No module named 'data.cache_manager'`

---

**BUG-003 — `visualize --plot-type heatmap` and `--plot-type distribution` crash with TypeError**

- **Severity:** Critical
- **Affected Command:** `visualize`
- **Affected File:** `main.py:3744, 3747`
- **Description:** `_create_plot_by_type` calls `_create_heatmap_plot(data, colormap, alpha)` (3 args) but the function signature is `_create_heatmap_plot(data, colormap, alpha, sns, plt)` (5 args). Similarly, `_create_distribution_plot(data, bins_val, alpha, grid)` (4 args) is called but the function requires `(data, bins_val, alpha, grid, plt)` (5 args). The `plt` (and `sns` for heatmap) references are available in the outer scope but are not passed through.
- **Reproduction Steps:**
  1. `python main.py visualize --input-file data.csv --plot-type heatmap`
  2. `python main.py visualize --input-file data.csv --plot-type distribution`
- **Expected Behavior:** Heatmap or distribution plot renders.
- **Actual Behavior:** `TypeError: _create_heatmap_plot() missing 2 required positional arguments: 'sns' and 'plt'` (and similarly for distribution).

---

**BUG-004 — `monitor-performance --command cross-species` uses wrong import path**

- **Severity:** Critical
- **Affected Command:** `monitor-performance`
- **Affected File:** `main.py:1701`
- **Description:** When `command == "cross-species"`, the code does `from APGI_Cross_Species_Scaling import CrossSpeciesScaling`. The actual file is named `APGI-Cross-Species-Scaling.py`. Python cannot import a hyphenated filename directly; dynamic loading via `importlib` is required. This will raise `ModuleNotFoundError`.
- **Reproduction Steps:** `python main.py monitor-performance --command cross-species`
- **Expected Behavior:** Cross-species analysis is benchmarked.
- **Actual Behavior:** `ModuleNotFoundError: No module named 'APGI_Cross_Species_Scaling'`

---

### High Severity

---

**BUG-005 — `config --set` is a silent no-op**

- **Severity:** High
- **Affected Command:** `config --set`
- **Affected File:** `main.py:2267–2276`
- **Description:** The `_set_config` function always sets `success = True` and prints "Configuration updated successfully" but the actual `set_parameter(section, param, value)` call is permanently commented out. No parameter value is ever written. The user receives a false confirmation.
- **Reproduction Steps:** `python main.py config --set model.tau_S=0.8` → outputs success message, but value is not persisted.
- **Expected Behavior:** The configuration value is updated in memory and/or saved to YAML.
- **Actual Behavior:** The function prints success but performs no write operation.

---

**BUG-006 — Duplicate `except ValueError` in `_set_config` — second handler is unreachable**

- **Severity:** High
- **Affected File:** `main.py:2287–2294`
- **Description:** The function has two consecutive `except` clauses — `except ValueError` followed by `except (ValueError, KeyError, AttributeError)`. In Python, once an exception is matched by the first handler, the second handler is never evaluated. `KeyError` and `AttributeError` that might arise from config operations will be caught by the first `except ValueError` handler, which prints a misleading usage message.
- **Reproduction Steps:** Inject a `KeyError` or `AttributeError` in `_set_config`.
- **Expected Behavior:** Distinct error messages for different exception types.
- **Actual Behavior:** `KeyError`/`AttributeError` triggers the `ValueError` handler, printing a usage message.

---

**BUG-007 — `logs --follow` option accepted but not implemented**

- **Severity:** High
- **Affected Command:** `logs --follow`
- **Affected File:** `main.py:3256, 3260–3327`
- **Description:** `--follow` is declared as a click option with help text "Follow log file in real-time" but the `follow` variable is accepted as a parameter and never referenced in the function body. No `tail -f`-style polling or inotify logic exists.
- **Reproduction Steps:** `python main.py logs --follow`
- **Expected Behavior:** Log file output streams in real-time.
- **Actual Behavior:** Displays the last 20 lines and exits immediately.

---

**BUG-008 — `gui --host` and `--port` options accepted but silently ignored**

- **Severity:** High
- **Affected Command:** `gui`
- **Affected File:** `main.py:3229–3251`
- **Description:** `--port` and `--host` are declared click options (default `8050` and `127.0.0.1`) but are never used inside the `gui()` function. The function dispatches to `_launch_validation_gui()`, `_launch_psychological_gui()`, or `_launch_analysis_gui()` without passing `host` or `port`. The GUI User Guide documents a web-based analysis interface on `localhost:8080`, but no such web server is launched.
- **Reproduction Steps:** `python main.py gui --gui-type analysis --host 0.0.0.0 --port 9000`
- **Expected Behavior:** Web analysis interface starts on port 9000.
- **Actual Behavior:** Attempts to run `APGI-Entropy-Implementation.py` as a desktop GUI, ignoring network parameters.

---

**BUG-009 — `Tests-GUI.py` imports missing `utils.theme_manager` module**

- **Severity:** High
- **Affected File:** `Tests-GUI.py:23–28`
- **Description:** `Tests-GUI.py` unconditionally attempts `from utils.theme_manager import ThemeManager`. The file `utils/theme_manager.py` does not exist in the repository. A warning is printed to stdout but the application continues with `THEME_MANAGER_AVAILABLE = False`. Since the GUI may use theme functionality that is not gracefully degraded throughout the file, this could cause `NameError` or `AttributeError` at runtime when theme functions are called.
- **Reproduction Steps:** Run `Tests-GUI.py` directly.
- **Expected Behavior:** Theme support either works or the application uses a built-in default theme.
- **Actual Behavior:** Prints "Warning: Theme manager not available" and theme support is silently disabled.

---

**BUG-010 — Unreachable code: redundant `if not save_file` check (L558)**

- **Severity:** High
- **Affected Command:** `formal-model`
- **Affected File:** `main.py:556–559`
- **Description:** Inside the `if save_file:` branch, there is a nested `if not save_file:` block (L558) that assigns a fallback filename. This inner condition can never be `True` because the outer branch guarantees `save_file` is truthy. The auto-generated timestamped filename for the case where `--output-file` is not provided is therefore dead code — if no output file is given, the `if save_file:` block is skipped entirely.
- **Impact:** Users who run `python main.py formal-model` without `--output-file` get no automatic save to a timestamped file despite the code's intent.

---

### Medium Severity

---

**BUG-011 — GUI User Guide documents wrong command syntax for `gui` subcommand**

- **Severity:** Medium
- **Affected File:** `docs/GUI-User-Guide.md`
- **Description:** The documentation shows:
  ```bash
  python main.py gui validation       # WRONG
  python main.py gui psychological    # WRONG
  python main.py gui analysis         # WRONG
  ```
  The actual CLI syntax requires the `--gui-type` flag:
  ```bash
  python main.py gui --gui-type validation
  python main.py gui --gui-type psychological
  python main.py gui --gui-type analysis
  ```
  The positional `validation`/`psychological`/`analysis` tokens would be silently ignored by Click (it only sees options, not positional arguments here), so users following the docs would always get the default `validation` GUI regardless of what they typed.
- **Expected Behavior:** Documentation accurately reflects CLI syntax.

---

**BUG-012 — `visualize` command: None defaults for required parameters cause TypeError in helpers**

- **Severity:** Medium
- **Affected Command:** `visualize`
- **Affected File:** `main.py:3867–3893`
- **Description:** Options such as `--plot-type`, `--style`, `--palette`, `--figsize`, `--bins`, `--linewidth`, `--markersize`, `--font-family`, `--font-size`, `--save-format`, `--aspect`, `--subplot-rows`, `--subplot-cols` have no `default=` values, so they resolve to `None` when not provided. The `_parse_visualization_parameters` function attempts `map(int, figsize.split(","))` on a `None` `figsize`, causing `AttributeError: 'NoneType' object has no attribute 'split'`. Similarly `_setup_plotting_style` calls `sns.set_style(style)` where `style` may be `None`.
- **Reproduction Steps:** `python main.py visualize --input-file data.csv`
- **Expected Behavior:** Uses sensible defaults (e.g., `figsize="12,8"`, `style="default"`).
- **Actual Behavior:** `AttributeError` crash.

---

**BUG-013 — Inconsistent output: 83 raw `print()` calls in newer commands**

- **Severity:** Medium
- **Affected Commands:** `neural-signatures`, `causal-manipulations`, `quantitative-fits`, `clinical-convergence`, `open-science`, `falsification`, `bayesian-estimation`, `comprehensive-validation`
- **Affected File:** `main.py` (multiple lines)
- **Description:** The codebase uses `console.print()` (Rich) for structured output in most commands. However, 83 raw `print()` calls were found, concentrated in the newer scientific validation and Bayesian estimation commands. These bypass Rich's terminal formatting (colors, panels, alignment) producing inconsistent visual output.

---

**BUG-014 — `_set_config` rejects values containing `=` sign unnecessarily**

- **Severity:** Medium
- **Affected Command:** `config --set`
- **Affected File:** `main.py:2248–2249`
- **Description:** `_set_config(key, value)` raises `ValueError("Parameter value cannot contain '=' separator")` if `value` contains `=`. However, the caller (`config` command) already splits `set` string on the first `=` before calling `_set_config`, so `value` would only contain `=` if the user explicitly provides something like `model.key=val=extra`. The validation is overly restrictive, rejecting valid values like URL parameters or mathematical expressions.

---

**BUG-015 — `log-level` CLI override has no effect**

- **Severity:** Medium
- **Affected Option:** `--log-level` (global)
- **Affected File:** `main.py:289–291`
- **Description:** The global `--log-level` option handler reads:
  ```python
  # set_parameter("logging", "level", log_level.upper())
  apgi_logger.logger.info(f"Log level overridden to: {log_level.upper()}")
  ```
  The `set_parameter` call is commented out. The log level is never actually changed; only an INFO message is emitted at the current (unconfigured) log level.

---

**BUG-016 — `violin` plot uses incorrect DataFrame method**

- **Severity:** Medium
- **Affected Command:** `visualize --plot-type violin`
- **Affected File:** `main.py:3752`
- **Description:** `data[numeric_cols].violinplot(alpha=float(alpha))` is called on a DataFrame. Pandas DataFrames do not have a `.violinplot()` method. The correct call would be `data[numeric_cols].plot.violin()` or use `matplotlib.pyplot.violinplot()`. This will raise `AttributeError: 'DataFrame' object has no attribute 'violinplot'`.
- **Reproduction Steps:** `python main.py visualize --input-file data.csv --plot-type violin`
- **Expected Behavior:** A violin plot is rendered.
- **Actual Behavior:** `AttributeError` crash.

---

**BUG-017 — `docs` Makefile target is a stub**

- **Severity:** Medium
- **Affected File:** `Makefile:36–37`
- **Description:** The `docs` target prints "Documentation generation not yet implemented" and does nothing. Given the extensive `docs/` directory exists, this target should generate API docs (e.g., via Sphinx or pdoc3).

---

### Low Severity

---

**BUG-018 — `log` `Modified` column shows raw Unix timestamp float**

- **Severity:** Low
- **Affected Command:** `logs`
- **Affected File:** `main.py:3297`
- **Description:** `modified = f"{log_file.stat().st_mtime}"` produces a raw float timestamp like `1740099600.0` instead of a human-readable date. Should use `datetime.fromtimestamp()`.

---

**BUG-019 — `__init__.py` references non-existent `APGI_Master_Validation` class**

- **Severity:** Low
- **Affected File:** `__init__.py:16`
- **Description:** `from .Validation.APGI_Master_Validation import APGIMasterValidator` uses an underscored module name. No file named `APGI_Master_Validation.py` exists in `Validation/`. The import is wrapped in a `try/except ImportError` so it silently fails, leaving `__all__ = []`.

---

**BUG-020 — `setup_environment.py` not referenced in Makefile or README**

- **Severity:** Low
- **Affected File:** `setup_environment.py`
- **Description:** A `setup_environment.py` file exists but is not invoked by `make install`, `make dev-install`, or any documented workflow. New contributors may not discover it.

---

**BUG-021 — `data[numeric_cols].violinplot` — see BUG-016**

---

**BUG-022 — `Validation-Protocol-*` files may lack `run_validation()` function**

- **Severity:** Low
- **Affected Command:** `validate`
- **Affected File:** `main.py:1877–1943`
- **Description:** The `validate` command checks `hasattr(protocol_module, "run_validation")` and falls back to printing "No validation function" rather than raising an error. If a protocol file exists but does not define `run_validation()`, the user gets a silent non-failure — results are stored as the string `"No validation function"` and saved to JSON. This may mislead users into thinking validation passed.

---

**BUG-023 — `export_data` catches `json.JSONEncodeError` (wrong exception name)**

- **Severity:** Low
- **Affected File:** `main.py:4113`
- **Description:** `except (FileNotFoundError, PermissionError, ValueError, json.JSONEncodeError)` — `json.JSONEncodeError` does not exist; the correct name is `json.JSONDecodeError`. Raised `JSONDecodeError` exceptions during error recovery would not be caught, propagating unexpectedly.

---

## Missing Features / Incomplete Implementations

| ID | Feature | Status | Evidence |
|----|---------|--------|----------|
| MF-001 | **Real-time log following (`logs --follow`)** | Declared, not implemented | `follow` parameter is accepted but never used in `logs()` body (`main.py:3260`) |
| MF-002 | **Web-based analysis interface (`gui analysis`)** | Misdirected to desktop script | `_launch_analysis_gui` points to `APGI-Entropy-Implementation.py` (a desktop script), not a Flask/Dash web server |
| MF-003 | **`--host`/`--port` web server in `gui` command** | Accepted but not used | GUI command accepts `--host` and `--port` but never starts a web server (`main.py:3229–3251`) |
| MF-004 | **Persistent configuration writes (`config --set`)** | Silent no-op | `set_parameter` permanently commented out (`main.py:2267–2276`) |
| MF-005 | **`--log-level` global override** | Commented out | `set_parameter("logging", "level", ...)` is commented (`main.py:290`) |
| MF-006 | **Documentation generation (`make docs`)** | Stub | Makefile `docs` target prints message only |
| MF-007 | **`utils/theme_manager.py`** | Missing file | `Tests-GUI.py` imports it but it doesn't exist in `utils/` |
| MF-008 | **`open-science` command** | Runtime failure | Resolves to wrong file path `Open_Science_Framework.py` |
| MF-009 | **`falsification` command** | Runtime failure | Resolves to wrong file path `Falsification_Framework.py` |
| MF-010 | **`bayesian-estimation` command** | Runtime failure | Resolves to wrong file path `Bayesian_Estimation_Framework.py` |
| MF-011 | **`comprehensive-validation` command** | Runtime failure | Calls `falsification` framework via wrong path |
| MF-012 | **Validation protocol descriptions in `validate` table** | Placeholder text | All protocols shown with generic `"Validation Protocol N"` description; no protocol metadata is loaded |
| MF-013 | **`__init__.py` public API** | Silently empty | Import of `APGIMasterValidator` fails silently; `__all__` is empty |
| MF-014 | **`Makefile docs` target** | Stub | No documentation generation tool configured |
| MF-015 | **`parameter_validator` module** | Conditional usage | `from parameter_validator import validate_parameters` — this is a relative import that may not resolve; falls back gracefully, but validation is skipped |

---

## Recommendations for Remediation

### P0 — Immediate Fixes (Breaking Functionality)

1. **Fix file path references (BUG-001, BUG-004):** Replace all occurrences of bare underscore-named paths with the correct hyphenated filenames using `importlib.util.spec_from_file_location`. For example:
   - `"Open_Science_Framework.py"` → `"APGI-Open-Science-Framework.py"`
   - `"Falsification_Framework.py"` → `"APGI-Falsification-Framework.py"`
   - `"Bayesian_Estimation_Framework.py"` → `"APGI-Bayesian-Estimation-Framework.py"`
   - Remove the direct `from APGI_Cross_Species_Scaling import ...` and use `importlib` instead.

2. **Fix `cache` import path (BUG-002):** Change `from data.cache_manager import CacheManager` to `from utils.cache_manager import CacheManager`.

3. **Fix `visualize` helper function calls (BUG-003, BUG-012, BUG-016):**
   - Pass `sns` and `plt` to `_create_heatmap_plot`.
   - Pass `plt` to `_create_distribution_plot`.
   - Add explicit defaults to all `visualize` click options (e.g., `default="auto"` for `--plot-type`, `default="default"` for `--style`, `default="12,8"` for `--figsize`, etc.).
   - Replace `data[numeric_cols].violinplot(...)` with `data[numeric_cols].plot.violin(...)`.

### P1 — High-Priority Fixes

4. **Implement config persistence (BUG-005, MF-004):** Uncomment and wire `set_parameter` calls in `_set_config` and `config --log-level` handler. Alternatively, implement direct YAML write using `config_manager.set_parameter`.

5. **Implement `logs --follow` (BUG-007, MF-001):** Add a polling loop (e.g., using `file.seek`, `time.sleep(0.5)`) or use `watchdog` library when `follow=True`.

6. **Fix duplicate `except ValueError` in `_set_config` (BUG-006):** Merge into `except (ValueError, KeyError, AttributeError)` as a single clause.

7. **Create `utils/theme_manager.py` stub or remove import (BUG-009, MF-007):** Either implement a minimal `ThemeManager` class or remove the import and replace usages with hardcoded defaults.

8. **Fix dead code in `formal-model` save path (BUG-010):** Move the auto-timestamped filename logic outside the `if save_file:` gate, so results are saved when no `--output-file` is given.

### P2 — Medium-Priority Improvements

9. **Fix GUI documentation (BUG-011):** Update `docs/GUI-User-Guide.md` to reflect actual `--gui-type` flag syntax.

10. **Implement or document `gui --host`/`--port` (BUG-008, MF-003):** Either wire up a Flask/Dash server using the provided parameters, or remove the options and update docs.

11. **Replace raw `print()` with `console.print()` (BUG-013):** Apply Rich formatting consistently across all commands. This is particularly important for the scientific validation results output.

12. **Fix `json.JSONEncodeError` (BUG-023):** Replace with `json.JSONDecodeError`.

13. **Fix `__init__.py` import (BUG-019):** Either create `Validation/APGI_Master_Validation.py` with the `APGIMasterValidator` class, or update the import to reference the correct module name.

### P3 — Low-Priority / Quality Improvements

14. **Human-readable timestamps in `logs` command (BUG-018):** Replace `f"{log_file.stat().st_mtime}"` with `datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")`.

15. **Implement `make docs` (BUG-017, MF-014):** Configure `pdoc` or `sphinx-apidoc` for automatic API documentation from docstrings.

16. **Load protocol metadata in `validate` table (MF-012):** Parse a `__doc__` or metadata dict from each protocol file to display meaningful descriptions.

17. **Install dependencies and run test suite:** The test suite cannot run in the current environment. Ensure `requirements.txt` packages are installable and CI executes `pytest` on every commit.

18. **Address `parameter_validator` import path (MF-015):** Ensure `from parameter_validator import validate_parameters` resolves correctly, or change to `from utils.parameter_validator import validate_parameters`.

---

## Test Infrastructure Assessment

- **Test files:** 3 (`test_basic.py`, `test_integration.py`, `test_performance.py`)
- **Fixtures:** Well-defined in `conftest.py` (`temp_dir`, `sample_config`, `sample_data`)
- **Markers:** Correctly configured (`unit`, `integration`, `performance`, `slow`)
- **Coverage target:** 60% (configured in `pytest.ini`)
- **Status:** Cannot be executed — no dependencies installed at system level (`pytest`, `numpy`, `pandas`, etc. all missing). Once dependencies are installed, tests are structurally sound.
- **Gap:** No tests cover the CLI commands in `main.py` (no `click.testing.CliRunner` tests). The 39 CLI commands are entirely untested.

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Total Python files audited | 63 |
| Syntax errors | 0 |
| CLI commands defined | 39 |
| CLI commands broken at runtime (critical path) | ≥5 |
| Commands with silent no-op behaviour | 1 (`config --set`) |
| Declared but unimplemented features | ≥3 (`--follow`, `--host`/`--port` for web, `make docs`) |
| Missing files referenced by code | 4 |
| Missing utility module | 1 (`theme_manager.py`) |
| Inconsistent `print()` vs `console.print()` calls | 83 raw `print()` occurrences |
| Total bugs documented | 23 |
| Critical severity bugs | 4 |
| High severity bugs | 6 |
| Medium severity bugs | 7 |
| Low severity bugs | 6 |

---

*This report was generated via automated static analysis and structural code inspection. Dynamic/runtime testing was not performed due to missing runtime dependencies. All findings should be verified against a fully-configured development environment before remediation.*
