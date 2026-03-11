# APGI Validation Framework -- Comprehensive Application Audit Report

**Date:** 2026-03-10
**Version Audited:** 1.3.0
**Auditor:** Automated Security & Quality Audit
**Scope:** End-to-end audit of all pages, interactive elements, settings, and user-facing options

---

## Executive Summary

The APGI (Adaptive Pattern Generation and Integration) Theory Framework is a sophisticated Python-based scientific research application comprising ~76 Python files, 35,000+ lines of code, 4 GUI applications, 12 validation protocols, 6 falsification protocols, and a CLI with 15+ commands. The framework implements computational models of consciousness and psychological state dynamics.

### Key Findings

| Metric | Value |
|--------|-------|
| **Total Issues Found** | **64** |
| Critical Severity | 9 |
| High Severity | 14 |
| Medium Severity | 22 |
| Low Severity | 19 |
| **Test Suite Status** | **100% FAILURE (76/76 tests error)** |
| Security Vulnerabilities | 8 |
| Missing/Incomplete Features | 12 |

**Overall Application Health: AT RISK** -- The application has strong architectural foundations and security awareness, but critical bugs prevent the test suite from running entirely, multiple CLI commands crash at runtime, and several GUI components have thread-safety violations that can cause hangs or data loss.

---

## KPI Scores Table

| Dimension | Score | Grade | Indicator |
|-----------|-------|-------|-----------|
| **Functional Completeness** | **38/100** | F | :red_circle: |
| **UI/UX Consistency** | **52/100** | D | :red_circle: |
| **Responsiveness & Performance** | **55/100** | D | :orange_circle: |
| **Error Handling & Resilience** | **42/100** | F | :red_circle: |
| **Implementation Quality** | **48/100** | F | :red_circle: |
| **Security Posture** | **68/100** | C | :orange_circle: |
| **Overall Score** | **50/100** | F | :red_circle: |

### Scoring Criteria

| Range | Grade | Indicator | Meaning |
|-------|-------|-----------|---------|
| 90-100 | A | :green_circle: | Excellent -- production-ready |
| 75-89 | B | :green_circle: | Good -- minor issues only |
| 60-74 | C | :orange_circle: | Fair -- notable gaps, needs attention |
| 40-59 | D | :orange_circle: | Poor -- significant issues |
| 0-39 | F | :red_circle: | Critical -- major rework needed |

### Dimension Scoring Rationale

**Functional Completeness (38/100):** 4 of 15+ CLI commands guaranteed to crash at import time. `config --set` is non-functional (changes silently discarded). The entire test suite (76 tests) fails due to a mathematical bug in entropy validation. Key advertised features like `tacs` intervention, `--modalities` option, and cross-species monitoring are broken or non-functional.

**UI/UX Consistency (52/100):** Validation GUI has solid architecture with queue-based thread-safe updates. However, Falsification GUI lacks window close handlers, has uninitialized variables, and a non-functional Stop button. Ctrl+C is overridden to clear output (breaks copy). Theme system has shared mutable references. No DPI/scaling awareness across any GUI. Documented keyboard shortcuts don't exist.

**Responsiveness & Performance (55/100):** All framework modules loaded eagerly at startup regardless of which command runs. Unbounded list growth in performance dashboard. Log files read entirely into memory. GUI update polling at 500ms feels sluggish. No operation timeouts that actually kill threads.

**Error Handling & Resilience (42/100):** 24 instances of `except Exception: pass` silently swallowing errors. `ErrorCategory.RUNTIME` referenced but doesn't exist in the enum. Crash recovery has no thread safety on state file writes. Backup integrity verification is fundamentally broken (hashes archive vs. extracted content). No `atexit` handlers for clean shutdown.

**Implementation Quality (48/100):** Inconsistent naming between files (hyphens vs underscores). Duplicate import paths. Dead code blocks throughout. Mixed `print()` vs `console.print()` vs `logger` usage. 60+ untyped `except Exception` catches masking bugs. Zero CLI command tests despite 15+ commands.

**Security Posture (68/100):** Good foundations -- HMAC-signed pickle, safe YAML loading everywhere, path traversal protection module, error sanitization infrastructure. Gaps: traceback leakage to dashboard HTML, inconsistent path validation, insecure HMAC fallback key, wide dependency version ranges allowing known-vulnerable versions.

---

## Prioritized Bug Inventory

### :red_circle: CRITICAL (9 issues)

| ID | Component | Description | File | Line(s) |
|----|-----------|-------------|------|---------|
| C-01 | Entropy Validation | **Shannon entropy calculation is mathematically broken** -- compares per-symbol entropy (max ~4 bits for hex chars) against total-bits threshold (448+ bits). A hex-encoded key can NEVER pass validation. This breaks the entire test suite (76/76 tests fail) and prevents module imports at runtime. | `utils/batch_processor.py` | 64-81 |
| C-02 | CLI Commands | **4 commands reference non-existent files** -- `open-science` loads `Open_Science_Framework.py` (actual: `APGI-Open-Science-Framework.py`), `falsification` loads `Falsification_Framework.py`, `bayesian-estimation` loads `Bayesian_Estimation_Framework.py`, `comprehensive-validation` loads same. All crash with `AttributeError` on `NoneType.loader`. | `main.py` | 2978, 3060, 3134, 3363 |
| C-03 | CLI Commands | **Cross-species monitoring imports wrong module name** -- `from APGI_Cross_Species_Scaling import CrossSpeciesScaling` but actual file is `APGI-Cross-Species-Scaling.py`. | `main.py` | 2066 |
| C-04 | CLI Commands | **Open-science command has typo in class name** -- instantiates `PrereregistrationTemplate` (triple "re") instead of `PreregistrationTemplate`. | `main.py` | 2991 |
| C-05 | Backup Manager | **`_load_history()` returns `None` when no history file exists** -- missing `return []` after the `if self.history_file.exists():` block. First backup attempt crashes with `AttributeError: 'NoneType' object has no attribute 'append'`. | `utils/backup_manager.py` | 132-178 |
| C-06 | Backup Manager | **Undefined variable `backup_id` in restore error path** -- `_restore_zip_backup()` references `backup_id` but method signature only receives `backup_file`. Raises `NameError` during integrity check failure. | `utils/backup_manager.py` | 568 |
| C-07 | Error Handler | **`ErrorCategory.RUNTIME` does not exist** -- `critical_error()` convenience function references this non-existent enum member. Any call raises `AttributeError`. | `utils/error_handler.py` | 559 |
| C-08 | Falsification GUI | **Uninitialized instance variables** -- `self.selected_protocol` and `self.parameter_values` used in multiple methods but never initialized in `__init__`. First interaction raises `AttributeError`. | `Falsification/APGI-Falsification-Protocol-GUI.py` | 699-845 |
| C-09 | Setup Script | **`PROJECT_ROOT` points to wrong directory** -- `Path(__file__).parent.parent` resolves to `/home/user/` instead of `/home/user/apgi-validation/`. Venv and dependencies installed in wrong location. | `setup_environment.py` | 15 |

**Reproduction Steps for C-01:**
```bash
export PICKLE_SECRET_KEY=$(openssl rand -hex 32)
export APGI_BACKUP_HMAC_KEY=$(openssl rand -hex 32)
python -m pytest tests/ -q
# Result: ALL 76 tests fail with:
# ValueError: PICKLE_SECRET_KEY has insufficient entropy (3.8 bits, requires at least 448 bits)
```

**Reproduction Steps for C-02:**
```bash
python main.py open-science --type preregistration
# Result: AttributeError: 'NoneType' object has no attribute 'loader'
```

---

### :orange_circle: HIGH (14 issues)

| ID | Component | Description | File | Line(s) |
|----|-----------|-------------|------|---------|
| H-01 | CLI Config | **`config --set` is completely non-functional** -- the actual `set_parameter()` call is commented out. Function always sets `success = True` and reports success, but changes are silently discarded. | `main.py` | 2626-2635 |
| H-02 | CLI Import | **Wrong import path for `parameter_validator`** -- `from parameter_validator import` should be `from utils.parameter_validator import`. | `main.py` | 421 |
| H-03 | CLI Commands | **Missing `tacs` intervention handler** -- `causal_manipulations` command accepts `--intervention tacs` but dispatch only handles `tms`, `pharmacological`, `metabolic`. | `main.py` | 2818-2825 |
| H-04 | Timeout Handler | **Timeouts do NOT kill operations** -- spawns daemon thread, waits with `join(timeout=...)`, raises `TimeoutError` if thread is alive, but thread continues running indefinitely. CPU/memory/file handles leak. | `utils/timeout_handler.py` | 160-221 |
| H-05 | Backup Integrity | **Backup checksum verification is fundamentally broken** -- `_calculate_checksum()` hashes the archive file; `_verify_restored_integrity()` hashes extracted content. These always produce different values. | `utils/backup_manager.py` | 209 vs 601 |
| H-06 | Validation GUI | **Ctrl+C overrides clipboard copy** -- `<Control-c>` is bound to `clear_output()`, breaking the standard copy shortcut for the entire window. | `Validation/APGI_Validation_GUI.py` | 105 |
| H-07 | Validation GUI | **Widget config called from worker thread** -- `_ensure_ui_consistency()` in the `_run_validation_worker` finally block directly calls `run_button.config()` from a background thread, violating tkinter thread safety. | `Validation/APGI_Validation_GUI.py` | 1473 |
| H-08 | Validation GUI | **Non-daemon validation thread** -- created without `daemon=True`. If close handler fails to stop it, Python process hangs indefinitely. | `Validation/APGI_Validation_GUI.py` | 1128 |
| H-09 | Falsification GUI | **No window close handler** -- no `WM_DELETE_WINDOW` protocol binding. Clicking X while protocol runs leaves background thread alive. | `Falsification/APGI-Falsification-Protocol-GUI.py` | N/A |
| H-10 | Falsification GUI | **`messagebox.showerror()` called from background thread** -- can crash or freeze the application on some platforms. | `Falsification/APGI-Falsification-Protocol-GUI.py` | 997 |
| H-11 | Falsification GUI | **`stop_protocol()` is empty** -- Stop button calls this method but body does nothing. Users cannot cancel running protocols. | `Falsification/APGI-Falsification-Protocol-GUI.py` | 916-917 |
| H-12 | Security | **Error information leakage** -- `str(e)` rendered directly into web dashboard HTML in 9 locations, exposing internal paths and stack traces. | `utils/performance_dashboard.py` | 136, 258, 281, etc. |
| H-13 | Security | **Output path validation applied inconsistently** -- `cross_species`, `causal_manipulations`, `quantitative_fits`, `clinical_convergence`, `open_science`, `bayesian_estimation`, `comprehensive_validation`, and `analyze_logs` write to user-provided paths with zero validation. | `main.py` | Various |
| H-14 | Utils GUI | **Double line read on Windows** -- `_read_output` reads `process.stdout.readline()` then reads again when `ready` is True, losing every other line. | `Utils-GUI.py` | 543-561 |

---

### :yellow_circle: MEDIUM (22 issues)

| ID | Component | Description | File | Line(s) |
|----|-----------|-------------|------|---------|
| M-01 | Exception Handling | 24 instances of `except Exception: pass` silently swallowing errors across 13 files | Multiple | Various |
| M-02 | Exception Handling | 60+ broad `except Exception as e` catches masking bugs in 8 CLI commands | `main.py` | Various |
| M-03 | Exception Handling | Duplicate/unreachable `ValueError` handler in `_set_config()` | `main.py` | 2646-2653 |
| M-04 | Memory | Unbounded list growth in `system_metrics` and `validation_results` | `utils/performance_dashboard.py` | 59-60 |
| M-05 | Memory | Global `_crash_recoveries` dict never cleaned up; accumulates instances with file handles and threads | `utils/crash_recovery.py` | 297 |
| M-06 | Memory | All 5 large framework modules loaded eagerly at startup regardless of which CLI command is invoked | `main.py` | 261 |
| M-07 | Race Condition | `signal.signal(SIGINT)` called potentially from non-main thread in parallel validation | `main.py` | 532 |
| M-08 | Race Condition | Dashboard lists accessed without consistent locking in Dash callbacks | `utils/performance_dashboard.py` | 59-61, 650 |
| M-09 | Thread Safety | Crash recovery state file writes have no lock; `_auto_save_worker` and main thread both write | `utils/crash_recovery.py` | 64-80, 119 |
| M-10 | Thread Safety | `SIGALRM` signal handlers not portable (Windows) and not thread-safe | `utils/input_validation.py` | 422 |
| M-11 | Cache Manager | `cached()` decorator cannot cache `None` return values; uses `is not None` check | `utils/cache_manager.py` | 496-498 |
| M-12 | Cache Manager | Broken import in `warm_cache()` -- `from data_validation import` should be `from utils.data_validation import` | `utils/cache_manager.py` | 305 |
| M-13 | Master Validator | `timeout_seconds` defined but never used; stuck protocols block orchestrator indefinitely | `Validation/Master_Validation.py` | 46, 152 |
| M-14 | Master Validator | `falsification_status` initialized but never populated | `Validation/Master_Validation.py` | 41-45 |
| M-15 | Error Handler | `ErrorSeverity.INFO` missing from severity-to-log-level mapping; falls through to `.error()` | `utils/error_handler.py` | 709-714 |
| M-16 | Validation GUI | Queue drain in finally block discards up to 50 queued final updates | `Validation/APGI_Validation_GUI.py` | 1475-1488 |
| M-17 | Validation GUI | Empty spinbox values pass validation, causing downstream errors | `Validation/APGI_Validation_GUI.py` | 1969, 1979 |
| M-18 | Falsification GUI | Theme radio buttons each create own `StringVar`; never visually track selection | `Falsification/APGI-Falsification-Protocol-GUI.py` | 338 |
| M-19 | Falsification GUI | Relative config path uses CWD instead of `PROJECT_ROOT` | `Falsification/APGI-Falsification-Protocol-GUI.py` | 773 |
| M-20 | Tests GUI | Run buttons never disabled during execution; users can start overlapping processes | `Tests-GUI.py` | N/A |
| M-21 | Dead Code | `_validate_output_file_path()` defined but never called (731-797), commented-out `set_parameter()` call (2626-2635), `cancel_message = []` appended to but never read (525-530) | `main.py` | Various |
| M-22 | CLI Output | Mixed `print()` / `console.print()` -- `print()` bypasses `--quiet` flag in 15+ locations | `main.py` | Various |

---

### :white_circle: LOW (19 issues)

| ID | Component | Description | File | Line(s) |
|----|-----------|-------------|------|---------|
| L-01 | Logging | 60+ uses of `print()` instead of logger across utils/ | Multiple | Various |
| L-02 | Logging | Config value logged in plaintext in `_set_config()` -- potential credential exposure | `main.py` | 2639 |
| L-03 | Naming | Confusing `APGI_Multimodal_Integration.py` (310 lines) vs `APGI-Multimodal-Integration.py` (3685 lines) -- different modules with similar names | Root | N/A |
| L-04 | CLI | Redundant `falsify` (works) vs `falsification` (broken) commands | `main.py` | 2418 vs 3043 |
| L-05 | CLI | `--modalities` option on `multimodal` command accepted but never used | `main.py` | 1066-1072 |
| L-06 | Tests | Zero CLI command tests despite 15+ commands; only protocols 1 and 9 tested of 12 | `tests/` | N/A |
| L-07 | Tests | No unit tests for error_handler, crash_recovery, timeout_handler, preprocessing_pipelines, path_security, performance_dashboard, static_dashboard_generator, progress_estimator | `tests/` | N/A |
| L-08 | Documentation | GUI User Guide documents Ctrl+N, Ctrl+O, F5 shortcuts that don't exist in code | `docs/GUI-User-Guide.md` | 420-433 |
| L-09 | Documentation | Tests-GUI, Utils-GUI, Falsification-GUI not mentioned in User Guide | `docs/GUI-User-Guide.md` | N/A |
| L-10 | Documentation | Documented "Psychological States GUI" and "Web-Based Analysis Interface" do not exist | `docs/GUI-User-Guide.md` | N/A |
| L-11 | GUI | Hardcoded font sizes (`Arial 10/16`) across all GUIs; no system font preferences | All GUIs | Various |
| L-12 | GUI | No DPI/scaling awareness in any GUI | All GUIs | Various |
| L-13 | Theme Manager | `themes["normal"] = themes["default"]` creates shared mutable reference | `utils/theme_manager.py` | 39 |
| L-14 | Theme Manager | `themes_dir` parameter accepted but never used (themes hardcoded) | `utils/theme_manager.py` | 17-19 |
| L-15 | Dependencies | Wide version ranges allow known-vulnerable packages -- SQLAlchemy >=1.4.0, PyTorch >=1.9.0, Jupyter >=1.0.0 | `requirements.txt` | N/A |
| L-16 | Crash Recovery | `crash_recovery_decorator` missing `@functools.wraps(func)` -- loses function metadata | `utils/crash_recovery.py` | 277 |
| L-17 | Crash Recovery | `set_recovery_callback` is dead code; `_recovery_callback` never read | `utils/crash_recovery.py` | 248 |
| L-18 | Cache Manager | `_evict_if_needed()` requires caller to hold lock but this isn't documented | `utils/cache_manager.py` | 142 |
| L-19 | Utils GUI | `quit()` then `destroy()` is redundant; can raise errors | `Utils-GUI.py` | 691-692 |

---

## Missing Features & Incomplete Implementations

Cross-referenced with documentation and project scope:

| ID | Feature | Status | Reference |
|----|---------|--------|-----------|
| MF-01 | **Configuration update system** | Non-functional -- `config --set` silently discards changes | `main.py:2626-2635` |
| MF-02 | **tACS causal intervention** | Declared in CLI options but not implemented | `main.py:2818-2825` |
| MF-03 | **Modality selection for multimodal analysis** | `--modalities` option accepted but ignored | `main.py:1066-1072` |
| MF-04 | **Falsification protocol cancellation** | Stop button exists but `stop_protocol()` is empty | `Falsification/...GUI.py:916` |
| MF-05 | **Falsification GUI window close handling** | No `WM_DELETE_WINDOW` handler | `Falsification/...GUI.py` |
| MF-06 | **Validation protocols 9-12** | Files exist but listed in `protocol_files` only up to Protocol 8 | `Validation/APGI_Validation_GUI.py:67-76` |
| MF-07 | **Keyboard shortcuts Ctrl+N, Ctrl+O, F5** | Documented in User Guide but not implemented | `docs/GUI-User-Guide.md:420-433` |
| MF-08 | **Psychological States GUI** | Referenced in documentation but does not exist as standalone application | `docs/GUI-User-Guide.md` |
| MF-09 | **Web-Based Analysis Interface** | Referenced in documentation but does not exist | `docs/GUI-User-Guide.md` |
| MF-10 | **Protocol execution timeouts** | `timeout_seconds` defined in Master Validator but never applied | `Validation/Master_Validation.py:46` |
| MF-11 | **Custom theme loading** | `themes_dir` parameter accepted but themes are hardcoded | `utils/theme_manager.py:17-19` |
| MF-12 | **Backup encryption** | HMAC integrity verification exists but data is stored in plaintext | `utils/backup_manager.py` |

---

## Security Vulnerability Summary

| ID | Severity | Category | Description | File |
|----|----------|----------|-------------|------|
| S-01 | Medium | Info Leakage | `str(e)` rendered in dashboard HTML (9 locations) -- exposes internal paths, configs | `utils/performance_dashboard.py` |
| S-02 | Medium | Path Traversal | `batch_processor.py` opens `input_file`/writes `output_file` from job params without path validation | `utils/batch_processor.py:450,493` |
| S-03 | Medium | Path Traversal | 8 CLI commands write to user-provided paths with zero validation | `main.py` |
| S-04 | Low | Info Leakage | `traceback.format_exc()` / `traceback.print_exc()` exposed in 6+ locations | `main.py`, `APGI-Equations.py`, etc. |
| S-05 | Low | Hardcoded Secret | Module-level `BACKUP_HMAC_KEY` falls back to `"default_backup_key_for_testing_32_chars"` | `utils/backup_manager.py:32` |
| S-06 | Low | Dependencies | Wide version ranges allow known-vulnerable SQLAlchemy 1.4.x, PyTorch 1.9.x, Jupyter 1.0.x | `requirements.txt` |
| S-07 | Low | Deserialization | `pickle.loads()` used with HMAC verification -- if key compromised, enables arbitrary code execution. No `RestrictedUnpickler` | `utils/batch_processor.py:157` |
| S-08 | Info | Portability | `SIGALRM`-based ReDoS protection fails on Windows (no fallback) and in worker threads | `utils/input_validation.py:422` |

---

## Actionable Recommendations

### Priority 1: Critical Fixes (Estimated effort: 1-2 days)

| # | Action | Owner | Effort | Files |
|---|--------|-------|--------|-------|
| 1 | **Fix entropy validation math** -- compare per-byte Shannon entropy (0-8 range) against per-byte threshold, not total bits against total threshold. Alternatively, check byte uniqueness ratio. | Core Team | 30 min | `utils/batch_processor.py:64-81` |
| 2 | **Fix 4 broken CLI module paths** -- correct file references to use actual hyphenated names with `secure_load_module()`. | Core Team | 30 min | `main.py:2978,3060,3134,3363` |
| 3 | **Fix cross-species import** and **parameter_validator import** paths. | Core Team | 15 min | `main.py:421,2066` |
| 4 | **Fix `PrereregistrationTemplate` typo** to `PreregistrationTemplate`. | Core Team | 5 min | `main.py:2991` |
| 5 | **Add `return []` to `_load_history()`** when history file doesn't exist. | Core Team | 5 min | `utils/backup_manager.py:132-178` |
| 6 | **Fix undefined `backup_id`** variable in `_restore_zip_backup()`. | Core Team | 10 min | `utils/backup_manager.py:568` |
| 7 | **Add `RUNTIME` to `ErrorCategory` enum** or change `critical_error()` to use an existing category. | Core Team | 5 min | `utils/error_handler.py:40-55,559` |
| 8 | **Initialize `selected_protocol` and `parameter_values`** in Falsification GUI `__init__`. | GUI Team | 15 min | `Falsification/...GUI.py` |
| 9 | **Fix `setup_environment.py` PROJECT_ROOT** to use `Path(__file__).parent`. | Core Team | 5 min | `setup_environment.py:15` |

### Priority 2: High-Severity Fixes (Estimated effort: 3-5 days)

| # | Action | Owner | Effort | Files |
|---|--------|-------|--------|-------|
| 10 | **Uncomment `config --set` implementation** or implement proper config persistence. | Core Team | 2 hrs | `main.py:2626-2635` |
| 11 | **Implement `tacs` intervention handler** or remove from CLI choices. | Science Team | 2 hrs | `main.py:2818-2825` |
| 12 | **Fix timeout handler** to actually terminate timed-out threads (use `multiprocessing` or `ctypes.pythonapi.PyThreadState_SetAsyncExc`). | Core Team | 4 hrs | `utils/timeout_handler.py:160-221` |
| 13 | **Fix backup checksum verification** to compare same data type (archive-to-archive or content-to-content). | Core Team | 2 hrs | `utils/backup_manager.py:209,601` |
| 14 | **Change Ctrl+C binding** to Ctrl+L or similar for `clear_output()`. | GUI Team | 15 min | `Validation/APGI_Validation_GUI.py:105` |
| 15 | **Move `_ensure_ui_consistency()` to GUI thread** via `root.after()` or the update queue. | GUI Team | 1 hr | `Validation/APGI_Validation_GUI.py:1473` |
| 16 | **Add `WM_DELETE_WINDOW` handler** to Falsification GUI with thread cleanup. | GUI Team | 1 hr | `Falsification/...GUI.py` |
| 17 | **Move `messagebox.showerror()`** to main thread via queue in Falsification GUI. | GUI Team | 1 hr | `Falsification/...GUI.py:997` |
| 18 | **Implement `stop_protocol()`** with `threading.Event` cancellation. | GUI Team | 2 hrs | `Falsification/...GUI.py:916-917` |
| 19 | **Add path validation** to all CLI commands that write output files. | Core Team | 2 hrs | `main.py` (8 commands) |
| 20 | **Replace raw `str(e)` in dashboard HTML** with generic error messages; log details server-side. | Core Team | 1 hr | `utils/performance_dashboard.py` |

### Priority 3: Medium/Low Improvements (Estimated effort: 1-2 weeks)

| # | Action | Owner | Effort | Files |
|---|--------|-------|--------|-------|
| 21 | **Add CLI command tests** using `click.testing.CliRunner` for all 15+ commands. | QA Team | 3 days | `tests/` |
| 22 | **Replace 24 `except: pass` blocks** with specific exception types and logging. | All Teams | 1 day | Multiple |
| 23 | **Implement lazy module loading** -- defer scientific module imports until command execution. | Core Team | 4 hrs | `main.py:261` |
| 24 | **Cap unbounded lists** in performance dashboard; add ring buffer or maxlen. | Core Team | 1 hr | `utils/performance_dashboard.py:59-60` |
| 25 | **Add thread locks** to crash recovery state file writes. | Core Team | 1 hr | `utils/crash_recovery.py` |
| 26 | **Narrow dependency version ranges** and run `pip-audit` / `safety` in CI. | DevOps | 2 hrs | `requirements.txt` |
| 27 | **Add missing tests** for error_handler, crash_recovery, timeout_handler, path_security. | QA Team | 3 days | `tests/` |
| 28 | **Update GUI User Guide** to reflect actual keyboard shortcuts and available GUIs. | Docs Team | 2 hrs | `docs/GUI-User-Guide.md` |
| 29 | **Add DPI/scaling awareness** to all GUI applications. | GUI Team | 4 hrs | All GUIs |
| 30 | **Implement validation protocols 9-12** in GUI protocol list. | GUI Team | 1 hr | `Validation/APGI_Validation_GUI.py:67-76` |

---

## Test Suite Analysis

### Current State: COMPLETE FAILURE

```
$ python -m pytest tests/ --tb=short -q
76 errors in 1.85s
```

**Root Cause:** `utils/batch_processor.py` line 87 calls `_validate_secret_key()` at module import time. The Shannon entropy calculation (line 69-76) computes per-symbol entropy (max ~4 bits for hex characters, ~8 bits for binary) but compares against `len(key_bytes) * 7` total bits. For a 64-character hex key: max possible = ~4 * 64 = 256 bits; threshold = 64 * 7 = 448 bits. **The threshold is mathematically impossible to meet with hex-encoded keys.**

This is triggered via the import chain: `tests/*.py` -> `utils/__init__.py` -> `utils/batch_processor.py`.

### Fix Required:
```python
# Current (broken):
min_entropy_bits = len(key_bytes) * 7  # Total bits, impossible threshold
if entropy < min_entropy_bits:

# Correct approach:
min_entropy_per_byte = 3.0  # bits per byte, ~37.5% of maximum
if entropy < min_entropy_per_byte:
```

### Test Coverage Gaps (after fix):
- 0/15+ CLI commands tested
- 2/12 validation protocols tested
- 1/6 falsification protocols tested
- 0/20 utility modules have dedicated tests
- No integration tests for GUI + protocol interaction
- No security-focused tests (path traversal, input validation)

---

## Appendix A: Files Audited

| Category | Count | Key Files |
|----------|-------|-----------|
| Core Modules | 15 | `main.py`, `APGI-Equations.py`, `APGI-Multimodal-Integration.py`, etc. |
| Validation | 14 | `Master_Validation.py`, `APGI_Validation_GUI.py`, Protocols 1-12 |
| Falsification | 8 | `APGI-Falsification-Protocol-GUI.py`, Protocols 1-6 |
| Utilities | 20 | `config_manager.py`, `error_handler.py`, `backup_manager.py`, etc. |
| Tests | 8 | `test_basic.py` through `test_utils.py`, `conftest.py` |
| GUIs | 4 | `APGI_Validation_GUI.py`, `APGI-Falsification-Protocol-GUI.py`, `Tests-GUI.py`, `Utils-GUI.py` |
| Config | 7 | `default.yaml`, `gui_config.yaml`, profiles, etc. |
| Documentation | 30+ | Architecture, equations, protocols, user guides |

## Appendix B: Environment Details

| Property | Value |
|----------|-------|
| Python Version | 3.11+ |
| Framework Type | Scientific Computing / CLI + GUI |
| GUI Framework | tkinter |
| Key Dependencies | numpy, scipy, pandas, torch, PyMC, click, rich |
| LOC (Python) | ~35,000+ |
| Total Files | 76 Python + 30+ docs + 7 config |

---

*Report generated 2026-03-10. All line numbers verified against current codebase.*
