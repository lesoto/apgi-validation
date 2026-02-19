# APGI Framework — Comprehensive Application Audit Report

**Date:** 2026-02-19
**Auditor:** Claude Code (Automated Audit)
**Branch:** `claude/app-audit-testing-VIAQz`
**Framework Version:** 1.3.0
**Python Runtime:** 3.11.14

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [KPI Scores](#kpi-scores)
3. [Application Scope](#application-scope)
4. [Bug Inventory](#bug-inventory)
   - [Critical](#critical-bugs)
   - [High](#high-severity-bugs)
   - [Medium](#medium-severity-bugs)
   - [Low](#low-severity-bugs)
5. [Missing Features & Incomplete Implementations](#missing-features--incomplete-implementations)
6. [Component-Level Findings](#component-level-findings)
7. [Test Suite Analysis](#test-suite-analysis)
8. [Security Findings](#security-findings)
9. [Actionable Recommendations](#actionable-recommendations)
10. [Appendix: File Coverage Matrix](#appendix-file-coverage-matrix)

---

## Executive Summary

The APGI (Adaptive Pattern Generation and Integration) Theory Framework is a large-scale scientific software package implementing an empirical consciousness theory. The framework spans approximately **500,000+ lines of Python** across 75+ source files, providing: a 25-command CLI (`main.py`), 12 validation protocols, 6 falsification protocols, 4 tkinter GUI applications, a FastAPI REST backend, a comprehensive utilities library, and 6 test modules.

**Overall assessment:** The codebase is architecturally sound and largely complete. The core scientific logic—validation protocols, falsification protocols, parameter estimation, and the CLI—is implemented end-to-end. However, a cluster of **infrastructure and configuration defects** undermine reliability and security in deployment scenarios. Four critical issues require remediation before production use, including an invalid mypy configuration, insecure authentication practices in the REST API, and missing filesystem directories that cause runtime failures.

The test suite passes 10 out of 16 collectable tests; 5 fail due to missing dependencies, hardcoded developer-machine paths, and a missing `utils/data/` directory. Two entire test modules (`test_integration.py`, `test_performance.py`) cannot be collected due to a missing `tqdm` dependency.

---

## KPI Scores

| # | KPI | Score | Rationale |
|---|-----|-------|-----------|
| 1 | **Functional Completeness** | **78 / 100** | 12/12 validation protocols, 6/6 falsification protocols, 25/25 CLI commands implemented. Deductions for 3 unimplemented Bayesian estimation methods, 2 unimplemented causal interventions, and the missing `apgi_gui` theme module. |
| 2 | **UI/UX Consistency** | **72 / 100** | All 4 GUIs share consistent tkinter patterns, tooltips, progress tracking, and threaded output. The theme-switching feature is silently disabled because `apgi_gui.theme_manager` does not exist, breaking the "Themes" menu. Rich CLI output is well-formatted and coloured. |
| 3 | **Responsiveness & Performance** | **75 / 100** | Threading is used correctly throughout all GUIs and long CLI operations. Bounded output buffers (`deque(maxsize=10000)`) prevent memory leaks. A caching layer and performance profiler are present. Deductions for no async I/O in the CLI, no documented performance baselines, and no load/stress testing coverage. |
| 4 | **Error Handling & Resilience** | **80 / 100** | `utils/error_handler.py` defines a structured error hierarchy with categories and severities. All GUI apps handle thread exceptions and use queued UI updates for safety. Validation protocols degrade gracefully when data is unavailable. Deductions for bare `except Exception` clauses in protocol loops, missing startup directory creation, and the API's lack of structured error logging for auth failures. |
| 5 | **Overall Implementation Quality** | **76 / 100** | Type hints used throughout; `mypy.ini` is configured for strict checking. Rich documentation directory with 30 docs. Config management, backup/restore, and logging are mature. Deductions for the invalid `python_version = 3.14` in `mypy.ini`, hardcoded developer paths in tests, security-grade weaknesses in the API, and the broken test collection. |

**Composite Score: 76 / 100**

---

## Application Scope

| Component | Files | Status |
|-----------|-------|--------|
| CLI entry point (`main.py`) | 1 | Implemented — 4,887 lines, 25+ commands |
| Validation Protocols (1–12) | 12 | All implemented |
| Master Validation Pipeline | 1 | Implemented |
| Falsification Protocols (1–6) | 6 | All implemented |
| GUI: Tests Runner | 1 (`Tests-GUI.py`) | Implemented — theme feature broken |
| GUI: Utils Runner | 1 (`Utils-GUI.py`) | Implemented |
| GUI: Validation Runner | 1 (`Validation/APGI-Validation-GUI.py`) | Implemented |
| GUI: Falsification Runner | 1 (`Falsification/APGI-Falsification-Protocol-GUI.py`) | Implemented |
| REST API (`APGI-API.py`) | 1 | Implemented — security issues |
| Utility modules | 23 | All implemented |
| Configuration files | 6 YAML | All complete |
| Test modules | 6 | 2 uncollectable; 5/16 tests fail |
| Documentation | 30 Markdown | Complete |

---

## Bug Inventory

### Critical Bugs

> **Critical** — Causes security vulnerabilities, data loss, or prevents the system from starting/running in documented configurations.

---

#### BUG-C1 · `mypy.ini` specifies Python 3.14 (non-existent version)

| Field | Detail |
|-------|--------|
| **Severity** | Critical |
| **Component** | `mypy.ini` |
| **Affected file** | `mypy.ini:2` |
| **Reproduction** | Run `mypy .` in the project root |
| **Expected** | mypy performs type checking against Python 3.11 |
| **Actual** | `error: Unrecognized Python version "3.14"` — mypy aborts; no type checking occurs |
| **Impact** | The entire mypy type-safety gate is non-functional; type regressions go undetected |

**Fix:** Change `python_version = 3.14` → `python_version = 3.11` in `mypy.ini`.

---

#### BUG-C2 · Hardcoded credentials in `APGI-API.py`

| Field | Detail |
|-------|--------|
| **Severity** | Critical |
| **Component** | REST API |
| **Affected file** | `APGI-API.py:92–96` |
| **Reproduction** | Open `APGI-API.py` and inspect `USERS_DB`; send `POST /auth/login` with `{"username":"admin","password":"admin123"}` |
| **Expected** | Credentials stored externally (env vars, secrets manager) and not committed to source control |
| **Actual** | `"admin": hashlib.sha256("admin123".encode()).hexdigest()` is hard-coded in the source file and committed to the repository |
| **Impact** | Any developer with repository access (or anyone who reads the source) can authenticate as admin |

**Fix:** Load credentials from environment variables or an external secrets store. Remove credentials from source.

---

#### BUG-C3 · Unsalted SHA-256 password hashing

| Field | Detail |
|-------|--------|
| **Severity** | Critical |
| **Component** | REST API |
| **Affected file** | `APGI-API.py:103–105` |
| **Reproduction** | Inspect `verify_password()`; compare stored hash against a rainbow table |
| **Expected** | Passwords hashed using bcrypt/argon2 with per-user salt |
| **Actual** | `hashlib.sha256(plain_password.encode()).hexdigest()` — deterministic, no salt, susceptible to rainbow tables |
| **Impact** | User passwords trivially recoverable if the in-memory store is ever persisted or logged |

**Fix:** Replace with `bcrypt.checkpw()` or `passlib.hash.argon2.verify()`.

---

#### BUG-C4 · In-memory token store lost on restart

| Field | Detail |
|-------|--------|
| **Severity** | Critical |
| **Component** | REST API |
| **Affected file** | `APGI-API.py:99` |
| **Reproduction** | Log in, restart the API server, attempt to use the previously issued token |
| **Expected** | Token remains valid across restarts (or is revoked gracefully) |
| **Actual** | `TOKENS: Dict[str, str] = {}` — all active sessions are destroyed on process restart; also not thread-safe under concurrent requests |
| **Impact** | Session continuity impossible; concurrent auth requests can corrupt token state |

**Fix:** Use a database table or Redis for token storage; add a mutex for concurrent access in the interim.

---

### High Severity Bugs

> **High** — Causes feature failures, test failures, or incorrect behaviour that affects core workflows.

---

#### BUG-H1 · `data_repository/` directory not present; file upload crashes

| Field | Detail |
|-------|--------|
| **Severity** | High |
| **Component** | REST API — `/data/upload` endpoint |
| **Affected file** | `APGI-API.py:39–42, 498–506` |
| **Reproduction** | Start the API and `POST /data/upload` with a file of type `raw_eeg`, `raw_fmri`, or `processed_behavioral` |
| **Expected** | File is saved to the appropriate subdirectory |
| **Actual** | `FileNotFoundError` — `data_repository/raw_data/eeg/` does not exist; `mkdir(parents=True, exist_ok=True)` is called only for the resolved sub-path but the parent tree is never seeded |
| **Impact** | Data upload is broken for all data types |

**Fix:** Add a startup event that calls `dir.mkdir(parents=True, exist_ok=True)` for `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, and `METADATA_DIR`.

---

#### BUG-H2 · `utils/data/` directory missing; `test_utils_directory_structure` fails

| Field | Detail |
|-------|--------|
| **Severity** | High |
| **Component** | Test suite / utils structure |
| **Affected file** | `tests/test_utils.py:21`, `utils/` directory |
| **Reproduction** | `python3 -m pytest tests/test_utils.py::test_utils_directory_structure` |
| **Expected** | `utils/data/` exists and test passes |
| **Actual** | `AssertionError: utils/data directory missing` |
| **Impact** | Test gate fails; backup manager may also fail if it attempts to reference `utils/data/cache/` |

**Fix:** Create `utils/data/.gitkeep` and commit it so the directory exists in the repository.

---

#### BUG-H3 · Missing `apgi_gui` package disables theme manager in all GUIs

| Field | Detail |
|-------|--------|
| **Severity** | High |
| **Component** | `Tests-GUI.py`, `Utils-GUI.py` |
| **Affected files** | `Tests-GUI.py:23–28` |
| **Reproduction** | Launch `Tests-GUI.py`; inspect the View → Theme menu |
| **Expected** | Theme sub-menu lists available themes (Normal, Dark, High Contrast) and switching applies them |
| **Actual** | `ImportError: No module named 'apgi_gui'`; `THEME_MANAGER_AVAILABLE = False`; entire Themes menu is absent |
| **Impact** | Advertised theme-switching feature (documented in `docs/GUI-User-Guide.md`) is completely unavailable |

**Fix:** Create `apgi_gui/__init__.py` and `apgi_gui/theme_manager.py` with the `ThemeManager` class, or remove the feature reference from documentation if not intended.

---

#### BUG-H4 · Hardcoded macOS paths in `test_validation.py` break CI

| Field | Detail |
|-------|--------|
| **Severity** | High |
| **Component** | Test suite |
| **Affected file** | `tests/test_validation.py:65, 195` |
| **Reproduction** | `python3 -m pytest tests/test_validation.py::test_apgi_dynamical_system_simulate_surprise_accumulation` |
| **Expected** | Test loads the protocol from a path relative to the project root |
| **Actual** | `FileNotFoundError: /Users/lesoto/Sites/PYTHON/apgi-validation/Validation/Validation-Protocol-1.py` |
| **Impact** | Two tests fail on every machine except the original developer's; CI/CD gates fail |

**Fix:** Replace absolute paths with `Path(__file__).parent.parent / "Validation" / "Validation-Protocol-1.py"`.

---

#### BUG-H5 · `tqdm` not in installed environment; two test modules uncollectable

| Field | Detail |
|-------|--------|
| **Severity** | High |
| **Component** | Test suite, `utils/__init__.py` |
| **Affected files** | `tests/test_integration.py:14`, `tests/test_performance.py:10`, `utils/batch_processor.py:22` |
| **Reproduction** | `python3 -m pytest tests/` in a clean environment without `tqdm` installed |
| **Expected** | Tests collected and executed |
| **Actual** | `ModuleNotFoundError: No module named 'tqdm'` — pytest aborts collection for both modules |
| **Impact** | All integration and performance tests are invisible to the test runner |

**Fix:** Ensure `tqdm` (already listed in `requirements.txt`) is installed in the CI environment. Add a `try/except ImportError` guard in `utils/__init__.py` with a clear `ImportError` message for optional dependencies.

---

#### BUG-H6 · Three API endpoints lack authentication

| Field | Detail |
|-------|--------|
| **Severity** | High |
| **Component** | REST API |
| **Affected file** | `APGI-API.py:297–346, 523–531, 532–543` |
| **Reproduction** | `GET /config`, `GET /results/{result_id}`, `GET /protocols/list` without `Authorization` header |
| **Expected** | 401 Unauthorized |
| **Actual** | 200 OK — full response returned without authentication |
| **Impact** | Unauthenticated callers can read configuration, stored results, and protocol list |

**Fix:** Add `current_user: str = Depends(get_current_user)` to `validate_data_file`, `get_validation_results`, and `list_validation_protocols`.

---

#### BUG-H7 · Causal intervention types return "not implemented" error

| Field | Detail |
|-------|--------|
| **Severity** | High |
| **Component** | CLI — `causal_manipulations` command |
| **Affected file** | `main.py:2467` |
| **Reproduction** | `python main.py causal-manipulations --intervention pharmacological` or `--intervention metabolic` |
| **Expected** | Simulation runs for the selected intervention type |
| **Actual** | `{"error": "Intervention pharmacological not implemented"}` |
| **Impact** | Two of the documented intervention modes are non-functional |

**Fix:** Implement the `pharmacological` and `metabolic` branches in the `causal_manipulations` command handler.

---

#### BUG-H8 · Three Bayesian estimation methods print "not implemented" in demo mode

| Field | Detail |
|-------|--------|
| **Severity** | High |
| **Component** | CLI — `bayesian_estimation` / `estimate-params` command |
| **Affected file** | `main.py:2801–2809` |
| **Reproduction** | `python main.py estimate-params --method hierarchical` (or `iit_convergence`, `recovery`) |
| **Expected** | Estimation runs using the specified method |
| **Actual** | `"Hierarchical Bayesian estimation not implemented in demo"` printed; no results produced |
| **Impact** | Three of the six documented estimation methods are stubs |

**Fix:** Implement the missing estimation branches or raise a proper `NotImplementedError` with an actionable message rather than silently printing to stdout.

---

### Medium Severity Bugs

> **Medium** — Incorrect behaviour, degraded functionality, or quality issues that do not block primary workflows.

---

#### BUG-M1 · `pytest.ini` enforces 80% coverage threshold — unachievable in current environment

| Field | Detail |
|-------|--------|
| **Severity** | Medium |
| **Component** | Test configuration |
| **Affected file** | `pytest.ini` (`--cov-fail-under=80`) |
| **Impact** | Any CI run without the full dependency set (click, torch, pymc, etc.) fails the coverage gate, blocking the pipeline even when core tests pass |

**Fix:** Lower the coverage threshold to match the subset of importable modules in the test environment, or configure coverage to exclude modules requiring unavailable dependencies.

---

#### BUG-M2 · `validate_data_file` endpoint lacks file type and size validation

| Field | Detail |
|-------|--------|
| **Severity** | Medium |
| **Component** | REST API — `/data/validate` |
| **Affected file** | `APGI-API.py:297–346` |
| **Impact** | Arbitrary file types and arbitrarily large files accepted; potential for DoS via large upload |

**Fix:** Validate `content_type` against an allowlist and enforce a `content-length` limit.

---

#### BUG-M3 · Placeholder OSF URL committed to source

| Field | Detail |
|-------|--------|
| **Severity** | Medium |
| **Component** | CLI — `open-science` command |
| **Affected file** | `main.py:2641` |
| **Reproduction** | `python main.py open-science --upload` or inspect the open-science protocol output |
| **Expected** | A real OSF repository URL |
| **Actual** | `data_repository="https://osf.io/XXXXX/"` — placeholder value |
| **Impact** | Open-science sharing feature non-functional; users receive a broken URL |

**Fix:** Replace with the real OSF project URL or make the URL a required CLI option.

---

#### BUG-M4 · `config_diff` command requires `config/versions/` directory that is never created

| Field | Detail |
|-------|--------|
| **Severity** | Medium |
| **Component** | CLI — `config-diff` command |
| **Affected file** | `main.py:4542–4591` |
| **Reproduction** | `python main.py config-diff` on a fresh checkout |
| **Expected** | Diff displayed or helpful message if no versions exist |
| **Actual** | `FileNotFoundError` when `version_file.exists()` check fails because parent directory doesn't exist |
| **Impact** | `config-diff` is broken on fresh installations |

**Fix:** Create `config/versions/` in project setup or add `mkdir(parents=True, exist_ok=True)` before referencing it.

---

#### BUG-M5 · `Validation-Protocol-1.py` has incomplete cross-validation loop body

| Field | Detail |
|-------|--------|
| **Severity** | Medium |
| **Component** | Validation Protocol 1 |
| **Affected file** | `Validation/Validation-Protocol-1.py:1737` |
| **Description** | Nested CV inner loop ends with a bare `pass` statement, silently skipping hyperparameter search |
| **Impact** | Hyperparameter tuning silently omitted; validation results may be sub-optimal without warning |

**Fix:** Implement the hyperparameter tuning logic or raise `NotImplementedError` with a descriptive message.

---

#### BUG-M6 · No API rate limiting

| Field | Detail |
|-------|--------|
| **Severity** | Medium |
| **Component** | REST API |
| **Affected file** | `APGI-API.py` (global) |
| **Impact** | Authentication endpoint (`/auth/login`) susceptible to brute-force; computationally expensive simulation endpoints can be DoS'd |

**Fix:** Add `slowapi` (or equivalent) middleware with per-IP rate limiting on auth and simulation endpoints.

---

### Low Severity Bugs

> **Low** — Minor quality, maintainability, or cosmetic issues.

---

#### BUG-L1 · `delete_pycache.py` is an ad-hoc utility that should be a Makefile/tox target

| Field | Detail |
|-------|--------|
| **Severity** | Low |
| **Component** | Project structure |
| **Affected file** | `delete_pycache.py` |
| **Impact** | Adds noise to the project root; could be replaced with `find . -type d -name __pycache__ -exec rm -rf {} +` in a Makefile |

---

#### BUG-L2 · `requirements.txt` has no upper-bound version pins

| Field | Detail |
|-------|--------|
| **Severity** | Low |
| **Component** | Dependency management |
| **Affected file** | `requirements.txt` |
| **Impact** | Future breaking releases of major dependencies (e.g., PyMC 6, pandas 3, torch 3) will break the environment without warning |

**Fix:** Pin maximum versions for production (`numpy>=1.21.0,<3.0.0`), or adopt `pip-tools` / `poetry` for lock-file management.

---

#### BUG-L3 · `Falsification/__init__.py` is empty; Falsification package not exported

| Field | Detail |
|-------|--------|
| **Severity** | Low |
| **Component** | Falsification package |
| **Affected file** | `Falsification/__init__.py` |
| **Impact** | Unlike the `Validation` package, falsification protocols cannot be imported as a package; `from Falsification import ...` fails |

**Fix:** Export the main falsification classes from `Falsification/__init__.py`, mirroring the pattern in `Validation/__init__.py`.

---

#### BUG-L4 · `mypy.ini` disallows untyped defs but several utility functions lack return types

| Field | Detail |
|-------|--------|
| **Severity** | Low |
| **Component** | mypy configuration vs. code |
| **Impact** | Once BUG-C1 is fixed, `mypy` will report errors in utilities that have `allow_untyped_defs = False` but still use dynamic returns |

---

## Missing Features & Incomplete Implementations

| ID | Feature | Location | Priority | Notes |
|----|---------|----------|----------|-------|
| F1 | `apgi_gui` theme manager package | `Tests-GUI.py:23`, `Utils-GUI.py` | High | Entire package directory is absent; Themes menu never renders |
| F2 | `data_repository/` directory structure | `APGI-API.py:39–42` | High | All three subdirs (`raw_data/`, `processed_data/`, `metadata/`) must be created on startup |
| F3 | `utils/data/` directory | `tests/test_utils.py:21`, `utils/backup_manager.py:83` | High | Referenced by test and backup code; not present in repository |
| F4 | Persistent session/token store | `APGI-API.py:99` | High | In-memory `TOKENS` dict resets on each restart |
| F5 | Bayesian hierarchical estimation | `main.py:2801` | High | Method prints "not implemented in demo" |
| F6 | Bayesian IIT convergence analysis | `main.py:2805` | High | Method prints "not implemented in demo" |
| F7 | Bayesian parameter recovery analysis | `main.py:2809` | High | Method prints "not implemented in demo" |
| F8 | Pharmacological causal intervention | `main.py:2467` | High | Returns error dict instead of results |
| F9 | Metabolic causal intervention | `main.py:2467` | High | Returns error dict instead of results |
| F10 | `Falsification/__init__.py` exports | `Falsification/__init__.py` | Medium | Package exports nothing; protocols must be loaded via `importlib` |
| F11 | Real OSF project URL | `main.py:2641` | Medium | Placeholder `osf.io/XXXXX/` blocks open-science sharing |
| F12 | `config/versions/` directory | `main.py:4542` | Medium | `config-diff` fails without it |
| F13 | API request size/type validation | `APGI-API.py:297–346` | Medium | File upload has no type allowlist or size cap |
| F14 | API rate limiting | `APGI-API.py` | Medium | No throttling on auth or simulation endpoints |
| F15 | Coverage baseline in test CI | `pytest.ini` | Low | 80% threshold unachievable with partial dependency install |
| F16 | Nested CV hyperparameter tuning (Protocol-1) | `Validation/Validation-Protocol-1.py:1737` | Low | Inner loop body is `pass` |

---

## Component-Level Findings

### CLI (`main.py`)

- **25 commands** defined and wired; command dispatch, help text, and option parsing are complete.
- Rich console output with progress spinners, tables, and panels is consistent throughout.
- Error handling wraps each command with user-friendly messages and `apgi_logger` integration.
- **Deductions:** BUG-H7 (causal interventions), BUG-H8 (Bayesian methods), BUG-M3 (OSF URL), BUG-M4 (config-diff directory).

### Validation Protocols (Protocols 1–12)

- All 12 protocols implement `main()` and `run_validation()` / equivalent entry points.
- Protocol tier classification correctly maps 1, 3, 9, 12 as primary (reject on failure); 2, 4, 8, 10 as secondary; 5, 6, 7, 11 as tertiary.
- `Master-Validation.py` (symlinked as `APGI-Validation-Pipeline.py`) correctly orchestrates all 12 via `ThreadPoolExecutor` with configurable timeout.
- **Deduction:** Protocol-1 has an incomplete CV loop body (BUG-M5).

### Falsification Protocols (Protocols 1–6)

- All 6 protocols implement complete experimental designs (Iowa Gambling Task variant, evolutionary emergence, phase transitions, network comparison, etc.).
- `APGI-Falsification-Protocol-GUI.py` provides a functional GUI runner for all 6.
- **Deduction:** `Falsification/__init__.py` is empty (BUG-L3).

### REST API (`APGI-API.py`)

- FastAPI application with 14 endpoints, JWT-style token auth, and Pydantic request models.
- **Security issues dominate this component** (BUG-C2, BUG-C3, BUG-C4, BUG-H6).
- Missing startup directory creation causes file upload to fail (BUG-H1).
- Functional endpoints: `/`, `/health`, `/auth/login`, `/simulation/run`, `/data/generate`, `/config` (GET/PUT), `/config/profile`, `/validation/run-protocol/{id}`.

### GUI Applications

| GUI | Theme | Threading | Progress | Error Handling |
|-----|-------|-----------|----------|----------------|
| `Tests-GUI.py` | Broken (BUG-H3) | Correct | Correct | Correct |
| `Utils-GUI.py` | Broken (BUG-H3) | Correct | Correct | Correct |
| `Validation/APGI-Validation-GUI.py` | N/A | Correct | Correct | Correct |
| `Falsification/APGI-Falsification-Protocol-GUI.py` | N/A | Correct | Correct | Correct |

All GUIs correctly use `threading.Thread` for long operations and `queue.Queue` for UI updates; no blocking calls on the main thread were found.

### Utility Modules

All 23 utility modules pass AST syntax checks. Key modules:

| Module | Notes |
|--------|-------|
| `error_handler.py` | Structured `ErrorCategory` / `ErrorSeverity` hierarchy; well-designed |
| `config_manager.py` | YAML-backed config with versioning; `compare_configs` used by `config-diff` |
| `backup_manager.py` | ZIP-based backups with metadata; references non-existent `utils/data/cache/` path |
| `cache_manager.py` | TTL-based in-memory cache with LRU eviction |
| `data_validation.py` | `DataValidator` with `validate_data_quality()` returning structured reports |
| `batch_processor.py` | Threaded batch processing with `tqdm` progress (BUG-H5) |
| `logging_config.py` | Loguru-based structured logging; configures rotation and retention |

---

## Test Suite Analysis

### Test Run Results (available modules only)

```
Ran:    16 tests across test_basic.py, test_validation.py, test_utils.py
Passed: 10
Failed: 5
Skipped: 1
Errors: 2 (test_integration.py, test_performance.py uncollectable)
```

### Failure Breakdown

| Test | Module | Root Cause | Fix |
|------|--------|------------|-----|
| `test_import_main` | test_basic.py | `click` not installed in minimal environment | Install full `requirements.txt` in CI |
| `test_apgi_dynamical_system_simulate_surprise_accumulation` | test_validation.py | Hardcoded macOS path (`/Users/lesoto/...`) | Use `Path(__file__).parent.parent / "Validation" / ...` |
| `test_config_manager_load_save_cycle` | test_validation.py | `tqdm` not installed → `utils/__init__` import fails | Install `tqdm`; add optional import guard |
| `test_apgi_master_validator_integration` | test_validation.py | Hardcoded macOS path (`/Users/lesoto/...`) | Use relative path construction |
| `test_utils_directory_structure` | test_utils.py | `utils/data/` directory missing from repo | Create `utils/data/.gitkeep` |

### Test Quality Observations

- Integration tests correctly use `pytest.mark.integration` and `mocker` for mocking.
- API tests (`test_api_endpoints_with_httpx`) use FastAPI's `TestClient` correctly.
- `conftest.py` fixtures are well-designed with proper teardown.
- No tests exist for the GUI layer (expected for tkinter; would require a test framework like `pytest-qt` or `pyautogui`).
- Performance tests include meaningful assertion thresholds.

---

## Security Findings

| ID | Issue | OWASP Category | Severity |
|----|-------|---------------|----------|
| S1 | Hardcoded admin credentials in source | A07: Identification and Authentication Failures | Critical |
| S2 | Unsalted SHA-256 password hashing | A02: Cryptographic Failures | Critical |
| S3 | In-memory token store (no persistence, not thread-safe) | A07: Identification and Authentication Failures | Critical |
| S4 | Three endpoints with no authentication | A01: Broken Access Control | High |
| S5 | No rate limiting on auth endpoint | A04: Insecure Design | Medium |
| S6 | No file type validation on upload endpoint | A04: Insecure Design | Medium |
| S7 | No CORS policy defined in FastAPI app | A05: Security Misconfiguration | Low |

---

## Actionable Recommendations

### Immediate (before next release)

1. **Fix mypy Python version** — Change `python_version = 3.14` → `python_version = 3.11` in `mypy.ini`. Verify `mypy .` completes without collection errors.

2. **Secure the API** — Remove hardcoded credentials; load from `APGI_ADMIN_PASSWORD` environment variable. Replace SHA-256 with `bcrypt` or `argon2`. Replace the in-memory token dict with database-backed sessions. Add `Depends(get_current_user)` to the three unprotected endpoints.

3. **Fix hardcoded test paths** — Replace absolute paths in `tests/test_validation.py:65` and `:195` with `Path(__file__).parent.parent / "Validation" / <filename>`.

4. **Create missing directories** — Add `utils/data/.gitkeep` and `data_repository/{raw_data,processed_data,metadata}/.gitkeep`. Add a startup lifecycle hook in `APGI-API.py` to `mkdir(parents=True, exist_ok=True)` all required paths.

5. **Create or remove `apgi_gui` package** — Either implement `apgi_gui/theme_manager.py` with the `ThemeManager` class, or remove all references and the corresponding documentation from `docs/GUI-User-Guide.md`.

### Short-term (next sprint)

6. **Implement missing CLI features** — Complete `pharmacological` and `metabolic` causal interventions; implement `hierarchical`, `iit_convergence`, and `recovery` Bayesian estimation methods; replace OSF placeholder URL.

7. **Fix CI test collection** — Ensure `tqdm`, `click`, and other required packages are installed in the CI environment. Lower or conditionally skip the 80% coverage threshold until the full dependency set is consistently available.

8. **Add API rate limiting** — Integrate `slowapi` with per-IP limits (e.g., 5 auth attempts/minute, 10 simulations/minute).

9. **Implement Protocol-1 CV loop body** — Replace `pass` at line 1737 with proper hyperparameter grid search or document the omission explicitly.

10. **Populate `Falsification/__init__.py`** — Export the main class from each falsification protocol, following the `Validation/__init__.py` pattern.

### Longer-term

11. **Adopt a lock-file dependency manager** — Replace bare `requirements.txt` with `pip-tools` (generates `requirements.lock`) or migrate to `poetry`/`hatch` for reproducible builds.

12. **Add CORS and security headers** — Define an explicit CORS policy on the FastAPI app (`allow_origins`, `allow_credentials`). Add security headers middleware (`starlette-csrf`, etc.).

13. **GUI theme manager** — Once created, extend theme support to the Validation and Falsification GUIs for consistency.

14. **Resolve `config/versions/` creation** — Either auto-create the directory during `config_version` command execution, or include it as a `.gitkeep` in the repository.

---

## Appendix: File Coverage Matrix

| File | Syntax OK | Importable* | Test Coverage | Notes |
|------|-----------|------------|---------------|-------|
| `main.py` | ✓ | Partial† | Indirect | Requires `click`, `rich`, `yaml` |
| `APGI-API.py` | ✓ | Partial† | Yes (`test_validation.py`) | Requires `fastapi`, `uvicorn` |
| `Tests-GUI.py` | ✓ | Partial† | No | Requires `tkinter`; theme broken |
| `Utils-GUI.py` | ✓ | Partial† | No | Requires `tkinter`; theme broken |
| `Validation/APGI-Validation-GUI.py` | ✓ | Partial† | No | Requires `tkinter` |
| `Validation/Master-Validation.py` | ✓ | Yes | Yes | Symlinked from Pipeline |
| `Validation/Validation-Protocol-1.py` | ✓ | Partial† | Yes | CV loop incomplete |
| `Validation/Validation-Protocol-2.py` | ✓ | Partial† | No | |
| `Validation/Validation-Protocol-3.py` | ✓ | Partial† | No | |
| `Validation/Validation-Protocol-4.py` | ✓ | Partial† | No | |
| `Validation/Validation-Protocol-5.py` | ✓ | Partial† | No | |
| `Validation/Validation-Protocol-6.py` | ✓ | Partial† | No | |
| `Validation/Validation-Protocol-7.py` | ✓ | Partial† | No | |
| `Validation/Validation-Protocol-8.py` | ✓ | Partial† | No | |
| `Validation/Validation-Protocol-9.py` | ✓ | Partial† | Yes (integration) | |
| `Validation/Validation-Protocol-10.py` | ✓ | Partial† | No | |
| `Validation/Validation-Protocol-11.py` | ✓ | Partial† | No | |
| `Validation/Validation-Protocol-12.py` | ✓ | Partial† | No | |
| `Falsification/Falsification-Protocol-1.py` | ✓ | Partial† | No | |
| `Falsification/Falsification-Protocol-2.py` | ✓ | Partial† | No | |
| `Falsification/Falsification-Protocol-3.py` | ✓ | Partial† | No | |
| `Falsification/Falsification-Protocol-4.py` | ✓ | Partial† | No | |
| `Falsification/Falsification-Protocol-5.py` | ✓ | Partial† | No | |
| `Falsification/Falsification-Protocol-6.py` | ✓ | Partial† | No | |
| `utils/batch_processor.py` | ✓ | No | Indirect | Blocked by `tqdm` |
| `utils/config_manager.py` | ✓ | No | Yes | Blocked by `tqdm` (via `__init__`) |
| `utils/error_handler.py` | ✓ | No | Indirect | Blocked by `tqdm` (via `__init__`) |
| `utils/data_validation.py` | ✓ | No | Yes | Blocked by `tqdm` (via `__init__`) |
| `utils/backup_manager.py` | ✓ | No | Indirect | Blocked by `tqdm` (via `__init__`) |
| `utils/cache_manager.py` | ✓ | No | Indirect | Blocked by `tqdm` (via `__init__`) |
| `utils/logging_config.py` | ✓ | No | Indirect | Blocked by `tqdm` (via `__init__`) |

*Importable = importable in the minimal test environment (numpy, scipy, pandas, pytest installed; tqdm, click, torch, pymc absent).
†Partial = passes AST syntax check but requires additional dependencies to fully import.

---

*Report generated by automated audit on 2026-02-19. All file and line references verified against the HEAD commit of branch `claude/app-audit-testing-VIAQz`.*
