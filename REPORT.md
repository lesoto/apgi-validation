# APGI Validation Framework — Comprehensive Audit & Coverage Report

**Date:** 2026-03-19
**Repository:** `apgi-validation`
**Branch:** `claude/audit-coverage-validation-VpA3v`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Security Audit](#2-security-audit)
3. [Test Coverage Analysis](#3-test-coverage-analysis)
4. [Recommendations](#4-recommendations)

---

## 1. Executive Summary

This report presents a comprehensive security audit and test coverage analysis of the APGI Validation Framework — a Python scientific computing framework for validating psychological and neurobiological theories.

### Key Findings

| Area | Status | Risk |
|------|--------|------|
| Input Validation | Excellent | Low |
| Path Traversal Protection | Excellent | Low |
| Secret Key Management | Excellent | Low |
| Hardcoded Credentials | None found | Low |
| Error Handling / Info Disclosure | Good | Low |
| Dangerous Function Usage | Safe / Controlled | Low |
| YAML Parsing | Safe (`safe_load` only) | Low |
| Subprocess Usage | Safe (list args, no `shell=True`) | Low |
| Regex ReDoS Protection | Good (timeout protection) | Low |
| Test Coverage (overall) | ~75-80% | Medium |
| CLI Command Coverage | ~47% of functions in `main.py` | High |
| Untested Utility Modules | 3 modules with zero coverage | High |
| Documentation Mismatch | CLAUDE.md describes non-existent FastAPI app | Medium |

**Overall Security Posture: GOOD** — The codebase demonstrates mature security practices for a CLI/research framework. No critical vulnerabilities found.

**Overall Test Coverage: NEEDS IMPROVEMENT** — Core scientific modules are well-tested, but CLI entry points and 3 utility modules lack coverage.

---

## 2. Security Audit

### 2.1 Input Validation

**Status:** EXCELLENT

**File:** `utils/input_validation.py` (542 LOC)

The `InputValidator` class provides comprehensive validation:
- Type validation (string, integer, float, boolean)
- Path validation with security checks and boundary enforcement
- File/directory existence and readability validation
- Email and URL validation with protocol restrictions
- Custom pattern matching with **regex timeout protection**:
  - Python 3.11+ uses built-in regex timeout parameter
  - Older versions use threading with 1-second timeout to prevent ReDoS

### 2.2 Path Traversal Protection

**Status:** EXCELLENT

Multiple layers of path traversal prevention:

| File | Mechanism |
|------|-----------|
| `utils/config_manager.py:60-102` | `_validate_file_path()` — rejects `..` sequences, resolves to canonical form, uses `Path.is_relative_to()` |
| `utils/batch_processor.py:285-327` | `_validate_secure_path()` — rejects `..`, uses `os.path.realpath()` to resolve symlinks, validates within project root |
| `main.py:75-105` | `secure_load_module()` — validates path within project root, restricts to `.py` files |

### 2.3 Secret / Key Management

**Status:** EXCELLENT with strong enforcement

**Environment Variable Enforcement** (`utils/__init__.py:24-54`):
- Required keys: `PICKLE_SECRET_KEY`, `APGI_BACKUP_HMAC_KEY`
- **Production mode** (`APGI_ENV=production`): raises `EnvironmentError` if missing
- **Development mode**: provides clear guidance for key generation via `secrets.token_hex(32)`

**Key Entropy Validation** (`utils/batch_processor.py:123-170`):
- Minimum 32 bytes (256 bits)
- Shannon entropy calculation
- Requires at least 192 bits for 32-byte keys (NIST-aligned)
- Thread-safe with `_keys_lock`

**HMAC-Based Integrity** (`utils/batch_processor.py:172-254`):
- SHA-256 HMAC for file signatures
- Timing-safe comparison via `hmac.compare_digest()`

**No hardcoded credentials found** — all secrets are environment-driven.

### 2.4 Dangerous Function Usage

| Function | Status | Details |
|----------|--------|---------|
| `eval()` / `exec()` | Safe | No direct usage; `spec.loader.exec_module()` is controlled with path validation |
| `subprocess` | Safe | List-based arguments only, no `shell=True` (`setup_environment.py`, `Tests-GUI.py`) |
| `pickle` | Not used | Framework uses HMAC-signed JSON instead |
| `yaml.load()` | Safe | All usage is `yaml.safe_load()` — never `yaml.load()`, `yaml.full_load()`, or `yaml.unsafe_load()` |

### 2.5 Error Handling & Information Disclosure

**Status:** GOOD

**Error Sanitization** (`utils/error_handler.py:125-150`):
- File paths redacted from tracebacks (`File "[path]"` → `File "[REDACTED]"`)
- Absolute paths replaced with `/[PATH]`
- Tracebacks truncated to 500 characters

**Secrets Redaction in Logs** (`utils/logging_config.py:42-84`):
- Comprehensive regex patterns for: API keys, tokens, passwords, JWTs, Bearer tokens, database URLs, email addresses, authorization headers
- `_sanitize_context()` recursively strips sensitive data from nested dicts with depth limit
- Control character removal to prevent log injection

### 2.6 Security Audit Logging

**File:** `utils/security_audit_logger.py` (500 LOC)
- Logs all file access with timestamps, operations, users, and success/failure status
- Maintains in-memory audit trail (last 1,000 entries)
- Note: lacks persistent storage — see recommendations

### 2.7 Web/API Security (Not Applicable)

The following categories are **not applicable** because APGI is a CLI/research framework, not a web API:

- Authentication / JWT — No login endpoints exist
- Authorization / RBAC — Single-user CLI tool
- CSRF Protection — No web sessions
- Rate Limiting — No network endpoints
- Cookie Security — No HTTP cookies
- CORS Configuration — No web server
- Middleware Stack — No HTTP middleware

> **Note:** The `CLAUDE.md` file describes a FastAPI web application with JWT auth, middleware, Celery workers, and PostgreSQL — none of which exist in the actual codebase. This documentation mismatch should be resolved.

---

## 3. Test Coverage Analysis

### 3.1 Overview

| Metric | Value |
|--------|-------|
| Total test files | 38 (excluding `__init__.py`, `conftest.py`) |
| Total source modules | ~50 (main + APGI modules + utils) |
| Coverage target (pytest.ini) | 80% |
| Estimated actual coverage | 75-80% |
| Hypothesis profiles | `dev` (10 examples), `ci` (100), `thorough` (1,000) |

### 3.2 Well-Tested Modules

| Module | Test File | Tests | Status |
|--------|-----------|-------|--------|
| `APGI_Equations.py` | `test_apgi_equations.py` | 28+ | Comprehensive |
| `APGI-Multimodal-Integration.py` | `test_apgi_multimodal_integration.py` | 50+ | Comprehensive |
| `APGI-Parameter-Estimation.py` | `test_apgi_parameter_estimation.py` | 50 | Comprehensive |
| `APGI-Entropy-Implementation.py` | `test_apgi_entropy_implementation.py` | ~35 | Good |
| `utils/input_validation.py` | `test_utils_modules.py` | Multiple | Good |
| `utils/batch_processor.py` | `test_utils_modules.py` | Multiple | Good |
| `utils/config_manager.py` | `test_utils_modules.py` | Multiple | Good |
| `utils/error_handler.py` | `test_error_handling.py` | Multiple | Good |
| Numerical boundaries | `test_numerical_boundaries.py` | Multiple | Good |

### 3.3 Critical Coverage Gaps

#### 3.3.1 Untested CLI Commands in `main.py` (18 functions)

`main.py` (5,718 LOC, 79 functions) has only 37 test functions — roughly 47% function coverage.

| Command | Tested? | Impact |
|---------|---------|--------|
| `validate()` | No | 17 validation protocols untested |
| `falsify()` | No | 17 falsification protocols untested |
| `estimate_params()` | No | Parameter estimation CLI untested |
| `cross_species()` | No | Cross-species analysis untested |
| `analyze_logs()` | No | Log parsing untested |
| `process_data()` | No | Data pipeline untested |
| `monitor_performance()` | No | Performance monitoring untested |
| `neural_signatures()` | No | — |
| `causal_manipulations()` | No | — |
| `quantitative_fits()` | No | — |
| `clinical_convergence()` | No | — |
| `open_science()` | No | — |
| `bayesian_estimation()` | No | — |
| `comprehensive_validation()` | No | — |
| `gui()` | No | GUI launch untested |
| `logs()` | No | Log display untested |
| `performance()` | No | Performance dashboard untested |
| `multimodal()` | Partial | Edge cases untested |

#### 3.3.2 Untested Utility Modules (3 modules, zero coverage)

| Module | LOC | Risk |
|--------|-----|------|
| `utils/dependency_scanner.py` | 413 | Subprocess execution, JSON parsing, timeout handling all untested |
| `utils/security_audit_logger.py` | 500 | Security logging functionality untested |
| `utils/security_logging_integration.py` | 148 | Security log integration untested |

#### 3.3.3 Untested Visualization Functions (9 functions)

- `_create_distribution_plot()`, `_create_figure_and_axes()`, `_create_heatmap_plot()`
- `_create_plot_by_type()`, `_create_scatter_plot()`, `_create_time_series_plot()`
- `_load_visualization_data()`, `_parse_visualization_parameters()`, `_setup_plotting_style()`

#### 3.3.4 Untested Data Processing & Protocol Execution

- `_process_csv_file()` — CSV processing
- `_run_demo_mode()` — Demo mode
- `_validate_input_file()` — Input file validation
- `_list_protocols()` — Protocol discovery
- `_run_parallel()` / `_run_sequential()` — Execution modes
- `_show_config()` / `_set_config()` / `_reset_config()` — Config management

### 3.4 Missing Error Condition Tests

| Condition | Status |
|-----------|--------|
| Out-of-memory scenarios | Not covered |
| Corrupted file formats | Not covered |
| Permission errors in file I/O | Partially covered |
| Network timeouts (dependency scanner) | Not covered |
| Malformed JSON configs | Partially covered |
| Signal interruption during long ops | Partially covered |
| NaN/Inf propagation through pipelines | Partially covered |
| GPU memory exhaustion (torch) | Not covered |
| Misaligned tensor shapes in batch processing | Not covered |

### 3.5 Mocking Concerns

Several tests rely on heavy mocking that may hide real bugs:

1. **File System Operations** — `test_main.py` mocks `secure_load_module()` rather than testing real file I/O
2. **Module Loading** — Real import errors and dependency resolution not tested end-to-end
3. **Subprocess Calls** — `dependency_scanner.py` has zero test coverage for actual subprocess execution
4. **Integration Tests** — `test_integration_workflows.py` has only 16 tests with heavy dependency mocking

### 3.6 Underutilized Test Infrastructure

The `conftest.py` defines fixtures that are rarely or never used:

| Fixture | Purpose | Usage |
|---------|---------|-------|
| `raises_fixture` | Custom exception testing | Minimal |
| `oom_fixture` | Out-of-memory simulation | Minimal |
| `mock_memory_error` | Memory error patching | Minimal |
| `flaky_operation` | Flaky operation testing | Never referenced |
| `exception_test_cases` | Predefined exceptions | Not heavily used |

### 3.7 Property-Based Testing (Hypothesis)

**Current:** ~20 property tests in `test_property_based_testing.py` covering mathematical properties, statistical invariants, and edge cases.

**Missing property tests for:**
- CLI argument parsing invariants
- Configuration transformation invariants
- Data pipeline properties
- Numerical stability across parameter ranges
- File format handling properties

---

## 4. Recommendations

### 4.1 Immediate (Critical)

1. **Add integration tests for all 18 untested CLI commands** — These are the primary user-facing entry points and represent the highest risk coverage gap.
2. **Add tests for the 3 untested utility modules** — `dependency_scanner.py`, `security_audit_logger.py`, and `security_logging_integration.py` need basic unit tests.
3. **Add real file I/O tests** for path validation — Supplement mocked tests with actual filesystem operations to catch permission and path resolution issues.

### 4.2 High Priority

4. **Implement visualization function tests** using matplotlib assertion helpers.
5. **Test data pipeline end-to-end** with real CSV files.
6. **Add property-based tests** for all Hypothesis-worthy mathematical functions.
7. **Test all validation/falsification protocols individually** — Current tests cover the framework but not individual protocol correctness.

### 4.3 Medium Priority

8. **Utilize defined fixtures** — `raises_fixture`, `oom_fixture`, and `flaky_operation` were created for a reason but never used.
9. **Add performance regression tests** for the `@pytest.mark.performance` category.
10. **Test concurrent access patterns** with `_config_lock`.
11. **Implement audit log persistence** — Current in-memory audit trail (1,000 entries) should write to disk for forensic analysis.

### 4.4 Documentation

12. **Update CLAUDE.md** to reflect the actual APGI CLI framework rather than the non-existent FastAPI web application it currently describes.

### 4.5 Security Maintenance

13. **Implement periodic key rotation** for `PICKLE_SECRET_KEY` and `APGI_BACKUP_HMAC_KEY`.
14. **Regularly audit dependencies** in `requirements.txt` for known vulnerabilities (especially PyTorch, scikit-learn, numpy).
15. **Add TOCTOU mitigation** — Consider file locking for critical file operations to fully mitigate time-of-check/time-of-use race conditions.

---

*Report generated as part of the APGI audit & coverage validation effort.*
