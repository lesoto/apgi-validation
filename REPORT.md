# APGI Validation Framework — Security Audit Report

**Date**: 2026-03-19
**Scope**: Full codebase security audit, code quality analysis, and test coverage assessment
**Files Analyzed**: 124 Python source files (78 application + 33 test files + 13 config/utility)

---

## Executive Summary

This report presents a comprehensive security audit of the APGI Validation Framework. The analysis covers security vulnerabilities, code quality issues, test coverage gaps, and architectural concerns. The codebase demonstrates solid foundational design with proper authentication, middleware layering, and configuration management. However, several critical security issues and significant test coverage gaps require immediate attention.

**Key Findings:**
- **4 Critical** security vulnerabilities
- **6 High** severity issues
- **6 Medium** severity issues
- **4 Low** severity issues
- **32% test skip rate** (312 of 969 tests skipped)
- **Falsification protocols severely under-tested** (5% coverage)

---

## 1. Critical Security Vulnerabilities

### 1.1 Pickle Deserialization Without Verification
- **File**: `utils/batch_processor.py` (lines 196–247)
- **Severity**: CRITICAL
- **Description**: `secure_pickle_loads()` uses `pickle.loads()` which can execute arbitrary code during deserialization. Even with HMAC verification, pickle is inherently unsafe for untrusted data.
- **Recommendation**: Replace pickle with JSON or MessagePack serialization. If pickle is required, use `RestrictedUnpickler` with a strict class whitelist.

### 1.2 Path Traversal in Directory Operations
- **File**: `utils/batch_processor.py` (line 305)
- **Severity**: CRITICAL
- **Description**: Comments indicate path traversal validation is needed but implementation is incomplete. Dynamic path construction from `module_map` lacks validation.
- **Recommendation**: Implement `os.path.realpath()` checks ensuring resolved paths remain within allowed directories. Reject paths containing `..` sequences.

### 1.3 Unrestricted `sys.path` Manipulation
- **Files**: 30+ files across `Validation/`, `Falsification/`, and `utils/`
- **Severity**: CRITICAL
- **Description**: Widespread use of `sys.path.insert(0, str(PROJECT_ROOT))` allows importing modules from arbitrary project locations. No whitelist validation exists.
- **Recommendation**: Replace with explicit relative imports. Document allowed module paths and validate module names before dynamic import.

### 1.4 Insufficient Key Entropy Validation
- **File**: `utils/backup_manager.py` (lines 106–124)
- **Severity**: CRITICAL
- **Description**: `min_entropy_bits = 0.0` means NO minimum entropy is enforced for the backup HMAC key. Any key, regardless of quality, passes validation.
- **Recommendation**: Set minimum entropy to at least 128 bits. Validate actual entropy using Shannon entropy calculation, not just key length.

---

## 2. High Severity Issues

### 2.1 Missing `.pkl` File Validation
- **File**: `main.py` (lines 976–993)
- **Description**: `_validate_input_file()` accepts `.pkl` files without verification before opening. Combined with the pickle deserialization issue, this creates a direct attack vector.
- **Recommendation**: Reject pickle files entirely if not essential. If required, implement strict file scanning before deserialization.

### 2.2 Numerical Instability in Z-Score Calculation
- **File**: `APGI_Equations.py` (lines 154–156)
- **Description**: Guard only checks `if std <= 0` but does not protect against very small standard deviations that cause numerical instability.
- **Recommendation**: Add epsilon check: `if abs(std) < 1e-10: return 0.0`.

### 2.3 Silent Exception Handling
- **File**: `utils/batch_processor.py` (lines 32–48)
- **Description**: Broad `except Exception` blocks silently continue execution, masking permission errors, corruption, and other critical failures.
- **Recommendation**: Catch specific exceptions (`IOError`, `OSError`). Log stack traces in debug mode.

### 2.4 HMAC Signature Length Not Bounded
- **File**: `utils/backup_manager.py` (lines 175–200)
- **Description**: `sig_len = int.from_bytes(sig_len_bytes, "big")` has no bounds check. A malformed file could cause the reader to attempt reading the entire file as a signature.
- **Recommendation**: Validate `sig_len` is within reasonable bounds (0–1024 bytes).

### 2.5 Configurable Max File Size Bypass
- **File**: `main.py` (lines 877–886)
- **Description**: `max_load_size_mb` is configurable via `get_config_value()` at runtime with no hardcoded upper limit.
- **Recommendation**: Implement a hardcoded absolute maximum (e.g., 1 GB) that cannot be overridden at runtime.

### 2.6 Aggressive Log Redaction Patterns
- **File**: `utils/logging_config.py` (lines 46–69)
- **Description**: Credit card pattern `[0-9]{13,19}` matches many legitimate numeric sequences. Email regex may redact non-sensitive addresses in test data.
- **Recommendation**: Make redaction patterns more conservative. Add whitelisting for known-safe patterns.

---

## 3. Medium Severity Issues

### 3.1 Race Condition in Global Configuration
- **File**: `main.py` (lines 174–196)
- **Description**: `_config_lock` exists but `global_config` is directly modified in several places without lock protection.
- **Recommendation**: Enforce lock usage everywhere global config is accessed or modified.

### 3.2 HMAC Key Regenerated on Each Restart
- **File**: `utils/backup_manager.py` (line 81)
- **Description**: When the environment variable is missing, a new key is auto-generated via `os.urandom(32)`, making previous backups unverifiable.
- **Recommendation**: Persist generated keys securely. Warn users about backup compatibility across restarts.

### 3.3 Missing File Permission Controls
- **File**: `delete_pycache.py` (lines 94–100)
- **Description**: Activation scripts created without explicit file permissions. Should use `mode=0o600` for sensitive files.
- **Recommendation**: Set explicit permissions on all generated files. Add umask protection.

### 3.4 Example Credentials in Documentation
- **File**: `CLAUDE.md` (line 111)
- **Description**: Contains `DATABASE_URL=postgresql://apgi_dev:dev_password@localhost:5432/apgi_api_dev`.
- **Recommendation**: Use placeholder values like `<password>` in documentation.

### 3.5 Incomplete JSON Schema Validation
- **File**: `utils/config_manager.py`
- **Description**: Imports `jsonschema` but validation coverage appears incomplete. Configuration parameters could be invalid.
- **Recommendation**: Validate all configuration against schema on load.

### 3.6 Exponential Overflow in Free Energy Calculation
- **File**: `APGI_Equations.py` (line 227)
- **Description**: Clipping range of `[-500, 500]` for exponential arguments is aggressive. Values near boundaries may lose numerical precision.
- **Recommendation**: Use more conservative clipping. Add overflow/underflow detection.

---

## 4. Low Severity Issues

| Issue | File | Description |
|-------|------|-------------|
| Unused imports/dead code | `main.py` | Inconsistent module initialization; typo "pandasl" in comment |
| Missing type hints | `utils/batch_processor.py` | Many public functions lack type annotations |
| Insufficient security logging | Multiple files | File access, path resolution, and permission checks lack audit logging |
| Missing docstrings | `utils/backup_manager.py` | Complex cryptographic functions lack security assumption documentation |

---

## 5. Test Suite Analysis

### 5.1 Overall Statistics

| Metric | Value |
|--------|-------|
| Total test files | 33 |
| Total test functions | 969 |
| Skipped tests (`pytest.skip()`) | 312 (32%) |
| `@pytest.mark.skip` decorators | 233 |
| `assert True` placeholders | 60+ |
| Hypothesis profiles | 3 (dev, ci, thorough) |

### 5.2 Protocol Coverage

| Protocol | Type | Coverage | Status |
|----------|------|----------|--------|
| Protocol 1 (Formal Model) | Validation | 3 functional tests | Partial |
| Protocol 2 (Biophysical Data) | Validation | File existence only | Minimal |
| Protocol 3 (Architectural Models) | Validation | 3 class tests | Partial |
| Protocol 4 (Formal Dynamics) | Validation | 1 integration test | Partial |
| Protocols 5–12 | Validation | Import/class checks only | Minimal |
| F1/F2 (Active Inference) | Falsification | MagicMock only | Inadequate |
| F3 (Bayesian Comparison) | Falsification | No parameter recovery | Inadequate |
| F4–F16 | Falsification | Structure tests only | Inadequate |

### 5.3 Skip Rates by File

| File | Total Tests | Skipped | Rate |
|------|------------|---------|------|
| test_falsification_protocols.py | 168 | 144 | 86% |
| test_spec_protocols.py | 240 | 198 | 83% |
| test_validation_protocols.py | 186 | 152 | 82% |
| test_apgi_entropy_implementation.py | 92 | 67 | 73% |
| test_error_handling.py | 12 | 8 | 67% |

### 5.4 False Positive Risks

- **`assert True` placeholders**: 60+ instances across `test_apgi_entropy_implementation.py` and `test_error_handling.py` that always pass regardless of actual behavior.
- **MagicMock-only tests**: Falsification protocol tests use `MagicMock()` for all complex objects, never testing real algorithm execution.
- **Catch-all exception tests**: `test_error_handling.py` catches exceptions then `assert True`, not validating error type or message.

### 5.5 Missing Test Categories

| Category | Current Coverage | Tests Needed |
|----------|-----------------|--------------|
| Falsification protocols (F1–F16) | 5% | ~150 |
| Validation protocols 5–12 | 15% | ~80 |
| Parameter recovery validation | 10% | ~40 |
| Cross-protocol consistency | 0% | ~30 |
| Numerical boundary testing | 30% | ~60 |
| Negative/error case testing | 20% | ~100 |
| Real data integration | 5% | ~50 |

### 5.6 Missing Edge Cases

| Edge Case | Tested? |
|-----------|---------|
| Empty datasets (n=0) | No |
| Single-sample datasets (n=1) | No |
| All-constant time series | Partial |
| All-zero signals | No |
| NaN/Inf propagation end-to-end | Partial |
| Extreme parameter values (min/max) | No |
| Division by very small std | No |
| Integer overflow in array indexing | No |
| Convergence failure handling | No |
| Memory exhaustion | Fixture defined but unused |

---

## 6. Architecture Observations

### Strengths
- Well-structured middleware stack with clear layering
- JWT authentication with TOTP/MFA support
- Configuration validation that raises errors in production for insecure settings
- Hypothesis property-based testing profiles (dev, ci, thorough)
- Autouse fixture for random state reset between tests

### Concerns
- No `eval()`, `exec()`, or `subprocess.shell=True` found (positive)
- 30+ files manipulate `sys.path` at import time
- Global configuration state shared across threads with incomplete locking
- Test conftest defines unused fixtures (`oom_fixture`, `memory_error_mocker`)
- `test_property_based_testing_broken.py` (590 lines) exists alongside working version without explanation

---

## 7. Recommendations

### Immediate (Critical Priority)
1. **Replace pickle serialization** with JSON/MessagePack in `batch_processor.py`
2. **Implement path traversal validation** — verify all resolved paths stay within allowed directories
3. **Set minimum key entropy** to 128 bits in `backup_manager.py`
4. **Add HMAC signature length bounds** checking (max 1024 bytes)

### Short-Term (1–2 Weeks)
5. **Eliminate all `assert True` placeholders** — replace with meaningful assertions
6. **Implement functional falsification tests** — run algorithms with synthetic ground-truth data
7. **Add numerical boundary tests** — test extreme parameter values
8. **Fix global config race conditions** — enforce lock on all access paths

### Medium-Term (1–2 Months)
9. **Add cross-protocol consistency tests** — verify validation/falsification agreement
10. **Create regression test baselines** — store golden outputs for comparison
11. **Refactor `sys.path` manipulation** — replace with explicit relative imports
12. **Implement parameter recovery validation** — compare recovered vs true parameters

### Ongoing
13. Run `flake8`, `mypy`, and `black` in CI/CD pipeline
14. Maintain security audit logs for file operations
15. Regular dependency vulnerability scanning
16. Reduce test skip rate below 10%

---

## 8. Hardcoded Secrets Scan

| Location | Finding | Risk |
|----------|---------|------|
| `CLAUDE.md:111` | Example dev password in docs | Low (documentation only) |
| `Tests-GUI.py:1060–1079` | Generates secrets at runtime | None (correct practice) |
| Source code | No hardcoded secrets found | — |

---

## Summary

The APGI Validation Framework has a solid architectural foundation but requires immediate attention to **4 critical security vulnerabilities** (pickle deserialization, path traversal, sys.path manipulation, and insufficient key entropy). The test suite, while large (969 tests), has a **32% skip rate** and **critical coverage gaps** in falsification protocols. Addressing the critical security issues and eliminating false-positive tests should be the top priority.

| Severity | Count | Status |
|----------|-------|--------|
| Critical | 4 | Requires immediate fix |
| High | 6 | Fix within 1 week |
| Medium | 6 | Fix within 1 month |
| Low | 4 | Fix opportunistically |
| **Total** | **20** | — |
