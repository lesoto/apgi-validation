# APGI Validation Framework — Comprehensive Security & Code Quality Audit Report

**Date**: 2026-03-24
**Repository**: lesoto/apgi-validation
**Branch**: claude/audit-security-report-tpxYP
**Scope**: Full codebase audit covering security, code quality, architecture, and test coverage

---

## Executive Summary

The APGI Validation Framework is a mature scientific computing project (~5,700 lines in `main.py`, 46 utility modules, 70 test files totalling 35,000+ lines) implementing Active Perception and Generative Inference theory. The codebase demonstrates good security intent in many areas (audit logging, path validation, HMAC-signed backups, dependency scanning) but contains several critical vulnerabilities, architectural defects, and test coverage gaps requiring immediate attention.

| Severity | Count |
|----------|-------|
| Critical | 5 |
| High | 12 |
| Medium | 18 |
| Test Coverage Gaps | 8+ |

---

## 1. Critical Findings

### 1.1 Unconditional `nn.Module` Subclass Outside PyTorch Guard

**File**: `Falsification/Falsification_InformationTheoretic_PhaseTransition.py:101`
**Severity**: CRITICAL — blocks all tests that import `utils`

`ThermodynamicEntropyCalculator` is defined as `class ThermodynamicEntropyCalculator(nn.Module)` at module scope, outside any `if HAS_TORCH:` guard. When PyTorch is not installed, `nn` is undefined and a `NameError` is raised at import time.

This module is unconditionally loaded at module level by `utils/batch_processor.py:262–270` via `importlib.util`, which is itself imported by `utils/__init__.py:132`. The cascade means **every test that imports `utils` fails immediately**.

**Fix**: Wrap the class definition inside `if HAS_TORCH:` and add a guard before the unconditional load in `batch_processor.py`.

---

### 1.2 Global Mutable State for Cryptographic Keys

**File**: `utils/batch_processor.py:83–108`
**Severity**: CRITICAL

```python
PICKLE_SECRET_KEY = None      # line 83
APGI_BACKUP_HMAC_KEY = None   # line 84
```

These module-level globals are mutated inside `_ensure_secure_keys()` at import time. Problems:

- Keys stored as plain module globals are trivially leaked in error messages, `repr()` calls, and memory dumps.
- Fallback key generation (lines 94–97, 101–104) uses `os.urandom(32).hex()` then stores the hex *string* directly — the key material is not derived via a proper KDF.
- No key lifecycle or rotation despite the existence of `utils/key_rotation_manager.py`.
- A thread lock is acquired and released *before* key usage, creating a TOCTOU window.

**Fix**: Replace globals with a `SecretStr`-style wrapper, retrieve keys only at point-of-use, and integrate `key_rotation_manager.py`.

---

### 1.3 HMAC Signature Length Over-Permissive in Backup Manager

**File**: `utils/backup_manager.py:194–250`
**Severity**: CRITICAL

Signature length is validated as "0–1024 bytes" during audit-log loading. SHA-256 produces exactly 32 bytes; SHA-512 produces 64 bytes. Accepting up to 1024 bytes allows length-extension or padding attacks and could let an attacker substitute a tampered history entry with a pre-computed oversized signature.

**Fix**: Enforce `len(signature) == 32` (or 64 for SHA-512), use `hmac.compare_digest()` for constant-time comparison, and add a signature-version header.

---

### 1.4 Bare Exception Silently Drops Audit Events

**File**: `utils/persistent_audit_logger.py:124–130`
**Severity**: CRITICAL

```python
except Exception:
    try:
        with open(self.log_file, "a") as f:
            f.write(f"{json.dumps(log_entry)}\n")
    except Exception as e2:
        print(f"Failed to write audit log: {e2}")
```

- Catches `BaseException` subclasses including `KeyboardInterrupt` and `SystemExit`.
- `_check_rotation()` (line 137) has `except Exception: pass` — rotation failures are silently ignored.
- If both paths fail, security events are **silently lost**.
- Unbounded fallback writes enable log-exhaustion DoS.

**Fix**: Catch specific exceptions (`OSError`, `ValueError`); never swallow `KeyboardInterrupt`/`SystemExit`; raise or alert on rotation failure.

---

### 1.5 `_check_file_size()` Unhandled OS Exceptions

**File**: `main.py:880`
**Severity**: CRITICAL

```python
size_mb = os.path.getsize(str(file_path)) / (1024 * 1024)
```

`os.path.getsize()` raises `FileNotFoundError` or `OSError` (permission denied, broken symlink) but neither is caught. The function signature advertises `ValueError` only. Unhandled OS errors crash the CLI.

**Fix**: Wrap in `try/except OSError` and convert to a descriptive `ValueError`.

---

## 2. High Priority Findings

### 2.1 Additional Unconditional `import torch` Locations

The following files import `torch` at module scope without try/except:

| File | Lines |
|------|-------|
| `Falsification/Falsification_NeuralNetwork_EnergyBenchmark.py` | 36–38 |
| `Validation/ActiveInference_AgentSimulations_Protocol3.py` | 28–30 |
| `Validation/NeuralNetwork_InductiveBias_ComputationalBenchmark.py` | 35–37 |
| `Validation/SyntheticEEG_MLClassification.py` | 32–34 |

**Fix**: Wrap each with `try: import torch; HAS_TORCH=True except ImportError: HAS_TORCH=False` and guard dependent code.

---

### 2.2 Zip Slip Vulnerability in Backup Restore

**File**: `utils/backup_manager.py:693–695`
**Severity**: HIGH

Path containment check uses:
```python
str(member.filename).startswith(str(extract_path))
```

String prefix matching is bypassable with crafted paths (e.g., `/safe/path../evil`).

**Fix**: Use `Path(extract_path / member.filename).resolve().relative_to(extract_path.resolve())` and catch `ValueError`.

---

### 2.3 Unvalidated Subprocess in Dependency Scanner

**File**: `utils/dependency_scanner.py:38–89`
**Severity**: HIGH

`str(self.requirements_file)` is passed to `subprocess.run()`. While a list is used (preventing shell injection), a requirements file path containing special characters or spaces could still trigger unexpected behavior. Additionally, the tool is located by name only — a rogue `pip-audit` earlier in `PATH` would be silently executed.

**Fix**: Resolve the tool via `shutil.which("pip-audit")` and assert the result is within expected system paths; validate `requirements_file` is within project root.

---

### 2.4 JSON Deserialization DoS — No Nesting/Element Limit

**File**: `utils/data_validation.py:126–127`
**Severity**: HIGH

`json.load(f)` is called without any nesting depth or element-count limit. A crafted JSON file with millions of nested arrays can exhaust the call stack or heap even if the file size is below the MB threshold.

**Fix**: Use a streaming JSON parser for large files or set `sys.setrecursionlimit` conservatively and catch `RecursionError`.

---

### 2.5 Dynamic Module Loading Without Interface Validation

**File**: `main.py:302–341, 482`
**Severity**: HIGH

`secure_load_module()` returns a module object, but callers immediately access attributes without checking they exist:
```python
SurpriseIgnitionSystem = module_info["module"].SurpriseIgnitionSystem
```
If the module lacks the expected class, an `AttributeError` propagates unhandled.

**Fix**: After loading, use `hasattr()` to validate required attributes; raise a descriptive error if missing.

---

### 2.6 TOCTOU File Size Check

**File**: `main.py:868–884`
**Severity**: HIGH

There is a time window between the file-size check and the actual file read. An attacker (or concurrent process) could replace the file between the two operations.

**Fix**: Capture `os.stat()` once, validate, then open with `O_NOFOLLOW` (Linux); alternatively, re-validate size inside the open file descriptor.

---

### 2.7 Cryptographic Keys Written to `os.environ` as Plain Hex

**File**: `utils/key_rotation_manager.py:173, 178`
**Severity**: HIGH

Rotated keys are written back to `os.environ` as plain hex strings. Environment variables are visible to all child processes, appear in `/proc/<pid>/environ`, and can be captured by `ps` on some systems.

**Fix**: Keep keys in memory only; never write secret material to environment variables beyond initial bootstrap.

---

### 2.8 Keys Stored as Base64 (Encoding ≠ Encryption) in Files

**File**: `utils/key_rotation_manager.py:118–120`
**Severity**: HIGH

Key files contain base64-encoded key material. Base64 is trivially reversible — it provides no confidentiality protection.

**Fix**: Encrypt key files at rest using `cryptography.fernet.Fernet` with a master key derived from a hardware secret or operator-supplied passphrase.

---

### 2.9 HMAC Key Persisted Without Immediate `chmod 600`

**File**: `utils/backup_manager.py:86–100`
**Severity**: HIGH

The backup HMAC key file is written, but `chmod 600` is applied *after* the write. There is a window during which the file is world-readable.

**Fix**: Open the file with `os.open(path, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)` to set permissions atomically.

---

### 2.10 Missing Validation Protocol Files

Tests reference protocol files that do not exist in the repository:

- `Validation/Validation_Protocol_3.py`
- `Validation/Validation_Protocol_5.py` through `Validation_Protocol_10.py`
- `Validation/Validation_Protocol_12.py`

Tests currently skip when these are absent. Missing implementations reduce falsification coverage.

---

### 2.11 Unregistered pytest Marks Cause `--strict-markers` Warnings

`pytest.ini` enables strict marker checking but the following marks appear in tests without being registered:

- `boundary`
- `regression`
- `parameter_recovery`
- `functional`

**Fix**: Add these marks to the `[pytest]` `markers =` section in `pytest.ini`.

---

### 2.12 Missing `requirements-dev.txt`

`CLAUDE.md` instructs developers to run `pip install -r requirements-dev.txt` but the file does not exist. Test and lint dependencies are mixed into `requirements.txt`, bloating production installs.

**Fix**: Create `requirements-dev.txt` with `pytest`, `hypothesis`, `black`, `isort`, `flake8`, `pytest-cov`, and related packages.

---

## 3. Medium Priority Findings

### 3.1 Signal Handler Restoration Logic Incorrect

**File**: `main.py:631–693`
**Severity**: MEDIUM

```python
if original_sigint:
    signal.signal(signal.SIGINT, original_sigint)
```

`signal.getsignal()` returns `signal.SIG_DFL` (integer `0`) or `signal.SIG_IGN` (integer `1`) for default/ignored signals — both are falsy. The condition incorrectly skips restoration for these cases.

**Fix**: Change to `if original_sigint is not None:`.

---

### 3.2 Configuration Schema Not Enforced at Load Time

**File**: `main.py:381–387`
**Severity**: MEDIUM

`config/config_schema.json` exists but is never applied to the loaded YAML configuration. Invalid configuration values can pass through silently.

**Fix**: Validate loaded config dict against `config_schema.json` using `jsonschema.validate()`.

---

### 3.3 Global Config Lock Protects Access but Not Mutable Values

**File**: `main.py:165–188`
**Severity**: MEDIUM

The `_config_lock` mutex protects dict-level reads and writes but not mutation of *values* that are themselves mutable objects (lists, dicts). Multi-threaded callers receiving the same list reference could corrupt it.

**Fix**: Return deep copies of mutable config values, or use an immutable config representation.

---

### 3.4 Incomplete Secret Redaction Patterns in Logging

**File**: `utils/logging_config.py:46–69`
**Severity**: MEDIUM

The log redaction filter only matches a subset of secret patterns. Keys with names like `api_token`, `auth_key`, `client_secret`, or `bearer` are not redacted and could appear in log output.

**Fix**: Expand the pattern list; consider a blocklist approach (redact anything matching common secret naming conventions) rather than an allowlist.

---

### 3.5 `.env` File Parsing Without Key/Value Validation

**File**: `utils/batch_processor.py:40–41`
**Severity**: MEDIUM

`.env` lines are split on `=` without validating that the key is a legitimate environment variable name. A crafted `.env` file could inject unexpected values.

**Fix**: Validate key names against a strict regex (`[A-Z_][A-Z0-9_]*`) before calling `os.environ.__setitem__`.

---

### 3.6 `subprocess.Popen` Accepts Arbitrary `**kwargs`

**File**: `utils/timeout_handler.py:256–257`
**Severity**: MEDIUM

Callers can pass arbitrary keyword arguments to `Popen`, potentially enabling shell injection via `shell=True` or disabling safety features.

**Fix**: Accept only an explicit allowlist of `Popen` parameters.

---

### 3.7 Import Fallback Chain Masks Real Errors

**File**: `utils/config_manager.py:38–51`
**Severity**: MEDIUM

Three-level import fallback silently proceeds past real `ImportError`s. If all three fail, the code attempts to use `logging_config` as a bare name, raising an opaque `NameError` at runtime.

**Fix**: Catch `ImportError` at all three levels with explicit logging; fail fast if all attempts fail.

---

### 3.8 JSON Decode Error Printed Twice

**File**: `main.py:526–538`
**Severity**: MEDIUM

When a parameter file contains invalid JSON, the error message is emitted twice (lines 528 and 534). Execution then continues with default parameters without a log entry.

**Fix**: Remove the duplicate `quiet_print`; emit a single structured error message and log it via the audit logger.

---

### 3.9 Known Unfixed Bugs Documented in Code Comments

The following comment-tagged bugs are acknowledged but unresolved:

| Tag | Location | Description |
|-----|----------|-------------|
| BUG-047 | `main.py` | Missing explicit file encoding specification in `open()` calls |
| BUG-049 | `main.py:5380–5400` | `max()`/`min()` called on potentially empty collections |

---

### 3.10 Inefficient Repeated Path Resolution in `_validate_file_path`

**File**: `main.py:850–863`
**Severity**: MEDIUM (performance)

`(project_root / allowed_dir).resolve()` is called inside the loop for every `allowed_dir` on every invocation. These paths are static.

**Fix**: Pre-compute and cache the resolved allowed-directory set at startup.

---

### 3.11 Thread-Unsafe Module Loading Cache

**File**: `utils/batch_processor.py:282`
**Severity**: MEDIUM

`_loaded_validation_modules` is a plain `dict` updated by multiple threads without a lock. Concurrent module loads can corrupt the dict or load the same module twice.

**Fix**: Protect with a `threading.Lock` or use `functools.lru_cache` with proper synchronization.

---

### 3.12 Error Handling Missing in Validation Protocol Execution Loop

**File**: `Validation/Master_Validation.py`
**Severity**: MEDIUM

Protocol execution loops have no `try/except`. A single failing protocol aborts the entire validation run with no partial results returned.

**Fix**: Catch exceptions per-protocol, record the failure, and continue with remaining protocols.

---

### 3.13–3.18 Additional Medium Issues

| # | File | Issue |
|---|------|-------|
| 3.13 | `utils/backup_manager.py:820–847` | Symlink resolution during restore uses string comparison; switch to `Path.resolve()` |
| 3.14 | `main.py:302–341` | Module loader exceptions caught but emitted only as warnings; propagate to caller |
| 3.15 | `utils/key_rotation_manager.py` | Key rotation does not invalidate in-flight references held by other modules |
| 3.16 | `requirements.txt` | Overly broad version ranges (`numpy>=1.26,<3.0.0`) will silently upgrade across breaking major versions |
| 3.17 | `config/config_schema.json` | Schema itself is not integrity-checked; a tampered schema could allow malicious configurations |
| 3.18 | `utils/data_validation.py` | File size limit in MB doesn't protect against deeply-nested-structure DoS (see 2.4) |

---

## 4. Test Coverage Gaps

### 4.1 Path Validation Edge Cases
No tests for:
- Symlink-based path traversal attempts
- Absolute path rejection after symlink resolution
- TOCTOU race in file size check

### 4.2 Environment Variable Initialization
No test verifying that absent `PICKLE_SECRET_KEY` or `APGI_BACKUP_HMAC_KEY` raises `EnvironmentError` in production mode. No test for the fallback key-generation path.

### 4.3 Signal Handler Restoration
No test verifying that the original SIGINT handler is restored after the simulation loop (including on exception paths).

### 4.4 File Operations in `_check_file_size`
No tests for: non-existent files, permission-denied files, zero-byte files, files exactly at the size limit.

### 4.5 Concurrency
`test_concurrent_config_access.py` exists but does not test:
- Race conditions in get-set config patterns
- Concurrent module loading races
- Stress scenarios with many threads

### 4.6 Backup Manager HMAC
No negative tests for:
- Tampered backup history detection
- Oversized signature acceptance
- Missing HMAC key at restore time

### 4.7 Dynamic Module Loading Robustness
No tests for:
- Modules missing expected attributes
- Malformed/syntax-error modules
- Circular import scenarios

### 4.8 Validation Protocol Failures
No tests for:
- Individual protocol raising an exception mid-run
- Partial failure recovery
- Protocol timeout handling

---

## 5. Dependency Analysis

**File**: `requirements.txt`

| Issue | Detail |
|-------|--------|
| Overly broad ranges | `numpy>=1.26.0,<3.0.0`, `torch>=2.1.0,<3.0.0` span breaking major versions |
| Missing `requirements-dev.txt` | Test/lint tools mixed into production dependencies |
| No `cryptography` package pinned | Used implicitly; version drift can introduce CVEs |
| No `requests` package listed | May be pulled in as transitive dep without version constraint |

---

## 6. Recommendations — Priority Order

### Immediate (Week 1)

1. **Fix `nn.Module` NameError** — wrap `ThermodynamicEntropyCalculator` in `if HAS_TORCH:` and add `ImportError` handling in `batch_processor.py`.
2. **Fix all unconditional `import torch`** in Validation and Falsification modules.
3. **Fix `_check_file_size()` OS exceptions** — add `try/except OSError`.
4. **Fix Zip Slip** in backup restore — use `Path.resolve().relative_to()`.
5. **Fix silent audit log swallowing** — catch specific exceptions; never swallow `KeyboardInterrupt`/`SystemExit`.
6. **Register missing pytest marks** and create `requirements-dev.txt`.

### Short-term (Week 2–3)

7. **Enforce configuration schema** with `jsonschema.validate()` at load time.
8. **Fix HMAC signature length validation** — enforce exact hash output size.
9. **Fix signal handler restoration** condition (`is not None`).
10. **Add unit tests** for all identified coverage gaps.
11. **Fix module interface validation** after dynamic loads.
12. **Patch key persistence** — use `os.open()` with `0o600` permissions atomically.

### Medium-term (Week 4+)

13. **Refactor global cryptographic key state** in `batch_processor.py`.
14. **Integrate key rotation** with in-flight module invalidation.
15. **Expand log redaction patterns**.
16. **Create stub implementations** for missing Validation Protocol files.
17. **Pin dependency versions** more conservatively; create `requirements-dev.txt`.
18. **Add fuzzing tests** for all public input-validation surfaces.

---

## 7. File Inventory

| Area | Files | Approx. Lines |
|------|-------|---------------|
| Main entry point | `main.py` | 5,720 |
| Utils | 46 modules | ~18,000 |
| Validation protocols | 17 files | ~14,000 |
| Falsification protocols | 21 files | ~9,500 |
| Tests | 70 files | ~35,155 |
| Configuration | 12 YAML/JSON files | ~800 |
| **Total** | | **~83,175** |

---

*Report generated by automated codebase audit — 2026-03-24*
