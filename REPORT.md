# APGI-Validation Codebase Audit Report

**Date:** 2026-03-24
**Branch:** `claude/audit-codebase-gaps-VOdKx`
**Repository:** `lesoto/apgi-validation`
**Auditor:** Claude Code (claude-sonnet-4-6)

---

## Executive Summary

This report documents a comprehensive audit of the APGI (Allostatic Precision-Gated Ignition)
Validation Framework codebase. The repository implements a scientific computing CLI for validating
and falsifying a computational theory of consciousness through 12 validation protocols and 14+
falsification protocols.

**Current Maturity Score: 47 / 100**
**Alignment Score: 62 / 100** (documented claims vs. working implementation)

The codebase contains a sophisticated architecture with genuine strengths — extensive security
infrastructure, 70+ test files, property-based tests, thread-safe configuration, HMAC-verified
backups, and comprehensive scientific documentation. However, a pervasive systemic bug (151
malformed `open()` calls) renders large portions of the CLI and test suite non-functional at
runtime. Combined with four missing production dependencies and a 5,771-line God-class entry
point, the gap between documented intent and deployable reality is significant.

---

## Scoring Breakdown

| Category | Score | Max | Notes |
|---|---|---|---|
| Code Quality & Correctness | 7 | 25 | 151 critical runtime bugs; missing type annotations |
| Architecture & Design | 12 | 25 | God-class `main.py`; duplicate threshold files; diverged content |
| Security | 12 | 20 | Good features but auto-key-gen flaw; no dashboard auth |
| Documentation | 13 | 20 | Extensive docs but CLAUDE.md is wrong; missing `scripts/` |
| Testing | 3 | 10 | 70 test files but majority broken by same malformed `open()` bug |
| **Total** | **47** | **100** | |

### Alignment Sub-Scores

| Claim | Alignment |
|---|---|
| CLI commands (documented vs. implemented) | 85% |
| Innovation claims (docs/Innovations.md vs. code) | 70% |
| Security features (claimed vs. working) | 60% |
| Threshold consistency (cross-file) | 85% |
| Test suite executability | 45% |

---

## Critical Issues (P0 — Runtime-Breaking)

### BUG-001: Pervasive Malformed `open()` Mode Strings

**Severity:** P0 — Raises `ValueError: invalid mode` at runtime
**Occurrences:** 151 across 20+ files
**Confirmed with:** `python3 -c "open('/tmp/t', ', encoding=\"utf-8\"w')"` → `ValueError: invalid mode: ', encoding="utf-8"w'`

Every affected call wraps the mode and encoding into one mangled string argument instead of two separate arguments. The pattern has been replicated consistently, suggesting a systematic find-and-replace that went wrong.

**Wrong (current):**
```python
with open(output_file, ', encoding="utf-8"w') as f:
```

**Correct:**
```python
with open(output_file, 'w', encoding='utf-8') as f:
```

**Affected files and occurrence counts:**

| File | Count |
|---|---|
| `main.py` | 25 |
| `tests/test_validation_falsification_protocols_individual.py` | 19 |
| `tests/test_cli_integration_comprehensive.py` | 16 |
| `tests/test_data_pipeline_end_to_end.py` | 13 |
| `tests/test_genome_data_extractor.py` | 8 |
| `Validation/APGI_Validation_GUI.py` | 7 |
| `tests/test_main.py` | 6 |
| `tests/test_property_based_additional.py` | 4 |
| `Validation/ActiveInference_AgentSimulations_Protocol3.py` | 4 |
| `utils/data_validation.py` | 3 |
| `tests/test_threshold_imports.py` | 3 |
| `tests/test_performance_benchmarks.py` | 3 |
| `tests/conftest.py` | 3 |
| `setup_environment.py` | 3 |
| `utils/crash_recovery.py` | 2 |
| `tests/test_regression_baselines.py` | 2 |
| `tests/test_error_handling.py` | 2 |
| `Validation/Psychophysical_ThresholdEstimation_Protocol1.py` | 2 |
| `Falsification/Falsification_ActiveInferenceAgents_F1F2.py` | 2 |
| `APGI_Parameter_Estimation.py` | 2 |
| `APGI_Multimodal_Integration.py` | 1 |
| *(others)* | ~25 |

**Impact:** Every CLI command that writes output, every test that creates a temporary file,
every session log, every backup confirmation — all fail with a `ValueError`. The pytest suite
is essentially non-executable for integration tests.

**Fix:**
```bash
# Automated sed fix across the entire repo
find . -name "*.py" | xargs sed -i \
  "s/open(\(.*\), ', encoding=\"utf-8\"\([rwab]*\)')/open(\1, '\2', encoding='utf-8')/g"
# Review and clean up double-encoding cases like ', encoding="utf-8"r', encoding="utf-8"
```

---

## High Priority Issues (P1 — Functional Gaps)

### BUG-002: Missing Production Dependencies in `requirements.txt`

**Severity:** P1 — `ImportError` at runtime for affected modules

The following packages are unconditionally imported (not inside `try/except ImportError`) in
production code but are absent from `requirements.txt`:

| Package | Missing From | Used In |
|---|---|---|
| `python-dotenv` | `requirements.txt` | `utils/__init__.py:16`, `utils/backup_manager.py:51,75`, `utils/config_manager.py:39`, `utils/batch_processor.py` |
| `pymc` | `requirements.txt` | `APGI_Parameter_Estimation.py:27`, `APGI_Multimodal_Classifier.py:34`, `Validation/BayesianModelComparison_ParameterRecovery.py:27`, `Validation/Validation_Protocol_11.py`, `main.py:1451`, and 5 others |
| `arviz` | `requirements.txt` | `APGI_Parameter_Estimation.py:23`, `APGI_Bayesian_Estimation_Framework.py:29`, `Validation/BayesianModelComparison_ParameterRecovery.py:21`, and 6 others |
| `sympy` | `requirements.txt` | `Falsification/Falsification_MathematicalConsistency_Equations.py:54-55` |

Note: `python-dotenv` is the most critical omission — it is imported at the top level of
`utils/__init__.py`, meaning **every single utils import fails** in a clean environment
without dotenv installed.

**Fix — append to `requirements.txt`:**
```
python-dotenv>=1.0.0,<2.0.0
pymc>=5.0.0,<6.0.0
arviz>=0.16.0,<1.0.0
sympy>=1.12,<2.0.0
```

Note that `pymc` v5 requires Python ≥ 3.10. Several existing version pins in `requirements.txt`
(e.g., `pandas>=1.3.0,<1.4.0`, `scikit-learn>=1.0.0,<1.1.0`) are years behind current stable
releases and will conflict with pymc's transitive dependencies. A dependency resolution pass is
needed.

---

### BUG-003: `utils/__init__.py` Raises `EnvironmentError` at Import Time

**File:** `utils/__init__.py`
**Severity:** P1 — Breaks all `utils` imports in clean CI environments

`utils/__init__.py` checks for `PICKLE_SECRET_KEY` and `APGI_BACKUP_HMAC_KEY` at module load
time and raises `EnvironmentError` if either is missing. This means:

- A fresh `pip install` + `pytest` run without setting environment variables fails before
  any test code executes.
- Any downstream consumer that imports from `utils.*` fails immediately.
- The documented approach (`pytest` with `--hypothesis-profile=ci`) gives a misleading
  `EnvironmentError` rather than a test failure.

**Fix:** Move the environment variable check into a lazy initializer or use a warning with a
default development key, similar to how `backup_manager.py` handles the same situation:
```python
# In utils/__init__.py — replace hard raise with warning + sentinel
import warnings
_PICKLE_SECRET_KEY = os.environ.get("PICKLE_SECRET_KEY")
if not _PICKLE_SECRET_KEY:
    warnings.warn(
        "PICKLE_SECRET_KEY not set. Pickle signing disabled. Set for production use.",
        RuntimeWarning, stacklevel=2
    )
```

---

## Architecture Issues (P2)

### ARCH-001: `main.py` God Object

**File:** `main.py`
**Metrics:** 5,771 lines | 93 functions | 39 CLI commands

`main.py` contains the CLI argument parsing, business logic, output formatting, module loading,
signal handling, thread-safe configuration, and file I/O for the entire framework. This violates
the Single Responsibility Principle and makes the file:

- Untestable in isolation (circular imports are masked by dynamic loading)
- Impossible to navigate without tooling
- A perpetual merge-conflict hotspot

**Recommended decomposition:**
```
cli/
├── __init__.py          # click group registration only
├── commands/
│   ├── validate.py      # validate, validate-entropy, validate-active-inference
│   ├── falsify.py       # falsify, falsify-*, causal
│   ├── visualize.py     # visualize, visualize-*
│   ├── benchmark.py     # benchmark, performance
│   ├── data.py          # process-data, validate-pipeline
│   ├── backup.py        # backup, restore, list-backups, cleanup-backups
│   └── config.py        # config, info, audit-dependencies
└── loader.py            # APGIModuleLoader (secure dynamic imports)
```

---

### ARCH-002: Duplicate `falsification_thresholds.py` with Diverged Content

**Files:**
- `utils/falsification_thresholds.py` — canonical version (imported by `utils/__init__.py`)
- `falsification_thresholds.py` — root-level duplicate with **7 additional constants**

The root-level file contains thresholds absent from the utils version:

```python
# In root falsification_thresholds.py ONLY — not in utils/ version:
F6_DELTA_AUROC_MIN: float = 0.05            # LNN AUROC superiority threshold
F2_3_MIN_RT_ADVANTAGE_MS: float = 50.0     # ≥50ms RT advantage
F2_3_MIN_BETA: float = 25.0
F2_3_MIN_STANDARDIZED_BETA: float = 0.40
F2_3_MIN_R2: float = 0.18
F2_3_ALPHA: float = 0.01
F2_CARDIAC_DETECTION_ADVANTAGE_MIN: float = 0.12  # 12% cardiac detection advantage
```

Any protocol importing from the root version uses stricter/different thresholds than one
importing from `utils/`. This is a silent correctness bug in falsification logic.

**Fix:** Delete the root-level duplicate and update all imports to `utils.falsification_thresholds`.
Port the 7 extra constants into `utils/falsification_thresholds.py` if they are intentional.

---

### ARCH-003: `BackupManager` Silent HMAC Key Auto-Generation

**File:** `utils/backup_manager.py:97`
**Severity:** P2 Security

When `APGI_BACKUP_HMAC_KEY` is not set, `BackupManager.__init__` silently:
1. Generates a new random 32-byte key via `os.urandom(32)`
2. Persists it to `backups/.backup_key` in plain text (mode 0o600)

This means:
- Different processes (or restarts) may use different keys, making previous backups
  unverifiable.
- The HMAC integrity guarantee is implicitly voided without any warning to the user.
- The key file is stored inside the backup directory — if the backup is copied without the
  key file, all backup integrity checks silently fail.

```python
# utils/backup_manager.py:97
backup_hmac_key = base64.b64encode(os.urandom(32)).decode("utf-8")
# ... persisted to backups/.backup_key — no warning emitted
```

**Fix:** Raise a `RuntimeError` with a clear message when the env var is missing, rather
than silently generating an ephemeral key. The current `utils/__init__.py` pattern
(EnvironmentError) is the right instinct, just needs better UX around it.

---

### ARCH-004: Missing `scripts/` Directory Referenced by Makefile

**File:** `Makefile` (threshold-lint target)
**Severity:** P2

```makefile
threshold-lint:
    python scripts/threshold_lint.py
```

The `scripts/` directory does not exist in the repository. Running `make threshold-lint`
fails immediately with `python: can't open file 'scripts/threshold_lint.py': [Errno 2]`.

**Fix:** Either create `scripts/threshold_lint.py` implementing the threshold linting logic
(cross-checking that both `falsification_thresholds.py` files are in sync), or remove the
Makefile target.

---

## Documentation Issues (P3)

### DOC-001: CLAUDE.md States Wrong CLI Framework

**File:** `CLAUDE.md`
**Line:** "Main entry point: `main.py` defines the unified CLI interface using `typer`..."

The codebase uses `click`, not `typer`. This is confirmed by:
- `main.py:50`: `import click`
- `tests/test_main.py:12`: `from click.testing import CliRunner`
- `requirements.txt`: `click>=8.0.0,<8.1.0` (typer is not listed)

**Fix:** Update CLAUDE.md to reference `click`.

---

### DOC-002: Outdated Dependency Versions

**File:** `requirements.txt`
**Severity:** P3

Several pinned version ranges are multiple major versions behind current stable releases
as of 2026-03-24:

| Package | Pinned Range | Current Stable |
|---|---|---|
| `pandas` | `>=1.3.0,<1.4.0` | 2.2.x |
| `scikit-learn` | `>=1.0.0,<1.1.0` | 1.5.x |
| `sqlalchemy` | `>=1.4.0,<1.5.0` | 2.0.x |
| `alembic` | `>=1.7.0,<1.8.0` | 1.13.x |
| `cryptography` | `>=41.0.0,<42.0.0` | 44.x |
| `rich` | `>=12.0.0,<13.0.0` | 13.x |
| `numba` | `>=0.56.0,<0.57.0` | 0.60.x |

The ultra-tight minor-version pins (`<1.4.0`) will conflict with pymc/arviz transitive
requirements. Python 3.11+ users will encounter immediate install failures.

---

### DOC-003: `test_property_based_testing_broken.py` Left in Test Suite

**File:** `tests/test_property_based_testing_broken.py`
**Severity:** P3

A file whose name explicitly advertises it as broken is committed to the test suite without
`@pytest.mark.skip` decorators. pytest.ini has `--strict-markers` and `--cov-fail-under=80`,
so this file either inflates skip counts or causes test collection errors.

**Fix:** Either fix the tests and rename the file, or add `pytestmark = pytest.mark.skip(reason="known broken - see issue #X")` at the module level.

---

## Security Observations

### SEC-001: Performance Dashboard Has No Authentication

**File:** `utils/performance_dashboard.py`
**Port:** 8050 (Dash default)

`ComprehensivePerformanceDashboard` starts a Dash/Flask web server with no authentication
middleware. Any user with network access to port 8050 can view system metrics, process lists,
and performance data. The server binds to `0.0.0.0` by default in Dash.

**Fix:** Add `server.secret_key` and basic HTTP auth, or bind to `127.0.0.1` only:
```python
app.run_server(host='127.0.0.1', port=8050, debug=False)
```

### SEC-002: Good Security Infrastructure (Positive)

The codebase has genuine security engineering worth preserving:
- `utils/toctou_mitigation.py` — fcntl-based file locking prevents TOCTOU races
- `utils/path_security.py` — path traversal prevention via `Path.resolve()` comparison
- `utils/security_audit_logger.py` + `utils/persistent_audit_logger.py` — dual audit trail
- `utils/key_rotation_manager.py` — 30-day Fernet key rotation with 5 backup keys
- `utils/logging_config.py:SecretsRedactionFilter` — regex redaction of API keys, JWTs, DB URLs
- `main.py:secure_load_module()` — importlib-based dynamic loading with path validation
- `utils/batch_processor.py` — HMAC-signed pickle serialization via `PICKLE_SECRET_KEY`

These are well-implemented. The primary gap is the auto-key-generation bypass in `BackupManager`.

---

## Test Suite Assessment

**Total test files:** 78 (in `tests/`)
**Estimated executable without the open() bug fix:** ~35% (files that don't call `open()`)
**Property-based test coverage:** Good — Hypothesis `@given` tests in 3 files covering core math

### Positive Observations
- Hypothesis profiles: `dev` (10 examples), `ci` (100 examples), `thorough` (1000 examples)
- `conftest.py` auto-detects CI via 10 environment indicators
- Extensive fixture library: `temp_dir`, `sample_config`, `sample_data`, `flaky_operation`, `oom_fixture`
- Custom markers: `slow`, `integration`, `unit`, `performance`, `hypothesis`, `boundary`, `regression`
- `tests/test_concurrent_config_access.py` — verifies thread-safety of `_config_lock`
- `tests/test_backup_hmac_validation.py` — covers HMAC integrity path
- `tests/test_persistent_audit_logger.py` — covers 10MB rotation and 5 backup files

### Gaps
- Duplicate test module coverage: `test_apgi_equations.py`, `test_apgi_equations_extended.py`,
  `test_apgi_equations_fixed.py` suggest iterative patching rather than test design
- No integration test verifies the full 12-protocol orchestration in `Master_Validation.py`
- No test exercises `DependencyScanner` against a real (sandboxed) `pip-audit` call
- `tests/test_property_based_testing_broken.py` is in the test suite without skip markers

---

## Inventory Summary

| Component | Count | Status |
|---|---|---|
| CLI commands | 39 | Blocked by BUG-001 on output paths |
| Validation protocols | 15 (VP-1 to VP-12 + extras) | Implemented; BUG-001 affects write paths |
| Falsification protocols | 23 | Implemented; BUG-001 affects write paths |
| Utils modules | 48 | BUG-002 (python-dotenv) breaks all at import |
| Test files | 78 | ~45% executable without fixes |
| Documentation files | 34 | Comprehensive but CLAUDE.md incorrect |
| Configuration files | 12 | Two duplicate config trees (root + utils/config/) |
| Core scientific modules | 13 | AST-valid; BUG-001 affects output paths |

---

## Prioritized Remediation Plan

### Phase 1 — Make It Run (1–2 days)

1. **Fix BUG-001** — automated sed to correct all 151 malformed `open()` calls:
   ```bash
   # Dry-run first
   grep -rn ', encoding="utf-8"[rwab]' --include="*.py" | wc -l
   # Apply fix
   find . -name "*.py" -exec perl -i -pe \
     's/open\(([^,]+), '"'"', encoding="utf-8"([rwab]+)'"'"'\)/open($1, '"'"'$2'"'"', encoding='"'"'utf-8'"'"')/g' {} \;
   # Handle double-encoding cases separately
   grep -rn "encoding=\"utf-8\", encoding=\"utf-8\"" --include="*.py"
   ```

2. **Fix BUG-002** — add missing dependencies to `requirements.txt`:
   ```
   python-dotenv>=1.0.0,<2.0.0
   pymc>=5.0.0,<6.0.0
   arviz>=0.16.0,<1.0.0
   sympy>=1.12,<2.0.0
   ```

3. **Fix BUG-003** — replace hard `EnvironmentError` in `utils/__init__.py` with a
   warning that allows development without keys set.

### Phase 2 — Make It Correct (1 week)

4. **Fix ARCH-002** — consolidate `falsification_thresholds.py` to single canonical file;
   port the 7 diverged constants; fix all imports.

5. **Fix ARCH-003** — `BackupManager` must not silently auto-generate HMAC keys.

6. **Create `scripts/threshold_lint.py`** — implement cross-file threshold consistency check
   as referenced by `Makefile`.

7. **Update `requirements.txt` version pins** — unpin to `>=X.Y,<X+1.0` bounds compatible
   with pymc 5.x transitive requirements.

### Phase 3 — Make It Maintainable (2–4 weeks)

8. **Decompose `main.py`** (ARCH-001) — extract 39 commands into `cli/commands/` modules.

9. **Fix DOC-001** — update CLAUDE.md to reference `click`, not `typer`.

10. **Fix `test_property_based_testing_broken.py`** — add skip markers or fix tests.

11. **Add type annotations** to at least `utils/` public interfaces (supports mypy enforcement
    already in `requirements-dev.txt`).

12. **Bind `performance_dashboard.py` to localhost** (SEC-001).

---

## Positive Findings Worth Preserving

The following represent genuine engineering quality that should be maintained through refactoring:

- **Statistical rigour**: `utils/statistical_tests.py` enforces N≥2 and rejects NaN/Inf before
  all tests; safe wrappers around `scipy.stats` functions.
- **Falsification framework**: `utils/meta_falsification.py` + `utils/criteria_registry.py`
  implement a Popperian framework with quantitative thresholds — scientifically sound.
- **Threshold registry**: `utils/falsification_thresholds.py` as a single-source dataclass for
  all falsification thresholds is excellent design; the duplication in BUG-002 undermines it.
- **Security audit trail**: dual-logger design (`SecurityAuditLogger` + `PersistentAuditLogger`)
  with 10MB rotation and in-memory ring buffer is production-grade.
- **Crash recovery**: `utils/crash_recovery.py` auto-save thread with exponential backoff
  (max 5 retries, 300s ceiling) is robust.
- **TOCTOU mitigation**: `utils/toctou_mitigation.py` using `fcntl` is correct and necessary.
- **Property-based tests**: Hypothesis-based tests for core APGI equations provide strong
  mathematical invariant coverage.
- **Cross-protocol consistency**: `utils/cross_protocol_consistency.py` checking that the same
  criteria produce consistent results across protocols is a sound meta-validation approach.

---

*Generated by Claude Code audit on 2026-03-24. Full exploration transcript available in task output.*
