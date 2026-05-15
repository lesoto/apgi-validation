# APGI Validation Framework — Comprehensive Application Audit Report

## Bug Inventory

### BUG-L03 — MCMC `cores=1` threading issue ✅ PARTIALLY FIXED

**File:** `Falsification/FP_10_BayesianEstimation_MCMC.py`

**Fix:** Replaced three hardcoded `cores=1` calls with `cores=_mcmc_safe_cores(n_chains)`. The helper (`_mcmc_safe_cores`) returns `1` when called from a daemon thread (GUI context, where fork-based multiprocessing hangs) and `n_chains` in all other contexts (CLI, headless, test runner). This restores parallel sampling for non-GUI usage.

The underlying daemon-thread/fork issue could be fully resolved with `mp_ctx="spawn"` in PyMC 4+. The comment in `_mcmc_safe_cores` documents this upgrade path.

---

### BUG-M01 — Falsification sub-tests F1.1 and F5 family ❌ OPEN

**File:** `Falsification/FP_01_ActiveInference.py`, lines ~337, ~456
**Severity:** Medium
**Status:** ❌ OPEN — Scientific implementation required. Sprint 3 scope.

```python
# TODO-1: Implement F1.1 threshold sensitivity test
# TODO-6, TODO-2: F5-family falsification tests
```

---

### BUG-M02 — Empirical data loading stub ❌ OPEN

**File:** `utils/bids_data_loaders.py`, line ~105
**Severity:** Medium
**Status:** ❌ OPEN — Requires MNE/neo integration. Sprint 3 scope.

```python
# TODO: Implement actual data loading with MNE or neo
```

`load_empirical_dataset()` falls back to synthetic data silently. `mne` is already in `requirements.txt`.
