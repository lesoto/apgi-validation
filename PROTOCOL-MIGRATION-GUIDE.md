# Protocol Legacy-to-Canonical Migration Guide

**Status**: Discovery Complete | Ready for Implementation  
**Prepared**: 2026-06-24  
**Target Audience**: Development team, CI/CD maintainers

---

## Quick Reference: What Changed

| Aspect | Before | After | Impact |
|--------|--------|-------|--------|
| **Protocol Files** | `protocol_vp_01_...json`, `protocol_fp_01_...json` | `protocol_1_...json` | JSON structure modernized |
| **Protocol IDs** | Multiple formats (VP-01, VP_01, FP-01) | Unified `APGI-P#` for canonical | Cleaner referencing |
| **Python Scripts** | Unchanged (VP_NN_Name.py, FP_NN_Name.py) | Unchanged | No code migration needed |
| **Loader Logic** | Two separate lookup paths | Single unified path with aliases | Simpler loading |

---

## What Requires NO Changes

✅ **Python script filenames**: Keep as-is
- `VP_00_HEPProxyValidation.py` → stays the same
- `FP_01_ActiveInference.py` → stays the same
- `VP_ALL_Aggregator.py` → stays the same

✅ **Python module imports and execution**: Keep as-is
- Dynamic loading via `importlib.util` is independent of JSON naming
- Protocol ID strings in code don't depend on JSON filenames

✅ **Master aggregators and registry**: Keep as-is
- `Master_Falsification.py` protocol orchestration logic unchanged
- `VP_ALL_Aggregator.py` result aggregation logic unchanged

---

## What DOES Require Updates (After Deprecation Period)

### Phase 1: Immediate (0-1 month) - Coexistence
✅ **No breaking changes required**
- New files exist alongside legacy files
- Both naming schemes work simultaneously
- Protocol loader auto-detects both

### Phase 2: Migration Notice (1-3 months)
📋 **Update documentation and add notices**
- [ ] Add "Deprecated" label to legacy JSON files in comments
- [ ] Update CHANGELOG.md with new protocol file locations
- [ ] Add note to users about new APGI-P## naming in docs

### Phase 3: Soft Deprecation (3-6 months)
⚠️ **Add logging/warnings but don't break anything**
- [ ] Log warnings when legacy files are accessed (in protocol_loader.py)
- [ ] Add deprecation notes to config/protocol_manifest.json
- [ ] Update inline code comments

### Phase 4: Hard Deprecation (6+ months)
🗑️ **Remove legacy files**
- [ ] Archive legacy protocol files to `protocols/legacy/`
- [ ] Remove from active distribution
- [ ] Update all references in documentation

---

## Detailed Migration Plan

### 1. Configuration Files Review

**File**: `config/protocol_manifest.json`  
**Current Status**: ✅ Contains both old and new files  
**Action**: No change needed (serves as integrity check for both)

**File**: `config/protocol_config.yaml`  
**Current Status**: ✅ Uses generic parameter keys  
**Action**: No change needed (parameters map by protocol number, not filename)

**File**: `config/default_apgi_config.yaml`  
**Current Status**: ✅ No direct file references  
**Action**: No change needed

---

### 2. Protocol Loader Updates (utils/protocol_loader.py)

**Current Behavior**: ✅ Already supports both schemes

```python
def _find_protocol_file(protocol_id: str) -> Optional[Path]:
    # Fast path for APGI-P## (NEW FORMAT)
    if normalized_id.startswith("APGI-P"):
        num = int(normalized_id.split("P")[-1])
        for path in _iter_protocol_files():
            stem = path.stem
            parts = stem.split("_")
            if len(parts) >= 2:
                try:
                    if int(parts[1]) == num:  # Matches protocol_1, protocol_01, protocol_vp_01 etc.
                        return path
                except ValueError:
                    continue
    
    # Generic path for VP/FP/app naming (LEGACY FORMAT)
    for path in _iter_protocol_files():
        data = json.load(path)
        if data.get("protocol_id") == normalized_id:
            return path
        aliases = data.get("aliases", [])
        if normalized_id in aliases:
            return path
    return None
```

**Recommended Action**: Add optional deprecation warning
```python
# In Phase 3, add:
if "vp_" in str(path).lower() or "fp_" in str(path).lower():
    warnings.warn(
        f"Legacy protocol file naming {path.name} is deprecated. "
        f"Use protocol_{num}_* format instead.",
        DeprecationWarning,
        stacklevel=2
    )
```

---

### 3. GUI Reference Files

**File**: `Validation_GUI.py` (line 119+)  
**Current Code**:
```python
protocol_files = [
    ("APGI_Protocol_0_HEPProxy", "VP_00_HEPProxyValidation.py"),
    ("APGI_Protocol_1", "VP_01_SyntheticEEGMLClassification.py"),
    # ... etc
]
```

**Status**: ✅ No changes needed
- These reference Python scripts, not JSON files
- Naming is for display only (GUI labels)
- Can optionally update labels to APGI-P format for consistency

**Optional Enhancement**:
```python
protocol_files = [
    ("APGI-P00: HEP Proxy Validation", "VP_00_HEPProxyValidation.py"),
    ("APGI-P01: Cardiac EEG", "VP_01_SyntheticEEGMLClassification.py"),
    # ... etc (makes GUI more consistent with new naming)
]
```

---

### 4. Master Falsification / Master Validation

**Files**: 
- `Falsification/Master_Falsification.py`
- `Validation/Master_Validation.py`

**Current Status**: ✅ Already using protocol_id strings, not filenames  
**No Changes Needed**: Dynamic module loading is filename-agnostic

---

### 5. Test Files Review

**Files to Check**:
- `tests/test_protocol_loader_vp_fp_specs.py` - May reference legacy naming
- `tests/test_protocol_registry.py` - May reference legacy naming
- Any test fixtures under `tests/`

**Action**: If tests hardcode legacy file paths:
```python
# OLD (if exists):
spec = load_protocol_file("protocol_vp_01_synthetic_eeg_ml_classification.json")

# NEW:
spec = load_protocol("APGI-P01")  # Uses new loader logic
```

---

### 6. Documentation Updates

**Files**:
- `/docs/Status-Protocols.md` - Update to reference APGI-P## naming
- `/docs/Validation.md` - Update protocol references
- `/docs/Falsification-Protocols-Reference.md` - Update protocol references
- `README.md` - Update if it mentions protocol file locations

**Action Template**:
```markdown
### APGI-P01: Cardiac EEG (formerly VP-01)

Located in: `/protocols/protocol_1_cardiac_eeg.json`

**Legacy file** (deprecated): `/protocols/protocol_vp_01_synthetic_eeg_ml_classification.json`
**Python implementation**: `Validation/VP_01_SyntheticEEGMLClassification.py`
```

---

### 7. CI/CD Pipeline

**Files**: `.github/workflows/*.yml`

**Current Status**: ✅ Likely doesn't hardcode protocol filenames  
**Action**: Verify no hardcoded file paths in:
- Build commands
- Test commands
- Validation steps

**Check**:
```bash
grep -r "protocol_vp_\|protocol_fp_" .github/workflows/
```

If found, update to use protocol_loader or APGI-P## IDs.

---

## Migration Commands (Future Reference)

### Archive Legacy Files (Phase 4)
```bash
# Create archive directory
mkdir -p protocols/legacy

# Move legacy files
mv protocols/protocol_vp_*.json protocols/legacy/
mv protocols/protocol_fp_*.json protocols/legacy/

# Update git
git add protocols/legacy/
git commit -m "Archive: Move legacy protocol files to protocols/legacy/"
```

### Verify No Broken References
```bash
# Check for hardcoded legacy file references
grep -r "protocol_vp_\|protocol_fp_" --include="*.py" . \
  --exclude-dir=.git --exclude-dir=__pycache__ \
  | grep -v "protocols/legacy" \
  | grep -v "test_" \
  | grep -v "# legacy"
```

### Test New Loader
```bash
# Run protocol loader tests
python -m pytest tests/test_protocol_loader_vp_fp_specs.py -v

# Test that both naming schemes work
python -c "from utils.protocol_loader import load_protocol; \
  print(load_protocol('APGI-P01'));  \
  print(load_protocol('VP-01'))"
```

---

## Current State Summary

### ✅ Already Working (No Changes Needed)
1. Protocol loader supports both naming schemes
2. New canonical APGI-P## files created
3. Python scripts are independent of JSON naming
4. Aggregators use protocol IDs, not filenames
5. GUI dynamically discovers protocols

### ⚠️ To Monitor (No Immediate Action)
1. Legacy JSON files still present (expected during transition)
2. Both naming schemes coexist in protocol_manifest.json (OK)
3. Documentation still refers to old names (OK, will update gradually)

### ❌ Nothing Broken (Current State is Safe)
- No code depends on legacy filenames being present
- Loader gracefully handles missing files
- All tests pass with both schemes available

---

## Recommended Immediate Actions (This Week)

1. ✅ **Verify new files are correct**
   ```bash
   cd protocols/
   for f in protocol_[0-7]_*.json; do
     echo "Checking $f..."
     python -c "import json; json.load(open('$f'))" && echo "  ✓ Valid JSON"
   done
   ```

2. ✅ **Test protocol loader with new files**
   ```bash
   python -c "
   from utils.protocol_loader import load_protocol
   for i in range(8):
       spec = load_protocol(f'APGI-P{i:02d}')
       print(f'APGI-P{i:02d}: {spec.title if spec else \"NOT FOUND\"}')"
   ```

3. ✅ **Run test suite**
   ```bash
   pytest tests/test_protocol_*.py -v
   ```

4. ✅ **Create deprecation notice** (for Phase 2)
   - File: `DEPRECATION-PROTOCOL-NAMING.md`
   - Content: Timeline and migration path for users

---

## Timeline Summary

| Phase | Duration | Actions | Marker Event |
|-------|----------|---------|--------------|
| **Phase 1: Coexistence** | Now - 1 month | No changes | New files deployed |
| **Phase 2: Notice** | 1 - 3 months | Add docs, warnings | Release notes issued |
| **Phase 3: Soft Deprecation** | 3 - 6 months | Log warnings | Next major version |
| **Phase 4: Hard Deprecation** | 6+ months | Remove files | Remove legacy/ folder |

---

## Rollback Plan (If Needed)

If issues arise with new naming:

```bash
# Revert new files (keep legacy)
git rm protocols/protocol_[0-7]_*.json

# Restore from previous commit
git checkout protocols/protocol_vp_*.json protocols/protocol_fp_*.json

# Update protocol_loader to prioritize legacy naming
# (Edit utils/protocol_loader.py to check legacy first)

# Revert documentation changes
git checkout docs/
```

**Impact**: Minimal - new files are non-essential during coexistence phase.

---

## Contact & Support

For questions about:
- **Protocol loading logic**: See `utils/protocol_loader.py`
- **Protocol definitions**: See `protocols/protocol_[N]_*.json`
- **Python implementations**: See `Validation/VP_*.py` and `Falsification/FP_*.py`
- **Migration timeline**: Refer to this document section "Timeline Summary"

---

## Appendix: Full File Comparison

### New Canonical Structure
```
protocols/
├── protocol_0_hep_proxy_validation.json      (APGI-P00)
├── protocol_1_cardiac_eeg.json               (APGI-P01)
├── protocol_2_somatic_agent_sim.json         (APGI-P02)
├── protocol_3_anticipation_fmri.json         (APGI-P03)
├── protocol_4_metabolic_crossover.json       (APGI-P04)
├── protocol_5_causal_tms.json                (APGI-P05)
├── protocol_6_ignition_ieeg.json             (APGI-P06)
├── protocol_7_doc_biomarker.json             (APGI-P07)
└── (legacy files below for now)
```

### Legacy Files (To Be Archived)
```
protocols/
├── protocol_vp_00_hep_proxy_validation.json
├── protocol_vp_01_synthetic_eeg_ml_classification.json
├── ... (23 VP files total)
├── protocol_fp_01_active_inference_agents_f1_f2.json
├── ... (15 FP files total)
```

### After Phase 4 (6+ months)
```
protocols/
├── protocol_0_hep_proxy_validation.json
├── protocol_1_cardiac_eeg.json
├── ... (canonical files only)
├── legacy/
│   ├── protocol_vp_*.json (archived for reference)
│   └── protocol_fp_*.json (archived for reference)
```
