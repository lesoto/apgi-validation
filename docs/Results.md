# APGI Output Standardization Guide

## Overview

All scripts in the APGI validation framework now use a unified output management system. This ensures consistent file organization, naming conventions, and metadata across all protocols and theory modules.

## Directory Structure

All outputs are organized in a single unified directory:

```
outputs/
├── validation/
│   ├── VP_01/
│   │   ├── results.json          # Main results file
│   │   ├── metadata.json         # Protocol metadata
│   │   ├── summary.txt           # Human-readable summary
│   │   └── logs/
│   │       └── VP_01.log         # Execution logs
│   ├── VP_02/
│   │   ├── results.json
│   │   ├── metadata.json
│   │   ├── summary.txt
│   │   └── logs/
│   │       └── VP_02.log
│   └── ... (VP_03 through VP_21)
│
├── falsification/
│   ├── FP_01/
│   │   ├── results.json
│   │   ├── metadata.json
│   │   ├── summary.txt
│   │   └── logs/
│   │       └── FP_01.log
│   ├── FP_02/
│   │   ├── results.json
│   │   ├── metadata.json
│   │   ├── summary.txt
│   │   └── logs/
│   │       └── FP_02.log
│   └── ... (FP_03 through FP_12)
│
└── theory/
    ├── APGI_Thermodynamic_Program_Aggregator/
    │   ├── results.json
    │   ├── metadata.json
    │   ├── summary.txt
    │   └── logs/
    │       └── APGI_Thermodynamic_Program_Aggregator.log
    ├── APGI_Information_Theoretic_Bandwidth/
    │   ├── results.json
    │   ├── metadata.json
    │   ├── summary.txt
    │   └── logs/
    │       └── APGI_Information_Theoretic_Bandwidth.log
    ├── APGI_Turing_Machine/
    │   ├── results.json
    │   ├── metadata.json
    │   ├── summary.txt
    │   └── logs/
    │       └── APGI_Turing_Machine.log
    └── ... (all other theory modules)
```

## File Naming Conventions

### Standard Output Files

| File | Purpose | Format |
|------|---------|--------|
| `results.json` | Main results data | JSON |
| `metadata.json` | Protocol/module metadata | JSON |
| `summary.txt` | Human-readable summary | Text |
| `logs/{protocol_id}.log` | Execution logs | Text |

### Naming Rules

1. **Protocol IDs**: Use exact protocol identifier (VP_01, FP_01, etc.)
2. **Theory Modules**: Use exact module name (APGI_Thermodynamic_Program_Aggregator)
3. **Timestamps**: Automatically added to results.json
4. **No Duplicates**: Each protocol/module has exactly one output directory

## Usage Examples

### For Validation Protocols

```python
from utils.unified_output_manager import UnifiedOutputManager

# Initialize manager
manager = UnifiedOutputManager()

# Get output directory for VP_01
output_dir = manager.get_protocol_output_dir("VP_01")

# Save results
results = {
    "protocol_id": "VP_01",
    "status": "pass",
    "tests_passed": 5,
    "tests_failed": 0,
    "execution_time": 12.34,
    "details": {
        "test_1": {"passed": True, "message": "Test passed"},
        "test_2": {"passed": True, "message": "Test passed"},
    }
}
manager.save_results("VP_01", results)

# Save metadata
metadata = {
    "version": "1.0.0",
    "author": "APGI Framework",
    "description": "Synthetic EEG ML Classification",
    "data_source": "synthetic",
    "sample_size": 100,
}
manager.save_metadata("VP_01", metadata)

# Save summary
summary = """
Validation Protocol VP_01: Synthetic EEG ML Classification
===========================================================

Objective: Validate APGI predictions on synthetic EEG data using ML classification

Results:
- Classification accuracy: 92.5%
- Precision: 0.91
- Recall: 0.93
- F1-score: 0.92

Conclusion: APGI predictions validated on synthetic EEG data.
"""
manager.save_summary("VP_01", summary)

# Get logger for protocol
logger = manager.get_logger_for_protocol("VP_01")
logger.info("Protocol execution started")
logger.info("Processing data...")
logger.info("Protocol execution completed")
```

### For Falsification Protocols

```python
# Same as validation, but with FP_XX protocol IDs
manager = UnifiedOutputManager()

results = {
    "protocol_id": "FP_01",
    "status": "pass",
    "falsification_criteria": {
        "criterion_1": {"met": False, "message": "Criterion not met"},
        "criterion_2": {"met": True, "message": "Criterion met"},
    }
}
manager.save_results("FP_01", results)
```

### For Theory Modules

```python
# Theory modules use is_theory=True and module_name parameter
manager = UnifiedOutputManager()

results = {
    "module": "APGI_Thermodynamic_Program_Aggregator",
    "status": "pass",
    "kappa": 4.625e14,
    "scaling_exponent": 0.414,
    "falsification_status": "FALSIFIED",
}
manager.save_results(
    "APGI_Thermodynamic_Program_Aggregator",
    results,
    is_theory=True,
    module_name="APGI_Thermodynamic_Program_Aggregator"
)

# Get logger for theory module
logger = manager.get_logger_for_protocol(
    "APGI_Thermodynamic_Program_Aggregator",
    is_theory=True,
    module_name="APGI_Thermodynamic_Program_Aggregator"
)
```

## Results JSON Schema

### Validation Protocol Results

```json
{
  "protocol_id": "VP_01",
  "timestamp": "2026-05-08T13:45:00.000000",
  "status": "pass",
  "execution_time": 12.34,
  "tests_passed": 5,
  "tests_failed": 0,
  "details": {
    "test_1": {
      "name": "Test Name",
      "passed": true,
      "message": "Test passed",
      "duration": 2.5
    }
  },
  "summary": {
    "total_tests": 5,
    "pass_rate": 1.0,
    "fail_rate": 0.0
  }
}
```

### Falsification Protocol Results

```json
{
  "protocol_id": "FP_01",
  "timestamp": "2026-05-08T13:45:00.000000",
  "status": "pass",
  "execution_time": 15.67,
  "falsification_criteria": {
    "criterion_1": {
      "name": "Criterion Name",
      "met": false,
      "message": "Criterion not met",
      "details": {}
    },
    "criterion_2": {
      "name": "Criterion Name",
      "met": true,
      "message": "Criterion met",
      "details": {}
    }
  },
  "summary": {
    "total_criteria": 2,
    "criteria_met": 1,
    "criteria_not_met": 1,
    "falsification_status": "INCONCLUSIVE"
  }
}
```

### Theory Module Results

```json
{
  "module": "APGI_Thermodynamic_Program_Aggregator",
  "timestamp": "2026-05-08T13:45:00.000000",
  "status": "pass",
  "execution_time": 8.92,
  "results": {
    "landauer_minimum": 0.058413,
    "neural_metabolic_cost": 1.85e15,
    "kappa": 4.625e14,
    "scaling_exponent": 0.414
  },
  "falsification_status": "FALSIFIED",
  "falsification_details": {
    "criterion": "Cross-species scaling exponent",
    "expected": 0.75,
    "observed": 0.414,
    "deviation": 0.336,
    "tolerance": 0.15,
    "message": "Empirical scaling exponent does not match APGI prediction"
  }
}
```

## Metadata JSON Schema

```json
{
  "protocol_id": "VP_01",
  "timestamp": "2026-05-08T13:45:00.000000",
  "output_dir": "/path/to/outputs/validation/VP_01",
  "version": "1.0.0",
  "author": "APGI Framework",
  "description": "Protocol description",
  "data_source": "synthetic|empirical",
  "sample_size": 100,
  "parameters": {
    "param_1": "value_1",
    "param_2": "value_2"
  },
  "dependencies": [
    "numpy",
    "scipy",
    "pymc"
  ]
}
```

## Integration with Existing Scripts

### Validation Protocols (VP_XX)

Update each validation script to use the unified output manager:

```python
# At the beginning of the script
from utils.unified_output_manager import UnifiedOutputManager

manager = UnifiedOutputManager()
logger = manager.get_logger_for_protocol("VP_01")

# ... protocol execution code ...

# At the end of the script
manager.save_results("VP_01", results)
manager.save_metadata("VP_01", metadata)
manager.save_summary("VP_01", summary_text)
```

### Falsification Protocols (FP_XX)

Same pattern as validation protocols:

```python
from utils.unified_output_manager import UnifiedOutputManager

manager = UnifiedOutputManager()
logger = manager.get_logger_for_protocol("FP_01")

# ... protocol execution code ...

manager.save_results("FP_01", results)
manager.save_metadata("FP_01", metadata)
manager.save_summary("FP_01", summary_text)
```

### Theory Modules

Theory modules use the `is_theory=True` parameter:

```python
from utils.unified_output_manager import UnifiedOutputManager

manager = UnifiedOutputManager()
module_name = "APGI_Thermodynamic_Program_Aggregator"
logger = manager.get_logger_for_protocol(
    module_name,
    is_theory=True,
    module_name=module_name
)

# ... module execution code ...

manager.save_results(
    module_name,
    results,
    is_theory=True,
    module_name=module_name
)
manager.save_metadata(
    module_name,
    metadata,
    is_theory=True,
    module_name=module_name
)
manager.save_summary(
    module_name,
    summary_text,
    is_theory=True,
    module_name=module_name
)
```

## Accessing Outputs

### Load Results

```python
manager = UnifiedOutputManager()

# Load validation protocol results
vp_results = manager.load_results("VP_01")

# Load falsification protocol results
fp_results = manager.load_results("FP_01")

# Load theory module results
theory_results = manager.load_results(
    "APGI_Thermodynamic_Program_Aggregator",
    is_theory=True,
    module_name="APGI_Thermodynamic_Program_Aggregator"
)
```

### Get All Outputs

```python
manager = UnifiedOutputManager()

# Get all outputs organized by type
all_outputs = manager.get_all_protocol_outputs()

# Access by type
validation_outputs = all_outputs["validation"]
falsification_outputs = all_outputs["falsification"]
theory_outputs = all_outputs["theory"]
```

### Generate Report

```python
manager = UnifiedOutputManager()

# Generate unified report
report = manager.generate_unified_report()
print(report)

# Save report
with open("unified_report.txt", "w") as f:
    f.write(report)
```

## Cleanup

### Remove Old Outputs

```python
manager = UnifiedOutputManager()

# Remove outputs older than 30 days
removed_count = manager.cleanup_old_outputs(days=30)
print(f"Removed {removed_count} old files")
```

## Best Practices

1. **Always use the UnifiedOutputManager** for all output operations
2. **Save results immediately** after protocol/module execution
3. **Include metadata** with every output for traceability
4. **Use descriptive summaries** for human readability
5. **Log all important events** during execution
6. **Validate JSON** before saving to catch errors early
7. **Use consistent naming** across all protocols and modules
8. **Archive old outputs** periodically to manage disk space

## Migration Checklist

- [ ] Update all VP_XX scripts to use UnifiedOutputManager
- [ ] Update all FP_XX scripts to use UnifiedOutputManager
- [ ] Update all Theory modules to use UnifiedOutputManager
- [ ] Update Master_Validation.py to aggregate outputs
- [ ] Update Master_Falsification.py to aggregate outputs
- [ ] Update Theory_GUI.py to read from unified outputs
- [ ] Update dashboards to read from unified outputs
- [ ] Test all scripts with new output system
- [ ] Verify output directory structure
- [ ] Document any custom output requirements

## Support

For questions or issues with the unified output system, refer to:
- `utils/unified_output_manager.py` - Implementation
- `utils/output_file_patterns.py` - Pattern matching (legacy)
- This guide - Usage documentation

