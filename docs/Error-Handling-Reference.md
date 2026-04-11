# Error Handling API Reference

## Overview

The APGI Framework includes a comprehensive error handling system that provides standardized error messages, consistent exception types, and helpful troubleshooting information.

## Core Classes

### `APGIError`

Base exception class for all APGI framework errors.

```python
from utils.error_handler import APGIError, ErrorSeverity

# Basic usage
raise APGIError(
    message="Something went wrong",
    severity=ErrorSeverity.MEDIUM,
    context={"module": "validation", "function": "run_protocol"},
    suggestion="Check input parameters"
)
```

**Parameters:**

- `message` (str): Error description
- `severity` (ErrorSeverity): Error severity level (LOW, MEDIUM, HIGH, CRITICAL)
- `context` (dict, optional): Additional context information
- `suggestion` (str, optional): Troubleshooting suggestion
- `original_error` (Exception, optional): Original exception that caused this error

### Specialized Error Classes

#### `ValidationError`

For data validation related errors.

```python
from utils.error_handler import ValidationError

raise ValidationError(
    message="Value out of range",
    data_field="threshold",
    context={"value": 1.5, "valid_range": [0.0, 1.0]},
    suggestion="Use a value between 0.0 and 1.0"
)
```

#### `ConfigurationError`

For configuration related errors.

```python
from utils.error_handler import ConfigurationError

raise ConfigurationError(
    message="Invalid configuration parameter",
    config_file="config/default.yaml",
    context={"parameter": "learning_rate", "value": "invalid"},
    suggestion="Use a numeric value for learning_rate"
)
```

#### `ProtocolError`

For falsification protocol related errors.

```python
from utils.error_handler import ProtocolError

raise ProtocolError(
    message="Protocol execution failed",
    protocol_name="Protocol-1",
    context={"step": "initialization", "error_code": 500},
    suggestion="Check protocol configuration and dependencies"
)
```

#### `DataError`

For data loading/processing related errors.

```python
from utils.error_handler import DataError

raise DataError(
    message="Failed to load data file",
    data_source="data_repository/participants.csv",
    context={"file_size": 0, "expected_format": "CSV"},
    suggestion="Ensure file exists and is in CSV format"
)
```

#### `ImportWarning`

For import/dependency related warnings.

```python
from utils.error_handler import ImportWarning

raise ImportWarning(
    message="Optional dependency not available",
    package="plotly",
    suggestion="Install with: pip install plotly"
)
```

## Utility Functions

### `handle_error()`

Standardized error handling utility.

```python
from utils.error_handler import handle_error, ErrorSeverity
import logging

logger = logging.getLogger(__name__)

try:
    # Some operation that might fail
    risky_operation()
except Exception as e:
    apgi_error = handle_error(
        error=e,
        logger=logger,
        reraise=False,  # Don't reraise, handle gracefully
        context={"operation": "risky_operation", "user_id": 123}
    )
    # apgi_error contains standardized error information
```

### `safe_execute()`

Safely execute a function with standardized error handling.

```python
from utils.error_handler import safe_execute

def divide_numbers(a, b):
    return a / b

# Safe execution with default return value on error
result = safe_execute(
    func=divide_numbers,
    args=[10, 2],
    error_message="Division failed",
    default_return=None,  # Return None on error
    logger=logger
)

# Safe execution that raises custom error
try:
    result = safe_execute(
        func=divide_numbers,
        args=[10, 0],
        error_message="Division by zero",
        error_type=ValidationError,
        reraise=True
    )
except ValidationError as e:
    print(f"Validation error: {e}")
```

### `format_error_message()`

Format standardized error messages.

```python
from utils.error_handler import format_error_message

# Use predefined templates
msg = format_error_message(
    "file_not_found",
    file_path="/path/to/missing/file.txt"
)
# Result: "File not found: /path/to/missing/file.txt"

msg = format_error_message(
    "missing_dependency",
    package="torch"
)
# Result: "Missing required dependency: torch"
```

## Error Severity Levels

```python
from utils.error_handler import ErrorSeverity

# Available severity levels
ErrorSeverity.LOW      # Minor issues, warnings
ErrorSeverity.MEDIUM   # Standard errors
ErrorSeverity.HIGH     # Serious errors affecting functionality
ErrorSeverity.CRITICAL # Critical system failures
```

## Standard Error Message Templates

The framework includes predefined error message templates:

```python
# Available templates
ERROR_MESSAGES = {
    "file_not_found": "File not found: {file_path}",
    "invalid_config": "Invalid configuration: {reason}",
    "missing_dependency": "Missing required dependency: {package}",
    "protocol_failed": "Protocol execution failed: {reason}",
    "data_validation": "Data validation failed: {field} - {reason}",
    "import_error": "Import error: {module} - {reason}",
    "permission_denied": "Permission denied: {action}",
    "network_error": "Network error: {operation} - {reason}",
    "timeout": "Operation timed out: {operation} (limit: {timeout}s)",
}
```

## Best Practices

### 1. Use Specific Error Types

```python
# Good
raise ValidationError(
    message="Invalid threshold value",
    data_field="ignition_threshold",
    context={"value": -0.5, "min_value": 0.0}
)

# Avoid
raise ValueError("Invalid threshold")
```

### 2. Provide Context and Suggestions

```python
# Good
raise ProtocolError(
    message="Protocol initialization failed",
    protocol_name="Protocol-1",
    context={
        "missing_config": "simulation_steps",
        "config_file": "config/protocol1.yaml"
    },
    suggestion="Add simulation_steps to protocol configuration"
)

# Avoid
raise RuntimeError("Protocol failed")
```

### 3. Use Safe Execution for Critical Operations

```python
# Good
result = safe_execute(
    func=load_data,
    args=[file_path],
    error_message="Failed to load participant data",
    error_type=DataError,
    default_return=None,
    logger=logger
)

if result is None:
    # Handle error gracefully
    return {"status": "error", "message": "Data loading failed"}

# Avoid
try:
    result = load_data(file_path)
except:
    return None  # Silent failure
```

### 4. Log Errors Appropriately

```python
import logging

logger = logging.getLogger(__name__)

try:
    operation()
except Exception as e:
    # This will log with appropriate level based on severity
    handle_error(e, logger=logger, context={"user": "admin"})
```

## Visual Troubleshooting Flows

The following flowcharts map specific APGIError classes to the relevant sections of the incident-response-playbook.md:

### APGIError → Incident Response Mapping

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  APGIError      │     │  Severity       │     │  Playbook       │
│  Class          │────▶│  Level          │────▶│  Section        │
└─────────────────┘     └─────────────────┘     └─────────────────┘

APGIError (base)        → MEDIUM     → incident-response-playbook.md#P3
├── ValidationError     → MEDIUM     → incident-response-playbook.md#P3
├── ConfigurationError  → MEDIUM     → incident-response-playbook.md#P3
├── ProtocolError       → HIGH       → incident-response-playbook.md#P2
├── DataError           → MEDIUM     → incident-response-playbook.md#P3
└── ImportWarning       → LOW        → incident-response-playbook.md#P4
```

### Troubleshooting Decision Tree

```text
                    ┌─────────────────┐
                    │   Error Raised  │
                    │   (APGIError)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐  ┌────▼─────┐  ┌─────▼──────┐
        │ CRITICAL  │  │  HIGH    │  │  MEDIUM    │
        │ P1 (≥15)  │  │  P2 (1hr)│  │  P3 (4hr)  │
        └─────┬─────┘  └────┬─────┘  └─────┬──────┘
              │             │              │
        ┌─────▼─────┐  ┌────▼─────┐  ┌─────▼──────┐
        │ Framework │  │ Protocol│  │ Validation │
        │ Failure   │  │ Logic   │  │ or Config  │
        │           │  │ Error   │  │ Issue      │
        └───────────┘  └─────────┘  └────────────┘
```

### Common Error Patterns & Playbook Sections

| Error Class | Example Scenario | Playbook Section | Response Time |
| ----------- | ---------------- | ---------------- | --------------- |
| `ValidationError` | Parameter out of range (θ₀ > 1.0) | P3 - Medium | 4 hours |
| `ConfigurationError` | Missing API key for PyMC | P3 - Medium | 4 hours |
| `ProtocolError` | FP-01 assertion failure | P2 - High | 1 hour |
| `DataError` | OpenNeuro download failure | P3 - Medium | 4 hours |
| `ImportWarning` | Optional dependency missing | P4 - Low | 24 hours |

### Severity Escalation Rules

1. **Auto-escalate to P2 (High)** if:
   - Same error occurs >3 times within 1 hour
   - Error blocks framework falsification (Condition A/B evaluation)
   - Error affects >2 protocols simultaneously

2. **Auto-escalate to P1 (Critical)** if:
   - Framework crashes during validation
   - Mathematical consistency check fails (FP-07)
   - Data corruption detected in empirical datasets

### Diagnostic Quick Reference

```bash
# Check error frequency
python utils/error_recovery.py --summary --hours=1

# View playbook section
less docs/incident-response-playbook.md | grep -A 20 "P2 - High"

# Trigger escalation
python utils/escalation_trigger.py --error-id=<ERROR_ID> --severity=<NEW_LEVEL>
```

---

## Integration with Existing Code

To update existing code to use the new error handling:

```python
# Before
def load_config(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Invalid YAML: {e}")
        return None

# After
from utils.error_handler import ConfigurationError, safe_execute

def load_config(file_path):
    def _load_config():
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)

    return safe_execute(
        func=_load_config,
        error_message="Failed to load configuration",
        error_type=ConfigurationError,
        context={"config_file": file_path},
        default_return=None
    )
```
