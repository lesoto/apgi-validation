# APGI Privacy Threat Model

## Threat Categories
1. Unauthorized read/write access to protocol artifacts.
2. Leakage of sensitive values via logs.
3. Path traversal and unsafe module import execution.
4. Insufficient deletion/audit lineage.

## Mitigations
- Mandatory audit decorators and role checks.
- Secret redaction in log pipeline.
- Path normalization and traversal rejection.
- Retention/deletion policy + compliance matrix governance references.
