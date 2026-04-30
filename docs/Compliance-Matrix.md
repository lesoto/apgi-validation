# APGI Compliance Control Matrix

This document maps APGI framework controls to formal compliance standards including ISO 27001, SOC 2, and NIST CSF, and outlines their corresponding code implementations and verification tests.

## Code & Test Implementation Mapping

| Control Category | Requirement | Implementation Module | Verification Test |
| :--- | :--- | :--- | :--- |
| **Access Control** | Role-based Access Control (RBAC) | `utils/auth_adapter.py` | `tests/test_auth_adapter.py` |
| **Access Control** | Least Privilege Default | `utils/auth_adapter.py` (Default: GUEST) | `tests/test_auth_adapter.py` |
| **Data Retention** | Configurable Retention Policies | `utils/backup_manager.py` | `tests/test_backup_manager.py` |
| **Data Retention** | Automated Cleanup/Minimization | `utils/data_protection.py` | `tests/test_data_protection.py` |
| **Auditability** | Persistent Action Logging | `utils/persistent_audit_logger.py` | `tests/test_persistent_audit_logger.py` |
| **Auditability** | Security Audit Trails | `utils/security_audit_logger.py` | `tests/test_security_audit.py` |
| **Auditability** | Action Immutable Export | `utils/audit_threshold_leakage.py` | `tests/test_audit_leakage.py` |
| **Encryption** | Data at Rest (Keys) | `utils/secure_key_manager.py` | `tests/test_secure_key_manager.py` |
| **Encryption** | Secret Policy Enforcement | `utils/secret_policy_enforcer.py` | `tests/test_env_var_initialization.py` |
| **Data Integrity** | TOCTOU Mitigation | `utils/toctou_mitigation.py` | `tests/test_toctou.py` |
| **Data Integrity** | Dependency Scanning & SBOMs | `utils/dependency_scanner.py` | `tests/test_dependency_scanner.py` |

## Enforcement Strategy
- **CI/CD Gates:** All mapped tests run sequentially in the `pytest` pipeline.
- **Fail-on-Regression:** Any failing test linked to a compliance control blocks the deployment.
- **Auditor Visibility:** The generated SBOM and test coverage reports serve as verifiable artifacts for external auditors.

## Control Framework Mapping

| Control ID | Control Description | ISO 27001 | SOC 2 | NIST CSF | Implementation Status |
| --- | --- | --- | --- | --- | --- |
| AC-01 | Access Control Policy | A.9 | CC6.1 | PR.AC | Implemented |
| AC-02 | User Access Management | A.9.2 | CC6.2 | PR.AC-1 | Implemented |
| AC-03 | Privileged Access Control | A.9.4 | CC6.7 | PR.AC-6 | Implemented |
| AC-04 | Authentication Mechanisms | A.9.3 | CC6.8 | PR.AC-7 | Implemented |
| AC-05 | Deny-by-Default Security | A.9.1 | CC6.1 | PR.AC-1 | Implemented |
| AU-01 | Audit Logging | A.12.7 | CC6.6 | AU.2 | Implemented |
| AU-02 | Audit Trail Protection | A.12.7.1 | CC6.6 | AU.3 | Implemented |
| AU-03 | Security Audit Automation | A.12.7 | CC6.6 | AU.6 | Implemented |
| CM-01 | Configuration Management | A.12.1 | CC8.1 | CM-2 | Partial |
| CM-02 | Change Control | A.12.1.2 | CC8.2 | CM-3 | Partial |
| CM-03 | Vulnerability Management | A.12.6 | CC8.6 | CM-8 | Implemented |
| DS-01 | Data Classification | A.8.2 | CC6.1 | DS-2 | Partial |
| DS-02 | Data Retention Policy | A.8.2.1 | CC6.1 | DS-2 | Implemented |
| DS-03 | Secure Data Deletion | A.8.2.2 | CC6.1 | DS-2 | Implemented |
| DS-04 | Data Access Logging | A.12.7 | CC6.6 | AU.2 | Implemented |
| DR-01 | Backup and Recovery | A.12.3 | CC8.3 | PR.DS-4 | Implemented |
| DR-02 | Backup Encryption | A.12.3.1 | CC8.3 | PR.DS-4 | Implemented |
| IR-01 | Incident Response Plan | A.16.1 | CC7.2 | IR-4 | Partial |
| IR-02 | Incident Reporting | A.16.1.2 | CC7.3 | IR-6 | Partial |
| SC-01 | System Hardening | A.12.2 | CC8.4 | PR.IP-1 | Partial |
| SC-02 | Secure Development | A.14.2 | CC8.5 | PR.IP-3 | Implemented |
| SC-03 | Dependency Scanning | A.14.2.1 | CC8.5 | PR.IP-3 | Implemented |

## Control Descriptions

### Access Control (AC)

#### AC-01: Access Control Policy
- Framework has defined security context with role-based access
- `utils/security_logging_integration.py` and `utils/auth_adapter.py` implement SecurityContext and RBAC
- Status: Implemented

#### AC-02: User Access Management
- Role-based authorization through SecurityContext
- Deny-by-default with explicit role assignment
- Status: Implemented

#### AC-03: Privileged Access Control
- No admin bypass mechanism
- Explicit role requirements for sensitive operations
- Status: Implemented

#### AC-04: Authentication Mechanisms
- Path validation and traversal protection
- Module loading security checks
- Status: Implemented

#### AC-05: Deny-by-Default Security
- Removed admin role bypass in `_require_role()`
- Default SecurityContext has empty roles set or GUEST role
- Status: Implemented

### Audit and Accountability (AU)

#### AU-01: Audit Logging
- Structured JSON logging with correlation IDs
- `utils/logging_config.py` and `utils/persistent_audit_logger.py` implement JSON formatting and persistence
- Status: Implemented

#### AU-02: Audit Trail Protection
- Secure audit logger in `utils/security_audit_logger.py`
- Immutable audit trail for security events, exported via `utils/compliance_export.py`
- Status: Implemented

#### AU-03: Security Audit Automation
- CI workflow `.github/workflows/security.yml` with automated SAST/DAST
- Bandit, Safety, Semgrep, pip-audit integration
- Status: Implemented

### Configuration Management (CM)

#### CM-01: Configuration Management
- `utils/config_manager.py` for centralized configuration
- Schema validation in `config/config_schema.json`
- Status: Partial - needs CI integration

#### CM-02: Change Control
- Git-based version control
- Manual change review process
- Status: Partial - needs automated change approval

#### CM-03: Vulnerability Management
- Automated dependency scanning in CI
- pip-audit for known vulnerabilities
- Status: Implemented

### Data Security (DS)

#### DS-01: Data Classification
- Data protection workflows in `utils/data_protection.py` with PII tagging logic
- Status: Partial - needs full classification schema

#### DS-02: Data Retention Policy
- `apply_retention_policy()` function with configurable max age
- Age distribution tracking
- Status: Implemented

#### DS-03: Secure Data Deletion
- `secure_delete()` with multi-pass overwriting
- SHA-256 hashing for verification
- PII minimization operations
- Status: Implemented

#### DS-04: Data Access Logging
- `log_data_access()` for audit trail
- User ID, purpose, and access type tracking
- Status: Implemented

### Disaster Recovery (DR)

#### DR-01: Backup and Recovery
- `utils/backup_manager.py` with HMAC validation
- Automated backup CLI commands
- Status: Implemented

#### DR-02: Backup Encryption
- Encrypted backup keys in `.keys/` directory
- HMAC validation for integrity
- Status: Implemented

### Incident Response (IR)

#### IR-01: Incident Response Plan
- `docs/Incident-response-playbook.md` exists
- Status: Partial - needs testing and updates

#### IR-02: Incident Reporting
- Manual incident reporting process
- Status: Partial - needs automated alerting

### System Security (SC)

#### SC-01: System Hardening
- Path security validation in `utils/path_security.py` and wrapped in `utils/file_ops.py`
- TOCTOU mitigation
- Secret policy enforcer integrated on startup
- Status: Partial - needs full hardening checklist

#### SC-02: Secure Development
- SAST/DAST in CI pipeline
- Static analysis with Bandit and Semgrep
- Status: Implemented

#### SC-03: Dependency Scanning
- `utils/dependency_scanner.py` for dependency analysis with SBOM export
- Automated vulnerability scanning with severity thresholds
- Status: Implemented

## Compliance Evidence

### CI/CD Integration

#### Security Workflow (`.github/workflows/security.yml`)
- Runs on push and pull_request
- SAST: Bandit, Semgrep
- Dependency scanning: pip-audit, Safety
- Reports uploaded as artifacts

### Code Evidence

#### Security Implementation
- `utils/security_logging_integration.py` / `utils/auth_adapter.py` - Authorization middleware
- `utils/security_audit_logger.py` / `utils/persistent_audit_logger.py` - Audit logging
- `utils/path_security.py` / `utils/file_ops.py` - Path validation and safe IO
- `utils/data_protection.py` - Data retention and deletion
- `utils/errors.py` - Typed error taxonomy
- `utils/dependency_scanner.py` - SBOM generation

#### Performance and Reliability
- `utils/performance_optimizer.py` - p50/p95/p99 latency tracking
- `utils/cache_manager.py` - Cache SLO monitoring & deterministic versioning
- `scripts/profile_optimization.py` - Benchmark corpus

#### Documentation Evidence
- `docs/Incident-response-playbook.md` - Incident response procedures
- `docs/Architecture.md` - System architecture
- `docs/Testing-Coverage.md` - Testing procedures
- `pytest.ini` - Quality gates (95% coverage)

## Gap Analysis

### High Priority Gaps
#### Data Classification Schema (DS-01)
- Need formal data classification levels
- Implement classification labels in data_protection.py

#### Change Control Automation (CM-02)
- Need automated change approval workflow
- Integrate with pull request process

#### Incident Alerting (IR-02)
- Need automated incident notification
- Integrate with monitoring system

### Medium Priority Gaps
#### System Hardening Checklist (SC-01)
- Document and implement hardening procedures
- Regular security audits

#### Configuration CI Integration (CM-01)
- Validate config changes in CI
- Automated config testing

#### Incident Response Testing (IR-01)
- Regular incident response drills
- Update playbook based on lessons learned

## Compliance Status Summary

- **Fully Implemented**: 13 controls
- **Partially Implemented**: 8 controls
- **Not Implemented**: 0 controls

### Overall Compliance
~75% (weighted by control priority)

## Next Steps

- Address high priority gaps (data classification, change control automation)
- Implement automated incident alerting
- Create system hardening checklist
- Regular compliance audits (quarterly)
- Update compliance matrix based on new standards
