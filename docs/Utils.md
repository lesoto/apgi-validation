# Essential Scripts (Core Functionality)

## Core Package

- `__init__.py` - Package initialization with 80+ module exports, defines centralized access patterns for all utility modules
- `constants.py` - Core model parameters and system defaults including ModelParameters dataclass, ThermodynamicConfig, and NeuralDataDefaults
- `config_schema.json` - JSON Schema for configuration validation
- `config_manager.py` - Comprehensive configuration management with YAML/JSON support, validation, versioning, and persistence
- `apgi_config.py` - APGI-specific configuration wrapper with profile management and runtime parameter access
- `apgi_engine.py` - Core APGI computation engine for surprise accumulation and ignition dynamics

## Protocol Management

- `protocol_registry.py` - Protocol registration system with metadata for 15 validation and 12 falsification protocols
- `protocol_schema.py` - Protocol result schemas with standardized data structures
- `protocol_contracts.py` - Protocol interface contracts and validation requirements
- `metadata_standardizer.py` - Metadata standardization across protocols
- `update_protocol_metadata.py` - Protocol metadata synchronization
- `interprotocol_schema.py` - Cross-protocol communication schemas

## Error Handling & Recovery

- `error_handler.py` - Centralized error handling with APGIError class, error categorization, and severity levels
- `error_recovery.py` - Automatic error recovery with retry logic and fallback mechanisms
- `crash_recovery.py` - System crash recovery with state preservation
- `signal_handler.py` - OS signal handling for graceful shutdown
- `timeout_handler.py` - Execution timeout enforcement with configurable limits

## Logging & Monitoring

- `logging_config.py` - Structured logging with rotation, formatting, and multiple output streams
- `log_analysis_tools.py` - Comprehensive log analysis with pattern detection, statistics, and filtering
- `monitoring_system.py` - System health monitoring for validation components
- `progress_estimator.py` - Progress tracking for long-running operations

## Input Validation & Security

- `input_validation.py` - Input sanitization and API parameter validation
- `input_sanitizer.py` - Advanced input sanitization for security
- `path_security.py` - Path traversal prevention and file path validation

## Falsification & Validation

- `falsification_thresholds.py` - Falsification criteria constants, defines threshold values for model falsification tests
- `criteria_registry.py` - Falsification criteria definitions, manages registration and execution of falsification criteria
- `meta_falsification.py` - Framework-level falsification, orchestrates falsification across multiple model components
- `shared_falsification.py` - Common falsification logic, provides reusable falsification utilities and helper functions
- `algorithmic_verification.py` - Core equation verification, validates mathematical equations and computational implementations
- `analytical_solutions.py` - Analytical solutions, provides mathematical solutions for validation benchmarks
- `audit_threshold_leakage.py` - Threshold leakage audit, detects and prevents threshold value leaks in validation
- `threshold_registry.py` - Threshold management, maintains and updates threshold values for various criteria
- `threshold_lint.py` - Threshold validation linter, checks threshold configurations for consistency and correctness
- `statistical_tests.py` - Shared statistical functions, implements common statistical tests and calculations
- `ordinal_logistic_regression.py` - Specialized statistical model, implements ordinal logistic regression for categorical outcomes
- `bayesian_model_comparison.py` - Bayesian model comparison utilities, computes Bayes factors and model evidence
- `cross_protocol_consistency.py` - Cross-protocol consistency checks, validates coherence across validation protocols
- `seven_standards_registry.py` - Seven standards compliance registry, tracks adherence to scientific validation standards
- `validation_pipeline_connector.py` - Pipeline connection, integrates validation steps into cohesive workflows
- `validation_runner.py` - Validation execution runner, orchestrates execution of validation protocols
- `validation_falsification_consistency.py` - Consistency validation, ensures alignment between validation and falsification results
- `interprotocol_schema.py` - Inter-protocol schema, defines schemas for cross-protocol communication
- `update_protocol_metadata.py` - Metadata updater, standardizes and updates protocol metadata
- `data_validation.py` - Data quality validation, ensures data integrity and format compliance before processing
- `data_quality_assessment.py` - Data quality assessment, evaluates and reports on overall data quality metrics

## Security & Audit

- `secure_key_manager.py` - Secure key management, handles encryption keys and secrets with hardware-backed security
- `key_rotation_manager.py` - Key rotation automation, manages scheduled rotation of encryption keys
- `persistent_audit_logger.py` - Persistent audit logging, maintains tamper-resistant audit trails
- `security_audit_logger.py` - Security audit logging, tracks security-relevant events and access patterns
- `security_logging_integration.py` - Security logging integration, integrates security events with logging systems
- `toctou_mitigation.py` - TOCTOU attack mitigation, prevents time-of-check-time-of-use vulnerabilities
- `dependency_scanner.py` - Dependency security scanner, checks for known vulnerabilities in dependencies

## Dashboard & Monitoring

- `performance_dashboard.py` - Web dashboard for monitoring, provides real-time visualization of system metrics
- `historical_dashboard.py` - Historical data dashboard, displays trends and historical performance metrics
- `static_dashboard_generator.py` - Static dashboard generator, creates offline dashboard reports
- `dashboard_integration.py` - Dashboard integration utilities, connects monitoring to external systems
- `monitoring_system.py` - System monitoring, tracks health and performance of validation components
- `performance_optimizer.py` - Performance optimization, identifies and resolves performance bottlenecks
- `progress_estimator.py` - Progress tracking, estimates and reports progress for long-running operations
- `protocol_visualization.py` - Protocol visualization, creates visual representations of validation results

## Data Processing & Generation

### Batch Processing

- `batch_processor.py` - Multi-threaded batch processing with TQDM progress tracking, error recovery, and checkpoint support
  - **Computational Complexity**: O(n × m) where n = batches, m = batch size
  - Memory: O(m) per batch, I/O: O(n) sequential
  - For 100k trials × 64-channel EEG (1GB): ~15 min on 8-core server
- `batch_config.py` - Batch processing configuration management

### Data Generation

- `sample_data_generator.py` - Synthetic dataset generation for testing and validation
- `empirical_data_generators.py` - Empirical data generation from theoretical distributions
- `generate_empirical_data.py` - Main script for generating validation datasets

### Signal Processing

- `eeg_processing.py` - EEG signal filtering, artifact removal, and feature extraction
- `eeg_simulator.py` - Synthetic EEG signal generation
- `spectral_analysis.py` - Frequency-domain analysis and spectral decomposition
- `hrf_utils.py` - Hemodynamic response function calculations for fMRI

### Data Management

- `data_collector.py` - Multi-source data aggregation
- `preprocessing_pipelines.py` - Data cleaning, normalization, and transformation
- `data_validation.py` - Data integrity and format validation
- `data_quality_assessment.py` - Data quality metrics and reporting
- `cache_manager.py` - Computed result caching for performance
- `parameter_validator.py` - Model parameter range validation
- `genome_data_extractor.py` - VP-5 genome data processing

## Backup & Recovery

- `backup_manager.py` - Comprehensive backup/restore with encryption, compression, and CLI integration

## Performance & Optimization

- `performance_optimizer.py` - Bottleneck identification and resolution
- `performance_gates.py` - Performance threshold enforcement
- `performance_governance.py` - Performance policy management

## Framework Verification

- `verify_framework_status.py` - Framework status verification for all protocols and components

## Utility Summary

**Total Utilities**: 80+ modules organized across:

- Core functionality (15 modules)
- Falsification & Validation (22 modules)
- Security & Audit (7 modules)
- Dashboard & Monitoring (8 modules)
- Data Processing (14 modules)
- Backup & Recovery (1 module)
- Performance (3 modules)
