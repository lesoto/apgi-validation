# Essential Scripts (Core Functionality)

__init__.py - Package initialization and imports, defines module-level exports and dependencies
constants.py - Core model parameters and system defaults, defines global constants for APGI framework
config_manager.py - Configuration management system, handles loading, validation, and persistence of configuration files
apgi_config.py - APGI-specific configuration, manages framework-specific settings and parameters
protocol_registry.py - Protocol registration system, maintains registry of available validation and falsification protocols
error_handler.py - Centralized error handling, provides consistent error logging and exception management
error_recovery.py - Error recovery mechanisms, handles graceful recovery from operational failures
crash_recovery.py - Crash recovery system, provides automatic recovery from unexpected failures
signal_handler.py - System signal handling, manages OS signals for graceful shutdown and cleanup
logging_config.py - Logging system configuration, sets up structured logging with rotation and formatting
log_analysis_tools.py - Log analysis utilities, parses and analyzes system logs for insights and debugging
timeout_handler.py - Timeout management, prevents indefinite hangs by enforcing execution time limits
input_validation.py - Input validation utilities, validates user inputs and API parameters
path_security.py - Path security utilities, validates and sanitizes file paths for safe operations

## Falsification & Validation

falsification_thresholds.py - Falsification criteria constants, defines threshold values for model falsification tests
criteria_registry.py - Falsification criteria definitions, manages registration and execution of falsification criteria
meta_falsification.py - Framework-level falsification, orchestrates falsification across multiple model components
shared_falsification.py - Common falsification logic, provides reusable falsification utilities and helper functions
algorithmic_verification.py - Core equation verification, validates mathematical equations and computational implementations
threshold_registry.py - Threshold management, maintains and updates threshold values for various criteria
threshold_lint.py - Threshold validation linter, checks threshold configurations for consistency and correctness
statistical_tests.py - Shared statistical functions, implements common statistical tests and calculations
ordinal_logistic_regression.py - Specialized statistical model, implements ordinal logistic regression for categorical outcomes
bayesian_model_comparison.py - Bayesian model comparison utilities, computes Bayes factors and model evidence
cross_protocol_consistency.py - Cross-protocol consistency checks, validates coherence across validation protocols
seven_standards_registry.py - Seven standards compliance registry, tracks adherence to scientific validation standards
validation_pipeline_connector.py - Pipeline connection, integrates validation steps into cohesive workflows
data_validation.py - Data quality validation, ensures data integrity and format compliance before processing
data_quality_assessment.py - Data quality assessment, evaluates and reports on overall data quality metrics

## Security & Audit

secure_key_manager.py - Secure key management, handles encryption keys and secrets with hardware-backed security
key_rotation_manager.py - Key rotation automation, manages scheduled rotation of encryption keys
persistent_audit_logger.py - Persistent audit logging, maintains tamper-resistant audit trails
security_audit_logger.py - Security audit logging, tracks security-relevant events and access patterns
security_logging_integration.py - Security logging integration, integrates security events with logging systems
toctou_mitigation.py - TOCTOU attack mitigation, prevents time-of-check-time-of-use vulnerabilities
dependency_scanner.py - Dependency security scanner, checks for known vulnerabilities in dependencies

## Dashboard & Monitoring

performance_dashboard.py - Web dashboard for monitoring, provides real-time visualization of system metrics
historical_dashboard.py - Historical data dashboard, displays trends and historical performance metrics
static_dashboard_generator.py - Static dashboard generator, creates offline dashboard reports
dashboard_integration.py - Dashboard integration utilities, connects monitoring to external systems
monitoring_system.py - System monitoring, tracks health and performance of validation components
performance_optimizer.py - Performance optimization, identifies and resolves performance bottlenecks
progress_estimator.py - Progress tracking, estimates and reports progress for long-running operations

## Data Processing

sample_data_generator.py - Synthetic data generation, creates realistic test data for validation and testing
preprocessing_pipelines.py - Data preprocessing, implements data cleaning, normalization, and transformation pipelines
data_collector.py - Data collection utilities, aggregates data from multiple sources for analysis
eeg_processing.py - EEG signal processing, handles EEG data filtering, artifact removal, and feature extraction
eeg_simulator.py - EEG simulation, generates synthetic EEG signals for testing
spectral_analysis.py - Spectral analysis tools, performs frequency-domain analysis of signals
genome_data_extractor.py - VP-5 specific utility, extracts and processes VP-5 genome data
batch_processor.py - Parallel processing, enables efficient execution of multiple tasks concurrently
batch_config.py - Batch configuration, manages configuration for batch processing operations
cache_manager.py - Caching system, manages temporary storage of computed results to improve performance
parameter_validator.py - Parameter validation, ensures model parameters are within valid ranges

## Backup & Recovery

backup_manager.py - Backup/restore functionality, handles data backup creation and recovery operations
