#!/usr/bin/env python3
"""
Centralized Error Handling System for APGI Framework
================================================

Provides standardized error messages, error categories, and centralized
error handling with proper logging and user-friendly messages.
"""

import functools
import sys
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union

from utils.logging_config import apgi_logger


class ErrorSeverity(Enum):
    """Error severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ErrorCategory(Enum):
    """Error categories for better organization."""

    CONFIGURATION = "CONFIGURATION"
    VALIDATION = "VALIDATION"
    SIMULATION = "SIMULATION"
    DATA = "DATA"
    IO = "IO"
    NETWORK = "NETWORK"
    MEMORY = "MEMORY"
    PERMISSION = "PERMISSION"
    IMPORT = "IMPORT"
    RUNTIME = "RUNTIME"
    USER_INPUT = "USER_INPUT"
    BACKUP = "BACKUP"
    CACHE = "CACHE"


@dataclass
class ErrorInfo:
    """Structured error information."""

    category: ErrorCategory
    severity: ErrorSeverity
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[list] = None
    user_action: Optional[str] = None


class APGIError(Exception):
    """Base exception class for APGI framework."""

    def __init__(self, error_info: ErrorInfo):
        self.error_info = error_info
        super().__init__(error_info.message)

    def __str__(self) -> str:
        return f"[{self.error_info.severity.value}] {self.error_info.category.value}: {self.error_info.message}"


class ErrorHandler:
    """Centralized error handling system."""

    # Error message templates
    ERROR_TEMPLATES = {
        # Configuration errors
        ErrorCategory.CONFIGURATION: {
            ErrorSeverity.CRITICAL: {
                "INVALID_CONFIG": "Configuration file is corrupted or invalid: {details}",
                "MISSING_CONFIG": "Required configuration file not found: {file_path}",
                "INVALID_PARAMETER": "Invalid configuration parameter '{param}': {details}",
                "CONFIG_VERSION_MISMATCH": "Configuration version mismatch. Expected: {expected}, Found: {found}",
            },
            ErrorSeverity.HIGH: {
                "CONFIG_LOAD_FAILED": "Failed to load configuration: {details}",
                "PARAMETER_OUT_OF_RANGE": "Parameter '{param}' value {value} is out of range [{min}, {max}]",
                "DEPENDENCY_MISSING": "Missing required dependency: {dependency}",
            },
            ErrorSeverity.MEDIUM: {
                "UNKNOWN_PARAMETER": "Unknown configuration parameter: {param}",
                "DEFAULT_VALUE_USED": "Using default value for '{param}': {default_value}",
            },
        },
        # Validation errors
        ErrorCategory.VALIDATION: {
            ErrorSeverity.HIGH: {
                "VALIDATION_FAILED": "Validation protocol failed: {protocol}",
                "CRITICAL_VALIDATION_ERROR": "Critical validation error: {details}",
            },
            ErrorSeverity.MEDIUM: {
                "VALIDATION_WARNING": "Validation warning: {details}",
                "PARTIAL_VALIDATION": "Partial validation success: {passed_count}/{total_count} tests passed",
            },
            ErrorSeverity.LOW: {
                "VALIDATION_INFO": "Validation information: {details}",
            },
        },
        # Simulation errors
        ErrorCategory.SIMULATION: {
            ErrorSeverity.CRITICAL: {
                "SIMULATION_CRASHED": "Simulation crashed: {details}",
                "NUMERICAL_INSTABILITY": "Numerical instability detected in simulation",
            },
            ErrorSeverity.HIGH: {
                "CONVERGENCE_FAILED": "Simulation failed to converge: {details}",
                "INVALID_PARAMETERS": "Invalid simulation parameters: {details}",
            },
            ErrorSeverity.MEDIUM: {
                "SLOW_CONVERGENCE": "Slow convergence detected: {details}",
                "SIMULATION_WARNING": "Simulation warning: {details}",
            },
        },
        # Data errors
        ErrorCategory.DATA: {
            ErrorSeverity.CRITICAL: {
                "DATA_CORRUPTION": "Data corruption detected: {details}",
                "DATA_LOSS": "Data loss detected: {details}",
            },
            ErrorSeverity.HIGH: {
                "INVALID_DATA_FORMAT": "Invalid data format: {format}",
                "MISSING_REQUIRED_FIELDS": "Missing required data fields: {fields}",
                "DATA_VALIDATION_FAILED": "Data validation failed: {details}",
            },
            ErrorSeverity.MEDIUM: {
                "MISSING_OPTIONAL_FIELDS": "Missing optional data fields: {fields}",
                "DATA_INCONSISTENCY": "Data inconsistency detected: {details}",
            },
            ErrorSeverity.LOW: {
                "DATA_QUALITY_WARNING": "Data quality warning: {details}",
            },
        },
        # I/O errors
        ErrorCategory.IO: {
            ErrorSeverity.CRITICAL: {
                "FILE_SYSTEM_ERROR": "Critical file system error: {details}",
            },
            ErrorSeverity.HIGH: {
                "FILE_NOT_FOUND": "File not found: {file_path}",
                "PERMISSION_DENIED": "Permission denied: {file_path}",
                "DISK_FULL": "Insufficient disk space",
            },
            ErrorSeverity.MEDIUM: {
                "FILE_READ_ERROR": "Error reading file: {file_path}",
                "FILE_WRITE_ERROR": "Error writing file: {file_path}",
            },
        },
        # Memory errors
        ErrorCategory.MEMORY: {
            ErrorSeverity.CRITICAL: {
                "OUT_OF_MEMORY": "Out of memory: {details}",
            },
            ErrorSeverity.HIGH: {
                "MEMORY_LEAK": "Potential memory leak detected: {details}",
            },
            ErrorSeverity.MEDIUM: {
                "HIGH_MEMORY_USAGE": "High memory usage: {usage_mb:.1f} MB",
            },
        },
        # Import errors
        ErrorCategory.IMPORT: {
            ErrorSeverity.HIGH: {
                "MODULE_NOT_FOUND": "Required module not found: {module}",
                "DEPENDENCY_ERROR": "Dependency error: {details}",
            },
            ErrorSeverity.MEDIUM: {
                "OPTIONAL_MODULE_MISSING": "Optional module missing: {module}. Some features may be limited.",
            },
        },
        # User input errors
        ErrorCategory.USER_INPUT: {
            ErrorSeverity.HIGH: {
                "INVALID_ARGUMENT": "Invalid argument: {argument}. {details}",
                "REQUIRED_ARGUMENT_MISSING": "Required argument missing: {argument}",
            },
            ErrorSeverity.MEDIUM: {
                "INVALID_VALUE": "Invalid value for {argument}: {value}. {details}",
                "DEPRECATED_ARGUMENT": "Deprecated argument: {argument}. Use {alternative} instead.",
            },
        },
        # Backup errors
        ErrorCategory.BACKUP: {
            ErrorSeverity.HIGH: {
                "BACKUP_FAILED": "Backup operation failed: {details}",
                "BACKUP_CORRUPTED": "Backup file is corrupted: {backup_id}",
            },
            ErrorSeverity.MEDIUM: {
                "BACKUP_WARNING": "Backup warning: {details}",
                "PARTIAL_BACKUP": "Partial backup completed: {details}",
            },
        },
        # Cache errors
        ErrorCategory.CACHE: {
            ErrorSeverity.MEDIUM: {
                "CACHE_ERROR": "Cache operation failed: {details}",
                "CACHE_MISS": "Cache miss: {details}",
            },
            ErrorSeverity.LOW: {
                "CACHE_WARNING": "Cache warning: {details}",
            },
        },
    }

    def __init__(self):
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.error_handlers: Dict[ErrorCategory, Callable] = {}

    def format_error(
        self, category: ErrorCategory, severity: ErrorSeverity, code: str, **kwargs
    ) -> str:
        """Format error message using templates."""
        try:
            template = self.ERROR_TEMPLATES[category][severity][code]
            return template.format(**kwargs)
        except KeyError:
            return f"Unknown error: {category.value}.{severity.value}.{code}: {kwargs}"

    def create_error(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        code: str,
        details: Optional[str] = None,
        suggestions: Optional[list] = None,
        user_action: Optional[str] = None,
        **format_kwargs,
    ) -> ErrorInfo:
        """Create structured error information."""
        message = self.format_error(category, severity, code, **format_kwargs)

        if details:
            format_kwargs["details"] = details

        return ErrorInfo(
            category=category,
            severity=severity,
            code=code,
            message=message,
            details=format_kwargs,
            suggestions=suggestions,
            user_action=user_action,
        )

    def handle_error(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity,
        code: str,
        details: Optional[str] = None,
        suggestions: Optional[list] = None,
        user_action: Optional[str] = None,
        **format_kwargs,
    ) -> APGIError:
        """Handle and log error with standard formatting."""
        # Create error info
        error_info = self.create_error(
            category, severity, code, details, suggestions, user_action, **format_kwargs
        )

        # Count errors by category
        self.error_counts[category] = self.error_counts.get(category, 0) + 1

        # Log error
        log_message = f"[{severity.value}] {category.value}: {error_info.message}"

        if severity == ErrorSeverity.CRITICAL:
            apgi_logger.logger.critical(log_message)
        elif severity == ErrorSeverity.HIGH:
            apgi_logger.logger.error(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            apgi_logger.logger.warning(log_message)
        else:
            apgi_logger.logger.info(log_message)

        # Add details to log if available
        if error_info.details:
            apgi_logger.logger.debug(f"Error details: {error_info.details}")

        # Add suggestions to log if available
        if error_info.suggestions:
            apgi_logger.logger.info(f"Suggestions: {error_info.suggestions}")

        # Call custom handler if registered
        if category in self.error_handlers:
            try:
                self.error_handlers[category](error_info)
            except Exception as e:
                apgi_logger.logger.error(f"Error in custom error handler: {e}")

        return APGIError(error_info)

    def register_handler(
        self, category: ErrorCategory, handler: Callable[[ErrorInfo], None]
    ) -> None:
        """Register custom error handler for a category."""
        self.error_handlers[category] = handler
        apgi_logger.logger.info(f"Registered error handler for category: {category.value}")

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors by category."""
        total_errors = sum(self.error_counts.values())

        return {
            "total_errors": total_errors,
            "by_category": {category.value: count for category, count in self.error_counts.items()},
            "most_common": (
                max(
                    self.error_counts.items(),
                    key=lambda x: x[1],
                    default=(ErrorCategory.RUNTIME, 0),
                )[0].value
                if self.error_counts
                else None
            ),
        }

    def reset_error_counts(self) -> None:
        """Reset error counts."""
        self.error_counts.clear()
        apgi_logger.logger.info("Error counts reset")


# Global error handler instance
error_handler = ErrorHandler()


# Decorator for automatic error handling
def handle_errors(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    code: str = "UNHANDLED_EXCEPTION",
    reraise: bool = True,
):
    """Decorator for automatic error handling."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except APGIError:
                # Re-raise APGI errors as-is
                if reraise:
                    raise
                return None
            except Exception as e:
                # Handle unexpected exceptions
                error = error_handler.handle_error(
                    category=category,
                    severity=severity,
                    code=code,
                    details=str(e),
                    suggestions=["Check function parameters", "Verify input data"],
                    user_action="Review the error details and try again",
                )
                if reraise:
                    raise error
                return None

        return wrapper

    return decorator


# Convenience functions for common error types
def config_error(code: str, **kwargs) -> APGIError:
    """Create configuration error."""
    return error_handler.handle_error(
        ErrorCategory.CONFIGURATION, ErrorSeverity.HIGH, code, **kwargs
    )


def validation_error(code: str, **kwargs) -> APGIError:
    """Create validation error."""
    return error_handler.handle_error(ErrorCategory.VALIDATION, ErrorSeverity.HIGH, code, **kwargs)


def simulation_error(code: str, **kwargs) -> APGIError:
    """Create simulation error."""
    return error_handler.handle_error(ErrorCategory.SIMULATION, ErrorSeverity.HIGH, code, **kwargs)


def data_error(code: str, **kwargs) -> APGIError:
    """Create data error."""
    return error_handler.handle_error(ErrorCategory.DATA, ErrorSeverity.HIGH, code, **kwargs)


def io_error(code: str, **kwargs) -> APGIError:
    """Create I/O error."""
    return error_handler.handle_error(ErrorCategory.IO, ErrorSeverity.HIGH, code, **kwargs)


def user_input_error(code: str, **kwargs) -> APGIError:
    """Create user input error."""
    return error_handler.handle_error(
        ErrorCategory.USER_INPUT, ErrorSeverity.MEDIUM, code, **kwargs
    )


def critical_error(code: str, **kwargs) -> APGIError:
    """Create critical error."""
    return error_handler.handle_error(ErrorCategory.RUNTIME, ErrorSeverity.CRITICAL, code, **kwargs)


# User-friendly error message formatter
def format_user_message(error: APGIError) -> str:
    """Format error message for user display."""
    info = error.error_info

    # Base message
    message = f"❌ {info.message}"

    # Add suggestions if available
    if info.suggestions:
        message += f"\n\n💡 Suggestions:"
        for suggestion in info.suggestions:
            message += f"\n   • {suggestion}"

    # Add user action if available
    if info.user_action:
        message += f"\n\n🔧 Action: {info.user_action}"

    return message


# Error recovery suggestions database
RECOVERY_SUGGESTIONS = {
    "FILE_NOT_FOUND": [
        "Check if the file path is correct",
        "Verify the file exists",
        "Check file permissions",
    ],
    "PERMISSION_DENIED": [
        "Run with appropriate permissions",
        "Check file/directory ownership",
        "Use sudo if necessary",
    ],
    "INVALID_DATA_FORMAT": [
        "Verify data format specification",
        "Check data file headers",
        "Validate data structure",
    ],
    "OUT_OF_MEMORY": [
        "Reduce data size",
        "Close other applications",
        "Increase system memory",
    ],
    "MODULE_NOT_FOUND": [
        "Install missing dependencies",
        "Check Python environment",
        "Verify package installation",
    ],
}


def get_recovery_suggestions(error_code: str) -> list:
    """Get recovery suggestions for error code."""
    return RECOVERY_SUGGESTIONS.get(error_code, ["Contact support if issue persists"])


if __name__ == "__main__":
    # Test error handling system
    print("Testing APGI Error Handler")
    print("=" * 40)

    # Test different error types
    try:
        # Test configuration error
        raise config_error(
            "INVALID_PARAMETER",
            param="tau_S",
            details="Value must be between 0.1 and 2.0",
        )
    except APGIError as e:
        print(f"Configuration error: {format_user_message(e)}")

    try:
        # Test validation error
        raise validation_error(
            "VALIDATION_FAILED",
            protocol="Protocol 1",
            details="Convergence criteria not met",
        )
    except APGIError as e:
        print(f"Validation error: {format_user_message(e)}")

    try:
        # Test data error
        raise data_error("MISSING_REQUIRED_FIELDS", fields=["subject_id", "timestamp", "value"])
    except APGIError as e:
        print(f"Data error: {format_user_message(e)}")

    # Show error summary
    summary = error_handler.get_error_summary()
    print(f"\nError Summary: {summary}")
