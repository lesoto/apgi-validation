#!/usr/bin/env python3
"""
Centralized Error Handling System for APGI Framework
================================================

Provides standardized error messages, error categories, and centralized
error handling with proper logging and user-friendly messages.
"""

import functools
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Type

try:
    from utils.logging_config import apgi_logger
except ImportError:
    # Fallback if running as standalone script
    import logging

    class MockAPGILogger:
        def __init__(self):
            self.logger = logging.getLogger(__name__)

    apgi_logger = MockAPGILogger()


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

    def __init__(
        self,
        error_info: Optional[ErrorInfo] = None,
        message: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.VALIDATION,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        original_error: Optional[Exception] = None,
        error_code: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        if error_info:
            self.error_info = error_info
            self.message = error_info.message
            self.severity = error_info.severity
            self.category = error_info.category
            self.context = error_info.details or {}
            self.suggestion = error_info.user_action
            self.original_error = original_error
            self.error_code = error_info.code
            self.timestamp = timestamp or datetime.now()
        else:
            self.error_info = None
            self.message = message or "Unknown error"
            self.severity = severity
            self.category = category
            self.context = context or {}
            self.suggestion = suggestion
            self.original_error = original_error
            self.error_code = error_code
            self.timestamp = timestamp or datetime.now()

        self.traceback = traceback.format_exc() if original_error else None
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.error_info:
            return f"[{self.error_info.severity.value}] {self.error_info.category.value}: {self.error_info.message}"
        else:
            base_msg = f"[{self.severity.value.upper()}] {self.message}"
            if self.error_code:
                base_msg = f"[{self.error_code}] {base_msg}"
            if self.suggestion:
                base_msg += f"\n💡 Suggestion: {self.suggestion}"
            if self.context:
                context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
                base_msg += f"\n📍 Context: {context_str}"
            return base_msg

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        # Sanitize traceback to prevent sensitive data exposure
        sanitized_traceback = None
        if self.traceback:
            # Remove file paths and potentially sensitive information from traceback
            import re

            sanitized_traceback = re.sub(
                r'File "[^"]*"', 'File "[REDACTED]"', self.traceback
            )
            sanitized_traceback = re.sub(r"(/[^\s]+)", "/[PATH]", sanitized_traceback)
            # Limit traceback length
            if len(sanitized_traceback) > 500:
                sanitized_traceback = sanitized_traceback[:500] + "...[TRUNCATED]"

        return {
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "error_code": self.error_code,
            "context": self.context,
            "suggestion": self.suggestion,
            "timestamp": self.timestamp.isoformat(),
            "traceback": sanitized_traceback,
        }


class ValidationError(APGIError):
    """Data validation related errors"""

    def __init__(self, message: str, data_field: Optional[str] = None, **kwargs):
        if data_field:
            message = f"Validation failed for field '{data_field}': {message}"
        super().__init__(message=message, category=ErrorCategory.VALIDATION, **kwargs)


class ConfigurationError(APGIError):
    """Configuration related errors"""

    def __init__(self, message: str, config_file: Optional[str] = None, **kwargs):
        if config_file:
            message = f"Configuration error in '{config_file}': {message}"
        super().__init__(
            message=message, category=ErrorCategory.CONFIGURATION, **kwargs
        )


class ProtocolError(APGIError):
    """Falsification protocol related errors"""

    def __init__(self, message: str, protocol_name: Optional[str] = None, **kwargs):
        if protocol_name:
            message = f"Protocol '{protocol_name}' error: {message}"
        super().__init__(message=message, category=ErrorCategory.SIMULATION, **kwargs)


class DataError(APGIError):
    """Data loading/processing related errors"""

    def __init__(self, message: str, data_source: Optional[str] = None, **kwargs):
        if data_source:
            message = f"Data error from '{data_source}': {message}"
        super().__init__(message=message, category=ErrorCategory.DATA, **kwargs)


class APGIImportWarning(APGIError):
    """Import/dependency related warnings"""

    def __init__(self, message: str, package: Optional[str] = None, **kwargs):
        if package:
            message = f"Import warning for package '{package}': {message}"
        super().__init__(message=message, severity=ErrorSeverity.LOW, **kwargs)


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
                "VERSION_CONFLICT": "Version conflict for {package}: required {required}, installed {installed}",
                "MISSING_PACKAGE": "Package not installed: {package}. Install with: pip install {package}",
            },
            ErrorSeverity.MEDIUM: {
                "OPTIONAL_MODULE_MISSING": "Optional module missing: {module}. Some features may be limited.",
                "IMPORT_WARNING": "Import warning: {details}",
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
            import string

            template = self.ERROR_TEMPLATES[category][severity][code]
            # Use string.Template for safer formatting with user input
            safe_template = string.Template(template)
            return safe_template.safe_substitute(**kwargs)
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

        # Count errors by category with cap to prevent unbounded growth
        self.error_counts[category] = min(self.error_counts.get(category, 0) + 1, 1000)

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
        apgi_logger.logger.info(
            f"Registered error handler for category: {category.value}"
        )

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors by category."""
        total_errors = sum(self.error_counts.values())

        return {
            "total_errors": total_errors,
            "by_category": {
                category.value: count for category, count in self.error_counts.items()
            },
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
    return error_handler.handle_error(
        ErrorCategory.VALIDATION, ErrorSeverity.HIGH, code, **kwargs
    )


def simulation_error(code: str, **kwargs) -> APGIError:
    """Create simulation error."""
    return error_handler.handle_error(
        ErrorCategory.SIMULATION, ErrorSeverity.HIGH, code, **kwargs
    )


def data_error(code: str, **kwargs) -> APGIError:
    """Create data error."""
    return error_handler.handle_error(
        ErrorCategory.DATA, ErrorSeverity.HIGH, code, **kwargs
    )


def io_error(code: str, **kwargs) -> APGIError:
    """Create I/O error."""
    return error_handler.handle_error(
        ErrorCategory.IO, ErrorSeverity.HIGH, code, **kwargs
    )


def import_error(code: str, **kwargs) -> APGIError:
    """Create import/dependency error."""
    return error_handler.handle_error(
        ErrorCategory.IMPORT, ErrorSeverity.HIGH, code, **kwargs
    )


def critical_error(code: str, **kwargs) -> APGIError:
    """Create critical error."""
    return error_handler.handle_error(
        ErrorCategory.RUNTIME, ErrorSeverity.CRITICAL, code, **kwargs
    )


def handle_import_error(module_name: str, error: Exception, context: str = "") -> None:
    """Handle import errors with specific, actionable messages."""
    error_msg = str(error)

    if "No module named" in error_msg:
        missing_module = error_msg.split("'")[1] if "'" in error_msg else module_name
        suggestions = {
            "click": "pip install click",
            "pandas": "pip install pandas",
            "numpy": "pip install numpy",
            "matplotlib": "pip install matplotlib",
            "scipy": "pip install scipy",
            "torch": "pip install torch",
            "sklearn": "pip install scikit-learn",
            "pymc": "pip install pymc",
            "arviz": "pip install arviz",
            "yaml": "pip install pyyaml",
            "seaborn": "pip install seaborn",
            "tqdm": "pip install tqdm",
        }

        suggestion = suggestions.get(missing_module, f"pip install {missing_module}")
        import_error("MISSING_PACKAGE", package=missing_module, install_cmd=suggestion)

    elif "DLL load failed" in error_msg or "shared library" in error_msg:
        import_error(
            "DEPENDENCY_ERROR",
            details=f"Library loading error for {module_name}: {error_msg}",
        )

    elif "Permission denied" in error_msg:
        import_error(
            "DEPENDENCY_ERROR",
            details=f"Permission error loading {module_name}: {error_msg}",
        )

    else:
        import_error(
            "DEPENDENCY_ERROR", details=f"Import error for {module_name}: {error_msg}"
        )

    if context:
        apgi_logger.logger.warning(f"Import context: {context}")


def format_user_message(error: APGIError) -> str:
    if error.error_info is None:
        # Fallback when error_info is not available
        message = f"❌ {error.message}"
        if error.suggestion:
            message += f"\n\n💡 Suggestion: {error.suggestion}"
        return message

    info = error.error_info

    # Base message
    message = f"❌ {info.message}"

    # Add suggestions if available
    if info.suggestions:
        message += "\n\n💡 Suggestions:"
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


def get_error_summary() -> Dict[str, Any]:
    """Get error summary from global error handler."""
    return error_handler.get_error_summary()


def handle_error(
    error: Exception,
    logger=None,
    reraise: bool = False,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[APGIError]:
    """
    Standardized error handling utility

    Args:
        error: The exception to handle
        logger: Logger instance for logging
        reraise: Whether to reraise exception
        context: Additional context information

    Returns:
        APGIError instance if not reraising
    """
    if logger is None:
        logger = apgi_logger.logger

    # Convert to APGIError if it's not already
    if not isinstance(error, APGIError):
        apgi_error = APGIError(
            message=str(error),
            severity=ErrorSeverity.MEDIUM,
            context=context,
            original_error=error,
        )
    else:
        apgi_error = error
        if context:
            apgi_error.context.update(context)

    # Log error
    log_level = {
        ErrorSeverity.LOW: apgi_logger.logger.warning,
        ErrorSeverity.MEDIUM: apgi_logger.logger.error,
        ErrorSeverity.HIGH: apgi_logger.logger.error,
        ErrorSeverity.CRITICAL: apgi_logger.logger.critical,
    }.get(apgi_error.severity, apgi_logger.logger.error)

    log_level(str(apgi_error))

    # Log traceback for debugging
    if apgi_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
        apgi_logger.logger.debug(f"Traceback: {apgi_error.traceback}")

    if reraise:
        raise apgi_error

    return apgi_error


def safe_execute(
    func: Callable,
    *args,
    error_message: str = "Operation failed",
    error_type=None,
    default_return: Any = None,
    logger=None,
    **kwargs,
) -> Any:
    """
    Safely execute a function with standardized error handling

    Args:
        func: Function to execute
        *args: Function arguments
        error_message: Custom error message
        error_type: Type of exception to raise
        default_return: Default return value on error
        logger: Logger instance
        **kwargs: Function keyword arguments

    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            handle_error(e, logger=logger, context={"function": func.__name__})

        if error_type and error_type != APGIError:
            raise error_type(f"{error_message}: {e}")
        else:
            if default_return is None:
                raise APGIError(
                    message=error_message,
                    context={"function": func.__name__},
                    original_error=e,
                )
            else:
                return default_return


def error_boundary(
    error_type=None,
    default_return: Any = None,
    logger=None,
):
    """
    Decorator for error boundary around functions

    Args:
        error_type: Type of error to raise
        default_return: Default return value on error
        logger: Logger instance
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return safe_execute(
                func,
                *args,
                error_message=f"Function {func.__name__} failed",
                error_type=error_type,
                default_return=default_return,
                logger=logger,
                **kwargs,
            )

        return wrapper

    return decorator


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator for retrying functions on specific exceptions

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to retry on
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            logger = apgi_logger.logger

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        import time

                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )

            raise last_exception

        return wrapper

    return decorator


# Standard error messages
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


def safe_import(module_name: str, fallback=None, context: str = ""):
    """
    Safely import a module with standardized error handling.

    Args:
        module_name: Name of the module to import
        fallback: Optional fallback value if import fails
        context: Additional context for error reporting

    Returns:
        Imported module or fallback value
    """
    try:
        import importlib

        return importlib.import_module(module_name)
    except ImportError as e:
        handle_import_error(module_name, e, context)
        return fallback
    except Exception as e:
        import_error(
            "DEPENDENCY_ERROR", details=f"Unexpected error importing {module_name}: {e}"
        )
        return fallback


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
        raise data_error(
            "MISSING_REQUIRED_FIELDS", fields=["subject_id", "timestamp", "value"]
        )
    except APGIError as e:
        print(f"Data error: {format_user_message(e)}")

    # Show error summary
    summary = error_handler.get_error_summary()
    print(f"\nError Summary: {summary}")
