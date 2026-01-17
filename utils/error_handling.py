"""
Centralized Error Handling for APGI Validation System
======================================================

Provides standardized error messages and exception handling utilities.
"""

import logging
import sys
import traceback
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, Union


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better organization"""

    CONFIGURATION = "configuration"
    DATA = "data"
    VALIDATION = "validation"
    SIMULATION = "simulation"
    IO = "io"
    NETWORK = "network"
    PERMISSION = "permission"
    MEMORY = "memory"
    COMPUTATION = "computation"


class APGIError(Exception):
    """Base exception class for APGI validation system"""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.VALIDATION,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        original_error: Optional[Exception] = None,
        error_code: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.suggestion = suggestion
        self.original_error = original_error
        self.error_code = error_code
        self.timestamp = timestamp or datetime.now()
        self.traceback = traceback.format_exc() if original_error else None

    def __str__(self):
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
        return {
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "error_code": self.error_code,
            "context": self.context,
            "suggestion": self.suggestion,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback,
        }


class ValidationError(APGIError):
    """Data validation related errors"""

    def __init__(self, message: str, data_field: Optional[str] = None, **kwargs):
        if data_field:
            message = f"Validation failed for field '{data_field}': {message}"
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)


class ConfigurationError(APGIError):
    """Configuration related errors"""

    def __init__(self, message: str, config_file: Optional[str] = None, **kwargs):
        if config_file:
            message = f"Configuration error in '{config_file}': {message}"
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)


class ProtocolError(APGIError):
    """Falsification protocol related errors"""

    def __init__(self, message: str, protocol_name: Optional[str] = None, **kwargs):
        if protocol_name:
            message = f"Protocol '{protocol_name}' error: {message}"
        super().__init__(message, category=ErrorCategory.SIMULATION, **kwargs)


class DataError(APGIError):
    """Data loading/processing related errors"""

    def __init__(self, message: str, data_source: Optional[str] = None, **kwargs):
        if data_source:
            message = f"Data error from '{data_source}': {message}"
        super().__init__(message, category=ErrorCategory.DATA, **kwargs)


class ImportWarning(APGIError):
    """Import/dependency related warnings"""

    def __init__(self, message: str, package: Optional[str] = None, **kwargs):
        if package:
            message = f"Import warning for package '{package}': {message}"
        super().__init__(message, severity=ErrorSeverity.LOW, **kwargs)


def handle_error(
    error: Exception,
    logger: Optional[logging.Logger] = None,
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
        logger = logging.getLogger(__name__)

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
        ErrorSeverity.LOW: logging.WARNING,
        ErrorSeverity.MEDIUM: logging.ERROR,
        ErrorSeverity.HIGH: logging.ERROR,
        ErrorSeverity.CRITICAL: logging.CRITICAL,
    }.get(apgi_error.severity, logging.ERROR)

    logger.log(log_level, str(apgi_error))

    # Log traceback for debugging
    if apgi_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
        logger.debug(f"Traceback: {apgi_error.traceback}")

    if reraise:
        raise apgi_error

    return apgi_error


def safe_execute(
    func: Callable,
    *args,
    error_message: str = "Operation failed",
    error_type: Type[Exception] = APGIError,
    default_return: Any = None,
    logger: Optional[logging.Logger] = None,
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

        if error_type != APGIError:
            raise error_type(f"{error_message}: {e}")
        else:
            raise (
                APGIError(
                    message=error_message,
                    context={"function": func.__name__},
                    original_error=e,
                )
                if default_return is None
                else default_return
            )


def error_boundary(
    error_type: Type[APGIError] = APGIError,
    default_return: Any = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Decorator for error boundary around functions

    Args:
        error_type: Type of error to raise
        default_return: Default return value on error
        logger: Logger instance
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
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
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        if logger:
                            logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                                f"Retrying in {current_delay}s..."
                            )
                        import time

                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        if logger:
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


def format_error_message(template_key: str, **kwargs) -> str:
    """Format a standardized error message"""
    template = ERROR_MESSAGES.get(template_key, "Unknown error: {details}")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        return f"Error formatting message: {template_key} - missing key: {e}"
