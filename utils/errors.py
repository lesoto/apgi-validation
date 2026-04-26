"""Typed domain exceptions for APGI framework error taxonomy."""

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCategory(Enum):
    """Categories of errors in the APGI framework."""

    VALIDATION = "validation"
    COMPUTATION = "computation"
    DATA = "data"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    PROTOCOL = "protocol"


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class DomainError(Exception):
    """Base class for all domain-specific exceptions."""

    def __init__(
        self,
        message: str,
        remediation: str,
        category: ErrorCategory = ErrorCategory.COMPUTATION,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.remediation = remediation
        self.category = category
        self.severity = severity
        self.context = context or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "remediation": self.remediation,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
        }


class ValidationError(DomainError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        remediation: str,
        field: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        full_context = context or {}
        if field:
            full_context["field"] = field
        super().__init__(
            message,
            remediation,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            context=full_context,
        )


class ComputationError(DomainError):
    """Raised when a computation fails."""

    def __init__(
        self,
        message: str,
        remediation: str,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        full_context = context or {}
        if operation:
            full_context["operation"] = operation
        super().__init__(
            message,
            remediation,
            category=ErrorCategory.COMPUTATION,
            severity=ErrorSeverity.HIGH,
            context=full_context,
        )


class DataError(DomainError):
    """Raised when data operations fail."""

    def __init__(
        self,
        message: str,
        remediation: str,
        data_source: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        full_context = context or {}
        if data_source:
            full_context["data_source"] = data_source
        super().__init__(
            message,
            remediation,
            category=ErrorCategory.DATA,
            severity=ErrorSeverity.MEDIUM,
            context=full_context,
        )


class ConfigurationError(DomainError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        remediation: str,
        config_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        full_context = context or {}
        if config_key:
            full_context["config_key"] = config_key
        super().__init__(
            message,
            remediation,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            context=full_context,
        )


class SecurityError(DomainError):
    """Raised when a security violation occurs."""

    def __init__(
        self,
        message: str,
        remediation: str,
        violation_type: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        full_context = context or {}
        if violation_type:
            full_context["violation_type"] = violation_type
        super().__init__(
            message,
            remediation,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            context=full_context,
        )


class InfrastructureError(DomainError):
    """Raised when infrastructure resources fail."""

    def __init__(
        self,
        message: str,
        remediation: str,
        resource: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        full_context = context or {}
        if resource:
            full_context["resource"] = resource
        super().__init__(
            message,
            remediation,
            category=ErrorCategory.INFRASTRUCTURE,
            severity=ErrorSeverity.HIGH,
            context=full_context,
        )


class ProtocolError(DomainError):
    """Raised when a validation/falsification protocol fails."""

    def __init__(
        self,
        message: str,
        remediation: str,
        protocol_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        full_context = context or {}
        if protocol_name:
            full_context["protocol_name"] = protocol_name
        super().__init__(
            message,
            remediation,
            category=ErrorCategory.PROTOCOL,
            severity=ErrorSeverity.HIGH,
            context=full_context,
        )
