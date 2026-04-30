"""Unified error taxonomy and exception hierarchy for APGI framework."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCategory(str, Enum):
    """Categories of errors in the APGI framework."""

    VALIDATION = "VALIDATION"
    COMPUTATION = "COMPUTATION"
    DATA = "DATA"
    CONFIGURATION = "CONFIGURATION"
    SECURITY = "SECURITY"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    PROTOCOL = "PROTOCOL"
    RUNTIME = "RUNTIME"
    SIMULATION = "SIMULATION"
    IO = "IO"
    MEMORY = "MEMORY"
    IMPORT = "IMPORT"
    USER_INPUT = "USER_INPUT"
    BACKUP = "BACKUP"
    CACHE = "CACHE"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ErrorCode(str, Enum):
    """Unique error codes for precise identification and telemetry."""

    # General / Runtime
    GEN_UNKNOWN = "GEN_000"
    GEN_NOT_IMPLEMENTED = "GEN_001"
    GEN_TIMEOUT = "GEN_002"
    GEN_RESOURCE_EXHAUSTED = "GEN_003"

    # Validation
    VAL_INVALID_INPUT = "VAL_001"
    VAL_SCHEMA_MISMATCH = "VAL_002"
    VAL_RANGE_ERROR = "VAL_003"
    VAL_TYPE_ERROR = "VAL_004"

    # Protocol
    PRT_EXECUTION_FAILED = "PRT_001"
    PRT_REGISTRATION_FAILED = "PRT_002"
    PRT_DEPENDENCY_MISSING = "PRT_003"
    PRT_INVALID_RESULT = "PRT_004"

    # Data
    DAT_LOAD_FAILED = "DAT_001"
    DAT_SAVE_FAILED = "DAT_002"
    DAT_CORRUPTION = "DAT_003"
    DAT_NOT_FOUND = "DAT_004"

    # Configuration
    CFG_MISSING_KEY = "CFG_001"
    CFG_INVALID_VALUE = "CFG_002"
    CFG_LOAD_FAILED = "CFG_003"

    # Security
    SEC_UNAUTHORIZED = "SEC_001"
    SEC_FORBIDDEN = "SEC_002"
    SEC_TOKEN_EXPIRED = "SEC_003"


class APGIException(Exception):
    """Base class for all APGI-specific exceptions with telemetry support."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.GEN_UNKNOWN,
        category: ErrorCategory = ErrorCategory.RUNTIME,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        remediation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.category = category
        self.severity = severity
        self.remediation = remediation
        self.context = context or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "code": self.code.value,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "remediation": self.remediation,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


class ValidationError(APGIException):
    """Raised when data or input validation fails."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", ErrorCode.VAL_INVALID_INPUT)
        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ProtocolError(APGIException):
    """Raised when a validation/falsification protocol fails."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", ErrorCode.PRT_EXECUTION_FAILED)
        kwargs.setdefault("category", ErrorCategory.PROTOCOL)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class DataError(APGIException):
    """Raised when data operations fail."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", ErrorCode.DAT_LOAD_FAILED)
        kwargs.setdefault("category", ErrorCategory.DATA)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class ConfigurationError(APGIException):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", ErrorCode.CFG_LOAD_FAILED)
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class SecurityError(APGIException):
    """Raised when a security violation occurs."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", ErrorCode.SEC_UNAUTHORIZED)
        kwargs.setdefault("category", ErrorCategory.SECURITY)
        kwargs.setdefault("severity", ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


class ComputationError(APGIException):
    """Raised when numerical or logic computation fails."""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("code", ErrorCode.GEN_UNKNOWN)
        kwargs.setdefault("category", ErrorCategory.COMPUTATION)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)
