"""
APGI Error Handler - Graceful Failure Recovery
===============================================

Comprehensive error handling with automatic recovery,
retry logic, and graceful degradation.
"""

import functools
import json
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

try:
    from utils.logging_config import apgi_logger
except ImportError:
    apgi_logger = None


class ErrorSeverity(Enum):
    """Error severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""

    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    DEGRADE = "degrade"


@dataclass
class ErrorContext:
    """Context information for errors."""

    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: float
    traceback_str: str
    context_data: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "traceback": self.traceback_str,
            "context": self.context_data,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
        }


class GracefulDegradationManager:
    """Manages graceful degradation of functionality."""

    def __init__(self) -> None:
        self.degradation_levels: Dict[str, int] = {}
        self.fallback_functions: Dict[str, Callable] = {}
        self.enabled_features: Dict[str, bool] = {}

    def register_fallback(
        self, feature_name: str, fallback_func: Callable[..., Any]
    ) -> None:
        """Register a fallback function for a feature."""
        self.fallback_functions[feature_name] = fallback_func
        self.enabled_features[feature_name] = True
        self.degradation_levels[feature_name] = 0

    def degrade_feature(self, feature_name: str) -> bool:
        """Degrade a feature to its fallback implementation."""
        if feature_name not in self.enabled_features:
            return False

        self.degradation_levels[feature_name] += 1

        if self.degradation_levels[feature_name] >= 3:
            # Disable feature after 3 degradation attempts
            self.enabled_features[feature_name] = False
            if apgi_logger:
                apgi_logger.logger.warning(
                    f"Feature {feature_name} disabled after multiple failures"
                )
            return False

        if apgi_logger:
            apgi_logger.logger.info(
                f"Feature {feature_name} degraded to level {self.degradation_levels[feature_name]}"
            )
        return True

    def get_fallback(self, feature_name: str) -> Optional[Callable]:
        """Get fallback function for a feature."""
        if not self.enabled_features.get(feature_name, True):
            return None
        return self.fallback_functions.get(feature_name)

    def is_feature_available(self, feature_name: str) -> bool:
        """Check if a feature is currently available."""
        return self.enabled_features.get(feature_name, True)

    def reset_feature(self, feature_name: str):
        """Reset a feature to full functionality."""
        self.degradation_levels[feature_name] = 0
        self.enabled_features[feature_name] = True


class RetryManager:
    """Manages retry logic with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retry_history: List[Dict] = []

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.base_delay * (self.exponential_base**attempt)
        return min(delay, self.max_delay)

    def should_retry(self, attempt: int, error: Exception) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.max_retries:
            return False

        # Don't retry certain errors
        non_retryable = (
            SyntaxError,
            ImportError,
            TypeError,
            ValueError,
        )
        if isinstance(error, non_retryable):
            return False

        return True

    def execute_with_retry(
        self,
        func: Callable,
        *args,
        context: str = "",
        **kwargs,
    ) -> Any:
        """Execute function with retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0 and apgi_logger:
                    apgi_logger.logger.info(
                        f"{context} succeeded after {attempt} retries"
                    )
                return result

            except Exception as e:
                last_error = e

                if not self.should_retry(attempt, e):
                    break

                delay = self.calculate_delay(attempt)
                self.retry_history.append(
                    {
                        "context": context,
                        "attempt": attempt,
                        "error": str(e),
                        "delay": delay,
                        "timestamp": time.time(),
                    }
                )

                if apgi_logger:
                    apgi_logger.logger.warning(
                        f"{context} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                time.sleep(delay)

        # All retries exhausted
        raise last_error


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == "closed":
            return True

        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                self.half_open_calls = 0
                if apgi_logger:
                    apgi_logger.logger.info("Circuit breaker entering half-open state")
                return True
            return False

        if self.state == "half-open":
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False

    def record_success(self):
        """Record successful operation."""
        if self.state == "half-open":
            self.state = "closed"
            self.failures = 0
            self.half_open_calls = 0
            if apgi_logger:
                apgi_logger.logger.info("Circuit breaker closed - service recovered")

    def record_failure(self):
        """Record failed operation."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.state == "half-open":
            self.state = "open"
            if apgi_logger:
                apgi_logger.logger.warning("Circuit breaker opened - recovery failed")
        elif self.failures >= self.failure_threshold:
            self.state = "open"
            if apgi_logger:
                apgi_logger.logger.warning(
                    f"Circuit breaker opened after {self.failures} failures"
                )


class ErrorRecoveryManager:
    """Main error recovery manager with graceful failure handling."""

    def __init__(self) -> None:
        self.degradation_manager = GracefulDegradationManager()
        self.retry_manager = RetryManager()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}

    def register_circuit_breaker(self, service_name: str, **kwargs) -> None:
        """Register a circuit breaker for a service."""
        self.circuit_breakers[service_name] = CircuitBreaker(**kwargs)

    def register_recovery_strategy(
        self, error_type: Type[Exception], strategy: RecoveryStrategy
    ) -> None:
        """Register recovery strategy for an error type."""
        self.recovery_strategies[error_type.__name__] = strategy

    def handle_error(
        self,
        error: Exception,
        context_data: Dict[str, Any],
        severity: ErrorSeverity = ErrorSeverity.ERROR,
    ) -> ErrorContext:
        """Handle an error and attempt recovery."""
        error_context = ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            timestamp=time.time(),
            traceback_str=traceback.format_exc(),
            context_data=context_data,
        )

        self.error_history.append(error_context)

        # Log error
        self._log_error(error_context)

        return error_context

    def _log_error(self, context: ErrorContext):
        """Log error based on severity."""
        if not apgi_logger:
            return

        log_func = {
            ErrorSeverity.DEBUG: apgi_logger.logger.debug,
            ErrorSeverity.INFO: apgi_logger.logger.info,
            ErrorSeverity.WARNING: apgi_logger.logger.warning,
            ErrorSeverity.ERROR: apgi_logger.logger.error,
            ErrorSeverity.CRITICAL: apgi_logger.logger.critical,
        }.get(context.severity, apgi_logger.logger.error)

        log_func(
            f"[{context.severity.value.upper()}] {context.error_type}: {context.error_message}"
        )

    def execute_with_recovery(
        self,
        func: Callable,
        *args,
        context: str = "",
        fallback_value: Any = None,
        circuit_breaker_name: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Execute function with comprehensive error recovery.

        Args:
            func: Function to execute
            *args: Function arguments
            context: Context description for logging
            fallback_value: Value to return on failure
            circuit_breaker_name: Name of circuit breaker to use
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback value
        """
        # Check circuit breaker
        if circuit_breaker_name:
            cb = self.circuit_breakers.get(circuit_breaker_name)
            if cb and not cb.can_execute():
                if apgi_logger:
                    apgi_logger.logger.warning(
                        f"Circuit breaker open for {circuit_breaker_name}, using fallback"
                    )
                return fallback_value

        try:
            # Try with retry logic
            result = self.retry_manager.execute_with_retry(
                func, *args, context=context, **kwargs
            )

            # Record success for circuit breaker
            if circuit_breaker_name and cb:
                cb.record_success()

            return result

        except Exception as e:
            # Record failure for circuit breaker
            if circuit_breaker_name and cb:
                cb.record_failure()

            # Handle error
            error_context = self.handle_error(
                e,
                {"context": context, "args": str(args), "kwargs": str(kwargs)},
                ErrorSeverity.ERROR,
            )
            error_context.recovery_attempted = True

            # Return fallback value
            error_context.recovery_successful = fallback_value is not None
            return fallback_value

    def get_error_summary(self, hours: float = 24.0) -> Dict[str, Any]:
        """Get summary of recent errors."""
        cutoff_time = time.time() - (hours * 3600)
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]

        if not recent_errors:
            return {"status": "healthy", "error_count": 0}

        severity_counts: dict[str, int] = {}
        error_types: dict[str, int] = {}

        for error in recent_errors:
            sev = error.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

            err_type = error.error_type
            error_types[err_type] = error_types.get(err_type, 0) + 1

        return {
            "status": "degraded"
            if severity_counts.get("critical", 0) > 0
            else "healthy",
            "error_count": len(recent_errors),
            "severity_breakdown": severity_counts,
            "top_error_types": sorted(
                error_types.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "recovery_success_rate": self._calculate_recovery_rate(recent_errors),
        }

    def _calculate_recovery_rate(self, errors: List[ErrorContext]) -> float:
        """Calculate recovery success rate."""
        recovery_attempted = [e for e in errors if e.recovery_attempted]
        if not recovery_attempted:
            return 1.0

        successful = sum(1 for e in recovery_attempted if e.recovery_successful)
        return successful / len(recovery_attempted)

    def export_error_report(self, filepath: str):
        """Export error history to JSON file."""
        report_data = {
            "generated_at": time.time(),
            "total_errors": len(self.error_history),
            "errors": [e.to_dict() for e in self.error_history[-100:]],  # Last 100
            "summary": self.get_error_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2, default=str)


# Decorator for automatic error recovery
def with_recovery(
    fallback_value: Any = None,
    max_retries: int = 3,
    context: str = "",
    circuit_breaker: Optional[str] = None,
):
    """Decorator for automatic error recovery."""

    def decorator(func: Callable) -> Callable:
        recovery_manager = ErrorRecoveryManager()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return recovery_manager.execute_with_recovery(
                func,
                *args,
                context=context or func.__name__,
                fallback_value=fallback_value,
                circuit_breaker_name=circuit_breaker,
                **kwargs,
            )

        return wrapper

    return decorator


# Global recovery manager instance
_recovery_manager: Optional[ErrorRecoveryManager] = None


def get_recovery_manager() -> ErrorRecoveryManager:
    """Get or create global recovery manager."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ErrorRecoveryManager()
    return _recovery_manager


def handle_error(error: Exception, context: Dict[str, Any]) -> ErrorContext:
    """Convenience function to handle an error."""
    return get_recovery_manager().handle_error(error, context)


def get_system_health() -> Dict[str, Any]:
    """Get current system health status."""
    return get_recovery_manager().get_error_summary()


if __name__ == "__main__":
    # Demo error handling
    print("APGI Error Recovery System Demo")
    print("=" * 40)

    manager = get_recovery_manager()

    # Register circuit breaker
    manager.register_circuit_breaker("database", failure_threshold=3)

    # Example function with recovery
    @with_recovery(fallback_value="default", max_retries=2)
    def risky_operation(fail: bool = False):
        if fail:
            raise RuntimeError("Simulated failure")
        return "success"

    print("\n1. Successful operation:")
    result = risky_operation(fail=False)
    print(f"   Result: {result}")

    print("\n2. Failed operation with recovery:")
    result = risky_operation(fail=True)
    print(f"   Recovered with: {result}")

    print("\n3. System health report:")
    health = get_system_health()
    print(f"   Status: {health}")

    print("\nError handling system ready!")
