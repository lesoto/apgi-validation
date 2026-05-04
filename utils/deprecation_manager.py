#!/usr/bin/env python3
"""
Centralized Deprecation Manager
==============================

Provides decorators and utilities to mark modules, classes, and functions
as deprecated with clear removal timelines and replacement suggestions.
"""

import functools
import logging
import warnings
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger("deprecation_manager")

T = TypeVar("T", bound=Callable[..., Any])


def deprecated(
    reason: str,
    replacement: Optional[str] = None,
    removal_version: Optional[str] = None,
    removal_date: Optional[str] = None,
) -> Callable[[T], T]:
    """
    Decorator to mark a function or class as deprecated.

    Args:
        reason: Why the item is deprecated.
        replacement: Suggested replacement (e.g., "apgi_core.model.APGIModel").
        removal_version: Version when the item will be removed.
        removal_date: Date when the item will be removed.
    """

    def decorator(func: T) -> T:
        message = f"{func.__name__} is deprecated: {reason}."
        if replacement:
            message += f" Use {replacement} instead."
        if removal_version:
            message += f" Will be removed in version {removal_version}."
        if removal_date:
            message += f" Scheduled removal date: {removal_date}."

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            logger.warning(f"DEPRECATED CALL: {message}")
            return func(*args, **kwargs)

        return cast(T, wrapper)

    return decorator


def warn_deprecated_module(
    module_name: str,
    reason: str,
    replacement: Optional[str] = None,
    removal_version: Optional[str] = None,
) -> None:
    """
    Issue a warning for a deprecated module.
    Should be called at the top of the deprecated module.
    """
    message = f"Module {module_name} is deprecated: {reason}."
    if replacement:
        message += f" Use {replacement} instead."
    if removal_version:
        message += f" Will be removed in version {removal_version}."

    warnings.warn(message, category=DeprecationWarning, stacklevel=2)
    logger.warning(f"DEPRECATED MODULE IMPORTED: {message}")


# Add cast for mypy if needed (imported from typing)
from typing import cast
