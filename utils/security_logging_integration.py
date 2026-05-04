"""Mandatory security audit middleware for APGI file and module operations."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# Add parent directory to path for standalone execution
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.auth_adapter import Role, get_auth_manager
from utils.error_handler import APGIError, ErrorCategory, ErrorSeverity
from utils.security_audit_logger import (
    audit_file_operation,
    audit_path_resolution,
    get_audit_logger,
)


class SecurityAuthorizationError(APGIError):
    """Authorization and policy violations for secured operations."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SECURITY,
        )


@dataclass(frozen=True)
class SecurityContext:
    """Security context used by middleware to authorize operations."""

    user_id: str = "system"
    roles: frozenset[str] = frozenset()
    jwt_role: Optional[Role] = None


_ROLE_PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.GUEST: frozenset({"reader"}),
    Role.AUDITOR: frozenset({"reader"}),
    Role.RESEARCHER: frozenset({"reader", "writer", "importer", "executor"}),
    Role.ADMIN: frozenset({"reader", "writer", "importer", "executor"}),
}


def _require_role(context: Optional[SecurityContext], required_role: str) -> None:
    """Require specific role - deny-by-default with no admin bypass."""
    ctx = context or SecurityContext()
    if required_role in ctx.roles:
        return
    if ctx.jwt_role is not None and required_role in _ROLE_PERMISSIONS.get(
        ctx.jwt_role, frozenset()
    ):
        return
    # No permission via explicit roles or JWT role mapping.
    # Deny by default.
    allowed = sorted(
        set(ctx.roles) | set(_ROLE_PERMISSIONS.get(ctx.jwt_role, frozenset()))
    )
    raise SecurityAuthorizationError(
        f"User '{ctx.user_id}' lacks required permission '{required_role}' (allowed: {allowed})"
    )


def security_context_from_token(token: str) -> SecurityContext:
    """Create a SecurityContext from a JWT token using the canonical auth adapter."""
    session = get_auth_manager().validate_token(token)
    if not session:
        raise SecurityAuthorizationError("Invalid or expired security token")
    return SecurityContext(user_id=session.user_id, jwt_role=session.role)


def _validate_path(file_path: str) -> Path:
    path = Path(file_path)
    if ".." in path.as_posix().split("/"):
        raise SecurityAuthorizationError("Path traversal attempt detected")
    return path


@audit_file_operation("read")
def secure_file_read(file_path: str, context: Optional[SecurityContext] = None) -> str:
    """Read file content with mandatory authorization and audit logging."""
    _require_role(context, "reader")
    path = _validate_path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@audit_file_operation("write")
def secure_file_write(
    file_path: str,
    content: str,
    context: Optional[SecurityContext] = None,
) -> None:
    """Write file content with mandatory authorization and audit logging."""
    _require_role(context, "writer")
    path = _validate_path(file_path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@audit_file_operation("delete")
def secure_file_delete(
    file_path: str, context: Optional[SecurityContext] = None
) -> None:
    """Delete file with mandatory authorization and audit logging."""
    _require_role(context, "writer")
    path = _validate_path(file_path)
    path.unlink()


@audit_file_operation("import")
def secure_import_module(
    module_name: str,
    module_path: str,
    context: Optional[SecurityContext] = None,
):
    """Import module with mandatory authorization and audit logging."""
    _require_role(context, "importer")
    path = _validate_path(module_path)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec for {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@audit_path_resolution("resolve")
def secure_resolve_path(path_str: str) -> str:
    """Resolve path with mandatory audit logging."""
    resolved = str(Path(path_str).resolve())
    logger = get_audit_logger()
    logger.log_path_resolution(path_str, resolved, "resolve", success=True)
    return resolved


def enforce_security_audit(operation: str) -> Callable:
    """Decorator factory for mandatory security-audited operations."""

    def decorator(func: Callable) -> Callable:
        return audit_file_operation(operation)(func)

    return decorator
