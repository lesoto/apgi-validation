"""
Security Gateway
=================
A minimal security gateway that centralizes authentication and authorization checks
for all CLI/GUI/protocol entry points.
It leverages the existing `utils.auth_adapter` to validate JWT tokens and role
permissions.
"""

from typing import List

try:
    from .auth_adapter import Role, get_auth_manager
except ImportError:
    from auth_adapter import Role, get_auth_manager  # type: ignore[no-redef]


class SecurityGateway:
    """Centralized security gateway.

    The gateway provides a simple ``check_access`` method that can be called by any
    entry point (CLI command, GUI action, protocol runner) to enforce that the
    provided JWT token grants one of the required roles.
    """

    def __init__(self):
        # In a full implementation this could load policies from configuration.
        # For now we simply expose the underlying auth manager (lazy-initialized).
        self._auth_manager = get_auth_manager()

    def check_access(self, token: str, required_roles: List[Role]) -> bool:
        """Return ``True`` if the token is valid and the user's role is allowed.

        Args:
            token: JWT token string.
            required_roles: List of ``Role`` enum values that are authorised.
        """
        return self._auth_manager.check_permission(token, required_roles)

    def require_roles(self, token: str, required_roles: List[Role]):
        """Raise ``PermissionError`` if the token does not have required access."""
        if not self.check_access(token, required_roles):
            raise PermissionError(
                f"Access denied. Token does not have one of the required roles: {[r.value for r in required_roles]}"
            )
