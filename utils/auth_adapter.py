#!/usr/bin/env python3
"""
Centralized Authentication & Authorization Adapter
================================================

Handles role binding, token validation, session abstractions,
and enforces least-privilege defaults for non-local deployments.
"""

import logging
import os
import time
from enum import Enum
from functools import wraps
from typing import Dict, List, Optional

import jwt

logger = logging.getLogger("auth_adapter")


class Role(Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    AUDITOR = "auditor"
    GUEST = "guest"


# Least privilege defaults
DEFAULT_ROLE = Role.GUEST


class AuthSession:
    """Represents an active authenticated session."""

    def __init__(self, user_id: str, role: Role, token: str, expires_at: float):
        self.user_id = user_id
        self.role = role
        self.token = token
        self.expires_at = expires_at

    def is_valid(self) -> bool:
        return time.time() < self.expires_at


class AuthAdapter:
    """Central adapter for AuthN and AuthZ operations."""

    def __init__(self) -> None:
        # In production, this should be securely loaded and rotated
        self.jwt_secret = os.getenv(
            "APGI_JWT_SECRET", "dev-fallback-secret-do-not-use-in-prod"
        )
        self.algorithm = "HS256"
        self._sessions: Dict[str, AuthSession] = {}

    def generate_token(
        self, user_id: str, role: Role, expiration_hours: int = 1
    ) -> str:
        """Generate a secure JWT token for a session."""
        payload = {
            "sub": user_id,
            "role": role.value,
            "exp": time.time() + (expiration_hours * 3600),
            "iat": time.time(),
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.algorithm)

    def validate_token(self, token: str) -> Optional[AuthSession]:
        """Validate a JWT token and establish/return a session."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.algorithm])
            user_id = payload.get("sub")
            role_str = payload.get("role", DEFAULT_ROLE.value)

            try:
                role = Role(role_str)
            except ValueError:
                role = DEFAULT_ROLE

            session = AuthSession(
                user_id=user_id,
                role=role,
                token=token,
                expires_at=payload.get("exp", 0),
            )
            self._sessions[token] = session
            return session

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            return None

    def check_permission(self, token: str, required_roles: List[Role]) -> bool:
        """Check if the given token has one of the required roles."""
        session = self.validate_token(token)
        if not session or not session.is_valid():
            return False

        if session.role in required_roles:
            return True

        # Admin overrides
        if session.role == Role.ADMIN:
            return True

        return False


# Global Auth Adapter
auth_manager = AuthAdapter()


def require_roles(roles: List[Role]):
    """Decorator to enforce role-based access control."""

    def decorator(func):
        @wraps(func)
        def wrapper(token: str, *args, **kwargs):
            if not auth_manager.check_permission(token, roles):
                raise PermissionError(
                    f"Access denied. Requires one of: {[r.value for r in roles]}"
                )
            return func(token, *args, **kwargs)

        return wrapper

    return decorator
