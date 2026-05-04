from __future__ import annotations

import pytest


def test_security_context_from_token_and_file_ops(tmp_path, monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    from utils.auth_adapter import Role, get_auth_manager
    from utils.security_logging_integration import (
        SecurityAuthorizationError,
        secure_file_delete,
        secure_file_read,
        secure_file_write,
        security_context_from_token,
    )

    token = get_auth_manager().generate_token(
        "user1", Role.RESEARCHER, expiration_hours=1
    )
    ctx = security_context_from_token(token)

    p = tmp_path / "a.txt"
    secure_file_write(str(p), "hello", context=ctx)
    assert secure_file_read(str(p), context=ctx) == "hello"
    secure_file_delete(str(p), context=ctx)
    assert not p.exists()

    bad_token = "not-a-jwt"
    with pytest.raises(SecurityAuthorizationError):
        _ = security_context_from_token(bad_token)


def test_guest_role_cannot_write(tmp_path, monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    from utils.auth_adapter import Role, get_auth_manager
    from utils.security_logging_integration import (
        SecurityAuthorizationError,
        secure_file_write,
        security_context_from_token,
    )

    token = get_auth_manager().generate_token("user2", Role.GUEST, expiration_hours=1)
    ctx = security_context_from_token(token)

    p = tmp_path / "b.txt"
    with pytest.raises(SecurityAuthorizationError):
        secure_file_write(str(p), "nope", context=ctx)
