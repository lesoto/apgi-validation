from __future__ import annotations

import pytest


def test_security_gateway_check_and_require(monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    from utils.auth_adapter import Role, get_auth_manager
    from utils.security_gateway import SecurityGateway

    token = get_auth_manager().generate_token(
        "gw-user", Role.RESEARCHER, expiration_hours=1
    )
    gw = SecurityGateway()

    assert gw.check_access(token, [Role.RESEARCHER]) is True
    gw.require_roles(token, [Role.RESEARCHER])

    with pytest.raises(PermissionError):
        gw.require_roles(token, [Role.AUDITOR])
