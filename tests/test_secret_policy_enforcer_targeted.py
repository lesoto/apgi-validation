from __future__ import annotations

import pytest


def test_secret_policy_allows_ephemeral_when_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")
    monkeypatch.delenv("APGI_MASTER_KEY", raising=False)

    from utils.secret_policy_enforcer import enforce_secret_policy

    # Should not exit
    enforce_secret_policy()


def test_secret_policy_requires_master_key_when_not_ephemeral(monkeypatch, tmp_path):
    monkeypatch.delenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", raising=False)
    monkeypatch.delenv("APGI_MASTER_KEY", raising=False)
    from utils.secret_policy_enforcer import enforce_secret_policy

    with pytest.raises(SystemExit):
        enforce_secret_policy()
