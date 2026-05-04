import pytest


def test_get_apgi_settings_returns_instance(monkeypatch, tmp_path):
    from utils.apgi_config import get_apgi_settings

    # Force an isolated config file to avoid relying on repo defaults.
    cfg_path = tmp_path / "apgi.yaml"
    cfg_path.write_text("theta_init: 0.25\n", encoding="utf-8")

    settings = get_apgi_settings(cfg_path, reload=True)
    assert settings.theta_init == 0.25


def test_env_overrides_yaml(monkeypatch, tmp_path):
    from utils.apgi_config import get_apgi_settings

    cfg_path = tmp_path / "apgi.yaml"
    cfg_path.write_text("theta_init: 0.25\n", encoding="utf-8")

    monkeypatch.setenv("APGI_THETA_INIT", "0.75")
    settings = get_apgi_settings(cfg_path, reload=True)
    assert settings.theta_init == 0.75


def test_master_key_required_without_ephemeral(monkeypatch, tmp_path):
    from utils.secure_key_manager import SecureKeyManager

    monkeypatch.delenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", raising=False)
    monkeypatch.delenv("APGI_MASTER_KEY", raising=False)

    with pytest.raises(RuntimeError):
        SecureKeyManager(keys_dir=str(tmp_path / ".keys"))


def test_get_jwt_secret_falls_back_to_pickle(monkeypatch, tmp_path):
    from utils.secure_key_manager import SecureKeyManager, get_jwt_secret

    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")
    monkeypatch.delenv("APGI_JWT_SECRET", raising=False)

    mgr = SecureKeyManager(keys_dir=str(tmp_path / ".keys"))
    pickle_key = mgr.get_pickle_secret_key()
    assert get_jwt_secret() == pickle_key


def test_env_hex_keys_short_circuit(monkeypatch):
    from utils.secure_key_manager import get_backup_hmac_key, get_pickle_secret_key

    hex_key = "a1" * 32
    monkeypatch.setenv("PICKLE_SECRET_KEY", hex_key)
    monkeypatch.setenv("APGI_BACKUP_HMAC_KEY", hex_key)

    assert get_pickle_secret_key() == hex_key
    assert get_backup_hmac_key() == hex_key


def test_rotate_keys_and_invalidate(monkeypatch, tmp_path):
    from utils.secure_key_manager import SecureKeyManager, invalidate_all_key_references

    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")
    # Clear env vars that would bypass key file rotation
    monkeypatch.delenv("PICKLE_SECRET_KEY", raising=False)
    # Ensure master key is a valid Fernet key for rotation operations.
    from cryptography.fernet import Fernet

    monkeypatch.setenv("APGI_MASTER_KEY", Fernet.generate_key().decode())

    mgr = SecureKeyManager(keys_dir=str(tmp_path / ".keys"))
    before_pickle = mgr.get_pickle_secret_key()

    rotated = mgr.rotate_keys()
    assert isinstance(rotated, dict)

    after_pickle = mgr.get_pickle_secret_key()
    assert before_pickle != after_pickle

    invalidate_all_key_references()
