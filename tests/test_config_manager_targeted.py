from __future__ import annotations

import json

import pytest


def test_config_manager_set_save_and_reset(tmp_path, monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    from utils.config_manager import ConfigManager

    cfg_path = tmp_path / "cfg.yaml"
    mgr = ConfigManager(str(cfg_path))

    # set a couple values (covers conversion + validation paths)
    assert mgr.set_parameter("simulation", "default_steps", 123) is True
    assert mgr.set_parameter("logging", "level", "INFO") is True

    # save to yaml
    out_path = tmp_path / "out.yaml"
    mgr.save_config(str(out_path))
    assert out_path.exists()

    # reset section and full reset
    mgr.reset_to_defaults("simulation")
    mgr.reset_to_defaults()


def test_config_manager_compare_and_diff(tmp_path, monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    from utils.config_manager import ConfigManager

    mgr = ConfigManager(str(tmp_path / "cfg.yaml"))
    from dataclasses import asdict

    current = asdict(mgr.get_config())

    modified = json.loads(json.dumps(current))
    modified.setdefault("simulation", {})
    modified["simulation"]["default_steps"] = (
        modified["simulation"].get("default_steps", 100) or 100
    ) + 1

    diff = mgr.compare_configs(current, modified)
    assert isinstance(diff, dict)


def test_validate_file_path_security(tmp_path, monkeypatch):
    from utils.config_manager import _validate_file_path

    # Allowed within project root is already enforced; for this unit test,
    # just ensure invalid traversal inputs raise.
    with pytest.raises(ValueError):
        _ = _validate_file_path("../secrets.txt", allowed_dirs=["config"])


def test_config_profiles_create_list_load(monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    import uuid
    from pathlib import Path

    from utils.config_manager import PROFILES_DIR, ConfigManager

    mgr = ConfigManager()
    name = f"pytest_{uuid.uuid4().hex[:8]}"
    profile_path = mgr.create_profile(
        name=name,
        description="pytest profile",
        category="test",
        tags=["pytest"],
    )
    assert Path(profile_path).exists()

    profiles = mgr.list_profiles(category="test")
    assert any(p.get("name") == name for p in profiles)

    assert mgr.load_profile(name) is True

    # Cleanup profile file
    p = PROFILES_DIR / f"{name}.yaml"
    if p.exists():
        p.unlink()
