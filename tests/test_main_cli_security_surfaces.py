from __future__ import annotations

from click.testing import CliRunner


def _make_valid_token() -> str:
    from utils.auth_adapter import Role, get_auth_manager

    auth = get_auth_manager()
    return auth.generate_token("test-user", Role.RESEARCHER, expiration_hours=1)


def test_cli_help_smoke(monkeypatch):
    # Ensure import-time policy enforcement doesn't hard-exit.
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    import main

    runner = CliRunner()
    result = runner.invoke(main.cli, ["--help"])
    assert result.exit_code == 0


def test_cli_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    import main

    runner = CliRunner()
    result = runner.invoke(
        main.cli, ["--token", "not-a-jwt", "config-group", "explain"]
    )
    assert result.exit_code != 0
    assert (
        "Invalid or expired security token" in result.output
        or "Security validation failed" in result.output
        or "Invalid token" in result.output
    )


def test_cli_accepts_valid_token_and_runs_command(monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    import main

    token = _make_valid_token()
    runner = CliRunner()
    # Use a low-impact command that exercises config + printing.
    result = runner.invoke(main.cli, ["--token", token, "config-group", "explain"])
    assert result.exit_code == 0
    assert "Configuration Precedence" in result.output


def test_config_command_show_set_reset(monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    import main

    token = _make_valid_token()
    runner = CliRunner()

    # show (covers _show_config path)
    r1 = runner.invoke(main.cli, ["--token", token, "config", "--show"])
    assert r1.exit_code == 0
    assert "Current Configuration" in r1.output

    # set (covers parsing + config_manager set_parameter path)
    r2 = runner.invoke(
        main.cli,
        ["--token", token, "config", "--set", "simulation.default_steps=123"],
    )
    assert r2.exit_code == 0

    # reset (covers reset_to_defaults path)
    r3 = runner.invoke(main.cli, ["--token", token, "config", "--reset"])
    assert r3.exit_code == 0

    # invalid set format hits validation branch
    r4 = runner.invoke(main.cli, ["--token", token, "config", "--set", "not_a_kv"])
    assert r4.exit_code == 0
    assert "must contain" in r4.output.lower()


def test_config_versioning_commands(monkeypatch):
    monkeypatch.setenv("APGI_ALLOW_EPHEMERAL_MASTER_KEY", "1")

    import re
    from pathlib import Path

    import main

    token = _make_valid_token()
    runner = CliRunner()

    r1 = runner.invoke(
        main.cli,
        [
            "--token",
            token,
            "config-version",
            "--description",
            "test",
            "--author",
            "pytest",
        ],
    )
    assert r1.exit_code == 0
    m = re.search(r"Configuration version created:\s*([\w-]+)", r1.output)
    assert m, r1.output
    version_id = m.group(1)

    r2 = runner.invoke(main.cli, ["--token", token, "config-versions", "--limit", "1"])
    assert r2.exit_code == 0

    r3 = runner.invoke(
        main.cli, ["--token", token, "config-restore", "--version-id", version_id]
    )
    assert r3.exit_code == 0

    r4 = runner.invoke(main.cli, ["--token", token, "config-diff"])
    assert r4.exit_code == 0

    # Cleanup created version artifact (repo-relative path used by CLI impl).
    version_file = Path("config/versions") / f"{version_id}.json"
    if version_file.exists():
        version_file.unlink()
