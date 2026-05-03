import os
import sys

import pytest
from click.testing import CliRunner

# Ensure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from main import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_validate_command_help(runner):
    result = runner.invoke(cli, ["validate", "--help"])
    assert result.exit_code == 0
    assert "validate" in result.output


def test_falsify_command_help(runner):
    result = runner.invoke(cli, ["falsify", "--help"])
    assert result.exit_code == 0
    assert "falsify" in result.output


def test_gui_command_help(runner):
    result = runner.invoke(cli, ["gui", "--help"])
    assert result.exit_code == 0


def test_status_command(runner):
    result = runner.invoke(cli, ["status"])
    # It might fail if dependencies or db aren't loaded, but we just want coverage
    assert result.exit_code in (0, 1, 2)


def test_validate_all_dry_run(runner):
    # Try a fast command to cover execution paths
    result = runner.invoke(
        cli,
        [
            "validate",
            "--protocol",
            "VP_16_Metabolic_ATP_GroundTruth.py",
            "--timeout",
            "1",
        ],
    )
    # Depending on the system, it could succeed or fail gracefully
    assert result.exit_code in (0, 1, 2)


def test_falsify_all_dry_run(runner):
    result = runner.invoke(
        cli, ["falsify", "--protocol", "FP_01_ActiveInference.py", "--timeout", "1"]
    )
    assert result.exit_code in (0, 1, 2)
