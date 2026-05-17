"""Cross-platform GUI smoke tests for APGI GUIs."""

import tkinter as tk
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize(
    "gui_factory",
    [
        pytest.param(
            lambda root: __import__("gui.script_runner_gui", fromlist=["ScriptRunnerGUI"]).ScriptRunnerGUI(root),
            id="Theory_GUI",
        ),
        pytest.param(
            lambda root: __import__("Validation_GUI", fromlist=["APGIValidationGUI"]).APGIValidationGUI(root),
            id="Validation_GUI",
        ),
        pytest.param(
            lambda root: __import__("Falsification_GUI", fromlist=["ProtocolRunnerGUI"]).ProtocolRunnerGUI(root),
            id="Falsification_GUI",
        ),
        pytest.param(
            lambda root: __import__("Tests_GUI", fromlist=["TestsGUI"]).TestsGUI(root),
            id="Tests_GUI",
        ),
        pytest.param(
            lambda root: __import__("Utils_GUI", fromlist=["UtilsRunnerGUI"]).UtilsRunnerGUI(root),
            id="Utils_GUI",
        ),
    ],
)
def test_gui_smoke_cross_platform(gui_factory):
    """Instantiate each GUI to validate basic window creation across platforms."""
    if isinstance(tk, MagicMock) or isinstance(getattr(tk, "Tk", None), MagicMock):
        pytest.skip("Tkinter is mocked in this test session; skipping real GUI smoke test")

    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("Tkinter not available in this environment")

    try:
        gui_factory(root)
        root.update_idletasks()
    except Exception as exc:
        pytest.fail(f"GUI initialization failed: {exc}")
    finally:
        root.destroy()
