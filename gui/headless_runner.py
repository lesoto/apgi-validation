"""HeadlessRunner — CI/headless validation runner for Theory scripts."""

import importlib.util
import os
import sys
from pathlib import Path

from gui.script_runner_gui import ScriptRunnerGUI

try:
    from utils.protocol_manifest import verify_protocol_file as _verify_protocol
except ImportError:
    _verify_protocol = None  # type: ignore[assignment]

# Project root is one level above this file's directory (gui/)
_PROJECT_ROOT = Path(__file__).parent.parent


class HeadlessRunner:
    """Run all Theory scripts without GUI for CI/headless validation."""

    def __init__(self):
        self.project_root = str(_PROJECT_ROOT)
        self.theory_dir = str(_PROJECT_ROOT / "Theory")
        self._messages = []

    def log_message(self, msg):
        """Print and buffer log messages."""
        print(msg)
        self._messages.append(msg)

    def _discover_protocols(self):
        """Reuse the GUI discovery logic without tkinter."""

        # Create a minimal mock object with just the log_message method
        class MockGUI:
            def log_message(self, msg):
                self.log_message(msg)

        mock = MockGUI()
        mock.log_message = self.log_message  # type: ignore[method-assign]
        return ScriptRunnerGUI._discover_protocols(mock, self.theory_dir)  # type: ignore[arg-type]

    def _execute_protocol(self, display_name, protocol_info):
        """Execute a single protocol and return (success, message)."""
        try:
            file_path = protocol_info.get("file_path", "")
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"

            _mod_key = protocol_info.get("module_name", protocol_info["file"].replace(".py", ""))
            _fp = Path(file_path)
            if _verify_protocol is not None and not _verify_protocol(_fp, "Theory"):
                return False, f"Protocol {_fp.name} failed manifest integrity check"

            spec = importlib.util.spec_from_file_location(_mod_key, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[_mod_key] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(_mod_key, None)

            exec_info = protocol_info.get("execution_info", {})
            exec_type = exec_info.get("type", "exec_module")

            if exec_type == "module_function":
                func_name = exec_info.get("function", "main")
                run_func = getattr(module, func_name, None)
                if run_func is None:
                    return False, f"Function '{func_name}' not found in module"
                try:
                    run_func()
                except TypeError:
                    run_func()

            elif exec_type == "class_method":
                class_name = exec_info.get("class")
                method_name = exec_info.get("method", "run_validation")
                if not class_name:
                    return False, "No class specified for class_method execution"
                cls = getattr(module, class_name)
                try:
                    instance = cls()
                except TypeError:
                    return False, f"Cannot instantiate {class_name} without arguments"
                method = getattr(instance, method_name, None)
                if method is None:
                    return False, f"Method '{method_name}' not found on {class_name}"
                method()

            return True, "OK"

        except Exception as exc:
            return False, str(exc)

    def run_all(self):
        """Discover and execute all Theory scripts. Returns exit code (0=all pass)."""
        self.log_message("=" * 70)
        self.log_message("APGI Theory GUI — Headless Validation Run")
        self.log_message("=" * 70)

        protocols = self._discover_protocols()

        if not protocols:
            self.log_message("[WARN] No theory scripts discovered in Theory/")
            return 1

        self.log_message(f"Discovered {len(protocols)} script(s).\n")
        passed, failed = [], []

        for i, (display_name, protocol_info) in enumerate(protocols.items(), 1):
            self.log_message(f"[{i:02d}/{len(protocols):02d}] Running: {display_name} ({protocol_info['file']})")
            ok, msg = self._execute_protocol(display_name, protocol_info)
            if ok:
                self.log_message("  ✓ PASS")
                passed.append(display_name)
            else:
                self.log_message(f"  ✗ FAIL: {msg}")
                failed.append((display_name, msg))
            self.log_message("")

        self.log_message("=" * 70)
        self.log_message("HEADLESS VALIDATION SUMMARY")
        self.log_message("=" * 70)
        self.log_message(f"  Total   : {len(protocols)}")
        self.log_message(f"  Passed  : {len(passed)}")
        self.log_message(f"  Failed  : {len(failed)}")

        if failed:
            self.log_message("\nFailed scripts:")
            for name, err in failed:
                self.log_message(f"  - {name}: {err}")

        self.log_message("=" * 70)
        return 0 if not failed else 1
