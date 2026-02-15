"""
APGI GUI Testing Framework
==========================

Automated GUI testing framework for APGI framework components.
Supports testing of tkinter and Dash-based GUI applications.
"""

import json
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import Mock, patch

try:
    import tkinter as tk
    from tkinter import ttk

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    import dash
    from dash.testing.application_runners import ThreadedRunner
    from dash.testing.browser import Browser

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


class GUITestResult:
    """Result container for GUI test execution."""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.success = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.screenshots: List[str] = []
        self.execution_time: float = 0.0
        self.metadata: Dict[str, Any] = {}

    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_name": self.test_name,
            "success": self.success,
            "errors": self.errors,
            "warnings": self.warnings,
            "screenshots": self.screenshots,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


class BaseGUITester:
    """Base class for GUI testing functionality."""

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.test_results: List[GUITestResult] = []
        self.screenshot_dir = Path("test_screenshots")
        self.screenshot_dir.mkdir(exist_ok=True)

    def run_test(self, test_name: str, test_func: Callable) -> GUITestResult:
        """Run a single GUI test."""
        result = GUITestResult(test_name)
        start_time = time.time()

        try:
            test_func(result)
        except Exception as e:
            result.add_error(f"Test execution failed: {str(e)}")
        finally:
            result.execution_time = time.time() - start_time
            self.test_results.append(result)

        return result

    def take_screenshot(self, filename: str, description: str = "") -> str:
        """Take a screenshot and save it."""
        # This is a placeholder - actual screenshot implementation
        # would depend on the GUI framework
        screenshot_path = self.screenshot_dir / f"{filename}.png"
        # In a real implementation, this would capture the actual GUI
        # For now, just create an empty file as placeholder
        screenshot_path.touch()
        return str(screenshot_path)

    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate test report."""
        report = {
            "app_name": self.app_name,
            "timestamp": time.time(),
            "total_tests": len(self.test_results),
            "passed_tests": len([r for r in self.test_results if r.success]),
            "failed_tests": len([r for r in self.test_results if not r.success]),
            "results": [r.to_dict() for r in self.test_results],
        }

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

        return report


class TkinterTester(BaseGUITester):
    """GUI testing for tkinter applications."""

    def __init__(self, app_name: str):
        super().__init__(app_name)
        if not TKINTER_AVAILABLE:
            raise ImportError(
                "tkinter not available. Install tkinter to test tkinter GUIs."
            )

        self.root = None

    def launch_app(self, app_class: type, *args, **kwargs) -> tk.Tk:
        """Launch tkinter application for testing."""

        def run_app():
            self.root = tk.Tk()
            # Hide the main window during testing
            self.root.withdraw()

            try:
                app = app_class(self.root, *args, **kwargs)
                # Run the app briefly to initialize
                self.root.update()
                return app
            except Exception as e:
                self.root.quit()
                raise e

        # Run in a separate thread to avoid blocking
        app_thread = threading.Thread(target=run_app, daemon=True)
        app_thread.start()

        # Wait for app to initialize
        time.sleep(0.5)

        if not self.root:
            raise RuntimeError("Failed to launch tkinter application")

        return self.root

    def close_app(self):
        """Close the tkinter application."""
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass
            finally:
                self.root = None

    def find_widget(
        self, widget_type: str, text: Optional[str] = None, name: Optional[str] = None
    ) -> Optional[tk.Widget]:
        """Find a widget in the application."""
        if not self.root:
            return None

        def search_widgets(widget):
            # Check current widget
            if widget.winfo_class() == widget_type:
                if text and hasattr(widget, "cget"):
                    try:
                        if widget.cget("text") == text:
                            return widget
                    except:
                        pass
                elif name and widget.winfo_name() == name:
                    return widget
                elif not text and not name:
                    return widget

            # Search children
            if hasattr(widget, "winfo_children"):
                for child in widget.winfo_children():
                    result = search_widgets(child)
                    if result:
                        return result
            return None

        return search_widgets(self.root)

    def simulate_click(self, widget: tk.Widget):
        """Simulate a button click."""
        if widget and hasattr(widget, "invoke"):
            widget.invoke()

    def simulate_text_input(self, widget: tk.Widget, text: str):
        """Simulate text input."""
        if widget and hasattr(widget, "delete"):
            widget.delete(0, tk.END)
        if widget and hasattr(widget, "insert"):
            widget.insert(0, text)

    def get_widget_text(self, widget: tk.Widget) -> str:
        """Get text from a widget."""
        if widget and hasattr(widget, "get"):
            try:
                return widget.get()
            except:
                pass
        if widget and hasattr(widget, "cget"):
            try:
                return widget.cget("text")
            except:
                pass
        return ""


class DashTester(BaseGUITester):
    """GUI testing for Dash applications."""

    def __init__(self, app_name: str):
        super().__init__(app_name)
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dash not available. Install dash[testing] to test Dash GUIs."
            )

        self.app = None
        self.runner = None
        self.browser = None

    def launch_app(self, app_instance, port: int = 8050) -> str:
        """Launch Dash application for testing."""
        self.app = app_instance

        try:
            self.runner = ThreadedRunner()
            self.runner.start(self.app, port=port)
            time.sleep(2)  # Wait for app to start

            url = f"http://127.0.0.1:{port}"
            self.browser = Browser(url)

            return url
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to launch Dash app: {e}")

    def cleanup(self):
        """Clean up Dash testing resources."""
        if self.browser:
            try:
                self.browser.quit()
            except:
                pass
            self.browser = None

        if self.runner:
            try:
                self.runner.stop()
            except:
                pass
            self.runner = None

    def find_element(self, selector: str, by: str = "css_selector"):
        """Find an element in the Dash app."""
        if not self.browser:
            return None

        try:
            if by == "css_selector":
                return self.browser.find_element_by_css_selector(selector)
            elif by == "id":
                return self.browser.find_element_by_id(selector)
            elif by == "xpath":
                return self.browser.find_element_by_xpath(selector)
            else:
                return self.browser.find_element_by_css_selector(selector)
        except Exception:
            return None

    def click_element(self, selector: str, by: str = "css_selector"):
        """Click an element."""
        element = self.find_element(selector, by)
        if element:
            element.click()
            time.sleep(0.5)  # Wait for action to complete

    def set_input_value(self, selector: str, value: str, by: str = "css_selector"):
        """Set value of an input element."""
        element = self.find_element(selector, by)
        if element:
            element.clear()
            element.send_keys(value)
            time.sleep(0.5)

    def get_element_text(self, selector: str, by: str = "css_selector") -> str:
        """Get text from an element."""
        element = self.find_element(selector, by)
        if element:
            return element.text
        return ""

    def wait_for_element(
        self, selector: str, timeout: int = 10, by: str = "css_selector"
    ) -> bool:
        """Wait for an element to appear."""
        if not self.browser:
            return False

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.find_element(selector, by):
                return True
            time.sleep(0.1)
        return False


class GUITestSuite(unittest.TestCase):
    """Base test suite for GUI testing."""

    def setUp(self):
        """Set up test environment."""
        self.tester = None

    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, "tester") and self.tester:
            if isinstance(self.tester, DashTester):
                self.tester.cleanup()
            elif isinstance(self.tester, TkinterTester):
                self.tester.close_app()


def test_tkinter_utils_gui():
    """Test the tkinter Utils GUI."""
    if not TKINTER_AVAILABLE:
        print("Skipping tkinter tests - tkinter not available")
        return

    tester = TkinterTester("Utils GUI")

    def test_basic_functionality(result: GUITestResult):
        try:
            # Import the GUI class
            sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))
            from Utils_GUI import UtilsRunnerGUI

            # Launch the app
            root = tester.launch_app(UtilsRunnerGUI)

            # Test that main window exists
            if not root:
                result.add_error("Failed to create main window")
                return

            # Look for basic GUI elements
            script_listbox = tester.find_widget("Listbox")
            if not script_listbox:
                result.add_warning("Script listbox not found")

            run_button = tester.find_widget("Button", text="Run")
            if not run_button:
                result.add_warning("Run button not found")

            # Test window title
            if root.title() != "APGI Utils Scripts Runner":
                result.add_warning(f"Unexpected window title: {root.title()}")

            result.metadata.update(
                {
                    "window_title": root.title(),
                    "widgets_found": (
                        ["Listbox", "Button"] if script_listbox and run_button else []
                    ),
                }
            )

        except Exception as e:
            result.add_error(f"GUI test failed: {str(e)}")

    # Run the test
    result = tester.run_test("tkinter_utils_basic", test_basic_functionality)

    # Generate report
    report = tester.generate_report("test_results/tkinter_utils_test.json")

    return result.success


def test_dash_interactive_dashboard():
    """Test the Dash interactive dashboard."""
    if not DASH_AVAILABLE:
        print("Skipping Dash tests - Dash not available")
        return

    tester = DashTester("Interactive Dashboard")

    def test_basic_functionality(result: GUITestResult):
        try:
            # Import the dashboard
            sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))
            from interactive_dashboard import create_dashboard_app

            # Create and launch the app
            app = create_dashboard_app()
            url = tester.launch_app(app, port=8051)

            # Test basic page load
            if not tester.wait_for_element("body", timeout=5):
                result.add_error("Dashboard page failed to load")
                return

            # Look for expected elements
            title_element = tester.find_element("h1, h2, h3")
            if not title_element:
                result.add_warning("No title element found")

            # Check for basic dashboard components
            graph_elements = tester.browser.find_elements_by_css_selector(".dash-graph")
            if len(graph_elements) == 0:
                result.add_warning("No graph elements found")

            result.metadata.update(
                {
                    "url": url,
                    "graphs_found": len(graph_elements),
                    "title_found": title_element is not None,
                }
            )

        except Exception as e:
            result.add_error(f"Dashboard test failed: {str(e)}")

    # Run the test
    result = tester.run_test("dash_dashboard_basic", test_basic_functionality)

    # Generate report
    report = tester.generate_report("test_results/dash_dashboard_test.json")

    return result.success


def run_gui_tests():
    """Run all available GUI tests."""
    print("APGI GUI Testing Framework")
    print("=" * 40)

    results = {}

    # Test tkinter GUI if available
    if TKINTER_AVAILABLE:
        print("Running tkinter GUI tests...")
        try:
            results["tkinter"] = test_tkinter_utils_gui()
            print(f"  Tkinter tests: {'PASSED' if results['tkinter'] else 'FAILED'}")
        except Exception as e:
            print(f"  Tkinter tests failed: {e}")
            results["tkinter"] = False
    else:
        print("Skipping tkinter tests - not available")
        results["tkinter"] = None

    # Test Dash GUI if available
    if DASH_AVAILABLE:
        print("Running Dash GUI tests...")
        try:
            results["dash"] = test_dash_interactive_dashboard()
            print(f"  Dash tests: {'PASSED' if results['dash'] else 'FAILED'}")
        except Exception as e:
            print(f"  Dash tests failed: {e}")
            results["dash"] = False
    else:
        print("Skipping Dash tests - not available")
        results["dash"] = None

    # Summary
    print("\nTest Summary:")
    print(f"  Tkinter available: {TKINTER_AVAILABLE}")
    print(f"  Dash available: {DASH_AVAILABLE}")

    passed_tests = sum(1 for r in results.values() if r is True)
    total_tests = sum(1 for r in results.values() if r is not None)

    print(f"\nResults: {passed_tests}/{total_tests} test suites passed")

    return results


if __name__ == "__main__":
    # Run GUI tests
    results = run_gui_tests()

    # Exit with appropriate code
    if any(r is False for r in results.values() if r is not None):
        sys.exit(1)
    else:
        sys.exit(0)
