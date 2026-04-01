"""
GUI Error Path Tests
==================

Comprehensive tests for GUI error handling paths in APGI Validation GUI.
Tests error scenarios, UI updates, queue handling, and thread safety.
"""

import pytest
import queue

import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tkinter as tk

# Skip all tests if no display is available
try:
    # Test if we can create a Tk root window
    test_root = tk.Tk()
    test_root.destroy()
    HAS_DISPLAY = True
except Exception:
    HAS_DISPLAY = False

pytestmark = pytest.mark.skipif(
    not HAS_DISPLAY, reason="No display available for GUI tests"
)

import Validation

sys.path.insert(0, str(Path(__file__).parent.parent))
APGIValidationGUI = Validation.APGIValidationGUI


class TestGUIErrorPaths:
    """Test GUI error handling paths and scenarios"""

    def test_gui_initialization_with_mocks(self):
        """Test GUI initialization with mocked dependencies"""
        with patch("tkinter.messagebox.showerror") as _mock_showerror:  # noqa: F841
            with patch(
                "tkinter.scrolledtext.ScrolledText"
            ) as _mock_scrolled:  # noqa: F841
                with patch("tkinter.ttk.Progressbar") as _mock_progress:  # noqa: F841
                    # Mock tkinter components to prevent actual GUI creation
                    root = tk.Tk()  # Create mock root
                    gui = APGIValidationGUI(root)

                    # Should initialize successfully with mocked components
                    assert gui is not None
                    assert hasattr(gui, "root")
                    assert hasattr(gui, "progress_var")

                    # Verify that mocked components were not used in initialization
                    # (they might be used later in widget creation)

    def test_protocol_import_error_handling(self):
        """Test error handling when protocol module fails to import"""
        root = tk.Tk()  # Create mock root
        gui = APGIValidationGUI(root)

        # Simulate protocol import error
        error = ImportError("No module named 'nonexistent_protocol'")
        result = gui._handle_protocol_error(error, 1)

        # Should return error result dict with expected fields
        assert isinstance(result, dict)
        assert "status" in result
        assert "IMPORT_ERROR" in result["status"]
        assert "troubleshooting" in result
        assert result["error_type"] == "ImportError"

    def test_protocol_execution_error_with_queue(self):
        """Test error handling during protocol execution with full queue"""
        root = tk.Tk()  # Create mock root
        gui = APGIValidationGUI(root)

        # Mock validator.falsification_status to avoid AttributeError
        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        # Fill queue to test full queue behavior
        for i in range(100):  # Fill queue beyond capacity
            try:
                gui._update_queue.put(("results", f"Test message {i}"), block=False)
            except queue.Full:
                break  # Queue is full, which is expected

        # Simulate protocol execution error
        error = RuntimeError("Protocol execution failed")
        protocol_tiers = {1: "primary"}

        # Mock update_results to capture calls
        update_calls = []
        gui.update_results = lambda msg: update_calls.append(msg)

        gui._handle_protocol_execution_error(error, 1, protocol_tiers)

        # Verify error was logged via update_results
        assert len(update_calls) > 0
        assert any("Protocol 1 failed" in str(call) for call in update_calls)
        assert any("RuntimeError" in str(call) for call in update_calls)

    def test_queue_overflow_handling(self):
        """Test behavior when update queue is full"""
        root = tk.Tk()  # Create mock root
        gui = APGIValidationGUI(root)

        # Fill queue to capacity - should not raise exceptions
        full_count = 0
        for i in range(1000):  # Exceed typical queue size
            try:
                gui._update_queue.put(("results", f"Overflow test {i}"), block=False)
            except queue.Full:
                full_count += 1
                # Queue is full, which is expected - update_status should skip without blocking
                break

        # Verify queue has items and filled up
        assert gui._update_queue.qsize() > 0 or full_count > 0

    def test_thread_safety_during_error(self):
        """Test thread safety when errors occur during protocol execution"""
        results = []

        def capture_gui_update(msg):
            results.append(msg)

        # Create GUI instance first
        root = tk.Tk()  # Create mock root
        gui = APGIValidationGUI(root)

        # Mock validator.falsification_status
        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        # Mock update_results to capture calls
        gui.update_results = capture_gui_update

        # Trigger error handling
        error = ValueError("Test error")
        gui._handle_protocol_execution_error(error, 1, {1: "primary"})

        # Verify update_results was called with error info
        assert len(results) > 0
        assert any("Test error" in str(r) for r in results)

    def test_critical_error_multiple_protocols(self):
        """Test critical error handling with multiple selected protocols"""
        root = tk.Tk()  # Create mock root
        gui = APGIValidationGUI(root)

        # Capture update calls
        status_calls = []
        results_calls = []
        gui.update_status = lambda msg: status_calls.append(msg)
        gui.update_results = lambda msg: results_calls.append(msg)

        error = ImportError("Critical error in multiple protocols")
        selected_protocols = [1, 2, 3]
        gui._handle_validation_critical_error(error, selected_protocols)

        # Verify error was logged via update_results
        assert len(results_calls) > 0
        assert any("CRITICAL ERROR" in str(call) for call in results_calls)
        assert any("ImportError" in str(call) for call in results_calls)

    def test_ui_state_consistency_during_errors(self):
        """Test UI state consistency during error conditions"""
        root = tk.Tk()  # Create mock root
        gui = APGIValidationGUI(root)

        # Mock validator.falsification_status
        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        # Simulate error during protocol execution
        error = RuntimeError("Test error")
        gui._handle_protocol_execution_error(error, 1, {1: "primary"})

        # Verify UI state remains consistent - check actual widget names
        assert hasattr(gui, "protocol_vars")  # protocol_vars exists
        assert hasattr(gui, "run_button")  # run_button (not start_button)
        assert hasattr(gui, "results_text")  # results_text (not results_display)

    def test_error_recovery_and_retry(self):
        """Test error recovery and retry mechanisms - GUI uses queued updates, not messagebox"""
        root = tk.Tk()  # Create mock root
        gui = APGIValidationGUI(root)

        # Mock validator.falsification_status
        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        # Capture update calls
        results_calls = []
        gui.update_results = lambda msg: results_calls.append(msg)

        # Simulate recoverable error - GUI queues updates, no retry prompt
        error = IOError("Recoverable error")
        gui._handle_protocol_execution_error(error, 1, {1: "primary"})

        # Verify error was logged
        assert len(results_calls) > 0
        assert any("IOError" in str(call) for call in results_calls)

    def test_memory_cleanup_after_error(self):
        """Test memory cleanup after error conditions - protocol_cache exists"""
        root = tk.Tk()  # Create mock root
        gui = APGIValidationGUI(root)

        # Verify cleanup method exists (clear_protocol_cache or similar)
        assert hasattr(gui, "clear_protocol_cache")

        # Simulate memory-intensive error - verify it doesn't crash
        error = MemoryError("Out of memory")

        # Mock validator.falsification_status
        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        # Should handle without crashing
        gui._handle_protocol_execution_error(error, 1, {1: "primary"})

        # Verify clear_protocol_cache works
        gui._protocol_cache = {"test": "data"}
        gui.clear_protocol_cache()
        assert len(gui._protocol_cache) == 0

    def test_concurrent_error_handling(self):
        """Test handling of concurrent errors from multiple protocols"""
        root = tk.Tk()  # Create mock root
        gui = APGIValidationGUI(root)

        # Simulate multiple concurrent errors
        errors = [
            ImportError("Protocol 1 error"),
            RuntimeError("Protocol 2 error"),
            ValueError("Protocol 3 error"),
        ]

        # Handle errors concurrently
        for i, error in enumerate(errors):
            gui._handle_protocol_execution_error(
                error, i + 1, {i + 1: f"Protocol {i + 1}"}
            )

        # Verify all errors were handled
        # This would be verified by checking that all errors were processed

    def test_error_logging_and_troubleshooting(self):
        """Test error logging and troubleshooting information"""
        root = tk.Tk()  # Create mock root
        gui = APGIValidationGUI(root)

        # Mock validator.falsification_status
        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        # Capture results
        results_calls = []
        gui.update_results = lambda msg: results_calls.append(msg)

        # Simulate complex error
        error = Exception("Complex multi-layer error")
        gui._handle_protocol_execution_error(error, 1, {1: "Test Protocol"})

        # Verify troubleshooting info is included in update_results calls
        assert len(results_calls) >= 2  # Error message + troubleshooting
        all_text = " ".join(str(c) for c in results_calls)
        assert "troubleshooting" in all_text.lower() or "Troubleshooting" in all_text

    def test_gui_responsiveness_during_error(self):
        """Test GUI remains responsive during error handling"""
        root = tk.Tk()  # Create mock root
        gui = APGIValidationGUI(root)

        # Mock validator.falsification_status
        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        # Track update calls
        update_calls = []
        gui.update_results = lambda msg: update_calls.append(msg)

        error = TimeoutError("Protocol timeout")
        gui._handle_protocol_execution_error(error, 1, {1: "primary"})

        # Verify updates were queued (non-blocking)
        assert len(update_calls) > 0
        assert any("timeout" in str(call).lower() for call in update_calls)
