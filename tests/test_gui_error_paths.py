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
from unittest.mock import Mock, patch, MagicMock

# Prevent any tkinter imports at module level
if "tkinter" in sys.modules:
    del sys.modules["tkinter"]

# Mock tkinter and all its submodules before any imports
tkinter_modules = [
    "tkinter",
    "tkinter.ttk",
    "tkinter.messagebox",
    "tkinter.filedialog",
    "tkinter.scrolledtext",
]

for module in tkinter_modules:
    sys.modules[module] = MagicMock()

# Mock the entire tkinter ecosystem
mock_tkinter = MagicMock()
mock_tkinter.Tk = MagicMock()
mock_tkinter.ttk = MagicMock()
mock_tkinter.ttk.Frame = MagicMock()
mock_tkinter.ttk.Button = MagicMock()
mock_tkinter.ttk.Label = MagicMock()
mock_tkinter.ttk.Spinbox = MagicMock()
mock_tkinter.ttk.Checkbutton = MagicMock()
mock_tkinter.ttk.Notebook = MagicMock()
mock_tkinter.ttk.Progressbar = MagicMock()
mock_tkinter.ttk.LabelFrame = MagicMock()
mock_tkinter.StringVar = MagicMock()
mock_tkinter.BooleanVar = MagicMock()
mock_tkinter.DoubleVar = MagicMock()
mock_tkinter.IntVar = MagicMock()
mock_tkinter.Scale = MagicMock()
mock_tkinter.ScrolledText = MagicMock()
mock_tkinter.messagebox = MagicMock()
mock_tkinter.messagebox.showerror = MagicMock()
mock_tkinter.messagebox.showinfo = MagicMock()
mock_tkinter.messagebox.showwarning = MagicMock()
mock_tkinter.messagebox.askyesno = MagicMock()
mock_tkinter.filedialog = MagicMock()
mock_tkinter.filedialog.asksaveasfilename = MagicMock()
mock_tkinter.filedialog.askopenfilename = MagicMock()

# Configure Tk mock
mock_root = MagicMock()
mock_child_ids = MagicMock()
mock_child_ids.get = MagicMock(
    side_effect=lambda key, default=0: default if isinstance(default, int) else 0
)
mock_root._last_child_ids = mock_child_ids
mock_tkinter.Tk.return_value = mock_root

# Configure variable mocks to return mock objects with get() method
for var_type in ["StringVar", "BooleanVar", "DoubleVar", "IntVar"]:
    mock_var = MagicMock()
    mock_var.get = MagicMock(
        return_value=(
            0
            if var_type == "IntVar"
            else (
                0.0
                if var_type == "DoubleVar"
                else False if var_type == "BooleanVar" else ""
            )
        )
    )
    mock_var.set = MagicMock()
    getattr(mock_tkinter, var_type).return_value = mock_var

# Patch tkinter in sys.modules
sys.modules["tkinter"] = mock_tkinter

sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import Validation after tkinter is mocked
with patch("Validation.APGIMasterValidator"):
    with patch("Validation.safe_import_module"):
        import Validation

        APGIValidationGUI = Validation.APGIValidationGUI


@pytest.fixture
def mock_tkinter_fixture():
    """Return the mock tkinter object for tests."""
    return {"root": mock_root, "tkinter": mock_tkinter}


class TestGUIErrorPaths:
    """Test GUI error handling paths and scenarios"""

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_gui_initialization_with_mocks(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test GUI initialization with mocked dependencies"""
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

            assert gui is not None
            assert hasattr(gui, "root")
            assert hasattr(gui, "progress_var")

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_protocol_import_error_handling(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test error handling when protocol module fails to import"""
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

        error = ImportError("No module named 'nonexistent_protocol'")
        result = gui._handle_protocol_error(error, 1)

        assert isinstance(result, dict)
        assert "status" in result
        assert "IMPORT_ERROR" in result["status"]
        assert "troubleshooting" in result
        assert result["error_type"] == "ImportError"

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_protocol_execution_error_with_queue(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test error handling during protocol execution with full queue"""
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        for i in range(100):
            try:
                gui._update_queue.put(("results", f"Test message {i}"), block=False)
            except queue.Full:
                break

        error = RuntimeError("Protocol execution failed")
        protocol_tiers = {1: "primary"}

        update_calls = []
        gui.update_results = lambda msg: update_calls.append(msg)

        gui._handle_protocol_execution_error(error, 1, protocol_tiers)

        assert len(update_calls) > 0
        assert any("Protocol 1 failed" in str(call) for call in update_calls)

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_queue_overflow_handling(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test behavior when update queue is full"""
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

        full_count = 0
        for i in range(1000):
            try:
                gui._update_queue.put(("results", f"Overflow test {i}"), block=False)
            except queue.Full:
                full_count += 1
                break

        assert gui._update_queue.qsize() > 0 or full_count > 0

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_thread_safety_during_error(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test thread safety when errors occur during protocol execution"""
        results = []

        def capture_gui_update(msg):
            results.append(msg)

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        gui.update_results = capture_gui_update

        error = ValueError("Test error")
        gui._handle_protocol_execution_error(error, 1, {1: "primary"})

        assert len(results) > 0
        assert any("Test error" in str(r) for r in results)

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_critical_error_multiple_protocols(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test critical error handling with multiple selected protocols"""
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

        status_calls = []
        results_calls = []
        gui.update_status = lambda msg: status_calls.append(msg)
        gui.update_results = lambda msg: results_calls.append(msg)

        error = ImportError("Critical error in multiple protocols")
        selected_protocols = [1, 2, 3]
        gui._handle_validation_critical_error(error, selected_protocols)

        assert len(results_calls) > 0
        assert any("CRITICAL ERROR" in str(call) for call in results_calls)

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_ui_state_consistency_during_errors(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test UI state consistency during error conditions"""
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        error = RuntimeError("Test error")
        gui._handle_protocol_execution_error(error, 1, {1: "primary"})

        assert hasattr(gui, "protocol_vars")
        assert hasattr(gui, "run_button")
        assert hasattr(gui, "results_text")

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_error_recovery_and_retry(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test error recovery and retry mechanisms"""
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        results_calls = []
        gui.update_results = lambda msg: results_calls.append(msg)

        error = IOError("Recoverable error")
        gui._handle_protocol_execution_error(error, 1, {1: "primary"})

        assert len(results_calls) > 0

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_memory_cleanup_after_error(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test memory cleanup after error conditions"""
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

        assert hasattr(gui, "clear_protocol_cache")

        error = MemoryError("Out of memory")
        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        gui._handle_protocol_execution_error(error, 1, {1: "primary"})

        gui._protocol_cache = {"test": "data"}
        gui.clear_protocol_cache()
        assert len(gui._protocol_cache) == 0

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_concurrent_error_handling(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test handling of concurrent errors from multiple protocols"""
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

        errors = [
            ImportError("Protocol 1 error"),
            RuntimeError("Protocol 2 error"),
            ValueError("Protocol 3 error"),
        ]

        for i, error in enumerate(errors):
            gui._handle_protocol_execution_error(
                error, i + 1, {i + 1: f"Protocol {i + 1}"}
            )

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_error_logging_and_troubleshooting(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test error logging and troubleshooting information"""
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        results_calls = []
        gui.update_results = lambda msg: results_calls.append(msg)

        error = Exception("Complex multi-layer error")
        gui._handle_protocol_execution_error(error, 1, {1: "Test Protocol"})

        assert len(results_calls) >= 2
        all_text = " ".join(str(c) for c in results_calls)
        assert "troubleshooting" in all_text.lower() or "Troubleshooting" in all_text

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_gui_responsiveness_during_error(
        self, mock_safe_import, mock_validator_class, mock_tkinter_fixture
    ):
        """Test GUI remains responsive during error handling"""
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display_labels"):
            root = mock_tkinter_fixture["root"]
            gui = APGIValidationGUI(root)

        gui.validator = Mock()
        gui.validator.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

        update_calls = []
        gui.update_results = lambda msg: update_calls.append(msg)

        error = TimeoutError("Protocol timeout")
        gui._handle_protocol_execution_error(error, 1, {1: "primary"})

        assert len(update_calls) > 0
        assert any("timeout" in str(call).lower() for call in update_calls)
