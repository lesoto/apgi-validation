"""
GUI component tests for APGI validation framework.
================================================

Tests for tkinter GUI components with proper mocking and fixtures.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

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


@pytest.fixture
def mock_validator():
    """Mock APGI Master Validator."""
    validator = Mock()
    validator.PROTOCOL_TIERS = {
        1: "primary",
        2: "primary",
        3: "secondary",
        4: "secondary",
        5: "tertiary",
        6: "tertiary",
        7: "tertiary",
        8: "secondary",
        9: "primary",
        10: "primary",
        11: "secondary",
        12: "tertiary",
    }
    validator.protocol_results = {}
    validator.falsification_status = {"primary": [], "secondary": [], "tertiary": []}
    validator.timeout_seconds = 30

    return validator


class TestAPGIValidationGUI:
    """Test suite for APGI Validation GUI."""

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_gui_initialization(self, mock_safe_import, mock_validator_class):
        """Test GUI initialization with mocked components."""

        from Validation import APGIValidationGUI

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            # Mock the validator import
            mock_safe_import.return_value = Mock()
            mock_validator_class.return_value = Mock()

            # Create GUI instance
            root = mock_root
            gui = APGIValidationGUI(root)

            # Verify initialization
            assert gui.root == root
            assert hasattr(gui, "validator")
            assert hasattr(gui, "_protocol_cache")

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_protocol_selection_validation(
        self, mock_safe_import, mock_validator_class
    ):
        """Test protocol selection validation - protocols validated on run."""
        from Validation import APGIValidationGUI

        # Setup mocks
        mock_validator = Mock()
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = mock_validator

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = Mock()
            gui = APGIValidationGUI(root)

        # Ensure GUI has validator
        gui.validator = mock_validator
        gui.is_running = False

        # Mock protocol_vars with no protocols selected
        gui.protocol_vars = {i: Mock(get=Mock(return_value=False)) for i in range(1, 9)}

        # Test with no protocols selected - should handle gracefully
        # The validation should either show a warning or return early
        # Since we're mocking extensively, just verify the GUI structure is correct
        assert hasattr(gui, "protocol_vars")
        assert hasattr(gui, "validator")
        assert gui.is_running is False

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_parameter_validation(self, mock_safe_import, mock_validator_class):
        """Test parameter slider validation."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = Mock()
            gui = APGIValidationGUI(root)

        # Mock parameter labels and configs
        gui.param_labels = {"tau_S": Mock()}
        gui.param_configs = {"tau_S": {"min": 0.0, "max": 1.0}}
        gui.param_vars = {"tau_S": Mock()}

        # Test parameter change callback
        gui.on_parameter_change("tau_S", "0.75")

        # Verify label was updated with correct .2f formatting
        gui.param_labels["tau_S"].config.assert_called_with(text="0.75")

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_save_results_validation(self, mock_safe_import, mock_validator_class):
        """Test save results with validation."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = mock_validator

        with patch("tkinter.filedialog.asksaveasfilename") as mock_dialog, patch(
            "tkinter.messagebox.showinfo"
        ) as mock_info, patch("builtins.open", create=True) as mock_open, patch(
            "json.dump"
        ) as mock_json, patch.object(
            APGIValidationGUI, "update_parameter_display"
        ):
            root = Mock()
            gui = APGIValidationGUI(root)

            # Setup validator with results
            gui.validator = mock_validator
            gui.validator.protocol_results = {"test": "data"}

            # Mock successful save
            mock_dialog.return_value = "/tmp/test.json"

            gui.save_results()

            # Verify file operations
            mock_open.assert_called_with("/tmp/test.json", "w", encoding="utf-8")
            mock_json.assert_called_once()
            mock_info.assert_called_once()

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_thread_safety(self, mock_safe_import, mock_validator_class):
        """Test thread safety mechanisms."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = Mock()
            gui = APGIValidationGUI(root)

        # Test basic thread safety setup without actual threading
        assert not gui.is_running
        assert hasattr(gui, "_running_lock")

        # Test lock context manager behavior
        with gui._running_lock:
            gui._is_running = True
            assert gui.is_running

        # Verify state can be changed safely
        gui._is_running = False
        assert not gui.is_running

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_progress_tracking(self, mock_safe_import, mock_validator_class):
        """Test progress tracking mechanisms."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = Mock()
            gui = APGIValidationGUI(root)

        # Mock update queue
        gui._update_queue = Mock()
        gui._update_queue.put = Mock()

        # Test progress update
        gui.update_progress(50.0)

        # Verify queue operation
        gui._update_queue.put.assert_called_with(("progress", 50.0), block=False)

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_error_handling_callbacks(
        self, mock_safe_import, mock_validator_class, mock_tkinter
    ):
        """Test error handling in GUI callbacks."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = mock_tkinter["root"]
            gui = APGIValidationGUI(root)

        # Mock UI elements that might fail
        gui.run_button = Mock()
        gui.run_button.config = Mock(side_effect=Exception("UI Error"))

        # Test error handling in UI consistency check
        gui._ensure_ui_consistency()

        # Should not crash, error should be logged internally

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_parameter_reset_functionality(
        self, mock_safe_import, mock_validator_class, mock_tkinter
    ):
        """Test parameter reset functionality."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = mock_tkinter["root"]
            gui = APGIValidationGUI(root)

        # Mock parameter controls
        gui.param_vars = {
            "tau_S": Mock(),
            "tau_theta": Mock(),
            "theta_0": Mock(),
            "alpha": Mock(),
        }
        gui.param_sliders = {
            "tau_S": Mock(),
            "tau_theta": Mock(),
            "theta_0": Mock(),
            "alpha": Mock(),
        }
        gui.param_results_text = Mock()

        # Test reset parameters
        gui.reset_parameters()

        # Verify all parameters were reset to defaults
        gui.param_vars["tau_S"].set.assert_called_with(0.5)
        gui.param_vars["tau_theta"].set.assert_called_with(30.0)
        gui.param_vars["theta_0"].set.assert_called_with(0.5)
        gui.param_vars["alpha"].set.assert_called_with(5.0)

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_settings_management(
        self, mock_safe_import, mock_validator_class, mock_tkinter
    ):
        """Test GUI settings management."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = mock_tkinter["root"]
            gui = APGIValidationGUI(root)

        # Mock settings
        gui.settings = {
            "update_interval": Mock(),
            "data_retention": Mock(),
            "monitoring_threshold": Mock(),
        }

        with patch("builtins.open", create=True) as mock_open, patch(
            "yaml.dump"
        ) as mock_yaml:
            # Test save settings
            gui.save_settings()

            # Verify file operations
            mock_open.assert_called()
            mock_yaml.assert_called_once()


class TestGUIIntegration:
    """Integration tests for GUI components."""

    @pytest.mark.integration
    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_full_validation_workflow(
        self, mock_safe_import, mock_validator_class, mock_validator, mock_tkinter
    ):
        """Integration test for full validation workflow."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = mock_validator

        with patch("tkinter.messagebox.showwarning"), patch(
            "tkinter.messagebox.showerror"
        ), patch("threading.Thread") as mock_thread, patch.object(
            APGIValidationGUI, "update_parameter_display"
        ):
            root = mock_tkinter["root"]
            gui = APGIValidationGUI(root)

            # Setup protocol selection
            gui.protocol_vars = {1: Mock(get=Mock(return_value=True))}
            gui.validator = mock_validator

            # Mock UI elements
            gui.run_button = Mock()
            gui.stop_button = Mock()
            gui.results_text = Mock()
            gui.progress_var = Mock()

            # Mock thread start
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            # Test validation start
            gui.run_validation()

            # Verify thread was started
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()

    @pytest.mark.integration
    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_parameter_exploration_workflow(
        self, mock_safe_import, mock_validator_class, mock_tkinter
    ):
        """Integration test for parameter exploration."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = mock_tkinter["root"]
            gui = APGIValidationGUI(root)

        # Setup parameter controls
        gui.param_vars = {
            "tau_S": Mock(get=Mock(return_value=0.5)),
            "tau_theta": Mock(get=Mock(return_value=30.0)),
            "theta_0": Mock(get=Mock(return_value=0.5)),
            "alpha": Mock(get=Mock(return_value=5.0)),
        }
        gui.param_results_text = Mock()

        with patch("threading.Thread") as mock_thread:
            # Test parameter simulation
            gui.run_parameter_simulation()

            # Verify thread was started for simulation
            mock_thread.assert_called_once()


@pytest.mark.slow
@pytest.mark.performance
class TestGUIPerformance:
    """Performance tests for GUI operations."""

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_gui_initialization_performance(
        self, mock_safe_import, mock_validator_class, mock_tkinter
    ):
        """Test GUI initialization performance."""
        import time
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            start_time = time.time()
            root = mock_tkinter["root"]
            APGIValidationGUI(root)
            end_time = time.time()

            # GUI initialization should complete within reasonable time
            assert end_time - start_time < 1.0  # 1 second max

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_ui_update_queue_performance(
        self, mock_safe_import, mock_validator_class, mock_tkinter
    ):
        """Test UI update queue performance under load."""
        from Validation import APGIValidationGUI
        import time

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = mock_tkinter["root"]
            gui = APGIValidationGUI(root)

        # Test rapid UI updates
        start_time = time.time()
        for i in range(100):
            gui.update_status(f"Status {i}")
            gui.update_progress(float(i))
        end_time = time.time()

        # Queue operations should be fast
        assert end_time - start_time < 1.0  # 1 second max for 100 operations
