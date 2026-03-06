"""
GUI component tests for APGI validation framework.
================================================

Tests for tkinter GUI components with proper mocking and fixtures.
"""

import pytest
from unittest.mock import Mock, patch

import Validation  # noqa: F401


@pytest.fixture
def mock_tkinter():
    """Mock tkinter to avoid GUI creation in tests."""
    with patch("tkinter.Tk") as mock_tk, patch(
        "tkinter.ttk.Frame"
    ) as mock_frame, patch("tkinter.ttk.Button") as mock_button, patch(
        "tkinter.ttk.Label"
    ) as mock_label, patch(
        "tkinter.StringVar"
    ) as mock_stringvar, patch(
        "tkinter.BooleanVar"
    ) as mock_boolvar, patch(
        "tkinter.DoubleVar"
    ) as mock_doublevar, patch(
        "tkinter.IntVar"
    ) as mock_intvar:
        # Configure mocks
        mock_root = Mock()
        mock_child_ids = Mock()
        mock_child_ids.get = Mock(
            side_effect=lambda key, default=0: default
            if isinstance(default, int)
            else 0
        )
        mock_root._last_child_ids = mock_child_ids
        mock_tk.return_value = mock_root

        # Configure variable mocks
        mock_stringvar.return_value = Mock()
        mock_boolvar.return_value = Mock()
        mock_doublevar.return_value = Mock()
        mock_intvar.return_value = Mock()

        yield {
            "tk": mock_tk,
            "root": mock_root,
            "frame": mock_frame,
            "button": mock_button,
            "label": mock_label,
            "stringvar": mock_stringvar,
            "boolvar": mock_boolvar,
            "doublevar": mock_doublevar,
            "intvar": mock_intvar,
        }


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
    }
    validator.protocol_results = {}
    validator.falsification_status = {"primary": [], "secondary": [], "tertiary": []}
    validator.timeout_seconds = 30

    return validator


class TestAPGIValidationGUI:
    """Test suite for APGI Validation GUI."""

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_gui_initialization(
        self, mock_safe_import, mock_validator_class, mock_tkinter
    ):
        """Test GUI initialization with mocked components."""
        from Validation import APGIValidationGUI

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            # Mock the validator import
            mock_safe_import.return_value = Mock()
            mock_validator_class.return_value = Mock()

            # Create GUI instance
            root = Mock()
            gui = APGIValidationGUI(root)

            # Verify initialization
            assert gui.root == root
            assert hasattr(gui, "validator")
            assert hasattr(gui, "_protocol_cache")

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_protocol_selection_validation(
        self, mock_safe_import, mock_validator_class, mock_validator, mock_tkinter
    ):
        """Test protocol selection validation."""
        from Validation import APGIValidationGUI

        # Setup mocks
        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = mock_validator

        with patch("tkinter.messagebox.showerror") as mock_error:
            root = mock_tkinter["root"]
            gui = APGIValidationGUI(root)

            # Mock protocol vars
            gui.protocol_vars = {1: Mock(get=Mock(return_value=True))}
            gui.validator = mock_validator

            # Test invalid protocol number
            gui.protocol_vars[1] = Mock(get=Mock(return_value=True))
            gui.validator.PROTOCOL_TIERS = {99: "invalid"}  # Invalid protocol

            # This should trigger validation error
            with patch.object(gui, "_ensure_ui_consistency"):
                gui.run_validation()

            # Verify error was shown
            mock_error.assert_called()

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_parameter_validation(self, mock_safe_import, mock_validator_class, mock_tkinter):
        """Test parameter slider validation."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = mock_tkinter["root"]
            gui = APGIValidationGUI(root)

        # Mock parameter labels
        gui.param_labels = {"tau_S": Mock()}

        # Test parameter change callback
        gui.on_parameter_change("tau_S", "0.75")

        # Verify label was updated
        gui.param_labels["tau_S"].config.assert_called_with(text="0.750")

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_save_results_validation(
        self, mock_safe_import, mock_validator_class, mock_validator, mock_tkinter
    ):
        """Test save results with validation."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = mock_validator

        with patch("tkinter.filedialog.asksaveasfilename") as mock_dialog, patch(
            "tkinter.messagebox.showinfo"
        ) as mock_info, patch("builtins.open", create=True) as mock_open, patch(
            "json.dump"
        ) as mock_json, patch.object(APGIValidationGUI, "update_parameter_display"):
            root = mock_tkinter["root"]
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
    def test_thread_safety(
        self, mock_safe_import, mock_validator_class, mock_validator, mock_tkinter
    ):
        """Test thread safety mechanisms."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = mock_validator

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = mock_tkinter["root"]
            gui = APGIValidationGUI(root)

        # Test running state thread safety
        assert not gui.is_running

        # Test with lock
        with gui._running_lock:
            gui._is_running = True
            assert gui.is_running

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_progress_tracking(self, mock_safe_import, mock_validator_class, mock_tkinter):
        """Test progress tracking mechanisms."""
        from Validation import APGIValidationGUI

        mock_safe_import.return_value = Mock()
        mock_validator_class.return_value = Mock()

        with patch.object(APGIValidationGUI, "update_parameter_display"):
            root = mock_tkinter["root"]
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
    def test_error_handling_callbacks(self, mock_safe_import, mock_validator_class, mock_tkinter):
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
    def test_settings_management(self, mock_safe_import, mock_validator_class, mock_tkinter):
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
        ), patch("threading.Thread") as mock_thread, patch.object(APGIValidationGUI, "update_parameter_display"):
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
            assert end_time - start_time < 5.0  # 5 seconds max

    @patch("Validation.APGIMasterValidator")
    @patch("Validation.safe_import_module")
    def test_ui_update_queue_performance(self, mock_safe_import, mock_validator_class, mock_tkinter):
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
