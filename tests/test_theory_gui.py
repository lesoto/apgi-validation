"""
Tests for Theory_GUI.py - Testing non-GUI components
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestColorPalette:
    """Test the color palette configuration."""

    def test_colors_defined(self):
        """Test COLORS dictionary is defined."""
        from Theory_GUI import COLORS

        assert isinstance(COLORS, dict)
        assert len(COLORS) > 0

    def test_colors_required_keys(self):
        """Test required color keys exist."""
        from Theory_GUI import COLORS

        required_keys = [
            "primary",
            "success",
            "alert",
            "background",
            "surface",
            "border",
            "text_primary",
            "text_secondary",
            "text_muted",
            "soft_gray",
            "info",
        ]
        for key in required_keys:
            assert key in COLORS

    def test_colors_valid_hex(self):
        """Test color values are valid hex codes."""
        from Theory_GUI import COLORS

        for color in COLORS.values():
            assert color.startswith("#")
            assert len(color) == 7


class TestFonts:
    """Test the font configuration."""

    def test_fonts_defined(self):
        """Test FONTS dictionary is defined."""
        from Theory_GUI import FONTS

        assert isinstance(FONTS, dict)
        assert len(FONTS) > 0

    def test_fonts_required_keys(self):
        """Test required font keys exist."""
        from Theory_GUI import FONTS

        required_keys = ["primary", "monospace", "academic"]
        for key in required_keys:
            assert key in FONTS


class TestFormatDisplayName:
    """Test display name formatting."""

    def test_format_display_name_basic(self):
        """Test basic name formatting."""
        from Theory_GUI import ScriptRunnerGUI

        gui = ScriptRunnerGUI.__new__(ScriptRunnerGUI)
        result = gui._format_display_name("APGI_Bayesian_Estimation")
        assert result == "Bayesian Estimation"

    def test_format_display_name_underscores(self):
        """Test underscore to space conversion."""
        from Theory_GUI import ScriptRunnerGUI

        gui = ScriptRunnerGUI.__new__(ScriptRunnerGUI)
        result = gui._format_display_name("cross_species_scaling")
        assert result == "Cross Species Scaling"

    def test_format_display_name_title_case(self):
        """Test title case conversion."""
        from Theory_GUI import ScriptRunnerGUI

        gui = ScriptRunnerGUI.__new__(ScriptRunnerGUI)
        result = gui._format_display_name("test_name")
        assert result == "Test Name"


class TestDetermineExecutionStrategy:
    """Test execution strategy determination."""

    def test_module_function_priority(self):
        """Test module-level functions have priority."""
        from Theory_GUI import ScriptRunnerGUI

        gui = ScriptRunnerGUI.__new__(ScriptRunnerGUI)
        result = gui._determine_execution_strategy(
            "test",
            [],
            ["run_validation", "other_func"],
        )
        assert result["type"] == "module_function"
        assert result["function"] == "run_validation"

    def test_class_method_fallback(self):
        """Test class methods as fallback."""
        from Theory_GUI import ScriptRunnerGUI

        gui = ScriptRunnerGUI.__new__(ScriptRunnerGUI)
        result = gui._determine_execution_strategy(
            "test",
            [{"name": "TestClass", "methods": ["run_validation"]}],
            [],
        )
        assert result["type"] == "class_method"
        assert result["class"] == "TestClass"
        assert result["method"] == "run_validation"

    def test_exec_module_default(self):
        """Test exec_module as default."""
        from Theory_GUI import ScriptRunnerGUI

        gui = ScriptRunnerGUI.__new__(ScriptRunnerGUI)
        result = gui._determine_execution_strategy("test", [], [], has_main_block=True)
        assert result["type"] == "exec_module"
        assert result["has_main_block"] is True


class TestInferParameters:
    """Test parameter inference from source code."""

    def test_infer_int_params(self):
        """Test integer parameter inference."""
        from Theory_GUI import ScriptRunnerGUI

        gui = ScriptRunnerGUI.__new__(ScriptRunnerGUI)
        source = "n_samples = 100\nn_trials = 50"
        exec_info = {"type": "module_function", "function": "run_validation"}
        result = gui._infer_parameters(source, exec_info)

        assert "n_samples" in result
        assert result["n_samples"]["type"] == "int"
        assert result["n_samples"]["default"] == 100

    def test_infer_float_params(self):
        """Test float parameter inference."""
        from Theory_GUI import ScriptRunnerGUI

        gui = ScriptRunnerGUI.__new__(ScriptRunnerGUI)
        source = "learning_rate = 0.01\nlr = 0.05"
        exec_info = {"type": "module_function", "function": "run_validation"}
        result = gui._infer_parameters(source, exec_info)

        assert "learning_rate" in result or "lr" in result

    def test_infer_threshold_params(self):
        """Test threshold parameter inference."""
        from Theory_GUI import ScriptRunnerGUI

        gui = ScriptRunnerGUI.__new__(ScriptRunnerGUI)
        source = "theta = 0.5\nalpha = 5.0"
        exec_info = {"type": "module_function", "function": "run_validation"}
        result = gui._infer_parameters(source, exec_info)

        assert "theta" in result or "alpha" in result

    def test_infer_dim_params(self):
        """Test dimension parameter inference."""
        from Theory_GUI import ScriptRunnerGUI

        gui = ScriptRunnerGUI.__new__(ScriptRunnerGUI)
        source = "hidden_dim = 128\ninput_dim = 64"
        exec_info = {"type": "module_function", "function": "run_validation"}
        result = gui._infer_parameters(source, exec_info)

        assert "hidden_dim" in result or "input_dim" in result

    def test_infer_empty_source(self):
        """Test parameter inference with empty source."""
        from Theory_GUI import ScriptRunnerGUI

        gui = ScriptRunnerGUI.__new__(ScriptRunnerGUI)
        source = ""
        exec_info = {"type": "module_function", "function": "run_validation"}
        result = gui._infer_parameters(source, exec_info)

        assert isinstance(result, dict)


class TestTorchAvailability:
    """Test torch availability check."""

    def test_torch_available_flag(self):
        """Test TORCH_AVAILABLE is defined."""
        from Theory_GUI import TORCH_AVAILABLE

        assert isinstance(TORCH_AVAILABLE, bool)


class TestMatplotlibBackend:
    """Test matplotlib backend configuration."""

    def test_matplotlib_backend_locked(self):
        """Test matplotlib backend is locked to Agg."""
        import matplotlib

        # Check that use function is locked
        assert hasattr(matplotlib, "use")
        # Try to switch to a GUI backend - should be blocked
        matplotlib.use("Qt5Agg")
        # Backend should still be Agg
        assert matplotlib.get_backend() == "Agg"
