"""
APGI GUI Testing Enhancement Module
====================================

Comprehensive GUI testing with:
- Headless browser testing using Playwright
- Screenshot comparison tests
- UI state transition testing
- Visual regression testing

This module provides advanced GUI testing capabilities for the APGI validation framework.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
import sys
import io
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field


@dataclass
class ScreenshotComparisonResult:
    """Result of screenshot comparison test."""

    test_name: str
    passed: bool
    similarity_score: float
    diff_pixels: int
    baseline_path: Optional[Path] = None
    current_path: Optional[Path] = None
    diff_path: Optional[Path] = None
    error_message: Optional[str] = None


@dataclass
class UIState:
    """Represents a UI state for testing."""

    state_name: str
    expected_widgets: List[str]
    expected_values: Dict[str, Any] = field(default_factory=dict)
    enabled_widgets: List[str] = field(default_factory=list)
    visible_widgets: List[str] = field(default_factory=list)


@dataclass
class UIStateTransition:
    """Represents a UI state transition."""

    from_state: str
    to_state: str
    trigger: str
    expected_side_effects: Dict[str, Any] = field(default_factory=dict)


class ScreenshotComparator:
    """Compare screenshots for visual regression testing."""

    def __init__(self, threshold: float = 0.95, diff_output_dir: Optional[Path] = None):
        """
        Initialize screenshot comparator.

        Args:
            threshold: Minimum similarity score (0.0-1.0) for test to pass
            diff_output_dir: Directory to save diff images
        """
        self.threshold = threshold
        self.diff_output_dir = diff_output_dir or Path("reports/screenshots")
        self.diff_output_dir.mkdir(parents=True, exist_ok=True)

    def compare_images(
        self, baseline: np.ndarray, current: np.ndarray, test_name: str
    ) -> ScreenshotComparisonResult:
        """
        Compare two images and calculate similarity.

        Args:
            baseline: Baseline image array
            current: Current image array
            test_name: Name of the test

        Returns:
            ScreenshotComparisonResult with comparison details
        """
        try:
            # Ensure images are same size
            if baseline.shape != current.shape:
                # Resize current to match baseline
                from PIL import Image

                current_pil = Image.fromarray(current)
                current_pil = current_pil.resize(
                    (baseline.shape[1], baseline.shape[0]), Image.Resampling.LANCZOS
                )
                current = np.array(current_pil)

            # Calculate pixel-wise difference
            diff = np.abs(baseline.astype(float) - current.astype(float))

            # Calculate similarity metrics
            if diff.size == 0:
                similarity = 1.0
                diff_pixels = 0
            else:
                # Normalize difference to 0-1 range
                max_diff = 255.0
                normalized_diff = diff / max_diff

                # Calculate mean similarity
                similarity = 1.0 - np.mean(normalized_diff)

                # Count pixels with significant difference (>5% of max)
                diff_pixels = int(np.sum(normalized_diff > 0.05))

            passed = similarity >= self.threshold

            # Save diff image if test failed
            diff_path = None
            if not passed:
                diff_path = self._save_diff_image(baseline, current, diff, test_name)

            return ScreenshotComparisonResult(
                test_name=test_name,
                passed=passed,
                similarity_score=float(similarity),
                diff_pixels=diff_pixels,
                diff_path=diff_path,
            )

        except Exception as e:
            return ScreenshotComparisonResult(
                test_name=test_name,
                passed=False,
                similarity_score=0.0,
                diff_pixels=0,
                error_message=str(e),
            )

    def _save_diff_image(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        diff: np.ndarray,
        test_name: str,
    ) -> Path:
        """Save diff visualization image."""
        from PIL import Image

        # Create side-by-side comparison
        h, w = baseline.shape[:2]
        comparison = np.zeros((h, w * 3, 3), dtype=np.uint8)

        # Convert grayscale to RGB if needed
        if len(baseline.shape) == 2:
            baseline_rgb = np.stack([baseline] * 3, axis=-1)
            current_rgb = np.stack([current] * 3, axis=-1)
            diff_rgb = np.stack([diff.astype(np.uint8)] * 3, axis=-1)
        else:
            baseline_rgb = baseline
            current_rgb = current
            # Highlight differences in red
            diff_normalized = (
                (diff / diff.max() * 255).astype(np.uint8)
                if diff.max() > 0
                else np.zeros_like(diff)
            )
            diff_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            diff_rgb[:, :, 0] = (
                diff_normalized
                if len(diff_normalized.shape) == 2
                else diff_normalized[:, :, 0]
            )

        comparison[:, :w] = baseline_rgb
        comparison[:, w : 2 * w] = current_rgb  # noqa: E226
        comparison[:, 2 * w :] = diff_rgb  # noqa: E226

        # Save image
        diff_path = self.diff_output_dir / f"{test_name}_diff.png"
        Image.fromarray(comparison).save(diff_path)

        return diff_path

    def update_baseline(self, current: np.ndarray, baseline_path: Path) -> None:
        """Update baseline image with current screenshot."""
        from PIL import Image

        Image.fromarray(current).save(baseline_path)


class HeadlessGUITester:
    """Headless GUI testing using Playwright or mock-based testing."""

    def __init__(self, headless: bool = True, browser_type: str = "chromium"):
        """
        Initialize headless GUI tester.

        Args:
            headless: Run browser in headless mode
            browser_type: Browser type (chromium, firefox, webkit)
        """
        self.headless = headless
        self.browser_type = browser_type
        self.playwright_available = self._check_playwright()
        self.browser: Optional[Any] = None
        self.context: Optional[Any] = None
        self.page: Optional[Any] = None

    def _check_playwright(self) -> bool:
        """Check if Playwright is available."""
        try:
            import playwright  # noqa: F401

            return True
        except ImportError:
            return False

    def _check_browsers_installed(self) -> bool:
        """Check if Playwright browsers are installed."""
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser_method = getattr(p, self.browser_type)
                # Try to launch in headless mode to verify browser exists
                browser = browser_method.launch()
                browser.close()
                return True
        except Exception:
            return False

    async def start_browser(self) -> None:
        """Start browser instance."""
        if not self.playwright_available:
            pytest.skip("Playwright not installed")

        from playwright.async_api import async_playwright

        self.playwright = await async_playwright().start()

        browser_method = getattr(self.playwright, self.browser_type)
        self.browser = await browser_method.launch(headless=self.headless)
        if self.browser is not None:
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080}
            )
            if self.context is not None:
                self.page = await self.context.new_page()

    async def close_browser(self) -> None:
        """Close browser instance."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, "playwright"):
            await self.playwright.stop()

    async def navigate_to(self, url: str) -> None:
        """Navigate to URL."""
        if self.page:
            await self.page.goto(url)

    async def take_screenshot(self, path: Optional[Path] = None) -> np.ndarray:
        """Take screenshot and return as numpy array."""
        if not self.page:
            pytest.skip("Browser page not initialized")

        screenshot_bytes = await self.page.screenshot()

        # Convert to numpy array
        from PIL import Image

        image = Image.open(io.BytesIO(screenshot_bytes))

        if path:
            image.save(path)

        return np.array(image)

    async def click_element(self, selector: str) -> None:
        """Click element by selector."""
        if self.page:
            await self.page.click(selector)

    async def fill_input(self, selector: str, value: str) -> None:
        """Fill input field."""
        if self.page:
            await self.page.fill(selector, value)

    async def select_option(self, selector: str, value: str) -> None:
        """Select option from dropdown."""
        if self.page:
            await self.page.select_option(selector, value)

    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> None:
        """Wait for element to appear."""
        if self.page:
            await self.page.wait_for_selector(selector, timeout=timeout)

    async def get_element_text(self, selector: str) -> str:
        """Get element text content."""
        if self.page:
            return await self.page.text_content(selector) or ""
        return ""

    async def is_element_visible(self, selector: str) -> bool:
        """Check if element is visible."""
        if self.page:
            element = await self.page.query_selector(selector)
            if element:
                return await element.is_visible()
        return False


class UIStateMachineTester:
    """Test UI state transitions and state machine logic."""

    def __init__(self) -> None:
        self.states: Dict[str, UIState] = {}
        self.transitions: List[UIStateTransition] = []
        self.current_state: Optional[str] = None
        self.state_history: List[str] = []

    def register_state(self, state: UIState) -> None:
        """Register a UI state."""
        self.states[state.state_name] = state

    def register_transition(self, transition: UIStateTransition) -> None:
        """Register a state transition."""
        self.transitions.append(transition)

    def get_transition(
        self, from_state: str, trigger: str
    ) -> Optional[UIStateTransition]:
        """Get transition for given state and trigger."""
        for t in self.transitions:
            if t.from_state == from_state and t.trigger == trigger:
                return t
        return None

    def test_state_transition(
        self, from_state: str, trigger: str, mock_ui: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Test a state transition.

        Args:
            from_state: Starting state
            trigger: Trigger event
            mock_ui: Mock UI state dictionary

        Returns:
            (success, error_message)
        """
        transition = self.get_transition(from_state, trigger)
        if not transition:
            return False, f"No transition found for {from_state} -> {trigger}"

        to_state = self.states.get(transition.to_state)
        if not to_state:
            return False, f"Target state {transition.to_state} not registered"

        # Verify expected side effects
        for key, expected_value in transition.expected_side_effects.items():
            if key not in mock_ui:
                return False, f"Expected side effect {key} not found in UI"
            if mock_ui[key] != expected_value:
                return (
                    False,
                    f"Side effect {key}: expected {expected_value}, got {mock_ui[key]}",
                )

        return True, None

    def verify_state_invariants(
        self, state_name: str, mock_ui: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Verify state invariants.

        Args:
            state_name: State to verify
            mock_ui: Mock UI state dictionary

        Returns:
            (passed, list of error messages)
        """
        state = self.states.get(state_name)
        if not state:
            return False, [f"State {state_name} not registered"]

        errors = []

        # Check expected widgets exist
        for widget in state.expected_widgets:
            if widget not in mock_ui:
                errors.append(
                    f"Expected widget {widget} not found in state {state_name}"
                )

        # Check expected values
        for key, expected_value in state.expected_values.items():
            if key not in mock_ui:
                errors.append(f"Expected value {key} not found in state {state_name}")
            elif mock_ui[key] != expected_value:
                errors.append(
                    f"State {state_name}: {key} expected {expected_value}, got {mock_ui[key]}"
                )

        return len(errors) == 0, errors

    def generate_transition_graph(self) -> Dict[str, List[str]]:
        """Generate state transition graph."""
        graph: Dict[str, List[str]] = {state: [] for state in self.states.keys()}
        for t in self.transitions:
            if t.from_state in graph:
                graph[t.from_state].append(t.to_state)
        return graph


@pytest.fixture
def screenshot_comparator():
    """Fixture providing ScreenshotComparator instance."""
    return ScreenshotComparator(threshold=0.95)


@pytest.fixture
def ui_state_machine():
    """Fixture providing UIStateMachineTester instance."""
    return UIStateMachineTester()


@pytest.fixture
def mock_tkinter_gui():
    """Fixture providing mock tkinter GUI for testing."""
    mock_gui = {
        "root": MagicMock(),
        "widgets": {},
        "state": "idle",
        "protocol_vars": {
            i: MagicMock(get=MagicMock(return_value=False)) for i in range(1, 18)
        },
        "param_vars": {},
        "param_labels": {},
        "progress_bar": MagicMock(),
        "run_button": MagicMock(),
        "save_button": MagicMock(),
        "status_label": MagicMock(),
    }
    return mock_gui


class TestGUIStateTransitions:
    """Test suite for GUI state transitions."""

    def test_idle_to_running_transition(self, ui_state_machine, mock_tkinter_gui):
        """Test transition from idle to running state."""
        # Register states
        ui_state_machine.register_state(
            UIState(
                state_name="idle",
                expected_widgets=["run_button", "protocol_checkboxes"],
                expected_values={"is_running": False},
                enabled_widgets=["run_button"],
            )
        )

        ui_state_machine.register_state(
            UIState(
                state_name="running",
                expected_widgets=["progress_bar", "cancel_button", "status_label"],
                expected_values={"is_running": True},
                enabled_widgets=["cancel_button"],
            )
        )

        # Register transition
        ui_state_machine.register_transition(
            UIStateTransition(
                from_state="idle",
                to_state="running",
                trigger="click_run",
                expected_side_effects={"is_running": True},
            )
        )

        # Test transition
        mock_ui = {"is_running": True}
        success, error = ui_state_machine.test_state_transition(
            "idle", "click_run", mock_ui
        )
        assert success, f"Transition failed: {error}"

    def test_running_to_completed_transition(self, ui_state_machine):
        """Test transition from running to completed state."""
        ui_state_machine.register_state(
            UIState(
                state_name="running",
                expected_widgets=["progress_bar"],
                expected_values={"is_running": True},
            )
        )

        ui_state_machine.register_state(
            UIState(
                state_name="completed",
                expected_widgets=["save_button", "results_label"],
                expected_values={"is_running": False, "has_results": True},
                enabled_widgets=["save_button"],
            )
        )

        ui_state_machine.register_transition(
            UIStateTransition(
                from_state="running",
                to_state="completed",
                trigger="validation_complete",
                expected_side_effects={"is_running": False, "has_results": True},
            )
        )

        mock_ui = {"is_running": False, "has_results": True}
        success, error = ui_state_machine.test_state_transition(
            "running", "validation_complete", mock_ui
        )
        assert success, f"Transition failed: {error}"

    def test_running_to_cancelled_transition(self, ui_state_machine):
        """Test transition from running to cancelled state."""
        ui_state_machine.register_state(
            UIState(
                state_name="running",
                expected_widgets=[],
                expected_values={"is_running": True},
            )
        )

        ui_state_machine.register_state(
            UIState(
                state_name="cancelled",
                expected_widgets=[],
                expected_values={"is_running": False, "was_cancelled": True},
            )
        )

        ui_state_machine.register_transition(
            UIStateTransition(
                from_state="running",
                to_state="cancelled",
                trigger="click_cancel",
                expected_side_effects={"is_running": False, "was_cancelled": True},
            )
        )

        mock_ui = {"is_running": False, "was_cancelled": True}
        success, error = ui_state_machine.test_state_transition(
            "running", "click_cancel", mock_ui
        )
        assert success, f"Transition failed: {error}"

    def test_state_invariant_verification(self, ui_state_machine):
        """Test state invariant verification."""
        ui_state_machine.register_state(
            UIState(
                state_name="test_state",
                expected_widgets=["widget1", "widget2"],
                expected_values={"value1": 42, "value2": "test"},
            )
        )

        # Valid state
        valid_ui = {"widget1": True, "widget2": True, "value1": 42, "value2": "test"}
        passed, errors = ui_state_machine.verify_state_invariants(
            "test_state", valid_ui
        )
        assert passed, f"Valid state failed: {errors}"

        # Invalid state - missing widget
        invalid_ui = {"widget1": True, "value1": 42, "value2": "test"}
        passed, errors = ui_state_machine.verify_state_invariants(
            "test_state", invalid_ui
        )
        assert not passed
        assert any("widget2" in e for e in errors)

    def test_invalid_state_transition(self, ui_state_machine):
        """Test handling of invalid state transitions."""
        ui_state_machine.register_state(
            UIState(state_name="state_a", expected_widgets=[])
        )

        ui_state_machine.register_state(
            UIState(state_name="state_b", expected_widgets=[])
        )

        # No transition registered
        mock_ui = {}
        success, error = ui_state_machine.test_state_transition(
            "state_a", "unknown_trigger", mock_ui
        )
        assert not success
        assert "No transition found" in error


class TestScreenshotComparison:
    """Test suite for screenshot comparison functionality."""

    def test_identical_images_comparison(self, screenshot_comparator):
        """Test comparison of identical images."""
        # Create identical test images
        baseline = np.ones((100, 100, 3), dtype=np.uint8) * 128
        current = np.ones((100, 100, 3), dtype=np.uint8) * 128

        result = screenshot_comparator.compare_images(
            baseline, current, "test_identical"
        )

        assert result.passed
        assert result.similarity_score == 1.0
        assert result.diff_pixels == 0

    def test_different_images_comparison(self, screenshot_comparator):
        """Test comparison of different images."""
        baseline = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White
        current = np.zeros((100, 100, 3), dtype=np.uint8)  # Black

        result = screenshot_comparator.compare_images(
            baseline, current, "test_different"
        )

        assert not result.passed
        assert result.similarity_score < 0.95
        assert result.diff_pixels > 0
        assert result.diff_path is not None

    def test_similar_images_comparison(self, screenshot_comparator):
        """Test comparison of similar images with minor differences."""
        baseline = np.ones((100, 100, 3), dtype=np.uint8) * 128
        current = baseline.copy()
        # Add small noise to 1% of pixels
        noise_pixels = np.random.choice(10000, 100, replace=False)
        current.flat[noise_pixels] = 130

        result = screenshot_comparator.compare_images(baseline, current, "test_similar")

        # Should pass with 99% similarity
        assert result.similarity_score > 0.95

    def test_different_size_images(self, screenshot_comparator):
        """Test comparison of images with different sizes."""
        baseline = np.ones((100, 100, 3), dtype=np.uint8) * 128
        current = np.ones((50, 50, 3), dtype=np.uint8) * 128

        result = screenshot_comparator.compare_images(baseline, current, "test_resize")

        # Should handle resize and pass (same content)
        assert result.passed

    def test_comparison_result_serialization(self, screenshot_comparator):
        """Test that comparison results can be serialized."""
        baseline = np.ones((100, 100, 3), dtype=np.uint8) * 128
        current = baseline.copy()

        result = screenshot_comparator.compare_images(
            baseline, current, "test_serializable"
        )

        # Verify result can be converted to dict
        result_dict = {
            "test_name": result.test_name,
            "passed": result.passed,
            "similarity_score": result.similarity_score,
            "diff_pixels": result.diff_pixels,
        }

        assert result_dict["test_name"] == "test_serializable"
        assert result_dict["passed"] == True  # noqa: E712 - numpy.bool_ requires ==


class TestHeadlessBrowser:
    """Test suite for headless browser testing."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HeadlessGUITester().playwright_available
        or not HeadlessGUITester()._check_browsers_installed(),
        reason="Playwright or browsers not installed",
    )
    async def test_browser_launch_and_navigation(self):
        """Test browser can launch and navigate."""
        tester = HeadlessGUITester(headless=True)
        await tester.start_browser()

        try:
            # Navigate to a test page
            await tester.navigate_to("about:blank")

            # Take screenshot
            screenshot = await tester.take_screenshot()
            assert screenshot is not None
            assert screenshot.shape[0] > 0
            assert screenshot.shape[1] > 0
        finally:
            await tester.close_browser()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not HeadlessGUITester().playwright_available
        or not HeadlessGUITester()._check_browsers_installed(),
        reason="Playwright or browsers not installed",
    )
    async def test_element_interaction(self, tmp_path):
        """Test element interaction."""
        tester = HeadlessGUITester(headless=True)
        await tester.start_browser()

        try:
            # Create a test HTML page
            html_content = """
            <html>
                <body>
                    <input id="test-input" type="text" />
                    <button id="test-button">Click</button>
                    <div id="test-output"></div>
                </body>
            </html>
            """

            # Write to temp file
            test_file = tmp_path / "test.html"
            test_file.write_text(html_content)

            # Navigate to test page
            await tester.navigate_to(f"file://{test_file}")

            # Fill input
            await tester.fill_input("#test-input", "test value")

            # Click button
            await tester.click_element("#test-button")

            # Verify element exists
            is_visible = await tester.is_element_visible("#test-input")
            assert is_visible
        finally:
            await tester.close_browser()


class TestGUIValidationWorkflow:
    """Test complete GUI validation workflows."""

    def test_protocol_selection_workflow(self, mock_tkinter_gui, ui_state_machine):
        """Test protocol selection and validation workflow."""
        # Register workflow states
        ui_state_machine.register_state(
            UIState(
                state_name="protocol_selection",
                expected_widgets=["protocol_checkboxes", "select_all_button"],
                expected_values={"has_selection": False},
            )
        )

        ui_state_machine.register_state(
            UIState(
                state_name="protocols_selected",
                expected_widgets=["run_button"],
                expected_values={"has_selection": True},
            )
        )

        ui_state_machine.register_transition(
            UIStateTransition(
                from_state="protocol_selection",
                to_state="protocols_selected",
                trigger="select_protocols",
                expected_side_effects={"has_selection": True},
            )
        )

        # Test selection
        mock_tkinter_gui["has_selection"] = True
        success, error = ui_state_machine.test_state_transition(
            "protocol_selection", "select_protocols", mock_tkinter_gui
        )
        assert success, f"Selection transition failed: {error}"

    def test_parameter_configuration_workflow(self, mock_tkinter_gui, ui_state_machine):
        """Test parameter configuration workflow."""
        ui_state_machine.register_state(
            UIState(
                state_name="parameter_config",
                expected_widgets=["param_sliders", "param_labels", "reset_button"],
                expected_values={"params_modified": False},
            )
        )

        ui_state_machine.register_state(
            UIState(
                state_name="params_modified",
                expected_widgets=["param_sliders", "apply_button"],
                expected_values={"params_modified": True},
            )
        )

        ui_state_machine.register_transition(
            UIStateTransition(
                from_state="parameter_config",
                to_state="params_modified",
                trigger="change_parameter",
                expected_side_effects={"params_modified": True},
            )
        )

        mock_ui = {"params_modified": True}
        success, error = ui_state_machine.test_state_transition(
            "parameter_config", "change_parameter", mock_ui
        )
        assert success, f"Parameter change failed: {error}"


class TestVisualRegression:
    """Test suite for visual regression testing."""

    def test_baseline_image_generation(self, screenshot_comparator, tmp_path):
        """Test baseline image generation and storage."""
        # Create test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 200

        # Save as baseline
        baseline_path = tmp_path / "baseline.png"
        screenshot_comparator.update_baseline(test_image, baseline_path)

        assert baseline_path.exists()

        # Load and verify
        from PIL import Image

        loaded = np.array(Image.open(baseline_path))
        np.testing.assert_array_equal(test_image, loaded)

    def test_regression_detection(self, screenshot_comparator, tmp_path):
        """Test visual regression detection."""
        # Create baseline
        baseline = np.ones((100, 100, 3), dtype=np.uint8) * 200
        baseline_path = tmp_path / "baseline.png"
        screenshot_comparator.update_baseline(baseline, baseline_path)

        # Create significantly different current image
        current = np.zeros((100, 100, 3), dtype=np.uint8)

        # Compare
        result = screenshot_comparator.compare_images(
            baseline, current, "test_regression"
        )

        assert not result.passed
        assert result.diff_path is not None


def run_gui_tests() -> None:
    """Entry point for running GUI tests."""
    print("=" * 80)
    print("APGI GUI Testing Suite")
    print("=" * 80)

    # Run tests using pytest
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_gui_enhanced.py",
            "-v",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    print(f"Tests {'PASSED' if result.returncode == 0 else 'FAILED'}")


if __name__ == "__main__":
    run_gui_tests()
    print("GUI tests completed")
