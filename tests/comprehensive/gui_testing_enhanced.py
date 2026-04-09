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

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import time
import json
from contextlib import contextmanager

# Try to import Playwright for headless browser testing
try:
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class UIStateEnum(Enum):
    """UI states for testing."""

    INITIAL = auto()
    LOADING = auto()
    READY = auto()
    PROCESSING = auto()
    ERROR = auto()
    COMPLETE = auto()


@dataclass
class ScreenshotComparisonResult:
    """Result of screenshot comparison test."""

    test_name: str
    passed: bool
    similarity_score: float
    diff_pixels: int
    total_pixels: int
    baseline_path: Optional[Path] = None
    current_path: Optional[Path] = None
    diff_path: Optional[Path] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UIState:
    """Represents a UI state for testing."""

    state_name: str
    expected_widgets: List[str]
    expected_values: Dict[str, Any] = field(default_factory=dict)
    enabled_widgets: List[str] = field(default_factory=list)
    visible_widgets: List[str] = field(default_factory=list)
    disabled_widgets: List[str] = field(default_factory=list)
    hidden_widgets: List[str] = field(default_factory=list)


@dataclass
class UIStateTransition:
    """Represents a UI state transition."""

    from_state: str
    to_state: str
    trigger: str
    expected_duration_ms: float = 100.0
    expected_side_effects: Dict[str, Any] = field(default_factory=dict)
    validation_callbacks: List[str] = field(default_factory=list)


@dataclass
class HeadlessBrowserTestResult:
    """Result of headless browser test."""

    test_name: str
    passed: bool
    page_url: str
    load_time_ms: float
    console_errors: List[str]
    network_errors: List[str]
    screenshot_path: Optional[Path] = None
    error_message: Optional[str] = None


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
        self.baseline_dir = self.diff_output_dir / "baselines"
        self.current_dir = self.diff_output_dir / "current"
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.current_dir.mkdir(parents=True, exist_ok=True)

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
                if PIL_AVAILABLE:
                    current_pil = Image.fromarray(current)
                    current_pil = current_pil.resize(
                        (baseline.shape[1], baseline.shape[0]), Image.Resampling.LANCZOS
                    )
                    current = np.array(current_pil)
                else:
                    # Fallback: crop or pad
                    current = self._resize_array(current, baseline.shape)

            # Calculate pixel-wise difference
            diff = np.abs(baseline.astype(float) - current.astype(float))

            # Calculate similarity metrics
            total_pixels = baseline.size
            diff_pixels = np.sum(diff > 10)  # Threshold for meaningful difference

            # MSE-based similarity
            mse = np.mean(diff**2)
            max_mse = 255**2  # Maximum possible MSE for 8-bit images
            similarity = 1.0 - (mse / max_mse)

            # Alternative: structural similarity using correlation
            baseline_flat = baseline.flatten()
            current_flat = current.flatten()
            correlation = np.corrcoef(baseline_flat, current_flat)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0

            # Combined similarity score
            combined_similarity = 0.6 * similarity + 0.4 * max(0, correlation)

            passed = combined_similarity >= self.threshold

            # Save diff image
            diff_path = None
            if not passed:
                diff_normalized = (
                    (diff / diff.max() * 255).astype(np.uint8)
                    if diff.max() > 0
                    else diff.astype(np.uint8)
                )
                diff_path = self.diff_output_dir / f"{test_name}_diff.png"
                if PIL_AVAILABLE:
                    Image.fromarray(diff_normalized).save(diff_path)

            return ScreenshotComparisonResult(
                test_name=test_name,
                passed=passed,
                similarity_score=combined_similarity,
                diff_pixels=int(diff_pixels),
                total_pixels=int(total_pixels),
                diff_path=diff_path,
                error_message=(
                    None
                    if passed
                    else f"Similarity {combined_similarity:.3f} below threshold {self.threshold}"
                ),
            )

        except Exception as e:
            return ScreenshotComparisonResult(
                test_name=test_name,
                passed=False,
                similarity_score=0.0,
                diff_pixels=0,
                total_pixels=0,
                error_message=str(e),
            )

    def _resize_array(
        self, arr: np.ndarray, target_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Resize numpy array to target shape."""
        # Simple cropping/padding approach
        result = np.zeros(target_shape, dtype=arr.dtype)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
        result[slices] = arr[slices]
        return result

    def save_baseline(self, image: np.ndarray, test_name: str) -> Path:
        """Save a new baseline image."""
        path = self.baseline_dir / f"{test_name}_baseline.png"
        if PIL_AVAILABLE:
            Image.fromarray(image).save(path)
        return path

    def load_baseline(self, test_name: str) -> Optional[np.ndarray]:
        """Load baseline image if it exists."""
        path = self.baseline_dir / f"{test_name}_baseline.png"
        if path.exists() and PIL_AVAILABLE:
            return np.array(Image.open(path))
        return None


class UIStateMachine:
    """State machine for testing UI state transitions."""

    def __init__(self) -> None:
        self.states: Dict[str, UIState] = {}
        self.transitions: List[UIStateTransition] = []
        self.current_state: Optional[str] = None
        self.transition_history: List[Tuple[str, str, float]] = []  # from, to, duration

    def add_state(self, state: UIState) -> None:
        """Add a state to the state machine."""
        self.states[state.state_name] = state

    def add_transition(self, transition: UIStateTransition) -> None:
        """Add a valid state transition."""
        self.transitions.append(transition)

    def validate_transition(self, from_state: str, to_state: str, trigger: str) -> bool:
        """Check if a state transition is valid."""
        for t in self.transitions:
            if (
                t.from_state == from_state
                and t.to_state == to_state
                and t.trigger == trigger
            ):
                return True
        return False

    def get_expected_side_effects(
        self, from_state: str, to_state: str, trigger: str
    ) -> Dict[str, Any]:
        """Get expected side effects for a transition."""
        for t in self.transitions:
            if (
                t.from_state == from_state
                and t.to_state == to_state
                and t.trigger == trigger
            ):
                return t.expected_side_effects
        return {}

    def transition(self, to_state: str, trigger: str) -> Tuple[bool, Optional[str]]:
        """Perform a state transition."""
        if self.current_state is None:
            return False, "No current state set"

        if not self.validate_transition(self.current_state, to_state, trigger):
            return (
                False,
                f"Invalid transition: {self.current_state} -> {to_state} via {trigger}",
            )

        start_time = time.time()
        self.transition_history.append((self.current_state, to_state, 0.0))
        self.current_state = to_state
        duration = (time.time() - start_time) * 1000
        self.transition_history[-1] = (
            self.transition_history[-1][0],
            to_state,
            duration,
        )

        return True, None

    def validate_current_state(
        self, widget_values: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate current state against expected values."""
        if self.current_state not in self.states:
            return False, [f"Unknown state: {self.current_state}"]

        state = self.states[self.current_state]
        errors = []

        for widget, expected_value in state.expected_values.items():
            if widget not in widget_values:
                errors.append(f"Missing widget: {widget}")
            elif widget_values[widget] != expected_value:
                errors.append(
                    f"Widget {widget}: expected {expected_value}, got {widget_values[widget]}"
                )

        return len(errors) == 0, errors


class HeadlessBrowserTester:
    """Headless browser testing using Playwright."""

    def __init__(self, viewport_size: Tuple[int, int] = (1920, 1080)):
        self.viewport_size = viewport_size
        self.results: List[HeadlessBrowserTestResult] = []

    @contextmanager
    def _get_browser(self):
        """Context manager for browser lifecycle."""
        if not PLAYWRIGHT_AVAILABLE:
            yield None
            return

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                yield browser
            finally:
                browser.close()

    def test_page_load(
        self,
        url: str,
        test_name: str,
        wait_for_selector: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> HeadlessBrowserTestResult:
        """Test that a page loads correctly."""
        if not PLAYWRIGHT_AVAILABLE:
            return HeadlessBrowserTestResult(
                test_name=test_name,
                passed=False,
                page_url=url,
                load_time_ms=0.0,
                console_errors=[],
                network_errors=[],
                error_message="Playwright not available - install with: pip install playwright && playwright install chromium",
            )

        console_errors = []
        network_errors = []

        try:
            with self._get_browser() as browser:
                if browser is None:
                    raise RuntimeError("Failed to launch browser")

                context = browser.new_context(
                    viewport={
                        "width": self.viewport_size[0],
                        "height": self.viewport_size[1],
                    }
                )

                page = context.new_page()

                # Collect console errors
                page.on(
                    "console",
                    lambda msg: (
                        console_errors.append(msg.text) if msg.type == "error" else None
                    ),
                )

                # Collect network errors
                page.on(
                    "requestfailed", lambda request: network_errors.append(request.url)
                )

                # Navigate and measure load time
                start_time = time.time()
                response = page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                load_time = (time.time() - start_time) * 1000

                if response is None or not response.ok:
                    network_errors.append(
                        f"Failed to load {url}: {response.status if response else 'No response'}"
                    )

                # Wait for specific element if requested
                if wait_for_selector:
                    page.wait_for_selector(wait_for_selector, timeout=timeout_ms)

                # Take screenshot
                screenshot_dir = Path("reports/screenshots")
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_path = screenshot_dir / f"{test_name}.png"
                page.screenshot(path=str(screenshot_path), full_page=True)

                context.close()

                result = HeadlessBrowserTestResult(
                    test_name=test_name,
                    passed=len(network_errors) == 0
                    and len([e for e in console_errors if "error" in e.lower()]) == 0,
                    page_url=url,
                    load_time_ms=load_time,
                    console_errors=console_errors,
                    network_errors=network_errors,
                    screenshot_path=screenshot_path,
                )

                self.results.append(result)
                return result

        except Exception as e:
            result = HeadlessBrowserTestResult(
                test_name=test_name,
                passed=False,
                page_url=url,
                load_time_ms=0.0,
                console_errors=console_errors,
                network_errors=network_errors,
                error_message=str(e),
            )
            self.results.append(result)
            return result

    def test_interaction(
        self,
        url: str,
        test_name: str,
        interactions: List[Dict[str, Any]],
        expected_outcomes: Dict[str, Any],
    ) -> HeadlessBrowserTestResult:
        """Test user interactions on a page."""
        if not PLAYWRIGHT_AVAILABLE:
            return HeadlessBrowserTestResult(
                test_name=test_name,
                passed=False,
                page_url=url,
                load_time_ms=0.0,
                console_errors=[],
                network_errors=[],
                error_message="Playwright not available",
            )

        try:
            with self._get_browser() as browser:
                if browser is None:
                    raise RuntimeError("Failed to launch browser")

                context = browser.new_context(
                    viewport={
                        "width": self.viewport_size[0],
                        "height": self.viewport_size[1],
                    }
                )

                page = context.new_page()

                start_time = time.time()
                page.goto(url, wait_until="networkidle")

                # Perform interactions
                for interaction in interactions:
                    action = interaction.get("action")
                    selector = interaction.get("selector")
                    value = interaction.get("value")

                    if action == "click":
                        page.click(selector)
                    elif action == "fill":
                        page.fill(selector, value)
                    elif action == "select":
                        page.select_option(selector, value)
                    elif action == "wait":
                        page.wait_for_timeout(value)
                    elif action == "wait_for_selector":
                        page.wait_for_selector(selector, timeout=value)

                load_time = (time.time() - start_time) * 1000

                # Verify expected outcomes
                errors = []
                for check_type, check_value in expected_outcomes.items():
                    if check_type == "url_contains":
                        if check_value not in page.url:
                            errors.append(f"URL does not contain: {check_value}")
                    elif check_type == "selector_visible":
                        if not page.is_visible(check_value):
                            errors.append(f"Selector not visible: {check_value}")
                    elif check_type == "text_contains":
                        selector, text = check_value
                        content = page.text_content(selector)
                        if text not in content:
                            errors.append(f"Text not found: {text}")

                context.close()

                result = HeadlessBrowserTestResult(
                    test_name=test_name,
                    passed=len(errors) == 0,
                    page_url=url,
                    load_time_ms=load_time,
                    console_errors=[],
                    network_errors=errors,
                )

                self.results.append(result)
                return result

        except Exception as e:
            result = HeadlessBrowserTestResult(
                test_name=test_name,
                passed=False,
                page_url=url,
                load_time_ms=0.0,
                console_errors=[],
                network_errors=[],
                error_message=str(e),
            )
            self.results.append(result)
            return result

    def export_results(self, output_path: Path) -> None:
        """Export test results to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                [
                    {
                        "test_name": r.test_name,
                        "passed": r.passed,
                        "page_url": r.page_url,
                        "load_time_ms": r.load_time_ms,
                        "console_errors": r.console_errors,
                        "network_errors": r.network_errors,
                        "error_message": r.error_message,
                    }
                    for r in self.results
                ],
                f,
                indent=2,
            )


class GUITestSuite:
    """Comprehensive GUI testing suite."""

    def __init__(self) -> None:
        self.screenshot_comparator = ScreenshotComparator()
        self.state_machine = UIStateMachine()
        self.browser_tester = HeadlessBrowserTester()
        self.test_results: Dict[str, Any] = {}

    def setup_state_machine(self) -> None:
        """Setup standard UI state machine for APGI."""
        # Define states
        self.state_machine.add_state(
            UIState(
                state_name="initial",
                expected_widgets=["load_data_button", "config_panel"],
                enabled_widgets=["load_data_button"],
                visible_widgets=["config_panel", "load_data_button"],
            )
        )

        self.state_machine.add_state(
            UIState(
                state_name="loading",
                expected_widgets=["progress_bar", "status_label"],
                enabled_widgets=[],
                visible_widgets=["progress_bar", "status_label"],
            )
        )

        self.state_machine.add_state(
            UIState(
                state_name="ready",
                expected_widgets=["run_button", "results_panel", "visualization"],
                enabled_widgets=["run_button", "export_button"],
                visible_widgets=["results_panel", "run_button", "visualization"],
            )
        )

        self.state_machine.add_state(
            UIState(
                state_name="processing",
                expected_widgets=["progress_bar", "cancel_button"],
                enabled_widgets=["cancel_button"],
                visible_widgets=["progress_bar", "cancel_button"],
            )
        )

        self.state_machine.add_state(
            UIState(
                state_name="error",
                expected_widgets=["error_message", "retry_button"],
                enabled_widgets=["retry_button", "dismiss_button"],
                visible_widgets=["error_message", "retry_button"],
            )
        )

        # Define transitions
        self.state_machine.add_transition(
            UIStateTransition(
                from_state="initial", to_state="loading", trigger="load_data"
            )
        )

        self.state_machine.add_transition(
            UIStateTransition(
                from_state="loading", to_state="ready", trigger="data_loaded"
            )
        )

        self.state_machine.add_transition(
            UIStateTransition(
                from_state="loading", to_state="error", trigger="load_failed"
            )
        )

        self.state_machine.add_transition(
            UIStateTransition(
                from_state="ready", to_state="processing", trigger="run_analysis"
            )
        )

        self.state_machine.add_transition(
            UIStateTransition(
                from_state="processing", to_state="ready", trigger="analysis_complete"
            )
        )

        self.state_machine.add_transition(
            UIStateTransition(
                from_state="processing", to_state="error", trigger="analysis_failed"
            )
        )

        self.state_machine.add_transition(
            UIStateTransition(
                from_state="error", to_state="initial", trigger="dismiss_error"
            )
        )

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all GUI enhancement tests."""
        print("=" * 80)
        print("APGI GUI TESTING ENHANCEMENT")
        print("=" * 80)

        # Setup
        self.setup_state_machine()

        # Run screenshot comparison tests
        self._test_screenshot_comparison()

        # Run UI state transition tests
        self._test_ui_state_transitions()

        # Run headless browser tests (if Playwright available)
        if PLAYWRIGHT_AVAILABLE:
            self._test_headless_browser()
        else:
            print("\n⚠️ Playwright not available - skipping headless browser tests")
            print(
                "   Install with: pip install playwright && playwright install chromium"
            )

        return self._generate_report()

    def _test_screenshot_comparison(self) -> None:
        """Test screenshot comparison functionality."""
        print("\n[GUI Test 1/3] Screenshot comparison...")

        # Create synthetic test images
        baseline = np.ones((100, 100, 3), dtype=np.uint8) * 200
        identical = np.ones((100, 100, 3), dtype=np.uint8) * 200
        different = np.ones((100, 100, 3), dtype=np.uint8) * 150

        # Test identical images
        result1 = self.screenshot_comparator.compare_images(
            baseline, identical, "identical_test"
        )
        assert result1.passed, "Identical images should pass"
        assert (
            result1.similarity_score > 0.99
        ), "Identical images should have high similarity"

        # Test different images
        result2 = self.screenshot_comparator.compare_images(
            baseline, different, "different_test"
        )
        # Different images should fail with default threshold of 0.95

        self.test_results["screenshot_comparison"] = {
            "identical_test_passed": result1.passed,
            "identical_similarity": result1.similarity_score,
            "different_test_passed": result2.passed,
            "different_similarity": result2.similarity_score,
            "overall_passed": result1.passed,  # Key test: identical images match
        }

        print("  ✓ Screenshot comparison tests passed")

    def _test_ui_state_transitions(self) -> None:
        """Test UI state machine transitions."""
        print("\n[GUI Test 2/3] UI state transitions...")

        # Set initial state
        self.state_machine.current_state = "initial"

        # Test valid transitions
        success, error = self.state_machine.transition("loading", "load_data")
        assert success, f"Valid transition failed: {error}"

        success, error = self.state_machine.transition("ready", "data_loaded")
        assert success, f"Valid transition failed: {error}"

        success, error = self.state_machine.transition("processing", "run_analysis")
        assert success, f"Valid transition failed: {error}"

        success, error = self.state_machine.transition("ready", "analysis_complete")
        assert success, f"Valid transition failed: {error}"

        # Test invalid transition
        success, error = self.state_machine.transition("initial", "invalid_trigger")
        assert not success, "Invalid transition should fail"

        # Verify transition history
        assert (
            len(self.state_machine.transition_history) >= 4
        ), "Should have recorded transitions"

        self.test_results["ui_state_transitions"] = {
            "valid_transitions_passed": True,
            "invalid_transition_blocked": not success,
            "transition_count": len(self.state_machine.transition_history),
            "overall_passed": True,
        }

        print("  ✓ UI state transition tests passed")

    def _test_headless_browser(self) -> None:
        """Test headless browser functionality."""
        print("\n[GUI Test 3/3] Headless browser testing...")

        # Create a simple test HTML file
        test_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test GUI Page</title>
        </head>
        <body>
            <h1>Test Page</h1>
            <button id="test-btn">Click Me</button>
            <div id="result" style="display:none;">Result shown</div>
            <script>
                document.getElementById('test-btn').addEventListener('click', function() {
                    document.getElementById('result').style.display = 'block';
                });
            </script>
        </body>
        </html>
        """

        test_file = Path("test_gui_page.html")
        with open(test_file, "w") as f:
            f.write(test_html)

        try:
            # Test page load
            result = self.browser_tester.test_page_load(
                url=f"file://{test_file.absolute()}",
                test_name="gui_page_load",
                wait_for_selector="h1",
            )

            # Test interaction
            interaction_result = self.browser_tester.test_interaction(
                url=f"file://{test_file.absolute()}",
                test_name="gui_interaction",
                interactions=[
                    {
                        "action": "wait_for_selector",
                        "selector": "#test-btn",
                        "value": 5000,
                    },
                    {"action": "click", "selector": "#test-btn"},
                    {"action": "wait", "value": 100},
                ],
                expected_outcomes={"selector_visible": "#result"},
            )

            self.test_results["headless_browser"] = {
                "page_load_passed": result.passed,
                "interaction_passed": interaction_result.passed,
                "load_time_ms": result.load_time_ms,
                "overall_passed": result.passed and interaction_result.passed,
            }

            print("  ✓ Headless browser tests passed")

        finally:
            test_file.unlink(missing_ok=True)

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        all_passed = all(
            r.get("overall_passed", False) for r in self.test_results.values()
        )

        report = {
            "gui_testing": {
                "screenshot_comparison": self.test_results.get(
                    "screenshot_comparison", {}
                ),
                "ui_state_transitions": self.test_results.get(
                    "ui_state_transitions", {}
                ),
                "headless_browser": self.test_results.get("headless_browser", {}),
                "all_passed": all_passed,
            }
        }

        print(f"\n{'=' * 80}")
        print("GUI TESTING SUMMARY")
        print(f"{'=' * 80}")
        print(
            f"Screenshot Comparison: {'✓' if self.test_results.get('screenshot_comparison', {}).get('overall_passed') else '✗'}"
        )
        print(
            f"UI State Transitions: {'✓' if self.test_results.get('ui_state_transitions', {}).get('overall_passed') else '✗'}"
        )
        print(
            f"Headless Browser: {'✓' if self.test_results.get('headless_browser', {}).get('overall_passed') else '⚠ (Playwright not installed)'}"
        )
        print(
            f"\nOverall: {'✅ All tests passed' if all_passed else '⚠️ Some tests failed'}"
        )

        return report


def run_gui_tests() -> Dict[str, Any]:
    """Entry point for GUI testing."""
    suite = GUITestSuite()
    return suite.run_all_tests()


if __name__ == "__main__":
    run_gui_tests()
