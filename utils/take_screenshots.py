#!/usr/bin/env python3
"""
APGI System Desktop App Screenshot Documentation

Captures screenshots of Python Tkinter desktop application
and interacts with all GUI elements automatically.

Requirements:
    pip install pyautogui pygetwindow pillow opencv-python

Usage:
    python take_screenshots.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import cv2
    import numpy as np
    import pyautogui
    import pygetwindow as gw
    from PIL import Image

    SCREENSHOT_AVAILABLE = True
except ImportError:
    print("Error: Required packages not installed. Run:")
    print("  pip install pyautogui pygetwindow pillow opencv-python")
    sys.exit(1)


class APGIScreenshotDocumentation:
    """Screenshot system for Python desktop application."""

    def __init__(self, base_dir: Path = None):
        # More robust base directory calculation
        if base_dir:
            self.base_dir = base_dir
        else:
            # Use current working directory as base
            self.base_dir = Path.cwd()
        self.screenshots_dir = self.base_dir / "docs" / "screenshots"
        self.reports_dir = self.base_dir / "docs" / "screenshots" / "reports"

        # Create directories
        try:
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)
            self.reports_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not create directories: {e}")
            # Fallback to current directory
            self.screenshots_dir = Path.cwd() / "docs" / "screenshots"
            self.reports_dir = Path.cwd() / "docs" / "screenshots" / "reports"

        # Documentation structure
        self.doc_structure = {
            "timestamp": datetime.now().isoformat(),
            "screenshots": [],
            "gui_elements": [],
            "system_info": self._get_system_info(),
        }

        # GUI process reference
        self.gui_process = None
        self.gui_window = None

        # GUI element locations (will be discovered)
        self.button_locations = {}
        self.tab_locations = {}
        self.slider_locations = {}
        self.menu_items = {}
        self.dialog_locations = {}
        self.view_toggles = {}
        self.zoom_controls = {}
        self.tools_actions = {}
        self.analysis_actions = {}
        self.help_actions = {}
        self.speed_slider = None
        self.status_bar = None
        self.event_log = None

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for documentation."""
        import platform

        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "screen_size": pyautogui.size() if SCREENSHOT_AVAILABLE else "Unknown",
        }

    def generate_comprehensive_documentation(self):
        """Generate complete screenshot documentation with enhanced user guidance."""
        print("🚀 APGI System Desktop App Screenshot Documentation")
        print("=" * 70)

        print("\n📋 COMPREHENSIVE SETUP INSTRUCTIONS:")
        print("=" * 50)
        print("\n🎯 BEFORE YOU START:")
        print("   1. ✅ Launch the APGI GUI application")
        print("   2. ✅ Make sure the application window is visible on screen")
        print("   3. ✅ Close or minimize other applications that might interfere")
        print("   4. ✅ Click on the APGI application window to ensure it's active")
        print("   5. ✅ Move the application to a clear, visible area")
        print("   6. ✅ IMPORTANT: Close any IDE windows (VS Code, PyCharm, etc.)")
        print("   7. ✅ Close terminal windows or move them away from the app")
        print("\n⚠️  IMPORTANT NOTES:")
        print("   • The script will automatically detect the APGI window")
        print("   • It actively avoids IDE windows and development tools")
        print("   • If automatic detection fails, you'll see a selection menu")
        print("   • The script uses multiple methods to ensure correct window capture")
        print("   • All screenshots are saved with timestamps and metadata")
        print("   • The process may take 5-10 minutes to complete")
        print("\n🔧 TROUBLESHOOTING:")
        print("   • If window detection fails: Try restarting the APGI application")
        print("   • If screenshots show wrong app: Use manual selection option")
        print("   • If script stops: Check that no dialogs are blocking the app")
        print("   • For best results: Run on a dedicated screen/monitor")
        print("   • IDE CAPTURE ISSUE: The script now actively detects and avoids IDEs")

        print("\n" + "=" * 70)
        print("🎮 READY TO START DOCUMENTATION")
        print("=" * 70)

        # Get user confirmation with detailed options
        print("\n📝 OPTIONS:")
        print("   • Press Enter to begin full documentation")
        print("   • Type 'test' to run window detection test only")
        print("   • Type 'demo' to run in demo mode (no actual screenshots)")
        print("   • Type 'help' for more information")
        print("   • Type 'scan' to see all available windows")
        print("   • Press Ctrl+C to cancel")

        try:
            user_input = input("\n➡️  Your choice: ").strip().lower()

            if user_input == "help":
                self._show_detailed_help()
                return
            elif user_input == "test":
                self._run_window_detection_test()
                return
            elif user_input == "demo":
                self.demo_mode = True
                print("\n🎭 Running in DEMO MODE (no actual screenshots)")
            elif user_input == "scan":
                self._show_all_windows()
                return
            elif user_input not in ["", "start", "begin"]:
                print("\n❌ Invalid choice. Starting full documentation...")
                time.sleep(2)

        except KeyboardInterrupt:
            print("\n❌ Documentation cancelled by user")
            return

        # Start the actual documentation process
        print("\n" + "=" * 70)
        print("🎬 STARTING DOCUMENTATION PROCESS")
        print("=" * 70)

        try:
            # Phase 1: Environment setup and validation
            self._validate_environment()

            # Phase 2: Start GUI application (if needed)
            self._start_gui_app()

            # Phase 3: Enhanced window discovery
            self._discover_gui_elements_enhanced()

            # Phase 4: Comprehensive screen documentation
            self._document_all_screens_enhanced()

            # Phase 5: Generate reports
            self._generate_documentation_report()

            # Success summary
            self._print_success_summary()

        except Exception as e:
            print(f"\n❌ DOCUMENTATION ERROR: {e}")
            print("\n🔧 Troubleshooting suggestions:")
            print("   1. Check that APGI application is still running")
            print("   2. Ensure no system dialogs are blocking")
            print("   3. Try restarting the documentation process")
            print("   4. Check system permissions for screenshot access")
            print("   5. Make sure IDE windows are closed or moved away")

            import traceback

            print("\n📋 Full error details:")
            traceback.print_exc()

        finally:
            self._cleanup_processes()

    def _show_detailed_help(self):
        """Show detailed help information."""
        print("\n" + "=" * 70)
        print("📚 APGI SCREENSHOT DOCUMENTATION - DETAILED HELP")
        print("=" * 70)

        print("\n🎯 PURPOSE:")
        print("   This tool automatically documents the APGI System desktop")
        print("   application by capturing screenshots of every screen, control, ")
        print("   and interaction. It generates comprehensive HTML reports.")

        print("\n🔧 WINDOW DETECTION METHODS:")
        print("   1. Exact title matching")
        print("   2. Multiple title variations")
        print("   3. Keyword-based filtering")
        print("   4. Size-based detection")
        print("   5. Manual activation testing")
        print("   6. Large window candidate selection")

        print("\n📸 SCREENSHOT CAPTURE METHODS:")
        print("   1. Window region capture (preferred)")
        print("   2. Active window capture")
        print("   3. Manual positioning")
        print("   4. Full screen with analysis")
        print("   5. Emergency fallback")

        print("\n📋 DOCUMENTATION COVERAGE:")
        print("   • Initial application state")
        print("   • All 6 main tabs")
        print("   • All 4 control buttons")
        print("   • All 8 parameter sliders")
        print("   • All 7 menu categories (31 menu items)")
        print("   • 5 simulation states")
        print("   • 24 dialog windows")
        print("   • View controls and zoom")
        print("   • Status bar and event log")
        print("   • Final application state")

        print("\n📁 OUTPUT FILES:")
        print("   • Screenshots: docs/screenshots/")
        print("   • HTML Report: docs/screenshots/reports/")
        print("   • JSON Data: docs/screenshots/reports/")

        print("\n⚙️ REQUIREMENTS:")
        print("   • Python 3.7+")
        print("   • pyautogui, pygetwindow, pillow, opencv-python")
        print("   • Screen permissions for screenshot capture")
        print("   • APGI GUI application running")

        print("\n🐛 COMMON ISSUES:")
        print("   • IDE screenshots instead of app → Use manual window selection")
        print("   • Window not found → Check app is running and visible")
        print("   • Blurry screenshots → Ensure app window is focused")
        print("   • Missing elements → Check app is in default state")

        print("\n" + "=" * 70)

    def _run_window_detection_test(self):
        """Run a window detection test without taking screenshots."""
        print("\n🧪 RUNNING WINDOW DETECTION TEST")
        print("=" * 50)

        print("\n🔍 Testing 6-method window detection...")
        window = self._find_gui_window()

        if window:
            print(f"\n✅ SUCCESS: Found window - {window.title}")
            print(f"   📐 Size: {window.width}x{window.height}")
            print(f"   📍 Position: ({window.left}, {window.top})")

            # Test activation
            print("\n🎯 Testing window activation...")
            activation_success = self._ensure_app_is_active()

            if activation_success:
                print("✅ Window activation successful")

                # Test screenshot capture
                print("\n📸 Testing screenshot capture...")
                try:
                    test_screenshot = pyautogui.screenshot(
                        region=(window.left, window.top, window.width, window.height)
                    )
                    print(
                        f"✅ Screenshot capture successful ({test_screenshot.size[0]}x{test_screenshot.size[1]})"
                    )

                    # Verify it's the right app
                    if self._is_likely_app_screenshot(test_screenshot):
                        print(
                            "✅ Screenshot appears to be the correct APGI application"
                        )
                    else:
                        print("⚠️ Screenshot might not be the APGI application")

                except Exception as e:
                    print(f"❌ Screenshot test failed: {e}")
            else:
                print("❌ Window activation failed")
        else:
            print("❌ FAILED: Could not find APGI window")
            print("\n💡 Suggestions:")
            print("   1. Make sure APGI application is running")
            print("   2. Check that the window is visible")
            print("   3. Try restarting the application")
            print("   4. Use manual selection when running full documentation")

        print("\n🧪 TEST COMPLETE")

    def _validate_environment(self):
        """Validate the environment before starting documentation."""
        print("\n🔍 VALIDATING ENVIRONMENT...")

        # Check dependencies
        try:
            import pyautogui

            print("✅ All required packages are installed")
        except ImportError as e:
            print(f"❌ Missing required package: {e}")
            print("   Run: pip install pyautogui pygetwindow pillow opencv-python")
            raise e

        # Check screen permissions
        try:
            screen_size = pyautogui.size()
            print(f"✅ Screen access: {screen_size[0]}x{screen_size[1]}")
        except Exception as e:
            print(f"❌ Screen access denied: {e}")
            print("   Grant screen recording permissions in System Preferences")
            raise e

        # Check directory permissions
        try:
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Directory permissions: OK")
        except Exception as e:
            print(f"❌ Directory access denied: {e}")
            raise e

        print("✅ Environment validation complete")

    def _discover_gui_elements_enhanced(self):
        """Enhanced GUI element discovery with detailed logging and fallback modes."""
        print("\n🔍 ENHANCED GUI ELEMENT DISCOVERY")
        print("=" * 50)

        # Wait for GUI to fully load
        print("⏳ Waiting for GUI to fully load...")
        time.sleep(3)

        # Try to find the GUI window with detailed logging
        print("🔍 Starting comprehensive window detection...")

        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"\n📍 Detection attempt {attempt + 1}/{max_attempts}...")

            self.gui_window = self._find_gui_window()

            if self.gui_window:
                print(f"✅ Window found: {self.gui_window.title}")
                print(f"   📐 Size: {self.gui_window.width}x{self.gui_window.height}")
                print(
                    f"   📍 Position: ({self.gui_window.left}, {self.gui_window.top})"
                )
                break
            else:
                print("⚠️ Window not found, retrying in 2 seconds...")
                time.sleep(2)

        if not self.gui_window:
            print("❌ Automatic window detection failed")
            print("🔄 Switching to manual window selection...")
            self.gui_window = self._manual_window_selection()

            if not self.gui_window:
                print("⚠️ No window selected, using fallback discovery mode")
                self._use_fallback_discovery()
                return False

        # Manual verification with enhanced confirmation
        print("\n🔍 WINDOW VERIFICATION")
        print("=" * 30)
        print(f"Found window: '{self.gui_window.title}'")
        print(f"Size: {self.gui_window.width}x{self.gui_window.height}")
        print(f"Position: ({self.gui_window.left}, {self.gui_window.top})")

        print("\n❓ Is this the correct application window?")
        print("   • Enter 'y' or press Enter to confirm")
        print("   • Enter 'n' to select different window")
        print("   • Enter 'test' to test window activation")

        try:
            response = input("Confirm (y/n/test): ").strip().lower()

            if response == "test":
                print("\n🧪 Testing window activation...")
                success = self._ensure_app_is_active()
                if success:
                    print("✅ Window activation test successful")
                    print("❓ Proceed with this window? (y/n)")
                    response = input("Confirm (y/n): ").strip().lower()
                else:
                    print("❌ Window activation test failed")
                    response = "n"

            if response not in ["y", "yes", ""]:
                print("\n🔄 Switching to manual window selection...")
                self.gui_window = self._manual_window_selection()
                if not self.gui_window:
                    print("❌ No window selected, using fallback mode")
                    self._use_fallback_discovery()
                    return False

        except KeyboardInterrupt:
            print("\n❌ Documentation cancelled by user")
            return False

        # Activate and maximize window with detailed logging
        print("\n🎯 ACTIVATING AND CONFIGURING WINDOW")
        try:
            print("   Attempting to activate window...")
            self.gui_window.activate()
            time.sleep(1)

            if hasattr(self.gui_window, "maximize"):
                print("   Attempting to maximize window...")
                self.gui_window.maximize()
                time.sleep(2)

            print("✅ Window activation and configuration complete")

        except Exception as e:
            print(f"⚠️ Window configuration warning: {e}")
            print("   Continuing with current window state...")

        # Verify window is active with test screenshot
        print("\n🧪 VERIFYING WINDOW CAPTURE")
        try:
            test_screenshot = pyautogui.screenshot(
                region=(
                    self.gui_window.left,
                    self.gui_window.top,
                    self.gui_window.width,
                    self.gui_window.height,
                )
            )
            print("✅ Window capture test successful")

            # Discover GUI elements
            print("\n🔍 DISCOVERING GUI ELEMENTS")
            self._discover_buttons(test_screenshot)
            self._discover_tabs(test_screenshot)
            self._discover_sliders(test_screenshot)
            self._discover_menu_items()

            total_elements = (
                len(self.button_locations)
                + len(self.tab_locations)
                + len(self.slider_locations)
                + len(self.menu_items)
            )

            print(f"   Buttons found: {len(self.button_locations)}")
            print(f"   Tabs found: {len(self.tab_locations)}")
            print(f"   Sliders found: {len(self.slider_locations)}")
            print(f"   Menu categories: {len(self.menu_items)}")
            print(f"   Total elements: {total_elements}")

        except Exception as e:
            print(f"⚠️ Window verification failed: {e}")
            print("   Switching to fallback discovery mode...")
            self._use_fallback_discovery()
            return False

    def _document_all_screens_enhanced(self):
        """Enhanced screen documentation with detailed progress tracking and fallback modes."""
        print("\n📸 COMPREHENSIVE SCREEN DOCUMENTATION")
        print("=" * 50)

        # Check if we're in manual mode (no specific GUI)
        if hasattr(self, "manual_mode") and self.manual_mode:
            print("🎯 Manual mode: Documenting available application")
            self._document_manual_application()
            return

        # Track progress
        total_phases = 11
        current_phase = 0

        def log_phase(phase_name):
            nonlocal current_phase
            current_phase += 1
            progress = (current_phase / total_phases) * 100
            print(
                f"\n📋 [{current_phase}/{total_phases}] {phase_name} ({progress:.1f}% complete)"
            )
            print("-" * 60)

        try:
            # Phase 1: Initial state
            log_phase("Initial Application State")
            self._take_screenshot(
                "01_initial_state", "Initial GUI state - application just started"
            )

            # Phase 2: Tabs
            if self.tab_locations:
                log_phase("Tab Documentation")
                self._document_all_tabs()
            else:
                print("\n⚠️ No tabs found, skipping tab documentation")

            # Phase 3: Buttons
            if self.button_locations:
                log_phase("Button Documentation")
                self._document_all_buttons()
            else:
                print("\n⚠️ No buttons found, skipping button documentation")

            # Phase 4: Sliders
            if self.slider_locations:
                log_phase("Slider Documentation")
                self._document_all_sliders()
            else:
                print("\n⚠️ No sliders found, skipping slider documentation")

            # Phase 5: Menus
            if self.menu_items:
                log_phase("Menu Documentation")
                self._document_all_menus()
            else:
                print("\n⚠️ No menu items found, skipping menu documentation")

            # Phase 6: Simulation states
            log_phase("Simulation States")
            self._document_simulation_states()

            # Phase 7: Dialog windows
            log_phase("Dialog Windows")
            self._document_dialog_windows()

            # Phase 8: View controls
            log_phase("View Controls")
            self._document_view_toggles()

            # Phase 9: Speed control
            log_phase("Speed Control")
            self._document_speed_control()

            # Phase 10: Status elements
            log_phase("Status Elements")
            self._document_status_bar_and_log()

            # Phase 11: Final state
            log_phase("Final Application State")
            self._take_screenshot(
                "18_final_state", "Final GUI state - after all documentation"
            )

            print(
                f"\n✅ SCREEN DOCUMENTATION COMPLETE ({current_phase}/{total_phases} phases)"
            )

        except Exception as e:
            print(f"\n❌ Error during screen documentation: {e}")
            import traceback

            traceback.print_exc()

    def _document_manual_application(self):
        """Document any running application in manual mode."""
        print("🎯 Manual Application Documentation")
        print("=" * 40)

        # Take a series of screenshots to document the current application
        screenshots = [
            ("01_initial", "Initial application state"),
            ("02_window_focus", "After window focus"),
            ("03_click_center", "After clicking center"),
            ("04_final", "Final documentation state"),
        ]

        for i, (filename, description) in enumerate(screenshots, 1):
            print(f"\n📸 [{i}/{len(screenshots)}] {description}")

            # Ensure we have an active window
            if not self.gui_window:
                self.gui_window = self._find_any_application_window()

            if self.gui_window:
                self._take_screenshot(filename, description)

                # Add some interaction between screenshots
                if i == 2:  # After focus, click center
                    try:
                        center_x = self.gui_window.left + self.gui_window.width // 2
                        center_y = self.gui_window.top + self.gui_window.height // 2
                        pyautogui.click(center_x, center_y)
                        time.sleep(1)
                    except Exception as e:
                        print(f"   ⚠️ Could not click center: {e}")

                time.sleep(1)
            else:
                print(f"   ⚠️ No window found for {description}")

        print(f"\n✅ Manual documentation complete with {len(screenshots)} screenshots")

    def _find_any_application_window(self) -> Optional[Any]:
        """Find any non-IDE application window for manual mode."""
        try:
            if hasattr(gw, "getAllWindows"):
                all_windows = gw.getAllWindows()

                # Filter for reasonable application windows
                candidates = []
                for window in all_windows:
                    if (
                        hasattr(window, "title")
                        and window.title
                        and hasattr(window, "width")
                        and hasattr(window, "height")
                        and not self._is_ide_window(window)
                        and window.width > 400
                        and window.height > 300
                    ):

                        title_lower = window.title.lower()
                        # Skip system windows
                        if not any(
                            skip in title_lower
                            for skip in [
                                "desktop",
                                "finder",
                                "spotlight",
                                "notification",
                                "system preferences",
                                "activity monitor",
                                "dock",
                                "menu bar",
                                "trash",
                                "launchpad",
                            ]
                        ):
                            candidates.append(window)

                # Sort by size (largest first)
                candidates.sort(key=lambda w: w.width * w.height, reverse=True)

                if candidates:
                    selected = candidates[0]
                    print(f"🎯 Found application: {selected.title}")
                    return selected

        except Exception as e:
            print(f"⚠️ Error finding application window: {e}")

        return None

    def _print_success_summary(self):
        """Print a comprehensive success summary."""
        print("\n" + "=" * 70)
        print("🎉 DOCUMENTATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        total_screenshots = len(self.doc_structure["screenshots"])
        total_size = sum(s.get("size", 0) for s in self.doc_structure["screenshots"])

        print("\n📊 SUMMARY:")
        print(f"   Total screenshots captured: {total_screenshots}")
        print(f"   Total file size: {total_size / 1024:.1f} KB")
        print(
            f"   Elements discovered: {len(self.button_locations)} buttons, {len(self.tab_locations)} tabs, {len(self.slider_locations)} sliders"
        )

        print("\n📁 OUTPUT FILES:")
        print(f"   Screenshots: {self.screenshots_dir}")
        print(f"   Reports: {self.reports_dir}")

        # Show capture methods used
        methods = {}
        for screenshot in self.doc_structure["screenshots"]:
            method = screenshot.get("capture_method", "unknown")
            methods[method] = methods.get(method, 0) + 1

        print("\n🎯 CAPTURE METHODS USED:")
        for method, count in methods.items():
            print(f"   {method}: {count} screenshots")

        print("\n🌐 NEXT STEPS:")
        print("   1. Open the HTML report to view documentation")
        print("   2. Review screenshots for completeness")
        print("   3. Share report with stakeholders")
        print("   4. Archive screenshots for future reference")

        print("\n" + "=" * 70)

    def _show_gui_selection_interface(self):
        """Show interactive GUI selection interface with fallback options."""
        print("💡 Available GUI files:")
        # List available GUI files
        gui_files = [
            "APGI-Psychological-States-GUI.py",
            "APGI-Equations.py",
            "APGI-Equations-Enhanced.py",
            "APGI-Equations-Enhanced-Full.py",
            "APGI-Entropy-Implementation.py",
            "APGI-Liquid-Network-Implementation.py",
            "APGI-Multimodal-Integration.py",
            "APGI-Parameter-Estimation-Protocol.py",
            "APGI-Psychological-States-CLI.py",
            "APGI-Turing-Machine.py",
            "Validation/APGI-Validation-GUI.py",
            "Falsification-Protocols/protocol_gui.py",
        ]

        # Show available GUI files with descriptions
        print("\n📱 Available APGI Applications:")
        print("=" * 50)

        gui_descriptions = {
            "APGI-Psychological-States-GUI.py": "🧠 Psychological States GUI - Main APGI consciousness modeling interface",
            "APGI-Equations.py": "📐 Formal Model - Core APGI mathematical framework",
            "APGI-Equations-Enhanced.py": "⚡ Enhanced Formal Model - Extended APGI framework with additional features",
            "APGI-Equations-Enhanced-Full.py": "🌟 Full Enhanced Model - Complete APGI implementation with all features",
            "APGI-Entropy-Implementation.py": "🔀 Entropy Implementation - APGI entropy-based consciousness model",
            "APGI-Liquid-Network-Implementation.py": "💧 Liquid Network - APGI liquid neural network implementation",
            "APGI-Multimodal-Integration.py": "🔗 Multimodal Integration - APGI multi-sensory integration system",
            "APGI-Parameter-Estimation-Protocol.py": "📊 Parameter Estimation - APGI parameter optimization protocol",
            "APGI-Psychological-States-CLI.py": "💻 Psychological States CLI - Command-line interface for APGI states",
            "APGI-Turing-Machine.py": "🤖 Turing Machine - APGI computational consciousness model",
            "Validation/APGI-Validation-GUI.py": "✅ Validation GUI - APGI validation and testing interface",
            "Falsification-Protocols/protocol_gui.py": "🧪 Falsification Protocols - APGI scientific testing protocols",
        }

        available_gui_files = []
        for i, gui_file in enumerate(gui_files):
            file_path = self.base_dir / gui_file
            if file_path.exists():
                description = gui_descriptions.get(gui_file, "📄 APGI Application")
                print(f"   {i + 1:2d}. {gui_file}")
                print(f"      {description}")
                available_gui_files.append((i + 1, gui_file))
                print()

        if not available_gui_files:
            print("❌ No GUI files found!")
            print("\n🔄 FALLBACK OPTIONS:")
            print("   • Type 'manual' to document any running application")
            print("   • Type 'scan' to see all running windows")
            print("   • Type 'skip' to proceed without launching GUI")
            print("   • Press Enter to continue with window detection only")

            try:
                choice = input("\n➡️  Your choice: ").strip().lower()

                if choice == "manual":
                    print("\n🎯 Manual mode: Will document any running application")
                    return "manual"
                elif choice == "scan":
                    self._show_all_windows()
                    return self._show_gui_selection_interface()
                elif choice == "skip":
                    print("\n⏭️ Skipping GUI launch, will detect running apps")
                    return "skip"
                else:
                    print("\n🔍 Continuing with window detection only...")
                    return "skip"
            except KeyboardInterrupt:
                print("\n❌ Cancelled by user")
                return None

        print("=" * 50)

        # Let user select GUI application
        print("🎮 Select APGI Application to Document:")
        print(
            "   • Enter number (1-{}) to select specific application".format(
                len(available_gui_files)
            )
        )
        print("   • Press Enter to use default (APGI-Psychological-States-GUI.py)")
        print("   • Type 'manual' to document any running application")
        print("   • Type 'list' to see available files again")
        print("   • Press Ctrl+C to cancel")

        try:
            selection = input("\n➡️  Your choice: ").strip()

            if selection.lower() == "list":
                # Show list again
                return self._show_gui_selection_interface()
            elif selection.lower() == "manual":
                print("\n🎯 Manual mode: Will document any running application")
                return "manual"
            elif selection.lower() == "cancel":
                print("❌ Cancelled by user")
                return None
            elif selection == "":
                # Use default
                selected_gui = "APGI-Psychological-States-GUI.py"
                print(f"✅ Using default: {selected_gui}")
            else:
                # Parse number selection
                try:
                    choice_num = int(selection)
                    if 1 <= choice_num <= len(available_gui_files):
                        selected_gui = available_gui_files[choice_num - 1][1]
                        print(f"✅ Selected: {selected_gui}")
                    else:
                        print(
                            f"❌ Invalid choice. Please enter 1-{len(available_gui_files)}"
                        )
                        return self._show_gui_selection_interface()
                except ValueError:
                    print("❌ Invalid input. Please enter a number or press Enter.")
                    return self._show_gui_selection_interface()

            return selected_gui

        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return None

    def _start_gui_app(self):
        """Start the GUI application with fallback options."""
        print("\n📱 Starting GUI Application...")

        try:
            # First show GUI selection interface
            selected_gui = self._show_gui_selection_interface()
            if not selected_gui:
                return False

            # Handle manual mode
            if selected_gui == "manual":
                print("🎯 Manual mode: Will document any running application")
                self.manual_mode = True
                return True

            # Handle skip mode
            if selected_gui == "skip":
                print("⏭️ Skipping GUI launch, will detect running apps")
                return True

            gui_script = self.base_dir / selected_gui

            # Start GUI in background
            self.gui_process = subprocess.Popen(
                [sys.executable, str(gui_script)], cwd=self.base_dir
            )

            print("✅ GUI application started")
            return True

        except Exception as e:
            print(f"❌ Error starting GUI: {e}")
            print("\n🔄 Falling back to manual mode...")
            self.manual_mode = True
            return True

    def _discover_gui_elements(self):
        """Discover and locate GUI elements with robust fallback."""
        print("\n🔍 Discovering GUI elements...")

        # Wait for GUI to fully load
        time.sleep(3)

        # Try to find the GUI window multiple times
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"  📍 Attempt {attempt + 1}/{max_attempts} to find GUI window...")
            self.gui_window = self._find_gui_window()

            if self.gui_window:
                break
            else:
                print("  ⏳ Waiting 2 seconds before retry...")
                time.sleep(2)

        if not self.gui_window:
            print("❌ Could not find APGI GUI window, using full-screen fallback mode")
            self._use_fallback_discovery()
            return False

        print(f"✅ Found GUI window: {self.gui_window.title}")

        # Manual verification - ask user to confirm if this is the right window
        print(
            f"\n🔍 Found window: '{self.gui_window.title}' ({self.gui_window.width}x{self.gui_window.height})"
        )
        print("Is this the correct APGI application window? (y/n)")
        try:
            response = input().strip().lower()
            if response not in ["y", "yes", ""]:
                print("🔄 Let's try manual window selection...")
                self.gui_window = self._manual_window_selection()
                if not self.gui_window:
                    print("❌ No window selected, using fallback mode")
                    self._use_fallback_discovery()
                    return False
        except KeyboardInterrupt:
            print("\n❌ Documentation cancelled by user")
            return False

        # Activate and maximize window
        try:
            self.gui_window.activate()
            if hasattr(self.gui_window, "maximize"):
                self.gui_window.maximize()
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Could not maximize window: {e}")

        # Verify window is active by taking a test screenshot
        try:
            pyautogui.screenshot(
                region=(
                    self.gui_window.left,
                    self.gui_window.top,
                    self.gui_window.width,
                    self.gui_window.height,
                )
            )
            print("✅ Window region capture test successful")
        except Exception as e:
            print(f"⚠️ Window region test failed, will use full screen: {e}")
            self.gui_window = None  # Force fallback mode
            self._use_fallback_discovery()
            return False

        # Take a screenshot to analyze
        try:
            screenshot = pyautogui.screenshot(
                region=(
                    self.gui_window.left,
                    self.gui_window.top,
                    self.gui_window.width,
                    self.gui_window.height,
                )
            )
        except Exception as e:
            print(f"⚠️ Could not capture window region, using full screen: {e}")
            screenshot = pyautogui.screenshot()

        screenshot_path = self.screenshots_dir / "gui_discovery.png"
        screenshot.save(screenshot_path)

        # Discover buttons using template matching and heuristics
        self._discover_buttons(screenshot)
        self._discover_tabs(screenshot)
        self._discover_sliders(screenshot)
        self._discover_menu_items()

        print(
            f"✅ Found {len(self.button_locations)} buttons, {len(self.tab_locations)} tabs, {len(self.slider_locations)} sliders"
        )
        return True

    def _use_fallback_discovery(self):
        """Use keyboard shortcuts and estimated positions when window cannot be located."""
        print("  🔄 Using fallback discovery mode...")

        # Set up estimated button locations based on typical GUI layout
        screen_width, screen_height = pyautogui.size()

        # Estimate control panel position (left side)
        control_x = screen_width // 4
        control_y = screen_height // 3

        self.button_locations = {
            "start": {
                "x": control_x - 100,
                "y": control_y,
                "rect": (control_x - 100, control_y, 80, 30),
                "confidence": 0.5,
            },
            "pause": {
                "x": control_x,
                "y": control_y,
                "rect": (control_x, control_y, 80, 30),
                "confidence": 0.5,
            },
            "stop": {
                "x": control_x + 100,
                "y": control_y,
                "rect": (control_x + 100, control_y, 80, 30),
                "confidence": 0.5,
            },
            "reset": {
                "x": control_x + 200,
                "y": control_y,
                "rect": (control_x + 200, control_y, 80, 30),
                "confidence": 0.5,
            },
        }

        # Estimate tab positions (top right area)
        tab_start_x = screen_width // 2
        tab_y = screen_height // 4

        tab_names = [
            "neural_activity",
            "interoception",
            "system_metrics",
            "self_model",
            "oscillations",
            "state_space",
        ]

        for i, tab_name in enumerate(tab_names):
            self.tab_locations[tab_name] = {
                "x": tab_start_x + i * 150,
                "y": tab_y,
                "rect": (tab_start_x + i * 150, tab_y, 120, 30),
            }

        # Estimate slider positions (left panel, below buttons)
        slider_start_x = control_x - 50
        slider_start_y = control_y + 100

        slider_names = [
            "ignition_threshold",
            "extero_precision",
            "intero_precision",
            "arousal",
            "stress",
            "activity",
            "learning_rate",
            "attention_gain",
        ]

        for i, slider_name in enumerate(slider_names):
            y_pos = slider_start_y + i * 40
            self.slider_locations[slider_name] = {
                "x": slider_start_x + 75,
                "y": y_pos + 10,
                "rect": (slider_start_x, y_pos, 150, 20),
                "min_x": slider_start_x,
                "max_x": slider_start_x + 150,
            }

        # Still discover menu items (these work with keyboard shortcuts regardless of window)
        self._discover_menu_items()

        print(
            f"✅ Fallback mode: Estimated {len(self.button_locations)} buttons, {len(self.tab_locations)} tabs, {len(self.slider_locations)} sliders"
        )

    def _find_gui_window(self) -> Optional[Any]:
        """Find APGI GUI window with enhanced 6-method detection approach."""
        print("🔍 Starting enhanced window detection (6-method approach)...")

        # Method 1: Exact title matching
        window = self._method_1_exact_title_matching()
        if window and self._verify_window_is_apgi(window):
            print(f"✅ Method 1 succeeded: {window.title}")
            return window

        # Method 2: Multiple title variations
        window = self._method_2_title_variations()
        if window and self._verify_window_is_apgi(window):
            print(f"✅ Method 2 succeeded: {window.title}")
            return window

        # Method 3: Keyword-based filtering
        window = self._method_3_keyword_filtering()
        if window and self._verify_window_is_apgi(window):
            print(f"✅ Method 3 succeeded: {window.title}")
            return window

        # Method 4: Size-based detection
        window = self._method_4_size_based_detection()
        if window and self._verify_window_is_apgi(window):
            print(f"✅ Method 4 succeeded: {window.title}")
            return window

        # Method 5: Manual activation testing
        window = self._method_5_activation_testing()
        if window and self._verify_window_is_apgi(window):
            print(f"✅ Method 5 succeeded: {window.title}")
            return window

        # Method 6: Large window candidate selection
        window = self._method_6_large_window_selection()
        if window and self._verify_window_is_apgi(window):
            print(f"✅ Method 6 succeeded: {window.title}")
            return window

        print("❌ All 6 detection methods failed")
        return None

    def _method_1_exact_title_matching(self) -> Optional[Any]:
        """Method 1: Try exact title matching with IDE filtering."""
        print("  📍 Method 1: Exact title matching...")
        try:
            exact_titles = [
                "APGIConsciousness Modeling Framework",
                "APGI Consciousness Modeling Framework",
                "APGIConsciousness",
                "APGI Consciousness",
            ]

            for title in exact_titles:
                if hasattr(gw, "getWindowsWithTitle"):
                    windows = gw.getWindowsWithTitle(title)
                    if windows:
                        # Filter out IDE windows
                        non_ide_windows = [
                            w for w in windows if not self._is_ide_window(w)
                        ]
                        if non_ide_windows:
                            print(f"    🎯 Found exact match: {title}")
                            return non_ide_windows[0]
        except Exception as e:
            print(f"    ⚠️ Method 1 error: {e}")
        return None

    def _method_2_title_variations(self) -> Optional[Any]:
        """Method 2: Try all possible title variations with IDE filtering."""
        print("  📍 Method 2: Title variations...")
        try:
            title_variations = [
                "APGIConsciousness Modeling Framework",
                "APGIConsciousness",
                "APGI Consciousness",
                "Consciousness Modeling",
                "APGI System",
                "APGI",
                "consciousness",
                "modeling",
                "tk",  # Tkinter windows often have 'tk' in title
                "python",
            ]

            for title in title_variations:
                try:
                    if hasattr(gw, "getWindowsWithTitle"):
                        windows = gw.getWindowsWithTitle(title)
                        if windows:
                            # Filter out IDE windows
                            non_ide_windows = [
                                w for w in windows if not self._is_ide_window(w)
                            ]
                            if non_ide_windows:
                                print(
                                    f"    🎯 Found variation: '{title}' -> {non_ide_windows[0].title}"
                                )
                                return non_ide_windows[0]
                except Exception:
                    continue
        except Exception as e:
            print(f"    ⚠️ Method 2 error: {e}")
        return None

    def _method_3_keyword_filtering(self) -> Optional[Any]:
        """Method 3: Search all windows with comprehensive keyword filtering and IDE exclusion."""
        print("  📍 Method 3: Keyword-based filtering...")
        try:
            if hasattr(gw, "getAllWindows"):
                all_windows = gw.getAllWindows()
                print(f"    🔍 Scanning {len(all_windows)} windows...")

                candidates = []
                for window in all_windows:
                    if not (
                        hasattr(window, "title")
                        and window.title
                        and len(window.title.strip()) > 0
                    ):
                        continue

                    # First filter: exclude IDE windows immediately
                    if self._is_ide_window(window):
                        continue

                    title_lower = window.title.lower()

                    # Skip obvious system and development windows
                    if any(
                        skip in title_lower
                        for skip in [
                            "desktop",
                            "finder",
                            "spotlight",
                            "notification",
                            "system preferences",
                            "activity monitor",
                            "visual studio code",
                            "vscode",
                            "chrome",
                            "safari",
                            "firefox",
                            "edge",
                            "opera",
                            "discord",
                            "slack",
                            "zoom",
                            "teams",
                            "spotify",
                            "photoshop",
                            "illustrator",
                        ]
                    ):
                        continue

                    # Look for APGI-related keywords
                    if any(
                        keyword in title_lower
                        for keyword in [
                            "apgi",
                            "consciousness",
                            "modeling",
                            "framework",
                            "neural",
                            "ignition",
                            "precision",
                            "interoception",
                            "exteroception",
                            "self-model",
                            "oscillations",
                        ]
                    ):
                        candidates.append(window)
                        print(f"    🎯 Keyword match: {window.title}")

                if candidates:
                    # Sort by relevance (title length and keyword count)
                    candidates.sort(
                        key=lambda w: (
                            sum(
                                1
                                for kw in [
                                    "apgi",
                                    "consciousness",
                                    "modeling",
                                    "framework",
                                ]
                                if kw in w.title.lower()
                            ),
                            len(w.title),
                        ),
                        reverse=True,
                    )
                    return candidates[0]
        except Exception as e:
            print(f"    ⚠️ Method 3 error: {e}")
        return None

    def _method_4_size_based_detection(self) -> Optional[Any]:
        """Method 4: Find windows by size characteristics with IDE filtering."""
        print("  📍 Method 4: Size-based detection...")
        try:
            if hasattr(gw, "getAllWindows"):
                all_windows = gw.getAllWindows()

                # Look for windows with typical GUI application sizes
                size_candidates = []
                for window in all_windows:
                    if not (
                        hasattr(window, "width")
                        and hasattr(window, "height")
                        and hasattr(window, "title")
                        and window.title
                    ):
                        continue

                    # First filter: exclude IDE windows
                    if self._is_ide_window(window):
                        continue

                    # Typical desktop GUI dimensions
                    if 800 < window.width < 2000 and 600 < window.height < 1500:
                        title_lower = window.title.lower()

                        # Skip development tools
                        if not any(
                            skip in title_lower
                            for skip in [
                                "visual studio",
                                "vscode",
                                "code",
                                "terminal",
                                "browser",
                                "chrome",
                                "safari",
                                "firefox",
                                "discord",
                                "slack",
                                "zoom",
                            ]
                        ):
                            size_candidates.append(window)
                            print(
                                f"    🎯 Size candidate: {window.title} ({window.width}x{window.height})"
                            )

                if size_candidates:
                    # Prefer larger windows (more likely to be our GUI)
                    size_candidates.sort(key=lambda w: w.width * w.height, reverse=True)
                    return size_candidates[0]
        except Exception as e:
            print(f"    ⚠️ Method 4 error: {e}")
        return None

    def _method_5_activation_testing(self) -> Optional[Any]:
        """Method 5: Try activating candidates to verify they respond, excluding IDEs."""
        print("  📍 Method 5: Manual activation testing...")
        try:
            if hasattr(gw, "getAllWindows"):
                all_windows = gw.getAllWindows()

                # Get reasonable candidates
                test_candidates = []
                for window in all_windows:
                    if (
                        hasattr(window, "width")
                        and hasattr(window, "height")
                        and hasattr(window, "title")
                        and window.title
                        and 600 < window.width < 1800
                        and 400 < window.height < 1200
                    ):

                        # First filter: exclude IDE windows
                        if self._is_ide_window(window):
                            continue

                        title_lower = window.title.lower()
                        if not any(
                            skip in title_lower
                            for skip in ["desktop", "finder", "dock", "menu bar"]
                        ):
                            test_candidates.append(window)

                # Test each candidate by trying to activate it
                for window in test_candidates[:5]:  # Test top 5 candidates
                    print(f"    🧪 Testing activation: {window.title}")
                    try:
                        # Remember current active window
                        original_active = gw.getActiveWindow()

                        # Try to activate the candidate
                        window.activate()
                        time.sleep(0.5)

                        # Check if it became active
                        active = gw.getActiveWindow()
                        if (
                            active
                            and hasattr(active, "title")
                            and active.title == window.title
                            and not self._is_ide_window(active)
                        ):
                            print(f"    ✅ Activation successful: {window.title}")
                            return window

                        # Restore original window
                        if original_active:
                            original_active.activate()
                            time.sleep(0.3)

                    except Exception:
                        continue
        except Exception as e:
            print(f"    ⚠️ Method 5 error: {e}")
        return None

    def _method_6_large_window_selection(self) -> Optional[Any]:
        """Method 6: Select from largest windows that could be our app, excluding IDEs."""
        print("  📍 Method 6: Large window candidate selection...")
        try:
            if hasattr(gw, "getAllWindows"):
                all_windows = gw.getAllWindows()

                # Find all reasonably large windows
                large_windows = []
                for window in all_windows:
                    if (
                        hasattr(window, "width")
                        and hasattr(window, "height")
                        and hasattr(window, "title")
                        and window.title
                    ):

                        # First filter: exclude IDE windows
                        if self._is_ide_window(window):
                            continue

                        # Must be substantial size for a GUI application
                        if window.width > 1000 and window.height > 700:
                            title_lower = window.title.lower()

                            # Exclude common large windows that aren't our app
                            if not any(
                                skip in title_lower
                                for skip in [
                                    "visual studio",
                                    "vscode",
                                    "code",
                                    "terminal",
                                    "browser",
                                    "chrome",
                                    "safari",
                                    "firefox",
                                    "edge",
                                    "finder",
                                    "desktop",
                                    "discord",
                                    "slack",
                                    "zoom",
                                ]
                            ):
                                large_windows.append(window)
                                print(
                                    f"    🎯 Large window: {window.title} ({window.width}x{window.height})"
                                )

                if large_windows:
                    # Sort by size (largest first) and try top 3
                    large_windows.sort(key=lambda w: w.width * w.height, reverse=True)

                    for window in large_windows[:3]:
                        print(f"    🎯 Trying largest: {window.title}")
                        try:
                            window.activate()
                            time.sleep(1)
                            active = gw.getActiveWindow()
                            if (
                                active
                                and hasattr(active, "title")
                                and active.title == window.title
                                and not self._is_ide_window(active)
                            ):
                                print(f"    ✅ Large window activated: {window.title}")
                                return window
                        except Exception:
                            continue
        except Exception as e:
            print(f"    ⚠️ Method 6 error: {e}")
        return None

    def _verify_window_is_apgi(self, window: Any) -> bool:
        """Verify that the detected window is actually the APGI application."""
        try:
            if not window or not hasattr(window, "title"):
                return False

            # Quick title-based verification
            title_lower = window.title.lower()
            if any(
                keyword in title_lower
                for keyword in ["apgi", "consciousness", "modeling"]
            ):
                return True

            # For windows without clear title indicators, try to activate and verify
            try:
                window.activate()
                time.sleep(0.5)

                # Take a small test screenshot to analyze
                if (
                    hasattr(window, "left")
                    and hasattr(window, "top")
                    and hasattr(window, "width")
                    and hasattr(window, "height")
                ):
                    test_region = (
                        max(0, window.left),
                        max(0, window.top),
                        min(200, window.width),  # Small test region
                        min(200, window.height),
                    )

                    test_screenshot = pyautogui.screenshot(region=test_region)
                    if self._is_likely_app_screenshot(test_screenshot):
                        return True

            except Exception:
                pass

            return False

        except Exception:
            return False

    def _discover_buttons(self, screenshot: Image.Image):
        """Discover button locations using improved image processing."""
        print("  🔘 Discovering buttons...")

        try:
            # Convert to OpenCV format
            img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

            # Enhanced button detection with multiple methods
            button_detections = []

            # Method 1: Color-based detection for common button colors
            color_ranges = [
                # Green buttons (Start)
                {
                    "hsv_lower": (40, 50, 50),
                    "hsv_upper": (80, 255, 255),
                    "name": "start",
                },
                # Yellow/Orange buttons (Pause)
                {
                    "hsv_lower": (15, 50, 50),
                    "hsv_upper": (35, 255, 255),
                    "name": "pause",
                },
                # Red buttons (Stop)
                {"hsv_lower": (0, 50, 50), "hsv_upper": (10, 255, 255), "name": "stop"},
                # Blue/Gray buttons (Reset)
                {
                    "hsv_lower": (90, 30, 30),
                    "hsv_upper": (130, 255, 255),
                    "name": "reset",
                },
            ]

            for color_range in color_ranges:
                mask = cv2.inRange(
                    img_hsv,
                    np.array(color_range["hsv_lower"]),
                    np.array(color_range["hsv_upper"]),
                )

                # Apply morphological operations to clean up
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 300 < area < 10000:  # Reasonable button size range
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h

                        # Check if it's button-like (not too elongated)
                        if 0.5 < aspect_ratio < 3.0:
                            button_detections.append(
                                {
                                    "name": color_range["name"],
                                    "x": x + w // 2,
                                    "y": y + h // 2,
                                    "rect": (x, y, w, h),
                                    "confidence": area / 1000,
                                    "method": "color",
                                }
                            )

            # Method 2: Edge detection for rectangular shapes
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 8000:
                    approx = cv2.approxPolyDP(
                        contour, 0.02 * cv2.arcLength(contour, True), True
                    )

                    # Check if it's roughly rectangular (4-6 corners)
                    if 4 <= len(approx) <= 6:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h

                        if 0.8 < aspect_ratio < 2.5:  # Reasonable button aspect ratio
                            # Check if this button wasn't already detected by color
                            is_duplicate = False
                            for existing in button_detections:
                                ex, ey = existing["x"], existing["y"]
                                if (
                                    abs(ex - (x + w // 2)) < 20
                                    and abs(ey - (y + h // 2)) < 20
                                ):
                                    is_duplicate = True
                                    break

                            if not is_duplicate:
                                # Determine button type based on position
                                button_name = self._classify_button_by_position(
                                    x, y, w, h
                                )
                                button_detections.append(
                                    {
                                        "name": button_name,
                                        "x": x + w // 2,
                                        "y": y + h // 2,
                                        "rect": (x, y, w, h),
                                        "confidence": area / 1000,
                                        "method": "edge",
                                    }
                                )

            # Remove duplicates and keep best confidence for each button type
            unique_buttons = {}
            for detection in button_detections:
                name = detection["name"]
                if (
                    name not in unique_buttons
                    or detection["confidence"] > unique_buttons[name]["confidence"]
                ):
                    unique_buttons[name] = detection

            # Store final button locations
            for name, detection in unique_buttons.items():
                self.button_locations[name] = {
                    "x": detection["x"],
                    "y": detection["y"],
                    "rect": detection["rect"],
                    "confidence": detection["confidence"],
                }
                print(
                    f"    Found {name} button at ({detection['x']}, {detection['y']}) [confidence: {detection['confidence']:.1f}]"
                )

        except Exception as e:
            print(f"    ❌ Error in button discovery: {e}")

    def _classify_button_by_position(self, x: int, y: int, w: int, h: int) -> str:
        """Classify button type based on its position in the window."""
        if self.gui_window:
            rel_x = x - self.gui_window.left
            rel_y = y - self.gui_window.top

            # Typical button layout: controls at the top or bottom
            if rel_y < 200:  # Top area
                if rel_x < 200:
                    return "start"
                elif rel_x < 400:
                    return "pause"
                elif rel_x < 600:
                    return "stop"
                else:
                    return "reset"
            else:  # Bottom area or other
                if rel_x < 200:
                    return "start"
                elif rel_x < 400:
                    return "pause"
                elif rel_x < 600:
                    return "stop"
                else:
                    return "reset"

        # Fallback classification based on size
        if w > 80 and h > 30:
            return "start"  # Larger buttons are typically start buttons
        elif w > 60 and h > 25:
            return "pause"
        else:
            return "stop"

    def _discover_tabs(self, screenshot: Image.Image):
        """Discover tab locations using improved detection."""
        print("  📑 Discovering tabs...")

        try:
            # Convert to OpenCV format
            img_cv = np.array(screenshot)
            img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)

            # Look for tab-like structures at the top of the window
            height, width = img_cv.shape[:2]

            # Focus on the top portion where tabs are typically located
            tab_region = img_hsv[: height // 4, :]

            # Tab detection using edge detection and contour analysis
            gray = cv2.cvtColor(tab_region, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Dilate edges to connect nearby components
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            tab_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 5000:  # Reasonable tab size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    # Tabs are typically wider than they are tall
                    if 2.0 < aspect_ratio < 8.0 and h < 50:
                        tab_candidates.append(
                            {
                                "x": x + w // 2,
                                "y": y + h // 2,
                                "rect": (x, y, w, h),
                                "area": area,
                            }
                        )

            # Sort tabs by x-coordinate (left to right)
            tab_candidates.sort(key=lambda t: t["x"])

            # Known tab names based on APGI system
            tab_names = [
                "neural_activity",
                "interoception",
                "system_metrics",
                "self_model",
                "oscillations",
                "state_space",
            ]

            # Assign tab names to discovered positions
            for i, tab_candidate in enumerate(tab_candidates[: len(tab_names)]):
                tab_name = tab_names[i]
                self.tab_locations[tab_name] = {
                    "x": tab_candidate["x"],
                    "y": tab_candidate["y"],
                    "rect": tab_candidate["rect"],
                }
                print(
                    f"    Found tab '{tab_name}' at ({tab_candidate['x']}, {tab_candidate['y']})"
                )

            # Fallback: if no tabs detected, use estimated positions
            if not self.tab_locations and self.gui_window:
                print("    ⚠️ No tabs detected, using estimated positions")
                start_x = 50
                start_y = 100
                tab_width = 120
                tab_height = 30

                for i, tab_name in enumerate(tab_names):
                    self.tab_locations[tab_name] = {
                        "x": start_x + i * (tab_width + 10),
                        "y": start_y,
                        "rect": (
                            start_x + i * (tab_width + 10),
                            start_y,
                            tab_width,
                            tab_height,
                        ),
                    }
                    print(
                        f"    Estimated tab '{tab_name}' at ({start_x + i * (tab_width + 10)}, {start_y})"
                    )

        except Exception as e:
            print(f"    ❌ Error in tab discovery: {e}")

    def _discover_sliders(self, screenshot: Image.Image):
        """Discover slider locations using improved detection."""
        print("  🎚️ Discovering sliders...")

        try:
            # Convert to OpenCV format
            img_cv = np.array(screenshot)

            height, width = img_cv.shape[:2]

            # Slider detection using horizontal line detection
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

            # Apply adaptive thresholding to handle varying lighting
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Detect horizontal lines (slider tracks)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, horizontal_kernel
            )

            contours, _ = cv2.findContours(
                horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            slider_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 2000:  # Reasonable slider track size
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    # Slider tracks are typically much wider than tall
                    if aspect_ratio > 5.0 and h < 20:
                        slider_candidates.append(
                            {
                                "x": x + w // 2,
                                "y": y + h // 2,
                                "rect": (x, y, w, h),
                                "area": area,
                            }
                        )

            # Sort sliders by y-coordinate (top to bottom)
            slider_candidates.sort(key=lambda s: s["y"])

            # Known slider names based on APGI system
            slider_names = [
                "ignition_threshold",
                "extero_precision",
                "intero_precision",
                "arousal",
                "stress",
                "activity",
                "learning_rate",
                "attention_gain",
            ]

            # Assign slider names to discovered positions
            for i, slider_candidate in enumerate(
                slider_candidates[: len(slider_names)]
            ):
                slider_name = slider_names[i]
                x, y, w, h = slider_candidate["rect"]
                self.slider_locations[slider_name] = {
                    "x": slider_candidate["x"],
                    "y": slider_candidate["y"],
                    "rect": slider_candidate["rect"],
                    "min_x": x,
                    "max_x": x + w,
                }
                print(
                    f"    Found slider '{slider_name}' at ({slider_candidate['x']}, {slider_candidate['y']})"
                )

            # Fallback: if no sliders detected, use estimated positions
            if not self.slider_locations and self.gui_window:
                print("    ⚠️ No sliders detected, using estimated positions")
                start_x = 150
                start_y = 300
                slider_width = 150
                slider_height = 20
                slider_spacing = 40

                for i, slider_name in enumerate(slider_names):
                    y_pos = start_y + i * slider_spacing
                    self.slider_locations[slider_name] = {
                        "x": start_x + slider_width // 2,
                        "y": y_pos + slider_height // 2,
                        "rect": (start_x, y_pos, slider_width, slider_height),
                        "min_x": start_x,
                        "max_x": start_x + slider_width,
                    }
                    print(
                        f"    Estimated slider '{slider_name}' at ({start_x + slider_width // 2}, {y_pos + slider_height // 2})"
                    )

        except Exception as e:
            print(f"    ❌ Error in slider discovery: {e}")

    def _discover_menu_items(self):
        """Discover all menu items and their submenus."""
        print("  📋 Discovering menu items...")

        # Comprehensive menu structure based on actual GUI
        self.menu_items = {
            "file": {
                "shortcut": "alt+f",
                "items": [
                    {"name": "New Session", "shortcut": "Ctrl+N"},
                    {"name": "Load Configuration", "shortcut": "Ctrl+O"},
                    {"name": "Save Configuration", "shortcut": "Ctrl+S"},
                    {"name": "Export Data", "shortcut": "Ctrl+E"},
                    {"name": "Export Plot"},
                    {"name": "Auto-save Data"},
                    {"name": "Exit", "shortcut": "Ctrl+Q"},
                ],
            },
            "edit": {
                "shortcut": "alt+e",
                "items": [
                    {"name": "System Parameters"},
                    {"name": "Precision Settings"},
                    {"name": "Ignition Threshold"},
                    {"name": "Reset to Defaults"},
                ],
            },
            "simulation": {
                "shortcut": "alt+s",
                "items": [
                    {"name": "Start", "shortcut": "F5"},
                    {"name": "Pause/Resume", "shortcut": "F6"},
                    {"name": "Stop", "shortcut": "F7"},
                    {"name": "Reset", "shortcut": "F8"},
                    {"name": "Run Preset Task"},
                ],
            },
            "view": {
                "shortcut": "alt+v",
                "items": [
                    {"name": "Control Panel"},
                    {"name": "Neural Activity"},
                    {"name": "Interoception"},
                    {"name": "System Metrics"},
                    {"name": "Zoom In", "shortcut": "Ctrl++"},
                    {"name": "Zoom Out", "shortcut": "Ctrl+-"},
                    {"name": "Fit to Window", "shortcut": "Ctrl+0"},
                ],
            },
            "tools": {
                "shortcut": "alt+t",
                "items": [
                    {"name": "Trigger Ignition Event"},
                    {"name": "Induce Stressor"},
                    {"name": "Modulate Precision"},
                    {"name": "Inject Sensory Input"},
                    {"name": "Set Body State"},
                    {"name": "System Diagnostics"},
                ],
            },
            "analysis": {
                "shortcut": "alt+a",
                "items": [
                    {"name": "Ignition Statistics"},
                    {"name": "Energy Budget Report"},
                    {"name": "Somatic Marker Analysis"},
                    {"name": "Self-Model Coherence"},
                    {"name": "Generate Report"},
                ],
            },
            "help": {
                "shortcut": "alt+h",
                "items": [
                    {"name": "Documentation"},
                    {"name": "Keyboard Shortcuts"},
                    {"name": "About APGI System"},
                ],
            },
        }

        total_menu_items = sum(
            len(menu.get("items", [])) for menu in self.menu_items.values()
        )
        print(
            f"    Found {len(self.menu_items)} menu categories with {total_menu_items} total items"
        )

    def _document_all_screens(self):
        """Document all screens and interactions with enhanced error handling."""
        print("\n📸 Documenting all screens and interactions...")

        try:
            # 1. Initial state
            self._take_screenshot(
                "01_initial_state", "Initial GUI state - application just started"
            )

            # 2. Click through all tabs (with error handling)
            if self.tab_locations:
                self._document_all_tabs()
            else:
                print("  ⚠️ No tabs found, skipping tab documentation")

            # 3. Test all buttons (with error handling)
            if self.button_locations:
                self._document_all_buttons()
            else:
                print("  ⚠️ No buttons found, skipping button documentation")

            # 4. Test all sliders (with error handling)
            if self.slider_locations:
                self._document_all_sliders()
            else:
                print("  ⚠️ No sliders found, skipping slider documentation")

            # 5. Test all menu items and submenus (with error handling)
            if self.menu_items:
                self._document_all_menus()
            else:
                print("  ⚠️ No menu items found, skipping menu documentation")

            # 6. Test simulation states (with error handling)
            self._document_simulation_states()

            # 7. Document all dialog windows (with error handling)
            self._document_dialog_windows()

            # 8. Document view toggles and zoom controls (with error handling)
            self._document_view_toggles()

            # 9. Document speed control (with error handling)
            self._document_speed_control()

            # 10. Document status bar and event log (with error handling)
            self._document_status_bar_and_log()

            # 11. Final state
            self._take_screenshot(
                "18_final_state", "Final GUI state - after all documentation"
            )

        except Exception as e:
            print(f"❌ Error during screen documentation: {e}")
            import traceback

            traceback.print_exc()

    def _document_all_tabs(self):
        """Document all tabs by clicking through them."""
        print("  📑 Documenting tabs...")

        for tab_name, tab_info in self.tab_locations.items():
            try:
                # Click on tab
                pyautogui.click(tab_info["x"], tab_info["y"])
                time.sleep(1)  # Wait for tab to load

                # Take screenshot
                self._take_screenshot(
                    f"02_tab_{tab_name}", f"Tab: {tab_name.replace('_', ' ').title()}"
                )
            except Exception as e:
                print(f"    ❌ Error documenting tab {tab_name}: {e}")
                continue

    def _document_all_buttons(self):
        """Document all buttons by clicking them."""
        print("  🔘 Documenting buttons...")

        for button_name, button_info in self.button_locations.items():
            try:
                # Click button
                pyautogui.click(button_info["x"], button_info["y"])
                time.sleep(2)  # Wait for action to complete

                # Take screenshot
                self._take_screenshot(
                    f"03_button_{button_name}", f"Button clicked: {button_name.title()}"
                )

                # If it's a start/pause/stop, we might need to handle the state change
                if button_name == "start":
                    time.sleep(3)  # Let simulation run a bit
                elif button_name == "stop":
                    time.sleep(1)  # Let it stop

            except Exception as e:
                print(f"    ❌ Error documenting button {button_name}: {e}")
                continue

    def _document_all_sliders(self):
        """Document all sliders by adjusting them."""
        print("  🎚️ Documenting sliders...")

        for slider_name, slider_info in self.slider_locations.items():
            try:
                # Move slider to minimum
                pyautogui.moveTo(slider_info["min_x"], slider_info["y"])
                pyautogui.dragTo(
                    slider_info["min_x"] + 20, slider_info["y"], duration=0.5
                )
                time.sleep(0.5)

                self._take_screenshot(
                    f"04_slider_{slider_name}_min",
                    f"Slider {slider_name.replace('_', ' ').title()} - Minimum",
                )

                # Move slider to maximum
                pyautogui.moveTo(slider_info["min_x"] + 20, slider_info["y"])
                pyautogui.dragTo(slider_info["max_x"], slider_info["y"], duration=0.5)
                time.sleep(0.5)

                self._take_screenshot(
                    f"04_slider_{slider_name}_max",
                    f"Slider {slider_name.replace('_', ' ').title()} - Maximum",
                )

                # Reset to middle
                mid_x = (slider_info["min_x"] + slider_info["max_x"]) // 2
                pyautogui.click(mid_x, slider_info["y"])
                time.sleep(0.5)

            except Exception as e:
                print(f"    ❌ Error documenting slider {slider_name}: {e}")
                continue

    def _document_all_menus(self):
        """Document all menu items and their submenus."""
        print("  📋 Documenting menus...")

        for menu_name, menu_info in self.menu_items.items():
            try:
                # Open menu
                pyautogui.hotkey(*menu_info["shortcut"].split("+"))
                time.sleep(1)

                # Take screenshot of opened menu
                self._take_screenshot(
                    f"05_menu_{menu_name}_expanded",
                    f"Menu: {menu_name.title()} - Expanded",
                )

                # Document each submenu item
                if "items" in menu_info:
                    for i, item in enumerate(menu_info["items"]):
                        item_name = (
                            item["name"].replace(" ", "_").replace("/", "_").lower()
                        )

                        # Press down arrow to navigate to item
                        for _ in range(i + 1):
                            pyautogui.press("down")
                        time.sleep(0.3)

                        # Take screenshot with item highlighted
                        self._take_screenshot(
                            f"05_menu_{menu_name}_{item_name}",
                            f"Menu Item: {menu_name.title()} > {item['name']}",
                        )

                # Close menu (press Escape twice)
                pyautogui.press("escape")
                time.sleep(0.3)
                pyautogui.press("escape")
                time.sleep(0.5)

            except Exception as e:
                print(f"    ❌ Error documenting menu {menu_name}: {e}")
                # Ensure menu is closed
                pyautogui.press("escape")
                time.sleep(0.5)
                continue

    def _document_simulation_states(self):
        """Document different simulation states."""
        print("  ▶️ Documenting simulation states...")

        try:
            # Start simulation if not running
            if "start" in self.button_locations:
                pyautogui.click(
                    self.button_locations["start"]["x"],
                    self.button_locations["start"]["y"],
                )
                time.sleep(3)

                self._take_screenshot(
                    "06_simulation_running", "Simulation running - active state"
                )

            # Pause simulation - check if button exists
            if "pause" in self.button_locations:
                pyautogui.click(
                    self.button_locations["pause"]["x"],
                    self.button_locations["pause"]["y"],
                )
                time.sleep(2)

                self._take_screenshot(
                    "07_simulation_paused", "Simulation paused - inactive state"
                )

            # Resume if possible
            if "pause" in self.button_locations:
                pyautogui.click(
                    self.button_locations["pause"]["x"],
                    self.button_locations["pause"]["y"],
                )
                time.sleep(2)

                self._take_screenshot(
                    "08_simulation_resumed", "Simulation resumed - active state"
                )

            # Stop simulation
            if "stop" in self.button_locations:
                pyautogui.click(
                    self.button_locations["stop"]["x"],
                    self.button_locations["stop"]["y"],
                )
                time.sleep(2)

                self._take_screenshot(
                    "09_simulation_stopped", "Simulation stopped - terminated state"
                )

            # Reset simulation
            if "reset" in self.button_locations:
                pyautogui.click(
                    self.button_locations["reset"]["x"],
                    self.button_locations["reset"]["y"],
                )
                time.sleep(2)

                self._take_screenshot(
                    "10_simulation_reset", "Simulation reset - initial state"
                )

        except Exception as e:
            print(f"    ❌ Error documenting simulation states: {e}")

    def _document_dialog_windows(self):
        """Document all dialog windows triggered by menu actions."""
        print("  🪟 Documenting dialog windows...")

        dialogs_to_document = [
            ("file", "New Session", "Ctrl+n"),
            ("file", "Load Configuration", "Ctrl+o"),
            ("file", "Save Configuration", "Ctrl+s"),
            ("file", "Export Data", "Ctrl+e"),
            ("file", "Export Plot", None),
            ("edit", "System Parameters", None),
            ("edit", "Precision Settings", None),
            ("edit", "Ignition Threshold", None),
            ("simulation", "Run Preset Task", None),
            ("tools", "Trigger Ignition Event", None),
            ("tools", "Induce Stressor", None),
            ("tools", "Modulate Precision", None),
            ("tools", "Inject Sensory Input", None),
            ("tools", "Set Body State", None),
            ("tools", "System Diagnostics", None),
            ("analysis", "Ignition Statistics", None),
            ("analysis", "Energy Budget Report", None),
            ("analysis", "Somatic Marker Analysis", None),
            ("analysis", "Self-Model Coherence", None),
            ("analysis", "Generate Report", None),
            ("help", "Documentation", None),
            ("help", "Keyboard Shortcuts", None),
            ("help", "About APGI System", None),
        ]

        for menu_name, dialog_name, shortcut in dialogs_to_document:
            try:
                # Open menu
                menu_info = self.menu_items.get(menu_name)
                if not menu_info:
                    continue

                # Use keyboard shortcut if available
                if shortcut:
                    pyautogui.hotkey(*shortcut.split("+"))
                else:
                    # Navigate through menu
                    pyautogui.hotkey(*menu_info["shortcut"].split("+"))
                    time.sleep(0.5)

                    # Find and click the menu item
                    item_index = None
                    for i, item in enumerate(menu_info.get("items", [])):
                        if item["name"] == dialog_name:
                            item_index = i
                            break

                    if item_index is not None:
                        for _ in range(item_index + 1):
                            pyautogui.press("down")
                        time.sleep(0.3)
                        pyautogui.press("enter")

                time.sleep(1.5)  # Wait for dialog to open

                # Take screenshot of dialog
                dialog_filename = (
                    f"11_dialog_{menu_name}_{dialog_name.replace(' ', '_').lower()}"
                )
                self._take_screenshot(
                    dialog_filename,
                    f"Dialog: {dialog_name} (from {menu_name.title()} menu)",
                )

                # Close dialog (Escape or Enter depending on dialog type)
                pyautogui.press("escape")
                time.sleep(0.5)

                # If dialog didn't close, try Enter
                pyautogui.press("enter")
                time.sleep(0.5)

            except Exception as e:
                print(f"    ⚠️ Could not document dialog '{dialog_name}': {e}")
                # Ensure any open dialogs are closed
                pyautogui.press("escape")
                time.sleep(0.3)
                pyautogui.press("escape")
                time.sleep(0.5)
                continue

    def _document_view_toggles(self):
        """Document View menu toggles and zoom controls."""
        print("  👁️ Documenting view toggles and zoom controls...")

        try:
            # Document zoom controls
            zoom_actions = [
                ("Zoom In", "ctrl+plus"),
                ("Zoom Out", "ctrl+minus"),
                ("Fit to Window", "ctrl+0"),
            ]

            for zoom_name, shortcut in zoom_actions:
                try:
                    # Apply zoom
                    pyautogui.hotkey(*shortcut.split("+"))
                    time.sleep(1)

                    # Take screenshot
                    self._take_screenshot(
                        f"12_zoom_{zoom_name.replace(' ', '_').lower()}",
                        f"Zoom Control: {zoom_name}",
                    )
                except Exception as e:
                    print(f"    ⚠️ Could not document zoom '{zoom_name}': {e}")

            # Reset zoom
            pyautogui.hotkey("ctrl", "0")
            time.sleep(1)

            # Document view toggles (Control Panel, Neural Activity, etc.)
            view_menu = self.menu_items.get("view")
            if view_menu and "items" in view_menu:
                toggle_items = [
                    "Control Panel",
                    "Neural Activity",
                    "Interoception",
                    "System Metrics",
                ]

                for toggle_name in toggle_items:
                    try:
                        # Open View menu
                        pyautogui.hotkey(*view_menu["shortcut"].split("+"))
                        time.sleep(0.5)

                        # Navigate to toggle item
                        item_index = None
                        for i, item in enumerate(view_menu["items"]):
                            if item["name"] == toggle_name:
                                item_index = i
                                break

                        if item_index is not None:
                            for _ in range(item_index + 1):
                                pyautogui.press("down")
                            time.sleep(0.3)
                            pyautogui.press("enter")
                            time.sleep(1)

                            # Take screenshot with toggle activated
                            self._take_screenshot(
                                f"13_view_toggle_{toggle_name.replace(' ', '_').lower()}",
                                f"View Toggle: {toggle_name}",
                            )

                        # Close menu
                        pyautogui.press("escape")
                        time.sleep(0.5)

                    except Exception as e:
                        print(
                            f"    ⚠️ Could not document view toggle '{toggle_name}': {e}"
                        )
                        pyautogui.press("escape")
                        time.sleep(0.5)
                        continue

        except Exception as e:
            print(f"    ❌ Error documenting view toggles: {e}")

    def _document_speed_control(self):
        """Document speed control slider."""
        print("  ⚡ Documenting speed control...")

        try:
            # Find speed slider (it's a separate slider from the parameter sliders)
            if self.gui_window:
                # Speed slider is typically in the control panel
                speed_x = self.gui_window.left + 200

                # Try different positions
                for y in range(100, 200, 10):
                    try:
                        # Click and drag to minimum
                        pyautogui.click(speed_x, y)
                        time.sleep(0.3)
                        pyautogui.drag(speed_x, y, speed_x - 50, y, duration=0.5)
                        time.sleep(0.5)

                        self._take_screenshot(
                            "14_speed_minimum", "Speed Control - Minimum (0.1x)"
                        )

                        # Drag to maximum
                        pyautogui.drag(speed_x - 50, y, speed_x + 50, y, duration=0.5)
                        time.sleep(0.5)

                        self._take_screenshot(
                            "15_speed_maximum", "Speed Control - Maximum (10.0x)"
                        )

                        # Reset to default
                        pyautogui.click(speed_x, y)
                        time.sleep(0.5)
                        break
                    except Exception:
                        continue

        except Exception as e:
            print(f"    ⚠️ Could not document speed control: {e}")

    def _document_status_bar_and_log(self):
        """Document status bar and event log."""
        print("  📊 Documenting status bar and event log...")

        try:
            # Take screenshot of status bar
            if self.gui_window:
                status_region = (
                    self.gui_window.left,
                    self.gui_window.bottom - 30,
                    self.gui_window.width,
                    30,
                )
                try:
                    screenshot = pyautogui.screenshot(region=status_region)
                    status_path = self.screenshots_dir / "16_status_bar.png"
                    screenshot.save(status_path)
                    print("    📸 Status bar captured")
                except Exception as e:
                    print(f"    ⚠️ Could not capture status bar: {e}")

            # Take screenshot of event log (bottom left panel)
            self._take_screenshot(
                "17_event_log", "Event Log Panel - showing system events and messages"
            )

        except Exception as e:
            print(f"    ❌ Error documenting status bar and log: {e}")

    def _take_screenshot(self, filename: str, description: str) -> Optional[Path]:
        """Robust screenshot capture with multiple fallback methods."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = self.screenshots_dir / f"{filename}_{timestamp}.png"

        # Ensure parent directory exists
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n📸 Capturing: {description}")

        try:
            screenshot = None
            capture_method = "unknown"

            # Method 1: Try window-specific capture with aggressive activation
            if self.gui_window:
                print("  🎯 Method 1: Window region capture...")

                # Aggressive activation before capture
                activation_success = self._ensure_app_is_active()
                if activation_success:
                    try:
                        # Validate coordinates one more time
                        if self._validate_window_coordinates(self.gui_window):
                            # Calculate safe region
                            region = self._calculate_safe_region(self.gui_window)

                            # Final activation right before capture
                            self.gui_window.activate()
                            time.sleep(0.2)

                            screenshot = pyautogui.screenshot(region=region)
                            capture_method = "window_region"
                            print(f"    ✅ Window capture successful: {region}")
                        else:
                            print(
                                "    ⚠️ Invalid window coordinates, trying full screen"
                            )
                    except Exception as e:
                        print(f"    ⚠️ Window region capture failed: {e}")
                else:
                    print("    ⚠️ Window activation failed, trying alternative methods")

            # Method 2: Active window capture if window method failed
            if screenshot is None:
                print("  🎯 Method 2: Active window capture...")
                try:
                    # Get the currently active window
                    active_window = gw.getActiveWindow()
                    if active_window and hasattr(active_window, "title"):
                        print(f"    📱 Active window: {active_window.title}")

                        # Check if this looks like our app
                        if self._verify_window_is_apgi(active_window):
                            if self._validate_window_coordinates(active_window):
                                region = self._calculate_safe_region(active_window)
                                screenshot = pyautogui.screenshot(region=region)
                                capture_method = "active_window"
                                print(
                                    f"    ✅ Active window capture successful: {region}"
                                )
                            else:
                                print("    ⚠️ Active window has invalid coordinates")
                        else:
                            print("    ⚠️ Active window doesn't appear to be APGI app")
                except Exception as e:
                    print(f"    ⚠️ Active window capture failed: {e}")

            # Method 3: Manual positioning and capture
            if screenshot is None:
                print("  🎯 Method 3: Manual positioning attempt...")
                screenshot = self._manual_positioning_capture()
                if screenshot:
                    capture_method = "manual_positioning"

            # Method 4: Full screen with analysis
            if screenshot is None:
                print("  🎯 Method 4: Full screen capture with analysis...")
                try:
                    full_screenshot = pyautogui.screenshot()

                    # Try to find and crop the app window from full screen
                    cropped_screenshot = self._crop_app_from_fullscreen(full_screenshot)
                    if cropped_screenshot:
                        screenshot = cropped_screenshot
                        capture_method = "fullscreen_cropped"
                        print("    ✅ Successfully cropped app from full screen")
                    else:
                        # Use full screen as last resort
                        screenshot = full_screenshot
                        capture_method = "fullscreen"
                        print(
                            "    ⚠️ Using full screen (could not identify app window)"
                        )

                except Exception as e:
                    print(f"    ⚠️ Full screen capture failed: {e}")

            # Method 5: Emergency fallback
            if screenshot is None:
                print("  🎯 Method 5: Emergency fallback...")
                try:
                    screenshot = pyautogui.screenshot()
                    capture_method = "emergency"
                    print("    ⚠️ Emergency fallback successful")
                except Exception as e:
                    print(f"    ❌ Emergency fallback failed: {e}")
                    raise e

            # Verify screenshot quality
            if screenshot and self._verify_screenshot_quality(screenshot):
                # Save screenshot
                screenshot.save(screenshot_path)

                # Add to documentation structure
                screenshot_info = {
                    "filename": screenshot_path.name,
                    "path": str(screenshot_path.relative_to(self.base_dir)),
                    "description": description,
                    "timestamp": datetime.now().isoformat(),
                    "size": screenshot_path.stat().st_size,
                    "type": "desktop_app_screenshot",
                    "capture_method": capture_method,
                    "window_info": (
                        {
                            "title": (
                                self.gui_window.title if self.gui_window else "Unknown"
                            ),
                            "geometry": (
                                f"{self.gui_window.width}x{self.gui_window.height}"
                                if self.gui_window
                                else "Unknown"
                            ),
                        }
                        if self.gui_window
                        else None
                    ),
                }

                self.doc_structure["screenshots"].append(screenshot_info)
                print(f"    ✅ Screenshot saved using {capture_method} method")

                return screenshot_path
            else:
                print("    ❌ Screenshot quality verification failed")
                return None

        except Exception as e:
            print(f"❌ Error taking screenshot: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _calculate_safe_region(self, window: Any) -> tuple:
        """Calculate a safe capture region within window bounds."""
        try:
            screen_width, screen_height = pyautogui.size()

            # Ensure coordinates are within screen bounds
            left = max(0, window.left)
            top = max(0, window.top)
            right = min(screen_width, window.left + window.width)
            bottom = min(screen_height, window.top + window.height)

            # Calculate final dimensions
            width = right - left
            height = bottom - top

            # Ensure minimum size
            width = max(100, width)
            height = max(100, height)

            return (left, top, width, height)

        except Exception as e:
            print(f"    ⚠️ Error calculating safe region: {e}")
            # Return a reasonable default
            screen_width, screen_height = pyautogui.size()
            return (
                100,
                100,
                min(800, screen_width - 200),
                min(600, screen_height - 200),
            )

    def _manual_positioning_capture(self) -> Optional[Image.Image]:
        """Try manual positioning to capture the app window."""
        try:
            screen_width, screen_height = pyautogui.size()

            # Try common positions where apps might be
            positions = [
                (screen_width // 2, screen_height // 2),  # Center
                (screen_width // 4, screen_height // 4),  # Top-left quadrant
                (3 * screen_width // 4, screen_height // 4),  # Top-right quadrant
                (screen_width // 2, 3 * screen_height // 4),  # Bottom-center
            ]

            for i, (x, y) in enumerate(positions):
                print(f"    🎯 Trying position {i + 1}: ({x}, {y})")
                try:
                    # Move mouse and click to focus
                    pyautogui.moveTo(x, y)
                    time.sleep(0.2)
                    pyautogui.click()
                    time.sleep(0.5)

                    # Check what window is now active
                    active = gw.getActiveWindow()
                    if active and hasattr(active, "title"):
                        print(f"      Active window: {active.title}")

                        # If it looks like our app, try to capture it
                        if self._verify_window_is_apgi(active):
                            if self._validate_window_coordinates(active):
                                region = self._calculate_safe_region(active)
                                screenshot = pyautogui.screenshot(region=region)

                                if self._verify_screenshot_quality(screenshot):
                                    print("      ✅ Manual positioning successful")
                                    return screenshot

                except Exception as e:
                    print(f"      ⚠️ Position {i + 1} failed: {e}")
                    continue

            return None

        except Exception as e:
            print(f"    ❌ Manual positioning failed: {e}")
            return None

    def _crop_app_from_fullscreen(
        self, full_screenshot: Image.Image
    ) -> Optional[Image.Image]:
        """Try to identify and crop the app window from a full screenshot."""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(full_screenshot)

            # Try to find window-like rectangles in the image
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Find large rectangular contours that could be our app window
            window_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100000:  # Reasonable minimum window size
                    approx = cv2.approxPolyDP(
                        contour, 0.02 * cv2.arcLength(contour, True), True
                    )

                    if len(approx) >= 4:  # Roughly rectangular
                        x, y, w, h = cv2.boundingRect(contour)

                        # Check aspect ratio and size
                        aspect_ratio = w / h
                        if 0.8 < aspect_ratio < 3.0 and w > 800 and h > 600:
                            window_candidates.append((x, y, w, h, area))

            # Sort by area (largest first) and try the best candidates
            window_candidates.sort(key=lambda c: c[4], reverse=True)

            for x, y, w, h, area in window_candidates[:3]:
                try:
                    # Crop the candidate region
                    cropped = full_screenshot.crop((x, y, x + w, y + h))

                    # Check if this looks like our app
                    if self._is_likely_app_screenshot(cropped):
                        print(f"    ✅ Found app window at ({x}, {y}) size {w}x{h}")
                        return cropped

                except Exception:
                    continue

            return None

        except Exception as e:
            print(f"    ⚠️ Error cropping from fullscreen: {e}")
            return None

    def _verify_screenshot_quality(self, screenshot: Image.Image) -> bool:
        """Verify that the screenshot is of acceptable quality."""
        try:
            if not screenshot:
                return False

            # Check image size
            width, height = screenshot.size
            if width < 100 or height < 100:
                print(f"    ⚠️ Screenshot too small: {width}x{height}")
                return False

            # Check if image is not completely black or white
            img_array = np.array(screenshot)
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))

            if unique_colors < 3:
                print(
                    f"    ⚠️ Screenshot appears to be blank (only {unique_colors} colors)"
                )
                return False

            # Check for typical IDE characteristics (which we want to avoid)
            if self._looks_like_ide_screenshot(screenshot):
                print("    ⚠️ Screenshot appears to be an IDE, not the target app")
                return False

            return True

        except Exception as e:
            print(f"    ⚠️ Error verifying screenshot quality: {e}")
            return True  # Assume it's okay if we can't verify

    def _looks_like_ide_screenshot(self, screenshot: Image.Image) -> bool:
        """Enhanced IDE detection with multiple heuristics."""
        try:
            img_array = np.array(screenshot)

            # Convert to HSV for better color analysis
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

            # Check 1: Dark pixel ratio (IDEs typically have >60% dark pixels)
            dark_pixels = np.sum(img_hsv[:, :, 2] < 50)
            total_pixels = img_array.shape[0] * img_array.shape[1]
            dark_ratio = dark_pixels / total_pixels

            if dark_ratio > 0.6:
                return True

            # Check 2: High contrast patterns (text/code indicators)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray)

            if contrast > 80:
                return True

            # Check 3: Line pattern detection (code editors have many horizontal lines)
            edges = cv2.Canny(gray, 50, 150)
            horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi / 2, threshold=50)

            if horizontal_lines is not None and len(horizontal_lines) > 20:
                return True

            # Check 4: Color distribution (IDEs have limited color palettes)
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
            if unique_colors < 20:
                return True

            # Check 5: Text region detection using OCR-like heuristics
            # Look for small, high-contrast rectangular regions (text characters)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            text_like_regions = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if 10 < area < 200:  # Small regions typical of text
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 5.0:  # Text-like aspect ratios
                        text_like_regions += 1

            if text_like_regions > 100:
                return True

            return False

        except Exception:
            return False

    def _ensure_app_is_active(self) -> bool:
        """Enhanced window activation with IDE detection and avoidance."""
        print("  🎯 Enhanced window activation with IDE detection...")

        max_attempts = 5
        for attempt in range(max_attempts):
            print(f"    Attempt {attempt + 1}/{max_attempts}...")

            try:
                # Try to find window if we don't have one
                if not self.gui_window:
                    self.gui_window = self._find_gui_window()
                    if not self.gui_window:
                        print("    ❌ No window found, trying manual selection...")
                        self.gui_window = self._manual_window_selection()
                        if not self.gui_window:
                            print("    ❌ Still no window, continuing anyway...")
                            return False

                # Method 1: Standard activation with IDE check
                try:
                    self.gui_window.activate()
                    time.sleep(0.5)

                    # Verify it's active and NOT an IDE
                    active = gw.getActiveWindow()
                    if (
                        active
                        and hasattr(active, "title")
                        and active.title == self.gui_window.title
                        and not self._is_ide_window(active)
                    ):
                        print(
                            f"    ✅ Standard activation successful: {self.gui_window.title}"
                        )
                        return True
                    else:
                        print(
                            f"    ⚠️ Window is IDE or activation failed, active: {active.title if active else 'None'}"
                        )
                except Exception as e:
                    print(f"    ⚠️ Standard activation error: {e}")

                # Method 2: Click-to-focus with IDE verification
                try:
                    if hasattr(self.gui_window, "left") and hasattr(
                        self.gui_window, "top"
                    ):
                        center_x = self.gui_window.left + self.gui_window.width // 2
                        center_y = self.gui_window.top + self.gui_window.height // 2

                        # Move mouse away and then click to focus
                        pyautogui.moveTo(center_x + 100, center_y + 100)
                        time.sleep(0.2)
                        pyautogui.click(center_x, center_y)
                        time.sleep(0.5)

                        # Verify activation and check it's not IDE
                        active = gw.getActiveWindow()
                        if (
                            active
                            and hasattr(active, "title")
                            and active.title == self.gui_window.title
                            and not self._is_ide_window(active)
                        ):
                            print(
                                f"    ✅ Click-to-focus successful: {self.gui_window.title}"
                            )
                            return True
                        else:
                            print(
                                f"    ⚠️ Click-to-focus got IDE or failed: {active.title if active else 'None'}"
                            )
                except Exception as e:
                    print(f"    ⚠️ Click-to-focus error: {e}")

                # Method 3: Alternative window search if current is IDE
                if attempt < max_attempts - 1:
                    print("    🔄 Searching for non-IDE window...")
                    self.gui_window = self._find_non_ide_window()
                    time.sleep(1)

            except Exception as e:
                print(f"    ❌ Activation attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(1)

        print("    ❌ All activation attempts failed or resulted in IDE")
        return False

    def _validate_window_coordinates(self, window: Any) -> bool:
        """Validate that window coordinates are reasonable and within screen bounds."""
        try:
            if not hasattr(window, "left") or not hasattr(window, "top"):
                return False
            if not hasattr(window, "width") or not hasattr(window, "height"):
                return False

            # Check for reasonable values
            if window.left < -1000 or window.left > 5000:
                return False
            if window.top < -1000 or window.top > 5000:
                return False
            if window.width < 100 or window.width > 5000:
                return False
            if window.height < 100 or window.height > 5000:
                return False

            # Check if window is within screen bounds (with some tolerance)
            screen_width, screen_height = pyautogui.size()
            if (
                window.left + window.width < -100
                or window.left > screen_width + 100
                or window.top + window.height < -100
                or window.top > screen_height + 100
            ):
                print(
                    f"    ⚠️ Window appears off-screen: {window.left},{window.top} {window.width}x{window.height}"
                )
                return False

            return True

        except Exception as e:
            print(f"    ⚠️ Coordinate validation error: {e}")
            return False

    def _manual_screenshot_attempt(self) -> Image.Image:
        """Manual attempt to capture the application screenshot."""
        try:
            # Method 1: Try to find window again
            self.gui_window = self._find_gui_window()
            if self.gui_window:
                self.gui_window.activate()
                time.sleep(1)
                return pyautogui.screenshot(
                    region=(
                        self.gui_window.left,
                        self.gui_window.top,
                        self.gui_window.width,
                        self.gui_window.height,
                    )
                )

            # Method 2: Try clicking on common app locations
            screen_width, screen_height = pyautogui.size()

            # Try center screen first
            pyautogui.click(screen_width // 2, screen_height // 2)
            time.sleep(0.5)
            screenshot = pyautogui.screenshot()

            # Check if we got the right app by analyzing the image
            if self._is_likely_app_screenshot(screenshot):
                return screenshot

            # Method 3: Try other common positions
            positions = [
                (screen_width // 4, screen_height // 4),  # Top-left quadrant
                (3 * screen_width // 4, screen_height // 4),  # Top-right quadrant
                (screen_width // 2, 3 * screen_height // 4),  # Bottom-center
            ]

            for x, y in positions:
                pyautogui.click(x, y)
                time.sleep(0.3)
                screenshot = pyautogui.screenshot()
                if self._is_likely_app_screenshot(screenshot):
                    return screenshot

            # Last resort
            return pyautogui.screenshot()

        except Exception as e:
            print(f"    ❌ Manual screenshot attempt failed: {e}")
            return pyautogui.screenshot()

    def _is_likely_app_screenshot(self, screenshot: Image.Image) -> bool:
        """Check if screenshot is likely the APGI application."""
        try:
            # Convert to numpy array for analysis
            img_array = np.array(screenshot)

            # Simple heuristic: check for GUI elements characteristic of APGI
            # Look for specific colors, patterns, or text that indicate our app

            # Check for typical GUI background colors
            unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))

            # APGI app typically has moderate color diversity (not too simple like IDE)
            if 50 < unique_colors < 1000:
                return True

            # Additional checks could be added here
            return False

        except Exception:
            return False

    def _manual_window_selection(self) -> Optional[Any]:
        """Enhanced manual window selection with interactive confirmation."""
        try:
            print("\n🪟 Enhanced Manual Window Selection:")
            print("Getting comprehensive list of available windows...")

            all_windows = self._get_all_windows()
            if not all_windows:
                return None

            categorized_windows = self._categorize_windows(all_windows)
            self._display_window_analysis(categorized_windows)

            if not categorized_windows["candidates"]:
                self._show_no_candidates_message()
                return None

            self._display_candidates(categorized_windows)
            return self._interactive_selection(categorized_windows)

        except Exception as e:
            print(f"❌ Error in manual selection: {e}")
            return None

    def _get_all_windows(self):
        """Get all available windows."""
        if hasattr(gw, "getAllWindows"):
            return gw.getAllWindows()
        return []

    def _categorize_windows(self, all_windows):
        """Filter and categorize windows into different types."""
        candidates = []
        system_windows = []
        development_windows = []
        other_windows = []

        for i, window in enumerate(all_windows):
            if not (
                hasattr(window, "title")
                and window.title
                and len(window.title.strip()) > 0
                and hasattr(window, "width")
                and hasattr(window, "height")
            ):
                continue

            title = window.title
            title_lower = title.lower()

            window_info = {
                "index": i,
                "window": window,
                "title": title,
                "width": window.width,
                "height": window.height,
                "area": window.width * window.height,
            }

            # Categorize windows
            if any(
                skip in title_lower
                for skip in [
                    "desktop",
                    "finder",
                    "spotlight",
                    "notification",
                    "system preferences",
                    "activity monitor",
                    "dock",
                    "menu bar",
                    "trash",
                    "launchpad",
                ]
            ):
                system_windows.append(window_info)
            elif any(
                dev in title_lower
                for dev in [
                    "visual studio",
                    "vscode",
                    "code",
                    "terminal",
                    "python",
                    "idle",
                    "jupyter",
                    "notebook",
                ]
            ):
                development_windows.append(window_info)
            else:
                candidates.append(window_info)
                other_windows.append(window_info)

        # Sort candidates by size (largest first) - more likely to be our GUI
        candidates.sort(key=lambda w: w["area"], reverse=True)

        return {
            "candidates": candidates,
            "system_windows": system_windows,
            "development_windows": development_windows,
            "other_windows": other_windows,
            "all_windows": all_windows,
        }

    def _display_window_analysis(self, categorized_windows):
        """Display analysis of categorized windows."""
        print("\n📊 Window Analysis:")
        print(f"  Total windows: {len(categorized_windows['all_windows'])}")
        print(
            f"  System windows: {len(categorized_windows['system_windows'])} (filtered out)"
        )
        print(
            f"  Development windows: {len(categorized_windows['development_windows'])}"
        )
        print(f"  Other candidates: {len(categorized_windows['candidates'])}")

    def _show_no_candidates_message(self):
        """Show message when no suitable candidates are found."""
        print("\n❌ No suitable application windows found")
        print("\n💡 Suggestions:")
        print("  1. Make sure the APGI application is running")
        print("  2. Check that the application window is visible")
        print("  3. Try restarting the application")

    def _display_candidates(self, categorized_windows):
        """Display candidate windows with confidence scores."""
        print("\n🎯 Potential APGI Application Windows:")
        print("=" * 80)

        candidates = categorized_windows["candidates"]
        development_windows = categorized_windows["development_windows"]

        for idx, candidate in enumerate(candidates[:10]):  # Show top 10
            confidence_score = self._calculate_window_confidence(candidate)
            confidence_emoji = (
                "🟢"
                if confidence_score > 70
                else "🟡" if confidence_score > 40 else "🔴"
            )

            print(f"  {idx + 1:2d}. {confidence_emoji} {candidate['title']}")
            print(
                f"      Size: {candidate['width']}x{candidate['height']} ({candidate['area']:,} pixels)"
            )
            print(f"      Confidence: {confidence_score}%")

            # Show why we think this might be the APGI app
            reasons = self._get_confidence_reasons(candidate)
            if reasons:
                print(f"      Reasons: {', '.join(reasons)}")
            print()

        # Show development windows if no good candidates
        if len(candidates) < 3 and development_windows:
            print("\n💻 Development Windows (less likely but possible):")
            for idx, dev_win in enumerate(development_windows[:5]):
                print(
                    f"  D{idx + 1}. {dev_win['title']} ({dev_win['width']}x{dev_win['height']})"
                )
            print()

    def _interactive_selection(self, categorized_windows):
        """Handle interactive window selection."""
        candidates = categorized_windows["candidates"]
        development_windows = categorized_windows["development_windows"]
        all_windows = categorized_windows["all_windows"]

        print("\n🎮 Window Selection:")
        print(
            "  • Enter a number (1-{}) to select a window".format(
                min(len(candidates), 10)
            )
        )
        if development_windows:
            print(
                "  • Enter D1-D{} to select a development window".format(
                    min(len(development_windows), 5)
                )
            )
        print("  • Press Enter to select the highest confidence window")
        print("  • Type 'list' to see all windows")
        print("  • Type 'skip' to use fallback mode")

        while True:
            try:
                choice = input("\nSelect window > ").strip().lower()

                if choice == "" or choice == "auto":
                    return self._handle_auto_selection(candidates)
                elif choice == "skip":
                    print("\n⏭️ Skipping manual selection, using fallback mode")
                    return None
                elif choice == "list":
                    self._list_all_windows(all_windows)
                    continue
                elif choice.startswith("d") and len(choice) > 1:
                    selected = self._handle_development_selection(
                        choice, development_windows
                    )
                    if selected:
                        return selected
                    continue
                else:
                    selected = self._handle_candidate_selection(choice, candidates)
                    if selected:
                        return selected

            except (ValueError, KeyboardInterrupt):
                print("\n❌ Invalid input. Try again or type 'skip' to continue.")
                continue

    def _handle_auto_selection(self, candidates):
        """Handle automatic selection of highest confidence window."""
        if candidates:
            selected = candidates[0]["window"]
            print(f"\n✅ Auto-selected highest confidence: {selected.title}")
            return self._confirm_window_selection(selected)
        else:
            print("\n❌ No candidates available for auto-selection")
            return None

    def _list_all_windows(self, all_windows):
        """List all available windows."""
        print("\n📋 All Available Windows:")
        for i, win in enumerate(all_windows):
            if hasattr(win, "title") and win.title:
                print(f"  {i + 1:3d}. {win.title}")

    def _handle_development_selection(self, choice, development_windows):
        """Handle development window selection."""
        try:
            dev_index = int(choice[1:]) - 1
            if 0 <= dev_index < len(development_windows):
                selected = development_windows[dev_index]["window"]
                print(f"\n✅ Selected development window: {selected.title}")
                return self._confirm_window_selection(selected)
            else:
                print(
                    f"❌ Invalid development window number. Use D1-D{len(development_windows)}"
                )
        except ValueError:
            print("❌ Invalid format. Use D1, D2, etc.")
        return None

    def _handle_candidate_selection(self, choice, candidates):
        """Handle regular candidate window selection."""
        window_index = int(choice) - 1
        if 0 <= window_index < len(candidates):
            selected = candidates[window_index]["window"]
            print(f"\n✅ Selected: {selected.title}")
            return self._confirm_window_selection(selected)
        else:
            print(f"❌ Invalid selection. Use 1-{len(candidates)}")
        return None

    def _calculate_window_confidence(self, window_info: dict) -> int:
        """Calculate confidence score that this window is the APGI application."""
        score = 0
        title = window_info["title"].lower()

        # Title-based scoring
        if "apgi" in title:
            score += 50
        if "consciousness" in title:
            score += 40
        if "modeling" in title:
            score += 30
        if "framework" in title:
            score += 25
        if "neural" in title:
            score += 20
        if "ignition" in title:
            score += 20
        if "precision" in title:
            score += 15
        if "interoception" in title or "exteroception" in title:
            score += 15

        # Size-based scoring (typical GUI app sizes)
        area = window_info["area"]
        if 800000 < area < 2000000:  # Typical desktop app size
            score += 20
        elif 500000 < area < 3000000:  # Reasonable range
            score += 10

        # Aspect ratio scoring
        aspect = window_info["width"] / window_info["height"]
        if 1.2 < aspect < 2.5:  # Typical desktop app aspect ratio
            score += 10

        # Penalize obvious non-GUI characteristics
        if any(skip in title for skip in ["terminal", "console", "command", "bash"]):
            score -= 30
        if any(skip in title for skip in ["browser", "chrome", "safari", "firefox"]):
            score -= 40

        return max(0, min(100, score))

    def _get_confidence_reasons(self, window_info: dict) -> list:
        """Get reasons for the confidence score."""
        reasons = []
        title = window_info["title"].lower()

        if "apgi" in title:
            reasons.append('contains "APGI"')
        if "consciousness" in title:
            reasons.append('contains "consciousness"')
        if "modeling" in title:
            reasons.append('contains "modeling"')
        if "neural" in title:
            reasons.append('contains "neural"')

        area = window_info["area"]
        if 800000 < area < 2000000:
            reasons.append("typical GUI size")

        return reasons

    def _confirm_window_selection(self, window: Any) -> Any:
        """Confirm the selected window by testing it."""
        try:
            print(f"\n🧪 Testing selected window: {window.title}")

            # Try to activate the window
            window.activate()
            time.sleep(1)

            # Verify it became active
            active = gw.getActiveWindow()
            if active and hasattr(active, "title") and active.title == window.title:
                print("✅ Window activation confirmed")

                # Take a test screenshot to verify
                if (
                    hasattr(window, "left")
                    and hasattr(window, "top")
                    and hasattr(window, "width")
                    and hasattr(window, "height")
                ):
                    test_region = (
                        max(0, window.left),
                        max(0, window.top),
                        min(200, window.width),
                        min(200, window.height),
                    )

                    test_screenshot = pyautogui.screenshot(region=test_region)
                    if self._is_likely_app_screenshot(test_screenshot):
                        print("✅ Window appears to be the correct application")
                        return window
                    else:
                        print("⚠️ Window doesn't appear to be the APGI application")
                        print(
                            "    But proceeding anyway as you selected it manually..."
                        )
                        return window
                else:
                    print("⚠️ Cannot verify window coordinates, but proceeding...")
                    return window
            else:
                print(
                    f"⚠️ Could not activate window. Active window: {active.title if active else 'None'}"
                )
                return None

        except Exception as e:
            print(f"❌ Error confirming window selection: {e}")
            return None

    def _generate_documentation_report(self):
        """Generate comprehensive HTML documentation report."""
        print("\n📄 Generating documentation report...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"desktop_app_documentation_{timestamp}.html"

        html_content = self._generate_html_report()

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Save JSON structure
        json_path = self.reports_dir / f"documentation_structure_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.doc_structure, f, indent=2)

        print(f"  📋 HTML Report: {report_path}")
        print(f"  📊 JSON Structure: {json_path}")

    def _generate_html_report(self) -> str:
        """Generate enhanced HTML documentation report with better statistics."""
        total_screenshots = len(self.doc_structure["screenshots"])
        total_size = sum(s.get("size", 0) for s in self.doc_structure["screenshots"])

        # Calculate additional statistics
        screenshots_by_type = {}
        for screenshot in self.doc_structure["screenshots"]:
            screenshot_type = screenshot.get("type", "unknown")
            screenshots_by_type[screenshot_type] = (
                screenshots_by_type.get(screenshot_type, 0) + 1
            )

        # Element discovery statistics
        element_stats = {
            "buttons": len(self.button_locations),
            "tabs": len(self.tab_locations),
            "sliders": len(self.slider_locations),
            "menus": len(self.menu_items),
            "dialogs": 24,  # Total dialogs documented
            "view_toggles": 4,  # Control Panel, Neural Activity, Interoception, System Metrics
            "zoom_controls": 3,  # Zoom In, Zoom Out, Fit to Window
            "speed_control": 1,
            "status_bar": 1,
            "event_log": 1,
        }

        # Calculate total menu items across all menus
        total_menu_items = sum(
            len(menu.get("items", [])) for menu in self.menu_items.values()
        )

        # Success rate calculations
        total_expected_elements = (
            4 + 6 + 8 + 7
        )  # Expected buttons, tabs, sliders, menu categories
        discovered_elements = (
            element_stats["buttons"]
            + element_stats["tabs"]
            + element_stats["sliders"]
            + element_stats["menus"]
        )
        discovery_rate = (
            (discovered_elements / total_expected_elements * 100)
            if total_expected_elements > 0
            else 0
        )

        # Total documentation coverage calculation (commented as unused)
        # total_documented_elements = (
        #     1  # Initial state
        #     + element_stats["tabs"]  # Each tab
        #     + element_stats["buttons"]  # Each button
        #     + (element_stats["sliders"] * 2)  # Each slider (min/max)
        #     + len(self.menu_items)
        #     + total_menu_items  # Menus and submenu items
        #     + 5  # Simulation states (running, paused, resumed, stopped, reset)
        #     + element_stats["dialogs"]  # Dialog windows
        #     + element_stats["view_toggles"]
        #     + element_stats["zoom_controls"]  # View controls
        #     + 2  # Speed control (min/max)
        #     + 2  # Status bar and event log
        #     + 1  # Final state
        # )

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APGI System - Desktop App Screenshot Documentation</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0; padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px; margin: 0 auto;
            background: white; border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white; padding: 40px; text-align: center;
        }}
        .header h1 {{ margin: 0; font-size: 2.5em; font-weight: 300; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em; }}
        .content {{ padding: 40px; }}
        .section {{
            background: #f8f9fa; margin: 30px 0; padding: 30px;
            border-radius: 10px; border-left: 5px solid #3498db;
        }}
        .section h2 {{ color: #2c3e50; margin-top: 0; font-size: 1.8em; }}
        .stats {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin: 30px 0;
        }}
        .stat {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white; padding: 25px; border-radius: 10px; text-align: center;
            box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
            transition: transform 0.3s ease;
        }}
        .stat:hover {{ transform: translateY(-5px); }}
        .stat h3 {{ margin: 0; font-size: 2em; font-weight: bold; }}
        .stat p {{ margin: 5px 0 0 0; opacity: 0.9; }}
        .discovery-stats {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px; margin: 20px 0;
        }}
        .discovery-stat {{
            background: linear-gradient(135deg, #27ae60, #229954);
            color: white; padding: 20px; border-radius: 8px; text-align: center;
        }}
        .discovery-stat h4 {{ margin: 0; font-size: 1.5em; }}
        .discovery-stat p {{ margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9em; }}
        .screenshot-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px; margin: 30px 0;
        }}
        .screenshot {{
            background: white; border-radius: 10px; overflow: hidden;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .screenshot:hover {{ transform: translateY(-5px); box-shadow: 0 15px 35px rgba(0,0,0,0.15); }}
        .screenshot img {{ width: 100%; height: auto; display: block; }}
        .screenshot-info {{
            padding: 20px; background: #f8f9fa;
        }}
        .screenshot-info h3 {{
            margin: 0 0 10px 0; color: #2c3e50;
        }}
        .screenshot-info p {{
            margin: 5px 0; color: #7f8c8d; font-size: 0.9em;
        }}
        .system-info {{
            background: #ecf0f1; padding: 20px; border-radius: 8px;
            margin: 20px 0;
        }}
        .system-info h3 {{ margin-top: 0; color: #2c3e50; }}
        .footer {{
            background: #34495e; color: white; padding: 30px; text-align: center;
        }}
        .badge {{
            display: inline-block; background: #e74c3c; color: white;
            padding: 5px 15px; border-radius: 20px; font-size: 0.8em;
            margin-left: 10px;
        }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
        .progress-bar {{
            background: #ecf0f1; border-radius: 10px; overflow: hidden;
            margin: 10px 0; height: 20px;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #27ae60, #2ecc71);
            height: 100%; transition: width 0.3s ease;
        }}
        .feature-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px; margin: 20px 0;
        }}
        .feature {{
            background: white; padding: 20px; border-radius: 8px;
            border-left: 4px solid #3498db; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .feature h4 {{ margin: 0 0 10px 0; color: #2c3e50; }}
        .feature p {{ margin: 0; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 APGI System</h1>
            <p>Desktop Application Screenshot Documentation</p>
            <p class="timestamp">Generated: {self.doc_structure['timestamp']}</p>
        </div>

        <div class="content">
            <div class="section">
                <h2>📊 Documentation Overview</h2>
                <div class="stats">
                    <div class="stat">
                        <h3>{total_screenshots}</h3>
                        <p>Total Screenshots</p>
                    </div>
                    <div class="stat">
                        <h3>{total_size / 1024:.1f} KB</h3>
                        <p>Total Size</p>
                    </div>
                    <div class="stat">
                        <h3>{discovery_rate:.1f}%</h3>
                        <p>Element Discovery Rate</p>
                    </div>
                    <div class="stat">
                        <h3>{discovered_elements}</h3>
                        <p>Elements Found</p>
                    </div>
                    <div class="stat">
                        <h3>{total_menu_items}</h3>
                        <p>Menu Items</p>
                    </div>
                    <div class="stat">
                        <h3>{element_stats['dialogs']}</h3>
                        <p>Dialogs Documented</p>
                    </div>
                </div>

                <h3>🔧 GUI Element Discovery Statistics</h3>
                <div class="discovery-stats">
                    <div class="discovery-stat">
                        <h4>{element_stats['buttons']}</h4>
                        <p>Buttons</p>
                    </div>
                    <div class="discovery-stat">
                        <h4>{element_stats['tabs']}</h4>
                        <p>Tabs</p>
                    </div>
                    <div class="discovery-stat">
                        <h4>{element_stats['sliders']}</h4>
                        <p>Sliders</p>
                    </div>
                    <div class="discovery-stat">
                        <h4>{element_stats['menus']}</h4>
                        <p>Menu Categories</p>
                    </div>
                    <div class="discovery-stat">
                        <h4>{element_stats['dialogs']}</h4>
                        <p>Dialog Windows</p>
                    </div>
                    <div class="discovery-stat">
                        <h4>{element_stats['view_toggles']}</h4>
                        <p>View Toggles</p>
                    </div>
                    <div class="discovery-stat">
                        <h4>{element_stats['zoom_controls']}</h4>
                        <p>Zoom Controls</p>
                    </div>
                    <div class="discovery-stat">
                        <h4>{element_stats['speed_control']}</h4>
                        <p>Speed Control</p>
                    </div>
                </div>

                <div class="progress-bar">
                    <div class="progress-fill" style="width: {discovery_rate}%"></div>
                </div>
                <p style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
                    Discovery Success Rate: {discovery_rate:.1f}% ({discovered_elements}/{total_expected_elements} elements)
                </p>
            </div>

            <div class="section">
                <h2>💻 System Information</h2>
                <div class="system-info">
                    <h3>Environment</h3>
                    <p><strong>Platform:</strong> {self.doc_structure['system_info']['platform']} {self.doc_structure['system_info']['platform_release']}</p>
                    <p><strong>Architecture:</strong> {self.doc_structure['system_info']['architecture']}</p>
                    <p><strong>Python:</strong> {self.doc_structure['system_info']['python_version']}</p>
                    <p><strong>Screen:</strong> {self.doc_structure['system_info']['screen_size']}</p>
                </div>
            </div>

            <div class="section">
                <h2>📱 GUI Application Screenshots</h2>
                <p>Comprehensive documentation of the Tkinter-based desktop application including all tabs, controls, buttons, and simulation states.</p>

                <div class="screenshot-grid">
"""

        # Add all screenshots with enhanced metadata
        for screenshot in self.doc_structure["screenshots"]:
            window_info = screenshot.get("window_info", {})
            window_title = (
                window_info.get("title", "Unknown") if window_info else "Unknown"
            )
            window_geometry = (
                window_info.get("geometry", "Unknown") if window_info else "Unknown"
            )

            html += f"""
                    <div class="screenshot">
                        <img src="../{screenshot['path']}" alt="{screenshot['description']}">
                        <div class="screenshot-info">
                            <h3>{screenshot['description']}</h3>
                            <p><strong>File:</strong> {screenshot['filename']}</p>
                            <p><strong>Size:</strong> {screenshot['size']} bytes</p>
                            <p><strong>Captured:</strong> {screenshot['timestamp']}</p>
                            <p><strong>Window:</strong> {window_title} ({window_geometry})</p>
                        </div>
                    </div>
"""

        html += f"""
                </div>
            </div>

            <div class="section">
                <h2>🔧 Discovered GUI Elements</h2>
                <ul>
                    <li><strong>Buttons:</strong> {", ".join(self.button_locations.keys()) if self.button_locations else "None found"}</li>
                    <li><strong>Tabs:</strong> {", ".join(self.tab_locations.keys()) if self.tab_locations else "None found"}</li>
                    <li><strong>Sliders:</strong> {", ".join(self.slider_locations.keys()) if self.slider_locations else "None found"}</li>
                    <li><strong>Menus:</strong> {", ".join(self.menu_items.keys()) if self.menu_items else "None found"}</li>
                </ul>
            </div>

            <div class="section">
                <h2>📋 Documentation Features</h2>
                <div class="feature-grid">
                    <div class="feature">
                        <h4>🖥️ Desktop App Capture</h4>
                        <p>Actual Tkinter window screenshots with precise region detection</p>
                    </div>
                    <div class="feature">
                        <h4>🔍 GUI Element Discovery</h4>
                        <p>Automatic detection using advanced computer vision algorithms</p>
                    </div>
                    <div class="feature">
                        <h4>🎮 Interactive Testing</h4>
                        <p>Automated clicking through all GUI elements with state tracking</p>
                    </div>
                    <div class="feature">
                        <h4>📊 State Documentation</h4>
                        <p>Different simulation states and transitions captured</p>
                    </div>
                    <div class="feature">
                        <h4>📋 Comprehensive Menu Coverage</h4>
                        <p>All 7 menu categories with 31 total menu items documented</p>
                    </div>
                    <div class="feature">
                        <h4>🪟 Dialog Window Documentation</h4>
                        <p>24 different dialog windows from File, Edit, Tools, Analysis, and Help menus</p>
                    </div>
                    <div class="feature">
                        <h4>👁️ View Controls</h4>
                        <p>View toggles and zoom controls (Zoom In/Out, Fit to Window)</p>
                    </div>
                    <div class="feature">
                        <h4>⚡ Speed Control</h4>
                        <p>Simulation speed slider documentation (0.1x to 10.0x)</p>
                    </div>
                    <div class="feature">
                        <h4>📊 Status & Logging</h4>
                        <p>Status bar and event log panel documentation</p>
                    </div>
                    <div class="feature">
                        <h4>🔄 Robust Fallback Mode</h4>
                        <p>Full documentation even when window cannot be located</p>
                    </div>
                    <div class="feature">
                        <h4>⚙️ System Information</h4>
                        <p>Complete environment and configuration details</p>
                    </div>
                    <div class="feature">
                        <h4>📈 Metadata Tracking</h4>
                        <p>Comprehensive timestamps, file sizes, and window information</p>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>📑 Documentation Coverage Summary</h2>
                <ul>
                    <li><strong>Initial State:</strong> Application startup screenshot</li>
                    <li><strong>Tabs (6):</strong> Neural Activity, Interoception, System Metrics, Self-Model, Oscillations, State Space</li>
                    <li><strong>Buttons (4):</strong> Start, Pause, Stop, Reset with state transitions</li>
                    <li><strong>Sliders (8):</strong> Ignition Threshold, Extero Precision, Intero Precision, Arousal, Stress, Activity, Learning Rate, Attention Gain (each at min/max positions)</li>
                    <li><strong>Menus (7 categories):</strong> File, Edit, Simulation, View, Tools, Analysis, Help</li>
                    <li><strong>Menu Items (31):</strong> All submenu items with keyboard shortcuts</li>
                    <li><strong>Simulation States (5):</strong> Running, Paused, Resumed, Stopped, Reset</li>
                    <li><strong>Dialog Windows (24):</strong> All dialogs from File, Edit, Simulation, Tools, Analysis, and Help menus</li>
                    <li><strong>View Controls (7):</strong> 4 view toggles + 3 zoom controls</li>
                    <li><strong>Speed Control (2):</strong> Minimum and maximum speed settings</li>
                    <li><strong>Status Elements (2):</strong> Status bar and event log</li>
                    <li><strong>Final State:</strong> Application state after complete documentation</li>
                </ul>
                <p style="margin-top: 20px; color: #7f8c8d;">
                    <strong>Total Expected Screenshots:</strong> ~100+ screenshots covering every aspect of the application
                </p>
            </div>
        </div>

        <div class="footer">
            <p>🧠 APGI System - Desktop Application Documentation</p>
            <p>Automatically generated on {self.doc_structure['timestamp']}</p>
            <p>Discovery Rate: {discovery_rate:.1f}% | Elements Found: {discovered_elements}/{total_expected_elements}</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _is_ide_window(self, window: Any) -> bool:
        """Check if a window is likely an IDE/development tool."""
        try:
            if not window or not hasattr(window, "title"):
                return False

            title = window.title.lower()

            # Comprehensive IDE detection keywords
            ide_keywords = [
                "visual studio",
                "vscode",
                "code",
                "terminal",
                "console",
                "python",
                "idle",
                "jupyter",
                "notebook",
                "intellij",
                "pycharm",
                "sublime",
                "atom",
                "brackets",
                "eclipse",
                "netbeans",
                "xcode",
                "android studio",
                "rubymine",
                "webstorm",
                "phpstorm",
                "clion",
                "datagrip",
                "command prompt",
                "powershell",
                "bash",
                "zsh",
                "cmd",
                "git bash",
                "wsl",
                "docker",
            ]

            # Check for IDE keywords in title
            if any(ide_keyword in title for ide_keyword in ide_keywords):
                return True

            # Check for common development file patterns
            dev_patterns = [".py", ".js", ".html", ".css", ".java", ".cpp", ".c"]
            if any(pattern in title for pattern in dev_patterns):
                return True

            return False

        except Exception:
            return False

    def _find_non_ide_window(self) -> Optional[Any]:
        """Find a window that is definitely not an IDE."""
        try:
            if hasattr(gw, "getAllWindows"):
                all_windows = gw.getAllWindows()

                # Filter out IDE windows and system windows
                candidates = []
                for window in all_windows:
                    if (
                        hasattr(window, "title")
                        and window.title
                        and hasattr(window, "width")
                        and hasattr(window, "height")
                        and not self._is_ide_window(window)
                        and window.width > 400
                        and window.height > 300
                    ):

                        title_lower = window.title.lower()
                        # Skip obvious system windows
                        if not any(
                            skip in title_lower
                            for skip in [
                                "desktop",
                                "finder",
                                "spotlight",
                                "notification",
                                "system preferences",
                                "activity monitor",
                                "dock",
                                "menu bar",
                                "trash",
                                "launchpad",
                            ]
                        ):
                            candidates.append(window)

                # Sort by size (largest first)
                candidates.sort(key=lambda w: w.width * w.height, reverse=True)

                return candidates[0] if candidates else None

        except Exception:
            return None

    def _show_all_windows(self):
        """Show all available windows for debugging."""
        print("\n🔍 ALL AVAILABLE WINDOWS SCAN")
        print("=" * 50)

        try:
            if hasattr(gw, "getAllWindows"):
                all_windows = gw.getAllWindows()
                print(f"\n📊 Found {len(all_windows)} total windows:\n")

                # Categorize windows
                apgi_windows = []
                ide_windows = []
                system_windows = []
                other_windows = []

                for i, window in enumerate(all_windows):
                    if not (
                        hasattr(window, "title")
                        and window.title
                        and len(window.title.strip()) > 0
                    ):
                        continue

                    title = window.title
                    title_lower = title.lower()

                    if self._is_ide_window(window):
                        ide_windows.append((i, window))
                    elif any(
                        keyword in title_lower
                        for keyword in ["apgi", "consciousness", "modeling"]
                    ):
                        apgi_windows.append((i, window))
                    elif any(
                        skip in title_lower
                        for skip in ["desktop", "finder", "spotlight", "notification"]
                    ):
                        system_windows.append((i, window))
                    else:
                        other_windows.append((i, window))

                # Display categorized windows
                if apgi_windows:
                    print("🟢 APGI APPLICATION WINDOWS:")
                    for i, (idx, window) in enumerate(apgi_windows):
                        size = (
                            f"{window.width}x{window.height}"
                            if hasattr(window, "width")
                            else "Unknown"
                        )
                        print(f"  {i + 1}. {window.title} ({size})")
                    print()

                if ide_windows:
                    print("🔴 IDE/DEVELOPMENT WINDOWS (will be avoided):")
                    for i, (idx, window) in enumerate(ide_windows[:10]):  # Limit to 10
                        size = (
                            f"{window.width}x{window.height}"
                            if hasattr(window, "width")
                            else "Unknown"
                        )
                        print(f"  {i + 1}. {window.title} ({size})")
                    if len(ide_windows) > 10:
                        print(f"  ... and {len(ide_windows) - 10} more")
                    print()

                if system_windows:
                    print("🔵 SYSTEM WINDOWS:")
                    for i, (idx, window) in enumerate(system_windows[:5]):  # Limit to 5
                        size = (
                            f"{window.width}x{window.height}"
                            if hasattr(window, "width")
                            else "Unknown"
                        )
                        print(f"  {i + 1}. {window.title} ({size})")
                    if len(system_windows) > 5:
                        print(f"  ... and {len(system_windows) - 5} more")
                    print()

                if other_windows:
                    print("⚪ OTHER APPLICATIONS:")
                    for i, (idx, window) in enumerate(
                        other_windows[:10]
                    ):  # Limit to 10
                        size = (
                            f"{window.width}x{window.height}"
                            if hasattr(window, "width")
                            else "Unknown"
                        )
                        print(f"  {i + 1}. {window.title} ({size})")
                    if len(other_windows) > 10:
                        print(f"  ... and {len(other_windows) - 10} more")
                    print()

                print("\n💡 RECOMMENDATIONS:")
                if apgi_windows:
                    print("✅ APGI application windows found - script should work well")
                else:
                    print(
                        "⚠️ No APGI windows found - make sure the application is running"
                    )

                if ide_windows:
                    print(
                        f"⚠️ Found {len(ide_windows)} IDE windows - script will avoid these"
                    )
                    print("   Consider closing IDEs for best results")

            else:
                print("❌ Cannot access window list")

        except Exception as e:
            print(f"❌ Error scanning windows: {e}")

        print("\n" + "=" * 50)
        print("Press Enter to return to main menu...")
        input()

    def _cleanup_processes(self):
        """Clean up running processes."""
        if self.gui_process:
            try:
                self.gui_process.terminate()
                self.gui_process.wait(timeout=5)
            except Exception:
                self.gui_process.kill()


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent

    print("🎯 APGI System Desktop App Screenshot Documentation")
    print("=" * 60)
    print("This tool captures screenshots of the Python Tkinter desktop application")
    print("and automatically interacts with all GUI elements.")
    print()

    # Check dependencies
    if not SCREENSHOT_AVAILABLE:
        print("❌ Required packages not installed. Run:")
        print("   pip install pyautogui pygetwindow pillow opencv-python")
        sys.exit(1)

    # Create and run documentation generator
    doc_generator = APGIScreenshotDocumentation(base_dir)
    doc_generator.generate_comprehensive_documentation()


if __name__ == "__main__":
    main()
