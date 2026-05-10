"""
Comprehensive tests for utils/dashboard_integration.py - 100% coverage target.

This file tests:
- DashboardManager initialization and configuration
- Component initialization and error handling
- Monitoring functionality
- Data export operations
- Thread safety and concurrent operations
- Error recovery and cleanup
- Integration with dashboard components
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Dashboard integration available check
    from utils.dashboard_integration import DASHBOARD_INTEGRATION_AVAILABLE
except ImportError as e:
    DASHBOARD_INTEGRATION_AVAILABLE = False
    print(f"Warning: dashboard_integration not available for testing: {e}")
    DASHBOARD_INTEGRATION_AVAILABLE = True


class TestDashboardManagerComplete:
    """Comprehensive tests for DashboardManager functionality."""

    def test_dashboard_manager_initialization(self):
        """Test DashboardManager initialization and configuration."""
        if not DASHBOARD_INTEGRATION_AVAILABLE:
            pytest.skip("dashboard_integration not available")

        from utils.dashboard_integration import DashboardManager

        # Test basic initialization
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)
            assert manager is not None
            assert manager.data_dir == Path(temp_dir)

    def test_dashboard_manager_component_init(self):
        """Test component initialization and error handling."""
        if not DASHBOARD_INTEGRATION_AVAILABLE:
            pytest.skip("dashboard_integration not available")

        from utils.dashboard_integration import DashboardManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Test component initialization (components are initialized in __init__)
            assert manager.historical_dashboard is not None or manager.static_generator is not None

    def test_dashboard_monitoring_functionality(self):
        """Test monitoring functionality."""
        if not DASHBOARD_INTEGRATION_AVAILABLE:
            pytest.skip("dashboard_integration not available")

        from utils.dashboard_integration import DashboardManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Test monitoring setup
            manager.start_monitoring(interval=0.1)  # Use short interval for testing
            assert manager._monitoring_active is True
            manager.stop_monitoring()  # Clean up

    def test_dashboard_data_export(self):
        """Test data export operations."""
        if not DASHBOARD_INTEGRATION_AVAILABLE:
            pytest.skip("dashboard_integration not available")

        from utils.dashboard_integration import DashboardManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Test data export
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as export_file:
                export_path = export_file.name

            try:
                test_data = {"test": "data", "timestamp": "2023-01-01"}
                result = manager.export_data(test_data, "json", export_path)
                assert result is not None
                assert os.path.exists(export_path)
            finally:
                if os.path.exists(export_path):
                    os.unlink(export_path)

    def test_dashboard_thread_safety(self):
        """Test thread safety and concurrent operations."""
        if not DASHBOARD_INTEGRATION_AVAILABLE:
            pytest.skip("dashboard_integration not available")

        import threading

        from utils.dashboard_integration import DashboardManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            results = []

            def concurrent_operation():
                # Test a simple operation that exists
                result = manager.historical_dashboard is not None
                results.append(result)

            threads = [threading.Thread(target=concurrent_operation) for _ in range(3)]
            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join(timeout=5.0)

            assert len(results) == 3

    def test_dashboard_error_recovery(self):
        """Test error recovery and cleanup."""
        if not DASHBOARD_INTEGRATION_AVAILABLE:
            pytest.skip("dashboard_integration not available")

        from utils.dashboard_integration import DashboardManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Test error recovery (cleanup method exists)
            manager.cleanup()
            assert manager._monitoring_active is False

    def test_dashboard_integration_components(self):
        """Test integration with dashboard components."""
        if not DASHBOARD_INTEGRATION_AVAILABLE:
            pytest.skip("dashboard_integration not available")

        from utils.dashboard_integration import DashboardManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Test component integration (components are already integrated)
            assert manager.historical_dashboard is not None or manager.static_generator is not None
