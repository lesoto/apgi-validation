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

import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.dashboard_integration import DashboardManager
    from utils.data_protection import (
        apply_retention_policy,
        minimize_data,
        secure_delete,
        tag_pii_in_data,
    )

    DASHBOARD_INTEGRATION_AVAILABLE = True
except ImportError as e:
    DASHBOARD_INTEGRATION_AVAILABLE = False
    print(f"Warning: dashboard_integration not available for testing: {e}")


class TestDashboardManagerComplete:
    """Comprehensive tests for DashboardManager functionality."""

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_dashboard_manager_initialization(self):
        """Test DashboardManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            assert manager.data_dir == Path(temp_dir)
            assert manager._monitoring_active is False
            assert manager._monitoring_thread is None
            assert isinstance(manager._callbacks, list)
            assert isinstance(manager._export_history, list)

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_component_initialization_with_errors(self):
        """Test component initialization error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock component imports to raise exceptions
            with patch("utils.dashboard_integration.STATIC_AVAILABLE", False):
                with patch("utils.dashboard_integration.HISTORICAL_AVAILABLE", False):
                    with patch(
                        "utils.dashboard_integration.PERFORMANCE_AVAILABLE", False
                    ):
                        manager = DashboardManager(data_dir=temp_dir)

                        # Should handle missing components gracefully
                        assert manager.static_generator is None
                        assert manager.historical_dashboard is None
                        assert manager.performance_dashboard is None

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_monitoring_functionality(self):
        """Test monitoring functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Test start monitoring
            manager.start_monitoring(interval=1.0)
            assert manager._monitoring_active is True
            assert manager._monitoring_thread is not None

            # Test stop monitoring
            manager.stop_monitoring()
            assert manager._monitoring_active is False

            # Wait for thread to finish
            if manager._monitoring_thread:
                manager._monitoring_thread.join(timeout=1.0)

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_callback_registration(self):
        """Test callback registration and execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Mock callback
            callback_calls = []

            def test_callback(data):
                callback_calls.append(data)

            # Register callback
            manager.register_callback(test_callback)
            assert len(manager._callbacks) == 1

            # Trigger callbacks
            test_data = {"timestamp": datetime.now(), "value": 42}
            manager._trigger_callbacks(test_data)

            assert len(callback_calls) == 1
            assert callback_calls[0] == test_data

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_data_export_json(self):
        """Test JSON data export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Test data
            test_data = {
                "protocol": "VP_01",
                "results": [1, 2, 3, 4, 5],
                "metadata": {"timestamp": "2023-01-01T00:00:00Z"},
            }

            # Export to JSON
            export_path = manager.export_data(
                test_data, format="json", filename="test_export"
            )

            assert export_path.exists()
            assert export_path.suffix == ".json"

            # Verify content
            with open(export_path, "r") as f:
                loaded_data = json.load(f)

            assert loaded_data == test_data

            # Check export history
            assert len(manager._export_history) == 1
            assert manager._export_history[0]["format"] == "json"

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_data_export_csv(self):
        """Test CSV data export functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Test data with tabular structure
            test_data = {
                "subjects": [1, 2, 3],
                "accuracy": [0.8, 0.9, 0.85],
                "rt": [500, 450, 600],
            }

            # Export to CSV
            export_path = manager.export_data(
                test_data, format="csv", filename="test_export"
            )

            assert export_path.exists()
            assert export_path.suffix == ".csv"

            # Check export history
            assert len(manager._export_history) == 1
            assert manager._export_history[0]["format"] == "csv"

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_export_error_handling(self):
        """Test export error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Test invalid format
            with pytest.raises(ValueError):
                manager.export_data({}, format="invalid", filename="test")

            # Test empty data
            with pytest.raises(ValueError):
                manager.export_data({}, format="json", filename="test")

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_thread_safety(self):
        """Test thread safety of dashboard operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            results = []
            errors = []

            def worker_function(worker_id):
                try:
                    # Simulate concurrent operations
                    for i in range(5):
                        data = {"worker": worker_id, "iteration": i}
                        export_path = manager.export_data(
                            data, format="json", filename=f"worker_{worker_id}_{i}"
                        )
                        results.append(export_path.name)
                except Exception as e:
                    errors.append(e)

            # Start multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker_function, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Should have no errors and all results
            assert len(errors) == 0
            assert len(results) == 15  # 3 workers * 5 iterations

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_static_dashboard_generation(self):
        """Test static dashboard generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Mock static generator
            mock_generator = Mock()
            manager.static_generator = mock_generator

            # Test dashboard generation
            test_data = {"metrics": {"accuracy": 0.85, "precision": 0.90}}

            manager.generate_static_dashboard(test_data, "test_dashboard")

            # Should call generator
            mock_generator.generate_dashboard.assert_called_once()

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_historical_data_storage(self):
        """Test historical data storage functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Mock historical dashboard
            mock_historical = Mock()
            manager.historical_dashboard = mock_historical

            # Test data storage
            test_data = {
                "timestamp": datetime.now(),
                "protocol": "VP_01",
                "results": {"accuracy": 0.85},
            }

            manager.store_historical_data(test_data)

            # Should call storage method
            mock_historical.store_data.assert_called_once()

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Mock performance dashboard
            mock_performance = Mock()
            manager.performance_dashboard = mock_performance

            # Test performance data
            perf_data = {"cpu_usage": 45.2, "memory_usage": 1024, "response_time": 0.15}

            manager.update_performance_metrics(perf_data)

            # Should call update method
            mock_performance.update_metrics.assert_called_once()

    @pytest.mark.skipif(
        not DASHBOARD_INTEGRATION_AVAILABLE,
        reason="dashboard_integration not available",
    )
    def test_cleanup_and_resource_management(self):
        """Test cleanup and resource management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)

            # Start monitoring with longer interval to avoid thread issues
            manager.start_monitoring(interval=1.0)
            assert manager._monitoring_active is True

            # Cleanup
            manager.cleanup()

            # Should stop monitoring and clean resources
            assert manager._monitoring_active is False
            assert manager._callbacks == []
            assert manager._export_history == []


class TestDataProtectionComplete:
    """Comprehensive tests for data protection functionality."""

    def test_pii_detection_email(self):
        """Test PII detection for email addresses."""
        test_data = "Contact us at support@example.com for help"
        result = tag_pii_in_data(test_data)

        assert "email" in result
        assert "support@example.com" in result["email"]

    def test_pii_detection_ssn(self):
        """Test PII detection for Social Security Numbers."""
        test_data = "My SSN is 123-45-6789 for verification"
        result = tag_pii_in_data(test_data)

        assert "ssn" in result
        assert "123-45-6789" in result["ssn"]

    def test_pii_detection_phone(self):
        """Test PII detection for phone numbers."""
        test_data = "Call me at 555-123-4567 for details"
        result = tag_pii_in_data(test_data)

        assert "phone" in result
        assert "555-123-4567" in result["phone"]

    def test_pii_detection_multiple_types(self):
        """Test PII detection for multiple PII types."""
        test_data = "Email john.doe@company.com, phone 555-987-6543, SSN 987-65-4321"
        result = tag_pii_in_data(test_data)

        assert len(result) == 3
        assert "email" in result
        assert "phone" in result
        assert "ssn" in result

    def test_pii_detection_no_pii(self):
        """Test PII detection with no PII present."""
        test_data = "This is a clean text with no personal information"
        result = tag_pii_in_data(test_data)

        assert len(result) == 0

    def test_data_minimization_email(self):
        """Test data minimization for email addresses."""
        test_data = "Contact support@example.com for help"
        result = minimize_data(test_data)

        assert "support@example.com" not in result
        # The email has 19 characters, so it should be replaced with 19 asterisks
        assert "*******************" in result

    def test_data_minimization_custom_char(self):
        """Test data minimization with custom redaction character."""
        test_data = "Call 555-123-4567 for details"
        result = minimize_data(test_data, redaction_char="#")

        assert "555-123-4567" not in result
        # The phone number has 12 characters (including dashes), so it should be replaced with 12 # characters
        assert "############" in result

    def test_data_minimization_multiple_pii(self):
        """Test data minimization with multiple PII types."""
        test_data = "Email john@doe.com, phone 555-123-4567"
        result = minimize_data(test_data)

        assert "john@doe.com" not in result
        assert "555-123-4567" not in result
        # john@doe.com has 11 characters, 555-123-4567 has 12 characters
        assert "***********" in result
        assert "************" in result

    def test_secure_delete_success(self):
        """Test successful secure file deletion."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"Sensitive data to be securely deleted")
            temp_path = temp_file.name

        # Verify file exists
        assert Path(temp_path).exists()

        # Secure delete
        result = secure_delete(temp_path)

        assert result is True
        assert not Path(temp_path).exists()

    def test_secure_delete_nonexistent_file(self):
        """Test secure delete of non-existent file."""
        result = secure_delete("/nonexistent/path/file.txt")

        assert result is False

    def test_secure_delete_directory(self):
        """Test secure delete attempt on directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = secure_delete(temp_dir)

            assert result is False

    def test_secure_delete_permission_error(self):
        """Test secure delete with permission error."""
        # Create a file and make it read-only
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"Test data")
            temp_path = temp_file.name

        # Make file read-only
        Path(temp_path).chmod(0o444)

        try:
            result = secure_delete(temp_path)
            assert result is False
        finally:
            # Restore permissions for cleanup
            Path(temp_path).chmod(0o644)
            Path(temp_path).unlink()

    def test_retention_policy_dry_run(self):
        """Test retention policy in dry run mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create old and new files
            old_file = Path(temp_dir) / "old_file.txt"
            new_file = Path(temp_dir) / "new_file.txt"

            old_file.write_text("old data")
            new_file.write_text("new data")

            # Make old file appear old
            old_time = time.time() - (400 * 24 * 3600)  # 400 days ago
            os.utime(old_file, (old_time, old_time))

            # Run retention policy in dry run mode
            result = apply_retention_policy(temp_dir, max_age_days=365, dry_run=True)

            assert "files_to_delete" in result
            assert len(result["files_to_delete"]) == 1
            assert "old_file.txt" in str(result["files_to_delete"][0])

            # Files should still exist in dry run
            assert old_file.exists()
            assert new_file.exists()

    def test_retention_policy_actual_deletion(self):
        """Test retention policy with actual deletion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create old file
            old_file = Path(temp_dir) / "old_file.txt"
            old_file.write_text("old data")

            # Make file appear old
            old_time = time.time() - (400 * 24 * 3600)  # 400 days ago
            os.utime(old_file, (old_time, old_time))

            # Run retention policy with actual deletion
            result = apply_retention_policy(temp_dir, max_age_days=365, dry_run=False)

            assert "deleted_count" in result
            assert result["deleted_count"] == 1
            assert not old_file.exists()

    def test_retention_policy_empty_directory(self):
        """Test retention policy on empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = apply_retention_policy(temp_dir, max_age_days=365, dry_run=True)

            assert "files_to_delete" in result
            assert len(result["files_to_delete"]) == 0

    def test_retention_policy_invalid_directory(self):
        """Test retention policy on invalid directory."""
        result = apply_retention_policy(
            "/nonexistent/directory", max_age_days=365, dry_run=True
        )

        assert "status" in result
        assert result["status"] == "error"
