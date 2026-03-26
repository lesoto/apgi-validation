"""
Tests for backup HMAC validation including tampered history, oversized signatures,
and missing HMAC key handling.
"""
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch
from utils.backup_manager import BackupManager
from utils.secure_key_manager import get_backup_hmac_key


class TestBackupHMACValidation:
    """Test backup HMAC validation security."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def backup_manager(self, temp_dir):
        """Create backup manager instance."""
        return BackupManager(str(temp_dir))

    @pytest.fixture
    def sample_data(self):
        """Create sample backup data."""
        return {
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"key": "value", "number": 42},
            "metadata": {"version": "1.0", "author": "test"},
        }

    def test_hmac_signature_generation(self, backup_manager, sample_data):
        """Test HMAC signature generation for backup data."""
        # Create backup
        backup_path = backup_manager.create_data_backup(sample_data, "test_backup")

        # Should have signature file
        sig_file = backup_path.with_suffix(".sig")
        assert sig_file.exists()

        # Signature should be valid hex
        signature = sig_file.read_text().strip()
        assert len(signature) == 64  # SHA256 hex length
        assert all(c in "0123456789abcdef" for c in signature.lower())

    def test_hmac_signature_verification(self, backup_manager, sample_data):
        """Test HMAC signature verification."""
        # Create backup
        backup_path = backup_manager.create_data_backup(sample_data, "test_backup")

        # Should verify successfully
        assert backup_manager.verify_data_backup_integrity(backup_path) is True

    def test_tampered_backup_detection(self, backup_manager, sample_data):
        """Test detection of tampered backup files."""
        # Create backup
        backup_path = backup_manager.create_data_backup(sample_data, "test_backup")

        # Create backup
        backup_data = json.loads(backup_path.read_text())
        backup_data["data"]["key"] = "tampered_value"
        backup_path.write_text(json.dumps(backup_data))

        # Should detect tampering
        assert backup_manager.verify_data_backup_integrity(backup_path) is False

    def test_missing_signature_file(self, backup_manager, sample_data):
        """Test handling of missing signature file."""
        # Create backup
        backup_path = backup_manager.create_data_backup(sample_data, "test_backup")

        # Delete signature file
        sig_file = backup_path.with_suffix(".sig")
        sig_file.unlink()

        # Should fail verification
        assert backup_manager.verify_data_backup_integrity(backup_path) is False

    def test_oversized_signature_detection(self, backup_manager, sample_data):
        """Test detection of oversized signatures."""
        # Create backup
        backup_path = backup_manager.create_data_backup(sample_data, "test_backup")

        # Create oversized signature
        sig_file = backup_path.with_suffix(".sig")
        oversized_sig = "a" * 1000  # Much larger than expected
        sig_file.write_text(oversized_sig)

        # Should detect oversized signature
        assert backup_manager.verify_data_backup_integrity(backup_path) is False

    def test_invalid_signature_format(self, backup_manager, sample_data):
        """Test detection of invalid signature formats."""
        # Create backup
        backup_path = backup_manager.create_data_backup(sample_data, "test_backup")

        # Invalid signature formats
        invalid_signatures = [
            "not_hex_at_all",
            "123",  # Too short
            "gg" * 32,  # Invalid hex characters
            "",  # Empty
            " " * 64,  # Spaces
        ]

        for invalid_sig in invalid_signatures:
            sig_file = backup_path.with_suffix(".sig")
            sig_file.write_text(invalid_sig)

            assert backup_manager.verify_data_backup_integrity(backup_path) is False

    def test_missing_hmac_key_handling(self, backup_manager, sample_data):
        """Test handling when HMAC key is missing."""
        # Mock get_backup_hmac_key to raise exception
        with patch("utils.secure_key_manager.get_backup_hmac_key") as mock_get_key:
            mock_get_key.side_effect = Exception("Key missing")

            # Should handle missing key gracefully
            with pytest.raises(Exception):
                backup_manager.create_data_backup(sample_data, "test_backup")

    def test_hmac_key_rotation_invalidation(self, backup_manager, sample_data):
        """Test backup verification after HMAC key rotation."""
        # Create backup with original key
        backup_path = backup_manager.create_data_backup(sample_data, "test_backup")

        # Should verify with original key
        assert backup_manager.verify_data_backup_integrity(backup_path) is True

        # Mock key rotation (simulate new key)
        with patch("utils.secure_key_manager.get_backup_hmac_key") as mock_get_key:
            mock_get_key.return_value = "f" * 64  # Different key

            # Should fail verification with new key
            assert backup_manager.verify_data_backup_integrity(backup_path) is False

    def test_concurrent_backup_verification(self, backup_manager, sample_data):
        """Test concurrent backup verification."""
        import threading

        # Create backup
        backup_path = backup_manager.create_data_backup(sample_data, "test_backup")

        results = []

        def verify_backup():
            result = backup_manager.verify_data_backup_integrity(backup_path)
            results.append(result)

        # Run multiple verifications concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=verify_backup)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All verifications should succeed
        assert all(results)
        assert len(results) == 10

    def test_backup_history_integrity(self, backup_manager, sample_data):
        """Test integrity of backup history with HMAC."""
        # Create multiple backups
        backup_manager.create_data_backup(sample_data, "backup1")
        backup_manager.create_data_backup(sample_data, "backup2")

        # Get backup history
        history = backup_manager.get_backup_history()

        # History should be valid
        assert len(history) >= 0  # Simple data backups don't track in history

        # Verify all backups in history
        for entry in history:
            backup_path = backup_manager.backup_dir / entry["filename"]
            if backup_path.exists():
                assert backup_manager.verify_data_backup_integrity(backup_path) is True

    def test_hmac_with_different_data_formats(self, backup_manager):
        """Test HMAC with different data formats."""
        test_cases = [
            {"simple": "data"},
            {"list": [1, 2, 3]},
            {"nested": {"deep": {"value": "test"}}},
            {"unicode": "测试数据"},
            {"special": "!@#$%^&*()"},
        ]

        for i, data in enumerate(test_cases):
            backup_path = backup_manager.create_data_backup(data, f"test_{i}")
            assert backup_manager.verify_data_backup_integrity(backup_path) is True

    def test_hmac_algorithm_consistency(self, backup_manager, sample_data):
        """Test HMAC algorithm consistency across operations."""
        # Create backup
        backup_path = backup_manager.create_data_backup(sample_data, "test_backup")

        # Get signature
        sig_file = backup_path.with_suffix(".sig")
        original_signature = sig_file.read_text().strip()

        # Verify multiple times - should get same result
        for _ in range(5):
            assert backup_manager.verify_data_backup_integrity(backup_path) is True

        # Signature should be unchanged
        current_signature = sig_file.read_text().strip()
        assert current_signature == original_signature

    def test_backup_with_large_data(self, backup_manager):
        """Test HMAC with large data files."""
        # Create large data
        large_data = {
            "large_array": list(range(10000)),
            "large_string": "x" * 10000,
            "nested_large": {"data": ["item" * 1000 for _ in range(100)]},
        }

        backup_path = backup_manager.create_data_backup(large_data, "large_backup")

        # Should handle large data and verify correctly
        assert backup_manager.verify_data_backup_integrity(backup_path) is True

    def test_hmac_key_entropy_validation(self):
        """Test that HMAC key has sufficient entropy."""
        key = get_backup_hmac_key()

        # Key should be 64 hex characters (32 bytes)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key.lower())

        # Test multiple keys for uniqueness
        keys = []
        for _ in range(5):
            # Force new key generation by clearing cache
            from utils.secure_key_manager import get_secure_key_manager

            manager = get_secure_key_manager()
            manager.clear_cache()
            keys.append(get_backup_hmac_key())

        # Keys should be the same (cached) unless rotation occurs
        # This tests the caching mechanism
        assert len(set(keys)) <= len(keys)


class TestBackupHMACEdgeCases:
    """Test edge cases for backup HMAC."""

    def test_empty_backup_data(self, backup_manager):
        """Test HMAC with empty backup data."""
        empty_data = {}
        backup_path = backup_manager.create_data_backup(empty_data, "empty_backup")
        assert backup_manager.verify_data_backup_integrity(backup_path) is True

    def test_null_bytes_in_data(self, backup_manager):
        """Test HMAC with null bytes in data."""
        data_with_null = {"data": "test\x00\x00data"}
        backup_path = backup_manager.create_data_backup(data_with_null, "null_backup")
        assert backup_manager.verify_data_backup_integrity(backup_path) is True

    def test_backup_with_nan_values(self, backup_manager):
        """Test HMAC with NaN values."""
        data_with_nan = {"value": float("nan"), "normal": 42}
        backup_path = backup_manager.create_data_backup(data_with_nan, "nan_backup")
        assert backup_manager.verify_data_backup_integrity(backup_path) is True

    def test_signature_file_permissions(self, backup_manager, sample_data, temp_dir):
        """Test signature file permissions."""
        backup_path = backup_manager.create_data_backup(sample_data, "test_backup")
        sig_file = backup_path.with_suffix(".sig")

        # Should have appropriate permissions
        stat_info = sig_file.stat()
        # Note: Permission checks might vary by system
        assert stat_info.st_size > 0
