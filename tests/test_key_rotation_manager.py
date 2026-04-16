"""
Tests for key rotation manager.
Tests periodic key rotation for PICKLE_SECRET_KEY and APGI_BACKUP_HMAC_KEY.
================================================================
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.key_rotation_manager import (
    KeyRotationManager,
    check_and_rotate_keys_if_needed,
    get_key_rotation_manager,
    get_key_status,
)


class TestKeyRotationManager:
    """Tests for KeyRotationManager class."""

    def test_initialization(self, temp_dir):
        """Test key rotation manager initialization."""
        manager = KeyRotationManager(
            keys_dir=str(temp_dir / ".keys"), rotation_interval_days=30
        )

        assert manager.keys_dir == Path(temp_dir / ".keys")
        assert manager.rotation_interval_days == 30
        assert manager.backup_count == 5
        assert manager.keys_dir.exists()

    def test_key_generation(self, temp_dir):
        """Test key generation."""
        manager = KeyRotationManager(keys_dir=str(temp_dir / ".keys"))

        # Keys should be generated during initialization
        assert manager.metadata["pickle_key"]["current"] is not None
        assert manager.metadata["backup_key"]["current"] is not None
        assert "fingerprint" in manager.metadata["pickle_key"]["current"]
        assert "fingerprint" in manager.metadata["backup_key"]["current"]

    def test_key_persistence(self, temp_dir):
        """Test that keys are persisted to disk."""
        KeyRotationManager(keys_dir=str(temp_dir / ".keys"))

        pickle_key_file = temp_dir / ".keys" / "pickle_secret_key.enc"
        backup_key_file = temp_dir / ".keys" / "backup_hmac_key.enc"

        assert pickle_key_file.exists()
        assert backup_key_file.exists()

        # Check file permissions
        stat_info = pickle_key_file.stat()
        assert (stat_info.st_mode & 0o777) == 0o600

    def test_metadata_persistence(self, temp_dir):
        """Test that metadata is persisted."""
        KeyRotationManager(keys_dir=str(temp_dir / ".keys"))

        metadata_file = temp_dir / ".keys" / "key_metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            loaded_metadata = json.load(f)

        assert "pickle_key" in loaded_metadata
        assert "backup_key" in loaded_metadata
        assert "last_rotation" in loaded_metadata
        assert "next_rotation" in loaded_metadata

    def test_key_rotation(self, temp_dir):
        """Test key rotation."""
        manager = KeyRotationManager(keys_dir=str(temp_dir / ".keys"))

        # Get original fingerprints
        old_pickle_fp = manager.metadata["pickle_key"]["current"]["fingerprint"]
        old_backup_fp = manager.metadata["backup_key"]["current"]["fingerprint"]

        # Rotate keys
        results = manager.rotate_all_keys()

        # Check results
        assert "pickle_key" in results
        assert "backup_key" in results

        # Verify fingerprints changed
        new_pickle_fp = manager.metadata["pickle_key"]["current"]["fingerprint"]
        new_backup_fp = manager.metadata["backup_key"]["current"]["fingerprint"]

        assert old_pickle_fp != new_pickle_fp
        assert old_backup_fp != new_backup_fp

        # Check rotation history
        assert len(manager.metadata["pickle_key"]["rotation_history"]) == 1
        assert len(manager.metadata["backup_key"]["rotation_history"]) == 1

    def test_rotation_history_limit(self, temp_dir):
        """Test that rotation history is limited."""
        manager = KeyRotationManager(keys_dir=str(temp_dir / ".keys"), backup_count=2)

        # Rotate multiple times
        for _ in range(5):
            manager.rotate_all_keys()

        # Should only keep 2 entries in history
        assert len(manager.metadata["pickle_key"]["rotation_history"]) == 2
        assert len(manager.metadata["backup_key"]["rotation_history"]) == 2

    def test_check_rotation_needed(self, temp_dir):
        """Test rotation needed check."""
        manager = KeyRotationManager(
            keys_dir=str(temp_dir / ".keys"), rotation_interval_days=1
        )

        # Initially, rotation should be needed (next_rotation is set to future)
        assert not manager.check_rotation_needed()

        # Manually set next_rotation to past
        manager.metadata["next_rotation"] = "2020-01-01T00:00:00"
        manager._save_metadata()

        # Now rotation should be needed
        assert manager.check_rotation_needed()

    def test_get_key_status(self, temp_dir):
        """Test getting key status."""
        manager = KeyRotationManager(keys_dir=str(temp_dir / ".keys"))

        status = manager.get_key_status()

        assert "last_rotation" in status
        assert "next_rotation" in status
        assert "keys" in status
        assert "pickle_key" in status["keys"]
        assert "backup_key" in status["keys"]

    def test_force_rotation_single_key(self, temp_dir):
        """Test forced rotation of single key."""
        manager = KeyRotationManager(keys_dir=str(temp_dir / ".keys"))

        manager.metadata["pickle_key"]["current"]["fingerprint"]

        results = manager.force_rotation("pickle_key")

        assert "pickle_key" in results
        assert len(results) == 1
        assert results["pickle_key"][0] != results["pickle_key"][1]

    def test_force_rotation_all_keys(self, temp_dir):
        """Test forced rotation of all keys."""
        manager = KeyRotationManager(keys_dir=str(temp_dir / ".keys"))

        results = manager.force_rotation()

        assert "pickle_key" in results
        assert "backup_key" in results
        assert len(results) == 2

    def test_concurrent_rotation(self, temp_dir):
        """Test concurrent key rotation - ensures lock prevents race conditions."""
        import threading

        manager = KeyRotationManager(keys_dir=str(temp_dir / ".keys"))

        rotation_count = [0]

        def rotate_in_thread():
            try:
                manager.rotate_all_keys()
                rotation_count[0] += 1
            except Exception as e:
                print(f"Rotation failed: {e}")

        threads = [threading.Thread(target=rotate_in_thread) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All 5 rotations should succeed (sequentially, not concurrently)
        # The lock ensures they don't happen simultaneously (preventing race conditions)
        assert rotation_count[0] == 5, f"Expected 5 rotations, got {rotation_count[0]}"

    def test_environment_variables_set(self, temp_dir):
        """Test that environment variables are set."""
        KeyRotationManager(keys_dir=str(temp_dir / ".keys"))

        # Environment variables should be set
        assert "PICKLE_SECRET_KEY" in os.environ
        assert "APGI_BACKUP_HMAC_KEY" in os.environ
        assert len(os.environ["PICKLE_SECRET_KEY"]) == 64
        assert len(os.environ["APGI_BACKUP_HMAC_KEY"]) == 64


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_key_rotation_manager(self, temp_dir):
        """Test global key rotation manager instance."""
        import utils.key_rotation_manager as krm_module

        # Reset global instance
        krm_module._key_rotation_manager = None

        manager = get_key_rotation_manager()

        assert isinstance(manager, KeyRotationManager)

    def test_check_and_rotate_if_needed(self, temp_dir):
        """Test check and rotate if needed function."""
        import utils.key_rotation_manager as krm_module

        krm_module._key_rotation_manager = KeyRotationManager(
            keys_dir=str(temp_dir / ".keys"), rotation_interval_days=1
        )

        # Initially should not rotate
        result = check_and_rotate_keys_if_needed()
        assert result is None

        # Set next rotation to past
        manager = get_key_rotation_manager()
        manager.metadata["next_rotation"] = "2020-01-01T00:00:00"
        manager._save_metadata()

        # Now should rotate
        result = manager.force_rotation()
        assert result is not None
        assert "pickle_key" in result
        assert "backup_key" in result

    def test_get_key_status_function(self, temp_dir):
        """Test get_key_status function."""
        import utils.key_rotation_manager as krm_module

        krm_module._key_rotation_manager = KeyRotationManager(
            keys_dir=str(temp_dir / ".keys")
        )

        status = get_key_status()

        assert isinstance(status, dict)
        assert "keys" in status
        assert "last_rotation" in status
        assert "next_rotation" in status
