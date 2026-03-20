"""
Periodic key rotation for PICKLE_SECRET_KEY and APGI_BACKUP_HMAC_KEY.
Implements secure key rotation with persistence and notification.
===========================================================
"""

import os
import json
import base64
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging


class KeyRotationManager:
    """Manages periodic rotation of security keys."""

    def __init__(
        self,
        rotation_interval_days: int = 30,
        backup_count: int = 5,
        keys_dir: str = ".keys",
        notification_enabled: bool = False,
    ):
        """
        Initialize key rotation manager.

        Args:
            rotation_interval_days: Days between key rotations
            backup_count: Number of old keys to keep as backups
            keys_dir: Directory to store keys
            notification_enabled: Whether to send rotation notifications
        """
        self.rotation_interval_days = rotation_interval_days
        self.backup_count = backup_count
        self.keys_dir = Path(keys_dir)
        self.notification_enabled = notification_enabled

        # Thread lock for thread-safe operations
        self._lock = threading.Lock()

        # Ensure keys directory exists
        self.keys_dir.mkdir(exist_ok=True)

        # Set restrictive permissions on keys directory
        self.keys_dir.chmod(0o700)

        # Initialize logger
        self.logger = logging.getLogger("key_rotation")
        self.logger.setLevel(logging.INFO)

        # Load or create key metadata
        self.metadata_file = self.keys_dir / "key_metadata.json"
        self.metadata = self._load_metadata()

        # Initialize keys
        self._initialize_keys()

    def _load_metadata(self) -> Dict:
        """Load key metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load metadata: {e}")
        return {
            "pickle_key": {"current": None, "rotation_history": []},
            "backup_key": {"current": None, "rotation_history": []},
            "last_rotation": None,
            "next_rotation": None,
        }

    def _save_metadata(self) -> None:
        """Save key metadata to file."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save metadata: {e}")

    def _initialize_keys(self) -> None:
        """Initialize or load keys."""
        with self._lock:
            # Initialize PICKLE_SECRET_KEY
            pickle_key_file = self.keys_dir / "pickle_secret_key.enc"
            if pickle_key_file.exists():
                self._load_key_from_file("pickle_key", pickle_key_file)
            else:
                self._generate_and_save_key("pickle_key", pickle_key_file)

            # Initialize APGI_BACKUP_HMAC_KEY
            backup_key_file = self.keys_dir / "backup_hmac_key.enc"
            if backup_key_file.exists():
                self._load_key_from_file("backup_key", backup_key_file)
            else:
                self._generate_and_save_key("backup_key", backup_key_file)

            # Set next rotation date
            self._schedule_next_rotation()

            # Set environment variables
            self._set_env_vars()

    def _generate_and_save_key(self, key_type: str, key_file: Path) -> None:
        """Generate and save a new key."""
        # Generate cryptographically secure key
        key_bytes = os.urandom(32)

        # Calculate key fingerprint
        fingerprint = hashlib.sha256(key_bytes).hexdigest()[:16]

        # Save encrypted key (in production, use proper encryption)
        # For now, we use base64 encoding with restricted permissions
        key_b64 = base64.b64encode(key_bytes).decode()
        key_file.write_text(key_b64)
        key_file.chmod(0o600)

        # Update metadata
        timestamp = datetime.utcnow().isoformat()
        self.metadata[f"{key_type}_key"]["current"] = {
            "fingerprint": fingerprint,
            "created_at": timestamp,
            "file": str(key_file),
        }

        self._save_metadata()
        self.logger.info(f"Generated new {key_type} with fingerprint {fingerprint}")

    def _load_key_from_file(self, key_type: str, key_file: Path) -> Path:
        """Load key from file."""
        try:
            key_b64 = key_file.read_text()
            key_bytes = base64.b64decode(key_b64)
            fingerprint = hashlib.sha256(key_bytes).hexdigest()[:16]

            self.metadata[f"{key_type}_key"]["current"] = {
                "fingerprint": fingerprint,
                "file": str(key_file),
            }
            self._save_metadata()

            return key_file
        except Exception as e:
            self.logger.error(f"Could not load {key_type}: {e}")
            raise

    def _schedule_next_rotation(self) -> None:
        """Schedule next key rotation."""
        if self.metadata["last_rotation"]:
            last_rotation = datetime.fromisoformat(self.metadata["last_rotation"])
        else:
            last_rotation = datetime.utcnow()

        next_rotation = last_rotation + timedelta(days=self.rotation_interval_days)
        self.metadata["next_rotation"] = next_rotation.isoformat()
        self._save_metadata()

        self.logger.info(f"Next key rotation scheduled for {next_rotation}")

    def _set_env_vars(self) -> None:
        """Set environment variables for current keys."""
        # Load current keys
        pickle_key_file = self.keys_dir / "pickle_secret_key.enc"
        backup_key_file = self.keys_dir / "backup_hmac_key.enc"

        if pickle_key_file.exists():
            pickle_key_b64 = pickle_key_file.read_text()
            pickle_key = base64.b64decode(pickle_key_b64).hex()
            os.environ["PICKLE_SECRET_KEY"] = pickle_key

        if backup_key_file.exists():
            backup_key_b64 = backup_key_file.read_text()
            backup_key = base64.b64decode(backup_key_b64).hex()
            os.environ["APGI_BACKUP_HMAC_KEY"] = backup_key

    def _rotate_key(self, key_type: str) -> Tuple[str, str]:
        """
        Rotate a key and save old key as backup.

        Args:
            key_type: Type of key to rotate ("pickle_key" or "backup_key")

        Returns:
            Tuple of (old_fingerprint, new_fingerprint)
        """
        # Get current key info
        current_key_info = self.metadata[f"{key_type}_key"]["current"]
        old_fingerprint = current_key_info["fingerprint"]
        old_file = Path(current_key_info["file"])

        # Generate new key
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        new_key_file = self.keys_dir / f"{key_type}_{timestamp}.enc"
        self._generate_and_save_key(key_type, new_key_file)
        new_key_info = self.metadata[f"{key_type}_key"]["current"]
        new_fingerprint = new_key_info["fingerprint"]

        # Add to rotation history
        rotation_entry = {
            "old_fingerprint": old_fingerprint,
            "new_fingerprint": new_fingerprint,
            "rotated_at": datetime.utcnow().isoformat(),
            "old_file": str(old_file),
            "new_file": str(new_key_file),
        }
        self.metadata[f"{key_type}_key"]["rotation_history"].append(rotation_entry)

        # Limit rotation history
        if (
            len(self.metadata[f"{key_type}_key"]["rotation_history"])
            > self.backup_count
        ):
            self.metadata[f"{key_type}_key"]["rotation_history"].pop(0)

        # Update last rotation
        self.metadata["last_rotation"] = datetime.utcnow().isoformat()
        self._schedule_next_rotation()
        self._save_metadata()

        # Update environment variables
        self._set_env_vars()

        self.logger.info(
            f"Rotated {key_type}: {old_fingerprint[:8]}... -> {new_fingerprint[:8]}..."
        )

        return old_fingerprint, new_fingerprint

    def rotate_all_keys(self) -> Dict[str, Tuple[str, str]]:
        """
        Rotate all keys.

        Returns:
            Dictionary mapping key types to (old_fingerprint, new_fingerprint)
        """
        with self._lock:
            results = {}

            # Rotate PICKLE_SECRET_KEY
            results["pickle_key"] = self._rotate_key("pickle_key")

            # Rotate APGI_BACKUP_HMAC_KEY
            results["backup_key"] = self._rotate_key("backup_key")

            # Send notification if enabled
            if self.notification_enabled:
                self._send_rotation_notification(results)

            return results

    def check_rotation_needed(self) -> bool:
        """
        Check if key rotation is needed.

        Returns:
            True if rotation is needed, False otherwise
        """
        if not self.metadata["next_rotation"]:
            return True

        next_rotation = datetime.fromisoformat(self.metadata["next_rotation"])
        return datetime.utcnow() >= next_rotation

    def get_key_status(self) -> Dict:
        """
        Get current key status.

        Returns:
            Dictionary with key status information
        """
        with self._lock:
            status = {
                "last_rotation": self.metadata["last_rotation"],
                "next_rotation": self.metadata["next_rotation"],
                "rotation_interval_days": self.rotation_interval_days,
                "keys": {},
            }

            for key_type in ["pickle_key", "backup_key"]:
                key_info = self.metadata[f"{key_type}_key"]
                status["keys"][key_type] = {
                    "fingerprint": key_info["current"]["fingerprint"],
                    "created_at": key_info["current"]["created_at"],
                    "rotation_count": len(key_info["rotation_history"]),
                }

            return status

    def _send_rotation_notification(self, rotation_results: Dict) -> None:
        """Send notification about key rotation."""
        # In production, this could send email, Slack message, etc.
        notification = {
            "timestamp": datetime.utcnow().isoformat(),
            "rotations": rotation_results,
        }

        # Log notification
        self.logger.info(
            f"Key rotation notification: {json.dumps(notification, indent=2)}"
        )

    def force_rotation(self, key_type: Optional[str] = None) -> Dict:
        """
        Force rotation of one or all keys.

        Args:
            key_type: Specific key to rotate, or None for all keys

        Returns:
            Dictionary mapping key types to (old_fingerprint, new_fingerprint)
        """
        if key_type:
            return {key_type: self._rotate_key(key_type)}
        else:
            return self.rotate_all_keys()


# Global instance
_key_rotation_manager: Optional[KeyRotationManager] = None


def get_key_rotation_manager() -> KeyRotationManager:
    """Get or create global key rotation manager instance."""
    global _key_rotation_manager
    if _key_rotation_manager is None:
        _key_rotation_manager = KeyRotationManager()
    return _key_rotation_manager


def check_and_rotate_keys_if_needed() -> Optional[Dict]:
    """
    Check if rotation is needed and rotate if so.

    Returns:
        Rotation results if rotation occurred, None otherwise
    """
    manager = get_key_rotation_manager()

    if manager.check_rotation_needed():
        return manager.rotate_all_keys()

    return None


def get_key_status() -> Dict:
    """Get current key status."""
    manager = get_key_rotation_manager()
    return manager.get_key_status()
