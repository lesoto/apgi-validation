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
from cryptography.fernet import Fernet

try:
    from .secure_key_manager import invalidate_all_key_references
except ImportError:
    try:
        from utils.secure_key_manager import invalidate_all_key_references
    except ImportError:
        # Fallback for standalone execution
        def invalidate_all_key_references():
            """Fallback function when secure_key_manager is not available."""
            pass


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
            with open(self.metadata_file, "w", encoding="utf-8") as f:
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

        # Save encrypted key
        master_key = os.environ.get("APGI_MASTER_KEY")
        if not master_key:
            master_key = Fernet.generate_key().decode()
            os.environ["APGI_MASTER_KEY"] = master_key

        fernet = Fernet(master_key.encode())
        key_b64 = base64.b64encode(key_bytes).decode("utf-8")
        encrypted = fernet.encrypt(key_b64.encode("utf-8"))

        # Write securely bypassing TOCTOU write-then-chmod
        fd = os.open(str(key_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, encrypted)
        finally:
            os.close(fd)

        # Update metadata
        timestamp = datetime.utcnow().isoformat()
        self.metadata[key_type]["current"] = {
            "fingerprint": fingerprint,
            "created_at": timestamp,
            "file": str(key_file),
        }

        self._save_metadata()
        self.logger.info(f"Generated new {key_type} with fingerprint {fingerprint}")

    def _load_key_from_file(self, key_type: str, key_file: Path) -> Path:
        """Load key from file."""
        try:
            master_key = os.environ.get("APGI_MASTER_KEY")
            if not master_key:
                master_key = Fernet.generate_key().decode()
                os.environ["APGI_MASTER_KEY"] = master_key
                self.logger.warning(
                    "APGI_MASTER_KEY not set in environment, generated ephemeral key. Prev keys may not decrypt."
                )

            fernet = Fernet(master_key.encode())
            encrypted = key_file.read_bytes()

            try:
                key_b64 = fernet.decrypt(encrypted).decode("utf-8")
            except Exception:
                # Fallback for old base64-only keys to allow migrating
                self.logger.warning(
                    f"Failed to decrypt {key_type}, falling back to legacy base64 load."
                )
                key_b64 = encrypted.decode("utf-8")

            key_bytes = base64.b64decode(key_b64)
            fingerprint = hashlib.sha256(key_bytes).hexdigest()[:16]

            self.metadata[key_type]["current"] = {
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
        """Set environment variables for current keys.

        .. deprecated::
            Writing cryptographic key material to ``os.environ`` exposes it to
            any child process spawned from this Python process.  Callers should
            read the key from the key file directly using ``_load_key_from_file``
            instead of relying on environment variables.
        """
        import warnings

        warnings.warn(
            "Setting cryptographic keys in os.environ exposes them to child "
            "processes.  Read keys directly from their key files instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Load current keys (decrypt them first)
        pickle_key_file = self.keys_dir / "pickle_secret_key.enc"
        backup_key_file = self.keys_dir / "backup_hmac_key.enc"

        if pickle_key_file.exists():
            try:
                master_key = os.environ.get("APGI_MASTER_KEY")
                if master_key:
                    fernet = Fernet(master_key.encode())
                    encrypted = pickle_key_file.read_bytes()
                    key_b64 = fernet.decrypt(encrypted).decode("utf-8")
                    pickle_key = base64.b64decode(key_b64).hex()
                    os.environ["PICKLE_SECRET_KEY"] = pickle_key
            except Exception as e:
                self.logger.warning(f"Could not set PICKLE_SECRET_KEY env var: {e}")

        if backup_key_file.exists():
            try:
                master_key = os.environ.get("APGI_MASTER_KEY")
                if master_key:
                    fernet = Fernet(master_key.encode())
                    encrypted = backup_key_file.read_bytes()
                    key_b64 = fernet.decrypt(encrypted).decode("utf-8")
                    backup_key = base64.b64decode(key_b64).hex()
                    os.environ["APGI_BACKUP_HMAC_KEY"] = backup_key
            except Exception as e:
                self.logger.warning(f"Could not set APGI_BACKUP_HMAC_KEY env var: {e}")

    def _rotate_key(self, key_type: str) -> Tuple[str, str]:
        """
        Rotate a key and save old key as backup.

        Args:
            key_type: Type of key to rotate ("pickle_key" or "backup_key")

        Returns:
            Tuple of (old_fingerprint, new_fingerprint)
        """
        # Get current key info
        current_key_info = self.metadata[key_type]["current"]
        old_fingerprint = current_key_info["fingerprint"]
        old_file = Path(current_key_info["file"])

        # Generate new key
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        new_key_file = self.keys_dir / f"{key_type}_{timestamp}.enc"
        self._generate_and_save_key(key_type, new_key_file)
        new_key_info = self.metadata[key_type]["current"]
        new_fingerprint = new_key_info["fingerprint"]

        # Add to rotation history
        rotation_entry = {
            "old_fingerprint": old_fingerprint,
            "new_fingerprint": new_fingerprint,
            "rotated_at": datetime.utcnow().isoformat(),
            "old_file": str(old_file),
            "new_file": str(new_key_file),
        }
        self.metadata[key_type]["rotation_history"].append(rotation_entry)

        # Limit rotation history
        if len(self.metadata[key_type]["rotation_history"]) > self.backup_count:
            self.metadata[key_type]["rotation_history"].pop(0)

        # Update last rotation
        self.metadata["last_rotation"] = datetime.utcnow().isoformat()
        self._schedule_next_rotation()
        self._save_metadata()

        # Update environment variables
        self._set_env_vars()

        # Invalidate all in-flight references across the application
        invalidate_all_key_references()

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
                key_info = self.metadata[key_type]
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


if __name__ == "__main__":
    """Test key rotation manager functionality."""
    print("Testing APGI Key Rotation Manager")
    print("=" * 50)

    try:
        # Get key rotation manager
        manager = get_key_rotation_manager()

        # Display current status
        status = get_key_status()
        print(f"Last rotation: {status['last_rotation']}")
        print(f"Next rotation: {status['next_rotation']}")
        print(f"Rotation interval: {status['rotation_interval_days']} days")

        # Display key information
        for key_type, key_info in status["keys"].items():
            print(f"{key_type}:")
            print(f"  Fingerprint: {key_info['fingerprint']}")
            print(f"  Created: {key_info['created_at']}")
            print(f"  Rotations: {key_info['rotation_count']}")

        # Check if rotation is needed
        rotation_needed = manager.check_rotation_needed()
        print(f"Rotation needed: {rotation_needed}")

        print("\nKey rotation manager test completed successfully!")

    except Exception as e:
        print(f"Error testing key rotation manager: {e}")
        import traceback

        traceback.print_exc()
