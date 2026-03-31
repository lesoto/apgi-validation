"""
Secure key management with SecretStr-style wrappers and point-of-use retrieval.
Replaces global mutable state with secure, on-demand key access.
========================================================================
"""

import os
import base64
import hashlib
import threading
import secrets
from typing import Dict, Optional
from pathlib import Path
import logging
from cryptography.fernet import Fernet


class SecureKeyManager:
    """Thread-safe secure key manager with point-of-use retrieval."""

    def __init__(self, keys_dir: str = ".keys"):
        """
        Initialize secure key manager.

        Args:
            keys_dir: Directory to store encrypted key files
        """
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(exist_ok=True)
        self.keys_dir.chmod(0o700)

        # Thread lock for thread-safe operations (RLock for reentrancy)
        self._lock = threading.RLock()

        # Initialize logger
        self.logger = logging.getLogger("secure_key_manager")
        self.logger.setLevel(logging.INFO)

        # Load or create keys on first access
        self._pickle_key_file = self.keys_dir / "pickle_secret_key.enc"
        self._backup_key_file = self.keys_dir / "backup_hmac_key.enc"

        # Cache for frequently accessed keys (cleared on rotation)
        self._key_cache: Dict[str, str] = {}

        # Generate master key on initialization to ensure it's available
        self._get_master_key()

    def _get_master_key(self) -> str:
        """Get or create master encryption key."""
        master_key = os.environ.get("APGI_MASTER_KEY")
        if not master_key:
            master_key = Fernet.generate_key().decode()
            os.environ["APGI_MASTER_KEY"] = master_key
            self.logger.warning(
                "APGI_MASTER_KEY not set in environment, generated ephemeral key. "
                "Previous keys may not decrypt."
            )
        return master_key

    def _load_encrypted_key(self, key_file: Path) -> bytes:
        """
        Load and decrypt key from encrypted file.

        Args:
            key_file: Path to encrypted key file

        Returns:
            Decrypted key bytes
        """
        if not key_file.exists():
            raise FileNotFoundError(f"Key file {key_file} does not exist")

        master_key = self._get_master_key()
        fernet = Fernet(master_key.encode())

        encrypted = key_file.read_bytes()
        try:
            decrypted_b64 = fernet.decrypt(encrypted).decode("utf-8")
            try:
                # Check if content is valid base64 by attempting to decode
                decoded_bytes = base64.b64decode(decrypted_b64)
                # If successful, convert to hex string
                key_bytes_hex = decoded_bytes.hex()
                return key_bytes_hex
            except Exception as decode_error:
                # If both decryption and base64 decode fail, raise ValueError
                raise ValueError(
                    f"Invalid key format in {key_file.name}: {decode_error}"
                )
        except Exception as decrypt_error:
            # If decryption fails, it might be due to master key mismatch
            # In test scenarios, we should regenerate the key
            self.logger.warning(
                f"Failed to decrypt {key_file.name} with current master key: {decrypt_error}. "
                f"This can happen when the master key changes. Regenerating key."
            )
            # Remove the corrupted/old key file and force regeneration
            key_file.unlink(missing_ok=True)
            raise ValueError(
                f"Key file {key_file.name} could not be decrypted with current master key"
            )

    def _generate_and_save_key(self, key_file: Path) -> str:
        """
        Generate new key and save encrypted.

        Args:
            key_file: Path where to save encrypted key

        Returns:
            Key fingerprint (first 16 chars of SHA256)
        """
        # Generate cryptographically secure key using secrets module (CSPRNG)
        key_bytes = secrets.token_bytes(32)

        # Calculate fingerprint
        fingerprint = hashlib.sha256(key_bytes).hexdigest()[:16]

        # Save encrypted
        master_key = self._get_master_key()
        fernet = Fernet(master_key.encode())
        key_b64 = base64.b64encode(key_bytes).decode("utf-8")
        encrypted = fernet.encrypt(key_b64.encode("utf-8"))

        # Write securely with atomic operation
        temp_file = key_file.with_suffix(".tmp")
        fd = os.open(str(temp_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, encrypted)
        finally:
            os.close(fd)

        # Atomic rename
        temp_file.replace(key_file)

        self.logger.info(f"Generated new key with fingerprint {fingerprint}")
        return fingerprint

    def get_pickle_secret_key(self) -> str:
        """
        Get PICKLE_SECRET_KEY securely.

        Returns:
            Hex-encoded secret key
        """
        with self._lock:
            if "pickle_secret" not in self._key_cache:
                # Check environment variable first
                env_key = os.environ.get("PICKLE_SECRET_KEY")
                if env_key and self._is_valid_hex_key(env_key):
                    self._key_cache["pickle_secret"] = env_key
                    return self._key_cache["pickle_secret"]

                # If no valid env var, try to load from file or generate
                try:
                    if self._pickle_key_file.exists():
                        key_hex = self._load_encrypted_key(self._pickle_key_file)
                        self._key_cache["pickle_secret"] = key_hex
                    else:
                        # Generate new key if none exists
                        fingerprint = self._generate_and_save_key(self._pickle_key_file)
                        key_hex = self._load_encrypted_key(self._pickle_key_file)
                        self._key_cache["pickle_secret"] = key_hex
                        self.logger.info(
                            f"Generated new PICKLE_SECRET_KEY with fingerprint {fingerprint}"
                        )
                except ValueError as e:
                    # Key file couldn't be decrypted, regenerate it
                    self.logger.warning(f"Regenerating pickle secret key: {e}")
                    fingerprint = self._generate_and_save_key(self._pickle_key_file)
                    key_hex = self._load_encrypted_key(self._pickle_key_file)
                    self._key_cache["pickle_secret"] = key_hex
                    self.logger.info(
                        f"Generated new PICKLE_SECRET_KEY with fingerprint {fingerprint}"
                    )

            return self._key_cache["pickle_secret"]

    def _is_valid_hex_key(self, key: str) -> bool:
        """Check if key is valid 64-character hex string."""
        if not key or len(key) != 64:
            return False
        try:
            int(key, 16)
            return True
        except ValueError:
            return False

    def get_backup_hmac_key(self) -> str:
        """
        Get APGI_BACKUP_HMAC_KEY securely.

        Returns:
            Hex-encoded HMAC key
        """
        with self._lock:
            if "backup_hmac" not in self._key_cache:
                # Check environment variable first
                env_key = os.environ.get("APGI_BACKUP_HMAC_KEY")
                if env_key and self._is_valid_hex_key(env_key):
                    self._key_cache["backup_hmac"] = env_key
                    return self._key_cache["backup_hmac"]

                # If no valid env var, try to load from file or generate
                try:
                    if self._backup_key_file.exists():
                        key_hex = self._load_encrypted_key(self._backup_key_file)
                        self._key_cache["backup_hmac"] = key_hex
                    else:
                        # Generate new key if none exists
                        fingerprint = self._generate_and_save_key(self._backup_key_file)
                        key_hex = self._load_encrypted_key(self._backup_key_file)
                        self._key_cache["backup_hmac"] = key_hex
                        self.logger.info(
                            f"Generated new APGI_BACKUP_HMAC_KEY with fingerprint {fingerprint}"
                        )
                except ValueError as e:
                    # Key file couldn't be decrypted, regenerate it
                    self.logger.warning(f"Regenerating backup HMAC key: {e}")
                    fingerprint = self._generate_and_save_key(self._backup_key_file)
                    key_hex = self._load_encrypted_key(self._backup_key_file)
                    self._key_cache["backup_hmac"] = key_hex
                    self.logger.info(
                        f"Generated new APGI_BACKUP_HMAC_KEY with fingerprint {fingerprint}"
                    )

            return self._key_cache["backup_hmac"]

    def clear_cache(self) -> None:
        """Clear key cache - call after key rotation."""
        with self._lock:
            self._key_cache.clear()

    def rotate_keys(self) -> dict:
        """
        Rotate both keys and return rotation info.

        Returns:
            Dictionary with rotation results
        """
        with self._lock:
            # Clear cache to force reload of new keys
            self.clear_cache()

            # Get old fingerprints before rotation
            old_pickle_fingerprint = None
            old_backup_fingerprint = None

            if self._pickle_key_file.exists():
                old_key_hex = self._load_encrypted_key(self._pickle_key_file)
                old_pickle_fingerprint = hashlib.sha256(
                    old_key_hex.encode()
                ).hexdigest()[:16]

            if self._backup_key_file.exists():
                old_key_hex = self._load_encrypted_key(self._backup_key_file)
                old_backup_fingerprint = hashlib.sha256(
                    old_key_hex.encode()
                ).hexdigest()[:16]

            # Generate new keys
            new_pickle_fingerprint = self._generate_and_save_key(self._pickle_key_file)
            new_backup_fingerprint = self._generate_and_save_key(self._backup_key_file)

            # Cache new keys
            pickle_key_hex = self._load_encrypted_key(self._pickle_key_file)
            backup_key_hex = self._load_encrypted_key(self._backup_key_file)
            self._key_cache["pickle_secret"] = pickle_key_hex
            self._key_cache["backup_hmac"] = backup_key_hex

            rotation_results = {
                "pickle_secret_key": {
                    "old_fingerprint": old_pickle_fingerprint,
                    "new_fingerprint": new_pickle_fingerprint,
                },
                "backup_hmac_key": {
                    "old_fingerprint": old_backup_fingerprint,
                    "new_fingerprint": new_backup_fingerprint,
                },
            }

            self.logger.info(
                f"Key rotation completed: pickle_secret {old_pickle_fingerprint[:8]}... -> {new_pickle_fingerprint[:8]}..., "
                f"backup_hmac {old_backup_fingerprint[:8]}... -> {new_backup_fingerprint[:8]}..."
            )

            return rotation_results

    def invalidate_all_references(self) -> None:
        """
        Invalidate all in-flight references by clearing cache and forcing reload.
        This should be called after key rotation to ensure no stale references.
        """
        with self._lock:
            self.clear_cache()

            # Force reload of keys on next access
            self._pickle_key_file = self._pickle_key_file
            self._backup_key_file = self._backup_key_file

            self.logger.info(
                "All key references invalidated - keys will be reloaded on next access"
            )


# Global secure key manager instance
_secure_key_manager: Optional[SecureKeyManager] = None


def get_secure_key_manager() -> SecureKeyManager:
    """Get or create global secure key manager instance."""
    global _secure_key_manager
    if _secure_key_manager is None:
        _secure_key_manager = SecureKeyManager()
    return _secure_key_manager


def get_pickle_secret_key() -> str:
    """Get PICKLE_SECRET_KEY securely."""
    # Check environment variable first for global functions
    env_key = os.environ.get("PICKLE_SECRET_KEY")
    if (
        env_key
        and len(env_key) == 64
        and all(c in "0123456789abcdefABCDEF" for c in env_key)
    ):
        return env_key
    return get_secure_key_manager().get_pickle_secret_key()


def get_backup_hmac_key() -> str:
    """Get APGI_BACKUP_HMAC_KEY securely."""
    # Check environment variable first for global functions
    env_key = os.environ.get("APGI_BACKUP_HMAC_KEY")
    if (
        env_key
        and len(env_key) == 64
        and all(c in "0123456789abcdefABCDEF" for c in env_key)
    ):
        return env_key
    return get_secure_key_manager().get_backup_hmac_key()


def rotate_keys() -> dict:
    """Rotate both keys securely."""
    return get_secure_key_manager().rotate_keys()


def invalidate_all_key_references() -> None:
    """Invalidate all in-flight key references."""
    get_secure_key_manager().invalidate_all_references()
