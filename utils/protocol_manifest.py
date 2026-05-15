"""
Protocol manifest integrity verification.

Loads config/protocol_manifest.json and exposes verify_protocol_file() so
GUI loaders can confirm a file's SHA-256 hash before exec_module() is called.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent.parent / "config" / "protocol_manifest.json"
_manifest: Optional[dict] = None


def _load_manifest() -> dict:
    global _manifest
    if _manifest is None:
        try:
            _manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not load protocol manifest: {e}")
            _manifest = {}
    return _manifest


def _manifest_path() -> Path:
    return _MANIFEST_PATH


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_protocol_file(file_path: Path, directory_key: str) -> bool:
    """Return True if file_path matches its recorded SHA-256 hash.

    Args:
        file_path: Absolute path to the protocol .py file.
        directory_key: One of "Theory", "Validation", or "Falsification".

    Returns:
        True if the file is in the manifest and its hash matches.
        False (with a warning) if missing from manifest or hash mismatch.
        True unconditionally when APGI_SKIP_MANIFEST_CHECK=1 is set (dev override).
    """
    if os.environ.get("APGI_SKIP_MANIFEST_CHECK", "0") == "1":
        logger.debug(f"Manifest check skipped for {file_path.name} (APGI_SKIP_MANIFEST_CHECK=1)")
        return True

    manifest = _load_manifest()
    dir_entries: dict = manifest.get("directories", {}).get(directory_key, {})

    if file_path.name not in dir_entries:
        logger.warning(
            f"SECURITY: {file_path.name} is not in the protocol manifest for {directory_key}/. "
            "File will not be loaded. Add it to config/protocol_manifest.json or set "
            "APGI_SKIP_MANIFEST_CHECK=1 to allow unregistered files during development."
        )
        return False

    expected = dir_entries[file_path.name]
    actual = _sha256(file_path)
    if actual != expected:
        logger.error(
            f"SECURITY: Hash mismatch for {file_path.name} in {directory_key}/. "
            f"Expected {expected[:16]}…, got {actual[:16]}…. "
            "File may have been modified externally and will not be loaded."
        )
        return False

    return True


def update_manifest_entry(file_path: Path, directory_key: str) -> None:
    """Recompute and persist the SHA-256 hash for one file.

    Useful after intentional protocol edits — call this from a dev CLI command
    rather than bypassing the manifest entirely.
    """
    manifest_file = _manifest_path()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception:
        manifest = {"version": "1.0", "directories": {}}

    manifest.setdefault("directories", {}).setdefault(directory_key, {})[file_path.name] = _sha256(file_path)
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    global _manifest
    _manifest = manifest
    logger.info(f"Manifest updated for {directory_key}/{file_path.name}")
