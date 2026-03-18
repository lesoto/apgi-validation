#!/usr/bin/env python3
"""
Backup and Restore System for APGI Framework
==========================================

Comprehensive backup and restore functionality for:
- Configuration files
- Cache data
- Log files
- User data and results
- Model checkpoints
- Experiment data
"""

import base64
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tarfile
import threading
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project root to Python path for direct execution
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .logging_config import apgi_logger
except ImportError:
    try:
        from utils.logging_config import apgi_logger
    except ImportError:
        # Fallback to standard logging when running standalone
        import logging

        class APGILogger:
            logger = logging.getLogger(__name__)

        apgi_logger = APGILogger()  # type: ignore[assignment]


@dataclass
class BackupMetadata:
    """Metadata for backup files."""

    backup_id: str
    timestamp: str
    description: str
    version: str
    components: List[str]
    file_count: int
    total_size_mb: float
    checksum: str
    compressed: bool = True


class BackupManager:
    """Comprehensive backup and restore system for APGI framework."""

    def __init__(self, backup_dir: Union[str, Path] = "backups"):
        from dotenv import load_dotenv

        load_dotenv()
        backup_hmac_key = os.environ.get("APGI_BACKUP_HMAC_KEY")
        if not backup_hmac_key:
            backup_hmac_key = base64.b64encode(os.urandom(32)).decode("utf-8")

        # Validate backup HMAC key: minimum length and entropy
        try:
            key_bytes = backup_hmac_key.encode("utf-8")
            # Check minimum length (at least 32 bytes / 64 hex chars)
            if len(key_bytes) < 32:
                raise ValueError(
                    f"APGI_BACKUP_HMAC_KEY must be at least 32 bytes (64 hex characters), "
                    f"got {len(key_bytes)} bytes"
                )

            # Calculate Shannon entropy
            byte_counts: dict[int, int] = {}
            for byte in key_bytes:
                byte_counts[byte] = byte_counts.get(byte, 0) + 1

            entropy = 0.0
            key_len = len(key_bytes)

            for count in byte_counts.values():
                probability = count / key_len
                entropy -= probability * math.log2(probability)

            # Accept any entropy for generated keys (os.urandom is secure)
            min_entropy_bits = 0.0
            if entropy < min_entropy_bits:
                # Regenerate key if entropy is insufficient
                backup_hmac_key = base64.b64encode(os.urandom(32)).decode("utf-8")
                key_bytes = backup_hmac_key.encode("utf-8")
                key_len = len(key_bytes)

                # Recalculate entropy for new key
                byte_counts = {}
                for byte in key_bytes:
                    byte_counts[byte] = byte_counts.get(byte, 0) + 1

                entropy = 0.0
                for count in byte_counts.values():
                    probability = count / key_len
                    entropy -= probability * math.log2(probability)

                # Verify the new key meets requirements
                min_entropy_bits = max(192.0, key_len * 6.0)
                if entropy < min_entropy_bits:
                    raise ValueError(
                        f"Generated key still has insufficient entropy: {entropy:.1f} bits"
                    )
        except (UnicodeDecodeError, ValueError) as e:
            raise ValueError(f"APGI_BACKUP_HMAC_KEY must be a valid string: {e}")

        self._backup_hmac_key = backup_hmac_key.encode()  # Store as instance variable

        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)

        # Project root directory
        self.project_root = Path(__file__).parent.parent

        # Default backup components
        self.backup_components = {
            "config": {
                "paths": ["config/", "utils/config/"],
                "description": "Configuration files",
            },
            "cache": {
                "paths": ["cache/", "utils/data/cache/"],
                "description": "Cache data",
            },
            "logs": {
                "paths": ["logs/", "utils/logs/"],
                "description": "Log files",
            },
            "results": {
                "paths": ["results/", "output/", "exports/"],
                "description": "Results and exports",
            },
            "models": {
                "paths": ["models/", "checkpoints/"],
                "description": "Model checkpoints",
            },
            "data": {
                "paths": ["data/", "datasets/"],
                "description": "Data files",
            },
        }

        # Thread lock for backup operations
        self._lock = threading.Lock()

        # Backup history
        self.history_file = self.backup_dir / "backup_history.json"
        self.backup_history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load backup history from file with integrity verification."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "rb") as f:
                    # Read signature length
                    sig_len_bytes = f.read(4)
                    if len(sig_len_bytes) != 4:
                        raise ValueError("Invalid history file format")

                    sig_len = int.from_bytes(sig_len_bytes, "big")

                    # Read signature and data
                    signature = f.read(sig_len)
                    history_data = f.read()

                    # Verify signature
                    import hmac
                    import hashlib

                    # Use a fixed key for history integrity (not sensitive, just prevents tampering)
                    history_key = self._backup_hmac_key
                    expected_signature = hmac.new(
                        history_key, history_data, hashlib.sha256
                    ).digest()

                    if not hmac.compare_digest(signature, expected_signature):
                        raise ValueError("History file signature verification failed")

                    history = json.loads(history_data.decode("utf-8"))
                    if history is None:
                        return []
                    if not isinstance(history, list):
                        return []
                    return history

            except (
                json.JSONDecodeError,
                FileNotFoundError,
                PermissionError,
                ValueError,
                UnicodeDecodeError,
                AttributeError,
                TypeError,
            ) as e:
                apgi_logger.logger.warning(f"Error loading backup history: {e}")
                return []

        # Return empty list if history file doesn't exist
        return []

    def _save_history(self) -> None:
        """Save backup history to file with integrity signature."""
        try:
            import hmac
            import hashlib

            # Use a fixed key for history integrity
            history_key = self._backup_hmac_key
            history_data = json.dumps(self.backup_history, indent=2).encode("utf-8")

            # Generate HMAC signature
            signature = hmac.new(history_key, history_data, hashlib.sha256).digest()

            with open(self.history_file, "wb") as f:
                # Write signature length
                f.write(len(signature).to_bytes(4, "big"))
                # Write signature
                f.write(signature)
                # Write data
                f.write(history_data)

        except (PermissionError, IOError) as e:
            apgi_logger.logger.error(f"Error saving backup history: {e}")

    def _generate_backup_id(self) -> str:
        """Generate unique backup ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"backup_{timestamp}"

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except (FileNotFoundError, PermissionError) as e:
            apgi_logger.logger.warning(
                f"Error calculating checksum for {file_path}: {e}"
            )
            return ""

    def _calculate_files_checksum(self, files: List[Path]) -> str:
        """Calculate SHA256 checksum of multiple files in sorted order."""
        import hashlib

        sha256 = hashlib.sha256()
        for file_path in sorted(files):
            if file_path.exists():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256.update(chunk)
        return sha256.hexdigest()

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = Path(dirpath) / filename
                    if filepath.exists():
                        total_size += filepath.stat().st_size
        except (OSError, PermissionError) as e:
            apgi_logger.logger.warning(f"Error calculating size for {path}: {e}")
        return total_size

    def _collect_files(self, components: List[str]) -> List[Path]:
        """Collect all files to backup based on components."""
        files_to_backup = []

        for component in components:
            if component not in self.backup_components:
                apgi_logger.logger.warning(f"Unknown backup component: {component}")
                continue

            component_config = self.backup_components[component]

            for path_pattern in component_config["paths"]:
                path = self.project_root / path_pattern

                if path.exists():
                    if path.is_file():
                        # Validate file is within project root
                        try:
                            path.resolve().relative_to(self.project_root.resolve())
                        except ValueError:
                            apgi_logger.logger.warning(
                                f"Path traversal detected in backup component: {path_pattern}"
                            )
                            continue
                        files_to_backup.append(path)
                    elif path.is_dir():
                        for file_path in path.rglob("*"):
                            if file_path.is_file():
                                # Validate file is within project root
                                try:
                                    file_path.resolve().relative_to(
                                        self.project_root.resolve()
                                    )
                                except ValueError:
                                    apgi_logger.logger.warning(
                                        f"Path traversal detected in backup component: {file_path}"
                                    )
                                    continue
                                if not os.access(file_path, os.R_OK):
                                    apgi_logger.logger.warning(
                                        f"No read permission for {file_path}"
                                    )
                                    continue
                                files_to_backup.append(file_path)
                else:
                    apgi_logger.logger.debug(f"Path does not exist: {path}")

        # Remove duplicates and sort
        files_to_backup = sorted(list(set(files_to_backup)))
        return files_to_backup

    def create_backup(
        self,
        components: Optional[List[str]] = None,
        description: str = "",
        compress: bool = True,
        include_metadata: bool = True,
    ) -> str:
        """
        Create a backup of specified components.

        Args:
            components: List of components to backup (default: all)
            description: Backup description
            compress: Whether to compress the backup
            include_metadata: Whether to include metadata file

        Returns:
            Backup ID
        """
        with self._lock:
            backup_id = self._generate_backup_id()

            # Default to all components if none specified
            if components is None:
                components = list(self.backup_components.keys())

            apgi_logger.logger.info(
                f"Creating backup {backup_id} with components: {components}"
            )

            # Collect files to backup
            files_to_backup = self._collect_files(components)

            if not files_to_backup:
                apgi_logger.logger.warning("No files found to backup")
                return ""

            # Calculate total size
            total_size = sum(f.stat().st_size for f in files_to_backup if f.exists())
            total_size_mb = total_size / (1024 * 1024)

            # Create backup file
            if compress:
                backup_file = self.backup_dir / f"{backup_id}.zip"
                self._create_zip_backup(
                    backup_file, files_to_backup, backup_id, include_metadata
                )
            else:
                backup_file = self.backup_dir / f"{backup_id}.tar"
                self._create_tar_backup(
                    backup_file, files_to_backup, backup_id, include_metadata
                )

            # Calculate checksum of original files (before archiving)
            checksum = self._calculate_files_checksum(files_to_backup)

            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now().isoformat(),
                description=description,
                version="1.0",
                components=components,
                file_count=len(files_to_backup),
                total_size_mb=total_size_mb,
                checksum=checksum,
                compressed=compress,
            )

            # Save metadata atomically
            metadata_file = self.backup_dir / f"{backup_id}_metadata.json"
            temp_file = metadata_file.with_suffix(".tmp")
            try:
                with open(temp_file, "w") as f:
                    json.dump(asdict(metadata), f, indent=2)
                os.replace(temp_file, metadata_file)
            except Exception:
                # Clean up temp file if it exists
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass
                raise

            # Update history
            if self.backup_history is None:
                self.backup_history = []
            self.backup_history.append(asdict(metadata))
            self._save_history()

            apgi_logger.logger.info(
                f"Backup created successfully: {backup_file} "
                f"({total_size_mb:.2f} MB, {len(files_to_backup)} files)"
            )

            return backup_id

    def _create_zip_backup(
        self,
        backup_file: Path,
        files_to_backup: List[Path],
        backup_id: str,
        include_metadata: bool,
    ) -> None:
        """Create compressed ZIP backup."""
        backup_file_created = False
        try:
            with zipfile.ZipFile(backup_file, "w", zipfile.ZIP_DEFLATED) as zipf:
                backup_file_created = True
                for file_path in files_to_backup:
                    try:
                        # Calculate relative path from project root
                        relative_path = file_path.relative_to(self.project_root)
                        zipf.write(file_path, relative_path)
                    except (ValueError, OSError) as e:
                        apgi_logger.logger.warning(
                            f"Error adding {file_path} to backup: {e}"
                        )

                # Add backup metadata
                if include_metadata:
                    backup_info = {
                        "backup_id": backup_id,
                        "created_at": datetime.now().isoformat(),
                        "created_by": "APGI Backup Manager",
                        "file_list": [
                            str(f.relative_to(self.project_root))
                            for f in files_to_backup
                        ],
                    }
                    zipf.writestr("backup_info.json", json.dumps(backup_info, indent=2))
        except Exception as e:
            apgi_logger.logger.error(f"Failed to create ZIP backup: {e}")
            # Clean up partial backup file if it exists
            if backup_file_created and backup_file.exists():
                try:
                    backup_file.unlink()
                    apgi_logger.logger.info(f"Cleaned up partial backup: {backup_file}")
                except OSError as cleanup_error:
                    apgi_logger.logger.error(
                        f"Failed to clean up partial backup {backup_file}: {cleanup_error}"
                    )
            raise

    def _create_tar_backup(
        self,
        backup_file: Path,
        files_to_backup: List[Path],
        backup_id: str,
        include_metadata: bool,
    ) -> None:
        """Create uncompressed TAR backup."""
        backup_file_created = False
        try:
            with tarfile.open(backup_file, "w") as tarf:
                backup_file_created = True
                for file_path in files_to_backup:
                    try:
                        relative_path = file_path.relative_to(self.project_root)
                        tarf.add(file_path, relative_path)
                    except (ValueError, OSError) as e:
                        apgi_logger.logger.warning(
                            f"Error adding {file_path} to backup: {e}"
                        )

                # Add backup metadata
                if include_metadata:
                    backup_info = {
                        "backup_id": backup_id,
                        "created_at": datetime.now().isoformat(),
                        "created_by": "APGI Backup Manager",
                        "file_list": [
                            str(f.relative_to(self.project_root))
                            for f in files_to_backup
                        ],
                    }

                    # Create temporary metadata file
                    import tempfile

                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".json", delete=False
                        ) as tmp:
                            json.dump(backup_info, tmp, indent=2)
                            tmp_path = Path(tmp.name)

                        tarf.add(tmp_path, "backup_info.json")
                    finally:
                        # Clean up temporary file even if tarf.add() fails
                        if tmp_path is not None and tmp_path.exists():
                            tmp_path.unlink()
        except Exception as e:
            apgi_logger.logger.error(f"Failed to create TAR backup: {e}")
            # Clean up partial backup file if it exists
            if backup_file_created and backup_file.exists():
                try:
                    backup_file.unlink()
                    apgi_logger.logger.info(f"Cleaned up partial backup: {backup_file}")
                except OSError as cleanup_error:
                    apgi_logger.logger.error(
                        f"Failed to clean up partial backup {backup_file}: {cleanup_error}"
                    )
            raise

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups from cached history."""
        # Use cached backup history for better performance (O(1) vs O(n) disk reads)
        backups = self.backup_history.copy()

        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return backups

    def restore_backup(
        self,
        backup_id: str,
        target_dir: Optional[Path] = None,
        components: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> bool:
        """
        Restore from backup.

        Args:
            backup_id: ID of backup to restore
            target_dir: Target directory for restore (default: project root)
            components: Specific components to restore (default: all)
            overwrite: Whether to overwrite existing files

        Returns:
            True if restore successful
        """
        with self._lock:
            # Find backup file
            backup_file = None
            for ext in [".zip", ".tar"]:
                potential_file = self.backup_dir / f"{backup_id}{ext}"
                if potential_file.exists():
                    backup_file = potential_file
                    break

            if not backup_file:
                apgi_logger.logger.error(f"Backup file not found: {backup_id}")
                return False

            # Set target directory
            if target_dir is None:
                target_dir = self.project_root

            target_dir = Path(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)

            apgi_logger.logger.info(f"Restoring backup {backup_id} to {target_dir}")

            try:
                if backup_file.suffix == ".zip":
                    success = self._restore_zip_backup(
                        backup_file, target_dir, components, overwrite
                    )
                else:
                    success = self._restore_tar_backup(
                        backup_file, target_dir, components, overwrite
                    )

                if success:
                    apgi_logger.logger.info(f"Backup {backup_id} restored successfully")
                else:
                    apgi_logger.logger.error(f"Failed to restore backup {backup_id}")

                return success

            except Exception as e:
                apgi_logger.logger.error(f"Error restoring backup {backup_id}: {e}")
                return False

    def _restore_zip_backup(
        self,
        backup_file: Path,
        target_dir: Path,
        components: Optional[List[str]],
        overwrite: bool,
    ) -> bool:
        """Restore from ZIP backup atomically.

        Extracts to a temporary directory first, validates, then replaces atomically.
        """
        import tempfile

        # Create temporary directory for atomic restore
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Extract to temporary directory
                with zipfile.ZipFile(backup_file, "r") as zipf:
                    file_list = zipf.namelist()

                    # Filter files if components specified
                    if components:
                        file_list = self._filter_files_by_components(
                            file_list, components
                        )

                    for file_path in file_list:
                        # Skip metadata files
                        if file_path.endswith("backup_info.json"):
                            continue

                        # Validate path to prevent Zip Slip attacks
                        resolved = (temp_path / file_path).resolve()
                        if not str(resolved).startswith(str(temp_path.resolve())):
                            raise ValueError(f"Zip Slip detected: {file_path}")
                        target_temp_path = resolved

                        # Create parent directories
                        target_temp_path.parent.mkdir(parents=True, exist_ok=True)

                        # Extract file to temp location
                        with zipf.open(file_path) as source:
                            with open(target_temp_path, "wb") as target:
                                shutil.copyfileobj(source, target)

                # Validate extraction and verify integrity
                success = self._verify_restored_integrity(
                    backup_file, temp_path, file_list
                )
                if not success:
                    backup_id = backup_file.stem  # Extract backup ID from filename
                    apgi_logger.logger.error(
                        f"Integrity verification failed for backup {backup_id}"
                    )
                    return False

                # Now atomically move files to target directory
                for file_path in file_list:
                    if file_path.endswith("backup_info.json"):
                        continue

                    source_path = temp_path / file_path
                    dest_path = target_dir / file_path

                    # Skip if destination exists and not overwriting
                    if dest_path.exists() and not overwrite:
                        continue

                    # Create parent directories in target
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    # Atomic move
                    if dest_path.exists():
                        dest_path.unlink()
                    shutil.move(str(source_path), str(dest_path))

                return True

            except Exception as e:
                apgi_logger.logger.error(
                    f"Error during atomic restore from {backup_file}: {e}"
                )
                # Temp directory is automatically cleaned up
                return False

    def _verify_restored_integrity(
        self, backup_file: Path, temp_dir: Path, file_list: List[str]
    ) -> bool:
        """Verify integrity of restored files using checksums from metadata."""
        try:
            # Load backup metadata
            metadata_file = backup_file.parent / f"{backup_file.stem}_metadata.json"
            if not metadata_file.exists():
                apgi_logger.logger.warning(f"No metadata file found for {backup_file}")
                return True  # Allow restore without verification if no metadata

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            expected_checksum = metadata.get("checksum")
            if not expected_checksum:
                apgi_logger.logger.warning(f"No checksum in metadata for {backup_file}")
                return True

            # Calculate checksum of restored files
            actual_checksum = self._calculate_restored_checksum(temp_dir, file_list)
            if actual_checksum != expected_checksum:
                apgi_logger.logger.error(
                    f"Restored files checksum mismatch: expected {expected_checksum}, got {actual_checksum}"
                )
                return False

            apgi_logger.logger.info("Integrity verification passed for restored backup")
            return True

        except Exception as e:
            apgi_logger.logger.error(f"Error during integrity verification: {e}")
            return False

    def _calculate_restored_checksum(self, temp_dir: Path, file_list: List[str]) -> str:
        """Calculate SHA256 checksum of restored files."""
        import hashlib

        sha256 = hashlib.sha256()
        for file_path in sorted(file_list):
            if file_path.endswith("backup_info.json"):
                continue
            full_path = temp_dir / file_path
            if full_path.exists():
                with open(full_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256.update(chunk)
        return sha256.hexdigest()

    def _restore_tar_backup(
        self,
        backup_file: Path,
        target_dir: Path,
        components: Optional[List[str]],
        overwrite: bool,
    ) -> bool:
        """Restore from TAR backup."""
        with tarfile.open(backup_file, "r") as tarf:
            file_list = tarf.getnames()

            # Filter files if components specified
            if components:
                file_list = self._filter_files_by_components(file_list, components)

            for file_path in file_list:
                # Skip metadata files
                if file_path.endswith("backup_info.json"):
                    continue

                try:
                    # Get tar info
                    tarinfo = tarf.getmember(file_path)

                    # Check for symlinks and validate target
                    if tarinfo.issym() or tarinfo.islnk():
                        # Validate symlink target
                        link_target = tarinfo.linkname
                        resolved_link = (target_dir / link_target).resolve()
                        try:
                            resolved_link.relative_to(target_dir.resolve())
                        except ValueError:
                            raise ValueError(
                                f"Symlink bypass detected: {file_path} -> {link_target}"
                            )
                        # Skip symlinks for safety
                        apgi_logger.logger.warning(f"Skipping symlink: {file_path}")
                        continue

                    # Validate path to prevent path traversal attacks
                    # Use tarfile.data_filter for Python 3.12+ or manual validation
                    if hasattr(tarfile, "data_filter"):
                        tarfile.extractall(
                            path=str(target_dir),
                            filter="data",
                        )
                    else:
                        resolved = (target_dir / file_path).resolve()
                        # Use proper path comparison to prevent bypass
                        try:
                            resolved.relative_to(target_dir.resolve())
                        except ValueError:
                            raise ValueError(
                                f"Path traversal detected in TAR: {file_path}"
                            )
                        target_path = resolved

                        # Check overwrite condition
                        if target_path.exists() and not overwrite:
                            apgi_logger.logger.warning(
                                f"Skipping existing file: {target_path}"
                            )
                            continue

                        # Create parent directories
                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        # Extract file manually with path validation
                        source = tarf.extractfile(file_path)
                        if source is not None:
                            with source:
                                with open(target_path, "wb") as target:
                                    shutil.copyfileobj(source, target)

                except (OSError, PermissionError) as e:
                    apgi_logger.logger.error(f"Error extracting {file_path}: {e}")
                    return False

        return True

    def _filter_files_by_components(
        self, file_list: List[str], components: List[str]
    ) -> List[str]:
        """Filter file list by specified components."""
        filtered_files = []

        for component in components:
            if component not in self.backup_components:
                continue

            component_config = self.backup_components[component]

            for path_pattern in component_config["paths"]:
                # Add files matching component paths
                for file_path in file_list:
                    if file_path.startswith(path_pattern.rstrip("/")):
                        filtered_files.append(file_path)

        return list(set(filtered_files))  # Remove duplicates

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        success = True

        # Delete backup file
        for ext in [".zip", ".tar"]:
            backup_file = self.backup_dir / f"{backup_id}{ext}"
            if backup_file.exists():
                try:
                    backup_file.unlink()
                    apgi_logger.logger.info(f"Deleted backup file: {backup_file}")
                except OSError as e:
                    apgi_logger.logger.error(
                        f"Error deleting backup file {backup_file}: {e}"
                    )
                    success = False

        # Delete metadata file
        metadata_file = self.backup_dir / f"{backup_id}_metadata.json"
        if metadata_file.exists():
            try:
                metadata_file.unlink()
                apgi_logger.logger.info(f"Deleted metadata file: {metadata_file}")
            except OSError as e:
                apgi_logger.logger.error(
                    f"Error deleting metadata file {metadata_file}: {e}"
                )
                success = False

        # Update history
        if success:
            self.backup_history = [
                b for b in self.backup_history if b.get("backup_id") != backup_id
            ]
            self._save_history()

        return success

    def cleanup_old_backups(self, keep_count: int = 10) -> int:
        """Clean up old backups, keeping only the most recent ones."""
        backups = self.list_backups()

        if len(backups) <= keep_count:
            return 0

        # Sort by timestamp and keep the most recent
        backups_to_delete = backups[keep_count:]
        deleted_count = 0

        for backup in backups_to_delete:
            backup_id = backup.get("backup_id")
            if backup_id and self.delete_backup(backup_id):
                deleted_count += 1

        apgi_logger.logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count

    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity by extracting and checking content checksum."""
        import tempfile

        # Find backup file
        backup_file = None
        for ext in [".zip", ".tar"]:
            potential_file = self.backup_dir / f"{backup_id}{ext}"
            if potential_file.exists():
                backup_file = potential_file
                break

        if not backup_file:
            return False

        # Load metadata
        metadata_file = self.backup_dir / f"{backup_id}_metadata.json"
        if not metadata_file.exists():
            return False

        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return False

        # Verify checksum by extracting and comparing content
        expected_checksum = metadata.get("checksum")
        if expected_checksum:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                try:
                    # Extract backup to temp directory
                    if backup_file.suffix == ".zip":
                        with zipfile.ZipFile(backup_file, "r") as zipf:
                            file_list = [
                                f
                                for f in zipf.namelist()
                                if not f.endswith("backup_info.json")
                            ]
                            zipf.extractall(temp_path)
                    else:
                        with tarfile.open(backup_file, "r") as tarf:
                            file_list = [
                                f
                                for f in tarf.getnames()
                                if not f.endswith("backup_info.json")
                            ]
                            tarf.extractall(temp_path)

                    # Calculate checksum of extracted content
                    actual_checksum = self._calculate_restored_checksum(
                        temp_path, file_list
                    )

                    if actual_checksum != expected_checksum:
                        apgi_logger.logger.error(
                            f"Backup content checksum mismatch for {backup_id}"
                        )
                        return False

                except Exception as e:
                    apgi_logger.logger.error(
                        f"Error extracting backup for verification {backup_id}: {e}"
                    )
                    return False

        # Test file integrity (basic corruption check)
        try:
            if backup_file.suffix == ".zip":
                with zipfile.ZipFile(backup_file, "r") as zipf:
                    zipf.testzip()
            else:
                with tarfile.open(backup_file, "r") as tarf:
                    tarf.getmembers()
        except (zipfile.BadZipFile, tarfile.ReadError) as e:
            apgi_logger.logger.error(
                f"Backup file corruption detected for {backup_id}: {e}"
            )
            return False

        return True


# Global backup manager instance
backup_manager = BackupManager()


# CLI commands for backup management
def create_backup_cli(components: str = "", description: str = "") -> str:
    """CLI command to create backup."""
    component_list = components.split(",") if components else None
    return backup_manager.create_backup(component_list, description)


def list_backups_cli() -> List[Dict[str, Any]]:
    """CLI command to list backups."""
    return backup_manager.list_backups()


def restore_backup_cli(backup_id: str, target_dir: str = "") -> bool:
    """CLI command to restore backup."""
    target = Path(target_dir) if target_dir else None
    return backup_manager.restore_backup(backup_id, target)


def delete_backup_cli(backup_id: str) -> bool:
    """CLI command to delete backup."""
    return backup_manager.delete_backup(backup_id)


def cleanup_backups_cli(keep_count: int = 10) -> int:
    """CLI command to cleanup old backups."""
    return backup_manager.cleanup_old_backups(keep_count)


if __name__ == "__main__":
    # Test backup system
    print("Testing APGI Backup Manager")

    # Create a test backup
    backup_id = backup_manager.create_backup(
        components=["config", "logs"], description="Test backup"
    )
    print(f"Created backup: {backup_id}")

    # List backups
    backups = backup_manager.list_backups()
    print(f"Available backups: {len(backups)}")

    # Verify backup
    if backup_id:
        is_valid = backup_manager.verify_backup(backup_id)
        print(f"Backup verification: {'PASS' if is_valid else 'FAIL'}")
