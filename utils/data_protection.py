"""Data protection workflows: retention, deletion, consent."""

import hashlib
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# PII Patterns for simple identification
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
}


def tag_pii_in_data(data: str) -> Dict[str, list]:
    """Tag potential PII in text data.

    Args:
        data: String content to analyze

    Returns:
        Dictionary of detected PII types and their matches
    """
    detected_pii = {}
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, data)
        if matches:
            detected_pii[pii_type] = list(set(matches))
    return detected_pii


def minimize_data(data: str, redaction_char: str = "*") -> str:
    """Redact PII from data string.

    Args:
        data: String content to redact
        redaction_char: Character to use for redaction

    Returns:
        Redacted string
    """
    minimized_data = data
    for pii_type, pattern in PII_PATTERNS.items():
        # Using a simplistic redaction for demonstration
        def repl(match):
            return redaction_char * len(match.group(0))

        minimized_data = re.sub(pattern, repl, minimized_data)
    return minimized_data


def secure_delete(path: str, passes: int = 3) -> bool:
    """Securely delete a file by overwriting it multiple times.

    Args:
        path: Path to the file to delete
        passes: Number of overwrite passes (default: 3)

    Returns:
        True if deletion succeeded, False otherwise
    """
    file_path = Path(path)

    if not file_path.exists():
        return False

    if not file_path.is_file():
        return False

    try:
        # Overwrite file with random data multiple times
        file_size = file_path.stat().st_size

        with open(file_path, "r+b") as f:
            for _ in range(passes):
                f.seek(0)
                # Generate random data
                random_data = os.urandom(file_size)
                f.write(random_data)
                f.flush()
                os.fsync(f.fileno())

        # Delete the file
        file_path.unlink()
        return True

    except (OSError, PermissionError, IOError) as e:
        print(f"Error during secure delete of {path}: {e}")
        return False


def apply_retention_policy(
    data_dir: str,
    max_age_days: int = 365,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Apply retention policy to delete data older than max_age_days.

    Args:
        data_dir: Directory containing data to check
        max_age_days: Maximum age in days before deletion (default: 365)
        dry_run: If True, only report what would be deleted (default: False)

    Returns:
        Dictionary with retention policy execution results
    """
    data_path = Path(data_dir)

    if not data_path.exists() or not data_path.is_dir():
        return {
            "status": "error",
            "message": f"Directory not found: {data_dir}",
            "deleted_count": 0,
            "deleted_size_bytes": 0,
            "files_to_delete": [],
        }

    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    files_to_delete = []
    total_size = 0

    # Find files older than max_age_days
    for file_path in data_path.rglob("*"):
        if file_path.is_file():
            file_mtime = file_path.stat().st_mtime
            if file_mtime < cutoff_time:
                file_size = file_path.stat().st_size
                files_to_delete.append(
                    {
                        "path": str(file_path),
                        "size_bytes": file_size,
                        "age_days": (time.time() - file_mtime) / (24 * 60 * 60),
                    }
                )
                total_size += file_size

    if dry_run:
        return {
            "status": "dry_run",
            "message": f"Would delete {len(files_to_delete)} files",
            "deleted_count": 0,
            "deleted_size_bytes": 0,
            "files_to_delete": files_to_delete,
        }

    # Actually delete the files
    deleted_count = 0
    deleted_size = 0
    errors = []

    for file_info in files_to_delete:
        file_path = Path(str(file_info["path"]))
        try:
            if secure_delete(str(file_path)):
                deleted_count += 1
                size_bytes = file_info.get("size_bytes", 0)
                if isinstance(size_bytes, (int, float)):
                    deleted_size += int(size_bytes)
            else:
                errors.append(f"Failed to delete: {file_path}")
        except Exception as e:
            errors.append(f"Error deleting {file_path}: {e}")

    return {
        "status": "completed",
        "message": f"Deleted {deleted_count} files ({deleted_size / (1024 * 1024):.2f} MB)",
        "deleted_count": deleted_count,
        "deleted_size_bytes": deleted_size,
        "files_to_delete": files_to_delete,
        "errors": errors,
    }


def log_data_access(
    data_path: str,
    user_id: str,
    purpose: str,
    access_type: str = "read",
) -> bool:
    """Log data access for audit trail.

    Args:
        data_path: Path to the data accessed
        user_id: Identifier of the user accessing the data
        purpose: Purpose of the access
        access_type: Type of access (read, write, delete) (default: "read")

    Returns:
        True if logging succeeded, False otherwise
    """
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_path": data_path,
            "user_id": user_id,
            "purpose": purpose,
            "access_type": access_type,
            "data_hash": _hash_file(data_path) if Path(data_path).exists() else None,
        }

        # In production, this would write to a secure audit log
        # For now, print to stdout
        print(f"[DATA_ACCESS_LOG] {log_entry}")
        return True

    except Exception as e:
        print(f"Error logging data access: {e}")
        return False


def _hash_file(file_path: str) -> Optional[str]:
    """Calculate SHA-256 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        Hexadecimal hash string or None if error
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception:
        return None


def get_data_retention_info(data_dir: str) -> Dict[str, Any]:
    """Get information about data retention status.

    Args:
        data_dir: Directory to analyze

    Returns:
        Dictionary with retention information
    """
    data_path = Path(data_dir)

    if not data_path.exists() or not data_path.is_dir():
        return {
            "status": "error",
            "message": f"Directory not found: {data_dir}",
        }

    file_count = 0
    total_size = 0
    age_distribution = {
        "0_30_days": 0,
        "30_90_days": 0,
        "90_180_days": 0,
        "180_365_days": 0,
        "365_plus_days": 0,
    }
    current_time = time.time()

    for file_path in data_path.rglob("*"):
        if file_path.is_file():
            file_count += 1
            file_size = file_path.stat().st_size
            total_size += file_size

            file_age_days = (current_time - file_path.stat().st_mtime) / (24 * 60 * 60)

            if file_age_days < 30:
                age_distribution["0_30_days"] += 1
            elif file_age_days < 90:
                age_distribution["30_90_days"] += 1
            elif file_age_days < 180:
                age_distribution["90_180_days"] += 1
            elif file_age_days < 365:
                age_distribution["180_365_days"] += 1
            else:
                age_distribution["365_plus_days"] += 1

    return {
        "status": "success",
        "file_count": file_count,
        "total_size_bytes": total_size,
        "total_size_mb": total_size / (1024 * 1024),
        "age_distribution": age_distribution,
    }
