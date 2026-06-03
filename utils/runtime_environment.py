"""Runtime cache configuration for protocol execution.

Several scientific dependencies compile or cache code during import. In the
Codex/test sandbox the default home-directory caches may be unwritable, so set
safe defaults before importing those packages.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Tuple


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_runtime_environment(project_root: Path | None = None) -> None:
    """Set writable cache locations for numba, PyTensor, and Matplotlib."""
    temp_root = Path(tempfile.gettempdir()) / "apgi-validation-runtime"
    _ensure_dir(temp_root)

    # Disable numba JIT completely to prevent cache errors
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(_ensure_dir(temp_root / "numba")))

    # Additional MNE-specific numba settings to prevent caching issues
    os.environ.setdefault("MNE_USE_NUMBA", "false")

    os.environ.setdefault("MPLCONFIGDIR", str(_ensure_dir(temp_root / "matplotlib")))

    pytensor_dir = _ensure_dir(temp_root / "pytensor")
    existing_flags = os.environ.get("PYTENSOR_FLAGS", "")
    if "compiledir=" not in existing_flags:
        compiledir_flag = f"compiledir={pytensor_dir}"
        os.environ["PYTENSOR_FLAGS"] = f"{existing_flags},{compiledir_flag}" if existing_flags else compiledir_flag

    # Ensure PyTensor compilation directory is writable
    try:
        _ensure_dir(pytensor_dir)
        # Test write permission
        test_file = pytensor_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except Exception:
        # Fallback to a subdirectory in the current working directory
        fallback_dir = Path.cwd() / ".pytensor_cache"
        _ensure_dir(fallback_dir)
        compiledir_flag = f"compiledir={fallback_dir}"
        os.environ["PYTENSOR_FLAGS"] = compiledir_flag

    if project_root is not None:
        os.environ.setdefault("APGI_PROJECT_ROOT", str(project_root))


def safe_import_mne() -> Tuple[bool, Any]:
    """Import MNE if possible, otherwise return an unavailable sentinel."""
    configure_runtime_environment()
    try:
        import mne  # type: ignore

        return True, mne
    except Exception:
        return False, None
