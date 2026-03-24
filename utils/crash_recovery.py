#!/usr/bin/env python3
"""
Crash Recovery Utility
====================

Provides automatic crash recovery and state restoration for applications.
"""

import json
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict

try:
    from utils.logging_config import apgi_logger
except ImportError:
    # Fallback to basic logging if utils.logging_config is not available
    import logging

    logging.basicConfig(level=logging.INFO)
    apgi_logger = logging.getLogger("crash_recovery")


@dataclass
class RecoveryState:
    """Represents the state of an application for recovery."""

    app_name: str
    timestamp: float
    state_data: Dict[str, Any]
    crash_info: Optional[Dict[str, Any]] = None
    recovery_attempts: int = 0


class CrashRecovery:
    """Handles automatic crash recovery and state restoration."""

    def __init__(self, app_name: str, recovery_dir: str = "recovery"):
        self.app_name = app_name
        self.recovery_dir = Path(recovery_dir)
        self.recovery_dir.mkdir(exist_ok=True)

        self.state_file = self.recovery_dir / f"{app_name}_state.json"
        self.crash_log_file = self.recovery_dir / f"{app_name}_crashes.log"

        self.current_state: Optional[RecoveryState] = None
        self.auto_save_enabled = True
        self.auto_save_interval = 30.0  # seconds
        self.max_recovery_attempts = 3
        self.max_auto_save_retries = 5  # Maximum number of auto-save retries
        self.auto_save_backoff_factor = 2.0  # Exponential backoff multiplier
        self.auto_save_max_delay = 300.0  # Maximum delay between retries (5 minutes)

        self._auto_save_thread: Optional[threading.Thread] = None
        self._stop_auto_save = threading.Event()

        # Check for previous crash on startup
        self._check_for_crash()

    def save_state(
        self, state_data: Dict[str, Any], crash_info: Optional[Dict[str, Any]] = None
    ):
        """Save the current application state."""
        self.current_state = RecoveryState(
            app_name=self.app_name,
            timestamp=time.time(),
            state_data=state_data,
            crash_info=crash_info,
            recovery_attempts=0,
        )

        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(asdict(self.current_state), f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save recovery state: {e}")

    def load_state(self) -> Optional[RecoveryState]:
        """Load the last saved application state."""
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return RecoveryState(**data)
        except Exception as e:
            print(f"Failed to load recovery state: {e}")
            return None

    def has_recovery_state(self) -> bool:
        """Check if recovery state is available."""
        return self.state_file.exists() and self.load_state() is not None

    def recover_state(self) -> Optional[Dict[str, Any]]:
        """Recover the last saved state."""
        if not self.has_recovery_state():
            return None

        state = self.load_state()
        if not state:
            return None

        # Check recovery attempts
        if state.recovery_attempts >= self.max_recovery_attempts:
            print(f"Max recovery attempts ({self.max_recovery_attempts}) exceeded")
            self._clear_recovery_state()
            return None

        # Update recovery attempts
        state.recovery_attempts += 1
        self.current_state = state

        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(asdict(state), f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to update recovery attempts: {e}")

        return state.state_data

    def start_auto_save(self, get_state_func: Callable[[], Dict[str, Any]]):
        """Start automatic state saving."""
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            return

        self._stop_auto_save.clear()
        self._auto_save_thread = threading.Thread(
            target=self._auto_save_worker, args=(get_state_func,), daemon=True
        )
        self._auto_save_thread.start()

    def stop_auto_save(self):
        """Stop automatic state saving."""
        self._stop_auto_save.set()
        if self._auto_save_thread:
            self._auto_save_thread.join(timeout=2.0)

    def _auto_save_worker(self, get_state_func: Callable[[], Dict[str, Any]]):
        """Worker thread for automatic state saving with exponential backoff and retry limits."""
        consecutive_failures = 0
        current_delay = self.auto_save_interval

        while not self._stop_auto_save.wait(current_delay):
            try:
                state_data = get_state_func()
                if state_data:
                    self.save_state(state_data)
                    consecutive_failures = 0  # Reset on success
                    current_delay = (
                        self.auto_save_interval
                    )  # Reset delay to base interval
                else:
                    consecutive_failures += 1
                    apgi_logger.logger.warning(
                        f"Auto-save: get_state_func returned None (failure {consecutive_failures}/{self.max_auto_save_retries})"
                    )
            except Exception as e:
                consecutive_failures += 1
                apgi_logger.logger.error(
                    f"Auto-save failed (failure {consecutive_failures}/{self.max_auto_save_retries}): {e}"
                )

            # Check if we've exceeded max retries
            if consecutive_failures >= self.max_auto_save_retries:
                apgi_logger.logger.critical(
                    f"Auto-save thread terminating after {consecutive_failures} consecutive failures. "
                    f"Max retries ({self.max_auto_save_retries}) exceeded."
                )
                break

            # Exponential backoff with jitter
            if consecutive_failures > 0:
                current_delay = min(
                    current_delay * self.auto_save_backoff_factor,
                    self.auto_save_max_delay,
                )
                # Add small random jitter to prevent thundering herd
                jitter = (
                    current_delay * 0.1 * (0.5 - time.time() % 1)
                )  # Small random variation
                current_delay = max(self.auto_save_interval, current_delay + jitter)

    def _check_for_crash(self):
        """Check if the application crashed previously."""
        if self.has_recovery_state():
            state = self.load_state()
            if state and state.crash_info:
                self._log_crash(state.crash_info)
                print(
                    f"Previous crash detected at {datetime.fromtimestamp(state.timestamp)}"
                )
                print(f"Recovery attempts: {state.recovery_attempts}")

    def _log_crash(self, crash_info: Dict[str, Any]):
        """Log crash information."""
        try:
            with open(self.crash_log_file, ', encoding="utf-8"a') as f:
                f.write(f"\n=== Crash at {datetime.now().isoformat()} ===\n")
                f.write(f"Application: {self.app_name}\n")
                f.write(f"Error: {crash_info.get('error', 'Unknown')}\n")
                f.write(f"Traceback:\n{crash_info.get('traceback', 'No traceback')}\n")
                f.write("=" * 50 + "\n")
        except Exception as e:
            print(f"Failed to log crash: {e}")

    def _clear_recovery_state(self):
        """Clear the recovery state file."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
        except Exception as e:
            print(f"Failed to clear recovery state: {e}")

    def clear_recovery_state(self):
        """Public method to clear recovery state."""
        self._clear_recovery_state()
        self.current_state = None

    def handle_crash(
        self, exception: Exception, additional_info: Optional[Dict[str, Any]] = None
    ):
        """Handle an application crash."""
        crash_info = {
            "error": str(exception),
            "type": type(exception).__name__,
            "traceback": traceback.format_exc(),
            "timestamp": time.time(),
            "additional_info": additional_info or {},
        }

        # Try to save current state if available
        current_state_data = None
        if hasattr(self, "_get_current_state"):
            try:
                current_state_data = self._get_current_state()
            except Exception:
                pass

        self.save_state(current_state_data or {}, crash_info)
        self._log_crash(crash_info)

    def set_recovery_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set a callback to be called when recovering state."""
        self._recovery_callback = callback

    def get_crash_log(self) -> Optional[str]:
        """Get the contents of the crash log."""
        if not self.crash_log_file.exists():
            return None

        try:
            with open(self.crash_log_file, ', encoding="utf-8"r') as f:
                return f.read()
        except Exception as e:
            print(f"Failed to read crash log: {e}")
            return None

    def clear_crash_log(self):
        """Clear the crash log."""
        try:
            if self.crash_log_file.exists():
                self.crash_log_file.unlink()
        except Exception as e:
            print(f"Failed to clear crash log: {e}")


def crash_recovery_decorator(recovery: CrashRecovery):
    """Decorator to automatically handle crashes in functions."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                recovery.handle_crash(
                    e,
                    {
                        "function": func.__name__,
                        "args": str(args)[:200],  # Limit length
                        "kwargs": str(kwargs)[:200],  # Limit length
                    },
                )
                raise

        return wrapper

    return decorator


# Global crash recovery instances
_crash_recoveries: Dict[str, CrashRecovery] = {}


def get_crash_recovery(app_name: str) -> CrashRecovery:
    """Get or create a crash recovery instance for an app."""
    if app_name not in _crash_recoveries:
        _crash_recoveries[app_name] = CrashRecovery(app_name)
    return _crash_recoveries[app_name]
