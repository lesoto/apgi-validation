#!/usr/bin/env python3
"""
Timeout Handler Utility
====================

Provides timeout handling for stuck operations and processes.
"""

import multiprocessing
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class TimeoutState(Enum):
    """Timeout states."""

    RUNNING = "running"
    TIMEOUT = "timeout"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TimeoutInfo:
    """Timeout information for an operation."""

    operation_id: str
    timeout_seconds: float
    start_time: float
    state: TimeoutState = TimeoutState.RUNNING
    callback: Optional[Callable[[str], None]] = None


class TimeoutHandler:
    """Handles timeouts for long-running operations."""

    def __init__(self) -> None:
        self.timeouts: Dict[str, TimeoutInfo] = {}
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

    def start_monitoring(self) -> None:
        """Start the timeout monitoring thread."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_timeouts, daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop the timeout monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def add_timeout(
        self,
        operation_id: str,
        timeout_seconds: float,
        callback: Optional[Callable[[str], None]] = None,
    ) -> float:
        """Add a timeout for an operation."""
        with self._lock:
            timeout_info = TimeoutInfo(
                operation_id=operation_id,
                timeout_seconds=timeout_seconds,
                start_time=time.time(),
                callback=callback,
            )
            self.timeouts[operation_id] = timeout_info

            if not self._running:
                self.start_monitoring()

        return timeout_info.start_time

    def remove_timeout(self, operation_id: str) -> bool:
        """Remove a timeout for an operation."""
        with self._lock:
            if operation_id in self.timeouts:
                timeout_info = self.timeouts[operation_id]
                timeout_info.state = TimeoutState.COMPLETED
                del self.timeouts[operation_id]
                return True
            return False

    def complete_operation(self, operation_id: str) -> bool:
        """Mark an operation as completed (remove timeout)."""
        return self.remove_timeout(operation_id)

    def extend_timeout(self, operation_id: str, additional_seconds: float) -> bool:
        """Extend the timeout for an operation."""
        with self._lock:
            if operation_id in self.timeouts:
                timeout_info = self.timeouts[operation_id]
                timeout_info.timeout_seconds += additional_seconds
                return True
            return False

    def get_time_remaining(self, operation_id: str) -> Optional[float]:
        """Get time remaining for an operation."""
        with self._lock:
            if operation_id not in self.timeouts:
                return None

            timeout_info = self.timeouts[operation_id]
            elapsed = time.time() - timeout_info.start_time
            remaining = timeout_info.timeout_seconds - elapsed
            return max(0, remaining)

    def _monitor_timeouts(self):
        """Monitor timeouts and trigger callbacks."""
        while self._running:
            try:
                current_time = time.time()
                timeouts_to_trigger = []

                with self._lock:
                    for operation_id, timeout_info in self.timeouts.items():
                        elapsed = current_time - timeout_info.start_time

                        if elapsed >= timeout_info.timeout_seconds:
                            timeout_info.state = TimeoutState.TIMEOUT
                            timeouts_to_trigger.append(operation_id)

                # Trigger callbacks outside the lock
                for operation_id in timeouts_to_trigger:
                    try:
                        timeout_info = self.timeouts.get(operation_id)
                        if timeout_info and timeout_info.callback:
                            timeout_info.callback(operation_id)
                    except (RuntimeError, KeyError, AttributeError, Exception) as e:
                        print(f"Error in timeout callback for {operation_id}: {e}")
                    finally:
                        # Remove the timed out operation
                        self.remove_timeout(operation_id)

                # Sleep for a short interval
                time.sleep(0.5)

            except (RuntimeError, KeyboardInterrupt, SystemExit) as e:
                print(f"Error in timeout monitor: {e}")
                time.sleep(1.0)


class TimeoutError(Exception):
    """Exception raised when an operation times out."""

    pass


def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to a function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Use multiprocessing.Manager to share results between processes
            manager = multiprocessing.Manager()
            result = manager.list()
            exception = manager.list()

            process = multiprocessing.Process(
                target=_timeout_target, args=(func, args, kwargs, result, exception)
            )
            process.start()
            process.join(timeout=timeout_seconds)

            if process.is_alive():
                # Terminate the process
                process.terminate()
                process.join(timeout=1.0)  # Give it a moment to terminate
                if process.is_alive():
                    # Force kill if terminate didn't work (Unix only)
                    try:
                        process.kill()  # This will raise AttributeError on Windows
                        process.join()
                    except AttributeError:
                        # On Windows, kill() is not available, terminate() is the only option
                        # Wait a bit more for terminate() to take effect
                        process.join(timeout=2.0)
                        if process.is_alive():
                            # Process is still alive, log warning but continue
                            # On Windows, this can happen if the process doesn't respond to TerminateProcess
                            print(
                                f"Warning: Process {process.pid} could not be killed after timeout"
                            )
                    raise TimeoutError(
                        f"Operation timed out after {timeout_seconds} seconds"
                    )

            if exception:
                raise exception[0]

            return result[0] if result else None

        return wrapper

    return decorator


# Module-level function to avoid pickling issues with local functions
def _timeout_target(func, args, kwargs, result_list, exception_list):
    """Target function for multiprocessing - must be module-level to be picklable."""
    try:
        result_list.append(func(*args, **kwargs))
    except Exception as e:
        exception_list.append(e)


def run_with_timeout(
    func: Callable, args: tuple = (), kwargs: dict = None, timeout_seconds: float = 30.0
) -> Any:
    """Run a function with a timeout."""
    if kwargs is None:
        kwargs = {}

    # Use multiprocessing.Manager to share results between processes
    manager = multiprocessing.Manager()
    result = manager.list()
    exception = manager.list()

    process = multiprocessing.Process(
        target=_timeout_target, args=(func, args, kwargs, result, exception)
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        # Terminate the process
        process.terminate()
        process.join(timeout=1.0)  # Give it a moment to terminate
        if process.is_alive():
            process.kill()  # Force kill if terminate didn't work
            process.join()
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

    if exception:
        raise exception[0]

    return result[0] if result else None


# Allowlisted safe keyword arguments for subprocess.Popen
_POPEN_SAFE_KWARGS = frozenset({"cwd", "env", "encoding"})


def run_subprocess_with_timeout(
    command: List[str],
    timeout_seconds: float = 300.0,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    encoding: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess with a timeout.

    Only accepts an explicit allowlist of Popen parameters to prevent
    injection of dangerous options such as ``shell=True``.
    """
    popen_kwargs: Dict[str, Any] = {}
    if cwd is not None:
        popen_kwargs["cwd"] = cwd
    if env is not None:
        popen_kwargs["env"] = env
    if encoding is not None:
        popen_kwargs["encoding"] = encoding

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,  # Explicitly disable shell to prevent injection
            cwd=cwd,
            env=env,
            encoding=encoding,
        )

        stdout, stderr = process.communicate(timeout=timeout_seconds)

        return subprocess.CompletedProcess(
            args=command, returncode=process.returncode, stdout=stdout, stderr=stderr
        )

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise TimeoutError(f"Subprocess timed out after {timeout_seconds} seconds")


# Global timeout handler instance
_global_timeout_handler = TimeoutHandler()


def get_timeout_handler() -> TimeoutHandler:
    """Get the global timeout handler instance."""
    return _global_timeout_handler
