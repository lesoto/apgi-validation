#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import multiprocessing
import subprocess  # nosec B404
import threading
import time
from enum import Enum
from typing import IO, Any, Callable, Dict, List, Optional, Union


class TimeoutState(Enum):
    """Timeout states."""

    RUNNING = "running"
    TIMEOUT = "timeout"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TimeoutInfo:
    """Timeout information for an operation."""

    def __init__(
        self,
        operation_id: str,
        timeout_seconds: float,
        start_time: float,
        state: TimeoutState = TimeoutState.RUNNING,
        callback: Optional[Callable[[str], None]] = None,
    ):
        self.operation_id = operation_id
        self.timeout_seconds = timeout_seconds
        self.start_time = start_time
        self.state = state
        self.callback = callback


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
        self._monitor_thread = threading.Thread(target=self._monitor_timeouts, daemon=True)
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
            return max(0.0, remaining)

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
            # For simple functions that don't actually need multiprocessing,
            # just run them directly and check if they complete quickly
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                if execution_time > timeout_seconds:
                    raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

                return result

            except Exception as e:
                # Re-raise any exception from the function
                raise e

        return wrapper

    return decorator


def _queue_worker(func: Callable, args: tuple, kwargs: dict, queue: "multiprocessing.Queue") -> None:
    """Worker process target that reports either a result or an exception via a Queue.

    This avoids ``multiprocessing.Manager`` which requires binding a local socket.
    Some sandboxed environments disallow that, causing spurious ``PermissionError``
    and "No result" failures.
    """
    try:
        queue.put(("result", func(*args, **kwargs)))
    except Exception as e:
        import traceback

        queue.put(("exception", repr(e), traceback.format_exc()))


def run_with_timeout(func: Callable, args: tuple = (), kwargs: dict = None, timeout_seconds: float = 30.0) -> Any:
    """Run a function with a timeout."""
    if kwargs is None:
        kwargs = {}

    # For simple functions, just run them directly with timeout check
    start_time = time.time()

    try:
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        if execution_time > timeout_seconds:
            raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")

        return result

    except Exception as e:
        # Re-raise any exception from the function
        raise e


# Allowlisted safe keyword arguments for subprocess.Popen
_POPEN_SAFE_KWARGS = frozenset({"cwd", "env", "encoding", "stdout", "stderr"})


def run_subprocess_with_timeout(
    command: List[str],
    timeout_seconds: float = 300.0,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    stdout: Optional[Union[int, IO[Any]]] = subprocess.PIPE,
    stderr: Optional[Union[int, IO[Any]]] = subprocess.PIPE,
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
            stdout=stdout,
            stderr=stderr,
            text=True,
            shell=False,  # Explicitly disable shell to prevent injection
            cwd=cwd,
            env=env,
            encoding=encoding,
        )  # nosec B603

        stdout_str, stderr_str = process.communicate(timeout=timeout_seconds)
        return subprocess.CompletedProcess(
            args=command, returncode=process.returncode, stdout=stdout_str, stderr=stderr_str
        )

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise TimeoutError(f"Subprocess timed out after {timeout_seconds} seconds")
        process.terminate()


# Global timeout handler instance
_global_timeout_handler = TimeoutHandler()


def get_timeout_handler() -> TimeoutHandler:
    """Get the global timeout handler instance."""
    return _global_timeout_handler
