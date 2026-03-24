#!/usr/bin/env python3
"""
Signal Handler Utility
====================

Provides signal handling functionality for graceful shutdown and cleanup.
"""

import signal
import threading
from contextlib import contextmanager
from typing import Optional, Callable


class SignalHandler:
    """Signal handler for graceful shutdown and cleanup."""

    def __init__(self, shutdown_callback: Optional[Callable] = None):
        """Initialize signal handler.

        Args:
            shutdown_callback: Optional callback to call on shutdown
        """
        self.shutdown_callback = shutdown_callback
        self.original_handlers = {}
        self._lock = threading.Lock()

    def __enter__(self):
        """Context manager entry - install signal handlers."""
        self._install_handlers()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore signal handlers."""
        self._restore_handlers()

    def _install_handlers(self):
        """Install signal handlers for SIGINT and SIGTERM."""
        with self._lock:
            # Store original handlers
            self.original_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
            self.original_handlers[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)

            # Install custom handlers
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

    def _restore_handlers(self):
        """Restore original signal handlers."""
        with self._lock:
            for sig, handler in self.original_handlers.items():
                signal.signal(sig, handler)

    def _handle_signal(self, signum, frame):
        """Handle incoming signals."""
        if self.shutdown_callback:
            try:
                self.shutdown_callback()
            except Exception:
                pass  # Ignore errors in shutdown callback

        # Restore original handlers and re-raise signal
        self._restore_handlers()

        # Re-raise the signal for default handling
        if hasattr(signal, "SIG_DFL"):
            signal.signal(signum, signal.SIG_DFL)

        # Raise KeyboardInterrupt for SIGINT to maintain expected behavior
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()


# Global handler instance
_global_handler = None
_handler_lock = threading.Lock()


def get_signal_handler() -> SignalHandler:
    """Get the global signal handler instance."""
    global _global_handler

    with _handler_lock:
        if _global_handler is None:
            _global_handler = SignalHandler()
        return _global_handler


def restore_signal_handlers():
    """Restore all signal handlers to their original state."""
    global _global_handler

    with _handler_lock:
        if _global_handler is not None:
            _global_handler._restore_handlers()


@contextmanager
def signal_context(shutdown_callback: Optional[Callable] = None):
    """Context manager for signal handling.

    Args:
        shutdown_callback: Optional callback to call on shutdown
    """
    handler = SignalHandler(shutdown_callback)
    try:
        yield handler
    finally:
        handler._restore_handlers()
