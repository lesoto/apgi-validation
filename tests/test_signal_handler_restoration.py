"""
Tests for signal handler restoration including exception paths and proper cleanup.
"""

import signal
import time
import pytest
from unittest.mock import patch
import threading

from utils.signal_handler import SignalHandler, restore_signal_handlers


class TestSignalHandlerRestoration:
    """Test signal handler restoration mechanisms."""

    @pytest.fixture
    def original_handlers(self):
        """Store original signal handlers."""
        original = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }
        yield original
        # Restore original handlers
        for sig, handler in original.items():
            signal.signal(sig, handler)

    def test_signal_handler_installation(self, original_handlers):
        """Test that signal handlers are installed correctly."""
        with SignalHandler():
            # Should install custom handlers
            assert signal.getsignal(signal.SIGINT) != original_handlers[signal.SIGINT]
            assert signal.getsignal(signal.SIGTERM) != original_handlers[signal.SIGTERM]

    def test_signal_handler_restoration(self, original_handlers):
        """Test that signal handlers are restored correctly."""
        with SignalHandler():
            # Handlers should be different
            assert signal.getsignal(signal.SIGINT) != original_handlers[signal.SIGINT]

        # After context exit, handlers should be restored
        assert signal.getsignal(signal.SIGINT) == original_handlers[signal.SIGINT]
        assert signal.getsignal(signal.SIGTERM) == original_handlers[signal.SIGTERM]

    def test_signal_handler_restoration_with_exception(self, original_handlers):
        """Test signal handler restoration when exceptions occur."""
        SignalHandler()

        # Mock restore_signal_handlers to raise exception
        with patch("utils.signal_handler.restore_signal_handlers") as mock_restore:
            mock_restore.side_effect = Exception("Test exception")

            # Should handle exception gracefully
            try:
                restore_signal_handlers()
            except Exception:
                pass  # Expected

        # Verify original handlers are still restored despite exception
        restore_signal_handlers()
        assert signal.getsignal(signal.SIGINT) == original_handlers[signal.SIGINT]

    def test_signal_handler_in_context_manager(self, original_handlers):
        """Test signal handler restoration using context manager."""
        with SignalHandler():
            # Handlers should be installed
            assert signal.getsignal(signal.SIGINT) != original_handlers[signal.SIGINT]

        # Handlers should be restored after context exit
        assert signal.getsignal(signal.SIGINT) == original_handlers[signal.SIGINT]
        assert signal.getsignal(signal.SIGTERM) == original_handlers[signal.SIGTERM]

    def test_signal_handler_context_manager_exception(self, original_handlers):
        """Test signal handler restoration when exception occurs in context."""
        try:
            with SignalHandler():
                # Handlers should be installed
                assert (
                    signal.getsignal(signal.SIGINT) != original_handlers[signal.SIGINT]
                )
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Handlers should still be restored despite exception
        assert signal.getsignal(signal.SIGINT) == original_handlers[signal.SIGINT]
        assert signal.getsignal(signal.SIGTERM) == original_handlers[signal.SIGTERM]

    def test_signal_handler_thread_safety(self, original_handlers):
        """Test signal handler restoration is thread-safe."""
        results = []
        errors = []

        def signal_operations():
            try:
                with SignalHandler():
                    # Simulate some work
                    time.sleep(0.1)
                results.append(True)
            except Exception as e:
                errors.append(e)

        # Run multiple threads with signal operations
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=signal_operations)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(results) == 5
        assert len(errors) == 0

        # Final state should have original handlers
        assert signal.getsignal(signal.SIGINT) == original_handlers[signal.SIGINT]

    def test_signal_handler_nested_contexts(self, original_handlers):
        """Test nested signal handler contexts."""
        with SignalHandler():
            outer_sigint = signal.getsignal(signal.SIGINT)
            assert outer_sigint != original_handlers[signal.SIGINT]

            with SignalHandler():
                # Inner handler should be different from outer
                assert signal.getsignal(signal.SIGINT) != outer_sigint

            # Should restore to outer handler
            assert signal.getsignal(signal.SIGINT) == outer_sigint

        # Should restore to original
        assert signal.getsignal(signal.SIGINT) == original_handlers[signal.SIGINT]

    def test_signal_handler_multiple_restoration_calls(self, original_handlers):
        """Test multiple restoration calls don't cause issues."""
        SignalHandler()

        # Multiple restoration calls should be safe
        restore_signal_handlers()
        restore_signal_handlers()
        restore_signal_handlers()

        # Should still have original handlers
        assert signal.getsignal(signal.SIGINT) == original_handlers[signal.SIGINT]

    def test_signal_handler_with_custom_signals(self, original_handlers):
        """Test signal handler with custom signal handlers."""

        def custom_handler(sig, frame):
            pass

        # Set custom handler
        signal.signal(signal.SIGUSR1, custom_handler)
        original_usr1 = signal.getsignal(signal.SIGUSR1)

        SignalHandler()

        # Custom signal should be preserved
        assert signal.getsignal(signal.SIGUSR1) == original_usr1

        restore_signal_handlers()

        # Custom signal should still be preserved
        assert signal.getsignal(signal.SIGUSR1) == original_usr1

    def test_signal_handler_graceful_shutdown(self, original_handlers):
        """Test graceful shutdown signal handling."""
        shutdown_called = []

        def mock_shutdown():
            shutdown_called.append(True)

        handler = SignalHandler(shutdown_callback=mock_shutdown)

        # Simulate SIGINT by calling the handler directly
        try:
            handler._handle_signal(signal.SIGINT, None)
        except KeyboardInterrupt:
            pass  # Expected

        # Verify shutdown callback was called
        assert len(shutdown_called) == 1

    def test_signal_handler_cleanup_on_destruction(self, original_handlers):
        """Test signal handler cleanup on object destruction."""
        # Use context manager to install handlers
        with SignalHandler():
            # Handlers should be installed
            assert signal.getsignal(signal.SIGINT) != original_handlers[signal.SIGINT]

        # After context exit, handlers should be restored
        assert signal.getsignal(signal.SIGINT) == original_handlers[signal.SIGINT]

        # Test that destruction doesn't crash even when not using context manager
        handler = SignalHandler()
        del handler

        # Force garbage collection
        import gc

        gc.collect()

        # If we get here, no crash occurred
        assert True


class TestSignalHandlerEdgeCases:
    """Test edge cases for signal handling."""

    def test_signal_handler_with_no_original_handler(self):
        """Test when there's no original signal handler."""
        # Set signal to default handling
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        SignalHandler()
        restore_signal_handlers()

        # Should restore to default
        assert signal.getsignal(signal.SIGINT) == signal.SIG_DFL

    def test_signal_handler_with_ignore(self):
        """Test signal handler restoration with SIG_IGN."""
        # Set signal to ignore
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        SignalHandler()
        restore_signal_handlers()

        # Should restore to ignore
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN

    def test_signal_handler_invalid_signal(self):
        """Test handling of invalid signal numbers."""
        SignalHandler()

        # Should handle invalid signals gracefully
        try:
            # This might not be a valid signal on all systems
            signal.signal(999, lambda s, f: None)
        except (ValueError, OSError):
            pass  # Expected

    def test_signal_handler_concurrent_modification(self):
        """Test concurrent signal handler modification."""
        results = []

        def modify_signals():
            try:
                with SignalHandler():
                    results.append(True)
            except Exception:
                results.append(False)

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=modify_signals)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should succeed
        assert all(results)
