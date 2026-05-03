import subprocess
import time
from unittest.mock import Mock, patch

import pytest

from utils.timeout_handler import (
    TimeoutError,
    TimeoutHandler,
    TimeoutInfo,
    TimeoutState,
    get_timeout_handler,
    run_subprocess_with_timeout,
    run_with_timeout,
    with_timeout,
)


def test_timeout_state_and_info():
    info = TimeoutInfo(operation_id="test", timeout_seconds=1.0, start_time=10.0)
    assert info.state == TimeoutState.RUNNING
    assert info.operation_id == "test"


def test_timeout_handler_lifecycle():
    handler = TimeoutHandler()
    assert not handler._running
    handler.start_monitoring()
    assert handler._running
    # Second call should return early
    handler.start_monitoring()
    assert handler._running
    handler.stop_monitoring()
    assert not handler._running
    # Stop again is safe
    handler.stop_monitoring()


def test_timeout_handler_operations():
    handler = TimeoutHandler()

    # Add timeout
    handler.add_timeout("op1", 10.0)
    assert "op1" in handler.timeouts
    assert handler._running  # Should auto-start

    # Get time remaining
    remaining = handler.get_time_remaining("op1")
    assert 0 <= remaining <= 10.0

    # Time remaining for non-existent
    assert handler.get_time_remaining("missing") is None

    # Extend timeout
    assert handler.extend_timeout("op1", 5.0)
    assert handler.timeouts["op1"].timeout_seconds == 15.0

    # Extend missing
    assert not handler.extend_timeout("missing", 5.0)

    # Complete operation
    assert handler.complete_operation("op1")
    assert "op1" not in handler.timeouts

    # Complete missing
    assert not handler.complete_operation("missing")

    handler.stop_monitoring()


def test_timeout_handler_monitor_trigger():
    handler = TimeoutHandler()
    callback = Mock()
    handler.add_timeout("op1", 0.1, callback)

    # Wait for monitor thread to trigger callback
    time.sleep(0.7)

    callback.assert_called_once_with("op1")
    assert "op1" not in handler.timeouts
    handler.stop_monitoring()


def test_timeout_handler_monitor_callback_error():
    handler = TimeoutHandler()

    def failing_callback(op_id):
        raise RuntimeError("Callback failed")

    handler.add_timeout("op1", 0.1, failing_callback)

    # Wait for monitor thread. It should handle the error and continue.
    time.sleep(0.7)

    assert "op1" not in handler.timeouts
    handler.stop_monitoring()


def test_with_timeout_success():
    @with_timeout(1.0)
    def fast_func():
        return "success"

    with patch("multiprocessing.Process") as mock_process:
        mock_p = Mock()
        mock_p.is_alive.return_value = False
        mock_process.return_value = mock_p

        # We need to simulate the target running and setting result
        def side_effect(*args, **kwargs):
            target = kwargs.get("target")
            if target:
                target()
            return mock_p

        mock_process.side_effect = side_effect

        assert fast_func() == "success"


def test_with_timeout_exception():
    @with_timeout(1.0)
    def failing_func():
        raise ValueError("Failed inside")

    with patch("multiprocessing.Process") as mock_process:
        mock_p = Mock()
        mock_p.is_alive.return_value = False
        mock_process.return_value = mock_p

        def side_effect(*args, **kwargs):
            target = kwargs.get("target")
            if target:
                target()
            return mock_p

        mock_process.side_effect = side_effect

        with pytest.raises(ValueError, match="Failed inside"):
            failing_func()


def test_with_timeout_expired():
    @with_timeout(0.1)
    def slow_func():
        return "done"

    with patch("multiprocessing.Process") as mock_process:
        mock_p = Mock()
        mock_p.is_alive.return_value = True
        mock_process.return_value = mock_p

        with pytest.raises(TimeoutError, match="Operation timed out"):
            slow_func()

        mock_p.terminate.assert_called_once()


def test_run_with_timeout_success():
    def fast_func(a, b=2):
        return a + b

    with patch("multiprocessing.Process") as mock_process:
        mock_p = Mock()
        mock_p.is_alive.return_value = False
        mock_process.return_value = mock_p

        def side_effect(*args, **kwargs):
            target = kwargs.get("target")
            if target:
                target()
            return mock_p

        mock_process.side_effect = side_effect

        assert (
            run_with_timeout(fast_func, args=(3,), kwargs={"b": 4}, timeout_seconds=1.0)
            == 7
        )


def test_run_with_timeout_exception():
    def failing_func():
        raise KeyError("Missing key")

    with patch("multiprocessing.Process") as mock_process:
        mock_p = Mock()
        mock_p.is_alive.return_value = False
        mock_process.return_value = mock_p

        def side_effect(*args, **kwargs):
            target = kwargs.get("target")
            if target:
                target()
            return mock_p

        mock_process.side_effect = side_effect

        with pytest.raises(KeyError, match="Missing key"):
            run_with_timeout(failing_func, timeout_seconds=1.0)


def test_run_with_timeout_expired():
    def slow_func():
        return "done"

    with patch("multiprocessing.Process") as mock_process:
        mock_p = Mock()
        mock_p.is_alive.return_value = True
        mock_process.return_value = mock_p

        with pytest.raises(TimeoutError, match="Operation timed out"):
            run_with_timeout(slow_func, timeout_seconds=0.1)

        mock_p.terminate.assert_called_once()


@patch("subprocess.Popen")
def test_run_subprocess_with_timeout_success(mock_popen):
    mock_process = Mock()
    mock_process.communicate.return_value = ("output", "error")
    mock_process.returncode = 0
    mock_popen.return_value = mock_process

    result = run_subprocess_with_timeout(
        ["ls"], timeout_seconds=1.0, cwd="/tmp", env={"A": "1"}, encoding="utf-8"
    )

    assert result.returncode == 0
    assert result.stdout == "output"
    assert result.stderr == "error"
    mock_popen.assert_called_once()
    kwargs = mock_popen.call_args[1]
    assert kwargs["cwd"] == "/tmp"
    assert kwargs["env"] == {"A": "1"}
    assert kwargs["encoding"] == "utf-8"


@patch("subprocess.Popen")
def test_run_subprocess_with_timeout_expired(mock_popen):
    mock_process = Mock()
    mock_process.communicate.side_effect = subprocess.TimeoutExpired(
        cmd=["ls"], timeout=0.1
    )
    mock_popen.return_value = mock_process

    with pytest.raises(TimeoutError, match="Subprocess timed out"):
        run_subprocess_with_timeout(["ls"], timeout_seconds=0.1)

    mock_process.kill.assert_called_once()
    mock_process.wait.assert_called_once()


def test_get_timeout_handler():
    handler = get_timeout_handler()
    assert isinstance(handler, TimeoutHandler)
