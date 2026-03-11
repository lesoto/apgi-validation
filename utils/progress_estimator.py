#!/usr/bin/env python3
"""
Progress Estimation Utility
========================

Provides progress estimation and time tracking for long-running operations.
"""

import time
import threading
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ProgressState(Enum):
    """Progress states for operations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressInfo:
    """Progress information for an operation."""

    operation_id: str
    operation_name: str
    current: int = 0
    total: int = 100
    state: ProgressState = ProgressState.PENDING
    start_time: Optional[float] = None
    estimated_end_time: Optional[float] = None
    message: str = ""
    error: Optional[str] = None


class ProgressEstimator:
    """Estimates progress and time remaining for operations."""

    def __init__(self, max_retained_operations: int = 100):
        self.operations: Dict[str, ProgressInfo] = {}
        self.progress_callbacks: Dict[str, Callable[[ProgressInfo], None]] = {}
        self.max_retained_operations = max_retained_operations
        self._lock = threading.Lock()

    def start_operation(
        self,
        operation_id: str,
        operation_name: str,
        total_steps: int = 100,
        callback: Optional[Callable[[ProgressInfo], None]] = None,
    ) -> ProgressInfo:
        """Start tracking a new operation."""
        progress_info = ProgressInfo(
            operation_id=operation_id,
            operation_name=operation_name,
            total=total_steps,
            state=ProgressState.RUNNING,
            start_time=time.time(),
        )

        self.operations[operation_id] = progress_info
        if callback:
            self.progress_callbacks[operation_id] = callback

        self._notify_progress(operation_id)
        return progress_info

    def update_progress(
        self, operation_id: str, current_step: int, message: str = ""
    ) -> Optional[ProgressInfo]:
        """Update progress for an operation."""
        if operation_id not in self.operations:
            return None

        progress_info = self.operations[operation_id]
        progress_info.current = min(current_step, progress_info.total)
        progress_info.message = message

        # Estimate remaining time
        if progress_info.start_time and progress_info.current > 0:
            elapsed = time.time() - progress_info.start_time
            rate = progress_info.current / elapsed
            if rate > 0:
                remaining_steps = progress_info.total - progress_info.current
                estimated_remaining = remaining_steps / rate
                progress_info.estimated_end_time = time.time() + estimated_remaining

        self._notify_progress(operation_id)
        return progress_info

    def complete_operation(
        self, operation_id: str, message: str = "Completed"
    ) -> Optional[ProgressInfo]:
        """Mark an operation as completed."""
        if operation_id not in self.operations:
            return None

        progress_info = self.operations[operation_id]
        progress_info.state = ProgressState.COMPLETED
        progress_info.current = progress_info.total
        progress_info.message = message

        # Auto-cleanup old completed operations to prevent memory leak
        self._cleanup_old_operations()

        self._notify_progress(operation_id)
        return progress_info

    def fail_operation(self, operation_id: str, error: str) -> Optional[ProgressInfo]:
        """Mark an operation as failed."""
        if operation_id not in self.operations:
            return None

        progress_info = self.operations[operation_id]
        progress_info.state = ProgressState.FAILED
        progress_info.error = error
        progress_info.message = f"Failed: {error}"

        self._notify_progress(operation_id)
        return progress_info

    def cancel_operation(self, operation_id: str) -> Optional[ProgressInfo]:
        """Cancel an operation."""
        if operation_id not in self.operations:
            return None

        progress_info = self.operations[operation_id]
        progress_info.state = ProgressState.CANCELLED
        progress_info.message = "Cancelled"

        self._notify_progress(operation_id)
        return progress_info

    def get_progress(self, operation_id: str) -> Optional[ProgressInfo]:
        """Get progress information for an operation."""
        return self.operations.get(operation_id)

    def get_all_progress(self) -> Dict[str, ProgressInfo]:
        """Get progress information for all operations."""
        return self.operations.copy()

    def _cleanup_old_operations(self):
        """Remove old completed operations to prevent memory leak."""
        with self._lock:
            # Separate completed and non-completed operations
            completed_ops = []
            active_ops = []
            for op_id, op_info in self.operations.items():
                if op_info.state in (
                    ProgressState.COMPLETED,
                    ProgressState.FAILED,
                    ProgressState.CANCELLED,
                ):
                    completed_ops.append((op_id, op_info))
                else:
                    active_ops.append((op_id, op_info))

            # Sort completed ops by start time (oldest first)
            completed_ops.sort(key=lambda x: x[1].start_time or 0)

            # Remove excess completed operations beyond retention limit
            if len(completed_ops) > self.max_retained_operations:
                excess_ops = len(completed_ops) - self.max_retained_operations
                for op_id, _ in completed_ops[:excess_ops]:
                    if op_id in self.operations:
                        del self.operations[op_id]
                    if op_id in self.progress_callbacks:
                        del self.progress_callbacks[op_id]

            # Keep only active operations and recently completed operations
            self.operations = {op_id: op_info for op_id, op_info in active_ops}
            for op_id, op_info in completed_ops[self.max_retained_operations :]:
                self.operations[op_id] = op_info

    def remove_operation(self, operation_id: str) -> bool:
        """Remove an operation from tracking."""
        with self._lock:
            if operation_id in self.operations:
                del self.operations[operation_id]
                if operation_id in self.progress_callbacks:
                    del self.progress_callbacks[operation_id]
                return True
            return False

    def _notify_progress(self, operation_id: str):
        """Notify progress callback if available."""
        if operation_id in self.progress_callbacks:
            progress_info = self.operations[operation_id]
            try:
                self.progress_callbacks[operation_id](progress_info)
            except Exception as e:
                print(f"Error in progress callback for {operation_id}: {e}")

    def estimate_time_remaining(self, operation_id: str) -> Optional[float]:
        """Get estimated time remaining for an operation."""
        progress_info = self.get_progress(operation_id)
        if not progress_info or not progress_info.estimated_end_time:
            return None

        return max(0, progress_info.estimated_end_time - time.time())

    def get_progress_percentage(self, operation_id: str) -> float:
        """Get progress percentage for an operation."""
        progress_info = self.get_progress(operation_id)
        if not progress_info or progress_info.total == 0:
            return 0.0

        return (progress_info.current / progress_info.total) * 100

    def format_time_remaining(self, seconds: float) -> str:
        """Format time remaining in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


# Global progress estimator instance
_global_progress_estimator = ProgressEstimator()


def get_progress_estimator() -> ProgressEstimator:
    """Get the global progress estimator instance."""
    return _global_progress_estimator
