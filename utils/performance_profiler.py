"""
APGI Performance Profiling Tools
================================

Advanced performance profiling and visualization tools for APGI framework.
Provides real-time monitoring, bottleneck identification, and performance optimization insights.
"""

import json
import pickle
import threading
import time
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns

# Project imports
try:
    from utils.logging_config import apgi_logger
except ImportError:
    # Fallback if running as standalone script
    import logging

    class MockAPGILogger:
        def __init__(self):
            self.logger = logging.getLogger(__name__)

    apgi_logger = MockAPGILogger()

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""

    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionProfile:
    """Profile information for a function."""

    name: str
    call_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    std_time: float = 0.0
    last_called: Optional[datetime] = None
    memory_usage: List[float] = field(default_factory=list)
    errors: int = 0


class SystemMonitor:
    """Real-time system resource monitoring."""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self.thread = None
        self.metrics_history = deque(maxlen=3600)  # 1 hour at 1s intervals
        self.start_time = datetime.now()

    def start_monitoring(self):
        """Start system monitoring in background thread."""
        if not self.monitoring:
            self.monitoring = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            apgi_logger.logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.thread is not None:
            self.thread.join(timeout=2)
            apgi_logger.logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()

                # Memory metrics
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()

                # Disk metrics
                disk = psutil.disk_usage(".")
                disk_io = psutil.disk_io_counters()

                # Network metrics
                network = psutil.net_io_counters()

                # Process metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()

                metric = {
                    "timestamp": datetime.now(),
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "cpu_freq_current": cpu_freq.current if cpu_freq else None,
                    "memory_percent": memory.percent,
                    "memory_used": memory.used,
                    "memory_available": memory.available,
                    "swap_percent": swap.percent,
                    "disk_percent": (disk.used / disk.total) * 100,
                    "disk_free": disk.free,
                    "process_memory_rss": process_memory.rss,
                    "process_memory_vms": process_memory.vms,
                    "process_cpu_percent": process_cpu,
                    "network_bytes_sent": network.bytes_sent,
                    "network_bytes_recv": network.bytes_recv,
                    "disk_read_bytes": disk_io.read_bytes if disk_io else 0,
                    "disk_write_bytes": disk_io.write_bytes if disk_io else 0,
                }

                self.metrics_history.append(metric)

            except Exception as e:
                apgi_logger.logger.warning(f"Error in system monitoring: {e}")

            time.sleep(self.update_interval)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return {}

    def get_metrics_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get summary of metrics for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [
            m for m in self.metrics_history if m["timestamp"] > cutoff_time
        ]

        if not recent_metrics:
            return {}

        summary = {}
        for key in ["cpu_percent", "memory_percent", "process_cpu_percent"]:
            values = [m[key] for m in recent_metrics if key in m and m[key] is not None]
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_max"] = np.max(values)
                summary[f"{key}_min"] = np.min(values)

        return summary


class PerformanceProfiler:
    """Advanced performance profiling for APGI framework."""

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path("apgi_output/performance_profiles")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.custom_metrics: List[PerformanceMetric] = []
        self.system_monitor = SystemMonitor()
        self.active_profiling = defaultdict(list)

        # Start system monitoring
        self.system_monitor.start_monitoring()

    def profile_function(self, category: str = "general"):
        """Decorator to profile function performance."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = f"{func.__module__}.{func.__name__}"
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss

                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = e
                    raise
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss

                    duration = end_time - start_time
                    memory_delta = (end_memory - start_memory) / (1024**2)  # MB

                    # Update function profile
                    if func_name not in self.function_profiles:
                        self.function_profiles[func_name] = FunctionProfile(
                            name=func_name
                        )

                    profile = self.function_profiles[func_name]
                    profile.call_count += 1
                    profile.total_time += duration
                    profile.avg_time = profile.total_time / profile.call_count
                    profile.min_time = min(profile.min_time, duration)
                    profile.max_time = max(profile.max_time, duration)
                    profile.last_called = datetime.now()
                    profile.memory_usage.append(memory_delta)

                    if not success:
                        profile.errors += 1

                    # Log performance metric
                    self.add_metric(
                        name=f"function_{func_name.split('.')[-1]}",
                        value=duration,
                        unit="seconds",
                        category=category,
                        metadata={
                            "function": func_name,
                            "success": success,
                            "memory_delta_mb": memory_delta,
                            "error": str(error) if error else None,
                        },
                    )

                return result

            return wrapper

        return decorator

    @contextmanager
    def profile_context(self, name: str, category: str = "context"):
        """Context manager for profiling code blocks."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            yield
            success = True
            error = None
        except Exception as e:
            success = False
            error = e
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            duration = end_time - start_time
            memory_delta = (end_memory - start_memory) / (1024**2)  # MB

            self.add_metric(
                name=name,
                value=duration,
                unit="seconds",
                category=category,
                metadata={
                    "success": success,
                    "memory_delta_mb": memory_delta,
                    "error": str(error) if error else None,
                },
            )

    def add_metric(
        self,
        name: str,
        value: float,
        unit: str,
        category: str = "custom",
        metadata: Dict[str, Any] = None,
    ):
        """Add a custom performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=category,
            metadata=metadata or {},
        )
        self.custom_metrics.append(metric)

        # Also log to the main logger
        apgi_logger.logger.log_performance_metric(name, value, unit)

    def get_top_functions(
        self, metric: str = "total_time", limit: int = 10
    ) -> List[FunctionProfile]:
        """Get top functions by specified metric."""
        if not self.function_profiles:
            return []

        sorted_functions = sorted(
            self.function_profiles.values(),
            key=lambda x: getattr(x, metric, 0),
            reverse=True,
        )
        return sorted_functions[:limit]

    def get_metrics_by_category(self, category: str) -> List[PerformanceMetric]:
        """Get all metrics for a specific category."""
        return [m for m in self.custom_metrics if m.category == category]

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": self.system_monitor.get_metrics_summary(),
            "function_profiles": {},
            "custom_metrics_summary": {},
            "bottlenecks": [],
            "recommendations": [],
        }

        # Function profiles summary
        if self.function_profiles:
            for name, profile in self.function_profiles.items():
                report["function_profiles"][name] = {
                    "call_count": profile.call_count,
                    "total_time": profile.total_time,
                    "avg_time": profile.avg_time,
                    "min_time": profile.min_time,
                    "max_time": profile.max_time,
                    "std_time": profile.std_time,
                    "errors": profile.errors,
                    "error_rate": (
                        profile.errors / profile.call_count
                        if profile.call_count > 0
                        else 0
                    ),
                    "avg_memory_mb": (
                        np.mean(profile.memory_usage) if profile.memory_usage else 0
                    ),
                }

        # Custom metrics summary
        if self.custom_metrics:
            metrics_df = pd.DataFrame(
                [
                    {
                        "name": m.name,
                        "value": m.value,
                        "unit": m.unit,
                        "category": m.category,
                        "timestamp": m.timestamp,
                    }
                    for m in self.custom_metrics
                ]
            )

            for category in metrics_df["category"].unique():
                category_metrics = metrics_df[metrics_df["category"] == category]
                report["custom_metrics_summary"][category] = {
                    "count": len(category_metrics),
                    "mean": category_metrics["value"].mean(),
                    "std": category_metrics["value"].std(),
                    "min": category_metrics["value"].min(),
                    "max": category_metrics["value"].max(),
                }

        # Identify bottlenecks
        bottlenecks = []

        # Slow functions
        slow_functions = self.get_top_functions("avg_time", 5)
        for func in slow_functions:
            if func.avg_time > 1.0:  # Functions taking > 1 second
                bottlenecks.append(
                    {
                        "type": "slow_function",
                        "name": func.name,
                        "value": func.avg_time,
                        "unit": "seconds",
                        "severity": "high" if func.avg_time > 5.0 else "medium",
                    }
                )

        # High memory functions
        for name, profile in self.function_profiles.items():
            if profile.memory_usage:
                avg_memory = np.mean(profile.memory_usage)
                if avg_memory > 100:  # > 100MB average
                    bottlenecks.append(
                        {
                            "type": "high_memory",
                            "name": name,
                            "value": avg_memory,
                            "unit": "MB",
                            "severity": "high" if avg_memory > 500 else "medium",
                        }
                    )

        # High error rate functions
        for name, profile in self.function_profiles.items():
            if profile.call_count > 10:  # At least 10 calls
                error_rate = profile.errors / profile.call_count
                if error_rate > 0.1:  # > 10% error rate
                    bottlenecks.append(
                        {
                            "type": "high_error_rate",
                            "name": name,
                            "value": error_rate,
                            "unit": "rate",
                            "severity": "high" if error_rate > 0.5 else "medium",
                        }
                    )

        report["bottlenecks"] = bottlenecks

        # Generate recommendations
        recommendations = []

        for bottleneck in bottlenecks:
            if bottleneck["type"] == "slow_function":
                recommendations.append(
                    {
                        "type": "optimization",
                        "priority": (
                            "high" if bottleneck["severity"] == "high" else "medium"
                        ),
                        "message": f"Consider optimizing {bottleneck['name']} (avg: {bottleneck['value']:.2f}s)",
                        "suggestions": [
                            "Add caching for expensive computations",
                            "Use more efficient algorithms",
                            "Consider parallelization",
                            "Profile internal function calls",
                        ],
                    }
                )
            elif bottleneck["type"] == "high_memory":
                recommendations.append(
                    {
                        "type": "memory_optimization",
                        "priority": (
                            "high" if bottleneck["severity"] == "high" else "medium"
                        ),
                        "message": f"Consider memory optimization for {bottleneck['name']} (avg: {bottleneck['value']:.1f}MB)",
                        "suggestions": [
                            "Use generators instead of lists",
                            "Implement memory pooling",
                            "Clear unused variables",
                            "Use memory-efficient data structures",
                        ],
                    }
                )
            elif bottleneck["type"] == "high_error_rate":
                recommendations.append(
                    {
                        "type": "error_handling",
                        "priority": "high",
                        "message": f"High error rate in {bottleneck['name']} ({bottleneck['value']:.1%})",
                        "suggestions": [
                            "Add input validation",
                            "Improve error handling",
                            "Add retry logic for transient errors",
                            "Review function logic",
                        ],
                    }
                )

        report["recommendations"] = recommendations

        return report

    def save_profile(self, filename: Optional[str] = None):
        """Save performance profile to file."""
        if filename is None:
            filename = (
                f"performance_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        filepath = self.save_dir / filename

        profile_data = {
            "timestamp": datetime.now().isoformat(),
            "function_profiles": {
                name: {
                    "call_count": profile.call_count,
                    "total_time": profile.total_time,
                    "avg_time": profile.avg_time,
                    "min_time": profile.min_time,
                    "max_time": profile.max_time,
                    "std_time": profile.std_time,
                    "last_called": (
                        profile.last_called.isoformat() if profile.last_called else None
                    ),
                    "memory_usage": profile.memory_usage,
                    "errors": profile.errors,
                }
                for name, profile in self.function_profiles.items()
            },
            "custom_metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp.isoformat(),
                    "category": metric.category,
                    "metadata": metric.metadata,
                }
                for metric in self.custom_metrics
            ],
            "system_metrics": list(self.system_monitor.metrics_history),
        }

        with open(filepath, "w") as f:
            json.dump(profile_data, f, indent=2, default=str)

        apgi_logger.logger.info(f"Performance profile saved to {filepath}")

    def load_profile(self, filepath: Path) -> bool:
        """Load performance profile from file."""
        try:
            with open(filepath, "r") as f:
                profile_data = json.load(f)

            # Load function profiles
            for name, data in profile_data.get("function_profiles", {}).items():
                profile = FunctionProfile(name=name)
                profile.call_count = data["call_count"]
                profile.total_time = data["total_time"]
                profile.avg_time = data["avg_time"]
                profile.min_time = data["min_time"]
                profile.max_time = data["max_time"]
                profile.std_time = data["std_time"]
                profile.last_called = (
                    datetime.fromisoformat(data["last_called"])
                    if data["last_called"]
                    else None
                )
                profile.memory_usage = data["memory_usage"]
                profile.errors = data["errors"]
                self.function_profiles[name] = profile

            # Load custom metrics
            for data in profile_data.get("custom_metrics", []):
                metric = PerformanceMetric(
                    name=data["name"],
                    value=data["value"],
                    unit=data["unit"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    category=data["category"],
                    metadata=data["metadata"],
                )
                self.custom_metrics.append(metric)

            apgi_logger.logger.info(f"Performance profile loaded from {filepath}")
            return True

        except Exception as e:
            apgi_logger.logger.error(f"Error loading performance profile: {e}")
            return False

    def create_visualization_dashboard(self, save_path: Optional[Path] = None) -> Path:
        """Create comprehensive performance visualization dashboard."""
        if save_path is None:
            save_path = (
                self.save_dir
                / f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle("APGI Performance Dashboard", fontsize=16, fontweight="bold")

        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. Top Functions by Total Time
        ax1 = fig.add_subplot(gs[0, 0:2])
        top_functions = self.get_top_functions("total_time", 8)
        if top_functions:
            names = [f.name.split(".")[-1] for f in top_functions]
            times = [f.total_time for f in top_functions]
            bars = ax1.barh(names, times)
            ax1.set_xlabel("Total Time (s)")
            ax1.set_title("Top Functions by Total Time")
            ax1.bar_label(bars, fmt="%.2f")

        # 2. Function Call Counts
        ax2 = fig.add_subplot(gs[0, 2:4])
        if self.function_profiles:
            functions_by_calls = sorted(
                self.function_profiles.values(),
                key=lambda x: x.call_count,
                reverse=True,
            )[:8]
            names = [f.name.split(".")[-1] for f in functions_by_calls]
            calls = [f.call_count for f in functions_by_calls]
            bars = ax2.barh(names, calls)
            ax2.set_xlabel("Call Count")
            ax2.set_title("Function Call Counts")
            ax2.bar_label(bars, fmt="%d")

        # 3. System Metrics Over Time
        ax3 = fig.add_subplot(gs[1, :])
        if self.system_monitor.metrics_history:
            metrics_df = pd.DataFrame(list(self.system_monitor.metrics_history))
            metrics_df.loc[:, "timestamp"] = pd.to_datetime(metrics_df["timestamp"])

            # Plot CPU and Memory
            ax3_twin = ax3.twinx()
            ax3.plot(
                metrics_df["timestamp"],
                metrics_df["cpu_percent"],
                "b-",
                label="CPU %",
                alpha=0.7,
            )
            ax3.plot(
                metrics_df["timestamp"],
                metrics_df["memory_percent"],
                "g-",
                label="Memory %",
                alpha=0.7,
            )
            ax3_twin.plot(
                metrics_df["timestamp"],
                metrics_df["process_memory_rss"] / (1024**3),
                "r-",
                label="Process Memory (GB)",
                alpha=0.7,
            )

            ax3.set_xlabel("Time")
            ax3.set_ylabel("CPU/Memory %")
            ax3_twin.set_ylabel("Process Memory (GB)")
            ax3.set_title("System Metrics Over Time")
            ax3.legend(loc="upper left")
            ax3_twin.legend(loc="upper right")
            ax3.grid(True, alpha=0.3)

        # 4. Performance Distribution
        ax4 = fig.add_subplot(gs[2, 0:2])
        if self.function_profiles:
            avg_times = [
                f.avg_time for f in self.function_profiles.values() if f.avg_time > 0
            ]
            if avg_times:
                ax4.hist(avg_times, bins=20, alpha=0.7, edgecolor="black")
                ax4.set_xlabel("Average Execution Time (s)")
                ax4.set_ylabel("Number of Functions")
                ax4.set_title("Distribution of Function Performance")
                ax4.axvline(
                    np.mean(avg_times),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {np.mean(avg_times):.3f}s",
                )
                ax4.legend()

        # 5. Memory Usage Distribution
        ax5 = fig.add_subplot(gs[2, 2:4])
        memory_usages = []
        function_names = []
        for name, profile in self.function_profiles.items():
            if profile.memory_usage:
                memory_usages.extend(profile.memory_usage)
                function_names.extend([name] * len(profile.memory_usage))

        if memory_usages:
            ax5.hist(memory_usages, bins=20, alpha=0.7, edgecolor="black")
            ax5.set_xlabel("Memory Delta (MB)")
            ax5.set_ylabel("Frequency")
            ax5.set_title("Memory Usage Distribution")
            ax5.axvline(
                np.mean(memory_usages),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(memory_usages):.1f}MB",
            )
            ax5.legend()

        # 6. Categories Performance
        ax6 = fig.add_subplot(gs[3, 0:2])
        if self.custom_metrics:
            categories = {}
            for metric in self.custom_metrics:
                if metric.category not in categories:
                    categories[metric.category] = []
                categories[metric.category].append(metric.value)

            if categories:
                cat_names = list(categories.keys())
                cat_means = [np.mean(categories[cat]) for cat in cat_names]
                cat_stds = [np.std(categories[cat]) for cat in cat_names]

                bars = ax6.bar(
                    cat_names, cat_means, yerr=cat_stds, capsize=5, alpha=0.7
                )
                ax6.set_ylabel("Average Value")
                ax6.set_title("Performance by Category")
                ax6.tick_params(axis="x", rotation=45)

        # 7. Error Rates
        ax7 = fig.add_subplot(gs[3, 2:4])
        error_rates = []
        func_names = []
        for name, profile in self.function_profiles.items():
            if profile.call_count > 0:
                error_rate = profile.errors / profile.call_count
                if error_rate > 0:
                    error_rates.append(error_rate)
                    func_names.append(name.split(".")[-1])

        if error_rates:
            bars = ax7.bar(func_names, error_rates, alpha=0.7, color="red")
            ax7.set_ylabel("Error Rate")
            ax7.set_title("Function Error Rates")
            ax7.tick_params(axis="x", rotation=45)
            ax7.bar_label(bars, fmt="%.2%")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        apgi_logger.logger.info(f"Performance dashboard saved to {save_path}")
        return save_path

    def __del__(self):
        """Cleanup when profiler is destroyed."""
        if hasattr(self, "system_monitor"):
            self.system_monitor.stop_monitoring()


# Global profiler instance
performance_profiler = PerformanceProfiler()


# Convenience decorators
def profile_function(category: str = "general"):
    """Convenience decorator for function profiling."""
    return performance_profiler.profile_function(category)


def profile_context(name: str, category: str = "context"):
    """Convenience context manager for profiling."""
    return performance_profiler.profile_context(name, category)


# Quick profiling functions
def quick_profile(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Quickly profile a function call."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    try:
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        success = False
        error = e
        raise
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss

        profile_result = {
            "duration": end_time - start_time,
            "memory_delta_mb": (end_memory - start_memory) / (1024**2),
            "success": success,
            "error": str(error) if error else None,
        }

    return result, profile_result
