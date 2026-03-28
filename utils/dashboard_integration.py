"""
APGI Dashboard Integration Module
=================================

Integrates historical analysis, export features, and monitoring
capabilities into the APGI Validation Framework.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

# APGI imports
try:
    from utils.logging_config import apgi_logger
except ImportError:
    apgi_logger = None

try:
    from utils.historical_dashboard import HistoricalDashboard

    HISTORICAL_AVAILABLE = True
except ImportError:
    HISTORICAL_AVAILABLE = False

try:
    from utils.static_dashboard_generator import StaticDashboardGenerator

    STATIC_AVAILABLE = True
except ImportError:
    STATIC_AVAILABLE = False

try:
    from utils.performance_dashboard import ComprehensivePerformanceDashboard

    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False


class DashboardManager:
    """
    Central manager for all dashboard and historical analysis features.

    Provides unified interface for:
    - Historical data analysis
    - Data export (JSON, CSV, Excel, PDF)
    - Real-time monitoring
    - Static dashboard generation
    """

    def __init__(self, data_dir: str = "data_repository/dashboard_data"):
        """Initialize the dashboard manager."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Dashboard instances
        self.historical_dashboard: Optional[HistoricalDashboard] = None
        self.performance_dashboard: Optional[Any] = None
        self.static_generator: Optional[StaticDashboardGenerator] = None

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []

        # Export history
        self._export_history: List[Dict] = []

        # Initialize available components
        self._init_components()

        if apgi_logger:
            apgi_logger.logger.info("DashboardManager initialized")

    def _init_components(self):
        """Initialize available dashboard components."""
        # Static dashboard generator
        if STATIC_AVAILABLE:
            try:
                self.static_generator = StaticDashboardGenerator(
                    output_dir=str(self.data_dir / "static_dashboards")
                )
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.warning(f"Failed to init static generator: {e}")

        # Historical dashboard
        if HISTORICAL_AVAILABLE:
            try:
                db_path = str(self.data_dir / "historical_data.db")
                self.historical_dashboard = HistoricalDashboard(
                    db_path=db_path, port=8051
                )
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.warning(
                        f"Failed to init historical dashboard: {e}"
                    )

        # Performance dashboard
        if PERFORMANCE_AVAILABLE:
            try:
                self.performance_dashboard = ComprehensivePerformanceDashboard(
                    port=8050
                )
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.warning(
                        f"Failed to init performance dashboard: {e}"
                    )

    def record_validation_result(
        self,
        protocol_number: int,
        protocol_name: str,
        status: str,
        execution_time: float,
        tests_passed: int,
        tests_failed: int,
        error_message: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Record a validation result to the historical database.

        Args:
            protocol_number: Protocol identifier
            protocol_name: Human-readable protocol name
            status: Pass/fail status
            execution_time: Execution time in seconds
            tests_passed: Number of tests passed
            tests_failed: Number of tests failed
            error_message: Optional error details
            metadata: Optional additional data

        Returns:
            True if successfully recorded
        """
        try:
            if self.historical_dashboard:
                # Calculate success rate
                total_tests = tests_passed + tests_failed
                success_rate = (
                    (tests_passed / total_tests * 100) if total_tests > 0 else 0
                )

                # Store in database via historical dashboard
                import sqlite3

                db_path = self.historical_dashboard.db_path

                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO validation_results 
                        (protocol_number, protocol_name, status, execution_time,
                         tests_passed, tests_failed, success_rate, error_message)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            protocol_number,
                            protocol_name,
                            status,
                            execution_time,
                            tests_passed,
                            tests_failed,
                            success_rate,
                            error_message,
                        ),
                    )
                    conn.commit()

                if apgi_logger:
                    apgi_logger.logger.info(
                        f"Recorded validation result for Protocol {protocol_number}: {status}"
                    )

                return True
            else:
                # Fallback: store in JSON file
                self._store_validation_fallback(
                    protocol_number,
                    protocol_name,
                    status,
                    execution_time,
                    tests_passed,
                    tests_failed,
                    error_message,
                    metadata,
                )
                return True

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Failed to record validation result: {e}")
            return False

    def _store_validation_fallback(
        self,
        protocol_number,
        protocol_name,
        status,
        execution_time,
        tests_passed,
        tests_failed,
        error_message,
        metadata,
    ):
        """Store validation result in JSON when database unavailable."""
        fallback_file = self.data_dir / "validation_history.json"

        history = []
        if fallback_file.exists():
            try:
                with open(fallback_file, "r") as f:
                    history = json.load(f)
            except (json.JSONDecodeError, OSError):
                history = []

        history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "protocol_number": protocol_number,
                "protocol_name": protocol_name,
                "status": status,
                "execution_time": execution_time,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "error_message": error_message,
                "metadata": metadata or {},
            }
        )

        # Keep only last 1000 entries
        history = history[-1000:]

        with open(fallback_file, "w") as f:
            json.dump(history, f, indent=2, default=str)

    def export_historical_data(
        self,
        format_type: str = "json",
        data_type: str = "all",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Export historical data to specified format.

        Args:
            format_type: Export format (json, csv, excel, pdf)
            data_type: Type of data to export (all, system, validation, performance)
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            output_dir: Optional output directory

        Returns:
            Path to exported file or None if failed
        """
        try:
            # Use historical dashboard if available
            if self.historical_dashboard:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"apgi_export_{data_type}_{timestamp}.{format_type}"

                if output_dir:
                    output_path = Path(output_dir) / filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_path = self.data_dir / "exports" / filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                # Export using historical dashboard
                success = self.historical_dashboard.export_historical_data(
                    format_type=format_type,
                    data_type=data_type,
                    start_date=start_date,
                    end_date=end_date,
                    filename=filename,
                )

                if success:
                    # Move to correct location if needed
                    source_path = Path("exports") / filename
                    if source_path.exists():
                        source_path.rename(output_path)

                    # Record export in history
                    self._export_history.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "format": format_type,
                            "data_type": data_type,
                            "filename": str(output_path),
                            "start_date": start_date,
                            "end_date": end_date,
                        }
                    )

                    if apgi_logger:
                        apgi_logger.logger.info(f"Exported data to {output_path}")

                    return str(output_path)

            # Fallback: export from JSON files
            return self._export_from_fallback(
                format_type, data_type, start_date, end_date
            )

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Export failed: {e}")
            return None

    def _export_from_fallback(
        self,
        format_type: str,
        data_type: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Optional[str]:
        """Export data from fallback JSON storage."""
        try:
            fallback_file = self.data_dir / "validation_history.json"
            if not fallback_file.exists():
                return None

            with open(fallback_file, "r") as f:
                data = json.load(f)

            # Filter by date if specified
            if start_date or end_date:
                filtered = []
                for item in data:
                    item_date = item.get("timestamp", "")
                    if start_date and item_date < start_date:
                        continue
                    if end_date and item_date > end_date:
                        continue
                    filtered.append(item)
                data = filtered

            # Generate output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = (
                self.data_dir / "exports" / f"apgi_export_{timestamp}.{format_type}"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type == "json":
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
            elif format_type == "csv":
                import csv

                if data:
                    with open(output_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
            else:
                # Excel/PDF require additional dependencies
                return None

            return str(output_path)

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Fallback export failed: {e}")
            return None

    def get_historical_summary(
        self, days: int = 30, protocol_filter: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Get summary of historical validation data.

        Args:
            days: Number of days to look back
            protocol_filter: Optional list of protocol numbers to include

        Returns:
            Dictionary with summary statistics
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            if self.historical_dashboard:
                data = self.historical_dashboard.get_historical_data(
                    "validation_results", start_date=cutoff_date
                )
            else:
                # Fallback to JSON
                fallback_file = self.data_dir / "validation_history.json"
                if fallback_file.exists():
                    with open(fallback_file, "r") as f:
                        all_data = json.load(f)
                    data = [
                        d for d in all_data if d.get("timestamp", "") >= cutoff_date
                    ]
                else:
                    data = []

            # Apply protocol filter
            if protocol_filter:
                data = [d for d in data if d.get("protocol_number") in protocol_filter]

            # Calculate statistics
            if not data:
                return {
                    "period_days": days,
                    "total_runs": 0,
                    "pass_rate": 0,
                    "avg_execution_time": 0,
                    "protocol_breakdown": {},
                }

            total = len(data)
            passed = sum(
                1
                for d in data
                if d.get("status", "").lower() in ["pass", "passed", "success"]
            )
            exec_times = [
                d.get("execution_time", 0) for d in data if d.get("execution_time")
            ]

            # Protocol breakdown
            by_protocol: Dict[int, Dict] = {}
            for d in data:
                pnum = d.get("protocol_number", 0)
                if pnum not in by_protocol:
                    by_protocol[pnum] = {"runs": 0, "passed": 0, "failed": 0}
                by_protocol[pnum]["runs"] += 1
                if d.get("status", "").lower() in ["pass", "passed", "success"]:
                    by_protocol[pnum]["passed"] += 1
                else:
                    by_protocol[pnum]["failed"] += 1

            return {
                "period_days": days,
                "total_runs": total,
                "pass_rate": (passed / total * 100) if total > 0 else 0,
                "avg_execution_time": sum(exec_times) / len(exec_times)
                if exec_times
                else 0,
                "protocol_breakdown": {
                    f"Protocol {k}": v for k, v in by_protocol.items()
                },
            }

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Failed to get historical summary: {e}")
            return {"error": str(e)}

    def generate_static_dashboards(self) -> List[str]:
        """Generate all static HTML dashboards."""
        if self.static_generator:
            return self.static_generator.generate_all_dashboards()
        return []

    def start_realtime_monitoring(self, interval_seconds: int = 30):
        """
        Start real-time monitoring thread.

        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self._monitoring_active:
            return

        self._monitoring_active = True

        def monitor_loop():
            while self._monitoring_active:
                try:
                    # Collect metrics
                    self._collect_system_metrics()

                    # Notify callbacks
                    for callback in self._callbacks:
                        try:
                            callback()
                        except Exception:
                            pass

                    time.sleep(interval_seconds)
                except Exception as e:
                    if apgi_logger:
                        apgi_logger.logger.error(f"Monitoring error: {e}")
                    time.sleep(interval_seconds)

        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()

        if apgi_logger:
            apgi_logger.logger.info(
                f"Started real-time monitoring (interval: {interval_seconds}s)"
            )

    def _collect_system_metrics(self):
        """Collect current system metrics."""
        try:
            import psutil

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3),
                "disk_usage_percent": psutil.disk_usage("/").percent,
                "network_connections": len(psutil.net_connections()),
                "load_average": psutil.getloadavg()[0]
                if hasattr(psutil, "getloadavg")
                else 0,
            }

            # Store in historical dashboard
            if self.historical_dashboard:
                import sqlite3

                db_path = self.historical_dashboard.db_path
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO system_metrics 
                        (cpu_percent, memory_percent, memory_used_gb, disk_usage_percent, 
                         network_connections, load_average)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            metrics["cpu_percent"],
                            metrics["memory_percent"],
                            metrics["memory_used_gb"],
                            metrics["disk_usage_percent"],
                            metrics["network_connections"],
                            metrics["load_average"],
                        ),
                    )
                    conn.commit()

        except ImportError:
            pass  # psutil not available
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.warning(f"Failed to collect system metrics: {e}")

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self._monitoring_active = False
        if apgi_logger:
            apgi_logger.logger.info("Stopped real-time monitoring")

    def add_monitoring_callback(self, callback: Callable):
        """Add a callback function to be called during monitoring."""
        self._callbacks.append(callback)

    def get_export_history(self) -> List[Dict]:
        """Get history of all exports performed."""
        return self._export_history.copy()

    def run_historical_dashboard(self, host: str = "127.0.0.1"):
        """Start the interactive historical dashboard server."""
        if self.historical_dashboard:
            self.historical_dashboard.run(host=host)
        else:
            raise RuntimeError("Historical dashboard not available")

    def run_performance_dashboard(self, host: str = "127.0.0.1"):
        """Start the performance dashboard server."""
        if self.performance_dashboard:
            self.performance_dashboard.run(host=host, debug=False, use_reloader=False)
        else:
            raise RuntimeError("Performance dashboard not available")


# Singleton instance
_dashboard_manager: Optional[DashboardManager] = None


def get_dashboard_manager() -> DashboardManager:
    """Get or create the singleton DashboardManager instance."""
    global _dashboard_manager
    if _dashboard_manager is None:
        _dashboard_manager = DashboardManager()
    return _dashboard_manager


def record_validation(
    protocol_number: int,
    protocol_name: str,
    status: str,
    execution_time: float,
    tests_passed: int,
    tests_failed: int,
    **kwargs,
) -> bool:
    """Convenience function to record a validation result."""
    manager = get_dashboard_manager()
    return manager.record_validation_result(
        protocol_number=protocol_number,
        protocol_name=protocol_name,
        status=status,
        execution_time=execution_time,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        **kwargs,
    )


def export_data(
    format_type: str = "json",
    data_type: str = "all",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[str]:
    """Convenience function to export historical data."""
    manager = get_dashboard_manager()
    return manager.export_historical_data(
        format_type=format_type,
        data_type=data_type,
        start_date=start_date,
        end_date=end_date,
    )


def get_summary(days: int = 30) -> Dict[str, Any]:
    """Convenience function to get historical summary."""
    manager = get_dashboard_manager()
    return manager.get_historical_summary(days=days)


if __name__ == "__main__":
    # Demo usage
    print("APGI Dashboard Integration Module")
    print("=================================")

    manager = get_dashboard_manager()

    # Record some sample data
    record_validation(
        protocol_number=1,
        protocol_name="Basic Validation",
        status="pass",
        execution_time=2.5,
        tests_passed=45,
        tests_failed=5,
    )

    # Get summary
    summary = get_summary(days=7)
    print("\nHistorical Summary (last 7 days):")
    print(json.dumps(summary, indent=2))

    # Generate static dashboards
    print("\nGenerating static dashboards...")
    dashboards = manager.generate_static_dashboards()
    for d in dashboards:
        print(f"  Generated: {d}")

    print("\nDashboard integration module ready!")
