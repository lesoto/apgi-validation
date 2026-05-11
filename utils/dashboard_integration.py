#!/usr/bin/env python3
"""
LEVEL DESIGNATION: Level 2 (information-theoretic)

Bridge to Level 1
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

# Import the new output file pattern matcher
try:
    from utils.output_file_patterns import OutputFilePatternMatcher

    PATTERN_MATCHER_AVAILABLE = True
except ImportError:
    PATTERN_MATCHER_AVAILABLE = False

# Data protection functions available but not imported as they're unused in this module
DATA_PROTECTION_AVAILABLE = False


class DashboardManager:
    """
    Central manager for all dashboard and historical analysis features.

    Provides unified interface for:
    - Historical data analysis
    - Data export (JSON, CSV, Excel, PDF)
    - Performance monitoring
    - Real-time status updates
    - Output file pattern detection
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        data_dir: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the dashboard manager.

        Args:
            project_root: Root directory of the APGI project
            data_dir: Alternative data directory path (for testing)
        """
        self.project_root = project_root or Path(__file__).parent.parent

        # Handle data_dir parameter for backward compatibility with tests
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = self.project_root / "data_repository/dashboard_data"

        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback to temp if no permissions
            import tempfile

            self.data_dir = Path(tempfile.gettempdir()) / "apgi_dashboard_data"
            self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database path
        if "db_path" in kwargs:
            self.db_path = Path(kwargs["db_path"])
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.db_path = Path(self.project_root / "data_repository/historical_data.db")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize pattern matcher if available
        if PATTERN_MATCHER_AVAILABLE:
            self.pattern_matcher = OutputFilePatternMatcher(self.project_root)
        else:
            self.pattern_matcher = None

        # Initialize other components if available
        if HISTORICAL_AVAILABLE:
            self.historical_dashboard = HistoricalDashboard(str(self.db_path))
        else:
            self.historical_dashboard = None

        if STATIC_AVAILABLE:
            self.static_generator = StaticDashboardGenerator(str(self.project_root))
        else:
            self.static_generator = None

        if PERFORMANCE_AVAILABLE:
            try:
                self.performance_dashboard = ComprehensivePerformanceDashboard(port=8050, debug=False)
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.warning(f"Failed to init performance dashboard: {e}")
                self.performance_dashboard = None
        else:
            self.performance_dashboard = None

        # Thread-safe data collection
        self.collection_lock = threading.Lock()
        self.collection_thread = None
        self.is_collecting = False

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        # Export history
        self._export_history: List[Dict] = []

        if apgi_logger:
            apgi_logger.logger.info("DashboardManager initialized")

    def _init_database(self):
        """Initialize SQLite database for historical data storage."""
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # System metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        cpu_percent REAL,
                        memory_percent REAL,
                        memory_used_gb REAL,
                        disk_usage_percent REAL,
                        network_connections INTEGER,
                        load_average REAL
                    )
                """)
                # Metadata table for validation results
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        data_json TEXT,
                        validation_type TEXT,
                        protocol_name TEXT,
                        status TEXT,
                        details TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            if apgi_logger:
                apgi_logger.error(f"Failed to initialize database: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_monitoring()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=1.0)
        self._callbacks.clear()
        self._export_history.clear()

    def add_monitoring_callback(self, callback: Callable) -> None:
        """Add a monitoring callback."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def _trigger_callbacks(self, data: Any) -> None:
        """Trigger all monitoring callbacks."""
        for callback in self._callbacks:
            try:
                callback(data)
            except Exception as e:
                # Log error for debugging purposes
                import logging

                logging.getLogger(__name__).debug(f"Callback execution failed: {e}")
                pass

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
                success_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

                # Store in database via historical dashboard
                import os
                import sqlite3

                db_path = self.historical_dashboard.db_path

                # Ensure database directory exists
                os.makedirs(os.path.dirname(db_path), exist_ok=True)

                try:
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            """
                            SELECT COUNT(*) FROM validation_results 
                            WHERE metadata IS NOT NULL 
                            AND timestamp >= ?
                            """,
                            (datetime.now().isoformat(),),
                        )
                        cursor.execute(
                            """
                            INSERT INTO validation_results 
                            (protocol_number, protocol_name, status, execution_time,
                             tests_passed, tests_failed, success_rate, error_message, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                                json.dumps(metadata) if metadata else None,
                            ),
                        )
                except Exception as db_error:
                    if apgi_logger:
                        apgi_logger.error(f"Failed to initialize database: {db_error}")
                    else:
                        print(f"Failed to initialize database: {db_error}")
                    return False

                if apgi_logger:
                    apgi_logger.logger.info(f"Recorded validation result for Protocol {protocol_number}: {status}")

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

                # Get historical data for the last 7 days
                history = self.historical_dashboard.get_historical_data(
                    table="dashboard_data",
                    start_date=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                    end_date=datetime.now().strftime("%Y-%m-%d"),
                )
                # Filter to only validation results
                validation_history = [item for item in history if item.get("protocol_name", "").startswith("Protocol")]
                if validation_history:
                    # Sort by timestamp (most recent first)
                    validation_history.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]), reverse=True)

                    # Write to file based on format_type
                    if format_type == "json":
                        import json

                        with open(output_path, "w") as f:
                            json.dump(validation_history, f, indent=2)
                    elif format_type == "csv":
                        import csv

                        with open(output_path, "w", newline="") as f:
                            if validation_history:
                                writer = csv.DictWriter(f, fieldnames=validation_history[0].keys())
                                writer.writeheader()
                                writer.writerows(validation_history)
                    elif format_type == "excel":
                        import pandas as pd

                        df = pd.DataFrame(validation_history)
                        df.to_excel(output_path, index=False)

                    return str(output_path)
                return None
                # Filter to only validation results
                validation_history = [item for item in history if item.get("protocol_name", "").startswith("Protocol")]
                if validation_history:
                    # Sort by timestamp (most recent first)
                    validation_history.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]), reverse=True)
                return validation_history
                # Filter to only validation results
                validation_history = [item for item in history if item.get("protocol_name", "").startswith("Protocol")]
                if validation_history:
                    # Sort by timestamp (most recent first)
                    validation_history.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]), reverse=True)
                return validation_history
                # Filter to only validation results
                validation_history = [item for item in history if item.get("protocol_name", "").startswith("Protocol")]
                if validation_history:
                    # Sort by timestamp (most recent first)
                    validation_history.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]), reverse=True)
                return validation_history
                # Filter to only validation results
                validation_history = [item for item in history if item.get("protocol_name", "").startswith("Protocol")]
                if validation_history:
                    # Sort by timestamp (most recent first)
                    validation_history.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]), reverse=True)
                return validation_history
                # Filter to only validation results
                validation_history = [item for item in history if item.get("protocol_name", "").startswith("Protocol")]
                if validation_history:
                    # Sort by timestamp (most recent first)
                    validation_history.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]), reverse=True)
                return validation_history
                # Filter to only validation results
                validation_history = [item for item in history if item.get("protocol_name", "").startswith("Protocol")]
                if validation_history:
                    # Sort by timestamp (most recent first)
                    validation_history.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]), reverse=True)
                return validation_history
                # Filter to only validation results
                validation_history = [item for item in history if item.get("protocol_name", "").startswith("Protocol")]
                if validation_history:
                    # Sort by timestamp (most recent first)
                    validation_history.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]), reverse=True)
                return validation_history
                # Filter to only validation results
                validation_history = [item for item in history if item.get("protocol_name", "").startswith("Protocol")]
                if validation_history:
                    # Sort by timestamp (most recent first)
                    validation_history.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]), reverse=True)
                return validation_history

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
            return self._export_from_fallback(format_type, data_type, start_date, end_date)

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
            output_path = self.data_dir / "exports" / f"apgi_export_{timestamp}.{format_type}"
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

    def get_historical_summary(self, days: int = 30, protocol_filter: Optional[List[int]] = None) -> Dict[str, Any]:
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
                data = self.historical_dashboard.get_historical_data("validation_results", start_date=cutoff_date)
            else:
                # Fallback to JSON
                fallback_file = self.data_dir / "validation_history.json"
                if fallback_file.exists():
                    with open(fallback_file, "r") as f:
                        all_data = json.load(f)
                    data = [d for d in all_data if d.get("timestamp", "") >= cutoff_date]
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
                    "pass_rate": 0.0,  # nosec B105
                    "avg_execution_time": 0,
                    "protocol_breakdown": {},
                }

            total_runs = len(data)
            passed_runs = sum(1 for d in data if d.get("status") == "pass")
            pass_rate = passed_runs / total_runs if total_runs > 0 else 0.0  # nosec B105  # nosec B105

            exec_times = [d.get("execution_time", 0) for d in data]
            avg_execution_time = sum(exec_times) / len(exec_times) if exec_times else 0

            # Protocol breakdown
            by_protocol = {}
            for d in data:
                protocol_num = d.get("protocol_number", "unknown")
                if protocol_num not in by_protocol:
                    by_protocol[protocol_num] = {"total": 0, "passed": 0}
                by_protocol[protocol_num]["total"] += 1
                if d.get("status") == "pass":
                    by_protocol[protocol_num]["passed"] += 1

            return {
                "period_days": days,
                "total_runs": total_runs,
                "pass_rate": pass_rate,
                "avg_execution_time": avg_execution_time,
                "protocol_breakdown": {f"Protocol {k}": v for k, v in by_protocol.items()},
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
                        except Exception as e:
                            # Log error for debugging purposes
                            import logging

                            logging.getLogger(__name__).debug(f"Monitoring callback failed: {e}")
                            pass

                    time.sleep(interval_seconds)
                except Exception as e:
                    if apgi_logger:
                        apgi_logger.logger.error(f"Monitoring error: {e}")
                    time.sleep(interval_seconds)

        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()

        if apgi_logger:
            apgi_logger.logger.info(f"Started real-time monitoring (interval: {interval_seconds}s)")

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
                "load_average": (psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0),
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

    def generate_static_dashboard(self, data: Dict[str, Any], dashboard_name: str) -> Optional[str]:
        """Generate a static dashboard with the provided data."""
        if self.static_generator:
            try:
                # Call generate_dashboard with the provided data and dashboard name
                output_path = self.static_generator.generate_dashboard(data, dashboard_name)
                return output_path
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Failed to generate static dashboard: {e}")
                return None
        else:
            raise RuntimeError("Static dashboard generator not available")

    def store_historical_data(self, data: Dict[str, Any]) -> bool:
        """Store historical data in the database."""
        if self.historical_dashboard:
            try:
                # Call the store_data method on the historical dashboard if it exists
                if hasattr(self.historical_dashboard, "store_data"):
                    self.historical_dashboard.store_data(data)
                else:
                    # Fallback: store it in the database directly
                    import sqlite3

                    db_path = self.historical_dashboard.db_path
                    timestamp = data.get("timestamp", datetime.now().isoformat())

                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()

                        # Store in a generic historical_data table
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS historical_data (
                                timestamp TEXT PRIMARY KEY,
                                data_json TEXT,
                                created_at TEXT DEFAULT CURRENT_TIMESTAMP
                            )
                            """)

                        cursor.execute(
                            "INSERT OR REPLACE INTO historical_data (timestamp, data_json) VALUES (?, ?)",
                            (timestamp, json.dumps(data, default=str)),
                        )
                        conn.commit()

                if apgi_logger:
                    apgi_logger.logger.info(f"Stored historical data for {data.get('timestamp', 'unknown')}")

                return True
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Failed to store historical data: {e}")
                return False
        else:
            raise RuntimeError("Historical dashboard not available")

    def update_performance_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Update performance metrics."""
        if self.performance_dashboard:
            try:
                # Call the update_metrics method on the performance dashboard if it exists
                if hasattr(self.performance_dashboard, "update_metrics"):
                    self.performance_dashboard.update_metrics(metrics)
                else:
                    # Fallback: store metrics in the database
                    import sqlite3

                    db_path = (
                        self.historical_dashboard.db_path
                        if self.historical_dashboard
                        else str(self.data_dir / "historical_data.db")
                    )
                    timestamp = metrics.get("timestamp", datetime.now().isoformat())

                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.cursor()

                        # Create performance_metrics table if it doesn't exist
                        cursor.execute("""
                            CREATE TABLE IF NOT EXISTS performance_metrics (
                                timestamp TEXT PRIMARY KEY,
                                cpu_usage REAL,
                                memory_usage INTEGER,
                                response_time REAL,
                                metrics_json TEXT,
                                created_at TEXT DEFAULT CURRENT_TIMESTAMP
                            )
                            """)

                        cursor.execute(
                            "INSERT OR REPLACE INTO performance_metrics (timestamp, cpu_usage, memory_usage, response_time, metrics_json) VALUES (?, ?, ?, ?, ?)",
                            (
                                timestamp,
                                metrics.get("cpu_usage", 0),
                                metrics.get("memory_usage", 0),
                                metrics.get("response_time", 0),
                                json.dumps(metrics, default=str),
                            ),
                        )
                        conn.commit()

                if apgi_logger:
                    apgi_logger.logger.info(f"Updated performance metrics for {timestamp}")

                return True
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Failed to update performance metrics: {e}")
                return False
        else:
            raise RuntimeError("Performance dashboard not available")

    def start_monitoring(self, interval: float = 30.0) -> None:
        """Start monitoring with backward compatibility."""
        if self._monitoring_active:
            return

        self._monitoring_active = True

        def monitor_loop():
            while self._monitoring_active:
                try:
                    self._collect_system_metrics()

                    for callback in self._callbacks:
                        try:
                            callback()
                        except Exception as e:
                            # Log error for debugging purposes
                            import logging

                            logging.getLogger(__name__).debug(f"Monitoring callback failed: {e}")
                            pass

                    time.sleep(interval)
                except Exception as e:
                    if apgi_logger:
                        apgi_logger.logger.error(f"Monitoring error: {e}")
                    time.sleep(interval)

        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()

        if apgi_logger:
            apgi_logger.logger.info(f"Started monitoring (interval: {interval}s)")

    def stop_monitoring_thread(self) -> None:
        """Stop monitoring thread with backward compatibility."""
        self._monitoring_active = False
        if apgi_logger:
            apgi_logger.logger.info("Stopped monitoring thread")

    def register_callback(self, callback: Callable) -> None:
        """Register a callback with backward compatibility."""
        self.add_monitoring_callback(callback)

    def export_data(self, data: Dict[str, Any], format: str, filename: str) -> Path:
        """Export data with backward compatibility."""
        if format not in ["json", "csv"]:
            raise ValueError(f"Unsupported format: {format}")

        if not data:
            raise ValueError("Data cannot be empty")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{filename}_{timestamp}.{format}"
        output_path = self.data_dir / "exports" / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "csv":
            import csv

            if data and isinstance(data, dict):
                # Flatten nested dict for CSV
                flattened_data = {}
                for key, value in data.items():
                    if isinstance(value, (str, int, float, bool)):
                        flattened_data[key] = value
                    elif isinstance(value, list) and value and isinstance(value[0], (str, int, float, bool)):
                        flattened_data[key] = ", ".join(str(v) for v in value)
                    else:
                        flattened_data[key] = str(value)

                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(flattened_data.keys()))
                    writer.writeheader()
                    writer.writerow(flattened_data)
            else:
                raise ValueError("Data must be a dictionary for CSV export")

        # Record in export history
        self._export_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "format": format,
                "filename": output_filename,
                "path": output_path,
            }
        )

        return Path(output_path)

    def _trigger_callbacks_v2(self, data: Dict[str, Any]) -> None:
        """Trigger all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(data)
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Callback error: {e}")

    def add_monitoring_callback_v2(self, callback: Callable) -> None:
        """Add a callback function to be called during monitoring."""
        self._callbacks.append(callback)

    def get_export_history(self) -> List[Dict]:
        """Get history of all exports performed."""
        return self._export_history.copy()

    def cleanup_dashboard(self) -> None:
        """Clean up dashboard resources."""
        self.stop_monitoring()
        self._callbacks = []
        self._export_history = []

        # Close database connections to prevent resource locking
        if self.historical_dashboard:
            try:
                # Force close any SQLite connections
                import sqlite3

                if hasattr(self.historical_dashboard, "db_path"):
                    db_path = self.historical_dashboard.db_path
                    # Connect and disconnect to ensure clean closure
                    conn = sqlite3.connect(db_path)
                    conn.close()
            except Exception as e:
                # Log cleanup errors for debugging
                import logging

                logging.getLogger(__name__).debug(f"Database cleanup error: {e}")
                pass  # Ignore cleanup errors

        if self.performance_dashboard:
            try:
                # Close any performance dashboard connections
                if hasattr(self.performance_dashboard, "close"):
                    self.performance_dashboard.close()
            except Exception as e:
                # Log cleanup errors for debugging
                import logging

                logging.getLogger(__name__).debug(f"Performance dashboard cleanup error: {e}")
                pass  # Ignore cleanup errors

        if apgi_logger:
            apgi_logger.logger.info("DashboardManager cleaned up")

    def run_historical_dashboard(self, host: str = "127.0.0.1"):
        """Start the interactive historical dashboard server."""
        if self.historical_dashboard:
            self.historical_dashboard.run(host=host)
        else:
            raise RuntimeError("Historical dashboard not available")

    def run_performance_dashboard(self, host: str = "127.0.0.1"):
        """Start the performance dashboard server."""
        if self.performance_dashboard:
            self.performance_dashboard.run(host=host)
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
