#!/usr/bin/env python3
"""
APGI Comprehensive Performance Dashboard
====================================

Integrated performance monitoring dashboard that combines real-time metrics,
system monitoring, validation results, and performance profiling
into a unified web-based dashboard.

Features:
- Real-time system resource monitoring
- Performance metrics collection and visualization
- Validation protocol results tracking
- Historical performance data analysis
- Interactive charts and alerts
- Export capabilities for reports
"""

import json
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

from dash import dcc, html, callback
import dash.dash as dash

# Try to import APGI components
try:
    from utils.logging_config import apgi_logger
    from utils.performance_profiler import performance_profiler
    from utils.interactive_dashboard import create_dashboard
    from utils.static_dashboard_generator import StaticDashboardGenerator
except ImportError:
    # Fallback for standalone usage
    import logging

    apgi_logger = logging.getLogger(__name__)

    class DummyProfiler:
        def get_current_metrics(self):
            return {}

        def get_performance_history(self):
            return []

    performance_profiler = DummyProfiler()

    def create_dashboard():
        return None

    class StaticDashboardGenerator:
        def generate_system_dashboard(self):
            return "<html><body><h1>Dashboard Not Available</h1></body></html>"


class ComprehensivePerformanceDashboard:
    """Comprehensive performance monitoring dashboard for APGI framework."""

    def __init__(self, port: int = 8050, debug: bool = False):
        """Initialize the comprehensive dashboard."""
        self.port = port
        self.debug = debug
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                "https://codepen.io/chriddyp/pen/oWLvVJ.css",
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
            ],
        )

        # Performance data storage
        self.performance_data = {
            "timestamps": [],
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "network_io": [],
            "validation_results": [],
            "system_metrics": [],
        }

        # Threading for real-time updates
        self.update_thread = None
        self.running = False

        # Initialize components
        self.static_generator = StaticDashboardGenerator()

        apgi_logger.info(
            f"Comprehensive Performance Dashboard initialized on port {port}"
        )

    def create_layout(self) -> html.Div:
        """Create the main dashboard layout."""
        return html.Div(
            [
                # Header
                html.Div(
                    [
                        html.H1(
                            "APGI Performance Dashboard",
                            className="text-center mb-4",
                            style={"color": "#2c3e50", "marginBottom": "20px"},
                        ),
                        html.Hr(style={"border": "1px solid #dee2e6"}),
                    ],
                    className="mb-4",
                ),
                # Metrics Overview Cards
                html.Div(
                    [
                        html.H2("System Overview", className="mb-3"),
                        html.Div([self._create_metric_cards()], className="row"),
                    ],
                    className="mb-4",
                ),
                # Charts Section
                html.Div(
                    [
                        html.H2("Performance Metrics"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="cpu-chart", className="border rounded"
                                        ),
                                        html.H3(
                                            "CPU Usage", className="text-center mt-3"
                                        ),
                                    ],
                                    className="col-md-6 mb-4",
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="memory-chart",
                                            className="border rounded",
                                        ),
                                        html.H3(
                                            "Memory Usage", className="text-center mt-3"
                                        ),
                                    ],
                                    className="col-md-6 mb-4",
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="validation-chart",
                                            className="border rounded",
                                        ),
                                        html.H3(
                                            "Validation Results",
                                            className="text-center mt-3",
                                        ),
                                    ],
                                    className="col-md-6 mb-4",
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="performance-timeline",
                                            className="border rounded",
                                        ),
                                        html.H3(
                                            "Performance Timeline",
                                            className="text-center mt-3",
                                        ),
                                    ],
                                    className="col-md-6 mb-4",
                                ),
                            ],
                            className="row",
                        ),
                    ],
                    className="mb-4",
                ),
                # Control Panel
                html.Div(
                    [html.H2("Controls"), self._create_control_panel()],
                    className="mb-4",
                ),
                # Data Table
                html.Div(
                    [
                        html.H2("Performance Data"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Button(
                                            "Export Data",
                                            id="export-btn",
                                            className="btn btn-primary me-2",
                                            n_clicks=0,
                                        ),
                                        html.Button(
                                            "Clear Data",
                                            id="clear-btn",
                                            className="btn btn-secondary me-2",
                                            n_clicks=0,
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                dcc.Loading(
                                    id="loading-output",
                                    type="default",
                                    children="Ready to monitor performance...",
                                ),
                                html.Div(id="data-table-container"),
                            ]
                        ),
                    ],
                    className="mb-4",
                ),
            ],
            className="container-fluid",
        )

    def _create_metric_cards(self) -> List[html.Div]:
        """Create metric overview cards."""
        current_metrics = self._get_current_metrics()

        cards = []

        # System Status Card
        status_color = (
            "#28a745" if current_metrics.get("cpu_percent", 0) < 80 else "#dc3545"
        )
        cards.append(
            html.Div(
                [
                    html.H4("System Status", className="card-title"),
                    html.P(
                        f"CPU: {current_metrics.get('cpu_percent', 0):.1f}%",
                        className="card-text",
                    ),
                    html.Div(
                        className="progress-bar",
                        style={
                            "width": f"{current_metrics.get('cpu_percent', 0)}%",
                            "backgroundColor": status_color,
                        },
                    ),
                ],
                className="metric-card",
                style={"backgroundColor": "#f8f9fa"},
            )
        )

        # Memory Usage Card
        memory_percent = current_metrics.get("memory_percent", 0)
        memory_color = "#28a745" if memory_percent < 80 else "#dc3545"
        cards.append(
            html.Div(
                [
                    html.H4("Memory Usage", className="card-title"),
                    html.P(f"Memory: {memory_percent:.1f}%", className="card-text"),
                    html.Div(
                        className="progress-bar",
                        style={
                            "width": f"{memory_percent}%",
                            "backgroundColor": memory_color,
                        },
                    ),
                ],
                className="metric-card",
                style={"backgroundColor": "#e3f2fd"},
            )
        )

        # Active Processes Card
        cards.append(
            html.Div(
                [
                    html.H4("Active Processes", className="card-title"),
                    html.P(
                        f"Processes: {current_metrics.get('active_processes', 0)}",
                        className="card-text",
                    ),
                ],
                className="metric-card",
                style={"backgroundColor": "#fff3cd"},
            )
        )

        return cards

    def _create_control_panel(self) -> html.Div:
        """Create control panel for dashboard."""
        return html.Div(
            [
                html.Div(
                    [
                        html.H4("Monitoring Controls"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Update Interval (seconds):"),
                                        dcc.Input(
                                            id="update-interval",
                                            type="number",
                                            value=5,
                                            min=1,
                                            max=60,
                                            step=1,
                                            className="form-control",
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.Div(
                                    [
                                        html.Label("Data Points to Keep:"),
                                        dcc.Input(
                                            id="data-points",
                                            type="number",
                                            value=1000,
                                            min=100,
                                            max=10000,
                                            step=100,
                                            className="form-control",
                                        ),
                                    ],
                                    className="mb-3",
                                ),
                                html.Button(
                                    "Start Monitoring",
                                    id="start-btn",
                                    className="btn btn-success me-2",
                                ),
                                html.Button(
                                    "Stop Monitoring",
                                    id="stop-btn",
                                    className="btn btn-danger me-2",
                                    disabled=True,
                                ),
                                html.Button(
                                    "Generate Report",
                                    id="report-btn",
                                    className="btn btn-info me-2",
                                ),
                            ],
                            className="control-group",
                        ),
                    ]
                )
            ]
        )

    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            import psutil

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk metrics
            disk = psutil.disk_usage("/")

            # Network metrics
            network = psutil.net_io_counters()

            # Process count
            process_count = len(psutil.pids())

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_read_mb": disk.read_bytes / (1024**2),
                "disk_write_mb": disk.write_bytes / (1024**2),
                "network_sent_mb": network.bytes_sent / (1024**2),
                "network_recv_mb": network.bytes_recv / (1024**2),
                "active_processes": process_count,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            apgi_logger.error(f"Error getting system metrics: {e}")
            return {
                "cpu_percent": 0,
                "memory_percent": 0,
                "memory_used_gb": 0,
                "memory_total_gb": 0,
                "disk_read_mb": 0,
                "disk_write_mb": 0,
                "network_sent_mb": 0,
                "network_recv_mb": 0,
                "active_processes": 0,
                "timestamp": datetime.now(),
            }

    def update_metrics(self):
        """Update performance metrics."""
        if not self.running:
            return

        try:
            metrics = self._get_current_metrics()

            # Store metrics
            self.performance_data["timestamps"].append(metrics["timestamp"])
            self.performance_data["cpu_usage"].append(metrics["cpu_percent"])
            self.performance_data["memory_usage"].append(metrics["memory_percent"])
            self.performance_data["disk_io"].append(
                {
                    "read_mb": metrics["disk_read_mb"],
                    "write_mb": metrics["disk_write_mb"],
                }
            )
            self.performance_data["network_io"].append(
                {
                    "sent_mb": metrics["network_sent_mb"],
                    "recv_mb": metrics["network_recv_mb"],
                }
            )
            self.performance_data["system_metrics"].append(
                {
                    "processes": metrics["active_processes"],
                    "memory_used": metrics["memory_used_gb"],
                }
            )

            # Keep only recent data points
            max_points = 1000
            for key in self.performance_data:
                if len(self.performance_data[key]) > max_points:
                    self.performance_data[key] = self.performance_data[key][
                        -max_points:
                    ]

            apgi_logger.debug(
                f"Updated metrics: CPU {metrics['cpu_percent']:.1f}%, "
                f"Memory {metrics['memory_percent']:.1f}%"
            )

        except Exception as e:
            apgi_logger.error(f"Error updating metrics: {e}")

    def create_charts(self) -> Dict[str, Any]:
        """Create chart data from performance metrics."""
        if not self.performance_data["timestamps"]:
            return {}

        # CPU Chart
        cpu_chart = {
            "data": [
                {
                    "x": self.performance_data["timestamps"][-100:],
                    "y": self.performance_data["cpu_usage"][-100:],
                    "type": "scatter",
                    "mode": "lines",
                    "name": "CPU Usage %",
                    "line": {"color": "#FF6B6B", "width": 2},
                }
            ],
            "layout": {
                "title": "CPU Usage Over Time",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "CPU Usage (%)", "range": [0, 100]},
            },
        }

        # Memory Chart
        memory_chart = {
            "data": [
                {
                    "x": self.performance_data["timestamps"][-100:],
                    "y": self.performance_data["memory_usage"][-100:],
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Memory Usage %",
                    "line": {"color": "#1f77b4", "width": 2},
                }
            ],
            "layout": {
                "title": "Memory Usage Over Time",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Memory Usage (%)", "range": [0, 100]},
            },
        }

        # Performance Timeline
        timeline_chart = {
            "data": [
                {
                    "x": self.performance_data["timestamps"][-50:],
                    "y": self.performance_data["cpu_usage"][-50:],
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Performance Timeline",
                    "line": {"color": "#0066CC", "width": 3},
                }
            ],
            "layout": {
                "title": "System Performance Timeline",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Performance Metric"},
            },
        }

        return {
            "cpu-chart": cpu_chart,
            "memory-chart": memory_chart,
            "performance-timeline": timeline_chart,
        }

    def export_data(self, filename: str = None) -> str:
        """Export performance data to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_data_{timestamp}.json"

        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "data_points": len(self.performance_data["timestamps"]),
                "time_range": {
                    "start": (
                        self.performance_data["timestamps"][0].isoformat()
                        if self.performance_data["timestamps"]
                        else None
                    ),
                    "end": (
                        self.performance_data["timestamps"][-1].isoformat()
                        if self.performance_data["timestamps"]
                        else None
                    ),
                },
                "metrics": self.performance_data,
            }

            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            return filename
        except Exception as e:
            apgi_logger.error(f"Error exporting data: {e}")
            return None

    def clear_data(self):
        """Clear all performance data."""
        self.performance_data = {
            "timestamps": [],
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "network_io": [],
            "validation_results": [],
            "system_metrics": [],
        }
        apgi_logger.info("Performance data cleared")

    def generate_report(self) -> Dict[str, Any]:
        """Generate performance summary report."""
        if not self.performance_data["timestamps"]:
            return {"error": "No data available for report generation"}

        try:
            # Calculate statistics
            cpu_avg = sum(self.performance_data["cpu_usage"]) / len(
                self.performance_data["cpu_usage"]
            )
            memory_avg = sum(self.performance_data["memory_usage"]) / len(
                self.performance_data["memory_usage"]
            )

            # Find peaks
            cpu_peak = max(self.performance_data["cpu_usage"])
            memory_peak = max(self.performance_data["memory_usage"])

            # Time range
            if len(self.performance_data["timestamps"]) > 1:
                time_range = (
                    self.performance_data["timestamps"][-1]
                    - self.performance_data["timestamps"][0]
                )
                time_range_seconds = time_range.total_seconds()
            else:
                time_range_seconds = 0

            report = {
                "summary": {
                    "monitoring_duration": str(time_range_seconds),
                    "total_data_points": len(self.performance_data["timestamps"]),
                    "cpu_avg": cpu_avg,
                    "cpu_peak": cpu_peak,
                    "memory_avg": memory_avg,
                    "memory_peak": memory_peak,
                },
                "recommendations": self._generate_recommendations(
                    cpu_avg, memory_avg, cpu_peak, memory_peak
                ),
                "timestamp": datetime.now().isoformat(),
            }

            return report
        except Exception as e:
            apgi_logger.error(f"Error generating report: {e}")
            return {"error": str(e)}

    def _generate_recommendations(
        self, cpu_avg: float, memory_avg: float, cpu_peak: float, memory_peak: float
    ) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []

        if cpu_avg > 80:
            recommendations.append(
                "High CPU usage detected. Consider optimizing algorithms or adding computational resources."
            )

        if memory_avg > 85:
            recommendations.append(
                "High memory usage detected. Consider memory optimization or increasing available RAM."
            )

        if cpu_peak > 95:
            recommendations.append(
                "CPU spikes detected. Investigate background processes or algorithm efficiency."
            )

        if memory_peak > 90:
            recommendations.append(
                "Memory spikes detected. Check for memory leaks or inefficient data structures."
            )

        if not recommendations:
            recommendations.append("System performance is within acceptable ranges.")

        return recommendations

    def setup_callbacks(self, app):
        """Setup dashboard callbacks."""

        @app.callback(
            callback.Output("loading-output", "children"),
            [callback.Input("start-btn", "n_clicks")],
            prevent_initial_call=True,
        )
        def start_monitoring(n_clicks):
            if n_clicks and not self.running:
                self.running = True
                self.update_thread = threading.Thread(
                    target=self._monitoring_loop, daemon=True
                )
                self.update_thread.start()
                return "Starting performance monitoring..."

        @app.callback(
            callback.Output("loading-output", "children"),
            [callback.Input("stop-btn", "n_clicks")],
            prevent_initial_call=True,
        )
        def stop_monitoring(n_clicks):
            if n_clicks and self.running:
                self.running = False
                return "Stopping performance monitoring..."

        @app.callback(
            callback.Output("loading-output", "children"),
            [callback.Input("report-btn", "n_clicks")],
            prevent_initial_call=True,
        )
        def generate_report_callback(n_clicks):
            if n_clicks:
                report = self.generate_report()
                return f"Report generated: {len(report.get('summary', {}).get('recommendations', []))} recommendations found"

        @app.callback(
            callback.Output("loading-output", "children"),
            [callback.Input("clear-btn", "n_clicks")],
            prevent_initial_call=True,
        )
        def clear_data_callback(n_clicks):
            if n_clicks:
                self.clear_data()
                return "Performance data cleared"

        @app.callback(
            [callback.Input("update-interval", "value")], prevent_initial_call=True
        )
        def update_interval(interval):
            return f"Update interval set to {interval} seconds"

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self.update_metrics()
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                apgi_logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)

    def run(self, host: str = "127.0.0.1", debug: bool = False):
        """Run the comprehensive performance dashboard."""
        try:
            # Setup layout
            self.app.layout = self.create_layout()

            # Setup callbacks
            self.setup_callbacks(self.app)

            # Start initial metrics update
            self.update_metrics()

            apgi_logger.info(
                f"Starting Comprehensive Performance Dashboard on http://{host}:{self.port}"
            )

            # Run the app
            self.app.run_server(debug=debug, host=host, port=self.port)

        except Exception as e:
            apgi_logger.error(f"Error running dashboard: {e}")
            raise

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        if not self.performance_data["timestamps"]:
            return {"status": "No data available"}

        return {
            "status": "Active",
            "data_points": len(self.performance_data["timestamps"]),
            "time_range": {
                "start": (
                    self.performance_data["timestamps"][0].isoformat()
                    if self.performance_data["timestamps"]
                    else None
                ),
                "end": (
                    self.performance_data["timestamps"][-1].isoformat()
                    if self.performance_data["timestamps"]
                    else None
                ),
            },
            "current_metrics": self._get_current_metrics(),
            "last_updated": (
                self.performance_data["timestamps"][-1].isoformat()
                if self.performance_data["timestamps"]
                else None
            ),
        }


def main():
    """Main function to run the comprehensive performance dashboard."""
    import argparse

    parser = argparse.ArgumentParser(
        description="APGI Comprehensive Performance Dashboard"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to run dashboard on"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    try:
        dashboard = ComprehensivePerformanceDashboard(port=args.port, debug=args.debug)
        dashboard.run()
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
