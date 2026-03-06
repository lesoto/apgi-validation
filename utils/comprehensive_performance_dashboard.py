#!/usr/bin/env python3
"""
APGI Theory Framework - Comprehensive Performance Dashboard
========================================================

Interactive web-based dashboard for real-time performance monitoring,
validation results tracking, and system metrics visualization using Dash.
"""

import json
import threading
import time
from datetime import datetime
from typing import List

try:
    import dash
    from dash import html, dcc, Input, Output
    import plotly.graph_objects as go

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from utils.logging_config import apgi_logger
except ImportError:
    apgi_logger = None


class ComprehensivePerformanceDashboard:
    """Comprehensive performance monitoring dashboard with proper error handling."""

    def __init__(self, port: int = 8050, debug: bool = False):
        """
        Initialize the dashboard.

        Args:
            port: Port to run the dashboard on
            debug: Enable debug mode
        """
        if not DASH_AVAILABLE:
            raise ImportError(
                "Dash is required for the performance dashboard. Install with: pip install dash plotly"
            )

        self.port = port
        self.debug = debug
        self.app = dash.Dash(__name__, title="APGI Performance Dashboard")

        # Data storage
        self.system_metrics = []
        self.validation_results = []
        self.performance_metrics = []

        # Threading lock for data access
        self.data_lock = threading.Lock()

        # Background monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False

        self._setup_layout()
        self._setup_callbacks()

        if apgi_logger:
            apgi_logger.logger.info(f"Initialized performance dashboard on port {port}")

    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div(
            [
                html.H1(
                    "APGI Comprehensive Performance Dashboard",
                    style={"textAlign": "center", "color": "#2c3e50"},
                ),
                # Navigation tabs
                dcc.Tabs(
                    id="main-tabs",
                    value="overview",
                    children=[
                        dcc.Tab(label="System Overview", value="overview"),
                        dcc.Tab(label="Performance Metrics", value="performance"),
                        dcc.Tab(label="Validation Results", value="validation"),
                        dcc.Tab(label="Error Monitoring", value="errors"),
                    ],
                ),
                # Content container
                html.Div(id="tab-content"),
                # Interval for updates
                dcc.Interval(
                    id="update-interval",
                    interval=5000,  # Update every 5 seconds
                    n_intervals=0,
                ),
                # Error display
                html.Div(id="error-display", style={"color": "red", "margin": "10px"}),
            ]
        )

    def _setup_callbacks(self):
        """Set up all dashboard callbacks with proper error handling."""

        @self.app.callback(
            Output("tab-content", "children"), Input("main-tabs", "value")
        )
        def update_tab_content(tab_value):
            """Update tab content with error handling."""
            try:
                if tab_value == "overview":
                    return self._create_overview_tab()
                elif tab_value == "performance":
                    return self._create_performance_tab()
                elif tab_value == "validation":
                    return self._create_validation_tab()
                elif tab_value == "errors":
                    return self._create_error_tab()
                else:
                    return html.Div("Unknown tab selected")
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Error updating tab content: {e}")
                return html.Div(
                    [
                        html.H3("Error Loading Tab Content", style={"color": "red"}),
                        html.P(f"An error occurred: {str(e)}"),
                        html.P("Please check the application logs for more details."),
                    ]
                )

        @self.app.callback(
            Output("error-display", "children"), Input("update-interval", "n_intervals")
        )
        def update_error_display(n_intervals):
            """Update error display with proper error handling."""
            try:
                # Collect current system metrics
                with self.data_lock:
                    self._collect_system_metrics()

                # Check for any system errors
                errors = self._check_for_errors()

                if errors:
                    return html.Div(
                        [
                            html.H4("System Alerts", style={"color": "orange"}),
                            html.Ul([html.Li(error) for error in errors]),
                        ]
                    )
                else:
                    return ""

            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Error updating error display: {e}")
                return html.Div(
                    [
                        html.Span(
                            "⚠️ Error monitoring system metrics", style={"color": "red"}
                        )
                    ]
                )

        @self.app.callback(
            [
                Output("system-metrics-graph", "figure"),
                Output("memory-usage-graph", "figure"),
            ],
            Input("update-interval", "n_intervals"),
        )
        def update_system_graphs(n_intervals):
            """Update system monitoring graphs with error handling."""
            try:
                with self.data_lock:
                    metrics = self.system_metrics[-20:]  # Last 20 data points

                if not metrics:
                    # Return empty figures if no data
                    empty_fig = go.Figure()
                    empty_fig.update_layout(title="No data available")
                    return empty_fig, empty_fig

                # CPU and Memory usage over time
                timestamps = [m.get("timestamp") for m in metrics]
                cpu_usage = [m.get("cpu_percent", 0) for m in metrics]
                memory_usage = [m.get("memory_percent", 0) for m in metrics]

                cpu_fig = go.Figure()
                cpu_fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=cpu_usage,
                        mode="lines+markers",
                        name="CPU Usage",
                        line=dict(color="blue"),
                    )
                )
                cpu_fig.update_layout(
                    title="CPU Usage Over Time",
                    xaxis_title="Time",
                    yaxis_title="CPU %",
                    yaxis_range=[0, 100],
                )

                memory_fig = go.Figure()
                memory_fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=memory_usage,
                        mode="lines+markers",
                        name="Memory Usage",
                        line=dict(color="green"),
                    )
                )
                memory_fig.update_layout(
                    title="Memory Usage Over Time",
                    xaxis_title="Time",
                    yaxis_title="Memory %",
                    yaxis_range=[0, 100],
                )

                return cpu_fig, memory_fig

            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Error updating system graphs: {e}")

                # Return error figures
                error_fig = go.Figure()
                error_fig.update_layout(title=f"Error loading data: {str(e)}")
                return error_fig, error_fig

        @self.app.callback(
            Output("performance-table", "data"), Input("update-interval", "n_intervals")
        )
        def update_performance_table(n_intervals):
            """Update performance metrics table with error handling."""
            try:
                with self.data_lock:
                    perf_data = self.performance_metrics[-10:]  # Last 10 entries

                if not perf_data:
                    return []

                return perf_data

            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Error updating performance table: {e}")
                return [
                    {
                        "metric": "Error",
                        "value": f"Failed to load performance data: {str(e)}",
                        "timestamp": datetime.now().isoformat(),
                    }
                ]

        @self.app.callback(
            Output("validation-results-display", "children"),
            Input("update-interval", "n_intervals"),
        )
        def update_validation_results(n_intervals):
            """Update validation results display with error handling."""
            try:
                with self.data_lock:
                    results = self.validation_results[-5:]  # Last 5 results

                if not results:
                    return html.Div("No validation results available yet.")

                result_items = []
                for result in results:
                    result_items.append(
                        html.Div(
                            [
                                html.H4(
                                    f"Validation Run: {result.get('timestamp', 'Unknown')}"
                                ),
                                html.P(f"Status: {result.get('status', 'Unknown')}"),
                                html.Pre(json.dumps(result.get("data", {}), indent=2)),
                                html.Hr(),
                            ]
                        )
                    )

                return html.Div(result_items)

            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Error updating validation results: {e}")
                return html.Div(
                    [
                        html.H4(
                            "Error Loading Validation Results", style={"color": "red"}
                        ),
                        html.P(f"An error occurred: {str(e)}"),
                    ]
                )

    def _create_overview_tab(self) -> html.Div:
        """Create the overview tab content."""
        try:
            return html.Div(
                [
                    html.H2("System Overview"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Graph(id="system-metrics-graph"),
                                ],
                                style={"width": "48%", "display": "inline-block"},
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="memory-usage-graph"),
                                ],
                                style={
                                    "width": "48%",
                                    "display": "inline-block",
                                    "float": "right",
                                },
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.H3("Recent System Metrics"),
                            html.Table(
                                [
                                    html.Thead(
                                        html.Tr(
                                            [
                                                html.Th("Metric"),
                                                html.Th("Value"),
                                                html.Th("Timestamp"),
                                            ]
                                        )
                                    ),
                                    html.Tbody(id="system-metrics-table"),
                                ]
                            ),
                        ]
                    ),
                ]
            )
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error creating overview tab: {e}")
            return html.Div(
                [
                    html.H3("Error Loading Overview", style={"color": "red"}),
                    html.P(f"Failed to create overview tab: {str(e)}"),
                ]
            )

    def _create_performance_tab(self) -> html.Div:
        """Create the performance metrics tab."""
        try:
            return html.Div(
                [
                    html.H2("Performance Metrics"),
                    html.Div(
                        [
                            html.H3("Performance Data Table"),
                            html.Table(
                                [
                                    html.Thead(
                                        html.Tr(
                                            [
                                                html.Th("Metric"),
                                                html.Th("Value"),
                                                html.Th("Timestamp"),
                                            ]
                                        )
                                    ),
                                    html.Tbody(id="performance-table-body"),
                                ],
                                id="performance-table",
                            ),
                        ]
                    ),
                ]
            )
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error creating performance tab: {e}")
            return html.Div(
                [
                    html.H3(
                        "Error Loading Performance Metrics", style={"color": "red"}
                    ),
                    html.P(f"Failed to create performance tab: {str(e)}"),
                ]
            )

    def _create_validation_tab(self) -> html.Div:
        """Create the validation results tab."""
        try:
            return html.Div(
                [
                    html.H2("Validation Results"),
                    html.Div(id="validation-results-display"),
                ]
            )
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error creating validation tab: {e}")
            return html.Div(
                [
                    html.H3("Error Loading Validation Results", style={"color": "red"}),
                    html.P(f"Failed to create validation tab: {str(e)}"),
                ]
            )

    def _create_error_tab(self) -> html.Div:
        """Create the error monitoring tab."""
        try:
            return html.Div(
                [
                    html.H2("Error Monitoring"),
                    html.Div(
                        [
                            html.H3("System Errors and Warnings"),
                            html.P(
                                "Error monitoring is handled in the main dashboard area."
                            ),
                        ]
                    ),
                ]
            )
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error creating error tab: {e}")
            return html.Div(
                [
                    html.H3("Error Loading Error Monitoring", style={"color": "red"}),
                    html.P(f"Failed to create error tab: {str(e)}"),
                ]
            )

    def _collect_system_metrics(self):
        """Collect current system metrics with error handling."""
        try:
            if not PSUTIL_AVAILABLE:
                return

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_usage": psutil.disk_usage("/").percent,
                "network_connections": len(psutil.net_connections()),
            }

            self.system_metrics.append(metrics)

            # Keep only last 100 entries
            if len(self.system_metrics) > 100:
                self.system_metrics = self.system_metrics[-100:]

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error collecting system metrics: {e}")

    def _check_for_errors(self) -> List[str]:
        """Check for system errors and return list of error messages."""
        errors = []

        try:
            if not self.system_metrics:
                return ["No system metrics available"]

            latest = self.system_metrics[-1] if self.system_metrics else {}

            # Check CPU usage
            if latest.get("cpu_percent", 0) > 90:
                errors.append(f"High CPU usage: {latest['cpu_percent']:.1f}%")

            # Check memory usage
            if latest.get("memory_percent", 0) > 90:
                errors.append(f"High memory usage: {latest['memory_percent']:.1f}%")

            # Check disk usage
            if latest.get("disk_usage", 0) > 95:
                errors.append(f"High disk usage: {latest['disk_usage']:.1f}%")

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error checking for system errors: {e}")
            errors.append(f"Error monitoring system: {str(e)}")

        return errors

    def _start_monitoring(self):
        """Start background monitoring thread."""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                with self.data_lock:
                    self._collect_system_metrics()

                    # Collect performance metrics
                    try:
                        from utils.logging_config import apgi_logger

                        if apgi_logger:
                            perf_summary = apgi_logger.get_performance_summary()
                            if perf_summary:
                                self.performance_metrics.append(
                                    {
                                        "timestamp": datetime.now().isoformat(),
                                        "metrics": perf_summary,
                                    }
                                )

                                # Keep only last 50 entries
                                if len(self.performance_metrics) > 50:
                                    self.performance_metrics = self.performance_metrics[
                                        -50:
                                    ]
                    except Exception as e:
                        if apgi_logger:
                            apgi_logger.logger.error(
                                f"Error collecting performance metrics: {e}"
                            )

                    # Collect validation results
                    try:
                        import glob

                        validation_files = glob.glob("validation_results/*.json")
                        for vf in validation_files[-3:]:  # Check last 3 files
                            try:
                                with open(vf, "r") as f:
                                    data = json.load(f)
                                    self.validation_results.append(
                                        {
                                            "timestamp": datetime.now().isoformat(),
                                            "file": vf,
                                            "status": "loaded",
                                            "data": data,
                                        }
                                    )

                                    # Keep only last 20 results
                                    if len(self.validation_results) > 20:
                                        self.validation_results = (
                                            self.validation_results[-20:]
                                        )

                            except (json.JSONDecodeError, FileNotFoundError) as e:
                                if apgi_logger:
                                    apgi_logger.logger.warning(
                                        f"Error reading validation file {vf}: {e}"
                                    )
                    except Exception as e:
                        if apgi_logger:
                            apgi_logger.logger.error(
                                f"Error collecting validation results: {e}"
                            )

            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Error in monitoring loop: {e}")

            # Sleep for 10 seconds
            time.sleep(10)

    def run(self, host: str = "127.0.0.1"):
        """
        Run the dashboard server.

        Args:
            host: Host to bind to
        """
        try:
            # Start monitoring thread
            self._start_monitoring()

            if apgi_logger:
                apgi_logger.logger.info(
                    f"Starting performance dashboard on {host}:{self.port}"
                )

            # Run the Dash app
            self.app.run_server(
                host=host,
                port=self.port,
                debug=self.debug,
                use_reloader=False,  # Disable reloader to avoid conflicts with monitoring thread
            )

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error running dashboard: {e}")
            raise
        finally:
            # Stop monitoring
            self.monitoring_active = False
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
