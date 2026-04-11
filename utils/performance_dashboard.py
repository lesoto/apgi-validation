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
from typing import Any, Dict, List

try:
    import dash
    import plotly.graph_objects as go
    from dash import Input, Output, dcc, html

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Warning: dash not available. Dashboard features disabled.")

    # Create stubs for dash components
    class _HtmlStub:
        class Div:
            def __init__(self, *args, **kwargs):
                self.children = args
                self.style = kwargs.get("style", {})

        class H1:
            def __init__(self, *args, **kwargs):
                self.children = args
                self.style = kwargs.get("style", {})

        class H2:
            def __init__(self, *args, **kwargs):
                self.children = args
                self.style = kwargs.get("style", {})

        class P:
            def __init__(self, *args, **kwargs):
                self.children = args
                self.style = kwargs.get("style", {})

        class Button:
            def __init__(self, *args, **kwargs):
                self.children = args
                self.style = kwargs.get("style", {})

        class Table:
            def __init__(self, *args, **kwargs):
                self.children = args
                self.style = kwargs.get("style", {})

        class Thead:
            def __init__(self, *args, **kwargs):
                self.children = args

        class Tbody:
            def __init__(self, *args, **kwargs):
                self.children = args

        class Tr:
            def __init__(self, *args, **kwargs):
                self.children = args

        class Th:
            def __init__(self, *args, **kwargs):
                self.children = args

        class Td:
            def __init__(self, *args, **kwargs):
                self.children = args

    class _DccStub:
        class Graph:
            def __init__(self, *args, **kwargs):
                self.id = kwargs.get("id")
                self.figure = kwargs.get("figure")

        class DatePickerRange:
            def __init__(self, *args, **kwargs):
                self.id = kwargs.get("id")
                self.start_date = kwargs.get("start_date")
                self.end_date = kwargs.get("end_date")

        class Dropdown:
            def __init__(self, *args, **kwargs):
                self.id = kwargs.get("id")
                self.options = kwargs.get("options")
                self.value = kwargs.get("value")

        class Tabs:
            def __init__(self, *args, **kwargs):
                self.id = kwargs.get("id")
                self.children = args

        class Tab:
            def __init__(self, *args, **kwargs):
                self.label = kwargs.get("label")
                self.children = args

    html = _HtmlStub()
    dcc = _DccStub()

    class _DashStub:
        def __init__(self):
            pass

        def run_server(self, *args, **kwargs):
            print("Dash not available - server not started")

    dash = _DashStub()

    # Stub for plotly
    class _PlotlyStub:
        class Figure:
            def __init__(self, *args, **kwargs):
                self.data = []
                self.layout = {}

    go = _PlotlyStub()

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
        self.system_metrics: List[Dict[str, Any]] = []
        self.validation_results: List[Dict[str, Any]] = []
        self.performance_metrics: List[Dict[str, Any]] = []

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
                        dcc.Tab(label="Data Exploration", value="exploration"),
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
            Output("tab-content", "children"),
            Input("main-tabs", "value"),
        )
        @self.app.callback(
            Output("export-data", "children"),
            Input("export-format", "value"),
            Input("download-export", "n_clicks"),
        )
        def handle_export_data(self, export_format, filename):
            """Handle export data request."""
            try:
                success = self.export_data(export_format, filename)
                if success:
                    if apgi_logger:
                        apgi_logger.logger.info(
                            f"Data exported successfully in {export_format} format"
                        )
                    return True
                else:
                    if apgi_logger:
                        apgi_logger.logger.error("Export failed")
                    return False
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Export error: {e}")
                return False

        @self.app.callback(
            Output("download-export", "children"),
            Input("export-format", "value"),
            Input("download-export", "n_clicks"),
            prevent_initial_call=True,
        )
        def handle_download_export(self, export_format, n_clicks):
            """Handle download export request."""
            if n_clicks > 0:
                try:
                    success = self.export_data(export_format)
                    if success:
                        if apgi_logger:
                            apgi_logger.logger.info(
                                f"Download triggered for {export_format} export"
                            )
                        return True
                    else:
                        if apgi_logger:
                            apgi_logger.logger.error("Download failed")
                        return False
                except Exception as e:
                    if apgi_logger:
                        apgi_logger.logger.error(f"Download error: {e}")
                    return False

        def update_tab_content(tab_value):
            """Update tab content with error handling."""
            try:
                if tab_value == "overview":
                    return self._create_overview_tab()
                elif tab_value == "performance":
                    return self._create_performance_tab()
                elif tab_value == "validation":
                    return self._create_validation_tab()
                elif tab_value == "exploration":
                    return self._create_exploration_tab()
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
                    # Return empty figures with clear error messages
                    empty_fig = go.Figure()
                    empty_fig.add_annotation(
                        text="No system metrics available yet.<br>Please wait for data collection to begin.",
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=14, color="red"),
                    )
                    empty_fig.update_layout(
                        title="System Metrics - No Data Available",
                        xaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                        yaxis=dict(
                            showgrid=False, zeroline=False, showticklabels=False
                        ),
                    )
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

    def _create_exploration_tab(self) -> html.Div:
        """Create the data exploration tab."""
        try:
            return html.Div(
                [
                    html.H2("Interactive Data Exploration"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Dataset Selection"),
                                    dcc.Dropdown(
                                        id="dataset-selector",
                                        options=[
                                            {
                                                "label": "System Metrics",
                                                "value": "system",
                                            },
                                            {
                                                "label": "Performance Data",
                                                "value": "performance",
                                            },
                                            {
                                                "label": "Validation Results",
                                                "value": "validation",
                                            },
                                            {
                                                "label": "Sample Multimodal Data",
                                                "value": "sample",
                                            },
                                        ],
                                        value="system",
                                        clearable=False,
                                    ),
                                ],
                                style={
                                    "width": "30%",
                                    "display": "inline-block",
                                    "marginRight": "20px",
                                },
                            ),
                            html.Div(
                                [
                                    html.H3("Visualization Type"),
                                    dcc.Dropdown(
                                        id="plot-type-selector",
                                        options=[
                                            {"label": "Line Chart", "value": "line"},
                                            {
                                                "label": "Scatter Plot",
                                                "value": "scatter",
                                            },
                                            {"label": "Bar Chart", "value": "bar"},
                                            {
                                                "label": "Histogram",
                                                "value": "histogram",
                                            },
                                            {"label": "Box Plot", "value": "box"},
                                        ],
                                        value="line",
                                        clearable=False,
                                    ),
                                ],
                                style={
                                    "width": "30%",
                                    "display": "inline-block",
                                    "marginRight": "20px",
                                },
                            ),
                            html.Div(
                                [
                                    html.H3("X-Axis"),
                                    dcc.Dropdown(
                                        id="x-axis-selector",
                                        options=[],  # Will be populated dynamically
                                        value=None,
                                    ),
                                ],
                                style={"width": "30%", "display": "inline-block"},
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Y-Axis"),
                                    dcc.Dropdown(
                                        id="y-axis-selector",
                                        options=[],  # Will be populated dynamically
                                        value=None,
                                        multi=True,
                                    ),
                                ],
                                style={
                                    "width": "45%",
                                    "display": "inline-block",
                                    "marginRight": "20px",
                                },
                            ),
                            html.Div(
                                [
                                    html.H3("Filters"),
                                    dcc.RangeSlider(
                                        id="time-range-slider",
                                        min=0,
                                        max=100,
                                        step=1,
                                        value=[0, 100],
                                        marks={0: "Start", 100: "End"},
                                    ),
                                ],
                                style={"width": "45%", "display": "inline-block"},
                            ),
                        ],
                        style={"marginTop": "20px"},
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Update Visualization", id="update-viz-btn", n_clicks=0
                            ),
                        ],
                        style={"marginTop": "20px", "marginBottom": "20px"},
                    ),
                    dcc.Graph(id="exploration-graph"),
                    html.Div(id="exploration-stats"),
                ]
            )
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error creating exploration tab: {e}")
            return html.Div(
                [
                    html.H3("Error Loading Data Exploration", style={"color": "red"}),
                    html.P(f"Failed to create exploration tab: {str(e)}"),
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
                                with open(vf, "r", encoding="utf-8") as f:
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

    def _stop_monitoring(self):
        self.monitoring_active = False

    def export_data(self, format_type: str = "json", filename: str = None):
        """
        Export dashboard data in specified format.

        Args:
            format_type: Export format ('json', 'csv', 'pdf')
            filename: Optional filename for export (auto-generated if None)

        Returns:
            bool: True if successful, False otherwise

        Raises:
            ValueError: If format_type is not supported
            IOError: If export fails
        """
        try:
            if not hasattr(self, "performance_data") or not hasattr(
                self, "validation_results"
            ):
                raise ValueError("No data available to export")

            # Prepare data for export
            export_data = {
                "performance_metrics": getattr(self, "performance_data", {}),
                "validation_results": getattr(self, "validation_results", {}),
                "system_metrics": getattr(self, "system_metrics", {}),
                "export_timestamp": datetime.now().isoformat(),
            }

            if format_type.lower() == "json":
                return self._export_json(export_data, filename)
            elif format_type.lower() == "csv":
                return self._export_csv(export_data, filename)
            elif format_type.lower() == "pdf":
                return self._export_pdf(export_data, filename)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Export failed: {e}")
            raise

    def _export_json(self, data: dict, filename: str = None) -> bool:
        """Export data to JSON format."""
        try:
            import json

            output_filename = (
                filename
                or f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

            if apgi_logger:
                apgi_logger.logger.info(f"Data exported to JSON: {output_filename}")
            return True

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"JSON export failed: {e}")
            return False

    def _export_csv(self, data: dict, filename: str = None) -> bool:
        """Export data to CSV format."""
        try:
            import pandas as pd

            output_filename = (
                filename
                or f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            # Convert nested data to flat structure for CSV
            csv_data = []

            # Performance metrics
            if "performance_metrics" in data:
                for timestamp, metrics in data["performance_metrics"].items():
                    csv_data.append(
                        {
                            "category": "performance",
                            "timestamp": timestamp,
                            "metric_name": "cpu_usage",
                            "value": metrics.get("cpu_usage", "N/A"),
                        }
                    )
                    csv_data.append(
                        {
                            "category": "performance",
                            "timestamp": timestamp,
                            "metric_name": "memory_usage",
                            "value": metrics.get("memory_usage", "N/A"),
                        }
                    )

            # Validation results
            if "validation_results" in data:
                for protocol, results in data["validation_results"].items():
                    csv_data.append(
                        {
                            "category": "validation",
                            "protocol": protocol,
                            "result": str(results.get("status", "unknown")),
                            "execution_time": results.get("execution_time", "N/A"),
                        }
                    )

            # System metrics
            if "system_metrics" in data:
                for timestamp, metrics in data["system_metrics"].items():
                    for metric_name, value in metrics.items():
                        csv_data.append(
                            {
                                "category": "system",
                                "timestamp": timestamp,
                                "metric_name": metric_name,
                                "value": str(value),
                            }
                        )

            df = pd.DataFrame(csv_data)
            df.to_csv(output_filename, index=False)

            if apgi_logger:
                apgi_logger.logger.info(f"Data exported to CSV: {output_filename}")
            return True

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"CSV export failed: {e}")
            return False

    def _export_pdf(self, data: dict, filename: str = None) -> bool:
        """Export data to PDF format."""
        try:
            import io

            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

            output_filename = (
                filename
                or f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )

            # Create PDF buffer
            buffer = io.BytesIO()

            # Build PDF document
            doc = SimpleDocTemplate(
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18,
            )

            # Add title
            styles = getSampleStyleSheet()
            title_style = styles["Heading1"]
            doc.append(Paragraph("APGI Framework Dashboard Export", title_style))
            doc.append(Spacer(1, 12))

            # Add performance metrics section
            if "performance_metrics" in data:
                doc.append(Paragraph("Performance Metrics", styles["Heading2"]))
                for timestamp, metrics in data["performance_metrics"].items():
                    doc.append(Paragraph(f"Timestamp: {timestamp}", styles["Normal"]))
                    for metric, value in metrics.items():
                        doc.append(Paragraph(f"  {metric}: {value}", styles["Normal"]))
                doc.append(Spacer(1, 12))

            # Add validation results section
            if "validation_results" in data:
                doc.append(Paragraph("Validation Results", styles["Heading2"]))
                for protocol, results in data["validation_results"].items():
                    doc.append(Paragraph(f"Protocol: {protocol}", styles["Normal"]))
                    doc.append(
                        Paragraph(
                            f"  Status: {results.get('status', 'unknown')}",
                            styles["Normal"],
                        )
                    )
                    doc.append(
                        Paragraph(
                            f"  Execution Time: {results.get('execution_time', 'N/A')}",
                            styles["Normal"],
                        )
                    )
                doc.append(Spacer(1, 12))

            # Add system metrics section
            if "system_metrics" in data:
                doc.append(Paragraph("System Metrics", styles["Heading2"]))
                for timestamp, metrics in data["system_metrics"].items():
                    doc.append(Paragraph(f"Timestamp: {timestamp}", styles["Normal"]))
                    for metric, value in metrics.items():
                        doc.append(Paragraph(f"  {metric}: {value}", styles["Normal"]))
                doc.append(Spacer(1, 12))

            # Build PDF
            doc.build([buffer])

            # Save to file
            with open(output_filename, "wb") as f:
                f.write(buffer.getvalue())

            if apgi_logger:
                apgi_logger.logger.info(f"Data exported to PDF: {output_filename}")
            return True

        except ImportError:
            if apgi_logger:
                apgi_logger.logger.error(
                    "PDF export requires reportlab. Install with: pip install reportlab"
                )
            return False
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"PDF export failed: {e}")
            return False

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
