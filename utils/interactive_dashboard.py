"""
APGI Real-time Interactive Dashboard
====================================

Interactive dashboard for monitoring APGI framework performance, validation results,
 and system metrics in real-time using Plotly Dash.
"""

# Try to import dash, provide fallback if not available
try:
    import dash
    import plotly.graph_objs as go
    from dash import Input, Output, dcc, html

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Warning: Dash not available. Interactive dashboard will not be available.")
    print("To install: pip install dash")

import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

try:
    from utils.logging_config import apgi_logger
except ImportError:
    # Fallback if running as standalone script
    import logging

    class MockAPGILogger:
        def __init__(self):
            self.logger = logging.getLogger(__name__)

    apgi_logger = MockAPGILogger()

# APGI imports
try:
    from utils.performance_profiler import performance_profiler
except ImportError:
    # Fallback if performance_profiler is not available
    performance_profiler = None


class DashboardData:
    """Data manager for the dashboard."""

    def __init__(self):
        self.update_interval = 2  # seconds
        self.running = False
        self.thread = None

        # Data storage
        self.system_metrics = deque(maxlen=1000)
        self.performance_metrics = deque(maxlen=1000)
        self.validation_results = deque(maxlen=100)
        self.alerts = deque(maxlen=50)

        # Start data collection
        self.start_collection()

    def start_collection(self):
        """Start data collection thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._collect_data, daemon=True)
            self.thread.start()

    def stop_collection(self):
        """Stop data collection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def _collect_data(self):
        """Background data collection."""
        while self.running:
            try:
                # Collect system metrics
                current_system = (
                    performance_profiler.system_monitor.get_current_metrics()
                )
                if current_system:
                    current_system["dashboard_timestamp"] = datetime.now()
                    self.system_metrics.append(current_system)

                # Collect performance metrics
                recent_metrics = performance_profiler.custom_metrics[
                    -10:
                ]  # Last 10 metrics
                for metric in recent_metrics:
                    self.performance_metrics.append(
                        {
                            "name": metric.name,
                            "value": metric.value,
                            "unit": metric.unit,
                            "category": metric.category,
                            "timestamp": metric.timestamp,
                            "dashboard_timestamp": datetime.now(),
                        }
                    )

                # Check for alerts
                self._check_alerts(current_system)

                time.sleep(self.update_interval)

            except Exception as e:
                apgi_logger.warning(f"Error in dashboard data collection: {e}")
                time.sleep(5)

    def _check_alerts(self, system_metrics: Dict[str, Any]):
        """Check for performance alerts."""
        if not system_metrics:
            return

        alerts = []

        # CPU alert
        if system_metrics.get("cpu_percent", 0) > 90:
            alerts.append(
                {
                    "type": "warning",
                    "message": f"High CPU usage: {system_metrics['cpu_percent']:.1f}%",
                    "timestamp": datetime.now(),
                    "severity": (
                        "high" if system_metrics["cpu_percent"] > 95 else "medium"
                    ),
                }
            )

        # Memory alert
        if system_metrics.get("memory_percent", 0) > 90:
            alerts.append(
                {
                    "type": "warning",
                    "message": f"High memory usage: {system_metrics['memory_percent']:.1f}%",
                    "timestamp": datetime.now(),
                    "severity": (
                        "high" if system_metrics["memory_percent"] > 95 else "medium"
                    ),
                }
            )

        # Process memory alert
        process_memory_gb = system_metrics.get("process_memory_rss", 0) / (1024**3)
        if process_memory_gb > 2:  # > 2GB
            alerts.append(
                {
                    "type": "warning",
                    "message": f"High process memory: {process_memory_gb:.1f}GB",
                    "timestamp": datetime.now(),
                    "severity": "high" if process_memory_gb > 4 else "medium",
                }
            )

        # Add alerts to queue
        for alert in alerts:
            self.alerts.append(alert)

    def get_system_metrics_df(self) -> pd.DataFrame:
        """Get system metrics as DataFrame."""
        if not self.system_metrics:
            return pd.DataFrame()

        df = pd.DataFrame(list(self.system_metrics))
        if "dashboard_timestamp" in df.columns:
            df.loc[:, "timestamp"] = pd.to_datetime(df["dashboard_timestamp"])
        return df

    def get_performance_metrics_df(self) -> pd.DataFrame:
        """Get performance metrics as DataFrame."""
        if not self.performance_metrics:
            return pd.DataFrame()

        df = pd.DataFrame(list(self.performance_metrics))
        if "timestamp" in df.columns:
            df.loc[:, "timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return list(self.alerts)[-limit:]


# Initialize dashboard data
dashboard_data = DashboardData()

# Create Dash app only if dash is available
if DASH_AVAILABLE:
    app = dash.Dash(__name__)
    app.title = "APGI Performance Dashboard"
else:
    app = None

# Layout and callbacks only if dash is available
if DASH_AVAILABLE:
    # Layout
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H1(
                        "APGI Performance Dashboard",
                        style={
                            "textAlign": "center",
                            "color": "#2c3e50",
                            "marginBottom": "30px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        "System Status", style={"color": "#34495e"}
                                    ),
                                    html.Div(
                                        id="system-status", style={"fontSize": "18px"}
                                    ),
                                ],
                                className="four columns",
                            ),
                            html.Div(
                                [
                                    html.H3("Last Update", style={"color": "#34495e"}),
                                    html.Div(
                                        id="last-update", style={"fontSize": "16px"}
                                    ),
                                ],
                                className="four columns",
                            ),
                            html.Div(
                                [
                                    html.H3(
                                        "Active Alerts", style={"color": "#34495e"}
                                    ),
                                    html.Div(
                                        id="alert-count", style={"fontSize": "18px"}
                                    ),
                                ],
                                className="four columns",
                            ),
                        ],
                        className="row",
                        style={"marginBottom": "30px"},
                    ),
                    # Alert section
                    html.Div(
                        [
                            html.H3("Recent Alerts", style={"color": "#e74c3c"}),
                            html.Div(id="alerts-container"),
                        ],
                        style={
                            "marginBottom": "30px",
                            "padding": "15px",
                            "backgroundColor": "#fdf2f2",
                            "borderRadius": "5px",
                        },
                    ),
                    # System metrics
                    html.Div(
                        [
                            html.H3(
                                "System Metrics",
                                style={"color": "#34495e", "marginBottom": "20px"},
                            ),
                            dcc.Graph(id="system-metrics-graph"),
                        ],
                        style={"marginBottom": "30px"},
                    ),
                    # Performance metrics
                    html.Div(
                        [
                            html.H3(
                                "Performance Metrics",
                                style={"color": "#34495e", "marginBottom": "20px"},
                            ),
                            dcc.Graph(id="performance-metrics-graph"),
                        ],
                        style={"marginBottom": "30px"},
                    ),
                    # Function performance
                    html.Div(
                        [
                            html.H3(
                                "Function Performance",
                                style={"color": "#34495e", "marginBottom": "20px"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [dcc.Graph(id="top-functions-graph")],
                                        className="six columns",
                                    ),
                                    html.Div(
                                        [dcc.Graph(id="function-calls-graph")],
                                        className="six columns",
                                    ),
                                ],
                                className="row",
                            ),
                        ],
                        style={"marginBottom": "30px"},
                    ),
                    # Memory usage
                    html.Div(
                        [
                            html.H3(
                                "Memory Usage",
                                style={"color": "#34495e", "marginBottom": "20px"},
                            ),
                            dcc.Graph(id="memory-usage-graph"),
                        ],
                        style={"marginBottom": "30px"},
                    ),
                    # Auto-refresh
                    dcc.Interval(
                        id="interval-component",
                        interval=2000,
                        n_intervals=0,  # 2 seconds
                    ),
                    # Hidden div to store data
                    html.Div(id="data-store", style={"display": "none"}),
                ],
                style={"padding": "20px", "fontFamily": "Arial, sans-serif"},
            )
        ]
    )


# Callbacks
@app.callback(
    [
        Output("system-status", "children"),
        Output("last-update", "children"),
        Output("alert-count", "children"),
        Output("alerts-container", "children"),
    ],
    [Input("interval-component", "n_intervals")],
)
def update_status_info(n):
    """Update status information."""
    try:
        # System status
        current_system = performance_profiler.system_monitor.get_current_metrics()
        if current_system:
            cpu_status = (
                "🟢 Normal"
                if current_system.get("cpu_percent", 0) < 80
                else (
                    "🟡 High"
                    if current_system.get("cpu_percent", 0) < 95
                    else "🔴 Critical"
                )
            )
            memory_status = (
                "🟢 Normal"
                if current_system.get("memory_percent", 0) < 80
                else (
                    "🟡 High"
                    if current_system.get("memory_percent", 0) < 95
                    else "🔴 Critical"
                )
            )

            system_status = html.Div(
                [
                    html.P(
                        f"CPU: {cpu_status} ({current_system.get('cpu_percent', 0):.1f}%)"
                    ),
                    html.P(
                        f"Memory: {memory_status} ({current_system.get('memory_percent', 0):.1f}%)"
                    ),
                ]
            )
        else:
            system_status = html.P("No data available", style={"color": "#e74c3c"})

        # Last update
        last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Alert count
        recent_alerts = dashboard_data.get_recent_alerts()
        alert_count = len(recent_alerts)
        alert_count_html = html.Span(
            [
                f"{alert_count} active alerts",
                html.Span(" ⚠️", style={"color": "#e74c3c"}) if alert_count > 0 else "",
            ]
        )

        # Alerts
        if recent_alerts:
            alerts_html = []
            for alert in recent_alerts[-5:]:  # Show last 5 alerts
                color = "#e74c3c" if alert["severity"] == "high" else "#f39c12"
                alerts_html.append(
                    html.Div(
                        [
                            html.Strong(
                                f"{alert['timestamp'].strftime('%H:%M:%S')} - "
                            ),
                            html.Span(alert["message"], style={"color": color}),
                        ],
                        style={
                            "marginBottom": "5px",
                            "padding": "5px",
                            "backgroundColor": "white",
                            "borderRadius": "3px",
                        },
                    )
                )
        else:
            alerts_html = [html.P("No recent alerts", style={"color": "#27ae60"})]

        return system_status, last_update, alert_count_html, alerts_html

    except Exception as e:
        error_msg = f"Error updating status: {e}"
        return (
            html.P(error_msg, style={"color": "#e74c3c"}),
            "",
            "",
            html.P(error_msg, style={"color": "#e74c3c"}),
        )


@app.callback(
    Output("system-metrics-graph", "figure"),
    [Input("interval-component", "n_intervals")],
)
def update_system_metrics(n):
    """Update system metrics graph."""
    try:
        df = dashboard_data.get_system_metrics_df()

        if df.empty:
            return {
                "data": [],
                "layout": {
                    "title": "No system data available",
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Value"},
                },
            }

        fig = go.Figure()

        # CPU percentage
        if "cpu_percent" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["cpu_percent"],
                    mode="lines",
                    name="CPU %",
                    line=dict(color="#3498db", width=2),
                )
            )

        # Memory percentage
        if "memory_percent" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["memory_percent"],
                    mode="lines",
                    name="Memory %",
                    line=dict(color="#e74c3c", width=2),
                )
            )

        # Process memory (GB)
        if "process_memory_rss" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["process_memory_rss"] / (1024**3),
                    mode="lines",
                    name="Process Memory (GB)",
                    line=dict(color="#f39c12", width=2),
                    yaxis="y2",
                )
            )

        fig.update_layout(
            title="System Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Percentage (%)",
            yaxis2=dict(title="Memory (GB)", overlaying="y", side="right"),
            hovermode="x unified",
            showlegend=True,
            height=400,
        )

        return fig

    except Exception as e:
        return {
            "data": [],
            "layout": {
                "title": f"Error: {e}",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Value"},
            },
        }


@app.callback(
    Output("performance-metrics-graph", "figure"),
    [Input("interval-component", "n_intervals")],
)
def update_performance_metrics(n):
    """Update performance metrics graph."""
    try:
        df = dashboard_data.get_performance_metrics_df()

        if df.empty:
            return {
                "data": [],
                "layout": {
                    "title": "No performance data available",
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Value"},
                },
            }

        # Group by category
        categories = df["category"].unique()

        fig = go.Figure()

        for category in categories:
            category_df = df[df["category"] == category]
            fig.add_trace(
                go.Scatter(
                    x=category_df["timestamp"],
                    y=category_df["value"],
                    mode="markers+lines",
                    name=category.title(),
                    text=category_df["name"],
                    hovertemplate="%{text}<br>%{y:.3f} %{customdata}<br>%{x}<extra></extra>",
                    customdata=category_df["unit"],
                )
            )

        fig.update_layout(
            title="Performance Metrics by Category",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode="x unified",
            showlegend=True,
            height=400,
        )

        return fig

    except Exception as e:
        return {
            "data": [],
            "layout": {
                "title": f"Error: {e}",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Value"},
            },
        }


@app.callback(
    [Output("top-functions-graph", "figure"), Output("function-calls-graph", "figure")],
    [Input("interval-component", "n_intervals")],
)
def update_function_performance(n):
    """Update function performance graphs."""
    try:
        # Get top functions by total time
        top_functions = performance_profiler.get_top_functions("total_time", 8)

        if not top_functions:
            empty_fig = {
                "data": [],
                "layout": {
                    "title": "No function data available",
                    "xaxis": {"title": "Function"},
                    "yaxis": {"title": "Time (s)"},
                },
            }
            return empty_fig, empty_fig

        # Top functions by total time
        names = [f.name.split(".")[-1] for f in top_functions]
        total_times = [f.total_time for f in top_functions]

        fig1 = go.Figure(data=[go.Bar(x=names, y=total_times, marker_color="#3498db")])
        fig1.update_layout(
            title="Top Functions by Total Time",
            xaxis_title="Function",
            yaxis_title="Total Time (s)",
            height=300,
        )

        # Function call counts
        call_counts = [f.call_count for f in top_functions]

        fig2 = go.Figure(data=[go.Bar(x=names, y=call_counts, marker_color="#e74c3c")])
        fig2.update_layout(
            title="Function Call Counts",
            xaxis_title="Function",
            yaxis_title="Call Count",
            height=300,
        )

        return fig1, fig2

    except Exception as e:
        error_fig = {
            "data": [],
            "layout": {
                "title": f"Error: {e}",
                "xaxis": {"title": "Function"},
                "yaxis": {"title": "Value"},
            },
        }
        return error_fig, error_fig


@app.callback(
    Output("memory-usage-graph", "figure"), [Input("interval-component", "n_intervals")]
)
def update_memory_usage(n):
    """Update memory usage graph."""
    try:
        df = dashboard_data.get_system_metrics_df()

        if df.empty:
            return {
                "data": [],
                "layout": {
                    "title": "No memory data available",
                    "xaxis": {"title": "Time"},
                    "yaxis": {"title": "Memory"},
                },
            }

        fig = go.Figure()

        # System memory
        if "memory_used" in df.columns and "memory_available" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["memory_used"] / (1024**3),
                    mode="lines",
                    name="Used Memory (GB)",
                    line=dict(color="#e74c3c", width=2),
                    stackgroup="one",
                )
            )

        # Process memory
        if "process_memory_rss" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["process_memory_rss"] / (1024**3),
                    mode="lines",
                    name="Process Memory (GB)",
                    line=dict(color="#f39c12", width=3),
                )
            )

        fig.update_layout(
            title="Memory Usage Over Time",
            xaxis_title="Time",
            yaxis_title="Memory (GB)",
            hovermode="x unified",
            showlegend=True,
            height=400,
        )

        return fig

    except Exception as e:
        return {
            "data": [],
            "layout": {
                "title": f"Error: {e}",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Memory (GB)"},
            },
        }


# End of DASH_AVAILABLE conditional


def run_dashboard(host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
    """Run the dashboard server."""
    if not DASH_AVAILABLE:
        print("Error: Dash is not available. Cannot run interactive dashboard.")
        print("To install: pip install dash")
        return False

    apgi_logger.info(f"Starting APGI dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
    return True


if __name__ == "__main__":
    run_dashboard(debug=True)
