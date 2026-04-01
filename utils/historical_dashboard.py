#!/usr/bin/env python3
"""
APGI Historical Dashboard - Enhanced Analytics and Export Features
================================================================

Advanced dashboard with historical data analysis, trend detection,
and comprehensive export capabilities for the APGI Validation Framework.
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import statistics

try:
    import dash
    from dash import html, dcc, Input, Output
    import pandas as pd

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

    # Stub for pandas
    class _PdStub:
        class DataFrame:
            def __init__(self, data):
                self.data = data

            def to_dict(self, *args, **kwargs):
                return self.data

    pd = _PdStub()

try:
    from utils.logging_config import apgi_logger
except ImportError:
    apgi_logger = None


class HistoricalDashboard:
    """Enhanced dashboard with historical analysis and export capabilities."""

    def __init__(
        self, db_path: str = "data_repository/historical_data.db", port: int = 8051
    ):
        """
        Initialize the historical dashboard.

        Args:
            db_path: Path to SQLite database for historical data
            port: Port to run the dashboard on
        """
        if not DASH_AVAILABLE:
            print("Warning: Running in test mode without dash functionality")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.port = port

        if DASH_AVAILABLE:
            self.app = dash.Dash(__name__, title="APGI Historical Dashboard")
        else:
            self.app = None

        # Initialize database
        self._init_database()

        # Data analysis cache
        self._analysis_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()

        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()

        if apgi_logger:
            apgi_logger.logger.info(f"Initialized historical dashboard on port {port}")

    def _init_database(self):
        """Initialize SQLite database for historical data storage."""
        try:
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

                # Validation results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS validation_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        protocol_number INTEGER,
                        protocol_name TEXT,
                        status TEXT,
                        execution_time REAL,
                        tests_passed INTEGER,
                        tests_failed INTEGER,
                        success_rate REAL,
                        error_message TEXT
                    )
                """)

                # Performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metric_category TEXT,
                        metric_name TEXT,
                        metric_value REAL,
                        unit TEXT,
                        metadata TEXT
                    )
                """)

                # Create indexes for better query performance
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_validation_timestamp ON validation_results(timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_protocol_number ON validation_results(protocol_number)"
                )

                conn.commit()

                if apgi_logger:
                    apgi_logger.logger.info("Database initialized successfully")

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Failed to initialize database: {e}")
            raise

    def _setup_layout(self):
        """Setup the dashboard layout with historical analysis tabs."""
        if not self.app:
            print("Skipping layout setup - dash not available")
            return

        self.app.layout = html.Div(
            [
                html.H1(
                    "🧠 APGI Historical Analytics Dashboard",
                    style={
                        "textAlign": "center",
                        "color": "#2c3e50",
                        "marginBottom": "30px",
                    },
                ),
                # Time range selector
                html.Div(
                    [
                        html.Label("Select Time Range:"),
                        dcc.DatePickerRange(
                            id="date-picker-range",
                            start_date=datetime.now() - timedelta(days=30),
                            end_date=datetime.now(),
                            display_format="YYYY-MM-DD",
                        ),
                        html.Button("Apply Range", id="apply-range-btn", n_clicks=0),
                    ],
                    style={"marginBottom": "20px", "textAlign": "center"},
                ),
                # Main tabs
                dcc.Tabs(
                    [
                        dcc.Tab(label="📊 Trends & Analysis", value="trends"),
                        dcc.Tab(label="📈 Performance History", value="performance"),
                        dcc.Tab(label="🔬 Validation Analytics", value="validation"),
                        dcc.Tab(label="📤 Export Reports", value="export"),
                        dcc.Tab(label="⚠️ Anomaly Detection", value="anomalies"),
                    ],
                    value="trends",
                    id="main-tabs",
                ),
                # Content container
                html.Div(id="tab-content", style={"marginTop": "20px"}),
                # Store for selected data
                dcc.Store(id="selected-data-store"),
                # Auto-refresh interval
                dcc.Interval(
                    id="auto-refresh", interval=30000, n_intervals=0  # 30 seconds
                ),
            ]
        )

    def _setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""
        if not self.app:
            print("Skipping callback setup - dash not available")
            return

        @self.app.callback(
            Output("tab-content", "children"),
            Input("main-tabs", "value"),
            Input("date-picker-range", "start_date"),
            Input("date-picker-range", "end_date"),
            Input("apply-range-btn", "n_clicks"),
        )
        def update_tab_content(active_tab, start_date, end_date, n_clicks):
            """Update tab content based on selected tab and date range."""
            try:
                if active_tab == "trends":
                    return self._create_trends_tab(start_date, end_date)
                elif active_tab == "performance":
                    return self._create_performance_tab(start_date, end_date)
                elif active_tab == "validation":
                    return self._create_validation_tab(start_date, end_date)
                elif active_tab == "export":
                    return self._create_export_tab(start_date, end_date)
                elif active_tab == "anomalies":
                    return self._create_anomalies_tab(start_date, end_date)
                else:
                    return html.Div("Tab not implemented")
            except Exception as e:
                if apgi_logger:
                    apgi_logger.logger.error(f"Error updating tab content: {e}")
                return html.Div(
                    [
                        html.H3("Error Loading Content", style={"color": "red"}),
                        html.P(f"Failed to load tab content: {str(e)}"),
                    ]
                )

    def _create_trends_tab(self, start_date: str, end_date: str) -> html.Div:
        """Create trends and analysis tab."""
        return html.Div(
            [
                html.H2("📊 System Trends & Analysis"),
                # Trend summary cards
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("CPU Trend"),
                                html.P(id="cpu-trend-summary", children="Analyzing..."),
                                html.Div(id="cpu-trend-indicator"),
                            ],
                            className="trend-card",
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "margin": "1%",
                            },
                        ),
                        html.Div(
                            [
                                html.H4("Memory Trend"),
                                html.P(
                                    id="memory-trend-summary", children="Analyzing..."
                                ),
                                html.Div(id="memory-trend-indicator"),
                            ],
                            className="trend-card",
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "margin": "1%",
                            },
                        ),
                        html.Div(
                            [
                                html.H4("Validation Success Rate"),
                                html.P(
                                    id="validation-trend-summary",
                                    children="Analyzing...",
                                ),
                                html.Div(id="validation-trend-indicator"),
                            ],
                            className="trend-card",
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "margin": "1%",
                            },
                        ),
                    ],
                    style={"marginBottom": "30px"},
                ),
                # Main trend charts
                dcc.Graph(id="main-trends-chart"),
                # Statistical analysis
                html.Div(
                    [
                        html.H3("Statistical Analysis"),
                        html.Div(id="statistical-analysis"),
                    ],
                    style={"marginTop": "30px"},
                ),
            ]
        )

    def _create_performance_tab(self, start_date: str, end_date: str) -> html.Div:
        """Create performance history tab."""
        return html.Div(
            [
                html.H2("📈 Performance History"),
                # Performance metrics selector
                html.Div(
                    [
                        html.Label("Select Metrics:"),
                        dcc.Checklist(
                            id="performance-metrics-selector",
                            options=[
                                {"label": "CPU Usage", "value": "cpu_percent"},
                                {"label": "Memory Usage", "value": "memory_percent"},
                                {"label": "Disk Usage", "value": "disk_usage_percent"},
                                {
                                    "label": "Network Connections",
                                    "value": "network_connections",
                                },
                            ],
                            value=["cpu_percent", "memory_percent"],
                            inline=True,
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                # Performance charts
                dcc.Graph(id="performance-history-chart"),
                # Performance table
                html.Div(
                    [
                        html.H3("Performance Data Table"),
                        html.Div(id="performance-data-table"),
                    ],
                    style={"marginTop": "30px"},
                ),
            ]
        )

    def _create_validation_tab(self, start_date: str, end_date: str) -> html.Div:
        """Create validation analytics tab."""
        return html.Div(
            [
                html.H2("🔬 Validation Analytics"),
                # Protocol selector
                html.Div(
                    [
                        html.Label("Filter by Protocol:"),
                        dcc.Dropdown(
                            id="protocol-selector",
                            options=[
                                {"label": f"Protocol {i}", "value": str(i)}
                                for i in range(1, 13)
                            ],
                            value="all",
                            multi=True,
                            placeholder="Select protocols (leave empty for all)",
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                # Validation success rate chart
                dcc.Graph(id="validation-success-chart"),
                # Execution time analysis
                dcc.Graph(id="execution-time-chart"),
                # Validation summary table
                html.Div(
                    [
                        html.H3("Validation Summary"),
                        html.Div(id="validation-summary-table"),
                    ],
                    style={"marginTop": "30px"},
                ),
            ]
        )

    def _create_export_tab(self, start_date: str, end_date: str) -> html.Div:
        """Create export reports tab."""
        return html.Div(
            [
                html.H2("📤 Export Reports"),
                # Export options
                html.Div(
                    [
                        html.H3("Export Configuration"),
                        html.Div(
                            [
                                html.Label("Export Format:"),
                                dcc.Dropdown(
                                    id="export-format",
                                    options=[
                                        {"label": "JSON", "value": "json"},
                                        {"label": "CSV", "value": "csv"},
                                        {"label": "PDF Report", "value": "pdf"},
                                        {"label": "Excel", "value": "excel"},
                                    ],
                                    value="json",
                                ),
                            ],
                            style={
                                "width": "45%",
                                "display": "inline-block",
                                "marginRight": "5%",
                            },
                        ),
                        html.Div(
                            [
                                html.Label("Data Type:"),
                                dcc.Dropdown(
                                    id="export-data-type",
                                    options=[
                                        {"label": "All Data", "value": "all"},
                                        {
                                            "label": "System Metrics Only",
                                            "value": "system",
                                        },
                                        {
                                            "label": "Validation Results Only",
                                            "value": "validation",
                                        },
                                        {
                                            "label": "Performance Metrics Only",
                                            "value": "performance",
                                        },
                                    ],
                                    value="all",
                                ),
                            ],
                            style={"width": "45%", "display": "inline-block"},
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                # Export buttons
                html.Div(
                    [
                        html.Button(
                            "📥 Generate Report",
                            id="generate-report-btn",
                            n_clicks=0,
                            style={"padding": "10px 20px", "fontSize": "16px"},
                        ),
                        html.Button(
                            "📧 Email Report",
                            id="email-report-btn",
                            n_clicks=0,
                            style={
                                "padding": "10px 20px",
                                "fontSize": "16px",
                                "marginLeft": "10px",
                            },
                        ),
                    ],
                    style={"marginBottom": "30px"},
                ),
                # Export status
                html.Div(id="export-status"),
                # Recent exports
                html.Div([html.H3("Recent Exports"), html.Div(id="recent-exports")]),
            ]
        )

    def _create_anomalies_tab(self, start_date: str, end_date: str) -> html.Div:
        """Create anomaly detection tab."""
        return html.Div(
            [
                html.H2("⚠️ Anomaly Detection"),
                # Anomaly settings
                html.Div(
                    [
                        html.Label("Sensitivity Level:"),
                        dcc.Slider(
                            id="anomaly-sensitivity",
                            min=1,
                            max=5,
                            step=1,
                            value=3,
                            marks={i: f"Level {i}" for i in range(1, 6)},
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                # Anomaly summary
                html.Div(id="anomaly-summary"),
                # Anomaly charts
                dcc.Graph(id="anomaly-chart"),
                # Anomaly details
                html.Div(
                    [html.H3("Anomaly Details"), html.Div(id="anomaly-details")],
                    style={"marginTop": "30px"},
                ),
            ]
        )

    def get_historical_data(
        self, table: str, start_date: str = None, end_date: str = None
    ) -> List[Dict]:
        """
        Retrieve historical data from database.

        Args:
            table: Table name to query
            start_date: Start date for filtering
            end_date: End date for filtering

        Returns:
            List of dictionaries containing the data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = f"SELECT * FROM {table}"
                params = []

                if start_date or end_date:
                    conditions = []
                    if start_date:
                        conditions.append("timestamp >= ?")
                        params.append(start_date)
                    if end_date:
                        conditions.append("timestamp <= ?")
                        params.append(end_date)
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY timestamp DESC"

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [dict(row) for row in rows]

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(
                    f"Error retrieving historical data from {table}: {e}"
                )
            return []

    def analyze_trends(self, data: List[Dict], metric_column: str) -> Dict[str, Any]:
        """
        Analyze trends in historical data.

        Args:
            data: List of data points
            metric_column: Column name to analyze

        Returns:
            Dictionary containing trend analysis results
        """
        if not data:
            return {"trend": "no_data", "change": 0, "direction": "neutral"}

        try:
            values = [
                row[metric_column] for row in data if row.get(metric_column) is not None
            ]

            if len(values) < 2:
                return {
                    "trend": "insufficient_data",
                    "change": 0,
                    "direction": "neutral",
                }

            # Calculate trend
            if len(values) >= 3:
                recent_avg = statistics.mean(values[-len(values) // 3 :])
            else:
                recent_avg = values[-1]
            if len(values) >= 3:
                older_avg = statistics.mean(values[: len(values) // 3])
            else:
                older_avg = values[0]

            change_percent = (
                ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0
            )

            # Determine trend direction
            if change_percent > 5:
                direction = "increasing"
                trend = "upward"
            elif change_percent < -5:
                direction = "decreasing"
                trend = "downward"
            else:
                direction = "stable"
                trend = "stable"

            # Calculate volatility
            if len(values) > 1:
                volatility = statistics.stdev(values) / statistics.mean(values) * 100
            else:
                volatility = 0

            return {
                "trend": trend,
                "direction": direction,
                "change_percent": change_percent,
                "volatility": volatility,
                "recent_avg": recent_avg,
                "older_avg": older_avg,
                "data_points": len(values),
            }

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error analyzing trends: {e}")
            return {"trend": "error", "change": 0, "direction": "neutral"}

    def export_historical_data(
        self,
        format_type: str = "json",
        data_type: str = "all",
        start_date: str = None,
        end_date: str = None,
        filename: str = None,
    ) -> bool:
        """
        Export historical data in specified format.

        Args:
            format_type: Export format (json, csv, pdf, excel)
            data_type: Type of data to export
            start_date: Start date for data range
            end_date: End date for data range
            filename: Optional custom filename

        Returns:
            True if successful, False otherwise
        """
        try:
            # Collect data based on type
            export_data: Dict[str, Any] = {}

            if data_type in ["all", "system"]:
                export_data["system_metrics"] = self.get_historical_data(
                    "system_metrics", start_date, end_date
                )

            if data_type in ["all", "validation"]:
                export_data["validation_results"] = self.get_historical_data(
                    "validation_results", start_date, end_date
                )

            if data_type in ["all", "performance"]:
                export_data["performance_metrics"] = self.get_historical_data(
                    "performance_metrics", start_date, end_date
                )

            # Add metadata
            export_data["export_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "date_range": {"start": start_date, "end": end_date},
                "data_type": data_type,
                "format": format_type,
                "total_records": sum(
                    len(data) for data in export_data.values() if isinstance(data, list)
                ),
            }

            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"apgi_historical_export_{timestamp}.{format_type}"

            # Export based on format
            if format_type == "json":
                return self._export_json(export_data, filename)
            elif format_type == "csv":
                return self._export_csv(export_data, filename)
            elif format_type == "excel":
                return self._export_excel(export_data, filename)
            elif format_type == "pdf":
                return self._export_pdf(export_data, filename)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Export failed: {e}")
            return False

    def _export_json(self, data: Dict, filename: str) -> bool:
        """Export data to JSON format."""
        try:
            output_path = Path("exports") / filename
            output_path.parent.mkdir(exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

            if apgi_logger:
                apgi_logger.logger.info(f"Data exported to JSON: {output_path}")
            return True

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"JSON export failed: {e}")
            return False

    def _export_csv(self, data: Dict, filename: str) -> bool:
        """Export data to CSV format."""
        try:
            output_path = Path("exports") / filename
            output_path.parent.mkdir(exist_ok=True)

            # Convert to DataFrame and save
            all_data = []

            for category, items in data.items():
                if isinstance(items, list) and items:
                    for item in items:
                        item["category"] = category
                        all_data.append(item)

            df = pd.DataFrame(all_data)
            df.to_csv(output_path, index=False)

            if apgi_logger:
                apgi_logger.logger.info(f"Data exported to CSV: {output_path}")
            return True

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"CSV export failed: {e}")
            return False

    def _export_excel(self, data: Dict, filename: str) -> bool:
        """Export data to Excel format with multiple sheets."""
        try:
            output_path = Path("exports") / filename
            output_path.parent.mkdir(exist_ok=True)

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                for category, items in data.items():
                    if isinstance(items, list) and items:
                        df = pd.DataFrame(items)
                        df.to_excel(
                            writer, sheet_name=category[:31], index=False
                        )  # Excel sheet name limit

            if apgi_logger:
                apgi_logger.logger.info(f"Data exported to Excel: {output_path}")
            return True

        except ImportError:
            if apgi_logger:
                apgi_logger.logger.error(
                    "Excel export requires openpyxl. Install with: pip install openpyxl"
                )
            return False
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Excel export failed: {e}")
            return False

    def _export_pdf(self, data: Dict, filename: str) -> bool:
        """Export data to PDF report format."""
        try:
            # This would require additional PDF generation libraries
            # For now, return a placeholder implementation
            if apgi_logger:
                apgi_logger.logger.warning("PDF export not yet implemented")
            return False

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"PDF export failed: {e}")
            return False

    def run(self, host: str = "127.0.0.1"):
        """Run the historical dashboard server."""
        try:
            if apgi_logger:
                apgi_logger.logger.info(
                    f"Starting historical dashboard on {host}:{self.port}"
                )

            self.app.run_server(
                host=host, port=self.port, debug=False, use_reloader=False
            )

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error running historical dashboard: {e}")
            raise


def create_historical_dashboard(port: int = 8051) -> HistoricalDashboard:
    """Create and return a historical dashboard instance."""
    return HistoricalDashboard(port=port)


import os

if __name__ == "__main__":
    # Run the historical dashboard when executed directly
    # Check if running in production mode (daemon/server mode)
    # Default to test mode for validation - set APGI_PRODUCTION_MODE=1 for server mode

    production_mode = os.environ.get("APGI_PRODUCTION_MODE", "0") == "1"

    if production_mode:
        # Run the full dashboard server (production mode)
        dashboard = create_historical_dashboard()
        dashboard.run()
    else:
        # Test mode: just initialize and verify the dashboard works
        print("Starting APGI Historical Dashboard in test mode...")
        try:
            dashboard = create_historical_dashboard()
            print(f"✓ Dashboard initialized successfully on port {dashboard.port}")
            print(f"✓ Database path: {dashboard.db_path}")
            print("✓ Test mode complete (server not started)")
        except Exception as e:
            print(f"✗ Dashboard initialization failed: {e}")
            raise
