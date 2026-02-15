"""
APGI Automated Report Generation
================================

Comprehensive automated report generation system for APGI validation results,
performance metrics, and analysis summaries in PDF and HTML formats.
"""

import base64
import json
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader

# PDF generation
try:
    from weasyprint import CSS, HTML

    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    WEASYPRINT_AVAILABLE = False
    # Handle different types of import errors gracefully
    # Use warnings since logger may not be available yet
    error_msg = str(e).lower()
    if (
        "libgobject" in error_msg
        or "library" in error_msg
        or "cannot load library" in error_msg
    ):
        # System library error - this is expected in some environments
        warnings.warn(
            "WeasyPrint system dependencies not available. "
            "PDF generation disabled. Install system dependencies with: brew install gtk+3",
            UserWarning,
        )
    else:
        # Standard import error
        warnings.warn(
            f"WeasyPrint import failed: {e}. "
            "PDF generation disabled. Install with: pip install weasyprint",
            UserWarning,
        )

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


class ReportGenerator:
    """Automated report generation for APGI framework."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("apgi_output/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Template directory
        self.template_dir = Path(__file__).parent / "templates"
        self.template_dir.mkdir(exist_ok=True)

        # Create default templates
        self._create_default_templates()

        # Setup Jinja2 environment
        self.jinja_env = Environment(loader=FileSystemLoader(str(self.template_dir)))

    def _create_default_templates(self):
        """Create default HTML and report templates."""

        # HTML template
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }
        .section { margin: 30px 0; }
        .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { color: #7f8c8d; font-size: 14px; }
        .success { color: #27ae60; }
        .warning { color: #f39c12; }
        .error { color: #e74c3c; }
        .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .table th { background-color: #f8f9fa; }
        .chart-container { text-align: center; margin: 20px 0; }
        .footer { margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 12px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .alert { padding: 10px; border-radius: 5px; margin: 10px 0; }
        .alert-high { background-color: #fdf2f2; border-left: 4px solid #e74c3c; }
        .alert-medium { background-color: #fef9e7; border-left: 4px solid #f39c12; }
        .alert-low { background-color: #f0f9ff; border-left: 4px solid #3498db; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated on {{ generation_date }}</p>
        <p>APGI Framework Validation Report</p>
    </div>

    {% if summary %}
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="grid">
            <div class="metric-card">
                <div class="metric-value">{{ summary.total_protocols }}</div>
                <div class="metric-label">Total Protocols</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {{ 'success' if summary.passed_protocols > summary.failed_protocols else 'error' }}">
                    {{ summary.passed_protocols }}/{{ summary.total_protocols }}
                </div>
                <div class="metric-label">Protocols Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {{ 'warning' if summary.avg_performance > 5 else 'success' }}">
                    {{ "%.2f"|format(summary.avg_performance) }}s
                </div>
                <div class="metric-label">Average Performance</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {{ 'success' if summary.error_rate < 0.1 else 'error' }}">
                    {{ "%.1f"|format(summary.error_rate * 100) }}%
                </div>
                <div class="metric-label">Error Rate</div>
            </div>
        </div>
    </div>
    {% endif %}

    {% if validation_results %}
    <div class="section">
        <h2>Validation Results</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>Protocol</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Key Metrics</th>
                    <th>Issues</th>
                </tr>
            </thead>
            <tbody>
                {% for result in validation_results %}
                <tr>
                    <td>{{ result.name }}</td>
                    <td class="{{ 'success' if result.passed else 'error' }}">
                        {{ 'PASSED' if result.passed else 'FAILED' }}
                    </td>
                    <td>{{ "%.3f"|format(result.duration) }}s</td>
                    <td>{{ result.key_metrics | join(', ') }}</td>
                    <td>{{ result.issues | length }} issues</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
    {% endif %}

    {% if performance_metrics %}
    <div class="section">
        <h2>Performance Analysis</h2>
        <div class="grid">
            {% for metric in performance_metrics %}
            <div class="metric-card">
                <div class="metric-value">{{ "%.3f"|format(metric.value) }} {{ metric.unit }}</div>
                <div class="metric-label">{{ metric.name }}</div>
                <div class="{{ 'success' if metric.status == 'good' else 'warning' if metric.status == 'warning' else 'error' }}">
                    {{ metric.status.upper() }}
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}

    {% if bottlenecks %}
    <div class="section">
        <h2>Performance Bottlenecks</h2>
        {% for bottleneck in bottlenecks %}
        <div class="alert alert-{{ bottleneck.severity }}">
            <strong>{{ bottleneck.type | title }}:</strong> {{ bottleneck.name }}
            <br>{{ bottleneck.description }}
            {% if bottleneck.recommendations %}
            <br><strong>Recommendations:</strong>
            <ul>
                {% for rec in bottleneck.recommendations %}
                <li>{{ rec }}</li>
                {% endfor %}
            </ul>
            {% endif %}
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if charts %}
    <div class="section">
        <h2>Visualizations</h2>
        {% for chart in charts %}
        <div class="chart-container">
            <h3>{{ chart.title }}</h3>
            <img src="data:image/png;base64,{{ chart.data }}" alt="{{ chart.title }}" style="max-width: 100%; height: auto;">
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {% if recommendations %}
    <div class="section">
        <h2>Recommendations</h2>
        {% for rec in recommendations %}
        <div class="alert alert-{{ rec.priority }}">
            <strong>{{ rec.type | title }}:</strong> {{ rec.message }}
            <br><em>Priority: {{ rec.priority | title }}</em>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="footer">
        <p>Generated by APGI Framework Automated Report Generator</p>
        <p>Report ID: {{ report_id }}</p>
    </div>
</body>
</html>
        """

        with open(self.template_dir / "report_template.html", "w") as f:
            f.write(html_template)

        # Summary template
        summary_template = """
# {{ title }}

**Generated:** {{ generation_date }}
**Report ID:** {{ report_id }}

## Executive Summary

{% if summary %}
- **Total Protocols:** {{ summary.total_protocols }}
- **Passed:** {{ summary.passed_protocols }}/{{ summary.total_protocols }}
- **Average Performance:** {{ "%.2f"|format(summary.avg_performance) }}s
- **Error Rate:** {{ "%.1f"|format(summary.error_rate * 100) }}%
{% endif %}

## Key Findings

{% for finding in key_findings %}
- {{ finding }}
{% endfor %}

## Recommendations

{% for rec in recommendations %}
1. **{{ rec.type | title }}** ({{ rec.priority | title }}): {{ rec.message }}
{% endfor %}

---
*Generated by APGI Framework Automated Report Generator*
        """

        with open(self.template_dir / "summary_template.md", "w") as f:
            f.write(summary_template)

    def _create_performance_charts(self) -> List[Dict[str, str]]:
        """Create performance visualization charts."""
        charts = []

        try:
            # Function performance chart
            if performance_profiler.function_profiles:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Top functions by total time
                top_functions = performance_profiler.get_top_functions("total_time", 8)
                if top_functions:
                    names = [f.name.split(".")[-1] for f in top_functions]
                    times = [f.total_time for f in top_functions]

                    ax1.barh(names, times)
                    ax1.set_xlabel("Total Time (s)")
                    ax1.set_title("Top Functions by Total Time")

                    # Add value labels
                    for i, v in enumerate(times):
                        ax1.text(v + 0.01, i, f"{v:.2f}s", va="center")

                # Function call counts
                call_counts = [f.call_count for f in top_functions]
                ax2.barh(names, call_counts, color="orange")
                ax2.set_xlabel("Call Count")
                ax2.set_title("Function Call Counts")

                # Add value labels
                for i, v in enumerate(call_counts):
                    ax2.text(v + max(call_counts) * 0.01, i, str(v), va="center")

                plt.tight_layout()

                # Convert to base64
                buffer = BytesIO()
                plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
                buffer.seek(0)
                chart_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                charts.append(
                    {"title": "Function Performance Overview", "data": chart_data}
                )

            # System metrics chart
            system_metrics = performance_profiler.system_monitor.get_metrics_summary()
            if system_metrics:
                fig, ax = plt.subplots(figsize=(10, 6))

                metrics = [
                    "cpu_percent_mean",
                    "memory_percent_mean",
                    "process_cpu_percent_mean",
                ]
                labels = ["CPU %", "Memory %", "Process CPU %"]
                values = [system_metrics.get(m, 0) for m in metrics]

                bars = ax.bar(labels, values, color=["blue", "red", "orange"])
                ax.set_ylabel("Percentage")
                ax.set_title("System Resource Usage")
                ax.set_ylim(0, 100)

                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 1,
                        f"{value:.1f}%",
                        ha="center",
                        va="bottom",
                    )

                plt.tight_layout()

                # Convert to base64
                buffer = BytesIO()
                plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
                buffer.seek(0)
                chart_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()

                charts.append({"title": "System Resource Usage", "data": chart_data})

        except Exception as e:
            apgi_logger.logger.warning(f"Error creating performance charts: {e}")

        return charts

    def _analyze_validation_data(
        self, validation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze validation data and extract insights."""
        summary = {
            "total_protocols": len(validation_data.get("protocols", {})),
            "passed_protocols": 0,
            "failed_protocols": 0,
            "total_duration": 0,
            "avg_performance": 0,
            "error_rate": 0,
            "issues_found": [],
        }

        protocols = validation_data.get("protocols", {})
        total_duration = 0
        total_errors = 0
        total_calls = 0

        for protocol_name, protocol_data in protocols.items():
            status = protocol_data.get("status", "unknown")
            duration = protocol_data.get("duration", 0)
            errors = protocol_data.get("errors", [])

            total_duration += duration
            total_errors += len(errors)

            if status == "passed":
                summary["passed_protocols"] += 1
            elif status == "failed":
                summary["failed_protocols"] += 1
                summary["issues_found"].append(
                    f"{protocol_name}: {protocol_data.get('error_message', 'Unknown error')}"
                )

        summary["total_duration"] = total_duration
        summary["avg_performance"] = total_duration / max(summary["total_protocols"], 1)

        # Calculate error rate from function profiles
        if performance_profiler.function_profiles:
            for profile in performance_profiler.function_profiles.values():
                total_calls += profile.call_count
                total_errors += profile.errors

        summary["error_rate"] = total_errors / max(total_calls, 1)

        return summary

    def _extract_performance_metrics(self) -> List[Dict[str, Any]]:
        """Extract key performance metrics."""
        metrics = []

        # System metrics
        system_summary = performance_profiler.system_monitor.get_metrics_summary()
        if system_summary:
            for key, value in system_summary.items():
                if "mean" in key:
                    metric_name = key.replace("_mean", "").replace("_", " ").title()

                    # Determine status
                    if "cpu" in key.lower():
                        status = (
                            "good"
                            if value < 70
                            else "warning" if value < 90 else "critical"
                        )
                    elif "memory" in key.lower():
                        status = (
                            "good"
                            if value < 70
                            else "warning" if value < 90 else "critical"
                        )
                    else:
                        status = "good"

                    metrics.append(
                        {
                            "name": metric_name,
                            "value": value,
                            "unit": "%",
                            "status": status,
                        }
                    )

        # Function performance metrics
        if performance_profiler.function_profiles:
            total_functions = len(performance_profiler.function_profiles)
            total_calls = sum(
                p.call_count for p in performance_profiler.function_profiles.values()
            )
            total_time = sum(
                p.total_time for p in performance_profiler.function_profiles.values()
            )
            avg_time = total_time / max(total_functions, 1)

            metrics.extend(
                [
                    {
                        "name": "Total Functions",
                        "value": total_functions,
                        "unit": "count",
                        "status": "good",
                    },
                    {
                        "name": "Total Function Calls",
                        "value": total_calls,
                        "unit": "count",
                        "status": "good",
                    },
                    {
                        "name": "Average Function Time",
                        "value": avg_time,
                        "unit": "s",
                        "status": (
                            "good"
                            if avg_time < 1.0
                            else "warning" if avg_time < 5.0 else "critical"
                        ),
                    },
                ]
            )

        return metrics

    def _generate_recommendations(
        self, validation_summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Validation recommendations
        if validation_summary["failed_protocols"] > 0:
            recommendations.append(
                {
                    "type": "validation",
                    "priority": "high",
                    "message": f"{validation_summary['failed_protocols']} validation protocols failed. Review error logs and fix issues.",
                }
            )

        if validation_summary["avg_performance"] > 10:
            recommendations.append(
                {
                    "type": "performance",
                    "priority": "medium",
                    "message": f"Average protocol execution time is {validation_summary['avg_performance']:.2f}s. Consider optimization.",
                }
            )

        if validation_summary["error_rate"] > 0.1:
            recommendations.append(
                {
                    "type": "reliability",
                    "priority": "high",
                    "message": f"Error rate is {validation_summary['error_rate']:.1%}. Improve error handling and input validation.",
                }
            )

        # Performance recommendations
        performance_report = performance_profiler.generate_performance_report()
        for bottleneck in performance_report.get("bottlenecks", []):
            if bottleneck["severity"] == "high":
                recommendations.append(
                    {
                        "type": "optimization",
                        "priority": "high",
                        "message": f"Critical bottleneck detected: {bottleneck['name']} ({bottleneck['value']:.2f} {bottleneck['unit']})",
                    }
                )

        # System recommendations
        system_summary = performance_profiler.system_monitor.get_metrics_summary()
        if system_summary:
            cpu_mean = system_summary.get("cpu_percent_mean", 0)
            memory_mean = system_summary.get("memory_percent_mean", 0)

            if cpu_mean > 80:
                recommendations.append(
                    {
                        "type": "system",
                        "priority": "medium",
                        "message": f"High CPU usage detected ({cpu_mean:.1f}%). Consider computational optimization.",
                    }
                )

            if memory_mean > 80:
                recommendations.append(
                    {
                        "type": "system",
                        "priority": "medium",
                        "message": f"High memory usage detected ({memory_mean:.1f}%). Consider memory optimization.",
                    }
                )

        return recommendations

    def generate_html_report(
        self,
        validation_data: Optional[Dict[str, Any]] = None,
        title: str = "APGI Validation Report",
        include_charts: bool = True,
    ) -> Path:
        """Generate comprehensive HTML report."""

        # Generate report ID
        report_id = f"apgi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Analyze data
        validation_summary = self._analyze_validation_data(validation_data or {})
        performance_metrics = self._extract_performance_metrics()
        recommendations = self._generate_recommendations(validation_summary)

        # Create charts
        charts = self._create_performance_charts() if include_charts else []

        # Extract bottlenecks
        performance_report = performance_profiler.generate_performance_report()
        bottlenecks = performance_report.get("bottlenecks", [])

        # Prepare validation results for template
        validation_results = []
        if validation_data:
            for protocol_name, protocol_data in validation_data.get(
                "protocols", {}
            ).items():
                validation_results.append(
                    {
                        "name": protocol_name,
                        "passed": protocol_data.get("status") == "passed",
                        "duration": protocol_data.get("duration", 0),
                        "key_metrics": list(protocol_data.get("metrics", {}).keys())[
                            :3
                        ],
                        "issues": protocol_data.get("errors", []),
                    }
                )

        # Load template
        template = self.jinja_env.get_template("report_template.html")

        # Render report
        html_content = template.render(
            title=title,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_id=report_id,
            summary=validation_summary,
            validation_results=validation_results,
            performance_metrics=performance_metrics,
            bottlenecks=bottlenecks,
            charts=charts,
            recommendations=recommendations,
            key_findings=[
                f"{validation_summary['passed_protocols']} of {validation_summary['total_protocols']} protocols passed",
                f"Average execution time: {validation_summary['avg_performance']:.2f}s",
                f"Overall error rate: {validation_summary['error_rate']:.1%}%",
                f"Total functions profiled: {len(performance_profiler.function_profiles)}",
            ],
        )

        # Save report
        report_path = self.output_dir / f"{report_id}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        apgi_logger.logger.info(f"HTML report generated: {report_path}")
        return report_path

    def generate_pdf_report(
        self,
        validation_data: Optional[Dict[str, Any]] = None,
        title: str = "APGI Validation Report",
    ) -> Optional[Path]:
        """Generate PDF report (requires WeasyPrint)."""

        if not WEASYPRINT_AVAILABLE:
            apgi_logger.logger.warning(
                "WeasyPrint not available. Cannot generate PDF report."
            )
            return None

        try:
            # Generate HTML first
            html_path = self.generate_html_report(
                validation_data, title, include_charts=True
            )

            # Convert to PDF
            report_id = html_path.stem
            pdf_path = self.output_dir / f"{report_id}.pdf"

            # Read HTML content
            with open(html_path, encoding="utf-8") as f:
                html_content = f.read()

            # Generate PDF
            html_doc = HTML(string=html_content)
            css = CSS(string="""
                @page { margin: 2cm; size: A4; }
                body { font-size: 10pt; }
                .metric-card { page-break-inside: avoid; }
                .chart-container { page-break-inside: avoid; }
                table { page-break-inside: avoid; }
            """)

            html_doc.write_pdf(pdf_path, stylesheets=[css])

            apgi_logger.logger.info(f"PDF report generated: {pdf_path}")
            return pdf_path

        except Exception as e:
            apgi_logger.logger.error(f"Error generating PDF report: {e}")
            return None

    def generate_summary_report(
        self,
        validation_data: Optional[Dict[str, Any]] = None,
        title: str = "APGI Validation Summary",
    ) -> Path:
        """Generate brief markdown summary report."""

        # Generate report ID
        report_id = f"apgi_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Analyze data
        validation_summary = self._analyze_validation_data(validation_data or {})
        recommendations = self._generate_recommendations(validation_summary)

        # Load template
        template = self.jinja_env.get_template("summary_template.md")

        # Render report
        markdown_content = template.render(
            title=title,
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            report_id=report_id,
            summary=validation_summary,
            recommendations=recommendations,
            key_findings=[
                f"{validation_summary['passed_protocols']} of {validation_summary['total_protocols']} protocols passed",
                f"Average execution time: {validation_summary['avg_performance']:.2f}s",
                f"Overall error rate: {validation_summary['error_rate']:.1%}%",
                f"Total functions profiled: {len(performance_profiler.function_profiles)}",
            ],
        )

        # Save report
        report_path = self.output_dir / f"{report_id}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        apgi_logger.logger.info(f"Summary report generated: {report_path}")
        return report_path

    def generate_comprehensive_report(
        self,
        validation_data: Optional[Dict[str, Any]] = None,
        title: str = "APGI Comprehensive Report",
    ) -> Dict[str, Path]:
        """Generate all report formats."""

        reports = {}

        # Generate HTML report
        reports["html"] = self.generate_html_report(validation_data, title)

        # Generate PDF report (if available)
        pdf_path = self.generate_pdf_report(validation_data, title)
        if pdf_path:
            reports["pdf"] = pdf_path

        # Generate summary report
        reports["summary"] = self.generate_summary_report(
            validation_data, f"{title} - Summary"
        )

        # Save raw data
        report_id = f"apgi_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        data_path = self.output_dir / f"{report_id}.json"

        report_data = {
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "validation_summary": self._analyze_validation_data(validation_data or {}),
            "performance_report": performance_profiler.generate_performance_report(),
            "validation_data": validation_data,
            "recommendations": self._generate_recommendations(
                self._analyze_validation_data(validation_data or {})
            ),
        }

        with open(data_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        reports["data"] = data_path

        apgi_logger.logger.info(
            f"Comprehensive report generated with {len(reports)} formats"
        )
        return reports


# Global report generator instance
report_generator = ReportGenerator()


# Convenience functions
def generate_quick_report(
    validation_data: Dict[str, Any], title: str = "APGI Quick Report"
) -> Path:
    """Generate a quick HTML report."""
    return report_generator.generate_html_report(validation_data, title)


def generate_full_report(
    validation_data: Dict[str, Any], title: str = "APGI Full Report"
) -> Dict[str, Path]:
    """Generate a comprehensive report in all formats."""
    return report_generator.generate_comprehensive_report(validation_data, title)
