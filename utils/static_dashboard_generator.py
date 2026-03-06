#!/usr/bin/env python3
"""
APGI Theory Framework - Static Dashboard Generator
===============================================

Generates static HTML dashboards for APGI framework components including
system monitoring, validation results, and performance metrics.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    PLOTLY_AVAILABLE = False  # Plotly not actually used in this module
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from utils.logging_config import apgi_logger
except ImportError:
    apgi_logger = None


class StaticDashboardGenerator:
    """Generates static HTML dashboards for APGI framework."""

    def __init__(self, output_dir: str):
        """
        Initialize dashboard generator.

        Args:
            output_dir: Directory to save generated dashboards
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if apgi_logger:
            apgi_logger.logger.info(
                f"Initialized static dashboard generator with output dir: {output_dir}"
            )

    def generate_system_dashboard(self) -> str:
        """Generate system monitoring dashboard."""
        try:
            # Collect system metrics
            system_metrics = self._collect_system_metrics()

            # Generate HTML content
            html_content = self._generate_system_html(system_metrics)

            # Save dashboard
            output_file = self.output_dir / "system_dashboard.html"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            return str(output_file)

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error generating system dashboard: {e}")
            raise

    def generate_validation_dashboard(self) -> str:
        """Generate validation results dashboard."""
        try:
            # Collect validation data
            validation_data = self._collect_validation_data()

            # Generate HTML content
            html_content = self._generate_validation_html(validation_data)

            # Save dashboard
            output_file = self.output_dir / "validation_dashboard.html"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)

            return str(output_file)

        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Error generating validation dashboard: {e}")
            raise

    def _collect_system_metrics(self) -> Dict:
        """Collect current system metrics."""
        try:
            import psutil

            return {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / (1024**3),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_usage": psutil.disk_usage("/").percent,
                "network_connections": len(psutil.net_connections()),
            }
        except ImportError:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": "psutil not available for system metrics",
            }
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": f"Error collecting system metrics: {e}",
            }

    def _collect_validation_data(self) -> Dict:
        """Collect validation results data."""
        try:
            # Look for validation results in results directory
            results_dir = Path("validation_results")
            if not results_dir.exists():
                return {"error": "No validation results found"}

            validation_files = list(results_dir.glob("*.json"))
            validation_data = []

            for vf in validation_files[:10]:  # Limit to 10 most recent
                try:
                    with open(vf, "r") as f:
                        data = json.load(f)
                        validation_data.append(
                            {
                                "file": vf.name,
                                "timestamp": data.get("timestamp", "unknown"),
                                "results": data.get("results", {}),
                            }
                        )
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    if apgi_logger:
                        apgi_logger.logger.warning(
                            f"Error reading validation file {vf}: {e}"
                        )
                    continue

            return {
                "timestamp": datetime.now().isoformat(),
                "validation_runs": validation_data,
                "total_files": len(validation_files),
            }

        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "error": f"Error collecting validation data: {e}",
            }

    def _generate_system_html(self, metrics: Dict) -> str:
        """Generate HTML for system dashboard."""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>APGI System Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .error {{ color: red; }}
        .timestamp {{ color: #666; font-size: 0.8em; }}
    </style>
</head>
<body>
    <h1>APGI System Dashboard</h1>
    <div class="timestamp">Generated: {metrics.get('timestamp', 'unknown')}</div>

    {'<div class="error">' + metrics.get('error', '') + '</div>' if 'error' in metrics else ''}

    {'<div class="metric"><h3>CPU Usage</h3><p>' + f"{metrics.get('cpu_percent', 0):.1f}%" + '</p></div>' if 'cpu_percent' in metrics else ''}

    {'<div class="metric"><h3>Memory Usage</h3><p>' + f"{metrics.get('memory_percent', 0):.1f}% ({metrics.get('memory_used_gb', 0):.1f} GB / {metrics.get('memory_total_gb', 0):.1f} GB)" + '</p></div>' if 'memory_percent' in metrics else ''}

    {'<div class="metric"><h3>Disk Usage</h3><p>' + f"{metrics.get('disk_usage', 0):.1f}%" + '</p></div>' if 'disk_usage' in metrics else ''}

    {'<div class="metric"><h3>Network Connections</h3><p>' + str(metrics.get('network_connections', 0)) + '</p></div>' if 'network_connections' in metrics else ''}
</body>
</html>
        """
        return html_template

    def _generate_validation_html(self, data: Dict) -> str:
        """Generate HTML for validation dashboard."""
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>APGI Validation Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .validation-run {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .error {{ color: red; }}
        .timestamp {{ color: #666; font-size: 0.8em; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>APGI Validation Dashboard</h1>
    <div class="timestamp">Generated: {data.get('timestamp', 'unknown')}</div>

    {'<div class="error">' + data.get('error', '') + '</div>' if 'error' in data else ''}

    {'<p>Total validation files: ' + str(data.get('total_files', 0)) + '</p>' if 'total_files' in data else ''}

    {''.join([f'''
    <div class="validation-run">
        <h3>{run.get('file', 'Unknown')}</h3>
        <p>Timestamp: {run.get('timestamp', 'unknown')}</p>
        <table>
            <tr><th>Protocol</th><th>Result</th></tr>
            {''.join([f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in run.get('results', {}).items()])}
        </table>
    </div>
    ''' for run in data.get('validation_runs', [])]) if 'validation_runs' in data else ''}
</body>
</html>
        """
        return html_template


def generate_dashboards(output_dir: str) -> List[str]:
    """
    Generate all static dashboards.

    Args:
        output_dir: Directory to save dashboards

    Returns:
        List of generated dashboard file paths
    """
    try:
        generator = StaticDashboardGenerator(output_dir)
        generated_files = []

        # Generate system dashboard
        try:
            system_file = generator.generate_system_dashboard()
            generated_files.append(system_file)
            if apgi_logger:
                apgi_logger.logger.info(f"Generated system dashboard: {system_file}")
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(f"Failed to generate system dashboard: {e}")

        # Generate validation dashboard
        try:
            validation_file = generator.generate_validation_dashboard()
            generated_files.append(validation_file)
            if apgi_logger:
                apgi_logger.logger.info(
                    f"Generated validation dashboard: {validation_file}"
                )
        except Exception as e:
            if apgi_logger:
                apgi_logger.logger.error(
                    f"Failed to generate validation dashboard: {e}"
                )

        return generated_files

    except Exception as e:
        if apgi_logger:
            apgi_logger.logger.error(f"Error in generate_dashboards: {e}")
        raise
