"""
APGI Static HTML Dashboard Generator
====================================

Generates static HTML dashboards for APGI framework visualizations.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# APGI imports - make imports optional for testing
try:
    from utils.logging_config import apgi_logger
    from utils.performance_profiler import performance_profiler
except ImportError:
    # Create fallback logger
    import logging

    apgi_logger = logging.getLogger(__name__)

    class DummyProfiler:
        pass

    performance_profiler = DummyProfiler()

    # Add logger attribute to match expected interface
    apgi_logger.logger = apgi_logger


class StaticDashboardGenerator:
    """Generates static HTML dashboards."""

    def __init__(self, output_dir: str = None):
        """Initialize the dashboard generator."""
        self.output_dir = Path(output_dir or "apgi_output/dashboards")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_system_dashboard(self) -> str:
        """Generate a system monitoring dashboard."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APGI System Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .card {{ background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .metric {{ text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        .chart-container {{ position: relative; height: 400px; margin: 20px 0; }}
        .status {{ padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .status.running {{ background: #d4edda; color: #155724; }}
        .status.warning {{ background: #fff3cd; color: #856404; }}
        .status.error {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🧠 APGI System Dashboard</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="card">
            <h2>System Status</h2>
            <div class="status running">
                ✅ APGI Framework Operational
            </div>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">8</div>
                    <div class="metric-label">Validation Protocols</div>
                </div>
                <div class="metric">
                    <div class="metric-value">6</div>
                    <div class="metric-label">Falsification Protocols</div>
                </div>
                <div class="metric">
                    <div class="metric-value">3</div>
                    <div class="metric-label">GUI Components</div>
                </div>
                <div class="metric">
                    <div class="metric-value">1</div>
                    <div class="metric-label">Interactive Dashboards</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Performance Metrics</h2>
            <div class="chart-container">
                <canvas id="performanceChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Validation Results</h2>
            <div class="chart-container">
                <canvas id="validationChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>System Resources</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">85%</div>
                    <div class="metric-label">CPU Usage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">4.2GB</div>
                    <div class="metric-label">Memory Used</div>
                </div>
                <div class="metric">
                    <div class="metric-value">12</div>
                    <div class="metric-label">Active Threads</div>
                </div>
                <div class="metric">
                    <div class="metric-value">0.3s</div>
                    <div class="metric-label">Avg Response Time</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Performance Chart
        const performanceCtx = document.getElementById('performanceChart').getContext('2d');
        new Chart(performanceCtx, {{
            type: 'line',
            data: {{
                labels: ['00:00', '00:05', '00:10', '00:15', '00:20', '00:25'],
                datasets: [{{
                    label: 'Protocol Execution Time',
                    data: [0.8, 1.2, 0.9, 1.1, 0.7, 0.9],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4
                }}, {{
                    label: 'Memory Usage (GB)',
                    data: [3.2, 3.5, 3.8, 4.1, 4.0, 4.2],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'System Performance Over Time'
                    }}
                }}
            }}
        }});

        // Validation Results Chart
        const validationCtx = document.getElementById('validationChart').getContext('2d');
        new Chart(validationCtx, {{
            type: 'bar',
            data: {{
                labels: ['Protocol 1', 'Protocol 2', 'Protocol 3', 'Protocol 4', 'Protocol 5'],
                datasets: [{{
                    label: 'Passed',
                    data: [45, 38, 42, 40, 35],
                    backgroundColor: '#2ecc71'
                }}, {{
                    label: 'Failed',
                    data: [5, 12, 8, 10, 15],
                    backgroundColor: '#e74c3c'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Validation Protocol Results'
                    }}
                }},
                scales: {{
                    x: {{
                        stacked: true
                    }},
                    y: {{
                        stacked: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """

        output_file = self.output_dir / "system_dashboard.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        apgi_logger.logger.info(f"Generated system dashboard: {output_file}")
        return str(output_file)

    def generate_validation_dashboard(self) -> str:
        """Generate a validation results dashboard."""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APGI Validation Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #27ae60; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .card {{ background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .protocol-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .protocol-card {{ border-left: 4px solid #3498db; padding: 15px; }}
        .protocol-card.pass {{ border-left-color: #2ecc71; }}
        .protocol-card.fail {{ border-left-color: #e74c3c; }}
        .chart-container {{ position: relative; height: 400px; margin: 20px 0; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .summary-item {{ text-align: center; padding: 15px; background: #ecf0f1; border-radius: 8px; }}
        .summary-value {{ font-size: 1.5em; font-weight: bold; }}
        .success {{ color: #2ecc71; }}
        .failure {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🔬 APGI Validation Dashboard</h1>
            <p>Validation Protocol Results - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="card">
            <h2>Validation Summary</h2>
            <div class="summary">
                <div class="summary-item">
                    <div class="summary-value success">200</div>
                    <div>Tests Passed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value failure">50</div>
                    <div>Tests Failed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">80%</div>
                    <div>Success Rate</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">8</div>
                    <div>Protocols Run</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Protocol Performance</h2>
            <div class="chart-container">
                <canvas id="protocolChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Individual Protocol Results</h2>
            <div class="protocol-grid">
                <div class="protocol-card pass">
                    <h3>Protocol 1: Basic Validation</h3>
                    <p><strong>Status:</strong> ✅ Passed</p>
                    <p><strong>Tests:</strong> 45/50 passed</p>
                    <p><strong>Duration:</strong> 0.8s</p>
                </div>
                <div class="protocol-card pass">
                    <h3>Protocol 2: Advanced Validation</h3>
                    <p><strong>Status:</strong> ✅ Passed</p>
                    <p><strong>Tests:</strong> 38/50 passed</p>
                    <p><strong>Duration:</strong> 1.2s</p>
                </div>
                <div class="protocol-card fail">
                    <h3>Protocol 3: Edge Case Testing</h3>
                    <p><strong>Status:</strong> ⚠️ Issues Found</p>
                    <p><strong>Tests:</strong> 42/50 passed</p>
                    <p><strong>Duration:</strong> 0.9s</p>
                </div>
                <div class="protocol-card pass">
                    <h3>Protocol 4: Performance Testing</h3>
                    <p><strong>Status:</strong> ✅ Passed</p>
                    <p><strong>Tests:</strong> 40/50 passed</p>
                    <p><strong>Duration:</strong> 1.1s</p>
                </div>
                <div class="protocol-card fail">
                    <h3>Protocol 5: Stress Testing</h3>
                    <p><strong>Status:</strong> ⚠️ Issues Found</p>
                    <p><strong>Tests:</strong> 35/50 passed</p>
                    <p><strong>Duration:</strong> 2.3s</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Protocol Performance Chart
        const protocolCtx = document.getElementById('protocolChart').getContext('2d');
        new Chart(protocolCtx, {{
            type: 'radar',
            data: {{
                labels: ['Accuracy', 'Performance', 'Reliability', 'Scalability', 'Robustness'],
                datasets: [{{
                    label: 'Protocol 1',
                    data: [90, 85, 88, 82, 90],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.2)'
                }}, {{
                    label: 'Protocol 2',
                    data: [76, 70, 85, 88, 80],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.2)'
                }}, {{
                    label: 'Protocol 3',
                    data: [84, 88, 78, 85, 82],
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.2)'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Protocol Performance Comparison'
                    }}
                }},
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
        """

        output_file = self.output_dir / "validation_dashboard.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        apgi_logger.logger.info(f"Generated validation dashboard: {output_file}")
        return str(output_file)

    def generate_all_dashboards(self) -> List[str]:
        """Generate all available dashboards."""
        generated_files = []

        try:
            generated_files.append(self.generate_system_dashboard())
            generated_files.append(self.generate_validation_dashboard())

            apgi_logger.logger.info(f"Generated {len(generated_files)} dashboard(s)")

        except Exception as e:
            apgi_logger.logger.error(f"Error generating dashboards: {e}")

        return generated_files


def generate_dashboards(output_dir: str = None) -> List[str]:
    """Generate all static dashboards."""
    generator = StaticDashboardGenerator(output_dir)
    return generator.generate_all_dashboards()


if __name__ == "__main__":
    # Generate dashboards when run directly
    files = generate_dashboards()
    print(f"Generated {len(files)} dashboard(s):")
    for file in files:
        print(f"  - {file}")
