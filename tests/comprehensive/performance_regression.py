"""
APGI Performance Regression Testing Module
==========================================

Performance regression testing with:
- Baseline establishment and tracking
- Performance trend analysis
- Automated degradation alerts
- Historical comparison reports

This module provides comprehensive performance monitoring for the APGI framework.
"""

import time
import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import numpy as np
import sys


@dataclass
class PerformanceBaseline:
    """Stores performance baseline data."""

    test_name: str
    mean_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    sample_count: int
    created_at: str
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceResult:
    """Result of a single performance test run."""

    test_name: str
    execution_time_ms: float
    timestamp: str
    baseline_mean_ms: Optional[float] = None
    deviation_percent: float = 0.0
    passed: bool = True
    alert_triggered: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Performance trend analysis results."""

    test_name: str
    slope: float  # Positive = getting slower
    r_squared: float
    recent_mean: float
    historical_mean: float
    trend_direction: str  # "improving", "stable", "degrading"
    confidence: float


@dataclass
class DegradationAlert:
    """Performance degradation alert."""

    test_name: str
    severity: str  # "warning", "critical"
    current_time_ms: float
    baseline_time_ms: float
    deviation_percent: float
    timestamp: str
    recommendation: str


class PerformanceBaselineManager:
    """Manages performance baselines and historical data."""

    def __init__(self, baseline_dir: Path = Path("reports/performance_baselines")):
        self.baseline_dir = baseline_dir
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = baseline_dir / "performance_history.json"
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.history: Dict[str, List[Dict]] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load existing baselines and history."""
        # Load baselines
        for baseline_file in self.baseline_dir.glob("*.baseline.json"):
            with open(baseline_file, "r") as f:
                data = json.load(f)
                baseline = PerformanceBaseline(**data)
                self.baselines[baseline.test_name] = baseline

        # Load history
        if self.history_file.exists():
            with open(self.history_file, "r") as f:
                self.history = json.load(f)

    def _save_baseline(self, baseline: PerformanceBaseline) -> None:
        """Save a baseline to disk."""
        baseline_file = self.baseline_dir / f"{baseline.test_name}.baseline.json"
        with open(baseline_file, "w") as f:
            json.dump(asdict(baseline), f, indent=2)
        self.baselines[baseline.test_name] = baseline

    def _save_history(self) -> None:
        """Save performance history to disk."""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def establish_baseline(
        self,
        test_name: str,
        test_func: Callable,
        warmup_runs: int = 3,
        measurement_runs: int = 10,
        metadata: Dict[str, Any] = None,
    ) -> PerformanceBaseline:
        """Establish a new performance baseline."""
        print(f"  Establishing baseline for {test_name}...")

        # Warmup
        for _ in range(warmup_runs):
            test_func()

        # Measure
        times = []
        for _ in range(measurement_runs):
            start = time.perf_counter()
            test_func()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        baseline = PerformanceBaseline(
            test_name=test_name,
            mean_time_ms=statistics.mean(times),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            min_time_ms=min(times),
            max_time_ms=max(times),
            sample_count=len(times),
            created_at=datetime.now().isoformat(),
            metadata=metadata or {},
        )

        self._save_baseline(baseline)
        print(
            f"    Baseline: {baseline.mean_time_ms:.2f}ms ± {baseline.std_dev_ms:.2f}ms"
        )

        return baseline

    def record_result(self, result: PerformanceResult) -> None:
        """Record a performance test result to history."""
        if result.test_name not in self.history:
            self.history[result.test_name] = []

        self.history[result.test_name].append(
            {
                "timestamp": result.timestamp,
                "execution_time_ms": result.execution_time_ms,
                "deviation_percent": result.deviation_percent,
                "passed": result.passed,
            }
        )

        # Keep only last 100 results
        self.history[result.test_name] = self.history[result.test_name][-100:]
        self._save_history()

    def get_baseline(self, test_name: str) -> Optional[PerformanceBaseline]:
        """Get baseline for a test."""
        return self.baselines.get(test_name)

    def get_history(self, test_name: str, limit: int = 50) -> List[Dict]:
        """Get performance history for a test."""
        return self.history.get(test_name, [])[-limit:]


class PerformanceTrendAnalyzer:
    """Analyzes performance trends over time."""

    def __init__(self, baseline_manager: PerformanceBaselineManager):
        self.baseline_manager = baseline_manager

    def analyze_trend(
        self, test_name: str, window_size: int = 20
    ) -> Optional[TrendAnalysis]:
        """Analyze performance trend for a test."""
        history = self.baseline_manager.get_history(test_name, limit=window_size)

        if len(history) < 10:
            return None

        times = [h["execution_time_ms"] for h in history]
        x = np.arange(len(times))

        # Linear regression
        slope, intercept, r_value, p_value, std_err = self._linear_regression(x, times)

        # Calculate trend direction
        recent = times[-window_size // 2 :]
        historical = times[: window_size // 2]

        recent_mean = statistics.mean(recent)
        historical_mean = statistics.mean(historical)

        # Determine trend
        if slope < -0.5:  # Getting faster
            direction = "improving"
            confidence = min(abs(slope) * 10, 1.0)
        elif slope > 0.5:  # Getting slower
            direction = "degrading"
            confidence = min(slope * 10, 1.0)
        else:
            direction = "stable"
            confidence = 1.0 - min(abs(slope) * 2, 1.0)

        return TrendAnalysis(
            test_name=test_name,
            slope=slope,
            r_squared=r_value**2,
            recent_mean=recent_mean,
            historical_mean=historical_mean,
            trend_direction=direction,
            confidence=confidence,
        )

    def _linear_regression(self, x: np.ndarray, y: List[float]):
        """Simple linear regression."""
        from scipy import stats

        return stats.linregress(x, y)

    def detect_anomalies(
        self, test_name: str, threshold_std: float = 2.0
    ) -> List[Dict]:
        """Detect anomalous performance results."""
        history = self.baseline_manager.get_history(test_name, limit=50)

        if len(history) < 10:
            return []

        times = [h["execution_time_ms"] for h in history]
        mean = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0

        anomalies = []
        for i, (h, t) in enumerate(zip(history, times)):
            if std_dev > 0 and abs(t - mean) > threshold_std * std_dev:
                anomalies.append(
                    {
                        "index": i,
                        "timestamp": h["timestamp"],
                        "value": t,
                        "deviation": (t - mean) / std_dev,
                    }
                )

        return anomalies


class PerformanceRegressionTester:
    """Main performance regression testing orchestrator."""

    # Alert thresholds
    WARNING_THRESHOLD = 20.0  # 20% slower
    CRITICAL_THRESHOLD = 50.0  # 50% slower

    def __init__(self, baseline_dir: Path = Path("reports/performance_baselines")):
        self.baseline_manager = PerformanceBaselineManager(baseline_dir)
        self.trend_analyzer = PerformanceTrendAnalyzer(self.baseline_manager)
        self.alerts: List[DegradationAlert] = []
        self.results: List[PerformanceResult] = []

    def run_test(
        self,
        test_name: str,
        test_func: Callable,
        establish_baseline_if_missing: bool = True,
    ) -> PerformanceResult:
        """Run a performance test and check for regression."""
        # Get baseline
        baseline = self.baseline_manager.get_baseline(test_name)

        if baseline is None and establish_baseline_if_missing:
            baseline = self.baseline_manager.establish_baseline(test_name, test_func)

        # Run test
        start = time.perf_counter()
        test_func()
        execution_time = (time.perf_counter() - start) * 1000

        # Calculate deviation
        if baseline:
            deviation = (
                (execution_time - baseline.mean_time_ms) / baseline.mean_time_ms
            ) * 100

            # Check thresholds
            alert_triggered = abs(deviation) > self.WARNING_THRESHOLD
            passed = deviation < self.CRITICAL_THRESHOLD

            if alert_triggered:
                self._create_alert(
                    test_name, execution_time, baseline.mean_time_ms, deviation
                )
        else:
            deviation = 0.0
            alert_triggered = False
            passed = True

        result = PerformanceResult(
            test_name=test_name,
            execution_time_ms=execution_time,
            timestamp=datetime.now().isoformat(),
            baseline_mean_ms=baseline.mean_time_ms if baseline else None,
            deviation_percent=deviation,
            passed=passed,
            alert_triggered=alert_triggered,
        )

        self.results.append(result)
        self.baseline_manager.record_result(result)

        return result

    def _create_alert(
        self, test_name: str, current: float, baseline: float, deviation: float
    ) -> None:
        """Create a degradation alert."""
        severity = "critical" if abs(deviation) > self.CRITICAL_THRESHOLD else "warning"

        if deviation > 0:
            recommendation = (
                "Investigate performance degradation - review recent changes"
            )
        else:
            recommendation = "Performance improved - consider updating baseline"

        alert = DegradationAlert(
            test_name=test_name,
            severity=severity,
            current_time_ms=current,
            baseline_time_ms=baseline,
            deviation_percent=deviation,
            timestamp=datetime.now().isoformat(),
            recommendation=recommendation,
        )

        self.alerts.append(alert)

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance regression tests."""
        print("=" * 80)
        print("APGI PERFORMANCE REGRESSION TESTING")
        print("=" * 80)
        print(f"Warning Threshold: {self.WARNING_THRESHOLD}% deviation")
        print(f"Critical Threshold: {self.CRITICAL_THRESHOLD}% deviation")

        # Define test functions
        tests = self._get_test_functions()

        print(f"\nRunning {len(tests)} performance tests...\n")

        for test_name, test_func in tests.items():
            print(f"[Performance Test] {test_name}...")
            result = self.run_test(test_name, test_func)

            status = "✓" if result.passed else "✗"
            if result.alert_triggered:
                status = "⚠"

            baseline_str = (
                f"(baseline: {result.baseline_mean_ms:.2f}ms)"
                if result.baseline_mean_ms
                else "(no baseline)"
            )
            print(
                f"  {status} {result.execution_time_ms:.2f}ms {baseline_str} ({result.deviation_percent:+.1f}%)"
            )

        # Analyze trends
        print("\n[Analyzing Trends]...")
        trends = self._analyze_all_trends()

        # Generate report
        report = self._generate_report(trends)

        return report

    def _get_test_functions(self) -> Dict[str, Callable]:
        """Get all performance test functions."""
        import numpy as np

        def test_eeg_filter():
            # Simulate EEG filtering
            signal = np.random.randn(64, 1000)
            kernel = np.ones(10) / 10
            filtered = np.array([np.convolve(ch, kernel, mode="same") for ch in signal])
            return filtered

        def test_matrix_multiplication():
            # Matrix operation typical in APGI
            A = np.random.randn(100, 100)
            B = np.random.randn(100, 100)
            return np.dot(A, B)

        def test_fft_transform():
            # FFT for spectral analysis
            signal = np.random.randn(1000)
            return np.fft.fft(signal)

        def test_data_normalization():
            # Data preprocessing
            data = np.random.randn(1000)
            return (data - np.mean(data)) / (np.std(data) + 1e-10)

        def test_surprise_computation():
            # Surprise computation
            error = np.random.randn(100)
            return 0.5 * error**2

        def test_threshold_update():
            # Threshold dynamics
            precision = 0.8
            surprise = 0.1
            return 0.5 * (1 / (1 + precision)) + 0.1 * surprise

        return {
            "eeg_filter": test_eeg_filter,
            "matrix_multiplication": test_matrix_multiplication,
            "fft_transform": test_fft_transform,
            "data_normalization": test_data_normalization,
            "surprise_computation": test_surprise_computation,
            "threshold_update": test_threshold_update,
        }

    def _analyze_all_trends(self) -> Dict[str, TrendAnalysis]:
        """Analyze trends for all tests."""
        trends = {}

        for test_name in self._get_test_functions().keys():
            trend = self.trend_analyzer.analyze_trend(test_name)
            if trend:
                trends[test_name] = trend
                direction_icon = {
                    "improving": "📈",
                    "stable": "➡️",
                    "degrading": "📉",
                }.get(trend.trend_direction, "❓")
                print(
                    f"  {direction_icon} {test_name}: {trend.trend_direction} (confidence: {trend.confidence:.1%})"
                )

        return trends

    def _generate_report(self, trends: Dict[str, TrendAnalysis]) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        alerts = len(self.alerts)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "passed": passed,
                "failed": failed,
                "alerts": alerts,
                "pass_rate": passed / len(self.results) * 100 if self.results else 0,
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "execution_time_ms": r.execution_time_ms,
                    "baseline_mean_ms": r.baseline_mean_ms,
                    "deviation_percent": r.deviation_percent,
                    "passed": r.passed,
                    "alert_triggered": r.alert_triggered,
                }
                for r in self.results
            ],
            "trends": [
                {
                    "test_name": t.test_name,
                    "direction": t.trend_direction,
                    "confidence": t.confidence,
                    "recent_mean": t.recent_mean,
                    "historical_mean": t.historical_mean,
                    "slope": t.slope,
                }
                for t in trends.values()
            ],
            "alerts": [
                {
                    "test_name": a.test_name,
                    "severity": a.severity,
                    "deviation_percent": a.deviation_percent,
                    "recommendation": a.recommendation,
                }
                for a in self.alerts
            ],
            "thresholds": {
                "warning": self.WARNING_THRESHOLD,
                "critical": self.CRITICAL_THRESHOLD,
            },
        }

        # Print summary
        print(f"\n{'=' * 80}")
        print("PERFORMANCE REGRESSION TEST SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']} ✓")
        print(
            f"Failed: {report['summary']['failed']} {'✓' if report['summary']['failed'] == 0 else '✗'}"
        )
        print(
            f"Alerts: {report['summary']['alerts']} {'✓' if report['summary']['alerts'] == 0 else '⚠️'}"
        )

        if self.alerts:
            print("\n⚠️ Performance Alerts:")
            for alert in self.alerts[:5]:
                icon = "🔴" if alert.severity == "critical" else "🟡"
                print(
                    f"  {icon} {alert.test_name}: {alert.deviation_percent:+.1f}% ({alert.severity})"
                )

        # Save report
        report_path = Path("reports/performance_regression_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {report_path}")

        return report

    def generate_html_report(self, report: Dict[str, Any]) -> Path:
        """Generate HTML performance report."""
        html_path = Path("reports/performance_regression_report.html")

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>APGI Performance Regression Report</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f5f7fa; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; }}
        .card {{ background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .pass {{ color: #27ae60; }}
        .fail {{ color: #e74c3c; }}
        .warning {{ color: #f39c12; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 APGI Performance Regression Report</h1>
        <p>Generated: {report['timestamp']}</p>
    </div>
    
    <div class="card">
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td>{report['summary']['total_tests']}</td></tr>
            <tr><td>Passed</td><td class="pass">{report['summary']['passed']}</td></tr>
            <tr><td>Failed</td><td class="{'pass' if report['summary']['failed'] == 0 else 'fail'}">{report['summary']['failed']}</td></tr>
            <tr><td>Alerts</td><td class="{'pass' if report['summary']['alerts'] == 0 else 'warning'}">{report['summary']['alerts']}</td></tr>
            <tr><td>Pass Rate</td><td>{report['summary']['pass_rate']:.1f}%</td></tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Test Results</h2>
        <table>
            <tr>
                <th>Test</th>
                <th>Time (ms)</th>
                <th>Baseline (ms)</th>
                <th>Deviation</th>
                <th>Status</th>
            </tr>
"""

        for r in report["results"]:
            status_class = "pass" if r["passed"] else "fail"
            status_icon = "✓" if r["passed"] else "✗"
            deviation_color = (
                "pass"
                if abs(r["deviation_percent"]) < 20
                else "warning" if abs(r["deviation_percent"]) < 50 else "fail"
            )

            html_content += f"""
            <tr>
                <td>{r['test_name']}</td>
                <td>{r['execution_time_ms']:.2f}</td>
                <td>{r['baseline_mean_ms']:.2f if r['baseline_mean_ms'] else 'N/A'}</td>
                <td class="{deviation_color}">{r['deviation_percent']:+.1f}%</td>
                <td class="{status_class}">{status_icon}</td>
            </tr>
"""

        html_content += """
        </table>
    </div>
</body>
</html>"""

        with open(html_path, "w") as f:
            f.write(html_content)

        return html_path


def run_performance_regression_tests() -> Dict[str, Any]:
    """Entry point for performance regression testing."""
    tester = PerformanceRegressionTester()
    report = tester.run_all_tests()

    # Generate HTML report
    html_path = tester.generate_html_report(report)
    print(f"\n📊 HTML report: {html_path}")

    return report


if __name__ == "__main__":
    results = run_performance_regression_tests()

    # Exit with error code if any critical alerts
    critical_alerts = sum(
        1 for a in results.get("alerts", []) if a["severity"] == "critical"
    )
    sys.exit(1 if critical_alerts > 0 else 0)
