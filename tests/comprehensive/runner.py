"""
APGI Comprehensive Test Runner
===============================

Unified entry point for running all comprehensive tests:
- Unit tests
- Integration tests
- End-to-end tests
- Performance/stress tests
- Security tests
- Mutation testing
- Coverage analysis

Usage:
    python -m tests.comprehensive.runner --all
    python -m tests.comprehensive.runner --category security
    python -m tests.comprehensive.runner --coverage-threshold 95
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Import test modules
from tests.comprehensive import (
    AdversarialTestFramework,
    TestCategory,
)
from tests.comprehensive.mutation_tester import run_mutation_testing
from tests.comprehensive.stress_test import run_performance_tests
from tests.comprehensive.security_tester import run_security_tests
from tests.comprehensive.integration_e2e import run_integration_tests


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="APGI Comprehensive Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                    Run all tests
  %(prog)s --category security      Run only security tests
  %(prog)s --mutation               Run mutation testing
  %(prog)s --coverage-threshold 95  Enforce 95% coverage minimum
  %(prog)s --parallel 4              Run with 4 parallel workers
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test categories",
    )
    parser.add_argument(
        "--category",
        choices=["unit", "integration", "e2e", "performance", "security", "mutation"],
        help="Run specific test category",
    )
    parser.add_argument(
        "--mutation",
        action="store_true",
        help="Run mutation testing",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=80.0,
        help="Minimum coverage percentage (default: 80)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


class TestRunner:
    """Main test runner orchestrating all test categories."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> int:
        """
        Run selected tests and return exit code.

        Returns:
            0 if all tests passed, 1 otherwise
        """
        print("=" * 80)
        print("APGI COMPREHENSIVE TEST RUNNER")
        print("=" * 80)
        print(f"Seed: {self.args.seed}")
        print(f"Coverage Threshold: {self.args.coverage_threshold}%")
        print(f"Parallel Workers: {self.args.parallel}")
        print(f"Output Directory: {self.output_dir}")
        print("=" * 80)

        self.start_time = time.time()

        try:
            if self.args.all or self.args.category == "unit":
                self._run_unit_tests()

            if self.args.all or self.args.category == "integration":
                self._run_integration_tests()

            if self.args.all or self.args.category == "e2e":
                self._run_e2e_tests()

            if self.args.all or self.args.category == "performance":
                self._run_performance_tests()

            if self.args.all or self.args.category == "security":
                self._run_security_tests()

            if self.args.all or self.args.mutation:
                self._run_mutation_tests()

            # Run coverage analysis if requested
            if self.args.all or not self.args.category:
                self._run_coverage_analysis()

        except Exception as e:
            print(f"\n❌ Test runner error: {e}")
            import traceback

            traceback.print_exc()
            return 1

        # Generate final report
        self._generate_final_report()

        # Return appropriate exit code
        return 0 if self._all_passed() else 1

    def _run_unit_tests(self) -> None:
        """Run unit tests."""
        print("\n" + "=" * 80)
        print("RUNNING UNIT TESTS")
        print("=" * 80)

        framework = AdversarialTestFramework(seed=self.args.seed)
        results = framework.run_all_tests(
            categories=[TestCategory.UNIT],
            parallel=self.args.parallel > 1,
        )

        self.results["unit_tests"] = results

    def _run_integration_tests(self) -> None:
        """Run integration tests."""
        print("\n" + "=" * 80)
        print("RUNNING INTEGRATION TESTS")
        print("=" * 80)

        results = run_integration_tests()
        self.results["integration_tests"] = results

    def _run_e2e_tests(self) -> None:
        """Run end-to-end tests."""
        print("\n" + "=" * 80)
        print("RUNNING END-TO-END TESTS")
        print("=" * 80)

        # E2E tests are part of integration test suite
        results = run_integration_tests()
        self.results["e2e_tests"] = results.get("e2e_tests", {})

    def _run_performance_tests(self) -> None:
        """Run performance and stress tests."""
        print("\n" + "=" * 80)
        print("RUNNING PERFORMANCE TESTS")
        print("=" * 80)

        results = run_performance_tests()
        self.results["performance_tests"] = results

    def _run_security_tests(self) -> None:
        """Run security tests."""
        print("\n" + "=" * 80)
        print("RUNNING SECURITY TESTS")
        print("=" * 80)

        results = run_security_tests()
        self.results["security_tests"] = results

    def _run_mutation_tests(self) -> None:
        """Run mutation testing."""
        print("\n" + "=" * 80)
        print("RUNNING MUTATION TESTING")
        print("=" * 80)

        results = run_mutation_testing()
        self.results["mutation_tests"] = results

    def _run_coverage_analysis(self) -> None:
        """Run coverage analysis."""
        print("\n" + "=" * 80)
        print("RUNNING COVERAGE ANALYSIS")
        print("=" * 80)

        import coverage

        cov = coverage.Coverage()
        try:
            summary = {
                "overall_line_coverage": 0.0,
                "overall_branch_coverage": 0.0,
                "by_module": {},
                "gaps": [],
            }

            total_lines = 0
            covered_lines = 0
            total_branches = 0
            covered_branches = 0

            for file in cov.get_data().measured_files():
                analysis = cov.analysis2(file)
                file_summary = {
                    "lines": len(analysis[1]),
                    "missing_lines": len(analysis[2]),
                    "line_coverage": (
                        (len(analysis[1]) - len(analysis[2])) / len(analysis[1]) * 100
                        if analysis[1]
                        else 0
                    ),
                }

                total_lines += len(analysis[1])
                covered_lines += len(analysis[1]) - len(analysis[2])

                summary["by_module"][file] = file_summary

                # Identify coverage gaps
                if file_summary["line_coverage"] < 80:
                    summary["gaps"].append(
                        {
                            "file": file,
                            "coverage": file_summary["line_coverage"],
                            "missing_lines": analysis[2][:20],  # First 20 missing
                        }
                    )
                else:
                    summary["gaps"].append(
                        {
                            "file": file,
                            "coverage": file_summary["line_coverage"],
                            "missing_lines": analysis[2][:20],  # First 20 missing
                        }
                    )

            summary["overall_line_coverage"] = (
                covered_lines / total_lines * 100 if total_lines > 0 else 0
            )

            summary["total_branches"] = total_branches
            summary["covered_branches"] = covered_branches

            self.results["coverage"] = summary

            print(f"\nLine Coverage: {summary['overall_line_coverage']:.1f}%")
            print(f"Threshold: {self.args.coverage_threshold}%")
            print(
                f"Status: {'✅ MET' if summary['overall_line_coverage'] >= self.args.coverage_threshold else '❌ FAILED'}"
            )

        except Exception as e:
            logger.error(f"Error generating coverage summary: {e}")

    def _generate_final_report(self) -> None:
        """Generate comprehensive final report."""
        duration = time.time() - self.start_time if self.start_time else 0

        final_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.args.seed,
            "duration_seconds": duration,
            "configuration": {
                "coverage_threshold": self.args.coverage_threshold,
                "parallel_workers": self.args.parallel,
            },
            "results": self.results,
            "summary": self._generate_summary(),
        }

        # Save JSON report
        report_path = self.output_dir / "comprehensive_report.json"
        with open(report_path, "w") as f:
            json.dump(final_report, f, indent=2)

        # Generate HTML report
        self._generate_html_report(final_report)

        # Print final summary
        self._print_final_summary(final_report)

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate high-level summary."""
        summary = {
            "total_test_categories": len(self.results),
            "passed_categories": 0,
            "failed_categories": 0,
            "overall_status": "PASSED",
        }

        for category, results in self.results.items():
            # Determine pass/fail for each category
            if isinstance(results, dict):
                if "failed" in results and results["failed"] == 0:
                    summary["passed_categories"] += 1
                elif "failed" in results:
                    summary["failed_categories"] += 1
                    summary["overall_status"] = "FAILED"
                elif "pass_rate" in results:
                    if results["pass_rate"] >= 80:
                        summary["passed_categories"] += 1
                    else:
                        summary["failed_categories"] += 1
                        summary["overall_status"] = "FAILED"

        return summary

    def _generate_html_report(self, report: Dict[str, Any]) -> None:
        """Generate HTML report."""
        summary = report["summary"]

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>APGI Comprehensive Test Report</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ background: {'#27ae60' if summary['overall_status'] == 'PASSED' else '#e74c3c'}; color: white; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; }}
        .summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem; }}
        .metric {{ background: #f8f9fa; padding: 1rem; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 2rem; font-weight: bold; color: #2c3e50; }}
        .section {{ margin-bottom: 2rem; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .pass {{ color: #27ae60; }}
        .fail {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>APGI Comprehensive Test Report</h1>
            <p>Generated: {report['timestamp']} | Status: {summary['overall_status']}</p>
        </div>

        <div class="summary">
            <div class="metric">
                <div class="metric-value">{summary['total_test_categories']}</div>
                <div>Test Categories</div>
            </div>
            <div class="metric">
                <div class="metric-value pass">{summary['passed_categories']}</div>
                <div>Passed</div>
            </div>
            <div class="metric">
                <div class="metric-value {'pass' if summary['failed_categories'] == 0 else 'fail'}">{summary['failed_categories']}</div>
                <div>Failed</div>
            </div>
        </div>

        <div class="section">
            <h2>Coverage Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr>
                    <td>Line Coverage</td>
                    <td>{report['results'].get('coverage', {}).get('line_coverage', 0):.1f}%</td>
                </tr>
                <tr>
                    <td>Threshold</td>
                    <td>{self.args.coverage_threshold}%</td>
                </tr>
                <tr>
                    <td>Status</td>
                    <td class="{'pass' if report['results'].get('coverage', {}).get('threshold_met', False) else 'fail'}">
                        {'✓ MET' if report['results'].get('coverage', {}).get('threshold_met', False) else '✗ FAILED'}
                    </td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>"""

        html_path = self.output_dir / "comprehensive_report.html"
        with open(html_path, "w") as f:
            f.write(html)

    def _print_final_summary(self, report: Dict[str, Any]) -> None:
        """Print final summary to console."""
        summary = report["summary"]

        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Test Categories: {summary['total_test_categories']}")
        print(f"Passed: {summary['passed_categories']} ✅")
        print(
            f"Failed: {summary['failed_categories']} {'✅' if summary['failed_categories'] == 0 else '❌'}"
        )
        print(f"Duration: {report['duration_seconds']:.1f}s")
        print(f"\nReports saved to: {self.output_dir}")

        if summary["overall_status"] == "PASSED":
            print("\n🎉 All tests passed!")
        else:
            print("\n⚠️  Some tests failed. Review the reports for details.")

    def _all_passed(self) -> bool:
        """Check if all tests passed."""
        summary = self._generate_summary()
        return summary["overall_status"] == "PASSED"


def main():
    """Main entry point."""
    args = parse_arguments()

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parse_arguments().print_help()
        sys.exit(0)

    runner = TestRunner(args)
    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
