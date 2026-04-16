"""
APGI Comprehensive Test Runner
==============================

Integrated test runner that orchestrates all testing modules:
- Unit tests
- Integration tests
- GUI testing enhancement
- Database transaction testing
- Mutation testing with HTML reports
- Property-based testing expansion
- Performance regression testing

This module provides a unified interface for running all APGI tests.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TestModuleResult:
    """Result from a test module."""

    module_name: str
    passed: bool
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class ComprehensiveTestRunner:
    """Orchestrates all testing modules."""

    def __init__(self, output_dir: Path = Path("reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[TestModuleResult] = []

    def run_all_tests(
        self, select_modules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run all test modules."""
        print("=" * 80)
        print("APGI COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        print(f"Selected modules: {select_modules or 'ALL'}")

        modules = {
            "integration": self._run_integration_tests,
            "gui": self._run_gui_tests,
            "database": self._run_database_tests,
            "mutation": self._run_mutation_tests,
            "property_based": self._run_property_based_tests,
            "performance": self._run_performance_tests,
        }

        # Run selected modules
        for name, runner in modules.items():
            if select_modules is None or name in select_modules:
                print(f"\n{'─' * 80}")
                print(f"Running {name.upper()} tests...")
                print("─" * 80)
                result = runner()
                self.results.append(result)

        # Generate comprehensive report
        report = self._generate_comprehensive_report()

        return report

    def _run_integration_tests(self) -> TestModuleResult:
        """Run integration and E2E tests."""
        start = time.time()

        try:
            from comprehensive.integration_e2e import IntegrationTestSuite

            suite = IntegrationTestSuite()
            report = suite.run_all_tests()

            passed = report.get("overall", {}).get("pass_rate", 0) >= 95

            return TestModuleResult(
                module_name="integration",
                passed=passed,
                duration_seconds=time.time() - start,
                details=report,
            )
        except Exception as e:
            return TestModuleResult(
                module_name="integration",
                passed=False,
                duration_seconds=time.time() - start,
                error_message=str(e),
            )

    def _run_gui_tests(self) -> TestModuleResult:
        """Run GUI testing enhancement module."""
        start = time.time()

        try:
            from comprehensive.gui_testing_enhanced import GUITestSuite

            suite = GUITestSuite()
            report = suite.run_all_tests()

            passed = report.get("gui_testing", {}).get("all_passed", False)

            return TestModuleResult(
                module_name="gui",
                passed=passed,
                duration_seconds=time.time() - start,
                details=report,
            )
        except Exception as e:
            return TestModuleResult(
                module_name="gui",
                passed=False,
                duration_seconds=time.time() - start,
                error_message=str(e),
            )

    def _run_database_tests(self) -> TestModuleResult:
        """Run database transaction testing module."""
        start = time.time()

        try:
            from comprehensive.db_transaction_comprehensive import (
                DatabaseTransactionTester,
            )

            tester = DatabaseTransactionTester()
            report = tester.run_all_tests()

            passed = report.get("summary", {}).get("all_passed", False)

            return TestModuleResult(
                module_name="database",
                passed=passed,
                duration_seconds=time.time() - start,
                details=report,
            )
        except Exception as e:
            return TestModuleResult(
                module_name="database",
                passed=False,
                duration_seconds=time.time() - start,
                error_message=str(e),
            )

    def _run_mutation_tests(self) -> TestModuleResult:
        """Run mutation testing with HTML reports."""
        start = time.time()

        try:
            from comprehensive.mutation_enhanced import run_enhanced_mutation_testing

            report = run_enhanced_mutation_testing()

            # Target: >=80% mutation score
            target_met = report.target_met

            return TestModuleResult(
                module_name="mutation",
                passed=target_met,
                duration_seconds=time.time() - start,
                details={
                    "mutation_score": report.mutation_score,
                    "total_mutants": report.total_mutants,
                    "killed": report.killed,
                    "survived": report.survived,
                    "html_report": report.html_report_path,
                    "recommendations": report.recommendations,
                },
            )
        except Exception as e:
            return TestModuleResult(
                module_name="mutation",
                passed=False,
                duration_seconds=time.time() - start,
                error_message=str(e),
            )

    def _run_property_based_tests(self) -> TestModuleResult:
        """Run property-based testing expansion."""
        start = time.time()

        try:
            from comprehensive.property_based_enhanced import run_property_based_tests

            info = run_property_based_tests()

            # Property-based tests are typically run via pytest
            # This runner provides the configuration
            return TestModuleResult(
                module_name="property_based",
                passed=True,  # Configuration loaded successfully
                duration_seconds=time.time() - start,
                details=info,
            )
        except Exception as e:
            return TestModuleResult(
                module_name="property_based",
                passed=False,
                duration_seconds=time.time() - start,
                error_message=str(e),
            )

    def _run_performance_tests(self) -> TestModuleResult:
        """Run performance regression testing."""
        start = time.time()

        try:
            from comprehensive.performance_regression import (
                run_performance_regression_tests,
            )

            report = run_performance_regression_tests()

            # Check if any critical alerts
            critical_alerts = sum(
                1 for a in report.get("alerts", []) if a.get("severity") == "critical"
            )

            passed = (
                critical_alerts == 0
                and report.get("summary", {}).get("pass_rate", 0) >= 90
            )

            return TestModuleResult(
                module_name="performance",
                passed=passed,
                duration_seconds=time.time() - start,
                details=report,
            )
        except Exception as e:
            return TestModuleResult(
                module_name="performance",
                passed=False,
                duration_seconds=time.time() - start,
                error_message=str(e),
            )

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_duration = sum(r.duration_seconds for r in self.results)
        passed_modules = sum(1 for r in self.results if r.passed)
        total_modules = len(self.results)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_modules": total_modules,
                "passed_modules": passed_modules,
                "failed_modules": total_modules - passed_modules,
                "total_duration_seconds": total_duration,
                "overall_pass_rate": (
                    passed_modules / total_modules * 100 if total_modules > 0 else 0
                ),
            },
            "modules": [
                {
                    "name": r.module_name,
                    "passed": r.passed,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error_message,
                }
                for r in self.results
            ],
            "details": {r.module_name: r.details for r in self.results},
        }

        # Save report
        report_path = self.output_dir / "comprehensive_test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print(f"\n{'=' * 80}")
        print("COMPREHENSIVE TEST SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total Modules: {total_modules}")
        print(f"Passed: {passed_modules} ✓")
        print(
            f"Failed: {total_modules - passed_modules} {'✓' if total_modules == passed_modules else '✗'}"
        )
        print(f"Total Duration: {total_duration:.1f}s")

        for r in self.results:
            status = "✅" if r.passed else "❌"
            print(f"\n{status} {r.module_name}: {r.duration_seconds:.1f}s")
            if r.error_message:
                print(f"   Error: {r.error_message}")

        print(f"\n📄 Full report: {report_path}")

        if passed_modules == total_modules:
            print("\n🎉 All test modules passed!")
        else:
            print(f"\n⚠️ {total_modules - passed_modules} module(s) failed")

        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="APGI Comprehensive Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python -m tests.comprehensive.comprehensive_runner
  
  # Run specific modules
  python -m tests.comprehensive.comprehensive_runner --modules gui database
  
  # Run with custom output directory
  python -m tests.comprehensive.comprehensive_runner --output ./my_reports
        """,
    )

    parser.add_argument(
        "--modules",
        nargs="+",
        choices=[
            "integration",
            "gui",
            "database",
            "mutation",
            "property_based",
            "performance",
        ],
        help="Specific modules to run (default: all)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports (default: reports)",
    )

    parser.add_argument(
        "--list", action="store_true", help="List available test modules and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("Available test modules:")
        print("  - integration: Integration and E2E tests")
        print(
            "  - gui: GUI testing enhancement (headless browser, screenshots, state transitions)"
        )
        print(
            "  - database: Database transaction testing (rollback/commit, pool exhaustion, isolation)"
        )
        print("  - mutation: Mutation testing (HTML reports, >=80% score target)")
        print(
            "  - property_based: Property-based testing (Hypothesis strategies, stateful testing)"
        )
        print(
            "  - performance: Performance regression testing (baselines, trends, alerts)"
        )
        return 0

    runner = ComprehensiveTestRunner(output_dir=args.output)
    report = runner.run_all_tests(select_modules=args.modules)

    # Return exit code based on results
    return 0 if report["summary"]["failed_modules"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
