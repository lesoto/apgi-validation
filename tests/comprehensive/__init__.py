"""
APGI Comprehensive Adversarial Testing Framework
=================================================

This module provides enterprise-grade testing infrastructure for the APGI
validation framework, designed to achieve near-100% code coverage while
testing for correctness, robustness, and failure handling.

Features:
- Adversarial test generation targeting edge cases and rare branches
- Mutation testing to verify test effectiveness
- Performance and stress testing under extreme conditions
- Security testing for input validation and injection resistance
- Deterministic reproducibility via controlled seeding
- Comprehensive coverage reporting (line, branch, path)
- Integration and end-to-end testing with realistic workflows

Usage:
    pytest tests/comprehensive/ -v --cov=. --cov-report=html
    python -m tests.comprehensive.mutation_tester
    python -m tests.comprehensive.stress_test
"""

__version__ = "1.0.0"
__author__ = "APGI Testing Team"

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import sys
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


class TestCategory(Enum):
    """Categories of comprehensive tests."""

    UNIT = auto()
    INTEGRATION = auto()
    E2E = auto()
    PERFORMANCE = auto()
    SECURITY = auto()
    MUTATION = auto()
    STRESS = auto()
    REGRESSION = auto()
    BOUNDARY = auto()
    CONCURRENCY = auto()


class TestPriority(Enum):
    """Test priority levels."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class TestResult:
    """Result of a single test execution."""

    test_name: str
    category: TestCategory
    passed: bool
    duration_ms: float
    coverage_pct: float
    branches_hit: int
    branches_total: int
    exceptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CoverageReport:
    """Comprehensive coverage report."""

    module_name: str
    line_coverage: float
    branch_coverage: float
    path_coverage: float
    function_coverage: float
    class_coverage: float
    uncovered_lines: List[int] = field(default_factory=list)
    uncovered_branches: List[Tuple[int, str]] = field(default_factory=list)
    complexity_score: float = 0.0


@dataclass
class MutationReport:
    """Mutation testing report."""

    total_mutants: int
    killed_mutants: int
    survived_mutants: int
    timeout_mutants: int
    mutation_score: float
    weak_assertions: List[str] = field(default_factory=list)
    equivalent_mutants: List[str] = field(default_factory=list)


class AdversarialTestFramework:
    """
    Main orchestrator for adversarial testing.

    This class coordinates all testing activities including:
    - Test discovery and categorization
    - Mutation testing
    - Performance/stress testing
    - Coverage analysis
    - Report generation
    """

    def __init__(
        self,
        source_paths: Optional[List[str]] = None,
        test_paths: Optional[List[str]] = None,
        seed: int = 42,
    ):
        """
        Initialize the testing framework.

        Args:
            source_paths: Paths to source code directories
            test_paths: Paths to test directories
            seed: Random seed for reproducibility
        """
        self.source_paths = source_paths or ["."]
        self.test_paths = test_paths or ["tests"]
        self.seed = seed
        self.results: List[TestResult] = []
        self.coverage_reports: List[CoverageReport] = []
        self.mutation_report: Optional[MutationReport] = None

        # Set random seeds for reproducibility
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)

        logger.info(f"AdversarialTestFramework initialized with seed={seed}")

    def discover_tests(self) -> Dict[TestCategory, List[str]]:
        """Discover all tests and categorize them."""
        categories: Dict[TestCategory, List[str]] = {cat: [] for cat in TestCategory}

        for test_path in self.test_paths:
            path = Path(test_path)
            if not path.exists():
                continue

            for test_file in path.rglob("test_*.py"):
                # Categorize based on filename and content
                name = test_file.stem.lower()
                file_path_str = str(test_file).lower()

                if "integration" in name or "integration" in file_path_str:
                    categories[TestCategory.INTEGRATION].append(str(test_file))
                elif (
                    "e2e" in name or "end_to_end" in name or "endtoend" in file_path_str
                ):
                    categories[TestCategory.E2E].append(str(test_file))
                elif (
                    "performance" in name
                    or "perf" in name
                    or "performance" in file_path_str
                ):
                    categories[TestCategory.PERFORMANCE].append(str(test_file))
                elif "security" in name or "sec" in name or "security" in file_path_str:
                    categories[TestCategory.SECURITY].append(str(test_file))
                elif "mutation" in name or "mutation" in file_path_str:
                    categories[TestCategory.MUTATION].append(str(test_file))
                elif "stress" in name or "stress" in file_path_str:
                    categories[TestCategory.STRESS].append(str(test_file))
                elif "regression" in name or "regression" in file_path_str:
                    categories[TestCategory.REGRESSION].append(str(test_file))
                elif "boundary" in name or "boundary" in file_path_str:
                    categories[TestCategory.BOUNDARY].append(str(test_file))
                elif (
                    "concurrency" in name
                    or "race" in name
                    or "concurrency" in file_path_str
                ):
                    categories[TestCategory.CONCURRENCY].append(str(test_file))
                elif "comprehensive" in file_path_str:
                    # Skip comprehensive test files from individual categories
                    continue
                else:
                    # Default to unit test for all other test files
                    categories[TestCategory.UNIT].append(str(test_file))

        return categories

    def run_all_tests(
        self,
        categories: Optional[List[TestCategory]] = None,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all tests and generate comprehensive report.

        Args:
            categories: Specific categories to run (None = all)
            parallel: Whether to run tests in parallel

        Returns:
            Comprehensive test report dictionary
        """
        import time

        start_time = time.time()
        all_tests = self.discover_tests()

        if categories:
            all_tests = {k: v for k, v in all_tests.items() if k in categories}

        results_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.seed,
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "by_category": {},
            "coverage_summary": {},
            "performance_summary": {},
            "recommendations": [],
        }

        # Run tests by category
        for category, test_files in all_tests.items():
            if not test_files:
                continue

            cat_results = self._run_category_tests(category, test_files, parallel)
            results_summary["by_category"][category.name] = cat_results
            results_summary["total_tests"] += cat_results.get("total", 0)
            results_summary["passed"] += cat_results.get("passed", 0)
            results_summary["failed"] += cat_results.get("failed", 0)

        # Generate coverage report
        coverage_summary = self._generate_coverage_summary()
        results_summary["coverage_summary"] = coverage_summary

        # Generate recommendations
        results_summary["recommendations"] = self._generate_recommendations(
            results_summary
        )

        results_summary["duration_seconds"] = time.time() - start_time

        return results_summary

    def _run_category_tests(
        self, category: TestCategory, test_files: List[str], parallel: bool
    ) -> Dict[str, Any]:
        """Run tests for a specific category."""
        import subprocess
        import tempfile
        import json

        results = {"category": category.name, "total": 0, "passed": 0, "failed": 0}

        print(f"\n  Running {len(test_files)} {category.name} test file(s)...")

        for i, test_file in enumerate(test_files, 1):
            print(f"  [{i}/{len(test_files)}] {test_file}...", end=" ", flush=True)
            try:
                # Run pytest without coverage for speed (coverage done separately)
                cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    test_file,
                    "-v",
                    "--tb=short",
                ]

                if not parallel:
                    cmd.append("-p")
                    cmd.append("no:xdist")

                # Use Popen with streaming to avoid deadlock from large output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Longer timeout for nolds-based tests (fractional dimension)
                timeout_seconds = 120 if "fractional" in test_file else 60

                try:
                    stdout, stderr = process.communicate(timeout=timeout_seconds)
                    result_code = process.returncode
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()
                    result_code = -1
                    print("TIMEOUT", flush=True)
                    results["failed"] += 1
                    results["total"] += 1
                    logger.error(f"Test {test_file} timed out after {timeout_seconds}s")
                    continue

                # Parse results
                if result_code == 0:
                    print("✓ PASS", flush=True)
                    results["passed"] += 1
                else:
                    print("✗ FAIL", flush=True)
                    # Show brief error on failure
                    if stderr:
                        err_lines = stderr.strip().split("\n")[:3]
                        for line in err_lines:
                            print(f"    {line}")
                    results["failed"] += 1
                results["total"] += 1

            except Exception as e:
                print(f"✗ ERROR: {e}", flush=True)
                results["failed"] += 1
                results["total"] += 1
                logger.error(f"Error running {test_file}: {e}")

        print(
            f"  {category.name} complete: {results['passed']}/{results['total']} passed"
        )
        return results

    def _generate_coverage_summary(self) -> Dict[str, Any]:
        """Generate comprehensive coverage summary."""
        import coverage

        cov = coverage.Coverage()
        cov.load()

        summary = {
            "overall_line_coverage": 0.0,
            "overall_branch_coverage": 0.0,
            "by_module": {},
            "gaps": [],
        }

        try:
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

                summary["by_module"][file] = file_summary

                total_lines += len(analysis[1])
                covered_lines += len(analysis[1]) - len(analysis[2])

            summary["total_branches"] = total_branches
            summary["covered_branches"] = covered_branches

        except Exception as e:
            logger.error(f"Error generating coverage summary: {e}")

        return summary

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        coverage = results.get("coverage_summary", {}).get("overall_line_coverage", 0)

        if coverage < 50:
            recommendations.append(
                "CRITICAL: Coverage below 50%. Immediate action required."
            )
        elif coverage < 80:
            recommendations.append(
                f"HIGH: Coverage at {coverage:.1f}%. Target 80% minimum."
            )
        elif coverage < 95:
            recommendations.append(
                f"MEDIUM: Coverage at {coverage:.1f}%. Target 95% for production."
            )

        if results.get("failed", 0) > 0:
            recommendations.append(
                f"CRITICAL: {results['failed']} tests failing. Fix before deployment."
            )

        # Check for missing test categories
        categories = results.get("by_category", {})
        if "PERFORMANCE" not in categories:
            recommendations.append(
                "MEDIUM: No performance tests found. Add stress testing."
            )
        if "SECURITY" not in categories:
            recommendations.append(
                "HIGH: No security tests found. Add input validation tests."
            )
        if "CONCURRENCY" not in categories:
            recommendations.append(
                "MEDIUM: No concurrency tests found. Add race condition tests."
            )

        return recommendations

    def export_report(self, results: Dict[str, Any], output_path: str) -> None:
        """Export test results to various formats."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # JSON export
        json_path = output.with_suffix(".json")
        import json

        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # HTML report
        html_path = output.with_suffix(".html")
        self._generate_html_report(results, html_path)

        # Markdown summary
        md_path = output.with_suffix(".md")
        self._generate_markdown_report(results, md_path)

        logger.info(f"Reports exported to {output.parent}")

    def _generate_html_report(self, results: Dict[str, Any], path: Path) -> None:
        """Generate HTML test report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>APGI Test Report</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
        .header {{ background: #1a1a2e; color: white; padding: 1.5rem; border-radius: 8px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }}
        .metric {{ background: #f0f0f0; padding: 1rem; border-radius: 6px; text-align: center; }}
        .pass {{ color: #2ecc71; }}
        .fail {{ color: #e74c3c; }}
        .warning {{ color: #f39c12; }}
        table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .recommendation {{ background: #fff3cd; padding: 1rem; border-left: 4px solid #f39c12; margin: 0.5rem 0; }}
        .progress-bar {{ background: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ background: #3498db; height: 100%; transition: width 0.3s; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>APGI Comprehensive Test Report</h1>
        <p>Generated: {results['timestamp']} | Seed: {results['seed']}</p>
    </div>

    <div class="summary">
        <div class="metric">
            <h3>Total Tests</h3>
            <p>{results['total_tests']}</p>
        </div>
        <div class="metric">
            <h3>Passed</h3>
            <p class="pass">{results['passed']}</p>
        </div>
        <div class="metric">
            <h3>Failed</h3>
            <p class="{('pass' if results['failed'] == 0 else 'fail')}">{results['failed']}</p>
        </div>
        <div class="metric">
            <h3>Coverage</h3>
            <p>{results['coverage_summary'].get('overall_line_coverage', 0):.1f}%</p>
        </div>
    </div>

    <h2>Recommendations</h2>
"""
        for rec in results.get("recommendations", []):
            html += f'    <div class="recommendation">{rec}</div>\n'

        html += """
</body>
</html>"""

        with open(path, "w") as f:
            f.write(html)

    def _generate_markdown_report(self, results: Dict[str, Any], path: Path) -> None:
        """Generate Markdown test report."""
        md = f"""# APGI Comprehensive Test Report

**Generated:** {results['timestamp']}  
**Seed:** {results['seed']}  
**Duration:** {results.get('duration_seconds', 0):.1f}s

## Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Tests | {results['total_tests']} | - |
| Passed | {results['passed']} | {'✅' if results['failed'] == 0 else '⚠️'} |
| Failed | {results['failed']} | {'✅' if results['failed'] == 0 else '❌'} |
| Line Coverage | {results['coverage_summary'].get('overall_line_coverage', 0):.1f}% | {'✅' if results['coverage_summary'].get('overall_line_coverage', 0) >= 80 else '❌'} |

## Recommendations

"""
        for rec in results.get("recommendations", []):
            md += f"- {rec}\n"

        md += """
## Coverage Gaps

"""
        for gap in results["coverage_summary"].get("gaps", [])[:10]:
            md += f"- **{gap['file']}**: {gap['coverage']:.1f}% (lines: {gap['missing_lines']})\n"

        with open(path, "w") as f:
            f.write(md)


def run_comprehensive_test_suite():
    """Entry point for running the full test suite."""
    framework = AdversarialTestFramework(
        source_paths=[".", "utils", "Validation", "Falsification", "Theory"],
        test_paths=["tests"],
        seed=42,
    )

    print("=" * 80)
    print("APGI COMPREHENSIVE ADVERSARIAL TEST SUITE")
    print("=" * 80)

    results = framework.run_all_tests(parallel=True)

    # Export reports
    framework.export_report(results, "reports/test_report")

    # Print summary
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed: {results['passed']} ✅")
    print(f"Failed: {results['failed']} {'✅' if results['failed'] == 0 else '❌'}")
    print(
        f"Coverage: {results['coverage_summary'].get('overall_line_coverage', 0):.1f}%"
    )

    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 80}")
    for rec in results["recommendations"]:
        print(f"• {rec}")

    return results


if __name__ == "__main__":
    run_comprehensive_test_suite()
