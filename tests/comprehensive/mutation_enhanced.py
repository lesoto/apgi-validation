"""
APGI Mutation Testing Enhancement Module
========================================

Enhanced mutation testing with:
- HTML report generation
- >=80% mutation score target
- Enhanced mutation operators
- Comprehensive coverage analysis

This module extends the base mutation_tester.py with advanced reporting.
"""

import html
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import base mutation testing
sys.path.insert(0, str(Path(__file__).parent))
from mutation_tester import MutationTester


@dataclass
class MutationScoreTarget:
    """Mutation score target configuration."""

    target_percentage: float = 80.0
    critical_modules: List[str] = field(default_factory=list)
    warning_threshold: float = 60.0
    fail_threshold: float = 40.0


@dataclass
class EnhancedMutationReport:
    """Enhanced mutation testing report."""

    total_mutants: int
    killed: int
    survived: int
    timeout: int
    error: int
    mutation_score: float
    target_met: bool
    survived_mutants: List[Dict[str, Any]]
    weak_assertions: List[str]
    coverage_by_module: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    html_report_path: Optional[str] = None


class HTMLReportGenerator:
    """Generate comprehensive HTML mutation testing reports."""

    def __init__(self, output_dir: Path = Path("reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self, report: Dict[str, Any], output_name: str = "mutation_enhanced"
    ) -> Path:
        """Generate comprehensive HTML report."""
        html_path = self.output_dir / f"{output_name}.html"

        mutation_score = report.get("mutation_score", 0.0)
        target_met = mutation_score >= 80.0

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APGI Mutation Testing Report</title>
    <style>
        :root {{
            --primary: #2c3e50;
            --success: #27ae60;
            --warning: #f39c12;
            --danger: #e74c3c;
            --info: #3498db;
            --light: #ecf0f1;
            --dark: #34495e;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: var(--primary);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}
        
        .score-card {{
            display: inline-block;
            background: white;
            color: var(--primary);
            padding: 1.5rem 2.5rem;
            border-radius: 12px;
            margin-top: 1rem;
            text-align: center;
        }}
        
        .score-value {{
            font-size: 3.5rem;
            font-weight: bold;
            color: {self._get_score_color(mutation_score)};
        }}
        
        .score-label {{
            font-size: 0.9rem;
            color: var(--dark);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .target-badge {{
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 1rem;
            background: {'#27ae60' if target_met else '#e74c3c'};
            color: white;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .card {{
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
            transition: transform 0.2s;
        }}
        
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }}
        
        .card h3 {{
            color: var(--primary);
            margin-bottom: 1rem;
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--light);
        }}
        
        .stat-row:last-child {{
            border-bottom: none;
        }}
        
        .stat-label {{
            color: var(--dark);
        }}
        
        .stat-value {{
            font-weight: bold;
            font-family: 'Courier New', monospace;
        }}
        
        .stat-value.killed {{ color: var(--success); }}
        .stat-value.survived {{ color: var(--danger); }}
        .stat-value.timeout {{ color: var(--warning); }}
        .stat-value.error {{ color: var(--danger); }}
        
        .progress-bar {{
            width: 100%;
            height: 8px;
            background: var(--light);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 0.5rem;
        }}
        
        .progress-fill {{
            height: 100%;
            background: {self._get_score_color(mutation_score)};
            transition: width 0.5s ease;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        
        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--light);
        }}
        
        th {{
            background: var(--primary);
            color: white;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }}
        
        tr:hover {{
            background: rgba(236, 240, 241, 0.5);
        }}
        
        .code-block {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 0.75rem;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        
        .original {{ border-left: 4px solid var(--success); }}
        .mutated {{ border-left: 4px solid var(--danger); }}
        
        .alert {{
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }}
        
        .alert-success {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }}
        
        .alert-warning {{
            background: #fff3cd;
            border: 1px solid #ffeeba;
            color: #856404;
        }}
        
        .alert-danger {{
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }}
        
        .recommendations {{
            list-style: none;
        }}
        
        .recommendations li {{
            padding: 0.75rem 0;
            padding-left: 1.5rem;
            position: relative;
        }}
        
        .recommendations li::before {{
            content: "→";
            position: absolute;
            left: 0;
            color: var(--info);
        }}
        
        .footer {{
            text-align: center;
            padding: 2rem;
            color: var(--dark);
            font-size: 0.9rem;
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 1rem;
            }}
            
            .grid {{
                grid-template-columns: 1fr;
            }}
            
            table {{
                font-size: 0.85rem;
            }}
            
            th, td {{
                padding: 0.75rem 0.5rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🔬 APGI Mutation Testing Report</h1>
            <p>Comprehensive mutation testing analysis with quality gates</p>
            <div class="score-card">
                <div class="score-value">{mutation_score:.1f}%</div>
                <div class="score-label">Mutation Score</div>
                <span class="target-badge">{'✓ TARGET MET (≥80%)' if target_met else '⚠ BELOW TARGET'}</span>
            </div>
        </header>
"""

        # Add summary cards
        html_content += self._generate_summary_cards(report)

        # Add survived mutants section
        html_content += self._generate_survived_mutants_section(report)

        # Add recommendations
        html_content += self._generate_recommendations(report)

        # Footer
        html_content += f"""
        <footer class="footer">
            <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} | APGI Validation Framework</p>
        </footer>
    </div>
</body>
</html>"""

        with open(html_path, "w") as f:
            f.write(html_content)

        return html_path

    def _get_score_color(self, score: float) -> str:
        """Get color for mutation score."""
        if score >= 80:
            return "var(--success)"
        elif score >= 60:
            return "var(--warning)"
        else:
            return "var(--danger)"

    def _generate_summary_cards(self, report: Dict[str, Any]) -> str:
        """Generate summary statistics cards."""
        total = report.get("total_mutants", 0)
        killed = report.get("killed", 0)
        survived = report.get("survived", 0)
        timeout = report.get("timeout", 0)
        error = report.get("error", 0)

        return f"""
        <div class="grid">
            <div class="card">
                <h3>📊 Mutation Statistics</h3>
                <div class="stat-row">
                    <span class="stat-label">Total Mutants</span>
                    <span class="stat-value">{total}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Killed</span>
                    <span class="stat-value killed">{killed}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Survived</span>
                    <span class="stat-value survived">{survived}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Timeout</span>
                    <span class="stat-value timeout">{timeout}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Error</span>
                    <span class="stat-value error">{error}</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🎯 Quality Gates</h3>
                <div class="stat-row">
                    <span class="stat-label">Target Score</span>
                    <span class="stat-value">80%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Actual Score</span>
                    <span class="stat-value {'killed' if report.get('mutation_score', 0) >= 80 else 'survived'}">{report.get('mutation_score', 0):.1f}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {min(report.get('mutation_score', 0), 100)}%"></div>
                </div>
                <div class="stat-row" style="margin-top: 1rem;">
                    <span class="stat-label">Status</span>
                    <span class="stat-value">{'✅ PASSED' if report.get('mutation_score', 0) >= 80 else '⚠️ NEEDS IMPROVEMENT'}</span>
                </div>
            </div>
            
            <div class="card">
                <h3>📈 Effectiveness</h3>
                <div class="stat-row">
                    <span class="stat-label">Kill Rate</span>
                    <span class="stat-value">{(killed / total * 100) if total > 0 else 0:.1f}%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Survival Rate</span>
                    <span class="stat-value survived">{(survived / total * 100) if total > 0 else 0:.1f}%</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Weak Assertions</span>
                    <span class="stat-value">{len(report.get('weak_assertions', []))}</span>
                </div>
            </div>
        </div>
"""

    def _generate_survived_mutants_section(self, report: Dict[str, Any]) -> str:
        """Generate survived mutants table."""
        survived = report.get("survived_mutants", [])

        if not survived:
            return """
        <div class="card">
            <h3>✅ Survived Mutants</h3>
            <div class="alert alert-success">
                No mutants survived! All mutations were detected by the test suite.
            </div>
        </div>
"""

        rows = ""
        for mutant in survived[:50]:  # Limit to first 50
            rows += f"""
                <tr>
                    <td>{html.escape(mutant.get('id', 'N/A'))}</td>
                    <td>{html.escape(mutant.get('type', 'N/A'))}</td>
                    <td>{mutant.get('line', 'N/A')}</td>
                    <td><div class="code-block original">{html.escape(str(mutant.get('original', 'N/A')))}</div></td>
                    <td><div class="code-block mutated">{html.escape(str(mutant.get('mutated', 'N/A')))}</div></td>
                </tr>
"""

        if len(survived) > 50:
            rows += f"""
                <tr>
                    <td colspan="5" style="text-align: center; color: var(--dark);">
                        ... and {len(survived) - 50} more survived mutants
                    </td>
                </tr>
"""

        return f"""
        <div class="card">
            <h3>⚠️ Survived Mutants ({len(survived)} total)</h3>
            <p>These mutants were not detected by the test suite and may indicate weak assertions:</p>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Type</th>
                        <th>Line</th>
                        <th>Original</th>
                        <th>Mutated</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
"""

    def _generate_recommendations(self, report: Dict[str, Any]) -> str:
        """Generate recommendations section."""
        recommendations = report.get("weak_assertions", [])
        mutation_score = report.get("mutation_score", 0)

        # Generate additional recommendations based on score
        additional_recs = []
        if mutation_score < 80:
            additional_recs.append(
                "Add more specific assertions to kill surviving mutants"
            )
            additional_recs.append("Consider adding boundary value tests")
        if mutation_score < 60:
            additional_recs.append("Review test coverage for critical paths")
            additional_recs.append(
                "Add integration tests that verify end-to-end behavior"
            )

        all_recs = recommendations + additional_recs

        if not all_recs:
            return """
        <div class="card">
            <h3>💡 Recommendations</h3>
            <div class="alert alert-success">
                No recommendations - mutation score meets target!
            </div>
        </div>
"""

        recs_html = "\n".join(f"<li>{html.escape(rec)}</li>" for rec in all_recs[:20])

        return f"""
        <div class="card">
            <h3>💡 Recommendations</h3>
            <ul class="recommendations">
                {recs_html}
            </ul>
        </div>
"""


class EnhancedMutationTester(MutationTester):
    """Enhanced mutation tester with HTML reporting and target enforcement."""

    def __init__(
        self,
        target_modules: List[str],
        test_modules: List[str],
        timeout_seconds: float = 5.0,
        parallel_workers: int = 4,
        score_target: MutationScoreTarget = None,
    ):
        super().__init__(
            target_modules, test_modules, timeout_seconds, parallel_workers
        )
        self.score_target = score_target or MutationScoreTarget()
        self.html_generator = HTMLReportGenerator()

    def run_enhanced(self) -> EnhancedMutationReport:
        """Run enhanced mutation testing with HTML report generation."""
        print("=" * 80)
        print("APGI ENHANCED MUTATION TESTING")
        print("=" * 80)
        print(f"Target Mutation Score: {self.score_target.target_percentage}%")
        print(f"Warning Threshold: {self.score_target.warning_threshold}%")
        print(f"Fail Threshold: {self.score_target.fail_threshold}%")
        print("=" * 80)

        # Run base mutation testing
        base_report = self.run()

        mutation_score = base_report.get("mutation_score", 0)
        target_met = mutation_score >= self.score_target.target_percentage

        # Generate recommendations
        recommendations = self._generate_recommendations(base_report)

        # Generate HTML report
        html_path = self.html_generator.generate_report(
            base_report, "mutation_enhanced"
        )

        # Build enhanced report
        enhanced_report = EnhancedMutationReport(
            total_mutants=base_report.get("total_mutants", 0),
            killed=base_report.get("killed", 0),
            survived=base_report.get("survived", 0),
            timeout=base_report.get("timeout", 0),
            error=base_report.get("error", 0),
            mutation_score=mutation_score,
            target_met=target_met,
            survived_mutants=base_report.get("survived_mutants", []),
            weak_assertions=base_report.get("weak_assertions", []),
            coverage_by_module=self._calculate_module_coverage(),
            recommendations=recommendations,
            html_report_path=str(html_path),
        )

        # Print enhanced summary
        self._print_enhanced_summary(enhanced_report)

        return enhanced_report

    def _generate_recommendations(self, base_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        mutation_score = base_report.get("mutation_score", 0)
        survived = base_report.get("survived_mutants", [])

        if mutation_score < self.score_target.target_percentage:
            recommendations.append(
                f"Mutation score {mutation_score:.1f}% is below target {self.score_target.target_percentage}%. "
                f"Add assertions to detect {len(survived)} surviving mutants."
            )

        # Check for specific mutation types that survived
        comparison_survivors = sum(
            1
            for s in survived
            if any(t in s.get("type", "") for t in ["GT_TO_GE", "LT_TO_LE", "EQ_TO_NE"])
        )
        if comparison_survivors > 0:
            recommendations.append(
                f"{comparison_survivors} comparison mutations survived. "
                "Consider adding more precise boundary assertions."
            )

        arithmetic_survivors = sum(
            1
            for s in survived
            if any(t in s.get("type", "") for t in ["ADD_TO_SUB", "MUL_TO_DIV"])
        )
        if arithmetic_survivors > 0:
            recommendations.append(
                f"{arithmetic_survivors} arithmetic mutations survived. "
                "Verify mathematical calculations with specific expected values."
            )

        return recommendations

    def _calculate_module_coverage(self) -> Dict[str, Dict[str, Any]]:
        """Calculate mutation coverage by module."""
        coverage = {}

        for result in self.results:
            module = result.mutant.module_path
            if module not in coverage:
                coverage[module] = {"total": 0, "killed": 0, "survived": 0}

            coverage[module]["total"] += 1
            if result.mutant.status == "killed":
                coverage[module]["killed"] += 1
            elif result.mutant.status == "survived":
                coverage[module]["survived"] += 1

        # Calculate percentages
        for module, stats in coverage.items():
            stats["mutation_score"] = int(
                stats["killed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            )

        return coverage

    def _print_enhanced_summary(self, report: EnhancedMutationReport) -> None:
        """Print enhanced summary."""
        print(f"\n{'=' * 80}")
        print("ENHANCED MUTATION TESTING SUMMARY")
        print(f"{'=' * 80}")
        print(f"Mutation Score: {report.mutation_score:.1f}%")
        print(f"Target Met: {'✅ YES' if report.target_met else '⚠️ NO'}")
        print(f"Total Mutants: {report.total_mutants}")
        print(f"Killed: {report.killed} ✓")
        print(f"Survived: {report.survived} {'✓' if report.survived == 0 else '⚠️'}")
        print(f"Timeout: {report.timeout}")
        print(f"Error: {report.error}")

        if report.html_report_path:
            print(f"\n📄 HTML Report: {report.html_report_path}")

        if report.recommendations:
            print("\n💡 Recommendations:")
            for rec in report.recommendations[:5]:
                print(f"  - {rec}")

        print(f"\n{'=' * 80}")

        if report.target_met:
            print("✅ Target mutation score of ≥80% achieved!")
        else:
            print(
                f"⚠️ Target mutation score of ≥80% not achieved. Current: {report.mutation_score:.1f}%"
            )


def run_enhanced_mutation_testing() -> EnhancedMutationReport:
    """Entry point for enhanced mutation testing."""
    tester = EnhancedMutationTester(
        target_modules=[
            "utils/eeg_processing.py",
            "utils/statistical_tests.py",
            "utils/protocol_schema.py",
            "utils/data_validation.py",
            "utils/data_processing_functions.py",
        ],
        test_modules=[
            "tests/test_eeg_processing.py",
            "tests/test_statistical_tests.py",
            "tests/test_data_processing_functions.py",
        ],
        timeout_seconds=10.0,
        parallel_workers=4,
        score_target=MutationScoreTarget(target_percentage=80.0),
    )

    return tester.run_enhanced()


if __name__ == "__main__":
    report = run_enhanced_mutation_testing()

    # Exit with appropriate code
    sys.exit(0 if report.target_met else 1)
