"""
meta_falsification.py
====================

Framework-level falsification layer as specified in the Epistemic Architecture Paper.

The entire APGI framework is falsified if:
1. "all 14+ predictions fail" 
2. "an alternative framework makes all the same predictions"

This module implements the meta-level check that aggregates results across all
validation protocols.

Usage::

    from utils.meta_falsification import (
        FrameworkFalsificationGate,
        MetaFalsificationEngine,
    )
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ProtocolResult:
    """Result from a single validation protocol"""

    protocol_name: str
    total_criteria: int
    passed_criteria: int
    failed_criteria: int
    passed_criteria_ids: List[str]
    failed_criteria_ids: List[str]


class FrameworkFalsificationGate:
    """
    Framework-level falsification gate.

    Implements the meta-falsification criterion from the Epistemic Architecture Paper:
    The entire APGI framework is falsified if all 14+ predictions fail.
    """

    def __init__(
        self,
        min_criteria_for_falsification: int = 14,
        fail_threshold: float = 0.8,  # 80% failure rate triggers framework falsification
    ):
        """
        Initialize framework falsification gate.

        Args:
            min_criteria_for_falsification: Minimum number of criteria to consider
            fail_threshold: Proportion of failed criteria that triggers falsification
        """
        self.min_criteria = min_criteria_for_falsification
        self.fail_threshold = fail_threshold
        self.protocol_results: List[ProtocolResult] = []

    def add_protocol_result(
        self,
        protocol_name: str,
        total_criteria: int,
        passed_criteria: int,
        failed_criteria: int,
        passed_criteria_ids: List[str],
        failed_criteria_ids: List[str],
    ):
        """
        Add results from a validation protocol.

        Args:
            protocol_name: Name of the validation protocol
            total_criteria: Total number of criteria tested
            passed_criteria: Number of passed criteria
            failed_criteria: Number of failed criteria
            passed_criteria_ids: IDs of passed criteria
            failed_criteria_ids: IDs of failed criteria
        """
        result = ProtocolResult(
            protocol_name=protocol_name,
            total_criteria=total_criteria,
            passed_criteria=passed_criteria,
            failed_criteria=failed_criteria,
            passed_criteria_ids=passed_criteria_ids,
            failed_criteria_ids=failed_criteria_ids,
        )
        self.protocol_results.append(result)

    def compute_framework_falsification(self) -> Dict[str, Any]:
        """
        Compute whether the entire APGI framework is falsified.

        Returns:
            Dictionary with falsification decision and metrics
        """
        if not self.protocol_results:
            return {
                "framework_falsified": False,
                "reason": "No protocol results available",
                "total_criteria": 0,
                "total_passed": 0,
                "total_failed": 0,
                "fail_rate": 0.0,
            }

        # Aggregate across all protocols
        total_criteria = sum(r.total_criteria for r in self.protocol_results)
        total_passed = sum(r.passed_criteria for r in self.protocol_results)
        total_failed = sum(r.failed_criteria for r in self.protocol_results)

        fail_rate = total_failed / total_criteria if total_criteria > 0 else 0.0

        # Check framework falsification criteria
        # 1. Minimum criteria tested
        sufficient_criteria = total_criteria >= self.min_criteria

        # 2. Fail rate exceeds threshold
        high_failure_rate = fail_rate >= self.fail_threshold

        # Framework is falsified if both conditions met
        framework_falsified = sufficient_criteria and high_failure_rate

        # Additional analysis: which protocols failed most
        protocol_fail_rates = {
            r.protocol_name: r.failed_criteria / r.total_criteria
            for r in self.protocol_results
            if r.total_criteria > 0
        }

        # Identify critical failure patterns
        all_passed_criteria_ids = []
        all_failed_criteria_ids = []

        for result in self.protocol_results:
            all_passed_criteria_ids.extend(result.passed_criteria_ids)
            all_failed_criteria_ids.extend(result.failed_criteria_ids)

        # Check for systematic failures (same criterion failing across protocols)
        from collections import Counter

        failed_counter = Counter(all_failed_criteria_ids)
        systematic_failures = {
            criterion_id: count
            for criterion_id, count in failed_counter.items()
            if count >= 2  # Failed in at least 2 protocols
        }

        return {
            "framework_falsified": framework_falsified,
            "reason": (
                f"Fail rate {fail_rate:.2%} exceeds threshold {self.fail_threshold:.0%}"
                if framework_falsified
                else "Framework not falsified"
            ),
            "total_criteria": total_criteria,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "fail_rate": fail_rate,
            "pass_rate": 1.0 - fail_rate,
            "sufficient_criteria": sufficient_criteria,
            "protocol_fail_rates": protocol_fail_rates,
            "systematic_failures": systematic_failures,
            "all_passed_criteria": all_passed_criteria_ids,
            "all_failed_criteria": all_failed_criteria_ids,
        }

    def compare_with_alternative_framework(
        self, alternative_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare APGI predictions with alternative framework.

        If an alternative framework makes all the same predictions, APGI is not
        uniquely falsifiable.

        Args:
            alternative_results: Results from alternative framework

        Returns:
            Dictionary with comparison results
        """
        framework_result = self.compute_framework_falsification()

        # Get APGI's prediction set
        apgi_predictions = set(framework_result["all_passed_criteria"])

        # Get alternative's prediction set
        alt_predictions = set(alternative_results.get("passed_criteria", []))

        # Compute overlap
        intersection = apgi_predictions.intersection(alt_predictions)
        overlap_ratio = (
            len(intersection) / len(apgi_predictions) if apgi_predictions else 0.0
        )

        # Check if alternative makes all same predictions
        identical_predictions = overlap_ratio >= 0.95

        return {
            "identical_predictions": identical_predictions,
            "overlap_ratio": overlap_ratio,
            "apgi_unique_predictions": apgi_predictions - alt_predictions,
            "alternative_unique_predictions": alt_predictions - apgi_predictions,
            "shared_predictions": intersection,
            "conclusion": (
                "Alternative framework makes identical predictions - APGI not uniquely falsifiable"
                if identical_predictions
                else "APGI predictions are distinct from alternative"
            ),
        }


class MetaFalsificationEngine:
    """
    Engine for meta-falsification across all validation protocols.

    Coordinates framework-level falsification checks and provides
    comprehensive reporting.
    """

    def __init__(self):
        """Initialize meta-falsification engine."""
        self.gate = FrameworkFalsificationGate()
        self.validation_reports: Dict[str, Any] = {}

    def add_validation_report(self, protocol_name: str, report: Dict[str, Any]):
        """
        Add validation report from a protocol.

        Args:
            protocol_name: Name of the validation protocol
            report: Validation report with passed/failed criteria
        """
        self.validation_reports[protocol_name] = report

        # Extract criteria counts
        passed_criteria = report.get("passed_criteria", [])
        failed_criteria = report.get("falsified_criteria", [])

        passed_ids = [c["code"] for c in passed_criteria]
        failed_ids = [c["code"] for c in failed_criteria]

        self.gate.add_protocol_result(
            protocol_name=protocol_name,
            total_criteria=len(passed_ids) + len(failed_ids),
            passed_criteria=len(passed_ids),
            failed_criteria=len(failed_ids),
            passed_criteria_ids=passed_ids,
            failed_criteria_ids=failed_ids,
        )

    def run_meta_falsification(
        self, alternative_framework_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run complete meta-falsification analysis.

        Args:
            alternative_framework_results: Optional results from alternative framework

        Returns:
            Comprehensive meta-falsification report
        """
        # Compute framework falsification
        framework_result = self.gate.compute_framework_falsification()

        # Compare with alternative if provided
        comparison_result = None
        if alternative_framework_results:
            comparison_result = self.gate.compare_with_alternative_framework(
                alternative_framework_results
            )

        # Generate comprehensive report
        meta_report = {
            "framework_falsified": framework_result["framework_falsified"],
            "framework_reason": framework_result["reason"],
            "summary": {
                "total_protocols_tested": len(self.validation_reports),
                "total_criteria": framework_result["total_criteria"],
                "total_passed": framework_result["total_passed"],
                "total_failed": framework_result["total_failed"],
                "pass_rate": framework_result["pass_rate"],
                "fail_rate": framework_result["fail_rate"],
            },
            "protocol_breakdown": framework_result["protocol_fail_rates"],
            "systematic_failures": framework_result["systematic_failures"],
            "all_validation_reports": self.validation_reports,
            "alternative_comparison": comparison_result,
        }

        # Final conclusion
        if comparison_result and comparison_result["identical_predictions"]:
            meta_report["final_conclusion"] = (
                "APGI framework NOT uniquely falsifiable - "
                "alternative framework makes identical predictions"
            )
        elif framework_result["framework_falsified"]:
            meta_report["final_conclusion"] = (
                "APGI framework FALSIFIED - "
                f"{framework_result['fail_rate']:.1%} of predictions fail"
            )
        else:
            meta_report["final_conclusion"] = (
                "APGI framework NOT falsified - "
                f"{framework_result['pass_rate']:.1%} of predictions pass"
            )

        return meta_report

    def generate_framework_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate human-readable framework falsification report.

        Args:
            save_path: Optional path to save report

        Returns:
            Markdown-formatted report
        """
        result = self.run_meta_falsification()

        report_lines = [
            "# APGI Framework Falsification Report",
            "",
            f"## Framework Status: **{'FALSIFIED' if result['framework_falsified'] else 'NOT FALSIFIED'}**",
            "",
            f"**Reason:** {result['framework_reason']}",
            "",
            f"**Final Conclusion:** {result['final_conclusion']}",
            "",
            "## Summary",
            "",
            f"- Protocols tested: {result['summary']['total_protocols_tested']}",
            f"- Total criteria: {result['summary']['total_criteria']}",
            f"- Passed: {result['summary']['total_passed']} ({result['summary']['pass_rate']:.1%})",
            f"- Failed: {result['summary']['total_failed']} ({result['summary']['fail_rate']:.1%})",
            "",
            "## Protocol Breakdown",
            "",
        ]

        for protocol, fail_rate in result["protocol_breakdown"].items():
            status = "❌ FAILED" if fail_rate > 0.5 else "✅ PASSED"
            report_lines.append(
                f"- **{protocol}**: {status} ({fail_rate:.1%} failure rate)"
            )

        if result["systematic_failures"]:
            report_lines.append("")
            report_lines.append("## Systematic Failures (failed in ≥2 protocols)")
            report_lines.append("")
            for criterion, count in result["systematic_failures"].items():
                report_lines.append(f"- **{criterion}**: Failed in {count} protocols")

        if result["alternative_comparison"]:
            report_lines.append("")
            report_lines.append("## Alternative Framework Comparison")
            report_lines.append("")
            comp = result["alternative_comparison"]
            report_lines.append(
                f"- Identical predictions: {'Yes' if comp['identical_predictions'] else 'No'}"
            )
            report_lines.append(f"- Overlap ratio: {comp['overlap_ratio']:.1%}")
            report_lines.append(f"- **{comp['conclusion']}**")

        report_text = "\n".join(report_lines)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_text)

        return report_text
