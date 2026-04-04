"""
Validation-Falsification Bidirectional Consistency Checks
===============================================================

This module implements cross-checks between validation and falsification results
to ensure mutual coherence and identify contradictions before framework evaluation.

Classes:
    ValidationFalsificationConsistency: Main class for consistency checking
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyIssue:
    """Individual consistency issue between validation and falsification results."""

    issue_type: str  # "CONTRADICTION", "MISSING_VALIDATION", "ASSUMPTION_VIOLATION"
    description: str  # Human-readable description
    fp_protocol: Optional[str]  # Related FP protocol
    vp_protocol: Optional[str]  # Related VP protocol
    prediction_id: Optional[str]  # Specific prediction involved
    severity: str  # "HIGH", "MEDIUM", "LOW"
    evidence: List[str]  # Supporting evidence
    recommendation: str  # Suggested resolution


class ValidationFalsificationConsistency:
    """Cross-check validation and falsification results for mutual coherence."""

    def __init__(self, fp_results: Dict[str, Any], vp_results: Dict[str, Any]):
        """
        Initialize consistency checker with validation and falsification results.

        Args:
            fp_results: Results from falsification protocols (FP-01 to FP-12)
            vp_results: Results from validation protocols (VP-01 to VP-15)
        """
        self.fp_results = fp_results
        self.vp_results = vp_results
        self.issues: List[ConsistencyIssue] = []

    def check_assumption_consistency(self) -> List[ConsistencyIssue]:
        """
        Validation should confirm FP assumptions. If VP contradicts FP,
        then either FP or VP has an error.

        Returns:
            List of consistency issues found
        """
        issues = []

        # Example: FP-01 assumes interoceptive precision modulates thresholds
        # VP-08 should confirm this psychophysically
        fp01_prediction = self.fp_results.get("FP_01_ActiveInference", {}).get(
            "P1.1", {}
        )
        vp08_validation = self.vp_results.get(
            "VP_08_Psychophysical_ThresholdEstimation", {}
        ).get("P1.1", {})

        if fp01_prediction.get("passed") and not vp08_validation.get("passed"):
            issues.append(
                ConsistencyIssue(
                    issue_type="CONTRADICTION",
                    description="FP-01 passed but VP-08 validation failed - possible error in VP-08 empirical data or FP-01 simulation",
                    fp_protocol="FP-01_ActiveInference",
                    vp_protocol="VP_08_Psychophysical_ThresholdEstimation",
                    prediction_id="P1.1",
                    severity="HIGH",
                    evidence=[
                        f"FP-01 P1.1: {fp01_prediction.get('passed')}",
                        f"VP-08 P1.1: {vp08_validation.get('passed')}",
                    ],
                    recommendation="Resolve contradiction before framework falsification evaluation",
                )
            )

        # Example: FP-02 assumes agent convergence
        # VP-03 should validate that agents actually converge
        fp02_convergence = self.fp_results.get(
            "FP_02_AgentComparison_ConvergenceBenchmark", {}
        ).get("P3.conv", {})
        vp03_validation = self.vp_results.get(
            "VP_03_ActiveInference_AgentSimulations", {}
        ).get("P3.conv", {})

        if fp02_convergence.get("passed") and not vp03_validation.get("passed"):
            issues.append(
                ConsistencyIssue(
                    issue_type="CONTRADICTION",
                    description="FP-02 shows agent convergence but VP-03 validation failed - possible simulation error or VP-03 measurement issue",
                    fp_protocol="FP_02_AgentComparison_ConvergenceBenchmark",
                    vp_protocol="VP_03_ActiveInference_AgentSimulations",
                    prediction_id="P3.conv",
                    severity="HIGH",
                    evidence=[
                        f"FP-02 P3.conv: {fp02_convergence.get('passed')}",
                        f"VP-03 P3.conv: {vp03_validation.get('passed')}",
                    ],
                    recommendation="Verify agent convergence simulation and VP-03 measurement accuracy",
                )
            )

        return issues

    def check_missing_validation_protocols(self) -> List[ConsistencyIssue]:
        """
        Check if critical validation protocols are missing that falsification assumes.

        Returns:
            List of missing validation issues
        """
        issues = []

        # Check for VP-05 (Evolutionary Emergence) which FP-01, FP-02, FP-03 depend on
        vp05_present = "VP_05_EvolutionaryEmergence" in self.vp_results

        if vp05_present:
            # Check if dependent FP protocols have results
            fp_dependents = [
                "FP_01_ActiveInference",
                "FP_02_AgentComparison_ConvergenceBenchmark",
                "FP_03_FrameworkLevel_MultiProtocol",
            ]
            missing_deps = []

            for fp_dep in fp_dependents:
                if fp_dep not in self.fp_results:
                    missing_deps.append(fp_dep)

            if missing_deps:
                issues.append(
                    ConsistencyIssue(
                        issue_type="MISSING_VALIDATION",
                        description=f"VP-05 present but dependent FP protocols missing: {', '.join(missing_deps)}",
                        fp_protocol=None,
                        vp_protocol="VP_05_EvolutionaryEmergence",
                        prediction_id=None,
                        severity="HIGH",
                        evidence=[f"Missing: {dep}" for dep in missing_deps],
                        recommendation="Run dependent FP protocols before VP-05",
                    )
                )

        return issues

    def check_framework_assumptions(self) -> List[ConsistencyIssue]:
        """
        Check if falsification framework assumptions are violated by validation results.

        Returns:
            List of assumption violation issues
        """
        issues = []

        # Check Framework Falsification Condition A prerequisites
        # All 14 core predictions must be present for evaluation
        core_predictions = [
            "P1.1",
            "P1.2",
            "P1.3",
            "P2.a",
            "P2.b",
            "P2.c",
            "P3.conv",
            "P3.bic",
            "P4.a",
            "P4.b",
            "P4.c",
            "P4.d",
            "P5.a",
            "P5.b",
        ]

        missing_core = []
        for pred_id in core_predictions:
            found = False
            # Check FP results
            for fp_name, fp_data in self.fp_results.items():
                if pred_id in fp_data.get("named_predictions", {}):
                    found = True
                    break

            # Check VP results (supplementary)
            for vp_name, vp_data in self.vp_results.items():
                if pred_id in vp_data.get("named_predictions", {}):
                    found = True
                    break

            if not found:
                missing_core.append(pred_id)

        if missing_core:
            issues.append(
                ConsistencyIssue(
                    issue_type="ASSUMPTION_VIOLATION",
                    description=f"Framework falsification assumes core predictions {', '.join(missing_core)} but none found",
                    fp_protocol=None,
                    vp_protocol=None,
                    prediction_id=None,
                    severity="HIGH",
                    evidence=[f"Missing predictions: {', '.join(missing_core)}"],
                    recommendation="Ensure all core predictions are generated before falsification evaluation",
                )
            )

        return issues

    def check_tms_causal_consistency(
        self,
        fp01_results: Dict[str, Any],
        vp07_results: Dict[str, Any],
        vp10_results: Dict[str, Any],
    ) -> Optional[ConsistencyIssue]:
        """
        **GAP 8 FIX**: Validate TMS effects consistency between FP-01 and VP-07/VP-10.

        VP-10 (TMS/Pharmacological Causal Manipulations) should validate predictions
        that FP-01 tests. This check ensures TMS effects align with active inference
        dynamics predictions.

        Consistency check:
        - If FP-01 predicts dlPFC drives threshold (τ_θ parameter)
        - Then VP-07/VP-10 TMS to dlPFC should shift threshold in predicted direction

        Args:
            fp01_results: Results from FP-01 Active Inference validation
            vp07_results: Results from VP-07 TMS Causal Interventions
            vp10_results: Results from VP-10 Causal Manipulations Priority 2

        Returns:
            ConsistencyIssue if inconsistency detected, or None if consistent
        """
        # Extract relevant predictions from FP-01
        # F1.4 tests threshold adaptation dynamics (τ_θ parameter)
        fp01_threshold_dynamics = fp01_results.get("F1.4", {})
        fp01_passed = fp01_threshold_dynamics.get("passed", False)

        # Extract VP-07 TMS results
        # V7.1 tests threshold reduction from dlPFC TMS
        vp07_threshold_effect = vp07_results.get("V7.1", {})
        vp07_passed = vp07_threshold_effect.get("passed", False)

        # Extract VP-10 TMS results
        # P2.a tests dlPFC TMS threshold shift
        vp10_dlpfc_tms = vp10_results.get("P2.a", {})
        vp10_passed = vp10_dlpfc_tms.get("passed", False)

        # Check consistency
        # Both FP-01 and TMS protocols should agree on threshold dynamics
        consistency_passed = fp01_passed and (vp07_passed or vp10_passed)

        if not consistency_passed:
            return ConsistencyIssue(
                issue_type="CONTRADICTION",
                description="TMS effects (VP-07/VP-10) inconsistent with FP-01 active inference predictions",
                fp_protocol="FP-01_ActiveInference",
                vp_protocol="VP-07_TMS_CausalInterventions / VP-10_CausalManipulations",
                prediction_id="F1.4 / P2.a",
                severity="HIGH",
                evidence=[
                    f"FP-01 F1.4 (threshold dynamics): {fp01_passed}",
                    f"VP-07 V7.1 (TMS threshold effect): {vp07_passed}",
                    f"VP-10 P2.a (dlPFC TMS): {vp10_passed}",
                ],
                recommendation="Align TMS intervention parameters with FP-01 threshold predictions or verify empirical measurement",
            )

        return None

    def validate_tms_causal_consistency(
        self,
        fp01_results: Optional[Dict[str, Any]] = None,
        vp07_results: Optional[Dict[str, Any]] = None,
        vp10_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        **GAP 8 FIX**: Cross-protocol validation check for TMS consistency.

        Ensure VP-10 TMS effects are consistent with FP-01 active inference model.

        This function can be called with explicit results OR will use results
        passed during initialization.

        Args:
            fp01_results: FP-01 results (uses self.fp_results if None)
            vp07_results: VP-07 results (uses self.vp_results if None)
            vp10_results: VP-10 results (uses self.vp_results if None)

        Returns:
            Dictionary with consistency check results:
            - consistency_check: Name of the check
            - passed: Boolean indicating if consistency passed
            - evidence: List of evidence strings
            - issue: ConsistencyIssue if failed
        """
        # Use provided results or fall back to initialized results
        fp01 = fp01_results or self.fp_results.get("FP_01_ActiveInference", {})
        vp07 = vp07_results or self.vp_results.get("VP_07_TMS_CausalInterventions", {})
        vp10 = vp10_results or self.vp_results.get(
            "VP_10_CausalManipulations_Priority2", {}
        )

        # Extract relevant predictions
        fp01_threshold_dynamics = fp01.get("F1.4", {})
        vp07_threshold_effect = vp07.get("V7.1", {})
        vp10_dlpfc_tms = vp10.get("P2.a", {})

        # Check consistency
        fp01_passed = fp01_threshold_dynamics.get("passed", False)
        vp07_passed = vp07_threshold_effect.get("passed", False)
        vp10_passed = vp10_dlpfc_tms.get("passed", False)

        passed = fp01_passed and (vp07_passed or vp10_passed)

        result = {
            "consistency_check": "FP-01 ↔ VP-07/VP-10 TMS coupling",
            "passed": passed,
            "evidence": [
                f"FP-01 F1.4 (threshold dynamics τ_θ): {fp01_passed}",
                f"VP-07 V7.1 (TMS threshold reduction): {vp07_passed}",
                f"VP-10 P2.a (dlPFC TMS threshold shift): {vp10_passed}",
            ],
            "details": {
                "fp01_f14": fp01_threshold_dynamics,
                "vp07_v71": vp07_threshold_effect,
                "vp10_p2a": vp10_dlpfc_tms,
            },
        }

        if not passed:
            result["alert"] = (
                "COUPLING ERROR: TMS effects inconsistent with active inference predictions"
            )
            result["recommendation"] = (
                "Verify that TMS to dlPFC shifts threshold in direction predicted by FP-01. "
                "If FP-01 predicts threshold adaptation (τ_θ), TMS should modulate this parameter."
            )

        return result

    def generate_consistency_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive consistency report with all issues and recommendations.

        Returns:
            Dictionary with consistency analysis results
        """
        # Run all consistency checks
        assumption_issues = self.check_framework_assumptions()
        validation_issues = self.check_missing_validation_protocols()
        contradiction_issues = self.check_assumption_consistency()

        all_issues = assumption_issues + validation_issues + contradiction_issues

        # Generate summary
        summary = {
            "total_issues": len(all_issues),
            "assumption_violations": len(assumption_issues),
            "missing_validations": len(validation_issues),
            "contradictions": len(contradiction_issues),
            "high_severity_issues": len(
                [i for i in all_issues if i.severity == "HIGH"]
            ),
            "medium_severity_issues": len(
                [i for i in all_issues if i.severity == "MEDIUM"]
            ),
            "low_severity_issues": len([i for i in all_issues if i.severity == "LOW"]),
            "issues": all_issues,
            "recommendations": self._generate_recommendations(all_issues),
            "ready_for_falsification": len(
                [i for i in all_issues if i.severity == "HIGH"]
            )
            == 0,
        }

        return summary

    def _generate_recommendations(self, issues: List[ConsistencyIssue]) -> List[str]:
        """Generate actionable recommendations from consistency issues."""
        recommendations = []

        # Group issues by type
        contradictions = [i for i in issues if i.issue_type == "CONTRADICTION"]
        missing_validations = [
            i for i in issues if i.issue_type == "MISSING_VALIDATION"
        ]
        assumptions = [i for i in issues if i.issue_type == "ASSUMPTION_VIOLATION"]

        # Prioritize high-severity issues
        high_priority = [i for i in issues if i.severity == "HIGH"]

        if high_priority:
            recommendations.append(
                "CRITICAL: Resolve high-severity consistency issues before framework falsification"
            )

        if contradictions:
            recommendations.append(
                "Review contradictory validation-falsification results and resolve discrepancies"
            )

        if missing_validations:
            recommendations.append(
                "Run missing validation protocols to complete framework evaluation"
            )

        if assumptions:
            recommendations.append(
                "Review framework assumptions and adjust validation protocol coverage"
            )

        return recommendations
