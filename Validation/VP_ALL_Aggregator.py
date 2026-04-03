"""
APGI Framework-Level Validation Aggregator (VP-ALL)

Implements comprehensive aggregation of results from all 14 validation protocols (VP-1 to VP-14).
Requires all validation protocol files to have produced JSON result files or returned
named prediction results.

Validation Criteria:
- Primary validations: Core APGI predictions from the framework paper
- Secondary validations: Extended empirical support
- Tertiary validations: Experimental and specialized protocols

Aggregates named predictions across all validation protocols for framework
validation status reporting.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json

NAMED_PREDICTIONS = {
    # VP-1: Synthetic EEG ML Classification (P1.1–P1.3)
    "V1.1": "ML classifier achieves >85% accuracy on synthetic APGI vs baseline EEG",
    "V1.2": "Feature importance aligns with APGI theoretical signatures",
    "V1.3": "Cross-validation shows robust generalization",
    # VP-2: Behavioral Bayesian Comparison (P1.1–P1.3, V2.1–V2.3, F2.1–F2.5)
    "V2.1": "Interoceptive precision (Πⁱ) modulates visual detection threshold (d=0.40–0.60)",
    "V2.2": "Arousal amplifies the Πⁱ–threshold relationship",
    "V2.3": "High-IA individuals show greater arousal benefit (d > 0.30)",
    # VP-3: Active Inference Agent Comparison
    "V3.1": "APGI agents converge in 50–80 trials",
    "V3.2": "APGI achieves lower BIC than StandardPP and GWTonly",
    "V3.3": "Active inference outperforms reactive baselines",
    # VP-4: Phase Transition / Epistemic Level 2
    "V4.1": "Phase transition observed at predicted critical threshold",
    "V4.2": "Epistemic level 2 shows distinct signature from level 1",
    "V4.3": "Hysteresis effects consistent with APGI predictions",
    # VP-5: Evolutionary Emergence
    "V5.1": "Consciousness-relevant architectures evolve under selection pressure",
    "V5.2": "Precision-weighted information integration emerges spontaneously",
    "V5.3": "Evolutionary trajectories converge on APGI-like architectures",
    # VP-6: Liquid Network Inductive Bias
    "V6.1": "Liquid networks show superior time-series processing",
    "V6.2": "APGI-inspired architectures outperform standard RNNs",
    "V6.3": "Adaptive timescale dynamics match theoretical predictions",
    # VP-7: TMS/Pharmacological Causal Interventions (CANONICAL for P2.a-P2.c)
    "V7.1": "dlPFC TMS shifts threshold >0.1 log units",
    "V7.2": "Insula TMS reduces HEP ~30% AND PCI ~20% (double dissociation)",
    "V7.3": "High-IA × insula TMS interaction confirmed",
    # VP-8: Psychophysical Threshold Estimation (Paper 1 — Protocol 1)
    "V8.1": "Detection thresholds follow APGI predicted curve",
    "V8.2": "Interoceptive precision correlates with threshold",
    "V8.3": "Metacognitive sensitivity matches model predictions",
    # VP-9: Convergent Neural Signatures — Empirical Priority 1
    "V9.1": "P3b amplitude correlates with ignition strength",
    "V9.2": "HEP amplitude correlates with interoceptive precision",
    "V9.3": "PCI and HEP show predicted dissociation patterns",
    # VP-10: Causal Manipulations TMS/Pharmacological — Priority 2
    # NOTE: V10.1-V10.3 are SUPPLEMENTARY to VP-07 and NOT counted in 14-prediction tally
    # They provide additional evidence but VP-07 is the canonical source for P2.a-P2.c
    # VP-11: MCMC / Cultural Neuroscience — Priority 3
    "V11.1": "MCMC convergence (Gelman-Rubin R̂ ≤ 1.01)",
    "V11.2": "Cultural parameters show predicted variation",
    "V11.3": "Cross-cultural replication confirms universality",
    # VP-12: Clinical Cross-Species Convergence
    "V12.1": "DoC patients show predicted neural signature patterns",
    "V12.2": "Animal models replicate key APGI phenomena",
    "V12.3": "Cross-species scaling follows predicted power law",
    # VP-13: Epistemic Architecture Predictions P5–P12 (Paper 4)
    "V13.1": "Epistemic architecture P5 predictions confirmed",
    "V13.2": "Epistemic architecture P6–P8 predictions confirmed",
    "V13.3": "Epistemic architecture P9–P12 predictions confirmed",
    # VP-14: fMRI Anticipation/Experience Protocol 5
    "V14.1": "vmPFC–SCR anticipatory correlation r > 0.40",
    "V14.2": "vmPFC uncorrelated with posterior insula (r < 0.20)",
    "V14.3": "Anticipation-experience dynamics match APGI model",
    # VP-15: fMRI Anticipation/vmPFC Protocol (STUB)
    "V15.1": "Anticipatory insula activation onset < 500ms pre-stimulus",
    "V15.2": "vmPFC–posterior insula anticipatory connectivity r > 0.40",
    "V15.3": "Anterior/posterior insula dissociation (anticipation vs. experience)",
}

# Protocol routing table - maps named predictions to validation protocols
PREDICTION_TO_PROTOCOL = {
    # VP-1: Synthetic EEG ML Classification
    "V1.1": "VP_01_SyntheticEEG_MLClassification",
    "V1.2": "VP_01_SyntheticEEG_MLClassification",
    "V1.3": "VP_01_SyntheticEEG_MLClassification",
    # VP-2: Behavioral Bayesian Comparison
    "V2.1": "VP_02_Behavioral_BayesianComparison",
    "V2.2": "VP_02_Behavioral_BayesianComparison",
    "V2.3": "VP_02_Behavioral_BayesianComparison",
    # VP-3: Active Inference Agent Comparison
    "V3.1": "VP_03_ActiveInference_AgentSimulations",
    "V3.2": "VP_03_ActiveInference_AgentSimulations",
    "V3.3": "VP_03_ActiveInference_AgentSimulations",
    # VP-4: Phase Transition / Epistemic Level 2
    "V4.1": "VP_04_PhaseTransition_EpistemicLevel2",
    "V4.2": "VP_04_PhaseTransition_EpistemicLevel2",
    "V4.3": "VP_04_PhaseTransition_EpistemicLevel2",
    # VP-5: Evolutionary Emergence
    "V5.1": "VP_05_EvolutionaryEmergence",
    "V5.2": "VP_05_EvolutionaryEmergence",
    "V5.3": "VP_05_EvolutionaryEmergence",
    # VP-6: Liquid Network Inductive Bias
    "V6.1": "VP_06_LiquidNetwork_InductiveBias",
    "V6.2": "VP_06_LiquidNetwork_InductiveBias",
    "V6.3": "VP_06_LiquidNetwork_InductiveBias",
    # VP-7: TMS/Pharmacological Causal Interventions (CANONICAL for P2.a-P2.c)
    "V7.1": "VP_07_TMS_CausalInterventions",
    "V7.2": "VP_07_TMS_CausalInterventions",
    "V7.3": "VP_07_TMS_CausalInterventions",
    # VP-8: Psychophysical Threshold Estimation
    "V8.1": "VP_08_Psychophysical_ThresholdEstimation",
    "V8.2": "VP_08_Psychophysical_ThresholdEstimation",
    "V8.3": "VP_08_Psychophysical_ThresholdEstimation",
    # VP-9: Convergent Neural Signatures
    "V9.1": "VP_09_NeuralSignatures_EmpiricalPriority1",
    "V9.2": "VP_09_NeuralSignatures_EmpiricalPriority1",
    "V9.3": "VP_09_NeuralSignatures_EmpiricalPriority1",
    # VP-10: Causal Manipulations (SUPPLEMENTARY - NOT in 14-prediction tally)
    # V10.1-V10.3 removed - VP-07 is canonical source for these predictions
    # VP-11: MCMC / Cultural Neuroscience
    "V11.1": "VP_11_MCMC_CulturalNeuroscience_Priority3",
    "V11.2": "VP_11_MCMC_CulturalNeuroscience_Priority3",
    "V11.3": "VP_11_MCMC_CulturalNeuroscience_Priority3",
    # VP-12: Clinical Cross-Species Convergence
    "V12.1": "VP_12_Clinical_CrossSpecies_Convergence",
    "V12.2": "VP_12_Clinical_CrossSpecies_Convergence",
    "V12.3": "VP_12_Clinical_CrossSpecies_Convergence",
    # VP-13: Epistemic Architecture P5–P12
    "V13.1": "VP_13_Epistemic_Architecture",
    "V13.2": "VP_13_Epistemic_Architecture",
    "V13.3": "VP_13_Epistemic_Architecture",
    # VP-14: fMRI Anticipation/Experience
    "V14.1": "VP_14_fMRI_Anticipation_Experience",
    "V14.2": "VP_14_fMRI_Anticipation_Experience",
    "V14.3": "VP_14_fMRI_Anticipation_Experience",
    # VP-15: fMRI Anticipation/vmPFC
    "V15.1": "VP_15_fMRI_Anticipation_vmPFC",
    "V15.2": "VP_15_fMRI_Anticipation_vmPFC",
    "V15.3": "VP_15_fMRI_Anticipation_vmPFC",
}

# Protocol tier classification
PROTOCOL_TIERS = {
    "VP_01_SyntheticEEG_MLClassification": "primary",
    "VP_02_Behavioral_BayesianComparison": "primary",
    "VP_03_ActiveInference_AgentSimulations": "secondary",
    "VP_04_PhaseTransition_EpistemicLevel2": "secondary",
    "VP_05_EvolutionaryEmergence": "tertiary",
    "VP_06_LiquidNetwork_InductiveBias": "secondary",
    "VP_07_TMS_CausalInterventions": "secondary",  # CANONICAL P2 source (VP-7)
    "VP_08_Psychophysical_ThresholdEstimation": "secondary",
    "VP_09_NeuralSignatures_EmpiricalPriority1": "tertiary",
    "VP_10_CausalManipulations_Priority2": "tertiary",  # SUPPLEMENTARY to VP-7
    "VP_11_MCMC_CulturalNeuroscience_Priority3": "secondary",
    "VP_12_Clinical_CrossSpecies_Convergence": "secondary",
    "VP_13_Epistemic_Architecture": "secondary",
    "VP_14_fMRI_Anticipation_Experience": "tertiary",
    "VP_15_fMRI_Anticipation_vmPFC": "tertiary",
}

# VP-07 / VP-10 Boundary Clarification
# ====================================
# VP-07 (TMS_CausalInterventions) is the CANONICAL source for P2.a–P2.c predictions
# VP-10 (CausalManipulations_Priority2) is SUPPLEMENTARY and provides:
#   - Pharmacological specificity testing (propranolol vs insula TMS circuitry)
#   - Medication × TMS interaction testing (Type I error control)
#   - Extended validation coverage at a different abstraction level
#
# AGGREGATOR LOGIC:
# - When both VP-07 and VP-10 report P2.a–P2.c results, VP-07 takes precedence
# - VP-10 results are tagged as "supplementary" and do not contribute to
#   the 14-prediction falsification tally independently
# - VP-10-specific tests (specificity, interaction) are tracked separately
#   as "VP-10-SUPPLEMENTARY" predictions

# Supplementary predictions (VP-10 specific, not counted in 14-prediction tally)
# VP-10 provides supplementary evidence for P2.a-P2.c; VP-07 is the canonical source
SUPPLEMENTARY_PREDICTIONS = {
    "V10.1": "TMS to dlPFC selectively disrupts ignition (supplementary to V7.1)",
    "V10.2": "Propranolol modulates precision-weighted processing (supplementary to V7.2)",
    "V10.3": "Atomoxetine enhances frontoparietal connectivity (supplementary to V7.3)",
    "V10.SPEC": "Pharmacological specificity: propranolol (peripheral) vs insula TMS (cortical)",
    "V10.INT": "Medication × TMS interaction test for Type I error control",
}


def aggregate_prediction_results(
    results_input: Union[Dict, List],
) -> Dict[str, Dict[str, Any]]:
    """Load results from protocols (paths or dicts) and tally prediction pass/fail.

    VP-07 / VP-10 Boundary Handling:
    - VP-07 is the CANONICAL source for P2.a–P2.c predictions
    - VP-10 results are marked as "supplementary" and do not independently
      contribute to the 14-prediction falsification tally
    - When both report P2 results, VP-07 takes precedence

    Args:
        results_input: Dictionary of results or list of file paths/result dicts

    Returns:
        Dict mapping prediction IDs to their pass/fail status and evidence
    """
    # Initialize tallies with proper structure
    tallies: Dict[str, Dict[str, Any]] = {
        k: {"passed": False, "evidence": [], "protocol_source": None}
        for k in NAMED_PREDICTIONS
    }

    # Handle dict of results, list of paths, or list of dicts
    items: List[Union[str, dict]] = []
    if isinstance(results_input, dict):
        items = list(results_input.values())
    elif isinstance(results_input, list):
        items = results_input

    for item in items:
        data = None
        if isinstance(item, str):
            try:
                with open(item) as f:
                    data = json.load(f)
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Protocol JSON load failed for {item}: {e}")
                continue
        elif isinstance(item, dict):
            data = item

        if not data:
            continue

        # Determine protocol source
        protocol_source = data.get("protocol_id", data.get("protocol", "unknown"))

        for pred_id, result_info in data.get("named_predictions", {}).items():
            # Fix 5: Mark V10 supplementary predictions to prevent double-counting
            skip_canonical_overwrite = pred_id in SUPPLEMENTARY_PREDICTIONS

            # Skip supplementary predictions if they would overwrite canonical results
            if skip_canonical_overwrite:
                # Check if canonical VP-07 result already exists for this prediction
                existing_source = tallies.get(pred_id, {}).get("protocol_source", "")
                if "VP-07" in str(existing_source) or "VP_07" in str(existing_source):
                    # VP-07 canonical result exists, skip VP-10 supplementary
                    logger = __import__("logging").getLogger(__name__)
                    logger.info(
                        f"Skipping supplementary {pred_id} from {protocol_source}: "
                        f"VP-07 canonical result already exists"
                    )
                    continue
                # Mark as supplementary in evidence tracking
                tallies[pred_id]["is_supplementary"] = True
                tallies[pred_id]["supplementary_source"] = protocol_source

            # Skip supplementary predictions from main tally if marked
            if pred_id in SUPPLEMENTARY_PREDICTIONS:
                continue

            if pred_id in tallies:
                # Check for VP-07/VP-10 boundary (legacy P2.a-P2.c naming)
                is_p2_prediction = pred_id in ["P2.a", "P2.b", "P2.c"]
                is_vp07 = "VP_07" in str(protocol_source) or "VP-07" in str(
                    protocol_source
                )
                is_vp10 = (
                    "VP_10" in str(protocol_source)
                    or "VP-10" in str(protocol_source)
                    or "supplementary" in str(result_info.get("source_type", ""))
                )

                passed = False
                if isinstance(result_info, dict):
                    passed = result_info.get("passed", False)
                elif isinstance(result_info, bool):
                    passed = result_info

                # VP-07/VP-10 boundary logic:
                # - If VP-07 reports P2, it takes precedence (canonical)
                # - VP-10 is supplementary and doesn't contribute to 14-prediction tally
                if is_p2_prediction:
                    existing_source = tallies[pred_id].get("protocol_source")
                    if is_vp07:
                        # VP-07 is canonical - overwrite any existing
                        tallies[pred_id]["passed"] = passed
                        tallies[pred_id]["protocol_source"] = "VP-07 (canonical)"
                        tallies[pred_id]["source_type"] = "canonical"
                    elif is_vp10:
                        # VP-10 is supplementary - should not reach here due to SUPPLEMENTARY_PREDICTIONS skip
                        # but kept for safety
                        if existing_source and "VP-07" in str(existing_source):
                            tallies[pred_id]["vp10_supplementary_passed"] = passed
                        else:
                            tallies[pred_id]["passed"] = passed
                            tallies[pred_id][
                                "protocol_source"
                            ] = "VP-10 (supplementary)"
                            tallies[pred_id]["source_type"] = "supplementary"
                    else:
                        # Other protocol - treat normally
                        tallies[pred_id]["passed"] |= passed
                        tallies[pred_id]["protocol_source"] = protocol_source
                else:
                    # Non-P2 prediction - normal aggregation
                    tallies[pred_id]["passed"] |= passed

                evidence_item = "result_dict" if isinstance(item, dict) else item
                _ = evidence_item  # Silence unused warning - kept for clarity
                tallies[pred_id]["evidence"].append(
                    {
                        "source": protocol_source,
                        "passed": passed,
                        "is_supplementary": is_vp10 and is_p2_prediction,
                    }
                )

    return tallies


def compute_validation_summary(
    predictions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute summary statistics from prediction results.

    Args:
        predictions: Dict of prediction results

    Returns:
        Summary dict with pass rates by tier and overall
    """
    total = len(predictions)
    passing = sum(1 for r in predictions.values() if r.get("passed"))

    # Count by protocol tier
    primary_preds = [
        p
        for p in PREDICTION_TO_PROTOCOL
        if PROTOCOL_TIERS.get(PREDICTION_TO_PROTOCOL[p]) == "primary"
    ]
    secondary_preds = [
        p
        for p in PREDICTION_TO_PROTOCOL
        if PROTOCOL_TIERS.get(PREDICTION_TO_PROTOCOL[p]) == "secondary"
    ]
    tertiary_preds = [
        p
        for p in PREDICTION_TO_PROTOCOL
        if PROTOCOL_TIERS.get(PREDICTION_TO_PROTOCOL[p]) == "tertiary"
    ]

    primary_passed = sum(
        1 for p in primary_preds if predictions.get(p, {}).get("passed")
    )
    secondary_passed = sum(
        1 for p in secondary_preds if predictions.get(p, {}).get("passed")
    )
    tertiary_passed = sum(
        1 for p in tertiary_preds if predictions.get(p, {}).get("passed")
    )

    return {
        "total_predictions": total,
        "total_passed": passing,
        "overall_pass_rate": passing / total if total > 0 else 0,
        "primary": {
            "total": len(primary_preds),
            "passed": primary_passed,
            "pass_rate": primary_passed / len(primary_preds) if primary_preds else 0,
        },
        "secondary": {
            "total": len(secondary_preds),
            "passed": secondary_passed,
            "pass_rate": (
                secondary_passed / len(secondary_preds) if secondary_preds else 0
            ),
        },
        "tertiary": {
            "total": len(tertiary_preds),
            "passed": tertiary_passed,
            "pass_rate": tertiary_passed / len(tertiary_preds) if tertiary_preds else 0,
        },
    }


def check_framework_validated(
    predictions: Dict[str, Dict[str, Any]],
    min_primary_rate: float = 0.8,
    min_overall_rate: float = 0.7,
) -> Dict[str, Any]:
    """Check if APGI framework meets validation criteria.

    Args:
        predictions: Dict of prediction results
        min_primary_rate: Minimum pass rate for primary validations (default 80%)
        min_overall_rate: Minimum overall pass rate (default 70%)

    Returns:
        Dict with validation status and criteria check results
    """
    summary = compute_validation_summary(predictions)

    primary_adequate = summary["primary"]["pass_rate"] >= min_primary_rate
    overall_adequate = summary["overall_pass_rate"] >= min_overall_rate

    return {
        "framework_validated": primary_adequate and overall_adequate,
        "primary_adequate": primary_adequate,
        "overall_adequate": overall_adequate,
        "min_primary_rate": min_primary_rate,
        "min_overall_rate": min_overall_rate,
        "summary": summary,
    }


def run_framework_validation(
    results_input: Union[Dict, List],
    min_primary_rate: float = 0.8,
    min_overall_rate: float = 0.7,
) -> Dict[str, Any]:
    """Run complete framework validation analysis.

    Args:
        results_input: List of JSON result files or dict of outcome dicts from all protocols.
        min_primary_rate: Minimum pass rate for primary validations
        min_overall_rate: Minimum overall pass rate

    Returns:
        dict: Complete validation results with pass rates and validation status
    """
    # Aggregate predictions
    predictions = aggregate_prediction_results(results_input)

    # Compute summary
    summary = compute_validation_summary(predictions)

    # Check validation criteria
    validation_status = check_framework_validated(
        predictions, min_primary_rate, min_overall_rate
    )

    return {
        "validation_complete": True,
        "framework_validated": validation_status["framework_validated"],
        "predictions": predictions,
        "summary": summary,
        "validation_status": validation_status,
        "named_prediction_count": len(NAMED_PREDICTIONS),
        "protocol_count": len(set(PREDICTION_TO_PROTOCOL.values())),
    }


class ValidationAggregator:
    """Master aggregator for APGI framework-level validation.

    Loads JSON results from all 14 validation protocols (VP-1 to VP-14),
    tallies the 42 named predictions, and applies validation criteria.

    Attributes:
        named_predictions: Dict of 42 named predictions with descriptions
        protocol_tiers: Dict mapping protocols to their tier classification
        prediction_to_protocol: Dict mapping predictions to their source protocol
    """

    def __init__(self):
        """Initialize the validation aggregator."""
        self.named_predictions = NAMED_PREDICTIONS
        self.protocol_tiers = PROTOCOL_TIERS
        self.prediction_to_protocol = PREDICTION_TO_PROTOCOL
        self._prediction_statuses: Dict[str, Dict[str, Any]] = {}

    def set_prediction(
        self, prediction_id: str, status: str, evidence: Optional[str] = None
    ) -> None:
        """Set the status of a named prediction manually.

        Used for marking predictions as PENDING_DATA when protocols
        return STUB status (e.g., VP-15 awaiting empirical fMRI data).

        Args:
            prediction_id: The prediction ID (e.g., "V15.1")
            status: Status string ("PENDING_DATA", "PASS", "FAIL", etc.)
            evidence: Optional evidence string explaining the status
        """
        if prediction_id not in self._prediction_statuses:
            self._prediction_statuses[prediction_id] = {}

        self._prediction_statuses[prediction_id]["status"] = status
        if evidence:
            self._prediction_statuses[prediction_id]["evidence"] = evidence

    def aggregate_results(
        self, results_input: Union[Dict, List]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate prediction results from all protocols.

        Handles VP-15 STUB status by marking predictions as PENDING_DATA.

        Args:
            results_input: List of file paths, dicts, or dict of results

        Returns:
            dict: Tally of pass/fail for each named prediction
        """
        results = aggregate_prediction_results(results_input)

        # Fix 1: Handle VP-15 STUB status - mark predictions as PENDING_DATA
        items = results_input if isinstance(results_input, list) else [results_input]
        for item in items:
            if isinstance(item, dict):
                protocol_id = item.get("protocol_id", item.get("protocol", ""))
                status = item.get("status", "").upper()

                if "VP-15" in str(protocol_id) and status == "STUB":
                    # Mark all VP-15 predictions as PENDING_DATA
                    for pred_id in ["V15.1", "V15.2", "V15.3"]:
                        if pred_id in results:
                            results[pred_id]["passed"] = None
                            results[pred_id]["status"] = "PENDING_DATA"
                            results[pred_id][
                                "evidence"
                            ] = "Awaiting empirical fMRI data"

        return results

    def compute_summary(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute validation summary statistics.

        Args:
            predictions: Dict of prediction results

        Returns:
            dict: Summary with pass rates by tier and overall
        """
        return compute_validation_summary(predictions)

    def check_validation(
        self,
        predictions: Dict[str, Dict[str, Any]],
        min_primary_rate: float = 0.8,
        min_overall_rate: float = 0.7,
    ) -> Dict[str, Any]:
        """Check if framework meets validation criteria.

        Args:
            predictions: Dict of prediction results
            min_primary_rate: Minimum pass rate for primary validations
            min_overall_rate: Minimum overall pass rate

        Returns:
            dict: Validation status and criteria check results
        """
        return check_framework_validated(
            predictions, min_primary_rate, min_overall_rate
        )

    def run_full_analysis(
        self,
        results_input: Union[Dict, List],
        min_primary_rate: float = 0.8,
        min_overall_rate: float = 0.7,
    ) -> Dict[str, Any]:
        """Run complete framework validation analysis.

        Args:
            results_input: Results from all validation protocols
            min_primary_rate: Minimum pass rate for primary validations
            min_overall_rate: Minimum overall pass rate

        Returns:
            dict: Complete validation report
        """
        return run_framework_validation(
            results_input, min_primary_rate, min_overall_rate
        )

    def get_protocol_predictions(self, protocol_name: str) -> List[str]:
        """Get all prediction IDs associated with a specific protocol.

        Args:
            protocol_name: Name of the protocol (e.g., "VP_02_Behavioral_BayesianComparison")

        Returns:
            List of prediction IDs for that protocol
        """
        return [
            p
            for p, proto in self.prediction_to_protocol.items()
            if proto == protocol_name
        ]

    def get_tier_predictions(self, tier: str) -> List[str]:
        """Get all prediction IDs for a specific tier.

        Args:
            tier: Tier name ("primary", "secondary", or "tertiary")

        Returns:
            List of prediction IDs for that tier
        """
        protocols = [p for p, t in self.protocol_tiers.items() if t == tier]
        return [
            p for p, proto in self.prediction_to_protocol.items() if proto in protocols
        ]


def load_protocol_results_from_directory(
    directory: Union[str, Path],
) -> List[Dict[str, Any]]:
    """Load all JSON result files from a directory.

    Args:
        directory: Path to directory containing JSON result files

    Returns:
        List of result dictionaries
    """
    directory = Path(directory)
    results: List[Any] = []

    if not directory.exists():
        return results

    for json_file in directory.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Protocol JSON load failed for {json_file}: {e}")
            continue

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("APGI Framework Validation Aggregator (VP-ALL)")
    print("=" * 60)
    print(f"\nNamed Predictions: {len(NAMED_PREDICTIONS)}")
    print(f"Validation Protocols: {len(set(PREDICTION_TO_PROTOCOL.values()))}")
    print("\nPrediction Mapping by Protocol:")

    current_protocol = None
    for pred_id, desc in NAMED_PREDICTIONS.items():
        proto = PREDICTION_TO_PROTOCOL.get(pred_id, "Unknown")
        tier = PROTOCOL_TIERS.get(proto, "unknown")
        if proto != current_protocol:
            current_protocol = proto
            print(f"\n{proto} [{tier}]")
        print(f"  {pred_id}: {desc[:50]}...")

    # Try to load results from data directory
    aggregator = ValidationAggregator()
    results_dir = Path(__file__).parent.parent / "data_repository" / "processed_data"

    if results_dir.exists():
        json_files = list(results_dir.glob("*.json"))
        print(f"\nFound {len(json_files)} result files in {results_dir}")
        if json_files:
            results = aggregator.aggregate_results([str(f) for f in json_files])
            print(
                "\nAggregation complete. Use aggregator.run_full_analysis() for full report."
            )
    else:
        print(f"\nNo results directory found at {results_dir}")
        print("Run individual validation protocols first to generate JSON results.")

    print("\n" + "=" * 60)
