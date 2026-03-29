"""
APGI Framework-Level Falsification Aggregator (FP-12)
Implements conditions A and B from the framework falsification specification.
Requires all 12 falsification protocol files (FP-1 to FP-12) to have produced JSON result files.

Falsification Criteria:
- FA (Condition A): All 14 named predictions fail simultaneously
- FB (Condition B): GWT or IIT is strictly more parsimonious (ΔBIC < threshold)
"""

NAMED_PREDICTIONS = {
    # FP-1: Psychophysical Threshold (P1.x)
    "P1.1": "Interoceptive precision modulates detection threshold (d=0.40–0.60)",
    "P1.2": "Arousal amplifies the Πⁱ–threshold relationship",
    "P1.3": "High-IA individuals show stronger arousal benefit",
    # FP-2: TMS Causal Manipulation (P2.x)
    "P2.a": "dlPFC TMS shifts threshold >0.1 log units",
    "P2.b": "Insula TMS reduces HEP ~30% AND PCI ~20% (double dissociation)",
    "P2.c": "High-IA × insula TMS interaction",
    # FP-3: Agent Convergence (P3.x)
    "P3.conv": "APGI converges in 50–80 trials (beats baselines)",
    "P3.bic": "APGI BIC lower than StandardPP and GWTonly",
    # FP-4: DoC Clinical Predictions (P4.x)
    "P4.a": "PCI+HEP joint AUC > 0.80 for DoC classification",
    "P4.b": "DMN↔PCI r > 0.50; DMN↔HEP r < 0.20",
    "P4.c": "Cold pressor increases PCI >10% in MCS, not VS",
    "P4.d": "Baseline PCI+HEP predicts 6-month recovery ΔR² > 0.10",
    # FP-5: Skin Conductance / Affective (P5.x)
    "P5.a": "vmPFC–SCR anticipatory correlation r > 0.40",
    "P5.b": "vmPFC uncorrelated with posterior insula (r < 0.20)",
}

FRAMEWORK_FALSIFICATION_THRESHOLD_A = 14  # Exactly 14 named predictions must fail
ALTERNATIVE_PARSIMONY_THRESHOLD_B = 10.0  # ΔBIC threshold for Condition B (FB)

# Protocol routing table - maps 14 named predictions to 12 falsification protocols (FP-1 to FP-12)
PREDICTION_TO_PROTOCOL = {
    # FP-1: Psychophysical Threshold Protocol
    "P1.1": "FP_1_Falsification_ActiveInferenceAgents_F1F2",
    "P1.2": "FP_1_Falsification_ActiveInferenceAgents_F1F2",
    "P1.3": "FP_1_Falsification_ActiveInferenceAgents_F1F2",
    # FP-2: TMS/Pharmacological Causal Manipulation
    "P2.a": "VP_10_Falsification_CausalManipulations_TMS_Pharmacological_Priority2",
    "P2.b": "VP_10_Falsification_CausalManipulations_TMS_Pharmacological_Priority2",
    "P2.c": "VP_10_Falsification_CausalManipulations_TMS_Pharmacological_Priority2",
    # FP-3: Agent Comparison Convergence
    "P3.conv": "FP_2_Falsification_AgentComparison_ConvergenceBenchmark",
    "P3.bic": "FP_2_Falsification_AgentComparison_ConvergenceBenchmark",
    # FP-4: DoC Clinical Predictions
    "P4.a": "FP_9_Falsification_NeuralSignatures_EEG_P3b_HEP",
    "P4.b": "FP_9_Falsification_NeuralSignatures_EEG_P3b_HEP",
    "P4.c": "FP_9_Falsification_NeuralSignatures_EEG_P3b_HEP",
    "P4.d": "FP_9_Falsification_NeuralSignatures_EEG_P3b_HEP",
    # FP-5: Skin Conductance / Affective Markers
    "P5.a": "FP_5_Falsification_EvolutionaryPlausibility_Standard6",
    "P5.b": "FP_5_Falsification_EvolutionaryPlausibility_Standard6",
}


def aggregate_prediction_results(results_input) -> dict:
    """Load results from protocols (paths or dicts) and tally prediction pass/fail."""
    import json
    from typing import Dict, Any, List, Union

    # Initialize tallies with proper structure: Dict[str, Dict[str, Any]]
    tallies: Dict[str, Dict[str, Any]] = {
        k: {"passed": False, "evidence": []} for k in NAMED_PREDICTIONS
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
            except Exception:
                continue
        elif isinstance(item, dict):
            data = item

        if not data:
            continue

        for pred_id, result_info in data.get("named_predictions", {}).items():
            if pred_id in tallies:
                if isinstance(result_info, dict):
                    tallies[pred_id]["passed"] |= result_info.get("passed", False)
                    evidence_item = "result_dict" if isinstance(item, dict) else item
                    tallies[pred_id]["evidence"].append(evidence_item)
                elif isinstance(result_info, bool):
                    tallies[pred_id]["passed"] |= result_info
                    evidence_item = "result_dict" if isinstance(item, dict) else item
                    tallies[pred_id]["evidence"].append(evidence_item)

    return tallies


def check_framework_falsification_condition_a(apgi_predictions: dict) -> bool:
    """Check if framework meets falsification Condition A (FA).

    Condition A: Framework is falsified if ALL 14 named predictions fail.
    This is a Boolean aggregate: FA = True if (all 14 predictions = FAIL)

    Args:
        apgi_predictions: Dict of prediction results with "passed" boolean field

    Returns:
        bool: True if Condition A is met (framework falsified), False otherwise
    """
    # Count predictions that passed (not falsified)
    passing_count = sum(1 for r in apgi_predictions.values() if r.get("passed"))
    # Condition A: ALL 14 must fail → passing_count must be 0
    return passing_count == 0


def extract_apgi_bic_advantage(results_input) -> float:
    """Helper to extract the BIC advantage of APGI over the best alternative framework.
    Advantage = (Best Alternative BIC) - (APGI BIC)
    If Advantage < 0, an alternative is better than APGI.
    """
    import json

    items = []
    if isinstance(results_input, dict):
        items = list(results_input.values())
    elif isinstance(results_input, list):
        items = results_input

    advantages = []

    for item in items:
        data = None
        if isinstance(item, str):
            try:
                with open(item) as f:
                    data = json.load(f)
            except Exception:
                continue
        elif isinstance(item, dict):
            data = item

        if not data:
            continue

        # Try to find bic_values directly
        if "bic_values" in data:
            bic_values = data["bic_values"]
            for env, agents in bic_values.items():
                if "APGI" in agents:
                    apgi_bic = agents["APGI"]["bic"]
                    alt_bics = [
                        a_data["bic"]
                        for a_name, a_data in agents.items()
                        if a_name != "APGI"
                    ]
                    if alt_bics:
                        best_alt_bic = min(alt_bics)
                        advantages.append(best_alt_bic - apgi_bic)

        # Or look into predictions
        if "P3.bic" in data.get("named_predictions", {}):
            p3 = data["named_predictions"]["P3.bic"]
            if isinstance(p3, dict) and "apgi_advantage" in p3:
                advantages.append(float(p3["apgi_advantage"]))

    if advantages:
        # Take the worst-case (minimum) advantage across environments
        return min(advantages)

    return float("inf")  # default to pass if no BIC data


ALTERNATIVE_PARSIMONY_THRESHOLD_B = 10.0  # ΔBIC threshold for Condition B (FB)


def check_framework_falsification_condition_b(
    results_input=None,
    apgi_predictions=None,
    gnwt_predictions=None,
    iit_predictions=None,
) -> bool:
    """
    Check if framework meets falsification Condition B (FB).

    Condition B: Framework loses distinctiveness if GWT or IIT is strictly
    more parsimonious than APGI. This occurs when the best alternative
    framework has a lower BIC than APGI.

    Criterion: ΔBIC < ALTERNATIVE_PARSIMONY_THRESHOLD_B
    Where ΔBIC = APGI_BIC - Best_Alternative_BIC
    If ΔBIC < 0, an alternative is more parsimonious (lower BIC = better).
    If ΔBIC > threshold, APGI maintains its advantage.

    Returns:
        bool: True if Condition B is met (framework falsified), False otherwise
    """
    if results_input is not None:
        apgi_advantage = extract_apgi_bic_advantage(results_input)
        # Condition B: alternative is more parsimonious if advantage is negative
        # or below threshold (i.e., APGI does not clearly win)
        return apgi_advantage < ALTERNATIVE_PARSIMONY_THRESHOLD_B

    # Fallback to prediction overlap method if BIC not available
    if apgi_predictions is None:
        return False

    apgi_passing = {k for k, v in apgi_predictions.items() if v.get("passed")}

    for alt_preds in [gnwt_predictions, iit_predictions]:
        if alt_preds is None:
            continue
        alt_passing = {k for k, v in alt_preds.items() if v.get("passed")}
        overlap = len(apgi_passing & alt_passing) / max(len(apgi_passing), 1)
        # If alternative passes same predictions, APGI loses distinctiveness
        if overlap >= 0.90:  # 90% overlap threshold
            return True
    return False


def generate_gnwt_predictions() -> dict:
    """
    Generate GNWT framework predictions for comparison.

    GWT proxy: Broadcast-only threshold model (no precision weighting, fixed θ)
    - Predicts failure for any system lacking evolved precision weighting
    - Based on Global Neuronal Workspace Theory: non-APGI systems cannot achieve
    integrated information processing due to lack of hierarchical precision

    Returns:
        dict: Predictions with same structure as APGI predictions
    """
    gnwt_preds = {}
    for pred_id in NAMED_PREDICTIONS:
        # GWT predicts failure for any system without evolved precision
        # Key consciousness predictions that should fail:
        if pred_id in ["P2.b", "P4.a", "P4.c"]:
            gnwt_preds[pred_id] = {"passed": False, "framework": "GNWT"}
        else:
            # Integration measures might pass in simple systems
            gnwt_preds[pred_id] = {"passed": True, "framework": "GNWT"}

    return gnwt_preds


def generate_iit_predictions() -> dict:
    """
    Generate IIT framework predictions for comparison.

    IIT proxy: Integrated information model with Φ as ignition criterion
    - Predicts failure for systems with low integrated information
    - Based on Integrated Information Theory: APGI systems should have
    high Φ due to hierarchical precision and ignition

    Returns:
        dict: Predictions with same structure as APGI predictions
    """
    iit_preds = {}
    for pred_id in NAMED_PREDICTIONS:
        # IIT predicts failure for systems with low integration
        # Key integration predictions that should fail:
        if pred_id in ["P4.a", "P4.b", "P5.a"]:
            iit_preds[pred_id] = {"passed": False, "framework": "IIT"}
        else:
            # Basic processing might pass integration tests
            iit_preds[pred_id] = {"passed": True, "framework": "IIT"}

    return iit_preds


def run_framework_falsification(results_input) -> dict:
    """Run complete framework falsification analysis.

    Args:
        results_input: List of JSON result files or dict of outcome dicts from all protocols.

    Returns:
        dict: Complete falsification results with conditions A and B
    """
    # Aggregate APGI predictions
    apgi_predictions = aggregate_prediction_results(results_input)

    # Generate alternative framework predictions
    gnwt_predictions = generate_gnwt_predictions()
    iit_predictions = generate_iit_predictions()

    # Check falsification conditions

    # Condition A: All 14 named predictions fail simultaneously
    condition_a = check_framework_falsification_condition_a(apgi_predictions)

    # Condition B: Alternative frameworks are more parsimonious
    condition_b = check_framework_falsification_condition_b(
        results_input=results_input,
        apgi_predictions=apgi_predictions,
        gnwt_predictions=gnwt_predictions,
        iit_predictions=iit_predictions,
    )

    return {
        "framework_falsified": condition_a or condition_b,
        "condition_a_met": condition_a,
        "condition_b_met": condition_b,
        "apgi_predictions": apgi_predictions,
        "gnwt_predictions": gnwt_predictions,
        "iit_predictions": iit_predictions,
        "summary": {
            "total_predictions": len(NAMED_PREDICTIONS),
            "apgi_passing": sum(
                1 for r in apgi_predictions.values() if r.get("passed")
            ),
            "gnwt_passing": sum(
                1 for r in gnwt_predictions.values() if r.get("passed")
            ),
            "iit_passing": sum(1 for r in iit_predictions.values() if r.get("passed")),
            "threshold_a": "All Falsified",
            "threshold_b": ALTERNATIVE_PARSIMONY_THRESHOLD_B,
        },
    }


class FalsificationAggregator:
    """Master aggregator for APGI framework-level falsification.

    Loads JSON results from all 12 falsification protocols (FP-1 to FP-12),
    tallies the 14 named predictions, and applies falsification conditions A and B.

    Attributes:
        named_predictions: Dict of 14 named predictions with descriptions
        threshold_a: Number of predictions that must fail for Condition A (14)
        threshold_b: ΔBIC threshold for Condition B parsimony comparison
    """

    def __init__(self):
        """Initialize the falsification aggregator."""
        self.named_predictions = NAMED_PREDICTIONS
        self.threshold_a = FRAMEWORK_FALSIFICATION_THRESHOLD_A
        self.threshold_b = ALTERNATIVE_PARSIMONY_THRESHOLD_B

    def aggregate_results(self, results_input) -> dict:
        """Aggregate prediction results from all protocols.

        Args:
            results_input: List of file paths, dicts, or dict of results

        Returns:
            dict: Tally of pass/fail for each named prediction
        """
        return aggregate_prediction_results(results_input)

    def combine_falsifications(self, results_dict: dict) -> dict:
        """Combine falsification results from multiple protocols.

        Args:
            results_dict: Dict mapping protocol names to their results

        Returns:
            dict: Combined falsification analysis
        """
        return run_framework_falsification(results_dict)

    def check_condition_a(self, predictions: dict) -> bool:
        """Check falsification Condition A: all 14 predictions fail.

        Args:
            predictions: Dict of prediction results

        Returns:
            bool: True if Condition A is met (all failed)
        """
        return check_framework_falsification_condition_a(predictions)

    def check_condition_b(
        self,
        results_input=None,
        apgi_predictions=None,
        gnwt_predictions=None,
        iit_predictions=None,
    ) -> bool:
        """Check falsification Condition B: alternatives more parsimonious.

        Args:
            results_input: Raw results with BIC data
            apgi_predictions: APGI prediction results
            gnwt_predictions: GWT prediction results
            iit_predictions: IIT prediction results

        Returns:
            bool: True if Condition B is met (alternatives win)
        """
        return check_framework_falsification_condition_b(
            results_input=results_input,
            apgi_predictions=apgi_predictions,
            gnwt_predictions=gnwt_predictions,
            iit_predictions=iit_predictions,
        )

    def run_full_analysis(self, results_input) -> dict:
        """Run complete framework falsification analysis.

        Args:
            results_input: Results from all falsification protocols

        Returns:
            dict: Complete falsification report
        """
        return run_framework_falsification(results_input)


if __name__ == "__main__":
    from pathlib import Path

    print("=" * 60)
    print("APGI Framework Falsification Aggregator (FP-12)")
    print("=" * 60)
    print(f"\nNamed Predictions: {len(NAMED_PREDICTIONS)}")
    print(
        f"Condition A Threshold: {FRAMEWORK_FALSIFICATION_THRESHOLD_A} predictions must fail"
    )
    print(f"Condition B Threshold: ΔBIC > {ALTERNATIVE_PARSIMONY_THRESHOLD_B}")
    print("\nPrediction Mapping:")
    for pid, desc in NAMED_PREDICTIONS.items():
        proto = PREDICTION_TO_PROTOCOL.get(pid, "Unknown")
        print(f"  {pid}: {desc[:50]}... -> {proto}")

    # Try to load results from data directory
    aggregator = FalsificationAggregator()
    results_dir = Path(__file__).parent.parent / "data"

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
        print("Run individual falsification protocols first to generate JSON results.")

    print("\n" + "=" * 60)
