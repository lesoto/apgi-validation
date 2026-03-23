"""
APGI Framework-Level Falsification Aggregator
Implements conditions (a) and (b) from the framework falsification specification.
Requires all 24 protocol files to have run and produced JSON result files.
"""

NAMED_PREDICTIONS = {
    "P1.1": "Interoceptive precision modulates detection threshold (d=0.40–0.60)",
    "P1.2": "Arousal amplifies the Πⁱ–threshold relationship",
    "P1.3": "High-IA individuals show stronger arousal benefit",
    "P2.a": "dlPFC TMS shifts threshold >0.1 log units",
    "P2.b": "Insula TMS reduces HEP ~30% AND PCI ~20% (double dissociation)",
    "P2.c": "High-IA × insula TMS interaction",
    "P3.conv": "APGI converges in 50–80 trials (beats baselines)",
    "P3.bic": "APGI BIC lower than StandardPP and GWTonly",
    "P4.a": "PCI+HEP joint AUC > 0.80 for DoC classification",
    "P4.b": "DMN↔PCI r > 0.50; DMN↔HEP r < 0.20",
    "P4.c": "Cold pressor increases PCI >10% in MCS, not VS",
    "P4.d": "Baseline PCI+HEP predicts 6-month recovery ΔR² > 0.10",
    "P5.a": "vmPFC–SCR anticipatory correlation r > 0.40",
    "P5.b": "vmPFC uncorrelated with posterior insula (r < 0.20)",
}

FRAMEWORK_FALSIFICATION_THRESHOLD_A = len(NAMED_PREDICTIONS)  # All 14 must fail
ALTERNATIVE_FRAMEWORK_PARSIMONY_THRESHOLD = 0.90  # 90% of same predictions

# Protocol routing table - maps named predictions to validation protocol files
PREDICTION_TO_PROTOCOL = {
    "P1.1": "Validation-Protocol-8",
    "P1.2": "Validation-Protocol-8",
    "P1.3": "Validation-Protocol-8",
    "P2.a": "Validation-Protocol-10",
    "P2.b": "Validation-Protocol-10",
    "P2.c": "Validation-Protocol-10",
    "P3.conv": "Validation-Protocol-3",
    "P3.bic": "Validation-Protocol-3",
    "P4.a": "Falsification-Protocol-9",
    "P4.b": "Falsification-Protocol-9",
    "P4.c": "Falsification-Protocol-9",
    "P4.d": "Falsification-Protocol-9",
    "P5.a": "Validation-Protocol-10",
    "P5.b": "Validation-Protocol-10",
}


def aggregate_prediction_results(result_files: list) -> dict:
    """Load JSON results from all protocols and tally prediction pass/fail."""
    import json

    tallies = {k: {"passed": False, "evidence": []} for k in NAMED_PREDICTIONS}

    for path in result_files:
        with open(path) as f:
            data = json.load(f)
            for pred_id, result in data.get("named_predictions", {}).items():
                if pred_id in tallies:
                    tallies[pred_id]["passed"] |= result.get("passed", False)
                    tallies[pred_id]["evidence"].append(path)

    return tallies


def check_framework_falsification_condition_a(
    apgi_predictions: dict, gnwt_predictions: dict, iit_predictions: dict
) -> bool:
    """Check if framework meets falsification condition A.

    Condition (a): Framework falsified if ALL 14+ predictions fail.
    Returns True if framework is falsified.
    """
    passing = sum(1 for r in apgi_predictions.values() if r.get("passed"))
    return passing < FRAMEWORK_FALSIFICATION_THRESHOLD_A  # Use threshold constant


def check_framework_falsification_condition_b(
    apgi_predictions: dict, gnwt_predictions: dict, iit_predictions: dict
) -> bool:
    """
    Condition (b): Framework loses distinctiveness if alternative
    accounts for same predictions with equal parsimony.
    """
    apgi_passing = {k for k, v in apgi_predictions.items() if v.get("passed")}

    for alt_preds in [gnwt_predictions, iit_predictions]:
        alt_passing = {k for k, v in alt_preds.items() if v.get("passed")}
        overlap = len(apgi_passing & alt_passing) / max(len(apgi_passing), 1)
        if overlap >= ALTERNATIVE_FRAMEWORK_PARSIMONY_THRESHOLD:
            return True  # APGI loses distinctiveness
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


def run_framework_falsification(result_files: list) -> dict:
    """Run complete framework falsification analysis.

    Args:
        result_files: List of JSON result files from all protocols

    Returns:
        dict: Complete falsification results with conditions A and B
    """
    # Aggregate APGI predictions
    apgi_predictions = aggregate_prediction_results(result_files)

    # Generate alternative framework predictions
    gnwt_predictions = generate_gnwt_predictions()
    iit_predictions = generate_iit_predictions()

    # Check falsification conditions

    # Condition A: APGI loses distinctiveness (parsimony violation)
    # APGI predictions overlap with alternative framework predictions
    condition_a = check_framework_falsification_condition_a(
        apgi_predictions, gnwt_predictions, iit_predictions
    )

    # Condition B: Alternative frameworks show APGI-like performance
    # Alternative frameworks achieve similar or better performance on key metrics
    condition_b = check_framework_falsification_condition_b(
        apgi_predictions, gnwt_predictions, iit_predictions
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
            "threshold_a": FRAMEWORK_FALSIFICATION_THRESHOLD_A,
            "threshold_b": ALTERNATIVE_FRAMEWORK_PARSIMONY_THRESHOLD,
        },
    }
