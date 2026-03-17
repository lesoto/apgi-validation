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


def aggregate_prediction_results(result_files: list) -> dict:
    """Load JSON results from all protocols and tally prediction pass/fail."""
    ...


def check_framework_falsification_condition_a(
    apgi_predictions: dict, gnwt_predictions: dict, iit_predictions: dict
) -> bool:
    """Check if framework meets falsification condition A.

    Condition (a): Framework falsified if ALL 14+ predictions fail.
    Returns True if framework is falsified.
    """
    passing = sum(1 for r in apgi_predictions.values() if r.get("passed"))
    return passing == 0  # Falsified only if zero predictions pass


def check_framework_falsification_condition_b(
    apgi_predictions: dict, gnwt_predictions: dict, iit_predictions: dict
) -> bool:
    """
    Condition (b): Framework loses distinctiveness if alternative
    accounts for same predictions with equal parsimony.
    """
