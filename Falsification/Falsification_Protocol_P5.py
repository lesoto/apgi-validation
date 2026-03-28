"""
Falsification Protocol P5: Evolutionary Emergence
================================================

This protocol implements the meta-falsification gate for Paper 5 (Evolutionary Emergence).
It verifies whether APGI-like architectural features (hierarchical precision, ignition, 
interoceptive priors) emerge under selection pressure for survival and homeostasis.
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Falsification.Falsification_EvolutionaryPlausibility_Standard6 import (
    EvolutionaryAPGIEmergence,
)
from utils.falsification_thresholds import (
    F5_1_MIN_PROPORTION,
    F5_2_MIN_PROPORTION,
    F5_3_MIN_PROPORTION,
    F5_4_MIN_PROPORTION,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_protocol_p5():
    """Run FP-5 validation suite."""
    logger.info("Starting Falsification Protocol P5: Evolutionary Emergence")

    # Initialize evolutionary simulation
    evo = EvolutionaryAPGIEmergence(population_size=30, n_generations=50)

    # Run evolution
    history = evo.run_evolution(max_time_seconds=60)

    # Analyze results
    analysis = evo.analyze_emergence(history)
    final_freqs = analysis["final_frequencies"]

    # Define sub-criteria based on emergence of specific traits
    results = {
        "F5.1": {
            "name": "Threshold Filtering Emergence",
            "passed": bool(final_freqs["has_threshold"] >= F5_1_MIN_PROPORTION),
            "score": float(final_freqs["has_threshold"]),
            "threshold": F5_1_MIN_PROPORTION,
        },
        "F5.2": {
            "name": "Precision-Weighted Coding Emergence",
            "passed": bool(
                final_freqs["has_precision_weighting"] >= F5_2_MIN_PROPORTION
            ),
            "score": float(final_freqs["has_precision_weighting"]),
            "threshold": F5_2_MIN_PROPORTION,
        },
        "F5.3": {
            "name": "Interoceptive Prioritization Emergence",
            "passed": bool(final_freqs["has_intero_weighting"] >= F5_3_MIN_PROPORTION),
            "score": float(final_freqs["has_intero_weighting"]),
            "threshold": F5_3_MIN_PROPORTION,
        },
        "F5.4": {
            "name": "Multi-Timescale Integration Emergence",
            "passed": bool(final_freqs["has_somatic_markers"] >= F5_4_MIN_PROPORTION),
            "score": float(final_freqs["has_somatic_markers"]),
            "threshold": F5_4_MIN_PROPORTION,
        },
    }

    # Named predictions for aggregator
    named_predictions = {
        "P5.emergence": {
            "passed": all(r["passed"] for r in results.values()),
            "details": f"Emergence frequencies: {final_freqs}",
        }
    }

    # Overall result
    protocol_passed = any(r["passed"] for r in results.values())

    report = {
        "protocol": "FP-5",
        "name": "Evolutionary Emergence",
        "passed": protocol_passed,
        "results": results,
        "named_predictions": named_predictions,
        "summary": analysis,
    }

    # Save results
    output_path = project_root / "results" / "FP5_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"FP-5 complete. Passed: {protocol_passed}")
    return report


if __name__ == "__main__":
    run_protocol_p5()
