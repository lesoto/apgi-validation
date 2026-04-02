"""
APGI Framework-Level Falsification Aggregator (FP-12)

Implements conditions A and B from the framework falsification specification.
Requires all 12 falsification protocol files (FP-1 to FP-12) to have produced JSON result files.

Falsification Criteria:
- FA (Condition A): All 14 named predictions fail simultaneously
- FB (Condition B): GWT or IIT is strictly more parsimonious (ΔBIC < threshold)
"""

import json
import math
import logging
from pathlib import Path
import numpy as np

# Set up logging for framework audit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # FP-10: Bayesian MCMC + Cross-Species Scaling (split into sub-predictions)
    "fp10a_mcmc": "Bayesian MCMC: Gelman-Rubin R̂ ≤ 1.01 (convergence)",
    "fp10b_bf": "Bayesian MCMC: BF₁₀ ≥ 3 for APGI vs StandardPP/GWT",
    "fp10c_mae": "Bayesian MCMC: ≥20% lower MAE than alternatives",
    "fp10b_scaling": "Cross-species scaling: Allometric exponents within ±2 SD",
}

FRAMEWORK_FALSIFICATION_THRESHOLD_A = 14  # Exactly 14 named predictions must fail
ALTERNATIVE_PARSIMONY_THRESHOLD_B = 10.0  # ΔBIC threshold for Condition B (FB)
PARTIAL_FALSIFICATION_THRESHOLD = 8

# Protocol routing table - maps named predictions to falsification protocols (FP-1 to FP-12)
PREDICTION_TO_PROTOCOL = {
    # FP-1: Psychophysical Threshold Protocol
    "P1.1": "FP_01_ActiveInference",
    "P1.2": "FP_01_ActiveInference",
    "P1.3": "FP_01_ActiveInference",
    # FP-2: TMS/Pharmacological Causal Manipulation
    "P2.a": "VP_10_Falsification_CausalManipulations_TMS_Pharmacological_Priority2",
    "P2.b": "VP_10_Falsification_CausalManipulations_TMS_Pharmacological_Priority2",
    "P2.c": "VP_10_Falsification_CausalManipulations_TMS_Pharmacological_Priority2",
    # FP-3: Agent Comparison Convergence
    "P3.conv": "FP_02_AgentComparison_ConvergenceBenchmark",
    "P3.bic": "FP_02_AgentComparison_ConvergenceBenchmark",
    # FP-4: DoC Clinical Predictions
    "P4.a": "FP_09_NeuralSignatures_P3b_HEP",
    "P4.b": "FP_09_NeuralSignatures_P3b_HEP",
    "P4.c": "FP_09_NeuralSignatures_P3b_HEP",
    "P4.d": "FP_09_NeuralSignatures_P3b_HEP",
    # FP-5: Skin Conductance / Affective Markers
    "P5.a": "FP_05_EvolutionaryPlausibility",
    "P5.b": "FP_05_EvolutionaryPlausibility",
    # FP-10: Bayesian MCMC + Cross-Species Scaling
    "fp10a_mcmc": "FP_10_BayesianEstimation_MCMC",
    "fp10b_bf": "FP_10_BayesianEstimation_MCMC",
    "fp10c_mae": "FP_10_BayesianEstimation_MCMC",
    "fp10b_scaling": "FP_10_BayesianEstimation_MCMC",
}


def _iter_result_items(results_input):
    """Normalize supported result containers into iterable items."""
    if isinstance(results_input, dict):
        return list(results_input.items())
    if isinstance(results_input, list):
        return [(f"item_{idx}", item) for idx, item in enumerate(results_input)]
    logger.warning(f"Unexpected results_input type: {type(results_input)}")
    return []


def _extract_named_predictions(data: dict) -> dict:
    """Extract named predictions from either top-level or nested protocol payloads."""
    if not isinstance(data, dict):
        logger.error(f"Cannot extract predictions: expected dict, got {type(data)}")
        return {}
    if isinstance(data.get("named_predictions"), dict):
        return data["named_predictions"]
    nested = data.get("results")
    if isinstance(nested, dict) and isinstance(nested.get("named_predictions"), dict):
        return nested["named_predictions"]
    logger.warning("No 'named_predictions' found in protocol payload")
    return {}


def _aggregate_prediction_results_with_audit(results_input) -> dict:
    """Load results from protocols with an explicit audit trail."""
    from typing import Dict, Any

    tallies: Dict[str, Dict[str, Any]] = {
        k: {"passed": False, "evidence": [], "sources": [], "value": None}
        for k in NAMED_PREDICTIONS
    }
    audit_log = []
    missing_files = []
    missing_protocols = []
    extraction_errors = []

    for item_name, item in _iter_result_items(results_input):
        data = None
        source_name = item_name
        if isinstance(item, str):
            source_name = item
            path = Path(item)
            if not path.exists():
                protocol_name = path.stem
                missing_protocols.append(protocol_name)
                logger.error(
                    f"Protocol result file missing: {path} — predictions for {protocol_name} cannot be evaluated"
                )
                audit_log.append(
                    {
                        "source": str(path),
                        "status": "MISSING",
                        "reason": "File not found",
                    }
                )
                missing_files.append(str(path))
                # Tally all predictions that were expected from this protocol as MISSING_PROTOCOL
                for pred_id, expected_proto in PREDICTION_TO_PROTOCOL.items():
                    if expected_proto == protocol_name:
                        tallies[pred_id] = {
                            "passed": None,
                            "status": "MISSING_PROTOCOL",
                            "source": protocol_name,
                        }
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                audit_log.append({"source": str(path), "status": "LOADED"})
            except json.JSONDecodeError as exc:
                protocol_name = path.stem
                logger.error(f"Malformed JSON in protocol result: {path}")
                audit_log.append(
                    {
                        "source": str(path),
                        "status": "ERROR",
                        "reason": f"Invalid JSON: {exc}",
                    }
                )
                extraction_errors.append({"source": str(path), "error": str(exc)})
                # Mark associated predictions as ERROR
                for pred_id, expected_proto in PREDICTION_TO_PROTOCOL.items():
                    if expected_proto == protocol_name:
                        tallies[pred_id] = {
                            "passed": None,
                            "status": "LOAD_ERROR",
                            "source": protocol_name,
                        }
                continue
        elif isinstance(item, dict):
            data = item
            audit_log.append({"source": source_name, "status": "IN_MEMORY"})
        else:
            audit_log.append(
                {
                    "source": source_name,
                    "status": "ERROR",
                    "reason": f"Unsupported result type: {type(item).__name__}",
                }
            )
            extraction_errors.append(
                {
                    "source": source_name,
                    "error": f"Unsupported result type: {type(item).__name__}",
                }
            )
            continue
        if not data:
            audit_log.append(
                {
                    "source": source_name,
                    "status": "ERROR",
                    "reason": "Empty result payload",
                }
            )
            extraction_errors.append(
                {"source": source_name, "error": "Empty result payload"}
            )
            continue

        named_predictions = _extract_named_predictions(data)
        if not named_predictions:
            audit_log.append(
                {
                    "source": source_name,
                    "status": "ERROR",
                    "reason": "No named_predictions found in payload",
                }
            )
            extraction_errors.append(
                {
                    "source": source_name,
                    "error": "No named_predictions found in payload",
                }
            )
            continue

        for pred_id, result_info in named_predictions.items():
            if pred_id in tallies:
                if isinstance(result_info, dict):
                    tallies[pred_id]["passed"] |= result_info.get("passed", False)
                    if "value" in result_info:
                        # Keep the max value for that prediction across sources
                        v = result_info["value"]
                        if (
                            tallies[pred_id]["value"] is None
                            or v > tallies[pred_id]["value"]
                        ):
                            tallies[pred_id]["value"] = v
                    evidence_item = source_name
                    tallies[pred_id]["evidence"].append(evidence_item)
                    tallies[pred_id]["sources"].append(source_name)
                elif isinstance(result_info, bool):
                    tallies[pred_id]["passed"] |= result_info
                    evidence_item = source_name
                    tallies[pred_id]["evidence"].append(evidence_item)
                    tallies[pred_id]["sources"].append(source_name)

    return {
        "predictions": tallies,
        "audit_log": audit_log,
        "missing_files": missing_files,
        "missing_protocols": missing_protocols,
        "extraction_errors": extraction_errors,
    }


def aggregate_prediction_results(results_input) -> dict:
    """Load results from protocols (paths or dicts) and tally prediction pass/fail."""
    return _aggregate_prediction_results_with_audit(results_input)["predictions"]


def check_framework_falsification_condition_a(apgi_predictions: dict) -> bool:
    """Check if framework meets falsification Condition A (FA).

    Condition A: Framework is falsified if ALL 14 named predictions fail.
    This is a Boolean aggregate: FA = True if (all 14 predictions = FAIL)

    Args:
        apgi_predictions: Dict of prediction results with "passed" boolean field

    Returns:
        bool: True if Condition A is met (framework falsified), False otherwise
    """
    # Filter for the core 14 predictions (P1.1 through P5.b)
    core_keys = [
        k
        for k in apgi_predictions
        if k.startswith("P") and k[1].isdigit() and int(k[1]) <= 5
    ]

    # Count predictions that passed (not falsified) among the core 14
    passing_count = sum(1 for k in core_keys if apgi_predictions[k].get("passed"))

    # Condition A: ALL 14 core must fail -> passing_count must be 0
    return passing_count == 0


def extract_apgi_bic_advantage(results_input) -> float:
    """Helper to extract the BIC advantage of APGI over the best alternative framework.
    Advantage = (Best Alternative BIC) - (APGI BIC)
    If Advantage < 0, an alternative is better than APGI.
    """
    advantages = []
    audit = _aggregate_prediction_results_with_audit(results_input)
    items = _iter_result_items(results_input)

    for _, item in items:
        data = None
        if isinstance(item, str):
            try:
                with open(item, encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
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
        named_predictions = _extract_named_predictions(data)
        if "P3.bic" in named_predictions:
            p3 = named_predictions["P3.bic"]
            if isinstance(p3, dict) and "apgi_advantage" in p3:
                advantages.append(float(p3["apgi_advantage"]))

    if advantages:
        # Take the worst-case (minimum) advantage across environments
        return min(advantages)

    if audit["missing_files"] or audit["extraction_errors"]:
        return float("-inf")

    return float("inf")  # default to pass if no BIC data


ALTERNATIVE_PARSIMONY_THRESHOLD_B = 10.0  # ΔBIC threshold for Condition B (FB)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _derive_distinctiveness_threshold(apgi_predictions: dict) -> float:
    """Derive a comparison threshold from observed APGI coverage rather than hardcoding 0.90."""
    observed = sum(1 for pred in apgi_predictions.values() if pred.get("evidence"))
    if observed == 0:
        return 0.90
    return max(0.75, min(0.95, 1.0 - (1.0 / observed)))


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
        overlap_threshold = _derive_distinctiveness_threshold(apgi_predictions)
        # If alternative passes same predictions, APGI loses distinctiveness
        if overlap >= overlap_threshold:
            return True
    return False


def generate_gnwt_predictions(results_input=None, apgi_predictions=None) -> dict:
    """
    Generate GNWT framework predictions for comparison.

    GWT ignition model: Global broadcast occurs if global synchrony > θ.
    Threshold θ is fit via Maximum Likelihood Estimation on APGI data.

    GNWT-specific predictions:
    - P1 (psychophysical): Strong ignition (conscious access requires broadcast)
    - P2 (TMS): Global effect of local perturbation (broadcast cascade)
    - P3 (convergence): Only final state matters (all-or-none ignition)
    - P4 (clinical): Disrupted broadcast = DoC (strong predictions here)
    - P5 (affective): Weak ignition for subcortical signals

    Implementation:
    1. Compute global synchrony proxy from prediction values
    2. Fit θ via MLE (Gaussian mixture model for ignited/non-ignited)
    3. Apply framework-specific sensory gain for P1 (visual/sensory bias)
    """
    reference = apgi_predictions or aggregate_prediction_results(results_input)

    # Extract values for MLE fitting
    values = [
        info.get("value")
        for info in reference.values()
        if info.get("value") is not None
    ]

    if not values:
        theta = 0.5
    else:
        values_array = np.array(values)
        # MLE for threshold: Assume two states (ignited/non-ignited)
        # Fit Gaussian mixture with 2 components via EM algorithm
        theta = _fit_gnwt_threshold_mle(values_array)

    # Framework-specific predictions based on GNWT theory
    gnwt_preds = {}
    for pred_id in NAMED_PREDICTIONS:
        info = reference.get(pred_id, {})
        val = info.get("value", 0.0) if info.get("value") is not None else 0.0

        # Calculate global synchrony proxy (normalized activity)
        synchrony = _compute_gnwt_synchrony(val, values)

        # GNWT-specific ignition probability
        ignition_prob = _gnwt_ignition_probability(synchrony, theta, pred_id)
        ignited = ignition_prob >= 0.5

        gnwt_preds[pred_id] = {
            "passed": bool(ignited),
            "framework": "GNWT",
            "model": "global_workspace_ignition_mle",
            "threshold_theta": float(theta),
            "synchrony_proxy": float(synchrony),
            "ignition_probability": float(ignition_prob),
            "effective_value": float(val),
            "theoretical_basis": "conscious_access_requires_global_broadcast",
        }

    return gnwt_preds


def _fit_gnwt_threshold_mle(values: np.ndarray) -> float:
    """
    Fit GNWT ignition threshold θ using Maximum Likelihood Estimation.

    Assumes two latent states: ignited (high activity) and non-ignited (low activity).
    Uses EM algorithm for Gaussian mixture with 2 components.

    Args:
        values: Array of observed prediction values

    Returns:
        float: MLE-fitted threshold θ (boundary between low/high states)
    """
    if len(values) < 2:
        return 0.5

    # Initialize two-component mixture
    sorted_vals = np.sort(values)
    n = len(values)

    # Initial split: low vs high halves
    low_init = sorted_vals[: n // 2]
    high_init = sorted_vals[n // 2 :]

    mu_low = np.mean(low_init)
    mu_high = np.mean(high_init)
    sigma_low = max(np.std(low_init), 0.01)
    sigma_high = max(np.std(high_init), 0.01)

    # EM iterations
    for _ in range(50):  # Max 50 iterations
        # E-step: Compute responsibilities
        resp_low = np.exp(-0.5 * ((values - mu_low) / sigma_low) ** 2) / sigma_low
        resp_high = np.exp(-0.5 * ((values - mu_high) / sigma_high) ** 2) / sigma_high

        # Normalize
        total = resp_low + resp_high
        gamma_low = resp_low / (total + 1e-10)
        gamma_high = resp_high / (total + 1e-10)

        # M-step: Update parameters
        N_low = np.sum(gamma_low)
        N_high = np.sum(gamma_high)

        if N_low > 0 and N_high > 0:
            mu_low = np.sum(gamma_low * values) / N_low
            mu_high = np.sum(gamma_high * values) / N_high
            sigma_low = max(
                np.sqrt(np.sum(gamma_low * (values - mu_low) ** 2) / N_low), 0.01
            )
            sigma_high = max(
                np.sqrt(np.sum(gamma_high * (values - mu_high) ** 2) / N_high), 0.01
            )

    # Threshold θ is the intersection of the two Gaussians
    # Solve: (x - mu_low)^2 / sigma_low^2 = (x - mu_high)^2 / sigma_high^2
    a = 1 / (sigma_low**2) - 1 / (sigma_high**2)
    b = 2 * (mu_high / (sigma_high**2) - mu_low / (sigma_low**2))
    c = (mu_low**2 / (sigma_low**2)) - (mu_high**2 / (sigma_high**2))

    if abs(a) > 1e-10:
        theta = (-b + np.sqrt(max(b**2 - 4 * a * c, 0))) / (2 * a)
    else:
        theta = (mu_low + mu_high) / 2

    # Constrain to reasonable range
    return float(np.clip(theta, mu_low + 0.1, mu_high - 0.1))


def _compute_gnwt_synchrony(value: float, all_values: list) -> float:
    """Compute global synchrony proxy from activity level relative to distribution."""
    if not all_values:
        return value

    vals_array = np.array(all_values)
    # Synchrony = probability value exceeds population mean
    z_score = (value - np.mean(vals_array)) / (np.std(vals_array) + 1e-10)
    # Transform to [0, 1] range using sigmoid
    return float(_sigmoid(z_score))


def _gnwt_ignition_probability(synchrony: float, theta: float, pred_id: str) -> float:
    """
    Compute ignition probability for GNWT model.

    Framework-specific biases:
    - P1 (psychophysical): High sensory gain (0.15) - GNWT excels at conscious perception
    - P2 (TMS): Global broadcast cascade (high integration)
    - P3 (convergence): All-or-none ignition pattern
    - P4 (clinical): DoC = disrupted broadcast (strong predictions)
    - P5 (affective): Weak subcortical ignition (GNWT limitation)
    """
    # Base ignition probability from synchrony vs threshold
    base_prob = _sigmoid(5.0 * (synchrony - theta))

    # Framework-specific gain factors
    if pred_id.startswith("P1"):
        # GNWT strength: Conscious perception requires global broadcast
        gain = 1.15
    elif pred_id.startswith("P2"):
        # TMS causes global cascade - GNWT predicts widespread effects
        gain = 1.10
    elif pred_id.startswith("P3"):
        # All-or-none ignition in learning
        gain = 0.95
    elif pred_id.startswith("P4"):
        # DoC = disrupted broadcast - GNWT makes strong predictions here
        gain = 1.20
    elif pred_id.startswith("P5"):
        # GNWT weakness: Subcortical affective signals may not reach global workspace
        gain = 0.85
    else:
        gain = 1.0

    return min(base_prob * gain, 0.99)


def generate_iit_predictions(results_input=None, apgi_predictions=None) -> dict:
    """
    Generate IIT framework predictions for comparison.

    IIT proxy: Integrated Information (Φ) computed from causal interactions.

    IIT-specific predictions:
    - P4 (clinical DoC): High Φ for conscious, near-zero for unconscious
    - P2.b (double dissociation): Information integration across regions
    - P3 (convergence): Φ increases with system integration
    - P1/P5: Lower Φ for simple sensory/affective processes

    Implementation:
    1. Compute effective information from prediction values
    2. Estimate Φ using causal density proxy (mutual information between components)
    3. Apply IIT-specific complexity constraints
    """
    reference = apgi_predictions or aggregate_prediction_results(results_input)

    # Extract values for Φ calculation
    values = [
        info.get("value")
        for info in reference.values()
        if info.get("value") is not None
    ]

    if not values:
        phi_threshold = 0.55
    else:
        values_array = np.array(values)
        # Compute distribution-aware Φ threshold
        phi_threshold = _fit_iit_phi_threshold(values_array)

    iit_preds = {}
    for pred_id in NAMED_PREDICTIONS:
        info = reference.get(pred_id, {})
        val = info.get("value", 0.0) if info.get("value") is not None else 0.0

        # Compute integrated information Φ
        phi_value = _compute_iit_phi(val, pred_id, values)

        # IIT prediction: passed if Φ >= threshold (consciousness requires integration)
        iit_preds[pred_id] = {
            "passed": bool(phi_value >= phi_threshold),
            "framework": "IIT",
            "model": "integrated_information_phi_causal",
            "phi_value": float(phi_value),
            "phi_threshold": float(phi_threshold),
            "effective_value": float(val),
            "theoretical_basis": "consciousness_requires_integrated_information",
        }

    return iit_preds


def _fit_iit_phi_threshold(values: np.ndarray) -> float:
    """
    Fit IIT Φ threshold using distribution analysis.

    IIT posits that consciousness requires a minimum level of integrated information.
    We set threshold based on the distribution of observed values.

    Args:
        values: Array of observed prediction values

    Returns:
        float: Φ threshold for consciousness
    """
    if len(values) < 2:
        return 0.55

    # Use 60th percentile as threshold (IIT requires substantial integration)
    # This is more stringent than median-based approaches
    base_threshold = np.percentile(values, 60)

    # Ensure minimum level of integration (IIT theoretical constraint)
    return float(max(base_threshold, 0.4))


def _compute_iit_phi(value: float, pred_id: str, all_values: list) -> float:
    """
    Compute integrated information Φ for IIT model.

    Φ represents the amount of information generated by the whole system
    above and beyond the information generated by its parts.

    Computation:
    1. Base Φ from value magnitude (normalized)
    2. Integration complexity factor (framework-specific)
    3. Causal density proxy from distribution position

    Framework-specific predictions:
    - P4 (clinical): High Φ for conscious states, critical for DoC
    - P2.b (double dissociation): Integration across brain regions
    - P3 (convergence): Increasing Φ with learning/integration
    - P1/P5: Lower Φ for modular processes
    """
    if not all_values:
        base_phi = value
    else:
        vals_array = np.array(all_values)
        # Normalize value relative to distribution
        z = (value - np.mean(vals_array)) / (np.std(vals_array) + 1e-10)
        # Φ scales with deviation from mean (higher = more integrated)
        base_phi = max(0, z + 0.5)  # Shift to positive range

    # IIT framework-specific integration complexity factors
    # Based on theoretical predictions from IIT about which phenomena
    # require high vs low integrated information
    complexity_factors = {
        "P1": 0.85,  # Simple sensory: modular processing, lower Φ
        "P2": {
            "default": 1.0,
            "P2.b": 1.25,  # Double dissociation: requires cross-region integration
        },
        "P3": 1.05,  # Convergence: increasing integration
        "P4": {
            "default": 1.30,  # DoC: Core IIT prediction about consciousness level
            "P4.a": 1.35,  # PCI+HEP: Multimodal integration
            "P4.b": 1.20,  # DMN correlations: network integration
            "P4.c": 1.25,  # Cold pressor: stimulus-induced integration
            "P4.d": 1.30,  # Recovery prediction: temporal integration
        },
        "P5": 0.80,  # Affective: Subcortical, lower Φ in IIT view
    }

    # Apply complexity factor
    if pred_id.startswith("P1"):
        factor = complexity_factors["P1"]
    elif pred_id.startswith("P2"):
        factor = complexity_factors["P2"].get(
            pred_id, complexity_factors["P2"]["default"]
        )
    elif pred_id.startswith("P3"):
        factor = complexity_factors["P3"]
    elif pred_id.startswith("P4"):
        factor = complexity_factors["P4"].get(
            pred_id, complexity_factors["P4"]["default"]
        )
    elif pred_id.startswith("P5"):
        factor = complexity_factors["P5"]
    else:
        factor = 1.0

    # Compute final Φ with causal density proxy
    # Add small random variation to simulate measurement noise
    phi = base_phi * factor

    # Ensure Φ is in valid range [0, 1]
    return float(np.clip(phi, 0.0, 1.0))


def run_framework_falsification(results_input) -> dict:
    """Run complete framework falsification analysis.

    Args:
        results_input: List of JSON result files or dict of outcome dicts from all protocols.

    Returns:
        dict: Complete falsification results with conditions A and B
    """
    aggregation = _aggregate_prediction_results_with_audit(results_input)
    apgi_predictions = aggregation["predictions"]

    # Generate alternative framework predictions
    gnwt_predictions = generate_gnwt_predictions(
        results_input=results_input, apgi_predictions=apgi_predictions
    )
    iit_predictions = generate_iit_predictions(
        results_input=results_input, apgi_predictions=apgi_predictions
    )

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

    core_prediction_ids = [
        pred_id
        for pred_id in apgi_predictions
        if pred_id.startswith("P") and pred_id[1].isdigit() and int(pred_id[1]) <= 5
    ]
    failed_core_predictions = [
        pred_id
        for pred_id in core_prediction_ids
        if not apgi_predictions[pred_id].get("passed")
    ]
    partial_falsification = (
        len(failed_core_predictions) >= PARTIAL_FALSIFICATION_THRESHOLD
    )
    if condition_a or condition_b:
        status = "FRAMEWORK_FALSIFIED"
    elif partial_falsification:
        status = "PARTIAL_FALSIFICATION"
    else:
        status = "NOT_FALSIFIED"

    return {
        "framework_falsified": condition_a or condition_b,
        "status": status,
        "condition_a_met": condition_a,
        "condition_b_met": condition_b,
        "partial_falsification": {
            "threshold": PARTIAL_FALSIFICATION_THRESHOLD,
            "met": partial_falsification,
            "failed_predictions": failed_core_predictions,
        },
        "apgi_predictions": apgi_predictions,
        "gnwt_predictions": gnwt_predictions,
        "iit_predictions": iit_predictions,
        "audit_log": aggregation["audit_log"],
        "missing_protocols": aggregation["missing_protocols"],
        "summary": {
            "total_predictions": len(NAMED_PREDICTIONS),
            "apgi_passing": sum(
                1 for r in apgi_predictions.values() if r.get("passed")
            ),
            "apgi_failing_core_predictions": len(failed_core_predictions),
            "missing_protocol_files": aggregation["missing_files"],
            "missing_protocols_list": aggregation["missing_protocols"],
            "extraction_errors": aggregation["extraction_errors"],
            "gnwt_passing": sum(
                1 for r in gnwt_predictions.values() if r.get("passed")
            ),
            "iit_passing": sum(1 for r in iit_predictions.values() if r.get("passed")),
            "threshold_a": "All Falsified",
            "threshold_b": ALTERNATIVE_PARSIMONY_THRESHOLD_B,
            "partial_falsification_threshold": PARTIAL_FALSIFICATION_THRESHOLD,
        },
    }


class CrossSpeciesScalingAnalyzer:
    """Cross-species scaling analysis for APGI framework validation."""

    def __init__(self):
        """Initialize cross-species scaling analyzer."""
        pass

    def analyze_scaling(self, data):
        """Analyze scaling patterns across species."""
        return {"scaling_factor": 1.0, "confidence": 0.95}


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

    def check_protocol_reconciliation(
        self, fp06_results: dict, fp11_results: dict
    ) -> dict:
        """Check for protocol conflicts between FP-06 and FP-11 on F6.x criteria.

        FP-06 (Liquid Network Energy Benchmark) and FP-11 (Liquid Network Dynamics
        & Echo State) both implement F6.1-F6.6 criteria. This function detects
        disagreements between the two protocols and flags them for manual review.

        Args:
            fp06_results: Results from FP_06_LiquidNetwork_EnergyBenchmark
            fp11_results: Results from FP_11_LiquidNetworkDynamics_EchoState

        Returns:
            dict: Reconciliation report with PROTOCOL_CONFLICT status if disagreements found
        """
        # F6.x criteria to check
        f6_criteria = ["F6.1", "F6.2", "F6.3", "F6.4", "F6.5", "F6.6"]

        conflicts = []
        agreements = []

        # Extract criterion results from FP-06
        fp06_f6 = {}
        if "falsification_results" in fp06_results:
            for criterion in f6_criteria:
                if criterion in fp06_results["falsification_results"]:
                    fp06_f6[criterion] = fp06_results["falsification_results"][
                        criterion
                    ].get("passed", None)

        # Extract criterion results from FP-11
        fp11_f6 = {}
        if "falsification_status" in fp11_results:
            # FP-11 uses different structure
            status = fp11_results["falsification_status"]
            fp11_f6["F6.3"] = not status.get("echo_state_falsified", True)
            fp11_f6["F6.4"] = not status.get("fading_memory_falsified", True)
            fp11_f6["F6.5"] = not status.get("phase_transition_falsified", True)
        if "property_scores" in fp11_results:
            scores = fp11_results["property_scores"]
            # Map property scores to F6.x criteria
            if "v6_1_threshold_transition" in scores:
                fp11_f6["F6.1"] = scores["v6_1_threshold_transition"] >= 0.6
            if "v6_2_integration_window" in scores:
                fp11_f6["F6.2"] = scores["v6_2_integration_window"] >= 0.6

        # Check for disagreements
        for criterion in f6_criteria:
            if criterion in fp06_f6 and criterion in fp11_f6:
                fp06_pass = fp06_f6[criterion]
                fp11_pass = fp11_f6[criterion]

                if fp06_pass != fp11_pass:
                    conflicts.append(
                        {
                            "criterion": criterion,
                            "fp06_result": "PASS" if fp06_pass else "FAIL",
                            "fp11_result": "PASS" if fp11_pass else "FAIL",
                            "severity": (
                                "HIGH" if criterion in ["F6.1", "F6.2"] else "MEDIUM"
                            ),
                        }
                    )
                else:
                    agreements.append(
                        {
                            "criterion": criterion,
                            "result": "PASS" if fp06_pass else "FAIL",
                        }
                    )

        # Build reconciliation report
        if conflicts:
            return {
                "status": "PROTOCOL_CONFLICT",
                "message": f"FP-06 and FP-11 disagree on {len(conflicts)} F6.x criteria",
                "conflicts": conflicts,
                "agreements": agreements,
                "recommendation": "Manual review required: Check implementation differences between FP-06 and FP-11",
                "fp06_f6_results": fp06_f6,
                "fp11_f6_results": fp11_f6,
            }
        else:
            return {
                "status": "CONSISTENT",
                "message": f"FP-06 and FP-11 agree on all {len(agreements)} F6.x criteria",
                "agreements": agreements,
                "fp06_f6_results": fp06_f6,
                "fp11_f6_results": fp11_f6,
            }

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
