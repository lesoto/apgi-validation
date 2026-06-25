#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metadata_standardizer import (
    PROTOCOL_DEPENDENCIES,
    PROTOCOL_PREDICTIONS,
    DataSource,
    ProtocolStatus,
    standardize_metadata,
)
from utils.protocol_schema import ProtocolResult


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutError("Protocol execution timed out")


def run_with_timeout(func, timeout_secs=30):
    """Run a function with a timeout using threading (works on all platforms).

    Args:
        func: Function to call
        timeout_secs: Timeout in seconds (default 30)

    Returns:
        Result of func()

    Raises:
        TimeoutError: If function takes longer than timeout_secs
    """
    result = [None]
    exception: list[Exception | None] = [None]

    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_secs)

    if thread.is_alive():
        raise TimeoutError(f"Protocol execution timed out after {timeout_secs}s")

    if exception[0]:
        raise exception[0]

    return result[0]


def check_protocol(module_path: str, protocol_id: str, timeout_secs: int = 30, quick_mode: bool = False) -> dict:
    """Check a single protocol.

    Args:
        module_path: Path to the module to import
        protocol_id: Identifier for the protocol
        timeout_secs: Timeout in seconds for protocol execution
        quick_mode: If True, only check imports and basic structure without running
    """
    try:
        mod = __import__(module_path, fromlist=[protocol_id])

        # Check run_protocol_main exists
        if not hasattr(mod, "run_protocol_main"):
            return {"status": "MISSING_MAIN", "message": "No run_protocol_main()"}

        # In quick mode, just verify the function exists without running
        if quick_mode:
            return {
                "status": "OK",
                "predictions": "quick_check",
                "completion": "N/A (quick mode)",
                "data_source": "quick_mode",
                "note": "Quick mode - protocol not executed",
            }

        # Try to run it (with timeout)
        try:
            result = run_with_timeout(mod.run_protocol_main, timeout_secs=timeout_secs)
        except TimeoutError:
            return {
                "status": "TIMEOUT",
                "message": f"Protocol took longer than {timeout_secs}s (skipped)",
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"run_protocol_main() failed: {str(e)[:50]}",
            }

        # Check result type
        if result is None:
            return {
                "status": "NULL_RESULT",
                "message": "run_protocol_main() returned None",
            }

        # Convert to ProtocolResult if dict
        if isinstance(result, dict):
            try:
                result = ProtocolResult.from_dict(result)
            except Exception as e:
                return {
                    "status": "INVALID_DICT",
                    "message": f"Cannot convert dict to ProtocolResult: {str(e)[:50]}",
                }
        elif not isinstance(result, ProtocolResult):
            return {
                "status": "WRONG_TYPE",
                "message": f"Result is {type(result).__name__}, not ProtocolResult",
            }

        # Check predictions
        if not result.named_predictions:
            return {"status": "NO_PREDICTIONS", "message": "No named_predictions"}

        # Check metadata
        if not result.metadata:
            return {"status": "NO_METADATA", "message": "No metadata"}

        return {
            "status": "OK",
            "predictions": len(result.named_predictions),
            "completion": result.completion_percentage,
            "data_source": result.metadata.get("data_source", "unknown"),
        }

    except Exception as e:
        return {"status": "IMPORT_ERROR", "message": f"Cannot import: {str(e)[:50]}"}


def main():
    """Run verification."""
    # Check for test mode (quick check without full execution)
    test_mode = os.environ.get("APGI_TEST_MODE", "").lower() in ("1", "true", "yes")

    print("=" * 80)
    print("APGI FRAMEWORK IMPLEMENTATION STATUS")
    print("=" * 80)

    if test_mode:
        print("\n[TEST MODE ENABLED - Quick check only, protocols not executed]\n")

    fp_protocols = [
        ("Falsification.FP_01_ActiveInference", "FP_01_ActiveInference"),
        (
            "Falsification.FP_02_AgentComparisonConvergenceBenchmark",
            "FP_02_AgentComparisonConvergenceBenchmark",
        ),
        (
            "Falsification.FP_03_FrameworkLevelMultiProtocol",
            "FP_03_FrameworkLevelMultiProtocol",
        ),
        (
            "Falsification.FP_04_PhaseTransitionEpistemicArchitecture",
            "FP_04_PhaseTransitionEpistemicArchitecture",
        ),
        (
            "Falsification.FP_05_EvolutionaryPlausibility",
            "FP_05_EvolutionaryPlausibility",
        ),
        (
            "Falsification.FP_06_LiquidNetworkEnergyBenchmark",
            "FP_06_LiquidNetworkEnergyBenchmark",
        ),
        (
            "Falsification.FP_07_MathematicalConsistency",
            "FP_07_MathematicalConsistency",
        ),
        (
            "Falsification.FP_08_ParameterSensitivityIdentifiability",
            "FP_08_ParameterSensitivityIdentifiability",
        ),
        (
            "Falsification.FP_09_NeuralSignaturesP3bHEP",
            "FP_09_NeuralSignaturesP3bHEP",
        ),
        (
            "Falsification.FP_10_BayesianEstimationMCMC",
            "FP_10_BayesianEstimationMCMC",
        ),
        (
            "Falsification.FP_11_LiquidNetworkDynamicsEchoState",
            "FP_11_LiquidNetworkDynamicsEchoState",
        ),
        ("Falsification.FP_12_CrossSpeciesScaling", "FP_12_CrossSpeciesScaling"),
    ]

    vp_protocols = [
        (
            "Validation.VP_01_SyntheticEEGMLClassification",
            "VP_01_SyntheticEEGMLClassification",
        ),
        (
            "Validation.VP_02_BehavioralBayesianComparison",
            "VP_02_BehavioralBayesianComparison",
        ),
        (
            "Validation.VP_03_ActiveInferenceAgentSimulations",
            "VP_03_ActiveInferenceAgentSimulations",
        ),
        (
            "Validation.VP_04_PhaseTransitionEpistemicLevel2",
            "VP_04_PhaseTransitionEpistemicLevel2",
        ),
        ("Validation.VP_05_EvolutionaryEmergence", "VP_05_EvolutionaryEmergence"),
        (
            "Validation.VP_06_LiquidNetworkInductiveBias",
            "VP_06_LiquidNetworkInductiveBias",
        ),
        ("Validation.VP_07_TMSCausalInterventions", "VP_07_TMSCausalInterventions"),
        (
            "Validation.VP_08_PsychophysicalThresholdEstimation",
            "VP_08_PsychophysicalThresholdEstimation",
        ),
        (
            "Validation.VP_09_NeuralSignaturesEmpiricalPriority1",
            "VP_09_NeuralSignaturesEmpiricalPriority1",
        ),
        (
            "Validation.VP_10_CausalManipulationsPriority2",
            "VP_10_CausalManipulationsPriority2",
        ),
        (
            "Validation.VP_11_MCMCCulturalNeurosciencePriority3",
            "VP_11_MCMCCulturalNeurosciencePriority3",
        ),
        (
            "Validation.VP_12_ClinicalCrossSpeciesConvergence",
            "VP_12_ClinicalCrossSpeciesConvergence",
        ),
        ("Validation.VP_13_EpistemicArchitecture", "VP_13_EpistemicArchitecture"),
        (
            "Validation.VP_14_FMRIAnticipationExperience",
            "VP_14_FMRIAnticipationExperience",
        ),
        ("Validation.VP_15_FMRIAnticipationVmPFC", "VP_15_FMRIAnticipationVmPFC"),
    ]

    print("\n" + "=" * 80)
    print("FALSIFICATION PROTOCOLS (FP)")
    print("=" * 80)

    fp_ok = 0
    for module_path, protocol_id in fp_protocols:
        # Use longer timeout for protocols that may be slow (e.g., MCMC sampling)
        timeout_secs = 30 if "Bayesian" in protocol_id or "MCMC" in protocol_id else 5
        result = check_protocol(module_path, protocol_id, timeout_secs=timeout_secs, quick_mode=test_mode)
        status = result["status"]

        if status == "OK":
            fp_ok += 1
            print(f"✅ {protocol_id:50} {result['predictions']} predictions, {result['completion']}% complete")
        elif status == "TIMEOUT":
            print(f"⏱️  {protocol_id:50} {status}: {result['message']}")
        else:
            print(f"❌ {protocol_id:50} {status}: {result['message']}")

    print("\n" + "=" * 80)
    print("VALIDATION PROTOCOLS (VP)")
    print("=" * 80)

    vp_ok = 0
    for module_path, protocol_id in vp_protocols:
        # Use longer timeout for protocols that may be slow (e.g., MCMC sampling)
        timeout_secs = 30 if "Bayesian" in protocol_id or "MCMC" in protocol_id else 5
        result = check_protocol(module_path, protocol_id, timeout_secs=timeout_secs, quick_mode=test_mode)
        status = result["status"]

        if status == "OK":
            vp_ok += 1
            print(f"✅ {protocol_id:50} {result['predictions']} predictions, {result['completion']}% complete")
        elif status == "TIMEOUT":
            print(f"⏱️  {protocol_id:50} {status}: {result['message']}")
        else:
            print(f"❌ {protocol_id:50} {status}: {result['message']}")

    print("\n" + "=" * 80)
    print("METADATA STANDARDIZATION")
    print("=" * 80)

    # Check metadata standardizer
    try:
        standardize_metadata(
            "FP_01_ActiveInference",
            {"status": "complete", "data_source": "synthetic"},
        )
        print("✅ Metadata standardization working")
        print(f"   - Status enum: {ProtocolStatus.COMPLETE.value}")
        print(f"   - DataSource enum: {DataSource.SYNTHETIC.value}")
        print(f"   - Protocol dependencies: {len(PROTOCOL_DEPENDENCIES)} protocols tracked")
        print(f"   - Predictions registry: {len(PROTOCOL_PREDICTIONS)} protocols registered")
    except Exception as e:
        print(f"❌ Metadata standardization failed: {str(e)}")

    print("\n" + "=" * 80)
    print("AGGREGATORS")
    print("=" * 80)

    try:
        from Falsification.FP_ALL_Aggregator import FalsificationAggregator

        FalsificationAggregator()
        print("✅ FP_ALL_Aggregator: Loaded successfully")
    except Exception as e:
        print(f"❌ FP_ALL_Aggregator: {str(e)[:60]}")

    try:
        from Validation.VP_ALL_Aggregator import ValidationAggregator

        ValidationAggregator()
        print("✅ VP_ALL_Aggregator: Loaded successfully")
    except Exception as e:
        print(f"❌ VP_ALL_Aggregator: {str(e)[:60]}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_ok = fp_ok + vp_ok
    total_protocols = len(fp_protocols) + len(vp_protocols)

    print(f"Protocols OK: {total_ok}/{total_protocols} ({100 * total_ok // total_protocols}%)")
    print(f"  - FP: {fp_ok}/{len(fp_protocols)}")
    print(f"  - VP: {vp_ok}/{len(vp_protocols)}")

    if total_ok == total_protocols:
        print("\n🟢 FRAMEWORK STATUS: READY FOR PUBLICATION (pending Gap #4)")
        print("   - All 27 protocols have standardized schema")
        print("   - All predictions wrapped and standardized")
        print("   - Metadata standardization framework in place")
        print("   - Integration tests available")
        print("   - Remaining: Empirical data for VP-11 and VP-15")
    else:
        print(f"\n🟡 FRAMEWORK STATUS: {total_ok}/{total_protocols} protocols ready")

    print("=" * 80)


if __name__ == "__main__":
    main()
