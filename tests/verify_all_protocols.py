"""
Comprehensive verification test for all 27 APGI protocols.
Validates that all FP and VP protocols return proper ProtocolResult objects.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# All 12 FP protocols
FP_PROTOCOLS = [
    ("FP-01", "Falsification.FP_01_ActiveInference"),
    ("FP-02", "Falsification.FP_02_AgentComparison_ConvergenceBenchmark"),
    ("FP-03", "Falsification.FP_03_FrameworkLevel_MultiProtocol"),
    ("FP-04", "Falsification.FP_04_PhaseTransition_EpistemicArchitecture"),
    ("FP-05", "Falsification.FP_05_EvolutionaryPlausibility"),
    ("FP-06", "Falsification.FP_06_LiquidNetwork_EnergyBenchmark"),
    ("FP-07", "Falsification.FP_07_MathematicalConsistency"),
    ("FP-08", "Falsification.FP_08_ParameterSensitivity_Identifiability"),
    ("FP-09", "Falsification.FP_09_NeuralSignatures_P3b_HEP"),
    ("FP-10", "Falsification.FP_10_BayesianEstimation_MCMC"),
    ("FP-11", "Falsification.FP_11_LiquidNetworkDynamics_EchoState"),
    ("FP-12", "Falsification.FP_12_CrossSpeciesScaling"),
]

# All 15 VP protocols
VP_PROTOCOLS = [
    ("VP-01", "Validation.VP_01_SyntheticEEG_MLClassification"),
    ("VP-02", "Validation.VP_02_Behavioral_BayesianComparison"),
    ("VP-03", "Validation.VP_03_ActiveInference_AgentSimulations"),
    ("VP-04", "Validation.VP_04_PhaseTransition_EpistemicLevel2"),
    ("VP-05", "Validation.VP_05_EvolutionaryEmergence"),
    ("VP-06", "Validation.VP_06_LiquidNetwork_InductiveBias"),
    ("VP-07", "Validation.VP_07_TMS_CausalInterventions"),
    ("VP-08", "Validation.VP_08_Psychophysical_ThresholdEstimation"),
    ("VP-09", "Validation.VP_09_NeuralSignatures_EmpiricalPriority1"),
    ("VP-10", "Validation.VP_10_CausalManipulations_Priority2"),
    ("VP-11", "Validation.VP_11_MCMC_CulturalNeuroscience_Priority3"),
    ("VP-12", "Validation.VP_12_Clinical_CrossSpecies_Convergence"),
    ("VP-13", "Validation.VP_13_Epistemic_Architecture"),
    ("VP-14", "Validation.VP_14_fMRI_Anticipation_Experience"),
    ("VP-15", "Validation.VP_15_fMRI_Anticipation_vmPFC"),
]


def verify_protocol(name, module_path, quick_mode=True):
    """Verify a protocol has working run_protocol_main."""
    try:
        module = __import__(module_path, fromlist=["run_protocol_main"])
        if not hasattr(module, "run_protocol_main"):
            return False, "Missing run_protocol_main function"

        if quick_mode:
            # Just verify the function exists and has correct signature
            import inspect

            sig = inspect.signature(module.run_protocol_main)
            return True, f"run_protocol_main{sig} available"
        else:
            # Actually run the protocol (slow)
            result = module.run_protocol_main()
            if hasattr(result, "protocol_id"):
                n_preds = len(getattr(result, "named_predictions", {}))
                return True, f"ProtocolResult with {n_preds} predictions"
            else:
                return False, f"Returns {type(result).__name__}, not ProtocolResult"
    except Exception as e:
        return False, str(e)[:60]


def main():
    print("=" * 70)
    print("APGI PROTOCOL VERIFICATION")
    print("=" * 70)

    quick_mode = "--quick" in sys.argv or "-q" in sys.argv
    if not quick_mode:
        print("\n(Use --quick for fast verification without running protocols)")

    print("\n--- FP Protocols (Falsification) ---")
    fp_ok = 0
    for name, path in FP_PROTOCOLS:
        ok, msg = verify_protocol(name, path, quick_mode)
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {msg}")
        if ok:
            fp_ok += 1

    print("\n--- VP Protocols (Validation) ---")
    vp_ok = 0
    for name, path in VP_PROTOCOLS:
        ok, msg = verify_protocol(name, path, quick_mode)
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {msg}")
        if ok:
            vp_ok += 1

    print("\n" + "=" * 70)
    print(
        f"RESULTS: {fp_ok}/{len(FP_PROTOCOLS)} FP + {vp_ok}/{len(VP_PROTOCOLS)} VP protocols OK"
    )
    total_ok = fp_ok + vp_ok
    total = len(FP_PROTOCOLS) + len(VP_PROTOCOLS)
    print(f"TOTAL: {total_ok}/{total} protocols verified ({100 * total_ok // total}%)")
    print("=" * 70)

    # Verify aggregators
    print("\n--- Aggregators ---")
    try:
        from Falsification.FP_ALL_Aggregator import NAMED_PREDICTIONS as FP_PREDICTIONS

        print(f"  ✓ FP_ALL_Aggregator: {len(FP_PREDICTIONS)} predictions defined")
    except Exception as e:
        print(f"  ✗ FP_ALL_Aggregator: {e}")

    try:
        from Validation.VP_ALL_Aggregator import NAMED_PREDICTIONS as VP_PREDICTIONS

        print(f"  ✓ VP_ALL_Aggregator: {len(VP_PREDICTIONS)} predictions defined")
    except Exception as e:
        print(f"  ✗ VP_ALL_Aggregator: {e}")

    return 0 if total_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
