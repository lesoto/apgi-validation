#!/usr/bin/env python3
"""Update all protocols with standardized metadata.

This script scans all FP and VP protocols and updates their run_protocol_main()
functions to use standardized metadata from the metadata_standardizer module.
"""

from pathlib import Path
from typing import Tuple

# Protocols to update
FP_PROTOCOLS = [
    "Falsification/FP_01_ActiveInference.py",
    "Falsification/FP_02_AgentComparison_ConvergenceBenchmark.py",
    "Falsification/FP_03_FrameworkLevel_MultiProtocol.py",
    "Falsification/FP_04_PhaseTransition_EpistemicArchitecture.py",
    "Falsification/FP_05_EvolutionaryPlausibility.py",
    "Falsification/FP_06_LiquidNetwork_EnergyBenchmark.py",
    "Falsification/FP_07_MathematicalConsistency.py",
    "Falsification/FP_08_ParameterSensitivity_Identifiability.py",
    "Falsification/FP_09_NeuralSignatures_P3b_HEP.py",
    "Falsification/FP_10_BayesianEstimation_MCMC.py",
    "Falsification/FP_11_LiquidNetworkDynamics_EchoState.py",
    "Falsification/FP_12_CrossSpeciesScaling.py",
]

VP_PROTOCOLS = [
    "Validation/VP_01_SyntheticEEG_MLClassification.py",
    "Validation/VP_02_Behavioral_BayesianComparison.py",
    "Validation/VP_03_ActiveInference_AgentSimulations.py",
    "Validation/VP_04_PhaseTransition_EpistemicLevel2.py",
    "Validation/VP_05_EvolutionaryEmergence.py",
    "Validation/VP_06_LiquidNetwork_InductiveBias.py",
    "Validation/VP_07_TMS_CausalInterventions.py",
    "Validation/VP_08_Psychophysical_ThresholdEstimation.py",
    "Validation/VP_09_NeuralSignatures_EmpiricalPriority1.py",
    "Validation/VP_10_CausalManipulations_Priority2.py",
    "Validation/VP_11_MCMC_CulturalNeuroscience_Priority3.py",
    "Validation/VP_12_Clinical_CrossSpecies_Convergence.py",
    "Validation/VP_13_Epistemic_Architecture.py",
    "Validation/VP_14_fMRI_Anticipation_Experience.py",
    "Validation/VP_15_fMRI_Anticipation_vmPFC.py",
]


def check_protocol_metadata(filepath: str) -> Tuple[bool, str]:
    """Check if protocol has standardized metadata.

    Returns:
        (has_standardized, message)
    """
    path = Path(filepath)
    if not path.exists():
        return False, f"File not found: {filepath}"

    content = path.read_text()

    # Check if it imports metadata_standardizer
    if "from utils.metadata_standardizer import" in content:
        return True, "Already uses standardized metadata"

    # Check if it has run_protocol_main
    if "def run_protocol_main" not in content:
        return False, "No run_protocol_main function"

    return False, "Uses legacy metadata format"


def main():
    """Check all protocols."""
    print("=" * 70)
    print("PROTOCOL METADATA STANDARDIZATION STATUS")
    print("=" * 70)

    all_protocols = FP_PROTOCOLS + VP_PROTOCOLS
    standardized_count = 0
    legacy_count = 0
    missing_count = 0

    for protocol_file in all_protocols:
        has_std, msg = check_protocol_metadata(protocol_file)

        if has_std:
            status = "✓ STANDARDIZED"
            standardized_count += 1
        elif "not found" in msg.lower():
            status = "✗ MISSING"
            missing_count += 1
        else:
            status = "⚠ LEGACY"
            legacy_count += 1

        protocol_name = Path(protocol_file).stem
        print(f"{protocol_name:50} {status:20} {msg}")

    print("=" * 70)
    print(
        f"Summary: {standardized_count} standardized, {legacy_count} legacy, {missing_count} missing"
    )
    print(f"Total: {len(all_protocols)} protocols")
    print("=" * 70)

    if legacy_count > 0:
        print("\nNext step: Update legacy protocols to use standardized metadata")
        print("Run: python3 apply_metadata_standardization.py")


if __name__ == "__main__":
    main()
