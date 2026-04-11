"""Standardize metadata across all protocols.

This module provides utilities to normalize metadata fields across all FP and VP
protocols, ensuring consistent status values, data source tracking, and dependency
management.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ProtocolStatus(str, Enum):
    """Standardized protocol execution status."""

    COMPLETE = "COMPLETE"
    PENDING_EMPIRICAL = "PENDING_EMPIRICAL"
    STUB = "STUB"
    FAILED = "FAILED"


class DataSource(str, Enum):
    """Standardized data source classification."""

    EMPIRICAL = "empirical"
    SYNTHETIC = "synthetic"
    HYBRID = "hybrid"


# Protocol dependency map: which protocols must run before others
PROTOCOL_DEPENDENCIES = {
    # FP protocols
    "FP_01_ActiveInference": [],
    "FP_02_AgentComparison_ConvergenceBenchmark": ["FP_01_ActiveInference"],
    "FP_03_FrameworkLevel_MultiProtocol": [
        "FP_01_ActiveInference",
        "FP_02_AgentComparison_ConvergenceBenchmark",
    ],
    "FP_04_PhaseTransition_EpistemicArchitecture": [
        "FP_03_FrameworkLevel_MultiProtocol"
    ],
    "FP_05_EvolutionaryPlausibility": [],
    "FP_06_LiquidNetwork_EnergyBenchmark": [],
    "FP_07_MathematicalConsistency": [],
    "FP_08_ParameterSensitivity_Identifiability": [],
    "FP_09_NeuralSignatures_P3b_HEP": [],
    "FP_10_BayesianEstimation_MCMC": [],
    "FP_11_LiquidNetworkDynamics_EchoState": ["FP_06_LiquidNetwork_EnergyBenchmark"],
    "FP_12_CrossSpeciesScaling": [],
    # VP protocols
    "VP_01_SyntheticEEG_MLClassification": [],
    "VP_02_Behavioral_BayesianComparison": [],
    "VP_03_ActiveInference_AgentSimulations": ["FP_01_ActiveInference"],
    "VP_04_PhaseTransition_EpistemicLevel2": [
        "FP_04_PhaseTransition_EpistemicArchitecture"
    ],
    "VP_05_EvolutionaryEmergence": ["FP_05_EvolutionaryPlausibility"],
    "VP_06_LiquidNetwork_InductiveBias": ["FP_06_LiquidNetwork_EnergyBenchmark"],
    "VP_07_TMS_CausalInterventions": [],
    "VP_08_Psychophysical_ThresholdEstimation": [],
    "VP_09_NeuralSignatures_EmpiricalPriority1": ["FP_09_NeuralSignatures_P3b_HEP"],
    "VP_10_CausalManipulations_Priority2": [],
    "VP_11_MCMC_CulturalNeuroscience_Priority3": [],
    "VP_12_Clinical_CrossSpecies_Convergence": ["FP_12_CrossSpeciesScaling"],
    "VP_13_Epistemic_Architecture": ["FP_04_PhaseTransition_EpistemicArchitecture"],
    "VP_14_fMRI_Anticipation_Experience": [],
    "VP_15_fMRI_Anticipation_vmPFC": ["FP_05_EvolutionaryPlausibility"],
}

# Prediction IDs for each protocol
PROTOCOL_PREDICTIONS = {
    "FP_01_ActiveInference": ["P1.1", "P1.2", "P1.3"],
    "FP_02_AgentComparison_ConvergenceBenchmark": ["P2.a", "P2.b"],
    "FP_03_FrameworkLevel_MultiProtocol": [
        "P3.1",
        "P3.2",
        "P3.3",
        "P3.4",
        "P3.5",
        "P3.6",
    ],
    "FP_04_PhaseTransition_EpistemicArchitecture": ["P3.a", "P3.b", "P3.c", "P3.d"],
    "FP_05_EvolutionaryPlausibility": ["P5.a", "P5.b"],
    "FP_06_LiquidNetwork_EnergyBenchmark": ["F1.1", "F1.2", "F1.3"],
    "FP_07_MathematicalConsistency": ["P7.1", "P7.2", "P7.3"],
    "FP_08_ParameterSensitivity_Identifiability": ["P8.1", "P8.2", "P8.3", "P8.4"],
    "FP_09_NeuralSignatures_P3b_HEP": ["P4.a", "P4.b", "P4.c", "P4.d"],
    "FP_10_BayesianEstimation_MCMC": ["fp10a", "fp10b", "fp10c", "fp10b_scaling"],
    "FP_11_LiquidNetworkDynamics_EchoState": ["F3.1", "F3.2", "F3.3"],
    "FP_12_CrossSpeciesScaling": ["F4.1", "F4.2", "F4.3"],
    "VP_01_SyntheticEEG_MLClassification": ["V1.1", "V1.2", "V1.3"],
    "VP_02_Behavioral_BayesianComparison": ["V2.1", "V2.2"],
    "VP_03_ActiveInference_AgentSimulations": ["V3.1", "V3.2", "V3.3"],
    "VP_04_PhaseTransition_EpistemicLevel2": ["V4.1", "V4.2"],
    "VP_05_EvolutionaryEmergence": ["V5.1", "V5.2"],
    "VP_06_LiquidNetwork_InductiveBias": ["V6.1", "V6.2", "V6.3"],
    "VP_07_TMS_CausalInterventions": ["V7.1", "V7.2"],
    "VP_08_Psychophysical_ThresholdEstimation": ["V8.1", "V8.2"],
    "VP_09_NeuralSignatures_EmpiricalPriority1": ["V9.1", "V9.2"],
    "VP_10_CausalManipulations_Priority2": ["V10.1", "V10.2"],
    "VP_11_MCMC_CulturalNeuroscience_Priority3": ["V11.1", "V11.2", "V11.3"],
    "VP_12_Clinical_CrossSpecies_Convergence": ["V12.1", "V12.2"],
    "VP_13_Epistemic_Architecture": ["V13.1", "V13.2"],
    "VP_14_fMRI_Anticipation_Experience": ["V14.1", "V14.2"],
    "VP_15_fMRI_Anticipation_vmPFC": ["V15.1", "V15.2", "V15.3"],
}


def normalize_status(status_string: Optional[str]) -> ProtocolStatus:
    """Map protocol-specific status strings to standard enum.

    Args:
        status_string: Status value from protocol metadata

    Returns:
        Standardized ProtocolStatus enum value
    """
    if not status_string:
        return ProtocolStatus.COMPLETE

    status_map = {
        "complete": ProtocolStatus.COMPLETE,
        "COMPLETE": ProtocolStatus.COMPLETE,
        "completed": ProtocolStatus.COMPLETE,
        "COMPLETED": ProtocolStatus.COMPLETE,
        "synthetic_pending_empirical": ProtocolStatus.PENDING_EMPIRICAL,
        "SYNTHETIC_PENDING_EMPIRICAL": ProtocolStatus.PENDING_EMPIRICAL,
        "pending_empirical": ProtocolStatus.PENDING_EMPIRICAL,
        "PENDING_EMPIRICAL": ProtocolStatus.PENDING_EMPIRICAL,
        "stub": ProtocolStatus.STUB,
        "STUB": ProtocolStatus.STUB,
        "simulation_only": ProtocolStatus.STUB,
        "SIMULATION_ONLY": ProtocolStatus.STUB,
        "simulation_validated_only": ProtocolStatus.STUB,
        "SIMULATION_VALIDATED_ONLY": ProtocolStatus.STUB,
        "failed": ProtocolStatus.FAILED,
        "FAILED": ProtocolStatus.FAILED,
        "error": ProtocolStatus.FAILED,
        "ERROR": ProtocolStatus.FAILED,
    }

    normalized = status_map.get(status_string.lower(), ProtocolStatus.COMPLETE)
    return normalized


def normalize_data_source(
    data_source_string: Optional[str], legacy_metadata: Optional[Dict] = None
) -> DataSource:
    """Map protocol-specific data source strings to standard enum.

    Args:
        data_source_string: Data source value from protocol metadata
        legacy_metadata: Full metadata dict for heuristic detection

    Returns:
        Standardized DataSource enum value
    """
    if not data_source_string and legacy_metadata:
        # Heuristic: check for empirical indicators
        if any(
            k in legacy_metadata
            for k in ["empirical_data", "real_data", "clinical_data", "fmri_data"]
        ):
            return DataSource.EMPIRICAL
        if any(
            k in legacy_metadata
            for k in ["synthetic_data", "simulation", "generated_data"]
        ):
            return DataSource.SYNTHETIC

    if not data_source_string:
        return DataSource.SYNTHETIC

    source_map = {
        "empirical": DataSource.EMPIRICAL,
        "EMPIRICAL": DataSource.EMPIRICAL,
        "real": DataSource.EMPIRICAL,
        "REAL": DataSource.EMPIRICAL,
        "clinical": DataSource.EMPIRICAL,
        "CLINICAL": DataSource.EMPIRICAL,
        "fmri": DataSource.EMPIRICAL,
        "fMRI": DataSource.EMPIRICAL,
        "eeg": DataSource.EMPIRICAL,
        "EEG": DataSource.EMPIRICAL,
        "synthetic": DataSource.SYNTHETIC,
        "SYNTHETIC": DataSource.SYNTHETIC,
        "simulation": DataSource.SYNTHETIC,
        "SIMULATION": DataSource.SYNTHETIC,
        "generated": DataSource.SYNTHETIC,
        "GENERATED": DataSource.SYNTHETIC,
        "hybrid": DataSource.HYBRID,
        "HYBRID": DataSource.HYBRID,
        "mixed": DataSource.HYBRID,
        "MIXED": DataSource.HYBRID,
    }

    normalized = source_map.get(data_source_string.lower(), DataSource.SYNTHETIC)
    return normalized


def standardize_metadata(
    protocol_id: str, legacy_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Convert protocol-specific metadata to standard format.

    Args:
        protocol_id: Protocol identifier (e.g., "FP_01_ActiveInference")
        legacy_metadata: Existing metadata dict from protocol

    Returns:
        Standardized metadata dict with all required fields
    """
    if legacy_metadata is None:
        legacy_metadata = {}

    # Extract and normalize fields
    status = normalize_status(legacy_metadata.get("status"))
    data_source = normalize_data_source(
        legacy_metadata.get("data_source"), legacy_metadata
    )
    completion_percentage = legacy_metadata.get("completion_percentage", 50)
    errors = legacy_metadata.get("errors", [])

    # Get dependencies and predictions for this protocol
    dependencies = PROTOCOL_DEPENDENCIES.get(protocol_id, [])
    predictions_evaluated = PROTOCOL_PREDICTIONS.get(protocol_id, [])

    # Build standardized metadata
    standardized = {
        "status": status.value,
        "data_source": data_source.value,
        "completion_percentage": int(completion_percentage),
        "errors": errors if isinstance(errors, list) else [str(errors)],
        "dependencies": dependencies,
        "predictions_evaluated": predictions_evaluated,
        "protocol_specific": {
            k: v
            for k, v in legacy_metadata.items()
            if k
            not in [
                "status",
                "data_source",
                "completion_percentage",
                "errors",
                "dependencies",
                "predictions_evaluated",
            ]
        },
    }

    return standardized


@dataclass
class StandardizedMetadata:
    """Dataclass for standardized protocol metadata."""

    protocol_id: str
    status: ProtocolStatus
    data_source: DataSource
    completion_percentage: int
    errors: List[str]
    dependencies: List[str]
    predictions_evaluated: List[str]
    protocol_specific: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "protocol_id": self.protocol_id,
            "status": self.status.value,
            "data_source": self.data_source.value,
            "completion_percentage": self.completion_percentage,
            "errors": self.errors,
            "dependencies": self.dependencies,
            "predictions_evaluated": self.predictions_evaluated,
            "protocol_specific": self.protocol_specific,
        }

    @classmethod
    def from_dict(
        cls, protocol_id: str, data: Dict[str, Any]
    ) -> "StandardizedMetadata":
        """Create from dictionary."""
        return cls(
            protocol_id=protocol_id,
            status=ProtocolStatus(data.get("status", "COMPLETE")),
            data_source=DataSource(data.get("data_source", "synthetic")),
            completion_percentage=int(data.get("completion_percentage", 50)),
            errors=data.get("errors", []),
            dependencies=data.get("dependencies", []),
            predictions_evaluated=data.get("predictions_evaluated", []),
            protocol_specific=data.get("protocol_specific", {}),
        )
