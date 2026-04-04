"""
Falsification Protocol Registry

This module provides a centralized registry mapping canonical protocol IDs
to their actual implementation filenames, resolving naming mismatches and providing
a unified interface for protocol discovery and loading.
"""

from pathlib import Path
from typing import Dict, Optional, List, Type
from dataclasses import dataclass


@dataclass
class ProtocolInfo:
    """Information about a falsification protocol."""

    canonical_id: str
    filename: str
    title: str
    description: str
    priority_level: str  # "P1", "P2", "F1", "F2", etc.
    category: str  # "Core", "Validation", "Benchmark", etc.
    implemented: bool = True
    aliases: List[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class PROTOCOL_REGISTRY:
    """
    Central registry for APGI falsification protocols.

    Maps canonical protocol IDs to actual filenames and provides metadata
    for protocol discovery and loading.
    """

    _protocols: Dict[str, ProtocolInfo] = {}

    @classmethod
    def register_protocol(cls, protocol_info: ProtocolInfo) -> None:
        """Register a protocol in the registry."""
        cls._protocols[protocol_info.canonical_id] = protocol_info

        # Also register aliases
        for alias in protocol_info.aliases:
            cls._protocols[alias] = protocol_info

    @classmethod
    def get_protocol(cls, protocol_id: str) -> Optional[ProtocolInfo]:
        """Get protocol information by ID or alias."""
        return cls._protocols.get(protocol_id)

    @classmethod
    def get_filename(cls, protocol_id: str) -> Optional[str]:
        """Get filename for a protocol by ID or alias."""
        protocol = cls.get_protocol(protocol_id)
        return protocol.filename if protocol else None

    @classmethod
    def list_protocols(cls, category: Optional[str] = None) -> List[ProtocolInfo]:
        """List all protocols, optionally filtered by category."""
        protocols = list(cls._protocols.values())

        # Remove duplicates (same protocol with different aliases)
        unique_protocols = {}
        for protocol in protocols:
            if protocol.canonical_id not in unique_protocols:
                unique_protocols[protocol.canonical_id] = protocol

        protocols = list(unique_protocols.values())

        if category:
            protocols = [p for p in protocols if p.category == category]

        return sorted(protocols, key=lambda p: p.canonical_id)

    @classmethod
    def resolve_protocol_file(cls, protocol_id: str, base_dir: Path) -> Optional[Path]:
        """Resolve a protocol ID to an actual file path."""
        filename = cls.get_filename(protocol_id)
        if filename:
            return base_dir / filename
        return None


# Initialize registry with known protocols
def _initialize_registry():
    """Initialize the protocol registry with all known falsification protocols."""

    # Core Protocols (P1, P2, etc.)
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="P1",
            filename="Falsification/FP_1_Falsification_ActiveInferenceAgents_F1F2.py",
            title="Active Inference Agents - Protocols F1 & F2",
            description="Core falsification protocols for active inference agents",
            priority_level="P1",
            category="Core",
            aliases=["F1", "F2", "ActiveInferenceAgents"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="P2",
            filename="Falsification/FP_2_Falsification_AgentComparison_ConvergenceBenchmark.py",
            title="Agent Comparison & Convergence Benchmark",
            description="Benchmark protocol for comparing agent convergence",
            priority_level="P2",
            category="Core",
            aliases=["AgentComparison", "ConvergenceBenchmark"],
        )
    )

    # Iowa Gambling Task Protocol
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="F2-Iowa",
            filename="Falsification/FP_2_Falsification_AgentComparison_ConvergenceBenchmark.py",
            title="Iowa Gambling Task Protocol",
            description="Iowa Gambling Task with AUC > 0.75 threshold",
            priority_level="F2",
            category="Validation",
            aliases=["IowaGamblingTask", "IGT"],
        )
    )

    # Cross-Species Scaling
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="P12",
            filename="Falsification/FP_12_CrossSpeciesScaling.py",
            title="Cross-Species Scaling Protocol",
            description="Cross-species validation of APGI predictions",
            priority_level="P12",
            category="Validation",
            aliases=["CrossSpecies", "Scaling"],
        )
    )

    # Information-Theoretic Phase Transition
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="F4",
            filename="Falsification/FP_4_Falsification_InformationTheoretic_PhaseTransition.py",
            title="Phase Transition with Bistability Protocol",
            description="Protocol F4 with bistability + critical slowing + hysteresis",
            priority_level="F4",
            category="Advanced",
            aliases=["PhaseTransition", "Bistability", "CriticalSlowing"],
        )
    )

    # Mathematical Consistency
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="MathematicalConsistency",
            filename="Falsification/FP_7_Falsification_MathematicalConsistency_Equations.py",
            title="Mathematical Consistency of Equations",
            description="Validation of mathematical consistency in APGI equations",
            priority_level="Validation",
            category="Core",
            aliases=["Equations", "MathConsistency"],
        )
    )

    # Bayesian Estimation Protocols
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="BayesianEstimation-MCMC",
            filename="Falsification/FP_10_BayesianEstimation_MCMC.py",
            title="Bayesian Estimation with MCMC",
            description="MCMC-based Bayesian parameter estimation",
            priority_level="Advanced",
            category="Estimation",
            aliases=["MCMC", "BayesianMCMC", "FP10"],
        )
    )

    # NOTE: FP_10_Falsification_BayesianEstimation_ParameterRecovery.py is DEPRECATED
    # and NOT registered as a standalone protocol. It re-exports from the canonical
    # FP_10_BayesianEstimation_MCMC.py file for backward compatibility only.
    # All FP-10 calls should route through the canonical MCMC file.

    # Evolutionary Plausibility
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="Standard6",
            filename="FP_5_Falsification_EvolutionaryPlausibility_Standard6.py",
            title="Evolutionary Plausibility - Standard 6",
            description="Evolutionary plausibility validation per standard 6",
            priority_level="Validation",
            category="Evolution",
            aliases=["Evolutionary", "Standard6", "EvoPlausibility"],
        )
    )

    # Neural Network Benchmarks
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="LiquidNetworkDynamics",
            filename="Falsification/FP_11_Falsification_LiquidNetworkDynamics_EchoState.py",
            title="Liquid Network Dynamics - Echo State",
            description="Liquid network dynamics validation with echo state networks",
            priority_level="Advanced",
            category="Neural",
            aliases=["LiquidNetwork", "EchoState", "LTC"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="NeuralNetwork-Energy",
            filename="Falsification/FP_6_Falsification_NeuralNetwork_EnergyBenchmark.py",
            title="Neural Network Energy Benchmark",
            description="Energy consumption benchmark for neural networks",
            priority_level="Benchmark",
            category="Neural",
            aliases=["EnergyBenchmark", "NeuralEnergy"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="NeuralSignatures-EEG",
            filename="Falsification/FP_9_Falsification_NeuralSignatures_EEG_P3b_HEP.py",
            title="Neural Signatures - EEG P3b & HEP",
            description="EEG signature validation for P3b and HEP markers",
            priority_level="Validation",
            category="Neural",
            aliases=["EEG", "P3b", "HEP", "NeuralSignatures"],
        )
    )

    # Parameter Analysis
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="ParameterSensitivity",
            filename="Falsification/FP_8_Falsification_ParameterSensitivity_Identifiability.py",
            title="Parameter Sensitivity & Identifiability",
            description="Analysis of parameter sensitivity and model identifiability",
            priority_level="Analysis",
            category="Analysis",
            aliases=["Sensitivity", "Identifiability", "ParamAnalysis"],
        )
    )

    # Framework Level
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FrameworkLevel",
            filename="Falsification/FP_3_Falsification_FrameworkLevel_MultiProtocol.py",
            title="Framework Level Multi-Protocol",
            description="Multi-protocol validation at framework level",
            priority_level="Integration",
            category="Framework",
            aliases=["MultiProtocol", "Framework", "Integration"],
        )
    )


# Initialize the registry
_initialize_registry()


def get_protocol_registry() -> Type[PROTOCOL_REGISTRY]:
    """Get the protocol registry instance."""
    return PROTOCOL_REGISTRY


# Convenience functions for external use
def resolve_protocol(
    protocol_id: str, base_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Resolve a protocol ID to its file path.

    Args:
        protocol_id: Canonical protocol ID or alias
        base_dir: Base directory for falsification files (defaults to project root)

    Returns:
        Path to the protocol file, or None if not found
    """
    if base_dir is None:
        from pathlib import Path

        base_dir = Path(__file__).parent.parent

    return PROTOCOL_REGISTRY.resolve_protocol_file(protocol_id, base_dir)


def list_available_protocols(category: Optional[str] = None) -> List[str]:
    """
    List available protocol IDs.

    Args:
        category: Optional category filter

    Returns:
        List of protocol canonical IDs
    """
    protocols = PROTOCOL_REGISTRY.list_protocols(category)
    return [p.canonical_id for p in protocols]


def get_protocol_info(protocol_id: str) -> Optional[ProtocolInfo]:
    """
    Get detailed information about a protocol.

    Args:
        protocol_id: Canonical protocol ID or alias

    Returns:
        ProtocolInfo object, or None if not found
    """
    return PROTOCOL_REGISTRY.get_protocol(protocol_id)
