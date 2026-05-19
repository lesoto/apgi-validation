"""
LEVEL DESIGNATION: Level 2 (information-theoretic)

Bridge to Level 1
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type


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
    """Initialize the protocol registry with all known protocols."""

    # ------------------------------------------------------------------
    # Experimental Protocol Specifications (JSON, protocols/ folder)
    # ------------------------------------------------------------------
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P01",
            filename="protocols/protocol_1_eeg_interoceptive_gating.json",
            title="EEG Study of Interoceptive Precision Gating",
            description="HEP-P3b coupling and cardiac-phase detection advantage (P1a–P1c)",
            priority_level="Empirical",
            category="ExperimentalProtocol",
            aliases=["P01", "EEG-Interoceptive", "HEP-P3b"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P02",
            filename="protocols/protocol_2_tms_insular_gating.json",
            title="TMS Causal Manipulation of Interoceptive Gating",
            description="Insula/thalamus TMS causal test of θₜ regulation (P2a–P2c)",
            priority_level="Empirical",
            category="ExperimentalProtocol",
            aliases=["P02", "TMS-Insular", "TMS-Causal"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P03",
            filename="protocols/protocol_3_active_inference_simulations.json",
            title="Active Inference Agent Simulations",
            description="Somatic marker M̂(c,a) and adaptive advantages (P3a–P3d)",
            priority_level="Computational",
            category="ExperimentalProtocol",
            aliases=["P03", "ActiveInference-Sim", "SomaticMarker"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P04",
            filename="protocols/protocol_4_disorders_of_consciousness.json",
            title="Disorders of Consciousness: HEP + PCI Joint Model",
            description="Joint HEP+PCI prediction of DoC recovery (P4a–P4d)",
            priority_level="Clinical",
            category="ExperimentalProtocol",
            aliases=["P04", "DoC", "Consciousness-Disorders"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P05",
            filename="protocols/protocol_5_fmri_anticipation.json",
            title="fMRI Anticipation vs Experience",
            description="vmPFC-posterior insula coupling during anticipation (P5a–P5d)",
            priority_level="Empirical",
            category="ExperimentalProtocol",
            aliases=["P05", "fMRI-Anticipation", "vmPFC-Coupling"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="APGI-P06",
            filename="protocols/protocol_6_icEEG_ignition_dynamics.json",
            title="intracranial EEG Ignition Bifurcation Dynamics",
            description="All-or-none bistable firing and critical slowing in iEEG (P6a–P6d)",
            priority_level="Empirical",
            category="ExperimentalProtocol",
            aliases=["P06", "iEEG-Bifurcation", "CriticalSlowing-iEEG"],
        )
    )

    # ------------------------------------------------------------------
    # Falsification Protocols (FP_01 – FP_12)
    # ------------------------------------------------------------------
    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-01",
            filename="Falsification/FP_01_ActiveInference.py",
            title="Active Inference Agents — F1 & F2",
            description="Core falsification protocols for active inference agents",
            priority_level="P1",
            category="Falsification",
            aliases=["P1", "F1", "F2", "ActiveInferenceAgents", "FP_01_ActiveInference"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-02",
            filename="Falsification/FP_02_AgentComparison_ConvergenceBenchmark.py",
            title="Agent Comparison & Convergence Benchmark",
            description="Benchmark protocol for comparing agent convergence",
            priority_level="P2",
            category="Falsification",
            aliases=[
                "P2",
                "AgentComparison",
                "ConvergenceBenchmark",
                "IGT",
                "IowaGamblingTask",
                "F2-Iowa",
                "FP_02_AgentComparison_ConvergenceBenchmark",
            ],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-03",
            filename="Falsification/FP_03_FrameworkLevel_MultiProtocol.py",
            title="Framework Level Multi-Protocol",
            description="Multi-protocol validation at framework level",
            priority_level="Integration",
            category="Falsification",
            aliases=[
                "FrameworkLevel",
                "MultiProtocol",
                "Framework",
                "Integration",
                "FP_03_FrameworkLevel_MultiProtocol",
            ],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-04",
            filename="Falsification/FP_04_PhaseTransition_EpistemicArchitecture.py",
            title="Phase Transition & Bistability (APGI-P06 linked)",
            description="Bistability + critical slowing + hysteresis; linked to APGI-P06 iEEG",
            priority_level="F4",
            category="Falsification",
            aliases=[
                "F4",
                "PhaseTransition",
                "Bistability",
                "CriticalSlowing",
                "FP_04_PhaseTransition_EpistemicArchitecture",
            ],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-05",
            filename="Falsification/FP_05_EvolutionaryPlausibility.py",
            title="Evolutionary Plausibility",
            description="Evolutionary plausibility validation",
            priority_level="Validation",
            category="Falsification",
            aliases=["Standard6", "Evolutionary", "EvoPlausibility", "FP_05_EvolutionaryPlausibility"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-06",
            filename="Falsification/FP_06_LiquidNetwork_EnergyBenchmark.py",
            title="Neural Network Energy Benchmark",
            description="Energy consumption benchmark for liquid/neural networks",
            priority_level="Benchmark",
            category="Falsification",
            aliases=["EnergyBenchmark", "NeuralEnergy", "FP_06_LiquidNetwork_EnergyBenchmark"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-07",
            filename="Falsification/FP_07_MathematicalConsistency.py",
            title="Mathematical Consistency of Equations",
            description="Validation of mathematical consistency in APGI equations",
            priority_level="Validation",
            category="Falsification",
            aliases=["MathematicalConsistency", "Equations", "MathConsistency", "FP_07_MathematicalConsistency"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-08",
            filename="Falsification/FP_08_ParameterSensitivity_Identifiability.py",
            title="Parameter Sensitivity & Identifiability",
            description="Analysis of parameter sensitivity and model identifiability",
            priority_level="Analysis",
            category="Falsification",
            aliases=[
                "ParameterSensitivity",
                "Sensitivity",
                "Identifiability",
                "ParamAnalysis",
                "FP_08_ParameterSensitivity_Identifiability",
            ],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-09",
            filename="Falsification/FP_09_NeuralSignatures_P3b_HEP.py",
            title="Neural Signatures — EEG P3b & HEP (APGI-P01 linked)",
            description="EEG P3b/HEP signature validation; linked to APGI-P01",
            priority_level="Validation",
            category="Falsification",
            aliases=["NeuralSignatures-EEG", "EEG", "P3b", "HEP", "NeuralSignatures", "FP_09_NeuralSignatures_P3b_HEP"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-10",
            filename="Falsification/FP_10_BayesianEstimation_MCMC.py",
            title="Bayesian Estimation with MCMC",
            description="MCMC-based Bayesian parameter estimation",
            priority_level="Advanced",
            category="Falsification",
            # NOTE: FP_10_Falsification_BayesianEstimation_ParameterRecovery.py is
            # DEPRECATED — all FP-10 calls route through the canonical MCMC file.
            aliases=["BayesianEstimation-MCMC", "MCMC", "BayesianMCMC", "FP10", "FP_10_BayesianEstimation_MCMC"],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-11",
            filename="Falsification/FP_11_LiquidNetworkDynamics_EchoState.py",
            title="Liquid Network Dynamics — Echo State",
            description="Liquid network dynamics validation with echo state networks",
            priority_level="Advanced",
            category="Falsification",
            aliases=[
                "LiquidNetworkDynamics",
                "LiquidNetwork",
                "EchoState",
                "LTC",
                "FP_11_LiquidNetworkDynamics_EchoState",
            ],
        )
    )

    PROTOCOL_REGISTRY.register_protocol(
        ProtocolInfo(
            canonical_id="FP-12",
            filename="Falsification/FP_12_CrossSpeciesScaling.py",
            title="Cross-Species Scaling",
            description="Cross-species validation of APGI predictions",
            priority_level="P12",
            category="Falsification",
            aliases=["P12", "CrossSpecies", "Scaling", "FP_12_CrossSpeciesScaling"],
        )
    )

    # ------------------------------------------------------------------
    # Validation Protocols (VP_01 – VP_21)
    # ------------------------------------------------------------------
    _vp_entries = [
        (
            "VP-01",
            "VP_01_SyntheticEEG_MLClassification.py",
            "Synthetic EEG ML Classification",
            "ML classification of synthetic EEG with precision-weighting",
            ["VP01", "SyntheticEEG", "MLClassification"],
        ),
        (
            "VP-02",
            "VP_02_Behavioral_BayesianComparison.py",
            "Behavioral Bayesian Model Comparison",
            "Bayesian model comparison on behavioral data",
            ["VP02", "BehavioralBayes", "BayesianComparison"],
        ),
        (
            "VP-03",
            "VP_03_ActiveInference_AgentSimulations.py",
            "Active Inference Agent Simulations (APGI-P03 linked)",
            "Agent-based active inference; linked to APGI-P03",
            ["VP03", "ActiveInference", "AgentSimulations"],
        ),
        (
            "VP-04",
            "VP_04_PhaseTransition_EpistemicLevel2.py",
            "Phase Transition Epistemic Level 2",
            "Phase transition dynamics at epistemic level 2",
            ["VP04", "PhaseTransitionEpistemic"],
        ),
        (
            "VP-05",
            "VP_05_EvolutionaryEmergence.py",
            "Evolutionary Emergence",
            "Evolutionary plausibility of APGI parameters",
            ["VP05", "EvolutionaryEmergence"],
        ),
        (
            "VP-06",
            "VP_06_LiquidNetwork_InductiveBias.py",
            "Liquid Network Inductive Bias",
            "Inductive bias testing for LTC networks",
            ["VP06", "LiquidNetworkBias"],
        ),
        (
            "VP-07",
            "VP_07_TMS_CausalInterventions.py",
            "TMS Causal Interventions (APGI-P02 linked)",
            "TMS/pharmacological intervention predictions; linked to APGI-P02",
            ["VP07", "TMS", "CausalInterventions"],
        ),
        (
            "VP-08",
            "VP_08_Psychophysical_ThresholdEstimation.py",
            "Psychophysical Threshold Estimation",
            "Threshold estimation from psychophysical data",
            ["VP08", "Psychophysical", "ThresholdEstimation"],
        ),
        (
            "VP-09",
            "VP_09_NeuralSignatures_EmpiricalPriority1.py",
            "Convergent Neural Signatures — Priority 1 (APGI-P01 linked)",
            "Neural signature convergence; linked to APGI-P01 EEG protocol",
            ["VP09", "NeuralSignatures", "Priority1", "HEP-VP"],
        ),
        (
            "VP-10",
            "VP_10_CausalManipulations_Priority2.py",
            "Causal Manipulations — Priority 2 (APGI-P02 linked)",
            "Causal manipulation validation; linked to APGI-P02 TMS",
            ["VP10", "CausalManipulations", "Priority2"],
        ),
        (
            "VP-11",
            "VP_11_MCMC_CulturalNeuroscience_Priority3.py",
            "MCMC Cultural Neuroscience — Priority 3",
            "MCMC-based cultural neuroscience testing",
            ["VP11", "MCMC-Cultural", "CulturalNeuroscience"],
        ),
        (
            "VP-12",
            "VP_12_Clinical_CrossSpecies_Convergence.py",
            "Clinical & Cross-Species Convergence (APGI-P04 linked)",
            "Clinical/cross-species convergence; linked to APGI-P04 DoC",
            ["VP12", "Clinical", "CrossSpeciesConvergence"],
        ),
        (
            "VP-13",
            "VP_13_Epistemic_Architecture.py",
            "Epistemic Architecture",
            "Epistemic architecture validation",
            ["VP13", "EpistemicArchitecture"],
        ),
        (
            "VP-14",
            "VP_14_fMRI_Anticipation_Experience.py",
            "fMRI Anticipation vs Experience (APGI-P05 linked)",
            "BOLD tracking of S(t)/M(t); linked to APGI-P05 fMRI",
            ["VP14", "fMRI-Anticipation"],
        ),
        (
            "VP-15",
            "VP_15_fMRI_Anticipation_vmPFC.py",
            "fMRI Anticipation vmPFC",
            "vmPFC-posterior insula connectivity in anticipation",
            ["VP15", "vmPFC-fMRI"],
        ),
        (
            "VP-16",
            "VP_16_Metabolic_ATP_GroundTruth.py",
            "Metabolic ATP Ground Truth",
            "Metabolic cost validation against ATP measurements",
            ["VP16", "MetabolicATP"],
        ),
        (
            "VP-17",
            "VP_17_AllenVisualCoding_Fatigue.py",
            "Allen Visual Coding Fatigue",
            "Neural fatigue effects in Allen Brain visual coding data",
            ["VP17", "AllenVisualCoding"],
        ),
        (
            "VP-18",
            "VP_18_EEG_Microstate_GFP_P3b.py",
            "EEG Microstate GFP P3b",
            "EEG microstate and global field power P3b analysis",
            ["VP18", "EEG-Microstate", "GFP-P3b"],
        ),
        (
            "VP-19",
            "VP_19_InformationErasure_MVPA.py",
            "Information Erasure MVPA",
            "MVPA-based information erasure testing",
            ["VP19", "InformationErasure", "MVPA"],
        ),
        (
            "VP-20",
            "VP_20_Empirical_iEEG.py",
            "Empirical intracranial EEG (APGI-P06 linked)",
            "iEEG bistability and critical slowing; linked to APGI-P06",
            ["VP20", "iEEG", "EmpiricalIEEG"],
        ),
        (
            "VP-21",
            "VP_21_FreeEnergy_PredictionError.py",
            "Free Energy Prediction Error",
            "Variational free energy and prediction error validation",
            ["VP21", "FreeEnergy", "PredictionError"],
        ),
    ]

    for canon_id, filename, title, description, aliases in _vp_entries:
        PROTOCOL_REGISTRY.register_protocol(
            ProtocolInfo(
                canonical_id=canon_id,
                filename=f"Validation/{filename}",
                title=title,
                description=description,
                priority_level="Validation",
                category="Validation",
                aliases=aliases,
            )
        )

    # ------------------------------------------------------------------
    # Theory Modules (Theory/ folder)
    # ------------------------------------------------------------------
    _theory_entries = [
        (
            "T-BayesianEstimation",
            "APGI_Bayesian_Estimation_Framework.py",
            "Bayesian Parameter Estimation Framework",
            "Hierarchical Bayesian modeling, model comparison, parameter recovery",
            ["BayesianFramework", "APGI_Bayesian"],
        ),
        (
            "T-Benchmarking",
            "APGI_Computational_Benchmarking.py",
            "Computational Benchmarking",
            "Compare APGI/FEP/GNW/IIT computational performance",
            ["ComputationalBenchmark", "APGI_Benchmarking"],
        ),
        (
            "T-CrossSpeciesScaling",
            "APGI_Cross_Species_Scaling.py",
            "Cross-Species Scaling Laws",
            "Allostatic scaling laws across human, macaque, mouse, fish",
            ["CrossSpeciesTheory", "APGI_CrossSpecies"],
        ),
        (
            "T-CulturalNeuroscience",
            "APGI_Cultural_Neuroscience.py",
            "Cultural Neuroscience Module",
            "Linguistic and contemplative effects on APGI parameters",
            ["CulturalNeuroscienceTheory", "APGI_Cultural"],
        ),
        (
            "T-Entropy",
            "APGI_Entropy_Implementation.py",
            "Three-Level Entropy Framework",
            "Thermodynamic, Shannon, and variational entropy",
            ["EntropyTheory", "APGI_Entropy"],
        ),
        (
            "T-FalsificationFramework",
            "APGI_Falsification_Framework.py",
            "Popperian Falsification Framework",
            "Core falsification infrastructure used by FP scripts",
            ["FalsificationTheory", "APGI_Falsification"],
        ),
        (
            "T-FractalThreshold",
            "APGI_Fractal_Threshold_Dynamics.py",
            "Fractal Threshold Dynamics",
            "Fractal dynamics of ignition threshold θₜ",
            ["FractalThreshold", "APGI_Fractal"],
        ),
        (
            "T-HierarchicalTiling",
            "APGI_Hierarchical_Temporal_Tiling.py",
            "Hierarchical Temporal Tiling",
            "Temporal integration across hierarchical levels",
            ["TemporalTiling", "APGI_HierarchicalTiling"],
        ),
        (
            "T-BandwidthInfo",
            "APGI_Information_Theoretic_Bandwidth.py",
            "Information-Theoretic Bandwidth (~40 bits/s)",
            "Level 2 derivation of conscious bandwidth from APGI parameters",
            ["BandwidthInfo", "APGI_Bandwidth"],
        ),
        (
            "T-BifurcationAnalysis",
            "APGI_LNN_Bifurcation_Analysis.py",
            "LNN Bifurcation Analysis (APGI-P06 linked)",
            "Bifurcation analysis of LTC dynamics; linked to APGI-P06",
            ["BifurcationAnalysis", "APGI_Bifurcation"],
        ),
        (
            "T-LiquidNetwork",
            "APGI_Liquid_Network_Implementation.py",
            "Liquid Time-Constant Network Implementation",
            "LTC neurons for APGI dynamics with volatility estimation",
            ["LiquidNetworkTheory", "APGI_LTC"],
        ),
        (
            "T-MultimodalClassifier",
            "APGI_Multimodal_Classifier.py",
            "Multimodal Classifier",
            "Multimodal stimulus classification with APGI",
            ["MultimodalClassifier", "APGI_Classifier"],
        ),
        (
            "T-MultimodalIntegration",
            "APGI_Multimodal_Integration.py",
            "Multimodal Integration",
            "Cross-modal integration with precision-weighting",
            ["MultimodalIntegration", "APGI_Integration"],
        ),
        (
            "T-Neuromodulation",
            "APGI_Neuromodulatory_Control_System.py",
            "Neuromodulatory Control System",
            "ACh, NE, DA, 5-HT effects on APGI parameters",
            ["Neuromodulation", "APGI_Neuromodulation"],
        ),
        (
            "T-OpenScience",
            "APGI_Open_Science_Framework.py",
            "Open Science Framework",
            "Reproducibility, pre-registration, and open data utilities",
            ["OpenScience", "APGI_OpenScience"],
        ),
        (
            "T-ParameterEstimation",
            "APGI_Parameter_Estimation.py",
            "Parameter Estimation",
            "Point estimation of APGI parameters from data",
            ["ParameterEstimationTheory", "APGI_ParamEst"],
        ),
        (
            "T-PsychologicalStates",
            "APGI_Psychological_States.py",
            "Psychological States (51-state space)",
            "51-state psychological state space inference",
            ["PsychStates", "APGI_PsychStates"],
        ),
        (
            "T-SomaticMarker",
            "APGI_Somatic_Marker_Identifiability.py",
            "Somatic Marker Identifiability (APGI-P03 linked)",
            "M̂(c,a) identifiability; validated by APGI-P03",
            ["SomaticMarkerIdentifiability", "APGI_SomaticMarker"],
        ),
        (
            "T-Thermodynamic",
            "APGI_Thermodynamic_Program_Aggregator.py",
            "Thermodynamic Program Aggregator (Level 1 bridge)",
            "Level 1 bridge: aggregates thermodynamic constraints",
            ["ThermodynamicAggregator", "APGI_Thermodynamic"],
        ),
        (
            "T-TuringMachine",
            "APGI_Turing_Machine.py",
            "APGI Turing Machine",
            "Computational completeness of APGI state transitions",
            ["TuringMachine", "APGI_Turing"],
        ),
    ]

    for canon_id, filename, title, description, aliases in _theory_entries:
        PROTOCOL_REGISTRY.register_protocol(
            ProtocolInfo(
                canonical_id=canon_id,
                filename=f"Theory/{filename}",
                title=title,
                description=description,
                priority_level="Theory",
                category="Theory",
                aliases=aliases,
            )
        )


# Initialize the registry
_initialize_registry()


def get_protocol_registry() -> Type[PROTOCOL_REGISTRY]:
    """Get the protocol registry instance."""
    return PROTOCOL_REGISTRY


# Convenience functions for external use
def resolve_protocol(protocol_id: str, base_dir: Optional[Path] = None) -> Optional[Path]:
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
