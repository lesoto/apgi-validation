"""
APGI Falsification Protocols Package
====================================

Contains protocols for testing and potentially falsifying APGI theory predictions.
"""

import importlib.util
import os
import warnings
from pathlib import Path

# Get the directory path
_dir = Path(__file__).parent

# Protocol 1: Active Inference Agent Testing - mapped to existing FP_1_Falsification_ActiveInferenceAgents_F1F2
try:
    _spec1 = importlib.util.spec_from_file_location(
        "Protocol_1", _dir / "FP_1_Falsification_ActiveInferenceAgents_F1F2.py"
    )
    if _spec1 and _spec1.loader:
        Protocol_1 = importlib.util.module_from_spec(_spec1)
        _spec1.loader.exec_module(Protocol_1)
    else:
        raise ImportError("Could not load Protocol 1")
except Exception as e:
    warnings.warn(f"Failed to load Falsification Protocol 1: {e}")
    Protocol_1 = None

# Protocol 2: Iowa Gambling Task Environment - mapped to existing protocol
# Note: No direct equivalent exists, using FrameworkLevel as placeholder
try:
    _spec2 = importlib.util.spec_from_file_location(
        "Protocol_2", _dir / "FP_3_Falsification_FrameworkLevel_MultiProtocol.py"
    )
    if _spec2 and _spec2.loader:
        Protocol_2 = importlib.util.module_from_spec(_spec2)
        _spec2.loader.exec_module(Protocol_2)
    else:
        raise ImportError("Could not load Protocol 2")
except Exception as e:
    warnings.warn(f"Failed to load Falsification Protocol 2: {e}")
    Protocol_2 = None

# Protocol 3: Agent Comparison Experiment - mapped to existing FP_2_Falsification_AgentComparison_ConvergenceBenchmark
try:
    _spec3 = importlib.util.spec_from_file_location(
        "Protocol_3",
        _dir / "FP_2_Falsification_AgentComparison_ConvergenceBenchmark.py",
    )
    if _spec3 and _spec3.loader:
        Protocol_3 = importlib.util.module_from_spec(_spec3)
        _spec3.loader.exec_module(Protocol_3)
    else:
        raise ImportError("Could not load Protocol 3")
except Exception as e:
    warnings.warn(f"Failed to load Falsification Protocol 3: {e}")
    Protocol_3 = None

# Protocol 4: Phase Transition Analysis - mapped to existing FP_4_Falsification_InformationTheoretic_PhaseTransition
try:
    _spec4 = importlib.util.spec_from_file_location(
        "Protocol_4",
        _dir / "FP_4_Falsification_InformationTheoretic_PhaseTransition.py",
    )
    if _spec4 and _spec4.loader:
        Protocol_4 = importlib.util.module_from_spec(_spec4)
        _spec4.loader.exec_module(Protocol_4)
    else:
        raise ImportError("Could not load Protocol 4")
except Exception as e:
    warnings.warn(f"Failed to load Falsification Protocol 4: {e}")
    Protocol_4 = None

# Protocol 5: Evolutionary APGI Emergence - uses existing Falsification_Protocol_P5
try:
    _spec5 = importlib.util.spec_from_file_location(
        "Protocol_5", _dir / "Falsification_Protocol_P5.py"
    )
    if _spec5 and _spec5.loader:
        Protocol_5 = importlib.util.module_from_spec(_spec5)
        _spec5.loader.exec_module(Protocol_5)
    else:
        raise ImportError("Could not load Protocol 5")
except Exception as e:
    warnings.warn(f"Failed to load Falsification Protocol 5: {e}")
    Protocol_5 = None

# Protocol 6: Network Comparison Experiment - uses existing Falsification_Protocol_P6
try:
    _spec6 = importlib.util.spec_from_file_location(
        "Protocol_6", _dir / "Falsification_Protocol_P6.py"
    )
    if _spec6 and _spec6.loader:
        Protocol_6 = importlib.util.module_from_spec(_spec6)
        _spec6.loader.exec_module(Protocol_6)
    else:
        raise ImportError("Could not load Protocol 6")
except Exception as e:
    warnings.warn(f"Failed to load Falsification Protocol 6: {e}")
    Protocol_6 = None

# Framework-Level Aggregator
try:
    _spec_aggregator = importlib.util.spec_from_file_location(
        "FP_12_Falsification_Aggregator", _dir / "FP_12_Falsification_Aggregator.py"
    )
    if _spec_aggregator and _spec_aggregator.loader:
        FP_12_Falsification_Aggregator = importlib.util.module_from_spec(
            _spec_aggregator
        )
        _spec_aggregator.loader.exec_module(FP_12_Falsification_Aggregator)
    else:
        raise ImportError("Could not load FP_12_Falsification_Aggregator")
except Exception as e:
    warnings.warn(f"Failed to load FP_12_Falsification_Aggregator: {e}")
    FP_12_Falsification_Aggregator = None

# GUI Components - Not loaded at import to avoid tkinter side effects
ProtocolRunnerGUI = None

# Protocol 1 exports
if Protocol_1:
    try:
        HierarchicalGenerativeModel = Protocol_1.HierarchicalGenerativeModel
        SomaticMarkerNetwork = Protocol_1.SomaticMarkerNetwork
        PolicyNetwork = Protocol_1.PolicyNetwork
        HabitualPolicy = Protocol_1.HabitualPolicy
        EpisodicMemory = Protocol_1.EpisodicMemory
        WorkingMemory = Protocol_1.WorkingMemory
        APGIActiveInferenceAgent = Protocol_1.APGIActiveInferenceAgent
        StandardPPAgent_P1 = Protocol_1.StandardPPAgent
        GWTOnlyAgent_P1 = Protocol_1.GWTOnlyAgent
        run_falsification_protocol_1 = Protocol_1.run_falsification
    except AttributeError as e:
        warnings.warn(f"Failed to load Protocol 1 exports: {e}")
    else:
        # Set defaults to None
        HierarchicalGenerativeModel = SomaticMarkerNetwork = PolicyNetwork = None
        HabitualPolicy = EpisodicMemory = WorkingMemory = None
        APGIActiveInferenceAgent = StandardPPAgent_P1 = GWTOnlyAgent_P1 = None
        run_falsification_protocol_1 = None
else:
    HierarchicalGenerativeModel = SomaticMarkerNetwork = PolicyNetwork = None
    HabitualPolicy = EpisodicMemory = WorkingMemory = None
    APGIActiveInferenceAgent = StandardPPAgent_P1 = GWTOnlyAgent_P1 = None
    run_falsification_protocol_1 = None

# Protocol 2 exports
if Protocol_2:
    try:
        IowaGamblingTaskEnvironment = Protocol_2.IowaGamblingTaskEnvironment
        VolatileForagingEnvironment = Protocol_2.VolatileForagingEnvironment
        ThreatRewardTradeoffEnvironment = Protocol_2.ThreatRewardTradeoffEnvironment
        run_falsification_protocol_2 = Protocol_2.run_falsification
    except AttributeError as e:
        warnings.warn(f"Failed to load Protocol 2 exports: {e}")
        IowaGamblingTaskEnvironment = VolatileForagingEnvironment = None
        ThreatRewardTradeoffEnvironment = run_falsification_protocol_2 = None
else:
    IowaGamblingTaskEnvironment = VolatileForagingEnvironment = None
    ThreatRewardTradeoffEnvironment = run_falsification_protocol_2 = None

# Protocol 3 exports
if Protocol_3:
    try:
        StandardPPAgent_P3 = Protocol_3.StandardPPAgent
        GWTOnlyAgent_P3 = Protocol_3.GWTOnlyAgent
        StandardActorCriticAgent = Protocol_3.StandardActorCriticAgent
        AgentComparisonExperiment = Protocol_3.AgentComparisonExperiment
        run_falsification_protocol_3 = Protocol_3.run_falsification
    except AttributeError as e:
        warnings.warn(f"Failed to load Protocol 3 exports: {e}")
        StandardPPAgent_P3 = GWTOnlyAgent_P3 = StandardActorCriticAgent = None
        AgentComparisonExperiment = run_falsification_protocol_3 = None
else:
    StandardPPAgent_P3 = GWTOnlyAgent_P3 = StandardActorCriticAgent = None
    AgentComparisonExperiment = run_falsification_protocol_3 = None

# Protocol 4 exports
if Protocol_4:
    try:
        SurpriseIgnitionSystem = Protocol_4.SurpriseIgnitionSystem
        InformationTheoreticAnalysis = Protocol_4.InformationTheoreticAnalysis
        run_falsification_protocol_4 = Protocol_4.run_falsification
    except AttributeError as e:
        warnings.warn(f"Failed to load Protocol 4 exports: {e}")
        SurpriseIgnitionSystem = InformationTheoreticAnalysis = None
        run_falsification_protocol_4 = None
else:
    SurpriseIgnitionSystem = InformationTheoreticAnalysis = None
    run_falsification_protocol_4 = None

# Protocol 5 exports
if Protocol_5:
    try:
        EvolvableAgent = Protocol_5.EvolvableAgent
        EvolutionaryAPGIEmergence = Protocol_5.EvolutionaryAPGIEmergence
        run_falsification_protocol_5 = Protocol_5.run_falsification
    except AttributeError as e:
        warnings.warn(f"Failed to load Protocol 5 exports: {e}")
        EvolvableAgent = EvolutionaryAPGIEmergence = run_falsification_protocol_5 = None
else:
    EvolvableAgent = EvolutionaryAPGIEmergence = run_falsification_protocol_5 = None

# Protocol 6 exports
if Protocol_6:
    try:
        APGIInspiredNetwork = Protocol_6.APGIInspiredNetwork
        ComparisonNetworks = Protocol_6.ComparisonNetworks
        NetworkComparisonExperiment = Protocol_6.NetworkComparisonExperiment
        run_falsification_protocol_6 = Protocol_6.run_falsification
    except AttributeError as e:
        warnings.warn(f"Failed to load Protocol 6 exports: {e}")
        APGIInspiredNetwork = ComparisonNetworks = NetworkComparisonExperiment = None
        run_falsification_protocol_6 = None
else:
    APGIInspiredNetwork = ComparisonNetworks = NetworkComparisonExperiment = None
    run_falsification_protocol_6 = None

# Version info
__version__ = "1.0.0"

# Export all main classes for easy access
__all__ = [
    # Protocols
    "Protocol_1",
    "Protocol_2",
    "Protocol_3",
    "Protocol_4",
    "Protocol_5",
    "Protocol_6",
    # Framework-Level Aggregator
    "FP_12_Falsification_Aggregator",
    # Protocol 1
    "HierarchicalGenerativeModel",
    "SomaticMarkerNetwork",
    "PolicyNetwork",
    "HabitualPolicy",
    "EpisodicMemory",
    "WorkingMemory",
    "APGIActiveInferenceAgent",
    "StandardPPAgent_P1",
    "GWTOnlyAgent_P1",
    "run_falsification_protocol_1",
    # Protocol 2
    "IowaGamblingTaskEnvironment",
    "VolatileForagingEnvironment",
    "ThreatRewardTradeoffEnvironment",
    "run_falsification_protocol_2",
    # Protocol 3
    "StandardPPAgent_P3",
    "GWTOnlyAgent_P3",
    "StandardActorCriticAgent",
    "AgentComparisonExperiment",
    "run_falsification_protocol_3",
    # Protocol 4
    "SurpriseIgnitionSystem",
    "InformationTheoreticAnalysis",
    "run_falsification_protocol_4",
    # Protocol 5
    "EvolvableAgent",
    "EvolutionaryAPGIEmergence",
    "run_falsification_protocol_5",
    # Protocol 6
    "APGIInspiredNetwork",
    "ComparisonNetworks",
    "NetworkComparisonExperiment",
    "run_falsification_protocol_6",
    # GUI
    "ProtocolRunnerGUI",
]
