"""
APGI Falsification Protocols Package
====================================

Contains protocols for testing and potentially falsifying APGI theory predictions.
"""

import importlib.util
import os
from pathlib import Path

# Get the directory path
_dir = Path(__file__).parent

# Protocol 1: Active Inference Agent Testing
_spec1 = importlib.util.spec_from_file_location("Protocol_1", _dir / "Falsification-Protocol-1.py")
Protocol_1 = importlib.util.module_from_spec(_spec1)
_spec1.loader.exec_module(Protocol_1)

# Protocol 2: Iowa Gambling Task Environment
_spec2 = importlib.util.spec_from_file_location("Protocol_2", _dir / "Falsification-Protocol-2.py")
Protocol_2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(Protocol_2)

# Protocol 3: Agent Comparison Experiment
_spec3 = importlib.util.spec_from_file_location("Protocol_3", _dir / "Falsification-Protocol-3.py")
Protocol_3 = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(Protocol_3)

# Protocol 4: Phase Transition Analysis
_spec4 = importlib.util.spec_from_file_location("Protocol_4", _dir / "Falsification-Protocol-4.py")
Protocol_4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(Protocol_4)

# Protocol 5: Evolutionary APGI Emergence
_spec5 = importlib.util.spec_from_file_location("Protocol_5", _dir / "Falsification-Protocol-5.py")
Protocol_5 = importlib.util.module_from_spec(_spec5)
_spec5.loader.exec_module(Protocol_5)

# Protocol 6: Network Comparison Experiment
_spec6 = importlib.util.spec_from_file_location("Protocol_6", _dir / "Falsification-Protocol-6.py")
Protocol_6 = importlib.util.module_from_spec(_spec6)
_spec6.loader.exec_module(Protocol_6)

# GUI Components
_gui_spec = importlib.util.spec_from_file_location(
    "protocol_gui", _dir / "APGI-Falsification-Protocol-GUI.py"
)
protocol_gui = importlib.util.module_from_spec(_gui_spec)
_gui_spec.loader.exec_module(protocol_gui)
ProtocolRunnerGUI = protocol_gui.ProtocolRunnerGUI

# Protocol 1 exports
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

# Protocol 2 exports
IowaGamblingTaskEnvironment = Protocol_2.IowaGamblingTaskEnvironment
VolatileForagingEnvironment = Protocol_2.VolatileForagingEnvironment
ThreatRewardTradeoffEnvironment = Protocol_2.ThreatRewardTradeoffEnvironment
run_falsification_protocol_2 = Protocol_2.run_falsification

# Protocol 3 exports
StandardPPAgent_P3 = Protocol_3.StandardPPAgent
GWTOnlyAgent_P3 = Protocol_3.GWTOnlyAgent
StandardActorCriticAgent = Protocol_3.StandardActorCriticAgent
AgentComparisonExperiment = Protocol_3.AgentComparisonExperiment
run_falsification_protocol_3 = Protocol_3.run_falsification

# Protocol 4 exports
SurpriseIgnitionSystem = Protocol_4.SurpriseIgnitionSystem
InformationTheoreticAnalysis = Protocol_4.InformationTheoreticAnalysis
run_falsification_protocol_4 = Protocol_4.run_falsification

# Protocol 5 exports
EvolvableAgent = Protocol_5.EvolvableAgent
EvolutionaryAPGIEmergence = Protocol_5.EvolutionaryAPGIEmergence
run_falsification_protocol_5 = Protocol_5.run_falsification

# Protocol 6 exports
APGIInspiredNetwork = Protocol_6.APGIInspiredNetwork
ComparisonNetworks = Protocol_6.ComparisonNetworks
NetworkComparisonExperiment = Protocol_6.NetworkComparisonExperiment
run_falsification_protocol_6 = Protocol_6.run_falsification

# Version info
__version__ = "1.0.0"

# Export all main classes for easy access
__all__ = [
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
