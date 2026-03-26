"""
APGI Theory Package
==================

Contains theoretical foundations and computational models for APGI theory.
"""

# Import main classes from each module
from .APGI_Equations import (
    FoundationalEquations,
    CoreIgnitionSystem,
    DynamicalSystemEquations,
    RunningStatistics,
    DerivedQuantities,
    APGIParameters,
    PsychologicalState,
)
from .APGI_Bayesian_Estimation_Framework import (
    APGIBayesianModel,
    ModelComparisonFramework,
    IITConvergenceBayesian,
    ParameterRecoveryAnalysis,
    BayesianValidationFramework,
    BAYESIAN_AVAILABLE,
)
from .APGI_Computational_Benchmarking import ComputationalBenchmarking
from .APGI_Cross_Species_Scaling import CrossSpeciesScaling
from .APGI_Cultural_Neuroscience import CulturalParameterModulator
from .APGI_Entropy_Implementation import EnhancedAPGIValidator
from .APGI_Falsification_Framework import PopperianFalsificationFramework
from .APGI_Full_Dynamic_Model import APGIFullDynamicModel
from .APGI_Liquid_Network_Implementation import APGILiquidNetwork
from .APGI_Multimodal_Classifier import APGIBayesianInversion
from .APGI_Multimodal_Integration import APGINormalizer
from .APGI_Open_Science_Framework import OpenScienceValidator
from .APGI_Parameter_Estimation import ParameterIdentifiabilityAnalyzer
from .APGI_Psychological_States import PsychologicalState
from .APGI_Turing_Machine import APGITuringMachine

__version__ = "1.0.0"
