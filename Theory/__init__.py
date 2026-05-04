"""
APGI Theory Package
==================

Contains theoretical foundations and computational models for APGI theory.
"""

from apgi_core.full_model import APGIFullDynamicModel

from .APGI_Bayesian_Estimation_Framework import (
    BAYESIAN_AVAILABLE,
    APGIBayesianModel,
    BayesianValidationFramework,
    IITConvergenceBayesian,
    ModelComparisonFramework,
    ParameterRecoveryAnalysis,
)
from .APGI_Computational_Benchmarking import ComputationalBenchmarking
from .APGI_Cross_Species_Scaling import CrossSpeciesScaling
from .APGI_Cultural_Neuroscience import CulturalParameterModulator
from .APGI_Entropy_Implementation import EnhancedAPGIValidator
from .APGI_Falsification_Framework import PopperianFalsificationFramework
from .APGI_Liquid_Network_Implementation import APGILiquidNetwork
from .APGI_Multimodal_Classifier import APGIBayesianInversion
from .APGI_Multimodal_Integration import APGINormalizer
from .APGI_Open_Science_Framework import OpenScienceValidator
from .APGI_Parameter_Estimation import ParameterIdentifiabilityAnalyzer
from .APGI_Psychological_States import PsychologicalState as PsychState
from .APGI_Turing_Machine import APGITuringMachine

__version__ = "1.0.0"
