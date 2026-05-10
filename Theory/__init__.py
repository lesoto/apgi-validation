"""
APGI Theory Package
==================

Contains theoretical foundations and computational models for APGI theory.

LEVEL DESIGNATION: All outputs are Level 3 (algorithmic/mathematical).
Bridge to Level 2 requires APGI_Information_Theoretic_Bandwidth.
Bridge to Level 1 requires APGI_Thermodynamic_Program_Aggregator.
This script does NOT claim thermodynamic or information-theoretic implications
without explicit bridge invocation.

FALSIFICATION_CRITERIA
----------------------
If the theory package fails to provide consistent theoretical foundations
across modules (interface inconsistency > 10%), or if the computational models
cannot reproduce theoretical predictions (deviation > 15%), then the APGI
theoretical framework claim is falsified. This would indicate that the
theoretical foundations are not internally consistent.
"""

from typing import Any, Dict

from apgi_core.full_model import APGIFullDynamicModel

# Lazy import for Bayesian module (heavy dependencies - PyMC, ArviZ)
# This prevents startup delays when only non-Bayesian modules are needed
_bayesian_cache: Dict[str, Any] = {}


def _get_bayesian_module():
    """Lazy load Bayesian module on first access."""
    if not _bayesian_cache:
        try:
            from .APGI_Bayesian_Estimation_Framework import (
                BAYESIAN_AVAILABLE,
                APGIBayesianModel,
                BayesianValidationFramework,
                IITConvergenceBayesian,
                ModelComparisonFramework,
                ParameterRecoveryAnalysis,
            )

            _bayesian_cache.update(
                {
                    "BAYESIAN_AVAILABLE": BAYESIAN_AVAILABLE,
                    "APGIBayesianModel": APGIBayesianModel,
                    "BayesianValidationFramework": BayesianValidationFramework,
                    "IITConvergenceBayesian": IITConvergenceBayesian,
                    "ModelComparisonFramework": ModelComparisonFramework,
                    "ParameterRecoveryAnalysis": ParameterRecoveryAnalysis,
                }
            )
        except ImportError as e:
            _bayesian_cache.update(
                {
                    "BAYESIAN_AVAILABLE": False,
                    "APGIBayesianModel": None,
                    "BayesianValidationFramework": None,
                    "IITConvergenceBayesian": None,
                    "ModelComparisonFramework": None,
                    "ParameterRecoveryAnalysis": None,
                    "_import_error": str(e),
                }
            )
    return _bayesian_cache


# Provide module-level access to Bayesian components
BAYESIAN_AVAILABLE = property(lambda self: _get_bayesian_module()["BAYESIAN_AVAILABLE"])

from .APGI_Computational_Benchmarking import ComputationalBenchmarking
from .APGI_Cross_Species_Scaling import CrossSpeciesScaling
from .APGI_Cultural_Neuroscience import CulturalParameterModulator
from .APGI_Entropy_Implementation import EnhancedAPGIValidator
from .APGI_Falsification_Framework import PopperianFalsificationFramework
from .APGI_Fractal_Threshold_Dynamics import MarkovianNullModel, NestedOUProcess, SpectralAnalyzer
from .APGI_Hierarchical_Temporal_Tiling import CrossLevelCouplingSimulator, GeometricTilingDerivation
from .APGI_Liquid_Network_Implementation import APGILiquidNetwork
from .APGI_Multimodal_Classifier import APGIBayesianInversion
from .APGI_Multimodal_Integration import APGINormalizer
from .APGI_Open_Science_Framework import OpenScienceValidator
from .APGI_Parameter_Estimation import ParameterIdentifiabilityAnalyzer
from .APGI_Psychological_States import PsychologicalState as PsychState
from .APGI_Thermodynamic_Program_Aggregator import (
    AggregatorConfig,
    AggregatorReport,
    CrossSpeciesScalingValidator,
    DoubleBridgeCalculator,
    LandauerMinimum,
    NeuralMetabolicCost,
    ThermodynamicProgramAggregator,
    run_aggregator,
)
from .APGI_Turing_Machine import APGITuringMachine

__version__ = "1.0.0"
