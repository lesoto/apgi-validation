"""
apgi_core — Canonical APGI Domain Package
==========================================

Single source-of-truth for all APGI logic.  All other modules should import
from here; none should import from the deleted ``apgi_implementation.py`` or
``utils/apgi_engine.py`` files.

Public API
----------
From :mod:`apgi_core.model` (high-level dynamical system):

    CONFIG, GenerativeModel, RunningStatsEMA,
    compute_precision, effective_interoceptive_precision,
    compute_signal, compute_information_value, update_threshold,
    ignition_probability, ignite,
    clip, enforce_stability,
    map_to_p3b_latency, map_to_hep_amplitude, map_to_reaction_time,
    HierarchicalLevel, HierarchicalProcessor,
    APGIModel

From :mod:`apgi_core.engine` (equation-level components):

    APGIPreProcessor, APGIPrecisionSystem, APGICoreSignal,
    APGIIgnitionMechanism, APGISystemDynamics, APGIAllostaticLayer,
    APGILiquidNeuralNetwork, APGIHierarchy, APGIRecovery,
    APGIValidationMetrics, APGISystem

From :mod:`apgi_core.equations`:

    verify_all_equations

"""

# ---------------------------------------------------------------------------
# engine.py exports
# ---------------------------------------------------------------------------
from apgi_core.engine import (
    APGIAllostaticLayer,
    APGICoreSignal,
    APGIHierarchy,
    APGIIgnitionMechanism,
    APGILiquidNeuralNetwork,
    APGIPrecisionSystem,
    APGIPreProcessor,
    APGIRecovery,
    APGISystem,
    APGISystemDynamics,
    APGIValidationMetrics,
)

# ---------------------------------------------------------------------------
# equations.py exports
# ---------------------------------------------------------------------------
from apgi_core.equations import verify_all_equations

# ---------------------------------------------------------------------------
# full_model.py exports
# ---------------------------------------------------------------------------
from apgi_core.full_model import APGIFullDynamicModel, APGIParameters, APGIState

# ---------------------------------------------------------------------------
# model.py exports
# ---------------------------------------------------------------------------
from apgi_core.model import (
    CONFIG,
    APGIModel,
    GenerativeModel,
    HierarchicalLevel,
    HierarchicalProcessor,
    RunningStatsEMA,
    clip,
    compute_information_value,
    compute_precision,
    compute_signal,
    effective_interoceptive_precision,
    enforce_stability,
    ignite,
    ignition_probability,
    map_to_hep_amplitude,
    map_to_p3b_latency,
    map_to_reaction_time,
    update_threshold,
)

__all__ = [
    # --- model ---
    "CONFIG",
    "GenerativeModel",
    "RunningStatsEMA",
    "compute_precision",
    "effective_interoceptive_precision",
    "compute_signal",
    "compute_information_value",
    "update_threshold",
    "ignition_probability",
    "ignite",
    "clip",
    "enforce_stability",
    "map_to_p3b_latency",
    "map_to_hep_amplitude",
    "map_to_reaction_time",
    "HierarchicalLevel",
    "HierarchicalProcessor",
    "APGIModel",
    # --- engine ---
    "APGIPreProcessor",
    "APGIPrecisionSystem",
    "APGICoreSignal",
    "APGIIgnitionMechanism",
    "APGISystemDynamics",
    "APGIAllostaticLayer",
    "APGILiquidNeuralNetwork",
    "APGIHierarchy",
    "APGIRecovery",
    "APGIValidationMetrics",
    "APGISystem",
    # --- equations ---
    "verify_all_equations",
    # --- full_model ---
    "APGIFullDynamicModel",
    "APGIParameters",
    "APGIState",
]
