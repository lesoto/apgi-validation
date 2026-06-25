"""
TIER DESIGNATION: Tier 1 (thermodynamic)

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
    APGIConfig,
    APGIModel,
    GenerativeModel,
    HierarchicalLevel,
    HierarchicalProcessor,
    RunningStatsEMA,
    _get_config,
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


def __getattr__(name: str):
    """Lazy/deprecated attribute access for backward compatibility."""
    if name == "CONFIG":
        import warnings

        warnings.warn(
            "apgi_core.CONFIG is deprecated — use APGIConfig.from_settings() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _get_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # --- model ---
    "APGIConfig",
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
