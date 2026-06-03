"""
Protocol interfaces for validation protocols and theory modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ValidationProtocolInterface(ABC):
    """Base interface for validation protocols (VP_01–VP_21)."""

    PROTOCOL_TIERS: Dict[int, str]

    @abstractmethod
    def run_validation(self, data_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Run the protocol and return results."""

    @abstractmethod
    def check_criteria(self) -> Dict[str, Any]:
        """Return criteria evaluation details for the protocol."""

    def validate(self, data_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Alias for run_validation for interface parity."""
        return self.run_validation(data_path=data_path, **kwargs)

    def run_full_validation(self, data_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Alias for run_validation for interface parity."""
        return self.run_validation(data_path=data_path, **kwargs)

    def run_all_protocols(self, data_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Alias for run_validation for interface parity."""
        return self.run_validation(data_path=data_path, **kwargs)

    def run_protocol(self, data_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Generic protocol entrypoint alias."""
        return self.run_validation(data_path=data_path, **kwargs)

    def run_protocol_main(self, data_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Generic protocol entrypoint alias used by tooling."""
        return self.run_validation(data_path=data_path, **kwargs)

    def run_all(self, data_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Generic protocol entrypoint alias."""
        return self.run_validation(data_path=data_path, **kwargs)

    def run(self, data_path: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Generic protocol entrypoint alias."""
        return self.run_validation(data_path=data_path, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Resolve legacy run_validation_vN aliases without code duplication."""
        if name.startswith("run_validation_v") and name[len("run_validation_v") :].isdigit():
            return self.run_validation
        if name in ("run_validation_alias", "run_validation_alias_compat", "run_all_protocols_alias"):
            return self.run_validation
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")


class TheoryModuleInterface(ABC):
    """Base interface for Theory computational modules.

    Theory modules are computational backends invoked by VP/FP scripts.
    They expose a consistent compute() entry-point and declare their
    falsification criteria so aggregators can collect them uniformly.
    """

    MODULE_LEVEL: str = "Level 3"

    @abstractmethod
    def compute(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the module's primary computation and return results."""

    @abstractmethod
    def get_falsification_criteria(self) -> Dict[str, Any]:
        """Return quantitative falsification thresholds for this module."""

    def as_protocol_result(self, protocol_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Wrap compute() output in a ProtocolResult dict for aggregation."""
        try:
            from utils.protocol_schema import PredictionResult, PredictionStatus, ProtocolResult
        except ImportError:
            return self.compute(**kwargs)

        raw = self.compute(**kwargs)
        criteria = self.get_falsification_criteria()
        named: Dict[str, PredictionResult] = {}
        for key, spec in criteria.items():
            passed = raw.get(key, {}).get("passed", False) if isinstance(raw.get(key), dict) else False
            named[key] = PredictionResult(
                passed=passed,
                threshold=spec.get("threshold"),
                status=PredictionStatus.PASSED if passed else PredictionStatus.NOT_EVALUATED,
                sources=[protocol_id],
                metadata=spec,
            )

        return ProtocolResult(
            protocol_id=protocol_id,
            named_predictions=named,
            completion_percentage=100 if named else 0,
            methodology=self.MODULE_LEVEL,
            metadata={"module_level": self.MODULE_LEVEL},
        ).to_dict()
