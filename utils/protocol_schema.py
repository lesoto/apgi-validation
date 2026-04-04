"""Standardized protocol result schema for unified aggregation across all FP/VP protocols.

This module defines the canonical data structures that ALL falsification and validation
protocols MUST return. Enforces consistent JSON structure regardless of protocol type or
implementation approach.

Classes:
    PredictionStatus: Enum for prediction evaluation state
    PredictionResult: Individual prediction with evidence and threshold
    ProtocolResult: Complete protocol execution result (standardized schema)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
from pathlib import Path
import json


class PredictionStatus(str, Enum):
    """Prediction evaluation status - canonical states."""

    PASSED = "passed"
    FAILED = "failed"
    MISSING_PROTOCOL = "missing_protocol"
    LOAD_ERROR = "load_error"
    DATA_UNAVAILABLE = "data_unavailable"
    NOT_EVALUATED = "not_evaluated"
    PARTIAL = "partial"  # Passed with caveats (e.g., some conditions met)


@dataclass
class PredictionResult:
    """Individual prediction result with threshold and evidence.

    Attributes:
        passed: Whether prediction passed (True/False)
        value: Observed/computed value (float or string for non-numeric thresholds)
        threshold: Expected threshold value or description
        status: Evaluation status (from PredictionStatus enum)
        evidence: List of supporting evidence strings (data points, effect sizes, etc.)
        sources: List of source protocols or data files contributing to this result
        metadata: Additional context (sample size, effect size, confidence intervals, etc.)
    """

    passed: bool
    value: Optional[Union[float, str]] = None
    threshold: Optional[Union[float, str]] = None
    status: PredictionStatus = PredictionStatus.NOT_EVALUATED
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "passed": self.passed,
            "value": self.value,
            "threshold": self.threshold,
            "status": self.status.value,
            "evidence": self.evidence,
            "sources": self.sources,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        status_str = "✓" if self.passed else "✗"
        return f"PredictionResult({status_str} {self.status.value}, value={self.value}, threshold={self.threshold})"


@dataclass
class ProtocolResult:
    """Standardized protocol result schema - CANONICAL OUTPUT FOR ALL PROTOCOLS.

    ALL falsification (FP_01-FP_12) and validation (VP_01-VP_15) protocols MUST
    return an instance of this class (or dict representation) to enable aggregation,
    framework falsification evaluation, and cross-protocol consistency checks.

    Attributes:
        protocol_id: Unique identifier ("FP_01_ActiveInference", "VP_03_AgentModeling", etc.)
        timestamp: ISO 8601 datetime of protocol execution
        named_predictions: Dict mapping prediction IDs to PredictionResult objects
                          Keys: "P1.1", "P1.2", "P2.a", "P3.b", "F1.1", etc.
        completion_percentage: Fraction of protocol implementation complete (0-100)
        data_sources: List of datasets/simulations used (["Iowa Gambling Task", "n=50 agents"])
        methodology: Approach used (agent_simulation, clinical_data, synthetic_data, etc.)
        errors: List of errors encountered (empty if clean run)
        metadata: Additional protocol-specific context (sample size, effect sizes, etc.)
    """

    protocol_id: str
    timestamp: str
    named_predictions: Dict[str, PredictionResult]
    completion_percentage: int
    data_sources: List[str]
    methodology: str
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary (JSON-serializable) representation."""
        return {
            "protocol_id": self.protocol_id,
            "timestamp": self.timestamp,
            "named_predictions": {
                pred_id: (
                    pred.to_dict() if isinstance(pred, PredictionResult) else pred
                )
                for pred_id, pred in self.named_predictions.items()
            },
            "completion_percentage": self.completion_percentage,
            "data_sources": self.data_sources,
            "methodology": self.methodology,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProtocolResult":
        """Load from dictionary, converting nested predictions to PredictionResult objects.

        Args:
            data: Dictionary with protocol result structure

        Returns:
            ProtocolResult instance
        """
        named_preds = {}
        for pred_id, pred_data in data.get("named_predictions", {}).items():
            if isinstance(pred_data, PredictionResult):
                named_preds[pred_id] = pred_data
            elif isinstance(pred_data, dict):
                named_preds[pred_id] = PredictionResult(
                    passed=pred_data.get("passed", False),
                    value=pred_data.get("value"),
                    threshold=pred_data.get("threshold"),
                    status=PredictionStatus(pred_data.get("status", "not_evaluated")),
                    evidence=pred_data.get("evidence", []),
                    sources=pred_data.get("sources", []),
                    metadata=pred_data.get("metadata", {}),
                )
            else:
                # If not dict/PredictionResult, treat as boolean pass/fail
                named_preds[pred_id] = PredictionResult(passed=bool(pred_data))

        return cls(
            protocol_id=data.get("protocol_id", "UNKNOWN"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            named_predictions=named_preds,
            completion_percentage=data.get("completion_percentage", 0),
            data_sources=data.get("data_sources", []),
            methodology=data.get("methodology", "unknown"),
            errors=data.get("errors", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_legacy_format(
        cls, protocol_name: str, legacy_result: dict
    ) -> "ProtocolResult":
        """Convert old-format results to standardized schema.

        Detects and converts from three legacy formats:
        1. {"named_predictions": {...}}
        2. {"results": {"named_predictions": {...}}}
        3. Direct predictions dict with prediction IDs as keys

        Args:
            protocol_name: Name of protocol (e.g., "FP_01_ActiveInference")
            legacy_result: Result dict in old format

        Returns:
            ProtocolResult instance in standardized schema
        """
        # Detect and extract named_predictions from various legacy formats
        named_preds_raw = None

        if isinstance(legacy_result, dict):
            # Format 1: Direct top-level "named_predictions"
            if "named_predictions" in legacy_result:
                named_preds_raw = legacy_result["named_predictions"]

            # Format 2: Nested under "results"
            elif "results" in legacy_result and isinstance(
                legacy_result["results"], dict
            ):
                if "named_predictions" in legacy_result["results"]:
                    named_preds_raw = legacy_result["results"]["named_predictions"]
                else:
                    named_preds_raw = legacy_result["results"]

            # Format 3: Assume entire dict is predictions (heuristic detection)
            elif any(
                k.startswith("P") and len(k) > 1 and k[1].isdigit()
                for k in legacy_result.keys()
            ):
                # P1.1, P1.2, P2.a style prediction IDs
                named_preds_raw = legacy_result
            elif any(
                k.startswith("F") and len(k) > 1 and k[1].isdigit()
                for k in legacy_result.keys()
            ):
                # F1.1, F2.a style prediction IDs
                named_preds_raw = legacy_result
            elif any(k.startswith("fp") and len(k) > 2 for k in legacy_result.keys()):
                # fp10a, fp10b style prediction IDs
                named_preds_raw = legacy_result
            else:
                named_preds_raw = {}

        # Convert raw predictions to PredictionResult objects
        named_preds = {}
        if isinstance(named_preds_raw, dict):
            for pred_id, pred_value in named_preds_raw.items():
                if isinstance(pred_value, PredictionResult):
                    # Already correct type
                    named_preds[pred_id] = pred_value
                elif isinstance(pred_value, bool):
                    # Simple boolean result
                    named_preds[pred_id] = PredictionResult(passed=pred_value)
                elif isinstance(pred_value, (int, float)):
                    # Numeric value - interpret as pass if > 0, fail if <= 0
                    named_preds[pred_id] = PredictionResult(
                        passed=bool(pred_value), value=pred_value
                    )
                elif isinstance(pred_value, dict):
                    # Complex dict - extract all available fields
                    named_preds[pred_id] = PredictionResult(
                        passed=pred_value.get("passed", False),
                        value=pred_value.get("value"),
                        threshold=pred_value.get("threshold"),
                        status=PredictionStatus(
                            pred_value.get("status", "not_evaluated")
                        ),
                        evidence=pred_value.get("evidence", []),
                        sources=pred_value.get("sources", []),
                        metadata=pred_value.get("metadata", {}),
                    )
                else:
                    # Unknown type - coerce to boolean
                    named_preds[pred_id] = PredictionResult(passed=bool(pred_value))

        return cls(
            protocol_id=protocol_name,
            timestamp=legacy_result.get("timestamp", datetime.now().isoformat()),
            named_predictions=named_preds,
            completion_percentage=legacy_result.get(
                "completion_percentage", legacy_result.get("completion", 0)
            ),
            data_sources=legacy_result.get(
                "data_sources", legacy_result.get("sources", [])
            ),
            methodology=legacy_result.get("methodology", "unknown"),
            errors=legacy_result.get("errors", []),
            metadata=legacy_result.get("metadata", {}),
        )

    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """Save protocol result to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> "ProtocolResult":
        """Load protocol result from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        pass_count = sum(1 for p in self.named_predictions.values() if p.passed)
        total_count = len(self.named_predictions)
        return (
            f"ProtocolResult(id={self.protocol_id}, "
            f"predictions={pass_count}/{total_count}, "
            f"completion={self.completion_percentage}%)"
        )
