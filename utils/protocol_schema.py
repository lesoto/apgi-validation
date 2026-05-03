"""Standardized protocol result schema for unified aggregation across all FP/VP protocols.

This module defines the canonical data structures that ALL falsification and validation
protocols MUST return. Enforces consistent JSON structure regardless of protocol type or
implementation approach using Pydantic for validation.

Classes:
    PredictionStatus: Enum for prediction evaluation state
    PredictionResult: Individual prediction with evidence and threshold
    ProtocolResult: Complete protocol execution result (standardized schema)
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class PredictionStatus(str, Enum):
    """Prediction evaluation status - canonical states."""

    PASSED = "passed"
    FAILED = "failed"
    MISSING_PROTOCOL = "missing_protocol"
    LOAD_ERROR = "load_error"
    DATA_UNAVAILABLE = "data_unavailable"
    NOT_EVALUATED = "not_evaluated"
    PARTIAL = "partial"  # Passed with caveats (e.g., some conditions met)


class PredictionResult(BaseModel):
    """Individual prediction result with threshold and evidence.

    Attributes:
        passed: Whether prediction passed (True/False)
        value: Observed/computed value (float or string for non-numeric thresholds)
        threshold: Expected threshold value or description
        status: Evaluation status (from PredictionStatus enum)
        name: Optional name/identifier for this prediction
        evidence: List of supporting evidence strings
        sources: List of source protocols or data files
        metadata: Additional context
    """

    model_config = ConfigDict(populate_by_name=True)

    passed: bool
    value: Optional[Union[float, str]] = None
    threshold: Optional[Union[float, str]] = None
    status: PredictionStatus = PredictionStatus.NOT_EVALUATED
    name: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return self.model_dump()

    def __repr__(self) -> str:
        status_str = "✓" if self.passed else "✗"
        return f"PredictionResult({status_str} {self.status.value}, value={self.value}, threshold={self.threshold})"


class ProtocolResult(BaseModel):
    """Standardized protocol result schema - CANONICAL OUTPUT FOR ALL PROTOCOLS.

    ALL falsification (FP_01-FP_12) and validation (VP_01-VP_15) protocols MUST
    return an instance of this class (or dict representation) to enable aggregation.
    """

    model_config = ConfigDict(populate_by_name=True)

    protocol_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    named_predictions: Dict[str, PredictionResult]
    completion_percentage: int = Field(ge=0, le=100)
    status: str = Field(
        default="success",
        pattern=r"^(success|failed|error|partial)$",
        description="Protocol execution status: success, failed, error, or partial",
    )
    data_sources: List[str] = Field(default_factory=list)
    methodology: str = "unknown"
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary (JSON-serializable) representation."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "ProtocolResult":
        """Load from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_legacy_format(
        cls, protocol_name: str, legacy_result: dict
    ) -> "ProtocolResult":
        """Convert old-format results to standardized schema."""
        # Detect and extract named_predictions from various legacy formats
        named_preds_raw = {}

        if isinstance(legacy_result, dict):
            if "named_predictions" in legacy_result:
                named_preds_raw = legacy_result["named_predictions"]
            elif "results" in legacy_result and isinstance(
                legacy_result["results"], dict
            ):
                if "named_predictions" in legacy_result["results"]:
                    named_preds_raw = legacy_result["results"]["named_predictions"]
                else:
                    named_preds_raw = legacy_result["results"]
            elif any(
                k.startswith(("P", "F", "fp")) and len(k) > 1
                for k in legacy_result.keys()
            ):
                named_preds_raw = legacy_result

        # Convert raw predictions to PredictionResult objects
        named_preds = {}
        for pred_id, pred_value in named_preds_raw.items():
            if isinstance(pred_value, PredictionResult):
                named_preds[pred_id] = pred_value
            elif isinstance(pred_value, bool):
                named_preds[pred_id] = PredictionResult(passed=pred_value)
            elif isinstance(pred_value, (int, float)):
                named_preds[pred_id] = PredictionResult(
                    passed=bool(pred_value), value=pred_value
                )
            elif isinstance(pred_value, dict):
                named_preds[pred_id] = PredictionResult(**pred_value)
            else:
                named_preds[pred_id] = PredictionResult(passed=bool(pred_value))

        # Build metadata from legacy result, including top-level fields
        metadata = legacy_result.get("metadata", {}).copy()
        # Copy legacy top-level fields into metadata if not already present
        for field in ["status", "message", "passed", "results"]:
            if field in legacy_result and field not in metadata:
                metadata[field] = legacy_result[field]

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
            metadata=metadata,
        )

    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """Save protocol result to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> "ProtocolResult":
        """Load protocol result from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            return cls.model_validate_json(f.read())

    def __repr__(self) -> str:
        pass_count = sum(1 for p in self.named_predictions.values() if p.passed)
        total_count = len(self.named_predictions)
        return (
            f"ProtocolResult(id={self.protocol_id}, "
            f"predictions={pass_count}/{total_count}, "
            f"completion={self.completion_percentage}%)"
        )
