"""
LEVEL DESIGNATION: Level 2 (information-theoretic)

Bridge to Level 1
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROTOCOLS_DIR = Path(__file__).parent.parent / "protocols"


@dataclass
class SubPrediction:
    """A single falsifiable sub-prediction from a protocol JSON."""

    id: str
    claim: str
    confirming_evidence: str = ""
    falsifying_evidence: str = ""
    key_measurement_equation: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubPrediction":
        return cls(
            id=data["id"],
            claim=data.get("claim", ""),
            confirming_evidence=data.get("confirming_evidence", ""),
            falsifying_evidence=data.get("falsifying_evidence", ""),
            key_measurement_equation=data.get("key_measurement_equation", ""),
        )


@dataclass
class ProtocolSpec:
    """Parsed representation of a JSON protocol specification."""

    protocol_id: str
    title: str
    version: str
    paper: str
    description: str
    validation_status: str
    apgi_parameters: Dict[str, Any]
    sub_predictions: List[SubPrediction]
    primary_hypothesis: str
    caveats: List[str]
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtocolSpec":
        return cls(
            protocol_id=data["protocol_id"],
            title=data.get("title", ""),
            version=data.get("version", "1.0"),
            paper=data.get("paper", ""),
            description=data.get("description", ""),
            validation_status=data.get("validation_status", "unspecified"),
            apgi_parameters=data.get("apgi_parameters", {}),
            sub_predictions=[
                SubPrediction.from_dict(p) for p in data.get("sub_predictions", [])
            ],
            primary_hypothesis=data.get("primary_hypothesis", ""),
            caveats=data.get("caveats", []),
            _raw=data,
        )

    def get_parameter(self, name: str, default: Any = None) -> Any:
        return self.apgi_parameters.get(name, default)

    def get_prediction(self, pred_id: str) -> Optional[SubPrediction]:
        for p in self.sub_predictions:
            if p.id == pred_id:
                return p
        return None

    def prediction_ids(self) -> List[str]:
        return [p.id for p in self.sub_predictions]

    def to_dict(self) -> Dict[str, Any]:
        return self._raw


def _find_protocol_file(protocol_id: str) -> Optional[Path]:
    """Resolve an APGI protocol ID (e.g. 'APGI-P01') to its JSON file."""
    if not _PROTOCOLS_DIR.exists():
        return None
    num_str = protocol_id.split("P")[-1]
    try:
        num = int(num_str)
    except ValueError:
        return None
    for path in _PROTOCOLS_DIR.glob("protocol_*.json"):
        stem = path.stem  # e.g. "protocol_1_eeg_interoceptive_gating"
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                if int(parts[1]) == num:
                    return path
            except ValueError:
                continue
    return None


def load_protocol(protocol_id: str) -> Optional[ProtocolSpec]:
    """Load a protocol spec by ID (e.g. 'APGI-P01'). Returns None if not found."""
    path = _find_protocol_file(protocol_id)
    if path is None:
        return None
    return load_protocol_file(path)


def load_protocol_file(path: Path) -> ProtocolSpec:
    """Load a ProtocolSpec from a specific JSON file path."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return ProtocolSpec.from_dict(data)


def load_all_protocols() -> Dict[str, ProtocolSpec]:
    """Load all protocol specs from the protocols/ directory, keyed by protocol_id."""
    specs: Dict[str, ProtocolSpec] = {}
    if not _PROTOCOLS_DIR.exists():
        return specs
    for path in sorted(_PROTOCOLS_DIR.glob("protocol_*.json")):
        try:
            spec = load_protocol_file(path)
            specs[spec.protocol_id] = spec
        except Exception:
            pass
    return specs


def get_apgi_parameters(protocol_id: str) -> Dict[str, Any]:
    """Return the apgi_parameters dict for a protocol, or empty dict if not found."""
    spec = load_protocol(protocol_id)
    return spec.apgi_parameters if spec is not None else {}
