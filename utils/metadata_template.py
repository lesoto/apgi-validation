"""
Standardized Metadata Template for APGI Protocols

This module defines the canonical metadata structure that ALL protocols
should use in their ProtocolResult objects.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StandardizedMetadata:
    """Standardized metadata structure for all APGI protocols.

    All 27 protocols (12 FP + 15 VP) should include these fields
    in their ProtocolResult.metadata for consistency.
    """

    # Data provenance (REQUIRED)
    data_sources: List[
        str
    ]  # e.g., ["Synthetic EEG data", "Iowa Gambling Task simulation"]
    empirical_vs_synthetic: str  # "empirical" | "synthetic" | "hybrid"
    sample_size: Optional[int] = None  # n=50, n=100, etc.

    # Protocol relationships (REQUIRED)
    dependencies: List[str] = field(default_factory=list)  # e.g., ["FP-05", "VP-03"]
    cross_protocol_validation: List[str] = field(default_factory=list)

    # Quality metrics (REQUIRED)
    completion_percentage: int = 0  # 0-100
    test_coverage: Optional[float] = None  # Percentage of tests passing

    # Analysis details (OPTIONAL but recommended)
    analysis_methods: List[str] = field(default_factory=list)
    statistical_tests: List[str] = field(default_factory=list)
    model_parameters: Dict[str, Any] = field(default_factory=dict)

    # Publication tracking (OPTIONAL)
    dataset_doi: Optional[str] = None
    preregistered: bool = False
    replication_status: str = (
        "not_attempted"  # "replicated" | "failed" | "not_attempted"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ProtocolResult.metadata."""
        return {
            "data_provenance": {
                "sources": self.data_sources,
                "type": self.empirical_vs_synthetic,
                "sample_size": self.sample_size,
            },
            "protocol_relationships": {
                "dependencies": self.dependencies,
                "cross_protocol_validation": self.cross_protocol_validation,
            },
            "quality_metrics": {
                "completion_percentage": self.completion_percentage,
                "test_coverage": self.test_coverage,
            },
            "analysis_details": {
                "methods": self.analysis_methods,
                "statistical_tests": self.statistical_tests,
                "model_parameters": self.model_parameters,
            },
            "publication_tracking": {
                "dataset_doi": self.dataset_doi,
                "preregistered": self.preregistered,
                "replication_status": self.replication_status,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardizedMetadata":
        """Create from dictionary (handles partial data)."""
        prov = data.get("data_provenance", {})
        rel = data.get("protocol_relationships", {})
        qual = data.get("quality_metrics", {})
        anal = data.get("analysis_details", {})
        pub = data.get("publication_tracking", {})

        return cls(
            data_sources=prov.get("sources", []),
            empirical_vs_synthetic=prov.get("type", "unknown"),
            sample_size=prov.get("sample_size"),
            dependencies=rel.get("dependencies", []),
            cross_protocol_validation=rel.get("cross_protocol_validation", []),
            completion_percentage=qual.get("completion_percentage", 0),
            test_coverage=qual.get("test_coverage"),
            analysis_methods=anal.get("methods", []),
            statistical_tests=anal.get("statistical_tests", []),
            model_parameters=anal.get("model_parameters", {}),
            dataset_doi=pub.get("dataset_doi"),
            preregistered=pub.get("preregistered", False),
            replication_status=pub.get("replication_status", "not_attempted"),
        )


# Predefined metadata templates for common protocol types

SYNTHETIC_AGENT_SIMULATION = StandardizedMetadata(
    data_sources=["Synthetic agent simulations"],
    empirical_vs_synthetic="synthetic",
    sample_size=100,
    analysis_methods=["agent-based modeling", "active inference"],
)

SYNTHETIC_EEG_DATA = StandardizedMetadata(
    data_sources=["Synthetic EEG data"],
    empirical_vs_synthetic="synthetic",
    sample_size=50,
    analysis_methods=["spectral analysis", "PAC analysis"],
    statistical_tests=["t-test", "ANOVA"],
)

BEHAVIORAL_SIMULATION = StandardizedMetadata(
    data_sources=["Iowa Gambling Task simulation", "Behavioral task simulation"],
    empirical_vs_synthetic="synthetic",
    sample_size=100,
    analysis_methods=["reinforcement learning", "decision modeling"],
)

EVOLUTIONARY_SIMULATION = StandardizedMetadata(
    data_sources=["Evolutionary simulation"],
    empirical_vs_synthetic="synthetic",
    sample_size=1000,
    analysis_methods=["genetic algorithm", "selection pressure modeling"],
    dependencies=["FP-05", "VP-05"],
)

# Empirical templates (for when real data is integrated)
EMPIRICAL_FMRI = StandardizedMetadata(
    data_sources=["fMRI dataset"],
    empirical_vs_synthetic="empirical",
    sample_size=30,
    analysis_methods=["fMRI preprocessing", "HRF convolution", "ROI analysis"],
    preregistered=True,
)

EMPIRICAL_EEG = StandardizedMetadata(
    data_sources=["EEG dataset"],
    empirical_vs_synthetic="empirical",
    sample_size=50,
    analysis_methods=["EEG preprocessing", "ERP analysis", "spectral analysis"],
    preregistered=True,
)


# Default template for test compatibility
DEFAULT_TEMPLATE = {
    "version": "1.0.0",
    "created": None,
    "modified": None,
    "fields": {},
}


class MetadataTemplate:
    """Template for metadata structures."""

    def __init__(self, fields: Optional[Dict[str, Any]] = None):
        self.fields: Dict[str, Any] = fields or {}

    def add_field(self, name: str, field_type: type, required: bool = False) -> None:
        self.fields[name] = {"type": field_type, "required": required}

    def validate(self, data: Dict[str, Any]) -> bool:
        for field_name, field_spec in self.fields.items():
            if field_spec.get("required", False) and field_name not in data:
                return False
            if field_name in data:
                expected_type = field_spec.get("type")
                if expected_type and not isinstance(data[field_name], expected_type):
                    return False
        return True


def create_metadata_template(type: str = "basic") -> MetadataTemplate:
    template = MetadataTemplate()
    if type == "basic":
        template.add_field("name", str)
        template.add_field("timestamp", str)
    elif type == "experimental":
        template.add_field("subject_id", str, required=True)
        template.add_field("session", int)
        template.add_field("condition", str)
    return template


def validate_metadata(
    data: Dict[str, Any], template: MetadataTemplate
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"valid": True, "missing": [], "errors": []}
    for field_name, field_spec in template.fields.items():
        if field_spec.get("required", False) and field_name not in data:
            result["valid"] = False
            result["missing"].append(field_name)
        if field_name in data:
            expected_type = field_spec.get("type")
            if expected_type and not isinstance(data[field_name], expected_type):
                result["valid"] = False
                result["errors"].append(f"Field '{field_name}' has wrong type")
    return result


def save_template(template: MetadataTemplate, filepath: Path) -> None:

    data = {
        "fields": {
            name: {
                "type": spec["type"].__name__,
                "required": spec.get("required", False),
            }
            for name, spec in template.fields.items()
        }
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_template(filepath: Path) -> Optional[MetadataTemplate]:

    if not filepath.exists():
        return None
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        template = MetadataTemplate()
        type_map = {"str": str, "int": int, "float": float, "bool": bool}
        for name, spec in data.get("fields", {}).items():
            field_type = type_map.get(spec.get("type"), str)
            template.add_field(name, field_type, spec.get("required", False))
        return template
    except (json.JSONDecodeError, KeyError):
        return None
