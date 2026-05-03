"""Empirical Validation Transition Report for VP-11 and VP-15.

Tracks the status of validation protocols transitioning from "Simulation Only"
to "Empirical Validation" using public neuroscience datasets.

Usage:
    python -m utils.empirical_transition_report
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ValidationMode(Enum):
    """Validation mode for protocols."""

    SIMULATION_ONLY = "SIMULATION_ONLY"
    SIMULATION_VALIDATED = "SIMULATION_VALIDATED"
    EMPIRICAL_PENDING = "EMPIRICAL_PENDING"  # Data identified but not yet ingested
    EMPIRICAL_VALIDATED = "EMPIRICAL_VALIDATED"
    HYBRID = "HYBRID"  # Some gates empirical, some simulation


@dataclass
class ProtocolTransitionStatus:
    """Status of a protocol's transition to empirical validation."""

    protocol_id: str
    protocol_name: str
    current_mode: ValidationMode
    target_mode: ValidationMode

    # Dataset mapping
    candidate_datasets: List[str]  # Dataset IDs that could validate this protocol
    available_datasets: List[str]  # Publicly available now
    pending_datasets: List[str]  # Author request / forthcoming

    # Transition readiness
    data_loaders_implemented: bool
    bids_support: bool

    # Gaps
    critical_gaps: List[str]

    # Timeline
    estimated_transition_date: Optional[str] = None
    blockers: List[str] = None

    def __post_init__(self):
        if self.blockers is None:
            self.blockers = []


class EmpiricalTransitionReport:
    """Generate reports on VP-11/VP-15 empirical validation transition."""

    def __init__(self) -> None:  # type: ignore[annotation-unchecked]
        self.protocols: Dict[str, ProtocolTransitionStatus] = {}
        self.report_date = datetime.now().isoformat()

    def analyze_vp11(self) -> ProtocolTransitionStatus:
        """Analyze VP-11 transition status."""
        return ProtocolTransitionStatus(
            protocol_id="VP-11",
            protocol_name="MCMC Cultural Neuroscience / Bayesian Estimation",
            current_mode=ValidationMode.SIMULATION_VALIDATED,
            target_mode=ValidationMode.EMPIRICAL_VALIDATED,
            candidate_datasets=["DS-01", "DS-02", "DS-15", "DS-12"],
            available_datasets=["DS-15", "DS-12"],  # Fully public
            pending_datasets=["DS-01", "DS-02"],  # Author request
            data_loaders_implemented=True,
            bids_support=True,
            critical_gaps=[
                "DS-01 (Sergent 2005): Author request required. Gold-standard for I-04/I-15.",
                "DS-02 (Melloni 2007): Author request required. Gamma synchrony test.",
            ],
            estimated_transition_date="2026-Q3",
            blockers=[
                "Contact authors for DS-01, DS-02 data access",
                "Implement THINGS-Data RSVP trial extraction for perceptual thresholds",
            ],
        )

    def analyze_vp15(self) -> ProtocolTransitionStatus:
        """Analyze VP-15 transition status."""
        return ProtocolTransitionStatus(
            protocol_id="VP-15",
            protocol_name="fMRI vmPFC Anticipation Paradigm",
            current_mode=ValidationMode.SIMULATION_VALIDATED,
            target_mode=ValidationMode.HYBRID,  # May need simulation for some gates
            candidate_datasets=["DS-07", "DS-09", "DS-10", "DS-11", "DS-16"],
            available_datasets=["DS-07", "DS-09", "DS-11"],  # Fully public
            pending_datasets=["DS-10", "DS-16"],  # Institutional / forthcoming
            data_loaders_implemented=True,
            bids_support=True,
            critical_gaps=[
                "GAP-1: No public dataset combines vmPFC BOLD + anterior insula + anticipatory paradigm.",
                "DS-09 (iEEG) lacks fMRI; DS-07 (fMRI) lacks insula-specific coverage.",
                "DS-16 (Cogitate fMRI, N=122) forthcoming - will provide highest-value validation.",
            ],
            estimated_transition_date="2026-Q4",
            blockers=[
                "DS-16 fMRI public release pending (expected 2026)",
                "Anticipatory paradigm not present in any available public dataset",
                "May need bespoke data collection for full V15.1-V15.3 validation",
            ],
        )

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive transition report."""
        vp11 = self.analyze_vp11()
        vp15 = self.analyze_vp15()

        total_public = len(vp11.available_datasets) + len(vp15.available_datasets)
        total_pending = len(vp11.pending_datasets) + len(vp15.pending_datasets)

        # Convert ValidationMode enums to strings for JSON serialization
        vp11_dict = asdict(vp11)
        vp11_dict["current_mode"] = vp11.current_mode.value
        vp11_dict["target_mode"] = vp11.target_mode.value

        vp15_dict = asdict(vp15)
        vp15_dict["current_mode"] = vp15.current_mode.value
        vp15_dict["target_mode"] = vp15.target_mode.value

        return {
            "report_title": "VP-11/VP-15 Empirical Validation Transition Report",
            "report_date": self.report_date,
            "catalogue_source": "PUBLIC DATASET CATALOGUE (Apr 22, 2026)",
            "summary": {
                "protocols_in_transition": 2,
                "total_public_datasets_available": total_public,
                "total_pending_datasets": total_pending,
                "estimated_completion": "2026-Q4",
            },
            "protocols": {
                "VP-11": vp11_dict,
                "VP-15": vp15_dict,
            },
            "critical_path": {
                "immediate_actions": [
                    "Download DS-15 (THINGS-Data) for VP-11 RSVP validation",
                    "Download DS-12 (OpenNeuro) for VP-11 depression stratification",
                    "Download DS-07 (Carhart-Harris) for VP-15 DMN connectivity",
                    "Download DS-09 (Cogitate) for VP-15 sustained ignition",
                ],
                "medium_term": [
                    "Contact Unicog lab for DS-01 (Sergent 2005) access",
                    "Contact Melloni lab for DS-02 (Melloni 2007) access",
                    "Monitor DS-16 (Cogitate fMRI) release status",
                ],
                "long_term": [
                    "Consider bespoke data collection for Joint HEP x PCI (Gap 1)",
                    "Consider bespoke data collection for anticipatory paradigm (Gap 2)",
                ],
            },
            "dataset_summary": {
                "immediately_available": [
                    {
                        "id": "DS-07",
                        "name": "Carhart-Harris Psychedelic fMRI",
                        "protocols": ["VP-15"],
                    },
                    {"id": "DS-09", "name": "Cogitate iEEG", "protocols": ["VP-15"]},
                    {"id": "DS-11", "name": "HCP-EP", "protocols": ["VP-15"]},
                    {
                        "id": "DS-12",
                        "name": "OpenNeuro EEG Depression",
                        "protocols": ["VP-11"],
                    },
                    {"id": "DS-15", "name": "THINGS-Data", "protocols": ["VP-11"]},
                ],
                "requires_access": [
                    {
                        "id": "DS-01",
                        "name": "Sergent 2005",
                        "access": "Author request",
                        "protocols": ["VP-11"],
                    },
                    {
                        "id": "DS-02",
                        "name": "Melloni 2007",
                        "access": "Author request",
                        "protocols": ["VP-11"],
                    },
                    {
                        "id": "DS-10",
                        "name": "Drysdale Biotypes",
                        "access": "Institutional DUA",
                        "protocols": ["VP-15"],
                    },
                ],
                "forthcoming": [
                    {
                        "id": "DS-16",
                        "name": "Cogitate fMRI/MEG",
                        "access": "Expected 2026",
                        "protocols": ["VP-11", "VP-15"],
                    },
                ],
            },
        }

    def print_report(self) -> None:
        """Print formatted report to console."""
        report = self.generate_report()

        print("=" * 80)
        print(report["report_title"])
        print(f"Generated: {report['report_date']}")
        print(f"Source: {report['catalogue_source']}")
        print("=" * 80)
        print()

        # Summary
        print("EXECUTIVE SUMMARY")
        print("-" * 40)
        summary = report["summary"]
        print(f"Protocols in Transition: {summary['protocols_in_transition']}")
        print(
            f"Public Datasets Available: {summary['total_public_datasets_available']}"
        )
        print(f"Pending Datasets: {summary['total_pending_datasets']}")
        print(f"Estimated Completion: {summary['estimated_completion']}")
        print()

        # Protocol details
        for proto_id, proto in report["protocols"].items():
            print(f"\n{proto_id}: {proto['protocol_name']}")
            print("-" * 60)
            print(f"  Current Mode: {proto['current_mode']}")
            print(f"  Target Mode:  {proto['target_mode']}")
            print(f"  Public Datasets: {', '.join(proto['available_datasets'])}")
            print(f"  Pending Datasets: {', '.join(proto['pending_datasets'])}")
            print(
                f"  Data Loaders: {'✓' if proto['data_loaders_implemented'] else '✗'}"
            )
            print(f"  BIDS Support: {'✓' if proto['bids_support'] else '✗'}")

            if proto["critical_gaps"]:
                print("\n  Critical Gaps:")
                for gap in proto["critical_gaps"]:
                    print(f"    • {gap}")

            if proto["blockers"]:
                print("\n  Blockers:")
                for blocker in proto["blockers"]:
                    print(f"    → {blocker}")

        # Critical Path
        print("\n\nCRITICAL PATH")
        print("=" * 80)

        cp = report["critical_path"]
        print("\nImmediate Actions (This Week):")
        for action in cp["immediate_actions"]:
            print(f"  [ ] {action}")

        print("\nMedium Term (1-3 Months):")
        for action in cp["medium_term"]:
            print(f"  [ ] {action}")

        print("\nLong Term (3-6 Months):")
        for action in cp["long_term"]:
            print(f"  [ ] {action}")

        print("\n" + "=" * 80)
        print("Report generation complete.")
        print("For detailed dataset info: python -m utils.empirical_dataset_catalog")


class EnumEncoder(json.JSONEncoder):
    """JSON encoder that handles Enum types."""

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


def main():
    """Generate and print transition report."""
    report = EmpiricalTransitionReport()
    report.print_report()

    # Also save JSON version
    output_path = Path("empirical_transition_report.json")
    with open(output_path, "w") as f:
        json.dump(report.generate_report(), f, indent=2, cls=EnumEncoder)
    print(f"\nJSON report saved to: {output_path}")


# Stubs for test compatibility
def generate_transition_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a transition report from data."""
    transitions = data.get("transitions", [])
    return {
        "transitions": transitions,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_transitions": len(transitions),
            "total_count": sum(t.get("count", 0) for t in transitions),
        },
    }


def format_report_as_markdown(report: Dict[str, Any]) -> str:
    """Format a report as Markdown."""
    title = report.get("title", "Transition Report")
    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"## Transitions ({len(report.get('transitions', []))} total)",
        "",
    ]
    for t in report.get("transitions", []):
        lines.append(
            f"| {t.get('from', 'unknown')} | {t.get('to', 'unknown')} | {t.get('count', t.get('probability', 0))} |"
        )
    return "\n".join(lines) if lines else "# Empty Report"


def save_report(report: Dict[str, Any], filepath: Path, format: str = "json") -> None:
    """Save a report to file."""
    if format.lower() == "json":
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)


def load_report(filepath: Path) -> Dict[str, Any]:
    """Load a report from file."""
    with open(filepath, "r") as f:
        return json.load(f)


def compare_reports(report1: Dict[str, Any], report2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two reports."""
    return {
        "identical": report1 == report2,
        "differences": {},
        "summary": {
            "fields_in_first": list(report1.keys()),
            "fields_in_second": list(report2.keys()),
        },
    }


if __name__ == "__main__":
    main()
