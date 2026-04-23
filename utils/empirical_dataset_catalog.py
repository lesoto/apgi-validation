"""Empirical Dataset Catalogue for APGI Validation Protocols.

This module maps public neuroscience datasets to specific validation protocols (VP-11, VP-15)
to enable empirical validation transitioning from simulation-only mode.

Based on: "PUBLIC DATASET CATALOGUE" (Last updated: Apr 22, 2026
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class DatasetTier(Enum):
    """APGI validation tier alignment."""

    THERMODYNAMIC = "Level 1"  # HEP, PCI, perturbational complexity
    INFORMATION_THEORETIC = "Level 2"  # Spectral, aperiodic, entropy
    COMPUTATIONAL = "Level 3"  # Attractor dynamics, ignition


class AccessStatus(Enum):
    """Dataset access status."""

    FULLY_PUBLIC = "green"  # BIDS, OpenNeuro, no barriers
    AUTHOR_REQUEST = "yellow"  # Contact authors
    INSTITUTIONAL = "red"  # DUA required
    FORTHCOMING = "forthcoming"  # Not yet released


@dataclass
class EmpiricalDataset:
    """Specification for a public neuroscience dataset."""

    id: str  # DS-XX identifier
    name: str  # Citation name
    tier: DatasetTier
    modality: str  # EEG, fMRI, iEEG, etc.
    access_status: AccessStatus
    primary_url: str
    sample_size: int
    key_measures: List[str] = field(default_factory=list)
    apgi_innovations: List[str] = field(default_factory=list)
    validation_protocols: List[str] = field(default_factory=list)
    bids_compliant: bool = False
    notes: str = ""


# ============================================================================
# DATASET REGISTRY - Public datasets mapped to APGI validation protocols
# ============================================================================

EMPIRICAL_DATASETS: Dict[str, EmpiricalDataset] = {
    # =========================================================================
    # LEVEL 3: Computational/Perceptual Paradigms (VP-11 candidates)
    # =========================================================================
    "DS-01": EmpiricalDataset(
        id="DS-01",
        name="Sergent, Baillet & Dehaene (2005): Attentional Blink / Near-Threshold Masking",
        tier=DatasetTier.COMPUTATIONAL,
        modality="EEG (128-channel)",
        access_status=AccessStatus.AUTHOR_REQUEST,
        primary_url="https://pubmed.ncbi.nlm.nih.gov/16158062/",
        sample_size=12,
        key_measures=[
            "P1, N1 amplitude",
            "late conscious-access wave (~270 ms)",
            "binary seen/unseen classification",
            "continuous visibility scale",
        ],
        apgi_innovations=[
            "I-04",
            "I-15",
        ],  # Attractor-Basin Bifurcation, Classic Paradigms
        validation_protocols=["VP-11"],
        bids_compliant=False,
        notes="Trial-by-trial visibility ratings enable Hill coefficient calculation. Pre-BIDS, needs harmonization.",
    ),
    "DS-02": EmpiricalDataset(
        id="DS-02",
        name="Melloni et al. (2007): Perceptual Detection with Gamma Synchrony",
        tier=DatasetTier.COMPUTATIONAL,
        modality="EEG (high-density)",
        access_status=AccessStatus.AUTHOR_REQUEST,
        primary_url="https://www.jneurosci.org/content/27/11/2858",
        sample_size=8,
        key_measures=[
            "Gamma-band synchrony (40 Hz)",
            "P300 / late cortical potential",
            "seen/unseen detection accuracy",
        ],
        apgi_innovations=["I-15"],  # Classic Perceptual Paradigms
        validation_protocols=["VP-11"],
        bids_compliant=False,
        notes="Gamma synchrony provides direct test of global workspace ignition. Very small N.",
    ),
    "DS-15": EmpiricalDataset(
        id="DS-15",
        name="THINGS-Data: Multimodal EEG, MEG & fMRI Object Representations",
        tier=DatasetTier.COMPUTATIONAL,
        modality="EEG/MEG/fMRI",
        access_status=AccessStatus.FULLY_PUBLIC,
        primary_url="https://doi.org/10.7554/eLife.82580",
        sample_size=10,  # EEG arm
        key_measures=[
            "EEG temporal dynamics (1 ms resolution)",
            "Representational similarity analysis (RSA)",
            "4.7M behavioral similarity judgments",
        ],
        apgi_innovations=[
            "I-04",
            "I-15",
        ],  # Reservoir attractor dynamics, Classic Paradigms
        validation_protocols=["VP-11"],
        bids_compliant=True,
        notes="Extraordinarily large stimulus set (1,854 concepts). RSVP paradigm comparable to DS-01.",
    ),
    # =========================================================================
    # LEVEL 2: Information-Theoretic / Spectral Dynamics
    # =========================================================================
    "DS-04": EmpiricalDataset(
        id="DS-04",
        name="Donoghue et al. (2020): specparam / FOOOF Aperiodic Parameterization",
        tier=DatasetTier.INFORMATION_THEORETIC,
        modality="EEG/LFP",
        access_status=AccessStatus.FULLY_PUBLIC,
        primary_url="https://github.com/fooof-tools/fooof",
        sample_size=100,  # Multi-dataset validation
        key_measures=[
            "Aperiodic exponent (slope)",
            "Aperiodic offset",
            "Periodic peak frequency/power",
            "E/I ratio proxy",
        ],
        apgi_innovations=["I-09"],  # 1/f Spectral Slope
        validation_protocols=[],
        bids_compliant=False,
        notes="Gold-standard method for aperiodic decomposition. Code open-source.",
    ),
    "DS-07": EmpiricalDataset(
        id="DS-07",
        name="Carhart-Harris et al. (2012-2019): Psychedelic EEG/fMRI",
        tier=DatasetTier.INFORMATION_THEORETIC,
        modality="fMRI/MEG/EEG",
        access_status=AccessStatus.FULLY_PUBLIC,
        primary_url="https://openneuro.org/datasets/ds003059",
        sample_size=15,  # Psilocybin fMRI
        key_measures=[
            "Global alpha power reduction",
            "Broadband spectral changes",
            "DMN connectivity",
            "Entropy measures",
        ],
        apgi_innovations=["I-19"],  # Flow vs. Psychedelic Dissolution
        validation_protocols=["VP-15"],  # fMRI connectivity patterns
        bids_compliant=True,
        notes="OpenNeuro ds003059 fully public. Tests precision landscape flattening prediction.",
    ),
    # =========================================================================
    # LEVEL 1: Thermodynamic / Perturbational Complexity (VP-15 candidates)
    # =========================================================================
    "DS-08": EmpiricalDataset(
        id="DS-08",
        name="Casali et al. (2013): TMS-EEG Perturbational Complexity Index (PCI)",
        tier=DatasetTier.THERMODYNAMIC,
        modality="TMS-EEG",
        access_status=AccessStatus.AUTHOR_REQUEST,
        primary_url="https://pubmed.ncbi.nlm.nih.gov/23946194/",
        sample_size=216,  # 108 healthy + 108 patients
        key_measures=[
            "PCI (Perturbational Complexity Index)",
            "TMS-evoked potential spatiotemporal complexity",
            "Lempel-Ziv compression",
        ],
        apgi_innovations=["I-20", "I-33"],  # Joint HEP x PCI, Cross-Species
        validation_protocols=["VP-15"],
        bids_compliant=False,
        notes="Gold standard for global ignition capacity. Code open (PCIst). CRITICAL GAP: No concurrent HEP.",
    ),
    "DS-09": EmpiricalDataset(
        id="DS-09",
        name="Cogitate Consortium (2025): Open Multi-Center iEEG Dataset",
        tier=DatasetTier.THERMODYNAMIC,
        modality="iEEG (38 patients, 3 centers)",
        access_status=AccessStatus.FULLY_PUBLIC,
        primary_url="https://www.nature.com/articles/s41597-025-04833-z",
        sample_size=38,
        key_measures=[
            "Broadband high-gamma",
            "Sustained vs. transient activity",
            "Ignition vs. local recurrence",
            "GNW vs. IIT predictions",
        ],
        apgi_innovations=["I-20", "I-33"],
        validation_protocols=["VP-15"],  # vmPFC connectivity, sustained ignition
        bids_compliant=True,
        notes="Largest public iEEG dataset for consciousness. Jupyter tutorial included. Forthcoming MEG/fMRI.",
    ),
    # =========================================================================
    # CLINICAL / PSYCHIATRIC STRATIFICATION (VP-15 candidates)
    # =========================================================================
    "DS-10": EmpiricalDataset(
        id="DS-10",
        name="Drysdale et al. (2017): fMRI Biotypes of Depression",
        tier=DatasetTier.COMPUTATIONAL,
        modality="rsfMRI",
        access_status=AccessStatus.INSTITUTIONAL,
        primary_url="https://pubmed.ncbi.nlm.nih.gov/27918562/",
        sample_size=1188,
        key_measures=[
            "mPFC-hippocampal connectivity",
            "Frontostriatal connectivity",
            "Biotype 1-4 classification",
            "TMS response prediction",
        ],
        apgi_innovations=[
            "I-10",
            "I-30",
        ],  # Psychiatric Biotyping, Depression Specifiers
        validation_protocols=["VP-15"],  # vmPFC connectivity
        bids_compliant=False,
        notes="Large N but replication concerns raised. Requires institutional DUA.",
    ),
    "DS-11": EmpiricalDataset(
        id="DS-11",
        name="HCP-EP: Human Connectome Project for Early Psychosis",
        tier=DatasetTier.COMPUTATIONAL,
        modality="rsfMRI/dMRI",
        access_status=AccessStatus.FULLY_PUBLIC,
        primary_url="https://humanconnectome.org/study/human-connectome-project-for-early-psychosis",
        sample_size=1100,
        key_measures=[
            "Functional connectivity matrices",
            "Structural connectivity",
            "PANSS scores",
            "Cognitive battery",
        ],
        apgi_innovations=["I-10"],  # Psychiatric Biotyping
        validation_protocols=["VP-15"],  # Functional connectivity
        bids_compliant=True,
        notes="Public via CCF. No EEG so temporal dynamics untestable.",
    ),
    "DS-12": EmpiricalDataset(
        id="DS-12",
        name="OpenNeuro ds003478: Resting-State EEG in Depression",
        tier=DatasetTier.COMPUTATIONAL,
        modality="EEG",
        access_status=AccessStatus.FULLY_PUBLIC,
        primary_url="https://openneuro.org/datasets/ds003478",
        sample_size=121,  # 46 MDD + 75 HC
        key_measures=[
            "Alpha power",
            "Theta power",
            "Frontal asymmetry",
            "Aperiodic exponent (extractable)",
        ],
        apgi_innovations=["I-30"],  # Depression Specifiers
        validation_protocols=["VP-11"],  # Could extend to EEG-based validation
        bids_compliant=True,
        notes="Fully public, no registration required. Eyes-open/closed conditions.",
    ),
    # =========================================================================
    # FORTHCOMING HIGH-VALUE DATASETS
    # =========================================================================
    "DS-16": EmpiricalDataset(
        id="DS-16",
        name="Cogitate: GNW x IIT Adversarial fMRI/MEG Dataset (Forthcoming)",
        tier=DatasetTier.THERMODYNAMIC,  # Multi-tier
        modality="fMRI/MEG/iEEG",
        access_status=AccessStatus.FORTHCOMING,
        primary_url="https://www.arc-cogitate.com",
        sample_size=256,  # Total across modalities
        key_measures=[
            "Sustained vs. transient ignition",
            "Late frontal amplification",
            "GNW vs. IIT discriminating predictions",
            "Broadband gamma",
        ],
        apgi_innovations=["I-20", "I-33"],
        validation_protocols=["VP-11", "VP-15"],
        bids_compliant=True,
        notes="Largest consciousness dataset ever. fMRI+MEG releases forthcoming (not yet April 2026).",
    ),
}


# ============================================================================
# VALIDATION PROTOCOL TO DATASET MAPPING
# ============================================================================

PROTOCOL_DATASET_MAPPING: Dict[str, List[str]] = {
    "VP-11": ["DS-01", "DS-02", "DS-15", "DS-12"],  # Perceptual paradigms, EEG
    "VP-15": ["DS-07", "DS-09", "DS-10", "DS-11", "DS-16"],  # fMRI, connectivity
}


def get_datasets_for_protocol(protocol_id: str) -> List[EmpiricalDataset]:
    """Get list of datasets that can validate a specific protocol."""
    dataset_ids = PROTOCOL_DATASET_MAPPING.get(protocol_id, [])
    return [
        EMPIRICAL_DATASETS[ds_id]
        for ds_id in dataset_ids
        if ds_id in EMPIRICAL_DATASETS
    ]


def get_accessible_datasets(protocol_id: str) -> List[EmpiricalDataset]:
    """Get datasets that are fully public (green) for immediate use."""
    all_datasets = get_datasets_for_protocol(protocol_id)
    return [ds for ds in all_datasets if ds.access_status == AccessStatus.FULLY_PUBLIC]


def get_dataset_by_id(dataset_id: str) -> Optional[EmpiricalDataset]:
    """Retrieve a specific dataset by its ID."""
    return EMPIRICAL_DATASETS.get(dataset_id)


def print_dataset_summary():
    """Print summary of available datasets for CLI/documentation."""
    print("=" * 80)
    print("APGI EMPIRICAL DATASET CATALOGUE")
    print("=" * 80)
    print()

    for protocol, dataset_ids in PROTOCOL_DATASET_MAPPING.items():
        print(f"\n{protocol} - Available Datasets:")
        print("-" * 60)

        for ds_id in dataset_ids:
            ds = EMPIRICAL_DATASETS.get(ds_id)
            if ds:
                status_icon = {
                    AccessStatus.FULLY_PUBLIC: "✅",
                    AccessStatus.AUTHOR_REQUEST: "📝",
                    AccessStatus.INSTITUTIONAL: "🔒",
                    AccessStatus.FORTHCOMING: "⏳",
                }.get(ds.access_status, "❓")

                print(f"  {status_icon} {ds.id}: {ds.name}")
                print(
                    f"     Tier: {ds.tier.value} | Modality: {ds.modality} | N={ds.sample_size}"
                )
                print(f"     APGI Innovations: {', '.join(ds.apgi_innovations)}")
                print(f"     URL: {ds.primary_url}")
                print()


if __name__ == "__main__":
    print_dataset_summary()
