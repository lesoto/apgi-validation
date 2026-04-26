"""BIDS-compliant data loaders for public neuroscience datasets.

Supports DS-09 (Cogitate iEEG), DS-12 (OpenNeuro EEG Depression), and DS-07 (Carhart-Harris fMRI).
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# BIDS Dataset Discovery
# ============================================================================


def discover_bids_dataset(bids_root: Union[str, Path]) -> Dict[str, Any]:
    """Discover and validate BIDS dataset structure.

    Args:
        bids_root: Path to BIDS dataset root directory

    Returns:
        Dictionary with dataset info: participants, modalities, tasks
    """
    bids_root = Path(bids_root)

    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    # Load dataset_description.json
    desc_path = bids_root / "dataset_description.json"
    dataset_info: Dict[str, Any] = {"root": str(bids_root)}

    if desc_path.exists():
        with open(desc_path) as f:
            dataset_info["description"] = json.load(f)

    # Load participants.tsv
    participants_path = bids_root / "participants.tsv"
    if participants_path.exists():
        participants_df = pd.read_csv(participants_path, sep="\t")
        dataset_info["participants"] = participants_df.to_dict("records")
        dataset_info["n_participants"] = len(participants_df)

    # Discover modalities
    modalities = []
    for mod_dir in ["eeg", "func", "ieeg", "anat"]:
        mod_path = bids_root / mod_dir
        if mod_path.exists():
            modalities.append(mod_dir)
    dataset_info["modalities"] = modalities

    return dataset_info


# ============================================================================
# DS-09: Cogitate iEEG Data Loader
# ============================================================================


def load_cogitate_ieeg_data(
    bids_root: Union[str, Path],
    subject_id: Optional[str] = None,
    extract_broadband_gamma: bool = True,
) -> Dict[str, Any]:
    """Load Cogitate Consortium iEEG data (DS-09).

    Cogitate 2025: Open multi-center intracranial EEG dataset with task probing
    conscious visual perception. N=38 patients, 3 centers, BIDS-formatted.

    APGI VP-15 Relevance:
        - vmPFC/sustained ignition predictions
        - Anterior/posterior insula dissociation
        - Connectivity patterns for V15.2 (vmPFC-insula connectivity)

    Args:
        bids_root: Path to Cogitate BIDS dataset
        subject_id: Specific subject to load (None = all)
        extract_broadband_gamma: Extract high-gamma band (70-150 Hz)

    Returns:
        Dictionary with iEEG data, electrode info, and behavioral data
    """
    bids_root = Path(bids_root)

    # Validate BIDS structure
    dataset_info = discover_bids_dataset(bids_root)
    logger.info(
        f"Loading Cogitate dataset: {dataset_info.get('n_participants', 'unknown')} participants"
    )

    # Find iEEG files
    ieeg_dir = bids_root / "ieeg"
    if not ieeg_dir.exists():
        raise FileNotFoundError(f"iEEG directory not found: {ieeg_dir}")

    # Look for data files
    data_files = list(ieeg_dir.glob("*_ieeg.*"))
    logger.info(f"Found {len(data_files)} iEEG files")

    # TODO: Implement actual data loading with MNE or neo
    # For now, return structure info for validation

    return {
        "dataset_id": "DS-09",
        "dataset_name": "Cogitate Consortium iEEG",
        "n_participants": dataset_info.get("n_participants", 38),
        "modalities": dataset_info.get("modalities", ["ieeg"]),
        "data_files": [str(f) for f in data_files],
        "electrode_info": "Loadable from *_electrodes.tsv",
        "events_info": "Loadable from *_events.tsv",
        "apgi_relevance": {
            "vp15_vmpfc": "Check for vmPFC electrode coverage",
            "vp15_insula": "Check for anterior/posterior insula coverage",
            "vp15_sustained": "Sustained vs transient ignition testable",
        },
        "note": "Full data loading requires MNE-python with iEEG support",
    }


# ============================================================================
# DS-12: OpenNeuro EEG Depression Data Loader
# ============================================================================


def load_openneuro_depression_eeg(
    bids_root: Union[str, Path],
    condition: str = "eyes_open",  # "eyes_open" or "eyes_closed"
) -> Dict[str, Any]:
    """Load OpenNeuro ds003478: Resting-State EEG in Depression.

    N=46 MDD patients, N=75 healthy controls. Eyes-open and eyes-closed conditions.
    Fully public, BIDS-compliant.

    APGI VP-11 Relevance:
        - Can extend to EEG-based validation of perceptual thresholds
        - Alpha/theta power as precision-weighting proxy
        - Aperiodic exponent extraction via specparam

    Args:
        bids_root: Path to ds003478 BIDS dataset
        condition: "eyes_open" or "eyes_closed"

    Returns:
        Dictionary with EEG data and participant info
    """
    bids_root = Path(bids_root)

    # Validate BIDS structure
    discover_bids_dataset(bids_root)

    # Map condition to BIDS task
    task_map = {"eyes_open": "restEyesOpen", "eyes_closed": "restEyesClosed"}
    task_name = task_map.get(condition, condition)

    eeg_dir = bids_root / "eeg"
    if not eeg_dir.exists():
        raise FileNotFoundError(f"EEG directory not found: {eeg_dir}")

    # Find EEG files for task
    eeg_files = list(eeg_dir.glob(f"*_task-{task_name}_eeg.*"))

    logger.info(f"DS-12: Found {len(eeg_files)} EEG files for {condition}")

    return {
        "dataset_id": "DS-12",
        "dataset_name": "OpenNeuro ds003478: Resting-State EEG in Depression",
        "condition": condition,
        "n_files": len(eeg_files),
        "n_mdd": 46,
        "n_controls": 75,
        "modalities": ["eeg"],
        "files": [str(f) for f in eeg_files[:5]],  # Sample first 5
        "apgi_relevance": {
            "vp11_aperiodic": "Extract βspec via specparam for each subject",
            "vp11_alpha": "Frontal alpha asymmetry as precision proxy",
            "depression_stratification": "MDD vs HC comparison for I-30",
        },
        "note": "Full loading requires MNE-python. Can compute aperiodic exponent with fooof-specparam.",
    }


# ============================================================================
# DS-07: Carhart-Harris Psychedelic fMRI Data Loader
# ============================================================================


def load_carhart_harris_fmri(
    bids_root: Union[str, Path],
    session: str = "psilocybin",  # or "placebo"
) -> Dict[str, Any]:
    """Load OpenNeuro ds003059: Carhart-Harris Psychedelic fMRI.

    Psilocybin and LSD resting-state fMRI. Tests APGI I-19 prediction:
    psychedelics flatten precision landscape and reduce βspec.

    APGI VP-15 Relevance:
        - DMN connectivity changes (vmPFC-PCC coupling)
        - Can test functional connectivity predictions

    Args:
        bids_root: Path to ds003059 BIDS dataset
        session: "psilocybin" or "placebo"

    Returns:
        Dictionary with fMRI data info
    """
    bids_root = Path(bids_root)

    # Validate BIDS structure
    discover_bids_dataset(bids_root)

    func_dir = bids_root / "func"
    if not func_dir.exists():
        raise FileNotFoundError(f"Functional MRI directory not found: {func_dir}")

    # Find resting-state fMRI files
    bold_files = list(func_dir.glob(f"*_ses-{session}_*_bold.nii.gz"))

    logger.info(f"DS-07: Found {len(bold_files)} BOLD files for {session}")

    return {
        "dataset_id": "DS-07",
        "dataset_name": "Carhart-Harris et al. Psychedelic fMRI",
        "session": session,
        "n_files": len(bold_files),
        "sample_size": 15,  # Psilocybin arm
        "modalities": ["fMRI", "MEG/EEG"],
        "files": [str(f) for f in bold_files[:5]],
        "apgi_relevance": {
            "vp15_dmn": "DMN connectivity (vmPFC-PCC) testable",
            "vp15_precision": "Precision landscape flattening prediction",
            "i19_flow_vs_dissolution": "Contrast psilocybin vs placebo states",
        },
        "note": "Full analysis requires nilearn/fmriprep for connectivity estimation",
    }


# ============================================================================
# DS-15: THINGS-Data Multimodal Loader
# ============================================================================


def load_things_data_eeg(
    data_root: Union[str, Path],
    subject_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load THINGS-Data EEG (rapid serial visual presentation paradigm).

    Extraordinarily large stimulus set (1,854 concepts). RSVP paradigm directly
    comparable to DS-01 (Sergent 2005) attentional blink paradigm.

    APGI VP-11 Relevance:
        - Trial-by-trial temporal dynamics for ignition onset
        - Object recognition as ignition proxy
        - Can validate threshold-adaptation predictions

    Args:
        data_root: Path to THINGS-EEG2 data
        subject_id: Specific subject (sub-01 through sub-10)

    Returns:
        Dictionary with EEG data info
    """
    data_root = Path(data_root)

    logger.info(f"Loading THINGS-Data from {data_root}")

    return {
        "dataset_id": "DS-15",
        "dataset_name": "THINGS-Data: Multimodal EEG/MEG/fMRI Object Representations",
        "n_subjects_eeg": 10,
        "n_subjects_fmri": 4,
        "n_concepts": 1854,
        "paradigm": "RSVP (100 ms/stimulus)",
        "apgi_relevance": {
            "vp11_rsvp": "RSVP comparable to attentional blink (DS-01)",
            "vp11_ignition": "Temporal dynamics of object recognition as ignition proxy",
            "vp11_reservoir": "Large stimulus set enables reservoir computing benchmarking",
        },
        "access": "Fully public via OSF, Zenodo, TIB",
        "note": "Data available via https://doi.org/10.7554/eLife.82580",
    }


# ============================================================================
# Generic Empirical Data Loader Router
# ============================================================================


def load_empirical_dataset(
    dataset_id: str,
    data_path: Union[str, Path],
    **kwargs,
) -> Dict[str, Any]:
    """Route to appropriate loader based on dataset ID.

    Args:
        dataset_id: One of DS-07, DS-09, DS-12, DS-15
        data_path: Path to dataset root
        **kwargs: Dataset-specific arguments

    Returns:
        Dataset info dictionary
    """
    loaders: Dict[str, Callable] = {
        "DS-07": load_carhart_harris_fmri,
        "DS-09": load_cogitate_ieeg_data,
        "DS-12": load_openneuro_depression_eeg,
        "DS-15": load_things_data_eeg,
    }

    if dataset_id not in loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_id}. Available: {list(loaders.keys())}"
        )

    logger.info(f"Loading {dataset_id} from {data_path}")
    return loaders[dataset_id](data_path, **kwargs)


# ============================================================================
# Validation Helpers
# ============================================================================


def check_dataset_availability(
    dataset_id: str, data_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Check if a dataset is available for empirical validation.

    Args:
        dataset_id: Dataset identifier (e.g., DS-09)
        data_root: Optional path to check

    Returns:
        Availability status dictionary
    """
    from utils.empirical_dataset_catalog import AccessStatus, get_dataset_by_id

    ds = get_dataset_by_id(dataset_id)
    if not ds:
        return {"available": False, "reason": "Unknown dataset ID"}

    status = {
        "dataset_id": dataset_id,
        "available": False,
        "access_status": ds.access_status.value,
        "public": ds.access_status == AccessStatus.FULLY_PUBLIC,
    }

    # Check if data path provided and exists
    if data_root and data_root.exists():
        status["path_exists"] = True
        try:
            discover_bids_dataset(data_root)
            status["bids_valid"] = True
            status["available"] = ds.access_status == AccessStatus.FULLY_PUBLIC
        except Exception as e:
            status["bids_valid"] = False
            status["error"] = str(e)
    elif ds.access_status == AccessStatus.FULLY_PUBLIC:
        status["reason"] = "Public dataset but no data path provided"
    else:
        status["reason"] = f"Requires {ds.access_status.value} access"

    return status


if __name__ == "__main__":
    # Test dataset discovery
    print("BIDS Data Loaders - Available Functions:")
    print("-" * 60)
    print("DS-07: load_carhart_harris_fmri() - Psychedelic fMRI")
    print("DS-09: load_cogitate_ieeg_data() - Multi-center iEEG")
    print("DS-12: load_openneuro_depression_eeg() - Resting EEG")
    print("DS-15: load_things_data_eeg() - RSVP object recognition")
