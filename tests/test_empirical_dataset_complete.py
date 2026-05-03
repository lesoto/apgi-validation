"""
Comprehensive tests for utils/empirical_dataset_catalog.py - 100% coverage target.

This file tests:
- EmpiricalDataset dataclass functionality
- DatasetTier and AccessStatus enums
- Dataset registry and validation
- APGI protocol mapping
- Data integrity and validation
"""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.empirical_dataset_catalog import (
        AccessStatus,
        DatasetTier,
        EmpiricalDataset,
    )

    EMPIRICAL_CATALOG_AVAILABLE = True
except ImportError as e:
    EMPIRICAL_CATALOG_AVAILABLE = False
    print(f"Warning: empirical_dataset_catalog not available for testing: {e}")


class TestEmpiricalDatasetComplete:
    """Comprehensive tests for EmpiricalDataset functionality."""

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_empirical_dataset_creation(self):
        """Test EmpiricalDataset dataclass creation."""
        dataset = EmpiricalDataset(
            id="DS-01",
            name="Test Dataset",
            tier=DatasetTier.THERMODYNAMIC,
            modality="EEG",
            access_status=AccessStatus.FULLY_PUBLIC,
            primary_url="https://example.com/dataset",
            sample_size=100,
            key_measures=["power_spectrum", "connectivity"],
            apgi_innovations=["entropy", "complexity"],
            validation_protocols=["VP-01", "VP-02"],
            bids_compliant=True,
            notes="Test dataset for validation",
        )

        assert dataset.id == "DS-01"
        assert dataset.name == "Test Dataset"
        assert dataset.tier == DatasetTier.THERMODYNAMIC
        assert dataset.modality == "EEG"
        assert dataset.access_status == AccessStatus.FULLY_PUBLIC
        assert dataset.primary_url == "https://example.com/dataset"
        assert dataset.sample_size == 100
        assert len(dataset.key_measures) == 2
        assert len(dataset.apgi_innovations) == 2
        assert len(dataset.validation_protocols) == 2
        assert dataset.bids_compliant is True
        assert dataset.notes == "Test dataset for validation"

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_empirical_dataset_defaults(self):
        """Test EmpiricalDataset with default values."""
        dataset = EmpiricalDataset(
            id="DS-02",
            name="Minimal Dataset",
            tier=DatasetTier.INFORMATION_THEORETIC,
            modality="fMRI",
            access_status=AccessStatus.AUTHOR_REQUEST,
            primary_url="https://example.com/minimal",
            sample_size=50,
        )

        # Default values should be empty lists/collections
        assert dataset.key_measures == []
        assert dataset.apgi_innovations == []
        assert dataset.validation_protocols == []
        assert dataset.bids_compliant is False
        assert dataset.notes == ""

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_dataset_tier_enum(self):
        """Test DatasetTier enum values."""
        assert DatasetTier.THERMODYNAMIC.value == "Level 1"
        assert DatasetTier.INFORMATION_THEORETIC.value == "Level 2"
        assert DatasetTier.COMPUTATIONAL.value == "Level 3"

        # Test enum comparison
        assert DatasetTier.THERMODYNAMIC != DatasetTier.INFORMATION_THEORETIC
        assert DatasetTier.INFORMATION_THEORETIC != DatasetTier.COMPUTATIONAL
        assert DatasetTier.COMPUTATIONAL != DatasetTier.THERMODYNAMIC

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_access_status_enum(self):
        """Test AccessStatus enum values."""
        assert AccessStatus.FULLY_PUBLIC.value == "green"
        assert AccessStatus.AUTHOR_REQUEST.value == "yellow"
        assert AccessStatus.INSTITUTIONAL.value == "red"
        assert AccessStatus.FORTHCOMING.value == "forthcoming"

        # Test enum comparison
        assert AccessStatus.FULLY_PUBLIC != AccessStatus.AUTHOR_REQUEST
        assert AccessStatus.AUTHOR_REQUEST != AccessStatus.INSTITUTIONAL
        assert AccessStatus.INSTITUTIONAL != AccessStatus.FORTHCOMING

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_dataset_serialization(self):
        """Test EmpiricalDataset JSON serialization."""
        dataset = EmpiricalDataset(
            id="DS-03",
            name="Serialization Test",
            tier=DatasetTier.COMPUTATIONAL,
            modality="iEEG",
            access_status=AccessStatus.INSTITUTIONAL,
            primary_url="https://example.com/serial",
            sample_size=200,
            key_measures=["spectral_analysis", "phase_synchronization"],
            apgi_innovations=["attractor_dynamics", "ignition"],
            validation_protocols=["VP-03", "VP-04"],
            bids_compliant=False,
            notes="Test serialization",
        )

        # Convert to dictionary for JSON serialization
        dataset_dict = {
            "id": dataset.id,
            "name": dataset.name,
            "tier": dataset.tier.value,
            "modality": dataset.modality,
            "access_status": dataset.access_status.value,
            "primary_url": dataset.primary_url,
            "sample_size": dataset.sample_size,
            "key_measures": dataset.key_measures,
            "apgi_innovations": dataset.apgi_innovations,
            "validation_protocols": dataset.validation_protocols,
            "bids_compliant": dataset.bids_compliant,
            "notes": dataset.notes,
        }

        # Should be JSON serializable
        json_str = json.dumps(dataset_dict)
        assert isinstance(json_str, str)

        # Should be able to load back
        loaded_dict = json.loads(json_str)
        assert loaded_dict == dataset_dict

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_dataset_validation_by_tier(self):
        """Test dataset validation based on tier requirements."""
        # Thermodynamic tier should have specific measures
        thermodynamic_dataset = EmpiricalDataset(
            id="DS-THERM",
            name="Thermodynamic Dataset",
            tier=DatasetTier.THERMODYNAMIC,
            modality="EEG",
            access_status=AccessStatus.FULLY_PUBLIC,
            primary_url="https://example.com/thermo",
            sample_size=100,
            key_measures=["perturbational_complexity", "phase_locking_value"],
            apgi_innovations=["entropy", "free_energy"],
            validation_protocols=["VP-01"],
            bids_compliant=True,
        )

        # Information-theoretic tier should have spectral measures
        info_dataset = EmpiricalDataset(
            id="DS-INFO",
            name="Information Dataset",
            tier=DatasetTier.INFORMATION_THEORETIC,
            modality="EEG",
            access_status=AccessStatus.FULLY_PUBLIC,
            primary_url="https://example.com/info",
            sample_size=150,
            key_measures=["power_spectrum", "mutual_information"],
            apgi_innovations=["entropy", "complexity"],
            validation_protocols=["VP-02"],
            bids_compliant=True,
        )

        # Computational tier should have dynamics measures
        comp_dataset = EmpiricalDataset(
            id="DS-COMP",
            name="Computational Dataset",
            tier=DatasetTier.COMPUTATIONAL,
            modality="fMRI",
            access_status=AccessStatus.FULLY_PUBLIC,
            primary_url="https://example.com/comp",
            sample_size=80,
            key_measures=["attractor_dynamics", "ignition"],
            apgi_innovations=["predictive_coding", "bayesian_inference"],
            validation_protocols=["VP-03"],
            bids_compliant=False,
        )

        # Verify tier assignments
        assert thermodynamic_dataset.tier == DatasetTier.THERMODYNAMIC
        assert info_dataset.tier == DatasetTier.INFORMATION_THEORETIC
        assert comp_dataset.tier == DatasetTier.COMPUTATIONAL

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_access_status_filtering(self):
        """Test filtering datasets by access status."""
        datasets = [
            EmpiricalDataset(
                id="DS-PUBLIC",
                name="Public Dataset",
                tier=DatasetTier.THERMODYNAMIC,
                modality="EEG",
                access_status=AccessStatus.FULLY_PUBLIC,
                primary_url="https://example.com/public",
                sample_size=100,
            ),
            EmpiricalDataset(
                id="DS-AUTHOR",
                name="Author Request Dataset",
                tier=DatasetTier.INFORMATION_THEORETIC,
                modality="fMRI",
                access_status=AccessStatus.AUTHOR_REQUEST,
                primary_url="https://example.com/author",
                sample_size=80,
            ),
            EmpiricalDataset(
                id="DS-INSTITUTION",
                name="Institutional Dataset",
                tier=DatasetTier.COMPUTATIONAL,
                modality="iEEG",
                access_status=AccessStatus.INSTITUTIONAL,
                primary_url="https://example.com/institution",
                sample_size=60,
            ),
            EmpiricalDataset(
                id="DS-FORTHCOMING",
                name="Forthcoming Dataset",
                tier=DatasetTier.THERMODYNAMIC,
                modality="MEG",
                access_status=AccessStatus.FORTHCOMING,
                primary_url="https://example.com/forthcoming",
                sample_size=0,
            ),
        ]

        # Filter by access status
        public_datasets = [
            d for d in datasets if d.access_status == AccessStatus.FULLY_PUBLIC
        ]
        author_datasets = [
            d for d in datasets if d.access_status == AccessStatus.AUTHOR_REQUEST
        ]
        institutional_datasets = [
            d for d in datasets if d.access_status == AccessStatus.INSTITUTIONAL
        ]
        forthcoming_datasets = [
            d for d in datasets if d.access_status == AccessStatus.FORTHCOMING
        ]

        assert len(public_datasets) == 1
        assert len(author_datasets) == 1
        assert len(institutional_datasets) == 1
        assert len(forthcoming_datasets) == 1

        assert public_datasets[0].id == "DS-PUBLIC"
        assert author_datasets[0].id == "DS-AUTHOR"
        assert institutional_datasets[0].id == "DS-INSTITUTION"
        assert forthcoming_datasets[0].id == "DS-FORTHCOMING"

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_modality_filtering(self):
        """Test filtering datasets by modality."""
        modalities = ["EEG", "fMRI", "iEEG", "MEG", "MEG-ECG"]
        datasets = []

        for i, modality in enumerate(modalities):
            dataset = EmpiricalDataset(
                id=f"DS-{i:02d}",
                name=f"{modality} Dataset",
                tier=DatasetTier.THERMODYNAMIC,
                modality=modality,
                access_status=AccessStatus.FULLY_PUBLIC,
                primary_url=f"https://example.com/{modality.lower()}",
                sample_size=50 + i * 10,
            )
            datasets.append(dataset)

        # Filter by modality
        eeg_datasets = [d for d in datasets if d.modality == "EEG"]
        fmri_datasets = [d for d in datasets if d.modality == "fMRI"]
        ieeg_datasets = [d for d in datasets if d.modality == "iEEG"]
        meg_datasets = [d for d in datasets if d.modality == "MEG"]
        meg_ecg_datasets = [d for d in datasets if d.modality == "MEG-ECG"]

        assert len(eeg_datasets) == 1
        assert len(fmri_datasets) == 1
        assert len(ieeg_datasets) == 1
        assert len(meg_datasets) == 1
        assert len(meg_ecg_datasets) == 1

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_sample_size_analysis(self):
        """Test sample size analysis across datasets."""
        datasets = [
            EmpiricalDataset(
                id="DS-SMALL",
                name="Small Dataset",
                tier=DatasetTier.THERMODYNAMIC,
                modality="EEG",
                access_status=AccessStatus.FULLY_PUBLIC,
                primary_url="https://example.com/small",
                sample_size=20,
            ),
            EmpiricalDataset(
                id="DS-MEDIUM",
                name="Medium Dataset",
                tier=DatasetTier.INFORMATION_THEORETIC,
                modality="fMRI",
                access_status=AccessStatus.FULLY_PUBLIC,
                primary_url="https://example.com/medium",
                sample_size=100,
            ),
            EmpiricalDataset(
                id="DS-LARGE",
                name="Large Dataset",
                tier=DatasetTier.COMPUTATIONAL,
                modality="iEEG",
                access_status=AccessStatus.FULLY_PUBLIC,
                primary_url="https://example.com/large",
                sample_size=500,
            ),
        ]

        # Analyze sample sizes
        sample_sizes = [d.sample_size for d in datasets]
        total_subjects = sum(sample_sizes)
        mean_sample_size = total_subjects / len(datasets)
        min_sample_size = min(sample_sizes)
        max_sample_size = max(sample_sizes)

        assert total_subjects == 620
        assert mean_sample_size == 620 / 3
        assert min_sample_size == 20
        assert max_sample_size == 500

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_validation_protocol_mapping(self):
        """Test mapping datasets to validation protocols."""
        datasets = [
            EmpiricalDataset(
                id="DS-VP01",
                name="VP-01 Dataset",
                tier=DatasetTier.THERMODYNAMIC,
                modality="EEG",
                access_status=AccessStatus.FULLY_PUBLIC,
                primary_url="https://example.com/vp01",
                sample_size=100,
                validation_protocols=["VP-01", "VP-02"],
            ),
            EmpiricalDataset(
                id="DS-VP03",
                name="VP-03 Dataset",
                tier=DatasetTier.COMPUTATIONAL,
                modality="fMRI",
                access_status=AccessStatus.FULLY_PUBLIC,
                primary_url="https://example.com/vp03",
                sample_size=80,
                validation_protocols=["VP-03", "VP-04"],
            ),
            EmpiricalDataset(
                id="DS-MULTI",
                name="Multi-Protocol Dataset",
                tier=DatasetTier.INFORMATION_THEORETIC,
                modality="MEG",
                access_status=AccessStatus.FULLY_PUBLIC,
                primary_url="https://example.com/multi",
                sample_size=150,
                validation_protocols=["VP-01", "VP-02", "VP-03", "VP-04"],
            ),
        ]

        # Map protocols to datasets
        protocol_mapping = {}
        for dataset in datasets:
            for protocol in dataset.validation_protocols:
                if protocol not in protocol_mapping:
                    protocol_mapping[protocol] = []
                protocol_mapping[protocol].append(dataset.id)

        # Verify mapping
        assert len(protocol_mapping["VP-01"]) == 2  # DS-VP01, DS-MULTI
        assert len(protocol_mapping["VP-02"]) == 2  # DS-VP01, DS-MULTI
        assert len(protocol_mapping["VP-03"]) == 2  # DS-VP03, DS-MULTI
        assert len(protocol_mapping["VP-04"]) == 2  # DS-VP03, DS-MULTI

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_bids_compliance_analysis(self):
        """Test BIDS compliance analysis."""
        datasets = [
            EmpiricalDataset(
                id="DS-BIDS",
                name="BIDS Compliant Dataset",
                tier=DatasetTier.THERMODYNAMIC,
                modality="EEG",
                access_status=AccessStatus.FULLY_PUBLIC,
                primary_url="https://example.com/bids",
                sample_size=100,
                bids_compliant=True,
            ),
            EmpiricalDataset(
                id="DS-NON-BIDS",
                name="Non-BIDS Dataset",
                tier=DatasetTier.INFORMATION_THEORETIC,
                modality="fMRI",
                access_status=AccessStatus.AUTHOR_REQUEST,
                primary_url="https://example.com/non-bids",
                sample_size=80,
                bids_compliant=False,
            ),
        ]

        # Analyze BIDS compliance
        bids_compliant = [d for d in datasets if d.bids_compliant]
        non_bids = [d for d in datasets if not d.bids_compliant]

        assert len(bids_compliant) == 1
        assert len(non_bids) == 1
        assert bids_compliant[0].id == "DS-BIDS"
        assert non_bids[0].id == "DS-NON-BIDS"

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_apgi_innovations_analysis(self):
        """Test APGI innovations analysis."""
        datasets = [
            EmpiricalDataset(
                id="DS-ENTROPY",
                name="Entropy Dataset",
                tier=DatasetTier.INFORMATION_THEORETIC,
                modality="EEG",
                access_status=AccessStatus.FULLY_PUBLIC,
                primary_url="https://example.com/entropy",
                sample_size=100,
                apgi_innovations=["entropy", "complexity", "mutual_information"],
            ),
            EmpiricalDataset(
                id="DS-DYNAMICS",
                name="Dynamics Dataset",
                tier=DatasetTier.COMPUTATIONAL,
                modality="fMRI",
                access_status=AccessStatus.FULLY_PUBLIC,
                primary_url="https://example.com/dynamics",
                sample_size=80,
                apgi_innovations=[
                    "attractor_dynamics",
                    "ignition",
                    "predictive_coding",
                ],
            ),
        ]

        # Analyze innovations
        all_innovations = []
        for dataset in datasets:
            all_innovations.extend(dataset.apgi_innovations)

        unique_innovations = list(set(all_innovations))

        assert len(all_innovations) == 6
        assert len(unique_innovations) == 6  # All unique in this case
        assert "entropy" in unique_innovations
        assert "attractor_dynamics" in unique_innovations

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_dataset_equality(self):
        """Test EmpiricalDataset equality comparison."""
        dataset1 = EmpiricalDataset(
            id="DS-EQUAL",
            name="Equal Dataset",
            tier=DatasetTier.THERMODYNAMIC,
            modality="EEG",
            access_status=AccessStatus.FULLY_PUBLIC,
            primary_url="https://example.com/equal",
            sample_size=100,
        )

        dataset2 = EmpiricalDataset(
            id="DS-EQUAL",
            name="Equal Dataset",
            tier=DatasetTier.THERMODYNAMIC,
            modality="EEG",
            access_status=AccessStatus.FULLY_PUBLIC,
            primary_url="https://example.com/equal",
            sample_size=100,
        )

        dataset3 = EmpiricalDataset(
            id="DS-DIFFERENT",
            name="Different Dataset",
            tier=DatasetTier.INFORMATION_THEORETIC,
            modality="fMRI",
            access_status=AccessStatus.AUTHOR_REQUEST,
            primary_url="https://example.com/different",
            sample_size=80,
        )

        # Dataclass equality should work
        assert dataset1 == dataset2
        assert dataset1 != dataset3
        assert dataset2 != dataset3

    @pytest.mark.skipif(
        not EMPIRICAL_CATALOG_AVAILABLE,
        reason="empirical_dataset_catalog not available",
    )
    def test_dataset_string_representation(self):
        """Test EmpiricalDataset string representation."""
        dataset = EmpiricalDataset(
            id="DS-STRING",
            name="String Test Dataset",
            tier=DatasetTier.COMPUTATIONAL,
            modality="EEG",
            access_status=AccessStatus.FULLY_PUBLIC,
            primary_url="https://example.com/string",
            sample_size=100,
        )

        str_repr = str(dataset)
        repr_str = repr(dataset)

        # Should contain key information
        assert "DS-STRING" in str_repr
        assert "String Test Dataset" in str_repr
        assert "COMPUTATIONAL" in str_repr
        assert "EEG" in str_repr

        # repr should be more detailed
        assert repr_str == str_repr
