# Empirical Data Integration Report for VP-11 and VP-15

This report documents the current state of empirical data integration
and provides a roadmap for achieving publication-ready status.

from pathlib import Path
from typing import Dict, List, Any

## CURRENT STATUS: Gap #4 - Empirical Data Integration

EMPIRICAL_DATA_STATUS = {
    "VP-11_MCMC_Cultural_Neuroscience": {
        "status": "SYNTHETIC_PENDING_EMPIRICAL",
        "current_score": 68,
        "target_score": 95,
        "has_loading_infrastructure": True,
        "data_loader": "load_cross_cultural_eeg_data()",
        "required_data_format": {
            "type": "cross-cultural EEG/fMRI/behavioral dataset",
            "min_subjects_per_culture": 15,
            "min_cultures": 3,
            "format": "CSV, NPZ, or BIDS",
        },
        "implementation_status": {
            "synthetic_mode": "✅ Working",
            "empirical_mode": "✅ Infrastructure exists",
            "real_data": "❌ Not integrated",
        },
        "integration_path": [
            "1. Obtain cross-cultural EEG dataset (e.g., from OpenNeuro or collaborators)",
            "2. Format data according to utils/empirical_data_generators.py specs",
            "3. Update VP-11 run_protocol_main() to pass empirical_data_path",
            "4. Validate parameter recovery on real data",
        ],
        "effort_estimate": "~4-6 hours (depends on data availability)",
    },

    "VP-15_fMRI_vmPFC_Anticipation": {
        "status": "STUB_IMPLEMENTATION",
        "current_score": 45,
        "target_score": 90,
        "has_loading_infrastructure": True,
        "data_loader": "load_fmri_data()",
        "required_data_format": {
            "type": "fMRI NIfTI files with vmPFC ROI",
            "min_subjects": 30,
            "paradigm": "resting-state + anticipation task",
            "format": ".nii, .nii.gz, or .npz",
        },
        "implementation_status": {
            "stub_mode": "✅ Returns placeholder",
            "simulation_mode": "✅ Synthetic BOLD generation works",
            "empirical_mode": "✅ Infrastructure exists (load_fmri_data())",
            "real_data": "❌ Not integrated",
        },
        "integration_path": [
            "1. Obtain fMRI dataset with vmPFC ROIs (e.g., from OpenNeuro ds001 confusion dataset)",
            "2. Preprocess data: motion correction, slice timing, HRF convolution",
            "3. Extract vmPFC time series using ROI mask",
            "4. Validate V15.1, V15.2, V15.3 predictions on real data",
        ],
        "effort_estimate": "~6-8 hours (depends on preprocessing complexity)",
    },
}

## DATA SOURCE RECOMMENDATIONS

RECOMMENDED_DATA_SOURCES = {
    "cross_cultural_eeg": [
        {
            "name": "OpenNeuro ds002578",
            "description": "Cross-cultural ERP study (N=60, US and Chinese participants)",
            "url": "https://openneuro.org/datasets/ds002578",
            "size": "~2GB",
            "compatibility": "High - ERP paradigm matches APGI requirements",
        },
        {
            "name": "Collaborative with Kitayama Lab",
            "description": "Cultural neuroscience datasets (individualism/collectivism)",
            "url": "Contact: kitayama@umich.edu",
            "compatibility": "High - Directly relevant to cultural parameters",
        },
    ],

    "fmri_vmpfc": [
        {
            "name": "OpenNeuro ds001 (Confusion dataset)",
            "description": "Anxiety/anticipation fMRI study with vmPFC activity",
            "url": "https://openneuro.org/datasets/ds001",
            "size": "~8GB",
            "compatibility": "High - Anticipation paradigm matches VP-15",
        },
        {
            "name": "Neurosynth vmPFC maps",
            "description": "Meta-analytic vmPFC functional connectivity",
            "url": "https://neurosynth.org/analyses/terms/vmpfc/",
            "compatibility": "Medium - Good for validation but not primary data",
        },
    ],
}

## PRIMARY INVESTIGATOR (PI) CONTACT DATABASE

PI_CONTACT_DATABASE = {
    "VP-11_MCMC_Cultural_Neuroscience": [
        {
            "pi_name": "Dr. Shinobu Kitayama",
            "institution": "University of Michigan",
            "email": "kitayama@umich.edu",
            "research_focus": "Cultural neuroscience, individualism/collectivism",
            "relevant_datasets": ["Cross-cultural ERP studies", "Cognitive style comparisons"],
            "collaboration_status": "Contacted - awaiting response",
            "data_sharing_policy": "Open via agreement",
            "last_contact": "2026-03-15",
        },
        {
            "pi_name": "Dr. Denise Park",
            "institution": "University of Texas at Dallas",
            "email": "denise.park@utdallas.edu",
            "research_focus": "Cognitive aging, cross-cultural neuroscience",
            "relevant_datasets": ["SCAN project - East Asian/US comparisons"],
            "collaboration_status": "Not yet contacted",
            "data_sharing_policy": "NDA required",
            "last_contact": None,
        },
        {
            "pi_name": "Dr. Yuri Miyamoto",
            "institution": "University of Wisconsin-Madison",
            "email": "miyamoto@wisc.edu",
            "research_focus": "Cultural psychology, cognitive processes",
            "relevant_datasets": ["Holistic vs. analytic processing EEG"],
            "collaboration_status": "Not yet contacted",
            "data_sharing_policy": "TBD",
            "last_contact": None,
        },
    ],

    "VP-15_fMRI_vmPFC_Anticipation": [
        {
            "pi_name": "Dr. Tor Wager",
            "institution": "Dartmouth College",
            "email": "tor.d.wager@dartmouth.edu",
            "research_focus": "Pain neuroimaging, affective neuroscience",
            "relevant_datasets": ["OpenNeuro ds001", "Pain anticipation fMRI"],
            "collaboration_status": "OpenNeuro data already public",
            "data_sharing_policy": "Open access",
            "last_contact": "N/A - public dataset",
        },
        {
            "pi_name": "Dr. Mauricio Delgado",
            "institution": "Rutgers University",
            "email": "delgado@psychology.rutgers.edu",
            "research_focus": "Reward anticipation, vmPFC function",
            "relevant_datasets": ["Reward processing fMRI datasets"],
            "collaboration_status": "Not yet contacted",
            "data_sharing_policy": "Collaboration agreement required",
            "last_contact": None,
        },
        {
            "pi_name": "Dr. Elizabeth Phelps",
            "institution": "Harvard University",
            "email": "liz_phelps@harvard.edu",
            "research_focus": "Emotion, fear conditioning, anticipation",
            "relevant_datasets": ["Fear anticipation neuroimaging"],
            "collaboration_status": "Not yet contacted",
            "data_sharing_policy": "TBD",
            "last_contact": None,
        },
    ],
}

def print_pi_database():
    """Print formatted PI contact database for empirical data integration."""
    print("=" * 80)
    print("PRIMARY INVESTIGATOR (PI) CONTACT DATABASE")
    print("=" * 80)
    print("\nFor VP-11 and VP-15 empirical data integration collaborative efforts.\n")

    for protocol, contacts in PI_CONTACT_DATABASE.items():
        print(f"\n{protocol}")
        print("-" * 80)
        for contact in contacts:
            print(f"  PI: {contact['pi_name']}")
            print(f"  Institution: {contact['institution']}")
            print(f"  Email: {contact['email']}")
            print(f"  Focus: {contact['research_focus']}")
            print(f"  Collaboration Status: {contact['collaboration_status']}")
            if contact['last_contact']:
                print(f"  Last Contact: {contact['last_contact']}")
            print()

## IMPLEMENTATION CHECKLIST

EMPIRICAL_INTEGRATION_CHECKLIST = """
## VP-11 Integration Checklist

- [ ] Download cross-cultural EEG dataset
- [ ] Verify data quality (minimum 15 subjects per culture)
- [ ] Convert to VP-11 expected format (CSV/ NPZ with columns: subject_id, culture, beta, alpha, tau)
- [ ] Run data through VP-11 empirical_data_path parameter
- [ ] Verify MCMC convergence on real data (R̂ < 1.01)
- [ ] Check parameter recovery rates
- [ ] Compare cultural parameter distributions
- [ ] Update protocol documentation with data source
- [ ] Add dataset citation to README
- [ ] Record any preprocessing steps

## VP-15 Integration Checklist

- [ ] Download fMRI dataset with vmPFC ROIs
- [ ] Verify data quality (minimum 30 subjects)
- [ ] Check TR, slice timing, and prep
- [ ] Run fMRIPrep or similar preprocessing pipeline
- [ ] Extract vmPFC ROI time series (using AAL or Harvard-Oxford atlas)
- [ ] Apply HRF convolution if not already done
- [ ] Quality control: motion parameters, tSNR maps
- [ ] Run VP-15 with preprocessed data
- [ ] Validate V15.1: Anticipatory onset < 500ms
- [ ] Validate V15.2: vmPFC-posterior insula connectivity r > 0.40
- [ ] Validate V15.3: Anterior/posterior dissociation
- [ ] Document preprocessing parameters
- [ ] Add dataset citation
- [ ] Update status from STUB to EMPIRICAL
"""

def print_empirical_data_report():
    """Print a formatted report of empirical data integration status."""
    print("=" * 80)
    print("EMPIRICAL DATA INTEGRATION REPORT - Gap #4")
    print("=" * 80)

    for protocol, info in EMPIRICAL_DATA_STATUS.items():
        print(f"\n{protocol}")
        print("-" * 80)
        print(f"  Status: {info['status']}")
        print(f"  Current Score: {info['current_score']}/100")
        print(f"  Target Score: {info['target_score']}/100")
        print(f"  Data Loader: {info['data_loader']}")
        print(f"  Effort Estimate: {info['effort_estimate']}")

        print("\n  Implementation Status:")
        for mode, status in info['implementation_status'].items():
            print(f"    {mode}: {status}")

        print("\n  Integration Path:")
        for step in info['integration_path']:
            print(f"    {step}")

    print("\n" + "=" * 80)
    print("RECOMMENDED DATA SOURCES")
    print("=" * 80)

    for data_type, sources in RECOMMENDED_DATA_SOURCES.items():
        print(f"\n{data_type.upper().replace('_', ' ')}:")
        for source in sources:
            print(f"\n  {source['name']}")
            print(f"    {source['description']}")
            print(f"    URL: {source['url']}")
            print(f"    Compatibility: {source['compatibility']}")

    print("\n" + "=" * 80)
    print(EMPIRICAL_INTEGRATION_CHECKLIST)
    print("=" * 80)

if __name__ == "__main__":
    print_empirical_data_report()
