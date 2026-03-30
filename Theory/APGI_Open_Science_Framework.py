"""
APGI Open Science Infrastructure
================================

Complete open science framework for APGI theory validation including:
- Data sharing protocols and repositories
- Preregistration templates for experiments
- Replication protocols and pipelines
- Open access publication templates
- Collaborative validation framework

"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def serialize(obj):
    """Custom JSON serializer for non-serializable objects"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, np.datetime64)):
        return obj.isoformat()
    else:
        return str(obj)


@dataclass
class PreregistrationTemplate:
    """Preregistration template for APGI experiments"""

    # Study metadata
    title: str
    authors: List[str]
    date_created: str
    predicted_completion: str

    # Research questions and hypotheses
    research_questions: List[str]
    hypotheses: List[str]
    theoretical_background: str

    # Experimental design
    design_type: str  # 'behavioral', 'neural', 'clinical', 'cross_species'
    paradigm: str  # 'masking', 'tms', 'psychophysics', etc.
    sample_size: int
    power_analysis: Dict[str, float]

    # APGI-specific predictions
    apgi_predictions: Dict[str, str]  # Parameter -> predicted effect
    falsification_criteria: List[str]

    # Analysis plan
    primary_analyses: List[str]
    secondary_analyses: List[str]
    exclusion_criteria: List[str]

    # Data management
    data_repository: str
    code_repository: str
    open_materials: bool
    open_data: bool

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for numpy arrays and datetime objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def to_json(self) -> str:
        """Export preregistration as JSON"""
        return json.dumps(asdict(self), indent=2, default=self._json_serializer)

    @classmethod
    def from_json(cls, json_str: str) -> "PreregistrationTemplate":
        """Import preregistration from JSON"""
        data = json.loads(json_str)
        return cls(**data)


class DataSharingProtocol:
    """Data sharing and repository management"""

    def __init__(self, repository_path: str = "data_repository"):
        self.repository_path = Path(repository_path)
        self.repository_path.mkdir(exist_ok=True)

        # Create subdirectories
        self.raw_data_path = self.repository_path / "raw_data"
        self.processed_data_path = self.repository_path / "processed_data"
        self.metadata_path = self.repository_path / "metadata"
        self.codebooks_path = self.repository_path / "codebooks"

        for path in [
            self.raw_data_path,
            self.processed_data_path,
            self.metadata_path,
            self.codebooks_path,
        ]:
            path.mkdir(exist_ok=True)

    def _serialize_metadata(self, obj: Any) -> Any:
        """Custom JSON serializer for metadata objects"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def create_dataset_metadata(self, dataset_name: str, metadata: Dict) -> str:
        """
        Create standardized metadata for a dataset

        Args:
            dataset_name: Name of the dataset
            metadata: Dictionary containing metadata

        Returns:
            Path to created metadata file
        """
        try:
            logger.info(f"Creating metadata for dataset: {dataset_name}")

            # Input validation
            if not dataset_name or not dataset_name.strip():
                raise ValueError("Dataset name cannot be empty")
            if not isinstance(metadata, dict):
                raise ValueError("Metadata must be a dictionary")

            required_fields = [
                "title",
                "description",
                "authors",
                "creation_date",
                "data_type",
                "sample_size",
                "variables",
                "license",
            ]

            for field in required_fields:
                if field not in metadata:
                    raise ValueError(f"Required metadata field missing: {field}")

            # Validate specific fields
            if (
                not isinstance(metadata.get("sample_size"), (int, float))
                or metadata["sample_size"] <= 0
            ):
                raise ValueError("Sample size must be a positive number")

            if (
                not isinstance(metadata.get("variables"), list)
                or len(metadata["variables"]) == 0
            ):
                raise ValueError("Variables must be a non-empty list")

            # Add APGI-specific metadata
            metadata["apgi_version"] = "1.0"
            metadata["validation_protocol"] = metadata.get(
                "validation_protocol", "Unknown"
            )
            metadata["data_standard"] = "APGI_Open_Science_v1.0"

            # Create metadata file
            metadata_file = self.metadata_path / f"{dataset_name}_metadata.json"

            try:
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, default=self._serialize_metadata)
                logger.info(f"Metadata created successfully: {metadata_file}")
                return str(metadata_file)
            except Exception as e:
                logger.error(f"Failed to write metadata file: {e}")
                raise RuntimeError(f"Could not write metadata file: {str(e)}")

        except Exception as e:
            logger.error(f"Failed to create dataset metadata: {e}")
            raise

    def validate_dataset_compliance(self, dataset_path: str) -> Dict:
        """
        Validate dataset compliance with open science standards

        Args:
            dataset_path: Path to dataset

        Returns:
            Compliance validation results
        """

        compliance_checks = {
            "has_metadata": False,
            "has_codebook": False,
            "data_anonymous": False,
            "license_specified": False,
            "apgi_standard": False,
            "reproducible_code": False,
        }

        dataset_name = Path(dataset_path).stem

        # Check for metadata
        metadata_file = self.metadata_path / f"{dataset_name}_metadata.json"
        if metadata_file.exists():
            compliance_checks["has_metadata"] = True

            # Load metadata to check other fields
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            compliance_checks["license_specified"] = "license" in metadata
            compliance_checks["apgi_standard"] = (
                metadata.get("data_standard") == "APGI_Open_Science_v1.0"
            )

        # Check for codebook
        codebook_file = self.codebooks_path / f"{dataset_name}_codebook.md"
        compliance_checks["has_codebook"] = codebook_file.exists()

        # Check for analysis code
        analysis_file = (
            self.repository_path / "analysis" / f"{dataset_name}_analysis.py"
        )
        compliance_checks["reproducible_code"] = analysis_file.exists()

        # Calculate compliance score
        compliance_score = sum(compliance_checks.values()) / len(compliance_checks)

        return {
            "compliance_checks": compliance_checks,
            "compliance_score": compliance_score,
            "fully_compliant": compliance_score == 1.0,
        }


class ReplicationProtocol:
    """Replication protocol management and validation"""

    def __init__(self):
        self.replication_templates = self._load_replication_templates()

    def _load_replication_templates(self) -> Dict:
        """Load replication protocol templates"""

        return {
            "neural_signatures": {
                "required_measures": [
                    "p3b_amplitude",
                    "frontoparietal_bold",
                    "gamma_coupling",
                ],
                "sample_size_min": 20,
                "key_predictions": [
                    "p3b_scales_sigmoidally_with_precision_surprise",
                    "frontoparietal_activation_contingent_on_ignition",
                    "theta_gamma_coupling_at_threshold_crossing",
                ],
            },
            "causal_manipulations": {
                "required_measures": ["detection_rates", "p3b_amplitude", "early_erps"],
                "sample_size_min": 15,
                "key_predictions": [
                    "tms_disrupts_ignition_at_200_300ms",
                    "pharmacological_precision_modulation",
                    "metabolic_threshold_elevation",
                ],
            },
            "quantitative_fits": {
                "required_measures": ["psychometric_curves", "model_parameters"],
                "sample_size_min": 25,
                "key_predictions": [
                    "sigmoid_better_than_linear",
                    "phase_transition_dynamics",
                    "spiking_lnn_reproduces_paradigms",
                ],
            },
            "clinical_convergence": {
                "required_measures": ["neural_measures", "behavioral_measures"],
                "sample_size_min": 12,
                "key_predictions": [
                    "vegetative_state_shows_p3b_loss",
                    "psychiatric_profiles_differentiated",
                    "cross_species_homologies",
                ],
            },
        }

    def create_replication_plan(
        self, original_study: str, replication_type: str
    ) -> Dict:
        """
        Create a replication plan for a study

        Args:
            original_study: Name of original study
            replication_type: Type of replication ('direct', 'conceptual', 'extension')

        Returns:
            Replication plan dictionary
        """

        plan = {
            "original_study": original_study,
            "replication_type": replication_type,
            "creation_date": datetime.now().isoformat(),
            "replication_steps": [],
            "validation_criteria": [],
            "power_analysis": {},
            "timeline": {},
        }

        if replication_type == "direct":
            plan["replication_steps"] = [
                "Reproduce exact experimental protocol",
                "Use identical analysis pipeline",
                "Compare effect sizes and statistical significance",
                "Assess reproducibility of key findings",
            ]
            plan["validation_criteria"] = [
                "Effect sizes within 95% CI of original",
                "Consistent statistical significance",
                "No systematic deviations detected",
            ]

        elif replication_type == "conceptual":
            plan["replication_steps"] = [
                "Adapt experimental paradigm to new context",
                "Maintain core theoretical predictions",
                "Test robustness across populations/settings",
                "Compare theoretical consistency",
            ]
            plan["validation_criteria"] = [
                "Theoretical predictions replicated",
                "Effect directions consistent",
                "Robustness across contexts demonstrated",
            ]

        return plan

    def validate_replication_attempt(
        self, original_results: Dict, replication_results: Dict
    ) -> Dict:
        """
        Validate a replication attempt

        Args:
            original_results: Results from original study
            replication_results: Results from replication

        Returns:
            Replication validation results
        """

        validation_results = {
            "effect_size_similarity": self._compare_effect_sizes(
                original_results, replication_results
            ),
            "statistical_consistency": self._check_statistical_consistency(
                original_results, replication_results
            ),
            "methodological_fidelity": self._assess_methodological_fidelity(
                original_results, replication_results
            ),
            "replication_successful": False,
        }

        # Determine overall replication success
        criteria_met = (
            validation_results["effect_size_similarity"]["within_ci"]
            and validation_results["statistical_consistency"]["consistent"]
            and validation_results["methodological_fidelity"]["high_fidelity"]
        )

        validation_results["replication_successful"] = criteria_met

        return validation_results

    def _compare_effect_sizes(self, original: Dict, replication: Dict) -> Dict:
        """Compare effect sizes between original and replication"""

        # Simplified effect size comparison
        original_effect = original.get("effect_size", 0.5)
        replication_effect = replication.get("effect_size", 0.5)

        # Assume 95% CI is ±0.2 for demonstration
        ci_lower = original_effect - 0.2
        ci_upper = original_effect + 0.2

        within_ci = ci_lower <= replication_effect <= ci_upper
        deviation = abs(original_effect - replication_effect)

        return {
            "original_effect": original_effect,
            "replication_effect": replication_effect,
            "within_ci": within_ci,
            "absolute_deviation": deviation,
        }

    def _check_statistical_consistency(self, original: Dict, replication: Dict) -> Dict:
        """Check statistical consistency"""

        original_sig = original.get("p_value", 0.5) < 0.05
        replication_sig = replication.get("p_value", 0.5) < 0.05

        consistent = original_sig == replication_sig

        return {
            "original_significant": original_sig,
            "replication_significant": replication_sig,
            "consistent": consistent,
        }

    def _assess_methodological_fidelity(
        self, original: Dict, replication: Dict
    ) -> Dict:
        """Assess methodological fidelity"""

        # Simplified fidelity assessment
        fidelity_score = (
            0.85  # Placeholder - would be calculated from protocol comparison
        )

        return {"fidelity_score": fidelity_score, "high_fidelity": fidelity_score > 0.8}


class OpenAccessPublicationTemplate:
    """Open access publication templates and guidelines"""

    def __init__(self):
        self.templates = self._load_publication_templates()

    def _load_publication_templates(self) -> Dict:
        """Load publication templates"""

        return {
            "neural_validation": {
                "required_sections": [
                    "Abstract",
                    "Introduction",
                    "Methods",
                    "Results",
                    "Discussion",
                    "Data Availability",
                    "Code Availability",
                ],
                "apgi_specific_sections": [
                    "APGI Theoretical Background",
                    "Neural Signatures Tested",
                    "Convergent Evidence",
                    "Falsification Attempts",
                ],
                "reporting_standards": [
                    "Signal detection theory metrics",
                    "Neural data preprocessing details",
                    "Statistical analysis pipeline",
                    "Effect size reporting",
                ],
            },
            "clinical_validation": {
                "required_sections": [
                    "Abstract",
                    "Introduction",
                    "Methods",
                    "Results",
                    "Discussion",
                    "Data Availability",
                    "Code Availability",
                    "Clinical Trial Registration",
                ],
                "apgi_specific_sections": [
                    "Clinical Populations Tested",
                    "Cross-species Comparisons",
                    "Theoretical Convergence",
                    "Clinical Implications",
                ],
                "reporting_standards": [
                    "Patient demographic details",
                    "Clinical assessment measures",
                    "Ethical approval information",
                    "Adverse event reporting",
                ],
            },
        }

    def generate_manuscript_template(self, study_type: str, title: str) -> str:
        """
        Generate a manuscript template

        Args:
            study_type: Type of study ('neural_validation', 'clinical_validation')
            title: Manuscript title

        Returns:
            Markdown template for manuscript
        """

        if study_type not in self.templates:
            raise ValueError(f"Unknown study type: {study_type}")

        manuscript = f"""# {title}

## Abstract

[250-300 words summarizing the study, methods, results, and implications for APGI theory]

## Introduction

### Background
- Current state of consciousness science
- Position of APGI within theoretical landscape
- Specific gap this study addresses

### Research Questions and Hypotheses
- Primary research questions
- Specific APGI predictions tested
- Falsification criteria

## Methods

### Participants/Sample
- Sample size and characteristics
- Inclusion/exclusion criteria
- Power analysis

### Experimental Design
- Detailed protocol description
- APGI parameter manipulations
- Control conditions

### Data Analysis
- Preprocessing pipeline
- Statistical analysis plan
- APGI model fitting procedures

## Results

### Primary Findings
- Key results for each hypothesis
- Effect sizes and confidence intervals
- Model comparison results

### Secondary Analyses
- Exploratory findings
- Robustness checks
- Individual differences

## Discussion

### Interpretation of Findings
- What results mean for APGI theory
- Theoretical implications
- Relation to existing literature

### Limitations and Future Directions
- Study limitations
- Next steps for APGI validation

## Data and Code Availability

All data and analysis code are available at [DOI/link].

### Data Repository
- Location: [Zenodo/GitHub/Dryad link]
- License: [CC-BY-SA 4.0/MIT/etc]

### Code Repository
- Location: [GitHub link]
- License: [MIT/GPL/etc]
- Documentation: [ReadTheDocs link]

## Author Contributions

[List specific contributions of each author]

## Acknowledgments

[Funding sources, collaborators, etc.]

## References

[APA format citations]
"""

        return manuscript


class CollaborativeValidationFramework:
    """Framework for collaborative APGI validation"""

    def __init__(self):
        self.collaborative_projects = {}
        self.validation_registry = []

    def register_validation_study(self, study_info: Dict) -> str:
        """
        Register a new validation study

        Args:
            study_info: Dictionary with study information

        Returns:
            Study registration ID
        """

        required_fields = ["title", "lead_author", "institution", "validation_priority"]

        for field in required_fields:
            if field not in study_info:
                raise ValueError(f"Required field missing: {field}")

        # Generate registration ID
        registration_id = f"APGI_VAL_{len(self.validation_registry) + 1:03d}"

        study_entry = {
            "registration_id": registration_id,
            "registration_date": datetime.now().isoformat(),
            "status": "registered",
            "collaborators": [study_info["lead_author"]],
            **study_info,
        }

        self.validation_registry.append(study_entry)

        return registration_id

    def find_collaboration_opportunities(
        self, research_interests: List[str]
    ) -> List[Dict]:
        """
        Find potential collaboration opportunities

        Args:
            research_interests: List of research interests

        Returns:
            List of matching studies
        """

        opportunities = []

        for study in self.validation_registry:
            if study["status"] in ["registered", "in_progress"]:
                # Check for interest overlap
                study_interests = study.get("research_interests", [])
                overlap = set(research_interests) & set(study_interests)

                if overlap:
                    opportunities.append(
                        {
                            "study": study,
                            "interest_overlap": list(overlap),
                            "contact": study["lead_author"],
                        }
                    )

        return opportunities

    def create_collaborative_project(
        self, project_name: str, collaborators: List[str], objectives: List[str]
    ) -> str:
        """
        Create a collaborative validation project

        Args:
            project_name: Name of the collaborative project
            collaborators: List of collaborator names/emails
            objectives: List of project objectives

        Returns:
            Project ID
        """

        project_id = f"APGI_COLLAB_{len(self.collaborative_projects) + 1:03d}"

        project = {
            "project_id": project_id,
            "project_name": project_name,
            "creation_date": datetime.now().isoformat(),
            "collaborators": collaborators,
            "objectives": objectives,
            "status": "active",
            "deliverables": [],
            "timeline": {},
        }

        self.collaborative_projects[project_id] = project

        return project_id


class OpenScienceValidator:
    """Complete open science validation framework"""

    def __init__(self):
        self.data_protocol = DataSharingProtocol()
        self.replication_protocol = ReplicationProtocol()
        self.publication_template = OpenAccessPublicationTemplate()
        self.collaboration_framework = CollaborativeValidationFramework()

    def validate_open_science_compliance(self, project_path: str) -> Dict:
        """
        Comprehensive open science compliance validation

        Args:
            project_path: Path to project directory

        Returns:
            Compliance validation results
        """

        compliance_results = {
            "data_sharing": {},
            "preregistration": {},
            "replication": {},
            "open_access": {},
            "collaboration": {},
            "overall_compliance_score": 0.0,
        }

        # Check data sharing compliance
        if (Path(project_path) / "data").exists():
            datasets = list((Path(project_path) / "data").glob("*"))
            if datasets:
                compliance_results["data_sharing"] = (
                    self.data_protocol.validate_dataset_compliance(str(datasets[0]))
                )

        # Check for preregistration
        prereg_files = list(Path(project_path).glob("*prereg*.json"))
        compliance_results["preregistration"] = {
            "has_preregistration": len(prereg_files) > 0,
            "preregistration_files": [str(f) for f in prereg_files],
        }

        # Check replication documentation
        replication_files = list(Path(project_path).glob("*replication*"))
        compliance_results["replication"] = {
            "has_replication_plan": len(replication_files) > 0,
            "replication_files": [str(f) for f in replication_files],
        }

        # Check open access readiness
        readme_files = list(Path(project_path).glob("README*"))
        license_files = list(Path(project_path).glob("LICENSE*"))
        compliance_results["open_access"] = {
            "has_readme": len(readme_files) > 0,
            "has_license": len(license_files) > 0,
            "readme_files": [str(f) for f in readme_files],
            "license_files": [str(f) for f in license_files],
        }

        # Check collaboration indicators
        compliance_results["collaboration"] = {
            "registered_studies": len(self.collaboration_framework.validation_registry),
            "active_projects": len(self.collaboration_framework.collaborative_projects),
        }

        # Calculate overall compliance score
        scores = []
        if compliance_results["data_sharing"]:
            scores.append(compliance_results["data_sharing"].get("compliance_score", 0))
        scores.append(
            1.0 if compliance_results["preregistration"]["has_preregistration"] else 0
        )
        scores.append(
            1.0 if compliance_results["replication"]["has_replication_plan"] else 0
        )
        scores.append(1.0 if all(compliance_results["open_access"].values()) else 0)
        scores.append(
            min(1.0, len(self.collaboration_framework.validation_registry) / 10)
        )  # Scale with registry size

        compliance_results["overall_compliance_score"] = (
            sum(scores) / len(scores) if scores else 0
        )

        return compliance_results

    def generate_open_science_report(self, compliance_results: Dict) -> str:
        """Generate a comprehensive open science compliance report"""

        report = f"""# APGI Open Science Compliance Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Compliance Score: {compliance_results['overall_compliance_score']:.3f}/1.0

## Detailed Compliance Assessment

### Data Sharing
- Compliance Score: {compliance_results['data_sharing'].get('compliance_score', 0):.3f}
- Fully Compliant: {compliance_results['data_sharing'].get('fully_compliant', False)}

### Preregistration
- Has Preregistration: {compliance_results['preregistration']['has_preregistration']}
- Number of Preregistration Files: {len(compliance_results['preregistration']['preregistration_files'])}

### Replication
- Has Replication Plan: {compliance_results['replication']['has_replication_plan']}
- Number of Replication Files: {len(compliance_results['replication']['replication_files'])}

### Open Access
- Has README: {compliance_results['open_access']['has_readme']}
- Has License: {compliance_results['open_access']['has_license']}

### Collaboration
- Registered Studies: {compliance_results['collaboration']['registered_studies']}
- Active Projects: {compliance_results['collaboration']['active_projects']}

## Recommendations for Improvement

"""

        if not compliance_results["data_sharing"].get("fully_compliant", False):
            report += "- Complete data sharing documentation and metadata\n"

        if not compliance_results["preregistration"]["has_preregistration"]:
            report += "- Create and register preregistration protocol\n"

        if not compliance_results["replication"]["has_replication_plan"]:
            report += "- Develop replication protocol for key findings\n"

        if not all(compliance_results["open_access"].values()):
            report += "- Add README and LICENSE files\n"

        if compliance_results["collaboration"]["registered_studies"] < 5:
            report += "- Register study with collaborative validation framework\n"

        return report


def main():
    """Demonstrate open science infrastructure"""

    # Initialize framework
    validator = OpenScienceValidator()

    # Create sample preregistration
    prereg = PreregistrationTemplate(
        title="APGI Neural Signatures Validation Study",
        authors=["Research Team"],
        date_created=datetime.now().isoformat(),
        predicted_completion="2027-01-01",
        research_questions=[
            "Does P3b amplitude scale sigmoidally with APGI surprise?",
            "Is frontoparietal activation contingent on ignition?",
        ],
        hypotheses=[
            "P3b amplitude will follow sigmoidal relationship with Π×|ε|",
            "Frontoparietal BOLD will activate only when S(t) > θ_t",
        ],
        theoretical_background="APGI theory predicts discrete ignition events...",
        design_type="neural",
        paradigm="visual_masking",
        sample_size=25,
        power_analysis={"effect_size": 0.8, "power": 0.9, "alpha": 0.05},
        apgi_predictions={
            "p3b_beta": "β ≥ 10 indicating phase transition",
            "frontoparietal_threshold": "activation only above ignition threshold",
        },
        falsification_criteria=[
            "If P3b fits linear better than sigmoidal, reject APGI",
            "If frontoparietal activates subthreshold, reject APGI",
        ],
        primary_analyses=[
            "Sigmoidal model fitting to P3b data",
            "GLM analysis of fMRI data with ignition regressor",
        ],
        secondary_analyses=[
            "Individual differences analysis",
            "Cross-validation with held-out data",
        ],
        exclusion_criteria=[
            "Poor data quality (EEG artifacts > 30%)",
            "Incomplete trials (> 20% missing)",
        ],
        data_repository="https://osf.io/apgi-validation/",
        code_repository="https://github.com/apgi-research/validation-study-1",
        open_materials=True,
        open_data=True,
    )

    # Validate the preregistration template
    try:
        # Validate required fields
        if not prereg.title or not prereg.title.strip():
            raise ValueError("Title cannot be empty")
        if not prereg.authors or len(prereg.authors) == 0:
            raise ValueError("At least one author is required")
        if not prereg.research_questions or len(prereg.research_questions) == 0:
            raise ValueError("At least one research question is required")
        if not prereg.hypotheses or len(prereg.hypotheses) == 0:
            raise ValueError("At least one hypothesis is required")

        # Validate dates
        try:
            datetime.fromisoformat(prereg.date_created)
            if prereg.predicted_completion:
                datetime.fromisoformat(prereg.predicted_completion)
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")

        # Validate sample size
        if prereg.sample_size <= 0:
            raise ValueError("Sample size must be positive")

        # Validate power analysis
        if prereg.power_analysis:
            required_keys = ["effect_size", "power", "alpha"]
            for key in required_keys:
                if key not in prereg.power_analysis:
                    raise ValueError(f"Power analysis missing required key: {key}")

        logger.info(f"Preregistration template validated: {prereg.title}")

    except Exception as e:
        logger.error(f"Preregistration validation failed: {e}")
        raise

    # Validate compliance (simulated)
    compliance = validator.validate_open_science_compliance(".")

    # Generate report
    report = validator.generate_open_science_report(compliance)

    print("APGI Open Science Infrastructure Demonstration")
    print("=" * 50)
    print(f"Overall Compliance Score: {compliance['overall_compliance_score']:.3f}")
    print("\nGenerated Report Preview:")
    print(report[:500] + "...")

    # Save preregistration
    with open("sample_preregistration.json", "w", encoding="utf-8") as f:
        f.write(prereg.to_json())

    print("\nSample preregistration saved to: sample_preregistration.json")


if __name__ == "__main__":
    main()
