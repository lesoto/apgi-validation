"""
Seven Standards Registry for APGI Validation

This module provides a centralized registry for the seven standards that APGI must meet
for empirical validation and falsification criteria. These standards define the
requirements for APGI to be considered a scientifically valid theory of consciousness.

The seven standards are:
1. Mathematical Consistency
2. Empirical Testability
3. Computational Tractability
4. Neurobiological Plausibility
5. Predictive Novelty
6. Cross-Species Generalizability
7. Clinical Applicability

This registry provides detailed criteria, falsification thresholds, and validation
methods for each standard.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class StandardCategory(Enum):
    """Categories of validation standards"""

    MATHEMATICAL = "Mathematical Consistency"
    EMPIRICAL = "Empirical Testability"
    COMPUTATIONAL = "Computational Tractability"
    NEUROBIOLOGICAL = "Neurobiological Plausibility"
    PREDICTIVE = "Predictive Novelty"
    CROSS_SPECIES = "Cross-Species Generalizability"
    CLINICAL = "Clinical Applicability"


@dataclass
class ValidationCriterion:
    """A specific validation criterion within a standard"""

    name: str
    description: str
    falsification_threshold: str
    test_method: str
    effect_size_requirement: str
    alternative_criterion: str
    priority_level: str  # "Critical", "High", "Medium", "Low"
    current_status: str  # "Implemented", "Partial", "Proposed", "Not Started"


@dataclass
class Standard:
    """A complete validation standard with multiple criteria"""

    standard_id: str
    name: str
    category: StandardCategory
    description: str
    rationale: str
    criteria: List[ValidationCriterion]
    overall_status: str
    implementation_notes: Optional[str] = None


class SevenStandardsRegistry:
    """
    Registry for the seven APGI validation standards.

    Provides centralized access to all validation criteria, falsification
    thresholds, and implementation status for each standard.
    """

    _standards: Dict[str, Standard] = {}

    @classmethod
    def register_standard(cls, standard: Standard) -> None:
        """Register a validation standard."""
        cls._standards[standard.standard_id] = standard

    @classmethod
    def get_standard(cls, standard_id: str) -> Optional[Standard]:
        """Get a standard by ID."""
        return cls._standards.get(standard_id)

    @classmethod
    def list_standards(
        cls, category: Optional[StandardCategory] = None
    ) -> List[Standard]:
        """List all standards, optionally filtered by category."""
        standards = list(cls._standards.values())

        if category:
            standards = [s for s in standards if s.category == category]

        return sorted(standards, key=lambda s: s.standard_id)

    @classmethod
    def get_criteria_by_priority(cls, priority: str) -> List[ValidationCriterion]:
        """Get all criteria with specified priority level."""
        criteria = []
        for standard in cls._standards.values():
            criteria.extend(
                [c for c in standard.criteria if c.priority_level == priority]
            )
        return criteria

    @classmethod
    def get_falsification_summary(cls) -> Dict[str, Dict]:
        """Get summary of falsification criteria for all standards."""
        summary = {}
        for standard_id, standard in cls._standards.items():
            summary[standard_id] = {
                "name": standard.name,
                "category": standard.category.value,
                "overall_status": standard.overall_status,
                "criteria_count": len(standard.criteria),
                "critical_criteria": len(
                    [c for c in standard.criteria if c.priority_level == "Critical"]
                ),
                "implemented_criteria": len(
                    [c for c in standard.criteria if "Implemented" in c.current_status]
                ),
            }
        return summary


def _initialize_standards():
    """Initialize the seven standards registry with all criteria."""

    # Standard 1: Mathematical Consistency
    standard_1 = Standard(
        standard_id="S1",
        name="Mathematical Consistency",
        category=StandardCategory.MATHEMATICAL,
        description="APGI equations must be mathematically consistent and well-posed",
        rationale="Theoretical frameworks must avoid contradictions and maintain internal logical consistency",
        criteria=[
            ValidationCriterion(
                name="ODE Well-Posedness",
                description="All differential equations must have unique solutions",
                falsification_threshold="No singularities or undefined expressions in parameter ranges",
                test_method="Analytical verification of Jacobian matrix rank",
                effect_size_requirement="Condition number < 1000 across parameter space",
                alternative_criterion="Falsified if singularities detected OR condition number > 1000",
                priority_level="Critical",
                current_status="Implemented",
            ),
            ValidationCriterion(
                name="Energy Conservation",
                description="Information flow must respect thermodynamic constraints",
                falsification_threshold="ΔE_total ≤ 5% over 1000 timesteps",
                test_method="Energy balance tracking in simulations",
                effect_size_requirement="Energy drift < 0.05 per timestep",
                alternative_criterion="Falsified if energy drift > 0.1 per timestep OR net energy creation",
                priority_level="High",
                current_status="Implemented",
            ),
            ValidationCriterion(
                name="Parameter Identifiability",
                description="Parameters must be uniquely identifiable from data",
                falsification_threshold="Fisher information matrix full rank",
                test_method="Parameter recovery analysis with synthetic data",
                effect_size_requirement="Parameter recovery error < 10% with realistic noise",
                alternative_criterion="Falsified if parameter correlations > 0.9 OR recovery error > 20%",
                priority_level="High",
                current_status="Implemented",
            ),
        ],
        overall_status="Implemented",
    )

    # Standard 2: Empirical Testability
    standard_2 = Standard(
        standard_id="S2",
        name="Empirical Testability",
        category=StandardCategory.EMPIRICAL,
        description="APGI predictions must be empirically testable with existing methods",
        rationale="Scientific theories must make falsifiable predictions that can be tested with current technology",
        criteria=[
            ValidationCriterion(
                name="Neural Correlates",
                description="Core variables must map to measurable neural signals",
                falsification_threshold="≥70% of variables have established neural correlates",
                test_method="Literature review + experimental validation",
                effect_size_requirement="Effect sizes d ≥ 0.5 for established correlates",
                alternative_criterion="Falsified if <50% have neural correlates OR effect sizes d < 0.3",
                priority_level="Critical",
                current_status="Implemented",
            ),
            ValidationCriterion(
                name="Behavioral Predictions",
                description="Model must predict observable behavioral phenomena",
                falsification_threshold="Accuracy ≥ 75% on benchmark tasks",
                test_method="Cross-validation with behavioral datasets",
                effect_size_requirement="Improvement over baseline models ≥ 10%",
                alternative_criterion="Falsified if accuracy < 60% OR no improvement over baseline",
                priority_level="High",
                current_status="Partial",
            ),
            ValidationCriterion(
                name="Temporal Dynamics",
                description="Time constants must match empirical measurements",
                falsification_threshold="τ_estimated / τ_empirical ∈ [0.5, 2.0]",
                test_method="Time series analysis comparison",
                effect_size_requirement="Correlation r ≥ 0.7 with empirical dynamics",
                alternative_criterion="Falsified if correlation r < 0.5 OR ratio outside [0.25, 4.0]",
                priority_level="High",
                current_status="Implemented",
            ),
        ],
        overall_status="Implemented",
    )

    # Standard 3: Computational Tractability
    standard_3 = Standard(
        standard_id="S3",
        name="Computational Tractability",
        category=StandardCategory.COMPUTATIONAL,
        description="APGI must be computationally feasible for realistic applications",
        rationale="Theories must be implementable and testable with available computational resources",
        criteria=[
            ValidationCriterion(
                name="Algorithmic Efficiency",
                description="Computations must scale polynomially with problem size",
                falsification_threshold="O(n^k) with k ≤ 3 for all core operations",
                test_method="Computational complexity analysis",
                effect_size_requirement="Runtime < 1 hour for standard datasets",
                alternative_criterion="Falsified if exponential scaling OR runtime > 24 hours",
                priority_level="High",
                current_status="Implemented",
            ),
            ValidationCriterion(
                name="Memory Requirements",
                description="Memory usage must be reasonable for target applications",
                falsification_threshold="Memory < 16GB for standard simulations",
                test_method="Memory profiling during execution",
                effect_size_requirement="Memory scaling O(n) or O(n log n)",
                alternative_criterion="Falsified if memory > 32GB OR O(n²) scaling",
                priority_level="Medium",
                current_status="Implemented",
            ),
            ValidationCriterion(
                name="Numerical Stability",
                description="Algorithms must be numerically stable in realistic conditions",
                falsification_threshold="No divergence for noise levels up to 20%",
                test_method="Stress testing with noisy inputs",
                effect_size_requirement="Condition numbers < 10^6 throughout parameter space",
                alternative_criterion="Falsified if divergence at <5% noise OR condition numbers > 10^8",
                priority_level="High",
                current_status="Implemented",
            ),
        ],
        overall_status="Implemented",
    )

    # Standard 4: Neurobiological Plausibility
    standard_4 = Standard(
        standard_id="S4",
        name="Neurobiological Plausibility",
        category=StandardCategory.NEUROBIOLOGICAL,
        description="APGI mechanisms must be consistent with known neurobiology",
        rationale="Theories of consciousness must align with established neuroscientific knowledge",
        criteria=[
            ValidationCriterion(
                name="Anatomical Consistency",
                description="Network structure must respect known neuroanatomy",
                falsification_threshold="≥80% of connections have anatomical basis",
                test_method="Comparison with connectome data",
                effect_size_requirement="Structural similarity metrics ≥ 0.7",
                alternative_criterion="Falsified if <60% anatomical consistency OR similarity < 0.5",
                priority_level="High",
                current_status="Partial",
            ),
            ValidationCriterion(
                name="Physiological Constraints",
                description="Parameters must respect physiological limits",
                falsification_threshold="All parameters within established biological ranges",
                test_method="Parameter validation against literature",
                effect_size_requirement="No violations of established constraints",
                alternative_criterion="Falsified if any parameter outside 2σ of biological mean",
                priority_level="Critical",
                current_status="Implemented",
            ),
            ValidationCriterion(
                name="Neurotransmitter Plausibility",
                description="Neuromodulation must follow known neurotransmitter dynamics",
                falsification_threshold="Time constants within factor 2 of empirical values",
                test_method="Comparison with pharmacological data",
                effect_size_requirement="Correlation r ≥ 0.6 with drug effects",
                alternative_criterion="Falsified if time constants differ by factor > 5 OR no correlation",
                priority_level="Medium",
                current_status="Proposed",
            ),
        ],
        overall_status="Partial",
    )

    # Standard 5: Predictive Novelty
    standard_5 = Standard(
        standard_id="S5",
        name="Predictive Novelty",
        category=StandardCategory.PREDICTIVE,
        description="APGI must generate novel, testable predictions",
        rationale="Valuable theories make predictions that go beyond existing knowledge",
        criteria=[
            ValidationCriterion(
                name="Novel Phenomena",
                description="Must predict phenomena not explained by existing theories",
                falsification_threshold="≥3 novel predictions with no existing explanations",
                test_method="Literature review + experimental validation",
                effect_size_requirement="Effect sizes significantly larger than competing theories",
                alternative_criterion="Falsified if <1 novel prediction OR all explanations exist",
                priority_level="High",
                current_status="Implemented",
            ),
            ValidationCriterion(
                name="Quantitative Precision",
                description="Predictions must be quantitative with specified precision",
                falsification_threshold="Prediction error < 20% for all novel predictions",
                test_method="Experimental validation with error analysis",
                effect_size_requirement="95% confidence intervals exclude competing theories",
                alternative_criterion="Falsified if error > 30% OR confidence intervals overlap competing theories",
                priority_level="High",
                current_status="Partial",
            ),
            ValidationCriterion(
                name="Cross-Domain Generalization",
                description="Predictions must generalize across multiple domains",
                falsification_threshold="Validated in ≥2 different domains (e.g., perception, action)",
                test_method="Multi-domain experimental validation",
                effect_size_requirement="Consistent effect sizes across domains (CV < 30%)",
                alternative_criterion="Falsified if only works in single domain OR high variability",
                priority_level="Medium",
                current_status="Proposed",
            ),
        ],
        overall_status="Partial",
    )

    # Standard 6: Cross-Species Generalizability
    standard_6 = Standard(
        standard_id="S6",
        name="Cross-Species Generalizability",
        category=StandardCategory.CROSS_SPECIES,
        description="APGI principles must apply across different species",
        rationale="Fundamental principles of consciousness should not be species-specific",
        criteria=[
            ValidationCriterion(
                name="Scaling Laws",
                description="Parameters must follow known allometric scaling",
                falsification_threshold="Exponent within 95% CI of empirical scaling laws",
                test_method="Cross-species parameter comparison",
                effect_size_requirement="R² ≥ 0.8 for scaling relationships",
                alternative_criterion="Falsified if exponents outside CI OR poor fit (R² < 0.6)",
                priority_level="High",
                current_status="Implemented",
            ),
            ValidationCriterion(
                name="Phylogenetic Consistency",
                description="Predictions must be consistent with phylogenetic relationships",
                falsification_threshold="Closely related species show similar APGI parameters",
                test_method="Comparative analysis across species",
                effect_size_requirement="Phylogenetic signal detected (p < 0.05)",
                alternative_criterion="Falsified if no phylogenetic signal OR random parameter distribution",
                priority_level="Medium",
                current_status="Partial",
            ),
            ValidationCriterion(
                name="Developmental Consistency",
                description="APGI development must follow known developmental trajectories",
                falsification_threshold="Parameters follow developmental stages consistently",
                test_method="Developmental time course analysis",
                effect_size_requirement="Monotonic or stage-appropriate changes",
                alternative_criterion="Falsified if inconsistent developmental patterns OR contradictions",
                priority_level="Medium",
                current_status="Proposed",
            ),
        ],
        overall_status="Partial",
    )

    # Standard 7: Clinical Applicability
    standard_7 = Standard(
        standard_id="S7",
        name="Clinical Applicability",
        category=StandardCategory.CLINICAL,
        description="APGI must have clinical relevance and applications",
        rationale="Theories of consciousness should contribute to clinical understanding and treatment",
        criteria=[
            ValidationCriterion(
                name="Diagnostic Utility",
                description="Must improve clinical diagnosis or classification",
                falsification_threshold="AUC ≥ 0.80 for disorder classification",
                test_method="Clinical validation with patient data",
                effect_size_requirement="Improvement over existing methods (ΔAUC ≥ 0.05)",
                alternative_criterion="Falsified if AUC < 0.75 OR no improvement over baseline",
                priority_level="Critical",
                current_status="Implemented",
            ),
            ValidationCriterion(
                name="Treatment Predictions",
                description="Must predict treatment response or outcomes",
                falsification_threshold="Treatment response prediction accuracy ≥ 70%",
                test_method="Prospective clinical prediction studies",
                effect_size_requirement="Significant prediction of clinical outcomes (p < 0.01)",
                alternative_criterion="Falsified if accuracy < 60% OR no predictive value",
                priority_level="High",
                current_status="Proposed",
            ),
            ValidationCriterion(
                name="Safety and Ethics",
                description="Clinical applications must be safe and ethical",
                falsification_threshold="No adverse effects from APGI-based interventions",
                test_method="Safety monitoring and ethical review",
                effect_size_requirement="Risk-benefit ratio favorable",
                alternative_criterion="Falsified if safety concerns OR ethical violations",
                priority_level="Critical",
                current_status="Not Started",
            ),
        ],
        overall_status="Partial",
    )

    # Register all standards
    SevenStandardsRegistry.register_standard(standard_1)
    SevenStandardsRegistry.register_standard(standard_2)
    SevenStandardsRegistry.register_standard(standard_3)
    SevenStandardsRegistry.register_standard(standard_4)
    SevenStandardsRegistry.register_standard(standard_5)
    SevenStandardsRegistry.register_standard(standard_6)
    SevenStandardsRegistry.register_standard(standard_7)


# Initialize the registry
_initialize_standards()


def get_seven_standards_registry() -> SevenStandardsRegistry:
    """Get the seven standards registry instance."""
    return SevenStandardsRegistry


# Convenience functions
def get_all_standards() -> List[Standard]:
    """Get all seven standards."""
    return SevenStandardsRegistry.list_standards()


def get_standard_by_id(standard_id: str) -> Optional[Standard]:
    """Get a specific standard by ID (S1-S7)."""
    return SevenStandardsRegistry.get_standard(standard_id)


def get_falsification_checklist() -> Dict[str, List[str]]:
    """Get a checklist of all falsification criteria."""
    checklist = {}
    for standard_id, standard in SevenStandardsRegistry._standards.items():
        checklist[standard_id] = [c.name for c in standard.criteria]
    return checklist


def get_implementation_status() -> Dict[str, str]:
    """Get current implementation status for all standards."""
    return {
        sid: std.overall_status
        for sid, std in SevenStandardsRegistry._standards.items()
    }


def get_critical_criteria() -> List[ValidationCriterion]:
    """Get all critical-priority falsification criteria."""
    return SevenStandardsRegistry.get_criteria_by_priority("Critical")


def generate_validation_report() -> str:
    """Generate a comprehensive validation status report."""
    report_lines = ["APGI SEVEN STANDARDS VALIDATION REPORT", "=" * 50, ""]

    summary = SevenStandardsRegistry.get_falsification_summary()

    for standard_id, info in summary.items():
        standard = SevenStandardsRegistry.get_standard(standard_id)
        report_lines.extend(
            [
                f"Standard {standard_id}: {info['name']}",
                f"Category: {info['category']}",
                f"Status: {info['overall_status']}",
                f"Criteria: {info['criteria_count']} total, {info['critical_criteria']} critical",
                f"Implemented: {info['implemented_criteria']}/{info['criteria_count']}",
                "",
            ]
        )

        if standard.implementation_notes:
            report_lines.append(f"Notes: {standard.implementation_notes}")
            report_lines.append("")

    return "\n".join(report_lines)
