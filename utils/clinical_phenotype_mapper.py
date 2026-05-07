"""
================================================================================
APGI Clinical Phenotype Simulation Mapper
================================================================================

Comprehensive mapping of DSM-5 disorders to APGI precision profiles with
hierarchical level-specific clinical predictions.

This module operationalizes the Disorder-Specific Precision Profiles described
in the APGI Empirical Credibility Roadmap, providing:

- Phenotype simulation for 60+ DSM-5 disorders
- Hierarchical level-specific clinical predictions
- APGI parameter profiles (prediction error, precision, threshold, somatic bias)
- Neural signature mappings
- Treatment implication frameworks

Disorder Hierarchy Mapping:
- Level 1: Sensory/perceptual (Panic, Tic disorders, Stereotypic movement)
- Level 2: Social/emotional (Social Anxiety, BPD, Attachment disorders)
- Level 3: Cognitive/semantic (GAD, OCD, Delusional disorder)
- Level 4: Narrative/state (Insomnia, Dissociative disorders, Consciousness)
- Multi-level: Global dysregulation (Schizophrenia, ASD, ADHD)
- Level coupling: Specific cross-level disruptions (PTSD)

Usage::

    from utils.clinical_phenotype_mapper import ClinicalPhenotypeMapper

    mapper = ClinicalPhenotypeMapper()

    # Get complete profile for a disorder
    profile = mapper.get_disorder_profile('panic-disorder')

    # Simulate phenotype for a specific disorder
    simulation = mapper.simulate_phenotype('generalized-anxiety-disorder', n_trials=100)

    # Compare disorders at specific hierarchical levels
    comparison = mapper.compare_level_profiles(level=2)

================================================================================
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


class HierarchicalLevel(Enum):
    """APGI Hierarchical Processing Levels"""

    LEVEL_1 = 1  # Sensory/perceptual processing
    LEVEL_2 = 2  # Social/emotional processing
    LEVEL_3 = 3  # Cognitive/semantic processing
    LEVEL_4 = 4  # Narrative/state processing
    GLOBAL = "global"  # Multi-level dysregulation
    COUPLING = "coupling"  # Cross-level coupling disruption


@dataclass
class APGIParameterProfile:
    """APGI quantitative parameter profile for a disorder"""

    # Prediction error magnitude (ε)
    epsilon_mean: float = 0.0
    epsilon_sigma_range: Tuple[float, float] = (0.0, 0.0)

    # Precision weighting (Π)
    pi_exteroceptive: float = 1.0  # Πₑ - external precision
    pi_interoceptive: float = 1.0  # Πᵢ - internal precision
    pi_dominant_channels: List[str] = field(default_factory=list)

    # Threshold deviation (θₜ)
    theta_t_percent: float = 0.0  # Percentage deviation from healthy
    theta_t_absolute: float = 0.5

    # Somatic bias (β)
    somatic_bias: float = 1.0

    # Hierarchical specificity
    primary_affected_level: Union[int, str] = 1
    cross_level_coupling: Dict[str, float] = field(default_factory=dict)


@dataclass
class NeuralSignature:
    """Neural signature profile for empirical validation"""

    eeg_markers: List[str] = field(default_factory=list)
    fmri_connectivity: List[str] = field(default_factory=list)
    structural_changes: List[str] = field(default_factory=list)
    autonomic_markers: List[str] = field(default_factory=list)


@dataclass
class TreatmentImplication:
    """Treatment implications based on APGI profile"""

    first_line: List[str] = field(default_factory=list)
    mechanism_targets: Dict[str, str] = field(default_factory=dict)
    precision_modulation: Dict[str, float] = field(default_factory=dict)


@dataclass
class DisorderPhenotype:
    """Complete phenotype specification for a disorder"""

    dsm5_code: str = ""
    category: str = ""
    description: str = ""
    apgi_profile: APGIParameterProfile = field(default_factory=APGIParameterProfile)
    neural_signatures: NeuralSignature = field(default_factory=NeuralSignature)
    treatment_implications: TreatmentImplication = field(
        default_factory=TreatmentImplication
    )
    mechanism: str = ""
    phenomenology: str = ""
    citations: List[str] = field(default_factory=list)


class ClinicalPhenotypeMapper:
    """
    Maps clinical disorders to APGI phenotype simulations.

    Provides comprehensive mappings for 60+ DSM-5 disorders with:
    - Disorder-specific precision profiles
    - Hierarchical level-specific predictions
    - Neural signature mappings
    - Treatment targeting
    """

    # Core APGI parameter table - Disorders & Conditions
    # Based on empirical literature and theoretical predictions
    DISORDER_PARAMETER_TABLE: Dict[str, Dict[str, Any]] = {
        # === NEURODEVELOPMENTAL DISORDERS ===
        "childhood-onset-fluency-disorder": {
            "dsm5_code": "F80.81",
            "category": "neurodevelopmental",
            "prediction_error": (2, 4),
            "prediction_error_type": "speech-motor",
            "precision_profile": {"motor_speech": "increased"},
            "theta_t_range": (-15, -25),
            "somatic_bias_range": (1.2, 1.5),
            "level": 1,
            "mechanism": "Speech-motor prediction error with elevated motor-speech precision",
        },
        "language-disorder": {
            "dsm5_code": "F80.9",
            "category": "neurodevelopmental",
            "prediction_error": (2, 4),
            "prediction_error_type": "linguistic",
            "precision_profile": {"language": "decreased"},
            "theta_t_range": (10, 25),
            "somatic_bias_range": (1.0, 1.0),
            "level": 3,
            "mechanism": "Reduced linguistic precision with elevated threshold for language processing",
        },
        "social-pragmatic-communication-disorder": {
            "dsm5_code": "F80.89",
            "category": "neurodevelopmental",
            "prediction_error": (3, 5),
            "prediction_error_type": "social-inference",
            "precision_profile": {"social": "decreased"},
            "theta_t_range": (10, 20),
            "somatic_bias_range": (0.9, 1.1),
            "level": 2,
            "mechanism": "Reduced social precision for pragmatic inference",
        },
        "speech-sound-disorder": {
            "dsm5_code": "F80.0",
            "category": "neurodevelopmental",
            "prediction_error": (2, 4),
            "prediction_error_type": "phonological",
            "precision_profile": {"motor_speech": "increased"},
            "theta_t_range": (-10, 20),
            "somatic_bias_range": (1.1, 1.4),
            "level": 1,
            "mechanism": "Phonological prediction error with elevated motor-speech precision",
        },
        "intellectual-disability": {
            "dsm5_code": "F79",
            "category": "neurodevelopmental",
            "prediction_error": (-2, -4),
            "prediction_error_type": "global",
            "precision_profile": {"all": "decreased"},
            "theta_t_range": (20, 40),
            "somatic_bias_range": (1.0, 1.0),
            "level": "global",
            "mechanism": "Global reduced precision across all processing levels",
        },
        "global-developmental-delay": {
            "dsm5_code": "F88",
            "category": "neurodevelopmental",
            "prediction_error": (-2, -3),
            "prediction_error_type": "global",
            "precision_profile": {"all": "decreased"},
            "theta_t_range": (20, 40),
            "somatic_bias_range": (1.0, 1.0),
            "level": "global",
            "mechanism": "Global reduced precision across developmental domains",
        },
        "developmental-coordination-disorder": {
            "dsm5_code": "F82",
            "category": "neurodevelopmental",
            "prediction_error": (3, 5),
            "prediction_error_type": "motor",
            "precision_profile": {"motor": "increased"},
            "theta_t_range": (-15, 25),
            "somatic_bias_range": (1.3, 1.7),
            "level": 1,
            "mechanism": "Motor prediction error with mismatched motor precision",
        },
        "stereotypic-movement-disorder": {
            "dsm5_code": "F98.4",
            "category": "neurodevelopmental",
            "prediction_error": (2, 4),
            "prediction_error_type": "motor-loop",
            "precision_profile": {"motor": "greatly_increased"},
            "theta_t_range": (-30, -50),
            "somatic_bias_range": (1.5, 2.5),
            "level": 1,
            "mechanism": "Self-reinforcing motor loops with pathologically elevated motor precision",
        },
        "adhd": {
            "dsm5_code": "F90.x",
            "category": "neurodevelopmental",
            "prediction_error": (2, 4),
            "prediction_error_type": "task-inconsistency",
            "precision_profile": {"executive": "decreased", "unstable": True},
            "theta_t_range": (-20, -40),
            "somatic_bias_range": (0.8, 1.2),
            "level": "global",
            "mechanism": "NE dysregulation → fluctuating gain; unstable executive precision",
        },
        "autism-spectrum-disorder": {
            "dsm5_code": "F84.x",
            "category": "neurodevelopmental",
            "prediction_error": (3, 6),
            "prediction_error_type": "social",
            "precision_profile": {"sensory": "increased", "social": "decreased"},
            "theta_t_range": (15, 35),
            "somatic_bias_range": (0.7, 1.1),
            "level": "multi-level",
            "mechanism": "Inflexible Π; high Πₑ for detail, low Πₑ for social; multi-level rigidity",
        },
        "specific-learning-disorder": {
            "dsm5_code": "F81.x",
            "category": "neurodevelopmental",
            "prediction_error": (3, 5),
            "prediction_error_type": "domain-specific",
            "precision_profile": {"domain": "decreased"},
            "theta_t_range": (10, 25),
            "somatic_bias_range": (1.0, 1.0),
            "level": 3,
            "mechanism": "Domain-specific precision reduction with elevated threshold",
        },
        # === SCHIZOPHRENIA SPECTRUM ===
        "brief-psychotic-disorder": {
            "dsm5_code": "F23",
            "category": "schizophrenia_spectrum",
            "prediction_error": (4, 7),
            "prediction_error_type": "delusional",
            "precision_profile": {"aberrant": "increased"},
            "theta_t_range": (-40, -60),
            "somatic_bias_range": (1.2, 2.0),
            "level": "global",
            "mechanism": "Transient aberrant salience with collapsed threshold",
        },
        "delusional-disorder": {
            "dsm5_code": "F22",
            "category": "schizophrenia_spectrum",
            "prediction_error": (3, 5),
            "prediction_error_type": "fixed-belief",
            "precision_profile": {"belief": "greatly_increased"},
            "theta_t_range": (-30, -45),
            "somatic_bias_range": (1.0, 1.4),
            "level": 3,
            "mechanism": "Fixed high-precision delusional priors resistant to updating",
        },
        "schizophrenia": {
            "dsm5_code": "F20.x",
            "category": "schizophrenia_spectrum",
            "prediction_error": (4, 8),
            "prediction_error_type": "internal",
            "precision_profile": {"sensory": "decreased", "internal": "increased"},
            "theta_t_range": (-50, -70),
            "somatic_bias_range": (0.8, 1.3),
            "level": "global",
            "mechanism": "Aberrant Π; DA tags noise as salient; internal priors dominate",
        },
        "schizoaffective-disorder": {
            "dsm5_code": "F25.x",
            "category": "schizophrenia_spectrum",
            "prediction_error": (4, 7),
            "prediction_error_type": "mood-psychotic",
            "precision_profile": {"alternating": True},
            "theta_t_range": (-40, -60),
            "somatic_bias_range": (1.0, 2.0),
            "level": "global",
            "mechanism": "Alternating mood and psychotic precision profiles",
        },
        "schizotypal-personality-disorder": {
            "dsm5_code": "F21",
            "category": "schizophrenia_spectrum",
            "prediction_error": (3, 5),
            "prediction_error_type": "magical-ideation",
            "precision_profile": {"associative": "increased"},
            "theta_t_range": (-20, -35),
            "somatic_bias_range": (0.8, 1.2),
            "level": 3,
            "mechanism": "Elevated associative precision with magical ideation",
        },
        # === BIPOLAR DISORDERS ===
        "bipolar-i": {
            "dsm5_code": "F31.x",
            "category": "bipolar",
            "prediction_error": (-2, 2),
            "prediction_error_type": "mania-depression",
            "precision_profile": {"reward": "greatly_increased"},
            "theta_t_range": (-40, -60),
            "somatic_bias_range": (0.7, 1.1),
            "level": "state_dependent",
            "mechanism": "Mania: -2σ error, Reward ↑↑, collapsed threshold; variable by state",
        },
        "bipolar-ii": {
            "dsm5_code": "F31.81",
            "category": "bipolar",
            "prediction_error": (-2, 2),
            "prediction_error_type": "hypomanic-oscillation",
            "precision_profile": {"reward": "increased"},
            "theta_t_range": (-25, -40),
            "somatic_bias_range": (0.8, 1.2),
            "level": "state_dependent",
            "mechanism": "Hypomanic oscillation with elevated reward precision",
        },
        "cyclothymia": {
            "dsm5_code": "F34.0",
            "category": "bipolar",
            "prediction_error": (-2, 2),
            "prediction_error_type": "chronic-oscillation",
            "precision_profile": {"reward": "unstable"},
            "theta_t_range": (-20, 30),
            "somatic_bias_range": (1.0, 1.0),
            "level": "state_dependent",
            "mechanism": "Chronic oscillation with unstable reward precision",
        },
        # === DEPRESSIVE DISORDERS ===
        "major-depressive-disorder": {
            "dsm5_code": "F33.x",
            "category": "depressive",
            "prediction_error": (-3, -5),
            "prediction_error_type": "reward",
            "precision_profile": {"positive": "decreased"},
            "theta_t_range": (30, 50),
            "somatic_bias_range": (0.7, 0.9),
            "level": 2,
            "mechanism": "Blunted interoception; high threshold; reduced reward precision",
        },
        "persistent-depressive-disorder": {
            "dsm5_code": "F34.1",
            "category": "depressive",
            "prediction_error": (-2, -4),
            "prediction_error_type": "reward",
            "precision_profile": {"positive": "decreased"},
            "theta_t_range": (25, 40),
            "somatic_bias_range": (0.8, 1.0),
            "level": 2,
            "mechanism": "Chronically reduced reward precision with moderately elevated threshold",
        },
        "pmdd": {
            "dsm5_code": "N94.3",
            "category": "depressive",
            "prediction_error": (2, 4),
            "prediction_error_type": "hormonal",
            "precision_profile": {"threat": "increased"},
            "theta_t_range": (-15, -30),
            "somatic_bias_range": (1.4, 2.0),
            "level": 2,
            "mechanism": "Cyclic threat precision elevation with hormonal modulation",
        },
        # === ANXIETY DISORDERS ===
        "generalized-anxiety-disorder": {
            "dsm5_code": "F41.1",
            "category": "anxiety",
            "prediction_error": (3, 6),
            "prediction_error_type": "threat",
            "precision_profile": {"threat": "greatly_increased"},
            "theta_t_range": (-25, -40),
            "somatic_bias_range": (1.8, 3.0),
            "level": 3,
            "mechanism": "High-confidence threat priors; Level 3 elevation; worry as cognitive avoidance",
        },
        "panic-disorder": {
            "dsm5_code": "F41.0",
            "category": "anxiety",
            "prediction_error": (4, 7),
            "prediction_error_type": "interoceptive",
            "precision_profile": {"body": "greatly_increased"},
            "theta_t_range": (-40, -60),
            "somatic_bias_range": (2.5, 4.0),
            "level": 1,
            "mechanism": "↑Πᵢ & ↓θₜ; Level 1 threshold collapse; somatic priors → catastrophe",
        },
        "social-anxiety-disorder": {
            "dsm5_code": "F40.10",
            "category": "anxiety",
            "prediction_error": (3, 5),
            "prediction_error_type": "social-threat",
            "precision_profile": {"social": "increased"},
            "theta_t_range": (-25, -40),
            "somatic_bias_range": (1.5, 2.5),
            "level": 2,
            "mechanism": "Context-specific elevation of Πᵢ and Πₑ during social evaluation",
        },
        "specific-phobia": {
            "dsm5_code": "F40.x",
            "category": "anxiety",
            "prediction_error": (4, 6),
            "prediction_error_type": "object-threat",
            "precision_profile": {"threat": "increased"},
            "theta_t_range": (-40, -60),
            "somatic_bias_range": (2.0, 3.5),
            "level": 1,
            "mechanism": "Object-specific threat precision with acute threshold reduction",
        },
        # === OBSESSIVE-COMPULSIVE ===
        "obsessive-compulsive-disorder": {
            "dsm5_code": "F42.x",
            "category": "obsessive_compulsive",
            "prediction_error": (3, 5),
            "prediction_error_type": "intrusive",
            "precision_profile": {"error_monitoring": "increased"},
            "theta_t_range": (-30, -45),
            "somatic_bias_range": (1.5, 2.5),
            "level": "3-4",
            "mechanism": "↑ Precision on error-monitoring; Level 3-4 elevation; compulsions reduce prediction error",
        },
        "hoarding-disorder": {
            "dsm5_code": "F42.3",
            "category": "obsessive_compulsive",
            "prediction_error": (3, 5),
            "prediction_error_type": "loss-prediction",
            "precision_profile": {"value": "increased"},
            "theta_t_range": (-25, -40),
            "somatic_bias_range": (1.2, 1.6),
            "level": 3,
            "mechanism": "Elevated precision on object value with loss aversion",
        },
        "body-dysmorphic-disorder": {
            "dsm5_code": "F45.22",
            "category": "obsessive_compulsive",
            "prediction_error": (4, 6),
            "prediction_error_type": "appearance",
            "precision_profile": {"visual_self": "increased"},
            "theta_t_range": (-30, -45),
            "somatic_bias_range": (1.8, 3.0),
            "level": 1,
            "mechanism": "Hyper-precise visual self-representation with appearance focus",
        },
        "trichotillomania": {
            "dsm5_code": "F63.3",
            "category": "obsessive_compulsive",
            "prediction_error": (3, 5),
            "prediction_error_type": "urge",
            "precision_profile": {"motor": "increased"},
            "theta_t_range": (-30, -45),
            "somatic_bias_range": (2.0, 3.0),
            "level": 1,
            "mechanism": "Elevated motor precision with urge-driven motor patterns",
        },
        # === TRAUMA/STRESS ===
        "ptsd": {
            "dsm5_code": "F43.10",
            "category": "trauma_stress",
            "prediction_error": (4, 7),
            "prediction_error_type": "threat-memory",
            "precision_profile": {"threat": "greatly_increased"},
            "theta_t_range": (-40, -60),
            "somatic_bias_range": (2.0, 3.5),
            "level": "2-3_coupling",
            "mechanism": "↑Πᵢ trauma; Level 2-3 coupling disruption; trauma bypasses narrative integration",
        },
        "acute-stress-disorder": {
            "dsm5_code": "F43.0",
            "category": "trauma_stress",
            "prediction_error": (3, 6),
            "prediction_error_type": "threat",
            "precision_profile": {"threat": "increased"},
            "theta_t_range": (-30, -50),
            "somatic_bias_range": (1.8, 3.0),
            "level": "2-3_coupling",
            "mechanism": "Acute threat precision elevation with partial coupling disruption",
        },
        # === DISSOCIATIVE ===
        "dissociative-identity-disorder": {
            "dsm5_code": "F44.81",
            "category": "dissociative",
            "prediction_error": (0, 0),
            "prediction_error_type": "self-model-split",
            "precision_profile": {"internal": "greatly_increased"},
            "theta_t_range": (40, 70),
            "somatic_bias_range": (0.1, 0.3),
            "level": 4,
            "mechanism": "Self-model split with compartmentalized high-precision internal states",
        },
        "depersonalization-disorder": {
            "dsm5_code": "F48.1",
            "category": "dissociative",
            "prediction_error": (-2, -4),
            "prediction_error_type": "body-error",
            "precision_profile": {"interoceptive": "decreased"},
            "theta_t_range": (40, 60),
            "somatic_bias_range": (0.2, 0.4),
            "level": 1,
            "mechanism": "Reduced interoceptive precision with body model decoupling",
        },
        "dissociative-amnesia": {
            "dsm5_code": "F44.0",
            "category": "dissociative",
            "prediction_error": (-2, -3),
            "prediction_error_type": "memory-access",
            "precision_profile": {"autobiographical": "decreased"},
            "theta_t_range": (30, 50),
            "somatic_bias_range": (0.5, 0.8),
            "level": 4,
            "mechanism": "Reduced autobiographical precision with memory access blockage",
        },
        # === SOMATIC SYMPTOM ===
        "somatic-symptom-disorder": {
            "dsm5_code": "F45.1",
            "category": "somatic",
            "prediction_error": (3, 5),
            "prediction_error_type": "body",
            "precision_profile": {"interoceptive": "increased"},
            "theta_t_range": (-20, -35),
            "somatic_bias_range": (2.0, 3.0),
            "level": 1,
            "mechanism": "Elevated interoceptive precision with amplified somatic signals",
        },
        "illness-anxiety-disorder": {
            "dsm5_code": "F45.21",
            "category": "somatic",
            "prediction_error": (3, 5),
            "prediction_error_type": "health-threat",
            "precision_profile": {"threat": "increased"},
            "theta_t_range": (-25, -40),
            "somatic_bias_range": (2.0, 3.0),
            "level": 3,
            "mechanism": "High-precision health threat priors with catastrophic illness interpretation",
        },
        "conversion-disorder": {
            "dsm5_code": "F44.x",
            "category": "somatic",
            "prediction_error": (2, 4),
            "prediction_error_type": "motor-sensory-mismatch",
            "precision_profile": {"motor": "increased"},
            "theta_t_range": (-30, -50),
            "somatic_bias_range": (1.5, 2.5),
            "level": 1,
            "mechanism": "Motor-sensory mismatch with elevated motor precision on conversion symptoms",
        },
        # === FEEDING/EATING ===
        "anorexia-nervosa": {
            "dsm5_code": "F50.01",
            "category": "feeding_eating",
            "prediction_error": (4, 6),
            "prediction_error_type": "body-weight",
            "precision_profile": {"control": "greatly_increased"},
            "theta_t_range": (-30, -50),
            "somatic_bias_range": (0.6, 0.9),
            "level": "1-2",
            "mechanism": "Distorted body model; ↓Πᵢ for hunger; Level 1-2 dissociation",
        },
        "bulimia-nervosa": {
            "dsm5_code": "F50.2",
            "category": "feeding_eating",
            "prediction_error": (-2, 4),
            "prediction_error_type": "control-oscillation",
            "precision_profile": {"reward": "increased"},
            "theta_t_range": (-25, 40),
            "somatic_bias_range": (1.5, 2.5),
            "level": "1-2",
            "mechanism": "Control oscillation with binge-purge precision cycling",
        },
        "binge-eating-disorder": {
            "dsm5_code": "F50.81",
            "category": "feeding_eating",
            "prediction_error": (3, 5),
            "prediction_error_type": "reward",
            "precision_profile": {"reward": "increased"},
            "theta_t_range": (-25, -40),
            "somatic_bias_range": (1.8, 2.8),
            "level": 1,
            "mechanism": "Elevated reward precision with impaired satiety signaling",
        },
        "arfid": {
            "dsm5_code": "F50.82",
            "category": "feeding_eating",
            "prediction_error": (3, 5),
            "prediction_error_type": "food-threat",
            "precision_profile": {"threat": "increased"},
            "theta_t_range": (-25, -40),
            "somatic_bias_range": (1.5, 2.5),
            "level": 1,
            "mechanism": "Food-related threat precision elevation with sensory aversion",
        },
        # === SLEEP-WAKE ===
        "insomnia-disorder": {
            "dsm5_code": "F51.01",
            "category": "sleep_wake",
            "prediction_error": (2, 4),
            "prediction_error_type": "sleep-prediction-error",
            "precision_profile": {"arousal": "increased"},
            "theta_t_range": (25, 40),
            "somatic_bias_range": (1.5, 2.5),
            "level": 4,
            "mechanism": "Failure to raise θₜ for sleep; Level 4 dysfunction; arousal blocks threshold elevation",
        },
        "hypersomnolence": {
            "dsm5_code": "F51.11",
            "category": "sleep_wake",
            "prediction_error": (-2, -3),
            "prediction_error_type": "wake-drive",
            "precision_profile": {"arousal": "decreased"},
            "theta_t_range": (30, 50),
            "somatic_bias_range": (0.8, 1.1),
            "level": 4,
            "mechanism": "Reduced arousal precision with excessive sleep drive",
        },
        "narcolepsy": {
            "dsm5_code": "G47.41",
            "category": "sleep_wake",
            "prediction_error": (-2, -4),
            "prediction_error_type": "state-boundary-error",
            "precision_profile": {"state": "decreased"},
            "theta_t_range": (-20, -40),
            "somatic_bias_range": (1.0, 1.0),
            "level": 4,
            "mechanism": "State boundary error with reduced state transition precision",
        },
        "nightmare-disorder": {
            "dsm5_code": "F51.5",
            "category": "sleep_wake",
            "prediction_error": (3, 5),
            "prediction_error_type": "threat-replay",
            "precision_profile": {"threat": "increased"},
            "theta_t_range": (-30, -50),
            "somatic_bias_range": (2.0, 3.0),
            "level": 4,
            "mechanism": "Threat replay during REM with elevated threat precision",
        },
        # === PERSONALITY DISORDERS ===
        "antisocial-personality-disorder": {
            "dsm5_code": "F60.2",
            "category": "personality",
            "prediction_error": (-3, -5),
            "prediction_error_type": "punishment",
            "precision_profile": {"threat": "decreased"},
            "theta_t_range": (-20, -35),
            "somatic_bias_range": (0.6, 0.9),
            "level": 2,
            "mechanism": "↓Πᵢ for empathy/fear; reduced punishment prediction error precision",
        },
        "borderline-personality-disorder": {
            "dsm5_code": "F60.3",
            "category": "personality",
            "prediction_error": (4, 6),
            "prediction_error_type": "attachment",
            "precision_profile": {"social": "increased"},
            "theta_t_range": (-40, -60),
            "somatic_bias_range": (2.0, 3.5),
            "level": 2,
            "mechanism": "Labile Πᵢ & θₜ; Level 2 instability; intense unstable emotional consciousness",
        },
        "avoidant-personality-disorder": {
            "dsm5_code": "F60.6",
            "category": "personality",
            "prediction_error": (3, 5),
            "prediction_error_type": "social-threat",
            "precision_profile": {"social": "increased"},
            "theta_t_range": (20, 35),
            "somatic_bias_range": (1.8, 3.0),
            "level": 2,
            "mechanism": "Elevated social precision with avoidant threshold elevation",
        },
        "narcissistic-personality-disorder": {
            "dsm5_code": "F60.81",
            "category": "personality",
            "prediction_error": (3, 5),
            "prediction_error_type": "ego-threat",
            "precision_profile": {"self": "increased"},
            "theta_t_range": (-20, -35),
            "somatic_bias_range": (0.8, 1.2),
            "level": 4,
            "mechanism": "Elevated self-precision with ego threat sensitivity",
        },
        "obsessive-compulsive-personality-disorder": {
            "dsm5_code": "F60.5",
            "category": "personality",
            "prediction_error": (2, 4),
            "prediction_error_type": "order-violation",
            "precision_profile": {"control": "increased"},
            "theta_t_range": (-15, -30),
            "somatic_bias_range": (1.2, 1.6),
            "level": 3,
            "mechanism": "Elevated control precision with order violation sensitivity",
        },
        # === SUBSTANCE-RELATED ===
        "substance-use-disorder": {
            "dsm5_code": "F1x.x",
            "category": "substance_related",
            "prediction_error": (4, 7),
            "prediction_error_type": "drug-cue",
            "precision_profile": {"reward": "greatly_increased"},
            "theta_t_range": (-40, -60),
            "somatic_bias_range": (1.5, 3.0),
            "level": 1,
            "mechanism": "Substance-induced θₜ/Π manipulation; drug cues dominate consciousness",
        },
        "gambling-disorder": {
            "dsm5_code": "F63.0",
            "category": "substance_related",
            "prediction_error": (4, 6),
            "prediction_error_type": "uncertainty-reward",
            "precision_profile": {"reward": "increased"},
            "theta_t_range": (-35, -55),
            "somatic_bias_range": (1.2, 2.0),
            "level": 1,
            "mechanism": "Elevated reward precision for uncertainty; near-miss effects",
        },
        # === NEUROCOGNITIVE ===
        "delirium": {
            "dsm5_code": "F05",
            "category": "neurocognitive",
            "prediction_error": (5, 8),
            "prediction_error_type": "global",
            "precision_profile": {"precision": "collapse"},
            "theta_t_range": (0, 0),
            "somatic_bias_range": (0.7, 1.3),
            "level": "global",
            "mechanism": "Global precision collapse with fluctuating awareness",
        },
        "major-neurocognitive-disorder": {
            "dsm5_code": "F02.x",
            "category": "neurocognitive",
            "prediction_error": (-4, -6),
            "prediction_error_type": "global",
            "precision_profile": {"all": "decreased"},
            "theta_t_range": (30, 60),
            "somatic_bias_range": (1.0, 1.0),
            "level": "global",
            "mechanism": "Global precision reduction across all processing levels",
        },
        "mild-neurocognitive-disorder": {
            "dsm5_code": "F06.7",
            "category": "neurocognitive",
            "prediction_error": (-2, -3),
            "prediction_error_type": "global",
            "precision_profile": {"executive": "decreased"},
            "theta_t_range": (15, 30),
            "somatic_bias_range": (1.0, 1.0),
            "level": 3,
            "mechanism": "Executive precision reduction with mild threshold elevation",
        },
        # === DISORDERS OF CONSCIOUSNESS ===
        "coma": {
            "dsm5_code": "R40.20",
            "category": "consciousness",
            "prediction_error": (0, 0),
            "prediction_error_type": "complete-disconnection",
            "precision_profile": {"all": "absent"},
            "theta_t_range": (100, 100),
            "somatic_bias_range": (0.0, 0.0),
            "level": "global",
            "mechanism": "Complete thalamocortical disconnection; absence of P3b, gamma, PCI ≈ 0",
        },
        "vegetative-state": {
            "dsm5_code": "R40.20",
            "category": "consciousness",
            "prediction_error": (0, 0),
            "prediction_error_type": "partial-disconnection",
            "precision_profile": {"minimal": "residual"},
            "theta_t_range": (80, 100),
            "somatic_bias_range": (0.1, 0.2),
            "level": "global",
            "mechanism": "Partial thalamocortical activity; no global broadcast; PCI < 0.3",
        },
        "minimally-conscious-state": {
            "dsm5_code": "R40.20",
            "category": "consciousness",
            "prediction_error": (1, 2),
            "prediction_error_type": "fluctuating",
            "precision_profile": {"fluctuating": True},
            "theta_t_range": (40, 70),
            "somatic_bias_range": (0.3, 0.5),
            "level": "global",
            "mechanism": "Fluctuating thalamocortical connectivity; intermittent P3b; PCI 0.3-0.5",
        },
    }

    # Hierarchical level-specific clinical predictions
    LEVEL_PREDICTIONS: Dict[Union[int, str], Dict[str, Any]] = {
        1: {
            "name": "Sensory/Perceptual Processing",
            "disorders": [
                "panic-disorder",
                "specific-phobia",
                "stereotypic-movement-disorder",
                "tic-disorders",
                "speech-sound-disorder",
                "childhood-onset-fluency-disorder",
                "developmental-coordination-disorder",
                "conversion-disorder",
                "somatic-symptom-disorder",
                "substance-use-disorder",
                "body-dysmorphic-disorder",
                "trichotillomania",
                "depersonalization-disorder",
                "binge-eating-disorder",
                "arfid",
                "nightmare-disorder",
            ],
            "consciousness_implication": "Bodily signals dominate conscious experience",
            "neural_signature": "Early sensory components (P1, N1), gamma band",
            "treatment_target": "Interoceptive exposure, sensory modulation",
        },
        2: {
            "name": "Social/Emotional Processing",
            "disorders": [
                "social-anxiety-disorder",
                "major-depressive-disorder",
                "persistent-depressive-disorder",
                "pmdd",
                "ptsd",
                "acute-stress-disorder",
                "borderline-personality-disorder",
                "antisocial-personality-disorder",
                "avoidant-personality-disorder",
                "social-pragmatic-communication-disorder",
            ],
            "consciousness_implication": "Emotional and social content shapes conscious field",
            "neural_signature": "P2, early posterior negativity, theta-alpha coupling",
            "treatment_target": "Social exposure, emotional regulation training",
        },
        3: {
            "name": "Cognitive/Semantic Processing",
            "disorders": [
                "generalized-anxiety-disorder",
                "obsessive-compulsive-disorder",
                "hoarding-disorder",
                "delusional-disorder",
                "schizotypal-personality-disorder",
                "obsessive-compulsive-personality-disorder",
                "language-disorder",
                "specific-learning-disorder",
                "illness-anxiety-disorder",
            ],
            "consciousness_implication": "Thought content dominates; rumination, worry",
            "neural_signature": "N400, P3b, semantic theta power",
            "treatment_target": "Cognitive restructuring, metacognitive training",
        },
        4: {
            "name": "Narrative/State Processing",
            "disorders": [
                "insomnia-disorder",
                "hypersomnolence",
                "narcolepsy",
                "nightmare-disorder",
                "dissociative-identity-disorder",
                "dissociative-amnesia",
                "narcissistic-personality-disorder",
            ],
            "consciousness_implication": "State transitions, narrative continuity disrupted",
            "neural_signature": "Slow oscillations, DMN connectivity, delta power",
            "treatment_target": "Sleep hygiene, state regulation, narrative integration",
        },
        "global": {
            "name": "Global/System-Wide Dysregulation",
            "disorders": [
                "schizophrenia",
                "bipolar-i",
                "bipolar-ii",
                "cyclothymia",
                "schizoaffective-disorder",
                "brief-psychotic-disorder",
                "autism-spectrum-disorder",
                "adhd",
                "intellectual-disability",
                "global-developmental-delay",
                "major-neurocognitive-disorder",
                "delirium",
                "vegetative-state",
                "minimally-conscious-state",
                "coma",
            ],
            "consciousness_implication": "Global changes in conscious state; all content affected",
            "neural_signature": "Wide-band coherence, PCI, global workspace metrics",
            "treatment_target": "Pharmacological modulation, global network targeting",
        },
        "coupling": {
            "name": "Cross-Level Coupling Disruption",
            "disorders": ["ptsd", "acute-stress-disorder"],
            "consciousness_implication": "Specific decoupling between hierarchical levels",
            "neural_signature": "Reduced cross-frequency coupling, L2-L3 decoupling",
            "treatment_target": "Level reintegration, coupling restoration",
        },
    }

    def __init__(self, profiles_dir: Optional[Path] = None):
        """
        Initialize the Clinical Phenotype Mapper.

        Args:
            profiles_dir: Directory containing YAML profile files
        """
        if profiles_dir is None:
            profiles_dir = Path(__file__).parent.parent / "config" / "profiles"
        self.profiles_dir = profiles_dir
        self._loaded_profiles: Dict[str, Dict[str, Any]] = {}

    def get_disorder_profile(self, disorder_key: str) -> Optional[DisorderPhenotype]:
        """
        Get complete phenotype profile for a disorder.

        Args:
            disorder_key: Disorder identifier (e.g., 'panic-disorder', 'adhd')

        Returns:
            DisorderPhenotype with complete APGI parameterization
        """
        # First check if we have a YAML profile
        yaml_profile = self._load_yaml_profile(disorder_key)
        if yaml_profile:
            return self._yaml_to_phenotype(yaml_profile)

        # Fall back to parameter table
        if disorder_key in self.DISORDER_PARAMETER_TABLE:
            return self._table_to_phenotype(disorder_key)

        return None

    def _load_yaml_profile(self, disorder_key: str) -> Optional[Dict[str, Any]]:
        """Load profile from YAML file if available"""
        if disorder_key in self._loaded_profiles:
            return self._loaded_profiles[disorder_key]

        yaml_path = self.profiles_dir / f"{disorder_key}.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                profile = yaml.safe_load(f)
                self._loaded_profiles[disorder_key] = profile
                return profile
        return None

    def _yaml_to_phenotype(self, yaml_profile: Dict[str, Any]) -> DisorderPhenotype:
        """Convert YAML profile to DisorderPhenotype"""
        params = yaml_profile.get("parameters", {}).get("model", {})
        precision = yaml_profile.get("precision_profile", {})
        hierarchical = yaml_profile.get("hierarchical_profile", {})

        apgi_profile = APGIParameterProfile(
            epsilon_mean=precision.get("prediction_error_epsilon", 0),
            epsilon_sigma_range=(
                precision.get("prediction_error_sigma", 0),
                precision.get("prediction_error_sigma", 0),
            ),
            pi_exteroceptive=precision.get("precision_pi_values", {}).get("Pi_e", 1.0),
            pi_interoceptive=precision.get("precision_pi_values", {}).get("Pi_i", 1.0),
            pi_dominant_channels=precision.get("precision_pi_dominant_channels", []),
            theta_t_percent=precision.get("threshold_theta_t_pct", 0),
            theta_t_absolute=params.get("theta_0", 0.5),
            somatic_bias=precision.get("somatic_bias_beta", 1.0),
            primary_affected_level=hierarchical.get("affected_level", 1),
            cross_level_coupling=hierarchical.get("cross_level_coupling", {}),
        )

        metadata = yaml_profile.get("metadata", {})

        return DisorderPhenotype(
            dsm5_code=metadata.get("dsm5_code", ""),
            category=yaml_profile.get("category", ""),
            description=yaml_profile.get("description", ""),
            apgi_profile=apgi_profile,
            neural_signatures=NeuralSignature(
                eeg_markers=metadata.get("neural_signatures", [])
            ),
            treatment_implications=TreatmentImplication(
                first_line=metadata.get("treatment_implications", [])
            ),
            mechanism=yaml_profile.get("hierarchical_profile", {}).get("mechanism", ""),
            phenomenology=metadata.get("phenomenology", ""),
            citations=yaml_profile.get("empirical_validation", {}).get("citations", []),
        )

    def _table_to_phenotype(self, disorder_key: str) -> DisorderPhenotype:
        """Convert parameter table entry to DisorderPhenotype"""
        table_entry = self.DISORDER_PARAMETER_TABLE[disorder_key]

        theta_range = table_entry["theta_t_range"]
        beta_range = table_entry["somatic_bias_range"]
        eps_range = table_entry["prediction_error"]

        apgi_profile = APGIParameterProfile(
            epsilon_mean=np.mean(eps_range),
            epsilon_sigma_range=eps_range,
            pi_exteroceptive=1.0,
            pi_interoceptive=1.0,
            pi_dominant_channels=list(table_entry["precision_profile"].keys()),
            theta_t_percent=int(np.mean(theta_range)),
            theta_t_absolute=0.5 + np.mean(theta_range) / 200,  # Convert to 0-1 scale
            somatic_bias=np.mean(beta_range),
            primary_affected_level=table_entry["level"],
        )

        return DisorderPhenotype(
            dsm5_code=table_entry["dsm5_code"],
            category=table_entry["category"],
            description=table_entry["mechanism"],
            apgi_profile=apgi_profile,
            mechanism=table_entry["mechanism"],
            phenomenology="See APGI Empirical Credibility Roadmap",
            citations=[],
        )

    def simulate_phenotype(
        self,
        disorder_key: str,
        n_trials: int = 100,
        duration: float = 10.0,
        dt: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Simulate phenotype dynamics for a disorder.

        Args:
            disorder_key: Disorder identifier
            n_trials: Number of simulation trials
            duration: Simulation duration in seconds
            dt: Time step

        Returns:
            Simulation results with time series and summary statistics
        """
        profile = self.get_disorder_profile(disorder_key)
        if profile is None:
            raise ValueError(f"Unknown disorder: {disorder_key}")

        apgi = profile.apgi_profile
        n_steps = int(duration / dt)

        # Initialize simulation variables
        S = np.zeros((n_trials, n_steps))  # Surprise accumulation
        theta = np.zeros((n_trials, n_steps))  # Threshold
        ignition = np.zeros((n_trials, n_steps))  # Ignition

        # Initial conditions based on disorder profile
        theta[:, 0] = apgi.theta_t_absolute

        # Simulation parameters
        tau = 0.2  # Signal decay time constant
        k = 3.0  # Sigmoid steepness

        for t in range(1, n_steps):
            # Generate prediction errors based on disorder profile
            epsilon_e = np.random.normal(
                apgi.epsilon_mean * 0.1, 0.5 + abs(apgi.epsilon_mean) * 0.1, n_trials
            )
            epsilon_i = np.random.normal(
                apgi.epsilon_mean * 0.05, 0.3 + abs(apgi.epsilon_mean) * 0.05, n_trials
            )

            # Effective interoceptive precision with somatic bias
            pi_i_eff = apgi.pi_interoceptive * (1 + apgi.somatic_bias * 0.5)

            # Signal accumulation
            we, wi = 0.5, 0.5
            dS = (
                -S[:, t - 1] / tau
                + we * apgi.pi_exteroceptive * np.abs(epsilon_e)
                + wi * pi_i_eff * np.abs(epsilon_i)
            ) * dt
            S[:, t] = S[:, t - 1] + dS

            # Threshold adaptation
            alpha = 0.9
            theta_0 = apgi.theta_t_absolute
            d_theta = alpha * (theta_0 - theta[:, t - 1]) * dt
            theta[:, t] = theta[:, t - 1] + d_theta

            # Ignition probability
            ignition[:, t] = 1 / (1 + np.exp(-k * (S[:, t] - theta[:, t])))

        return {
            "disorder": disorder_key,
            "n_trials": n_trials,
            "duration": duration,
            "dt": dt,
            "apgi_profile": apgi,
            "ignition_rate": np.mean(ignition[:, -int(1 / dt) :]),  # Last second
            "ignition_variability": np.std(np.mean(ignition, axis=1)),
            "mean_threshold": np.mean(theta[:, -1]),
            "threshold_stability": np.std(theta[:, -1]),
            "surprise_accumulation": np.mean(S[:, -1]),
            "time_series": {
                "S": S,
                "theta": theta,
                "ignition": ignition,
            },
        }

    def compare_level_profiles(self, level: Union[int, str]) -> Dict[str, Any]:
        """
        Compare all disorders at a specific hierarchical level.

        Args:
            level: Hierarchical level (1-4, 'global', or 'coupling')

        Returns:
            Comparison of disorders at the specified level
        """
        level_info = self.LEVEL_PREDICTIONS.get(level, {})
        disorders = level_info.get("disorders", [])

        profiles = {}
        for disorder_key in disorders:
            profile = self.get_disorder_profile(disorder_key)
            if profile:
                profiles[disorder_key] = profile

        return {
            "level": level,
            "level_name": level_info.get("name", ""),
            "consciousness_implication": level_info.get(
                "consciousness_implication", ""
            ),
            "neural_signature": level_info.get("neural_signature", ""),
            "treatment_target": level_info.get("treatment_target", ""),
            "n_disorders": len(profiles),
            "disorders": profiles,
            "mean_theta_t": (
                np.mean([p.apgi_profile.theta_t_percent for p in profiles.values()])
                if profiles
                else 0
            ),
            "mean_somatic_bias": (
                np.mean([p.apgi_profile.somatic_bias for p in profiles.values()])
                if profiles
                else 1.0
            ),
        }

    def get_disorders_by_category(self, category: str) -> List[str]:
        """Get all disorders in a specific category"""
        return [
            key
            for key, params in self.DISORDER_PARAMETER_TABLE.items()
            if params["category"] == category
        ]

    def list_all_disorders(self) -> List[str]:
        """List all available disorders"""
        yaml_disorders = [
            p.stem
            for p in self.profiles_dir.glob("*.yaml")
            if p.stem not in ["research-default"]
        ]
        table_disorders = list(self.DISORDER_PARAMETER_TABLE.keys())
        return sorted(set(yaml_disorders + table_disorders))

    def get_precision_summary_table(self) -> pd.DataFrame:
        """Generate comprehensive precision summary table for all disorders"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for table generation")

        data = []
        for disorder_key in self.list_all_disorders():
            profile = self.get_disorder_profile(disorder_key)
            if profile:
                apgi = profile.apgi_profile
                data.append(
                    {
                        "Disorder": disorder_key,
                        "DSM-5 Code": profile.dsm5_code,
                        "Category": profile.category,
                        "ε (σ)": f"{apgi.epsilon_mean:.1f}",
                        "Πₑ": f"{apgi.pi_exteroceptive:.1f}",
                        "Πᵢ": f"{apgi.pi_interoceptive:.1f}",
                        "θₜ (%)": f"{apgi.theta_t_percent:.0f}",
                        "β": f"{apgi.somatic_bias:.1f}",
                        "Level": apgi.primary_affected_level,
                        "Mechanism": (
                            profile.mechanism[:60] + "..."
                            if len(profile.mechanism) > 60
                            else profile.mechanism
                        ),
                    }
                )

        return pd.DataFrame(data)


def create_clinical_profile_summary() -> str:
    """Create markdown summary of all clinical profiles"""
    mapper = ClinicalPhenotypeMapper()

    lines = [
        "# APGI Clinical Phenotype Summary\n",
        "## Disorder-Specific Precision Profiles\n",
        "### Core Anxiety/Depression/Psychosis Profiles\n",
    ]

    # Core disorders from user's request
    core_disorders = [
        "generalized-anxiety-disorder",
        "panic-disorder",
        "social-anxiety-disorder",
        "major-depressive-disorder",
        "bipolar-i",
        "schizophrenia",
        "ptsd",
        "obsessive-compulsive-disorder",
        "adhd",
        "autism-spectrum-disorder",
    ]

    for disorder_key in core_disorders:
        profile = mapper.get_disorder_profile(disorder_key)
        if profile:
            apgi = profile.apgi_profile
            lines.append(f"\n#### {disorder_key.replace('-', ' ').title()}\n")
            lines.append(f"- **DSM-5**: {profile.dsm5_code}\n")
            lines.append(f"- **Primary Dysregulation**: {profile.category}\n")
            lines.append(f"- **Affected Level**: {apgi.primary_affected_level}\n")
            lines.append(
                f"- **Πₑ**: {apgi.pi_exteroceptive:.1f}, **Πᵢ**: {apgi.pi_interoceptive:.1f}\n"
            )
            lines.append(f"- **θₜ**: {apgi.theta_t_percent:+.0f}%\n")
            lines.append(f"- **β**: {apgi.somatic_bias:.1f}\n")
            lines.append(f"- **Mechanism**: {profile.mechanism}\n")

    return "".join(lines)


if __name__ == "__main__":
    # Demo usage
    mapper = ClinicalPhenotypeMapper()

    print("=" * 70)
    print("APGI Clinical Phenotype Mapper - Demo")
    print("=" * 70)

    # List all available disorders
    disorders = mapper.list_all_disorders()
    print(f"\nTotal disorders mapped: {len(disorders)}")

    # Get and display a specific profile
    disorder = "panic-disorder"
    profile = mapper.get_disorder_profile(disorder)
    if profile:
        separator = "=" * 70
        print(f"\n{separator}")
        print(f"Profile: {disorder}")
        print(separator)
        print(f"DSM-5 Code: {profile.dsm5_code}")
        print(f"Category: {profile.category}")
        print(f"Mechanism: {profile.mechanism[:100]}...")
        print("\nAPGI Parameters:")
        print(f"  ε (prediction error): {profile.apgi_profile.epsilon_mean:.1f}σ")
        print(f"  Πₑ (exteroceptive): {profile.apgi_profile.pi_exteroceptive:.1f}")
        print(f"  Πᵢ (interoceptive): {profile.apgi_profile.pi_interoceptive:.1f}")
        print(f"  θₜ (threshold): {profile.apgi_profile.theta_t_percent:+.0f}%")
        print(f"  β (somatic bias): {profile.apgi_profile.somatic_bias:.1f}")
        print(f"  Affected Level: {profile.apgi_profile.primary_affected_level}")

    # Simulate phenotype
    sim_separator = "=" * 70
    print(f"\n{sim_separator}")
    print(f"Simulation: {disorder}")
    print(sim_separator)
    sim = mapper.simulate_phenotype(disorder, n_trials=10, duration=5.0)
    print(f"Ignition rate: {sim['ignition_rate']:.3f} ")
    print(f"Mean threshold: {sim['mean_threshold']:.3f} ")
    print(f"Surprise accumulation: {sim['surprise_accumulation']:.3f} ")

    # Compare level profiles
    print(f"\n{separator}")
    print("Level 1 Disorders (Sensory/Perceptual)")
    print(f"{separator}")
    level_comparison = mapper.compare_level_profiles(level=1)
    print(f"Level: {level_comparison['level_name']} ")
    print(f"Disorders: {level_comparison['n_disorders']} ")
    print(f"Mean θₜ: {level_comparison['mean_theta_t']:.1f}% ")
    print(
        f"Consciousness implication: {level_comparison['consciousness_implication']} "
    )
