# APGI Protocol Tier Classification Rationale

## Overview

The APGI validation framework organizes validation protocols into three tiers based on their importance, scope, and relationship to core APGI theory. This classification enables weighted scoring and prioritized validation efforts.

## Tier Definitions

### Primary Protocols (Protocols 1-2)

**Rationale:** Core validation protocols that test fundamental APGI properties. These are essential for establishing basic APGI validity and must pass for the framework to be considered viable.

### Protocol 1: Basic Equation Validation

- Tests fundamental APGI mathematical equations
- Validates surprise accumulation, threshold adaptation, and ignition dynamics
- Checks parameter bounds and numerical stability
- **Why Primary:** Without passing Protocol 1, the APGI model itself is fundamentally broken

### Protocol 2: Parameter Consistency Checks

- Validates parameter interactions and constraints
- Tests parameter estimation accuracy
- Checks for parameter identifiability
- **Why Primary:** Parameter consistency is foundational to all other validations

**Weight in Scoring:** 50% (0.5)

### Secondary Protocols (Protocols 3-4, 8, 11-12)

**Rationale:** Extended validation covering specific aspects of APGI. These test important but not absolutely foundational properties. They provide deeper insights into APGI's behavior and falsifiability.

### Protocol 3: Behavioral Pattern Validation

- Tests whether APGI agents produce expected behavioral patterns
- Validates decision-making and action selection
- Compares APGI predictions to empirical behavioral data
- **Why Secondary:** Important for real-world applicability but builds on core equations

### Protocol 4: State Transition Verification

- Tests phase transition properties of ignition
- Validates information-theoretic signatures
- Checks for critical phenomena in APGI dynamics
- **Why Secondary:** Tests specific predictions about consciousness-related phenomena

### Protocol 8: Cross-Species Scaling Validation

- Tests whether APGI scales appropriately across species
- Validates scaling laws and allometric relationships
- Checks species-specific parameter adjustments
- **Why Secondary:** Important for generalizability but not core to APGI theory

### Protocol 11: Cultural Neuroscience Validation

- Tests cultural and contextual influences on APGI
- Validates cross-cultural parameter variations
- Checks for cultural universals vs. specifics
- **Why Secondary:** Important for ecological validity but not foundational

### Protocol 12: Liquid Network Validation

- Tests liquid network properties and dynamics
- Validates network topology and connectivity effects
- Checks for phase transitions in network states
- **Why Secondary:** Tests advanced network properties but not core requirements

**Weight in Scoring:** 30% (0.3)

### Tertiary Protocols (Protocols 5-7, 9-10)

**Rationale:** Specialized and experimental protocols that test niche aspects, computational implementations, or exploratory hypotheses. These provide supplementary evidence but are not critical for core validation.

### Protocol 5: Computational Benchmarking

- Tests computational efficiency and performance
- Validates implementation quality
- Benchmarks against alternative architectures
- **Why Tertiary:** Important for practical use but not theoretical validity

### Protocol 6: Bayesian Estimation Framework

- Tests parameter estimation using Bayesian methods
- Validates uncertainty quantification
- Checks posterior predictive validity
- **Why Tertiary:** Advanced statistical methods but not required for basic validation

### Protocol 7: Multimodal Integration

- Tests integration of multiple data modalities
- Validates cross-modal consistency
- Checks for modality-specific parameter adjustments
- **Why Tertiary:** Important for real-world applications but not core theory

### Protocol 9: Psychological States Validation

- Tests APGI predictions about psychological states
- Validates state-dependent parameter changes
- Checks for emotion and motivation effects
- **Why Tertiary:** Tests specific predictions but not fundamental

### Protocol 10: Turing Machine Validation

- Tests computational universality properties
- Validates APGI as a computational model
- Checks for algorithmic completeness
- **Why Tertiary:** Theoretical curiosity but not required for validation

**Weight in Scoring:** 20% (0.2)

## Weighted Scoring Rationale

The tier weights (Primary: 0.5, Secondary: 0.3, Tertiary: 0.2) reflect:

1. **Criticality:** Primary protocols must pass for APGI to be viable
2. **Scope:** Secondary protocols cover important but not essential aspects
3. **Supplementary:** Tertiary protocols provide additional evidence

This weighting ensures that:

- Core APGI validity dominates the overall assessment
- Important but not critical aspects contribute meaningfully
- Supplementary evidence provides context without overwhelming core results

## Decision Thresholds

Based on weighted scores:

- **PASS (≥0.85):** Strong validation support - APGI is well-validated
- **MARGINAL (0.60-0.84):** Moderate validation support - APGI has issues but may be viable
- **FAIL (<0.60):** Insufficient validation support - APGI requires major revisions

## Implementation Notes

- Tier classification is defined in `Validation/Master_Validation.py`
- Weighted scoring is implemented in `generate_master_report()` method
- Each protocol's tier is stored in `PROTOCOL_TIERS` dictionary
- Falsification status is tracked per tier in `falsification_status` dictionary

## References

- See `Validation/Master_Validation.py` lines 27-56 for implementation
- See `tests/test_gui.py` lines 93-106 for test fixture with tier definitions
- See `docs/APGI-Parameter-Specifications.md` for parameter-level details
