"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

from typing import Any, Dict, Union

from utils.protocol_loader import get_template_protocol
from utils.protocol_schema import PredictionResult, PredictionStatus


def validate_against_template_benchmark(
    protocol_key: str, benchmark_id: str, observed_value: Union[float, int]
) -> PredictionResult:
    """Check if observed value meets template benchmark criteria.

    Args:
        protocol_key: Protocol key from template (e.g., 'P01', 'P02')
        benchmark_id: Benchmark identifier (e.g., 'P1a_P3b_amplitude_shift')
        observed_value: Observed/computed value to validate

    Returns:
        PredictionResult with validation outcome
    """
    template_proto = get_template_protocol(protocol_key)
    if not template_proto:
        return PredictionResult(passed=False, status=PredictionStatus.MISSING_PROTOCOL)

    benchmark = template_proto.get("benchmarks", {}).get(benchmark_id)
    if not benchmark:
        return PredictionResult(passed=False, status=PredictionStatus.NOT_EVALUATED)

    expected = benchmark.get("value")
    operator = benchmark.get("operator", "in_range")
    unit = benchmark.get("unit", "")

    # Determine if the benchmark passes
    if operator == "greater_than":
        passed = observed_value > expected
    elif operator == "less_than":
        passed = observed_value < expected
    elif operator == "approximately":
        # Allow 10% tolerance for "approximately"
        tolerance = 0.1 * expected if isinstance(expected, (int, float)) else 0
        passed = abs(observed_value - expected) <= tolerance
    elif isinstance(expected, list):
        # Range check
        passed = expected[0] <= observed_value <= expected[1]
    else:
        # Exact match
        passed = observed_value == expected

    return PredictionResult(
        passed=passed,
        value=observed_value,
        threshold=f"{expected} {unit}" if unit else expected,
        status=PredictionStatus.PASSED if passed else PredictionStatus.FAILED,
        name=benchmark_id,
        evidence=[f"Benchmark role: {benchmark.get('role', 'unknown')}"],
        metadata={
            "operator": operator,
            "unit": unit,
            "note": benchmark.get("note", ""),
        },
    )


def check_falsification_threshold(
    protocol_key: str, threshold_id: str, observed_value: Union[float, int, bool]
) -> PredictionResult:
    """Check if observed value violates falsification threshold.

    Args:
        protocol_key: Protocol key from template (e.g., 'P01', 'P02')
        threshold_id: Threshold identifier (e.g., 'HEP_P3b_r_below')
        observed_value: Observed/computed value to check

    Returns:
        PredictionResult where passed=True means threshold NOT violated (good)
    """
    template_proto = get_template_protocol(protocol_key)
    if not template_proto:
        return PredictionResult(passed=False, status=PredictionStatus.MISSING_PROTOCOL)

    thresholds = template_proto.get("falsification_thresholds", {})
    if threshold_id not in thresholds:
        return PredictionResult(passed=False, status=PredictionStatus.NOT_EVALUATED)

    threshold_value = thresholds[threshold_id]

    # Threshold names ending in "_below" mean observed must be ABOVE threshold to pass
    # Threshold names ending in "_above" mean observed must be BELOW threshold to pass
    if threshold_id.endswith("_below"):
        passed = observed_value > threshold_value
    elif threshold_id.endswith("_above"):
        passed = observed_value < threshold_value
    elif isinstance(threshold_value, bool):
        passed = observed_value == threshold_value
    else:
        # Default: observed must be less than or equal to threshold
        passed = observed_value <= threshold_value

    return PredictionResult(
        passed=passed,
        value=observed_value,
        threshold=threshold_value,
        status=PredictionStatus.PASSED if passed else PredictionStatus.FAILED,
        name=threshold_id,
        evidence=[f"Falsification threshold check: {threshold_id}"],
        metadata={"threshold_type": "falsification"},
    )


def get_template_benchmarks(protocol_key: str) -> Dict[str, Any]:
    """Get all benchmarks for a protocol from the template.

    Args:
        protocol_key: Protocol key from template (e.g., 'P01', 'P02')

    Returns:
        Dictionary of benchmark_id -> benchmark_data
    """
    template_proto = get_template_protocol(protocol_key)
    if not template_proto:
        return {}
    return template_proto.get("benchmarks", {})


def get_template_falsification_thresholds(protocol_key: str) -> Dict[str, Any]:
    """Get all falsification thresholds for a protocol from the template.

    Args:
        protocol_key: Protocol key from template (e.g., 'P01', 'P02')

    Returns:
        Dictionary of threshold_id -> threshold_value
    """
    template_proto = get_template_protocol(protocol_key)
    if not template_proto:
        return {}
    return template_proto.get("falsification_thresholds", {})


def validate_all_benchmarks(
    protocol_key: str, observed_values: Dict[str, Union[float, int]]
) -> Dict[str, PredictionResult]:
    """Validate multiple benchmarks against observed values.

    Args:
        protocol_key: Protocol key from template (e.g., 'P01', 'P02')
        observed_values: Dictionary of benchmark_id -> observed_value

    Returns:
        Dictionary of benchmark_id -> PredictionResult
    """
    results = {}
    for benchmark_id, observed_value in observed_values.items():
        results[benchmark_id] = validate_against_template_benchmark(protocol_key, benchmark_id, observed_value)
    return results
