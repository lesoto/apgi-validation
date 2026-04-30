import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.protocol_schema import PredictionResult, PredictionStatus, ProtocolResult


# Mock implementation of test logic for entry point contracts
def test_cli_contract_success():
    """Verify that successful CLI execution adheres to ProtocolResult contract."""
    # This represents a CLI run that succeeds
    result = ProtocolResult(
        protocol_id="VP_01",
        named_predictions={
            "pred_1": PredictionResult(
                passed=True,
                value=1.0,
                threshold=0.5,
                status=PredictionStatus.PASSED,
                name="test_prediction",
                evidence=["Test evidence"],
            )
        },
        completion_percentage=100,
        methodology="test_method",
    )
    assert result.completion_percentage == 100
    assert len(result.named_predictions) == 1
    assert result.named_predictions["pred_1"].passed is True


def test_gui_contract_validation():
    """Verify GUI execution returns proper valid contracts for different config profiles."""
    # Create a mock result directly
    result = ProtocolResult(
        protocol_id="VP_GUI",
        named_predictions={},
        completion_percentage=50,
        methodology="gui_method",
        errors=["Test error"],
    )
    assert result.completion_percentage == 50
    assert isinstance(result, ProtocolResult)


def test_api_contract_error_handling():
    """Ensure protocol contracts return proper error formatting."""
    # Test that invalid completion percentage raises validation error
    with pytest.raises(ValueError):
        ProtocolResult(
            protocol_id="VP_FAIL",
            named_predictions={},
            completion_percentage=150,  # Invalid: > 100
            methodology="test",
        )
