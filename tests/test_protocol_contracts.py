"""Protocol contract and dependency graph validation tests."""

from Validation.Master_Validation import APGIMasterValidator


def test_master_validator_builds_contract_diagnostics():
    validator = APGIMasterValidator()
    assert validator.contract_diagnostics
    assert "Protocol-1" in validator.contract_diagnostics


def test_master_validator_dependency_graph_validation_runs():
    validator = APGIMasterValidator()
    validator._validate_dependency_graph()
