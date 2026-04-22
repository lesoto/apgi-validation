"""Compliance matrix traceability checks."""

import json
from pathlib import Path


def test_control_matrix_contains_required_frameworks():
    matrix_path = Path("docs/compliance/control_matrix.json")
    data = json.loads(matrix_path.read_text(encoding="utf-8"))

    standards = data["standards"]
    assert "SOC2" in standards
    assert "GDPR" in standards
    assert "HIPAA" in standards


def test_control_matrix_has_governance_artifacts():
    matrix_path = Path("docs/compliance/control_matrix.json")
    data = json.loads(matrix_path.read_text(encoding="utf-8"))

    # Check that the control matrix has the required structure
    assert "standards" in data
    assert "scientific_standards" in data
    assert "gaps_and_recommendations" in data

    # Verify each standard has required fields
    for standard_name, standard_data in data["standards"].items():
        assert "name" in standard_data
        assert "status" in standard_data
        assert "controls" in standard_data
