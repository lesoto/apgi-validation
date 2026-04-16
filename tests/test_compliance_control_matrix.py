"""Compliance matrix traceability checks."""

import json
from pathlib import Path


def test_control_matrix_contains_required_frameworks():
    matrix_path = Path("docs/compliance/control_matrix.json")
    data = json.loads(matrix_path.read_text(encoding="utf-8"))

    frameworks = data["frameworks"]
    assert "SOC2" in frameworks
    assert "GDPR" in frameworks
    assert "HIPAA" in frameworks


def test_control_matrix_has_governance_artifacts():
    matrix_path = Path("docs/compliance/control_matrix.json")
    data = json.loads(matrix_path.read_text(encoding="utf-8"))

    governance = data["governance"]
    for _, rel_path in governance.items():
        assert Path(rel_path).exists()
