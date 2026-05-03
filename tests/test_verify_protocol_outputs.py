"""Tests for Verify Protocol Outputs module - comprehensive coverage."""

import json

from utils.verify_protocol_outputs import (
    OutputVerifier,
    VerificationResult,
    check_output_structure,
    compare_protocol_outputs,
    validate_output_files,
    verify_protocol_output,
)


class TestVerificationResult:
    """Test Verification Result dataclass."""

    def test_result_creation(self):
        """Test creating verification result."""
        result = VerificationResult(
            valid=True, errors=[], warnings=["minor issue"], details={"checked": 10}
        )
        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1

    def test_result_is_success(self):
        """Test success status."""
        result = VerificationResult(valid=True, errors=[], warnings=[], details={})
        assert result.is_success() is True

        result_invalid = VerificationResult(
            valid=False, errors=["error"], warnings=[], details={}
        )
        assert result_invalid.is_success() is False


class TestVerifyProtocolOutput:
    """Test protocol output verification."""

    def test_verify_valid_output(self, tmp_path):
        """Test verifying valid output."""
        # Create valid output file
        output_file = tmp_path / "output.json"
        output_data = {"status": "success", "results": [1, 2, 3]}
        with open(output_file, "w") as f:
            json.dump(output_data, f)

        result = verify_protocol_output(output_file)
        assert isinstance(result, VerificationResult)

    def test_verify_nonexistent_output(self, tmp_path):
        """Test verifying non-existent output."""
        output_file = tmp_path / "nonexistent.json"
        result = verify_protocol_output(output_file)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_verify_invalid_json(self, tmp_path):
        """Test verifying invalid JSON."""
        output_file = tmp_path / "invalid.json"
        output_file.write_text("not valid json{")

        result = verify_protocol_output(output_file)
        assert result.valid is False


class TestCheckOutputStructure:
    """Test output structure checking."""

    def test_check_valid_structure(self):
        """Test checking valid structure."""
        data = {"required_field": "value", "optional_field": 123}
        required_fields = ["required_field"]

        result = check_output_structure(data, required_fields)
        assert result["valid"] is True

    def test_check_missing_fields(self):
        """Test checking with missing required fields."""
        data = {"other_field": "value"}
        required_fields = ["required_field"]

        result = check_output_structure(data, required_fields)
        assert result["valid"] is False
        assert "missing_fields" in result


class TestValidateOutputFiles:
    """Test output file validation."""

    def test_validate_existing_files(self, tmp_path):
        """Test validating existing files."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.touch()
        file2.touch()

        files = [file1, file2]
        result = validate_output_files(files)
        assert result["valid"] is True

    def test_validate_missing_files(self, tmp_path):
        """Test validating with missing files."""
        existing = tmp_path / "exists.txt"
        existing.touch()
        missing = tmp_path / "missing.txt"

        files = [existing, missing]
        result = validate_output_files(files)
        assert result["valid"] is False
        assert "missing" in result


class TestCompareProtocolOutputs:
    """Test protocol output comparison."""

    def test_compare_identical_outputs(self):
        """Test comparing identical outputs."""
        output1 = {"status": "success", "data": [1, 2, 3]}
        output2 = {"status": "success", "data": [1, 2, 3]}

        result = compare_protocol_outputs(output1, output2)
        assert result["identical"] is True

    def test_compare_different_outputs(self):
        """Test comparing different outputs."""
        output1 = {"status": "success", "data": [1, 2, 3]}
        output2 = {"status": "success", "data": [1, 2, 4]}

        result = compare_protocol_outputs(output1, output2)
        assert result["identical"] is False


class TestOutputVerifier:
    """Test Output Verifier class."""

    def test_init(self):
        """Test initialization."""
        verifier = OutputVerifier()
        assert verifier is not None

    def test_verify_with_schema(self, tmp_path):
        """Test verifying with schema."""
        verifier = OutputVerifier()

        output_file = tmp_path / "output.json"
        output_data = {"name": "test", "value": 42}
        with open(output_file, "w") as f:
            json.dump(output_data, f)

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = verifier.verify(output_file, schema=schema)
        assert isinstance(result, VerificationResult)
