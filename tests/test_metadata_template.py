"""Tests for Metadata Template module - comprehensive coverage."""

from utils.metadata_template import (
    DEFAULT_TEMPLATE,
    MetadataTemplate,
    create_metadata_template,
    load_template,
    save_template,
    validate_metadata,
)


class TestMetadataTemplate:
    """Test Metadata Template class."""

    def test_init_default(self):
        """Test default initialization."""
        template = MetadataTemplate()
        assert template.fields is not None

    def test_init_with_fields(self):
        """Test initialization with custom fields."""
        fields = {"name": str, "value": float}
        template = MetadataTemplate(fields=fields)
        assert "name" in template.fields
        assert "value" in template.fields

    def test_add_field(self):
        """Test adding a field."""
        template = MetadataTemplate()
        template.add_field("new_field", str)
        assert "new_field" in template.fields

    def test_validate_valid_data(self):
        """Test validation with valid data."""
        template = MetadataTemplate()
        template.add_field("name", str)

        data = {"name": "test"}
        result = template.validate(data)
        assert result is True

    def test_validate_invalid_data(self):
        """Test validation with invalid data."""
        template = MetadataTemplate()
        template.add_field("count", int)

        data = {"count": "not an integer"}
        result = template.validate(data)
        assert result is False


class TestCreateMetadataTemplate:
    """Test template creation function."""

    def test_create_basic_template(self):
        """Test creating basic template."""
        template = create_metadata_template(type="basic")
        assert isinstance(template, MetadataTemplate)

    def test_create_experimental_template(self):
        """Test creating experimental template."""
        template = create_metadata_template(type="experimental")
        assert isinstance(template, MetadataTemplate)
        # Should have experimental-specific fields


class TestValidateMetadata:
    """Test metadata validation."""

    def test_validate_against_template(self):
        """Test validating against a template."""
        template = MetadataTemplate()
        template.add_field("subject_id", str)
        template.add_field("session", int)

        data = {"subject_id": "sub-01", "session": 1}
        result = validate_metadata(data, template)
        assert result["valid"] is True

    def test_validate_missing_fields(self):
        """Test validation with missing required fields."""
        template = MetadataTemplate()
        template.add_field("required_field", str, required=True)

        data = {}
        result = validate_metadata(data, template)
        assert result["valid"] is False
        assert "missing" in result


class TestLoadAndSaveTemplate:
    """Test template persistence."""

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading JSON template."""
        template = MetadataTemplate()
        template.add_field("test_field", str)

        filepath = tmp_path / "template.json"
        save_template(template, filepath)

        loaded = load_template(filepath)
        assert loaded is not None


class TestDefaultTemplate:
    """Test default template."""

    def test_default_template_exists(self):
        """Test that default template exists and is valid."""
        assert DEFAULT_TEMPLATE is not None
        assert isinstance(DEFAULT_TEMPLATE, dict)

    def test_default_template_has_required_fields(self):
        """Test default template has required fields."""
        required_fields = ["version", "created", "modified"]
        for field in required_fields:
            assert field in DEFAULT_TEMPLATE
