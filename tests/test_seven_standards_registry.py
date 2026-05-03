"""Tests for Seven Standards Registry module - comprehensive coverage."""

from utils.seven_standards_registry import (
    SEVEN_STANDARDS,
    ComplianceLevel,
    Standard,
    StandardsRegistry,
    check_compliance,
    get_standard,
    list_standards,
    register_standard,
)


class TestStandard:
    """Test Standard dataclass."""

    def test_standard_creation(self):
        """Test creating a standard."""
        standard = Standard(
            id="TEST-001",
            name="Test Standard",
            description="A test standard",
            requirements=["req1", "req2"],
        )
        assert standard.id == "TEST-001"
        assert standard.name == "Test Standard"


class TestComplianceLevel:
    """Test Compliance Level enum."""

    def test_compliance_levels(self):
        """Test compliance level values."""
        assert ComplianceLevel.NONE.value == 0
        assert ComplianceLevel.PARTIAL.value == 1
        assert ComplianceLevel.FULL.value == 2


class TestStandardsRegistry:
    """Test Standards Registry."""

    def test_init(self):
        """Test registry initialization."""
        registry = StandardsRegistry()
        assert registry is not None

    def test_register_standard(self):
        """Test registering a standard."""
        registry = StandardsRegistry()
        standard = Standard(
            id="REG-001",
            name="Registration Test",
            description="Test registration",
            requirements=["req1"],
        )
        registry.register(standard)
        assert "REG-001" in registry

    def test_get_standard(self):
        """Test retrieving a standard."""
        registry = StandardsRegistry()
        standard = Standard(
            id="GET-001",
            name="Get Test",
            description="Test get",
            requirements=["req1"],
        )
        registry.register(standard)

        retrieved = registry.get("GET-001")
        assert retrieved == standard

    def test_list_standards(self):
        """Test listing all standards."""
        registry = StandardsRegistry()
        standards = registry.list_all()
        assert isinstance(standards, list)


class TestRegisterStandard:
    """Test standalone register function."""

    def test_register_global_standard(self):
        """Test registering to global registry."""
        standard = Standard(
            id="GLOBAL-001",
            name="Global Test",
            description="Test global registration",
            requirements=["req1"],
        )
        register_standard(standard)
        # Should be accessible via get_standard
        retrieved = get_standard("GLOBAL-001")
        assert retrieved is not None


class TestGetStandard:
    """Test standalone get function."""

    def test_get_existing_standard(self):
        """Test getting an existing standard."""
        # Register first
        standard = Standard(
            id="EXIST-001",
            name="Existing",
            description="Test existing",
            requirements=["req1"],
        )
        register_standard(standard)

        result = get_standard("EXIST-001")
        assert result is not None
        assert result.name == "Existing"

    def test_get_nonexistent_standard(self):
        """Test getting non-existent standard."""
        result = get_standard("NONEXISTENT-999")
        assert result is None


class TestListStandards:
    """Test listing standards."""

    def test_list_returns_standards(self):
        """Test that listing returns standards."""
        standards = list_standards()
        assert isinstance(standards, list)

    def test_list_filtered_by_category(self):
        """Test filtering standards by category."""
        standards = list_standards(category="validation")
        assert isinstance(standards, list)


class TestCheckCompliance:
    """Test compliance checking."""

    def test_check_full_compliance(self):
        """Test checking full compliance."""
        standard = Standard(
            id="COMP-001",
            name="Compliance Test",
            description="Test compliance",
            requirements=["req1", "req2"],
        )
        register_standard(standard)

        implementation = {"req1": True, "req2": True}
        result = check_compliance("COMP-001", implementation)
        assert result.level == ComplianceLevel.FULL

    def test_check_partial_compliance(self):
        """Test checking partial compliance."""
        standard = Standard(
            id="PART-001",
            name="Partial Test",
            description="Test partial",
            requirements=["req1", "req2", "req3"],
        )
        register_standard(standard)

        implementation = {"req1": True, "req2": False, "req3": True}
        result = check_compliance("PART-001", implementation)
        assert result.level == ComplianceLevel.PARTIAL


class TestSevenStandards:
    """Test the seven standards constant."""

    def test_seven_standards_exist(self):
        """Test that seven standards are defined."""
        assert SEVEN_STANDARDS is not None
        assert len(SEVEN_STANDARDS) == 7

    def test_seven_standards_are_valid(self):
        """Test that all seven standards are valid."""
        for standard in SEVEN_STANDARDS:
            assert isinstance(standard, Standard)
            assert standard.id is not None
            assert standard.name is not None
