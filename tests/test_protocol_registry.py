"""
Comprehensive Tests for Protocol Registry Module
================================================

Target: 100% coverage for utils/protocol_registry.py
"""

import sys
from pathlib import Path

import pytest

# Ensure utils is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.protocol_registry import (
    PROTOCOL_REGISTRY,
    ProtocolInfo,
    _initialize_registry,
    get_protocol_info,
    get_protocol_registry,
    list_available_protocols,
    resolve_protocol,
)


class TestProtocolInfo:
    """Test ProtocolInfo dataclass"""

    def test_protocol_info_basic(self):
        """Test basic ProtocolInfo creation"""
        info = ProtocolInfo(
            canonical_id="TEST1",
            filename="test.py",
            title="Test Protocol",
            description="A test protocol",
            priority_level="P1",
            category="Test",
        )
        assert info.canonical_id == "TEST1"
        assert info.filename == "test.py"
        assert info.title == "Test Protocol"
        assert info.description == "A test protocol"
        assert info.priority_level == "P1"
        assert info.category == "Test"
        assert info.implemented is True
        assert info.aliases == []

    def test_protocol_info_with_aliases(self):
        """Test ProtocolInfo with aliases"""
        info = ProtocolInfo(
            canonical_id="TEST2",
            filename="test2.py",
            title="Test Protocol 2",
            description="Another test",
            priority_level="P2",
            category="Core",
            implemented=False,
            aliases=["alias1", "alias2"],
        )
        assert info.implemented is False
        assert info.aliases == ["alias1", "alias2"]

    def test_protocol_info_post_init_none_aliases(self):
        """Test that None aliases are converted to empty list"""
        info = ProtocolInfo(
            canonical_id="TEST3",
            filename="test3.py",
            title="Test",
            description="Test desc",
            priority_level="P1",
            category="Test",
            aliases=None,
        )
        assert info.aliases == []


class TestProtocolRegistry:
    """Test PROTOCOL_REGISTRY class methods"""

    def test_register_protocol(self):
        """Test registering a new protocol"""
        # Clear registry first
        PROTOCOL_REGISTRY._protocols.clear()

        info = ProtocolInfo(
            canonical_id="REG1",
            filename="reg1.py",
            title="Registered Protocol",
            description="Test registration",
            priority_level="P1",
            category="Test",
            aliases=["REG1_ALIAS"],
        )

        PROTOCOL_REGISTRY.register_protocol(info)

        # Check canonical ID
        retrieved = PROTOCOL_REGISTRY.get_protocol("REG1")
        assert retrieved is not None
        assert retrieved.canonical_id == "REG1"

        # Check alias
        retrieved_alias = PROTOCOL_REGISTRY.get_protocol("REG1_ALIAS")
        assert retrieved_alias is not None
        assert retrieved_alias.canonical_id == "REG1"

    def test_get_protocol_not_found(self):
        """Test getting non-existent protocol"""
        result = PROTOCOL_REGISTRY.get_protocol("NONEXISTENT")
        assert result is None

    def test_get_filename(self):
        """Test getting filename for protocol"""
        PROTOCOL_REGISTRY._protocols.clear()

        info = ProtocolInfo(
            canonical_id="FILE1",
            filename="file1.py",
            title="File Test",
            description="Testing filename retrieval",
            priority_level="P1",
            category="Test",
        )
        PROTOCOL_REGISTRY.register_protocol(info)

        filename = PROTOCOL_REGISTRY.get_filename("FILE1")
        assert filename == "file1.py"

    def test_get_filename_not_found(self):
        """Test getting filename for non-existent protocol"""
        filename = PROTOCOL_REGISTRY.get_filename("NOTFOUND")
        assert filename is None

    def test_list_protocols_all(self):
        """Test listing all protocols"""
        PROTOCOL_REGISTRY._protocols.clear()

        # Add multiple protocols
        for i in range(3):
            info = ProtocolInfo(
                canonical_id=f"LIST{i}",
                filename=f"list{i}.py",
                title=f"List Test {i}",
                description=f"Testing list {i}",
                priority_level="P1",
                category="Category1" if i < 2 else "Category2",
            )
            PROTOCOL_REGISTRY.register_protocol(info)

        protocols = PROTOCOL_REGISTRY.list_protocols()
        assert len(protocols) == 3

    def test_list_protocols_filtered(self):
        """Test listing protocols filtered by category"""
        PROTOCOL_REGISTRY._protocols.clear()

        # Add protocols with different categories
        for i in range(4):
            info = ProtocolInfo(
                canonical_id=f"FILT{i}",
                filename=f"filt{i}.py",
                title=f"Filter Test {i}",
                description=f"Testing filter {i}",
                priority_level="P1",
                category="CatA" if i < 2 else "CatB",
            )
            PROTOCOL_REGISTRY.register_protocol(info)

        cat_a = PROTOCOL_REGISTRY.list_protocols("CatA")
        cat_b = PROTOCOL_REGISTRY.list_protocols("CatB")

        assert len(cat_a) == 2
        assert len(cat_b) == 2
        assert all(p.category == "CatA" for p in cat_a)
        assert all(p.category == "CatB" for p in cat_b)

    def test_list_protocols_deduplication(self):
        """Test that list_protocols removes duplicates from aliases"""
        PROTOCOL_REGISTRY._protocols.clear()

        info = ProtocolInfo(
            canonical_id="DEDUP",
            filename="dedup.py",
            title="Deduplication Test",
            description="Testing dedup",
            priority_level="P1",
            category="Test",
            aliases=["ALIAS1", "ALIAS2"],
        )
        PROTOCOL_REGISTRY.register_protocol(info)

        # Registry has 3 entries (canonical + 2 aliases)
        assert len(PROTOCOL_REGISTRY._protocols) == 3

        # But list_protocols returns only 1 (unique canonical IDs)
        protocols = PROTOCOL_REGISTRY.list_protocols()
        assert len(protocols) == 1
        assert protocols[0].canonical_id == "DEDUP"

    def test_resolve_protocol_file(self):
        """Test resolving protocol ID to file path"""
        PROTOCOL_REGISTRY._protocols.clear()

        info = ProtocolInfo(
            canonical_id="RESOLVE",
            filename="subdir/resolve.py",
            title="Resolve Test",
            description="Testing resolution",
            priority_level="P1",
            category="Test",
        )
        PROTOCOL_REGISTRY.register_protocol(info)

        base_dir = Path("/test/dir")
        result = PROTOCOL_REGISTRY.resolve_protocol_file("RESOLVE", base_dir)

        assert result == Path("/test/dir/subdir/resolve.py")

    def test_resolve_protocol_file_not_found(self):
        """Test resolving non-existent protocol"""
        result = PROTOCOL_REGISTRY.resolve_protocol_file("NOEXIST", Path("/test"))
        assert result is None


class TestInitializeRegistry:
    """Test the _initialize_registry function"""

    def test_initialize_registry(self):
        """Test that _initialize_registry populates known protocols"""
        PROTOCOL_REGISTRY._protocols.clear()

        _initialize_registry()

        # Check that known protocols are registered
        assert PROTOCOL_REGISTRY.get_protocol("P1") is not None
        assert PROTOCOL_REGISTRY.get_protocol("P2") is not None
        assert PROTOCOL_REGISTRY.get_protocol("F2-Iowa") is not None
        assert PROTOCOL_REGISTRY.get_protocol("BayesianEstimation-MCMC") is not None

    def test_initialize_registry_aliases(self):
        """Test that aliases are properly registered"""
        PROTOCOL_REGISTRY._protocols.clear()

        _initialize_registry()

        # Check aliases work
        assert PROTOCOL_REGISTRY.get_protocol("F1") is not None  # Alias for P1
        assert PROTOCOL_REGISTRY.get_protocol("F2") is not None  # Alias for P1
        assert (
            PROTOCOL_REGISTRY.get_protocol("MCMC") is not None
        )  # Alias for BayesianEstimation-MCMC


class TestConvenienceFunctions:
    """Test module-level convenience functions"""

    def test_get_protocol_registry(self):
        """Test get_protocol_registry function"""
        registry = get_protocol_registry()
        assert registry is PROTOCOL_REGISTRY

    def test_resolve_protocol(self):
        """Test resolve_protocol function"""
        PROTOCOL_REGISTRY._protocols.clear()

        info = ProtocolInfo(
            canonical_id="CONV1",
            filename="conv1.py",
            title="Convenience Test",
            description="Testing convenience func",
            priority_level="P1",
            category="Test",
        )
        PROTOCOL_REGISTRY.register_protocol(info)

        result = resolve_protocol("CONV1")
        assert result is not None
        assert result.name == "conv1.py"

    def test_resolve_protocol_with_base_dir(self):
        """Test resolve_protocol with custom base dir"""
        PROTOCOL_REGISTRY._protocols.clear()

        info = ProtocolInfo(
            canonical_id="CONV2",
            filename="custom/path.py",
            title="Custom Path Test",
            description="Testing custom path",
            priority_level="P1",
            category="Test",
        )
        PROTOCOL_REGISTRY.register_protocol(info)

        custom_dir = Path("/custom/base")
        result = resolve_protocol("CONV2", custom_dir)
        assert result == Path("/custom/base/custom/path.py")

    def test_resolve_protocol_not_found(self):
        """Test resolve_protocol with non-existent protocol"""
        result = resolve_protocol("NONEXISTENT_PROTOCOL")
        assert result is None

    def test_list_available_protocols(self):
        """Test list_available_protocols function"""
        PROTOCOL_REGISTRY._protocols.clear()

        # Add test protocols
        for i in range(3):
            info = ProtocolInfo(
                canonical_id=f"AVAIL{i}",
                filename=f"avail{i}.py",
                title=f"Available {i}",
                description=f"Testing available {i}",
                priority_level="P1",
                category="AvailableTest",
            )
            PROTOCOL_REGISTRY.register_protocol(info)

        result = list_available_protocols()
        assert len(result) == 3
        assert all(f"AVAIL{i}" in result for i in range(3))

    def test_list_available_protocols_filtered(self):
        """Test list_available_protocols with category filter"""
        PROTOCOL_REGISTRY._protocols.clear()

        # Add protocols in different categories
        for i in range(4):
            info = ProtocolInfo(
                canonical_id=f"FILTER{i}",
                filename=f"filter{i}.py",
                title=f"Filter {i}",
                description=f"Testing filter {i}",
                priority_level="P1",
                category="FilterCatA" if i < 2 else "FilterCatB",
            )
            PROTOCOL_REGISTRY.register_protocol(info)

        result_a = list_available_protocols("FilterCatA")
        result_b = list_available_protocols("FilterCatB")

        assert len(result_a) == 2
        assert len(result_b) == 2

    def test_get_protocol_info(self):
        """Test get_protocol_info function"""
        PROTOCOL_REGISTRY._protocols.clear()

        info = ProtocolInfo(
            canonical_id="INFO1",
            filename="info1.py",
            title="Info Test",
            description="Testing get_protocol_info",
            priority_level="P1",
            category="Test",
        )
        PROTOCOL_REGISTRY.register_protocol(info)

        result = get_protocol_info("INFO1")
        assert result is not None
        assert result.canonical_id == "INFO1"
        assert result.title == "Info Test"

    def test_get_protocol_info_not_found(self):
        """Test get_protocol_info with non-existent protocol"""
        result = get_protocol_info("NOT_REAL")
        assert result is None


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_registry_list(self):
        """Test listing protocols from empty registry"""
        PROTOCOL_REGISTRY._protocols.clear()

        protocols = PROTOCOL_REGISTRY.list_protocols()
        assert protocols == []

    def test_empty_registry_available(self):
        """Test listing available protocols from empty registry"""
        PROTOCOL_REGISTRY._protocols.clear()

        available = list_available_protocols()
        assert available == []

    def test_protocol_info_empty_strings(self):
        """Test ProtocolInfo with empty strings"""
        info = ProtocolInfo(
            canonical_id="",
            filename="",
            title="",
            description="",
            priority_level="",
            category="",
        )
        assert info.canonical_id == ""
        assert info.aliases == []

    def test_register_protocol_overwrite(self):
        """Test that registering same ID overwrites previous"""
        PROTOCOL_REGISTRY._protocols.clear()

        info1 = ProtocolInfo(
            canonical_id="OVER",
            filename="v1.py",
            title="Version 1",
            description="First version",
            priority_level="P1",
            category="Test",
        )
        PROTOCOL_REGISTRY.register_protocol(info1)

        info2 = ProtocolInfo(
            canonical_id="OVER",
            filename="v2.py",
            title="Version 2",
            description="Second version",
            priority_level="P2",
            category="Test",
        )
        PROTOCOL_REGISTRY.register_protocol(info2)

        result = PROTOCOL_REGISTRY.get_protocol("OVER")
        assert result.filename == "v2.py"
        assert result.priority_level == "P2"

    def test_register_protocol_multiple_aliases_same_id(self):
        """Test registering protocol with multiple aliases pointing to same"""
        PROTOCOL_REGISTRY._protocols.clear()

        info = ProtocolInfo(
            canonical_id="MULTI",
            filename="multi.py",
            title="Multi Alias",
            description="Testing multiple aliases",
            priority_level="P1",
            category="Test",
            aliases=["A", "B", "C", "D"],
        )
        PROTOCOL_REGISTRY.register_protocol(info)

        # All aliases should point to same protocol
        assert PROTOCOL_REGISTRY.get_protocol("A").canonical_id == "MULTI"
        assert PROTOCOL_REGISTRY.get_protocol("B").canonical_id == "MULTI"
        assert PROTOCOL_REGISTRY.get_protocol("C").canonical_id == "MULTI"
        assert PROTOCOL_REGISTRY.get_protocol("D").canonical_id == "MULTI"


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry to known state after each test"""
    yield
    # Re-initialize with known protocols after tests
    PROTOCOL_REGISTRY._protocols.clear()
    _initialize_registry()
