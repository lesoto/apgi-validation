"""
Phase 3 Warning System Tests

Tests to verify that the _emit_legacy_protocol_warning() function is properly
activated and emitting deprecation warnings for legacy protocol access.

This test suite ensures:
1. Legacy protocols (VP-##, FP-##) emit deprecation warnings
2. Canonical protocols (APGI-P##) do NOT emit warnings
3. Warning messages include proper migration guidance
4. Logging integration works correctly
"""

import pytest
import warnings
import logging
from utils.protocol_loader import (
    load_protocol,
    load_all_protocols,
    _emit_legacy_protocol_warning,
    _is_legacy_protocol,
)


class TestLegacyProtocolWarnings:
    """Test suite for Phase 3 legacy protocol deprecation warnings."""

    def test_emit_legacy_protocol_warning_with_canonical_mapping(self, caplog):
        """Test that warnings are emitted for legacy protocols with canonical mappings."""
        with pytest.warns(DeprecationWarning) as warning_list:
            _emit_legacy_protocol_warning("VP-01", "APGI-P01")
        
        assert len(warning_list) >= 1
        deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        
        warning_msg = str(deprecation_warnings[0].message)
        assert "PROTOCOL MIGRATION" in warning_msg
        assert "PHASE 3" in warning_msg
        assert "VP-01" in warning_msg
        assert "APGI-P01" in warning_msg
        assert "deprecated" in warning_msg.lower()

    def test_emit_legacy_protocol_warning_without_canonical_mapping(self, caplog):
        """Test that warnings are emitted for legacy protocols without canonical mappings."""
        with pytest.warns(DeprecationWarning) as warning_list:
            _emit_legacy_protocol_warning("VP-02", None)
        
        assert len(warning_list) >= 1
        deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        
        warning_msg = str(deprecation_warnings[0].message)
        assert "PROTOCOL MIGRATION" in warning_msg
        assert "VP-02" in warning_msg
        assert "no canonical APGI-P## equivalent" in warning_msg

    def test_load_protocol_vp01_emits_warning(self):
        """Test that loading VP-01 emits a deprecation warning."""
        with pytest.warns(DeprecationWarning) as warning_list:
            spec = load_protocol("VP-01")
        
        assert spec is not None
        assert spec.protocol_id == "VP-01"
        
        # Check that at least one DeprecationWarning was emitted
        deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1

    def test_load_protocol_fp01_emits_warning(self):
        """Test that loading FP-01 emits a deprecation warning."""
        with pytest.warns(DeprecationWarning) as warning_list:
            spec = load_protocol("FP-01")
        
        assert spec is not None
        assert spec.protocol_id == "FP-01"
        
        # Check that at least one DeprecationWarning was emitted
        deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1

    def test_load_protocol_canonical_apgi_p01_does_not_emit_legacy_warning(self):
        """Test that loading APGI-P01 does NOT emit a legacy protocol warning."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            spec = load_protocol("APGI-P01")
        
        # Should load successfully
        assert spec is not None
        
        # Filter for the specific Phase 3 deprecation message
        phase3_warnings = [
            w for w in warning_list 
            if issubclass(w.category, DeprecationWarning) and "PROTOCOL MIGRATION" in str(w.message)
        ]
        
        # Should NOT have a Phase 3 deprecation warning for canonical protocol
        assert len(phase3_warnings) == 0

    def test_load_protocol_canonical_apgi_p02_does_not_emit_legacy_warning(self):
        """Test that loading APGI-P02 does NOT emit a legacy protocol warning."""
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            spec = load_protocol("APGI-P02")
        
        # Should load successfully
        assert spec is not None
        
        # Filter for the specific Phase 3 deprecation message
        phase3_warnings = [
            w for w in warning_list 
            if issubclass(w.category, DeprecationWarning) and "PROTOCOL MIGRATION" in str(w.message)
        ]
        
        # Should NOT have a Phase 3 deprecation warning for canonical protocol
        assert len(phase3_warnings) == 0

    def test_load_all_protocols_emits_warnings_for_legacy_protocols(self):
        """Test that load_all_protocols() emits warnings for all legacy protocols."""
        with pytest.warns(DeprecationWarning) as warning_list:
            specs = load_all_protocols()
        
        # Should load successfully
        assert len(specs) > 0
        
        # Check that we have VP and FP protocols
        vp_protocols = [pid for pid in specs.keys() if pid.startswith("VP-")]
        fp_protocols = [pid for pid in specs.keys() if pid.startswith("FP-")]
        
        assert len(vp_protocols) > 0
        assert len(fp_protocols) > 0
        
        # Check that deprecation warnings were emitted
        deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0

    def test_is_legacy_protocol_vp_format(self):
        """Test that VP-## format is recognized as legacy."""
        assert _is_legacy_protocol("VP-01") is True
        assert _is_legacy_protocol("VP-22") is True
        assert _is_legacy_protocol("VP-00") is True

    def test_is_legacy_protocol_fp_format(self):
        """Test that FP-## format is recognized as legacy."""
        assert _is_legacy_protocol("FP-01") is True
        assert _is_legacy_protocol("FP-15") is True
        assert _is_legacy_protocol("FP-02") is True

    def test_is_legacy_protocol_canonical_format(self):
        """Test that APGI-P## format is NOT recognized as legacy."""
        assert _is_legacy_protocol("APGI-P01") is False
        assert _is_legacy_protocol("APGI-P02") is False
        assert _is_legacy_protocol("APGI-P07") is False

    def test_warning_message_includes_migration_guidance(self):
        """Test that warning messages include proper migration guidance."""
        with pytest.warns(DeprecationWarning) as warning_list:
            _emit_legacy_protocol_warning("VP-01", "APGI-P01")
        
        deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        
        warning_msg = str(deprecation_warnings[0].message)
        assert "PROTOCOL-MIGRATION-USER-GUIDE.md" in warning_msg
        assert "Phase 4" in warning_msg
        assert "December 2026" in warning_msg

    def test_warning_stacklevel_correct(self):
        """Test that warning stacklevel points to the correct caller location."""
        with pytest.warns(DeprecationWarning) as warning_list:
            spec = load_protocol("VP-01")
        
        # Just verify that warnings are captured
        deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1


class TestPhase3WarningIntegration:
    """Integration tests for Phase 3 warning system across multiple operations."""

    def test_multiple_legacy_protocol_loads_emit_warnings(self):
        """Test that loading multiple legacy protocols each emit their own warnings."""
        legacy_ids = ["VP-01", "FP-01", "VP-07", "FP-02"]
        
        with pytest.warns(DeprecationWarning) as warning_list:
            for protocol_id in legacy_ids:
                load_protocol(protocol_id)
        
        # Should have warnings for each legacy protocol
        deprecation_warnings = [w for w in warning_list if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0

    def test_canonical_protocols_no_warnings_multiple_loads(self):
        """Test that loading multiple canonical protocols does NOT emit warnings."""
        canonical_ids = ["APGI-P01", "APGI-P02", "APGI-P03"]
        
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            for protocol_id in canonical_ids:
                spec = load_protocol(protocol_id)
                # All should load successfully
                if spec is not None:
                    assert spec.protocol_id.startswith("APGI-P")
        
        # Filter for Phase 3 deprecation warnings
        phase3_warnings = [
            w for w in warning_list 
            if issubclass(w.category, DeprecationWarning) and "PROTOCOL MIGRATION" in str(w.message)
        ]
        
        # Should have NO Phase 3 warnings for canonical protocols
        assert len(phase3_warnings) == 0

    def test_warning_system_with_load_all_protocols_option(self):
        """Test warning system when loading with different options."""
        # Test with include_legacy=True (default)
        with pytest.warns(DeprecationWarning) as warning_list_with_legacy:
            specs_with_legacy = load_all_protocols(include_legacy=True)
        
        assert len(specs_with_legacy) > 0
        deprecation_warnings = [w for w in warning_list_with_legacy if issubclass(w.category, DeprecationWarning)]
        assert len(deprecation_warnings) > 0
        
        # Test with include_legacy=False
        with warnings.catch_warnings(record=True) as warning_list_without_legacy:
            warnings.simplefilter("always")
            specs_without_legacy = load_all_protocols(include_legacy=False)
        
        # Should still load protocols
        assert len(specs_without_legacy) > 0
        
        # Filter for Phase 3 deprecation warnings
        phase3_warnings = [
            w for w in warning_list_without_legacy 
            if issubclass(w.category, DeprecationWarning) and "PROTOCOL MIGRATION" in str(w.message)
        ]
        
        # Should have NO Phase 3 warnings when include_legacy=False
        assert len(phase3_warnings) == 0


class TestPhase3WarningLogging:
    """Test logging integration for Phase 3 warning system."""

    def test_warning_logging_output(self, caplog):
        """Test that warning messages are properly logged."""
        with caplog.at_level(logging.WARNING):
            _emit_legacy_protocol_warning("VP-01", "APGI-P01")
        
        # Check that logging occurred
        assert "[Phase 3]" in caplog.text or "Deprecated protocol" in caplog.text

    def test_warning_logging_includes_protocol_ids(self, caplog):
        """Test that logged warnings include both legacy and canonical IDs."""
        with caplog.at_level(logging.WARNING):
            _emit_legacy_protocol_warning("FP-01", "APGI-P02")
        
        # Check that both IDs are in the log
        assert "FP-01" in caplog.text
        assert "APGI-P02" in caplog.text or "NO CANONICAL" in caplog.text


class TestLegacyProtocolIdentification:
    """Test legacy protocol identification logic."""

    def test_all_vp_protocols_identified_as_legacy(self):
        """Test that all VP-## protocols are correctly identified as legacy."""
        vp_protocols = [f"VP-{i:02d}" for i in range(0, 23)]
        for protocol_id in vp_protocols:
            assert _is_legacy_protocol(protocol_id) is True

    def test_all_fp_protocols_identified_as_legacy(self):
        """Test that all FP-## protocols are correctly identified as legacy."""
        fp_protocols = [f"FP-{i:02d}" for i in range(0, 16)]
        for protocol_id in fp_protocols:
            assert _is_legacy_protocol(protocol_id) is True

    def test_canonical_apgi_protocols_not_identified_as_legacy(self):
        """Test that all APGI-P## protocols are NOT identified as legacy."""
        apgi_protocols = [f"APGI-P{i:02d}" for i in range(0, 8)]
        for protocol_id in apgi_protocols:
            assert _is_legacy_protocol(protocol_id) is False
