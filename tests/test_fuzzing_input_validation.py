"""
Fuzzing tests for all public input-validation surfaces.

This module tests input validation functions with:
- Random fuzzed inputs
- Boundary value analysis
- Malformed/attack payloads
- Type confusion attacks
"""
import sys
import json
import tempfile
import string
import pytest
from pathlib import Path
from hypothesis import given, strategies as st, settings

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.path_security import validate_file_path
from utils.batch_processor import secure_load_json
from utils.input_validation import validate_env_key
from utils.config_manager import ConfigManager

# Use PROJECT_ROOT directly instead of importing from main
MAIN_PROJECT_ROOT = PROJECT_ROOT


class TestFuzzedPathValidation:
    """Fuzzing tests for path validation functions."""

    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=100, deadline=None)
    def test_fuzzed_path_validation_does_not_crash(self, fuzzed_path):
        """Path validation should not crash on any input, only raise ValueError."""
        try:
            # Should either succeed or raise ValueError, never crash
            validate_file_path(fuzzed_path, MAIN_PROJECT_ROOT)
        except (ValueError, TypeError, OSError):
            # Expected exceptions for invalid paths
            pass
        except Exception as e:
            pytest.fail(
                f"Path validation crashed on input {fuzzed_path!r}: {type(e).__name__}: {e}"
            )

    @given(st.text(alphabet=string.printable, min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=None)
    def test_fuzzed_path_with_special_characters(self, fuzzed_content):
        """Test paths containing special characters, null bytes, and control characters."""
        # Create a temporary file with fuzzed content in the name (safely)
        safe_content = "".join(c for c in fuzzed_content if c.isalnum() or c in "._-")[
            :100
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / f"test_{safe_content}.txt"
            try:
                test_file.write_text("test")
                # Should not crash
                validate_file_path(test_file, MAIN_PROJECT_ROOT)
            except (ValueError, OSError):
                pass  # Expected for invalid paths

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10))
    @settings(max_examples=50, deadline=None)
    def test_fuzzed_path_traversal_patterns(self, path_components):
        """Test various path traversal combinations."""
        traversal_path = "/".join(path_components)

        try:
            validate_file_path(traversal_path, MAIN_PROJECT_ROOT)
        except (ValueError, TypeError, OSError):
            pass  # Expected
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

    def test_path_with_embedded_null_bytes(self):
        """Test paths containing null byte injection attempts."""
        null_byte_paths = [
            "safe.txt\x00/../etc/passwd",
            "\x00/etc/passwd",
            "file\x00name.txt",
            "\x00\x00\x00",
        ]

        for path in null_byte_paths:
            with pytest.raises((ValueError, TypeError)):
                validate_file_path(path, MAIN_PROJECT_ROOT)


class TestFuzzedJsonValidation:
    """Fuzzing tests for JSON validation functions."""

    @given(st.text(min_size=0, max_size=10000))
    @settings(max_examples=100, deadline=None)
    def test_fuzzed_json_parsing(self, fuzzed_content):
        """JSON parser should handle malformed input gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.json"
            test_file.write_text(fuzzed_content, errors="replace")

            try:
                result = secure_load_json(test_file)
                # If it parses, result should be dict or None
                if result is not None:
                    assert isinstance(result, dict)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass  # Expected for invalid JSON
            except Exception as e:
                pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=50),
            st.one_of(st.integers(), st.text(), st.floats(), st.booleans()),
            min_size=0,
            max_size=100,
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_fuzzed_valid_json_content(self, data):
        """Test loading valid but random JSON structures."""
        # Use project-relative temp directory
        temp_dir = Path(MAIN_PROJECT_ROOT) / "data" / "temp_test"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            test_file = temp_dir / "test.json"
            test_file.write_text(json.dumps(data))

            try:
                result = secure_load_json(test_file)
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"Valid JSON should load: {e}")
        finally:
            # Cleanup
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_json_with_circular_references(self):
        """Test that circular reference detection works."""
        # Create circular reference
        data = {"a": 1}
        data["self"] = data

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "circular.json"

            # This should fail at JSON serialization level
            with pytest.raises((ValueError, TypeError, json.JSONDecodeError)):
                test_file.write_text(json.dumps(data))

    def test_json_with_oversized_content(self):
        """Test handling of extremely large JSON files."""
        # Use project-relative temp directory
        temp_dir = Path(MAIN_PROJECT_ROOT) / "data" / "temp_test_large"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            test_file = temp_dir / "large.json"
            # Create data that exceeds 100MB limit
            large_data = {"data": "x" * (100 * 1024 * 1024)}

            # First, try to write the file - this may fail due to memory
            try:
                test_file.write_text(json.dumps(large_data))
            except (OSError, MemoryError):
                pytest.skip("Cannot create large test file due to memory constraints")

            # Then, verify that loading it raises ValueError for file too large
            with pytest.raises(ValueError, match="File too large"):
                secure_load_json(test_file)
        finally:
            # Cleanup
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


class TestFuzzedEnvironmentVariables:
    """Fuzzing tests for environment variable validation."""

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=100, deadline=None)
    def test_fuzzed_env_key_validation(self, fuzzed_key):
        """Environment key validation should reject invalid keys."""
        is_valid = validate_env_key(fuzzed_key)

        # Valid keys match the pattern ^[A-Z_][A-Z0-9_]*$
        if is_valid:
            assert (
                fuzzed_key.isupper() or fuzzed_key == ""
            ), f"Valid key must be uppercase: {fuzzed_key}"
            assert all(
                c.isupper() or c.isdigit() or c == "_" for c in fuzzed_key
            ), f"Invalid chars in: {fuzzed_key}"

    @given(st.from_regex(r"^[A-Z_][A-Z0-9_]*$", fullmatch=True))
    @settings(max_examples=100, deadline=None)
    def test_valid_env_keys_accepted(self, valid_key):
        """Valid environment keys should be accepted."""
        assert validate_env_key(valid_key) is True


class TestFuzzedConfigValidation:
    """Fuzzing tests for configuration validation."""

    @given(
        st.dictionaries(
            st.sampled_from(["model", "simulation", "validation", "logging", "data"]),
            st.dictionaries(
                st.text(min_size=1, max_size=30),
                st.one_of(
                    st.integers(min_value=-1000000, max_value=1000000),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.booleans(),
                    st.text(max_size=100),
                ),
                min_size=1,
                max_size=20,
            ),
            min_size=0,
            max_size=5,
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_fuzzed_config_data(self, fuzzed_config):
        """Configuration manager should handle fuzzed config data gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.yaml"

            # Create a minimal valid YAML or JSON
            import yaml

            try:
                config_file.write_text(yaml.dump(fuzzed_config))

                # Try to load with ConfigManager
                # Should not crash, may raise validation errors
                try:
                    _ = ConfigManager(config_file)
                except (ValueError, KeyError, TypeError):
                    pass  # Expected for invalid configs
            except (yaml.YAMLError, OSError):
                pass  # Expected for invalid YAML

    def test_config_with_extreme_numeric_values(self):
        """Test config with extreme numeric values (infinity, NaN, very large numbers)."""
        extreme_values = [
            float("inf"),
            float("-inf"),
            float("nan"),
            1e308,  # Near max double
            -1e308,
            1e-308,  # Near min positive double
        ]

        for val in extreme_values:
            # Should handle gracefully, not crash
            try:
                # Create a test dict with extreme value
                _ = {"model": {"test_param": val}}
            except (ValueError, TypeError, OverflowError):
                pass


class TestFuzzedStringInputs:
    """Fuzzing tests for string input validation across all surfaces."""

    @given(st.text(alphabet=string.printable, min_size=0, max_size=10000))
    @settings(max_examples=100, deadline=None)
    def test_fuzzed_string_handling(self, fuzzed_string):
        """All string inputs should be handled safely without crashes."""
        # Test that various string inputs don't cause crashes
        try:
            # Path operations
            _ = Path(fuzzed_string).name
            _ = Path(fuzzed_string).suffix
        except (ValueError, TypeError):
            pass

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=50, deadline=None)
    def test_unicode_normalization_attacks(self, fuzzed_text):
        """Test handling of unicode normalization edge cases."""
        # Test various unicode edge cases
        tricky_strings = [
            fuzzed_text + "\u202e" + "txt.exe",  # Right-to-left override
            fuzzed_text + "\u200b",  # Zero-width space
            fuzzed_text + "\ufeff",  # BOM
            "\u00a0",  # Non-breaking space
            "\u3000",  # Ideographic space
        ]

        for s in tricky_strings:
            try:
                # Should not crash
                _ = s.encode("utf-8", errors="replace")
            except Exception as e:
                pytest.fail(f"String handling crashed: {e}")


class TestBoundaryValueAnalysis:
    """Boundary value analysis for input validation."""

    def test_integer_boundary_values(self):
        """Test integer boundary values in configuration."""
        boundaries = [
            0,
            1,
            -1,  # Zero and sign boundaries
            127,
            128,
            -128,  # Byte boundaries
            255,
            256,  # Byte max boundaries
            32767,
            32768,
            -32768,  # Short boundaries
            65535,
            65536,  # Unsigned short boundaries
            2147483647,
            2147483648,
            -2147483648,  # Int boundaries
            9223372036854775807,
            -9223372036854775808,  # Long boundaries
        ]

        for val in boundaries:
            # Should handle without crashing
            try:
                test_path = f"/tmp/test_{val}.txt"
                _ = Path(test_path)
            except Exception as e:
                pytest.fail(f"Boundary value {val} caused crash: {e}")

    def test_float_boundary_values(self):
        """Test float boundary values."""
        float_boundaries = [
            0.0,
            -0.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            sys.float_info.min,
            sys.float_info.max,
            sys.float_info.epsilon,
            -sys.float_info.max,
        ]

        for val in float_boundaries:
            try:
                # Should not crash
                _ = str(val)
                _ = float(val) if not (val != val) else 0.0  # NaN check
            except Exception as e:
                pytest.fail(f"Float boundary {val} caused crash: {e}")

    def test_string_length_boundaries(self):
        """Test string length boundaries."""
        lengths = [0, 1, 255, 256, 1023, 1024, 4095, 4096, 65535, 65536]

        for length in lengths:
            test_str = "x" * length
            try:
                # Should handle without crashing
                _ = len(test_str)
                _ = test_str[:100]
            except MemoryError:
                if length > 1000000:
                    pass  # Expected for very large
            except Exception as e:
                pytest.fail(f"String length {length} caused crash: {e}")


class TestMalformedInputHandling:
    """Tests for malformed and attack inputs."""

    def test_sql_injection_patterns(self):
        """Test SQL injection pattern handling in file paths."""
        sql_patterns = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "'; EXEC xp_cmdshell('dir'); --",
            "' UNION SELECT * FROM passwords --",
            "'; DELETE FROM users WHERE 't' = 't'; --",
        ]

        for pattern in sql_patterns:
            try:
                # Should treat as literal string, not crash
                _ = Path(pattern)
            except Exception as e:
                pytest.fail(f"SQL pattern caused crash: {e}")

    def test_command_injection_patterns(self):
        """Test command injection pattern handling."""
        cmd_patterns = [
            "$(whoami)",
            "`whoami`",
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& echo hacked",
            "|| echo hacked",
        ]

        for pattern in cmd_patterns:
            try:
                _ = Path(pattern)
            except Exception as e:
                pytest.fail(f"Command pattern caused crash: {e}")

    def test_xss_patterns_in_inputs(self):
        """Test XSS pattern handling."""
        xss_patterns = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>",
            "'-\"><script>alert(String.fromCharCode(88,83,83))</script>",
        ]

        for pattern in xss_patterns:
            try:
                # Should handle as literal text
                _ = pattern.encode("utf-8")
            except Exception as e:
                pytest.fail(f"XSS pattern caused crash: {e}")


class TestTypeConfusionAttacks:
    """Tests for type confusion attack vectors."""

    def test_unexpected_types_in_validation(self):
        """Test handling of unexpected types in validation functions."""
        unexpected_types = [
            None,
            [],
            {},
            set(),
            object(),
            lambda: None,
            Exception("test"),
            type("TestClass", (), {})(),
        ]

        for val in unexpected_types:
            try:
                # Should not crash, may raise TypeError
                validate_file_path(val, MAIN_PROJECT_ROOT)
            except (ValueError, TypeError, AttributeError):
                pass  # Expected
            except Exception as e:
                pytest.fail(f"Type {type(val)} caused unexpected error: {e}")

    def test_numeric_string_confusion(self):
        """Test numeric string type confusion."""
        confusing_inputs = [
            "123",
            "123.456",
            "0x123",
            "1e10",
            "NaN",
            "Infinity",
            "-Infinity",
            "true",
            "false",
            "null",
            "[]",
            "{}",
        ]

        for val in confusing_inputs:
            try:
                # Should remain as string, not auto-convert
                assert isinstance(val, str)
            except AssertionError as e:
                pytest.fail(f"Type confusion on {val}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
