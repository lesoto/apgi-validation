"""
Comprehensive tests for path validation security including symlink traversal,
TOCTOU race conditions, absolute path rejection, and other security edge cases.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch
import threading

from utils.path_security import validate_file_path

# Define PROJECT_ROOT locally to avoid main.py import issues
PROJECT_ROOT = Path(__file__).parent.parent


class TestPathValidationSecurity:
    """Test path validation security mechanisms."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def safe_file(self, temp_dir):
        """Create a safe test file."""
        safe_file = temp_dir / "safe_file.txt"
        safe_file.write_text("safe content")
        return safe_file

    def test_symlink_traversal_prevention(self, temp_dir):
        """Test that symlink traversal attacks are prevented."""
        # Create a file outside temp directory
        outside_file = temp_dir.parent / "outside.txt"
        outside_file.write_text("sensitive data")

        # Create symlink pointing outside temp directory
        malicious_link = temp_dir / "malicious_link"
        malicious_link.symlink_to(outside_file)

        # Should raise ValueError for symlink traversal
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_file_path(malicious_link, PROJECT_ROOT)

    def test_relative_path_traversal_prevention(self, temp_dir):
        """Test that relative path traversal is prevented."""
        # Test various traversal patterns
        traversal_patterns = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
        ]

        for pattern in traversal_patterns:
            malicious_path = temp_dir / pattern
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_path(malicious_path, PROJECT_ROOT)

    def test_absolute_path_rejection(self):
        """Test that absolute paths outside allowed directories are rejected."""
        absolute_paths = [
            Path("/etc/passwd"),
            Path("C:\\Windows\\System32\\config"),
            Path("/var/log/auth.log"),
        ]

        for abs_path in absolute_paths:
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_path(abs_path, PROJECT_ROOT)

    def test_toctou_race_condition_protection(self, temp_dir):
        """Test Time-of-Check-Time-of-Use race condition protection."""
        # Create initial safe file
        safe_file = temp_dir / "initial_file.txt"
        safe_file.write_text("initial content")

        # Mock the file operations to simulate TOCTOU
        with patch("pathlib.Path.exists") as mock_exists, patch(
            "pathlib.Path.is_file"
        ) as mock_is_file:
            # First call returns True (file exists)
            mock_exists.return_value = True
            mock_is_file.return_value = True

            # Should pass validation initially
            validate_file_path(safe_file, PROJECT_ROOT)

            # Simulate race condition - second call returns False
            mock_is_file.return_value = False

            # Should detect the race condition
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_path(safe_file, PROJECT_ROOT)

    def test_concurrent_path_validation(self, temp_dir):
        """Test concurrent path validation doesn't have race conditions."""
        safe_file = temp_dir / "concurrent_test.txt"
        safe_file.write_text("concurrent content")

        results = []
        errors = []

        def validate_path():
            try:
                validate_file_path(safe_file)
                results.append(True)
            except ValueError as e:
                errors.append(e)

        # Run multiple validations concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=validate_path)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All validations should succeed without race conditions
        assert len(results) == 10
        assert len(errors) == 0

    def test_path_with_null_bytes(self, temp_dir):
        """Test that paths with null bytes are rejected."""
        null_byte_paths = [
            temp_dir / "file\x00.txt",
            temp_dir / "safe\x00../../../etc/passwd",
            Path("test\x00file"),
        ]

        for path in null_byte_paths:
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_path(path, PROJECT_ROOT)

    def test_path_with_unicode_tricks(self, temp_dir):
        """Test that Unicode-based path tricks are rejected."""
        unicode_tricks = [
            temp_dir / "file\u202etxt",  # Unicode right-to-left override
            temp_dir / "file\u200b.txt",  # Zero-width space
            temp_dir / "file\ufeff.txt",  # BOM character
        ]

        for path in unicode_tricks:
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_path(path, PROJECT_ROOT)

    def test_file_permission_checks(self, temp_dir):
        """Test file permission validation."""
        # Create file with restrictive permissions
        restricted_file = temp_dir / "restricted.txt"
        restricted_file.write_text("restricted")
        restricted_file.chmod(0o000)  # No permissions

        # Should reject files with no read permissions
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_file_path(restricted_file, PROJECT_ROOT)

    def test_file_size_validation(self, temp_dir):
        """Test file size validation."""
        # Create oversized file
        oversized_file = temp_dir / "oversized.txt"
        with open(oversized_file, "wb") as f:
            f.write(b"x" * (1024 * 1024 * 1024 + 1))  # > 1GB

        # Should reject oversized files
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_file_path(oversized_file, PROJECT_ROOT)

    def test_safe_path_acceptance(self, safe_file):
        """Test that safe paths are accepted."""
        # Valid file should pass validation
        validate_file_path(safe_file, PROJECT_ROOT)

        # Valid directory should pass validation
        validate_file_path(safe_file.parent)

    def test_path_normalization(self, temp_dir):
        """Test path normalization attacks."""
        # Test various normalization attacks
        normalization_attacks = [
            temp_dir / "test/../etc/passwd",  # Directory traversal
            temp_dir / "test/./../../etc/passwd",  # Current directory
            temp_dir / "test//etc/passwd",  # Double slash
        ]

        for attack_path in normalization_attacks:
            with pytest.raises(ValueError, match="Path traversal detected"):
                validate_file_path(attack_path, temp_dir)


class TestPathValidationEdgeCases:
    """Test edge cases for path validation."""

    def test_empty_path(self):
        """Test empty path rejection."""
        with pytest.raises(ValueError):
            validate_file_path(Path(""))

    def test_none_path(self):
        """Test None path rejection."""
        with pytest.raises(TypeError):
            validate_file_path(None)

    def test_path_with_spaces(self, temp_dir):
        """Test paths with spaces are handled correctly."""
        spaced_file = temp_dir / "file with spaces.txt"
        spaced_file.write_text("content")

        # Should accept valid paths with spaces
        validate_file_path(spaced_file)

    def test_very_long_path(self, temp_dir):
        """Test very long path handling."""
        # Create path that exceeds typical limits
        long_name = "a" * 255
        long_path = temp_dir / long_name

        # Should handle long paths appropriately
        with pytest.raises(ValueError, match="Path traversal detected"):
            validate_file_path(long_path)
