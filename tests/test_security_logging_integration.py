"""
Tests for utils/security_logging_integration.py
===============================================
Comprehensive tests for security logging integration utilities.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.security_logging_integration import (secure_file_delete,
                                                secure_file_read,
                                                secure_file_write,
                                                secure_import_module,
                                                secure_resolve_path)


class TestSecureFileRead:
    """Test secure_file_read function."""

    def test_secure_file_read_success(self, tmp_path):
        """Test successful secure file read."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        content = secure_file_read(str(test_file))

        assert content == "test content"

    def test_secure_file_read_failure(self, tmp_path):
        """Test secure file read with non-existent file."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            secure_file_read(str(test_file))


class TestSecureFileWrite:
    """Test secure_file_write function."""

    def test_secure_file_write_success(self, tmp_path):
        """Test successful secure file write."""
        test_file = tmp_path / "test.txt"

        secure_file_write(str(test_file), "test content")

        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_secure_file_write_failure(self, tmp_path):
        """Test secure file write to invalid path."""
        invalid_path = "/root/test.txt"  # Permission denied

        with pytest.raises((PermissionError, OSError)):
            secure_file_write(invalid_path, "test content")


class TestSecureFileDelete:
    """Test secure_file_delete function."""

    def test_secure_file_delete_success(self, tmp_path):
        """Test successful secure file delete."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        secure_file_delete(str(test_file))

        assert not test_file.exists()

    def test_secure_file_delete_failure(self, tmp_path):
        """Test secure file delete with non-existent file."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            secure_file_delete(str(test_file))


class TestSecureImportModule:
    """Test secure_import_module function."""

    def test_secure_import_module_success(self, tmp_path):
        """Test successful secure module import."""
        module_file = tmp_path / "test_module.py"
        module_file.write_text("test_value = 42\n")

        module = secure_import_module("test_module", str(module_file))

        assert hasattr(module, "test_value")
        assert module.test_value == 42

    def test_secure_import_module_failure(self, tmp_path):
        """Test secure module import with invalid file."""
        module_file = tmp_path / "invalid.py"
        module_file.write_text("invalid syntax here\n")

        with pytest.raises(SyntaxError):
            secure_import_module("invalid", str(module_file))


class TestSecureResolvePath:
    """Test secure_resolve_path function."""

    def test_secure_resolve_path_success(self, tmp_path):
        """Test successful secure path resolution."""
        relative_path = "test.txt"

        resolved = secure_resolve_path(relative_path)

        assert Path(resolved).is_absolute()

    def test_secure_resolve_path_failure(self, tmp_path):
        """Test secure path resolution with invalid path."""
        # This should still resolve, just might not exist
        resolved = secure_resolve_path("nonexistent/path")

        assert Path(resolved).is_absolute()
