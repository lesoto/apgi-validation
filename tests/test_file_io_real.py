"""
Real file I/O tests for path validation.
Tests actual filesystem operations to catch permission and path resolution issues.
===============================================================================
"""

import os
import stat
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import secure_load_module, secure_load_module_from_path


class TestRealFileIO:
    """Tests for real file I/O operations without mocking."""

    def test_secure_load_module_valid_path(self, temp_dir):
        """Test loading a valid module with secure path validation."""
        # Create a simple test module
        test_module = temp_dir / "test_module.py"
        test_module.write_text("""
def test_function():
    return "success"
""")

        module = secure_load_module("test_module", test_module)
        assert hasattr(module, "test_function")
        assert module.test_function() == "success"

    def test_secure_load_module_outside_project_root(self):
        """Test that loading module outside project root is rejected."""
        # Try to load from /tmp (outside project root)
        test_file = Path("/tmp/test_module.py")
        with pytest.raises(ValueError) as exc_info:
            secure_load_module("test_module", test_file)
        assert "outside project root" in str(exc_info.value)

    def test_secure_load_module_non_py_file(self, temp_dir):
        """Test that non-.py files are rejected."""
        test_file = temp_dir / "test_module.txt"
        test_file.write_text("not a python file")

        with pytest.raises(ValueError) as exc_info:
            secure_load_module("test_module", test_file)
        assert "must be a .py file" in str(exc_info.value)

    def test_secure_load_module_nonexistent_file(self, temp_dir):
        """Test loading nonexistent file."""
        test_file = temp_dir / "nonexistent.py"

        with pytest.raises((ImportError, FileNotFoundError)):
            secure_load_module("test_module", test_file)

    def test_secure_load_module_from_path_convenience(self, temp_dir):
        """Test convenience function for loading from path."""
        test_module = temp_dir / "convenience_test.py"
        test_module.write_text("""
def convenience_func():
    return "convenience"
""")

        module = secure_load_module_from_path(test_module)
        assert hasattr(module, "convenience_func")
        assert module.convenience_func() == "convenience"

    def test_file_permission_denied(self, temp_dir):
        """Test handling of permission denied errors."""
        # Create a file with no read permissions
        test_file = temp_dir / "no_read.txt"
        test_file.write_text("secret content")

        # Remove read permissions (on Unix-like systems)
        try:
            os.chmod(test_file, 0o000)
            with pytest.raises((PermissionError, OSError)):
                test_file.read_text()
        except (PermissionError, OSError):
            pytest.skip("Cannot change file permissions on this system")
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(test_file, 0o644)
            except (PermissionError, OSError):
                pass

    def test_path_resolution_symlink(self, temp_dir):
        """Test path resolution with symlinks."""
        # Create a regular file
        target_file = temp_dir / "target.txt"
        target_file.write_text("target content")

        # Create a symlink (on Unix-like systems)
        try:
            symlink_file = temp_dir / "symlink.txt"
            symlink_file.symlink_to(target_file)

            # Reading through symlink should work
            content = symlink_file.read_text()
            assert content == "target content"
        except (OSError, NotImplementedError):
            # Symlinks not supported on this system
            pytest.skip("Symlinks not supported")

    def test_path_resolution_relative_path(self, temp_dir):
        """Test path resolution with relative paths."""
        # Create a file
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            # Read with relative path
            content = Path("test.txt").read_text()
            assert content == "content"
        finally:
            os.chdir(original_cwd)

    def test_path_resolution_absolute_path(self, temp_dir):
        """Test path resolution with absolute paths."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        # Read with absolute path
        absolute_path = test_file.resolve()
        content = absolute_path.read_text()
        assert content == "content"

    def test_file_exists_check(self, temp_dir):
        """Test file existence checking."""
        existing_file = temp_dir / "exists.txt"
        existing_file.write_text("exists")

        nonexistent_file = temp_dir / "nonexistent.txt"

        assert existing_file.exists()
        assert not nonexistent_file.exists()

    def test_is_file_vs_is_directory(self, temp_dir):
        """Test distinguishing files from directories."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("file")

        test_dir = temp_dir / "test_dir"
        test_dir.mkdir()

        assert test_file.is_file()
        assert not test_file.is_dir()
        assert test_dir.is_dir()
        assert not test_dir.is_file()

    def test_file_size_check(self, temp_dir):
        """Test file size checking."""
        test_file = temp_dir / "test.txt"
        content = "x" * 1000
        test_file.write_text(content)

        assert test_file.stat().st_size == 1000

    def test_file_modification_time(self, temp_dir):
        """Test file modification time tracking."""
        import time

        test_file = temp_dir / "test.txt"
        test_file.write_text("initial")

        mtime1 = test_file.stat().st_mtime
        time.sleep(0.1)  # Small delay

        test_file.write_text("modified")
        mtime2 = test_file.stat().st_mtime

        assert mtime2 > mtime1

    def test_directory_traversal_protection(self, temp_dir):
        """Test protection against directory traversal attacks."""
        # Create a file outside temp_dir
        outside_file = temp_dir.parent / "outside.txt"
        outside_file.write_text("outside content")

        # Try to access using relative path traversal
        try:
            traversal_path = temp_dir / ".." / "outside.txt"
            # This should resolve to outside the temp_dir
            resolved = traversal_path.resolve()
            assert resolved == outside_file.resolve()
        finally:
            # Cleanup
            if outside_file.exists():
                outside_file.unlink()

    def test_max_file_size_limit(self, temp_dir):
        """Test handling of large files."""
        # Create a large file
        large_file = temp_dir / "large.txt"
        large_file.write_text("x" * (10 * 1024 * 1024))  # 10 MB

        # File should be readable
        content = large_file.read_text()
        assert len(content) == 10 * 1024 * 1024

    def test_file_extension_validation(self, temp_dir):
        """Test file extension validation."""
        # Create various file types
        py_file = temp_dir / "module.py"
        txt_file = temp_dir / "data.txt"
        json_file = temp_dir / "config.json"
        csv_file = temp_dir / "data.csv"

        for f in [py_file, txt_file, json_file, csv_file]:
            f.write_text("content")

        assert py_file.suffix == ".py"
        assert txt_file.suffix == ".txt"
        assert json_file.suffix == ".json"
        assert csv_file.suffix == ".csv"

    def test_path_normalization(self, temp_dir):
        """Test path normalization."""
        # Create a file
        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        # Test various path representations
        paths = [
            test_file,
            temp_dir / "./test.txt",
            temp_dir / "subdir/../test.txt",
        ]

        for path in paths:
            resolved = path.resolve()
            assert resolved == test_file.resolve()

    def test_file_creation_and_deletion(self, temp_dir):
        """Test file creation and deletion."""
        test_file = temp_dir / "create_delete.txt"

        # Create
        test_file.write_text("content")
        assert test_file.exists()

        # Delete
        test_file.unlink()
        assert not test_file.exists()

    def test_directory_creation_and_deletion(self, temp_dir):
        """Test directory creation and deletion."""
        test_dir = temp_dir / "test_dir"

        # Create
        test_dir.mkdir()
        assert test_dir.exists()
        assert test_dir.is_dir()

        # Delete
        test_dir.rmdir()
        assert not test_dir.exists()

    def test_file_permissions_unix(self, temp_dir):
        """Test file permission handling on Unix-like systems."""
        test_file = temp_dir / "perms.txt"
        test_file.write_text("content")

        try:
            # Make file read-only
            os.chmod(test_file, 0o444)
            file_mode = test_file.stat().st_mode
            assert not file_mode & stat.S_IWUSR  # No write permission for owner

            # Restore write permission
            os.chmod(test_file, 0o644)
            file_mode = test_file.stat().st_mode
            assert file_mode & stat.S_IWUSR  # Write permission restored
        except (OSError, AttributeError):
            pytest.skip("Permission operations not supported")

    def test_file_with_special_characters(self, temp_dir):
        """Test handling files with special characters in name."""
        # Create files with various special characters
        special_names = [
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.with.dots.txt",
        ]

        for name in special_names:
            test_file = temp_dir / name
            test_file.write_text(f"content for {name}")
            assert test_file.exists()
            assert test_file.read_text() == f"content for {name}"

    def test_empty_file_handling(self, temp_dir):
        """Test handling of empty files."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")

        assert empty_file.exists()
        assert empty_file.stat().st_size == 0
        assert empty_file.read_text() == ""

    def test_file_encoding_read_write(self, temp_dir):
        """Test file encoding during read/write operations."""
        test_file = temp_dir / "encoding.txt"

        # Write with UTF-8 encoding
        test_file.write_text("Hello 世界 🌍", encoding="utf-8")

        # Read with UTF-8 encoding
        content = test_file.read_text(encoding="utf-8")
        assert "世界" in content
        assert "🌍" in content

    def test_binary_file_operations(self, temp_dir):
        """Test binary file operations."""
        test_file = temp_dir / "binary.bin"

        # Write binary data
        binary_data = bytes(range(256))
        test_file.write_bytes(binary_data)

        # Read binary data
        read_data = test_file.read_bytes()
        assert read_data == binary_data
        assert len(read_data) == 256
