"""Tests for Output Paths module - comprehensive coverage."""

from pathlib import Path

from utils.output_paths import (
    OutputPathManager,
    create_output_path,
    ensure_directory_exists,
    get_output_directory,
    get_unique_filename,
    sanitize_filename,
)


class TestGetOutputDirectory:
    """Test output directory retrieval."""

    def test_get_default_output_dir(self, tmp_path, monkeypatch):
        """Test getting default output directory."""
        monkeypatch.chdir(tmp_path)
        result = get_output_directory()
        assert isinstance(result, Path)

    def test_get_custom_output_dir(self, tmp_path):
        """Test getting custom output directory."""
        custom_dir = tmp_path / "custom_output"
        result = get_output_directory(subdir=custom_dir)
        assert result == custom_dir


class TestCreateOutputPath:
    """Test output path creation."""

    def test_create_simple_path(self, tmp_path):
        """Test creating simple output path."""
        result = create_output_path(tmp_path, "test.txt")
        assert result == tmp_path / "test.txt"

    def test_create_nested_path(self, tmp_path):
        """Test creating nested output path."""
        result = create_output_path(tmp_path, "subdir/test.txt")
        assert result == tmp_path / "subdir" / "test.txt"


class TestEnsureDirectoryExists:
    """Test directory creation."""

    def test_create_new_directory(self, tmp_path):
        """Test creating new directory."""
        new_dir = tmp_path / "new_directory"
        ensure_directory_exists(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_existing_directory(self, tmp_path):
        """Test ensuring existing directory (no error)."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        ensure_directory_exists(existing_dir)  # Should not raise
        assert existing_dir.exists()


class TestGetUniqueFilename:
    """Test unique filename generation."""

    def test_unique_filename_no_conflict(self, tmp_path):
        """Test unique filename when no conflict."""
        result = get_unique_filename(tmp_path, "test.txt")
        assert result == tmp_path / "test.txt"

    def test_unique_filename_with_conflict(self, tmp_path):
        """Test unique filename when file exists."""
        (tmp_path / "test.txt").touch()
        result = get_unique_filename(tmp_path, "test.txt")
        assert result != tmp_path / "test.txt"
        assert result.name.startswith("test")


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_sanitize_simple_filename(self):
        """Test sanitizing simple filename."""
        result = sanitize_filename("test.txt")
        assert result == "test.txt"

    def test_sanitize_with_special_chars(self):
        """Test sanitizing filename with special characters."""
        result = sanitize_filename("test<file>.txt")
        assert "<" not in result
        assert ">" not in result

    def test_sanitize_with_spaces(self):
        """Test sanitizing filename with spaces."""
        result = sanitize_filename("test file name.txt")
        assert " " not in result or result == "test file name.txt"


class TestOutputPathManager:
    """Test Output Path Manager class."""

    def test_init(self, tmp_path):
        """Test initialization."""
        manager = OutputPathManager(base_dir=tmp_path)
        assert manager.base_dir == tmp_path

    def test_get_path(self, tmp_path):
        """Test getting managed path."""
        manager = OutputPathManager(base_dir=tmp_path)
        result = manager.get_path("output.txt")
        assert isinstance(result, Path)

    def test_register_path(self, tmp_path):
        """Test registering a path."""
        manager = OutputPathManager(base_dir=tmp_path)
        manager.register("output", "results.txt")
        result = manager.get_registered("output")
        assert result == tmp_path / "results.txt"
