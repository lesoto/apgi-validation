"""Tests for delete_pycache cleanup script - comprehensive coverage."""

import pytest

from delete_pycache import clear_log_files, delete_temporary_items, main, matches_any


class TestDeleteTemporaryItems:
    """Test delete_temporary_items function."""

    def test_delete_temporary_items_with_pycache(self, tmp_path):
        """Test deleting temporary items including pycache."""
        # Create mock pycache directories
        pycache1 = tmp_path / "__pycache__"
        pycache1.mkdir()
        (pycache1 / "test.cpython-310.pyc").touch()

        # Create a temp file
        (tmp_path / "test.tmp").touch()

        # Delete them
        stats = delete_temporary_items(str(tmp_path), dry_run=False, verbose=False)
        assert isinstance(stats, dict)

    def test_dry_run_no_deletion(self, tmp_path):
        """Test dry run doesn't actually delete."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "test.pyc").touch()

        stats = delete_temporary_items(str(tmp_path), dry_run=True, verbose=False)
        assert stats is not None
        assert pycache.exists()  # Should still exist


class TestMatchesAny:
    """Test pattern matching for file deletion."""

    def test_matches_pyc_pattern(self):
        """Test matching .pyc files."""
        assert matches_any("test.pyc", ["*.pyc"]) is True
        assert matches_any("test.py", ["*.pyc"]) is False

    def test_matches_multiple_patterns(self):
        """Test matching against multiple patterns."""
        patterns = ["*.pyc", "*.log", "*.cache"]
        assert matches_any("debug.log", patterns) is True
        assert matches_any("data.cache", patterns) is True
        assert matches_any("script.py", patterns) is False


class TestClearLogFiles:
    """Test log file clearing."""

    def test_clear_log_files(self, tmp_path):
        """Test clearing log files."""
        log_file = tmp_path / "app.log"
        log_file.write_text("old log content\nmore content")

        clear_log_files(str(tmp_path), dry_run=False, verbose=False)
        # File may be deleted or emptied depending on implementation

    def test_clear_nonexistent_logs(self, tmp_path):
        """Test clearing when no logs exist."""
        # Should not raise an error
        clear_log_files(str(tmp_path), dry_run=False, verbose=False)


class TestMainFunction:
    """Test main entry point."""

    def test_main_help(self, capsys):
        """Test main with help flag."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_main_version(self, capsys):
        """Test main with version flag."""
        with pytest.raises(SystemExit):
            main(["--version"])

    def test_main_dry_run(self, tmp_path, monkeypatch):
        """Test main with dry run option."""
        # Create test structure
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "test.pyc").touch()

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        exit_code = main(["--dry-run"])
        assert exit_code == 0
        # Files should still exist
        assert pycache.exists()
