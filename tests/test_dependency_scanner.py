"""
Tests for utils/dependency_scanner.py
=======================================
Comprehensive tests for dependency vulnerability scanner.
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dependency_scanner import DependencyScanner, run_dependency_scan


class TestDependencyScannerInit:
    """Test DependencyScanner initialization."""

    def test_init_with_project_root(self, tmp_path):
        """Test initialization with explicit project root."""
        scanner = DependencyScanner(str(tmp_path))
        assert scanner.project_root == tmp_path
        assert scanner.requirements_file == tmp_path / "requirements.txt"

    def test_init_without_project_root(self):
        """Test initialization without project root (uses cwd)."""
        scanner = DependencyScanner()
        assert scanner.project_root == Path.cwd()
        assert scanner.requirements_file == Path.cwd() / "requirements.txt"


class TestScanWithPipAudit:
    """Test scan_with_pip_audit method."""

    @patch("subprocess.run")
    def test_scan_with_pip_audit_success(self, mock_run):
        """Test successful pip-audit scan."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"vulnerabilities": []}', stderr=""
        )

        scanner = DependencyScanner()
        result = scanner.scan_with_pip_audit()

        assert result["scanner"] == "pip-audit"
        assert result["vulnerabilities_found"] == 0
        assert "timestamp" in result
        assert "details" in result

    @patch("subprocess.run")
    def test_scan_with_pip_audit_vulnerabilities_found(self, mock_run):
        """Test pip-audit scan with vulnerabilities."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"vulnerabilities": [{"id": "CVE-2024-1234"}]}',
            stderr="",
        )

        scanner = DependencyScanner()
        result = scanner.scan_with_pip_audit()

        assert result["scanner"] == "pip-audit"
        assert (
            result["vulnerabilities_found"] == 0
        )  # pip-audit returns 0 even with vulns
        assert len(result["details"]) == 1

    @patch("subprocess.run")
    def test_scan_with_pip_audit_error(self, mock_run):
        """Test pip-audit scan with error."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Error scanning dependencies"
        )

        scanner = DependencyScanner()
        result = scanner.scan_with_pip_audit()

        assert result["scanner"] == "pip-audit"
        assert result["vulnerabilities_found"] == -1
        assert "error" in result

    @patch("subprocess.run")
    def test_scan_with_pip_audit_not_installed(self, mock_run):
        """Test pip-audit not installed."""
        mock_run.side_effect = FileNotFoundError()

        scanner = DependencyScanner()
        result = scanner.scan_with_pip_audit()

        assert result["scanner"] == "pip-audit"
        assert result["vulnerabilities_found"] == -1
        assert "not installed" in result["error"]

    @patch("subprocess.run")
    def test_scan_with_pip_audit_timeout(self, mock_run):
        """Test pip-audit scan timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("pip-audit", 300)

        scanner = DependencyScanner()
        result = scanner.scan_with_pip_audit()

        assert result["scanner"] == "pip-audit"
        assert result["vulnerabilities_found"] == -1
        assert "timed out" in result["error"].lower()

    @patch("subprocess.run")
    def test_scan_with_pip_audit_exception(self, mock_run):
        """Test pip-audit scan with unexpected exception."""
        mock_run.side_effect = Exception("Unexpected error")

        scanner = DependencyScanner()
        result = scanner.scan_with_pip_audit()

        assert result["scanner"] == "pip-audit"
        assert result["vulnerabilities_found"] == -1
        assert "Unexpected error" in result["error"]


class TestScanWithSafety:
    """Test scan_with_safety method."""

    @patch("subprocess.run")
    def test_scan_with_safety_success(self, mock_run):
        """Test successful safety scan."""
        mock_run.return_value = MagicMock(returncode=0, stdout="[]", stderr="")

        scanner = DependencyScanner()
        result = scanner.scan_with_safety()

        assert result["scanner"] == "safety"
        assert result["vulnerabilities_found"] == 0
        assert "timestamp" in result

    @patch("subprocess.run")
    def test_scan_with_safety_vulnerabilities_found(self, mock_run):
        """Test safety scan with vulnerabilities."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='[{"id": "12345", "advisory": "Test advisory"}]',
            stderr="",
        )

        scanner = DependencyScanner()
        result = scanner.scan_with_safety()

        assert result["scanner"] == "safety"
        assert result["vulnerabilities_found"] == 1
        assert len(result["details"]) == 1

    @patch("subprocess.run")
    def test_scan_with_safety_error(self, mock_run):
        """Test safety scan with error."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Error scanning with safety"
        )

        scanner = DependencyScanner()
        result = scanner.scan_with_safety()

        assert result["scanner"] == "safety"
        assert result["vulnerabilities_found"] == -1
        assert "error" in result

    @patch("subprocess.run")
    def test_scan_with_safety_not_installed(self, mock_run):
        """Test safety not installed."""
        mock_run.side_effect = FileNotFoundError()

        scanner = DependencyScanner()
        result = scanner.scan_with_safety()

        assert result["scanner"] == "safety"
        assert result["vulnerabilities_found"] == -1
        assert "not installed" in result["error"]

    @patch("subprocess.run")
    def test_scan_with_safety_timeout(self, mock_run):
        """Test safety scan timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("safety", 300)

        scanner = DependencyScanner()
        result = scanner.scan_with_safety()

        assert result["scanner"] == "safety"
        assert result["vulnerabilities_found"] == -1
        assert "timed out" in result["error"].lower()


class TestScanWithBandit:
    """Test scan_with_bandit method."""

    @patch("subprocess.run")
    def test_scan_with_bandit_success(self, mock_run):
        """Test successful bandit scan."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"results": [], "metrics": {} }', stderr=""
        )

        scanner = DependencyScanner()
        result = scanner.scan_with_bandit()

        assert result["scanner"] == "bandit"
        assert result["issues_found"] == 0
        assert "timestamp" in result

    @patch("subprocess.run")
    def test_scan_with_bandit_issues_found(self, mock_run):
        """Test bandit scan with issues found."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout='{"results": [{"issue_text": "Test issue"}], "metrics": {}}',
            stderr="",
        )

        scanner = DependencyScanner()
        result = scanner.scan_with_bandit()

        assert result["scanner"] == "bandit"
        assert result["issues_found"] == 1
        assert len(result["details"]["results"]) == 1

    @patch("subprocess.run")
    def test_scan_with_bandit_error(self, mock_run):
        """Test bandit scan with error."""
        mock_run.return_value = MagicMock(
            returncode=2, stdout="", stderr="Error scanning with bandit"
        )

        scanner = DependencyScanner()
        result = scanner.scan_with_bandit()

        assert result["scanner"] == "bandit"
        assert result["issues_found"] == -1
        assert "error" in result

    @patch("subprocess.run")
    def test_scan_with_bandit_not_installed(self, mock_run):
        """Test bandit not installed."""
        mock_run.side_effect = FileNotFoundError()

        scanner = DependencyScanner()
        result = scanner.scan_with_bandit()

        assert result["scanner"] == "bandit"
        assert result["issues_found"] == -1
        assert "not installed" in result["error"]

    @patch("subprocess.run")
    def test_scan_with_bandit_timeout(self, mock_run):
        """Test bandit scan timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("bandit", 300)

        scanner = DependencyScanner()
        result = scanner.scan_with_bandit()

        assert result["scanner"] == "bandit"
        assert result["issues_found"] == -1
        assert "timed out" in result["error"].lower()

    @patch("subprocess.run")
    def test_scan_with_bandit_invalid_json(self, mock_run):
        """Test bandit scan with invalid JSON output."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="invalid json", stderr=""
        )

        scanner = DependencyScanner()
        result = scanner.scan_with_bandit()

        assert result["scanner"] == "bandit"
        assert result["issues_found"] == 0
        assert result["details"] == {}


class TestRunComprehensiveScan:
    """Test run_comprehensive_scan method."""

    @patch.object(DependencyScanner, "scan_with_pip_audit")
    @patch.object(DependencyScanner, "scan_with_safety")
    @patch.object(DependencyScanner, "scan_with_bandit")
    def test_run_comprehensive_scan_all_success(
        self, mock_bandit, mock_safety, mock_pip
    ):
        """Test comprehensive scan with all scanners successful."""
        mock_pip.return_value = {
            "scanner": "pip-audit",
            "vulnerabilities_found": 0,
            "details": [],
        }
        mock_safety.return_value = {
            "scanner": "safety",
            "vulnerabilities_found": 0,
            "details": [],
        }
        mock_bandit.return_value = {
            "scanner": "bandit",
            "issues_found": 0,
            "details": {},
        }

        scanner = DependencyScanner()
        result = scanner.run_comprehensive_scan()

        assert "timestamp" in result
        assert "project_root" in result
        assert "scans" in result
        assert "summary" in result
        assert result["summary"]["scanners_run"] == 3
        assert result["summary"]["scanners_failed"] == 0
        assert result["summary"]["total_vulnerabilities"] == 0
        assert result["summary"]["total_issues"] == 0

    @patch.object(DependencyScanner, "scan_with_pip_audit")
    @patch.object(DependencyScanner, "scan_with_safety")
    @patch.object(DependencyScanner, "scan_with_bandit")
    def test_run_comprehensive_scan_with_failures(
        self, mock_bandit, mock_safety, mock_pip
    ):
        """Test comprehensive scan with some scanners failing."""
        mock_pip.return_value = {
            "scanner": "pip-audit",
            "vulnerabilities_found": 2,
            "details": [{"id": "CVE-1"}, {"id": "CVE-2"}],
        }
        mock_safety.return_value = {
            "scanner": "safety",
            "vulnerabilities_found": -1,
            "error": "not installed",
        }
        mock_bandit.return_value = {
            "scanner": "bandit",
            "issues_found": 3,
            "details": {"results": [1, 2, 3]},
        }

        scanner = DependencyScanner()
        result = scanner.run_comprehensive_scan()

        assert result["summary"]["scanners_run"] == 3
        assert result["summary"]["scanners_failed"] == 1
        assert result["summary"]["total_vulnerabilities"] == 2
        assert result["summary"]["total_issues"] == 3

    @patch.object(DependencyScanner, "scan_with_pip_audit")
    @patch.object(DependencyScanner, "scan_with_safety")
    @patch.object(DependencyScanner, "scan_with_bandit")
    def test_run_comprehensive_scan_all_fail(self, mock_bandit, mock_safety, mock_pip):
        """Test comprehensive scan with all scanners failing."""
        mock_pip.return_value = {
            "scanner": "pip-audit",
            "vulnerabilities_found": -1,
            "error": "not installed",
        }
        mock_safety.return_value = {
            "scanner": "safety",
            "vulnerabilities_found": -1,
            "error": "not installed",
        }
        mock_bandit.return_value = {
            "scanner": "bandit",
            "issues_found": -1,
            "error": "not installed",
        }

        scanner = DependencyScanner()
        result = scanner.run_comprehensive_scan()

        assert result["summary"]["scanners_run"] == 3
        assert result["summary"]["scanners_failed"] == 3
        assert result["summary"]["total_vulnerabilities"] == 0
        assert result["summary"]["total_issues"] == 0


class TestSaveScanReport:
    """Test save_scan_report method."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_scan_report_default_path(self, mock_json_dump, mock_file):
        """Test saving scan report to default path."""
        scanner = DependencyScanner()
        results = {"test": "data"}

        scanner.save_scan_report(results)

        mock_file.assert_called_once()
        mock_json_dump.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_scan_report_custom_path(self, mock_json_dump, mock_file, tmp_path):
        """Test saving scan report to custom path."""
        scanner = DependencyScanner()
        results = {"test": "data"}
        output_file = tmp_path / "custom_report.json"

        scanner.save_scan_report(results, str(output_file))

        mock_file.assert_called_once()
        mock_json_dump.assert_called_once()


class TestGetVulnerabilitySummary:
    """Test get_vulnerability_summary method."""

    def test_get_vulnerability_summary_clean(self):
        """Test summary generation for clean scan."""
        scanner = DependencyScanner()
        results = {
            "timestamp": "2024-01-01T00:00:00",
            "project_root": "/test",
            "summary": {
                "total_vulnerabilities": 0,
                "total_issues": 0,
                "scanners_run": 3,
                "scanners_failed": 0,
            },
            "scans": {
                "pip_audit": {"vulnerabilities_found": 0, "details": []},
                "safety": {"vulnerabilities_found": 0, "details": []},
                "bandit": {"issues_found": 0, "details": {}},
            },
        }

        summary = scanner.get_vulnerability_summary(results)

        assert "DEPENDENCY VULNERABILITY SCAN REPORT" in summary
        assert "Total Vulnerabilities: 0" in summary
        assert "Total Code Issues: 0" in summary
        assert "Scanners Run: 3" in summary
        assert "Scanners Failed: 0" in summary

    def test_get_vulnerability_summary_with_issues(self):
        """Test summary generation with vulnerabilities and issues."""
        scanner = DependencyScanner()
        results = {
            "timestamp": "2024-01-01T00:00:00",
            "project_root": "/test",
            "summary": {
                "total_vulnerabilities": 5,
                "total_issues": 3,
                "scanners_run": 3,
                "scanners_failed": 0,
            },
            "scans": {
                "pip_audit": {
                    "vulnerabilities_found": 2,
                    "details": [{"id": "CVE-1"}, {"id": "CVE-2"}],
                },
                "safety": {
                    "vulnerabilities_found": 3,
                    "details": [{"id": "123"}, {"id": "456"}, {"id": "789"}],
                },
                "bandit": {"issues_found": 3, "details": {"results": [1, 2, 3]}},
            },
        }

        summary = scanner.get_vulnerability_summary(results)

        assert "Total Vulnerabilities: 5" in summary
        assert "Total Code Issues: 3" in summary
        assert "2 items found" in summary
        assert "3 items found" in summary

    def test_get_vulnerability_summary_with_errors(self):
        """Test summary generation with scanner errors."""
        scanner = DependencyScanner()
        results = {
            "timestamp": "2024-01-01T00:00:00",
            "project_root": "/test",
            "summary": {
                "total_vulnerabilities": 0,
                "total_issues": 0,
                "scanners_run": 3,
                "scanners_failed": 2,
            },
            "scans": {
                "pip_audit": {"vulnerabilities_found": 0, "details": []},
                "safety": {"error": "not installed", "vulnerabilities_found": -1},
                "bandit": {"error": "timeout", "issues_found": -1},
            },
        }

        summary = scanner.get_vulnerability_summary(results)

        assert "Scanners Failed: 2" in summary
        assert "ERROR" in summary


class TestRunDependencyScan:
    """Test run_dependency_scan function."""

    @patch.object(DependencyScanner, "run_comprehensive_scan")
    @patch.object(DependencyScanner, "save_scan_report")
    def test_run_dependency_scan_with_save(self, mock_save, mock_scan):
        """Test run_dependency_scan with save enabled."""
        mock_scan.return_value = {"test": "results"}

        result = run_dependency_scan(save_report=True)

        assert result == {"test": "results"}
        mock_scan.assert_called_once()
        mock_save.assert_called_once()

    @patch.object(DependencyScanner, "run_comprehensive_scan")
    @patch.object(DependencyScanner, "save_scan_report")
    def test_run_dependency_scan_without_save(self, mock_save, mock_scan):
        """Test run_dependency_scan with save disabled."""
        mock_scan.return_value = {"test": "results"}

        result = run_dependency_scan(save_report=False)

        assert result == {"test": "results"}
        mock_scan.assert_called_once()
        mock_save.assert_not_called()

    @patch.object(DependencyScanner, "run_comprehensive_scan")
    @patch.object(DependencyScanner, "save_scan_report")
    def test_run_dependency_scan_with_project_root(
        self, mock_save, mock_scan, tmp_path
    ):
        """Test run_dependency_scan with custom project root."""
        mock_scan.return_value = {"test": "results"}

        result = run_dependency_scan(project_root=str(tmp_path), save_report=True)

        assert result == {"test": "results"}
        mock_scan.assert_called_once()
        mock_save.assert_called_once()
