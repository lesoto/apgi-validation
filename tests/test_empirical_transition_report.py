"""Tests for Empirical Transition Report module - comprehensive coverage."""

from utils.empirical_transition_report import (
    compare_reports,
    format_report_as_markdown,
    generate_transition_report,
    load_report,
    save_report,
)


class TestGenerateTransitionReport:
    """Test transition report generation."""

    def test_generate_basic_report(self):
        """Test generating a basic transition report."""
        data = {
            "transitions": [
                {"from": "state_a", "to": "state_b", "count": 10},
                {"from": "state_b", "to": "state_c", "count": 5},
            ]
        }
        report = generate_transition_report(data)
        assert isinstance(report, dict)
        assert "transitions" in report

    def test_generate_empty_report(self):
        """Test generating report with empty data."""
        data = {"transitions": []}
        report = generate_transition_report(data)
        assert isinstance(report, dict)
        assert report["transitions"] == []


class TestFormatReportAsMarkdown:
    """Test Markdown formatting of reports."""

    def test_format_basic_report(self):
        """Test formatting a basic report."""
        report = {
            "title": "Test Report",
            "transitions": [{"from": "A", "to": "B", "probability": 0.8}],
        }
        markdown = format_report_as_markdown(report)
        assert isinstance(markdown, str)
        assert "Test Report" in markdown

    def test_format_empty_report(self):
        """Test formatting empty report."""
        report = {}
        markdown = format_report_as_markdown(report)
        assert isinstance(markdown, str)


class TestSaveAndLoadReport:
    """Test report persistence."""

    def test_save_and_load_json(self, tmp_path):
        """Test saving and loading JSON report."""
        report = {"title": "Test", "data": [1, 2, 3]}
        filepath = tmp_path / "report.json"

        save_report(report, filepath, format="json")
        assert filepath.exists()

        loaded = load_report(filepath)
        assert loaded["title"] == "Test"
        assert loaded["data"] == [1, 2, 3]


class TestCompareReports:
    """Test report comparison functionality."""

    def test_compare_identical_reports(self):
        """Test comparing identical reports."""
        report1 = {"transitions": [{"from": "A", "to": "B"}]}
        report2 = {"transitions": [{"from": "A", "to": "B"}]}

        comparison = compare_reports(report1, report2)
        assert isinstance(comparison, dict)

    def test_compare_different_reports(self):
        """Test comparing different reports."""
        report1 = {"transitions": [{"from": "A", "to": "B"}]}
        report2 = {"transitions": [{"from": "A", "to": "C"}]}

        comparison = compare_reports(report1, report2)
        assert isinstance(comparison, dict)
