"""
Comprehensive Tests for Log Analysis Tools Module
==================================================

Target: 100% coverage for utils/log_analysis_tools.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.log_analysis_tools import (
    AnomalyDetector,
    IntegrityVerifier,
    LogAggregator,
    LogAnalyzer,
    LogEntry,
    LogLevel,
)


class TestLogLevel:
    """Test LogLevel enum"""

    def test_log_level_values(self):
        """Test log level enum values"""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestLogEntry:
    """Test LogEntry dataclass"""

    def test_log_entry_creation(self):
        """Test log entry creation"""
        entry = LogEntry(
            timestamp="2024-01-01 12:00:00",
            level="INFO",
            message="Test message",
            module="test_module",
            function="test_func",
            line=42,
            file_path="/test/file.py",
            hash_value="abc123",
            raw_data="raw log line",
        )
        assert entry.timestamp == "2024-01-01 12:00:00"
        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert entry.module == "test_module"

    def test_log_entry_defaults(self):
        """Test log entry default values"""
        entry = LogEntry(
            timestamp="2024-01-01 12:00:00",
            level="INFO",
            message="Test",
            module="test",
            function=None,
            line=None,
            file_path=None,
            hash_value=None,
            raw_data=None,
        )
        assert entry.chain_hash is None
        assert entry.entry_number is None
        assert entry.anomalies == []


class TestLogAnalyzer:
    """Test LogAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        return LogAnalyzer()

    @pytest.fixture
    def sample_log_file(self, tmp_path):
        """Create sample log file"""
        log_file = tmp_path / "test.log"
        log_content = """2024-01-01 12:00:00,123 - INFO - test_module - Test message 1
2024-01-01 12:00:01,456 - WARNING - test_module - Warning message
2024-01-01 12:00:02,789 - ERROR - test_module - Error message
2024-01-01 12:00:03,012 - INFO - test_module - Test message 2
"""
        log_file.write_text(log_content)
        return log_file

    def test_analyzer_creation(self, analyzer):
        """Test analyzer creation"""
        assert isinstance(analyzer.entries, list)
        assert isinstance(analyzer.anomalies, list)

    def test_parse_log_file(self, analyzer, sample_log_file):
        """Test log file parsing"""
        entries = analyzer.parse_log_file(str(sample_log_file))
        assert len(entries) == 4
        assert entries[0].level == "INFO"
        assert entries[1].level == "WARNING"

    def test_parse_nonexistent_file(self, analyzer):
        """Test parsing non-existent file"""
        entries = analyzer.parse_log_file("/nonexistent/file.log")
        assert entries == []

    def test_detect_anomalies(self, analyzer):
        """Test anomaly detection"""
        # Add entries with high frequency
        for i in range(100):
            analyzer.entries.append(
                LogEntry(
                    timestamp=f"2024-01-01 12:00:{i:02d}",
                    level="ERROR",
                    message=f"Error {i}",
                    module="test",
                    function=None,
                    line=None,
                    file_path=None,
                    hash_value=None,
                    raw_data=None,
                )
            )

        anomalies = analyzer.detect_anomalies()
        assert len(anomalies) > 0

    def test_analyze_levels(self, analyzer, sample_log_file):
        """Test level analysis"""
        analyzer.parse_log_file(str(sample_log_file))
        level_stats = analyzer.analyze_levels()

        assert "INFO" in level_stats
        assert "WARNING" in level_stats
        assert "ERROR" in level_stats

    def test_filter_by_level(self, analyzer, sample_log_file):
        """Test filtering by level"""
        analyzer.parse_log_file(str(sample_log_file))
        errors = analyzer.filter_by_level("ERROR")

        assert len(errors) == 1
        assert errors[0].level == "ERROR"

    def test_filter_by_time_range(self, analyzer, sample_log_file):
        """Test filtering by time range"""
        analyzer.parse_log_file(str(sample_log_file))
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 0, 2, 800000)  # Include up to 02.8s

        filtered = analyzer.filter_by_time_range(start_time, end_time)
        assert len(filtered) == 3

    def test_search_pattern(self, analyzer, sample_log_file):
        """Test pattern search"""
        analyzer.parse_log_file(str(sample_log_file))
        results = analyzer.search_pattern("Error")

        assert len(results) == 1
        assert "Error" in results[0].message


class TestIntegrityVerifier:
    """Test IntegrityVerifier class"""

    @pytest.fixture
    def verifier(self):
        return IntegrityVerifier()

    @pytest.fixture
    def sample_log_file(self, tmp_path):
        """Create sample log file"""
        log_file = tmp_path / "test.log"
        log_content = """2024-01-01 12:00:00,123 - INFO - test_module - Test message 1
2024-01-01 12:00:01,456 - WARNING - test_module - Warning message
"""
        log_file.write_text(log_content)
        return log_file

    def test_verifier_creation(self, verifier):
        """Test verifier creation"""
        assert isinstance(verifier, IntegrityVerifier)

    def test_compute_file_hash(self, verifier, sample_log_file):
        """Test file hash computation"""
        hash_value = verifier.compute_file_hash(str(sample_log_file))
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA-256 hash length

    def test_verify_file_integrity(self, verifier, sample_log_file):
        """Test file integrity verification"""
        expected_hash = verifier.compute_file_hash(str(sample_log_file))
        result = verifier.verify_file_integrity(str(sample_log_file), expected_hash)

        assert result is True

    def test_verify_corrupted_file(self, verifier, sample_log_file):
        """Test verification with corrupted file"""
        wrong_hash = "0" * 64
        result = verifier.verify_file_integrity(str(sample_log_file), wrong_hash)

        assert result is False


class TestLogAggregator:
    """Test LogAggregator class"""

    @pytest.fixture
    def aggregator(self):
        return LogAggregator()

    def test_aggregator_creation(self, aggregator):
        """Test aggregator creation"""
        assert isinstance(aggregator, LogAggregator)

    def test_add_log_source(self, aggregator, tmp_path):
        """Test adding log source"""
        log_file = tmp_path / "test.log"
        log_file.write_text("Test log content")

        aggregator.add_log_source("test_source", str(log_file))
        assert "test_source" in aggregator.sources


class TestAnomalyDetector:
    """Test AnomalyDetector class"""

    @pytest.fixture
    def detector(self):
        return AnomalyDetector()

    def test_detector_creation(self, detector):
        """Test detector creation"""
        assert isinstance(detector, AnomalyDetector)

    def test_detect_burst_errors(self, detector):
        """Test burst error detection"""
        entries = [
            LogEntry(
                timestamp=f"2024-01-01 12:00:{i:02d}",
                level="ERROR",
                message=f"Error {i}",
                module="test",
                function=None,
                line=None,
                file_path=None,
                hash_value=None,
                raw_data=None,
            )
            for i in range(20)
        ]

        anomalies = detector.detect_burst_errors(entries, threshold=5)
        assert len(anomalies) > 0


class TestEdgeCases:
    """Test edge cases"""

    def test_empty_log_file(self, tmp_path):
        """Test parsing empty log file"""
        log_file = tmp_path / "empty.log"
        log_file.write_text("")

        analyzer = LogAnalyzer()
        entries = analyzer.parse_log_file(str(log_file))
        assert entries == []

    def test_malformed_log_entries(self, tmp_path):
        """Test parsing malformed log entries"""
        log_file = tmp_path / "malformed.log"
        log_content = """Not a valid log line
Another invalid line
"""
        log_file.write_text(log_content)

        analyzer = LogAnalyzer()
        entries = analyzer.parse_log_file(str(log_file))
        # Should handle gracefully, possibly skip or mark as unknown
        assert isinstance(entries, list)

    def test_large_log_file(self, tmp_path):
        """Test parsing large log file"""
        log_file = tmp_path / "large.log"

        # Create a large log file
        with open(log_file, "w") as f:
            for i in range(1000):
                f.write(f"2024-01-01 12:00:{i % 60:02d} - INFO - test - Message {i}\n")

        analyzer = LogAnalyzer()
        entries = analyzer.parse_log_file(str(log_file))
        assert len(entries) == 1000
