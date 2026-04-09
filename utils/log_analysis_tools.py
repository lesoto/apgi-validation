"""
Automated log analysis and integrity verification tools for APGI Framework.

This module provides comprehensive log analysis capabilities including:
- Log parsing and aggregation
- Anomaly detection and alerting
- Integrity verification through cryptographic hashes
- Performance metric extraction
- Report generation

Classes:
- LogAnalyzer: Comprehensive log analysis with pattern matching
- IntegrityVerifier: Cryptographic integrity verification for log files
- LogAggregator: Unified log data collection and analysis
- ReportGenerator: Automated report generation from log data
"""

import re
import hashlib
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter

# Import logging configuration
try:
    from utils.logging_config import apgi_logger
except ImportError:
    apgi_logger = None  # type: ignore


@dataclass
class LogEntry:
    """Single log entry with metadata."""

    timestamp: str
    level: str
    message: str
    module: str
    function: Optional[str]
    line: Optional[int]
    file_path: Optional[str]
    hash_value: Optional[str]
    raw_data: Optional[str]
    chain_hash: Optional[str] = None  # Link to previous entry for chain integrity
    entry_number: Optional[int] = None  # Sequential entry number
    anomalies: List[Dict] = field(default_factory=list)  # Detected anomalies


class LogAnalysis:
    """Container for log analysis results."""

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        total_entries: int = 0,
        entries_by_level: Optional[Dict[str, List[LogEntry]]] = None,
        entries_by_module: Optional[Dict[str, List[LogEntry]]] = None,
        entries_by_function: Optional[Dict[str, List[LogEntry]]] = None,
        anomalies: Optional[List[Dict[str, Any]]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        summary: str = "",
        file_integrity: Optional[Dict[str, bool]] = None,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.total_entries = total_entries
        self.entries_by_level = entries_by_level or {}
        self.entries_by_module = entries_by_module or {}
        self.entries_by_function = entries_by_function or {}
        self.anomalies = anomalies or []
        self.performance_metrics = performance_metrics or {}
        self.summary = summary
        self.file_integrity = file_integrity or {}


class LogAnalyzer:
    """Advanced log analysis with pattern matching and anomaly detection."""

    def __init__(self, log_patterns: Optional[Dict[str, List[str]]] = None):
        self.log_patterns = log_patterns or {
            "error_patterns": [
                r"(?i)error|exception|failed|crashed|timeout",
                r"(?i)critical|fatal|emergency",
                r"(?i)security|breach|unauthorized|injection",
                r"(?i)performance|slow|timeout|memory",
            ],
            "warning_patterns": [
                r"(?i)deprecated|warning|obsolete",
                r"(?i)retry|backoff|circuit_breaker",
            ],
            "info_patterns": [
                r"(?i)info|debug|trace",
                r"(?i)started|completed|finished|initialized",
            ],
            "success_patterns": [
                r"(?i)success|completed|passed|validated",
            ],
        }

    def parse_log_file(self, log_file_path: Path) -> List[LogEntry]:
        """Parse a single log file and extract structured data."""
        entries = []

        try:
            with open(log_file_path, "r", encoding="utf-8", errors="replace") as f:
                for line_num, line in enumerate(f, 1):
                    entry = self._parse_log_line(line, line_num, log_file_path)
                    if entry:
                        entries.append(entry)

            return entries
        except Exception as e:
            if apgi_logger:
                apgi_logger.error(f"Error parsing log file {log_file_path}: {e}")
            return []

    def _parse_log_line(
        self, line: str, line_num: int, log_file_path: Path
    ) -> Optional[LogEntry]:
        """Parse a single log line and extract structured data."""
        line = line.strip()
        if not line:
            return None

        # Extract timestamp
        timestamp_match = re.search(
            r"(\d{4}-\d{2}-\d{4}T\d{2}:\d{2}:\d{2}\.\d{6}", line
        )
        timestamp = (
            timestamp_match.group(0) if timestamp_match else datetime.now().isoformat()
        )

        # Extract log level
        level_match = re.search(r"\b(ERROR|WARNING|INFO|DEBUG|CRITICAL)\b", line)
        level = level_match.group(0).lower() if level_match else "info"

        # Extract module
        module_match = re.search(r"\[(\w+)\.+\]", line)
        module = module_match.group(1) if module_match else "unknown"

        # Extract function
        function_match = re.search(r"\b(\w+)\.\w+\([^)]+)\(", line)
        function = function_match.group(2) if function_match else None

        # Extract message
        message_match = re.search(r"\b([^]]+)\b", line)
        message = message_match.group(1) if message_match else line

        # Extract file path (if present) - not currently used but extracted for future use
        file_match = re.search(r"\b(at|in|on)\s+([^]]+)\b", line)
        _ = (
            file_match.group(1) if file_match else None
        )  # Suppress unused variable warning

        # Calculate hash for integrity verification
        content_to_hash = line + str(line_num)
        hash_value = hashlib.sha256(content_to_hash.encode()).hexdigest()[:16]

        return LogEntry(
            timestamp=timestamp,
            level=level,
            module=module,
            function=function,
            message=message,
            line=line_num,
            file_path=str(log_file_path),
            hash_value=hash_value,
            raw_data=line,
        )

    def analyze_logs(self, log_files: List[Path]) -> LogAnalysis:
        """Analyze multiple log files and generate comprehensive analysis."""
        analysis = LogAnalysis(start_time=datetime.now(), end_time=datetime.now())

        total_entries = 0
        all_entries = []

        for log_file in log_files:
            if not log_file.exists():
                if apgi_logger:
                    apgi_logger.warning(f"Log file not found: {log_file}")
                continue

            entries = self.parse_log_file(log_file)
            total_entries += len(entries)
            all_entries.extend(entries)

            # Analyze entries for patterns
            for entry in entries:
                self._detect_patterns(entry)

        analysis.end_time = datetime.now()
        analysis.total_entries = total_entries
        analysis.entries_by_level = self._group_entries_by_level(all_entries)
        analysis.entries_by_module = self._group_entries_by_module(all_entries)
        analysis.entries_by_function = self._group_entries_by_function(all_entries)

        # Generate summary
        analysis.summary = self._generate_summary(analysis)

        return analysis

    def _detect_patterns(self, entry: LogEntry):
        """Detect patterns and anomalies in log entry."""
        for pattern_name, patterns in self.log_patterns.items():
            for pattern in patterns:
                if re.search(pattern, entry.message, re.IGNORECASE):
                    entry.anomalies.append(
                        {
                            "type": pattern_name,
                            "description": f"Pattern detected: {pattern_name}",
                            "details": entry.message,
                        }
                    )

                    # Mark entry level based on pattern
                    if pattern_name in ["error_patterns", "critical"]:
                        entry.level = "error"
                    elif pattern_name in ["warning_patterns"]:
                        entry.level = "warning"

        # Detect performance issues
        if "timeout" in entry.message.lower():
            entry.anomalies.append(
                {"type": "performance", "description": "Timeout detected in operation"}
            )

        # Detect security issues
        if any(
            pattern in entry.message.lower()
            for pattern in self.log_patterns["security_patterns"]
        ):
            entry.anomalies.append(
                {"type": "security", "description": "Security pattern detected"}
            )

    def _group_entries_by_level(
        self, entries: List[LogEntry]
    ) -> Dict[str, List[LogEntry]]:
        """Group entries by log level."""
        grouped = defaultdict(list)
        for entry in entries:
            grouped[entry.level].append(entry)
        return dict(grouped)

    def _group_entries_by_module(
        self, entries: List[LogEntry]
    ) -> Dict[str, List[LogEntry]]:
        """Group entries by module."""
        grouped = defaultdict(list)
        for entry in entries:
            if entry.module:
                grouped[entry.module].append(entry)
        return dict(grouped)

    def _group_entries_by_function(
        self, entries: List[LogEntry]
    ) -> Dict[str, List[LogEntry]]:
        """Group entries by function."""
        grouped = defaultdict(list)
        for entry in entries:
            if entry.function:
                grouped[entry.function].append(entry)
        return dict(grouped)

    def _verify_file_integrity(self, analysis: LogAnalysis) -> Dict[str, bool]:
        """Verify integrity of log files using chain verification."""
        integrity_results = {}
        verifier = ChainIntegrityVerifier()

        # Get unique file paths from entries
        file_paths = set()
        for entries in analysis.entries_by_module.values():
            for entry in entries:
                if entry.file_path:
                    file_paths.add(entry.file_path)

        for file_path in file_paths:
            if file_path:
                result = verifier.verify_log_chain(Path(file_path))
                integrity_results[file_path] = result["verified"]

        return integrity_results

    def _generate_summary(self, analysis: LogAnalysis) -> str:
        """Generate comprehensive analysis summary."""
        summary_parts = []

        # Level distribution
        level_counts: Dict[str, int] = Counter()
        for entries in analysis.entries_by_level.values():
            for entry in entries:
                level_counts[entry.level] += 1
        summary_parts.append(f"Log Levels: {dict(level_counts)}")

        # Module distribution
        module_counts: Dict[str, int] = Counter()
        for entries in analysis.entries_by_module.values():
            for entry in entries:
                module_counts[entry.module] += 1
        summary_parts.append(f"Modules: {dict(module_counts)}")

        # Top anomalies
        if analysis.anomalies:
            anomaly_types = Counter(ano["type"] for ano in analysis.anomalies)
            summary_parts.append(f"Anomalies: {dict(anomaly_types)}")

        # Performance issues
        performance_issues = [
            ano for ano in analysis.anomalies if ano["type"] == "performance"
        ]
        summary_parts.append(f"Performance Issues: {len(performance_issues)}")

        # Security issues
        security_issues = [
            ano for ano in analysis.anomalies if ano["type"] == "security"
        ]
        summary_parts.append(f"Security Issues: {len(security_issues)}")

        # File integrity
        integrity_results = self._verify_file_integrity(analysis)
        summary_parts.append(f"File Integrity: {integrity_results}")

        return "\n".join(summary_parts)


class ChainIntegrityVerifier:
    """
    Verify cryptographic integrity of log files using blockchain-style chain hashing.

    This creates a tamper-evident audit trail where each entry contains a hash
    of the previous entry, making any modification detectable.
    """

    def __init__(self, integrity_file: Optional[str] = None):
        """
        Initialize chain integrity verifier.

        Args:
            integrity_file: Path to store integrity metadata (chain hashes, etc.)
        """
        self.integrity_file = (
            Path(integrity_file) if integrity_file else Path(".log_integrity")
        )
        self.chain_hashes: Dict[str, str] = {}
        self.entry_counters: Dict[str, int] = {}
        self._load_integrity_data()

    def _load_integrity_data(self) -> None:
        """Load previously stored integrity data."""
        if self.integrity_file.exists():
            try:
                with open(self.integrity_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.chain_hashes = data.get("chain_hashes", {})
                    self.entry_counters = data.get("entry_counters", {})
            except (json.JSONDecodeError, IOError) as e:
                if apgi_logger:
                    apgi_logger.warning(f"Could not load integrity data: {e}")

    def _save_integrity_data(self) -> None:
        """Save integrity data to disk."""
        try:
            with open(self.integrity_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "chain_hashes": self.chain_hashes,
                        "entry_counters": self.entry_counters,
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except IOError as e:
            if apgi_logger:
                apgi_logger.error(f"Could not save integrity data: {e}")

    def _calculate_entry_hash(self, entry: LogEntry, previous_hash: str = "0") -> str:
        """
        Calculate cryptographic hash for a log entry.

        Args:
            entry: Log entry to hash
            previous_hash: Hash of previous entry in chain

        Returns:
            Hex digest of entry hash
        """
        content = f"{entry.timestamp}:{entry.level}:{entry.message}:{entry.module}:{previous_hash}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def verify_log_chain(self, log_file: Path) -> Dict[str, Any]:
        """
        Verify integrity of a log file using chain verification.

        Args:
            log_file: Path to log file to verify

        Returns:
            Dictionary with verification results
        """
        if not log_file.exists():
            return {
                "file": str(log_file),
                "verified": False,
                "error": "File not found",
                "tampered_entries": [],
                "missing_entries": [],
            }

        file_key = str(log_file.resolve())
        last_stored_hash = self.chain_hashes.get(file_key, "0")
        expected_counter = self.entry_counters.get(file_key, 0)

        results: Dict[str, Any] = {
            "file": str(log_file),
            "verified": True,
            "total_entries": 0,
            "tampered_entries": [],
            "missing_entries": [],
            "chain_valid": True,
            "details": [],
        }

        previous_hash = "0"
        entry_count = 0

        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                entry = self._parse_line_to_entry(line, line_num, log_file)
                if not entry:
                    continue

                entry_count += 1
                expected_hash = self._calculate_entry_hash(entry, previous_hash)

                # Check if entry has stored chain hash
                if entry.chain_hash:
                    if entry.chain_hash != expected_hash:
                        results["tampered_entries"].append(
                            {
                                "line": line_num,
                                "expected_hash": expected_hash,
                                "actual_hash": entry.chain_hash,
                                "timestamp": entry.timestamp,
                            }
                        )
                        results["chain_valid"] = False
                        results["verified"] = False

                # Update previous hash for next entry
                previous_hash = expected_hash

                results["details"].append(
                    {
                        "line": line_num,
                        "hash": expected_hash,
                        "timestamp": entry.timestamp,
                    }
                )

            results["total_entries"] = entry_count

            # Check for missing entries (gap in sequence)
            if expected_counter > 0 and entry_count < expected_counter:
                missing_count = expected_counter - entry_count
                results["missing_entries"].append(
                    {
                        "expected_count": expected_counter,
                        "actual_count": entry_count,
                        "missing": missing_count,
                    }
                )
                results["verified"] = False

            # Verify final chain hash matches stored value
            if last_stored_hash != "0" and previous_hash != last_stored_hash:
                results["chain_valid"] = False
                results["verified"] = False
                results["final_hash_mismatch"] = {
                    "expected": last_stored_hash,
                    "actual": previous_hash,
                }

        except Exception as e:
            results["verified"] = False
            results["error"] = str(e)

        return results

    def _parse_line_to_entry(
        self, line: str, line_num: int, log_file: Path
    ) -> Optional[LogEntry]:
        """Parse a log line into a LogEntry."""
        line = line.strip()
        if not line:
            return None

        # Try to extract timestamp
        timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", line)
        timestamp = (
            timestamp_match.group(0) if timestamp_match else datetime.now().isoformat()
        )

        # Extract log level
        level_match = re.search(r"\b(ERROR|WARNING|INFO|DEBUG|CRITICAL)\b", line)
        level = level_match.group(0).lower() if level_match else "info"

        # Extract message (rest of line after timestamp/level)
        message = line

        # Extract chain hash if present (format: [CHAIN:hash])
        chain_match = re.search(r"\[CHAIN:([a-f0-9]{32})\]", line)
        chain_hash = chain_match.group(1) if chain_match else None

        # Extract entry number if present (format: [ENTRY:num])
        entry_match = re.search(r"\[ENTRY:(\d+)\]", line)
        entry_number = int(entry_match.group(1)) if entry_match else None

        return LogEntry(
            timestamp=timestamp,
            level=level,
            message=message,
            module="unknown",
            function=None,
            line=line_num,
            file_path=str(log_file),
            hash_value=None,
            raw_data=line,
            chain_hash=chain_hash,
            entry_number=entry_number,
        )

    def update_chain_hash(self, log_file: Path, entries: List[LogEntry]) -> None:
        """
        Update chain hash for a log file after new entries.

        Args:
            log_file: Path to log file
            entries: List of entries to add to chain
        """
        file_key = str(log_file.resolve())

        # Get last known hash
        last_hash = self.chain_hashes.get(file_key, "0")
        current_counter = self.entry_counters.get(file_key, 0)

        # Calculate new hashes
        for i, entry in enumerate(entries):
            entry.entry_number = current_counter + i + 1
            entry_hash = self._calculate_entry_hash(entry, last_hash)
            entry.chain_hash = entry_hash
            last_hash = entry_hash

        # Store updated hash
        self.chain_hashes[file_key] = last_hash
        self.entry_counters[file_key] = current_counter + len(entries)

        # Save integrity data
        self._save_integrity_data()

    def generate_integrity_certificate(self, log_file: Path) -> Dict[str, Any]:
        """
        Generate integrity certificate for a log file.

        Args:
            log_file: Path to log file

        Returns:
            Dictionary containing integrity certificate
        """
        file_key = str(log_file.resolve())
        verification = self.verify_log_chain(log_file)

        certificate = {
            "file": str(log_file),
            "generated_at": datetime.now().isoformat(),
            "file_size": log_file.stat().st_size if log_file.exists() else 0,
            "file_modified": (
                datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                if log_file.exists()
                else None
            ),
            "total_entries": verification.get("total_entries", 0),
            "chain_hash": self.chain_hashes.get(file_key, "0"),
            "entry_count": self.entry_counters.get(file_key, 0),
            "verification_status": (
                "VERIFIED" if verification["verified"] else "TAMPERED"
            ),
            "chain_valid": verification.get("chain_valid", False),
            "tampered_entries": len(verification.get("tampered_entries", [])),
            "missing_entries": len(verification.get("missing_entries", [])),
            "certificate_hash": None,
        }

        # Calculate certificate hash
        cert_content = f"{certificate['file']}:{certificate['chain_hash']}:{certificate['total_entries']}"
        certificate["certificate_hash"] = hashlib.sha256(
            cert_content.encode()
        ).hexdigest()[:32]

        return certificate


class IntegrityVerifier:
    """Verify cryptographic integrity of log files (backward compatibility wrapper)."""

    def __init__(self, expected_hashes: Optional[Dict[str, str]] = None):
        self.expected_hashes = expected_hashes or {}
        self.chain_verifier = ChainIntegrityVerifier()

    def _verify_file_integrity(self, analysis: Any) -> Dict[str, bool]:
        """Verify integrity of log files using chain verification."""
        integrity_results = {}

        for file_path in analysis.entries_by_module.keys():
            if file_path:
                result = self.chain_verifier.verify_log_chain(Path(file_path))
                integrity_results[file_path] = result["verified"]

        return integrity_results

    def verify_log_chain(self, log_file: Path) -> Dict[str, Any]:
        """Delegate to ChainIntegrityVerifier for chain verification."""
        return self.chain_verifier.verify_log_chain(log_file)


class LogAggregator:
    """Aggregate log data from multiple sources for unified analysis."""

    def __init__(self, log_sources: Dict[str, str]):
        """Initialize aggregator with multiple log sources."""
        self.log_sources = log_sources
        self.analysis_cache: Dict[str, LogAnalysis] = {}

    def aggregate_logs(
        self, time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> LogAnalysis:
        """Aggregate logs from multiple sources within time range."""
        if not self.log_sources:
            return LogAnalysis(
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_entries=0,
                entries_by_level={},
                entries_by_module={},
                entries_by_function={},
                anomalies=[],
                performance_metrics={},
                summary="No log sources provided",
            )

        all_entries = []
        start_time = (
            time_range[0]
            if time_range and isinstance(time_range[0], datetime)
            else datetime.now() - timedelta(days=7)
        )
        end_time = (
            time_range[1]
            if time_range and isinstance(time_range[1], datetime)
            else datetime.now()
        )

        # Collect entries from all sources within time range
        for source_name, source_path in self.log_sources.items():
            source_file = Path(source_path)
            if not source_file.exists():
                if apgi_logger:
                    apgi_logger.warning(f"Log source not found: {source_path}")
                continue

            # Read and parse recent entries
            recent_entries: List[LogEntry] = []
            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()[-1000:]  # Read last 1000 lines
                    for line in lines:
                        entry = self._parse_log_line(
                            line, len(recent_entries), source_path
                        )
                        if (
                            entry
                            and datetime.fromisoformat(entry.timestamp) >= start_time
                            and datetime.fromisoformat(entry.timestamp) <= end_time
                        ):
                            recent_entries.append(entry)
            except Exception as e:
                if apgi_logger:
                    apgi_logger.error(f"Error reading log source {source_path}: {e}")

            all_entries.extend(recent_entries)

        analysis = LogAnalysis(
            start_time=start_time,
            end_time=end_time,
            total_entries=len(all_entries),
            entries_by_level=self._group_entries_by_level(all_entries),
            entries_by_module=self._group_entries_by_module(all_entries),
            entries_by_function=self._group_entries_by_function(all_entries),
            anomalies=self._detect_patterns_batch(all_entries),
            summary=self._generate_summary(
                LogAnalysis(
                    start_time=start_time,
                    end_time=end_time,
                    total_entries=len(all_entries),
                    entries_by_level=self._group_entries_by_level(all_entries),
                    entries_by_module=self._group_entries_by_module(all_entries),
                    entries_by_function=self._group_entries_by_function(all_entries),
                    anomalies=[],
                    performance_metrics={},
                    summary=f"Aggregated {len(self.log_sources)} sources from {start_time} to {end_time}",
                )
            ),
        )

        # Cache the analysis
        self.analysis_cache[f"{start_time.isoformat()}_{end_time.isoformat()}"] = (
            analysis
        )

        return analysis

    def _parse_log_line(
        self, line: str, line_num: int, log_file_path: str
    ) -> Optional[LogEntry]:
        """Parse a single log line and extract structured data."""
        line = line.strip()
        if not line:
            return None

        # Extract timestamp
        timestamp_match = re.search(
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6})", line
        )
        timestamp = (
            timestamp_match.group(0) if timestamp_match else datetime.now().isoformat()
        )

        # Extract log level
        level_match = re.search(r"\b(ERROR|WARNING|INFO|DEBUG|CRITICAL)\b", line)
        level = level_match.group(0).lower() if level_match else "info"

        # Extract module
        module_match = re.search(r"\[(\w+)\.+\]", line)
        module = module_match.group(1) if module_match else "unknown"

        # Extract function
        function_match = re.search(r"\b(\w+)\.\w+\([^)]+)\(", line)
        function = function_match.group(2) if function_match else None

        # Extract message
        message_match = re.search(r"\b([^]]+)\b", line)
        message = message_match.group(1) if message_match else line

        # Calculate hash for integrity verification
        content_to_hash = line + str(line_num)
        hash_value = hashlib.sha256(content_to_hash.encode()).hexdigest()[:16]

        return LogEntry(
            timestamp=timestamp,
            level=level,
            module=module,
            function=function,
            message=message,
            line=line_num,
            file_path=log_file_path,
            hash_value=hash_value,
            raw_data=line,
        )

    def _group_entries_by_level(
        self, entries: List[LogEntry]
    ) -> Dict[str, List[LogEntry]]:
        """Group entries by log level."""
        grouped = defaultdict(list)
        for entry in entries:
            grouped[entry.level].append(entry)
        return dict(grouped)

    def _group_entries_by_module(
        self, entries: List[LogEntry]
    ) -> Dict[str, List[LogEntry]]:
        """Group entries by module."""
        grouped = defaultdict(list)
        for entry in entries:
            if entry.module:
                grouped[entry.module].append(entry)
        return dict(grouped)

    def _group_entries_by_function(
        self, entries: List[LogEntry]
    ) -> Dict[str, List[LogEntry]]:
        """Group entries by function."""
        grouped = defaultdict(list)
        for entry in entries:
            if entry.function:
                grouped[entry.function].append(entry)
        return dict(grouped)

    def _detect_patterns_batch(self, entries: List[LogEntry]) -> List[Dict[str, Any]]:
        """Detect patterns in a batch of entries."""
        anomalies = []
        for entry in entries:
            # Simple pattern detection
            if "error" in entry.message.lower() or "exception" in entry.message.lower():
                anomalies.append(
                    {
                        "type": "error",
                        "description": "Error pattern detected",
                        "details": entry.message,
                    }
                )
            if "timeout" in entry.message.lower():
                anomalies.append(
                    {
                        "type": "performance",
                        "description": "Timeout detected",
                        "details": entry.message,
                    }
                )
        return anomalies

    def _generate_summary(self, analysis: LogAnalysis) -> str:
        """Generate summary for analysis."""
        return f"Analyzed {analysis.total_entries} entries from {len(analysis.entries_by_module)} sources"


class ReportGenerator:
    """Generate comprehensive reports from log analysis data."""

    def __init__(self, output_dir: str = "reports"):
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_integrity_report(self, analysis: LogAnalysis) -> str:
        """Generate integrity verification report."""
        report_lines = [
            "APGI Framework - Log Integrity Report",
            "=" * 80,
            f"Generated: {analysis.end_time.isoformat()}",
            (
                f"Time Range: {analysis.start_time.isoformat()} to {analysis.end_time.isoformat()}"
                if analysis.start_time and analysis.end_time
                else "N/A"
            ),
        ]

        # Add integrity results
        for file_path, is_valid in analysis.file_integrity.items():
            status = "✅ VERIFIED" if is_valid else "❌ FAILED"
            report_lines.append(f"  {file_path}: {status}")

        report_lines.append("")

        return "\n".join(report_lines)

    def generate_performance_report(self, analysis: LogAnalysis) -> str:
        """Generate performance metrics report."""
        if not analysis.performance_metrics:
            return "No performance metrics available"

        report_lines = [
            "APGI Framework - Performance Metrics Report",
            "=" * 80,
            f"Generated: {analysis.end_time.isoformat()}",
            (
                f"Time Range: {analysis.start_time.isoformat()} to {analysis.end_time.isoformat()}"
                if analysis.start_time and analysis.end_time
                else "N/A"
            ),
        ]

        # Add performance metrics by level
        for level, metrics in analysis.performance_metrics.items():
            report_lines.append(f"{level.upper()} Metrics:")
            for metric, value in metrics.items():
                report_lines.append(f"  {metric}: {value}")

        return "\n".join(report_lines)

    def generate_security_report(self, analysis: LogAnalysis) -> str:
        """Generate security issues report."""
        if not analysis.anomalies:
            return "No security issues detected"

        report_lines = [
            "APGI Framework - Security Report",
            "=" * 80,
            f"Generated: {analysis.end_time.isoformat()}",
            (
                f"Time Range: {analysis.start_time.isoformat()} to {analysis.end_time.isoformat()}"
                if analysis.start_time and analysis.end_time
                else "N/A"
            ),
        ]

        # Add security issues
        security_issues = [
            ano for ano in analysis.anomalies if ano.get("type") == "security"
        ]
        for issue in security_issues:
            report_lines.append(
                f"  {issue['description']} (File: {issue.get('file_path', 'Unknown')})"
            )

        return "\n".join(report_lines)

    def generate_anomaly_report(self, analysis: LogAnalysis) -> str:
        """Generate anomaly detection report."""
        if not analysis.anomalies:
            return "No anomalies detected"

        report_lines = [
            "APGI Framework - Anomaly Detection Report",
            "=" * 80,
            f"Generated: {analysis.end_time.isoformat()}",
            (
                f"Time Range: {analysis.start_time.isoformat()} to {analysis.end_time.isoformat()}"
                if analysis.start_time and analysis.end_time
                else "N/A"
            ),
        ]

        # Add anomalies by type
        anomaly_types = Counter(ano["type"] for ano in analysis.anomalies)
        for anomaly_type, count in anomaly_types.items():
            report_lines.append(f"{anomaly_type.upper()} Anomalies: {count}")
            for ano in analysis.anomalies:
                if ano["type"] == anomaly_type:
                    report_lines.append(f"  - {ano['description']}")

        return "\n".join(report_lines)

    def generate_summary_report(self, analysis: LogAnalysis) -> str:
        """Generate comprehensive summary report."""
        report_lines = [
            "APGI Framework - Log Analysis Summary",
            "=" * 80,
            f"Generated: {analysis.end_time.isoformat()}",
            (
                f"Time Range: {analysis.start_time.isoformat()} to {analysis.end_time.isoformat()}"
                if analysis.start_time and analysis.end_time
                else "N/A"
            ),
        ]

        # Add overview statistics
        report_lines.append(f"Total Entries Analyzed: {analysis.total_entries}")
        report_lines.append(f"Sources Analyzed: {len(analysis.entries_by_module)}")
        report_lines.append(f"Anomalies Detected: {len(analysis.anomalies)}")
        report_lines.append(
            f"Performance Issues: {int(len([ano for ano in analysis.anomalies if ano.get('type') == 'performance']))}"
        )
        report_lines.append(
            f"Security Issues: {int(len([ano for ano in analysis.anomalies if ano.get('type') == 'security']))}"
        )

        # Add file integrity summary
        integrity_valid = all(is_valid for is_valid in analysis.file_integrity.values())
        valid_count = sum(
            1 for is_valid in analysis.file_integrity.values() if is_valid
        )
        report_lines.append(
            f"File Integrity: {'✅ VERIFIED' if integrity_valid else '❌ FAILED'} ({valid_count}/{len(analysis.file_integrity)})"
        )

        report_lines.append("")
        report_lines.append("")

        return "\n".join(report_lines)

    def generate_executive_summary(
        self,
        analysis: LogAnalysis,
        period_name: str = "Current Period",
        include_recommendations: bool = True,
    ) -> str:
        """
        Generate executive summary for leadership and stakeholders.

        Provides high-level overview with key metrics, trends, and
        actionable insights suitable for executive reporting.

        Args:
            analysis: Log analysis data
            period_name: Name of the reporting period
            include_recommendations: Whether to include recommendations

        Returns:
            Executive summary report as formatted string
        """
        report_lines = [
            "APGI FRAMEWORK - EXECUTIVE SUMMARY",
            "=" * 80,
            f"Period: {period_name}",
            f"Generated: {datetime.now().isoformat()}",
            (
                f"Analysis Time Range: {analysis.start_time.strftime('%Y-%m-%d %H:%M')} to {analysis.end_time.strftime('%Y-%m-%d %H:%M')}"
                if analysis.start_time and analysis.end_time
                else "Analysis Time Range: N/A"
            ),
            "",
            "KEY METRICS",
            "-" * 40,
        ]

        # Calculate key metrics
        total_entries = analysis.total_entries
        error_count = len(analysis.entries_by_level.get("error", []))
        warning_count = len(analysis.entries_by_level.get("warning", []))
        _ = len(
            analysis.entries_by_level.get("info", [])
        )  # info count not currently used

        error_rate = (error_count / total_entries * 100) if total_entries > 0 else 0
        warning_rate = (warning_count / total_entries * 100) if total_entries > 0 else 0

        # Security and performance metrics
        security_issues = [a for a in analysis.anomalies if a.get("type") == "security"]
        performance_issues = [
            a for a in analysis.anomalies if a.get("type") == "performance"
        ]
        integrity_issues = [
            a for a in analysis.anomalies if a.get("type") == "integrity"
        ]

        # System health score (0-100)
        health_score: float = 100.0
        health_score -= min(30.0, error_rate * 3)  # Up to 30 points for errors
        health_score -= min(20.0, warning_rate * 2)  # Up to 20 points for warnings
        health_score -= min(25, len(security_issues) * 5)  # Up to 25 for security
        health_score -= min(15, len(performance_issues) * 3)  # Up to 15 for performance
        health_score -= min(10, len(integrity_issues) * 10)  # Up to 10 for integrity
        health_score = max(0, health_score)

        # Health indicator
        if health_score >= 90:
            health_status = "✅ HEALTHY"
        elif health_score >= 70:
            health_status = "⚠️  DEGRADED"
        else:
            health_status = "❌ CRITICAL"

        report_lines.extend(
            [
                f"System Health Score: {health_score:.1f}/100 {health_status}",
                f"Total Log Entries: {total_entries:,}",
                f"Error Rate: {error_rate:.2f}% ({error_count} errors)",
                f"Warning Rate: {warning_rate:.2f}% ({warning_count} warnings)",
                f"Security Issues: {len(security_issues)}",
                f"Performance Issues: {len(performance_issues)}",
                f"Integrity Violations: {len(integrity_issues)}",
                "",
                "CRITICAL FINDINGS",
                "-" * 40,
            ]
        )

        # Critical findings section
        critical_findings = []

        if security_issues:
            critical_findings.append(
                f"🔒 SECURITY: {len(security_issues)} security issues detected requiring immediate attention"
            )

        if integrity_issues:
            critical_findings.append(
                f"🔐 INTEGRITY: {len(integrity_issues)} log integrity violations detected"
            )

        if error_rate > 5:
            critical_findings.append(
                f"❌ ERRORS: High error rate of {error_rate:.1f}% indicates system instability"
            )

        if performance_issues:
            critical_findings.append(
                f"⚡ PERFORMANCE: {len(performance_issues)} performance anomalies detected"
            )

        if not critical_findings:
            report_lines.append(
                "No critical findings. System operating within normal parameters."
            )
        else:
            for finding in critical_findings:
                report_lines.append(finding)

        report_lines.append("")

        # Top affected modules
        if analysis.entries_by_module:
            report_lines.extend(
                [
                    "TOP ACTIVITY SOURCES",
                    "-" * 40,
                ]
            )
            module_counts = {
                module: len(entries)
                for module, entries in analysis.entries_by_module.items()
            }
            sorted_modules = sorted(
                module_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
            for module, count in sorted_modules:
                report_lines.append(f"  {module}: {count:,} entries")
            report_lines.append("")

        # Recommendations
        if include_recommendations:
            report_lines.extend(
                [
                    "RECOMMENDATIONS",
                    "-" * 40,
                ]
            )

            recommendations = []

            if health_score < 70:
                recommendations.append(
                    "🚨 PRIORITY: System health is CRITICAL. Initiate incident response procedures."
                )

            if security_issues:
                recommendations.append(
                    "🔒 SECURITY: Review security logs immediately. Consider rotating credentials and auditing access."
                )

            if integrity_issues:
                recommendations.append(
                    "🔐 INTEGRITY: Log tampering detected. Conduct forensic analysis and verify backup integrity."
                )

            if error_rate > 10:
                recommendations.append(
                    "❌ STABILITY: High error rate suggests systemic issues. Review recent deployments and configuration changes."
                )

            if performance_issues:
                recommendations.append(
                    "⚡ PERFORMANCE: Performance degradation detected. Consider scaling resources or optimizing hot paths."
                )

            if not recommendations:
                recommendations.append(
                    "✅ System is healthy. Continue standard monitoring and maintenance procedures."
                )

            for rec in recommendations:
                report_lines.append(rec)

            report_lines.append("")

        # Footer
        report_lines.extend(
            [
                "=" * 80,
                "For detailed technical analysis, see accompanying technical report.",
                "This summary was automatically generated by APGI Log Analysis System.",
            ]
        )

    def generate_statistical_significance_report(
        self,
        current_analysis: LogAnalysis,
        baseline_analysis: Optional[LogAnalysis] = None,
        confidence_level: float = 0.95,
    ) -> str:
        """
        Generate statistical significance report comparing current vs baseline.

        Performs statistical tests to determine if observed changes are
        statistically significant or due to random variation.

        Args:
            current_analysis: Current period log analysis
            baseline_analysis: Baseline period for comparison (optional)
            confidence_level: Confidence level for significance testing (default 0.95)

        Returns:
            Statistical significance report as formatted string
        """
        from math import sqrt

        report_lines = [
            "APGI FRAMEWORK - STATISTICAL SIGNIFICANCE REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            f"Confidence Level: {confidence_level * 100:.0f}%",
            "",
            "CURRENT PERIOD STATISTICS",
            "-" * 40,
        ]

        # Current period metrics
        current_total = current_analysis.total_entries
        current_errors = len(current_analysis.entries_by_level.get("error", []))
        current_warnings = len(current_analysis.entries_by_level.get("warning", []))
        current_info = len(current_analysis.entries_by_level.get("info", []))

        current_error_rate = (
            (current_errors / current_total * 100) if current_total > 0 else 0
        )
        current_warning_rate = (
            (current_warnings / current_total * 100) if current_total > 0 else 0
        )

        report_lines.extend(
            [
                f"Total Entries: {current_total:,}",
                f"Error Rate: {current_error_rate:.2f}% ({current_errors} errors)",
                f"Warning Rate: {current_warning_rate:.2f}% ({current_warnings} warnings)",
                f"Info Rate: {(current_info / current_total * 100):.2f}% ({current_info} info)",
                "",
            ]
        )

        # Statistical significance testing if baseline provided
        if baseline_analysis:
            baseline_total = baseline_analysis.total_entries
            baseline_errors = len(baseline_analysis.entries_by_level.get("error", []))
            baseline_warnings = len(
                baseline_analysis.entries_by_level.get("warning", [])
            )

            baseline_error_rate = (
                (baseline_errors / baseline_total * 100) if baseline_total > 0 else 0
            )
            baseline_warning_rate = (
                (baseline_warnings / baseline_total * 100) if baseline_total > 0 else 0
            )

            report_lines.extend(
                [
                    "BASELINE PERIOD STATISTICS",
                    "-" * 40,
                    f"Total Entries: {baseline_total:,}",
                    f"Error Rate: {baseline_error_rate:.2f}% ({baseline_errors} errors)",
                    f"Warning Rate: {baseline_warning_rate:.2f}% ({baseline_warnings} warnings)",
                    "",
                    "COMPARATIVE ANALYSIS",
                    "-" * 40,
                ]
            )

            # Calculate percent changes
            error_change = current_error_rate - baseline_error_rate
            warning_change = current_warning_rate - baseline_warning_rate
            volume_change = (
                ((current_total - baseline_total) / baseline_total * 100)
                if baseline_total > 0
                else 0
            )

            # Simple z-test for proportions approximation
            def calculate_z_score(p1, p2, n1, n2):
                """Calculate z-score for two-proportion z-test."""
                if n1 == 0 or n2 == 0:
                    return 0
                p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
                if p_pooled == 0 or p_pooled == 1:
                    return 0
                se = sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
                if se == 0:
                    return 0
                return (p1 - p2) / se

            # Critical z-value for given confidence level
            z_critical = (
                1.96
                if confidence_level == 0.95
                else 2.576 if confidence_level == 0.99 else 1.645
            )

            # Test error rate change
            error_z = calculate_z_score(
                current_error_rate / 100,
                baseline_error_rate / 100,
                current_total,
                baseline_total,
            )
            error_significant = abs(error_z) > z_critical

            # Test warning rate change
            warning_z = calculate_z_score(
                current_warning_rate / 100,
                baseline_warning_rate / 100,
                current_total,
                baseline_total,
            )
            warning_significant = abs(warning_z) > z_critical

            report_lines.extend(
                [
                    f"Error Rate Change: {error_change:+.2f}%",
                    f"  Statistical Significance: {'✅ SIGNIFICANT' if error_significant else '❌ Not Significant'}",
                    f"  Z-Score: {error_z:.3f} (threshold: ±{z_critical:.3f})",
                    f"  Interpretation: {'Change is unlikely due to chance' if error_significant else 'Change may be due to random variation'}",
                    "",
                    f"Warning Rate Change: {warning_change:+.2f}%",
                    f"  Statistical Significance: {'✅ SIGNIFICANT' if warning_significant else '❌ Not Significant'}",
                    f"  Z-Score: {warning_z:.3f} (threshold: ±{z_critical:.3f})",
                    f"  Interpretation: {'Change is unlikely due to chance' if warning_significant else 'Change may be due to random variation'}",
                    "",
                    f"Log Volume Change: {volume_change:+.1f}%",
                    f"  Current: {current_total:,} vs Baseline: {baseline_total:,}",
                    "",
                ]
            )

            # Trend analysis
            report_lines.extend(
                [
                    "TREND ANALYSIS",
                    "-" * 40,
                ]
            )

            if error_significant and error_change > 0:
                report_lines.append(
                    "🔴 ERROR TREND: Significant increase in error rate detected"
                )
            elif error_significant and error_change < 0:
                report_lines.append(
                    "🟢 ERROR TREND: Significant decrease in error rate (improvement)"
                )
            else:
                report_lines.append(
                    "⚪ ERROR TREND: No significant change in error rate"
                )

            if warning_significant and warning_change > 0:
                report_lines.append(
                    "🟡 WARNING TREND: Significant increase in warning rate"
                )
            elif warning_significant and warning_change < 0:
                report_lines.append(
                    "🟢 WARNING TREND: Significant decrease in warning rate"
                )
            else:
                report_lines.append(
                    "⚪ WARNING TREND: No significant change in warning rate"
                )

            report_lines.append("")

        else:
            report_lines.extend(
                [
                    "COMPARATIVE ANALYSIS",
                    "-" * 40,
                    "No baseline data provided. Cannot perform statistical significance testing.",
                    "To enable comparison, provide baseline_analysis parameter with historical data.",
                    "",
                ]
            )

        # Summary statistics
        report_lines.extend(
            [
                "SUMMARY",
                "-" * 40,
                f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Methodology: Two-proportion z-test at {confidence_level * 100:.0f}% confidence level",
                "Note: Statistical significance indicates whether observed changes are",
                "      likely due to actual system changes rather than random variation.",
                "",
                "=" * 80,
            ]
        )

        return "\n".join(report_lines)

    def generate_cross_protocol_correlation_report(
        self,
        protocol_results: Dict[str, LogAnalysis],
    ) -> str:
        """
        Generate cross-protocol correlation analysis report.

        Analyzes relationships and correlations between different protocols
        to identify systemic patterns and dependencies.

        Args:
            protocol_results: Dictionary mapping protocol names to their analyses

        Returns:
            Cross-protocol correlation report as formatted string
        """
        report_lines = [
            "APGI FRAMEWORK - CROSS-PROTOCOL CORRELATION ANALYSIS",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            f"Protocols Analyzed: {len(protocol_results)}",
            "",
            "PROTOCOL OVERVIEW",
            "-" * 40,
        ]

        # Protocol summary table
        for protocol_name, analysis in protocol_results.items():
            error_count = len(analysis.entries_by_level.get("error", []))
            _ = len(analysis.entries_by_level.get("warning", []))  # not currently used
            total = analysis.total_entries

            error_rate = (error_count / total * 100) if total > 0 else 0

            report_lines.append(
                f"  {protocol_name:20s}: {total:8,} entries | "
                f"{error_count:5d} errors ({error_rate:5.2f}%) | "
                f"{len(analysis.anomalies):3d} anomalies"
            )

        report_lines.append("")

        # Error correlation analysis
        if len(protocol_results) > 1:
            report_lines.extend(
                [
                    "ERROR CORRELATION ANALYSIS",
                    "-" * 40,
                ]
            )

            # Calculate error rates for each protocol
            protocol_error_rates = {}
            for protocol_name, analysis in protocol_results.items():
                error_count = len(analysis.entries_by_level.get("error", []))
                total = analysis.total_entries
                protocol_error_rates[protocol_name] = (
                    (error_count / total * 100) if total > 0 else 0
                )

            # Find protocols with similar error patterns
            high_error_protocols = [
                name for name, rate in protocol_error_rates.items() if rate > 5.0
            ]
            low_error_protocols = [
                name for name, rate in protocol_error_rates.items() if rate < 1.0
            ]

            if len(high_error_protocols) > 1:
                report_lines.append(
                    f"🚨 HIGH ERROR CORRELATION: {len(high_error_protocols)} protocols show elevated error rates (>5%):"
                )
                for protocol in high_error_protocols:
                    report_lines.append(
                        f"   - {protocol}: {protocol_error_rates[protocol]:.2f}%"
                    )
                report_lines.append("")

            if len(low_error_protocols) > 1:
                report_lines.append(
                    f"✅ STABLE PROTOCOLS: {len(low_error_protocols)} protocols show low error rates (<1%):"
                )
                for protocol in low_error_protocols[:5]:  # Show top 5
                    report_lines.append(
                        f"   - {protocol}: {protocol_error_rates[protocol]:.2f}%"
                    )
                report_lines.append("")

            # Anomaly pattern correlation
            report_lines.extend(
                [
                    "ANOMALY PATTERN CORRELATION",
                    "-" * 40,
                ]
            )

            # Collect anomaly types across protocols
            all_anomaly_types: Dict[str, List[str]] = {}
            for protocol_name, analysis in protocol_results.items():
                for anomaly in analysis.anomalies:
                    anomaly_type = anomaly.get("type", "unknown")
                    if anomaly_type not in all_anomaly_types:
                        all_anomaly_types[anomaly_type] = []
                    all_anomaly_types[anomaly_type].append(protocol_name)

            # Find common anomaly patterns
            common_patterns = {
                atype: protocols
                for atype, protocols in all_anomaly_types.items()
                if len(protocols) > 1
            }

            if common_patterns:
                report_lines.append(
                    f"Found {len(common_patterns)} anomaly patterns affecting multiple protocols:"
                )
                for atype, protocols in sorted(
                    common_patterns.items(), key=lambda x: -len(x[1])
                ):
                    report_lines.append(
                        f"   - {atype}: affects {len(protocols)} protocols ({', '.join(protocols[:3])}{'...' if len(protocols) > 3 else ''})"
                    )
            else:
                report_lines.append(
                    "No common anomaly patterns detected across protocols."
                )

            report_lines.append("")

            # Systemic issues identification
            report_lines.extend(
                [
                    "SYSTEMIC ISSUE DETECTION",
                    "-" * 40,
                ]
            )

            systemic_issues = []

            # Check for widespread issues
            protocols_with_errors = sum(
                1
                for analysis in protocol_results.values()
                if len(analysis.entries_by_level.get("error", [])) > 0
            )

            if protocols_with_errors > len(protocol_results) * 0.5:
                systemic_issues.append(
                    f"⚠️  WIDESPREAD ERRORS: {protocols_with_errors}/{len(protocol_results)} protocols reporting errors"
                )

            protocols_with_security = sum(
                1
                for analysis in protocol_results.values()
                if any(a.get("type") == "security" for a in analysis.anomalies)
            )

            if protocols_with_security > 1:
                systemic_issues.append(
                    f"🔒 SECURITY CONCERN: {protocols_with_security} protocols have security-related anomalies"
                )

            protocols_with_performance = sum(
                1
                for analysis in protocol_results.values()
                if any(a.get("type") == "performance" for a in analysis.anomalies)
            )

            if protocols_with_performance > 1:
                systemic_issues.append(
                    f"⚡ PERFORMANCE IMPACT: {protocols_with_performance} protocols showing performance issues"
                )

            if systemic_issues:
                for issue in systemic_issues:
                    report_lines.append(issue)
            else:
                report_lines.append(
                    "✅ No systemic issues detected. Problems appear isolated to individual protocols."
                )

            report_lines.append("")

        # Recommendations
        report_lines.extend(
            [
                "RECOMMENDATIONS",
                "-" * 40,
            ]
        )

        recommendations = []

        # Check for correlated failures
        high_error_count = sum(
            1
            for analysis in protocol_results.values()
            if len(analysis.entries_by_level.get("error", [])) > 10
        )

        if high_error_count > len(protocol_results) * 0.3:
            recommendations.append(
                "🚨 CRITICAL: Multiple protocols experiencing high error rates. Investigate shared infrastructure."
            )

        if len(protocol_results) > 3:
            avg_entries = sum(a.total_entries for a in protocol_results.values()) / len(
                protocol_results
            )
            low_volume = [
                name
                for name, a in protocol_results.items()
                if a.total_entries < avg_entries * 0.1
            ]
            if low_volume:
                recommendations.append(
                    f"📊 VOLUME: {len(low_volume)} protocols have significantly lower activity than average"
                )

        if not recommendations:
            recommendations.append(
                "✅ Protocols appear to be operating independently without significant correlations."
            )

        for rec in recommendations:
            report_lines.append(rec)

        report_lines.extend(
            [
                "",
                "=" * 80,
                "Note: Correlation analysis helps identify whether issues are systemic or isolated.",
                "High correlation suggests shared root causes requiring coordinated response.",
            ]
        )

        return "\n".join(report_lines)


# Convenience functions for common operations
def analyze_log_files(
    log_files: Dict[str, str], time_range: Optional[Tuple[datetime, datetime]] = None
) -> LogAnalysis:
    """Convenience function to analyze multiple log files."""
    aggregator = LogAggregator(log_files)
    return aggregator.aggregate_logs(time_range)


def verify_log_integrity(
    log_files: Dict[str, str], expected_hashes: Dict[str, str] = None
) -> Dict[str, bool]:
    """Convenience function to verify log file integrity."""
    aggregator = LogAggregator(log_files)
    analysis = aggregator.aggregate_logs()

    verifier = IntegrityVerifier(expected_hashes)
    return verifier._verify_file_integrity(analysis)


def verify_log_chain_integrity(log_file: str) -> Dict[str, Any]:
    """
    Verify log file integrity using blockchain-style chain verification.

    Args:
        log_file: Path to log file to verify

    Returns:
        Verification results with tamper detection details
    """
    verifier = ChainIntegrityVerifier()
    return verifier.verify_log_chain(Path(log_file))


def generate_integrity_certificate(log_file: str) -> Dict[str, Any]:
    """
    Generate integrity certificate for a log file.

    Args:
        log_file: Path to log file

    Returns:
        Dictionary containing integrity certificate
    """
    verifier = ChainIntegrityVerifier()
    return verifier.generate_integrity_certificate(Path(log_file))


def generate_comprehensive_report(
    log_files: Dict[str, str],
    output_dir: str = "reports",
    time_range: Optional[Tuple[datetime, datetime]] = None,
    expected_hashes: Optional[Dict[str, str]] = None,
) -> str:
    """Generate comprehensive analysis and integrity reports."""
    # Analyze logs
    analysis = analyze_log_files(log_files, time_range)

    # Generate reports
    report_generator = ReportGenerator(output_dir)

    integrity_report = report_generator.generate_integrity_report(analysis)
    performance_report = report_generator.generate_performance_report(analysis)
    security_report = report_generator.generate_security_report(analysis)
    anomaly_report = report_generator.generate_anomaly_report(analysis)
    summary_report = report_generator.generate_summary_report(analysis)

    # Combine all reports
    full_report = "\n".join(
        [
            integrity_report,
            performance_report,
            security_report,
            anomaly_report,
            summary_report,
        ]
    )

    return full_report


class AutomatedLogAnalyzer:
    """
    Automated log analysis with scheduling and alerting capabilities.

    Provides continuous monitoring of log files with configurable thresholds
    for anomaly detection and alerting.
    """

    def __init__(
        self,
        log_sources: Dict[str, str],
        analysis_interval: int = 3600,  # Default 1 hour
        alert_thresholds: Optional[Dict[str, int]] = None,
        output_dir: str = "reports/automated",
    ):
        """
        Initialize automated log analyzer.

        Args:
            log_sources: Dictionary of source names to log file paths
            analysis_interval: Time between analyses in seconds
            alert_thresholds: Thresholds for triggering alerts
            output_dir: Directory for storing analysis reports
        """
        self.log_sources = log_sources
        self.analysis_interval = analysis_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "error_count": 10,
            "warning_count": 50,
            "security_issues": 1,
            "performance_issues": 5,
            "tampered_entries": 1,
        }

        self.analyzer = LogAnalyzer()
        self.aggregator = LogAggregator(log_sources)
        self.report_generator = ReportGenerator(str(self.output_dir))
        self.integrity_verifier = ChainIntegrityVerifier()

        # Alert history
        self.alert_history: List[Dict[str, Any]] = []
        self.max_alert_history = 1000

        # Running state
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def analyze_and_alert(self) -> Dict[str, Any]:
        """
        Perform analysis and generate alerts if thresholds are exceeded.

        Returns:
            Analysis results with any triggered alerts
        """
        timestamp = datetime.now().isoformat()
        results: Dict[str, Any] = {
            "timestamp": timestamp,
            "analysis": None,
            "alerts": [],
            "integrity_checks": {},
            "reports_generated": [],
        }

        try:
            # Aggregate logs from all sources
            analysis = self.aggregator.aggregate_logs()
            results["analysis"] = {
                "total_entries": analysis.total_entries,
                "entries_by_level": {
                    level: len(entries)
                    for level, entries in analysis.entries_by_level.items()
                },
                "anomaly_count": len(analysis.anomalies),
            }

            # Check thresholds and generate alerts
            alerts = self._check_thresholds(analysis)
            results["alerts"] = alerts

            # Verify log integrity for each source
            for source_name, source_path in self.log_sources.items():
                source_file = Path(source_path)
                if source_file.exists():
                    integrity_result = self.integrity_verifier.verify_log_chain(
                        source_file
                    )
                    results["integrity_checks"][source_name] = integrity_result

                    # Alert if tampered entries found
                    if integrity_result.get("tampered_entries"):
                        alert = {
                            "type": "integrity_violation",
                            "severity": "critical",
                            "source": source_name,
                            "message": f"Tampered entries detected in {source_name}",
                            "details": integrity_result["tampered_entries"],
                            "timestamp": timestamp,
                        }
                        results["alerts"].append(alert)
                        self._record_alert(alert)

            # Generate reports if issues detected
            if alerts or analysis.anomalies:
                report_files = self._generate_reports(analysis, timestamp)
                results["reports_generated"] = report_files

        except Exception as e:
            error_alert = {
                "type": "analysis_error",
                "severity": "error",
                "message": f"Analysis failed: {str(e)}",
                "timestamp": timestamp,
            }
            results["alerts"].append(error_alert)
            self._record_alert(error_alert)

        return results

    def _check_thresholds(self, analysis: LogAnalysis) -> List[Dict[str, Any]]:
        """Check analysis against alert thresholds."""
        alerts = []
        timestamp = datetime.now().isoformat()

        # Count entries by level
        error_count = len(analysis.entries_by_level.get("error", []))
        warning_count = len(analysis.entries_by_level.get("warning", []))

        # Check error threshold
        if error_count >= self.alert_thresholds["error_count"]:
            alert = {
                "type": "error_threshold",
                "severity": "high",
                "message": f"Error count ({error_count}) exceeded threshold ({self.alert_thresholds['error_count']})",
                "count": error_count,
                "threshold": self.alert_thresholds["error_count"],
                "timestamp": timestamp,
            }
            alerts.append(alert)
            self._record_alert(alert)

        # Check warning threshold
        if warning_count >= self.alert_thresholds["warning_count"]:
            alert = {
                "type": "warning_threshold",
                "severity": "medium",
                "message": f"Warning count ({warning_count}) exceeded threshold ({self.alert_thresholds['warning_count']})",
                "count": warning_count,
                "threshold": self.alert_thresholds["warning_count"],
                "timestamp": timestamp,
            }
            alerts.append(alert)
            self._record_alert(alert)

        # Check security issues
        security_issues = [a for a in analysis.anomalies if a.get("type") == "security"]
        if len(security_issues) >= self.alert_thresholds["security_issues"]:
            alert = {
                "type": "security_alert",
                "severity": "critical",
                "message": f"Security issues detected: {len(security_issues)}",
                "count": len(security_issues),
                "details": security_issues[:5],  # Include first 5
                "timestamp": timestamp,
            }
            alerts.append(alert)
            self._record_alert(alert)

        # Check performance issues
        performance_issues = [
            a for a in analysis.anomalies if a.get("type") == "performance"
        ]
        if len(performance_issues) >= self.alert_thresholds["performance_issues"]:
            alert = {
                "type": "performance_alert",
                "severity": "high",
                "message": f"Performance issues detected: {len(performance_issues)}",
                "count": len(performance_issues),
                "details": performance_issues[:5],
                "timestamp": timestamp,
            }
            alerts.append(alert)
            self._record_alert(alert)

        return alerts

    def _record_alert(self, alert: Dict[str, Any]) -> None:
        """Record alert in history."""
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_alert_history:
            self.alert_history.pop(0)

    def _generate_reports(self, analysis: LogAnalysis, timestamp: str) -> List[str]:
        """Generate analysis reports."""
        report_files = []
        base_filename = f"analysis_{timestamp.replace(':', '-')}"

        # Generate summary report
        summary = self.report_generator.generate_summary_report(analysis)
        summary_file = self.output_dir / f"{base_filename}_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)
        report_files.append(str(summary_file))

        # Generate security report if security issues exist
        security_issues = [a for a in analysis.anomalies if a.get("type") == "security"]
        if security_issues:
            security = self.report_generator.generate_security_report(analysis)
            security_file = self.output_dir / f"{base_filename}_security.txt"
            with open(security_file, "w", encoding="utf-8") as f:
                f.write(security)
            report_files.append(str(security_file))

        return report_files

    def start_monitoring(self) -> None:
        """Start continuous monitoring in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()

        if apgi_logger:
            apgi_logger.info("Automated log monitoring started")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        if apgi_logger:
            apgi_logger.info("Automated log monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self.analyze_and_alert()
            except Exception as e:
                if apgi_logger:
                    apgi_logger.error(f"Monitoring loop error: {e}")

            # Wait for next interval
            time.sleep(self.analysis_interval)

    def get_alert_summary(
        self, hours: int = 24, severity_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get summary of alerts from recent time period.

        Args:
            hours: Number of hours to look back
            severity_filter: Optional list of severities to include

        Returns:
            Alert summary statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_alerts = []
        for alert in self.alert_history:
            alert_time = datetime.fromisoformat(alert["timestamp"])
            if alert_time >= cutoff_time:
                if not severity_filter or alert.get("severity") in severity_filter:
                    filtered_alerts.append(alert)

        # Group by type
        by_type: Dict[str, List[Dict]] = {}
        by_severity: Dict[str, List[Dict]] = {}

        for alert in filtered_alerts:
            alert_type = alert.get("type", "unknown")
            severity = alert.get("severity", "unknown")

            by_type.setdefault(alert_type, []).append(alert)
            by_severity.setdefault(severity, []).append(alert)

        return {
            "period_hours": hours,
            "total_alerts": len(filtered_alerts),
            "by_type": {t: len(a) for t, a in by_type.items()},
            "by_severity": {s: len(a) for s, a in by_severity.items()},
            "recent_alerts": filtered_alerts[:10],  # Last 10 alerts
        }

    def export_alert_history(self, output_file: str) -> None:
        """Export alert history to file."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "exported_at": datetime.now().isoformat(),
                    "total_alerts": len(self.alert_history),
                    "alerts": self.alert_history,
                },
                f,
                indent=2,
            )


# Convenience function for automated analysis
def run_automated_analysis(
    log_sources: Dict[str, str],
    analysis_interval: int = 3600,
    alert_thresholds: Optional[Dict[str, int]] = None,
    output_dir: str = "reports/automated",
) -> AutomatedLogAnalyzer:
    """
    Create and configure automated log analyzer.

    Args:
        log_sources: Dictionary of source names to log file paths
        analysis_interval: Time between analyses in seconds
        alert_thresholds: Thresholds for triggering alerts
        output_dir: Directory for storing reports

    Returns:
        Configured AutomatedLogAnalyzer instance
    """
    analyzer = AutomatedLogAnalyzer(
        log_sources=log_sources,
        analysis_interval=analysis_interval,
        alert_thresholds=alert_thresholds,
        output_dir=output_dir,
    )
    return analyzer


if __name__ == "__main__":
    # Example usage
    log_files = {
        "validation": "/path/to/validation.log",
        "falsification": "/path/to/falsification.log",
    }

    # Generate comprehensive report
    report = generate_comprehensive_report(log_files=log_files, output_dir="./reports")

    print("Report generated successfully")
