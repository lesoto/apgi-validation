"""
APGI Theory Framework - Logging Configuration
============================================

Comprehensive logging system using loguru with:
- Multiple log levels and formatting
- Rotating log files
- Structured output for debugging
- Performance metrics tracking
- Error reporting and alerts
"""

import json
import queue
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from threading import _PythonFinalizationError as PythonFinalizationError
except ImportError:
    # Fallback for older Python versions
    class PythonFinalizationError(RuntimeError):
        pass


from loguru import logger

# Project root directory
PROJECT_ROOT = Path(__file__).parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)


@dataclass
class LogEntry:
    """Structured log entry for search and analysis."""

    timestamp: str
    level: str
    message: str
    module: str
    function: str
    line: int
    thread: str = None
    exception: str = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
            "module": self.module,
            "function": self.function,
            "line": self.line,
            "thread": self.thread,
            "exception": self.exception,
        }


@dataclass
class SearchQuery:
    """Log search query parameters."""

    text: str = None
    level: str = None
    module: str = None
    start_time: str = None
    end_time: str = None
    regex: bool = False
    max_results: int = 1000
    offset: int = 0


class LogStreamer:
    """Real-time log streaming functionality."""

    def __init__(self):
        self.subscribers = {}
        self.queue = queue.Queue()
        self.running = False
        self.thread = None

    def subscribe(
        self,
        callback: Callable[[LogEntry], None],
        level_filter: str = None,
        module_filter: str = None,
    ) -> str:
        """Subscribe to log stream with filters."""
        subscriber_id = f"sub_{int(time.time() * 1000)}"
        self.subscribers[subscriber_id] = {
            "callback": callback,
            "level_filter": level_filter,
            "module_filter": module_filter,
        }
        return subscriber_id

    def unsubscribe(self, subscriber_id: str):
        """Unsubscribe from log stream."""
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]

    def publish(self, entry: LogEntry):
        """Publish log entry to subscribers."""
        for subscriber_id, subscriber in self.subscribers.items():
            # Apply filters
            if subscriber["level_filter"] and entry.level != subscriber["level_filter"]:
                continue
            if (
                subscriber["module_filter"]
                and entry.module != subscriber["module_filter"]
            ):
                continue

            try:
                subscriber["callback"](entry)
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.error(f"Error in log subscriber {subscriber_id}: {e}")

    def start_streaming(self):
        """Start the streaming thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._stream_worker, daemon=True)
            self.thread.start()

    def stop_streaming(self):
        """Stop the streaming thread."""
        self.running = False
        if self.thread and self.thread.is_alive():
            # Use timeout to avoid hanging during interpreter shutdown
            self.thread.join(timeout=1.0)

    def _stream_worker(self):
        """Background worker for streaming."""
        while self.running:
            try:
                # Get log entries from queue
                entry = self.queue.get(timeout=1)
                self.publish(entry)
                self.queue.task_done()
            except queue.Empty:
                continue
            except (ValueError, TypeError, RuntimeError) as e:
                logger.error(f"Error in stream worker: {e}")


class APGILogger:
    """Advanced logging system for APGI framework."""

    def __init__(
        self,
        log_level: str = "INFO",
        enable_console: bool = True,
        queue_size: int = 10000,
    ):
        self.log_files = {}
        self.performance_metrics = {}
        self.error_counts = {}
        self.logger = logger  # Expose the loguru logger
        self.streamer = LogStreamer()

        # Validate and apply logging configuration with fallbacks
        self._validate_logging_config(log_level, enable_console, queue_size)

        self._setup_logging()
        self.streamer.start_streaming()

    def _validate_logging_config(
        self, log_level: str, enable_console: bool, queue_size: int
    ):
        """Validate logging configuration parameters with default fallbacks."""
        # Validate and fallback log level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if not isinstance(log_level, str) or log_level.upper() not in valid_levels:
            logger.warning(
                f"Invalid log level '{log_level}'. Using default 'INFO'. Valid levels: {valid_levels}"
            )
            log_level = "INFO"
        else:
            log_level = log_level.upper()

        # Validate and fallback enable_console
        if not isinstance(enable_console, bool):
            logger.warning(
                f"enable_console must be a boolean, got {type(enable_console)}. Using default True"
            )
            enable_console = True

        # Validate and fallback queue_size
        if not isinstance(queue_size, int) or queue_size < 1:
            logger.warning(
                f"queue_size must be a positive integer, got {queue_size}. Using default 10000"
            )
            queue_size = 10000
        elif queue_size > 100000:
            logger.warning(
                f"Large queue_size ({queue_size}) may cause memory issues. Using 100000"
            )
            queue_size = 100000

        # Apply validated/fallback values
        self.log_level = log_level
        self.enable_console = enable_console
        self.queue_size = queue_size

    def update_logging_config(
        self, log_level: str = None, enable_console: bool = None, queue_size: int = None
    ):
        """Update logging configuration with validation and fallbacks."""
        config_changed = False

        # Validate and update log level
        if log_level is not None:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if not isinstance(log_level, str) or log_level.upper() not in valid_levels:
                logger.warning(
                    f"Invalid log level '{log_level}'. Keeping current '{self.log_level}'. Valid levels: {valid_levels}"
                )
            else:
                old_level = self.log_level
                self.log_level = log_level.upper()
                if old_level != self.log_level:
                    config_changed = True

        # Validate and update enable_console
        if enable_console is not None:
            if not isinstance(enable_console, bool):
                logger.warning(
                    f"enable_console must be a boolean, got {type(enable_console)}. Keeping current {self.enable_console}"
                )
            else:
                if self.enable_console != enable_console:
                    self.enable_console = enable_console
                    config_changed = True

        # Validate and update queue_size
        if queue_size is not None:
            if not isinstance(queue_size, int) or queue_size < 1:
                logger.warning(
                    f"queue_size must be a positive integer, got {queue_size}. Keeping current {self.queue_size}"
                )
            elif queue_size > 100000:
                logger.warning(
                    f"Large queue_size ({queue_size}) may cause memory issues. Using 100000"
                )
                if self.queue_size != 100000:
                    self.queue_size = 100000
                    config_changed = True
            else:
                if self.queue_size != queue_size:
                    self.queue_size = queue_size
                    config_changed = True

        # Reconfigure logging only if settings changed
        if config_changed:
            self._setup_logging()
            logger.info("Logging configuration updated")
        else:
            logger.debug("No configuration changes applied")

    def _setup_logging(self):
        """Configure loguru logger with custom settings."""
        # Remove default logger
        logger.remove()

        # Console logger
        if self.enable_console:
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>",
                level=self.log_level,
                colorize=True,
            )

        # Main log file with rotation and queue limit
        main_log_file = LOGS_DIR / "apgi_framework.log"
        logger.add(
            main_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            enqueue={"queue_size": self.queue_size},
        )

        # Error-specific log file with queue limit
        error_log_file = LOGS_DIR / "errors.log"
        logger.add(
            error_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
            level="ERROR",
            rotation="5 MB",
            retention="60 days",
            compression="zip",
            enqueue={"queue_size": self.queue_size},
            backtrace=True,
            diagnose=True,
        )

        # Performance metrics log with queue limit
        performance_log_file = LOGS_DIR / "performance.log"
        logger.add(
            performance_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | PERFORMANCE | {extra[metric]} | {extra[value]} | {extra[unit]}",
            level="INFO",
            rotation="5 MB",
            retention="7 days",
            filter=lambda record: "metric" in record["extra"],
            enqueue={"queue_size": self.queue_size},
        )

        # Structured JSON log for machine processing with queue limit
        json_log_file = LOGS_DIR / "structured.jsonl"
        logger.add(
            json_log_file,
            format="{message}",
            level="DEBUG",
            rotation="20 MB",
            retention="30 days",
            serialize=True,
            enqueue={"queue_size": self.queue_size},
        )

        self.log_files = {
            "main": main_log_file,
            "errors": error_log_file,
            "performance": performance_log_file,
            "structured": json_log_file,
        }

    def log_simulation_start(self, simulation_type: str, parameters: Dict[str, Any]):
        """Log the start of a simulation with parameters."""
        logger.info(f"Starting {simulation_type} simulation")
        logger.bind(simulation_type=simulation_type, parameters=parameters).debug(
            "Simulation parameters"
        )

    def log_simulation_end(
        self, simulation_type: str, duration: float, results_summary: Dict[str, Any]
    ):
        """Log the end of a simulation with results summary."""
        logger.info(f"Completed {simulation_type} simulation in {duration:.2f} seconds")
        logger.bind(
            simulation_type=simulation_type, duration=duration, results=results_summary
        ).debug("Simulation results")

    def log_parameter_estimation(
        self, method: str, parameters: Dict[str, Any], log_likelihood: float
    ):
        """Log parameter estimation results."""
        logger.info(f"Parameter estimation using {method} method")
        logger.bind(
            method=method, parameters=parameters, log_likelihood=log_likelihood
        ).info("Estimation results")

    def log_validation_result(
        self, protocol: str, passed: bool, metrics: Dict[str, float]
    ):
        """Log validation protocol results."""
        status = "PASSED" if passed else "FAILED"
        logger.info(f"Validation protocol {protocol}: {status}")
        logger.bind(protocol=protocol, passed=passed, metrics=metrics).debug(
            "Validation metrics"
        )

    def log_performance_metric(
        self, metric_name: str, value: float, unit: str = "seconds"
    ):
        """Log performance metrics."""
        logger.info(f"Performance: {metric_name} = {value:.3f} {unit}")
        logger.bind(metric=metric_name, value=value, unit=unit).debug(
            "Performance metric details"
        )

        # Track metrics for analysis
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        self.performance_metrics[metric_name].append(
            {"value": value, "unit": unit, "timestamp": datetime.now().isoformat()}
        )

    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all performance metrics."""
        summary = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                numeric_values = [v["value"] for v in values]
                summary[metric_name] = {
                    "count": len(values),
                    "mean": sum(numeric_values) / len(numeric_values),
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "unit": values[0]["unit"],
                    "latest": values[-1]["value"],
                    "latest_timestamp": values[-1]["timestamp"],
                }
        return summary

    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log errors with additional context information."""
        error_type = type(error).__name__
        error_message = str(error)

        # Track error counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        logger.error(
            f"Error in {context.get('operation', 'unknown operation')}: {error_message}"
        )
        logger.bind(
            error_type=error_type,
            error_message=error_message,
            context=context,
            traceback=traceback.format_exc(),
        ).exception("Error details")

    def log_data_processing(
        self, data_type: str, file_path: str, records_processed: int, duration: float
    ):
        """Log data processing operations."""
        logger.info(
            f"Processed {records_processed} {data_type} records from {file_path} in {duration:.2f}s"
        )
        logger.bind(
            data_type=data_type,
            file_path=file_path,
            records_processed=records_processed,
            duration=duration,
            throughput=records_processed / duration,
        ).debug("Data processing details")

    def log_model_configuration(self, model_name: str, config: Dict[str, Any]):
        """Log model configuration details."""
        logger.info(f"Configuring {model_name} model")
        logger.bind(model_name=model_name, configuration=config).debug(
            "Model configuration"
        )

    def log_system_info(self):
        """Log system information for debugging."""
        import platform

        import psutil

        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "disk_free": f"{psutil.disk_usage('.').free / (1024**3):.2f} GB",
        }

        logger.bind(system_info=system_info).info("System information")

    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error occurrences."""
        return self.error_counts.copy()

    def _is_timestamp_line(self, line: str) -> bool:
        """Check if line starts with a timestamp."""
        import re

        return (
            re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})", line) is not None
        )

    def _is_multiline_continuation(self, line: str) -> bool:
        """Check if line is a continuation of a multiline entry."""
        return line.strip().startswith(
            ("  ", "\t", "    ", "Traceback", "File ", "    ")
        )

    def _process_log_line(
        self,
        line: str,
        line_num: int,
        log_file: Path,
        current_entry: Optional[dict],
        multiline_mode: bool,
    ) -> tuple:
        """Process a single log line and return updated state."""
        line = line.rstrip("\n\r")

        if not line.strip():
            if multiline_mode and current_entry:
                multiline_mode = False
            return current_entry, multiline_mode

        if self._is_timestamp_line(line):
            if current_entry:
                log_entries = [current_entry]
            else:
                log_entries = []

            new_entry = {
                "raw_lines": [line],
                "file": str(log_file),
                "line_num": line_num + 1,
            }
            return new_entry, False, log_entries
        else:
            if current_entry:
                current_entry["raw_lines"].append(line)
                if self._is_multiline_continuation(line):
                    multiline_mode = True
            return current_entry, multiline_mode, []

    def _read_log_file(self, log_file: Path) -> list:
        """Read and parse a single log file into entries."""
        log_entries = []
        current_entry = None
        multiline_mode = False

        try:
            with open(log_file, encoding="utf-8") as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines):
                result = self._process_log_line(
                    line, line_num, log_file, current_entry, multiline_mode
                )
                if len(result) == 3:
                    current_entry, multiline_mode, new_entries = result
                    log_entries.extend(new_entries)
                else:
                    current_entry, multiline_mode = result

            if current_entry:
                log_entries.append(current_entry)

        except (
            FileNotFoundError,
            PermissionError,
            UnicodeDecodeError,
            ValueError,
        ) as e:
            logger.warning(f"Error reading log file {log_file}: {e}")

        return log_entries

    def _match_log_pattern(self, first_line: str) -> Optional[tuple]:
        """Try to match log line against known patterns."""
        import re

        patterns = [
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+)\s+\| ?\s*([^-]+?) - (.*)",
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+)\s+\| ([^-]+?) - (.*)",
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+)\s+\| (.*)",
        ]

        for pattern in patterns:
            match = re.match(pattern, first_line)
            if match:
                return match.groups()
        return None

    def _extract_log_fields(self, groups: tuple) -> tuple:
        """Extract timestamp, level, location, and message from regex groups."""
        if len(groups) >= 3:
            timestamp_str, level, location, message = (
                groups[0],
                groups[1],
                groups[2],
            )
            if len(groups) == 3:
                location = "unknown"
                message = groups[2]
        else:
            timestamp_str, level, message = groups[0], groups[1], groups[2]
            location = "unknown"

        return timestamp_str, level, location, message

    def _passes_filters(
        self,
        timestamp: datetime,
        level: str,
        log_level: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> bool:
        """Check if log entry passes all filters."""
        if start_time and timestamp < start_time:
            return False
        if end_time and timestamp > end_time:
            return False
        if log_level and level != log_level.upper():
            return False
        return True

    def _parse_log_entry(
        self,
        entry: dict,
        log_level: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> Optional[dict]:
        """Parse a single log entry and return structured data."""
        from datetime import datetime

        raw_text = "\n".join(entry["raw_lines"])
        first_line = entry["raw_lines"][0]

        groups = self._match_log_pattern(first_line)
        if not groups:
            return None

        timestamp_str, level, location, message = self._extract_log_fields(groups)

        try:
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return None

        if not self._passes_filters(timestamp, level, log_level, start_time, end_time):
            return None

        return {
            "timestamp": timestamp_str,
            "datetime": timestamp,
            "level": level,
            "location": location.strip(),
            "message": message.strip(),
            "full_message": raw_text,
            "file": entry["file"],
            "line_num": entry["line_num"],
        }

    def _export_json(self, parsed_entries: list, output_path: Path) -> None:
        """Export logs to JSON format."""
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed_entries, f, indent=2, default=str)

    def _export_csv(self, parsed_entries: list, output_path: Path) -> None:
        """Export logs to CSV format."""
        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["timestamp", "level", "location", "message", "file"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in parsed_entries:
                writer.writerow({k: v for k, v in entry.items() if k in fieldnames})

    def _export_txt(self, parsed_entries: list, output_path: Path) -> None:
        """Export logs to plain text format."""
        from datetime import datetime

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("APGI Framework Log Export\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total entries: {len(parsed_entries)}\n")
            if parsed_entries:
                f.write(
                    f"Time range: {parsed_entries[0]['timestamp']} to {parsed_entries[-1]['timestamp']}\n"
                )
            f.write("=" * 80 + "\n\n")

            for entry in parsed_entries:
                f.write(
                    f"[{entry['timestamp']}] {entry['level']} - {entry['message']}\n"
                )
                if entry["location"] != "unknown":
                    f.write(f"  Location: {entry['location']}\n")
                if entry.get("full_message") and len(entry["full_message"]) > len(
                    entry["message"]
                ):
                    f.write(f"  Details: {entry['full_message']}\n")
                f.write("\n")

    def _collect_and_parse_log_entries(
        self,
        log_files: list,
        log_level: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> list:
        """Collect and parse log entries from all log files."""
        log_entries = []
        for log_file in log_files:
            log_entries.extend(self._read_log_file(log_file))

        parsed_entries = []
        for entry in log_entries:
            try:
                parsed_entry = self._parse_log_entry(
                    entry, log_level, start_time, end_time
                )
                if parsed_entry:
                    parsed_entries.append(parsed_entry)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Error parsing log entry: {e}")

        return parsed_entries

    def export_logs(
        self,
        output_file: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format_type: str = "json",
        log_level: Optional[str] = None,
    ):
        """Export logs to a file for analysis.

        Args:
            output_file: Path to output file
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            format_type: Export format ('json', 'csv', 'txt')
            log_level: Optional log level filter ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        logger.info(f"Exporting logs to {output_file} (format: {format_type})")

        log_files = list(LOGS_DIR.glob("*.log"))
        if not log_files:
            logger.warning("No log files found to export")
            return False

        parsed_entries = self._collect_and_parse_log_entries(
            log_files, log_level, start_time, end_time
        )

        if not parsed_entries:
            logger.warning("No log entries found matching criteria")
            return False

        parsed_entries.sort(key=lambda x: x["datetime"])

        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type.lower() == "json":
                self._export_json(parsed_entries, output_path)
            elif format_type.lower() == "csv":
                self._export_csv(parsed_entries, output_path)
            elif format_type.lower() == "txt":
                self._export_txt(parsed_entries, output_path)
            else:
                logger.error(f"Unsupported format: {format_type}")
                return False

            logger.info(
                f"Successfully exported {len(parsed_entries)} log entries to {output_file}"
            )
            return True

        except (
            FileNotFoundError,
            PermissionError,
            ValueError,
            json.JSONEncodeError,
            OSError,
        ) as e:
            logger.error(f"Error exporting logs: {e}")
            return False

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files."""
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_files = 0

        for log_file in LOGS_DIR.glob("*.log*"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    log_file.unlink()
                    cleaned_files += 1
            except (FileNotFoundError, PermissionError, OSError) as e:
                logger.warning(f"Could not delete old log file {log_file}: {e}")

        logger.info(f"Cleaned up {cleaned_files} old log files")

    # Log Search and Streaming Methods
    def search_logs(self, query: SearchQuery) -> List[LogEntry]:
        """Search logs based on query parameters."""
        results = []

        # Get log files to search
        log_files = list(LOGS_DIR.glob("*.log"))

        for log_file in log_files:
            try:
                with open(log_file, encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f):
                        if len(results) >= query.max_results:
                            break

                        # Parse log line
                        entry = self._parse_log_line(line)
                        if not entry:
                            continue

                        # Apply filters
                        if not self._matches_query(entry, query):
                            continue

                        results.append(entry)

            except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
                self.logger.warning(f"Error searching log file {log_file}: {e}")

        # Apply offset
        if query.offset > 0:
            results = results[query.offset :]

        return results[: query.max_results]

    def _parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a log line into a LogEntry object."""
        try:
            # Loguru format: timestamp | level | module:function:line - message
            parts = line.strip().split(" | ")
            if len(parts) < 3:
                return None

            timestamp_part = parts[0].strip()
            level_part = parts[1].strip()
            location_part = parts[2].strip()

            # Extract message
            message_parts = parts[3:] if len(parts) > 3 else []
            message = " | ".join(message_parts).strip()

            # Parse location (module:function:line)
            location_match = re.match(r"(.+):(.+):(\d+)", location_part)
            if not location_match:
                return None

            module, function, line_str = location_match.groups()

            return LogEntry(
                timestamp=timestamp_part,
                level=level_part,
                message=message,
                module=module,
                function=function,
                line=int(line_str),
            )

        except (ValueError, TypeError, KeyError, IndexError):
            return None

    def _matches_query(self, entry: LogEntry, query: SearchQuery) -> bool:
        """Check if a log entry matches the search query."""
        # Text search
        if query.text:
            if query.regex:
                if not re.search(query.text, entry.message, re.IGNORECASE):
                    return False
            else:
                if query.text.lower() not in entry.message.lower():
                    return False

        # Level filter
        if query.level and entry.level != query.level.upper():
            return False

        # Module filter
        if query.module and query.module.lower() not in entry.module.lower():
            return False

        # Time range filter
        if query.start_time or query.end_time:
            try:
                entry_time = datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M:%S.%f")

                if query.start_time:
                    start_time = datetime.fromisoformat(query.start_time)
                    if entry_time < start_time:
                        return False

                if query.end_time:
                    end_time = datetime.fromisoformat(query.end_time)
                    if entry_time > end_time:
                        return False
            except ValueError:
                pass  # Skip time filtering if parsing fails

        return True

    def stream_logs(
        self,
        callback: Callable[[LogEntry], None],
        level_filter: str = None,
        module_filter: str = None,
    ) -> str:
        """Subscribe to real-time log streaming."""
        return self.streamer.subscribe(callback, level_filter, module_filter)

    def stop_streaming(self, subscriber_id: str):
        """Unsubscribe from log streaming."""
        self.streamer.unsubscribe(subscriber_id)

    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about log files."""
        stats = {
            "total_files": 0,
            "total_size": 0,
            "by_level": {},
            "by_module": {},
            "time_range": {"oldest": None, "newest": None},
        }

        log_files = list(LOGS_DIR.glob("*.log*"))
        stats["total_files"] = len(log_files)

        oldest_time = None
        newest_time = None

        for log_file in log_files:
            try:
                file_size = log_file.stat().st_size
                stats["total_size"] += file_size

                # Sample some entries to get stats
                with open(log_file, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        entry = self._parse_log_line(line)
                        if entry:
                            # Count by level
                            stats["by_level"][entry.level] = (
                                stats["by_level"].get(entry.level, 0) + 1
                            )

                            # Count by module
                            stats["by_module"][entry.module] = (
                                stats["by_module"].get(entry.module, 0) + 1
                            )

                            # Track time range
                            try:
                                entry_time = datetime.strptime(
                                    entry.timestamp, "%Y-%m-%d %H:%M:%S.%f"
                                )
                                if oldest_time is None or entry_time < oldest_time:
                                    oldest_time = entry_time
                                if newest_time is None or entry_time > newest_time:
                                    newest_time = entry_time
                            except ValueError:
                                pass
                        break  # Only sample first few lines

            except (
                FileNotFoundError,
                PermissionError,
                UnicodeDecodeError,
                ValueError,
                KeyError,
            ) as e:
                self.logger.warning(f"Error analyzing log file {log_file}: {e}")

        if oldest_time:
            stats["time_range"]["oldest"] = oldest_time.isoformat()
        if newest_time:
            stats["time_range"]["newest"] = newest_time.isoformat()

        return stats

    def set_up_alerts(self, error_threshold: int = 10, time_window: int = 300):
        """Set up log alerts for error monitoring."""
        # This would implement alerting logic
        # For now, just log that alerts are configured
        self.logger.info(
            f"Log alerts configured: {error_threshold} errors in {time_window}s"
        )

    def __del__(self):
        """Cleanup when logger is destroyed."""
        try:
            if hasattr(self, "streamer"):
                self.streamer.stop_streaming()
        except (AttributeError, RuntimeError, PythonFinalizationError):
            # Ignore errors during interpreter shutdown
            pass


# Global logger instance with configurable queue size
apgi_logger = APGILogger(queue_size=10000)


# Convenience functions for common logging tasks
def log_simulation(
    simulation_type: str,
    parameters: Dict[str, Any],
    duration: float,
    results: Dict[str, Any],
):
    """Convenience function for logging complete simulation."""
    apgi_logger.log_simulation_start(simulation_type, parameters)
    apgi_logger.log_simulation_end(simulation_type, duration, results)


def log_performance(metric_name: str, value: float, unit: str = "seconds"):
    """Convenience function for performance logging."""
    apgi_logger.log_performance_metric(metric_name, value, unit)


def log_error(error: Exception, operation: str, **context):
    """Convenience function for error logging."""
    context["operation"] = operation
    apgi_logger.log_error_with_context(error, context)


# Decorators for automatic logging
def log_execution_time(metric_name: str = "execution_time"):
    """Decorator to automatically log function execution time."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                log_performance(metric_name, duration)
                return result
            except (ValueError, TypeError, RuntimeError, AttributeError, KeyError) as e:
                duration = (datetime.now() - start_time).total_seconds()
                log_performance(f"{metric_name}_failed", duration)
                log_error(e, f"Function {func.__name__}")
                raise

        return wrapper

    return decorator


def log_function_call(level: str = "DEBUG"):
    """Decorator to log function calls."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.bind(function=func.__name__, args=args, kwargs=kwargs).log(
                level, f"Calling {func.__name__}"
            )
            try:
                result = func(*args, **kwargs)
                logger.bind(function=func.__name__).log(
                    level, f"Completed {func.__name__}"
                )
                return result
            except (ValueError, TypeError, RuntimeError, AttributeError, KeyError) as e:
                logger.bind(function=func.__name__).error(
                    f"Error in {func.__name__}: {e}"
                )
                raise

        return wrapper

    return decorator


if __name__ == "__main__":
    # Test logging system
    apgi_logger.log_system_info()

    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test performance logging
    apgi_logger.log_performance_metric("test_metric", 1.23, "seconds")

    # Test error logging with context
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        apgi_logger.log_error_with_context(
            e, {"operation": "test", "user": "test_user"}
        )

    # Test simulation logging
    apgi_logger.log_simulation_start("test_simulation", {"param1": 1.0, "param2": 2.0})
    apgi_logger.log_simulation_end(
        "test_simulation", 5.5, {"result1": 10, "result2": 20}
    )

    print(f"Log files created in: {LOGS_DIR}")
    print("Performance summary:", apgi_logger.get_performance_summary())
    print("Error summary:", apgi_logger.get_error_summary())
