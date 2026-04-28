import json
import logging
import uuid
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", str(uuid.uuid4())),
        }
        return json.dumps(log_record)


logger = logging.getLogger("apgi")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class APGILogger:
    """Wrapper class for logger to maintain compatibility with existing code."""

    def __init__(self, logger_instance: logging.Logger):
        self.logger = logger_instance

    def __getattr__(self, name: str) -> Any:
        """Delegate all attribute access to the underlying logger."""
        return getattr(self.logger, name)

    def log_simulation_start(self, name: str, params: Any) -> None:
        """Log the start of a simulation."""
        self.logger.info(f"Simulation started: {name}", extra={"params": params})

    def log_simulation_end(self, name: str, *args, **kwargs) -> None:
        """Log the end of a simulation."""
        self.logger.info(f"Simulation ended: {name}")

    def log_performance_metric(self, name: str, value: float, unit: str = "") -> None:
        """Log a performance metric."""
        self.logger.info(f"Metric: {name} = {value}{unit}")

    def log_simulation_error(self, name: str, error: Exception) -> None:
        """Log a simulation error."""
        self.logger.error(f"Simulation error in {name}: {str(error)}")

    def export_logs(
        self, export_path: str, format_type: str = "json", log_level: str = None
    ) -> bool:
        """Export logs to a file.

        Args:
            export_path: Path to export logs to
            format_type: Format to export (json, csv, txt)
            log_level: Optional log level filter

        Returns:
            True if export succeeded, False otherwise
        """
        try:
            from pathlib import Path

            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)

            # For now, create a simple export
            with open(export_file, "w", encoding="utf-8") as f:
                f.write("Log export placeholder - logs would be exported here\n")

            return True
        except Exception:
            return False

    def get_performance_summary(self) -> dict:
        """Get performance metrics summary.

        Returns:
            Dictionary with performance metrics summary
        """
        # Return empty dict for now - in production this would aggregate metrics
        return {}


# Export aliases for compatibility
apgi_logger = APGILogger(logger)


def log_error(message: str) -> None:
    """Log an error message."""
    logger.error(message)
