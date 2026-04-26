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


# Export aliases for compatibility
apgi_logger = APGILogger(logger)


def log_error(message: str) -> None:
    """Log an error message."""
    logger.error(message)
