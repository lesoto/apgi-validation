#!/usr/bin/env python3
"""
APGI API Server with Rate Limiting
==================================

Flask-based API server for the APGI framework with rate limiting middleware.
"""

import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from slowapi import Limiter as SlowAPILimiter
from slowapi.util import get_remote_address as slowapi_get_remote_address
from slowapi.middleware import SlowAPIMiddleware

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_manager import get_config
from utils.logging_config import apgi_logger

# Initialize Flask app
app = Flask(__name__)

# Configure rate limiting with SlowAPI
limiter = SlowAPILimiter(
    app,
    key_func=slowapi_get_remote_address,
    default_limits=["100 per hour", "10 per minute"],
)

# Add SlowAPI middleware
app.wsgi_app = SlowAPIMiddleware(app.wsgi_app)

# Configure Flask-Limiter as alternative
flask_limiter = Limiter(
    app, key_func=get_remote_address, default_limits=["100 per hour", "10 per minute"]
)


@app.route("/health", methods=["GET"])
@limiter.limit("10 per minute")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "APGI API", "version": "1.0.0"})


@app.route("/config", methods=["GET"])
@flask_limiter.limit("5 per minute")
def get_configuration():
    """Get current configuration."""
    try:
        config = get_config()
        return jsonify(
            {
                "status": "success",
                "config": config.to_dict()
                if hasattr(config, "to_dict")
                else str(config),
            }
        )
    except Exception as e:
        apgi_logger.logger.error(f"Config retrieval error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/validate/<int:protocol>", methods=["POST"])
@limiter.limit("20 per hour")
def run_validation(protocol):
    """Run validation protocol."""
    try:
        # Import validation logic
        from utils.validation_pipeline_connector import ValidationPipelineConnector

        connector = ValidationPipelineConnector()
        result = connector.run_validation_with_pipeline(
            validation_protocol=protocol, use_synthetic=True
        )

        return jsonify(result)

    except Exception as e:
        apgi_logger.logger.error(f"Validation error: {e}")
        return jsonify({"status": "error", "protocol": protocol, "error": str(e)}), 500


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded."""
    return (
        jsonify(
            {
                "status": "error",
                "error": "Rate limit exceeded",
                "retry_after": e.description,
            }
        ),
        429,
    )


if __name__ == "__main__":
    # Get configuration
    config = get_config()

    # Set host and port
    host = os.environ.get("APGI_API_HOST", "0.0.0.0")
    port = int(os.environ.get("APGI_API_PORT", "5000"))

    apgi_logger.logger.info(f"Starting APGI API server on {host}:{port}")
    app.run(host=host, port=port, debug=False)
