#!/usr/bin/env python3
"""
Input Validation Utility
======================

Provides comprehensive input validation for user inputs across the application.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ValidationType(Enum):
    """Types of validation available."""

    REQUIRED = "required"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    PATH = "path"
    FILE_PATH = "file_path"
    DIRECTORY_PATH = "directory_path"
    EMAIL = "email"
    URL = "url"
    RANGE = "range"
    LENGTH = "length"
    PATTERN = "pattern"
    CUSTOM = "custom"


@dataclass
class ValidationRule:
    """Represents a single validation rule."""

    type: ValidationType
    params: Dict[str, Any]
    message: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    message: str
    value: Any = None


class InputValidator:
    """Comprehensive input validation system."""

    def __init__(self):
        self.custom_validators: Dict[str, Callable[[Any], ValidationResult]] = {}

    def validate(self, value: Any, rules: List[ValidationRule]) -> ValidationResult:
        """Validate a value against a list of rules."""
        for rule in rules:
            result = self._apply_rule(value, rule)
            if not result.is_valid:
                return result

        return ValidationResult(True, "Valid", value)

    def validate_field(
        self, field_name: str, value: Any, schema: Dict[str, Any]
    ) -> ValidationResult:
        """Validate a field against a schema definition."""
        rules = self._parse_schema(schema)
        result = self.validate(value, rules)

        if not result.is_valid:
            # Add field name to message if not present
            if field_name not in result.message:
                result.message = f"{field_name}: {result.message}"

        return result

    def validate_form(
        self, data: Dict[str, Any], schema: Dict[str, Dict[str, Any]]
    ) -> Dict[str, ValidationResult]:
        """Validate an entire form against a schema."""
        results = {}

        for field_name, field_schema in schema.items():
            field_value = data.get(field_name)
            results[field_name] = self.validate_field(
                field_name, field_value, field_schema
            )

        return results

    def add_custom_validator(
        self, name: str, validator: Callable[[Any], ValidationResult]
    ):
        """Add a custom validator function."""
        self.custom_validators[name] = validator

    def _apply_rule(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Apply a single validation rule."""
        if rule.type == ValidationType.REQUIRED:
            return self._validate_required(value, rule.params)
        elif rule.type == ValidationType.STRING:
            return self._validate_string(value, rule.params)
        elif rule.type == ValidationType.INTEGER:
            return self._validate_integer(value, rule.params)
        elif rule.type == ValidationType.FLOAT:
            return self._validate_float(value, rule.params)
        elif rule.type == ValidationType.BOOLEAN:
            return self._validate_boolean(value, rule.params)
        elif rule.type == ValidationType.PATH:
            return self._validate_path(value, rule.params)
        elif rule.type == ValidationType.FILE_PATH:
            return self._validate_file_path(value, rule.params)
        elif rule.type == ValidationType.DIRECTORY_PATH:
            return self._validate_directory_path(value, rule.params)
        elif rule.type == ValidationType.EMAIL:
            return self._validate_email(value, rule.params)
        elif rule.type == ValidationType.URL:
            return self._validate_url(value, rule.params)
        elif rule.type == ValidationType.RANGE:
            return self._validate_range(value, rule.params)
        elif rule.type == ValidationType.LENGTH:
            return self._validate_length(value, rule.params)
        elif rule.type == ValidationType.PATTERN:
            return self._validate_pattern(value, rule.params)
        elif rule.type == ValidationType.CUSTOM:
            return self._validate_custom(value, rule.params)
        else:
            return ValidationResult(False, f"Unknown validation type: {rule.type}")

    def _validate_required(
        self, value: Any, params: Dict[str, Any]
    ) -> ValidationResult:
        """Validate required field."""
        if value is None or (isinstance(value, str) and not value.strip()):
            message = params.get("message", "This field is required")
            return ValidationResult(False, message)
        return ValidationResult(True, "Valid")

    def _validate_string(self, value: Any, params: Dict[str, Any]) -> ValidationResult:
        """Validate string input."""
        if not isinstance(value, str):
            message = params.get("message", "Must be a string")
            return ValidationResult(False, message)

        # Check length constraints
        min_length = params.get("min_length")
        max_length = params.get("max_length")

        if min_length is not None and len(value) < min_length:
            message = params.get("message", f"Must be at least {min_length} characters")
            return ValidationResult(False, message)

        if max_length is not None and len(value) > max_length:
            message = params.get(
                "message", f"Must be no more than {max_length} characters"
            )
            return ValidationResult(False, message)

        return ValidationResult(True, "Valid")

    def _validate_integer(self, value: Any, params: Dict[str, Any]) -> ValidationResult:
        """Validate integer input."""
        try:
            int_value = int(value)
        except (ValueError, TypeError):
            message = params.get("message", "Must be an integer")
            return ValidationResult(False, message)

        # Check range constraints
        min_value = params.get("min")
        max_value = params.get("max")

        if min_value is not None and int_value < min_value:
            message = params.get("message", f"Must be at least {min_value}")
            return ValidationResult(False, message)

        if max_value is not None and int_value > max_value:
            message = params.get("message", f"Must be no more than {max_value}")
            return ValidationResult(False, message)

        return ValidationResult(True, "Valid", int_value)

    def _validate_float(self, value: Any, params: Dict[str, Any]) -> ValidationResult:
        """Validate float input."""
        try:
            float_value = float(value)
        except (ValueError, TypeError):
            message = params.get("message", "Must be a number")
            return ValidationResult(False, message)

        # Check range constraints
        min_value = params.get("min")
        max_value = params.get("max")

        if min_value is not None and float_value < min_value:
            message = params.get("message", f"Must be at least {min_value}")
            return ValidationResult(False, message)

        if max_value is not None and float_value > max_value:
            message = params.get("message", f"Must be no more than {max_value}")
            return ValidationResult(False, message)

        return ValidationResult(True, "Valid", float_value)

    def _validate_boolean(self, value: Any, params: Dict[str, Any]) -> ValidationResult:
        """Validate boolean input."""
        if isinstance(value, bool):
            return ValidationResult(True, "Valid", value)

        if isinstance(value, str):
            lower_value = value.lower().strip()
            if lower_value in ("true", "1", "yes", "on"):
                return ValidationResult(True, "Valid", True)
            elif lower_value in ("false", "0", "no", "off"):
                return ValidationResult(True, "Valid", False)

        message = params.get("message", "Must be true or false")
        return ValidationResult(False, message)

    def _validate_path(self, value: Any, params: Dict[str, Any]) -> ValidationResult:
        """Validate path input."""
        if not isinstance(params, dict):
            return ValidationResult(False, "Invalid validation parameters")

        if not isinstance(value, str):
            message = params.get("message", "Must be a string")
            return ValidationResult(False, message)

        try:
            from pathlib import Path

            path_obj = Path(value)

            # Check if path should exist
            must_exist = params.get("must_exist", False)
            if must_exist and not path_obj.exists():
                message = params.get("message", "Path does not exist")
                return ValidationResult(False, message)

            # Check if path should be absolute
            must_be_absolute = params.get("must_be_absolute", False)
            if must_be_absolute and not path_obj.is_absolute():
                message = params.get("message", "Must be an absolute path")
                return ValidationResult(False, message)

            return ValidationResult(True, "Valid", str(path_obj))

        except Exception:
            message = params.get("message", "Invalid path format")
            return ValidationResult(False, message)

    def _validate_file_path(
        self, value: Any, params: Dict[str, Any]
    ) -> ValidationResult:
        """Validate file path input."""
        if not isinstance(params, dict):
            return ValidationResult(False, "Invalid validation parameters")

        result = self._validate_path(value, params)
        if not result.is_valid:
            return result

        import os

        path_str = result.value

        # Check if file exists and is accessible (reduce TOCTOU window)
        if not os.path.exists(path_str):
            message = params.get("message", "File does not exist")
            return ValidationResult(False, message)

        if not os.path.isfile(path_str):
            message = params.get("message", "Path exists but is not a file")
            return ValidationResult(False, message)

        if not os.access(path_str, os.R_OK):
            message = params.get("message", "File exists but is not readable")
            return ValidationResult(False, message)

        return result

    def _validate_directory_path(
        self, value: Any, params: Dict[str, Any]
    ) -> ValidationResult:
        """Validate directory path input."""
        if not isinstance(params, dict):
            return ValidationResult(False, "Invalid validation parameters")

        result = self._validate_path(value, params)
        if not result.is_valid:
            return result

        path_obj = Path(result.value)
        if path_obj.exists() and not path_obj.is_dir():
            message = params.get("message", "Path exists but is not a directory")
            return ValidationResult(False, message)

        return result

    def _validate_email(self, value: Any, params: Dict[str, Any]) -> ValidationResult:
        """Validate email input."""
        if not isinstance(params, dict):
            return ValidationResult(False, "Invalid validation parameters")

        if not isinstance(value, str):
            message = params.get("message", "Must be a valid email address")
            return ValidationResult(False, message)

        # Basic email regex
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, value):
            message = params.get("message", "Must be a valid email address")
            return ValidationResult(False, message)

        return ValidationResult(True, "Valid")

    def _validate_url(self, value: Any, params: Dict[str, Any]) -> ValidationResult:
        """Validate URL input."""
        if not isinstance(value, str):
            message = params.get("message", "Must be a valid URL")
            return ValidationResult(False, message)

        # Only allow http and https protocols for security
        if not (value.startswith("http://") or value.startswith("https://")):
            message = params.get(
                "message", "Must be a valid URL starting with http:// or https://"
            )
            return ValidationResult(False, message)

        # Basic URL regex for structure validation
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        if not re.match(url_pattern, value):
            message = params.get("message", "Must be a valid URL (http:// or https://)")
            return ValidationResult(False, message)

        return ValidationResult(True, "Valid")

    def _validate_range(self, value: Any, params: Dict[str, Any]) -> ValidationResult:
        """Validate range input (like [min, max])."""
        if not isinstance(value, str):
            message = params.get("message", "Must be in format [min, max]")
            return ValidationResult(False, message)

        try:
            import json

            # Check payload size to prevent memory exhaustion
            max_json_size = 10000  # 10KB limit for JSON strings
            if len(value) > max_json_size:
                message = params.get(
                    "message",
                    f"JSON payload too large (max {max_json_size} characters)",
                )
                return ValidationResult(False, message)

            range_list = json.loads(value)
            if not isinstance(range_list, list) or len(range_list) != 2:
                raise ValueError()

            min_val, max_val = range_list
            if min_val >= max_val:
                message = params.get("message", "Minimum must be less than maximum")
                return ValidationResult(False, message)

            return ValidationResult(True, "Valid", [min_val, max_val])

        except Exception:
            message = params.get("message", "Must be in format [min, max]")
            return ValidationResult(False, message)

    def _validate_length(self, value: Any, params: Dict[str, Any]) -> ValidationResult:
        """Validate length of iterable input."""
        if not isinstance(params, dict):
            return ValidationResult(False, "Invalid validation parameters")

        try:
            length = len(value)
        except TypeError:
            message = params.get("message", "Value must have a length")
            return ValidationResult(False, message)

        min_length = params.get("min")
        max_length = params.get("max")

        if min_length is not None and length < min_length:
            message = params.get("message", f"Must have at least {min_length} items")
            return ValidationResult(False, message)

        if max_length is not None and length > max_length:
            message = params.get(
                "message", f"Must have no more than {max_length} items"
            )
            return ValidationResult(False, message)

        return ValidationResult(True, "Valid")

    def _validate_pattern(self, value: Any, params: Dict[str, Any]) -> ValidationResult:
        """Validate against a regex pattern."""
        if not isinstance(params, dict):
            return ValidationResult(False, "Invalid validation parameters")

        if not isinstance(value, str):
            message = params.get("message", "Must be a string")
            return ValidationResult(False, message)

        pattern = params.get("pattern")
        if not pattern:
            return ValidationResult(False, "Pattern not specified")

        try:
            # Add timeout to prevent ReDoS attacks
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Regex matching timed out")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(1)  # 1 second timeout

            try:
                result = re.match(pattern, value)
                signal.alarm(0)  # Cancel alarm

                if not result:
                    message = params.get("message", "Invalid format")
                    return ValidationResult(False, message)

                return ValidationResult(True, "Valid")
            except TimeoutError:
                signal.alarm(0)  # Cancel alarm
                return ValidationResult(False, "Regex matching timed out")

        except re.error:
            return ValidationResult(False, "Invalid regex pattern")

    def _validate_custom(self, value: Any, params: Dict[str, Any]) -> ValidationResult:
        """Validate using a custom validator."""
        if not isinstance(params, dict):
            return ValidationResult(False, "Invalid validation parameters")

        validator_name = params.get("validator")
        if not validator_name or validator_name not in self.custom_validators:
            return ValidationResult(False, "Custom validator not found")

        return self.custom_validators[validator_name](value)

    def _parse_schema(self, schema: Dict[str, Any]) -> List[ValidationRule]:
        """Parse a schema definition into validation rules."""
        rules = []

        for rule_type, rule_params in schema.items():
            if isinstance(rule_type, str):
                try:
                    validation_type = ValidationType(rule_type)
                    rules.append(ValidationRule(validation_type, rule_params))
                except ValueError:
                    continue  # Skip unknown validation types

        return rules


# Global validator instance
_global_validator = InputValidator()


def get_validator() -> InputValidator:
    """Get the global input validator instance."""
    return _global_validator


# Common validation schemas
COMMON_SCHEMAS = {
    "positive_integer": {
        "type": "integer",
        "min": 1,
        "message": "Must be a positive integer",
    },
    "non_negative_integer": {
        "type": "integer",
        "min": 0,
        "message": "Must be zero or positive",
    },
    "percentage": {
        "type": "float",
        "min": 0,
        "max": 100,
        "message": "Must be between 0 and 100",
    },
    "required_string": {"type": "required", "message": "This field is required"},
    "email_address": {"type": "email", "message": "Must be a valid email address"},
    "http_url": {
        "type": "url",
        "message": "Must be a valid URL starting with http:// or https://",
    },
}
