#!/usr/bin/env python3
"""
Schema Version Manager
=====================
Provides versioned backward-compatibility strategy with schema evolution contracts.
Handles configuration schema migration, data format compatibility, and API versioning.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import semver
import yaml

logger = logging.getLogger("schema_version_manager")


class CompatibilityLevel(Enum):
    """Compatibility level for schema changes."""

    BACKWARD_COMPATIBLE = "backward_compatible"  # Old clients can work with new schema
    FORWARD_COMPATIBLE = "forward_compatible"  # New clients can work with old schema
    BREAKING = "breaking"  # Incompatible changes require migration


class SchemaVersion:
    """Represents a schema version with compatibility information."""

    def __init__(
        self,
        version: str,
        compatibility: CompatibilityLevel,
        migration_path: Optional[str] = None,
    ):
        self.version = version
        self.compatibility = compatibility
        self.migration_path = migration_path

    def __str__(self) -> str:
        return f"v{self.version} ({self.compatibility.value})"


class MigrationContract:
    """Defines migration rules between schema versions."""

    def __init__(
        self,
        from_version: str,
        to_version: str,
        migration_func: Callable,
        description: str = "",
    ):
        self.from_version = from_version
        self.to_version = to_version
        self.migration_func = migration_func
        self.description = description


class SchemaVersionManager:
    """Manages schema versioning and migration contracts."""

    def __init__(self, schema_registry_path: Optional[Path] = None):
        self.current_version = "1.3.0"
        self.min_supported_version = "1.0.0"
        self.schema_registry_path = (
            schema_registry_path or Path(__file__).parent.parent / "config"
        )
        self.migrations: Dict[Tuple[str, str], MigrationContract] = {}
        self._register_default_migrations()

    def _register_default_migrations(self) -> None:
        """Register default migration contracts for known versions."""

        # Migration 1.0 -> 1.1: Add new model parameters
        def migrate_1_0_to_1_1(config: Dict[str, Any]) -> Dict[str, Any]:
            """Migrate config from v1.0 to v1.1."""
            if "model" not in config:
                config["model"] = {}

            # Add new parameters with defaults
            model = config["model"]
            model.setdefault("tau_M", 0.1)  # New somatic marker time constant
            model.setdefault("gamma_M", 0.5)  # New metabolic cost weight
            model.setdefault("gamma_A", 0.3)  # New arousal modulation weight
            model.setdefault("beta", 0.2)  # New somatic bias weight

            return config

        self.register_migration(
            "1.0.0",
            "1.1.0",
            migrate_1_0_to_1_1,
            "Add somatic marker parameters (tau_M, gamma_M, gamma_A, beta)",
        )

        # Migration 1.1 -> 1.2: Add falsification thresholds
        def migrate_1_1_to_1_2(config: Dict[str, Any]) -> Dict[str, Any]:
            """Migrate config from v1.1 to v1.2."""
            if "falsification" not in config:
                config["falsification"] = {}

            # Add falsification protocol thresholds
            falsification = config["falsification"]
            falsification.setdefault("cumulative_reward_advantage_threshold", 15.0)
            falsification.setdefault("cohens_d_threshold", 0.8)
            falsification.setdefault("significance_level", 0.05)
            falsification.setdefault("threshold_reduction_min", 20.0)

            return config

        self.register_migration(
            "1.1.0",
            "1.2.0",
            migrate_1_1_to_1_2,
            "Add falsification protocol thresholds",
        )

        # Migration 1.2 -> 1.3: Add PAC bands configuration
        def migrate_1_2_to_1_3(config: Dict[str, Any]) -> Dict[str, Any]:
            """Migrate config from v1.2 to v1.3."""
            if "pac_bands" not in config:
                config["pac_bands"] = {}

            # Add default PAC band configurations
            pac_bands = config["pac_bands"]
            pac_bands.setdefault(
                "L1_L2",
                {
                    "phase": [4.0, 8.0],  # Theta band
                    "amplitude": [30.0, 100.0],  # Gamma band
                    "description": "Level 1-2 coupling: theta-gamma PAC",
                },
            )
            pac_bands.setdefault(
                "L2_L3",
                {
                    "phase": [1.0, 4.0],  # Delta band
                    "amplitude": [4.0, 8.0],  # Theta band
                    "description": "Level 2-3 coupling: delta-theta PAC",
                },
            )

            return config

        self.register_migration(
            "1.2.0",
            "1.3.0",
            migrate_1_2_to_1_3,
            "Add phase-amplitude coupling band configurations",
        )

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migration_func: Callable,
        description: str = "",
    ) -> None:
        """Register a migration contract between two versions."""
        key = (from_version, to_version)
        self.migrations[key] = MigrationContract(
            from_version, to_version, migration_func, description
        )
        logger.info(f"Registered migration: {from_version} -> {to_version}")

    def validate_version(self, version: str) -> bool:
        """Check if a version is supported."""
        try:
            parsed = semver.Version.parse(version)
            min_parsed = semver.Version.parse(self.min_supported_version)
            current_parsed = semver.Version.parse(self.current_version)

            return min_parsed <= parsed <= current_parsed
        except ValueError:
            return False

    def get_compatibility_level(
        self, from_version: str, to_version: str
    ) -> CompatibilityLevel:
        """Determine compatibility level between versions."""
        try:
            from_v = semver.Version.parse(from_version)
            to_v = semver.Version.parse(to_version)

            if from_v.major != to_v.major:
                return CompatibilityLevel.BREAKING
            elif from_v.minor != to_v.minor:
                return CompatibilityLevel.BACKWARD_COMPATIBLE
            else:
                return CompatibilityLevel.FORWARD_COMPATIBLE
        except ValueError:
            return CompatibilityLevel.BREAKING

    def migrate_config(
        self, config: Dict[str, Any], target_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Migrate a configuration to the target version."""
        source_version = config.get("version", "1.0.0")
        target_version = target_version or self.current_version

        if source_version == target_version:
            return config

        if not self.validate_version(source_version):
            raise ValueError(f"Unsupported source version: {source_version}")

        if not self.validate_version(target_version):
            raise ValueError(f"Unsupported target version: {target_version}")

        # Find migration path
        migration_path = self._find_migration_path(source_version, target_version)
        if not migration_path:
            raise ValueError(
                f"No migration path from {source_version} to {target_version}"
            )

        # Apply migrations sequentially
        current_config = config.copy()
        current_version = source_version

        for next_version in migration_path[1:]:  # Skip first element (source version)
            key = (current_version, next_version)
            if key not in self.migrations:
                raise ValueError(
                    f"No migration defined for {current_version} -> {next_version}"
                )

            migration = self.migrations[key]
            logger.info(f"Applying migration: {current_version} -> {next_version}")
            current_config = migration.migration_func(current_config)
            current_config["version"] = next_version
            current_version = next_version

        logger.info(
            f"Successfully migrated config from {source_version} to {target_version}"
        )
        return current_config

    def _find_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Find a migration path from source to target version."""
        try:
            from_v = semver.Version.parse(from_version)
            to_v = semver.Version.parse(to_version)
        except ValueError:
            return []

        if from_v == to_v:
            return [from_version]

        # Simple path finding for sequential versions
        path = [from_version]
        current = from_v

        while current < to_v:
            # Try to increment patch version first
            next_patch = semver.Version(current.major, current.minor, current.patch + 1)
            if self._has_migration(str(current), str(next_patch)):
                path.append(str(next_patch))
                current = next_patch
                continue

            # Try to increment minor version
            next_minor = semver.Version(current.major, current.minor + 1, 0)
            if self._has_migration(str(current), str(next_minor)):
                path.append(str(next_minor))
                current = next_minor
                continue

            # Try to increment major version
            next_major = semver.Version(current.major + 1, 0, 0)
            if self._has_migration(str(current), str(next_major)):
                path.append(str(next_major))
                current = next_major
                continue

            # No direct migration found
            break

        if str(current) == to_version:
            return path

        return []

    def _has_migration(self, from_version: str, to_version: str) -> bool:
        """Check if a migration exists between two versions."""
        return (from_version, to_version) in self.migrations

    def validate_config_schema(
        self, config: Dict[str, Any], schema_version: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """Validate a configuration against its schema version."""
        version = config.get("version", "1.0.0")
        schema_version = schema_version or version

        if not self.validate_version(version):
            return False, [f"Unsupported configuration version: {version}"]

        # Load appropriate schema
        schema_path = self.schema_registry_path / f"config_schema_v{version}.json"
        if not schema_path.exists():
            # Fallback to current schema
            schema_path = self.schema_registry_path / "config_schema.json"

        try:
            with open(schema_path, "r") as f:
                schema = json.load(f)

            # Basic validation (would use jsonschema in production)
            errors: List[str] = []
            self._validate_recursive(config, schema, "", errors)

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Schema validation error: {e}"]

    def _validate_recursive(
        self, data: Any, schema: Dict[str, Any], path: str, errors: List[str]
    ) -> None:
        """Recursively validate data against schema."""
        if not isinstance(schema, dict) or "type" not in schema:
            return

        expected_type = schema["type"]
        current_path = (
            f"{path}.{schema.get('title', '')}" if path else schema.get("title", "")
        )

        if expected_type == "object":
            if not isinstance(data, dict):
                errors.append(
                    f"Expected object at {current_path}, got {type(data).__name__}"
                )
                return

            properties = schema.get("properties", {})
            for prop_name, prop_schema in properties.items():
                if prop_name in data:
                    self._validate_recursive(
                        data[prop_name], prop_schema, current_path, errors
                    )

        elif expected_type == "array":
            if not isinstance(data, list):
                errors.append(
                    f"Expected array at {current_path}, got {type(data).__name__}"
                )
                return

        elif expected_type == "string":
            if not isinstance(data, str):
                errors.append(
                    f"Expected string at {current_path}, got {type(data).__name__}"
                )

        elif expected_type == "number":
            if not isinstance(data, (int, float)):
                errors.append(
                    f"Expected number at {current_path}, got {type(data).__name__}"
                )
            else:
                # Check constraints
                if "minimum" in schema and data < schema["minimum"]:
                    errors.append(
                        f"Value at {current_path} below minimum: {data} < {schema['minimum']}"
                    )
                if "maximum" in schema and data > schema["maximum"]:
                    errors.append(
                        f"Value at {current_path} above maximum: {data} > {schema['maximum']}"
                    )

        elif expected_type == "boolean":
            if not isinstance(data, bool):
                errors.append(
                    f"Expected boolean at {current_path}, got {type(data).__name__}"
                )

        elif expected_type == "integer":
            if not isinstance(data, int):
                errors.append(
                    f"Expected integer at {current_path}, got {type(data).__name__}"
                )


# Global instance
_schema_manager: Optional[SchemaVersionManager] = None


def get_schema_manager() -> SchemaVersionManager:
    """Get the global schema version manager instance."""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = SchemaVersionManager()
    return _schema_manager


def migrate_config_file(file_path: Path, target_version: Optional[str] = None) -> bool:
    """Migrate a configuration file to the current version."""
    try:
        with open(file_path, "r") as f:
            if file_path.suffix.lower() == ".json":
                config = json.load(f)
            else:
                config = yaml.safe_load(f)

        manager = get_schema_manager()
        migrated_config = manager.migrate_config(config, target_version)

        # Write migrated config back
        with open(file_path, "w") as f:
            if file_path.suffix.lower() == ".json":
                json.dump(migrated_config, f, indent=2)
            else:
                yaml.dump(migrated_config, f, default_flow_style=False)

        return True

    except Exception as e:
        logger.error(f"Failed to migrate config file {file_path}: {e}")
        return False
