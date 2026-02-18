"""
APGI Theory Framework - Configuration Management
================================================

Comprehensive configuration system for the APGI validation framework.
Provides flexible configuration management with validation, profiling,
and runtime updates for all APGI parameters and settings.

Features:
- YAML and JSON configuration file support
- Environment variable integration
- Parameter validation with schemas
- Default configuration management
- Runtime configuration updates
- Named configuration profiles with easy switching
- Configuration comparison and diff functionality
- Configuration versioning and rollback capabilities
- Hierarchical configuration inheritance
- Configuration templates and presets

Example:
    >>> config_manager = ConfigManager()
    >>> config = config_manager.get_config("validation")
    >>> config_manager.set_parameter("validation", "threshold", "0.5")
    >>> profile_path = config_manager.create_profile("research", "Research settings")

Author: APGI Research Team
Date: 2026
Version: 1.0
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonschema
import yaml
from dotenv import load_dotenv

try:
    from .logging_config import apgi_logger, log_error
except ImportError:
    try:
        from utils.logging_config import apgi_logger, log_error
    except ImportError:
        # When running directly from utils directory
        import logging_config

        apgi_logger = logging_config.apgi_logger
        log_error = logging_config.log_error

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
PROFILES_DIR = CONFIG_DIR / "profiles"
VERSIONS_DIR = CONFIG_DIR / "versions"

# Ensure directories exist
CONFIG_DIR.mkdir(exist_ok=True)
PROFILES_DIR.mkdir(exist_ok=True)
VERSIONS_DIR.mkdir(exist_ok=True)


@dataclass
class ConfigProfile:
    """Enhanced configuration profile definition."""

    name: str
    description: str
    category: str  # "disorder", "research", "clinical", etc.
    parameters: Dict[str, Any]
    created_at: str
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    author: str = "APGI Framework"
    dependencies: List[str] = field(default_factory=list)
    compatibility: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.tags:
            self.tags = []
        if not self.dependencies:
            self.dependencies = []
        if not self.compatibility:
            self.compatibility = {"min_version": "1.0", "max_version": "2.0"}
        if not self.metadata:
            self.metadata = {}
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ConfigVersion:
    """Configuration version tracking."""

    version_id: str
    timestamp: str
    config_hash: str
    description: str
    author: str = "system"
    changes: List[str] = None

    def __post_init__(self):
        if self.changes is None:
            self.changes = []


@dataclass
class ModelParameters:
    """Default model parameters for APGI formal model."""

    # Timescales (in seconds)
    tau_S: float = 0.5  # 500 ms (Range: 100-1000ms)
    tau_theta: float = 30.0  # 30 s   (Range: 5-60s)

    # Threshold parameters
    theta_0: float = 0.5  # Baseline threshold (Range: 0.1-1.0 AU)

    # Sigmoid parameters
    alpha: float = 5.0  # Sharpness (Range: 3.0-8.0)

    # Sensitivities
    gamma_M: float = -0.3  # Metabolic sensitivity (Range: -0.5 to 0.5)
    gamma_A: float = 0.1  # Arousal sensitivity (Range: -0.3 to 0.3)

    # Reset dynamics
    rho: float = 0.7  # Reset fraction (Range: 0.3-0.9)

    # Noise strengths
    sigma_S: float = 0.05
    sigma_theta: float = 0.02


@dataclass
class SimulationConfig:
    """Simulation configuration settings."""

    default_steps: int = 1000
    default_dt: float = 0.01
    max_steps: int = 100000
    enable_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 150
    save_results: bool = True
    results_format: str = "csv"


@dataclass
class LoggingConfig:
    """Logging configuration settings."""

    level: str = "INFO"
    enable_console: bool = True
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"
    enable_performance_logging: bool = True
    enable_structured_logging: bool = True


@dataclass
class DataConfig:
    """Data processing configuration."""

    default_data_dir: str = "data"
    supported_formats: list = None
    max_file_size_mb: int = 100
    enable_caching: bool = True
    cache_dir: str = "cache"

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["csv", "json", "xlsx", "pkl"]


@dataclass
class ValidationConfig:
    """Validation protocol configuration."""

    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_sensitivity_analysis: bool = True
    sensitivity_samples: int = 100
    enable_robustness_tests: bool = True
    significance_level: float = 0.05


@dataclass
class APGIConfig:
    """Main configuration container."""

    model: ModelParameters = None
    simulation: SimulationConfig = None
    logging: LoggingConfig = None
    data: DataConfig = None
    validation: ValidationConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelParameters()
        if self.simulation is None:
            self.simulation = SimulationConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.validation is None:
            self.validation = ValidationConfig()


class ConfigManager:
    """Advanced configuration management system."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or CONFIG_DIR / "default.yaml"
        self.config = APGIConfig()
        self.schema = self._load_schema()
        self._load_environment()
        self._load_config()
        self.initialize_default_profiles()

    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema for configuration validation."""
        schema = {
            "type": "object",
            "properties": {
                "model": {
                    "type": "object",
                    "properties": {
                        "tau_S": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                        "tau_theta": {
                            "type": "number",
                            "minimum": 5.0,
                            "maximum": 60.0,
                        },
                        "theta_0": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                        "alpha": {"type": "number", "minimum": 1.0, "maximum": 15.0},
                        "gamma_M": {"type": "number", "minimum": -0.5, "maximum": 0.5},
                        "gamma_A": {"type": "number", "minimum": -0.3, "maximum": 0.3},
                        "rho": {"type": "number", "minimum": 0.3, "maximum": 0.9},
                        "sigma_S": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "sigma_theta": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                },
                "simulation": {
                    "type": "object",
                    "properties": {
                        "default_steps": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000000,
                        },
                        "default_dt": {
                            "type": "number",
                            "minimum": 0.001,
                            "maximum": 1.0,
                        },
                        "max_steps": {"type": "integer", "minimum": 1},
                        "enable_plots": {"type": "boolean"},
                        "plot_format": {
                            "type": "string",
                            "enum": ["png", "jpg", "svg", "pdf"],
                        },
                        "plot_dpi": {"type": "integer", "minimum": 50, "maximum": 300},
                        "save_results": {"type": "boolean"},
                        "results_format": {
                            "type": "string",
                            "enum": ["csv", "json", "pkl"],
                        },
                    },
                },
                "logging": {
                    "type": "object",
                    "properties": {
                        "level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                        },
                        "enable_console": {"type": "boolean"},
                        "log_rotation": {"type": "string"},
                        "log_retention": {"type": "string"},
                        "enable_performance_logging": {"type": "boolean"},
                        "enable_structured_logging": {"type": "boolean"},
                    },
                },
                "data": {
                    "type": "object",
                    "properties": {
                        "default_data_dir": {"type": "string"},
                        "supported_formats": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "max_file_size_mb": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10000,
                        },
                        "enable_caching": {"type": "boolean"},
                        "cache_dir": {"type": "string"},
                    },
                },
                "validation": {
                    "type": "object",
                    "properties": {
                        "enable_cross_validation": {"type": "boolean"},
                        "cv_folds": {
                            "type": "integer",
                            "minimum": 2,
                            "maximum": 20,
                        },
                        "enable_sensitivity_analysis": {"type": "boolean"},
                        "sensitivity_samples": {
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 10000,
                        },
                        "enable_robustness_tests": {"type": "boolean"},
                        "significance_level": {
                            "type": "number",
                            "minimum": 0.001,
                            "maximum": 0.5,
                        },
                    },
                },
            },
        }
        return schema

    def _load_environment(self):
        """Load environment variables from .env file."""
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            load_dotenv(env_file)

    def _load_config(self):
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    if self.config_file.suffix.lower() == ".yaml":
                        config_data = yaml.safe_load(f)
                    elif self.config_file.suffix.lower() == ".json":
                        config_data = json.load(f)
                    else:
                        raise ValueError(
                            f"Unsupported config file format: {self.config_file.suffix}"
                        )

                # Validate configuration
                self._validate_config(config_data)

                # Update configuration
                self._update_config(config_data)

                apgi_logger.logger.info(f"Loaded configuration from {self.config_file}")

            except (
                FileNotFoundError,
                PermissionError,
                yaml.YAMLError,
                json.JSONDecodeError,
                ValueError,
                KeyError,
            ) as e:
                apgi_logger.log_error_with_context(
                    e, {"operation": "load_config", "file": str(self.config_file)}
                )
                apgi_logger.logger.warning(
                    f"Using default configuration due to error: {e}"
                )
        else:
            self._save_default_config()
            apgi_logger.logger.info(
                f"Created default configuration at {self.config_file}"
            )

    def _validate_config(self, config_data: Dict[str, Any]):
        """Validate configuration against schema."""
        try:
            jsonschema.validate(config_data, self.schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")

    def _update_config(self, config_data: Dict[str, Any]):
        """Update configuration with loaded data."""
        if "model" in config_data:
            self._update_dataclass(self.config.model, config_data["model"])
        if "simulation" in config_data:
            self._update_dataclass(self.config.simulation, config_data["simulation"])
        if "logging" in config_data:
            self._update_dataclass(self.config.logging, config_data["logging"])
        if "data" in config_data:
            self._update_dataclass(self.config.data, config_data["data"])
        if "validation" in config_data:
            self._update_dataclass(self.config.validation, config_data["validation"])

    def _update_dataclass(self, dataclass_instance, data: Dict[str, Any]):
        """Update dataclass fields from dictionary with validation."""
        for key, value in data.items():
            if hasattr(dataclass_instance, key):
                # Validate the value before setting
                if self._validate_field_value(dataclass_instance, key, value):
                    setattr(dataclass_instance, key, value)
                else:
                    apgi_logger.logger.warning(
                        f"Invalid value for {key}: {value}. Keeping existing value."
                    )
            else:
                apgi_logger.logger.warning(f"Unknown field {key} in configuration")

    def _validate_field_value(
        self, dataclass_instance, field_name: str, value: Any
    ) -> bool:
        """Validate a field value against the schema."""
        try:
            # Get the class name to find the right schema section
            class_name = (
                dataclass_instance.__class__.__name__.lower()
                .replace("parameters", "")
                .replace("config", "")
            )

            # Map class names to schema sections
            schema_mapping = {
                "model": "model",
                "simulation": "simulation",
                "logging": "logging",
                "data": "data",
                "validation": "validation",
            }

            schema_section = schema_mapping.get(class_name)
            if not schema_section or schema_section not in self.schema.get(
                "properties", {}
            ):
                return True  # No validation available, allow

            field_schema = self.schema["properties"][schema_section]["properties"].get(
                field_name
            )
            if not field_schema:
                return True  # No schema for this field, allow

            # Validate based on schema type
            field_type = field_schema.get("type")

            if field_type == "number":
                if not isinstance(value, (int, float)):
                    return False
                # Check min/max constraints
                if "minimum" in field_schema and value < field_schema["minimum"]:
                    return False
                if "maximum" in field_schema and value > field_schema["maximum"]:
                    return False

            elif field_type == "integer":
                if not isinstance(value, int):
                    return False
                # Check min/max constraints
                if "minimum" in field_schema and value < field_schema["minimum"]:
                    return False
                if "maximum" in field_schema and value > field_schema["maximum"]:
                    return False

            elif field_type == "boolean":
                if not isinstance(value, bool):
                    return False

            elif field_type == "string":
                if not isinstance(value, str):
                    return False
                # Check enum constraints
                if "enum" in field_schema and value not in field_schema["enum"]:
                    return False

            return True

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            apgi_logger.logger.warning(f"Validation error for {field_name}: {e}")
            return False

    def _save_default_config(self):
        """Save default configuration to file."""
        config_dict = asdict(self.config)

        with open(self.config_file, "w") as f:
            if self.config_file.suffix.lower() == ".yaml":
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif self.config_file.suffix.lower() == ".json":
                json.dump(config_dict, f, indent=2)

    def get_config(self, section: Optional[str] = None) -> Union[APGIConfig, Any]:
        """Get configuration section or entire config."""
        if section is None:
            return self.config
        elif hasattr(self.config, section):
            return getattr(self.config, section)
        else:
            raise ValueError(f"Unknown configuration section: {section}")

    def set_parameter(self, section: str, parameter: str, value: Any) -> bool:
        """Set a specific configuration parameter."""
        if not hasattr(self.config, section):
            raise ValueError(f"Unknown configuration section: {section}")

        section_obj = getattr(self.config, section)
        if not hasattr(section_obj, parameter):
            raise ValueError(f"Unknown parameter: {parameter} in section: {section}")

        # Convert string values to appropriate types
        converted_value = self._convert_parameter_value(section, parameter, value)

        # Validate parameter value
        self._validate_parameter(section, parameter, converted_value)

        setattr(section_obj, parameter, converted_value)
        apgi_logger.logger.info(f"Updated {section}.{parameter} = {converted_value}")

    def _convert_parameter_value(self, section: str, parameter: str, value: Any) -> Any:
        """Convert parameter value to appropriate type."""
        # Get current value to determine target type
        section_obj = getattr(self.config, section)
        current_value = getattr(section_obj, parameter)

        # If value is already the correct type, return as-is
        if isinstance(value, type(current_value)):
            return value

        # Convert string values
        if isinstance(value, str):
            # Boolean conversion
            if isinstance(current_value, bool):
                return value.lower() in ["true", "1", "yes", "on"]
            # Integer conversion
            elif isinstance(current_value, int):
                return int(value)
            # Float conversion
            elif isinstance(current_value, float):
                return float(value)
            # List conversion (comma-separated)
            elif isinstance(current_value, list):
                return [item.strip() for item in value.split(",")]

        return value

    def _validate_parameter(self, section: str, parameter: str, value: Any):
        """Validate individual parameter value."""
        # Get schema for this parameter
        section_schema = self.schema.get("properties", {}).get(section, {})
        param_schema = section_schema.get("properties", {}).get(parameter, {})

        if param_schema:
            try:
                jsonschema.validate(value, param_schema)
            except jsonschema.ValidationError as e:
                raise ValueError(
                    f"Invalid value for {section}.{parameter}: {e.message}"
                )

    def reset_to_defaults(self, section: Optional[str] = None):
        """Reset configuration to defaults."""
        if section is None:
            self.config = APGIConfig()
            apgi_logger.logger.info("Reset all configuration to defaults")
        else:
            if hasattr(self.config, section):
                default_section = getattr(APGIConfig(), section)
                setattr(self.config, section, default_section)
                apgi_logger.logger.info(f"Reset {section} configuration to defaults")
            else:
                raise ValueError(f"Unknown configuration section: {section}")

    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = file_path or self.config_file
        config_dict = asdict(self.config)

        with open(save_path, "w") as f:
            if save_path.suffix.lower() == ".yaml":
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif save_path.suffix.lower() == ".json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {save_path.suffix}")

        apgi_logger.logger.info(f"Configuration saved to {save_path}")

    # Configuration Profiles and Versioning
    def create_profile(
        self, name: str, description: str, category: str, tags: List[str] = None
    ) -> str:
        """Create a new configuration profile from current settings."""
        profile = ConfigProfile(
            name=name,
            description=description,
            category=category,
            parameters=asdict(self.config),
            created_at=datetime.now().isoformat(),
            tags=tags or [],
        )

        profile_file = PROFILES_DIR / f"{name}.yaml"
        with open(profile_file, "w") as f:
            yaml.dump(asdict(profile), f, default_flow_style=False, indent=2)

        apgi_logger.logger.info(f"Created configuration profile: {name}")
        return str(profile_file)

    def load_profile(self, name: str) -> bool:
        """Load a configuration profile."""
        profile_file = PROFILES_DIR / f"{name}.yaml"
        if not profile_file.exists():
            apgi_logger.logger.error(f"Profile not found: {name}")
            return False

        try:
            with open(profile_file, "r") as f:
                profile_data = yaml.safe_load(f)

            # Create version before loading
            self.create_version(f"Loaded profile: {name}")

            # Apply profile parameters
            self._update_dataclass(self.config, profile_data["parameters"])
            apgi_logger.logger.info(f"Loaded configuration profile: {name}")
            return True

        except (
            FileNotFoundError,
            PermissionError,
            yaml.YAMLError,
            ValueError,
            KeyError,
            AttributeError,
        ) as e:
            apgi_logger.logger.error(f"Error loading profile {name}: {e}")
            return False

    def list_profiles(self, category: str = None) -> List[Dict[str, Any]]:
        """List available configuration profiles."""
        profiles = []

        for profile_file in PROFILES_DIR.glob("*.yaml"):
            try:
                with open(profile_file, "r") as f:
                    profile_data = yaml.safe_load(f)

                if category is None or profile_data.get("category") == category:
                    profiles.append(
                        {
                            "name": profile_data.get("name"),
                            "description": profile_data.get("description"),
                            "category": profile_data.get("category"),
                            "created_at": profile_data.get("created_at"),
                            "version": profile_data.get("version"),
                            "tags": profile_data.get("tags", []),
                        }
                    )
            except (FileNotFoundError, PermissionError, yaml.YAMLError, KeyError) as e:
                apgi_logger.logger.warning(f"Error reading profile {profile_file}: {e}")

        return profiles

    def delete_profile(self, name: str) -> bool:
        """Delete a configuration profile."""
        profile_file = PROFILES_DIR / f"{name}.yaml"
        if profile_file.exists():
            profile_file.unlink()
            apgi_logger.logger.info(f"Deleted configuration profile: {name}")
            return True
        return False

    def create_version(self, description: str, author: str = "user") -> str:
        """Create a version snapshot of current configuration."""
        config_dict = asdict(self.config)
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        version = ConfigVersion(
            version_id=f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            config_hash=config_hash,
            description=description,
            author=author,
        )

        version_file = VERSIONS_DIR / f"{version.version_id}.json"
        version_data = {"version": asdict(version), "config": config_dict}

        with open(version_file, "w") as f:
            json.dump(version_data, f, indent=2)

        apgi_logger.logger.info(f"Created configuration version: {version.version_id}")
        return version.version_id

    def list_versions(self) -> List[Dict[str, Any]]:
        """List available configuration versions."""
        versions = []

        for version_file in sorted(VERSIONS_DIR.glob("*.json"), reverse=True):
            try:
                with open(version_file, "r") as f:
                    version_data = json.load(f)

                version_info = version_data["version"]
                versions.append(
                    {
                        "version_id": version_info["version_id"],
                        "timestamp": version_info["timestamp"],
                        "description": version_info["description"],
                        "author": version_info["author"],
                        "config_hash": version_info["config_hash"],
                    }
                )
            except (
                FileNotFoundError,
                PermissionError,
                json.JSONDecodeError,
                KeyError,
            ) as e:
                apgi_logger.logger.warning(f"Error reading version {version_file}: {e}")

        return versions

    def restore_version(self, version_id: str) -> bool:
        """Restore configuration from a version snapshot."""
        version_file = VERSIONS_DIR / f"{version_id}.json"
        if not version_file.exists():
            apgi_logger.logger.error(f"Version not found: {version_id}")
            return False

        try:
            with open(version_file, "r") as f:
                version_data = json.load(f)

            # Create version before restoring
            self.create_version(f"Restored from version: {version_id}")

            # Restore configuration
            self._update_dataclass(self.config, version_data["config"])
            apgi_logger.logger.info(f"Restored configuration version: {version_id}")
            return True

        except (
            FileNotFoundError,
            PermissionError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
        ) as e:
            apgi_logger.logger.error(f"Error restoring version {version_id}: {e}")
            return False

    def compare_configs(
        self, config1: Dict[str, Any], config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two configurations and return differences."""
        differences = {"added": [], "removed": [], "modified": []}

        def deep_compare(dict1, dict2, path=""):
            for key in dict1:
                current_path = f"{path}.{key}" if path else key
                if key not in dict2:
                    differences["removed"].append(current_path)
                elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    deep_compare(dict1[key], dict2[key], current_path)
                elif dict1[key] != dict2[key]:
                    differences["modified"].append(
                        {"path": current_path, "old": dict1[key], "new": dict2[key]}
                    )

            for key in dict2:
                current_path = f"{path}.{key}" if path else key
                if key not in dict1:
                    differences["added"].append(current_path)

        deep_compare(config1, config2)
        return differences

    def initialize_default_profiles(self):
        """Initialize default configuration profiles."""
        default_profiles = [
            {
                "name": "anxiety-disorder",
                "description": "Optimized for anxiety disorder research",
                "category": "disorder",
                "tags": ["anxiety", "clinical", "research"],
                "parameters": {
                    "model": {
                        "tau_S": 0.3,
                        "tau_theta": 20.0,
                        "theta_0": 0.4,
                        "alpha": 12.0,
                        "gamma_M": -0.4,
                        "gamma_A": 0.15,
                    }
                },
            },
            {
                "name": "adhd",
                "description": "Optimized for ADHD research",
                "category": "disorder",
                "tags": ["adhd", "attention", "clinical"],
                "parameters": {
                    "model": {
                        "tau_S": 0.7,
                        "tau_theta": 40.0,
                        "theta_0": 0.6,
                        "alpha": 8.0,
                        "gamma_M": -0.2,
                        "gamma_A": 0.2,
                    }
                },
            },
            {
                "name": "research-default",
                "description": "Standard research configuration",
                "category": "research",
                "tags": ["default", "research", "general"],
                "parameters": {
                    "simulation": {"default_steps": 10000, "default_dt": 0.01},
                    "validation": {"enable_cross_validation": True, "cv_folds": 10},
                },
            },
        ]

        for profile_data in default_profiles:
            profile_file = PROFILES_DIR / f"{profile_data['name']}.yaml"
            if not profile_file.exists():
                profile = ConfigProfile(
                    name=profile_data["name"],
                    description=profile_data["description"],
                    category=profile_data["category"],
                    parameters=profile_data["parameters"],
                    created_at=datetime.now().isoformat(),
                    tags=profile_data["tags"],
                )

                with open(profile_file, "w") as f:
                    yaml.dump(asdict(profile), f, default_flow_style=False, indent=2)

                apgi_logger.logger.info(
                    f"Created default profile: {profile_data['name']}"
                )

    def export_config_template(self, file_path: str, format: str = "yaml"):
        """Export configuration template with comments."""
        template_config = APGIConfig()
        config_dict = asdict(template_config)

        # Add comments for YAML format
        if format.lower() == "yaml":
            template_content = """# APGI Theory Framework Configuration
# ======================================

# Model Parameters
# ---------------
# Core parameters for the APGI formal model
model:
  # Timescales (in seconds)
  tau_S: 0.5      # Surprise accumulation timescale (100-1000ms)
  tau_theta: 30.0 # Threshold adaptation timescale (5-60s)

  # Threshold parameters
  theta_0: 0.5    # Baseline threshold (0.1-1.0 AU)

  # Sigmoid parameters
  alpha: 10.0     # Sigmoid sharpness (1-15)

  # Sensitivities
  gamma_M: -0.3   # Metabolic sensitivity (-0.5 to 0.5)
  gamma_A: 0.1    # Arousal sensitivity (-0.3 to 0.3)

  # Reset dynamics
  rho: 0.7        # Reset fraction (0.3-0.9)

  # Noise strengths
  sigma_S: 0.05
  sigma_theta: 0.02

# Simulation Configuration
# ----------------------
simulation:
  default_steps: 1000
  default_dt: 0.01
  max_steps: 100000
  enable_plots: true
  plot_format: "png"
  plot_dpi: 150
  save_results: true
  results_format: "csv"

# Logging Configuration
# --------------------
logging:
  level: "INFO"
  enable_console: true
  log_rotation: "10 MB"
  log_retention: "30 days"
  enable_performance_logging: true
  enable_structured_logging: true

# Data Configuration
# -----------------
data:
  default_data_dir: "data"
  supported_formats: ["csv", "json", "xlsx", "pkl"]
  max_file_size_mb: 100
  enable_caching: true
  cache_dir: "cache"

# Validation Configuration
# -----------------------
validation:
  enable_cross_validation: true
  cv_folds: 5
  enable_sensitivity_analysis: true
  sensitivity_samples: 100
  enable_robustness_tests: true
  significance_level: 0.05
"""
            with open(file_path, "w") as f:
                f.write(template_content)
        else:
            # JSON format (no comments)
            with open(file_path, "w") as f:
                json.dump(config_dict, f, indent=2)

        apgi_logger.logger.info(f"Configuration template exported to {file_path}")

    def get_environment_override(self, parameter: str) -> Optional[str]:
        """Get environment variable override for configuration parameter."""
        env_var = f"APGI_{parameter.upper()}"
        return os.getenv(env_var)

    def apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            "APGI_LOG_LEVEL": ("logging", "level"),
            "APGI_ENABLE_PLOTS": ("simulation", "enable_plots"),
            "APGI_DATA_DIR": ("data", "default_data_dir"),
            "APGI_TAU_S": ("model", "tau_S"),
            "APGI_TAU_THETA": ("model", "tau_theta"),
            "APGI_THETA_0": ("model", "theta_0"),
        }

        for env_var, (section, param) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string to appropriate type
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif value.replace(".", "").isdigit():
                    value = float(value) if "." in value else int(value)

                self.set_parameter(section, param, value)
                apgi_logger.logger.info(
                    f"Applied environment override: {env_var} -> {section}.{param} = {value}"
                )


# Global configuration manager instance
config_manager = ConfigManager()


class EnhancedConfigManager(ConfigManager):
    """Enhanced configuration manager with profile switching and comparison."""

    def __init__(self):
        super().__init__()
        self.current_profile = None
        self.profile_history = []
        self.max_history = 10

    def create_profile(
        self,
        name: str,
        description: str,
        category: str = "custom",
        tags: List[str] = None,
        author: str = "APGI Framework",
    ) -> ConfigProfile:
        """Create a new configuration profile from current settings."""

        # Get current configuration
        current_config = self.get_config()

        # Create profile
        profile = ConfigProfile(
            name=name,
            description=description,
            category=category,
            parameters=asdict(current_config),
            created_at=datetime.now().isoformat(),
            tags=tags or [],
            author=author,
        )

        # Save profile
        profile_path = PROFILES_DIR / f"{name}.yaml"
        with open(profile_path, "w") as f:
            yaml.dump(asdict(profile), f, default_flow_style=False)

        apgi_logger.logger.info(f"Created configuration profile: {name}")
        return profile

    def load_profile(self, name: str) -> bool:
        """Load a configuration profile."""
        profile_path = PROFILES_DIR / f"{name}.yaml"

        if not profile_path.exists():
            apgi_logger.logger.error(f"Profile not found: {name}")
            return False

        try:
            with open(profile_path, "r") as f:
                profile_data = yaml.safe_load(f)

            profile = ConfigProfile(**profile_data)

            # Save current state to history
            if self.current_profile:
                self.profile_history.append(self.current_profile)
                if len(self.profile_history) > self.max_history:
                    self.profile_history.pop(0)

            # Apply profile
            self.config = self._dict_to_config(profile.parameters)
            self.current_profile = profile

            apgi_logger.logger.info(f"Loaded configuration profile: {name}")
            return True

        except (
            FileNotFoundError,
            PermissionError,
            yaml.YAMLError,
            ValueError,
            KeyError,
            TypeError,
        ) as e:
            apgi_logger.logger.error(f"Error loading profile {name}: {e}")
            return False

    def list_profiles(self, category: Optional[str] = None) -> List[ConfigProfile]:
        """List available configuration profiles."""
        profiles = []

        for profile_file in PROFILES_DIR.glob("*.yaml"):
            try:
                with open(profile_file, "r") as f:
                    profile_data = yaml.safe_load(f)

                profile = ConfigProfile(**profile_data)

                if category is None or profile.category == category:
                    profiles.append(profile)

            except (
                FileNotFoundError,
                PermissionError,
                yaml.YAMLError,
                KeyError,
                TypeError,
            ) as e:
                apgi_logger.logger.warning(f"Error reading profile {profile_file}: {e}")

        return profiles

    def compare_profiles(
        self, profile1_name: str, profile2_name: str
    ) -> Dict[str, Any]:
        """Compare two configuration profiles."""
        profile1_path = PROFILES_DIR / f"{profile1_name}.yaml"
        profile2_path = PROFILES_DIR / f"{profile2_name}.yaml"

        if not profile1_path.exists() or not profile2_path.exists():
            return {"error": "One or both profiles not found"}

        try:
            with open(profile1_path, "r") as f:
                profile1_data = yaml.safe_load(f)

            with open(profile2_path, "r") as f:
                profile2_data = yaml.safe_load(f)

            # Compare parameters
            params1 = profile1_data["parameters"]
            params2 = profile2_data["parameters"]

            differences = self._compare_dicts(params1, params2)

            return {
                "profile1": profile1_name,
                "profile2": profile2_name,
                "differences": differences,
                "similarity_score": self._calculate_similarity(params1, params2),
            }

        except (
            FileNotFoundError,
            PermissionError,
            yaml.YAMLError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as e:
            return {"error": f"Error comparing profiles: {e}"}

    def _compare_dicts(
        self, dict1: Dict, dict2: Dict, path: str = ""
    ) -> List[Dict[str, Any]]:
        """Recursively compare two dictionaries."""
        differences = []

        all_keys = set(dict1.keys()) | set(dict2.keys())

        for key in all_keys:
            current_path = f"{path}.{key}" if path else key

            if key not in dict1:
                differences.append(
                    {"path": current_path, "type": "added", "value": dict2[key]}
                )
            elif key not in dict2:
                differences.append(
                    {"path": current_path, "type": "removed", "value": dict1[key]}
                )
            elif dict1[key] != dict2[key]:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    differences.extend(
                        self._compare_dicts(dict1[key], dict2[key], current_path)
                    )
                else:
                    differences.append(
                        {
                            "path": current_path,
                            "type": "changed",
                            "old_value": dict1[key],
                            "new_value": dict2[key],
                        }
                    )

        return differences

    def _calculate_similarity(self, dict1: Dict, dict2: Dict) -> float:
        """Calculate similarity score between two configurations."""
        all_keys = set(dict1.keys()) | set(dict2.keys())
        if not all_keys:
            return 1.0

        matching_keys = set(dict1.keys()) & set(dict2.keys())
        value_matches = 0

        for key in matching_keys:
            if dict1[key] == dict2[key]:
                value_matches += 1
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                # Recursive similarity for nested dicts
                nested_similarity = self._calculate_similarity(dict1[key], dict2[key])
                value_matches += nested_similarity

        return value_matches / len(all_keys)

    def switch_to_profile(self, name: str) -> bool:
        """Switch to a different configuration profile."""
        return self.load_profile(name)

    def rollback_profile(self) -> bool:
        """Rollback to previous profile."""
        if not self.profile_history:
            apgi_logger.logger.warning("No previous profile to rollback to")
            return False

        previous_profile = self.profile_history.pop()
        return self.load_profile(previous_profile.name)

    def get_profile_history(self) -> List[str]:
        """Get list of profile names in history."""
        return [profile.name for profile in self.profile_history]

    def delete_profile(self, name: str) -> bool:
        """Delete a configuration profile."""
        profile_path = PROFILES_DIR / f"{name}.yaml"

        if not profile_path.exists():
            apgi_logger.logger.error(f"Profile not found: {name}")
            return False

        try:
            profile_path.unlink()
            apgi_logger.logger.info(f"Deleted configuration profile: {name}")
            return True

        except (FileNotFoundError, PermissionError, OSError) as e:
            apgi_logger.logger.error(f"Error deleting profile {name}: {e}")
            return False

    def export_profile(self, name: str, format: str = "yaml") -> Optional[str]:
        """Export profile in specified format."""
        profile_path = PROFILES_DIR / f"{name}.yaml"

        if not profile_path.exists():
            apgi_logger.logger.error(f"Profile not found: {name}")
            return None

        try:
            with open(profile_path, "r") as f:
                profile_data = yaml.safe_load(f)

            if format.lower() == "json":
                export_path = PROFILES_DIR / f"{name}.json"
                with open(export_path, "w") as f:
                    json.dump(profile_data, f, indent=2)
                return str(export_path)
            elif format.lower() == "yaml":
                return str(profile_path)
            else:
                apgi_logger.logger.error(f"Unsupported export format: {format}")
                return None

        except (
            FileNotFoundError,
            PermissionError,
            yaml.YAMLError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
        ) as e:
            apgi_logger.logger.error(f"Error exporting profile {name}: {e}")
            return None

    def import_profile(self, file_path: str, name: Optional[str] = None) -> bool:
        """Import profile from file."""
        file_path = Path(file_path)

        if not file_path.exists():
            apgi_logger.logger.error(f"File not found: {file_path}")
            return False

        try:
            with open(file_path, "r") as f:
                if file_path.suffix.lower() == ".json":
                    profile_data = json.load(f)
                else:
                    profile_data = yaml.safe_load(f)

            # Use provided name or extract from file
            if name:
                profile_data["name"] = name
            elif "name" not in profile_data:
                profile_data["name"] = file_path.stem

            # Save as profile
            profile = ConfigProfile(**profile_data)
            profile_path = PROFILES_DIR / f"{profile.name}.yaml"

            with open(profile_path, "w") as f:
                yaml.dump(asdict(profile), f, default_flow_style=False)

            apgi_logger.logger.info(f"Imported configuration profile: {profile.name}")
            return True

        except (
            FileNotFoundError,
            PermissionError,
            yaml.YAMLError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
            TypeError,
        ) as e:
            apgi_logger.logger.error(f"Error importing profile from {file_path}: {e}")
            return False

    def validate_profile(self, name: str) -> Dict[str, Any]:
        """Validate a configuration profile."""
        profile_path = PROFILES_DIR / f"{name}.yaml"

        if not profile_path.exists():
            return {"valid": False, "error": "Profile not found"}

        try:
            with open(profile_path, "r") as f:
                profile_data = yaml.safe_load(f)

            profile = ConfigProfile(**profile_data)

            # Validate structure
            validation_errors = []

            if not profile.name:
                validation_errors.append("Profile name is required")

            if not profile.description:
                validation_errors.append("Profile description is required")

            if not profile.parameters:
                validation_errors.append("Profile parameters are required")

            # Validate parameter structure
            required_sections = ["model", "simulation", "logging", "data", "validation"]
            for section in required_sections:
                if section not in profile.parameters:
                    validation_errors.append(f"Missing required section: {section}")

            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "profile": profile,
            }

        except (
            FileNotFoundError,
            PermissionError,
            yaml.YAMLError,
            ValueError,
            KeyError,
            TypeError,
        ) as e:
            return {"valid": False, "error": f"Validation error: {e}"}

    def get_profile_stats(self) -> Dict[str, Any]:
        """Get statistics about configuration profiles."""
        profiles = self.list_profiles()

        stats = {
            "total_profiles": len(profiles),
            "categories": {},
            "total_versions": len(list(VERSIONS_DIR.glob("*.json"))),
            "current_profile": (
                self.current_profile.name if self.current_profile else None
            ),
            "history_length": len(self.profile_history),
        }

        for profile in profiles:
            category = profile.category
            if category not in stats["categories"]:
                stats["categories"][category] = 0
            stats["categories"][category] += 1

        return stats


# Enhanced configuration manager instance
enhanced_config_manager = EnhancedConfigManager()


# Convenience functions for enhanced features
def create_config_profile(
    name: str, description: str, category: str = "custom"
) -> bool:
    """Create a new configuration profile."""
    try:
        enhanced_config_manager.create_profile(name, description, category)
        return True
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        apgi_logger.logger.error(f"Error creating profile: {e}")
        return False


def switch_config_profile(name: str) -> bool:
    """Switch to a configuration profile."""
    return enhanced_config_manager.switch_to_profile(name)


def list_config_profiles(category: Optional[str] = None) -> List[Dict[str, str]]:
    """List available configuration profiles."""
    profiles = enhanced_config_manager.list_profiles(category)
    return [
        {
            "name": p.name,
            "description": p.description,
            "category": p.category,
            "version": p.version,
            "author": p.author,
            "tags": p.tags,
        }
        for p in profiles
    ]


def compare_config_profiles(profile1: str, profile2: str) -> Dict[str, Any]:
    """Compare two configuration profiles."""
    return enhanced_config_manager.compare_profiles(profile1, profile2)


def rollback_config_profile() -> bool:
    """Rollback to previous configuration profile."""
    return enhanced_config_manager.rollback_profile()


# Convenience functions
def get_config(section: Optional[str] = None):
    """Get configuration section or entire config."""
    return config_manager.get_config(section)


def set_parameter(section: str, parameter: str, value: Any):
    """Set a specific configuration parameter."""
    try:
        config_manager.set_parameter(section, parameter, value)
        return True
    except (ValueError, KeyError, AttributeError, TypeError) as e:
        return False


def reset_config(section: Optional[str] = None):
    """Reset configuration to defaults."""
    config_manager.reset_to_defaults(section)


def save_config(file_path: Optional[str] = None):
    """Save current configuration."""
    config_manager.save_config(file_path)


if __name__ == "__main__":
    # Test configuration system
    print("Testing APGI Configuration Management System")
    print("=" * 50)

    # Display current configuration
    config = config_manager.get_config()
    print("Current configuration:")
    print(f"  Model tau_S: {config.model.tau_S}")
    print(f"  Simulation steps: {config.simulation.default_steps}")
    print(f"  Logging level: {config.logging.level}")

    # Test parameter setting
    set_parameter("model", "tau_S", 0.8)
    print(f"Updated model tau_S: {config.model.tau_S}")

    # Test environment overrides
    os.environ["APGI_LOG_LEVEL"] = "DEBUG"
    config_manager.apply_environment_overrides()
    print(f"Logging level after env override: {config.logging.level}")

    # Export template
    template_file = CONFIG_DIR / "config_template.yaml"
    config_manager.export_config_template(str(template_file))
    print(f"Configuration template exported to: {template_file}")

    print("Configuration system test completed successfully!")
