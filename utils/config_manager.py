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

"""

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import jsonschema
import yaml
from dotenv import load_dotenv

# Import logging with fallback for different execution contexts
_logger_source = "unknown"

# First, try relative import (when imported as part of package)
try:
    from .logging_config import apgi_logger, log_error

    _logger_source = "relative_import"
except ImportError:
    pass

# If relative import failed, use importlib to load from the correct location
if _logger_source == "unknown":
    try:
        import importlib.util
        import sys

        # Get the directory containing this file
        this_file = Path(__file__).resolve()
        utils_dir = this_file.parent
        logging_config_path = utils_dir / "logging_config.py"

        # Load the module explicitly from the correct path
        spec = importlib.util.spec_from_file_location(
            "logging_config_local", logging_config_path
        )
        if spec and spec.loader:
            logging_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(logging_module)
            apgi_logger = logging_module.apgi_logger
            log_error = logging_module.log_error
            _logger_source = "importlib_local"
    except Exception:
        pass

# Final fallback: create a basic logger
if _logger_source == "unknown":
    import logging
    import sys

    logger: logging.Logger = logging.getLogger("apgi")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    # Create a type-compatible APGILogger wrapper
    _FallbackLogger = Any  # Forward reference for type compatibility

    class _FallbackAPGILogger:
        """Wrapper class for logger to maintain compatibility."""

        def __init__(self, logger_instance: logging.Logger) -> None:
            self.logger = logger_instance

        def __getattr__(self, name: str) -> Any:
            """Delegate all attribute access to the underlying logger."""
            return getattr(self.logger, name)

        def info(self, msg: str) -> None:
            self.logger.info(msg)

        def error(self, msg: str) -> None:
            self.logger.error(msg)

        def warning(self, msg: str) -> None:
            self.logger.warning(msg)

        def debug(self, msg: str) -> None:
            self.logger.debug(msg)

    apgi_logger: Any = _FallbackAPGILogger(logger)  # type: ignore[no-redef]

    def log_error(message: str) -> None:
        logger.error(message)

    _logger_source = "fallback_created"

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
PROFILES_DIR = CONFIG_DIR / "profiles"
VERSIONS_DIR = CONFIG_DIR / "versions"

# Module-level flag to prevent duplicate config loading messages
_config_loaded = False
_config_lock = threading.Lock()


def _validate_file_path(file_path: str, allowed_dirs: List[str] = None) -> Path:
    """Validate file path to prevent directory traversal attacks.

    Args:
        file_path: The file path to validate
        allowed_dirs: List of allowed directory names (e.g., ["config", "data"])

    Returns:
        Validated Path object

    Raises:
        ValueError: If path validation fails
    """
    try:
        path = Path(file_path).resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid file path: {file_path}") from e

    # Check if path is within project root or allowed directories
    if path.is_absolute():
        # For absolute paths, ensure it's within project root or allowed subdirectories
        if allowed_dirs:
            # Check if path is within any allowed directory
            allowed_paths = [PROJECT_ROOT / d for d in allowed_dirs]
            if not any(
                path.is_relative_to(allowed_path) for allowed_path in allowed_paths
            ):
                raise ValueError(
                    f"File path must be within allowed directories: {allowed_dirs}"
                )
        else:
            # If no allowed_dirs specified, allow only project root and immediate subdirectories
            if not path.is_relative_to(PROJECT_ROOT):
                raise ValueError(
                    f"File path must be within project directory: {PROJECT_ROOT}"
                )
    else:
        # For relative paths, resolve relative to project root
        # Don't allow .. to escape project directory
        if ".." in str(path):
            raise ValueError("Path traversal not allowed")

    return path


@dataclass
class ConfigProfile:
    """Enhanced configuration profile definition.

    Supports empirical validation tracking for clinical and research profiles,
    including data source classification and validation status.
    """

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
    # Empirical validation tracking (NEW)
    data_source: str = "synthetic"  # "empirical", "synthetic", "hybrid"
    empirical_validation: Dict[str, Any] = field(default_factory=dict)

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
        if not self.empirical_validation:
            self.empirical_validation = {
                "status": (
                    "COMPLETE"
                    if self.data_source == "empirical"
                    else "SYNTHETIC_PENDING_EMPIRICAL"
                ),
                "citations": [],
                "pending_datasets": [],
            }

    @property
    def validation_status(self) -> str:
        """Get the validation status of this profile."""
        return self.empirical_validation.get("status", "UNKNOWN")

    @property
    def is_empirically_validated(self) -> bool:
        """Check if profile has empirical validation."""
        return self.data_source == "empirical" and self.validation_status == "COMPLETE"

    @property
    def pending_datasets(self) -> List[str]:
        """Get list of pending datasets for empirical validation."""
        return self.empirical_validation.get("pending_datasets", [])


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
    tau_M: float = 1.52  # Somatic marker time constant [seconds]

    # Threshold parameters
    theta_0: float = 0.5  # Baseline threshold (Range: 0.1-1.0 AU)

    # Sigmoid parameters
    alpha: float = 5.0  # Sharpness (Range: 3.0-8.0)

    # Sensitivities
    gamma_M: float = -0.3  # Metabolic sensitivity (Range: -0.5 to 0.5)
    gamma_A: float = 0.1  # Arousal sensitivity (Range: -0.3 to 0.3)
    beta: float = 1.2  # Somatic bias weight

    # Precision parameters
    Pi_e: float = 1.0  # Exteroceptive precision (baseline)
    Pi_i_baseline: float = 1.0  # Interoceptive precision (baseline)
    beta_Pi_i: float = 1.0  # Composite interoceptive parameter

    # Reset dynamics
    rho: float = 0.7  # Reset fraction (Range: 0.3-0.9)

    # Noise strengths
    sigma_S: float = 0.05
    sigma_theta: float = 0.02
    sigma_noise: float = 0.8  # Sensory noise amplitude

    # Interoceptive parameters
    somatic_gain: float = 1.0  # Somatic marker gain
    homeostatic_cost_weight: float = 0.2  # Weight for homeostatic predictions

    # Network architecture parameters
    hidden_dim: int = 64  # Default hidden dimension
    somatic_hidden_dim: int = 32  # Somatic marker hidden dimension

    # Learning parameters
    learning_rate: float = 0.01  # Network learning rate
    value_learning_rate: float = 0.001  # Value network learning rate

    # Simulation parameters
    dt: float = 0.05  # Simulation time step
    duration: float = 1.0  # Simulation duration

    # F6.2 parameters
    F6_2_MIN_INTEGRATION_RATIO: float = 4.0  # ≥4× RNN  (spec criterion)
    F6_2_FALSIFICATION_RATIO: float = 2.5  # falsified if ratio < 2.5×
    F6_2_MIN_CURVE_FIT_R2: float = 0.85  # R² ≥ 0.85
    F6_2_WILCOXON_ALPHA: float = 0.05

    # Innovation 29 (LNN AUROC superiority) parameters
    F6: dict = None  # Will be initialized in __post_init__

    def __post_init__(self):
        if self.F6 is None:
            self.F6 = {"delta_auroc_min": 0.05}


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
    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_sensitivity_analysis: bool = True
    sensitivity_samples: int = 100
    enable_robustness_tests: bool = True
    significance_level: float = 0.05


@dataclass
class LoggingConfig:
    """Logging configuration settings."""

    level: str = "INFO"
    enable_console: bool = True
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"
    enable_performance_logging: bool = True
    log_directory: str = "logs/"
    enable_structured_logging: bool = True


@dataclass
class DataConfig:
    """Data processing configuration."""

    default_data_dir: str = "data"
    supported_formats: list = None
    max_file_size_mb: int = 100
    enable_caching: bool = True
    cache_dir: str = "cache"
    sample_size: int = 1000
    time_series_length: int = 100
    eeg_channels: int = 32
    sampling_rate: int = 1000
    max_load_size_mb: int = 100  # Maximum file size for loading (in MB)

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
    n_simulations: int = 100


@dataclass
class BatchConfig:
    """Batch processing configuration settings."""

    max_workers: int = 4
    use_processes: bool = False
    chunk_size: int = 100
    timeout: int = 300  # 5 minutes
    auto_scale: bool = True


@dataclass
class FalsificationConfig:
    """Falsification testing configuration."""

    cumulative_reward_advantage_threshold: float = 18.0
    cohens_d_threshold: float = 0.60
    significance_level: float = 0.01
    threshold_reduction_min: float = 20.0
    cohens_d_adaptation_threshold: float = 0.70
    tau_theta_min: float = 10.0
    tau_theta_max: float = 100.0


@dataclass
class APGIConfig:
    """Main configuration container."""

    model: ModelParameters = field(default_factory=ModelParameters)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    falsification: FalsificationConfig = field(default_factory=FalsificationConfig)


class ConfigManager:
    """Advanced configuration management system with hot-reload support."""

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.config_file = Path(config_file or CONFIG_DIR / "default.yaml")
        self.config = APGIConfig()
        self.schema = self._load_schema()
        self._load_environment()
        self._load_config()
        self.initialize_default_profiles()

        # Hot-reload attributes
        self._hot_reload_enabled: bool = False
        self._hot_reload_thread: Optional[threading.Thread] = None
        self._hot_reload_stop_event = threading.Event()
        self._last_config_mtime: float = 0
        self._last_config_hash: Optional[str] = None
        self._config_change_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._hot_reload_check_interval = 2.0  # seconds

    def enable_hot_reload(
        self,
        check_interval: float = 2.0,
        on_change: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        """Enable configuration hot-reload.

        BUG-013 Fix: Configuration Hot-Reload - Monitors config file for changes
        and automatically reloads without requiring application restart.

        Args:
            check_interval: Seconds between file change checks (default: 2.0)
            on_change: Optional callback function called when config changes.
                      Receives dict of changed fields with old/new values.

        Returns:
            True if hot-reload was enabled successfully, False otherwise

        Example:
            >>> def on_config_change(changes):
            ...     print(f"Config changed: {changes}")
            >>> config_manager.enable_hot_reload(on_change=on_config_change)
        """
        if self._hot_reload_enabled:
            apgi_logger.warning("Hot-reload is already enabled")
            return False

        # Validate config file exists
        if not self.config_file.exists():
            apgi_logger.error(
                f"Cannot enable hot-reload: config file not found: {self.config_file}"
            )
            return False

        # Store initial state
        self._hot_reload_check_interval = max(0.5, check_interval)  # Min 0.5s
        self._last_config_mtime = self._get_file_mtime()
        self._last_config_hash = self._compute_config_hash()

        # Register callback if provided
        if on_change:
            self._config_change_callbacks.append(on_change)

        # Start watcher thread
        self._hot_reload_stop_event.clear()
        self._hot_reload_thread = threading.Thread(
            target=self._hot_reload_worker,
            name="ConfigHotReload",
            daemon=True,
        )
        self._hot_reload_thread.start()
        self._hot_reload_enabled = True

        apgi_logger.info(
            f"Configuration hot-reload enabled (interval: {self._hot_reload_check_interval}s)"
        )
        return True

    def disable_hot_reload(self) -> bool:
        """Disable configuration hot-reload.

        Returns:
            True if hot-reload was disabled successfully
        """
        if not self._hot_reload_enabled:
            return False

        self._hot_reload_stop_event.set()
        if self._hot_reload_thread and self._hot_reload_thread.is_alive():
            self._hot_reload_thread.join(timeout=5.0)

        self._hot_reload_enabled = False
        self._config_change_callbacks.clear()

        apgi_logger.info("Configuration hot-reload disabled")
        return True

    def register_change_callback(
        self, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register a callback to be called when configuration changes.

        Args:
            callback: Function receiving dict of changed fields
        """
        if callback not in self._config_change_callbacks:
            self._config_change_callbacks.append(callback)

    def unregister_change_callback(
        self, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Unregister a configuration change callback."""
        if callback in self._config_change_callbacks:
            self._config_change_callbacks.remove(callback)

    def _get_file_mtime(self) -> float:
        """Get modification time of config file."""
        try:
            return os.path.getmtime(self.config_file)
        except (OSError, FileNotFoundError):
            return 0

    def _compute_config_hash(self) -> str:
        """Compute MD5 hash of current config file contents."""
        try:
            with open(self.config_file, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except (OSError, FileNotFoundError):
            return ""

    def _hot_reload_worker(self):
        """Background worker thread that monitors config file for changes."""
        while not self._hot_reload_stop_event.is_set():
            try:
                # Check if file has been modified
                current_mtime = self._get_file_mtime()
                if current_mtime != self._last_config_mtime:
                    # Double-check with hash to avoid false positives
                    current_hash = self._compute_config_hash()
                    if current_hash != self._last_config_hash:
                        # File changed - reload
                        self._reload_config()
                        self._last_config_mtime = current_mtime
                        self._last_config_hash = current_hash

                # Wait for next check or stop event
                self._hot_reload_stop_event.wait(self._hot_reload_check_interval)

            except Exception as e:
                apgi_logger.error(f"Error in hot-reload worker: {e}")
                self._hot_reload_stop_event.wait(self._hot_reload_check_interval)

    def _reload_config(self):
        """Reload configuration from file and notify callbacks."""
        try:
            # Store old config for comparison
            old_config_dict = asdict(self.config)

            # Reload config
            self._load_config()

            # Get new config
            new_config_dict = asdict(self.config)

            # Compute differences
            changes = self._compute_config_diff(old_config_dict, new_config_dict)

            if changes:
                apgi_logger.info(
                    f"Configuration hot-reloaded: {len(changes)} change(s) detected"
                )

                # Notify callbacks
                for callback in self._config_change_callbacks:
                    try:
                        callback(changes)
                    except Exception as e:
                        apgi_logger.error(f"Error in config change callback: {e}")
            else:
                apgi_logger.debug("Configuration file changed but no differences found")

        except Exception as e:
            apgi_logger.error(f"Failed to hot-reload configuration: {e}")

    def _compute_config_diff(
        self, old: Dict[str, Any], new: Dict[str, Any], path: str = ""
    ) -> Dict[str, Dict[str, Any]]:
        """Compute differences between old and new configuration.

        Returns dict of changed paths with old and new values.
        """
        changes = {}

        all_keys = set(old.keys()) | set(new.keys())
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key

            if key not in old:
                changes[current_path] = {"old": None, "new": new[key]}
            elif key not in new:
                changes[current_path] = {"old": old[key], "new": None}
            elif isinstance(old[key], dict) and isinstance(new[key], dict):
                nested_changes = self._compute_config_diff(
                    old[key], new[key], current_path
                )
                changes.update(nested_changes)
            elif old[key] != new[key]:
                changes[current_path] = {"old": old[key], "new": new[key]}

        return changes

    def get_hot_reload_status(self) -> Dict[str, Any]:
        """Get current hot-reload status.

        Returns:
            Dict with enabled status, check interval, and callback count
        """
        return {
            "enabled": self._hot_reload_enabled,
            "check_interval": self._hot_reload_check_interval,
            "registered_callbacks": len(self._config_change_callbacks),
            "config_file": str(self.config_file),
            "last_modified": (
                datetime.fromtimestamp(self._last_config_mtime).isoformat()
                if self._last_config_mtime
                else None
            ),
        }

    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema for configuration validation."""
        schema_file = Path(__file__).parent.parent / "config" / "config_schema.json"
        if schema_file.exists():
            try:
                with open(schema_file, "r", encoding="utf-8") as f:
                    schema_data = json.load(f)

                # Verify schema integrity before using it
                if not self._verify_schema_integrity(schema_file, schema_data):
                    apgi_logger.error(
                        f"Schema integrity verification failed for {schema_file}. "
                        "Using fallback schema for security."
                    )
                    return self._get_fallback_schema()

                return schema_data
            except (json.JSONDecodeError, IOError) as e:
                apgi_logger.warning(f"Failed to load schema from {schema_file}: {e}")
                return self._get_fallback_schema()
        else:
            apgi_logger.warning(f"Schema file not found: {schema_file}")
            return self._get_fallback_schema()

    def _get_fallback_schema(self) -> Dict[str, Any]:
        """Fallback schema if external schema file cannot be loaded."""
        return {
            "type": "object",
            "properties": {
                "validation": {
                    "type": "object",
                    "properties": {
                        "enable_cross_validation": {"type": "boolean"},
                        "cv_folds": {"type": "integer", "minimum": 2},
                        "enable_sensitivity_analysis": {"type": "boolean"},
                        "sensitivity_samples": {"type": "integer", "minimum": 10},
                        "enable_robustness_tests": {"type": "boolean"},
                        "significance_level": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                },
                "batch": {
                    "type": "object",
                    "properties": {
                        "max_workers": {"type": "integer", "minimum": 1},
                        "use_processes": {"type": "boolean"},
                        "chunk_size": {"type": "integer", "minimum": 10},
                        "timeout": {"type": "integer", "minimum": 60},
                        "auto_scale": {"type": "boolean"},
                    },
                },
                "model": {
                    "type": "object",
                    "properties": {
                        "tau_S": {"type": "number", "minimum": 0.1},
                        "tau_theta": {"type": "number", "minimum": 5.0},
                        "theta_0": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                        "alpha": {"type": "number", "minimum": 1.0},
                        "gamma_M": {"type": "number"},
                        "gamma_A": {"type": "number"},
                        "rho": {"type": "number", "minimum": 0.3, "maximum": 0.9},
                        "sigma_S": {"type": "number", "minimum": 0.0},
                        "sigma_theta": {"type": "number", "minimum": 0.0},
                    },
                },
                "simulation": {
                    "type": "object",
                    "properties": {
                        "default_steps": {"type": "integer", "minimum": 1},
                        "default_dt": {"type": "number", "minimum": 0.001},
                        "enable_plots": {"type": "boolean"},
                    },
                },
            },
        }

    def _verify_schema_integrity(
        self, schema_file: Path, schema_data: Dict[str, Any]
    ) -> bool:
        """Verify schema file integrity using SHA-256 hash.

        Args:
            schema_file: Path to the schema file
            schema_data: Loaded schema data to verify

        Returns:
            True if schema integrity is verified, False otherwise
        """
        # Expected SHA-256 hash of the legitimate schema
        # This hash should be updated when schema changes are made
        EXPECTED_SCHEMA_HASH = (
            "13c1767e524b7ca5b709f8d25202ec084f5debdf97e84c1676975469816f9cc4"
        )

        try:
            # Calculate hash of the loaded schema data
            schema_json = json.dumps(schema_data, sort_keys=True, separators=(",", ":"))
            actual_hash = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()

            if actual_hash != EXPECTED_SCHEMA_HASH:
                apgi_logger.error(
                    f"Schema integrity check failed for {schema_file}: "
                    f"expected {EXPECTED_SCHEMA_HASH}, got {actual_hash}"
                )
                return False

            # Only log schema verification once per process
            global _config_loaded
            with _config_lock:
                if not _config_loaded:
                    apgi_logger.info(f"Schema integrity verified for {schema_file}")
                    _config_loaded = True
            return True

        except Exception as e:
            apgi_logger.error(
                f"Error during schema integrity verification for {schema_file}: {e}"
            )
            return False

    def _load_environment(self):
        """Load environment variables from .env file."""
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            load_dotenv(env_file)

    def _load_config(self):
        """Load configuration from file."""
        # Validate config file path for security
        try:
            validated_config_path = _validate_file_path(
                str(self.config_file), allowed_dirs=["config"]
            )
            self.config_file = validated_config_path
        except ValueError as e:
            apgi_logger.warning(f"Config file path validation failed: {e}")
            apgi_logger.warning("Using default configuration")
            self._save_default_config()
            return

        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
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

                # Only log config loading once per process
                global _config_loaded
                with _config_lock:
                    if not _config_loaded:
                        apgi_logger.info(
                            f"Loaded configuration from {self.config_file}"
                        )
                        _config_loaded = True
                    else:
                        apgi_logger.debug(
                            f"Configuration reloaded from {self.config_file}"
                        )

            except (
                FileNotFoundError,
                PermissionError,
                yaml.YAMLError,
            ) as e:
                apgi_logger.log_error_with_context(
                    e,
                    {
                        "operation": "load_config",
                        "file": str(self.config_file),
                    },
                )
                apgi_logger.warning(f"Using default configuration due to error: {e}")
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Re-raise JSON and validation errors for tests
                apgi_logger.log_error_with_context(
                    e,
                    {
                        "operation": "load_config",
                        "file": str(self.config_file),
                    },
                )
                raise
        else:
            self._save_default_config()
            apgi_logger.info(f"Created default configuration at {self.config_file}")

    def _validate_config(self, config_data: Dict[str, Any]):
        """Validate configuration against schema."""
        try:
            jsonschema.validate(config_data, self.schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> APGIConfig:
        """Convert dictionary to APGIConfig object."""
        config = APGIConfig()

        if "model" in config_dict:
            self._update_dataclass(config.model, config_dict["model"])
        if "simulation" in config_dict:
            self._update_dataclass(config.simulation, config_dict["simulation"])
        if "logging" in config_dict:
            self._update_dataclass(config.logging, config_dict["logging"])
        if "data" in config_dict:
            self._update_dataclass(config.data, config_dict["data"])
        if "validation" in config_dict:
            self._update_dataclass(config.validation, config_dict["validation"])
        if "batch" in config_dict:
            self._update_dataclass(config.batch, config_dict["batch"])

        return config

    def _update_config(self, config_data: Dict[str, Any]):
        """Update configuration with loaded data."""
        # Import main module to sync top-level config values
        import sys

        main_module = sys.modules.get("main")

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
        if "batch" in config_data:
            self._update_dataclass(self.config.batch, config_data["batch"])

        # Sync top-level metadata to global_config for CLI display
        if main_module and hasattr(main_module, "global_config"):
            if "version" in config_data:
                main_module.global_config["version"] = config_data["version"]
            if "project_name" in config_data:
                main_module.global_config["project_name"] = config_data["project_name"]

    def _update_dataclass(self, dataclass_instance, data: Dict[str, Any]):
        """Update dataclass fields from dictionary with validation."""
        for key, value in data.items():
            if hasattr(dataclass_instance, key):
                # Validate the value before setting
                if self._validate_field_value(dataclass_instance, key, value):
                    setattr(dataclass_instance, key, value)
                else:
                    # Raise ValueError for invalid types
                    raise ValueError(
                        f"Invalid value type for {key}: {value} (type: {type(value).__name__})"
                    )
            else:
                apgi_logger.warning(f"Unknown field {key} in configuration")

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
                "batch": "batch",
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
            apgi_logger.warning(f"Validation error for {field_name}: {e}")
            return False

    def _save_default_config(self):
        """Save default configuration to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
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

        # Dataclass configuration
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

        apgi_logger.info(f"Updated {section}.{parameter} = {converted_value}")
        return True

    def _convert_parameter_value(self, section: str, parameter: str, value: Any) -> Any:
        """Convert parameter value to appropriate type."""
        # Get current value to determine target type
        if hasattr(self.config, "__dataclass_fields__"):
            # Dataclass configuration
            section_obj = getattr(self.config, section)
            current_value = getattr(section_obj, parameter)
        else:
            # Dictionary configuration - convert to dict if needed
            config_dict = cast(Dict[str, Any], self.config)
            if section in config_dict and parameter in config_dict[section]:
                current_value = config_dict[section][parameter]
            else:
                # Default to string if no existing value
                return value

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

    def validate_configuration(
        self, config_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive configuration validation with cross-parameter checks.

        This method performs both schema validation and cross-parameter validation
        to ensure configuration consistency. It generates a detailed validation
        report with error messages and suggestions.

        Args:
            config_data: Configuration data to validate. If None, validates current config.

        Returns:
            Validation report with structure:
            {
                "valid": bool,
                "schema_errors": list of schema validation errors,
                "cross_parameter_errors": list of cross-parameter consistency errors,
                "warnings": list of warnings,
                "suggestions": list of suggested fixes,
                "summary": str
            }
        """
        if config_data is None:
            config_data = asdict(self.config)

        report: Dict[str, Any] = {
            "valid": True,
            "schema_errors": [],
            "cross_parameter_errors": [],
            "warnings": [],
            "suggestions": [],
            "summary": "",
        }

        # 1. Schema validation
        try:
            jsonschema.validate(config_data, self.schema)
        except jsonschema.ValidationError as e:
            report["schema_errors"].append(
                {
                    "message": e.message,
                    "path": list(e.path),
                    "validator": e.validator,
                    "validator_value": e.validator_value,
                }
            )
            report["valid"] = False
        except jsonschema.SchemaError as e:
            report["schema_errors"].append(
                {
                    "message": f"Schema error: {e.message}",
                }
            )
            report["valid"] = False

        # 2. Cross-parameter validation
        model = config_data.get("model", {})
        simulation = config_data.get("simulation", {})
        validation = config_data.get("validation", {})

        # Timescale consistency: tau_S should be much smaller than tau_theta
        tau_S = model.get("tau_S")
        tau_theta = model.get("tau_theta")
        if tau_S is not None and tau_theta is not None:
            if tau_S >= tau_theta:
                report["cross_parameter_errors"].append(
                    {
                        "type": "timescale_inconsistency",
                        "message": f"tau_S ({tau_S}s) should be much smaller than tau_theta ({tau_theta}s)",
                        "parameters": ["model.tau_S", "model.tau_theta"],
                        "severity": "error",
                    }
                )
                report["suggestions"].append(
                    "Set tau_S < tau_theta (suggested: tau_S = 0.5s, tau_theta = 30s)"
                )
                report["valid"] = False
            elif tau_S * 10 > tau_theta:
                report["warnings"].append(
                    {
                        "type": "timescale_ratio",
                        "message": f"tau_S ({tau_S}s) is close to tau_theta ({tau_theta}s). "
                        f"Recommended: tau_theta >= 10 * tau_S for proper separation",
                    }
                )

        # Noise parameters should be reasonable relative to signal
        sigma_S = model.get("sigma_S")
        theta_0 = model.get("theta_0")
        if sigma_S is not None and theta_0 is not None:
            if sigma_S > theta_0 * 0.5:
                report["warnings"].append(
                    {
                        "type": "high_noise",
                        "message": f"sigma_S ({sigma_S}) is high relative to theta_0 ({theta_0}). "
                        f"This may cause unstable simulations.",
                    }
                )

        # Reset fraction validation
        rho = model.get("rho")
        if rho is not None:
            if rho < 0.3:
                report["warnings"].append(
                    {
                        "type": "low_reset_fraction",
                        "message": f"rho ({rho}) is very low. Threshold will reset almost completely.",
                    }
                )
            elif rho > 0.9:
                report["warnings"].append(
                    {
                        "type": "high_reset_fraction",
                        "message": f"rho ({rho}) is very high. Threshold will barely reset.",
                    }
                )

        # Sensitivity parameter consistency
        gamma_M = model.get("gamma_M")
        gamma_A = model.get("gamma_A")
        if gamma_M is not None and gamma_A is not None:
            if abs(gamma_M) < 0.1 and abs(gamma_A) < 0.1:
                report["warnings"].append(
                    {
                        "type": "low_sensitivity",
                        "message": "Both gamma_M and gamma_A are close to zero. "
                        "Metabolic and arousal modulation will be minimal.",
                    }
                )

        # Simulation parameters validation
        dt = simulation.get("default_dt")
        duration = simulation.get("duration", model.get("duration"))
        if dt is not None and duration is not None:
            steps = duration / dt
            if steps < 100:
                report["warnings"].append(
                    {
                        "type": "few_simulation_steps",
                        "message": f"Simulation will only run {int(steps)} steps. "
                        f"Consider increasing duration or decreasing dt for better accuracy.",
                    }
                )
            elif steps > 100000:
                report["warnings"].append(
                    {
                        "type": "many_simulation_steps",
                        "message": f"Simulation will run {int(steps)} steps. "
                        f"This may be computationally expensive.",
                    }
                )

        # Cross-validation consistency
        cv_folds = validation.get("cv_folds")
        enable_cv = validation.get("enable_cross_validation")
        if enable_cv and cv_folds is not None:
            if cv_folds < 2:
                report["cross_parameter_errors"].append(
                    {
                        "type": "invalid_cv_folds",
                        "message": f"cv_folds ({cv_folds}) must be >= 2 for cross-validation",
                        "parameters": ["validation.cv_folds"],
                        "severity": "error",
                    }
                )
                report["valid"] = False
                report["suggestions"].append("Set cv_folds >= 2 (suggested: 5 or 10)")

        # Generate summary
        total_errors = len(report["schema_errors"]) + len(
            report["cross_parameter_errors"]
        )
        total_warnings = len(report["warnings"])

        if total_errors == 0 and total_warnings == 0:
            report["summary"] = "Configuration is valid."
        elif total_errors == 0:
            report["summary"] = (
                f"Configuration is valid with {total_warnings} warning(s)."
            )
        else:
            report["summary"] = (
                f"Configuration has {total_errors} error(s) and {total_warnings} warning(s)."
            )

        return report

    def generate_validation_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a detailed validation report and optionally save to file.

        Args:
            output_file: Optional path to save the report. If None, returns report as string.

        Returns:
            Formatted validation report string.
        """
        report = self.validate_configuration()

        lines = [
            "=" * 60,
            "APGI Configuration Validation Report",
            "=" * 60,
            "",
            f"Status: {'✓ VALID' if report['valid'] else '✗ INVALID'}",
            f"Summary: {report['summary']}",
            "",
        ]

        if report["schema_errors"]:
            lines.extend(
                [
                    "Schema Validation Errors:",
                    "-" * 40,
                ]
            )
            for i, error in enumerate(report["schema_errors"], 1):
                lines.append(f"  {i}. {error['message']}")
                if error.get("path"):
                    lines.append(
                        f"     Path: {'.'.join(str(p) for p in error['path'])}"
                    )
            lines.append("")

        if report["cross_parameter_errors"]:
            lines.extend(
                [
                    "Cross-Parameter Validation Errors:",
                    "-" * 40,
                ]
            )
            for i, error in enumerate(report["cross_parameter_errors"], 1):
                lines.append(f"  {i}. {error['message']}")
                lines.append(f"     Parameters: {', '.join(error['parameters'])}")
            lines.append("")

        if report["warnings"]:
            lines.extend(
                [
                    "Warnings:",
                    "-" * 40,
                ]
            )
            for i, warning in enumerate(report["warnings"], 1):
                lines.append(f"  {i}. {warning['message']}")
            lines.append("")

        if report["suggestions"]:
            lines.extend(
                [
                    "Suggested Fixes:",
                    "-" * 40,
                ]
            )
            for i, suggestion in enumerate(report["suggestions"], 1):
                lines.append(f"  {i}. {suggestion}")
            lines.append("")

        lines.extend(
            [
                "=" * 60,
                f"Report generated at: {datetime.now().isoformat()}",
                "=" * 60,
            ]
        )

        report_text = "\n".join(lines)

        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            apgi_logger.info(f"Validation report saved to: {output_file}")

        return report_text

    def fix_common_issues(
        self, config_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Attempt to automatically fix common configuration issues.

        Args:
            config_data: Configuration data to fix. If None, fixes current config.

        Returns:
            Fixed configuration data.
        """
        if config_data is None:
            config_data = asdict(self.config)

        fixed = config_data.copy()
        fixes_applied = []

        # Fix 1: Ensure tau_S < tau_theta
        model = fixed.get("model", {})
        tau_S = model.get("tau_S")
        tau_theta = model.get("tau_theta")
        if tau_S is not None and tau_theta is not None and tau_S >= tau_theta:
            fixed["model"]["tau_S"] = tau_theta / 10
            fixes_applied.append(
                f"Fixed: tau_S reduced from {tau_S} to {tau_theta / 10}"
            )

        # Fix 2: Ensure cv_folds >= 2
        validation = fixed.get("validation", {})
        cv_folds = validation.get("cv_folds")
        if cv_folds is not None and cv_folds < 2:
            fixed["validation"]["cv_folds"] = 5
            fixes_applied.append(f"Fixed: cv_folds increased from {cv_folds} to 5")

        # Fix 3: Set reasonable defaults for missing critical parameters
        if "model" not in fixed:
            fixed["model"] = {}
        defaults = {
            "tau_S": 0.5,
            "tau_theta": 30.0,
            "theta_0": 0.5,
            "alpha": 10.0,
            "rho": 0.7,
        }
        for param, default_val in defaults.items():
            if param not in fixed["model"]:
                fixed["model"][param] = default_val
                fixes_applied.append(
                    f"Added missing parameter: model.{param} = {default_val}"
                )

        if fixes_applied:
            apgi_logger.info(f"Applied {len(fixes_applied)} automatic fixes:")
            for fix in fixes_applied:
                apgi_logger.info(f"  - {fix}")

        return fixed

    def reset_to_defaults(self, section: Optional[str] = None):
        """Reset configuration to defaults."""
        if section is None:
            self.config = APGIConfig()
            apgi_logger.info("Reset all configuration to defaults")
        else:
            if hasattr(self.config, section):
                default_section = getattr(APGIConfig(), section)
                setattr(self.config, section, default_section)
                apgi_logger.info(f"Reset {section} configuration to defaults")
            else:
                raise ValueError(f"Unknown configuration section: {section}")

    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file."""
        save_path = file_path or self.config_file
        save_path = Path(save_path)

        # Handle both dataclass and dictionary configs
        if hasattr(self.config, "__dataclass_fields__"):
            config_dict = asdict(self.config)
        else:
            config_dict = cast(Dict[str, Any], self.config)

        with open(save_path, "w") as f:
            if save_path.suffix.lower() == ".yaml":
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif save_path.suffix.lower() == ".json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {save_path.suffix}")

        apgi_logger.info(f"Configuration saved to {save_path}")

    # Configuration Profiles and Versioning
    def create_profile(
        self,
        name: str,
        description: str,
        category: str,
        tags: List[str] = None,
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
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        with open(profile_file, "w") as f:
            yaml.dump(asdict(profile), f, default_flow_style=False, indent=2)

        apgi_logger.info(f"Created configuration profile: {name}")
        return str(profile_file)

    def load_profile(self, name: str) -> bool:
        """Load a configuration profile."""
        if not PROFILES_DIR.exists():
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        profile_file = PROFILES_DIR / f"{name}.yaml"
        if not profile_file.exists():
            apgi_logger.error(f"Profile not found: {name}")
            return False

        try:
            with open(profile_file, "r") as f:
                profile_data = yaml.safe_load(f)

            # Create version before loading
            self.create_version(f"Loaded profile: {name}")

            # Apply profile parameters
            self._update_dataclass(self.config, profile_data["parameters"])
            apgi_logger.info(f"Loaded configuration profile: {name}")
            return True

        except (
            FileNotFoundError,
            PermissionError,
            yaml.YAMLError,
            ValueError,
            KeyError,
            AttributeError,
        ) as e:
            apgi_logger.error(f"Error loading profile {name}: {e}")
            return False

    def list_profiles(self, category: str = None) -> List[Dict[str, Any]]:
        """List available configuration profiles."""
        if not PROFILES_DIR.exists():
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        profiles = []

        for profile_file in PROFILES_DIR.glob("*.yaml"):
            try:
                with open(profile_file, "r") as f:
                    profile_data = yaml.safe_load(f)

                if category is None or profile_data.get("category") == category:
                    # Extract validation status from empirical_validation field
                    empirical_validation = profile_data.get("empirical_validation", {})
                    profiles.append(
                        {
                            "name": profile_data.get("name"),
                            "description": profile_data.get("description"),
                            "category": profile_data.get("category"),
                            "created_at": profile_data.get("created_at"),
                            "version": profile_data.get("version"),
                            "tags": profile_data.get("tags", []),
                            "data_source": profile_data.get("data_source", "synthetic"),
                            "validation_status": empirical_validation.get(
                                "status", "UNKNOWN"
                            ),
                            "is_empirically_validated": (
                                profile_data.get("data_source") == "empirical"
                                and empirical_validation.get("status") == "COMPLETE"
                            ),
                        }
                    )
            except (
                FileNotFoundError,
                PermissionError,
                yaml.YAMLError,
                KeyError,
            ) as e:
                apgi_logger.warning(f"Error reading profile {profile_file}: {e}")

        return profiles

    def delete_profile(self, name: str) -> bool:
        """Delete a configuration profile."""
        if not PROFILES_DIR.exists():
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        profile_file = PROFILES_DIR / f"{name}.yaml"
        if profile_file.exists():
            profile_file.unlink()
            apgi_logger.info(f"Deleted configuration profile: {name}")
            return True
        return False

    def get_profile_validation_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed validation status for a clinical or research profile.

        Args:
            name: Profile name

        Returns:
            Dict with validation status details, or None if profile not found
        """
        profile_file = PROFILES_DIR / f"{name}.yaml"
        if not profile_file.exists():
            return None

        try:
            with open(profile_file, "r") as f:
                profile_data = yaml.safe_load(f)

            empirical_validation = profile_data.get("empirical_validation", {})
            return {
                "name": profile_data.get("name"),
                "data_source": profile_data.get("data_source", "synthetic"),
                "validation_status": empirical_validation.get("status", "UNKNOWN"),
                "citations": empirical_validation.get("citations", []),
                "pending_datasets": empirical_validation.get("pending_datasets", []),
                "is_empirically_validated": (
                    profile_data.get("data_source") == "empirical"
                    and empirical_validation.get("status") == "COMPLETE"
                ),
                "parameter_origin": profile_data.get("metadata", {}).get(
                    "parameter_origin", "unknown"
                ),
                "validation_level": profile_data.get("metadata", {}).get(
                    "validation_level", "unknown"
                ),
            }
        except (
            FileNotFoundError,
            PermissionError,
            yaml.YAMLError,
            KeyError,
        ) as e:
            apgi_logger.warning(f"Error reading profile {profile_file}: {e}")
            return None

    def list_pending_empirical_profiles(self) -> List[Dict[str, Any]]:
        """List all profiles awaiting empirical validation.

        Returns:
            List of profiles with SYNTHETIC_PENDING_EMPIRICAL status
        """
        all_profiles = self.list_profiles()
        pending_profiles = []

        for profile_summary in all_profiles:
            # Get full validation status
            full_status = self.get_profile_validation_status(profile_summary["name"])
            if (
                full_status
                and full_status["validation_status"] == "SYNTHETIC_PENDING_EMPIRICAL"
            ):
                pending_profiles.append(full_status)

        return pending_profiles

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
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        version_data = {"version": asdict(version), "config": config_dict}

        with open(version_file, "w") as f:
            json.dump(version_data, f, indent=2)

        apgi_logger.info(f"Created configuration version: {version.version_id}")
        return version.version_id

    def list_versions(self) -> List[Dict[str, Any]]:
        """List available configuration versions."""
        if not VERSIONS_DIR.exists():
            VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
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
                apgi_logger.warning(f"Error reading version {version_file}: {e}")

        return versions

    def restore_version(self, version_id: str) -> bool:
        """Restore configuration from a version snapshot."""
        if not VERSIONS_DIR.exists():
            VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        version_file = VERSIONS_DIR / f"{version_id}.json"
        if not version_file.exists():
            apgi_logger.error(f"Version not found: {version_id}")
            return False

        try:
            with open(version_file, "r") as f:
                version_data = json.load(f)

            # Create version before restoring
            self.create_version(f"Restored from version: {version_id}")

            # Restore configuration
            self._update_dataclass(self.config, version_data["config"])
            apgi_logger.info(f"Restored configuration version: {version_id}")
            return True

        except (
            FileNotFoundError,
            PermissionError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
        ) as e:
            apgi_logger.error(f"Error restoring version {version_id}: {e}")
            return False

    def compare_configs(
        self, config1: Dict[str, Any], config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two configurations and return differences."""
        differences: Dict[str, Any] = {"added": [], "removed": [], "modified": []}

        def deep_compare(dict1, dict2, path=""):
            for key in dict1:
                current_path = f"{path}.{key}" if path else key
                if key not in dict2:
                    differences["removed"].append(current_path)
                elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    deep_compare(dict1[key], dict2[key], current_path)
                elif dict1[key] != dict2[key]:
                    differences["modified"].append(
                        {
                            "path": current_path,
                            "old": dict1[key],
                            "new": dict2[key],
                        }
                    )

            for key in dict2:
                current_path = f"{path}.{key}" if path else key
                if key not in dict1:
                    differences["added"].append(current_path)

        deep_compare(config1, config2)
        return differences

    def initialize_default_profiles(self):
        """Initialize default configuration profiles."""
        if not PROFILES_DIR.exists():
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
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
                    "validation": {
                        "enable_cross_validation": True,
                        "cv_folds": 10,
                    },
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

                apgi_logger.info(f"Created default profile: {profile_data['name']}")

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

        apgi_logger.info(f"Configuration template exported to {file_path}")

    def get_environment_override(self, parameter: str) -> Optional[str]:
        """Get environment variable override for configuration parameter."""
        env_var = f"APGI_{parameter.upper()}"
        return os.getenv(env_var)

    def apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        # Load whitelisted environment variables from .env file
        env_file = Path(".env")
        if env_file.exists():
            try:
                # Read .env file manually and only load whitelisted variables
                whitelisted_vars = {}
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, value = line.split("=", 1)
                            key = key.strip()
                            value = value.strip()
                            # Remove quotes if present
                            if (value.startswith('"') and value.endswith('"')) or (
                                value.startswith("'") and value.endswith("'")
                            ):
                                value = value[1:-1]
                            whitelisted_vars[key] = value

                # Only set whitelisted environment variables
                env_mappings = {
                    "APGI_LOG_LEVEL": ("logging", "level"),
                    "APGI_ENABLE_PLOTS": ("simulation", "enable_plots"),
                    "APGI_DATA_DIR": ("data", "default_data_dir"),
                    "APGI_TAU_S": ("model", "tau_S"),
                    "APGI_TAU_THETA": ("model", "tau_theta"),
                    "APGI_THETA_0": ("model", "theta_0"),
                    "APGI_ALPHA": ("model", "alpha"),
                }

                for env_var, (section, param) in env_mappings.items():
                    value = whitelisted_vars.get(env_var) or os.getenv(env_var)
                    if value is not None:
                        # Convert string values to appropriate types
                        if value.lower() in ["true", "false"]:
                            value = value.lower() == "true"
                        else:
                            try:
                                # Try to convert to float (handles negatives and decimals)
                                value = float(value)
                                # If it's an integer, convert to int
                                if value.is_integer():
                                    value = int(value)
                            except ValueError:
                                # Not a number, keep as string
                                pass

                        self.set_parameter(section, param, value)
                        apgi_logger.info(
                            f"Applied environment override: {env_var} -> {section}.{param} = {value}"
                        )
            except (
                FileNotFoundError,
                PermissionError,
                UnicodeDecodeError,
            ) as e:
                apgi_logger.warning(f"Could not read .env file: {e}")
                # Fall back to regular environment variables
                self._apply_env_mappings_from_system()


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
    ) -> str:
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

        apgi_logger.info(f"Created configuration profile: {name}")
        return str(profile_path)

    def load_profile(self, name: str) -> bool:
        """Load a configuration profile."""
        profile_path = PROFILES_DIR / f"{name}.yaml"

        if not profile_path.exists():
            apgi_logger.error(f"Profile not found: {name}")
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

            apgi_logger.info(f"Loaded configuration profile: {name}")
            return True

        except (
            FileNotFoundError,
            PermissionError,
            yaml.YAMLError,
            ValueError,
            KeyError,
            TypeError,
        ) as e:
            apgi_logger.error(f"Error loading profile {name}: {e}")
            return False

    def list_profiles(self, category: Optional[str] = None) -> list[dict[str, Any]]:
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

            except (
                FileNotFoundError,
                PermissionError,
                yaml.YAMLError,
                KeyError,
                TypeError,
            ) as e:
                apgi_logger.warning(f"Error reading profile {profile_file}: {e}")

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
                    {
                        "path": current_path,
                        "type": "added",
                        "value": dict2[key],
                    }
                )
            elif key not in dict2:
                differences.append(
                    {
                        "path": current_path,
                        "type": "removed",
                        "value": dict1[key],
                    }
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
        value_matches: float = 0.0

        for key in matching_keys:
            if dict1[key] == dict2[key]:
                value_matches += 1.0
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
            apgi_logger.warning("No previous profile to rollback to")
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
            apgi_logger.error(f"Profile not found: {name}")
            return False

        try:
            profile_path.unlink()
            apgi_logger.info(f"Deleted configuration profile: {name}")
            return True

        except (FileNotFoundError, PermissionError, OSError) as e:
            apgi_logger.error(f"Error deleting profile {name}: {e}")
            return False

    def export_profile(self, name: str, format: str = "yaml") -> Optional[str]:
        """Export profile in specified format."""
        profile_path = PROFILES_DIR / f"{name}.yaml"

        if not profile_path.exists():
            apgi_logger.error(f"Profile not found: {name}")
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
                apgi_logger.error(f"Unsupported export format: {format}")
                return None

        except (
            FileNotFoundError,
            PermissionError,
            yaml.YAMLError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
        ) as e:
            apgi_logger.error(f"Error exporting profile {name}: {e}")
            return None

    def import_profile(self, file_path: str, name: Optional[str] = None) -> bool:
        """Import profile from file."""
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            apgi_logger.error(f"File not found: {file_path_obj}")
            return False

        try:
            with open(file_path_obj, "r") as f:
                if file_path_obj.suffix.lower() == ".json":
                    profile_data = json.load(f)
                else:
                    profile_data = yaml.safe_load(f)

            # Use provided name or extract from file
            if name:
                profile_data["name"] = name
            elif "name" not in profile_data:
                profile_data["name"] = file_path_obj.stem

            # Save as profile
            profile = ConfigProfile(**profile_data)
            profile_path = PROFILES_DIR / f"{profile.name}.yaml"

            with open(profile_path, "w") as f:
                yaml.dump(asdict(profile), f, default_flow_style=False)

            apgi_logger.info(f"Imported configuration profile: {profile.name}")
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
            apgi_logger.error(f"Error importing profile from {file_path}: {e}")
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
            built_in_profiles = [
                "adhd",
                "anxiety-disorder",
                "research-default",
            ]
            if profile.name not in built_in_profiles:
                # For custom profiles, require all sections
                required_sections = [
                    "model",
                    "simulation",
                    "logging",
                    "data",
                    "validation",
                ]
                for section in required_sections:
                    if section not in profile.parameters:
                        validation_errors.append(f"Missing required section: {section}")
            else:
                # For built-in profiles, only require the sections they actually have
                if not profile.parameters:
                    validation_errors.append("Profile parameters are required")

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
            category = profile["category"]
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
        apgi_logger.error(f"Error creating profile: {e}")
        return False


def switch_config_profile(name: str) -> bool:
    """Switch to a configuration profile."""
    return enhanced_config_manager.switch_to_profile(name)


def list_config_profiles(
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List available configuration profiles."""
    profiles = enhanced_config_manager.list_profiles(category)
    result: List[Dict[str, Any]] = []
    for p in profiles:
        result.append(
            {
                "name": p["name"],
                "description": p["description"],
                "category": p["category"],
                "version": p["version"],
                "author": p.get("author"),
                "tags": p["tags"],
            }
        )
    return result


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
    except (ValueError, KeyError, AttributeError, TypeError):
        return False


def reset_config(section: Optional[str] = None):
    """Reset configuration to defaults."""
    config_manager.reset_to_defaults(section)


def get_batch_config():
    """Get batch processing configuration."""
    return config_manager.get_config("batch")


def set_batch_parameter(parameter: str, value: Any):
    """Set a batch processing parameter."""
    try:
        config_manager.set_parameter("batch", parameter, value)
        return True
    except (ValueError, KeyError, AttributeError, TypeError):
        return False


def get_max_workers() -> int:
    """Get optimal number of workers based on batch configuration and system."""
    batch_config = config_manager.get_config("batch")
    max_workers: int = getattr(batch_config, "max_workers", 4)

    if getattr(batch_config, "auto_scale", False):
        # Scale based on CPU count
        cpu_count = os.cpu_count() or 4
        max_workers = min(max_workers, cpu_count)

        # Reduce workers if system is under load (simple heuristic)
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                max_workers = max(1, max_workers // 2)
        except ImportError:
            pass  # psutil not available, use configured value

    return max_workers


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
