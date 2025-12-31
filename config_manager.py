"""
APGI Theory Framework - Configuration Management
================================================

Comprehensive configuration system with:
- YAML and JSON configuration files
- Environment variable support
- Parameter validation
- Default configuration management
- Runtime configuration updates
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import jsonschema
from logging_config import apgi_logger, log_error

# Project root directory
PROJECT_ROOT = Path(__file__).parent
CONFIG_DIR = PROJECT_ROOT / 'config'

# Ensure config directory exists
CONFIG_DIR.mkdir(exist_ok=True)


@dataclass
class ModelParameters:
    """Default model parameters for APGI formal model."""
    # Timescales (in seconds)
    tau_S: float = 0.5      # 500 ms (Range: 100-1000ms)
    tau_theta: float = 30.0 # 30 s   (Range: 5-60s)
    
    # Threshold parameters
    theta_0: float = 0.5    # Baseline threshold (Range: 0.1-1.0 AU)
    
    # Sigmoid parameters
    alpha: float = 10.0     # Sharpness (Range: 1-15)
    
    # Sensitivities
    gamma_M: float = -0.3   # Metabolic sensitivity (Range: -0.5 to 0.5)
    gamma_A: float = 0.1    # Arousal sensitivity (Range: -0.3 to 0.3)
    
    # Reset dynamics
    rho: float = 0.7        # Reset fraction (Range: 0.3-0.9)
    
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
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load JSON schema for configuration validation."""
        schema = {
            "type": "object",
            "properties": {
                "model": {
                    "type": "object",
                    "properties": {
                        "tau_S": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                        "tau_theta": {"type": "number", "minimum": 5.0, "maximum": 60.0},
                        "theta_0": {"type": "number", "minimum": 0.1, "maximum": 1.0},
                        "alpha": {"type": "number", "minimum": 1.0, "maximum": 15.0},
                        "gamma_M": {"type": "number", "minimum": -0.5, "maximum": 0.5},
                        "gamma_A": {"type": "number", "minimum": -0.3, "maximum": 0.3},
                        "rho": {"type": "number", "minimum": 0.3, "maximum": 0.9},
                        "sigma_S": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "sigma_theta": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    }
                },
                "simulation": {
                    "type": "object",
                    "properties": {
                        "default_steps": {"type": "integer", "minimum": 1, "maximum": 1000000},
                        "default_dt": {"type": "number", "minimum": 0.001, "maximum": 1.0},
                        "max_steps": {"type": "integer", "minimum": 1},
                        "enable_plots": {"type": "boolean"},
                        "plot_format": {"type": "string", "enum": ["png", "jpg", "svg", "pdf"]},
                        "plot_dpi": {"type": "integer", "minimum": 50, "maximum": 300},
                        "save_results": {"type": "boolean"},
                        "results_format": {"type": "string", "enum": ["csv", "json", "pkl"]}
                    }
                },
                "logging": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                        "enable_console": {"type": "boolean"},
                        "log_rotation": {"type": "string"},
                        "log_retention": {"type": "string"},
                        "enable_performance_logging": {"type": "boolean"},
                        "enable_structured_logging": {"type": "boolean"}
                    }
                }
            }
        }
        return schema
    
    def _load_environment(self):
        """Load environment variables from .env file."""
        env_file = PROJECT_ROOT / '.env'
        if env_file.exists():
            load_dotenv(env_file)
    
    def _load_config(self):
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    if self.config_file.suffix.lower() == '.yaml':
                        config_data = yaml.safe_load(f)
                    elif self.config_file.suffix.lower() == '.json':
                        config_data = json.load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {self.config_file.suffix}")
                
                # Validate configuration
                self._validate_config(config_data)
                
                # Update configuration
                self._update_config(config_data)
                
                apgi_logger.logger.info(f"Loaded configuration from {self.config_file}")
                
            except Exception as e:
                apgi_logger.log_error_with_context(e, {"operation": "load_config", "file": str(self.config_file)})
                apgi_logger.logger.warning(f"Using default configuration due to error: {e}")
        else:
            self._save_default_config()
            apgi_logger.logger.info(f"Created default configuration at {self.config_file}")
    
    def _validate_config(self, config_data: Dict[str, Any]):
        """Validate configuration against schema."""
        try:
            jsonschema.validate(config_data, self.schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")
    
    def _update_config(self, config_data: Dict[str, Any]):
        """Update configuration with loaded data."""
        if 'model' in config_data:
            self._update_dataclass(self.config.model, config_data['model'])
        if 'simulation' in config_data:
            self._update_dataclass(self.config.simulation, config_data['simulation'])
        if 'logging' in config_data:
            self._update_dataclass(self.config.logging, config_data['logging'])
        if 'data' in config_data:
            self._update_dataclass(self.config.data, config_data['data'])
        if 'validation' in config_data:
            self._update_dataclass(self.config.validation, config_data['validation'])
    
    def _update_dataclass(self, dataclass_instance, data: Dict[str, Any]):
        """Update dataclass fields from dictionary."""
        for key, value in data.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def _save_default_config(self):
        """Save default configuration to file."""
        config_dict = asdict(self.config)
        
        with open(self.config_file, 'w') as f:
            if self.config_file.suffix.lower() == '.yaml':
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif self.config_file.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
    
    def get_config(self, section: Optional[str] = None) -> Union[APGIConfig, Any]:
        """Get configuration section or entire config."""
        if section is None:
            return self.config
        elif hasattr(self.config, section):
            return getattr(self.config, section)
        else:
            raise ValueError(f"Unknown configuration section: {section}")
    
    def set_parameter(self, section: str, parameter: str, value: Any):
        """Set a specific configuration parameter."""
        if not hasattr(self.config, section):
            raise ValueError(f"Unknown configuration section: {section}")
        
        section_obj = getattr(self.config, section)
        if not hasattr(section_obj, parameter):
            raise ValueError(f"Unknown parameter: {parameter} in section: {section}")
        
        # Validate parameter value
        self._validate_parameter(section, parameter, value)
        
        setattr(section_obj, parameter, value)
        apgi_logger.logger.info(f"Updated {section}.{parameter} = {value}")
    
    def _validate_parameter(self, section: str, parameter: str, value: Any):
        """Validate individual parameter value."""
        # Get schema for this parameter
        section_schema = self.schema.get('properties', {}).get(section, {})
        param_schema = section_schema.get('properties', {}).get(parameter, {})
        
        if param_schema:
            try:
                jsonschema.validate(value, param_schema)
            except jsonschema.ValidationError as e:
                raise ValueError(f"Invalid value for {section}.{parameter}: {e.message}")
    
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
        
        with open(save_path, 'w') as f:
            if save_path.suffix.lower() == '.yaml':
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif save_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {save_path.suffix}")
        
        apgi_logger.logger.info(f"Configuration saved to {save_path}")
    
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
            with open(file_path, 'w') as f:
                f.write(template_content)
        else:
            # JSON format (no comments)
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        apgi_logger.logger.info(f"Configuration template exported to {file_path}")
    
    def get_environment_override(self, parameter: str) -> Optional[str]:
        """Get environment variable override for configuration parameter."""
        env_var = f"APGI_{parameter.upper()}"
        return os.getenv(env_var)
    
    def apply_environment_overrides(self):
        """Apply environment variable overrides to configuration."""
        env_mappings = {
            'APGI_LOG_LEVEL': ('logging', 'level'),
            'APGI_ENABLE_PLOTS': ('simulation', 'enable_plots'),
            'APGI_DATA_DIR': ('data', 'default_data_dir'),
            'APGI_TAU_S': ('model', 'tau_S'),
            'APGI_TAU_THETA': ('model', 'tau_theta'),
            'APGI_THETA_0': ('model', 'theta_0'),
        }
        
        for env_var, (section, param) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.replace('.', '').isdigit():
                    value = float(value) if '.' in value else int(value)
                
                self.set_parameter(section, param, value)
                apgi_logger.logger.info(f"Applied environment override: {env_var} -> {section}.{param} = {value}")


# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions
def get_config(section: Optional[str] = None):
    """Get configuration section or entire config."""
    return config_manager.get_config(section)

def set_parameter(section: str, parameter: str, value: Any):
    """Set a specific configuration parameter."""
    config_manager.set_parameter(section, parameter, value)

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
    set_parameter('model', 'tau_S', 0.8)
    print(f"Updated model tau_S: {config.model.tau_S}")
    
    # Test environment overrides
    os.environ['APGI_LOG_LEVEL'] = 'DEBUG'
    config_manager.apply_environment_overrides()
    print(f"Logging level after env override: {config.logging.level}")
    
    # Export template
    template_file = CONFIG_DIR / "config_template.yaml"
    config_manager.export_config_template(str(template_file))
    print(f"Configuration template exported to: {template_file}")
    
    print("Configuration system test completed successfully!")
