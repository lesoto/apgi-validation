#!/usr/bin/env python3
"""
Validation Pipeline Connector
==========================

Connects preprocessing pipelines with validation protocols to enable end-to-end
workflow automation for the APGI framework.
"""

import json
import logging
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

try:
    from utils.preprocessing_pipelines import (
        MultimodalPreprocessingPipeline,
        PreprocessingConfig,
    )
except ImportError:
    # Fallback if utils.preprocessing_pipelines is not available
    import warnings

    warnings.warn(
        "utils.preprocessing_pipelines not available - pipeline connector may be limited",
        ImportWarning,
    )
    MultimodalPreprocessingPipeline = None
    PreprocessingConfig = None
from utils.sample_data_generator import (
    SampleDataGenerator,
    generate_sample_multimodal_data,
)

# Configure logging
logger = logging.getLogger(__name__)


class ValidationPipelineConnector:
    """Connects preprocessing pipelines with validation protocols."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize connector with optional preprocessing configuration."""
        self.config = config or PreprocessingConfig()
        self.preprocessor = MultimodalPreprocessingPipeline(self.config)
        self.data_generator = SampleDataGenerator()
        self.connection_log = []

    def prepare_data_for_validation(
        self,
        validation_protocol: int,
        input_data: Optional[Union[str, Path]] = None,
        use_synthetic: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare data for specific validation protocol.

        Args:
            validation_protocol: Protocol number (1-12)
            input_data: Path to input data file
            use_synthetic: Generate synthetic data if no input provided
            **kwargs: Additional parameters for specific protocols

        Returns:
            Dictionary with prepared data and metadata
        """
        logger.info(f"Preparing data for Validation Protocol {validation_protocol}")

        try:
            # Generate or load data
            if use_synthetic or not input_data:
                data = self._generate_protocol_specific_data(
                    validation_protocol, **kwargs
                )
                self.connection_log.append(
                    f"Generated synthetic data for Protocol {validation_protocol}"
                )
            else:
                data = self._load_and_preprocess_data(input_data, validation_protocol)
                self.connection_log.append(
                    f"Loaded and preprocessed {input_data} for Protocol {validation_protocol}"
                )

            # Validate data compatibility
            compatibility_check = self._validate_protocol_compatibility(
                data, validation_protocol
            )

            return {
                "status": "success",
                "protocol": validation_protocol,
                "data": data,
                "metadata": {
                    "source": "synthetic" if use_synthetic else "file",
                    "compatibility": compatibility_check,
                    "preprocessing_applied": not use_synthetic,
                    "data_shape": data.shape if hasattr(data, "shape") else len(data),
                },
            }

        except Exception as e:
            logger.error(
                f"Failed to prepare data for Protocol {validation_protocol}: {e}"
            )
            return {
                "status": "error",
                "protocol": validation_protocol,
                "error": str(e),
                "metadata": {"source": "failed_preparation"},
            }

    def _generate_protocol_specific_data(self, protocol: int, **kwargs) -> pd.DataFrame:
        """Generate synthetic data specific to validation protocol requirements."""

        if protocol == 1:
            # Protocol 1: Synthetic neural data for ML classification
            n_samples = kwargs.get("n_samples", 1000)
            return generate_sample_multimodal_data(
                n_samples=n_samples, sampling_rate=100.0, duration_minutes=10
            )

        elif protocol == 2:
            # Protocol 2: Time-series data for dynamics validation
            return generate_sample_multimodal_data(
                n_samples=2000, sampling_rate=200.0, duration_minutes=5
            )

        elif protocol == 3:
            # Protocol 3: Agent simulation data
            return generate_sample_multimodal_data(
                n_samples=1500, sampling_rate=50.0, duration_minutes=15
            )

        elif protocol in [4, 5, 6]:
            # Protocols 4-6: Cross-validation data
            return generate_sample_multimodal_data(
                n_samples=800, sampling_rate=100.0, duration_minutes=8
            )

        elif protocol in [7, 8]:
            # Protocols 7-8: Individual differences data
            return generate_sample_multimodal_data(
                n_samples=1200, sampling_rate=100.0, duration_minutes=12
            )

        else:
            # Default for protocols 9-12
            return generate_sample_multimodal_data(
                n_samples=1000, sampling_rate=100.0, duration_minutes=10
            )

    def _load_and_preprocess_data(
        self, input_path: Union[str, Path], protocol: int
    ) -> pd.DataFrame:
        """Load and preprocess data for validation protocol."""
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input data file not found: {input_path}")

        # Run preprocessing pipeline
        preprocessing_result = self.preprocessor.run_complete_pipeline(
            input_path,
            output_dir=f"data/processed/protocol_{protocol}",
            show_progress=True,
        )

        if preprocessing_result["status"] != "success":
            raise RuntimeError(
                f"Preprocessing failed: {preprocessing_result.get('error', 'Unknown error')}"
            )

        # Load processed data
        processed_file = preprocessing_result["output_file"]
        if processed_file.suffix == ".json":
            with open(processed_file, "r") as f:
                data = json.load(f)
                return pd.DataFrame(data)
        else:
            return pd.read_csv(processed_file)

    def _validate_protocol_compatibility(
        self, data: pd.DataFrame, protocol: int
    ) -> Dict[str, Any]:
        """Validate that data meets protocol requirements."""
        compatibility = {
            "valid": True,
            "warnings": [],
            "required_columns": [],
            "missing_columns": [],
        }
        # Define required columns for each protocol
        protocol_requirements = {
            1: ["EEG_Cz", "pupil_diameter"],  # Basic neural data
            2: ["EEG_Cz", "pupil_diameter", "eda"],  # Time-series dynamics
            3: ["EEG_Cz", "pupil_diameter", "eda", "heart_rate"],  # Agent simulation
            4: ["EEG_Cz", "pupil_diameter"],  # Cross-validation
            5: ["EEG_Cz", "pupil_diameter", "eda"],  # Cross-validation
            6: ["EEG_Cz", "pupil_diameter", "eda", "heart_rate"],  # Cross-validation
            7: ["EEG_Cz", "pupil_diameter"],  # Individual differences
            8: ["EEG_Cz", "pupil_diameter", "eda"],  # Individual differences
            9: ["EEG_Cz", "pupil_diameter"],  # Neural signatures
            10: ["EEG_Cz", "pupil_diameter"],  # Behavioral validation
            11: ["EEG_Cz", "pupil_diameter", "eda"],  # Clinical validation
            12: ["EEG_Cz", "pupil_diameter"],  # Meta-analysis
        }

        required_cols = protocol_requirements.get(
            protocol, ["EEG_Cz", "pupil_diameter"]
        )
        compatibility["required_columns"] = required_cols
        # Check for missing columns
        data_columns = list(data.columns) if hasattr(data, "columns") else []
        missing_cols = [col for col in required_cols if col not in data_columns]
        compatibility["missing_columns"] = missing_cols

        if missing_cols:
            compatibility["valid"] = False
            compatibility["warnings"].append(
                f"Missing required columns: {', '.join(missing_cols)}"
            )
        # Check data size
        if hasattr(data, "shape"):
            if len(data) < 100:
                compatibility["warnings"].append("Small dataset size (< 100 samples)")
            elif len(data) < 500:
                compatibility["warnings"].append(
                    "Moderate dataset size (< 500 samples)"
                )

        return compatibility

    def run_validation_with_pipeline(
        self,
        validation_protocol: int,
        input_data: Optional[Union[str, Path]] = None,
        use_synthetic: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run validation protocol with integrated preprocessing pipeline.

        Args:
            validation_protocol: Protocol number (1-12)
            input_data: Path to input data file
            use_synthetic: Generate synthetic data if no input provided
            **kwargs: Additional parameters for validation protocol

        Returns:
            Dictionary with validation results
        """
        logger.info(
            f"Running Validation Protocol {validation_protocol} with integrated pipeline"
        )
        # Prepare data
        data_preparation = self.prepare_data_for_validation(
            validation_protocol, input_data, use_synthetic, **kwargs
        )

        if data_preparation["status"] != "success":
            return {
                "status": "error",
                "protocol": validation_protocol,
                "error": f"Data preparation failed: {data_preparation['error']}",
                "pipeline_metadata": data_preparation["metadata"],
            }

        try:
            # Import and run the validation protocol
            validation_module = self._import_validation_protocol(validation_protocol)
            # Pass prepared data to validation protocol
            validation_result = self._execute_validation_protocol(
                validation_module, data_preparation["data"], **kwargs
            )
            # Combine results
            combined_result = {
                "status": "success",
                "protocol": validation_protocol,
                "validation_result": validation_result,
                "pipeline_metadata": data_preparation["metadata"],
                "connection_log": self.connection_log[-5:],  # Last 5 log entries
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            logger.info(f"Protocol {validation_protocol} completed successfully")
            return combined_result

        except Exception as e:
            logger.error(f"Validation Protocol {validation_protocol} failed: {e}")
            return {
                "status": "error",
                "protocol": validation_protocol,
                "error": str(e),
                "pipeline_metadata": data_preparation["metadata"],
                "connection_log": self.connection_log[-5:],
            }

    def _import_validation_protocol(self, protocol: int):
        """Dynamically import validation protocol module."""
        import importlib.util

        protocol_file = (
            Path(__file__).parent.parent
            / "Validation"
            / f"Validation-Protocol-{protocol}.py"
        )

        if not protocol_file.exists():
            raise FileNotFoundError(
                f"Validation Protocol {protocol} not found: {protocol_file}"
            )

        spec = importlib.util.spec_from_file_location(
            f"validation_protocol_{protocol}", protocol_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return module

    def _execute_validation_protocol(self, module, data: pd.DataFrame, **kwargs):
        """Execute validation protocol with prepared data."""
        # Try to find main validation function or class
        if hasattr(module, "main"):
            # Check if main() accepts data parameter
            sig = inspect.signature(module.main)
            if "data" in sig.parameters:
                return module.main(data=data, **kwargs)
            else:
                # For protocols that don't accept data parameter, save temporarily
                temp_file = Path("temp_validation_data.csv")
                data.to_csv(temp_file, index=False)
                try:
                    result = module.main()
                    return result
                finally:
                    if temp_file.exists():
                        temp_file.unlink()
        elif hasattr(module, "run_validation"):
            # For protocols that use run_validation() without parameters
            # We need to temporarily save data for them to load
            temp_file = Path("temp_validation_data.csv")
            data.to_csv(temp_file, index=False)
            try:
                result = module.run_validation()
                return result
            finally:
                if temp_file.exists():
                    temp_file.unlink()
        else:
            raise AttributeError("No main validation function found in Protocol module")

    def get_connection_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline connections and operations."""
        return {
            "total_connections": len(self.connection_log),
            "recent_logs": self.connection_log[-10:],
            "supported_protocols": list(range(1, 13)),
            "preprocessing_config": {
                "eeg_bandpass_low": self.config.eeg_bandpass_low,
                "eeg_bandpass_high": self.config.eeg_bandpass_high,
                "target_sampling_rate": getattr(
                    self.config, "target_sampling_rate", 100.0
                ),
            },
        }


def main():
    """Demonstrate validation pipeline connector."""
    print("APGI Framework - Validation Pipeline Connector")
    print("=" * 50)
    # Initialize connector
    connector = ValidationPipelineConnector()
    # Test with Protocol 1 using synthetic data
    print("\n1. Testing Protocol 1 with synthetic data...")
    result1 = connector.run_validation_with_pipeline(
        validation_protocol=1, use_synthetic=True, n_samples=500
    )

    print(f"Protocol 1 Status: {result1['status']}")
    if result1["status"] == "success":
        print(f"  Data shape: {result1['pipeline_metadata']['data_shape']}")
        print(
            f"  Compatibility: {result1['pipeline_metadata']['compatibility']['valid']}"
        )
    else:
        print(f"  Error: {result1['error']}")
    # Get connection summary
    summary = connector.get_connection_summary()
    print("\n2. Connection Summary:")
    print(f"  Total connections: {summary['total_connections']}")
    print(f"  Supported protocols: {summary['supported_protocols']}")

    print("\nValidation pipeline connector ready!")


if __name__ == "__main__":
    main()
