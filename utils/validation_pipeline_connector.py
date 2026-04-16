#!/usr/bin/env python3
"""
Validation Pipeline Connector
==========================

Connects preprocessing pipelines with validation protocols to enable end-to-end
workflow automation for APGI framework.
"""

import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.error_handler import ConfigurationError, DataError, ValidationError

try:
    from utils.preprocessing_pipelines import (
        MultimodalPreprocessingPipeline,
        PreprocessingConfig,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import preprocessing pipelines. Ensure utils.preprocessing_pipelines exists."
    ) from exc

try:
    from utils.sample_data_generator import (
        SampleDataGenerator as SampleDataGeneratorClass,
    )
    from utils.sample_data_generator import generate_sample_multimodal_data
except ImportError as exc:
    raise ImportError(
        "sample_data_generator is required for ValidationPipelineConnector. "
        "Install project dependencies and verify utils/sample_data_generator.py is importable."
    ) from exc

# Configure logging
logger = logging.getLogger(__name__)


class ValidationPipelineConnector:
    """Connects preprocessing pipelines with validation protocols."""

    def __init__(
        self, config: Optional[Union[Dict[str, Any], PreprocessingConfig]] = None
    ):
        """Initialize connector with optional preprocessing configuration."""
        if config is None:
            self.config = PreprocessingConfig()
        elif isinstance(config, dict):
            self.config = PreprocessingConfig(**config)
        else:
            self.config = config

        self.preprocessor = MultimodalPreprocessingPipeline(self.config)
        self.data_generator = SampleDataGeneratorClass()
        self.connection_log: List[Union[str, Dict[str, Any]]] = []
        self._log_lock = threading.Lock()

    def prepare_data_for_validation(
        self,
        validation_protocol: int,
        input_data: Optional[Union[str, Path]] = None,
        use_synthetic: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare data for specific validation protocol."""
        logger.info("Preparing data for Validation Protocol %s", validation_protocol)

        try:
            if use_synthetic or not input_data:
                data = self._generate_protocol_specific_data(validation_protocol, **kwargs)
                with self._log_lock:
                    self.connection_log.append(
                        f"Generated synthetic data for Protocol {validation_protocol}"
                    )
            else:
                data = self._load_and_preprocess_data(input_data, validation_protocol)
                with self._log_lock:
                    self.connection_log.append(
                        f"Loaded and preprocessed {input_data} for Protocol {validation_protocol}"
                    )

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

        except (ValidationError, DataError, FileNotFoundError, RuntimeError) as exc:
            logger.error(
                "Failed to prepare data for Protocol %s: %s",
                validation_protocol,
                exc,
            )
            return {
                "status": "error",
                "protocol": validation_protocol,
                "error": str(exc),
                "metadata": {"source": "failed_preparation"},
            }

    def _generate_protocol_specific_data(self, protocol: int, **kwargs) -> pd.DataFrame:
        """Generate synthetic data specific to validation protocol requirements."""
        generation_map = {
            1: {"n_samples": kwargs.get("n_samples", 1000), "sampling_rate": 100.0, "duration_minutes": 10},
            2: {"n_samples": 2000, "sampling_rate": 200.0, "duration_minutes": 5},
            3: {"n_samples": 1500, "sampling_rate": 50.0, "duration_minutes": 15},
            4: {"n_samples": 800, "sampling_rate": 100.0, "duration_minutes": 8},
            5: {"n_samples": 800, "sampling_rate": 100.0, "duration_minutes": 8},
            6: {"n_samples": 800, "sampling_rate": 100.0, "duration_minutes": 8},
            7: {"n_samples": 1200, "sampling_rate": 100.0, "duration_minutes": 12},
            8: {"n_samples": 1200, "sampling_rate": 100.0, "duration_minutes": 12},
        }
        config = generation_map.get(
            protocol,
            {"n_samples": 1000, "sampling_rate": 100.0, "duration_minutes": 10},
        )
        return generate_sample_multimodal_data(**config)

    def _load_and_preprocess_data(
        self, input_path: Union[str, Path], protocol: int
    ) -> pd.DataFrame:
        """Load and preprocess data for validation protocol."""
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input data file not found: {input_path}")

        preprocessing_result = self.preprocessor.run_complete_pipeline(
            input_path,
            output_dir=f"data_repository/processed/protocol_{protocol}",
            show_progress=True,
        )

        if preprocessing_result["status"] != "success":
            raise DataError(
                f"Preprocessing failed: {preprocessing_result.get('error', 'Unknown error')}",
                data_source=str(input_path),
            )

        processed_file = preprocessing_result["output_file"]
        if processed_file.suffix == ".json":
            with open(processed_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return pd.DataFrame(data)
        return pd.read_csv(processed_file)

    def _validate_protocol_compatibility(
        self, data: pd.DataFrame, protocol: int
    ) -> Dict[str, Any]:
        """Validate that data meets protocol requirements."""
        if not isinstance(data, pd.DataFrame):
            raise ValidationError("Expected pandas DataFrame", data_field="data")

        compatibility: Dict[str, Any] = {
            "valid": True,
            "warnings": [],
            "required_columns": [],
            "missing_columns": [],
        }
        protocol_requirements = {
            1: ["EEG_Cz", "pupil_diameter"],
            2: ["EEG_Cz", "pupil_diameter", "eda"],
            3: ["EEG_Cz", "pupil_diameter", "eda", "heart_rate"],
            4: ["EEG_Cz", "pupil_diameter"],
            5: ["EEG_Cz", "pupil_diameter", "eda"],
            6: ["EEG_Cz", "pupil_diameter", "eda", "heart_rate"],
            7: ["EEG_Cz", "pupil_diameter"],
            8: ["EEG_Cz", "pupil_diameter", "eda"],
            9: ["EEG_Cz", "pupil_diameter"],
            10: ["EEG_Cz", "pupil_diameter"],
            11: ["EEG_Cz", "pupil_diameter", "eda"],
            12: ["EEG_Cz", "pupil_diameter"],
        }

        required_cols = protocol_requirements.get(protocol, ["EEG_Cz", "pupil_diameter"])
        compatibility["required_columns"] = required_cols
        data_columns = list(data.columns)
        missing_cols = [col for col in required_cols if col not in data_columns]
        compatibility["missing_columns"] = missing_cols

        if missing_cols:
            compatibility["valid"] = False
            compatibility["warnings"].append(
                f"Missing required columns: {', '.join(missing_cols)}"
            )

        if len(data) < 100:
            compatibility["warnings"].append("Small dataset size (< 100 samples)")
        elif len(data) < 500:
            compatibility["warnings"].append("Moderate dataset size (< 500 samples)")

        return compatibility


if __name__ == "__main__":
    # Keep simple smoke execution path explicit.
    try:
        connector = ValidationPipelineConnector()
        test_result = connector.prepare_data_for_validation(1, use_synthetic=True)
        print(f"Connector test result: {test_result['status']}")
    except (ImportError, ConfigurationError) as exc:
        raise SystemExit(f"Initialization failed: {exc}") from exc
