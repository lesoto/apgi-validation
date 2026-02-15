#!/usr/bin/env python3
"""
APGI Framework REST API
========================

REST API for running simulations, validations, and accessing results.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, Form, HTTPException, UploadFile
from pydantic import BaseModel

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils import data_validation, sample_data_generator
from utils.config_manager import ConfigManager

app = FastAPI(
    title="APGI Framework API",
    description="REST API for the Active Predictive Processing and Ignition (APGI) framework",
    version="1.0.0",
)

# Initialize components
config_manager = ConfigManager()

# Data repository paths
DATA_REPO = PROJECT_ROOT / "data_repository"
RAW_DATA_DIR = DATA_REPO / "raw_data"
PROCESSED_DATA_DIR = DATA_REPO / "processed_data"
METADATA_DIR = DATA_REPO / "metadata"


class SimulationRequest(BaseModel):
    """Request model for simulation."""

    tau_S: float = 0.5
    tau_theta: float = 30.0
    theta_0: float = 0.5
    alpha: float = 5.0
    steps: int = 1000
    dt: float = 0.01


class ValidationRequest(BaseModel):
    """Request model for validation."""

    protocol: str = "protocol_1"
    parameters: Optional[Dict] = None


class DataGenerationRequest(BaseModel):
    """Request model for data generation."""

    n_samples: int = 1000
    sampling_rate: float = 100.0
    duration_minutes: int = 1
    include_artifacts: bool = True


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "APGI Framework API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}


@app.post("/simulation/run")
async def run_simulation(request: SimulationRequest):
    """Run a simulation with given parameters."""
    try:
        # Create generator
        generator = sample_data_generator.SampleDataGenerator(
            sampling_rate=int(1 / request.dt), duration=int(request.steps * request.dt)
        )

        # Generate data (placeholder - would run actual APGI simulation)
        eeg_signal, p300_events = generator.generate_eeg_data()

        return {
            "status": "completed",
            "parameters": request.dict(),
            "results": {
                "signal_length": len(eeg_signal),
                "p300_events": len(p300_events),
                "eeg_sample": eeg_signal[:100].tolist(),  # First 100 samples
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@app.post("/data/generate")
async def generate_data(request: DataGenerationRequest):
    """Generate sample multimodal data."""
    try:
        data_df = sample_data_generator.generate_sample_multimodal_data(
            n_samples=request.n_samples,
            sampling_rate=request.sampling_rate,
            duration_minutes=request.duration_minutes,
            include_artifacts=request.include_artifacts,
        )

        return {
            "status": "completed",
            "data_shape": data_df.shape,
            "columns": list(data_df.columns),
            "sample": data_df.head(5).to_dict("records"),  # First 5 rows
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")


@app.post("/data/validate")
async def validate_data_file(file_path: str):
    """Validate a data file."""
    try:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise HTTPException(status_code=404, detail="File not found")

        validator = data_validation.DataValidator()

        # Validate format
        format_result = validator.validate_file_format(file_path_obj)

        if not format_result["is_readable"]:
            raise HTTPException(status_code=400, detail="File not readable")

        # Load data and assess quality
        if file_path_obj.suffix.lower() == ".csv":
            import pandas as pd

            data_df = pd.read_csv(file_path_obj)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        quality_report = validator.validate_data_quality(data_df)

        return {
            "status": "completed",
            "file_info": format_result,
            "quality_report": quality_report,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.get("/config")
async def get_config(section: Optional[str] = None):
    """Get current configuration."""
    try:
        config = config_manager.get_config(section)
        if section:
            return {section: config.__dict__ if hasattr(config, "__dict__") else config}
        else:
            return config.__dict__
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Config retrieval failed: {str(e)}"
        )


@app.put("/config/{section}/{parameter}")
async def set_config_parameter(section: str, parameter: str, value: str):
    """Set a configuration parameter."""
    try:
        # Convert value to appropriate type
        config_manager.set_parameter(section, parameter, value)
        return {
            "status": "updated",
            "section": section,
            "parameter": parameter,
            "value": value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config update failed: {str(e)}")


@app.post("/config/profile")
async def create_config_profile(name: str, description: str, category: str = "custom"):
    """Create a new configuration profile."""
    try:
        profile_path = config_manager.create_profile(name, description, category)
        return {"status": "created", "profile": name, "path": str(profile_path)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Profile creation failed: {str(e)}"
        )


@app.get("/config/profiles")
async def list_config_profiles(category: Optional[str] = None):
    """List available configuration profiles."""
    try:
        profiles = config_manager.list_profiles(category)
        return {"profiles": profiles}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile listing failed: {str(e)}")


@app.post("/validation/run-protocol/{protocol_id}")
async def run_validation_protocol(protocol_id: str, request: ValidationRequest):
    """Run a specific validation protocol."""
    try:
        protocol_id_int = int(protocol_id)
        if protocol_id_int not in [9, 10, 11, 12]:
            raise HTTPException(
                status_code=400, detail=f"Protocol {protocol_id} not supported"
            )

        # Import and run the appropriate protocol
        if protocol_id_int == 9:
            from Validation.Validation_Protocol_9 import APGINeuralSignaturesValidator

            validator = APGINeuralSignaturesValidator()

            # Extract data paths from request parameters
            eeg_path = (
                request.parameters.get("eeg_data_path") if request.parameters else None
            )
            fmri_path = (
                request.parameters.get("fmri_data_path") if request.parameters else None
            )
            behavioral_path = (
                request.parameters.get("behavioral_data_path")
                if request.parameters
                else None
            )

            results = validator.validate_convergent_signatures(
                eeg_data_path=eeg_path,
                fmri_data_path=fmri_path,
                behavioral_data_path=behavioral_path,
            )

        elif protocol_id_int == 10:
            from Validation.Validation_Protocol_10 import CausalManipulationsValidator

            validator = CausalManipulationsValidator()
            # Implement similar to protocol 9
            results = {"status": "placeholder", "protocol": protocol_id}

        elif protocol_id_int == 11:
            from Validation.Validation_Protocol_11 import QuantitativeModelFitsValidator

            validator = QuantitativeModelFitsValidator()
            # Implement similar to protocol 9
            results = {"status": "placeholder", "protocol": protocol_id}

        elif protocol_id_int == 12:
            from Validation.Validation_Protocol_12 import ClinicalCrossSpeciesValidator

            validator = ClinicalCrossSpeciesValidator()
            # Implement similar to protocol 9
            results = {"status": "placeholder", "protocol": protocol_id}

        return {"status": "completed", "protocol": protocol_id, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/data/upload")
async def upload_data_file(file: UploadFile, data_type: str = Form(...)):
    """Upload data files for validation."""
    try:
        # Save file to data_repository
        data_dir = {
            "raw_eeg": RAW_DATA_DIR / "eeg",
            "raw_fmri": RAW_DATA_DIR / "fmri",
            "processed_behavioral": PROCESSED_DATA_DIR,
            "metadata": METADATA_DIR,
        }.get(data_type, PROCESSED_DATA_DIR)

        data_dir.mkdir(parents=True, exist_ok=True)

        file_path = data_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {
            "status": "uploaded",
            "file_path": str(file_path),
            "data_type": data_type,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/results/{result_id}")
async def get_validation_results(result_id: str):
    """Retrieve validation results by ID."""
    try:
        # In a real implementation, results would be stored in a database
        # For now, return placeholder
        return {
            "result_id": result_id,
            "status": "available",
            "results": {"placeholder": "Results would be stored here"},
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Result retrieval failed: {str(e)}"
        )


@app.get("/protocols/list")
async def list_validation_protocols():
    """List available validation protocols."""
    try:
        protocols = {
            "9": {
                "name": "Convergent Neural Signatures",
                "description": "Validate P3b scaling, frontoparietal coactivation, theta-gamma coupling",
                "priority": "1",
                "data_requirements": ["EEG", "fMRI", "behavioral"],
            },
            "10": {
                "name": "Causal Manipulations",
                "description": "TMS/tACS disruption of ignition parameters",
                "priority": "2",
                "data_requirements": ["EEG", "behavioral"],
            },
            "11": {
                "name": "Quantitative Model Fits",
                "description": "Psychometric functions and spiking network fits",
                "priority": "3",
                "data_requirements": ["behavioral"],
            },
            "12": {
                "name": "Clinical and Cross-Species Convergence",
                "description": "Clinical populations and cross-species homologies",
                "priority": "4",
                "data_requirements": ["clinical_data", "cross_species"],
            },
        }

        return {"protocols": protocols}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Protocol listing failed: {str(e)}"
        )
