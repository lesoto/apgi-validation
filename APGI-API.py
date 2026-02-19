#!/usr/bin/env python3
"""
APGI Framework REST API
========================

REST API for running simulations, validations, and accessing results.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
import secrets
import hashlib
from datetime import datetime, timedelta
import numpy as np

from fastapi import FastAPI, Form, HTTPException, UploadFile, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from werkzeug.utils import secure_filename

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


class LoginRequest(BaseModel):
    """Request model for user login."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Response model for token generation."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


# Authentication setup
security = HTTPBearer()

# Simple in-memory user store (in production, use proper database)
USERS_DB = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),  # Default admin user
    "researcher": hashlib.sha256(
        "research123".encode()
    ).hexdigest(),  # Default researcher user
}

# Token store (in production, use Redis or database)
TOKENS = {}
TOKEN_EXPIRY_HOURS = 24


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password


def create_access_token(username: str) -> str:
    """Create a new access token for the user."""
    token = secrets.token_urlsafe(32)
    expiry = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)
    TOKENS[token] = {"username": username, "expiry": expiry}
    return token


def verify_token(token: str) -> Optional[str]:
    """Verify a token and return the associated username if valid."""
    if token not in TOKENS:
        return None

    token_data = TOKENS[token]
    if datetime.utcnow() > token_data["expiry"]:
        # Token expired, remove it
        del TOKENS[token]
        return None

    return token_data["username"]


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """Dependency to get the current authenticated user."""
    token = credentials.credentials
    username = verify_token(token)

    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return username


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "APGI Framework API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from datetime import datetime

    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    """Authenticate user and return access token."""
    if request.username not in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not verify_password(request.password, USERS_DB[request.username]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    # Create access token
    access_token = create_access_token(request.username)

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=TOKEN_EXPIRY_HOURS * 3600,  # Convert to seconds
    )


@app.post("/simulation/run")
async def run_simulation(
    request: SimulationRequest, current_user: str = Depends(get_current_user)
):
    """Run a simulation with given parameters."""
    try:
        # Import APGI simulation components
        import importlib.util

        # Load the SurpriseIgnitionSystem
        module_path = PROJECT_ROOT / "Falsification" / "Falsification-Protocol-4.py"
        spec = importlib.util.spec_from_file_location("apgi_simulation", module_path)
        if spec and spec.loader:
            simulation_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(simulation_module)
            SurpriseIgnitionSystem = simulation_module.SurpriseIgnitionSystem
        else:
            raise HTTPException(
                status_code=500, detail="Could not load APGI simulation module"
            )

        # Initialize the APGI system with request parameters
        system_params = {
            "alpha": getattr(request, "alpha", 10.0),
            "tau_S": getattr(request, "tau_S", 0.5),
            "tau_theta": getattr(request, "tau_theta", 30.0),
            "theta_0": getattr(request, "theta_0", 0.5),
            "gamma_M": getattr(request, "gamma_M", -0.3),
            "gamma_A": getattr(request, "gamma_A", 0.1),
            "rho": getattr(request, "rho", 0.7),
            "sigma_S": getattr(request, "sigma_S", 0.05),
            "sigma_theta": getattr(request, "sigma_theta", 0.02),
        }

        system = SurpriseIgnitionSystem(**system_params)

        # Run simulation
        steps = getattr(request, "steps", 1000)
        dt = getattr(request, "dt", 0.01)

        # Prepare input generator (simplified for API)
        def input_gen(t):
            return {
                "Pi_e": 1.0 + 0.3 * np.sin(2 * np.pi * t / 10),  # Oscillating precision
                "eps_e": np.random.normal(0.5, 0.3),  # Surprise input
                "beta": 1.0,  # Somatic bias
                "Pi_i": 1.0,  # Internal precision
                "eps_i": np.random.normal(0.2, 0.2),  # Internal error
                "M": 1.0,  # Metabolic state
                "A": 0.5,  # Arousal
            }

        # Run the simulation
        history = system.simulate(steps, dt, input_gen)

        # Extract key results
        ignition_events = np.sum(history["B"] > 0.5)  # Count ignition events
        avg_surprise = np.mean(history["S"])
        final_surprise = history["S"][-1]
        final_threshold = history["theta"][-1]

        # Calculate ignition statistics
        ignition_times = np.where(history["B"] > 0.5)[0]
        ignition_probability = len(ignition_times) / steps if steps > 0 else 0

        return {
            "status": "completed",
            "parameters": request.dict(),
            "results": {
                "steps_simulated": steps,
                "dt": dt,
                "ignition_events": int(ignition_events),
                "ignition_probability": float(ignition_probability),
                "avg_surprise": float(avg_surprise),
                "final_surprise": float(final_surprise),
                "final_threshold": float(final_threshold),
                "time_series": {
                    "time": history["time"][::10].tolist(),  # Sample every 10th point
                    "surprise": history["S"][::10].tolist(),
                    "threshold": history["theta"][::10].tolist(),
                    "ignition": history["B"][::10].tolist(),
                },
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@app.post("/data/generate")
async def generate_data(
    request: DataGenerationRequest, current_user: str = Depends(get_current_user)
):
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
async def validate_data_file(file: UploadFile):
    """Validate an uploaded data file."""
    try:
        # Save uploaded file temporarily with secure filename
        temp_dir = PROJECT_ROOT / "temp"
        temp_dir.mkdir(exist_ok=True)
        secure_name = secure_filename(file.filename)
        temp_path = temp_dir / secure_name

        # Write uploaded file to temp location
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        try:
            validator = data_validation.DataValidator()

            # Validate format
            format_result = validator.validate_file_format(temp_path)

            if not format_result["is_readable"]:
                raise HTTPException(status_code=400, detail="File not readable")

            # Load data and assess quality
            if temp_path.suffix.lower() == ".csv":
                import pandas as pd

                data_df = pd.read_csv(temp_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")

            quality_report = validator.validate_data_quality(data_df)

            return {
                "status": "completed",
                "file_info": format_result,
                "quality_report": quality_report,
            }

        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

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
async def set_config_parameter(
    section: str,
    parameter: str,
    value: str,
    current_user: str = Depends(get_current_user),
):
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
async def create_config_profile(
    name: str,
    description: str,
    category: str = "custom",
    current_user: str = Depends(get_current_user),
):
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
async def run_validation_protocol(
    protocol_id: str,
    request: ValidationRequest,
    current_user: str = Depends(get_current_user),
):
    """Run a specific validation protocol."""
    try:
        protocol_id_int = int(protocol_id)
        if protocol_id_int not in [9, 10, 11, 12]:
            raise HTTPException(
                status_code=400, detail=f"Protocol {protocol_id} not supported"
            )

        # Import and run the appropriate protocol
        if protocol_id_int == 9:
            # Load the module with hyphen in filename
            import importlib.util

            module_path = PROJECT_ROOT / "Validation" / "Validation-Protocol-9.py"
            spec = importlib.util.spec_from_file_location(
                "validation_protocol_9", module_path
            )
            if spec and spec.loader:
                validation_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(validation_module)
                APGINeuralSignaturesValidator = (
                    validation_module.APGINeuralSignaturesValidator
                )
            else:
                raise HTTPException(
                    status_code=500, detail="Could not load Validation Protocol 9"
                )

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
            raise HTTPException(
                status_code=501,
                detail="Not Implemented: Protocol 10 (Causal Manipulations) not yet implemented",
            )

        elif protocol_id_int == 11:
            raise HTTPException(
                status_code=501,
                detail="Not Implemented: Protocol 11 (Quantitative Model Fits) not yet implemented",
            )

        elif protocol_id_int == 12:
            raise HTTPException(
                status_code=501,
                detail="Not Implemented: Protocol 12 (Clinical Cross-Species) not yet implemented",
            )

        return {"status": "completed", "protocol": protocol_id, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/data/upload")
async def upload_data_file(
    file: UploadFile,
    data_type: str = Form(...),
    current_user: str = Depends(get_current_user),
):
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

        file_path = data_dir / secure_filename(file.filename)
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
    raise HTTPException(
        status_code=501,
        detail="Not Implemented: Results persistence not yet implemented",
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
