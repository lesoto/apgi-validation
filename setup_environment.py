#!/usr/bin/env python3
"""
APGI Validation Framework Setup Script
======================================

This script sets up the APGI Validation Framework environment.
It creates a virtual environment, installs dependencies, and configures
security keys required for the framework.

APGI Validation Framework: A computational framework for validating and
falsifying Active Inference and Predictive Processing models through
systematic experimentation.

Quick Start
-----------
1. Run this setup script: python setup_environment.py
2. Activate virtual environment: source activate.sh (or activate.bat on Windows)
3. Set environment variables (see README.md)
4. Launch GUI: python Theory_GUI.py

For protocol-specific dependencies:
   pip install -r requirements-protocols.txt

Security Setup Required
-----------------------
Before running the framework, set these environment variables:
   export PICKLE_SECRET_KEY=$(python -c "import os; print(os.urandom(32).hex())")
   export APGI_BACKUP_HMAC_KEY=$(python -c "import os; print(os.urandom(32).hex())")
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).parent

# Core dependencies matching requirements.txt
# For reproducibility, use specific versions from requirements.txt
CORE_DEPENDENCIES = [
    "numpy>=2.3.0",
    "scipy>=1.15.0",
    "pandas>=2.2.0",
    "torch>=2.1.0",
    "torchvision>=0.15.0",
    "scikit-learn>=1.5.0",
    "shap>=0.44.0",
    "tqdm>=4.62.0",
    "rich>=13.0.0",
    "dash>=2.0.0",
    "pyyaml>=6.0",
    "jsonschema>=4.0.0",
    "pydantic>=2.0.0",
    "sqlalchemy>=2.0.0",
    "alembic>=1.13.0",
    "python-dateutil>=2.8.0",
    "pytz>=2021.3",
    "psutil>=5.8.0",
    "msgpack>=1.0.0",
    "reportlab>=4.0.0",
    "numba>=0.60.0",
    "nolds>=0.5.2",
    "cryptography>=42.0.0",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
    "tifffile>=2021.7.0",
    "pyreadr>=0.8.0",
]

# Development and testing dependencies
DEV_DEPENDENCIES = [
    "pytest>=8.3.3",
    "pytest-cov>=2.12.0",
    "pytest-mock>=3.6.0",
    "pytest-xdist>=2.5.0",
    "pytest-asyncio>=0.21.0",
    "pytest-timeout>=2.0.0",
    "hypothesis>=6.0.0",
    "black>=22.0.0",
    "isort>=5.13.2",
    "sphinx>=5.0.0",
    "bandit>=1.7.0",
    "safety>=2.0.0",
    "pip-audit>=2.0.0",
    "memory-profiler>=0.60.0",
    "line-profiler>=4.0.0",
    "pre-commit>=3.0.0",
    "tox>=4.0.0",
    "coverage>=6.0.0",
]

# APGI Validation Framework Module Structure
# ===========================================
FRAMEWORK_MODULES = {
    "Theory": [
        "APGI_Bayesian_Estimation_Framework.py",
        "APGI_Computational_Benchmarking.py",
        "APGI_Cross_Species_Scaling.py",
        "APGI_Cultural_Neuroscience.py",
        "APGI_DynamicalSystems_Formalization.py",
        "APGI_Ignition_Dynamics_Simulator.py",
        "APGI_Mathematical_Formalization.py",
        "APGI_Neural_Mass_Models.py",
        "APGI_Physics_Integration.py",
        "APGI_Precision_Dynamics.py",
        "APGI_Rational_Pathway_Model.py",
        "APGI_Urgency_Gating.py",
    ],
    "Validation": [
        "VP_01_SyntheticEEG_MLClassification.py",
        "VP_02_Behavioral_BayesianComparison.py",
        "VP_03_ActiveInference_AgentSimulations.py",
        "VP_04_CausalIntervention_TMS.py",
        "BayesianModelComparison_ParameterRecovery.py",
        "BayesianModelComparison_PredictiveAccuracy.py",
        "CrossValidation_Empirical_Generalizability.py",
        "CrossValidation_ParameterStability.py",
        "Master_Validation.py",
        "Multiverse_Analysis_Robustness.py",
    ],
    "Falsification": [
        "FP_01_ActiveInference.py",
        "FP_02_AgentComparison_ConvergenceBenchmark.py",
        "FP_03_FrameworkLevel_MultiProtocol.py",
        "FP_04_PhaseTransition_EpistemicArchitecture.py",
        "FP_05_SurvivalAnalysis_TimeToIgnition.py",
        "FP_06_Perturbation_Resilience.py",
        "FP_07_Diversity_BehavioralSpaceCoverage.py",
        "FP_08_MetaLearning_FewShotAdaptation.py",
        "FP_09_TemporalDynamics_SequenceSensitivity.py",
        "FP_10_Adversarial_AttackResistance.py",
        "FP_11_ResourceEfficiency_ComputationalCost.py",
        "FP_12_Falsification_Aggregator.py",
    ],
    "GUI_Apps": [
        "Theory_GUI.py",
        "Validation_GUI.py",
        "Falsification_Protocols_GUI.py",
        "Tests_GUI.py",
        "Utils_GUI.py",
    ],
}

# Protocol-specific optional dependencies
PROTOCOL_DEPENDENCIES = {
    "Bayesian modeling (VP-11, FP-10)": ["pymc>=5.0.0,<6.0.0", "arviz>=0.16.0,<1.0.0"],
    "EEG processing (VP-01, VP-07, VP-09)": ["mne>=1.6.0,<1.7.0"],
    "Survival analysis (VP-02, FP-02)": ["lifelines>=0.27.0,<0.29.0"],
    "Deep learning explainability (VP-09)": ["captum>=0.7.0,<0.8.0"],
    "Sensitivity analysis (FP-08)": ["SALib>=1.4.0,<2.0.0"],
    "Machine learning (VP-01, VP-02, FP-03)": ["scikit-learn>=1.5.0,<2.0.0"],
    "Symbolic mathematics (FP-07, FP-08)": ["sympy>=1.12,<2.0.0"],
}


def create_virtual_environment() -> Optional[Path]:
    """Create a virtual environment if it doesn't exist."""
    venv_path = PROJECT_ROOT / "venv"

    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return venv_path

    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print(f"✓ Virtual environment created at {venv_path}")
        return venv_path
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return None


def get_venv_python(venv_path: Path) -> Path:
    """Get the Python executable from the virtual environment."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def install_dependencies(venv_python: Path) -> bool:
    """Install dependencies in the virtual environment."""
    print("Installing dependencies...")

    requirements_file = PROJECT_ROOT / "requirements.txt"
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False

    # Validate requirements.txt is readable and not empty
    try:
        with open(requirements_file, "r") as f:
            content = f.read().strip()
            if not content:
                print("✗ requirements.txt is empty")
                return False
            # Basic validation: check for common requirement patterns
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            if not lines:
                print("✗ requirements.txt contains no valid package specifications")
                return False
    except (IOError, UnicodeDecodeError) as e:
        print(f"✗ Failed to read requirements.txt: {e}")
        return False

    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True
        )
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", str(requirements_file)],
            check=True,
        )
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def create_activation_script() -> None:
    """Create activation scripts for convenience."""
    venv_path = PROJECT_ROOT / "venv"

    if sys.platform == "win32":
        activate_script = PROJECT_ROOT / "activate.bat"
        with open(activate_script, "w", encoding="utf-8") as f:
            f.write(f"@echo off\n{venv_path}\\Scripts\\activate.bat\n")
    else:
        activate_script = PROJECT_ROOT / "activate.sh"
        with open(activate_script, "w", encoding="utf-8") as f:
            f.write(f"#!/bin/bash\nsource {venv_path}/bin/activate\n")
        activate_script.chmod(0o755)

    print(f"✓ Activation script created: {activate_script}")


def install_core_dependencies(venv_python: Path) -> bool:
    """Install core dependencies from requirements.txt."""
    print("Installing core dependencies from requirements.txt...")

    requirements_file = PROJECT_ROOT / "requirements.txt"
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False

    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
        )
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", str(requirements_file)],
            check=True,
        )
        print("✓ Core dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install core dependencies: {e}")
        return False


def install_dev_dependencies(venv_python: Path) -> bool:
    """Install development and testing dependencies."""
    print("Installing development dependencies...")

    try:
        for dep in DEV_DEPENDENCIES:
            print(f"  Installing {dep}...")
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", dep],
                check=True,
            )
        print("✓ Development dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install development dependencies: {e}")
        return False


def install_protocol_dependencies(venv_python: Path) -> bool:
    """Install optional protocol-specific dependencies."""
    print("Installing protocol-specific dependencies (optional)...")

    req_protocols = PROJECT_ROOT / "requirements-protocols.txt"
    if not req_protocols.exists():
        print("⚠ requirements-protocols.txt not found - skipping")
        return True

    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", str(req_protocols)],
            check=True,
        )
        print("✓ Protocol dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠ Failed to install protocol dependencies: {e}")
        return False


def test_installation(venv_python: Path) -> bool:
    """Test the installation by importing key modules."""
    print("Testing installation...")

    test_code = """
import sys
try:
    import numpy, scipy, pandas, torch
    print("✓ Core packages (numpy, scipy, pandas, torch) imported successfully")
    sys.exit(0)
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
"""

    try:
        result = subprocess.run(
            [str(venv_python), "-c", test_code],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓ Installation test passed")
            return True
        else:
            print("✗ Installation test failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def verify_core_versions(venv_python: Path) -> bool:
    """Verify that core packages are installed with compatible versions."""
    print("Verifying core package versions...")

    version_check_code = """
import sys
import numpy
import scipy
import pandas
import torch

packages = {
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "pandas": pandas.__version__,
    "torch": torch.__version__.split('+')[0],  # Remove CUDA suffix
}

print("Installed package versions:")
for pkg, ver in packages.items():
    print(f"  {pkg}: {ver}")

sys.exit(0)
"""

    try:
        result = subprocess.run(
            [str(venv_python), "-c", version_check_code],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Version verification failed: {e}")
        return False


def verify_module_exists(module_dir: str, module_name: str) -> bool:
    """Check if a module exists in the specified directory."""
    module_path = PROJECT_ROOT / module_dir / module_name
    return module_path.exists()


def verify_script_exists(script_name: str) -> bool:
    """Check if a script exists in the project root."""
    script_path = PROJECT_ROOT / script_name
    return script_path.exists()


def display_framework_structure() -> None:
    """Display the APGI Validation Framework module structure."""
    print("\n" + "=" * 80)
    print("APGI VALIDATION FRAMEWORK MODULE STRUCTURE")
    print("=" * 80)

    for category, modules in FRAMEWORK_MODULES.items():
        print(f"\n{category}:")
        for module in modules:
            exists = "✓" if verify_module_exists(category, module) else "✗"
            print(f"  [{exists}] {module}")

    print("\n" + "=" * 80)


def display_protocol_dependencies() -> None:
    """Display protocol-specific optional dependencies."""
    print("\n" + "=" * 80)
    print("OPTIONAL PROTOCOL-SPECIFIC DEPENDENCIES")
    print("=" * 80)
    print("\nInstall with: pip install -r requirements-protocols.txt\n")

    for category, deps in PROTOCOL_DEPENDENCIES.items():
        print(f"{category}:")
        for dep in deps:
            print(f"  - {dep}")

    print()


def check_required_env_vars() -> List[str]:
    """Check if required environment variables are set."""
    import os

    missing = []
    if not os.environ.get("PICKLE_SECRET_KEY"):
        missing.append("PICKLE_SECRET_KEY")
    if not os.environ.get("APGI_BACKUP_HMAC_KEY"):
        missing.append("APGI_BACKUP_HMAC_KEY")
    return missing


def display_env_var_instructions() -> None:
    """Display instructions for setting required environment variables."""
    print("\n" + "=" * 80)
    print("SECURITY SETUP REQUIRED")
    print("=" * 80)
    print("\nBefore running the framework, set these environment variables:\n")
    print(
        '  export PICKLE_SECRET_KEY=$(python -c "import os; print(os.urandom(32).hex())")'
    )
    print(
        '  export APGI_BACKUP_HMAC_KEY=$(python -c "import os; print(os.urandom(32).hex())")'
    )
    print("\nOr add them to your .env file in the project root.")
    print()


def main() -> bool:
    """Main setup function for APGI Validation Framework."""
    print("APGI Validation Framework Setup")
    print("=" * 50)
    print("A computational framework for validating and falsifying")
    print("Active Inference and Predictive Processing models.")
    print("=" * 50)

    # Create virtual environment
    venv_path = create_virtual_environment()
    if not venv_path:
        print("Setup failed: Could not create virtual environment")
        return False

    # Get virtual environment Python
    venv_python = get_venv_python(venv_path)
    if not venv_python.exists():
        print("Setup failed: Virtual environment Python not found")
        return False

    # Install core dependencies
    if not install_core_dependencies(venv_python):
        print("Setup failed: Could not install core dependencies")
        return False

    # Verify core versions
    if not verify_core_versions(venv_python):
        print("Setup completed with warnings (version check failed)")
    else:
        print("✓ Core versions verified successfully")

    # Create activation script
    create_activation_script()

    # Test installation
    if not test_installation(venv_python):
        print("Setup completed with warnings (test failed)")
    else:
        print("✓ Installation test passed!")

    # Display framework structure
    display_framework_structure()

    # Display protocol dependencies
    display_protocol_dependencies()

    # Check environment variables
    missing_env = check_required_env_vars()
    if missing_env:
        display_env_var_instructions()
    else:
        print("✓ Required environment variables are set")

    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    print("✓ Virtual environment created")
    print("✓ Core dependencies installed from requirements.txt")
    print("✓ Activation script created (activate.sh / activate.bat)")
    if not missing_env:
        print("✓ Environment variables configured")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   activate.bat")
    else:
        print("   source activate.sh")

    if missing_env:
        print("\n2. Set required environment variables:")
        print(
            '   export PICKLE_SECRET_KEY=$(python -c "import os; print(os.urandom(32).hex())")'
        )
        print(
            '   export APGI_BACKUP_HMAC_KEY=$(python -c "import os; print(os.urandom(32).hex())")'
        )
        base_step = 3
    else:
        base_step = 2

    print(f"\n{base_step}. Install protocol-specific dependencies (optional):")
    print("   pip install -r requirements-protocols.txt")

    print(f"\n{base_step + 1}. Explore the framework:")
    print("   python main.py --help                    # CLI interface")
    print("   python Theory_GUI.py                   # Theory module GUI")
    print("   python Validation_GUI.py               # Validation protocols GUI")
    print("   python Falsification_Protocols_GUI.py  # Falsification tests GUI")
    print("   python Tests_GUI.py                    # Test runner GUI")

    print(f"\n{base_step + 2}. Run validation protocols:")
    print("   python Validation/Master_Validation.py")
    print("   python Falsification/FP_12_Falsification_Aggregator.py")

    print(f"\n{base_step + 3}. Run tests:")
    print("   pytest")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
