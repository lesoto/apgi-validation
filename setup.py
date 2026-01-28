#!/usr/bin/env python3
"""
APGI Theory Framework Setup Script
==================================

This script helps set up the APGI framework environment.
It creates a virtual environment and installs dependencies.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def create_virtual_environment():
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


def get_venv_python(venv_path):
    """Get the Python executable from the virtual environment."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def install_dependencies(venv_python):
    """Install dependencies in the virtual environment."""
    print("Installing dependencies...")

    requirements_file = PROJECT_ROOT / "requirements.txt"
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
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


def create_activation_script():
    """Create activation scripts for convenience."""
    venv_path = PROJECT_ROOT / "venv"

    if sys.platform == "win32":
        activate_script = PROJECT_ROOT / "activate.bat"
        with open(activate_script, "w") as f:
            f.write(f"@echo off\n{venv_path}\\Scripts\\activate.bat\n")
    else:
        activate_script = PROJECT_ROOT / "activate.sh"
        with open(activate_script, "w") as f:
            f.write(f"#!/bin/bash\nsource {venv_path}/bin/activate\n")
        activate_script.chmod(0o755)

    print(f"✓ Activation script created: {activate_script}")


def test_installation(venv_python):
    """Test the installation by running the test script."""
    print("Testing installation...")

    try:
        result = subprocess.run(
            [str(venv_python), "test_framework.py"],
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


def main():
    """Main setup function."""
    print("APGI Theory Framework Setup")
    print("=" * 40)

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

    # Install dependencies
    if not install_dependencies(venv_python):
        print("Setup failed: Could not install dependencies")
        return False

    # Create activation script
    create_activation_script()

    # Test installation
    if not test_installation(venv_python):
        print("Setup completed with warnings (test failed)")
    else:
        print("Setup completed successfully!")

    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   activate.bat")
    else:
        print("   source activate.sh")
    print("2. Test the CLI:")
    print("   python main.py --help")
    print("3. Run a simulation:")
    print("   python main.py formal-model --simulation-steps 100")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
