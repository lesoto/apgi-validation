#!/usr/bin/env python3
"""
APGI Framework Dependency Installer
==================================

Simple script to install all required dependencies for the APGI framework.
This handles the installation of both core and optional dependencies.

Usage:
    python install_dependencies.py [--optional]
"""

import subprocess
import sys
from pathlib import Path

try:
    from .error_handling import (
        APGIError,
        ConfigurationError,
        DataError,
        ErrorSeverity,
        format_error_message,
        handle_error,
        safe_execute,
    )
except ImportError:
    # Fallback for standalone execution
    try:
        from error_handler import (
            APGIError,
            ConfigurationError,
            DataError,
            ErrorSeverity,
            format_error_message,
            handle_error,
            safe_execute,
        )
    except ImportError:
        # Minimal fallback if error_handler is not available
        class APGIError(Exception):
            pass

        class ConfigurationError(APGIError):
            pass

        class DataError(APGIError):
            pass

        class ErrorSeverity:
            CRITICAL = "CRITICAL"
            HIGH = "HIGH"
            MEDIUM = "MEDIUM"
            LOW = "LOW"
            INFO = "INFO"

        def format_error_message(error, context=None):
            return str(error)

        def handle_error(error, context=None):
            print(f"Error: {format_error_message(error, context)}")

        def safe_execute(func, *args, **kwargs):
            return func(*args, **kwargs)


def run_command(command, description):
    """Run a command and handle errors gracefully."""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        error = ConfigurationError(
            message=format_error_message(
                "protocol_failed",
                reason=f"Command failed with return code {e.returncode}",
            ),
            context={
                "command": command,
                "description": description,
                "return_code": e.returncode,
                "stderr": e.stderr,
            },
            suggestion="Check command syntax and system permissions",
        )
        print(f"❌ {error}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        error = ConfigurationError(
            message="Python 3.8+ is required for this framework",
            context={
                "current_version": f"{version.major}.{version.minor}.{version.micro}",
                "required_version": "3.8+",
            },
            suggestion="Please upgrade Python to version 3.8 or higher",
        )
        print(f"❌ {error}")
        return False
    print(
        f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible"
    )
    return True


def install_core_dependencies():
    """Install core dependencies from requirements.txt."""
    requirements_path = Path(__file__).parent.parent / "requirements.txt"

    if not requirements_path.exists():
        error = DataError(
            message=format_error_message(
                "file_not_found", file_path=str(requirements_path)
            ),
            suggestion="Ensure requirements.txt exists in the project root directory",
        )
        print(f"❌ {error}")
        return False

    print(
        "📦 Note: This script uses --break-system-packages flag to install dependencies."
    )
    print("💡 For production use, consider creating a virtual environment:")
    print("   python3 -m venv venv")
    print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("   pip install -r requirements.txt")
    print()

    # Upgrade pip first
    run_command(
        f"{sys.executable} -m pip install --upgrade pip --break-system-packages",
        "Upgrading pip",
    )

    # Install core requirements
    success = run_command(
        f"{sys.executable} -m pip install -r {requirements_path} --break-system-packages",
        "Installing core dependencies",
    )

    return success


def install_optional_dependencies():
    """Install optional dependencies for enhanced functionality."""
    optional_packages = [
        "jupyter>=1.0.0",
        "ipykernel>=6.0.0",
        "notebook>=6.0.0",
        "ipywidgets>=7.6.0",
    ]

    for package in optional_packages:
        run_command(
            f"{sys.executable} -m pip install {package} --break-system-packages",
            f"Installing {package}",
        )


def verify_installation():
    """Verify that critical dependencies are installed."""
    critical_packages = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "sklearn",
        "torch",
        "click",
        "rich",
    ]

    print("\n🔍 Verifying installation...")
    failed = []

    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            error = ImportWarning(
                message=format_error_message("missing_dependency", package=package),
                package=package,
                suggestion=f"Install with: pip install {package}",
            )
            print(f"❌ {package}")
            failed.append(package)

    if failed:
        error = ConfigurationError(
            message=f"Failed to install critical dependencies: {', '.join(failed)}",
            context={"failed_packages": failed},
            suggestion="Run the installer again or install packages manually",
        )
        print(f"\n⚠️  {error}")
        return False
    else:
        print("\n✅ All critical dependencies installed successfully!")
        return True


def main():
    """Main installation function."""
    print("🧠 APGI Framework Dependency Installer")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install core dependencies
    if not install_core_dependencies():
        error = ConfigurationError(
            message="Core dependency installation failed",
            suggestion="Check system permissions and internet connection",
        )
        print(f"\n❌ {error}")
        sys.exit(1)

    # Check for optional flag
    install_optional = "--optional" in sys.argv
    if install_optional:
        print("\n📦 Installing optional dependencies...")
        install_optional_dependencies()

    # Verify installation
    if verify_installation():
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python main.py --help' to see available commands")
        print("2. Try 'python main.py validate --all-protocols' to test the framework")
        print("3. Use 'python main.py gui --gui-type analysis' for the web interface")
    else:
        error = ConfigurationError(
            message="Installation verification failed",
            suggestion="Check error messages above and resolve dependency issues",
        )
        print(f"\n❌ {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
