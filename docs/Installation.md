# APGI Theory Framework - Installation Guide

## Quick Start

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python main.py --help
```

## Troubleshooting

### Issue: "externally-managed-environment" Error

This occurs on systems with managed Python installations. Solution:

```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or use --break-system-packages (not recommended)
pip install --break-system-packages -r requirements.txt
```

### Issue: Missing Dependencies

If some packages fail to install, you can install them individually:

```bash
# Core dependencies
pip install numpy scipy pandas matplotlib
pip install scikit-learn torch torchvision
pip install pymc arviz tqdm

# CLI and utilities
pip install click rich pyyaml
pip install loguru python-dotenv jsonschema pydantic

# Development tools (optional)
pip install pytest black flake8
```

### Issue: Python Version Compatibility

The framework requires Python 3.8+. Some packages may have version restrictions:

- **PyMC**: Works best with Python 3.9-3.11
- **PyTorch**: Supports Python 3.8-3.12
- **NumPy/SciPy**: Generally compatible with all Python 3.8+

If you encounter compatibility issues, consider using Python 3.10 or 3.11.

## Verification

### Basic Test (No Dependencies Required)

```bash
python -m pytest tests/test_basic.py -v
```

### Full Test (Dependencies Required)

```bash
python -m pytest tests/ -v
```

### CLI Test

```bash
python main.py --help
python main.py info
python main.py formal-model --simulation-steps 100
```

## Project Structure After Installation

```text
apgi-theory/
├── README.md                    # Main documentation
├── docs/Install.md              # This installation guide
├── requirements.txt             # Dependencies list
├── setup.py                     # Automated setup script
├── main.py                      # Unified CLI interface
├── minimal_test.py              # Basic functionality test
├── test_framework.py            # Comprehensive test suite
├── config_manager.py            # Configuration system
├── logging_config.py            # Logging framework
├── config/
│   └── default.yaml            # Default configuration
├── logs/                        # Log files (created during runtime)
├── data_repository/              # Data directory (raw_data/, processed_data/, metadata/, etc.)
├── cache/                       # Cache directory
├── venv/                        # Virtual environment (created)
└── [existing APGI modules...]   # Original framework components
```

## Next Steps

1. **Run the CLI**: `python main.py --help`
2. **Test a simulation**: `python main.py formal-model --simulation-steps 1000`
3. **View configuration**: `python main.py config --show`
4. **Check logs**: `python main.py logs --tail 20`

## Support

If you encounter issues:

1. Check the logs directory for error messages
2. Run `python minimal_test.py` to verify basic functionality
3. Ensure all dependencies are installed correctly
4. Verify Python version compatibility (3.8+)

For detailed usage instructions, see the main README.md file.
