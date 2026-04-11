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

## Container Template (Conda/Docker)

To completely eliminate dependency-related installation variance across different lab environments, use the following container templates:

### Conda Environment Template

```yaml
# environment.yml
name: apgi-framework
channels:
  - conda-forge
  - pytorch
dependencies:
  - python=3.10
  - numpy>=1.21
  - scipy>=1.7
  - pandas>=1.3
  - matplotlib>=3.4
  - scikit-learn>=0.24
  - pytorch>=1.10
  - cpuonly  # Use 'pytorch::pytorch-cuda' for GPU
  - pymc>=5.10
  - arviz>=0.17
  - pip
  - pip:
    - click>=8.0
    - rich>=10.0
    - pyyaml>=6.0
    - loguru>=0.6
    - python-dotenv>=0.19
    - jsonschema>=4.0
    - pydantic>=1.8
```

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate apgi-framework
```

### Docker Container Template

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run tests to verify installation
RUN python -m pytest tests/test_basic.py -v

# Default command
CMD ["python", "main.py", "--help"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  apgi:
    build: .
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - PYTHONPATH=/app
      - LOG_LEVEL=INFO
    command: python main.py formal-model --simulation-steps 1000
```

### Usage Instructions

```bash
# Build and run with Docker
docker build -t apgi-framework .
docker run -it --rm -v $(pwd)/data:/app/data apgi-framework

# Or use docker-compose
docker-compose up

# For reproducible analysis across labs
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/results:/app/results \
  apgi-framework python main.py validate --all-protocols
```

---

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
