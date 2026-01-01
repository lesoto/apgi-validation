# APGI Theory Framework - Troubleshooting Guide

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Configuration Problems](#configuration-problems)
3. [Runtime Errors](#runtime-errors)
4. [Performance Issues](#performance-issues)
5. [Data Issues](#data-issues)
6. [GUI Problems](#gui-problems)
7. [Validation Failures](#validation-failures)
8. [Memory and Resource Issues](#memory-and-resource-issues)
9. [Logging and Debugging](#logging-and-debugging)
10. [Common Error Messages](#common-error-messages)

## Installation Issues

### Virtual Environment Problems

**Problem:** Cannot activate virtual environment
```bash
# Error
source: No such file 'venv/bin/activate'
```

**Solution:**
```bash
# Create virtual environment
python3 -m venv venv

# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate

# Verify activation
which python  # Should show venv path
```

**Problem:** Dependencies not installing
```bash
# Error
ERROR: Could not install packages due to EnvironmentError
```

**Solution:**
```bash
# Upgrade pip first
pip install --upgrade pip

# Install dependencies one by one to identify problematic package
pip install numpy
pip install scipy
pip install pandas
# ... continue with other packages

# If specific package fails, try alternative installation
pip install --no-cache-dir <package-name>
# or
conda install <package-name>
```

### Module Import Errors

**Problem:** ModuleNotFoundError for specific modules
```python
# Error
ModuleNotFoundError: No module named 'click'
```

**Solution:**
```bash
# Install missing dependency
pip install click

# Or install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import click; print(click.__version__)"
```

**Problem:** Import errors for APGI modules
```python
# Error
ModuleNotFoundError: No module named 'config_manager'
```

**Solution:**
```bash
# Check current directory
pwd
# Should be in apgi-theory directory

# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/apgi-theory"

# Or use absolute imports in Python
import sys
sys.path.append('/path/to/apgi-theory')
```

## Configuration Problems

### Configuration File Issues

**Problem:** Configuration file not found
```bash
# Error
FileNotFoundError: [Errno 2] No such file or directory: 'config/default.yaml'
```

**Solution:**
```bash
# Create config directory
mkdir -p config

# Reset configuration
python main.py config --reset

# Check configuration
python main.py config --show
```

**Problem:** Invalid configuration values
```python
# Error
ValueError: Invalid value for simulation.default_steps: '500' is not of type 'integer'
```

**Solution:**
```bash
# Set correct type
python main.py config --set simulation.default_steps=500

# Or edit config file directly
nano config/default.yaml
```

### Parameter Validation Errors

**Problem:** Parameter validation fails
```python
# Error
jsonschema.exceptions.ValidationError: 'invalid_value' is not of type 'number'
```

**Solution:**
```python
# Check valid parameter ranges
from config_manager import get_config
config = get_config()
print(config.simulation)  # Show valid parameters

# Use correct data type
set_parameter('simulation', 'default_steps', 500)  # integer, not string
```

## Runtime Errors

### Simulation Errors

**Problem:** Simulation crashes with NaN values
```python
# Error
RuntimeWarning: invalid value encountered in divide
```

**Solution:**
```python
# Check input data for invalid values
import numpy as np
data = np.array(your_data)
print(f"NaN values: {np.isnan(data).any()}")
print(f"Infinite values: {np.isinf(data).any()}")

# Clean data
data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

# Use smaller time steps
system.step(dt=0.001, inputs=inputs)  # Smaller dt
```

**Problem:** Model parameters out of range
```python
# Error
ValueError: tau_S must be positive
```

**Solution:**
```python
# Check parameter constraints
valid_params = {
    'tau_S': 0.01,      # Must be positive
    'tau_theta': 0.1,   # Must be positive
    'theta_0': 2.0,     # Should be reasonable
    'alpha': 0.5,       # Should be between 0 and 1
}

# Use valid parameters
system = SurpriseIgnitionSystem(params=valid_params)
```

### Integration Errors

**Problem:** Multimodal integration fails
```python
# Error
KeyError: 'exteroceptive' not found in data
```

**Solution:**
```python
# Check data structure
required_keys = ['exteroceptive', 'interoceptive', 'somatic']
data = {'EEG': [...], 'Pupil': [...], 'EDA': [...]}

# Map keys correctly
key_mapping = {
    'EEG': 'exteroceptive',
    'Pupil': 'interoceptive', 
    'EDA': 'somatic'
}

mapped_data = {key_mapping.get(k, k): v for k, v in data.items()}
```

## Performance Issues

### Slow Performance

**Problem:** Simulations running very slowly
```python
# Symptoms
- 1000 steps taking > 10 seconds
- High CPU usage
- Memory growing continuously
```

**Solution:**
```python
# Use vectorized operations
import numpy as np

# Instead of loops
for i in range(len(data)):
    result = process_single(data[i])

# Use vectorized operations
results = process_vectorized(data)

# Enable parallel processing
python main.py validate --all-protocols --parallel

# Use batch processing
processor = APGIBatchProcessor(normalizer, config, batch_size=1000)
```

**Problem:** Memory usage too high
```python
# Symptoms
- Out of memory errors
- System becomes unresponsive
- Memory usage grows with time
```

**Solution:**
```python
# Process data in chunks
chunk_size = 1000
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    process_chunk(chunk)
    
    # Clean up memory
    import gc
    gc.collect()

# Use generators instead of lists
def data_generator():
    for item in large_dataset:
        yield process_item(item)

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")
```

### GPU Acceleration Issues

**Problem:** GPU not being used despite availability
```python
# Symptoms
- torch.cuda.is_available() returns False
- CPU usage high, GPU idle
```

**Solution:**
```python
# Check PyTorch GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# Install GPU-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Move tensors to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = tensor.to(device)
```

## Data Issues

### Data Loading Problems

**Problem:** Cannot read data files
```python
# Error
FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'
```

**Solution:**
```bash
# Check file exists
ls -la data.csv

# Use absolute path
python main.py multimodal --input-data /full/path/to/data.csv

# Check current directory
pwd
```

**Problem:** Data format issues
```python
# Error
pandas.errors.ParserError: Error tokenizing data
```

**Solution:**
```python
# Check file format
import pandas as pd

# Try different encodings
for encoding in ['utf-8', 'latin-1', 'cp1252']:
    try:
        data = pd.read_csv('data.csv', encoding=encoding)
        print(f"Success with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        continue

# Specify delimiter and parameters
data = pd.read_csv('data.csv', delimiter=',', quotechar='"', escapechar='\\')
```

### Data Quality Issues

**Problem:** Missing or invalid data
```python
# Symptoms
- NaN values in results
- Division by zero errors
- Invalid parameter values
```

**Solution:**
```python
# Check data quality
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

# Check for missing values
print(f"Missing values:\n{data.isnull().sum()}")

# Check data types
print(f"Data types:\n{data.dtypes}")

# Check for infinite values
numeric_cols = data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    inf_count = np.isinf(data[col]).sum()
    if inf_count > 0:
        print(f"{col}: {inf_count} infinite values")

# Clean data
data_clean = data.dropna()  # Remove rows with missing values
data_clean = data_clean.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
data_clean = data_clean.dropna()  # Remove inf values

# Fill missing values if appropriate
data_filled = data.fillna(data.mean())  # Fill with mean
# or
data_filled = data.fillna(method='ffill')  # Forward fill
```

## GUI Problems

### GUI Not Starting

**Problem:** GUI fails to launch
```bash
# Error
tkinter.TclError: no display name and no $DISPLAY environment variable
```

**Solution:**
```bash
# For headless systems, use web interface
python main.py gui --gui-type analysis --host 0.0.0.0 --port 8080

# Install required GUI dependencies
pip install tkinter  # Usually comes with Python
pip install PyQt5  # Alternative GUI framework

# Check X11 forwarding (SSH)
ssh -X user@server
```

**Problem:** Flask not available for web interface
```bash
# Error
ImportError: No module named 'flask'
```

**Solution:**
```bash
# Install Flask
pip install flask

# Or use alternative web framework
pip install fastapi uvicorn
```

### GUI Display Issues

**Problem:** GUI windows not rendering properly
```python
# Symptoms
- Blank windows
- Distorted graphics
- Controls not responding
```

**Solution:**
```python
# Check matplotlib backend
import matplotlib
print(f"Backend: {matplotlib.get_backend()}")

# Set appropriate backend
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg' for headless

# Update GUI libraries
pip install --upgrade matplotlib PyQt5
```

## Validation Failures

### Protocol Execution Failures

**Problem:** Validation protocols not running
```python
# Error
AttributeError: module has no attribute 'run_validation'
```

**Solution:**
```python
# Check protocol structure
# Each protocol should have:
def run_validation():
    """Run validation and return results."""
    return results

def main():
    """Main entry point."""
    results = run_validation()
    print(results)

# Verify protocol file
python Validation/APGI-Protocol-1.py
```

**Problem:** Validation results inconsistent
```python
# Symptoms
- Different results on multiple runs
- Random failures
- Non-deterministic behavior
```

**Solution:**
```python
# Set random seeds
import numpy as np
import random
np.random.seed(42)
random.seed(42)

# Use deterministic algorithms
from sklearn.utils import check_random_state
rng = check_random_state(42)

# Run multiple times and average
results = []
for i in range(10):
    result = run_validation()
    results.append(result)

mean_result = np.mean(results, axis=0)
std_result = np.std(results, axis=0)
```

## Memory and Resource Issues

### Out of Memory Errors

**Problem:** System runs out of memory
```bash
# Error
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Reduce batch size
batch_size = 100  # Instead of 1000

# Use memory-efficient data types
import numpy as np
data = np.array(data, dtype=np.float32)  # Instead of float64

# Process incrementally
def process_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

# Use generators
def data_stream():
    for item in large_dataset:
        yield process_item(item)

# Monitor memory usage
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB")

# Clean up explicitly
del large_variable
gc.collect()
```

### File Handle Limits

**Problem:** Too many open files
```bash
# Error
OSError: [Errno 24] Too many open files
```

**Solution:**
```python
# Close files explicitly
with open('file.txt', 'r') as f:
    data = f.read()
# File automatically closed

# Check open file handles
import psutil
process = psutil.Process()
open_files = process.open_files()
print(f"Open files: {len(open_files)}")

# Increase file limit (Linux/macOS)
ulimit -n 4096  # Temporary
# Add to ~/.bashrc for permanent:
# ulimit -n 4096
```

## Logging and Debugging

### Enabling Debug Mode

**Problem:** Need more detailed error information

**Solution:**
```bash
# Set debug log level
python main.py --log-level DEBUG formal-model --simulation-steps 100

# Check configuration
python main.py config --show

# View recent logs
python main.py logs --tail 50 --level DEBUG
```

### Custom Debugging

**Problem:** Need to debug specific component

**Solution:**
```python
# Add debug prints
import logging
logging.basicConfig(level=logging.DEBUG)

# Use logging in your code
logger = logging.getLogger(__name__)
logger.debug(f"Variable value: {variable}")

# Use pdb for interactive debugging
import pdb; pdb.set_trace()

# Or use ipdb for better debugging
pip install ipdb
import ipdb; ipdb.set_trace()
```

### Performance Profiling

**Problem:** Need to identify performance bottlenecks

**Solution:**
```python
# Use cProfile
python -m cProfile -o profile.stats main.py formal-model --simulation-steps 1000

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"

# Use line_profiler for detailed analysis
pip install line_profiler

# Add @profile decorator to functions
@profile
def slow_function():
    # Your code here

# Run with profiler
kernprof -l -v your_script.py
```

## Common Error Messages

### Import Errors

```
ModuleNotFoundError: No module named 'X'
```
**Solution:** Install missing module with `pip install X`

```
ImportError: cannot import name 'Y' from 'X'
```
**Solution:** Check module version or use correct import name

### Configuration Errors

```
ValueError: Invalid value for parameter
```
**Solution:** Check parameter type and range constraints

```
FileNotFoundError: Configuration file not found
```
**Solution:** Run `python main.py config --reset` to create default config

### Runtime Errors

```
RuntimeWarning: invalid value encountered in divide
```
**Solution:** Check for division by zero or NaN values

```
MemoryError: Unable to allocate array
```
**Solution:** Reduce data size or process in chunks

### Data Errors

```
pandas.errors.ParserError: Error tokenizing data
```
**Solution:** Check file format and encoding

```
KeyError: 'required_column' not found
```
**Solution:** Verify column names in data file

### Validation Errors

```
AttributeError: module has no attribute 'run_validation'
```
**Solution:** Ensure validation protocol has required functions

```
AssertionError: Validation failed
```
**Solution:** Check validation logic and input data

## Getting Additional Help

### Check System Status

```bash
# Check framework status
python main.py info

# Run validation protocols
python main.py validate --all-protocols

# Check logs for errors
python main.py logs --tail 50 --level ERROR
```

### Report Issues

When reporting issues, include:

1. **System Information**
   ```bash
   python --version
   pip list | grep -E "(numpy|scipy|pandas|pymc)"
   uname -a
   ```

2. **Error Messages**
   - Full error traceback
   - Command that caused the error
   - Input data (if applicable)

3. **Configuration**
   ```bash
   python main.py config --show > config.txt
   ```

4. **Logs**
   ```bash
   python main.py logs --export debug_logs.json
   ```

### Community Resources

- Check the GitHub repository for known issues
- Review the API documentation
- Consult the tutorial for usage examples
- Run validation protocols to verify installation

### Performance Tuning

For optimal performance:

1. **Use appropriate data types** (float32 instead of float64 when possible)
2. **Process data in chunks** for large datasets
3. **Enable parallel processing** when available
4. **Monitor memory usage** and clean up resources
5. **Use vectorized operations** instead of loops
6. **Profile code** to identify bottlenecks

This troubleshooting guide should help resolve most common issues with the APGI Theory Framework. For persistent problems, consult the full documentation or seek community support.
