# APGI Utils Scripts Runner GUI

A simple tkinter-based GUI to run all scripts in the utils folder with real-time output display and error handling.

## Features

- **Run Individual Scripts**: Select and run any script from the utils folder
- **Run All Scripts**: Execute all utils scripts sequentially
- **Real-time Output**: See script output as it runs
- **Error Handling**: Clear error messages and status indicators
- **Progress Tracking**: Visual progress bar during script execution
- **Stop Functionality**: Stop running scripts if needed

## Usage

### Method 1: Direct Launch

```bash
python utils_runner_gui.py
```

### Method 2: Using the Launcher

```bash
python launch_utils_gui.py
```

### Method 3: Make executable and run

```bash
chmod +x launch_utils_gui.py
./launch_utils_gui.py
```

## GUI Controls

- **Scripts List**: Browse all available Python scripts in the utils folder
- **Run Selected**: Execute the currently selected script
- **Run All Scripts**: Execute all scripts one by one
- **Stop Selected**: Stop the currently running script
- **Clear Output**: Clear the output display area

## Output Color Coding

- **Black**: Normal information messages
- **Green**: Success messages (✅)
- **Red**: Error messages (❌)
- **Orange**: Warning messages

## Available Scripts

The GUI automatically detects all `.py` files in the utils folder, including:

- `backup_manager.py` - Backup and restore system
- `batch_processor.py` - Batch processing for simulations
- `config_manager.py` - Configuration management
- `data_quality_assessment.py` - Data quality tools
- `error_handler.py` - Error handling utilities
- `logging_config.py` - Logging configuration
- `parameter_validator.py` - Parameter validation
- `performance_profiler.py` - Performance profiling
- And many more...

## Troubleshooting

### Common Issues

1. **Import Errors**: Some scripts may have dependency issues. The GUI will show these errors in the output.

2. **Missing Dependencies**: Ensure all required packages are installed:

   ```bash
   pip install -r requirements.txt
   ```

3. **Permission Issues**: Make sure the scripts have execute permissions:

   ```bash
   chmod +x utils/*.py
   ```

### Script-Specific Notes

- **backup_manager.py**: Creates backups in the `backups/` directory
- **batch_processor.py**: May take time to run due to simulations
- **validation protocols**: Require the correct validation files to exist

## Requirements

- Python 3.7+
- tkinter (usually comes with Python)
- All dependencies listed in `requirements.txt`

## Installation

### Install Dependencies

Before running the GUI, install all required dependencies:

```bash
# If you have pip restrictions (externally managed environment)
pip install --break-system-packages -r requirements.txt

# Or if you're in a virtual environment
pip install -r requirements.txt
```

### Common Missing Dependencies

If you encounter missing module errors, you may need to install specific packages:

```bash
# For logging issues
pip install --break-system-packages loguru

# For data processing
pip install --break-system-packages numpy pandas scipy

# For progress bars
pip install --break-system-packages tqdm

# For configuration management
pip install --break-system-packages pyyaml
```

## File Structure

```text
apgi-validation/
├── utils_runner_gui.py          # Main GUI application
├── launch_utils_gui.py          # Launcher script
├── utils/                       # Utils folder with scripts
│   ├── backup_manager.py
│   ├── batch_processor.py
│   ├── config_manager.py
│   └── ... (other scripts)
└── README_UTILS_GUI.md          # This file
```

## Tips

1. **Start Small**: Test individual scripts before running all scripts
2. **Monitor Output**: Watch the output for any errors or warnings
3. **Stop When Needed**: Use the stop button if a script is taking too long
4. **Clear Output**: Use clear output to clean up the display between runs

## Development

The GUI is built using:

- **tkinter**: For the GUI framework
- **threading**: For non-blocking script execution
- **subprocess**: For running scripts in separate processes
- **pathlib**: For file system operations

Feel free to modify the GUI to add more features or customize the interface!
