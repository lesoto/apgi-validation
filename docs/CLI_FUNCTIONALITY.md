# APGI Theory Framework - CLI Functionality Summary

## Overview
The APGI validation framework provides a comprehensive command-line interface with **41 available commands** organized across multiple functional categories.

## Global Options (Available with all commands)

```bash
python3 main.py [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS]
```

| Option | Flag | Description |
|--------|------|-------------|
| `--version` | - | Show the version and exit |
| `--config-file TEXT` | - | Override configuration file path |
| `--log-level TEXT` | - | Override logging level |
| `--verbose` | `-v` | Enable verbose output |
| `--quiet` | `-q` | Suppress non-error output |
| `--token TEXT` | - | JWT authentication token for secured operations |
| `--help` | - | Show help message and exit |

## Available Commands (41 total)

### 1. Core Information & Framework
- **`info`** - Show framework information and status
- **`performance`** - Show performance metrics and statistics
  - `--detailed` - Show detailed performance breakdown

### 2. Monitoring & Logging
- **`logs`** - View and monitor log files
  - `--tail N` - Show last N lines
  - `--follow` - Follow log output in real-time
  - `--level` - Filter by log level
  - `--export` - Export logs to file
  
- **`errors`** - Show error summary and statistics
  - `--category` - Filter by error category
  - `--severity` - Filter by severity level
  - `--reset` - Clear error log
  
- **`analyze-logs`** - Analyze log files for patterns and insights
  - `--limit` - Limit number of entries to analyze
  
- **`monitor-performance`** - Monitor performance metrics for APGI operations
  - `--sample-rate` - Set sampling rate (0.0-1.0)
  
- **`test-errors`** - Test error handling system
  - `--test-config` - Test configuration errors
  - `--test-validation` - Test validation errors
  - `--test-data` - Test data errors

### 3. Configuration Management
- **`config`** - Manage configuration settings for APGI framework
  - `--show` - Display current configuration
  - `--set KEY VALUE` - Set configuration value
  - `--reset` - Reset to defaults
  
- **`config-group`** - Manage APGI configuration (group)
  - `explain` - Show configuration precedence and resolution order
  
- **`config-version`** - Create a version snapshot of current configuration
  - `--description` - Add description
  - `--author` - Set author name
  
- **`config-versions`** - List configuration versions
  - `--limit` - Limit number of versions shown
  
- **`config-restore`** - Restore configuration from version
  - `--version-id` - Specific version to restore
  
- **`config-diff`** - Compare current configuration with last version

### 4. Backup & Restore
- **`backup`** - Create backup of APGI framework data
  - `--components` - Specify components to backup
  - `--description` - Add backup description
  - `--compress` - Compress backup files
  
- **`backups`** - List available backups
  - `--limit` - Limit number of backups shown
  
- **`restore`** - Restore from backup
  - `--backup-id` - Specific backup to restore
  - `--keep-current` - Keep current version
  
- **`delete-backup`** - Delete backup(s)
  - `--backup-id` - Specific backup to delete
  - `--keep-count` - Keep N most recent backups
  - `--cleanup-all` - Delete all backups

### 5. Data Operations
- **`process-data`** - Run data processing pipelines on raw data
  - `--input-file` - Input data file
  - `--output-file` - Output file path
  - `--format` - Data format specification
  - `--normalize` - Normalize data
  - `--validate` - Validate data
  
- **`export-data`** - Export data in various formats
  - `--input-file` - Input data file
  - `--output-file` - Output file path
  - `--format` - Export format (csv, json, hdf5, etc)
  - `--compress` - Compress output
  
- **`import-data`** - Import data from various formats into CSV
  - `--input-file` - Input data file
  - `--output-file` - Output CSV file
  - `--format` - Input format
  - `--validate` - Validate imported data
  
- **`cache-cmd`** - Manage cache operations
  - `action` - Cache action (clear, list, stats)
  - `--sources` - Specify cache sources
  - `--max-workers` - Number of parallel workers

### 6. Visualization
- **`visualize`** - Create visualizations of APGI results and data
  - `--input-file` - Data file to visualize
  - `--output-file` - Output visualization file
  - `--type` - Plot type (timeseries, scatter, heatmap, distribution)
  - `--figsize` - Figure size (width,height)
  - `--style` - Plot style
  - `--palette` - Color palette
  
- **`dashboard`** - Generate static HTML dashboards for APGI framework
  - `--output-dir` - Output directory
  - `--dashboard-type` - Type of dashboard
  - `--open-browser` - Open in browser automatically
  
- **`performance-dashboard`** - Launch comprehensive performance monitoring dashboard
  - `--host` - Server host (default: localhost)
  - `--port` - Server port (default: 5000)
  - `--debug` - Enable debug mode

### 7. GUI & Interactive Interfaces
- **`gui`** - Launch graphical user interface for APGI framework
  - `--gui-type` - GUI type (validation, psychological, analysis)
  - `--port` - Server port
  - `--host` - Server host
  - `--debug` - Enable debug mode

### 8. Validation Protocols
- **`validate`** - Run validation protocols
  - `--protocol` - Specific protocol to run
  - `--parallel` - Run protocols in parallel
  - `--sequential` - Run protocols sequentially
  - `--output-dir` - Output directory for results
  
- **`validate-pipeline`** - Run validation protocols with integrated preprocessing pipeline
  - Similar options to `validate`
  
- **`comprehensive-validation`** - Run comprehensive validation across all APGI priorities
  - `--all-protocols` - Run all validation protocols
  - `--priorities` - Specific priorities to validate

### 9. Falsification/Testing
- **`falsify`** - Execute falsification testing protocols
  - `--protocol` - Specific protocol to falsify
  - `--output-dir` - Output directory
  
- **`falsification`** - Run falsification testing protocols
  - Same options as `falsify`

### 10. Advanced Analysis & Theory
- **`formal-model`** - Run formal model simulations
  - `--simulation-steps` - Number of simulation steps
  - `--parameters` - Model parameters
  - `--output-file` - Output file
  
- **`multimodal`** - Execute multimodal data integration
  - `--input-file` - Input data file
  - `--modalities` - Specify modalities to integrate
  - `--output-file` - Output file
  
- **`estimate-params`** - Perform Bayesian parameter estimation for APGI framework
  - `--input-data` - Input data file
  - `--iterations` - MCMC iterations
  - `--output-file` - Results file
  
- **`bayesian-estimation`** - Run Bayesian parameter estimation and model comparison
  - Extended parameter estimation with model comparison
  
- **`cross-species`** - Run cross-species scaling analysis for consciousness measurements
  - `--input-file` - Input data
  - `--species` - Species to analyze
  
- **`neural-signatures`** - Run Priority 1: Convergent Neural Signatures validation
  - Priority-specific validation
  
- **`causal-manipulations`** - Run Priority 2: Causal Manipulations validation
  - TMS intervention analysis
  
- **`quantitative-fits`** - Run Priority 3: Quantitative Model Fits validation
  - Model fitting and quality assessment
  
- **`clinical-convergence`** - Run Priority 4: Clinical and Cross-Species Convergence validation
  - Clinical and comparative analysis

### 11. Open Science & Infrastructure
- **`open-science`** - Manage open science infrastructure
  - `--init` - Initialize OSF connection
  - `--upload` - Upload results to OSF
  - `--download` - Download data from OSF
  - `--sync` - Sync with OSF project

## Example Usage

```bash
# View framework information
python3 main.py info

# Run performance monitoring with verbose output
python3 main.py -v performance --detailed

# Process and export data
python3 main.py process-data --input-file data.csv --output-file processed.csv
python3 main.py export-data --input-file processed.csv --output-file result.json --format json

# Create visualization dashboard
python3 main.py dashboard --output-dir ./dashboards --open-browser

# Run validation protocols
python3 main.py validate --protocol VP_01 --parallel

# Run falsification testing
python3 main.py falsify --output-dir ./results

# Launch GUI
python3 main.py gui --gui-type validation --port 8080

# Perform Bayesian analysis
python3 main.py bayesian-estimation --input-data data.csv --iterations 5000

# Create configuration snapshot
python3 main.py config-version --description "Pre-validation config" --author "researcher"

# Monitor performance in real-time
python3 main.py monitor-performance --sample-rate 0.5

# View logs with filtering
python3 main.py logs --tail 50 --level INFO --export logs.txt
```
