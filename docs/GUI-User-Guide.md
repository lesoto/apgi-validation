# APGI Theory Framework - GUI User Guide

## Overview

The APGI Framework provides several graphical user interfaces (GUIs) for different use cases. This guide covers how to use each GUI application effectively.

## Available GUI Applications

### 1. Validation GUI

**Purpose:** Run validation protocols and analyze results visually
**Launch Command:** `python main.py gui validation`

### 2. Psychological States GUI

**Purpose:** Model and analyze psychological states dynamics
**Launch Command:** `python main.py gui psychological`

### 3. Web-Based Analysis Interface

**Purpose:** Browser-based analysis and visualization
**Launch Command:** `python main.py gui analysis --host localhost --port 8080`

---

## Validation GUI User Guide

### Launching the Validation GUI

```bash
# Basic launch
python main.py gui validation

# With debug mode
python main.py gui validation --debug

# With custom configuration
python main.py gui validation --config profiles/anxiety-disorder.yaml
```

### Main Interface

The Validation GUI consists of several key components:

#### 1. Protocol Selection

- **Protocol 1:** Basic validation tests
- **Protocol 2:** Cross-validation analysis
- **Protocol 3:** Robustness testing

#### 2. Parameter Configuration

- **Model Parameters:** Adjust tau_S, tau_theta, alpha, etc.
- **Simulation Settings:** Set steps, dt, and output options
- **Validation Settings:** Configure cross-validation folds and significance levels

#### 3. Results Visualization

- **Performance Plots:** Real-time visualization of model performance
- **Statistical Summaries:** View validation metrics and confidence intervals
- **Data Export:** Export results in CSV, JSON, or PDF formats

### Step-by-Step Workflow

1. **Select Protocol**

   - Choose the validation protocol you want to run
   - Each protocol has different validation criteria and tests

2. **Configure Parameters**

   - Adjust model parameters using the sliders or input fields
   - Load a predefined profile if desired
   - Validate parameter ranges before running

3. **Run Validation**

   - Click "Run Validation" to start the process
   - Monitor progress in the status bar
   - View real-time results in the plots panel

4. **Analyze Results**
   - Review performance metrics in the results panel
   - Compare against baseline or previous runs
   - Export results for further analysis

### Validation GUI Use Cases

#### Research Validation

```bash
# Launch with research profile
python main.py gui validation --config config/profiles/research-default.yaml
```

#### Clinical Analysis

```bash
# Launch with anxiety disorder profile
python main.py gui validation --config config/profiles/anxiety-disorder.yaml
```

#### Parameter Sensitivity Analysis

1. Select "Parameter Sensitivity" from the analysis menu

2. Choose parameters to analyze

3. Set range and step size

4. Run sensitivity analysis

5. View tornado plots and sensitivity indices

---

## Psychological States GUI User Guide

### Launching the Psychological States GUI

```bash
# Basic launch
python main.py gui psychological

# With debug mode
python main.py gui psychological --debug

# With specific disorder profile
python main.py gui psychological --config config/profiles/adhd.yaml
```

### Psychological States GUI Interface

The Psychological States GUI provides tools for modeling and analyzing psychological dynamics:

#### 1. State Space Visualization

- **2D/3D Plots:** Visualize psychological state trajectories

- **Phase Portraits:** Analyze system dynamics

- **Attractor Maps:** Identify stable states and transitions

#### 2. Parameter Control Panel

- **Cognitive Parameters:** Attention, memory, processing speed

- **Emotional Parameters:** Arousal, valence, mood

- **Behavioral Parameters:** Response patterns, decision making

#### 3. Simulation Controls

- **Time Controls:** Play, pause, reset simulation

- **Speed Control:** Adjust simulation speed

- **Initial Conditions:** Set starting psychological states

### Psychological States Workflow

1. **Set Initial Conditions**
   - Configure initial psychological state values
   - Choose from preset conditions or custom values
   - Validate initial state feasibility

2. **Configure Parameters**

   - Adjust cognitive, emotional, and behavioral parameters

   - Load disorder-specific profiles if needed

   - Use parameter constraints to ensure realistic values

3. **Run Simulation**

   - Start the simulation and observe state evolution

   - Monitor key metrics in real-time

   - Pause to analyze specific time points

4. **Analyze Dynamics**

   - Examine state trajectories and phase portraits

   - Identify attractors and transition points

   - Calculate stability metrics and transition probabilities

### Psychological States GUI Use Cases

#### ADHD Analysis

```bash
# Launch with ADHD profile
python main.py gui psychological --config config/profiles/adhd.yaml
```

- Analyze attention dynamics

- Model hyperactivity patterns

- Study impulse control mechanisms

#### Anxiety Disorder Modeling

```bash
# Launch with anxiety profile
python main.py gui psychological --config config/profiles/anxiety-disorder.yaml
```

- Model anxiety state dynamics

- Study worry and rumination patterns

- Analyze threat response mechanisms

#### Cognitive Performance

```bash
# Launch with default cognitive profile
python main.py gui psychological
```

- Model working memory dynamics

- Study attention and focus patterns

- Analyze decision-making processes

---

## Web-Based Analysis Interface User Guide

### Launching the Web Interface

```bash
# Default launch (localhost:8080)
python main.py gui analysis

# Custom host and port
python main.py gui analysis --host 0.0.0.0 --port 9000

# With authentication
python main.py gui analysis --auth --username admin --password secure123
```

### Accessing the Interface

1. Open your web browser

2. Navigate to `http://localhost:8080` (or your custom host/port)

3. Login if authentication is enabled

### Main Features

#### 1. Dashboard

- **System Status:** Overview of framework status and resources

- **Recent Analyses:** Quick access to recent analysis results

- **Quick Actions:** Common tasks and shortcuts

#### 2. Analysis Tools

- **Formal Model Analysis:** Run formal model simulations

- **Parameter Estimation:** Estimate model parameters from data

- **Validation Protocols:** Execute validation tests

- **Performance Profiling:** Analyze system performance

#### 3. Data Management

- **Upload Data:** Import experimental data for analysis

- **Data Visualization:** Interactive plots and charts

- **Export Results:** Download analysis results

#### 4. Configuration Management

- **Parameter Tuning:** Adjust model parameters

- **Profile Management:** Save and load parameter profiles

- **System Settings:** Configure framework behavior

### Web Interface Workflow

1. **Upload Data**

   - Navigate to the Data Management section

   - Upload your experimental data (CSV, JSON, etc.)

   - Validate data format and structure

2. **Configure Analysis**
   - Select analysis type from the Analysis Tools menu

   - Configure parameters and settings

   - Choose output format and visualization options

3. **Run Analysis**
   - Start the analysis job

   - Monitor progress in real-time

   - Receive notifications when complete

4. **View Results**
   - Examine interactive plots and visualizations

   - Download detailed reports

   - Share results with collaborators

### API Access

The web interface also provides a REST API for programmatic access:

```python
import requests

# Run analysis via API
response = requests.post('http://localhost:8080/api/analysis/formal-model',
                        json={'parameters': {'tau_S': 0.5, 'tau_theta': 30.0}})
results = response.json()

# Get analysis status
status = requests.get(f'http://localhost:8080/api/analysis/{results["id"]}/status')
```

---

## Troubleshooting

### Common Issues

#### GUI Won't Launch

1. **Check Dependencies:** Ensure all required packages are installed

   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Python Path:** Make sure you're using the correct Python environment

   ```bash
   which python
   ```

3. **Check Display:** For GUI applications, ensure display is available

   ```bash
   echo $DISPLAY
   ```

#### Web Interface Not Accessible

1. **Check Port:** Ensure port is not already in use

   ```bash
   lsof -i :8080
   ```

2. **Firewall:** Check if firewall is blocking the port

   ```bash
   # On macOS
   sudo lsof -i :8080
   ```

3. **Network:** Verify network connectivity and host binding

#### Performance Issues

1. **Memory Usage:** Monitor system resources

   ```bash
   htop  # or Activity Monitor on macOS
   ```

2. **GPU Acceleration:** Enable GPU acceleration if available

   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

3. **Parallel Processing:** Configure number of parallel processes

   ```bash
   export OMP_NUM_THREADS=4
   ```

### Getting Help

1. **Debug Mode:** Launch GUIs with `--debug` flag for detailed logging

2. **Documentation:** Refer to the full documentation in the `docs/` directory

3. **Community:** Check the project repository for issues and discussions

4. **Logs:** Check log files in the `logs/` directory for error details

---

## Additional Resources

For more help and troubleshooting:

### Validation GUI

- `Ctrl+N`: New validation run

- `Ctrl+O`: Open configuration file

- `Ctrl+S`: Save current configuration

- `Ctrl+R`: Run validation

- `Ctrl+E`: Export results

- `F5`: Refresh plots

- `Ctrl+Q`: Quit application

### Psychological States GUI

- `Space`: Play/Pause simulation

- `R`: Reset simulation

- `Ctrl+S`: Save current state

- `Ctrl+L`: Load saved state

- `Ctrl+E`: Export trajectory data

- `F11`: Toggle fullscreen

- `Ctrl+Q`: Quit application

### Web Interface Shortcuts

- `Ctrl+/`: Toggle keyboard shortcuts help

- `Ctrl+Shift+D`: Toggle dark mode

- `Esc`: Close modal dialogs

- `Enter`: Confirm actions

- `Tab`: Navigate between form fields

---

## Tips and Best Practices

### Performance Optimization

1. **Use Appropriate Data Sizes:** Start with smaller datasets for testing

2. **Configure Caching:** Enable result caching for repeated analyses

3. **Parallel Processing:** Use multiple cores for computationally intensive tasks

4. **Memory Management:** Clear unused data and results regularly

### Data Management

1. **Backup Configurations:** Save important parameter configurations

2. **Version Control:** Track changes to analysis parameters

3. **Data Validation:** Always validate input data before analysis

4. **Documentation:** Keep detailed notes on analysis parameters and results

### Collaboration

1. **Share Profiles:** Export and share parameter profiles with collaborators

2. **Reproducible Results:** Use fixed random seeds for reproducible simulations

3. **Standard Formats:** Use standard data formats for sharing results

4. **Documentation:** Document analysis workflows for reproducibility
