# APGI Framework Data Directory

This directory contains sample datasets, data validation utilities, preprocessing pipelines, and cache management for the APGI Theory Framework.

## Directory Structure

```text
data/
├── sample_data_generator.py      # Generate realistic multimodal datasets
├── data_validation.py            # Data validation and quality assessment
├── preprocessing_pipelines.py    # Specialized preprocessing pipelines
├── cache_manager.py             # Advanced caching system
├── README.md                    # This file
├── cache/                       # Cache storage directory
│   ├── cache_metadata.json     # Cache metadata
│   └── *.cache                  # Cached data files
├── processed/                   # Processed data output
│   ├── *_processed.json        # Processed datasets
│   └── *_processing_report.json # Processing reports
└── sample_datasets/             # Generated sample data
    ├── sub_01_sess_01.csv       # Subject 1, Session 1
    ├── sub_01_sess_01.json      # JSON format
    ├── sub_01_sess_01_metadata.json # Metadata
    ├── demo_demo.csv             # Demo dataset (smaller)
    ├── demo_demo.json            # Demo JSON format
    └── ...                      # Additional subject sessions
```

## Sample Datasets

The framework includes realistic sample datasets for testing and demonstration:

### Generated Datasets

- **sub_01_sess_01/02/03**: Multiple subjects with different sessions
- **demo_demo**: Smaller demo dataset for quick testing
- **Formats**: CSV and JSON for flexibility

### Data Features

Each dataset contains synchronized multimodal physiological data:

- **EEG Signals**: Fz and Pz channels with P300 events
- **Pupil Diameter**: Continuous measurement with blink artifacts
- **EDA**: Electrodermal activity with phasic/tonic components
- **Heart Rate**: BPM with respiratory sinus arrhythmia
- **Event Markers**: Stimulus onset markers
- **Metadata**: Subject ID, session ID, timestamps

### Data Characteristics

- **Sampling Rate**: 1000 Hz (full datasets), 250 Hz (demo)
- **Duration**: 60 seconds (full), 30 seconds (demo)
- **Quality**: Realistic noise, artifacts, and physiological variations
- **Events**: P300-like events every 10-15 seconds

## Data Validation

### Validation Features

- **Format Validation**: CSV and JSON structure checking
- **Data Quality Assessment**: Missing data, outliers, signal quality
- **Range Validation**: Physiologically plausible ranges
- **Temporal Consistency**: Sampling rate and timestamp validation
- **Quality Scoring**: Overall data quality metrics

### Usage

```python
from data.data_validation import DataValidator

# Initialize validator
validator = DataValidator(strict_mode=True)

# Validate file
report = validator.generate_validation_report('data/sub_01_sess_01.csv')
print(f"Quality score: {report['data_quality']['overall_score']:.1f}")
```

## Preprocessing Pipelines

### Available Pipelines

- **EEG Preprocessing**: Bandpass filtering, notch filtering, artifact correction
- **Pupil Preprocessing**: Blink detection, interpolation, normalization
- **EDA Preprocessing**: Lowpass filtering, phasic/tonic extraction
- **Heart Rate Preprocessing**: Outlier detection, interpolation, smoothing
- **Multimodal Integration**: Combined preprocessing with resampling

### Configuration

```python
from data.preprocessing_pipelines import PreprocessingConfig, MultimodalPreprocessingPipeline

# Configure preprocessing
config = PreprocessingConfig(
    eeg_bandpass_low=0.5,
    eeg_bandpass_high=40.0,
    target_sampling_rate=250.0
)

# Run pipeline
pipeline = MultimodalPreprocessingPipeline(config)
result = pipeline.run_complete_pipeline('data/demo_demo.csv')
```

## Cache Management

### Cache Features

- **Intelligent Caching**: Automatic caching of computed results
- **LRU Eviction**: Least Recently Used eviction policy
- **TTL Support**: Time-to-live for cached entries
- **Size Management**: Configurable cache size limits
- **Thread Safety**: Safe for concurrent access

### Cache Types

- **Data Cache**: Preprocessed data caching
- **Simulation Cache**: Model simulation results
- **Validation Cache**: Protocol validation results
- **General Cache**: Arbitrary data caching

### Cache Usage

```python
from data.cache_manager import CacheManager, DataCache

# Initialize cache
cache = CacheManager(max_size_mb=100)
data_cache = DataCache(cache)

# Cache preprocessed data
data_cache.cache_preprocessed_data('input.csv', config, processed_data)

# Retrieve cached data
cached_data = data_cache.get_preprocessed_data('input.csv', config)
```

## Quick Start Examples

### 1. Generate Sample Data

```bash
python data/sample_data_generator.py
```

### 2. Validate Data Quality

```python
from data.data_validation import DataValidator

validator = DataValidator()
report = validator.generate_validation_report('data/demo_demo.csv')
```

### 3. Preprocess Data

```python
from data.preprocessing_pipelines import MultimodalPreprocessingPipeline

pipeline = MultimodalPreprocessingPipeline()
result = pipeline.run_complete_pipeline('data/demo_demo.csv')
```

### 4. Use Caching

```python
from data.cache_manager import cached, CacheManager

cache = CacheManager()

@cached(ttl=3600, cache_manager=cache)
def expensive_computation(params):
    # Your expensive computation here
    return result

# Results will be cached for 1 hour
result = expensive_computation(my_params)
```

## Data Quality Guidelines

### Good Data Quality

- **Missing Data**: < 5% missing values
- **Outliers**: < 5% extreme values
- **Signal Quality**: > 80% quality score
- **Temporal Consistency**: Regular sampling intervals
- **Physiological Ranges**: Within expected biological limits

### Data Issues to Watch

- **Flat Segments**: No signal variation
- **Extreme Values**: Beyond physiological limits
- **Irregular Sampling**: Variable time intervals
- **Artifacts**: Sudden jumps or drops
- **Missing Events**: No stimulus markers

## Integration with CLI

The data infrastructure integrates seamlessly with the APGI CLI:

```bash
# Validate data
python main.py validate --all-protocols

# Process data with caching
python main.py multimodal --input-data data/demo_demo.csv --output-file results.csv

# Visualize data quality
python main.py visualize --input-file data/demo_demo.csv --plot-type heatmap
```

## Performance Considerations

### Memory Usage

- **Large Datasets**: Process in chunks to avoid memory issues
- **Cache Size**: Monitor cache usage and adjust limits as needed
- **Preprocessing**: Use appropriate sampling rates for your use case

### Processing Speed

- **Caching**: Enable caching for repeated computations
- **Parallel Processing**: Use parallel validation when available
- **Efficient Algorithms**: Optimized filtering and preprocessing

## Troubleshooting

### Common Issues

1. **File Not Found**: Check file paths and ensure data directory exists
2. **Memory Errors**: Reduce dataset size or use chunked processing
3. **Cache Issues**: Clear cache if corrupted entries detected
4. **Validation Errors**: Check data format and required columns

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run validation with detailed output
validator = DataValidator(strict_mode=True)
report = validator.generate_validation_report('data.csv')
```

## Extending the Data Infrastructure

### Adding New Preprocessors

1. Create specialized preprocessor class
2. Implement required methods
3. Add to main preprocessing pipeline
4. Update configuration options

### Adding New Validation Rules

1. Extend DataValidator class
2. Add new validation methods
3. Update quality scoring algorithm
4. Add to validation report

### Adding New Cache Types

1. Extend DataCache class
2. Add specialized caching methods
3. Update cache key generation
4. Add TTL configuration options

## Best Practices

1. **Always validate** data before processing
2. **Use caching** for expensive computations
3. **Monitor cache size** and clean up expired entries
4. **Document preprocessing** steps and parameters
5. **Test with sample data** before using real data
6. **Backup important data** before preprocessing
7. **Version control** preprocessing configurations
8. **Log processing steps** for reproducibility

## Support

For issues with the data infrastructure:

1. Check the troubleshooting guide in `docs/Troubleshooting.md`
2. Run validation protocols to diagnose issues
3. Check log files for error messages
4. Use the test framework to verify functionality

## Data Privacy and Security

- **No Real Subject Data**: All sample data is synthetically generated
- **Local Processing**: Data stays on your local machine
- **Secure Caching**: Cache files are local and can be cleared
- **Metadata Only**: Cached metadata contains no sensitive information
