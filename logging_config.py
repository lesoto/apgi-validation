"""
APGI Theory Framework - Logging Configuration
============================================

Comprehensive logging system using loguru with:
- Multiple log levels and formatting
- Rotating log files
- Structured output for debugging
- Performance metrics tracking
- Error reporting and alerts
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import traceback
from typing import Optional, Dict, Any
from loguru import logger

# Project root directory
PROJECT_ROOT = Path(__file__).parent
LOGS_DIR = PROJECT_ROOT / 'logs'

# Ensure logs directory exists
LOGS_DIR.mkdir(exist_ok=True)

class APGILogger:
    """Advanced logging system for APGI framework."""
    
    def __init__(self, log_level: str = "INFO", enable_console: bool = True):
        self.log_level = log_level.upper()
        self.enable_console = enable_console
        self.log_files = {}
        self.performance_metrics = {}
        self.error_counts = {}
        self.logger = logger  # Expose the loguru logger
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure loguru logger with custom settings."""
        # Remove default logger
        logger.remove()
        
        # Console logger
        if self.enable_console:
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                       "<level>{message}</level>",
                level=self.log_level,
                colorize=True
            )
        
        # Main log file with rotation
        main_log_file = LOGS_DIR / "apgi_framework.log"
        logger.add(
            main_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            enqueue=True
        )
        
        # Error-specific log file
        error_log_file = LOGS_DIR / "errors.log"
        logger.add(
            error_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
            level="ERROR",
            rotation="5 MB",
            retention="60 days",
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
        
        # Performance metrics log
        performance_log_file = LOGS_DIR / "performance.log"
        logger.add(
            performance_log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | PERFORMANCE | {extra[metric]} | {extra[value]} | {extra[unit]}",
            level="INFO",
            rotation="5 MB",
            retention="7 days",
            filter=lambda record: "metric" in record["extra"]
        )
        
        # Structured JSON log for machine processing
        json_log_file = LOGS_DIR / "structured.jsonl"
        logger.add(
            json_log_file,
            format="{message}",
            level="DEBUG",
            rotation="20 MB",
            retention="30 days",
            serialize=True,
            enqueue=True
        )
        
        self.log_files = {
            'main': main_log_file,
            'errors': error_log_file,
            'performance': performance_log_file,
            'structured': json_log_file
        }
    
    def log_simulation_start(self, simulation_type: str, parameters: Dict[str, Any]):
        """Log the start of a simulation with parameters."""
        logger.info(f"Starting {simulation_type} simulation")
        logger.bind(simulation_type=simulation_type, parameters=parameters).debug("Simulation parameters")
    
    def log_simulation_end(self, simulation_type: str, duration: float, results_summary: Dict[str, Any]):
        """Log the end of a simulation with results summary."""
        logger.info(f"Completed {simulation_type} simulation in {duration:.2f} seconds")
        logger.bind(simulation_type=simulation_type, duration=duration, results=results_summary).debug("Simulation results")
    
    def log_parameter_estimation(self, method: str, parameters: Dict[str, Any], log_likelihood: float):
        """Log parameter estimation results."""
        logger.info(f"Parameter estimation using {method} method")
        logger.bind(method=method, parameters=parameters, log_likelihood=log_likelihood).info("Estimation results")
    
    def log_validation_result(self, protocol: str, passed: bool, metrics: Dict[str, float]):
        """Log validation protocol results."""
        status = "PASSED" if passed else "FAILED"
        logger.info(f"Validation protocol {protocol}: {status}")
        logger.bind(protocol=protocol, passed=passed, metrics=metrics).debug("Validation metrics")
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "seconds"):
        """Log performance metrics."""
        logger.info(f"Performance: {metric_name} = {value:.3f} {unit}")
        logger.bind(metric=metric_name, value=value, unit=unit).debug("Performance metric details")
        
        # Track metrics for analysis
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        self.performance_metrics[metric_name].append({
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all performance metrics."""
        summary = {}
        for metric_name, values in self.performance_metrics.items():
            if values:
                numeric_values = [v['value'] for v in values]
                summary[metric_name] = {
                    'count': len(values),
                    'mean': sum(numeric_values) / len(numeric_values),
                    'min': min(numeric_values),
                    'max': max(numeric_values),
                    'unit': values[0]['unit'],
                    'latest': values[-1]['value'],
                    'latest_timestamp': values[-1]['timestamp']
                }
        return summary
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log errors with additional context information."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Track error counts
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        logger.error(f"Error in {context.get('operation', 'unknown operation')}: {error_message}")
        logger.bind(
            error_type=error_type,
            error_message=error_message,
            context=context,
            traceback=traceback.format_exc()
        ).exception("Error details")
    
    def log_data_processing(self, data_type: str, file_path: str, records_processed: int, duration: float):
        """Log data processing operations."""
        logger.info(f"Processed {records_processed} {data_type} records from {file_path} in {duration:.2f}s")
        logger.bind(
            data_type=data_type,
            file_path=file_path,
            records_processed=records_processed,
            duration=duration,
            throughput=records_processed/duration
        ).debug("Data processing details")
    
    def log_model_configuration(self, model_name: str, config: Dict[str, Any]):
        """Log model configuration details."""
        logger.info(f"Configuring {model_name} model")
        logger.bind(model_name=model_name, configuration=config).debug("Model configuration")
    
    def log_system_info(self):
        """Log system information for debugging."""
        import platform
        import psutil
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            'disk_free': f"{psutil.disk_usage('.').free / (1024**3):.2f} GB"
        }
        
        logger.bind(system_info=system_info).info("System information")
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error occurrences."""
        return self.error_counts.copy()
    
    def export_logs(self, output_file: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, 
                   format_type: str = "json", log_level: Optional[str] = None):
        """Export logs to a file for analysis.
        
        Args:
            output_file: Path to output file
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            format_type: Export format ('json', 'csv', 'txt')
            log_level: Optional log level filter ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        """
        import re
        import csv
        from datetime import datetime
        
        logger.info(f"Exporting logs to {output_file} (format: {format_type})")
        
        # Find all log files
        log_files = list(LOGS_DIR.glob("*.log"))
        if not log_files:
            logger.warning("No log files found to export")
            return False
        
        # Parse log entries
        log_entries = []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse log line format - handle multiple patterns with flexible spacing
                        patterns = [
                            # Standard format with newlines: "2025-12-31 19:22:13.802 | INFO     | \n config_manager:_load_config:208 - message"
                            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+)\s+\| ?\s*([^-]+) - (.+)',
                            # Alternative format: "2025-12-31 19:22:13.802 | INFO | module:function - message"
                            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+)\s+\| ([^-]+) - (.+)',
                            # Simple format: "2025-12-31 19:22:13.802 | INFO | message"
                            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+)\s+\| (.+)'
                        ]
                        
                        match = None
                        for pattern in patterns:
                            match = re.match(pattern, line, re.DOTALL)
                            if match:
                                break
                        
                        if match:
                            if len(match.groups()) == 4:
                                timestamp_str, level, location, message = match.groups()
                            elif len(match.groups()) == 3:
                                timestamp_str, level, message = match.groups()
                                location = "unknown"
                            else:
                                continue
                            
                            try:
                                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                            except ValueError:
                                continue
                            
                            # Apply filters
                            if start_time and timestamp < start_time:
                                continue
                            if end_time and timestamp > end_time:
                                continue
                            if log_level and level != log_level.upper():
                                continue
                            
                            log_entry = {
                                'timestamp': timestamp_str,
                                'datetime': timestamp,
                                'level': level,
                                'location': location.strip(),
                                'message': message.strip(),
                                'file': str(log_file)
                            }
                            log_entries.append(log_entry)
                            
            except Exception as e:
                logger.warning(f"Error reading log file {log_file}: {e}")
        
        if not log_entries:
            logger.warning("No log entries found matching the criteria")
            return False
        
        # Sort by timestamp
        log_entries.sort(key=lambda x: x['datetime'])
        
        # Export based on format
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type.lower() == 'json':
                # JSON format
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(log_entries, f, indent=2, default=str)
                    
            elif format_type.lower() == 'csv':
                # CSV format
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['timestamp', 'level', 'location', 'message', 'file']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for entry in log_entries:
                        writer.writerow({k: v for k, v in entry.items() if k in fieldnames})
                        
            elif format_type.lower() == 'txt':
                # Plain text format
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"APGI Framework Log Export\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total entries: {len(log_entries)}\n")
                    f.write(f"Time range: {log_entries[0]['timestamp']} to {log_entries[-1]['timestamp']}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for entry in log_entries:
                        f.write(f"{entry['timestamp']} | {entry['level']:8} | {entry['location']} | {entry['message']}\n")
                        
            else:
                logger.error(f"Unsupported export format: {format_type}")
                return False
            
            logger.info(f"Successfully exported {len(log_entries)} log entries to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting logs: {e}")
            return False
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files."""
        import glob
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_files = 0
        
        for log_file in LOGS_DIR.glob("*.log*"):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    log_file.unlink()
                    cleaned_files += 1
            except Exception as e:
                logger.warning(f"Could not delete old log file {log_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_files} old log files")


# Global logger instance
apgi_logger = APGILogger()

# Convenience functions for common logging tasks
def log_simulation(simulation_type: str, parameters: Dict[str, Any], duration: float, results: Dict[str, Any]):
    """Convenience function for logging complete simulation."""
    apgi_logger.log_simulation_start(simulation_type, parameters)
    apgi_logger.log_simulation_end(simulation_type, duration, results)

def log_performance(metric_name: str, value: float, unit: str = "seconds"):
    """Convenience function for performance logging."""
    apgi_logger.log_performance_metric(metric_name, value, unit)

def log_error(error: Exception, operation: str, **context):
    """Convenience function for error logging."""
    context['operation'] = operation
    apgi_logger.log_error_with_context(error, context)

# Decorators for automatic logging
def log_execution_time(metric_name: str = "execution_time"):
    """Decorator to automatically log function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                log_performance(metric_name, duration)
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                log_performance(f"{metric_name}_failed", duration)
                log_error(e, f"Function {func.__name__}")
                raise
        return wrapper
    return decorator

def log_function_call(level: str = "DEBUG"):
    """Decorator to log function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.bind(function=func.__name__, args=args, kwargs=kwargs).log(level, f"Calling {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.bind(function=func.__name__).log(level, f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger.bind(function=func.__name__).error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test logging system
    apgi_logger.log_system_info()
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test performance logging
    apgi_logger.log_performance_metric("test_metric", 1.23, "seconds")
    
    # Test error logging with context
    try:
        raise ValueError("Test error for logging")
    except Exception as e:
        apgi_logger.log_error_with_context(e, {"operation": "test", "user": "test_user"})
    
    # Test simulation logging
    apgi_logger.log_simulation_start("test_simulation", {"param1": 1.0, "param2": 2.0})
    apgi_logger.log_simulation_end("test_simulation", 5.5, {"result1": 10, "result2": 20})
    
    print(f"Log files created in: {LOGS_DIR}")
    print("Performance summary:", apgi_logger.get_performance_summary())
    print("Error summary:", apgi_logger.get_error_summary())
