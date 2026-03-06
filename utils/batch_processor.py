#!/usr/bin/env python3
"""
APGI Theory Framework - Batch Processing Utility
============================================

Advanced batch processing system for running multiple simulations,
validation protocols, and analyses in parallel with progress tracking.
"""

import importlib.util
import json
import pickle
import sys
import time
import hashlib
import hmac
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Import batch configuration
try:
    from .batch_config import BatchProcessorConfig
except ImportError:
    try:
        from utils.batch_config import BatchProcessorConfig
    except ImportError:
        # Fallback to basic config if not available
        class BatchProcessorConfig:
            def __init__(self, config_file=None):
                self.config_file = config_file

            def get_max_workers(self):
                return 4

            def get(self, key, default=None):
                return default


# Optional import for tqdm
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None  # type: ignore

import os

# Secure pickle functions with HMAC signing
PICKLE_SECRET_KEY = os.environ.get("PICKLE_SECRET_KEY")
if PICKLE_SECRET_KEY is not None:
    PICKLE_SECRET_KEY = PICKLE_SECRET_KEY.encode()


def secure_pickle_dump(obj: Any, file_path: Path) -> None:
    """Securely pickle an object with HMAC signature"""
    if PICKLE_SECRET_KEY is None:
        from utils.error_handler import error_handler, ErrorCategory, ErrorSeverity

        raise error_handler.handle_error(
            category=ErrorCategory.IMPORT,
            severity=ErrorSeverity.CRITICAL,
            code="MISSING_ENV_VAR",
            message="PICKLE_SECRET_KEY environment variable not set. Cannot use secure pickle.",
            suggestions=["Set PICKLE_SECRET_KEY environment variable"],
            user_action="Set the required environment variable before using secure pickle functions",
        )
    # Generate HMAC signature
    pickle_data = pickle.dumps(obj)
    signature = hmac.new(PICKLE_SECRET_KEY, pickle_data, hashlib.sha256).digest()

    # Write signature + data
    with open(file_path, "wb") as f:
        f.write(len(signature).to_bytes(4, "big"))  # Signature length
        f.write(signature)
        f.write(pickle_data)


def secure_pickle_load(file_path: Path) -> Any:
    """Securely load a pickled object with HMAC verification"""
    if PICKLE_SECRET_KEY is None:
        from utils.error_handler import error_handler, ErrorCategory, ErrorSeverity

        raise error_handler.handle_error(
            category=ErrorCategory.IMPORT,
            severity=ErrorSeverity.CRITICAL,
            code="MISSING_ENV_VAR",
            message="PICKLE_SECRET_KEY environment variable not set. Cannot use secure pickle.",
            suggestions=["Set PICKLE_SECRET_KEY environment variable"],
            user_action="Set the required environment variable before using secure pickle functions",
        )
    with open(file_path, "rb") as f:
        # Read signature length
        sig_len_bytes = f.read(4)
        if len(sig_len_bytes) != 4:
            raise ValueError("Invalid pickle file format")

        sig_len = int.from_bytes(sig_len_bytes, "big")

        # Read signature and data
        signature = f.read(sig_len)
        pickle_data = f.read()

        # Verify signature
        expected_signature = hmac.new(
            PICKLE_SECRET_KEY, pickle_data, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError(
                "Pickle file signature verification failed - possible tampering"
            )

        return pickle.loads(pickle_data)


# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load modules with hyphens using importlib
formal_model_spec = importlib.util.spec_from_file_location(
    "SurpriseIgnitionSystem",
    PROJECT_ROOT / "Falsification" / "Falsification-Protocol-4.py",
)
formal_model_module = importlib.util.module_from_spec(formal_model_spec)
formal_model_spec.loader.exec_module(formal_model_module)
SurpriseIgnitionSystem = formal_model_module.SurpriseIgnitionSystem

# Import logging_config using importlib to avoid package init issues
logging_config_spec = importlib.util.spec_from_file_location(
    "logging_config", PROJECT_ROOT / "utils" / "logging_config.py"
)
logging_config = importlib.util.module_from_spec(logging_config_spec)
logging_config_spec.loader.exec_module(logging_config)
apgi_logger = logging_config.apgi_logger


def load_validation_module(protocol):
    """Load validation module by name."""
    module_map = {
        "protocol_1": "Validation/Validation-Protocol-1.py",
        "protocol_2": "Validation/Validation-Protocol-2.py",
        "protocol_3": "Validation/Validation-Protocol-3.py",
        "protocol_4": "Validation/Validation-Protocol-4.py",
        "protocol_5": "Validation/Validation-Protocol-5.py",
        "protocol_6": "Validation/Validation-Protocol-6.py",
        "protocol_7": "Validation/Validation-Protocol-7.py",
        "protocol_8": "Validation/Validation-Protocol-8.py",
    }

    if protocol not in module_map:
        raise ValueError(f"Unknown validation protocol: {protocol}")

    module_path = PROJECT_ROOT / module_map[protocol]
    spec = importlib.util.spec_from_file_location(protocol, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class BatchJob:
    """Represents a single batch processing job."""

    job_id: str
    job_type: str  # 'simulation', 'validation', 'analysis'
    parameters: Dict[str, Any]
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    status: str = "pending"  # 'pending', 'running', 'completed', 'failed'
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class BatchProcessor:
    """Advanced batch processing system for APGI framework."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: Optional[bool] = None,
        config_file: Optional[Path] = None,
    ):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum number of parallel workers (overrides config)
            use_processes: Use processes instead of threads (overrides config)
            config_file: Path to configuration file
        """
        # Load configuration
        self.config = BatchProcessorConfig(config_file)

        # Use provided values or fall back to config
        self.max_workers = max_workers or self.config.get_max_workers()
        self.use_processes = (
            use_processes
            if use_processes is not None
            else self.config.get("use_processes", False)
        )

        self.jobs: List[BatchJob] = []
        self.results: Dict[str, Any] = {}

        # Choose executor type
        self.executor_class = (
            ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        )

    def add_simulation_job(
        self,
        job_id: str,
        params: Dict[str, Any],
        steps: int = 1000,
        dt: float = 0.01,
        output_file: Optional[str] = None,
    ) -> str:
        """Add a simulation job to the batch."""
        job = BatchJob(
            job_id=job_id,
            job_type="simulation",
            parameters={"params": params, "steps": steps, "dt": dt},
            output_file=output_file,
        )
        self.jobs.append(job)
        return job_id

    def add_validation_job(
        self,
        job_id: str,
        protocol: str,
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> str:
        """Add a validation job to the batch."""
        job = BatchJob(
            job_id=job_id,
            job_type="validation",
            parameters={"protocol": protocol},
            input_file=input_file,
            output_file=output_file,
        )
        self.jobs.append(job)
        return job_id

    def add_analysis_job(
        self,
        job_id: str,
        analysis_type: str,
        input_file: str,
        parameters: Dict[str, Any] = None,
        output_file: Optional[str] = None,
    ) -> str:
        """Add an analysis job to the batch."""
        job = BatchJob(
            job_id=job_id,
            job_type="analysis",
            parameters={
                "analysis_type": analysis_type,
                "input_file": input_file,
                "parameters": parameters or {},
            },
            input_file=input_file,
            output_file=output_file,
        )
        self.jobs.append(job)
        return job_id

    def run_job(self, job: BatchJob) -> BatchJob:
        """Execute a single job."""
        job.status = "running"
        job.start_time = time.time()

        try:
            if job.job_type == "simulation":
                job.result = self._run_simulation(job)
            elif job.job_type == "validation":
                job.result = self._run_validation(job)
            elif job.job_type == "analysis":
                job.result = self._run_analysis(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")

            job.status = "completed"

            # Save result if output file specified
            if job.output_file:
                self._save_result(job)

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            apgi_logger.logger.error(f"Job {job.job_id} failed: {e}")

        job.end_time = time.time()
        return job

    def _run_simulation(self, job: BatchJob) -> Dict[str, Any]:
        """Run a simulation job."""
        params = job.parameters["params"]
        steps = job.parameters["steps"]
        dt = job.parameters["dt"]

        # Initialize system (no params in constructor)
        system = SurpriseIgnitionSystem()

        # Apply parameters to system if needed
        if hasattr(system, "__dict__"):
            for key, value in params.items():
                if hasattr(system, key):
                    setattr(system, key, value)

        # Create a simple input generator
        def input_generator(t):
            return {
                "Pi_e": np.random.normal(0, 1),
                "Pi_i": np.random.normal(0, 1),
                "eps_e": np.random.normal(0, 0.1),
                "eps_i": np.random.normal(0, 0.1),
                "beta": params.get("beta", 1.2),
            }

        # Run simulation
        duration = steps * dt
        results = system.simulate(
            duration=duration, dt=dt, input_generator=input_generator
        )

        return {
            "job_id": job.job_id,
            "parameters": params,
            "results": results,
            "summary": {
                "total_ignitions": np.sum(results["B"]),
                "mean_surprise": np.mean(results["S"])
                if len(results["S"]) > 0
                else 0.0,
                "mean_threshold": np.mean(results["theta"])
                if len(results["theta"]) > 0
                else 0.0,
                "final_state": {
                    "S": results["S"][-1] if len(results["S"]) > 0 else 0.0,
                    "theta": results["theta"][-1] if len(results["theta"]) > 0 else 0.0,
                    "B": results["B"][-1] if len(results["B"]) > 0 else 0.0,
                },
            },
        }

    def _run_validation(self, job: BatchJob) -> Dict[str, Any]:
        """Run a validation job."""
        protocol = job.parameters["protocol"]

        try:
            # Load validation module dynamically
            validation_module = load_validation_module(protocol)

            # Run validation
            results = validation_module.run_validation()

            return {
                "job_id": job.job_id,
                "protocol": protocol,
                "results": results,
                "status": "completed",
            }

        except Exception as e:
            raise ImportError(f"Could not run validation protocol {protocol}: {e}")

    def _run_analysis(self, job: BatchJob) -> Dict[str, Any]:
        """Run an analysis job."""
        analysis_type = job.parameters["analysis_type"]
        input_file = job.parameters["input_file"]

        # Load data
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        data = pd.read_csv(input_file)

        # Perform analysis based on type
        if analysis_type == "statistical_summary":
            result = {
                "job_id": job.job_id,
                "analysis_type": analysis_type,
                "summary": data.describe().to_dict(),
                "shape": data.shape,
                "columns": list(data.columns),
            }
        elif analysis_type == "correlation_analysis":
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            correlation_matrix = data[numeric_cols].corr()

            result = {
                "job_id": job.job_id,
                "analysis_type": analysis_type,
                "correlation_matrix": correlation_matrix.to_dict(),
                "strong_correlations": [],
            }

            # Find strong correlations
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        result["strong_correlations"].append(
                            {
                                "var1": correlation_matrix.columns[i],
                                "var2": correlation_matrix.columns[j],
                                "correlation": corr_val,
                            }
                        )

        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        return result

    def _save_result(self, job: BatchJob):
        """Save job result to file."""
        output_path = Path(job.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if job.output_file.endswith(".json"):
            with open(output_path, "w") as f:
                json.dump(job.result, f, indent=2, default=str)
        elif job.output_file.endswith(".pkl"):
            secure_pickle_dump(job.result, output_path)
        elif job.output_file.endswith(".csv"):
            if isinstance(job.result, dict) and "results" in job.result:
                # Save simulation results as CSV
                df = pd.DataFrame(job.result["results"])
                df.to_csv(output_path, index=False)
            else:
                # Save summary as CSV
                df = pd.DataFrame([job.result])
                df.to_csv(output_path, index=False)
        else:
            # Default to JSON
            with open(output_path.with_suffix(".json"), "w") as f:
                json.dump(job.result, f, indent=2, default=str)

    def run_batch(self, show_progress: bool = True) -> Dict[str, Any]:
        """Run all jobs in the batch."""
        if not self.jobs:
            print("No jobs to run")
            return {"status": "no_jobs"}

        print(f"Running {len(self.jobs)} jobs with {self.max_workers} workers...")

        # Create progress bar (or simple counter if tqdm not available)
        if show_progress and TQDM_AVAILABLE:
            pbar = tqdm(total=len(self.jobs), desc="Processing jobs")
        else:
            pbar = None
            if show_progress:
                print("Progress tracking disabled (tqdm not available)")

        # Separate and run different job types
        memory_intensive_jobs, regular_jobs = self._categorize_jobs()

        completed_jobs = []
        failed_jobs = []

        # Run jobs and collect results
        self._run_regular_jobs(regular_jobs, completed_jobs, failed_jobs, pbar)
        self._run_memory_intensive_jobs(
            memory_intensive_jobs, completed_jobs, failed_jobs, pbar
        )
        self._retry_failed_jobs(failed_jobs, completed_jobs, pbar)

        # Cleanup progress bar
        if pbar and TQDM_AVAILABLE:
            pbar.close()

        # Compile and return results
        return self._compile_results(completed_jobs)

    def _categorize_jobs(self):
        """Separate memory-intensive jobs from regular jobs."""
        memory_intensive_jobs = []
        regular_jobs = []

        for job in self.jobs:
            if (
                job.job_type == "validation"
                and job.parameters.get("protocol") == "protocol_2"
            ):
                memory_intensive_jobs.append(job)
            else:
                regular_jobs.append(job)

        return memory_intensive_jobs, regular_jobs

    def _run_regular_jobs(self, regular_jobs, completed_jobs, failed_jobs, pbar):
        """Run regular jobs in parallel."""
        if not regular_jobs:
            return

        print(f"Running {len(regular_jobs)} regular jobs in parallel...")
        try:
            with self.executor_class(max_workers=self.max_workers) as executor:
                # Submit regular jobs
                future_to_job = {
                    executor.submit(self.run_job, job): job for job in regular_jobs
                }

                # Collect results
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        completed_job = future.result()
                        completed_jobs.append(completed_job)
                    except Exception as e:
                        job.status = "failed"
                        job.error = str(e)
                        failed_jobs.append(job)
                        print(f"\nJob {job.job_id} failed: {e}")

                    if pbar:
                        pbar.update(1)

        except Exception as e:
            print(f"\nExecutor error: {e}")
            # Add all regular jobs to failed list for sequential processing
            failed_jobs.extend(regular_jobs)

    def _run_memory_intensive_jobs(
        self, memory_intensive_jobs, completed_jobs, failed_jobs, pbar
    ):
        """Run memory-intensive jobs sequentially."""
        if not memory_intensive_jobs:
            return

        print(
            f"Running {len(memory_intensive_jobs)} memory-intensive jobs sequentially..."
        )
        for job in memory_intensive_jobs:
            try:
                print(f"Running {job.job_id} (memory-intensive)...")
                completed_job = self.run_job(job)
                completed_jobs.append(completed_job)
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                failed_jobs.append(job)
                print(f"Memory-intensive job {job.job_id} failed: {e}")

            if pbar:
                pbar.update(1)

    def _retry_failed_jobs(self, failed_jobs, completed_jobs, pbar):
        """Try to run failed jobs sequentially."""
        if not failed_jobs:
            return

        print(f"Attempting to run {len(failed_jobs)} failed jobs sequentially...")
        for job in failed_jobs[:]:  # Copy list to allow modification during iteration
            try:
                print(f"Running {job.job_id} sequentially...")
                completed_job = self.run_job(job)
                completed_jobs.append(completed_job)
                failed_jobs.remove(job)
            except Exception as e2:
                job.status = "failed"
                job.error = str(e2)
                print(f"Sequential run also failed for {job.job_id}: {e2}")

            if pbar:
                pbar.update(1)

    def _compile_results(self, completed_jobs):
        """Compile and return batch processing results."""
        self.results = {
            "total_jobs": len(self.jobs),
            "completed": len([j for j in completed_jobs if j.status == "completed"]),
            "failed": len([j for j in completed_jobs if j.status == "failed"]),
            "jobs": [asdict(job) for job in completed_jobs],
        }

        # Print summary
        print("\nBatch processing complete:")
        print(f"  Total jobs: {self.results['total_jobs']}")
        print(f"  Completed: {self.results['completed']}")
        print(f"  Failed: {self.results['failed']}")

        if self.results["failed"] > 0:
            print("\nFailed jobs:")
            for job in completed_jobs:
                if job.status == "failed":
                    print(f"  {job.job_id}: {job.error}")

        return self.results

    def save_batch_report(self, output_file: str):
        """Save a comprehensive batch processing report."""
        report = {
            "timestamp": time.time(),
            "configuration": {
                "max_workers": self.max_workers,
                "use_processes": self.use_processes,
            },
            "results": self.results,
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Batch report saved to: {output_file}")


def create_parameter_grid(param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Create a grid of parameter combinations for batch simulation."""
    import itertools

    keys = list(param_ranges.keys())
    values = list(param_ranges.values())

    combinations = list(itertools.product(*values))

    return [dict(zip(keys, combo)) for combo in combinations]


def main():
    """Example usage of batch processing system."""
    # Create batch processor
    processor = BatchProcessor(max_workers=4, use_processes=True)

    # Example 1: Parameter sweep simulation
    param_ranges = {
        "tau_S": [0.3, 0.5, 0.7],
        "alpha": [8.0, 10.0, 12.0],
        "theta_0": [0.3, 0.5, 0.7],
    }

    param_combinations = create_parameter_grid(param_ranges)

    print(f"Adding {len(param_combinations)} simulation jobs...")
    for i, params in enumerate(param_combinations):
        processor.add_simulation_job(
            job_id=f"sim_{i:03d}",
            params=params,
            steps=1000,
            dt=0.01,
            output_file=f"results/sim_{i:03d}.json",
        )

    # Example 2: Validation protocols
    protocols = ["protocol_1", "protocol_2", "protocol_3"]
    for protocol in protocols:
        processor.add_validation_job(
            job_id=f"val_{protocol}",
            protocol=protocol,
            output_file=f"results/val_{protocol}.json",
        )

    # Example 3: Analysis jobs
    if Path("data/sample_data.csv").exists():
        processor.add_analysis_job(
            job_id="analysis_summary",
            analysis_type="statistical_summary",
            input_file="data/sample_data.csv",
            output_file="results/analysis_summary.json",
        )

    # Run batch
    processor.run_batch(show_progress=True)

    # Save report
    processor.save_batch_report("results/batch_report.json")


if __name__ == "__main__":
    main()
