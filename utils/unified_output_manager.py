#!/usr/bin/env python3
"""
TIER DESIGNATION: Tier 2 (information-theoretic)

Bridge to Level 1
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class UnifiedOutputManager:
    """Manages standardized output for all APGI scripts."""

    # Output directory structure
    OUTPUT_ROOT = "outputs"
    SUBDIRS = {
        "validation": "validation",
        "falsification": "falsification",
        "theory": "theory",
    }

    # Standard output files
    RESULTS_FILE = "results.json"
    METADATA_FILE = "metadata.json"
    LOGS_DIR = "logs"
    SUMMARY_FILE = "summary.txt"

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the output manager.

        Args:
            project_root: Root directory of the APGI project
        """
        self.project_root = project_root or Path(__file__).parent.parent
        self.output_root = self.project_root / self.OUTPUT_ROOT

        # Create output directories
        self._ensure_directories()

        # Setup logging
        self.logger = self._setup_logging()

    def _ensure_directories(self) -> None:
        """Ensure all output directories exist."""
        for subdir in self.SUBDIRS.values():
            (self.output_root / subdir).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the output manager."""
        logger = logging.getLogger("UnifiedOutputManager")
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(console_handler)

        return logger

    def get_protocol_output_dir(self, protocol_id: str, create: bool = True) -> Path:
        """Get the output directory for a protocol.

        Args:
            protocol_id: Protocol identifier (e.g., "VP_01", "FP_01")
            create: Whether to create the directory if it doesn't exist

        Returns:
            Path to the protocol output directory
        """
        if protocol_id.startswith("VP_"):
            subdir = self.SUBDIRS["validation"]
        elif protocol_id.startswith("FP_"):
            subdir = self.SUBDIRS["falsification"]
        elif protocol_id.startswith("APGI_"):
            # Handle theory modules that start with APGI_
            subdir = self.SUBDIRS["theory"]
        else:
            raise ValueError(f"Unknown protocol type: {protocol_id}")

        output_dir = self.output_root / subdir / protocol_id

        if create:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / self.LOGS_DIR).mkdir(exist_ok=True)

        return output_dir

    def get_theory_output_dir(self, module_name: str, create: bool = True) -> Path:
        """Get the output directory for a theory module.

        Args:
            module_name: Theory module name (e.g., "APGI_Thermodynamic_Program_Aggregator")
            create: Whether to create the directory if it doesn't exist

        Returns:
            Path to the theory module output directory
        """
        output_dir = self.output_root / self.SUBDIRS["theory"] / module_name

        if create:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / self.LOGS_DIR).mkdir(exist_ok=True)

        return output_dir

    def save_results(
        self,
        protocol_id: str,
        results: Dict[str, Any],
        is_theory: bool = False,
        module_name: Optional[str] = None,
    ) -> Path:
        """Save results to standardized output file.

        Args:
            protocol_id: Protocol identifier (e.g., "VP_01", "FP_01")
            results: Results dictionary to save
            is_theory: Whether this is a theory module output
            module_name: Theory module name (required if is_theory=True)

        Returns:
            Path to the saved results file
        """
        if is_theory:
            if not module_name:
                raise ValueError("module_name required for theory outputs")
            output_dir = self.get_theory_output_dir(module_name)
        else:
            output_dir = self.get_protocol_output_dir(protocol_id)

        results_file = output_dir / self.RESULTS_FILE

        # Add timestamp if not present
        if "timestamp" not in results:
            results["timestamp"] = datetime.now().isoformat()

        # Save results
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Saved results to {results_file}")
        return results_file

    def save_metadata(
        self,
        protocol_id: str,
        metadata: Dict[str, Any],
        is_theory: bool = False,
        module_name: Optional[str] = None,
    ) -> Path:
        """Save metadata to standardized metadata file.

        Args:
            protocol_id: Protocol identifier
            metadata: Metadata dictionary to save
            is_theory: Whether this is a theory module output
            module_name: Theory module name (required if is_theory=True)

        Returns:
            Path to the saved metadata file
        """
        if is_theory:
            if not module_name:
                raise ValueError("module_name required for theory outputs")
            output_dir = self.get_theory_output_dir(module_name)
        else:
            output_dir = self.get_protocol_output_dir(protocol_id)

        metadata_file = output_dir / self.METADATA_FILE

        # Add standard metadata
        metadata.setdefault("protocol_id", protocol_id)
        metadata.setdefault("timestamp", datetime.now().isoformat())
        metadata.setdefault("output_dir", str(output_dir))

        # Save metadata
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.info(f"Saved metadata to {metadata_file}")
        return metadata_file

    def get_logger_for_protocol(
        self,
        protocol_id: str,
        is_theory: bool = False,
        module_name: Optional[str] = None,
    ) -> logging.Logger:
        """Get a logger that writes to the protocol's log directory.

        Args:
            protocol_id: Protocol identifier
            is_theory: Whether this is a theory module output
            module_name: Theory module name (required if is_theory=True)

        Returns:
            Logger instance configured for the protocol
        """
        if is_theory:
            if not module_name:
                raise ValueError("module_name required for theory outputs")
            output_dir = self.get_theory_output_dir(module_name)
            logger_name = f"theory.{module_name}"
        else:
            output_dir = self.get_protocol_output_dir(protocol_id)
            logger_name = f"protocol.{protocol_id}"

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        logger.handlers = []

        # File handler
        log_file = output_dir / self.LOGS_DIR / f"{protocol_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def save_summary(
        self,
        protocol_id: str,
        summary_text: str,
        is_theory: bool = False,
        module_name: Optional[str] = None,
    ) -> Path:
        """Save a text summary of the protocol execution.

        Args:
            protocol_id: Protocol identifier
            summary_text: Summary text to save
            is_theory: Whether this is a theory module output
            module_name: Theory module name (required if is_theory=True)

        Returns:
            Path to the saved summary file
        """
        if is_theory:
            if not module_name:
                raise ValueError("module_name required for theory outputs")
            output_dir = self.get_theory_output_dir(module_name)
        else:
            output_dir = self.get_protocol_output_dir(protocol_id)

        summary_file = output_dir / self.SUMMARY_FILE

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Protocol: {protocol_id}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(summary_text)

        self.logger.info(f"Saved summary to {summary_file}")
        return summary_file

    def load_results(
        self,
        protocol_id: str,
        is_theory: bool = False,
        module_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load results from a protocol's output directory.

        Args:
            protocol_id: Protocol identifier
            is_theory: Whether this is a theory module output
            module_name: Theory module name (required if is_theory=True)

        Returns:
            Results dictionary, or None if not found
        """
        if is_theory:
            if not module_name:
                raise ValueError("module_name required for theory outputs")
            output_dir = self.get_theory_output_dir(module_name, create=False)
        else:
            output_dir = self.get_protocol_output_dir(protocol_id, create=False)

        results_file = output_dir / self.RESULTS_FILE

        if not results_file.exists():
            self.logger.warning(f"Results file not found: {results_file}")
            return None

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Error loading results: {e}")
            return None

    def get_all_protocol_outputs(self) -> Dict[str, Dict[str, Any]]:
        """Get all protocol outputs organized by type.

        Returns:
            Dictionary with validation, falsification, and theory outputs
        """
        outputs: Dict[str, Dict[str, Any]] = {
            "validation": {},
            "falsification": {},
            "theory": {},
        }

        # Validation protocols
        validation_dir = self.output_root / self.SUBDIRS["validation"]
        if validation_dir.exists():
            for protocol_dir in validation_dir.iterdir():
                if protocol_dir.is_dir():
                    results = self.load_results(protocol_dir.name)
                    if results:
                        outputs["validation"][protocol_dir.name] = results

        # Falsification protocols
        falsification_dir = self.output_root / self.SUBDIRS["falsification"]
        if falsification_dir.exists():
            for protocol_dir in falsification_dir.iterdir():
                if protocol_dir.is_dir():
                    results = self.load_results(protocol_dir.name)
                    if results:
                        outputs["falsification"][protocol_dir.name] = results

        # Theory modules
        theory_dir = self.output_root / self.SUBDIRS["theory"]
        if theory_dir.exists():
            for module_dir in theory_dir.iterdir():
                if module_dir.is_dir():
                    results = self.load_results(module_dir.name)
                    if results:
                        outputs["theory"][module_dir.name] = results

        return outputs

    def generate_unified_report(self) -> str:
        """Generate a unified report of all protocol outputs.

        Returns:
            Formatted report string
        """
        outputs = self.get_all_protocol_outputs()

        report = []
        report.append("=" * 80)
        report.append("APGI UNIFIED OUTPUT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}\n")

        # Validation protocols
        report.append("VALIDATION PROTOCOLS")
        report.append("-" * 80)
        if outputs["validation"]:
            for protocol_id, results in sorted(outputs["validation"].items()):
                status = results.get("status", "unknown")
                report.append(f"  {protocol_id}: {status}")
        else:
            report.append("  No validation protocol outputs found")

        # Falsification protocols
        report.append("\nFALSIFICATION PROTOCOLS")
        report.append("-" * 80)
        if outputs["falsification"]:
            for protocol_id, results in sorted(outputs["falsification"].items()):
                status = results.get("status", "unknown")
                report.append(f"  {protocol_id}: {status}")
        else:
            report.append("  No falsification protocol outputs found")

        # Theory modules
        report.append("\nTHEORY MODULES")
        report.append("-" * 80)
        if outputs["theory"]:
            for module_name, results in sorted(outputs["theory"].items()):
                status = results.get("status", "unknown")
                report.append(f"  {module_name}: {status}")
        else:
            report.append("  No theory module outputs found")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def cleanup_old_outputs(self, days: int = 30) -> int:
        """Remove output files older than specified days.

        Args:
            days: Number of days to keep

        Returns:
            Number of files removed
        """
        import time

        cutoff_time = time.time() - (days * 86400)
        removed_count = 0

        for output_dir in self.output_root.rglob("*"):
            if output_dir.is_file():
                if output_dir.stat().st_mtime < cutoff_time:
                    output_dir.unlink()
                    removed_count += 1
                    self.logger.info(f"Removed old file: {output_dir}")

        return removed_count


def main():
    """Test the unified output manager."""
    manager = UnifiedOutputManager()

    print("Unified Output Manager Test")
    print("=" * 80)

    # Test validation protocol output
    print("\n1. Testing Validation Protocol Output (VP_01)...")
    vp_dir = manager.get_protocol_output_dir("VP_01")
    print(f"   Output directory: {vp_dir}")

    # Save sample results
    sample_results = {
        "protocol_id": "VP_01",
        "status": "pass",
        "tests_passed": 5,
        "tests_failed": 0,
        "execution_time": 12.34,
    }
    manager.save_results("VP_01", sample_results)

    # Save sample metadata
    sample_metadata = {
        "version": "1.0.0",
        "author": "APGI Framework",
        "description": "Sample validation protocol",
    }
    manager.save_metadata("VP_01", sample_metadata)

    # Test falsification protocol output
    print("\n2. Testing Falsification Protocol Output (FP_01)...")
    fp_dir = manager.get_protocol_output_dir("FP_01")
    print(f"   Output directory: {fp_dir}")

    # Test theory module output
    print("\n3. Testing Theory Module Output...")
    theory_dir = manager.get_theory_output_dir("APGI_Thermodynamic_Program_Aggregator")
    print(f"   Output directory: {theory_dir}")

    # Save sample theory results
    theory_results = {
        "module": "APGI_Thermodynamic_Program_Aggregator",
        "status": "pass",
        "kappa": 4.625e14,
        "scaling_exponent": 0.414,
    }
    manager.save_results(
        "APGI_Thermodynamic_Program_Aggregator",
        theory_results,
        module_name="APGI_Thermodynamic_Program_Aggregator",
    )

    # Generate unified report
    print("\n4. Generating Unified Report...")
    report = manager.generate_unified_report()
    print(report)

    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    main()
