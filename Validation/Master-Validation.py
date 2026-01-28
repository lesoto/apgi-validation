"""
APGI Master Validation Pipeline
================================

Executes all 8 protocols and applies hierarchical falsification decision tree
per validation roadmap page 26.

Falsification Logic:
- Primary tests (Protocols 1, 3): Failure → Framework rejected
- Secondary tests (2+ failures): Major revision required
- Tertiary tests (3+ failures): Scope restriction
"""

import importlib
import importlib.util
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class APGIMasterValidator:
    # Protocol tier classification - class constant for consistency
    PROTOCOL_TIERS = {
        1: "primary",  # Primary tests: Failure → Framework rejected
        2: "secondary",  # Secondary tests: 2+ failures → Major revision
        3: "primary",  # Primary tests: Failure → Framework rejected
        4: "secondary",  # Secondary tests: 2+ failures → Major revision
        5: "tertiary",  # Tertiary tests: 3+ failures → Scope restriction
        6: "tertiary",  # Tertiary tests: 3+ failures → Scope restriction
        7: "tertiary",  # Tertiary tests: 3+ failures → Scope restriction
        8: "secondary",  # Secondary tests: 2+ failures → Major revision
    }

    def __init__(self, timeout_seconds: int = 300) -> None:
        """Initialize validator with configurable timeout."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self.protocol_results: Dict[str, Any] = {}
        self.falsification_status: Dict[str, List[Dict[str, Any]]] = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }

    def run_all_protocols(self) -> None:
        """Execute all 8 protocols in sequence."""
        for protocol_num in self.PROTOCOL_TIERS.keys():
            self._run_protocol(protocol_num)

    def _validate_protocol_number(self, protocol_num: int) -> bool:
        """Validate protocol number is within valid range."""
        return isinstance(protocol_num, int) and protocol_num in self.PROTOCOL_TIERS

    def _run_protocol(self, protocol_num: int) -> None:
        """Execute a single protocol with comprehensive error handling and timeout."""
        if not self._validate_protocol_number(protocol_num):
            self._handle_protocol_error(
                protocol_num,
                "INVALID_PROTOCOL",
                f"Protocol number must be between 1 and 8, got {protocol_num}",
            )
            return

        try:
            protocol_file = f"Validation-Protocol-{protocol_num}.py"
            protocol_path = Path(__file__).parent / protocol_file

            if not protocol_path.exists():
                raise ImportError(f"Protocol file {protocol_file} not found")

            spec = importlib.util.spec_from_file_location(
                f"Validation_Protocol_{protocol_num}", protocol_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create spec for {protocol_file}")

            protocol_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(protocol_module)

            # Execute protocol with timeout
            result = self._execute_protocol_with_timeout(protocol_module)
            passed = result.get("passed", False) if isinstance(result, dict) else False
            tier = self.PROTOCOL_TIERS[protocol_num]

            self.protocol_results[f"protocol_{protocol_num}"] = result
            self.falsification_status[tier].append(
                {"protocol": protocol_num, "passed": passed, "result": result}
            )

            logger.info(f"Protocol {protocol_num}: {'PASSED' if passed else 'FAILED'}")

        except KeyError as e:
            self._handle_protocol_error(
                protocol_num, "PROTOCOL_KEY_ERROR", f"Invalid protocol tier lookup: {e}"
            )
        except ImportError as e:
            self._handle_protocol_error(protocol_num, "IMPORT_ERROR", str(e))
        except AttributeError as e:
            self._handle_protocol_error(protocol_num, "INTERFACE_ERROR", str(e))
        except (ValueError, TypeError) as e:
            self._handle_protocol_error(protocol_num, "PARAMETER_ERROR", str(e))
        except RuntimeError as e:
            self._handle_protocol_error(protocol_num, "RUNTIME_ERROR", str(e))
        except Exception as e:
            import traceback

            error_result = {
                "status": "UNEXPECTED_ERROR",
                "error": f"Unexpected error: {e}",
                "traceback": traceback.format_exc(),
                "passed": False,
            }
            # Validate protocol number before accessing PROTOCOL_TIERS
            if self._validate_protocol_number(protocol_num):
                tier = self.PROTOCOL_TIERS[protocol_num]
                self.protocol_results[f"protocol_{protocol_num}"] = error_result
                self.falsification_status[tier].append(
                    {"protocol": protocol_num, "passed": False, "result": error_result}
                )
                logger.error(f"Protocol {protocol_num}: Unexpected error - {e}")
            else:
                self.protocol_results[f"protocol_{protocol_num}"] = error_result
                logger.error(f"Protocol {protocol_num}: Unexpected error - {e}")

    def _execute_protocol_validation(self, protocol_module: Any) -> Dict[str, Any]:
        """Execute validation function from protocol module with strict interface detection."""
        # Primary validation functions - strict interface
        validation_functions = [
            "run_validation",
            "validate",
        ]

        for func_name in validation_functions:
            if hasattr(protocol_module, func_name):
                func = getattr(protocol_module, func_name)
                if callable(func):
                    try:
                        result = func()
                        if self._validate_protocol_result(result):
                            return result
                    except Exception as e:
                        return {
                            "status": "EXECUTION_ERROR",
                            "error": f"Error executing {func_name}: {e}",
                            "passed": False,
                        }

        # Secondary functions with explicit validation in name
        secondary_functions = ["main", "execute", "run"]

        for func_name in secondary_functions:
            if hasattr(protocol_module, func_name):
                func = getattr(protocol_module, func_name)
                if callable(func):
                    try:
                        result = func()
                        if self._validate_protocol_result(result):
                            return result
                    except Exception as e:
                        return {
                            "status": "EXECUTION_ERROR",
                            "error": f"Error executing {func_name}: {e}",
                            "passed": False,
                        }

        return {
            "status": "NO_VALIDATION_FUNCTION",
            "error": "No suitable validation function found. Protocol must implement run_validation, validate, main, or execute",
            "passed": False,
        }

    def _handle_protocol_error(
        self, protocol_num: int, status: str, error: str
    ) -> None:
        """Handle protocol errors consistently."""
        error_result = {"status": status, "error": error, "passed": False}

        if self._validate_protocol_number(protocol_num):
            tier = self.PROTOCOL_TIERS[protocol_num]
            self.protocol_results[f"protocol_{protocol_num}"] = error_result
            self.falsification_status[tier].append(
                {"protocol": protocol_num, "passed": False, "result": error_result}
            )
        else:
            self.protocol_results[f"protocol_{protocol_num}"] = error_result

        logger.error(f"Protocol {protocol_num}: {status} - {error}")

    def apply_decision_tree(self) -> str:
        """
        Apply hierarchical falsification logic

        Returns:
            'VALIDATED', 'MAJOR_REVISION', 'SCOPE_RESTRICTION', or 'REJECTED'
        """
        # Count failures at each tier
        primary_failures = len(
            [r for r in self.falsification_status["primary"] if not r["passed"]]
        )
        secondary_failures = len(
            [r for r in self.falsification_status["secondary"] if not r["passed"]]
        )
        tertiary_failures = len(
            [r for r in self.falsification_status["tertiary"] if not r["passed"]]
        )

        # Decision tree
        if primary_failures >= 1:
            return "REJECTED"
        elif secondary_failures >= 2:
            return "MAJOR_REVISION"
        elif tertiary_failures >= 3:
            return "SCOPE_RESTRICTION"
        else:
            return "VALIDATED"

    def generate_master_report(self) -> Dict[str, Any]:
        """Comprehensive validation report."""
        return {
            "protocol_results": self.protocol_results,
            "falsification_status": self.falsification_status,
            "overall_decision": self.apply_decision_tree(),
        }

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy and other non-serializable types to Python types."""
        import numpy as np

        if obj is None:
            return None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return self._convert_to_serializable(obj.__dict__)
        else:
            return obj

    def _execute_protocol_with_timeout(self, protocol_module: Any) -> Dict[str, Any]:
        """Execute protocol with timeout protection."""
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._execute_protocol_validation, protocol_module)
            try:
                result = future.result(timeout=self.timeout_seconds)
                return result
            except FutureTimeoutError:
                return {
                    "status": "TIMEOUT_ERROR",
                    "error": f"Protocol timed out after {self.timeout_seconds} seconds",
                    "passed": False,
                }
        except Exception as e:
            return {
                "status": "EXECUTION_ERROR",
                "error": f"Failed to execute protocol: {e}",
                "passed": False,
            }
        finally:
            executor.shutdown(wait=False)

    def _validate_protocol_result(self, result: Any) -> bool:
        """Validate that protocol result is properly structured with strict checking."""
        if result is None:
            return False

        if isinstance(result, dict):
            # Require explicit 'passed' key - no defaults
            if "passed" not in result:
                return False

            # Validate passed is boolean
            if not isinstance(result["passed"], bool):
                return False

            # Ensure status key exists
            if "status" not in result:
                return False

            return True

        return False


if __name__ == "__main__":
    # Run master validation with configurable timeout
    import argparse

    parser = argparse.ArgumentParser(description="APGI Master Validation Pipeline")
    parser.add_argument(
        "--timeout", type=int, default=300, help="Protocol timeout in seconds"
    )
    args = parser.parse_args()

    validator = APGIMasterValidator(timeout_seconds=args.timeout)

    logger.info("Starting APGI Master Validation Pipeline...")
    logger.info("=" * 50)

    # Execute all protocols
    validator.run_all_protocols()

    logger.info("=" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 50)

    # Generate and display report
    report = validator.generate_master_report()

    logger.info(f"Overall Decision: {report['overall_decision']}")

    # Print tier summaries
    for tier, results in report["falsification_status"].items():
        failures = len([r for r in results if not r["passed"]])
        total = len(results)
        logger.info(f"{tier.capitalize()} tier: {failures}/{total} failed")

    # Save detailed report
    report_serializable = validator._convert_to_serializable(report)
    with open("APGI-Master-Validation-Report.json", "w") as f:
        json.dump(report_serializable, f, indent=2)

    logger.info(f"Detailed report saved to: APGI-Master-Validation-Report.json")
