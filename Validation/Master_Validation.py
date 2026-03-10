#!/usr/bin/env python3
"""
APGI Master Validation Orchestrator
===================================

Coordinates execution of all validation protocols and aggregates results.
"""

import importlib.util
from pathlib import Path
from typing import Dict, List

# Try to import logging config
try:
    from utils.logging_config import apgi_logger as logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class APGIMasterValidator:
    """Orchestrates execution of all APGI validation protocols"""

    def __init__(self):
        self.protocol_results = {}
        self.PROTOCOL_TIERS = {
            1: "primary",
            2: "primary",
            3: "secondary",
            4: "secondary",
            5: "tertiary",
            6: "tertiary",
            7: "tertiary",
            8: "secondary",
            9: "tertiary",
            10: "tertiary",
            11: "secondary",
            12: "secondary",
        }
        self.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }
        self.timeout_seconds = 30
        self.available_protocols = {
            "Protocol-1": {
                "file": "Validation-Protocol-1.py",
                "function": "run_validation",
                "description": "Synthetic Neural Data Generation and ML Classification",
            },
            "Protocol-2": {
                "file": "Validation-Protocol-2.py",
                "function": "run_validation",
                "description": "Behavioral Validation Protocol",
            },
            "Protocol-3": {
                "file": "Validation-Protocol-3.py",
                "function": "run_validation",
                "description": "Agent Comparison Experiment",
            },
            "Protocol-4": {
                "file": "Validation-Protocol-4.py",
                "function": "run_validation",
                "description": "Phase Transition Analysis",
            },
            "Protocol-5": {
                "file": "Validation-Protocol-5.py",
                "function": "run_validation",
                "description": "Evolutionary Emergence",
            },
            "Protocol-6": {
                "file": "Validation-Protocol-6.py",
                "function": "run_validation",
                "description": "Network Comparison",
            },
            "Protocol-7": {
                "file": "Validation-Protocol-7.py",
                "function": "run_validation",
                "description": "Mathematical Consistency",
            },
            "Protocol-8": {
                "file": "Validation-Protocol-8.py",
                "function": "run_validation",
                "description": "Parameter Sensitivity",
            },
            "Protocol-9": {
                "file": "Validation-Protocol-9.py",
                "function": "run_validation",
                "description": "Neural Signatures Validation",
            },
            "Protocol-10": {
                "file": "Validation-Protocol-10.py",
                "function": "run_validation",
                "description": "Cross-Species Scaling",
            },
            "Protocol-11": {
                "file": "Validation-Protocol-11.py",
                "function": "run_validation",
                "description": "Bayesian Estimation",
            },
            "Protocol-12": {
                "file": "Validation-Protocol-12.py",
                "function": "run_validation",
                "description": "Computational Benchmarking",
            },
        }

    def run_validation(self, protocols: List[str], **kwargs) -> Dict[str, Dict]:
        """
        Run specified validation protocols

        Args:
            protocols: List of protocol names (e.g., ["Protocol-1", "Protocol-2"])
            **kwargs: Additional arguments passed to protocol functions

        Returns:
            Dictionary of protocol results
        """
        results = {}

        for protocol_name in protocols:
            if protocol_name not in self.available_protocols:
                logger.warning(f"Unknown protocol: {protocol_name}")
                results[protocol_name] = {
                    "status": "error",
                    "message": "Unknown protocol",
                }
                continue

            try:
                logger.info(f"Running {protocol_name}...")
                protocol_info = self.available_protocols[protocol_name]
                result = self._run_single_protocol(protocol_info, **kwargs)
                results[protocol_name] = result
                logger.info(
                    f"{protocol_name} completed: {result.get('status', 'unknown')}"
                )

            except Exception as e:
                logger.error(f"Error running {protocol_name}: {e}")
                results[protocol_name] = {
                    "status": "error",
                    "message": str(e),
                    "passed": False,
                }

        self.protocol_results.update(results)
        return results

    def _run_single_protocol(self, protocol_info: Dict, **kwargs) -> Dict:
        """Run a single validation protocol"""
        file_path = Path(__file__).parent / protocol_info["file"]

        if not file_path.exists():
            return {
                "status": "error",
                "message": f"Protocol file not found: {file_path}",
                "passed": False,
            }

        # Load the protocol module
        spec = importlib.util.spec_from_file_location(protocol_info["file"], file_path)
        if spec is None or spec.loader is None:
            return {
                "status": "error",
                "message": "Could not load protocol module",
                "passed": False,
            }

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the validation function
        func_name = protocol_info["function"]
        if not hasattr(module, func_name):
            return {
                "status": "error",
                "message": f"Validation function '{func_name}' not found in {protocol_info['file']}",
                "passed": False,
            }

        validation_func = getattr(module, func_name)

        # Run the validation
        try:
            result = validation_func(**kwargs)
            if isinstance(result, dict):
                return result
            else:
                # Assume boolean result
                return {
                    "status": "success" if result else "failed",
                    "passed": bool(result),
                }
        except Exception as e:
            return {"status": "error", "message": str(e), "passed": False}

    def generate_master_report(self) -> Dict:
        """Generate comprehensive validation report"""
        if not self.protocol_results:
            return {
                "overall_decision": "No protocols run",
                "summary": "Run validation protocols first",
                "protocol_results": {},
            }

        total_protocols = len(self.protocol_results)
        passed_protocols = sum(
            1 for r in self.protocol_results.values() if r.get("passed", False)
        )
        success_rate = passed_protocols / total_protocols if total_protocols > 0 else 0

        # Determine overall decision
        if success_rate >= 0.8:
            overall_decision = "PASS: Strong validation support"
        elif success_rate >= 0.6:
            overall_decision = "MARGINAL: Moderate validation support"
        else:
            overall_decision = "FAIL: Insufficient validation support"

        return {
            "overall_decision": overall_decision,
            "total_protocols": total_protocols,
            "passed_protocols": passed_protocols,
            "success_rate": success_rate,
            "protocol_results": self.protocol_results,
            "summary": f"Validated {passed_protocols}/{total_protocols} protocols ({success_rate:.1%})",
        }

    def get_available_protocols(self) -> Dict[str, Dict]:
        """Get list of available validation protocols"""
        return self.available_protocols.copy()

    def clear_results(self):
        """Clear all protocol results"""
        self.protocol_results.clear()
