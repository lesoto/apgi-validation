#!/usr/bin/env python3
"""
APGI Master Validation Orchestrator
===================================

Coordinates execution of all validation protocols and aggregates results.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to sys.path for Falsification imports
_proj_root = Path(__file__).parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from Falsification.FP_12_Falsification_Aggregator import FalsificationAggregator

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
        # Protocol tier classification rationale:
        # - Primary (1-2): Core validation protocols that test fundamental APGI properties
        #   Protocol 1: Basic equation validation
        #   Protocol 2: Parameter consistency checks
        # - Secondary (3-4, 8, 11-12): Extended validation covering specific aspects
        #   Protocol 3: Behavioral pattern validation
        #   Protocol 4: State transition verification
        #   Protocol 8: Cross-species scaling validation
        #   Protocol 11: Cultural neuroscience validation
        #   Protocol 12: Liquid network validation
        # - Tertiary (5-7, 9-10): Specialized and experimental protocols
        #   Protocol 5: Computational benchmarking
        #   Protocol 6: Bayesian estimation framework
        #   Protocol 7: Multimodal integration
        #   Protocol 9: Psychological states validation
        #   Protocol 10: Turing machine validation
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
            "P4-Epistemic": "secondary",
            "FP-5": "tertiary",
            "FP-6": "tertiary",
            "FP-7": "tertiary",
        }
        self.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }
        self.timeout_seconds = 30
        # Protocol dependencies: protocols that must run before others
        self.protocol_dependencies = {
            "Protocol-1": [],  # No dependencies
            "Protocol-2": [],
            "Protocol-3": [],
            "Protocol-4": [],
            "Protocol-5": [],  # Evolutionary - independent
            "Protocol-6": [],
            "Protocol-7": [],
            "Protocol-8": [],
            "Protocol-9": [],
            "Protocol-10": [],
            "Protocol-11": [],
            "Protocol-12": [],
            "Protocol-P4-Epistemic": [],
            "Falsification-Protocol-P5": [],
            "Falsification-Protocol-P6": [],
            "Falsification-Protocol-P7": [],
        }
        self.available_protocols = {
            "Protocol-1": {
                "file": "VP_1_SyntheticEEG_MLClassification.py",
                "function": "run_validation",
                "description": "Synthetic Neural Data Generation and ML Classification",
            },
            "Protocol-2": {
                "file": "VP_2_Validation_Protocol_2.py",
                "function": "run_validation",
                "description": "Behavioral Validation Protocol",
            },
            "Protocol-3": {
                "file": "VP_3_ActiveInference_AgentSimulations_Protocol3.py",
                "function": "run_validation",
                "description": "Agent Comparison Experiment",
            },
            "Protocol-4": {
                "file": "VP_4_InformationTheoretic_PhaseTransition_Level2.py",
                "function": "run_validation",
                "description": "Phase Transition Analysis",
            },
            "Protocol-5": {
                "file": "VP_5_EvolutionaryEmergence_AnalyticalValidation.py",
                "function": "run_validation",
                "description": "Evolutionary Emergence",
            },
            "Protocol-6": {
                "file": "VP_6_NeuralNetwork_InductiveBias_ComputationalBenchmark.py",
                "function": "run_validation",
                "description": "Network Comparison",
            },
            "Protocol-7": {
                "file": "VP_7_TMS_Pharmacological_CausalIntervention_Protocol2.py",
                "function": "run_validation",
                "description": "Causal Manipulations",
            },
            "Protocol-8": {
                "file": "VP_8_Psychophysical_ThresholdEstimation_Protocol1.py",
                "function": "run_validation",
                "description": "Psychophysical Thresholds",
            },
            "Protocol-9": {
                "file": "VP_9_ConvergentNeuralSignatures_Priority1_EmpiricalRoadmap.py",
                "function": "run_validation",
                "description": "Neural Signatures Validation",
            },
            "Protocol-P4-Epistemic": {
                "file": "Validation_Protocol_P4_Epistemic.py",
                "function": "run_validation",
                "description": "Paper 4 Epistemic Architecture Predictions (P5-P12)",
            },
            "Protocol-10": {
                "file": "VP_10_Falsification_CausalManipulations_TMS_Pharmacological_Priority2.py",
                "function": "run_validation",
                "description": "Cross-Species Scaling",
            },
            "Protocol-11": {
                "file": "VP_11_Validation_Protocol_11.py",
                "function": "run_validation",
                "description": "Bayesian Estimation",
            },
            "Protocol-12": {
                "file": "VP_12_Clinical_CrossSpecies_Convergence_Protocol4.py",
                "function": "run_validation",
                "description": "Computational Benchmarking",
            },
            "Falsification-Protocol-P5": {
                "file": "../Falsification/Falsification_Protocol_P5.py",
                "function": "run_protocol_p5",
                "description": "Evolutionary Emergence (Falsification)",
            },
            "Falsification-Protocol-P6": {
                "file": "../Falsification/Falsification_Protocol_P6.py",
                "function": "run_protocol_p6",
                "description": "Temporal Dynamics / LTCN (Falsification)",
            },
            "Falsification-Protocol-P7": {
                "file": "../Falsification/Falsification_Protocol_P7.py",
                "function": "run_protocol_p7",
                "description": "Causal Manipulations (Falsification)",
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
        """Generate comprehensive validation report with weighted scoring."""
        total_protocols = len(self.protocol_results)
        if total_protocols == 0:
            return {
                "overall_decision": "No protocols run",
                "total_protocols": 0,
                "passed_protocols": 0,
                "success_rate": 0,
                "weighted_score": 0,
                "protocol_results": {},
                "falsification_status": self.falsification_status,
                "summary": "Run validation protocols first",
            }

        passed_protocols = sum(
            1 for r in self.protocol_results.values() if r.get("passed", False)
        )
        success_rate = passed_protocols / total_protocols

        # Equal weighting across all protocols (1/N) unless papers specify differential evidential weight
        # This prevents systematic underweighting of protocols covering Paper 4's Level 1/2 predictions
        tier_weights = {"primary": 1.0, "secondary": 1.0, "tertiary": 1.0}
        tier_stats = {
            "primary": {"passed": 0, "total": 0},
            "secondary": {"passed": 0, "total": 0},
            "tertiary": {"passed": 0, "total": 0},
        }

        for p_name, result in self.protocol_results.items():
            # Extract identifier from "Protocol-X" or "Falsification-Protocol-PX"
            try:
                if "P4-Epistemic" in p_name:
                    tier = self.PROTOCOL_TIERS.get("P4-Epistemic", "secondary")
                elif "-P5" in p_name:
                    tier = self.PROTOCOL_TIERS.get("FP-5", "tertiary")
                elif "-P6" in p_name:
                    tier = self.PROTOCOL_TIERS.get("FP-6", "tertiary")
                elif "-P7" in p_name:
                    tier = self.PROTOCOL_TIERS.get("FP-7", "tertiary")
                else:
                    p_num = int(p_name.split("-")[-1])
                    tier = self.PROTOCOL_TIERS.get(p_num, "tertiary")
            except (ValueError, IndexError):
                tier = "tertiary"

            tier_stats[tier]["total"] += 1
            if result.get("passed", False):
                tier_stats[tier]["passed"] += 1

        weighted_score = 0.0
        total_weight_used = 0.0
        for tier, stats in tier_stats.items():
            if stats["total"] > 0:
                tier_success = stats["passed"] / stats["total"]
                weight = tier_weights[tier]
                weighted_score += tier_success * weight
                total_weight_used += weight

        # Normalize if not all tiers were run
        if total_weight_used > 0:
            weighted_score /= total_weight_used

        # Determine overall decision
        if weighted_score >= 0.85:
            overall_decision = "PASS: Strong validation support"
        elif weighted_score >= 0.60:
            overall_decision = "MARGINAL: Moderate validation support"
        else:
            overall_decision = "FAIL: Insufficient validation support"

        summary = f"Validated {passed_protocols}/{total_protocols} protocols (Raw: {success_rate:.1%}, Weighted Score: {weighted_score:.2f})"

        return {
            "overall_decision": overall_decision,
            "total_protocols": total_protocols,
            "passed_protocols": passed_protocols,
            "success_rate": success_rate,
            "weighted_score": weighted_score,
            "tier_summary": tier_stats,
            "protocol_results": self.protocol_results,
            "falsification_status": self.falsification_status,
            "summary": summary,
        }

    def get_available_protocols(self) -> Dict[str, Dict]:
        """Get list of available validation protocols"""
        return self.available_protocols.copy()

    def clear_results(self):
        """Clear all protocol results"""
        self.protocol_results.clear()

    def run_all_protocols(
        self, seed: Optional[int] = None, **kwargs
    ) -> Dict[str, Dict]:
        """
        Run all validation protocols in dependency order

        Args:
            seed: Random seed for reproducibility
            **kwargs: Additional arguments passed to protocol functions

        Returns:
            Dictionary of all protocol results
        """
        if seed is not None:
            np.random.seed(seed)
            kwargs["seed"] = seed

        # Get all available protocols
        all_protocols = list(self.available_protocols.keys())

        # Run in dependency order (topological sort)
        executed = set()
        results = {}

        for protocol_name in all_protocols:
            if protocol_name in executed:
                continue

            # Check dependencies
            dependencies = self.protocol_dependencies.get(protocol_name, [])
            for dep in dependencies:
                if dep not in executed:
                    # Run dependency first
                    dep_results = self.run_validation([dep], **kwargs)
                    results.update(dep_results)
                    executed.add(dep)

            # Run current protocol
            protocol_results = self.run_validation([protocol_name], **kwargs)
            results.update(protocol_results)
            executed.add(protocol_name)

        return results

    def generate_reproducibility_package(self, output_dir: str = None) -> Dict:
        """
        Generate reproducibility package with all parameters, seeds, and outputs

        Args:
            output_dir: Directory to save reproducibility package

        Returns:
            Dictionary with reproducibility information
        """
        from pathlib import Path
        import json
        from datetime import datetime

        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "reproducibility"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate reproducibility data
        reproducibility_data = {
            "timestamp": datetime.now().isoformat(),
            "protocol_results": self.protocol_results,
            "tier_classification": self.PROTOCOL_TIERS,
            "tier_weights": {"primary": 1.0, "secondary": 1.0, "tertiary": 1.0},
            "available_protocols": self.available_protocols,
            "protocol_dependencies": self.protocol_dependencies,
        }

        # Save reproducibility data
        output_file = (
            output_dir
            / f"validation_reproducibility_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(reproducibility_data, f, indent=2)

        # Save results as CSV
        results_df = self._results_to_dataframe()
        csv_file = (
            output_dir
            / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        results_df.to_csv(csv_file, index=False)

        return {
            "reproducibility_data": reproducibility_data,
            "output_files": {
                "json": str(output_file),
                "csv": str(csv_file),
            },
            "output_directory": str(output_dir),
        }

    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert protocol results to pandas DataFrame"""
        rows = []
        for protocol_name, result in self.protocol_results.items():
            row = {
                "protocol": protocol_name,
                "status": result.get("status", "unknown"),
                "passed": result.get("passed", False),
                "message": result.get("message", ""),
            }

            # Extract identifier for tier classification
            try:
                if "P4-Epistemic" in protocol_name:
                    row["tier"] = self.PROTOCOL_TIERS.get("P4-Epistemic", "unknown")
                elif "-P5" in protocol_name:
                    row["tier"] = self.PROTOCOL_TIERS.get("FP-5", "unknown")
                elif "-P6" in protocol_name:
                    row["tier"] = self.PROTOCOL_TIERS.get("FP-6", "unknown")
                elif "-P7" in protocol_name:
                    row["tier"] = self.PROTOCOL_TIERS.get("FP-7", "unknown")
                else:
                    p_num = int(protocol_name.split("-")[-1])
                    row["tier"] = self.PROTOCOL_TIERS.get(p_num, "unknown")
            except (ValueError, IndexError):
                row["tier"] = "unknown"

            rows.append(row)

        return pd.DataFrame(rows)


def main():
    """Main entry point for Master Validation"""
    validator = APGIMasterValidator()
    print("\n" + "=" * 80)
    print(" APGI MASTER VALIDATION ORCHESTRATOR ".center(80, "="))
    print("=" * 80)

    # Run all available protocols
    summary = validator.run_all_protocols(save_outputs=True)

    # Print results table
    print("\nVALDIATION RESULTS:")
    print("-" * 80)
    # Print nice table header
    print(f"{'PROTOCOL':<15} | {'TIER':<10} | {'STATUS':<10} | {'MESSAGE'}")
    print("-" * 80)

    for p_name in sorted(validator.protocol_results.keys()):
        res = validator.protocol_results[p_name]
        try:
            if "P4-Epistemic" in p_name:
                tier = validator.PROTOCOL_TIERS.get("P4-Epistemic", "unknown")
            elif "-P5" in p_name:
                tier = validator.PROTOCOL_TIERS.get("FP-5", "unknown")
            elif "-P6" in p_name:
                tier = validator.PROTOCOL_TIERS.get("FP-6", "unknown")
            elif "-P7" in p_name:
                tier = validator.PROTOCOL_TIERS.get("FP-7", "unknown")
            else:
                p_num = int(p_name.split("-")[-1])
                tier = validator.PROTOCOL_TIERS.get(p_num, "unknown")
        except (ValueError, IndexError):
            tier = "unknown"
        status = "✓ PASS" if res.get("passed") else "✗ FAIL"
        msg = res.get("message", "No description provided")
        print(f"{p_name:<15} | {tier:<10} | {status:<10} | {msg}")

    print("-" * 80)
    passed_count = sum(
        1 for p in validator.protocol_results.values() if p.get("passed", False)
    )
    total_count = len(validator.protocol_results)

    print(f"\nFinal Summary: {passed_count}/{total_count} protocols passed.")
    if passed_count == total_count:
        print("\nOVERALL STATUS: ✓ PASS (100/100 ALIGNMENT REACHED)")
    else:
        print("\nOVERALL STATUS: ✗ FAIL (Requires remediation)")

    print(f"\nReport saved to: {summary['output_files']['json']}")

    # Run Joint Falsification Aggregator
    print("\n" + "=" * 80)
    print(" JOINT FALSIFICATION AGGREGATION ".center(80, "="))
    print("=" * 80)
    aggregator = FalsificationAggregator()
    joint_report = aggregator.generate_master_report()

    print(
        f"\nFRAMEWORK FALSIFICATION STATUS: {'✓ PASSED' if joint_report['framework_falsified'] else '✗ FAILED'}"
    )
    print(
        f"Compliance Ratio: {joint_report['falsified_count']}/{joint_report['total_predictions']}"
    )
    print(f"Compliance Score: {joint_report['compliance_score']:.1f}/100")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
