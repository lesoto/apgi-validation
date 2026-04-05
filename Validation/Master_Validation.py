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

# Add project root to sys.path for imports
_proj_root = Path(__file__).parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

# Try to import logging config
try:
    from utils.logging_config import apgi_logger as logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[misc,assignment]

# Import validation-falsification consistency checker
try:
    from utils.validation_falsification_consistency import (
        ValidationFalsificationConsistency,
        ConsistencyIssue,
    )
except ImportError:
    # Fallback placeholders when module is not available
    ValidationFalsificationConsistency = None  # type: ignore[misc]
    ConsistencyIssue = None  # type: ignore[misc]


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
        # - Tertiary (6-7, 9-10): Specialized and experimental protocols
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
            7: "secondary",
            8: "secondary",
            9: "tertiary",
            10: "tertiary",
            11: "secondary",
            12: "secondary",
            13: "secondary",
            14: "tertiary",
            15: "tertiary",
            16: "secondary",  # Protocol ALL (Aggregator)
        }
        self.falsification_status = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }
        self.timeout_seconds = 3600  # Increased to prevent GUI hanging or premature aborts of complex protocols like VP-05
        # Protocol dependencies: protocols that must run before others
        self.protocol_dependencies = {
            "Protocol-1": {"dependencies": []},
            "Protocol-2": {"dependencies": []},
            "Protocol-3": {"dependencies": []},
            "Protocol-4": {"dependencies": []},
            "Protocol-5": {
                "dependencies": [],
                "must_run_before": ["FP_01", "FP_02", "FP_03", "FP_05", "FP_06"],
            },  # VP-05: Evolutionary Emergence
            "Protocol-6": {"dependencies": []},
            "Protocol-7": {"dependencies": []},
            "Protocol-8": {"dependencies": []},
            "Protocol-9": {"dependencies": []},
            "Protocol-10": {"dependencies": []},
            "Protocol-11": {"dependencies": []},
            "Protocol-12": {"dependencies": []},
            "Protocol-13": {"dependencies": []},
            "Protocol-14": {"dependencies": []},
            "Protocol-15": {"dependencies": []},
        }

        # Falsification protocol dependencies: VP-05 must complete before certain falsification protocols
        self.falsification_dependencies = {
            "FP-01": ["Protocol-5"],  # ActiveInference depends on VP-05
            "FP-02": ["Protocol-5"],  # AgentComparison depends on VP-05
            "FP-03": ["Protocol-5"],  # FrameworkLevel depends on VP-05
            "FP-05": ["Protocol-5"],  # EvolutionaryPlausibility depends on VP-05
            "FP-06": ["Protocol-5"],  # LiquidNetwork_EnergyBenchmark depends on VP-05
        }
        # Pending protocols: awaiting empirical data (excluded from scoring denominator)
        self.PENDING_PROTOCOLS = []  # All current protocols implemented
        self.tier_weights = {
            "primary": 2.0,
            "secondary": 1.5,
            "tertiary": 1.0,
        }  # Initialize tier weights
        self.available_protocols = {
            "Protocol-1": {
                "file": "VP_01_SyntheticEEG_MLClassification.py",
                "function": "run_validation",
                "description": "Computational Support: Synthetic Neural Data Simulations for Protocol 1 (NOT Paper Protocol 1)",
            },
            "Protocol-2": {
                "file": "VP_02_Behavioral_BayesianComparison.py",
                "function": "run_validation",
                "description": "Behavioral Bayesian Comparison (P1.1–P1.3, V2.1–V2.3, F2.1–F2.5)",
            },
            "Protocol-3": {
                "file": "VP_03_ActiveInference_AgentSimulations.py",
                "function": "run_validation",
                "description": "Active Inference Agent Comparison Experiment",
            },
            "Protocol-4": {
                "file": "VP_04_PhaseTransition_EpistemicLevel2.py",
                "function": "run_validation",
                "description": "Phase Transition / Epistemic Architecture Level 2",
            },
            "Protocol-5": {
                "file": "VP_05_EvolutionaryEmergence.py",
                "function": "run_validation",
                "description": "Evolutionary Emergence of APGI Architectures",
            },
            "Protocol-6": {
                "file": "VP_06_LiquidNetwork_InductiveBias.py",
                "function": "run_validation",
                "description": "Liquid Network Inductive Bias Benchmark (Paper 2)",
            },
            "Protocol-7": {
                "file": "VP_07_TMS_CausalInterventions.py",
                "function": "run_validation",
                "description": "TMS/Pharmacological Causal Interventions (Paper 1 — Protocol 2)",
            },
            "Protocol-8": {
                "file": "VP_08_Psychophysical_ThresholdEstimation.py",
                "function": "run_validation",
                "description": "Psychophysical Threshold Estimation (Paper 1 — Protocol 1)",
            },
            "Protocol-9": {
                "file": "VP_09_NeuralSignatures_EmpiricalPriority1.py",
                "function": "run_validation",
                "description": "Convergent Neural Signatures — Empirical Roadmap Priority 1",
            },
            "Protocol-10": {
                "file": "VP_10_CausalManipulations_Priority2.py",
                "function": "run_validation",
                "description": "Causal Manipulations TMS/Pharmacological — Priority 2",
            },
            "Protocol-11": {
                "file": "VP_11_MCMC_CulturalNeuroscience_Priority3.py",
                "function": "run_validation",
                "description": "MCMC / Cultural Neuroscience — Priority 3 (Gelman-Rubin R̂ ≤ 1.01)",
            },
            "Protocol-12": {
                "file": "VP_12_Clinical_CrossSpecies_Convergence.py",
                "function": "run_validation",
                "description": "Clinical Cross-Species Convergence (Paper 1 — Protocol 4)",
            },
            "Protocol-13": {
                "file": "VP_13_Epistemic_Architecture.py",
                "function": "run_validation",
                "description": "Epistemic Architecture Predictions P5–P12 (Paper 4)",
            },
            "Protocol-14": {
                "file": "VP_14_fMRI_Anticipation_Experience.py",
                "function": "run_validation",
                "description": "fMRI Anticipation/Experience Protocol 14 (Simulation-Validated, Awaiting Empirical)",
            },
            "Protocol-15": {
                "file": "VP_15_fMRI_Anticipation_vmPFC.py",
                "function": "run_validation",
                "description": "fMRI Anticipation vmPFC (STUB — Awaiting Data)",
            },
        }
        self.PROTOCOL_DESCRIPTIONS = {
            k: v["description"] for k, v in self.available_protocols.items()
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
                    "passed": "false",  # String for type consistency
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
        except NotImplementedError as e:
            # Special handling for VP-15 and other stub protocols
            if "VP-15" in protocol_info["file"] or "VP_15" in protocol_info["file"]:
                logger.warning(f"VP-15 is a stub awaiting empirical data: {e}")
                return {
                    "status": "STUB",
                    "passed": None,
                    "protocol_id": "VP-15",
                    "protocol_name": "fMRI vmPFC Anticipation Paradigm",
                    "named_predictions": {
                        "V15.1": {
                            "passed": None,
                            "reason": "Awaiting empirical fMRI data",
                        },
                        "V15.2": {
                            "passed": None,
                            "reason": "Awaiting empirical fMRI data",
                        },
                        "V15.3": {
                            "passed": None,
                            "reason": "Awaiting empirical fMRI data",
                        },
                    },
                    "data_source": None,
                    "reason": "Awaiting empirical fMRI data for vmPFC anticipation paradigm",
                }
            return {"status": "error", "message": str(e), "passed": False}
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
                "pending_protocols": 0,
                "success_rate": 0,
                "weighted_score": 0,
                "protocol_results": {},
                "falsification_status": self.falsification_status,
                "summary": "Run validation protocols first",
            }

        # Count protocols by status: passed, failed, pending
        passed_protocols = sum(
            1 for r in self.protocol_results.values() if r.get("passed", False)
        )
        pending_protocols = sum(
            1
            for r in self.protocol_results.values()
            if r.get("passed") is None or r.get("status") == "STUB_AWAITING_DATA"
        )
        completed_protocols = total_protocols - pending_protocols

        # Success rate only over completed protocols (pending excluded from denominator)
        success_rate = (
            passed_protocols / completed_protocols if completed_protocols > 0 else 0
        )

        # Equal weighting across completed protocols (excluding pending)
        tier_weights = {"primary": 2.0, "secondary": 1.5, "tertiary": 1.0}
        self.tier_weights = (
            tier_weights  # Store as instance attribute for external access
        )
        tier_stats = {
            "primary": {"passed": 0, "total": 0, "pending": 0},
            "secondary": {"passed": 0, "total": 0, "pending": 0},
            "tertiary": {"passed": 0, "total": 0, "pending": 0},
        }

        for p_name, result in self.protocol_results.items():
            # Extract protocol number from "Protocol-X"
            p_num = None  # Initialize to None to avoid UnboundLocalError
            try:
                p_num = int(p_name.split("-")[-1])
                tier = self.PROTOCOL_TIERS.get(p_num, "tertiary")
            except (ValueError, IndexError):
                tier = "tertiary"

            # Check if pending (awaiting data)
            is_pending = (
                result.get("passed") is None
                or (
                    p_num is not None
                    and p_num in getattr(self, "PENDING_PROTOCOLS", [])
                )
                or result.get("status") == "STUB_AWAITING_DATA"
            )

            if is_pending:
                tier_stats[tier]["pending"] += 1
            else:
                tier_stats[tier]["total"] += 1
                if result.get("passed", False):
                    tier_stats[tier]["passed"] += 1

        weighted_score = 0.0
        total_weight_used = 0.0
        for tier, stats in tier_stats.items():
            # Exclude pending from tier calculations
            completed = stats["total"]  # Already excludes pending
            if completed > 0:
                tier_success = stats["passed"] / completed
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

        summary = (
            f"Validated {passed_protocols}/{completed_protocols} completed protocols "
            f"({pending_protocols} pending) (Raw: {success_rate:.1%}, Weighted: {weighted_score:.2f})"
        )

        return {
            "overall_decision": overall_decision,
            "total_protocols": total_protocols,
            "completed_protocols": completed_protocols,
            "passed_protocols": passed_protocols,
            "pending_protocols": pending_protocols,
            "success_rate": success_rate,
            "weighted_score": weighted_score,
            "tier_summary": tier_stats,
            "pending_list": getattr(self, "PENDING_PROTOCOLS", []),
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

        # Resolve ordering before executing the protocol queue
        in_degree = {p: 0 for p in all_protocols}
        graph: Dict[str, List[str]] = {p: [] for p in all_protocols}

        for p in all_protocols:
            deps_entry = self.protocol_dependencies.get(p, {})
            deps = (
                deps_entry.get("dependencies", [])
                if isinstance(deps_entry, dict)
                else deps_entry
            )

            # backward deps
            for d in deps:
                if d in all_protocols:
                    graph[d].append(p)
                    in_degree[p] += 1

            # forward deps (must_run_before)
            if isinstance(deps_entry, dict):
                forward_deps = deps_entry.get("must_run_before", [])
                for fd in forward_deps:
                    if fd in all_protocols:
                        graph[p].append(fd)
                        in_degree[fd] += 1

        # Topological Sort
        queue = []
        zero_in_degree = [p for p in all_protocols if in_degree[p] == 0]
        while zero_in_degree:
            curr = zero_in_degree.pop(0)
            queue.append(curr)
            for neighbor in graph.get(curr, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    zero_in_degree.append(neighbor)

        for p in all_protocols:
            if p not in queue:
                queue.append(p)

        all_protocols = queue

        # Run in dependency order
        executed = set()
        results = {}

        for protocol_name in all_protocols:
            if protocol_name in executed:
                continue

            # Run current protocol
            protocol_results = self.run_validation([protocol_name], **kwargs)
            results.update(protocol_results)
            executed.add(protocol_name)

        return results

    def generate_reproducibility_package(
        self, output_dir: Optional[Path] = None
    ) -> Dict:
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
            output_path = Path(__file__).parent.parent / "reproducibility"
        else:
            output_path = Path(output_dir)

        output_path.mkdir(parents=True, exist_ok=True)

        # Generate reproducibility data
        reproducibility_data = {
            "timestamp": datetime.now().isoformat(),
            "protocol_results": self.protocol_results,
            "tier_classification": self.PROTOCOL_TIERS,
            "tier_weights": {"primary": 2.0, "secondary": 1.5, "tertiary": 1.0},
            "instance_tier_weights": self.tier_weights,
            "available_protocols": self.available_protocols,
            "protocol_dependencies": self.protocol_dependencies,
            "falsification_dependencies": getattr(
                self, "falsification_dependencies", {}
            ),
        }

        # Save reproducibility data
        output_file = (
            output_path
            / f"validation_reproducibility_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(reproducibility_data, f, indent=2)

        # Save results as CSV
        results_df = self._results_to_dataframe()
        csv_file = (
            output_path
            / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        results_df.to_csv(csv_file, index=False)

        return {
            "reproducibility_data": reproducibility_data,
            "output_files": {
                "json": str(output_file),
                "csv": str(csv_file),
            },
            "output_directory": str(output_path),
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
                p_num = int(protocol_name.split("-")[-1])
                row["tier"] = self.PROTOCOL_TIERS.get(p_num, "unknown")
            except (ValueError, IndexError):
                row["tier"] = "unknown"

            rows.append(row)

        return pd.DataFrame(rows)


def main():
    """Main entry point for Master Validation"""
    import argparse

    parser = argparse.ArgumentParser(description="APGI Master Validation Orchestrator")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run with reduced trials for speed"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    validator = APGIMasterValidator()
    print("\n" + "=" * 80)
    print(" APGI MASTER VALIDATION ORCHESTRATOR ".center(80, "="))
    print("=" * 80)

    # Run all available protocols
    run_kwargs = {"save_outputs": True, "seed": args.seed}
    if args.dry_run:
        run_kwargs.update({"n_trials": 5, "n_agents": 2, "epochs": 1})
        print("NOTE: Running in DRY-RUN mode with reduced parameters.")

    _ = validator.run_all_protocols(**run_kwargs)
    summary = validator.generate_master_report()

    # Print results table
    print("\nVALIDATION RESULTS:")
    print("-" * 80)
    print(f"{'PROTOCOL':<15} | {'TIER':<10} | {'STATUS':<10} | {'MESSAGE'}")
    print("-" * 80)

    for p_name in sorted(validator.protocol_results.keys()):
        res = validator.protocol_results[p_name]
        p_num_str = p_name.split("-")[-1]
        try:
            p_num = int(p_num_str)
            tier = validator.PROTOCOL_TIERS.get(p_num, "unknown")
        except (ValueError, IndexError):
            tier = "unknown"
        status = "✓ PASS" if res.get("passed") else "✗ FAIL"
        msg = res.get("message", "No description provided")
        print(f"{p_name:<15} | {tier:<10} | {status:<10} | {msg}")

    print("-" * 80)
    passed_count = summary.get("passed_protocols", 0)
    total_count = summary.get("completed_protocols", 0)

    print(f"\nFinal Summary: {passed_count}/{total_count} protocols passed.")
    if summary.get("weighted_score", 0) >= 0.85:
        print("\nOVERALL STATUS: ✓ PASS (HIGH-ALIGNMENT REACHED)")
    else:
        print("\nOVERALL STATUS: ✗ FAIL (Requires remediation)")

    if "output_files" in summary and "json" in summary["output_files"]:
        print(f"\nReport saved to: {summary['output_files']['json']}")

    # Only run falsification if NOT in dry-run (falsification is heavy)
    if not args.dry_run:
        print("\n" + "=" * 80)
        print(" JOINT FALSIFICATION AGGREGATION ".center(80, "="))
        print("=" * 80)
        from Falsification.FP_ALL_Aggregator import (
            aggregate_prediction_results,
            check_framework_falsification_condition_a,
            check_framework_falsification_condition_b,
        )
        from utils.validation_falsification_consistency import (
            ValidationFalsificationConsistency,
        )

        preds = aggregate_prediction_results(validator.protocol_results)

        # Run validation-falsification consistency checks
        if ValidationFalsificationConsistency is not None:
            consistency_checker = ValidationFalsificationConsistency(
                fp_results=validator.protocol_results, vp_results={}
            )

            # Extract VP results for consistency checking
            # VP results are stored with different keys, so we need to extract them
            vp_results_for_consistency = {}
            for protocol_name, protocol_data in validator.protocol_results.items():
                if protocol_name.startswith("VP_"):
                    vp_results_for_consistency[protocol_name] = protocol_data

            consistency_report = consistency_checker.generate_consistency_report()

            # Add consistency report to summary
            if consistency_report["total_issues"] > 0:
                print("\n" + "=" * 80)
                print(" VALIDATION-FALSIFICATION CONSISTENCY CHECKS ".center(80, "="))
                print("=" * 80)

                print(f"Total Issues Found: {consistency_report['total_issues']}")
                print(
                    f"High Severity Issues: {consistency_report['high_severity_issues']}"
                )

                if consistency_report["assumption_violations"] > 0:
                    print("⚠️  ASSUMPTION VIOLATIONS DETECTED:")
                    for issue in consistency_report["assumption_violations"]:
                        print(f"  - {issue.description}")

                if consistency_report["contradictions"] > 0:
                    print("⚠️  CONTRADICTIONS DETECTED:")
                    for issue in consistency_report["contradictions"]:
                        print(f"  - {issue.description}")

                if consistency_report["missing_validations"] > 0:
                    print("⚠️  MISSING VALIDATION PROTOCOLS:")
                    for issue in consistency_report["missing_validations"]:
                        print(f"  - {issue.description}")

                print("\nRecommendations:")
                for rec in consistency_report["recommendations"]:
                    print(f"  - {rec}")

                # Check if ready for falsification
                if not consistency_report["ready_for_falsification"]:
                    print(
                        "\n🚨 CRITICAL: High-severity consistency issues must be resolved before falsification"
                    )
                    print("HALTING falsification evaluation")
                    return summary

                print("=" * 80)

        fa = check_framework_falsification_condition_a(preds)
        fb = check_framework_falsification_condition_b(
            results_input=validator.protocol_results
        )

        print(
            f"\nCondition A (Simultaneous Failure): {'✗ FAILED' if fa else '✓ PASSED'}"
        )
        print(f"Condition B (Parsimony / BIC): {'✗ FAILED' if fb else '✓ PASSED'}")

        # Generate detailed report with weighted scoring
        report = f"""
# APGI Master Validation Report
================================================================================

## Summary Statistics
- **Total Protocols**: {len(validator.protocol_results)}
- **Completed Protocols**: {summary.get("completed_protocols", 0)}
- **Failed Protocols**: {summary.get("failed_protocols", 0)}
- **Overall Success Rate**: {summary.get("success_rate", 0.0):.1%}
- **Weighted Score**: {summary.get("weighted_score", 0.0):.3f}

## Protocol Performance (Weighted by Tier)
"""

        for protocol_id, result in validator.protocol_results.items():
            if protocol_id in validator.PROTOCOL_TIERS:
                tier = validator.PROTOCOL_TIERS[protocol_id]
                weight = validator.tier_weights.get(tier, 1.0)
                score = result.get("score", 0.0) if result.get("passed", False) else 0.0
                weighted_score = score * weight

                report += f"""
### {protocol_id} (Tier: {tier})
- **Status**: {'✅ PASSED' if result.get('passed', False) else '❌ FAILED'}
- **Raw Score**: {result.get('score', 0.0):.3f}
- **Weight**: {weight:.1f}
- **Weighted Score**: {weighted_score:.3f}
"""

        report += """
## Detailed Results
"""

        # Add detailed results for each protocol
        for protocol_id, result in validator.protocol_results.items():
            if protocol_id in validator.PROTOCOL_TIERS:
                tier = validator.PROTOCOL_TIERS[protocol_id]
                weight = validator.tier_weights.get(tier, 1.0)
                score = result.get("score", 0.0) if result.get("passed", False) else 0.0
                weighted_score = score * weight

                report += f"""
**Protocol {protocol_id}**:
- Description: {validator.PROTOCOL_DESCRIPTIONS.get(protocol_id, 'Unknown protocol')}
- Status: {'✅ PASSED' if result.get('passed', False) else '❌ FAILED'}
- Raw Score: {result.get('score', 0.0):.3f}
- Weight: {weight:.1f}
- Weighted Score: {weighted_score:.3f}
"""
                if isinstance(result, dict) and "details" in result:
                    report += f"""
- Details: {result['details']}
"""

        report += """

## Compliance Analysis
The weighted scoring system prioritizes protocols by their scientific importance:
- **Primary protocols** (2.0x weight): Core agent behaviors and active inference
- **Secondary protocols** (1.5x weight): Multi-agent systems and convergence analysis  
- **Tertiary protocols** (1.0x weight): Supporting analyses and validation

Final weighted score reflects overall APGI framework performance.
"""

        print(report)

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
