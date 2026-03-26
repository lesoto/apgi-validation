"""
Cross-Protocol Consistency Verification

This module implements verification that results from different validation protocols
are internally consistent. It checks that the same falsification criteria produce
consistent results across protocols that share them.
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class CrossProtocolConsistencyChecker:
    """
    Verify consistency across validation protocols.

    Checks that:
    - Same falsification criteria produce consistent results across protocols
    - Parameter estimates are consistent across protocols
    - Statistical thresholds are applied consistently
    - Data processing pipelines are consistent
    """

    def __init__(self):
        self.consistency_results: Dict[str, Any] = {}
        self.protocol_results: Dict[str, Dict] = {}

    def add_protocol_results(self, protocol_name: str, results: Dict) -> None:
        """
        Add validation results from a protocol.

        Args:
            protocol_name: Name of the protocol (e.g., "Validation_Protocol_1")
            results: Dictionary containing the protocol's validation results
        """
        self.protocol_results[protocol_name] = results
        logger.info(f"Added results from {protocol_name} to consistency checker")

    def verify_falsification_criteria_consistency(self) -> Dict[str, Any]:
        """
        Verify that falsification criteria are applied consistently across protocols.

        Returns:
            Dictionary with consistency check results
        """
        consistency_results = {
            "criteria_consistency": {},
            "inconsistencies_found": [],
            "overall_consistency_score": 0.0,
        }

        # Define shared falsification criteria
        shared_criteria = [
            "F1.1",  # APGI performance advantage
            "F1.2",  # Somatic marker advantage
            "F1.3",  # Arousal interaction
            "F1.4",  # Threshold adaptation
            "F1.5",  # PAC modulation
            "F1.6",  # Spectral slope
            "F2.1",  # Advantageous selection
            "F2.2",  # Cost correlation
            "F2.3",  # RT advantage
            "F2.4",  # Confidence effect
            "F2.5",  # Time to criterion
            "F3.1",  # APGI advantage (repetition)
            "F3.2",  # Interoceptive advantage
            "F3.3",  # Threshold reduction
            "F3.4",  # Precision reduction
            "F3.5",  # Performance retention
            "F3.6",  # Sample efficiency
            "F5.1",  # Threshold agents
            "F5.2",  # Precision agents
            "F5.3",  # Interoceptive agents
            "F5.4",  # Multiscale agents
            "F5.5",  # PCA variance
            "F5.6",  # Performance difference
            "F6.1",  # LTCN transition
            "F6.2",  # Integration window
        ]

        # Collect criteria values from all protocols
        criteria_values = {}
        for criterion in shared_criteria:
            criteria_values[criterion] = {}

            for protocol_name, results in self.protocol_results.items():
                if "criteria" in results and criterion in results["criteria"]:
                    criterion_data = results["criteria"][criterion]
                    criteria_values[criterion][protocol_name] = criterion_data

        # Check consistency for each criterion
        consistent_count = 0
        total_checks = 0

        for criterion in shared_criteria:
            criterion_data = criteria_values[criterion]
            protocols_with_criterion = list(criterion_data.keys())

            if len(protocols_with_criterion) < 2:
                continue  # Need at least 2 protocols to compare

            # Check if pass/fail status is consistent
            pass_values = []
            for protocol in protocols_with_criterion:
                criterion_data[protocol]["passed"]
                pass_values.append(criterion_data[protocol].get("passed", False))

            # All protocols should have same pass/fail status
            all_consistent = all(p == pass_values[0] for p in pass_values)

            # Check if thresholds are consistent
            threshold_values = []
            for protocol in protocols_with_criterion:
                threshold_values.append(criterion_data[protocol].get("threshold", ""))

            thresholds_consistent = len(set(threshold_values)) <= 1

            # Check if p-values are consistent (within tolerance)
            p_values = []
            for protocol in protocols_with_criterion:
                p_values.append(criterion_data[protocol].get("p_value", 1.0))

            # Check if p-values are consistent (within 0.01 tolerance)
            p_consistent = max(p_values) - min(p_values) < 0.01

            criterion_consistent = (
                all_consistent and thresholds_consistent and p_consistent
            )

            consistency_results["criteria_consistency"][criterion] = {
                "consistent": criterion_consistent,
                "protocols": protocols_with_criterion,
                "pass_values": pass_values,
                "thresholds": threshold_values,
                "p_values": p_values,
                "all_consistent": all_consistent,
                "thresholds_consistent": thresholds_consistent,
                "p_values_consistent": p_consistent,
            }

            if criterion_consistent:
                consistent_count += 1
            else:
                inconsistency = {
                    "criterion": criterion,
                    "issue": "Inconsistent results",
                    "protocols": protocols_with_criterion,
                    "pass_values": pass_values,
                    "thresholds": threshold_values,
                    "p_values": p_values,
                }
                consistency_results["inconsistencies_found"].append(inconsistency)

            total_checks += 1

        # Calculate overall consistency score
        if total_checks > 0:
            consistency_results["overall_consistency_score"] = (
                consistent_count / total_checks
            )
        else:
            consistency_results["overall_consistency_score"] = 0.0

        logger.info(
            f"Criteria consistency: {consistent_count}/{total_checks} "
            f"({consistency_results['overall_consistency_score']:.2f})"
        )

        return consistency_results

    def verify_parameter_consistency(self) -> Dict[str, Any]:
        """
        Verify that APGI parameter estimates are consistent across protocols.

        Returns:
            Dictionary with parameter consistency results
        """
        parameter_results = {
            "parameter_consistency": {},
            "inconsistencies_found": [],
            "overall_consistency_score": 0.0,
        }

        # Define key APGI parameters
        key_parameters = [
            "beta",  # Phase transition steepness
            "theta",  # Ignition threshold
            "Pi_e",  # Interoceptive precision
            "Pi_i",  # Inhibitory precision
            "alpha",  # Sensitivity parameter
        ]

        # Collect parameter estimates from all protocols
        parameter_values = {}
        for param in key_parameters:
            parameter_values[param] = {}

            for protocol_name, results in self.protocol_results.items():
                # Look for parameter estimates in various locations
                param_estimate = None

                # Check in different possible locations
                if "posterior_summary" in results:
                    posterior = results["posterior_summary"]
                    if param in posterior:
                        param_estimate = posterior[param].get("mean", None)
                elif "parameter_estimates" in results:
                    param_estimate = results["parameter_estimates"].get(param, None)
                elif "fitted_parameters" in results:
                    param_estimate = results["fitted_parameters"].get(param, None)

                if param_estimate is not None:
                    parameter_values[param][protocol_name] = param_estimate

        # Check consistency for each parameter
        consistent_count = 0
        total_checks = 0

        for param in key_parameters:
            param_data = parameter_values[param]
            protocols_with_param = list(param_data.keys())

            if len(protocols_with_param) < 2:
                continue  # Need at least 2 protocols to compare

            # Calculate coefficient of variation (CV)
            param_values_list = [
                param_data[protocol] for protocol in protocols_with_param
            ]
            mean_param = np.mean(param_values_list)
            std_param = np.std(param_values_list)
            cv = std_param / mean_param if mean_param != 0 else 0

            # Check if CV is acceptable (< 0.20 for consistency)
            cv_acceptable = cv < 0.20

            # Check if values are within reasonable range
            # Beta should be 8-16, Theta should be 0.3-0.7, Pi should be 0.5-2.0
            reasonable_range = True
            if param == "beta":
                reasonable_range = 8.0 <= mean_param <= 16.0
            elif param == "theta":
                reasonable_range = 0.3 <= mean_param <= 0.7
            elif param == "Pi_e" or param == "Pi_i":
                reasonable_range = 0.5 <= mean_param <= 2.0
            elif param == "alpha":
                reasonable_range = 2.0 <= mean_param <= 10.0

            param_consistent = cv_acceptable and reasonable_range

            parameter_results["parameter_consistency"][param] = {
                "consistent": param_consistent,
                "protocols": protocols_with_param,
                "values": param_values_list,
                "mean": float(mean_param),
                "std": float(std_param),
                "cv": float(cv),
                "cv_acceptable": cv_acceptable,
                "reasonable_range": reasonable_range,
            }

            if param_consistent:
                consistent_count += 1
            else:
                inconsistency = {
                    "parameter": param,
                    "issue": "Inconsistent parameter estimate",
                    "protocols": protocols_with_param,
                    "values": param_values_list,
                    "mean": float(mean_param),
                    "std": float(std_param),
                    "cv": float(cv),
                    "cv_acceptable": cv_acceptable,
                    "reasonable_range": reasonable_range,
                }
                parameter_results["inconsistencies_found"].append(inconsistency)

            total_checks += 1

        # Calculate overall consistency score
        if total_checks > 0:
            parameter_results["overall_consistency_score"] = (
                consistent_count / total_checks
            )
        else:
            parameter_results["overall_consistency_score"] = 0.0

        logger.info(
            f"Parameter consistency: {consistent_count}/{total_checks} "
            f"({parameter_results['overall_consistency_score']:.2f})"
        )

        return parameter_results

    def verify_statistical_threshold_consistency(self) -> Dict[str, Any]:
        """
        Verify that statistical thresholds are applied consistently across protocols.

        Returns:
            Dictionary with threshold consistency results
        """
        threshold_results = {
            "threshold_consistency": {},
            "inconsistencies_found": [],
            "overall_consistency_score": 0.0,
        }

        # Define key statistical thresholds
        key_thresholds = {
            "alpha_level": 0.05,
            "cohens_d_minimum": 0.30,
            "eta_squared_minimum": 0.20,
            "r_squared_minimum": 0.70,
            "bayes_factor_strong": 6.0,
        }

        # Collect threshold values from all protocols
        threshold_values = {}
        for threshold, target_value in key_thresholds.items():
            threshold_values[threshold] = {}

            for protocol_name, results in self.protocol_results.items():
                # Look for threshold in various locations
                threshold_value = None

                # Check in different possible locations
                if "thresholds" in results:
                    threshold_value = results["thresholds"].get(threshold, None)
                elif "statistical_thresholds" in results:
                    threshold_value = results["statistical_thresholds"].get(
                        threshold, None
                    )
                elif "criteria" in results:
                    for criterion_data in results["criteria"].values():
                        if "threshold" in criterion_data:
                            threshold_value = criterion_data["threshold"]
                            break

                if threshold_value is not None:
                    # Ensure threshold_value is a float
                    try:
                        # Handle string thresholds like "r ≥ 0.70" by extracting the numeric part
                        if isinstance(threshold_value, str):
                            import re

                            # Extract numeric values from strings like "r ≥ 0.70", "h ≥ 0.30", "p < 0.05"
                            numeric_match = re.search(r"(\d+\.?\d*)", threshold_value)
                            if numeric_match:
                                threshold_value = float(numeric_match.group(1))
                            else:
                                raise ValueError(
                                    "No numeric value found in threshold string"
                                )
                        else:
                            threshold_value = float(threshold_value)
                        threshold_values[threshold][protocol_name] = threshold_value
                    except (ValueError, TypeError):
                        # Skip invalid threshold values
                        continue

        # Check consistency for each threshold
        consistent_count = 0
        total_checks = 0

        for threshold, target_value in key_thresholds.items():
            threshold_data = threshold_values[threshold]
            protocols_with_threshold = list(threshold_data.keys())

            if len(protocols_with_threshold) < 2:
                continue  # Need at least 2 protocols to compare

            # Check if threshold values are consistent (within 0.01 tolerance)
            threshold_values_list = [
                threshold_data[protocol] for protocol in protocols_with_threshold
            ]

            # Check if all values are close to target
            all_close_to_target = all(
                abs(tv - target_value) < 0.01 for tv in threshold_values_list
            )

            # Check if all values are consistent with each other
            all_close_to_each_other = all(
                abs(tv - threshold_values_list[0]) < 0.01
                for tv in threshold_values_list[1:]
            )

            threshold_consistent = all_close_to_target and all_close_to_each_other

            threshold_results["threshold_consistency"][threshold] = {
                "consistent": threshold_consistent,
                "protocols": protocols_with_threshold,
                "values": threshold_values_list,
                "target_value": target_value,
                "all_close_to_target": all_close_to_target,
                "all_close_to_each_other": all_close_to_each_other,
            }

            if threshold_consistent:
                consistent_count += 1
            else:
                inconsistency = {
                    "threshold": threshold,
                    "issue": "Inconsistent threshold value",
                    "protocols": protocols_with_threshold,
                    "values": threshold_values_list,
                    "target_value": target_value,
                }
                threshold_results["inconsistencies_found"].append(inconsistency)

            total_checks += 1

        # Calculate overall consistency score
        if total_checks > 0:
            threshold_results["overall_consistency_score"] = (
                consistent_count / total_checks
            )
        else:
            threshold_results["overall_consistency_score"] = 0.0

        logger.info(
            f"Threshold consistency: {consistent_count}/{total_checks} "
            f"({threshold_results['overall_consistency_score']:.2f})"
        )

        return threshold_results

    def generate_consistency_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive consistency report.

        Returns:
            Dictionary with all consistency check results
        """
        logger.info("Generating cross-protocol consistency report")

        report = {
            "protocol_count": len(self.protocol_results),
            "criteria_consistency": self.verify_falsification_criteria_consistency(),
            "parameter_consistency": self.verify_parameter_consistency(),
            "threshold_consistency": self.verify_statistical_threshold_consistency(),
            "summary": {},
        }

        # Calculate overall consistency score
        criteria_score = report["criteria_consistency"]["overall_consistency_score"]
        parameter_score = report["parameter_consistency"]["overall_consistency_score"]
        threshold_score = report["threshold_consistency"]["overall_consistency_score"]

        overall_score = (criteria_score + parameter_score + threshold_score) / 3

        report["summary"] = {
            "overall_consistency_score": overall_score,
            "criteria_score": criteria_score,
            "parameter_score": parameter_score,
            "threshold_score": threshold_score,
            "total_inconsistencies": (
                len(report["criteria_consistency"]["inconsistencies_found"])
                + len(report["parameter_consistency"]["inconsistencies_found"])
                + len(report["threshold_consistency"]["inconsistencies_found"])
            ),
            "protocol_results": self.protocol_results,
        }

        logger.info(f"Overall consistency score: {overall_score:.3f}")

        return report

    def check_protocol_data_pipeline_consistency(self) -> Dict[str, Any]:
        """
        Verify that data processing pipelines are consistent across protocols.

        Returns:
            Dictionary with pipeline consistency results
        """
        pipeline_results = {
            "pipeline_consistency": {},
            "inconsistencies_found": [],
            "overall_consistency_score": 0.0,
        }

        # Check for consistent data preprocessing
        preprocessing_checks = {
            "normalization_applied": False,
            "outlier_removal_applied": False,
            "missing_data_handled": False,
            "bootstrap_ci_applied": False,
        }

        for protocol_name, results in self.protocol_results.items():
            # Check for evidence of preprocessing
            if "normalized_data" in results or "z_scores" in results:
                preprocessing_checks["normalization_applied"] = True
            if "outliers_removed" in results or "robust_statistics" in results:
                preprocessing_checks["outlier_removal_applied"] = True
            if "missing_data_handled" in results or "imputation" in results:
                preprocessing_checks["missing_data_handled"] = True
            if "bootstrap_ci" in results or "confidence_intervals" in results:
                preprocessing_checks["bootstrap_ci_applied"] = True

        # Check for consistent statistical test usage
        statistical_tests_used = set()
        for protocol_name, results in self.protocol_results.items():
            if "criteria" in results:
                for criterion_data in results["criteria"].values():
                    if "p_value" in criterion_data:
                        statistical_tests_used.add("p_value")
                    if "t_statistic" in criterion_data:
                        statistical_tests_used.add("t_statistic")
                    if "cohens_d" in criterion_data:
                        statistical_tests_used.add("cohens_d")
                    if "eta_squared" in criterion_data:
                        statistical_tests_used.add("eta_squared")

        pipeline_results["statistical_tests_used"] = list(statistical_tests_used)

        # Calculate pipeline consistency score
        preprocessing_score = sum(preprocessing_checks.values()) / len(
            preprocessing_checks
        )
        statistical_score = (
            len(statistical_tests_used) / 10
        )  # At least 10 different tests
        overall_score = (preprocessing_score + statistical_score) / 2

        pipeline_results["overall_consistency_score"] = overall_score
        pipeline_results["preprocessing_checks"] = preprocessing_checks
        pipeline_results["statistical_tests_used"] = list(statistical_tests_used)

        logger.info(f"Pipeline consistency score: {overall_score:.3f}")

        return pipeline_results


def run_cross_protocol_consistency_check(
    protocol_results: Dict[str, Dict]
) -> Dict[str, Any]:
    """
    Run cross-protocol consistency check.

    Args:
        protocol_results: Dictionary mapping protocol names to their validation results

    Returns:
        Dictionary with consistency check results
    """
    checker = CrossProtocolConsistencyChecker()

    # Add protocol results
    for protocol_name, results in protocol_results.items():
        checker.add_protocol_results(protocol_name, results)

    # Generate consistency report
    report = checker.generate_consistency_report()

    return report


if __name__ == "__main__":
    # Example usage
    print("Cross-Protocol Consistency Verification")
    print("=" * 50)

    # Create mock protocol results
    mock_results = {
        "Validation_Protocol_1": {
            "criteria": {
                "F1.1": {"passed": True, "p_value": 0.01, "threshold": "r ≥ 0.70"},
                "F1.2": {"passed": True, "p_value": 0.02, "threshold": "h ≥ 0.30"},
                "F1.3": {
                    "passed": True,
                    "p_value": 0.01,
                    "threshold": "interaction p < 0.05",
                },
            },
            "thresholds": {
                "alpha_level": 0.05,
                "cohens_d_minimum": 0.30,
                "eta_squared_minimum": 0.20,
                "bayes_factor_strong": 6.0,
            },
            "posterior_summary": {
                "beta": {"mean": 12.0, "std": 1.5},
                "theta": {"mean": 0.5, "std": 0.08},
            },
        },
        "Validation_Protocol_2": {
            "criteria": {
                "F1.1": {"passed": True, "p_value": 0.01, "threshold": "r ≥ 0.70"},
                "F1.2": {"passed": True, "p_value": 0.02, "threshold": "h ≥ 0.30"},
            },
            "thresholds": {
                "alpha_level": 0.05,
                "cohens_d_minimum": 0.30,
                "eta_squared_minimum": 0.20,
                "bayes_factor_strong": 6.0,
            },
            "posterior_summary": {
                "beta": {"mean": 12.5, "std": 1.8},
                "theta": {"mean": 0.52, "std": 0.09},
            },
        },
        "Validation_Protocol_3": {
            "criteria": {
                "F1.1": {"passed": True, "p_value": 0.01, "threshold": "r ≥ 0.70"},
                "F1.2": {"passed": True, "p_value": 0.02, "threshold": "h ≥ 0.30"},
            },
            "thresholds": {
                "alpha_level": 0.05,
                "cohens_d_minimum": 0.30,
                "eta_squared_minimum": 0.20,
                "bayes_factor_strong": 6.0,
            },
            "posterior_summary": {
                "beta": {"mean": 11.8, "std": 1.2},
                "theta": {"mean": 0.48, "std": 0.07},
            },
        },
    }

    # Run consistency check
    report = run_cross_protocol_consistency_check(mock_results)

    print("\nConsistency Report:")
    print("-" * 50)
    print(
        f"Overall consistency score: {report['summary']['overall_consistency_score']:.3f}"
    )
    print(f"Total inconsistencies: {report['summary']['total_inconsistencies']}")

    print("\nDetailed Results:")
    print(
        f"Criteria consistency: {report['criteria_consistency']['overall_consistency_score']:.3f}"
    )
    print(
        f"Parameter consistency: {report['parameter_consistency']['overall_consistency_score']:.3f}"
    )
    print(
        f"Threshold consistency: {report['threshold_consistency']['overall_consistency_score']:.3f}"
    )

    if report["summary"]["total_inconsistencies"] > 0:
        print("\nInconsistencies found:")
        for inconsistency in report["criteria_consistency"]["inconsistencies_found"]:
            print(f"  - {inconsistency}")
        for inconsistency in report["parameter_consistency"]["inconsistencies_found"]:
            print(f"  - {inconsistency}")
        for inconsistency in report["threshold_consistency"]["inconsistencies_found"]:
            print(f"  - {inconsistency}")

    print("\nProtocols analyzed:", report["protocol_count"])
