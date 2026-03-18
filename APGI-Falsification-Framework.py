"""
APGI Falsification Testing Framework
====================================

Complete falsification testing framework for APGI theory including:
- Specific null predictions for each priority
- Statistical falsification criteria
- Popperian falsification protocols
- Robustness testing against alternative explanations

"""

import logging
import warnings
from enum import Enum
from typing import Dict, List, Union

import numpy as np
import scipy.stats as stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


class TestStatistic(Enum):
    """Enum for supported test statistics"""

    MEAN_DIFFERENCE = "mean_difference"
    CORRELATION = "correlation"
    MODEL_COMPARISON = "model_comparison"
    EFFECT_SIZE = "effect_size"


class FalsificationCriterion:
    """Individual falsification criterion"""

    def __init__(
        self,
        name: str,
        description: str,
        test_statistic: str,
        threshold: float,
        direction: str,
        alpha: float = 0.05,
    ):
        """
        Args:
            name: Short name for the criterion
            description: Detailed description
            test_statistic: Statistical test to perform
            threshold: Critical value for falsification
            direction: 'greater', 'less', or 'two_sided'
            alpha: Significance level
        """
        # Input validation
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Criterion name must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Criterion description must be a non-empty string")
        # Validate test_statistic using enum
        try:
            TestStatistic(test_statistic)
        except ValueError as e:
            raise ValueError(
                f"Unsupported test statistic: {test_statistic}. "
                f"Supported values: {[s.value for s in TestStatistic]}"
            ) from e
        if direction not in ["greater", "less", "two_sided"]:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'greater', 'less', or 'two_sided'"
            )
        if not (0 < alpha <= 1):
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")

        self.name = name
        self.description = description
        self.test_statistic = test_statistic
        self.threshold = threshold
        self.direction = direction
        self.alpha = alpha

        logger.debug(f"Initialized falsification criterion: {name}")

    def test(self, data: Union[np.ndarray, Dict]) -> Dict:
        """Test the falsification criterion"""

        if self.test_statistic == "mean_difference":
            return self._test_mean_difference(data)
        elif self.test_statistic == "correlation":
            return self._test_correlation(data)
        elif self.test_statistic == "model_comparison":
            return self._test_model_comparison(data)
        elif self.test_statistic == "effect_size":
            return self._test_effect_size(data)
        else:
            return {"error": f"Unknown test statistic: {self.test_statistic}"}

    def _test_mean_difference(self, data: Dict) -> Dict:
        """Test difference between means"""
        try:
            group1 = data.get("group1", [])
            group2 = data.get("group2", [])

            # Input validation
            if not isinstance(group1, (list, np.ndarray)) or not isinstance(
                group2, (list, np.ndarray)
            ):
                return {"error": "Groups must be lists or numpy arrays"}

            group1 = np.array(group1, dtype=float)
            group2 = np.array(group2, dtype=float)

            if len(group1) == 0 or len(group2) == 0:
                return {"error": "Insufficient data for mean difference test"}

            if len(group1) < 3 or len(group2) < 3:
                logger.warning(
                    f"Small sample sizes: group1={len(group1)}, group2={len(group2)}"
                )

            # Check for normality (optional warning)
            try:
                _, normality_p1 = stats.shapiro(group1)
                _, normality_p2 = stats.shapiro(group2)
                if normality_p1 < 0.05 or normality_p2 < 0.05:
                    logger.warning(
                        "Data may not be normally distributed, consider non-parametric tests"
                    )
            except Exception as e:
                logger.debug(f"Normality test failed: {e}")

            t_stat, p_value = stats.ttest_ind(group1, group2)

            # Handle NaN results
            if np.isnan(t_stat) or np.isnan(p_value):
                return {"error": "Statistical test produced NaN results"}

            if self.direction == "greater":
                falsified = t_stat < self.threshold and p_value < self.alpha
            elif self.direction == "less":
                falsified = t_stat > self.threshold and p_value < self.alpha
            else:
                falsified = abs(t_stat) > self.threshold and p_value < self.alpha

            logger.info(
                f"Criterion {self.name}: t={t_stat:.3f}, p={p_value:.3f}, falsified={falsified}"
            )

            return {
                "test_statistic": float(t_stat),
                "p_value": float(p_value),
                "falsified": bool(falsified),
                "threshold": self.threshold,
                "direction": self.direction,
                "sample_sizes": {"group1": len(group1), "group2": len(group2)},
                "means": {
                    "group1": float(np.mean(group1)),
                    "group2": float(np.mean(group2)),
                },
            }

        except Exception as e:
            logger.error(
                f"Error in mean difference test for criterion {self.name}: {e}"
            )
            return {"error": f"Test execution failed: {str(e)}"}

    def _test_correlation(self, data: Dict) -> Dict:
        """Test correlation strength"""
        try:
            x = data.get("x", [])
            y = data.get("y", [])

            # Input validation
            if not isinstance(x, (list, np.ndarray)) or not isinstance(
                y, (list, np.ndarray)
            ):
                return {"error": "Data must be lists or numpy arrays"}

            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            if len(x) == 0 or len(y) == 0:
                return {"error": "Insufficient data for correlation test"}

            if len(x) != len(y):
                return {
                    "error": f"Arrays must have same length: x={len(x)}, y={len(y)}"
                }

            if len(x) < 3:
                logger.warning(f"Very small sample size for correlation: n={len(x)}")

            # Check for constant values
            if np.std(x) == 0 or np.std(y) == 0:
                return {"error": "Cannot compute correlation with constant values"}

            corr, p_value = stats.pearsonr(x, y)

            # Handle NaN results
            if np.isnan(corr) or np.isnan(p_value):
                return {"error": "Correlation test produced NaN results"}

            if self.direction == "greater":
                falsified = corr < self.threshold and p_value < self.alpha
            elif self.direction == "less":
                falsified = corr > self.threshold and p_value < self.alpha
            else:
                falsified = abs(corr) < self.threshold and p_value < self.alpha

            logger.info(
                f"Criterion {self.name}: r={corr:.3f}, p={p_value:.3f}, falsified={falsified}"
            )

            return {
                "test_statistic": float(corr),
                "p_value": float(p_value),
                "falsified": bool(falsified),
                "threshold": self.threshold,
                "direction": self.direction,
                "sample_size": len(x),
            }

        except Exception as e:
            logger.error(f"Error in correlation test for criterion {self.name}: {e}")
            return {"error": f"Test execution failed: {str(e)}"}

    def _test_model_comparison(self, data: Dict) -> Dict:
        """Test model comparison (e.g., AIC/BIC difference)"""
        try:
            model1_fit = data.get("model1", {})
            model2_fit = data.get("model2", {})

            # Input validation
            if not isinstance(model1_fit, dict) or not isinstance(model2_fit, dict):
                return {"error": "Model fits must be dictionaries"}

            if not model1_fit or not model2_fit:
                return {"error": "Model comparison requires both models"}

            # Support both AIC and BIC
            aic1 = model1_fit.get("aic")
            aic2 = model2_fit.get("aic")
            bic1 = model1_fit.get("bic")
            bic2 = model2_fit.get("bic")

            if aic1 is None and bic1 is None:
                return {"error": "Model1 must contain 'aic' or 'bic' key"}
            if aic2 is None and bic2 is None:
                return {"error": "Model2 must contain 'aic' or 'bic' key"}

            # Prefer AIC, fall back to BIC
            if aic1 is not None and aic2 is not None:
                criterion_name = "AIC"
                crit1, crit2 = aic1, aic2
            elif bic1 is not None and bic2 is not None:
                criterion_name = "BIC"
                crit1, crit2 = bic1, bic2
            else:
                return {"error": "Both models must use the same information criterion"}

            # Validate criterion values
            if not isinstance(crit1, (int, float)) or not isinstance(
                crit2, (int, float)
            ):
                return {"error": f"{criterion_name} values must be numeric"}

            aic_diff = crit2 - crit1  # Positive means model2 is better

            # Check for convergence issues
            if np.isinf(crit1) or np.isinf(crit2):
                logger.warning(f"Infinite {criterion_name} values detected")

            if self.direction == "greater":
                falsified = aic_diff < self.threshold
            elif self.direction == "less":
                falsified = aic_diff > self.threshold
            else:
                falsified = abs(aic_diff) < self.threshold

            logger.info(
                f"Criterion {self.name}: {criterion_name} diff={aic_diff:.1f}, falsified={falsified}"
            )

            return {
                "test_statistic": float(aic_diff),
                "p_value": None,  # Model comparison doesn't use p-values
                "falsified": bool(falsified),
                "threshold": self.threshold,
                "direction": self.direction,
                "criterion": criterion_name,
                f"{criterion_name.lower()}_difference": float(
                    aic_diff
                ),  # Keep for backward compatibility
                f"model1_{criterion_name.lower()}": float(crit1),
                f"model2_{criterion_name.lower()}": float(crit2),
            }

        except Exception as e:
            logger.error(
                f"Error in model comparison test for criterion {self.name}: {e}"
            )
            return {"error": f"Test execution failed: {str(e)}"}

    def _test_effect_size(self, data: Dict) -> Dict:
        """Test effect size magnitude"""
        try:
            effect_size = data.get("effect_size")

            # Input validation
            if effect_size is None:
                return {"error": "Effect size data is required"}

            if not isinstance(effect_size, (int, float)):
                return {"error": "Effect size must be a numeric value"}

            # Check for reasonable effect size values
            if not (-10 <= effect_size <= 10):
                logger.warning(f"Effect size {effect_size} seems unusually large")

            if self.direction == "greater":
                falsified = effect_size < self.threshold
            elif self.direction == "less":
                falsified = effect_size > self.threshold
            else:
                falsified = abs(effect_size) < self.threshold

            logger.info(
                f"Criterion {self.name}: effect_size={effect_size:.3f}, falsified={falsified}"
            )

            return {
                "test_statistic": float(effect_size),
                "p_value": None,  # Effect size test doesn't use p-values
                "falsified": bool(falsified),
                "threshold": self.threshold,
                "direction": self.direction,
                "effect_size": float(effect_size),  # Keep for backward compatibility
            }

        except Exception as e:
            logger.error(f"Error in effect size test for criterion {self.name}: {e}")
            return {"error": f"Test execution failed: {str(e)}"}


class APGIFalsificationProtocol:
    """Complete falsification protocol for APGI theory"""

    def __init__(self):
        self.falsification_criteria = self._initialize_criteria()

    def _initialize_criteria(self) -> Dict[str, List[FalsificationCriterion]]:
        """Initialize falsification criteria for each priority"""

        return {
            "priority_1_neural_signatures": [
                FalsificationCriterion(
                    name="p3b_linear_better",
                    description="P3b amplitude fits linear model better than sigmoidal",
                    test_statistic="model_comparison",
                    threshold=-10,  # AIC difference favoring linear
                    direction="greater",
                ),
                FalsificationCriterion(
                    name="no_frontoparietal_contingency",
                    description="Frontoparietal activation occurs subthreshold",
                    test_statistic="mean_difference",
                    threshold=0.1,  # Effect size threshold
                    direction="greater",
                ),
                FalsificationCriterion(
                    name="absent_gamma_coupling",
                    description="No theta-gamma coupling at threshold crossing",
                    test_statistic="correlation",
                    threshold=0.3,  # Correlation threshold
                    direction="less",
                ),
            ],
            "priority_2_causal_manipulations": [
                FalsificationCriterion(
                    name="tms_no_timing_effect",
                    description="TMS disrupts ignition equally across all timings",
                    test_statistic="mean_difference",
                    threshold=0.05,  # Small effect size
                    direction="less",
                ),
                FalsificationCriterion(
                    name="pharmacology_affects_early_erp",
                    description="Pharmacological manipulation affects N1/P2 amplitude",
                    test_statistic="effect_size",
                    threshold=0.2,  # Effect size threshold
                    direction="greater",
                ),
                FalsificationCriterion(
                    name="metabolic_no_threshold_shift",
                    description="Metabolic challenge does not elevate detection thresholds",
                    test_statistic="mean_difference",
                    threshold=0.1,  # Threshold shift threshold
                    direction="less",
                ),
            ],
            "priority_3_quantitative_fits": [
                FalsificationCriterion(
                    name="no_phase_transition",
                    description="Psychometric function shows no phase transition (β < 5)",
                    test_statistic="effect_size",
                    threshold=5,  # Beta threshold
                    direction="less",
                ),
                FalsificationCriterion(
                    name="lnn_fails_paradigm",
                    description="Spiking LNN cannot reproduce consciousness paradigms",
                    test_statistic="model_comparison",
                    threshold=-20,  # Large AIC disadvantage
                    direction="greater",
                ),
                FalsificationCriterion(
                    name="bayesian_no_convergence",
                    description="Bayesian estimation shows no parameter convergence",
                    test_statistic="effect_size",
                    threshold=0.1,  # R-hat threshold
                    direction="greater",
                ),
            ],
            "priority_4_clinical_convergence": [
                FalsificationCriterion(
                    name="vegetative_normal_p3b",
                    description="Vegetative state patients show normal P3b amplitudes",
                    test_statistic="mean_difference",
                    threshold=-0.5,  # Effect size favoring normal
                    direction="greater",
                ),
                FalsificationCriterion(
                    name="psychiatric_no_differentiation",
                    description="APGI cannot differentiate psychiatric disorders",
                    test_statistic="effect_size",
                    threshold=0.6,  # Classification accuracy threshold
                    direction="less",
                ),
                FalsificationCriterion(
                    name="no_species_homology",
                    description="No conserved APGI relationships across species",
                    test_statistic="correlation",
                    threshold=0.2,  # Correlation threshold
                    direction="less",
                ),
            ],
        }

    def test_priority_falsification(self, priority: str, test_data: Dict) -> Dict:
        """
        Test all falsification criteria for a priority

        Args:
            priority: Priority name (e.g., 'priority_1_neural_signatures')
            test_data: Dictionary containing test data for each criterion

        Returns:
            Falsification test results
        """
        try:
            logger.info(f"Starting falsification testing for priority: {priority}")

            if priority not in self.falsification_criteria:
                logger.error(f"Unknown priority requested: {priority}")
                return {"error": f"Unknown priority: {priority}"}

            if not isinstance(test_data, dict):
                logger.error("Test data must be a dictionary")
                return {"error": "Test data must be a dictionary"}

            results = {
                "priority": priority,
                "criteria_results": [],
                "overall_falsified": False,
                "falsification_strength": 0.0,
            }

            criteria_list = self.falsification_criteria[priority]
            falsified_count = 0
            total_tests = len(criteria_list)

            logger.debug(f"Testing {total_tests} criteria for priority {priority}")

            for criterion in criteria_list:
                criterion_data = test_data.get(criterion.name, {})

                if not criterion_data:
                    logger.warning(
                        f"No test data provided for criterion: {criterion.name}"
                    )
                    result = {
                        "criterion": criterion.name,
                        "error": "No test data provided",
                    }
                else:
                    try:
                        result = criterion.test(criterion_data)
                        result["criterion"] = criterion.name
                        result["description"] = criterion.description

                        if result.get("falsified", False):
                            falsified_count += 1
                            logger.warning(
                                f"Criterion {criterion.name} falsified the theory"
                            )

                    except Exception as e:
                        logger.error(f"Failed to test criterion {criterion.name}: {e}")
                        result = {
                            "criterion": criterion.name,
                            "error": f"Test execution failed: {str(e)}",
                        }

                results["criteria_results"].append(result)

            # Determine overall falsification
            results["falsified_criteria"] = falsified_count
            results["total_criteria"] = total_tests
            results["falsification_rate"] = (
                falsified_count / total_tests if total_tests > 0 else 0
            )

            # Theory is falsified if ANY criterion is met (strict Popperian)
            results["overall_falsified"] = falsified_count > 0

            # Falsification strength (0-1, higher = stronger evidence against theory)
            results["falsification_strength"] = (
                falsified_count / total_tests if total_tests > 0 else 0
            )

            logger.info(
                f"Priority {priority}: {falsified_count}/{total_tests} criteria falsified"
            )
            if results["overall_falsified"]:
                logger.warning(f"Priority {priority} falsifies the theory")

            return results

        except Exception as e:
            logger.error(f"Critical error in priority falsification testing: {e}")
            return {"error": f"Priority testing failed: {str(e)}"}

    def run_comprehensive_falsification(self, all_test_data: Dict) -> Dict:
        """
        Run falsification tests across all priorities

        Args:
            all_test_data: Dictionary with test data for all priorities

        Returns:
            Comprehensive falsification results
        """
        try:
            logger.info(
                "Starting comprehensive falsification testing across all priorities"
            )

            if not isinstance(all_test_data, dict):
                logger.error("All test data must be a dictionary")
                return {"error": "Test data must be a dictionary"}

            comprehensive_results = {
                "priority_results": [],
                "overall_falsification": False,
                "falsification_summary": {},
                "theory_status": "supported",  # Default
            }

            total_falsified = 0
            total_criteria = 0
            priorities_tested = 0

            for priority in self.falsification_criteria.keys():
                priority_data = all_test_data.get(priority, {})

                try:
                    result = self.test_priority_falsification(priority, priority_data)
                    comprehensive_results["priority_results"].append(result)

                    if "error" not in result:
                        total_falsified += result.get("falsified_criteria", 0)
                        total_criteria += result.get("total_criteria", 0)
                        priorities_tested += 1
                    else:
                        logger.error(
                            f"Failed to test priority {priority}: {result['error']}"
                        )

                except Exception as e:
                    logger.error(
                        f"Exception during priority testing for {priority}: {e}"
                    )
                    error_result = {
                        "priority": priority,
                        "error": f"Priority testing failed: {str(e)}",
                    }
                    comprehensive_results["priority_results"].append(error_result)

            # Overall assessment
            if total_criteria > 0:
                comprehensive_results["total_falsified_criteria"] = total_falsified
                comprehensive_results["total_criteria"] = total_criteria
                comprehensive_results["overall_falsification_rate"] = (
                    total_falsified / total_criteria
                )

                # Theory status determination
                if total_falsified > 0:
                    comprehensive_results["overall_falsification"] = True
                    if total_falsified / total_criteria > 0.5:
                        comprehensive_results["theory_status"] = "strongly_falsified"
                    else:
                        comprehensive_results["theory_status"] = "weakly_falsified"
                else:
                    comprehensive_results["theory_status"] = "supported"
            else:
                logger.warning("No criteria were successfully tested")
                comprehensive_results["total_falsified_criteria"] = 0
                comprehensive_results["total_criteria"] = 0
                comprehensive_results["overall_falsification_rate"] = 0.0
                comprehensive_results["theory_status"] = "not_tested"

            # Summary by priority
            comprehensive_results["falsification_summary"] = {
                priority_result.get("priority", "unknown"): {
                    "falsified": priority_result.get("falsified_criteria", 0),
                    "total": priority_result.get("total_criteria", 0),
                    "rate": priority_result.get("falsification_rate", 0),
                    "error": "error" in priority_result,
                }
                for priority_result in comprehensive_results["priority_results"]
            }

            logger.info(
                f"Comprehensive testing complete. Theory status: {comprehensive_results['theory_status']}"
            )
            logger.info(
                f"Overall falsification rate: {comprehensive_results.get('overall_falsification_rate', 0):.3f}"
            )

            return comprehensive_results

        except Exception as e:
            logger.error(f"Critical error in comprehensive falsification: {e}")
            return {"error": f"Comprehensive testing failed: {str(e)}"}


class RobustnessTestingFramework:
    """Robustness testing against alternative explanations"""

    def __init__(self):
        self.alternative_models = self._initialize_alternative_models()

    def _initialize_alternative_models(self) -> Dict:
        """Initialize alternative models for comparison"""

        return {
            "gnw_global_neural_workspace": {
                "description": "Global Neural Workspace theory",
                "key_predictions": [
                    "Late P3b component indicates conscious access",
                    "Frontoparietal network ignition",
                    "All-or-none phenomenology",
                ],
                "discriminating_tests": [
                    "Precision weighting absent in GNW",
                    "Metabolic allostasis not predicted",
                    "Dynamic threshold not specified",
                ],
            },
            "iit_integrated_information": {
                "description": "Integrated Information Theory",
                "key_predictions": [
                    "Consciousness = integrated information (Φ)",
                    "Substrate-independent",
                    "Causal structure determines experience",
                ],
                "discriminating_tests": [
                    "Φ computation vs ignition event",
                    "Metabolic dependence not predicted",
                    "Algorithmic vs implementational level",
                ],
            },
            "predictive_processing": {
                "description": "Generic predictive processing",
                "key_predictions": [
                    "Hierarchical prediction and error minimization",
                    "Precision weighting for attention",
                    "Homeostatic regulation",
                ],
                "discriminating_tests": [
                    "Discrete ignition vs continuous minimization",
                    "Allostatic threshold not specified",
                    "Phase transition dynamics absent",
                ],
            },
            "attention_schema": {
                "description": "Attention Schema theory",
                "key_predictions": [
                    "Consciousness as attention simulator",
                    "Evolutionary adaptation",
                    "Computational model of awareness",
                ],
                "discriminating_tests": [
                    "No precision-weighted ignition",
                    "No metabolic allostasis",
                    "Different causal structure",
                ],
            },
        }

    def test_model_discrimination(self, empirical_data: Dict) -> Dict:
        """
        Test ability to discriminate APGI from alternative models

        Args:
            empirical_data: Dictionary with empirical test results

        Returns:
            Model discrimination results
        """

        discrimination_results = {}

        for model_name, model_info in self.alternative_models.items():
            discrimination_results[model_name] = {
                "model_info": model_info,
                "discriminating_power": self._assess_discriminating_power(
                    model_name, empirical_data
                ),
                "apgi_superior": False,  # To be determined
            }

        # Overall assessment
        apgi_discriminating_tests = [
            "p3b_sigmoidal_vs_linear",
            "metabolic_threshold_elevation",
            "phase_transition_dynamics",
            "precision_expectation_gap_anxiety",
        ]

        successful_discriminations = sum(
            1
            for test in apgi_discriminating_tests
            if empirical_data.get(test, {}).get("passed", False)
        )

        discrimination_results["overall_assessment"] = {
            "apgi_unique_predictions_tested": len(apgi_discriminating_tests),
            "apgi_unique_predictions_passed": successful_discriminations,
            "discrimination_success_rate": successful_discriminations
            / len(apgi_discriminating_tests),
            "theory_discriminated": successful_discriminations
            > len(apgi_discriminating_tests) * 0.7,
        }

        return discrimination_results

    def _assess_discriminating_power(
        self, model_name: str, empirical_data: Dict
    ) -> Dict:
        """Assess how well data discriminates APGI from alternative"""

        # Check if APGI makes unique predictions that alternatives don't
        unique_predictions = {
            "precision_expectation_gap_anxiety": empirical_data.get(
                "precision_expectation_gap_anxiety", {}
            ).get("passed", False),
            "metabolic_allostasis": empirical_data.get(
                "metabolic_threshold_elevation", {}
            ).get("passed", False),
            "phase_transition": empirical_data.get("phase_transition_dynamics", {}).get(
                "passed", False
            ),
            "discrete_ignition": empirical_data.get("discrete_ignition_events", {}).get(
                "passed", False
            ),
        }

        discriminating_power = sum(unique_predictions.values()) / len(
            unique_predictions
        )

        return {
            "unique_predictions_tested": unique_predictions,
            "discriminating_power": discriminating_power,
            "discriminates_from_alternative": discriminating_power > 0.5,
        }


class PopperianFalsificationFramework:
    """Popperian scientific methodology implementation"""

    def __init__(self):
        self.falsification_protocol = APGIFalsificationProtocol()
        self.robustness_framework = RobustnessTestingFramework()

    def conduct_falsification_test(self, empirical_results: Dict) -> Dict:
        """
        Conduct comprehensive Popperian falsification test

        Args:
            empirical_results: Results from all validation protocols

        Returns:
            Complete falsification assessment
        """

        # Extract test data for falsification
        test_data = self._extract_test_data(empirical_results)

        # Run falsification tests
        falsification_results = (
            self.falsification_protocol.run_comprehensive_falsification(test_data)
        )

        # Test robustness against alternatives
        discrimination_results = self.robustness_framework.test_model_discrimination(
            empirical_results
        )

        # Overall scientific assessment
        scientific_assessment = self._assess_scientific_status(
            falsification_results, discrimination_results
        )

        return {
            "falsification_results": falsification_results,
            "discrimination_results": discrimination_results,
            "scientific_assessment": scientific_assessment,
            "methodology": "popperian_falsification",
        }

    def _extract_test_data(self, empirical_results: Dict) -> Dict:
        """Extract falsification-relevant data from empirical results"""

        test_data = {}

        # Priority 1: Neural signatures
        test_data["priority_1_neural_signatures"] = {
            "p3b_linear_better": {
                "model1": empirical_results.get("p3b_sigmoidal_fit", {}),
                "model2": empirical_results.get("p3b_linear_fit", {}),
            },
            "no_frontoparietal_contingency": {
                "group1": empirical_results.get("suprathreshold_activation", []),
                "group2": empirical_results.get("subthreshold_activation", []),
            },
            "absent_gamma_coupling": {
                "x": empirical_results.get("gamma_power", []),
                "y": empirical_results.get("ignition_probability", []),
            },
        }

        # Priority 2: Causal manipulations
        test_data["priority_2_causal_manipulations"] = {
            "tms_no_timing_effect": {
                "group1": empirical_results.get("tms_ignition_window", []),
                "group2": empirical_results.get("tms_control_window", []),
            },
            "pharmacology_affects_early_erp": {
                "effect_size": empirical_results.get("pharmacology_early_erp_effect", 0)
            },
            "metabolic_no_threshold_shift": {
                "group1": empirical_results.get("metabolic_baseline_threshold", []),
                "group2": empirical_results.get("metabolic_elevated_threshold", []),
            },
        }

        # Priority 3: Quantitative fits
        test_data["priority_3_quantitative_fits"] = {
            "no_phase_transition": {
                "effect_size": empirical_results.get("psychometric_beta", 0)
            },
            "lnn_fails_paradigm": {
                "model1": empirical_results.get("apgi_lnn_fit", {}),
                "model2": empirical_results.get("alternative_fit", {}),
            },
            "bayesian_no_convergence": {
                "effect_size": empirical_results.get("bayesian_rhat", 1.0)
            },
        }

        # Priority 4: Clinical convergence
        test_data["priority_4_clinical_convergence"] = {
            "vegetative_normal_p3b": {
                "group1": empirical_results.get("vegetative_p3b", []),
                "group2": empirical_results.get("healthy_p3b", []),
            },
            "psychiatric_no_differentiation": {
                "effect_size": empirical_results.get(
                    "psychiatric_classification_accuracy", 0
                )
            },
            "no_species_homology": {
                "x": empirical_results.get("species_brain_size", []),
                "y": empirical_results.get("species_ignition_latency", []),
            },
        }

        return test_data

    def _assess_scientific_status(
        self, falsification_results: Dict, discrimination_results: Dict
    ) -> Dict:
        """Assess overall scientific status using Popperian criteria"""

        # Corroboration vs falsification
        falsification_rate = falsification_results.get("overall_falsification_rate", 0)
        discrimination_success = discrimination_results.get(
            "overall_assessment", {}
        ).get("theory_discriminated", False)

        # Scientific status determination
        if falsification_results.get("overall_falsification", False):
            if falsification_rate > 0.5:
                status = "falsified"
                confidence = "high"
            else:
                status = "provisionally_falsified"
                confidence = "moderate"
        elif discrimination_success:
            status = "corroborated"
            confidence = "high"
        else:
            status = "not_yet_tested"
            confidence = "low"

        # Detailed assessment
        assessment = {
            "scientific_status": status,
            "confidence_level": confidence,
            "falsification_rate": falsification_rate,
            "discrimination_success": discrimination_success,
            "testability_score": 1 - falsification_rate,  # How well tested
            "corroboration_score": discrimination_results.get(
                "overall_assessment", {}
            ).get("discrimination_success_rate", 0),
            "recommendations": self._generate_recommendations(
                status, falsification_rate
            ),
        }

        return assessment

    def _generate_recommendations(
        self, status: str, falsification_rate: float
    ) -> List[str]:
        """Generate recommendations based on falsification results"""

        recommendations = []

        if status == "falsified":
            recommendations.extend(
                [
                    "Revise or abandon APGI theory based on failed predictions",
                    "Identify which core assumptions led to falsification",
                    "Consider alternative theoretical frameworks",
                    "Design new theory that avoids falsified predictions",
                ]
            )
        elif status == "provisionally_falsified":
            recommendations.extend(
                [
                    "Conduct replication studies on falsified predictions",
                    "Check for methodological artifacts or confounds",
                    "Refine theory to accommodate problematic findings",
                    "Design more stringent tests of remaining predictions",
                ]
            )
        elif status == "corroborated":
            recommendations.extend(
                [
                    "Continue validation with additional predictions",
                    "Extend theory to new domains",
                    "Compare with alternative theories more rigorously",
                    "Develop applications based on corroborated predictions",
                ]
            )
        else:  # not_yet_tested
            recommendations.extend(
                [
                    "Design and conduct proper falsification tests",
                    "Ensure predictions are specific and testable",
                    "Improve experimental methodology",
                    "Increase statistical power of tests",
                ]
            )

        return recommendations


def main():
    """Demonstrate falsification testing framework"""

    # Initialize framework
    framework = PopperianFalsificationFramework()

    # Simulated empirical results (would come from actual validation protocols)
    simulated_results = {
        "p3b_sigmoidal_vs_linear": {"passed": True},
        "metabolic_threshold_elevation": {"passed": True},
        "phase_transition_dynamics": {"passed": True},
        "precision_expectation_gap_anxiety": {"passed": True},
        "discrete_ignition_events": {"passed": True},
        # Include some test data
        "p3b_sigmoidal_fit": {"aic": -100},
        "p3b_linear_fit": {"aic": -80},
        "suprathreshold_activation": [0.8, 0.9, 0.7],
        "subthreshold_activation": [0.2, 0.1, 0.3],
        "psychometric_beta": 12.0,
        "apgi_lnn_fit": {"aic": -150},
        "alternative_fit": {"aic": -120},
        "vegetative_p3b": [0.1, 0.2, 0.15],
        "healthy_p3b": [0.9, 0.8, 0.85],
    }

    # Run comprehensive falsification test
    falsification_assessment = framework.conduct_falsification_test(simulated_results)

    print("APGI Falsification Testing Framework Results")
    print("=" * 50)
    print(
        f"Scientific Status: {falsification_assessment['scientific_assessment']['scientific_status']}"
    )
    print(
        f"Confidence Level: {falsification_assessment['scientific_assessment']['confidence_level']}"
    )
    print(
        f"Testability Score: {falsification_assessment['scientific_assessment']['testability_score']:.3f}"
    )

    print("\nFalsification Results by Priority:")
    for priority_result in falsification_assessment["falsification_results"][
        "priority_results"
    ]:
        print(
            f"  {priority_result['priority']}: {priority_result['falsified_criteria']}/{priority_result['total_criteria']} falsified"
        )

    print("\nRecommendations:")
    for rec in falsification_assessment["scientific_assessment"]["recommendations"]:
        print(f"  • {rec}")


if __name__ == "__main__":
    main()
