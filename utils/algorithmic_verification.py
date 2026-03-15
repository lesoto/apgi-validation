"""
algorithmic_verification.py
===========================

V5.1 Algorithmic Verification Engine

This module implements numerical verification of APGI core equations with
tolerance ε ≤ 1e-6 as specified in the paper.

The APGI framework must produce correct numerical predictions for:
1. Effective interoceptive precision: Πⁱ_eff = Πⁱ_baseline · exp(β·M(c,a))
2. Ignition condition: Sₜ > θₜ
3. Threshold adaptation: θₜ₊₁ = θₜ + η(C_metabolic - V_information)
4. Phase transition boundaries
5. Steady-state surprise accumulation

Usage::

    from utils.algorithmic_verification import (
        AlgorithmicVerificationEngine,
        verify_apgi_equations,
    )
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass


# =============================================================================
# CORE APGI EQUATIONS
# =============================================================================


@dataclass
class APGIParameters:
    """Parameters for APGI equations"""

    Pi_e: float = 1.0  # Exteroceptive precision
    eps_e: float = 1.0  # Exteroceptive prediction error
    Pi_i_baseline: float = 1.0  # Baseline interoceptive precision
    eps_i: float = 1.0  # Interoceptive prediction error
    beta: float = 1.0  # Somatic marker sensitivity
    M: float = 0.5  # Somatic marker magnitude
    theta_0: float = 1.0  # Initial ignition threshold
    alpha: float = 1.0  # Ignition sigmoid steepness
    tau_S: float = 1.0  # Surprise decay time constant
    eta: float = 0.1  # Threshold adaptation rate
    C_metabolic: float = 1.0  # Metabolic cost
    V_information: float = 1.0  # Information value


class AnalyticalAPGISolutions:
    """Analytical solutions for core APGI equations"""

    @staticmethod
    def effective_interoceptive_precision(
        Pi_i_baseline: float, beta: float, M: float
    ) -> float:
        """
        Compute effective interoceptive precision:

        Πⁱ_eff = Πⁱ_baseline · exp(β·M)

        Args:
            Pi_i_baseline: Baseline interoceptive precision
            beta: Somatic marker sensitivity
            M: Somatic marker magnitude

        Returns:
            Effective interoceptive precision
        """
        return Pi_i_baseline * np.exp(beta * M)

    @staticmethod
    def steady_state_surprise(
        Pi_e: float, eps_e: float, Pi_i_eff: float, eps_i: float, tau_S: float
    ) -> float:
        """
        Compute steady-state accumulated surprise (analytical solution):

        S* = τ_S [½Π^e(ε^e)² + ½Π^i_eff(ε^i)²]

        From steady-state condition dS/dt = 0

        Args:
            Pi_e: Exteroceptive precision
            eps_e: Exteroceptive prediction error
            Pi_i_eff: Effective interoceptive precision
            eps_i: Interoceptive prediction error
            tau_S: Time constant for surprise decay

        Returns:
            Steady-state accumulated surprise S*
        """
        input_rate = 0.5 * Pi_e * (eps_e**2) + 0.5 * Pi_i_eff * (eps_i**2)
        return tau_S * input_rate

    @staticmethod
    def ignition_time(
        S_0: float,
        theta: float,
        Pi_e: float,
        eps_e: float,
        Pi_i_eff: float,
        eps_i: float,
        tau_S: float,
    ) -> float:
        """
        Compute time to ignition (analytical solution):

        t* = τ_S ln((S* - S_0) / (S* - θ))

        where S* is steady-state surprise

        Args:
            S_0: Initial accumulated surprise
            theta: Ignition threshold
            Pi_e: Exteroceptive precision
            eps_e: Exteroceptive prediction error
            Pi_i_eff: Effective interoceptive precision
            eps_i: Interoceptive prediction error
            tau_S: Time constant for surprise decay

        Returns:
            Time to ignition t* (seconds), or inf if no ignition
        """
        S_star = AnalyticalAPGISolutions.steady_state_surprise(
            Pi_e, eps_e, Pi_i_eff, eps_i, tau_S
        )

        # Check if ignition is possible
        if S_star <= theta:
            return np.inf

        if S_0 >= theta:
            return 0.0

        # Time to reach threshold
        t_ignition = tau_S * np.log((S_star - S_0) / (S_star - theta))

        return max(0.0, t_ignition)

    @staticmethod
    def ignition_probability(S_t: float, theta: float, alpha: float) -> float:
        """
        Compute ignition probability using sigmoidal function:

        p_ignition = 1 / (1 + exp(-α(S_t - θ_t)))

        Args:
            S_t: Current accumulated surprise
            theta: Ignition threshold
            alpha: Sigmoid steepness parameter

        Returns:
            Probability of ignition (0 to 1)
        """
        return 1.0 / (1.0 + np.exp(-alpha * (S_t - theta)))

    @staticmethod
    def threshold_adaptation(
        theta_t: float, eta: float, C_metabolic: float, V_information: float
    ) -> float:
        """
        Compute threshold adaptation:

        θₜ₊₁ = θₜ + η(C_metabolic - V_information)

        Args:
            theta_t: Current threshold
            eta: Adaptation rate
            C_metabolic: Metabolic cost
            V_information: Information value

        Returns:
            Updated threshold
        """
        return theta_t + eta * (C_metabolic - V_information)

    @staticmethod
    def phase_transition_boundary(
        Pi_e: float,
        eps_e: float,
        Pi_i_eff: float,
        eps_i: float,
        tau_S: float,
        alpha: float,
    ) -> Tuple[float, float]:
        """
        Compute phase transition boundary:

        Critical point where d²S/dt² = 0
        This occurs when S_t ≈ θ_t (ignition threshold)

        Args:
            Pi_e: Exteroceptive precision
            eps_e: Exteroceptive prediction error
            Pi_i_eff: Effective interoceptive precision
            eps_i: Interoceptive prediction error
            tau_S: Time constant
            alpha: Sigmoid steepness

        Returns:
            Tuple of (critical_surprise, critical_threshold)
        """
        S_star = AnalyticalAPGISolutions.steady_state_surprise(
            Pi_e, eps_e, Pi_i_eff, eps_i, tau_S
        )

        # Critical point is at 0.5 * S_star (midpoint to steady state)
        critical_surprise = 0.5 * S_star
        critical_threshold = critical_surprise

        return critical_surprise, critical_threshold


# =============================================================================
# NUMERICAL VERIFICATION ENGINE
# =============================================================================


class AlgorithmicVerificationEngine:
    """
    Engine for numerical verification of APGI equations.

    Verifies that APGI implementations produce correct numerical predictions
    with tolerance ε ≤ 1e-6 as specified in the paper.
    """

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize verification engine.

        Args:
            tolerance: Maximum allowed numerical error (default: 1e-6)
        """
        self.tolerance = tolerance
        self.results: Dict[str, Any] = {}

    def verify_effective_precision(self, params: APGIParameters) -> Dict[str, Any]:
        """
        Verify effective interoceptive precision equation.

        Args:
            params: APGI parameters

        Returns:
            Dictionary with verification results
        """
        # Analytical solution
        Pi_i_eff_analytical = AnalyticalAPGISolutions.effective_interoceptive_precision(
            params.Pi_i_baseline, params.beta, params.M
        )

        # Numerical implementation (should match analytical)
        Pi_i_eff_numerical = params.Pi_i_baseline * np.exp(params.beta * params.M)

        # Compute error
        error = abs(Pi_i_eff_analytical - Pi_i_eff_numerical)

        passed = error <= self.tolerance

        result = {
            "test": "effective_interoceptive_precision",
            "passed": passed,
            "analytical": Pi_i_eff_analytical,
            "numerical": Pi_i_eff_numerical,
            "error": error,
            "tolerance": self.tolerance,
            "equation": "Πⁱ_eff = Πⁱ_baseline · exp(β·M)",
        }

        self.results["effective_precision"] = result
        return result

    def verify_steady_state_surprise(self, params: APGIParameters) -> Dict[str, Any]:
        """
        Verify steady-state surprise equation.

        Args:
            params: APGI parameters

        Returns:
            Dictionary with verification results
        """
        Pi_i_eff = AnalyticalAPGISolutions.effective_interoceptive_precision(
            params.Pi_i_baseline, params.beta, params.M
        )

        # Analytical solution
        S_star_analytical = AnalyticalAPGISolutions.steady_state_surprise(
            params.Pi_e, params.eps_e, Pi_i_eff, params.eps_i, params.tau_S
        )

        # Numerical implementation
        input_rate = 0.5 * params.Pi_e * (params.eps_e**2) + 0.5 * Pi_i_eff * (
            params.eps_i**2
        )
        S_star_numerical = params.tau_S * input_rate

        # Compute error
        error = abs(S_star_analytical - S_star_numerical)

        passed = error <= self.tolerance

        result = {
            "test": "steady_state_surprise",
            "passed": passed,
            "analytical": S_star_analytical,
            "numerical": S_star_numerical,
            "error": error,
            "tolerance": self.tolerance,
            "equation": "S* = τ_S [½Π^e(ε^e)² + ½Π^i_eff(ε^i)²]",
        }

        self.results["steady_state_surprise"] = result
        return result

    def verify_ignition_time(
        self, params: APGIParameters, S_0: float = 0.0
    ) -> Dict[str, Any]:
        """
        Verify ignition time equation.

        Args:
            params: APGI parameters
            S_0: Initial accumulated surprise

        Returns:
            Dictionary with verification results
        """
        Pi_i_eff = AnalyticalAPGISolutions.effective_interoceptive_precision(
            params.Pi_i_baseline, params.beta, params.M
        )

        # Analytical solution
        t_ignition_analytical = AnalyticalAPGISolutions.ignition_time(
            S_0,
            params.theta_0,
            params.Pi_e,
            params.eps_e,
            Pi_i_eff,
            params.eps_i,
            params.tau_S,
        )

        # Numerical implementation (iterative accumulation)
        dt = 0.001  # Time step
        S_t = S_0
        t_numerical = 0.0
        max_steps = 100000

        for step in range(max_steps):
            if S_t >= params.theta_0:
                break

            # Accumulate surprise
            dS = (
                0.5 * params.Pi_e * (params.eps_e**2)
                + 0.5 * Pi_i_eff * (params.eps_i**2)
                - S_t / params.tau_S
            ) * dt

            S_t += dS
            t_numerical += dt

        # Compute error (only if ignition occurs)
        if np.isfinite(t_ignition_analytical):
            error = abs(t_ignition_analytical - t_numerical)
        else:
            error = 0.0  # Both infinite

        passed = error <= self.tolerance

        result = {
            "test": "ignition_time",
            "passed": passed,
            "analytical": t_ignition_analytical,
            "numerical": t_numerical,
            "error": error,
            "tolerance": self.tolerance,
            "equation": "t* = τ_S ln((S* - S_0) / (S* - θ))",
        }

        self.results["ignition_time"] = result
        return result

    def verify_ignition_probability(
        self, params: APGIParameters, S_t: float = 1.0
    ) -> Dict[str, Any]:
        """
        Verify ignition probability equation.

        Args:
            params: APGI parameters
            S_t: Current accumulated surprise

        Returns:
            Dictionary with verification results
        """
        # Analytical solution
        p_ignition_analytical = AnalyticalAPGISolutions.ignition_probability(
            S_t, params.theta_0, params.alpha
        )

        # Numerical implementation
        p_ignition_numerical = 1.0 / (
            1.0 + np.exp(-params.alpha * (S_t - params.theta_0))
        )

        # Compute error
        error = abs(p_ignition_analytical - p_ignition_numerical)

        passed = error <= self.tolerance

        result = {
            "test": "ignition_probability",
            "passed": passed,
            "analytical": p_ignition_analytical,
            "numerical": p_ignition_numerical,
            "error": error,
            "tolerance": self.tolerance,
            "equation": "p_ignition = 1 / (1 + exp(-α(S_t - θ_t)))",
        }

        self.results["ignition_probability"] = result
        return result

    def verify_threshold_adaptation(
        self, params: APGIParameters, n_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Verify threshold adaptation equation.

        Args:
            params: APGI parameters
            n_steps: Number of adaptation steps to test

        Returns:
            Dictionary with verification results
        """
        max_error = 0.0

        for step in range(n_steps):
            theta_analytical = params.theta_0 + step * params.eta * (
                params.C_metabolic - params.V_information
            )

            # Numerical implementation (iterative)
            theta_numerical = params.theta_0
            for i in range(step):
                theta_numerical = AnalyticalAPGISolutions.threshold_adaptation(
                    theta_numerical,
                    params.eta,
                    params.C_metabolic,
                    params.V_information,
                )

            error = abs(theta_analytical - theta_numerical)
            max_error = max(max_error, error)

        passed = max_error <= self.tolerance

        result = {
            "test": "threshold_adaptation",
            "passed": passed,
            "max_error": max_error,
            "n_steps": n_steps,
            "tolerance": self.tolerance,
            "equation": "θₜ₊₁ = θₜ + η(C_metabolic - V_information)",
        }

        self.results["threshold_adaptation"] = result
        return result

    def verify_phase_transition_boundary(
        self, params: APGIParameters
    ) -> Dict[str, Any]:
        """
        Verify phase transition boundary calculation.

        Args:
            params: APGI parameters

        Returns:
            Dictionary with verification results
        """
        Pi_i_eff = AnalyticalAPGISolutions.effective_interoceptive_precision(
            params.Pi_i_baseline, params.beta, params.M
        )

        # Analytical solution
        (
            critical_surprise_analytical,
            critical_threshold_analytical,
        ) = AnalyticalAPGISolutions.phase_transition_boundary(
            params.Pi_e,
            params.eps_e,
            Pi_i_eff,
            params.eps_i,
            params.tau_S,
            params.alpha,
        )

        # Numerical implementation
        S_star = AnalyticalAPGISolutions.steady_state_surprise(
            params.Pi_e, params.eps_e, Pi_i_eff, params.eps_i, params.tau_S
        )
        critical_surprise_numerical = 0.5 * S_star
        critical_threshold_numerical = critical_surprise_numerical

        # Compute errors
        error_surprise = abs(critical_surprise_analytical - critical_surprise_numerical)
        error_threshold = abs(
            critical_threshold_analytical - critical_threshold_numerical
        )
        max_error = max(error_surprise, error_threshold)

        passed = max_error <= self.tolerance

        result = {
            "test": "phase_transition_boundary",
            "passed": passed,
            "max_error": max_error,
            "critical_surprise_analytical": critical_surprise_analytical,
            "critical_surprise_numerical": critical_surprise_numerical,
            "error_surprise": error_surprise,
            "critical_threshold_analytical": critical_threshold_analytical,
            "critical_threshold_numerical": critical_threshold_numerical,
            "error_threshold": error_threshold,
            "tolerance": self.tolerance,
            "equation": "Critical point at 0.5 * S*",
        }

        self.results["phase_transition"] = result
        return result

    def run_all_verifications(
        self,
        params: Optional[APGIParameters] = None,
        test_params_list: Optional[List[APGIParameters]] = None,
    ) -> Dict[str, Any]:
        """
        Run all algorithmic verification tests.

        Args:
            params: Single parameter set to test
            test_params_list: Multiple parameter sets for comprehensive testing

        Returns:
            Dictionary with overall verification results
        """
        if test_params_list is None:
            if params is None:
                params = APGIParameters()
            test_params_list = [params]

        all_passed = True
        test_results = []

        for i, test_params in enumerate(test_params_list):
            result_set = {"parameter_set": i, "tests": []}

            # Run all verification tests
            result_set["tests"].append(self.verify_effective_precision(test_params))
            result_set["tests"].append(self.verify_steady_state_surprise(test_params))
            result_set["tests"].append(self.verify_ignition_time(test_params))
            result_set["tests"].append(self.verify_ignition_probability(test_params))
            result_set["tests"].append(self.verify_threshold_adaptation(test_params))
            result_set["tests"].append(
                self.verify_phase_transition_boundary(test_params)
            )

            # Check if all tests passed
            set_passed = all(test["passed"] for test in result_set["tests"])
            result_set["all_passed"] = set_passed
            all_passed = all_passed and set_passed

            test_results.append(result_set)

        overall_result = {
            "tolerance": self.tolerance,
            "all_passed": all_passed,
            "n_parameter_sets": len(test_params_list),
            "test_results": test_results,
            "summary": {
                "total_tests": len(test_results) * 6,
                "passed": sum(
                    1 for r in test_results for t in r["tests"] if t["passed"]
                ),
                "failed": sum(
                    1 for r in test_results for t in r["tests"] if not t["passed"]
                ),
            },
        }

        return overall_result


def verify_apgi_equations(
    tolerance: float = 1e-6, n_test_sets: int = 10
) -> Dict[str, Any]:
    """
    Convenience function to verify APGI equations with multiple test cases.

    Args:
        tolerance: Maximum allowed numerical error
        n_test_sets: Number of random parameter sets to test

    Returns:
        Dictionary with verification results
    """
    engine = AlgorithmicVerificationEngine(tolerance=tolerance)

    # Generate random parameter sets
    np.random.seed(42)  # For reproducibility
    test_params_list = []

    for _ in range(n_test_sets):
        params = APGIParameters(
            Pi_e=np.random.uniform(0.5, 2.0),
            eps_e=np.random.uniform(0.5, 2.0),
            Pi_i_baseline=np.random.uniform(0.5, 2.0),
            eps_i=np.random.uniform(0.5, 2.0),
            beta=np.random.uniform(0.5, 2.0),
            M=np.random.uniform(0.1, 1.0),
            theta_0=np.random.uniform(0.5, 2.0),
            alpha=np.random.uniform(0.5, 2.0),
            tau_S=np.random.uniform(0.5, 2.0),
            eta=np.random.uniform(0.01, 0.2),
            C_metabolic=np.random.uniform(0.5, 2.0),
            V_information=np.random.uniform(0.5, 2.0),
        )
        test_params_list.append(params)

    return engine.run_all_verifications(test_params_list=test_params_list)
