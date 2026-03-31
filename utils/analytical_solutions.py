"""
APGI Analytical Solutions Utility
=================================

Provides exact analytical solutions for core APGI equations used for
mathematical consistency verification and protocol validation.
"""

import numpy as np


class AnalyticalAPGISolutions:
    """Analytical solutions for core APGI equations"""

    @staticmethod
    def steady_state_surprise(
        Pi_e: float,
        eps_e: float,
        Pi_i_eff: float,
        eps_i: float,
        tau_S: float,
    ) -> float:
        """
        Compute steady-state accumulated surprise (analytical solution):
        S* = τ_S [½Π^e(ε^e)² + ½Π^i_eff(ε^i)²]
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
        """
        S_star = AnalyticalAPGISolutions.steady_state_surprise(
            Pi_e, eps_e, Pi_i_eff, eps_i, tau_S
        )

        if theta >= S_star:
            return float("inf")
        if S_0 >= theta:
            return 0.0

        t_star = tau_S * np.log((S_star - S_0) / (S_star - theta))
        return max(0.0, t_star)

    @staticmethod
    def ignition_threshold_critical_point(
        Pi_e: float,
        eps_e: float,
        Pi_i_eff: float,
        eps_i: float,
        tau_S: float,
    ) -> float:
        """Compute critical threshold for ignition (analytical)"""
        return AnalyticalAPGISolutions.steady_state_surprise(
            Pi_e, eps_e, Pi_i_eff, eps_i, tau_S
        )

    @staticmethod
    def phase_boundary(
        theta: float,
        tau_S: float,
        Pi_e: float,
        eps_e: float,
        Pi_i_eff: float,
        eps_i: float,
    ) -> dict:
        """Compute phase boundary parameters (analytical)"""
        S_star = AnalyticalAPGISolutions.steady_state_surprise(
            Pi_e, eps_e, Pi_i_eff, eps_i, tau_S
        )
        return {
            "steady_state_surprise": S_star,
            "threshold": theta,
            "phase_gap": S_star - theta,
            "ignition_possible": theta < S_star,
        }

    @staticmethod
    def effective_interoceptive_precision_analytical(
        Pi_i_baseline: float,
        M: float,
        M_0: float,
        beta: float,
    ) -> float:
        """Compute effective interoceptive precision (analytical)"""
        return Pi_i_baseline * np.exp(beta * M)

    @staticmethod
    def ignition_probability_analytical(
        S: float,
        theta: float,
        alpha: float,
    ) -> float:
        """Compute ignition probability (analytical)"""
        z = alpha * (S - theta)
        if z >= 0:
            return 1.0 / (1.0 + np.exp(-z))
        else:
            z_exp = np.exp(z)
            return z_exp / (1.0 + z_exp)

    @staticmethod
    def critical_sigmoid_steepness(
        S: float,
        theta: float,
        target_probability: float = 0.5,
    ) -> float:
        """
        Compute critical sigmoid steepness for target probability:
        P = σ(α(S - θ)) → α = ln(P/(1-P)) / (S - θ)
        """
        if S == theta:
            return 0.0  # Any α gives P = 0.5 at S = θ

        if target_probability <= 0 or target_probability >= 1:
            raise ValueError("Target probability must be in (0, 1)")

        logit = np.log(target_probability / (1.0 - target_probability))
        return logit / (S - theta)
