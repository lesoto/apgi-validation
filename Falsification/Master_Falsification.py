#!/usr/bin/env python3
"""
APGI Master Falsification Orchestrator
=======================================

Coordinates execution of all falsification protocols and aggregates results.
Central registry for all falsification criteria (F1.x - F12.x).

This module provides:
- Unified interface to all 12 falsification protocols (FP-01 to FP-12)
- Centralized falsification criteria registry
- Tier-based protocol classification
- Framework-level falsification aggregation
- Named prediction tracking across all protocols
"""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to sys.path for imports
_proj_root = Path(__file__).parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

# Import the framework-level aggregator
from Falsification.FP_ALL_Aggregator import (NAMED_PREDICTIONS,
                                             FalsificationAggregator,
                                             run_framework_falsification)

# Try to import logging config
try:
    from utils.logging_config import APGILogger
    from utils.logging_config import apgi_logger as _logger
except ImportError:
    import logging

    _logger = logging.getLogger(__name__)  # type: ignore[no-redef,assignment]
    APGILogger = logging.Logger  # type: ignore[misc,assignment,no-redef]

logger = _logger  # type: ignore[assignment]


@dataclass
class FalsificationResults:
    """Results from a falsification protocol test.

    Attributes:
        protocol_id: Unique identifier for the protocol
        hypothesis_tested: Description of the hypothesis being tested
        p_value: Statistical p-value from the test
        effect_size: Effect size measure
        confidence_interval: Tuple of (lower, upper) confidence bounds
        falsified: Boolean indicating if the hypothesis was falsified
    """

    protocol_id: str
    hypothesis_tested: str
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    falsified: bool


@dataclass
class ProtocolResult:
    """Result from running a single protocol.

    Attributes:
        protocol_name: Name of the protocol
        status: Execution status (completed, failed, etc.)
        outcome: Outcome classification (success, failure, etc.)
        metrics: Dictionary of result metrics
    """

    protocol_name: str
    status: str
    outcome: str
    metrics: Dict[str, Any]


class APGIMasterFalsifier:
    """Orchestrates execution of all APGI falsification protocols.

    Coordinates 12 falsification protocols (FP-01 to FP-12) that test
    specific falsifiable predictions of the APGI framework. Each protocol
    contains multiple falsification criteria (F1.x - F12.x).

    Protocol Tier Classification:
    - Primary (FP-01, FP-02): Core architectural falsification
      Tests fundamental APGI properties that must hold for viability
    - Secondary (FP-03, FP-04, FP-08, FP-11, FP-12): Extended validation
      Tests important but not absolutely foundational properties
    - Tertiary (FP-05, FP-06, FP-07, FP-09, FP-10): Specialized tests
      Tests specific predictions and edge cases

    Framework-Level Falsification Conditions:
    - Condition A (FA): ALL 14 named predictions fail simultaneously
    - Condition B (FB): Alternative frameworks (GWT, IIT) are more parsimonious
    """

    def __init__(self) -> None:
        self.protocol_results: Dict[str, Any] = {}
        self.falsification_aggregator = FalsificationAggregator()

        # Protocol tier classification
        self.PROTOCOL_TIERS: Dict[int, str] = {
            1: "primary",  # Active Inference Agents (F1.x, F2.x)
            2: "primary",  # Agent Comparison / Convergence (F3.x)
            3: "secondary",  # Framework-Level Multi-Protocol
            4: "secondary",  # Phase Transition / Epistemic
            5: "tertiary",  # Evolutionary Plausibility
            6: "tertiary",  # Liquid Network / Energy Benchmark
            7: "tertiary",  # Mathematical Consistency
            8: "secondary",  # Parameter Sensitivity / Identifiability
            9: "tertiary",  # Neural Signatures (P3b, HEP)
            10: "tertiary",  # Bayesian Estimation / Cross-Species
            11: "secondary",  # Liquid Network Dynamics
            12: "secondary",  # Framework Aggregator
        }

        # Centralized falsification criteria registry
        # Each entry maps to the protocol that tests it
        self.FALSIFICATION_CRITERIA: Dict[str, Dict[str, Any]] = {
            # FP-01: Active Inference / F1 & F2
            "F1.1": {
                "protocol": 1,
                "description": "APGI Agent Performance Advantage",
                "threshold": "≥18% higher cumulative reward",
            },
            "F1.2": {
                "protocol": 1,
                "description": "Hierarchical Level Emergence",
                "threshold": "≥3 distinct temporal clusters",
            },
            "F1.3": {
                "protocol": 1,
                "description": "Level-Specific Precision Weighting",
                "threshold": "L1 precision 25-40% higher than L3",
            },
            "F1.4": {
                "protocol": 1,
                "description": "Threshold Adaptation Dynamics",
                "threshold": "τ_θ=10-100s, >20% reduction",
            },
            "F1.5": {
                "protocol": 1,
                "description": "Cross-Level PAC",
                "threshold": "MI ≥ 0.012, ≥30% increase during ignition",
            },
            "F1.6": {
                "protocol": 1,
                "description": "1/f Spectral Slope",
                "threshold": "α_spec=0.8-1.2 active, 1.5-2.0 low-arousal",
            },
            "F2.1": {
                "protocol": 1,
                "description": "Somatic Marker Advantage",
                "threshold": "≥22% higher selection advantageous decks",
            },
            "F2.2": {
                "protocol": 1,
                "description": "Interoceptive Cost Sensitivity",
                "threshold": "r = -0.45 to -0.65",
            },
            "F2.3": {
                "protocol": 1,
                "description": "vmPFC-like Anticipatory Bias",
                "threshold": "≥35ms faster RT, β_cost ≥ 25ms/unit",
            },
            "F2.4": {
                "protocol": 1,
                "description": "Precision-Weighted Integration",
                "threshold": "≥30% greater influence high-confidence",
            },
            "F2.5": {
                "protocol": 1,
                "description": "Learning Trajectory Discrimination",
                "threshold": "≤55 trials to 70% criterion",
            },
            # FP-02: Agent Comparison / F3.x
            "F3.1": {
                "protocol": 2,
                "description": "Overall Performance Advantage",
                "threshold": "≥18% over non-APGI baselines",
            },
            "F3.2": {
                "protocol": 2,
                "description": "Interoceptive Task Specificity",
                "threshold": "≥28% advantage in interoceptive tasks",
            },
            "F3.3": {
                "protocol": 2,
                "description": "Threshold Gating Necessity",
                "threshold": "≥25% reduction when removed",
            },
            "F3.4": {
                "protocol": 2,
                "description": "Precision Weighting Necessity",
                "threshold": "≥20% reduction with uniform precision",
            },
            "F3.5": {
                "protocol": 2,
                "description": "Computational Efficiency",
                "threshold": "≤1.5× compute vs. alternatives",
            },
            "F3.6": {
                "protocol": 2,
                "description": "Sample Efficiency",
                "threshold": "≤200 trials to 80% performance",
            },
            # FP-03: Framework-Level / Named Predictions
            "P3.conv": {
                "protocol": 3,
                "description": "Convergence 50-80 trials",
                "threshold": "50-80 trials to criterion",
            },
            "P3.bic": {
                "protocol": 3,
                "description": "APGI BIC advantage",
                "threshold": "APGI BIC < alternatives",
            },
            # FP-04: Phase Transition / Information-Theoretic
            "F4.1": {
                "protocol": 4,
                "description": "Ignition Threshold Sharpness",
                "threshold": "d' ≥ 2.0 at threshold",
            },
            "F4.2": {
                "protocol": 4,
                "description": "Hysteresis Width",
                "threshold": "0.08 ≤ H ≤ 0.25",
            },
            "F4.3": {
                "protocol": 4,
                "description": "Critical Slowing Down",
                "threshold": "τ increases ≥40% near threshold",
            },
            "F4.4": {
                "protocol": 4,
                "description": "Metabolic Cost Ceiling",
                "threshold": "≤20% above baseline ATP",
            },
            "F4.5": {
                "protocol": 4,
                "description": "Long-Range Correlations",
                "threshold": "Hurst H = 0.6-0.8",
            },
            "F4.6": {
                "protocol": 4,
                "description": "Bandwidth Constraint",
                "threshold": "≤40 bits/s (biological ceiling)",
            },
            # FP-05: Evolutionary Plausibility
            "F5.1": {
                "protocol": 5,
                "description": "Threshold Evolution Proportion",
                "threshold": "≥75% evolve θ mechanisms",
            },
            "F5.2": {
                "protocol": 5,
                "description": "Precision Weighting Emergence",
                "threshold": "≥70% evolve Πⁱ weighting",
            },
            "F5.3": {
                "protocol": 5,
                "description": "Interoceptive Gain Bias",
                "threshold": "≥60% show interoceptive bias",
            },
            "F5.4": {
                "protocol": 5,
                "description": "Multi-Timescale Integration",
                "threshold": "Peak separation ≥3.0 log units",
            },
            "F5.5": {
                "protocol": 5,
                "description": "PCA Clustering",
                "threshold": "≥70% variance, loadings ≥0.60",
            },
            "F5.6": {
                "protocol": 5,
                "description": "Non-APGI Performance Gap",
                "threshold": "≥15% worse without APGI components",
            },
            # FP-06: Liquid Network / Energy Benchmark
            "F6.1": {
                "protocol": 6,
                "description": "LTCN Fast Transition",
                "threshold": "<50ms ignition transition",
            },
            "F6.2": {
                "protocol": 6,
                "description": "LTCN Extended Window",
                "threshold": "200-500ms integration, ≥4× standard RNN",
            },
            "F6.3": {
                "protocol": 6,
                "description": "Sparsity Reduction",
                "threshold": "≥30% reduction in active units",
            },
            "F6.4": {
                "protocol": 6,
                "description": "Memory Decay Tau",
                "threshold": "τ between 50-500ms",
            },
            "F6.5": {
                "protocol": 6,
                "description": "Bifurcation/Hysteresis",
                "threshold": "0.08 ≤ H ≤ 0.25",
            },
            "F6.6": {
                "protocol": 6,
                "description": "RNN Add-On Requirement",
                "threshold": "≥2 add-ons needed",
            },
            # FP-07: Mathematical Consistency
            "F7.1": {
                "protocol": 7,
                "description": "Surprise Accumulation ODE",
                "threshold": "dS/dt = ε²/2 - Πⁱ·|εᵢ|·σ(·)",
            },
            "F7.2": {
                "protocol": 7,
                "description": "Ignition Sigmoid",
                "threshold": "σ(Πⁱ·|εᵢ| - θₜ) with proper form",
            },
            "F7.3": {
                "protocol": 7,
                "description": "Precision Update Rule",
                "threshold": "dΠⁱ/dt = α(Πⁱ - Πⁱ²ε²)",
            },
            "F7.4": {
                "protocol": 7,
                "description": "Threshold Dynamics",
                "threshold": "dθ/dt = β(S - θ) with τ = 1/β",
            },
            # FP-08: Parameter Sensitivity
            "F8.1": {
                "protocol": 8,
                "description": "θ₀ Identifiability",
                "threshold": "FIM diagonal > 0.1 for θ₀",
            },
            "F8.2": {
                "protocol": 8,
                "description": "Πⁱ Identifiability",
                "threshold": "FIM diagonal > 0.1 for Πⁱ",
            },
            "F8.3": {
                "protocol": 8,
                "description": "β Identifiability",
                "threshold": "FIM diagonal > 0.1 for β",
            },
            "F8.4": {
                "protocol": 8,
                "description": "Recovery Stability",
                "threshold": "r ≥ 0.82 core params",
            },
            "F8.5": {
                "protocol": 8,
                "description": "Sobol Sensitivity",
                "threshold": "Sᵢ > 0.10 for at least 2 params",
            },
            # FP-09: Neural Signatures (P3b, HEP)
            "P4.a": {
                "protocol": 9,
                "description": "PCI+HEP Joint AUC",
                "threshold": "AUC > 0.80",
            },
            "P4.b": {
                "protocol": 9,
                "description": "DMN Correlations",
                "threshold": "DMN↔PCI r>0.50, DMN↔HEP r<0.20",
            },
            "P4.c": {
                "protocol": 9,
                "description": "Cold Pressor Response",
                "threshold": "PCI >10% increase in MCS",
            },
            "P4.d": {
                "protocol": 9,
                "description": "Recovery Prediction",
                "threshold": "ΔR² > 0.10 at 6 months",
            },
            "P5.a": {
                "protocol": 9,
                "description": "vmPFC-SCR Correlation",
                "threshold": "r > 0.40",
            },
            "P5.b": {
                "protocol": 9,
                "description": "vmPFC-Insula Discrimination",
                "threshold": "r < 0.20",
            },
            # FP-10: Bayesian MCMC / Cross-Species
            "F10.MCMC": {
                "protocol": 10,
                "description": "MCMC Convergence",
                "threshold": "R̂ ≤ 1.01",
            },
            "F10.BF": {
                "protocol": 10,
                "description": "Bayes Factor",
                "threshold": "BF₁₀ ≥ 3",
            },
            "F10.MAE": {
                "protocol": 10,
                "description": "MAE Advantage",
                "threshold": "≥20% lower MAE",
            },
            "P12.a": {
                "protocol": 10,
                "description": "Cross-Species Scaling",
                "threshold": "Allometric exponents within ±2 SD",
            },
            "P12.b": {
                "protocol": 10,
                "description": "Brain-Body Scaling",
                "threshold": "0.65 ≤ b ≤ 0.75",
            },
            # FP-11: Liquid Network Dynamics
            "F11.1": {
                "protocol": 11,
                "description": "Spectral Radius Guard",
                "threshold": "ρ < 1.0",
            },
            "F11.2": {
                "protocol": 11,
                "description": "Echo State Property",
                "threshold": "Fading memory confirmed",
            },
            "F11.3": {
                "protocol": 11,
                "description": "Liquid Time Constants",
                "threshold": "τ distributed 10-500ms",
            },
            # FP-12: Framework Aggregator (meta)
            "FA": {
                "protocol": 12,
                "description": "Framework Falsification A",
                "threshold": "All 14 predictions fail",
            },
            "FB": {
                "protocol": 12,
                "description": "Framework Falsification B",
                "threshold": "ΔBIC < 10 (alternatives win)",
            },
        }

        # Available protocol configurations
        self.available_protocols = {
            "FP-01": {
                "file": "Falsification/FP_01_ActiveInference.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Active Inference Agents: F1.x (Architecture) + F2.x (Somatic Markers)",
                "tier": "primary",
            },
            "FP-02": {
                "file": "Falsification/FP_02_AgentComparison_ConvergenceBenchmark.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Agent Comparison: F3.x (Performance Benchmarks)",
                "tier": "primary",
            },
            "FP-03": {
                "file": "Falsification/FP_03_FrameworkLevel_MultiProtocol.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Framework-Level Multi-Protocol Synthesis",
                "tier": "secondary",
            },
            "FP-04": {
                "file": "Falsification/FP_04_PhaseTransition_EpistemicArchitecture.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Phase Transition / Epistemic Architecture Level 2",
                "tier": "secondary",
            },
            "FP-05": {
                "file": "Falsification/FP_05_EvolutionaryPlausibility.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Evolutionary Plausibility: F5.x",
                "tier": "tertiary",
            },
            "FP-06": {
                "file": "Falsification/FP_06_LiquidNetwork_EnergyBenchmark.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Liquid Network / Energy Benchmark: F6.x",
                "tier": "tertiary",
            },
            "FP-07": {
                "file": "Falsification/FP_07_MathematicalConsistency.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Mathematical Consistency: F7.x",
                "tier": "tertiary",
            },
            "FP-08": {
                "file": "Falsification/FP_08_ParameterSensitivity_Identifiability.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Parameter Sensitivity / Identifiability: F8.x",
                "tier": "secondary",
            },
            "FP-09": {
                "file": "Falsification/FP_09_NeuralSignatures_P3b_HEP.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Neural Signatures: P4.x, P5.x (P3b, HEP)",
                "tier": "tertiary",
            },
            "FP-10": {
                "file": "Falsification/FP_10_BayesianEstimation_MCMC.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Bayesian MCMC + Cross-Species Scaling",
                "tier": "tertiary",
            },
            "FP-10a": {
                "file": "Falsification/FP_10_BayesianEstimation_MCMC.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Bayesian MCMC Estimation (standalone)",
                "tier": "tertiary",
            },
            "FP-10b": {
                "file": "Falsification/FP_12_CrossSpeciesScaling.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Cross-Species Scaling (standalone)",
                "tier": "tertiary",
            },
            "FP-11": {
                "file": "Falsification/FP_11_LiquidNetworkDynamics_EchoState.py",
                "function": "run_falsification",
                "class": "None",
                "description": "Liquid Network Dynamics / Echo State",
                "tier": "secondary",
            },
            "FP-12": {
                "file": "Falsification/FP_ALL_Aggregator.py",
                "function": "run_framework_falsification",
                "class": "None",
                "description": "Framework-Level Aggregator (Conditions A & B)",
                "tier": "secondary",
            },
        }

        self.timeout_seconds = 300  # 5 minutes default timeout

    def get_criteria_by_protocol(self, protocol_num: int) -> Dict[str, Dict]:
        """Get all falsification criteria for a specific protocol.

        Args:
            protocol_num: Protocol number (1-12)

        Returns:
            Dictionary of criteria IDs to their definitions
        """
        return {
            k: v
            for k, v in self.FALSIFICATION_CRITERIA.items()
            if v.get("protocol") == protocol_num
        }

    def get_all_criteria(self) -> Dict[str, Dict]:
        """Return complete falsification criteria registry.

        Returns:
            Dictionary of all falsification criteria
        """
        return self.FALSIFICATION_CRITERIA.copy()

    def get_named_predictions(self) -> Dict[str, str]:
        """Return the 14 named predictions for framework-level falsification.

        Returns:
            Dictionary mapping prediction IDs to descriptions
        """
        return NAMED_PREDICTIONS.copy()

    def _prepare_vp5_genome_data(self) -> Dict[str, Any]:
        """
        Run the VP-5 evolutionary engine before FP-05 and extract genome-level metrics.

        This enforces the required VP-5 -> FP-05 ordering so FP-05 never runs
        without the evolutionary genome payload it expects.
        """
        project_root = Path(__file__).parent.parent
        vp5_path = project_root / "Validation" / "VP_05_EvolutionaryEmergence.py"
        spec = importlib.util.spec_from_file_location("Validation.VP_05", vp5_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load VP-05 module from {vp5_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        optimizer = module.EvolutionaryOptimizer(
            population_size=100,
            n_generations=100,
            mutation_rate=0.01,
            crossover_rate=0.7,
            tournament_size=5,
            elitism=5,
        )
        history = optimizer.run_evolution()
        best_genomes = history.get("best_genomes", [])
        genome_dicts = [
            genome.to_dict() if hasattr(genome, "to_dict") else genome
            for genome in best_genomes
        ]

        intero_gain_ratios = [
            float(genome.get("beta", 1.0))
            for genome in genome_dicts
            if genome.get("has_intero_weighting", False)
        ]
        alpha_values = [float(genome.get("alpha", 0.0)) for genome in genome_dicts]

        return {
            "source": "VP-05",
            "history_length": len(history.get("generation", [])),
            "intero_gain_ratios": intero_gain_ratios,
            "alpha_values": alpha_values,
            "best_genomes": genome_dicts,
            "architecture_frequencies": history.get("architecture_frequencies", []),
        }

    def _prepare_vp3_agent(self) -> Any:
        """
        CRIT-04 FIX: Run VP-03 to instantiate APGIAgent for FP-08 dependency injection.

        FP-08 (Parameter Sensitivity) requires a live APGIAgent for F8.SA (Sobol analysis).
        Per FP-8 specification, synthetic oracles are NOT permitted for sensitivity analysis.
        This method enforces the VP-03 -> FP-08 dependency chain.

        Returns:
            APGIAgent instance ready for sensitivity analysis

        Raises:
            ImportError: If VP-03 module cannot be loaded
            RuntimeError: If APGIAgent cannot be instantiated
        """
        project_root = Path(__file__).parent.parent
        vp3_path = (
            project_root / "Validation" / "VP_03_ActiveInference_AgentSimulations.py"
        )

        logger.info(
            "CRIT-04 FIX: Running VP-03 prerequisite to prepare APGIAgent for FP-08..."
        )

        spec = importlib.util.spec_from_file_location("Validation.VP_03", vp3_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load VP-03 module from {vp3_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get APGIAgent class from module
        if not hasattr(module, "APGIAgent"):
            raise RuntimeError("VP-03 module does not export APGIAgent class")

        APGIAgent = module.APGIAgent

        # Instantiate agent with default config for sensitivity analysis
        try:
            # Try to get APGIConfig if available
            if hasattr(module, "APGIConfig"):
                config = module.APGIConfig()
                agent = APGIAgent(config=config)
            else:
                agent = APGIAgent()

            logger.info(
                f"CRIT-04 FIX: Successfully instantiated APGIAgent ({type(agent).__name__})"
            )
            return agent

        except Exception as e:
            raise RuntimeError(f"Failed to instantiate APGIAgent from VP-03: {e}")

    def run_falsification(self, protocols: List[str], **kwargs) -> Dict[str, Any]:
        """Run specified falsification protocols.

        Args:
            protocols: List of protocol names (e.g., ["FP-01", "FP-02"])
            **kwargs: Additional arguments passed to protocol functions

        Returns:
            Dictionary of protocol results with aggregated falsification status
        """
        results: Dict[str, Dict[str, Any]] = {}

        for protocol_name in protocols:
            if protocol_name not in self.available_protocols:
                logger.warning(f"Unknown falsification protocol: {protocol_name}")
                results[protocol_name] = {
                    "status": "error",
                    "message": f"Unknown protocol: {protocol_name}",
                }
                continue

            try:
                logger.info(f"Running falsification protocol {protocol_name}...")
                protocol_info = self.available_protocols[protocol_name]
                protocol_kwargs = dict(kwargs)
                if protocol_name == "FP-05":
                    if "genome_data" not in protocol_kwargs:
                        logger.info(
                            "Running VP-05 prerequisite before FP-05 to prepare genome_data."
                        )
                        protocol_kwargs["genome_data"] = self._prepare_vp5_genome_data()
                    # Set n_replicates=5 for FP-05 as per specification
                    protocol_kwargs.setdefault("n_replicates", 5)

                # CRIT-04 FIX: Inject APGIAgent for FP-08
                if protocol_name == "FP-08":
                    if "agent_instance" not in protocol_kwargs:
                        try:
                            logger.info(
                                "CRIT-04 FIX: Running VP-03 prerequisite before FP-08 to prepare APGIAgent."
                            )
                            protocol_kwargs["agent_instance"] = (
                                self._prepare_vp3_agent()
                            )
                        except Exception as agent_error:
                            logger.error(
                                f"CRIT-04 FIX: Failed to prepare APGIAgent for FP-08: {agent_error}. "
                                "FP-08 will use synthetic fallback (violates F8.SA specification)."
                            )
                            # Don't fail entirely - let FP-08 handle the fallback

                result = self._run_single_protocol(protocol_info, **protocol_kwargs)
                results[protocol_name] = result
                logger.info(
                    f"{protocol_name} completed: {result.get('status', 'unknown')}"
                )
            except Exception as e:
                logger.error(f"Error running {protocol_name}: {e}")
                results[protocol_name] = {
                    "status": "error",
                    "message": str(e),
                    "falsified": "true",  # Error = falsified (conservative)
                }

        self.protocol_results.update(results)
        return results

    def _run_single_protocol(
        self, protocol_info: Dict[str, str], **kwargs
    ) -> Dict[str, Any]:
        """Run a single falsification protocol.

        Args:
            protocol_info: Protocol configuration dict
            **kwargs: Arguments passed to protocol

        Returns:
            Protocol execution results
        """
        # Ensure project root is in sys.path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Load the protocol module
        file_path = project_root / protocol_info["file"]

        if not file_path.exists():
            return {
                "status": "error",
                "message": f"File not found: {file_path}",
                "falsified": True,
            }

        spec = importlib.util.spec_from_file_location(
            protocol_info["file"].replace("/", ".").replace(".py", ""),
            file_path,
        )
        if spec is None or spec.loader is None:
            return {
                "status": "error",
                "message": f"Could not load spec for {file_path}",
                "falsified": True,
            }

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the entry point (function or class method)
        func_name = protocol_info.get("function", "run_falsification")
        class_name = protocol_info.get("class", "None")

        if class_name and class_name != "None":
            # Instantiate class and call method
            cls = getattr(module, class_name)
            instance = cls()
            if hasattr(instance, func_name):
                run_func = getattr(instance, func_name)
            elif hasattr(instance, "run_full_experiment"):
                run_func = instance.run_full_experiment
            else:
                return {
                    "status": "error",
                    "message": f"No runnable method found in {class_name}",
                    "falsified": True,
                }
        else:
            # Direct function call
            run_func = getattr(module, func_name, None)
            if run_func is None:
                return {
                    "status": "error",
                    "message": f"Function {func_name} not found",
                    "falsified": True,
                }

        # Run the protocol
        result = run_func(**kwargs)

        # Standardize result format
        if isinstance(result, dict):
            if "falsified" not in result:
                # Infer falsified status from passed/passed_criteria
                passed = result.get("passed", False)
                result["falsified"] = not passed
            if "status" not in result:
                result["status"] = "falsified" if result["falsified"] else "passed"
        else:
            result = {
                "status": "passed" if result else "falsified",
                "falsified": not bool(result),
                "raw_result": result,
            }

        return result

    def run_all_protocols(self, **kwargs) -> Dict[str, Any]:
        """Run all available falsification protocols.

        Args:
            **kwargs: Arguments passed to all protocols

        Returns:
            Complete falsification results with framework-level aggregation
        """
        all_protocols = list(self.available_protocols.keys())

        # Exclude meta-protocols from direct execution
        run_protocols = [p for p in all_protocols if p not in ["FP-10", "FP-12"]]

        logger.info(f"Running {len(run_protocols)} falsification protocols...")
        results = self.run_falsification(run_protocols, **kwargs)

        # Aggregate results at framework level
        framework_results = self.aggregate_framework_falsification(results)

        return {
            "protocol_results": results,
            "framework_falsification": framework_results,
            "summary": self._generate_summary(results, framework_results),
        }

    def aggregate_framework_falsification(
        self, protocol_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply framework-level falsification conditions (A and B).

        Args:
            protocol_results: Results from individual protocols

        Returns:
            Framework-level falsification analysis
        """
        return run_framework_falsification(protocol_results)

    def _generate_summary(
        self, protocol_results: Dict[str, Any], framework_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary of falsification results.

        Args:
            protocol_results: Individual protocol results
            framework_results: Aggregated framework results

        Returns:
            Summary statistics
        """
        total = len(protocol_results)
        passed = sum(
            1 for r in protocol_results.values() if not r.get("falsified", True)
        )
        failed = total - passed

        # Weighted scoring by protocol tier
        tier_weights = {
            "primary": 2.0,  # Highest weight for primary protocols
            "secondary": 1.5,  # Intermediate weight for secondary protocols
            "tertiary": 1.0,  # Standard weight for tertiary protocols
        }

        # Use tier_weights to avoid unused variable warning
        _ = tier_weights
        weighted_score = self._calculate_weighted_score(protocol_results)

        # Count by tier
        tier_counts = {
            "primary": {"total": 0, "passed": 0},
            "secondary": {"total": 0, "passed": 0},
            "tertiary": {"total": 0, "passed": 0},
        }
        for name, result in protocol_results.items():
            # Extract protocol number
            try:
                protocol_num = int(name.split("_")[-1])
                if protocol_num <= 3:
                    tier = "primary"
                elif protocol_num <= 6:
                    tier = "secondary"
                else:
                    tier = "tertiary"
            except (ValueError, IndexError):
                tier = "tertiary"

            tier_counts[tier]["total"] += 1
            if isinstance(result, dict) and not result.get("falsified", True):
                tier_counts[tier]["passed"] += 1

        return {
            "total_protocols": total,
            "passed_protocols": passed,
            "failed_protocols": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "weighted_score": weighted_score,
            "tier_counts": tier_counts,
            "framework_results": framework_results,
        }

    def _calculate_weighted_score(self, protocol_results: Dict[str, Any]) -> float:
        """Calculate weighted score based on protocol results."""
        tier_weights = {
            "primary": 2.0,  # Highest weight for primary protocols
            "secondary": 1.5,  # Intermediate weight for secondary protocols
            "tertiary": 1.0,  # Standard weight for tertiary protocols
        }

        weighted_score = 0.0
        total_weight = 0.0

        for protocol_name, result in protocol_results.items():
            if isinstance(result, dict) and "passed" in result:
                # Determine tier based on protocol name
                if any(
                    keyword in protocol_name.lower()
                    for keyword in ["vp01", "vp02", "vp03"]
                ):
                    tier = "primary"
                elif any(
                    keyword in protocol_name.lower()
                    for keyword in ["vp04", "vp05", "vp06"]
                ):
                    tier = "secondary"
                else:
                    tier = "tertiary"

                weight = tier_weights.get(tier, 1.0)
                weighted_score += weight * (1.0 if result["passed"] else 0.0)
                total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def get_falsification_report(self) -> str:
        """Generate a formatted falsification criteria report.

        Returns:
            Markdown-formatted string with all falsification criteria
        """
        report = []
        report.append("# APGI Falsification Criteria Registry")
        report.append("")
        report.append(
            "This document contains all falsification criteria organized by protocol tier."
        )
        report.append("")

        # Group criteria by tier
        tier_groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
            "primary": [],
            "secondary": [],
            "tertiary": [],
        }
        for criterion, info in self.FALSIFICATION_CRITERIA.items():
            protocol_num = info["protocol"]
            tier = self.PROTOCOL_TIERS.get(protocol_num, "unknown")
            tier_groups[tier].append((criterion, info))

        # Generate tier sections
        for tier in ["primary", "secondary", "tertiary"]:
            if tier_groups[tier]:
                report.append(f"## {tier.title()} Protocols")
                report.append("")

                for criterion, info in tier_groups[tier]:
                    report.append(f"### {criterion}")
                    report.append(f"**Protocol:** FP-{info['protocol']:02d}")
                    report.append(f"**Description:** {info['description']}")
                    if "threshold" in info:
                        report.append(f"**Threshold:** {info['threshold']}")
                    report.append("")

        return "\n".join(report)


def main():
    """Run master falsification protocol."""
    falsifier = APGIMasterFalsifier()
    return falsifier.run_master_falsification()


if __name__ == "__main__":
    main()


def get_named_predictions() -> Dict[str, str]:
    """Convenience function: Return 14 named predictions.

    Returns:
        Dictionary of named predictions
    """
    return NAMED_PREDICTIONS.copy()


def run_all_falsification_protocols(**kwargs) -> Dict[str, Any]:
    """Convenience function: Run all falsification protocols.

    Args:
        **kwargs: Arguments passed to protocols

    Returns:
        Complete falsification results
    """
    master = APGIMasterFalsifier()
    return master.run_all_protocols(**kwargs)


def generate_falsification_report() -> str:
    """Convenience function: Generate falsification report.

    Returns:
        Markdown-formatted report
    """
    master = APGIMasterFalsifier()
    return master.get_falsification_report()


if __name__ == "__main__":
    print("=" * 70)
    print("APGI Master Falsification Orchestrator")
    print("=" * 70)
    print()

    master = APGIMasterFalsifier()

    # Print criteria registry
    print(master.get_falsification_report())

    # Print available protocols
    print("\n" + "=" * 70)
    print("Available Falsification Protocols")
    print("=" * 70)
    print()

    for name, info in master.available_protocols.items():
        print(f"  {name}: {info['description']}")
        print(f"    File: {info['file']}")
        print(f"    Tier: {info['tier']}")
        print()

    print("\n" + "=" * 70)
    print("To run all protocols:")
    print("  master = APGIMasterFalsifier()")
    print("  results = master.run_all_protocols()")
    print("=" * 70)
