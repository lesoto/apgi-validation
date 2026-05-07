#!/usr/bin/env python3
"""
APGI Master Validation Module
==========================

Provides APGIMasterValidator class for validation protocol management.
This is a compatibility layer that wraps the APGIMasterFalsifier.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path for imports
_proj_root = Path(__file__).parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

# Import the actual master falsifier
from Falsification.Master_Falsification import APGIMasterFalsifier


class APGIMasterValidator:
    """
    Compatibility wrapper for APGIMasterFalsifier.

    Provides the interface expected by tests while delegating to the actual
    master falsifier implementation.
    """

    def __init__(self, timeout_seconds: int = 3600):
        """Initialize the master validator."""
        self._falsifier = APGIMasterFalsifier()
        self.timeout_seconds = timeout_seconds

        # Expose the falsifier's attributes for test compatibility
        self.protocol_results = self._falsifier.protocol_results
        self.PROTOCOL_TIERS = self._falsifier.PROTOCOL_TIERS
        self.falsification_status = self._falsifier.falsification_status
        self.contract_diagnostics = self._generate_contract_diagnostics()

    def _generate_contract_diagnostics(self) -> Dict[str, Any]:
        """Generate contract diagnostics information."""
        return {
            "Protocol-1": "Active Inference Agents (F1.x, F2.x)",
            "Protocol-2": "Agent Comparison / Convergence (F3.x)",
            "Protocol-3": "Framework-Level Multi-Protocol",
            "total_protocols": len(self.PROTOCOL_TIERS),
            "primary_protocols": len(
                [k for k, v in self.PROTOCOL_TIERS.items() if v == "primary"]
            ),
            "secondary_protocols": len(
                [k for k, v in self.PROTOCOL_TIERS.items() if v == "secondary"]
            ),
            "tertiary_protocols": len(
                [k for k, v in self.PROTOCOL_TIERS.items() if v == "tertiary"]
            ),
        }

    def _validate_dependency_graph(self) -> None:
        """Validate protocol dependency graph."""
        # Delegate to the falsifier's validation if it exists
        if hasattr(self._falsifier, "_validate_dependency_graph"):
            self._falsifier._validate_dependency_graph()
        else:
            # Default implementation - just ensure all protocols have tiers
            for protocol_id in self.PROTOCOL_TIERS:
                if protocol_id not in self.PROTOCOL_TIERS:
                    raise ValueError(f"Protocol {protocol_id} missing tier assignment")

    def _is_protocol_passed(self, result: Any) -> bool:
        """
        Check if a protocol result indicates passing status.

        Args:
            result: Protocol result object with metadata and named_predictions

        Returns:
            True if protocol passed, False otherwise
        """
        if result is None:
            return False

        # Check metadata for explicit pass flag
        if hasattr(result, "metadata") and result.metadata:
            if isinstance(result.metadata, dict) and result.metadata.get("passed"):
                return True

        # Check named predictions - all must pass
        if hasattr(result, "named_predictions") and result.named_predictions:
            if not result.named_predictions:  # Empty predictions
                return False

            # All predictions must have passed=True
            for pred_name, pred_obj in result.named_predictions.items():
                if hasattr(pred_obj, "passed") and not pred_obj.passed:
                    return False

        # Default to False if no clear pass indication
        return False


# Export logger if available from the falsifier
try:
    pass
except ImportError:
    import logging
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from utils.logging_config import APGILogger
    else:
        APGILogger = logging.Logger

    master_logger: APGILogger = logging.getLogger(__name__)  # type: ignore[assignment]

    def run_validation(
        self, protocol_names: List[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Run validation protocols and return results.

        Args:
            protocol_names: List of protocol names to run. If None, runs all available protocols.
            **kwargs: Additional parameters passed to protocols (n_participants, seed, etc.)

        Returns:
            Dictionary mapping protocol names to their results
        """
        if protocol_names is None:
            protocol_names = list(self.PROTOCOL_TIERS.keys())

        results = {}
        for protocol_name in protocol_names:
            if protocol_name in self.PROTOCOL_TIERS:
                try:
                    # Construct protocol_info dict for falsifier
                    protocol_info = self.PROTOCOL_TIERS.get(protocol_name, {})
                    result = self._falsifier._run_single_protocol(
                        protocol_info, **kwargs
                    )
                    results[protocol_name] = result
                except Exception as e:
                    master_logger.error(f"Protocol {protocol_name} failed: {e}")
                    results[protocol_name] = {
                        "status": "error",
                        "message": str(e),
                        "passed": False,
                    }
            else:
                results[protocol_name] = {
                    "status": "error",
                    "message": f"Unknown protocol: {protocol_name}",
                    "passed": False,
                }

        return results

    def run_all_protocols(self, **kwargs) -> Dict[str, Any]:
        """
        Run all available validation protocols.

        Args:
            **kwargs: Additional parameters passed to protocols

        Returns:
            Dictionary mapping all protocol names to their results
        """
        return self.run_validation(protocol_names=None, **kwargs)

    def generate_master_report(self) -> Any:
        """
        Generate a comprehensive master validation report.

        Returns:
            Master report object containing aggregated results
        """
        return self._falsifier.generate_master_report()

    def get_available_protocols(self) -> List[str]:
        """
        Get list of all available validation protocols.

        Returns:
            List of available protocol names
        """
        return list(self.PROTOCOL_TIERS.keys())
