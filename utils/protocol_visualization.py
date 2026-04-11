"""
Protocol Visualization Utility

Standardized PNG output generation for all APGI falsification protocols.
Each protocol should output protocolNUMBER.png when run.
"""

import logging
from pathlib import Path
from typing import Any, Dict

# Try to import matplotlib with fallback
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

logger = logging.getLogger(__name__)


class ProtocolVisualizer:
    """Standardized visualization for APGI protocols"""

    def __init__(self, protocol_number: int, output_dir: str = None):
        self.protocol_number = protocol_number
        self.protocol_name = f"FP-{protocol_number:02d}"

        # Set output directory
        if output_dir is None:
            self.output_dir = (
                Path(__file__).parent.parent / "results" / "visualizations"
            )
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / f"protocol{protocol_number:02d}.png"

    def create_summary_plot(self, results: Dict[str, Any]) -> bool:
        """Create a standardized summary plot for protocol results"""
        if not HAS_MATPLOTLIB:
            logger.warning(
                f"Matplotlib not available for {self.protocol_name} visualization"
            )
            return False

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(
                f"{self.protocol_name} Falsification Results",
                fontsize=16,
                fontweight="bold",
            )

            # Flatten axes for easier indexing
            axes = axes.flatten()

            # Plot 1: Overall Status
            self._plot_status(axes[0], results)

            # Plot 2: Criteria Results
            self._plot_criteria(axes[1], results)

            # Plot 3: Key Metrics
            self._plot_metrics(axes[2], results)

            # Plot 4: Summary Statistics
            self._plot_summary(axes[3], results)

            plt.tight_layout()
            plt.savefig(self.output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(
                f"Saved {self.protocol_name} visualization to {self.output_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create {self.protocol_name} visualization: {e}")
            return False

    def _plot_status(self, ax, results: Dict[str, Any]):
        """Plot overall falsification status"""
        status = results.get("status", "UNKNOWN")

        # Color based on status
        colors = {
            "PASS": "#2ecc71",  # Green
            "FAIL": "#e74c3c",  # Red
            "ERROR": "#f39c12",  # Orange
            "UNKNOWN": "#95a5a6",  # Gray
        }

        color = colors.get(status, "#95a5a6")

        # Create status indicator
        ax.bar(1, 1, color=color, width=0.5)
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1.2)
        ax.text(
            1,
            0.5,
            status,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="white",
        )
        ax.set_title("Overall Status")
        ax.set_xticks([])
        ax.set_yticks([])

    def _plot_criteria(self, ax, results: Dict[str, Any]):
        """Plot criteria results"""
        criteria = results.get("criteria", {})
        if not criteria:
            ax.text(0.5, 0.5, "No Criteria Data", ha="center", va="center")
            ax.set_title("Criteria Results")
            return

        # Count passed/failed criteria
        passed = sum(1 for c in criteria.values() if c.get("passed", False))
        failed = sum(1 for c in criteria.values() if not c.get("passed", False))
        total = len(criteria)

        # Create pie chart
        if total > 0:
            sizes = [passed, failed]
            labels = ["Passed", "Failed"]
            colors = ["#2ecc71", "#e74c3c"]

            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors, autopct="%d%%"
            )
            ax.set_title(f"Criteria ({passed}/{total} Passed)")
        else:
            ax.text(0.5, 0.5, "No Criteria", ha="center", va="center")
            ax.set_title("Criteria Results")

    def _plot_metrics(self, ax, results: Dict[str, Any]):
        """Plot key metrics"""
        metrics = {}

        # Extract common metrics from different protocols
        if "summary" in results:
            summary = results["summary"]
            metrics.update(
                {
                    "Total Criteria": summary.get("total_criteria", 0),
                    "Passed": summary.get("passed", 0),
                    "Failed": summary.get("failed", 0),
                }
            )

        # Add protocol-specific metrics
        for key, value in results.items():
            if isinstance(value, (int, float)) and key not in metrics:
                if abs(value) < 1000:  # Avoid very large numbers
                    metrics[key] = value

        if not metrics:
            ax.text(0.5, 0.5, "No Metrics Data", ha="center", va="center")
            ax.set_title("Key Metrics")
            return

        # Create bar chart
        names = list(metrics.keys())[:5]  # Limit to 5 metrics
        values = [metrics[name] for name in names]

        bars = ax.bar(range(len(names)), values, color="#3498db")
        ax.set_title("Key Metrics")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

    def _plot_summary(self, ax, results: Dict[str, Any]):
        """Plot summary statistics"""
        summary_text = []

        # Basic info
        summary_text.append(f"Protocol: {self.protocol_name}")
        summary_text.append(f"Status: {results.get('status', 'UNKNOWN')}")

        # Add criteria summary
        criteria = results.get("criteria", {})
        if criteria:
            passed = sum(1 for c in criteria.values() if c.get("passed", False))
            total = len(criteria)
            summary_text.append(f"Criteria: {passed}/{total} Passed")

        # Add timestamp
        from datetime import datetime

        summary_text.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Display as text
        ax.text(
            0.05,
            0.95,
            "\n".join(summary_text),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        ax.set_title("Summary")
        ax.set_xticks([])
        ax.set_yticks([])

    def create_custom_plot(self, plot_func, title: str = None) -> bool:
        """Create a custom plot using a provided function"""
        if not HAS_MATPLOTLIB:
            logger.warning(
                f"Matplotlib not available for {self.protocol_name} visualization"
            )
            return False

        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Call the custom plotting function
            if plot_func(fig, ax):
                if title:
                    ax.set_title(
                        f"{self.protocol_name}: {title}", fontsize=16, fontweight="bold"
                    )
                else:
                    ax.set_title(
                        f"{self.protocol_name} Results", fontsize=16, fontweight="bold"
                    )

                plt.tight_layout()
                plt.savefig(self.output_path, dpi=300, bbox_inches="tight")
                plt.close()

                logger.info(
                    f"Saved custom {self.protocol_name} visualization to {self.output_path}"
                )
                return True
            else:
                plt.close()
                return False

        except Exception as e:
            logger.error(
                f"Failed to create custom {self.protocol_name} visualization: {e}"
            )
            return False


def add_standard_png_output(
    protocol_number: int,
    results: Dict[str, Any],
    custom_plot_func=None,
    title: str = None,
) -> bool:
    """
    Convenience function to add standardized PNG output to any protocol

    Args:
        protocol_number: Protocol number (1-12)
        results: Protocol results dictionary
        custom_plot_func: Optional custom plotting function
        title: Optional custom title

    Returns:
        True if successful, False otherwise
    """
    visualizer = ProtocolVisualizer(protocol_number)

    if custom_plot_func is not None:
        return visualizer.create_custom_plot(custom_plot_func, title)
    else:
        return visualizer.create_summary_plot(results)


# Protocol-specific plotting functions
def fp01_custom_plot(fig, ax):
    """Custom plot for FP-01 Active Inference"""
    # This would be implemented with FP-01 specific data
    return False  # Fallback to standard plot


def fp02_custom_plot(fig, ax):
    """Custom plot for FP-02 Agent Comparison"""
    # This would be implemented with FP-02 specific data
    return False  # Fallback to standard plot


def fp03_custom_plot(fig, ax):
    """Custom plot for FP-03 Framework Level"""
    # This would be implemented with FP-03 specific data
    return False  # Fallback to standard plot


def fp04_custom_plot(fig, ax):
    """Custom plot for FP-04 Phase Transition"""
    # This would be implemented with FP-04 specific data
    return False  # Fallback to standard plot


def fp05_custom_plot(fig, ax):
    """Custom plot for FP-05 Evolutionary Plausibility"""
    # This would be implemented with FP-05 specific data
    return False  # Fallback to standard plot
