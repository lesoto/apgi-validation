"""
APGI Pipeline Visualization
===========================

Comprehensive pipeline visualization system for APGI framework.
Provides visual representation of data processing pipelines, validation workflows,
 and system architecture with interactive diagrams.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# APGI imports
from logging_config import apgi_logger
from matplotlib.patches import FancyBboxPatch


class NodeType(Enum):
    """Types of nodes in pipeline diagrams."""

    INPUT = "input"
    PREPROCESSOR = "preprocessor"
    TRANSFORMER = "transformer"
    VALIDATOR = "validator"
    OUTPUT = "output"
    DECISION = "decision"
    MERGE = "merge"
    SPLIT = "split"


@dataclass
class PipelineNode:
    """Node in a pipeline diagram."""

    id: str
    name: str
    node_type: NodeType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    position: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineEdge:
    """Edge in a pipeline diagram."""

    source: str
    target: str
    description: str = ""
    data_type: str = "data"
    condition: Optional[str] = None


@dataclass
class PipelineDiagram:
    """Complete pipeline diagram."""

    name: str
    description: str
    nodes: List[PipelineNode]
    edges: List[PipelineEdge]
    layout: str = "hierarchical"
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineVisualizer:
    """Advanced pipeline visualization system."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("apgi_output/pipeline_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color schemes for different node types
        self.colors = {
            NodeType.INPUT: "#2ecc71",  # Green
            NodeType.PREPROCESSOR: "#3498db",  # Blue
            NodeType.TRANSFORMER: "#9b59b6",  # Purple
            NodeType.VALIDATOR: "#e74c3c",  # Red
            NodeType.OUTPUT: "#f39c12",  # Orange
            NodeType.DECISION: "#95a5a6",  # Gray
            NodeType.MERGE: "#1abc9c",  # Turquoise
            NodeType.SPLIT: "#34495e",  # Dark gray
        }

        # Node shapes
        self.shapes = {
            NodeType.INPUT: "ellipse",
            NodeType.PREPROCESSOR: "box",
            NodeType.TRANSFORMER: "diamond",
            NodeType.VALIDATOR: "hexagon",
            NodeType.OUTPUT: "ellipse",
            NodeType.DECISION: "diamond",
            NodeType.MERGE: "circle",
            NodeType.SPLIT: "circle",
        }

    def create_preprocessing_pipeline_diagram(self) -> PipelineDiagram:
        """Create preprocessing pipeline diagram."""

        nodes = [
            PipelineNode(
                id="raw_data",
                name="Raw Data",
                node_type=NodeType.INPUT,
                description="Input multimodal data (EEG, Pupil, EDA, HR)",
                metadata={"formats": ["CSV", "EDF", "MAT"]},
            ),
            PipelineNode(
                id="data_validation",
                name="Data Validation",
                node_type=NodeType.VALIDATOR,
                description="Validate data format and structure",
                parameters={"strict_mode": True, "schema_check": True},
            ),
            PipelineNode(
                id="eeg_preprocessor",
                name="EEG Preprocessor",
                node_type=NodeType.PREPROCESSOR,
                description="Process EEG signals with artifact removal",
                parameters={
                    "bandpass_filter": [0.5, 40],
                    "notch_filter": 50,
                    "ica_artifact_removal": True,
                },
            ),
            PipelineNode(
                id="pupil_preprocessor",
                name="Pupil Preprocessor",
                node_type=NodeType.PREPROCESSOR,
                description="Process pupil diameter data",
                parameters={
                    "blink_detection": True,
                    "interpolation": "cubic",
                    "baseline_correction": True,
                },
            ),
            PipelineNode(
                id="eda_preprocessor",
                name="EDA Preprocessor",
                node_type=NodeType.PREPROCESSOR,
                description="Process electrodermal activity",
                parameters={
                    "lowpass_filter": 5.0,
                    "decomposition": True,
                    "peak_detection": True,
                },
            ),
            PipelineNode(
                id="hr_preprocessor",
                name="Heart Rate Preprocessor",
                node_type=NodeType.PREPROCESSOR,
                description="Process heart rate data",
                parameters={
                    "artifact_removal": True,
                    "hrv_calculation": True,
                    "bandpass_filter": [0.5, 4.0],
                },
            ),
            PipelineNode(
                id="quality_assessment",
                name="Quality Assessment",
                node_type=NodeType.VALIDATOR,
                description="Assess data quality and detect anomalies",
                parameters={"threshold": 0.8, "anomaly_detection": True},
            ),
            PipelineNode(
                id="data_integration",
                name="Data Integration",
                node_type=NodeType.MERGE,
                description="Integrate preprocessed multimodal data",
                parameters={"synchronization": True, "interpolation": "linear"},
            ),
            PipelineNode(
                id="feature_extraction",
                name="Feature Extraction",
                node_type=NodeType.TRANSFORMER,
                description="Extract features for analysis",
                parameters={
                    "frequency_features": True,
                    "time_features": True,
                    "connectivity_features": True,
                },
            ),
            PipelineNode(
                id="processed_data",
                name="Processed Data",
                node_type=NodeType.OUTPUT,
                description="Clean, integrated, and feature-rich data",
                metadata={"format": "HDF5", "compression": True},
            ),
        ]

        edges = [
            PipelineEdge("raw_data", "data_validation", "Validate input data"),
            PipelineEdge("data_validation", "eeg_preprocessor", "Route EEG data", "EEG"),
            PipelineEdge("data_validation", "pupil_preprocessor", "Route pupil data", "Pupil"),
            PipelineEdge("data_validation", "eda_preprocessor", "Route EDA data", "EDA"),
            PipelineEdge("data_validation", "hr_preprocessor", "Route HR data", "HR"),
            PipelineEdge("eeg_preprocessor", "quality_assessment", "EEG quality check"),
            PipelineEdge("pupil_preprocessor", "quality_assessment", "Pupil quality check"),
            PipelineEdge("eda_preprocessor", "quality_assessment", "EDA quality check"),
            PipelineEdge("hr_preprocessor", "quality_assessment", "HR quality check"),
            PipelineEdge("quality_assessment", "data_integration", "Integrate quality data"),
            PipelineEdge("data_integration", "feature_extraction", "Extract features"),
            PipelineEdge("feature_extraction", "processed_data", "Output processed data"),
        ]

        return PipelineDiagram(
            name="APGI Data Preprocessing Pipeline",
            description="Complete multimodal data preprocessing workflow",
            nodes=nodes,
            edges=edges,
            layout="hierarchical",
        )

    def create_validation_pipeline_diagram(self) -> PipelineDiagram:
        """Create validation pipeline diagram."""

        nodes = [
            PipelineNode(
                id="validation_start",
                name="Validation Start",
                node_type=NodeType.INPUT,
                description="Begin validation process",
            ),
            PipelineNode(
                id="protocol_selection",
                name="Protocol Selection",
                node_type=NodeType.DECISION,
                description="Select validation protocol",
                parameters={"protocols": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]},
            ),
            PipelineNode(
                id="protocol_1",
                name="Protocol 1: Synthetic Data",
                node_type=NodeType.VALIDATOR,
                description="Test with synthetic consciousness data",
            ),
            PipelineNode(
                id="protocol_2",
                name="Protocol 2: Bayesian Comparison",
                node_type=NodeType.VALIDATOR,
                description="Compare with existing datasets",
            ),
            PipelineNode(
                id="protocol_3",
                name="Protocol 3: Agent Simulations",
                node_type=NodeType.VALIDATOR,
                description="Run active inference agent simulations",
            ),
            PipelineNode(
                id="protocol_4",
                name="Protocol 4: Cross-Modal",
                node_type=NodeType.VALIDATOR,
                description="Cross-modal replication analysis",
            ),
            PipelineNode(
                id="protocol_5",
                name="Protocol 5: Falsification",
                node_type=NodeType.VALIDATOR,
                description="Computational falsification tests",
            ),
            PipelineNode(
                id="protocol_6",
                name="Protocol 6: Real-Time",
                node_type=NodeType.VALIDATOR,
                description="Real-time implementation tests",
            ),
            PipelineNode(
                id="protocol_7",
                name="Protocol 7: Clinical",
                node_type=NodeType.VALIDATOR,
                description="Clinical translation validation",
            ),
            PipelineNode(
                id="protocol_8",
                name="Protocol 8: Psychometric",
                node_type=NodeType.VALIDATOR,
                description="Psychometric validation",
            ),
            PipelineNode(
                id="result_aggregation",
                name="Result Aggregation",
                node_type=NodeType.MERGE,
                description="Aggregate validation results",
            ),
            PipelineNode(
                id="meta_analysis",
                name="Meta-Analysis",
                node_type=NodeType.TRANSFORMER,
                description="Perform meta-analysis across protocols",
            ),
            PipelineNode(
                id="decision_point",
                name="Validation Decision",
                node_type=NodeType.DECISION,
                description="Make final validation decision",
            ),
            PipelineNode(
                id="report_generation",
                name="Report Generation",
                node_type=NodeType.OUTPUT,
                description="Generate validation report",
            ),
        ]

        edges = [
            PipelineEdge("validation_start", "protocol_selection", "Select protocol"),
            PipelineEdge("protocol_selection", "protocol_1", "Select P1", "P1"),
            PipelineEdge("protocol_selection", "protocol_2", "Select P2", "P2"),
            PipelineEdge("protocol_selection", "protocol_3", "Select P3", "P3"),
            PipelineEdge("protocol_selection", "protocol_4", "Select P4", "P4"),
            PipelineEdge("protocol_selection", "protocol_5", "Select P5", "P5"),
            PipelineEdge("protocol_selection", "protocol_6", "Select P6", "P6"),
            PipelineEdge("protocol_selection", "protocol_7", "Select P7", "P7"),
            PipelineEdge("protocol_selection", "protocol_8", "Select P8", "P8"),
            PipelineEdge("protocol_1", "result_aggregation", "P1 results"),
            PipelineEdge("protocol_2", "result_aggregation", "P2 results"),
            PipelineEdge("protocol_3", "result_aggregation", "P3 results"),
            PipelineEdge("protocol_4", "result_aggregation", "P4 results"),
            PipelineEdge("protocol_5", "result_aggregation", "P5 results"),
            PipelineEdge("protocol_6", "result_aggregation", "P6 results"),
            PipelineEdge("protocol_7", "result_aggregation", "P7 results"),
            PipelineEdge("protocol_8", "result_aggregation", "P8 results"),
            PipelineEdge("result_aggregation", "meta_analysis", "Analyze results"),
            PipelineEdge("meta_analysis", "decision_point", "Make decision"),
            PipelineEdge("decision_point", "report_generation", "Generate report"),
        ]

        return PipelineDiagram(
            name="APGI Validation Pipeline",
            description="Complete validation protocol workflow",
            nodes=nodes,
            edges=edges,
            layout="hierarchical",
        )

    def create_system_architecture_diagram(self) -> PipelineDiagram:
        """Create system architecture diagram."""

        nodes = [
            PipelineNode(
                id="user_interface",
                name="User Interface",
                node_type=NodeType.INPUT,
                description="CLI and GUI interfaces",
                metadata={"components": ["CLI", "GUI", "Dashboard"]},
            ),
            PipelineNode(
                id="config_manager",
                name="Configuration Manager",
                node_type=NodeType.PREPROCESSOR,
                description="Manage configuration profiles and settings",
            ),
            PipelineNode(
                id="data_layer",
                name="Data Layer",
                node_type=NodeType.PREPROCESSOR,
                description="Data processing and quality assessment",
                metadata={"components": ["Preprocessing", "Quality Assessment", "Cache"]},
            ),
            PipelineNode(
                id="apgi_core",
                name="APGI Core Engine",
                node_type=NodeType.TRANSFORMER,
                description="Core APGI computations and algorithms",
                metadata={"components": ["Formal Model", "Dynamics", "State Estimation"]},
            ),
            PipelineNode(
                id="validation_engine",
                name="Validation Engine",
                node_type=NodeType.VALIDATOR,
                description="Validation protocols and falsification tests",
            ),
            PipelineNode(
                id="performance_monitor",
                name="Performance Monitor",
                node_type=NodeType.VALIDATOR,
                description="Performance profiling and monitoring",
            ),
            PipelineNode(
                id="report_generator",
                name="Report Generator",
                node_type=NodeType.TRANSFORMER,
                description="Automated report generation",
            ),
            PipelineNode(
                id="visualization_engine",
                name="Visualization Engine",
                node_type=NodeType.TRANSFORMER,
                description="Charts, graphs, and interactive dashboards",
            ),
            PipelineNode(
                id="output_layer",
                name="Output Layer",
                node_type=NodeType.OUTPUT,
                description="Final outputs and reports",
                metadata={"formats": ["PDF", "HTML", "JSON", "CSV"]},
            ),
        ]

        edges = [
            PipelineEdge("user_interface", "config_manager", "Load configuration"),
            PipelineEdge("config_manager", "data_layer", "Configure data processing"),
            PipelineEdge("data_layer", "apgi_core", "Provide processed data"),
            PipelineEdge("apgi_core", "validation_engine", "Validate results"),
            PipelineEdge("apgi_core", "performance_monitor", "Monitor performance"),
            PipelineEdge("validation_engine", "report_generator", "Provide validation results"),
            PipelineEdge("performance_monitor", "report_generator", "Provide performance data"),
            PipelineEdge("report_generator", "visualization_engine", "Create visualizations"),
            PipelineEdge("visualization_engine", "output_layer", "Generate outputs"),
        ]

        return PipelineDiagram(
            name="APGI System Architecture",
            description="High-level system architecture overview",
            nodes=nodes,
            edges=edges,
            layout="hierarchical",
        )

    def calculate_layout(self, diagram: PipelineDiagram) -> Dict[str, Tuple[float, float]]:
        """Calculate node positions for the diagram."""
        G = nx.DiGraph()

        # Add nodes and edges
        for node in diagram.nodes:
            G.add_node(node.id)

        for edge in diagram.edges:
            G.add_edge(edge.source, edge.target)

        # Calculate layout
        if diagram.layout == "hierarchical":
            pos = nx.spring_layout(G, k=3, iterations=50)
        elif diagram.layout == "circular":
            pos = nx.circular_layout(G)
        elif diagram.layout == "random":
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)

        return pos

    def draw_diagram(
        self,
        diagram: PipelineDiagram,
        save_path: Optional[Path] = None,
        figsize: Tuple[int, int] = (14, 10),
        show_labels: bool = True,
        show_descriptions: bool = True,
    ) -> Path:
        """Draw pipeline diagram."""

        if save_path is None:
            save_path = (
                self.output_dir
                / f"{diagram.name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

        # Calculate layout
        positions = self.calculate_layout(diagram)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(diagram.description, fontsize=16, fontweight="bold", pad=20)

        # Draw edges
        for edge in diagram.edges:
            if edge.source in positions and edge.target in positions:
                source_pos = positions[edge.source]
                target_pos = positions[edge.target]

                # Draw edge
                ax.annotate(
                    "",
                    xy=target_pos,
                    xytext=source_pos,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="gray",
                        lw=1.5,
                        alpha=0.7,
                        connectionstyle="arc3,rad=0.1",
                    ),
                )

                # Add edge label
                if show_descriptions and edge.description:
                    mid_x = (source_pos[0] + target_pos[0]) / 2
                    mid_y = (source_pos[1] + target_pos[1]) / 2
                    ax.text(
                        mid_x,
                        mid_y,
                        edge.description,
                        fontsize=8,
                        ha="center",
                        va="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    )

        # Draw nodes
        for node in diagram.nodes:
            if node.id in positions:
                pos = positions[node.id]
                color = self.colors[node.node_type]

                # Determine node shape
                if node.node_type in [NodeType.INPUT, NodeType.OUTPUT]:
                    # Ellipse for input/output
                    circle = plt.Circle(pos, 0.08, color=color, alpha=0.8, transform=ax.transData)
                    ax.add_patch(circle)
                elif node.node_type == NodeType.DECISION:
                    # Diamond for decision nodes
                    diamond = patches.RegularPolygon(
                        pos,
                        4,
                        radius=0.08,
                        orientation=np.pi / 4,
                        color=color,
                        alpha=0.8,
                        transform=ax.transData,
                    )
                    ax.add_patch(diamond)
                elif node.node_type == NodeType.MERGE:
                    # Circle for merge nodes
                    circle = plt.Circle(pos, 0.06, color=color, alpha=0.8, transform=ax.transData)
                    ax.add_patch(circle)
                else:
                    # Rectangle for other nodes
                    rect = FancyBboxPatch(
                        (pos[0] - 0.08, pos[1] - 0.04),
                        0.16,
                        0.08,
                        boxstyle="round,pad=0.02",
                        color=color,
                        alpha=0.8,
                        transform=ax.transData,
                    )
                    ax.add_patch(rect)

                # Add node label
                if show_labels:
                    ax.text(
                        pos[0],
                        pos[1] + 0.12,
                        node.name,
                        fontsize=9,
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                    # Add description if enabled
                    if show_descriptions and node.description:
                        # Split long descriptions
                        words = node.description.split()
                        lines = []
                        current_line = []
                        for word in words:
                            current_line.append(word)
                            if len(" ".join(current_line)) > 20:
                                lines.append(" ".join(current_line[:-1]))
                                current_line = [word]
                        if current_line:
                            lines.append(" ".join(current_line))

                        for i, line in enumerate(lines[:2]):  # Max 2 lines
                            ax.text(
                                pos[0],
                                pos[1] - 0.06 - i * 0.03,
                                line,
                                fontsize=7,
                                ha="center",
                                va="top",
                                style="italic",
                            )

        # Add legend
        legend_elements = []
        for node_type, color in self.colors.items():
            legend_elements.append(patches.Patch(color=color, label=node_type.value.title()))
        ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

        # Set axis properties
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add grid
        ax.grid(True, alpha=0.1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        apgi_logger.logger.info(f"Pipeline diagram saved: {save_path}")
        return save_path

    def create_interactive_diagram(self, diagram: PipelineDiagram) -> Dict[str, Any]:
        """Create interactive diagram data for web visualization."""

        positions = self.calculate_layout(diagram)

        # Prepare nodes
        nodes_data = []
        for node in diagram.nodes:
            pos = positions.get(node.id, (0, 0))
            nodes_data.append(
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.node_type.value,
                    "description": node.description,
                    "parameters": node.parameters,
                    "metadata": node.metadata,
                    "position": {"x": pos[0], "y": pos[1]},
                    "color": self.colors[node.node_type],
                }
            )

        # Prepare edges
        edges_data = []
        for edge in diagram.edges:
            edges_data.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "description": edge.description,
                    "data_type": edge.data_type,
                    "condition": edge.condition,
                }
            )

        return {
            "name": diagram.name,
            "description": diagram.description,
            "layout": diagram.layout,
            "nodes": nodes_data,
            "edges": edges_data,
            "metadata": diagram.metadata,
        }

    def generate_pipeline_documentation(self, diagram: PipelineDiagram) -> Path:
        """Generate pipeline documentation."""

        doc_path = self.output_dir / f"{diagram.name.replace(' ', '_').lower()}_documentation.md"

        with open(doc_path, "w") as f:
            f.write(f"# {diagram.name}\n\n")
            f.write(f"{diagram.description}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Pipeline Components\n\n")

            # Group nodes by type
            nodes_by_type = defaultdict(list)
            for node in diagram.nodes:
                nodes_by_type[node.node_type].append(node)

            for node_type, nodes in nodes_by_type.items():
                f.write(f"### {node_type.value.title()}\n\n")
                for node in nodes:
                    f.write(f"#### {node.name}\n")
                    f.write(f"- **ID**: {node.id}\n")
                    f.write(f"- **Description**: {node.description}\n")
                    if node.parameters:
                        f.write(f"- **Parameters**:\n")
                        for param, value in node.parameters.items():
                            f.write(f"  - `{param}`: {value}\n")
                    if node.metadata:
                        f.write("- **Metadata**:\n")
                        for key, value in node.metadata.items():
                            f.write(f"  - `{key}`: {value}\n")
                    f.write("\n")

            f.write("## Data Flow\n\n")
            f.write("| Source | Target | Description | Data Type |\n")
            f.write("|--------|--------|-------------|-----------|\n")

            for edge in diagram.edges:
                condition = f" (if {edge.condition})" if edge.condition else ""
                f.write(
                    f"| {edge.source} | {edge.target} | {edge.description}{condition} | {edge.data_type} |\n"
                )

            f.write("\n## Pipeline Statistics\n\n")
            f.write(f"- **Total Nodes**: {len(diagram.nodes)}\n")
            f.write(f"- **Total Edges**: {len(diagram.edges)}\n")
            f.write(f"- **Node Types**: {list(nodes_by_type.keys())}\n")
            f.write(f"- **Layout**: {diagram.layout}\n")

        apgi_logger.logger.info(f"Pipeline documentation generated: {doc_path}")
        return doc_path

    def create_all_diagrams(self) -> Dict[str, Path]:
        """Create all standard pipeline diagrams."""

        diagrams = {
            "preprocessing": self.create_preprocessing_pipeline_diagram(),
            "validation": self.create_validation_pipeline_diagram(),
            "architecture": self.create_system_architecture_diagram(),
        }

        generated_diagrams = {}

        for name, diagram in diagrams.items():
            # Draw static diagram
            diagram_path = self.draw_diagram(diagram)
            generated_diagrams[f"{name}_diagram"] = diagram_path

            # Generate documentation
            doc_path = self.generate_pipeline_documentation(diagram)
            generated_diagrams[f"{name}_documentation"] = doc_path

            # Create interactive data
            interactive_data = self.create_interactive_diagram(diagram)
            interactive_path = self.output_dir / f"{name}_interactive.json"
            with open(interactive_path, "w") as f:
                json.dump(interactive_data, f, indent=2)
            generated_diagrams[f"{name}_interactive"] = interactive_path

        return generated_diagrams


# Global pipeline visualizer instance
pipeline_visualizer = PipelineVisualizer()


# Convenience functions
def create_preprocessing_diagram() -> Path:
    """Create preprocessing pipeline diagram."""
    diagram = pipeline_visualizer.create_preprocessing_pipeline_diagram()
    return pipeline_visualizer.draw_diagram(diagram)


def create_validation_diagram() -> Path:
    """Create validation pipeline diagram."""
    diagram = pipeline_visualizer.create_validation_pipeline_diagram()
    return pipeline_visualizer.draw_diagram(diagram)


def create_architecture_diagram() -> Path:
    """Create system architecture diagram."""
    diagram = pipeline_visualizer.create_system_architecture_diagram()
    return pipeline_visualizer.draw_diagram(diagram)


def generate_all_pipeline_visualizations() -> Dict[str, Path]:
    """Generate all pipeline visualizations."""
    return pipeline_visualizer.create_all_diagrams()
