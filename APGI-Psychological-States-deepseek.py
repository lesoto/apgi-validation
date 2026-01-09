"""
APGI Psychological State Parameter Library with Advanced Visualizations
=============================================================================

Complete parameter mappings for 51 psychological states based on the
Active Posterior Global Integration (APGI) framework.

This enhanced version includes:
1. Modern, eloquent data visualizations for psychological state analysis
2. Interactive 3D state mapping with force-directed layouts
3. Temporal dynamics and transition pathway visualizations
4. Multi-dimensional comparison tools with innovative visual metaphors
5. Export capabilities for presentations and publications

=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum, auto
import warnings
from math import pi
import datetime

# Visualization imports with graceful fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio

    PLOTLY_AVAILABLE = True
    # Set modern theme
    pio.templates.default = "plotly_white+plotly_dark"
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    from matplotlib.patches import Circle, Wedge, Polygon
    import matplotlib.cm as cm

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available. Install with: pip install pandas")

# GUI imports with graceful fallbacks
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    from tkinter.scrolledtext import ScrolledText
    from traceback import format_exc
    import warnings
    import os
    import tempfile
    import threading
    from typing import Dict, List, Optional, Tuple, Union, Any

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    warnings.warn("Tkinter not available for GUI interface")


@dataclass
class APGIParameters:
    """APGI parameter set with proper type safety"""

    Pi_e: float  # Exteroceptive precision ∈ [0.1, 10]
    Pi_i_baseline: float  # Baseline interoceptive precision ∈ [0.1, 10]
    Pi_i_eff: float  # Effective interoceptive precision (modulated)
    theta_t: float  # Ignition threshold (z-score)
    S_t: float  # Accumulated surprise signal
    M_ca: float  # Somatic marker value ∈ [-2, +2]
    beta: float  # Somatic influence gain ∈ [0.3, 0.8]
    z_e: float  # Exteroceptive z-score
    z_i: float  # Interoceptive z-score

    def __post_init__(self):
        """Validate parameters are within physiological bounds"""
        assert 0.1 <= self.Pi_e <= 10.0, f"Pi_e must be in [0.1, 10], got {self.Pi_e}"
        assert (
            0.1 <= self.Pi_i_baseline <= 10.0
        ), f"Pi_i_baseline must be in [0.1, 10], got {self.Pi_i_baseline}"
        assert (
            0.1 <= self.Pi_i_eff <= 10.0
        ), f"Pi_i_eff must be in [0.1, 10], got {self.Pi_i_eff}"
        assert -2.0 <= self.M_ca <= 2.0, f"M_ca must be in [-2, 2], got {self.M_ca}"
        assert 0.3 <= self.beta <= 0.8, f"beta must be in [0.3, 0.8], got {self.beta}"

    def compute_ignition_probability(self) -> float:
        """Compute P(ignite) = σ(S_t - θ_t)"""
        return 1.0 / (1.0 + np.exp(-(self.S_t - self.theta_t)))

    def verify_S_t(self) -> bool:
        """Verify S_t matches the formula: S_t = Π_e·|z_e| + Π_i_eff·|z_i|"""
        computed = self.Pi_e * abs(self.z_e) + self.Pi_i_eff * abs(self.z_i)
        return np.isclose(self.S_t, computed, rtol=0.01)

    def verify_Pi_i_eff(self) -> bool:
        """Verify Π_i_eff matches the formula: Π_i_eff = Π_i_baseline · exp(β·M)"""
        computed = self.Pi_i_baseline * np.exp(self.beta * self.M_ca)
        computed = np.clip(computed, 0.1, 10.0)
        return np.isclose(self.Pi_i_eff, computed, rtol=0.05)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for visualization"""
        return {
            "Pi_e": self.Pi_e,
            "Pi_i_baseline": self.Pi_i_baseline,
            "Pi_i_eff": self.Pi_i_eff,
            "theta_t": self.theta_t,
            "S_t": self.S_t,
            "M_ca": self.M_ca,
            "beta": self.beta,
            "z_e": self.z_e,
            "z_i": self.z_i,
            "ignition_probability": self.compute_ignition_probability(),
        }


@dataclass
class PsychologicalState:
    """Extended state representation with metadata"""

    name: str
    parameters: APGIParameters
    category: str
    description: str = ""
    phenomenology: List[str] = field(default_factory=list)
    distinguishing_features: Dict[str, str] = field(default_factory=dict)
    pathological_variant: Optional[str] = None
    temporal_dynamics: Optional[str] = None
    color: Optional[str] = None  # For visualization


class StateCategory(Enum):
    """Categories of psychological states with colors"""

    OPTIMAL_FUNCTIONING = ("#2E86AB", "Optimal Functioning")  # Calming blue
    POSITIVE_AFFECTIVE = ("#48BF84", "Positive Affective")  # Growth green
    COGNITIVE_ATTENTIONAL = ("#FF9F1C", "Cognitive/Attentional")  # Focus orange
    AVERSIVE_AFFECTIVE = ("#E63946", "Aversive Affective")  # Alert red
    PATHOLOGICAL_EXTREME = ("#7209B7", "Pathological/Extreme")  # Purple
    ALTERED_BOUNDARY = ("#8338EC", "Altered/Boundary")  # Mystic purple
    TRANSITIONAL_CONTEXTUAL = ("#06D6A0", "Transitional/Contextual")  # Teal
    UNELABORATED = ("#8D99AE", "Unelaborated")  # Neutral gray

    def __init__(self, color: str, display_name: str):
        self.color = color
        self.display_name = display_name


# =============================================================================
# ENHANCED VISUALIZATION ENGINE
# =============================================================================


class APGIVisualizer:
    """Modern, eloquent visualizations for APGI psychological states"""

    # Color palettes inspired by psychological states
    PALETTES = {
        "categorical": {
            StateCategory.OPTIMAL_FUNCTIONING: "#2E86AB",
            StateCategory.POSITIVE_AFFECTIVE: "#48BF84",
            StateCategory.COGNITIVE_ATTENTIONAL: "#FF9F1C",
            StateCategory.AVERSIVE_AFFECTIVE: "#E63946",
            StateCategory.PATHOLOGICAL_EXTREME: "#7209B7",
            StateCategory.ALTERED_BOUNDARY: "#8338EC",
            StateCategory.TRANSITIONAL_CONTEXTUAL: "#06D6A0",
            StateCategory.UNELABORATED: "#8D99AE",
        },
        "sequential": [
            "#003f5c",
            "#2f4b7c",
            "#665191",
            "#a05195",
            "#d45087",
            "#f95d6a",
            "#ff7c43",
            "#ffa600",
        ],
        "diverging": [
            "#2166ac",
            "#4393c3",
            "#92c5de",
            "#d1e5f0",
            "#f7f7f7",
            "#fddbc7",
            "#f4a582",
            "#d6604d",
            "#b2182b",
        ],
    }

    def __init__(
        self,
        states_dict: Dict[str, APGIParameters],
        categories_dict: Dict[str, StateCategory],
    ):
        """
        Initialize visualizer with states and categories.

        Args:
            states_dict: Dictionary of state names to APGIParameters
            categories_dict: Dictionary of state names to StateCategory
        """
        self.states = states_dict
        self.categories = categories_dict

        if PANDAS_AVAILABLE:
            self.df = self._create_dataframe()
        else:
            self.df = None

    def _create_dataframe(self) -> pd.DataFrame:
        """Create a pandas DataFrame for visualization"""
        data = []
        for name, params in self.states.items():
            row = params.to_dict()
            row["name"] = name
            row["category"] = self.categories.get(name, StateCategory.UNELABORATED).name
            row["category_display"] = self.categories.get(
                name, StateCategory.UNELABORATED
            ).display_name
            row["category_color"] = self.categories.get(
                name, StateCategory.UNELABORATED
            ).color
            data.append(row)

        df = pd.DataFrame(data)

        # Add derived metrics
        df["precision_ratio"] = df["Pi_i_eff"] / df["Pi_e"]
        df["somatic_engagement"] = df["M_ca"] * df["beta"]
        df["prediction_error_total"] = df["z_e"] + df["z_i"]

        return df

    def plot_state_network_3d(
        self,
        dimension1: str = "Pi_e",
        dimension2: str = "Pi_i_eff",
        dimension3: str = "theta_t",
        show_transitions: bool = True,
        transition_threshold: float = 5.0,
        save_path: Optional[str] = None,
    ) -> Optional[go.Figure]:
        """
        Create an interactive 3D network visualization of psychological states.

        Args:
            dimension1: First parameter for x-axis
            dimension2: Second parameter for y-axis
            dimension3: Third parameter for z-axis
            show_transitions: Whether to show transition pathways
            transition_threshold: Maximum transition cost to show edges
            save_path: Path to save HTML file

        Returns:
            Plotly Figure object if Plotly available, else None
        """
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            warnings.warn("Plotly or Pandas not available for 3D network visualization")
            return None

        fig = go.Figure()

        # Create node positions using force-directed layout simulation
        positions = self._compute_force_layout()

        # Add nodes (states)
        for idx, row in self.df.iterrows():
            state_name = row["name"]
            x, y, z = positions.get(
                state_name, (row[dimension1], row[dimension2], row[dimension3])
            )

            # Node properties
            size = 15 + (row["S_t"] * 2)  # Scale by surprise
            color = row["category_color"]

            # Add node trace
            fig.add_trace(
                go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode="markers+text",
                    marker=dict(
                        size=size,
                        color=color,
                        opacity=0.8,
                        line=dict(width=2, color="white"),
                        symbol="circle",
                    ),
                    text=[state_name.replace("_", " ").title()],
                    textposition="top center",
                    hoverinfo="text",
                    hovertext=self._create_hover_text(state_name, row),
                    name=state_name,
                    showlegend=False,
                )
            )

        # Add transition edges if requested
        if show_transitions:
            edges = self._compute_transition_edges(transition_threshold)
            for from_state, to_state, cost in edges:
                if from_state in positions and to_state in positions:
                    x1, y1, z1 = positions[from_state]
                    x2, y2, z2 = positions[to_state]

                    # Edge properties
                    width = max(1, 5 - cost)  # Thicker for easier transitions
                    opacity = max(0.1, 0.8 - cost / 10)

                    fig.add_trace(
                        go.Scatter3d(
                            x=[x1, x2, None],
                            y=[y1, y2, None],
                            z=[z1, z2, None],
                            mode="lines",
                            line=dict(
                                width=width,
                                color="rgba(150, 150, 150, {})".format(opacity),
                                dash="dot" if cost > 3 else "solid",
                            ),
                            hoverinfo="none",
                            showlegend=False,
                        )
                    )

        # Layout configuration
        fig.update_layout(
            title="APGI Psychological State Network",
            scene=dict(
                xaxis_title=dimension1,
                yaxis_title=dimension2,
                zaxis_title=dimension3,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, b=0, t=40),
            hovermode="closest",
        )

        # Add category legend
        self._add_category_legend(fig)

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_ignition_landscape(
        self,
        focus_state: Optional[str] = None,
        parameter1: str = "Pi_e",
        parameter2: str = "theta_t",
        resolution: int = 50,
        save_path: Optional[str] = None,
    ) -> Optional[go.Figure]:
        """
        Create a 3D ignition probability landscape.

        Args:
            focus_state: State to highlight on landscape
            parameter1: First parameter for x-axis
            parameter2: Second parameter for y-axis
            resolution: Grid resolution for landscape
            save_path: Path to save visualization

        Returns:
            Plotly Figure object if available
        """
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available for ignition landscape")
            return None

        # Create parameter grid
        p1_range = np.linspace(
            self.df[parameter1].min() * 0.8, self.df[parameter1].max() * 1.2, resolution
        )
        p2_range = np.linspace(
            self.df[parameter2].min() * 0.8, self.df[parameter2].max() * 1.2, resolution
        )

        P1, P2 = np.meshgrid(p1_range, p2_range)

        # Compute ignition probability (simplified formula)
        # Using S_t ~ Pi_e * |z_e| + Pi_i_eff * |z_i| with average errors
        avg_z_e = self.df["z_e"].mean()
        avg_z_i = self.df["z_i"].mean()
        avg_Pi_i_eff = self.df["Pi_i_eff"].mean()

        # For landscape, assume fixed other parameters at average values
        S_t = P1 * avg_z_e + avg_Pi_i_eff * avg_z_i
        Z = 1.0 / (1.0 + np.exp(-(S_t - P2)))  # Ignition probability

        fig = go.Figure(
            data=[
                go.Surface(
                    z=Z,
                    x=P1,
                    y=P2,
                    colorscale="Viridis",
                    opacity=0.8,
                    contours={
                        "z": {
                            "show": True,
                            "usecolormap": True,
                            "highlightcolor": "limegreen",
                            "project": {"z": True},
                        }
                    },
                    hoverongaps=False,
                    hovertemplate="%{x} vs %{y}<br>Ignition Probability: %{z:.3f}<extra></extra>",
                )
            ]
        )

        # Add state markers
        scatter_x = []
        scatter_y = []
        scatter_z = []
        scatter_colors = []
        scatter_names = []

        for idx, row in self.df.iterrows():
            # Compute ignition probability for actual state
            S_t_actual = row["Pi_e"] * row["z_e"] + row["Pi_i_eff"] * row["z_i"]
            ignition_prob = 1.0 / (1.0 + np.exp(-(S_t_actual - row["theta_t"])))

            scatter_x.append(row[parameter1])
            scatter_y.append(row[parameter2])
            scatter_z.append(ignition_prob)
            scatter_colors.append(row["category_color"])
            scatter_names.append(row["name"])

        fig.add_trace(
            go.Scatter3d(
                x=scatter_x,
                y=scatter_y,
                z=scatter_z,
                mode="markers+text",
                marker=dict(
                    size=8,
                    color=scatter_colors,
                    opacity=1.0,
                    line=dict(width=2, color="white"),
                ),
                text=[name.replace("_", " ").title() for name in scatter_names],
                textposition="top center",
                hoverinfo="text",
                hovertext=[
                    f"{name}<br>P(ignition)={z:.2%}"
                    for name, z in zip(scatter_names, scatter_z)
                ],
                name="Psychological States",
            )
        )

        # Highlight focus state if provided
        if focus_state and focus_state in self.states:
            focus_idx = self.df[self.df["name"] == focus_state].index[0]
            focus_row = self.df.iloc[focus_idx]

            fig.add_trace(
                go.Scatter3d(
                    x=[focus_row[parameter1]],
                    y=[focus_row[parameter2]],
                    z=[focus_row["ignition_probability"]],
                    mode="markers",
                    marker=dict(
                        size=20,
                        color="gold",
                        opacity=1.0,
                        line=dict(width=3, color="white"),
                        symbol="diamond",
                    ),
                    name=f"Focus: {focus_state}",
                    hoverinfo="text",
                    hovertext=f"<b>{focus_state}</b><br>"
                    f"P(ignition)={focus_row['ignition_probability']:.2%}<br>"
                    f"{parameter1}={focus_row[parameter1]:.2f}<br>"
                    f"{parameter2}={focus_row[parameter2]:.2f}",
                )
            )

        # Layout
        fig.update_layout(
            title="Ignition Probability Landscape",
            scene=dict(
                xaxis_title=parameter1,
                yaxis_title=parameter2,
                zaxis_title="P(Ignition)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=40),
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_transition_pathways(
        self,
        from_state: str,
        to_state: str,
        show_intermediate: bool = True,
        save_path: Optional[str] = None,
    ) -> Optional[go.Figure]:
        """
        Visualize transition pathways between states.

        Args:
            from_state: Starting state name
            to_state: Target state name
            show_intermediate: Show intermediate suggested states
            save_path: Path to save visualization

        Returns:
            Plotly Figure object if available
        """
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available for transition pathways")
            return None

        # Generate pathway
        pathway = get_transition_pathway(from_state, to_state)

        if not show_intermediate:
            pathway = [from_state, to_state]

        # Create parallel coordinates plot for parameter changes
        dimensions = []
        param_names = [
            "Pi_e",
            "Pi_i_eff",
            "theta_t",
            "M_ca",
            "S_t",
            "ignition_probability",
        ]

        for param in param_names:
            values = []
            for state in pathway:
                if param == "ignition_probability":
                    values.append(self.states[state].compute_ignition_probability())
                else:
                    values.append(getattr(self.states[state], param))

            dimensions.append(
                dict(
                    range=[min(values) * 0.9, max(values) * 1.1],
                    label=param,
                    values=values,
                )
            )

        # Colors for states in pathway
        colors = [
            self.categories.get(state, StateCategory.UNELABORATED).color
            for state in pathway
        ]

        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=colors,
                    colorscale=[[0, colors[0]], [1, colors[-1]]],
                    showscale=True,
                    cmin=0,
                    cmax=1,
                ),
                dimensions=dimensions,
            )
        )

        # Add pathway visualization as network
        network_fig = go.Figure()

        # Node positions in a circular layout
        angles = np.linspace(0, 2 * np.pi, len(pathway), endpoint=False)
        radius = 5
        x_pos = radius * np.cos(angles)
        y_pos = radius * np.sin(angles)

        for i, (state, x, y) in enumerate(zip(pathway, x_pos, y_pos)):
            color = self.categories.get(state, StateCategory.UNELABORATED).color

            network_fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(
                        size=30, color=color, line=dict(width=2, color="white")
                    ),
                    text=[state.replace("_", " ").title()],
                    textposition="top center",
                    name=state,
                    hoverinfo="text",
                    hovertext=f"<b>{state}</b><br>"
                    f"Step {i+1}/{len(pathway)}<br>"
                    f"Category: {self.categories[state].display_name}",
                )
            )

        # Add edges with arrows
        for i in range(len(pathway) - 1):
            network_fig.add_annotation(
                x=x_pos[i + 1],
                y=y_pos[i + 1],
                ax=x_pos[i],
                ay=y_pos[i],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(100, 100, 100, 0.5)",
            )

        network_fig.update_layout(
            title=f"Transition Pathway: {from_state} → {to_state}",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, b=0, t=40),
        )

        # Create subplot figure
        final_fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Parameter Evolution", "Transition Pathway"),
            specs=[[{"type": "parcoords"}, {"type": "scatter"}]],
        )

        # Add traces to subplot
        final_fig.add_trace(fig.data[0], row=1, col=1)

        for trace in network_fig.data:
            final_fig.add_trace(trace, row=1, col=2)

        # Update layout
        final_fig.update_layout(
            title_text=f"State Transition Analysis: {from_state} → {to_state}",
            showlegend=False,
            height=500,
        )

        if save_path:
            final_fig.write_html(save_path)

        return final_fig

    def plot_state_radar(
        self,
        state_names: List[str],
        normalize: bool = True,
        save_path: Optional[str] = None,
    ) -> Optional[go.Figure]:
        """
        Create a radar chart comparing multiple states.

        Args:
            state_names: List of state names to compare
            normalize: Whether to normalize parameters to [0, 1]
            save_path: Path to save visualization

        Returns:
            Plotly Figure object if available
        """
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            warnings.warn("Plotly or Pandas not available for radar chart")
            return None

        # Parameters to include in radar
        params = ["Pi_e", "Pi_i_eff", "theta_t", "M_ca", "S_t", "z_e", "z_i", "beta"]

        # Prepare data
        categories = params
        fig = go.Figure()

        for state_name in state_names:
            if state_name not in self.states:
                continue

            params_obj = self.states[state_name]
            values = []

            for param in params:
                value = getattr(params_obj, param)

                if normalize:
                    # Normalize based on all states' ranges
                    all_values = self.df[param]
                    if param in ["theta_t", "M_ca"]:  # These can be negative
                        value = (value - all_values.min()) / (
                            all_values.max() - all_values.min()
                        )
                    else:
                        value = value / all_values.max()

                values.append(value)

            # Close the radar by repeating first value
            values.append(values[0])

            color = self.categories.get(state_name, StateCategory.UNELABORATED).color

            # Convert color to rgba format with transparency if it's not already in that format
            if color.startswith("#"):
                if len(color) == 7:  # #RRGGBB format
                    fill_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.25)"
                    line_color = color
                elif len(color) == 9:  # #RRGGBBAA format
                    fill_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {int(color[7:9], 16)/255:.2f})"
                    line_color = f"#{color[1:7]}"  # Remove alpha for line color
                else:
                    fill_color = (
                        f"rgba(128, 128, 128, 0.25)"  # Fallback gray with transparency
                    )
                    line_color = "#808080"
            else:
                fill_color = (
                    f"rgba(128, 128, 128, 0.25)"  # Fallback gray with transparency
                )
                line_color = "#808080"

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],  # Close the shape
                    fill="toself",
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=2),
                    name=state_name.replace("_", " ").title(),
                    hoverinfo="text",
                    hovertext=self._create_hover_text(state_name, None),
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.1] if normalize else None)
            ),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.1),
            title="State Comparison Radar",
            margin=dict(l=100, r=100, b=50, t=50),
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_parameter_correlation_heatmap(
        self, parameters: Optional[List[str]] = None, save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Create a correlation heatmap of APGI parameters.

        Args:
            parameters: List of parameters to include (default: all)
            save_path: Path to save visualization

        Returns:
            Plotly Figure object if available
        """
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            warnings.warn("Plotly or Pandas not available for heatmap")
            return None

        if parameters is None:
            parameters = [
                "Pi_e",
                "Pi_i_baseline",
                "Pi_i_eff",
                "theta_t",
                "S_t",
                "M_ca",
                "beta",
                "z_e",
                "z_i",
                "ignition_probability",
            ]

        # Calculate correlation matrix
        corr_matrix = self.df[parameters].corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=parameters,
                y=parameters,
                colorscale="RdBu",
                zmid=0,
                hoverongaps=False,
                hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="APGI Parameter Correlation Matrix",
            xaxis_title="Parameters",
            yaxis_title="Parameters",
            width=700,
            height=600,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def create_state_summary_dashboard(
        self, state_name: str, save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Create a comprehensive dashboard for a single state.

        Args:
            state_name: State to visualize
            save_path: Path to save HTML dashboard

        Returns:
            Plotly Figure object if available
        """
        if not PLOTLY_AVAILABLE or state_name not in self.states:
            warnings.warn("Plotly not available or state not found")
            return None

        params = self.states[state_name]
        category = self.categories.get(state_name, StateCategory.UNELABORATED)

        # Create subplots: 2x2 grid
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Parameter Profile",
                "Ignition Dynamics",
                "Category Comparison",
                "Transition Analysis",
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "radar"}, {"type": "network"}],
            ],
        )

        # 1. Parameter Profile (Bar Chart)
        param_names = [
            "Pi_e",
            "Pi_i_baseline",
            "Pi_i_eff",
            "theta_t",
            "M_ca",
            "beta",
            "z_e",
            "z_i",
        ]
        param_values = [getattr(params, p) for p in param_names]
        param_colors = ["#2E86AB" if v > 0 else "#E63946" for v in param_values]

        fig.add_trace(
            go.Bar(
                x=param_names,
                y=param_values,
                marker_color=param_colors,
                name="Parameters",
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 2. Ignition Dynamics (Scatter with threshold line)
        # Simulate S_t variations
        S_t_range = np.linspace(0, params.S_t * 2, 100)
        ignition_probs = 1.0 / (1.0 + np.exp(-(S_t_range - params.theta_t)))

        fig.add_trace(
            go.Scatter(
                x=S_t_range,
                y=ignition_probs,
                mode="lines",
                line=dict(color=category.color, width=3),
                name="Ignition Probability",
                fill="tozeroy",
                fillcolor=category.color + "20",
            ),
            row=1,
            col=2,
        )

        # Add current state marker
        fig.add_trace(
            go.Scatter(
                x=[params.S_t],
                y=[params.compute_ignition_probability()],
                mode="markers",
                marker=dict(size=15, color="gold", line=dict(width=2, color="black")),
                name="Current State",
                hovertext=f"S_t={params.S_t:.2f}, P={params.compute_ignition_probability():.2%}",
            ),
            row=1,
            col=2,
        )

        # 3. Category Comparison (Radar)
        category_states = [
            name
            for name, cat in self.categories.items()
            if cat == category and name != state_name
        ][:4]
        if category_states:
            radar_data = []

            for comp_state in [state_name] + category_states:
                comp_params = self.states[comp_state]
                values = [
                    comp_params.Pi_e / 10,  # Normalize to [0, 1]
                    comp_params.Pi_i_eff / 10,
                    (comp_params.theta_t + 3) / 6,  # Normalize [-3, 3] to [0, 1]
                    (comp_params.M_ca + 2) / 4,  # Normalize [-2, 2] to [0, 1]
                    comp_params.compute_ignition_probability(),
                ]
                values.append(values[0])  # Close radar

                radar_data.append(values)

            radar_params = ["Π_e", "Π_i_eff", "θ_t", "M_ca", "P(ign)"]

            for i, (comp_state, values) in enumerate(
                zip([state_name] + category_states, radar_data)
            ):
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=radar_params + [radar_params[0]],
                        fill="toself" if i == 0 else "none",
                        line=dict(width=2 if i == 0 else 1),
                        opacity=1.0 if i == 0 else 0.6,
                        name=comp_state.replace("_", " ").title(),
                        showlegend=True,
                    ),
                    row=2,
                    col=1,
                )

        # 4. Transition Network (Simplified)
        # Find nearest states for transitions
        all_states = list(self.states.keys())
        all_states.remove(state_name)

        # Calculate distances to other states
        distances = []
        for other_state in all_states[:6]:  # Limit to 6 closest
            cost_dict = compute_transition_cost(state_name, other_state)
            distances.append((other_state, cost_dict["total"]))

        # Sort by distance
        distances.sort(key=lambda x: x[1])
        closest_states = [state_name] + [d[0] for d in distances[:3]]

        # Simple network layout
        angles = np.linspace(0, 2 * np.pi, len(closest_states), endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)

        for i, (state, x, y) in enumerate(zip(closest_states, x_pos, y_pos)):
            state_color = self.categories.get(state, StateCategory.UNELABORATED).color

            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(
                        size=30 if state == state_name else 20,
                        color=state_color,
                        line=dict(width=3 if state == state_name else 1, color="white"),
                    ),
                    text=[state.replace("_", " ").title()],
                    textposition="top center",
                    name=state,
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

        # Add edges
        for i in range(1, len(closest_states)):
            fig.add_annotation(
                x=x_pos[i],
                y=y_pos[i],
                ax=x_pos[0],
                ay=y_pos[0],
                xref="x4",
                yref="y4",
                axref="x4",
                ayref="y4",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(100, 100, 100, 0.5)",
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title_text=f"APGI State Dashboard: {state_name.replace('_', ' ').title()}",
            showlegend=True,
            height=800,
            template="plotly_white",
        )

        # Update axes for each subplot
        fig.update_xaxes(title_text="Parameters", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)

        fig.update_xaxes(title_text="Accumulated Surprise (S_t)", row=1, col=2)
        fig.update_yaxes(title_text="Ignition Probability", range=[0, 1], row=1, col=2)

        fig.update_polars(radialaxis_range=[0, 1], row=2, col=1)

        fig.update_xaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=2, col=2
        )
        fig.update_yaxes(
            showgrid=False, zeroline=False, showticklabels=False, row=2, col=2
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _compute_force_layout(self) -> Dict[str, Tuple[float, float, float]]:
        """Compute force-directed layout positions for states"""
        positions = {}

        # Simple layout based on parameter similarity
        for i, (name, params) in enumerate(self.states.items()):
            # Use parameters to determine position
            x = params.Pi_e + params.z_e * 0.5
            y = params.Pi_i_eff + params.z_i * 0.5
            z = params.theta_t + params.M_ca

            # Add some noise for visual separation
            x += np.random.normal(0, 0.3)
            y += np.random.normal(0, 0.3)
            z += np.random.normal(0, 0.2)

            positions[name] = (x, y, z)

        return positions

    def _compute_transition_edges(
        self, max_cost: float = 5.0
    ) -> List[Tuple[str, str, float]]:
        """Compute transition edges between states"""
        edges = []
        state_names = list(self.states.keys())

        # Limit to avoid O(n²) computation for large state sets
        for i in range(min(20, len(state_names))):  # Sample first 20 states
            for j in range(i + 1, min(20, len(state_names))):
                cost_dict = compute_transition_cost(state_names[i], state_names[j])
                cost = cost_dict["total"]

                if cost <= max_cost:
                    edges.append((state_names[i], state_names[j], cost))

        return edges

    def _create_hover_text(
        self, state_name: str, row: Optional[pd.Series] = None
    ) -> str:
        """Create hover text for state visualization"""
        params = self.states[state_name]

        if row is None and PANDAS_AVAILABLE:
            row = self.df[self.df["name"] == state_name].iloc[0]

        text = f"<b>{state_name.replace('_', ' ').title()}</b><br>"
        text += f"Category: {self.categories[state_name].display_name}<br>"
        text += f"Π_e: {params.Pi_e:.2f}<br>"
        text += f"Π_i_eff: {params.Pi_i_eff:.2f}<br>"
        text += f"θ_t: {params.theta_t:+.2f}<br>"
        text += f"M_ca: {params.M_ca:+.2f}<br>"
        text += f"S_t: {params.S_t:.2f}<br>"
        text += f"P(ignition): {params.compute_ignition_probability():.2%}"

        return text

    def _add_category_legend(self, fig: go.Figure) -> None:
        """Add category legend to figure"""
        if not PLOTLY_AVAILABLE:
            return

        # Add invisible traces for legend
        for category in StateCategory:
            fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=10, color=category.color),
                    name=category.display_name,
                    showlegend=True,
                )
            )

    def export_visualization_report(
        self, output_dir: str = "./apgi_visualizations"
    ) -> Dict[str, str]:
        """
        Export a comprehensive set of visualizations.

        Args:
            output_dir: Directory to save visualizations

        Returns:
            Dictionary of visualization file paths
        """
        import os

        os.makedirs(output_dir, exist_ok=True)
        report_files = {}

        # 1. Main network visualization
        network_file = os.path.join(output_dir, "apgi_state_network_3d.html")
        self.plot_state_network_3d(save_path=network_file)
        report_files["network"] = network_file

        # 2. Ignition landscape
        landscape_file = os.path.join(output_dir, "ignition_landscape.html")
        self.plot_ignition_landscape(save_path=landscape_file)
        report_files["landscape"] = landscape_file

        # 3. Correlation heatmap
        heatmap_file = os.path.join(output_dir, "parameter_correlations.html")
        self.plot_parameter_correlation_heatmap(save_path=heatmap_file)
        report_files["correlations"] = heatmap_file

        # 4. Example transition pathways
        transition_file = os.path.join(output_dir, "transition_pathways.html")
        self.plot_transition_pathways("anxiety", "calm", save_path=transition_file)
        report_files["transitions"] = transition_file

        # 5. Category radar comparison
        radar_file = os.path.join(output_dir, "category_radar.html")

        # Select representative states from each category
        rep_states = []
        for category in StateCategory:
            category_states = [s for s, c in self.categories.items() if c == category]
            if category_states:
                rep_states.append(category_states[0])

        self.plot_state_radar(rep_states[:4], save_path=radar_file)
        report_files["radar"] = radar_file

        # 6. Example dashboard
        dashboard_file = os.path.join(output_dir, "state_dashboard_flow.html")
        self.create_state_summary_dashboard("flow", save_path=dashboard_file)
        report_files["dashboard"] = dashboard_file

        # 7. Create a summary index HTML
        self._create_report_index(output_dir, report_files)

        return report_files

    def _create_report_index(
        self, output_dir: str, report_files: Dict[str, str]
    ) -> None:
        """Create an HTML index page for the visualization report"""
        index_file = os.path.join(output_dir, "index.html")

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>APGI Psychological States Visualization Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .visualization-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }}
        .viz-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            border-left: 5px solid #3498db;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .viz-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .viz-card h3 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        .viz-card a {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .stats-box {{
            background: linear-gradient(135deg, #74b9ff, #0984e3);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 APGI Psychological States Visualization Report</h1>
        
        <div class="stats-box">
            <h2>📈 Library Statistics</h2>
            <p><strong>Total States:</strong> {len(self.states)}</p>
            <p><strong>Categories:</strong> {len(set(self.categories.values()))}</p>
            <p><strong>Visualizations Generated:</strong> {len(report_files)}</p>
            <p><strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <h2>🎨 Interactive Visualizations</h2>
        <div class="visualization-grid">
            <div class="viz-card">
                <h3>🔗 3D State Network</h3>
                <p>Interactive network showing all 51 psychological states positioned by their parameters. States are colored by category and sized by surprise accumulation.</p>
                <a href="{os.path.basename(report_files['network'])}" target="_blank">Open Visualization</a>
            </div>
            
            <div class="viz-card">
                <h3>🏔️ Ignition Landscape</h3>
                <p>3D surface showing ignition probability as a function of key parameters. States are plotted on the landscape showing their current position.</p>
                <a href="{os.path.basename(report_files['landscape'])}" target="_blank">Open Visualization</a>
            </div>
            
            <div class="viz-card">
                <h3>📊 Parameter Correlations</h3>
                <p>Heatmap showing correlations between all APGI parameters. Reveals underlying relationships in the psychological state space.</p>
                <a href="{os.path.basename(report_files['correlations'])}" target="_blank">Open Visualization</a>
            </div>
            
            <div class="viz-card">
                <h3>🔄 Transition Pathways</h3>
                <p>Visualization of state transitions from anxiety to calm, showing parameter evolution and suggested intermediate states.</p>
                <a href="{os.path.basename(report_files['transitions'])}" target="_blank">Open Visualization</a>
            </div>
            
            <div class="viz-card">
                <h3>🎯 Category Radar</h3>
                <p>Radar chart comparing representative states from each category across key parameters (normalized for comparison).</p>
                <a href="{os.path.basename(report_files['radar'])}" target="_blank">Open Visualization</a>
            </div>
            
            <div class="viz-card">
                <h3>📱 State Dashboard</h3>
                <p>Comprehensive dashboard for the "flow" state showing parameter profile, ignition dynamics, category comparison, and transitions.</p>
                <a href="{os.path.basename(report_files['dashboard'])}" target="_blank">Open Visualization</a>
            </div>
        </div>
        
        <div style="margin-top: 40px; padding: 20px; background: #f1f2f6; border-radius: 10px;">
            <h3>📝 How to Use These Visualizations</h3>
            <ol>
                <li><strong>Interactive Features:</strong> Hover over elements to see detailed information</li>
                <li><strong>3D Rotation:</strong> Click and drag to rotate 3D visualizations</li>
                <li><strong>Zoom:</strong> Use scroll wheel or pinch gestures to zoom</li>
                <li><strong>Export:</strong> Click the camera icon in Plotly visualizations to export as PNG</li>
                <li><strong>Embedding:</strong> HTML files can be embedded in presentations or web pages</li>
            </ol>
            
            <h3>🔧 Python API Usage</h3>
            <pre style="background: #2c3e50; color: white; padding: 15px; border-radius: 5px;">
# Create visualizer instance
visualizer = APGIVisualizer(PSYCHOLOGICAL_STATES, STATE_CATEGORIES)

# Generate specific visualization
fig = visualizer.plot_state_network_3d()
# Display in embedded viewer or save to file

# Export comprehensive report
report = visualizer.export_visualization_report()</pre>
        </div>
    </div>
</body>
</html>
        """

        with open(index_file, "w") as f:
            f.write(html_content)

        report_files["index"] = index_file


# =============================================================================
# GUI INTERFACE FOR INTERACTIVE VISUALIZATION
# =============================================================================


class APGIVisualizerGUI:
    """Interactive GUI for APGI Psychological States Visualization"""

    def __init__(self):
        """Initialize the GUI application"""
        if not TKINTER_AVAILABLE:
            raise ImportError("Tkinter is required for GUI interface")
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            raise ImportError("Plotly and Pandas are required for visualization")

        self.root = tk.Tk()
        self.root.title("APGI Psychological States Visualizer")
        self.root.geometry("1200x800")

        # Initialize visualizer with states and categories
        try:
            self.visualizer = APGIVisualizer(PSYCHOLOGICAL_STATES, STATE_CATEGORIES)
            self.current_visualization = None

            # Setup GUI
            self.setup_gui()

            # Populate state dropdowns
            self.populate_state_dropdowns()

            # Set initial state
            self.status_var.set("Ready - Select visualization type and click Generate")
            self.update_info(
                "APGI Visualizer initialized successfully!\n\n"
                "Available states: {}\n\n"
                "Choose a visualization type and click 'Generate Visualization' to begin.".format(
                    len(PSYCHOLOGICAL_STATES)
                )
            )
        except Exception as e:
            messagebox.showerror(
                "Initialization Error", f"Failed to initialize visualizer: {str(e)}"
            )
            self.root.destroy()
            raise

    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="APGI Psychological States Visualizer",
            font=("Arial", 16, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Control Panel (Left)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(
            row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10)
        )

        # Visualization Type
        ttk.Label(control_frame, text="Visualization Type:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        self.viz_type = ttk.Combobox(
            control_frame,
            values=[
                "3D State Network",
                "Ignition Landscape",
                "State Radar Comparison",
                "Parameter Correlation Heatmap",
                "State Dashboard",
                "Transition Pathways",
            ],
            state="readonly",
        )
        self.viz_type.set("3D State Network")
        self.viz_type.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # State Selection
        ttk.Label(control_frame, text="Select State:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.state_var = tk.StringVar()
        self.state_combo = ttk.Combobox(
            control_frame, textvariable=self.state_var, state="readonly"
        )
        self.state_combo.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Multiple States for Radar
        ttk.Label(control_frame, text="Multiple States (Radar):").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        self.states_text = tk.Text(control_frame, height=3, width=25)
        self.states_text.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.states_text.insert("1.0", "flow, anxiety, calm, focus")

        # Transition States
        ttk.Label(control_frame, text="From State:").grid(
            row=6, column=0, sticky=tk.W, pady=5
        )
        self.from_state_var = tk.StringVar()
        self.from_state_combo = ttk.Combobox(
            control_frame, textvariable=self.from_state_var, state="readonly"
        )
        self.from_state_combo.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Label(control_frame, text="To State:").grid(
            row=8, column=0, sticky=tk.W, pady=5
        )
        self.to_state_var = tk.StringVar()
        self.to_state_combo = ttk.Combobox(
            control_frame, textvariable=self.to_state_var, state="readonly"
        )
        self.to_state_combo.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Buttons
        ttk.Button(
            control_frame,
            text="Generate Visualization",
            command=self.generate_visualization,
        ).grid(row=10, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(
            control_frame, text="Save Visualization", command=self.save_visualization
        ).grid(row=11, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(
            control_frame,
            text="Generate Full Report",
            command=self.generate_full_report,
        ).grid(row=12, column=0, sticky=(tk.W, tk.E), pady=5)

        # Status Bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = ttk.Label(
            main_frame, textvariable=self.status_var, relief=tk.SUNKEN
        )
        status_bar.grid(
            row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        # Preview Area (Right)
        preview_frame = ttk.LabelFrame(
            main_frame, text="Visualization Preview", padding="10"
        )
        preview_frame.grid(
            row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

        # Info text
        self.info_text = tk.Text(preview_frame, height=20, width=60, wrap=tk.WORD)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbar for info text
        scrollbar = ttk.Scrollbar(
            preview_frame, orient=tk.VERTICAL, command=self.info_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text["yscrollcommand"] = scrollbar.set

        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        # Configure control frame
        control_frame.columnconfigure(0, weight=1)

    def populate_state_dropdowns(self):
        """Populate all state selection dropdowns with available states"""
        try:
            if not PSYCHOLOGICAL_STATES:
                self.status_var.set("Error: No states available")
                return

            # Get sorted list of state names
            state_names = sorted(PSYCHOLOGICAL_STATES.keys())

            # Set values for all state dropdowns
            for combo in [self.state_combo, self.from_state_combo, self.to_state_combo]:
                combo["values"] = state_names

            # Set default selections if available
            if state_names:
                self.state_combo.set(state_names[0])

                # Try to set common states if they exist, otherwise use first available
                if "anxiety" in state_names:
                    self.from_state_combo.set("anxiety")
                else:
                    self.from_state_combo.set(state_names[0])

                if "calm" in state_names:
                    self.to_state_combo.set("calm")
                else:
                    self.to_state_combo.set(state_names[-1])  # Use last state as target

            self.status_var.set("Ready - Select visualization type and click Generate")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.update_info(f"Error populating state dropdowns: {str(e)}")
            raise

    def init_visualizer(self):
        """Initialize the visualizer components (kept for backward compatibility)"""
        self.populate_state_dropdowns()

    def update_info(self, text):
        """Update the info text area"""
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", text)

    def generate_visualization(self):
        """Generate the selected visualization"""
        if not self.visualizer:
            messagebox.showerror("Error", "Visualizer not initialized")
            return

        viz_type = self.viz_type.get()
        self.status_var.set(f"Generating {viz_type}...")
        self.root.update()  # Update the UI to show the status

        try:
            # Clear previous visualization
            if hasattr(self, "html_frame"):
                self.html_frame.destroy()

            # Generate the appropriate visualization
            if viz_type == "3D State Network":
                fig = self.visualizer.plot_state_network_3d()
                title = "3D State Network Visualization"
            elif viz_type == "Ignition Landscape":
                state = self.state_var.get()
                if not state:
                    messagebox.showerror("Error", "Please select a state")
                    return
                fig = self.visualizer.plot_ignition_landscape(state)
                title = f"Ignition Landscape: {state}"
            elif viz_type == "State Radar Comparison":
                states_text = self.states_text.get("1.0", tk.END).strip()
                states = [s.strip() for s in states_text.split(",") if s.strip()]
                if not states:
                    messagebox.showerror("Error", "Please enter states to compare")
                    return
                fig = self.visualizer.plot_state_radar(states)
                title = "State Comparison: " + ", ".join(states)
            elif viz_type == "Parameter Correlation Heatmap":
                fig = self.visualizer.plot_parameter_correlation_heatmap()
                title = "Parameter Correlation Heatmap"
            elif viz_type == "State Dashboard":
                state = self.state_var.get()
                if not state:
                    messagebox.showerror("Error", "Please select a state")
                    return
                fig = self.visualizer.create_state_summary_dashboard(state)
                title = f"State Dashboard: {state}"
            elif viz_type == "Transition Pathways":
                from_state = self.from_state_var.get()
                to_state = self.to_state_var.get()
                if not from_state or not to_state:
                    messagebox.showerror(
                        "Error", "Please select both 'From' and 'To' states"
                    )
                    return
                fig = self.visualizer.plot_transition_pathways(from_state, to_state)
                title = f"Transition: {from_state} → {to_state}"
            else:
                messagebox.showerror("Error", "Unknown visualization type")
                return

            if fig:
                self.current_visualization = fig
                self.status_var.set(f"Generated {viz_type} visualization")

                # Create HTML content with responsive design
                plot_data = fig.to_json()
                plot_layout = fig.layout.to_plotly_json()

                # Create the HTML content with proper string formatting
                html_content = f"""<!DOCTYPE html>
                <html>
                <head>
                    <title>{title}</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <style>
                        html, body {{
                            margin: 0;
                            padding: 0;
                            width: 100%;
                            height: 100%;
                            overflow: hidden;
                        }}
                        #plot {{
                            width: 100%;
                            height: 100%;
                            border: none;
                        }}
                    </style>
                </head>
                <body>
                    <div id="plot"></div>
                    <script>
                        var plotData = {plot_data};
                        var layout = {plot_layout};
                        
                        // Make the plot responsive
                        layout.autosize = true;
                        layout.margin = {{l: 50, r: 50, b: 50, t: 50, pad: 4}};
                        
                        // Create the plot
                        Plotly.newPlot('plot', plotData, layout, {{responsive: true}});
                        
                        // Handle window resizing
                        window.addEventListener('resize', function() {{
                            Plotly.Plots.resize('plot');
                        }});
                    </script>
                </body>
                </html>"""

                # Try to use tkinterweb for embedded display
                try:
                    from tkinterweb import HtmlFrame

                    # Create HTML frame and load content
                    self.html_frame = HtmlFrame(
                        self.preview_frame,
                        messages_enabled=False,
                        vertical_scrollbar=False,
                        horizontal_scrollbar=False,
                    )
                    self.html_frame.grid(row=0, column=0, sticky="nsew")
                    self.html_frame.load_html(html_content)

                    # Configure grid weights for proper resizing
                    self.preview_frame.columnconfigure(0, weight=1)
                    self.preview_frame.rowconfigure(0, weight=1)

                except ImportError:
                    # Fallback message if embedded display fails
                    state_count = len(self.states) if hasattr(self, "states") else 0
                    self.status_var.set(
                        "Visualization generated (embedded display unavailable)"
                    )
                    self.update_info(
                        f"Generated {title} successfully!\n\n"
                        f"Visualization Type: {viz_type}\n"
                        f"States: {state_count} psychological states\n\n"
                        f"Note: Embedded display requires additional dependencies. "
                        f"The visualization data has been generated and can be saved."
                    )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate visualization: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            self.update_info(
                f"Error generating visualization: {str(e)}\n\n{format_exc()}"
            )

    def save_visualization(self):
        """Save the current visualization"""
        if not self.current_visualization:
            messagebox.showwarning(
                "Warning", "No visualization to save. Generate one first."
            )
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
        )

        if filename:
            try:
                self.current_visualization.write_html(filename)
                messagebox.showinfo("Success", f"Visualization saved to {filename}")
                self.status_var.set(f"Saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {str(e)}")

    def generate_full_report(self):
        """Generate comprehensive visualization report"""
        if not self.visualizer:
            messagebox.showerror("Error", "Visualizer not initialized")
            return

        self.status_var.set("Generating full report...")

        def generate():
            try:
                report_dir = filedialog.askdirectory(
                    title="Select directory for report"
                )
                if report_dir:
                    report_files = self.visualizer.export_visualization_report(
                        report_dir
                    )

                    # Update GUI in main thread
                    self.root.after(
                        0,
                        lambda: self.status_var.set(
                            f"Report generated with {len(report_files)} files"
                        ),
                    )
                    self.root.after(
                        0,
                        lambda: self.update_info(
                            f"Comprehensive report generated!\n\n"
                            f"Directory: {report_dir}\n\n"
                            f"Files created:\n"
                            f"• 3D State Network\n"
                            f"• Ignition Landscape\n"
                            f"• Parameter Correlations\n"
                            f"• Transition Pathways\n"
                            f"• Category Radar\n"
                            f"• State Dashboard\n"
                            f"• Index HTML\n\n"
                            f"Report generated successfully!"
                        ),
                    )

            except Exception as e:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Error", f"Failed to generate report: {str(e)}"
                    ),
                )
                self.root.after(
                    0, lambda: self.status_var.set("Error generating report")
                )

        # Run in background thread
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()


def create_apgi_params(
    Pi_e: float,
    Pi_i_baseline: float,
    M_ca: float,
    beta: float,
    z_e: float,
    z_i: float,
    theta_t: float,
) -> APGIParameters:
    """
    Factory function that computes derived parameters automatically.

    Computes:
    - Pi_i_eff = Pi_i_baseline · exp(β·M_ca)
    - S_t = Π_e·|z_e| + Π_i_eff·|z_i|
    """
    # Compute effective interoceptive precision with somatic modulation
    Pi_i_eff = Pi_i_baseline * np.exp(beta * M_ca)
    Pi_i_eff = np.clip(Pi_i_eff, 0.1, 10.0)

    # Compute accumulated surprise
    S_t = Pi_e * abs(z_e) + Pi_i_eff * abs(z_i)

    return APGIParameters(
        Pi_e=Pi_e,
        Pi_i_baseline=Pi_i_baseline,
        Pi_i_eff=Pi_i_eff,
        theta_t=theta_t,
        S_t=S_t,
        M_ca=M_ca,
        beta=beta,
        z_e=z_e,
        z_i=z_i,
    )


# =============================================================================
# STATE DEFINITIONS (51 PSYCHOLOGICAL STATES)
# =============================================================================

# =============================================================================
# CATEGORY 1: OPTIMAL FUNCTIONING STATES (States 1-4)
# =============================================================================

STATE_01_FLOW = create_apgi_params(
    Pi_e=6.5,  # High precision on task-relevant content
    Pi_i_baseline=1.5,  # Low baseline interoceptive (body recedes)
    M_ca=0.3,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.4,  # Low prediction error (mastery)
    z_i=0.2,  # Minimal interoceptive error
    theta_t=1.8,  # Elevated threshold (effortless filtering)
)

STATE_02_FOCUS = create_apgi_params(
    Pi_e=8.0,  # Very high precision on target
    Pi_i_baseline=1.2,  # Low interoceptive baseline
    M_ca=0.25,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.8,  # Moderate prediction error (challenge)
    z_i=0.3,  # Low interoceptive error
    theta_t=-0.5,  # Actively suppressed threshold (effortful)
)

STATE_03_SERENITY = create_apgi_params(
    Pi_e=1.5,  # Low, broadly distributed precision
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.7,  # Moderate somatic bias (peaceful embodiment)
    beta=0.5,  # Neutral somatic gain
    z_e=0.2,  # Minimal prediction error
    z_i=0.3,  # Minimal interoceptive error
    theta_t=1.5,  # Elevated threshold (filters trivial content)
)

STATE_04_MINDFULNESS = create_apgi_params(
    Pi_e=3.0,  # Moderate, present-moment precision
    Pi_i_baseline=3.5,  # Elevated interoceptive baseline
    M_ca=0.9,  # Moderate-high somatic bias (embodied presence)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.6,  # Moderate errors (observed without reactivity)
    z_i=0.5,  # Moderate interoceptive errors
    theta_t=0.0,  # Neutral threshold (flexible, non-reactive)
)


# =============================================================================
# CATEGORY 2: POSITIVE AFFECTIVE STATES (States 5-11)
# =============================================================================

STATE_05_AMUSEMENT = create_apgi_params(
    Pi_e=4.0,  # Moderate-high precision on incongruity
    Pi_i_baseline=1.0,  # Low interoceptive (cognitive, not embodied)
    M_ca=-0.1,  # Slightly negative somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=1.2,  # Moderate PE (benign incongruity)
    z_i=0.2,  # Minimal interoceptive error
    theta_t=-0.3,  # Slightly lowered for incongruity resolution
)

STATE_06_JOY = create_apgi_params(
    Pi_e=5.0,  # High precision on reward-relevant content
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.8,  # Moderate-high somatic bias (warmth, expansion)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=1.0,  # Positive prediction error (better than expected)
    z_i=0.7,  # Moderate interoceptive involvement
    theta_t=-0.8,  # Lowered for positive content
)

STATE_07_PRIDE = create_apgi_params(
    Pi_e=4.5,  # High precision on self-model
    Pi_i_baseline=3.0,  # Elevated interoceptive baseline
    M_ca=1.1,  # High somatic bias (postural expansion)
    beta=0.6,  # Elevated somatic gain
    z_e=1.2,  # Positive self-referential PE
    z_i=0.9,  # Elevated interoceptive (embodied pride)
    theta_t=-0.6,  # Lowered for self-enhancing content
)

STATE_08_ROMANTIC_LOVE_EARLY = create_apgi_params(
    Pi_e=7.5,  # Very high precision on partner
    Pi_i_baseline=4.0,  # High interoceptive baseline
    M_ca=1.8,  # Very high somatic bias
    beta=0.7,  # High somatic gain
    z_e=1.5,  # High PE (partner constantly surprising)
    z_i=1.3,  # High interoceptive involvement
    theta_t=-1.5,  # Very low threshold for partner cues
)

STATE_08B_ROMANTIC_LOVE_SUSTAINED = create_apgi_params(
    Pi_e=5.0,  # Moderate-high precision on partner
    Pi_i_baseline=3.0,  # Moderate interoceptive baseline
    M_ca=1.2,  # High but not extreme somatic bias
    beta=0.6,  # Moderate-high somatic gain
    z_e=0.5,  # Low PE (partner predictable, integrated)
    z_i=0.6,  # Moderate interoceptive involvement
    theta_t=-0.8,  # Lowered but not extreme
)

STATE_09_GRATITUDE = create_apgi_params(
    Pi_e=4.0,  # Moderate-high precision on affiliative confirmation
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.8,  # Moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.3,  # Low PE (prediction confirmed)
    z_i=0.5,  # Moderate interoceptive (warmth)
    theta_t=-0.4,  # Slightly lowered
)

STATE_10_HOPE = create_apgi_params(
    Pi_e=5.0,  # High precision on specific future outcomes
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.6,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.9,  # Moderate PE (future uncertainty)
    z_i=0.4,  # Low interoceptive error
    theta_t=-0.7,  # Lowered for desired futures
)

STATE_11_OPTIMISM = create_apgi_params(
    Pi_e=3.0,  # Diffuse moderate precision
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.4,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.4,  # Low PE (positive priors)
    z_i=0.3,  # Low interoceptive error
    theta_t=-0.5,  # Globally slightly lowered
)


# =============================================================================
# CATEGORY 3: COGNITIVE AND ATTENTIONAL STATES (States 12-19)
# =============================================================================

STATE_12_CURIOSITY = create_apgi_params(
    Pi_e=6.0,  # High precision on reducible uncertainty
    Pi_i_baseline=1.0,  # Low interoceptive baseline
    M_ca=-0.2,  # Slightly negative somatic bias (cognitive)
    beta=0.45,  # Slightly low somatic gain
    z_e=1.4,  # Moderate-high PE (novelty)
    z_i=0.2,  # Minimal interoceptive error
    theta_t=-0.9,  # Lowered for novel information
)

STATE_13_BOREDOM = create_apgi_params(
    Pi_e=0.8,  # Collapsed precision on current task
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=-0.3,  # Slightly negative somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.1,  # Very low PE (no surprise)
    z_i=0.2,  # Low interoceptive error
    theta_t=-1.0,  # Lowered for novel inputs (seeking)
)

STATE_14_CREATIVITY = create_apgi_params(
    Pi_e=4.0,  # Moderate precision (loosened high-level)
    Pi_i_baseline=1.0,  # Low interoceptive baseline
    M_ca=-0.3,  # Slightly negative somatic bias (cognitive)
    beta=0.45,  # Slightly low somatic gain
    z_e=1.2,  # Variable PE (tolerated)
    z_i=0.2,  # Minimal interoceptive error
    theta_t=-1.2,  # Lowered for unconventional content
)

STATE_15_INSPIRATION = create_apgi_params(
    Pi_e=8.5,  # Very high precision spike on emergent pattern
    Pi_i_baseline=1.5,  # Low-moderate interoceptive baseline
    M_ca=0.4,  # Low-moderate somatic bias (aha feeling)
    beta=0.5,  # Neutral somatic gain
    z_e=2.0,  # High PE (unexpected convergence)
    z_i=0.4,  # Low interoceptive error
    theta_t=-2.0,  # Acutely suppressed threshold
)

STATE_16_HYPERFOCUS = create_apgi_params(
    Pi_e=9.5,  # Extreme precision on single target
    Pi_i_baseline=0.5,  # Very low interoceptive baseline
    M_ca=-0.8,  # Negative somatic bias (body disappears)
    beta=0.4,  # Low somatic gain
    z_e=0.6,  # Low PE within domain
    z_i=0.1,  # Minimal interoceptive (ignored)
    theta_t=2.5,  # Very elevated for all else
)

STATE_17_FATIGUE = create_apgi_params(
    Pi_e=1.5,  # Globally reduced precision
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.4,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.3,  # Low PE
    z_i=0.4,  # Moderate interoceptive (tiredness signals)
    theta_t=1.8,  # Elevated (metabolic conservation)
)

STATE_18_DECISION_FATIGUE = create_apgi_params(
    Pi_e=2.5,  # Undifferentiated precision across options
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=0.3,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.8,  # Moderate PE per option
    z_i=0.3,  # Low interoceptive error
    theta_t=1.5,  # Elevated (metabolic conservation)
)

STATE_19_MIND_WANDERING = create_apgi_params(
    Pi_e=0.8,  # Collapsed external precision
    Pi_i_baseline=3.5,  # Elevated interoceptive/DMN baseline
    M_ca=0.6,  # Moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.2,  # Low external PE (ignored)
    z_i=0.9,  # Moderate internal PE (narrative unfolds)
    theta_t=1.5,  # High external / Low internal (asymmetric)
)


# =============================================================================
# CATEGORY 4: AVERSIVE AFFECTIVE STATES (States 20-26)
# =============================================================================

STATE_20_FEAR = create_apgi_params(
    Pi_e=8.0,  # Very high precision on threat
    Pi_i_baseline=3.0,  # Elevated interoceptive baseline
    M_ca=1.9,  # Near-maximum somatic bias
    beta=0.75,  # High somatic gain
    z_e=2.5,  # Very high PE (threat detection)
    z_i=2.0,  # High interoceptive PE (body mobilization)
    theta_t=-2.5,  # Strongly suppressed threshold
)

STATE_21_ANXIETY = create_apgi_params(
    Pi_e=6.5,  # High precision on potential threats
    Pi_i_baseline=3.5,  # Elevated interoceptive baseline
    M_ca=1.5,  # High somatic bias
    beta=0.65,  # Elevated somatic gain
    z_e=1.5,  # Moderate-high PE
    z_i=1.3,  # Elevated interoceptive PE
    theta_t=-1.5,  # Lowered for threat-relevant content
)

STATE_22_ANGER = create_apgi_params(
    Pi_e=7.5,  # Very high precision on obstacle/transgressor
    Pi_i_baseline=3.0,  # Elevated interoceptive baseline
    M_ca=1.5,  # High somatic bias (mobilization)
    beta=0.65,  # Elevated somatic gain
    z_e=2.0,  # High PE (goal blockage)
    z_i=1.4,  # Elevated interoceptive PE
    theta_t=-1.2,  # Lowered for target; elevated for mitigation
)

STATE_23_GUILT = create_apgi_params(
    Pi_e=5.0,  # High precision on self-model violations
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.8,  # Moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=1.3,  # Moderate PE (action-specific)
    z_i=0.9,  # Moderate interoceptive PE
    theta_t=-0.8,  # Lowered for rumination
)

STATE_24_SHAME = create_apgi_params(
    Pi_e=7.0,  # Very high precision on social prediction errors
    Pi_i_baseline=3.0,  # Elevated interoceptive baseline
    M_ca=1.3,  # High somatic bias
    beta=0.6,  # Elevated somatic gain
    z_e=1.8,  # High PE (global self-threat)
    z_i=1.2,  # Elevated interoceptive PE
    theta_t=-1.5,  # Strongly lowered (broad ignition)
)

STATE_25_LONELINESS = create_apgi_params(
    Pi_e=5.5,  # High precision on social absence
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.8,  # Moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=1.4,  # Moderate-high PE (absence signals)
    z_i=0.9,  # Moderate interoceptive PE
    theta_t=-1.0,  # Lowered for social threat
)

STATE_26_OVERWHELM = create_apgi_params(
    Pi_e=3.0,  # Fragmented, unstable precision
    Pi_i_baseline=3.0,  # Elevated interoceptive baseline
    M_ca=1.2,  # High somatic bias (freeze)
    beta=0.6,  # Elevated somatic gain
    z_e=2.8,  # Very high concurrent PE
    z_i=1.5,  # High interoceptive PE
    theta_t=0.0,  # Chaotic (multiple crossings)
)


# =============================================================================
# CATEGORY 5: PATHOLOGICAL AND EXTREME STATES (States 27-33)
# =============================================================================

STATE_27_DEPRESSION = create_apgi_params(
    Pi_e=2.0,  # Low precision on positive; high on negative
    Pi_i_baseline=1.5,  # Dysregulated interoceptive baseline
    M_ca=0.3,  # Dysregulated somatic (oscillating)
    beta=0.5,  # Neutral somatic gain
    z_e=0.4,  # Blunted PE
    z_i=0.8,  # Moderate interoceptive (allostatic disruption)
    theta_t=1.5,  # Elevated for positive; lowered for negative
)

STATE_28_LEARNED_HELPLESSNESS = create_apgi_params(
    Pi_e=1.5,  # Collapsed precision on controllability
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.5,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.2,  # Extinguished PE (actions don't matter)
    z_i=0.4,  # Low interoceptive PE
    theta_t=2.0,  # Elevated for action initiation
)

STATE_29_PESSIMISTIC_DEPRESSION = create_apgi_params(
    Pi_e=2.5,  # High precision on negative priors
    Pi_i_baseline=2.0,  # Moderate-dysregulated interoceptive
    M_ca=0.7,  # Moderate somatic bias (heaviness)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.3,  # Low PE (positive surprise extinguished)
    z_i=0.6,  # Moderate interoceptive PE
    theta_t=1.8,  # Elevated for positive evidence
)

STATE_30_PANIC = create_apgi_params(
    Pi_e=4.0,  # Moderate external (overwhelmed by internal)
    Pi_i_baseline=5.0,  # Very high interoceptive baseline
    M_ca=2.0,  # Maximum somatic bias
    beta=0.8,  # Maximum somatic gain
    z_e=1.5,  # Moderate external PE
    z_i=3.0,  # Extreme interoceptive PE
    theta_t=-3.0,  # Catastrophically suppressed threshold
)

STATE_31_DISSOCIATION = create_apgi_params(
    Pi_e=2.0,  # Reduced external precision
    Pi_i_baseline=0.5,  # Collapsed interoceptive baseline
    M_ca=-1.5,  # Strongly negative somatic bias (disconnection)
    beta=0.35,  # Low somatic gain
    z_e=0.8,  # Moderate external PE
    z_i=0.1,  # Minimal interoceptive (disconnected)
    theta_t=2.0,  # Elevated for embodiment
)

STATE_32_DEPERSONALIZATION = create_apgi_params(
    Pi_e=3.0,  # Moderate external precision
    Pi_i_baseline=0.8,  # Low interoceptive baseline
    M_ca=-1.2,  # Negative somatic bias (self disconnected)
    beta=0.4,  # Low somatic gain
    z_e=1.0,  # Moderate PE (self-mismatch)
    z_i=0.5,  # Low interoceptive (foreign)
    theta_t=1.5,  # Elevated for self-as-subject
)

STATE_33_DEREALIZATION = create_apgi_params(
    Pi_e=1.5,  # Collapsed external precision
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=-0.8,  # Negative somatic bias
    beta=0.45,  # Slightly low somatic gain
    z_e=1.2,  # Moderate PE (world-mismatch)
    z_i=0.4,  # Low interoceptive PE
    theta_t=1.8,  # Elevated for perceptual integration
)


# =============================================================================
# CATEGORY 6: ALTERED AND BOUNDARY STATES (States 34-39)
# =============================================================================

STATE_34_AWE = create_apgi_params(
    Pi_e=3.5,  # Reduced precision on high-level priors
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.8,  # Variable, moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=2.8,  # Very high hierarchical PE
    z_i=0.7,  # Moderate interoceptive PE
    theta_t=-1.5,  # Briefly suppressed threshold
)

STATE_35_TRANCE = create_apgi_params(
    Pi_e=1.0,  # Suppressed external precision
    Pi_i_baseline=4.0,  # Very high internal precision
    M_ca=0.4,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.2,  # Low external PE (ignored)
    z_i=0.6,  # Moderate internal PE
    theta_t=2.0,  # High external / Low internal
)

STATE_36_MEDITATION_FOCUSED = create_apgi_params(
    Pi_e=7.0,  # Very high precision on meditation object
    Pi_i_baseline=3.5,  # Elevated interoceptive (breath-focused)
    M_ca=1.0,  # Moderate-high somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.5,  # Low PE (maintained attention)
    z_i=0.6,  # Moderate interoceptive PE
    theta_t=1.5,  # Elevated for distractions
)

STATE_36B_MEDITATION_OPEN = create_apgi_params(
    Pi_e=3.0,  # Broadly distributed precision
    Pi_i_baseline=3.0,  # Moderate interoceptive baseline
    M_ca=0.7,  # Moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.8,  # Variable PE (observed)
    z_i=0.6,  # Moderate interoceptive PE
    theta_t=0.0,  # Neutral (flexible)
)

STATE_36C_MEDITATION_NONDUAL = create_apgi_params(
    Pi_e=2.0,  # Collapsed content precision
    Pi_i_baseline=1.5,  # Reduced interoceptive baseline
    M_ca=0.5,  # Variable somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.2,  # Very low PE (no self-reference)
    z_i=0.2,  # Very low interoceptive PE
    theta_t=2.0,  # Elevated for conceptual; lowered for immediate
)

STATE_37_HYPNOSIS = create_apgi_params(
    Pi_e=2.0,  # Suppressed reality-testing precision
    Pi_i_baseline=3.5,  # Elevated for suggested content
    M_ca=0.6,  # Variable (suggestion-dependent)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.3,  # Low PE (suggestions = "true")
    z_i=0.8,  # Moderate interoceptive PE
    theta_t=-1.5,  # Very low for suggestions
)

STATE_38_HYPNAGOGIA = create_apgi_params(
    Pi_e=2.5,  # Unstable external precision
    Pi_i_baseline=4.0,  # Elevated memory-trace precision
    M_ca=0.7,  # Moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.6,  # Moderate PE (blended)
    z_i=1.0,  # Elevated interoceptive PE
    theta_t=0.5,  # Unstable (transitional)
)

STATE_39_DEJA_VU = create_apgi_params(
    Pi_e=4.5,  # Anomalously high familiarity precision
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=0.2,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.4,  # Low PE (false confirmation)
    z_i=0.2,  # Low interoceptive PE
    theta_t=-0.8,  # Premature threshold crossing
)


# =============================================================================
# CATEGORY 7: TRANSITIONAL/CONTEXTUAL STATES (States 40-46)
# =============================================================================

STATE_40_MORNING_FLOW = create_apgi_params(
    Pi_e=5.5,  # High precision on routine tasks
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.5,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.3,  # Low PE (practiced routines)
    z_i=0.3,  # Low interoceptive PE
    theta_t=1.2,  # Elevated for non-routine
)

STATE_41_EVENING_FATIGUE = create_apgi_params(
    Pi_e=1.2,  # Globally reduced precision
    Pi_i_baseline=3.0,  # Elevated interoceptive (tiredness signals)
    M_ca=1.0,  # Moderate-high somatic bias (heaviness)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.2,  # Low external PE
    z_i=0.7,  # Moderate interoceptive PE
    theta_t=2.2,  # Elevated (metabolic conservation)
)

STATE_42_CREATIVE_INSPIRATION = create_apgi_params(
    Pi_e=8.0,  # Very high precision on novel synthesis
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=0.3,  # Low somatic bias (mild aha)
    beta=0.5,  # Neutral somatic gain
    z_e=2.2,  # High PE (unexpected connection)
    z_i=0.3,  # Low interoceptive PE
    theta_t=-1.8,  # Acutely suppressed threshold
)

STATE_43_ANXIOUS_RUMINATION = create_apgi_params(
    Pi_e=6.0,  # Very high precision on threat possibilities
    Pi_i_baseline=3.5,  # Elevated interoceptive baseline
    M_ca=1.4,  # High somatic bias (sustained tension)
    beta=0.65,  # Elevated somatic gain
    z_e=1.6,  # Moderate-high PE (threat scenarios)
    z_i=1.2,  # Elevated interoceptive PE
    theta_t=-1.2,  # Lowered for threat; elevated for reassurance
)

STATE_44_CALM = create_apgi_params(
    Pi_e=1.8,  # Low, broadly distributed precision
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.5,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.2,  # Minimal PE
    z_i=0.3,  # Minimal interoceptive PE
    theta_t=1.2,  # Moderately elevated
)

STATE_45_PRODUCTIVE_FOCUS = create_apgi_params(
    Pi_e=7.0,  # High precision on task
    Pi_i_baseline=1.5,  # Low interoceptive baseline
    M_ca=0.3,  # Low somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.7,  # Low-moderate PE (manageable challenge)
    z_i=0.3,  # Low interoceptive PE
    theta_t=-0.3,  # Slightly lowered for task
)

STATE_46_SECOND_WIND = create_apgi_params(
    Pi_e=5.5,  # Restored precision
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.5,  # Low-moderate somatic bias
    beta=0.55,  # Slightly elevated somatic gain
    z_e=0.9,  # Moderate PE (renewed engagement)
    z_i=0.5,  # Moderate interoceptive PE
    theta_t=-0.8,  # Lowered (catecholamine release)
)


# =============================================================================
# CATEGORY 8: PREVIOUSLY UNELABORATED STATES (States 47-51)
# =============================================================================

STATE_47_HYPERVIGILANCE = create_apgi_params(
    Pi_e=8.5,  # Very high threat-scanning precision
    Pi_i_baseline=4.0,  # Elevated interoceptive baseline
    M_ca=1.7,  # Very high somatic bias
    beta=0.7,  # High somatic gain
    z_e=1.8,  # Moderate-high PE (scanning)
    z_i=1.5,  # Elevated interoceptive PE
    theta_t=-2.0,  # Very low for threat; high for safety
)

STATE_48_SADNESS = create_apgi_params(
    Pi_e=4.5,  # High precision on loss-relevant content
    Pi_i_baseline=2.5,  # Moderate interoceptive baseline
    M_ca=0.9,  # Moderate somatic bias (heaviness)
    beta=0.55,  # Slightly elevated somatic gain
    z_e=1.2,  # Moderate PE (loss detection)
    z_i=0.8,  # Moderate interoceptive PE
    theta_t=-0.6,  # Lowered for loss reminders
)

STATE_49_CHOICE_PARALYSIS = create_apgi_params(
    Pi_e=2.5,  # Equal, undifferentiated precision
    Pi_i_baseline=2.0,  # Moderate interoceptive baseline
    M_ca=0.5,  # Low-moderate somatic bias
    beta=0.5,  # Neutral somatic gain
    z_e=0.9,  # Moderate PE per option
    z_i=0.5,  # Moderate interoceptive PE
    theta_t=1.5,  # Elevated (metabolic conservation)
)

STATE_50_MENTAL_PARALYSIS = create_apgi_params(
    Pi_e=2.0,  # Fragmented, unstable precision
    Pi_i_baseline=3.5,  # Elevated interoceptive baseline
    M_ca=1.3,  # High somatic bias (freeze)
    beta=0.65,  # Elevated somatic gain
    z_e=3.0,  # Very high concurrent PE
    z_i=1.8,  # High interoceptive PE
    theta_t=0.5,  # Chaotic (multiple crossings)
)

STATE_51_CURIOUS_EXPLORATION = create_apgi_params(
    Pi_e=6.5,  # High precision on novel stimuli
    Pi_i_baseline=1.0,  # Low interoceptive baseline
    M_ca=-0.1,  # Slightly negative somatic bias (exteroceptive)
    beta=0.45,  # Slightly low somatic gain
    z_e=1.6,  # Moderate-high PE (novelty)
    z_i=0.2,  # Minimal interoceptive PE
    theta_t=-1.0,  # Lowered for novelty
)

# =============================================================================
# MASTER DICTIONARY OF ALL STATES
# =============================================================================

PSYCHOLOGICAL_STATES: Dict[str, APGIParameters] = {
    # Optimal Functioning States (1-4)
    "flow": STATE_01_FLOW,
    "focus": STATE_02_FOCUS,
    "serenity": STATE_03_SERENITY,
    "mindfulness": STATE_04_MINDFULNESS,
    # Positive Affective States (5-11)
    "amusement": STATE_05_AMUSEMENT,
    "joy": STATE_06_JOY,
    "pride": STATE_07_PRIDE,
    "romantic_love_early": STATE_08_ROMANTIC_LOVE_EARLY,
    "romantic_love_sustained": STATE_08B_ROMANTIC_LOVE_SUSTAINED,
    "gratitude": STATE_09_GRATITUDE,
    "hope": STATE_10_HOPE,
    "optimism": STATE_11_OPTIMISM,
    # Cognitive and Attentional States (12-19)
    "curiosity": STATE_12_CURIOSITY,
    "boredom": STATE_13_BOREDOM,
    "creativity": STATE_14_CREATIVITY,
    "inspiration": STATE_15_INSPIRATION,
    "hyperfocus": STATE_16_HYPERFOCUS,
    "fatigue": STATE_17_FATIGUE,
    "decision_fatigue": STATE_18_DECISION_FATIGUE,
    "mind_wandering": STATE_19_MIND_WANDERING,
    # Aversive Affective States (20-26)
    "fear": STATE_20_FEAR,
    "anxiety": STATE_21_ANXIETY,
    "anger": STATE_22_ANGER,
    "guilt": STATE_23_GUILT,
    "shame": STATE_24_SHAME,
    "loneliness": STATE_25_LONELINESS,
    "overwhelm": STATE_26_OVERWHELM,
    # Pathological and Extreme States (27-33)
    "depression": STATE_27_DEPRESSION,
    "learned_helplessness": STATE_28_LEARNED_HELPLESSNESS,
    "pessimistic_depression": STATE_29_PESSIMISTIC_DEPRESSION,
    "panic": STATE_30_PANIC,
    "dissociation": STATE_31_DISSOCIATION,
    "depersonalization": STATE_32_DEPERSONALIZATION,
    "derealization": STATE_33_DEREALIZATION,
    # Altered and Boundary States (34-39)
    "awe": STATE_34_AWE,
    "trance": STATE_35_TRANCE,
    "meditation_focused": STATE_36_MEDITATION_FOCUSED,
    "meditation_open": STATE_36B_MEDITATION_OPEN,
    "meditation_nondual": STATE_36C_MEDITATION_NONDUAL,
    "hypnosis": STATE_37_HYPNOSIS,
    "hypnagogia": STATE_38_HYPNAGOGIA,
    "deja_vu": STATE_39_DEJA_VU,
    # Transitional/Contextual States (40-46)
    "morning_flow": STATE_40_MORNING_FLOW,
    "evening_fatigue": STATE_41_EVENING_FATIGUE,
    "creative_inspiration": STATE_42_CREATIVE_INSPIRATION,
    "anxious_rumination": STATE_43_ANXIOUS_RUMINATION,
    "calm": STATE_44_CALM,
    "productive_focus": STATE_45_PRODUCTIVE_FOCUS,
    "second_wind": STATE_46_SECOND_WIND,
    # Previously Unelaborated States (47-51)
    "hypervigilance": STATE_47_HYPERVIGILANCE,
    "sadness": STATE_48_SADNESS,
    "choice_paralysis": STATE_49_CHOICE_PARALYSIS,
    "mental_paralysis": STATE_50_MENTAL_PARALYSIS,
    "curious_exploration": STATE_51_CURIOUS_EXPLORATION,
}

# State category mapping
STATE_CATEGORIES: Dict[str, StateCategory] = {
    "flow": StateCategory.OPTIMAL_FUNCTIONING,
    "focus": StateCategory.OPTIMAL_FUNCTIONING,
    "serenity": StateCategory.OPTIMAL_FUNCTIONING,
    "mindfulness": StateCategory.OPTIMAL_FUNCTIONING,
    "amusement": StateCategory.POSITIVE_AFFECTIVE,
    "joy": StateCategory.POSITIVE_AFFECTIVE,
    "pride": StateCategory.POSITIVE_AFFECTIVE,
    "romantic_love_early": StateCategory.POSITIVE_AFFECTIVE,
    "romantic_love_sustained": StateCategory.POSITIVE_AFFECTIVE,
    "gratitude": StateCategory.POSITIVE_AFFECTIVE,
    "hope": StateCategory.POSITIVE_AFFECTIVE,
    "optimism": StateCategory.POSITIVE_AFFECTIVE,
    "curiosity": StateCategory.COGNITIVE_ATTENTIONAL,
    "boredom": StateCategory.COGNITIVE_ATTENTIONAL,
    "creativity": StateCategory.COGNITIVE_ATTENTIONAL,
    "inspiration": StateCategory.COGNITIVE_ATTENTIONAL,
    "hyperfocus": StateCategory.COGNITIVE_ATTENTIONAL,
    "fatigue": StateCategory.COGNITIVE_ATTENTIONAL,
    "decision_fatigue": StateCategory.COGNITIVE_ATTENTIONAL,
    "mind_wandering": StateCategory.COGNITIVE_ATTENTIONAL,
    "fear": StateCategory.AVERSIVE_AFFECTIVE,
    "anxiety": StateCategory.AVERSIVE_AFFECTIVE,
    "anger": StateCategory.AVERSIVE_AFFECTIVE,
    "guilt": StateCategory.AVERSIVE_AFFECTIVE,
    "shame": StateCategory.AVERSIVE_AFFECTIVE,
    "loneliness": StateCategory.AVERSIVE_AFFECTIVE,
    "overwhelm": StateCategory.AVERSIVE_AFFECTIVE,
    "depression": StateCategory.PATHOLOGICAL_EXTREME,
    "learned_helplessness": StateCategory.PATHOLOGICAL_EXTREME,
    "pessimistic_depression": StateCategory.PATHOLOGICAL_EXTREME,
    "panic": StateCategory.PATHOLOGICAL_EXTREME,
    "dissociation": StateCategory.PATHOLOGICAL_EXTREME,
    "depersonalization": StateCategory.PATHOLOGICAL_EXTREME,
    "derealization": StateCategory.PATHOLOGICAL_EXTREME,
    "awe": StateCategory.ALTERED_BOUNDARY,
    "trance": StateCategory.ALTERED_BOUNDARY,
    "meditation_focused": StateCategory.ALTERED_BOUNDARY,
    "meditation_open": StateCategory.ALTERED_BOUNDARY,
    "meditation_nondual": StateCategory.ALTERED_BOUNDARY,
    "hypnosis": StateCategory.ALTERED_BOUNDARY,
    "hypnagogia": StateCategory.ALTERED_BOUNDARY,
    "deja_vu": StateCategory.ALTERED_BOUNDARY,
    "morning_flow": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "evening_fatigue": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "creative_inspiration": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "anxious_rumination": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "calm": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "productive_focus": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "second_wind": StateCategory.TRANSITIONAL_CONTEXTUAL,
    "hypervigilance": StateCategory.UNELABORATED,
    "sadness": StateCategory.UNELABORATED,
    "choice_paralysis": StateCategory.UNELABORATED,
    "mental_paralysis": StateCategory.UNELABORATED,
    "curious_exploration": StateCategory.UNELABORATED,
}

# =============================================================================
# ENHANCED UTILITY FUNCTIONS
# =============================================================================


def get_state(name: str) -> APGIParameters:
    """Retrieve parameters for a named psychological state"""
    if name not in PSYCHOLOGICAL_STATES:
        raise KeyError(
            f"Unknown state: {name}. Available: {list(PSYCHOLOGICAL_STATES.keys())}"
        )
    return PSYCHOLOGICAL_STATES[name]


def get_states_by_category(category: StateCategory) -> Dict[str, APGIParameters]:
    """Retrieve all states belonging to a category"""
    return {
        name: params
        for name, params in PSYCHOLOGICAL_STATES.items()
        if STATE_CATEGORIES.get(name) == category
    }


def compare_states(state1: str, state2: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Compare two states and return parameter differences.

    Returns:
        Dict mapping parameter names to (state1_value, state2_value, difference)
    """
    p1 = get_state(state1)
    p2 = get_state(state2)

    params = [
        "Pi_e",
        "Pi_i_baseline",
        "Pi_i_eff",
        "theta_t",
        "S_t",
        "M_ca",
        "beta",
        "z_e",
        "z_i",
    ]

    comparison = {}
    for param in params:
        v1 = getattr(p1, param)
        v2 = getattr(p2, param)
        comparison[param] = (v1, v2, v2 - v1)

    return comparison


def find_nearest_state(params: APGIParameters) -> Tuple[str, float]:
    """
    Find the psychological state nearest to given parameters.

    Uses Euclidean distance in normalized parameter space.

    Returns:
        (state_name, distance)
    """
    # Normalization ranges (approximate)
    ranges = {
        "Pi_e": (0.1, 10.0),
        "Pi_i_baseline": (0.1, 10.0),
        "Pi_i_eff": (0.1, 10.0),
        "theta_t": (-3.0, 3.0),
        "S_t": (0.0, 50.0),
        "M_ca": (-2.0, 2.0),
        "beta": (0.3, 0.8),
        "z_e": (0.0, 3.5),
        "z_i": (0.0, 3.5),
    }

    def normalize(value: float, param: str) -> float:
        min_v, max_v = ranges[param]
        return (value - min_v) / (max_v - min_v)

    def distance(p1: APGIParameters, p2: APGIParameters) -> float:
        total = 0.0
        for param in ranges.keys():
            v1 = normalize(getattr(p1, param), param)
            v2 = normalize(getattr(p2, param), param)
            total += (v1 - v2) ** 2
        return np.sqrt(total)

    min_dist = float("inf")
    nearest = None

    for name, state_params in PSYCHOLOGICAL_STATES.items():
        d = distance(params, state_params)
        if d < min_dist:
            min_dist = d
            nearest = name

    return nearest, min_dist


def compute_transition_cost(from_state: str, to_state: str) -> Dict[str, float]:
    """
    Compute the parameter changes required to transition between states.

    Returns:
        Dict mapping parameter names to required change magnitudes
    """
    comparison = compare_states(from_state, to_state)

    # Weight certain parameters as more "costly" to change
    weights = {
        "Pi_e": 1.0,  # Precision shifts are moderate cost
        "Pi_i_baseline": 1.2,  # Interoceptive precision harder to shift
        "Pi_i_eff": 0.8,  # Effective is derived, less "real" cost
        "theta_t": 1.5,  # Threshold changes are effortful
        "S_t": 0.5,  # Accumulated signal is outcome, not lever
        "M_ca": 1.3,  # Somatic bias is sticky
        "beta": 2.0,  # Trait-like, hardest to change
        "z_e": 0.7,  # Environment-dependent
        "z_i": 0.8,  # Body-dependent
    }

    costs = {}
    total_cost = 0.0

    for param, (v1, v2, diff) in comparison.items():
        cost = abs(diff) * weights.get(param, 1.0)
        costs[param] = cost
        total_cost += cost

    costs["total"] = total_cost
    return costs


def get_transition_pathway(from_state: str, to_state: str) -> List[str]:
    """
    Suggest intermediate states for gradual transition.

    Uses parameter interpolation to find natural waypoints.
    """
    p1 = get_state(from_state)
    p2 = get_state(to_state)

    # Generate interpolated parameter sets
    pathway = [from_state]

    for alpha in [0.33, 0.66]:
        interpolated = create_apgi_params(
            Pi_e=p1.Pi_e + alpha * (p2.Pi_e - p1.Pi_e),
            Pi_i_baseline=p1.Pi_i_baseline
            + alpha * (p2.Pi_i_baseline - p1.Pi_i_baseline),
            M_ca=p1.M_ca + alpha * (p2.M_ca - p1.M_ca),
            beta=np.clip(p1.beta + alpha * (p2.beta - p1.beta), 0.3, 0.8),
            z_e=p1.z_e + alpha * (p2.z_e - p1.z_e),
            z_i=p1.z_i + alpha * (p2.z_i - p1.z_i),
            theta_t=p1.theta_t + alpha * (p2.theta_t - p1.theta_t),
        )
        nearest, _ = find_nearest_state(interpolated)
        if nearest not in pathway and nearest != to_state:
            pathway.append(nearest)

    pathway.append(to_state)
    return pathway


def get_state_summary(name: str) -> str:
    """Generate a human-readable summary of a state's parameters"""
    params = get_state(name)
    category = STATE_CATEGORIES.get(name, StateCategory.UNELABORATED)
    ignition_prob = params.compute_ignition_probability()

    summary = f"""
═══════════════════════════════════════════════════════════════════
State: {name.upper().replace('_', ' ')}
Category: {category.display_name}
═══════════════════════════════════════════════════════════════════

PRECISION PARAMETERS
────────────────────────────────────────────────────────────────────
  Exteroceptive (Π_e):        {params.Pi_e:5.2f}  {"█" * int(params.Pi_e)}
  Interoceptive baseline:     {params.Pi_i_baseline:5.2f}  {"█" * int(params.Pi_i_baseline)}
  Interoceptive effective:    {params.Pi_i_eff:5.2f}  {"█" * int(params.Pi_i_eff)}

PREDICTION ERROR
────────────────────────────────────────────────────────────────────
  Exteroceptive (z_e):        {params.z_e:5.2f}  {"▓" * int(params.z_e * 3)}
  Interoceptive (z_i):        {params.z_i:5.2f}  {"▓" * int(params.z_i * 3)}

THRESHOLD & SOMATIC
────────────────────────────────────────────────────────────────────
  Ignition threshold (θ_t):   {params.theta_t:+5.2f}  {"↑" if params.theta_t > 0 else "↓"} {"▒" * abs(int(params.theta_t * 2))}
  Somatic marker (M_ca):      {params.M_ca:+5.2f}  {"+" if params.M_ca > 0 else "-"} {"░" * abs(int(params.M_ca * 2))}
  Somatic gain (β):           {params.beta:5.2f}

DERIVED VALUES
────────────────────────────────────────────────────────────────────
  Accumulated surprise (S_t): {params.S_t:6.2f}
  Ignition probability:       {ignition_prob:6.2%}
  
Formula verification:
  Π_i_eff = Π_i_baseline · exp(β·M):  {"✓" if params.verify_Pi_i_eff() else "✗"}
  S_t = Π_e·|z_e| + Π_i_eff·|z_i|:    {"✓" if params.verify_S_t() else "✗"}
═══════════════════════════════════════════════════════════════════
"""
    return summary


def generate_state_comparison_table(states: List[str]) -> str:
    """Generate a formatted comparison table for multiple states"""
    headers = [
        "State",
        "Π_e",
        "Π_i_eff",
        "θ_t",
        "S_t",
        "M_ca",
        "β",
        "z_e",
        "z_i",
        "P(ign)",
    ]

    rows = []
    for name in states:
        p = get_state(name)
        rows.append(
            [
                name[:20],
                f"{p.Pi_e:.1f}",
                f"{p.Pi_i_eff:.1f}",
                f"{p.theta_t:+.1f}",
                f"{p.S_t:.1f}",
                f"{p.M_ca:+.1f}",
                f"{p.beta:.2f}",
                f"{p.z_e:.1f}",
                f"{p.z_i:.1f}",
                f"{p.compute_ignition_probability():.0%}",
            ]
        )

    # Format table
    col_widths = [
        max(len(str(row[i])) for row in [headers] + rows) + 2
        for i in range(len(headers))
    ]

    lines = []
    lines.append("┌" + "┬".join("─" * w for w in col_widths) + "┐")
    lines.append(
        "│"
        + "│".join(headers[i].center(col_widths[i]) for i in range(len(headers)))
        + "│"
    )
    lines.append("├" + "┼".join("─" * w for w in col_widths) + "┤")

    for row in rows:
        lines.append(
            "│"
            + "│".join(str(row[i]).center(col_widths[i]) for i in range(len(row)))
            + "│"
        )

    lines.append("└" + "┴".join("─" * w for w in col_widths) + "┘")

    return "\n".join(lines)


# =============================================================================
# ENHANCED DEMONSTRATION WITH VISUALIZATIONS
# =============================================================================


def run_enhanced_demo():
    """Run enhanced demonstration with visualizations"""
    print("\n" + "=" * 70)
    print("🎨 APGI PSYCHOLOGICAL STATE LIBRARY - ENHANCED DEMONSTRATION")
    print("=" * 70)

    # Check visualization capabilities
    has_viz = PLOTLY_AVAILABLE and PANDAS_AVAILABLE
    if has_viz:
        print("✅ Visualization packages available")
        print("   Plotly:", PLOTLY_AVAILABLE)
        print("   Pandas:", PANDAS_AVAILABLE)
    else:
        print("⚠️  Some visualization packages not available")
        print("   Install with: pip install plotly pandas matplotlib")

    # Create visualizer instance
    if has_viz:
        print("\n📊 CREATING VISUALIZER INSTANCE")
        print("-" * 40)
        visualizer = APGIVisualizer(PSYCHOLOGICAL_STATES, STATE_CATEGORIES)
        print(
            f"   Loaded {len(PSYCHOLOGICAL_STATES)} states across {len(set(STATE_CATEGORIES.values()))} categories"
        )

    # 1. Basic state retrieval
    print("\n1. BASIC STATE RETRIEVAL")
    print("-" * 40)
    fear_params = get_state("fear")
    print(f"Fear parameters:")
    print(f"  Π_e = {fear_params.Pi_e}")
    print(f"  Π_i_eff = {fear_params.Pi_i_eff}")
    print(f"  θ_t = {fear_params.theta_t}")
    print(f"  M_ca = {fear_params.M_ca}")
    print(f"  P(ignition) = {fear_params.compute_ignition_probability():.2%}")

    # 2. Category retrieval
    print("\n2. CATEGORY RETRIEVAL")
    print("-" * 40)
    optimal_states = get_states_by_category(StateCategory.OPTIMAL_FUNCTIONING)
    print(f"Optimal functioning states: {list(optimal_states.keys())}")

    # 3. State comparison
    print("\n3. STATE COMPARISON: Fear vs Anxiety")
    print("-" * 40)
    comparison = compare_states("fear", "anxiety")
    print(f"{'Parameter':<15} {'Fear':>8} {'Anxiety':>8} {'Δ':>8}")
    print("-" * 40)
    for param, (v1, v2, diff) in comparison.items():
        print(f"{param:<15} {v1:>8.2f} {v2:>8.2f} {diff:>+8.2f}")

    # 4. State summary
    print("\n4. DETAILED STATE SUMMARY")
    print(get_state_summary("flow"))

    # 5. Comparison table
    print("\n5. AVERSIVE STATES COMPARISON TABLE")
    print(
        generate_state_comparison_table(
            ["fear", "anxiety", "anger", "guilt", "shame", "loneliness", "overwhelm"]
        )
    )

    # 6. Find nearest state
    print("\n6. NEAREST STATE DETECTION")
    print("-" * 40)
    test_params = create_apgi_params(
        Pi_e=7.0, Pi_i_baseline=3.0, M_ca=1.6, beta=0.7, z_e=2.2, z_i=1.8, theta_t=-2.0
    )
    nearest, distance = find_nearest_state(test_params)
    print(f"Test parameters most similar to: {nearest} (distance: {distance:.3f})")

    # 7. Transition pathway
    print("\n7. STATE TRANSITION PATHWAY")
    print("-" * 40)
    pathway = get_transition_pathway("anxiety", "calm")
    print(f"Suggested pathway from 'anxiety' to 'calm':")
    print(f"  {' → '.join(pathway)}")

    # 8. Transition cost
    print("\n8. TRANSITION COST ANALYSIS")
    print("-" * 40)
    costs = compute_transition_cost("anxiety", "calm")
    print(f"Cost to transition from 'anxiety' to 'calm':")
    for param, cost in sorted(
        costs.items(), key=lambda x: -x[1] if x[0] != "total" else 0
    ):
        if param != "total":
            print(f"  {param:<15}: {cost:.2f}")
    print(f"  {'TOTAL':<15}: {costs['total']:.2f}")

    # 9. Visualization demonstrations
    if has_viz:
        print("\n🎨 VISUALIZATION DEMONSTRATIONS")
        print("-" * 40)

        # 9.1 Generate visualizations
        print("9.1 Generating comprehensive visualization report...")
        try:
            report_files = visualizer.export_visualization_report("./apgi_report")
            print(f"✅ Report generated with {len(report_files)} files")
            print(f"   Main index: {report_files.get('index', 'Not generated')}")

            # 9.2 Show example visualization
            print("\n9.2 Creating example 3D network visualization...")
            fig = visualizer.plot_state_network_3d()
            if fig:
                print("✅ 3D Network visualization created")
                print("   Use embedded viewer or save to file")

            # 9.3 Create state dashboard
            print("\n9.3 Creating state dashboard for 'flow'...")
            dashboard = visualizer.create_state_summary_dashboard("flow")
            if dashboard:
                print("✅ State dashboard created")

        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
            print("   Ensure all dependencies are installed: pip install plotly pandas")

    # 10. Library statistics
    print("\n10. LIBRARY STATISTICS")
    print("-" * 40)

    all_pi_e = [p.Pi_e for p in PSYCHOLOGICAL_STATES.values()]
    all_theta = [p.theta_t for p in PSYCHOLOGICAL_STATES.values()]
    all_m_ca = [p.M_ca for p in PSYCHOLOGICAL_STATES.values()]
    all_ignition = [
        p.compute_ignition_probability() for p in PSYCHOLOGICAL_STATES.values()
    ]

    print(f"Total states: {len(PSYCHOLOGICAL_STATES)}")
    print(
        f"\nΠ_e range: {min(all_pi_e):.1f} - {max(all_pi_e):.1f} (mean: {np.mean(all_pi_e):.2f})"
    )
    print(
        f"θ_t range: {min(all_theta):+.1f} - {max(all_theta):+.1f} (mean: {np.mean(all_theta):+.2f})"
    )
    print(
        f"M_ca range: {min(all_m_ca):+.1f} - {max(all_m_ca):+.1f} (mean: {np.mean(all_m_ca):+.2f})"
    )
    print(
        f"P(ignition) range: {min(all_ignition):.0%} - {max(all_ignition):.0%} (mean: {np.mean(all_ignition):.0%})"
    )

    # 11. States by category count
    print("\n11. STATES BY CATEGORY")
    print("-" * 40)
    for category in StateCategory:
        states = get_states_by_category(category)
        print(f"  {category.display_name:<25}: {len(states):>2} states")

    print("\n" + "=" * 70)
    print("✅ ENHANCED DEMONSTRATION COMPLETE")

    if has_viz:
        print("\n📁 Visualizations saved to './apgi_report/' directory")
        print("🔧 Use APGIVisualizer class for custom visualizations")

    print("=" * 70)


# =============================================================================
# QUICK START FUNCTION
# =============================================================================


def quick_start():
    """Quick start function for new users"""
    print("\n⚡ APGI VISUALIZATION QUICK START")
    print("=" * 50)

    # Check dependencies
    if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
        print("\n📦 Required packages not installed.")
        print("   Run: pip install plotly pandas matplotlib")
        print("   Then restart your Python session.")
        return

    print("\n1. Creating visualizer instance...")
    visualizer = APGIVisualizer(PSYCHOLOGICAL_STATES, STATE_CATEGORIES)

    print("2. Generating example visualization...")
    fig = visualizer.plot_state_network_3d()

    print("3. Exporting comprehensive report...")
    report = visualizer.export_visualization_report()

    print("\n✅ Quick start complete!")
    print(f"\n📊 Available visualizations:")
    print(f"   • 3D Network: {report.get('network', 'Generated')}")
    print(f"   • Ignition Landscape: {report.get('landscape', 'Generated')}")
    print(f"   • Correlation Heatmap: {report.get('correlations', 'Generated')}")
    print(f"   • Transition Pathways: {report.get('transitions', 'Generated')}")
    print(f"   • Report Index: {report.get('index', 'Generated')}")

    print("\n🔧 Try these commands:")
    print("   visualizer.plot_state_radar(['flow', 'anxiety', 'calm'])")
    print("   visualizer.create_state_summary_dashboard('flow')")
    print("   visualizer.plot_ignition_landscape('focus')")


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main entry point for the APGI Psychological States application."""
    # Run validation first
    print("\n🔍 Validating APGI state library...")

    def validate_all_states() -> Dict[str, Dict[str, bool]]:
        """Validate all state parameters and formulas"""
        results = {}

        for name, params in PSYCHOLOGICAL_STATES.items():
            results[name] = {
                "Pi_i_eff_valid": params.verify_Pi_i_eff(),
                "S_t_valid": params.verify_S_t(),
                "ignition_prob_valid": 0.0
                <= params.compute_ignition_probability()
                <= 1.0,
                "bounds_valid": True,  # Already checked in __post_init__
            }

        return results

    results = validate_all_states()
    all_valid = all(all(checks.values()) for checks in results.values())

    if all_valid:
        print("✅ All states validated successfully")

        # Main program loop
        while True:
            # Ask user what they want to run
            print("\n" + "=" * 50)
            print("APGI PSYCHOLOGICAL STATES VISUALIZATION SYSTEM")
            print("=" * 50)
            print("\nChoose an option:")
            print("1. Run enhanced demonstration (recommended)")
            print("2. Quick start with visualizations")
            print("3. Generate full visualization report")
            print("4. Launch Interactive GUI")
            print("5. Exit")

            try:
                choice = input("\nEnter choice (1-5): ").strip()

                if choice == "1":
                    run_enhanced_demo()
                elif choice == "2":
                    quick_start()
                elif choice == "3":
                    if PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
                        visualizer = APGIVisualizer(
                            PSYCHOLOGICAL_STATES, STATE_CATEGORIES
                        )
                        report = visualizer.export_visualization_report()
                        print(f"\n✅ Report generated with {len(report)} files")
                        print(f"📂 Report generated in './apgi_report/' directory")
                    else:
                        print("❌ Visualization packages not available")
                        print("   Install with: pip install plotly pandas")
                elif choice == "4":
                    if TKINTER_AVAILABLE and PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
                        try:
                            gui = APGIVisualizerGUI()
                            gui.run()
                            # After GUI closes, prompt to continue
                            input("\nPress Enter to return to the main menu...")
                        except Exception as e:
                            print(f"❌ Error starting GUI: {e}")
                            print("Falling back to command line interface...")
                            run_enhanced_demo()
                    else:
                        print("❌ GUI dependencies not available")
                        print("   Install with: pip install plotly pandas")
                        if not TKINTER_AVAILABLE:
                            print("   Tkinter is required for the GUI interface")
                        input("\nPress Enter to continue...")
                elif choice == "5":
                    print("👋 Exiting")
                    return
                else:
                    print("⚠️  Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\n👋 Interrupted by user")
                return
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Running basic validation report instead...")
                run_enhanced_demo()
    else:
        print("❌ Some states failed validation")
        print_validation_report()


if __name__ == "__main__":
    main()


def print_validation_report():
    """Print a validation report for all states"""
    results = validate_all_states()

    print("\n" + "=" * 70)
    print("APGI STATE LIBRARY VALIDATION REPORT")
    print("=" * 70)

    all_valid = True
    for name, checks in results.items():
        status = "✓" if all(checks.values()) else "✗"
        if not all(checks.values()):
            all_valid = False

        failed = [k for k, v in checks.items() if not v]
        if failed:
            print(f"  {status} {name}: FAILED - {', '.join(failed)}")
        else:
            print(f"  {status} {name}")

    print("-" * 70)
    print(f"Total states: {len(results)}")
    print(f"Overall status: {'ALL VALID ✓' if all_valid else 'SOME FAILURES ✗'}")
    print("=" * 70)
