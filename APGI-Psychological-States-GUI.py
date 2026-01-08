"""
APGI Psychological State Parameter Library with Advanced Visualizations
Enhanced GUI with Embedded Visualization Panel
=============================================================================

Complete parameter mappings for 51 psychological states with embedded 
interactive visualizations displayed exclusively within the GUI application.

All visualizations are rendered directly in the right panel of the application
with no external browser dependencies, save options, or display capabilities.

=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum, auto
import warnings
from math import pi
import datetime
import os
import tempfile
import json
from pathlib import Path

# Visualization imports with graceful fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
    pio.templates.default = "plotly_white"
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    from matplotlib.patches import Circle, Wedge, Polygon
    import matplotlib.cm as cm
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
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
    from tkinter import ttk, messagebox
    from tkinter.scrolledtext import ScrolledText
    import threading
    from traceback import format_exc
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    warnings.warn("Tkinter not available for GUI interface")

# Try to import tkinterweb for HTML rendering, fallback to built-in approach
try:
    try:
        from tkinterweb import HTMLFrame
        TKINTERWEB_AVAILABLE = True
    except ImportError:
        TKINTERWEB_AVAILABLE = False
except Exception as e:
    TKINTERWEB_AVAILABLE = False
    print(f"Warning: Could not import tkinterweb: {e}")

# No external browser dependencies or save options needed - all visualizations are embedded


@dataclass
class APGIParameters:
    """APGI parameter set with proper type safety"""
    Pi_e: float
    Pi_i_baseline: float
    Pi_i_eff: float
    theta_t: float
    S_t: float
    M_ca: float
    beta: float
    z_e: float
    z_i: float
    
    def __post_init__(self):
        """Validate parameters are within physiological bounds"""
        assert 0.1 <= self.Pi_e <= 10.0, f"Pi_e must be in [0.1, 10], got {self.Pi_e}"
        assert 0.1 <= self.Pi_i_baseline <= 10.0, f"Pi_i_baseline must be in [0.1, 10], got {self.Pi_i_baseline}"
        assert 0.1 <= self.Pi_i_eff <= 10.0, f"Pi_i_eff must be in [0.1, 10], got {self.Pi_i_eff}"
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
            'Pi_e': self.Pi_e,
            'Pi_i_baseline': self.Pi_i_baseline,
            'Pi_i_eff': self.Pi_i_eff,
            'theta_t': self.theta_t,
            'S_t': self.S_t,
            'M_ca': self.M_ca,
            'beta': self.beta,
            'z_e': self.z_e,
            'z_i': self.z_i,
            'ignition_probability': self.compute_ignition_probability()
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
    color: Optional[str] = None


class StateCategory(Enum):
    """Categories of psychological states with colors"""
    OPTIMAL_FUNCTIONING = ("#2E86AB", "Optimal Functioning")
    POSITIVE_AFFECTIVE = ("#48BF84", "Positive Affective")
    COGNITIVE_ATTENTIONAL = ("#FF9F1C", "Cognitive/Attentional")
    AVERSIVE_AFFECTIVE = ("#E63946", "Aversive Affective")
    PATHOLOGICAL_EXTREME = ("#7209B7", "Pathological/Extreme")
    ALTERED_BOUNDARY = ("#8338EC", "Altered/Boundary")
    TRANSITIONAL_CONTEXTUAL = ("#06D6A0", "Transitional/Contextual")
    UNELABORATED = ("#8D99AE", "Unelaborated")
    
    def __init__(self, color: str, display_name: str):
        self.color = color
        self.display_name = display_name


# =============================================================================
# ENHANCED EMBEDDED VISUALIZATION ENGINE
# =============================================================================

class EmbeddedVisualizationRenderer:
    """Render Plotly visualizations for embedded display in Tkinter"""
    
    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize renderer with optional temp directory"""
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="apgi_viz_")
        else:
            self.temp_dir = temp_dir
            os.makedirs(self.temp_dir, exist_ok=True)
        
        self.current_file = None
    
    def render_figure_to_html(self, fig: go.Figure, filename: str = "current.html") -> str:
        """
        Render a Plotly figure to HTML with embedded resources.
        
        Returns:
            Path to the generated HTML file
        """
        filepath = os.path.join(self.temp_dir, filename)
        
        # Create HTML with responsive sizing and proper scaling
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APGI Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        html, body {{
            width: 100%;
            height: 100%;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f8f9fa;
        }}
        #plot {{
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .loading {{
            font-size: 16px;
            color: #666;
        }}
        .info-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.95);
            padding: 12px 16px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            font-size: 12px;
            max-width: 300px;
            z-index: 1000;
            display: none;
        }}
        .info-panel.show {{
            display: block;
        }}
    </style>
</head>
<body>
    <div id="plot" class="loading">Loading visualization...</div>
    <div class="info-panel" id="info-panel"></div>
    
    <script>
        let plotData = null;
        let layout = null;
        
        // Function to initialize plot
        function initPlot() {{
            const plotJson = {json.dumps(fig.to_json())};
            const figure = JSON.parse(plotJson);
            
            // Ensure responsive sizing
            figure.layout.autosize = true;
            figure.layout.margin = {{l: 50, r: 50, b: 50, t: 50, pad: 4}};
            figure.layout.paper_bgcolor = '#f8f9fa';
            figure.layout.plot_bgcolor = 'white';
            
            // Create the plot with responsive config
            Plotly.newPlot('plot', figure.data, figure.layout, {{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            }});
            
            // Handle window resizing
            window.addEventListener('resize', function() {{
                Plotly.Plots.resize('plot');
            }});
            
            // Remove loading message
            const plotDiv = document.getElementById('plot');
            if (plotDiv.classList.contains('loading')) {{
                plotDiv.classList.remove('loading');
            }}
        }}
        
        // Initialize on load
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initPlot);
        }} else {{
            initPlot();
        }}
        
        // Hover info handler
        document.getElementById('plot').addEventListener('plotly_hover', function(data) {{
            const infoPanel = document.getElementById('info-panel');
            if (data.points && data.points.length > 0) {{
                const point = data.points[0];
                let text = '';
                if (point.customdata) {{
                    text = point.customdata;
                }} else if (point.text) {{
                    text = point.text;
                }} else {{
                    text = 'Hover over data points for information';
                }}
                infoPanel.textContent = text;
                infoPanel.classList.add('show');
            }}
        }});
        
        document.getElementById('plot').addEventListener('plotly_unhover', function() {{
            document.getElementById('info-panel').classList.remove('show');
        }});
    </script>
</body>
</html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.current_file = filepath
        return filepath


class APGIVisualizer:
    """Modern, eloquent visualizations for APGI psychological states"""
    
    PALETTES = {
        'categorical': {
            StateCategory.OPTIMAL_FUNCTIONING: '#2E86AB',
            StateCategory.POSITIVE_AFFECTIVE: '#48BF84',
            StateCategory.COGNITIVE_ATTENTIONAL: '#FF9F1C',
            StateCategory.AVERSIVE_AFFECTIVE: '#E63946',
            StateCategory.PATHOLOGICAL_EXTREME: '#7209B7',
            StateCategory.ALTERED_BOUNDARY: '#8338EC',
            StateCategory.TRANSITIONAL_CONTEXTUAL: '#06D6A0',
            StateCategory.UNELABORATED: '#8D99AE'
        },
        'sequential': ['#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087', '#f95d6a', '#ff7c43', '#ffa600'],
        'diverging': ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    }
    
    def __init__(self, states_dict: Dict[str, APGIParameters], 
                 categories_dict: Dict[str, StateCategory]):
        """Initialize visualizer with states and categories."""
        self.states = states_dict
        self.categories = categories_dict
        self.renderer = EmbeddedVisualizationRenderer()
        
        if PANDAS_AVAILABLE:
            self.df = self._create_dataframe()
        else:
            self.df = None
    
    def _create_dataframe(self) -> 'pd.DataFrame':
        """Create a pandas DataFrame for visualization"""
        data = []
        for name, params in self.states.items():
            row = params.to_dict()
            row['name'] = name
            row['category'] = self.categories.get(name, StateCategory.UNELABORATED).name
            row['category_display'] = self.categories.get(name, StateCategory.UNELABORATED).display_name
            row['category_color'] = self.categories.get(name, StateCategory.UNELABORATED).color
            data.append(row)
        
        df = pd.DataFrame(data)
        df['precision_ratio'] = df['Pi_i_eff'] / df['Pi_e']
        df['somatic_engagement'] = df['M_ca'] * df['beta']
        df['prediction_error_total'] = df['z_e'] + df['z_i']
        
        return df
    
    def plot_state_network_3d(self, 
                            dimension1: str = 'Pi_e',
                            dimension2: str = 'Pi_i_eff', 
                            dimension3: str = 'theta_t') -> Optional[go.Figure]:
        """Create an interactive 3D network visualization of psychological states."""
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            warnings.warn("Plotly or Pandas not available")
            return None
        
        fig = go.Figure()
        
        for idx, row in self.df.iterrows():
            state_name = row['name']
            size = 15 + (row['S_t'] * 2)
            color = row['category_color']
            
            fig.add_trace(go.Scatter3d(
                x=[row[dimension1]], y=[row[dimension2]], z=[row[dimension3]],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color,
                    opacity=0.8,
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                text=[state_name.replace('_', ' ').title()],
                textposition="top center",
                hoverinfo='text',
                hovertext=self._create_hover_text(state_name, row),
                name=state_name,
                showlegend=False
            ))
        
        fig.update_layout(
            title="APGI Psychological State Network (3D)",
            scene=dict(
                xaxis_title=dimension1,
                yaxis_title=dimension2,
                zaxis_title=dimension3,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                bgcolor='rgba(240,240,240,0.9)'
            ),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=0, r=0, b=0, t=40),
            hovermode='closest',
            template='plotly_white'
        )
        
        self._add_category_legend(fig)
        return fig
    
    def plot_ignition_landscape(self,
                              focus_state: Optional[str] = None,
                              parameter1: str = 'Pi_e',
                              parameter2: str = 'theta_t',
                              resolution: int = 50) -> Optional[go.Figure]:
        """Create a 3D ignition probability landscape."""
        if not PLOTLY_AVAILABLE:
            warnings.warn("Plotly not available")
            return None
        
        p1_range = np.linspace(self.df[parameter1].min() * 0.8, 
                              self.df[parameter1].max() * 1.2, resolution)
        p2_range = np.linspace(self.df[parameter2].min() * 0.8,
                              self.df[parameter2].max() * 1.2, resolution)
        
        P1, P2 = np.meshgrid(p1_range, p2_range)
        
        avg_z_e = self.df['z_e'].mean()
        avg_z_i = self.df['z_i'].mean()
        avg_Pi_i_eff = self.df['Pi_i_eff'].mean()
        
        S_t = P1 * avg_z_e + avg_Pi_i_eff * avg_z_i
        Z = 1.0 / (1.0 + np.exp(-(S_t - P2)))
        
        fig = go.Figure(data=[go.Surface(
            z=Z, x=P1, y=P2,
            colorscale='Viridis',
            opacity=0.8,
            contours={"z": {"show": True, "usecolormap": True, 
                           "highlightcolor": "limegreen", "project": {"z": True}}},
            hoverongaps=False,
            hovertemplate='%{x:.2f} vs %{y:.2f}<br>P(ignition): %{z:.3f}<extra></extra>'
        )])
        
        # Add state markers
        scatter_x = []
        scatter_y = []
        scatter_z = []
        scatter_colors = []
        scatter_names = []
        
        for idx, row in self.df.iterrows():
            S_t_actual = row['Pi_e'] * row['z_e'] + row['Pi_i_eff'] * row['z_i']
            ignition_prob = 1.0 / (1.0 + np.exp(-(S_t_actual - row['theta_t'])))
            
            scatter_x.append(row[parameter1])
            scatter_y.append(row[parameter2])
            scatter_z.append(ignition_prob)
            scatter_colors.append(row['category_color'])
            scatter_names.append(row['name'])
        
        fig.add_trace(go.Scatter3d(
            x=scatter_x, y=scatter_y, z=scatter_z,
            mode='markers+text',
            marker=dict(size=8, color=scatter_colors, opacity=1.0, line=dict(width=2, color='white')),
            text=[name.replace('_', ' ').title() for name in scatter_names],
            textposition="top center",
            hoverinfo='text',
            hovertext=[f"{name}<br>P(ignition)={z:.2%}" for name, z in zip(scatter_names, scatter_z)],
            name='Psychological States'
        ))
        
        fig.update_layout(
            title="Ignition Probability Landscape",
            scene=dict(
                xaxis_title=parameter1,
                yaxis_title=parameter2,
                zaxis_title='P(Ignition)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=40),
            template='plotly_white'
        )
        
        return fig
    
    def plot_state_radar(self, state_names: List[str], normalize: bool = True) -> Optional[go.Figure]:
        """Create a radar chart comparing multiple states."""
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            warnings.warn("Plotly or Pandas not available")
            return None
        
        params = ['Pi_e', 'Pi_i_eff', 'theta_t', 'M_ca', 'S_t', 'z_e', 'z_i', 'beta']
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
                    all_values = self.df[param]
                    if param in ['theta_t', 'M_ca']:
                        value = (value - all_values.min()) / (all_values.max() - all_values.min())
                    else:
                        value = value / all_values.max()
                
                values.append(value)
            
            values.append(values[0])
            color = self.categories.get(state_name, StateCategory.UNELABORATED).color
            
            if color.startswith('#'):
                if len(color) == 7:
                    fill_color = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.25)"
                    line_color = color
                else:
                    fill_color = f"rgba(128, 128, 128, 0.25)"
                    line_color = "#808080"
            else:
                fill_color = f"rgba(128, 128, 128, 0.25)"
                line_color = "#808080"
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                fillcolor=fill_color,
                line=dict(color=line_color, width=2),
                name=state_name.replace('_', ' ').title(),
                hoverinfo='text',
                hovertext=self._create_hover_text(state_name, None)
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.1] if normalize else None)),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
            title="State Comparison Radar Chart",
            margin=dict(l=100, r=100, b=50, t=50),
            template='plotly_white'
        )
        
        return fig
    
    def plot_parameter_correlation_heatmap(self, 
                                         parameters: Optional[List[str]] = None) -> Optional[go.Figure]:
        """Create a correlation heatmap of APGI parameters."""
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            warnings.warn("Plotly or Pandas not available")
            return None
        
        if parameters is None:
            parameters = ['Pi_e', 'Pi_i_baseline', 'Pi_i_eff', 'theta_t', 
                         'S_t', 'M_ca', 'beta', 'z_e', 'z_i', 'ignition_probability']
        
        corr_matrix = self.df[parameters].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=parameters,
            y=parameters,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="APGI Parameter Correlation Matrix",
            xaxis_title="Parameters",
            yaxis_title="Parameters",
            width=700,
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_state_summary_dashboard(self, state_name: str) -> Optional[go.Figure]:
        """Create a comprehensive dashboard for a single state."""
        if not PLOTLY_AVAILABLE or state_name not in self.states:
            warnings.warn("Plotly not available or state not found")
            return None
        
        params = self.states[state_name]
        category = self.categories.get(state_name, StateCategory.UNELABORATED)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Parameter Profile",
                "Ignition Dynamics",
                "Category Comparison",
                "State Distribution"
            ),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "radar"}, {"type": "histogram"}]]
        )
        
        # 1. Parameter Profile
        param_names = ['Pi_e', 'Pi_i_baseline', 'Pi_i_eff', 'theta_t', 'M_ca', 'beta', 'z_e', 'z_i']
        param_values = [getattr(params, p) for p in param_names]
        param_colors = ['#2E86AB' if v > 0 else '#E63946' for v in param_values]
        
        fig.add_trace(
            go.Bar(x=param_names, y=param_values, marker_color=param_colors, name="Parameters",
                   hovertemplate="%{x}: %{y:.2f}<extra></extra>"),
            row=1, col=1
        )
        
        # 2. Ignition Dynamics
        S_t_range = np.linspace(0, params.S_t * 2, 100)
        ignition_probs = 1.0 / (1.0 + np.exp(-(S_t_range - params.theta_t)))
        
        fig.add_trace(
            go.Scatter(x=S_t_range, y=ignition_probs, mode='lines',
                      line=dict(color=category.color, width=3),
                      name="Ignition Probability", fill='tozeroy',
                      fillcolor=category.color + '20'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=[params.S_t], y=[params.compute_ignition_probability()],
                      mode='markers', marker=dict(size=15, color='gold', line=dict(width=2, color='black')),
                      name="Current State",
                      hovertext=f"S_t={params.S_t:.2f}, P={params.compute_ignition_probability():.2%}"),
            row=1, col=2
        )
        
        # 3. Category Comparison
        category_states = [name for name, cat in self.categories.items() 
                          if cat == category and name != state_name][:4]
        if category_states:
            for comp_state in [state_name] + category_states:
                comp_params = self.states[comp_state]
                values = [
                    comp_params.Pi_e / 10,
                    comp_params.Pi_i_eff / 10,
                    (comp_params.theta_t + 3) / 6,
                    (comp_params.M_ca + 2) / 4,
                    comp_params.compute_ignition_probability()
                ]
                values.append(values[0])
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=['Π_e', 'Π_i_eff', 'θ_t', 'M_ca', 'P(ign)', 'Π_e'],
                        fill='toself' if comp_state == state_name else 'none',
                        line=dict(width=2 if comp_state == state_name else 1),
                        opacity=1.0 if comp_state == state_name else 0.6,
                        name=comp_state.replace('_', ' ').title(),
                        showlegend=True
                    ),
                    row=2, col=1
                )
        
        # 4. Distribution histogram
        all_ignition = [s.compute_ignition_probability() for s in self.states.values()]
        fig.add_trace(
            go.Histogram(x=all_ignition, nbinsx=20, name="P(ignition) Distribution",
                        marker_color='rgba(99,110,250,0.7)',
                        hovertemplate="P(ignition): %{x:.2%}<br>Count: %{y}<extra></extra>"),
            row=2, col=2
        )
        
        fig.add_vline(x=params.compute_ignition_probability(), line_dash="dash", line_color="red",
                     annotation_text=f"Current: {params.compute_ignition_probability():.0%}",
                     row=2, col=2)
        
        fig.update_xaxes(title_text="Parameters", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        
        fig.update_xaxes(title_text="Accumulated Surprise (S_t)", row=1, col=2)
        fig.update_yaxes(title_text="Ignition Probability", range=[0, 1], row=1, col=2)
        
        fig.update_polars(radialaxis_range=[0, 1], row=2, col=1)
        
        fig.update_xaxes(title_text="P(ignition)", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        fig.update_layout(
            title_text=f"APGI State Dashboard: {state_name.replace('_', ' ').title()}",
            showlegend=True,
            height=800,
            template="plotly_white"
        )
        
        return fig
    
    def _create_hover_text(self, state_name: str, row: Optional['pd.Series'] = None) -> str:
        """Create hover text for state visualization"""
        params = self.states[state_name]
        
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
        
        for category in StateCategory:
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color=category.color),
                name=category.display_name,
                showlegend=True
            ))
    


# =============================================================================
# ENHANCED GUI WITH EMBEDDED VISUALIZATION PANEL
# =============================================================================

class APGIVisualizerGUI:
    """Enhanced GUI for APGI Psychological States Visualization with Embedded Panel
    
    All visualizations are displayed exclusively in the embedded right panel
    with no external browser options, dependencies, or save capabilities.
    """
    
    def __init__(self):
        """Initialize the GUI application with embedded visualization support"""
        if not TKINTER_AVAILABLE:
            raise ImportError("Tkinter is required for GUI interface")
        if not PLOTLY_AVAILABLE or not PANDAS_AVAILABLE:
            raise ImportError("Plotly and Pandas are required for visualization")
        
        self.root = tk.Tk()
        self.root.title("APGI Psychological States Visualizer - Enhanced GUI")
        self.root.geometry("1400x900")
        
        try:
            self.visualizer = APGIVisualizer(PSYCHOLOGICAL_STATES, STATE_CATEGORIES)
            self.current_visualization = None
            self.current_html_file = None
            
            # Setup GUI
            self.setup_gui()
            self.populate_state_dropdowns()
            
            self.status_var.set("Ready - Select visualization type and click Generate")
            self.update_info("APGI Visualizer initialized successfully!\n\n"
                           "Available states: {}\n\n"
                           "Choose a visualization type and click 'Generate Visualization' to begin."
                           .format(len(PSYCHOLOGICAL_STATES)))
        except Exception as e:
            messagebox.showerror("Initialization Error", 
                              f"Failed to initialize visualizer: {str(e)}")
            self.root.destroy()
            raise
    
    def setup_gui(self):
        """Setup the enhanced GUI layout with embedded visualization panel"""
        # Main container with better layout
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🧠 APGI Psychological States Visualizer", 
                               font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 15))
        
        # Control Panel (Left) - Enhanced
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="12")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Visualization Type
        ttk.Label(control_frame, text="Visualization Type:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(5, 2))
        self.viz_type = ttk.Combobox(control_frame, values=[
            "3D State Network",
            "Ignition Landscape", 
            "State Radar Comparison",
            "Parameter Correlation Heatmap",
            "State Dashboard",
        ], state="readonly", font=('Arial', 9))
        self.viz_type.set("3D State Network")
        self.viz_type.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # State Selection
        ttk.Label(control_frame, text="Select State:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=(5, 2))
        self.state_var = tk.StringVar()
        self.state_combo = ttk.Combobox(control_frame, textvariable=self.state_var, state="readonly", font=('Arial', 9))
        self.state_combo.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Multiple States for Radar
        ttk.Label(control_frame, text="States to Compare\n(comma-separated):", font=('Arial', 9, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=(5, 2))
        self.states_text = tk.Text(control_frame, height=3, width=25, font=('Courier', 8))
        self.states_text.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.states_text.insert("1.0", "flow\nanxiety\ncalm")
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=6, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Buttons with better styling
        button_style = {'width': 20}
        
        ttk.Button(control_frame, text="Generate Visualization", 
                  command=self.generate_visualization).grid(row=7, column=0, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(control_frame, text="Clear Display", 
                  command=self.clear_display).grid(row=8, column=0, sticky=(tk.W, tk.E), pady=5)
        
        control_frame.columnconfigure(0, weight=1)
        
        # Visualization Panel (Right) - Enhanced with embedded display
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization Panel", padding="5")
        viz_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Create embedded web view
        self.embedded_display = EmbeddedDisplayPanel(viz_frame)
        self.embedded_display.pack(fill=tk.BOTH, expand=True)
        
        # Info Panel (Bottom) - Smaller
        info_frame = ttk.LabelFrame(main_frame, text="Information Panel", padding="8")
        info_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        self.info_text = tk.Text(info_frame, height=4, width=80, wrap=tk.WORD, font=('Arial', 9))
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.info_text['yscrollcommand'] = scrollbar.set
        
        # Status Bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, 
                             font=('Arial', 9))
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def populate_state_dropdowns(self):
        """Populate state selection dropdowns"""
        if not PSYCHOLOGICAL_STATES:
            self.status_var.set("Error: No states available")
            return
        
        state_names = sorted(PSYCHOLOGICAL_STATES.keys())
        self.state_combo['values'] = state_names
        
        if state_names:
            self.state_combo.set(state_names[0])
        
        self.status_var.set("Ready - Select visualization type and click Generate")
    
    def update_info(self, text):
        """Update the info text area"""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", text)
        self.info_text.config(state=tk.DISABLED)
    
    def generate_visualization(self):
        """Generate the selected visualization with embedded display"""
        viz_type = self.viz_type.get()
        self.status_var.set(f"Generating {viz_type}...")
        self.root.update()
        
        try:
            fig = None
            
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
                states = [s.strip() for s in states_text.split('\n') if s.strip()]
                if not states:
                    messagebox.showerror("Error", "Please enter states to compare")
                    return
                fig = self.visualizer.plot_state_radar(states)
                title = "State Comparison Radar"
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
            else:
                messagebox.showerror("Error", "Unknown visualization type")
                return
            
            if fig:
                self.current_visualization = fig
                
                # Render to HTML for tkinterweb
                if self.embedded_display.display_method == "tkinterweb":
                    html_file = self.visualizer.renderer.render_figure_to_html(fig, "current_viz.html")
                    self.current_html_file = html_file
                    self.embedded_display.load_html_file(html_file)
                else:
                    # Pass the actual figure to matplotlib fallback
                    self.embedded_display.display_plotly_figure(fig)
                
                self.status_var.set(f"✓ Generated {viz_type}")
                self.update_info(f"Visualization: {title}\n\n"
                               f"Use the controls on the left to generate different visualizations.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate visualization: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            self.update_info(f"Error generating visualization:\n{str(e)}\n\n{format_exc()}")
    
    def clear_display(self):
        """Clear the visualization panel"""
        self.embedded_display.clear()
        self.status_var.set("Display cleared")
        self.update_info("Visualization cleared. Select a new visualization type and click Generate.")
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()


class EmbeddedDisplayPanel(ttk.Frame):
    """Custom embedded panel for displaying HTML content exclusively within the GUI
    
    This panel handles all visualization rendering internally with no external
    browser dependencies, opening options, or save capabilities.
    """
    
    def __init__(self, parent, **kwargs):
        """Initialize the embedded display panel"""
        super().__init__(parent, **kwargs)
        
        # Try different display methods
        self.display_method = self._setup_display()
    
    def _setup_display(self):
        """Setup the display backend (tkinterweb or fallback)"""
        if TKINTERWEB_AVAILABLE:
            try:
                self.html_frame = HtmlFrame(self, messages_enabled=False)
                self.html_frame.pack(fill=tk.BOTH, expand=True)
                return "tkinterweb"
            except Exception as e:
                print(f"Failed to initialize HtmlFrame: {e}, using fallback")
        
        # Fallback display for when tkinterweb is not available
        self._setup_fallback_display()
        return "fallback"
    
    def _setup_fallback_display(self):
        """Setup fallback display using matplotlib canvas"""
        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True)
        
        if MATPLOTLIB_AVAILABLE:
            # Create matplotlib canvas for fallback rendering
            self.canvas_frame = frame
            self.matplotlib_canvas = None
            self.toolbar = None
            
            # Initial message
            self.info_label = ttk.Label(frame, 
                text="📊 Visualization Panel\n\n"
                     "Visualizations will be displayed here using matplotlib.\n\n"
                     "Generate a visualization to see it in this panel.",
                font=('Arial', 12), justify=tk.CENTER)
            self.info_label.pack(fill=tk.BOTH, expand=True)
        else:
            # No matplotlib available
            label = ttk.Label(frame, 
                text="📊 Visualization Panel\n\n"
                     "Neither tkinterweb nor matplotlib are available.\n\n"
                     "Install tkinterweb: pip install tkinterweb\n"
                     "or matplotlib: pip install matplotlib",
                font=('Arial', 12), justify=tk.CENTER)
            label.pack(fill=tk.BOTH, expand=True)
            self.info_label = label
    
    def display_plotly_figure(self, fig):
        """Display a Plotly figure using matplotlib fallback"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        try:
            # Clear existing widgets
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            
            # Create matplotlib figure
            mpl_fig = Figure(figsize=(12, 8))
            
            # Try to create a meaningful static visualization
            if hasattr(fig, 'data') and fig.data:
                # Create subplots based on the type of visualization
                if len(fig.data) > 0:
                    ax = mpl_fig.add_subplot(111)
                    
                    # Extract data from first trace
                    trace = fig.data[0]
                    
                    if hasattr(trace, 'x') and hasattr(trace, 'y') and trace.x is not None:
                        # 2D scatter plot
                        ax.scatter(trace.x, trace.y, alpha=0.7, s=50)
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_title('APGI Visualization (Static Version)')
                        ax.grid(True, alpha=0.3)
                    elif hasattr(trace, 'z') and hasattr(trace, 'x') and hasattr(trace, 'y'):
                        # 3D scatter plot - project to 2D
                        ax.scatter(trace.x, trace.y, c=trace.z, alpha=0.7, s=50, cmap='viridis')
                        ax.set_xlabel('X')
                        ax.set_ylabel('Y')
                        ax.set_title('APGI 3D Visualization (2D Projection)')
                        ax.grid(True, alpha=0.3)
                        plt.colorbar(mpl_fig.gca().collections[0], ax=ax, label='Z value')
                    else:
                        # Generic visualization info
                        ax.text(0.5, 0.5, 
                               f"📊 {fig.layout.title.text if hasattr(fig.layout, 'title') else 'APGI Visualization'}\n\n"
                               f"Interactive Plotly visualization\n\n"
                               f"Type: {type(trace).__name__}\n\n"
                               f"For full interactivity, install:\n"
                               f"pip install tkinterweb\n\n"
                               f"Then restart the application.",
                               ha='center', va='center', fontsize=12, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.axis('off')
                else:
                    # No data case
                    ax = mpl_fig.add_subplot(111)
                    ax.text(0.5, 0.5, 
                           "📊 APGI Visualization\n\n"
                           "No data available for this visualization type.\n\n"
                           "For full interactive visualizations, install:\n"
                           "pip install tkinterweb",
                           ha='center', va='center', fontsize=12, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
            else:
                # Fallback message
                ax = mpl_fig.add_subplot(111)
                ax.text(0.5, 0.5, 
                       "📊 APGI Visualization\n\n"
                       "Interactive Plotly visualization created.\n\n"
                       "For full interactivity, install tkinterweb:\n"
                       "pip install tkinterweb\n\n"
                       "Then restart the application.",
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            
            # Create canvas and toolbar
            self.matplotlib_canvas = FigureCanvasTkAgg(mpl_fig, self.canvas_frame)
            self.toolbar = NavigationToolbar2Tk(self.matplotlib_canvas, self.canvas_frame)
            
            # Pack toolbar and canvas
            self.toolbar.update()
            self.matplotlib_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Draw the figure
            self.matplotlib_canvas.draw()
            
        except Exception as e:
            print(f"Error creating matplotlib visualization: {e}")
            # Show error message
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            
            error_label = ttk.Label(self.canvas_frame, 
                text=f"❌ Error rendering visualization\n\n{str(e)}\n\n"
                     "For best results, install tkinterweb:\n"
                     "pip install tkinterweb",
                font=('Arial', 11), justify=tk.CENTER, foreground='red')
            error_label.pack(fill=tk.BOTH, expand=True)

    def _plotly_to_matplotlib(self, fig):
        """Convert a Plotly figure to matplotlib for fallback display"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        return self.display_plotly_figure(fig)
    
    def load_html_file(self, filepath):
        """Load and display an HTML file"""
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            return
        
        if self.display_method == "tkinterweb":
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                self.html_frame.load_html(html_content)
            except Exception as e:
                print(f"Error loading HTML: {e}")
        else:
            # Fallback: Show matplotlib version
            if hasattr(self, 'info_label') and self.info_label.winfo_exists():
                self.info_label.destroy()
            
            # Create a simple matplotlib display
            mpl_fig = self._plotly_to_matplotlib(None)
            if mpl_fig and self.matplotlib_canvas:
                self.matplotlib_canvas.draw()
    
    def clear(self):
        """Clear display"""
        if self.display_method == "tkinterweb":
            try:
                self.html_frame.load_html("<html><body><h2>Cleared</h2></body></html>")
            except (AttributeError, Exception):
                pass  # HTML frame not available or error loading
        else:
            # Clear matplotlib canvas
            if hasattr(self, 'matplotlib_canvas') and self.matplotlib_canvas:
                # Clear existing widgets
                for widget in self.canvas_frame.winfo_children():
                    widget.destroy()
                
                # Show initial message again
                self.info_label = ttk.Label(self.canvas_frame, 
                    text="📊 Visualization Panel\n\n"
                         "Visualizations will be displayed here using matplotlib.\n\n"
                         "Generate a visualization to see it in this panel.",
                    font=('Arial', 12), justify=tk.CENTER)
                self.info_label.pack(fill=tk.BOTH, expand=True)
                self.matplotlib_canvas = None
                self.toolbar = None


# =============================================================================
# STATE DEFINITIONS AND FACTORY FUNCTION
# =============================================================================

def create_apgi_params(
    Pi_e: float,
    Pi_i_baseline: float,
    M_ca: float,
    beta: float,
    z_e: float,
    z_i: float,
    theta_t: float
) -> APGIParameters:
    """Factory function that computes derived parameters automatically"""
    Pi_i_eff = Pi_i_baseline * np.exp(beta * M_ca)
    Pi_i_eff = np.clip(Pi_i_eff, 0.1, 10.0)
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
        z_i=z_i
    )


# =============================================================================
# STATE DEFINITIONS (51 PSYCHOLOGICAL STATES) - COMPREHENSIVE
# =============================================================================

# Category 1: Optimal Functioning States
STATE_01_FLOW = create_apgi_params(Pi_e=6.5, Pi_i_baseline=1.5, M_ca=0.3, beta=0.5, z_e=0.4, z_i=0.2, theta_t=1.8)
STATE_02_FOCUS = create_apgi_params(Pi_e=8.0, Pi_i_baseline=1.2, M_ca=0.25, beta=0.5, z_e=0.8, z_i=0.3, theta_t=-0.5)
STATE_03_SERENITY = create_apgi_params(Pi_e=1.5, Pi_i_baseline=2.0, M_ca=0.7, beta=0.5, z_e=0.2, z_i=0.3, theta_t=1.5)
STATE_04_MINDFULNESS = create_apgi_params(Pi_e=3.0, Pi_i_baseline=3.5, M_ca=0.9, beta=0.55, z_e=0.6, z_i=0.5, theta_t=0.0)

# Category 2: Positive Affective States
STATE_05_AMUSEMENT = create_apgi_params(Pi_e=4.0, Pi_i_baseline=1.0, M_ca=-0.1, beta=0.5, z_e=1.2, z_i=0.2, theta_t=-0.3)
STATE_06_JOY = create_apgi_params(Pi_e=5.0, Pi_i_baseline=2.5, M_ca=0.8, beta=0.55, z_e=1.0, z_i=0.7, theta_t=-0.8)
STATE_07_PRIDE = create_apgi_params(Pi_e=4.5, Pi_i_baseline=3.0, M_ca=1.1, beta=0.6, z_e=1.2, z_i=0.9, theta_t=-0.6)
STATE_08_ROMANTIC_LOVE_EARLY = create_apgi_params(Pi_e=7.5, Pi_i_baseline=4.0, M_ca=1.8, beta=0.7, z_e=1.5, z_i=1.3, theta_t=-1.5)
STATE_08B_ROMANTIC_LOVE_SUSTAINED = create_apgi_params(Pi_e=5.0, Pi_i_baseline=3.0, M_ca=1.2, beta=0.6, z_e=0.5, z_i=0.6, theta_t=-0.8)
STATE_09_GRATITUDE = create_apgi_params(Pi_e=4.0, Pi_i_baseline=2.5, M_ca=0.8, beta=0.55, z_e=0.3, z_i=0.5, theta_t=-0.4)
STATE_10_HOPE = create_apgi_params(Pi_e=5.0, Pi_i_baseline=2.0, M_ca=0.6, beta=0.5, z_e=0.9, z_i=0.4, theta_t=-0.7)
STATE_11_OPTIMISM = create_apgi_params(Pi_e=3.0, Pi_i_baseline=2.0, M_ca=0.4, beta=0.5, z_e=0.4, z_i=0.3, theta_t=-0.5)

# Category 3: Cognitive and Attentional States
STATE_12_CURIOSITY = create_apgi_params(Pi_e=6.0, Pi_i_baseline=1.0, M_ca=-0.2, beta=0.45, z_e=1.4, z_i=0.2, theta_t=-0.9)
STATE_13_BOREDOM = create_apgi_params(Pi_e=0.8, Pi_i_baseline=1.5, M_ca=-0.3, beta=0.5, z_e=0.1, z_i=0.2, theta_t=-1.0)
STATE_14_CREATIVITY = create_apgi_params(Pi_e=4.0, Pi_i_baseline=1.0, M_ca=-0.3, beta=0.45, z_e=1.2, z_i=0.2, theta_t=-1.2)
STATE_15_INSPIRATION = create_apgi_params(Pi_e=8.5, Pi_i_baseline=1.5, M_ca=0.4, beta=0.5, z_e=2.0, z_i=0.4, theta_t=-2.0)
STATE_16_HYPERFOCUS = create_apgi_params(Pi_e=9.5, Pi_i_baseline=0.5, M_ca=-0.8, beta=0.4, z_e=0.6, z_i=0.1, theta_t=2.5)
STATE_17_FATIGUE = create_apgi_params(Pi_e=1.5, Pi_i_baseline=2.0, M_ca=0.4, beta=0.5, z_e=0.3, z_i=0.4, theta_t=1.8)
STATE_18_DECISION_FATIGUE = create_apgi_params(Pi_e=2.5, Pi_i_baseline=1.5, M_ca=0.3, beta=0.5, z_e=0.8, z_i=0.3, theta_t=1.5)
STATE_19_MIND_WANDERING = create_apgi_params(Pi_e=0.8, Pi_i_baseline=3.5, M_ca=0.6, beta=0.55, z_e=0.2, z_i=0.9, theta_t=1.5)

# Category 4: Aversive Affective States
STATE_20_FEAR = create_apgi_params(Pi_e=8.0, Pi_i_baseline=3.0, M_ca=1.9, beta=0.75, z_e=2.5, z_i=2.0, theta_t=-2.5)
STATE_21_ANXIETY = create_apgi_params(Pi_e=6.5, Pi_i_baseline=3.5, M_ca=1.5, beta=0.65, z_e=1.5, z_i=1.3, theta_t=-1.5)
STATE_22_ANGER = create_apgi_params(Pi_e=7.5, Pi_i_baseline=3.0, M_ca=1.5, beta=0.65, z_e=2.0, z_i=1.4, theta_t=-1.2)
STATE_23_GUILT = create_apgi_params(Pi_e=5.0, Pi_i_baseline=2.5, M_ca=0.8, beta=0.55, z_e=1.3, z_i=0.9, theta_t=-0.8)
STATE_24_SHAME = create_apgi_params(Pi_e=7.0, Pi_i_baseline=3.0, M_ca=1.3, beta=0.6, z_e=1.8, z_i=1.2, theta_t=-1.5)
STATE_25_LONELINESS = create_apgi_params(Pi_e=5.5, Pi_i_baseline=2.5, M_ca=0.8, beta=0.55, z_e=1.4, z_i=0.9, theta_t=-1.0)
STATE_26_OVERWHELM = create_apgi_params(Pi_e=3.0, Pi_i_baseline=3.0, M_ca=1.2, beta=0.6, z_e=2.8, z_i=1.5, theta_t=0.0)

# Category 5: Pathological and Extreme States
STATE_27_DEPRESSION = create_apgi_params(Pi_e=2.0, Pi_i_baseline=1.5, M_ca=0.3, beta=0.5, z_e=0.4, z_i=0.8, theta_t=1.5)
STATE_28_LEARNED_HELPLESSNESS = create_apgi_params(Pi_e=1.5, Pi_i_baseline=2.0, M_ca=0.5, beta=0.5, z_e=0.2, z_i=0.4, theta_t=2.0)
STATE_29_PESSIMISTIC_DEPRESSION = create_apgi_params(Pi_e=2.5, Pi_i_baseline=2.0, M_ca=0.7, beta=0.55, z_e=0.3, z_i=0.6, theta_t=1.8)
STATE_30_PANIC = create_apgi_params(Pi_e=4.0, Pi_i_baseline=5.0, M_ca=2.0, beta=0.8, z_e=1.5, z_i=3.0, theta_t=-3.0)
STATE_31_DISSOCIATION = create_apgi_params(Pi_e=2.0, Pi_i_baseline=0.5, M_ca=-1.5, beta=0.35, z_e=0.8, z_i=0.1, theta_t=2.0)
STATE_32_DEPERSONALIZATION = create_apgi_params(Pi_e=3.0, Pi_i_baseline=0.8, M_ca=-1.2, beta=0.4, z_e=1.0, z_i=0.5, theta_t=1.5)
STATE_33_DEREALIZATION = create_apgi_params(Pi_e=1.5, Pi_i_baseline=1.5, M_ca=-0.8, beta=0.45, z_e=1.2, z_i=0.4, theta_t=1.8)

# Category 6: Altered and Boundary States
STATE_34_AWE = create_apgi_params(Pi_e=3.5, Pi_i_baseline=2.5, M_ca=0.8, beta=0.55, z_e=2.8, z_i=0.7, theta_t=-1.5)
STATE_35_TRANCE = create_apgi_params(Pi_e=1.0, Pi_i_baseline=4.0, M_ca=0.4, beta=0.5, z_e=0.2, z_i=0.6, theta_t=2.0)
STATE_36_MEDITATION_FOCUSED = create_apgi_params(Pi_e=7.0, Pi_i_baseline=3.5, M_ca=1.0, beta=0.55, z_e=0.5, z_i=0.6, theta_t=1.5)
STATE_36B_MEDITATION_OPEN = create_apgi_params(Pi_e=3.0, Pi_i_baseline=3.0, M_ca=0.7, beta=0.5, z_e=0.8, z_i=0.6, theta_t=0.0)
STATE_36C_MEDITATION_NONDUAL = create_apgi_params(Pi_e=2.0, Pi_i_baseline=1.5, M_ca=0.5, beta=0.5, z_e=0.2, z_i=0.2, theta_t=2.0)
STATE_37_HYPNOSIS = create_apgi_params(Pi_e=2.0, Pi_i_baseline=3.5, M_ca=0.6, beta=0.55, z_e=0.3, z_i=0.8, theta_t=-1.5)
STATE_38_HYPNAGOGIA = create_apgi_params(Pi_e=2.5, Pi_i_baseline=4.0, M_ca=0.7, beta=0.55, z_e=0.6, z_i=1.0, theta_t=0.5)
STATE_39_DEJA_VU = create_apgi_params(Pi_e=4.5, Pi_i_baseline=1.5, M_ca=0.2, beta=0.5, z_e=0.4, z_i=0.2, theta_t=-0.8)

# Category 7: Transitional/Contextual States
STATE_40_MORNING_FLOW = create_apgi_params(Pi_e=5.5, Pi_i_baseline=2.0, M_ca=0.5, beta=0.5, z_e=0.3, z_i=0.3, theta_t=1.2)
STATE_41_EVENING_FATIGUE = create_apgi_params(Pi_e=1.2, Pi_i_baseline=3.0, M_ca=1.0, beta=0.55, z_e=0.2, z_i=0.7, theta_t=2.2)
STATE_42_CREATIVE_INSPIRATION = create_apgi_params(Pi_e=8.0, Pi_i_baseline=1.5, M_ca=0.3, beta=0.5, z_e=2.2, z_i=0.3, theta_t=-1.8)
STATE_43_ANXIOUS_RUMINATION = create_apgi_params(Pi_e=6.0, Pi_i_baseline=3.5, M_ca=1.4, beta=0.65, z_e=1.6, z_i=1.2, theta_t=-1.2)
STATE_44_CALM = create_apgi_params(Pi_e=1.8, Pi_i_baseline=2.0, M_ca=0.5, beta=0.5, z_e=0.2, z_i=0.3, theta_t=1.2)
STATE_45_PRODUCTIVE_FOCUS = create_apgi_params(Pi_e=7.0, Pi_i_baseline=1.5, M_ca=0.3, beta=0.5, z_e=0.7, z_i=0.3, theta_t=-0.3)
STATE_46_SECOND_WIND = create_apgi_params(Pi_e=5.5, Pi_i_baseline=2.5, M_ca=0.5, beta=0.55, z_e=0.9, z_i=0.5, theta_t=-0.8)

# Category 8: Previously Unelaborated States
STATE_47_HYPERVIGILANCE = create_apgi_params(Pi_e=8.5, Pi_i_baseline=4.0, M_ca=1.7, beta=0.7, z_e=1.8, z_i=1.5, theta_t=-2.0)
STATE_48_SADNESS = create_apgi_params(Pi_e=4.5, Pi_i_baseline=2.5, M_ca=0.9, beta=0.55, z_e=1.2, z_i=0.8, theta_t=-0.6)
STATE_49_CHOICE_PARALYSIS = create_apgi_params(Pi_e=2.5, Pi_i_baseline=2.0, M_ca=0.5, beta=0.5, z_e=0.9, z_i=0.5, theta_t=1.5)
STATE_50_MENTAL_PARALYSIS = create_apgi_params(Pi_e=2.0, Pi_i_baseline=3.5, M_ca=1.3, beta=0.65, z_e=3.0, z_i=1.8, theta_t=0.5)
STATE_51_CURIOUS_EXPLORATION = create_apgi_params(Pi_e=6.5, Pi_i_baseline=1.0, M_ca=-0.1, beta=0.45, z_e=1.6, z_i=0.2, theta_t=-1.0)

# Master dictionary
PSYCHOLOGICAL_STATES: Dict[str, APGIParameters] = {
    'flow': STATE_01_FLOW,
    'focus': STATE_02_FOCUS,
    'serenity': STATE_03_SERENITY,
    'mindfulness': STATE_04_MINDFULNESS,
    'amusement': STATE_05_AMUSEMENT,
    'joy': STATE_06_JOY,
    'pride': STATE_07_PRIDE,
    'romantic_love_early': STATE_08_ROMANTIC_LOVE_EARLY,
    'romantic_love_sustained': STATE_08B_ROMANTIC_LOVE_SUSTAINED,
    'gratitude': STATE_09_GRATITUDE,
    'hope': STATE_10_HOPE,
    'optimism': STATE_11_OPTIMISM,
    'curiosity': STATE_12_CURIOSITY,
    'boredom': STATE_13_BOREDOM,
    'creativity': STATE_14_CREATIVITY,
    'inspiration': STATE_15_INSPIRATION,
    'hyperfocus': STATE_16_HYPERFOCUS,
    'fatigue': STATE_17_FATIGUE,
    'decision_fatigue': STATE_18_DECISION_FATIGUE,
    'mind_wandering': STATE_19_MIND_WANDERING,
    'fear': STATE_20_FEAR,
    'anxiety': STATE_21_ANXIETY,
    'anger': STATE_22_ANGER,
    'guilt': STATE_23_GUILT,
    'shame': STATE_24_SHAME,
    'loneliness': STATE_25_LONELINESS,
    'overwhelm': STATE_26_OVERWHELM,
    'depression': STATE_27_DEPRESSION,
    'learned_helplessness': STATE_28_LEARNED_HELPLESSNESS,
    'pessimistic_depression': STATE_29_PESSIMISTIC_DEPRESSION,
    'panic': STATE_30_PANIC,
    'dissociation': STATE_31_DISSOCIATION,
    'depersonalization': STATE_32_DEPERSONALIZATION,
    'derealization': STATE_33_DEREALIZATION,
    'awe': STATE_34_AWE,
    'trance': STATE_35_TRANCE,
    'meditation_focused': STATE_36_MEDITATION_FOCUSED,
    'meditation_open': STATE_36B_MEDITATION_OPEN,
    'meditation_nondual': STATE_36C_MEDITATION_NONDUAL,
    'hypnosis': STATE_37_HYPNOSIS,
    'hypnagogia': STATE_38_HYPNAGOGIA,
    'deja_vu': STATE_39_DEJA_VU,
    'morning_flow': STATE_40_MORNING_FLOW,
    'evening_fatigue': STATE_41_EVENING_FATIGUE,
    'creative_inspiration': STATE_42_CREATIVE_INSPIRATION,
    'anxious_rumination': STATE_43_ANXIOUS_RUMINATION,
    'calm': STATE_44_CALM,
    'productive_focus': STATE_45_PRODUCTIVE_FOCUS,
    'second_wind': STATE_46_SECOND_WIND,
    'hypervigilance': STATE_47_HYPERVIGILANCE,
    'sadness': STATE_48_SADNESS,
    'choice_paralysis': STATE_49_CHOICE_PARALYSIS,
    'mental_paralysis': STATE_50_MENTAL_PARALYSIS,
    'curious_exploration': STATE_51_CURIOUS_EXPLORATION,
}

# State category mapping
STATE_CATEGORIES: Dict[str, StateCategory] = {
    'flow': StateCategory.OPTIMAL_FUNCTIONING,
    'focus': StateCategory.OPTIMAL_FUNCTIONING,
    'serenity': StateCategory.OPTIMAL_FUNCTIONING,
    'mindfulness': StateCategory.OPTIMAL_FUNCTIONING,
    'amusement': StateCategory.POSITIVE_AFFECTIVE,
    'joy': StateCategory.POSITIVE_AFFECTIVE,
    'pride': StateCategory.POSITIVE_AFFECTIVE,
    'romantic_love_early': StateCategory.POSITIVE_AFFECTIVE,
    'romantic_love_sustained': StateCategory.POSITIVE_AFFECTIVE,
    'gratitude': StateCategory.POSITIVE_AFFECTIVE,
    'hope': StateCategory.POSITIVE_AFFECTIVE,
    'optimism': StateCategory.POSITIVE_AFFECTIVE,
    'curiosity': StateCategory.COGNITIVE_ATTENTIONAL,
    'boredom': StateCategory.COGNITIVE_ATTENTIONAL,
    'creativity': StateCategory.COGNITIVE_ATTENTIONAL,
    'inspiration': StateCategory.COGNITIVE_ATTENTIONAL,
    'hyperfocus': StateCategory.COGNITIVE_ATTENTIONAL,
    'fatigue': StateCategory.COGNITIVE_ATTENTIONAL,
    'decision_fatigue': StateCategory.COGNITIVE_ATTENTIONAL,
    'mind_wandering': StateCategory.COGNITIVE_ATTENTIONAL,
    'fear': StateCategory.AVERSIVE_AFFECTIVE,
    'anxiety': StateCategory.AVERSIVE_AFFECTIVE,
    'anger': StateCategory.AVERSIVE_AFFECTIVE,
    'guilt': StateCategory.AVERSIVE_AFFECTIVE,
    'shame': StateCategory.AVERSIVE_AFFECTIVE,
    'loneliness': StateCategory.AVERSIVE_AFFECTIVE,
    'overwhelm': StateCategory.AVERSIVE_AFFECTIVE,
    'depression': StateCategory.PATHOLOGICAL_EXTREME,
    'learned_helplessness': StateCategory.PATHOLOGICAL_EXTREME,
    'pessimistic_depression': StateCategory.PATHOLOGICAL_EXTREME,
    'panic': StateCategory.PATHOLOGICAL_EXTREME,
    'dissociation': StateCategory.PATHOLOGICAL_EXTREME,
    'depersonalization': StateCategory.PATHOLOGICAL_EXTREME,
    'derealization': StateCategory.PATHOLOGICAL_EXTREME,
    'awe': StateCategory.ALTERED_BOUNDARY,
    'trance': StateCategory.ALTERED_BOUNDARY,
    'meditation_focused': StateCategory.ALTERED_BOUNDARY,
    'meditation_open': StateCategory.ALTERED_BOUNDARY,
    'meditation_nondual': StateCategory.ALTERED_BOUNDARY,
    'hypnosis': StateCategory.ALTERED_BOUNDARY,
    'hypnagogia': StateCategory.ALTERED_BOUNDARY,
    'deja_vu': StateCategory.ALTERED_BOUNDARY,
    'morning_flow': StateCategory.TRANSITIONAL_CONTEXTUAL,
    'evening_fatigue': StateCategory.TRANSITIONAL_CONTEXTUAL,
    'creative_inspiration': StateCategory.TRANSITIONAL_CONTEXTUAL,
    'anxious_rumination': StateCategory.TRANSITIONAL_CONTEXTUAL,
    'calm': StateCategory.TRANSITIONAL_CONTEXTUAL,
    'productive_focus': StateCategory.TRANSITIONAL_CONTEXTUAL,
    'second_wind': StateCategory.TRANSITIONAL_CONTEXTUAL,
    'hypervigilance': StateCategory.UNELABORATED,
    'sadness': StateCategory.UNELABORATED,
    'choice_paralysis': StateCategory.UNELABORATED,
    'mental_paralysis': StateCategory.UNELABORATED,
    'curious_exploration': StateCategory.UNELABORATED,
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_state(name: str) -> APGIParameters:
    """Retrieve parameters for a named psychological state"""
    if name not in PSYCHOLOGICAL_STATES:
        raise KeyError(f"Unknown state: {name}")
    return PSYCHOLOGICAL_STATES[name]


def get_states_by_category(category: StateCategory) -> Dict[str, APGIParameters]:
    """Retrieve all states belonging to a category"""
    return {
        name: params 
        for name, params in PSYCHOLOGICAL_STATES.items()
        if STATE_CATEGORIES.get(name) == category
    }


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main entry point for the APGI Psychological States Visualization System"""
    print("\n🧠 APGI Psychological States Visualization System")
    print("=" * 70)
    
    # Check dependencies
    required = {
        "Tkinter": TKINTER_AVAILABLE,
        "Plotly": PLOTLY_AVAILABLE,
        "Pandas": PANDAS_AVAILABLE,
    }
    
    missing = [name for name, available in required.items() if not available]
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("   pip install plotly pandas")
        return
    
    print("✅ All dependencies available")
    print(f"   • Tkinter: {TKINTER_AVAILABLE}")
    print(f"   • Plotly: {PLOTLY_AVAILABLE}")
    print(f"   • Pandas: {PANDAS_AVAILABLE}")
    print(f"   • TkinterWeb: {TKINTERWEB_AVAILABLE}")
    
    print(f"\n📊 Loaded {len(PSYCHOLOGICAL_STATES)} psychological states across {len(set(STATE_CATEGORIES.values()))} categories")
    
    try:
        print("\n🚀 Launching interactive GUI...")
        gui = APGIVisualizerGUI()
        gui.run()
    except Exception as e:
        print(f"\n❌ Error launching GUI: {e}")
        print(format_exc())


if __name__ == "__main__":
    main()