#!/usr/bin/env python
"""
Loops through the random files for each .npz 
A metric is calculated for each sparsity value and task similarity
"""

from __future__ import annotations
import sys, re, os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipdb import set_trace
import pickle
import jax
import matplotlib.pyplot as plt

from loss_plotter import load_grouped_metric_data, MetricSeries, calc_metric, _log

# Define publication-friendly color scales
COLOR_SCALES = {
    'rem': 'Viridis',
    'ft_t1': 'Plasma',
    'ft_t2': 'Inferno',
    'zs': 'Cividis'}

METRIC_TITLES = {
    'rem': 'Remembering',
    'ft_t1': 'Forward Transfer Task1',
    'ft_t2': 'Forward Transfer Task2',
    'zs': 'Zero-Shot'
}

LossDict = Dict[str, Any]
FILE_REGEX = re.compile(r"g_type_([^_]+)_v_(\d+\.\d+)_spars_(\d+\.\d+)_overlap_(\d+\.\d+)\.npz") # f"g_type_{g_type}_v_{v:.2f}_spars_{sparsity:.2f}.npz"


def load_overlap_metric_data(data_dir: Path, verbose: bool=True) -> Tuple[Dict, Dict]:
    """Load and validate data from both experiment directories (random/determ)."""
    g_type = 'overlap'

    grouped = {g_type:[]}
    meta_ref = None
    dir_path = data_dir
    for npz_path in dir_path.glob("*.npz"):
        m = FILE_REGEX.match(npz_path.name)
        if not m:
            continue

        try:
            with np.load(npz_path) as dat:
                if not all(k in dat for k in ["test_loss1", "test_loss2", "epochs", "switch_point", "sparsity", "overlap_output", "overlap"]):
                    continue
                # set_trace()
                meta = {k: dat[k].item() if dat[k].shape == () else dat[k] 
                        for k in ["switch_point", "epochs", "g_type", "sparsity"] if k in dat}
                if meta_ref is None:
                    meta_ref = meta
                else:
                    for k, v in meta_ref.items():
                        if k != 'g_type' and not np.array_equal(meta.get(k, None), v):
                            raise ValueError(f"Meta mismatch in {npz_path.name}: {k}")

                grouped[meta['g_type']].append({
                    "v": float(m.group(2)),
                    "g_type": meta['g_type'],
                    "epochs": dat["epochs"],
                    "loss1_raw": np.clip(dat["test_loss1"], 1e-15, 1e9),
                    "loss2_raw": np.clip(dat["test_loss2"], 1e-15, 1e9),
                    "switch_point": int(meta["switch_point"]),
                    "num_runs": dat["test_loss1"].shape[1],
                    "overlap_output": dat["overlap_output"],
                    "overlap_var": dat["overlap"]
                })
        except Exception as e:
            _log(f"Error reading {npz_path.name}: {e}", verbose)
    
    return grouped, meta_ref


def get_metrics(data_dir: Path, calc_method: str = 'auc', verbose: bool = True, metric_names: list = ["rem"]) -> Tuple[Dict, Dict, Dict]:
    """Calculate metrics for all experiments in a directory."""
    min_runs_for_std = 2
    grouped, meta_ref = load_overlap_metric_data(data_dir, verbose)
    series: Dict[str, Dict[str, list]] = {
        'overlap': {m: {'v': [], 'mu': [], 'sig': [], 'overlap': [], "overlap_var":[]} for m in metric_names}
    }
    # Define metric mapping
    METRIC_MAP = {
        'rem': 0,
        'ft_t1': 1,
        'ft_t2': 2,
        'zs': 3
    }
    
    for g_type, exps in grouped.items():
        for exp in sorted(exps, key=lambda e: e["v"]):
            switch_idx = int(np.argmin(np.abs(exp["epochs"] - exp["switch_point"])))
            
            # Calculate average overlap for this experiment
            overlap_avg = np.mean(exp["overlap_output"])
            
            per_run = [
                calc_metric(
                    exp["loss1_raw"], exp["loss2_raw"],
                    r, exp["v"], switch_idx, calc_method, meta_ref['epochs']
                )
                for r in range(exp["loss1_raw"].shape[1])
            ]
            
            if len(per_run) < min_runs_for_std:
                continue

            all_metrics = list(map(np.asarray, zip(*per_run)))
            v_val = float(exp["v"])

            # Store metrics with overlap
            for metric in metric_names:
                values = all_metrics[METRIC_MAP[metric]]
                series[g_type][metric]['v'].append(v_val)
                series[g_type][metric]['mu'].append(values.mean())
                series[g_type][metric]['sig'].append(values.std() / np.sqrt(len(values)))
                series[g_type][metric]['overlap'].append(overlap_avg)
                series[g_type][metric]['overlap_var'].append(exp['overlap_var'])
    return series, grouped, meta_ref


# ==================== Plotting Functions (REVAMPED) ====================
def create_scatter_plots(metrics_data: Dict[str, Dict[str, np.ndarray]], calc_method: str, metric_names) -> List[go.Figure]:
    """Create individual scatter plots for each metric with color bars."""
    figures = []
    for metric in metric_names:
        title = f"{METRIC_TITLES[metric]} ({calc_method})"
        fig = go.Figure()
        
        scatter = go.Scatter(
            x=metrics_data[metric]['v'],
            y=metrics_data[metric]['overlap'],
            mode='markers',
            marker=dict(
                size=30,
                symbol='square',
                color=metrics_data[metric]['mu'],
                colorscale=COLOR_SCALES[metric],
                showscale=True,
                colorbar=dict(title='Metric Value', len=0.75),
                opacity=0.8
            ),
            name=title,
            text=[f"v: {v:.2f}<br>Overlap: {o:.4f}<br>Value: {val:.4f}" 
                  for v, o, val in zip(metrics_data[metric]['v'], 
                                       metrics_data[metric]['overlap'], 
                                       metrics_data[metric]['mu'])]
        )
        
        fig.add_trace(scatter)
        fig.update_layout(
            title=title,
            xaxis_title="Task Similarity (v)",
            yaxis_title="Overlap",
            template="plotly_white",
            font=dict(family="Arial", size=14),
            height=600,
            width=800
        )
        figures.append(fig)
    
    return figures

def create_3d_plots(processed_3d: Dict[str, Dict[str, np.ndarray]], calc_method: str, metric_names) -> List[go.Figure]:
    """Create individual 3D surface plots for each metric with color bars."""
    figures = []
    for metric in metric_names:
        title = f"{METRIC_TITLES[metric]} ({calc_method})"
        fig = go.Figure()
        
        surface = go.Surface(
            x=processed_3d[metric]['v_grid'],
            y=processed_3d[metric]['overlap_grid'],
            z=processed_3d[metric]['mu_grid'],
            colorscale=COLOR_SCALES[metric],
            showscale=True,
            name=title,
            opacity=0.9,
            contours=dict(
                x=dict(show=True, color='gray', width=1),
                y=dict(show=True, color='gray', width=1),
                z=dict(show=True, width=1)
            ),
            hoverinfo="x+y+z+name",
            hovertemplate=(
                "Task Similarity: %{x:.2f}<br>"
                "Overlap: %{y:.4f}<br>"
                "Metric Value: %{z:.4f}<br>"
            )
        )
        
        fig.add_trace(surface)
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Task Similarity (v)",
                yaxis_title="Overlap",
                zaxis_title="Metric Value"
            ),
            template="plotly_white",
            font=dict(family="Arial", size=14),
            height=800,
            width=1200
        )
        figures.append(fig)
    
    return figures

def create_max_value_plots(max_values: Dict[str, np.ndarray], 
                           max_overlaps: Dict[str, np.ndarray], 
                           max_sigs: Dict[str, np.ndarray], 
                           v_arr: np.ndarray, 
                           calc_method: str,
                           metric_names: list) -> List[go.Figure]:
    """Create plots for maximum metric values vs task similarity with overlap in hover."""
    figures = []
    for metric in metric_names:
        title = f"{METRIC_TITLES[metric]} ({calc_method})"
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=v_arr,
                y=max_values[metric],
                mode='markers+lines',
                marker=dict(size=10, color='#2ca02c'),
                line=dict(width=3, color='#2ca02c'),
                name=title,
                # UPDATED hover text with overlap
                text=[f"v: {v:.2f}<br>Max Value: {val:.4f}<br>Overlap: {ov:.4f}" 
                      for v, val, ov in zip(v_arr, max_values[metric], max_overlaps[metric])]
            )
        )
        # Add shaded region (error bar)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([v_arr, v_arr[::-1]]),  # X, then reversed X
                y=np.concatenate([
                    np.array(max_values[metric]) + np.array(max_sigs[metric]),  # Upper bound
                    (np.array(max_values[metric]) - np.array(max_sigs[metric]))[::-1]  # Lower bound (reversed)
                ]),
                fill='toself',
                fillcolor='rgba(44, 160, 44, 0.2)',  # Semi-transparent green
                line=dict(color='rgba(255,255,255,0)'),  # Invisible line
                hoverinfo="skip",
                showlegend=False
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Task Similarity (v)",
            yaxis_title="Max Metric Value",
            template="plotly_white",
            font=dict(family="Arial", size=14),
            height=500,
            width=800
        )
        figures.append(fig)
    
    return figures


# ==================== Data Processing ====================
def prepare_3d_data(metrics_data: Dict[str, Dict[str, list]], metric_names: list) -> Dict[str, Dict[str, np.ndarray]]:
    """Prepare gridded data for 3D surface plots."""
    processed = {}
    
    for metric in metric_names:
        # Extract raw data
        v = np.array(metrics_data[metric]['v'])
        overlap = np.array(metrics_data[metric]['overlap'])
        mu = np.array(metrics_data[metric]['mu'])
        
        # Create grid for surface plot
        v_min, v_max = v.min(), v.max()
        o_min, o_max = overlap.min(), overlap.max()
        
        v_grid = np.linspace(v_min, v_max, 50)
        o_grid = np.linspace(o_min, o_max, 50)
        v_mesh, o_mesh = np.meshgrid(v_grid, o_grid)
        
        # Interpolate metric values onto the grid
        from scipy.interpolate import griddata
        mu_grid = griddata((v, overlap), mu, (v_mesh, o_mesh), method='cubic')
        
        processed[metric] = {
            'v_grid': v_mesh,
            'overlap_grid': o_mesh,
            'mu_grid': mu_grid,
            'v': v,
            'overlap': overlap,
            'mu': mu
        }
    
    return processed

# Step 2: Update calculate_max_values to return max_sigs
def calculate_max_values(metrics_data: Dict[str, Dict[str, list]], metric_names: list) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    max_values = {metric: [] for metric in metric_names}
    max_overlaps = {metric: [] for metric in metric_names}
    max_sigs = {metric: [] for metric in metric_names}  # NEW: store sig at max
    # Get unique v values from all metrics
    all_v = []
    for metric in metric_names:
        all_v.extend(metrics_data[metric]['v'])
    unique_v = sorted(set(all_v))
    
    for v_val in unique_v:
        for metric in metric_names:
            indices = [i for i, v in enumerate(metrics_data[metric]['v']) if v == v_val]
            if not indices:
                max_values[metric].append(np.nan)
                max_overlaps[metric].append(np.nan)
                max_sigs[metric].append(np.nan)  # NEW
                continue
                
            values = [metrics_data[metric]['mu'][i] for i in indices]
            overlaps = [metrics_data[metric]['overlap'][i] for i in indices]
            sigs = [metrics_data[metric]['sig'][i] for i in indices]  # NEW: get sigs
            
            max_idx = np.argmax(values)
            max_values[metric].append(values[max_idx])
            max_overlaps[metric].append(overlaps[max_idx])
            max_sigs[metric].append(sigs[max_idx])  # NEW: store corresponding sig
    
    return max_values, max_overlaps, max_sigs, np.array(unique_v)  # UPDATED return

def create_overlap_traces_plot(metrics_data: Dict[str, Dict[str, list]], calc_method: str, metric_names: list) -> List[go.Figure]:
    """
    Creates plots with traces for each overlap_var value using a continuous color scale.
    Y-axis: metric value (mu)
    X-axis: task similarity (v)
    Color: overlap_var value (continuous color bar)
    Hover: shows overlap_output value
    Includes error bars for standard deviation (sig)
    """
    import plotly.colors
    figures = []
    for metric in metric_names:
        title = f"{METRIC_TITLES[metric]} ({calc_method})"
        fig = go.Figure()
        # Extract data for this metric
        v_arr = np.array(metrics_data[metric]['v'])
        mu_arr = np.array(metrics_data[metric]['mu'])
        sig_arr = np.array(metrics_data[metric]['sig'])  # Standard deviation
        overlap_output_arr = np.array(metrics_data[metric]['overlap'])  # noisy overlap_output
        overlap_var_arr = np.array(metrics_data[metric]['overlap_var'])  # declared overlap_var
        
        # Get unique overlap_var values and sort them
        unique_overlaps = sorted(set(overlap_var_arr))
        
        # Get min/max for color scaling
        min_ov = min(unique_overlaps)
        max_ov = max(unique_overlaps)
        
        # Choose a color scale (Viridis works well for continuous values)
        color_scale = COLOR_SCALES.get(metric, 'Viridis')
        
        # Create a trace for each unique overlap_var
        for ov in unique_overlaps:
            # Get indices for this specific overlap_var
            indices = np.where(np.abs(overlap_var_arr - ov) < 1e-5)[0]
            
            # Sort by v for clean line connections
            sorted_indices = indices[np.argsort(v_arr[indices])]
            
            # Get sorted data for this trace
            trace_v = v_arr[sorted_indices]
            trace_mu = mu_arr[sorted_indices]
            trace_sig = sig_arr[sorted_indices]
            trace_output = overlap_output_arr[sorted_indices]
            
            # Normalize overlap for color mapping
            norm_ov = (ov - min_ov) / (max_ov - min_ov)
            line_color = plotly.colors.sample_colorscale(color_scale, norm_ov)[0]
            
            # Create hover text with actual overlap_output
            hover_text = [
                f"Task Similarity: {v:.2f}<br>"
                f"Metric Value: {mu:.4f} ± {sig:.4f}<br>"
                f"Declared Overlap: {ov:.2f}<br>"
                f"Actual Overlap: {actual:.4f}"
                for v, mu, sig, actual in zip(trace_v, trace_mu, trace_sig, trace_output)
            ]
            
            # Add main trace with error bars
            fig.add_trace(go.Scatter(
                x=trace_v,
                y=trace_mu,
                mode='lines+markers',
                name=f'Overlap={ov:.2f}',
                line=dict(color=line_color, width=3),
                marker=dict(color=line_color, size=8),
                text=hover_text,
                hoverinfo='text',
                showlegend=False,  # We'll use the color bar instead
                error_y=dict(
                    type='data',
                    array=trace_sig,
                    visible=True,
                    thickness=1.5,
                    width=4
                )
            ))
        
        # Add a dummy trace for the color bar
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                colorscale=color_scale,
                cmin=min_ov,
                cmax=max_ov,
                colorbar=dict(
                    title='Declared Overlap',
                    thickness=20,
                    len=0.6,
                    tickvals=np.linspace(min_ov, max_ov, 6),
                    tickformat='.2f'
                ),
                showscale=True
            ),
            hoverinfo='none',
            showlegend=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Task Similarity (v)",
            yaxis_title="Metric Value",
            template="plotly_white",
            font=dict(family="Arial", size=14),
            height=600,
            width=900
        )
        figures.append(fig)
    
    return figures


# ==================== HTML Saving Helpers ====================
def save_figures_to_html(figures: List[go.Figure], filename: Path):
    """Save multiple figures to a single HTML file."""
    with open(filename, 'w') as f:
        f.write('<html><head><script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body>')
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs=False))
        f.write('</body></html>')


# ==================== Main Processing (REVAMPED) ====================
def main():
    # Configuration
    data_path = "/home/users/MTrappett/mtrl/RepresentationSimilarityCL/loss_data/overlap_search/under_capacity_twolayer"
    plots_path = Path(data_path + "_plots")
    plots_path.mkdir(parents=True, exist_ok=True)
    
    # Get all sparsity directories
    sparsity_dirs = [d for d in Path(data_path).iterdir() if d.is_dir() and 'sparsity' in d.name]
    sparsity_dirs.sort(key=lambda x: float(x.name.split('_')[7]))  # Sort by sparsity value
    
    # Initialize figure aggregators
    all_scatter_figs = []
    all_3d_figs = []
    all_max_figs = []
    all_overlap_trace_figs = []
    
    # Define calculation methods
    calc_methods = ['auc', 'mean', 'median', 'min']
    
    # Process each calculation method
    for calc_method in calc_methods:
        # Initialize data structures for this method
        metric_names = ['rem', 'ft_t2', 'zs']
        metrics_data = {m: {'v': [], 'overlap': [], 'mu': [], 'sig': [], "overlap_var": []} for m in metric_names}
        # Process each sparsity directory
        for sparsity_dir in sparsity_dirs:
            try:
                series, grouped, meta_ref = get_metrics(sparsity_dir, calc_method, True, metric_names)

                # Collect data across all directories
                for metric in metric_names:
                    metrics_data[metric]['v'].extend(series['overlap'][metric]['v'])
                    metrics_data[metric]['overlap'].extend(series['overlap'][metric]['overlap'])
                    metrics_data[metric]['overlap_var'].extend(series['overlap'][metric]['overlap_var'])
                    metrics_data[metric]['mu'].extend(series['overlap'][metric]['mu'])
                    metrics_data[metric]['sig'].extend(series['overlap'][metric]['sig'])

            except Exception as e:
                print(f"Error processing {sparsity_dir.name}: {str(e)}")
                continue

        # Convert to arrays
        for metric in metric_names:
            for key in ['v', 'overlap', 'mu', 'overlap_var']:
                metrics_data[metric][key] = np.array(metrics_data[metric][key])

        # Prepare 3D data for surface plots
        processed_3d = prepare_3d_data(metrics_data, metric_names)

        # Calculate maximum values per task similarity
        max_values, max_overlaps, max_sigs, unique_v = calculate_max_values(metrics_data, metric_names)  # UPDATED

        # Create all plots for this method
        scatter_figs = create_scatter_plots(metrics_data, calc_method, metric_names)
        three_d_figs = create_3d_plots(processed_3d, calc_method, metric_names)
        max_figs = create_max_value_plots(max_values, max_overlaps, max_sigs, unique_v, calc_method, metric_names)  # UPDATED
        overlap_trace_figs = create_overlap_traces_plot(metrics_data, calc_method, metric_names)

        # Aggregate figures
        all_scatter_figs.extend(scatter_figs)
        all_3d_figs.extend(three_d_figs)
        all_max_figs.extend(max_figs)
        all_overlap_trace_figs.extend(overlap_trace_figs)

       # Replace the entire np.savez block with:
        save_data = {'unique_v': unique_v}

        # Add v from the first metric (should be same for all)
        if metric_names:
            save_data['v'] = metrics_data[metric_names[0]]['v']

        for metric in metric_names:
            save_data.update({
                f'overlap_{metric}': metrics_data[metric]['overlap'],
                metric: metrics_data[metric]['mu'],
                f'max_{metric}': max_values[metric],
                f'max_overlap_{metric}': max_overlaps[metric],
                f'max_sig_{metric}': max_sigs[metric],
            })

        np.savez(plots_path / f"overlap_metrics_{calc_method}.npz", **save_data)

    
    # Save all figures to unified HTML files
    save_figures_to_html(all_scatter_figs, plots_path / "scatter_plots.html")
    save_figures_to_html(all_3d_figs, plots_path / "3d_plots.html")
    save_figures_to_html(all_max_figs, plots_path / "max_value_plots.html")
    save_figures_to_html(all_overlap_trace_figs, plots_path / "overlap_traces_plots.html")

    print(f"Processing complete. Results saved to: {plots_path}")

if __name__=='__main__':
    main()