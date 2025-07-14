#!/usr/bin/env python
"""
Plot max metric values from multiple .npz files with explicit folder paths and labels
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ==================== CONFIGURATION - MODIFY THESE VALUES ====================
# Define your folders and labels here (folder path : display label)
FOLDERS = {
    "/home/users/MTrappett/mtrl/RepresentationSimilarityCL/loss_data/overlap_search/negative_capacity_plots": "Under Capacity",
    "/home/users/MTrappett/mtrl/RepresentationSimilarityCL/loss_data/overlap_search_plots": "Equal Capacity",
    "/home/users/MTrappett/mtrl/RepresentationSimilarityCL/loss_data/overlap_search/over_capacity_plots": "Over Capacity",
    "/home/users/MTrappett/mtrl/RepresentationSimilarityCL/loss_data/overlap_search/under_capacity_twolayer_plots": "Two Layers",
    "/home/users/MTrappett/mtrl/RepresentationSimilarityCL/loss_data/single_layer_ngg_equalCap_plots": "Noise Gated-Grad"
    # Add more folders as needed:
    # "/path/to/another/folder": "Another Experiment",
}

# Output directory for plots
OUTPUT_DIR = "./loss_data/max_overlap_metric_plots_ngg"

# Plot configuration
METRIC_TITLES = {
    'rem': 'Remembering',
    'ft_t2': 'Forward Transfer Task2',
    'zs': 'Zero-Shot'
}
METRIC_COLORS = {
    'Under Capacity': 'rgb(31,118,180)',     # Blue
    'Equal Capacity': 'rgb(255,127,14)',              # Orange
    'Over Capacity' : 'rgb(190,174,212)',
    'Two Layers'    : 'rgb(100, 100, 100)',
    "Noise Gated-Grad": 'rgb(75, 125, 225)',
    # Add more colors as needed:
    # 'Another Experiment': '#2ca02c',  # Green
}
SUB_METRICS = ['rem', 'ft_t2', 'zs']   # Metrics to show in subplots
METRIC_FILES = {
    'AUC': 'overlap_metrics_auc.npz',
    'mean': 'overlap_metrics_mean.npz',
    'median': 'overlap_metrics_median.npz',
    'min': 'overlap_metrics_min.npz'
}
# =============================================================================

def load_npz_data(folder_path: Path, metric_key: str) -> dict:
    """Load NPZ data from specified folder for a specific metric"""
    npz_path = folder_path / METRIC_FILES[metric_key]
    if not npz_path.exists():
        print(f"⚠️ File not found: {npz_path}. Skipping...")
        return None
    return dict(np.load(npz_path))

def create_metric_plot(metric_data: dict, metric_name: str) -> go.Figure:
    """Create consolidated plot for a specific metric"""
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=[METRIC_TITLES[m] for m in SUB_METRICS],
                        horizontal_spacing=0.08)
    
    # Track which labels we've already added to legend
    legend_added = {label: False for label in metric_data.keys()}
    
    for idx, sub_metric in enumerate(SUB_METRICS, start=1):
        for label, data in metric_data.items():
            if data is None:
                continue
                
            v = data['unique_v']
            max_vals = data[f'max_{sub_metric}']
            max_sigs = data[f'max_sig_{sub_metric}']
            # Get the overlap values at which max was achieved
            max_overlaps = data.get(f'max_overlap_{sub_metric}', np.full_like(max_vals, np.nan))
            
            # Filter out NaN values
            mask = ~np.isnan(max_vals)
            if not any(mask):
                continue
                
            v = v[mask]
            max_vals = max_vals[mask]
            max_sigs = max_sigs[mask]
            max_overlaps = max_overlaps[mask]

            # Only show legend for first subplot
            show_legend = not legend_added[label]
            legend_added[label] = True

            # Create hover text with overlap information
            hover_text = [
                f"<b>{label}</b><br>"
                f"Task Similarity: {x_val:.2f}<br>"
                f"Max Metric Value: {y_val:.4f}<br>"
                f"Error: ±{err:.4f}<br>"
                f"Overlap: {ov:.4f}"
                for x_val, y_val, err, ov in zip(v, max_vals, max_sigs, max_overlaps)
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=v,
                    y=max_vals,
                    mode='lines+markers',
                    name=label,
                    legendgroup=label,
                    showlegend=show_legend,
                    line=dict(color=METRIC_COLORS.get(label, 'rgb(51,51,51)')), 
                    marker=dict(size=8),
                    error_y=dict(
                        type='data',
                        array=max_sigs,
                        visible=True,
                        thickness=1.5,
                        width=4
                    ),
                    text=hover_text,
                    hoverinfo='text',
                ),
                row=1, col=idx
            )
            
            # Add shaded error region
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([v, v[::-1]]),
                    y=np.concatenate([max_vals + max_sigs, (max_vals - max_sigs)[::-1]]),
                    fill='toself',
                    fillcolor=METRIC_COLORS.get(label, 'rgb(51,51,51)').replace('rgb', 'rgba').replace(')', ', 0.2)'),
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo='skip',
                    showlegend=False,
                    legendgroup=label
                ),
                row=1, col=idx
            )
        
        # Update axis labels
        fig.update_xaxes(title_text="Task Similarity (v)", row=1, col=idx)
        fig.update_yaxes(title_text="Max Metric Value", row=1, col=idx)
    
    # Final layout adjustments
    fig.update_layout(
        title_text=f"Max Values Comparison ({metric_name})",
        height=500,
        width=1500,
        template="plotly_white",
        font=dict(family="Arial", size=14),
        # legend=dict(
        #     # orientation="h",
        #     yanchor="bottom",
        #     y=1,
        #     xanchor="right",
        #     x=1.02,
            
        # ),
        margin=dict(t=100, b=80, l=60, r=60)
    )
    return fig

def main():
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting plot generation...")
    print(f"Folders being processed:")
    for path, label in FOLDERS.items():
        print(f"  {label}: {path}")
    
    # Process each metric type
    for metric_name in METRIC_FILES.keys():
        metric_data = {}
        print(f"\nProcessing {metric_name} metric...")
        
        for folder_path, label in FOLDERS.items():
            folder = Path(folder_path)
            print(f"  Loading data for '{label}'...")
            data = load_npz_data(folder, metric_name)
            metric_data[label] = data
        
        # Skip if no valid data found
        if all(data is None for data in metric_data.values()):
            print(f"⚠️ No valid data found for {metric_name}. Skipping...")
            continue
            
        # Create and save plot
        fig = create_metric_plot(metric_data, metric_name)
        fig.write_html(output_dir / f"max_values_{metric_name}.html")
        print(f"✅ Saved plot to {output_dir}/max_values_{metric_name}.html")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()