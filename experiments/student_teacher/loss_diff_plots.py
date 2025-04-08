# -*- coding: utf-8 -*-
"""
Module to load, analyze, and plot loss data from task-switching experiments.
Provides a function `generate_loss_plots` to perform the analysis and
return the generated figures.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ipdb import set_trace

# Type alias for clarity
ResultDict = Dict[str, Any]


def generate_loss_plots(
    data_dir: str | Path,
    switch_point: int,
    steps_after_switch: int = 500000,
    min_runs_for_std: int = 2,
    loss_file_pattern: str = r"losses_v_(\d+\.\d+)\.npz",
    show_plots: bool = False, # Control whether to .show() plots internally
    verbose: bool = True # Control print statements
) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    """
    Loads loss data, analyzes differences, and generates plots.

    Args:
        data_dir: Path to the directory containing the .npz loss files.
        steps_after_switch: Number of steps after the switch epoch to analyze
                            the loss difference.
        min_runs_for_std: Minimum number of runs required to calculate std dev.
        loss_file_pattern: Regex pattern to match filenames and extract 'v'.
        show_plots: If True, call fig.show() for each generated plot.
        verbose: If True, print progress messages and warnings.

    Returns:
        A tuple containing two Plotly Figure objects:
        (fig_loss_curves, fig_difference_plots)
        Either figure can be None if data loading or calculation fails.
    """

    # --- Helper Functions (nested inside or defined at module level) ---
    # Keeping them nested here for full encapsulation within this function's context

    def _log(message: str):
        """Helper to print only if verbose is True."""
        if verbose:
            print(message)

    def _calculate_error_difference_stats(
        results_list: List[ResultDict],
        switch_epoch_val: int,
        target_epoch_after_val: int,
        loss_key: str
    ) -> Tuple[List[float], List[float], List[float], int, int]:
        """ Calculates mean/std dev of error differences across runs. (Internal Helper) """
        v_values: List[float] = []
        mean_diffs: List[float] = []
        std_diffs: List[float] = []
        actual_switch_epochs: List[int] = []
        actual_target_epochs: List[int] = []

        task_num_str = '1' if 'loss1' in loss_key else '2'
        _log(f"\nCalculating Task {task_num_str} error difference between epochs ~{switch_epoch_val} and ~{target_epoch_after_val}")

        for res in results_list:
            epochs = res['epochs']
            loss_raw = res[loss_key] # Shape: [epochs, runs]
            v = res['v']
            num_runs = res['num_runs']

            if epochs.size == 0 or loss_raw.size == 0:
                _log(f"Warning: Skipping v={v:.2f} for Task {task_num_str} due to empty data.")
                continue

            # Find closest indices and actual epoch values
            switch_idx = np.argmin(np.abs(epochs - switch_epoch_val))
            actual_switch = int(epochs[switch_idx])
            target_idx = np.argmin(np.abs(epochs - target_epoch_after_val))
            actual_target = int(epochs[target_idx])

            if actual_target <= actual_switch:
                _log(f"Warning: Skipping v={v:.2f} for Task {task_num_str}. Closest target epoch ({actual_target}) <= closest switch epoch ({actual_switch}). Increase steps_after_switch?")
                continue

            # Calculate difference for each run
            losses_at_switch = loss_raw[switch_idx, :]      # Shape: [runs]
            losses_at_target = loss_raw[target_idx, :]      # Shape: [runs]
            differences_all_runs = losses_at_target - losses_at_switch

            # Calculate mean and std dev of differences
            mean_diff = float(np.mean(differences_all_runs))
            std_diff = 0.0
            if num_runs >= min_runs_for_std:
                std_diff = float(np.std(differences_all_runs))
            else:
                _log(f"Warning: v={v:.2f} for Task {task_num_str} has only {num_runs} run(s) (< {min_runs_for_std}), std dev of difference set to 0.")

            _log(f"  Task {task_num_str} v={v:.2f}: Using epochs {actual_switch} & {actual_target}. Mean Diff = {mean_diff:.4e}, Std Diff = {std_diff:.4e} ({num_runs} runs)")

            v_values.append(v)
            mean_diffs.append(mean_diff)
            std_diffs.append(std_diff)
            actual_switch_epochs.append(actual_switch)
            actual_target_epochs.append(actual_target)

        report_switch_epoch = actual_switch_epochs[0] if actual_switch_epochs else switch_epoch_val
        report_target_epoch = actual_target_epochs[0] if actual_target_epochs else target_epoch_after_val

        return v_values, mean_diffs, std_diffs, report_switch_epoch, report_target_epoch

    def _add_mean_error_difference_subplot(
        fig: go.Figure,
        row: int, col: int,
        v_values: List[float],
        mean_diffs: List[float],
        std_diffs: List[float],
        task_label: str
    ) -> None:
        """ Adds a scatter plot of mean error difference vs. 'v' to a subplot. (Internal Helper) """
        if not v_values:
            _log(f"\nCannot create difference plot for {task_label}: No valid data points calculated.")
            fig.add_annotation(text=f"No difference data<br>to plot for {task_label}",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                               font=dict(size=10, color="grey"),
                               row=row, col=col)
            return

        v_np = np.array(v_values)
        mean_np = np.array(mean_diffs)
        std_np = np.array(std_diffs)

        sort_indices = np.argsort(v_np)
        v_sorted = v_np[sort_indices]
        mean_sorted = mean_np[sort_indices]
        std_sorted = std_np[sort_indices]

        upper_bound = mean_sorted + std_sorted
        lower_bound = mean_sorted - std_sorted

        # Find min/max v for consistent color scaling
        min_v, max_v = (min(v_sorted), max(v_sorted)) if v_sorted.size > 0 else (0, 1)

        # Add shaded standard deviation band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([v_sorted, v_sorted[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill="toself", fillcolor='rgba(0,100,80,0.15)',
                line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip",
                showlegend=False, name=f'{task_label} Std Dev Range'
            ), row=row, col=col
        )
        # Add the mean line
        fig.add_trace(
            go.Scatter(
                x=v_sorted, y=mean_sorted, mode='lines',
                line=dict(color='grey', width=1.5, dash='dash'),
                showlegend=False, name=f'{task_label} Mean Difference (Line)'
            ), row=row, col=col
        )
        # Add markers colored by v
        marker_colors = [px.colors.sample_colorscale('viridis', (v - min_v) / (max_v - min_v) if max_v > min_v else 0.5)[0] for v in v_sorted]
        fig.add_trace(
            go.Scatter(
                x=v_sorted, y=mean_sorted, mode='markers',
                marker=dict(color=marker_colors, size=10),
                name=f'{task_label} Mean Difference',
                customdata=np.stack((std_sorted,), axis=-1),
                hovertemplate=(f"<b>{task_label}</b><br>Similarity (v): %{{x:.2f}}<br>"
                               "Mean Difference: %{y:.3e}<br>"
                               "Std Dev Diff: %{customdata[0]:.3e}<extra></extra>"),
                showlegend=False
            ), row=row, col=col
        )

    # --- Main Function Logic ---

    # 1. Data Loading & Preprocessing
    data_path = Path(data_dir)
    results: List[ResultDict] = []
    required_keys = ["test_loss1", "test_loss2", "epochs"]
    pattern = re.compile(loss_file_pattern)
    fig_loss: Optional[go.Figure] = None
    fig_diff: Optional[go.Figure] = None

    if not data_path.is_dir():
        _log(f"Error: Data directory not found: {data_path}")
        return None, None

    _log(f"Loading data from: {data_path}")
    num_files_processed = 0
    for filepath in data_path.glob("*.npz"):
        match = pattern.match(filepath.name)
        if not match:
            _log(f"Skipping file with unexpected format: {filepath.name}")
            continue

        v_value = float(match.group(1))
        try:
            with np.load(filepath) as data:
                if not all(key in data for key in required_keys):
                    _log(f"Warning: Skipping {filepath.name}, missing required keys ({required_keys}).")
                    continue

                epochs_data = data['epochs']
                loss1_data = data["test_loss1"]
                loss2_data = data["test_loss2"]

                if any(arr.size == 0 for arr in [epochs_data, loss1_data, loss2_data]):
                    _log(f"Warning: Skipping {filepath.name}, contains empty arrays.")
                    continue
                if loss1_data.ndim != 2 or loss2_data.ndim != 2:
                    _log(f"Warning: Skipping {filepath.name}, loss data not 2D. Shapes: {loss1_data.shape}, {loss2_data.shape}")
                    continue

                num_runs = loss1_data.shape[1]
                if num_runs != loss2_data.shape[1]:
                    _log(f"Warning: Skipping {filepath.name}, mismatched runs: {num_runs} vs {loss2_data.shape[1]}.")
                    continue
                if loss1_data.shape[0] != epochs_data.shape[0] or loss2_data.shape[0] != epochs_data.shape[0]:
                    _log(f"Warning: Skipping {filepath.name}, mismatched samples: epochs({epochs_data.shape[0]}) vs losses({loss1_data.shape[0]}, {loss2_data.shape[0]}).")
                    continue

                results.append({
                    'v': v_value,
                    'epochs': epochs_data,
                    'test_loss1_mean': np.mean(loss1_data, axis=1),
                    'test_loss1_std': np.std(loss1_data, axis=1),
                    'test_loss1_raw': loss1_data,
                    'test_loss2_mean': np.mean(loss2_data, axis=1),
                    'test_loss2_std': np.std(loss2_data, axis=1),
                    'test_loss2_raw': loss2_data,
                    'num_runs': num_runs
                })
                num_files_processed += 1
        except Exception as e:
            _log(f"Error loading or processing {filepath.name}: {e}")
            continue

    if not results:
        _log("Error: No valid data loaded after processing files.")
        return None, None

    results.sort(key=lambda x: x['v'])
    _log(f"Successfully loaded and processed data for {len(results)} 'v' values from {num_files_processed} files.")

    # 2. Determine Switch Epoch
    first_epochs = results[0].get('epochs')
    if first_epochs is None or first_epochs.size == 0:
        _log("Error: Epochs array is empty in the first result. Cannot determine switch epoch.")
        return None, None

    # if first_epochs.size < 2:
    #     _log("Warning: Very few epochs found. Using the first epoch as switch point.")
    #     switch_epoch_index = 0
    # else:
    #     switch_epoch_index = first_epochs.size // 2

    switch_epoch_index = (first_epochs==switch_point).nonzero()[0][0]
    
    switch_epoch = int(first_epochs[switch_epoch_index])
    assert switch_epoch == switch_point, set_trace()
    target_epoch_after = switch_epoch + steps_after_switch
    _log(f"Switch epoch determined as: {switch_epoch} (index {switch_epoch_index})")
    _log(f"Target epoch for difference calculation: {target_epoch_after}")

    # 3. Plotting the Original Loss Curves
    _log("\nGenerating original loss curves plot...")
    fig_loss = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Task 1 Test Loss", "Task 2 Test Loss"))

    v_s = [res['v'] for res in results if res.get('epochs', np.array([])).size > 0]
    min_v, max_v = (min(v_s), max(v_s)) if v_s else (0, 1)
    # set_trace()
    for res in results:
        epochs = res.get('epochs')
        if epochs is None or epochs.size == 0: continue
        v = res['v']
        # Normalize v for colorscale lookup
        norm_v = (v - min_v) / (max_v - min_v) if max_v > min_v else 0.5
        color = px.colors.sample_colorscale('viridis', norm_v)[0]

        # Task 1
        mean1, std1 = res['test_loss1_mean'], res['test_loss1_std']
        upper1, lower1 = mean1 + std1, mean1 - std1
        fig_loss.add_trace(go.Scatter(x=np.concatenate([epochs, epochs[::-1]]), y=np.concatenate([upper1, lower1[::-1]]), fill="toself", fillcolor=color, line=dict(color='rgba(255,255,255,0)'), opacity=0.2, showlegend=False, hoverinfo='skip'), row=1, col=1)
        fig_loss.add_trace(go.Scatter(x=epochs, y=mean1, mode='lines', line=dict(color=color), name=f"v={v:.2f}", legendgroup=f"v={v:.2f}", showlegend=False), row=1, col=1)

        # Task 2
        mean2, std2 = res['test_loss2_mean'], res['test_loss2_std']
        upper2, lower2 = mean2 + std2, mean2 - std2
        fig_loss.add_trace(go.Scatter(x=np.concatenate([epochs, epochs[::-1]]), y=np.concatenate([upper2, lower2[::-1]]), fill="toself", fillcolor=color, line=dict(color='rgba(255,255,255,0)'), opacity=0.2, showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig_loss.add_trace(go.Scatter(x=epochs, y=mean2, mode='lines', line=dict(color=color), name=f"v={v:.2f}", legendgroup=f"v={v:.2f}", showlegend=False), row=2, col=1)

    # Add dummy traces for legend
    added_legends = set()
    for res in results:
         v = res['v']
         if v not in added_legends:
             norm_v = (v - min_v) / (max_v - min_v) if max_v > min_v else 0.5
             color = px.colors.sample_colorscale('viridis', norm_v)[0]
             fig_loss.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=color), legendgroup=f"v={v:.2f}", name=f"v={v:.2f}"))
             added_legends.add(v)

    fig_loss.add_vline(x=switch_epoch, line=dict(color='black', dash='dash', width=1.5), annotation_text="Switch Epoch", annotation_position="top right")
    fig_loss.update_yaxes(type="log", title_text='Loss (Log Scale)', exponentformat="e", row=1, col=1)
    fig_loss.update_yaxes(type="log", title_text='Loss (Log Scale)', exponentformat="e", row=2, col=1)
    fig_loss.update_xaxes(title_text="Training Steps", row=2, col=1)
    fig_loss.update_layout(height=700, width=900, title_text="Test Loss During Training", template='plotly_white', hovermode='x unified', legend_title_text='Similarity (v)')

    if show_plots:
        _log("Displaying loss curves plot...")
        fig_loss.show()

    # 4. Calculate and Plot Mean/Std Error Difference
    _log("\nCalculating difference stats for plotting...")
    v1, mean1, std1, epoch_s1, epoch_t1 = _calculate_error_difference_stats(results, switch_epoch, target_epoch_after, 'test_loss1_raw')
    v2, mean2, std2, epoch_s2, epoch_t2 = _calculate_error_difference_stats(results, switch_epoch, target_epoch_after, 'test_loss2_raw')

    used_switch_epoch = epoch_s2 # Use consistent reported epochs
    used_target_epoch = epoch_t2

    if not v1 and not v2:
        _log("Warning: Could not calculate error differences for either task. Skipping difference plot generation.")
        # fig_diff remains None
    else:
        _log("\nGenerating mean error difference subplots...")
        fig_diff = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=(f'Task 1 Loss Difference (Epoch {used_target_epoch} - Epoch {used_switch_epoch})',
                            f'Task 2 Loss Difference (Epoch {used_target_epoch} - Epoch {used_switch_epoch})')
        )
        _add_mean_error_difference_subplot(fig=fig_diff, row=1, col=1, v_values=v1, mean_diffs=mean1, std_diffs=std1, task_label='Task 1')
        _add_mean_error_difference_subplot(fig=fig_diff, row=2, col=1, v_values=v2, mean_diffs=mean2, std_diffs=std2, task_label='Task 2')

        fig_diff.update_layout(
            height=800, width=800,
            title=f'Mean Loss Difference vs. Task Similarity (v)<br><sup>Comparing loss {steps_after_switch} steps after switch (epoch ~{used_switch_epoch})</sup>',
            template='plotly_white', hovermode='closest', showlegend=False
        )
        y_axis_title = f'Mean Loss Difference'
        fig_diff.update_yaxes(title_text=y_axis_title, exponentformat="e", row=1, col=1)
        fig_diff.update_yaxes(title_text=y_axis_title, exponentformat="e", zeroline=True, zerolinecolor='lightgrey', zerolinewidth=1, row=2, col=1)
        fig_diff.update_xaxes(title_text='Task Similarity (v)', row=2, col=1)

        if show_plots:
             _log("Displaying mean error difference plot...")
             fig_diff.show()

    _log("\nPlot generation function finished.")
    return fig_loss, fig_diff


# --- Example Usage ---
# This block will only run when the script is executed directly
# (e.g., python your_script_name.py)
# It will NOT run when the `generate_loss_plots` function is imported.
if __name__ == "__main__":
    print("Running example usage of generate_loss_plots...")

    # Define the directory where your data lives for the example
    EXAMPLE_DATA_DIR = "/home/users/MTrappett/mtrl/RepresentationSimilarityCL/loss_data/tests/new_switch_point/" # Adjust if needed

    # Call the main function
    # Set show_plots=True if you want plots to display automatically here
    figure1, figure2 = generate_loss_plots(
        data_dir=EXAMPLE_DATA_DIR,
        switch_point=1_500_000,
        steps_after_switch=500000, # Example value
        show_plots=True,           # Show plots when run directly
        verbose=True               # Show log messages
    )

    # You can now work with the returned figures if needed
    if figure1:
        print("Loss curve figure generated.")
        # Example: Save the figure
        # figure1.write_html("example_loss_curves.html")
        pass
    else:
        print("Loss curve figure generation failed or was skipped.")

    if figure2:
        print("Difference plot figure generated.")
        # Example: Save the figure
        # figure2.write_html("example_difference_plot.html")
        pass
    else:
        print("Difference plot figure generation failed or was skipped.")

    print("\nExample usage finished.")