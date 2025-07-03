#!/usr/bin/env python
# loss_plotter.py
"""
Reads *.npz result files produced by sparsity.py and makes two Plotly figures:
(1) Task-specific test-loss curves (with  ±1 σ ribbons)
(2) Mean loss-difference ΔL = L(t_switch+Δ) − L(t_switch) as a function of similarity v
Only the experiment directory is required as a command-line argument.
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

# ============  utility ============

LossDict = Dict[str, Any]
FILE_REGEX = re.compile(r"g_type_([^_]+)_v_(\d+\.\d+)_spars_(\d+\.\d+)\.npz") # f"g_type_{g_type}_v_{v:.2f}_spars_{sparsity:.2f}.npz"

def _log(msg: str, verbose: bool):
    if verbose:
        print(msg)

# ============  main plotting routine ============
# ============  New Metric Functions & Helpers ============
import jax.numpy as jnp

def my_trap(y,x):
    return jnp.trapezoid(y, x)

def my_med(y, x):
    return jnp.median(y)

def my_mean(y, x):
    return jnp.mean(y)

def my_min(y, x):
    return jnp.min(y)

def get_metric_func(metric: str):
    if metric == 'auc':
        return my_trap
    elif metric == 'median':
        return my_med
    elif metric == 'mean':
        return my_mean
    elif metric == 'min':
        return my_min
    raise ValueError(f"Unknown metric: {metric}")


def load_expert_metric_data(data_dir: Path, verbose: bool=True) -> Tuple[Dict, Dict]:
    expert_regex = re.compile(r"v_(\d+\.\d+).npz") 
    meta_ref = None
    grouped={}
    for dir_path in [data_dir]:
        print(f"{dir_path.exists()=}")
        if not dir_path.exists():
            continue
            
        for npz_path in dir_path.glob("*.npz"):
            m = expert_regex.match(npz_path.name)
            if not m:
                continue
            try:
                with np.load(npz_path) as dat:
                    if not all(k in dat for k in ["train_loss", "test_loss", "epochs"]):
                        continue
                    grouped[m.group(1)] = {
                        "v": float(m.group(1)),
                        "epochs": dat["epochs"],
                        "train_loss": dat["train_loss"],
                        "test_loss": dat["test_loss"],
                        "num_runs": 20,
                    }
            except Exception as e:
                _log(f"Error reading{npz_path.name}: {e}", verbose)
    
    return grouped, meta_ref

def get_expert_losses():
    path_to_experts = Path("/home/users/MTrappett/mtrl/RepresentationSimilarityCL/loss_data/d_ht_expert/d_h_200_d_hs_200_lr_0.1/")
    metric_data, _ = load_expert_metric_data(path_to_experts)
    expert_losses = {}
    for k,v in metric_data.items():
        # print(f"auc for {v["v"]}: {np.trapezoid(v['test_loss'].T, v['epochs'].T).shape=}")
        expert_losses[v["v"]] = v['test_loss'] # shape is (149,20)

    return expert_losses

expert_losses = get_expert_losses()

def calc_metric(data_t1: np.ndarray, data_t2: np.ndarray, run: int, v: int, switch_idx: int, metric: str = 'auc', epochs: list = []):
    # input: data_t1: array shape: (downsampled_epochs,runs) test loss data for task 1
    #        data_t2: array shape: (downsample_epochs, runs) test loss data for task 2
    #        run: which run we are using, index for expert
    #        v: which similarity expert we are comparing against
    #        switch_idx: When we switch task training
    #       metric: how we are agregating over the data, defualt to AUC
    #       
    # return metrics: rem, transfer and zeroshot
    
    i_x         = epochs[:switch_idx]
    j_x         = epochs[switch_idx:]
    data_ii     = data_t1[:switch_idx, run]
    data_ij     = data_t1[switch_idx:, run]
    metric_func = get_metric_func(metric)
    metric_ii   = metric_func(jnp.array(data_ii), i_x)
    metric_ij   = metric_func(jnp.array(data_ij), j_x)
    exp_metric  = metric_func(expert_losses[v].T[:switch_idx,run], i_x)
    data_ji     = data_t2[:switch_idx, run]
    data_jj     = data_t2[switch_idx:, run]
    metric_ji   = metric_func(jnp.array(data_ji), i_x)
    metric_jj   = metric_func(jnp.array(data_jj), j_x)
    t1          = data_t2[0, run]
    t2          = data_t2[switch_idx-1, run]
    # =============calculate metrics=============: 
    rem = (metric_ii - metric_ij)/(metric_ii+metric_ij)
    ft_t1  = (exp_metric - metric_ii)/(exp_metric + metric_ii)
    # set_trace()
    zs = (t1 - t2)/(t1+t2)
    # zs = - metric_ji
    ft_t2 = (exp_metric - metric_jj)/(exp_metric + metric_jj)
    return rem, ft_t1, ft_t2, zs


# ============  Refactored Data Loading ============
def load_grouped_metric_data(data_dir: Path, verbose: bool=True) -> Tuple[Dict, Dict]:
    """Load and validate data from both experiment directories (random/determ)."""
    current_dir_name = data_dir.name
    if 'random' in current_dir_name:
        g_types = ['random', 'determ']
    elif 'determ' in current_dir_name:
        g_types = ['determ', 'random']
    else:
        raise ValueError("Directory name must contain 'random' or 'determ'")

    grouped = {gt: [] for gt in g_types}
    meta_ref = None

    for dir_path in [data_dir, data_dir.parent / current_dir_name.replace(g_types[0], g_types[1], 1)]:
        if not dir_path.exists():
            continue
            
        for npz_path in dir_path.glob("*.npz"):
            m = FILE_REGEX.match(npz_path.name)
            if not m:
                continue

            try:
                with np.load(npz_path) as dat:
                    if not all(k in dat for k in ["test_loss1", "test_loss2", "epochs", "switch_point", "sparsity"]):
                        continue

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
                        "loss1_raw": dat["test_loss1"],
                        "loss2_raw": dat["test_loss2"],
                        "switch_point": int(meta["switch_point"]),
                        "num_runs": dat["test_loss1"].shape[1],
                        "overlap": dat["overlap"],
                    })
            except Exception as e:
                _log(f"Error reading {npz_path.name}: {e}", verbose)
    
    return grouped, meta_ref


# ----------------------------------------------------------------------
# Helper container -----------------------------------------------------
# ----------------------------------------------------------------------
@dataclass
class MetricSeries:
    """Holds v‑values plus mean / std for one metric curve."""
    v:   List[float] = field(default_factory=list)
    mu:  List[float] = field(default_factory=list)
    sig: List[float] = field(default_factory=list)
    overlap_mu: List[float] = field(default_factory=list)
    overlap_sig: List[float] = field(default_factory=list)

    def sorted_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        order = np.argsort(self.v)
        return (np.asarray(self.v)[order],
                np.asarray(self.mu)[order],
                np.asarray(self.sig)[order],
                np.asarray(self.overlap_mu)[order],
                np.asarray(self.overlap_sig)[order])

    

# ============  Main Metric Plot Function ============
def generate_metric_plot(
    data_dir: str | Path,
    metric: str = 'auc',
    min_runs_for_std: int = 2,
    show_plots: bool = False,
    verbose: bool = True
) -> Optional[go.Figure]:
    """Generate plot of specified metric vs similarity for both tasks."""
    data_dir = Path(data_dir)
    grouped, meta_ref = load_grouped_metric_data(data_dir, verbose)
    if not grouped or meta_ref is None:
        return None
    # set_trace()
    # ========== Load overlap max metrics ==========
    try:
        # Get the base path from data_dir (assuming structure: base_path/loss_data/sparsity_X)
        base_path = data_dir.parent.parent
        # Path to overlap metrics
        overlap_path = Path("/home/users/MTrappett/mtrl/RepresentationSimilarityCL/loss_data/overlap_search_plots")
        overlap_file = overlap_path / f"overlap_metrics_{metric}.npz"
        
        if overlap_file.exists():
            overlap_data = np.load(overlap_file)
            print(f"Loaded overlap metrics from: {overlap_file}")
        else:
            print(f"Overlap file not found: {overlap_file}")
            overlap_data = None
    except Exception as e:
        print(f"Error loading overlap metrics: {str(e)}")
        overlap_data = None
    # ========== END overlap ==========

    metric_names = ("rem", "ft_t1", "ft_t2", "zs")
    colors = {
        'random': 'rgba(255,0,0,1)',
        'determ': 'rgba(0,0,255,1)',
        'overlap_max': 'rgba(0,128,0,1)'  # Green for overlap max
    }
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                       subplot_titles=[f"{name.upper()} {metric.upper()}" for name in metric_names])
        # g_type  -> metric_name -> MetricSeries
    series: Dict[str, Dict[str, MetricSeries]] = {
        g: {m: MetricSeries() for m in metric_names} for g in colors
    }

    # ------------------------------------------------------------------
    # Aggregate stats ---------------------------------------------------
    # ------------------------------------------------------------------
    for g_type, exps in grouped.items():
        for exp in sorted(exps, key=lambda e: e["v"]):
            switch_idx = int(np.argmin(np.abs(exp["epochs"] - exp["switch_point"])))

            # run‑wise metric computation (your existing helper)
            per_run = [
                calc_metric(
                    exp["loss1_raw"], exp["loss2_raw"],
                    r, exp["v"], switch_idx, metric, meta_ref['epochs']
                )
                for r in range(exp["loss1_raw"].shape[1])
            ]
            if len(per_run) < min_runs_for_std:
                continue

            rem, ft1, ft2, zs = map(np.asarray, zip(*per_run))
            v = float(exp["v"])

            overlap_mu = np.mean(exp["overlap"])
            overlap_sig = np.std(exp["overlap"])

            # Store metrics
            for metric_name, values in zip(metric_names, [rem, ft1, ft2, zs]):
                series[g_type][metric_name].v.append(v)
                series[g_type][metric_name].mu.append(values.mean())
                series[g_type][metric_name].sig.append(values.std())
                series[g_type][metric_name].overlap_mu.append(overlap_mu)
                series[g_type][metric_name].overlap_sig.append(overlap_sig)

    # ------------------------------------------------------------------
    # Plotting ----------------------------------------------------------
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=1, cols=4, shared_xaxes=True, vertical_spacing=0.07, horizontal_spacing=0.06,
        subplot_titles=[f"{name.upper()} {metric.upper()}" for name in metric_names],
    )

    for col, metric_name in enumerate(metric_names, start=1):
        for g_type, colr in colors.items():
            curve = series[g_type][metric_name]
            if not curve.v:
                continue

            v, m, s, om, osig = curve.sorted_arrays()

            # ribbon
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([v, v[::-1]]),
                    y=np.concatenate([m + s, (m - s)[::-1]]),
                    fill="toself",
                    fillcolor=colr.replace("1)", "0.18)"),
                    line_color="rgba(0,0,0,0)",
                    showlegend=False,
                    legendgroup=g_type,
                    hoverinfo='none',
                ),
                row=1, col=col,
            )

            # mean line
            fig.add_trace(
                go.Scatter(
                    x=v, y=m,
                    line=dict(color=colr, width=4),
                    name=g_type if col == 1 else None,   # one legend entry
                    legendgroup=g_type,
                    showlegend=(col == 1),
                    # HIGHLIGHT START
                    customdata=np.stack([om, osig], axis=-1),
                    hovertemplate=(
                        f"<b>{g_type}</b><br>"
                        "v: %{x:.2f}<br>"
                        f"{metric_name}: %{{y:.3f}}<br>"
                        "overlap: %{customdata[0]:.3f} ± %{customdata[1]:.3f}"
                        "<extra></extra>"
                    ),
                    # HIGHLIGHT END
                ),
                row=1, col=col,
            )       

        # axis labels row‑wise
        fig.update_yaxes(title_text=f"{metric_name}", row=1, col=col, exponentformat='e')
        # ========== Plot overlap max metrics ==========
        if overlap_data is not None:
            # Get overlap data for this metric
            v_arr = overlap_data['unique_v']
            max_vals = overlap_data[f'max_{metric_name}']
            max_sigs = overlap_data[f'max_sig_{metric_name}']
            max_overlaps = overlap_data[f'max_overlap_{metric_name}']

            # Overlap max line
            # Overlap error region
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([v_arr, v_arr[::-1]]),
                    y=np.concatenate([max_vals + max_sigs, (max_vals - max_sigs)[::-1]]),
                    fill='toself',
                    fillcolor=colors['overlap_max'].replace("1)", "0.2)"),
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    legendgroup='overlap_max',
                    hoverinfo='none',
                ),
                row=1, col=col,
            )
            
            fig.add_trace(
                go.Scatter(
                    x=v_arr,
                    y=max_vals,
                    mode='lines+markers',
                    line=dict(color=colors['overlap_max'], width=4, dash='dash'),
                    marker=dict(size=8, color=colors['overlap_max']),
                    name='Overlap Max' if col == 1 else None,
                    legendgroup='overlap_max',
                    showlegend=(col == 1),
                    customdata=max_overlaps,
                    hovertemplate=(
                    "<b>Overlap Max</b><br>"
                    "v: %{x:.2f}<br>"
                    f"{metric_name}: %{{y:.3f}}<br>"
                    "overlap: %{customdata:.3f}"
                    "<extra></extra>"
                    ),
                ),
                row=1, col=col,
            )

            
        # ========== END overlap plot ==========

    fig.update_xaxes(title_text="similarity v")
    fig.update_layout(
        title=f"CL Metrics: {metric.upper()} curves (mean ± std); Sparsity: {meta_ref['sparsity']}",
        height=400,
        width=2000,
        template="plotly_white",
        hovermode="closest", 
        font=dict(
                    family="DejaVu Sans Bold",
                    size=20,
                )
    )

    if show_plots:
        fig.show()

    return fig

def generate_diff_plot(
    data_dir: str | Path,
    *,
    steps_after_switch: int = 500_000,
    min_runs_for_std: int = 2,
    show_plots: bool = False,
    verbose: bool = True
) -> Optional[go.Figure]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Determine the other directory based on 'random' or 'determ' in the current directory name
    current_dir_name = data_dir.name
    if 'random' in current_dir_name:
        current_g_type = 'random'
        other_g_type = 'determ'
    elif 'determ' in current_dir_name:
        current_g_type = 'determ'
        other_g_type = 'random'
    else:
        raise ValueError(f"Directory name must contain 'random' or 'determ', got: {current_dir_name}")
    
    other_dir = data_dir.parent / current_dir_name.replace(current_g_type, other_g_type, 1)
    if not other_dir.exists():
        raise FileNotFoundError(f"Other directory not found for {other_g_type}: {other_dir}")

    results: List[Dict] = []
    meta_reference: Optional[Dict] = None

    # Load data from both directories
    for dir_path in [data_dir, other_dir]:
        for npz_path in dir_path.glob("*.npz"):
            m = FILE_REGEX.match(npz_path.name)
            if not m:
                continue
            g_type = m.group(1)
            v_val = float(m.group(2))

            try:
                with np.load(npz_path) as dat:
                    required = ["test_loss1", "test_loss2", "epochs", "switch_point"]
                    if not all(k in dat for k in required):
                        continue

                    meta = {k: dat[k].item() if dat[k].shape == () else dat[k]
                            for k in ["switch_point", "epochs", "lr", "num_epochs",
                                      "sparsity", "d_hs", "d_in", "g_type"]
                            if k in dat}

                    if meta_reference is None:
                        meta_reference = meta
                    else:
                        # Skip 'g_type' check, enforce others
                        for k, ref_val in meta_reference.items():
                            if k == 'g_type':
                                continue
                            if k not in meta or not np.array_equal(meta[k], ref_val):
                                raise ValueError(f"Meta-data mismatch in {npz_path.name}: {k} differs.")

                    results.append({
                        "v": v_val,
                        "g_type": g_type,
                        "epochs": dat["epochs"],
                        "loss1_raw": dat["test_loss1"],
                        "loss2_raw": dat["test_loss2"],
                        "num_runs": dat["test_loss1"].shape[1],
                    })
            except Exception as e:
                _log(f"Error reading {npz_path.name}: {e}", verbose)

    if not results:
        return None

    switch_point = int(meta_reference["switch_point"])
    results.sort(key=lambda x: (x['g_type'], x['v']))

    # Group results by g_type
    grouped = {}
    for d in results:
        key = d['g_type']
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(d)

    # Define colors for each g_type
    colors = {'random': 'rgba(255,0,0,1)', 'determ': 'rgba(0,0,255,1)'}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=[f"Task {i} ΔL (step {switch_point + steps_after_switch} - {switch_point})"
                                       for i in (1, 2)])

    def compute_delta_stats(group: List[Dict], task_key: str, mete_ref: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        v, mean, sd = [], [], []
        for d in group:
            epochs = d['epochs']
            t_switch_idx = np.argmin(np.abs(epochs - switch_point))
            t_target_idx = np.argmin(np.abs(epochs - (switch_point + steps_after_switch)))
            if t_target_idx <= t_switch_idx:
                continue

            losses = d[task_key]
            delta = losses[t_target_idx] - losses[t_switch_idx]
            if len(delta) < min_runs_for_std:
                continue

            v.append(d['v'])
            mean.append(np.mean(delta))
            sd.append(np.std(delta))
        return np.array(v), np.array(mean), np.array(sd)

    for row, task_key in enumerate(['loss1_raw', 'loss2_raw'], start=1):
        for g_type in ['random', 'determ']:
            if g_type not in grouped:
                continue
            v, m, s = compute_delta_stats(grouped[g_type], task_key, meta_reference)
            if len(v) == 0:
                continue

            order = np.argsort(v)
            v_sorted = v[order]
            m_sorted = m[order]
            s_sorted = s[order]

            # Confidence band
            fig.add_trace(go.Scatter(
                x=np.concatenate([v_sorted, v_sorted[::-1]]),
                y=np.concatenate([m_sorted + s_sorted, (m_sorted - s_sorted)[::-1]]),
                fill='toself',
                fillcolor=colors[g_type].replace('1)', '0.2)'),  # Adjust alpha for fill
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                legendgroup=g_type,
            ), row=row, col=1)

            # Line trace
            fig.add_trace(go.Scatter(
                x=v_sorted,
                y=m_sorted,
                mode='lines',
                line=dict(color=colors[g_type], dash='solid'),
                name=g_type,
                legendgroup=g_type,
                showlegend=(row == 1),
            ), row=row, col=1)

    fig.update_yaxes(title_text="Δ loss")
    fig.update_xaxes(title_text="similarity v", row=2, col=1)
    fig.update_layout(
        title=f"Loss Difference by g_type\nSparsity: {meta_reference['sparsity']}",
        template="plotly_white",
        height=700,
        width=800,
        hovermode="x unified"
    )

    if show_plots:
        fig.show()

    return fig


def generate_loss_plots(
    data_dir: str | Path,
    *,
    steps_after_switch: int = 500_000,   # can be overridden on CLI if desired
    min_runs_for_std: int = 2,
    show_plots: bool = False,
    verbose: bool = True
) -> Tuple[Optional[go.Figure], Optional[go.Figure]]:
    """
    Parameters
    ----------
    data_dir : str | Path
        Directory in which the training script stored all `*.npz` files for one grid search. 
    steps_after_switch : int
        The Δ between the switch epoch and the second evaluation point when computing ΔL.
    min_runs_for_std : int
        Require at least this many runs to draw a std-dev ribbon.
    show_plots : bool
        Call `fig.show()` so that the plots pop up immediately.
    verbose : bool
        Print progress information.
    """
    # set_trace()
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # -----------------------------------------------------------
    # 1. Load all files and collect meta data
    # -----------------------------------------------------------
    results: List[LossDict] = []
    meta_reference: Dict[str, Any] | None = None   # keep the first file’s meta data
    for npz_path in data_dir.glob("*.npz"):
        m = FILE_REGEX.match(npz_path.name)
        if not m:
            _log(f"Skipping   {npz_path.name}   (does not match naming scheme).", verbose)
            continue
        v_val = float(m.group(2))
        # set_trace()
        try:
            with np.load(npz_path) as dat:
                # -------- sanity checks --------
                required = ["test_loss1", "test_loss2", "epochs", "switch_point"]
                if not all(k in dat for k in required):
                    _log(f"Missing keys in {npz_path.name}; skipping.", verbose)
                    continue
                if dat["test_loss1"].ndim != 2:
                    _log(f"Expected 2-D loss arrays in {npz_path.name}; skipping.", verbose)
                    continue
                # -------- keep meta data --------
                meta = {k: dat[k].item() if dat[k].shape == () else dat[k]
                        for k in ["switch_point", "epochs", "lr", "num_epochs",
                                  "sparsity", "d_hs", "d_in", "g_type"]
                        if k in dat}
                if meta_reference is None:
                    meta_reference = meta
                else:
                    # Make sure every file belongs to the *same* experiment
                    for k, ref_val in meta_reference.items():
                        if k not in meta or not np.array_equal(meta[k], ref_val):
                            raise ValueError(
                                f"Meta-data mismatch in {npz_path.name}: {k} differs."
                            )
                overlap_arr = dat["overlap"] # old determ and random sill have overlap as key
                # set_trace()
                results.append({
                    "v"            : v_val,
                    "epochs"       : dat["epochs"],
                    "loss1_raw"    : dat["test_loss1"],
                    "loss2_raw"    : dat["test_loss2"],
                    "loss1_mean"   : np.mean(dat["test_loss1"], axis=1),
                    "loss1_std"    : np.std (dat["test_loss1"], axis=1),
                    "loss2_mean"   : np.mean(dat["test_loss2"], axis=1),
                    "loss2_std"    : np.std (dat["test_loss2"], axis=1),
                    "train_mean"   : np.mean(dat["train_loss"], axis=1),
                    "train_std"    : np.std(dat["train_loss"], axis=1),
                    "num_runs"     : dat["test_loss1"].shape[1],
                    "overlap_mean": np.mean(overlap_arr).item(),               # ### NEW ###
                    "overlap_std":  np.std(overlap_arr).item(), 
                })
                # set_trace()
        except Exception as e:
            _log(f"Could not read {npz_path.name}: {e}", verbose)

    if not results:
        raise RuntimeError("No valid result files found.")

    results.sort(key=lambda d: d["v"])   # sort by similarity parameter
    # unpack reference meta data
    switch_point  = int(meta_reference["switch_point"])
    num_epochs    = int(meta_reference["num_epochs"])
    # ------------------------------------------------------------------
    # 2. Build loss-curve figure
    # ------------------------------------------------------------------
    fig_loss = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                             subplot_titles=("Task 1 test loss", "Task 2 test loss", "Training Loss"))

    v_values = [d["v"] for d in results]
    v_min, v_max = min(v_values), max(v_values)
    for d in results:
        col = px.colors.sample_colorscale("viridis",
                                          (d["v"] - v_min)/(v_max - v_min) if v_max>v_min else 0.5)[0]

        for row, (mean, sd) in enumerate([(d["loss1_mean"], d["loss1_std"]),
                                          (d["loss2_mean"], d["loss2_std"]),
                                          (d["train_mean"], d["train_std"])], start=1):
            epochs = d["epochs"]
            custom = np.column_stack([
                np.full_like(epochs, d["overlap_mean"], dtype=float),
                np.full_like(epochs, d["overlap_std"], dtype=float)
            ])   
            fig_loss.add_trace(
                go.Scatter(x=np.concatenate([epochs, epochs[::-1]]),
                           y=np.concatenate([mean+sd, (mean-sd)[::-1]]),
                           fill="toself", fillcolor=col.replace("rgb(", "rgba(").replace(")", ",0.2)"), line=dict(color="rgba(0,0,0,0)"),
                           hoverinfo="skip", legendgroup=f"v={d['v']:.2f}", showlegend=False),
                row=row, col=1)
            fig_loss.add_trace(
                go.Scatter(x=epochs, y=mean, mode="lines",
                           line=dict(color=col, width=4), name=f"v={d['v']:.2f}",
                           legendgroup=f"v={d['v']:.2f}", showlegend=(row==1),customdata=custom,                            # ### NEW ###
                    hovertemplate=(
                        f"v   :{d['v']:.2f}<br>"
                        "step: %{x}<br>"
                        "loss: %{y:.3e}<br>"
                        "overlap: %{customdata[0]:.3f} ± %{customdata[1]:.3f}"
                        "<extra></extra>"
                    )  ),
                row=row, col=1)

    fig_loss.add_vline(x=switch_point, line=dict(color="black", dash="dash"),)
    fig_loss.update_yaxes(type="log", exponentformat="e")
    fig_loss.update_xaxes(title_text="training step", row=3, col=1)
    fig_loss.update_layout(title=f"Losses during training\nSparsity: {meta_reference['sparsity']}, Gating Type: {meta['g_type']}",
                           template="plotly_white", height=700, width=900, font=dict(
                               family="DejaVu Sans Bold",
                               size=20,
                           ))
                            # hovermode="x unified" )

    if show_plots:
        fig_loss.show()

    # ------------------------------------------------------------------
    # 3. Compute ΔL and make difference figure
    # ------------------------------------------------------------------
    def delta_stats(task_key: str
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        v, mean, sd = [], [], []
        for d in results:
            epochs = d["epochs"]
            t_switch = np.argmin(np.abs(epochs - switch_point)) - 1
            t_target = np.argmin(np.abs(epochs - (switch_point + steps_after_switch)))
            # set_trace()
            if t_target <= t_switch:
                continue
            delta = d[task_key][t_target] - d[task_key][t_switch]   # vector over runs
            v.append(d["v"])
            mean.append(np.mean(delta))
            sd.append(np.std(delta) if d["num_runs"] >= min_runs_for_std else 0.0)
        return np.asarray(v), np.asarray(mean), np.asarray(sd)

    v1, m1, s1 = delta_stats("loss1_raw")
    v2, m2, s2 = delta_stats("loss2_raw")

    if v1.size == 0 and v2.size == 0:
        fig_diff = None
    else:
        fig_diff = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                                 subplot_titles=[f"Task {i}  ΔL (step {switch_point+steps_after_switch} – {switch_point})"
                                                 for i in (1,2)])
        for row, (v, m, s) in enumerate([(v1, m1, s1), (v2, m2, s2)], start=1):
            if v.size == 0:
                continue
            order = np.argsort(v)
            v, m, s = v[order], m[order], s[order]
            col_band = "rgba(0,100,80,0.15)"
            fig_diff.add_trace(go.Scatter(
                x=np.concatenate([v, v[::-1]]),
                y=np.concatenate([m+s, (m-s)[::-1]]),
                fill="toself", fillcolor=col_band, line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip", showlegend=False), row=row, col=1)
            col_pts = [px.colors.sample_colorscale("viridis", (vv-v_min)/(v_max-v_min) if v_max>v_min else 0.5)[0]
                       for vv in v]
            fig_diff.add_trace(go.Scatter(
                x=v, y=m, mode="markers+lines",
                marker=dict(color=col_pts, size=10), line=dict(color="grey", dash="dash"),
                showlegend=False), row=row, col=1)

        fig_diff.update_yaxes(title_text="Δ loss")
        fig_diff.update_xaxes(title_text="similarity v", row=2, col=1)
        fig_diff.update_layout(title=f"Loss jump after task switch\nSparsity: {meta_reference['sparsity']}",
                               template="plotly_white", height=700, width=800)
        if show_plots:
            fig_diff.show()
        

    return fig_loss, fig_diff


def save_param_hist(param_sets: dict,
                    out_dir: str | Path,
                    tag: str,
                    bins: int = 60,
                    dpi: int = 150) -> None:
    """
    Make three histograms – initial, after‑task‑1, final – and save as a PNG.

    Parameters
    ----------
    param_sets : dict
        Output from vectorized_train_for_v:  keys = 'initial_params' | 'intermediate_params' | 'final_params'.
    out_dir    : directory where the `param_hists/` sub‑folder should be created.
    tag        : filename stem (e.g. 'g_type_random_v_0.20_spars_0.50').
    """
    # ------- flatten each PyTree into a single 1‑D array ----------
    def _flatten(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return np.concatenate([np.asarray(l).ravel() for l in leaves])

    vals_init = _flatten(param_sets["initial_params"])
    vals_mid  = _flatten(param_sets["intermediate_params"])
    vals_last = _flatten(param_sets["final_params"])

    # ------- plot ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, vals, title in zip(
        axes,
        (vals_init, vals_mid, vals_last),
        ("initial", "after task 1", "final")
    ):
        ax.hist(vals, bins=bins, density=True, alpha=0.80)
        ax.set_title(title)
        ax.set_xlabel("weight value")
    axes[0].set_ylabel("probability density")
    fig.suptitle(f"{tag} – parameter distributions")

    # ------- save ----------
    out_dir  = Path(out_dir) / "param_hists"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / f"{tag}_param_hist.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)                       # free memory for long sweeps
    print(f"Saved parameter‑hist → {out_png.resolve()}")


# ============  CLI ============

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Plot results of sparsity.py")
    p.add_argument("data_dir", help="directory that contains *.npz files")
    p.add_argument("--after", type=int, default=50_000,
                   help="steps after switch for ΔL (default: 500 k)")
    p.add_argument("--show", action="store_true", help="pop up figures")
    p.add_argument("--quiet", action="store_true", help="suppress log messages")
    args = p.parse_args()

    generate_loss_plots(args.data_dir,
                        steps_after_switch=args.after,
                        show_plots=args.show,
                        verbose=not args.quiet)
