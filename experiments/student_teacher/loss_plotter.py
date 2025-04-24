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

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipdb import set_trace

# ============  utility ============

LossDict = Dict[str, Any]
FILE_REGEX = re.compile(r"g_type_([^_]+)_v_(\d+\.\d+)_spars_(\d+\.\d+)\.npz") # f"g_type_{g_type}_v_{v:.2f}_spars_{sparsity:.2f}.npz"

def _log(msg: str, verbose: bool):
    if verbose:
        print(msg)

# ============  main plotting routine ============

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

                results.append({
                    "v"            : v_val,
                    "epochs"       : dat["epochs"],
                    "loss1_raw"    : dat["test_loss1"],
                    "loss2_raw"    : dat["test_loss2"],
                    "loss1_mean"   : np.mean(dat["test_loss1"], axis=1),
                    "loss1_std"    : np.std (dat["test_loss1"], axis=1),
                    "loss2_mean"   : np.mean(dat["test_loss2"], axis=1),
                    "loss2_std"    : np.std (dat["test_loss2"], axis=1),
                    "num_runs"     : dat["test_loss1"].shape[1],
                    "num_epochs"   : dat["num_epochs"]
                })
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
    fig_loss = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                             subplot_titles=("Task 1 test loss", "Task 2 test loss"))

    v_values = [d["v"] for d in results]
    v_min, v_max = min(v_values), max(v_values)
    for d in results:
        col = px.colors.sample_colorscale("viridis",
                                          (d["v"] - v_min)/(v_max - v_min) if v_max>v_min else 0.5)[0]
        for row, (mean, sd) in enumerate([(d["loss1_mean"], d["loss1_std"]),
                                          (d["loss2_mean"], d["loss2_std"])], start=1):
            epochs = d["epochs"]
            fig_loss.add_trace(
                go.Scatter(x=np.concatenate([epochs, epochs[::-1]]),
                           y=np.concatenate([mean+sd, (mean-sd)[::-1]]),
                           fill="toself", fillcolor=col, line=dict(color="rgba(0,0,0,0)"),
                           hoverinfo="skip", legendgroup=f"v={d['v']:.2f}", showlegend=False),
                row=row, col=1)
            fig_loss.add_trace(
                go.Scatter(x=epochs, y=mean, mode="lines",
                           line=dict(color=col), name=f"v={d['v']:.2f}",
                           legendgroup=f"v={d['v']:.2f}", showlegend=(row==1)),
                row=row, col=1)

    fig_loss.add_vline(x=switch_point, line=dict(color="black", dash="dash"),
                       annotation_text="switch", annotation_position="top right")
    fig_loss.update_yaxes(type="log", exponentformat="e")
    fig_loss.update_xaxes(title_text="training step", row=2, col=1)
    fig_loss.update_layout(title="Test loss during training",
                           template="plotly_white", height=650, width=900)

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
            t_switch = np.argmin(np.abs(epochs - switch_point))
            t_target = np.argmin(np.abs(epochs - (switch_point + steps_after_switch)))
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
        fig_diff.update_layout(title="Loss jump after task switch",
                               template="plotly_white", height=700, width=800)
        if show_plots:
            fig_diff.show()
        

    return fig_loss, fig_diff

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
