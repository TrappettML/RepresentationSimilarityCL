import os
# Option A: read the external env (if launching with CUDA_VISIBLE_DEVICES=...)
print("ENV CUDA_VISIBLE_DEVICES (before imports):", os.environ.get("CUDA_VISIBLE_DEVICES"))

import argparse
import pickle
from functools import partial
from pathlib import Path

import jax

print("jax.devices():", jax.devices())
print("jax.local_devices():", jax.local_devices())

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import plotly.graph_objects as go
import plotly.express.colors as px_colors
from flax import linen as nn
from jax import jit, lax, random, vmap
from ipdb import set_trace

from make_masks import create_masks
from timer_class import Timer

from single_expert import train_experts, scaled_normal_init

def plot_losses(npz_path: str | Path, log_y: bool = True) -> None:
    """
    Plots training and test losses from a results file using Plotly.
    This version is robust to missing or invalid expert data and corrects
    visual artifacts.
    """
    npz_path = Path(npz_path)
    try:
        data = np.load(npz_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Results file not found at {npz_path}")
        return

    # --- Data Loading ---
    epochs = data["epochs"]
    train_loss = data["train_loss"]
    test_losses = data["test_losses"]
    switch_points = data.get("switch_points", []) # Use .get for safety
    ntasks = test_losses.shape[-1]
    
    epochs_per_task = int(data['epochs_per_task'])
    sample_rate = int(data['sample_rate'])

    
    # --- Extract similarity and calculate overlap stats ---
    similarity = data.get('similarity', float('nan'))
    overlaps = data.get('overlaps', [])
    overlap_mean = np.mean(overlaps) if len(overlaps) > 0 else float('nan')
    overlap_std = np.std(overlaps) if len(overlaps) > 0 else float('nan')

    # --- Mean and Std Calculation ---
    train_mean = train_loss.mean(axis=1)
    train_sem = train_loss.std(axis=1)/jnp.sqrt(train_loss.shape[1])
    test_means = test_losses.mean(axis=1)
    test_sems = test_losses.std(axis=1)/jnp.sqrt(train_loss.shape[1])

    fig = go.Figure()
    colors = px_colors.qualitative.Plotly

    def add_band(x, y_mean, y_std, name, color, dash=None):
        """Helper to add a line with a shaded error band."""
        is_finite = np.isfinite(y_mean) & np.isfinite(y_std)
        x, y_mean, y_std = x[is_finite], y_mean[is_finite], y_std[is_finite]
        if len(x) == 0: return

        y_upper, y_lower = y_mean + y_std, y_mean - y_std
        
        # *** FIX 1: Corrected alpha value for transparency ***
        rgba_color = color.replace('rgb', 'rgba').replace(')', ', 0.2)')

        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill='toself', fillcolor=rgba_color, opacity=0.2, line_width=0,
            hoverinfo="none", showlegend=False, legendgroup=name,
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, name=name,
            mode='lines', line=dict(color=color, dash=dash, width=2.5),
            legendgroup=name,
        ))

    # --- Plot Student Performance ---
    add_band(epochs, train_mean, train_sem, name="Student Train Loss", color='rgb(0,0,0)')
    for i in range(ntasks):
        color = colors[i % len(colors)]
        add_band(epochs, test_means[:, i], test_sems[:, i], name=f"Student on Task {i+1}", color=color)

    # --- Plot Expert Performance (with Data Validation) ---
    # *** FIX 2: Check for valid expert data before attempting to plot ***
    if 'expert_test_losses' not in data or not np.any(np.isfinite(data['expert_test_losses'])):
        print("\nWarning: 'expert_test_losses' key not found in npz file or contains no valid data.")
        print("         Skipping expert plots. Please check the 'train_experts' function.\n")
    else:
        expert_losses = data['expert_test_losses']
        expert_means = expert_losses.mean(axis=0)
        expert_sems = expert_losses.std(axis=0)/jnp.sqrt(expert_losses.shape[0])
        expert_sampled_steps = np.arange(0, epochs_per_task, sample_rate)

        for i in range(ntasks):
            # *** FIX 3: Safeguard against length mismatch ***
            y_mean_segment = expert_means[i, :]
            num_samples = len(y_mean_segment)
            x_segment = expert_sampled_steps[:num_samples] + (i * epochs_per_task)

            if len(x_segment) != len(y_mean_segment):
                print(f"Warning: Skipping expert plot for task {i+1} due to data length mismatch.")
                continue

            add_band(
                x_segment, y_mean_segment, expert_sems[i, :],
                name=f"Expert on Task {i+1}", color=colors[i % len(colors)], dash='dash'
            )

    # --- Add Task Switch Lines ---
    for sp in switch_points:
        fig.add_vline(x=sp, line_width=1.5, line_dash="dash", line_color="rgba(255, 0, 0, 0.8)")
    

    # --- Final Touches ---
    fig.update_layout(
        title_text=(
            f"Student vs. Expert Performance on {ntasks} Tasks<br>"
            f"Task Similarity (v): {similarity:.2f}, "
            f"Overlap: mean={overlap_mean:.2f}, std={overlap_std:.2f}"
        ),
        xaxis_title="Training Step", yaxis_title="Mean Squared Error",
        yaxis_type="log" if log_y else "linear",
        legend_title_text="Legend", template="plotly_white",
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        legend=dict(groupclick="togglegroup")
    )

    out_html = npz_path.with_suffix(".html")
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"Saved figure -> {out_html.resolve()}")



# Define a function with custom forward and backward behavior
@jax.custom_vjp
def mask_backward(x, mask):
    # Standard forward pass - no masking
    return x

# Define forward rule - same as standard forward pass
def mask_backward_fwd(x, mask):
    return mask_backward(x, mask), mask

# Define backward rule - apply mask to gradients
def mask_backward_bwd(mask, g):
    # Return gradient masked by the provided mask
    return (g * mask, None)

# Register the custom VJP
mask_backward.defvjp(mask_backward_fwd, mask_backward_bwd)


class StudentHead(nn.Module):
    """A single output head for the student network."""
    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            features=1,
            use_bias=False,
            name="head_out",
            kernel_init=scaled_normal_init(1.0 / jnp.sqrt(x.shape[-1])),
        )(x)

class StudentNetwork(nn.Module):
    """Student network with a shared backbone and task-specific masked heads."""
    hidden_dim: int
    head_hidden_dim: int # In this refactor, this is effectively the same as hidden_dim
    d_in: int
    def setup(self):
        # The single head layer, whose weights will be applied to different masked inputs
        self.head_layer = StudentHead(name="head")
        self.layer1 = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            name="masked_layer1",
            kernel_init=scaled_normal_init(jnp.sqrt(2 / self.d_in))
        )

    @nn.compact
    def __call__(self, x, masks: jnp.ndarray):
        # masks is a single array of shape (ntasks, d_hs)
        h = nn.relu(self.layer1(x))

        def apply_mask_and_head(mask):
            """Applies a single mask and passes the result through the head."""
            norm = jnp.sqrt(jnp.maximum(mask.sum(), 1e-8))
            # masked_h = (h * mask) / norm
            hidden = mask_backward(h, mask) / norm
            return self.head_layer(hidden).squeeze(-1)

        # vmap over the masks to get an output for each task
        masks_array = jnp.stack(masks, axis=0)
        outputs = vmap(apply_mask_and_head)(masks_array)
        return outputs

def teacher_forward(params: dict, x:jnp.ndarray)->jnp.ndarray:
    """A simple two-layer teacher network forward pass for a batch of inputs."""
    # x shape: (batch_size, d_in)
    # params['w1'] shape: (d_in, d_ht)
    h = nn.relu(x @ params['w1']) # Result shape: (batch_size, d_ht)
    # params['w2'] shape: (d_ht, 1)
    return (h @ params['w2']).squeeze(-1) # Result shape: (batch_size)


# --- Core Training & Evaluation Logic ---
@partial(jit, static_argnames=("apply_fn",))
def compute_grads(params, batch, teacher_params, task_idx, apply_fn, masks):
    """Computes loss and gradients for a single task."""
    def loss_fn(p):
        # Student produces outputs for all tasks
        all_preds = apply_fn({'params': p}, batch, masks)
        # We only care about the prediction for the current training task
        task_pred = all_preds[task_idx]
        
        task_targets = teacher_forward(teacher_params, batch)
        loss = jnp.mean((task_pred - task_targets) ** 2)
        return loss, task_pred.mean() # Also return mean pred for monitoring
        
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss, grads

@partial(jit, static_argnames=("apply_fn",))
def evaluate_metrics(params, apply_fn, test_inputs, all_teacher_targets, masks):
    """Computes test losses against all teachers."""
    all_preds = apply_fn({'params': params}, test_inputs, masks) # (ntasks, test_size)
    
    # Vmap the MSE calculation over all tasks
    def mse_loss(preds, targets):
        return jnp.mean((preds - targets)**2)

    losses = vmap(mse_loss)(all_preds, all_teacher_targets)
    return losses


@partial(jit, static_argnames=(
    "epochs_per_task", "sample_rate", "batch_size", "d_in",
    "model_apply_fn", "optimizer", "task_idx"
))
def train_single_task(
    initial_carry: tuple,
    task_idx: int,
    current_teacher_params: dict,
    all_teacher_test_targets: jnp.ndarray,
    masks: tuple,
    test_inputs: jnp.ndarray,
    # Static args
    epochs_per_task: int,
    sample_rate: int,
    batch_size: int,
    d_in: int,
    model_apply_fn,
    optimizer,
):
    """
    Runs a training loop for a single task using lax.scan.
    This function is JIT-compiled for performance.
    """
    
    # Vmapped helpers for operating on the batch of runs
    vmap_compute_grads = vmap(compute_grads, in_axes=(0, 0, 0, None, None, 0))
    vmap_evaluate_metrics = vmap(evaluate_metrics, in_axes=(0, None, None, 0, 0))
    
    @jit
    def vmapped_update(grads, opt_state, params):
        updates, new_opt_state = vmap(optimizer.update)(grads, opt_state, params)
        new_params = vmap(optax.apply_updates)(params, updates)
        return new_params, new_opt_state

    def step_fn(carry, step_idx):
        params_batch, opt_state_batch, keys_batch, masks_batch = carry

        # Generate a new batch of data for all runs
        split_fn = vmap(lambda k: random.split(k, 2))
        all_keys = split_fn(keys_batch)
        keys_batch = all_keys[:, 0, :]  # New keys for next iteration
        subkeys_batch = all_keys[:, 1, :]  # Subkeys for data generation
        batch_data = vmap(lambda k: random.normal(k, (batch_size, d_in)))(subkeys_batch)

        # Compute gradients for the current task
        train_loss_batch, grads_batch = vmap_compute_grads(
            params_batch, batch_data, current_teacher_params, task_idx, model_apply_fn, masks_batch
        )
        
        # Update parameters
        new_params_batch, new_opt_state_batch = vmapped_update(
            grads_batch, opt_state_batch, params_batch
        )

        # Periodically evaluate on the test set against ALL teachers
        def perform_eval():
            return vmap_evaluate_metrics(new_params_batch, model_apply_fn, test_inputs, all_teacher_test_targets, masks_batch)

        def skip_eval():
            num_runs, ntasks = all_teacher_test_targets.shape[:2]
            return jnp.full((num_runs, ntasks), jnp.nan)

        test_losses_batch = lax.cond(
            (step_idx % sample_rate == 0), perform_eval, skip_eval
        )
        
        new_carry = (new_params_batch, new_opt_state_batch, keys_batch, masks_batch)
        outputs = (train_loss_batch, test_losses_batch)
        return new_carry, outputs

    # Run the scan loop for the duration of this task
    final_carry, all_outputs = lax.scan(
        step_fn, initial_carry, jnp.arange(epochs_per_task)
    )
    return final_carry, all_outputs


def run_training_loop(
    key, config: dict, student_model, optimizer, test_inputs, all_teacher_params
):
    """
    Orchestrates the training process by looping through tasks.
    """
    # Unpack config
    d_in, d_h, d_hs, ntasks = config['d_in'], config['d_h'], config['d_hs'], config['ntasks']
    num_runs, epochs_per_task, sample_rate = config['num_runs'], config['epochs_per_task'], config['sample_rate']
    
    # --- Create Initial Student States and Masks for all runs ---
    mask_keys, model_keys, run_keys = random.split(key, 3)

    # vmap mask creation over the number of runs
    vmap_create_masks = vmap(create_masks, in_axes=(None, None, None, None, None, 0, None))
    masks_batch, m_overlap = vmap_create_masks(
        d_hs, config['sparsity'], config['similarity'], config['overlap'],
        config['g_type'], random.split(mask_keys, num_runs), ntasks
    ) # masks_batch is a PyTree of shape (num_runs, ntasks, ...)

    # vmap state creation over the number of runs
    student_net = StudentNetwork(hidden_dim=d_h, head_hidden_dim=d_hs, d_in=config['d_in'])
    vmap_create_state = vmap(
        lambda k, m: student_net.init(k, jnp.ones((1, d_in)), m)['params'],
        in_axes=(0, 0)
    )
    initial_params_batch = vmap_create_state(random.split(model_keys, num_runs), masks_batch)
    initial_opt_state_batch = vmap(optimizer.init)(initial_params_batch)
    
    # --- Pre-compute all test targets ---
    # `all_teacher_params` has shape (num_runs, ntasks, ...)
    # We want `all_teacher_targets` to be (num_runs, ntasks, test_size)
    # for k,v in all_teacher_params.items():
    #     jax.debug.print("all_teacher_params keys {x}", x=k)
    #     jax.debug.print("all teacher shapes: {x}", x=v.shape)
    # jax.debug.print("test_inputs shape: {x}", x=test_inputs.shape)
    vmap_teacher_forward = vmap(vmap(teacher_forward, in_axes=(0, None)), in_axes=(0, None))
    all_teacher_test_targets = vmap_teacher_forward(all_teacher_params, test_inputs)

    # --- Sequentially train on each task ---
    current_carry = (
        initial_params_batch,
        initial_opt_state_batch,
        random.split(run_keys, num_runs),
        masks_batch
    )
    
    collected_train_losses = []
    collected_test_losses = []
    
    print(f"Starting training loop for {ntasks} tasks...")
    for i in range(ntasks):
        print(f"  Training on Task {i+1}/{ntasks}...")
        with Timer(f"  Task {i+1} duration"):
            # Select the teacher parameters for the current task
            current_teacher_params_batch = jax.tree_util.tree_map(lambda x: x[:, i], all_teacher_params)

            current_carry, (train_losses, test_losses) = train_single_task(
                initial_carry=current_carry,
                task_idx=i,
                current_teacher_params=current_teacher_params_batch,
                all_teacher_test_targets=all_teacher_test_targets,
                masks=masks_batch,
                test_inputs=test_inputs,
                # Static args from config
                epochs_per_task=epochs_per_task,
                sample_rate=sample_rate,
                batch_size=config['batch_size'],
                d_in=d_in,
                model_apply_fn=student_model.apply,
                optimizer=optimizer,
            )
            collected_train_losses.append(train_losses)
            collected_test_losses.append(test_losses)

    # --- Aggregate and Sample Results ---
    # Concatenate results from all tasks
    full_train_loss = jnp.concatenate(collected_train_losses, axis=0) # (total_epochs, num_runs)
    full_test_losses = jnp.concatenate(collected_test_losses, axis=0) # (total_epochs, num_runs, ntasks)

    # Sample the results at the specified rate
    eval_indices = jnp.arange(0, ntasks * epochs_per_task, sample_rate)
    sampled_train_loss = full_train_loss[eval_indices]
    sampled_test_losses = full_test_losses[eval_indices]
    
    return sampled_train_loss, sampled_test_losses, masks_batch[0], m_overlap # Return one sample of masks for inspection


@partial(jit, static_argnames=('K',))
def get_C(v: int, K: int) -> jnp.array:
    one_matrix = jnp.ones((K,K))
    ident = jnp.eye(K)
    alpha = jnp.sqrt(1-v)
    beta = (-jnp.sqrt(1-v)+jnp.sqrt((1-v)+K*v))/(K)
    return alpha*ident + beta*one_matrix

@jax.jit
def compute_weight_vectors(C:jnp.array, basis_vectors: jnp.array)->jnp.array:
    # basis_vectors shape: (d_ht*d_in, K)
    # C shape: (K, K) and symetric
    # Result shape: (d_ht*d_in, K)
    # return basis_vectors @ C # for shape (d_ht,k)
    return vmap(lambda b: b@C)(basis_vectors) # returns shape (d_ht, K)


@partial(jax.jit, static_argnames=('d_ht', 'd_in'))
def generate_single_teacher_w1(v: int, U: jnp.ndarray, d_ht: int, d_in: int) -> jnp.ndarray:
    K = U.shape[1]
    C = get_C(v, K)
    W = U @ C # should be (d_in*d_ht, K)
    return W.T.reshape((K, d_in, d_ht))

    
def get_all_teacher_weights(key: jnp.ndarray, num_runs: int, ntasks: int, d_in: int, d_ht: int, similarity: int):
    """
    Generates teacher weights for all runs and tasks based on a single simlarity.
    
    This implementation uses a shared orthonormal basis `U` for each run. For each task,
    a set of correlated vectors is generated from this basis using the task-specific
    similarity value `v`, and one vector is randomly chosen to be the teacher's `w1`.
    """
    
    run_keys = random.split(key, num_runs) # One key per run

    @jax.jit
    def create_teachers_for_one_run(run_key: jnp.ndarray):
        # --- 1. Setup for the run ---
        U_key, w2_key = random.split(run_key, 2)

        # --- 2. Create shared orthonormal basis `U` for this run ---
        # K (number of basis vectors) is set to ntasks.
        K = ntasks
        flattened_dim = d_in * d_ht
        # U contains K orthonormal vectors, each of size flattened_dim.
        U = random.orthogonal(U_key, n=flattened_dim, m=K) # Shape: (d_in * d_ht, K)
        
        # generate similar task weight w_1
        all_w1s = generate_single_teacher_w1(similarity, U, d_ht, d_in)

        # --- 4. Generate a single shared `w2` for this run ---
        w2 = random.normal(w2_key, (d_ht, 1)) # Shape: (d_ht, 1)
        # we don't need to normalize we could if we wanted to
        # w2 /= jnp.linalg.norm(w2, axis=0, keepdims=True) + 1e-8 # Normalize 

        # --- 5. Combine and return ---
        # We stack w2 to match the number of tasks for consistent tree structure.
        all_w2s = jnp.stack([w2] * ntasks, axis=0) # Shape: (ntasks, d_ht, 1)
        return {'w1': all_w1s, 'w2': all_w2s}

    # Vmap the entire run-generation process over the number of runs.
    all_params = vmap(create_teachers_for_one_run)(run_keys) # Final shape e.g. w1: (num_runs, ntasks, d_in, d_ht)
    # jax.debug.breakpoint()
    return all_params


def setup_and_run_experiment(config):
    """Sets up and runs a full experiment for a given configuration."""
    key = random.PRNGKey(config['seed'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print(f"Running experiment with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    
    # --- Create Model and Optimizer ---
    student_model = StudentNetwork(hidden_dim=config['d_h'], head_hidden_dim=config['d_hs'], d_in=config['d_in'])
    optimizer = optax.sgd(config['lr'])
    expert_optimizer = optax.sgd(config['lr'])

    # --- Create Teacher Weights and Test Data ---
    teacher_key, test_key, run_key = random.split(key, 3)
    
    all_teacher_params = get_all_teacher_weights(
        teacher_key, config['num_runs'], config['ntasks'],
        config['d_in'], config['d_ht'], jnp.array(config['similarity'])
    )

    test_inputs = random.normal(test_key, (config['test_size'], config['d_in']))

    # --- Run Student Training ---
    with Timer("Total student training and evaluation"):
        train_loss, test_losses, masks, overlaps = run_training_loop(
            run_key, config, student_model, optimizer, test_inputs, all_teacher_params
        )

    # --- Run Expert Training ---
    expert_key = random.fold_in(run_key, 42) # Derived key for reproducibility
    expert_test_losses = train_experts(
        expert_key, config, expert_optimizer, test_inputs, all_teacher_params
    )

    # --- Save Results ---
    total_epochs = config['ntasks'] * config['epochs_per_task']
    epochs_array = np.arange(0, total_epochs, config['sample_rate'])
    switch_points = [i * config['epochs_per_task'] for i in range(1, config['ntasks'])]

    # Use a dictionary that can be loaded with allow_pickle=True
    config_to_save = {k: v for k, v in config.items() if isinstance(v, (int, float, str, list, tuple))}

    results = {
        "train_loss": np.array(train_loss),
        "test_losses": np.array(test_losses),
        "epochs": epochs_array,
        "switch_points": switch_points,
        "expert_test_losses": np.array(expert_test_losses),
        "masks": {f"mask_{i}": mask for i, mask in enumerate(masks)},
        "overlaps": overlaps,
        **config_to_save
    }
    
    filename = output_dir / f"ntasks_{config['ntasks']}_sim_{config['similarity']}.npz"
    np.savez(filename, **results)
    print(f"Results saved to {filename}")

    # --- Plot ---
    plot_losses(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Generalized Student-Teacher Experiment")
    # parser.add_argument("--d_hs", type=int, default=1000, help="Student hidden dimension")
    # parser.add_argument("--d_ht", type=int, default=200, help="Teacher hidden dimension")
    parser.add_argument("--d_in", type=int, default=800, help="Input dimension")
    parser.add_argument("--ntasks", type=int, default=2, help="Number of tasks to train on sequentially")
    parser.add_argument("--num_epochs", type=int, default=50_000, help="Total training steps for each task")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    # parser.add_argument("--sparsity", type=float, default=0.5, help="Sparsity level for masks")
    parser.add_argument("--g_type", type=str, default="overlap", help="Mask generation type ('random', 'determ', 'overlap')")
    parser.add_argument("--overlap", type=float, default=0.5, help="Shared units between task masks")
    parser.add_argument("--num_runs", type=int, default=15, help="Number of independent runs to average over")
    parser.add_argument("--v", type=float, default=0.5, help='Simimlarity between teacher weights for all tasks')
    parser.add_argument("--path", type=str, default="./results/ntasks_grad_mask_demo/overlap_search/", help="Parent directory for output")

    args = parser.parse_args()
    density = 1/args.ntasks
    sparsity = 1 - density # assures that we can have non-overlapping masks
    similarity = args.v
    d_ht = 200 # we want our network to be capable of perfectly learning the tasks w/ zero overlap
    d_hs = 200 * args.ntasks
    # --- Build Configuration ---
    epochs_per_task = args.num_epochs 
    total_epochs = args.num_epochs *  args.ntasks
    # sim_str_for_path = "_".join([f"{s:.2f}" for s in similarities]).replace('.', 'p')
    output_dir_str = (
        f"{args.path}/ntasks_{args.ntasks}_sparsity_{sparsity:.2f}"
        f"_overlap_{args.overlap:.2f}_gtype_{args.g_type}_v_{args.v}/"
    )


    config = {
        "d_in": args.d_in,
        "d_h": d_hs,  # Shared and masked layers have same dimension
        "d_hs": d_hs,
        "d_ht": d_ht,
        "ntasks": args.ntasks,
        "num_runs": args.num_runs,
        "lr": args.lr,
        "total_epochs": total_epochs,
        "epochs_per_task": epochs_per_task,
        "sample_rate": 1_000,
        "batch_size": 256,
        "test_size": 10_000,
        "sparsity": sparsity, # Enough capacity for separate subnetworks
        "similarity": args.v,
        "overlap": args.overlap,
        "g_type": args.g_type,
        "seed": 42,
        "output_dir": output_dir_str,
    }


    # Execute the experiment
    setup_and_run_experiment(config)
    print(f"\n--- Experiment finished ---\nResults saved in: {output_dir_str}")
