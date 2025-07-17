import os
import argparse
import pickle
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from jax import jit, lax, random, vmap

from make_masks import create_masks
from timer_class import Timer

# --- Plotting ---
def plot_losses(npz_path: str | Path, log_y: bool = True, dpi: int = 150) -> None:
    """Plots training and test losses for all tasks from a results file."""
    npz_path = Path(npz_path)
    data = np.load(npz_path)

    epochs = data["epochs"]
    train_loss = data["train_loss"]          # (num_samples, num_runs)
    test_losses = data["test_losses"]        # (num_samples, num_runs, ntasks)
    switch_points = data["switch_points"]
    ntasks = test_losses.shape[-1]

    train_mean = train_loss.mean(axis=1)
    train_std = train_loss.std(axis=1)
    test_means = test_losses.mean(axis=1)
    test_stds = test_losses.std(axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot train loss
    ax.plot(epochs, train_mean, label="Train Loss", color="black")
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                    color="black", alpha=0.2)

    # Plot test losses for each task
    colors = plt.cm.viridis(np.linspace(0, 1, ntasks))
    for i in range(ntasks):
        ax.plot(epochs, test_means[:, i], label=f"Test Loss (Task {i+1})", color=colors[i])
        ax.fill_between(epochs, test_means[:, i] - test_stds[:, i],
                        test_means[:, i] + test_stds[:, i],
                        color=colors[i], alpha=0.2)
    
    # Add vertical lines at each task switch
    for sp in switch_points:
        ax.axvline(x=sp, color='r', linestyle='--', alpha=0.7)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title(npz_path.stem)
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)

    out_png = npz_path.with_suffix(".png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure -> {out_png.resolve()}")
    plt.close(fig)


# --- Model Definitions ---
def scaled_normal_init(scale: float):
    """Returns a function for scaled normal initialization."""
    def init(key, shape, dtype=jnp.float32):
        return scale * random.normal(key, shape, dtype)
    return init

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

    def setup(self):
        # The single head layer, whose weights will be applied to different masked inputs
        self.head_layer = StudentHead(name="head")
        self.layer1 = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            name="masked_layer1",
            kernel_init=scaled_normal_init(jnp.sqrt(2 / self.hidden_dim))
        )

    @nn.compact
    def __call__(self, x, masks: jnp.ndarray):
        # masks is a single array of shape (ntasks, d_hs)
        h = nn.relu(self.layer1(x))

        def apply_mask_and_head(mask):
            """Applies a single mask and passes the result through the head."""
            norm = jnp.sqrt(jnp.maximum(mask.sum(), 1e-8))
            masked_h = (h * mask) / norm
            return self.head_layer(masked_h).squeeze(-1)

        # vmap over the masks to get an output for each task
        masks_array = jnp.stack(masks, axis=0)
        outputs = vmap(apply_mask_and_head)(masks_array)
        return outputs

def teacher_forward(params, x):
    """A simple two-layer teacher network forward pass."""
    h = nn.relu(x @ params['w1'])
    return h @ params['w2']


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
        keys_batch, subkeys_batch = jnp.split(random.split(keys_batch, 2 * keys_batch.shape[0]).reshape(keys_batch.shape[0], 2, -1), 2, axis=1)
        keys_batch, subkeys_batch = keys_batch.squeeze(1), subkeys_batch.squeeze(1)
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
        d_hs, config['sparsity'], config['similarities'], config['overlap'], ## this will throw an error
        config['g_type'], random.split(mask_keys, num_runs), ntasks
    ) # masks_batch is a PyTree of shape (num_runs, ntasks, ...)

    # vmap state creation over the number of runs
    student_net = StudentNetwork(hidden_dim=d_h, head_hidden_dim=d_hs)
    vmap_create_state = vmap(
        lambda k, m: student_net.init(k, jnp.ones((1, d_in)), m)['params'],
        in_axes=(0, 0)
    )
    initial_params_batch = vmap_create_state(random.split(model_keys, num_runs), masks_batch)
    initial_opt_state_batch = vmap(optimizer.init)(initial_params_batch)
    
    # --- Pre-compute all test targets ---
    # `all_teacher_params` has shape (num_runs, ntasks, ...)
    # We want `all_teacher_targets` to be (num_runs, ntasks, test_size)
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
def generate_single_teacher_w1(key: jnp.ndarray, v: int, U: jnp.ndarray, d_ht: int, d_in: int) -> jnp.ndarray:
    K = U.shape[1]
    C = get_C(v, K)
    W = compute_weight_vectors(C, U)
    k_select = jax.random.randint(key, shape=(), minval=0, maxval=K)
    W_k = W[:, k_select]
    return W_k.reshape((d_in, d_ht))

    
def get_all_teacher_weights(key: jnp.ndarray, num_runs: int, ntasks: int, d_in: int, d_ht: int, similarities: jnp.ndarray):
    """
    Generates teacher weights for all runs and tasks based on a vector of similarities.
    
    This implementation uses a shared orthonormal basis `U` for each run. For each task,
    a set of correlated vectors is generated from this basis using the task-specific
    similarity value `v`, and one vector is randomly chosen to be the teacher's `w1`.
    """
    assert similarities.shape == (ntasks,), "Length of similarities vector must equal ntasks."
    
    run_keys = random.split(key, num_runs) # One key per run

    @jax.jit
    def create_teachers_for_one_run(run_key: jnp.ndarray, similarities_vec: jnp.ndarray):
        # --- 1. Setup for the run ---
        U_key, w2_key, task_keys_key = random.split(run_key, 3)
        task_keys = random.split(task_keys_key, ntasks) # One key per task for selection

        # --- 2. Create shared orthonormal basis `U` for this run ---
        # K (number of basis vectors) is set to ntasks.
        K = ntasks
        flattened_dim = d_in * d_ht
        # U contains K orthonormal vectors, each of size flattened_dim.
        U = random.orthogonal(U_key, n=flattened_dim, m=K) # Shape: (d_in * d_ht, K)
        
        # --- 3. Generate `w1` for each task by vmapping over similarities ---
        # We vmap the generation function over the similarities vector and task keys.
        vmap_gen_w1 = vmap(
            generate_single_teacher_w1,
            in_axes=(0, 0, None, None, None), # vmap over key and v; U is broadcast
        )
        all_w1s = vmap_gen_w1(task_keys, similarities_vec, U, d_in, d_ht) # Shape: (ntasks, d_in, d_ht)

        # --- 4. Generate a single shared `w2` for this run ---
        w2 = random.normal(w2_key, (d_ht, 1)) # Shape: (d_ht, 1)
        w2 /= jnp.linalg.norm(w2, axis=0, keepdims=True) + 1e-8 # Normalize

        # --- 5. Combine and return ---
        # We stack w2 to match the number of tasks for consistent tree structure.
        all_w2s = jnp.stack([w2] * ntasks, axis=0) # Shape: (ntasks, d_ht, 1)
        return {'w1': all_w1s, 'w2': all_w2s}

    # Vmap the entire run-generation process over the number of runs.
    # The similarities vector is the same for all runs.
    all_params = vmap(
        create_teachers_for_one_run,
        in_axes=(0, None), # vmap over run_keys; similarities_vec is broadcast
    )(run_keys, similarities) # Final shape e.g. w1: (num_runs, ntasks, d_in, d_ht)

    return all_params


def setup_and_run_experiment(config):
    """Sets up and runs a full experiment for a given configuration."""
    # --- Setup ---
    key = random.PRNGKey(config['seed'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running experiment with config:\n{config}")
    
    # --- Create Model and Optimizer ---
    student_model = StudentNetwork(hidden_dim=config['d_h'], head_hidden_dim=config['d_hs'])
    optimizer = optax.sgd(config['lr'])

    # --- Create Teacher Weights and Test Data ---
    teacher_key, test_key, run_key = random.split(key, 3)
    
    # Generate weights for all runs and tasks using the new method
    all_teacher_params = get_all_teacher_weights(
        teacher_key, config['num_runs'], config['ntasks'],
        config['d_in'], config['d_ht'], jnp.array(config['similarities']) # Pass similarities vector
    )

    # Create a single batch of test data, shared across all runs
    test_inputs = random.normal(test_key, (config['test_size'], config['d_in']))

    # --- Run Training ---
    with Timer("Total training and evaluation"):
        train_loss, test_losses, masks = run_training_loop(
            run_key, config, student_model, optimizer, test_inputs, all_teacher_params
        )

    # --- Save Results ---
    total_epochs = config['ntasks'] * config['epochs_per_task']
    epochs_array = np.arange(0, total_epochs, config['sample_rate'])
    switch_points = [i * config['epochs_per_task'] for i in range(1, config['ntasks'])]

    results = {
        "train_loss": np.array(train_loss),
        "test_losses": np.array(test_losses),
        "epochs": epochs_array,
        "switch_points": switch_points,
        "masks": {f"mask_{i}": mask for i, mask in enumerate(masks)},
        **config # Save the config for reproducibility
    }
    
    sim_str = "_".join([f"{s:.2f}" for s in config['similarities']]).replace('.', 'p')
    filename = output_dir / f"ntasks_{config['ntasks']}_sim_{sim_str}.npz"
    np.savez(filename, **results)
    print(f"Results saved to {filename}")

    # --- Plot ---
    plot_losses(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Generalized Student-Teacher Experiment")
    parser.add_argument("--d_hs", type=int, default=1000, help="Student head hidden dimension")
    parser.add_argument("--d_ht", type=int, default=100, help="Teacher hidden dimension")
    parser.add_argument("--d_in", type=int, default=800, help="Input dimension")
    parser.add_argument("--ntasks", type=int, default=3, help="Number of tasks to train on sequentially")
    parser.add_argument("--num_epochs", type=int, default=250_000, help="Total training steps across all tasks")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    # parser.add_argument("--sparsity", type=float, default=0.5, help="Sparsity level for masks")
    parser.add_argument("--g_type", type=str, default="random", help="Mask generation type ('random', 'determ', 'overlap')")
    parser.add_argument("--overlap", type=float, default=0.0, help="Shared units between task masks")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of independent runs to average over")
    parser.add_argument("--path", type=str, default="./results/n_task_demo", help="Parent directory for output")

    args = parser.parse_args()
    sparsity = 1/args.ntasks
    similarities = [i/args.ntasks for i in range(args.ntasks)]
    """    # --- Process similarities argument ---
    sim_values_str = args.similarities.split(',')
    try:
        similarities = [float(v) for v in sim_values_str]
    except ValueError:
        raise ValueError(f"Invalid format for --similarities. Expected comma-separated floats, got {args.similarities}")

    if len(similarities) == 1 and args.ntasks > 1:
        # If one value is provided, repeat it for all tasks
        similarities = similarities * args.ntasks
    elif len(similarities) != args.ntasks:
        raise ValueError(f"Number of similarity values ({len(similarities)}) must match ntasks ({args.ntasks}) or be 1.")"""

    # --- Build Configuration ---
    epochs_per_task = args.num_epochs 
    total_epochs = args.num_epochs *  args.ntasks
    # sim_str_for_path = "_".join([f"{s:.2f}" for s in similarities]).replace('.', 'p')
    output_dir_str = (
        f"{args.path}/ntasks_{args.ntasks}_sparsity_{sparsity:.2f}"
        f"_overlap_{args.overlap:.2f}_gtype_{args.g_type}/"
    )


    config = {
        "d_in": args.d_in,
        "d_h": args.d_hs,  # Shared and masked layers have same dimension
        "d_hs": args.d_hs,
        "d_ht": args.d_ht,
        "ntasks": args.ntasks,
        "num_runs": args.num_runs,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "epochs_per_task": epochs_per_task,
        "sample_rate": 10_000,
        "batch_size": 256,
        "test_size": 10_000,
        "sparsity": sparsity, # Enough capacity for separate subnetworks
        "similarities": similarities,
        "overlap": args.overlap,
        "g_type": args.g_type,
        "seed": 42,
        "output_dir": output_dir_str,
    }


    # Execute the experiment
    setup_and_run_experiment(config)
    print(f"\n--- Experiment finished ---\nResults saved in: {output_dir_str}")