import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
import optax
from flax import linen as nn
import flax
from flax.training import train_state
import numpy as np
from functools import partial
from jax.scipy.special import erf
from timer_class import Timer
import sys
from make_masks import create_masks
from ipdb import set_trace
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from loss_plotter import save_param_hist

# Optional: Explicit device selection
devices = jax.devices('gpu')
if len(devices) >= 3:
    jax.config.update('jax_default_device', devices[2])

avail_gpus = jax.devices()
print("Available devices:", avail_gpus) 

def plot_loss(npz_path: str | Path,
              head: int = 1,
              log_y: bool = True,
              dpi: int = 150) -> tuple:
    """
    Parameters
    ----------
    npz_path : str | Path
        Path to the results file produced by sparsity.py.
    head : {1, 2}
        Which test‑head loss to display (test_loss1 or test_loss2).
    log_y : bool
        Use a log‑scaled y‑axis if True.
    dpi : int
        Resolution of the saved PNG.

    Returns
    -------
    fig, ax : Matplotlib figure and axis objects (for further tweaking).
    """
    npz_path = Path(npz_path)
    data     = np.load(npz_path)

    epochs     = data["epochs"]                          # (num_samples,)
    train_loss = data["train_loss"].mean(axis=1)         # (num_samples,)
    test_loss  = data[f"test_loss{head}"].mean(axis=1)   # (num_samples,)

    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, label="train loss")
    ax.plot(epochs, test_loss,  label=f"test loss (head {head})")

    ax.set_xlabel("training step")
    ax.set_ylabel("MSE")
    if log_y:
        ax.set_yscale("log")
    ax.set_title(npz_path.stem)
    ax.legend()

    # --- automatic file name -------------------------------------------------
    out_png = npz_path.with_name(f"{npz_path.stem}_head{head}_loss.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure → {out_png.resolve()}")

    return fig, ax

# -------------------------
# Model/Helper Definitions (keep as before)
# -------------------------
def scaled_error(x):
    return erf(x / jnp.sqrt(2.0))

def scaled_normal_init(scale: float):
    def _init(key, shape, dtype=jnp.float32):
        return scale * jax.random.normal(key, shape, dtype)
    return _init


class StudentHead(nn.Module):
    @nn.compact
    def __call__(self, x):
        out = nn.Dense(
            1,
            use_bias=False,
            name="head_out",
            kernel_init=scaled_normal_init(1 / jnp.sqrt(x.shape[-1])),  
            # kernel_init=scaled_normal_init(0.001),
        )(x)
        return out

class StudentNetwork(nn.Module):
    '''
    hidden_dim: shared representation
    head_hidden_dim: sparse representation, linearly multiplied with unique student heads.
    masks: binary vectors to turn off representation units
    '''
    hidden_dim: int
    head_hidden_dim: int

    def setup(self):
        self.head_layer = StudentHead(name="head1")

    @nn.compact
    def __call__(self, x, masks):
        # Backbone layer with scaled init
        # remove shared layer
        # x =  nn.Dense(
        #     self.hidden_dim,
        #     use_bias=False,
        #     name="backbone_dense",
        #     kernel_init=scaled_normal_init(1),
        #     )(x) # / jnp.sqrt(x.shape[-1])  # Keep normalization as-is
        # # x = scaled_error(x)
        
        x = nn.Dense(self.head_hidden_dim, use_bias=False,
                     name="masked_layer",
                     kernel_init=scaled_normal_init(jnp.sqrt(2 / x.shape[-1]))
                    # kernel_init=scaled_normal_init(0.001),
                     )(x)
        x = nn.relu(x)
        hidden_s1 = x*masks[0] 
        hidden_s1 /= jnp.sqrt(masks[0].sum()) # normalize based on n active units
        hidden_s2 = x*masks[1]  
        hidden_s2 /= jnp.sqrt(masks[1].sum()) # normalize based on n active units
        s1_out = self.head_layer(hidden_s1)
        s2_out = self.head_layer(hidden_s2) # share the final head weight, single output because of single regression.
        return s1_out, s2_out

def create_initial_state_parts(rng, mask_pair,  *, optimizer, sample_input, d_hs, d_h):
    # rng is the *model* key now – no split needed
    params = StudentNetwork(hidden_dim=d_h,
                            head_hidden_dim=d_hs).init(
                rng, sample_input, mask_pair
             )['params']
    return params, optimizer.init(params)

@jit
def teacher_forward(x, w1, w2):
    # Ensure d_in is available or shape is correctly inferred
    h = jnp.dot(x, w1) # / jnp.sqrt(x.shape[-1])
    # return jnp.dot(scaled_error(h), w2)
    return jnp.dot(nn.relu(h), w2)

# -------------------------
# Vectorized Training Step Components
# -------------------------

# Note: train_step now focuses on grads, update happens separately
@partial(jit, static_argnums=(4 ,5,)) # head_idx and apply_fn are static
def compute_grads(params, batch, teacher_w1, teacher_w2, head_idx, apply_fn, masks):
    """Computes loss and gradients for a single run's state."""
    l2_reg = 0.01
    def loss_fn(p):
        pred1, pred2 = apply_fn({'params': p}, batch, masks)
        # pred1, pred2 = jax.checkpoint(apply_fn)({'params': p}, batch)  # Rematerialization
        pred = pred1 if head_idx == 0 else pred2
        targets = teacher_forward(batch, teacher_w1, teacher_w2)
        loss = jnp.mean((pred - targets) ** 2)
        return loss
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

# Evaluation step (similar, but uses vmap internally for test data)
@partial(jit, static_argnums=(1,)) # apply_fn is static
def evaluate_metrics(params, apply_fn, test_inputs, t1_targets, t2_targets, masks):
    """Computes test losses for a single run's parameters."""
    pred1, pred2 = apply_fn({'params': params}, test_inputs, masks)
    loss1 = jnp.mean((pred1 - t1_targets)**2)
    loss2 = jnp.mean((pred2 - t2_targets)**2)
    return loss1, loss2

# -------------------------
# Vectorized Main Training Loop (using lax.scan)
# -------------------------
# Make static args explicit for clarity
@partial(jit, static_argnames=("switch_point", "sample_rate",
                               "d_in", "batch_size", "model_apply_fn", "optimizer", 
                               "num_epochs", "sparsity", "g_type", "d_hs", "d_h", "num_runs", "d_out"))
def vectorized_train_for_v(
    # initial_params_batch, # Shape (num_runs, ...) PyTree
    initial_keys_batch, # Shape (num_runs, 2)
    master_key,
    t1_w1, t1_w2, 
    t2_w1, t2_w2, 
    switch_point, # static
    d_in, # static
    sample_rate, # static
    model_apply_fn, # Pass the model's apply function
    optimizer, # Pass the optax optimizer instance
    test_inputs, # static
    t1_test_targets, 
    t2_test_targets,  
    num_epochs, # static
    batch_size, # static
    # for mask and net creation
    sparsity,
    similarity,
    g_type,
    d_hs,
    d_h,
    num_runs,
    d_out
    ):
    # create num_runs keys for masks and models, separate than batch keys (initial_keys_batch)
    mask_master_key, model_master_key = jax.random.split(master_key)
    mask_keys = jax.random.split(mask_master_key, num_runs)  # -> (num_runs, 2) array
    model_keys = jax.random.split(model_master_key, num_runs)

    # then vmap create_masks over that:
    masks_batch = vmap(
        create_masks,
        in_axes=(None, None, None, None, 0),
    )(
        d_hs, sparsity, similarity, g_type, mask_keys
    )
    mask1_batch, mask2_batch, overlap_batch = masks_batch 
    # temporarily setting masks to ones for testing convergence
    # mask1_batch, mask2_batch = (jnp.ones_like(mask1_batch), jnp.ones_like(mask2_batch))
    if g_type == 'determ':
        mask1_batch, mask2_batch = mask1_batch.squeeze(axis=1), mask2_batch.squeeze(axis=1)  
    # # Generate initial states inside using num_runs keys for model, model_keys
    # vmap_create_state = vmap(partial(create_initial_state_parts, optimizer=optimizer,
    #                                 sample_input=jnp.ones((1, d_in)),
    #                                 sparsity=sparsity, g_type=g_type, v=v,
    #                                 d_hs=d_hs, d_h=d_h))
    
    # initial_params_batch, initial_opt_state_batch, _ = vmap_create_state(model_keys)
    vmap_create_state = vmap(
                            partial(create_initial_state_parts,
                                    optimizer=optimizer,
                                    sample_input=jnp.ones((1, d_in)),
                                    d_hs=d_hs, d_h=d_h),
                            in_axes=(0, (0, 0))        # model_key, (mask1, mask2)
                            )
    initial_params_batch, initial_opt_state_batch = vmap_create_state(model_keys, (mask1_batch, mask2_batch))
    initial_params_np = initial_params_batch

    # After initialization:
    # jax.debug.print(jnp.all(mask1_batch[0] == mask1_batch[1]))  # Should be False
    # jax.debug.print(jnp.all(initial_params_batch['backbone_dense']['kernel'][0] == 
    #             initial_params_batch['backbone_dense']['kernel'][1]))  # Should be False   
    # batch_size = 1 # Or make it a static arg if > 1
    # Vmapped function to apply optimizer update
    @jit
    def vmapped_optimizer_update(grads_batch, opt_state_batch, params_batch):
        updates, new_opt_state = vmap(optimizer.update)(grads_batch, opt_state_batch, params_batch)
        new_params = vmap(optax.apply_updates)(params_batch, updates)
        return new_params, new_opt_state

    vmap_compute_grads = vmap(compute_grads, in_axes=(0, 0, 0, 0, None, None, (0,0)))
    vmap_eval = vmap(evaluate_metrics, in_axes=(0, None, 0, 0, 0, (0, 0)))

    # helper used in both step_fn1 / step_fn2
    def _next_batch(keys_batch):
        # keys_batch: (R, 2)  ‑‑ R == num_runs
        split_keys = jax.vmap(lambda k: random.split(k, 2))(keys_batch)   # (R, 2, 2)
        new_keys = split_keys[:, 0, :]   # carry forward
        subkeys  = split_keys[:, 1, :]   # use for this minibatch
        # produces (R, batch_size, d_in)
        batch = vmap(lambda k: random.normal(k, (batch_size, d_in)))(subkeys)
        return new_keys, batch
        
    
    # --- Scan Step for Task 1 ---
    def step_fn1(carry, step_idx):
        # keys batch holds the number of repeats, 5 keys = 5 repeats
        params_batch, opt_state_batch, keys_batch = carry
        new_keys_batch, batch_data = _next_batch(keys_batch)
        
        # 2. Compute loss and gradients (vmapped over runs)
        # Pass teacher 1 weights, head_idx 0
        loss_batch, grads_batch = vmap_compute_grads(params_batch, 
                                                     batch_data, 
                                                     t1_w1, t1_w2, 0, model_apply_fn, (mask1_batch, mask2_batch))
        # 3. Apply optimizer updates (vmapped over runs)
        new_params_batch, new_opt_state_batch = vmapped_optimizer_update(grads_batch, 
                                                                        opt_state_batch, 
                                                                        params_batch
                                                                        )

        # 4. Evaluate metrics periodically (using lax.cond)
        def eval_true():
             test_loss1_batch, test_loss2_batch = vmap_eval(
                 new_params_batch, model_apply_fn, test_inputs, t1_test_targets, t2_test_targets, (mask1_batch, mask2_batch)
             )
             return test_loss1_batch, test_loss2_batch

        def eval_false():
            # Return NaNs or 0s when not evaluating
            dummy_shape = (initial_params_batch['masked_layer']['kernel'].shape[0],) # num_runs
            return jnp.full(dummy_shape, jnp.nan), jnp.full(dummy_shape, jnp.nan)

        test_loss1_batch, test_loss2_batch = lax.cond((step_idx % sample_rate == 0),
                                                       eval_true, 
                                                       eval_false
                                                    )

        new_carry = (new_params_batch, new_opt_state_batch, new_keys_batch)
        outputs = (loss_batch, test_loss1_batch, test_loss2_batch)
        return new_carry, outputs

    # --- Run Scan for Task 1 ---
    carry_init_1 = (initial_params_batch, initial_opt_state_batch, initial_keys_batch)
    (final_params_1, final_opt_state_1, final_keys_1), outputs1 = lax.scan(
        step_fn1, carry_init_1, jnp.arange(switch_point)
    )
    intermediate_params_np = final_params_1

    # --- Scan Step for Task 2 ---
    def step_fn2(carry, step_idx):
        # Use final state from task 1 as starting point
        params_batch, opt_state_batch, keys_batch = carry
        new_keys_batch, batch_data = _next_batch(keys_batch)


        # 2. Compute loss and gradients (vmapped over runs)
        # Pass teacher 2 weights, head_idx 1
        loss_batch, grads_batch = vmap_compute_grads(params_batch, 
                                                     batch_data, 
                                                     t2_w1, 
                                                     t2_w2, 
                                                     1, 
                                                     model_apply_fn,
                                                    (mask1_batch, mask2_batch))

        # 3. Apply optimizer updates (vmapped over runs)
        new_params_batch, new_opt_state_batch = vmapped_optimizer_update(
            grads_batch, opt_state_batch, params_batch
        )

        # 4. Evaluate metrics periodically (same logic as step_fn1)
        def eval_true():
             test_loss1_batch, test_loss2_batch = vmap_eval(
                 new_params_batch, model_apply_fn, test_inputs, t1_test_targets, t2_test_targets, (mask1_batch, mask2_batch)
             )
             return test_loss1_batch, test_loss2_batch
        def eval_false():
             dummy_shape = (initial_params_batch['masked_layer']['kernel'].shape[0],) # num_runs
             return jnp.full(dummy_shape, jnp.nan), jnp.full(dummy_shape, jnp.nan)
        test_loss1_batch, test_loss2_batch = lax.cond(
            (step_idx % sample_rate == 0), eval_true, eval_false
        )

        new_carry = (new_params_batch, new_opt_state_batch, new_keys_batch)
        outputs = (loss_batch, test_loss1_batch, test_loss2_batch)
        return new_carry, outputs

    # --- Run Scan for Task 2 ---
    carry_init_2 = (final_params_1, final_opt_state_1, final_keys_1) # Start from Task 1 end state
    (final_params_2, _, _), outputs2 = lax.scan(
        step_fn2, carry_init_2, jnp.arange(int(num_epochs - switch_point))
    )
    # Shape: (num_epochs_per_task, num_runs) for each
    final_params_np = final_params_2

    # --- Combine and Sample Results ---
    all_losses = jnp.concatenate([outputs1[0], outputs2[0]], axis=0)
    all_test1 = jnp.concatenate([outputs1[1], outputs2[1]], axis=0)
    all_test2 = jnp.concatenate([outputs1[2], outputs2[2]], axis=0)

    # Generate evaluation indices for both tasks
    eval_indices_task1 = jnp.arange(0, switch_point, sample_rate)
    eval_indices_task2 = jnp.arange(switch_point, num_epochs, sample_rate)
    eval_indices = jnp.concatenate([eval_indices_task1, eval_indices_task2])

    # Use these indices to sample the results
    sampled_losses = all_losses[eval_indices]
    sampled_test1 = all_test1[eval_indices]
    sampled_test2 = all_test2[eval_indices]

    # drift
    dW = final_params_2['head1']['head_out']['kernel'] - initial_params_batch['head1']['head_out']['kernel']
    drift_per_row = jnp.linalg.norm(dW, axis=1)

    # sampled arrays should have shape (num_samples, num_runs)
    return sampled_losses, sampled_test1, sampled_test2, overlap_batch, (mask1_batch, mask2_batch), drift_per_row, {"initial_params": initial_params_np,
                                                                                                                    "intermediate_params": intermediate_params_np,
                                                                                                                    "final_params": final_params_np}

# -------------------------
# Main Execution Block
# -------------------------
if __name__ == "__main__":
    '''example function call from commandline:
    python ~/mtrl/experiments/student_teacher/sparsity.py d_hs sparsity overlap g_type path
    '''
    # p = argparse.ArgumentParser(description="Run StudentTeacher Single Layer Experiment")
    # p.add_argument("d_hs", default=200, help="size of hidden dimensions")
    args = sys.argv[1:]
    arg_names = ['d_hs', 'sparsity', 'g_type', 'path']
    arg_dict = dict(zip(arg_names, args))
    d_hs = int(arg_dict.setdefault('d_hs', int(200)))
    sparsity = float(arg_dict.setdefault('sparsity', 0.5))
    g_type = arg_dict.setdefault('g_type', 'Determ').lower()
    parent_path = arg_dict.setdefault('path', "/loss_data/tests/gating_method/")
    avail_gpus = jax.devices()
    print(jax.devices())
    # --- Configuration ---
    d_in = 800
    num_epochs = 500_000 # Total steps (will be split per task)
    switch_point = int(num_epochs/2)
    lr = 0.1 # lr is 1 for d_in=10_000, 0.1 for d_in=1_000 ### but 1 breaks
    sample_rate = 10_000 # Sample every N steps
    v_values = np.linspace(0, 1, 11)
    num_runs = 15
    test_size = 50000
    batch_size = 200 # make larger than input size for better generalization
    d_h = d_hs # make them equal # now there is no shared hidden layer
    d_ht = int((d_hs*sparsity)/2) # teacher hidden layer size = sparse units divided by #tasks
    d_out = 1
    output_dir = f"{parent_path}/d_h_{d_h}_d_hs_{d_hs}_sparsity_{sparsity}_g_type_{g_type}_lr_{lr}/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"{arg_dict}")
    

    with Timer(print_time=True, show_memory=False):

        # --- Master key ---
        script_master_key = random.PRNGKey(42)

        # --- Teacher 1 ---
        t1_key_base, t2_key_base, script_master_key = random.split(script_master_key, 3)
        # t1_w1_key, t1_w2_key = random.split(t1_key)
        t1_keys = random.split(t1_key_base, num_runs)
        t_w2_key, script_master_key = random.split(script_master_key)
        t_w2_batch = random.normal(t_w2_key, (num_runs, d_ht, d_out))
        t_w2_batch /= jnp.linalg.norm(t_w2_batch, axis=1, keepdims=True) + 1e-8


        @partial(jax.jit, static_argnames=('dim'))
        def generate_t1_single_ortho(key, dim):
            t_key, r_key = random.split(key)
            t = random.normal(t_key, (dim,))
            t /= jnp.linalg.norm(t) + 1e-8
            r = random.normal(r_key, (dim,))
            proj_r_on_t = (r@t)/(t@t) * t
            ortho_to_t = r - proj_r_on_t
            ortho_to_t /= jnp.linalg.norm(ortho_to_t) + 1e-8
            return t, ortho_to_t

        @partial(jax.jit, static_argnums=(1,2))
        def gen_t1_w1(key, d_in, d_ht):
            d_ht_keys = random.split(key, d_ht)
            t, ortho_t = vmap(generate_t1_single_ortho, in_axes=(0, None))(d_ht_keys, d_in) # make d_ht number of vectors, each size of d_in
            # yields a size of d_ht, d_in, want it d_in, d_ht
            return t.T, ortho_t.T
        
        t1_w1_batch, ortho_t1_w1_batch = vmap(gen_t1_w1, in_axes=(0, None, None))(t1_keys, d_in, d_ht)
        # t1_w1_batch, ortho_t1_w1_batch = jnp.transpose(t1_w1_batch, axes=(2,0,1)), jnp.transpose(ortho_t1_w1_batch, axes=(2,0,1))

        jax.debug.print("Teacher1 identical across runs? {}", 
                jnp.allclose(t1_w1_batch[0], t1_w1_batch[1]))


        # --- Test Data (once) ---
        test_key_base, script_master_key = random.split(script_master_key, 2)
        test_keys = random.split(test_key_base, num_runs)
        test_inputs = vmap(jax.random.normal, in_axes=(0, None))(test_keys,
                                                         (test_size, d_in))

        # # --- Model and Optimizer (once) ---
        student_model = StudentNetwork(hidden_dim=d_h, head_hidden_dim=d_hs)
        model_apply_fn = student_model.apply
        optimizer = optax.sgd(lr) # Create optimizer instance

        print(f"Using {g_type}:\nSparsity: {sparsity:.2f}%\n")
        # --- Loop over similarity values ---
        for v_idx, v in enumerate(v_values):
            # --- Prepare for vectorized run ---
            print(f"\n--- Starting Similarity v={v:.2f} ({v_idx+1}/{len(v_values)}) ---")

            # --- Prepare for vectorized run ---
            v_master_key, teacher_key, script_master_key = random.split(script_master_key, 3)
            run_keys = random.split(v_master_key, num_runs)
            
            # Create teacher 2 (only needs one instance per v)
            t2_w1_batch = v*t1_w1_batch + jnp.sqrt(1-v**2)*ortho_t1_w1_batch
            t2_w1_batch /= jnp.linalg.norm(t2_w1_batch, axis=1, keepdims=True) + 1e-8
            teacher_similarity = jnp.mean(jnp.sum(t1_w1_batch * t2_w1_batch, axis=1))
            print(f"Similarity between t1_w1 and t2_w1: {teacher_similarity}")
            # Precompute teacher targets for test_inputs
            t1_test_targets = vmap(teacher_forward, in_axes=(0,0,0))(test_inputs, t1_w1_batch, t_w2_batch)
            # t1_test_targets = teacher_forward(test_inputs, t1_w1_batch, t1_w2_batch)
            t2_test_targets = vmap(teacher_forward, in_axes=(0,0,0))(test_inputs, t2_w1_batch, t_w2_batch)
            # t2_test_targets = teacher_forward(test_inputs, t2_w1_batch, t2_w1_batch)

            # --- Execute the vectorized training ---
            train_losses, test_losses1, test_losses2, overlap, masks_tuple, drift, params = vectorized_train_for_v(
                run_keys, # Pass the num_run keys for generating data inside scan
                script_master_key,
                t1_w1_batch, t_w2_batch,
                t2_w1_batch, t_w2_batch,
                switch_point,
                d_in,
                sample_rate,
                model_apply_fn,
                optimizer,      
                test_inputs,
                t1_test_targets,
                t2_test_targets,
                num_epochs,
                batch_size,
                sparsity,
                v,
                g_type,
                d_hs,
                d_h,
                num_runs,
                d_out
            )
            # train_losses etc have shape (num_samples, num_runs)

            # --- Prepare results for saving ---
            num_samples = train_losses.shape[0]
            epochs_array = np.arange(0, num_epochs, sample_rate) # Correct epochs

            final_results_for_v = {
                "train_loss": np.array(train_losses),
                "test_loss1": np.array(test_losses1),
                "test_loss2": np.array(test_losses2),
                "epochs": epochs_array[:num_samples], # Ensure epochs match samples dim
                "num_epochs": num_epochs,
                "overlap": overlap,
                "switch_point": switch_point,
                "lr": lr,
                "sparsity": sparsity,
                "d_hs": d_hs,
                "d_in": d_in,
                "g_type": g_type,
                "mask1_batch": np.array(masks_tuple[0]),
                "mask2_batch": np.array(masks_tuple[1]),
                "drift": drift,
            }
            params = {k:jax.device_get(params) for k,params in params.items()}
            # --- Save the aggregated results ---
            npz_path = f"g_type_{g_type}_v_{v:.2f}_spars_{sparsity:.2f}.npz"
            filename = os.path.join(output_dir, npz_path)
            np.savez(filename, **final_results_for_v)
            pickle_path = f"g_type_{g_type}_v_{v:.2f}_spars_{sparsity:.2f}_params.pkl"
            params_path = os.path.join(output_dir, pickle_path)
            with open(params_path, "wb") as f:
                pickle.dump(params, f)
            file_tag = f"g_type_{g_type}_v_{v:.2f}_spars_{sparsity:.2f}"
            save_param_hist(params, output_dir, file_tag)
            print(f"Results Saved for v={v:.2f} (Shape e.g., train_loss: {final_results_for_v['train_loss'].shape})")
            plot_loss(filename, head=1)
            plot_loss(filename, head=2)

        print(f"\n--- All experiments finished ---\nSaved at {output_dir}\nWith sparse values: {arg_dict}")


