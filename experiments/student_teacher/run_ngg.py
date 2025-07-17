import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from loss_plotter import save_param_hist

import single_expert as expert

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
    train_loss = data["train_loss"]                      # (num_samples, num_runs)
    test_loss  = data[f"test_loss{head}"]                # (num_samples, num_runs)

    # Compute means and standard deviations
    train_loss_mean = train_loss.mean(axis=1)
    train_loss_std = train_loss.std(axis=1)
    test_loss_mean = test_loss.mean(axis=1)
    test_loss_std = test_loss.std(axis=1)

    fig, ax = plt.subplots()
    
    # Plot train loss with shaded region
    ax.plot(epochs, train_loss_mean, label="train loss")
    ax.fill_between(epochs, 
                   train_loss_mean - train_loss_std, 
                   train_loss_mean + train_loss_std,
                   alpha=0.3)
    
    # Plot test loss with shaded region
    test_label = f"test loss (head {head})"
    ax.plot(epochs, test_loss_mean, label=test_label)
    ax.fill_between(epochs, 
                   test_loss_mean - test_loss_std, 
                   test_loss_mean + test_loss_std,
                   alpha=0.3)

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
    plt.close()
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
    noise_scale: float = 1.0

    def setup(self):
        self.head_layer = StudentHead(name="head1")
                # Separate layers with distinct parameters
        self.layer1 = nn.Dense(self.hidden_dim, use_bias=False,
                              name="masked_layer1",
                              kernel_init=scaled_normal_init(jnp.sqrt(2 / self.hidden_dim)))
        

    @nn.compact
    def __call__(self, x, masks, test: bool):
        x = self.layer1(x)
        x = nn.relu(x)
        eps = 1e-8

        if not test:
            noise_rng = self.make_rng('noise')
            norm_s1 = jnp.sqrt(jnp.maximum(masks[0].sum(), eps))
            norm_s2 = jnp.sqrt(jnp.maximum(masks[1].sum(), eps))    
            hidden_s1 = x*masks[0] 
            noise1 = self.noise_scale * random.normal(noise_rng, x.shape) * (1-masks[0])
            hidden_s1 = (hidden_s1 + noise1) / norm_s1 # normalize based on n active units
            noise2 = self.noise_scale * random.normal(noise_rng, x.shape) * (1-masks[1])
            hidden_s2 = x*masks[1]  
            hidden_s2 = (hidden_s1 + noise2)/norm_s1 # normalize based on n active units
        else:
            norm = jnp.sqrt(x.shape[-1])
            hidden_s1 = x/norm
            hidden_s2 = hidden_s1

        s1_out = self.head_layer(hidden_s1)
        s2_out = self.head_layer(hidden_s2) 
        return s1_out, s2_out

def create_initial_state_parts(rng_dict, mask_pair,  *, optimizer, sample_input, d_hs, d_h):
    # rng is the *model* key now – no split needed
    params = StudentNetwork(hidden_dim=d_h,
                            head_hidden_dim=d_hs,
                            ).init(
                rng_dict, sample_input, mask_pair, test=False
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
@partial(jit, static_argnums=(5, 6,)) # head_idx and apply_fn are static
def compute_grads(params, batch, noise_key, teacher_w1, teacher_w2, head_idx, apply_fn, masks):
    """Computes loss and gradients for a single run's state."""
    def loss_fn(p):
        pred1, pred2 = apply_fn({'params': p}, batch, masks, test=False,
                                rngs={'noise': noise_key})
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
    pred1, pred2 = apply_fn({'params': params}, test_inputs, masks, test=True)
    loss1 = jnp.mean((pred1 - t1_targets)**2)
    loss2 = jnp.mean((pred2 - t2_targets)**2)
    return loss1, loss2


# -------------------------
# Vectorized Main Training Loop (using lax.scan)
# -------------------------
# Make static args explicit for clarity
@partial(jit, static_argnames=("switch_point", "sample_rate",
                               "d_in", "batch_size", "model_apply_fn", "optimizer", 
                               "num_epochs", "sparsity", "overlap", "g_type", "d_hs", "d_h", "num_runs", "lr")) # , "d_out"
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
    overlap,
    similarity,
    g_type,
    d_hs,
    d_h,
    num_runs,
    lr,
    # d_out
    ):
    # create num_runs keys for masks and models, separate than batch keys (initial_keys_batch)
    mask_master_key, model_master_key, noise_master_key = jax.random.split(master_key, 3)
    mask_keys = jax.random.split(mask_master_key, num_runs)  # -> (num_runs, 2) array
    model_keys = jax.random.split(model_master_key, num_runs)
    noise_keys = jax.random.split(noise_master_key, num_runs)
    # then vmap create_masks over that:
    masks_batch = vmap(
        create_masks,
        in_axes=(None, None, None, None, None, 0),
    )(
        d_hs, sparsity, similarity, overlap, g_type, mask_keys
    )
    mask1_batch, mask2_batch, overlap_batch = masks_batch 

    if g_type in ['determ','overlap']:
        mask1_batch, mask2_batch = mask1_batch.squeeze(axis=1), mask2_batch.squeeze(axis=1)  
    
    vmap_create_state = vmap(
                            partial(create_initial_state_parts,
                                    optimizer=optimizer,
                                    sample_input=jnp.ones((1, d_in)),
                                    d_hs=d_hs, d_h=d_h),
                            in_axes=({'params':0, 'noise': 0}, (0, 0))        # model_key, (mask1, mask2)
                            )
    initial_params_batch, initial_opt_state_batch = vmap_create_state({'params':model_keys,'noise': noise_keys}, (mask1_batch, mask2_batch))
    initial_params_np = initial_params_batch

    @jit
    def vmapped_optimizer_update(grads_batch, opt_state_batch, params_batch):
        updates, new_opt_state = vmap(optimizer.update)(grads_batch, opt_state_batch, params_batch)
        new_params = vmap(optax.apply_updates)(params_batch, updates)
        return new_params, new_opt_state

    vmap_compute_grads = vmap(compute_grads, in_axes=(0, 0, 0, 0, 0, None, None, (0,0)))
    vmap_eval = vmap(evaluate_metrics, in_axes=(0, None, None, 0, 0, (0, 0)))

    # helper used in both step_fn1 / step_fn2
    def _next_batch(keys_batch):
        # keys_batch: (R, 2)  ‑‑ R == num_runs
        split_keys = jax.vmap(lambda k: random.split(k, 3))(keys_batch)   # (R, 3, 2) # add one for noise
        new_keys = split_keys[:, 0, :]   # carry forward
        subkeys  = split_keys[:, 1, :]   # use for this minibatch
        noise_keys = split_keys[:,2, :]
        # produces (R, batch_size, d_in)
        batch = vmap(lambda k: random.normal(k, (batch_size, d_in)))(subkeys)
        return new_keys, batch, noise_keys
        
    
    # --- Scan Step for Task 1 ---
    def step_fn1(carry, step_idx):
        # keys batch holds the number of repeats, 5 keys = 5 repeats
        params_batch, opt_state_batch, keys_batch = carry
        new_keys_batch, batch_data, noise_keys_batch = _next_batch(keys_batch)
        
        # 2. Compute loss and gradients (vmapped over runs)
        # Pass teacher 1 weights, head_idx 0
        loss_batch, grads_batch = vmap_compute_grads(params_batch, 
                                                     batch_data,
                                                     noise_keys_batch, 
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
            dummy_shape = (initial_params_batch['masked_layer1']['kernel'].shape[0],) # num_runs
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
        new_keys_batch, batch_data, noise_keys_batch = _next_batch(keys_batch)


        # 2. Compute loss and gradients (vmapped over runs)
        # Pass teacher 2 weights, head_idx 1
        loss_batch, grads_batch = vmap_compute_grads(params_batch, 
                                                     batch_data,
                                                     noise_keys_batch, 
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
             dummy_shape = (initial_params_batch['masked_layer1']['kernel'].shape[0],) # num_runs
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
    # dW = final_params_2['head1']['head_out']['kernel'] - initial_params_batch['head1']['head_out']['kernel']
    # drift_per_row = jnp.linalg.norm(dW, axis=1)

    # sampled arrays should have shape (num_samples, num_runs)
    return sampled_losses, sampled_test1, sampled_test2, overlap_batch, (mask1_batch, mask2_batch),  {"initial_params": initial_params_np,
                                                                                                                    "intermediate_params": intermediate_params_np,
                                                                                                                    "final_params": final_params_np} # drift_per_row,


# process param helper function
def process_params_results(batch_params_results, index):
    return_dict = dict(initial_params={'head1':batch_params_results['initial_params']['head1']['head_out']['kernel'][index], 
                                       'masked_layer1':batch_params_results['initial_params']['masked_layer1']['kernel'][index],
                                    #    'masked_layer2':batch_params_results['initial_params']['masked_layer2']['kernel'][index]
                                       },
                       intermediate_params={'head1':batch_params_results['intermediate_params']['head1']['head_out']['kernel'][index], 
                                            'masked_layer1':batch_params_results['intermediate_params']['masked_layer1']['kernel'][index],
                                            # 'masked_layer2':batch_params_results['intermediate_params']['masked_layer2']['kernel'][index]
                                            },
                       final_params={'head1': batch_params_results['final_params']['head1']['head_out']['kernel'][index], 
                                     'masked_layer1':batch_params_results['final_params']['masked_layer1']['kernel'][index],
                                    #  'masked_layer2':batch_params_results['final_params']['masked_layer2']['kernel'][index]
                                     })
    return return_dict

# -------------------------
# Main Execution Block
# -------------------------
if __name__ == "__main__":
    '''example function call from commandline:
    python ~/mtrl/experiments/student_teacher/sparsity.py d_hs sparsity g_type path
    '''
    # default_d_hs = 400
    # default_d_ht = 200 # half the size of the student hidden dim
    parser = argparse.ArgumentParser(description="Run StudentTeacher Single Layer Experiment")
    parser.add_argument("--d_hs", type=int, default=400, help="full size of student hidden layer (w/ no sparsity)")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Sparsity level")
    parser.add_argument("--g_type", type=str, default="overlap", help="Mask generation type, one of ('random','determ','overlap')")
    parser.add_argument("--overlap", type=float, default=0.0, help="amount of shared units between two tasks")
    parser.add_argument("--d_ht", type=int, default=200, help="Hidden size for teacher nets.")
    parser.add_argument("--path", type=str, default="./loss_data/single_layer_ngg_equalCap", help="Output directory path")
    args = parser.parse_args()

    d_hs = args.d_hs
    d_ht = args.d_ht
    sparsity = args.sparsity
    g_type = args.g_type
    parent_path = args.path
    overlap = args.overlap

    avail_gpus = jax.devices()
    print(jax.devices())
    # --- Configuration ---
    d_in = 800
    num_epochs = 500_000 # Total steps (will be split per task)
    switch_point = int(num_epochs/2)
    lr = 0.1 # lr is 1 for d_in=10_000, 0.1 for d_in=1_000 ### but 1 breaks
    sample_rate = 10_000 # Sample every N steps
    v_values = np.linspace(0, 1, 11)
    num_runs = 10
    test_size = 10000
    batch_size = 200 # make larger than input size for better generalization
    d_h = d_hs # make them equal # now there is no shared hidden layer
    expert_dh = (d_h, d_h)
    d_out = 1
    output_dir = f"{parent_path}/d_ht_{d_ht}_d_hs_{d_hs}_sparsity_{sparsity:.2f}_g_type_{g_type}_lr_{lr}_overlap_{overlap:.2f}/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Args: {args}")

    # Get available devices
    devices = jax.local_devices()
    num_devices = len(devices)

    with Timer(print_time=True, show_memory=False):

        # --- Master key ---
        script_master_key = random.PRNGKey(42)

        # --- Teacher 1 ---
        t1_key_base, t_w2_key_base, script_master_key = random.split(script_master_key, 3)
        # t1_w1_key, t1_w2_key = random.split(t1_key)
        t1_keys = random.split(t1_key_base, num_runs)
        t_w2_batch = random.normal(t_w2_key_base, (num_runs, d_ht, d_out))
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

        test_inputs = jax.device_put(random.normal(test_key_base, (test_size, d_in)),
                                     jax.local_devices()[0]  # Put on first device
                                    )
        # Precompute teacher targets for test_inputs
        t1_test_targets = vmap(teacher_forward, in_axes=(None,0,0))(test_inputs, t1_w1_batch, t_w2_batch)

        # # --- Model and Optimizer (once) ---
        student_model = StudentNetwork(hidden_dim=d_h, head_hidden_dim=d_hs)
        model_apply_fn = student_model.apply
        # optimizer = optax.sgd(lr) # Create optimizer instance
        optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=lr)

        print(f"Using {g_type}:\nSparsity: {sparsity:.2f}%\n")

        def run_one_v(v, key_for_v):
            model_master_key, data_key, expert_key = jax.random.split(key_for_v, 3)
            run_keys = jax.random.split(data_key, num_runs)
            # Create teacher 2 (only needs one instance per v)
            t2_w1_batch = v*t1_w1_batch + jnp.sqrt(1-v**2)*ortho_t1_w1_batch
            t2_w1_batch /= jnp.linalg.norm(t2_w1_batch, axis=1, keepdims=True) + 1e-8
            teacher_similarity = jnp.mean(jnp.sum(t1_w1_batch * t2_w1_batch, axis=1))
            t2_test_targets = vmap(teacher_forward, in_axes=(None,0,0))(test_inputs, t2_w1_batch, t_w2_batch)

            # --- Execute the vectorized training ---
            results = vectorized_train_for_v(
                run_keys, # Pass the num_run keys for generating data inside scan
                model_master_key,
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
                overlap,
                v,
                g_type,
                d_hs,
                d_h,
                num_runs,
                lr,
            )
            
            print(f'Training Expert')
            expert_model = expert.ExpertNetwork(features=expert_dh, head_hidden_dim=d_hs)
            expert_run_keys = jax.random.split(expert_key, num_runs)
            test_inputs_batch = jnp.repeat(test_inputs[None, ...], num_runs, axis=0)  # Shape: (num_runs, 10000, 800)
            _, expert_test_losses = expert.vectorized_train_single_task(
                                                                        expert_run_keys,
                                                                        t2_w1_batch,
                                                                        t_w2_batch,
                                                                        sample_rate,
                                                                        d_in,
                                                                        expert_model.apply,
                                                                        optimizer,
                                                                        test_inputs_batch,
                                                                        switch_point,  # Train for the same duration as the student on Task 2
                                                                        batch_size,
                                                                        d_hs,
                                                                        expert_dh,
                                                                        )
            final_expert_loss_t2 = expert_test_losses[-1]

            return (v, *results, teacher_similarity, final_expert_loss_t2)

        
        num_devices = jax.local_device_count()
        keys_for_v = jax.random.split(script_master_key, len(v_values))
        pmapped_run_one_v = jax.pmap(run_one_v, in_axes=(0,0))

        for start_idx in range(0, len(v_values), num_devices):
            end_idx = min(start_idx + num_devices, len(v_values))
            batch_v = v_values[start_idx:end_idx]
            batch_keys = keys_for_v[start_idx:end_idx]

            # pad batch if smaller than num_devices
            if len(batch_v) < num_devices:
                pad_size = num_devices - len(batch_v)
                batch_v = np.concatenate([batch_v, np.zeros(pad_size)])
                dummy_keys = jax.random.split(jax.random.PRNGKey(0), pad_size)
                batch_keys = np.concatenate([batch_keys, dummy_keys], axis=0)

            batch_v = jnp.array(batch_v)
            batch_keys = jnp.array(batch_keys)
            batch_results = pmapped_run_one_v(jax.device_put(batch_v, devices[0]),
                                        jax.device_put(batch_keys, devices[0])
                                            )   
            # set_trace()
            # process results
            for i in range(len(batch_v) - (0 if end_idx <= len(v_values) else pad_size)):
                v = batch_results[0][i]
                sampled_losses = batch_results[1][i]
                sampled_test1 = batch_results[2][i]
                sampled_test2 = batch_results[3][i]
                overlap_outs = batch_results[4][i]
                masks_tuple = (batch_results[5][0][i], batch_results[5][1][i])
                params_dict = process_params_results(batch_results[6], i)
                teacher_similarity = batch_results[7][i]
                final_expert_loss = batch_results[8][i]
                
                print(f"Similarity between t1_w1 and t2_w1: {teacher_similarity}")
                # --- Prepare results for saving ---
                num_samples = sampled_losses.shape[0]
                epochs_array = np.arange(0, num_epochs, sample_rate) # Correct epochs

                final_results_for_v = {
                    "train_loss": np.array(sampled_losses),
                    "test_loss1": np.array(sampled_test1),
                    "test_loss2": np.array(sampled_test2),
                    "expert_loss": np.array(final_expert_loss),
                    "epochs": epochs_array[:num_samples], # Ensure epochs match samples dim
                    "num_epochs": num_epochs,
                    "overlap_output": overlap_outs,
                    "overlap": overlap,
                    "switch_point": switch_point,
                    "lr": lr,
                    "sparsity": sparsity,
                    "d_hs": d_hs,
                    "d_in": d_in,
                    "g_type": g_type,
                    "mask1_batch": np.array(masks_tuple[0]),
                    "mask2_batch": np.array(masks_tuple[1]),
                    # "drift": drift,
                    "v": teacher_similarity,
                }
                params = {k:jax.device_get(params) for k,params in params_dict.items()}
                # --- Save the aggregated results ---
                npz_path = f"g_type_{g_type}_v_{v:.2f}_spars_{sparsity:.2f}_overlap_{overlap:.2f}.npz"
                filename = os.path.join(output_dir, npz_path)
                np.savez(filename, **final_results_for_v)
                pickle_path = f"g_type_{g_type}_v_{v:.2f}_spars_{sparsity:.2f}_overlap_{overlap:.2f}_params.pkl"
                params_path = os.path.join(output_dir, pickle_path)
                with open(params_path, "wb") as f:
                    pickle.dump(params, f)
                file_tag = f"g_type_{g_type}_v_{v:.2f}_spars_{sparsity:.2f}_overlap_{overlap:.2f}"
                # save_param_hist(params, output_dir, file_tag)
                # print(f"Results Saved for v={v:.2f} (Shape e.g., train_loss: {final_results_for_v['train_loss'].shape})")
                plot_loss(filename, head=1)
                plot_loss(filename, head=2)

        print(f"\n--- All experiments finished ---\nSaved at {output_dir}\nWith hyper-params: {args}")


