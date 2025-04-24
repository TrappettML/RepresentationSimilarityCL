import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
import optax
from flax import linen as nn
import flax
from flax.training import train_state
import numpy as np
import os
from functools import partial
from jax.scipy.special import erf
from timer_class import Timer
from loss_diff_plots import generate_loss_plots
import sys
from make_masks import create_masks
from ipdb import set_trace


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
            kernel_init=scaled_normal_init(0.001),  
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

    @nn.compact
    def __call__(self, x, masks):
        # Backbone layer with scaled init
        x = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            name="backbone_dense",
            kernel_init=scaled_normal_init(0.001),
            )(x) / jnp.sqrt(x.shape[-1])  # Keep normalization as-is
        x = scaled_error(x)
        x = nn.Dense(self.head_hidden_dim, use_bias=False,
                     name="shared_head_layer",
                     kernel_init=scaled_normal_init(0.001))(x)
        
        hidden_s1 = x*masks[0]
        hidden_s2 = x*masks[1]
        s1_out = StudentHead(name="head1")(hidden_s1)
        s2_out = StudentHead(name="head2")(hidden_s2)
        return s1_out, s2_out

# We might manage params and opt_state directly for easier vmapping
# than the full TrainState object.
# def create_initial_state_parts(rng, model, optimizer, sample_input):
#     params = model.init(rng, sample_input)['params']
#     opt_state = optimizer.init(params)
#     return params, opt_state # Return optimizer too

def create_initial_state_parts(rng, optimizer, sample_input, sparsity, g_type, v, d_hs, d_h):
    mask_key, model_key = random.split(rng)
    mask1, mask2, overlap = create_masks(d_in=d_hs, d_out=1, sparsity=sparsity, v=v, m_type=g_type, key=mask_key)
    model = StudentNetwork(hidden_dim=d_h, head_hidden_dim=d_hs)
    params = model.init(model_key, sample_input, (mask1, mask2))['params']
    opt_state = optimizer.init(params)
    return params, opt_state, overlap

@jit
def teacher_forward(x, w1, w2):
    # Ensure d_in is available or shape is correctly inferred
    h = jnp.dot(x, w1) / jnp.sqrt(x.shape[-1])
    return jnp.dot(scaled_error(h), w2)

# -------------------------
# Vectorized Training Step Components
# -------------------------

# Note: train_step now focuses on grads, update happens separately
@partial(jit, static_argnums=(4 ,5,)) # head_idx and apply_fn are static
def compute_grads(params, batch, teacher_w1, teacher_w2, head_idx, apply_fn, masks):
    """Computes loss and gradients for a single run's state."""
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
    # Teacher forward should handle batches if test_inputs is batched
    # targets1 = teacher_forward(test_inputs, t1_w1, t1_w2)
    # targets2 = teacher_forward(test_inputs, t2_w1, t2_w2)
    loss1 = jnp.mean((pred1 - t1_targets)**2)
    loss2 = jnp.mean((pred2 - t2_targets)**2)
    return loss1, loss2


# -------------------------
# Vectorized Main Training Loop (using lax.scan)
# -------------------------
# Make static args explicit for clarity
@partial(jit, static_argnames=("switch_point", "sample_rate",
                               "d_in", "batch_size", "model_apply_fn", "optimizer", 
                               "num_epochs", "sparsity", "g_type", "d_hs", "d_h", "d_out"))
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
    d_out
    ):
    # create num_runs keys for masks and models, separate than batch keys (initial_keys_batch)
    mask_master_key, model_master_key = jax.random.split(master_key)
    mask_keys = jax.random.split(mask_master_key, num_runs)  # -> (num_runs, 2) array
    model_keys = jax.random.split(model_master_key, num_runs)

    # then vmap create_masks over that:
    masks_batch = vmap(
        create_masks,
        in_axes=(None, None, None, None, None, 0),
    )(
        d_hs, d_out, sparsity, similarity, g_type, mask_keys
    )
    mask1_batch, mask2_batch, overlap_batch = masks_batch   
    # Generate initial states inside using num_runs keys for model, model_keys
    vmap_create_state = vmap(partial(create_initial_state_parts, optimizer=optimizer,
                                    sample_input=jnp.ones((1, d_in)),
                                    sparsity=sparsity, g_type=g_type, v=v,
                                    d_hs=d_hs, d_h=d_h))
    
    initial_params_batch, initial_opt_state_batch, _ = vmap_create_state(model_keys)


    # batch_size = 1 # Or make it a static arg if > 1
    # Vmapped function to apply optimizer update
    @jit
    def vmapped_optimizer_update(grads_batch, opt_state_batch, params_batch):
        updates, new_opt_state = vmap(optimizer.update)(grads_batch, opt_state_batch, params_batch)
        new_params = vmap(optax.apply_updates)(params_batch, updates)
        return new_params, new_opt_state

    vmap_compute_grads = vmap(compute_grads, in_axes=(0, 0, None, None, None, None, (0,0)))
    vmap_eval = vmap(evaluate_metrics, in_axes=(0, None, None, None, None, (0, 0)))
    # --- Scan Step for Task 1 ---
    def step_fn1(carry, step_idx):
        # keys batch holds the number of repeats, 5 keys = 5 repeats
        params_batch, opt_state_batch, keys_batch = carry
        # # 1. Generate keys and batch per run
        # TODO: Adapt if batch_size > 1
        keys_split_pairs = vmap(lambda k: random.split(k, 2))(keys_batch) # Output shape (5, 2, 2)
        # Use one key from the pair for this iteration's randomness
        iter_keys = keys_split_pairs[:, 0, :]      # Shape (5, 2)
        # Use the other key from the pair to carry forward
        new_keys_batch = keys_split_pairs[:, 1, :] # Shape (5, 2)
        # batch_data = vmap(lambda k: random.normal(k, (batch_size, d_in)))(iter_keys)
        shape = (int(batch_size), int(d_in))
        batch_data = vmap(lambda k: random.normal(k, shape))(iter_keys)
        
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
            dummy_shape = (initial_params_batch['backbone_dense']['kernel'].shape[0],) # num_runs
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
    # outputs1 = (all_losses_1, all_test1_1, all_test2_1)
    # Shape: (switch_point, num_runs) for each

    # --- Scan Step for Task 2 ---
    def step_fn2(carry, step_idx):
        # Use final state from task 1 as starting point
        params_batch, opt_state_batch, keys_batch = carry

        # 1. Generate keys and batch per run
        keys_split_pairs = vmap(lambda k: random.split(k, 2))(keys_batch) # Output shape (5, 2, 2)
        # Use one key from the pair for this iteration's randomness
        iter_keys = keys_split_pairs[:, 0, :]      # Shape (5, 2)
        # Use the other key from the pair to carry forward
        new_keys_batch = keys_split_pairs[:, 1, :] # Shape (5, 2)
        # batch_data = vmap(lambda k: random.normal(k, (batch_size, d_in)))(iter_keys)
        shape = (int(batch_size), int(d_in))
        batch_data = vmap(lambda k: random.normal(k, shape))(iter_keys)


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
             dummy_shape = (initial_params_batch['backbone_dense']['kernel'].shape[0],) # num_runs
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
    # outputs2 = (all_losses_2, all_test1_2, all_test2_2)
    # Shape: (num_epochs_per_task, num_runs) for each

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

    # sampled arrays should have shape (num_samples, num_runs)
    return sampled_losses, sampled_test1, sampled_test2, overlap_batch

# -------------------------
# Main Execution Block
# -------------------------
if __name__ == "__main__":
    '''example function call from commandline:
    python ~/mtrl/experiments/student_teacher/sparsity.py d_hs sparsity overlap g_type path
    '''
    args = sys.argv[1:]
    arg_names = ['d_hs', 'sparsity', 'g_type', 'path']
    arg_dict = dict(zip(arg_names, args))
    d_hs = int(arg_dict.setdefault('d_hs', 200))
    sparsity = float(arg_dict.setdefault('sparsity', 0.5))
    g_type = arg_dict.setdefault('g_type', 'Determ').lower()
    parent_path = arg_dict.setdefault('path', "/loss_data/tests/gating_method/")
    avail_gpus = jax.devices()
    print(jax.devices())
    # --- Configuration ---
    d_in = 800
    num_epochs = 2_000_000 # Total steps (will be split per task)
    switch_point = int(num_epochs/2)
    lr = 0.3 # lr is 1 for d_in=10_000, 0.1 for d_in=1_000 ### but 1 breaks
    sample_rate = 10_000 # Sample every N steps
    v_values = np.linspace(0, 1, 11)
    num_runs = 20
    test_size = 50000
    batch_size = 200
    d_h = d_hs # make them equal
    d_out = 1
    output_dir = f".{parent_path}/d_h_{d_h}_d_hs_{d_hs}_sparsity_{sparsity}_g_type_{g_type}_lr_{lr}/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"{arg_dict}")
    

    with Timer(print_time=True, show_memory=False):

        # --- Master key ---
        script_master_key = random.PRNGKey(42)

        # --- Teacher 1 ---
        t1_key, script_master_key = random.split(script_master_key)
        t1_w1_key, t1_w2_key = random.split(t1_key)
        teacher1_w1 = random.normal(t1_w1_key, (d_in, 1))
        teacher1_w1 = teacher1_w1 / (jnp.linalg.norm(teacher1_w1, axis=0, keepdims=True) + 1e-8)
        teacher1_w2 = random.normal(t1_w2_key, (1, 1))
        teacher1_w2 = teacher1_w2 / (jnp.linalg.norm(teacher1_w2, axis=0, keepdims=True) + 1e-8)

        # --- Teacher 2 Creation Function ---
        # (Keep your create_teacher2 function, ensure it handles norms robustly)
        def create_teacher2(v, key):
            key, subkey, w2_key = random.split(key, 3)
            random_vec = random.normal(subkey, (d_in, 1))
            proj = (teacher1_w1.T @ random_vec) * teacher1_w1
            ortho = random_vec - proj
            ortho = ortho / (jnp.linalg.norm(ortho, axis=0, keepdims=True) + 1e-8)
            teacher2_w1 = v * teacher1_w1 + jnp.sqrt(jnp.maximum(1 - v**2, 0.0)) * ortho # Ensure sqrt >= 0
            teacher2_w1 = teacher2_w1 / (jnp.linalg.norm(teacher2_w1, axis=0, keepdims=True) + 1e-8)
            teacher2_w2 = random.normal(w2_key, (1, 1))
            teacher2_w2 = teacher2_w2 / (jnp.linalg.norm(teacher2_w2, axis=0, keepdims=True) + 1e-8)
            return teacher2_w1, teacher2_w2

        # --- Test Data (once) ---
        test_key, script_master_key = random.split(script_master_key)
        test_inputs = jax.random.normal(test_key, (test_size, d_in))


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
            v_master_key, script_master_key = random.split(script_master_key)
            run_keys = random.split(v_master_key, num_runs)

            # Create teacher 2 (only needs one instance per v)
            create_t2_key, _ = random.split(run_keys[0]) # Use one key just for t2 creation
            teacher2_w1, teacher2_w2 = create_teacher2(v, create_t2_key)

            # Precompute teacher targets for test_inputs
            t1_test_targets = teacher_forward(test_inputs, teacher1_w1, teacher1_w2)
            t2_test_targets = teacher_forward(test_inputs, teacher2_w1, teacher2_w2)

            # --- Execute the vectorized training ---
            train_losses, test_losses1, test_losses2, overlap = vectorized_train_for_v(
                # initial_params_batch,
                # initial_opt_state_batch,
                run_keys, # Pass the run keys for generating data inside scan
                script_master_key,
                teacher1_w1, teacher1_w2,
                teacher2_w1, teacher2_w2,
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
            }

            # --- Save the aggregated results ---
            filename = os.path.join(output_dir, f"g_type_{g_type}_v_{v:.2f}_spars_{sparsity:.2f}.npz")
            np.savez(filename, **final_results_for_v)
            print(f"Results Saved for v={v:.2f} (Shape e.g., train_loss: {final_results_for_v['train_loss'].shape})")

        print(f"\n--- All experiments finished ---\nSaved at {output_dir}\nWith sparse values: {arg_dict}")


