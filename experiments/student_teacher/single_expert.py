import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
import optax
from flax import linen as nn
import numpy as np
from functools import partial
from jax.scipy.special import erf
import sys
from typing import Sequence

# -------------------------
# Model Definitions
# -------------------------
def scaled_error(x):
    return erf(x / jnp.sqrt(2.0))

def scaled_normal_init(scale: float):
    def _init(key, shape, dtype=jnp.float32):
        return scale * jax.random.normal(key, shape, dtype)
    return _init

@jit
def teacher_forward(x, w1, w2):
    # Ensure d_in is available or shape is correctly inferred
    h = jnp.dot(x, w1) # / jnp.sqrt(x.shape[-1])
    # return jnp.dot(scaled_error(h), w2)
    return jnp.dot(nn.relu(h), w2)


class ExpertNetwork(nn.Module):
    features: Sequence[int]
    # hidden_dim: int
    # head_hidden_dim: int

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f'hidden_{i}')(x)/ jnp.sqrt(x.shape[-1])
            x = nn.relu(x)
        
        x = nn.Dense(
            1,
            use_bias=False,
            name="head_out",
            kernel_init=scaled_normal_init(jnp.sqrt(1 / x.shape[-1])),
            # kernel_init=scaled_normal_init(0.001),
        )(x) # / jnp.sqrt(x.shape[-1])
        return x

# -------------------------
# Training Utilities
# -------------------------
def create_initial_state(rng, optimizer, sample_input, features):
    model = ExpertNetwork(features=features) # , head_hidden_dim=d_hs
    params = model.init(rng, sample_input)['params']
    opt_state = optimizer.init(params)
    return params, opt_state

@partial(jit, static_argnums=(4,))
def compute_grads(params, batch, teacher_w1, teacher_w2, apply_fn):
    l2_reg = 0.01
    def loss_fn(p):
        pred = apply_fn({'params': p}, batch)
        targets = teacher_forward(batch, teacher_w1, teacher_w2)
        loss = jnp.mean((pred - targets) ** 2)
        return loss
        # mse_loss = jnp.mean((pred - targets) ** 2)

        # # L2 regularization term
        # l2_loss = sum([jnp.sum(jnp.square(param)) for param in jax.tree_util.tree_leaves(p)])
        # total_loss = mse_loss  + l2_reg * l2_loss
        # return total_loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

@partial(jit, static_argnums=(1,))
def evaluate_metrics(params, apply_fn, test_inputs, teacher_w1, teacher_w2):
    pred = apply_fn({'params': params}, test_inputs)
    targets = teacher_forward(test_inputs, teacher_w1, teacher_w2)
    return jnp.mean((pred - targets) ** 2)

# -------------------------
# Vectorized Training Loop
# -------------------------
# @partial(jit, static_argnames=("sample_rate", "d_in", "batch_size", "model_apply_fn", 
#                               "optimizer", "num_epochs", "d_hs", "d_h", "num_runs"))
@partial(jax.jit,
         static_argnums=(3,4,5,6,8,9,10))  # all static args after initial arrays
def vectorized_train_single_task(
    initial_keys_batch,
    teacher_w1,
    teacher_w2,
    sample_rate,
    d_in,
    model_apply_fn,
    optimizer,
    test_inputs,
    num_epochs,
    batch_size,
    # d_hs,
    d_h,
):  
    local_runs = initial_keys_batch.shape[0]
    vmap_create_state = vmap(partial(create_initial_state, 
                                   optimizer=optimizer,
                                   sample_input=jnp.ones((1, d_in)),
                                   features=d_h))
    initial_params_batch, initial_opt_state_batch = vmap_create_state(initial_keys_batch)

    @jit
    def vmapped_optimizer_update(grads_batch, opt_state_batch, params_batch):
        updates, new_opt_state = vmap(optimizer.update)(grads_batch, opt_state_batch, params_batch)
        new_params = vmap(optax.apply_updates)(params_batch, updates)
        return new_params, new_opt_state

    vmap_compute_grads = vmap(compute_grads, in_axes=(0, 0, 0, 0, None))
    vmap_eval = vmap(evaluate_metrics, in_axes=(0, None, 0, 0, 0))

    def step_fn(carry, step_idx):
        params_batch, opt_state_batch, keys_batch = carry
        keys_split = vmap(lambda k: random.split(k, 2))(keys_batch)
        iter_keys, new_keys = keys_split[:, 0], keys_split[:, 1]
        batch_data = vmap(lambda k: random.normal(k, (batch_size, d_in)))(iter_keys)

        loss_batch, grads_batch = vmap_compute_grads(
            params_batch, batch_data, teacher_w1, teacher_w2, model_apply_fn
        )
        new_params, new_opt_state = vmapped_optimizer_update(grads_batch, opt_state_batch, params_batch)

        test_loss_batch = lax.cond(
            step_idx % sample_rate == 0,
            lambda: vmap_eval(new_params, model_apply_fn, test_inputs, teacher_w1, teacher_w2),
            lambda: jnp.full((local_runs,), jnp.nan)
        )

        return (new_params, new_opt_state, new_keys), (loss_batch, test_loss_batch)

    carry_init = (initial_params_batch, initial_opt_state_batch, initial_keys_batch)
    (final_params, _, _), (losses, test_losses) = lax.scan(
        step_fn, carry_init, jnp.arange(num_epochs)
    )
    
    eval_indices = jnp.arange(0, num_epochs, sample_rate)
    return losses[eval_indices], test_losses[eval_indices]

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    args = sys.argv[1:]
    arg_names = ['d_hs', 'path']
    arg_dict = dict(zip(arg_names, args))
    d_hs = int(arg_dict.get('d_hs', 200))
    parent_path = arg_dict.get('path', "./loss_data/expert_test_mul_layer/")
    print(jax.devices())
    # Configuration
    d_in = 800
    d_h = (d_hs, d_hs)
    d_out = 1
    num_epochs = 250_000
    lr = 0.1
    d_ht = int((d_hs)/2) # teacher hidden layer size = sparse units divided by #tasks
    sample_rate = 10_000
    batch_size = 200
    test_size = 50000
    num_runs = 21
    v_values = np.linspace(0, 1, 11)
    n_devices = jax.local_device_count()          # e.g. 8
    assert num_runs % n_devices == 0, "num_runs must be divisible by number of GPUs"
    runs_per_device = num_runs // n_devices

    
    output_dir = f"{parent_path}/d_h_{d_h}_d_hs_{d_hs}_lr_{lr}/"
    os.makedirs(output_dir, exist_ok=True)

    # --- Master key ---
    script_master_key = random.PRNGKey(42)

    # --- Teacher 1 ---
    t1_key_base, t2_key_base, script_master_key = random.split(script_master_key, 3)
    # t1_w1_key, t1_w2_key = random.split(t1_key)
    t1_keys = random.split(t1_key_base, num_runs)
    t_w2_key, script_master_key = random.split(script_master_key)
    t_w2_batch = random.normal(t_w2_key, (num_runs, d_ht, d_out))
    t_w2_batch /= jnp.linalg.norm(t_w2_batch, axis=1, keepdims=True) + 1e-8
    test_key, script_master_key = random.split(script_master_key)

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
    # Test data
    test_inputs = random.normal(test_key, (num_runs, test_size, d_in))

    # Model & Optimizer
    student = ExpertNetwork(features=d_h) # , head_hidden_dim=d_hs
    optimizer = optax.sgd(lr)
    
    def split_for_devices(x):
        """Reshape leading axis [num_runs, …] → [n_devices, runs_per_device, …]"""
        return x.reshape((n_devices, runs_per_device) + x.shape[1:])

    # teacher weights / test inputs
    t1_w1_dev = jax.device_put_sharded(list(split_for_devices(t1_w1_batch)), jax.local_devices())
    t1_w2_dev = jax.device_put_sharded(list(split_for_devices(t_w2_batch)), jax.local_devices())
    test_inputs_dev  = jax.device_put_sharded(list(split_for_devices(test_inputs)), jax.local_devices())

    pmapped_train_fn = jax.pmap(
        vectorized_train_single_task,
        axis_name="device",
        static_broadcasted_argnums=(3,4,5,6,8,9,10)
    )
    for v in v_values:
        print(f"Begining {v=}")
        key, run_key, t2_key, rand_vec_key = random.split(script_master_key, 4)
        run_keys = random.split(run_key, num_runs)
        # t2_w1_key, t2_w2_key = random.split(t2_key)
        rand_vecs = random.normal(rand_vec_key, (num_runs, d_in, 1))
        
        # Create teacher 2 (only needs one instance per v)
        # Create teacher 2 (only needs one instance per v)
        t2_w1_batch = v*t1_w1_batch + jnp.sqrt(1-v**2)*ortho_t1_w1_batch
        t2_w1_batch /= jnp.linalg.norm(t2_w1_batch, axis=1, keepdims=True) + 1e-8
        teacher_similarity = jnp.mean(jnp.sum(t1_w1_batch * t2_w1_batch, axis=1))
        print(f"Similarity between t1_w1 and t2_w1: {teacher_similarity}")

        t2_w1_dev = jax.device_put_sharded(list(split_for_devices(t2_w1_batch)), jax.local_devices())
        t2_w2_dev = jax.device_put_sharded(list(split_for_devices(t_w2_batch)), jax.local_devices())
        run_keys_dev = jax.device_put_sharded(list(split_for_devices(run_keys)), jax.local_devices())

        # Train
        train_loss_dev, test_loss_dev = pmapped_train_fn(
            run_keys_dev,
            t2_w1_dev,
            t2_w2_dev,
            sample_rate,
            d_in,
            student.apply,
            optimizer,
            test_inputs_dev,
            num_epochs,
            batch_size,
            # d_hs,
            d_h,
        )

        # Save results
        results = {
            "train_loss": np.array(train_loss_dev).reshape(num_runs, -1),
            "test_loss": np.array(test_loss_dev).reshape(num_runs, -1),
            "epochs": np.arange(0, num_epochs, sample_rate),
            "v": v,
            "d_hs": d_hs,
            "lr": lr,
            "num_runs": num_runs
        }
        np.savez(os.path.join(output_dir, f"v_{v:.2f}.npz"), **results)

    print("Run complete.")