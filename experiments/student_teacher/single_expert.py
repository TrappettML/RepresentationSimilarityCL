import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

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
from timer_class import Timer
import matplotlib.pyplot as plt


def teacher_forward(params: dict, x:jnp.ndarray)->jnp.ndarray:
    """A simple two-layer teacher network forward pass for a batch of inputs."""
    # x shape: (batch_size, d_in)
    # params['w1'] shape: (d_in, d_ht)
    h = nn.relu(x @ params['w1']) # Result shape: (batch_size, d_ht)
    # params['w2'] shape: (d_ht, 1)
    return (h @ params['w2']).squeeze(-1) # Result shape: (batch_size)


# --- Model Definitions ---
def scaled_normal_init(scale: float):
    """Returns a function for scaled normal initialization."""
    def init(key, shape, dtype=jnp.float32):
        return scale * random.normal(key, shape, dtype)
    return init


class ExpertNetwork(nn.Module):
    """Simplified expert network without masks and with reduced hidden dimension."""
    hidden_dim: int  # d_ht
    
    @nn.compact
    def __call__(self, x):
        # jax.debug.print(f"{self.hiden_dim=}")
        # jax.debug.print(f"{x.shape=}")
        h = nn.Dense(
            self.hidden_dim,
            use_bias=False,
            kernel_init=scaled_normal_init(jnp.sqrt(2 / x.shape[-1]))
        )(x)
        h = nn.relu(h)
        h = h/jnp.sqrt(self.hidden_dim)
        out = nn.Dense(1, use_bias=False,
            kernel_init=scaled_normal_init(1.0 / jnp.sqrt(self.hidden_dim)))(h)
        return out.squeeze(-1)
    

@partial(jit, static_argnames=(
    "epochs_per_task", "sample_rate", "batch_size", "d_in",
    "model_apply_fn", "optimizer"
))
def train_single_expert(
    initial_carry: tuple,
    teacher_params: dict,
    test_targets: jnp.ndarray,
    test_inputs: jnp.ndarray,
    # Static args
    epochs_per_task: int,
    sample_rate: int,
    batch_size: int,
    d_in: int,
    model_apply_fn,
    optimizer,
):
    """JIT-compiled training loop for a single expert using lax.scan."""
    params, opt_state, key = initial_carry
    
    grad_fn = jit(jax.value_and_grad(
        lambda p, data, targets: jnp.mean((model_apply_fn({'params': p}, data) - targets)**2)
    ))
    
    def evaluate_expert(p):
        preds = model_apply_fn({'params': p}, test_inputs)
        return jnp.mean((preds - test_targets)**2)

    def step_fn(carry, step_idx):
        params, opt_state, key = carry
        
        key, data_key = random.split(key)
        batch_data = random.normal(data_key, (batch_size, d_in))
        batch_targets = teacher_forward(teacher_params, batch_data)
        
        loss, grads = grad_fn(params, batch_data, batch_targets)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        test_loss = lax.cond(
            step_idx % sample_rate == 0,
            lambda: evaluate_expert(new_params),
            lambda: jnp.nan
        )
        
        new_carry = (new_params, new_opt_state, key)
        return new_carry, test_loss
    
    final_carry, all_test_losses_with_nans = lax.scan(
        step_fn, (params, opt_state, key), jnp.arange(epochs_per_task)
    )
    
    # *** FIX ***
    # Return the full array with NaNs. The dynamic filtering is moved outside this JIT'd function.
    return final_carry, all_test_losses_with_nans

# --- Refactored Main Expert Training Orchestrator ---

def train_experts(
    key, config: dict, optimizer, test_inputs, all_teacher_params
):
    """Trains experts (one per task per run) in parallel."""
    num_runs, ntasks, d_in = config['num_runs'], config['ntasks'], config['d_in']
    epochs_per_task, sample_rate = config['epochs_per_task'], config['sample_rate']
    batch_size = config['batch_size']
    
    expert_hidden_dim = config['d_ht']
    total_experts = num_runs * ntasks
    
    print("-" * 50)
    print(f"Training {total_experts} experts in parallel...")
    print(f"  - Expert hidden dimension: {expert_hidden_dim}")
    
    # --- Prepare Batched Data ---
    expert_teacher_params = jax.tree_util.tree_map(
        lambda x: x.reshape((total_experts,) + x.shape[2:]), all_teacher_params
    )
    vmap_teacher_forward = vmap(teacher_forward, in_axes=(0, None))
    expert_teacher_test_targets = vmap_teacher_forward(expert_teacher_params, test_inputs)
    # jax.debug.breakpoint()
    # --- Initialize Batched Expert States ---
    model_keys, run_keys = random.split(key, 2)
    expert_model = ExpertNetwork(hidden_dim=expert_hidden_dim)
    
    vmap_init = vmap(expert_model.init, in_axes=(0, None))
    initial_params_batch = vmap_init(
        random.split(model_keys, total_experts), jnp.ones((1, d_in))
    )['params']
    initial_opt_state_batch = vmap(optimizer.init)(initial_params_batch)
    
    # --- Run Training in Parallel ---
    with Timer("Expert training duration"):
        vmapped_train_fn = vmap(
            train_single_expert,
            in_axes=((0, 0, 0), 0, 0, None, None, None, None, None, None, None)
        )
        
        initial_carry = (initial_params_batch, initial_opt_state_batch, random.split(run_keys, total_experts))
        
        _, final_losses_with_nans = vmapped_train_fn(
            initial_carry, expert_teacher_params, expert_teacher_test_targets, test_inputs,
            epochs_per_task, sample_rate, batch_size, d_in, expert_model.apply, optimizer
        )
    
    # *** FIX ***
    # The filtering now happens here, outside the JIT.
    # We use static integer indexing instead of a dynamic boolean mask.
    sampled_indices = jnp.arange(0, epochs_per_task, sample_rate)
    final_test_losses = final_losses_with_nans[:, sampled_indices]
    
    # Reshape back to (num_runs, ntasks, num_samples_per_task)
    num_samples = epochs_per_task // sample_rate
    final_test_losses = final_test_losses.reshape((num_runs, ntasks, num_samples))
    
    print("Expert training complete.")
    print("-" * 50)
    
    return final_test_losses


