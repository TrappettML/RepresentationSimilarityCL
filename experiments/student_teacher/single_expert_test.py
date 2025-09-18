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
from run_ntasks_fb_gates import get_all_teacher_weights
from single_expert import train_experts



# -------------------------
# Test Function
# -------------------------
def run_expert_test():
    """Runs a small-scale test to verify expert training works."""
    print("\n" + "="*50)
    print("Running Expert Test Suite")
    print("="*50 + "\n")
    
    # Test Configuration
    test_config = {
        'num_runs': 3,
        'ntasks': 2,
        'd_in': 800,
        'd_ht': 200,
        'epochs_per_task': 50000,
        'sample_rate': 100,
        'batch_size': 256,
        'test_size': 10000,
        'lr': 0.01
    }
    
    # Create test data
    key = random.PRNGKey(42)
    d_ht = test_config['d_ht']
    test_inputs = random.normal(key, (test_config['test_size'], test_config['d_in']))
    test_inputs = test_inputs[jnp.newaxis, :]  # add batch dim

    # Generate teacher parameters
    key, teacher_key = random.split(key, 2)
    all_teacher_params = get_all_teacher_weights(teacher_key, 
                                                 test_config['num_runs'],
                                                 test_config['ntasks'],
                                                 test_config['d_in'],
                                                 test_config['d_ht'],
                                                 0)

    optimizer = optax.sgd(test_config['lr']) 

    # Run expert training
    key, subkey = random.split(key)
    test_losses = train_experts(
        subkey,
        test_config,
        optimizer,
        test_inputs,
        all_teacher_params
    )

    # Check and print results
    num_samples = test_config['epochs_per_task'] // test_config['sample_rate']
    expected_shape = (test_config['num_runs'], test_config['ntasks'], num_samples)
    
    print("\nTest Results:")
    print(f"Test losses shape: {test_losses.shape}")
    print(f"Expected shape: {expected_shape}")
    print(f"Sample losses: {test_losses[0, 0, :5]}")  # First 5 samples
    
    # Simple validity checks
    assert test_losses.shape == expected_shape, \
        f"Unexpected test losses shape. Got {test_losses.shape}, expected {expected_shape}"
    # assert jnp.all(jnp.isfinite(test_losses)), "NaN/Inf values found in losses"
    
    print("\n" + "="*50)
    print("Expert Tests PASSED")
    print("="*50 + "\n")

    fig, axs = plt.subplots(test_config['ntasks'], sharex=True)
    x = jnp.arange(0, test_config['epochs_per_task'], test_config['sample_rate'])
    for t in range(test_config['ntasks']):
        t_mean = jnp.mean(test_losses[:,t,:], axis=0)
        t_std = jnp.std(test_losses[:,t,:], axis=0)
        axs[t].plot(x, t_mean)
        axs[t].fill_between(x, t_mean-t_std, t_mean+t_std, alpha=0.2)
        axs[t].set_title(f'Expert Test Loss for Task {t}')
        axs[t].set_yscale('log')
    
    fig.savefig('./single_expert_test.png')

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    run_expert_test()