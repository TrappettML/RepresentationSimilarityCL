import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
import optax
from flax import linen as nn
import matplotlib.pyplot as plt
import numpy as np
from make_masks import create_masks
from single_expert import teacher_forward, scaled_normal_init
from run_ntasks_fb_gates import get_all_teacher_weights


# --- Configuration ---
class Config:
    def __init__(self):
        self.d_in = 400  # Input dimension
        self.d_ht = 100   # Teacher hidden dimension
        self.d_hs = 200  # Student hidden dimension (should be same as teacher for parameter count)
        self.num_runs = 20  # Number of independent runs
        self.epochs = 50000
        self.sample_interval = 500
        self.batch_size = 256
        self.test_size = 1000
        self.lr = 0.1
        self.seed = 42
        self.sparsity = 0.5  # Added for mask creation
        self.similarity = 0.5  # Added for mask creation
        self.overlap = 0.5  # Added for mask creation
        self.g_type = 'random'  # Added for mask creation

config = Config()


# --- Expert Networks ---
class StandardExpert(nn.Module):
    """Standard expert network."""
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x, mask=None):
        # h = nn.Dense(self.hidden_dim, use_bias=False, name="dense_1")(x)
        h = nn.Dense(
            self.hidden_dim,
            use_bias=False, 
            name="dense_1",
            kernel_init=scaled_normal_init(jnp.sqrt(2 / x.shape[-1]))
        )(x)
        h = nn.relu(h)
        h = h/jnp.sqrt(self.hidden_dim)
        out = nn.Dense(1, use_bias=False, name="dense_2",
            kernel_init=scaled_normal_init(jnp.sqrt(1.0 / self.hidden_dim)))(h)
        return out

class GatedExpert(nn.Module):
    """Gated expert network with external masks."""
    hidden_dim: int  # Full hidden dimension
    
    @nn.compact
    def __call__(self, x, mask):
        h = nn.Dense(
            self.hidden_dim,
            use_bias=False, 
            name="dense_1",
            kernel_init=scaled_normal_init(jnp.sqrt(2 / x.shape[-1]))
        )(x)
        h = nn.relu(h)
        h = h/jnp.sqrt(self.hidden_dim/2)
        h = h*mask
        out = nn.Dense(1, use_bias=False, name="dense_2",
            kernel_init=scaled_normal_init(jnp.sqrt(1.0 / int(self.hidden_dim/2))))(h)
        return out

# --- Vectorized Training Function ---
def train_experts_vectorized(keys, teachers_params, expert_model, experts_params, 
                            test_inputs, masks, config):
    """Train experts on teachers in a vectorized manner."""
    optimizer = optax.sgd(config.lr)
    vmap_optimizer_init = vmap(optimizer.init)
    vmap_optimizer_update = vmap(optimizer.update)
    opt_states = vmap_optimizer_init(experts_params)

    # Loss function for a single run
    def loss_fn(expert_params, teacher_params, inputs, mask):
        teacher_predictions = teacher_forward(teacher_params, inputs)
        expert_predictions = expert_model.apply(expert_params, inputs, mask).squeeze(1)
        return jnp.mean((expert_predictions - teacher_predictions)**2)
    
    # Vectorized loss function over all runs
    vmap_loss_fn = vmap(loss_fn, in_axes=(0, 0, 0, 0))
    
    @jit
    def train_step(carry, step_idx):
        experts_params, opt_states, key, masks = carry
        
        key, data_key = random.split(key)
        data_keys = random.split(data_key, config.num_runs)
        train_inputs = vmap(lambda k: random.normal(k, (config.batch_size, config.d_in)))(data_keys)
        
        # Calculate loss and gradients for each run separately
        losses = vmap_loss_fn(experts_params, teachers_params, train_inputs, masks)
        grads = vmap(grad(loss_fn))(experts_params, teachers_params, train_inputs, masks)
        
        # Update each expert separately
        updates, new_opt_states = vmap_optimizer_update(grads, opt_states, experts_params)
        new_experts_params = vmap(optax.apply_updates)(experts_params, updates)
        
        # Calculate test losses at intervals
        test_losses = jnp.where(
            (step_idx + 1) % config.sample_interval == 0,
            vmap_loss_fn(new_experts_params, teachers_params, test_inputs, masks),
            jnp.full(config.num_runs, jnp.nan)
        )
        new_carry = (new_experts_params, new_opt_states, key, masks)
        return new_carry, (losses, test_losses)
    
    initial_carry = (experts_params, opt_states, keys[0], masks)
    steps = jnp.arange(config.epochs)
    final_carry, (train_losses, test_losses) = jax.lax.scan(train_step, initial_carry, steps)
    
    sampled_indices = jnp.arange(config.sample_interval - 1, config.epochs, config.sample_interval)
    sampled_train_losses = train_losses[sampled_indices]
    sampled_test_losses = test_losses[sampled_indices]
    
    return sampled_train_losses, sampled_test_losses

# --- Main Function ---
def main():
    key = random.PRNGKey(config.seed)
    key, teacher_key, test_key, std_init_key, gate_init_key, mask_key, std_train_key, gate_train_key = random.split(key, 8)
    
    teacher_keys = random.split(teacher_key, config.num_runs)
    test_keys = random.split(test_key, config.num_runs)
    std_init_keys = random.split(std_init_key, config.num_runs)
    gate_init_keys = random.split(gate_init_key, config.num_runs)
    mask_keys = random.split(mask_key, config.num_runs)
    std_train_keys = random.split(std_train_key, config.num_runs)
    gate_train_keys = random.split(gate_train_key, config.num_runs)
    
    dummy_input = jnp.ones((1, config.d_in))

    # --- Teacher Network Initialization ---
    all_teacher_params = get_all_teacher_weights(teacher_key, 
                                                config.num_runs,
                                                1, # only training on a single task
                                                config.d_in,
                                                config.d_ht,
                                                0)
    # --- Prepare Batched Data ---
    all_teacher_params = jax.tree_util.tree_map( # remove the task dimension
        lambda x: x.reshape((config.num_runs,) + x.shape[2:]), all_teacher_params
    )
    vmap_teacher_forward = vmap(teacher_forward, in_axes=(0, 0))
    test_inputs = vmap(lambda k: random.normal(k, (config.test_size, config.d_in)))(test_keys)
    expert_teacher_test_targets = vmap_teacher_forward(all_teacher_params, test_inputs)
    # --- Standard Expert Initialization ---
    std_expert_model = StandardExpert(hidden_dim=config.d_hs)
    std_params = vmap(std_expert_model.init, in_axes=(0, None))(std_init_keys, dummy_input)
    
    # --- Gated Expert Initialization ---
    # Create masks for gated experts
    vmap_create_masks = vmap(create_masks, in_axes=(None, None, None, None, None, 0, None))
    masks_batch, m_overlap = vmap_create_masks(
        config.d_hs*2, config.sparsity, config.similarity, config.overlap,
        config.g_type, mask_keys, 1  # Only 1 task
    )
    masks = masks_batch[:, 0]  # Extract masks for the single task
    
    gate_expert_model = GatedExpert(hidden_dim=config.d_hs*2)
    gate_params = vmap(gate_expert_model.init, in_axes=(0, None, None))(gate_init_keys, dummy_input, masks)
    
    # --- Training ---
    print("Training standard experts...")
    std_train_losses, std_test_losses = train_experts_vectorized(
        std_train_keys, all_teacher_params, std_expert_model, std_params,
        test_inputs, jnp.ones((config.num_runs, config.d_hs)), config  # Use ones mask for standard experts
    )
    
    print("Training gated experts...")
    gate_train_losses, gate_test_losses = train_experts_vectorized(
        gate_train_keys, all_teacher_params, gate_expert_model, gate_params,
        test_inputs, masks, config
    )
    
    # --- Plotting ---
    epochs_sampled = jnp.arange(config.sample_interval, config.epochs + 1, config.sample_interval)
    
    std_train_mean = jnp.nanmean(std_train_losses, axis=1)
    std_train_std = jnp.nanstd(std_train_losses, axis=1)
    std_test_mean = jnp.nanmean(std_test_losses, axis=1)
    std_test_std = jnp.nanstd(std_test_losses, axis=1)
    
    gate_train_mean = jnp.nanmean(gate_train_losses, axis=1)
    gate_train_std = jnp.nanstd(gate_train_losses, axis=1)
    gate_test_mean = jnp.nanmean(gate_test_losses, axis=1)
    gate_test_std = jnp.nanstd(gate_test_losses, axis=1)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(epochs_sampled, std_train_mean, label='Standard Expert', color='blue')
    plt.fill_between(epochs_sampled, std_train_mean - std_train_std, std_train_mean + std_train_std, alpha=0.2, color='blue')
    plt.semilogy(epochs_sampled, gate_train_mean, label='Gated Expert', color='red')
    plt.fill_between(epochs_sampled, gate_train_mean - gate_train_std, gate_train_mean + gate_train_std, alpha=0.2, color='red')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    plt.subplot(1, 2, 2)
    plt.semilogy(epochs_sampled, std_test_mean, label='Standard Expert', color='blue')
    plt.fill_between(epochs_sampled, std_test_mean - std_test_std, std_test_mean + std_test_std, alpha=0.2, color='blue')
    plt.semilogy(epochs_sampled, gate_test_mean, label='Gated Expert', color='red')
    plt.fill_between(epochs_sampled, gate_test_mean - gate_test_std, gate_test_mean + gate_test_std, alpha=0.2, color='red')
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.savefig('expert_comparison_vectorized.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFinal Performance (mean ± std across {config.num_runs} runs):")
    print(f"Standard Expert - Test: {std_test_mean[-1]:.6f} ± {std_test_std[-1]:.6f}")
    print(f"Gated Expert    - Test: {gate_test_mean[-1]:.6f} ± {gate_test_std[-1]:.6f}")

if __name__ == "__main__":
    main()