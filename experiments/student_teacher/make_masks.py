import jax
import jax.numpy as jnp
from typing import Tuple
from ipdb import set_trace


def random_gates(r_key, s, d_in, d_out, n_vectors):
    d_in = int(d_in)
    d_out = int(d_out)
    random_keys = jax.random.split(r_key, n_vectors)
    return jax.vmap(lambda k: jax.random.bernoulli(k, s, (d_in, d_out)))(random_keys).squeeze(-1).astype(jnp.float32)

def deterministic_gates(key, v, s, d_in, d_out, n_vectors):
    h = v + 0.4 * jnp.pi
    theta = jnp.array([jnp.cos(h*jnp.pi), jnp.sin(h*jnp.pi)])
    determ_keys = jax.random.split(key, n_vectors*2)

    base = jax.vmap(lambda k: jax.random.uniform(k, (d_in, d_out)))(determ_keys)
    base = jnp.reshape(base, (n_vectors, d_in, d_out, 2))
    # print(f"{base.shape=}")
    z = base @ theta.T
    # print(f"{z.shape=}")
    ones = jnp.ones_like(z)
    g_determ = jnp.heaviside(z - (s-0.5), ones)
    # print(f"{base.shape=}")
    #####
    # m = 10
    # z_sigmoid = 1.5/ (1 + jnp.exp(-m*(z)))
    # # # Create smooth threshold
    # s = s-0.01
    # g_determ = (z_sigmoid > s).astype(jnp.float32)
    #######
    return g_determ.squeeze(-1).astype(jnp.float32)
    

def create_masks(d_in: int,
                  d_out: int,
                  sparsity: float,
                  v: float, # task similarity; 0: orthog, 1: same
                  m_type: str, # determ or random
                  key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    masks = None
    m_type = m_type.lower()
    if m_type=='random':
        masks = random_gates(key, sparsity, d_in, d_out, 2) # get two repeats to return each mask
    elif m_type=='determ':
        masks = deterministic_gates(key, v, sparsity, d_in, d_out, 2)

    assert masks != None, f"type not possible, got {m_type=}, should be either 'determ' or 'random'."
    mask1 = masks[0, :]
    mask2 = masks[1, :]
    overlap = jnp.sum(jnp.multiply(mask1, mask2), axis=-1)/d_in
    return mask1, mask2, overlap


def _create_masks(d_hs: int, # total num of student hidden neurons
                 n_active: int, # num of active neurons
                 shared_neurons: int, # 
                 mask_type: str, # determ or random/shuffled 
                 key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Creates student masks with controlled overlap and sparsity,
    then shuffles them randomly while preserving properties.
    
    Args:
        d_hs: Total number of neurons (mask size)
        n_active: Number of active neurons per student
        shared_neurons: Number of overlapping active neurons
        mask_type: either 'determ' or 'random'
        key: JAX PRNG key
    
    Returns:
        tuple: (mask1, mask2, actual_overlap_percent)
    """
    # Validate inputs
    assert shared_neurons <= n_active, f"Overlap can't exceed active neurons per head\nshared_neurons: {shared_neurons}\nn_active: {n_active}"
    assert (2*n_active - shared_neurons) <= d_hs, f"Not enough total neurons\nshared_neurons:{shared_neurons}\nn_active: {n_active}\nd_hs: {d_hs}"
    assert mask_type.lower() in ['determ', 'random'], f"Unknown Masking Type, got {mask_type}"
    
    # Initialize masks with zeros
    mask1 = jnp.zeros(d_hs)
    mask2 = jnp.zeros(d_hs)
    
    # Create indices for mask components
    indices = jnp.arange(d_hs)
    
    # Common components
    common = indices[:shared_neurons]
    
    # Unique components
    unique1 = indices[shared_neurons:n_active]
    unique2 = indices[n_active:2*n_active - shared_neurons]
    
    # Set values in masks
    mask1 = mask1.at[common].set(1)
    mask1 = mask1.at[unique1].set(1)
    
    mask2 = mask2.at[common].set(1)
    mask2 = mask2.at[unique2].set(1)
    
    # Shuffle if requested
    if mask_type.lower() == 'random':
        perm = jax.random.permutation(key, d_hs)
        mask1 = mask1[perm]
        mask2 = mask2[perm]
    
    # Calculate actual overlap percentage
    actual_overlap = jnp.sum(mask1 * mask2)
    overlap_percent = (actual_overlap / n_active) * 100

    return mask1, mask2, overlap_percent

# Example usage
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    num_repeats = 100

    # Format: (d_hs, n_active, shared_neurons, target_overlap_percent)
    test_cases = [
        (100,  .2, 1),    # completely separate
        (100,  .2, .60),    # half shared
        (100,  .40, .50),    # quarter shared
        (100,  .30, .70),    # high overlap
        (100, .50, .70)     # medium overlap
    ]

    for mask_type in ['determ', 'random']:
        print(f"\n=== Testing {mask_type.upper()} masks over {num_repeats} repeats ===")
        for idx, (d_hs, sparsity, v_param) in enumerate(test_cases, 1):

            overlaps = []
            # generate a fresh subkey each repeat
            for rep in range(num_repeats):
                subkey = jax.random.fold_in(key, rep)
                mask1, mask2, ovlp = create_masks(
                    d_in=d_hs,
                    d_out=1,
                    sparsity=sparsity,
                    v=v_param,
                    type=mask_type,
                    key=subkey
                )
                overlaps.append(float(ovlp))

            mean_ovlp = jnp.mean(jnp.array(overlaps))
            std_ovlp  = jnp.std(jnp.array(overlaps))

            print(f"\nCase {idx}: d_hs={d_hs}, {sparsity=}, {v_param=}")
            print(f" Mean overlap: {mean_ovlp:.2f}%  ± {std_ovlp:.1f}% ")
            print(f"{mask1=}\n{mask2}\n")

            # # quick sanity check on mean
            # assert abs(mean_ovlp - target) < 5.0, (
            #     f"Mean overlap {mean_ovlp:.1f}% deviates more than 1% from target {target}%"
            # )

        print("✓ Done with", mask_type)