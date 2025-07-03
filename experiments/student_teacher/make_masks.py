import jax
import jax.numpy as jnp
from typing import Tuple
from ipdb import set_trace


def random_gates(r_key, s: float, d_in: int, n_vectors: int):
    d_in = int(d_in)
    random_keys = jax.random.split(r_key, n_vectors)
    return jax.vmap(lambda k: jax.random.bernoulli(k, 1-s, (d_in)))(random_keys).astype(jnp.float32)

def deterministic_gates2(key, v, s, d_in, n_vectors):
    ''' ============== type 1 =============='''
    # b1_key, b2_key = jax.random.split(key, 2)
    # determ_keys = jax.random.split(b1_key, n_vectors)
    # pi = jnp.clip((0.5*jnp.tanh(b*((1-s)-m))+0.5), min=0.01, max=1)
    # g_1 = jax.vmap(lambda k: jax.random.bernoulli(k, 1-pi, (d_in)))(determ_keys).astype(jnp.float32)
    # determ_keys2 = jax.random.split(b2_key, n_vectors)
    # g_2 = jax.vmap(lambda k: jax.random.bernoulli(k, 1-pi, (d_in)))(determ_keys2).astype(jnp.float32)
    ''' ============== type 2 =============='''
    b = 5
    m = 0.5
    pi = (1-s)
    key1, key2 = jax.random.split(key, 2)
    determ_keys = jax.random.split(key1, n_vectors)
    g_1 = jax.vmap(lambda k: jax.random.bernoulli(k, pi, (d_in)))(determ_keys).astype(jnp.float32)
    not_g1 = 1 - g_1 # will be 1's where g_1 had zeros, use for p0
    u = jnp.clip((0.5*jnp.tanh(b*(v-m))+0.5), min=0.0, max=1.0)
    # u = jnp.clip((4*(v-0.5)**2), min=0.05, max=1.0)
    p1 = g_1 * u
    p0 = not_g1 * jnp.clip(((1 - s)/(s+1e-12)) * (1 - u), min=0.0, max=1.0)
    p0 = not_g1 * (1-u)
    P = p1 + p0 # should be same shape as g_1
    g2_key, key = jax.random.split(key2)
    g_2 = jax.random.bernoulli(g2_key, P, (g_1.shape)).astype(jnp.float32)
    return g_1, g_2

    
def overlap_gates(key, v, s, u, d_in, n_vectors):
    '''Sparsity will be fixed at 0.5 
        s will dictate the overlap'''
    pi = s
    key1, key2 = jax.random.split(key, 2)
    determ_keys = jax.random.split(key1, n_vectors)
    g_1 = jax.vmap(lambda k: jax.random.bernoulli(k, pi, (d_in)))(determ_keys).astype(jnp.float32)
    not_g1 = 1 - g_1 # will be 1's where g_1 had zeros, use for p0
    p1 = g_1 * u
    p0 = not_g1 * (1-u)
    P = p1 + p0 # should be same shape as g_1
    g2_key, key = jax.random.split(key2)
    g_2 = jax.random.bernoulli(g2_key, P, (g_1.shape)).astype(jnp.float32)
    return g_1, g_2


def create_masks(d_in: int,
                  sparsity: float,
                  v: float, # task similarity; 0: orthog, 1: same
                  u: float, # overlap
                  m_type: str, # determ or random
                  key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    masks = None
    m_type = m_type.lower()
    if m_type=='random':
        masks = random_gates(key, sparsity, d_in, 2) # get two repeats to return each mask
        mask1 = masks[0, :]
        mask2 = masks[1, :]
    elif m_type=='determ':
        # masks = deterministic_gates(key, v, sparsity, d_in, 2)
        mask1, mask2 = deterministic_gates2(key, v, sparsity, d_in, 1)
    elif m_type=='overlap':
        mask1, mask2 = overlap_gates(key, v, sparsity, u, d_in, 1)
    # set_trace()

    assert mask1 != None, f"type not possible, got {m_type=}, should be one of ('determ','random','overlap')."

    overlap = jnp.sum(jnp.multiply(mask1, mask2), axis=-1)/jnp.sum(mask1)
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
                    sparsity=sparsity,
                    v=v_param,
                    m_type=mask_type,
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