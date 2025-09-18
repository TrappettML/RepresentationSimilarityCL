import jax
import jax.numpy as jnp
from typing import Tuple, List
from ipdb import set_trace
from math import isclose
from copy import copy



def random_gates(r_key, s: float, d_in: int, n_vectors: int):
    d_in = int(d_in)
    random_keys = jax.random.split(r_key, n_vectors)
    return jax.vmap(lambda k: jax.random.bernoulli(k, 1-s, (d_in)))(random_keys).astype(jnp.float32)

def deterministic_gates(key, v, s, d_in, n_vectors):
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
    key1, key2, g2_key = jax.random.split(key, 3)
    g_1 = jax.random.bernoulli(key1, pi, (d_in,)).astype(jnp.float32)
    not_g1 = 1 - g_1 # will be 1's where g_1 had zeros, use for p0
    u = jnp.clip((0.5*jnp.tanh(b*(v-m))+0.5), min=0.0, max=1.0)
    # u = jnp.clip((4*(v-0.5)**2), min=0.05, max=1.0)
    p1 = g_1 * u
    p0 = not_g1 * jnp.clip(((1 - s)/(s + 1e-8)) * (1 - u), min=0.0, max=1.0)
    # p0 = not_g1 * (1-u)
    P = p1 + p0 # should be same shape as g_1
    g2_key, key = jax.random.split(key2)
    g_2 = jax.random.bernoulli(g2_key, P, (g_1.shape)).astype(jnp.float32)
    return [g_1, g_2]


def overlap_gates_old(key, v, s, u, d_in, n_vectors):
    '''Sparsity will be fixed at 0.5 
        s will dictate the overlap'''
    pi = s
    key1, key2, g2_key = jax.random.split(key, 3)
    g_1 = jax.random.bernoulli(key1, pi, (d_in,)).astype(jnp.float32)
    not_g1 = 1 - g_1 # will be 1's where g_1 had zeros, use for p0
    p1 = g_1 * u
    p0 = not_g1 * (1-u)
    P = p1 + p0 # should be same shape as g_1
    g2_key, key = jax.random.split(key2)
    g_2 = jax.random.bernoulli(g2_key, P, (g_1.shape)).astype(jnp.float32)
    return [g_1, g_2]

    
def overlap_gates(key: jax.random.PRNGKey, s: float, u: float, d_in: int, ntasks: int) -> List[jnp.ndarray]:
    """
    Generates n-task masks with constant expected cosine similarity 'u' to the global pool.

    This function implements the "global pool" model, where each new mask is
    generated based on the union of all previous masks. It dynamically adjusts
    sampling probabilities to ensure the expected cosine similarity between a
    new mask and the growing pool remains constant, correcting for the natural
    tendency of similarity to decrease as the pool size increases.

    Args:
        key: JAX random key.
        s: Sparsity of each mask (proportion of zeros).
        u: Target cosine similarity (0 to 1) with the global pool.
        d_in: The dimension of the masks.
        ntasks: The total number of masks to generate.
    
    Returns:
        A list of 'ntasks' mask tensors, each of shape (d_in,).
    """
    if ntasks == 0:
        return []
        
    gates = []
    pi = (1 - s)
    n_target_active = d_in * pi

    # Generate the first mask randomly
    g_key, key = jax.random.split(key)
    g1 = jax.random.bernoulli(g_key, pi, (d_in,)).astype(jnp.float32)
    gates.append(g1)
    
    if ntasks == 1:
        return gates
        
    all_gates = g1

    for _ in range(ntasks - 1):
        n_already_active = jnp.sum(all_gates)
        n_available_inactive = d_in - n_already_active

        # --- Corrected Logic to Target Constant Cosine Similarity ---
        
        # 1. Calculate the number of shared units needed to achieve similarity 'u'
        # E[shared] / (sqrt(E[norm_new]) * sqrt(E[norm_pool])) ≈ u
        # E[shared] ≈ u * sqrt(n_target_active * n_already_active)
        target_shared_units = u * jnp.sqrt(n_target_active * n_already_active)

        # 2. Derive the per-neuron probability 'p1' for the active pool
        # E[shared] = p1 * n_already_active
        p1 = target_shared_units / (n_already_active + 1e-8)

        # 3. Derive 'p0' for the inactive pool to maintain overall density
        # E[total_active] = p1*n_already_active + p0*n_available_inactive = n_target_active
        # target_shared_units + p0*n_available_inactive = n_target_active
        p0 = (n_target_active - target_shared_units) / (n_available_inactive + 1e-8)
        
        # 4. Construct the final probability mask with safety clips
        P = (all_gates * jnp.clip(p1, 0.0, 1.0) +
             (1 - all_gates) * jnp.clip(p0, 0.0, 1.0))

        # Generate the next mask and update the global pool
        g_key, key = jax.random.split(key)
        g_next = jax.random.bernoulli(g_key, P).astype(jnp.float32)
        gates.append(g_next)
        all_gates = jnp.clip(all_gates + g_next, 0, 1)
        
    return gates


def create_masks(d_in: int,
                  sparsity: float,
                  v: float, # task similarity; 0: orthog, 1: same
                  u: float, # overlap
                  m_type: str, # determ or random
                  key: jax.random.PRNGKey,
                  ntasks: int) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    ######### need to fix this to take in a list of similarities ############
    masks = None
    m_type = m_type.lower()
    # if not isclose(sparsity, 1 - 1/ntasks, abs_tol=0.1):
    #     print("Warning: sparsity must be less than 1/ntasks if we want no overlap")
    #     print(f"Got: {1-sparsity=} and {1/ntasks=}")
    if m_type=='random':
        masks = random_gates(key, sparsity, d_in, ntasks) # get two repeats to return each mask
        # mask1 = masks[0, :]
        # mask2 = masks[1, :]
        masks = [masks[i] for i in range(ntasks)]
    elif m_type=='determ':
        # masks = deterministic_gates(key, v, sparsity, d_in, 2)
        if ntasks > 2:
            print("Deterministic gates not set up for more than 2 tasks. Please Fix")
        masks = deterministic_gates(key, v, sparsity, d_in, 1)
    elif m_type=='overlap':
        masks = overlap_gates(key, sparsity, u, d_in, ntasks)
    # set_trace()

    assert masks[0] != None, f"type not possible, got {m_type=}, should be one of ('determ','random','overlap')."

    # overlap = jnp.sum(jnp.multiply(mask1, mask2), axis=-1)/jnp.sum(mask1)
    overlap = None
    if ntasks > 1:
        overlap = mean_mask_overlap(masks)

    return jnp.stack(masks), overlap

def mean_mask_overlap(masks):
    all_similarities = []
    for i in range(len(masks)):
        cp_masks = copy(masks)
        mask_i = cp_masks.pop(i)
        all_other_active = jnp.clip(jnp.sum(jnp.array(cp_masks), axis=0), min=0, max=1.0)
        # set_trace()
        sim_i = jnp.matmul(mask_i, all_other_active.T)/(jnp.linalg.norm(all_other_active) * jnp.linalg.norm(mask_i))
        # print(f"{sim_i=}")
        all_similarities.append(sim_i)
    overlap = jnp.nanmean(jnp.stack(all_similarities))
    # print(f"{overlap=}")
    # set_trace()
    return overlap



# Example usage
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    num_repeats = 100
    n_neurons = 200
    ntasks = 10
    overlap=0.00

    # Format: (d_hs, n_active, shared_neurons, target_overlap_percent)
    test_cases = [
        (n_neurons,  1-1/ntasks, 1),    # completely separate
        (n_neurons,  1-1/ntasks, .60),    # half shared
        (n_neurons,  1-1/ntasks, .50),    # quarter shared
        (n_neurons,  1-1/ntasks, .70),    # high overlap
        (n_neurons, 1-1/ntasks, .70)     # medium overlap
    ]

    for mask_type in ['overlap']:#,'determ', 'random'
        print(f"\n=== Testing {mask_type.upper()} masks over {num_repeats} repeats ===")
        for idx, (d_hs, sparsity, v_param) in enumerate(test_cases, 1):

            overlaps = []
            active_units = []
            # generate a fresh subkey each repeat
            for rep in range(num_repeats):
                subkey = jax.random.fold_in(key, rep)
                masks, ovlp = create_masks(
                    d_in=d_hs,
                    sparsity=sparsity,
                    v=v_param,
                    u=overlap,
                    m_type=mask_type,
                    key=subkey,
                    ntasks=ntasks,
                )
                overlaps.append(float(ovlp))
                active_units.append([jnp.sum(m) for m in masks])


            mean_ovlp = jnp.mean(jnp.array(overlaps))
            std_ovlp  = jnp.std(jnp.array(overlaps))
            mean_act  = [jnp.mean(jnp.array(an)) for an in zip(*active_units)]

            print(f"\nCase {idx}: d_hs={d_hs}, {sparsity=}, {v_param=}")
            aun_str = f"Mean active units:"
            for i,m_act in enumerate(mean_act):
                aun_str += f"t{i}: {m_act}; "
            print(aun_str)
            print(f" Mean overlap: {mean_ovlp*100:.0f}%  ± {std_ovlp*100:.0f}% ")
            m_str = f""
            for i,m in enumerate(masks):
                m_str += f"Mask[i]: {m}\n"
            print(m_str)
            print(f"{type(masks)=}\n{masks[0].shape=}")
            # # quick sanity check on mean
            # assert abs(mean_ovlp - target) < 5.0, (
            #     f"Mean overlap {mean_ovlp:.1f}% deviates more than 1% from target {target}%"
            # )

        print("✓ Done with", mask_type)
        subkeys = jax.random.split(key, 5)
        masks, ovlp = jax.vmap(
            create_masks, in_axes=(None, None, None, None, None, 0, None),
            )(
               d_hs,sparsity,v_param,1.00,mask_type,subkeys,3,
            )
        print(f"{masks[0].shape=}")