import jax
import jax.numpy as jnp
from typing import Tuple
from ipdb import set_trace


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
    key1, key2 = jax.random.split(key, 2)
    determ_keys = jax.random.split(key1, n_vectors)
    g_1 = jax.vmap(lambda k: jax.random.bernoulli(k, pi, (d_in)))(determ_keys).astype(jnp.float32)
    not_g1 = 1 - g_1 # will be 1's where g_1 had zeros, use for p0
    u = jnp.clip((0.5*jnp.tanh(b*(v-m))+0.5), min=0.0, max=1.0)
    # u = jnp.clip((4*(v-0.5)**2), min=0.05, max=1.0)
    p1 = g_1 * u
    p0 = not_g1 * jnp.clip(((1 - s)/(s)) * (1 - u), min=0.0, max=1.0)
    p0 = not_g1 * (1-u)
    P = p1 + p0 # should be same shape as g_1
    g2_key, key = jax.random.split(key2)
    g_2 = jax.random.bernoulli(g2_key, P, (g_1.shape)).astype(jnp.float32)
    return g_1, g_2


def overlap_gates_old(key, v, s, u, d_in, n_vectors):
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

    
def overlap_gates(key, s, u, d_in, n_vectors, ntasks):
    '''Sparsity will be fixed at 0.5 
        s will dictate the overlap'''
    gates = []
    pi = (1-s) # density=(1-s) = 1/n_tasks
    g1_key, key = jax.random.split(key, 2)
    g1_keys = jax.random.split(g1_key, n_vectors)
    g_1 = jax.vmap(lambda k: jax.random.bernoulli(k, pi, (d_in)))(g1_keys).astype(jnp.float32)
    gates.append(g_1)
    for i in range(ntasks - 1):
        prev_g = gates[-1]
        not_g = 1 - prev_g # will be 1's where g_1 had zeros, use for p0
        p1 = prev_g * u
        p0 = not_g * jnp.clip(((1 - s)/(s)) * (1 - u), min=0.0, max=1.0)
        P = p1 + p0 # should be same shape as g_1
        next_g_key, key = jax.random.split(key)
        next_g = jax.random.bernoulli(next_g_key, P, (prev_g.shape)).astype(jnp.float32)
        gates.append(next_g)
    return gates


def create_masks(d_in: int,
                  sparsity: float,
                  v: list[float], # task similarity; 0: orthog, 1: same
                  u: float, # overlap
                  m_type: str, # determ or random
                  key: jax.random.PRNGKey,
                  ntasks: int) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    ######### need to fix this to take in a list of similarities ############
    masks = None
    m_type = m_type.lower()
    if (1-sparsity) > 1/ntasks:
        print("Warning: sparsity must be less than 1/ntasks if we want ")
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
        masks = overlap_gates(key, sparsity, u, d_in, 1, ntasks)
    # set_trace()

    assert masks[0] != None, f"type not possible, got {m_type=}, should be one of ('determ','random','overlap')."

    # overlap = jnp.sum(jnp.multiply(mask1, mask2), axis=-1)/jnp.sum(mask1)
    overlap = mean_mask_overlap(masks)
    return jnp.stack(masks), overlap

def mean_mask_overlap(masks):
    all_similarities = []
    for i in range(len(masks)):
        for j in range(i+1,len(masks)):
            all_similarities.append(jnp.sum(jnp.multiply(masks[i], masks[j]), axis=-1)/jnp.sum(masks[0]))
    
    return jnp.mean(jnp.array(all_similarities))



# Example usage
if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    num_repeats = 100
    n_neurons = 200

    # Format: (d_hs, n_active, shared_neurons, target_overlap_percent)
    test_cases = [
        (n_neurons,  .5, 1),    # completely separate
        (n_neurons,  .6, .60),    # half shared
        (n_neurons,  .7, .50),    # quarter shared
        (n_neurons,  .80, .70),    # high overlap
        (n_neurons, .90, .70)     # medium overlap
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
                    u=1.00,
                    m_type=mask_type,
                    key=subkey,
                    ntasks=3,
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