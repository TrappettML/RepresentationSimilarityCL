import jax
import jax.numpy as jnp
from typing import Tuple

def create_masks(d_hs: int, # total num of student hidden neurons
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
    for m_type in ['determ', 'random']:
        # Example 1: ([0,0,1,1], [1,1,0,0]) --> shuffled
        mask1, mask2, overlap = create_masks(d_hs=4, n_active=2, shared_neurons=0, mask_type=m_type, key=key)
        print(f"Example 1 {m_type}:\n{mask1}\n{mask2}\nOverlap: {overlap:.0f}%\n")

        # Example 1.1: ([0,1,1,1], [1,1,1,0]) --> shuffled
        mask1, mask2, overlap = create_masks(d_hs=4, n_active=3, shared_neurons=2, mask_type=m_type, key=key)
        print(f"Example 3 {m_type}:\n{mask1}\n{mask2}\nOverlap: {overlap:.0f}%")

        # Example 2: ([0,1,1], [1,1,0]) --> shuffled
        mask1, mask2, overlap = create_masks(d_hs=3, n_active=2, shared_neurons=1, mask_type=m_type, key=key)
        print(f"Example 2 {m_type}:\n{mask1}\n{mask2}\nOverlap: {overlap:.0f}%\n")

        # Example 3: ([0,0,0,1,1,1,1], [1,1,1,1,0,0,0]) --> shuffled
        mask1, mask2, overlap = create_masks(d_hs=7, n_active=4, shared_neurons=1, mask_type=m_type, key=key)
        print(f"Example 3 {m_type}:\n{mask1}\n{mask2}\nOverlap: {overlap:.0f}%")

    # Format: (d_hs, k, shared_neurons, expected_overlap_percent)
    test_cases = [
        (4, 2, 0, 0),    # Completely separate
        (3, 2, 1, 50),   # Half shared
        (7, 4, 1, 25),   # Quarter shared
        (5, 3, 2, 66.7), # High overlap
        (10, 5, 3, 60)   # Medium overlap
    ]

    for mask_type in ['determ', 'random']:
        print(f"\n=== Testing {mask_type.upper()} masks ===")
        
        for case_idx, (d_hs, k, shared, target_ovlp) in enumerate(test_cases, 1):
            print(f"\nCase {case_idx}: d_hs={d_hs}, k={k}, shared={shared}")
            
            # Generate masks
            m1, m2, ovlp = create_masks(d_hs, k, shared, mask_type, key)
            
            # Print results
            print(f"Mask1: {m1.astype(int)}")
            print(f"Mask2: {m2.astype(int)}")
            print(f"Overlap: {ovlp:.1f}% (Target: {target_ovlp}%)")
            
            # Automated checks
            assert jnp.sum(m1) == k, f"Mask1 sparsity violated! Expected {k} active"
            assert jnp.sum(m2) == k, f"Mask2 sparsity violated! Expected {k} active"
            assert abs(ovlp - target_ovlp) < 0.1, "Overlap percentage mismatch!"
            
            if mask_type == 'determ':
                # Additional deterministic checks
                actual_shared = jnp.sum(m1 * m2)
                assert actual_shared == shared, "Shared neuron count mismatch!"
                
            print("✓ All checks passed!")