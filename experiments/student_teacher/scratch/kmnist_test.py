import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from functools import partial

# Updated load_data function with batching
def load_data(batch_size=128):
    ds_builder = tfds.builder('kmnist')
    ds_builder.download_and_prepare()
    
    # Create batched datasets
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(
            split='train',
            batch_size=batch_size,
            shuffle_files=True
        )
    )
    test_ds = tfds.as_numpy(
        ds_builder.as_dataset(
            split='test',
            batch_size=batch_size
        )
    )
    
    # Filter function with reshaping
    def filter_and_preprocess(batch):
        mask = batch['label'] < 20
        images = batch['image'][mask].astype(jnp.float32) / 255.
        labels = batch['label'][mask]
        return {
            'image': images.reshape(-1, 28*28),
            'label': labels
        }
    
    return train_ds, test_ds, filter_and_preprocess

def scaled_normal_init(scale: float):
    def _init(key, shape, dtype=jnp.float32):
        return scale * jax.random.normal(key, shape, dtype)
    return _init

# Define model
class WideNetwork(nn.Module):
    hidden_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim,
            # kernel_init=scaled_normal_init(2 / jnp.sqrt(x.shape[-1]))
            )(x)
        x = nn.relu(x)
        # x /= jnp.sqrt(x.shape[-1])
        x = nn.Dense(20,
            # kernel_init=scaled_normal_init(1 / jnp.sqrt(x.shape[-1]))
            )(x)
        return x

# Train function
@partial(jax.jit, static_argnames=('apply_fn', 'num_classes'))
def train_step(apply_fn, state, batch, num_classes):
    def loss_fn(params):
        logits = apply_fn({'params': params}, batch['image'])
        one_hot = jax.nn.one_hot(batch['label'], num_classes)
        ce_loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        l2_loss = 0.0
        for p in jax.tree_util.tree_leaves(params):
            l2_loss += jnp.sum(p**2)
        l2_loss *= 1e-4  # small weight‐decay factor
        _acc = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
        return ce_loss + l2_loss, (ce_loss, _acc)
    
    (_, (loss, acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc

# Eval function
@partial(jax.jit, static_argnames=('apply_fn', 'num_classes') )
def eval_step(apply_fn, params, batch, num_classes):
    logits = apply_fn({'params': params}, batch['image'])
    one_hot = jax.nn.one_hot(batch['label'], num_classes)
    loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    return loss, acc

def main():
    # Hyperparameters
    num_epochs = 20
    learning_rate = 0.0001
    seed = 42
    num_classes = 20
    batch_size = 512
    hidden = 2000
    
    # Load data with batching
    train_ds, test_ds, preprocess_fn = load_data(batch_size)

    # print(f"Training samples: {len(train_ds['label'])}")
    # print(f"Testing samples: {len(test_ds['label'])}")
    
    # Initialize model
    model = WideNetwork(hidden_dim=hidden)
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((1, 28*28)))['params']
    
    # Initialize optimizer
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    
    # Training loop
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(num_epochs):
        # Train
        epoch_losses, epoch_accs = [], []
        for batch in train_ds:
            batch = preprocess_fn(batch)
            state, loss, acc = train_step(
                model.apply, state, batch, num_classes
            )
            epoch_losses.append(loss)
            epoch_accs.append(acc)
        train_losses.append(jnp.mean(jnp.array(epoch_losses)))
        train_accs.append(jnp.mean(jnp.array(epoch_accs)))


        
        # Eval epoch
        etest_losses, etest_accs = [], []
        for batch in test_ds:
            batch = preprocess_fn(batch)
            loss, acc = eval_step(
                model.apply, state.params, batch, num_classes
            )
            etest_losses.append(loss)
            etest_accs.append(acc)
        
        test_losses.append(jnp.mean(jnp.array(etest_losses)))
        test_accs.append(jnp.mean(jnp.array(etest_accs)))
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train: Loss={loss:.4f}, Acc={acc:.4f}")
        print(f"  Test:  Loss={test_losses[-1]:.4f}, Acc={test_accs[-1]:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('kmnist_training.png')
    plt.show()

if __name__ == "__main__":
    main()