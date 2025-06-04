#!/bin/bash
#SBATCH --job-name=jax_gpu_test
#SBATCH --output=jax_test_%j.out
#SBATCH --error=jax_test_%j.err
#SBATCH --partition=gpu,gpulong  # Change to your GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1         # Request 1 GPU
#SBATCH --time=00:30:00  # 30 minute test
#SBATCH --mem=70G
#SBATCH --account=tau  ### Account used for job submission

# Load required modules (adjust for your cluster)
module purge
module load cuda/12.4.1

# load env
source /home/mtrappet/stu_teach/RepresentationSimilarityCL/rspy/bin/activate

# Test script
echo "Running JAX/Flax GPU test..."
python - <<EOF
import jax
import jax.numpy as jnp
from flax import linen as nn

print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

# Simple network test
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        return nn.Dense(10)(x)

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (1, 784))
model = MLP()
params = model.init(key, x)
print("Network initialized successfully!")

# GPU matrix mult test
x = jax.random.normal(key, (1000, 1000))
y = jax.numpy.dot(x, x.T)
print(f"Matrix multiplication result sum: {y.sum():.2f}")
print("All GPU tests completed successfully!")

# --- GPU Spec Reporting ---
print("\n=== JAX Device Info ===")
devices = jax.devices()
for idx, device in enumerate(devices):
    print(f"Device {idx}:")
    print(f"  Platform: {device.platform}")
    print(f"  Device Type: {device.device_kind}")
    print(f"  Memory: {device.client.device_memory_info().total / 1e9:.2f} GB (Total)")
    print(f"  Process Memory: {device.client.memory_usage() / 1e9:.2f} GB (Used)")

# --- Alternative: System-Level Info ---
try:
    import pynvml
    print("\n=== System GPU Info ===")
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU Name: {pynvml.nvmlDeviceGetName(handle)}")
    print(f"Total Memory: {mem_info.total/1e9:.2f} GB")
    print(f"Free Memory: {mem_info.free/1e9:.2f} GB")
    print(f"Used Memory: {mem_info.used/1e9:.2f} GB")
except ImportError:
    print("\nInstall pynvml for detailed system GPU info: pip install pynvml")

EOF

deactivate