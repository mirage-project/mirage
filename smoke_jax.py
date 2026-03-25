import jax
import jax.numpy as jnp

print("process_index:", jax.process_index())
print("local_device_count:", jax.local_device_count())
print("device_count:", jax.device_count())
print("devices:", jax.devices())

x = jnp.arange(8, dtype=jnp.float32)
y = x + 1
print("y:", y)