import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_vectors_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

@jax.jit
def add_vectors(x, y):
    return pl.pallas_call(
        add_vectors_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
    )(x, y)

x = jnp.arange(8, dtype=jnp.int32)
y = jnp.arange(8, dtype=jnp.int32)
out = add_vectors(x, y)
print(out)