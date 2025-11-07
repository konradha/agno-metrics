import jax.numpy as jnp
import numpy as np


def node_pos_encode(x: jnp.ndarray, freq: int = 4) -> jnp.ndarray:
    f = jnp.arange(1, freq + 1, dtype=x.dtype)[None, :, None]
    phi = np.pi * (x + 1.0)
    a = f * phi[:, None, :]
    s = jnp.sin(a)
    c = jnp.cos(a)
    y = jnp.concatenate([s, c], axis=2)
    y = y.reshape((x.shape[0], -1))
    return y
