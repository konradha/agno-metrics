import jax.numpy as jnp
import numpy as np


def fourier_posenc(coords, num_freqs):
    f = jnp.arange(1, num_freqs + 1, dtype=coords.dtype)[None, :, None]
    phi = np.pi * coords[:, None, :]
    a = f * phi
    s = jnp.sin(a)
    c = jnp.cos(a)
    y = jnp.concatenate([s, c], axis=2)
    y = y.reshape((coords.shape[0], -1))
    return y


def time_posenc(times, num_freqs):
    t = times[:, None]
    f = jnp.arange(1, num_freqs + 1, dtype=t.dtype)[None, :]
    a = f * t
    s = jnp.sin(a)
    c = jnp.cos(a)
    y = jnp.concatenate([s, c], axis=1)
    return y
