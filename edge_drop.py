import jax
import jax.numpy as jnp


def bernoulli_mask(rng, size, p_keep: float):
    return jax.random.bernoulli(rng, p_keep, shape=size)


def apply_edge_mask(values: jnp.ndarray, mask: jnp.ndarray):
    if values.ndim == 1:
        return values * mask.astype(values.dtype)
    else:
        return values * mask.astype(values.dtype)[..., None]
