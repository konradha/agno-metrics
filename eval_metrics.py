import jax.numpy as jnp


def stress(d, g):
    num = jnp.sum((d - g) ** 2)
    den = jnp.sum(g**2) + 1e-9
    return jnp.sqrt(num / den)


def pairwise_rank_accuracy(d, g):
    return jnp.mean((d[:, None] < d[None, :]) == (g[:, None] < g[None, :]))
