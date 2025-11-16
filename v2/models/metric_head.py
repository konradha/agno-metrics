import jax.numpy as jnp


def pairwise_distance(hu, hv, metric="l2"):
    if metric == "l2":
        return jnp.linalg.norm(hu - hv, axis=-1)
    if metric == "cosine":
        a = hu / jnp.clip(jnp.linalg.norm(hu, axis=-1, keepdims=True), 1e-9)
        b = hv / jnp.clip(jnp.linalg.norm(hv, axis=-1, keepdims=True), 1e-9)
        return 1.0 - jnp.sum(a * b, axis=-1)
    return jnp.linalg.norm(hu - hv, axis=-1)
