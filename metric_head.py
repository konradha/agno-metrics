import jax.numpy as jnp
import flax.linen as nn
from typing import Literal


class PairwiseMetric(nn.Module):
    metric: Literal["l2", "cosine", "mahalanobis"] = "l2"
    emb_dim: int = 128

    @nn.compact
    def __call__(self, hu, hv):
        if self.metric == "l2":
            return jnp.linalg.norm(hu - hv, axis=-1)
        if self.metric == "cosine":
            a = hu / jnp.clip(jnp.linalg.norm(hu, axis=-1, keepdims=True), 1e-9)
            b = hv / jnp.clip(jnp.linalg.norm(hv, axis=-1, keepdims=True), 1e-9)
            return 1.0 - jnp.sum(a * b, axis=-1)
        B = self.param(
            "B", nn.initializers.lecun_normal(), (self.emb_dim, self.emb_dim)
        )
        Au = jnp.dot(hu, B)
        Av = jnp.dot(hv, B)
        return jnp.linalg.norm(Au - Av, axis=-1)
