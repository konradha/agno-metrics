from typing import Literal
import jax.numpy as jnp
import flax.linen as nn


class Readout(nn.Module):
    method: Literal["mean", "max"] = "mean"
    hidden: int = 128
    out: int = 128

    @nn.compact
    def __call__(self, Z):
        if Z.ndim == 3:
            z = Z
        else:
            z = Z[None, ...]
        if self.method == "mean":
            h = jnp.mean(z, axis=1)
        else:
            h = jnp.max(z, axis=1)
        h = nn.Dense(self.hidden)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.out)(h)
        return h
