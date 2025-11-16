import jax.numpy as jnp
import flax.linen as nn
from .mlp import MLP


class GlobalReadout(nn.Module):
    method: str = "mean"
    hidden_dim: int = 128
    out_dim: int = 128

    @nn.compact
    def __call__(self, tokens):
        if tokens.ndim == 2:
            z = tokens[None, ...]
        else:
            z = tokens
        if self.method == "mean":
            h = jnp.mean(z, axis=1)
        else:
            h = jnp.max(z, axis=1)
        h = MLP((self.hidden_dim, self.out_dim))(h)
        return h
