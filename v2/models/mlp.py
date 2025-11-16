import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    sizes: tuple

    @nn.compact
    def __call__(self, x):
        for i, s in enumerate(self.sizes):
            x = nn.Dense(s)(x)
            if i < len(self.sizes) - 1:
                x = nn.gelu(x)
        return x
