from typing import Sequence, Callable
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    sizes: Sequence[int]
    act: Callable = nn.gelu
    out_bias: bool = True

    @nn.compact
    def __call__(self, x):
        for i, s in enumerate(self.sizes):
            x = nn.Dense(s, use_bias=True)(x)
            if i < len(self.sizes) - 1:
                x = self.act(x)
        return x


class LinearChannelMLP(nn.Module):
    sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return MLP(self.sizes)(x)
