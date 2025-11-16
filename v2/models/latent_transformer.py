import jax.numpy as jnp
import flax.linen as nn


class LatentTransformer(nn.Module):
    num_layers: int
    model_dim: int
    num_heads: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, tokens, train=True):
        x = tokens
        if x.ndim == 2:
            x = x[None, ...]
        for _ in range(self.num_layers):
            y = nn.LayerNorm()(x)
            y = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.model_dim,
                out_features=self.model_dim,
                dropout_rate=self.dropout_rate,
            )(y, deterministic=not train)
            x = x + y
            y2 = nn.LayerNorm()(x)
            y2 = nn.Dense(self.model_dim * 4)(y2)
            y2 = nn.gelu(y2)
            y2 = nn.Dense(self.model_dim)(y2)
            x = x + y2
        return x
