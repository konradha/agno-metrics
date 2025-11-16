import jax.numpy as jnp
import flax.linen as nn
from .magno_layer import MAGNOStack
from ..geometry.geom_features import build_geometry_embedding


class GAOTEncoder(nn.Module):
    num_layers: int
    hidden_dim: int
    coord_dim: int = 2
    attention_dim: int = 64
    use_attention: bool = True
    attention_type: str = "cosine"

    @nn.compact
    def __call__(self, coords, fields, multiscale_csr):
        geom = build_geometry_embedding(coords, multiscale_csr)
        x = jnp.concatenate([fields, geom], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = MAGNOStack(
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            coord_dim=self.coord_dim,
            attention_dim=self.attention_dim,
            use_attention=self.use_attention,
            attention_type=self.attention_type,
        )(coords, x, multiscale_csr, geom)
        return x
