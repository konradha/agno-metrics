import flax.linen as nn
from .gaot_encoder import GAOTEncoder
from .latent_transformer import LatentTransformer
from .readout import GlobalReadout


class GAOTMetricModel(nn.Module):
    encoder_layers: int
    encoder_hidden_dim: int
    coord_dim: int = 2
    attention_dim: int = 64
    use_attention: bool = True
    attention_type: str = "cosine"
    transformer_layers: int = 0
    transformer_dim: int = 128
    transformer_heads: int = 4
    readout_hidden_dim: int = 128
    readout_out_dim: int = 128
    readout_method: str = "mean"

    @nn.compact
    def __call__(self, coords, fields, multiscale_csr, train=True):
        x = GAOTEncoder(
            num_layers=self.encoder_layers,
            hidden_dim=self.encoder_hidden_dim,
            coord_dim=self.coord_dim,
            attention_dim=self.attention_dim,
            use_attention=self.use_attention,
            attention_type=self.attention_type,
        )(coords, fields, multiscale_csr)
        if self.transformer_layers > 0:
            x = LatentTransformer(
                num_layers=self.transformer_layers,
                model_dim=self.transformer_dim,
                num_heads=self.transformer_heads,
            )(x, train=train)
        emb = GlobalReadout(
            method=self.readout_method,
            hidden_dim=self.readout_hidden_dim,
            out_dim=self.readout_out_dim,
        )(x)
        return emb
