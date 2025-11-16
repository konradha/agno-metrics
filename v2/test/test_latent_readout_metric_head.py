import jax
import jax.numpy as jnp
from ..models.latent_transformer import LatentTransformer
from ..models.readout import GlobalReadout
from ..models.metric_head import pairwise_distance


def test_latent_transformer_forward():
    tokens = jnp.ones((2, 5, 16))
    model = LatentTransformer(num_layers=1, model_dim=16, num_heads=2)
    params = model.init(jax.random.PRNGKey(0), tokens, True)
    y = model.apply(params, tokens, True)
    assert y.shape == tokens.shape


def test_global_readout_and_metric_head():
    tokens = jnp.ones((2, 5, 16))
    readout = GlobalReadout(hidden_dim=32, out_dim=8)
    params = readout.init(jax.random.PRNGKey(0), tokens)
    emb = readout.apply(params, tokens)
    assert emb.shape == (2, 8)
    d = pairwise_distance(emb[0:1], emb[1:2], "l2")
    assert d.shape == (1,)
