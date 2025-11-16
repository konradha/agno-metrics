import jax.numpy as jnp
from ..models.mlp import MLP


def test_mlp_forward_shape():
    x = jnp.ones((4, 3))
    model = MLP((5, 7))
    params = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(params, x)
    assert y.shape == (4, 7)
