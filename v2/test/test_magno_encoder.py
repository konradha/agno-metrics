import numpy as np
import jax
import jax.numpy as jnp
from ..geometry.coords_graph import build_multiscale_graph
from ..models.magno_layer import MAGNOStack
from ..models.gaot_encoder import GAOTEncoder


def test_magno_stack_forward():
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    fields = jnp.ones((3, 4))
    mg = build_multiscale_graph(np.array(coords), [1.5])
    model = MAGNOStack(num_layers=2, hidden_dim=8)
    params = model.init(jax.random.PRNGKey(0), coords, fields, mg, None)
    y = model.apply(params, coords, fields, mg, None)
    assert y.shape == (3, 8)


def test_gaot_encoder_forward():
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    fields = jnp.ones((3, 2))
    mg = build_multiscale_graph(np.array(coords), [1.5])
    model = GAOTEncoder(num_layers=2, hidden_dim=8)
    params = model.init(jax.random.PRNGKey(0), coords, fields, mg)
    y = model.apply(params, coords, fields, mg)
    assert y.shape == (3, 8)
