import pytest
import jax.numpy as jnp
from jax import random
from jax_pde_metric.agno import AGNO


def test_agno_unbatched_no_attention_shapes():
    key = random.PRNGKey(0)
    N, M, Dmlp = 5, 3, 6
    y = jnp.stack([jnp.linspace(-1, 1, N), jnp.linspace(1, -1, N)], axis=1)
    x = y[:M]
    indices = jnp.array([0, 1, 2, 2, 3, 4], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 4, 6], dtype=jnp.int32)
    csr = {"indices": indices, "indptr": indptr}
    f_y = None
    mdl = AGNO(
        mlp_sizes=(8, Dmlp), transform_type="linear", use_attn=False, coord_dim=2
    )
    params = mdl.init(key, y, csr, x, f_y, None, None)
    out = mdl.apply(params, y, csr, x, f_y, None, None)
    assert out.shape == (M, Dmlp)


@pytest.mark.parametrize("attn_type", ["cosine", "dot_product"])
def test_agno_batched_attention_shapes(attn_type):
    key = random.PRNGKey(1)
    N, M, C, B, Dmlp = 7, 4, 3, 2, 8
    y = jnp.stack([jnp.linspace(-1, 1, N), jnp.linspace(1, -1, N)], axis=1)
    x = y[:M]
    f_y = random.normal(key, (B, N, C))
    indices = jnp.array([0, 1, 1, 2, 2, 3, 4, 5], dtype=jnp.int32)
    indptr = jnp.array([0, 2, 4, 6, 8], dtype=jnp.int32)
    csr = {"indices": indices, "indptr": indptr}
    mdl = AGNO(
        mlp_sizes=(16, Dmlp),
        transform_type="linear",
        use_attn=True,
        attention_type=attn_type,
        coord_dim=2,
    )
    params = mdl.init(key, y, csr, x, f_y, None, None)
    out = mdl.apply(params, y, csr, x, f_y, None, None)
    assert out.shape == (B, M, Dmlp)
