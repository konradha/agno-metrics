import pytest
import jax.numpy as jnp
from jax import random
from jax_pde_metric.metric_head import PairwiseMetric


def test_metric_l2_values():
    key = random.PRNGKey(0)
    hu = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    hv = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    mod = PairwiseMetric(metric="l2", emb_dim=2)
    params = mod.init(key, hu, hv)
    d = mod.apply(params, hu, hv)
    expected = jnp.array([1.0, jnp.sqrt(2.0)])
    assert d.shape == (2,)
    assert jnp.allclose(d, expected, atol=1e-6)


def test_metric_cosine_range():
    key = random.PRNGKey(1)
    hu = jnp.array([[1.0, 0.0], [1.0, 0.0]], dtype=jnp.float32)
    hv = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    mod = PairwiseMetric(metric="cosine", emb_dim=2)
    params = mod.init(key, hu, hv)
    d = mod.apply(params, hu, hv)
    assert d.shape == (2,)
    assert jnp.allclose(d[0], 0.0, atol=1e-6)
    assert jnp.allclose(d[1], 1.0, atol=1e-6)


def test_metric_mahalanobis_shapes():
    key = random.PRNGKey(2)
    hu = jnp.ones((4, 3), dtype=jnp.float32)
    hv = jnp.zeros((4, 3), dtype=jnp.float32)
    mod = PairwiseMetric(metric="mahalanobis", emb_dim=3)
    params = mod.init(key, hu, hv)
    d = mod.apply(params, hu, hv)
    assert d.shape == (4,)
    assert jnp.all(d >= 0.0)
