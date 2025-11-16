import jax.numpy as jnp
from ..geometry.segment_ops import (
    csr_repeat_ids,
    segment_sum,
    segment_mean,
    segment_softmax,
)


def test_csr_repeat_ids_and_sum():
    indptr = jnp.array([0, 2, 5], dtype=jnp.int32)
    segids = csr_repeat_ids(indptr)
    values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = segment_sum(values, segids, 2)
    assert out.shape == (2,)
    assert float(out[0]) == 3.0
    assert float(out[1]) == 12.0


def test_segment_mean_and_softmax():
    indptr = jnp.array([0, 3, 5], dtype=jnp.int32)
    segids = csr_repeat_ids(indptr)
    scores = jnp.array([0.0, 1.0, 2.0, -1.0, 0.5])
    mean = segment_mean(scores, segids, 2)
    assert mean.shape == (2,)
    sm = segment_softmax(scores, segids, 2)
    assert sm.shape == scores.shape
    assert jnp.allclose(
        jnp.bincount(segids, weights=sm, length=2), jnp.ones(2), atol=1e-5
    )
