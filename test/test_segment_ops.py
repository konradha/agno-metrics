import pytest
import jax.numpy as jnp
from jax_pde_metric.segment_ops import (
    csr_repeat_ids,
    segment_sum,
    segment_max,
    segment_mean,
    segment_softmax,
)


@pytest.mark.parametrize(
    "indptr,expected",
    [
        (
            jnp.array([0, 2, 5, 5], dtype=jnp.int32),
            jnp.array([0, 0, 1, 1, 1], dtype=jnp.int32),
        ),
        (
            jnp.array([0, 3, 6], dtype=jnp.int32),
            jnp.array([0, 0, 0, 1, 1, 1], dtype=jnp.int32),
        ),
    ],
)
def test_csr_repeat_ids(indptr, expected):
    segids = csr_repeat_ids(indptr)
    assert segids.dtype == expected.dtype
    assert segids.shape == expected.shape
    assert jnp.all(segids == expected)


def test_segment_reducers_sum_max_mean():
    indptr = jnp.array([0, 2, 5, 5], dtype=jnp.int32)
    segids = csr_repeat_ids(indptr)
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nsegs = indptr.shape[0] - 1

    s = segment_sum(x, segids, nsegs)
    assert jnp.allclose(s, jnp.array([3.0, 12.0, 0.0]))

    mx = segment_max(x, segids, nsegs)
    assert jnp.allclose(mx, jnp.array([2.0, 5.0, -jnp.inf]))

    mu = segment_mean(x[:, None], segids, nsegs)
    assert mu.shape == (nsegs, 1)
    assert jnp.allclose(mu[0, 0], 1.5)
    assert jnp.allclose(mu[1, 0], (3.0 + 4.0 + 5.0) / 3.0)
    assert jnp.allclose(mu[2, 0], 0.0)


def test_segment_softmax_normalizes_per_segment():
    indptr = jnp.array([0, 3, 6], dtype=jnp.int32)
    segids = csr_repeat_ids(indptr)
    scores = jnp.array([0.1, 0.2, 0.3, -1.0, 2.0, 0.0])
    nsegs = indptr.shape[0] - 1

    a = segment_softmax(scores, segids, nsegs)
    assert a.shape == scores.shape

    s1 = jnp.sum(a[indptr[0] : indptr[1]])
    s2 = jnp.sum(a[indptr[1] : indptr[2]])
    assert jnp.allclose(jnp.stack([s1, s2]), jnp.ones(2), atol=1e-6)
