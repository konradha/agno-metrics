from typing import Tuple
import jax.numpy as jnp
import jax
from jax import lax


def csr_rowcounts(indptr: jnp.ndarray) -> jnp.ndarray:
    return indptr[1:] - indptr[:-1]


def csr_repeat_ids(indptr: jnp.ndarray) -> jnp.ndarray:
    counts = csr_rowcounts(indptr)
    return jnp.repeat(jnp.arange(counts.shape[0], dtype=jnp.int32), counts)


def segment_sum(values, segids, nsegs):
    out = jnp.zeros((nsegs,) + values.shape[1:], values.dtype)
    return out.at[segids].add(values)


def segment_max(values, segids, nsegs):
    out = jnp.full((nsegs,) + values.shape[1:], -jnp.inf, values.dtype)
    return out.at[segids].max(values)


def segment_mean(values, segids, nsegs):
    s = segment_sum(values, segids, nsegs)
    cnt = jnp.bincount(segids, length=nsegs)
    cnt = jnp.clip(cnt, 1) if values.ndim == 1 else jnp.clip(cnt, 1)[:, None]
    return s / cnt


def segment_softmax(
    scores: jnp.ndarray, segids: jnp.ndarray, nsegs: int
) -> jnp.ndarray:
    m = segment_max(scores, segids, nsegs)
    mexp = m[segids]
    x = scores - mexp
    ex = jnp.exp(x)
    z = segment_sum(ex, segids, nsegs)
    zexp = z[segids]
    return ex / (zexp + 1e-9)
