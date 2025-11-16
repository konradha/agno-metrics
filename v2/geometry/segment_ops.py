import jax.numpy as jnp


def csr_rowcounts(indptr):
    return indptr[1:] - indptr[:-1]


def csr_repeat_ids(indptr):
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
    if values.ndim == 1:
        cnt = jnp.clip(cnt, 1)
    else:
        cnt = jnp.clip(cnt, 1)[:, None]
    return s / cnt


def segment_softmax(scores, segids, nsegs):
    m = segment_max(scores, segids, nsegs)
    mexp = m[segids]
    x = scores - mexp
    ex = jnp.exp(x)
    z = segment_sum(ex, segids, nsegs)
    zexp = z[segids]
    return ex / (zexp + 1e-9)
