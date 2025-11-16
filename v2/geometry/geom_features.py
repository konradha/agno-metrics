import jax.numpy as jnp
from .segment_ops import csr_repeat_ids, segment_mean
from .posenc import fourier_posenc


def local_geometry_features(coords, csr_graph):
    idx = csr_graph.indices
    indptr = csr_graph.indptr
    segids = csr_repeat_ids(jnp.asarray(indptr))
    c_rep = coords[idx]
    c_self = coords
    c_self_rep = jnp.repeat(c_self, indptr[1:] - indptr[:-1], axis=0)
    diff = c_rep - c_self_rep
    dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)
    mean_diff = segment_mean(diff, segids, coords.shape[0])
    mean_dist = segment_mean(dist, segids, coords.shape[0])
    feats = jnp.concatenate([mean_diff, mean_dist], axis=-1)
    return feats


def build_geometry_embedding(coords, multiscale_csr):
    pos = fourier_posenc(coords, 4)
    feats = []
    for graph in multiscale_csr.graphs:
        g = local_geometry_features(coords, graph)
        feats.append(g)
    if len(feats) == 0:
        geom = pos
    else:
        geom = jnp.concatenate([pos] + feats, axis=-1)
    return geom
