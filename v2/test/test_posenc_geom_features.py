import numpy as np
import jax.numpy as jnp
from ..geometry.coords_graph import build_radius_graph, MultiscaleCSR
from ..geometry.posenc import fourier_posenc
from ..geometry.geom_features import (
    local_geometry_features,
    build_geometry_embedding,
)


def test_fourier_posenc_shape():
    coords = jnp.array([[0.0, 0.0], [0.5, -0.5]])
    pe = fourier_posenc(coords, 3)
    assert pe.shape[0] == coords.shape[0]
    assert pe.shape[1] == coords.shape[1] * 2 * 3


def test_local_geometry_features_shape():
    coords = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.3, 0.0]])
    g = build_radius_graph(np.array(coords), 0.2)
    feats = local_geometry_features(coords, g)
    assert feats.shape[0] == coords.shape[0]


def test_build_geometry_embedding_shape():
    coords = jnp.array([[0.0, 0.0], [0.1, 0.0], [0.3, 0.0]])
    g1 = build_radius_graph(np.array(coords), 0.2)
    g2 = build_radius_graph(np.array(coords), 0.5)
    mg = MultiscaleCSR([g1, g2])
    emb = build_geometry_embedding(coords, mg)
    assert emb.shape[0] == coords.shape[0]
    assert emb.shape[1] > 0
