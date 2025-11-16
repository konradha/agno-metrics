import numpy as np
from ..geometry.coords_graph import (
    build_radius_graph,
    build_multiscale_graph,
    CSRGraph,
    MultiscaleCSR,
)


def test_build_radius_graph_basic():
    coords = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0]])
    g = build_radius_graph(coords, 0.2)
    assert isinstance(g, CSRGraph)
    assert g.indices.ndim == 1
    assert g.indptr.shape[0] == coords.shape[0] + 1


def test_build_multiscale_graph_basic():
    coords = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0]])
    mg = build_multiscale_graph(coords, [0.2, 1.0])
    assert isinstance(mg, MultiscaleCSR)
    assert len(mg.graphs) == 2
