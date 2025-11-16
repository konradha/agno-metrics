import numpy as np


class CSRGraph:
    def __init__(self, indices, indptr, weights=None):
        self.indices = np.asarray(indices)
        self.indptr = np.asarray(indptr)
        self.weights = None if weights is None else np.asarray(weights)


class MultiscaleCSR:
    def __init__(self, graphs):
        self.graphs = list(graphs)


def build_radius_graph(coords, radius):
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    lists = tree.query_ball_point(coords, r=radius)
    indices = []
    indptr = [0]
    for lst in lists:
        indices.extend(lst)
        indptr.append(len(indices))
    return CSRGraph(
        np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)
    )


def build_multiscale_graph(coords, radii):
    graphs = []
    for r in radii:
        graphs.append(build_radius_graph(coords, r))
    return MultiscaleCSR(graphs)
