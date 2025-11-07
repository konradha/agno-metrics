import numpy as np
from scipy.spatial import cKDTree
import os


def radius_csr(data: np.ndarray, queries: np.ndarray, radius: float):
    tree = cKDTree(data)
    lists = tree.query_ball_point(queries, r=radius)
    indptr = [0]
    indices = []
    for L in lists:
        indices.extend(L)
        indptr.append(len(indices))
    return np.asarray(indices, dtype=np.int32), np.asarray(indptr, dtype=np.int32)


def save_npz(
    path: str, indices: np.ndarray, indptr: np.ndarray, weights: np.ndarray = None
):
    if weights is None:
        np.savez(path, indices=indices, indptr=indptr)
    else:
        np.savez(path, indices=indices, indptr=indptr, weights=weights)


def load_npz(path: str):
    z = np.load(path)
    indices = z["indices"]
    indptr = z["indptr"]
    weights = z["weights"] if "weights" in z else None
    return indices, indptr, weights
