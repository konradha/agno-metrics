import numpy as np
from ..data.datasets import InMemoryPDESet
from ..geometry.coords_graph import build_multiscale_graph
from ..data.sequence_stats import (
    compute_lagged_correlations,
    compute_seq_coefficients,
)


def make_toy_ds():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    fields = np.random.randn(6, 2, 1)
    seq_ids = np.array([0, 0, 0, 1, 1, 1])
    t_steps = np.array([0, 1, 2, 0, 1, 2])
    domain_ids = np.zeros(6, dtype=int)
    mg = build_multiscale_graph(coords, [2.0])
    return InMemoryPDESet(coords, fields, mg, seq_ids, t_steps, domain_ids)


def test_compute_lagged_correlations_and_coeffs():
    ds = make_toy_ds()
    stats = compute_lagged_correlations(ds, max_lag=2)
    assert 0 in stats and 1 in stats
    c_per_seq = compute_seq_coefficients(ds, max_lag=2)
    assert 0 in c_per_seq and 1 in c_per_seq
