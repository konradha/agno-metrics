import numpy as np
from ..data.pair_sampling import (
    sample_pairs,
    build_supervised_pairs,
    build_triplets,
    batch_iterator,
)
from ..data.datasets import InMemoryPDESet
from ..geometry.coords_graph import build_multiscale_graph


def make_small_ds():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    fields = np.random.randn(4, 2, 1)
    seq_ids = np.array([0, 0, 1, 1])
    t_steps = np.array([0, 1, 0, 1])
    domain_ids = np.zeros(4, dtype=int)
    mg = build_multiscale_graph(coords, [2.0])
    return InMemoryPDESet(coords, fields, mg, seq_ids, t_steps, domain_ids)


def test_sample_pairs_shapes():
    ds = make_small_ds()
    i, j, same = sample_pairs(ds.seq_ids, ds.t_steps, 10, 0)
    assert i.shape == j.shape == same.shape
    assert i.shape[0] == 10


def test_build_supervised_pairs_targets():
    ds = make_small_ds()
    seq_len = {0: 2, 1: 2}
    c_per_seq = {0: 1.0, 1: 1.0}
    i, j, d = build_supervised_pairs(ds.seq_ids, ds.t_steps, seq_len, c_per_seq, 10, 0)
    assert i.shape == j.shape == d.shape
    assert np.all(d >= 0.0)
    assert np.all(d <= 1.0)


def test_build_triplets_shapes():
    ds = make_small_ds()
    a, p, n = build_triplets(ds.seq_ids, ds.t_steps, 7, 0)
    assert a.shape == p.shape == n.shape
    assert a.shape[0] == 7


def test_batch_iterator_yields():
    ds = make_small_ds()
    it = batch_iterator(ds, np.arange(len(ds)), 2, 0)
    coords, fields, csr = next(it)
    assert coords.shape[0] == ds.coords.shape[0]
    assert fields.shape[0] == 2
