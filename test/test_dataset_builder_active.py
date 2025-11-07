import pytest
import numpy as np
from datasets import InMemoryPDESet
from pairbuilder import build_supervised_pairs, build_triplets
from active_sampling import max_min_novelty, novelty_uncertainty_tradeoff

def make_dummy():
    N = 8
    M = 4
    C = 3
    B = 12
    y = np.stack([np.linspace(-1,1,N), np.linspace(1,-1,N)], axis=1).astype(np.float32)
    x = y[:M]
    f = np.random.RandomState(0).randn(B, N, C).astype(np.float32)
    indices = np.array([0,1,1,2,2,3,4,5], dtype=np.int32)
    indptr = np.array([0,2,4,6,8], dtype=np.int32)
    seq_ids = np.repeat(np.arange(3), 4)[:B]
    t_steps = np.tile(np.arange(4), 3)[:B]
    domains = np.random.RandomState(1).randint(0,2,size=B)
    ds = InMemoryPDESet(y, x, f, indices, indptr, seq_ids, t_steps, domains)
    return ds

def test_dataset_getitem_and_lengths():
    ds = make_dummy()
    y, x, f_y, csr, meta = ds[0]
    assert y.shape[1] == 2 and x.shape[1] == 2
    assert f_y.ndim == 2
    assert "indices" in csr and "indptr" in csr
    assert "seq_id" in meta and "t" in meta

def test_pairbuilder_targets_and_triplets():
    ds = make_dummy()
    seq_len = {int(s): 4 for s in np.unique(ds.seq_ids)}
    c_map = {int(s): 2.0 for s in np.unique(ds.seq_ids)}
    i, j, g = build_supervised_pairs(ds.seq_ids, ds.t_steps, seq_len, c_map, 16, 0)
    assert i.shape == j.shape == g.shape
    assert (g >= 0).all() and (g <= 1.0 + 1e-9).all()
    a, p, n = build_triplets(ds.seq_ids, ds.t_steps, 8, 1)
    assert a.shape == p.shape == n.shape

def test_active_sampling_scores():
    rs = np.random.RandomState(0)
    bank = rs.randn(5, 8).astype(np.float32)
    pool = rs.randn(20, 8).astype(np.float32)
    sel = max_min_novelty(bank, pool, 3)
    assert sel.shape == (3,)
    u = rs.rand(20).astype(np.float32)
    sel2 = novelty_uncertainty_tradeoff(bank, pool, u, 0.6, 5)
    assert sel2.shape == (5,)

