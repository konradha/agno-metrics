import numpy as np
from ..data.entropy_targets import (
    normalized_gap,
    entropy_distance,
    targets_from_pairs,
)


def test_normalized_gap_range():
    w = normalized_gap(0, 5, 6)
    assert 0.0 <= w <= 1.0
    w2 = normalized_gap(5, 0, 6)
    assert w == w2


def test_entropy_distance_monotone():
    w = np.linspace(0.0, 1.0, 10)
    d = entropy_distance(w, c=1.0)
    assert d.shape == w.shape
    assert d[0] == 0.0
    assert d[-1] <= 1.0
    assert np.all(np.diff(d) >= -1e-8)


def test_targets_from_pairs_same_and_diff():
    seq_ids = np.array([0, 0, 1, 1])
    t_steps = np.array([0, 1, 0, 1])
    seq_len = {0: 2, 1: 2}
    c_per_seq = {0: 1.0, 1: 1.0}
    idx_a = np.array([0, 0, 2])
    idx_b = np.array([1, 2, 3])
    d = targets_from_pairs(idx_a, idx_b, seq_ids, t_steps, seq_len, c_per_seq)
    assert d.shape[0] == 3
    assert d[2] == 1.0
    assert d[0] < 1.0
