import pytest
import numpy as np
from entropy_targets import (
    normalized_entropy_distance,
    gaps_from_indices,
    fit_entropy_rate_from_corr,
    targets_from_sequences,
)


def test_entropy_distance_range_and_order():
    w = np.linspace(0, 1, 11)
    d = normalized_entropy_distance(w, c=2.0)
    assert d.min() >= 0.0 and d.max() <= 1.0 + 1e-9
    assert np.all(np.diff(d) >= -1e-9)


def test_gaps_from_indices():
    i = np.array([0, 5, 9])
    j = np.array([9, 5, 0])
    w = gaps_from_indices(i, j, 10)
    assert np.allclose(w, np.array([1.0, 0.0, 1.0]))


def test_fit_entropy_rate_from_corr():
    lags = np.arange(1, 6)
    corr = np.exp(-lags / 3.0)
    c = fit_entropy_rate_from_corr(lags, corr)
    assert c > 0


def test_targets_from_sequences_shape_and_bounds():
    i = np.array([0, 3, 5])
    j = np.array([5, 4, 5])
    t = targets_from_sequences(i, j, 6, 2.0)
    assert t.shape == (3,)
    assert (t >= 0).all() and (t <= 1.0 + 1e-9).all()
