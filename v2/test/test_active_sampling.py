import numpy as np
from ..utils.active_sampling import max_min_novelty, novelty_uncertainty_tradeoff


def test_max_min_novelty_empty_bank():
    emb_bank = np.zeros((0, 4))
    emb_pool = np.random.randn(10, 4)
    idx = max_min_novelty(emb_bank, emb_pool, 3)
    assert idx.shape[0] == 3


def test_max_min_novelty_nonempty_bank():
    emb_bank = np.random.randn(5, 4)
    emb_pool = np.random.randn(10, 4)
    idx = max_min_novelty(emb_bank, emb_pool, 4)
    assert idx.shape[0] == 4


def test_novelty_uncertainty_tradeoff():
    emb_bank = np.random.randn(5, 4)
    emb_pool = np.random.randn(10, 4)
    uncert = np.random.rand(10)
    idx = novelty_uncertainty_tradeoff(emb_bank, emb_pool, uncert, 0.5, 3)
    assert idx.shape[0] == 3
