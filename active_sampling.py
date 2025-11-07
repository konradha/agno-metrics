import numpy as np

def max_min_novelty(emb_bank, emb_pool, k):
    if emb_bank.shape[0] == 0:
        d = np.linalg.norm(emb_pool[:, None, :] - 0.0, axis=-1).max(axis=1)
        return np.argsort(-d)[:k]
    dists = np.linalg.norm(emb_pool[:, None, :] - emb_bank[None, :, :], axis=-1)
    mm = dists.min(axis=1)
    return np.argsort(-mm)[:k]

def novelty_uncertainty_tradeoff(emb_bank, emb_pool, uncert_pool, alpha, k):
    idx = max_min_novelty(emb_bank, emb_pool, emb_pool.shape[0])
    mm = np.linalg.norm(emb_pool[:, None, :] - emb_bank[None, :, :], axis=-1).min(axis=1) if emb_bank.shape[0] else np.linalg.norm(emb_pool, axis=1)
    u = uncert_pool
    s = alpha * mm + (1 - alpha) * u
    return np.argsort(-s)[:k]

