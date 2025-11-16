import numpy as np
import jax
import jax.numpy as jnp


def max_min_novelty(emb_bank, emb_pool, k):
    emb_bank = np.asarray(emb_bank)
    emb_pool = np.asarray(emb_pool)
    if emb_bank.shape[0] == 0:
        d = np.linalg.norm(emb_pool, axis=1)
        idx = np.argsort(-d)[:k]
        return idx
    dists = np.linalg.norm(emb_pool[:, None, :] - emb_bank[None, :, :], axis=-1)
    mm = dists.min(axis=1)
    idx = np.argsort(-mm)[:k]
    return idx


def novelty_uncertainty_tradeoff(emb_bank, emb_pool, uncert_pool, alpha, k):
    emb_bank = np.asarray(emb_bank)
    emb_pool = np.asarray(emb_pool)
    u = np.asarray(uncert_pool)
    if emb_bank.shape[0] == 0:
        s = u
        idx = np.argsort(-s)[:k]
        return idx
    dists = np.linalg.norm(emb_pool[:, None, :] - emb_bank[None, :, :], axis=-1)
    mm = dists.min(axis=1)
    s = alpha * mm + (1.0 - alpha) * u
    idx = np.argsort(-s)[:k]
    return idx


def run_active_sampling(
    params, model, ds_bank, ds_pool, batch_size, k, metric_type="l2", uncert_model=None
):
    import math

    coords = ds_bank.coords

    def encode_indices(ds, indices):
        indices = np.asarray(indices, dtype=np.int64)
        embs = []
        for start in range(0, indices.shape[0], batch_size):
            sub = indices[start : start + batch_size]
            fields = ds.fields[sub]
            emb = model.apply(params, coords, fields, ds.multiscale_csr, False)
            embs.append(np.array(emb))
        if len(embs) == 0:
            return np.zeros((0, model.readout_out_dim))
        return np.concatenate(embs, axis=0)

    bank_idx = np.arange(len(ds_bank))
    pool_idx = np.arange(len(ds_pool))
    emb_bank = encode_indices(ds_bank, bank_idx)
    emb_pool = encode_indices(ds_pool, pool_idx)
    if uncert_model is None:
        uncert_pool = np.zeros(emb_pool.shape[0], dtype=np.float32)
    else:
        uncert_pool = uncert_model(emb_pool)
    sel_local = max_min_novelty(emb_bank, emb_pool, k)
    return pool_idx[sel_local]
