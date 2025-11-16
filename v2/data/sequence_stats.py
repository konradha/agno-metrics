import numpy as np


def compute_lagged_correlations(ds, max_lag):
    seq_ids = np.asarray(ds.seq_ids)
    t_steps = np.asarray(ds.t_steps)
    fields = np.asarray(ds.fields)
    seq_to_idx = {}
    for i, s in enumerate(seq_ids):
        if s not in seq_to_idx:
            seq_to_idx[s] = []
        seq_to_idx[s].append(i)
    result = {}
    for s, idx_list in seq_to_idx.items():
        idx_arr = np.array(idx_list)
        order = np.argsort(t_steps[idx_arr])
        idx_arr = idx_arr[order]
        f_seq = fields[idx_arr]
        T = f_seq.shape[0]
        f_flat = f_seq.reshape(T, -1)
        f_flat = f_flat - f_flat.mean(axis=1, keepdims=True)
        lags = []
        corrs = []
        for lag in range(1, max_lag + 1):
            if lag >= T:
                break
            x = f_flat[:-lag]
            y = f_flat[lag:]
            num = np.sum(x * y, axis=1)
            den = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1) + 1e-12)
            c = num / (den + 1e-12)
            corrs.append(c.mean())
            lags.append(lag)
        if len(lags) == 0:
            lags_arr = np.array([1], dtype=np.float64)
            corrs_arr = np.array([0.0], dtype=np.float64)
        else:
            lags_arr = np.asarray(lags, dtype=np.float64)
            corrs_arr = np.asarray(corrs, dtype=np.float64)
        result[int(s)] = (lags_arr, corrs_arr)
    return result


def fit_entropy_rate_from_corr(lags, corr):
    lags = np.asarray(lags, dtype=np.float64)
    corr = np.asarray(corr, dtype=np.float64)
    mask = (lags > 0) & (corr > 0)
    if mask.sum() < 2:
        return 1.0
    x = lags[mask]
    y = np.log(corr[mask] + 1e-12)
    A = np.vstack([x, np.ones_like(x)]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    tau = -1.0 / max(1e-8, k)
    c = np.clip(1.0 / max(1e-6, tau), 0.1, 6.0)
    return float(c)


def compute_seq_coefficients(ds, max_lag):
    corr_stats = compute_lagged_correlations(ds, max_lag)
    c_per_seq = {}
    for s, (lags, corr) in corr_stats.items():
        c_per_seq[int(s)] = fit_entropy_rate_from_corr(lags, corr)
    return c_per_seq
