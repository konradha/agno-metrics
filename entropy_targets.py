import numpy as np

def normalized_entropy_distance(w, c):
    w = np.asarray(w, dtype=np.float64)
    num = np.log10(np.power(10.0, c) * w + 1.0)
    den = np.log10(np.power(10.0, c) + 1.0)
    return (num / den).astype(np.float64)

def gaps_from_indices(t_i, t_j, length):
    t_i = np.asarray(t_i, dtype=np.int64)
    t_j = np.asarray(t_j, dtype=np.int64)
    w = np.abs(t_j - t_i) / float(max(1, length - 1))
    return w

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

def targets_from_sequences(idx_a, idx_b, seq_len, c):
    w = gaps_from_indices(idx_a, idx_b, seq_len)
    return normalized_entropy_distance(w, c)
