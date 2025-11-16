import numpy as np


def normalized_gap(i, j, length):
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    w = np.abs(j - i) / float(max(1, length - 1))
    return w


def entropy_distance(w, c):
    w = np.asarray(w, dtype=np.float64)
    num = np.log10(np.power(10.0, c) * w + 1.0)
    den = np.log10(np.power(10.0, c) + 1.0)
    return (num / den).astype(np.float64)


def targets_from_pairs(idx_a, idx_b, seq_ids, t_steps, seq_len_dict, c_per_seq):
    idx_a = np.asarray(idx_a, dtype=np.int64)
    idx_b = np.asarray(idx_b, dtype=np.int64)
    seq_ids = np.asarray(seq_ids, dtype=np.int64)
    t_steps = np.asarray(t_steps, dtype=np.int64)
    d = np.zeros_like(idx_a, dtype=np.float64)
    same = seq_ids[idx_a] == seq_ids[idx_b]
    if same.any():
        sa = seq_ids[idx_a[same]]
        ta = t_steps[idx_a[same]]
        tb = t_steps[idx_b[same]]
        w = np.zeros_like(ta, dtype=np.float64)
        for k in range(sa.shape[0]):
            s = int(sa[k])
            L = int(seq_len_dict.get(s, 2))
            c = float(c_per_seq.get(s, 1.0))
            w[k] = normalized_gap(ta[k], tb[k], L)
        d_same = entropy_distance(w, c)
        d[same] = d_same
    d[~same] = 1.0
    return d
