import numpy as np
from entropy_targets import targets_from_sequences


def sample_pairs(meta_seq, meta_t, num_pairs, rng):
    n = meta_seq.shape[0]
    rs = np.random.RandomState(rng)
    i = rs.randint(0, n, size=num_pairs)
    j = rs.randint(0, n, size=num_pairs)
    same = meta_seq[i] == meta_seq[j]
    return i, j, same


def build_supervised_pairs(meta_seq, meta_t, seq_lengths, c_per_seq, num_pairs, rng):
    i, j, same = sample_pairs(meta_seq, meta_t, num_pairs, rng)
    c = np.array([c_per_seq.get(int(s), 1.0) for s in meta_seq[i]])
    L = np.array([seq_lengths.get(int(s), 2) for s in meta_seq[i]])
    g = np.zeros_like(i, dtype=np.float64)
    m = same
    if m.any():
        g[m] = targets_from_sequences(meta_t[i[m]], meta_t[j[m]], L[m], c[m])
    g[~m] = 1.0
    return i, j, g


def build_triplets(meta_seq, meta_t, num_triplets, rng):
    rs = np.random.RandomState(rng)
    a = rs.permutation(len(meta_t))[:num_triplets]
    p = np.empty(num_triplets, dtype=int)
    n = np.empty(num_triplets, dtype=int)
    for k, ai in enumerate(a):
        same = np.nonzero(meta_seq == meta_seq[ai])[0]
        diff = np.nonzero(meta_seq != meta_seq[ai])[0]
        p[k] = rs.choice(same) if same.size > 0 else rs.randint(0, len(meta_t))
        n[k] = rs.choice(diff) if diff.size > 0 else rs.randint(0, len(meta_t))
    return a, p, n
