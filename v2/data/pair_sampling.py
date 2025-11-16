import numpy as np
from .entropy_targets import targets_from_pairs


def sample_pairs(seq_ids, t_steps, num_pairs, rng):
    seq_ids = np.asarray(seq_ids, dtype=np.int64)
    t_steps = np.asarray(t_steps, dtype=np.int64)
    n = seq_ids.shape[0]
    rs = np.random.RandomState(rng)
    i = rs.randint(0, n, size=num_pairs)
    j = rs.randint(0, n, size=num_pairs)
    same = seq_ids[i] == seq_ids[j]
    return i, j, same


def build_supervised_pairs(seq_ids, t_steps, seq_len_dict, c_per_seq, num_pairs, rng):
    i, j, same = sample_pairs(seq_ids, t_steps, num_pairs, rng)
    d = targets_from_pairs(i, j, seq_ids, t_steps, seq_len_dict, c_per_seq)
    return i, j, d


def build_triplets(seq_ids, t_steps, num_triplets, rng):
    seq_ids = np.asarray(seq_ids, dtype=np.int64)
    n = seq_ids.shape[0]
    rs = np.random.RandomState(rng)
    a = rs.permutation(n)[:num_triplets]
    p = np.empty(num_triplets, dtype=np.int64)
    n_idx = np.empty(num_triplets, dtype=np.int64)
    for k, ai in enumerate(a):
        same = np.nonzero(seq_ids == seq_ids[ai])[0]
        diff = np.nonzero(seq_ids != seq_ids[ai])[0]
        if same.size > 1:
            same = same[same != ai]
        if same.size == 0:
            p[k] = rs.randint(0, n)
        else:
            p[k] = rs.choice(same)
        if diff.size == 0:
            n_idx[k] = rs.randint(0, n)
        else:
            n_idx[k] = rs.choice(diff)
    return a, p, n_idx


def batch_iterator(ds, indices, batch_size, rng):
    indices = np.asarray(indices, dtype=np.int64)
    rs = np.random.RandomState(rng)
    while True:
        perm = rs.permutation(indices.shape[0])
        idx = indices[perm]
        for start in range(0, idx.shape[0], batch_size):
            sub = idx[start : start + batch_size]
            coords = ds.coords
            fields = ds.fields[sub]
            multiscale_csr = ds.multiscale_csr
            yield coords, fields, multiscale_csr
