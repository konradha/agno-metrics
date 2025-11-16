import numpy as np


def compute_seq_lengths(seq_ids, t_steps):
    seq_ids = np.asarray(seq_ids, dtype=np.int64)
    t_steps = np.asarray(t_steps, dtype=np.int64)
    seq_to_t = {}
    for s, t in zip(seq_ids, t_steps):
        s = int(s)
        if s not in seq_to_t:
            seq_to_t[s] = []
        seq_to_t[s].append(int(t))
    seq_len = {}
    for s, ts in seq_to_t.items():
        arr = np.sort(np.asarray(ts, dtype=np.int64))
        if arr.size == 0:
            seq_len[s] = 0
        else:
            seq_len[s] = int(arr.size)
    return seq_len
