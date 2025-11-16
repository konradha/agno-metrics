import numpy as np
from .datasets import InMemoryPDESet
from ..geometry.coords_graph import build_multiscale_graph


def build_dataset_from_raw(raw_path, radii):
    z = np.load(raw_path, allow_pickle=True)
    coords = z["coords"]
    fields = z["fields"]
    seq_ids = z["seq_ids"]
    t_steps = z["t_steps"]
    domain_ids = z["domain_ids"]
    multiscale_csr = build_multiscale_graph(coords, radii)
    ds = InMemoryPDESet(coords, fields, multiscale_csr, seq_ids, t_steps, domain_ids)
    return ds


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
