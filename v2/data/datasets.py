import numpy as np
from ..geometry.coords_graph import CSRGraph, MultiscaleCSR


class InMemoryPDESet:
    def __init__(self, coords, fields, multiscale_csr, seq_ids, t_steps, domain_ids):
        self.coords = np.asarray(coords)
        self.fields = np.asarray(fields)
        self.multiscale_csr = multiscale_csr
        self.seq_ids = np.asarray(seq_ids)
        self.t_steps = np.asarray(t_steps)
        self.domain_ids = np.asarray(domain_ids)

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, i):
        coords = self.coords
        field = self.fields[i]
        csr = self.multiscale_csr
        meta = {
            "seq_id": int(self.seq_ids[i]),
            "t": int(self.t_steps[i]),
            "domain": int(self.domain_ids[i]),
        }
        return coords, field, csr, meta


def load_pde_dataset(path):
    z = np.load(path, allow_pickle=True)
    coords = z["coords"]
    fields = z["fields"]
    seq_ids = z["seq_ids"]
    t_steps = z["t_steps"]
    domain_ids = z["domain_ids"]
    indices_keys = sorted([k for k in z.files if k.startswith("indices_")])
    indptr_keys = sorted([k for k in z.files if k.startswith("indptr_")])
    graphs = []
    for ik, pk in zip(indices_keys, indptr_keys):
        indices = z[ik]
        indptr = z[pk]
        graphs.append(CSRGraph(indices, indptr))
    multiscale_csr = MultiscaleCSR(graphs)
    return InMemoryPDESet(coords, fields, multiscale_csr, seq_ids, t_steps, domain_ids)


def save_pde_dataset(ds, path):
    data = {
        "coords": ds.coords,
        "fields": ds.fields,
        "seq_ids": ds.seq_ids,
        "t_steps": ds.t_steps,
        "domain_ids": ds.domain_ids,
    }
    if ds.multiscale_csr is not None:
        for k, graph in enumerate(ds.multiscale_csr.graphs):
            data[f"indices_{k}"] = graph.indices
            data[f"indptr_{k}"] = graph.indptr
    np.savez(path, **data)


def split_indices(ds, train_frac, val_frac, seed):
    n = len(ds)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return train_idx, val_idx, test_idx
