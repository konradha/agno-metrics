import numpy as np

class InMemoryPDESet:
    def __init__(self, coords_y, coords_x, features, csr_indices, csr_indptr, seq_ids, t_steps, domain_ids):
        self.coords_y = coords_y
        self.coords_x = coords_x
        self.features = features
        self.csr_indices = csr_indices
        self.csr_indptr = csr_indptr
        self.seq_ids = np.asarray(seq_ids, dtype=np.int64)
        self.t_steps = np.asarray(t_steps, dtype=np.int64)
        self.domain_ids = np.asarray(domain_ids, dtype=np.int64)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, i):
        y = self.coords_y
        x = self.coords_x
        f_y = self.features[i]
        csr = {"indices": self.csr_indices, "indptr": self.csr_indptr}
        meta = {"seq_id": self.seq_ids[i], "t": self.t_steps[i], "domain": self.domain_ids[i]}
        return y, x, f_y, csr, meta

    def filter_by_seq(self, seq_id):
        mask = self.seq_ids == seq_id
        return np.nonzero(mask)[0]

    def indices_by_domain(self, domain_id):
        return np.nonzero(self.domain_ids == domain_id)[0]

