import numpy as np
from ..data.datasets import InMemoryPDESet, save_pde_dataset, load_pde_dataset
from ..geometry.coords_graph import build_multiscale_graph


def test_inmemory_pdeset_len_and_get(tmp_path):
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    fields = np.random.randn(5, 2, 3)
    seq_ids = np.arange(5)
    t_steps = np.arange(5)
    domain_ids = np.zeros(5, dtype=int)
    mg = build_multiscale_graph(coords, [1.5])
    ds = InMemoryPDESet(coords, fields, mg, seq_ids, t_steps, domain_ids)
    assert len(ds) == 5
    c, f, csr, meta = ds[0]
    assert c.shape == coords.shape
    assert f.shape == fields[0].shape
    assert "seq_id" in meta


def test_save_and_load_dataset(tmp_path):
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    fields = np.random.randn(5, 2, 3)
    seq_ids = np.arange(5)
    t_steps = np.arange(5)
    domain_ids = np.zeros(5, dtype=int)
    mg = build_multiscale_graph(coords, [1.5])
    ds = InMemoryPDESet(coords, fields, mg, seq_ids, t_steps, domain_ids)
    path = tmp_path / "ds.npz"
    save_pde_dataset(ds, str(path))
    ds2 = load_pde_dataset(str(path))
    assert np.allclose(ds2.coords, coords)
    assert ds2.fields.shape == fields.shape
