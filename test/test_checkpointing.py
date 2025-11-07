import pickle, json
import numpy as np
from checkpointing import (
    save_params,
    load_params,
    save_metrics,
    load_metrics,
    save_arrayz,
    load_arrayz,
)


class Dummy:
    def __init__(self, x):
        self.x = x


def test_save_load_params(tmp_path):
    p = Dummy(3)
    f = tmp_path / "ckpt" / "params.pkl"
    save_params(p, str(f))
    q = load_params(str(f))
    assert isinstance(q, Dummy) and q.x == 3


def test_save_load_metrics(tmp_path):
    m = {"a": 1.0, "b": 2.0}
    f = tmp_path / "logs" / "m.json"
    save_metrics(m, str(f))
    n = load_metrics(str(f))
    assert n == m


def test_save_load_arrayz(tmp_path):
    f = tmp_path / "arrs" / "z.npz"
    save_arrayz(str(f), x=np.arange(5), y=np.ones(3))
    d = load_arrayz(str(f))
    assert "x" in d and "y" in d
