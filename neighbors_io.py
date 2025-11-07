from dataclasses import dataclass
import jax.numpy as jnp
from typing import Optional, Dict, List, Tuple


@dataclass
class CSRNeighbors:
    indices: jnp.ndarray
    indptr: jnp.ndarray
    weights: Optional[jnp.ndarray] = None


def validate_csr(csr: CSRNeighbors, n_data: int, n_query: int) -> None:
    assert csr.indptr.shape[0] == n_query + 1
    assert csr.indices.dtype in (jnp.int32, jnp.int64)
    assert csr.indices.shape[0] == csr.indptr[-1]
    assert jnp.all((csr.indices >= 0) & (csr.indices < n_data))


def pack_multiscale(csrs: List[CSRNeighbors]) -> Dict[str, object]:
    return {
        "indices": [c.indices for c in csrs],
        "indptr": [c.indptr for c in csrs],
        "weights": [c.weights for c in csrs],
    }
