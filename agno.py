from typing import Optional, Dict
import jax.numpy as jnp
import flax.linen as nn
from .mlp import MLP
from .segment_ops import csr_repeat_ids, segment_softmax, segment_sum


class AGNO(nn.Module):
    mlp_sizes: tuple
    transform_type: str = "linear"
    use_attn: bool = True
    attention_type: str = "cosine"
    coord_dim: int = 2
    attention_dim: int = 64

    @nn.compact
    def __call__(
        self,
        y,
        neighbors: Dict[str, jnp.ndarray],
        x: Optional[jnp.ndarray] = None,
        f_y: Optional[jnp.ndarray] = None,
        weights: Optional[jnp.ndarray] = None,
        drop_mask: Optional[jnp.ndarray] = None,
    ):
        if x is None:
            x = y
        idx = neighbors["indices"]
        indptr = neighbors["indptr"]
        segids = csr_repeat_ids(indptr)
        rep_y = y[idx]
        num_reps = indptr[1:] - indptr[:-1]
        self_x = jnp.repeat(x, num_reps, axis=0)

        if self.use_attn:
            if self.attention_type == "dot_product":
                q = nn.Dense(self.attention_dim, use_bias=False)(
                    self_x[:, : self.coord_dim]
                )
                k = nn.Dense(self.attention_dim, use_bias=False)(
                    rep_y[:, : self.coord_dim]
                )
                s = jnp.sum(q * k, axis=-1) / jnp.sqrt(self.attention_dim)
            else:
                qx = self_x[:, : self.coord_dim]
                ky = rep_y[:, : self.coord_dim]
                qn = qx / jnp.clip(jnp.linalg.norm(qx, axis=-1, keepdims=True), 1e-9)
                kn = ky / jnp.clip(jnp.linalg.norm(ky, axis=-1, keepdims=True), 1e-9)
                s = jnp.sum(qn * kn, axis=-1)
            a = segment_softmax(s, segids, int(x.shape[0]))
        else:
            a = None

        agg = jnp.concatenate([rep_y, self_x], axis=-1)
        if f_y is not None and (
            self.transform_type in ("nonlinear", "nonlinear_kernelonly")
        ):
            if f_y.ndim == 3:
                b = f_y[:, idx, :].reshape((-1, f_y.shape[-1]))
                B = f_y.shape[0]
                K = idx.shape[0]
                M = x.shape[0]
                rep_y = jnp.tile(rep_y, (B, 1))
                self_x = jnp.tile(self_x, (B, 1))
                agg = jnp.concatenate([rep_y, self_x, b], axis=-1)
                if a is not None:
                    a = jnp.tile(a, (B,))
                segids = (
                    jnp.tile(segids, (B,))
                    + jnp.repeat(jnp.arange(B, dtype=segids.dtype), K) * M
                )
            else:
                b = f_y[idx]
                agg = jnp.concatenate([agg, b], axis=-1)

        kmlp = MLP(self.mlp_sizes)
        m = kmlp(agg)

        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            if f_y.ndim == 3:
                B = f_y.shape[0]
                K = idx.shape[0]
                M = x.shape[0]
                fy = f_y[:, idx, :].reshape((-1, f_y.shape[-1]))
                if self.transform_type not in ("nonlinear", "nonlinear_kernelonly"):
                    m = jnp.tile(m, (B, 1))
                    if a is not None:
                        a = jnp.tile(a, (B,))
                    segids = (
                        jnp.tile(segids, (B,))
                        + jnp.repeat(jnp.arange(B, dtype=segids.dtype), K) * M
                    )
            else:
                fy = f_y[idx]
            if f_y is not None:
                if m.shape[-1] != fy.shape[-1]:
                    fy = nn.Dense(m.shape[-1], use_bias=False)(fy)
                m = m * fy

        if a is not None:
            m = m * a[:, None]

        if weights is not None:
            w = weights[idx]
            m = m * w[:, None]
            red = "sum"
        else:
            red = "sum" if a is not None else "mean"

        nsegs = (
            int(x.shape[0] * f_y.shape[0])
            if (f_y is not None and f_y.ndim == 3)
            else int(x.shape[0])
        )
        out = segment_sum(m, segids, nsegs)

        if red == "mean" and (a is None):
            counts = (indptr[1:] - indptr[:-1]).clip(1)
            if f_y is not None and f_y.ndim == 3:
                counts_b = jnp.tile(counts, (f_y.shape[0],))
                out = out / counts_b[:, None]
            else:
                out = out / counts[:, None]

        if f_y is not None and f_y.ndim == 3:
            out = out.reshape((f_y.shape[0], x.shape[0], -1))
        else:
            out = out.reshape((1, x.shape[0], -1)) if f_y is not None else out
        return out
