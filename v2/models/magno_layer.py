import jax.numpy as jnp
import flax.linen as nn
from ..geometry.segment_ops import csr_repeat_ids, segment_sum, segment_softmax


class MAGNOLayer(nn.Module):
    hidden_dim: int
    coord_dim: int = 2
    attention_dim: int = 64
    use_attention: bool = True
    attention_type: str = "cosine"

    @nn.compact
    def __call__(self, coords, feats, multiscale_csr, geom_embs=None):
        x = feats
        if geom_embs is not None:
            x = jnp.concatenate([x, geom_embs], axis=-1)
        outputs = []
        for graph in multiscale_csr.graphs:
            idx = graph.indices
            indptr = graph.indptr
            segids = csr_repeat_ids(jnp.asarray(indptr))
            x_rep = x[idx]
            c_rep = coords[idx]
            c_self = coords
            if self.use_attention:
                if self.attention_type == "dot_product":
                    q = nn.Dense(self.attention_dim, use_bias=False)(
                        c_self[:, : self.coord_dim]
                    )
                    k = nn.Dense(self.attention_dim, use_bias=False)(
                        c_rep[:, : self.coord_dim]
                    )
                    q_rep = jnp.repeat(q, indptr[1:] - indptr[:-1], axis=0)
                    scores = jnp.sum(q_rep * k, axis=-1) / jnp.sqrt(self.attention_dim)
                else:
                    q = c_self[:, : self.coord_dim]
                    k = c_rep[:, : self.coord_dim]
                    q_rep = jnp.repeat(q, indptr[1:] - indptr[:-1], axis=0)
                    qn = q_rep / jnp.clip(
                        jnp.linalg.norm(q_rep, axis=-1, keepdims=True), 1e-9
                    )
                    kn = k / jnp.clip(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-9)
                    scores = jnp.sum(qn * kn, axis=-1)
                a = segment_softmax(scores, segids, x.shape[0])
                x_msg = x_rep * a[:, None]
            else:
                x_msg = x_rep
            m = nn.Dense(self.hidden_dim)(x_msg)
            m = nn.gelu(m)
            m = nn.Dense(self.hidden_dim)(m)
            agg = segment_sum(m, segids, x.shape[0])
            outputs.append(agg)
        if len(outputs) == 1:
            out = outputs[0]
        else:
            out = jnp.stack(outputs, axis=0).sum(axis=0)
        out = nn.LayerNorm()(out)
        return out


class MAGNOStack(nn.Module):
    num_layers: int
    hidden_dim: int
    coord_dim: int = 2
    attention_dim: int = 64
    use_attention: bool = True
    attention_type: str = "cosine"

    @nn.compact
    def __call__(self, coords, feats, multiscale_csr, geom_embs=None):
        x = feats
        for _ in range(self.num_layers):
            x_res = x
            x = MAGNOLayer(
                hidden_dim=self.hidden_dim,
                coord_dim=self.coord_dim,
                attention_dim=self.attention_dim,
                use_attention=self.use_attention,
                attention_type=self.attention_type,
            )(coords, x, multiscale_csr, geom_embs)
            x = x + x_res
        return x
