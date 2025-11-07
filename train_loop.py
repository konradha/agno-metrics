from typing import Callable, Dict
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from .config import TrainConfig, AGNOConfig, ReadoutConfig, MetricConfig
from .agno import AGNO
from .embedding_readout import Readout
from .metric_head import PairwiseMetric
from .losses import mse_loss, corr_penalty, triplet_loss


class Model(nn.Module):
    agno_cfg: AGNOConfig
    readout_cfg: ReadoutConfig
    metric_cfg: MetricConfig

    @nn.compact
    def __call__(self, y, x, f_y, csr):
        z = AGNO(
            self.agno_cfg.mlp_layers,
            self.agno_cfg.transform_type,
            self.agno_cfg.use_attn,
            self.agno_cfg.attention_type,
            self.agno_cfg.coord_dim,
            self.agno_cfg.attention_dim,
        )(y, csr, x, f_y, None, None)
        if z.ndim == 3:
            z = z
        else:
            z = z[None, ...]
        h = Readout(
            self.readout_cfg.method, self.readout_cfg.hidden, self.readout_cfg.out
        )(z)
        return h


def create_state(rng, model, tx, sample_batch):
    params = model.init(rng, *sample_batch)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train(
    ds_iter: Callable,
    target_pairs_iter: Callable,
    cfg: TrainConfig,
    agno_cfg: AGNOConfig,
    readout_cfg: ReadoutConfig,
    metric_cfg: MetricConfig,
    steps: int,
):
    model = Model(agno_cfg, readout_cfg, metric_cfg)
    tx = optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)
    y, x, f_y, csr = next(ds_iter())
    state = create_state(jax.random.PRNGKey(cfg.seed), model, tx, (y, x, f_y, csr))
    metric = PairwiseMetric(metric_cfg.type, readout_cfg.out)

    @jax.jit
    def step(state, batch, pair_targets, triplet_idx):
        y, x, f_y, csr = batch
        hu = model.apply(state.params, y, x, f_y, csr)
        hv = model.apply(state.params, y, x, f_y, csr)
        i, j = pair_targets
        di = metric(hu[i], hv[j])
        tg = jnp.asarray(triplet_idx)
        a, p, n = tg
        d_ap = metric(hu[a], hv[p])
        d_an = metric(hu[a], hv[n])
        loss = (
            mse_loss(di, pair_targets[2])
            + cfg.loss_corr * corr_penalty(di, pair_targets[2])
            + cfg.loss_triplet * triplet_loss(d_ap, d_an, cfg.margin)
        )
        grads = jax.grad(
            lambda p: step_loss(p, batch, pair_targets, triplet_idx, model, metric, cfg)
        )(state.params)
        return state.apply_gradients(grads=grads), loss

    def step_loss(params, batch, pairs, trip_idx, model, metric, cfg):
        y, x, f_y, csr = batch
        hu = model.apply(params, y, x, f_y, csr)
        hv = model.apply(params, y, x, f_y, csr)
        i, j, t = pairs
        di = metric(hu[i], hv[j])
        a, p, n = trip_idx
        d_ap = metric(hu[a], hv[p])
        d_an = metric(hu[a], hv[n])
        return (
            mse_loss(di, t)
            + cfg.loss_corr * corr_penalty(di, t)
            + cfg.loss_triplet * triplet_loss(d_ap, d_an, cfg.margin)
        )

    it = ds_iter()
    pt = target_pairs_iter()
    for _ in range(steps):
        batch = next(it)
        pairs = next(pt)
        trip = (pairs[0], pairs[1], pairs[1])
        state, loss = step(state, batch, pairs, trip)
    return state
