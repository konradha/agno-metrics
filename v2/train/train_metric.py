import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from ..models.gaot_metric_model import GAOTMetricModel
from ..models.metric_head import pairwise_distance
from .losses import combined_loss


class MetricTrainState(TrainState):
    pass


def create_train_state(rng, model, learning_rate, weight_decay, sample_batch):
    params = model.init(rng, *sample_batch)
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    return MetricTrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_step(
    state,
    batch,
    pair_indices,
    pair_targets,
    triplets,
    loss_weights,
    margin,
    metric_type,
):
    coords, fields, multiscale_csr = batch
    i, j = pair_indices
    a, p, n = triplets
    w_mse, w_corr, w_triplet = loss_weights

    def loss_fn(params):
        emb = state.apply_fn(params, coords, fields, multiscale_csr, True)
        d_ij = pairwise_distance(emb[i], emb[j], metric_type)
        d_ap = pairwise_distance(emb[a], emb[p], metric_type)
        d_an = pairwise_distance(emb[a], emb[n], metric_type)
        loss = combined_loss(
            d_ij, pair_targets, d_ap, d_an, w_mse, w_corr, w_triplet, margin
        )
        return loss, {
            "loss": loss,
            "mse": jnp.mean((d_ij - pair_targets) ** 2),
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logs), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, logs


def train_loop(
    state,
    data_iter,
    pair_sampler,
    triplet_sampler,
    steps,
    loss_weights,
    margin,
    metric_type,
    log_fn=None,
):
    for step in range(steps):
        batch = next(data_iter)
        pair_indices, pair_targets = next(pair_sampler)
        triplets = next(triplet_sampler)
        state, logs = train_step(
            state,
            batch,
            pair_indices,
            pair_targets,
            triplets,
            loss_weights,
            margin,
            metric_type,
        )
        if log_fn is not None:
            log_fn(step, logs)
    return state
