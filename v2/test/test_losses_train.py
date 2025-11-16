import numpy as np
import jax
import jax.numpy as jnp
from ..train.losses import (
    mse_loss,
    pearson_corr,
    corr_penalty,
    triplet_hinge,
    combined_loss,
)
from ..train.train_metric import create_train_state, train_step
from ..models.gaot_metric_model import GAOTMetricModel
from ..geometry.coords_graph import build_multiscale_graph


def test_losses_basic():
    pred = jnp.array([0.0, 0.5, 1.0])
    target = jnp.array([0.0, 0.4, 1.0])
    mse = mse_loss(pred, target)
    corr = pearson_corr(pred, target)
    pen = corr_penalty(pred, target)
    d_ap = jnp.array([0.2, 0.3])
    d_an = jnp.array([0.5, 0.6])
    th = triplet_hinge(d_ap, d_an, margin=0.1)
    total = combined_loss(pred, target, d_ap, d_an, 1.0, 1.0, 1.0, 0.1)
    assert mse >= 0.0
    assert -1.0 <= corr <= 1.0
    assert pen >= 0.0
    assert th >= 0.0
    assert total >= 0.0


def test_train_step_runs():
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    fields = jnp.ones((4, 2, 1))
    mg = build_multiscale_graph(np.array(coords), [1.5])
    model = GAOTMetricModel(
        encoder_layers=1,
        encoder_hidden_dim=8,
        coord_dim=2,
        attention_dim=8,
        use_attention=True,
        attention_type="cosine",
        transformer_layers=0,
        transformer_dim=8,
        transformer_heads=2,
        readout_hidden_dim=8,
        readout_out_dim=4,
        readout_method="mean",
    )
    sample_batch = (coords, fields[:2], mg)
    state = create_train_state(jax.random.PRNGKey(0), model, 1e-3, 0.0, sample_batch)
    batch = (coords, fields, mg)
    pair_indices = (jnp.array([0, 1]), jnp.array([2, 3]))
    pair_targets = jnp.array([0.2, 0.8])
    triplets = (jnp.array([0, 1]), jnp.array([1, 2]), jnp.array([2, 3]))
    loss_weights = (1.0, 0.5, 0.5)
    margin = 0.1
    metric_type = "l2"
    new_state, logs = train_step(
        state,
        batch,
        pair_indices,
        pair_targets,
        triplets,
        loss_weights,
        margin,
        metric_type,
    )
    assert "loss" in logs
