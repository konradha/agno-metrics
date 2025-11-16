import jax.numpy as jnp


def stress(d_pred, d_true):
    d_pred = jnp.asarray(d_pred)
    d_true = jnp.asarray(d_true)
    num = jnp.sum((d_pred - d_true) ** 2)
    den = jnp.sum(d_true**2) + 1e-9
    return jnp.sqrt(num / den)


def pairwise_rank_accuracy(d_pred, d_true):
    d_pred = jnp.asarray(d_pred)
    d_true = jnp.asarray(d_true)
    dp = d_pred[:, None]
    dt = d_true[:, None]
    pred_ord = dp < dp.T
    true_ord = dt < dt.T
    return jnp.mean(pred_ord == true_ord)


def basic_metrics(d_pred, d_true):
    d_pred = jnp.asarray(d_pred)
    d_true = jnp.asarray(d_true)
    mse = jnp.mean((d_pred - d_true) ** 2)
    mae = jnp.mean(jnp.abs(d_pred - d_true))
    x = d_pred - jnp.mean(d_pred)
    y = d_true - jnp.mean(d_true)
    num = jnp.sum(x * y)
    den = jnp.sqrt(jnp.sum(x * x) * jnp.sum(y * y) + 1e-8)
    pearson = num / (den + 1e-8)
    s = stress(d_pred, d_true)
    pra = pairwise_rank_accuracy(d_pred, d_true)
    return {
        "mse": float(mse),
        "mae": float(mae),
        "pearson": float(pearson),
        "stress": float(s),
        "pairwise_rank_accuracy": float(pra),
    }
