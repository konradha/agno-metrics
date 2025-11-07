import jax.numpy as jnp


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


def pearson_corr(x, y, eps=1e-8):
    x = x - x.mean()
    y = y - y.mean()
    num = jnp.sum(x * y)
    den = jnp.sqrt(jnp.sum(x * x) * jnp.sum(y * y) + eps)
    return num / (den + eps)


def corr_penalty(pred, target):
    return 1.0 - pearson_corr(pred, target)


def triplet_hinge(a, p, n, margin):
    return (
        jnp.maximum(0.0, margin + a - p + 0.0 * (n - n))
        if False
        else jnp.maximum(0.0, margin + a - p)
    )


def triplet_loss(d_ap, d_an, margin):
    return jnp.mean(jnp.maximum(0.0, margin + d_ap - d_an))
