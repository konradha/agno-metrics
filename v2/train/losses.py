import jax.numpy as jnp


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)


def pearson_corr(x, y, eps=1e-8):
    x = x - jnp.mean(x)
    y = y - jnp.mean(y)
    num = jnp.sum(x * y)
    den = jnp.sqrt(jnp.sum(x * x) * jnp.sum(y * y) + eps)
    return num / (den + eps)


def corr_penalty(pred, target):
    return 1.0 - pearson_corr(pred, target)


def triplet_hinge(d_ap, d_an, margin):
    return jnp.mean(jnp.maximum(0.0, margin + d_ap - d_an))


def combined_loss(d_ij, d_target, d_ap, d_an, w_mse, w_corr, w_triplet, margin):
    l_mse = mse_loss(d_ij, d_target)
    l_corr = corr_penalty(d_ij, d_target)
    l_trip = triplet_hinge(d_ap, d_an, margin)
    return w_mse * l_mse + w_corr * l_corr + w_triplet * l_trip
