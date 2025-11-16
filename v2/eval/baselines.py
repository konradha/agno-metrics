import numpy as np


def l2_baseline(field_u, field_v):
    u = np.asarray(field_u)
    v = np.asarray(field_v)
    diff = u - v
    return np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=-1)


def pcc_baseline(field_u, field_v):
    u = np.asarray(field_u)
    v = np.asarray(field_v)
    u_flat = u.reshape(u.shape[0], -1)
    v_flat = v.reshape(v.shape[0], -1)
    u_flat = u_flat - u_flat.mean(axis=1, keepdims=True)
    v_flat = v_flat - v_flat.mean(axis=1, keepdims=True)
    num = np.sum(u_flat * v_flat, axis=1)
    den = np.sqrt(
        np.sum(u_flat * u_flat, axis=1) * np.sum(v_flat * v_flat, axis=1) + 1e-12
    )
    return num / (den + 1e-12)


def compute_baseline_distances(ds, pair_indices):
    i, j = pair_indices
    fields_i = ds.fields[i]
    fields_j = ds.fields[j]
    d_l2 = l2_baseline(fields_i, fields_j)
    d_pcc = pcc_baseline(fields_i, fields_j)
    return {
        "l2": d_l2,
        "pcc": d_pcc,
    }
