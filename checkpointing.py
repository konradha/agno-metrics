import os, json, pickle
import numpy as np


def save_params(params, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(params, f)


def load_params(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_metrics(metrics: dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f)


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def save_arrayz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)


def load_arrayz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}
