import os
import json
from flax.serialization import to_bytes, from_bytes


def save_params(params, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    b = to_bytes(params)
    with open(path, "wb") as f:
        f.write(b)


def load_params(path, empty_tree):
    with open(path, "rb") as f:
        b = f.read()
    params = from_bytes(empty_tree, b)
    return params


def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)
