import json, os
import numpy as np
from eval_metrics import stress
from losses import pearson_corr


def compute_metrics(d_pred: np.ndarray, d_tgt: np.ndarray):
    m = {}
    m["mse"] = float(np.mean((d_pred - d_tgt) ** 2))
    m["mae"] = float(np.mean(np.abs(d_pred - d_tgt)))
    m["pearson"] = float(pearson_corr(d_pred, d_tgt))
    m["stress"] = float(stress(d_pred, d_tgt))
    return m


def log_jsonl(metrics: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")
