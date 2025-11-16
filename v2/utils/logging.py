import os
import json
import time


def setup_experiment_dirs(base_dir):
    ts = time.strftime("%Y%m%d-%H%M%S")
    root = os.path.join(base_dir, ts)
    ckpt_dir = os.path.join(root, "checkpoints")
    log_dir = os.path.join(root, "logs")
    plot_dir = os.path.join(root, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return {
        "root": root,
        "ckpt_dir": ckpt_dir,
        "log_dir": log_dir,
        "plot_dir": plot_dir,
    }


def log_metrics(step, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = dict(metrics)
    rec["step"] = int(step)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
