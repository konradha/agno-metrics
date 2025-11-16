import json
import os


def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def save_config(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
