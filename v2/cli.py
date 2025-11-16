import argparse
import os
import json
import numpy as np
import jax
from .config import load_config, save_config
from .data.sequence_generation import build_dataset_from_raw, compute_seq_lengths
from .data.datasets import save_pde_dataset, load_pde_dataset, split_indices
from .data.sequence_stats import compute_seq_coefficients
from .data.pair_sampling import build_supervised_pairs, build_triplets, batch_iterator
from .eval.metrics import basic_metrics
from .eval.calibration import fit_calibrator, apply_calibrator
from .eval.viz import scatter_distance_vs_target, calibration_curve
from .eval.baselines import compute_baseline_distances
from .models.gaot_metric_model import GAOTMetricModel
from .train.train_metric import create_train_state, train_loop
from .utils.checkpointing import save_params, load_params, save_metrics
from .utils.logging import setup_experiment_dirs, log_metrics


def cmd_prepare_dataset(args):
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    radii = data_cfg["radii"]
    max_lag = data_cfg.get("max_lag", 10)
    raw_path = data_cfg["raw_path"]
    out_dir = data_cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    ds = build_dataset_from_raw(raw_path, radii)
    ds_path = os.path.join(out_dir, "dataset.npz")
    save_pde_dataset(ds, ds_path)
    c_per_seq = compute_seq_coefficients(ds, max_lag)
    seq_len = compute_seq_lengths(ds.seq_ids, ds.t_steps)
    with open(os.path.join(out_dir, "seq_coefficients.json"), "w") as f:
        json.dump(c_per_seq, f)
    with open(os.path.join(out_dir, "seq_lengths.json"), "w") as f:
        json.dump(seq_len, f)
    data_cfg["dataset_path"] = ds_path
    data_cfg["coefficients_path"] = os.path.join(out_dir, "seq_coefficients.json")
    data_cfg["seq_lengths_path"] = os.path.join(out_dir, "seq_lengths.json")
    save_config(cfg, args.config)


def make_pair_sampler(ds, indices, seq_len_dict, c_per_seq, num_pairs, seed):
    from .data.pair_sampling import build_supervised_pairs

    rs = np.random.RandomState(seed)
    seq_ids_sub = ds.seq_ids[indices]
    t_steps_sub = ds.t_steps[indices]
    while True:
        sub_seed = int(rs.randint(0, 2**31 - 1))
        i_local, j_local, d = build_supervised_pairs(
            seq_ids_sub, t_steps_sub, seq_len_dict, c_per_seq, num_pairs, sub_seed
        )
        i = indices[i_local]
        j = indices[j_local]
        yield (i, j), d


def make_triplet_sampler(ds, indices, num_triplets, seed):
    from .data.pair_sampling import build_triplets

    rs = np.random.RandomState(seed)
    seq_ids_sub = ds.seq_ids[indices]
    t_steps_sub = ds.t_steps[indices]
    while True:
        sub_seed = int(rs.randint(0, 2**31 - 1))
        a_local, p_local, n_local = build_triplets(
            seq_ids_sub, t_steps_sub, num_triplets, sub_seed
        )
        a = indices[a_local]
        p = indices[p_local]
        n = indices[n_local]
        yield (a, p, n)


def cmd_train_metric(args):
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    metric_cfg = cfg["metric"]
    out_cfg = cfg["log"]
    ds = load_pde_dataset(data_cfg["dataset_path"])
    with open(data_cfg["coefficients_path"], "r") as f:
        c_per_seq = json.load(f)
    with open(data_cfg["seq_lengths_path"], "r") as f:
        seq_len_dict = json.load(f)
    train_idx, val_idx, test_idx = split_indices(
        ds, data_cfg["train_frac"], data_cfg["val_frac"], train_cfg["seed"]
    )
    dirs = setup_experiment_dirs(out_cfg["base_dir"])
    log_path = os.path.join(dirs["log_dir"], "train_metrics.jsonl")
    batch_size = train_cfg["batch_size"]
    rng = jax.random.PRNGKey(train_cfg["seed"])
    coords = ds.coords
    fields_sample = ds.fields[train_idx[:batch_size]]
    multiscale_csr = ds.multiscale_csr
    model = GAOTMetricModel(
        encoder_layers=model_cfg["encoder_layers"],
        encoder_hidden_dim=model_cfg["encoder_hidden_dim"],
        coord_dim=model_cfg.get("coord_dim", 2),
        attention_dim=model_cfg.get("attention_dim", 64),
        use_attention=model_cfg.get("use_attention", True),
        attention_type=model_cfg.get("attention_type", "cosine"),
        transformer_layers=model_cfg.get("transformer_layers", 0),
        transformer_dim=model_cfg.get(
            "transformer_dim", model_cfg["encoder_hidden_dim"]
        ),
        transformer_heads=model_cfg.get("transformer_heads", 4),
        readout_hidden_dim=model_cfg.get("readout_hidden_dim", 128),
        readout_out_dim=model_cfg.get("readout_out_dim", 128),
        readout_method=model_cfg.get("readout_method", "mean"),
    )
    sample_batch = (coords, fields_sample, multiscale_csr)
    state = create_train_state(
        rng, model, train_cfg["learning_rate"], train_cfg["weight_decay"], sample_batch
    )
    data_iter = batch_iterator(ds, train_idx, batch_size, train_cfg["seed"])
    num_pairs = train_cfg.get("pairs_per_step", batch_size)
    num_triplets = train_cfg.get("triplets_per_step", batch_size)
    pair_sampler = make_pair_sampler(
        ds, train_idx, seq_len_dict, c_per_seq, num_pairs, train_cfg["seed"] + 1
    )
    triplet_sampler = make_triplet_sampler(
        ds, train_idx, num_triplets, train_cfg["seed"] + 2
    )
    loss_weights = (
        train_cfg["loss_mse"],
        train_cfg["loss_corr"],
        train_cfg["loss_triplet"],
    )
    metric_type = metric_cfg.get("type", "l2")
    margin = train_cfg["margin"]

    def log_fn(step, logs):
        log_metrics(step, {k: float(v) for k, v in logs.items()}, log_path)

    state = train_loop(
        state,
        data_iter,
        pair_sampler,
        triplet_sampler,
        train_cfg["steps"],
        loss_weights,
        margin,
        metric_type,
        log_fn,
    )
    ckpt_path = os.path.join(dirs["ckpt_dir"], "model_params.msgpack")
    save_params(state.params, ckpt_path)
    cfg["log"]["last_run_dir"] = dirs["root"]
    cfg["log"]["last_ckpt"] = ckpt_path
    save_config(cfg, args.config)


def cmd_eval_metric(args):
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    metric_cfg = cfg["metric"]
    out_cfg = cfg["log"]
    ds = load_pde_dataset(data_cfg["dataset_path"])
    with open(data_cfg["coefficients_path"], "r") as f:
        c_per_seq = json.load(f)
    with open(data_cfg["seq_lengths_path"], "r") as f:
        seq_len_dict = json.load(f)
    train_idx, val_idx, test_idx = split_indices(
        ds, data_cfg["train_frac"], data_cfg["val_frac"], train_cfg["seed"]
    )
    ckpt_path = out_cfg["last_ckpt"]
    dirs = {
        "root": out_cfg["last_run_dir"],
        "ckpt_dir": os.path.join(out_cfg["last_run_dir"], "checkpoints"),
        "log_dir": os.path.join(out_cfg["last_run_dir"], "logs"),
        "plot_dir": os.path.join(out_cfg["last_run_dir"], "plots"),
    }
    coords = ds.coords
    multiscale_csr = ds.multiscale_csr
    rng = jax.random.PRNGKey(train_cfg["seed"])
    fields_sample = ds.fields[train_idx[: train_cfg["batch_size"]]]
    model = GAOTMetricModel(
        encoder_layers=model_cfg["encoder_layers"],
        encoder_hidden_dim=model_cfg["encoder_hidden_dim"],
        coord_dim=model_cfg.get("coord_dim", 2),
        attention_dim=model_cfg.get("attention_dim", 64),
        use_attention=model_cfg.get("use_attention", True),
        attention_type=model_cfg.get("attention_type", "cosine"),
        transformer_layers=model_cfg.get("transformer_layers", 0),
        transformer_dim=model_cfg.get(
            "transformer_dim", model_cfg["encoder_hidden_dim"]
        ),
        transformer_heads=model_cfg.get("transformer_heads", 4),
        readout_hidden_dim=model_cfg.get("readout_hidden_dim", 128),
        readout_out_dim=model_cfg.get("readout_out_dim", 128),
        readout_method=model_cfg.get("readout_method", "mean"),
    )
    sample_batch = (coords, fields_sample, multiscale_csr)
    empty_state = create_train_state(
        rng, model, train_cfg["learning_rate"], train_cfg["weight_decay"], sample_batch
    )
    params = load_params(ckpt_path, empty_state.params)
    metric_type = metric_cfg.get("type", "l2")
    num_pairs_eval = cfg["eval"].get("num_pairs", 10000)
    rs = np.random.RandomState(train_cfg["seed"] + 3)
    test_perm = rs.permutation(test_idx.shape[0])
    use_idx = test_idx[test_perm[: min(num_pairs_eval, test_idx.shape[0])]]
    i = rs.randint(0, use_idx.shape[0], size=num_pairs_eval)
    j = rs.randint(0, use_idx.shape[0], size=num_pairs_eval)
    idx_i = use_idx[i]
    idx_j = use_idx[j]
    batch_size = train_cfg["batch_size"]

    def encode_indices(indices):
        indices = np.asarray(indices, dtype=np.int64)
        embs = []
        for start in range(0, indices.shape[0], batch_size):
            sub = indices[start : start + batch_size]
            fields = ds.fields[sub]
            emb = model.apply(params, coords, fields, multiscale_csr, False)
            embs.append(np.array(emb))
        if len(embs) == 0:
            return np.zeros((0, model_cfg.get("readout_out_dim", 128)))
        return np.concatenate(embs, axis=0)

    uniq = np.unique(np.concatenate([idx_i, idx_j], axis=0))
    emb_all = encode_indices(uniq)
    id_to_pos = {int(i): k for k, i in enumerate(uniq)}
    pos_i = np.array([id_to_pos[int(a)] for a in idx_i], dtype=np.int64)
    pos_j = np.array([id_to_pos[int(b)] for b in idx_j], dtype=np.int64)
    from .models.metric_head import pairwise_distance as pw

    d_pred = pw(emb_all[pos_i], emb_all[pos_j], metric_type)
    d_pred = np.array(d_pred)
    from .data.entropy_targets import targets_from_pairs

    d_true = targets_from_pairs(
        idx_i, idx_j, ds.seq_ids, ds.t_steps, seq_len_dict, c_per_seq
    )
    m = basic_metrics(d_pred, d_true)
    metrics_path = os.path.join(dirs["log_dir"], "eval_metrics.json")
    save_metrics(m, metrics_path)
    scatter_path = os.path.join(dirs["plot_dir"], "scatter_eval.png")
    scatter_distance_vs_target(d_pred, d_true, scatter_path)


def cmd_calibrate_metric(args):
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    metric_cfg = cfg["metric"]
    out_cfg = cfg["log"]
    ds = load_pde_dataset(data_cfg["dataset_path"])
    with open(data_cfg["coefficients_path"], "r") as f:
        c_per_seq = json.load(f)
    with open(data_cfg["seq_lengths_path"], "r") as f:
        seq_len_dict = json.load(f)
    train_idx, val_idx, test_idx = split_indices(
        ds, data_cfg["train_frac"], data_cfg["val_frac"], train_cfg["seed"]
    )
    ckpt_path = out_cfg["last_ckpt"]
    dirs = {
        "root": out_cfg["last_run_dir"],
        "ckpt_dir": os.path.join(out_cfg["last_run_dir"], "checkpoints"),
        "log_dir": os.path.join(out_cfg["last_run_dir"], "logs"),
        "plot_dir": os.path.join(out_cfg["last_run_dir"], "plots"),
    }
    coords = ds.coords
    multiscale_csr = ds.multiscale_csr
    rng = jax.random.PRNGKey(train_cfg["seed"])
    fields_sample = ds.fields[train_idx[: train_cfg["batch_size"]]]
    model = GAOTMetricModel(
        encoder_layers=model_cfg["encoder_layers"],
        encoder_hidden_dim=model_cfg["encoder_hidden_dim"],
        coord_dim=model_cfg.get("coord_dim", 2),
        attention_dim=model_cfg.get("attention_dim", 64),
        use_attention=model_cfg.get("use_attention", True),
        attention_type=model_cfg.get("attention_type", "cosine"),
        transformer_layers=model_cfg.get("transformer_layers", 0),
        transformer_dim=model_cfg.get(
            "transformer_dim", model_cfg["encoder_hidden_dim"]
        ),
        transformer_heads=model_cfg.get("transformer_heads", 4),
        readout_hidden_dim=model_cfg.get("readout_hidden_dim", 128),
        readout_out_dim=model_cfg.get("readout_out_dim", 128),
        readout_method=model_cfg.get("readout_method", "mean"),
    )
    sample_batch = (coords, fields_sample, multiscale_csr)
    empty_state = create_train_state(
        rng, model, train_cfg["learning_rate"], train_cfg["weight_decay"], sample_batch
    )
    params = load_params(ckpt_path, empty_state.params)
    metric_type = metric_cfg.get("type", "l2")
    num_pairs_val = cfg["eval"].get("num_pairs_val", 10000)
    rs = np.random.RandomState(train_cfg["seed"] + 4)
    val_perm = rs.permutation(val_idx.shape[0])
    use_idx = val_idx[val_perm[: min(num_pairs_val, val_idx.shape[0])]]
    i = rs.randint(0, use_idx.shape[0], size=num_pairs_val)
    j = rs.randint(0, use_idx.shape[0], size=num_pairs_val)
    idx_i = use_idx[i]
    idx_j = use_idx[j]
    batch_size = train_cfg["batch_size"]

    def encode_indices(indices):
        indices = np.asarray(indices, dtype=np.int64)
        embs = []
        for start in range(0, indices.shape[0], batch_size):
            sub = indices[start : start + batch_size]
            fields = ds.fields[sub]
            emb = model.apply(params, coords, multiscale_csr, False)
            embs.append(np.array(emb))
        if len(embs) == 0:
            return np.zeros((0, model_cfg.get("readout_out_dim", 128)))
        return np.concatenate(embs, axis=0)

    uniq = np.unique(np.concatenate([idx_i, idx_j], axis=0))
    emb_all = encode_indices(uniq)
    id_to_pos = {int(i): k for k, i in enumerate(uniq)}
    pos_i = np.array([id_to_pos[int(a)] for a in idx_i], dtype=np.int64)
    pos_j = np.array([id_to_pos[int(b)] for b in idx_j], dtype=np.int64)
    from .models.metric_head import pairwise_distance as pw

    d_pred = pw(emb_all[pos_i], emb_all[pos_j], metric_type)
    d_pred = np.array(d_pred)
    from .data.entropy_targets import targets_from_pairs

    d_true = targets_from_pairs(
        idx_i, idx_j, ds.seq_ids, ds.t_steps, seq_len_dict, c_per_seq
    )
    cal = fit_calibrator(d_pred, d_true)
    d_cal = apply_calibrator(cal, d_pred)
    m = basic_metrics(d_cal, d_true)
    metrics_path = os.path.join(dirs["log_dir"], "calibration_metrics.json")
    save_metrics(m, metrics_path)
    curve_path = os.path.join(dirs["plot_dir"], "calibration_curve.png")
    calibration_curve(d_pred, d_cal, curve_path)


def cmd_active_sampling(args):
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    out_cfg = cfg["log"]
    ds = load_pde_dataset(data_cfg["dataset_path"])
    ckpt_path = out_cfg["last_ckpt"]
    coords = ds.coords
    multiscale_csr = ds.multiscale_csr
    rng = jax.random.PRNGKey(train_cfg["seed"])
    fields_sample = ds.fields[: train_cfg["batch_size"]]
    model = GAOTMetricModel(
        encoder_layers=model_cfg["encoder_layers"],
        encoder_hidden_dim=model_cfg["encoder_hidden_dim"],
        coord_dim=model_cfg.get("coord_dim", 2),
        attention_dim=model_cfg.get("attention_dim", 64),
        use_attention=model_cfg.get("use_attention", True),
        attention_type=model_cfg.get("attention_type", "cosine"),
        transformer_layers=model_cfg.get("transformer_layers", 0),
        transformer_dim=model_cfg.get(
            "transformer_dim", model_cfg["encoder_hidden_dim"]
        ),
        transformer_heads=model_cfg.get("transformer_heads", 4),
        readout_hidden_dim=model_cfg.get("readout_hidden_dim", 128),
        readout_out_dim=model_cfg.get("readout_out_dim", 128),
        readout_method=model_cfg.get("readout_method", "mean"),
    )
    from .train.train_metric import create_train_state

    sample_batch = (coords, fields_sample, multiscale_csr)
    empty_state = create_train_state(
        rng, model, train_cfg["learning_rate"], train_cfg["weight_decay"], sample_batch
    )
    from .utils.checkpointing import load_params

    params = load_params(ckpt_path, empty_state.params)
    from .utils.active_sampling import run_active_sampling

    n_bank = args.bank_size
    n_pool = args.pool_size
    bank_indices = np.arange(0, min(n_bank, len(ds)))
    pool_indices = np.arange(min(n_bank, len(ds)), min(n_bank + n_pool, len(ds)))

    class BankView:
        def __init__(self, ds, indices):
            self.coords = ds.coords
            self.fields = ds.fields[indices]
            self.multiscale_csr = ds.multiscale_csr

    ds_bank = BankView(ds, bank_indices)
    ds_pool = BankView(ds, pool_indices)
    selected = run_active_sampling(
        params, model, ds_bank, ds_pool, train_cfg["batch_size"], args.k
    )
    out_path = args.out_selection
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, pool_indices[selected])


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare-dataset")
    pp.add_argument("--config", required=True)
    pp.set_defaults(func=cmd_prepare_dataset)
    pt = sub.add_parser("train-metric")
    pt.add_argument("--config", required=True)
    pt.set_defaults(func=cmd_train_metric)
    pe = sub.add_parser("eval-metric")
    pe.add_argument("--config", required=True)
    pe.set_defaults(func=cmd_eval_metric)
    pc = sub.add_parser("calibrate-metric")
    pc.add_argument("--config", required=True)
    pc.set_defaults(func=cmd_calibrate_metric)
    pa = sub.add_parser("active-sampling")
    pa.add_argument("--config", required=True)
    pa.add_argument("--bank-size", type=int, default=100)
    pa.add_argument("--pool-size", type=int, default=1000)
    pa.add_argument("--k", type=int, default=100)
    pa.add_argument("--out-selection", required=True)
    pa.set_defaults(func=cmd_active_sampling)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
