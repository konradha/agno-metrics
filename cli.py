import argparse, os, numpy as np
from datasets import InMemoryPDESet
from pairbuilder import build_supervised_pairs
from entropy_targets import fit_entropy_rate_from_corr, targets_from_sequences
from calibrator import IsotonicCalibrator
from logging_utils import compute_metrics, log_jsonl
from viz import scatter_distance_vs_target, calibration_curve
from checkpointing import save_params, load_params, save_metrics


def cmd_calibrate(args):
    d = np.load(args.pred_path)
    g = np.load(args.tgt_path)
    cal = IsotonicCalibrator().fit(d, g)
    z = cal.transform(d)
    m = compute_metrics(z, g)
    save_params(cal, args.out_calibrator)
    save_metrics(m, args.out_metrics)
    if args.out_plot:
        calibration_curve(d, z, args.out_plot)


def cmd_evaluate(args):
    d = np.load(args.pred_path)
    g = np.load(args.tgt_path)
    m = compute_metrics(d, g)
    save_metrics(m, args.out_metrics)
    if args.out_plot:
        scatter_distance_vs_target(d, g, args.out_plot)


def cmd_curate(args):
    emb = np.load(args.emb_path)
    k = args.keep
    from active_sampling import max_min_novelty

    sel = max_min_novelty(emb, emb, k)
    np.save(args.selection_out, sel)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate")
    pc.add_argument("--pred-path", required=True)
    pc.add_argument("--tgt-path", required=True)
    pc.add_argument("--out-calibrator", required=True)
    pc.add_argument("--out-metrics", required=True)
    pc.add_argument("--out-plot", default=None)
    pc.set_defaults(func=cmd_calibrate)

    pe = sub.add_parser("evaluate")
    pe.add_argument("--pred-path", required=True)
    pe.add_argument("--tgt-path", required=True)
    pe.add_argument("--out-metrics", required=True)
    pe.add_argument("--out-plot", default=None)
    pe.set_defaults(func=cmd_evaluate)

    pr = sub.add_parser("curate")
    pr.add_argument("--emb-path", required=True)
    pr.add_argument("--keep", type=int, required=True)
    pr.add_argument("--selection-out", required=True)
    pr.set_defaults(func=cmd_curate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
