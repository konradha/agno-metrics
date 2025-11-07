import os
import numpy as np
import matplotlib.pyplot as plt


def scatter_distance_vs_target(d_pred, d_tgt, path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(d_tgt, d_pred, s=8)
    ax.set_xlabel("target")
    ax.set_ylabel("predicted")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def calibration_curve(raw_d, cal_d, path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    idx = np.argsort(raw_d)
    ax.plot(raw_d[idx], cal_d[idx])
    ax.set_xlabel("raw distance")
    ax.set_ylabel("calibrated")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def coverage_hist(min_dists, path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(min_dists, bins=30)
    ax.set_xlabel("min distance to bank")
    ax.set_ylabel("count")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig
