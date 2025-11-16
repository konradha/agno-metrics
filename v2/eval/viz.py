import os
import numpy as np
import matplotlib.pyplot as plt


def scatter_distance_vs_target(d_pred, d_true, path=None):
    d_pred = np.asarray(d_pred)
    d_true = np.asarray(d_true)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(d_true, d_pred, s=8)
    ax.set_xlabel("target")
    ax.set_ylabel("predicted")
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def calibration_curve(raw_d, cal_d, path=None):
    raw_d = np.asarray(raw_d)
    cal_d = np.asarray(cal_d)
    idx = np.argsort(raw_d)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(raw_d[idx], cal_d[idx])
    ax.set_xlabel("raw distance")
    ax.set_ylabel("calibrated")
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig


def coverage_hist(min_dists, path=None):
    min_dists = np.asarray(min_dists)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(min_dists, bins=30)
    ax.set_xlabel("min distance to bank")
    ax.set_ylabel("count")
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig
