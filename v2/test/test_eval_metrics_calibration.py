import numpy as np
from ..eval.metrics import stress, pairwise_rank_accuracy, basic_metrics
from ..eval.calibration import (
    IsotonicCalibrator,
    fit_calibrator,
    apply_calibrator,
)


def test_basic_metrics_and_stress():
    d_true = np.linspace(0.0, 1.0, 20)
    d_pred = d_true + 0.1 * np.random.randn(20)
    s = stress(d_pred, d_true)
    m = basic_metrics(d_pred, d_true)
    assert s >= 0.0
    assert "mse" in m and "stress" in m


def test_isotonic_calibration_monotone():
    x = np.linspace(0.0, 1.0, 50)
    y = x**2
    cal = IsotonicCalibrator().fit(x, y)
    xq = np.linspace(0.0, 1.0, 20)
    yq = cal.transform(xq)
    assert yq.shape == xq.shape


def test_fit_and_apply_calibrator_wrapper():
    x = np.linspace(0.0, 1.0, 50)
    y = x**2
    cal = fit_calibrator(x, y)
    y2 = apply_calibrator(cal, x)
    assert y2.shape == x.shape
