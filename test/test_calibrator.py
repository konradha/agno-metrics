import pytest
import numpy as np
from calibrator import IsotonicCalibrator

def test_isotonic_monotone_and_interpolation():
    x = np.linspace(0, 1, 50)
    y = x**2 + 0.05*np.random.RandomState(0).randn(50)
    cal = IsotonicCalibrator().fit(x, y)
    z = cal.transform(x)
    assert np.all(np.diff(z) >= -1e-9)
    t = cal.transform(np.array([0.25, 0.75]))
    assert t.shape == (2,)

