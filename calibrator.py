import numpy as np

class IsotonicCalibrator:
    def __init__(self):
        self.x_ = None
        self.y_ = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        y = self._pav(y)
        self.x_ = x
        self.y_ = y
        return self

    def _pav(self, y):
        y = y.copy()
        n = len(y)
        w = np.ones(n, dtype=np.float64)
        i = 0
        while i < n - 1:
            if y[i] <= y[i + 1]:
                i += 1
                continue
            j = i
            s = y[i] * w[i] + y[i + 1] * w[i + 1]
            wj = w[i] + w[i + 1]
            y[i] = s / wj
            w[i] = wj
            y = np.delete(y, i + 1)
            w = np.delete(w, i + 1)
            n -= 1
            while i > 0 and y[i - 1] > y[i]:
                s = y[i - 1] * w[i - 1] + y[i] * w[i]
                wj = w[i - 1] + w[i]
                y[i - 1] = s / wj
                w[i - 1] = wj
                y = np.delete(y, i)
                w = np.delete(w, i)
                n -= 1
                i -= 1
        y_expanded = np.repeat(y, w.astype(int))
        if y_expanded.shape[0] < w.sum():
            pad = int(w.sum() - y_expanded.shape[0])
            y_expanded = np.concatenate([y_expanded, np.full(pad, y_expanded[-1])])
        return y_expanded[:int(w.sum())]

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        idx = np.searchsorted(self.x_, x, side="left")
        idx = np.clip(idx, 1, len(self.x_) - 1)
        x0 = self.x_[idx - 1]
        x1 = self.x_[idx]
        y0 = self.y_[idx - 1]
        y1 = self.y_[idx]
        t = np.where(x1 > x0, (x - x0) / (x1 - x0), 0.0)
        return y0 + t * (y1 - y0)

