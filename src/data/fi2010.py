"""FI-2010 limit-order-book dataset: loader, labels, and leakage-safe windowing.

FI-2010 (Ntakaris et al., 2018) is the standard public LOB benchmark. Each file is a ``(149, N_events)``
array: rows 0-39 are the 40 raw LOB features (10 levels x [ask price, ask size, bid price, bid size]); the
last 5 rows are the direction labels for prediction horizons ``k in {1, 2, 3, 5, 10}`` (values 1/2/3 = up /
stationary / down), which we map to classes ``{0, 1, 2}``.

A sample is the 100-event window ending at event ``e``: ``X = features[:, e-99 : e+1]`` reshaped to
``(1, 100, 40)``, with label = the horizon-``k`` label at event ``e``. Windows are built **within a single
contiguous array**, so a window never spans the official train/test boundary (load the Train and Test files as
separate datasets). The window uses only events up to and including ``e``; the label is the *future* move over
horizon ``k`` — that is the supervised target, not leakage.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["HORIZONS", "load_fi2010_file", "build_windows", "FI2010Dataset"]

# Column order of the 5 label rows at the end of an FI-2010 file.
HORIZONS = (1, 2, 3, 5, 10)
N_FEATURES = 40
WINDOW = 100


def load_fi2010_file(path: str) -> np.ndarray:
    """Load a raw FI-2010 text file into a ``(149, N_events)`` float array (whitespace-delimited)."""
    return np.loadtxt(path)


def _horizon_index(horizon: int) -> int:
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")
    return HORIZONS.index(horizon)


def _split_features_labels(data: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (features[:40], label_row) for a horizon, mapping labels to {0,1,2}."""
    if data.shape[0] < N_FEATURES + len(HORIZONS):
        raise ValueError(
            f"expected at least {N_FEATURES + len(HORIZONS)} rows, got {data.shape[0]}"
        )
    features = data[:N_FEATURES, :].astype(np.float32)
    label_row = data[-len(HORIZONS):, :][_horizon_index(horizon), :]
    labels = np.rint(label_row).astype(np.int64)
    offset = 1 if labels.min() >= 1 else 0  # FI-2010 ships labels as {1,2,3}
    return features, labels - offset


def build_windows(
    features: np.ndarray, labels: np.ndarray, window: int = WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize all windows (for small data / tests).

    Parameters
    ----------
    features : np.ndarray, shape (40, N)
        Feature rows (time along axis 1).
    labels : np.ndarray, shape (N,)
        Per-event class labels in {0,1,2}.
    window : int
        Window length (events).

    Returns
    -------
    X : np.ndarray, shape (N - window + 1, 1, window, 40)
    y : np.ndarray, shape (N - window + 1,)
        ``y[i]`` is the label at the window's **last** event ``i + window - 1`` (no look-ahead in features).
    """
    n = features.shape[1]
    if n < window:
        return np.empty((0, 1, window, N_FEATURES), np.float32), np.empty((0,), np.int64)
    n_win = n - window + 1
    feat_t = features.T  # (N, 40)
    X = np.empty((n_win, 1, window, N_FEATURES), dtype=np.float32)
    for i in range(n_win):
        X[i, 0] = feat_t[i : i + window]
    y = labels[window - 1 : window - 1 + n_win].astype(np.int64)
    return X, y


class FI2010Dataset(Dataset):
    """Lazy windowed FI-2010 dataset yielding ``(X: (1,100,40) float, y: long in {0,1,2})``."""

    def __init__(self, data: np.ndarray, horizon: int = 10, window: int = WINDOW):
        self.window = window
        self.horizon = horizon
        self.features, self.labels = _split_features_labels(data, horizon)
        self._feat_t = torch.from_numpy(self.features.T).contiguous()  # (N, 40)
        self._labels = torch.from_numpy(self.labels)
        self.n_events = self.features.shape[1]

    def __len__(self) -> int:
        return max(0, self.n_events - self.window + 1)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if i < 0 or i >= len(self):
            raise IndexError(i)
        x = self._feat_t[i : i + self.window].unsqueeze(0)   # (1, window, 40)
        y = self._labels[i + self.window - 1]                 # label at last event
        return x, y

    def class_counts(self) -> np.ndarray:
        """Count of each class over the windowed labels (for class-weighting)."""
        y = self.labels[self.window - 1 :]
        return np.bincount(y, minlength=3)
