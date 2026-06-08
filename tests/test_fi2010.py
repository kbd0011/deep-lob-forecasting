"""Tests for the FI-2010 loader and leakage-safe windowing (no data file, tiny synthetic array)."""
import numpy as np
import torch
from src.data.fi2010 import HORIZONS, FI2010Dataset, build_windows


def _synthetic(n_events=12):
    """Build a (45, N) FI-2010-like array: rows 0-39 features (feature f, time t = f + 0.001*t),
    last 5 rows are per-horizon labels with a distinct, checkable pattern."""
    rows = 40 + len(HORIZONS)
    data = np.zeros((rows, n_events), dtype=float)
    for f in range(40):
        data[f, :] = f + 0.001 * np.arange(n_events)
    # Label rows (last 5): horizon h gets values that encode (event index, horizon position) -> in {1,2,3}.
    for hi in range(len(HORIZONS)):
        data[40 + hi, :] = (np.arange(n_events) + hi) % 3 + 1  # values 1..3
    return data


def test_build_windows_shapes_and_feature_content():
    data = _synthetic(n_events=12)
    features = data[:40].astype(np.float32)
    labels = (data[-5:][4] - 1).astype(np.int64)  # horizon=10 (last row), mapped to {0,1,2}
    X, y = build_windows(features, labels, window=4)
    assert X.shape == (12 - 4 + 1, 1, 4, 40)       # 9 windows
    assert y.shape == (9,)
    # X[0,0] should be features[:, 0:4].T -> X[0,0,j,f] == features[f, j].
    assert X[0, 0, 0, 5] == np.float32(features[5, 0])
    assert X[0, 0, 3, 7] == np.float32(features[7, 3])
    # Window 2 starts at event 2.
    assert X[2, 0, 0, 9] == np.float32(features[9, 2])


def test_label_aligns_to_last_event_of_window():
    data = _synthetic(n_events=12)
    features = data[:40].astype(np.float32)
    raw_label_row = data[-5:][4]                 # horizon=10
    labels = (raw_label_row - 1).astype(np.int64)
    X, y = build_windows(features, labels, window=4)
    # y[i] must equal the label at event i+window-1 (the LAST event in the window), not the first.
    for i in range(len(y)):
        assert y[i] == labels[i + 4 - 1]


def test_dataset_matches_build_windows_and_horizon_selection():
    data = _synthetic(n_events=15)
    ds = FI2010Dataset(data, horizon=10, window=5)
    assert len(ds) == 15 - 5 + 1
    x0, y0 = ds[0]
    assert x0.shape == (1, 5, 40)
    assert isinstance(y0.item(), int)
    # Horizon selection: horizon=1 uses the FIRST of the 5 label rows, horizon=10 the LAST.
    ds_h1 = FI2010Dataset(data, horizon=1, window=5)
    last_event = 5 - 1
    assert ds_h1[0][1].item() == int((data[-5:][0][last_event]) - 1)
    assert ds[0][1].item() == int((data[-5:][4][last_event]) - 1)


def test_classes_in_range_and_counts():
    data = _synthetic(n_events=20)
    ds = FI2010Dataset(data, horizon=5, window=4)
    ys = torch.stack([ds[i][1] for i in range(len(ds))])
    assert ys.min() >= 0 and ys.max() <= 2
    counts = ds.class_counts()
    assert counts.sum() == len(ds)


def test_too_short_returns_empty():
    data = _synthetic(n_events=3)
    ds = FI2010Dataset(data, horizon=10, window=10)
    assert len(ds) == 0
