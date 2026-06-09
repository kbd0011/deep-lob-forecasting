"""Train DeepLOB on the real FI-2010 benchmark and report macro-F1 by prediction horizon.

Loads the canonical FI-2010 ZScore CF_7 split (149 x N arrays: 40 LOB features + 5 horizon label rows),
trains a DeepLOB classifier per horizon, and reports macro-F1 on the held-out test set. For a CPU/MPS-feasible
demonstration the event axis is subsampled (disclosed); scale EVENTS up for full-paper replication.

FI-2010 label rows map to prediction horizons k ∈ {10, 20, 30, 50, 100} events.

Usage (from repo root):  python scripts/run_fi2010.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.data.fi2010 import FI2010Dataset
from src.train import TrainConfig, build_model, train

TRAIN_EVENTS = 50_000   # subsample of 254,750 for a tractable demo
TEST_EVENTS = 20_000    # subsample of 55,478
EPOCHS = 6
# FI-2010 label-row index (via the loader's HORIZONS tuple) -> actual prediction horizon k.
HORIZON_TO_K = {1: 10, 3: 30, 5: 50, 10: 100}


def fast_load(path: str, max_events: int | None = None) -> np.ndarray:
    """Fast (149, N) loader: np.fromstring per line (C-level), far quicker than np.loadtxt on 600MB."""
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(np.fromstring(line, sep=" ", dtype=np.float32))
    arr = np.vstack(rows)
    return arr[:, :max_events] if max_events else arr


def main() -> None:
    warnings.filterwarnings("ignore")
    data = Path("data")
    train_arr = fast_load(str(data / "Train_Dst_NoAuction_ZScore_CF_7.txt"), TRAIN_EVENTS)
    test_arr = fast_load(str(data / "Test_Dst_NoAuction_ZScore_CF_7.txt"), TEST_EVENTS)
    print(f"loaded train {train_arr.shape}, test {test_arr.shape}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    results = {}
    for hz, k in HORIZON_TO_K.items():
        tr = FI2010Dataset(train_arr, horizon=hz)
        te = FI2010Dataset(test_arr, horizon=hz)
        counts = tr.class_counts()
        torch.manual_seed(0)
        model = build_model("deeplob")
        cfg = TrainConfig(epochs=EPOCHS, batch_size=64, lr=1e-3, device=device, patience=3,
                          use_class_weights=True, num_workers=0)
        res = train(model, tr, te, cfg)
        results[k] = res["best_macro_f1"]
        print(f"k={k:3d}: test macro-F1={res['best_macro_f1']:.4f} "
              f"(epochs_ran={res['epochs_ran']}, train class balance={counts.tolist()})")

    # Figure: macro-F1 by prediction horizon.
    ks = list(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([str(k) for k in ks], [results[k] for k in ks], color="steelblue")
    ax.set_xlabel("prediction horizon k (events)")
    ax.set_ylabel("macro F1 (test)")
    ax.set_ylim(0, 1)
    ax.set_title(f"DeepLOB on FI-2010 (ZScore CF_7)\ntrain {TRAIN_EVENTS:,} / test {TEST_EVENTS:,} events "
                 f"(subset), {EPOCHS} epochs")
    for i, k in enumerate(ks):
        ax.text(i, results[k] + 0.02, f"{results[k]:.3f}", ha="center")
    fig.tight_layout()
    out = Path("assets/fi2010_deeplob_f1.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("\nSUMMARY macro-F1 by horizon:", {k: round(v, 4) for k, v in results.items()})
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
