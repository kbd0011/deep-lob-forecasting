"""Failure analysis: where and when does LOB predictability actually hold?

Replicating DeepLOB's headline F1 is table stakes. The credible contribution is the candid breakdown:
predictability is much higher for **large-tick** instruments (a coarser grid is easier to forecast) and it
**decays over time** as markets adapt. This module computes macro-F1 by tick-size stratum and by year and
renders the two diagnostics, so a strong aggregate number is never reported without its caveats.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "macro_f1",
    "assign_tick_stratum",
    "f1_by_column",
    "f1_by_tick_and_year",
    "plot_strata",
]

TICK_LABELS = ("small", "medium", "large")


def macro_f1(y_true, y_pred) -> float:
    """Macro-averaged F1 over the three direction classes (robust to class imbalance)."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0))


def assign_tick_stratum(tick_size, labels: tuple[str, ...] = TICK_LABELS) -> np.ndarray:
    """Bucket instruments into ordered tick-size strata by rank thirds (small/medium/large).

    Rank-based so it always yields balanced, fully-populated strata even with tied tick sizes.
    """
    s = pd.Series(np.asarray(tick_size, dtype=float))
    ordinal = s.rank(method="first").to_numpy() - 1  # 0-based ordinal rank
    n_labels = len(labels)
    idx = np.minimum((ordinal * n_labels // len(s)).astype(int), n_labels - 1)
    return np.array([labels[i] for i in idx])


def f1_by_column(
    df: pd.DataFrame, group_col: str, y_true: str = "y_true", y_pred: str = "y_pred"
) -> pd.Series:
    """Macro-F1 within each group of ``group_col``."""
    out = {
        key: macro_f1(g[y_true].to_numpy(), g[y_pred].to_numpy())
        for key, g in df.groupby(group_col, observed=True)
    }
    return pd.Series(out, name="macro_f1").rename_axis(group_col)


def f1_by_tick_and_year(
    df: pd.DataFrame,
    tick_col: str = "tick_stratum",
    year_col: str = "year",
    y_true: str = "y_true",
    y_pred: str = "y_pred",
) -> tuple[pd.Series, pd.Series]:
    """Return (F1 by tick stratum, F1 by year). Tick strata are ordered small < medium < large."""
    by_tick = f1_by_column(df, tick_col, y_true, y_pred)
    ordered = [s for s in TICK_LABELS if s in by_tick.index]
    by_tick = by_tick.reindex(ordered)
    by_year = f1_by_column(df, year_col, y_true, y_pred).sort_index()
    return by_tick, by_year


def plot_strata(by_tick: pd.Series, by_year: pd.Series, out_path: str) -> str:
    """Save a two-panel figure: F1 by tick stratum (bars) and F1 decay by year (line)."""
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar([str(i) for i in by_tick.index], by_tick.to_numpy(), color="steelblue")
    axes[0].set_ylabel("macro F1")
    axes[0].set_title("Predictability by tick-size stratum")
    axes[0].set_ylim(0, 1)

    axes[1].plot([str(i) for i in by_year.index], by_year.to_numpy(), marker="o", color="firebrick")
    axes[1].set_xlabel("year")
    axes[1].set_ylabel("macro F1")
    axes[1].set_title("Temporal decay of predictability")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)
