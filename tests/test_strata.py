"""Tests for tick-strata and temporal-decay analysis (no network)."""
import numpy as np
import pandas as pd
from src.eval.strata import (
    assign_tick_stratum,
    f1_by_column,
    f1_by_tick_and_year,
    macro_f1,
    plot_strata,
)


def test_assign_tick_stratum_balanced_thirds():
    strata = assign_tick_stratum([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    assert list(strata) == ["small", "small", "medium", "medium", "large", "large"]


def test_macro_f1_perfect_and_chance():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y, y) == 1.0
    assert macro_f1(y, np.zeros_like(y)) < 1.0


def _results(seed=0):
    """Synthetic predictions where large-tick is more predictable and F1 decays across years."""
    rng = np.random.default_rng(seed)
    rows = []
    acc_by_tick = {"small": 0.40, "medium": 0.60, "large": 0.88}
    acc_by_year = {2019: 0.85, 2020: 0.78, 2021: 0.70, 2022: 0.60, 2023: 0.50}
    for tick, a_t in acc_by_tick.items():
        for year, a_y in acc_by_year.items():
            acc = (a_t + a_y) / 2
            n = 400
            y_true = rng.integers(0, 3, size=n)
            correct = rng.uniform(size=n) < acc
            y_pred = np.where(correct, y_true, (y_true + rng.integers(1, 3, size=n)) % 3)
            rows.append(pd.DataFrame({"tick_stratum": tick, "year": year,
                                      "y_true": y_true, "y_pred": y_pred}))
    return pd.concat(rows, ignore_index=True)


def test_large_tick_more_predictable_and_decay_over_years():
    df = _results()
    by_tick, by_year = f1_by_tick_and_year(df)
    # Ordered small < medium < large and increasing F1.
    assert list(by_tick.index) == ["small", "medium", "large"]
    assert by_tick["large"] > by_tick["medium"] > by_tick["small"]
    # Predictability decays: 2019 F1 clearly above 2023 F1.
    assert by_year.loc[2019] > by_year.loc[2023]


def test_f1_by_column_matches_manual():
    # Group a covers all 3 classes and is perfectly classified -> macro-F1 over {0,1,2} == 1.0.
    # Group b is perfect on the classes present but misses class 1, so macro-F1 over the fixed 3 classes < 1.
    df = pd.DataFrame({
        "g": ["a", "a", "a", "b", "b"],
        "y_true": [0, 1, 2, 0, 2],
        "y_pred": [0, 1, 2, 0, 0],
    })
    res = f1_by_column(df, "g")
    assert res["a"] == 1.0
    assert res["b"] < 1.0


def test_plot_saves_figure(tmp_path):
    df = _results()
    by_tick, by_year = f1_by_tick_and_year(df)
    out = plot_strata(by_tick, by_year, str(tmp_path / "reports" / "strata.png"))
    assert (tmp_path / "reports" / "strata.png").exists()
    assert (tmp_path / "reports" / "strata.png").stat().st_size > 0
    assert out.endswith("strata.png")
