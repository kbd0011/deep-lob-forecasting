"""Tests for the cost-aware trading evaluation (no network)."""
import numpy as np
import pytest
from src.backtest.trade_eval import (
    TradeCostConfig,
    evaluate_strategy,
    predictions_to_positions,
    trade_pnl,
)


def test_predictions_to_positions_mapping():
    preds = np.array([0, 1, 2, 0, 2])
    pos = predictions_to_positions(preds, {0: 1, 1: 0, 2: -1})
    assert list(pos) == [1.0, 0.0, -1.0, 1.0, -1.0]


def test_net_is_below_gross_when_costs_positive():
    rng = np.random.default_rng(0)
    preds = rng.integers(0, 3, size=500)
    r = rng.normal(0, 0.001, size=500)
    cfg = TradeCostConfig(spread_bps=2.0, latency_bps=1.0)
    pos = predictions_to_positions(preds, cfg.position_map)
    pnl = trade_pnl(pos, r, cfg)
    assert pnl["net"].sum() < pnl["gross"].sum()
    assert (pnl["cost"] >= 0).all()


def test_perfect_predictor_profits_without_costs():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.001, size=1000)
    # Perfect: long when the move is up, short when down (class 0=up,2=down); stationary on tiny moves.
    preds = np.where(r > 0, 0, 2)
    cfg = TradeCostConfig(spread_bps=0.0, latency_bps=0.0)
    res = evaluate_strategy(preds, r, cfg)
    assert res["net_total"] > 0
    assert res["hit_rate"] == pytest.approx(1.0)
    assert res["net_sharpe"] > 0


def test_accuracy_does_not_imply_profit_under_turnover_and_costs():
    # A predictor that is right >50% of the time but flips position every step pays heavy turnover costs.
    rng = np.random.default_rng(2)
    n = 4000
    r = rng.normal(0, 0.0002, size=n)          # tiny edges
    preds = np.where(r > 0, 0, 2)              # correct direction (high accuracy)
    # Force maximal turnover by alternating flat/active is implicit; large costs dominate the tiny gross.
    cfg = TradeCostConfig(spread_bps=10.0, latency_bps=5.0)
    res = evaluate_strategy(preds, r, cfg)
    assert res["hit_rate"] == pytest.approx(1.0)   # perfectly accurate...
    assert res["net_total"] < res["gross_total"]   # ...yet costs eat the edge
    assert res["net_total"] <= 0                    # net unprofitable despite perfect accuracy


def test_flat_predictions_have_zero_pnl_and_turnover():
    r = np.random.default_rng(3).normal(0, 0.001, size=200)
    preds = np.ones(200, dtype=int)  # all stationary -> flat
    res = evaluate_strategy(preds, r)
    assert res["net_total"] == pytest.approx(0.0)
    assert res["turnover_total"] == pytest.approx(0.0)
    assert res["n_trades"] == 0
    assert np.isnan(res["hit_rate"])


def test_deflated_sharpe_below_psr_under_multiple_trials():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0001, 0.001, size=2000)
    preds = np.where(r > 0, 0, 2)
    res = evaluate_strategy(preds, r, TradeCostConfig(spread_bps=0.0, latency_bps=0.0),
                            n_trials=50, sr_variance=0.01)
    assert 0.0 <= res["dsr"] <= 1.0
    assert res["dsr"] <= res["psr"] + 1e-9  # deflation cannot increase confidence
