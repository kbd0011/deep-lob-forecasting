"""Cost-aware trading evaluation for LOB direction models.

A DeepLOB/TLOB clone with high macro-F1 is *not* a profitable strategy. This module does the part that
separates a credible result from a misleading one: convert predicted directions into positions, charge
**half-spread plus a queue/latency penalty on every position change**, and report net P&L, turnover, hit
rate, and the **Deflated Sharpe Ratio** (which corrects for the many configurations one tries). Directional
accuracy and P&L are reported side by side precisely because they diverge once costs and turnover bite.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shared.validation import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_stats,
)

__all__ = ["TradeCostConfig", "predictions_to_positions", "trade_pnl", "evaluate_strategy"]


@dataclass
class TradeCostConfig:
    """Cost model and class->position mapping.

    ``position_map`` maps predicted class to a target position; the default assumes class 0 = up (long),
    1 = stationary (flat), 2 = down (short) — adjust to your label convention.
    """

    spread_bps: float = 1.0   # full quoted spread; crossing costs half per unit turnover
    latency_bps: float = 0.5  # queue/latency slippage charged per unit turnover (the honest penalty)
    position_map: dict[int, int] = field(default_factory=lambda: {0: 1, 1: 0, 2: -1})


def predictions_to_positions(preds, position_map: dict[int, int]) -> np.ndarray:
    """Map predicted classes to target positions in {-1, 0, +1}."""
    p = np.asarray(preds)
    out = np.zeros(p.shape, dtype=float)
    for cls, pos in position_map.items():
        out[p == cls] = pos
    return out


def trade_pnl(positions: np.ndarray, forward_returns, cfg: TradeCostConfig) -> dict:
    """Gross/net per-period P&L and turnover for a position series.

    ``forward_returns[t]`` is the return realized by holding ``positions[t]`` over the prediction horizon.
    Turnover at ``t`` is ``|pos_t - pos_{t-1}|`` (starting flat); cost = turnover * (half-spread + latency).
    """
    pos = np.asarray(positions, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    if pos.shape != r.shape:
        raise ValueError("positions and forward_returns must have the same shape")
    prev = np.concatenate([[0.0], pos[:-1]])
    turnover = np.abs(pos - prev)
    cost = turnover * (cfg.spread_bps / 2.0 + cfg.latency_bps) * 1e-4
    gross = pos * r
    net = gross - cost
    return {"gross": gross, "net": net, "turnover": turnover, "cost": cost}


def evaluate_strategy(
    preds,
    forward_returns,
    cfg: TradeCostConfig | None = None,
    n_trials: int = 1,
    sr_variance: float = 0.0,
) -> dict:
    """Full honest evaluation: net P&L, turnover, hit rate, and PSR/Deflated Sharpe.

    Parameters
    ----------
    preds : array-like of int
        Predicted classes.
    forward_returns : array-like of float
        Realized forward return aligned to each prediction.
    cfg : TradeCostConfig, optional
    n_trials : int
        Number of configurations tried (for the Deflated Sharpe selection-bias correction).
    sr_variance : float
        Cross-trial variance of Sharpe ratios; with ``n_trials > 1`` this drives the deflation.

    Returns
    -------
    dict
        Totals and per-period stats, ``net_sharpe`` (per period), ``psr``, ``dsr``, ``hit_rate``,
        ``turnover_total``, ``n_trades``, and ``directional_accuracy`` (for the accuracy-vs-P&L contrast).
    """
    cfg = cfg or TradeCostConfig()
    pos = predictions_to_positions(preds, cfg.position_map)
    r = np.asarray(forward_returns, dtype=float)
    pnl = trade_pnl(pos, r, cfg)
    net = pnl["net"]

    stats = sharpe_stats(net)
    psr = probabilistic_sharpe_ratio(stats["sr"], stats["n_obs"], stats["skew"], stats["kurt"])
    if n_trials > 1 and sr_variance > 0:
        dsr = deflated_sharpe_ratio(
            stats["sr"], stats["n_obs"], stats["skew"], stats["kurt"], n_trials, sr_variance
        )
    else:
        dsr = psr

    active = pos != 0
    hit_rate = float((np.sign(pos[active]) == np.sign(r[active])).mean()) if active.any() else float("nan")
    return {
        "gross_total": float(pnl["gross"].sum()),
        "net_total": float(net.sum()),
        "mean_net": float(net.mean()),
        "net_sharpe": stats["sr"],
        "psr": float(psr),
        "dsr": float(dsr),
        "turnover_total": float(pnl["turnover"].sum()),
        "n_trades": int((pnl["turnover"] > 0).sum()),
        "hit_rate": hit_rate,
        "directional_accuracy": hit_rate,
        "n_obs": int(net.size),
    }
