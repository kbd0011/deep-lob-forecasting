# Implementation Logic — Deep LOB

## Data & labels
- **FI-2010** (free benchmark): 10 levels x {ask price, ask size, bid price, bid size} = 40 features; pre-derived
  normalized representations (z-score / decimal-precision). Uses the published train/test files.
- **Label** (FI-2010 convention): smoothed mid-price m(t); future mean m_+(t,k) over horizon k; direction =
  up/stationary/down via a threshold alpha on (m_+(t,k) - m(t)) / m(t). The loader indexes the five label rows
  as horizons {1, 2, 3, 5, 10}; `scripts/run_fi2010.py` reports them as k = 10, 20, 30, 50, 100 events.
- **Window**: 100 time steps x 40 features per sample, labeled at the window's last event.
- **LOBSTER** free samples (AAPL/MSFT/etc., a few days): planned; reconstruct the LOB from message + orderbook
  files and build the same 100x40 windows.

## Models
1. **DeepLOB** (`src/models/deeplob.py`): conv blocks compress the 40-wide price/size dimension to 1 while
   LeakyReLU+BN extract local structure; an Inception module captures multi-scale temporal patterns; an
   LSTM(64) captures longer dependencies; a Dense(3) layer outputs direction logits.
2. **TLOB** (`src/models/tlob.py`): a dual-attention transformer (Berti & Kasneci, 2025) attending across both
   feature and time axes, with the same input/output shapes as DeepLOB. Implemented and unit-tested; not yet
   trained on FI-2010.

## Evaluation
Implemented:
- **Classification:** macro-F1 per horizon (`src/train.py`), reported for DeepLOB on an FI-2010 subset.
- **Cost-aware trading eval** (`src/backtest/trade_eval.py`): signal -> positions; subtract half-spread and a
  queue/latency penalty on turnover; report net P&L, turnover, hit rate, and the Probabilistic / Deflated
  Sharpe Ratio (`shared/validation.py`). Directional accuracy != profit.
- **Tick-size strata and temporal decay** (`src/eval/strata.py`): macro-F1 by tick-size stratum and by year,
  with plots; demonstrated on synthetic predictions because FI-2010 lacks instrument and date metadata.

Planned:
- A full-data FI-2010 run with a validation split taken from the training days, compared with paper numbers.
- DeepLOB vs TLOB per horizon.
- Strata and decay analysis on data with instrument and date metadata (e.g. LOBSTER samples).
- The trading evaluation applied to model predictions with aligned forward returns.

## Pitfalls
- Train/test leakage across the FI-2010 normalization horizon; shuffle within time blocks only.
- Reporting accuracy on a class-imbalanced label without macro-F1.
- Claiming profitability from accuracy without a cost/queue model.
- Overfitting to FI-2010's specific market regime (check on LOBSTER samples).
