# Implementation Logic — Project 1 (Deep LOB)

## Data & labels
- **FI-2010** (free benchmark): 10 levels x {ask price, ask size, bid price, bid size} = 40 features; pre-derived
  normalized representations (z-score / decimal-precision). Use the published train/test split.
- **LOBSTER** free samples (AAPL/MSFT/etc., a few days): reconstruct LOB from message + orderbook files.
- **Label** (FI-2010 convention): smoothed mid-price m(t); future mean m_+(t,k) over horizon k; direction =
  up/stationary/down via a threshold alpha on (m_+(t,k) - m(t)) / m(t). Provide horizons k in {1,2,3,5,10}.
- **Window**: 100 time steps x 40 features per sample.

## Models
1. **DeepLOB** (implemented in `src/models/deeplob.py`): conv blocks compress the 40-wide price/size dimension
   to 1 while LeakyReLU+BN extract local structure; an Inception module captures multi-scale temporal patterns;
   an LSTM(64) captures longer dependencies; a Dense(3)+softmax outputs direction.
2. **TLOB** (TODO): a dual-attention transformer (Berti & Kasneci, 2025) attending across both feature and time
   axes. Replicate, then compare F1 vs DeepLOB per horizon.

## Evaluation (the differentiator)
- Classification: F1 (macro) per horizon; replicate paper numbers as a correctness check.
- **Tick-size strata**: split instruments into small/medium/large-tick; report predictability per stratum.
- **Temporal decay**: train on year Y, test on Y+1..; show the F1 decline over time (a candid, real finding).
- **Honest trading eval**: convert signal -> positions; subtract transaction costs and a queue/latency penalty;
  report net P&L, turnover, and **Deflated Sharpe Ratio** (shared/validation.py). Directional accuracy != profit.

## Pitfalls
- Train/test leakage across the FI-2010 normalization horizon; shuffle within time blocks only.
- Reporting accuracy on a class-imbalanced label without macro-F1.
- Claiming profitability from accuracy without a cost/queue model.
- Overfitting to FI-2010's specific market regime (validate on LOBSTER samples).

## Per-lane positioning
- **Trader/QR**: microstructure understanding + honest backtest + decay analysis.
- **Strats**: clean engineering, reproducible pipeline, model comparison.
