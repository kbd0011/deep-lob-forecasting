# Project 1 — Deep LOB Forecasting + Honest Trading Evaluation (RANK #3)

**Thesis.** Reimplement DeepLOB (Zhang, Zohren & Roberts, 2019) and a modern transformer (TLOB, 2025) for
short-horizon mid-price direction on limit-order-book data — then do the part everyone skips: **failure
analysis** (tick-size strata, signal decay over time) and a **transaction-cost-aware trading evaluation**
that shows whether directional accuracy survives as P&L.

A DeepLOB clone is now a baseline, not an achievement. Your contribution is the rigorous "when/why does it
work, and does the edge survive costs?" analysis.

## Why it ranks #3
Highest prestige with prop shops and the clearest microstructure signal — but constrained by free-data limits
(FI-2010 is dated; LOBSTER free samples are small; full LOB data is paid) and a crowded GitHub field.

## What "done" looks like
> "Replicated DeepLOB F1 within ~1 pt of paper on FI-2010; TLOB beats it by ~3 F1. Predictability is far higher
> for large-tick names and **decays materially across 2019→2023**. After 10 bps costs and queue/latency
> assumptions, the naive signal's P&L edge is ~0 for liquid names and survives only in [regime]. Deflated Sharpe
> reported over all configs tried."

## Layout
```
project1_deep_lob/
├── IMPLEMENTATION_LOGIC.md
├── DATA.md
├── LLM_PROMPTS.md
├── requirements.txt
├── src/models/deeplob.py     # IMPLEMENTED: faithful DeepLOB (PyTorch)
├── src/models/tlob.py        # IMPLEMENTED: TLOB dual-attention transformer (einops)
├── src/data/fi2010.py        # IMPLEMENTED: FI-2010 loader, horizon labels, leakage-safe windowing
├── src/train.py              # IMPLEMENTED: Hydra trainer (class weights, macro-F1 early stop, wandb)
├── src/backtest/trade_eval.py # IMPLEMENTED: cost-aware net P&L, turnover, Deflated Sharpe
└── src/eval/strata.py        # IMPLEMENTED: tick-strata + temporal-decay F1 + plots
```

## Status
All modules implemented with tests (CPU-only, tiny tensors; no network/GPU; FI-2010 windowing tested on a
synthetic array). DeepLOB and TLOB share the `(B,1,100,40)->(B,3)` contract so they are directly comparable.
The trading evaluation makes the project's thesis concrete — its tests show a *perfectly accurate* predictor
still loses money once half-spread + queue/latency costs are charged against tick-sized edges. Drop in
FI-2010 (`Train_*`/`Test_*` files under `data/`) and run `python -m src.train` to reproduce.
