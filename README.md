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
├── src/data/                 # TODO: FI-2010 loader, labels, normalization
├── src/models/tlob.py        # TODO: transformer (dual attention)
├── src/backtest/             # TODO: cost-aware eval (use ../../shared/validation.py)
└── src/eval/                 # TODO: tick-strata + decay analysis
```
