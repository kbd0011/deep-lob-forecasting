# LLM Prompts — Project 1 (Deep LOB)

Use the SYSTEM prompt from `../LLM_PROMPTING_GUIDE.md`. Run in order.

### Prompt 1 — FI-2010 loader + labels (`src/data/fi2010.py`)
> CONTEXT: IMPLEMENTATION_LOGIC.md "Data & labels". TASK: implement a loader yielding (X: (N,1,100,40),
> y: (N,) in {0,1,2}) for a chosen horizon k, using the official train/test split, with no time leakage in
> windowing. CONSTRAINTS: numpy, torch Dataset. ACCEPTANCE: code + test on a tiny synthetic array verifying
> window shapes and that labels align to the correct future horizon.

### Prompt 2 — training loop (`src/train.py`)
> CONTEXT: `src/models/deeplob.py`. TASK: Hydra-config-driven trainer (Adam, early stopping, class weights for
> imbalance), logging macro-F1 per epoch to wandb, checkpointing. CONSTRAINTS: torch, hydra, wandb. ACCEPTANCE:
> code + a 2-epoch smoke test on synthetic data (CPU) that runs end to end.

### Prompt 3 — TLOB transformer (`src/models/tlob.py`)
> CONTEXT: replicate TLOB (Berti & Kasneci, 2025) dual attention over feature and time axes; same input/output
> shapes as DeepLOB. CONSTRAINTS: torch, einops. ACCEPTANCE: code + a forward-shape test (B,1,100,40)->(B,3).

### Prompt 4 — cost-aware trading eval (`src/backtest/trade_eval.py`)
> CONTEXT: IMPLEMENTATION_LOGIC.md "Evaluation" + `../../shared/validation.py`. TASK: map predicted direction
> to positions, apply a transaction-cost + queue/latency penalty, compute net P&L, turnover, and Deflated
> Sharpe. CONSTRAINTS: pandas, numpy, import shared.validation. ACCEPTANCE: code + test on synthetic predictions.

### Prompt 5 — tick-strata + decay analysis (`src/eval/strata.py`)
> CONTEXT: IMPLEMENTATION_LOGIC.md "Evaluation". TASK: stratify instruments by tick size and by year; produce
> F1-by-stratum and F1-by-year tables + Matplotlib plots saved to reports/. ACCEPTANCE: code + test + figures.
