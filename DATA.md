# Data — Project 1 (FREE)

## FI-2010 (primary benchmark)
- The standard public LOB benchmark (Ntakaris et al.). Search "FI-2010 limit order book dataset"; available via
  academic mirrors / Kaggle. Contains normalized 40-feature representations + k-horizon direction labels.
- Pros: free, standard, comparable to papers. Cons: dated (2010), small-tick Nordic stocks, regime-specific.

## LOBSTER sample data (validation)
- LOBSTER provides FREE sample days (e.g., AAPL, MSFT, INTC) as message + orderbook CSVs at a few depth levels.
- Reconstruct the LOB; build the same 100x40 windows. Full historical data is PAID — use only samples and say so.

## Optional: synthetic LOB (your strength)
- Generate synthetic LOB sequences (e.g., a Hawkes-process or a simple agent-based simulator) to stress-test
  generalization. Frame as a robustness ablation, not a primary result.

## Limitations to disclose
- No paid tick data; FI-2010 is old; LOBSTER samples are tiny. Your decay/strata analysis is illustrative on the
  available data, not a claim about current live markets.
