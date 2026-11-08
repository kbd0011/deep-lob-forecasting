# Data

## FI-2010 (primary benchmark)
- The standard public LOB benchmark (Ntakaris et al., 2018), available via academic mirrors and Kaggle. Contains
  normalized 40-feature representations plus k-horizon direction labels.
- Used here: the NoAuction ZScore CF_7 train and test files, downloaded by `make fi2010` into `data/`
  (gitignored). `make train` uses the DecPre CF_7 files named in `configs/train.yaml`.
- Pros: free, standard, comparable to papers. Cons: dated (2010), small-tick Nordic stocks, regime-specific,
  and no per-instrument tick-size or date metadata in the normalized arrays.

## LOBSTER sample data (planned)
- LOBSTER provides free sample days (e.g., AAPL, MSFT, INTC) as message + orderbook CSVs at a few depth levels.
- Plan: reconstruct the LOB and build the same 100x40 windows. Full historical data is paid, so only the
  sample days would be used.

## Synthetic data
- The tick-strata and temporal-decay figure uses synthetic predictions with the effects built in; it
  demonstrates the tooling only. The unit tests also use small synthetic arrays and returns.

## Limitations
- No paid tick data; FI-2010 is old; LOBSTER samples are small. Any decay or strata result on this data would be
  illustrative, not a claim about current live markets.
